//! BatchNorm1d／2d 層（イシュー #1732・親 #1608。`docs/compat-api-scope.md`
//! §1.2 Tier 1・`docs/batch-norm-ops-design.md`）。
//!
//! `nn::LayerNorm`／`RmsNorm`（`norm.rs`）と同じ「本体（`Tensor<f32>` を
//! 永続保持する層パラメータ）→ `bind(&tape)` で `Var` 化した
//! `BatchNormVars`（1 ステップ分のテープ登録済みパラメータ）」の分離
//! パターンを踏襲するが、BatchNorm は本クレート内で**初めて train／
//! eval でモードにより挙動が変わる層**であり、running mean／var
//! （`RefCell`。`Module::forward`／`forward_host` が `&self` のため
//! train モードでの更新に内部可変性が必要——`nn::optim::device_store`
//! の `RefCell<Option<GradStaging>>` が先例）と `training: bool`
//! （plain フィールド。`Module::set_training` は `&mut self`）を保持する。
//!
//! # チャネル軸・rank 契約
//!
//! チャネル軸は常に dim 1（NCHW／NCL 固定。`fandhe_ai_tensor_core::
//! batch_norm_layout` が `(n, c, spatial)` を導出）。`BatchNorm1d` は
//! rank 2（`[N, C]`）／rank 3（`[N, C, L]`）を受理し、`BatchNorm2d` は
//! rank 4（`[N, C, H, W]`）のみを受理する（`Var::batch_norm`／
//! `batch_norm_infer` 自体は rank 2〜4 を一様に受理するため、この
//! rank 限定は本層が forward 時に追加検査する）。
//!
//! # running stats 更新契約
//!
//! `running = (1−momentum)·running + momentum·batch_stat`（`f64` で
//! 計算し 1 回 downcast。`.claude/rules/coding-rust.md` の縮約契約と
//! 同じ精度規律）。`running_var` には **unbiased** 分散
//! （`batch_var · M/(M−1)`。`M = n·spatial`）を用いる一方、出力の
//! 正規化自体は biased 分散（PyTorch `torch.nn.functional.batch_norm`
//! 互換）。train モードの forward は呼ぶたび必ず running stats を
//! 更新する（PyTorch の functional と同じ契約。eval モードは更新
//! しない）。
//!
//! # `BatchNormCore` の可視性
//!
//! `impl Module for BatchNorm1d`／`BatchNorm2d`（`module.rs`）が
//! `forward_host`／`set_training`／`named_parameters` から `core`
//! フィールドへアクセスする必要があるため、`BatchNormCore` 自体とその
//! メソッド群は `pub(crate)` とする（クレート外には非公開のまま——
//! `BatchNorm1d`／`BatchNorm2d` 自体の公開 API はこのモジュール末尾の
//! `pub` メソッドに限られる）。
//!
//! # 対象外（設計 doc `docs/batch-norm-ops-design.md` §7 参照）
//!
//! `momentum=None`（累積移動平均）・`track_running_stats=false`・
//! `named_buffers`／state_dict 直列化（running stats は buffer であり
//! `named_parameters` に含めない）・rank 5（BatchNorm3d）・
//! channels-last レイアウトはいずれも本 issue の対象外。

use std::cell::{Cell, Ref, RefCell};

use fandhe_ai_tensor_core::{ShapeError, Tensor, batch_norm_layout};

use crate::error::AutodiffError;
use crate::eval::dense_vec;
use crate::tape::Tape;
use crate::var::Var;

/// BatchNorm1d／2d 共通の既定 `eps`（PyTorch `nn.BatchNorm1d`／
/// `BatchNorm2d` の既定値）。
pub const BATCH_NORM_DEFAULT_EPS: f32 = 1e-5;
/// BatchNorm1d／2d 共通の既定 `momentum`（PyTorch 既定値）。
pub const BATCH_NORM_DEFAULT_MOMENTUM: f32 = 0.1;

/// `BatchNorm1d` が受理する rank（rank 2: 非空間入力、rank 3: 空間
/// 入力 `[N, C, L]`）。[`crate::nn::module`] の `Module::forward_host`
/// 実装からも参照する。
pub(crate) const BATCH_NORM_1D_RANKS: &[usize] = &[2, 3];
/// `BatchNorm2d` が受理する rank（rank 4: `[N, C, H, W]` のみ）。
pub(crate) const BATCH_NORM_2D_RANKS: &[usize] = &[4];

/// `eps` の fail-closed 検査（`nn::norm::validate_eps` と同じ「有限かつ
/// 非負」契約）。
fn validate_eps(eps: f32, who: &str) -> Result<(), AutodiffError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{who}: eps must be finite and non-negative, got {eps}"
        )));
    }
    Ok(())
}

/// `momentum` の fail-closed 検査（有限かつ `[0.0, 1.0]` の範囲。
/// running stats 更新式 `(1−m)·running + m·stat` が発散しないための
/// 契約。OWASP A03: 外部由来の構築引数を計算前に検証する）。
fn validate_momentum(momentum: f32, who: &str) -> Result<(), AutodiffError> {
    if !momentum.is_finite() || !(0.0..=1.0).contains(&momentum) {
        return Err(AutodiffError::InvalidArgument(format!(
            "{who}: momentum must be finite and within [0.0, 1.0], got {momentum}"
        )));
    }
    Ok(())
}

/// `BatchNorm1d`／`BatchNorm2d` 共通のパラメータ本体（クレート内公開。
/// 両層が `core` フィールドとして保持する。モジュール doc comment
/// 「`BatchNormCore` の可視性」参照）。
#[derive(Debug)]
pub(crate) struct BatchNormCore {
    num_features: usize,
    eps: f32,
    momentum: f32,
    weight: Option<Tensor<f32>>,
    bias: Option<Tensor<f32>>,
    running_mean: RefCell<Tensor<f32>>,
    running_var: RefCell<Tensor<f32>>,
    num_batches_tracked: Cell<u64>,
    training: bool,
}

impl BatchNormCore {
    /// `weight` を全要素 `1.0`・`bias` を全要素 `0.0`・`running_mean` を
    /// 全要素 `0.0`・`running_var` を全要素 `1.0` で初期化する
    /// （`elementwise_affine=true`・`track_running_stats=true` 相当。
    /// PyTorch 既定）。
    fn new(num_features: usize, eps: f32, momentum: f32, who: &str) -> Result<Self, AutodiffError> {
        validate_eps(eps, who)?;
        validate_momentum(momentum, who)?;
        if num_features == 0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "{who}: num_features must be non-zero"
            )));
        }
        // `Tensor::full` は `vec![value; numel]` を確保する前に
        // `checked_numel_for::<f32>` でバイトサイズ上限（`isize::MAX`）
        // まで検査する。ここを素の `vec![1.0f32; num_features]` に
        // 置き換えると `Tensor::new` の検査へ到達する前に `vec!` 自体が
        // capacity overflow で panic し、本番経路 panic 禁止規約
        // （`.claude/rules/coding-rust.md`）に反する（イシュー #1732・
        // PR #1874 codex-review P1 是正）。
        let weight = Tensor::full(&[num_features], 1.0f32)?;
        let bias = Tensor::full(&[num_features], 0.0f32)?;
        let running_mean = Tensor::full(&[num_features], 0.0f32)?;
        let running_var = Tensor::full(&[num_features], 1.0f32)?;
        Ok(Self {
            num_features,
            eps,
            momentum,
            weight: Some(weight),
            bias: Some(bias),
            running_mean: RefCell::new(running_mean),
            running_var: RefCell::new(running_var),
            num_batches_tracked: Cell::new(0),
            training: true,
        })
    }

    /// `weight`／`bias` を持たない構成（`elementwise_affine=false`
    /// 相当）。running stats は [`Self::new`] と同じ初期化。
    fn without_affine(
        num_features: usize,
        eps: f32,
        momentum: f32,
        who: &str,
    ) -> Result<Self, AutodiffError> {
        validate_eps(eps, who)?;
        validate_momentum(momentum, who)?;
        if num_features == 0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "{who}: num_features must be non-zero"
            )));
        }
        // `new` と同じ理由で `Tensor::full` を使い `vec!` 前に確保上限を
        // 検査する（PR #1874 codex-review P1 是正）。
        let running_mean = Tensor::full(&[num_features], 0.0f32)?;
        let running_var = Tensor::full(&[num_features], 1.0f32)?;
        Ok(Self {
            num_features,
            eps,
            momentum,
            weight: None,
            bias: None,
            running_mean: RefCell::new(running_mean),
            running_var: RefCell::new(running_var),
            num_batches_tracked: Cell::new(0),
            training: true,
        })
    }

    /// 明示的な `weight`／`bias`／`running_mean`／`running_var` から
    /// 構築する（safetensors ロード等向け。`nn::LayerNorm::
    /// from_parameters` と同じ位置付け）。`running_mean`／
    /// `running_var` は必ず rank 1・同一長を要求し、`num_features`
    /// はその長さから導出する。`weight`／`bias` を渡す場合はそれぞれ
    /// 独立に rank 1・同一長を要求する（A03: 外部由来パラメータを
    /// 計算前に検証する契約。`.claude/rules/security.md`）。
    #[allow(clippy::too_many_arguments)]
    fn from_parameters(
        weight: Option<Tensor<f32>>,
        bias: Option<Tensor<f32>>,
        running_mean: Tensor<f32>,
        running_var: Tensor<f32>,
        eps: f32,
        momentum: f32,
        who: &str,
    ) -> Result<Self, AutodiffError> {
        validate_eps(eps, who)?;
        validate_momentum(momentum, who)?;
        if running_mean.rank() != 1 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 1,
                actual: running_mean.rank(),
            }));
        }
        if running_var.shape() != running_mean.shape() {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: running_var.shape().to_vec(),
                rhs: running_mean.shape().to_vec(),
            }));
        }
        let num_features = running_mean.shape()[0];
        // `new`／`without_affine` と同じ契約（チャネル数ゼロ拒否）を
        // `from_parameters` にも課す。running_mean の shape から
        // `num_features` を間接導出する経路であるため、ここで検査
        // しないと `new`／`without_affine` では拒否される
        // `num_features == 0` 構成がこのコンストラクタだけ素通りして
        // しまい、コンストラクタ間の入力契約が不統一になる
        // （PR #1874 codex-review P2・Cursor Bugbot Low 是正）。
        if num_features == 0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "{who}: num_features must be non-zero"
            )));
        }
        if let Some(w) = &weight
            && w.shape() != running_mean.shape()
        {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: w.shape().to_vec(),
                rhs: running_mean.shape().to_vec(),
            }));
        }
        if let Some(b) = &bias
            && b.shape() != running_mean.shape()
        {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: b.shape().to_vec(),
                rhs: running_mean.shape().to_vec(),
            }));
        }
        Ok(Self {
            num_features,
            eps,
            momentum,
            weight,
            bias,
            running_mean: RefCell::new(running_mean),
            running_var: RefCell::new(running_var),
            num_batches_tracked: Cell::new(0),
            training: true,
        })
    }

    /// `Var::batch_norm_with_batch_stats`／`batch_norm_infer` の shape
    /// 検査は `weight`／`bias` が `Some` のときのみ入力チャネル数 `c`
    /// を間接検証する（`w.shape() == [c]` かつ `w` は `[num_features]`
    /// で構築済みのため）。`without_affine`（`weight`／`bias` とも
    /// `None`）構成では `c` を検証する経路が無くなるため、呼び出し元
    /// （[`BatchNormCore::forward_var`]／`batch_norm_forward_host`）が
    /// `c == num_features` を明示検査するのに使うアクセサ（codex-review
    /// P1・イシュー #1732 fix ループ）。
    pub(crate) fn num_features(&self) -> usize {
        self.num_features
    }

    pub(crate) fn weight(&self) -> Option<&Tensor<f32>> {
        self.weight.as_ref()
    }

    pub(crate) fn bias(&self) -> Option<&Tensor<f32>> {
        self.bias.as_ref()
    }

    /// `Ref` の漏出を避けるため clone した `Tensor<f32>` を返す
    /// （公開アクセサ。`RefCell<Tensor<f32>>` 自体は公開しない）。
    pub(crate) fn running_mean(&self) -> Tensor<f32> {
        self.running_mean.borrow().clone()
    }

    pub(crate) fn running_var(&self) -> Tensor<f32> {
        self.running_var.borrow().clone()
    }

    /// [`Self::running_mean`]／[`Self::running_var`] の clone を避けたい
    /// 呼び出し元（`forward_host` 等）向けの `Ref` 直接借用。
    pub(crate) fn running_mean_ref(&self) -> Ref<'_, Tensor<f32>> {
        self.running_mean.borrow()
    }

    pub(crate) fn running_var_ref(&self) -> Ref<'_, Tensor<f32>> {
        self.running_var.borrow()
    }

    pub(crate) fn num_batches_tracked(&self) -> u64 {
        self.num_batches_tracked.get()
    }

    pub(crate) fn eps(&self) -> f32 {
        self.eps
    }

    pub(crate) fn momentum(&self) -> f32 {
        self.momentum
    }

    pub(crate) fn training(&self) -> bool {
        self.training
    }

    pub(crate) fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// train モードの forward が返すバッチ統計から running stats を
    /// 更新する（モジュール doc comment「running stats 更新契約」
    /// 参照）。`m`（= `n*spatial`。チャネルごとの縮約要素数）は
    /// 呼び出し元が `Var::batch_norm_with_batch_stats`／
    /// `batch_norm_train_with_fallback` と同じ入力から導出済みの
    /// ものを渡す。
    pub(crate) fn update_running_stats(
        &self,
        batch_mean: &Tensor<f32>,
        batch_var: &Tensor<f32>,
        m: usize,
    ) {
        let momentum = self.momentum as f64;
        let unbiased_factor = if m > 1 {
            m as f64 / (m as f64 - 1.0)
        } else {
            1.0
        };
        let new_mean: Vec<f32> = {
            let running = dense_vec(&self.running_mean.borrow());
            let batch = dense_vec(batch_mean);
            running
                .iter()
                .zip(batch.iter())
                .map(|(&r, &b)| ((1.0 - momentum) * r as f64 + momentum * b as f64) as f32)
                .collect()
        };
        let new_var: Vec<f32> = {
            let running = dense_vec(&self.running_var.borrow());
            let batch = dense_vec(batch_var);
            running
                .iter()
                .zip(batch.iter())
                .map(|(&r, &b)| {
                    let unbiased_b = b as f64 * unbiased_factor;
                    ((1.0 - momentum) * r as f64 + momentum * unbiased_b) as f32
                })
                .collect()
        };
        // `new_mean`／`new_var` の長さは常に `num_features`（呼び出し元
        // が `Var::batch_norm_with_batch_stats` の契約検証済み戻り値
        // `batch_mean`／`batch_var`〈shape `[c]`〉を渡すため）。万一の
        // shape 不変条件違反は running stats を変えずに保持する
        // （本番経路 panic 禁止規約。`.claude/rules/coding-rust.md`）。
        let mean_tensor = Tensor::new(new_mean, &[self.num_features]).unwrap_or_else(|_| {
            debug_assert!(
                false,
                "BatchNormCore::update_running_stats: batch_mean の長さが num_features と不一致"
            );
            self.running_mean.borrow().clone()
        });
        let var_tensor = Tensor::new(new_var, &[self.num_features]).unwrap_or_else(|_| {
            debug_assert!(
                false,
                "BatchNormCore::update_running_stats: batch_var の長さが num_features と不一致"
            );
            self.running_var.borrow().clone()
        });
        *self.running_mean.borrow_mut() = mean_tensor;
        *self.running_var.borrow_mut() = var_tensor;
        self.num_batches_tracked
            .set(self.num_batches_tracked.get().saturating_add(1));
    }

    pub(crate) fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = Vec::new();
        if let Some(w) = self.weight() {
            out.push(("weight".to_string(), w));
        }
        if let Some(b) = self.bias() {
            out.push(("bias".to_string(), b));
        }
        out
    }

    /// [`crate::nn::module::Module::set_parameter`]（`BatchNorm1d`／
    /// `BatchNorm2d` 実装。`module.rs` 参照）の本体。`nn::norm::
    /// LayerNorm::set_parameter` と同型（`"weight"`／`"bias"` を受理し、
    /// 対応するフィールドが `None`〈`without_affine` 構成〉の場合は
    /// 未知名扱いで拒否）。shape 保存置換のみ（running stats
    /// 〈`running_mean`／`running_var`〉は buffer であり
    /// `named_parameters` に含めないため、本メソッドの対象にも
    /// 含めない。`Module::set_parameter` doc「オーバーライド指針」の
    /// `named_parameters` と対で実装する契約を満たす。PR #1874
    /// codex-review P1・Cursor Bugbot Medium 是正・イシュー #1732）。
    pub(crate) fn set_parameter(
        &mut self,
        name: &str,
        value: Tensor<f32>,
    ) -> Result<(), AutodiffError> {
        let slot = match name {
            "weight" => &mut self.weight,
            "bias" => &mut self.bias,
            _ => {
                return Err(AutodiffError::InvalidArgument(format!(
                    "BatchNormCore::set_parameter: no parameter named `{name}`"
                )));
            }
        };
        match slot {
            Some(current) => {
                if value.shape() != current.shape() {
                    return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                        lhs: value.shape().to_vec(),
                        rhs: current.shape().to_vec(),
                    }));
                }
                *current = value;
                Ok(())
            }
            None => Err(AutodiffError::InvalidArgument(format!(
                "BatchNormCore::set_parameter: no parameter named `{name}`"
            ))),
        }
    }

    /// train／eval モードに応じて `Var::batch_norm_with_batch_stats`／
    /// `batch_norm_infer` へ委譲する tape 経路の forward 本体
    /// （`BatchNormVars::forward` と `Module::forward_host` 双方が
    /// 参照する rank 限定契約を伴わない共通ロジック）。`accepted_ranks`
    /// の検査は呼び出し元（`BatchNormVars::forward`／`forward_host`）
    /// が先に行う。
    ///
    /// `weight`／`bias` が `Some` の構成では `Var::
    /// batch_norm_with_batch_stats`／`batch_norm_infer` 内部の
    /// `require_same_shape(w.shape(), &[c])` が `w` の構築時 shape
    /// （`[num_features]`）経由で `c == num_features` を間接検証する
    /// が、`without_affine`（両方 `None`）構成ではその経路が無い。
    /// ここで `c == num_features` を明示検査しないと、train モードで
    /// `update_running_stats` へ長さ `c` の `batch_mean`／`batch_var`
    /// が渡り、running stats（長さ `num_features`）との `zip` が
    /// `c < num_features` では debug_assert panic（debug）・
    /// `c > num_features` では黙った切り詰め（release）を招く
    /// （codex-review P1・Cursor Bugbot 指摘。イシュー #1732 fix
    /// ループ。eval モードは `batch_norm_infer` が `running_mean`
    /// shape `[num_features]` を `[c]` と検査するため既に安全）。
    fn forward_var<'t>(
        &self,
        input: &Var<'t>,
        weight: Option<&Var<'t>>,
        bias: Option<&Var<'t>>,
    ) -> Result<Var<'t>, AutodiffError> {
        let (n, c, spatial) = batch_norm_layout(&input.shape())?;
        if c != self.num_features {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![c],
                rhs: vec![self.num_features],
            }));
        }
        if self.training {
            let (out, batch_mean, batch_var) =
                input.batch_norm_with_batch_stats(weight, bias, self.eps)?;
            self.update_running_stats(&batch_mean, &batch_var, n * spatial);
            Ok(out)
        } else {
            let running_mean = self.running_mean.borrow();
            let running_var = self.running_var.borrow();
            input.batch_norm_infer(weight, bias, &running_mean, &running_var, self.eps)
        }
    }
}

/// `BatchNorm1d`／`BatchNorm2d::bind` が返す、1 ステップ分のテープに
/// 登録済みパラメータ。`accepted_ranks` は呼び出し元（1d／2d）が渡す
/// rank 限定契約（モジュール doc comment「チャネル軸・rank 契約」）。
pub struct BatchNormVars<'t, 'a> {
    core: &'a BatchNormCore,
    accepted_ranks: &'static [usize],
    /// `BatchNorm*::bind` 時点の `weight`（affine なし構成の場合は
    /// `None`）をこの `tape` へ登録した `Var`。`Tape::backward` 後に
    /// `Gradients::get(&vars.weight)`（`Some` の場合）で `dweight` を
    /// 取得する（呼び出し側の責務。[`crate::nn::norm::LayerNormVars`]
    /// と同じ理由）。
    pub weight: Option<Var<'t>>,
    /// `BatchNorm*::bind` 時点の `bias`（`weight` と独立に `None` を
    /// 取りうる）をこの `tape` へ登録した `Var`。
    pub bias: Option<Var<'t>>,
}

impl<'t, 'a> BatchNormVars<'t, 'a> {
    /// `self.core` の train／eval モードに応じて train（`Var::
    /// batch_norm_with_batch_stats`。running stats を更新する）／eval
    /// （`Var::batch_norm_infer`。固定統計）へ委譲する。
    pub fn forward(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let rank = input.shape().len();
        if !self.accepted_ranks.contains(&rank) {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: self.accepted_ranks[0],
                actual: rank,
            }));
        }
        self.core
            .forward_var(input, self.weight.as_ref(), self.bias.as_ref())
    }
}

/// `BatchNorm1d`（rank 2 `[N, C]`／rank 3 `[N, C, L]` の入力を受理。
/// PyTorch `nn.BatchNorm1d` 相当）。
#[derive(Debug)]
pub struct BatchNorm1d {
    pub(crate) core: BatchNormCore,
}

impl BatchNorm1d {
    /// affine あり（`weight`／`bias` を持つ）`BatchNorm1d` を構築する。
    /// `weight` は 1・`bias` は 0・`running_mean` は 0・`running_var` は 1
    /// で初期化し、初期モードは train（[`Self::bind`] 直後は
    /// `Var::batch_norm_with_batch_stats` 経由で running stats を
    /// 更新する）。`num_features` は非 0 を要求する。対象入力 shape は
    /// rank 2 `[N, C]`／rank 3 `[N, C, L]`（チャネル軸は常に dim 1）。
    pub fn new(num_features: usize, eps: f32, momentum: f32) -> Result<Self, AutodiffError> {
        Ok(Self {
            core: BatchNormCore::new(num_features, eps, momentum, "BatchNorm1d::new")?,
        })
    }

    /// affine なし（`weight`／`bias` を持たない）`BatchNorm1d` を
    /// 構築する。PyTorch `nn.BatchNorm*d(..., affine=False)` 相当。
    /// `running_mean`／`running_var`・初期モード（train）は
    /// [`Self::new`] と同じ。`num_features` は非 0 を要求する。
    pub fn without_affine(
        num_features: usize,
        eps: f32,
        momentum: f32,
    ) -> Result<Self, AutodiffError> {
        Ok(Self {
            core: BatchNormCore::without_affine(
                num_features,
                eps,
                momentum,
                "BatchNorm1d::without_affine",
            )?,
        })
    }

    /// 既存の `weight`／`bias`／`running_mean`／`running_var` から
    /// `BatchNorm1d` を構築する（チェックポイント復元等）。`weight`
    /// と `bias` は独立に `Some`／`None` を取れる。`running_mean`／
    /// `running_var` は形状が一致する非 0 長の rank 1 テンソルを
    /// 要求し、`num_features` はその長さから導出する。初期モードは
    /// train。
    pub fn from_parameters(
        weight: Option<Tensor<f32>>,
        bias: Option<Tensor<f32>>,
        running_mean: Tensor<f32>,
        running_var: Tensor<f32>,
        eps: f32,
        momentum: f32,
    ) -> Result<Self, AutodiffError> {
        Ok(Self {
            core: BatchNormCore::from_parameters(
                weight,
                bias,
                running_mean,
                running_var,
                eps,
                momentum,
                "BatchNorm1d::from_parameters",
            )?,
        })
    }

    /// affine の重み（shape `[num_features]`）。`without_affine` 構成
    /// では `None`。
    pub fn weight(&self) -> Option<&Tensor<f32>> {
        self.core.weight()
    }

    /// affine のバイアス（shape `[num_features]`）。`without_affine`
    /// 構成では `None`。
    pub fn bias(&self) -> Option<&Tensor<f32>> {
        self.core.bias()
    }

    /// 現在の running mean（shape `[num_features]`）の clone。train
    /// モードの forward を呼ぶたびに momentum に従って更新される
    /// （モジュール doc comment「running stats 更新契約」参照）。
    pub fn running_mean(&self) -> Tensor<f32> {
        self.core.running_mean()
    }

    /// 現在の running variance（shape `[num_features]`。unbiased）の
    /// clone。更新契約は [`Self::running_mean`] と同じ。
    pub fn running_var(&self) -> Tensor<f32> {
        self.core.running_var()
    }

    /// train モードの forward を呼んだ回数（`update_running_stats` の
    /// 呼び出し回数）。`saturating_add` で飽和する。
    pub fn num_batches_tracked(&self) -> u64 {
        self.core.num_batches_tracked()
    }

    /// 分散へ加える数値安定化定数（構築時に固定・非負かつ有限）。
    pub fn eps(&self) -> f32 {
        self.core.eps()
    }

    /// running stats 更新の momentum（構築時に固定）。
    pub fn momentum(&self) -> f32 {
        self.core.momentum()
    }

    /// このステップの `tape` へ `weight`／`bias`（あれば）を葉ノード
    /// として登録し、`forward` を呼べる [`BatchNormVars`] を返す。
    /// 受理する入力 rank は rank 2 `[N, C]`／rank 3 `[N, C, L]`（`accepted_ranks`）に限る。
    pub fn bind<'t>(&self, tape: &'t Tape) -> BatchNormVars<'t, '_> {
        let weight = self.core.weight.as_ref().map(|w| tape.var(w));
        let bias = self.core.bias.as_ref().map(|b| tape.var(b));
        BatchNormVars {
            core: &self.core,
            accepted_ranks: BATCH_NORM_1D_RANKS,
            weight,
            bias,
        }
    }
}

/// `BatchNorm2d`（rank 4 `[N, C, H, W]` の入力のみを受理。PyTorch
/// `nn.BatchNorm2d` 相当）。
#[derive(Debug)]
pub struct BatchNorm2d {
    pub(crate) core: BatchNormCore,
}

impl BatchNorm2d {
    /// affine あり（`weight`／`bias` を持つ）`BatchNorm2d` を構築する。
    /// `weight` は 1・`bias` は 0・`running_mean` は 0・`running_var` は 1
    /// で初期化し、初期モードは train（[`Self::bind`] 直後は
    /// `Var::batch_norm_with_batch_stats` 経由で running stats を
    /// 更新する）。`num_features` は非 0 を要求する。対象入力 shape は
    /// rank 4 `[N, C, H, W]`（チャネル軸は常に dim 1）。
    pub fn new(num_features: usize, eps: f32, momentum: f32) -> Result<Self, AutodiffError> {
        Ok(Self {
            core: BatchNormCore::new(num_features, eps, momentum, "BatchNorm2d::new")?,
        })
    }

    /// affine なし（`weight`／`bias` を持たない）`BatchNorm2d` を
    /// 構築する。PyTorch `nn.BatchNorm*d(..., affine=False)` 相当。
    /// `running_mean`／`running_var`・初期モード（train）は
    /// [`Self::new`] と同じ。`num_features` は非 0 を要求する。
    pub fn without_affine(
        num_features: usize,
        eps: f32,
        momentum: f32,
    ) -> Result<Self, AutodiffError> {
        Ok(Self {
            core: BatchNormCore::without_affine(
                num_features,
                eps,
                momentum,
                "BatchNorm2d::without_affine",
            )?,
        })
    }

    /// 既存の `weight`／`bias`／`running_mean`／`running_var` から
    /// `BatchNorm2d` を構築する（チェックポイント復元等）。`weight`
    /// と `bias` は独立に `Some`／`None` を取れる。`running_mean`／
    /// `running_var` は形状が一致する非 0 長の rank 1 テンソルを
    /// 要求し、`num_features` はその長さから導出する。初期モードは
    /// train。
    pub fn from_parameters(
        weight: Option<Tensor<f32>>,
        bias: Option<Tensor<f32>>,
        running_mean: Tensor<f32>,
        running_var: Tensor<f32>,
        eps: f32,
        momentum: f32,
    ) -> Result<Self, AutodiffError> {
        Ok(Self {
            core: BatchNormCore::from_parameters(
                weight,
                bias,
                running_mean,
                running_var,
                eps,
                momentum,
                "BatchNorm2d::from_parameters",
            )?,
        })
    }

    /// affine の重み（shape `[num_features]`）。`without_affine` 構成
    /// では `None`。
    pub fn weight(&self) -> Option<&Tensor<f32>> {
        self.core.weight()
    }

    /// affine のバイアス（shape `[num_features]`）。`without_affine`
    /// 構成では `None`。
    pub fn bias(&self) -> Option<&Tensor<f32>> {
        self.core.bias()
    }

    /// 現在の running mean（shape `[num_features]`）の clone。train
    /// モードの forward を呼ぶたびに momentum に従って更新される
    /// （モジュール doc comment「running stats 更新契約」参照）。
    pub fn running_mean(&self) -> Tensor<f32> {
        self.core.running_mean()
    }

    /// 現在の running variance（shape `[num_features]`。unbiased）の
    /// clone。更新契約は [`Self::running_mean`] と同じ。
    pub fn running_var(&self) -> Tensor<f32> {
        self.core.running_var()
    }

    /// train モードの forward を呼んだ回数（`update_running_stats` の
    /// 呼び出し回数）。`saturating_add` で飽和する。
    pub fn num_batches_tracked(&self) -> u64 {
        self.core.num_batches_tracked()
    }

    /// 分散へ加える数値安定化定数（構築時に固定・非負かつ有限）。
    pub fn eps(&self) -> f32 {
        self.core.eps()
    }

    /// running stats 更新の momentum（構築時に固定）。
    pub fn momentum(&self) -> f32 {
        self.core.momentum()
    }

    /// このステップの `tape` へ `weight`／`bias`（あれば）を葉ノード
    /// として登録し、`forward` を呼べる [`BatchNormVars`] を返す。
    /// 受理する入力 rank は rank 4 `[N, C, H, W]`（`accepted_ranks`）に限る。
    pub fn bind<'t>(&self, tape: &'t Tape) -> BatchNormVars<'t, '_> {
        let weight = self.core.weight.as_ref().map(|w| tape.var(w));
        let bias = self.core.bias.as_ref().map(|b| tape.var(b));
        BatchNormVars {
            core: &self.core,
            accepted_ranks: BATCH_NORM_2D_RANKS,
            weight,
            bias,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn dense(t: &Tensor<f32>) -> Vec<f32> {
        t.as_slice()
            .expect("test: expected contiguous tensor")
            .to_vec()
    }

    #[test]
    fn batch_norm_1d_new_initializes_weight_ones_bias_zeros_running_stats() {
        let bn = BatchNorm1d::new(3, BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM).unwrap();
        assert_eq!(dense(bn.weight().unwrap()), vec![1.0f32; 3]);
        assert_eq!(dense(bn.bias().unwrap()), vec![0.0f32; 3]);
        assert_eq!(dense(&bn.running_mean()), vec![0.0f32; 3]);
        assert_eq!(dense(&bn.running_var()), vec![1.0f32; 3]);
        assert_eq!(bn.num_batches_tracked(), 0);
    }

    #[test]
    fn batch_norm_1d_without_affine_has_no_weight_or_bias() {
        let bn =
            BatchNorm1d::without_affine(3, BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM)
                .unwrap();
        assert!(bn.weight().is_none());
        assert!(bn.bias().is_none());
    }

    #[test]
    fn batch_norm_rejects_zero_num_features() {
        let err =
            BatchNorm1d::new(0, BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    /// `num_features=usize::MAX` は `Tensor::new(vec![...; num_features],
    /// ..)` の旧実装では `Tensor::new` の検査へ到達する前に `vec!` 自体が
    /// capacity overflow で panic していた（本番経路 panic 禁止規約
    /// `.claude/rules/coding-rust.md` 違反。PR #1874 codex-review P1）。
    /// `Tensor::full` 採用後は型付きエラーへ収束することを確認する
    /// （panic しないこと自体が本テストの主目的）。
    #[test]
    fn batch_norm_1d_new_rejects_huge_num_features_without_panicking() {
        let err = BatchNorm1d::new(
            usize::MAX,
            BATCH_NORM_DEFAULT_EPS,
            BATCH_NORM_DEFAULT_MOMENTUM,
        )
        .unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }

    /// `without_affine`／`BatchNorm2d::new` も同じ `Tensor::full` 経路を
    /// 通るため、同型の huge `num_features` で panic しないことを
    /// 確認する。
    #[test]
    fn batch_norm_1d_without_affine_rejects_huge_num_features_without_panicking() {
        let err = BatchNorm1d::without_affine(
            usize::MAX,
            BATCH_NORM_DEFAULT_EPS,
            BATCH_NORM_DEFAULT_MOMENTUM,
        )
        .unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }

    #[test]
    fn batch_norm_2d_new_rejects_huge_num_features_without_panicking() {
        let err = BatchNorm2d::new(
            usize::MAX,
            BATCH_NORM_DEFAULT_EPS,
            BATCH_NORM_DEFAULT_MOMENTUM,
        )
        .unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }

    #[test]
    fn batch_norm_rejects_non_finite_eps() {
        let err = BatchNorm1d::new(3, f32::NAN, BATCH_NORM_DEFAULT_MOMENTUM).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn batch_norm_rejects_out_of_range_momentum() {
        let err = BatchNorm1d::new(3, BATCH_NORM_DEFAULT_EPS, 1.5).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        let err = BatchNorm1d::new(3, BATCH_NORM_DEFAULT_EPS, -0.1).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn batch_norm_from_parameters_rejects_shape_mismatch() {
        let running_mean = Tensor::new(vec![0.0f32; 3], &[3]).unwrap();
        let running_var = Tensor::new(vec![1.0f32; 4], &[4]).unwrap();
        let err = BatchNorm1d::from_parameters(
            None,
            None,
            running_mean,
            running_var,
            BATCH_NORM_DEFAULT_EPS,
            BATCH_NORM_DEFAULT_MOMENTUM,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
        ));
    }

    /// `from_parameters` は `running_mean`／`running_var` の shape から
    /// `num_features` を間接導出するため、`new`／`without_affine` と
    /// 同じ「チャネル数ゼロ拒否」契約を課さないとコンストラクタ間で
    /// 入力契約が不統一になる（PR #1874 codex-review P2・Cursor
    /// Bugbot Low 是正）。
    #[test]
    fn batch_norm_from_parameters_rejects_zero_num_features() {
        let running_mean = Tensor::new(Vec::<f32>::new(), &[0]).unwrap();
        let running_var = Tensor::new(Vec::<f32>::new(), &[0]).unwrap();
        let err = BatchNorm1d::from_parameters(
            None,
            None,
            running_mean,
            running_var,
            BATCH_NORM_DEFAULT_EPS,
            BATCH_NORM_DEFAULT_MOMENTUM,
        )
        .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn batch_norm_bind_forward_train_matches_direct_var_call() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]).unwrap());
        let bn = BatchNorm1d::new(2, 0.0, 0.1).unwrap();

        let via_vars = bn.bind(&tape).forward(&x).unwrap().to_tensor();
        let w = tape.var(bn.weight().unwrap());
        let b = tape.var(bn.bias().unwrap());
        let via_direct = x.batch_norm(Some(&w), Some(&b), 0.0).unwrap().to_tensor();

        assert_eq!(dense(&via_vars), dense(&via_direct));
    }

    #[test]
    fn batch_norm_1d_rejects_rank4_input() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0f32; 8], &[1, 2, 2, 2]).unwrap());
        let bn = BatchNorm1d::new(2, BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM).unwrap();
        let err = bn.bind(&tape).forward(&x).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::RankMismatch { .. })
        ));
    }

    #[test]
    fn batch_norm_without_affine_train_rejects_channel_mismatch() {
        // affine なし（`weight`／`bias` とも `None`）構成では
        // `Var::batch_norm_with_batch_stats` 内部の `require_same_shape
        // (w.shape(), &[c])` が働かないため、`forward_var` 自身が
        // `c == num_features` を検査しないと `update_running_stats`
        // （長さ `num_features` の running stats と入力由来の長さ `c`
        // の batch 統計を `zip` する）で debug_assert panic（debug）・
        // 黙った切り詰め（release）を招く（codex-review P1 指摘。
        // イシュー #1732 fix ループ）。ここでは num_features=3 に対し
        // c=2 の入力を渡し、panic せず型付きエラーで拒否されることを
        // 確認する。
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]).unwrap());
        let bn =
            BatchNorm1d::without_affine(3, BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM)
                .unwrap();
        let err = bn.bind(&tape).forward(&x).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
        ));
        // running stats・num_batches_tracked は拒否時に不変のまま
        // （`forward_var` が `update_running_stats` 呼び出し前に
        // エラーで早期 return するため）。
        assert_eq!(bn.num_batches_tracked(), 0);
    }

    #[test]
    fn batch_norm_2d_rejects_rank2_input() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]).unwrap());
        let bn = BatchNorm2d::new(2, BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM).unwrap();
        let err = bn.bind(&tape).forward(&x).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::RankMismatch { .. })
        ));
    }

    #[test]
    fn batch_norm_train_updates_running_stats_and_tracked_count() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]).unwrap());
        let bn = BatchNorm1d::new(2, 0.0, 1.0).unwrap();
        // momentum=1.0: running <- batch_stat そのもの（unbiased 化は
        // running_var にのみ適用）。
        bn.bind(&tape).forward(&x).unwrap();
        assert_eq!(bn.num_batches_tracked(), 1);
        // ch0: [1, -1] mean=0.0 biased_var=1.0 -> unbiased (M=2) = 2.0
        // ch1: [2, 0.5] mean=1.25 biased_var=0.5625 -> unbiased = 1.125
        let mean = dense(&bn.running_mean());
        let var = dense(&bn.running_var());
        assert!((mean[0] as f64 - 0.0).abs() < 1e-5);
        assert!((mean[1] as f64 - 1.25).abs() < 1e-5);
        assert!((var[0] as f64 - 2.0).abs() < 1e-4);
        assert!((var[1] as f64 - 1.125).abs() < 1e-4);
    }

    #[test]
    fn batch_norm_eval_does_not_update_running_stats() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]).unwrap());
        let mut bn = BatchNorm1d::new(2, 0.0, 1.0).unwrap();
        bn.core.set_training(false);
        bn.bind(&tape).forward(&x).unwrap();
        assert_eq!(bn.num_batches_tracked(), 0);
        assert_eq!(dense(&bn.running_mean()), vec![0.0f32; 2]);
        assert_eq!(dense(&bn.running_var()), vec![1.0f32; 2]);
    }

    #[test]
    fn batch_norm_eval_uses_running_stats_not_batch_stats() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        // running_mean/var far from batch stats -> eval output must
        // reflect running stats, not the batch's own mean/var.
        let running_mean = Tensor::new(vec![100.0f32, 100.0], &[2]).unwrap();
        let running_var = Tensor::new(vec![1.0f32, 1.0], &[2]).unwrap();
        let mut bn = BatchNorm1d::from_parameters(
            None,
            None,
            running_mean,
            running_var,
            0.0,
            BATCH_NORM_DEFAULT_MOMENTUM,
        )
        .unwrap();
        bn.core.set_training(false);
        let x = tape.var(&Tensor::new(vec![100.0, 100.0, 100.0, 100.0], &[2, 2]).unwrap());
        let out = bn.bind(&tape).forward(&x).unwrap().to_tensor();
        // (100-100)/sqrt(1) == 0.0 for every element
        assert_eq!(dense(&out), vec![0.0f32; 4]);
    }
}
