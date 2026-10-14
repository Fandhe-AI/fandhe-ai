//! 全結合層（Linear/Dense）。`nn` モジュール（TASK-9.1a・#91）の第 1 分割。
//!
//! `docs/spec/05-tasks.md` TASK-9.1（REQ-9・M3）に基づき、自作テンソル
//! （`tensor-core`）・自作 autodiff（`Tape`/`Var`）の上に PyTorch
//! `nn.Linear` 相当の層を再実装する。参照実装は PoC-v2-2 の MLP
//! （`docs/spec/03-poc/poc-v2-2-autodiff/code/rust/src/mlp.rs`）の
//! `x.matmul(w) → add(bias)` 構成。
//!
//! **`Tape` ライフサイクルとの関係**（`tape.rs` の「学習ループでの運用」
//! 節参照）: `Tape` はステップごとに生成・破棄される前提のため、
//! パラメータの永続的な値（`Tensor<f32>`）を保持する `Linear` 本体と、
//! 特定ステップのテープへ登録した後の `Var` を保持する `LinearVars` を
//! 分離する。呼び出し元は毎ステップ `Linear::bind(&tape)` で
//! `LinearVars` を作り直し、`forward` を呼ぶ。

use std::cell::RefCell;

use fandhe_ai_tensor_core::{
    Activation, BackendOps, ScalarDType, ShapeError, Tensor, broadcast_shape, gemm_out_shape,
};

use crate::error::AutodiffError;
use crate::nn::init::{
    BIAS_SEED_SALT, SPECTRAL_U_SEED_SALT, SPECTRAL_V_SEED_SALT, WEIGHT_SEED_SALT, derive_seed,
    uniform_init,
};
use crate::tape::Tape;
use crate::var::Var;
use crate::weight_reparam_ops::{
    SPECTRAL_NORM_INIT_POWER_ITERATIONS, SpectralNormState, norm_except_dim,
};

/// `Linear` の `weight` スロットを再パラメータ化した状態（イシュー #3048。
/// 決定記録 `docs/facade-weight-reparam-wiring-decision.md` §5〜§10 の
/// 「(a) 層内再パラメータ化」）。`Linear` 型自身が保持するため、
/// `as_linear()` で `Linear` として観測される層の経路（`Sequential` の bind・
/// 学習・保存）が型を変えずに再パラメータ化を扱える。
///
/// - `WeightNorm`: `weight` スロットが `v`、ここが `g`（`dim` 以外が 1 の keepdim 形、
///   `dim = None` は rank 0）。有効重みは `Var::weight_norm(v, g, dim)`。
/// - `SpectralNorm`: `weight` スロットが `original`。有効重みは
///   `Var::spectral_norm(original, state, training)`。状態は forward が `&self` のため
///   `RefCell`（`nn/batch_norm.rs` の running stats が先例）。
enum WeightReparam {
    WeightNorm { g: Tensor<f32>, dim: Option<usize> },
    SpectralNorm { state: RefCell<SpectralNormState> },
}

/// 全結合層のパラメータ本体。`weight` は `[in_features, out_features]`
/// （PoC-v2-2 の `x.matmul(w)` 慣習。転置は持たない）、`bias` は
/// `Some` の場合 `[out_features]`。
pub struct Linear {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
    /// 層別 `requires_grad` 凍結フラグ（イシュー #2137。
    /// `Module::set_requires_grad`／`freeze` の実体）。`bind` が
    /// `weight`／`bias` を葉登録する際にこの値を渡す。既定 `true`
    /// （全コンストラクタで初期化。`nn/module.rs`「反映タイミング」
    /// 節参照）。
    requires_grad: bool,
    /// `weight` スロットの再パラメータ化状態（イシュー #3048）。`None` が従来の
    /// 素の重み。`Some` のとき `weight` は「生の一次テンソル」（weight_norm の `v`・
    /// spectral_norm の `original`）を指す。全コンストラクタで `None` 初期化。
    reparam: Option<WeightReparam>,
    /// train／eval モード（イシュー #3048）。`spectral_norm` が train の forward でだけ
    /// 状態を更新するために保持する（`Module::set_training` の実体）。既定 `true`。
    training: bool,
}

impl Linear {
    /// 決定的シードで `U(-1/√in_features, 1/√in_features)` の一様初期化
    /// を行う（PyTorch `nn.Linear` 既定初期化と同じ有効範囲。
    /// `nn/init.rs` 参照）。`bias: true` で `[out_features]` の bias を
    /// 同じ範囲・独立した導出シードで初期化する（weight/bias のシード
    /// 導出は `nn/init.rs::derive_seed` を参照。「同じ呼び出しシードから
    /// 2 系統を作る」設計上、単純な線形オフセットでは連番呼び出しシード
    /// で層を重ねる使い方の際に系列が衝突しうるため、ビットミキシング
    /// で独立させている）。
    ///
    /// `out_features == 0` は実行時に構築できる空テンソル（`tensor-core`
    /// はサイズ 0 軸を妥当な shape として扱う。`ops_shape.rs` の
    /// `matmul_zero_size_axis` 参照）としてそのまま受理する。一方
    /// `in_features == 0` は `bound = 1/√in_features` が非有限（inf）に
    /// なるため、テンソル生成に進む前に引数として弾く。この失敗は
    /// 「生成済み・生成中のテンソルの shape 不整合」ではなく「コンス
    /// トラクタ引数がそもそも構築不可能」という性質のため、
    /// `tensor-core::ShapeError` の既存 variant（`RankMismatch` 等）は
    /// いずれも意味的に適合せず、`AutodiffError::InvalidArgument` で
    /// 表現する（`error.rs` の doc 参照。review 指摘 #91: 当初
    /// `ShapeError::AxisOutOfRange` へ転用していたが撤回した）。
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Linear, AutodiffError> {
        if in_features == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Linear::new: in_features must be > 0 (1/sqrt(in_features) would be non-finite)"
                    .to_string(),
            ));
        }
        let bound = 1.0 / (in_features as f32).sqrt();
        let weight_seed = derive_seed(seed, WEIGHT_SEED_SALT);
        let weight_data = uniform_init(in_features * out_features, bound, weight_seed);
        let weight = Tensor::new(weight_data, &[in_features, out_features])?;
        let bias = if bias {
            let bias_seed = derive_seed(seed, BIAS_SEED_SALT);
            let bias_data = uniform_init(out_features, bound, bias_seed);
            Some(Tensor::new(bias_data, &[out_features])?)
        } else {
            None
        };
        Ok(Linear {
            weight,
            bias,
            requires_grad: true,
            reparam: None,
            training: true,
        })
    }

    /// 明示的な重み・バイアスから構築する（テスト・将来の safetensors
    /// ロード経路（REQ-7 系。本イシューではスコープ外）向けの入口）。
    /// `weight` は rank 2、`bias` を渡す場合は rank 1 かつ
    /// `bias.shape() == [weight.shape()[1]]` を要求する（A03: 外部由来
    /// パラメータを計算前に検証する契約。`.claude/rules/security.md`）。
    pub fn from_parameters(
        weight: Tensor<f32>,
        bias: Option<Tensor<f32>>,
    ) -> Result<Linear, AutodiffError> {
        if weight.rank() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: weight.rank(),
            }));
        }
        // `weight.shape()[0]`（in_features）が 0 の場合、`tensor-core::ops_shape::gemm_out_shape`
        // は `lhs[1]==rhs[0]==0` を zero-K パスとして妥当な shape 扱いで許容してしまうため、
        // forward はエラーにならず全要素 0.0 の出力を静かに返す（review 指摘 #91）。
        // `Linear::new` が同条件を `AutodiffError::InvalidArgument` で明示的に拒否しているのに
        // 対し、safetensors ロード等の外部由来パラメータ入口である `from_parameters` がこの
        // 検証を欠くと、壊れた／欠損した checkpoint（shape が `[0, N]` に縮退したもの）を
        // エラーにせず読み込み、学習・推論が常時ゼロ出力のまま進行しうる（A03: 外部由来
        // パラメータを計算前に検証する契約。`.claude/rules/security.md`）。`out_features == 0`
        // は妥当な shape として引き続き許容する（`new` と対称。docstring 参照）。
        if weight.shape()[0] == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Linear::from_parameters: weight.shape()[0] (in_features) must be > 0 \
                 (zero-K matmul would silently produce an all-zero output)"
                    .to_string(),
            ));
        }
        if let Some(ref b) = bias {
            if b.rank() != 1 {
                return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                    expected: 1,
                    actual: b.rank(),
                }));
            }
            let out_features = weight.shape()[1];
            if b.shape() != [out_features] {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: b.shape().to_vec(),
                    rhs: vec![out_features],
                }));
            }
        }
        Ok(Linear {
            weight,
            bias,
            requires_grad: true,
            reparam: None,
            training: true,
        })
    }

    /// このステップの `tape` へ `weight`/`bias` を葉ノードとして登録し、
    /// `forward` を呼べる `LinearVars` を返す。`Tape::var`（`tape.rs`）を
    /// 経由するため、返る `Var` はこの `tape` に属する（クロステープ
    /// 検査の対象になる）。
    pub fn bind<'t>(&self, tape: &'t Tape) -> LinearVars<'t> {
        let weight = tape.var_with_requires_grad(&self.weight, self.requires_grad);
        let bias = self
            .bias
            .as_ref()
            .map(|b| tape.var_with_requires_grad(b, self.requires_grad));
        LinearVars { weight, bias }
    }

    /// `weight` スロットのテンソル。**再パラメータ化済み（[`Self::is_parametrized`]）の層では
    /// 有効重みではなく生の一次テンソル**（weight_norm の `v`・spectral_norm の `original`）を
    /// 返す（イシュー #3048。シグネチャ不変の代わりに意味が変わる点を呼び出し側が
    /// `is_parametrized` で判別する）。有効重みを使う計算は [`Self::forward_bound`] 経由に
    /// 限る。
    pub fn weight(&self) -> &Tensor<f32> {
        &self.weight
    }

    /// `weight` が再パラメータ化（weight_norm／spectral_norm）済みか（イシュー #3048）。
    /// `Sequential` の bind・学習・常駐・host・ONNX の各消費箇所が、生の `weight()` を有効重みと
    /// 誤読しないよう分岐するための判定。
    pub fn is_parametrized(&self) -> bool {
        self.reparam.is_some()
    }

    /// weight_norm 適用済みなら `g`（`dim` 以外が 1 の keepdim 形、`dim = None` は rank 0）。
    pub fn weight_norm_g(&self) -> Option<&Tensor<f32>> {
        match &self.reparam {
            Some(WeightReparam::WeightNorm { g, .. }) => Some(g),
            _ => None,
        }
    }

    /// weight_norm 適用済みなら適用時の `dim`（`Some(None)` は `dim = None`）。
    pub fn weight_norm_dim(&self) -> Option<Option<usize>> {
        match &self.reparam {
            Some(WeightReparam::WeightNorm { dim, .. }) => Some(*dim),
            _ => None,
        }
    }

    /// spectral_norm 適用済みなら現在の状態の複製（`u`／`v`・`eps`・反復回数）。複製のため
    /// 取得後に層の状態は進まない（参照実装との突合用）。
    pub fn spectral_norm_state(&self) -> Option<SpectralNormState> {
        match &self.reparam {
            Some(WeightReparam::SpectralNorm { state }) => {
                state.try_borrow().ok().map(|st| st.clone())
            }
            _ => None,
        }
    }

    /// `weight` を weight_norm（`w = g · v / ‖v‖`）へ再パラメータ化する（イシュー #3048。
    /// 決定記録 §5 D-A (a)）。`weight` スロットは `v` になり、`g` は
    /// `norm_except_dim(weight, dim)` で初期化するため、適用直後の有効重みは元の重みと一致する。
    ///
    /// `dim` は `None`（テンソル全体）／`Some(0)`（in 軸ごと）／`Some(1)`（out 軸ごと。
    /// PyTorch `nn.Linear` の `dim=0`〈out 軸ごと〉相当は、転置を持たない `[in, out]` の本層では
    /// `Some(1)`）。検証・計算を先に済ませてから代入するため、失敗時は層が変化しない。
    /// 二重適用は `InvalidArgument`。compile 後の適用は facade
    /// `Sequential::parametrize_weight_norm` が拒否する。
    pub fn parametrize_weight_norm(&mut self, dim: Option<usize>) -> Result<(), AutodiffError> {
        if self.reparam.is_some() {
            return Err(AutodiffError::InvalidArgument(
                "Linear::parametrize_weight_norm: already parametrized".to_string(),
            ));
        }
        if matches!(dim, Some(d) if d > 1) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Linear::parametrize_weight_norm: dim must be None, Some(0) or Some(1) for a \
                 rank-2 weight (dim={dim:?})"
            )));
        }
        let g = norm_except_dim(&self.weight, dim)?;
        self.reparam = Some(WeightReparam::WeightNorm { g, dim });
        Ok(())
    }

    /// `weight` を spectral_norm（`w = W / σ`）へ再パラメータ化する（イシュー #3048。決定記録
    /// §5 D-A (a)）。`weight` スロットは `original` になる。行軸は `[in, out]` の out 軸（`dim = 1`）
    /// 固定。`u0`（長さ out）・`v0`（長さ in）は `seed` から決定的に導出し
    /// （`nn/init.rs` の `SPECTRAL_U_SEED_SALT`／`SPECTRAL_V_SEED_SALT`）、PyTorch の初期化と同じく
    /// `SPECTRAL_NORM_INIT_POWER_ITERATIONS` 回の予備反復を行う。`n_power_iterations >= 1`・
    /// `eps` が有限かつ非負は `SpectralNormState::from_vectors` が検証する。失敗時は層が変化しない。
    pub fn parametrize_spectral_norm(
        &mut self,
        n_power_iterations: usize,
        eps: f32,
        seed: u64,
    ) -> Result<(), AutodiffError> {
        if self.reparam.is_some() {
            return Err(AutodiffError::InvalidArgument(
                "Linear::parametrize_spectral_norm: already parametrized".to_string(),
            ));
        }
        let in_features = self.weight.shape()[0];
        let out_features = self.weight.shape()[1];
        let bound = 1.0 / (in_features.max(1) as f32).sqrt();
        let u0 = uniform_init(out_features, bound, derive_seed(seed, SPECTRAL_U_SEED_SALT));
        let v0 = uniform_init(in_features, bound, derive_seed(seed, SPECTRAL_V_SEED_SALT));
        let mut state = SpectralNormState::from_vectors(
            self.weight.shape(),
            1,
            &u0,
            &v0,
            n_power_iterations,
            eps,
        )?;
        state.power_iterate(&self.weight, SPECTRAL_NORM_INIT_POWER_ITERATIONS)?;
        self.reparam = Some(WeightReparam::SpectralNorm {
            state: RefCell::new(state),
        });
        Ok(())
    }

    /// [`Self::parametrize_weight_norm`] の `self` 消費版（ビルダー用）。
    pub fn with_weight_norm(mut self, dim: Option<usize>) -> Result<Linear, AutodiffError> {
        self.parametrize_weight_norm(dim)?;
        Ok(self)
    }

    /// [`Self::parametrize_spectral_norm`] の `self` 消費版（ビルダー用）。
    pub fn with_spectral_norm(
        mut self,
        n_power_iterations: usize,
        eps: f32,
        seed: u64,
    ) -> Result<Linear, AutodiffError> {
        self.parametrize_spectral_norm(n_power_iterations, eps, seed)?;
        Ok(self)
    }

    /// weight_norm 適用済みの層の `g` を `tape` の葉として登録する（それ以外は `None`）。
    /// `LinearVars` へフィールドを足すと外部の struct リテラル構築を壊す（破壊的変更）ため、
    /// `bind` は不変のまま補助葉だけ別名で登録する（決定記録 §14.1）。`requires_grad` は
    /// `weight` と同じフラグを使う。
    pub fn bind_weight_norm_g<'t>(&self, tape: &'t Tape) -> Option<Var<'t>> {
        self.weight_norm_g()
            .map(|g| tape.var_with_requires_grad(g, self.requires_grad))
    }

    /// 再パラメータ化を含む 1 ステップ分の forward（イシュー #3048）。`vars` は [`Self::bind`]、
    /// `g` は [`Self::bind_weight_norm_g`] の結果。`fuse_relu` は次層が ReLU の場合の epilogue 融合。
    ///
    /// 非再パラメータ化層は従来の `LinearVars::forward`／`forward_with_activation(Relu)` と
    /// bit 同一。再パラメータ化層は有効重みを tape 上で再構成してから同じ演算列を適用する。
    /// 入力 shape は有効重み再構成（spectral_norm は状態を進める）より前に検査するため、
    /// shape 不一致の入力では状態が進まない。spectral_norm の更新は train モードの forward 1 回につき
    /// 1 回（PyTorch と同じ契約。同一 step で複数回 forward すると複数回進む）。
    pub fn forward_bound<'t>(
        &self,
        vars: &LinearVars<'t>,
        g: Option<&Var<'t>>,
        input: &Var<'t>,
        fuse_relu: bool,
    ) -> Result<Var<'t>, AutodiffError> {
        let w_eff = match &self.reparam {
            None => {
                return if fuse_relu {
                    vars.forward_with_activation(input, Activation::Relu)
                } else {
                    vars.forward(input)
                };
            }
            Some(reparam) => {
                // rank≥3 を暗黙受理しない 2 次元厳密検査（`LinearVars::forward` と同じ境界）を、
                // 状態更新を伴う再構成より先に行う。
                gemm_out_shape(&input.shape(), &vars.weight.shape())?;
                match reparam {
                    WeightReparam::WeightNorm { dim, .. } => {
                        let g = g.ok_or_else(|| {
                            AutodiffError::InvalidArgument(
                                "Linear::forward_bound: weight_norm の g が bind されていない"
                                    .to_string(),
                            )
                        })?;
                        vars.weight.weight_norm(g, *dim)?
                    }
                    WeightReparam::SpectralNorm { state } => {
                        let mut st = state.try_borrow_mut().map_err(|_| {
                            AutodiffError::InvalidArgument(
                                "Linear::forward_bound: spectral_norm 状態が使用中（再入）"
                                    .to_string(),
                            )
                        })?;
                        vars.weight.spectral_norm(&mut st, self.training)?
                    }
                }
            }
        };
        if fuse_relu {
            input.linear_act(&w_eff, vars.bias.as_ref(), Activation::Relu)
        } else {
            let y = input.matmul(&w_eff)?;
            match &vars.bias {
                Some(bias) => y.add(bias),
                None => Ok(y),
            }
        }
    }

    /// `Module::set_training` の実体（イシュー #3048。spectral_norm の更新可否のみに効く）。
    ///
    /// **再パラメータ化済みの層だけがモードを保持する**。非再パラメータ化の `Linear` は従来どおり
    /// train／eval で挙動が変わらない無状態層として扱い、`training()` は常に `true` のまま
    /// （`nn/module.rs` の `set_training` 契約と `tests/nn_module_freeze.rs` の既存契約を変えない）。
    /// 再パラメータ化の適用前に呼んだ値は保持されないため、適用後に呼び直す
    /// （facade `Sequential::parametrize_*` が適用直後にコンテナのモードを同期する）。
    pub(crate) fn set_training(&mut self, training: bool) {
        if self.reparam.is_some() {
            self.training = training;
        }
    }

    /// `Module::training` の実体。
    pub(crate) fn training(&self) -> bool {
        self.training
    }

    pub fn bias(&self) -> Option<&Tensor<f32>> {
        self.bias.as_ref()
    }

    /// [`crate::nn::module::Module::set_requires_grad`]（`Linear` 実装。
    /// `module.rs` 参照）の本体。`weight`／`bias` の両方を一括で切り替
    /// える（per-layer 粒度。イシュー #2137）。
    pub(crate) fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// [`crate::nn::module::Module::requires_grad`]（`Linear` 実装）の
    /// 本体。
    pub(crate) fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// [`crate::nn::module::Module::set_parameter`]（`Linear` 実装。
    /// `module.rs` 参照）の本体。`"weight"`（常に）／`"bias"`（`Some`
    /// の場合のみ）を受理する（`named_parameters` の命名契約と対称。
    /// イシュー #1752）。shape 保存置換のみ（`Linear::from_parameters`
    /// の #91 契約と同じ）。
    pub(crate) fn set_parameter(
        &mut self,
        name: &str,
        value: Tensor<f32>,
    ) -> Result<(), AutodiffError> {
        // 再パラメータ化済みは分解パラメータのキー（`named_parameters` と対称。
        // イシュー #3048）だけ受理し、`weight` は拒否する。いずれも shape 保存置換のみで、
        // 再パラメータ化状態・`training`・`requires_grad` は保持する。
        if let Some(reparam) = &mut self.reparam {
            match (reparam, name) {
                (WeightReparam::WeightNorm { g, .. }, "parametrizations.weight.original0") => {
                    return replace_same_shape(g, value);
                }
                (WeightReparam::WeightNorm { .. }, "parametrizations.weight.original1")
                | (WeightReparam::SpectralNorm { .. }, "parametrizations.weight.original") => {
                    return replace_same_shape(&mut self.weight, value);
                }
                (_, "weight") => {
                    return Err(AutodiffError::InvalidArgument(
                        "Linear::set_parameter: `weight` is parametrized; use the \
                         `parametrizations.weight.*` keys"
                            .to_string(),
                    ));
                }
                _ => {}
            }
        }
        match name {
            "weight" if self.reparam.is_none() => replace_same_shape(&mut self.weight, value),
            "bias" => match &mut self.bias {
                Some(current) => replace_same_shape(current, value),
                None => Err(AutodiffError::InvalidArgument(format!(
                    "Linear::set_parameter: no parameter named `{name}` (this layer has no bias)"
                ))),
            },
            _ => Err(AutodiffError::InvalidArgument(format!(
                "Linear::set_parameter: no parameter named `{name}`"
            ))),
        }
    }

    /// [`crate::nn::module::Module::forward_host`]（`Linear` 実装。
    /// `gemm_out_shape` → `ops.gemm` → `ops.add` の非融合合成。
    /// `nn/module.rs` 参照）の epilogue 融合版（イシュー #1218・
    /// `docs/perf/cpu-infer-predict-profile.md`）。[`LinearVars::
    /// forward_with_activation`]（tape 経路。`Var::linear_act` 経由）と
    /// 対をなす tape 不要版で、`y = act(input.matmul(weight) (+ bias))`
    /// を `ops.gemm_bias_act` へ 1 呼び出しで委ねる。
    ///
    /// **呼び出し元が守る契約**: `ops.gemm_bias_act` の融合オーバーライドが
    /// 非融合合成（`gemm` → `add` → `act`）と bit 完全一致するバックエンド
    /// でのみ使うこと。CPU（CPU バックエンドクレートの `CpuBackendOps`）は
    /// `crates/backend-cpu/tests/gemm_epilogue_parity.rs` が MR/NR/MC/KC/NC
    /// 境界を跨ぐ形状グリッドで bit 完全一致を hard assert 済み
    /// （`docs/perf/cpu-gemm-epilogue-fusion.md`「数値一致」節）。CUDA／
    /// Metal の融合オーバーライドはこの一致が未保証のため（`docs/
    /// inference-forward-fixed-cost-design.md` §3.1「段階 A」の
    /// bit-exactness 契約は `Module::forward_host`〈trait レベル・汎用
    /// バックエンド向け〉に適用され続ける）、汎用 `&dyn BackendOps` を
    /// 受け取る `Module::forward_host` 自体はこのメソッドを使わず非融合
    /// のまま維持する。呼び出し元は現状 CPU 固定経路
    /// （`fandhe_ai_facade::compat::sequential::Sequential::
    /// predict_tape_free_with_ops`。Linear→ReLU の先読み結線）に限定する。
    ///
    /// **エラー型の一致契約**: `forward_host` と同じ理由（同ファイル
    /// `forward_host` doc 参照）で、`ops.gemm_bias_act` の `?` に検査を
    /// 任せず `gemm_out_shape`／`broadcast_shape` を先に呼び、shape
    /// 不整合を `forward_host` と同じ `AutodiffError::Shape` として返す
    /// （`AutodiffError::Backend` へ variant が変わらないようにする）。
    pub fn forward_host_with_activation(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
        act: Activation,
    ) -> Result<Tensor<f32>, AutodiffError> {
        if self.is_parametrized() {
            return Err(AutodiffError::Backend(
                fandhe_ai_tensor_core::BackendError::Unsupported(
                    "Linear::forward_host_with_activation: weight が再パラメータ化済み（有効重みを \
                     host 経路で再構成しない。イシュー #3048）"
                        .to_string(),
                ),
            ));
        }
        let out_shape = gemm_out_shape(input.shape(), self.weight.shape())?;
        if let Some(ref bias) = self.bias {
            broadcast_shape(&out_shape, bias.shape())?;
        }
        Ok(ops.gemm_bias_act(input, &self.weight, self.bias.as_ref(), act)?)
    }
}

/// shape 保存置換の共通処理（`Linear::set_parameter`）。shape 不一致は `ShapeMismatch`。
fn replace_same_shape(slot: &mut Tensor<f32>, value: Tensor<f32>) -> Result<(), AutodiffError> {
    if value.shape() != slot.shape() {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: value.shape().to_vec(),
            rhs: slot.shape().to_vec(),
        }));
    }
    *slot = value;
    Ok(())
}

/// `Linear::bind` が返す、1 ステップ分のテープに登録済みパラメータ。
/// `weight`/`bias` を公開する理由: `Tape::backward` 後に
/// `Gradients::get(&vars.weight)` で勾配を取り出すのは呼び出し側
/// （optimizer。#192・本イシューではスコープ外）の責務であり、
/// `LinearVars` 自身は勾配更新 API を持たない。
pub struct LinearVars<'t> {
    pub weight: Var<'t>,
    pub bias: Option<Var<'t>>,
}

impl<'t> LinearVars<'t> {
    /// `y = input.matmul(weight) (+ bias)`。`input` は `[batch,
    /// in_features]`（2 次元）を想定し、出力は `[batch, out_features]`。
    /// bias 加算は `Var::add` の broadcast（`[batch, out_features]` +
    /// `[out_features]`）に委ねるため、bias 勾配の batch 軸縮約は既存の
    /// `reduce_to_shape` 機構（`grad.rs`）でそのまま成立する（TASK-9.1a
    /// 計画 §2 参照）。
    ///
    /// **rank≥3 は対象外（イシュー #1715）**: `Var::matmul` 自体は
    /// イシュー #1715 でバッチ次元（rank≥3）を受理するよう拡張された
    /// が、`nn::Linear`（PyTorch 互換の rank≥3 入力〈`[*, in_features]`
    /// の任意の先頭次元〉対応）は本イシューのスコープ外（実装計画
    /// §8）。本メソッドは `gemm_out_shape`（2 次元厳密版）で明示検査
    /// することで、その挙動変更を `Var::matmul` 側の拡張から独立に
    /// 固定する（`input.matmul(&self.weight)?` が rank≥3 を暗黙に
    /// 受理してしまうのを防ぐ fail-closed な境界）。
    pub fn forward(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        gemm_out_shape(&input.shape(), &self.weight.shape())?;
        let y = input.matmul(&self.weight)?;
        match &self.bias {
            Some(bias) => y.add(bias),
            None => Ok(y),
        }
    }

    /// [`Self::forward`] の epilogue 融合版（イシュー #1044・`docs/
    /// kernel-fusion.md` §2.2「学習経路への結線」）。`y = act(input.matmul(
    /// weight) (+ bias))` を `Var::linear_act`（`var.rs`）経由で 1 ノード
    /// （`crate::tape::Op::LinearAct`。非公開のためコードスパン表記で
    /// 参照しリンク化しない）として記録し、`BackendOps::
    /// gemm_bias_act`（epilogue 融合カーネル。CPU／CUDA／Metal とも
    /// オーバーライド済み）へ直接委ねる。
    ///
    /// 唯一の呼び出し元は `fandhe_ai_facade::compat::sequential::
    /// Sequential`（次層が `ReLU` かを先読みし、その場合のみ
    /// `Activation::Relu` を渡して `ReLU` 層自体のノード追加をスキップ
    /// する。次層が `ReLU` でなければ `Activation::None` で bias のみ
    /// 融合する）。`Var::linear_act` が bias の broadcast 可否を
    /// `Var::add` と同じ判定で検査するため、本メソッド自体は追加の
    /// shape 検査を持たない。
    pub fn forward_with_activation(
        &self,
        input: &Var<'t>,
        act: fandhe_ai_tensor_core::Activation,
    ) -> Result<Var<'t>, AutodiffError> {
        input.linear_act(&self.weight, self.bias.as_ref(), act)
    }
}

/// [`LinearVars::forward_with_activation`] の opt-in 低精度版（イシュー
/// #1960・親 #1626／#1648）。`y = act(input.matmul(weight) (+ bias))` を
/// `dtype`（[`ScalarDType::F16`]／[`ScalarDType::Bf16`]）で指定した
/// 精度で計算する（**f32 master weight**: `vars.weight`／`vars.bias`
/// 自体は変更せず、`TypedOps<T>` 経由の forward 計算のみが低精度）。
///
/// **`LinearVars` へメソッドとして追加しない理由**: `LinearVars` は
/// `pub weight`／`pub bias` の公開フィールドを持つ struct であり
/// （`crates/facade/tests/mha_backend_parity.rs` 等が外部から struct
/// literal で構築する。facade crates.io 公開互換のため破壊的変更を
/// 避ける必要がある）、フィールド追加なしに opt-in 精度指定を渡す
/// 手段としてメソッドではなく自由関数を選んだ（実装計画 §2.2）。
///
/// **backward は常に f32**: `Var::linear_act_low_precision`
/// （唯一の呼び出し元）が記録する `Op::LinearAct` ノードは `grad::vjp`
/// から見れば通常の `LinearAct` と同一であり、`compute_dtype`
/// フィールドは forward 経路の記録専用（VJP は読まない）。真の混合
/// 精度（勾配も低精度で計算する経路）・`DeviceParamStore` 常駐経路
/// への結線は本関数のスコープ外（`docs/autodiff-low-precision-linear-
/// design.md` §6「スコープ外」）。
///
/// **`Activation` が `#[non_exhaustive]`**：未対応 variant（`None`／
/// `Relu` 以外）は `tensor-core::linear_forward_low_precision` 側で
/// `AutodiffError::Backend(BackendError::InvalidArgument(_))` として
/// fail-closed に拒否される。`typed_ops_f16`／`typed_ops_bf16`
/// accessor が `None`（低精度カーネル未実装のバックエンド）の場合も
/// 同様に `AutodiffError::Backend(BackendError::Unsupported(_))` を
/// 返す（f32 へのフォールバックはしない）。
///
/// **rank≥3 は対象外**: [`LinearVars::forward`] と同じく
/// `gemm_out_shape`（2 次元厳密版）で明示検査する。
pub fn linear_forward_low_precision<'t>(
    vars: &LinearVars<'t>,
    input: &Var<'t>,
    act: Activation,
    dtype: ScalarDType,
) -> Result<Var<'t>, AutodiffError> {
    gemm_out_shape(&input.shape(), &vars.weight.shape())?;
    input.linear_act_low_precision(&vars.weight, vars.bias.as_ref(), act, dtype)
}

#[cfg(test)]
mod tests {
    //! [`Linear::forward_host_with_activation`]（イシュー #1218・`docs/
    //! perf/cpu-infer-predict-profile.md`）の単体テスト。ここでは
    //! `BackendOps::gemm_bias_act` の**トレイト既定実装**（`gemm` →
    //! `add` → `act` の合成。`tensor-core/src/backend_ops.rs` 参照）を
    //! 経由させるため、`gemm`/`add`/`relu` を素朴に実装するだけの
    //! テスト用 `BackendOps`（CPU バックエンドクレートへの新規依存を
    //! 避ける。同クレートの `gemm_bias_act` 単体テストが同型の
    //! `ComputingMockOps` を使う先例と同じ方式）を使う。

    use super::*;
    use crate::nn::module::Module;
    use fandhe_ai_tensor_core::{Activation, BackendError, BackendOps, device::Device};

    struct ComputingMockOps;

    impl BackendOps for ComputingMockOps {
        fn device(&self) -> Device {
            Device::Cpu
        }

        fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            let (m, k) = (a.shape()[0], a.shape()[1]);
            let n = b.shape()[1];
            let a_data = a.as_slice().expect("test: a must be contiguous");
            let b_data = b.as_slice().expect("test: b must be contiguous");
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc = a_data[i * k + p].mul_add(b_data[p * n + j], acc);
                    }
                    out[i * n + j] = acc;
                }
            }
            Tensor::new(out, &[m, n]).map_err(BackendError::ShapeMismatch)
        }

        fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            let a_shape = a.shape().to_vec();
            let a_data = a.as_slice().expect("test: a must be contiguous");
            let b_data = b.as_slice().expect("test: b must be contiguous");
            let n = a_shape[1];
            let out: Vec<f32> = a_data
                .iter()
                .enumerate()
                .map(|(idx, x)| x + b_data[idx % n])
                .collect();
            Tensor::new(out, &a_shape).map_err(BackendError::ShapeMismatch)
        }

        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: mul".into()))
        }

        fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            let data = a.as_slice().expect("test: a must be contiguous");
            let out: Vec<f32> = data.iter().map(|x| x.max(0.0)).collect();
            Tensor::new(out, a.shape()).map_err(BackendError::ShapeMismatch)
        }

        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: exp".into()))
        }

        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: tanh".into()))
        }

        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: sum".into()))
        }

        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: max".into()))
        }
    }

    fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
        t.as_slice()
            .expect("test: expected contiguous tensor")
            .to_vec()
    }

    #[test]
    fn forward_host_with_activation_relu_matches_forward_host_then_relu() {
        let ops = ComputingMockOps;
        let linear = Linear::new(4, 6, true, 7).unwrap();
        let input = Tensor::new(
            vec![0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
            &[2, 4],
        )
        .unwrap();

        let fused = linear
            .forward_host_with_activation(&ops, &input, Activation::Relu)
            .unwrap();
        let non_fused = {
            let y = linear.forward_host(&ops, &input).unwrap();
            ops.relu(&y).unwrap()
        };

        assert_eq!(dense_vec(&fused), dense_vec(&non_fused));
    }

    #[test]
    fn forward_host_with_activation_none_matches_forward_host() {
        let ops = ComputingMockOps;
        let linear = Linear::new(3, 5, false, 11).unwrap();
        let input = Tensor::new(vec![0.2_f32, -0.1, 0.4, 0.3, -0.5, 0.1], &[2, 3]).unwrap();

        let fused = linear
            .forward_host_with_activation(&ops, &input, Activation::None)
            .unwrap();
        let non_fused = linear.forward_host(&ops, &input).unwrap();

        assert_eq!(dense_vec(&fused), dense_vec(&non_fused));
    }

    #[test]
    fn forward_host_with_activation_rejects_matmul_shape_mismatch_as_shape_error() {
        // `forward_host` と同じエラー variant 一致契約（`nn/module.rs`
        // doc 参照）: `ops.gemm_bias_act` の `?` に検査を任せず
        // `gemm_out_shape` を先に呼ぶため、shape 不整合は
        // `AutodiffError::Shape` として返る（`AutodiffError::Backend`
        // ではない）。
        let ops = ComputingMockOps;
        let linear = Linear::new(4, 6, true, 7).unwrap();
        // in_features=4 の重みに対し in_features=5 の入力を渡す。
        let input = Tensor::new(vec![0.0_f32; 2 * 5], &[2, 5]).unwrap();

        let err = linear
            .forward_host_with_activation(&ops, &input, Activation::Relu)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));

        // 同じ入力で forward_host も同じ variant を返すことを併せて確認する
        // （新旧 2 メソッド間でエラー variant が食い違わないことの回帰）。
        let err2 = linear.forward_host(&ops, &input).unwrap_err();
        assert!(matches!(err2, AutodiffError::Shape(_)));
    }

    #[test]
    fn forward_host_with_activation_rejects_bias_broadcast_mismatch_as_shape_error() {
        let ops = ComputingMockOps;
        // bias が weight の out_features（6）と食い違う shape になる
        // ケースを直接構築する（`Linear::new` は bias を自動導出するため
        // `from_parameters` で意図的に壊れた bias を渡す）。
        let weight = Tensor::new(vec![0.1_f32; 4 * 6], &[4, 6]).unwrap();
        let bad_bias = Tensor::new(vec![0.0_f32; 3], &[3]).unwrap();
        let linear = Linear {
            weight,
            bias: Some(bad_bias),
            requires_grad: true,
            reparam: None,
            training: true,
        };
        let input = Tensor::new(vec![0.0_f32; 2 * 4], &[2, 4]).unwrap();

        let err = linear
            .forward_host_with_activation(&ops, &input, Activation::Relu)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }

    // イシュー #1566・PR #1659→#1665→#1666 取り込み後の追加ユーザー
    // 承認（2026-09-12）: `LinearVars::forward`（`matmul → add` の
    // 非融合合成。本テストモジュールの対象）の bias 勾配も
    // `crate::grad::reduce_bias_grad`（f64 相当）へ統一されたことの
    // 回帰テスト（`crate::grad::vjp` の `Op::Add` 分岐）。

    /// `[1e8, 1.0, -1e8]` のような相殺入力で、`f32` 逐次和（旧
    /// `reduce_to_shape`）なら失われる寄与（真値 `1.0`）が
    /// `LinearVars::forward` 経由の bias 勾配でも保持されることを
    /// 確認する（`eval::reduce_bias_grad_rows_tests::preserves_
    /// cancelling_contribution_via_f64_accumulator` の
    /// `LinearVars::forward` 版）。`weight` は `x` を素通りさせない
    /// 値（`x` 自体を全 0 にして matmul の寄与をゼロにし、bias の
    /// broadcast 加算だけで `pred` を決める）ため、`d_pred`（本テストの
    /// 重み付き和の重み `s` そのもの）が bias の縮約対象 `g` に一致する。
    #[test]
    fn linear_vars_forward_bias_grad_preserves_cancelling_contribution() {
        let tape = Tape::new();
        // x: [3, 1] 全 0（matmul 結果を 0 に固定し、pred = 0 + bias の
        // broadcast のみで決まるようにする）。weight: [1, 1] は任意
        // （x が 0 のため forward 結果に影響しない）。
        let x = tape.var(&Tensor::new(vec![0.0f32; 3], &[3, 1]).unwrap());
        let weight = tape.var(&Tensor::new(vec![5.0f32], &[1, 1]).unwrap());
        let bias = tape.var(&Tensor::new(vec![0.0f32], &[1]).unwrap());
        let linear_vars = LinearVars {
            weight,
            bias: Some(bias),
        };

        let pred = linear_vars.forward(&x).unwrap();
        // L = sum(pred ⊙ s)（`s = [1e8, 1.0, -1e8]`）とすると
        // dL/dpred = s（`grad.rs` 冒頭 doc の「固定の重みテンソルによる
        // スカラー射影」と同じ手法）。bias の VJP（`Op::Add`）は
        // `reduce_bias_grad(s, [1])` = `s` の列ごとの和（列は 1 本の
        // ため単純合計）となり、真値は `1e8 + 1.0 + (-1e8) = 1.0`。
        let s = tape.var(&Tensor::new(vec![1.0e8f32, 1.0, -1.0e8], &[3, 1]).unwrap());
        let loss = pred.mul(&s).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let bias_grad = grads
            .get(&bias)
            .unwrap()
            .expect("bias は backward で到達するはず");

        assert_eq!(bias_grad.shape(), &[1]);
        assert_eq!(
            bias_grad.get(&[0]).unwrap(),
            1.0f32,
            "LinearVars::forward（Op::Add 経由）の bias 勾配は f64 相当の \
             アキュムレータ（reduce_bias_grad）を使うため、f32 逐次和なら \
             失われる寄与（1e8 + 1.0 + (-1e8) = 1.0）を保持するはず"
        );
    }

    /// `LinearVars::forward`（`Op::Add` 経由）と `LinearVars::
    /// forward_with_activation`（`Op::LinearAct` 経由。epilogue 融合）の
    /// bias 勾配が bit 完全一致することを確認する（fresh〈非融合〉／
    /// fused の数値方式統一。resident〈`Op::LinearResident`〉側は
    /// `crates/facade/tests/device_param_store_grad_readout.rs::
    /// param_grads_to_host_matches_host_only_path_two_layer` が
    /// カバーする——host 側はいずれも同一の `reduce_bias_grad` を
    /// 経由するため、入力が同一なら bit 完全一致する）。
    #[test]
    fn add_path_bias_grad_matches_linear_act_bias_grad_bit_exact() {
        let x_data = Tensor::new(vec![1.0, 2.0, -3.0, 4.0, 0.5, -0.5], &[3, 2]).unwrap();
        let w_data = Tensor::new(vec![1.0, 0.5, -0.5, 1.0], &[2, 2]).unwrap();
        let bias_data = Tensor::new(vec![0.1, -0.2], &[2]).unwrap();
        let target_data = Tensor::new(vec![0.0, 1.0, -1.0, 0.5, 0.2, -0.3], &[3, 2]).unwrap();

        // `Op::Add` 経由（`LinearVars::forward` と同型の非融合合成）。
        let add_tape = Tape::new();
        let x1 = add_tape.var(&x_data);
        let w1 = add_tape.var(&w_data);
        let b1 = add_tape.var(&bias_data);
        let t1 = add_tape.var(&target_data);
        let linear_vars1 = LinearVars {
            weight: w1,
            bias: Some(b1),
        };
        let pred1 = linear_vars1.forward(&x1).unwrap();
        let loss1 = pred1.mse_loss(&t1).unwrap();
        let grads1 = add_tape.backward(&loss1).unwrap();
        let bias_grad_add = grads1
            .get(&b1)
            .unwrap()
            .expect("bias は backward で到達するはず")
            .clone();

        // `Op::LinearAct` 経由（`forward_with_activation`。epilogue 融合。
        // `Activation::None` のため数式上は `Op::Add` 経由と同一）。
        let act_tape = Tape::new();
        let x2 = act_tape.var(&x_data);
        let w2 = act_tape.var(&w_data);
        let b2 = act_tape.var(&bias_data);
        let t2 = act_tape.var(&target_data);
        let linear_vars2 = LinearVars {
            weight: w2,
            bias: Some(b2),
        };
        let pred2 = linear_vars2
            .forward_with_activation(&x2, Activation::None)
            .unwrap();
        let loss2 = pred2.mse_loss(&t2).unwrap();
        let grads2 = act_tape.backward(&loss2).unwrap();
        let bias_grad_act = grads2
            .get(&b2)
            .unwrap()
            .expect("bias は backward で到達するはず")
            .clone();

        assert_eq!(
            bias_grad_add.contiguous().as_slice().unwrap(),
            bias_grad_act.contiguous().as_slice().unwrap(),
            "Op::Add 経由（LinearVars::forward 相当）と Op::LinearAct 経由の bias 勾配は \
             いずれもホスト reduce_bias_grad を経由するため bit 完全一致するはず"
        );
    }

    /// 個別シード API（`Linear::new(.., seed)`）とプロセスグローバル
    /// 決定的 RNG（`tensor-core::rng::manual_seed`）が完全に独立した
    /// 別機構であることを機構的に固定する（イシュー #1724・受入基準
    /// 「既存の個別シード API との整合性」）。`manual_seed` を任意の値で
    /// 何度呼んでも `Linear::new(.., seed)` の出力（weight/bias）が
    /// 一切変わらないことを確認する。
    #[test]
    fn linear_new_is_unaffected_by_global_manual_seed_state() {
        // 他の rng テスト（`tensor-core::rng` の `#[cfg(test)]` 直列化
        // ロック）とは別クレートのテストのため直接の競合はないが、
        // `manual_seed` はプロセスグローバルであり `cargo test` の
        // 既定並列実行下で他クレートのテストと同時に走りうる。本テスト
        // は「`manual_seed` を呼んだ後でも `Linear::new` の出力が
        // 変わらない」ことのみを主張するため、他プロセス外テストとの
        // 競合があっても主張自体は影響を受けない（`manual_seed` の値に
        // 依存しない契約を検証している）。
        let baseline = Linear::new(4, 3, true, 7).unwrap();

        fandhe_ai_tensor_core::rng::manual_seed(1);
        let after_seed_1 = Linear::new(4, 3, true, 7).unwrap();

        fandhe_ai_tensor_core::rng::manual_seed(999_999);
        let after_seed_2 = Linear::new(4, 3, true, 7).unwrap();

        assert_eq!(
            baseline.weight.contiguous().as_slice().unwrap(),
            after_seed_1.weight.contiguous().as_slice().unwrap(),
            "manual_seed(1) の前後で Linear::new(.., 7) の weight が変化した \
             （グローバル RNG と個別シード API は独立のはず）"
        );
        assert_eq!(
            baseline.weight.contiguous().as_slice().unwrap(),
            after_seed_2.weight.contiguous().as_slice().unwrap(),
            "manual_seed(999_999) の前後で Linear::new(.., 7) の weight が変化した"
        );
        assert_eq!(
            baseline
                .bias
                .as_ref()
                .unwrap()
                .contiguous()
                .as_slice()
                .unwrap(),
            after_seed_1
                .bias
                .as_ref()
                .unwrap()
                .contiguous()
                .as_slice()
                .unwrap(),
            "manual_seed(1) の前後で Linear::new(.., 7) の bias が変化した"
        );
        assert_eq!(
            baseline
                .bias
                .as_ref()
                .unwrap()
                .contiguous()
                .as_slice()
                .unwrap(),
            after_seed_2
                .bias
                .as_ref()
                .unwrap()
                .contiguous()
                .as_slice()
                .unwrap(),
            "manual_seed(999_999) の前後で Linear::new(.., 7) の bias が変化した"
        );
    }

    // `Linear::set_parameter`（イシュー #1752・`state_dict`/
    // `load_state_dict` の基盤）の単体テスト。

    #[test]
    fn set_parameter_replaces_weight_in_place() {
        let mut linear = Linear::new(3, 2, true, 1).unwrap();
        let new_weight = Tensor::new(vec![9.0f32; 6], &[3, 2]).unwrap();
        Linear::set_parameter(&mut linear, "weight", new_weight.clone()).unwrap();
        assert_eq!(
            linear.weight().contiguous().as_slice().unwrap(),
            new_weight.contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn set_parameter_replaces_bias_in_place() {
        let mut linear = Linear::new(3, 2, true, 1).unwrap();
        let new_bias = Tensor::new(vec![5.0f32, 6.0f32], &[2]).unwrap();
        Linear::set_parameter(&mut linear, "bias", new_bias.clone()).unwrap();
        assert_eq!(
            linear.bias().unwrap().contiguous().as_slice().unwrap(),
            new_bias.contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn set_parameter_rejects_unknown_name() {
        let mut linear = Linear::new(3, 2, true, 1).unwrap();
        let dummy = linear.weight().clone();
        let err = Linear::set_parameter(&mut linear, "bogus", dummy)
            .expect_err("未知のパラメータ名は Err を返すはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn set_parameter_rejects_shape_mismatch() {
        let mut linear = Linear::new(3, 2, true, 1).unwrap();
        let wrong_shape = Tensor::new(vec![1.0f32; 4], &[2, 2]).unwrap();
        let err = Linear::set_parameter(&mut linear, "weight", wrong_shape)
            .expect_err("shape 不一致は Err を返すはず");
        assert!(matches!(err, AutodiffError::Shape(_)));
    }

    #[test]
    fn set_parameter_rejects_bias_when_layer_has_none() {
        let mut linear = Linear::new(3, 2, false, 1).unwrap();
        let new_bias = Tensor::new(vec![1.0f32, 2.0f32], &[2]).unwrap();
        let err = Linear::set_parameter(&mut linear, "bias", new_bias)
            .expect_err("bias を持たない層への `bias` 指定は Err を返すはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn set_parameter_err_leaves_layer_unchanged() {
        let mut linear = Linear::new(3, 2, true, 1).unwrap();
        let before = linear.weight().contiguous().as_slice().unwrap().to_vec();
        let wrong_shape = Tensor::new(vec![1.0f32; 4], &[2, 2]).unwrap();
        let _ = Linear::set_parameter(&mut linear, "weight", wrong_shape);
        assert_eq!(linear.weight().contiguous().as_slice().unwrap(), before);
    }
}
