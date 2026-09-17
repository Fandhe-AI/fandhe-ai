//! Keras 風 `compile()`／`fit()`／`evaluate()` 最小版（イシュー #1761・
//! 親 #1618・`docs/compat-api-scope.md` §1.2「Keras 風 `Sequential` の層
//! 追加と `compile()`／`fit()`／`evaluate()`／callbacks の最小版」行）。
//!
//! `Sequential::bind → forward → loss → backward → trainable_grads →
//! optimizer.step → apply_parameters` という定型（`sequential.rs`
//! モジュール doc の doctest・`tests/compat_sequential_train.rs`）を
//! `compile`／`fit`／`evaluate` の 3 メソッドへ畳み込む。**新規
//! `Op`／`BackendOps` メソッド／VJP／カーネルは一切追加しない**——
//! 既存公開 API（`Sequential::bind` 等・[`crate::optim`] の
//! optimizer・[`crate::data::DataLoader`]・`Var::mse_loss`／
//! `cross_entropy_loss`）の合成のみで実装する（REQ-9「薄いラッパーに
//! 徹する」）。
//!
//! callbacks（`EarlyStopping`／`ModelCheckpoint`／LR スケジューラ連携）・
//! `validation_data` は [`Sequential::fit_with_callbacks`]（イシュー
//! #1763・親 #1618・`super::callbacks`）で実装済み。[`Sequential::fit`]
//! は引き続き既存契約のまま（`fit_with_callbacks(x, y, config, None,
//! &mut [])` への委譲）。metrics・`DataLoader` を直接受ける `fit` 入口は
//! 対象外のまま（`docs/compat-callbacks-design.md` §8 参照）。

use crate::optim::{
    Adam, AdamConfig, AdamW, AdamWConfig, GradScaler, GradScalerConfig, Sgd, SgdConfig,
};
use crate::{AutodiffError, Tensor};
use fandhe_ai_autodiff::Reduction;
use fandhe_ai_tensor_core::Element;
use fandhe_ai_tensor_core::ScalarDType;
use fandhe_ai_tensor_core::data::{DataLoader, DataLoaderConfig, TensorDataset};

use super::callbacks::Callback;

use super::sequential::Sequential;

/// `compile_with_amp()` の低精度 forward dtype 指定（イシュー #1961・
/// 親 #1958。`docs/autodiff-low-precision-linear-design.md` §7「facade
/// 統合（#1961）」）。
///
/// **facade ローカルに閉じる理由**: `fandhe_ai_tensor_core::ScalarDType`
/// 自体の facade 再エクスポートは `docs/compat-api-scope.md` §5 未承認の
/// まま（イシュー #1939）——本 enum はその承認を経ずに `compile_with_amp`
/// が dtype を受け取れるようにするための facade 専用の薄い写像であり、
/// `ScalarDType` を facade 公開面へ直接持ち出さない（`to_scalar_dtype`
/// が内部でのみ変換する）。`#[non_exhaustive]` は `ScalarDType` 自体の
/// バリアント追加余地に追従するため。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmpDType {
    /// IEEE 754 半精度（仮数 10bit）。
    F16,
    /// bfloat16（仮数 7bit・指数幅は f32 と同一）。
    Bf16,
}

impl AmpDType {
    fn to_scalar_dtype(self) -> ScalarDType {
        match self {
            AmpDType::F16 => ScalarDType::F16,
            AmpDType::Bf16 => ScalarDType::Bf16,
        }
    }
}

/// [`Sequential::compile_with_amp`] の構成（イシュー #1961）。
///
/// `compute_dtype`（[`AmpDType`]。`Linear` 層 forward の低精度化）と
/// `grad_scaler`（[`GradScalerConfig`]。損失スケーリングのハイパー
/// パラメータ）を束ねる。両者は独立の機構——`compute_dtype` は
/// forward 経路（`linear_forward_low_precision`。backward は常に f32）、
/// `grad_scaler` は backward 後の勾配スケーリング（`GradScaler`）——
/// であり、本 struct は `fit` へ渡す前にこれらをまとめて検証・保持する
/// ための単なる入れ物（新規数値ロジックなし）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmpConfig {
    compute_dtype: AmpDType,
    grad_scaler: GradScalerConfig,
}

impl AmpConfig {
    /// `grad_scaler` は [`GradScalerConfig::default`]（PyTorch
    /// `torch.cuda.amp.GradScaler` の既定と同一値）で初期化する。
    pub fn new(compute_dtype: AmpDType) -> Self {
        AmpConfig {
            compute_dtype,
            grad_scaler: GradScalerConfig::default(),
        }
    }

    /// `grad_scaler` を明示的に差し替える（ビルダー）。
    pub fn grad_scaler(mut self, config: GradScalerConfig) -> Self {
        self.grad_scaler = config;
        self
    }
}

/// [`Compiled`] が AMP 有効時のみ保持する状態（`dtype` は `compile_with_amp`
/// 呼び出し時点で固定・`scaler` は `fit` 呼び出しをまたいで継続する
/// `GradScaler` 本体）。`GradScaler` は `Debug` を実装しないため
/// [`OptimizerState`] と同様に手書き `Debug` を用意する。
struct AmpState {
    dtype: ScalarDType,
    scaler: GradScaler,
}

impl std::fmt::Debug for AmpState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmpState")
            .field("dtype", &self.dtype)
            .field("scale", &self.scaler.scale())
            .finish()
    }
}

/// `compile()` の `loss` 引数（`Reduction::Mean` 固定。`#[non_exhaustive]`
/// のため後続の損失追加〈#1763 以降〉が既存呼び出し元の非網羅的
/// `match` を破壊しない）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    /// [`crate::Var::mse_loss`]（`Reduction::Mean`）。`target` は
    /// `Tensor<f32>`（`pred` と同 shape）。
    Mse,
    /// [`crate::Var::cross_entropy_loss`]（`class_dim = 1`・
    /// `Reduction::Mean`）。`logits` は `[N, C]`・`target` は
    /// `Tensor<i32>` `[N]`（クラス添字）。
    CrossEntropy,
}

/// `compile()` の `optimizer` 引数（既存 [`crate::optim`] の
/// `*Config` 型を保持する variant のみ。ハイパーパラメータ検証は
/// `compile()` 内で各 `*::new` へ委譲する）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Optimizer {
    Sgd(SgdConfig),
    AdamW(AdamWConfig),
    Adam(AdamConfig),
}

/// `fit()` の構成（Keras `fit(epochs=, batch_size=, shuffle=)` の
/// サブセット）。
///
/// **`shuffle` の既定値（`false`）について**: Keras `fit` の既定
/// `shuffle=True` とは異なり、[`fandhe_ai_tensor_core::data::
/// DataLoaderConfig::new`] と同じ「既定でグローバル RNG を消費しない」
/// 側へ揃えた（`docs/compat-fit-evaluate-design.md` 参照）。Keras
/// 相当にしたい呼び出し元は [`Self::shuffle`]`(true)` を明示する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitConfig {
    epochs: usize,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl FitConfig {
    /// `epochs`（学習を繰り返す回数）・`batch_size`（[`Sequential::fit`] へ
    /// 渡す `x`／`y` を分割する 1 バッチあたりのサンプル数）を指定して
    /// 構築する（Keras `fit(epochs=, batch_size=)` 相当）。
    ///
    /// 既定値: [`Self::shuffle`] は `false`（型ドキュメント冒頭の
    /// 「`shuffle` の既定値」節を参照）・[`Self::drop_last`] も `false`
    /// （末尾の端数バッチも切り捨てずに使う）。
    ///
    /// `epochs == 0`／`batch_size == 0` はここでは検査しない
    /// （両者とも `usize` の有効値であり、この時点では「不正な引数」
    /// ではなく「呼び出し方によっては無意味な設定」であるため）。
    /// 実際の検査は [`Sequential::fit`] 呼び出し時に行う: `epochs == 0` は
    /// `AutodiffError::InvalidArgument` を即座に返し、`batch_size == 0`
    /// は [`fandhe_ai_tensor_core::data::DataLoaderConfig::new`] 経由で
    /// 検査され同様に `InvalidArgument` へマッピングされる。
    pub fn new(epochs: usize, batch_size: usize) -> Self {
        FitConfig {
            epochs,
            batch_size,
            shuffle: false,
            drop_last: false,
        }
    }

    /// `true` で各 epoch 開始前にサンプル順をシャッフルする
    /// （[`fandhe_ai_tensor_core::data::DataLoaderConfig::shuffle`] への
    /// 委譲。グローバル RNG〈[`crate::manual_seed`]〉を消費する）。
    pub fn shuffle(mut self, on: bool) -> Self {
        self.shuffle = on;
        self
    }

    /// `true` で `batch_size` に満たない末尾バッチを切り捨てる
    /// （[`fandhe_ai_tensor_core::data::DataLoaderConfig::drop_last`]
    /// への委譲）。
    pub fn drop_last(mut self, on: bool) -> Self {
        self.drop_last = on;
        self
    }
}

/// `fit()`／`fit_with_callbacks()` の戻り値（Keras `History.history`
/// 相当。`["loss", "val_loss", "lr"]` の 3 キーに対応する 3 フィールド）。
///
/// `loss[i]` は epoch `i`（0-indexed）の学習損失を、各バッチの
/// サンプル数で重み付けした平均（`Σ(batch_loss * batch_n) / Σ batch_n`。
/// `f64` で集計し最後に 1 回 `f32` へ downcast）として記録する。
///
/// `val_loss[i]`（イシュー #1763）は `fit_with_callbacks` の
/// `validation` 引数が `Some` の場合のみ epoch `i` 末の検証損失
/// （[`Sequential::evaluate`] と同じサンプル数重み付き平均集計方式）で
/// 埋まる。`validation = None`（[`Sequential::fit`] を含む）の場合は
/// 空のまま（`Vec::new()`）。
///
/// `lr[i]`（イシュー #1763）は epoch `i` の**全バッチが実際に使った**
/// optimizer の学習率（`callbacks` の有無に関わらず常に記録する。
/// [`super::callbacks::LrSchedule`] が存在しない場合は `compile()` 時に
/// 設定した学習率が epoch を通じて一定のまま記録される）。
///
/// いずれの `Vec` も `i` は `fit_with_callbacks` 呼び出し内のローカル
/// epoch 番号（0-indexed。`super::callbacks` モジュール doc「epoch
/// 番号の数え方」節が定義する callback 側の通算 epoch 番号とは別物）。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct History {
    pub loss: Vec<f32>,
    pub val_loss: Vec<f32>,
    pub lr: Vec<f32>,
}

/// `fit()`／`evaluate()` の target 要素型（sealed。[`crate::CastElement`]
/// と同型の非公開 `private::Sealed` 経由）。`f32`（[`Loss::Mse`]）・
/// `i32`（[`Loss::CrossEntropy`]）のみ実装する——他の型を受け付ける
/// 誤用をコンパイル時に排除する（`.claude/rules/security.md` A03 の
/// 精神を型で担保する）。
pub trait FitTarget: Element + private::Sealed {
    /// `pred`（forward 出力）と `target_batch`（このバッチの正解値）から
    /// `loss` に応じた損失 `Var` を構築する。`loss` と `Self` の組が
    /// 対応しない場合（例: `Loss::Mse` × `i32` target）は
    /// `AutodiffError::InvalidArgument` を返す（fail-closed。loss と
    /// target の dtype 不整合は compile 時点では target 型が未知のため
    /// fit／evaluate 呼び出し時にのみ検出できる）。
    #[doc(hidden)]
    fn loss_for<'t>(
        loss: Loss,
        tape: &'t crate::Tape,
        pred: &crate::Var<'t>,
        target_batch: &Tensor<Self>,
    ) -> Result<crate::Var<'t>, AutodiffError>;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for i32 {}
}

impl FitTarget for f32 {
    fn loss_for<'t>(
        loss: Loss,
        tape: &'t crate::Tape,
        pred: &crate::Var<'t>,
        target_batch: &Tensor<f32>,
    ) -> Result<crate::Var<'t>, AutodiffError> {
        match loss {
            Loss::Mse => {
                let target_var = tape.var_no_grad(target_batch);
                pred.mse_loss_with(&target_var, Reduction::Mean)
            }
            Loss::CrossEntropy => Err(AutodiffError::InvalidArgument(
                "Sequential::fit/evaluate: Loss::CrossEntropy には Tensor<i32> の \
                 target（クラス添字）が必要（Tensor<f32> が渡された）"
                    .to_string(),
            )),
        }
    }
}

impl FitTarget for i32 {
    fn loss_for<'t>(
        loss: Loss,
        _tape: &'t crate::Tape,
        pred: &crate::Var<'t>,
        target_batch: &Tensor<i32>,
    ) -> Result<crate::Var<'t>, AutodiffError> {
        match loss {
            Loss::CrossEntropy => pred.cross_entropy_loss(target_batch, 1, Reduction::Mean),
            Loss::Mse => Err(AutodiffError::InvalidArgument(
                "Sequential::fit/evaluate: Loss::Mse には Tensor<f32> の target が必要\
                 （Tensor<i32> が渡された）"
                    .to_string(),
            )),
        }
    }
}

/// `compile()` で構築した optimizer 本体（[`crate::optim::Sgd`]／
/// [`crate::optim::AdamW`]／[`crate::optim::Adam`] のいずれか）。
/// 3 者は `step` のシグネチャが異なる（`Sgd::step` は位置対応スライス
/// 2 本・`AdamW`／`Adam::step` はタプルスライス 1 本）ため、ここで
/// [`Sequential::trainable_parameters`]／`grad_refs` から共通の
/// `Result<Vec<Tensor<f32>>, AutodiffError>` へ橋渡しする。
enum OptimizerState {
    Sgd(Sgd),
    AdamW(AdamW),
    Adam(Adam),
}

impl std::fmt::Debug for OptimizerState {
    // `AdamW`／`Adam` は `Debug` を実装していない（内部の `m`／`v`
    // モーメントバッファを丸ごと出力する `Debug` 導出をあえて設けて
    // いない設計。`nn::optim::adamw`／`adam` 参照）ため、variant 名の
    // みを出す非網羅的な `Debug` を手書きする（`derive` 不可）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerState::Sgd(sgd) => f.debug_tuple("Sgd").field(sgd).finish(),
            OptimizerState::AdamW(_) => f.debug_tuple("AdamW").finish(),
            OptimizerState::Adam(_) => f.debug_tuple("Adam").finish(),
        }
    }
}

impl OptimizerState {
    fn new(optimizer: Optimizer) -> Result<Self, AutodiffError> {
        match optimizer {
            Optimizer::Sgd(config) => Ok(OptimizerState::Sgd(Sgd::new(config)?)),
            Optimizer::AdamW(config) => Ok(OptimizerState::AdamW(AdamW::new(config)?)),
            Optimizer::Adam(config) => Ok(OptimizerState::Adam(Adam::new(config)?)),
        }
    }

    /// 現在の学習率（LR scheduler 連携用。イシュー #1763）。
    fn lr(&self) -> f32 {
        match self {
            OptimizerState::Sgd(sgd) => sgd.config().lr,
            OptimizerState::AdamW(adamw) => adamw.config().lr,
            OptimizerState::Adam(adam) => adam.config().lr,
        }
    }

    /// 学習率のみを書き換える（各 optimizer の `set_lr` へ委譲。
    /// イシュー #1763。`Sgd`／`AdamW`／`Adam::set_lr` doc の「PyTorch の
    /// `param_group["lr"]` 書き換えと同じ意味論」節を参照——momentum
    /// バッファ／moment 推定値／`step_count` は一切リセットしない）。
    fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError> {
        match self {
            OptimizerState::Sgd(sgd) => sgd.set_lr(new_lr),
            OptimizerState::AdamW(adamw) => adamw.set_lr(new_lr),
            OptimizerState::Adam(adam) => adam.set_lr(new_lr),
        }
    }

    /// `params.len() != grads.len()` を各 `step` 実装（`Sgd::step` は
    /// 検査済み）へ委譲する前に、`AdamW`／`Adam::step` が要求する
    /// `&[(&Tensor, &Tensor)]` への zip 変換自体が短い側で黙って
    /// 切り詰められてしまう（fail-closed 違反）のを防ぐため、ここで
    /// 明示的に事前検査する。
    fn step(
        &mut self,
        params: &[&Tensor<f32>],
        grads: &[&Tensor<f32>],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        if params.len() != grads.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::fit: trainable_parameters().len() ({}) != \
                 trainable_grads().len() ({})",
                params.len(),
                grads.len()
            )));
        }
        match self {
            OptimizerState::Sgd(sgd) => sgd.step(params, grads),
            OptimizerState::AdamW(adamw) => {
                let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
                    params.iter().copied().zip(grads.iter().copied()).collect();
                adamw.step(&pairs)
            }
            OptimizerState::Adam(adam) => {
                let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
                    params.iter().copied().zip(grads.iter().copied()).collect();
                adam.step(&pairs)
            }
        }
    }
}

/// [`Sequential::compiled`] が保持する optimizer／loss の組
/// （非公開・`pub(super)` で `sequential.rs` からのみ到達可能）。
#[derive(Debug)]
pub(super) struct Compiled {
    optimizer: OptimizerState,
    loss: Loss,
    /// [`Sequential::compile_with_amp`] で設定された AMP 状態（既定
    /// `None`。イシュー #1961）。`GradScaler` は fit 呼び出しをまたいで
    /// 継続する状態のため `FitConfig`（`Copy`＋`Eq` 導出済み）ではなく
    /// ここに保持する（`optimizer` と同じ理由）。
    amp: Option<AmpState>,
}

/// 未 compile のモデルへ `fit`／`evaluate` を呼んだ場合の共通エラー。
fn not_compiled(method: &str) -> AutodiffError {
    AutodiffError::InvalidArgument(format!(
        "Sequential::{method}: compile() が呼ばれていない（optimizer／loss が未設定）"
    ))
}

impl Sequential {
    /// optimizer／loss を設定する（Keras `model.compile(optimizer, loss)`
    /// 相当）。ハイパーパラメータ検証（`lr < 0.0` 等）は各
    /// `Sgd`／`AdamW`／`Adam::new` へ委譲する。
    ///
    /// 既に compile 済みのモデルへ再度呼ぶと、それまでの optimizer
    /// 状態（momentum バッファ・m/v 等）を破棄して新しい `optimizer`／
    /// `loss` へ置き換える（Keras の再 `compile()` と同じ挙動）。
    pub fn compile(&mut self, optimizer: Optimizer, loss: Loss) -> Result<(), AutodiffError> {
        self.compiled = Some(Compiled {
            optimizer: OptimizerState::new(optimizer)?,
            loss,
            amp: None,
        });
        Ok(())
    }

    /// [`Self::compile`] の AMP（自動混合精度）版（イシュー #1961・親
    /// #1958）。`optimizer`／`loss` は [`Self::compile`] と同じ意味だが、
    /// 追加で `amp`（[`AmpConfig`]）を渡すことで [`Self::fit`]／
    /// [`Self::fit_with_callbacks`] の 1 step が次の演算列になる
    /// （`crate::optim` モジュール doc「AMP（GradScaler）を使う場合」
    /// 節・`docs/autodiff-low-precision-linear-design.md` §7 参照）:
    ///
    /// 1. `Linear` 層 forward を `amp.compute_dtype`（f32 master weight・
    ///    backward は常に f32）で計算する（`Linear` 以外の層は f32 の
    ///    まま。[`crate::compat::SequentialVars::forward_with_precision`]
    ///    doc 参照）
    /// 2. **scale 前**の素の loss を `History::loss` へ記録する（非有限
    ///    でも overflow を可視化するためそのまま記録する）
    /// 3. `scale_loss → backward → unscale`（非有限検出込み）
    /// 4. `should_skip_step()` が `true` ならこの step の
    ///    `optimizer.step`／`apply_parameters` を**両方**スキップする
    ///    （`optimizer` の `step_count` も進めない）
    /// 5. `false` なら unscale 済み勾配で `optimizer.step` →
    ///    `apply_parameters`
    /// 6. 最後に必ず `scaler.update(found_non_finite)`
    ///
    /// [`Self::evaluate`]・`validation_data`（[`Self::fit_with_callbacks`]）・
    /// [`Self::predict`][crate::compat::Sequential::predict] は f32 の
    /// ままで AMP の対象外（2h 粒度のスコープ外。実装計画 §8）。
    ///
    /// 再度 [`Self::compile`]／[`Self::compile_with_amp`] を呼ぶと
    /// optimizer 状態と同様に AMP 状態（`GradScaler` のスケール値・
    /// backoff 履歴）も破棄される（`Self::compile` doc の再 compile
    /// 契約と同じ）。
    ///
    /// # Errors
    ///
    /// `optimizer`（[`OptimizerState::new`]）または `amp.grad_scaler`
    /// （[`GradScaler::new`]）の検証に失敗した場合 `InvalidArgument`
    /// （fail-closed。いずれかが失敗した場合 `self.compiled` は
    /// 変更しない——[`Self::compile`] は失敗時に前の `compiled` を
    /// そのまま残す契約とは異なる〈`OptimizerState::new` 失敗時は
    /// 元々 `self.compiled` へ代入する前に return する〉ため、本
    /// メソッドも同じく「全構築成功後にのみ代入する」規約で揃える）。
    pub fn compile_with_amp(
        &mut self,
        optimizer: Optimizer,
        loss: Loss,
        amp: AmpConfig,
    ) -> Result<(), AutodiffError> {
        let optimizer_state = OptimizerState::new(optimizer)?;
        let scaler = GradScaler::new(amp.grad_scaler)?;
        self.compiled = Some(Compiled {
            optimizer: optimizer_state,
            loss,
            amp: Some(AmpState {
                dtype: amp.compute_dtype.to_scalar_dtype(),
                scaler,
            }),
        });
        Ok(())
    }

    /// [`Self::compile`] 済みかどうか。
    pub fn is_compiled(&self) -> bool {
        self.compiled.is_some()
    }

    /// 現在の AMP スケール値（[`GradScaler::scale`]）。AMP 未使用
    /// （[`Self::compile`] のみ・[`Self::compile_with_amp`] 未呼び出し）
    /// または未 compile の場合は `None`（イシュー #1961。AC-a を facade
    /// のみで観測するための読み取り専用アクセサ）。
    pub fn amp_loss_scale(&self) -> Option<f32> {
        self.compiled
            .as_ref()?
            .amp
            .as_ref()
            .map(|amp| amp.scaler.scale())
    }

    /// `x`（`[N, ...]`）・`y`（`[N, ...]`）を `config.epochs` 回学習する
    /// （Keras `model.fit(x, y, epochs=, batch_size=)` 相当）。
    ///
    /// [`Self::fit_with_callbacks`]（`validation=None`・`callbacks=&mut
    /// []`）への薄い委譲——`fit_with_callbacks_empty_matches_fit_bit_
    /// exact` が両者の演算列・戻り値の完全一致を検証する。callbacks／
    /// `validation_data`／LR スケジューラ連携（イシュー #1763）が
    /// 必要な場合は [`Self::fit_with_callbacks`] を直接使う。
    ///
    /// 1 バッチあたりの演算列・train／eval モードの扱い・エラー契約は
    /// [`Self::fit_with_callbacks`] doc を参照（本メソッドはそのうち
    /// `validation`／`callbacks` を使わないサブセット）。
    pub fn fit<T: FitTarget>(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        config: FitConfig,
    ) -> Result<History, AutodiffError> {
        self.fit_with_callbacks_named("fit", x, y, config, None, &mut [])
    }

    /// [`Self::fit`] の拡張版（イシュー #1763・親 #1618）:
    /// `validation`（検証データ。`Some((x_val, y_val))`）・
    /// `callbacks`（[`super::callbacks::Callback`] の可変スライス）を
    /// 追加で受け取る。`validation=None`・`callbacks=&mut []` のとき
    /// [`Self::fit`] と完全に同一の演算列・戻り値になる。
    ///
    /// # 1 epoch の処理順序
    ///
    /// 1. **epoch 開始 LR 同期**: `callbacks` 中の
    ///    [`super::callbacks::Callback::LrSchedule`] それぞれについて
    ///    `lr_for_epoch_begin()`（`callbacks` モジュール doc
    ///    「LR 同期のタイミング」節）を読み `compiled.optimizer.set_lr`
    ///    へ書き込む（複数存在する場合は `callbacks` の並び順で最後の
    ///    ものが勝つ）。この時点の optimizer の学習率を
    ///    `history.lr` へ記録する（`LrSchedule` が 1 つもない場合も
    ///    常に記録する。既存学習率が epoch を通じて一定のまま
    ///    記録される）。
    /// 2. [`Self::fit`] と同一のバッチループ（`bind → forward →
    ///    T::loss_for → backward → trainable_grads → optimizer.step →
    ///    apply_parameters`）を回し `history.loss` へ記録する。
    /// 3. `validation` が `Some` なら eval モードへ一時的に切り替えて
    ///    [`Self::evaluate`] と同じ演算列（`run_evaluate`。
    ///    `config.batch_size` を使う）で評価し `history.val_loss` へ
    ///    記録する。
    /// 4. **epoch 末 callbacks**: `callbacks` を**スライス順**で処理
    ///    する（いずれかが学習打ち切りを要求しても、当該 epoch の他の
    ///    callback は必ず処理してから打ち切る）。
    ///    - [`super::callbacks::Callback::ModelCheckpoint`][]: 監視値と
    ///      `self` から `observe`（スナップショット更新）する。
    ///    - [`super::callbacks::Callback::LrSchedule`]: `advance` して
    ///      内部状態（`Plateau` なら `ReduceLrOnPlateau::step`）を
    ///      進める（optimizer への書き戻しは次 epoch 開始時のみ。
    ///      「LR 同期のタイミング」節参照）。
    ///    - [`super::callbacks::Callback::EarlyStopping`]: `observe` し、
    ///      学習打ち切りを要求されたら（同一 epoch の他 callback 処理
    ///      後に）ループを抜ける。
    ///
    /// # fit 呼び出しをまたぐ状態の扱い
    ///
    /// - [`super::callbacks::Callback::EarlyStopping`][]: 本メソッド
    ///   呼び出しのたびに内部状態（`best`／`wait`／`stopped_epoch` 等）
    ///   を**リセット**する（Keras `on_train_begin` と同じ。前回の
    ///   呼び出しで停止済みの `EarlyStopping` を再利用しても即座には
    ///   停止しない）。
    /// - [`super::callbacks::Callback::ModelCheckpoint`]・
    ///   [`super::callbacks::Callback::LrSchedule`]: 内部状態は
    ///   **継続**する（optimizer 状態が `fit` 呼び出しをまたいで継続
    ///   する既存契約——`fit(1)+fit(1) == fit(2)`——と整合する）。
    ///
    /// # `restore_best_weights`
    ///
    /// `callbacks` 中に `restore_best_weights(true)` の
    /// [`super::callbacks::Callback::EarlyStopping`] があり、かつ
    /// best スナップショットが記録されていれば、本メソッドが返る
    /// 直前に [`Self::load_state_dict`] でそこへ復元する（Keras 3 の
    /// `restore_best_weights=True` と同じ意味論）。**この復元は
    /// 学習打ち切り・完走・下記「エラー」節のエラーによる早期終了
    /// （`Err` を返す経路）のいずれでも必ず行われる**——PR #1883
    /// レビュー指摘の是正（イシュー #1763）: `EarlyStopping` が既に
    /// 保持していたベストスナップショットが、epoch 途中のエラーに
    /// よって黙って失われることはない。複数の `EarlyStopping` が
    /// 該当する場合は `callbacks` の並び順で最後のものが勝つ。
    ///
    /// # エラー
    ///
    /// [`Self::fit`] の既存エラー契約に加え:
    /// - `callbacks` のいずれかが `monitor == Monitor::ValLoss`
    ///   （[`super::callbacks::EarlyStopping`]／
    ///   [`super::callbacks::ModelCheckpoint`] の既定・
    ///   [`super::callbacks::LrSchedule::plateau`] の既定）で
    ///   `validation.is_none()` の場合 → `InvalidArgument`
    ///   （train／eval モード変更前に検査するため復元は不要）
    /// - `LrSchedule::advance`／`optimizer.set_lr` が失敗した場合
    ///   （例: ユーザー定義 `LrScheduler` が非有限値を返した）→
    ///   `restore_best_weights` を（該当すれば）適用したうえで、
    ///   その時点のエラーをそのまま返す（[`Self::fit`] と同じ
    ///   fail-closed 契約: compile 済み状態は維持したまま返す）
    pub fn fit_with_callbacks<T: FitTarget>(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        config: FitConfig,
        validation: Option<(&Tensor<f32>, &Tensor<T>)>,
        callbacks: &mut [Callback],
    ) -> Result<History, AutodiffError> {
        self.fit_with_callbacks_named("fit_with_callbacks", x, y, config, validation, callbacks)
    }

    fn fit_with_callbacks_named<T: FitTarget>(
        &mut self,
        method: &'static str,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        config: FitConfig,
        validation: Option<(&Tensor<f32>, &Tensor<T>)>,
        callbacks: &mut [Callback],
    ) -> Result<History, AutodiffError> {
        // (1) 未 compile 検査・compiled の一時取り出し（借用衝突回避。
        // `run_fit` 内で `bind`〈&self 借用〉と `self.compiled`〈&mut
        // self 借用〉を同時に持てないため、compiled を一旦取り外して
        // 別変数として扱う。結果を問わず必ず書き戻す）。
        let mut compiled = self.compiled.take().ok_or_else(|| not_compiled(method))?;

        // (2) 引数検査（モード変更・DataLoader 構築より前。早期 Err は
        // モード変更前のため復元不要——ただし compiled は既に take 済み
        // なのでここで明示的に書き戻す）。
        if config.epochs == 0 {
            self.compiled = Some(compiled);
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: epochs == 0"
            )));
        }
        if validation.is_none()
            && let Some(offending) = callbacks.iter().find(|cb| cb.requires_validation())
        {
            self.compiled = Some(compiled);
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: callback {offending:?} は \
                 Monitor::ValLoss を監視するが validation が None"
            )));
        }

        // (2.5) EarlyStopping はこの fit 呼び出しの開始時に必ずリセット
        // する（本メソッド doc「fit 呼び出しをまたぐ状態の扱い」節）。
        for cb in callbacks.iter_mut() {
            if let Callback::EarlyStopping(es) = cb {
                es.reset_for_fit();
            }
        }

        // (3) train モードへ切り替え（Dropout 入りモデルのマスク適用の
        // ため）。復元は成功・失敗いずれの経路でも必ず行う。
        let prev_training = self.training();
        self.set_training(true);

        let result = self.run_fit(method, &mut compiled, x, y, config, validation, callbacks);

        // (4) モード復元・compiled の書き戻し（結果を問わず必ず行う。
        // fail-closed: 失敗した fit の後もモデルを「未 compile」状態へ
        // 落とさない）。
        self.set_training(prev_training);
        self.compiled = Some(compiled);
        result
    }

    /// [`Self::fit_with_callbacks`] の本体（compiled を取り出し済みの
    /// 状態で呼ばれる。`&mut self` と `compiled: &mut Compiled` を
    /// 独立した借用として受け取ることで、`self.bind(&tape)`〈`&self`〉
    /// と `compiled.optimizer.step`〈`&mut compiled`〉を同時に生かせる）。
    /// `method`（呼び出し元の公開メソッド名。イシュー #1763 PR #1883
    /// レビュー指摘の是正）を追加したことで引数が 8 個になった
    /// （既存の 7 引数構成を維持したまま追加したため。呼び出し元は
    /// [`Self::fit_with_callbacks_named`] の 1 箇所のみで、これ以上
    /// 引数が増える見込みも薄いため構造体化はせず許容する）。
    #[allow(clippy::too_many_arguments)]
    fn run_fit<T: FitTarget>(
        &mut self,
        method: &'static str,
        compiled: &mut Compiled,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        config: FitConfig,
        validation: Option<(&Tensor<f32>, &Tensor<T>)>,
        callbacks: &mut [Callback],
    ) -> Result<History, AutodiffError> {
        let to_invalid_arg = |e: fandhe_ai_tensor_core::data::DataError| {
            AutodiffError::InvalidArgument(format!("Sequential::{method}: {e}"))
        };
        let x_dataset = TensorDataset::new(x.clone()).map_err(to_invalid_arg)?;
        let y_dataset = TensorDataset::new(y.clone()).map_err(to_invalid_arg)?;
        let loader = DataLoader::new(
            (x_dataset, y_dataset),
            DataLoaderConfig::new(config.batch_size)
                .shuffle(config.shuffle)
                .drop_last(config.drop_last),
        )
        .map_err(to_invalid_arg)?;

        // `Vec::with_capacity` は capacity overflow（`config.epochs`
        // が巨大・`usize::MAX` 近辺等）で panic する（本番経路の panic
        // 禁止。`.claude/rules/security.md` A03 の精神）。`try_reserve_exact`
        // で失敗可能にし、確保失敗は `InvalidArgument` へマッピングして
        // 呼び出し元（[`Self::fit_with_callbacks`]）の既存復元経路
        // （train／eval モード巻き戻し・`compiled` 復元）へ返す。
        let mut loss = Vec::new();
        loss.try_reserve_exact(config.epochs).map_err(|e| {
            AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: History.loss 用の確保に失敗した \
                 (epochs={}): {e}",
                config.epochs
            ))
        })?;
        // `val_loss` は `validation.is_some()` のときのみ epochs 分
        // 確保する（`None` の場合は空のまま。`History::val_loss` doc
        // 参照）。
        let mut val_loss = Vec::new();
        if validation.is_some() {
            val_loss.try_reserve_exact(config.epochs).map_err(|e| {
                AutodiffError::InvalidArgument(format!(
                    "Sequential::{method}: History.val_loss 用の確保に失敗した \
                     (epochs={}): {e}",
                    config.epochs
                ))
            })?;
        }
        let mut lr = Vec::new();
        lr.try_reserve_exact(config.epochs).map_err(|e| {
            AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: History.lr 用の確保に失敗した \
                 (epochs={}): {e}",
                config.epochs
            ))
        })?;
        let mut history = History { loss, val_loss, lr };

        // 本体は `'epochs_block` ラベル付きブロックへ包み、`?` による
        // 早期 return を `break 'epochs_block Err(..)` に置き換える
        // （イシュー #1763 PR #1883 レビュー指摘: 従来はここで直接
        // `return Err(..)`／`?` していたため、epoch 途中のエラー
        // （`LrSchedule::advance` の失敗・バリデーション失敗・
        // `count == 0` 等）が下記「fit 終了時」の `restore_best_weights`
        // を素通りしてしまい、`EarlyStopping` が既に保持していたベスト
        // 重みスナップショットが失われたまま次回 `fit_with_callbacks`
        // 呼び出しで `reset_for_fit` により消えていた）。この block 式
        // により、正常完走・`break 'epochs`（patience 打ち切り）・
        // エラーのいずれの経路でも必ずブロック直後の
        // `restore_best_weights` 処理へ到達する。
        let epoch_result: Result<(), AutodiffError> = 'epochs_block: {
            'epochs: for epoch_local in 0..config.epochs {
                // (1) epoch 開始 LR 同期（[`Self::fit_with_callbacks`] doc
                // 「1 epoch の処理順序」節）。`LrSchedule` が 1 つもなくても
                // `history.lr` は常に記録する。
                for cb in callbacks.iter() {
                    if let Callback::LrSchedule(ls) = cb
                        && let Err(e) = compiled.optimizer.set_lr(ls.lr_for_epoch_begin())
                    {
                        break 'epochs_block Err(e);
                    }
                }
                history.lr.push(compiled.optimizer.lr());

                // (2) バッチループ（[`Self::fit`] と同一の演算列）。
                let mut weighted_sum = 0.0f64;
                let mut count = 0usize;

                for batch in &loader {
                    let (x_batch, y_batch) = match batch {
                        Ok(v) => v,
                        Err(e) => {
                            break 'epochs_block Err(AutodiffError::InvalidArgument(format!(
                                "Sequential::{method}: {e}"
                            )));
                        }
                    };
                    let n_batch = x_batch.shape().first().copied().unwrap_or(0);

                    // `updated == None` は AMP 有効時に非有限勾配で
                    // この step をスキップしたことを表す（イシュー
                    // #1961 実装計画 §2.3 手順 4。`optimizer.step`／
                    // `apply_parameters` を両方スキップし、`optimizer`
                    // の `step_count` も進めない）。AMP 無効
                    // （`compiled.amp.is_none()`）のときは常に `Some`
                    // であり、下記のブロック全体・エラー経路は
                    // AMP 導入前の実装と完全に同一（bit 同一契約）。
                    let updated: Option<Vec<Tensor<f32>>> = {
                        let tape = crate::tape();
                        let bound = self.bind(&tape);
                        let x_var = tape.var(&x_batch);

                        let low_precision_dtype = compiled.amp.as_ref().map(|amp| amp.dtype);
                        let pred = match bound.forward_with_precision(
                            &tape,
                            &x_var,
                            low_precision_dtype,
                        ) {
                            Ok(v) => v,
                            Err(e) => break 'epochs_block Err(e),
                        };
                        let loss_var = match T::loss_for(compiled.loss, &tape, &pred, &y_batch) {
                            Ok(v) => v,
                            Err(e) => break 'epochs_block Err(e),
                        };
                        // 適用順序契約（手順 2）: **scale 前**の素の loss
                        // を常に記録する（非有限でも overflow をそのまま
                        // 可視化する。AMP 無効時は scale が存在しない
                        // ためこの値がそのまま記録対象）。
                        let loss_scalar = match loss_var.to_tensor().get(&[]) {
                            Some(v) => v,
                            None => {
                                break 'epochs_block Err(AutodiffError::InvalidArgument(format!(
                                    "Sequential::{method}: loss の shape が [] ではない\
                                         （loss 演算の契約違反）"
                                )));
                            }
                        };
                        weighted_sum += loss_scalar as f64 * n_batch as f64;
                        count += n_batch;

                        if let Some(amp) = compiled.amp.as_mut() {
                            // 適用順序契約（手順 3）: scale_loss → backward → unscale。
                            let scaled_loss = match amp.scaler.scale_loss(&loss_var) {
                                Ok(v) => v,
                                Err(e) => break 'epochs_block Err(e),
                            };
                            let grads = match tape.backward(&scaled_loss) {
                                Ok(v) => v,
                                Err(e) => break 'epochs_block Err(e),
                            };
                            let grad_refs = match bound.trainable_grads(&grads) {
                                Ok(v) => v,
                                Err(e) => break 'epochs_block Err(e),
                            };
                            let unscale_result = match amp.scaler.unscale(&grad_refs) {
                                Ok(v) => v,
                                Err(e) => break 'epochs_block Err(e),
                            };
                            if unscale_result.should_skip_step() {
                                // 適用順序契約（手順 4）: skip でも
                                // `scaler.update` は必ず呼ぶ。
                                if let Err(e) = amp.scaler.update(true) {
                                    break 'epochs_block Err(e);
                                }
                                None
                            } else {
                                // 適用順序契約（手順 5）: unscale 済み
                                // 勾配で optimizer.step。
                                let unscaled_refs: Vec<&Tensor<f32>> =
                                    unscale_result.grads.iter().collect();
                                let param_refs = self.trainable_parameters();
                                let stepped =
                                    match compiled.optimizer.step(&param_refs, &unscaled_refs) {
                                        Ok(v) => v,
                                        Err(e) => break 'epochs_block Err(e),
                                    };
                                // 適用順序契約（手順 6）: 非 skip step も
                                // `scaler.update` を必ず呼ぶ（`amp` は
                                // `compiled.amp.as_mut()` から借用済みの
                                // まま・再取得しない）。
                                if let Err(e) = amp.scaler.update(false) {
                                    break 'epochs_block Err(e);
                                }
                                Some(stepped)
                            }
                        } else {
                            let grads = match tape.backward(&loss_var) {
                                Ok(v) => v,
                                Err(e) => break 'epochs_block Err(e),
                            };
                            let grad_refs = match bound.trainable_grads(&grads) {
                                Ok(v) => v,
                                Err(e) => break 'epochs_block Err(e),
                            };
                            let param_refs = self.trainable_parameters();
                            match compiled.optimizer.step(&param_refs, &grad_refs) {
                                Ok(v) => Some(v),
                                Err(e) => break 'epochs_block Err(e),
                            }
                        }
                    };
                    if let Some(updated) = updated
                        && let Err(e) = self.apply_parameters(updated)
                    {
                        break 'epochs_block Err(e);
                    }
                }

                if count == 0 {
                    // `drop_last = true` で全バッチが落ちた場合（0 除算を
                    // 黙って NaN にしない。fail-closed）。
                    break 'epochs_block Err(AutodiffError::InvalidArgument(format!(
                        "Sequential::{method}: この epoch で処理されたサンプルが 0 件\
                         （drop_last により全バッチが切り捨てられた可能性がある）"
                    )));
                }
                history.loss.push((weighted_sum / count as f64) as f32);

                // (3) validation（[`Self::fit_with_callbacks`] doc「1 epoch
                // の処理順序」節）。
                if let Some((x_val, y_val)) = validation {
                    self.set_training(false);
                    let v = self.run_evaluate::<T>(
                        x_val,
                        y_val,
                        config.batch_size,
                        compiled.loss,
                        method,
                    );
                    self.set_training(true);
                    match v {
                        Ok(v) => history.val_loss.push(v),
                        Err(e) => break 'epochs_block Err(e),
                    }
                }

                // (4) epoch 末 callbacks（スライス順。いずれかが学習打ち切り
                // を要求しても、当該 epoch の他 callback はすべて処理して
                // から打ち切る）。
                let mut stop = false;
                for cb in callbacks.iter_mut() {
                    match cb {
                        Callback::ModelCheckpoint(mc) => {
                            if let Some(value) = mc.monitor_value_at(&history, epoch_local) {
                                mc.observe(value, self);
                            }
                        }
                        Callback::LrSchedule(ls) => {
                            let value = ls.monitor_value_at(&history, epoch_local);
                            if let Err(e) = ls.advance(value) {
                                break 'epochs_block Err(e);
                            }
                        }
                        Callback::EarlyStopping(es) => {
                            if let Some(value) = es.monitor_value_at(&history, epoch_local)
                                && es.observe(value, epoch_local, || self.state_dict())
                            {
                                stop = true;
                            }
                        }
                    }
                }
                if stop {
                    break 'epochs;
                }
            }
            Ok(())
        };

        // fit 終了時: `restore_best_weights`（[`Self::fit_with_callbacks`]
        // doc「`restore_best_weights`」節）。学習打ち切り・完走・
        // エラーによる早期終了（`epoch_result` が `Err`）いずれの
        // 経路でもここへ到達する（イシュー #1763 PR #1883 レビュー
        // 指摘の是正）。エラー経路でも `EarlyStopping` が既に保持して
        // いたベストスナップショットをベストエフォートで適用してから
        // 元のエラーを返す——復元自体が失敗した場合（`load_state_dict`
        // が架構不一致等で `Err` を返す。通常は同一モデル自身の
        // スナップショットのため発生しない）のみ、その復元エラーで
        // `epoch_result` を上書きする（呼び出し元へ「復元にも失敗した」
        // ことを fail-closed に伝える）。
        let mut restore_err: Option<AutodiffError> = None;
        for cb in callbacks.iter_mut() {
            if let Callback::EarlyStopping(es) = cb
                && let Some(state) = es.take_restore_state()
                && let Err(e) = self.load_state_dict(state)
            {
                restore_err = Some(e);
            }
        }

        epoch_result?;
        if let Some(e) = restore_err {
            return Err(e);
        }

        Ok(history)
    }
    /// `x`／`y` に対する平均損失を評価する（Keras
    /// `model.evaluate(x, y, batch_size=)` 相当。学習は行わない）。
    ///
    /// `batch_size == x.shape()[0]`（全件 1 バッチ）のとき、
    /// [`Sequential::forward`] → `T::loss_for` の直接計算と bit 完全
    /// 一致する。`batch_size` を分割した場合はサンプル数重み付き平均
    /// （[`Self::fit`] の `History::loss` と同じ集計方式）。
    ///
    /// # train／eval モード
    ///
    /// [`Self::fit`] と同様、呼び出し前のモードを記憶し評価中は eval
    /// モード（[`Self::eval`]）へ切り替える（Dropout 入りモデルでも
    /// 決定的な評価を得るため）。成功・失敗いずれの場合も元のモードへ
    /// 復元する。
    ///
    /// # エラー
    /// [`Self::fit`] と同じ検査（未 compile・サンプル数不一致・
    /// `batch_size == 0`・`loss` と `T` の不整合）に従う。
    pub fn evaluate<T: FitTarget>(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        batch_size: usize,
    ) -> Result<f32, AutodiffError> {
        let loss = self
            .compiled
            .as_ref()
            .ok_or_else(|| not_compiled("evaluate"))?
            .loss;

        let prev_training = self.training();
        self.set_training(false);
        let result = self.run_evaluate::<T>(x, y, batch_size, loss, "evaluate");
        self.set_training(prev_training);
        result
    }

    /// `method` には呼び出し元の公開メソッド名（`"evaluate"`、または
    /// [`Self::fit_with_callbacks_named`] の validation フェーズ経由
    /// なら `"fit"`／`"fit_with_callbacks"`）を渡し、エラーメッセージが
    /// 実際に呼ばれた公開メソッド名を名乗るようにする（イシュー #1763
    /// PR #1883 レビュー指摘の是正）。
    fn run_evaluate<T: FitTarget>(
        &self,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        batch_size: usize,
        loss: Loss,
        method: &str,
    ) -> Result<f32, AutodiffError> {
        let to_invalid_arg = |e: fandhe_ai_tensor_core::data::DataError| {
            AutodiffError::InvalidArgument(format!("Sequential::{method}: {e}"))
        };
        let x_dataset = TensorDataset::new(x.clone()).map_err(to_invalid_arg)?;
        let y_dataset = TensorDataset::new(y.clone()).map_err(to_invalid_arg)?;
        let loader = DataLoader::new((x_dataset, y_dataset), DataLoaderConfig::new(batch_size))
            .map_err(to_invalid_arg)?;

        let mut weighted_sum = 0.0f64;
        let mut count = 0usize;

        for batch in &loader {
            let (x_batch, y_batch) = batch.map_err(|e| {
                AutodiffError::InvalidArgument(format!("Sequential::{method}: {e}"))
            })?;
            let n_batch = x_batch.shape().first().copied().unwrap_or(0);

            let tape = crate::tape();
            let x_var = tape.var(&x_batch);
            let pred = self.forward(&tape, &x_var)?;
            let loss_var = T::loss_for(loss, &tape, &pred, &y_batch)?;
            let loss_scalar = loss_var.to_tensor().get(&[]).ok_or_else(|| {
                AutodiffError::InvalidArgument(format!(
                    "Sequential::{method}: loss の shape が [] ではない\
                     （loss 演算の契約違反）"
                ))
            })?;
            weighted_sum += loss_scalar as f64 * n_batch as f64;
            count += n_batch;
        }

        if count == 0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: 処理されたサンプルが 0 件"
            )));
        }
        Ok((weighted_sum / count as f64) as f32)
    }
}
