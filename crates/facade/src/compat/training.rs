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
//! &mut [])` への委譲）。分類 metrics（accuracy・precision・recall・
//! F1・confusion matrix）は [`Sequential::fit_with_metrics`]（イシュー
//! #2072・親 #2059・`super::metrics`）で実装済み。`DataLoader` を直接
//! 受ける `fit` 入口は対象外のまま（`docs/compat-callbacks-design.md`
//! §8 参照）。
//!
//! **カスタム学習 step フック（イシュー #2184・親 #2131）について**:
//! [`Sequential::run_fit`] の既定バッチ処理を丸ごと差し替える内部
//! フック（`CustomStepHook`）を配線済みだが、facade 公開面（Keras
//! `Model.train_step()` 相当）は承認待ちのため未公開のまま
//! （`docs/compat-train-step-hook-decision.md`。
//! `crate::TrainStepHoldDoctestGuard` が機械的に固定する）。既存 3 入口
//! （[`Sequential::fit`]／[`Sequential::fit_with_callbacks`]／
//! [`Sequential::fit_with_metrics`]）はいずれもフック `None` で
//! [`Sequential::run_fit`] へ委譲するため挙動は変わらない。

use crate::optim::{
    Adagrad, AdagradConfig, Adam, AdamConfig, AdamW, AdamWConfig, GradScaler, GradScalerConfig,
    Lamb, LambConfig, LbfgsConfig, RmsProp, RmsPropConfig, Sgd, SgdConfig,
};
use crate::{AutodiffError, Tensor};
use fandhe_ai_autodiff::Reduction;
// 非 `pub` の use のみ（`loss_ops` モジュール自体は facade へ公開しない。
// `LossOpsHoldDoctestGuard` の保留を維持。#2509）。
use fandhe_ai_autodiff::loss_ops::l1_loss;
// `Lbfgs` は #2502 で `crate::optim` から公開済み（同一型）。本ファイルの
// `OptimizerState::Lbfgs` は内部クレートの型を直接 import して保持する
// （非 `pub use`。再エクスポートは `optim.rs` が担う）。
use fandhe_ai_autodiff::nn::optim::Lbfgs;
// イシュー #2372: `save_model`／`load_model` が optimizer 内部状態と GradScaler の
// 状態を往復させるための内部専用 import（`pub use` にしない。facade 公開面へ
// 出ないことは `tests/api_surface.rs` の `grad_scaler_from_state`／
// `OptimizerStateDict` ガードが機械的に固定する）。
use fandhe_ai_autodiff::nn::optim::OptimizerStateDict;
use fandhe_ai_autodiff::nn::optim::amp::grad_scaler_from_state;
use fandhe_ai_tensor_core::Element;
use fandhe_ai_tensor_core::ScalarDType;
use fandhe_ai_tensor_core::data::{DataLoader, DataLoaderConfig, TensorDataset};
use std::collections::HashMap;

use super::callbacks::Callback;
use super::metrics::{ConfusionAccumulator, Metrics, MetricsResult};

use super::sequential::Sequential;

/// `compile_with_amp()` の低精度 forward dtype 指定（イシュー #1961・
/// 親 #1958。`docs/autodiff-low-precision-linear-design.md` §7「facade
/// 統合（#1961）」）。
///
/// **facade ローカルに閉じる理由**: `fandhe_ai_tensor_core::ScalarDType`
/// 自体は #1939（`docs/compat-api-scope.md` §5 経路 2 承認）で
/// `crate::ScalarDType` として facade 再エクスポート済みだが、
/// `compile_with_amp` が受理する dtype は F16／Bf16 の 2 種のみに限る
/// （`ScalarDType` は `#[non_exhaustive]` で他 variant にも将来拡張
/// されうる）ため、本 enum は受理集合をこの 2 種へ型で狭める facade
/// 専用の薄い写像として維持する（`to_scalar_dtype` が内部でのみ
/// 変換する）。`AmpDType` 自体を `ScalarDType` へ置換する統合は本
/// 変更のスコープ外（イシュー #1939 実装記録参照）。`#[non_exhaustive]`
/// は `ScalarDType` 自体のバリアント追加余地に追従するため。
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
    /// `dtype` の元になった facade 公開型。`save_model` が manifest の
    /// `amp.dtype`（`f16`／`bf16` の allowlist）へ書くために保持する
    /// （`ScalarDType` は `#[non_exhaustive]` で逆写像が全域でないため）。
    amp_dtype: AmpDType,
    /// `GradScaler` は config の getter を持たない（公開面を広げないため
    /// autodiff へ足さない）ので、構築時の [`GradScalerConfig`] をここへ記録する
    /// （`save_model` の manifest `amp.grad_scaler_config`。イシュー #2372）。
    grad_scaler_config: GradScalerConfig,
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

/// `compile()` の `loss` 引数（`Reduction::Mean` 固定の 9 種。`#[non_exhaustive]`
/// のため後続の損失追加が既存呼び出し元の非網羅的 `match` を破壊しない）。
///
/// target の dtype は `Mse`・`L1`・`Bce`・`BceWithLogits`・`KlDiv`・`Huber`・
/// `SmoothL1` が `Tensor<f32>`、`CrossEntropy`・`Nll` が `Tensor<i32>`
/// （クラス添字）。不整合は fit／evaluate 時に `InvalidArgument` で拒否する。
/// パラメータ付き損失は固定値（`Huber` の delta・`SmoothL1` の beta は 1.0、
/// `KlDiv` は `log_target = false`）で、可変化は本 enum の範囲外。
///
/// 決定記録は `docs/facade-compile-loss-variants-decision.md`（イシュー #2509）。
///
/// # 例
///
/// ```
/// use fandhe_ai::compat::{Loss, Optimizer, Sequential};
/// use fandhe_ai::optim::SgdConfig;
///
/// let model = Sequential::new()
///     .add_linear(2, 1, 0)
///     .unwrap()
///     .add_sigmoid()
///     .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Bce)
///     .unwrap();
/// let _ = model;
/// ```
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
    /// 平均絶対誤差（内部クレートの `loss_ops::l1_loss`・`Reduction::Mean`）。
    /// `target` は `Tensor<f32>`（`pred` と同 shape）。
    L1,
    /// [`crate::Var::bce_loss`]（`Reduction::Mean`）。`pred` は `[0, 1]` の
    /// 確率・`target` は `Tensor<f32>`（同 shape）。範囲外は型付きエラー。
    Bce,
    /// [`crate::Var::bce_with_logits_loss`]（`Reduction::Mean`）。`pred` は
    /// logits・`target` は `Tensor<f32>`（同 shape）。
    BceWithLogits,
    /// [`crate::Var::nll_loss`]（`class_dim = 1`・`Reduction::Mean`）。`pred` は
    /// `[N, C]` の log 確率・`target` は `Tensor<i32>` `[N]`（クラス添字）。
    Nll,
    /// [`crate::Var::kl_div_loss`]（`log_target = false`・`Reduction::Mean`）。
    /// `pred` は log 確率・`target` は確率の `Tensor<f32>`（同 shape）。
    KlDiv,
    /// [`crate::Var::huber_loss`]（`delta = 1.0`・`Reduction::Mean`）。
    /// `target` は `Tensor<f32>`（同 shape）。
    Huber,
    /// [`crate::Var::smooth_l1_loss`]（`beta = 1.0`・`Reduction::Mean`）。
    /// `target` は `Tensor<f32>`（同 shape）。
    SmoothL1,
}

/// `compile()` の `optimizer` 引数（既存 [`crate::optim`] の
/// `*Config` 型を保持する variant のみ。ハイパーパラメータ検証は
/// `compile()` 内で各 `*::new` へ委譲する）。
///
/// **`RmsProp`／`Adagrad`／`Lamb`（イシュー #2170・親 #2131）の LR
/// スケジューラ非対応について**: [`crate::optim::RmsProp`]／
/// [`crate::optim::Adagrad`]／[`crate::optim::Lamb`] は（`Sgd`／
/// `AdamW`／`Adam` と異なり）`set_lr` を持たない値型のため、これら 3
/// variant を compile した状態で [`super::callbacks::Callback::LrSchedule`]
/// を含む `callbacks` を渡すと [`Sequential::fit_with_callbacks`] は
/// `InvalidArgument` を返す（`OptimizerState::set_lr` doc 参照）。
/// `set_lr` の追加は facade 公開面の拡張のためユーザー承認事項であり
/// 本イシューのスコープ外（`docs/compat-api-scope.md` §1.3 参照）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Optimizer {
    Sgd(SgdConfig),
    AdamW(AdamWConfig),
    Adam(AdamConfig),
    /// RMSprop（イシュー #1743・親 #1610。`docs/compat-api-scope.md`
    /// §1.3）。LR スケジューラ非対応（本 enum doc 参照）。
    RmsProp(RmsPropConfig),
    /// Adagrad（イシュー #1743・親 #1610）。LR スケジューラ非対応
    /// （本 enum doc 参照）。
    Adagrad(AdagradConfig),
    /// LAMB（layer-wise adaptive、イシュー #1744・親 #1610）。LR
    /// スケジューラ非対応（本 enum doc 参照）。
    Lamb(LambConfig),
    /// L-BFGS（closure・strong Wolfe line search。イシュー #2197・親
    /// #2172。2026-09-27 所有者承認〈#2172 コメント〉で本 variant・
    /// `LbfgsConfig` の facade 再エクスポート・`compile()`/`fit()` 統合を
    /// 実装した）。
    ///
    /// 他 6 者と異なり、`fandhe_ai_autodiff::nn::optim::Lbfgs` は
    /// 「パラメータ列 → `(損失, 勾配列)`」を返す closure を内部で複数回
    /// 評価する形（1 step あたり勾配評価 1 回を前提とする既存 optimizer
    /// とは API 形状が異なる。`lbfgs.rs` モジュール doc 参照）。
    /// `Sequential::run_fit`（内部専用。facade 公開 API ではない）は
    /// この variant のみ既定バッチ処理（`forward → backward →
    /// optimizer.step`）を迂回し、専用ヘルパー `lbfgs_batch_step` へ
    /// 分岐する（本ファイル「L-BFGS（closure 駆動 optimizer）」節参照）。
    ///
    /// **非対応の組み合わせ（いずれも fail-closed に `InvalidArgument`）**:
    /// [`Sequential::compile_with_amp`]（AMP。損失スケーリングが closure
    /// 複数回評価と両立しないため）・`FitConfig` の `accumulate_steps >
    /// 1`（勾配累積のウィンドウ処理が closure ベースの outer step と
    /// 意味的に合わないため）・カスタム学習 step フック（フックは
    /// `&Sequential`〈不変参照〉しか受け取らないため、trial パラメータを
    /// 書き込む closure を内部で駆動できない）。`Callback::LrSchedule`
    /// は [`Lbfgs::set_lr`] へ委譲できるため対応する。`OptimizerStateDict`
    /// （#2304）・param groups（#2173）は facade 側がいずれも別途保留中
    /// のため、本 variant 固有の追加対応は不要（保留解除時に横断対応
    /// する）。
    ///
    /// **line search**: `Lbfgs`／`LbfgsLineSearch` は #2502 で facade
    /// （`crate::optim`）から公開済みのため、`LbfgsConfig::line_search` に
    /// `LbfgsLineSearch::StrongWolfe` を指定できる（既定は固定ステップの
    /// `LbfgsLineSearch::None`。`crate::optim` モジュール doc「L-BFGS」節）。
    Lbfgs(LbfgsConfig),
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
    /// 勾配累積のウィンドウ幅（イシュー #2180・親 #2131）。既定 `1`
    /// （既存挙動と bit 完全一致。R3）。`1` より大きい場合、
    /// [`Sequential::run_fit`] は `accumulate_steps` 回の backward ごとに
    /// 1 回だけ `optimizer.step` → `apply_parameters` を行う（PyTorch の
    /// `.grad +=` に相当する f32 逐次和で累積。正規化〈N で割る処理〉は
    /// 行わない——大バッチ相当にしたい呼び出し元は学習率を `1/N` にする
    /// 必要がある）。
    ///
    /// [`Self::accumulate_steps`] で設定する（イシュー #2508 で公開。
    /// フィールド自体は非公開のまま。`Copy + Eq` も維持）。
    accumulate_steps: u32,
}

impl FitConfig {
    /// `epochs`（学習を繰り返す回数）・`batch_size`（[`Sequential::fit`] へ
    /// 渡す `x`／`y` を分割する 1 バッチあたりのサンプル数）を指定して
    /// 構築する（Keras `fit(epochs=, batch_size=)` 相当）。
    ///
    /// 既定値: [`Self::shuffle`] は `false`（型ドキュメント冒頭の
    /// 「`shuffle` の既定値」節を参照）・[`Self::drop_last`] も `false`
    /// （末尾の端数バッチも切り捨てずに使う）・[`Self::accumulate_steps`] は
    /// `1`（勾配累積なし）。
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
            accumulate_steps: 1,
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

    /// 勾配累積のウィンドウ幅を `n` にする（PyTorch の gradient
    /// accumulation〈`.grad +=`〉相当。イシュー #2508 で公開。#2180 の
    /// 実装を `docs/compat-grad-accumulation-decision.md` §5 の承認形で
    /// 公開した追加 API）。
    ///
    /// `n` 回の backward ごとに 1 回だけ `optimizer.step` を呼ぶ。epoch 末の
    /// 端数ウィンドウ（`n` に満たないマイクロステップ）も flush する。
    /// 正規化〈`N` で割る処理〉は行わないため、大バッチ相当にしたい場合は
    /// 呼び出し元が学習率を `1/N` にする（等価になるのは SGD〈momentum
    /// なし〉で各ウィンドウのサイズがそろう場合に限る。決定記録 §3）。
    ///
    /// 既定は `1` で、既存の `fit` と bit 完全一致する。`n == 0` はここでは
    /// 検査せず（[`Self::new`] の `epochs == 0` と同じ方針）、
    /// [`Sequential::fit`] 呼び出し時に `AutodiffError::InvalidArgument` を
    /// 返す。`n > 1` は AMP（`compile_with_amp`）・`Optimizer::Lbfgs`・
    /// カスタム step フックとの併用を fail-closed で
    /// `InvalidArgument` として拒否する。
    ///
    /// ```
    /// use fandhe_ai::compat::FitConfig;
    ///
    /// let cfg = FitConfig::new(10, 32).accumulate_steps(4);
    /// assert_ne!(cfg, FitConfig::new(10, 32));
    /// assert_eq!(FitConfig::new(10, 32).accumulate_steps(1), FitConfig::new(10, 32));
    /// ```
    pub fn accumulate_steps(mut self, n: u32) -> Self {
        self.accumulate_steps = n;
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
/// `val_metrics[i]`（イシュー #2072）は [`Sequential::fit_with_metrics`]
/// の `metrics` 引数が非空かつ `validation` が `Some` の場合のみ epoch
/// `i` 末の検証 metrics（[`MetricsResult`]。`val_loss` と同じ
/// validation バッチ列から算出）で埋まる。`metrics` が空、または
/// `validation = None`（[`Sequential::fit`]／`fit_with_callbacks` を
/// 含む）の場合は空のまま（`Vec::new()`）。
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
    pub val_metrics: Vec<MetricsResult>,
}

/// `fit()`／`evaluate()` の target 要素型（sealed。[`crate::CastElement`]
/// と同型の非公開 `private::Sealed` 経由）。`f32`（[`Loss::Mse`] 等の f32 系 7 種）・
/// `i32`（[`Loss::CrossEntropy`]・[`Loss::Nll`]）のみ実装する——他の型を受け付ける
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

    /// `fit_with_metrics`（イシュー #2072）の metrics は分類（クラス
    /// 添字 target）でのみ定義される指標のため、`T = f32`
    /// （[`Loss::Mse`]）での呼び出しを演算列に入る前に拒否する型
    /// ゲート。`i32`（[`Loss::CrossEntropy`]）は既定（`Ok(())`）のまま
    /// 許可、`f32` のみ上書きして拒否する（sealed のため実装追加は
    /// 非破壊）。戻り値自体に意味はなく成否のみを表す。
    #[doc(hidden)]
    fn require_metrics_support() -> Result<(), AutodiffError> {
        Ok(())
    }

    /// `target_batch` をクラス添字 `Tensor<i32>` として参照できるなら
    /// `Some` を返す（`i32` 実装のみ上書き。`Self = i32` のときは
    /// `Tensor<Self> = Tensor<i32>` のため型変換なしでそのまま返せる）。
    /// [`Self::require_metrics_support`] が `Ok` を返す呼び出し経路
    /// （`run_evaluate_with_metrics`）でのみ使い、`None` は「呼び出し元
    /// の事前検査が抜けている」ことを示す防御的分岐として扱う
    /// （`.claude/rules/coding-rust.md` 本番経路 panic 禁止のため
    /// `unwrap`／`expect` ではなく `Option` のまま返し、呼び出し元が
    /// `InvalidArgument` へ写像する）。
    #[doc(hidden)]
    fn as_class_targets(target_batch: &Tensor<Self>) -> Option<&Tensor<i32>> {
        let _ = target_batch;
        None
    }
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
        let t = || tape.var_no_grad(target_batch);
        match loss {
            Loss::Mse => pred.mse_loss_with(&t(), Reduction::Mean),
            Loss::L1 => l1_loss(pred, &t(), Reduction::Mean),
            Loss::Bce => pred.bce_loss(&t(), Reduction::Mean),
            Loss::BceWithLogits => pred.bce_with_logits_loss(&t(), Reduction::Mean),
            Loss::KlDiv => pred.kl_div_loss(&t(), Reduction::Mean),
            Loss::Huber => pred.huber_loss(&t(), 1.0, Reduction::Mean),
            Loss::SmoothL1 => pred.smooth_l1_loss(&t(), 1.0, Reduction::Mean),
            Loss::CrossEntropy | Loss::Nll => Err(AutodiffError::InvalidArgument(format!(
                "Sequential::fit/evaluate: Loss::{loss:?} には Tensor<i32> の \
                 target（クラス添字）が必要（Tensor<f32> が渡された）"
            ))),
        }
    }

    fn require_metrics_support() -> Result<(), AutodiffError> {
        Err(AutodiffError::InvalidArgument(
            "Sequential::fit_with_metrics: metrics はクラス添字 target（Loss::CrossEntropy／\
             Loss::Nll・Tensor<i32>）でのみ計算できる（Tensor<f32> target が渡された）"
                .to_string(),
        ))
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
            // Nll の pred は log 確率。その argmax は logits の argmax と一致するため
            // metrics（分類指標）もそのまま有効（`as_class_targets` が Some を返す）。
            Loss::Nll => pred.nll_loss(target_batch, 1, Reduction::Mean),
            Loss::Mse
            | Loss::L1
            | Loss::Bce
            | Loss::BceWithLogits
            | Loss::KlDiv
            | Loss::Huber
            | Loss::SmoothL1 => Err(AutodiffError::InvalidArgument(format!(
                "Sequential::fit/evaluate: Loss::{loss:?} には Tensor<f32> の target が必要\
                 （Tensor<i32> が渡された）"
            ))),
        }
    }

    fn as_class_targets(target_batch: &Tensor<i32>) -> Option<&Tensor<i32>> {
        Some(target_batch)
    }
}

/// `compile()` で構築した optimizer 本体（[`crate::optim::Sgd`]／
/// [`crate::optim::AdamW`]／[`crate::optim::Adam`]／
/// [`crate::optim::RmsProp`]／[`crate::optim::Adagrad`]／
/// [`crate::optim::Lamb`]／`Lbfgs`〈内部クレート限定。イシュー #2172〉
/// のいずれか。イシュー #2170 で `RmsProp`／`Adagrad`／`Lamb` を、
/// イシュー #2172 で `Lbfgs` を追加）。先頭 6 者は `step` のシグネチャが
/// 異なる（`Sgd::step` は位置対応スライス 2 本・他 5 者の `step` は
/// タプルスライス 1 本）ため、ここで [`Sequential::trainable_parameters`]／
/// `grad_refs` から共通の `Result<Vec<Tensor<f32>>, AutodiffError>` へ
/// 橋渡しする。`Lbfgs` はさらに異なり、`step`（本 impl）経由では更新
/// できず closure 駆動の `lbfgs_batch_step`（本ファイル下部）のみが
/// 呼べる——[`Sequential::run_fit`] がバッチ処理の入口で分岐するため
/// （本モジュール冒頭 doc 参照）。
enum OptimizerState {
    Sgd(Sgd),
    AdamW(AdamW),
    Adam(Adam),
    RmsProp(RmsProp),
    Adagrad(Adagrad),
    Lamb(Lamb),
    Lbfgs(Lbfgs),
}

impl std::fmt::Debug for OptimizerState {
    // `AdamW`／`Adam`／`RmsProp`／`Adagrad`／`Lamb`／`Lbfgs` は `Debug` を
    // 実装していない（内部のモーメント・累積バッファ・L-BFGS 履歴を
    // 丸ごと出力する `Debug` 導出をあえて設けていない設計。
    // `nn::optim::{adamw, adam, rmsprop, adagrad, lamb, lbfgs}` 参照）
    // ため、variant 名のみを出す非網羅的な `Debug` を手書きする
    // （`derive` 不可）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerState::Sgd(sgd) => f.debug_tuple("Sgd").field(sgd).finish(),
            OptimizerState::AdamW(_) => f.debug_tuple("AdamW").finish(),
            OptimizerState::Adam(_) => f.debug_tuple("Adam").finish(),
            OptimizerState::RmsProp(_) => f.debug_tuple("RmsProp").finish(),
            OptimizerState::Adagrad(_) => f.debug_tuple("Adagrad").finish(),
            OptimizerState::Lamb(_) => f.debug_tuple("Lamb").finish(),
            OptimizerState::Lbfgs(_) => f.debug_tuple("Lbfgs").finish(),
        }
    }
}

impl OptimizerState {
    fn new(optimizer: Optimizer) -> Result<Self, AutodiffError> {
        match optimizer {
            Optimizer::Sgd(config) => Ok(OptimizerState::Sgd(Sgd::new(config)?)),
            Optimizer::AdamW(config) => Ok(OptimizerState::AdamW(AdamW::new(config)?)),
            Optimizer::Adam(config) => Ok(OptimizerState::Adam(Adam::new(config)?)),
            Optimizer::RmsProp(config) => Ok(OptimizerState::RmsProp(RmsProp::new(config)?)),
            Optimizer::Adagrad(config) => Ok(OptimizerState::Adagrad(Adagrad::new(config)?)),
            Optimizer::Lamb(config) => Ok(OptimizerState::Lamb(Lamb::new(config)?)),
            Optimizer::Lbfgs(config) => Ok(OptimizerState::Lbfgs(Lbfgs::new(config)?)),
        }
    }

    /// 現在の学習率（LR scheduler 連携用。イシュー #1763）。`RmsProp`／
    /// `Adagrad`／`Lamb` は `set_lr` を持たないが `lr()`（`history.lr`
    /// 記録用）は `config().lr` で常に取得できる。`Lbfgs` も
    /// `config().lr` で取得できる（`Lbfgs::set_lr` あり）。
    fn lr(&self) -> f32 {
        match self {
            OptimizerState::Sgd(sgd) => sgd.config().lr,
            OptimizerState::AdamW(adamw) => adamw.config().lr,
            OptimizerState::Adam(adam) => adam.config().lr,
            OptimizerState::RmsProp(rmsprop) => rmsprop.config().lr,
            OptimizerState::Adagrad(adagrad) => adagrad.config().lr,
            OptimizerState::Lamb(lamb) => lamb.config().lr,
            OptimizerState::Lbfgs(lbfgs) => lbfgs.config().lr,
        }
    }

    /// 学習率のみを書き換える（各 optimizer の `set_lr` へ委譲。
    /// イシュー #1763。`Sgd`／`AdamW`／`Adam::set_lr` doc の「PyTorch の
    /// `param_group["lr"]` 書き換えと同じ意味論」節を参照——momentum
    /// バッファ／moment 推定値／`step_count` は一切リセットしない）。
    /// `Lbfgs::set_lr`（イシュー #2172）も同じ意味論（`n_iter`／履歴／
    /// `d`／`t` 等は不変）。
    ///
    /// **`RmsProp`／`Adagrad`／`Lamb`（イシュー #2170）**: いずれも
    /// `set_lr` を持たない値型のため常に `InvalidArgument` を返す。
    /// 呼び出し元 [`Sequential::fit_with_callbacks_named`] は「compiled
    /// optimizer がこの 3 者かつ `callbacks` に
    /// [`super::callbacks::Callback::LrSchedule`] を含む」場合を
    /// バッチループ・モード変更より前の引数検査で既に拒否しているため、
    /// 本 arm は通常到達しない防御的二重化（fail-closed）である。
    fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError> {
        match self {
            OptimizerState::Sgd(sgd) => sgd.set_lr(new_lr),
            OptimizerState::AdamW(adamw) => adamw.set_lr(new_lr),
            OptimizerState::Adam(adam) => adam.set_lr(new_lr),
            OptimizerState::Lbfgs(lbfgs) => lbfgs.set_lr(new_lr),
            OptimizerState::RmsProp(_) | OptimizerState::Adagrad(_) | OptimizerState::Lamb(_) => {
                Err(AutodiffError::InvalidArgument(
                    "OptimizerState::set_lr: RmsProp／Adagrad／Lamb は set_lr を \
                     提供しないため LrSchedule callback と併用できない \
                     （Sequential::fit_with_callbacks_named の引数検査で \
                     通常は事前に拒否される）"
                        .to_string(),
                ))
            }
        }
    }

    /// `params.len() != grads.len()` を各 `step` 実装（`Sgd::step` は
    /// 検査済み）へ委譲する前に、タプルスライスを要求する 5 者
    /// （`AdamW`／`Adam`／`RmsProp`／`Adagrad`／`Lamb::step`）が要求する
    /// `&[(&Tensor, &Tensor)]` への zip 変換自体が短い側で黙って
    /// 切り詰められてしまう（fail-closed 違反）のを防ぐため、ここで
    /// 明示的に事前検査する。
    ///
    /// **`Lbfgs`（イシュー #2172）は本メソッド経由では更新できない**
    /// （closure 駆動の `try_step_closure` のみが更新経路のため、
    /// `(param, grad)` の 1 回評価を前提とする本メソッドのシグネチャに
    /// 収まらない）。[`Sequential::run_fit`] は `OptimizerState::Lbfgs`
    /// を検出した時点でバッチ処理そのものを `lbfgs_batch_step` へ分岐
    /// させ、本メソッドを一切呼ばないため、この arm は通常到達しない
    /// 防御的フォールバック（fail-closed。黙って no-op にしない）。
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
            OptimizerState::RmsProp(rmsprop) => {
                let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
                    params.iter().copied().zip(grads.iter().copied()).collect();
                rmsprop.step(&pairs)
            }
            OptimizerState::Adagrad(adagrad) => {
                let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
                    params.iter().copied().zip(grads.iter().copied()).collect();
                adagrad.step(&pairs)
            }
            OptimizerState::Lamb(lamb) => {
                let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
                    params.iter().copied().zip(grads.iter().copied()).collect();
                lamb.step(&pairs)
            }
            OptimizerState::Lbfgs(_) => Err(AutodiffError::InvalidArgument(
                "OptimizerState::step: Lbfgs は closure 駆動の \
                 try_step_closure のみで更新するため、この汎用 step 経由の \
                 呼び出しには非対応（Sequential::run_fit は Lbfgs 専用の \
                 lbfgs_batch_step 経路へ分岐する）"
                    .to_string(),
            )),
        }
    }

    /// compiled optimizer が `set_lr` を持たない 3 者（`RmsProp`／
    /// `Adagrad`／`Lamb`。イシュー #2170）かどうか（`callbacks` に
    /// [`super::callbacks::Callback::LrSchedule`] が含まれる場合の
    /// fail-closed 拒否判定に使う。`OptimizerState::set_lr` doc 参照）。
    /// `Lbfgs`（イシュー #2172）は `set_lr` を持つため対応（`true`）。
    fn supports_lr_schedule(&self) -> bool {
        !matches!(
            self,
            OptimizerState::RmsProp(_) | OptimizerState::Adagrad(_) | OptimizerState::Lamb(_)
        )
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

/// `save_model` が取り出す compile 状態の写し（イシュー #2372）。
///
/// `optimizer` は `set_lr` 反映後の**現在の config**、`optimizer_state` は
/// `OptimizerStateDict::state_dict()` のキー（`optimizer.` 接頭辞なし）そのまま。
/// `Lbfgs`（#2373）は `Lbfgs::state_dict()`（接頭辞なしキー）と履歴ペア件数
/// `lbfgs_history_len`（`Lbfgs` のときだけ `Some`。load 側の照合・`load_state_dict` の
/// `expected_history_len` に使う）を持つ。AMP との併用は `compile_with_amp` が拒否済みで、
/// 保存・復元の両側でも `amp` が `Some` の `Lbfgs` を拒否する。
pub(super) struct CompiledSnapshot {
    pub(super) loss: Loss,
    pub(super) optimizer: Optimizer,
    pub(super) optimizer_state: HashMap<String, Tensor<f32>>,
    pub(super) lbfgs_history_len: Option<usize>,
    pub(super) amp: Option<AmpSnapshot>,
}

/// [`CompiledSnapshot`] の AMP 部分（`GradScaler` の save 時点の状態）。
pub(super) struct AmpSnapshot {
    pub(super) dtype: AmpDType,
    pub(super) grad_scaler_config: GradScalerConfig,
    pub(super) scale: f32,
    pub(super) growth_tracker: u64,
}

impl Sequential {
    /// compile 状態を取り出す（未 compile は `None`）。`save_model` から呼ばれる。
    pub(super) fn snapshot_compiled(&self) -> Result<Option<CompiledSnapshot>, AutodiffError> {
        let Some(compiled) = self.compiled.as_ref() else {
            return Ok(None);
        };
        // `config()` は `set_lr` 反映後の現在値を返す（LR scheduler の書き換えを含む）。
        let (optimizer, optimizer_state, lbfgs_history_len) = match &compiled.optimizer {
            OptimizerState::Sgd(o) => (Optimizer::Sgd(*o.config()), o.state_dict()?, None),
            OptimizerState::AdamW(o) => (Optimizer::AdamW(*o.config()), o.state_dict()?, None),
            OptimizerState::Adam(o) => (Optimizer::Adam(*o.config()), o.state_dict()?, None),
            OptimizerState::RmsProp(o) => (Optimizer::RmsProp(*o.config()), o.state_dict()?, None),
            OptimizerState::Adagrad(o) => (Optimizer::Adagrad(*o.config()), o.state_dict()?, None),
            OptimizerState::Lamb(o) => (Optimizer::Lamb(*o.config()), o.state_dict()?, None),
            OptimizerState::Lbfgs(o) => (
                Optimizer::Lbfgs(*o.config()),
                o.state_dict()?,
                Some(o.history_len()),
            ),
        };
        let amp = compiled.amp.as_ref().map(|a| AmpSnapshot {
            dtype: a.amp_dtype,
            grad_scaler_config: a.grad_scaler_config,
            scale: a.scaler.scale(),
            growth_tracker: a.scaler.growth_tracker(),
        });
        Ok(Some(CompiledSnapshot {
            loss: compiled.loss,
            optimizer,
            optimizer_state,
            lbfgs_history_len,
            amp,
        }))
    }

    /// 訓練対象パラメータの shape 列（`Lbfgs::load_state_dict` の `slot_shapes`）。
    fn lbfgs_slot_shapes(&self) -> Vec<Vec<usize>> {
        self.trainable_parameters()
            .iter()
            .map(|p| p.shape().to_vec())
            .collect()
    }

    /// 保存側の「書いたものは読める」検証（#2373）。`Lbfgs` の snapshot を新しい `Lbfgs` へ
    /// 試験復元する。`check_slot_shapes` は `state.` 接頭辞のキーしか見ないため `Lbfgs` には
    /// 効かず、fit 後の `add_*` によるパラメータ構成のずれや非有限の状態を保存前に拒否する
    /// ために `model_io::prepare_save` が呼ぶ。`Lbfgs` 以外は何もしない。
    pub(super) fn check_lbfgs_restorable(
        &self,
        snap: &CompiledSnapshot,
    ) -> Result<(), AutodiffError> {
        let Optimizer::Lbfgs(config) = snap.optimizer else {
            return Ok(());
        };
        let Some(n) = snap.lbfgs_history_len else {
            return Err(AutodiffError::InvalidArgument(
                "Lbfgs の snapshot に履歴件数がありません".to_string(),
            ));
        };
        Lbfgs::new(config)?.load_state_dict(
            snap.optimizer_state.clone(),
            &self.lbfgs_slot_shapes(),
            n,
        )
    }

    /// [`Self::snapshot_compiled`] の写しから compile 状態を復元する（`load_model` から呼ばれる）。
    ///
    /// construct-before-assign: optimizer の構築・状態の load・GradScaler の復元が
    /// すべて成功してから `self.compiled` へ代入する（失敗時は変更しない）。値の範囲検証は
    /// 各コンストラクタ（`*::new`・`grad_scaler_from_state`）へ委ねる。`Lbfgs`（#2373）は
    /// `load_state_dict` が履歴件数・shape・有限性・到達可能性を検証する。
    pub(super) fn restore_compiled(&mut self, snap: CompiledSnapshot) -> Result<(), AutodiffError> {
        let CompiledSnapshot {
            loss,
            optimizer,
            optimizer_state,
            lbfgs_history_len,
            amp,
        } = snap;
        // 多層防御: 履歴件数は Lbfgs のときだけ持つ。AMP との併用は compile_with_amp が拒否する組合せ。
        let is_lbfgs = matches!(optimizer, Optimizer::Lbfgs(_));
        if is_lbfgs != lbfgs_history_len.is_some() || (is_lbfgs && amp.is_some()) {
            return Err(AutodiffError::InvalidArgument(
                "Sequential の compile 状態の復元: Lbfgs の履歴件数・AMP の組合せが不正です"
                    .to_string(),
            ));
        }
        let slot_shapes = self.lbfgs_slot_shapes();
        let mut state = OptimizerState::new(optimizer)?;
        match &mut state {
            OptimizerState::Sgd(o) => o.load_state_dict(optimizer_state)?,
            OptimizerState::AdamW(o) => o.load_state_dict(optimizer_state)?,
            OptimizerState::Adam(o) => o.load_state_dict(optimizer_state)?,
            OptimizerState::RmsProp(o) => o.load_state_dict(optimizer_state)?,
            OptimizerState::Adagrad(o) => o.load_state_dict(optimizer_state)?,
            OptimizerState::Lamb(o) => o.load_state_dict(optimizer_state)?,
            OptimizerState::Lbfgs(o) => o.load_state_dict(
                optimizer_state,
                &slot_shapes,
                lbfgs_history_len.unwrap_or(0),
            )?,
        }
        let amp = match amp {
            None => None,
            Some(a) => Some(AmpState {
                dtype: a.dtype.to_scalar_dtype(),
                amp_dtype: a.dtype,
                grad_scaler_config: a.grad_scaler_config,
                scaler: grad_scaler_from_state(a.grad_scaler_config, a.scale, a.growth_tracker)?,
            }),
        };
        self.compiled = Some(Compiled {
            optimizer: state,
            loss,
            amp,
        });
        Ok(())
    }
}

/// 未 compile のモデルへ `fit`／`evaluate` を呼んだ場合の共通エラー。
fn not_compiled(method: &str) -> AutodiffError {
    AutodiffError::InvalidArgument(format!(
        "Sequential::{method}: compile() が呼ばれていない（optimizer／loss が未設定）"
    ))
}

/// カスタム学習 step フック（イシュー #2184・親 #2131。Keras
/// `Model.train_step()` 相当）。[`Sequential::run_fit`] の既定バッチ
/// 処理（`bind → forward → T::loss_for → backward → trainable_grads →
/// optimizer.step → apply_parameters`）を、このフックの呼び出しへ
/// 丸ごと差し替える。
///
/// **facade 公開面は承認待ちのため未公開**（本エイリアス自体も
/// `pub` にしない。crate 内部専用）。承認後の公開面案
/// （`TrainStepFn`／`TrainStepOptimizer`／`TrainStepOutput`）は
/// `docs/compat-train-step-hook-decision.md` §5 を参照。
///
/// 引数は `(&Sequential, x_batch, y_batch, &mut OptimizerState)`。
/// `&Sequential`（`&mut` ではない）にしているのは、フック内から
/// `compile`／`fit`／`set_training` を再入呼び出しする事故を型で防ぐ
/// ため——`&Sequential` からは `apply_parameters`（`&mut self` 必須）を
/// 呼べないため、パラメータ更新は戻り値の `Option<Vec<Tensor<f32>>>`
/// 経由で [`Sequential::run_fit`] 側が行う（`Some` で反映・`None` で
/// スキップ。AMP の非有限勾配スキップと同じ意味論）。戻り値タプルの
/// 第 1 要素はそのバッチの損失スカラーで、既存経路と同じ
/// サンプル数重み付き平均集計（`weighted_sum += loss as f64 * n_batch
/// as f64`）へそのまま合流する。
///
/// `FnMut`（`Fn` ではない）にしているのは、状態を持つ自作 optimizer や
/// 呼び出し回数の計測など、呼び出しごとに内部状態を変える利用を許す
/// ため。
type CustomStepHook<'h, T> = dyn FnMut(
        &Sequential,
        &Tensor<f32>,
        &Tensor<T>,
        &mut OptimizerState,
    ) -> Result<(f32, Option<Vec<Tensor<f32>>>), AutodiffError>
    + 'h;

/// 勾配累積（イシュー #2180）: `acc`（累積中の勾配。位置は
/// [`Sequential::trainable_grads`] と同じ順序契約）へ `grads`（このマイクロ
/// バッチの勾配）を要素ごとに f32 で逐次加算する（`acc[i] += grads[i]`。
/// PyTorch `.grad +=` と同じ意味論——[`crate::Tape::backward_accumulate`]
/// が使う `vjp_elementwise_add`〈f32 の `a + b`〉と同じ数値形式。
/// `docs/autodiff-retain-graph-accumulate-decision.md`）。正規化（`N` で
/// 割る処理）は行わない。
///
/// `.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64` アキュムレータ」
/// 原則は、カーネル内の行方向縮約（dw・bias・rstd 等、1 要素あたり多数の
/// 項を畳み込む縮約）が対象であり、本関数は縮約済みの勾配テンソルを
/// 高々 `accumulate_steps` 回（実用上小さい回数）だけ足すだけのホスト側
/// 加算のため対象外——新たな `f64` アキュムレータ契約は導入しない。
///
/// 新しい `Vec` を組み立て終えてから `*acc` へ書き戻す（原子的。途中で
/// shape 不一致等により失敗しても `acc` は変更前のまま残る）。累積
/// バッファ・加算結果バッファの確保（`try_reserve_exact`）が失敗した
/// 場合は非アロケーションな `AutodiffError::Shape(ShapeError::
/// ElementCountOverflow)`（`super::alloc_failed`。イシュー #2249）を
/// 返す。長さ不一致・shape 不一致（内部不変条件違反）は引き続き
/// `InvalidArgument` を使う。
fn accumulate_grads_into(
    acc: &mut Vec<Tensor<f32>>,
    grads: &[&Tensor<f32>],
    method: &str,
) -> Result<(), AutodiffError> {
    if acc.len() != grads.len() {
        return Err(AutodiffError::InvalidArgument(format!(
            "Sequential::{method}: 勾配累積バッファの長さ ({}) が今回の \
             勾配数 ({}) と一致しない（内部不変条件違反）",
            acc.len(),
            grads.len()
        )));
    }
    let mut next: Vec<Tensor<f32>> = Vec::new();
    next.try_reserve_exact(acc.len())
        .map_err(|_| super::alloc_failed())?;
    for (a, g) in acc.iter().zip(grads.iter()) {
        if a.shape() != g.shape() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: 勾配累積の shape が一致しない \
                 (累積側: {:?}, 今回: {:?})",
                a.shape(),
                g.shape()
            )));
        }
        let a_slice = a.host_slice();
        let g_slice = g.host_slice();
        let mut summed: Vec<f32> = Vec::new();
        summed
            .try_reserve_exact(a_slice.len())
            .map_err(|_| super::alloc_failed())?;
        summed.extend(a_slice.iter().zip(g_slice.iter()).map(|(x, y)| x + y));
        next.push(Tensor::new(summed, a.shape())?);
    }
    *acc = next;
    Ok(())
}

/// L-BFGS（イシュー #2172・親 #2131。2026-09-27 所有者承認）: 1 バッチ
/// あたりの outer step を 1 回実行する（[`Sequential::run_fit`] の
/// `OptimizerState::Lbfgs` 分岐から呼ばれる）。
///
/// 手順（`docs/autodiff-lbfgs-decision.md` §9・`crates/facade/tests/
/// compat_sequential_lbfgs_manual.rs::lbfgs_step_with_closure` と同じ
/// fail-closed 復元契約）:
///
/// 1. [`Sequential::trainable_parameters`] を snapshot として保持する。
/// 2. `lbfgs.try_step_closure` へ snapshot と closure を渡す。closure は
///    呼ばれるたびに `model.apply_parameters(trial)`（trial パラメータの
///    書き込み）→ `bind` → `forward_with_precision`（AMP 非対応のため
///    低精度 dtype は常に `None`）→ `T::loss_for` → `tape.backward` →
///    `trainable_grads` の owned clone、を行い `(損失, 勾配列)` を返す
///    （既存の非 L-BFGS 経路と同じ演算列。`nn::Dropout`／`BatchNorm` の
///    RNG・running stats は closure 評価のたびに更新される——PyTorch と
///    同じ意味論であり拒否しない。`compat::training::Optimizer::Lbfgs`
///    doc「Dropout／BatchNorm」相当の注記）。
/// 3. `Ok(updated)`: `model.apply_parameters(updated)` を行い、
///    `lbfgs.last_loss()`（その step の**初回評価**損失。PyTorch
///    `orig_loss` 相当）を返す。
/// 4. `Err(e)`: closure が既に trial パラメータを書き込んでいる可能性が
///    あるため、必ず `model.apply_parameters(snapshot)` で復元してから
///    `e` を返す（trainable params のみ復元——running stats は他
///    optimizer の失敗経路と同じく復元しない）。
fn lbfgs_batch_step<T: FitTarget>(
    model: &mut Sequential,
    lbfgs: &mut Lbfgs,
    loss: Loss,
    x_batch: &Tensor<f32>,
    y_batch: &Tensor<T>,
    method: &str,
) -> Result<f32, AutodiffError> {
    let snapshot: Vec<Tensor<f32>> = model.trainable_parameters().into_iter().cloned().collect();

    let result = lbfgs.try_step_closure(&snapshot, |trial| {
        model.apply_parameters(trial.to_vec())?;
        let tape = crate::tape();
        let bound = model.bind(&tape);
        let x_var = tape.var(x_batch);
        let pred = bound.forward_with_precision(&tape, &x_var, None)?;
        let loss_var = T::loss_for(loss, &tape, &pred, y_batch)?;
        let loss_scalar = loss_var.to_tensor().get(&[]).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: loss の shape が [] ではない\
                 （loss 演算の契約違反）"
            ))
        })?;
        let grads = tape.backward(&loss_var)?;
        let grad_refs = bound.trainable_grads(&grads)?;
        let grads_owned: Vec<Tensor<f32>> = grad_refs.into_iter().cloned().collect();
        Ok((loss_scalar, grads_owned))
    });

    match result {
        Ok(updated) => {
            model.apply_parameters(updated)?;
            lbfgs.last_loss().ok_or_else(|| {
                AutodiffError::InvalidArgument(format!(
                    "Sequential::{method}: Lbfgs::try_step_closure 成功後に \
                     last_loss が None（内部不変条件違反）"
                ))
            })
        }
        Err(err) => {
            // closure が既に試行パラメータを書き込んでいる可能性が
            // あるため、元のエラーを返す前に snapshot へ復元する
            // （fail-closed。復元自体が失敗した場合は復元エラーを
            // 優先して返す——`self.compiled` 書き戻し等、他の復元処理と
            // 同じ「復元不能は隠さず表に出す」方針）。
            model.apply_parameters(snapshot)?;
            Err(err)
        }
    }
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
    /// 1. `Linear`・`Conv2d`・`MultiheadAttention` 層 forward を
    ///    `amp.compute_dtype`（f32 master weight・backward は常に f32）
    ///    で計算する（イシュー #2071 で `Conv2d`／`MultiheadAttention`
    ///    へ拡張済み。それ以外の層〈`Conv1d`／`LayerNorm`／
    ///    `TransformerEncoderLayer` 等〉は f32 のまま。`crate::compat::
    ///    SequentialVars::forward_with_precision` doc 参照）
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
    /// `optimizer`（`OptimizerState::new`）または `amp.grad_scaler`
    /// （[`GradScaler::new`]）の検証に失敗した場合 `InvalidArgument`
    /// （fail-closed。いずれかが失敗した場合 `self.compiled` は
    /// 変更しない——[`Self::compile`] と同じ construct-before-assign
    /// 〈`?` が構造体リテラル内で先に評価されるため代入前に return
    /// する〉により、本メソッドも「全構築成功後にのみ代入する」規約
    /// で揃えている）。
    pub fn compile_with_amp(
        &mut self,
        optimizer: Optimizer,
        loss: Loss,
        amp: AmpConfig,
    ) -> Result<(), AutodiffError> {
        // イシュー #2172（L-BFGS）: AMP（損失スケーリング・非有限検出）は
        // closure 駆動の `Lbfgs::try_step_closure`（1 outer step 内で
        // forward／backward を複数回やり直す）と両立しない
        // （`GradScaler` は「1 step = 1 回の scale_loss→backward→unscale」
        // を前提とするが、L-BFGS の 1 outer step は複数回評価するため
        // scale／unscale をどの評価に適用するかが定義できない。PyTorch の
        // `GradScaler` も closure 型 `LBFGS` を公式に非対応としている）。
        // `OptimizerState::new`／`GradScaler::new` を呼ぶ前に拒否し、
        // 失敗時 `self.compiled` を変更しない（construct-before-assign。
        // `Self::compile_with_amp` doc 冒頭の契約と同じ）。
        if matches!(optimizer, Optimizer::Lbfgs(_)) {
            return Err(AutodiffError::InvalidArgument(
                "Sequential::compile_with_amp: Optimizer::Lbfgs は AMP \
                 （損失スケーリング）と併用できない（イシュー #2172。\
                 closure 駆動の outer step 内で forward／backward を \
                 複数回評価するため scale／unscale の適用箇所が定義でき \
                 ない）"
                    .to_string(),
            ));
        }
        let optimizer_state = OptimizerState::new(optimizer)?;
        let scaler = GradScaler::new(amp.grad_scaler)?;
        self.compiled = Some(Compiled {
            optimizer: optimizer_state,
            loss,
            amp: Some(AmpState {
                dtype: amp.compute_dtype.to_scalar_dtype(),
                amp_dtype: amp.compute_dtype,
                grad_scaler_config: amp.grad_scaler,
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
        self.fit_with_callbacks_named("fit", x, y, config, None, &mut [], &[], None)
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
    ///      [`super::callbacks::ModelCheckpoint::to_file`] でパスを
    ///      指定していれば、スナップショット更新時に safetensors
    ///      ファイルへも書き出す（イシュー #2073。失敗した場合は
    ///      下記「エラー」節参照）。
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
    /// - `config.epochs` が巨大で `History`（`loss`／`val_loss`／`lr`／
    ///   `val_metrics`）や勾配累積バッファの確保に失敗した場合（イシュー
    ///   #2249）→ 非アロケーションな `AutodiffError::Shape(ShapeError::
    ///   ElementCountOverflow)`（`super::alloc_failed`。以前の
    ///   `InvalidArgument(String)` から変更）。train／eval モードの
    ///   復元・`compiled` の書き戻しは他のエラーと同様に行われる
    /// - `callbacks` のいずれかが `monitor == Monitor::ValLoss`
    ///   （[`super::callbacks::EarlyStopping`]／
    ///   [`super::callbacks::ModelCheckpoint`] の既定・
    ///   [`super::callbacks::LrSchedule::plateau`] の既定）で
    ///   `validation.is_none()` の場合 → `InvalidArgument`
    ///   （train／eval モード変更前に検査するため復元は不要）
    /// - compile 済み optimizer が `RmsProp`／`Adagrad`／`Lamb`
    ///   （イシュー #2170。いずれも `set_lr` 非対応）で `callbacks` に
    ///   [`super::callbacks::Callback::LrSchedule`] を含む場合 →
    ///   `InvalidArgument`（train／eval モード変更前に検査するため
    ///   復元は不要。[`Optimizer`] enum doc 参照）
    /// - `LrSchedule::advance`／`optimizer.set_lr` が失敗した場合
    ///   （例: ユーザー定義 `LrScheduler` が非有限値を返した）→
    ///   `restore_best_weights` を（該当すれば）適用したうえで、
    ///   その時点のエラーをそのまま返す（[`Self::fit`] と同じ
    ///   fail-closed 契約: compile 済み状態は維持したまま返す）
    /// - `ModelCheckpoint::to_file` 指定時にファイル保存が失敗した
    ///   場合（イシュー #2073）→ `InvalidArgument`。in-memory
    ///   スナップショット（`best`／`best_epoch`／`state`）は永続化に
    ///   成功した場合のみ前進させる契約のため、保存失敗した epoch の
    ///   更新はコミットされず改善前の値のまま据え置かれる
    ///   （`ModelCheckpoint::observe` 節参照。一時的な保存失敗が
    ///   解消した次回以降の `fit_with_callbacks` 呼び出しで再試行
    ///   できる）。上記と同じく `restore_best_weights` を（該当すれば）
    ///   適用したうえで返す
    pub fn fit_with_callbacks<T: FitTarget>(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        config: FitConfig,
        validation: Option<(&Tensor<f32>, &Tensor<T>)>,
        callbacks: &mut [Callback],
    ) -> Result<History, AutodiffError> {
        self.fit_with_callbacks_named(
            "fit_with_callbacks",
            x,
            y,
            config,
            validation,
            callbacks,
            &[],
            None,
        )
    }

    /// [`Self::fit_with_callbacks`] の拡張版（イシュー #2072・親
    /// #2059）: `metrics`（[`super::metrics::Metrics`] の集合）を追加で
    /// 受け取り、`validation` が `Some` の場合のみ epoch 末に
    /// [`super::metrics::MetricsResult`] を計算して
    /// [`History::val_metrics`] へ積む。`metrics = &[]` のとき
    /// [`Self::fit_with_callbacks`] と完全に同一の演算列・戻り値になる
    /// （`fit_with_metrics_empty_matches_fit_with_callbacks_bit_exact`
    /// で検証）。
    ///
    /// metrics 計算は validation フェーズ（[`Self::evaluate`] と同じ
    /// eval モード）でのみ行うため、学習の演算列（`bind → forward →
    /// T::loss_for → backward → trainable_grads → optimizer.step →
    /// apply_parameters`）・`history.loss`／`lr` は
    /// [`Self::fit_with_callbacks`] と bit 完全一致する。
    ///
    /// [`super::callbacks::Callback::ModelCheckpoint`]／
    /// [`super::callbacks::Callback::EarlyStopping`]／
    /// [`super::callbacks::Callback::LrSchedule`] は
    /// [`super::callbacks::Monitor::ValMetric`] で metrics を監視
    /// できる（`MonitorMode` の既定は `Min` のまま——accuracy 等を
    /// 監視する場合は呼び出し側が `.mode(MonitorMode::Max)` を明示する
    /// 必要がある。自動推定はしない）。
    ///
    /// # エラー
    ///
    /// [`Self::fit_with_callbacks`] の既存エラー契約に加え:
    /// - `!metrics.is_empty() && validation.is_none()` →
    ///   `InvalidArgument`（metrics は validation set 上で定義される
    ///   指標のため）
    /// - `callbacks` のいずれかが `Monitor::ValMetric(m)` を監視し、
    ///   `m` が `metrics` に含まれない、または `m ==
    ///   Metrics::ConfusionMatrix`（非スカラーのため監視値として
    ///   定義できない）の場合 → `InvalidArgument`（callback が黙って
    ///   スキップされる穴を防ぐ）
    /// - `metrics` が非空かつ `T = f32`（[`Loss::Mse`]）の場合 →
    ///   `InvalidArgument`（metrics は分類 target（`Tensor<i32>`）
    ///   でのみ定義される）
    #[allow(clippy::too_many_arguments)]
    pub fn fit_with_metrics<T: FitTarget>(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        config: FitConfig,
        validation: Option<(&Tensor<f32>, &Tensor<T>)>,
        callbacks: &mut [Callback],
        metrics: &[Metrics],
    ) -> Result<History, AutodiffError> {
        self.fit_with_callbacks_named(
            "fit_with_metrics",
            x,
            y,
            config,
            validation,
            callbacks,
            metrics,
            None,
        )
    }

    /// `method`（呼び出し元の公開メソッド名。[`Self::fit`]／
    /// [`Self::fit_with_callbacks`]／[`Self::fit_with_metrics`] の
    /// いずれか）を渡し、エラーメッセージが実際に呼ばれた公開メソッド
    /// 名を名乗るようにする（イシュー #1763 PR #1883 レビュー指摘の
    /// 是正を踏襲）。`metrics` はイシュー #2072 で追加した引数——
    /// [`Self::fit`]／[`Self::fit_with_callbacks`] は `&[]` で委譲する
    /// ため、両者の演算列・戻り値は本引数追加の前後で変化しない。
    /// `custom_step`（イシュー #2184）は [`CustomStepHook`] doc 参照——
    /// 既存 3 入口はいずれも `None` で委譲するため、本引数追加自体は
    /// それらの演算列・戻り値を一切変えない（R3）。
    #[allow(clippy::too_many_arguments)]
    fn fit_with_callbacks_named<T: FitTarget>(
        &mut self,
        method: &'static str,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        config: FitConfig,
        validation: Option<(&Tensor<f32>, &Tensor<T>)>,
        callbacks: &mut [Callback],
        metrics: &[Metrics],
        custom_step: Option<&mut CustomStepHook<'_, T>>,
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
        // イシュー #2398: `add_module` で積んだパラメータ持ちの独自層は `bind` が追跡せず
        // 学習されないため、モード変更・DataLoader 構築より前に型付き拒否する。
        if let Err(e) = self.reject_untracked_parametric_layer(&format!("Sequential::{method}")) {
            self.compiled = Some(compiled);
            return Err(e);
        }
        // イシュー #2530: batch_first=false の MHA は DataLoader の第 0 軸分割と不整合のため拒否。
        if let Err(e) = self.reject_seq_first_layer_for_fit(&format!("Sequential::{method}")) {
            self.compiled = Some(compiled);
            return Err(e);
        }
        // (1.5) 勾配累積（イシュー #2180）の引数検査。`accumulate_steps
        // == 0` はウィンドウ幅として意味を持たない（fail-closed）。
        if config.accumulate_steps == 0 {
            self.compiled = Some(compiled);
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: accumulate_steps == 0"
            )));
        }
        // AMP（[`Self::compile_with_amp`]）と `accumulate_steps > 1` の
        // 組み合わせは未実装（実装計画 §3.4「代替」方式）。AMP は
        // `GradScaler::update` を「窓ごとに 1 回」呼ぶ必要があるが、
        // 現在の実装は `accumulate_steps == 1` の場合と bit 同一を保つ
        // ため各マイクロバッチで step／update するAMP 経路をそのまま
        // 維持しており、`accumulate_steps > 1` と組み合わせると
        // `scaler.update` が窓ごとではなくマイクロバッチごとに呼ばれて
        // しまう（scale の意味論が崩れる）。fail-closed に拒否する。
        if config.accumulate_steps > 1 && compiled.amp.is_some() {
            self.compiled = Some(compiled);
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: accumulate_steps > 1 は \
                 compile_with_amp（AMP）と併用できない（イシュー #2180 \
                 スコープ外）"
            )));
        }
        // (1.6) カスタム学習 step フック（イシュー #2184）の組み合わせ
        // 検査。いずれも fail-closed（実装計画 §3.4）:
        // - AMP: `GradScaler::scale_loss`／`unscale`／`update` は既定
        //   step の一部で、フックが迂回すると scaler 状態が崩れる。
        // - 勾配累積: 更新の粒度をフック側が持つため、`accumulate_steps`
        //   のウィンドウ処理と両立しない。
        if custom_step.is_some() && compiled.amp.is_some() {
            self.compiled = Some(compiled);
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: カスタム学習 step フックは \
                 compile_with_amp（AMP）と併用できない（イシュー #2184）"
            )));
        }
        if custom_step.is_some() && config.accumulate_steps > 1 {
            self.compiled = Some(compiled);
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: カスタム学習 step フックは \
                 accumulate_steps > 1（勾配累積）と併用できない \
                 （イシュー #2184）"
            )));
        }
        // (1.7) L-BFGS（イシュー #2172）の組み合わせ検査。いずれも
        // fail-closed（`compat::training::Optimizer::Lbfgs` doc「非対応の
        // 組み合わせ」節参照）:
        // - `accumulate_steps > 1`: L-BFGS の 1 outer step は closure を
        //   複数回評価して 1 回のパラメータ更新を行う独自のウィンドウ
        //   処理を内包しており、マイクロバッチ勾配の逐次加算という
        //   `accumulate_steps` の意味論とかみ合わない。
        // - カスタム学習 step フック: フックのシグネチャは `&Sequential`
        //   （不変参照）しかモデルへ渡さないため、trial パラメータを
        //   都度書き込む L-BFGS の closure（`model.apply_parameters`
        //   に `&mut Sequential` を要求する）をフック内から駆動できない。
        if config.accumulate_steps > 1 && matches!(compiled.optimizer, OptimizerState::Lbfgs(_)) {
            self.compiled = Some(compiled);
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: accumulate_steps > 1 は \
                 Optimizer::Lbfgs と併用できない（イシュー #2172）"
            )));
        }
        if custom_step.is_some() && matches!(compiled.optimizer, OptimizerState::Lbfgs(_)) {
            self.compiled = Some(compiled);
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: カスタム学習 step フックは \
                 Optimizer::Lbfgs と併用できない（イシュー #2172。フックは \
                 &Sequential〈不変参照〉しか受け取らず closure 駆動の \
                 trial パラメータ書き込みを実行できない）"
            )));
        }
        if validation.is_none()
            && let Some(offending) = callbacks.iter().find(|cb| cb.requires_validation())
        {
            self.compiled = Some(compiled);
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: callback {offending:?} は \
                 Monitor::ValLoss または Monitor::ValMetric を監視するが \
                 validation が None"
            )));
        }
        // (2.05) LrSchedule × set_lr 非対応 optimizer の fail-closed
        // 拒否（イシュー #2170）: `RmsProp`／`Adagrad`／`Lamb` は
        // `set_lr` を提供しないため、`Callback::LrSchedule` と組み
        // 合わせても黙って LR 更新が効かないまま学習が進んでしまう
        // （`Optimizer` enum doc・`OptimizerState::set_lr` doc 参照）。
        // モード変更・パラメータ更新より前に検査する。
        if !compiled.optimizer.supports_lr_schedule()
            && callbacks
                .iter()
                .any(|cb| matches!(cb, Callback::LrSchedule(_)))
        {
            self.compiled = Some(compiled);
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: compile() した optimizer（RmsProp／\
                 Adagrad／Lamb のいずれか）は set_lr を提供しないため \
                 Callback::LrSchedule と併用できない"
            )));
        }
        // (2.1) metrics（イシュー #2072）: validation set 上でのみ定義
        // される指標のため、metrics 非空かつ validation が None なら
        // 拒否する。
        if !metrics.is_empty() && validation.is_none() {
            self.compiled = Some(compiled);
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: metrics が非空だが validation が None\
                 （metrics は validation set 上で定義される指標のため）"
            )));
        }
        // (2.2) callbacks が Monitor::ValMetric(m) を監視する場合、
        // `m` が `metrics` に含まれる（`value_at` が黙って `None` を
        // 返し callback がスキップされる穴を防ぐ）・非スカラー
        // （`Metrics::ConfusionMatrix`）でないことを検査する。
        for cb in callbacks.iter() {
            if let Some(super::callbacks::Monitor::ValMetric(m)) = cb.monitor() {
                if m == Metrics::ConfusionMatrix {
                    self.compiled = Some(compiled);
                    return Err(AutodiffError::InvalidArgument(format!(
                        "Sequential::{method}: callback {cb:?} は \
                         Monitor::ValMetric(Metrics::ConfusionMatrix)（非スカラー）を \
                         監視できない"
                    )));
                }
                if !metrics.contains(&m) {
                    self.compiled = Some(compiled);
                    return Err(AutodiffError::InvalidArgument(format!(
                        "Sequential::{method}: callback {cb:?} は Monitor::ValMetric({m:?}) を \
                         監視するが、{m:?} が metrics 引数に含まれない"
                    )));
                }
            }
        }
        // (2.3) metrics（イシュー #2072）は分類 target（`Tensor<i32>`）
        // でのみ定義される。`T = f32`（`Loss::Mse`）での呼び出しを
        // 演算列に入る前に拒否する。
        if !metrics.is_empty()
            && let Err(e) = T::require_metrics_support()
        {
            self.compiled = Some(compiled);
            return Err(e);
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

        let result = self.run_fit(
            method,
            &mut compiled,
            x,
            y,
            config,
            validation,
            callbacks,
            metrics,
            custom_step,
        );

        // (4) モード復元・compiled の書き戻し（結果を問わず必ず行う。
        // fail-closed: 失敗した fit の後もモデルを「未 compile」状態へ
        // 落とさない）。
        self.set_training(prev_training);
        self.compiled = Some(compiled);
        result
    }

    /// [`Self::fit_with_callbacks`]／[`Self::fit_with_metrics`] の本体
    /// （compiled を取り出し済みの状態で呼ばれる。`&mut self` と
    /// `compiled: &mut Compiled` を独立した借用として受け取ることで、
    /// `self.bind(&tape)`〈`&self`〉と `compiled.optimizer.step`〈`&mut
    /// compiled`〉を同時に生かせる）。`method`（呼び出し元の公開
    /// メソッド名。イシュー #1763 PR #1883 レビュー指摘の是正）・
    /// `metrics`（イシュー #2072）を追加したことで引数が 9 個になり、
    /// `custom_step`（イシュー #2184。[`CustomStepHook`] doc 参照）で
    /// 10 個になった。呼び出し元は [`Self::fit_with_callbacks_named`]
    /// の 1 箇所のみのため、引数の構造体化は公開入口が確定する承認後に
    /// まとめて検討する（`docs/compat-train-step-hook-decision.md`）。
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
        metrics: &[Metrics],
        mut custom_step: Option<&mut CustomStepHook<'_, T>>,
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
        // で失敗可能にし、確保失敗は非アロケーションな
        // `AutodiffError::Shape(ShapeError::ElementCountOverflow)`
        // （[`super::alloc_failed`]。イシュー #2249）へマッピングして
        // 呼び出し元（[`Self::fit_with_callbacks`]）の既存復元経路
        // （train／eval モード巻き戻し・`compiled` 復元）へ返す。確保
        // 失敗の*報告*自体が新たなヒープ確保（`format!` の `String`）を
        // 行わないようにするため、診断メッセージは持たせない。
        let mut loss = Vec::new();
        loss.try_reserve_exact(config.epochs)
            .map_err(|_| super::alloc_failed())?;
        // `val_loss` は `validation.is_some()` のときのみ epochs 分
        // 確保する（`None` の場合は空のまま。`History::val_loss` doc
        // 参照）。
        let mut val_loss = Vec::new();
        if validation.is_some() {
            val_loss
                .try_reserve_exact(config.epochs)
                .map_err(|_| super::alloc_failed())?;
        }
        let mut lr = Vec::new();
        lr.try_reserve_exact(config.epochs)
            .map_err(|_| super::alloc_failed())?;
        // `val_metrics` は `validation.is_some() && !metrics.is_empty()`
        // のときのみ epochs 分確保する（`History::val_metrics` doc
        // 参照）。
        let mut val_metrics = Vec::new();
        if validation.is_some() && !metrics.is_empty() {
            val_metrics
                .try_reserve_exact(config.epochs)
                .map_err(|_| super::alloc_failed())?;
        }
        let mut history = History {
            loss,
            val_loss,
            lr,
            val_metrics,
        };

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
                // 勾配累積（イシュー #2180）: `acc` は累積中の勾配
                // （[`Sequential::trainable_grads`] と同じ位置対応順序）・
                // `micro` はこのウィンドウで処理したマイクロバッチ数。
                // 各 epoch の開始時に必ず空の状態から始まり、ウィンドウの
                // 境界（`micro == config.accumulate_steps`）または epoch
                // 末の端数フラッシュで必ず消費し尽くされる（fit 呼び出し
                // をまたいで持ち越さない）。`accumulate_steps == 1`
                // （既定）のとき、1 マイクロバッチ目で常に境界へ到達する
                // ため毎回 clone のみ（加算は一度も走らない）で
                // `compiled.optimizer.step` を呼ぶ——既存の非累積経路と
                // bit 完全一致する（R3）。
                let mut acc: Option<Vec<Tensor<f32>>> = None;
                let mut micro: u32 = 0;

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
                    // カスタム学習 step フック（イシュー #2184）: `Some`
                    // なら既定のバッチ処理（このブロック直下の
                    // `else`）を丸ごと迂回し、フックの呼び出しだけで
                    // このバッチの loss・パラメータ更新を決める。
                    // AMP・勾配累積との併用は引数検査で fail-closed
                    // 拒否済み（本関数の呼び出し元 doc 参照）のため、
                    // フック使用時に `compiled.amp`／`micro`／`acc` へ
                    // 触れることはない。
                    let updated: Option<Vec<Tensor<f32>>> = if let Some(hook) = custom_step.as_mut()
                    {
                        let (loss_scalar, updated_from_hook) =
                            match hook(&*self, &x_batch, &y_batch, &mut compiled.optimizer) {
                                Ok(v) => v,
                                Err(e) => break 'epochs_block Err(e),
                            };
                        // 既存経路（本関数 doc「適用順序契約」節）と同一の
                        // サンプル数重み付き平均集計式。非有限値もそのまま
                        // 記録する（既存経路と同じ）。
                        weighted_sum += loss_scalar as f64 * n_batch as f64;
                        count += n_batch;
                        updated_from_hook
                    } else if let OptimizerState::Lbfgs(lbfgs) = &mut compiled.optimizer {
                        // L-BFGS（イシュー #2172）: `lbfgs_batch_step` が
                        // snapshot 取得・closure 駆動の outer step・
                        // 成功時の `apply_parameters`／失敗時の snapshot
                        // 復元までを一元的に担う（`compat::training::
                        // Optimizer::Lbfgs` doc 参照）。AMP・勾配累積との
                        // 併用は引数検査で fail-closed 拒否済みのため、
                        // この分岐で `compiled.amp`／`micro`／`acc` へ
                        // 触れることはない（`compiled.amp` は
                        // `compile_with_amp` が `Optimizer::Lbfgs` を拒否
                        // するため常に `None`）。
                        let loss_scalar = match lbfgs_batch_step(
                            self,
                            lbfgs,
                            compiled.loss,
                            &x_batch,
                            &y_batch,
                            method,
                        ) {
                            Ok(v) => v,
                            Err(e) => break 'epochs_block Err(e),
                        };
                        weighted_sum += loss_scalar as f64 * n_batch as f64;
                        count += n_batch;
                        // パラメータは `lbfgs_batch_step` 内で既に
                        // `apply_parameters` 済みのため、本ループ末尾の
                        // 共通 `apply_parameters` 呼び出し（`updated` が
                        // `Some` の場合のみ実行）を起動しないよう `None`
                        // を返す。
                        None
                    } else {
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

                            // 勾配累積（イシュー #2180）。AMP は
                            // `accumulate_steps > 1` と併用できない
                            // （引数検査で fail-closed 拒否済み）ため、
                            // この分岐にのみ累積ロジックを持つ。
                            micro += 1;
                            if micro == 1 {
                                // R3: 1 マイクロステップ目は clone のみ
                                // （算術を一切行わない）。
                                // `accumulate_steps == 1` のときは常に
                                // ここで境界に到達し、加算は一度も
                                // 起きないため既存経路と bit 同一になる。
                                let mut cloned: Vec<Tensor<f32>> = Vec::new();
                                if cloned.try_reserve_exact(grad_refs.len()).is_err() {
                                    break 'epochs_block Err(super::alloc_failed());
                                }
                                cloned.extend(grad_refs.iter().map(|g| (*g).clone()));
                                acc = Some(cloned);
                            } else {
                                let acc_buf = match acc.as_mut() {
                                    Some(buf) => buf,
                                    None => {
                                        break 'epochs_block Err(AutodiffError::InvalidArgument(
                                            format!(
                                                "Sequential::{method}: 勾配累積バッファが\
                                                 初期化されていない（内部不変条件違反）"
                                            ),
                                        ));
                                    }
                                };
                                if let Err(e) = accumulate_grads_into(acc_buf, &grad_refs, method) {
                                    break 'epochs_block Err(e);
                                }
                            }

                            if micro == config.accumulate_steps {
                                let param_refs = self.trainable_parameters();
                                let acc_buf = match acc.as_ref() {
                                    Some(buf) => buf,
                                    None => {
                                        break 'epochs_block Err(AutodiffError::InvalidArgument(
                                            format!(
                                                "Sequential::{method}: 勾配累積バッファが\
                                                 初期化されていない（内部不変条件違反）"
                                            ),
                                        ));
                                    }
                                };
                                let acc_refs: Vec<&Tensor<f32>> = acc_buf.iter().collect();
                                let stepped = match compiled.optimizer.step(&param_refs, &acc_refs)
                                {
                                    Ok(v) => v,
                                    Err(e) => break 'epochs_block Err(e),
                                };
                                acc = None;
                                micro = 0;
                                Some(stepped)
                            } else {
                                None
                            }
                        }
                    };
                    if let Some(updated) = updated
                        && let Err(e) = self.apply_parameters(updated)
                    {
                        break 'epochs_block Err(e);
                    }
                }

                // 勾配累積（イシュー #2180）: epoch 末の端数ウィンドウ
                // flush。`micro > 0` は「`accumulate_steps` に満たない
                // まま epoch が終わった」ことを表す（例: `N = 3` でバッチ
                // 数 7 の場合、最後の 1 バッチ分）。validation・callbacks・
                // `ModelCheckpoint`／`EarlyStopping` が epoch 末に必ず
                // 更新済みのパラメータを見られるよう、ここで強制的に
                // step する（fit 呼び出しをまたいで持ち越さない契約。
                // 実装計画 §3.3「step の境界」参照）。
                if micro > 0 {
                    let param_refs = self.trainable_parameters();
                    let acc_buf = match acc.as_ref() {
                        Some(buf) => buf,
                        None => {
                            break 'epochs_block Err(AutodiffError::InvalidArgument(format!(
                                "Sequential::{method}: 勾配累積バッファが初期化されていない\
                                 （内部不変条件違反）"
                            )));
                        }
                    };
                    let acc_refs: Vec<&Tensor<f32>> = acc_buf.iter().collect();
                    let stepped = match compiled.optimizer.step(&param_refs, &acc_refs) {
                        Ok(v) => v,
                        Err(e) => break 'epochs_block Err(e),
                    };
                    if let Err(e) = self.apply_parameters(stepped) {
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
                // の処理順序」節。metrics 計算はイシュー #2072・
                // `run_evaluate_with_metrics` doc 参照）。
                if let Some((x_val, y_val)) = validation {
                    self.set_training(false);
                    let v = self.run_evaluate_with_metrics::<T>(
                        x_val,
                        y_val,
                        config.batch_size,
                        compiled.loss,
                        metrics,
                        method,
                    );
                    self.set_training(true);
                    match v {
                        Ok((loss_v, metrics_v)) => {
                            history.val_loss.push(loss_v);
                            if let Some(m) = metrics_v {
                                history.val_metrics.push(m);
                            }
                        }
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
                            if let Some(value) = mc.monitor_value_at(&history, epoch_local)
                                && let Err(e) = mc.observe(value, self)
                            {
                                // `to_file` 指定時のファイル保存失敗
                                // （イシュー #2073）。`observe` は永続化
                                // に成功した場合のみ in-memory 側
                                // （`best`／`best_epoch`／`state`）を
                                // 前進させる契約のため、この epoch の
                                // 更新はコミットされず改善前の値のまま
                                // 据え置かれた状態で `'epochs_block` を
                                // 抜ける（次回以降の
                                // `fit_with_callbacks` 呼び出しで再試行
                                // できる）。後続の `EarlyStopping::
                                // restore_best_weights` 復元・train／
                                // eval モード復元・`compiled` 書き戻しは
                                // 通常どおり実行される（`callbacks.rs`
                                // モジュール冒頭 doc「`ModelCheckpoint`
                                // のファイル保存」節参照）。
                                break 'epochs_block Err(AutodiffError::InvalidArgument(format!(
                                    "Sequential::{method}: ModelCheckpoint::to_file の保存に失敗した: {e}"
                                )));
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
        let result = self
            .run_evaluate_with_metrics::<T>(x, y, batch_size, loss, &[], "evaluate")
            .map(|(loss_v, _metrics_v)| loss_v);
        self.set_training(prev_training);
        result
    }

    /// `method` には呼び出し元の公開メソッド名（`"evaluate"`、または
    /// [`Self::fit_with_callbacks_named`] の validation フェーズ経由
    /// なら `"fit"`／`"fit_with_callbacks"`／`"fit_with_metrics"`）を
    /// 渡し、エラーメッセージが実際に呼ばれた公開メソッド名を名乗る
    /// ようにする（イシュー #1763 PR #1883 レビュー指摘の是正）。
    ///
    /// `metrics`（イシュー #2072）が空の場合、loss の演算列
    /// （`forward → T::loss_for → to_tensor().get(&[])`）は本引数追加
    /// 前と完全に同一（bit 一致契約）。非空の場合のみ、同じ `pred`
    /// （forward 出力。`cross_entropy_loss` は融合対象外のため既に
    /// 実体化済み）に対して `pred.argmax(Some(1))`（非微分・[`crate::
    /// Var::argmax`] 契約）を追加で計算し
    /// [`super::metrics::ConfusionAccumulator`] へ蓄積する。
    fn run_evaluate_with_metrics<T: FitTarget>(
        &self,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        batch_size: usize,
        loss: Loss,
        metrics: &[Metrics],
        method: &str,
    ) -> Result<(f32, Option<MetricsResult>), AutodiffError> {
        self.reject_seq_first_layer_for_fit(&format!("Sequential::{method}"))?;
        let to_invalid_arg = |e: fandhe_ai_tensor_core::data::DataError| {
            AutodiffError::InvalidArgument(format!("Sequential::{method}: {e}"))
        };
        let x_dataset = TensorDataset::new(x.clone()).map_err(to_invalid_arg)?;
        let y_dataset = TensorDataset::new(y.clone()).map_err(to_invalid_arg)?;
        let loader = DataLoader::new((x_dataset, y_dataset), DataLoaderConfig::new(batch_size))
            .map_err(to_invalid_arg)?;

        let mut weighted_sum = 0.0f64;
        let mut count = 0usize;
        let mut accumulator: Option<ConfusionAccumulator> = None;

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

            if !metrics.is_empty() {
                let target_batch = T::as_class_targets(&y_batch).ok_or_else(|| {
                    AutodiffError::InvalidArgument(format!(
                        "Sequential::{method}: metrics はクラス添字 target（Loss::CrossEntropy／\
                         Loss::Nll・Tensor<i32>）でのみ計算できる"
                    ))
                })?;
                // `Var::shape` は `pub(crate)`（autodiff クレート内限定）
                // のため facade からは到達できず、`to_tensor()`
                // （公開 API）で実体化した `Tensor<f32>` 経由で shape を
                // 読む（`loss_var` 計算で `pred` は既に実体化済みのため
                // 追加の演算コストはメモリコピー相当のみ）。
                let pred_tensor = pred.to_tensor();
                let pred_shape = pred_tensor.shape();
                if pred_shape.len() != 2 {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "Sequential::{method}: metrics には pred（logits）が rank 2 [N, C] \
                         である必要がある (実際は {pred_shape:?})"
                    )));
                }
                let num_classes = pred_shape[1];
                let pred_classes = pred.argmax(Some(1))?;
                match accumulator.as_mut() {
                    Some(acc) => acc.observe(&pred_classes, target_batch)?,
                    None => {
                        let mut acc = ConfusionAccumulator::new(num_classes)?;
                        acc.observe(&pred_classes, target_batch)?;
                        accumulator = Some(acc);
                    }
                }
            }
        }

        if count == 0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::{method}: 処理されたサンプルが 0 件"
            )));
        }
        let avg_loss = (weighted_sum / count as f64) as f32;
        let metrics_result = match accumulator {
            Some(acc) => Some(acc.finish(metrics)?),
            None => None,
        };
        Ok((avg_loss, metrics_result))
    }
}

/// カスタム学習 step フック（イシュー #2184）のテスト専用入口。
/// [`CustomStepHook`]（非公開・crate 内部専用）を型に含むシグネチャの
/// ため `pub(crate)` にはできない（`private_interfaces` lint に掛かる）。
/// 同じ `training.rs` モジュール内の `#[cfg(test)] mod` はこの非公開
/// メソッドを問題なく呼べる。
#[cfg(test)]
impl Sequential {
    #[allow(clippy::too_many_arguments)]
    fn fit_custom_step_for_test<T: FitTarget>(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        config: FitConfig,
        validation: Option<(&Tensor<f32>, &Tensor<T>)>,
        callbacks: &mut [Callback],
        metrics: &[Metrics],
        hook: &mut CustomStepHook<'_, T>,
    ) -> Result<History, AutodiffError> {
        self.fit_with_callbacks_named(
            "fit",
            x,
            y,
            config,
            validation,
            callbacks,
            metrics,
            Some(hook),
        )
    }
}

/// カスタム学習 step フック（イシュー #2184・親 #2131）の単体テスト。
/// `Sequential::fit_custom_step_for_test`（`#[cfg(test)]` 限定）は本
/// クレート内からしか呼べないため、`crates/facade/tests/*` ではなく
/// ここに置く（勾配累積の外部テストは
/// `tests/compat_sequential_accumulate.rs`）。
#[cfg(test)]
mod train_step_tests {
    use super::super::callbacks::EarlyStopping;
    use super::*;

    /// [`CustomStepHook`] の戻り値型（`clippy::type_complexity` 回避の
    /// テスト局所エイリアス。本体側は `type CustomStepHook` の定義自体が
    /// 「型定義への factoring」を満たすため対象外だが、テスト側の各
    /// フック関数・クロージャの戻り値注釈は同じ複合型を直書きするため
    /// 必要）。
    type StepOutcome = Result<(f32, Option<Vec<Tensor<f32>>>), AutodiffError>;

    const D_IN: usize = 3;
    const D_HIDDEN: usize = 4;
    const D_OUT: usize = 2;
    const SEED_L1: u64 = 0x7570_1111;
    const SEED_L2: u64 = 0x7570_2222;

    fn build_model() -> Sequential {
        Sequential::new()
            .add_linear(D_IN, D_HIDDEN, SEED_L1)
            .unwrap_or_else(|e| panic!("test fixture: 層 1 の構築に失敗: {e}"))
            .add_relu()
            .add_linear(D_HIDDEN, D_OUT, SEED_L2)
            .unwrap_or_else(|e| panic!("test fixture: 層 2 の構築に失敗: {e}"))
    }

    /// `tests/compat_sequential_accumulate.rs` の `deterministic_fill` と同型の局所実装
    /// （splitmix64。値域 `(-0.5, 0.5)`）。
    fn deterministic_fill(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
            })
            .collect()
    }

    fn gen_regression_data(seed: u64, n: usize) -> (Tensor<f32>, Tensor<f32>) {
        let x = deterministic_fill(seed, n * D_IN);
        let y = deterministic_fill(seed ^ 0x5555_5555_5555_5555, n * D_OUT);
        (
            Tensor::new(x, &[n, D_IN])
                .unwrap_or_else(|e| panic!("test fixture: x の shape 構築に失敗: {e}")),
            Tensor::new(y, &[n, D_OUT])
                .unwrap_or_else(|e| panic!("test fixture: y の shape 構築に失敗: {e}")),
        )
    }

    fn params_bit_exact(a: &[&Tensor<f32>], b: &[&Tensor<f32>]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (x, y) in a.iter().zip(b.iter()) {
            let xd = x.host_slice();
            let yd = y.host_slice();
            if xd.len() != yd.len() {
                return false;
            }
            if xd
                .iter()
                .zip(yd.iter())
                .any(|(xv, yv)| xv.to_bits() != yv.to_bits())
            {
                return false;
            }
        }
        true
    }

    /// 既定 step（`bind → forward → mse_loss(Reduction::Mean) →
    /// backward → trainable_grads → opt.step`）を手で再実装したフック。
    /// T1 が使う——既定の `fit` と bit 完全一致することの証明になる。
    fn default_step_hook(
        m: &Sequential,
        x_batch: &Tensor<f32>,
        y_batch: &Tensor<f32>,
        opt: &mut OptimizerState,
    ) -> StepOutcome {
        let tape = crate::tape();
        let bound = m.bind(&tape);
        let x_var = tape.var(x_batch);
        let pred = bound.forward_with_precision(&tape, &x_var, None)?;
        let target_var = tape.var(y_batch);
        let loss_var = pred.mse_loss_with(&target_var, Reduction::Mean)?;
        let loss_scalar = loss_var.to_tensor().get(&[]).ok_or_else(|| {
            AutodiffError::InvalidArgument("test fixture: loss の shape が [] ではない".to_string())
        })?;
        let grads = tape.backward(&loss_var)?;
        let grad_refs = bound.trainable_grads(&grads)?;
        let param_refs = m.trainable_parameters();
        let stepped = opt.step(&param_refs, &grad_refs)?;
        Ok((loss_scalar, Some(stepped)))
    }

    /// T1（配線・n_batch 重み付け・step/apply 順序の証明）: 既定 step を
    /// 手で再実装したフックで fit すると、`History.loss`・学習後
    /// パラメータが既定の `fit` と bit 完全一致する（SGD 構成）。
    /// epochs=3・端数バッチを含む構成（N=7・batch=2）。
    #[test]
    fn custom_step_reimplementing_default_matches_fit_bit_exact_sgd() {
        const N: usize = 7;
        const BATCH: usize = 2;
        const EPOCHS: usize = 3;
        let (x, y) = gen_regression_data(0x7570_AAAA, N);

        let mut default_model = build_model();
        default_model
            .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
            .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));
        let default_history = default_model
            .fit(&x, &y, FitConfig::new(EPOCHS, BATCH))
            .unwrap_or_else(|e| panic!("既定 fit に失敗: {e}"));

        let mut hook_model = build_model();
        hook_model
            .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
            .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));
        let hook_history = hook_model
            .fit_custom_step_for_test(
                &x,
                &y,
                FitConfig::new(EPOCHS, BATCH),
                None,
                &mut [],
                &[],
                &mut default_step_hook,
            )
            .unwrap_or_else(|e| panic!("フック fit に失敗: {e}"));

        assert_eq!(
            hook_history.loss, default_history.loss,
            "既定 step を再実装したフックの History.loss（SGD）が既定の fit と bit 一致しない"
        );
        assert!(
            params_bit_exact(
                &default_model.trainable_parameters(),
                &hook_model.trainable_parameters()
            ),
            "既定 step を再実装したフックの学習後パラメータ（SGD）が既定の fit と bit 一致しない"
        );
    }

    /// T1 の AdamW 構成版（optimizer 種別に依存しないことの確認）。
    #[test]
    fn custom_step_reimplementing_default_matches_fit_bit_exact_adamw() {
        const N: usize = 7;
        const BATCH: usize = 2;
        const EPOCHS: usize = 3;
        let (x, y) = gen_regression_data(0x7570_BBBB, N);

        let mut default_model = build_model();
        default_model
            .compile(
                Optimizer::AdamW(AdamWConfig {
                    lr: 0.01,
                    ..Default::default()
                }),
                Loss::Mse,
            )
            .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));
        let default_history = default_model
            .fit(&x, &y, FitConfig::new(EPOCHS, BATCH))
            .unwrap_or_else(|e| panic!("既定 fit に失敗: {e}"));

        let mut hook_model = build_model();
        hook_model
            .compile(
                Optimizer::AdamW(AdamWConfig {
                    lr: 0.01,
                    ..Default::default()
                }),
                Loss::Mse,
            )
            .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));
        let hook_history = hook_model
            .fit_custom_step_for_test(
                &x,
                &y,
                FitConfig::new(EPOCHS, BATCH),
                None,
                &mut [],
                &[],
                &mut default_step_hook,
            )
            .unwrap_or_else(|e| panic!("フック fit に失敗: {e}"));

        assert_eq!(
            hook_history.loss, default_history.loss,
            "既定 step を再実装したフックの History.loss（AdamW）が既定の fit と bit 一致しない"
        );
        assert!(
            params_bit_exact(
                &default_model.trainable_parameters(),
                &hook_model.trainable_parameters()
            ),
            "既定 step を再実装したフックの学習後パラメータ（AdamW）が既定の fit と bit 一致しない"
        );
    }

    /// T2（R4）: 損失を自作（`Reduction::Sum`。`Loss` enum に存在しない
    /// 組み合わせ）し、optimizer を使わずホスト側で `p - lr * g` を
    /// 計算する手書き SGD で更新するフック。(a) 独立に組んだ手動ループ
    /// （facade 公開 `DataLoader` を直接使い、同じフック関数を呼ぶ）と
    /// `History.loss`・パラメータが bit 完全一致し、(b) 固定シードで
    /// 決定的に収束する（epoch 0 の loss より最終 epoch の loss が
    /// 小さい）ことを確認する。
    fn custom_loss_host_sgd_hook(
        m: &Sequential,
        x_batch: &Tensor<f32>,
        y_batch: &Tensor<f32>,
        _opt: &mut OptimizerState,
    ) -> StepOutcome {
        const LR: f32 = 0.02;
        let tape = crate::tape();
        let bound = m.bind(&tape);
        let x_var = tape.var(x_batch);
        let pred = bound.forward_with_precision(&tape, &x_var, None)?;
        let target_var = tape.var(y_batch);
        let loss_var = pred.mse_loss_with(&target_var, Reduction::Sum)?;
        let loss_scalar = loss_var.to_tensor().get(&[]).ok_or_else(|| {
            AutodiffError::InvalidArgument("test fixture: loss の shape が [] ではない".to_string())
        })?;
        let grads = tape.backward(&loss_var)?;
        let grad_refs = bound.trainable_grads(&grads)?;
        let param_refs = m.trainable_parameters();
        let mut updated: Vec<Tensor<f32>> = Vec::new();
        for (p, g) in param_refs.iter().zip(grad_refs.iter()) {
            let ps = p.host_slice();
            let gs = g.host_slice();
            let new_vals: Vec<f32> = ps
                .iter()
                .zip(gs.iter())
                .map(|(pv, gv)| pv - LR * gv)
                .collect();
            updated.push(Tensor::new(new_vals, p.shape())?);
        }
        Ok((loss_scalar, Some(updated)))
    }

    #[test]
    fn custom_loss_and_host_sgd_converges() {
        const N: usize = 16;
        const BATCH: usize = 4;
        const EPOCHS: usize = 30;
        let (x, y) = gen_regression_data(0x7570_CCCC, N);

        let mut hook_model = build_model();
        hook_model
            .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
            .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));
        let hook_history = hook_model
            .fit_custom_step_for_test(
                &x,
                &y,
                FitConfig::new(EPOCHS, BATCH),
                None,
                &mut [],
                &[],
                &mut custom_loss_host_sgd_hook,
            )
            .unwrap_or_else(|e| panic!("フック fit に失敗: {e}"));

        // (a) 独立に組んだ手動ループ（facade 公開 `DataLoader` を直接
        // 使う。同じ初期重み・同じフック関数・同じバッチ順序〈shuffle
        // なし〉のため bit 完全一致するはず）。
        let mut manual_model = build_model();
        let mut dummy_opt = OptimizerState::new(Optimizer::Sgd(SgdConfig::new(0.1)))
            .unwrap_or_else(|e| panic!("test fixture: OptimizerState 構築に失敗: {e}"));
        let x_dataset = TensorDataset::new(x.clone())
            .unwrap_or_else(|e| panic!("test fixture: x データセット構築に失敗: {e}"));
        let y_dataset = TensorDataset::new(y.clone())
            .unwrap_or_else(|e| panic!("test fixture: y データセット構築に失敗: {e}"));
        let loader = DataLoader::new((x_dataset, y_dataset), DataLoaderConfig::new(BATCH))
            .unwrap_or_else(|e| panic!("test fixture: DataLoader 構築に失敗: {e}"));
        let mut manual_loss: Vec<f32> = Vec::new();
        for _ in 0..EPOCHS {
            let mut weighted_sum = 0.0f64;
            let mut count = 0usize;
            for batch in &loader {
                let (xb, yb) =
                    batch.unwrap_or_else(|e| panic!("test fixture: バッチ取得に失敗: {e}"));
                let n_batch = xb.shape().first().copied().unwrap_or(0);
                let (loss_scalar, updated) =
                    custom_loss_host_sgd_hook(&manual_model, &xb, &yb, &mut dummy_opt)
                        .unwrap_or_else(|e| panic!("手動ループのフック呼び出しに失敗: {e}"));
                weighted_sum += loss_scalar as f64 * n_batch as f64;
                count += n_batch;
                if let Some(u) = updated {
                    manual_model
                        .apply_parameters(u)
                        .unwrap_or_else(|e| panic!("手動ループの apply_parameters に失敗: {e}"));
                }
            }
            manual_loss.push((weighted_sum / count as f64) as f32);
        }

        assert_eq!(
            hook_history.loss, manual_loss,
            "フック fit の History.loss が独立な手動ループと bit 一致しない"
        );
        assert!(
            params_bit_exact(
                &manual_model.trainable_parameters(),
                &hook_model.trainable_parameters()
            ),
            "フック fit の学習後パラメータが独立な手動ループと bit 一致しない"
        );

        // (b) 決定的な収束確認（固定シードのためフレーキーにならない）。
        let first = *hook_history
            .loss
            .first()
            .unwrap_or_else(|| panic!("test fixture: history.loss が空"));
        let last = *hook_history
            .loss
            .last()
            .unwrap_or_else(|| panic!("test fixture: history.loss が空"));
        assert!(
            last < first,
            "自作損失・ホスト SGD の学習ループが収束していない（first={first}, last={last}）"
        );
    }

    /// T3: 常に `(loss, None)` を返すフックはパラメータを一切更新せず
    /// （bit 完全一致で不変）、loss は記録される。
    #[test]
    fn custom_step_none_update_leaves_params_unchanged() {
        const N: usize = 6;
        const BATCH: usize = 2;
        const EPOCHS: usize = 2;
        let (x, y) = gen_regression_data(0x7570_DDDD, N);

        let mut model = build_model();
        model
            .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
            .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));
        let before: Vec<Tensor<f32>> = model.trainable_parameters().into_iter().cloned().collect();

        let mut hook = |m: &Sequential,
                        x_batch: &Tensor<f32>,
                        y_batch: &Tensor<f32>,
                        _opt: &mut OptimizerState|
         -> StepOutcome {
            let tape = crate::tape();
            let bound = m.bind(&tape);
            let x_var = tape.var(x_batch);
            let pred = bound.forward_with_precision(&tape, &x_var, None)?;
            let target_var = tape.var(y_batch);
            let loss_var = pred.mse_loss_with(&target_var, Reduction::Mean)?;
            let loss_scalar = loss_var.to_tensor().get(&[]).ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "test fixture: loss の shape が [] ではない".to_string(),
                )
            })?;
            Ok((loss_scalar, None))
        };

        let history = model
            .fit_custom_step_for_test(
                &x,
                &y,
                FitConfig::new(EPOCHS, BATCH),
                None,
                &mut [],
                &[],
                &mut hook,
            )
            .unwrap_or_else(|e| panic!("フック fit に失敗: {e}"));

        assert_eq!(history.loss.len(), EPOCHS);
        let before_refs: Vec<&Tensor<f32>> = before.iter().collect();
        assert!(
            params_bit_exact(&before_refs, &model.trainable_parameters()),
            "(loss, None) を返すフックでパラメータが変化してしまった"
        );
    }

    /// T4: フックが `Err` を返すと fit が `Err` になり、その後も
    /// `is_compiled()` は true・training モードは呼び出し前の値に戻る。
    #[test]
    fn custom_step_error_propagates_and_keeps_compiled() {
        const N: usize = 4;
        const BATCH: usize = 2;
        let (x, y) = gen_regression_data(0x7570_EEEE, N);

        let mut model = build_model();
        model
            .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
            .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));
        let prev_training = model.training();

        let mut hook = |_m: &Sequential,
                        _x_batch: &Tensor<f32>,
                        _y_batch: &Tensor<f32>,
                        _opt: &mut OptimizerState|
         -> StepOutcome {
            Err(AutodiffError::InvalidArgument(
                "test fixture: フックの意図的な失敗".to_string(),
            ))
        };

        let err = model
            .fit_custom_step_for_test(
                &x,
                &y,
                FitConfig::new(1, BATCH),
                None,
                &mut [],
                &[],
                &mut hook,
            )
            .expect_err("フックが Err を返したので fit も Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        assert!(
            model.is_compiled(),
            "エラー後も compiled 状態が維持されるはず"
        );
        assert_eq!(
            model.training(),
            prev_training,
            "エラー後も training モードが呼び出し前の値へ復元されるはず"
        );
    }

    /// T5a: カスタム学習 step フックは AMP（`compile_with_amp`）と
    /// 併用できない（fail-closed）。
    #[test]
    fn custom_step_rejected_with_amp() {
        const N: usize = 4;
        const BATCH: usize = 2;
        let (x, y) = gen_regression_data(0x7570_1AAA, N);

        let mut model = build_model();
        model
            .compile_with_amp(
                Optimizer::Sgd(SgdConfig::new(0.1)),
                Loss::Mse,
                AmpConfig::new(AmpDType::F16),
            )
            .unwrap_or_else(|e| panic!("test fixture: compile_with_amp に失敗: {e}"));
        let before: Vec<Tensor<f32>> = model.trainable_parameters().into_iter().cloned().collect();

        let mut hook = default_step_hook;
        let err = model
            .fit_custom_step_for_test(
                &x,
                &y,
                FitConfig::new(1, BATCH),
                None,
                &mut [],
                &[],
                &mut hook,
            )
            .expect_err("カスタム学習 step フックと AMP の併用は Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        assert!(
            model.is_compiled(),
            "エラー後も compiled 状態が維持されるはず"
        );
        let before_refs: Vec<&Tensor<f32>> = before.iter().collect();
        assert!(
            params_bit_exact(&before_refs, &model.trainable_parameters()),
            "拒否されたはずのフックでパラメータが変化してしまった"
        );
    }

    /// T5b: カスタム学習 step フックは勾配累積（`accumulate_steps >
    /// 1`）と併用できない（fail-closed）。
    #[test]
    fn custom_step_rejected_with_accumulate_steps_gt_one() {
        const N: usize = 4;
        const BATCH: usize = 2;
        let (x, y) = gen_regression_data(0x7570_1BBB, N);

        let mut model = build_model();
        model
            .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
            .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));
        let before: Vec<Tensor<f32>> = model.trainable_parameters().into_iter().cloned().collect();

        let config = FitConfig::new(1, BATCH).accumulate_steps(2);
        let mut hook = default_step_hook;
        let err = model
            .fit_custom_step_for_test(&x, &y, config, None, &mut [], &[], &mut hook)
            .expect_err("カスタム学習 step フックと accumulate_steps > 1 の併用は Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        assert!(
            model.is_compiled(),
            "エラー後も compiled 状態が維持されるはず"
        );
        let before_refs: Vec<&Tensor<f32>> = before.iter().collect();
        assert!(
            params_bit_exact(&before_refs, &model.trainable_parameters()),
            "拒否されたはずのフックでパラメータが変化してしまった"
        );
    }

    /// T5c（イシュー #2172）: カスタム学習 step フックは
    /// `Optimizer::Lbfgs` とも併用できない（fail-closed。フックは
    /// `&Sequential`〈不変参照〉しか受け取らず、trial パラメータを
    /// 書き込む L-BFGS closure をフック内から駆動できないため）。
    #[test]
    fn custom_step_rejected_with_lbfgs() {
        const N: usize = 4;
        const BATCH: usize = 2;
        let (x, y) = gen_regression_data(0x7570_1CCC, N);

        let mut model = build_model();
        model
            .compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse)
            .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));
        let before: Vec<Tensor<f32>> = model.trainable_parameters().into_iter().cloned().collect();

        let mut hook = default_step_hook;
        let err = model
            .fit_custom_step_for_test(
                &x,
                &y,
                FitConfig::new(1, BATCH),
                None,
                &mut [],
                &[],
                &mut hook,
            )
            .expect_err("カスタム学習 step フックと Optimizer::Lbfgs の併用は Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        assert!(
            model.is_compiled(),
            "エラー後も compiled 状態が維持されるはず"
        );
        let before_refs: Vec<&Tensor<f32>> = before.iter().collect();
        assert!(
            params_bit_exact(&before_refs, &model.trainable_parameters()),
            "拒否されたはずのフックでパラメータが変化してしまった"
        );
    }

    /// T6: validation・callbacks（`EarlyStopping`）併用時も
    /// `val_loss` が埋まり、callbacks が動く（学習側の演算列はフックが
    /// 決めるが、validation フェーズは既存の `run_evaluate` 経路の
    /// ままであることの確認）。
    #[test]
    fn custom_step_with_validation_and_callbacks() {
        const N: usize = 8;
        const BATCH: usize = 2;
        const EPOCHS: usize = 3;
        let (x, y) = gen_regression_data(0x7570_2CCC, N);
        let (x_val, y_val) = gen_regression_data(0x7570_2DDD, 4);

        let mut model = build_model();
        model
            .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
            .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));

        let mut callbacks = [Callback::EarlyStopping(EarlyStopping::new(100))];
        let mut hook = default_step_hook;
        let history = model
            .fit_custom_step_for_test(
                &x,
                &y,
                FitConfig::new(EPOCHS, BATCH),
                Some((&x_val, &y_val)),
                &mut callbacks,
                &[],
                &mut hook,
            )
            .unwrap_or_else(|e| panic!("フック fit（validation・callbacks 併用）に失敗: {e}"));

        assert_eq!(
            history.val_loss.len(),
            EPOCHS,
            "validation 併用時は val_loss が epoch 数分埋まるはず"
        );
        assert_eq!(history.loss.len(), EPOCHS);
    }

    /// T7: フックが shape の違うパラメータを返すと、`apply_parameters`
    /// 経由で `Err` になる。
    #[test]
    fn custom_step_shape_mismatch_update_is_rejected() {
        const N: usize = 4;
        const BATCH: usize = 2;
        let (x, y) = gen_regression_data(0x7570_3EEE, N);

        let mut model = build_model();
        model
            .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
            .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));

        let mut hook = |m: &Sequential,
                        x_batch: &Tensor<f32>,
                        y_batch: &Tensor<f32>,
                        _opt: &mut OptimizerState|
         -> StepOutcome {
            let tape = crate::tape();
            let bound = m.bind(&tape);
            let x_var = tape.var(x_batch);
            let pred = bound.forward_with_precision(&tape, &x_var, None)?;
            let target_var = tape.var(y_batch);
            let loss_var = pred.mse_loss_with(&target_var, Reduction::Mean)?;
            let loss_scalar = loss_var.to_tensor().get(&[]).ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "test fixture: loss の shape が [] ではない".to_string(),
                )
            })?;
            // 意図的に shape の違うパラメータ集合を返す（1 要素だけ、
            // 元の 1 個目のパラメータと異なる shape）。
            let bogus = Tensor::new(vec![0.0f32; 1], &[1])?;
            Ok((loss_scalar, Some(vec![bogus])))
        };

        let err = model
            .fit_custom_step_for_test(
                &x,
                &y,
                FitConfig::new(1, BATCH),
                None,
                &mut [],
                &[],
                &mut hook,
            )
            .expect_err("shape の違うパラメータ更新は Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }
}

/// `Sequential::run_fit` の `OptimizerState::Lbfgs` 分岐（`lbfgs_batch_step`）
/// の失敗時復元契約を固定する単体テスト（codex-review 指摘・PR #2319:
/// closure を 1 回以上評価した後に失敗する経路の復元・`compiled` 状態
/// 維持・再度 fit/evaluate 可能であることが未検証だった）。
///
/// `lbfgs_batch_step` は crate 内部専用の非公開関数（`pub` にできない
/// 契約はない——単に facade 公開面に含めない設計。`compat::training::
/// Optimizer::Lbfgs` doc 参照）のため、外部統合テストクレート
/// （`crates/facade/tests/*`）からは到達できず、本ファイル内の
/// `#[cfg(test)]` から `Sequential::fit`（公開 API）経由で間接的に
/// 検証する。
#[cfg(test)]
mod lbfgs_fit_failure_tests {
    use super::*;

    const D_IN: usize = 3;
    const D_HIDDEN: usize = 4;
    const D_OUT: usize = 2;
    const N: usize = 4;
    const SEED_L1: u64 = 0x1BF6_1111;
    const SEED_L2: u64 = 0x1BF6_2222;

    fn build_model() -> Sequential {
        Sequential::new()
            .add_linear(D_IN, D_HIDDEN, SEED_L1)
            .unwrap_or_else(|e| panic!("test fixture: 層 1 の構築に失敗: {e}"))
            .add_relu()
            .add_linear(D_HIDDEN, D_OUT, SEED_L2)
            .unwrap_or_else(|e| panic!("test fixture: 層 2 の構築に失敗: {e}"))
    }

    /// `tests/compat_sequential_accumulate.rs` の `deterministic_fill` と同型の局所実装
    /// （splitmix64。値域 `(-0.5, 0.5)`）。
    fn deterministic_fill(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
            })
            .collect()
    }

    fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
        let x = deterministic_fill(seed, N * D_IN);
        let y = deterministic_fill(seed ^ 0x5555_5555_5555_5555, N * D_OUT);
        (
            Tensor::new(x, &[N, D_IN])
                .unwrap_or_else(|e| panic!("test fixture: x の shape 構築に失敗: {e}")),
            Tensor::new(y, &[N, D_OUT])
                .unwrap_or_else(|e| panic!("test fixture: y の shape 構築に失敗: {e}")),
        )
    }

    fn params_bit_exact(a: &[&Tensor<f32>], b: &[&Tensor<f32>]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| {
            let xd = x.contiguous();
            let yd = y.contiguous();
            let xs = xd.as_slice().expect("test fixture: contiguous 化済み");
            let ys = yd.as_slice().expect("test fixture: contiguous 化済み");
            xs.len() == ys.len()
                && xs
                    .iter()
                    .zip(ys.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        })
    }

    /// `docs/autodiff-lbfgs-decision.md` §8「`run_fit` のバッチ処理」節が
    /// 予定する復元契約: L-BFGS の 1 outer step 内で closure を 2 回以上
    /// 評価した後（＝ 1 回目の評価は成功し、内部反復の固定ステップ更新後
    /// の 2 回目以降の評価で失敗する）に `Lbfgs::try_step_closure` が
    /// `Err` を返した場合、`lbfgs_batch_step` は
    /// （1）`fit` 呼び出しが `Err` を返す・
    /// （2）`Sequential::trainable_parameters()` が `fit` 呼び出し前
    ///    （この step 開始前と同じ。1 epoch・フルバッチのため両者は
    ///    一致する）と bit 完全一致で復元される・
    /// （3）`Sequential::is_compiled()` が維持される・
    /// （4）同じモデルで再度 `fit`／`evaluate` が呼び出し可能である
    /// ことを満たす。
    ///
    /// **決定的な失敗誘発の仕組み**: `LbfgsConfig::line_search`
    /// （既定 `LbfgsLineSearch::None`。固定ステップ）でも、固定ステップの
    /// 反復は「`x += t·d` 更新直後の反復末尾で closure を再評価し次反復の
    /// `loss`／`flat_grad` を得る」契約（`lbfgs.rs` モジュール doc）を
    /// 持つため `max_iter >= 2` で closure が複数回呼ばれる。極端に大きい
    /// `lr`（`1e30`）を与えると、1 回目の評価（元のパラメータ・有限）で
    /// 得た勾配方向へ 1 回目のステップを踏んだ時点でパラメータが桁違いに
    /// 巨大化する（それ自体はまだ有限）。この巨大パラメータで forward
    /// した 2 回目の closure 評価は、MSE loss の二乗項が `f32::MAX`
    /// （約 3.4e38）を超えて `inf` になる（`(1e30)^2 = 1e60`）ため
    /// 非有限となり、`Lbfgs` 内部の closure 戻り値検証（`lbfgs.rs`
    /// モジュール doc「closure 戻り値の検証」節）が `InvalidArgument` を
    /// 返す。パラメータ自体（`x += t·d` 直後の値）は有限のままのため、
    /// `Lbfgs::try_step_closure` はこの巨大パラメータで closure を実際に
    /// 呼び出す（`lbfgs_batch_step` の closure が
    /// `model.apply_parameters(trial)` を実行してから forward する）ため、
    /// 「trial 書き込み後の失敗」という復元経路を確実に踏む。
    #[test]
    fn lbfgs_fit_restores_params_and_keeps_compiled_after_multi_eval_failure() {
        let (x, y) = gen_regression_data(0x1BF6_D474);
        let mut model = build_model();
        model
            .compile(
                Optimizer::Lbfgs(LbfgsConfig {
                    lr: 1e30,
                    max_iter: 2,
                    ..LbfgsConfig::default()
                }),
                Loss::Mse,
            )
            .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));

        let snapshot_before: Vec<Tensor<f32>> =
            model.trainable_parameters().into_iter().cloned().collect();
        let eval_before = model
            .evaluate(&x, &y, N)
            .unwrap_or_else(|e| panic!("test fixture: 事前 evaluate に失敗: {e}"));

        let result = model.fit(&x, &y, FitConfig::new(1, N));
        let err = result.expect_err(
            "極端に大きい lr による非有限 loss は Lbfgs::try_step_closure を \
             Err にするはず（テスト前提が崩れている場合はここで失敗する）",
        );
        assert!(
            matches!(err, AutodiffError::InvalidArgument(_)),
            "非有限 loss の検出は InvalidArgument のはず: {err:?}"
        );
        // テスト前提の固定: 失敗が 1 回目の closure 評価（元パラメータ・
        // 有限）ではなく、固定ステップ更新後の 2 回目以降の評価
        // （非有限 loss）で起きていることをエラーメッセージで確認する
        // （`Lbfgs::try_step_closure` の非有限検出メッセージは
        // `lbfgs.rs` 側で "closure returned non-finite loss" を含む）。
        let msg = err.to_string();
        assert!(
            msg.contains("non-finite loss"),
            "想定した失敗経路（2 回目以降の closure 評価での非有限 loss \
             検出）ではない可能性がある: {msg}"
        );

        // (2) パラメータは fit 呼び出し前と bit 完全一致で復元される
        // （trial 書き込み〈`model.apply_parameters(trial)`〉が発生した
        // 後の失敗でも、`lbfgs_batch_step` の Err 経路が
        // `model.apply_parameters(snapshot)` で元へ戻すため）。
        let snapshot_after = model.trainable_parameters();
        assert_eq!(snapshot_before.len(), snapshot_after.len());
        let before_refs: Vec<&Tensor<f32>> = snapshot_before.iter().collect();
        assert!(
            params_bit_exact(&before_refs, &snapshot_after),
            "L-BFGS closure 失敗後にパラメータが fit 呼び出し前の snapshot \
             から変化している"
        );

        // (3) compiled 状態が維持される。
        assert!(
            model.is_compiled(),
            "L-BFGS closure 失敗後も is_compiled() が true のまま維持される \
             はず"
        );

        // (4) 同じモデルで再度 evaluate が呼べ、パラメータ復元により
        // fit 呼び出し前と同じ損失が得られる（bit 完全一致は要求しない
        // ——evaluate 自体は決定的だが余分な許容誤差を持ち込まないため
        // 数値比較で十分）。
        let eval_after = model
            .evaluate(&x, &y, N)
            .unwrap_or_else(|e| panic!("L-BFGS closure 失敗後の evaluate に失敗: {e}"));
        assert_eq!(
            eval_before, eval_after,
            "パラメータが復元されているなら fit 前後で evaluate の損失は \
             完全一致するはず"
        );

        // (4) 続けて正常な学習率で fit を再実行できる（compiled 状態が
        // 壊れていないことの追加確認。`Lbfgs` 内部状態〈n_iter 等〉は
        // 失敗時不変契約により初回呼び出し前のまま残っているため、
        // 新しい outer step として正常に走る）。
        model
            .compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse)
            .unwrap_or_else(|e| panic!("test fixture: 再 compile に失敗: {e}"));
        let history = model
            .fit(&x, &y, FitConfig::new(1, N))
            .unwrap_or_else(|e| panic!("再 compile 後の fit に失敗: {e}"));
        assert_eq!(history.loss.len(), 1);
        assert!(history.loss[0].is_finite());
    }
}

/// イシュー #2530・PR #2724 レビュー指摘: `batch_first=false` の MultiheadAttention を含む
/// モデルは、DataLoader の第 0 軸分割と不整合になるため `fit`／`evaluate` が型付きで拒否する。
#[cfg(test)]
mod seq_first_mha_fit_rejection_tests {
    use super::super::MultiheadAttentionConfig;
    use super::*;

    fn data() -> (Tensor<f32>, Tensor<f32>) {
        (
            Tensor::new(vec![0.1; 4 * 2 * 8], &[4, 2, 8]).unwrap(),
            Tensor::new(vec![0.2; 4 * 2 * 8], &[4, 2, 8]).unwrap(),
        )
    }

    fn compiled(batch_first: bool) -> Sequential {
        let cfg = MultiheadAttentionConfig::new(8, 2).with_batch_first(batch_first);
        let mut m = Sequential::new()
            .add_multihead_attention_with_config(cfg, 1)
            .unwrap();
        m.compile(Optimizer::Sgd(SgdConfig::new(0.01)), Loss::Mse)
            .unwrap();
        m
    }

    #[test]
    fn fit_and_evaluate_reject_batch_first_false() {
        let (x, y) = data();
        let mut m = compiled(false);
        let err = m.fit(&x, &y, FitConfig::new(1, 2)).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        assert!(m.is_compiled(), "拒否後も compiled を保持する");
        let err = m.evaluate(&x, &y, 2).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn batch_first_true_still_accepted() {
        let (x, y) = data();
        let mut m = compiled(true);
        m.fit(&x, &y, FitConfig::new(1, 2)).unwrap();
        m.evaluate(&x, &y, 2).unwrap();
    }
}
