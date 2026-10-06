//! composition root（TASK-9.3・イシュー #410・spec 確定は
//! fandhe-ai-spec#52／spec PR #53。`docs/spec/05-tasks.md:315`）。
//!
//! `facade` は 2 つの責務を担う（spec 確定内容）。
//!
//! 1. **composition root（本クレート・本イシューで実装）**: [`Device`]
//!    識別子を受け取り、対応する具体 `BackendOps` 実装
//!    （`fandhe_ai_backend_cpu::CpuBackendOps`／`fandhe_ai_backend_cuda::CudaBackendOps`／
//!    `fandhe_ai_backend_metal::MetalBackendOps`）を構築して `fandhe_ai_autodiff::Tape` へ
//!    結線する。この結線ロジックを持つのは本クレートのみであり、
//!    `tensor-core`／`autodiff`／`backend-*` は互いに他バックエンドを
//!    直接参照しない構造的境界（REQ-9・`docs/fusion-graph-design.md`
//!    §3.4「`autodiff` は具体クレートへの依存を一切持たない」）を、
//!    上位でここに一本化する。
//! 2. **compat 公開面**（[`compat::array`]／[`compat::Sequential`]）:
//!    TASK-9.4（イシュー #411）で `fandhe_ai_autodiff::compat` から本クレートへ
//!    移設し実装済み。サポート境界の明文化（`facade` が唯一のサポート
//!    対象公開 API 面であり `tensor-core`／`autodiff`／`backend-*` は
//!    内部クレート）は `docs/compat-api-scope.md` を参照。
//!
//! 3. **optim 公開面**（[`optim`]。イシュー #961・親 #960。Adam〈coupled
//!    L2 weight decay〉は #1742・RMSprop／Adagrad は #1743・親 #1610・
//!    LAMB は #1744）: SGD・AdamW・Adam・RMSprop・Adagrad・LAMB・
//!    gradient clipping・LR スケジューラを `fandhe_ai::optim` の単一入口へ再エクスポートする。
//!    値型・純関数のみのため REQ-12 と矛盾しない（詳細は [`optim`]
//!    モジュール doc）。
//!
//! 4. **data 公開面**（[`data`]。イシュー #1615・親 #1602）: map-style
//!    データセット（`Dataset`／`TensorDataset`）とミニバッチ供給
//!    （`DataLoader`／`DataLoaderConfig`）を `fandhe_ai::data` の単一
//!    入口へ再エクスポートする。ホスト側だけで完結し `Op`／
//!    `BackendOps`／VJP を経由しないため REQ-12 と矛盾しない（詳細は
//!    [`data`] モジュール doc）。
//!
//! 5. **nn 公開面**（[`nn::rnn`]。イシュー #1955）: RNN／LSTM／GRU の
//!    Sequence レベル API（`Rnn`／`Lstm`／`Gru`。実体は
//!    `fandhe_ai_autodiff::nn`。#1647）を `fandhe_ai::nn::rnn` の単一
//!    入口へ純再エクスポートする。`forward_seq` が生 `Tape` を引数に
//!    取るため facade 利用者からは直接呼べず、`impl Tape` に追加した
//!    `rnn_forward_seq`／`lstm_forward_seq`／`gru_forward_seq`（`&self.0`
//!    を渡すだけの薄い委譲。`step_device_param_store` 等と同型）が
//!    入口となる。イシュー #2535 で多層・双方向・層間 dropout 版
//!    （`RnnConfig`・`StackedRnn`／`StackedLstm`／`StackedGru`・戻り値型 2 型）
//!    も同モジュールへ再エクスポートし、`stacked_rnn_forward_seq`／
//!    `stacked_lstm_forward_seq`／`stacked_gru_forward_seq` を追加した。
//!    値型の再エクスポート＋薄い委譲のみで任意
//!    `BackendOps` 注入経路を新設しないため REQ-12 と矛盾しない
//!    （詳細は [`nn::rnn`] モジュール doc）。同じく `nn` 配下に、PyTorch
//!    `torch.nn.init.*` 相当の初期化関数 9 個と補助 4 名を [`nn::init`] へ
//!    純再エクスポートする（イシュー #2504。グローバル RNG に従う）。
//!
//! 6. **model 公開面**（[`model`]。イシュー #2087・親 #2082）:
//!    ホームディレクトリ配下のキャッシュディレクトリを基盤とする
//!    ローカル限定モデルレジストリ [`model::ModelRegistry`] を提供
//!    する。ホスト側のパス管理と [`interop::safetensors`] への委譲
//!    のみで完結し `BackendOps` 注入経路を新設しないため REQ-12 と
//!    矛盾しない（詳細は [`model`] モジュール doc・`docs/facade-
//!    model-registry-decision.md` 参照）。
//!
//! # 公開面の設計（REQ-12: 任意 `BackendOps` 注入の公開 API を設けない）
//!
//! 利用者向けに公開するのは [`Device`] 識別子を受け取る 2 関数
//! （[`tape`]・[`tape_for`]）と、それに必要な最小限の型再エクスポート
//! （[`Device`]・[`BackendError`]）のみである。`fandhe_ai_autodiff::Tape`・
//! `fandhe_ai_tensor_core::BackendOps` は本クレートから再エクスポートしない
//! （`fandhe_ai::Tape::new_with_ops(ops)` という経路がサポート面に露出すると
//! 「任意 `BackendOps` 実装を注入できる公開 API を設けない」（REQ-12）と
//! 矛盾するため）。この制約は `tests/api_surface.rs` がソース走査で
//! 機械的に固定する。
//!
//! **`Tape`（composition root が構築する値）の扱い（codex-review PR #424
//! P1 是正）**: [`tape`]・[`tape_for`] の戻り値は `fandhe_ai_autodiff::Tape` を
//! そのまま返すのではなく、本クレート所有の newtype [`Tape`] でラップする。
//! `fandhe_ai_autodiff::Tape` を素通しすると `Tape::new_with_ops` という
//! `BackendOps` 注入経路が facade 型として到達可能になり REQ-12 と矛盾する
//! ため、[`Tape`] は [`Tape::var`]／[`Tape::backward`] の 2 メソッドのみを
//! 再委譲する（構築は [`tape`]／[`tape_for`] 経由のみで、`fandhe_ai_autodiff::Tape`
//! の任意構築経路は到達不能なまま）。[`compat::Sequential::forward`]／
//! [`compat::Sequential::bind`] もこの [`Tape`] 型を引数に取る（旧実装は
//! `fandhe_ai_autodiff::Tape` を直に引数へ取っており、内部クレートの型が facade の
//! 公開シグネチャへ直接露出していた。codex-review 指摘）。
//!
//! [`TapeRef`]（#2394。[`nn::Module::forward`] の第 1 引数）は `var`／`var_from`／`var_no_grad` のみを持つ借用ハンドルで、
//! [`Tape`] の公開メソッドは変えない。crate 外の入口は `From<&Tape>` のみ。
//!
//! **`Var`／`Gradients`／`AutodiffError`／`LinearVars`（`autodiff` 由来）・
//! `Tensor`（`tensor_core` 由来）の扱い**: これらは `BackendOps` 注入の
//! 迂回経路を持たない値型・エラー型であるため、`facade` の正式な公開契約
//! として本クレートから再エクスポートする（下記 `pub use`）。
//! `compat::{array, Sequential}` の公開シグネチャはこの再エクスポート
//! パス（`crate::{AutodiffError, Tensor}` 等）を使う。
//!
//! # `Device::Cuda(_)`／`Device::Metal` の構築規則
//!
//! spec は本規則を「TASK-9.3 実装時にユーザー承認を得て確定」とする
//! 未決事項として残しているため、以下の最小・自明な規則を採用した
//! （イシュー #410 実装計画 §2-2。PR 本文でユーザー確認を仰ぐ）。
//!
//! - `Device::Cpu` → `CpuBackendOps::new()`（常に利用可能なため検証不要）。
//! - `Device::Cuda(ordinal)` → `CudaDeviceProvider::select` で存在検証
//!   （driver 不在・範囲外 ordinal は [`BackendError`] を返す fail-fast）
//!   したうえで `CudaBackendOps::new(ordinal)` を構築する。イシュー #929:
//!   `CudaBackendOps` の各演算メソッドは `backend-cuda` 側のプロセス内
//!   キャッシュ（`crate::context_cache`。`ordinal` キー）を経由するため、
//!   同一プロセス内で 2 回目以降に `tape_for(Device::Cuda(_))` を呼んでも
//!   `CudaContext` 生成・NVRTC コンパイルは再実行されない。`resolve_ops`
//!   自体（この関数）は毎回新しい `CudaBackendOps`／`Tape` を構築する
//!   軽量な値であり、重い初期化コストはバックエンド側キャッシュが吸収する。
//! - `Device::Metal`（`cfg(target_os = "macos")`）→ `MetalDeviceProvider::select`
//!   で検証したうえで `MetalBackendOps::new()` を構築する。
//!
//! 既定デバイスの自動選択（GPU フォールバック等）・デバイス列挙の集約入口
//! （`Device::available()` 相当の facade 結線）は本イシューのスコープ外
//! （`docs/public-api-design.md` §4.1 の未決事項を尊重。
//! out-of-scope-tracking.md 対象）。

use fandhe_ai_tensor_core::device::select_from;
use fandhe_ai_tensor_core::{BackendOps, DeviceProvider};

/// numpy/Keras 慣習の互換 API 層（compat 公開面。TASK-9.4・#411）。
/// [`compat::array`]・[`compat::Sequential`] を提供する（詳細はモジュール
/// doc・`docs/compat-api-scope.md` 参照）。
pub mod compat;

/// optimizer 公開面（イシュー #961・親 #960。Adam は #1742・RMSprop／
/// Adagrad は #1743・親 #1610・LAMB は #1744）。SGD・AdamW・Adam・RMSprop・
/// Adagrad・LAMB・gradient clipping・LR スケジューラを再エクスポートする（詳細・
/// 適用順序契約はモジュール doc 参照）。
pub mod optim;

/// Dataset／DataLoader 公開面（イシュー #1615・親 #1602）。map-style
/// データセット（[`data::Dataset`]・[`data::TensorDataset`]）とミニ
/// バッチ供給（[`data::DataLoader`]・[`data::DataLoaderConfig`]）を
/// 再エクスポートする（詳細はモジュール doc・`docs/dataset-dataloader-
/// design.md` 参照）。
pub mod data;

/// `nn` 公開面（イシュー #1955）。[`nn::rnn`]
/// （`Rnn`／`Lstm`／`Gru` と多層版 `Stacked*`・`RnnConfig` の Sequence レベル
/// API の純再エクスポート）と、
/// facade 独自の [`nn::Module`]（#2395）を提供する。`forward_seq` の呼び出しには [`Tape::rnn_forward_seq`]
/// 等（本モジュール自体ではなく `impl Tape` の薄い委譲メソッド）を
/// 使う（詳細は [`nn::rnn`] モジュール doc 参照）。
pub mod nn;

/// 相互運用（interop）公開面の入口（イシュー #2017・#2018・#2019）。
/// [`interop::onnx`]（ONNX import／export。`OnnxModel`／`OnnxValue`／
/// `OnnxError`・`OnnxModel::{from_bytes, from_path, from_path_with_limits, run,
/// to_bytes, to_path}`・`OnnxExportOptions`・`OnnxExternalDataLimits`
/// 〈external data 読み込み予算。#2360〉。export は #2018 で公開済み・roundtrip
/// export ラッパー限定）に加え、[`interop::safetensors`]（safetensors
/// save／load 純再エクスポート。イシュー #2019）を提供する
/// （`docs/facade-onnx-export-exposure-decision.md`・`docs/facade-
/// safetensors-exposure-decision.md`）。
pub mod interop;

/// 6. **model 公開面**（[`model`]。イシュー #2087・親 #2082）: ホーム
///    ディレクトリ配下のキャッシュディレクトリを基盤とするローカル
///    限定モデルレジストリ [`model::ModelRegistry`] を提供する。
///    ホスト側のパス管理と [`interop::safetensors`] への委譲のみで
///    完結し `BackendOps` 注入経路を新設しないため REQ-12 と矛盾しない
///    （詳細は [`model`] モジュール doc・
///    `docs/facade-model-registry-decision.md` 参照）。
pub mod model;

/// バッチ推論（DataLoader 反復 `predict`）・推論フェーズ計測の内部実装
/// （イシュー #2192・親 #2131）。facade 公開は承認待ちのため**非公開**
/// （`pub mod` にしない）。詳細は [`inference::batch`] モジュール doc・
/// `docs/facade-predict-batches-phase-metrics-decision.md` を参照。
mod inference;

/// 非公開。`model` と `compat::model_io`（#2369）が共有する no-follow 葉オープンと
/// サイズ上限（`fs_guard` モジュール doc 参照）。
mod fs_guard;

// 公開面として再エクスポートする型（モジュール冒頭「公開面の設計」参照）。
// `fandhe_ai_autodiff::Tape`（生の型）・`fandhe_ai_tensor_core::BackendOps` は意図的に含めない
// （`Tape::new_with_ops` という BackendOps 注入経路が到達可能になるため。
// REQ-12）。`Var`／`Gradients`／`AutodiffError`／`LinearVars`・`Tensor` は
// 迂回経路を持たない値型・エラー型のため facade の正式な公開契約として
// 再エクスポートする（codex-review PR #424 P1 是正）。
// `tests/api_surface.rs::facade_does_not_reexport_tape_or_backend_ops` は
// `pub use` を行単位（`trimmed.starts_with("pub use")`）で走査するため、
// 複数行に折り返す `pub use fandhe_ai_autodiff::{ ... };` ブロックは
// 開き括弧の行しか検査対象に入らず、ブロック内部に `Tape` が紛れ込んでも
// 検出できない（レビュー指摘対応）。1 文 1 行を維持する。
//
// `SgdConfig` はここ（クレート root）と `crate::optim::SgdConfig`
// （イシュー #961）の 2 経路から再エクスポートされるが、いずれも
// `fandhe_ai_autodiff::optim::SgdConfig` を指す同一型である（再エクスポート
// 経路が 2 つあるだけで型は 1 つ。`Tape::step_device_param_store` の
// 引数型として本経路が既に使われているため、`optim` モジュール新設に
// 伴いこちら側を除去・付け替えることはしない）。
//
// `DeviceParamStore`／`Tape::step_device_param_store`（デバイス常駐更新
// 経路）は REQ-9 の 2026-08-29 追記（正本 spec
// `docs/spec/04-requirements.md:213`。実装リポ #984／#986）で `tape()`系・
// `compat`／`optim` と並ぶ確定入口となった（`docs/compat-api-scope.md` §0）。
pub use fandhe_ai_autodiff::optim::{DeviceParamStore, ResidentLeaf, SgdConfig};
// `Tape::step_device_param_store_adam`／`_adamw`（イシュー #1959）の
// 引数型として使う。`AdamConfig`／`AdamWConfig` 自体は `crate::optim`
// （`optim.rs`）から利用者向けにも再エクスポート済みのため、ここでは
// 型を参照するためだけの `use`（`pub use` ではない）とする。
use fandhe_ai_autodiff::nn::optim::{
    AdagradConfig, AdamConfig, AdamWConfig, GradScaler, LambConfig, RmsPropConfig,
};
pub use fandhe_ai_autodiff::{AutodiffError, Gradients, Var, nn::LinearVars};
// ユーザー定義 forward／backward（カスタム VJP）の trait を facade の
// crate ルートへ再エクスポートする（イシュー #2549。ルート #2499 の一括
// 承認・`docs/autodiff-custom-function-decision.md` §16.1 (a)・
// `docs/compat-api-scope.md` §5 経路 2）。利用者は `Tape::custom` へ
// `Arc<dyn fandhe_ai::CustomFunction>` を渡す。1 文 1 行・別名なし・
// グループ形なしを維持する（`tests/api_surface.rs` が行単位で固定する）。
pub use fandhe_ai_autodiff::CustomFunction;
// `VarHostView`（借用ビュー読み出し API。イシュー #1335）は 1 文 1 行を
// 維持する（`tests/api_surface.rs` が `pub use` を行単位で走査するため。
// 上記コメント「1 文 1 行を維持する」参照）。
pub use fandhe_ai_autodiff::VarHostView;
// 線形代数（イシュー #1621・`docs/autodiff-linalg-design.md`）の多出力
// 戻り値型（`Var::qr`／`Var::svd`）は 1 文 1 行で再エクスポートする
// （上記コメント「1 文 1 行を維持する」と同じ理由）。
pub use fandhe_ai_autodiff::QrVars;
pub use fandhe_ai_autodiff::SvdVars;
// `topk_unique_ops`（イシュー #2153 実装・#2519 公開）の入出力型。
// `Var::topk_with_options`／`unique_with_options`／`unique_consecutive` の
// 引数（`TopkOptions`／`UniqueOptions`）と戻り値（`UniqueOutput`）で、これらが
// 無いと委譲メソッドを呼べない。モジュール `topk_unique_ops` 自体は
// 再エクスポートしない（型だけを autodiff ルート経由で 1 文 1 行で公開する。
// `api_surface.rs::facade_reexports_topk_unique_types_only_in_approved_shape`）。
pub use fandhe_ai_autodiff::{TopkOptions, UniqueOptions, UniqueOutput};
pub use fandhe_ai_tensor_core::{BackendError, Device, PoolStats, Tensor};
// `RngError`（イシュー #1725）: `randint` の戻り値型（`low >= high` の
// 範囲不正を表す）。`Tape`／`BackendOps` を含まない単純なエラー型のため
// 1 行の `pub use` で足りる（`api_surface.rs::facade_does_not_reexport_tape_or_backend_ops`
// は行単位で `Tape`／`BackendOps`／`new_with_ops` を検査するのみで抵触
// しない）。
pub use fandhe_ai_tensor_core::RngError;
// `ShapeError`（イシュー #1725）: `randn`／`rand`（本 PR で新設したトップ
// レベル `pub fn`）の戻り値型（形状不正を表す）。`Tensor::zeros` 等の
// 既存メソッドも同型を返すが、それらは既存の再エクスポート型
// （`Tensor`）のメソッドであるのに対し `randn`／`rand` は本 PR 新設の
// トップレベル関数であり、facade が「唯一のサポートされる公開 API 面」
// である方針（CLAUDE.md）に照らし `RngError` と同様に 1 行の
// `pub use` で再エクスポートする（codex-review 指摘対応。上記
// `RngError` コメントと同じ理由で `api_surface.rs` の走査にも抵触
// しない）。
pub use fandhe_ai_tensor_core::ShapeError;
// `CreationError`（イシュー #1726）: `arange`／`linspace`（本ファイルで
// 新設したトップレベル `pub fn`）の戻り値型（`step` 不正・非有限値を
// 表す）。`RngError`／`ShapeError` と同じ理由で 1 行の `pub use` で
// 再エクスポートする（`api_surface.rs` の走査にも抵触しない）。
pub use fandhe_ai_tensor_core::CreationError;
// `ChecksumReadout`／`GemmChecksum`（イシュー #1339・`Var::matmul_checksum`
// の戻り値・引数型）も 1 文 1 行で再エクスポートする（上記コメント
// 「1 文 1 行を維持する」と同じ理由）。
pub use fandhe_ai_tensor_core::{ChecksumReadout, GemmChecksum};
// 線形代数（イシュー #1621）の値型（`BackendOps::linalg_qr`／
// `linalg_svd`／`linalg_matrix_norm` の入出力）も 1 文 1 行で
// 再エクスポートする（`QrFactors`／`SvdFactors`／`MatrixNormOrd` は
// `Var::qr`／`svd`／`matrix_norm` の呼び出し側からは直接見えないが、
// バックエンド実装を跨いで自作するテスト・診断コードのために公開する。
// 上記コメント「1 文 1 行を維持する」と同じ理由）。
pub use fandhe_ai_tensor_core::MatrixNormOrd;
pub use fandhe_ai_tensor_core::QrFactors;
pub use fandhe_ai_tensor_core::SvdFactors;
// `InterpolateMode`（イシュー #1757・`Var::interpolate` の `mode`
// 引数型）も 1 文 1 行で再エクスポートする（上記コメント「1 文 1 行を
// 維持する」と同じ理由）。`Bilinear { align_corners }` variant
// （イシュー #1762）追加時も新規公開アイテムは発生しない
// （`InterpolateMode` 自体の再エクスポートのみで完結する）。
pub use fandhe_ai_tensor_core::InterpolateMode;
// `GlobalPoolMode`（イシュー #2527。`compat::Sequential::add_global_pool`
// の `mode` 引数型。`#[non_exhaustive]`）も 1 文 1 行で再エクスポートする。
// `GlobalPool` 層型・`AdaptiveMaxPool2d`／`AdaptiveMaxPool1d` 層型の
// 再エクスポートは未承認のため行わない（`AdaptiveMaxGlobalPoolHoldDoctestGuard`
// と `tests/api_surface.rs` の正ガードが固定）。
pub use fandhe_ai_autodiff::nn::GlobalPoolMode;
// `EmbeddingBagMode`（イシュー #2528。`compat::Sequential::add_embedding_bag`・
// `Var::embedding_bag` の `mode` 引数型。`#[non_exhaustive]`）も 1 文 1 行で
// 再エクスポートする。`Dropout2d`／`AlphaDropout`／`EmbeddingBag`／
// `EmbeddingBagVars` 層型の再エクスポートは未承認のため行わない
// （`DropoutEmbeddingBagHoldDoctestGuard` と `tests/api_surface.rs` の正ガードが固定）。
pub use fandhe_ai_autodiff::nn::EmbeddingBagMode;
// `CreateGraphResult`（子テープ方式の高階微分結果型。イシュー #2545。
// `Tape::backward_create_graph` の戻り値型）も 1 文 1 行で再エクスポートする。
// 承認根拠はルート #2499 の一括承認（2026-10-04）と
// `docs/autodiff-higher-order-grad-decision.md` §17.2 の確定形。フィールドは
// 非公開で利用者は構築できない。`TapeRef` 版・子テープ構築ヘルパーは
// §17.3 により公開しない（`tests/api_surface.rs` の正ガードが固定）。
pub use fandhe_ai_autodiff::CreateGraphResult;
// `CastDType`／`CastElement`（イシュー #1750。`Var::cast`／`Tape::
// var_from` の型境界・dtype タグ）も 1 文 1 行で再エクスポートする
// （上記コメント「1 文 1 行を維持する」と同じ理由）。`CastOps`（動的
// ディスパッチ面）は再エクスポートしない——利用者は `Var::cast`／
// `Tape::var_from` 経由で到達し、`CastOps` 自体を直接構築・実装する
// 経路は facade の公開契約に含めない（`docs/tensor-core-cast-design.md`
// 参照）。
pub use fandhe_ai_tensor_core::{CastDType, CastElement};
// `Scalar`／`ScalarDType`／`TypedOps`（イシュー #1939・
// `docs/compat-api-scope.md` §5 経路 2 承認・`docs/backend-dtype-
// dispatch-design.md`）: `Tensor<T>` を直接対象とする dtype 別演算集合
// （`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max` の 8 演算
// 限定）への capability accessor 面を facade へ昇格する。到達経路は
// `Tape::typed_ops_f64`／`_f16`／`_bf16`（本ファイル下部）が返す
// `Option<&dyn TypedOps<T>>` のみで、`Var`／autograd は経由しない
// （勾配は付かない）。`half::f16`／`half::bf16` を名指しする利用者は
// `half`（本 workspace と同一バージョン）へ直接依存する前提とし、
// facade は `half` 自体を再エクスポートしない（承認事項）。上記コメント
// 「1 文 1 行を維持する」と同じ理由で 1 行にまとめる。
pub use fandhe_ai_tensor_core::{Scalar, ScalarDType, TypedOps};

// `TypedOps<T>` の戻り値型シグネチャで `half::f16`／`half::bf16` を
// 名指しするための非公開 `use`（facade 自体は `half` を再エクスポート
// しない。上記 `Scalar`／`ScalarDType`／`TypedOps` コメント参照）。
use fandhe_ai_tensor_core::{bf16, f16};

/// composition root（[`tape`]／[`tape_for`]）が構築する `Tape` の
/// newtype ラッパー（codex-review PR #424 P1 是正）。
///
/// `fandhe_ai_autodiff::Tape` をそのまま公開すると `Tape::new_with_ops(ops)`
/// （任意 `BackendOps` 注入経路）が facade の型として到達可能になり
/// REQ-12「任意 `BackendOps` 実装を注入できる公開 API を設けない」と
/// 矛盾する。本型は内部の `fandhe_ai_autodiff::Tape`（フィールド `0`。`pub(crate)`
/// のためクレート外から直接構築・分解できない）が持つメソッドのうち
/// [`Tape::var`]／[`Tape::backward`] のみを再委譲し、それ以外（特に
/// `new_with_ops`）を到達不能に保つ。構築できるのは [`tape`]／[`tape_for`]
/// のみ（`pub(crate)` タプルフィールドのため crate 外からのフィールド
/// アクセス・構築は不能。`tests/api_surface.rs` が機械的に固定する）。
pub struct Tape(pub(crate) fandhe_ai_autodiff::Tape);

impl std::fmt::Debug for Tape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // 内部の `fandhe_ai_autodiff::Tape` も同じ理由（`ops` が `Debug` 非実装）で
        // `finish_non_exhaustive` を使う（`fandhe_ai_autodiff::tape::Tape` の
        // `Debug` 実装と同じ方針。`crates/autodiff/src/tape.rs` 参照）。
        f.debug_struct("Tape").finish_non_exhaustive()
    }
}

impl Tape {
    /// 入力テンソルをテープ上の葉ノード `Var` として登録する
    /// （`fandhe_ai_autodiff::Tape::var` への委譲）。
    pub fn var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.0.var(tensor)
    }

    /// 非 f32 dtype の入力テンソルを f32 へ変換したうえでテープ上の
    /// 葉ノード `Var` として登録する（[`Self::var`] の dtype 変換版。
    /// イシュー #1750・`fandhe_ai_autodiff::Tape::var_from` への委譲）。
    pub fn var_from<T: CastElement>(&self, tensor: &Tensor<T>) -> Result<Var<'_>, AutodiffError> {
        self.0.var_from(tensor)
    }

    /// 入力テンソルを `requires_grad == false` の葉ノードとして
    /// テープ上へ登録する（イシュー #1748・PyTorch
    /// `requires_grad=False` 相当。`fandhe_ai_autodiff::Tape::
    /// var_no_grad` への委譲。`Var::detach`〈既存 `Var` 再エクスポート
    /// 経由で到達〉と対になる no_grad 側の入口）。
    pub fn var_no_grad(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.0.var_no_grad(tensor)
    }

    /// `loss` から逆伝播し勾配を計算する（`fandhe_ai_autodiff::Tape::backward`
    /// への委譲）。
    pub fn backward(&self, loss: &Var<'_>) -> Result<Gradients, AutodiffError> {
        self.0.backward(loss)
    }

    /// `loss` から逆伝播し、その結果を既存の `into`（同一世代の
    /// `Gradients`）へ加算する（イシュー #1749・`docs/compat-api-scope.md`
    /// §5 経路 2〈#1612 承認〉。`fandhe_ai_autodiff::Tape::backward_
    /// accumulate` への委譲）。PyTorch の複数回 `loss.backward()` に
    /// よる `.grad` 蓄積相当の opt-in API。`retain_graph`（テープは
    /// `reset`／drop まで常時グラフを保持し、同一グラフへの複数回
    /// `backward` は無条件で成功する契約）自体は追加の API なしで
    /// 常時成立している（`fandhe_ai_autodiff::Tape` doc「`retain_graph`
    /// 契約」参照）。
    pub fn backward_accumulate(
        &self,
        loss: &Var<'_>,
        into: &mut Gradients,
    ) -> Result<(), AutodiffError> {
        self.0.backward_accumulate(loss, into)
    }

    /// ノード列を葉プレフィックスまで切り詰め、次のステップで同一
    /// `Tape` を再利用可能にする（イシュー #1048。
    /// `fandhe_ai_autodiff::Tape::reset` への委譲。同メソッドの doc
    /// 「学習ループでの運用」参照）。学習ループ・reuse GEMM が step
    /// ごとに新しい `Tape` を生成・破棄する運用の代替となる。
    pub fn reset(&mut self) {
        self.0.reset();
    }

    /// [`Tape::reset`] 後も保持される葉の個数（`fandhe_ai_autodiff::
    /// Tape::leaf_count` への委譲）。
    pub fn leaf_count(&self) -> usize {
        self.0.leaf_count()
    }

    /// 保持される葉 `index` 番目（登録順）の `Var` を再取得する
    /// （`fandhe_ai_autodiff::Tape::leaf` への委譲。範囲外・非 `Var`
    /// 表現の葉は `None`）。
    pub fn leaf(&self, index: usize) -> Option<Var<'_>> {
        self.0.leaf(index)
    }

    /// [`DeviceParamStore::step`] への委譲入口（イシュー #935）。
    ///
    /// `DeviceParamStore` の状態機械メソッドは `fandhe_ai_autodiff::Tape`
    /// （内部クレートの生の型）を直接引数に取るため、`facade::Tape`
    /// （本型。内部フィールド `0` は `pub(crate)`）の利用者からは直接
    /// 呼べない。本メソッドは `&self.0` を渡すだけの薄い委譲であり、
    /// `BackendOps`／`MemoryOps` を利用者向け公開面へ露出しない
    /// （`crate::lib.rs` モジュール doc「公開面の設計」・REQ-12）。
    pub fn step_device_param_store(
        &self,
        store: &mut DeviceParamStore,
        grads: &Gradients,
        config: &SgdConfig,
    ) -> Result<(), BackendError> {
        store.step(&self.0, grads, config)
    }

    /// [`DeviceParamStore::step_adam`] への委譲入口（イシュー #1959）。
    /// `step_device_param_store`（SGD）と同じ理由の薄い委譲。
    pub fn step_device_param_store_adam(
        &self,
        store: &mut DeviceParamStore,
        grads: &Gradients,
        config: &AdamConfig,
    ) -> Result<(), BackendError> {
        store.step_adam(&self.0, grads, config)
    }

    /// [`DeviceParamStore::step_adamw`] への委譲入口（イシュー #1959）。
    /// `step_device_param_store`（SGD）と同じ理由の薄い委譲。
    pub fn step_device_param_store_adamw(
        &self,
        store: &mut DeviceParamStore,
        grads: &Gradients,
        config: &AdamWConfig,
    ) -> Result<(), BackendError> {
        store.step_adamw(&self.0, grads, config)
    }

    /// [`DeviceParamStore::step_rmsprop`] への委譲入口（イシュー
    /// #2175）。`step_device_param_store`（SGD）と同じ理由の薄い委譲。
    pub fn step_device_param_store_rmsprop(
        &self,
        store: &mut DeviceParamStore,
        grads: &Gradients,
        config: &RmsPropConfig,
    ) -> Result<(), BackendError> {
        store.step_rmsprop(&self.0, grads, config)
    }

    /// [`DeviceParamStore::step_adagrad`] への委譲入口（イシュー
    /// #2175）。`step_device_param_store`（SGD）と同じ理由の薄い委譲。
    pub fn step_device_param_store_adagrad(
        &self,
        store: &mut DeviceParamStore,
        grads: &Gradients,
        config: &AdagradConfig,
    ) -> Result<(), BackendError> {
        store.step_adagrad(&self.0, grads, config)
    }

    /// [`DeviceParamStore::step_lamb`] への委譲入口（イシュー #2175）。
    /// `step_device_param_store`（SGD）と同じ理由の薄い委譲。
    pub fn step_device_param_store_lamb(
        &self,
        store: &mut DeviceParamStore,
        grads: &Gradients,
        config: &LambConfig,
    ) -> Result<(), BackendError> {
        store.step_lamb(&self.0, grads, config)
    }

    /// [`DeviceParamStore::step_amp`] への委譲入口（イシュー #2181。AMP
    /// を常駐 step へ結線する。`docs/device-resident-update-design.md`
    /// 追補）。`step_device_param_store`（SGD）と同じ理由の薄い委譲だが、
    /// 戻り値は `Result<(), BackendError>` ではなく `Result<bool,
    /// AutodiffError>`（内部で [`GradScaler::update`] も呼ぶため
    /// `AutodiffError`。`bool` は「この step が非有限勾配により skip
    /// されたか」）。
    ///
    /// 1 step の使い方: `scaler.scale_loss(&loss)` → `tape.
    /// backward_device_param_store(&scaled, &store)`（`Op::
    /// LinearResident` を含むグラフの場合。含まない場合は素の
    /// [`Tape::backward`]）→ 本メソッド。戻り値 `true` は skip
    /// （どのパラメータも更新されず、forward 登録のみ消費された）ことを
    /// 示し、呼び出し元は次の step へそのまま進めばよい
    /// （`crate::optim::GradScaler` doc「1 step の使い方」参照）。
    pub fn step_device_param_store_amp(
        &self,
        store: &mut DeviceParamStore,
        grads: &Gradients,
        config: &SgdConfig,
        scaler: &mut GradScaler,
    ) -> Result<bool, AutodiffError> {
        store.step_amp(&self.0, grads, config, scaler)
    }

    /// [`DeviceParamStore::step_adam_amp`] への委譲入口（イシュー
    /// #2181）。[`Self::step_device_param_store_amp`] と同じ理由の
    /// 薄い委譲。
    pub fn step_device_param_store_adam_amp(
        &self,
        store: &mut DeviceParamStore,
        grads: &Gradients,
        config: &AdamConfig,
        scaler: &mut GradScaler,
    ) -> Result<bool, AutodiffError> {
        store.step_adam_amp(&self.0, grads, config, scaler)
    }

    /// [`DeviceParamStore::step_adamw_amp`] への委譲入口（イシュー
    /// #2181）。[`Self::step_device_param_store_amp`] と同じ理由の
    /// 薄い委譲。
    pub fn step_device_param_store_adamw_amp(
        &self,
        store: &mut DeviceParamStore,
        grads: &Gradients,
        config: &AdamWConfig,
        scaler: &mut GradScaler,
    ) -> Result<bool, AutodiffError> {
        store.step_adamw_amp(&self.0, grads, config, scaler)
    }

    /// [`DeviceParamStore::backward`] への委譲入口（イシュー #1022）。
    ///
    /// `Sequential::forward_resident`（`compat::sequential`）で forward
    /// したグラフ（`Op::LinearResident` を含む）は、weight のデバイス
    /// バッファを解決できる本メソッドで backward する必要がある——
    /// 素の [`Tape::backward`] はこの解決手段を持たず型付きエラーで
    /// 拒否する（`fandhe_ai_autodiff::optim::device_store::
    /// DeviceParamStore::backward` doc・`docs/device-resident-update-
    /// design.md` §3.3e 参照）。`step_device_param_store` と同じ理由の
    /// 薄い委譲であり、`BackendOps`／`MemoryOps` を利用者向け公開面へ
    /// 露出しない（`crate::lib.rs` モジュール doc「公開面の設計」・
    /// REQ-12）。
    pub fn backward_device_param_store(
        &self,
        loss: &Var<'_>,
        store: &DeviceParamStore,
    ) -> Result<Gradients, AutodiffError> {
        store.backward(&self.0, loss)
    }

    /// [`DeviceParamStore::sync_to_host`] への委譲入口（イシュー #935）。
    /// 上記 `step_device_param_store` と同じ理由の薄い委譲。
    pub fn sync_device_param_store_to_host(
        &self,
        store: &DeviceParamStore,
    ) -> Result<Vec<Tensor<f32>>, BackendError> {
        store.sync_to_host(&self.0)
    }

    /// [`DeviceParamStore::resident_grads_to_host`] への委譲入口
    /// （イシュー #1479）。resident 経路（`GradStaging`）で新鮮に充填
    /// 済みの勾配のみをホストへ読み出す。weight slot は CPU・CUDA
    /// 〈#1559〉・Metal〈#1555〉のいずれも resident 経由で `Some`。bias
    /// slot は Metal〈#1566 以降〉のみ resident 経由で `Some` を返し、
    /// CPU・CUDA は引き続き `None`（bias 縮約はホスト経由の
    /// `Gradients` へ回る。イシュー #1898・`crates/backend-metal/src/
    /// ops.rs::MetalBackendOps::gemm_fp32_strict_into_with_bias_reduce_
    /// tracked` doc 参照）。resident 未対応バックエンド（`gemm_fp32_
    /// strict_into` 未実装のモック等）では [`BackendError::Unsupported`]
    /// を返す（panic なし）。全パラメータの勾配をバックエンド横断で
    /// 読みたい場合は [`Self::param_grads_to_host`] を使う。上記
    /// `sync_device_param_store_to_host` と同じ理由の薄い委譲。
    pub fn resident_grads_to_host(
        &self,
        store: &DeviceParamStore,
        grads: &Gradients,
    ) -> Result<Vec<Option<Tensor<f32>>>, BackendError> {
        store.resident_grads_to_host(&self.0, grads)
    }

    /// [`DeviceParamStore::param_grads_to_host`] への委譲入口
    /// （イシュー #1479）。resident 経由で充填済みの slot は staging
    /// から、それ以外は `grads` からのフォールバックで、全パラメータの
    /// 勾配を 3 バックエンド共通の読み出し窓として返す（CUDA／Metal も
    /// `Ok` を返す。§設計は `fandhe_ai_autodiff::optim::device_store::
    /// DeviceParamStore::param_grads_to_host` doc 参照）。上記と同じ
    /// 理由の薄い委譲。
    pub fn param_grads_to_host(
        &self,
        store: &DeviceParamStore,
        grads: &Gradients,
    ) -> Result<Vec<Tensor<f32>>, BackendError> {
        store.param_grads_to_host(&self.0, grads)
    }

    /// この `Tape` が結線されているデバイス（イシュー #1614）。
    /// `fandhe_ai_autodiff::Tape::device` への委譲。
    pub fn device(&self) -> Device {
        self.0.device()
    }

    /// 別の `facade::Tape`（`self` 自身でもよい）上の `source` を、
    /// この `Tape` 上へ新しい葉ノードとして転送する（イシュー #1614。
    /// `fandhe_ai_autodiff::Var::to_tape` への委譲）。
    ///
    /// facade 利用者は `fandhe_ai_autodiff::Tape`（生の内部型）を
    /// 取り出せない（本型のフィールド `0` は `pub(crate)`）ため、
    /// facade 経由でクロスデバイス転送を行う唯一の入口がこのメソッド
    /// になる。`self` と `source` が同じ `Tape` を指す場合は恒等
    /// （`Var::to_tape` の契約どおり）。値の数値契約・`requires_grad`
    /// 引き継ぎ・`reset` 契約は `fandhe_ai_autodiff::Var::to_tape` の
    /// doc comment を参照。
    pub fn transfer(&self, source: &Var<'_>) -> Result<Var<'_>, AutodiffError> {
        source.to_tape(&self.0)
    }

    /// 子テープ方式の高階微分（`create_graph`）入口（イシュー #2545・
    /// `fandhe_ai_autodiff::Tape::backward_create_graph` への薄い委譲。
    /// 公開形は `docs/autodiff-higher-order-grad-decision.md` §17.2）。
    ///
    /// `loss` を `self`（親テープ）上で逆伝播し、1 階勾配の計算過程を
    /// `child`（子テープ）へ `Var` 演算として記録する。返る
    /// [`CreateGraphResult`] の `grad`／`child_var` で得た子テープ上の
    /// `Var` を `child.backward(..)` でさらに微分すれば二階勾配（Hessian・
    /// HVP）が得られる。`first_order()` は通常の [`Tape::backward`] と
    /// bit 同一。
    ///
    /// `child` は [`tape`]／[`tape_for`] で作った**空**の `Tape` で、`self`
    /// と別のものでなければならない。`BackendOps` を露出しない薄い委譲
    /// （REQ-12）。エラー契約は既存 [`AutodiffError`] の variant のみ:
    /// `Backward`（同一テープ・非空の子・checkpoint・未対応 Op）／
    /// `TapeMismatch`／`DeviceMismatch`／`GradientTrackingDisabled`。
    /// 拒否時は `child` へ何も書き込まない。
    ///
    /// ```
    /// use fandhe_ai::{tape, Tensor};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let t = tape();
    /// let child = tape();
    /// let x = t.var(&Tensor::new(vec![2.0_f32], &[1])?);
    /// let loss = x.mul(&x)?.mul(&x)?.sum(None)?; // x^3
    /// let cg = t.backward_create_graph(&loss, &child)?;
    /// // 1 階: 3x^2 = 12
    /// let first = cg.first_order().get(&x)?.unwrap().as_slice().unwrap().to_vec();
    /// assert!((first[0] - 12.0).abs() < 1e-4);
    /// // 2 階: 6x = 12（子テープ上でもう一度逆伝播する）
    /// let g = cg.grad(&x)?.unwrap();
    /// let grads2 = child.backward(&g.sum(None)?)?;
    /// let xc = cg.child_var(&x)?.unwrap();
    /// let second = grads2.get(&xc)?.unwrap().as_slice().unwrap().to_vec();
    /// assert!((second[0] - 12.0).abs() < 1e-4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn backward_create_graph<'c>(
        &self,
        loss: &Var<'_>,
        child: &'c Tape,
    ) -> Result<CreateGraphResult<'c>, AutodiffError> {
        self.0.backward_create_graph(loss, &child.0)
    }

    /// [`fandhe_ai_autodiff::Tape::custom`] への委譲入口（イシュー #2549。
    /// ルート #2499 の一括承認・`docs/autodiff-custom-function-decision.md`
    /// §16.1 (b)(c)(d)）。
    ///
    /// 利用者実装の [`CustomFunction`]（host の `Tensor<f32>` だけを受け渡す
    /// forward／backward 対）を 1 ノードとして Tape へ記録し、出力 [`Var`]
    /// を返す。`facade::Tape` は内部の autodiff `Tape` を隠すため、本メソッド
    /// が facade 経由の唯一の入口になる（`Var::custom` は設けない）。
    ///
    /// # 契約
    ///
    /// - `forward`／`backward` は常に host 実行で `BackendOps` を経由しない
    ///   ため、REQ-2 のバックエンド間数値一致の判定対象外
    /// - `backward` は `Tape::backward_accumulate` 等で複数回呼ばれうる。
    ///   実装は入力だけに依存する純関数・冪等にすること
    /// - 二階微分（`create_graph`）には非対応（fail-closed で `Err`）
    ///
    /// # エラー
    ///
    /// - 空の `inputs` → [`AutodiffError::InvalidArgument`]
    /// - 別の Tape の入力 → [`AutodiffError::TapeMismatch`]
    /// - 宣言 shape と実出力の不一致 → `Shape`／`Backward` 系
    /// - `requires_grad = true` の入力に `None` を返した backward →
    ///   `Backward`（逆伝播時）
    ///
    /// # 使用例
    ///
    /// ```
    /// use std::sync::Arc;
    /// use fandhe_ai::{AutodiffError, CustomFunction, Tensor};
    ///
    /// // y = 2x（要素ごと）。backward は upstream を 2 倍する。
    /// struct Double;
    ///
    /// impl CustomFunction for Double {
    ///     fn name(&self) -> &str {
    ///         "double"
    ///     }
    ///     fn output_shape(&self, shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
    ///         Ok(shapes[0].to_vec())
    ///     }
    ///     fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
    ///         let d: Vec<f32> = inputs[0].host_slice().iter().map(|v| v * 2.0).collect();
    ///         Tensor::new(d, inputs[0].shape()).map_err(AutodiffError::Shape)
    ///     }
    ///     fn backward(
    ///         &self,
    ///         inputs: &[&Tensor<f32>],
    ///         _out_value: &Tensor<f32>,
    ///         upstream: &Tensor<f32>,
    ///         _requires_grad: &[bool],
    ///     ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
    ///         let d: Vec<f32> = upstream.host_slice().iter().map(|g| g * 2.0).collect();
    ///         let g = Tensor::new(d, inputs[0].shape()).map_err(AutodiffError::Shape)?;
    ///         Ok(vec![Some(g)])
    ///     }
    /// }
    ///
    /// # fn main() -> Result<(), AutodiffError> {
    /// let tape = fandhe_ai::tape();
    /// let x = tape.var(&Tensor::new(vec![1.0, -2.0, 3.0], &[3]).unwrap());
    /// let y = tape.custom(Arc::new(Double), &[x])?;
    /// let loss = y.sum(None)?;
    /// let grads = tape.backward(&loss)?;
    /// let dx = grads.get(&x)?.expect("x は loss に寄与する");
    /// assert_eq!(dx.host_slice().as_ref(), &[2.0, 2.0, 2.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn custom<'t>(
        &'t self,
        func: std::sync::Arc<dyn CustomFunction>,
        inputs: &[Var<'t>],
    ) -> Result<Var<'t>, AutodiffError> {
        self.0.custom(func, inputs)
    }

    /// [`fandhe_ai_autodiff::nn::Rnn::forward_seq`] への委譲入口
    /// （イシュー #1955）。
    ///
    /// `Rnn::forward_seq` は生の `fandhe_ai_autodiff::Tape` を第 1
    /// 引数に取るため、`facade::Tape`（本型。内部フィールド `0` は
    /// `pub(crate)`）の利用者からは直接呼べない。本メソッドは
    /// `&self.0` を渡すだけの薄い委譲であり、`BackendOps` を利用者向け
    /// 公開面へ露出しない（`step_device_param_store` と同じ理由。
    /// `crate::lib.rs` モジュール doc「公開面の設計」・REQ-12）。
    /// `h0` 省略時はゼロ初期化される（`Rnn::forward_seq` の契約を
    /// 参照）。
    pub fn rnn_forward_seq<'t>(
        &'t self,
        rnn: &nn::rnn::Rnn,
        x: &Tensor<f32>,
        h0: Option<&Var<'t>>,
    ) -> Result<nn::rnn::RnnSeqOutput<'t, nn::rnn::RnnCellVars<'t>>, AutodiffError> {
        rnn.forward_seq(&self.0, x, h0)
    }

    /// [`fandhe_ai_autodiff::nn::Lstm::forward_seq`] への委譲入口
    /// （イシュー #1955）。上記 [`Self::rnn_forward_seq`] と同じ理由の
    /// 薄い委譲。`h0`／`c0` 省略時はいずれもゼロ初期化される
    /// （`Lstm::forward_seq` の契約を参照）。
    pub fn lstm_forward_seq<'t>(
        &'t self,
        lstm: &nn::rnn::Lstm,
        x: &Tensor<f32>,
        h0: Option<&Var<'t>>,
        c0: Option<&Var<'t>>,
    ) -> Result<nn::rnn::LstmSeqOutput<'t>, AutodiffError> {
        lstm.forward_seq(&self.0, x, h0, c0)
    }

    /// [`fandhe_ai_autodiff::nn::Gru::forward_seq`] への委譲入口
    /// （イシュー #1955）。上記 [`Self::rnn_forward_seq`] と同じ理由の
    /// 薄い委譲。`h0` 省略時はゼロ初期化される（`Gru::forward_seq`
    /// の契約を参照）。
    pub fn gru_forward_seq<'t>(
        &'t self,
        gru: &nn::rnn::Gru,
        x: &Tensor<f32>,
        h0: Option<&Var<'t>>,
    ) -> Result<nn::rnn::RnnSeqOutput<'t, nn::rnn::GruCellVars<'t>>, AutodiffError> {
        gru.forward_seq(&self.0, x, h0)
    }

    /// 多層・双方向・層間 dropout 付き RNN の `Tape` 委譲メソッド
    /// （イシュー #2535・`docs/autodiff-rnn-stacked-config-decision.md`
    /// §9・`docs/compat-api-scope.md` §5 経路 2）。[`Self::rnn_forward_seq`]
    /// と同じ理由（生の `fandhe_ai_autodiff::Tape` を取り出せない）の
    /// 薄い委譲で、`&self.0` を渡すだけ。
    ///
    /// `h0` は `Some` の場合、長さ `num_layers * num_directions`
    /// （index = `layer * num_directions + direction`）でなければならない。
    /// 双方向時の各 step 出力は `[B, 2H]`（forward と reverse の連結）。
    ///
    /// **既知の制限**: `StackedRnn::new` は training=true で構築され、
    /// `Module::set_training` は facade から到達できないため、`dropout > 0`
    /// の場合は常に層間 dropout を適用してグローバル RNG
    /// （[`manual_seed`] で再現可能。消費順は layer 昇順 → t 昇順）を
    /// 消費する。推論用途では `dropout = 0.0` で構築する。
    pub fn stacked_rnn_forward_seq<'t>(
        &'t self,
        rnn: &nn::rnn::StackedRnn,
        x: &Tensor<f32>,
        h0: Option<&[Var<'t>]>,
    ) -> Result<nn::rnn::StackedRnnSeqOutput<'t, nn::rnn::RnnCellVars<'t>>, AutodiffError> {
        rnn.forward_seq(&self.0, x, h0)
    }

    /// [`Self::stacked_rnn_forward_seq`] の LSTM 版（イシュー #2535）。
    /// `h0`／`c0` は `Some` の場合ともに長さ `num_layers * num_directions`。
    /// eval モード不可・dropout の RNG 消費は同メソッドの doc を参照。
    pub fn stacked_lstm_forward_seq<'t>(
        &'t self,
        lstm: &nn::rnn::StackedLstm,
        x: &Tensor<f32>,
        h0: Option<&[Var<'t>]>,
        c0: Option<&[Var<'t>]>,
    ) -> Result<nn::rnn::StackedLstmSeqOutput<'t>, AutodiffError> {
        lstm.forward_seq(&self.0, x, h0, c0)
    }

    /// [`Self::stacked_rnn_forward_seq`] の GRU 版（イシュー #2535）。
    pub fn stacked_gru_forward_seq<'t>(
        &'t self,
        gru: &nn::rnn::StackedGru,
        x: &Tensor<f32>,
        h0: Option<&[Var<'t>]>,
    ) -> Result<nn::rnn::StackedRnnSeqOutput<'t, nn::rnn::GruCellVars<'t>>, AutodiffError> {
        gru.forward_seq(&self.0, x, h0)
    }

    /// この `Tape` が結線されているバックエンドの `f64` 演算本体
    /// （[`TypedOps<f64>`]）への capability accessor（イシュー #1939・
    /// `docs/compat-api-scope.md` §5 経路 2 承認・`docs/backend-dtype-
    /// dispatch-design.md`）。
    ///
    /// `Tensor<f64>` を直接対象とする 8 演算（`gemm`／`add`／`mul`／
    /// `relu`／`exp`／`tanh`／`sum`／`max`）限定の capability であり、
    /// `Var`／autograd を経由しない（呼び出しは tape に記録されず
    /// 勾配は付かない）。バックエンドが対応しない場合は `None`
    /// （fail-closed。例: Metal は f64 型自体が既定非対応）。`Some` が
    /// 返っても個々の演算が [`BackendError::Unsupported`] を返す場合が
    /// ある（例: CUDA の `TypedOps<f64>` は 8 演算すべて
    /// `Unsupported`）——`Some`/`None` は型としての対応可否のみを表す。
    ///
    /// REQ-12: 返すのは `TypedOps<f64>` の不変借用のみで、`BackendOps`
    /// 自体の注入経路は増えない（`fandhe_ai_autodiff::Tape::ops()` は
    /// 引き続き `pub(crate)`）。
    pub fn typed_ops_f64(&self) -> Option<&dyn TypedOps<f64>> {
        self.0.typed_ops_f64()
    }

    /// `half::f16` 演算本体（[`TypedOps<f16>`]）への capability
    /// accessor。契約は [`Self::typed_ops_f64`] と同じ（イシュー
    /// #1939）。`half::f16` を名指しするには利用者が `half` クレート
    /// （本 workspace と同一バージョン）へ直接依存する（facade は
    /// `half` を再エクスポートしない）。
    pub fn typed_ops_f16(&self) -> Option<&dyn TypedOps<f16>> {
        self.0.typed_ops_f16()
    }

    /// `half::bf16` 演算本体（[`TypedOps<bf16>`]）への capability
    /// accessor。契約は [`Self::typed_ops_f64`] と同じ（イシュー
    /// #1939）。
    pub fn typed_ops_bf16(&self) -> Option<&dyn TypedOps<bf16>> {
        self.0.typed_ops_bf16()
    }
}

/// autodiff コンテナ内で facade 独自層へ渡す、`var` 系メソッドのみの借用ハンドル。
///
/// 役割: `fandhe_ai::nn::Module::forward`（#2395）の第 1 引数、および autodiff
/// 側コンテナ（`compat::Sequential` 内部）から facade 独自層を呼ぶアダプタ（#2397）
/// が `&fandhe_ai_autodiff::Tape` から構築する橋渡しである。生の
/// `fandhe_ai_autodiff::Tape` から facade の [`Tape`]（newtype）を作る安全な手段は
/// なく、`#[repr(transparent)]` と `unsafe` の参照キャストは不採用（#2338
/// 2026-09-29 承認事項 4「案 1: 借用ハンドル型・`unsafe` なし」）。
///
/// REQ-12: 生の `Tape`／`BackendOps` を露出しないため、公開メソッドは
/// [`Self::var`]・[`Self::var_from`]・[`Self::var_no_grad`] の 3 件のみで、
/// `backward` 等は委譲しない。フィールドは `pub(crate)`。crate 内の構築は
/// `from_autodiff`、crate 外の入口は `From<&Tape>` のみ。
/// `tests/api_surface.rs` がこの面を機械的に固定する。
#[derive(Clone, Copy)]
pub struct TapeRef<'t>(pub(crate) &'t fandhe_ai_autodiff::Tape);

impl<'t> TapeRef<'t> {
    /// `&fandhe_ai_autodiff::Tape` から借用ハンドルを作る crate 内専用の構築経路
    /// （#2397 のアダプタが使う。`From<&Tape>` もここへ集約する）。
    pub(crate) fn from_autodiff(tape: &'t fandhe_ai_autodiff::Tape) -> Self {
        Self(tape)
    }

    /// [`Tape::var`] と同じ（葉ノード登録）。戻り値の寿命は借用元テープ `'t` に結び付く。
    pub fn var(&self, tensor: &Tensor<f32>) -> Var<'t> {
        self.0.var(tensor)
    }

    /// [`Tape::var_from`] と同じ（dtype 変換つきの葉ノード登録）。
    pub fn var_from<T: CastElement>(&self, tensor: &Tensor<T>) -> Result<Var<'t>, AutodiffError> {
        self.0.var_from(tensor)
    }

    /// [`Tape::var_no_grad`] と同じ（`requires_grad == false` の葉ノード登録）。
    pub fn var_no_grad(&self, tensor: &Tensor<f32>) -> Var<'t> {
        self.0.var_no_grad(tensor)
    }
}

impl std::fmt::Debug for TapeRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // facade `Tape` の `Debug` と同じ方針（内部 `Tape` の表現に依存しない）。
        f.debug_struct("TapeRef").finish_non_exhaustive()
    }
}

impl<'t> From<&'t Tape> for TapeRef<'t> {
    fn from(tape: &'t Tape) -> Self {
        Self::from_autodiff(&tape.0)
    }
}

/// 既定バックエンド（CPU・TASK-2.5 ユーザー承認済み。
/// `docs/public-api-design.md:429`）で [`Tape`] を構築する。
///
/// CPU は常に利用可能であるため非 fallible。`CpuBackendOps::new()`
/// （`run_fused` を `run_fused_elementwise` へオーバーライド済み＝融合
/// 有効。`crates/backend-cpu/src/ops.rs`）を結線する唯一の入口。
pub fn tape() -> Tape {
    Tape(fandhe_ai_autodiff::Tape::new_with_ops(Box::new(
        fandhe_ai_backend_cpu::CpuBackendOps::new(),
    )))
}

/// 指定した [`Device`] へ明示的に結線した [`Tape`] を構築する。
///
/// 存在しないデバイス・範囲外 ordinal・driver 不在は [`BackendError`]
/// を返す（fail-fast。本番経路で `panic!`／`unwrap()` しない。
/// `.claude/rules/coding-rust.md`）。構築規則はモジュール冒頭コメント
/// 参照。
pub fn tape_for(device: Device) -> Result<Tape, BackendError> {
    let ops = resolve_ops(device)?;
    Ok(Tape(fandhe_ai_autodiff::Tape::new_with_ops(ops)))
}

/// 実行時に利用可能な [`Device`] を列挙する（`docs/public-api-design.md`
/// §4.1 `Device::available()` の集約入口。イシュー #1614。TASK-1.9a
/// （#44。`tensor-core::device` モジュール doc）が「集約入口をどの層で
/// 結線するかは後続イシューへ引き継ぐ」としていた未決事項をここで解消
/// する）。
///
/// `tensor-core::device::enumerate_all`（`&[&dyn DeviceProvider]` を
/// 横断して列挙する下位ヘルパー。`tensor-core` は 3 バックエンドを
/// 直接参照できないため依存逆転構成を取る）へ、composition root
/// （本クレート）が 3 バックエンドの `DeviceProvider` を束ねて渡す
/// 唯一の入口。返すのは [`Device`] 識別子のみで、`DeviceProvider`／
/// `DeviceInfo`（デバイス名・メモリ容量等）は再エクスポートしない
/// （REQ-12 と同じ「利用者向け公開面は `Device` 識別子のみ」の方針。
/// `tests/api_surface.rs` が機械的に固定する）。
///
/// **順序契約**: `Device::Cpu`（常に含まれる）→ `Device::Cuda(0..n)`
/// （ordinal 昇順。CUDA driver 不在なら 0 件）→ `Device::Metal`
/// （`cfg(target_os = "macos")` 限定。デバイス検出 0 件なら含まれない）
/// の順で、同一プロセス内の複数回呼び出しでも決定的。
///
/// **副作用**: `CudaDeviceProvider::enumerate` は検出した CUDA
/// ordinal ごとに `context_cache` のコンテキストを初期化する
/// （`crates/backend-cuda/src/device.rs`。2 回目以降はキャッシュ
/// ヒットで軽量）。これは後で [`tape_for`] が払うはずのコストを
/// 前倒しするだけであり、追加コストではない。
///
/// **一貫性契約**: 本関数が返した各 `Device` に対して [`tape_for`]
/// を呼べば `Ok` になることを意図する（ただし列挙と `tape_for` の
/// 呼び出しの間でデバイス状態が変化する TOCTOU は契約外——ドライバの
/// 抜き差し等の外部要因まではカバーしない）。
///
/// **fail-safe**: 個々のバックエンドの列挙が失敗しても `panic!`／
/// `unwrap()` せず、その分を除いた結果を返す（`enumerate_all` の
/// fail-safe 方針をそのまま引き継ぐ）。
pub fn available_devices() -> Vec<Device> {
    let cpu = fandhe_ai_backend_cpu::CpuDeviceProvider::new();
    let cuda = fandhe_ai_backend_cuda::CudaDeviceProvider::new();
    #[cfg(target_os = "macos")]
    let metal = fandhe_ai_backend_metal::MetalDeviceProvider::new();

    // macOS 以外では 2 要素で固定（`providers` を `mut` にすると
    // 非 macOS ビルドで `push` が到達不能になり `unused_mut` 警告
    // （`-D warnings` で fail）になるため、cfg ごとに構築を分ける）。
    #[cfg(target_os = "macos")]
    let providers: Vec<&dyn DeviceProvider> = vec![&cpu, &cuda, &metal];
    #[cfg(not(target_os = "macos"))]
    let providers: Vec<&dyn DeviceProvider> = vec![&cpu, &cuda];

    fandhe_ai_tensor_core::enumerate_all(&providers)
        .into_iter()
        .map(|info| info.device)
        .collect()
}

/// PyTorch `torch.manual_seed` 相当。プロセス全体で共有されるグローバル
/// 決定的 RNG（[`randn`]／[`rand`]／[`randint`] が消費する。#1602 本文の
/// 設計方針どおりホスト生成のみで `BackendOps` は経由しない）の状態を
/// `seed` からやり直す（イシュー #1724）。
///
/// `fandhe_ai_autodiff::manual_seed`（実体は `fandhe_ai_tensor_core::rng`）
/// への薄い委譲（composition root。`docs/compat-api-scope.md` §0 の確定
/// 公開面）。既存の個別シード API（`nn::Linear::new(.., seed)` 等）とは
/// 独立した別機構であり、本関数を呼んでもそれらの挙動には一切影響しない
/// （設計判断・スレッド安全性の範囲は `docs/rng-global-contract-design.md`）。
pub fn manual_seed(seed: u64) {
    fandhe_ai_autodiff::manual_seed(seed);
}

/// PyTorch `torch.use_deterministic_algorithms` 相当の決定論モードを
/// プロセス全体で切り替える（イシュー #2507。実装は #2157）。既定は
/// `false`（opt-in）。
///
/// `fandhe_ai_autodiff::determinism::set_deterministic` への薄い委譲
/// （composition root。`set_cuda_gemm_precision` と同型の crate ルート
/// 自由関数）。現時点では状態を記録するだけの no-op 契約で、ON／OFF で
/// forward／backward の結果は bit 一致する。保証範囲は CPU（同一
/// バイナリ・同一マシン）に限り、CUDA／Metal の `Tape` には等しく no-op
/// で決定性は未検証である（並行スレッドからの RNG 消費順も対象外）。
/// 設計判断・契約は `docs/autodiff-determinism-mode-design.md` §3 を参照。
///
/// ```
/// fandhe_ai::set_deterministic(true);
/// assert!(fandhe_ai::is_deterministic());
/// fandhe_ai::set_deterministic(false);
/// assert!(!fandhe_ai::is_deterministic());
/// ```
pub fn set_deterministic(enabled: bool) {
    fandhe_ai_autodiff::determinism::set_deterministic(enabled);
}

/// [`set_deterministic`] で設定した決定論モードの現在値を返す（既定
/// `false`。イシュー #2507）。`fandhe_ai_autodiff::determinism::
/// is_deterministic` への薄い委譲で、保証範囲・no-op 契約は
/// [`set_deterministic`] を参照。
pub fn is_deterministic() -> bool {
    fandhe_ai_autodiff::determinism::is_deterministic()
}

/// 標準正規分布 `N(0, 1)` に従う乱数テンソルを生成する（PyTorch
/// `torch.randn` 相当。イシュー #1725）。[`manual_seed`] が設定した
/// プロセスグローバル決定的 RNG をホスト側だけで消費し（`BackendOps` を
/// 経由しない。#1602 本文の設計方針）、返る [`Tensor`] は [`Tape::var`]
/// で任意のデバイスへアップロードできる。
///
/// `fandhe_ai_autodiff::randn`（実体は `fandhe_ai_tensor_core::rng::randn`）
/// への薄い委譲（composition root）。決定性の範囲（同一プロセス・同一
/// プラットフォーム限定）・アルゴリズムは `docs/rng-global-contract-design.md`
/// を参照。
pub fn randn(shape: &[usize]) -> Result<Tensor<f32>, ShapeError> {
    fandhe_ai_autodiff::randn(shape)
}

/// `[0, 1)` の一様分布に従う乱数テンソルを生成する（PyTorch `torch.rand`
/// 相当。イシュー #1725）。設計・到達経路は [`randn`] と同じ。整数演算
/// のみで構成されるためプラットフォーム横断で bit 同一の決定性を持つ
/// （`fandhe_ai_tensor_core::rng` モジュール doc 参照）。
pub fn rand(shape: &[usize]) -> Result<Tensor<f32>, ShapeError> {
    fandhe_ai_autodiff::rand(shape)
}

/// `[low, high)` の一様分布に従う整数乱数テンソルを生成する（PyTorch
/// `torch.randint` 相当。イシュー #1725）。dtype は `i32`
/// （本リポの index／targets 型契約に合わせた意図的な差異。
/// `docs/rng-global-contract-design.md`）。`low >= high` は
/// [`RngError::InvalidRange`] を返す。
pub fn randint(low: i32, high: i32, shape: &[usize]) -> Result<Tensor<i32>, RngError> {
    fandhe_ai_autodiff::randint(low, high, shape)
}

/// `[start, end)` を `step` 刻みで並べたテンソルを生成する（PyTorch
/// `torch.arange` 相当。イシュー #1726）。[`randn`]／[`rand`]／
/// [`randint`] と同じくホスト側だけで完結し `BackendOps` を経由しない
/// （`Op`／VJP も追加しない。微分不能な葉値のため）。数値契約
/// （プラットフォーム横断で bit 同一）は
/// `fandhe_ai_tensor_core::creation` モジュール doc 参照。
///
/// `fandhe_ai_autodiff::arange`（実体は
/// `fandhe_ai_tensor_core::creation::arange`）への薄い委譲
/// （composition root）。
pub fn arange(start: f32, end: f32, step: f32) -> Result<Tensor<f32>, CreationError> {
    fandhe_ai_autodiff::arange(start, end, step)
}

/// `[start, end]`（両端を含む）を `steps` 個の等間隔値で埋めたテンソルを
/// 生成する（PyTorch `torch.linspace` 相当。イシュー #1726）。設計・
/// 到達経路は [`arange`] と同じ。
pub fn linspace(start: f32, end: f32, steps: usize) -> Result<Tensor<f32>, CreationError> {
    fandhe_ai_autodiff::linspace(start, end, steps)
}

/// `n x n` の単位行列を生成する（PyTorch `torch.eye` 相当。長方形版は
/// 対象外。イシュー #1726）。dtype は `f32` 固定（facade 公開面は他の
/// 生成系トップレベル関数と同じく `f32` 限定。汎用版は
/// `fandhe_ai_tensor_core::creation::eye::<T>` を参照）。
pub fn eye(n: usize) -> Result<Tensor<f32>, ShapeError> {
    fandhe_ai_autodiff::eye::<f32>(n)
}

/// `like` と同じ shape・全要素 `0.0` の新規テンソルを生成する（PyTorch
/// `torch.zeros_like` 相当。イシュー #1726）。`like` が転置・broadcast
/// 由来の非 contiguous view であっても shape のみを引き継ぎ、strides
/// は保存しない新規 contiguous バッファを返す
/// （`fandhe_ai_tensor_core::creation` モジュール doc 参照）。
pub fn zeros_like(like: &Tensor<f32>) -> Result<Tensor<f32>, ShapeError> {
    fandhe_ai_autodiff::zeros_like(like)
}

/// `like` と同じ shape・全要素 `1.0` の新規テンソルを生成する（PyTorch
/// `torch.ones_like` 相当。イシュー #1726）。契約は [`zeros_like`] と
/// 同じ。
pub fn ones_like(like: &Tensor<f32>) -> Result<Tensor<f32>, ShapeError> {
    fandhe_ai_autodiff::ones_like(like)
}

/// `device` に対応する具体 `BackendOps` を解決する（非公開）。
///
/// composition root の中核: `Device` → 具体バックエンドクレートの
/// `BackendOps` 実装への唯一の変換点。呼び出し元は [`tape_for`] のみ。
fn resolve_ops(device: Device) -> Result<Box<dyn BackendOps + Send>, BackendError> {
    match device {
        Device::Cpu => Ok(Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new())),
        Device::Cuda(ordinal) => {
            // `CudaDeviceProvider::select` で存在検証してから構築する
            // （driver 不在・範囲外 ordinal を fail-fast で弾く。
            // `crates/backend-cuda/src/device.rs::CudaDeviceProvider`）。
            let provider = fandhe_ai_backend_cuda::CudaDeviceProvider::new();
            let providers: [&dyn DeviceProvider; 1] = [&provider];
            select_from(&providers, device)?;
            Ok(Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(
                ordinal,
            )))
        }
        #[cfg(target_os = "macos")]
        Device::Metal => {
            // `MetalDeviceProvider::select` で存在検証してから構築する
            // （`crates/backend-metal/src/device.rs::MetalDeviceProvider`）。
            //
            // イシュー #2114: 存在確認（`probe_all` = IOKit + `MTLCopyAllDevices`）は
            // `tape_for`／`predict_resident` の呼び出しごとに走り、tape_build の主因と推定される。
            // `verify_device_cached` は既定 OFF で従来と同じく毎回 `select_from` を実行する。
            // opt-in（backend-metal 側の `#[doc(hidden)]` ガード）の ON 時のみ、成功済みの
            // 存在確認を再利用する（失敗はキャッシュしない。設計は
            // `docs/perf/metal-tape-build-infer-fixedcost.md`）。
            fandhe_ai_backend_metal::fixed_cost_diag::verify_device_cached(|| {
                let provider = fandhe_ai_backend_metal::MetalDeviceProvider::new();
                let providers: [&dyn DeviceProvider; 1] = [&provider];
                select_from(&providers, device).map(|_| ())
            })?;
            Ok(Box::new(fandhe_ai_backend_metal::MetalBackendOps::new()))
        }
    }
}

/// REQ-14 の明示解放 API（イシュー #1018 ツリー・#1020 CUDA・#1021
/// Metal）。`device` に対応するバックエンドのデバイスメモリプール
/// （`backend-cuda`／`backend-metal` のサイズクラス別プール実装）が
/// アイドル保持している分を即座に実解放する。プールを持たない
/// バックエンド（CPU 等）は何もせず `Ok(())` を返す（`BackendOps::
/// release_cached_device_memory` の既定契約。`docs/device-memory-pool-
/// design.md` §3.1「facade からの再公開」）。
///
/// `fandhe_ai_tensor_core::BackendOps::release_cached_device_memory` への
/// 薄い委譲（composition root。`docs/compat-api-scope.md` §0 の確定
/// 公開面）。`Device` は `tensor-core` 由来の外部型（識別子 enum）の
/// ため facade は inherent メソッドを追加できず（orphan rule。
/// `docs/facade-device-handle-design.md` の「案 B のみ採用」方針）、
/// [`tape_for`] と同型の自由関数として公開する。存在しないデバイス・
/// 範囲外 ordinal・driver 不在は [`BackendError`] を返す（`tape_for` と
/// 同じ fail-fast。`resolve_ops` を経由するため検証規則も同一）。
pub fn release_cached_memory(device: Device) -> Result<(), BackendError> {
    resolve_ops(device)?.release_cached_device_memory()
}

/// `device` のデバイスメモリプールの統計スナップショット（診断用。
/// イシュー #1020・#1021）。POD [`PoolStats`] のみを返し、内部ハンドル
/// 表現は含まない（`docs/device-memory-pool-design.md` §3.1「facade
/// からの再公開」）。プールを持たないバックエンドは `Ok(None)`
/// （`BackendOps::device_memory_pool_stats` の既定実装）。存在しない
/// デバイス・範囲外 ordinal・driver 不在は [`release_cached_memory`]
/// と同じく [`BackendError`] を返す。
pub fn memory_pool_stats(device: Device) -> Result<Option<PoolStats>, BackendError> {
    Ok(resolve_ops(device)?.device_memory_pool_stats())
}

/// CUDA GEMM（`fandhe_ai::tape().var(a).matmul(b)` 等が最終的に到達する
/// `CudaBackendOps::gemm`）の TF32 Tensor Core 経路を opt-in で有効化・
/// 無効化する（イシュー #1042。親ツリー #1029 Phase 2）。
///
/// `fandhe_ai_backend_cuda::precision::set_tf32_gemm_enabled` への薄い
/// 委譲（composition root。`docs/compat-api-scope.md` §0 の確定公開面）。
/// **既定は無効（FP32 厳密）**。有効化すると以降の全スレッド・全 CUDA
/// device の `gemm` 呼び出しがプロセスワイドに TF32 Tensor Core 経路へ
/// 切り替わる（`Device` 単位ではない。`fandhe_ai_backend_cuda::precision`
/// モジュール冒頭コメントの契約参照）。有効時に TF32 カーネルが使用不能
/// （cc<8.0・NVRTC コンパイル失敗等）な環境では `gemm` 呼び出しが
/// [`BackendError`] を返す（fail-closed。FP32 への黙示フォールバックは
/// しない）。数値一致許容誤差（相対 1e-3 未満 または 絶対 1e-5 未満）・
/// 適用範囲（`gemm_bias_act`・`gemm_resident_*`・学習経路は対象外）は
/// 変更しない（`docs/cuda-tf32-optin-api-decision.md`）。
pub fn set_cuda_tf32_gemm_enabled(enabled: bool) {
    fandhe_ai_backend_cuda::precision::set_tf32_gemm_enabled(enabled);
}

/// [`set_cuda_tf32_gemm_enabled`] で設定した現在の opt-in 状態を返す
/// （既定 `false`）。
pub fn cuda_tf32_gemm_enabled() -> bool {
    fandhe_ai_backend_cuda::precision::tf32_gemm_enabled()
}

/// 学習 step の update 区間（`BackendOps::sgd_step_device_tracked`）を
/// CUDA Graph で capture・再利用する経路を opt-in で有効化・無効化する
/// （イシュー #1349。親 #1348・ルート #1341 → #1269）。
///
/// `fandhe_ai_backend_cuda::graph::set_step_graph_enabled` への薄い委譲
/// （composition root。`docs/compat-api-scope.md` §0 の確定公開面）。
/// **既定は無効**。有効化すると以降の全スレッド・全 CUDA device の学習
/// step update 区間がプロセスワイドに capture 対象となる（`Device` 単位
/// ではない。`fandhe_ai_backend_cuda::graph` モジュール冒頭コメントの
/// 契約参照）。
///
/// **重要な制約（設定タイミング）**: `CudaDevice` は ordinal ごとに
/// プロセス内で 1 回だけ構築・キャッシュされる（`context_cache::
/// cached_device`）ため、本関数は**最初の CUDA デバイス初期化より前**
/// （`fandhe_ai::tape_for(Device::Cuda(_))` の最初の呼び出しより前）に
/// 呼ぶ必要がある。有効化がデバイス初期化に間に合わなかった場合、以降の
/// 学習 step は `BackendError::Unsupported` を返し fail-closed に拒否
/// する（silent に非 capture 経路へフォールバックしない。`docs/backend-
/// cuda-graph-step-capture-design.md` §4.7）。
///
/// capture 対象は update 区間のみ（forward／backward は対象外。同 design
/// doc §1・§3.2）であり、`bit` 単位で非 capture 経路と同一の損失・勾配・
/// パラメータになることを受け入れ条件とする（同 doc §7 受け入れ (a)）。
/// 非対応バックエンド（CPU／Metal）・opt-in OFF では
/// `BackendOps::captured_segment_key` が常に `Ok(None)` を返すため、
/// `DeviceParamStore::step` は現行の直接実行経路のまま動作する。
pub fn set_cuda_graph_step_enabled(enabled: bool) {
    fandhe_ai_backend_cuda::graph::set_step_graph_enabled(enabled);
}

/// [`set_cuda_graph_step_enabled`] で設定した現在の opt-in 状態を返す
/// （既定 `false`）。
pub fn cuda_graph_step_enabled() -> bool {
    fandhe_ai_backend_cuda::graph::step_graph_enabled()
}

/// [`fandhe_ai_backend_cuda::graph::StepGraphMode`] の再公開（composition
/// root。イシュー #1350）。`framework-compare` の `bench-fandhe` が
/// `--graph stream-only` 起動時に「環境変数 `FANDHE_AI_CUDA_GRAPH_STEP=
/// stream-only` が実際に反映されたか」を確認するために使う診断用途の
/// 公開面（`docs/compat-api-scope.md` §0）。`set_cuda_graph_step_enabled`
/// と異なり API からモードを設定する手段はない（`stream-only` は環境
/// 変数専用の中間状態。`fandhe_ai_backend_cuda::graph` モジュール冒頭
/// コメント参照）。
pub type CudaGraphStepMode = fandhe_ai_backend_cuda::graph::StepGraphMode;

/// [`fandhe_ai_backend_cuda::graph::StepGraphStats`] の再公開（同上。
/// launch 固定費の診断用スナップショット）。
pub type CudaGraphStepStats = fandhe_ai_backend_cuda::graph::StepGraphStats;

/// 現在の CUDA Graph step capture の opt-in モードを返す（イシュー
/// #1350。[`CudaGraphStepMode`] 参照）。
pub fn cuda_graph_step_mode() -> CudaGraphStepMode {
    fandhe_ai_backend_cuda::graph::step_graph_mode_public()
}

/// launch 固定費の診断用スナップショットを返す（イシュー #1350。
/// [`CudaGraphStepStats`] 参照）。計測ウィンドウの外（`framework-compare`
/// の record 生成時）で 1 回だけ呼ぶことを想定しており、呼び出し自体は
/// 計測時間へ計上されない設計（`fandhe_ai_backend_cuda::graph::
/// step_graph_stats` doc コメント参照）。
pub fn cuda_graph_step_stats() -> CudaGraphStepStats {
    fandhe_ai_backend_cuda::graph::step_graph_stats()
}

/// CUDA GEMM 精度モード（既定 `Fp32Strict`・単発 `Tf32`・3×TF32
/// `Tf32x3`。イシュー #1355。親ツリー #1354・承認元 #1338）の再公開。
/// `set_cuda_gemm_precision`／`cuda_gemm_precision` の戻り値・引数型
/// として使う（`fandhe_ai_backend_cuda::precision::CudaGemmPrecision` の
/// 薄い再公開。[`PoolStats`] と同じ前例）。
pub use fandhe_ai_backend_cuda::precision::CudaGemmPrecision;

/// CUDA GEMM（`fandhe_ai::tape().var(a).matmul(b)` 等が最終的に到達する
/// `CudaBackendOps::gemm`）の精度モードを設定する（イシュー #1355。
/// [`set_cuda_tf32_gemm_enabled`] の 3 モード拡張版・同型の composition
/// root 委譲）。
///
/// `fandhe_ai_backend_cuda::precision::set_gemm_precision` への薄い委譲。
/// **既定は [`CudaGemmPrecision::Fp32Strict`]**。`CudaGemmPrecision::
/// Tf32x3` を指定すると、以降の全スレッド・全 CUDA device の `gemm`
/// 呼び出しが 3×TF32（split-single 法。hi/lo 分割・3 回の `mma.sync`
/// 累積）経路へプロセスワイドに切り替わる（`Device` 単位ではない。
/// `fandhe_ai_backend_cuda::precision` モジュール冒頭コメントの契約
/// 参照）。有効時にモード固有のカーネルが使用不能（cc<8.0・NVRTC
/// コンパイル失敗・整列制約不成立等）な環境では `gemm` 呼び出しが
/// [`BackendError`] を返す（fail-closed。FP32 への黙示フォールバックは
/// しない）。`Tf32x3` は f32 SIMT と bit 一致しない（
/// `.claude/rules/coding-rust.md` FMA 契約統一節の明示的例外。数値一致
/// 許容誤差自体は変更しない）。適用範囲（`gemm_bias_act`・
/// `gemm_resident_*`・学習経路は対象外）は [`set_cuda_tf32_gemm_enabled`]
/// と同じ（`docs/cuda-tf32-optin-api-decision.md`・
/// `docs/cuda-tf32x3-split-single-decision.md`）。
///
/// [`set_cuda_tf32_gemm_enabled`]／[`cuda_tf32_gemm_enabled`]（旧 2 値
/// API）は互換ラッパーとして維持する: `set_cuda_tf32_gemm_enabled(false)`
/// はどのモードからでも `Fp32Strict` へ戻す。`cuda_tf32_gemm_enabled()`
/// は単発 `Tf32` のときのみ `true` を返す（`Tf32x3` では `false`）。
pub fn set_cuda_gemm_precision(mode: CudaGemmPrecision) {
    fandhe_ai_backend_cuda::precision::set_gemm_precision(mode);
}

/// [`set_cuda_gemm_precision`] で設定した現在の精度モードを返す
/// （既定 [`CudaGemmPrecision::Fp32Strict`]）。
pub fn cuda_gemm_precision() -> CudaGemmPrecision {
    fandhe_ai_backend_cuda::precision::gemm_precision()
}

/// CUDA `DeviceBuffer` の確保配置（`alloc_zeroed`／`upload`）を managed
/// memory（`cuMemAllocManaged`）へ opt-in で切り替える（イシュー #1352。
/// 親 #1351「GB10 物理統合メモリ向けゼロコピー割当の試作・実測」）。
///
/// `fandhe_ai_backend_cuda::placement::set_managed_placement_enabled` への
/// 薄い委譲（[`set_cuda_tf32_gemm_enabled`] と同型の composition root。
/// `docs/compat-api-scope.md` §0 の確定公開面）。**既定は無効
/// （`cuMemAlloc` による device-only 配置）**。有効化すると以降の全
/// スレッド・全 CUDA device の確保呼び出しがプロセスワイドに managed
/// 配置へ切り替わる（`Device` 単位ではない。`fandhe_ai_backend_cuda::
/// placement` モジュール冒頭コメントの契約参照）。
///
/// **本イシュー時点のスコープ**（`docs/backend-cuda-managed-placement-
/// decision.md` 参照）: `MemoryOps::alloc_zeroed`／`upload`／`download`
/// と、これらを経由する GEMM／SGD 常駐経路
/// （`gemm_resident_rhs`／`gemm_resident_lhs`（NT 転置分岐を除く）／
/// `linear_forward_device`／`sgd_step_device`）は managed 配置に対応
/// 済み。fresh モードの素の `CudaBackendOps::gemm`（`run_tiled_f32` 系。
/// `CudaMemory` を経由せず `clone_htod`／`alloc_zeros` を直接呼ぶ）・
/// `gemm_resident_lhs` の NT 転置分岐・その他の演算（elementwise・
/// rmsnorm・softmax 等）は本イシューでは managed 化していない
/// （既定 device-only のまま変更なし。opt-in 時にこれらの経路を通ると
/// 従来どおり device-only 配置で動作する。managed 化の可否は #1353 の
/// 実測結果を踏まえ後続で判断する）。
///
/// 対象デバイスが managed memory 非対応（`CU_DEVICE_ATTRIBUTE_
/// MANAGED_MEMORY`／`CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS` の
/// いずれかが 0）の場合、有効時の確保呼び出しは
/// [`BackendError::Unsupported`] を返す（fail-closed。device-only への
/// 黙示フォールバックはしない。`set_cuda_tf32_gemm_enabled` と同じ方針）。
///
/// 出力の数値契約: 配置はメモリの物理的な置き場所のみを変え、確保
/// バッファを読み書きするカーネル本体・起動 config は device-only／
/// managed の両配置で完全に共有するため、出力は配置に依らず bit
/// 同一となる（`fandhe_ai_backend_cuda::memory` モジュール冒頭コメント
/// 「配置（managed 拡張）」参照）。既定（無効）時の経路・出力は本
/// イシュー導入前と完全に不変。
pub fn set_cuda_managed_memory_enabled(enabled: bool) {
    fandhe_ai_backend_cuda::placement::set_managed_placement_enabled(enabled);
}

/// [`set_cuda_managed_memory_enabled`] で設定した現在の opt-in 状態を
/// 返す（既定 `false`）。
pub fn cuda_managed_memory_enabled() -> bool {
    fandhe_ai_backend_cuda::placement::managed_placement_enabled()
}

/// CUDA H2D（ホスト→デバイス転送）を、pageable ホストメモリからの直接
/// 転送ではなく pinned（page-locked）ステージングバッファ経由で発行する
/// opt-in スイッチ（イシュー #1585。低レイヤー診断
/// `docs/perf/lowlayer-diagnosis-2026-09-12.md` §7 表 B-3 行で起票された
/// 候補）。
///
/// `fandhe_ai_backend_cuda::{set_pinned_h2d_enabled}` への薄い委譲
/// （[`set_cuda_managed_memory_enabled`] と同型の composition root。
/// `docs/compat-api-scope.md` §0 の確定公開面）。**既定は無効**（pageable
/// ホストメモリからの `clone_htod`／`memcpy_htod` 直接発行。導入前と
/// 経路・出力は bit 同一）。有効化すると以降の全スレッド・全
/// `CudaMemory`／`CudaGemm` インスタンスの H2D 発行（`upload`／
/// `upload_into`・`Var::matmul` 等が最終的に到達する各 GEMM `run_*`
/// 系）がプロセスワイドに pinned staging 経由へ切り替わる。
///
/// **本イシュー時点のスコープ**（`docs/perf/cuda-h2d-pinned-staging.md`
/// 参照）: f32 経路（`MemoryOps::upload`／`upload_into`・fresh／resident
/// GEMM の各 `run_*`／`launch_*` 系）のみ対応。TF32 Tensor Core 経路
/// （`run_wmma_tf32` が到達する `run_wmma_f32_kernel`／
/// `run_wmma_tf32_opt_kernel`／`run_wmma_tf32_staged_kernel`）は入力が
/// `&[f32]` のため `upload_h2d_new` を経由し、有効化時は pinned staging
/// の対象に含まれる。**対象外のまま**なのは f16 Tensor Core 経路
/// （`gemm_mma.rs` の `run_f16` 系・`gemm.rs::run_f16_kernel`。f16 データ
/// は `upload_h2d_new` の型〈`&[f32]`〉と一致しないため構造的に非到達）・
/// elementwise／rmsnorm／softmax／transpose／mse のみ（既定どおり
/// pageable のまま変更なし）。
///
/// 数値契約: pinned バッファへ同期コピーしてから DMA を発行するだけで
/// 転送対象の内容自体は変わらないため、出力は経路に依らず bit 同一
/// （`fandhe_ai_backend_cuda::host_staging` モジュール「H2D 用
/// ステージング」節）。既定（無効）時の経路・出力は本イシュー導入前と
/// 完全に不変。
pub fn set_cuda_pinned_h2d_enabled(enabled: bool) {
    fandhe_ai_backend_cuda::set_pinned_h2d_enabled(enabled);
}

/// [`set_cuda_pinned_h2d_enabled`] で設定した現在の opt-in 状態を返す
/// （既定 `false`）。
pub fn cuda_pinned_h2d_enabled() -> bool {
    fandhe_ai_backend_cuda::pinned_h2d_enabled()
}

/// Metal GEMM（`fandhe_ai::tape().var(a).matmul(b)` 等が最終的に到達する
/// `MetalBackendOps::gemm`）の split-K 2 パス経路（イシュー #1516 で
/// 本番結線・#1544 で既定有効化）を実行時に無効化する opt-out スイッチ
/// （イシュー #1545）。
///
/// `fandhe_ai_backend_metal::split_k_runtime::set_split_k_enabled` への
/// 薄い委譲（[`set_cuda_tf32_gemm_enabled`] と同型の composition root。
/// `docs/compat-api-scope.md` §0 の確定公開面）。**既定は有効
/// （`true`）**——コンパイル時ゲート（`SPLIT_K_DISPATCH_AUTO_PRODUCTION_
/// ENABLED`）・インスタンス単位ゲート（`MetalGemm::split_k_auto_enabled`）
/// に続く 3 段目のゲートであり、本関数を呼ばない限り #1544 で確定した
/// 本番既定挙動は完全に不変。`false` にすると、以降の全スレッドの
/// `dispatch_auto`（`MetalBackendOps::gemm` が通る本番 NN 入口）が
/// split-K 到達形状も含めて常に classic 経路へ固定され、結線前
/// （`fandhe-ai =0.8.0` 相当）と bit 同一の出力を返す（fail-closed に
/// 「安全な既知の経路」へ倒す設計。`true` に戻すと従来どおりの経路選択
/// に戻る）。プロセスワイドな設定であり（`Device` 単位ではない）、
/// `fandhe_ai_backend_metal::split_k_runtime` モジュール冒頭コメントの
/// 3 段ゲートの関係・`docs/backend-metal-splitk-decision.md` §5
/// 「実行時トグル」を参照。
///
/// split-K 到達形状（K 支配的な非正方形状。`tile::should_split_k`）では
/// classic 経路と bit 一致しない（実測 baseline 非後退方式による受け入れ。
/// `docs/backend-metal-splitk-parity-judgment-decision.md` §7）。数値一致
/// 許容誤差そのものは変更しない。適用範囲は `dispatch_auto` を経由する
/// `MetalBackendOps::gemm` 系のみ（`gemm_bias_act` 融合カーネルは元々
/// classic 経路のみのため対象外）。
#[cfg(target_os = "macos")]
pub fn set_metal_split_k_gemm_enabled(enabled: bool) {
    fandhe_ai_backend_metal::split_k_runtime::set_split_k_enabled(enabled);
}

/// [`set_metal_split_k_gemm_enabled`] で設定した現在の実行時トグル状態を
/// 返す（既定 `true`）。
#[cfg(target_os = "macos")]
pub fn metal_split_k_gemm_enabled() -> bool {
    fandhe_ai_backend_metal::split_k_runtime::split_k_enabled()
}

/// [`crate::interop::onnx::OnnxModel::run`] の GPU 実行（CUDA）を opt-in
/// で有効化・無効化する（イシュー #2077。`docs/onnx-gpu-execution-
/// decision.md`）。
///
/// `crate::interop::onnx` の `pub(crate)` 状態への薄い委譲（composition
/// root。[`set_cuda_tf32_gemm_enabled`] と同型）。**既定は無効
/// （`false`）**——無効時の `OnnxModel::run` の経路・出力は本イシュー
/// 導入前と bit 完全に不変。有効化すると、以降の全スレッドの
/// `OnnxModel::run` 呼び出しが `fandhe_ai_backend_cuda::CudaBackendOps`
/// （ordinal 0 固定）経由の device 実行を試みる（プロセスワイド。
/// `Device` 単位の切替 API は設けない）。CUDA driver 不在・ordinal 0 が
/// 存在しない環境では `run` 自体が [`crate::interop::onnx::OnnxError::
/// Execution`] を返す（ホスト CPU への黙示フォールバックはしない。
/// fail-closed。OWASP A08）。両フラグ（本関数・
/// `set_metal_onnx_gpu_execution_enabled`〈macOS 限定 cfg のため非
/// macOS ビルドでは存在せずリンク化しない〉）が有効な場合は CUDA を
/// 優先する（評価順固定）。対象 op・数値契約（REQ-2 統一複合判定）は
/// `crate::interop::onnx::OnnxModel` のドキュメンテーションコメントを
/// 正とする。
pub fn set_cuda_onnx_gpu_execution_enabled(enabled: bool) {
    crate::interop::onnx::set_cuda_onnx_gpu_execution_enabled(enabled);
}

/// [`set_cuda_onnx_gpu_execution_enabled`] で設定した現在の opt-in 状態を
/// 返す（既定 `false`）。
pub fn cuda_onnx_gpu_execution_enabled() -> bool {
    crate::interop::onnx::cuda_onnx_gpu_execution_enabled()
}

/// [`set_cuda_onnx_gpu_execution_enabled`] の Metal 版（macOS 限定。
/// イシュー #2077）。既定・fail-closed 方針は同一で、有効化すると
/// `fandhe_ai_backend_metal::MetalBackendOps` 経由の device 実行を試みる。
#[cfg(target_os = "macos")]
pub fn set_metal_onnx_gpu_execution_enabled(enabled: bool) {
    crate::interop::onnx::set_metal_onnx_gpu_execution_enabled(enabled);
}

/// [`set_metal_onnx_gpu_execution_enabled`] で設定した現在の opt-in 状態を
/// 返す（既定 `false`。macOS 限定）。
#[cfg(target_os = "macos")]
pub fn metal_onnx_gpu_execution_enabled() -> bool {
    crate::interop::onnx::metal_onnx_gpu_execution_enabled()
}

/// `Var`／`Sequential` の `custom`／`add_custom` と `Tape::add_custom`
/// （ユーザー定義 forward／backward 抽象の合成入口。イシュー #2064
/// §12.5）が facade の公開面から到達不能であることを、
/// `crates/autodiff/tests/architecture_boundaries.rs`・`crates/facade/
/// tests/api_surface.rs` のソース文字列走査（heuristics）とは独立に、
/// **コンパイラそのもの**で固定するための非公開足場
/// （`docs/autodiff-custom-function-decision.md` 「否定ガードの多層防御」
/// 節参照）。
///
/// # 方針転換の経緯（2026-09。PR #2212 codex-review 指摘）
///
/// 旧実装は `custom`／`add_custom` を実際に呼び出す 3 本の
/// ```` ```compile_fail,E0599 ```` doctest ブロックだった。しかし
/// **stable rustdoc（実測 rustc 1.98.1）は `compile_fail,EXXXX` の
/// エラーコードを一切照合しない**（`E0599` を無関係な `E0308` へ書き
/// 換えても doctest は合格した。実測に基づく事実）。この特性下では、
/// facade の公開面へ引数付きシグネチャで `custom`／`add_custom` が
/// 漏れ出しても `.custom()` の呼び出しは「引数の個数が違う」
/// （E0061）・「モジュールが見つからない」（E0432）・「glob の衝突で
/// 名前が曖昧」（E0659）等、`E0599` 以外の何らかのエラーで**依然として
/// コンパイルに失敗する**ため、`compile_fail`（エラーコード不問）は
/// 空合格してしまい、部分公開を検出できない。
///
/// そこで本足場は「失敗するはずの例を用意する」方式から「**成功する
/// はずの正のプローブを 1 つ用意し、それが実際にコンパイルできること**
/// を固定する」方式へ転換した。原理: facade の全 `pub mod`（ネスト含む）
/// を glob import したスコープに、ローカルにのみ存在する
/// `__FandheHoldProbe` トレイト（`custom`／`add_custom` という名前の
/// メソッドを持つ）を用意し、`Var`／`Tape`／`compat::Sequential` へ実装
/// したうえで、各型に対しメソッド形・型パス形の両方で呼び出す。facade
/// 側にこれと同名の実体（trait 経由の公開・inherent メソッドとしての
/// 転送メソッド追加のいずれであっても）が漏れ出すと、
/// - 別の trait が同名メソッドを提供する形の漏れ → 呼び出しが
///   複数のトレイト実装のどちらを指すか一意に定まらず曖昧になり、
///   エラーコードに依存せず必ずコンパイルが失敗する
/// - facade 独自の inherent メソッドとして漏れる形の漏れ（`Tape::custom`
///   への転送メソッド追加。§12.4「入口」参照）→ inherent メソッドが
///   優先解決され、戻り値の型がプローブの期待型と一致せず型不一致で
///   必ずコンパイルが失敗する
/// のいずれかとなり、**特定のエラーコードに依存せず**部分公開を
/// fail-closed に検出できる。1 ブロックにまとめても検出できるため、
/// 旧実装が抱えていた「3 種の禁止呼び出しを独立ブロックへ分割する
/// 必要性」（部分公開の見逃し防止）はこの転換によって解消された。
///
/// **本プローブ単独の既知の限界**: `&Var`（`Var` そのものではなく
/// その参照型）に対する 2 段目の autoref を経由した trait 実装、および
/// facade 型が内部型への `Deref` を実装した場合のフィールドアクセス
/// 経由の到達は、本プローブの呼び出し形（メソッド形・型パス形の
/// いずれも `Var`／`Tape`／`compat::Sequential` 自体に対する呼び出し）
/// だけでは拾いきれない可能性がある。この限界は、[`crates/facade/
/// tests/api_surface.rs::workspace_declares_custom_fn_only_on_tape_and_facade_delegation`]
/// （workspace 全体を対象に `fn custom`／`fn add_custom` の**定義元**を
/// インベントリする多層防御）が、facade からの到達可能性とは独立に
/// 「そもそも `crates/autodiff/src/tape.rs`（`Tape::custom`）以外の
/// どこにも `custom`／`add_custom` を定義させない」という定義元側の
/// 制約で補完する。
///
/// glob import する `pub mod` 集合（下記 doctest 内の `use` 一覧）と
/// 本クレートの実際の `pub mod` 宣言（ネスト含めて `src/` 全体を再帰
/// 走査した集合）がドリフトしないことは `crates/facade/tests/
/// api_surface.rs::custom_function_hold_doctest_globs_all_pub_modules`
/// が機械的に固定する（`collect_public_module_paths` が `lib.rs` の
/// `pub mod` 宣言から解決先ファイルを再帰的にたどり `nn::rnn` 等の
/// ネストしたパスも含めた集合を得たうえで doctest 内の `use
/// fandhe_ai::<mod>::*;` の集合と突き合わせる）。加えて本ブロックの
/// フェンス・本文が固定文言からドリフトしていないことは同ファイルの
/// `custom_function_hold_doctest_probe_body_matches_fixed_contract`
/// が固定する。`crates/facade/tests/api_surface.
/// rs::facade_source_declares_no_custom_fn_in_any_context` は、可視性
/// キーワード・宣言文脈（inherent impl・trait impl・trait 定義・自由
/// 関数のいずれか）を問わず facade 全ソースに `fn custom`／
/// `fn add_custom` 宣言が存在しないことを固定する（本 doctest 内の
/// トレイト・関数宣言は `///` コメントの中身のため、コメント除去後の
/// トークン走査には現れず誤検出しない）。
///
/// `Tape::custom` はイシュー #2549（ルート #2499 の一括承認・
/// `docs/autodiff-custom-function-decision.md` §16.1）で承認形のまま公開済み
/// のため、`__probe_tape` から `Tape::custom` の 2 行だけを外した。本ガードは
/// 承認範囲外（`Var`／`Sequential` の `custom`／`add_custom`・
/// `Tape::add_custom`）の非公開を固定し続ける（§16.4）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// struct __FandheHoldMarker;
///
/// trait __FandheHoldProbe {
///     fn custom(&self) -> __FandheHoldMarker;
///     fn add_custom(&self) -> __FandheHoldMarker;
/// }
///
/// impl<'t> __FandheHoldProbe for fandhe_ai::Var<'t> {
///     fn custom(&self) -> __FandheHoldMarker {
///         __FandheHoldMarker
///     }
///     fn add_custom(&self) -> __FandheHoldMarker {
///         __FandheHoldMarker
///     }
/// }
///
/// impl __FandheHoldProbe for fandhe_ai::Tape {
///     fn custom(&self) -> __FandheHoldMarker {
///         __FandheHoldMarker
///     }
///     fn add_custom(&self) -> __FandheHoldMarker {
///         __FandheHoldMarker
///     }
/// }
///
/// impl __FandheHoldProbe for fandhe_ai::compat::Sequential {
///     fn custom(&self) -> __FandheHoldMarker {
///         __FandheHoldMarker
///     }
///     fn add_custom(&self) -> __FandheHoldMarker {
///         __FandheHoldMarker
///     }
/// }
///
/// fn __probe_var(x: &fandhe_ai::Var<'_>) {
///     let _: __FandheHoldMarker = fandhe_ai::Var::custom(x);
///     let _: __FandheHoldMarker = x.custom();
///     let _: __FandheHoldMarker = fandhe_ai::Var::add_custom(x);
///     let _: __FandheHoldMarker = x.add_custom();
/// }
///
/// fn __probe_tape(x: &fandhe_ai::Tape) {
///     let _: __FandheHoldMarker = fandhe_ai::Tape::add_custom(x);
///     let _: __FandheHoldMarker = x.add_custom();
/// }
///
/// fn __probe_sequential(x: &fandhe_ai::compat::Sequential) {
///     let _: __FandheHoldMarker = fandhe_ai::compat::Sequential::custom(x);
///     let _: __FandheHoldMarker = x.custom();
///     let _: __FandheHoldMarker = fandhe_ai::compat::Sequential::add_custom(x);
///     let _: __FandheHoldMarker = x.add_custom();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct VarCustomHoldDoctestGuard;

/// イシュー #2141（親 #2131）の facade 公開保留を固定する doctest 足場。
/// #2510（ルート #2499 一括承認）で比較 6 種（`gt_bool`／`ge_bool`／
/// `lt_bool`／`le_bool`／`eq_bool`／`ne_bool`）と `masked_select` の 7 件は
/// `Var` の委譲メソッドとして公開済みのため、本ガードが固定するのは
/// 次の「承認形外」の配置に絞られる:
///
/// - logical 3 種（`logical_and`／`logical_or`／`logical_not`）の `Var`
///   への配置（公開形は未決・#2594）
/// - 10 関数の `Tensor<bool>`／`Tensor<f32>`／`Tape` 上への配置
/// - `bool_ops` モジュールの再エクスポート・別名 `pub use`
///
/// `VarCustomHoldDoctestGuard`／`NnModuleHoldDoctestGuard`（#2396 で削除
/// 済み）と同型の「正のプローブ 1 ブロック方式」を採る: facade の全
/// `pub mod` を glob import したスコープに、本ブロック内でのみ定義した
/// ローカルの自由関数群（`__fandhe_bool_hold_probe::bool_ops::*`）と
/// トレイト（`__FandheBoolHoldProbe`）を導入し、実際に使う関数を書く。
/// 未承認の経路でモジュール名・関数名が公開されると、ローカル定義との
/// glob 衝突（モジュール名の場合）または呼び出しシグネチャの不一致
/// （inherent メソッドがトレイトメソッドより優先解決されるため、引数なしの
/// `x.logical_and()` が inherent 側に解決されて型・引数数エラーになる）で
/// コンパイルが失敗する。`Var` 用トレイト impl の 7 件分は inherent に
/// 隠れて呼ばれないが、トレイトが 10 メソッドを要求するため残す。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// bool_ops_hold_doctest_globs_all_pub_modules`・`bool_ops_hold_
/// doctest_probe_body_matches_fixed_contract`・`facade_does_not_
/// reexport_or_declare_bool_ops`・`workspace_declares_bool_ops_fn_
/// names_in_approved_places_only`）との多層防御の位置づけは
/// `docs/autodiff-bool-ops-exposure-decision.md` §6「承認事項」・§6.1
/// を参照。
///
/// 撤去条件: #2594 で logical 3 種の公開形が承認・実装されたとき、本
/// モジュール・本 doctest 自体を削除する（ソース走査側と同時に外す）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_bool_hold_probe {
///     pub mod bool_ops {
///         pub fn gt_bool() {}
///         pub fn ge_bool() {}
///         pub fn lt_bool() {}
///         pub fn le_bool() {}
///         pub fn eq_bool() {}
///         pub fn ne_bool() {}
///         pub fn logical_and() {}
///         pub fn logical_or() {}
///         pub fn logical_not() {}
///         pub fn masked_select() {}
///     }
/// }
/// use __fandhe_bool_hold_probe::*;
///
/// struct __FandheBoolMarker;
///
/// trait __FandheBoolHoldProbe {
///     fn gt_bool(&self) -> __FandheBoolMarker;
///     fn ge_bool(&self) -> __FandheBoolMarker;
///     fn lt_bool(&self) -> __FandheBoolMarker;
///     fn le_bool(&self) -> __FandheBoolMarker;
///     fn eq_bool(&self) -> __FandheBoolMarker;
///     fn ne_bool(&self) -> __FandheBoolMarker;
///     fn logical_and(&self) -> __FandheBoolMarker;
///     fn logical_or(&self) -> __FandheBoolMarker;
///     fn logical_not(&self) -> __FandheBoolMarker;
///     fn masked_select(&self) -> __FandheBoolMarker;
/// }
///
/// impl<'t> __FandheBoolHoldProbe for fandhe_ai::Var<'t> {
///     fn gt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn ge_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn lt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn le_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn eq_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn ne_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn logical_and(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn logical_or(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn logical_not(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn masked_select(&self) -> __FandheBoolMarker { __FandheBoolMarker }
/// }
///
/// impl __FandheBoolHoldProbe for fandhe_ai::Tensor<bool> {
///     fn gt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn ge_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn lt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn le_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn eq_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn ne_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn logical_and(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn logical_or(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn logical_not(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn masked_select(&self) -> __FandheBoolMarker { __FandheBoolMarker }
/// }
///
/// impl __FandheBoolHoldProbe for fandhe_ai::Tensor<f32> {
///     fn gt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn ge_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn lt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn le_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn eq_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn ne_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn logical_and(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn logical_or(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn logical_not(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn masked_select(&self) -> __FandheBoolMarker { __FandheBoolMarker }
/// }
///
/// impl __FandheBoolHoldProbe for fandhe_ai::Tape {
///     fn gt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn ge_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn lt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn le_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn eq_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn ne_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn logical_and(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn logical_or(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn logical_not(&self) -> __FandheBoolMarker { __FandheBoolMarker }
///     fn masked_select(&self) -> __FandheBoolMarker { __FandheBoolMarker }
/// }
///
/// fn __probe_free_fns() {
///     // `bool_ops::` を経由した経路解決（`use fandhe_ai::*;` が
///     // 同名モジュールを glob 公開していれば、名前解決自体が
///     // 曖昧になり E0659 でコンパイル失敗する。バレ識別子の
///     // 未使用 glob 衝突は rustc が検出しないため、経路として
///     // 実際に `bool_ops` を解決させる必要がある）。
///     bool_ops::gt_bool();
///     bool_ops::ge_bool();
///     bool_ops::lt_bool();
///     bool_ops::le_bool();
///     bool_ops::eq_bool();
///     bool_ops::ne_bool();
///     bool_ops::logical_and();
///     bool_ops::logical_or();
///     bool_ops::logical_not();
///     bool_ops::masked_select();
/// }
///
/// fn __probe_var(x: &fandhe_ai::Var<'_>) {
///     let _: __FandheBoolMarker = fandhe_ai::Var::logical_and(x);
///     let _: __FandheBoolMarker = x.logical_and();
///     let _: __FandheBoolMarker = fandhe_ai::Var::logical_or(x);
///     let _: __FandheBoolMarker = x.logical_or();
///     let _: __FandheBoolMarker = fandhe_ai::Var::logical_not(x);
///     let _: __FandheBoolMarker = x.logical_not();
/// }
///
/// fn __probe_tensor_bool(x: &fandhe_ai::Tensor<bool>) {
///     let _: __FandheBoolMarker = fandhe_ai::Tensor::logical_and(x);
///     let _: __FandheBoolMarker = x.logical_and();
///     let _: __FandheBoolMarker = fandhe_ai::Tensor::logical_not(x);
///     let _: __FandheBoolMarker = x.logical_not();
/// }
///
/// fn __probe_tensor_f32(x: &fandhe_ai::Tensor<f32>) {
///     let _: __FandheBoolMarker = fandhe_ai::Tensor::gt_bool(x);
///     let _: __FandheBoolMarker = x.gt_bool();
/// }
///
/// fn __probe_tape(x: &fandhe_ai::Tape) {
///     let _: __FandheBoolMarker = fandhe_ai::Tape::gt_bool(x);
///     let _: __FandheBoolMarker = x.gt_bool();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct VarBoolOpsHoldDoctestGuard;

/// イシュー #2139（親 #2138・#2131）の facade 公開保留を固定する doctest
/// 足場。`VarCustomHoldDoctestGuard`／`NnModuleHoldDoctestGuard`（#2396 で削除済み）／
/// `VarBoolOpsHoldDoctestGuard`（直前の宣言）と同型の「正のプローブ 1
/// ブロック方式」を採るが、本ガードは 2 種類の衝突プローブを併用する:
///
/// (a) 型・モジュール名の衝突プローブ（`NnModuleHoldDoctestGuard`〈#2396 で削除済み〉方式）。
/// facade の全 `pub mod` を glob import したスコープに、本ブロック内でのみ
/// 定義したローカル型（`__fandhe_hooks_hold_probe::{HookHandle,
/// ForwardHooked, ForwardHookCtx}`）とモジュール（`hooks`）を導入し、
/// 実際に使う関数を書く。facade が同名の型・モジュールを glob で公開
/// すると、名前解決が曖昧になり（E0659 等）コンパイルが失敗する。
///
/// (b) メソッド名の衝突プローブ（`VarBoolOpsHoldDoctestGuard` 方式）。
/// `register_forward_hook`／`register_backward_hook`／`register_hook`／
/// `remove_hook`／`remove_backward_hook` の 5 メソッドをトレイト
/// （`__FandheHooksHoldProbe`）として `fandhe_ai::Var<'t>`／
/// `fandhe_ai::Tape`／`fandhe_ai::compat::Sequential` に実装し、UFCS 形・
/// メソッド呼び出し形の両方で呼ぶ。inherent メソッドはトレイトメソッド
/// より優先して解決されるため、これらの型に同名の inherent メソッドが
/// 追加されると、引数の数や型の不一致でコンパイルが失敗する。
///
/// facade の `Tape` は `pub struct Tape(pub(crate) fandhe_ai_autodiff::Tape)`
/// という newtype で `Deref` を持たないため（`crate::tape::Tape` 参照）、
/// 本プローブが検出できるのは facade 側に追加されたメソッドのみである。
/// autodiff 側の `Tape` に追加された定義は `crates/facade/tests/
/// api_surface.rs::workspace_declares_no_hook_registration_fns`（workspace
/// 全体のソース走査）が捕捉する分担とする。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// hooks_hold_doctest_globs_all_pub_modules`・`hooks_hold_doctest_probe_
/// body_matches_fixed_contract`・`workspace_declares_no_hook_registration_
/// fns`）との多層防御の位置づけ・承認未取得の経緯は
/// `docs/autodiff-forward-backward-hooks-design.md` §13「実装保留記録
/// （イシュー #2139）」を参照。
///
/// 承認（設計 doc §11 の 5 項目）を得て facade 公開を実施する日が来たら、
/// 本モジュール・本 doctest 自体を削除する（ソース走査側の対応する
/// 否定ガードと同時に外す）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_hooks_hold_probe {
///     pub struct HookHandle;
///     pub struct ForwardHooked;
///     pub struct ForwardHookCtx;
///     pub mod hooks {
///         pub fn __probe() {}
///     }
/// }
/// use __fandhe_hooks_hold_probe::*;
///
/// fn __probe_types(_: HookHandle, _: ForwardHooked, _: ForwardHookCtx) {
///     hooks::__probe();
/// }
///
/// struct __FandheHooksMarker;
///
/// trait __FandheHooksHoldProbe {
///     fn register_forward_hook(&self) -> __FandheHooksMarker;
///     fn register_backward_hook(&self) -> __FandheHooksMarker;
///     fn register_hook(&self) -> __FandheHooksMarker;
///     fn remove_hook(&self) -> __FandheHooksMarker;
///     fn remove_backward_hook(&self) -> __FandheHooksMarker;
/// }
///
/// impl<'t> __FandheHooksHoldProbe for fandhe_ai::Var<'t> {
///     fn register_forward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
///     fn register_backward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
///     fn register_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
///     fn remove_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
///     fn remove_backward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
/// }
///
/// impl __FandheHooksHoldProbe for fandhe_ai::Tape {
///     fn register_forward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
///     fn register_backward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
///     fn register_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
///     fn remove_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
///     fn remove_backward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
/// }
///
/// impl __FandheHooksHoldProbe for fandhe_ai::compat::Sequential {
///     fn register_forward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
///     fn register_backward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
///     fn register_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
///     fn remove_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
///     fn remove_backward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }
/// }
///
/// fn __probe_var(x: &fandhe_ai::Var<'_>) {
///     let _: __FandheHooksMarker = fandhe_ai::Var::register_forward_hook(x);
///     let _: __FandheHooksMarker = x.register_forward_hook();
///     let _: __FandheHooksMarker = fandhe_ai::Var::register_backward_hook(x);
///     let _: __FandheHooksMarker = x.register_backward_hook();
///     let _: __FandheHooksMarker = fandhe_ai::Var::register_hook(x);
///     let _: __FandheHooksMarker = x.register_hook();
///     let _: __FandheHooksMarker = fandhe_ai::Var::remove_hook(x);
///     let _: __FandheHooksMarker = x.remove_hook();
///     let _: __FandheHooksMarker = fandhe_ai::Var::remove_backward_hook(x);
///     let _: __FandheHooksMarker = x.remove_backward_hook();
/// }
///
/// fn __probe_tape(x: &fandhe_ai::Tape) {
///     let _: __FandheHooksMarker = fandhe_ai::Tape::register_forward_hook(x);
///     let _: __FandheHooksMarker = x.register_forward_hook();
///     let _: __FandheHooksMarker = fandhe_ai::Tape::register_backward_hook(x);
///     let _: __FandheHooksMarker = x.register_backward_hook();
///     let _: __FandheHooksMarker = fandhe_ai::Tape::register_hook(x);
///     let _: __FandheHooksMarker = x.register_hook();
///     let _: __FandheHooksMarker = fandhe_ai::Tape::remove_hook(x);
///     let _: __FandheHooksMarker = x.remove_hook();
///     let _: __FandheHooksMarker = fandhe_ai::Tape::remove_backward_hook(x);
///     let _: __FandheHooksMarker = x.remove_backward_hook();
/// }
///
/// fn __probe_sequential(x: &fandhe_ai::compat::Sequential) {
///     let _: __FandheHooksMarker = fandhe_ai::compat::Sequential::register_forward_hook(x);
///     let _: __FandheHooksMarker = x.register_forward_hook();
///     let _: __FandheHooksMarker = fandhe_ai::compat::Sequential::register_backward_hook(x);
///     let _: __FandheHooksMarker = x.register_backward_hook();
///     let _: __FandheHooksMarker = fandhe_ai::compat::Sequential::register_hook(x);
///     let _: __FandheHooksMarker = x.register_hook();
///     let _: __FandheHooksMarker = fandhe_ai::compat::Sequential::remove_hook(x);
///     let _: __FandheHooksMarker = x.remove_hook();
///     let _: __FandheHooksMarker = fandhe_ai::compat::Sequential::remove_backward_hook(x);
///     let _: __FandheHooksMarker = x.remove_backward_hook();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct VarHooksHoldDoctestGuard;

/// イシュー #2084（親 #2059。設計正本 `docs/kv-cache-design.md` §6
/// 承認事項 2・§10）の facade 公開（K-2: `KvCache`／`StatefulAttention`・
/// `compat::Sequential::add_stateful_attention` 相当）保留を固定する
/// doctest 足場。`VarCustomHoldDoctestGuard`／`NnModuleHoldDoctestGuard`（#2396 で削除済み）／
/// `VarBoolOpsHoldDoctestGuard`（直前の宣言）と同型の「正のプローブ 1
/// ブロック方式」を採る: facade の全 `pub mod` を glob import したスコープ
/// に、本ブロック内でのみ定義したローカル `__fandhe_kv_hold_probe::
/// {KvCache, StatefulAttention, add_stateful_attention}` を導入し、実際に
/// 使う関数を書く。facade がどの経路（単一行・複数行・ネストした group
/// での `pub use`・別名エクスポート・facade 独自の `struct`／`type` 宣言・
/// `compat::Sequential` への inherent メソッド追加）で `KvCache`／
/// `StatefulAttention`／`add_stateful_attention` という名前を公開しても、
/// ローカル定義との glob 衝突（型名の場合。E0659 等）または呼び出し
/// シグネチャの不一致（`compat::Sequential` への inherent メソッドは
/// トレイトメソッドより優先解決されるため、本プローブの trait 経由呼び出し
/// が型・引数不一致でコンパイル失敗する）でエラーコードに依存せず
/// コンパイルが失敗する。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// facade_does_not_expose_kv_cache_stateful_attention`〈直列 1 行走査。
/// 内側の層として維持〉・`facade_does_not_reexport_or_declare_kv_cache_
/// items`〈トークン方式。複数行・別名・独自宣言を検出〉）との多層防御の
/// 位置づけ・承認未取得の経緯・承認依頼用の K-2 事前設計は
/// `docs/kv-cache-design.md` §10「facade 公開（K-2）の
/// 保留固定と承認依頼用の事前設計」を参照。本 doctest が glob import する
/// `pub mod` 集合と `src/lib.rs` の実宣言集合のドリフトは
/// `kv_cache_hold_doctest_globs_all_pub_modules` が、本文（glob 以外）の
/// 固定文言からのドリフトは `kv_cache_hold_doctest_probe_body_matches_
/// fixed_contract` が固定する（`extract_hold_doctest_guard_doc`・
/// [`KV_CACHE_HOLD_PROBE_BODY`] 参照）。
///
/// K-2 facade 公開（`docs/kv-cache-design.md` §6 承認事項 2）がユーザー
/// 承認され実施する日が来たら、本モジュール・本 doctest 自体を削除する
/// （ソース走査側の対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_kv_hold_probe {
///     pub struct KvCache;
///     pub struct StatefulAttention;
///     pub struct __FandheKvHoldMarker;
///     pub fn add_stateful_attention() -> __FandheKvHoldMarker {
///         __FandheKvHoldMarker
///     }
/// }
/// use __fandhe_kv_hold_probe::*;
///
/// trait __FandheKvHoldProbe {
///     fn add_stateful_attention(&self) -> __FandheKvHoldMarker;
/// }
///
/// impl __FandheKvHoldProbe for fandhe_ai::compat::Sequential {
///     fn add_stateful_attention(&self) -> __FandheKvHoldMarker {
///         __FandheKvHoldMarker
///     }
/// }
///
/// fn __probe(_: KvCache, _: StatefulAttention, x: &fandhe_ai::compat::Sequential) {
///     let _: __FandheKvHoldMarker = add_stateful_attention();
///     let _: __FandheKvHoldMarker = fandhe_ai::compat::Sequential::add_stateful_attention(x);
///     let _: __FandheKvHoldMarker = x.add_stateful_attention();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct KvCacheHoldDoctestGuard;

/// イシュー #2146（親 #2131）の facade 公開のうち、承認形外の配置を引き続き
/// 固定する doctest 足場（#2516 で `Var` 委譲メソッド 5 件を公開したため部分反転。
/// `VarBoolOpsHoldDoctestGuard` と同じ #2510 型の部分反転）。承認形は
/// `Var::{mish, hardtanh, relu6, prelu, glu}` の 1 行委譲のみで、その正ガードは
/// `crates/facade/tests/api_surface.rs::var_activation_ops_methods_are_thin_delegations`・
/// `var_activation_ops_are_reachable_via_facade_only` が担う。本ガードは 2 種類の
/// 衝突プローブを併用する（`VarHooksHoldDoctestGuard` と同じ理由）:
///
/// (a) 承認形外の配置の衝突プローブ。facade の全 `pub mod` を glob import した
/// スコープに、本ブロック内でのみ定義したローカルの自由関数群
/// （`__fandhe_activation_hold_probe::activation_ops::{mish, hardtanh,
/// relu6, prelu, glu}`）とトレイト（`__FandheActivationHoldProbe`）を
/// 導入し、`Tensor<f32>`／`Tape` に実装して実際に使う。facade が
/// `activation_ops` というモジュール名や 5 個の関数名を `Tensor<f32>`／`Tape`
/// 上へ公開しても、ローカル定義との glob 衝突、または呼び出しシグネチャの
/// 不一致（inherent メソッドがトレイトメソッドより優先解決されるため）で
/// コンパイルが失敗する。`Var` への実装と `__probe_var` は、`Var` に inherent
/// メソッドが載ったことで引数なし呼び出しが inherent 側へ解決され型エラーに
/// なるため削除した。
///
/// (b) `compat::Sequential::add_*` 5 種は #2529 で承認形（`add_mish`・
/// `add_hardtanh`・`add_relu6`・`add_glu`・`add_prelu`）として公開済みのため、
/// 衝突プローブ（`__FandheActivationAddProbe`）は撤去した。承認形の正ガードは
/// `crates/facade/tests/api_surface.rs::
/// compat_sequential_activation_layers_add_methods_have_approved_signatures` が担う。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// activation_ops_hold_doctest_globs_all_pub_modules`・
/// `activation_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_activation_ops`・
/// `workspace_declares_activation_ops_fn_names_only_in_approved_locations`）
/// との多層防御の位置づけは `docs/autodiff-activation-ops-decision.md` §6
/// 「承認事項」を参照。
///
/// 撤去条件: `activation_ops` モジュールの再エクスポート・`Tensor<f32>`／`Tape` 上への
/// 配置が承認されたら、本 doctest 自体を削除する（ソース走査側の対応する否定ガードと
/// 同時に外す）。`compat::Sequential::add_*` の部分は #2529 で正ガードへ反転済み
/// （#2528 の `DropoutEmbeddingBagHoldDoctestGuard` 縮小と同じ部分反転）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_activation_hold_probe {
///     pub mod activation_ops {
///         pub fn mish() {}
///         pub fn hardtanh() {}
///         pub fn relu6() {}
///         pub fn prelu() {}
///         pub fn glu() {}
///     }
/// }
/// use __fandhe_activation_hold_probe::*;
///
/// struct __FandheActivationMarker;
///
/// trait __FandheActivationHoldProbe {
///     fn mish(&self) -> __FandheActivationMarker;
///     fn hardtanh(&self) -> __FandheActivationMarker;
///     fn relu6(&self) -> __FandheActivationMarker;
///     fn prelu(&self) -> __FandheActivationMarker;
///     fn glu(&self) -> __FandheActivationMarker;
/// }
///
/// impl __FandheActivationHoldProbe for fandhe_ai::Tensor<f32> {
///     fn mish(&self) -> __FandheActivationMarker { __FandheActivationMarker }
///     fn hardtanh(&self) -> __FandheActivationMarker { __FandheActivationMarker }
///     fn relu6(&self) -> __FandheActivationMarker { __FandheActivationMarker }
///     fn prelu(&self) -> __FandheActivationMarker { __FandheActivationMarker }
///     fn glu(&self) -> __FandheActivationMarker { __FandheActivationMarker }
/// }
///
/// impl __FandheActivationHoldProbe for fandhe_ai::Tape {
///     fn mish(&self) -> __FandheActivationMarker { __FandheActivationMarker }
///     fn hardtanh(&self) -> __FandheActivationMarker { __FandheActivationMarker }
///     fn relu6(&self) -> __FandheActivationMarker { __FandheActivationMarker }
///     fn prelu(&self) -> __FandheActivationMarker { __FandheActivationMarker }
///     fn glu(&self) -> __FandheActivationMarker { __FandheActivationMarker }
/// }
///
/// fn __probe_free_fns() {
///     // `activation_ops::` を経由した経路解決（`use fandhe_ai::*;` が
///     // 同名モジュールを glob 公開していれば、名前解決自体が曖昧に
///     // なり E0659 でコンパイル失敗する）。
///     activation_ops::mish();
///     activation_ops::hardtanh();
///     activation_ops::relu6();
///     activation_ops::prelu();
///     activation_ops::glu();
/// }
///
/// fn __probe_tensor_f32(x: &fandhe_ai::Tensor<f32>) {
///     let _: __FandheActivationMarker = fandhe_ai::Tensor::hardtanh(x);
///     let _: __FandheActivationMarker = x.hardtanh();
/// }
///
/// fn __probe_tape(x: &fandhe_ai::Tape) {
///     let _: __FandheActivationMarker = fandhe_ai::Tape::prelu(x);
///     let _: __FandheActivationMarker = x.prelu();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct VarActivationOpsHoldDoctestGuard;

/// イシュー #2156（親 #2131）の facade 公開保留を固定する doctest 足場。
/// `VarMatrixOpsHoldDoctestGuard`（#2144。#2513 で削除済み）・`KvCacheHoldDoctestGuard`
/// （#2084）と同型の「正のプローブ 1 ブロック方式」を採る: facade の
/// 全 `pub mod` を glob import したスコープに、本ブロック内でのみ定義
/// したローカルの自由関数群（`__fandhe_rng_dist_hold_probe::{bernoulli,
/// multinomial, normal, Generator}`）とトレイト
/// （`__FandheRngDistHoldProbe`）を導入し、実際に使う関数・型を書く。
/// facade がどの経路（`pub use fandhe_ai_tensor_core::rng::{bernoulli,
/// multinomial, normal, Generator};` のような再エクスポート・`Tensor`／
/// `Var` への inherent メソッド追加・別名 `pub use`・facade 独自の
/// `struct Generator` 宣言）でこれらの名前を公開しても、ローカル定義
/// との glob 衝突（自由関数・型名の場合。E0659 等）または呼び出し
/// シグネチャの不一致（inherent メソッドがトレイトメソッドより優先
/// 解決されるため、引数なしの `x.bernoulli()` 呼び出しが実際のシグネチャ
/// （`Tensor<f32>::bernoulli(probs: &Tensor<f32>)` 等）と型・引数数不一致
/// になる）でコンパイルが失敗する。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// rng_distributions_hold_doctest_globs_all_pub_modules`・`rng_
/// distributions_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_rng_distributions`・
/// `workspace_declares_rng_distribution_names_only_in_allowed_locations`）
/// との多層防御の位置づけ・承認未取得の経緯は
/// `docs/rng-distributions-generator-decision.md` §「facade 保留と承認
/// 依頼用の事前設計」を参照。
///
/// **注意**: `normal` は `crates/autodiff/src/nn/init.rs` の
/// `nn::init::normal`（PyTorch `nn.init.normal_` 相当）と同名の別機能で
/// ある。`nn::init` はイシュー #2504 で先に facade 公開されたため、自由
/// 関数 `normal` の衝突プローブは `nn::init` を除く全 `pub mod` を glob
/// した入れ子モジュール `__fandhe_rng_dist_normal_scope` に分離している
/// （`docs/rng-distributions-generator-decision.md` §5）。`tensor_core::
/// rng::normal` を公開する際は、この分離と `api_surface.rs` の
/// ソース走査の経路限定許可を合わせて撤去する。
///
/// facade 公開（ユーザー承認）がされる日が来たら、本モジュール・本
/// doctest 自体を削除する（ソース走査側の対応する否定ガードも同時に
/// 正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_rng_dist_hold_probe {
///     pub struct Generator;
///     pub fn bernoulli() {}
///     pub fn multinomial() {}
/// }
/// use __fandhe_rng_dist_hold_probe::*;
///
/// struct __FandheRngDistMarker;
///
/// trait __FandheRngDistHoldProbe {
///     fn bernoulli(&self) -> __FandheRngDistMarker;
///     fn multinomial(&self) -> __FandheRngDistMarker;
///     fn normal(&self) -> __FandheRngDistMarker;
/// }
///
/// impl<'t> __FandheRngDistHoldProbe for fandhe_ai::Var<'t> {
///     fn bernoulli(&self) -> __FandheRngDistMarker { __FandheRngDistMarker }
///     fn multinomial(&self) -> __FandheRngDistMarker { __FandheRngDistMarker }
///     fn normal(&self) -> __FandheRngDistMarker { __FandheRngDistMarker }
/// }
///
/// impl __FandheRngDistHoldProbe for fandhe_ai::Tensor<f32> {
///     fn bernoulli(&self) -> __FandheRngDistMarker { __FandheRngDistMarker }
///     fn multinomial(&self) -> __FandheRngDistMarker { __FandheRngDistMarker }
///     fn normal(&self) -> __FandheRngDistMarker { __FandheRngDistMarker }
/// }
///
/// fn __probe_free_fns(_: Generator) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     bernoulli();
///     multinomial();
/// }
///
/// // `normal` だけは `nn::init::normal`（#2504 で公開済み）が同名の
/// // 別機能として facade に存在するため、`nn::init` を除く全 `pub mod`
/// // だけを glob したスコープで衝突検査する（`nn::init` を含めると
/// // 常に曖昧になる）。
/// mod __fandhe_rng_dist_normal_scope {
///     use fandhe_ai::*;
///     use fandhe_ai::compat::*;
///     use fandhe_ai::optim::*;
///     use fandhe_ai::data::*;
///     use fandhe_ai::nn::*;
///     use fandhe_ai::nn::rnn::*;
///     use fandhe_ai::interop::*;
///     use fandhe_ai::interop::onnx::*;
///     use fandhe_ai::interop::safetensors::*;
///     use fandhe_ai::model::*;
///
///     mod __fandhe_rng_dist_normal_probe {
///         pub fn normal() {}
///     }
///     use __fandhe_rng_dist_normal_probe::*;
///
///     pub fn __probe_normal() {
///         normal();
///     }
/// }
///
/// fn __probe_var(x: &fandhe_ai::Var<'_>) {
///     let _: __FandheRngDistMarker = fandhe_ai::Var::bernoulli(x);
///     let _: __FandheRngDistMarker = x.bernoulli();
///     let _: __FandheRngDistMarker = fandhe_ai::Var::normal(x);
///     let _: __FandheRngDistMarker = x.normal();
/// }
///
/// fn __probe_tensor_f32(x: &fandhe_ai::Tensor<f32>) {
///     let _: __FandheRngDistMarker = fandhe_ai::Tensor::multinomial(x);
///     let _: __FandheRngDistMarker = x.multinomial();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct RngDistributionsHoldDoctestGuard;

/// イシュー #2159（親 #2131。設計正本 `docs/autodiff-spatial-layers-
/// decision.md` §6 承認事項 1）の facade 公開保留を固定する doctest
/// 足場。`KvCacheHoldDoctestGuard`（#2084）と同型の「正のプローブ 1
/// ブロック方式」を採る。**イシュー #2522 で `add_upsample`／
/// `add_zero_pad2d`／`add_identity` の 3 メソッドは承認・公開済み**のため
/// メソッド／自由関数プローブから外した（型名 `Upsample`／`ZeroPad2d`／
/// `Identity` の再エクスポートは未承認のため型名プローブは維持する。
/// 公開済み 3 メソッドは `api_surface.rs::compat_sequential_exposes_
/// spatial_layer_add_methods_issue_2522` が正ガードで固定する）。残る保留は
/// `ConvTranspose1d`／`Unflatten` 系と 5 型名。facade の全 `pub mod` を
/// glob import したスコープに、本ブロック内でのみ定義したローカル
/// `__fandhe_spatial_hold_probe::{ConvTranspose1d, Upsample, ZeroPad2d,
/// Identity, Unflatten, add_conv_transpose1d, add_unflatten}` を導入し、実際に使う
/// 関数を書く。facade がどの経路（単一行・複数行・ネストした group で
/// の `pub use`・別名エクスポート・facade 独自の `struct`／`type`
/// 宣言・`compat::Sequential`／`Var` への inherent メソッド追加）でこれら
/// の名前を公開しても、ローカル定義との glob 衝突（型名の場合。E0659
/// 等）または呼び出しシグネチャの不一致（`compat::Sequential`／`Var`
/// への inherent メソッドはトレイトメソッドより優先解決されるため、
/// 本プローブの trait 経由呼び出しが型・引数不一致でコンパイル失敗
/// する）でエラーコードに依存せずコンパイルが失敗する。
///
/// **イシュー #2521・#2522 での更新**: 5 層すべての `add_*`／`Var` メソッド
/// （`add_conv_transpose1d`／`add_unflatten`／`add_upsample`／`add_zero_pad2d`／
/// `add_identity`・`Var::conv_transpose1d`／`Var::unflatten`）は承認形として
/// 公開済みのため、trait 経由プローブ（`__FandheSpatialAddProbe`・
/// `__FandheSpatialVarProbe`）は本ブロックから外した（承認形だけを許す正ガードは
/// `api_surface.rs` の `*_spatial_*` テスト群）。型名の衝突プローブと自由関数の
/// プローブは、型の再エクスポート・自由関数での公開が未承認のまま残るため維持する。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// spatial_layers_hold_doctest_globs_all_pub_modules`・`spatial_
/// layers_hold_doctest_probe_body_matches_fixed_contract`）との
/// 多層防御の位置づけ・承認未取得の経緯は
/// `docs/autodiff-spatial-layers-decision.md` §6「承認事項」節を参照。
///
/// 型の再エクスポート・自由関数での公開は引き続き未承認のため、型名・自由関数の
/// 衝突プローブのみを本 doctest に残す。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_spatial_hold_probe {
///     pub struct ConvTranspose1d;
///     pub struct Upsample;
///     pub struct ZeroPad2d;
///     pub struct Identity;
///     pub struct Unflatten;
///     pub struct __FandheSpatialHoldMarker;
///     pub fn add_conv_transpose1d() -> __FandheSpatialHoldMarker {
///         __FandheSpatialHoldMarker
///     }
///     pub fn add_unflatten() -> __FandheSpatialHoldMarker {
///         __FandheSpatialHoldMarker
///     }
/// }
/// use __fandhe_spatial_hold_probe::*;
///
/// fn __probe(
///     _: ConvTranspose1d,
///     _: Upsample,
///     _: ZeroPad2d,
///     _: Identity,
///     _: Unflatten,
/// ) {
///     let _: __FandheSpatialHoldMarker = add_conv_transpose1d();
///     let _: __FandheSpatialHoldMarker = add_unflatten();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct SpatialLayersHoldDoctestGuard;

/// イシュー #2158（親 #2131）の facade 公開保留を固定する doctest 足場。
/// イシュー #2524（ルート #2499 の一括承認）で `Var::conv3d`（委譲メソッド）と
/// `compat::Sequential::add_conv3d` の 2 形を公開したため、本ガードは
/// **承認済みの 2 形のプローブだけを外して縮小**し、承認外の形への衝突
/// プローブを残している（#2716 の `SpatialLayersHoldDoctestGuard` と同じ判断）。
/// `VarActivationOpsHoldDoctestGuard`（イシュー #2146）と同型の「正の
/// プローブ 1 ブロック方式」を採り、2 種類の衝突プローブを併用する:
///
/// (a) 自由関数の衝突プローブ（`conv3d_ops::conv3d`。`VarActivationOps
/// HoldDoctestGuard` の `__probe_free_fns` と同方式）。facade の全
/// `pub mod` を glob import したスコープに、本ブロック内でのみ定義した
/// ローカルの自由関数（`__fandhe_conv3d_hold_probe::conv3d_ops::
/// conv3d`）を導入し、`conv3d_ops::conv3d()` という経路解決で使う。
/// facade が `pub use fandhe_ai_autodiff::conv3d_ops;` のようなモジュール
/// 再エクスポートで `conv3d_ops` を公開すれば、ローカル定義との glob
/// 衝突（E0659）でコンパイルが失敗する（承認範囲外のため引き続き禁止）。
///
/// (b) `Tensor<f32>`／`Tape` への `conv3d` メソッド衝突プローブ
/// （`__FandheConv3dHoldProbe` トレイト。`VarActivationOpsHoldDoctestGuard`
/// の `__FandheActivationHoldProbe` と同方式）。`Tensor::conv3d`／
/// `Tape::conv3d` の inherent メソッドが追加されれば、トレイトメソッドより
/// 優先解決されるため戻り値型が `__FandheConv3dMarker` ではなくなり型エラー
/// になる（`Var::conv3d` は承認済みのためプローブから外した）。
///
/// 撤去した項目（イシュー #2524 で承認・公開済み。以降の正ガードは
/// `api_surface.rs` の `add_conv3d_signature_matches_approved_contract` 等が担う）:
/// `Var` への `__FandheConv3dHoldProbe` impl と `__probe_var`、
/// `__FandheConv3dAddProbe` トレイトと `__probe_sequential_add`。
///
/// facade の `nn` は `pub mod rnn;` にのみ固定済み
/// （`nn_mod_declares_only_rnn_submodule`）のため、`Conv3d` 型自体の
/// glob 衝突プローブは不要（`nn::Conv3d` を追加しても `nn` モジュール
/// 自体の公開面〈`pub mod rnn;` のみ〉には現れない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// conv3d_hold_doctest_globs_all_pub_modules`・`conv3d_hold_doctest_
/// probe_body_matches_fixed_contract`・`facade_declares_conv3d_names_only_
/// in_approved_form`・`workspace_declares_conv3d_fn_names_only_in_
/// allowed_locations`）との多層防御の位置づけは `docs/conv-ops-
/// design.md` §16.7 を参照。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_conv3d_hold_probe {
///     pub mod conv3d_ops {
///         pub fn conv3d() {}
///     }
/// }
/// use __fandhe_conv3d_hold_probe::*;
///
/// struct __FandheConv3dMarker;
///
/// trait __FandheConv3dHoldProbe {
///     fn conv3d(&self) -> __FandheConv3dMarker;
/// }
///
/// impl __FandheConv3dHoldProbe for fandhe_ai::Tensor<f32> {
///     fn conv3d(&self) -> __FandheConv3dMarker { __FandheConv3dMarker }
/// }
///
/// impl __FandheConv3dHoldProbe for fandhe_ai::Tape {
///     fn conv3d(&self) -> __FandheConv3dMarker { __FandheConv3dMarker }
/// }
///
/// fn __probe_free_fns() {
///     // `conv3d_ops::` を経由した経路解決（`use fandhe_ai::*;` が
///     // 同名モジュールを glob 公開していれば、名前解決自体が曖昧に
///     // なり E0659 でコンパイル失敗する）。
///     conv3d_ops::conv3d();
/// }
///
/// fn __probe_tensor_f32(x: &fandhe_ai::Tensor<f32>) {
///     let _: __FandheConv3dMarker = fandhe_ai::Tensor::conv3d(x);
///     let _: __FandheConv3dMarker = x.conv3d();
/// }
///
/// fn __probe_tape(x: &fandhe_ai::Tape) {
///     let _: __FandheConv3dMarker = fandhe_ai::Tape::conv3d(x);
///     let _: __FandheConv3dMarker = x.conv3d();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct VarConv3dHoldDoctestGuard;

/// イシュー #2160（親 #2131）の facade 公開保留のうち、イシュー #2527
/// （親 #2520・ルート #2499 の一括承認）の後も残る分を固定する doctest
/// 足場。`VarConv3dHoldDoctestGuard`（#2524 の縮小形）と同型の「正のプローブ
/// 1 ブロック方式」を採る。
///
/// イシュー #2527 で公開済みの `Var::adaptive_max_pool2d`／
/// `adaptive_max_pool1d`・`compat::Sequential::add_adaptive_max_pool2d`／
/// `add_adaptive_max_pool1d`／`add_global_pool` のプローブ（inherent メソッド
/// 追加を trait 経由呼び出しの不一致で検出していた
/// `__FandheAdaptiveMaxPoolAddProbe` と `__probe_var`／
/// `__probe_sequential_add`）は撤去した。これらの承認形（定義元集合・1 行
/// 委譲本体・シグネチャ・実値）は `crates/facade/tests/api_surface.rs` の
/// 正ガード群（`workspace_declares_adaptive_max_global_pool_facade_fn_names_
/// only_in_approved_locations` ほか）が固定する。
///
/// まだ承認していない `Tensor<f32>`／`Tape` への同名メソッド追加は、
/// `__FandheAdaptiveMaxPoolHoldProbe` トレイトの衝突プローブ（inherent
/// メソッドはトレイトメソッドより優先解決されるため、追加されれば戻り値型が
/// `__FandheAdaptiveMaxPoolMarker` ではなくなり型エラーになる）で引き続き
/// 禁止する。層型（`AdaptiveMaxPool2d`／`AdaptiveMaxPool1d`／`GlobalPool`）の
/// facade 再エクスポートは未承認で、`api_surface.rs` の
/// `facade_reexports_global_pool_mode_only_in_approved_shape` が固定する
/// （facade の `nn` は `pub mod rnn;` にのみ固定済みのため glob 衝突プローブは
/// 不要。`GlobalPoolMode` だけがルート再エクスポートの承認対象）。
///
/// ソース走査ガード（`api_surface.rs::adaptive_max_global_pool_hold_doctest_
/// globs_all_pub_modules`・`adaptive_max_global_pool_hold_doctest_probe_body_
/// matches_fixed_contract`）との多層防御の位置づけ・承認の経緯は
/// `docs/autodiff-adaptive-max-global-pool-decision.md` §6・§8 を参照。
///
/// `Tensor<f32>`／`Tape` への追加が承認される日が来たら、本モジュール・本
/// doctest 自体を削除する（対応する固定文言も同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// struct __FandheAdaptiveMaxPoolMarker;
///
/// trait __FandheAdaptiveMaxPoolHoldProbe {
///     fn adaptive_max_pool2d(&self) -> __FandheAdaptiveMaxPoolMarker;
///     fn adaptive_max_pool1d(&self) -> __FandheAdaptiveMaxPoolMarker;
/// }
///
/// impl __FandheAdaptiveMaxPoolHoldProbe for fandhe_ai::Tensor<f32> {
///     fn adaptive_max_pool2d(&self) -> __FandheAdaptiveMaxPoolMarker { __FandheAdaptiveMaxPoolMarker }
///     fn adaptive_max_pool1d(&self) -> __FandheAdaptiveMaxPoolMarker { __FandheAdaptiveMaxPoolMarker }
/// }
///
/// impl __FandheAdaptiveMaxPoolHoldProbe for fandhe_ai::Tape {
///     fn adaptive_max_pool2d(&self) -> __FandheAdaptiveMaxPoolMarker { __FandheAdaptiveMaxPoolMarker }
///     fn adaptive_max_pool1d(&self) -> __FandheAdaptiveMaxPoolMarker { __FandheAdaptiveMaxPoolMarker }
/// }
///
/// fn __probe_tensor_f32(x: &fandhe_ai::Tensor<f32>) {
///     let _: __FandheAdaptiveMaxPoolMarker = fandhe_ai::Tensor::adaptive_max_pool2d(x);
///     let _: __FandheAdaptiveMaxPoolMarker = x.adaptive_max_pool2d();
///     let _: __FandheAdaptiveMaxPoolMarker = fandhe_ai::Tensor::adaptive_max_pool1d(x);
///     let _: __FandheAdaptiveMaxPoolMarker = x.adaptive_max_pool1d();
/// }
///
/// fn __probe_tape(x: &fandhe_ai::Tape) {
///     let _: __FandheAdaptiveMaxPoolMarker = fandhe_ai::Tape::adaptive_max_pool2d(x);
///     let _: __FandheAdaptiveMaxPoolMarker = x.adaptive_max_pool2d();
///     let _: __FandheAdaptiveMaxPoolMarker = fandhe_ai::Tape::adaptive_max_pool1d(x);
///     let _: __FandheAdaptiveMaxPoolMarker = x.adaptive_max_pool1d();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct AdaptiveMaxGlobalPoolHoldDoctestGuard;

/// イシュー #2161（親 #2131。設計正本 `docs/autodiff-dropout-embedding-
/// bag-decision.md` §6 承認事項）の facade 公開保留のうち、イシュー #2528
/// （親 #2520・ルート #2499 の一括承認）の後も残る分を固定する doctest
/// 足場。`PixelShuffleHoldDoctestGuard`（#2526 の縮小形）と同型の「正の
/// プローブ 1 ブロック方式」を採る: facade の全 `pub mod` を glob
/// import したスコープに、本ブロック内でのみ定義したローカル
/// `__fandhe_dropout_embedding_bag_hold_probe::{Dropout2d, AlphaDropout,
/// EmbeddingBag, EmbeddingBagVars, add_dropout2d, add_alpha_dropout,
/// add_embedding_bag}` を導入し、実際に使う関数を書く。facade がどの経路
/// （`pub use`・別名エクスポート・facade 独自の `struct`／`type` 宣言・
/// 自由関数での公開）でこれらの名前を公開しても、ローカル定義との glob
/// 衝突でエラーコードに依存せずコンパイルが失敗する。
///
/// イシュー #2528 で公開済みの `compat::Sequential::add_dropout2d`／
/// `add_alpha_dropout`／`add_embedding_bag`・`Var::dropout2d`／
/// `alpha_dropout`／`embedding_bag`・ルートの `EmbeddingBagMode` 再エクスポート
/// のプローブ（inherent メソッド追加を trait 経由呼び出しの不一致で検出して
/// いた `__FandheDropoutEmbeddingBagAddProbe`／`__FandheDropoutEmbeddingBagVarProbe`
/// と `seq.*`／`v.*` の呼び出し・`EmbeddingBagMode` の型プローブ）は撤去した。
/// これらの承認形（定義元集合・1 行委譲本体・シグネチャ・再エクスポート形・
/// 実値）は `crates/facade/tests/api_surface.rs` の正ガード群
/// （`workspace_declares_dropout_embedding_bag_facade_fn_names_only_in_
/// approved_locations` ほか）が固定する。層型の再エクスポートと自由関数での
/// 公開は未承認のため、本プローブで引き続き禁止する。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// dropout_embedding_bag_hold_doctest_globs_all_pub_modules`・
/// `dropout_embedding_bag_hold_doctest_probe_body_matches_fixed_
/// contract`）との多層防御の位置づけ・承認の経緯は
/// `docs/autodiff-dropout-embedding-bag-decision.md` §6・§8 を参照。
///
/// 層型の再エクスポート・自由関数公開が承認される日が来たら、本モジュール・
/// 本 doctest 自体を削除する（対応する固定文言も同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_dropout_embedding_bag_hold_probe {
///     pub struct Dropout2d;
///     pub struct AlphaDropout;
///     pub struct EmbeddingBag;
///     pub struct EmbeddingBagVars;
///     pub struct __FandheDropoutEmbeddingBagHoldMarker;
///     pub fn add_dropout2d() -> __FandheDropoutEmbeddingBagHoldMarker {
///         __FandheDropoutEmbeddingBagHoldMarker
///     }
///     pub fn add_alpha_dropout() -> __FandheDropoutEmbeddingBagHoldMarker {
///         __FandheDropoutEmbeddingBagHoldMarker
///     }
///     pub fn add_embedding_bag() -> __FandheDropoutEmbeddingBagHoldMarker {
///         __FandheDropoutEmbeddingBagHoldMarker
///     }
/// }
/// use __fandhe_dropout_embedding_bag_hold_probe::*;
///
/// fn __probe(
///     _: Dropout2d,
///     _: AlphaDropout,
///     _: EmbeddingBag,
///     _: EmbeddingBagVars,
/// ) {
///     let _: __FandheDropoutEmbeddingBagHoldMarker = add_dropout2d();
///     let _: __FandheDropoutEmbeddingBagHoldMarker = add_alpha_dropout();
///     let _: __FandheDropoutEmbeddingBagHoldMarker = add_embedding_bag();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct DropoutEmbeddingBagHoldDoctestGuard;

/// イシュー #2164（親 #2131。設計正本 `docs/autodiff-rnn-stacked-config-
/// decision.md` §8）の facade 公開保留のうち、イシュー #2535（親 #2534・
/// ルート #2499 の一括承認）の後も残る分を固定する doctest 足場
/// （`PixelShuffleHoldDoctestGuard` の #2526 縮小形と同型）。
///
/// #2535 で公開済みのもの（`RnnConfig`・`Stacked*` 6 型の再エクスポートと
/// `Tape::{stacked_rnn_forward_seq, stacked_lstm_forward_seq,
/// stacked_gru_forward_seq}`）の保留プローブは撤去した。承認形は
/// `crates/facade/tests/api_surface.rs` の正ガード
/// （`nn_rnn_module_reexports_exactly_expected_surface`・
/// `nn_rnn_stacked_types_are_reachable_via_facade_only`・
/// `tape_stacked_rnn_methods_are_thin_delegations` 等）が固定する。
///
/// 残す保留は `fandhe_ai::nn::rnn::{Rnn, Lstm, Gru}::with_config`（既存型への
/// `RnnConfig` 引数付きコンストラクタ追加）のみ。決定記録 §2.1 で不採用、
/// §8 の承認範囲にも含まれない（trait 経由のプローブ呼び出しが inherent
/// メソッドの型・引数不一致でコンパイル失敗する）。
///
/// ソース走査ガード（`rnn_config_hold_doctest_globs_all_pub_modules`・
/// `rnn_config_hold_doctest_probe_body_matches_fixed_contract`・
/// `workspace_declares_no_rnn_with_config_fn`）との多層防御。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// struct __FandheRnnConfigMarker;
///
/// trait __FandheRnnConfigWithConfigProbe {
///     fn with_config(&self) -> __FandheRnnConfigMarker;
/// }
///
/// impl __FandheRnnConfigWithConfigProbe for fandhe_ai::nn::rnn::Rnn {
///     fn with_config(&self) -> __FandheRnnConfigMarker {
///         __FandheRnnConfigMarker
///     }
/// }
///
/// impl __FandheRnnConfigWithConfigProbe for fandhe_ai::nn::rnn::Lstm {
///     fn with_config(&self) -> __FandheRnnConfigMarker {
///         __FandheRnnConfigMarker
///     }
/// }
///
/// impl __FandheRnnConfigWithConfigProbe for fandhe_ai::nn::rnn::Gru {
///     fn with_config(&self) -> __FandheRnnConfigMarker {
///         __FandheRnnConfigMarker
///     }
/// }
///
/// fn __probe(
///     rnn: &fandhe_ai::nn::rnn::Rnn,
///     lstm: &fandhe_ai::nn::rnn::Lstm,
///     gru: &fandhe_ai::nn::rnn::Gru,
/// ) {
///     let _: __FandheRnnConfigMarker = fandhe_ai::nn::rnn::Rnn::with_config(rnn);
///     let _: __FandheRnnConfigMarker = rnn.with_config();
///     let _: __FandheRnnConfigMarker = fandhe_ai::nn::rnn::Lstm::with_config(lstm);
///     let _: __FandheRnnConfigMarker = lstm.with_config();
///     let _: __FandheRnnConfigMarker = fandhe_ai::nn::rnn::Gru::with_config(gru);
///     let _: __FandheRnnConfigMarker = gru.with_config();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct RnnConfigHoldDoctestGuard;

/// イシュー #2162（親 #2131）の facade 公開保留のうち、イシュー #2526
/// （親 #2520・ルート #2499 の一括承認）の後も残る分を固定する doctest
/// 足場。`SpatialLayersHoldDoctestGuard`（#2521 の縮小形）と同型の「正の
/// プローブ 1 ブロック方式」を採る: facade の全 `pub mod` を glob
/// import したスコープに、本ブロック内でのみ定義したローカル
/// `__fandhe_pixel_shuffle_hold_probe::{PixelShuffle, PixelUnshuffle,
/// add_pixel_shuffle, add_pixel_unshuffle}` を導入し、実際に使う関数を
/// 書く。facade がどの経路（`pub use`・別名エクスポート・facade 独自の
/// `struct`／`type` 宣言・自由関数での公開）でこれらの名前を公開しても、
/// ローカル定義との glob 衝突でエラーコードに依存せずコンパイルが失敗する。
///
/// イシュー #2526 で公開済みの `compat::Sequential::add_pixel_shuffle`／
/// `add_pixel_unshuffle`・`Var::pixel_shuffle`／`pixel_unshuffle` のプローブ
/// （inherent メソッド追加を trait 経由呼び出しの不一致で検出していた
/// `__FandhePixelShuffleAddProbe`／`__FandhePixelShuffleVarProbe` と
/// `seq.*`／`v.*` の呼び出し）は撤去した。これらの承認形（定義元集合・
/// 1 行委譲本体・シグネチャ・実値）は `crates/facade/tests/api_surface.rs`
/// の正ガード群（`workspace_declares_pixel_shuffle_facade_fn_names_only_in_
/// approved_locations` ほか）が固定する。型の再エクスポートと自由関数での
/// 公開は未承認のため、本プローブで引き続き禁止する。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// pixel_shuffle_hold_doctest_globs_all_pub_modules`・`pixel_shuffle_
/// hold_doctest_probe_body_matches_fixed_contract`）との多層防御の位置
/// づけ・承認の経緯は `docs/autodiff-pixel-shuffle-decision.md` §6 を参照。
///
/// 型の再エクスポート・自由関数公開が承認される日が来たら、本モジュール・
/// 本 doctest 自体を削除する（対応する固定文言も同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_pixel_shuffle_hold_probe {
///     pub struct PixelShuffle;
///     pub struct PixelUnshuffle;
///     pub struct __FandhePixelShuffleHoldMarker;
///     pub fn add_pixel_shuffle() -> __FandhePixelShuffleHoldMarker {
///         __FandhePixelShuffleHoldMarker
///     }
///     pub fn add_pixel_unshuffle() -> __FandhePixelShuffleHoldMarker {
///         __FandhePixelShuffleHoldMarker
///     }
/// }
/// use __fandhe_pixel_shuffle_hold_probe::*;
///
/// fn __probe(
///     _: PixelShuffle,
///     _: PixelUnshuffle,
/// ) {
///     let _: __FandhePixelShuffleHoldMarker = add_pixel_shuffle();
///     let _: __FandhePixelShuffleHoldMarker = add_pixel_unshuffle();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct PixelShuffleHoldDoctestGuard;

/// イシュー #2166（親 #2131）の facade 公開保留を固定する doctest 足場。
/// イシュー #2167（親 #2131）で、距離ベースの損失 3 種
/// （`cosine_embedding_loss`・`margin_ranking_loss`・
/// `triplet_margin_loss`）と `poisson_nll_loss` を同じ保留対象へ追加
/// した。イシュー #2168（親 #2131）で `ctc_loss`（CTC 損失）を追加した
/// （7 関数名。`crate::loss_ops` モジュール doc「facade 非公開
/// （意図的）」と同じ判断枠組み）。
/// `VarReduceOpsHoldDoctestGuard`（イシュー #2147。#2514 で削除済み）と同型の「正の
/// プローブ 1 ブロック方式」を採る: facade の全 `pub mod` を glob
/// import したスコープに、本ブロック内でのみ定義したローカルの自由
/// 関数群（`__fandhe_loss_hold_probe::loss_ops::{l1_loss,
/// cross_entropy_loss_with, cosine_embedding_loss, margin_ranking_loss,
/// triplet_margin_loss, poisson_nll_loss, ctc_loss}`）とトレイト
/// （`__FandheLossHoldProbe`）を導入し、実際に使う関数を書く。facade が
/// どの経路（`pub use fandhe_ai_autodiff::loss_ops;` のようなモジュール
/// 再エクスポート・`Var` への inherent メソッド追加・別名 `pub use`）で
/// `loss_ops` という名前や 7 個の関数名を公開しても、ローカル定義との
/// glob 衝突（モジュール名の場合）または呼び出しシグネチャの不一致
/// （inherent メソッドがトレイトメソッドより優先解決されるため、引数
/// なしの `x.cosine_embedding_loss()` 呼び出しが実際の inherent メソッド
/// 〈引数数不一致〉に解決されて型・引数数エラーになる）で
/// コンパイルが失敗する。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// loss_ops_hold_doctest_globs_all_pub_modules`・`loss_ops_hold_
/// doctest_probe_body_matches_fixed_contract`・`facade_does_not_
/// reexport_or_declare_loss_ops`・`workspace_declares_loss_ops_
/// fn_names_only_in_approved_locations`）との多層防御の位置づけは
/// `docs/autodiff-loss-ops-decision.md` §5「承認事項」を参照。
///
/// **#2538 での部分反転**（ルート #2499 一括承認。`VarActivationOpsHoldDoctestGuard`
/// の #2516 型）: `Var::l1_loss`・`Var::cross_entropy_loss_with` は委譲メソッド
/// として facade 公開済みのため、`__probe_var` から該当 4 行だけを削除した
/// （正ガードは `api_surface.rs::var_loss_ops_methods_are_thin_delegations`・
/// `var_loss_ops_are_reachable_via_facade_var`）。`loss_ops` モジュール・
/// `Tensor`／`Tape` 上の同名メソッド・残り 5 名の `Var` への公開は引き続き拒否する。
///
/// **#2539 での追加の部分反転**（同一の一括承認）: `Var::cosine_embedding_loss`・
/// `Var::margin_ranking_loss`・`Var::triplet_margin_loss`・`Var::poisson_nll_loss`
/// も委譲メソッドとして公開済みのため、`__probe_var` から該当 8 行を削除した
/// （正ガードは `var_loss_ops_methods_are_thin_delegations` 6 件化・
/// `var_distance_poisson_loss_ops_are_reachable_via_facade_var`）。`__probe_var` には
/// `ctc_loss` の 2 行のみが残っていた。
///
/// **#2540 での反転**（同一の一括承認）: `Var::ctc_loss` を委譲メソッドとして公開した
/// ため、`__probe_var` と `Var` 向けトレイト impl を撤去した（正ガードは
/// `var_loss_ops_methods_are_thin_delegations` 7 件化・`var_ctc_loss_is_reachable_via_facade_var`）。
///
/// 撤去条件: `Var` 側の撤去は #2540 で完了。`Tensor`／`Tape`／モジュール再エクスポート／
/// オプション型の拒否は維持する（解除には別途承認が必要）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_loss_hold_probe {
///     pub mod loss_ops {
///         pub fn l1_loss() {}
///         pub fn cross_entropy_loss_with() {}
///         pub fn cosine_embedding_loss() {}
///         pub fn margin_ranking_loss() {}
///         pub fn triplet_margin_loss() {}
///         pub fn poisson_nll_loss() {}
///         pub fn ctc_loss() {}
///     }
/// }
/// use __fandhe_loss_hold_probe::*;
///
/// struct __FandheLossMarker;
///
/// trait __FandheLossHoldProbe {
///     fn l1_loss(&self) -> __FandheLossMarker;
///     fn cross_entropy_loss_with(&self) -> __FandheLossMarker;
///     fn cosine_embedding_loss(&self) -> __FandheLossMarker;
///     fn margin_ranking_loss(&self) -> __FandheLossMarker;
///     fn triplet_margin_loss(&self) -> __FandheLossMarker;
///     fn poisson_nll_loss(&self) -> __FandheLossMarker;
///     fn ctc_loss(&self) -> __FandheLossMarker;
/// }
///
/// impl __FandheLossHoldProbe for fandhe_ai::Tensor<f32> {
///     fn l1_loss(&self) -> __FandheLossMarker { __FandheLossMarker }
///     fn cross_entropy_loss_with(&self) -> __FandheLossMarker { __FandheLossMarker }
///     fn cosine_embedding_loss(&self) -> __FandheLossMarker { __FandheLossMarker }
///     fn margin_ranking_loss(&self) -> __FandheLossMarker { __FandheLossMarker }
///     fn triplet_margin_loss(&self) -> __FandheLossMarker { __FandheLossMarker }
///     fn poisson_nll_loss(&self) -> __FandheLossMarker { __FandheLossMarker }
///     fn ctc_loss(&self) -> __FandheLossMarker { __FandheLossMarker }
/// }
///
/// impl __FandheLossHoldProbe for fandhe_ai::Tape {
///     fn l1_loss(&self) -> __FandheLossMarker { __FandheLossMarker }
///     fn cross_entropy_loss_with(&self) -> __FandheLossMarker { __FandheLossMarker }
///     fn cosine_embedding_loss(&self) -> __FandheLossMarker { __FandheLossMarker }
///     fn margin_ranking_loss(&self) -> __FandheLossMarker { __FandheLossMarker }
///     fn triplet_margin_loss(&self) -> __FandheLossMarker { __FandheLossMarker }
///     fn poisson_nll_loss(&self) -> __FandheLossMarker { __FandheLossMarker }
///     fn ctc_loss(&self) -> __FandheLossMarker { __FandheLossMarker }
/// }
///
/// fn __probe_free_fns() {
///     // `loss_ops::` を経由した経路解決（`use fandhe_ai::*;` が
///     // 同名モジュールを glob 公開していれば、名前解決自体が曖昧に
///     // なり E0659 でコンパイル失敗する）。
///     loss_ops::l1_loss();
///     loss_ops::cross_entropy_loss_with();
///     loss_ops::cosine_embedding_loss();
///     loss_ops::margin_ranking_loss();
///     loss_ops::triplet_margin_loss();
///     loss_ops::poisson_nll_loss();
///     loss_ops::ctc_loss();
/// }
///
/// fn __probe_tensor_f32(x: &fandhe_ai::Tensor<f32>) {
///     let _: __FandheLossMarker = fandhe_ai::Tensor::l1_loss(x);
///     let _: __FandheLossMarker = x.l1_loss();
///     let _: __FandheLossMarker = fandhe_ai::Tensor::cosine_embedding_loss(x);
///     let _: __FandheLossMarker = x.cosine_embedding_loss();
///     let _: __FandheLossMarker = fandhe_ai::Tensor::margin_ranking_loss(x);
///     let _: __FandheLossMarker = x.margin_ranking_loss();
///     let _: __FandheLossMarker = fandhe_ai::Tensor::ctc_loss(x);
///     let _: __FandheLossMarker = x.ctc_loss();
/// }
///
/// fn __probe_tape(x: &fandhe_ai::Tape) {
///     let _: __FandheLossMarker = fandhe_ai::Tape::cross_entropy_loss_with(x);
///     let _: __FandheLossMarker = x.cross_entropy_loss_with();
///     let _: __FandheLossMarker = fandhe_ai::Tape::triplet_margin_loss(x);
///     let _: __FandheLossMarker = x.triplet_margin_loss();
///     let _: __FandheLossMarker = fandhe_ai::Tape::poisson_nll_loss(x);
///     let _: __FandheLossMarker = x.poisson_nll_loss();
///     let _: __FandheLossMarker = fandhe_ai::Tape::ctc_loss(x);
///     let _: __FandheLossMarker = x.ctc_loss();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct LossOpsHoldDoctestGuard;

/// イシュー #2173（親 #2131。設計正本
/// `docs/autodiff-param-groups-decision.md` §5「承認事項」）の facade
/// 公開保留を固定する doctest 足場。`RnnConfigHoldDoctestGuard`
/// （イシュー #2164）と同型の「正のプローブ 1 ブロック方式」を採る:
/// facade の全 `pub mod` を glob import したスコープに、本ブロック内
/// でのみ定義したローカル `__fandhe_param_groups_hold_probe::
/// {ParamGroup, ParamGroupStep}` を導入する。facade がどの経路
/// （単一行・複数行・ネストした group での `pub use`・別名エクスポート）
/// でこれらの名前を公開しても、ローカル定義との glob 衝突（E0659）で
/// コンパイルが失敗する。glob 衝突は名前を実際に参照したときにだけ
/// 発生するため（未参照のままでは検出できない。イシュー #2304 の
/// レビュー指摘）、`ParamGroupStep` は `fn __probe_trait<T: ?Sized +
/// ParamGroupStep>() {}` という generic 境界で明示的に参照する。
///
/// あわせて `fandhe_ai::compat::Sequential::compile_with_param_groups`
/// （承認事項の設計案「`compile()` とは別の param_groups 対応エントリ」）
/// と `fandhe_ai::optim::{Sgd, AdamW, Adam, RmsProp, Adagrad, Lamb}::
/// step_with_groups`（`fandhe_ai_autodiff::nn::optim::ParamGroupStep`
/// の facade 側 inherent 再エクスポートに相当する経路）も同じブロックで
/// 保留固定する（trait 経由のプローブ呼び出しが inherent メソッドの
/// 型・引数不一致でコンパイル失敗する。`RnnConfigHoldDoctestGuard` の
/// `__FandheRnnConfigWithConfigProbe` と同方式）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// param_groups_hold_doctest_globs_all_pub_modules`・
/// `param_groups_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_param_groups`）との多層防御の
/// 位置づけ・承認未取得の経緯は
/// `docs/autodiff-param-groups-decision.md` §5「承認事項」節を参照。
///
/// 承認（facade への `ParamGroup`／`ParamGroupStep` 再エクスポート・
/// `compile_with_param_groups` 新設）を得た日が来たら、本モジュール・
/// 本 doctest 自体を削除する（ソース走査側の対応する否定ガードも
/// 同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_param_groups_hold_probe {
///     pub struct ParamGroup;
///     pub trait ParamGroupStep {}
/// }
/// use __fandhe_param_groups_hold_probe::*;
///
/// fn __probe_trait<T: ?Sized + ParamGroupStep>() {}
///
/// struct __FandheParamGroupsMarker;
///
/// trait __FandheParamGroupsCompileProbe {
///     fn compile_with_param_groups(&self) -> __FandheParamGroupsMarker;
/// }
///
/// impl __FandheParamGroupsCompileProbe for fandhe_ai::compat::Sequential {
///     fn compile_with_param_groups(&self) -> __FandheParamGroupsMarker {
///         __FandheParamGroupsMarker
///     }
/// }
///
/// trait __FandheParamGroupsStepProbe {
///     fn step_with_groups(&self) -> __FandheParamGroupsMarker;
/// }
///
/// impl __FandheParamGroupsStepProbe for fandhe_ai::optim::Sgd {
///     fn step_with_groups(&self) -> __FandheParamGroupsMarker {
///         __FandheParamGroupsMarker
///     }
/// }
///
/// impl __FandheParamGroupsStepProbe for fandhe_ai::optim::AdamW {
///     fn step_with_groups(&self) -> __FandheParamGroupsMarker {
///         __FandheParamGroupsMarker
///     }
/// }
///
/// impl __FandheParamGroupsStepProbe for fandhe_ai::optim::Adam {
///     fn step_with_groups(&self) -> __FandheParamGroupsMarker {
///         __FandheParamGroupsMarker
///     }
/// }
///
/// impl __FandheParamGroupsStepProbe for fandhe_ai::optim::RmsProp {
///     fn step_with_groups(&self) -> __FandheParamGroupsMarker {
///         __FandheParamGroupsMarker
///     }
/// }
///
/// impl __FandheParamGroupsStepProbe for fandhe_ai::optim::Adagrad {
///     fn step_with_groups(&self) -> __FandheParamGroupsMarker {
///         __FandheParamGroupsMarker
///     }
/// }
///
/// impl __FandheParamGroupsStepProbe for fandhe_ai::optim::Lamb {
///     fn step_with_groups(&self) -> __FandheParamGroupsMarker {
///         __FandheParamGroupsMarker
///     }
/// }
///
/// fn __probe(
///     _: ParamGroup,
///     seq: &fandhe_ai::compat::Sequential,
///     sgd: &fandhe_ai::optim::Sgd,
///     adamw: &fandhe_ai::optim::AdamW,
///     adam: &fandhe_ai::optim::Adam,
///     rmsprop: &fandhe_ai::optim::RmsProp,
///     adagrad: &fandhe_ai::optim::Adagrad,
///     lamb: &fandhe_ai::optim::Lamb,
/// ) {
///     let _: __FandheParamGroupsMarker =
///         fandhe_ai::compat::Sequential::compile_with_param_groups(seq);
///     let _: __FandheParamGroupsMarker = seq.compile_with_param_groups();
///     let _: __FandheParamGroupsMarker = fandhe_ai::optim::Sgd::step_with_groups(sgd);
///     let _: __FandheParamGroupsMarker = sgd.step_with_groups();
///     let _: __FandheParamGroupsMarker = fandhe_ai::optim::AdamW::step_with_groups(adamw);
///     let _: __FandheParamGroupsMarker = adamw.step_with_groups();
///     let _: __FandheParamGroupsMarker = fandhe_ai::optim::Adam::step_with_groups(adam);
///     let _: __FandheParamGroupsMarker = adam.step_with_groups();
///     let _: __FandheParamGroupsMarker = fandhe_ai::optim::RmsProp::step_with_groups(rmsprop);
///     let _: __FandheParamGroupsMarker = rmsprop.step_with_groups();
///     let _: __FandheParamGroupsMarker = fandhe_ai::optim::Adagrad::step_with_groups(adagrad);
///     let _: __FandheParamGroupsMarker = adagrad.step_with_groups();
///     let _: __FandheParamGroupsMarker = fandhe_ai::optim::Lamb::step_with_groups(lamb);
///     let _: __FandheParamGroupsMarker = lamb.step_with_groups();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct ParamGroupsHoldDoctestGuard;

/// イシュー #2174（親 #2131。設計正本 `docs/autodiff-optimizer-state-
/// dict-decision.md` §5「承認事項」）の facade 公開保留を固定する
/// doctest 足場。`ParamGroupsHoldDoctestGuard`（イシュー #2173）と同型の
/// 「正のプローブ 1 ブロック方式」を採る: facade の全 `pub mod` を
/// glob import したスコープに、本ブロック内でのみ定義したローカル
/// `__fandhe_optim_state_dict_hold_probe::OptimizerStateDict`（trait）を
/// 導入する。facade がどの経路（単一行・複数行・ネストした group での
/// `pub use`・別名エクスポート）でこの名前を公開しても、ローカル定義
/// との glob 衝突（E0659）でコンパイルが失敗する。glob 衝突は名前を
/// 実際に参照したときにだけ発生するため（未参照のままでは検出できない。
/// イシュー #2304 のレビュー指摘）、`OptimizerStateDict` は
/// `fn __probe_trait<T: ?Sized + OptimizerStateDict>() {}` という
/// generic 境界で明示的に参照する。
///
/// あわせて `fandhe_ai::optim::{AdamW, Adam, RmsProp, Adagrad, Lamb}::
/// state_dict`／`load_state_dict`（`fandhe_ai_autodiff::nn::optim::
/// OptimizerStateDict` の facade 側 inherent 再エクスポートに相当する
/// 経路）も同じブロックで保留固定する（trait 経由のプローブ呼び出しが
/// inherent メソッドの型・引数不一致でコンパイル失敗する。
/// `ParamGroupsHoldDoctestGuard` の `__FandheParamGroupsStepProbe` と
/// 同方式）。`compat::Sequential` は既存の inherent `state_dict`／
/// `load_state_dict` を持つためプローブ対象にしない。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// optimizer_state_dict_hold_doctest_globs_all_pub_modules`・
/// `optimizer_state_dict_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_optimizer_state_dict`）との
/// 多層防御の位置づけ・承認未取得の経緯は `docs/autodiff-optimizer-
/// state-dict-decision.md` §5「承認事項」節を参照。
///
/// 承認（facade への `OptimizerStateDict` 再エクスポート・各型への
/// inherent `state_dict`／`load_state_dict` 追加）を得た日が来たら、
/// 本モジュール・本 doctest 自体を削除する（ソース走査側の対応する
/// 否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_optim_state_dict_hold_probe {
///     pub trait OptimizerStateDict {}
/// }
/// use __fandhe_optim_state_dict_hold_probe::*;
///
/// fn __probe_trait<T: ?Sized + OptimizerStateDict>() {}
///
/// struct __FandheOptimizerStateDictMarker;
///
/// trait __FandheOptimizerStateDictProbe {
///     fn state_dict(&self) -> __FandheOptimizerStateDictMarker;
///     fn load_state_dict(&mut self) -> __FandheOptimizerStateDictMarker;
/// }
///
/// impl __FandheOptimizerStateDictProbe for fandhe_ai::optim::AdamW {
///     fn state_dict(&self) -> __FandheOptimizerStateDictMarker {
///         __FandheOptimizerStateDictMarker
///     }
///     fn load_state_dict(&mut self) -> __FandheOptimizerStateDictMarker {
///         __FandheOptimizerStateDictMarker
///     }
/// }
///
/// impl __FandheOptimizerStateDictProbe for fandhe_ai::optim::Adam {
///     fn state_dict(&self) -> __FandheOptimizerStateDictMarker {
///         __FandheOptimizerStateDictMarker
///     }
///     fn load_state_dict(&mut self) -> __FandheOptimizerStateDictMarker {
///         __FandheOptimizerStateDictMarker
///     }
/// }
///
/// impl __FandheOptimizerStateDictProbe for fandhe_ai::optim::RmsProp {
///     fn state_dict(&self) -> __FandheOptimizerStateDictMarker {
///         __FandheOptimizerStateDictMarker
///     }
///     fn load_state_dict(&mut self) -> __FandheOptimizerStateDictMarker {
///         __FandheOptimizerStateDictMarker
///     }
/// }
///
/// impl __FandheOptimizerStateDictProbe for fandhe_ai::optim::Adagrad {
///     fn state_dict(&self) -> __FandheOptimizerStateDictMarker {
///         __FandheOptimizerStateDictMarker
///     }
///     fn load_state_dict(&mut self) -> __FandheOptimizerStateDictMarker {
///         __FandheOptimizerStateDictMarker
///     }
/// }
///
/// impl __FandheOptimizerStateDictProbe for fandhe_ai::optim::Lamb {
///     fn state_dict(&self) -> __FandheOptimizerStateDictMarker {
///         __FandheOptimizerStateDictMarker
///     }
///     fn load_state_dict(&mut self) -> __FandheOptimizerStateDictMarker {
///         __FandheOptimizerStateDictMarker
///     }
/// }
///
/// fn __probe(
///     adamw: &mut fandhe_ai::optim::AdamW,
///     adam: &mut fandhe_ai::optim::Adam,
///     rmsprop: &mut fandhe_ai::optim::RmsProp,
///     adagrad: &mut fandhe_ai::optim::Adagrad,
///     lamb: &mut fandhe_ai::optim::Lamb,
/// ) {
///     let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::AdamW::state_dict(adamw);
///     let _: __FandheOptimizerStateDictMarker = adamw.state_dict();
///     let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::AdamW::load_state_dict(adamw);
///     let _: __FandheOptimizerStateDictMarker = adamw.load_state_dict();
///     let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::Adam::state_dict(adam);
///     let _: __FandheOptimizerStateDictMarker = adam.state_dict();
///     let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::Adam::load_state_dict(adam);
///     let _: __FandheOptimizerStateDictMarker = adam.load_state_dict();
///     let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::RmsProp::state_dict(rmsprop);
///     let _: __FandheOptimizerStateDictMarker = rmsprop.state_dict();
///     let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::RmsProp::load_state_dict(rmsprop);
///     let _: __FandheOptimizerStateDictMarker = rmsprop.load_state_dict();
///     let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::Adagrad::state_dict(adagrad);
///     let _: __FandheOptimizerStateDictMarker = adagrad.state_dict();
///     let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::Adagrad::load_state_dict(adagrad);
///     let _: __FandheOptimizerStateDictMarker = adagrad.load_state_dict();
///     let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::Lamb::state_dict(lamb);
///     let _: __FandheOptimizerStateDictMarker = lamb.state_dict();
///     let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::Lamb::load_state_dict(lamb);
///     let _: __FandheOptimizerStateDictMarker = lamb.load_state_dict();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct OptimizerStateDictHoldDoctestGuard;

/// イシュー #2179（親 #2131「PyTorch／TF 置き換えの API 網羅」）の
/// facade 公開保留を固定する doctest 足場。`OptimizerExtHoldDoctestGuard`
/// （#2171。#2501 で削除済み）と同型の、2 系統の否定ガードを 1 ブロックで兼ねる方式を
/// 採る。
///
/// [`fandhe_ai_autodiff::nn::ExponentialMovingAverage`]（内部クレート
/// 限定。`crates/autodiff/src/nn/ema.rs`。イシュー #2179）は facade
/// （`fandhe_ai::optim` 等）から再エクスポートしておらず、`compat::
/// FitConfig`（`fit(use_ema=true)` 相当のフィールド／メソッド追加）・
/// `compat::Sequential`（EMA 適用・復元メソッド追加）への接続も未実装
/// のまま保留する（`docs/autodiff-ema-decision.md` §4「承認事項」節。
/// 親 #2131 の「facade 公開面拡張は設計判断記録 → 承認 → 実装の 2 段」
/// 規則に基づく）。
///
/// 1. **型名の再エクスポート・独自宣言**（`OptimizerExtHoldDoctestGuard`（#2501 で削除済み）
///    と同じ glob 衝突方式）: facade の全 `pub mod` を glob import した
///    スコープに、本ブロック内でのみ定義したローカル
///    `__fandhe_ema_hold_probe::ExponentialMovingAverage` を導入し、
///    それを引数に取る `__probe_type` 関数を書く。facade がどの経路
///    でこの名前を公開しても、ローカル定義との glob 衝突（E0659 等）で
///    コンパイルが失敗する。ソース走査ガードは `crates/facade/tests/
///    api_surface.rs::facade_does_not_reexport_or_declare_ema_items`。
/// 2. **`compat::FitConfig`／`compat::Sequential` への inherent メソッド
///    追加**（`VarCustomHoldDoctestGuard`〈#2064〉と同じマーカー型
///    トレイト方式）: ローカル `__FandheEmaHoldProbe` トレイト
///    （`use_ema`／`ema_decay` という名前のメソッドを持つ）を両型へ
///    実装し、メソッド形・型パス形の両方で呼び出す。facade がどちらか
///    の型へ同名の inherent メソッドを追加すると、優先解決される
///    inherent メソッドの戻り値の型がプローブの期待型と一致せず型
///    不一致でコンパイルが失敗する（承認後の実際のメソッド名・シグ
///    ネチャは `docs/autodiff-ema-decision.md` §5「承認後の facade
///    仕様案」で未確定のため、本足場の `use_ema`／`ema_decay` は
///    「これらの名前を持つ inherent メソッドが facade 型に生えたら
///    検出する」という最小契約に留める）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// ema_hold_doctest_globs_all_pub_modules`・
/// `ema_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_ema_items`）との多層防御の
/// 位置づけは `docs/autodiff-ema-decision.md` §4 を参照。
///
/// facade 公開（ユーザー承認）がされる日が来たら、本モジュール・本
/// doctest 自体を削除する（ソース走査側の対応する否定ガードも同時に
/// 正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_ema_hold_probe {
///     pub struct ExponentialMovingAverage;
/// }
/// use __fandhe_ema_hold_probe::*;
///
/// fn __probe_type(_: ExponentialMovingAverage) {}
///
/// struct __FandheEmaHoldMarker;
///
/// trait __FandheEmaHoldProbe {
///     fn use_ema(&self) -> __FandheEmaHoldMarker;
///     fn ema_decay(&self) -> __FandheEmaHoldMarker;
/// }
///
/// impl __FandheEmaHoldProbe for fandhe_ai::compat::FitConfig {
///     fn use_ema(&self) -> __FandheEmaHoldMarker {
///         __FandheEmaHoldMarker
///     }
///     fn ema_decay(&self) -> __FandheEmaHoldMarker {
///         __FandheEmaHoldMarker
///     }
/// }
///
/// impl __FandheEmaHoldProbe for fandhe_ai::compat::Sequential {
///     fn use_ema(&self) -> __FandheEmaHoldMarker {
///         __FandheEmaHoldMarker
///     }
///     fn ema_decay(&self) -> __FandheEmaHoldMarker {
///         __FandheEmaHoldMarker
///     }
/// }
///
/// fn __probe_fit_config(x: &fandhe_ai::compat::FitConfig) {
///     let _: __FandheEmaHoldMarker = fandhe_ai::compat::FitConfig::use_ema(x);
///     let _: __FandheEmaHoldMarker = x.use_ema();
///     let _: __FandheEmaHoldMarker = fandhe_ai::compat::FitConfig::ema_decay(x);
///     let _: __FandheEmaHoldMarker = x.ema_decay();
/// }
///
/// fn __probe_sequential(x: &fandhe_ai::compat::Sequential) {
///     let _: __FandheEmaHoldMarker = fandhe_ai::compat::Sequential::use_ema(x);
///     let _: __FandheEmaHoldMarker = x.use_ema();
///     let _: __FandheEmaHoldMarker = fandhe_ai::compat::Sequential::ema_decay(x);
///     let _: __FandheEmaHoldMarker = x.ema_decay();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct EmaHoldDoctestGuard;

/// イシュー #2177（親 #2131「PyTorch／TF 置き換えの API 網羅（対応表の
/// 行内深掘り）」）の facade 公開保留を固定する doctest 足場。
/// 旧 MHA 保留ガード（#2163。#2530 で撤去済み）と同型の「正のプローブ 1
/// ブロック方式」を採る。
///
/// 候補の公開面は 2 種に分かれる: (a) まだ存在しない新規型 `FitWeights`
/// （型名の glob 衝突プローブ）、(b) 既存の公開型 `fandhe_ai::compat::
/// {FitConfig, Sequential}` へのメソッド追加（トレイトプローブ）。
///
/// (a) `__fandhe_fit_weighting_hold_probe::FitWeights` をローカル
/// 宣言する。facade が同名の型を `pub use`／`pub struct`／`pub type` の
/// いずれで公開しても、`use fandhe_ai::compat::*;` の glob が同名を
/// 持ち込み、ローカル定義との衝突（E0659）でコンパイルが失敗する
/// （`CsvLogger` 等の旧保留ガード〈`CallbacksLoggersHoldDoctestGuard`。#2571 で削除済み〉と同方式）。
///
/// (b) `validation_split`／`class_weight`／`sample_weight` を
/// `fandhe_ai::compat::FitConfig` へ、`fit_with_weights`／
/// `fit_weighted` を `fandhe_ai::compat::Sequential` へ実装する
/// `__FandheFitWeightHoldProbe` トレイトの衝突プローブ（`
/// __FandheMhaOptionsAddProbe` と同方式）。`class_weight`／
/// `sample_weight` を `FitConfig` 側のプローブに含めるのは、受入基準の
/// 字面（`FitConfig` へ直接フィールド追加）どおりに実装すると
/// `#[derive(Copy, Eq)]`（`crates/facade/src/compat/training.rs:175`）
/// と衝突し 0.9.0 非破壊契約に反するため、その禁止された実装経路
/// 自体も検出対象に含める設計判断による（`docs/compat-fit-sample-
/// weighting-decision.md` §2）。**UFCS 形のみ**（`fandhe_ai::compat::
/// FitConfig::validation_split(cfg)` 等）で呼ぶ（`FitConfig` の
/// ビルダーは値で `self` を取る inherent メソッド、`Sequential::fit*`
/// は `&mut self` を取る inherent メソッドのため、メソッド呼び出し形
/// だと inherent 側が優先解決され衝突を検出できない——
/// 旧 MHA 保留ガード（#2163。#2530 で撤去済み） doc の同一理由）。本プローブの
/// トレイトメソッドはいずれも引数なし `&self` のみを取るため、facade
/// 側に実引数を要求する本物の実装が追加されれば UFCS 呼び出しの引数
/// 個数・型が一致せずコンパイルが失敗する。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// fit_weighting_hold_doctest_globs_all_pub_modules`・
/// `fit_weighting_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_fit_weighting_items`・
/// `fit_config_keeps_copy_eq_for_0_9_0_compat`）との多層防御の位置
/// づけは `docs/compat-fit-sample-weighting-decision.md` §7 を参照。
///
/// facade 公開（ユーザー承認）がされる日が来たら、本モジュール・本
/// doctest 自体を削除する（ソース走査側の対応する否定ガードも同時に
/// 正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_fit_weighting_hold_probe {
///     pub struct FitWeights;
/// }
/// use __fandhe_fit_weighting_hold_probe::*;
///
/// struct __FandheFitWeightHoldMarker;
///
/// trait __FandheFitWeightHoldProbe {
///     fn validation_split(&self) -> __FandheFitWeightHoldMarker;
///     fn class_weight(&self) -> __FandheFitWeightHoldMarker;
///     fn sample_weight(&self) -> __FandheFitWeightHoldMarker;
///     fn fit_with_weights(&self) -> __FandheFitWeightHoldMarker;
///     fn fit_weighted(&self) -> __FandheFitWeightHoldMarker;
/// }
///
/// impl __FandheFitWeightHoldProbe for fandhe_ai::compat::FitConfig {
///     fn validation_split(&self) -> __FandheFitWeightHoldMarker {
///         __FandheFitWeightHoldMarker
///     }
///     fn class_weight(&self) -> __FandheFitWeightHoldMarker {
///         __FandheFitWeightHoldMarker
///     }
///     fn sample_weight(&self) -> __FandheFitWeightHoldMarker {
///         __FandheFitWeightHoldMarker
///     }
///     fn fit_with_weights(&self) -> __FandheFitWeightHoldMarker {
///         __FandheFitWeightHoldMarker
///     }
///     fn fit_weighted(&self) -> __FandheFitWeightHoldMarker {
///         __FandheFitWeightHoldMarker
///     }
/// }
///
/// impl __FandheFitWeightHoldProbe for fandhe_ai::compat::Sequential {
///     fn validation_split(&self) -> __FandheFitWeightHoldMarker {
///         __FandheFitWeightHoldMarker
///     }
///     fn class_weight(&self) -> __FandheFitWeightHoldMarker {
///         __FandheFitWeightHoldMarker
///     }
///     fn sample_weight(&self) -> __FandheFitWeightHoldMarker {
///         __FandheFitWeightHoldMarker
///     }
///     fn fit_with_weights(&self) -> __FandheFitWeightHoldMarker {
///         __FandheFitWeightHoldMarker
///     }
///     fn fit_weighted(&self) -> __FandheFitWeightHoldMarker {
///         __FandheFitWeightHoldMarker
///     }
/// }
///
/// fn __probe(_: FitWeights, cfg: &fandhe_ai::compat::FitConfig, seq: &fandhe_ai::compat::Sequential) {
///     let _: __FandheFitWeightHoldMarker =
///         fandhe_ai::compat::FitConfig::validation_split(cfg);
///     let _: __FandheFitWeightHoldMarker =
///         fandhe_ai::compat::FitConfig::class_weight(cfg);
///     let _: __FandheFitWeightHoldMarker =
///         fandhe_ai::compat::FitConfig::sample_weight(cfg);
///     let _: __FandheFitWeightHoldMarker =
///         fandhe_ai::compat::Sequential::fit_with_weights(seq);
///     let _: __FandheFitWeightHoldMarker =
///         fandhe_ai::compat::Sequential::fit_weighted(seq);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct FitWeightingHoldDoctestGuard;

/// イシュー #2184（親 #2131）・#2568 の train_step フック公開形を固定する
/// doctest 足場。承認形（`TrainStepFn`／`TrainStepOptimizer`／
/// `TrainStepOutput` の 3 型と inherent `Sequential::fit_with_train_step`）
/// は #2568 で公開済みのため、本プローブは**残る禁止経路**だけを固定する:
/// `FitConfig::train_step_fn`（フックを設定へ持たせる形。`FitConfig` は
/// `Copy + Eq` を維持するため不可）・`FitConfig::fit_with_train_step`・
/// `Sequential::train_step_fn`（承認形にない名前）。
/// `FitWeightingHoldDoctestGuard`（#2177）と同型の「正のプローブ 1 ブロック
/// 方式」で、facade の全 `pub mod` を glob import したスコープに、これらの
/// 名前を持つトレイト（`__FandheTrainStepHoldProbe`）を導入して UFCS で
/// 呼ぶ。facade が inherent メソッドとして公開すると trait 経由呼び出しが
/// 型・引数不一致でコンパイル失敗する（inherent が優先解決されるため）。
///
/// ソース走査の正ガード（#2569 で反転済み。`crates/facade/tests/api_surface.rs::
/// facade_train_step_public_surface_matches_approved_contract`・
/// `facade_train_step_optimizer_and_output_shapes_match_approved_contract`・
/// `workspace_declares_train_step_fn_names_only_in_approved_location`・
/// `train_step_types_are_reachable_via_facade_only`）が承認形の存在と形を固定し、
/// 本 doctest は禁止経路専用として型検査レベルで維持する
/// （`train_step_hold_doctest_globs_all_pub_modules`・
/// `train_step_hold_doctest_probe_body_matches_fixed_contract` がドリフトを検出）。
/// 多層防御の位置づけは `docs/compat-train-step-hook-decision.md` §8.4・§9 を参照。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// struct __FandheTrainStepHoldMarker;
///
/// trait __FandheTrainStepHoldProbe {
///     fn train_step_fn(&self) -> __FandheTrainStepHoldMarker;
///     fn fit_with_train_step(&self) -> __FandheTrainStepHoldMarker;
/// }
///
/// impl __FandheTrainStepHoldProbe for fandhe_ai::compat::FitConfig {
///     fn train_step_fn(&self) -> __FandheTrainStepHoldMarker {
///         __FandheTrainStepHoldMarker
///     }
///     fn fit_with_train_step(&self) -> __FandheTrainStepHoldMarker {
///         __FandheTrainStepHoldMarker
///     }
/// }
///
/// impl __FandheTrainStepHoldProbe for fandhe_ai::compat::Sequential {
///     fn train_step_fn(&self) -> __FandheTrainStepHoldMarker {
///         __FandheTrainStepHoldMarker
///     }
///     fn fit_with_train_step(&self) -> __FandheTrainStepHoldMarker {
///         __FandheTrainStepHoldMarker
///     }
/// }
///
/// fn __probe(cfg: &fandhe_ai::compat::FitConfig, seq: &fandhe_ai::compat::Sequential) {
///     let _: __FandheTrainStepHoldMarker =
///         fandhe_ai::compat::FitConfig::train_step_fn(cfg);
///     let _: __FandheTrainStepHoldMarker =
///         fandhe_ai::compat::FitConfig::fit_with_train_step(cfg);
///     let _: __FandheTrainStepHoldMarker =
///         fandhe_ai::compat::Sequential::train_step_fn(seq);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct TrainStepHoldDoctestGuard;

/// イシュー #2189（親 #2131）の facade 公開保留を固定する doctest 足場。
/// `RngDistributionsHoldDoctestGuard`（#2156）と同型の「正のプローブ 1
/// ブロック方式」を採る: facade の全 `pub mod` を glob import した
/// スコープに、本ブロック内でのみ定義したローカル
/// `__fandhe_npy_io_hold_probe::{NpyError, load_npy, save_npy, load_npz,
/// save_npz, npy, npz}` を導入し、実際に使う名前・呼び出しを書く。
/// facade がどの経路（`interop`／`tensor_io` 配下への自由関数としての
/// 再エクスポート・`Tensor<f32>` への inherent メソッド追加・facade 独自
/// の `pub mod npy`／`pub mod npz`／`struct NpyError` 宣言）でこれらの
/// 名前を公開しても、ローカル定義との glob 衝突（自由関数・モジュール名・
/// 型名の場合。E0659 等）または呼び出しシグネチャの不一致（inherent
/// メソッドがトレイトメソッドより優先解決されるため、本プローブの trait
/// 経由呼び出しが型・引数不一致でコンパイル失敗する）でエラーコードに
/// 依存せずコンパイルが失敗する。
///
/// `tensor-core` 側の実装（`crates/tensor-core/src/io/{mod,npy,npz,
/// crc32,inflate}.rs`）は完了済みで、保留対象は facade 公開面 4 件
/// （`load_npy`／`save_npy`／`load_npz`／`save_npz`。`NpyError` の
/// 再エクスポートを含む）のみ。`Tensor` に inherent メソッドを追加しない
/// のは facade が `pub use fandhe_ai_tensor_core::{..., Tensor, ...};`
/// で `Tensor` を再エクスポートしているため、inherent メソッド追加が
/// それだけで facade の公開面を広げてしまうから（#2156 の前例。
/// `docs/rng-distributions-generator-decision.md:28`）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// npy_io_hold_doctest_globs_all_pub_modules`・`npy_io_hold_doctest_
/// probe_body_matches_fixed_contract`・`facade_does_not_reexport_or_
/// declare_npy_io`・`workspace_declares_npy_io_names_only_in_allowed_
/// locations`）との多層防御の位置づけ・承認未取得の経緯は
/// `docs/tensor-core-npy-npz-io-decision.md` §「承認事項」節を参照。
///
/// facade 公開（ユーザー承認）がされる日が来たら、本モジュール・本
/// doctest 自体を削除する（ソース走査側の対応する否定ガードも同時に
/// 正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_npy_io_hold_probe {
///     pub struct NpyError;
///     pub fn load_npy() {}
///     pub fn save_npy() {}
///     pub fn load_npz() {}
///     pub fn save_npz() {}
///     pub mod npy {
///         pub fn __mark() {}
///     }
///     pub mod npz {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_npy_io_hold_probe::*;
///
/// struct __FandheNpyIoHoldMarker;
///
/// trait __FandheNpyIoHoldProbe {
///     fn load_npy(&self) -> __FandheNpyIoHoldMarker;
///     fn save_npy(&self) -> __FandheNpyIoHoldMarker;
///     fn load_npz(&self) -> __FandheNpyIoHoldMarker;
///     fn save_npz(&self) -> __FandheNpyIoHoldMarker;
/// }
///
/// impl __FandheNpyIoHoldProbe for fandhe_ai::Tensor<f32> {
///     fn load_npy(&self) -> __FandheNpyIoHoldMarker { __FandheNpyIoHoldMarker }
///     fn save_npy(&self) -> __FandheNpyIoHoldMarker { __FandheNpyIoHoldMarker }
///     fn load_npz(&self) -> __FandheNpyIoHoldMarker { __FandheNpyIoHoldMarker }
///     fn save_npz(&self) -> __FandheNpyIoHoldMarker { __FandheNpyIoHoldMarker }
/// }
///
/// fn __probe_free_fns(_: NpyError) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     load_npy();
///     save_npy();
///     load_npz();
///     save_npz();
///     npy::__mark();
///     npz::__mark();
/// }
///
/// fn __probe_tensor(x: &fandhe_ai::Tensor<f32>) {
///     let _: __FandheNpyIoHoldMarker = fandhe_ai::Tensor::load_npy(x);
///     let _: __FandheNpyIoHoldMarker = x.save_npy();
///     let _: __FandheNpyIoHoldMarker = fandhe_ai::Tensor::load_npz(x);
///     let _: __FandheNpyIoHoldMarker = x.save_npz();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct NpyIoHoldDoctestGuard;

/// イシュー #2188（親 #2131「PyTorch／TF 置き換えの API 網羅（対応表の
/// 行内深掘り）」）で導入し、イシュー #2369（親 #2362）で**代替案専用へ縮小**
/// した facade 公開保留の doctest 足場。
/// `TrainStepHoldDoctestGuard`（#2184）と同型の「正のプローブ 1 ブロック
/// 方式」を採る: facade の全 `pub mod` を glob import したスコープに、
/// 本ブロック内でのみ定義した、`fandhe_ai::compat::Sequential` への inherent
/// メソッドを装うトレイト（`__FandheModelIoHoldProbe`）を導入し、実際に使う
/// 呼び出しを書く。facade が `Sequential` へ inherent メソッド
/// （`save`／`load`／`save_model`／`load_model`）を追加すると、inherent メソッドが
/// トレイトメソッドより優先解決されるため、本プローブの trait 経由呼び出しが
/// 型・引数不一致でエラーコードに依存せずコンパイル失敗する。
///
/// `docs/compat-model-io-decision.md` §2 は代替公開 API 案として
/// `Sequential::save(&self, dir)`／`Sequential::load(dir)`（`_model`
/// 接尾辞なしの inherent メソッド）も併記しているため、本トレイトは
/// `save_model`／`load_model` に加えて `save`／`load` も
/// `__FandheModelIoHoldProbe` のメソッドとして持つ（PR #2317 review
/// 指摘: 旧版は `_model` 接尾辞ありの 2 名だけを検出しており、代替名
/// `Sequential::save`／`Sequential::load` を facade が追加しても本
/// プローブ・`api_surface.rs` のソース走査のいずれも検出できなかった）。
///
/// **#2369 で自由関数 `compat::save_model`／`compat::load_model` と
/// `compat::ModelIoError` は公開済み**（親 #2362 でユーザー承認済みの主案。
/// 承認事項 1）。そのため、旧版が持っていたローカル定義（`model_io` モジュール・
/// 自由関数・エラー型のプローブ）は glob 衝突で本 doctest 自体を壊すため撤去し、
/// 正の確認は `api_surface.rs` の正ガード
/// （`model_io_items_are_reachable_via_facade` 等）へ移した。**残る保留**は
/// 代替案の inherent メソッド `Sequential::save`／`Sequential::load`（および
/// 同じく承認外の `Sequential::save_model`／`Sequential::load_model`）で、
/// 本 doctest と `api_surface.rs` の `Sequential` 内 `fn save`／`fn load`
/// 走査・workspace インベントリ（0 件固定）が固定する
/// （`docs/compat-model-io-decision.md` §7・§10）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// model_io_hold_doctest_globs_all_pub_modules`・
/// `model_io_hold_doctest_probe_body_matches_fixed_contract`・
/// `workspace_declares_sequential_alt_save_load_fn_names_only_in_allowed_locations`）
/// との多層防御の位置づけは同 doc §7 を参照。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// struct __FandheModelIoHoldMarker;
///
/// trait __FandheModelIoHoldProbe {
///     fn save_model(&self) -> __FandheModelIoHoldMarker;
///     fn load_model(&self) -> __FandheModelIoHoldMarker;
///     fn save(&self) -> __FandheModelIoHoldMarker;
///     fn load(&self) -> __FandheModelIoHoldMarker;
/// }
///
/// impl __FandheModelIoHoldProbe for fandhe_ai::compat::Sequential {
///     fn save_model(&self) -> __FandheModelIoHoldMarker {
///         __FandheModelIoHoldMarker
///     }
///     fn load_model(&self) -> __FandheModelIoHoldMarker {
///         __FandheModelIoHoldMarker
///     }
///     fn save(&self) -> __FandheModelIoHoldMarker {
///         __FandheModelIoHoldMarker
///     }
///     fn load(&self) -> __FandheModelIoHoldMarker {
///         __FandheModelIoHoldMarker
///     }
/// }
///
/// fn __probe_inherent_method(seq: &fandhe_ai::compat::Sequential) {
///     let _: __FandheModelIoHoldMarker =
///         fandhe_ai::compat::Sequential::save_model(seq);
///     let _: __FandheModelIoHoldMarker =
///         fandhe_ai::compat::Sequential::load_model(seq);
///     let _: __FandheModelIoHoldMarker =
///         fandhe_ai::compat::Sequential::save(seq);
///     let _: __FandheModelIoHoldMarker =
///         fandhe_ai::compat::Sequential::load(seq);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct ModelIoHoldDoctestGuard;

/// イシュー #2191（`generate()` 自己回帰ループ）の facade 公開保留を
/// 固定する doctest 足場。`KvCacheHoldDoctestGuard`（#2084）と同型の
/// 「正のプローブ 1 ブロック方式」を採る: facade の全 `pub mod` を glob
/// import したスコープに、本ブロック内でのみ定義したローカル
/// `__fandhe_generate_hold_probe::{inference::{generate, GenerateConfig,
/// SamplingStrategy, AutoregressiveModel}, generate, GenerateConfig,
/// SamplingStrategy, AutoregressiveModel}` を導入し、実際に使う関数を
/// 書く。facade がどの経路（`pub mod inference` 配下・crate ルート直下の
/// 自由関数・別名エクスポート・facade 独自の `struct`／`trait` 宣言・
/// `Tape`／`compat::Sequential` への inherent メソッド追加）でこれらの
/// 名前を公開しても、ローカル定義との glob 衝突（型・モジュール名の
/// 場合。E0659 等）または呼び出しシグネチャの不一致（inherent メソッドが
/// トレイトメソッドより優先解決されるため、本プローブの trait 経由
/// 呼び出しが型・引数不一致でコンパイル失敗する）でエラーコードに
/// 依存せずコンパイルが失敗する。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// facade_does_not_expose_generate_items`〈直列 1 行走査。内側の層として
/// 維持〉・`facade_does_not_reexport_or_declare_generate_items`〈トークン
/// 方式。複数行・別名・独自宣言を検出〉）との多層防御の位置づけ・
/// 承認未取得の経緯・承認依頼用の事前設計は
/// `docs/facade-generate-decision.md` §8「承認事項」を参照。
///
/// `workspace` 全体（facade 以外のクレート内部の private 宣言も含む）の
/// 名前インベントリは、KvCache（#2084 §10.1）が codex-review 指摘
/// （無関係な内部宣言まで固定してしまう）を受けて撤回した経緯と同じ
/// 理由（`crates/self-repair` に `fn generate` を持つトレイトが複数
/// あるため）で採用しない（`docs/facade-generate-decision.md` §7）。
///
/// facade 公開（承認事項: `docs/facade-generate-decision.md` §8）が
/// ユーザー承認され実施する日が来たら、本モジュール・本 doctest 自体を
/// 削除する（ソース走査側の対応する否定ガードも同時に正ガードへ
/// 置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_generate_hold_probe {
///     pub mod inference {
///         pub fn generate() {}
///         pub struct GenerateConfig;
///         pub struct SamplingStrategy;
///         pub struct AutoregressiveModel;
///     }
///     pub fn generate() {}
///     pub struct GenerateConfig;
///     pub struct SamplingStrategy;
///     pub struct AutoregressiveModel;
/// }
/// use __fandhe_generate_hold_probe::*;
///
/// struct __FandheGenerateHoldMarker;
///
/// trait __FandheGenerateHoldProbe {
///     fn generate(&self) -> __FandheGenerateHoldMarker;
/// }
///
/// impl __FandheGenerateHoldProbe for fandhe_ai::Tape {
///     fn generate(&self) -> __FandheGenerateHoldMarker {
///         __FandheGenerateHoldMarker
///     }
/// }
///
/// impl __FandheGenerateHoldProbe for fandhe_ai::compat::Sequential {
///     fn generate(&self) -> __FandheGenerateHoldMarker {
///         __FandheGenerateHoldMarker
///     }
/// }
///
/// fn __probe_module_path() {
///     inference::generate();
///     let _ = inference::GenerateConfig;
///     let _ = inference::SamplingStrategy;
///     let _ = inference::AutoregressiveModel;
/// }
///
/// fn __probe_free_fn() {
///     generate();
///     let _ = GenerateConfig;
///     let _ = SamplingStrategy;
///     let _ = AutoregressiveModel;
/// }
///
/// fn __probe_inherent_method(tape: &fandhe_ai::Tape, seq: &fandhe_ai::compat::Sequential) {
///     let _: __FandheGenerateHoldMarker = fandhe_ai::Tape::generate(tape);
///     let _: __FandheGenerateHoldMarker = fandhe_ai::compat::Sequential::generate(seq);
///     let _: __FandheGenerateHoldMarker = tape.generate();
///     let _: __FandheGenerateHoldMarker = seq.generate();
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct GenerateHoldDoctestGuard;

/// イシュー #2192（親 #2131）の facade 公開保留を固定する doctest 足場。
/// `NpyIoHoldDoctestGuard`（#2189）と同型の「正のプローブ 1 ブロック
/// 方式」を採る: facade の全 `pub mod` を glob import したスコープに、
/// 本ブロック内でのみ定義したローカル
/// `__fandhe_predict_batches_hold_probe::{PhaseMetrics, get_phase_metrics,
/// current_phase_metrics, reset_phase_metrics, inference::__mark}` を
/// 導入し、実際に使う名前・呼び出しを書く。facade がどの経路（`pub use`
/// による再エクスポート・型宣言・`Sequential` への `predict_batches`／
/// `get_phase_metrics`／`current_phase_metrics` inherent メソッド追加・
/// `pub mod inference` の新設）でこれらの名前を公開しても、ローカル定義
/// との glob 衝突（自由関数・モジュール名・型名の場合。E0659 等）または
/// 呼び出しシグネチャの不一致（inherent メソッドがトレイトメソッドより
/// 優先解決されるため、本プローブの trait 経由呼び出しが型・引数不一致で
/// コンパイル失敗する）でエラーコードに依存せずコンパイルが失敗する。
///
/// `crate::inference::batch`（DataLoader 反復推論・phase 計測の内部実装。
/// `#[cfg(test)]` 限定で `Sequential::run_loader_inference` から到達
/// 可能）は実装済みで、保留対象は facade 公開面 3 件（`predict_batches`・
/// `PhaseMetrics`（`get_phase_metrics`／`current_phase_metrics`／
/// `reset_phase_metrics` を含む計測アクセサ一式）・`pub mod inference`
/// の新設）のみ。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// predict_batches_hold_doctest_globs_all_pub_modules`・
/// `predict_batches_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_predict_batches_items`・
/// `workspace_declares_predict_batches_fn_names_nowhere`）との多層防御の
/// 位置づけ・承認未取得の経緯は `docs/facade-predict-batches-phase-
/// metrics-decision.md` §5「承認事項」節を参照。
///
/// 承認（`predict_batches`・`PhaseMetrics` 一式・`pub mod inference` の
/// 新設）を得た日が来たら、本モジュール・本 doctest 自体を削除する
/// （ソース走査側の対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_predict_batches_hold_probe {
///     pub struct PhaseMetrics;
///     pub fn get_phase_metrics() {}
///     pub fn current_phase_metrics() {}
///     pub fn reset_phase_metrics() {}
///     pub mod inference {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_predict_batches_hold_probe::*;
///
/// struct __FandhePredictBatchesHoldMarker;
///
/// trait __FandhePredictBatchesHoldProbe {
///     fn predict_batches(&self) -> __FandhePredictBatchesHoldMarker;
///     fn get_phase_metrics(&self) -> __FandhePredictBatchesHoldMarker;
///     fn current_phase_metrics(&self) -> __FandhePredictBatchesHoldMarker;
/// }
///
/// impl __FandhePredictBatchesHoldProbe for fandhe_ai::compat::Sequential {
///     fn predict_batches(&self) -> __FandhePredictBatchesHoldMarker {
///         __FandhePredictBatchesHoldMarker
///     }
///     fn get_phase_metrics(&self) -> __FandhePredictBatchesHoldMarker {
///         __FandhePredictBatchesHoldMarker
///     }
///     fn current_phase_metrics(&self) -> __FandhePredictBatchesHoldMarker {
///         __FandhePredictBatchesHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(_: PhaseMetrics) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     get_phase_metrics();
///     current_phase_metrics();
///     reset_phase_metrics();
///     inference::__mark();
/// }
///
/// fn __probe_sequential(seq: &fandhe_ai::compat::Sequential) {
///     let _: __FandhePredictBatchesHoldMarker =
///         fandhe_ai::compat::Sequential::predict_batches(seq);
///     let _: __FandhePredictBatchesHoldMarker = seq.get_phase_metrics();
///     let _: __FandhePredictBatchesHoldMarker =
///         fandhe_ai::compat::Sequential::current_phase_metrics(seq);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct PredictBatchesHoldDoctestGuard;

/// イシュー #2631・#2632・#2633（親 #2630・ルート #2499 Phase 4。`rfft`／`irfft`／
/// `fft`／`ifft`／`stft`／`istft`）の facade 公開保留を固定する doctest 足場。`PredictBatchesHoldDoctestGuard`（#2192）と同型の
/// 「正のプローブ 1 ブロック方式」を採る: facade の全 `pub mod` を glob import
/// したスコープに、本ブロック内でのみ定義したローカル
/// `__fandhe_fft_hold_probe::{FftNorm, StftOptions, IstftOptions, StftPadMode,
/// fft_ops::{rfft, irfft, fft, ifft, stft, istft}, fft::__mark}`
/// を導入し、実際に使う名前・呼び出しを書く。facade がどの経路（`pub use` に
/// よる再エクスポート・型宣言・`Var`／`Tape` への `rfft`／`irfft`／`fft`／`ifft`／`stft`／`istft` inherent
/// メソッド追加・`pub mod fft_ops`／`pub mod fft` の新設）でこれらの名前を
/// 公開しても、ローカル定義との glob 衝突（モジュール名・型名の場合。E0659
/// 等）または呼び出しシグネチャの不一致（inherent メソッドがトレイトメソッド
/// より優先解決されるため、本プローブの `Type::method` 形式の呼び出しが型
/// 不一致でコンパイル失敗する）でエラーコードに依存せずコンパイルが失敗する。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::fft_ops::{rfft,
/// irfft, fft, ifft, stft, istft, StftOptions, IstftOptions}`・
/// `fandhe_ai_tensor_core::{FftNorm, StftPadMode, fft}`）。保留対象は
/// facade 公開面（`Var::rfft`／`Var::irfft`／`Var::fft`／`Var::ifft`／
/// `Var::stft`／`Var::istft` の委譲メソッドと `FftNorm`・`StftOptions`・
/// `IstftOptions`・`StftPadMode` の再エクスポート）のみで、公開形は未承認（承認依頼は #2677・公開自体は
/// 承認後の #2678。推奨案は `docs/autodiff-fft-ops-decision.md` §7）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// fft_ops_hold_doctest_globs_all_pub_modules`・
/// `fft_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_fft_ops`・
/// `workspace_declares_fft_ops_fn_names_only_in_allowed_locations`）との
/// 多層防御の位置づけは同決定記録 §9 を参照。
///
/// 承認を得た日が来たら、本モジュール・本 doctest 自体を削除する（ソース
/// 走査側の対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_fft_hold_probe {
///     pub struct FftNorm;
///     pub struct StftOptions;
///     pub struct IstftOptions;
///     pub struct StftPadMode;
///     pub mod fft_ops {
///         pub fn rfft() {}
///         pub fn irfft() {}
///         pub fn fft() {}
///         pub fn ifft() {}
///         pub fn stft() {}
///         pub fn istft() {}
///     }
///     pub mod fft {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_fft_hold_probe::*;
///
/// struct __FandheFftHoldMarker;
///
/// trait __FandheFftHoldProbe {
///     fn rfft(&self) -> __FandheFftHoldMarker;
///     fn irfft(&self) -> __FandheFftHoldMarker;
///     fn fft(&self) -> __FandheFftHoldMarker;
///     fn ifft(&self) -> __FandheFftHoldMarker;
///     fn stft(&self) -> __FandheFftHoldMarker;
///     fn istft(&self) -> __FandheFftHoldMarker;
/// }
///
/// impl<'t> __FandheFftHoldProbe for fandhe_ai::Var<'t> {
///     fn rfft(&self) -> __FandheFftHoldMarker {
///         __FandheFftHoldMarker
///     }
///     fn irfft(&self) -> __FandheFftHoldMarker {
///         __FandheFftHoldMarker
///     }
///     fn fft(&self) -> __FandheFftHoldMarker {
///         __FandheFftHoldMarker
///     }
///     fn ifft(&self) -> __FandheFftHoldMarker {
///         __FandheFftHoldMarker
///     }
///     fn stft(&self) -> __FandheFftHoldMarker {
///         __FandheFftHoldMarker
///     }
///     fn istft(&self) -> __FandheFftHoldMarker {
///         __FandheFftHoldMarker
///     }
/// }
///
/// impl __FandheFftHoldProbe for fandhe_ai::Tape {
///     fn rfft(&self) -> __FandheFftHoldMarker {
///         __FandheFftHoldMarker
///     }
///     fn irfft(&self) -> __FandheFftHoldMarker {
///         __FandheFftHoldMarker
///     }
///     fn fft(&self) -> __FandheFftHoldMarker {
///         __FandheFftHoldMarker
///     }
///     fn ifft(&self) -> __FandheFftHoldMarker {
///         __FandheFftHoldMarker
///     }
///     fn stft(&self) -> __FandheFftHoldMarker {
///         __FandheFftHoldMarker
///     }
///     fn istft(&self) -> __FandheFftHoldMarker {
///         __FandheFftHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(_: FftNorm, _: StftOptions, _: IstftOptions, _: StftPadMode) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     fft_ops::rfft();
///     fft_ops::irfft();
///     fft_ops::fft();
///     fft_ops::ifft();
///     fft_ops::stft();
///     fft_ops::istft();
///     fft::__mark();
/// }
///
/// fn __probe_methods(v: &fandhe_ai::Var<'_>, tape: &fandhe_ai::Tape) {
///     let _: __FandheFftHoldMarker = fandhe_ai::Var::rfft(v);
///     let _: __FandheFftHoldMarker = fandhe_ai::Var::irfft(v);
///     let _: __FandheFftHoldMarker = fandhe_ai::Var::fft(v);
///     let _: __FandheFftHoldMarker = fandhe_ai::Var::ifft(v);
///     let _: __FandheFftHoldMarker = fandhe_ai::Var::stft(v);
///     let _: __FandheFftHoldMarker = fandhe_ai::Var::istft(v);
///     let _: __FandheFftHoldMarker = fandhe_ai::Tape::rfft(tape);
///     let _: __FandheFftHoldMarker = fandhe_ai::Tape::irfft(tape);
///     let _: __FandheFftHoldMarker = fandhe_ai::Tape::fft(tape);
///     let _: __FandheFftHoldMarker = fandhe_ai::Tape::ifft(tape);
///     let _: __FandheFftHoldMarker = fandhe_ai::Tape::stft(tape);
///     let _: __FandheFftHoldMarker = fandhe_ai::Tape::istft(tape);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct FftOpsHoldDoctestGuard;

/// 逆三角関数・双曲線関数（`atan`・`asin`・`acos`・`atan2`・`sinh`・`cosh`・
/// `asinh`・`acosh`・`atanh`。イシュー #2634・親 #2625・ルート #2499
/// Phase 4）を facade 公開面から締め出す保留ガード（`FftOpsHoldDoctestGuard`
/// と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカル
/// モジュール `trig_ops` と 9 メソッドを持つプローブ用トレイトを置き、
/// 修飾なしの関数呼び出しと `Var`／`Tape` の修飾付きメソッド呼び出しの両方を
/// 行う。facade が同名のモジュール・関数を glob 可能な位置へ公開するか、
/// `Var`／`Tape` へ同名の inherent メソッドを公開すると、名前解決の曖昧性または
/// 呼び出しシグネチャの不一致でエラーコードに依存せずコンパイルが失敗する。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::trig_ops`・
/// `fandhe_ai_tensor_core::{ScalarUnaryOp, ScalarBinaryOp}` の追加 variant）。
/// 保留対象は facade 公開面（`Var::atan` 等の委譲メソッド）のみで、公開形は
/// 未承認（承認依頼は #2677・公開自体は承認後の #2678。推奨案は
/// `docs/autodiff-trig-ops-decision.md` §7。同記録は推奨案の記録であり
/// 承認記録ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// trig_ops_hold_doctest_globs_all_pub_modules`・
/// `trig_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_trig_ops`・
/// `workspace_declares_trig_ops_fn_names_only_in_allowed_locations`）との
/// 多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の
/// 対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_trig_ops_hold_probe {
///     pub mod trig_ops {
///         pub fn atan() {}
///         pub fn asin() {}
///         pub fn acos() {}
///         pub fn sinh() {}
///         pub fn cosh() {}
///         pub fn asinh() {}
///         pub fn acosh() {}
///         pub fn atanh() {}
///         pub fn atan2() {}
///     }
/// }
/// use __fandhe_trig_ops_hold_probe::*;
///
/// struct __FandheTrigOpsHoldMarker;
///
/// trait __FandheTrigOpsHoldProbe {
///     fn atan(&self) -> __FandheTrigOpsHoldMarker;
///     fn asin(&self) -> __FandheTrigOpsHoldMarker;
///     fn acos(&self) -> __FandheTrigOpsHoldMarker;
///     fn sinh(&self) -> __FandheTrigOpsHoldMarker;
///     fn cosh(&self) -> __FandheTrigOpsHoldMarker;
///     fn asinh(&self) -> __FandheTrigOpsHoldMarker;
///     fn acosh(&self) -> __FandheTrigOpsHoldMarker;
///     fn atanh(&self) -> __FandheTrigOpsHoldMarker;
///     fn atan2(&self) -> __FandheTrigOpsHoldMarker;
/// }
///
/// impl<'t> __FandheTrigOpsHoldProbe for fandhe_ai::Var<'t> {
///     fn atan(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn asin(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn acos(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn sinh(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn cosh(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn asinh(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn acosh(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn atanh(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn atan2(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
/// }
///
/// impl __FandheTrigOpsHoldProbe for fandhe_ai::Tape {
///     fn atan(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn asin(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn acos(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn sinh(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn cosh(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn asinh(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn acosh(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn atanh(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
///     fn atan2(&self) -> __FandheTrigOpsHoldMarker {
///         __FandheTrigOpsHoldMarker
///     }
/// }
///
/// fn __probe_free_fns() {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     trig_ops::atan();
///     trig_ops::asin();
///     trig_ops::acos();
///     trig_ops::sinh();
///     trig_ops::cosh();
///     trig_ops::asinh();
///     trig_ops::acosh();
///     trig_ops::atanh();
///     trig_ops::atan2();
/// }
///
/// fn __probe_methods(v: &fandhe_ai::Var<'_>, tape: &fandhe_ai::Tape) {
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Var::atan(v);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Var::asin(v);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Var::acos(v);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Var::sinh(v);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Var::cosh(v);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Var::asinh(v);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Var::acosh(v);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Var::atanh(v);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Var::atan2(v);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Tape::atan(tape);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Tape::asin(tape);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Tape::acos(tape);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Tape::sinh(tape);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Tape::cosh(tape);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Tape::asinh(tape);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Tape::acosh(tape);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Tape::atanh(tape);
///     let _: __FandheTrigOpsHoldMarker = fandhe_ai::Tape::atan2(tape);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct TrigOpsHoldDoctestGuard;

/// 非有限値の判定・置換（`isnan`・`isinf`・`isfinite`・`nan_to_num`。イシュー #2635・
/// 親 #2625・ルート #2499 Phase 4）を facade 公開面から締め出す保留ガード
/// （`TrigOpsHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカル
/// モジュール `nonfinite_ops` と 4 メソッドを持つプローブ用トレイトを置き、
/// 修飾なしの関数呼び出しと `Var`／`Tape`／`Tensor<f32>`／`Tensor<bool>` の
/// 修飾付きメソッド呼び出しの両方を行う。facade が同名のモジュール・関数を glob
/// 可能な位置へ公開するか、これらの型へ同名の inherent メソッドを公開すると、
/// 名前解決の曖昧性または呼び出しシグネチャの不一致でエラーコードに依存せず
/// コンパイルが失敗する（`Tensor` は facade から再エクスポートされるため、
/// `tensor-core` 側への同名メソッド追加も検出する）。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::nonfinite_ops`・
/// `fandhe_ai_tensor_core::ScalarUnaryOp` の追加 variant）。保留対象は facade
/// 公開面（`Var::isnan` 等の委譲メソッド）のみで、公開形は未承認（承認依頼は
/// #2677・公開自体は承認後の #2678。推奨案は
/// `docs/autodiff-nonfinite-ops-decision.md` §7。同記録は推奨案の記録であり
/// 承認記録ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// nonfinite_ops_hold_doctest_globs_all_pub_modules`・
/// `nonfinite_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_nonfinite_ops`・
/// `workspace_declares_nonfinite_ops_fn_names_only_in_allowed_locations`）との
/// 多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の
/// 対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_nonfinite_hold_probe {
///     pub mod nonfinite_ops {
///         pub fn isnan() {}
///         pub fn isinf() {}
///         pub fn isfinite() {}
///         pub fn nan_to_num() {}
///     }
/// }
/// use __fandhe_nonfinite_hold_probe::*;
///
/// struct __FandheNonfiniteOpsHoldMarker;
///
/// trait __FandheNonfiniteOpsHoldProbe {
///     fn isnan(&self) -> __FandheNonfiniteOpsHoldMarker;
///     fn isinf(&self) -> __FandheNonfiniteOpsHoldMarker;
///     fn isfinite(&self) -> __FandheNonfiniteOpsHoldMarker;
///     fn nan_to_num(&self) -> __FandheNonfiniteOpsHoldMarker;
/// }
///
/// impl<'t> __FandheNonfiniteOpsHoldProbe for fandhe_ai::Var<'t> {
///     fn isnan(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
///     fn isinf(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
///     fn isfinite(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
///     fn nan_to_num(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
/// }
///
/// impl __FandheNonfiniteOpsHoldProbe for fandhe_ai::Tape {
///     fn isnan(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
///     fn isinf(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
///     fn isfinite(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
///     fn nan_to_num(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
/// }
///
/// impl __FandheNonfiniteOpsHoldProbe for fandhe_ai::Tensor<f32> {
///     fn isnan(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
///     fn isinf(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
///     fn isfinite(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
///     fn nan_to_num(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
/// }
///
/// impl __FandheNonfiniteOpsHoldProbe for fandhe_ai::Tensor<bool> {
///     fn isnan(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
///     fn isinf(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
///     fn isfinite(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
///     fn nan_to_num(&self) -> __FandheNonfiniteOpsHoldMarker {
///         __FandheNonfiniteOpsHoldMarker
///     }
/// }
///
/// fn __probe_free_fns() {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     nonfinite_ops::isnan();
///     nonfinite_ops::isinf();
///     nonfinite_ops::isfinite();
///     nonfinite_ops::nan_to_num();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
///     tb: &fandhe_ai::Tensor<bool>,
/// ) {
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Var::isnan(v);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Var::isinf(v);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Var::isfinite(v);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Var::nan_to_num(v);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Tape::isnan(tape);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Tape::isinf(tape);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Tape::isfinite(tape);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Tape::nan_to_num(tape);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Tensor::<f32>::isnan(tf);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Tensor::<f32>::isinf(tf);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Tensor::<f32>::isfinite(tf);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Tensor::<f32>::nan_to_num(tf);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Tensor::<bool>::isnan(tb);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Tensor::<bool>::isinf(tb);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Tensor::<bool>::isfinite(tb);
///     let _: __FandheNonfiniteOpsHoldMarker = fandhe_ai::Tensor::<bool>::nan_to_num(tb);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct NonfiniteOpsHoldDoctestGuard;

/// 累積最大・最小・累積 logsumexp（`cummax`・`cummin`・`logcumsumexp`。イシュー
/// #2636・親 #2625・ルート #2499 Phase 4）を facade 公開面から締め出す保留ガード
/// （`NonfiniteOpsHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカル
/// モジュール `cumulative_ops`／`cumulative` と 3 メソッドを持つプローブ用
/// トレイトを置き、修飾なしの関数呼び出しと `Var`／`Tape`／`Tensor<f32>` の
/// 修飾付きメソッド呼び出しの両方を行う。facade が同名のモジュール・関数を glob
/// 可能な位置へ公開するか、これらの型へ同名の inherent メソッドを公開すると、
/// 名前解決の曖昧性または呼び出しシグネチャの不一致でエラーコードに依存せず
/// コンパイルが失敗する（`Tensor` は facade から再エクスポートされるため、
/// `tensor-core` 側への同名メソッド追加も検出する）。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::cumulative_ops`・
/// `fandhe_ai_tensor_core::cumulative`・`BackendOps::scan_*`）。保留対象は facade
/// 公開面（`Var::cummax` 等の委譲メソッド）のみで、公開形は未承認（承認依頼は
/// #2677・公開自体は承認後の #2678。推奨案は
/// `docs/autodiff-cumulative-ops-decision.md` §7。同記録は推奨案の記録であり
/// 承認記録ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// cumulative_ops_hold_doctest_globs_all_pub_modules`・
/// `cumulative_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_cumulative_ops`・
/// `workspace_declares_cumulative_ops_fn_names_only_in_allowed_locations`）との
/// 多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の
/// 対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_cumulative_hold_probe {
///     pub mod cumulative_ops {
///         pub fn cummax() {}
///         pub fn cummin() {}
///         pub fn logcumsumexp() {}
///     }
///     pub mod cumulative {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_cumulative_hold_probe::*;
///
/// struct __FandheCumulativeOpsHoldMarker;
///
/// trait __FandheCumulativeOpsHoldProbe {
///     fn cummax(&self) -> __FandheCumulativeOpsHoldMarker;
///     fn cummin(&self) -> __FandheCumulativeOpsHoldMarker;
///     fn logcumsumexp(&self) -> __FandheCumulativeOpsHoldMarker;
/// }
///
/// impl<'t> __FandheCumulativeOpsHoldProbe for fandhe_ai::Var<'t> {
///     fn cummax(&self) -> __FandheCumulativeOpsHoldMarker {
///         __FandheCumulativeOpsHoldMarker
///     }
///     fn cummin(&self) -> __FandheCumulativeOpsHoldMarker {
///         __FandheCumulativeOpsHoldMarker
///     }
///     fn logcumsumexp(&self) -> __FandheCumulativeOpsHoldMarker {
///         __FandheCumulativeOpsHoldMarker
///     }
/// }
///
/// impl __FandheCumulativeOpsHoldProbe for fandhe_ai::Tape {
///     fn cummax(&self) -> __FandheCumulativeOpsHoldMarker {
///         __FandheCumulativeOpsHoldMarker
///     }
///     fn cummin(&self) -> __FandheCumulativeOpsHoldMarker {
///         __FandheCumulativeOpsHoldMarker
///     }
///     fn logcumsumexp(&self) -> __FandheCumulativeOpsHoldMarker {
///         __FandheCumulativeOpsHoldMarker
///     }
/// }
///
/// impl __FandheCumulativeOpsHoldProbe for fandhe_ai::Tensor<f32> {
///     fn cummax(&self) -> __FandheCumulativeOpsHoldMarker {
///         __FandheCumulativeOpsHoldMarker
///     }
///     fn cummin(&self) -> __FandheCumulativeOpsHoldMarker {
///         __FandheCumulativeOpsHoldMarker
///     }
///     fn logcumsumexp(&self) -> __FandheCumulativeOpsHoldMarker {
///         __FandheCumulativeOpsHoldMarker
///     }
/// }
///
/// fn __probe_free_fns() {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     cumulative_ops::cummax();
///     cumulative_ops::cummin();
///     cumulative_ops::logcumsumexp();
///     cumulative::__mark();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
/// ) {
///     let _: __FandheCumulativeOpsHoldMarker = fandhe_ai::Var::cummax(v);
///     let _: __FandheCumulativeOpsHoldMarker = fandhe_ai::Var::cummin(v);
///     let _: __FandheCumulativeOpsHoldMarker = fandhe_ai::Var::logcumsumexp(v);
///     let _: __FandheCumulativeOpsHoldMarker = fandhe_ai::Tape::cummax(tape);
///     let _: __FandheCumulativeOpsHoldMarker = fandhe_ai::Tape::cummin(tape);
///     let _: __FandheCumulativeOpsHoldMarker = fandhe_ai::Tape::logcumsumexp(tape);
///     let _: __FandheCumulativeOpsHoldMarker = fandhe_ai::Tensor::<f32>::cummax(tf);
///     let _: __FandheCumulativeOpsHoldMarker = fandhe_ai::Tensor::<f32>::cummin(tf);
///     let _: __FandheCumulativeOpsHoldMarker = fandhe_ai::Tensor::<f32>::logcumsumexp(tf);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct CumulativeOpsHoldDoctestGuard;

/// 順序統計・NaN 無視縮約（`median`・`kthvalue`・`quantile`・`nanmean`・`nansum`。
/// イシュー #2637・親 #2625・ルート #2499 Phase 4）を facade 公開面から締め出す保留
/// ガード（`CumulativeOpsHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカル
/// モジュール `stat_reduce_ops`／`stat_reduce`・型 `QuantileInterpolation`／
/// `StatReduceError`・6 メソッドを持つプローブ用トレイトを置き、修飾なしの関数呼び出しと
/// `Var`／`Tape`／`Tensor<f32>` の修飾付きメソッド呼び出しの両方を行う。facade が
/// 同名のモジュール・型・関数を glob 可能な位置へ公開するか、これらの型へ同名の
/// inherent メソッドを公開すると、名前解決の曖昧性または呼び出しシグネチャの不一致で
/// エラーコードに依存せずコンパイルが失敗する（`Tensor` は facade から再エクスポート
/// されるため、`tensor-core` 側への同名メソッド追加も検出する）。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::stat_reduce_ops`・
/// `fandhe_ai_tensor_core::{stat_reduce, QuantileInterpolation, StatReduceError}`・
/// `BackendOps::stat_*`）。保留対象は facade 公開面（`Var::median` 等の委譲メソッドと
/// 引数型 `QuantileInterpolation` の再エクスポート）のみで、公開形は未承認（承認依頼は
/// #2677・公開自体は承認後の #2678。推奨案は
/// `docs/autodiff-stat-reduce-ops-decision.md` §7。同記録は推奨案の記録であり承認記録
/// ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// stat_reduce_ops_hold_doctest_globs_all_pub_modules`・
/// `stat_reduce_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_stat_reduce_ops`・
/// `workspace_declares_stat_reduce_ops_fn_names_only_in_allowed_locations`）との
/// 多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の
/// 対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_stat_reduce_hold_probe {
///     pub struct QuantileInterpolation;
///     pub struct StatReduceError;
///     pub mod stat_reduce_ops {
///         pub fn median() {}
///         pub fn median_with_indices() {}
///         pub fn kthvalue() {}
///         pub fn quantile() {}
///         pub fn nanmean() {}
///         pub fn nansum() {}
///     }
///     pub mod stat_reduce {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_stat_reduce_hold_probe::*;
///
/// struct __FandheStatReduceOpsHoldMarker;
///
/// trait __FandheStatReduceOpsHoldProbe {
///     fn median(&self) -> __FandheStatReduceOpsHoldMarker;
///     fn median_with_indices(&self) -> __FandheStatReduceOpsHoldMarker;
///     fn kthvalue(&self) -> __FandheStatReduceOpsHoldMarker;
///     fn quantile(&self) -> __FandheStatReduceOpsHoldMarker;
///     fn nanmean(&self) -> __FandheStatReduceOpsHoldMarker;
///     fn nansum(&self) -> __FandheStatReduceOpsHoldMarker;
/// }
///
/// impl<'t> __FandheStatReduceOpsHoldProbe for fandhe_ai::Var<'t> {
///     fn median(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn median_with_indices(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn kthvalue(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn quantile(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn nanmean(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn nansum(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
/// }
///
/// impl __FandheStatReduceOpsHoldProbe for fandhe_ai::Tape {
///     fn median(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn median_with_indices(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn kthvalue(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn quantile(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn nanmean(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn nansum(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
/// }
///
/// impl __FandheStatReduceOpsHoldProbe for fandhe_ai::Tensor<f32> {
///     fn median(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn median_with_indices(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn kthvalue(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn quantile(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn nanmean(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
///     fn nansum(&self) -> __FandheStatReduceOpsHoldMarker {
///         __FandheStatReduceOpsHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(_: QuantileInterpolation, _: StatReduceError) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     stat_reduce_ops::median();
///     stat_reduce_ops::median_with_indices();
///     stat_reduce_ops::kthvalue();
///     stat_reduce_ops::quantile();
///     stat_reduce_ops::nanmean();
///     stat_reduce_ops::nansum();
///     stat_reduce::__mark();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
/// ) {
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Var::median(v);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Var::median_with_indices(v);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Var::kthvalue(v);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Var::quantile(v);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Var::nanmean(v);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Var::nansum(v);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Tape::median(tape);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Tape::median_with_indices(tape);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Tape::kthvalue(tape);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Tape::quantile(tape);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Tape::nanmean(tape);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Tape::nansum(tape);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Tensor::<f32>::median(tf);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Tensor::<f32>::median_with_indices(tf);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Tensor::<f32>::kthvalue(tf);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Tensor::<f32>::quantile(tf);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Tensor::<f32>::nanmean(tf);
///     let _: __FandheStatReduceOpsHoldMarker = fandhe_ai::Tensor::<f32>::nansum(tf);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct StatReduceOpsHoldDoctestGuard;

/// ヒストグラム・二分探索系（`histc`・`bincount`・`searchsorted`・`bucketize`。
/// イシュー #2638・親 #2625・ルート #2499 Phase 4）を facade 公開面から締め出す保留
/// ガード（`StatReduceOpsHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカル
/// モジュール `binning_ops`／`binning`・型 `BinningError`・5 メソッドを持つ
/// プローブ用トレイトを置き、修飾なしの関数呼び出しと `Var`／`Tape`／`Tensor<f32>`／
/// `Tensor<i32>`（`bincount` の入力型）の修飾付きメソッド呼び出しの両方を行う。facade が
/// 同名のモジュール・型・関数を glob 可能な位置へ公開するか、これらの型へ同名の
/// inherent メソッドを公開すると、名前解決の曖昧性または呼び出しシグネチャの不一致で
/// エラーコードに依存せずコンパイルが失敗する（`Tensor` は facade から再エクスポート
/// されるため、`tensor-core` 側への同名メソッド追加も検出する）。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::binning_ops`・
/// `fandhe_ai_tensor_core::{binning, BinningError}`・`BackendOps::binning_*`）。保留対象は
/// facade 公開面（`Var::histc` 等の委譲メソッド）のみで、公開形は未承認（承認依頼は
/// #2677・公開自体は承認後の #2678・#2679。推奨案は
/// `docs/autodiff-binning-ops-decision.md` §7。同記録は推奨案の記録であり承認記録
/// ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// binning_ops_hold_doctest_globs_all_pub_modules`・
/// `binning_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_binning_ops`・
/// `workspace_declares_binning_ops_fn_names_only_in_allowed_locations`）との
/// 多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の
/// 対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_binning_hold_probe {
///     pub struct BinningError;
///     pub mod binning_ops {
///         pub fn histc() {}
///         pub fn bincount() {}
///         pub fn bincount_weighted() {}
///         pub fn searchsorted() {}
///         pub fn bucketize() {}
///     }
///     pub mod binning {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_binning_hold_probe::*;
///
/// struct __FandheBinningOpsHoldMarker;
///
/// trait __FandheBinningOpsHoldProbe {
///     fn histc(&self) -> __FandheBinningOpsHoldMarker;
///     fn bincount(&self) -> __FandheBinningOpsHoldMarker;
///     fn bincount_weighted(&self) -> __FandheBinningOpsHoldMarker;
///     fn searchsorted(&self) -> __FandheBinningOpsHoldMarker;
///     fn bucketize(&self) -> __FandheBinningOpsHoldMarker;
/// }
///
/// impl<'t> __FandheBinningOpsHoldProbe for fandhe_ai::Var<'t> {
///     fn histc(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn bincount(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn bincount_weighted(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn searchsorted(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn bucketize(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
/// }
///
/// impl __FandheBinningOpsHoldProbe for fandhe_ai::Tape {
///     fn histc(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn bincount(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn bincount_weighted(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn searchsorted(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn bucketize(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
/// }
///
/// impl __FandheBinningOpsHoldProbe for fandhe_ai::Tensor<f32> {
///     fn histc(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn bincount(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn bincount_weighted(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn searchsorted(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn bucketize(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
/// }
///
/// impl __FandheBinningOpsHoldProbe for fandhe_ai::Tensor<i32> {
///     fn histc(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn bincount(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn bincount_weighted(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn searchsorted(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
///     fn bucketize(&self) -> __FandheBinningOpsHoldMarker {
///         __FandheBinningOpsHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(_: BinningError) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     binning_ops::histc();
///     binning_ops::bincount();
///     binning_ops::bincount_weighted();
///     binning_ops::searchsorted();
///     binning_ops::bucketize();
///     binning::__mark();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
///     ti: &fandhe_ai::Tensor<i32>,
/// ) {
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Var::histc(v);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Var::bincount(v);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Var::bincount_weighted(v);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Var::searchsorted(v);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Var::bucketize(v);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tape::histc(tape);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tape::bincount(tape);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tape::bincount_weighted(tape);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tape::searchsorted(tape);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tape::bucketize(tape);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tensor::<f32>::histc(tf);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tensor::<f32>::bincount(tf);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tensor::<f32>::bincount_weighted(tf);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tensor::<f32>::searchsorted(tf);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tensor::<f32>::bucketize(tf);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tensor::<i32>::histc(ti);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tensor::<i32>::bincount(ti);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tensor::<i32>::bincount_weighted(ti);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tensor::<i32>::searchsorted(ti);
///     let _: __FandheBinningOpsHoldMarker = fandhe_ai::Tensor::<i32>::bucketize(ti);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct BinningOpsHoldDoctestGuard;

/// `Var`／`Tape` の低精度 forward 入口（`matmul_low_precision`・
/// `add_low_precision`・`mul_low_precision`・`relu_low_precision`・
/// `exp_low_precision`・`tanh_low_precision`。イシュー #2628・親 #2626・
/// ルート #2499 Phase 4）を facade 公開面から締め出す保留ガード（
/// `FftOpsHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカル
/// モジュール `low_precision_ops`（と 6 関数のルート直下定義）および 6 メソッド
/// を持つプローブ用トレイトを置き、修飾なし／修飾付きの両方で呼ぶ。facade が
/// 同名のモジュール・関数を glob 可能な位置へ公開するか、`Var`／`Tape` へ同名の
/// inherent メソッドを公開すると、名前解決の曖昧性または呼び出しシグネチャの
/// 不一致でエラーコードに依存せずコンパイルが失敗する。内部の
/// `pub(crate)` な `Var::matmul_low_precision` は別クレートから見えず、同名
/// トレイトメソッドを隠さないため 6 名すべてを対象にできる（`pub` に
/// すると inherent が優先され本プローブが失敗する）。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::low_precision_ops`・
/// `fandhe_ai_tensor_core` の `*_low_precision`）。保留対象は facade 公開面
/// （`Var::{matmul,add,mul,relu,exp,tanh}_low_precision(.., dtype)` の委譲
/// メソッド）のみで、公開形は未承認（承認依頼は #2677・公開自体は承認後の
/// #2678。推奨案は `docs/autodiff-low-precision-op-extension-decision.md`。
/// 同記録は推奨案の記録であり承認記録ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// var_low_precision_ops_hold_doctest_globs_all_pub_modules`・
/// `var_low_precision_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_low_precision_ops`・
/// `workspace_declares_low_precision_ops_fn_names_only_in_allowed_locations`）
/// との多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の
/// 対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_low_precision_ops_hold_probe {
///     pub mod low_precision_ops {
///         pub fn matmul_low_precision() {}
///         pub fn add_low_precision() {}
///         pub fn mul_low_precision() {}
///         pub fn relu_low_precision() {}
///         pub fn exp_low_precision() {}
///         pub fn tanh_low_precision() {}
///     }
///     pub fn matmul_low_precision() {}
///     pub fn add_low_precision() {}
///     pub fn mul_low_precision() {}
///     pub fn relu_low_precision() {}
///     pub fn exp_low_precision() {}
///     pub fn tanh_low_precision() {}
/// }
/// use __fandhe_low_precision_ops_hold_probe::*;
///
/// struct __FandheLowPrecisionHoldMarker;
///
/// trait __FandheLowPrecisionHoldProbe {
///     fn matmul_low_precision(&self) -> __FandheLowPrecisionHoldMarker;
///     fn add_low_precision(&self) -> __FandheLowPrecisionHoldMarker;
///     fn mul_low_precision(&self) -> __FandheLowPrecisionHoldMarker;
///     fn relu_low_precision(&self) -> __FandheLowPrecisionHoldMarker;
///     fn exp_low_precision(&self) -> __FandheLowPrecisionHoldMarker;
///     fn tanh_low_precision(&self) -> __FandheLowPrecisionHoldMarker;
/// }
///
/// impl<'t> __FandheLowPrecisionHoldProbe for fandhe_ai::Var<'t> {
///     fn matmul_low_precision(&self) -> __FandheLowPrecisionHoldMarker {
///         __FandheLowPrecisionHoldMarker
///     }
///     fn add_low_precision(&self) -> __FandheLowPrecisionHoldMarker {
///         __FandheLowPrecisionHoldMarker
///     }
///     fn mul_low_precision(&self) -> __FandheLowPrecisionHoldMarker {
///         __FandheLowPrecisionHoldMarker
///     }
///     fn relu_low_precision(&self) -> __FandheLowPrecisionHoldMarker {
///         __FandheLowPrecisionHoldMarker
///     }
///     fn exp_low_precision(&self) -> __FandheLowPrecisionHoldMarker {
///         __FandheLowPrecisionHoldMarker
///     }
///     fn tanh_low_precision(&self) -> __FandheLowPrecisionHoldMarker {
///         __FandheLowPrecisionHoldMarker
///     }
/// }
///
/// impl __FandheLowPrecisionHoldProbe for fandhe_ai::Tape {
///     fn matmul_low_precision(&self) -> __FandheLowPrecisionHoldMarker {
///         __FandheLowPrecisionHoldMarker
///     }
///     fn add_low_precision(&self) -> __FandheLowPrecisionHoldMarker {
///         __FandheLowPrecisionHoldMarker
///     }
///     fn mul_low_precision(&self) -> __FandheLowPrecisionHoldMarker {
///         __FandheLowPrecisionHoldMarker
///     }
///     fn relu_low_precision(&self) -> __FandheLowPrecisionHoldMarker {
///         __FandheLowPrecisionHoldMarker
///     }
///     fn exp_low_precision(&self) -> __FandheLowPrecisionHoldMarker {
///         __FandheLowPrecisionHoldMarker
///     }
///     fn tanh_low_precision(&self) -> __FandheLowPrecisionHoldMarker {
///         __FandheLowPrecisionHoldMarker
///     }
/// }
///
/// fn __probe_free_fns() {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     low_precision_ops::matmul_low_precision();
///     low_precision_ops::add_low_precision();
///     low_precision_ops::mul_low_precision();
///     low_precision_ops::relu_low_precision();
///     low_precision_ops::exp_low_precision();
///     low_precision_ops::tanh_low_precision();
///     matmul_low_precision();
///     add_low_precision();
///     mul_low_precision();
///     relu_low_precision();
///     exp_low_precision();
///     tanh_low_precision();
/// }
///
/// fn __probe_methods(v: &fandhe_ai::Var<'_>, tape: &fandhe_ai::Tape) {
///     let _: __FandheLowPrecisionHoldMarker = fandhe_ai::Var::matmul_low_precision(v);
///     let _: __FandheLowPrecisionHoldMarker = fandhe_ai::Var::add_low_precision(v);
///     let _: __FandheLowPrecisionHoldMarker = fandhe_ai::Var::mul_low_precision(v);
///     let _: __FandheLowPrecisionHoldMarker = fandhe_ai::Var::relu_low_precision(v);
///     let _: __FandheLowPrecisionHoldMarker = fandhe_ai::Var::exp_low_precision(v);
///     let _: __FandheLowPrecisionHoldMarker = fandhe_ai::Var::tanh_low_precision(v);
///     let _: __FandheLowPrecisionHoldMarker = fandhe_ai::Tape::matmul_low_precision(tape);
///     let _: __FandheLowPrecisionHoldMarker = fandhe_ai::Tape::add_low_precision(tape);
///     let _: __FandheLowPrecisionHoldMarker = fandhe_ai::Tape::mul_low_precision(tape);
///     let _: __FandheLowPrecisionHoldMarker = fandhe_ai::Tape::relu_low_precision(tape);
///     let _: __FandheLowPrecisionHoldMarker = fandhe_ai::Tape::exp_low_precision(tape);
///     let _: __FandheLowPrecisionHoldMarker = fandhe_ai::Tape::tanh_low_precision(tape);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct VarLowPrecisionOpsHoldDoctestGuard;

/// 形状演算 6 種（`unbind`・`movedim`・`swapaxes`・`tensor_split`・`meshgrid`・`rot90`。
/// イシュー #2639・親 #2625・ルート #2499 Phase 4）を facade 公開面から締め出す保留
/// ガード（`StatReduceOpsHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカル
/// モジュール `shape_view_ops`・型 `MeshgridIndexing`・7 メソッドを持つプローブ用
/// トレイトを置き、修飾なしの関数呼び出しと `Var`／`Tape`／`Tensor<f32>` の修飾付き
/// メソッド呼び出しの両方を行う。facade が同名のモジュール・型・関数を glob 可能な位置へ
/// 公開するか、これらの型へ同名の inherent メソッドを公開すると、名前解決の曖昧性または
/// 呼び出しシグネチャの不一致でエラーコードに依存せずコンパイルが失敗する（`Tensor` は
/// facade から再エクスポートされるため、`tensor-core` 側への同名メソッド追加も検出する）。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::shape_view_ops`。新規 `Op`・
/// `BackendOps` メソッドはない）。保留対象は facade 公開面（`Var::unbind` 等の委譲メソッドと
/// 引数型 `MeshgridIndexing` の再エクスポート）のみで、公開形は未承認（承認依頼は
/// #2677・公開自体は承認後の #2678。推奨案は
/// `docs/autodiff-shape-view-ops-decision.md` §7。同記録は推奨案の記録であり承認記録
/// ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// shape_view_ops_hold_doctest_globs_all_pub_modules`・
/// `shape_view_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_shape_view_ops`・
/// `workspace_declares_shape_view_ops_fn_names_only_in_allowed_locations`）との
/// 多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の
/// 対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_shape_view_hold_probe {
///     pub struct MeshgridIndexing;
///     pub mod shape_view_ops {
///         pub fn unbind() {}
///         pub fn movedim() {}
///         pub fn swapaxes() {}
///         pub fn tensor_split() {}
///         pub fn tensor_split_indices() {}
///         pub fn meshgrid() {}
///         pub fn rot90() {}
///     }
/// }
/// use __fandhe_shape_view_hold_probe::*;
///
/// struct __FandheShapeViewOpsHoldMarker;
///
/// trait __FandheShapeViewOpsHoldProbe {
///     fn unbind(&self) -> __FandheShapeViewOpsHoldMarker;
///     fn movedim(&self) -> __FandheShapeViewOpsHoldMarker;
///     fn swapaxes(&self) -> __FandheShapeViewOpsHoldMarker;
///     fn tensor_split(&self) -> __FandheShapeViewOpsHoldMarker;
///     fn tensor_split_indices(&self) -> __FandheShapeViewOpsHoldMarker;
///     fn meshgrid(&self) -> __FandheShapeViewOpsHoldMarker;
///     fn rot90(&self) -> __FandheShapeViewOpsHoldMarker;
/// }
///
/// impl<'t> __FandheShapeViewOpsHoldProbe for fandhe_ai::Var<'t> {
///     fn unbind(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn movedim(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn swapaxes(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn tensor_split(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn tensor_split_indices(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn meshgrid(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn rot90(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
/// }
///
/// impl __FandheShapeViewOpsHoldProbe for fandhe_ai::Tape {
///     fn unbind(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn movedim(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn swapaxes(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn tensor_split(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn tensor_split_indices(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn meshgrid(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn rot90(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
/// }
///
/// impl __FandheShapeViewOpsHoldProbe for fandhe_ai::Tensor<f32> {
///     fn unbind(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn movedim(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn swapaxes(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn tensor_split(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn tensor_split_indices(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn meshgrid(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
///     fn rot90(&self) -> __FandheShapeViewOpsHoldMarker {
///         __FandheShapeViewOpsHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(_: MeshgridIndexing) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     shape_view_ops::unbind();
///     shape_view_ops::movedim();
///     shape_view_ops::swapaxes();
///     shape_view_ops::tensor_split();
///     shape_view_ops::tensor_split_indices();
///     shape_view_ops::meshgrid();
///     shape_view_ops::rot90();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
/// ) {
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Var::unbind(v);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Var::movedim(v);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Var::swapaxes(v);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Var::tensor_split(v);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Var::tensor_split_indices(v);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Var::meshgrid(v);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Var::rot90(v);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tape::unbind(tape);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tape::movedim(tape);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tape::swapaxes(tape);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tape::tensor_split(tape);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tape::tensor_split_indices(tape);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tape::meshgrid(tape);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tape::rot90(tape);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tensor::<f32>::unbind(tf);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tensor::<f32>::movedim(tf);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tensor::<f32>::swapaxes(tf);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tensor::<f32>::tensor_split(tf);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tensor::<f32>::tensor_split_indices(tf);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tensor::<f32>::meshgrid(tf);
///     let _: __FandheShapeViewOpsHoldMarker = fandhe_ai::Tensor::<f32>::rot90(tf);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct ShapeViewOpsHoldDoctestGuard;

/// 索引付き更新 4 種（`scatter_reduce`・`index_add`・`index_copy`・`masked_scatter`。
/// イシュー #2641・親 #2625・ルート #2499 Phase 4）を facade 公開面から締め出す保留
/// ガード（`ShapeViewOpsHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカル
/// モジュール `indexed_update_ops`・`indexed_update`・型 `ScatterReduceMode`・4 メソッドを持つ
/// プローブ用トレイトを置き、修飾なしの関数呼び出しと `Var`／`Tape`／`Tensor<f32>` の修飾付き
/// メソッド呼び出しの両方を行う。facade が同名のモジュール・型・関数を glob 可能な位置へ
/// 公開するか、これらの型へ同名の inherent メソッドを公開すると、名前解決の曖昧性または
/// 呼び出しシグネチャの不一致でエラーコードに依存せずコンパイルが失敗する（`Tensor` は
/// facade から再エクスポートされるため、`tensor-core` 側への同名メソッド追加も検出する）。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::indexed_update_ops`・
/// `fandhe_ai_tensor_core::indexed_update`。新規 `Op` は `scatter_reduce` 用の 1 つ、
/// `BackendOps` メソッドは `indexed_scatter_reduce` のみ）。保留対象は facade 公開面
/// （`Var::scatter_reduce` 等の委譲メソッドと引数型 `ScatterReduceMode` の再エクスポート）のみで、
/// 公開形は未承認（承認依頼は #2677・公開自体は承認後の #2678・#2679。推奨案は
/// `docs/autodiff-indexed-update-ops-decision.md` §7。同記録は推奨案の記録であり承認記録
/// ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// indexed_update_ops_hold_doctest_globs_all_pub_modules`・
/// `indexed_update_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_indexed_update_ops`・
/// `workspace_declares_indexed_update_ops_fn_names_only_in_allowed_locations`）との
/// 多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の
/// 対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_indexed_update_hold_probe {
///     pub struct ScatterReduceMode;
///     pub mod indexed_update_ops {
///         pub fn scatter_reduce() {}
///         pub fn index_add() {}
///         pub fn index_copy() {}
///         pub fn masked_scatter() {}
///     }
///     pub mod indexed_update {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_indexed_update_hold_probe::*;
///
/// struct __FandheIndexedUpdateOpsHoldMarker;
///
/// trait __FandheIndexedUpdateOpsHoldProbe {
///     fn scatter_reduce(&self) -> __FandheIndexedUpdateOpsHoldMarker;
///     fn index_add(&self) -> __FandheIndexedUpdateOpsHoldMarker;
///     fn index_copy(&self) -> __FandheIndexedUpdateOpsHoldMarker;
///     fn masked_scatter(&self) -> __FandheIndexedUpdateOpsHoldMarker;
/// }
///
/// impl<'t> __FandheIndexedUpdateOpsHoldProbe for fandhe_ai::Var<'t> {
///     fn scatter_reduce(&self) -> __FandheIndexedUpdateOpsHoldMarker {
///         __FandheIndexedUpdateOpsHoldMarker
///     }
///     fn index_add(&self) -> __FandheIndexedUpdateOpsHoldMarker {
///         __FandheIndexedUpdateOpsHoldMarker
///     }
///     fn index_copy(&self) -> __FandheIndexedUpdateOpsHoldMarker {
///         __FandheIndexedUpdateOpsHoldMarker
///     }
///     fn masked_scatter(&self) -> __FandheIndexedUpdateOpsHoldMarker {
///         __FandheIndexedUpdateOpsHoldMarker
///     }
/// }
///
/// impl __FandheIndexedUpdateOpsHoldProbe for fandhe_ai::Tape {
///     fn scatter_reduce(&self) -> __FandheIndexedUpdateOpsHoldMarker {
///         __FandheIndexedUpdateOpsHoldMarker
///     }
///     fn index_add(&self) -> __FandheIndexedUpdateOpsHoldMarker {
///         __FandheIndexedUpdateOpsHoldMarker
///     }
///     fn index_copy(&self) -> __FandheIndexedUpdateOpsHoldMarker {
///         __FandheIndexedUpdateOpsHoldMarker
///     }
///     fn masked_scatter(&self) -> __FandheIndexedUpdateOpsHoldMarker {
///         __FandheIndexedUpdateOpsHoldMarker
///     }
/// }
///
/// impl __FandheIndexedUpdateOpsHoldProbe for fandhe_ai::Tensor<f32> {
///     fn scatter_reduce(&self) -> __FandheIndexedUpdateOpsHoldMarker {
///         __FandheIndexedUpdateOpsHoldMarker
///     }
///     fn index_add(&self) -> __FandheIndexedUpdateOpsHoldMarker {
///         __FandheIndexedUpdateOpsHoldMarker
///     }
///     fn index_copy(&self) -> __FandheIndexedUpdateOpsHoldMarker {
///         __FandheIndexedUpdateOpsHoldMarker
///     }
///     fn masked_scatter(&self) -> __FandheIndexedUpdateOpsHoldMarker {
///         __FandheIndexedUpdateOpsHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(_: ScatterReduceMode) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     indexed_update_ops::scatter_reduce();
///     indexed_update_ops::index_add();
///     indexed_update_ops::index_copy();
///     indexed_update_ops::masked_scatter();
///     indexed_update::__mark();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
/// ) {
///     let _: __FandheIndexedUpdateOpsHoldMarker = fandhe_ai::Var::scatter_reduce(v);
///     let _: __FandheIndexedUpdateOpsHoldMarker = fandhe_ai::Var::index_add(v);
///     let _: __FandheIndexedUpdateOpsHoldMarker = fandhe_ai::Var::index_copy(v);
///     let _: __FandheIndexedUpdateOpsHoldMarker = fandhe_ai::Var::masked_scatter(v);
///     let _: __FandheIndexedUpdateOpsHoldMarker = fandhe_ai::Tape::scatter_reduce(tape);
///     let _: __FandheIndexedUpdateOpsHoldMarker = fandhe_ai::Tape::index_add(tape);
///     let _: __FandheIndexedUpdateOpsHoldMarker = fandhe_ai::Tape::index_copy(tape);
///     let _: __FandheIndexedUpdateOpsHoldMarker = fandhe_ai::Tape::masked_scatter(tape);
///     let _: __FandheIndexedUpdateOpsHoldMarker = fandhe_ai::Tensor::<f32>::scatter_reduce(tf);
///     let _: __FandheIndexedUpdateOpsHoldMarker = fandhe_ai::Tensor::<f32>::index_add(tf);
///     let _: __FandheIndexedUpdateOpsHoldMarker = fandhe_ai::Tensor::<f32>::index_copy(tf);
///     let _: __FandheIndexedUpdateOpsHoldMarker = fandhe_ai::Tensor::<f32>::masked_scatter(tf);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct IndexedUpdateOpsHoldDoctestGuard;

/// テンソル積・距離・外積 4 演算（`kron`・`tensordot`〈`tensordot_axes`〉・`cdist`・`cross`。
/// イシュー #2640・親 #2625・ルート #2499 Phase 4）を facade 公開面から締め出す保留
/// ガード（`ShapeViewOpsHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカル
/// モジュール `tensor_product_ops`・5 メソッドを持つプローブ用トレイトを置き、修飾なしの
/// 関数呼び出しと `Var`／`Tape`／`Tensor<f32>` の修飾付きメソッド呼び出しの両方を行う。
/// facade が同名のモジュール・関数を glob 可能な位置へ公開するか、これらの型へ同名の
/// inherent メソッドを公開すると、名前解決の曖昧性または呼び出しシグネチャの不一致で
/// エラーコードに依存せずコンパイルが失敗する（新規型はないため型のプローブは置かない）。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::tensor_product_ops`。新規 `Op`・
/// `BackendOps` メソッドはない）。保留対象は facade 公開面（`Var::kron` 等の委譲メソッド）
/// のみで、公開形は未承認（承認依頼は #2677・公開自体は承認後の #2678。推奨案は
/// `docs/autodiff-tensor-product-ops-decision.md` §7。同記録は推奨案の記録であり承認記録
/// ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// tensor_product_ops_hold_doctest_globs_all_pub_modules`・
/// `tensor_product_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_tensor_product_ops`・
/// `workspace_declares_tensor_product_ops_fn_names_only_in_allowed_locations`）との
/// 多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の
/// 対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_tensor_product_hold_probe {
///     pub mod tensor_product_ops {
///         pub fn kron() {}
///         pub fn tensordot() {}
///         pub fn tensordot_axes() {}
///         pub fn cdist() {}
///         pub fn cross() {}
///     }
/// }
/// use __fandhe_tensor_product_hold_probe::*;
///
/// struct __FandheTensorProductOpsHoldMarker;
///
/// trait __FandheTensorProductOpsHoldProbe {
///     fn kron(&self) -> __FandheTensorProductOpsHoldMarker;
///     fn tensordot(&self) -> __FandheTensorProductOpsHoldMarker;
///     fn tensordot_axes(&self) -> __FandheTensorProductOpsHoldMarker;
///     fn cdist(&self) -> __FandheTensorProductOpsHoldMarker;
///     fn cross(&self) -> __FandheTensorProductOpsHoldMarker;
/// }
///
/// impl<'t> __FandheTensorProductOpsHoldProbe for fandhe_ai::Var<'t> {
///     fn kron(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
///     fn tensordot(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
///     fn tensordot_axes(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
///     fn cdist(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
///     fn cross(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
/// }
///
/// impl __FandheTensorProductOpsHoldProbe for fandhe_ai::Tape {
///     fn kron(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
///     fn tensordot(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
///     fn tensordot_axes(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
///     fn cdist(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
///     fn cross(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
/// }
///
/// impl __FandheTensorProductOpsHoldProbe for fandhe_ai::Tensor<f32> {
///     fn kron(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
///     fn tensordot(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
///     fn tensordot_axes(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
///     fn cdist(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
///     fn cross(&self) -> __FandheTensorProductOpsHoldMarker {
///         __FandheTensorProductOpsHoldMarker
///     }
/// }
///
/// fn __probe_free_fns() {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     tensor_product_ops::kron();
///     tensor_product_ops::tensordot();
///     tensor_product_ops::tensordot_axes();
///     tensor_product_ops::cdist();
///     tensor_product_ops::cross();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
/// ) {
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Var::kron(v);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Var::tensordot(v);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Var::tensordot_axes(v);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Var::cdist(v);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Var::cross(v);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Tape::kron(tape);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Tape::tensordot(tape);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Tape::tensordot_axes(tape);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Tape::cdist(tape);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Tape::cross(tape);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Tensor::<f32>::kron(tf);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Tensor::<f32>::tensordot(tf);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Tensor::<f32>::tensordot_axes(tf);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Tensor::<f32>::cdist(tf);
///     let _: __FandheTensorProductOpsHoldMarker = fandhe_ai::Tensor::<f32>::cross(tf);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct TensorProductOpsHoldDoctestGuard;

/// `pad` の非定数モード（`pad_with_mode`・`PadMode`。reflect／replicate／circular。
/// イシュー #2642・親 #2625・ルート #2499 Phase 4）を facade 公開面から締め出す保留
/// ガード（`IndexedUpdateOpsHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカル
/// モジュール `pad_ops`／`pad_modes`・型 `PadMode`／`PadModeError`・メソッド
/// `pad_with_mode` を持つプローブ用トレイトを置き、修飾なしの関数呼び出しと
/// `Var`／`Tape`／`Tensor<f32>` の修飾付きメソッド呼び出しの両方を行う。facade が
/// 同名のモジュール・型・関数を glob 可能な位置へ公開するか、これらの型へ同名の
/// inherent メソッドを公開すると、名前解決の曖昧性または呼び出しシグネチャの不一致で
/// エラーコードに依存せずコンパイルが失敗する（`Tensor` は facade から再エクスポート
/// されるため、`tensor-core` 側への同名メソッド追加も検出する）。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::pad_ops`・
/// `fandhe_ai_tensor_core::{pad_modes, PadMode, PadModeError}`・
/// `BackendOps::pad_modes_forward`）。保留対象は facade 公開面
/// （`Var::pad_with_mode` の委譲メソッドと `PadMode` の再エクスポート）のみで、公開形は
/// 未承認（承認依頼は #2677・公開自体は承認後の #2678・#2679。推奨案は
/// `docs/autodiff-pad-modes-decision.md` §7。同記録は推奨案の記録であり承認記録
/// ではない）。既存の `Var::pad`（定数埋め）・`StftPadMode` には触れない。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// pad_modes_hold_doctest_globs_all_pub_modules`・
/// `pad_modes_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_pad_modes`・
/// `workspace_declares_pad_modes_fn_names_only_in_allowed_locations`）との
/// 多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の
/// 対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_pad_modes_hold_probe {
///     pub struct PadMode;
///     pub struct PadModeError;
///     pub mod pad_ops {
///         pub fn pad_with_mode() {}
///     }
///     pub mod pad_modes {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_pad_modes_hold_probe::*;
///
/// struct __FandhePadModesHoldMarker;
///
/// trait __FandhePadModesHoldProbe {
///     fn pad_with_mode(&self) -> __FandhePadModesHoldMarker;
/// }
///
/// impl<'t> __FandhePadModesHoldProbe for fandhe_ai::Var<'t> {
///     fn pad_with_mode(&self) -> __FandhePadModesHoldMarker {
///         __FandhePadModesHoldMarker
///     }
/// }
///
/// impl __FandhePadModesHoldProbe for fandhe_ai::Tape {
///     fn pad_with_mode(&self) -> __FandhePadModesHoldMarker {
///         __FandhePadModesHoldMarker
///     }
/// }
///
/// impl __FandhePadModesHoldProbe for fandhe_ai::Tensor<f32> {
///     fn pad_with_mode(&self) -> __FandhePadModesHoldMarker {
///         __FandhePadModesHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(_: PadMode, _: PadModeError) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     pad_ops::pad_with_mode();
///     pad_modes::__mark();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
/// ) {
///     let _: __FandhePadModesHoldMarker = fandhe_ai::Var::pad_with_mode(v);
///     let _: __FandhePadModesHoldMarker = fandhe_ai::Tape::pad_with_mode(tape);
///     let _: __FandhePadModesHoldMarker = fandhe_ai::Tensor::<f32>::pad_with_mode(tf);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct PadModesHoldDoctestGuard;

/// 3D プーリング（`max_pool3d`・`avg_pool3d`。イシュー #2643・親 #2625・ルート #2499
/// Phase 4）を facade 公開面から締め出す保留ガード（`PadModesHoldDoctestGuard` と同型の
/// 正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカル
/// モジュール `pool3d_ops`／`pool3d`・型 `Pool3dParams`／`MaxPool3d`／`AvgPool3d`・メソッド
/// `max_pool3d`／`avg_pool3d`（`Var`／`Tape`／`Tensor<f32>`）と
/// `add_max_pool3d`／`add_avg_pool3d`（`compat::Sequential`）を持つプローブ用トレイトを
/// 置き、修飾なしの関数呼び出しと修飾付きメソッド呼び出しの両方を行う。facade が同名の
/// モジュール・型・関数を glob 可能な位置へ公開するか、これらの型へ同名の inherent
/// メソッドを公開すると、名前解決の曖昧性または呼び出しシグネチャの不一致で
/// エラーコードに依存せずコンパイルが失敗する（`Tensor` は facade から再エクスポート
/// されるため、`tensor-core` 側への同名メソッド追加も検出する）。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::pool3d_ops`・
/// `fandhe_ai_tensor_core::pool3d`・`BackendOps::pool3d_max`／`pool3d_avg`）。保留対象は
/// facade 公開面（`Var::max_pool3d`／`avg_pool3d` の委譲メソッド）のみで、公開形は
/// 未承認（承認依頼は #2677・公開自体は承認後の #2678・#2679。推奨案は
/// `docs/autodiff-pool3d-ops-decision.md` §7。同記録は推奨案の記録であり承認記録では
/// ない）。層化（`compat::Sequential::add_max_pool3d`／`add_avg_pool3d`）も同じ保留に
/// 含める（#2679 の層結線がもう一方の漏出経路のため）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// pool3d_ops_hold_doctest_globs_all_pub_modules`・
/// `pool3d_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_pool3d_ops`・
/// `workspace_declares_pool3d_ops_fn_names_only_in_allowed_locations`）との
/// 多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の
/// 対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_pool3d_hold_probe {
///     pub struct Pool3dParams;
///     pub struct MaxPool3d;
///     pub struct AvgPool3d;
///     pub mod pool3d_ops {
///         pub fn max_pool3d() {}
///         pub fn avg_pool3d() {}
///     }
///     pub mod pool3d {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_pool3d_hold_probe::*;
///
/// struct __FandhePool3dHoldMarker;
///
/// trait __FandhePool3dHoldProbe {
///     fn max_pool3d(&self) -> __FandhePool3dHoldMarker;
///     fn avg_pool3d(&self) -> __FandhePool3dHoldMarker;
/// }
///
/// impl<'t> __FandhePool3dHoldProbe for fandhe_ai::Var<'t> {
///     fn max_pool3d(&self) -> __FandhePool3dHoldMarker {
///         __FandhePool3dHoldMarker
///     }
///     fn avg_pool3d(&self) -> __FandhePool3dHoldMarker {
///         __FandhePool3dHoldMarker
///     }
/// }
///
/// impl __FandhePool3dHoldProbe for fandhe_ai::Tape {
///     fn max_pool3d(&self) -> __FandhePool3dHoldMarker {
///         __FandhePool3dHoldMarker
///     }
///     fn avg_pool3d(&self) -> __FandhePool3dHoldMarker {
///         __FandhePool3dHoldMarker
///     }
/// }
///
/// impl __FandhePool3dHoldProbe for fandhe_ai::Tensor<f32> {
///     fn max_pool3d(&self) -> __FandhePool3dHoldMarker {
///         __FandhePool3dHoldMarker
///     }
///     fn avg_pool3d(&self) -> __FandhePool3dHoldMarker {
///         __FandhePool3dHoldMarker
///     }
/// }
///
/// trait __FandhePool3dHoldSequentialProbe {
///     fn add_max_pool3d(&self) -> __FandhePool3dHoldMarker;
///     fn add_avg_pool3d(&self) -> __FandhePool3dHoldMarker;
/// }
///
/// impl __FandhePool3dHoldSequentialProbe for fandhe_ai::compat::Sequential {
///     fn add_max_pool3d(&self) -> __FandhePool3dHoldMarker {
///         __FandhePool3dHoldMarker
///     }
///     fn add_avg_pool3d(&self) -> __FandhePool3dHoldMarker {
///         __FandhePool3dHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(_: Pool3dParams, _: MaxPool3d, _: AvgPool3d) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     pool3d_ops::max_pool3d();
///     pool3d_ops::avg_pool3d();
///     pool3d::__mark();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
///     seq: &fandhe_ai::compat::Sequential,
/// ) {
///     let _: __FandhePool3dHoldMarker = fandhe_ai::Var::max_pool3d(v);
///     let _: __FandhePool3dHoldMarker = fandhe_ai::Var::avg_pool3d(v);
///     let _: __FandhePool3dHoldMarker = fandhe_ai::Tape::max_pool3d(tape);
///     let _: __FandhePool3dHoldMarker = fandhe_ai::Tape::avg_pool3d(tape);
///     let _: __FandhePool3dHoldMarker = fandhe_ai::Tensor::<f32>::max_pool3d(tf);
///     let _: __FandhePool3dHoldMarker = fandhe_ai::Tensor::<f32>::avg_pool3d(tf);
///     let _: __FandhePool3dHoldMarker = fandhe_ai::compat::Sequential::add_max_pool3d(seq);
///     let _: __FandhePool3dHoldMarker = fandhe_ai::compat::Sequential::add_avg_pool3d(seq);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct Pool3dOpsHoldDoctestGuard;

/// 3D 転置畳み込みと MaxUnpool（`conv_transpose3d`・`max_unpool1d`・`max_unpool2d`・`max_unpool3d`。
/// イシュー #2644・親 #2625・ルート #2499 Phase 4）を facade 公開面から締め出す保留ガード
/// （`Pool3dOpsHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカルモジュール
/// `conv_transpose3d_ops`／`max_unpool_ops`／`conv_transpose3d`／`max_unpool`・型
/// `ConvTranspose3d`／`MaxUnpool1d`／`MaxUnpool2d`／`MaxUnpool3d`／`MaxUnpoolLayout`・メソッド
/// `conv_transpose3d`／`max_unpool1d`／`max_unpool2d`／`max_unpool3d`（`Var`／`Tape`／
/// `Tensor<f32>`）と `add_conv_transpose3d`／`add_max_unpool1d`／`add_max_unpool2d`／
/// `add_max_unpool3d`（`compat::Sequential`）を持つプローブ用トレイトを置き、修飾なしの関数
/// 呼び出しと修飾付きメソッド呼び出しの両方を行う。facade が同名のモジュール・型・関数を glob
/// 可能な位置へ公開するか、これらの型へ同名の inherent メソッドを公開すると、名前解決の曖昧性
/// または呼び出しシグネチャの不一致でエラーコードに依存せずコンパイルが失敗する（`Tensor` は
/// facade から再エクスポートされるため、`tensor-core` 側への同名メソッド追加も検出する）。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::conv_transpose3d_ops`・
/// `fandhe_ai_autodiff::max_unpool_ops`・`fandhe_ai_tensor_core::conv_transpose3d`・
/// `fandhe_ai_tensor_core::max_unpool`）。保留対象は facade 公開面（`Var::conv_transpose3d`／
/// `Var::max_unpool1d/2d/3d` の委譲メソッド）のみで、公開形は未承認（承認依頼は #2677・公開自体は
/// 承認後の #2678・#2679。推奨案は `docs/autodiff-conv-transpose3d-max-unpool-decision.md` §7。
/// 同記録は推奨案の記録であり承認記録ではない）。層化（`compat::Sequential::add_conv_transpose3d`／
/// `add_max_unpool1d/2d/3d`）も同じ保留に含める（#2679 の層結線がもう一方の漏出経路のため）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// conv_transpose3d_max_unpool_hold_doctest_globs_all_pub_modules`・
/// `conv_transpose3d_max_unpool_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_conv_transpose3d_max_unpool`・
/// `workspace_declares_conv_transpose3d_max_unpool_fn_names_only_in_allowed_locations`）との
/// 多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の対応する否定ガードも
/// 同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_conv_transpose3d_max_unpool_hold_probe {
///     pub struct ConvTranspose3d;
///     pub struct MaxUnpool1d;
///     pub struct MaxUnpool2d;
///     pub struct MaxUnpool3d;
///     pub struct MaxUnpoolLayout;
///     pub mod conv_transpose3d_ops {
///         pub fn conv_transpose3d() {}
///     }
///     pub mod max_unpool_ops {
///         pub fn max_unpool1d() {}
///         pub fn max_unpool2d() {}
///         pub fn max_unpool3d() {}
///     }
///     pub mod conv_transpose3d {
///         pub fn __mark() {}
///     }
///     pub mod max_unpool {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_conv_transpose3d_max_unpool_hold_probe::*;
///
/// struct __FandheConvTranspose3dMaxUnpoolHoldMarker;
///
/// trait __FandheConvTranspose3dMaxUnpoolHoldProbe {
///     fn conv_transpose3d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker;
///     fn max_unpool1d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker;
///     fn max_unpool2d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker;
///     fn max_unpool3d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker;
/// }
///
/// impl<'t> __FandheConvTranspose3dMaxUnpoolHoldProbe for fandhe_ai::Var<'t> {
///     fn conv_transpose3d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
///     fn max_unpool1d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
///     fn max_unpool2d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
///     fn max_unpool3d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
/// }
///
/// impl __FandheConvTranspose3dMaxUnpoolHoldProbe for fandhe_ai::Tape {
///     fn conv_transpose3d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
///     fn max_unpool1d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
///     fn max_unpool2d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
///     fn max_unpool3d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
/// }
///
/// impl __FandheConvTranspose3dMaxUnpoolHoldProbe for fandhe_ai::Tensor<f32> {
///     fn conv_transpose3d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
///     fn max_unpool1d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
///     fn max_unpool2d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
///     fn max_unpool3d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
/// }
///
/// trait __FandheConvTranspose3dMaxUnpoolHoldSequentialProbe {
///     fn add_conv_transpose3d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker;
///     fn add_max_unpool1d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker;
///     fn add_max_unpool2d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker;
///     fn add_max_unpool3d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker;
/// }
///
/// impl __FandheConvTranspose3dMaxUnpoolHoldSequentialProbe for fandhe_ai::compat::Sequential {
///     fn add_conv_transpose3d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
///     fn add_max_unpool1d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
///     fn add_max_unpool2d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
///     fn add_max_unpool3d(&self) -> __FandheConvTranspose3dMaxUnpoolHoldMarker {
///         __FandheConvTranspose3dMaxUnpoolHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(_: ConvTranspose3d, _: MaxUnpool1d, _: MaxUnpool2d, _: MaxUnpool3d, _: MaxUnpoolLayout) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して
///     // いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     conv_transpose3d_ops::conv_transpose3d();
///     max_unpool_ops::max_unpool1d();
///     max_unpool_ops::max_unpool2d();
///     max_unpool_ops::max_unpool3d();
///     conv_transpose3d::__mark();
///     max_unpool::__mark();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
///     seq: &fandhe_ai::compat::Sequential,
/// ) {
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::Var::conv_transpose3d(v);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::Var::max_unpool1d(v);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::Var::max_unpool2d(v);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::Var::max_unpool3d(v);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::Tape::conv_transpose3d(tape);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::Tape::max_unpool1d(tape);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::Tape::max_unpool2d(tape);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::Tape::max_unpool3d(tape);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::Tensor::<f32>::conv_transpose3d(tf);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::Tensor::<f32>::max_unpool1d(tf);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::Tensor::<f32>::max_unpool2d(tf);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::Tensor::<f32>::max_unpool3d(tf);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::compat::Sequential::add_conv_transpose3d(seq);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::compat::Sequential::add_max_unpool1d(seq);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::compat::Sequential::add_max_unpool2d(seq);
///     let _: __FandheConvTranspose3dMaxUnpoolHoldMarker = fandhe_ai::compat::Sequential::add_max_unpool3d(seq);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct ConvTranspose3dMaxUnpoolHoldDoctestGuard;

/// Fold と Unfold（`fold`・`unfold`。`F.fold`／`F.unfold`・`nn.Fold`／`nn.Unfold` 相当。イシュー #2645・
/// 親 #2625・ルート #2499 Phase 4）を facade 公開面から締め出す保留ガード
/// （`ConvTranspose3dMaxUnpoolHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカルモジュール `fold_ops`／`fold`・
/// 型 `Fold`／`Unfold`・メソッド `fold`／`unfold`（`Var`／`Tape`／`Tensor<f32>`）と
/// `add_fold`／`add_unfold`（`compat::Sequential`）を持つプローブ用トレイトを置き、修飾なしの関数呼び出しと
/// 修飾付きメソッド呼び出しの両方を行う。facade が同名のモジュール・型・関数を glob 可能な位置へ公開するか、
/// これらの型へ同名の inherent メソッドを公開すると、名前解決の曖昧性または呼び出しシグネチャの不一致で
/// エラーコードに依存せずコンパイルが失敗する（`Tensor` は facade から再エクスポートされるため、
/// `tensor-core` 側への同名メソッド追加も検出する）。検出範囲はこれらの名前・型に限り、マクロ生成や別名経由の
/// メソッドまでは保証しない。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::fold_ops`・`fandhe_ai_tensor_core::fold`）。
/// 保留対象は facade 公開面（`Var::fold`／`Var::unfold` の委譲メソッド）のみで、公開形は未承認
/// （承認依頼は #2677・公開自体は承認後の #2678・#2679。推奨案は `docs/autodiff-fold-unfold-decision.md` §7。
/// 同記録は推奨案の記録であり承認記録ではない）。層化（`compat::Sequential::add_fold`／`add_unfold`）も同じ
/// 保留に含める（#2679 の層結線がもう一方の漏出経路のため）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::fold_unfold_hold_doctest_globs_all_pub_modules`・
/// `fold_unfold_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_fold_unfold`・
/// `workspace_declares_fold_unfold_fn_names_only_in_allowed_locations`）との多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の対応する否定ガードも同時に
/// 正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_fold_unfold_hold_probe {
///     pub struct Fold;
///     pub struct Unfold;
///     pub mod fold_ops {
///         pub fn fold() {}
///         pub fn unfold() {}
///     }
///     pub mod fold {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_fold_unfold_hold_probe::*;
///
/// struct __FandheFoldUnfoldHoldMarker;
///
/// trait __FandheFoldUnfoldHoldProbe {
///     fn fold(&self) -> __FandheFoldUnfoldHoldMarker;
///     fn unfold(&self) -> __FandheFoldUnfoldHoldMarker;
/// }
///
/// impl<'t> __FandheFoldUnfoldHoldProbe for fandhe_ai::Var<'t> {
///     fn fold(&self) -> __FandheFoldUnfoldHoldMarker {
///         __FandheFoldUnfoldHoldMarker
///     }
///     fn unfold(&self) -> __FandheFoldUnfoldHoldMarker {
///         __FandheFoldUnfoldHoldMarker
///     }
/// }
///
/// impl __FandheFoldUnfoldHoldProbe for fandhe_ai::Tape {
///     fn fold(&self) -> __FandheFoldUnfoldHoldMarker {
///         __FandheFoldUnfoldHoldMarker
///     }
///     fn unfold(&self) -> __FandheFoldUnfoldHoldMarker {
///         __FandheFoldUnfoldHoldMarker
///     }
/// }
///
/// impl __FandheFoldUnfoldHoldProbe for fandhe_ai::Tensor<f32> {
///     fn fold(&self) -> __FandheFoldUnfoldHoldMarker {
///         __FandheFoldUnfoldHoldMarker
///     }
///     fn unfold(&self) -> __FandheFoldUnfoldHoldMarker {
///         __FandheFoldUnfoldHoldMarker
///     }
/// }
///
/// trait __FandheFoldUnfoldHoldSequentialProbe {
///     fn add_fold(&self) -> __FandheFoldUnfoldHoldMarker;
///     fn add_unfold(&self) -> __FandheFoldUnfoldHoldMarker;
/// }
///
/// impl __FandheFoldUnfoldHoldSequentialProbe for fandhe_ai::compat::Sequential {
///     fn add_fold(&self) -> __FandheFoldUnfoldHoldMarker {
///         __FandheFoldUnfoldHoldMarker
///     }
///     fn add_unfold(&self) -> __FandheFoldUnfoldHoldMarker {
///         __FandheFoldUnfoldHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(_: Fold, _: Unfold) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開していれば、名前解決自体が曖昧になり
///     // E0659 でコンパイル失敗する）。
///     fold_ops::fold();
///     fold_ops::unfold();
///     fold::__mark();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
///     seq: &fandhe_ai::compat::Sequential,
/// ) {
///     let _: __FandheFoldUnfoldHoldMarker = fandhe_ai::Var::fold(v);
///     let _: __FandheFoldUnfoldHoldMarker = fandhe_ai::Var::unfold(v);
///     let _: __FandheFoldUnfoldHoldMarker = fandhe_ai::Tape::fold(tape);
///     let _: __FandheFoldUnfoldHoldMarker = fandhe_ai::Tape::unfold(tape);
///     let _: __FandheFoldUnfoldHoldMarker = fandhe_ai::Tensor::<f32>::fold(tf);
///     let _: __FandheFoldUnfoldHoldMarker = fandhe_ai::Tensor::<f32>::unfold(tf);
///     let _: __FandheFoldUnfoldHoldMarker = fandhe_ai::compat::Sequential::add_fold(seq);
///     let _: __FandheFoldUnfoldHoldMarker = fandhe_ai::compat::Sequential::add_unfold(seq);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct FoldUnfoldHoldDoctestGuard;

/// LocalResponseNorm と重み再パラメータ化（`local_response_norm`・`weight_norm`・`norm_except_dim`・
/// `spectral_norm`。`F.local_response_norm`・`torch._weight_norm`・`parametrizations.spectral_norm` 相当。
/// イシュー #2646・親 #2625・ルート #2499 Phase 4）を facade 公開面から締め出す保留ガード
/// （`FoldUnfoldHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカルモジュール `lrn_ops`／
/// `weight_reparam_ops`／`lrn`／`weight_reparam`・型 `LocalResponseNorm`／`WeightNorm`／`SpectralNorm`／
/// `SpectralNormState`・メソッド `local_response_norm`／`weight_norm`／`spectral_norm`／`norm_except_dim`
/// （`Var`／`Tape`／`Tensor<f32>`）と `add_local_response_norm`／`add_weight_norm`／`add_spectral_norm`
/// （`compat::Sequential`）を持つプローブ用トレイトを置き、修飾なしの関数呼び出しと修飾付きメソッド呼び出しの
/// 両方を行う。facade が同名のモジュール・型・関数を glob 可能な位置へ公開するか、これらの型へ同名の inherent
/// メソッドを公開すると、名前解決の曖昧性または呼び出しシグネチャの不一致でエラーコードに依存せずコンパイルが
/// 失敗する（`Tensor` は facade から再エクスポートされるため、`tensor-core` 側への同名メソッド追加も検出する）。
/// 検出範囲はこれらの名前・型に限り、マクロ生成や別名経由のメソッドまでは保証しない。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::lrn_ops`・`fandhe_ai_autodiff::weight_reparam_ops`・
/// `fandhe_ai_tensor_core::lrn`・`fandhe_ai_tensor_core::weight_reparam`）。保留対象は facade 公開面
/// （`Var::local_response_norm`／`Var::weight_norm`／`Var::spectral_norm` の委譲メソッドと
/// `SpectralNormState` の公開位置）のみで、公開形は未承認（承認依頼は #2677・公開自体は承認後の #2678・
/// #2679。推奨案は `docs/autodiff-lrn-weight-reparam-decision.md` §7。同記録は推奨案の記録であり承認記録では
/// ない）。層化（`compat::Sequential::add_*`・`Linear`／`Conv` への parametrization 結線）も同じ保留に
/// 含める（#2679 の層結線がもう一方の漏出経路のため）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::lrn_weight_reparam_hold_doctest_globs_all_pub_modules`・
/// `lrn_weight_reparam_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_lrn_weight_reparam`・
/// `workspace_declares_lrn_weight_reparam_fn_names_only_in_allowed_locations`）との多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の対応する否定ガードも同時に
/// 正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_lrn_weight_reparam_hold_probe {
///     pub struct LocalResponseNorm;
///     pub struct WeightNorm;
///     pub struct SpectralNorm;
///     pub struct SpectralNormState;
///     pub mod lrn_ops {
///         pub fn local_response_norm() {}
///     }
///     pub mod weight_reparam_ops {
///         pub fn weight_norm() {}
///         pub fn norm_except_dim() {}
///         pub fn spectral_norm() {}
///     }
///     pub mod lrn {
///         pub fn __mark() {}
///     }
///     pub mod weight_reparam {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_lrn_weight_reparam_hold_probe::*;
///
/// struct __FandheLrnWeightReparamHoldMarker;
///
/// trait __FandheLrnWeightReparamHoldProbe {
///     fn local_response_norm(&self) -> __FandheLrnWeightReparamHoldMarker;
///     fn weight_norm(&self) -> __FandheLrnWeightReparamHoldMarker;
///     fn spectral_norm(&self) -> __FandheLrnWeightReparamHoldMarker;
///     fn norm_except_dim(&self) -> __FandheLrnWeightReparamHoldMarker;
/// }
///
/// impl<'t> __FandheLrnWeightReparamHoldProbe for fandhe_ai::Var<'t> {
///     fn local_response_norm(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
///     fn weight_norm(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
///     fn spectral_norm(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
///     fn norm_except_dim(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
/// }
///
/// impl __FandheLrnWeightReparamHoldProbe for fandhe_ai::Tape {
///     fn local_response_norm(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
///     fn weight_norm(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
///     fn spectral_norm(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
///     fn norm_except_dim(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
/// }
///
/// impl __FandheLrnWeightReparamHoldProbe for fandhe_ai::Tensor<f32> {
///     fn local_response_norm(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
///     fn weight_norm(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
///     fn spectral_norm(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
///     fn norm_except_dim(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
/// }
///
/// trait __FandheLrnWeightReparamHoldSequentialProbe {
///     fn add_local_response_norm(&self) -> __FandheLrnWeightReparamHoldMarker;
///     fn add_weight_norm(&self) -> __FandheLrnWeightReparamHoldMarker;
///     fn add_spectral_norm(&self) -> __FandheLrnWeightReparamHoldMarker;
/// }
///
/// impl __FandheLrnWeightReparamHoldSequentialProbe for fandhe_ai::compat::Sequential {
///     fn add_local_response_norm(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
///     fn add_weight_norm(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
///     fn add_spectral_norm(&self) -> __FandheLrnWeightReparamHoldMarker {
///         __FandheLrnWeightReparamHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(_: LocalResponseNorm, _: WeightNorm, _: SpectralNorm, _: SpectralNormState) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開していれば、名前解決自体が曖昧になり
///     // E0659 でコンパイル失敗する）。
///     lrn_ops::local_response_norm();
///     weight_reparam_ops::weight_norm();
///     weight_reparam_ops::norm_except_dim();
///     weight_reparam_ops::spectral_norm();
///     lrn::__mark();
///     weight_reparam::__mark();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
///     seq: &fandhe_ai::compat::Sequential,
/// ) {
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::Var::local_response_norm(v);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::Var::weight_norm(v);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::Var::spectral_norm(v);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::Var::norm_except_dim(v);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::Tape::local_response_norm(tape);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::Tape::weight_norm(tape);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::Tape::spectral_norm(tape);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::Tape::norm_except_dim(tape);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::Tensor::<f32>::local_response_norm(tf);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::Tensor::<f32>::weight_norm(tf);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::Tensor::<f32>::spectral_norm(tf);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::Tensor::<f32>::norm_except_dim(tf);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::compat::Sequential::add_local_response_norm(seq);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::compat::Sequential::add_weight_norm(seq);
///     let _: __FandheLrnWeightReparamHoldMarker = fandhe_ai::compat::Sequential::add_spectral_norm(seq);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct LrnWeightReparamHoldDoctestGuard;

/// 可変長系列の pack／unpack（`pack_padded_sequence`・`pad_packed_sequence`・`PackedSequence`）と RNN 系の
/// packed 実行（`rnn_forward_packed`・`gru_forward_packed`・`lstm_forward_packed`・`stacked_*_forward_packed`。
/// `torch.nn.utils.rnn` 相当。イシュー #2647・親 #2625・ルート #2499 Phase 4）を facade 公開面から締め出す
/// 保留ガード（`LrnWeightReparamHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカルモジュール `packed_sequence`・型
/// `PackedSequence`／`PackedRnnSeqOutput`／`PackedLstmSeqOutput`／`StackedPackedRnnSeqOutput`／
/// `StackedPackedLstmSeqOutput`・トップレベル自由関数 8 名と、プローブ用トレイトのメソッド
/// （`Var`／`Tape`／`Tensor<f32>` の `pack_padded_sequence`／`pad_packed_sequence`、`Tape` の `*_forward_packed` 6 名、
/// `nn::rnn::{Rnn, Lstm, Gru, StackedRnn, StackedLstm, StackedGru}` の `forward_packed`）を置き、修飾なしの
/// 関数呼び出しと修飾付きメソッド呼び出しの両方を行う。facade が同名のモジュール・型・関数を glob 可能な位置へ
/// 公開するか、これらの型へ同名の inherent メソッドを公開すると、名前解決の曖昧性または呼び出しシグネチャの
/// 不一致でエラーコードに依存せずコンパイルが失敗する。推奨する公開形が自由関数の再エクスポートのため、関数名
/// そのものの glob 衝突も検出対象にしている。検出範囲は列挙したこれらの名前・型に限り、マクロ生成や別名経由の
/// メソッドまでは保証しない。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::nn::packed_sequence`）。保留対象は facade 公開面のみで、
/// 公開形は未承認（承認依頼は #2677・公開自体は承認後の #2678・#2679。推奨案は
/// `docs/autodiff-packed-sequence-decision.md` §7。同記録は推奨案の記録であり承認記録ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::packed_sequence_hold_doctest_globs_all_pub_modules`・
/// `packed_sequence_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_packed_sequence`・
/// `workspace_declares_packed_sequence_fn_names_only_in_allowed_locations`）との多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の対応する否定ガードも同時に
/// 正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_packed_sequence_hold_probe {
///     pub struct PackedSequence;
///     pub struct PackedRnnSeqOutput;
///     pub struct PackedLstmSeqOutput;
///     pub struct StackedPackedRnnSeqOutput;
///     pub struct StackedPackedLstmSeqOutput;
///     pub fn pack_padded_sequence() {}
///     pub fn pad_packed_sequence() {}
///     pub fn rnn_forward_packed() {}
///     pub fn gru_forward_packed() {}
///     pub fn lstm_forward_packed() {}
///     pub fn stacked_rnn_forward_packed() {}
///     pub fn stacked_gru_forward_packed() {}
///     pub fn stacked_lstm_forward_packed() {}
///     pub mod packed_sequence {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_packed_sequence_hold_probe::*;
///
/// struct __FandhePackedSequenceHoldMarker;
///
/// trait __FandhePackedSequenceHoldProbe {
///     fn pack_padded_sequence(&self) -> __FandhePackedSequenceHoldMarker;
///     fn pad_packed_sequence(&self) -> __FandhePackedSequenceHoldMarker;
/// }
///
/// trait __FandhePackedSequenceHoldTapeProbe {
///     fn rnn_forward_packed(&self) -> __FandhePackedSequenceHoldMarker;
///     fn gru_forward_packed(&self) -> __FandhePackedSequenceHoldMarker;
///     fn lstm_forward_packed(&self) -> __FandhePackedSequenceHoldMarker;
///     fn stacked_rnn_forward_packed(&self) -> __FandhePackedSequenceHoldMarker;
///     fn stacked_gru_forward_packed(&self) -> __FandhePackedSequenceHoldMarker;
///     fn stacked_lstm_forward_packed(&self) -> __FandhePackedSequenceHoldMarker;
/// }
///
/// trait __FandhePackedSequenceHoldRnnProbe {
///     fn forward_packed(&self) -> __FandhePackedSequenceHoldMarker;
/// }
///
/// impl<'t> __FandhePackedSequenceHoldProbe for fandhe_ai::Var<'t> {
///     fn pack_padded_sequence(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
///     fn pad_packed_sequence(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
/// }
///
/// impl __FandhePackedSequenceHoldProbe for fandhe_ai::Tape {
///     fn pack_padded_sequence(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
///     fn pad_packed_sequence(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
/// }
///
/// impl __FandhePackedSequenceHoldProbe for fandhe_ai::Tensor<f32> {
///     fn pack_padded_sequence(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
///     fn pad_packed_sequence(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
/// }
///
/// impl __FandhePackedSequenceHoldTapeProbe for fandhe_ai::Tape {
///     fn rnn_forward_packed(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
///     fn gru_forward_packed(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
///     fn lstm_forward_packed(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
///     fn stacked_rnn_forward_packed(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
///     fn stacked_gru_forward_packed(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
///     fn stacked_lstm_forward_packed(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
/// }
///
/// impl __FandhePackedSequenceHoldRnnProbe for fandhe_ai::nn::rnn::Rnn {
///     fn forward_packed(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
/// }
///
/// impl __FandhePackedSequenceHoldRnnProbe for fandhe_ai::nn::rnn::Lstm {
///     fn forward_packed(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
/// }
///
/// impl __FandhePackedSequenceHoldRnnProbe for fandhe_ai::nn::rnn::Gru {
///     fn forward_packed(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
/// }
///
/// impl __FandhePackedSequenceHoldRnnProbe for fandhe_ai::nn::rnn::StackedRnn {
///     fn forward_packed(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
/// }
///
/// impl __FandhePackedSequenceHoldRnnProbe for fandhe_ai::nn::rnn::StackedLstm {
///     fn forward_packed(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
/// }
///
/// impl __FandhePackedSequenceHoldRnnProbe for fandhe_ai::nn::rnn::StackedGru {
///     fn forward_packed(&self) -> __FandhePackedSequenceHoldMarker {
///         __FandhePackedSequenceHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(
///     _0: PackedSequence,
///     _1: PackedRnnSeqOutput,
///     _2: PackedLstmSeqOutput,
///     _3: StackedPackedRnnSeqOutput,
///     _4: StackedPackedLstmSeqOutput,
/// ) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開していれば、名前解決自体が曖昧になり
///     // E0659 でコンパイル失敗する）。
///     pack_padded_sequence();
///     pad_packed_sequence();
///     rnn_forward_packed();
///     gru_forward_packed();
///     lstm_forward_packed();
///     stacked_rnn_forward_packed();
///     stacked_gru_forward_packed();
///     stacked_lstm_forward_packed();
///     packed_sequence::__mark();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
///     rnn: &fandhe_ai::nn::rnn::Rnn,
///     lstm: &fandhe_ai::nn::rnn::Lstm,
///     gru: &fandhe_ai::nn::rnn::Gru,
///     stackedrnn: &fandhe_ai::nn::rnn::StackedRnn,
///     stackedlstm: &fandhe_ai::nn::rnn::StackedLstm,
///     stackedgru: &fandhe_ai::nn::rnn::StackedGru,
/// ) {
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::Var::pack_padded_sequence(v);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::Tape::pack_padded_sequence(tape);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::Tensor::<f32>::pack_padded_sequence(tf);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::Var::pad_packed_sequence(v);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::Tape::pad_packed_sequence(tape);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::Tensor::<f32>::pad_packed_sequence(tf);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::Tape::rnn_forward_packed(tape);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::Tape::gru_forward_packed(tape);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::Tape::lstm_forward_packed(tape);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::Tape::stacked_rnn_forward_packed(tape);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::Tape::stacked_gru_forward_packed(tape);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::Tape::stacked_lstm_forward_packed(tape);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::nn::rnn::Rnn::forward_packed(rnn);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::nn::rnn::Lstm::forward_packed(lstm);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::nn::rnn::Gru::forward_packed(gru);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::nn::rnn::StackedRnn::forward_packed(stackedrnn);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::nn::rnn::StackedLstm::forward_packed(stackedlstm);
///     let _: __FandhePackedSequenceHoldMarker = fandhe_ai::nn::rnn::StackedGru::forward_packed(stackedgru);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct PackedSequenceHoldDoctestGuard;

/// SELU・CELU・Softsign・Hardsigmoid・LogSigmoid（`selu`・`celu`・`softsign`・
/// `hardsigmoid`・`log_sigmoid`。イシュー #2649・親 #2648）を facade 公開面から
/// 締め出す保留ガード（`PackedSequenceHoldDoctestGuard` と同型の正のプローブ 1
/// ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカルの型 5 個・
/// 関数 5 個・モジュール `activation_scalar_ops` と、`Var`／`Tape`／`Tensor<f32>`
/// 上の 5 メソッド・`compat::Sequential` 上の `add_*` 5 メソッドを持つプローブ用
/// トレイトを置き、修飾なしの関数呼び出しと修飾付きメソッド呼び出し（UFCS）の両方を
/// 行う。facade が同名のモジュール・型・関数を glob 可能な位置へ公開するか、
/// 同名の inherent メソッドを公開すると、名前解決の曖昧性または呼び出し
/// シグネチャの不一致でエラーコードに依存せずコンパイルが失敗する。
///
/// 検出範囲は本プローブが列挙した名前・型に限る（マクロ生成や別名経由の公開までは
/// 保証しない）。実装は内部クレートに閉じている
/// （`fandhe_ai_autodiff::activation_scalar_ops`・`nn::activation` の層 5 型・
/// `fandhe_ai_tensor_core::ScalarUnaryOp` の追加 variant）。保留対象は facade 公開面
/// （`Var` の委譲メソッド・`compat::Sequential::add_*`）のみで、公開形は未承認
/// （承認依頼は #2677・公開自体は承認後の #2678・#2679。推奨案は
/// `docs/autodiff-activation-scalar-ops-decision.md` §7。同記録は推奨案の記録であり
/// 承認記録ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::
/// activation_scalar_ops_hold_doctest_globs_all_pub_modules`・
/// `activation_scalar_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_activation_scalar_ops`・
/// `workspace_declares_activation_scalar_ops_fn_names_only_in_allowed_locations`）との
/// 多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の
/// 対応する否定ガードも同時に正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_activation_scalar_ops_hold_probe {
///     pub struct Selu;
///     pub struct Celu;
///     pub struct Softsign;
///     pub struct Hardsigmoid;
///     pub struct LogSigmoid;
///     pub fn selu() {}
///     pub fn celu() {}
///     pub fn softsign() {}
///     pub fn hardsigmoid() {}
///     pub fn log_sigmoid() {}
///     pub mod activation_scalar_ops {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_activation_scalar_ops_hold_probe::*;
///
/// struct __FandheActivationScalarOpsHoldMarker;
///
/// trait __FandheActivationScalarOpsHoldProbe {
///     fn selu(&self) -> __FandheActivationScalarOpsHoldMarker;
///     fn celu(&self) -> __FandheActivationScalarOpsHoldMarker;
///     fn softsign(&self) -> __FandheActivationScalarOpsHoldMarker;
///     fn hardsigmoid(&self) -> __FandheActivationScalarOpsHoldMarker;
///     fn log_sigmoid(&self) -> __FandheActivationScalarOpsHoldMarker;
/// }
///
/// trait __FandheActivationScalarOpsHoldAddProbe {
///     fn add_selu(&self) -> __FandheActivationScalarOpsHoldMarker;
///     fn add_celu(&self) -> __FandheActivationScalarOpsHoldMarker;
///     fn add_softsign(&self) -> __FandheActivationScalarOpsHoldMarker;
///     fn add_hardsigmoid(&self) -> __FandheActivationScalarOpsHoldMarker;
///     fn add_log_sigmoid(&self) -> __FandheActivationScalarOpsHoldMarker;
/// }
///
/// impl<'t> __FandheActivationScalarOpsHoldProbe for fandhe_ai::Var<'t> {
///     fn selu(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn celu(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn softsign(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn hardsigmoid(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn log_sigmoid(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
/// }
///
/// impl __FandheActivationScalarOpsHoldProbe for fandhe_ai::Tape {
///     fn selu(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn celu(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn softsign(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn hardsigmoid(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn log_sigmoid(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
/// }
///
/// impl __FandheActivationScalarOpsHoldProbe for fandhe_ai::Tensor<f32> {
///     fn selu(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn celu(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn softsign(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn hardsigmoid(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn log_sigmoid(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
/// }
///
/// impl __FandheActivationScalarOpsHoldAddProbe for fandhe_ai::compat::Sequential {
///     fn add_selu(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn add_celu(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn add_softsign(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn add_hardsigmoid(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
///     fn add_log_sigmoid(&self) -> __FandheActivationScalarOpsHoldMarker {
///         __FandheActivationScalarOpsHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(
///     _0: Selu,
///     _1: Celu,
///     _2: Softsign,
///     _3: Hardsigmoid,
///     _4: LogSigmoid,
/// ) {
///     // 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開していれば、名前解決自体が曖昧になり
///     // E0659 でコンパイル失敗する）。
///     selu();
///     celu();
///     softsign();
///     hardsigmoid();
///     log_sigmoid();
///     activation_scalar_ops::__mark();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
///     seq: &fandhe_ai::compat::Sequential,
/// ) {
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Var::selu(v);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Tape::selu(tape);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Tensor::<f32>::selu(tf);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Var::celu(v);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Tape::celu(tape);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Tensor::<f32>::celu(tf);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Var::softsign(v);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Tape::softsign(tape);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Tensor::<f32>::softsign(tf);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Var::hardsigmoid(v);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Tape::hardsigmoid(tape);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Tensor::<f32>::hardsigmoid(tf);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Var::log_sigmoid(v);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Tape::log_sigmoid(tape);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::Tensor::<f32>::log_sigmoid(tf);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::compat::Sequential::add_selu(seq);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::compat::Sequential::add_celu(seq);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::compat::Sequential::add_softsign(seq);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::compat::Sequential::add_hardsigmoid(seq);
///     let _: __FandheActivationScalarOpsHoldMarker = fandhe_ai::compat::Sequential::add_log_sigmoid(seq);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct ActivationScalarOpsHoldDoctestGuard;

/// 活性化 4 種（`softmin`・`tanhshrink`・`threshold`・`rrelu`／`rrelu_with_noise` と層 `Softmin`・`Tanhshrink`・
/// `Threshold`・`RRelu`。`F.softmin`／`F.tanhshrink`／`F.threshold`／`F.rrelu` 相当。イシュー #2650・親 #2648・
/// Phase 親 #2625）を facade 公開面から締め出す保留ガード（`PackedSequenceHoldDoctestGuard` と同型の
/// 正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカルモジュール `softmin_threshold_ops`／
/// `softmin_threshold`・型 `Softmin`／`Tanhshrink`／`RRelu`／`Threshold` と、プローブ用トレイトのメソッド
/// （`Var`／`Tape`／`Tensor<f32>` の `softmin`／`tanhshrink`／`threshold`／`rrelu`／`rrelu_with_noise`、
/// `compat::Sequential` の `add_softmin`／`add_tanhshrink`／`add_threshold`／`add_rrelu`）を置き、モジュール経由の
/// 関数呼び出しと修飾付きメソッド呼び出しの両方を行う。facade が同名のモジュール・型を glob 可能な位置へ
/// 公開するか、これらの型へ同名の inherent メソッドを公開すると、名前解決の曖昧性または呼び出しシグネチャの
/// 不一致でエラーコードに依存せずコンパイルが失敗する。検出範囲は列挙したこれらの名前・型に限り、マクロ生成や
/// 別名経由のメソッドまでは保証しない。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::softmin_threshold_ops`・
/// `fandhe_ai_autodiff::nn::softmin_threshold`）。保留対象は facade 公開面のみで、公開形は未承認（承認依頼は
/// #2677・公開自体は承認後の #2678・#2679。推奨案は `docs/autodiff-softmin-threshold-ops-decision.md` §7。
/// 同記録は推奨案の記録であり承認記録ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::softmin_threshold_ops_hold_doctest_globs_all_pub_modules`・
/// `softmin_threshold_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_softmin_threshold_ops`・
/// `workspace_declares_softmin_threshold_ops_fn_names_only_in_allowed_locations`）との多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の対応する否定ガードも同時に
/// 正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_softmin_threshold_ops_hold_probe {
///     pub struct Softmin;
///     pub struct Tanhshrink;
///     pub struct RRelu;
///     pub struct Threshold;
///     pub mod softmin_threshold_ops {
///         pub fn softmin() {}
///         pub fn tanhshrink() {}
///         pub fn threshold() {}
///         pub fn rrelu() {}
///         pub fn rrelu_with_noise() {}
///     }
///     pub mod softmin_threshold {
///         pub fn __mark() {}
///     }
/// }
/// use __fandhe_softmin_threshold_ops_hold_probe::*;
///
/// struct __FandheSoftminThresholdOpsHoldMarker;
///
/// trait __FandheSoftminThresholdOpsHoldProbe {
///     fn softmin(&self) -> __FandheSoftminThresholdOpsHoldMarker;
///     fn tanhshrink(&self) -> __FandheSoftminThresholdOpsHoldMarker;
///     fn threshold(&self) -> __FandheSoftminThresholdOpsHoldMarker;
///     fn rrelu(&self) -> __FandheSoftminThresholdOpsHoldMarker;
///     fn rrelu_with_noise(&self) -> __FandheSoftminThresholdOpsHoldMarker;
/// }
///
/// trait __FandheSoftminThresholdOpsHoldSequentialProbe {
///     fn add_softmin(&self) -> __FandheSoftminThresholdOpsHoldMarker;
///     fn add_tanhshrink(&self) -> __FandheSoftminThresholdOpsHoldMarker;
///     fn add_threshold(&self) -> __FandheSoftminThresholdOpsHoldMarker;
///     fn add_rrelu(&self) -> __FandheSoftminThresholdOpsHoldMarker;
/// }
///
/// impl<'t> __FandheSoftminThresholdOpsHoldProbe for fandhe_ai::Var<'t> {
///     fn softmin(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn tanhshrink(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn threshold(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn rrelu(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn rrelu_with_noise(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
/// }
///
/// impl __FandheSoftminThresholdOpsHoldProbe for fandhe_ai::Tape {
///     fn softmin(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn tanhshrink(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn threshold(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn rrelu(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn rrelu_with_noise(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
/// }
///
/// impl __FandheSoftminThresholdOpsHoldProbe for fandhe_ai::Tensor<f32> {
///     fn softmin(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn tanhshrink(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn threshold(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn rrelu(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn rrelu_with_noise(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
/// }
///
/// impl __FandheSoftminThresholdOpsHoldSequentialProbe for fandhe_ai::compat::Sequential {
///     fn add_softmin(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn add_tanhshrink(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn add_threshold(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
///     fn add_rrelu(&self) -> __FandheSoftminThresholdOpsHoldMarker {
///         __FandheSoftminThresholdOpsHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(
///     _0: Softmin,
///     _1: Tanhshrink,
///     _2: RRelu,
///     _3: Threshold,
/// ) {
///     // モジュール経由の呼び出し（`use fandhe_ai::*;` が同名モジュールを glob 公開していれば、
///     // 名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     softmin_threshold_ops::softmin();
///     softmin_threshold_ops::tanhshrink();
///     softmin_threshold_ops::threshold();
///     softmin_threshold_ops::rrelu();
///     softmin_threshold_ops::rrelu_with_noise();
///     softmin_threshold::__mark();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
///     seq: &fandhe_ai::compat::Sequential,
/// ) {
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Var::softmin(v);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Var::tanhshrink(v);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Var::threshold(v);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Var::rrelu(v);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Var::rrelu_with_noise(v);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Tape::softmin(tape);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Tape::tanhshrink(tape);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Tape::threshold(tape);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Tape::rrelu(tape);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Tape::rrelu_with_noise(tape);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Tensor::<f32>::softmin(tf);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Tensor::<f32>::tanhshrink(tf);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Tensor::<f32>::threshold(tf);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Tensor::<f32>::rrelu(tf);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::Tensor::<f32>::rrelu_with_noise(tf);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::compat::Sequential::add_softmin(seq);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::compat::Sequential::add_tanhshrink(seq);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::compat::Sequential::add_threshold(seq);
///     let _: __FandheSoftminThresholdOpsHoldMarker = fandhe_ai::compat::Sequential::add_rrelu(seq);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct SoftminThresholdOpsHoldDoctestGuard;

/// pos_weight 付き BCEWithLogits・HingeEmbedding・SoftMargin・GaussianNLL（イシュー #2652・親 #2651）を
/// facade 公開面から締め出す保留ガード（`SoftminThresholdOpsHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカルモジュール `elementwise_loss_ops`・
/// 型 `BceWithLogitsOptions`／`GaussianNllOptions` と、プローブ用トレイトのメソッド（`Var`／`Tape`／`Tensor<f32>` の
/// `bce_with_logits_loss_with`／`hinge_embedding_loss`／`soft_margin_loss`／`gaussian_nll_loss`）を置き、
/// モジュール経由の関数呼び出しと修飾付きメソッド呼び出しの両方を行う。facade が同名のモジュール・型を glob 可能な
/// 位置へ公開するか、これらの型へ同名の inherent メソッドを公開すると、名前解決の曖昧性または呼び出しシグネチャの
/// 不一致でエラーコードに依存せずコンパイルが失敗する。検出範囲は列挙したこれらの名前・型に限り、マクロ生成や
/// 別名経由のメソッドまでは保証しない。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::elementwise_loss_ops`）。保留対象は facade 公開面のみで、
/// 公開形は未承認（承認依頼は #2677・公開自体は承認後の #2678。推奨案は
/// `docs/autodiff-elementwise-loss-ops-decision.md` §7。同記録は推奨案の記録であり承認記録ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::elementwise_loss_ops_hold_doctest_globs_all_pub_modules`・
/// `elementwise_loss_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_elementwise_loss_ops`・
/// `workspace_declares_elementwise_loss_ops_fn_names_only_in_allowed_locations`）との多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の対応する否定ガードも同時に
/// 正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_elementwise_loss_ops_hold_probe {
///     pub struct BceWithLogitsOptions;
///     pub struct GaussianNllOptions;
///     pub mod elementwise_loss_ops {
///         pub fn bce_with_logits_loss_with() {}
///         pub fn hinge_embedding_loss() {}
///         pub fn soft_margin_loss() {}
///         pub fn gaussian_nll_loss() {}
///     }
/// }
/// use __fandhe_elementwise_loss_ops_hold_probe::*;
///
/// struct __FandheElementwiseLossOpsHoldMarker;
///
/// trait __FandheElementwiseLossOpsHoldProbe {
///     fn bce_with_logits_loss_with(&self) -> __FandheElementwiseLossOpsHoldMarker;
///     fn hinge_embedding_loss(&self) -> __FandheElementwiseLossOpsHoldMarker;
///     fn soft_margin_loss(&self) -> __FandheElementwiseLossOpsHoldMarker;
///     fn gaussian_nll_loss(&self) -> __FandheElementwiseLossOpsHoldMarker;
/// }
///
/// impl<'t> __FandheElementwiseLossOpsHoldProbe for fandhe_ai::Var<'t> {
///     fn bce_with_logits_loss_with(&self) -> __FandheElementwiseLossOpsHoldMarker {
///         __FandheElementwiseLossOpsHoldMarker
///     }
///     fn hinge_embedding_loss(&self) -> __FandheElementwiseLossOpsHoldMarker {
///         __FandheElementwiseLossOpsHoldMarker
///     }
///     fn soft_margin_loss(&self) -> __FandheElementwiseLossOpsHoldMarker {
///         __FandheElementwiseLossOpsHoldMarker
///     }
///     fn gaussian_nll_loss(&self) -> __FandheElementwiseLossOpsHoldMarker {
///         __FandheElementwiseLossOpsHoldMarker
///     }
/// }
///
/// impl __FandheElementwiseLossOpsHoldProbe for fandhe_ai::Tape {
///     fn bce_with_logits_loss_with(&self) -> __FandheElementwiseLossOpsHoldMarker {
///         __FandheElementwiseLossOpsHoldMarker
///     }
///     fn hinge_embedding_loss(&self) -> __FandheElementwiseLossOpsHoldMarker {
///         __FandheElementwiseLossOpsHoldMarker
///     }
///     fn soft_margin_loss(&self) -> __FandheElementwiseLossOpsHoldMarker {
///         __FandheElementwiseLossOpsHoldMarker
///     }
///     fn gaussian_nll_loss(&self) -> __FandheElementwiseLossOpsHoldMarker {
///         __FandheElementwiseLossOpsHoldMarker
///     }
/// }
///
/// impl __FandheElementwiseLossOpsHoldProbe for fandhe_ai::Tensor<f32> {
///     fn bce_with_logits_loss_with(&self) -> __FandheElementwiseLossOpsHoldMarker {
///         __FandheElementwiseLossOpsHoldMarker
///     }
///     fn hinge_embedding_loss(&self) -> __FandheElementwiseLossOpsHoldMarker {
///         __FandheElementwiseLossOpsHoldMarker
///     }
///     fn soft_margin_loss(&self) -> __FandheElementwiseLossOpsHoldMarker {
///         __FandheElementwiseLossOpsHoldMarker
///     }
///     fn gaussian_nll_loss(&self) -> __FandheElementwiseLossOpsHoldMarker {
///         __FandheElementwiseLossOpsHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(
///     _0: BceWithLogitsOptions,
///     _1: GaussianNllOptions,
/// ) {
///     // モジュール経由の呼び出し（`use fandhe_ai::*;` が同名モジュールを glob 公開していれば、
///     // 名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     elementwise_loss_ops::bce_with_logits_loss_with();
///     elementwise_loss_ops::hinge_embedding_loss();
///     elementwise_loss_ops::soft_margin_loss();
///     elementwise_loss_ops::gaussian_nll_loss();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
/// ) {
///     let _: __FandheElementwiseLossOpsHoldMarker = fandhe_ai::Var::bce_with_logits_loss_with(v);
///     let _: __FandheElementwiseLossOpsHoldMarker = fandhe_ai::Var::hinge_embedding_loss(v);
///     let _: __FandheElementwiseLossOpsHoldMarker = fandhe_ai::Var::soft_margin_loss(v);
///     let _: __FandheElementwiseLossOpsHoldMarker = fandhe_ai::Var::gaussian_nll_loss(v);
///     let _: __FandheElementwiseLossOpsHoldMarker = fandhe_ai::Tape::bce_with_logits_loss_with(tape);
///     let _: __FandheElementwiseLossOpsHoldMarker = fandhe_ai::Tape::hinge_embedding_loss(tape);
///     let _: __FandheElementwiseLossOpsHoldMarker = fandhe_ai::Tape::soft_margin_loss(tape);
///     let _: __FandheElementwiseLossOpsHoldMarker = fandhe_ai::Tape::gaussian_nll_loss(tape);
///     let _: __FandheElementwiseLossOpsHoldMarker = fandhe_ai::Tensor::<f32>::bce_with_logits_loss_with(tf);
///     let _: __FandheElementwiseLossOpsHoldMarker = fandhe_ai::Tensor::<f32>::hinge_embedding_loss(tf);
///     let _: __FandheElementwiseLossOpsHoldMarker = fandhe_ai::Tensor::<f32>::soft_margin_loss(tf);
///     let _: __FandheElementwiseLossOpsHoldMarker = fandhe_ai::Tensor::<f32>::gaussian_nll_loss(tf);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct ElementwiseLossOpsHoldDoctestGuard;

/// MultiMargin・MultiLabelMargin・MultiLabelSoftMargin・sigmoid focal loss（イシュー #2653・親 #2651）を
/// facade 公開面から締め出す保留ガード（`SoftminThresholdOpsHoldDoctestGuard` と同型の正のプローブ 1 ブロック方式）。
///
/// 下の doctest は全 `pub mod` を glob import したスコープへ、ローカルモジュール `margin_focal_loss_ops`・
/// 型 `MultiMarginOptions`／`MultiLabelSoftMarginOptions`／`SigmoidFocalLossOptions` と、プローブ用トレイトのメソッド（`Var`／`Tape`／`Tensor<f32>` の
/// `multi_margin_loss`／`multilabel_margin_loss`／`multilabel_soft_margin_loss`／`sigmoid_focal_loss`）を置き、
/// モジュール経由の関数呼び出しと修飾付きメソッド呼び出しの両方を行う。facade が同名のモジュール・型を glob 可能な
/// 位置へ公開するか、これらの型へ同名の inherent メソッドを公開すると、名前解決の曖昧性または呼び出しシグネチャの
/// 不一致でエラーコードに依存せずコンパイルが失敗する。検出範囲は列挙したこれらの名前・型に限り、マクロ生成や
/// 別名経由のメソッドまでは保証しない。
///
/// 実装は内部クレートに閉じている（`fandhe_ai_autodiff::margin_focal_loss_ops`）。保留対象は facade 公開面のみで、
/// 公開形は未承認（承認依頼は #2677・公開自体は承認後の #2678。推奨案は
/// `docs/autodiff-margin-focal-loss-ops-decision.md` §7。同記録は推奨案の記録であり承認記録ではない）。
///
/// ソース走査ガード（`crates/facade/tests/api_surface.rs::margin_focal_loss_ops_hold_doctest_globs_all_pub_modules`・
/// `margin_focal_loss_ops_hold_doctest_probe_body_matches_fixed_contract`・
/// `facade_does_not_reexport_or_declare_margin_focal_loss_ops`・
/// `workspace_declares_margin_focal_loss_ops_fn_names_only_in_allowed_locations`）との多層防御として働く。
///
/// 承認を得た日が来たら、本構造体・本 doctest 自体を削除する（ソース走査側の対応する否定ガードも同時に
/// 正ガードへ置き換える）。
///
/// # 正のプローブ: 全 `pub mod` glob import 済みのスコープでコンパイル
/// できること
///
/// ```
/// use fandhe_ai::*;
/// use fandhe_ai::compat::*;
/// use fandhe_ai::optim::*;
/// use fandhe_ai::data::*;
/// use fandhe_ai::nn::*;
/// use fandhe_ai::nn::init::*;
/// use fandhe_ai::nn::rnn::*;
/// use fandhe_ai::interop::*;
/// use fandhe_ai::interop::onnx::*;
/// use fandhe_ai::interop::safetensors::*;
/// use fandhe_ai::model::*;
///
/// mod __fandhe_margin_focal_loss_ops_hold_probe {
///     pub struct MultiMarginOptions;
///     pub struct MultiLabelSoftMarginOptions;
///     pub struct SigmoidFocalLossOptions;
///     pub mod margin_focal_loss_ops {
///         pub fn multi_margin_loss() {}
///         pub fn multilabel_margin_loss() {}
///         pub fn multilabel_soft_margin_loss() {}
///         pub fn sigmoid_focal_loss() {}
///     }
/// }
/// use __fandhe_margin_focal_loss_ops_hold_probe::*;
///
/// struct __FandheMarginFocalLossOpsHoldMarker;
///
/// trait __FandheMarginFocalLossOpsHoldProbe {
///     fn multi_margin_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker;
///     fn multilabel_margin_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker;
///     fn multilabel_soft_margin_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker;
///     fn sigmoid_focal_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker;
/// }
///
/// impl<'t> __FandheMarginFocalLossOpsHoldProbe for fandhe_ai::Var<'t> {
///     fn multi_margin_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker {
///         __FandheMarginFocalLossOpsHoldMarker
///     }
///     fn multilabel_margin_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker {
///         __FandheMarginFocalLossOpsHoldMarker
///     }
///     fn multilabel_soft_margin_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker {
///         __FandheMarginFocalLossOpsHoldMarker
///     }
///     fn sigmoid_focal_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker {
///         __FandheMarginFocalLossOpsHoldMarker
///     }
/// }
///
/// impl __FandheMarginFocalLossOpsHoldProbe for fandhe_ai::Tape {
///     fn multi_margin_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker {
///         __FandheMarginFocalLossOpsHoldMarker
///     }
///     fn multilabel_margin_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker {
///         __FandheMarginFocalLossOpsHoldMarker
///     }
///     fn multilabel_soft_margin_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker {
///         __FandheMarginFocalLossOpsHoldMarker
///     }
///     fn sigmoid_focal_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker {
///         __FandheMarginFocalLossOpsHoldMarker
///     }
/// }
///
/// impl __FandheMarginFocalLossOpsHoldProbe for fandhe_ai::Tensor<f32> {
///     fn multi_margin_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker {
///         __FandheMarginFocalLossOpsHoldMarker
///     }
///     fn multilabel_margin_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker {
///         __FandheMarginFocalLossOpsHoldMarker
///     }
///     fn multilabel_soft_margin_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker {
///         __FandheMarginFocalLossOpsHoldMarker
///     }
///     fn sigmoid_focal_loss(&self) -> __FandheMarginFocalLossOpsHoldMarker {
///         __FandheMarginFocalLossOpsHoldMarker
///     }
/// }
///
/// fn __probe_free_fns(
///     _0: MultiMarginOptions,
///     _1: MultiLabelSoftMarginOptions,
///     _2: SigmoidFocalLossOptions,
/// ) {
///     // モジュール経由の呼び出し（`use fandhe_ai::*;` が同名モジュールを glob 公開していれば、
///     // 名前解決自体が曖昧になり E0659 でコンパイル失敗する）。
///     margin_focal_loss_ops::multi_margin_loss();
///     margin_focal_loss_ops::multilabel_margin_loss();
///     margin_focal_loss_ops::multilabel_soft_margin_loss();
///     margin_focal_loss_ops::sigmoid_focal_loss();
/// }
///
/// fn __probe_methods(
///     v: &fandhe_ai::Var<'_>,
///     tape: &fandhe_ai::Tape,
///     tf: &fandhe_ai::Tensor<f32>,
/// ) {
///     let _: __FandheMarginFocalLossOpsHoldMarker = fandhe_ai::Var::multi_margin_loss(v);
///     let _: __FandheMarginFocalLossOpsHoldMarker = fandhe_ai::Var::multilabel_margin_loss(v);
///     let _: __FandheMarginFocalLossOpsHoldMarker = fandhe_ai::Var::multilabel_soft_margin_loss(v);
///     let _: __FandheMarginFocalLossOpsHoldMarker = fandhe_ai::Var::sigmoid_focal_loss(v);
///     let _: __FandheMarginFocalLossOpsHoldMarker = fandhe_ai::Tape::multi_margin_loss(tape);
///     let _: __FandheMarginFocalLossOpsHoldMarker = fandhe_ai::Tape::multilabel_margin_loss(tape);
///     let _: __FandheMarginFocalLossOpsHoldMarker = fandhe_ai::Tape::multilabel_soft_margin_loss(tape);
///     let _: __FandheMarginFocalLossOpsHoldMarker = fandhe_ai::Tape::sigmoid_focal_loss(tape);
///     let _: __FandheMarginFocalLossOpsHoldMarker = fandhe_ai::Tensor::<f32>::multi_margin_loss(tf);
///     let _: __FandheMarginFocalLossOpsHoldMarker = fandhe_ai::Tensor::<f32>::multilabel_margin_loss(tf);
///     let _: __FandheMarginFocalLossOpsHoldMarker = fandhe_ai::Tensor::<f32>::multilabel_soft_margin_loss(tf);
///     let _: __FandheMarginFocalLossOpsHoldMarker = fandhe_ai::Tensor::<f32>::sigmoid_focal_loss(tf);
/// }
/// ```
#[cfg(doctest)]
#[allow(dead_code)]
struct MarginFocalLossOpsHoldDoctestGuard;

#[cfg(test)]
mod tape_ref_tests {
    use super::*;

    /// `pub(crate)` の `from_autodiff` 経路（統合テストから呼べない）で作った葉が
    /// facade `Tape::backward` で勾配を得られること。
    #[test]
    fn from_autodiff_leaf_receives_gradient() {
        let t = tape();
        let r = TapeRef::from_autodiff(&t.0);
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).expect("テスト入力の構築");
        let v = r.var(&x);
        let loss = v.mul(&v).expect("mul").sum(None).expect("sum");
        let g = t.backward(&loss).expect("backward");
        assert!(g.get(&v).expect("get").is_some());
    }
}
