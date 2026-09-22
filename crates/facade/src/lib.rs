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
//!    入口となる。値型の再エクスポート＋薄い委譲のみで任意
//!    `BackendOps` 注入経路を新設しないため REQ-12 と矛盾しない
//!    （詳細は [`nn::rnn`] モジュール doc）。
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

/// `nn` 公開面（イシュー #1955）。現時点は [`nn::rnn`]
/// （`Rnn`／`Lstm`／`Gru` の Sequence レベル API の純再エクスポート）
/// のみを提供する。`forward_seq` の呼び出しには [`Tape::rnn_forward_seq`]
/// 等（本モジュール自体ではなく `impl Tape` の薄い委譲メソッド）を
/// 使う（詳細は [`nn::rnn`] モジュール doc 参照）。
pub mod nn;

/// 相互運用（interop）公開面の入口（イシュー #2017・#2018・#2019）。
/// [`interop::onnx`]（ONNX import／export。`OnnxModel`／`OnnxValue`／
/// `OnnxError`・`OnnxModel::{from_bytes, from_path, run, to_bytes,
/// to_path}`・`OnnxExportOptions`。export は #2018 で公開済み・roundtrip
/// export ラッパー限定）に加え、[`interop::safetensors`]（safetensors
/// save／load 純再エクスポート。イシュー #2019）を提供する
/// （`docs/facade-onnx-export-exposure-decision.md`・`docs/facade-
/// safetensors-exposure-decision.md`）。
pub mod interop;

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
use fandhe_ai_autodiff::nn::optim::{AdamConfig, AdamWConfig};
pub use fandhe_ai_autodiff::{AutodiffError, Gradients, Var, nn::LinearVars};
// `VarHostView`（借用ビュー読み出し API。イシュー #1335）は 1 文 1 行を
// 維持する（`tests/api_surface.rs` が `pub use` を行単位で走査するため。
// 上記コメント「1 文 1 行を維持する」参照）。
pub use fandhe_ai_autodiff::VarHostView;
// 線形代数（イシュー #1621・`docs/autodiff-linalg-design.md`）の多出力
// 戻り値型（`Var::qr`／`Var::svd`）は 1 文 1 行で再エクスポートする
// （上記コメント「1 文 1 行を維持する」と同じ理由）。
pub use fandhe_ai_autodiff::QrVars;
pub use fandhe_ai_autodiff::SvdVars;
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
            let provider = fandhe_ai_backend_metal::MetalDeviceProvider::new();
            let providers: [&dyn DeviceProvider; 1] = [&provider];
            select_from(&providers, device)?;
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
/// [`set_metal_onnx_gpu_execution_enabled`]）が有効な場合は CUDA を
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
