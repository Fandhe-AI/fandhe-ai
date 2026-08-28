//! composition root（TASK-9.3・イシュー #410・spec 確定は
//! rust-ai-library-spec#52／spec PR #53。`docs/spec/05-tasks.md:315`）。
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

// 公開面として再エクスポートする型（モジュール冒頭「公開面の設計」参照）。
// `fandhe_ai_autodiff::Tape`（生の型）・`fandhe_ai_tensor_core::BackendOps` は意図的に含めない
// （`Tape::new_with_ops` という BackendOps 注入経路が到達可能になるため。
// REQ-12）。`Var`／`Gradients`／`AutodiffError`／`LinearVars`・`Tensor` は
// 迂回経路を持たない値型・エラー型のため facade の正式な公開契約として
// 再エクスポートする（codex-review PR #424 P1 是正）。
pub use fandhe_ai_autodiff::{AutodiffError, Gradients, Var, nn::LinearVars};
pub use fandhe_ai_tensor_core::{BackendError, Device, Tensor};

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

    /// `loss` から逆伝播し勾配を計算する（`fandhe_ai_autodiff::Tape::backward`
    /// への委譲）。
    pub fn backward(&self, loss: &Var<'_>) -> Result<Gradients, AutodiffError> {
        self.0.backward(loss)
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
