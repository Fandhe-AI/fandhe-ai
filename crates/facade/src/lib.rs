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
//! 3. **optim 公開面**（[`optim`]。イシュー #961・親 #960）: SGD・AdamW・
//!    gradient clipping・LR スケジューラを `fandhe_ai::optim` の単一入口
//!    へ再エクスポートする。値型・純関数のみのため REQ-12 と矛盾しない
//!    （詳細は [`optim`] モジュール doc）。
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

/// optimizer 公開面（イシュー #961・親 #960）。SGD・AdamW・gradient
/// clipping・LR スケジューラを再エクスポートする（詳細・適用順序契約は
/// モジュール doc 参照）。
pub mod optim;

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
    /// 済みの重み勾配のみをホストへ読み出す（bias 等の未充填 slot は
    /// `None`）。resident 未対応バックエンド（CUDA。`gemm_fp32_
    /// strict_into` 未実装。CPU・Metal はイシュー #1555 で対応済み）では
    /// [`BackendError::Unsupported`] を返す（panic なし）。全パラメータの
    /// 勾配をバックエンド横断で読みたい場合は
    /// [`Self::param_grads_to_host`] を使う。上記 `sync_device_param_
    /// store_to_host` と同じ理由の薄い委譲。
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
/// GEMM の各 `run_*`／`launch_*` 系）のみ対応。TF32／f16 Tensor Core
/// 経路（`run_wmma_*`／`run_f16_kernel`）・elementwise／rmsnorm／
/// softmax／transpose／mse は本イシューでは pinned 化していない
/// （既定どおり pageable のまま変更なし）。
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
