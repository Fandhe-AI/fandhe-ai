//! CUDA `DeviceBuffer` の確保配置（managed／device-only）を opt-in で
//! 選択する公開スイッチ（イシュー #1352。親 #1351「GB10 物理統合メモリ
//! 向けゼロコピー割当の試作・実測」・兄弟 #1353「実測に基づく既定化
//! 可否判断」）。
//!
//! DGX Spark GB10 のようなホスト・GPU 物理統合メモリ環境では
//! `cuMemAlloc` + `cuMemcpyHtoD`/`cuMemcpyDtoH` の往復（H2D/D2H）が
//! 本来不要になりうる。本モジュールはプロセスワイドな `AtomicBool`
//! フラグ（`crate::precision::TF32_GEMM_ENABLED` と同型の opt-in 方式）
//! で、`memory.rs` の確保経路（`alloc_zeroed`／`upload`）が
//! `cuMemAllocManaged`（`cudarc::driver::safe::unified_memory::
//! CudaContext::alloc_unified`）由来の `UnifiedSlice` を使うか、従来の
//! `cuMemAlloc`（`CudaSlice`）を使うかを切り替える。
//!
//! **契約（既定 OFF・プロセスワイド・fail-closed）**:
//! - 既定値は `false`（従来の `cuMemAlloc` 経路）。フラグ OFF 時の
//!   `memory.rs`／`ops.rs`／`sgd.rs`／`gemm.rs` の経路・出力は本イシュー
//!   導入前と bit-exact に不変（`memory.rs::CudaStorage` ドキュメンテー
//!   ションコメント参照）。
//! - opt-in（`true`）時、対象デバイスが managed memory 非対応
//!   （`crate::device::CudaDevice::managed_memory_supported()` が
//!   `false`）の場合は **fail-closed**: `CudaError::
//!   ManagedMemoryUnsupported` を返し、`cuMemAlloc` への黙示フォール
//!   バックはしない（`crate::precision` の TF32 opt-in と同じ方針。
//!   明示 opt-in の計測条件を静かに崩さない）。
//! - 出力の数値的な同一性契約: managed 配置はメモリの物理的な置き場所
//!   （ホスト・デバイス統合メモリ上か、デバイス専用メモリ上か）のみを
//!   変える。カーネル本体・起動 config は device-only 経路と完全に共有
//!   するため（`memory.rs::CudaArg`／`CudaArgMut` が両配置を同一
//!   `PushKernelArg` 経路へ橋渡しする）、出力は配置に依らず bit
//!   同一となる契約（`docs/backend-cuda-managed-placement-decision.md`
//!   参照）。
//! - `PooledMemory<CudaMemory>`（`tensor-core::pool`）等でバッファが
//!   再利用される場合、配置は確保時点のフラグ値で固定される（プール
//!   再利用時に配置を付け替える機構は持たない。現状の本番経路は
//!   `PooledMemory` を CUDA バックエンドで構築しないため実害はない）。
//!
//! プロセスワイドである理由: `crate::precision` と同じく、`facade` の
//! 公開 API（`fandhe_ai::set_cuda_managed_memory_enabled`）はデバイス
//! ハンドルを介さないグローバルスイッチとして設計する（`Device` 単位の
//! インスタンスを都度引き回す設計は呼び出し側の負担が大きい）。
//! `AtomicBool`（`Ordering::SeqCst`。頻度が低い設定変更のため緩い順序
//! による最適化は不要）はスレッド間で安全に共有できるため `Mutex` 等は
//! 要さない。

use std::sync::atomic::{AtomicBool, Ordering};

/// managed 配置の opt-in フラグ本体。既定 `false`（`cuMemAlloc` 経路）。
static MANAGED_PLACEMENT_ENABLED: AtomicBool = AtomicBool::new(false);

/// CUDA `DeviceBuffer` の確保配置（`memory.rs::CudaMemory::
/// alloc_zeroed`／`upload`）を managed memory へ切り替える。
/// プロセスワイドな設定であり、以降の全スレッド・全 `CudaMemory`
/// インスタンスの確保呼び出しに反映される（`facade::
/// set_cuda_managed_memory_enabled` から委譲される。モジュール冒頭
/// コメントの契約を参照）。
pub fn set_managed_placement_enabled(enabled: bool) {
    MANAGED_PLACEMENT_ENABLED.store(enabled, Ordering::SeqCst);
}

/// 現在の managed 配置 opt-in 状態を返す（既定 `false`）。
pub fn managed_placement_enabled() -> bool {
    MANAGED_PLACEMENT_ENABLED.load(Ordering::SeqCst)
}

/// `MANAGED_PLACEMENT_ENABLED` を操作するテスト間で共有する直列化ロック
/// （`cfg(test)` 限定）。`crate::precision::test_support::
/// tf32_flag_test_lock` と同じ理由（`cargo test` の既定並列実行下でも
/// プロセスグローバルフラグの直列化を保つ）で、本フラグ専用の別ロック
/// として用意する（TF32 フラグとは独立の軸のため共有しない）。
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    /// フラグ操作テスト全体で共有する単一ロックを返す。
    pub(crate) fn placement_flag_test_lock() -> &'static Mutex<()> {
        static LOCK: Mutex<()> = Mutex::new(());
        &LOCK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// フラグはプロセスグローバルのため、他のテストとの競合を避けて
    /// 直列化・原状復帰する RAII ガード（`crate::precision::tests::
    /// FlagGuard` と同型）。
    struct FlagGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        original: bool,
    }

    impl FlagGuard {
        fn acquire() -> Self {
            let lock = test_support::placement_flag_test_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let original = managed_placement_enabled();
            Self {
                _lock: lock,
                original,
            }
        }
    }

    impl Drop for FlagGuard {
        fn drop(&mut self) {
            set_managed_placement_enabled(self.original);
        }
    }

    #[test]
    fn default_is_disabled_when_no_prior_test_left_it_enabled() {
        let _guard = FlagGuard::acquire();
        set_managed_placement_enabled(false);
        assert!(!managed_placement_enabled());
    }

    #[test]
    fn set_true_then_false_round_trips() {
        let _guard = FlagGuard::acquire();
        set_managed_placement_enabled(true);
        assert!(managed_placement_enabled());
        set_managed_placement_enabled(false);
        assert!(!managed_placement_enabled());
    }
}
