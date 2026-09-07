//! `fandhe_ai::set_cuda_graph_step_enabled`／`fandhe_ai::cuda_graph_step_enabled`
//! （イシュー #1349。学習 step の update 区間を CUDA Graph で capture・
//! 再利用する経路を opt-in で選択する公開 API）の受入テスト。
//!
//! `cuda_tf32_gemm_optin.rs` と同じ理由で、setter/getter の往復自体は
//! デバイス（CUDA driver）を要さない（`fandhe_ai_backend_cuda::graph`
//! の `AtomicU8` フラグはプロセスグローバルであり `CudaDevice`／driver
//! 初期化を経由しない）ため CUDA 非搭載環境でも常に実行できる。
//!
//! opt-in 時の実際の capture・再生（`SegmentRun::Captured`／`Replayed`）
//! の検証は `crates/backend-cuda/tests/graph_capture_real_device.rs`
//! （`#[ignore]`。実機必須）・受け入れ条件 (a)（10 step の bit 同一）は
//! `cuda_graph_step_bit_identity.rs`（同）が担う。本テストは facade
//! 公開面の往復のみを担当し責務を分離する。

/// フラグはプロセスグローバル（`fandhe_ai_backend_cuda::graph`）のため、
/// 他のテスト・他の `#[test]` 関数との並列実行下での競合を避けて
/// 直列化・原状復帰する RAII ガード（`cuda_tf32_gemm_optin.rs::
/// Tf32FlagGuard` と同型）。
struct GraphStepFlagGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    original: bool,
}

impl GraphStepFlagGuard {
    fn acquire() -> Self {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let original = fandhe_ai::cuda_graph_step_enabled();
        Self {
            _lock: lock,
            original,
        }
    }
}

impl Drop for GraphStepFlagGuard {
    fn drop(&mut self) {
        fandhe_ai::set_cuda_graph_step_enabled(self.original);
    }
}

/// 既定値は無効である契約を、明示的に `false` へ戻した直後の観測で
/// 確認する（他テストが有効化したまま残す可能性があるため、プロセス
/// 起動直後の真の初期値そのものは検証しない。`cuda_tf32_gemm_optin.rs::
/// default_is_disabled` と同じ理由）。
#[test]
fn default_is_disabled() {
    let _guard = GraphStepFlagGuard::acquire();
    fandhe_ai::set_cuda_graph_step_enabled(false);
    assert!(!fandhe_ai::cuda_graph_step_enabled());
}

/// setter/getter の往復契約。
#[test]
fn set_true_then_false_round_trips() {
    let _guard = GraphStepFlagGuard::acquire();
    fandhe_ai::set_cuda_graph_step_enabled(true);
    assert!(fandhe_ai::cuda_graph_step_enabled());
    fandhe_ai::set_cuda_graph_step_enabled(false);
    assert!(!fandhe_ai::cuda_graph_step_enabled());
}
