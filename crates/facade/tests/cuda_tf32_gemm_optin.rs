//! `fandhe_ai::set_cuda_tf32_gemm_enabled`／`fandhe_ai::cuda_tf32_gemm_enabled`
//! （イシュー #1042。CUDA GEMM の TF32 Tensor Core 経路を opt-in で選択
//! する公開 API）の受入テスト。
//!
//! setter/getter の往復自体はデバイス（CUDA driver）を要さないため、
//! CUDA 非搭載環境（Linux CI・macOS 開発機）でも常に実行できる
//! （`crates/backend-cuda/src/precision.rs` の `AtomicBool` フラグは
//! プロセスグローバルであり `CudaDevice`／driver 初期化を経由しない）。
//! opt-in 時の実際の GEMM 経路切り替え（TF32 Tensor Core カーネルへの
//! ルーティング検証）は `crates/backend-cuda/src/ops.rs` の
//! 環境適応テスト（`gemm_routes_to_tf32_path_when_optin_flag_is_enabled_
//! env_adaptive`）が担う（本テストは facade 公開面の往復のみを担当し、
//! 責務を分離する）。

/// フラグはプロセスグローバル（`fandhe_ai_backend_cuda::precision`）の
/// ため、他のテスト・他の `#[test]` 関数との並列実行下での競合を避けて
/// 直列化・原状復帰する RAII ガード（`precision.rs::tests::FlagGuard`・
/// `ops.rs::tests::Tf32FlagGuard` と同型）。
struct Tf32FlagGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    original: bool,
}

impl Tf32FlagGuard {
    fn acquire() -> Self {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let original = fandhe_ai::cuda_tf32_gemm_enabled();
        Self {
            _lock: lock,
            original,
        }
    }
}

impl Drop for Tf32FlagGuard {
    fn drop(&mut self) {
        fandhe_ai::set_cuda_tf32_gemm_enabled(self.original);
    }
}

/// 既定値は無効（FP32 厳密）である契約（イシュー #1042 実装計画 §2.1）を、
/// 明示的に `false` へ戻した直後の観測で確認する（他テストが有効化した
/// まま残す可能性があるため、プロセス起動直後の真の初期値そのものは
/// 検証しない。`precision.rs::tests::default_is_disabled_when_no_prior_
/// test_left_it_enabled` と同じ理由）。
#[test]
fn default_is_disabled() {
    let _guard = Tf32FlagGuard::acquire();
    fandhe_ai::set_cuda_tf32_gemm_enabled(false);
    assert!(!fandhe_ai::cuda_tf32_gemm_enabled());
}

/// setter/getter の往復契約。
#[test]
fn set_true_then_false_round_trips() {
    let _guard = Tf32FlagGuard::acquire();
    fandhe_ai::set_cuda_tf32_gemm_enabled(true);
    assert!(fandhe_ai::cuda_tf32_gemm_enabled());
    fandhe_ai::set_cuda_tf32_gemm_enabled(false);
    assert!(!fandhe_ai::cuda_tf32_gemm_enabled());
}
