//! `fandhe_ai::set_cuda_tf32_gemm_enabled`／`fandhe_ai::cuda_tf32_gemm_enabled`
//! （イシュー #1042。CUDA GEMM の TF32 Tensor Core 経路を opt-in で選択
//! する公開 API）と、その 3 モード拡張版 `fandhe_ai::
//! set_cuda_gemm_precision`／`fandhe_ai::cuda_gemm_precision`
//! （イシュー #1355。親ツリー #1354・承認元 #1338）の受入テスト。
//!
//! setter/getter の往復自体はデバイス（CUDA driver）を要さないため、
//! CUDA 非搭載環境（Linux CI・macOS 開発機）でも常に実行できる
//! （`crates/backend-cuda/src/precision.rs` の `AtomicU8` フラグは
//! プロセスグローバルであり `CudaDevice`／driver 初期化を経由しない）。
//! opt-in 時の実際の GEMM 経路切り替え（TF32／3×TF32 Tensor Core
//! カーネルへのルーティング検証）は `crates/backend-cuda/src/ops.rs` の
//! 環境適応テスト（`gemm_routes_to_tf32_path_when_optin_flag_is_enabled_
//! env_adaptive`・`gemm_routes_to_tf32x3_path_when_precision_is_tf32x3_
//! env_adaptive`）が担う（本テストは facade 公開面の往復のみを担当し、
//! 責務を分離する）。

/// フラグはプロセスグローバル（`fandhe_ai_backend_cuda::precision`）の
/// ため、他のテスト・他の `#[test]` 関数との並列実行下での競合を避けて
/// 直列化・原状復帰する RAII ガード（`precision.rs::tests::FlagGuard`・
/// `ops.rs::tests::Tf32FlagGuard` と同型）。
///
/// **enum 保存/復元（イシュー #1355）**: `original` を `bool` ではなく
/// `fandhe_ai::CudaGemmPrecision` で保存する（`ops.rs::tests::
/// Tf32FlagGuard` と同じ理由。`Tf32x3` 状態の lossy な復元を防ぐ）。
struct Tf32FlagGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    original: fandhe_ai::CudaGemmPrecision,
}

impl Tf32FlagGuard {
    fn acquire() -> Self {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let original = fandhe_ai::cuda_gemm_precision();
        Self {
            _lock: lock,
            original,
        }
    }
}

impl Drop for Tf32FlagGuard {
    fn drop(&mut self) {
        fandhe_ai::set_cuda_gemm_precision(self.original);
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

/// `set_cuda_gemm_precision`／`cuda_gemm_precision`（イシュー #1355）の
/// 3 モード往復契約。
#[test]
fn set_cuda_gemm_precision_round_trips_across_all_three_modes() {
    let _guard = Tf32FlagGuard::acquire();
    for mode in [
        fandhe_ai::CudaGemmPrecision::Fp32Strict,
        fandhe_ai::CudaGemmPrecision::Tf32,
        fandhe_ai::CudaGemmPrecision::Tf32x3,
    ] {
        fandhe_ai::set_cuda_gemm_precision(mode);
        assert_eq!(fandhe_ai::cuda_gemm_precision(), mode);
    }
}

/// 互換ラッパーの意味論（イシュー #1355。`precision.rs` モジュール冒頭
/// コメントの契約）: `set_cuda_tf32_gemm_enabled(false)` はどのモード
/// からでも `Fp32Strict` へ戻す。`cuda_tf32_gemm_enabled()` は単発
/// `Tf32` のときのみ `true`（`Tf32x3` では `false`）。
#[test]
fn legacy_bool_api_maps_correctly_to_and_from_tf32x3() {
    let _guard = Tf32FlagGuard::acquire();

    fandhe_ai::set_cuda_gemm_precision(fandhe_ai::CudaGemmPrecision::Tf32x3);
    assert!(
        !fandhe_ai::cuda_tf32_gemm_enabled(),
        "cuda_tf32_gemm_enabled() は Tf32x3 では false を返す契約"
    );

    fandhe_ai::set_cuda_tf32_gemm_enabled(false);
    assert_eq!(
        fandhe_ai::cuda_gemm_precision(),
        fandhe_ai::CudaGemmPrecision::Fp32Strict,
        "set_cuda_tf32_gemm_enabled(false) はどのモードからでも Fp32Strict へ戻す契約"
    );
}
