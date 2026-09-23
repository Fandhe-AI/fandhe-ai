//! GPU `run_fused` の elementwise allowlist 融合（区分 B-1・イシュー
//! #2085）の CUDA 実機検証。
//!
//! `cast_parity.rs`・`gemm_tf32_optin.rs` と同じ構成方針: 環境適応
//! スモーク（属性なし。CUDA 非搭載環境では `BackendError::
//! CudaUnavailable` を確認して panic しないことのみ検証。ゲート
//! OFF・ゲート ON 双方のデバイス非依存分岐は本クレートの単体テスト
//! 〈`fused_elementwise.rs::tests`〉で既に検証済みのため、ここでは
//! デバイス到達性の確認に限る）と、実機必須の網羅
//! （`#[ignore]`。DGX Spark GB10 等）を分離する。
//!
//! **数値契約**: ゲート ON 時、融合カーネル出力は同一バックエンドの
//! per-op 合成（`ops.add`／`mul`／`relu`／`exp`／`tanh` の逐次呼び出し）
//! と **bit 完全一致**（`kernels_fused_elementwise.rs` モジュール冒頭
//! 「数値契約」）。CPU 融合結果とは REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）で突合する。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test fused_elementwise_parity -- --ignored --nocapture
//! ```

use std::sync::Mutex;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::fused_elementwise::run_fused_elementwise;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_backend_cuda::fused_elementwise::{
    gpu_elementwise_fusion_enabled, set_gpu_elementwise_fusion_enabled,
};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

/// ゲートはプロセスグローバルのため、`cargo test` の既定並列実行下でも
/// 直列化する（`gemm_tf32_optin.rs::FLAG_LOCK` と同型）。
static GATE_LOCK: Mutex<()> = Mutex::new(());

struct GateGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    original: bool,
}

impl GateGuard {
    fn acquire(enabled: bool) -> Self {
        let lock = GATE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let original = gpu_elementwise_fusion_enabled();
        set_gpu_elementwise_fusion_enabled(enabled);
        Self {
            _lock: lock,
            original,
        }
    }
}

impl Drop for GateGuard {
    fn drop(&mut self) {
        set_gpu_elementwise_fusion_enabled(self.original);
    }
}

/// 4 段連鎖（`add → relu → exp → tanh`）の `FusionPlan` を組み立てる
/// （実装計画 §5.2 のパターンの 1 つ）。
fn build_4_chain_plan(numel: usize) -> FusionPlan {
    FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Input { leaf_index: 1 },
            FusedOpKind::Add { lhs: 0, rhs: 1 },
            FusedOpKind::Relu { input: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Tanh { input: 4 },
        ],
        vec![numel],
        DType::F32,
        2,
    )
    .expect("valid plan")
}

#[test]
fn fused_elementwise_smoke_env_adaptive() {
    let _guard = GateGuard::acquire(true);
    let cuda = CudaBackendOps::new(0);
    let plan = build_4_chain_plan(8);
    let a = Tensor::new(vec![0.1f32; 8], &[8]).expect("valid tensor");
    let b = Tensor::new(vec![0.2f32; 8], &[8]).expect("valid tensor");
    match cuda.run_fused(&plan, &[&a, &b]) {
        Ok(out) => {
            assert_eq!(out.shape(), &[8]);
        }
        Err(BackendError::CudaUnavailable(_)) => {
            // CUDA 非搭載環境（通常 CI）。panic せず終了する。
        }
        Err(other) => panic!("unexpected error on CUDA-equipped runner: {other}"),
    }
}

/// ゲート OFF はデバイス非依存で `Unsupported` を返す
/// （`ops.rs::run_fused` の呼び出し順序契約。CUDA 非搭載環境でも
/// `CudaUnavailable` にならないことを確認する）。
#[test]
fn fused_elementwise_gate_off_is_unsupported_without_device_access() {
    let _guard = GateGuard::acquire(false);
    let cuda = CudaBackendOps::new(0);
    let plan = build_4_chain_plan(8);
    let a = Tensor::new(vec![0.1f32; 8], &[8]).expect("valid tensor");
    let b = Tensor::new(vec![0.2f32; 8], &[8]).expect("valid tensor");
    let result = cuda.run_fused(&plan, &[&a, &b]);
    assert!(
        matches!(result, Err(BackendError::Unsupported(_))),
        "gate OFF must return Unsupported (not CudaUnavailable) without touching the device: \
         {result:?}"
    );
}

/// より広い形状・パターン網羅を実機で確認する（DGX Spark GB10 等。
/// `#[ignore]`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn fused_elementwise_matches_per_op_and_cpu_across_shapes() {
    let _guard = GateGuard::acquire(true);
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    for (i, &n) in [1usize, 2, 255, 256, 257, 4096, 1 << 16].iter().enumerate() {
        let mut rng = Xorshift64Star::new(2_085_000 + i as u64);
        let a_data = rng.fill_vec(n);
        let b_data = rng.fill_vec(n);
        let a = Tensor::new(a_data.clone(), &[n]).expect("valid tensor");
        let b = Tensor::new(b_data.clone(), &[n]).expect("valid tensor");
        let plan = build_4_chain_plan(n);

        let fused = cuda
            .run_fused(&plan, &[&a, &b])
            .unwrap_or_else(|e| panic!("cuda run_fused failed at n={n}: {e}"));

        // per-op 合成（同一バックエンド。bit 完全一致契約）。
        let add = cuda.add(&a, &b).expect("cuda add");
        let relu = cuda.relu(&add).expect("cuda relu");
        let exp = cuda.exp(&relu).expect("cuda exp");
        let tanh = cuda.tanh(&exp).expect("cuda tanh");
        assert_eq!(
            fused.as_slice().expect("contiguous"),
            tanh.as_slice().expect("contiguous"),
            "fused vs per-op mismatch at n={n}"
        );

        // CPU 融合カーネルとの REQ-2 統一複合判定。
        let cpu_fused = run_fused_elementwise(&plan, &[&a, &b]).expect("cpu run_fused_elementwise");
        assert_parity(
            &format!("fused_elementwise n={n}"),
            fused.as_slice().expect("contiguous"),
            cpu_fused.as_slice().expect("contiguous"),
        );

        let _ = &cpu;
    }
}

/// ゲート OFF 時は per-op フォールバックが機能する（`Unsupported` を
/// 検出した呼び出し元が `Tape` 実体化経路で per-op を使う契約の裏付け。
/// 実機で `Unsupported` が確実に返ることを確認する意味も持つ）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn fused_elementwise_gate_off_returns_unsupported_on_real_device() {
    let _guard = GateGuard::acquire(false);
    let cuda = CudaBackendOps::new(0);
    let plan = build_4_chain_plan(16);
    let a = Tensor::new(vec![0.1f32; 16], &[16]).expect("valid tensor");
    let b = Tensor::new(vec![0.2f32; 16], &[16]).expect("valid tensor");
    assert!(matches!(
        cuda.run_fused(&plan, &[&a, &b]),
        Err(BackendError::Unsupported(_))
    ));
}
