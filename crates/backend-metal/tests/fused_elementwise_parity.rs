//! GPU `run_fused` の elementwise allowlist 融合（区分 B-1・イシュー
//! #2085）の Metal 実機検証（CUDA 側
//! `crates/backend-cuda/tests/fused_elementwise_parity.rs` の Metal
//! 対応版）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`cast_parity.rs` と同方針。`#![cfg(target_os = "macos")]` により
//! Linux CI ではコンパイル対象外になり、`#[ignore]` により通常の
//! `cargo test` からも除外される）。
//!
//! **数値契約**: ゲート ON 時、融合カーネル出力は同一バックエンドの
//! per-op 合成（`ops.add`／`mul`／`relu`／`exp`／`tanh` の逐次呼び出し）
//! と **bit 完全一致**（`fused_elementwise_source.rs` モジュール冒頭
//! 「数値契約」）。CPU 融合結果とは REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）で突合する。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test fused_elementwise_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::sync::Mutex;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::fused_elementwise::run_fused_elementwise;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_backend_metal::fused_elementwise::{
    gpu_elementwise_fusion_enabled, set_gpu_elementwise_fusion_enabled,
};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

/// ゲートはプロセスグローバルのため直列化する（CUDA 側
/// `gemm_tf32_optin.rs::FLAG_LOCK` と同型）。
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

/// ゲート OFF はデバイス非依存で `Unsupported` を返す
/// （`ops.rs::run_fused` の呼び出し順序契約）。Metal はシステム
/// デフォルトデバイス 1 台のみを扱う前提のため CUDA 側の
/// 「CudaUnavailable にならない」確認とは異なり、ここでは
/// `context_cache::cached_context` 自体が呼ばれずに `Unsupported` が
/// 返ることのみを確認する（デバイス到達性自体は既存
/// `elementwise_matches_cpu_across_ops` 等が別途検証済み）。
#[test]
fn fused_elementwise_gate_off_is_unsupported() {
    let _guard = GateGuard::acquire(false);
    let metal = MetalBackendOps::new();
    let plan = build_4_chain_plan(8);
    let a = Tensor::new(vec![0.1f32; 8], &[8]).expect("valid tensor");
    let b = Tensor::new(vec![0.2f32; 8], &[8]).expect("valid tensor");
    let result = metal.run_fused(&plan, &[&a, &b]);
    assert!(
        matches!(result, Err(BackendError::Unsupported(_))),
        "gate OFF must return Unsupported: {result:?}"
    );
}

/// より広い形状・パターン網羅を実機で確認する（Apple Silicon 実機。
/// `#[ignore]`）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn fused_elementwise_matches_per_op_and_cpu_across_shapes() {
    let _guard = GateGuard::acquire(true);
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    for (i, &n) in [1usize, 2, 255, 256, 257, 4096, 1 << 16].iter().enumerate() {
        let mut rng = Xorshift64Star::new(2_085_100 + i as u64);
        let a_data = rng.fill_vec(n);
        let b_data = rng.fill_vec(n);
        let a = Tensor::new(a_data.clone(), &[n]).expect("valid tensor");
        let b = Tensor::new(b_data.clone(), &[n]).expect("valid tensor");
        let plan = build_4_chain_plan(n);

        let fused = metal
            .run_fused(&plan, &[&a, &b])
            .unwrap_or_else(|e| panic!("metal run_fused failed at n={n}: {e}"));

        let add = metal.add(&a, &b).expect("metal add");
        let relu = metal.relu(&add).expect("metal relu");
        let exp = metal.exp(&relu).expect("metal exp");
        let tanh = metal.tanh(&exp).expect("metal tanh");
        assert_eq!(
            fused.as_slice().expect("contiguous"),
            tanh.as_slice().expect("contiguous"),
            "fused vs per-op mismatch at n={n}"
        );

        let cpu_fused = run_fused_elementwise(&plan, &[&a, &b]).expect("cpu run_fused_elementwise");
        assert_parity(
            &format!("fused_elementwise n={n}"),
            fused.as_slice().expect("contiguous"),
            cpu_fused.as_slice().expect("contiguous"),
        );

        let _ = &cpu;
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn fused_elementwise_gate_off_returns_unsupported_on_real_device() {
    let _guard = GateGuard::acquire(false);
    let metal = MetalBackendOps::new();
    let plan = build_4_chain_plan(16);
    let a = Tensor::new(vec![0.1f32; 16], &[16]).expect("valid tensor");
    let b = Tensor::new(vec![0.2f32; 16], &[16]).expect("valid tensor");
    assert!(matches!(
        metal.run_fused(&plan, &[&a, &b]),
        Err(BackendError::Unsupported(_))
    ));
}
