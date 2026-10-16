//! `fandhe_ai::set_cuda_elementwise_fusion_enabled`／
//! `fandhe_ai::cuda_elementwise_fusion_enabled`（および macOS 限定の Metal 版）
//! の受入テスト（イシュー #3385。機構は #2085・公開形は #3115 の
//! `docs/facade-graph-tier-b-optin-exposure-decision.md` §3）。
//!
//! setter/getter の往復は `AtomicBool` を触るだけで GPU driver を要さないため、
//! CUDA 非搭載の CI でも常に実行できる。
//!
//! **観測の限界**: B-1 には推論 capture の `infer_graph_stats` に当たる経路
//! カウンタがなく、融合出力は同一バックエンドの per-op と bit 一致する契約の
//! ため ON/OFF の出力比較では経路を区別できない。そこで facade の setter が
//! backend のゲートへ効いていることは、`BackendOps::run_fused` を直接呼んだ
//! 返り値の分類（OFF は `Unsupported`、ON は成功か到達不能エラー）で確認する。
//! 実機での ON 時 parity は `gpu_elementwise_fusion_parity_real_device.rs`
//! （`#[ignore]`）が担う。

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

/// フラグはプロセスグローバルのため、並列テスト間で直列化し原状復帰する
/// RAII ガード（`cuda_graph_infer_optin.rs::InferFlagGuard` と同型）。
struct FusionFlagGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    original_cuda: bool,
    #[cfg(target_os = "macos")]
    original_metal: bool,
}

impl FusionFlagGuard {
    fn acquire() -> Self {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        Self {
            _lock: lock,
            original_cuda: fandhe_ai::cuda_elementwise_fusion_enabled(),
            #[cfg(target_os = "macos")]
            original_metal: fandhe_ai::metal_elementwise_fusion_enabled(),
        }
    }
}

impl Drop for FusionFlagGuard {
    fn drop(&mut self) {
        fandhe_ai::set_cuda_elementwise_fusion_enabled(self.original_cuda);
        #[cfg(target_os = "macos")]
        fandhe_ai::set_metal_elementwise_fusion_enabled(self.original_metal);
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
    .expect("test fixture: valid plan")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// 既定無効の契約を、明示的に `false` へ戻した直後の観測で確認する
/// （他テストが有効化したまま残す可能性があるため、起動直後の真の初期値
/// そのものは検証しない。`cuda_graph_infer_optin.rs::default_is_disabled`
/// と同じ理由）。
#[test]
fn cuda_default_is_disabled() {
    let _guard = FusionFlagGuard::acquire();
    fandhe_ai::set_cuda_elementwise_fusion_enabled(false);
    assert!(!fandhe_ai::cuda_elementwise_fusion_enabled());
}

/// 往復契約と、facade の getter が backend の同一 static を見ていること。
#[test]
fn cuda_set_true_then_false_round_trips() {
    let _guard = FusionFlagGuard::acquire();
    fandhe_ai::set_cuda_elementwise_fusion_enabled(true);
    assert!(fandhe_ai::cuda_elementwise_fusion_enabled());
    assert!(fandhe_ai_backend_cuda::fused_elementwise::gpu_elementwise_fusion_enabled());
    fandhe_ai::set_cuda_elementwise_fusion_enabled(false);
    assert!(!fandhe_ai::cuda_elementwise_fusion_enabled());
    assert!(!fandhe_ai_backend_cuda::fused_elementwise::gpu_elementwise_fusion_enabled());
}

/// facade setter が backend のゲートへ効いている（環境適応）。
#[test]
fn cuda_gate_wiring_env_adaptive() {
    use fandhe_ai_backend_cuda::CudaBackendOps;
    let _guard = FusionFlagGuard::acquire();
    let cuda = CudaBackendOps::new(0);
    let plan = build_4_chain_plan(8);
    let a = Tensor::new(vec![0.1f32; 8], &[8]).expect("valid tensor");
    let b = Tensor::new(vec![0.2f32; 8], &[8]).expect("valid tensor");

    fandhe_ai::set_cuda_elementwise_fusion_enabled(false);
    let off = cuda.run_fused(&plan, &[&a, &b]);
    assert!(
        matches!(off, Err(BackendError::Unsupported(_))),
        "OFF はデバイスに触れず Unsupported: {off:?}"
    );

    fandhe_ai::set_cuda_elementwise_fusion_enabled(true);
    match cuda.run_fused(&plan, &[&a, &b]) {
        Ok(out) => assert_eq!(out.shape(), &[8]),
        Err(BackendError::CudaUnavailable(_)) => {}
        Err(other) => panic!("unexpected error on CUDA-equipped runner: {other}"),
    }
}

/// CPU のみの環境で OFF → ON → OFF と切り替えても panic せず、CPU tape の
/// 出力が bit 完全に一致する（非対応環境では効果を持たない契約）。
#[test]
fn toggling_on_cpu_keeps_output_bit_identical() {
    let _guard = FusionFlagGuard::acquire();
    let run = || {
        let tape = fandhe_ai::tape();
        let a = tape.var(&Tensor::new(vec![0.1f32, -0.4, 0.7, 1.3], &[4]).unwrap());
        let b = tape.var(&Tensor::new(vec![0.2f32, 0.5, -0.9, 0.3], &[4]).unwrap());
        let y = a.add(&b).unwrap().relu().exp().tanh();
        bits(&y.to_tensor())
    };

    fandhe_ai::set_cuda_elementwise_fusion_enabled(false);
    let off1 = run();
    fandhe_ai::set_cuda_elementwise_fusion_enabled(true);
    let on = run();
    fandhe_ai::set_cuda_elementwise_fusion_enabled(false);
    let off2 = run();

    assert_eq!(off1, on);
    assert_eq!(off1, off2);
}

#[cfg(target_os = "macos")]
mod metal {
    use super::*;
    use fandhe_ai_backend_metal::MetalBackendOps;

    #[test]
    fn metal_default_is_disabled() {
        let _guard = FusionFlagGuard::acquire();
        fandhe_ai::set_metal_elementwise_fusion_enabled(false);
        assert!(!fandhe_ai::metal_elementwise_fusion_enabled());
    }

    #[test]
    fn metal_set_true_then_false_round_trips() {
        let _guard = FusionFlagGuard::acquire();
        fandhe_ai::set_metal_elementwise_fusion_enabled(true);
        assert!(fandhe_ai::metal_elementwise_fusion_enabled());
        assert!(fandhe_ai_backend_metal::fused_elementwise::gpu_elementwise_fusion_enabled());
        fandhe_ai::set_metal_elementwise_fusion_enabled(false);
        assert!(!fandhe_ai::metal_elementwise_fusion_enabled());
    }

    #[test]
    fn metal_gate_wiring_env_adaptive() {
        let _guard = FusionFlagGuard::acquire();
        let metal = MetalBackendOps::new();
        let plan = build_4_chain_plan(8);
        let a = Tensor::new(vec![0.1f32; 8], &[8]).expect("valid tensor");
        let b = Tensor::new(vec![0.2f32; 8], &[8]).expect("valid tensor");

        fandhe_ai::set_metal_elementwise_fusion_enabled(false);
        let off = metal.run_fused(&plan, &[&a, &b]);
        assert!(
            matches!(off, Err(BackendError::Unsupported(_))),
            "OFF は Unsupported: {off:?}"
        );

        fandhe_ai::set_metal_elementwise_fusion_enabled(true);
        if let Ok(out) = metal.run_fused(&plan, &[&a, &b]) {
            assert_eq!(out.shape(), &[8]);
        }
    }
}
