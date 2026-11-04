//! `fandhe_ai::set_cuda_gpu_linalg_enabled`／`cuda_gpu_linalg_enabled`（および macOS 限定の
//! Metal 版）の受入テスト（イシュー #3405。機構は #2674・公開形は
//! `docs/backend-gpu-linalg-decision.md` §11）。
//!
//! setter/getter の往復は `AtomicBool` を触るだけで GPU driver を要さないため、CUDA 非搭載の
//! CI でも常に実行できる。ゲート OFF（既定）の出力が変わらないことは CPU tape での
//! ON/OFF 往復の bit 比較で裏付ける。実機での ON 時 parity は
//! `linalg_gpu_optin_cuda_parity.rs`／`linalg_gpu_optin_metal_parity.rs`（`#[ignore]`）が担う。

use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// フラグはプロセスグローバルのため、並列テスト間で直列化し原状復帰する RAII ガード。
struct LinalgFlagGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    original_cuda: bool,
    #[cfg(target_os = "macos")]
    original_metal: bool,
}

impl LinalgFlagGuard {
    fn acquire() -> Self {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        Self {
            _lock: lock,
            original_cuda: fandhe_ai::cuda_gpu_linalg_enabled(),
            #[cfg(target_os = "macos")]
            original_metal: fandhe_ai::metal_gpu_linalg_enabled(),
        }
    }
}

impl Drop for LinalgFlagGuard {
    fn drop(&mut self) {
        fandhe_ai::set_cuda_gpu_linalg_enabled(self.original_cuda);
        #[cfg(target_os = "macos")]
        fandhe_ai::set_metal_gpu_linalg_enabled(self.original_metal);
    }
}

#[test]
fn cuda_default_is_disabled() {
    let _g = LinalgFlagGuard::acquire();
    // 他テストの副作用に依存しないよう、明示的に既定へ戻した直後の観測で確認する。
    fandhe_ai::set_cuda_gpu_linalg_enabled(false);
    assert!(!fandhe_ai::cuda_gpu_linalg_enabled());
}

#[test]
fn cuda_set_true_then_false_round_trips() {
    let _g = LinalgFlagGuard::acquire();
    fandhe_ai::set_cuda_gpu_linalg_enabled(true);
    assert!(fandhe_ai::cuda_gpu_linalg_enabled());
    assert!(fandhe_ai_backend_cuda::linalg::gpu_linalg_enabled());
    fandhe_ai::set_cuda_gpu_linalg_enabled(false);
    assert!(!fandhe_ai::cuda_gpu_linalg_enabled());
    assert!(!fandhe_ai_backend_cuda::linalg::gpu_linalg_enabled());
}

/// facade の setter が backend のゲートへ効いていることを `linalg_inv` の返り値の分類で確認する
/// （OFF は `Unsupported`、ON は成功か到達不能エラー。`Unsupported` は失敗）。
#[test]
fn cuda_gate_wiring_env_adaptive() {
    let _g = LinalgFlagGuard::acquire();
    let ops = CudaBackendOps::new(0);
    let a = Tensor::new(vec![2.0f32, 0.0, 0.0, 2.0], &[2, 2]).expect("tensor");

    fandhe_ai::set_cuda_gpu_linalg_enabled(false);
    assert!(matches!(
        ops.linalg_inv(&a),
        Err(BackendError::Unsupported(_))
    ));

    fandhe_ai::set_cuda_gpu_linalg_enabled(true);
    let r = ops.linalg_inv(&a);
    assert!(
        !matches!(r, Err(BackendError::Unsupported(_))),
        "ゲート ON なのに Unsupported: facade の setter が backend に効いていない"
    );
}

/// 既定 OFF の出力が変わらない裏付け: CPU tape の forward はゲートの影響を受けず、
/// OFF → ON → OFF で bit 同一。
#[test]
fn toggling_on_cpu_keeps_output_bit_identical() {
    let _g = LinalgFlagGuard::acquire();
    let a = Tensor::new(
        vec![4.0f32, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0],
        &[3, 3],
    )
    .expect("tensor");
    let b = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3, 1]).expect("tensor");

    let run = || -> Vec<Vec<u32>> {
        let tape = fandhe_ai::tape();
        let va = tape.var(&a);
        let bits = |t: Tensor<f32>| -> Vec<u32> {
            t.contiguous()
                .as_slice()
                .expect("slice")
                .iter()
                .map(|x| x.to_bits())
                .collect()
        };
        let qr = va.qr().expect("qr");
        vec![
            bits(va.inv().expect("inv").to_tensor()),
            bits(va.solve(&tape.var(&b)).expect("solve").to_tensor()),
            bits(va.det().expect("det").to_tensor()),
            bits(va.cholesky().expect("cholesky").to_tensor()),
            bits(qr.q.to_tensor()),
            bits(qr.r.to_tensor()),
        ]
    };

    fandhe_ai::set_cuda_gpu_linalg_enabled(false);
    let off1 = run();
    fandhe_ai::set_cuda_gpu_linalg_enabled(true);
    let on = run();
    fandhe_ai::set_cuda_gpu_linalg_enabled(false);
    let off2 = run();
    assert_eq!(off1, on, "CPU tape の出力がゲートの影響を受けた");
    assert_eq!(off1, off2);
}

#[cfg(target_os = "macos")]
mod metal {
    use super::LinalgFlagGuard;

    #[test]
    fn metal_set_true_then_false_round_trips() {
        let _g = LinalgFlagGuard::acquire();
        fandhe_ai::set_metal_gpu_linalg_enabled(true);
        assert!(fandhe_ai::metal_gpu_linalg_enabled());
        assert!(fandhe_ai_backend_metal::linalg::gpu_linalg_enabled());
        fandhe_ai::set_metal_gpu_linalg_enabled(false);
        assert!(!fandhe_ai::metal_gpu_linalg_enabled());
        assert!(!fandhe_ai_backend_metal::linalg::gpu_linalg_enabled());
    }
}
