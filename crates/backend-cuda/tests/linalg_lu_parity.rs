//! イシュー #2674: CUDA LU 系線形代数（`linalg_inv`／`linalg_solve`／`linalg_det`）の
//! ゲート・検証・CPU 比較テスト。
//!
//! 属性なしテスト（通常 CI で実行。CUDA 非搭載でも panic しない）はゲート書き換えを
//! `static Mutex` で直列化し RAII で原状復帰する。`#[ignore]` テストは DGX Spark GB10 実機用:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test linalg_lu_parity -- --ignored --nocapture
//! ```
//!
//! カーネル起動（新規 unsafe）はユーザー承認待ちのため、parity テストはゲート ON で
//! `Unsupported`（承認待ち）を受けたら承認待ちを明示して fail させる（green と取り違えない）。

use std::sync::Mutex;

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::linalg::{CudaLinalg, set_gpu_linalg_enabled};
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

static GATE_LOCK: Mutex<()> = Mutex::new(());

/// ゲートを ON にし、drop で OFF へ戻す。
struct GateOn<'a>(#[allow(dead_code)] std::sync::MutexGuard<'a, ()>);

impl<'a> GateOn<'a> {
    fn new(on: bool) -> Self {
        let g = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_gpu_linalg_enabled(on);
        Self(g)
    }
}

impl Drop for GateOn<'_> {
    fn drop(&mut self) {
        set_gpu_linalg_enabled(false);
    }
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("valid tensor")
}

#[test]
fn gate_off_returns_unsupported() {
    let _g = GateOn::new(false);
    let cuda = CudaBackendOps::new(0);
    let a = t(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]);
    let b = t(vec![1.0, 2.0], &[2, 1]);
    assert!(matches!(
        cuda.linalg_inv(&a),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        cuda.linalg_solve(&a, &b),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        cuda.linalg_det(&a),
        Err(BackendError::Unsupported(_))
    ));
}

#[test]
fn gate_on_validates_shapes_without_touching_device() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let rect = t(vec![0.0; 6], &[2, 3]);
    let rank1 = t(vec![0.0; 2], &[2]);
    let a = t(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]);
    let b_bad = t(vec![0.0; 3], &[3, 1]);
    let b_rank1 = t(vec![0.0; 2], &[2]);
    assert!(matches!(
        cuda.linalg_inv(&rect),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        cuda.linalg_det(&rect),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        cuda.linalg_inv(&rank1),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        cuda.linalg_solve(&a, &b_bad),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        cuda.linalg_solve(&a, &b_rank1),
        Err(BackendError::ShapeMismatch(_))
    ));
}

#[test]
fn gate_on_empty_matrix_matches_cpu() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    let a = t(vec![], &[0, 0]);
    let b = t(vec![], &[0, 3]);
    let inv = cuda.linalg_inv(&a).expect("n=0 inv");
    assert_eq!(inv.shape(), cpu.linalg_inv(&a).expect("cpu").shape());
    let sol = cuda.linalg_solve(&a, &b).expect("n=0 solve");
    assert_eq!(sol.shape(), cpu.linalg_solve(&a, &b).expect("cpu").shape());
    let det = cuda.linalg_det(&a).expect("n=0 det");
    assert_eq!(det.shape(), cpu.linalg_det(&a).expect("cpu").shape());
    assert_eq!(det.contiguous().as_slice().expect("slice"), &[1.0f32]);
}

/// CUDA 非搭載環境でも panic せず型付きエラーまたは承認待ち `Unsupported` を返す。
#[test]
fn gate_on_non_empty_does_not_panic() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let a = t(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]);
    let r = cuda.linalg_det(&a);
    assert!(matches!(
        r,
        Err(BackendError::Unsupported(_) | BackendError::CudaUnavailable(_)) | Ok(_)
    ));
}

/// 実機: 2 カーネルが GB10 で NVRTC コンパイル・ロードできる（Stage A で検証可能な項目）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要"]
fn nvrtc_compiles_and_loads_lu_kernels_on_device() {
    let device = CudaDevice::new(0).expect("CUDA device 0");
    let lin = CudaLinalg::new(&device).expect("NVRTC compile/load");
    assert_eq!(lin.ordinal(), 0);
}

/// 実機: ゲート ON の det を CPU と統一複合判定で比較する。カーネル起動が承認待ちの間は
/// 承認待ちを明示して fail する（green と取り違えない）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。カーネル起動は新規 unsafe の承認待ち（#2674）"]
fn lu_det_matches_cpu_on_device() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    let a = t(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]);
    let got = match cuda.linalg_det(&a) {
        Ok(v) => v,
        Err(BackendError::Unsupported(m)) => {
            panic!("カーネル起動の unsafe 承認待ち（#2674）のため未実装: {m}")
        }
        Err(e) => panic!("linalg_det failed: {e:?}"),
    };
    let want = cpu.linalg_det(&a).expect("cpu det");
    fandhe_ai_backend_cpu::assert_parity(
        "linalg_det 2x2",
        got.contiguous().as_slice().expect("slice"),
        want.contiguous().as_slice().expect("slice"),
    );
}
