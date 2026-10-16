//! イシュー #2675: CUDA 分解系線形代数（`linalg_cholesky`／`linalg_qr`／`linalg_eigh`）の
//! ゲート・検証・CPU 比較テスト（設計記録 `docs/backend-gpu-linalg-decision.md` §5）。
//!
//! 属性なしテスト（通常 CI で実行。CUDA 非搭載でも panic しない）はゲート書き換えを
//! `static Mutex` で直列化し RAII で原状復帰する。`#[ignore]` テストは DGX Spark GB10 実機用:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test linalg_factor_parity -- --ignored --nocapture
//! ```
//!
//! カーネル起動はイシュー #3136 で実装済み（実機での一致確認は #3138）。ゲート ON の
//! 非空入力は `Unsupported` を返さない（CUDA 不在なら `CudaUnavailable`）。

use std::sync::Mutex;

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::linalg::{CudaLinalgFactor, set_gpu_linalg_enabled};
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

static GATE_LOCK: Mutex<()> = Mutex::new(());

/// ゲートを設定し、drop で OFF へ戻す。
struct GateOn<'a> {
    _guard: std::sync::MutexGuard<'a, ()>,
}

impl<'a> GateOn<'a> {
    fn new(on: bool) -> Self {
        let g = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_gpu_linalg_enabled(on);
        Self { _guard: g }
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

fn spd(n: usize, scale: f32) -> Tensor<f32> {
    // 対角優位の対称正定値行列（κ が小さい）。
    let mut d = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = if i == j {
                (n as f32 + 1.0) * scale
            } else {
                scale / (1.0 + (i as f32 - j as f32).abs())
            };
        }
    }
    t(d, &[n, n])
}

fn dense(m: usize, n: usize, scale: f32) -> Tensor<f32> {
    let d: Vec<f32> = (0..m * n)
        .map(|i| (((i * 37 + 11) % 23) as f32 - 11.0) / 7.0 * scale)
        .collect();
    t(d, &[m, n])
}

fn s(x: &Tensor<f32>) -> Vec<f32> {
    x.contiguous().as_slice().expect("slice").to_vec()
}

#[test]
fn gate_off_returns_unsupported() {
    let _g = GateOn::new(false);
    let cuda = CudaBackendOps::new(0);
    let a = t(vec![4.0, 1.0, 1.0, 3.0], &[2, 2]);
    assert!(matches!(
        cuda.linalg_cholesky(&a),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        cuda.linalg_qr(&a),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        cuda.linalg_eigh(&a),
        Err(BackendError::Unsupported(_))
    ));
}

#[test]
fn gate_on_validates_shapes_without_touching_device() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let rect = t(vec![0.0; 6], &[2, 3]);
    let rank1 = t(vec![0.0; 2], &[2]);
    assert!(matches!(
        cuda.linalg_cholesky(&rank1),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        cuda.linalg_qr(&rank1),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        cuda.linalg_eigh(&rank1),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        cuda.linalg_cholesky(&rect),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        cuda.linalg_eigh(&rect),
        Err(BackendError::InvalidArgument(_))
    ));
    // 非正方の QR は形状エラーにならない（CUDA 不在まで進む。`Unsupported` は不合格）。
    assert!(matches!(
        cuda.linalg_qr(&rect),
        Err(BackendError::CudaUnavailable(_)) | Ok(_)
    ));
}

#[test]
fn gate_on_empty_inputs_match_cpu_shapes() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    let z = t(vec![], &[0, 0]);
    assert_eq!(
        cuda.linalg_cholesky(&z).expect("n=0").shape(),
        cpu.linalg_cholesky(&z).expect("cpu").shape()
    );
    let ez = cuda.linalg_eigh(&z).expect("n=0 eigh");
    let cz = cpu.linalg_eigh(&z).expect("cpu eigh");
    assert_eq!(ez.eigenvalues.shape(), cz.eigenvalues.shape());
    assert_eq!(ez.eigenvectors.shape(), cz.eigenvectors.shape());
    for shape in [[0usize, 5], [5, 0], [0, 0]] {
        let a = t(vec![], &shape);
        let g = cuda.linalg_qr(&a).expect("empty qr");
        let c = cpu.linalg_qr(&a).expect("cpu qr");
        assert_eq!(g.q.shape(), c.q.shape(), "q {shape:?}");
        assert_eq!(g.r.shape(), c.r.shape(), "r {shape:?}");
    }
}

/// CUDA 非搭載環境でも panic せず型付きエラー（`CudaUnavailable`）を返す。
/// ゲート ON の非空入力が `Unsupported` に落ちない（実起動へ結線済み）ことも固定する。
#[test]
fn gate_on_non_empty_does_not_panic() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let a = t(vec![4.0, 1.0, 1.0, 3.0], &[2, 2]);
    assert!(matches!(
        cuda.linalg_cholesky(&a),
        Err(BackendError::CudaUnavailable(_)) | Ok(_)
    ));
    assert!(matches!(
        cuda.linalg_qr(&a),
        Err(BackendError::CudaUnavailable(_)) | Ok(_)
    ));
    assert!(matches!(
        cuda.linalg_eigh(&a),
        Err(BackendError::CudaUnavailable(_)) | Ok(_)
    ));
}

/// 実機: 3 カーネルが GB10 で NVRTC コンパイル・ロードできる（Stage A で検証可能な項目）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要（実測は #3138）"]
fn nvrtc_compiles_and_loads_factor_kernels_on_device() {
    let device = CudaDevice::new(0).expect("CUDA device 0");
    let lin = CudaLinalgFactor::new(&device).expect("NVRTC compile/load");
    assert_eq!(lin.ordinal(), 0);
}

fn ok<T>(r: Result<T, BackendError>, op: &str) -> T {
    r.unwrap_or_else(|e| panic!("{op} failed: {e:?}"))
}

/// 既知の条件数（約 1e3）を持つ SPD。対角 `d_i = 1e3^(i/(n-1))` に決定的な Givens 回転を
/// f64 で施し、対称化してから f32 化する（設計記録 §5）。
fn spd_known_kappa(n: usize) -> Tensor<f32> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        let e = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.0
        };
        a[i * n + i] = 1e3f64.powf(e);
    }
    for p in 0..n {
        for q in (p + 1)..n {
            let th = 0.3 + 0.01 * (p * n + q) as f64;
            let (c, sn) = (th.cos(), th.sin());
            // A <- G A Gᵀ（G は (p,q) 平面の回転）。
            for k in 0..n {
                let (x, y) = (a[p * n + k], a[q * n + k]);
                a[p * n + k] = c * x - sn * y;
                a[q * n + k] = sn * x + c * y;
            }
            for k in 0..n {
                let (x, y) = (a[k * n + p], a[k * n + q]);
                a[k * n + p] = c * x - sn * y;
                a[k * n + q] = sn * x + c * y;
            }
        }
    }
    let mut d = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = (0.5 * (a[i * n + j] + a[j * n + i])) as f32;
        }
    }
    t(d, &[n, n])
}

/// 実機: cholesky を CPU と統一複合判定で比較する（SPD・大小スケール）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要（実測は #3138）"]
fn cholesky_matches_cpu_on_device() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    for &n in &[1usize, 2, 3, 8, 33, 64] {
        for &scale in &[1.0f32, 1e-3, 1e3, 1e20, 1e-20] {
            let a = spd(n, scale);
            let got = ok(cuda.linalg_cholesky(&a), "cholesky");
            let want = cpu.linalg_cholesky(&a).expect("cpu cholesky");
            fandhe_ai_backend_cpu::assert_parity(
                &format!("cholesky n={n} scale={scale}"),
                &s(&got),
                &s(&want),
            );
        }
        let a = spd_known_kappa(n);
        let got = ok(cuda.linalg_cholesky(&a), "cholesky kappa");
        let want = cpu.linalg_cholesky(&a).expect("cpu cholesky kappa");
        fandhe_ai_backend_cpu::assert_parity(
            &format!("cholesky known-kappa n={n}"),
            &s(&got),
            &s(&want),
        );
    }
}

/// 実機: QR（縦長・横長・rank 落ち・大小スケール）を符号規約込みで比較する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要（実測は #3138）"]
fn qr_matches_cpu_on_device() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    let mut cases: Vec<(String, Tensor<f32>)> = Vec::new();
    for &(m, n) in &[(64usize, 8usize), (8, 64), (5, 5), (33, 17)] {
        for &scale in &[1.0f32, 1e20, 1e-20] {
            cases.push((format!("{m}x{n} scale={scale}"), dense(m, n, scale)));
        }
    }
    // rank 落ち（ゼロ列を含む）。
    let mut zc = dense(6, 4, 1.0);
    let mut data = s(&zc);
    for r in 0..6 {
        data[r * 4 + 1] = 0.0;
    }
    zc = t(data, &[6, 4]);
    cases.push(("zero column".into(), zc));
    cases.push(("1e20 * I2".into(), t(vec![1e20, 0.0, 0.0, 1e20], &[2, 2])));
    for (name, a) in cases {
        let got = ok(cuda.linalg_qr(&a), "qr");
        let want = cpu.linalg_qr(&a).expect("cpu qr");
        fandhe_ai_backend_cpu::assert_parity(&format!("qr.q {name}"), &s(&got.q), &s(&want.q));
        fandhe_ai_backend_cpu::assert_parity(&format!("qr.r {name}"), &s(&got.r), &s(&want.r));
    }
}

/// 実機: eigh（重複固有値を含む）を固有値の並び・固有ベクトルの符号込みで比較する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要（実測は #3138）"]
fn eigh_matches_cpu_on_device() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    let mut cases: Vec<(String, Tensor<f32>)> = Vec::new();
    for &n in &[1usize, 2, 3, 8, 33] {
        cases.push((format!("spd n={n}"), spd(n, 1.0)));
    }
    cases.push((
        "repeated eigenvalues".into(),
        t(vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0], &[3, 3]),
    ));
    cases.push(("1e20 * I2".into(), t(vec![1e20, 0.0, 0.0, 1e20], &[2, 2])));
    for &scale in &[1e20f32, 1e-20] {
        cases.push((format!("spd n=8 scale={scale}"), spd(8, scale)));
    }
    for &n in &[2usize, 8, 33] {
        cases.push((format!("known-kappa n={n}"), spd_known_kappa(n)));
    }
    for (name, a) in cases {
        let got = ok(cuda.linalg_eigh(&a), "eigh");
        let want = cpu.linalg_eigh(&a).expect("cpu eigh");
        fandhe_ai_backend_cpu::assert_parity(
            &format!("eigh.values {name}"),
            &s(&got.eigenvalues),
            &s(&want.eigenvalues),
        );
        fandhe_ai_backend_cpu::assert_parity(
            &format!("eigh.vectors {name}"),
            &s(&got.eigenvectors),
            &s(&want.eigenvectors),
        );
    }
}

/// 実機: 非正定値の cholesky は CUDA・CPU ともに `InvalidArgument`（A08・不変条件 (f)）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要（実測は #3138）"]
fn cholesky_non_spd_is_invalid_argument_on_device() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    for a in [t(vec![1.0, 2.0, 2.0, 1.0], &[2, 2]), t(vec![0.0], &[1, 1])] {
        assert!(matches!(
            cuda.linalg_cholesky(&a),
            Err(BackendError::InvalidArgument(_))
        ));
        assert!(matches!(
            cpu.linalg_cholesky(&a),
            Err(BackendError::InvalidArgument(_))
        ));
    }
}

/// 実機: NaN を含む対称行列の eigh は巡回 Jacobi が収束せず CUDA・CPU ともに
/// `InvalidArgument`。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要（実測は #3138）"]
fn eigh_non_convergence_is_invalid_argument_on_device() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    let a = t(vec![1.0, f32::NAN, f32::NAN, 1.0], &[2, 2]);
    assert!(matches!(
        cuda.linalg_eigh(&a),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        cpu.linalg_eigh(&a),
        Err(BackendError::InvalidArgument(_))
    ));
}
