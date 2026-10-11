//! イシュー #2676: Metal 線形代数（LU 系・cholesky・QR）のゲート・検証・CPU 比較テスト
//! （設計記録 `docs/backend-gpu-linalg-decision.md` §5）。
//!
//! 属性なしテスト（macOS の通常 `cargo test` で実行。デバイスに触れない）はゲート書き換えを
//! `static Mutex` で直列化し RAII で原状復帰する。`#[ignore]` テストは Apple Silicon 実機用:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test linalg_parity -- --ignored --nocapture
//! ```
//!
//! カーネル起動（新規 unsafe）はユーザー承認待ちのため、parity テストはゲート ON で
//! `Unsupported`（承認待ち）を受けたら承認待ちを明示して fail させる（green と取り違えない）。

#![cfg(target_os = "macos")]

use std::sync::Mutex;

use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};
use fandhe_ai_backend_metal::linalg::set_gpu_linalg_enabled;
use fandhe_ai_backend_metal::linalg_kernel::MetalLinalg;
use fandhe_ai_backend_metal::{MetalBackendOps, MetalContext};
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

fn s(x: &Tensor<f32>) -> Vec<f32> {
    x.contiguous().as_slice().expect("slice").to_vec()
}

fn spd(n: usize) -> Tensor<f32> {
    let mut d = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = if i == j {
                n as f32 + 1.0
            } else {
                1.0 / (1.0 + (i as f32 - j as f32).abs())
            };
        }
    }
    t(d, &[n, n])
}

fn ops() -> MetalBackendOps {
    MetalBackendOps::new()
}

#[test]
fn gate_off_returns_unsupported() {
    let _g = GateOn::new(false);
    let metal = ops();
    let a = t(vec![4.0, 1.0, 1.0, 3.0], &[2, 2]);
    let b = t(vec![1.0, 2.0], &[2, 1]);
    assert!(matches!(
        metal.linalg_inv(&a),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        metal.linalg_solve(&a, &b),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        metal.linalg_det(&a),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        metal.linalg_cholesky(&a),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        metal.linalg_qr(&a),
        Err(BackendError::Unsupported(_))
    ));
    // eigh は Metal では対象外（trait 既定のまま）。
    assert!(matches!(
        metal.linalg_eigh(&a),
        Err(BackendError::Unsupported(_))
    ));
}

#[test]
fn gate_on_validates_shapes_without_touching_device() {
    let _g = GateOn::new(true);
    let metal = ops();
    let rect = t(vec![0.0; 6], &[2, 3]);
    let rank1 = t(vec![0.0; 2], &[2]);
    let sq = t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let bad_b = t(vec![0.0; 3], &[3, 1]);
    for r in [
        metal.linalg_inv(&rank1).map(|_| ()),
        metal.linalg_det(&rank1).map(|_| ()),
        metal.linalg_cholesky(&rank1).map(|_| ()),
        metal.linalg_qr(&rank1).map(|_| ()),
    ] {
        assert!(matches!(r, Err(BackendError::ShapeMismatch(_))));
    }
    for r in [
        metal.linalg_inv(&rect).map(|_| ()),
        metal.linalg_det(&rect).map(|_| ()),
        metal.linalg_cholesky(&rect).map(|_| ()),
        metal.linalg_solve(&sq, &bad_b).map(|_| ()),
    ] {
        assert!(matches!(r, Err(BackendError::InvalidArgument(_))));
    }
    // 非正方の QR は形状エラーにならない（承認待ちの Unsupported まで進む）。
    assert!(matches!(
        metal.linalg_qr(&rect),
        Err(BackendError::Unsupported(_))
    ));
}

#[test]
fn gate_on_empty_inputs_match_cpu_shapes() {
    let _g = GateOn::new(true);
    let metal = ops();
    let cpu = CpuBackendOps::new();
    let z = t(vec![], &[0, 0]);
    assert_eq!(
        metal.linalg_inv(&z).expect("n=0").shape(),
        cpu.linalg_inv(&z).expect("cpu").shape()
    );
    assert_eq!(
        metal.linalg_cholesky(&z).expect("n=0").shape(),
        cpu.linalg_cholesky(&z).expect("cpu").shape()
    );
    assert_eq!(s(&metal.linalg_det(&z).expect("det")), vec![1.0]);
    assert_eq!(
        metal.linalg_det(&z).expect("det").shape(),
        cpu.linalg_det(&z).expect("cpu").shape()
    );
    let b0 = t(vec![], &[0, 4]);
    assert_eq!(
        metal.linalg_solve(&z, &b0).expect("solve").shape(),
        cpu.linalg_solve(&z, &b0).expect("cpu").shape()
    );
    for shape in [[0usize, 5], [5, 0], [0, 0]] {
        let a = t(vec![], &shape);
        let m = metal.linalg_qr(&a).expect("empty qr");
        let c = cpu.linalg_qr(&a).expect("cpu qr");
        assert_eq!(m.q.shape(), c.q.shape());
        assert_eq!(m.r.shape(), c.r.shape());
    }
}

#[test]
fn gate_on_non_empty_is_fail_closed_unsupported() {
    let _g = GateOn::new(true);
    let metal = ops();
    let a = t(vec![4.0, 1.0, 1.0, 3.0], &[2, 2]);
    assert!(matches!(
        metal.linalg_inv(&a),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        metal.linalg_cholesky(&a),
        Err(BackendError::Unsupported(_))
    ));
}

/// 実機: 4 カーネルの MSL がコンパイルされパイプラインが構築できること
/// （MSL 構文・属性の検証。起動は含まない）。
#[test]
#[ignore = "Apple Silicon 実機が必要"]
fn pipeline_compiles_on_device() {
    let ctx = MetalContext::new().expect("Metal デバイス");
    MetalLinalg::new(&ctx).expect("linalg.metal のコンパイルとパイプライン構築");
}

fn expect_launch_or_fail<T>(r: Result<T, BackendError>, what: &str) -> T {
    match r {
        Ok(v) => v,
        Err(BackendError::Unsupported(msg)) => panic!(
            "{what}: カーネル起動は新規 unsafe の承認待ち（Stage B 未実施）のため parity 未実測: {msg}"
        ),
        Err(e) => panic!("{what}: {e}"),
    }
}

#[test]
#[ignore = "Apple Silicon 実機が必要。カーネル起動は承認待ち（Stage B）"]
fn cholesky_matches_cpu_on_device() {
    let _g = GateOn::new(true);
    let cpu = CpuBackendOps::new();
    for n in [1usize, 8, 64] {
        let a = spd(n);
        let got = expect_launch_or_fail(ops().linalg_cholesky(&a), "cholesky");
        assert_parity(
            &format!("cholesky n={n}"),
            &s(&got),
            &s(&cpu.linalg_cholesky(&a).expect("cpu")),
        );
    }
}

#[test]
#[ignore = "Apple Silicon 実機が必要。カーネル起動は承認待ち（Stage B）"]
fn lu_family_matches_cpu_on_device() {
    let _g = GateOn::new(true);
    let cpu = CpuBackendOps::new();
    for n in [2usize, 8, 17] {
        let a = spd(n);
        let got = expect_launch_or_fail(ops().linalg_inv(&a), "inv");
        assert_parity(
            &format!("inv n={n}"),
            &s(&got),
            &s(&cpu.linalg_inv(&a).expect("cpu")),
        );
        let det = expect_launch_or_fail(ops().linalg_det(&a), "det");
        assert_parity(
            &format!("det n={n}"),
            &s(&det),
            &s(&cpu.linalg_det(&a).expect("cpu")),
        );
    }
}

#[test]
#[ignore = "Apple Silicon 実機が必要。カーネル起動は承認待ち（Stage B）"]
fn qr_matches_cpu_on_device() {
    let _g = GateOn::new(true);
    let cpu = CpuBackendOps::new();
    for (m, n) in [(8usize, 3usize), (3, 8), (17, 17)] {
        let d: Vec<f32> = (0..m * n)
            .map(|i| (((i * 37 + 11) % 23) as f32 - 11.0) / 7.0)
            .collect();
        let a = t(d, &[m, n]);
        let got = expect_launch_or_fail(ops().linalg_qr(&a), "qr");
        let reference = cpu.linalg_qr(&a).expect("cpu");
        assert_parity(&format!("qr R {m}x{n}"), &s(&got.r), &s(&reference.r));
        assert_parity(&format!("qr Q {m}x{n}"), &s(&got.q), &s(&reference.q));
    }
}
