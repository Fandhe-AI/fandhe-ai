//! `CpuBackendOps::gemm_device` の受け入れ基準対応テスト（イシュー #3250 P4）。
//!
//! upload 済みの入力に対する行列積が、ホスト入力の `gemm` と bit 完全一致すること
//! （同一カーネル。REQ-2 の複合判定ではなく bit 比較）と、不正入力を型付きエラーで
//! 拒否することを確かめる。`facade::Tape::matmul_resident` の CPU 経路の土台。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{CpuBackendOps, CpuMemory};
use fandhe_ai_tensor_core::buffer::MemoryOps;
use fandhe_ai_tensor_core::{BackendError, BackendOps, Tensor};

fn t(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(Xorshift64Star::new(seed).fill_vec(n), shape).unwrap()
}

#[test]
fn gemm_device_matches_gemm_bitwise() {
    let ops = CpuBackendOps;
    let mem = CpuMemory::new();
    for &(m, k, n) in &[
        (1, 1, 1),
        (1, 5, 3),
        (7, 13, 5),
        (64, 784, 256),
        (256, 256, 256),
        (0, 4, 3),
        (3, 0, 4),
        (3, 4, 0),
    ] {
        let a = t(1, &[m, k]);
        let b = t(2, &[k, n]);
        let expect = ops.gemm(&a, &b).unwrap();
        let ab = mem.upload(&a).unwrap();
        let bb = mem.upload(&b).unwrap();
        let out = ops.gemm_device(&ab, &bb).unwrap();
        assert_eq!(out.shape(), &[m, n]);
        let got = mem.download(&out).unwrap();
        let (e, g) = (expect.as_slice().unwrap(), got.as_slice().unwrap());
        assert_eq!(e.len(), g.len());
        for (x, y) in e.iter().zip(g) {
            assert_eq!(x.to_bits(), y.to_bits(), "shape {m}x{k}x{n}");
        }
    }
}

#[test]
fn gemm_device_rejects_bad_shapes() {
    let ops = CpuBackendOps;
    let mem = CpuMemory::new();
    let a = mem.upload(&t(1, &[2, 3])).unwrap();
    let b = mem.upload(&t(2, &[4, 5])).unwrap();
    assert!(matches!(
        ops.gemm_device(&a, &b),
        Err(BackendError::ShapeMismatch(_))
    ));
    let v = mem.upload(&t(3, &[3])).unwrap();
    assert!(matches!(
        ops.gemm_device(&v, &b),
        Err(BackendError::ShapeMismatch(_))
    ));
}
