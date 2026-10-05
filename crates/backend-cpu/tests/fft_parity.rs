//! `CpuBackendOps::fft_rfft`／`fft_irfft` の直接呼び出し統合テスト
//! （イシュー #2631・`docs/autodiff-fft-ops-decision.md`）。
//!
//! 共有カーネル（`fandhe_ai_tensor_core::fft`）の単体テストは tensor-core
//! 側にあり、本ファイルは `BackendOps` 経由の到達性・解析解との REQ-2
//! 複合判定（`assert_parity`）・決定性（同一入力 2 回の bit 一致）・不正
//! 引数の型付きエラー（fail-closed）を固定する。

use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, FftNorm, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    tensor.host_slice().to_vec()
}

#[test]
fn rfft_delta_is_constant() {
    let ops = CpuBackendOps::new();
    let y = ops
        .fft_rfft(&t(vec![1.0, 0.0, 0.0, 0.0], &[4]), 4, 0, FftNorm::Backward)
        .unwrap();
    assert_eq!(y.shape(), &[3, 2]);
    assert_parity("rfft_delta", &dense(&y), &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn rfft_known_solution_odd_n_ortho() {
    let ops = CpuBackendOps::new();
    let y = ops
        .fft_rfft(&t(vec![1.0, 2.0, 3.0], &[3]), 3, 0, FftNorm::Ortho)
        .unwrap();
    let s = 1.0 / 3.0f32.sqrt();
    assert_parity(
        "rfft_odd_ortho",
        &dense(&y),
        &[6.0 * s, 0.0, -1.5 * s, 0.866_025_4 * s],
    );
}

#[test]
fn irfft_roundtrip_and_default_shape() {
    let ops = CpuBackendOps::new();
    let x = t(vec![0.5, -1.0, 2.0, 0.25, 3.0, 1.5], &[2, 3]);
    let y = ops.fft_rfft(&x, 3, 1, FftNorm::Forward).unwrap();
    assert_eq!(y.shape(), &[2, 2, 2]);
    let z = ops.fft_irfft(&y, 3, 1, FftNorm::Forward).unwrap();
    assert_eq!(z.shape(), &[2, 3]);
    assert_parity("roundtrip", &dense(&z), &dense(&x));
}

#[test]
fn deterministic_across_runs() {
    let ops = CpuBackendOps::new();
    let x = t((0..14).map(|i| (i as f32 * 0.37).sin()).collect(), &[2, 7]);
    let a = ops.fft_rfft(&x, 7, 1, FftNorm::Backward).unwrap();
    let b = ops.fft_rfft(&x, 7, 1, FftNorm::Backward).unwrap();
    assert!(
        dense(&a)
            .iter()
            .zip(dense(&b))
            .all(|(p, q)| p.to_bits() == q.to_bits())
    );
}

#[test]
fn invalid_arguments_are_typed_errors() {
    let ops = CpuBackendOps::new();
    let x = t(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    // n = 0
    assert!(matches!(
        ops.fft_rfft(&x, 0, 0, FftNorm::Backward),
        Err(BackendError::InvalidArgument(_))
    ));
    // dim 範囲外
    assert!(matches!(
        ops.fft_rfft(&x, 4, 1, FftNorm::Backward),
        Err(BackendError::InvalidArgument(_))
    ));
    // 末尾次元 != 2
    let c = t(vec![1.0; 6], &[2, 3]);
    assert!(matches!(
        ops.fft_irfft(&c, 2, 0, FftNorm::Backward),
        Err(BackendError::ShapeMismatch(_))
    ));
    // 確保不能な巨大 n は確保前に拒否
    assert!(matches!(
        ops.fft_rfft(&x, usize::MAX / 2, 0, FftNorm::Backward),
        Err(BackendError::ShapeMismatch(_))
    ));
}

// --- fft／ifft（c2c。イシュー #2632） ---

#[test]
fn fft_delta_and_odd_n_ortho_known_solutions() {
    let ops = CpuBackendOps::new();
    let y = ops
        .fft_fft(
            &t(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]),
            4,
            0,
            FftNorm::Backward,
        )
        .unwrap();
    assert_eq!(y.shape(), &[4, 2]);
    assert_parity(
        "fft_delta",
        &dense(&y),
        &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    );
    let y = ops
        .fft_fft(
            &t(vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0], &[3, 2]),
            3,
            0,
            FftNorm::Ortho,
        )
        .unwrap();
    let s = 1.0 / 3.0f32.sqrt();
    assert_parity(
        "fft_odd_ortho",
        &dense(&y),
        &[
            6.0 * s,
            0.0,
            -1.5 * s,
            0.866_025_4 * s,
            -1.5 * s,
            -0.866_025_4 * s,
        ],
    );
}

#[test]
fn fft_ifft_roundtrip_deterministic_and_padded_shape() {
    let ops = CpuBackendOps::new();
    let x = t(
        (0..24).map(|i| (i as f32 * 0.41).sin()).collect(),
        &[2, 6, 2],
    );
    let y = ops.fft_fft(&x, 6, 1, FftNorm::Forward).unwrap();
    let z = ops.fft_ifft(&y, 6, 1, FftNorm::Forward).unwrap();
    assert_parity("c2c_roundtrip", &dense(&z), &dense(&x));
    let padded = ops.fft_fft(&x, 8, 1, FftNorm::Backward).unwrap();
    assert_eq!(padded.shape(), &[2, 8, 2]);
    let a = ops.fft_ifft(&x, 6, 1, FftNorm::Ortho).unwrap();
    let b = ops.fft_ifft(&x, 6, 1, FftNorm::Ortho).unwrap();
    assert!(
        dense(&a)
            .iter()
            .zip(dense(&b))
            .all(|(p, q)| p.to_bits() == q.to_bits())
    );
}

#[test]
fn c2c_invalid_arguments_are_typed_errors() {
    let ops = CpuBackendOps::new();
    let x = t(vec![1.0; 8], &[4, 2]);
    for f in [CpuBackendOps::fft_fft, CpuBackendOps::fft_ifft] {
        assert!(matches!(
            f(&ops, &x, 0, 0, FftNorm::Backward),
            Err(BackendError::InvalidArgument(_))
        ));
        assert!(matches!(
            f(&ops, &x, 4, 1, FftNorm::Backward),
            Err(BackendError::InvalidArgument(_))
        ));
        let bad = t(vec![1.0; 6], &[2, 3]);
        assert!(matches!(
            f(&ops, &bad, 2, 0, FftNorm::Backward),
            Err(BackendError::ShapeMismatch(_))
        ));
        assert!(matches!(
            f(&ops, &x, usize::MAX / 2, 0, FftNorm::Backward),
            Err(BackendError::ShapeMismatch(_))
        ));
    }
}
