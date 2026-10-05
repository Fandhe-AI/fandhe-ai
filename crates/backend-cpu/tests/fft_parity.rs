//! `CpuBackendOps::fft_rfft`／`fft_irfft`／`fft_fft`／`fft_ifft`／`fft_stft`／
//! `fft_istft` の直接呼び出し統合テスト
//! （イシュー #2631・#2632・#2633・`docs/autodiff-fft-ops-decision.md`）。
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

// ---- stft / istft（イシュー #2633） ----

use fandhe_ai_tensor_core::StftPadMode;
use fandhe_ai_tensor_core::fft::{IstftParams, StftParams, stft_window};

fn sp(n: usize, hop: Option<usize>, center: bool, one: bool) -> StftParams {
    StftParams::new(n, hop, center, StftPadMode::Reflect, false, one).unwrap()
}

#[test]
fn stft_hop_equals_nfft_matches_blockwise_rfft() {
    let ops = CpuBackendOps::new();
    let x: Vec<f32> = (0..8).map(|i| (i as f32 * 0.7).cos()).collect();
    let w = t(vec![1.0; 4], &[4]);
    let y = ops
        .fft_stft(&t(x.clone(), &[8]), &w, &sp(4, Some(4), false, true))
        .unwrap();
    assert_eq!(y.shape(), &[3, 2, 2]);
    for blk in 0..2 {
        let r = ops
            .fft_rfft(
                &t(x[blk * 4..blk * 4 + 4].to_vec(), &[4]),
                4,
                0,
                FftNorm::Backward,
            )
            .unwrap();
        let r = dense(&r);
        let y = dense(&y);
        for k in 0..3 {
            assert_eq!(y[(k * 2 + blk) * 2], r[k * 2]);
            assert_eq!(y[(k * 2 + blk) * 2 + 1], r[k * 2 + 1]);
        }
    }
}

#[test]
fn stft_istft_roundtrip_deterministic() {
    let ops = CpuBackendOps::new();
    let x = t((0..40).map(|i| (i as f32 * 0.37).sin()).collect(), &[40]);
    let win: Vec<f32> = (0..8)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 8.0).cos()) as f32)
        .collect();
    let w = t(win, &[8]);
    let s = ops.fft_stft(&x, &w, &sp(8, Some(2), true, true)).unwrap();
    let s2 = ops.fft_stft(&x, &w, &sp(8, Some(2), true, true)).unwrap();
    assert!(
        dense(&s)
            .iter()
            .zip(dense(&s2))
            .all(|(p, q)| p.to_bits() == q.to_bits())
    );
    let ip = IstftParams::new(8, Some(2), None, true, false, None, Some(40)).unwrap();
    let y = ops.fft_istft(&s, &w, &ip).unwrap();
    assert_eq!(y.shape(), &[40]);
    assert_parity("stft_roundtrip", &dense(&y), &dense(&x));
}

#[test]
fn stft_invalid_arguments_are_typed_errors() {
    let ops = CpuBackendOps::new();
    let w = t(stft_window(None, None, 8).unwrap(), &[8]);
    // 反射パディング幅が信号長以上。
    assert!(matches!(
        ops.fft_stft(&t(vec![1.0; 4], &[4]), &w, &sp(8, Some(2), true, true)),
        Err(BackendError::InvalidArgument(_))
    ));
    // rank 違い。
    assert!(matches!(
        ops.fft_stft(
            &t(vec![1.0; 8], &[1, 2, 4]),
            &w,
            &sp(8, Some(2), true, true)
        ),
        Err(BackendError::ShapeMismatch(_))
    ));
    // 窓長が n_fft と一致しない。
    assert!(matches!(
        ops.fft_stft(
            &t(vec![1.0; 32], &[32]),
            &t(vec![1.0; 7], &[7]),
            &sp(8, Some(2), true, true)
        ),
        Err(BackendError::InvalidArgument(_))
    ));
    // NOLA 違反（Hann・center=false）はバックエンドでも拒否される。
    let hann = t(
        (0..8)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 8.0).cos()) as f32)
            .collect(),
        &[8],
    );
    let ip = IstftParams::new(8, Some(2), None, false, false, None, None).unwrap();
    assert!(matches!(
        ops.fft_istft(&t(vec![0.0; 5 * 17 * 2], &[5, 17, 2]), &hann, &ip),
        Err(BackendError::InvalidArgument(_))
    ));
}
