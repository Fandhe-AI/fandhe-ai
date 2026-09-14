//! イシュー #1757: `BackendOps::interpolate`（`torch.nn.functional.
//! interpolate(mode='nearest')` 相当）の CPU-CUDA 数値一致検証。
//!
//! `gather_scatter_parity.rs`（#1777）・`constant_pad_parity.rs`
//! （#1756 相当テンプレート）と同じ構成方針を踏襲する: 環境適応
//! スモーク（属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `BackendError::CudaUnavailable` を確認して panic しないことのみ
//! 検証。デバイス初期化より前に返る shape 検査経路は GPU 有無に依らず
//! 検証する）と、実機必須の形状網羅（`#[ignore]`。DGX Spark GB10 等）
//! を分離する。
//!
//! **契約は bit 完全一致**（interpolate は算術を含まない純粋な
//! コピー演算のため。`.claude/rules/coding-rust.md` 数値契約節参照）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test interpolate_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, InterpolateMode, Tensor};

fn assert_interpolate_parity(seed: u64, in_shape: &[usize], size: &[usize]) {
    let numel_in: usize = in_shape.iter().product();

    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let input =
        Tensor::new(Xorshift64Star::new(seed).fill_vec(numel_in), in_shape).expect("valid tensor");

    let cpu_out = cpu
        .interpolate(&input, size, InterpolateMode::Nearest)
        .expect("cpu interpolate always succeeds for valid input");
    let cuda_out = cuda
        .interpolate(&input, size, InterpolateMode::Nearest)
        .expect("cuda interpolate must succeed on CUDA-equipped test runner");

    assert_eq!(cpu_out.shape(), cuda_out.shape());
    let cpu_slice = cpu_out.as_slice().expect("contiguous");
    let cuda_slice = cuda_out.as_slice().expect("contiguous");
    assert_eq!(
        cpu_slice, cuda_slice,
        "interpolate(in_shape={in_shape:?}, size={size:?}): CPU と CUDA は bit 完全一致のはず"
    );

    // run-to-run で bit 同一（決定的カーネル）。
    let cuda_out2 = cuda
        .interpolate(&input, size, InterpolateMode::Nearest)
        .expect("cuda interpolate rerun");
    assert_eq!(
        cuda_out2.as_slice().expect("contiguous"),
        cuda_slice,
        "interpolate: run-to-run で bit 同一のはず"
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA 不在なら
/// `BackendError::CudaUnavailable` を確認して早期 return する
/// （`gather_scatter_parity_smoke_env_adaptive` と同じ分岐パターン）。
/// デバイス初期化より前に返る shape 検査経路（`interpolate_out_shape`
/// の再検査）は GPU 有無に依らず検証する。
#[test]
fn interpolate_parity_smoke_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();

    let input = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).expect("valid tensor");

    match cuda.interpolate(&input, &[6], InterpolateMode::Nearest) {
        Ok(_) => {
            assert_interpolate_parity(9201, &[3], &[6]);
            assert_interpolate_parity(9202, &[8], &[3]); // 非整数比ダウンサンプル
            assert_interpolate_parity(9203, &[2, 4], &[4]); // 先頭 batch 軸付き
            assert_interpolate_parity(9204, &[3, 4], &[3, 4]); // 恒等サイズ

            // shape 不一致（rank 超過）は `BackendError::ShapeMismatch`
            // を返す（実装側の再検査。`.claude/rules/security.md`
            // A08）。
            let err = cuda
                .interpolate(&input, &[2, 2], InterpolateMode::Nearest)
                .expect_err("size.len() > rank must be rejected");
            assert!(matches!(err, BackendError::ShapeMismatch(_)));

            // 空間軸 0 は `BackendError::ShapeMismatch` を返す
            // （ゼロ除算回避の事前拒否）。
            let err = cuda
                .interpolate(&input, &[0], InterpolateMode::Nearest)
                .expect_err("zero output spatial axis must be rejected");
            assert!(matches!(err, BackendError::ShapeMismatch(_)));
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");

            // shape 検査はデバイス初期化より前に走るため CUDA 非搭載
            // 環境でも検証できる（CPU の返す値と一致することも確認）。
            let err = cuda
                .interpolate(&input, &[2, 2], InterpolateMode::Nearest)
                .expect_err("size.len() > rank must be rejected even without CUDA");
            assert!(matches!(err, BackendError::ShapeMismatch(_)));

            let err = cuda
                .interpolate(&input, &[0], InterpolateMode::Nearest)
                .expect_err("zero output spatial axis must be rejected even without CUDA");
            assert!(matches!(err, BackendError::ShapeMismatch(_)));

            let cpu_err = cpu
                .interpolate(&input, &[0], InterpolateMode::Nearest)
                .expect_err("cpu must also reject zero output spatial axis");
            assert!(matches!(cpu_err, BackendError::ShapeMismatch(_)));
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::interpolate: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn interpolate_matches_cpu_across_shapes() {
    let cases: &[(&[usize], &[usize])] = &[
        (&[3], &[6]),          // 1-D 整数倍アップサンプル
        (&[8], &[3]),          // 1-D 非整数比ダウンサンプル
        (&[4], &[4]),          // 恒等サイズ
        (&[2, 3], &[7]),       // 先頭 batch 軸付き（末尾 1 軸のみ空間軸）
        (&[2, 4, 4], &[9, 9]), // 2 軸空間（先頭 batch 軸付き）
        (&[1 << 10, 3], &[7]), // ブロック境界をまたぐ大きさ
        (&[5], &[1]),          // 極端ダウンサンプル（全出力が同一入力）
    ];
    let mut seed = 20_000u64;
    for &(in_shape, size) in cases {
        seed += 7;
        assert_interpolate_parity(seed, in_shape, size);
    }

    // 非 contiguous な input（transpose view）を渡しても contiguous 化
    // 後に一致する。
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let base = Tensor::new((0..12).map(|v| v as f32).collect(), &[3, 4]).expect("valid tensor");
    let transposed = base.transpose(0, 1).expect("valid transpose"); // shape [4, 3]
    let cpu_out = cpu
        .interpolate(&transposed, &[6], InterpolateMode::Nearest)
        .expect("cpu interpolate");
    let cuda_out = cuda
        .interpolate(&transposed, &[6], InterpolateMode::Nearest)
        .expect("cuda interpolate");
    assert_eq!(
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous")
    );

    // NaN／±inf の通過（算術を含まないコピー演算のため bit 単位で
    // そのまま伝播する）。
    let input_special = Tensor::new(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0], &[4])
        .expect("valid tensor");
    let cpu_special = cpu
        .interpolate(&input_special, &[8], InterpolateMode::Nearest)
        .expect("cpu interpolate");
    let cuda_special = cuda
        .interpolate(&input_special, &[8], InterpolateMode::Nearest)
        .expect("cuda interpolate");
    let cpu_slice = cpu_special.as_slice().expect("contiguous");
    let cuda_slice = cuda_special.as_slice().expect("contiguous");
    for (c, g) in cpu_slice.iter().zip(cuda_slice.iter()) {
        if c.is_nan() {
            assert!(g.is_nan(), "NaN must pass through interpolate unchanged");
        } else {
            assert_eq!(c.to_bits(), g.to_bits());
        }
    }
}
