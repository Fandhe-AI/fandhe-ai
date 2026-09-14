//! イシュー #1756: `BackendOps::pad`（`torch.nn.functional.pad
//! (mode='constant')` 相当）の CPU-CUDA 数値一致検証。
//!
//! `gather_scatter_parity.rs`（#1777）と同じ構成方針を踏襲する:
//! 環境適応スモーク（属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `BackendError::CudaUnavailable` を確認して panic しないことのみ検証。
//! デバイス初期化より前に返る shape 検査経路は GPU 有無に依らず検証する）
//! と、実機必須の形状網羅（`#[ignore]`。DGX Spark GB10 等）を分離する。
//!
//! **契約は bit 完全一致**（pad は算術を含まない純粋なコピー演算のため。
//! `.claude/rules/coding-rust.md` 数値契約節参照）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test constant_pad_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// `BackendError` は `PartialEq` を実装しないため、`ShapeMismatch` の
/// 内側 `ShapeError`（`PartialEq` 実装済み）だけを取り出して比較する。
fn expect_shape_mismatch(err: BackendError) -> fandhe_ai_tensor_core::ShapeError {
    match err {
        BackendError::ShapeMismatch(inner) => inner,
        other => panic!("expected BackendError::ShapeMismatch, got {other}"),
    }
}

fn assert_pad_parity(seed: u64, in_shape: &[usize], pads: &[(usize, usize)], value: f32) {
    let numel_in: usize = in_shape.iter().product();

    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let input =
        Tensor::new(Xorshift64Star::new(seed).fill_vec(numel_in), in_shape).expect("valid tensor");

    let cpu_out = cpu
        .pad(&input, pads, value)
        .expect("cpu pad always succeeds for valid input");
    let cuda_out = cuda
        .pad(&input, pads, value)
        .expect("cuda pad must succeed on real device");

    assert_eq!(cpu_out.shape(), cuda_out.shape());
    let cpu_slice = cpu_out.as_slice().expect("contiguous");
    let cuda_slice = cuda_out.as_slice().expect("contiguous");
    assert_eq!(
        cpu_slice, cuda_slice,
        "pad(in_shape={in_shape:?}, pads={pads:?}, value={value}): CPU と CUDA は bit 完全一致のはず"
    );

    // run-to-run で bit 同一（決定的カーネル）。
    let cuda_out2 = cuda.pad(&input, pads, value).expect("cuda pad rerun");
    assert_eq!(
        cuda_out2.as_slice().expect("contiguous"),
        cuda_slice,
        "pad: run-to-run で bit 同一のはず"
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA 不在なら
/// `BackendError::CudaUnavailable` を確認して早期 return する
/// （`gather_scatter_parity_smoke_env_adaptive` と同じ分岐パターン）。
/// デバイス初期化より前に返る shape 検査経路（`pad_out_shape` の
/// 再検査）は GPU 有無に依らず検証する。
#[test]
fn constant_pad_parity_smoke_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();

    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("valid tensor");

    match cuda.pad(&input, &[(1, 0), (0, 1)], 0.0) {
        Ok(_) => {
            assert_pad_parity(9101, &[2, 3], &[(1, 0), (0, 1)], -1.0);
            assert_pad_parity(9102, &[4], &[(2, 3)], 0.0);
            assert_pad_parity(9103, &[3, 4], &[(0, 0), (0, 0)], 0.0); // 恒等（pads 全 0）

            // 空入力 → 非空出力（value で埋まる）。
            let empty_input = Tensor::new(Vec::<f32>::new(), &[0, 3]).expect("valid tensor");
            let cpu_empty = cpu
                .pad(&empty_input, &[(1, 0), (0, 0)], 7.0)
                .expect("cpu pad on empty input");
            let cuda_empty = cuda
                .pad(&empty_input, &[(1, 0), (0, 0)], 7.0)
                .expect("cuda pad on empty input");
            assert_eq!(
                cuda_empty.as_slice().expect("contiguous"),
                cpu_empty.as_slice().expect("contiguous")
            );

            // shape 不一致（rank mismatch）は `BackendError::
            // ShapeMismatch` を返す（実装側の再検査。`.claude/rules/
            // security.md` A08）。
            let err = cuda
                .pad(&input, &[(1, 0)], 0.0)
                .expect_err("rank mismatch must be rejected");
            assert!(matches!(err, BackendError::ShapeMismatch(_)));
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");

            // shape 検査はデバイス初期化より前に走るため CUDA 非搭載
            // 環境でも検証できる（CPU の返す値と一致することも確認）。
            let cpu_err = expect_shape_mismatch(
                cpu.pad(&input, &[(1, 0)], 0.0)
                    .expect_err("cpu must reject rank mismatch"),
            );
            let cuda_err = expect_shape_mismatch(
                cuda.pad(&input, &[(1, 0)], 0.0)
                    .expect_err("rank mismatch must be rejected even without CUDA"),
            );
            assert_eq!(cpu_err, cuda_err);
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::pad: {other}"),
    }
}

/// `(in_shape, pads)` の組（clippy `type_complexity` 回避のためのエイリ
/// アス。本ファイル限定の局所定義）。
type ShapePads<'a> = (&'a [usize], &'a [(usize, usize)]);

/// 実機必須の形状網羅（受け入れ条件の本体）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn pad_matches_cpu_across_shapes() {
    let shapes_pads: &[ShapePads] = &[
        (&[4], &[(1, 2)]),
        (&[2, 3], &[(1, 0), (0, 1)]),
        (&[2, 3], &[(0, 0), (0, 0)]), // 恒等
        (&[2, 3, 4], &[(1, 1), (0, 2), (2, 0)]),
        (&[1 << 12, 3], &[(1, 0), (0, 0)]), // ブロック境界をまたぐ大きさ
    ];
    let mut seed = 20_000u64;
    for &(in_shape, pads) in shapes_pads {
        seed += 7;
        assert_pad_parity(seed, in_shape, pads, -2.5);
    }

    // 非 contiguous な input（transpose view）を渡しても contiguous 化後
    // に一致する。
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let base = Tensor::new((0..12).map(|v| v as f32).collect(), &[3, 4]).expect("valid tensor");
    let transposed = base.transpose(0, 1).expect("valid transpose");
    let pads = [(1usize, 0usize), (0usize, 1usize)];
    let cpu_out = cpu.pad(&transposed, &pads, 0.0).expect("cpu pad");
    let cuda_out = cuda.pad(&transposed, &pads, 0.0).expect("cuda pad");
    assert_eq!(
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous")
    );

    // NaN／±inf の通過（入力）・`value` に NaN を使うクラス一致確認。
    let input_special = Tensor::new(
        vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0],
        &[2, 2],
    )
    .expect("valid tensor");
    let pads_special = [(1usize, 0usize), (0usize, 1usize)];
    let cpu_special = cpu
        .pad(&input_special, &pads_special, f32::NAN)
        .expect("cpu pad");
    let cuda_special = cuda
        .pad(&input_special, &pads_special, f32::NAN)
        .expect("cuda pad");
    let cpu_special_slice = cpu_special.as_slice().expect("contiguous");
    let cuda_special_slice = cuda_special.as_slice().expect("contiguous");
    for (c, g) in cpu_special_slice.iter().zip(cuda_special_slice.iter()) {
        if c.is_nan() {
            assert!(g.is_nan(), "NaN must remain NaN through pad");
        } else {
            assert_eq!(c, g);
        }
    }
}
