//! イシュー #1700: `BackendOps::scalar_unary`／`scalar_binary`（`Sqrt`／
//! `Sub`／`Div`／`Pow`）の CPU-CUDA 数値一致検証。
//!
//! `where_masked_fill_parity.rs`（#1637）・`gemm_bias_act_parity.rs`
//! （#599）と同じ構成方針を踏襲する: 環境適応スモーク（属性なし。通常
//! CI で実行し、CUDA 非搭載環境では `BackendError::CudaUnavailable` を
//! 確認して panic しないことのみ検証）と、実機必須の形状網羅
//! （`#[ignore]`。DGX Spark GB10 等）を分離する。判定式・許容誤差は
//! 再定義せず `fandhe_ai_backend_cpu::parity` を唯一の参照とする
//! （`.claude/rules/coding-rust.md`）。
//!
//! `Sqrt`／`Sub`／`Div` は NVRTC 既定オプション（`prec-div`／`prec-sqrt`
//! が true）により IEEE 754 丸めとなりホスト `f32` 演算と bit 同一に
//! なる想定（`kernels_scalar_op.rs` モジュール doc「NVRTC 既定オプション
//! と数値契約」参照）のため、`assert_parity`（REQ-2 複合判定）に加え
//! bit 同一（より強い検証）も併記する。`Pow` は超越関数の合成近似の
//! ため `assert_parity` のみで検証する（既存 `exp`／`tanh` と同じ扱い）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test scalar_op_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, ScalarBinaryOp, ScalarUnaryOp, Tensor};

mod common;

/// `Sqrt` 用に定義域外（負値）を避けた正の乱数（`[0.1, 2.1)`）を生成
/// する（`backend-cpu/tests/scalar_op_parity.rs::positive_data` と同じ
/// オフセット方針）。
fn positive_data(rng: &mut Xorshift64Star, len: usize) -> Vec<f32> {
    rng.fill_vec(len)
        .into_iter()
        .map(|v| v.abs() + 0.1)
        .collect()
}

fn assert_unary_parity(op: ScalarUnaryOp, seed: u64, shape: &[usize], bit_exact: bool) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let a = Tensor::new(positive_data(&mut Xorshift64Star::new(seed), numel), shape)
        .expect("valid tensor");

    let cpu_result = cpu
        .scalar_unary(op, &a)
        .expect("cpu scalar_unary always succeeds for implemented kinds");
    let cuda_result = cuda
        .scalar_unary(op, &a)
        .expect("cuda scalar_unary must succeed on CUDA-equipped test runner");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let cuda_slice = cuda_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("scalar_unary({op:?}) cpu-cuda parity shape={shape:?}"),
        cuda_slice,
        cpu_slice,
    );
    if bit_exact {
        assert_eq!(
            cuda_slice, cpu_slice,
            "scalar_unary({op:?}): IEEE 754 丸め契約により bit 同一のはず（shape={shape:?}）"
        );
    }
}

fn assert_binary_parity(
    op: ScalarBinaryOp,
    seed_a: u64,
    seed_b: u64,
    shape: &[usize],
    bit_exact: bool,
) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let a = Tensor::new(
        positive_data(&mut Xorshift64Star::new(seed_a), numel),
        shape,
    )
    .expect("valid tensor");
    let b = Tensor::new(
        positive_data(&mut Xorshift64Star::new(seed_b), numel),
        shape,
    )
    .expect("valid tensor");

    let cpu_result = cpu
        .scalar_binary(op, &a, &b)
        .expect("cpu scalar_binary always succeeds for implemented kinds");
    let cuda_result = cuda
        .scalar_binary(op, &a, &b)
        .expect("cuda scalar_binary must succeed on CUDA-equipped test runner");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let cuda_slice = cuda_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("scalar_binary({op:?}) cpu-cuda parity shape={shape:?}"),
        cuda_slice,
        cpu_slice,
    );
    if bit_exact {
        assert_eq!(
            cuda_slice, cpu_slice,
            "scalar_binary({op:?}): IEEE 754 丸め契約により bit 同一のはず（shape={shape:?}）"
        );
    }
}

/// broadcast 形状（`[m, n]` + `[n]`）での `scalar_binary` parity。
fn assert_binary_broadcast_parity(op: ScalarBinaryOp, seed_a: u64, seed_b: u64) {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let a = Tensor::new(positive_data(&mut Xorshift64Star::new(seed_a), 12), &[3, 4])
        .expect("valid tensor");
    let b = Tensor::new(positive_data(&mut Xorshift64Star::new(seed_b), 4), &[4])
        .expect("valid tensor");

    let cpu_result = cpu.scalar_binary(op, &a, &b).expect("cpu succeeds");
    let cuda_result = cuda.scalar_binary(op, &a, &b).expect("cuda succeeds");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let cuda_slice = cuda_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("scalar_binary({op:?}) broadcast cpu-cuda parity"),
        cuda_slice,
        cpu_slice,
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA 不在なら
/// `BackendError::CudaUnavailable` を確認して早期 return する。実機なら
/// 形状網羅ケースまで実行する。
#[test]
fn scalar_op_parity_smoke_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let a = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], &[2, 2]).expect("valid tensor");

    match cuda.scalar_unary(ScalarUnaryOp::Sqrt, &a) {
        Ok(_) => {
            common::parity_baseline::assert_tolerance_constants_pinned();

            assert_unary_parity(ScalarUnaryOp::Sqrt, 1701, &[4], true);
            assert_unary_parity(ScalarUnaryOp::Sqrt, 1702, &[3, 5], true);

            assert_binary_parity(ScalarBinaryOp::Sub, 1703, 1704, &[4], true);
            assert_binary_parity(ScalarBinaryOp::Div, 1705, 1706, &[4], true);
            assert_binary_parity(ScalarBinaryOp::Pow, 1707, 1708, &[4], false);

            assert_binary_broadcast_parity(ScalarBinaryOp::Sub, 1709, 1710);

            // スコープ境界の回帰ガード（fail-closed 契約の確認）: 未実装
            // kind は `BackendError::Unsupported` を返す（意図せず余分な
            // kind を実装してしまっていないかの検出）。
            let unsupported = cuda.scalar_unary(ScalarUnaryOp::Log, &a);
            assert!(matches!(unsupported, Err(BackendError::Unsupported(_))));
            let unsupported_bin = cuda.scalar_binary(ScalarBinaryOp::Add, &a, &a);
            assert!(matches!(unsupported_bin, Err(BackendError::Unsupported(_))));

            // 形状不一致は `BackendError::ShapeMismatch` を返す
            // （`elementwise_binary` の既存契約と同様の再検査）。`a` は
            // `[2, 2]` のため、末尾次元が `1`／`2` 以外でブロードキャスト
            // 不能な `[3]` を使う（`[2]` は `Tensor::broadcast_with` の
            // 契約上ブロードキャスト可能で shape mismatch にならない。
            // codex-review 指摘・PR #1781）。
            let bad = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).expect("valid tensor");
            let err = cuda
                .scalar_binary(ScalarBinaryOp::Sub, &a, &bad)
                .expect_err("shape mismatch (non-broadcastable) must be rejected");
            assert!(matches!(err, BackendError::ShapeMismatch(_)));
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::scalar_unary: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`PARALLEL_THRESHOLD`
/// （CPU 側 rayon 閾値。境界前後）・0/inf/NaN 通過を検証する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn scalar_op_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let shapes: &[&[usize]] = &[
        &[1],
        &[4],
        &[3, 5],
        &[2, 3, 4],
        &[1 << 15], // PARALLEL_THRESHOLD ちょうど（backend-cpu::elementwise）
        &[(1 << 15) - 1],
        &[(1 << 15) + 1],
    ];
    let mut seed = 9000u64;
    for &shape in shapes {
        seed += 4;
        assert_unary_parity(ScalarUnaryOp::Sqrt, seed, shape, true);
        assert_binary_parity(ScalarBinaryOp::Sub, seed + 1, seed + 2, shape, true);
        assert_binary_parity(ScalarBinaryOp::Div, seed + 1, seed + 2, shape, true);
        assert_binary_parity(ScalarBinaryOp::Pow, seed + 1, seed + 2, shape, false);
    }

    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    // Sqrt(0.0) == 0.0・Sqrt(負値) == NaN（IEEE 754 のまま伝播。定義域外
    // を panic せず処理することの確認）。
    let a = Tensor::new(vec![0.0, -1.0, f32::INFINITY, 4.0], &[4]).expect("valid tensor");
    let cpu_result = cpu
        .scalar_unary(ScalarUnaryOp::Sqrt, &a)
        .expect("cpu succeeds");
    let cuda_result = cuda
        .scalar_unary(ScalarUnaryOp::Sqrt, &a)
        .expect("cuda succeeds");
    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let cuda_slice = cuda_result.as_slice().expect("contiguous");
    for (i, (&c, &g)) in cpu_slice.iter().zip(cuda_slice.iter()).enumerate() {
        if c.is_nan() {
            assert!(g.is_nan(), "Sqrt NaN propagation mismatch at index {i}");
        } else {
            assert_eq!(
                c.to_bits(),
                g.to_bits(),
                "Sqrt cpu/cuda bit mismatch at index {i}"
            );
        }
    }

    // Div: `b == 0.0` を含む（inf／-inf／NaN 伝播確認。IEEE 754 のまま
    // panic しない）。
    let a = Tensor::new(vec![1.0, -1.0, 0.0, 5.0], &[4]).expect("valid tensor");
    let b = Tensor::new(vec![0.0, 0.0, 0.0, 2.0], &[4]).expect("valid tensor");
    let cpu_result = cpu
        .scalar_binary(ScalarBinaryOp::Div, &a, &b)
        .expect("cpu succeeds");
    let cuda_result = cuda
        .scalar_binary(ScalarBinaryOp::Div, &a, &b)
        .expect("cuda succeeds");
    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let cuda_slice = cuda_result.as_slice().expect("contiguous");
    for (i, (&c, &g)) in cpu_slice.iter().zip(cuda_slice.iter()).enumerate() {
        if c.is_nan() {
            assert!(g.is_nan(), "Div NaN propagation mismatch at index {i}");
        } else {
            assert_eq!(
                c.to_bits(),
                g.to_bits(),
                "Div cpu/cuda bit mismatch at index {i}"
            );
        }
    }

    // Pow: `a == 0.0`／`b == 0.0`（IEEE `0^0 = 1` 契約）・負の底＋非整数
    // 指数（NaN 伝播）を REQ-2 複合判定のみで確認する。
    let a = Tensor::new(vec![0.0, 2.0, -2.0, 0.0], &[4]).expect("valid tensor");
    let b = Tensor::new(vec![0.0, 3.0, 0.5, 5.0], &[4]).expect("valid tensor");
    let cpu_result = cpu
        .scalar_binary(ScalarBinaryOp::Pow, &a, &b)
        .expect("cpu succeeds");
    let cuda_result = cuda
        .scalar_binary(ScalarBinaryOp::Pow, &a, &b)
        .expect("cuda succeeds");
    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let cuda_slice = cuda_result.as_slice().expect("contiguous");
    // index 2（`(-2.0).powf(0.5)`）は CPU・CUDA いずれも NaN になるため、
    // NaN 要素はクラス一致（両方 NaN）で個別確認し、`assert_parity` には
    // 有限値の要素のみを渡す（NaN 同士は数値比較上不合格になり、後続の
    // 空テンソル検証等へ到達できなくなるため。Sqrt／Div の既存 NaN 個別
    // 確認と同じ扱い。codex-review 指摘・PR #1781）。
    let finite_cpu: Vec<f32> = cpu_slice
        .iter()
        .zip(cuda_slice.iter())
        .filter(|(c, _)| !c.is_nan())
        .map(|(&c, _)| c)
        .collect();
    let finite_cuda: Vec<f32> = cpu_slice
        .iter()
        .zip(cuda_slice.iter())
        .filter(|(c, _)| !c.is_nan())
        .map(|(_, &g)| g)
        .collect();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "scalar_binary(Pow) 0^0/negative-base edge cases",
        &finite_cuda,
        &finite_cpu,
    );
    for (i, (&c, &g)) in cpu_slice.iter().zip(cuda_slice.iter()).enumerate() {
        if c.is_nan() {
            assert!(g.is_nan(), "Pow NaN propagation mismatch at index {i}");
        }
    }
    assert_eq!(
        cpu_slice[0], 1.0,
        "0^0 must be 1.0 per IEEE 754 pow contract"
    );
    assert!(
        cuda_slice[2].is_nan(),
        "negative base with non-integer exponent must propagate NaN on CUDA"
    );

    // numel == 0（空 shape）の早期 return。
    let empty = Tensor::new(Vec::<f32>::new(), &[0]).expect("valid tensor");
    let cuda_empty_unary = cuda
        .scalar_unary(ScalarUnaryOp::Sqrt, &empty)
        .expect("empty tensor must succeed");
    assert_eq!(cuda_empty_unary.shape(), &[0]);
    let cuda_empty_binary = cuda
        .scalar_binary(ScalarBinaryOp::Sub, &empty, &empty)
        .expect("empty tensor must succeed");
    assert_eq!(cuda_empty_binary.shape(), &[0]);
}
