//! イシュー #1700/#1701/#1702/#1713: `BackendOps::scalar_unary`／
//! `scalar_binary`（`Sqrt`／`Sub`／`Div`／`Pow`・`Neg`／`Abs`／`Log`／
//! `Log2`／`Log10`／`Sin`／`Cos`／`Tan`・比較演算 6 種（`Gt`／`Ge`／
//! `Lt`／`Le`／`Eq`／`Ne`）・`Clamp`・`Gelu`／`GeluTanh`／`Softplus`）の
//! CPU-CUDA 数値一致検証。
//!
//! `where_masked_fill_parity.rs`（#1637）・`gemm_bias_act_parity.rs`
//! （#599）と同じ構成方針を踏襲する: 環境適応スモーク（属性なし。通常
//! CI で実行し、CUDA 非搭載環境では `BackendError::CudaUnavailable` を
//! 確認して panic しないことのみ検証）と、実機必須の形状網羅
//! （`#[ignore]`。DGX Spark GB10 等）を分離する。判定式・許容誤差は
//! 再定義せず `fandhe_ai_backend_cpu::parity` を唯一の参照とする
//! （`.claude/rules/coding-rust.md`）。
//!
//! `Sqrt`／`Sub`／`Div`・`Neg`／`Abs` は NVRTC 既定オプション
//! （`prec-div`／`prec-sqrt` が true）により IEEE 754 丸めとなりホスト
//! `f32` 演算と bit 同一になる想定（`kernels_scalar_op.rs` モジュール
//! doc「NVRTC 既定オプションと数値契約」参照）のため、`assert_parity`
//! （REQ-2 複合判定）に加え bit 同一（より強い検証）も併記する。
//! `Pow`・`Log`／`Log2`／`Log10`／`Sin`／`Cos`／`Tan`（超越関数。CUDA
//! libm の ulp 誤差がホスト側 glibc libm と一致する保証がない）は
//! `assert_parity` のみで検証する（既存 `exp`／`tanh` と同じ扱い）。
//! 比較演算 6 種・`Clamp` は算術を含まない純粋な選択・比較演算のため
//! bit 同一（`assert_eq!`）で検証する（`kernels_scalar_op.rs::unary_expr`
//! の `Clamp` 分岐コメント・`binary_expr` の比較演算コメント参照）。
//! `Gelu`／`GeluTanh`（`erff`／`tanhf`）・`Softplus`（`log1pf`／`expf`）
//! は超越関数のため `assert_parity` のみで検証する（イシュー #1713）。
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

/// `Sqrt`／`Log`／`Log2`／`Log10` 用に定義域外（負値／0 近傍）を避けた
/// 正の乱数（`[0.1, 2.1)`）を生成する（`backend-cpu/tests/
/// scalar_op_parity.rs::positive_data` と同じオフセット方針）。
fn positive_data(rng: &mut Xorshift64Star, len: usize) -> Vec<f32> {
    rng.fill_vec(len)
        .into_iter()
        .map(|v| v.abs() + 0.1)
        .collect()
}

/// `Neg`／`Abs`／`Sin`／`Cos`／`Tan` 用の符号あり乱数（`[-1, 1)`）。
/// 正の値だけでは `Abs` が恒等になり検証にならず、`Tan` は
/// `positive_data` の範囲（`[0.1, 2.1)`）が `π/2` をまたぐため検証意図
/// が不明瞭になる（モジュール doc・計画参照）。
fn signed_data(rng: &mut Xorshift64Star, len: usize) -> Vec<f32> {
    rng.fill_vec(len)
}

/// [`assert_unary_parity`] が使う入力生成器の選択（kind の定義域に応じ
/// 呼び出し元が指定する）。
#[derive(Clone, Copy)]
enum Domain {
    Positive,
    Signed,
}

fn gen_data(domain: Domain, rng: &mut Xorshift64Star, len: usize) -> Vec<f32> {
    match domain {
        Domain::Positive => positive_data(rng, len),
        Domain::Signed => signed_data(rng, len),
    }
}

fn assert_unary_parity(
    op: ScalarUnaryOp,
    domain: Domain,
    seed: u64,
    shape: &[usize],
    bit_exact: bool,
) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let a = Tensor::new(
        gen_data(domain, &mut Xorshift64Star::new(seed), numel),
        shape,
    )
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

/// 比較演算（`Gt`／`Ge`／`Lt`／`Le`／`Eq`／`Ne`）用の `(a, b)` ペアを
/// 生成する。独立乱数 2 本では `Eq` が全 `0.0`・`Ne` が全 `1.0` になり
/// 検証にならないため、`a` から意図的に等値・やや大きい値・やや小さい
/// 値を混在させて派生させる（イシュー #1702）。
fn comparison_pair_data(seed: u64, len: usize) -> (Vec<f32>, Vec<f32>) {
    let a = positive_data(&mut Xorshift64Star::new(seed), len);
    let b = a
        .iter()
        .enumerate()
        .map(|(i, &v)| match i % 3 {
            0 => v,        // 等値ケース
            1 => v + 0.25, // a < b
            _ => v - 0.25, // a > b
        })
        .collect();
    (a, b)
}

/// 比較演算 [`ScalarBinaryOp`]（`Gt`／`Ge`／`Lt`／`Le`／`Eq`／`Ne`）の
/// CPU-CUDA parity を bit 同一（0.0/1.0 リテラルの純粋な比較演算のため。
/// モジュール doc参照）で検証する。
fn assert_comparison_parity(op: ScalarBinaryOp, seed: u64, shape: &[usize]) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let (a_data, b_data) = comparison_pair_data(seed, numel);
    let a = Tensor::new(a_data, shape).expect("valid tensor");
    let b = Tensor::new(b_data, shape).expect("valid tensor");

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
    assert_eq!(
        cuda_slice, cpu_slice,
        "scalar_binary({op:?}): 比較演算は 0.0/1.0 の純粋な比較のため bit 同一のはず（shape={shape:?}）"
    );
}

/// [`ScalarUnaryOp::Clamp`] の CPU-CUDA parity を bit 同一（算術を含ま
/// ない純粋な選択演算のため）で検証する。
fn assert_clamp_parity(min: f32, max: f32, seed: u64, shape: &[usize]) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let op = ScalarUnaryOp::Clamp { min, max };

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
        &format!("scalar_unary(Clamp{{min={min},max={max}}}) cpu-cuda parity shape={shape:?}"),
        cuda_slice,
        cpu_slice,
    );
    assert_eq!(
        cuda_slice, cpu_slice,
        "scalar_unary(Clamp{{min={min},max={max}}}): 純粋な選択演算のため bit 同一のはず（shape={shape:?}）"
    );
}

/// [`ScalarUnaryOp::Softplus`] の CPU-CUDA parity を REQ-2 複合判定
/// （超越関数〈`log1pf`／`expf`〉のため bit 同一は主張しない。
/// `assert_unary_parity` の `bit_exact=false` 経路と同型）で検証する
/// （イシュー #1713）。
fn assert_softplus_parity(beta: f32, threshold: f32, seed: u64, shape: &[usize]) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let op = ScalarUnaryOp::Softplus { beta, threshold };

    let a = Tensor::new(signed_data(&mut Xorshift64Star::new(seed), numel), shape)
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
        &format!(
            "scalar_unary(Softplus{{beta={beta},threshold={threshold}}}) cpu-cuda parity \
             shape={shape:?}"
        ),
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

            assert_unary_parity(ScalarUnaryOp::Sqrt, Domain::Positive, 1701, &[4], true);
            assert_unary_parity(ScalarUnaryOp::Sqrt, Domain::Positive, 1702, &[3, 5], true);

            assert_binary_parity(ScalarBinaryOp::Sub, 1703, 1704, &[4], true);
            assert_binary_parity(ScalarBinaryOp::Div, 1705, 1706, &[4], true);
            assert_binary_parity(ScalarBinaryOp::Pow, 1707, 1708, &[4], false);

            assert_binary_broadcast_parity(ScalarBinaryOp::Sub, 1709, 1710);

            // #1701 の対象 8 kind（超越関数系）のスモーク。
            assert_unary_parity(ScalarUnaryOp::Neg, Domain::Signed, 1711, &[4], true);
            assert_unary_parity(ScalarUnaryOp::Abs, Domain::Signed, 1712, &[4], true);
            assert_unary_parity(ScalarUnaryOp::Log, Domain::Positive, 1713, &[4], false);
            assert_unary_parity(ScalarUnaryOp::Log2, Domain::Positive, 1714, &[4], false);
            assert_unary_parity(ScalarUnaryOp::Log10, Domain::Positive, 1715, &[4], false);
            assert_unary_parity(ScalarUnaryOp::Sin, Domain::Signed, 1716, &[4], false);
            assert_unary_parity(ScalarUnaryOp::Cos, Domain::Signed, 1717, &[4], false);
            assert_unary_parity(ScalarUnaryOp::Tan, Domain::Signed, 1718, &[4], false);

            // 比較演算・Clamp（イシュー #1702）の代表ケース。
            assert_comparison_parity(ScalarBinaryOp::Gt, 1719, &[4]);
            assert_comparison_parity(ScalarBinaryOp::Eq, 1720, &[4]);
            assert_clamp_parity(0.5, 1.5, 1721, &[4]);

            // GELU（誤差関数版・tanh 近似版）・Softplus（イシュー #1713）
            // のスモーク。既定 beta/threshold（1.0/20.0）に加え、恒等
            // 分岐（`x*beta > threshold`）が発火する組合せも確認する。
            assert_unary_parity(ScalarUnaryOp::Gelu, Domain::Signed, 1722, &[4], false);
            assert_unary_parity(ScalarUnaryOp::GeluTanh, Domain::Signed, 1723, &[4], false);
            assert_softplus_parity(1.0, 20.0, 1724, &[4]);
            assert_softplus_parity(2.0, 1.0, 1725, &[4]);

            // スコープ境界の回帰ガード（fail-closed 契約の確認）: 未実装
            // kind は `BackendError::Unsupported` を返す（意図せず余分な
            // kind を実装してしまっていないかの検出）。番兵 kind は
            // `Relu`（#1701 で `Log` が実装済みになったため、#1702 が
            // 担当する残りの未実装 kind へ付け替え）。
            let unsupported = cuda.scalar_unary(ScalarUnaryOp::Relu, &a);
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
        seed += 14;
        assert_unary_parity(ScalarUnaryOp::Sqrt, Domain::Positive, seed, shape, true);
        assert_binary_parity(ScalarBinaryOp::Sub, seed + 1, seed + 2, shape, true);
        assert_binary_parity(ScalarBinaryOp::Div, seed + 1, seed + 2, shape, true);
        assert_binary_parity(ScalarBinaryOp::Pow, seed + 1, seed + 2, shape, false);

        assert_unary_parity(ScalarUnaryOp::Neg, Domain::Signed, seed + 3, shape, true);
        assert_unary_parity(ScalarUnaryOp::Abs, Domain::Signed, seed + 5, shape, true);
        assert_unary_parity(ScalarUnaryOp::Log, Domain::Positive, seed + 6, shape, false);
        assert_unary_parity(
            ScalarUnaryOp::Log2,
            Domain::Positive,
            seed + 7,
            shape,
            false,
        );
        assert_unary_parity(
            ScalarUnaryOp::Log10,
            Domain::Positive,
            seed + 8,
            shape,
            false,
        );
        assert_unary_parity(ScalarUnaryOp::Sin, Domain::Signed, seed + 9, shape, false);
        assert_unary_parity(ScalarUnaryOp::Cos, Domain::Signed, seed + 10, shape, false);
        assert_unary_parity(ScalarUnaryOp::Tan, Domain::Signed, seed + 11, shape, false);

        for op in [
            ScalarBinaryOp::Gt,
            ScalarBinaryOp::Ge,
            ScalarBinaryOp::Lt,
            ScalarBinaryOp::Le,
            ScalarBinaryOp::Eq,
            ScalarBinaryOp::Ne,
        ] {
            assert_comparison_parity(op, seed + 12, shape);
        }
        assert_clamp_parity(0.5, 1.5, seed + 13, shape);

        assert_unary_parity(ScalarUnaryOp::Gelu, Domain::Signed, seed + 14, shape, false);
        assert_unary_parity(
            ScalarUnaryOp::GeluTanh,
            Domain::Signed,
            seed + 15,
            shape,
            false,
        );
        assert_softplus_parity(1.0, 20.0, seed + 16, shape);
        assert_softplus_parity(2.0, 1.0, seed + 17, shape);
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

    // #1701 の対象 8 kind: 特殊値（0／-0／inf／-inf／NaN）の伝播確認。
    // `Neg`／`Abs` は bit 同一（NaN はクラス一致）、超越関数系は有限
    // 入力のみ `assert_parity`（NaN 混入時のフィルタは Pow と同様）で
    // 検証する。

    // Neg: `0.0`/`-0.0` の符号反転が両バックエンドで一致することを bit
    // 単位で確認する。
    let a = Tensor::new(vec![0.0, -0.0, 1.5, f32::INFINITY], &[4]).expect("valid tensor");
    let cpu_result = cpu.scalar_unary(ScalarUnaryOp::Neg, &a).expect("succeeds");
    let cuda_result = cuda.scalar_unary(ScalarUnaryOp::Neg, &a).expect("succeeds");
    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let cuda_slice = cuda_result.as_slice().expect("contiguous");
    // `assert_eq!` を f32 スライスへ直接使うと `+0.0 == -0.0` が真になり
    // 符号反転の回帰を検出できないため、Abs と同様に `to_bits()` で
    // ビット単位比較する（NaN 混入時はクラス一致で確認する）。
    for (i, (&c, &g)) in cpu_slice.iter().zip(cuda_slice.iter()).enumerate() {
        if c.is_nan() {
            assert!(g.is_nan(), "Neg NaN propagation mismatch at index {i}");
        } else {
            assert_eq!(
                c.to_bits(),
                g.to_bits(),
                "Neg cpu/cuda bit mismatch at index {i}"
            );
        }
    }
    assert_eq!(cpu_slice[0].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(cpu_slice[1].to_bits(), (0.0_f32).to_bits());

    // Abs: `-0.0` は `+0.0` へ、NaN はクラス一致で確認する。
    let a = Tensor::new(vec![-0.0, -1.5, f32::INFINITY, f32::NAN], &[4]).expect("valid tensor");
    let cpu_result = cpu.scalar_unary(ScalarUnaryOp::Abs, &a).expect("succeeds");
    let cuda_result = cuda.scalar_unary(ScalarUnaryOp::Abs, &a).expect("succeeds");
    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let cuda_slice = cuda_result.as_slice().expect("contiguous");
    for (i, (&c, &g)) in cpu_slice.iter().zip(cuda_slice.iter()).enumerate() {
        if c.is_nan() {
            assert!(g.is_nan(), "Abs NaN propagation mismatch at index {i}");
        } else {
            assert_eq!(
                c.to_bits(),
                g.to_bits(),
                "Abs cpu/cuda bit mismatch at index {i}"
            );
        }
    }
    assert_eq!(cpu_slice[0].to_bits(), (0.0_f32).to_bits());

    // Log/Log2/Log10: `Log(0.0) == -inf`・`Log(負値) == NaN`・
    // `Log(1.0) == 0.0`・`Log(inf) == inf` を両バックエンドで確認する。
    for op in [
        ScalarUnaryOp::Log,
        ScalarUnaryOp::Log2,
        ScalarUnaryOp::Log10,
    ] {
        let a = Tensor::new(vec![0.0, -1.0, 1.0, f32::INFINITY], &[4]).expect("valid tensor");
        let cpu_result = cpu.scalar_unary(op, &a).expect("succeeds");
        let cuda_result = cuda.scalar_unary(op, &a).expect("succeeds");
        let cpu_slice = cpu_result.as_slice().expect("contiguous");
        let cuda_slice = cuda_result.as_slice().expect("contiguous");
        assert!(
            cpu_slice[0].is_infinite() && cpu_slice[0] < 0.0,
            "{op:?}(0.0) must be -inf on cpu"
        );
        assert!(
            cuda_slice[0].is_infinite() && cuda_slice[0] < 0.0,
            "{op:?}(0.0) must be -inf on cuda"
        );
        assert!(cpu_slice[1].is_nan(), "{op:?}(-1.0) must be NaN on cpu");
        assert!(cuda_slice[1].is_nan(), "{op:?}(-1.0) must be NaN on cuda");
        assert_eq!(cpu_slice[2], 0.0, "{op:?}(1.0) must be 0.0 on cpu");
        assert_eq!(cuda_slice[2], 0.0, "{op:?}(1.0) must be 0.0 on cuda");
        assert!(
            cpu_slice[3].is_infinite() && cpu_slice[3] > 0.0,
            "{op:?}(inf) must be +inf on cpu"
        );
        assert!(
            cuda_slice[3].is_infinite() && cuda_slice[3] > 0.0,
            "{op:?}(inf) must be +inf on cuda"
        );
        // 有限値（index 2 のみ）を assert_parity へ渡す。
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("scalar_unary({op:?}) edge case finite element"),
            &cuda_slice[2..3],
            &cpu_slice[2..3],
        );
    }

    // Sin/Cos/Tan: `0.0` の厳密値・inf/NaN 入力の NaN 伝播を確認する。
    let a = Tensor::new(vec![0.0, -0.0, f32::INFINITY, f32::NAN], &[4]).expect("valid tensor");
    for (op, zero_expect) in [
        (ScalarUnaryOp::Sin, 0.0_f32),
        (ScalarUnaryOp::Cos, 1.0_f32),
        (ScalarUnaryOp::Tan, 0.0_f32),
    ] {
        let cpu_result = cpu.scalar_unary(op, &a).expect("succeeds");
        let cuda_result = cuda.scalar_unary(op, &a).expect("succeeds");
        let cpu_slice = cpu_result.as_slice().expect("contiguous");
        let cuda_slice = cuda_result.as_slice().expect("contiguous");
        assert_eq!(
            cpu_slice[0], zero_expect,
            "{op:?}(0.0) unexpected cpu value"
        );
        assert_eq!(
            cuda_slice[0], zero_expect,
            "{op:?}(0.0) unexpected cuda value"
        );
        for i in [2usize, 3] {
            assert!(
                cpu_slice[i].is_nan(),
                "{op:?} at index {i} must be NaN on cpu (inf/NaN input)"
            );
            assert!(
                cuda_slice[i].is_nan(),
                "{op:?} at index {i} must be NaN on cuda (inf/NaN input)"
            );
        }
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("scalar_unary({op:?}) edge case finite elements"),
            &cuda_slice[0..2],
            &cpu_slice[0..2],
        );
    }

    // 比較演算エッジ（イシュー #1702）: `NaN`／`-0.0`／`inf` を含む。
    // `Eq(NaN, NaN) == 0.0`／`Ne(NaN, NaN) == 1.0`（NaN は自身とも等しく
    // ない。IEEE 754）・`Eq(-0.0, 0.0) == 1.0`（符号付きゼロは数値として
    // 等しい）・`Ge(inf, inf) == 1.0` を CPU と bit 同一で確認する。
    let a = Tensor::new(vec![f32::NAN, -0.0, f32::INFINITY, 1.0, f32::NAN], &[5])
        .expect("valid tensor");
    let b = Tensor::new(vec![f32::NAN, 0.0, f32::INFINITY, 2.0, 1.0], &[5]).expect("valid tensor");
    for op in [
        ScalarBinaryOp::Gt,
        ScalarBinaryOp::Ge,
        ScalarBinaryOp::Lt,
        ScalarBinaryOp::Le,
        ScalarBinaryOp::Eq,
        ScalarBinaryOp::Ne,
    ] {
        let cpu_result = cpu.scalar_binary(op, &a, &b).expect("cpu succeeds");
        let cuda_result = cuda.scalar_binary(op, &a, &b).expect("cuda succeeds");
        let cpu_slice = cpu_result.as_slice().expect("contiguous");
        let cuda_slice = cuda_result.as_slice().expect("contiguous");
        assert_eq!(
            cuda_slice, cpu_slice,
            "scalar_binary({op:?}) edge case (NaN/-0.0/inf) cpu/cuda bit mismatch"
        );
    }
    assert_eq!(
        cpu.scalar_binary(ScalarBinaryOp::Eq, &a, &b)
            .expect("cpu succeeds")
            .as_slice()
            .expect("contiguous")[0],
        0.0,
        "Eq(NaN, NaN) must be 0.0 per IEEE 754"
    );
    assert_eq!(
        cpu.scalar_binary(ScalarBinaryOp::Ne, &a, &b)
            .expect("cpu succeeds")
            .as_slice()
            .expect("contiguous")[0],
        1.0,
        "Ne(NaN, NaN) must be 1.0 per IEEE 754"
    );
    assert_eq!(
        cpu.scalar_binary(ScalarBinaryOp::Eq, &a, &b)
            .expect("cpu succeeds")
            .as_slice()
            .expect("contiguous")[1],
        1.0,
        "Eq(-0.0, 0.0) must be 1.0 (signed zeros compare equal)"
    );

    // Clamp エッジ（イシュー #1702）: `NaN`／`inf`／`-inf`／`-0.0` を含む
    // 入力・通常の `Clamp{-1,1}` と `min > max`（常に `max` を返す。§3.4
    // 数値規約）の `Clamp{5.0, 1.0}` の両方を確認する。
    let clamp_input = Tensor::new(
        vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0],
        &[5],
    )
    .expect("valid tensor");
    for (min, max) in [(-1.0f32, 1.0f32), (5.0f32, 1.0f32), (0.0f32, 1.0f32)] {
        let op = ScalarUnaryOp::Clamp { min, max };
        let cpu_result = cpu.scalar_unary(op, &clamp_input).expect("cpu succeeds");
        let cuda_result = cuda.scalar_unary(op, &clamp_input).expect("cuda succeeds");
        let cpu_slice = cpu_result.as_slice().expect("contiguous");
        let cuda_slice = cuda_result.as_slice().expect("contiguous");
        for (i, (&c, &g)) in cpu_slice.iter().zip(cuda_slice.iter()).enumerate() {
            if c.is_nan() {
                assert!(
                    g.is_nan(),
                    "Clamp{{min={min},max={max}}} NaN propagation mismatch at index {i}"
                );
            } else {
                assert_eq!(
                    c.to_bits(),
                    g.to_bits(),
                    "Clamp{{min={min},max={max}}} cpu/cuda bit mismatch at index {i}"
                );
            }
        }
    }
    // `-0.0` の保持（`Clamp{0.0, 1.0}` は `x < min` の比較で `-0.0 < 0.0`
    // は false のため `-0.0` がそのまま通過する。IEEE 754 の符号付き
    // ゼロ比較規約どおり）を bit で確認する。
    let clamp_zero_result = cpu
        .scalar_unary(ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 }, &clamp_input)
        .expect("cpu succeeds");
    assert_eq!(
        clamp_zero_result.as_slice().expect("contiguous")[3].to_bits(),
        (-0.0f32).to_bits(),
        "Clamp{{0.0,1.0}} must preserve -0.0 sign bit"
    );

    // numel == 0（空 shape）の早期 return。`Log` でも確認する（#1701）。
    let empty = Tensor::new(Vec::<f32>::new(), &[0]).expect("valid tensor");
    let cuda_empty_unary = cuda
        .scalar_unary(ScalarUnaryOp::Sqrt, &empty)
        .expect("empty tensor must succeed");
    assert_eq!(cuda_empty_unary.shape(), &[0]);
    let cuda_empty_log = cuda
        .scalar_unary(ScalarUnaryOp::Log, &empty)
        .expect("empty tensor must succeed");
    assert_eq!(cuda_empty_log.shape(), &[0]);
    let cuda_empty_binary = cuda
        .scalar_binary(ScalarBinaryOp::Sub, &empty, &empty)
        .expect("empty tensor must succeed");
    assert_eq!(cuda_empty_binary.shape(), &[0]);
    let cuda_empty_clamp = cuda
        .scalar_unary(ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 }, &empty)
        .expect("empty tensor must succeed (Clamp)");
    assert_eq!(cuda_empty_clamp.shape(), &[0]);
}
