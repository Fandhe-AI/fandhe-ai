//! イシュー #1707: `BackendOps::scalar_unary`／`scalar_binary`（`Sqrt`／
//! `Sub`／`Div`／`Pow`）の CPU-Metal 数値一致検証（CUDA 側
//! `backend-cuda::tests::scalar_op_parity`〈#1700〉の Metal 対応版）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`where_masked_fill_parity.rs`〈#1637〉と同方針。`#![cfg(target_os =
//! "macos")]` により Linux CI ではコンパイル対象外になり、`#[ignore]`
//! により通常の `cargo test` からも除外される。Metal は Linux で
//! `MetalBackendOps` が存在しないため CUDA 側のような環境適応スモーク
//! テストは持てない）。
//!
//! 判定式・許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の
//! 参照とする（`.claude/rules/coding-rust.md`）。`Sqrt`／`Sub`／`Div` は
//! `metal::precise::sqrt`／`+ - * /` が correctly rounded であることに
//! より IEEE 754 丸めとなりホスト `f32` 演算と bit 同一になる想定
//! （`crate::scalar_op_source` モジュール doc「コンパイルオプションと
//! 数値契約」参照）のため、`assert_parity`（REQ-2 複合判定）に加え
//! bit 同一（より強い検証）も併記する。`Pow`（超越関数。`metal::
//! precise::pow` の ulp 誤差がホスト側 libm と一致する保証がない）は
//! `assert_parity` のみで検証する（既存 `exp`／`tanh` と同じ扱い）。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test scalar_op_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, ScalarBinaryOp, ScalarUnaryOp, Tensor};

/// `Sqrt` 用に定義域外（負値・0 近傍）を避けた正の乱数（`[0.1, 2.1)`）を
/// 生成する（CUDA 側 `backend-cuda/tests/scalar_op_parity.rs::
/// positive_data` と同じオフセット方針）。
fn positive_data(rng: &mut Xorshift64Star, len: usize) -> Vec<f32> {
    rng.fill_vec(len)
        .into_iter()
        .map(|v| v.abs() + 0.1)
        .collect()
}

fn assert_sqrt_parity(seed: u64, shape: &[usize]) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a = Tensor::new(positive_data(&mut Xorshift64Star::new(seed), numel), shape)
        .expect("valid tensor");

    let cpu_result = cpu
        .scalar_unary(ScalarUnaryOp::Sqrt, &a)
        .expect("cpu scalar_unary always succeeds for implemented kinds");
    let metal_result = metal
        .scalar_unary(ScalarUnaryOp::Sqrt, &a)
        .expect("metal scalar_unary must succeed on Metal-equipped test runner");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let metal_slice = metal_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("scalar_unary(Sqrt) cpu-metal parity shape={shape:?}"),
        metal_slice,
        cpu_slice,
    );
    assert_eq!(
        metal_slice, cpu_slice,
        "scalar_unary(Sqrt): IEEE 754 丸め契約により bit 同一のはず（shape={shape:?}）"
    );
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
    let metal = MetalBackendOps::new();

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
    let metal_result = metal
        .scalar_binary(op, &a, &b)
        .expect("metal scalar_binary must succeed on Metal-equipped test runner");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let metal_slice = metal_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("scalar_binary({op:?}) cpu-metal parity shape={shape:?}"),
        metal_slice,
        cpu_slice,
    );
    if bit_exact {
        assert_eq!(
            metal_slice, cpu_slice,
            "scalar_binary({op:?}): IEEE 754 丸め契約により bit 同一のはず（shape={shape:?}）"
        );
    }
}

/// broadcast 形状（`[m, n]` + `[n]`）での `scalar_binary` parity。
fn assert_binary_broadcast_parity(op: ScalarBinaryOp, seed_a: u64, seed_b: u64) {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a = Tensor::new(positive_data(&mut Xorshift64Star::new(seed_a), 12), &[3, 4])
        .expect("valid tensor");
    let b = Tensor::new(positive_data(&mut Xorshift64Star::new(seed_b), 4), &[4])
        .expect("valid tensor");

    let cpu_result = cpu.scalar_binary(op, &a, &b).expect("cpu succeeds");
    let metal_result = metal.scalar_binary(op, &a, &b).expect("metal succeeds");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let metal_slice = metal_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("scalar_binary({op:?}) broadcast cpu-metal parity"),
        metal_slice,
        cpu_slice,
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`Sqrt`（unary）・
/// `Sub`／`Div`（bit 同一）・`Pow`（REQ-2 のみ）の各 kind を、
/// `PARALLEL_THRESHOLD`（CPU 側 rayon 閾値）境界前後を含む形状で検証する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn scalar_op_matches_cpu_across_shapes() {
    let shapes: &[&[usize]] = &[
        &[1],
        &[4],
        &[3, 5],
        &[2, 3, 4],
        &[1 << 15], // PARALLEL_THRESHOLD ちょうど
        &[(1 << 15) - 1],
        &[(1 << 15) + 1],
    ];
    let mut seed = 11_000u64;
    for &shape in shapes {
        seed += 3;
        assert_sqrt_parity(seed, shape);
        assert_binary_parity(ScalarBinaryOp::Sub, seed, seed + 1, shape, true);
        assert_binary_parity(ScalarBinaryOp::Div, seed, seed + 1, shape, true);
        assert_binary_parity(ScalarBinaryOp::Pow, seed, seed + 1, shape, false);
    }
}

/// broadcast 形状（`[3, 4]` + `[4]`）の parity。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn scalar_binary_broadcast_matches_cpu() {
    assert_binary_broadcast_parity(ScalarBinaryOp::Sub, 12_001, 12_002);
    assert_binary_broadcast_parity(ScalarBinaryOp::Div, 12_003, 12_004);
    assert_binary_broadcast_parity(ScalarBinaryOp::Pow, 12_005, 12_006);
}

/// `0.0`／`inf`／`NaN` 混入時の通過（クラス一致で比較。`NaN` の payload
/// は処理系依存のため bit 同一ではなくクラス一致で検証する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn scalar_op_handles_zero_inf_nan() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    // Sqrt: 0 は 0・負値は NaN・+inf は +inf。
    let a = Tensor::new(vec![0.0, -1.0, f32::INFINITY, 4.0], &[4]).expect("valid tensor");
    let cpu_out = cpu
        .scalar_unary(ScalarUnaryOp::Sqrt, &a)
        .expect("cpu succeeds");
    let metal_out = metal
        .scalar_unary(ScalarUnaryOp::Sqrt, &a)
        .expect("metal succeeds");
    for (i, (&c, &g)) in cpu_out
        .as_slice()
        .unwrap()
        .iter()
        .zip(metal_out.as_slice().unwrap().iter())
        .enumerate()
    {
        if c.is_nan() {
            assert!(g.is_nan(), "index {i}: expected NaN, got {g}");
        } else {
            assert_eq!(c.to_bits(), g.to_bits(), "index {i}: {c} != {g}");
        }
    }

    // Div: 0/0 は NaN・x/0（x!=0）は ±inf。
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 4.0], &[4]).expect("valid tensor");
    let y = Tensor::new(vec![0.0, 0.0, 0.0, 2.0], &[4]).expect("valid tensor");
    let cpu_out = cpu
        .scalar_binary(ScalarBinaryOp::Div, &x, &y)
        .expect("cpu succeeds");
    let metal_out = metal
        .scalar_binary(ScalarBinaryOp::Div, &x, &y)
        .expect("metal succeeds");
    for (i, (&c, &g)) in cpu_out
        .as_slice()
        .unwrap()
        .iter()
        .zip(metal_out.as_slice().unwrap().iter())
        .enumerate()
    {
        if c.is_nan() {
            assert!(g.is_nan(), "index {i}: expected NaN, got {g}");
        } else {
            assert_eq!(c.to_bits(), g.to_bits(), "index {i}: {c} != {g}");
        }
    }
}

/// `0^0 == 1`（IEEE 754 `powf` 規約）を CPU-Metal 両方で確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn pow_zero_pow_zero_is_one() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let a = Tensor::new(vec![0.0f32], &[1]).expect("valid tensor");
    let b = Tensor::new(vec![0.0f32], &[1]).expect("valid tensor");
    let cpu_out = cpu
        .scalar_binary(ScalarBinaryOp::Pow, &a, &b)
        .expect("cpu succeeds");
    let metal_out = metal
        .scalar_binary(ScalarBinaryOp::Pow, &a, &b)
        .expect("metal succeeds");
    assert_eq!(cpu_out.as_slice().unwrap()[0], 1.0);
    assert_eq!(metal_out.as_slice().unwrap()[0], 1.0);
}

/// 負の底に非整数指数を与えると `NaN` を返す（実数域での定義域外）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn pow_negative_base_noninteger_exponent_is_nan() {
    let metal = MetalBackendOps::new();
    let a = Tensor::new(vec![-2.0f32], &[1]).expect("valid tensor");
    let b = Tensor::new(vec![0.5f32], &[1]).expect("valid tensor");
    let out = metal
        .scalar_binary(ScalarBinaryOp::Pow, &a, &b)
        .expect("metal succeeds");
    assert!(out.as_slice().unwrap()[0].is_nan());
}

/// 空 shape（`numel == 0`）は空の結果を返す。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn scalar_op_empty_shape_returns_empty() {
    let metal = MetalBackendOps::new();
    let a = Tensor::new(Vec::<f32>::new(), &[0]).expect("valid tensor");
    let out = metal
        .scalar_unary(ScalarUnaryOp::Sqrt, &a)
        .expect("metal succeeds on empty input");
    assert_eq!(out.as_slice().unwrap().len(), 0);

    let b = Tensor::new(Vec::<f32>::new(), &[0]).expect("valid tensor");
    let out = metal
        .scalar_binary(ScalarBinaryOp::Sub, &a, &b)
        .expect("metal succeeds on empty input");
    assert_eq!(out.as_slice().unwrap().len(), 0);
}

/// 未実装 kind（`Log`〈#1708 スコープ〉・`Clamp`〈#1709 スコープ〉・
/// `Add`／`Maximum`〈いずれの sub issue にも含まれない〉）は
/// `BackendError::Unsupported` を返し、パニックしないことの回帰ガード
/// （`crate::scalar_op_source` モジュール doc「スコープ」参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn unimplemented_kinds_return_unsupported() {
    let metal = MetalBackendOps::new();
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).expect("valid tensor");
    let b = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).expect("valid tensor");

    let err = metal
        .scalar_unary(ScalarUnaryOp::Log, &a)
        .expect_err("Log is not implemented by #1707");
    assert!(matches!(err, BackendError::Unsupported(_)));

    let err = metal
        .scalar_unary(ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 }, &a)
        .expect_err("Clamp is not implemented by #1707");
    assert!(matches!(err, BackendError::Unsupported(_)));

    let err = metal
        .scalar_binary(ScalarBinaryOp::Add, &a, &b)
        .expect_err("Add is not implemented by #1707");
    assert!(matches!(err, BackendError::Unsupported(_)));

    let err = metal
        .scalar_binary(ScalarBinaryOp::Maximum, &a, &b)
        .expect_err("Maximum is not implemented by #1707");
    assert!(matches!(err, BackendError::Unsupported(_)));
}

/// 非ブロードキャスト可能形状（`[2, 2]` vs `[3]`）は
/// `BackendError::ShapeMismatch` を返す。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn scalar_binary_incompatible_shape_is_rejected() {
    let metal = MetalBackendOps::new();
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).expect("valid tensor");
    let err = metal
        .scalar_binary(ScalarBinaryOp::Sub, &a, &b)
        .expect_err("incompatible shapes must be rejected");
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}
