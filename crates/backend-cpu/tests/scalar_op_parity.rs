//! `CpuBackendOps::scalar_unary`／`scalar_binary`（イシュー #1634）の
//! 受け入れ条件テスト。`docs/scalar-op-dispatch-design.md` §6.2 対応。
//!
//! 判定式・許容誤差の独自定義は行わない。CPU 単体（同一マシン・
//! 同一 `f32` std 関数のため `rayon` 分割は数値に影響しない。
//! `crates/backend-cpu/src/scalar_elementwise.rs` モジュール doc）の
//! **bit 同一**（`f32::to_bits()` 比較。`NaN` を含む `==` の落とし穴を
//! 避ける）を判定基準とする。バックエンド間（CPU/CUDA/Metal）横断の
//! REQ-2 複合判定は #1635／#1636（GPU カーネル実装イシュー）のスコープ。
//!
//! 入力生成は `bench_harness::rng::Xorshift64Star`（決定的シード。
//! `.claude/rules/coding-rust.md`）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, ScalarBinaryOp, ScalarUnaryOp, Tensor};

fn unary_variants() -> Vec<ScalarUnaryOp> {
    vec![
        ScalarUnaryOp::Neg,
        ScalarUnaryOp::Abs,
        ScalarUnaryOp::Sqrt,
        ScalarUnaryOp::Log,
        ScalarUnaryOp::Log2,
        ScalarUnaryOp::Log10,
        ScalarUnaryOp::Sin,
        ScalarUnaryOp::Cos,
        ScalarUnaryOp::Tan,
        ScalarUnaryOp::Relu,
        ScalarUnaryOp::Exp,
        ScalarUnaryOp::Tanh,
        ScalarUnaryOp::Sigmoid,
        ScalarUnaryOp::Gelu,
        ScalarUnaryOp::GeluTanh,
        ScalarUnaryOp::Silu,
        ScalarUnaryOp::Hardswish,
        ScalarUnaryOp::LeakyRelu {
            negative_slope: 0.1,
        },
        ScalarUnaryOp::Elu { alpha: 1.0 },
        ScalarUnaryOp::Softplus {
            beta: 1.0,
            threshold: 20.0,
        },
        ScalarUnaryOp::Clamp {
            min: -0.5,
            max: 0.5,
        },
        ScalarUnaryOp::PowScalar { exponent: 2.0 },
    ]
}

fn binary_variants() -> Vec<ScalarBinaryOp> {
    vec![
        ScalarBinaryOp::Add,
        ScalarBinaryOp::Sub,
        ScalarBinaryOp::Mul,
        ScalarBinaryOp::Div,
        ScalarBinaryOp::Pow,
        ScalarBinaryOp::Maximum,
        ScalarBinaryOp::Minimum,
        ScalarBinaryOp::Gt,
        ScalarBinaryOp::Ge,
        ScalarBinaryOp::Lt,
        ScalarBinaryOp::Le,
        ScalarBinaryOp::Eq,
        ScalarBinaryOp::Ne,
    ]
}

/// `Sqrt`/`Log`/`Log2`/`Log10` は定義域が正のため、それ以外の変則的
/// 挙動（`0` 除算等）を避けつつ全 variant を同一データ生成器で試せる
/// よう、絶対値へオフセットを加えた正の範囲 `[0.1, 2.1)` を使う
/// （`Div`/`Pow` の分母・底が 0 に近すぎないようにする副次効果もある）。
fn positive_data(rng: &mut Xorshift64Star, len: usize) -> Vec<f32> {
    rng.fill_vec(len)
        .into_iter()
        .map(|v| v.abs() + 0.1)
        .collect()
}

fn bits_eq(a: &Tensor<f32>, b: &[f32]) -> bool {
    let av = a.contiguous();
    let av = av.as_slice().expect("contiguous() は常に as_slice() 可能");
    av.len() == b.len() && av.iter().zip(b).all(|(&x, &y)| x.to_bits() == y.to_bits())
}

// ---------------------------------------------------------------------
// 1. 全 unary variant × 全 binary variant: 逐次ホスト参照（op.apply()
//    の逐次適用）と bit 同一（PARALLEL_THRESHOLD 未満の小サイズ）
// ---------------------------------------------------------------------

#[test]
fn scalar_unary_all_variants_match_sequential_host_reference() {
    let ops = CpuBackendOps::new();
    let mut rng = Xorshift64Star::new(0x1634_0001);
    let len = 37; // PARALLEL_THRESHOLD 未満（逐次実行経路）
    let data = positive_data(&mut rng, len);
    let a = Tensor::new(data.clone(), &[len]).unwrap();

    for op in unary_variants() {
        let expected: Vec<f32> = data.iter().map(|&x| op.apply(x)).collect();
        let out = ops.scalar_unary(op, &a).unwrap();
        assert!(
            bits_eq(&out, &expected),
            "scalar_unary({op:?}): 逐次ホスト参照と bit 一致しない"
        );
    }
}

#[test]
fn scalar_binary_all_variants_match_sequential_host_reference() {
    let ops = CpuBackendOps::new();
    let mut rng = Xorshift64Star::new(0x1634_0002);
    let len = 41;
    let a_data = positive_data(&mut rng, len);
    let b_data = positive_data(&mut rng, len);
    let a = Tensor::new(a_data.clone(), &[len]).unwrap();
    let b = Tensor::new(b_data.clone(), &[len]).unwrap();

    for op in binary_variants() {
        let expected: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| op.apply(x, y))
            .collect();
        let out = ops.scalar_binary(op, &a, &b).unwrap();
        assert!(
            bits_eq(&out, &expected),
            "scalar_binary({op:?}): 逐次ホスト参照と bit 一致しない"
        );
    }
}

// ---------------------------------------------------------------------
// 2. PARALLEL_THRESHOLD 境界（ちょうど／±1）テストは
//    `crates/backend-cpu/src/scalar_elementwise.rs` のクレート内単体
//    テストへ移設済み（PR #1686 codex-review 指摘 P1。`PARALLEL_
//    THRESHOLD` は `pub(crate)` のため統合テストからは見えず、値の
//    複製を避けるため）。
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// 3. 非 contiguous 入力（transpose view・broadcast）が contiguous
//    参照実装と一致する
// ---------------------------------------------------------------------

#[test]
fn scalar_unary_transposed_view_matches_contiguous_reference() {
    let ops = CpuBackendOps::new();
    let mut rng = Xorshift64Star::new(0x1634_0005);
    let data = rng.fill_vec(12);
    let a = Tensor::new(data, &[3, 4]).unwrap();
    let a_t = a.transpose_2d().unwrap(); // 非 contiguous view（[4, 3]）

    let out_view = ops.scalar_unary(ScalarUnaryOp::Exp, &a_t).unwrap();
    let out_contig = ops
        .scalar_unary(ScalarUnaryOp::Exp, &a_t.contiguous())
        .unwrap();
    assert!(
        bits_eq(&out_view, out_contig.contiguous().as_slice().unwrap()),
        "scalar_unary: transpose view と contiguous 実体化が不一致"
    );
}

#[test]
fn scalar_binary_broadcast_matches_expanded_reference() {
    let ops = CpuBackendOps::new();
    // bias パターン（[2,3] + [3]）と同じ broadcast 意味論
    // （`elementwise::add` と同一の `elementwise_out_shape`）。
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::new(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    let out = ops.scalar_binary(ScalarBinaryOp::Mul, &a, &b).unwrap();
    let expected = [10.0, 40.0, 90.0, 40.0, 100.0, 180.0];
    assert!(bits_eq(&out, &expected), "scalar_binary broadcast mismatch");
}

// ---------------------------------------------------------------------
// 4. 既存 5 演算（Add/Mul/Relu/Exp/Tanh）との bit 同一（非 NaN 入力）
// ---------------------------------------------------------------------

#[test]
fn scalar_ops_match_existing_five_ops_for_non_nan_inputs() {
    let ops = CpuBackendOps::new();
    let mut rng = Xorshift64Star::new(0x1634_0006);
    let len = 53;
    let a_data = rng.fill_vec(len);
    let b_data = rng.fill_vec(len);
    let a = Tensor::new(a_data, &[len]).unwrap();
    let b = Tensor::new(b_data, &[len]).unwrap();

    let existing_add = ops.add(&a, &b).unwrap();
    let scalar_add = ops.scalar_binary(ScalarBinaryOp::Add, &a, &b).unwrap();
    assert!(bits_eq(
        &existing_add,
        scalar_add.contiguous().as_slice().unwrap()
    ));

    let existing_mul = ops.mul(&a, &b).unwrap();
    let scalar_mul = ops.scalar_binary(ScalarBinaryOp::Mul, &a, &b).unwrap();
    assert!(bits_eq(
        &existing_mul,
        scalar_mul.contiguous().as_slice().unwrap()
    ));

    for (existing, op) in [
        (ops.relu(&a).unwrap(), ScalarUnaryOp::Relu),
        (ops.exp(&a).unwrap(), ScalarUnaryOp::Exp),
        (ops.tanh(&a).unwrap(), ScalarUnaryOp::Tanh),
    ] {
        let scalar_out = ops.scalar_unary(op, &a).unwrap();
        assert!(
            bits_eq(&existing, scalar_out.contiguous().as_slice().unwrap()),
            "{op:?}: 既存 5 演算と非 NaN 入力で不一致"
        );
    }
}

/// `ScalarUnaryOp::Relu` は既存 `BackendOps::relu` と異なり `NaN` を
/// 伝播する（意図的な差異。`tensor-core::scalar_op::ScalarUnaryOp::Relu`
/// doc・`docs/scalar-op-dispatch-design.md` §3.6 参照）。
#[test]
fn scalar_relu_diverges_from_existing_relu_only_for_nan_input() {
    let ops = CpuBackendOps::new();
    let a = Tensor::new(vec![f32::NAN], &[1]).unwrap();

    let existing = ops.relu(&a).unwrap();
    assert_eq!(existing.get(&[0]), Some(0.0), "既存 relu(NaN) == 0.0 契約");

    let scalar = ops.scalar_unary(ScalarUnaryOp::Relu, &a).unwrap();
    assert!(
        scalar.get(&[0]).unwrap().is_nan(),
        "ScalarUnaryOp::Relu は NaN を伝播する契約"
    );
}

// ---------------------------------------------------------------------
// 5. NaN／-0.0／inf 入力の伝播・比較演算の 0/1 出力・Clamp(min>max)
// ---------------------------------------------------------------------

#[test]
fn scalar_unary_nan_inf_propagation() {
    let ops = CpuBackendOps::new();
    let a = Tensor::new(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY], &[3]).unwrap();

    let clamp_out = ops
        .scalar_unary(
            ScalarUnaryOp::Clamp {
                min: -1.0,
                max: 1.0,
            },
            &a,
        )
        .unwrap();
    assert!(clamp_out.get(&[0]).unwrap().is_nan(), "clamp(NaN) は NaN");
    assert_eq!(clamp_out.get(&[1]).unwrap(), 1.0, "clamp(+inf) は max");
    assert_eq!(clamp_out.get(&[2]).unwrap(), -1.0, "clamp(-inf) は min");
}

#[test]
fn scalar_binary_comparisons_return_zero_or_one_and_nan_is_not_equal() {
    let ops = CpuBackendOps::new();
    let a = Tensor::new(vec![1.0, 2.0, f32::NAN], &[3]).unwrap();
    let b = Tensor::new(vec![1.0, 1.0, f32::NAN], &[3]).unwrap();

    let eq = ops.scalar_binary(ScalarBinaryOp::Eq, &a, &b).unwrap();
    assert_eq!(eq.get(&[0]).unwrap(), 1.0);
    assert_eq!(eq.get(&[1]).unwrap(), 0.0);
    assert_eq!(eq.get(&[2]).unwrap(), 0.0, "NaN == NaN は false");

    let ne = ops.scalar_binary(ScalarBinaryOp::Ne, &a, &b).unwrap();
    assert_eq!(ne.get(&[2]).unwrap(), 1.0, "NaN != NaN は true");
}

#[test]
fn scalar_unary_clamp_min_greater_than_max_always_returns_max() {
    let ops = CpuBackendOps::new();
    let a = Tensor::new(vec![-100.0, 0.0, 100.0], &[3]).unwrap();
    let out = ops
        .scalar_unary(ScalarUnaryOp::Clamp { min: 5.0, max: 1.0 }, &a)
        .unwrap();
    for i in 0..3 {
        assert_eq!(out.get(&[i]).unwrap(), 1.0, "min>max は常に max");
    }
}

// ---------------------------------------------------------------------
// 6. shape 不一致（binary のみ。broadcast 不能な組合せ）は
//    fail-closed で `BackendError::ShapeMismatch`
// ---------------------------------------------------------------------

#[test]
fn scalar_binary_rejects_non_broadcastable_shape_mismatch() {
    let ops = CpuBackendOps::new();
    let a = Tensor::new(vec![1.0; 6], &[2, 3]).unwrap();
    let b = Tensor::new(vec![1.0; 4], &[2, 2]).unwrap();
    let err = ops.scalar_binary(ScalarBinaryOp::Add, &a, &b).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}
