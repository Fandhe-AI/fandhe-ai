//! MatMul と elementwise 5 演算（Add／Mul／Relu／Exp／Tanh）の opt-in 低精度
//! forward の自由関数（イシュー #2628・親 #2626「`compute_dtype` の Op 拡張」。
//! ルート #2499 Phase 4）。
//!
//! **facade 非公開（保留）**: 公開形（`Var::{matmul,add,mul,relu,exp,tanh}
//! _low_precision(.., dtype)` の委譲メソッド）は未承認で、承認依頼は #2677
//! （公開自体は承認後の #2678）。本モジュールは内部クレート限定の入口で、
//! `Var` に `pub` の inherent メソッドを足さない（`fft_ops` と同型）。保留は
//! `crates/facade/src/lib.rs` の `VarLowPrecisionOpsHoldDoctestGuard` と
//! `crates/facade/tests/api_surface.rs` の否定ガードが機械的に固定する
//! （`docs/autodiff-low-precision-op-extension-decision.md`。同記録は推奨案の
//! 記録であり承認記録ではない）。
//!
//! **数値契約**: 入力（f32 master）を `dtype`（`F16`／`Bf16` のみ）へ丸め、
//! `TypedOps<T>` の演算を行い、結果を 1 回だけ f32 へ昇格して `Tape` へ積む
//! （`fandhe_ai_tensor_core::low_precision` が正。`Linear`／`Conv2d`／MHA の
//! 低精度 forward と同方式）。`Var`／`Tape` 自体の dtype は f32 のまま。
//!
//! **backward 契約**: `grad.rs` は変更せず既存の f32 VJP がそのまま働く
//! （案 C: backward と master 値は f32）。`Exp`／`Tanh` は丸め済みの forward
//! 記録値を読み（straight-through）、`Mul`／`MatMul` は丸め前の master 入力、
//! `Add` は shape のみ、`Relu` は master 入力の符号でマスクする。
//!
//! **fail-closed**: `typed_ops_f16`／`typed_ops_bf16` を持たないバックエンドは
//! `AutodiffError::Backend(BackendError::Unsupported)`、`F16`／`Bf16` 以外の
//! dtype は `Backend(InvalidArgument)`。**f32 へのフォールバックも
//! ホスト計算フォールバックも持たない**（低精度を指定したのに f32 で計算
//! される無言の精度後退を避ける。CUDA／Metal は既存の `TypedOps` accessor で
//! 到達する）。失敗時はテープへノードを積まない。
//!
//! **他機構との関係**: 結果ノードは `push_eager` で値を持ち（融合連鎖
//! `FusionPlan`〈f32 固定〉に参加しない）、`TapeNode::low_precision` を立てる。
//! これにより ① activation checkpoint の解放対象外 ② `create_graph` が
//! `requires_grad` な祖先に含まれた時点で拒否（子テープでの f32 再生による
//! 精度後退を塞ぐ。`create_graph::validate_ancestors`）の 2 経路を塞ぐ。

use fandhe_ai_tensor_core::{
    BackendError, BackendOps, ScalarDType, ShapeError, Tensor, broadcast_shape,
};

use crate::error::AutodiffError;
use crate::tape::{NodeId, Op, Tape, materialize_fallible};
use crate::var::Var;

/// 単項低精度 forward の関数型（`fandhe_ai_tensor_core::*_low_precision`）。
type UnaryForward =
    fn(&dyn BackendOps, ScalarDType, &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;

/// 二項低精度 forward の関数型。
type BinaryForward = fn(
    &dyn BackendOps,
    ScalarDType,
    &Tensor<f32>,
    &Tensor<f32>,
) -> Result<Tensor<f32>, BackendError>;

/// `x` を層 1 で実体化した `Tensor<f32>` を返す（`RefCell` 借用はここで閉じる。
/// 未実体化の遅延連鎖は f32 のまま実体化され、poison 済み入力は `Err`）。
fn materialize_one<'t>(x: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = x.tape().nodes.borrow();
    Ok(materialize_fallible(&nodes, x.tape().ops(), x.node_id())?.clone())
}

/// バックエンド（実装依存の broadcast）が返した shape が事前に確定した
/// 期待 shape と一致することを検証する。
fn verify_shape(actual: &[usize], expected: &[usize]) -> Result<(), AutodiffError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: actual.to_vec(),
                rhs: expected.to_vec(),
            },
        )))
    }
}

/// 値を持つノード（`push_eager`）を積み、低精度 forward 契約フラグを立てる
/// （`Var::matmul_low_precision` と同じ事後設定パターンを 1 箇所へ集約）。
fn push_low_precision<'t>(
    tape: &'t Tape,
    op: Op,
    value: Tensor<f32>,
    expected_shape: &[usize],
) -> Result<Var<'t>, AutodiffError> {
    verify_shape(value.shape(), expected_shape)?;
    let id: NodeId = tape.push_eager(op, value);
    tape.nodes.borrow_mut()[id.0].low_precision = true;
    Ok(Var::from_raw(tape, id))
}

fn unary<'t>(
    x: &Var<'t>,
    dtype: ScalarDType,
    forward: UnaryForward,
    make_op: fn(NodeId) -> Op,
) -> Result<Var<'t>, AutodiffError> {
    let input = materialize_one(x)?;
    let value = forward(x.tape().ops(), dtype, &input).map_err(AutodiffError::Backend)?;
    let expected = x.shape();
    push_low_precision(x.tape(), make_op(x.node_id()), value, &expected)
}

fn binary<'t>(
    lhs: &Var<'t>,
    rhs: &Var<'t>,
    dtype: ScalarDType,
    forward: BinaryForward,
    make_op: fn(NodeId, NodeId) -> Op,
) -> Result<Var<'t>, AutodiffError> {
    lhs.check_same_tape(rhs)?;
    let expected = broadcast_shape(&lhs.shape(), &rhs.shape())?;
    let l = materialize_one(lhs)?;
    let r = materialize_one(rhs)?;
    let value = forward(lhs.tape().ops(), dtype, &l, &r).map_err(AutodiffError::Backend)?;
    push_low_precision(
        lhs.tape(),
        make_op(lhs.node_id(), rhs.node_id()),
        value,
        &expected,
    )
}

/// バッチ行列積の opt-in 低精度 forward（`Var::matmul_low_precision` への
/// 1 行委譲。rank≥3 の `create_graph` は非対応）。
pub fn matmul_low_precision<'t>(
    lhs: &Var<'t>,
    rhs: &Var<'t>,
    dtype: ScalarDType,
) -> Result<Var<'t>, AutodiffError> {
    lhs.matmul_low_precision(rhs, dtype)
}

/// elementwise 加算（NumPy 互換 broadcast）の opt-in 低精度 forward。
pub fn add_low_precision<'t>(
    lhs: &Var<'t>,
    rhs: &Var<'t>,
    dtype: ScalarDType,
) -> Result<Var<'t>, AutodiffError> {
    binary(
        lhs,
        rhs,
        dtype,
        fandhe_ai_tensor_core::add_low_precision,
        Op::Add,
    )
}

/// elementwise 乗算（NumPy 互換 broadcast）の opt-in 低精度 forward。
pub fn mul_low_precision<'t>(
    lhs: &Var<'t>,
    rhs: &Var<'t>,
    dtype: ScalarDType,
) -> Result<Var<'t>, AutodiffError> {
    binary(
        lhs,
        rhs,
        dtype,
        fandhe_ai_tensor_core::mul_low_precision,
        Op::Mul,
    )
}

/// ReLU の opt-in 低精度 forward（backward は master 入力の符号でマスク）。
pub fn relu_low_precision<'t>(x: &Var<'t>, dtype: ScalarDType) -> Result<Var<'t>, AutodiffError> {
    unary(
        x,
        dtype,
        fandhe_ai_tensor_core::relu_low_precision,
        Op::Relu,
    )
}

/// `exp` の opt-in 低精度 forward（f16 の表現範囲を超える出力は `+inf`。
/// backward は丸め済みの forward 記録値を読む）。
pub fn exp_low_precision<'t>(x: &Var<'t>, dtype: ScalarDType) -> Result<Var<'t>, AutodiffError> {
    unary(x, dtype, fandhe_ai_tensor_core::exp_low_precision, Op::Exp)
}

/// `tanh` の opt-in 低精度 forward（backward は丸め済みの forward 記録値を読む）。
pub fn tanh_low_precision<'t>(x: &Var<'t>, dtype: ScalarDType) -> Result<Var<'t>, AutodiffError> {
    unary(
        x,
        dtype,
        fandhe_ai_tensor_core::tanh_low_precision,
        Op::Tanh,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{low_precision_test_ops, test_ops};
    use fandhe_ai_tensor_core::{bf16, f16};

    const DTYPES: [ScalarDType; 2] = [ScalarDType::F16, ScalarDType::Bf16];

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data.to_vec(), shape).unwrap()
    }

    fn round_scalar(dtype: ScalarDType, v: f32) -> f32 {
        match dtype {
            ScalarDType::F16 => f16::from_f32(v).to_f32(),
            ScalarDType::Bf16 => bf16::from_f32(v).to_f32(),
            other => panic!("test: unsupported dtype {other:?}"),
        }
    }

    fn round_t(dtype: ScalarDType, x: &Tensor<f32>) -> Tensor<f32> {
        let data: Vec<f32> = x
            .host_slice()
            .iter()
            .map(|&v| round_scalar(dtype, v))
            .collect();
        t(&data, x.shape())
    }

    fn bits(x: &Tensor<f32>) -> Vec<u32> {
        x.host_slice().iter().map(|v| v.to_bits()).collect()
    }

    fn value_of(v: &Var<'_>) -> Tensor<f32> {
        materialize_one(v).unwrap()
    }

    fn lp_tape() -> Tape {
        Tape::new_with_ops(low_precision_test_ops())
    }

    /// 丸めオラクル: `round(f32_op(round(x)))`（`TestOps` の f32 演算を使う。
    /// 被検実装〈`tensor-core::low_precision`〉とは独立の経路）。
    fn oracle_unary(
        dtype: ScalarDType,
        x: &Tensor<f32>,
        f: fn(&Tensor<f32>) -> Tensor<f32>,
    ) -> Tensor<f32> {
        round_t(dtype, &f(&round_t(dtype, x)))
    }

    fn oracle_binary(
        dtype: ScalarDType,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        f: fn(&Tensor<f32>, &Tensor<f32>) -> Tensor<f32>,
    ) -> Tensor<f32> {
        round_t(dtype, &f(&round_t(dtype, a), &round_t(dtype, b)))
    }

    fn x23() -> Tensor<f32> {
        t(&[-1.5, -0.1, 0.0, 0.1, 0.7, 3.3], &[2, 3])
    }

    fn y3() -> Tensor<f32> {
        t(&[0.1, -0.7, 2.3], &[3])
    }

    // --- P1: forward が丸めオラクルと bit 一致 ---

    #[test]
    fn unary_forward_matches_rounding_oracle() {
        type Case = (
            for<'a, 'b> fn(&'b Var<'a>, ScalarDType) -> Result<Var<'a>, AutodiffError>,
            fn(&Tensor<f32>) -> Tensor<f32>,
        );
        let cases: [Case; 3] = [
            (relu_low_precision, crate::eval::relu),
            (exp_low_precision, crate::eval::exp),
            (tanh_low_precision, crate::eval::tanh),
        ];
        for dtype in DTYPES {
            for (lp, f32_op) in cases {
                let tape = lp_tape();
                let x = tape.var(&x23());
                let y = lp(&x, dtype).unwrap();
                assert_eq!(
                    bits(&value_of(&y)),
                    bits(&oracle_unary(dtype, &x23(), f32_op)),
                    "{dtype:?}"
                );
            }
        }
    }

    #[test]
    fn binary_forward_matches_rounding_oracle_with_broadcast() {
        type Case = (
            for<'a, 'b> fn(&'b Var<'a>, &'b Var<'a>, ScalarDType) -> Result<Var<'a>, AutodiffError>,
            fn(&Tensor<f32>, &Tensor<f32>) -> Tensor<f32>,
        );
        let cases: [Case; 2] = [
            (add_low_precision, crate::eval::add),
            (mul_low_precision, crate::eval::mul),
        ];
        for dtype in DTYPES {
            for (lp, f32_op) in cases {
                let tape = lp_tape();
                let x = tape.var(&x23());
                let y = tape.var(&y3());
                // bias パターン `[2,3] + [3]` と同形。
                let z = lp(&x, &y, dtype).unwrap();
                assert_eq!(z.shape(), vec![2, 3]);
                assert_eq!(
                    bits(&value_of(&z)),
                    bits(&oracle_binary(dtype, &x23(), &y3(), f32_op)),
                    "{dtype:?}"
                );
            }
        }
    }

    #[test]
    fn matmul_forward_matches_rounding_oracle() {
        let a = x23();
        let b = t(&[0.3, -0.2, 1.1, 0.05, -0.6, 0.9], &[3, 2]);
        for dtype in DTYPES {
            let tape = lp_tape();
            let va = tape.var(&a);
            let vb = tape.var(&b);
            let c = matmul_low_precision(&va, &vb, dtype).unwrap();
            assert_eq!(
                bits(&value_of(&c)),
                bits(&oracle_binary(dtype, &a, &b, crate::eval::matmul)),
                "{dtype:?}"
            );
        }
    }

    // --- P2: backward（master／丸め済み記録値の読み分け） ---

    fn grad_of(tape: &Tape, loss: &Var<'_>, v: &Var<'_>) -> Tensor<f32> {
        let grads = tape.backward(loss).unwrap();
        grads
            .get(v)
            .unwrap()
            .expect("勾配が到達するはず")
            .contiguous()
    }

    #[test]
    fn exp_backward_reads_rounded_forward_value() {
        for dtype in DTYPES {
            let tape = lp_tape();
            let x = tape.var(&x23());
            let y = exp_low_precision(&x, dtype).unwrap();
            let loss = y.sum(None).unwrap();
            let dx = grad_of(&tape, &loss, &x);
            // upstream = 1。丸め済み forward 値そのもの（master の exp とは異なる）。
            assert_eq!(bits(&dx), bits(&value_of(&y)), "{dtype:?}");
            assert_ne!(
                bits(&dx),
                bits(&crate::eval::exp(&x23())),
                "{dtype:?}: master の exp とは区別できる入力のはず"
            );
        }
    }

    #[test]
    fn tanh_backward_reads_rounded_forward_value() {
        for dtype in DTYPES {
            let tape = lp_tape();
            let x = tape.var(&x23());
            let y = tanh_low_precision(&x, dtype).unwrap();
            let loss = y.sum(None).unwrap();
            let dx = grad_of(&tape, &loss, &x);
            let v = value_of(&y);
            let expected: Vec<f32> = v.host_slice().iter().map(|&v| 1.0 - v * v).collect();
            assert_eq!(bits(&dx), bits(&t(&expected, &[2, 3])), "{dtype:?}");
        }
    }

    #[test]
    fn mul_backward_reads_master_inputs() {
        assert_ne!(f16::from_f32(0.1).to_f32(), 0.1f32);
        let a = t(&[0.1, 0.2, -0.3, 0.4], &[2, 2]);
        let b = t(&[0.7, -0.1, 0.3, 0.9], &[2, 2]);
        for dtype in DTYPES {
            let tape = lp_tape();
            let va = tape.var(&a);
            let vb = tape.var(&b);
            let y = mul_low_precision(&va, &vb, dtype).unwrap();
            let loss = y.sum(None).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let da = grads.get(&va).unwrap().unwrap().contiguous();
            let db = grads.get(&vb).unwrap().unwrap().contiguous();
            assert_eq!(bits(&da), bits(&b), "{dtype:?}: dA は b の master 値");
            assert_eq!(bits(&db), bits(&a), "{dtype:?}: dB は a の master 値");
        }
    }

    #[test]
    fn add_backward_is_ones_and_reduces_bias_pattern() {
        for dtype in DTYPES {
            let tape = lp_tape();
            let x = tape.var(&x23());
            let b = tape.var(&y3());
            let y = add_low_precision(&x, &b, dtype).unwrap();
            let loss = y.sum(None).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let dx = grads.get(&x).unwrap().unwrap().contiguous();
            let db = grads.get(&b).unwrap().unwrap().contiguous();
            assert_eq!(dx.host_slice().as_ref(), &[1.0f32; 6], "{dtype:?}");
            assert_eq!(db.host_slice().as_ref(), &[2.0f32; 3], "{dtype:?}");
        }
    }

    #[test]
    fn relu_backward_masks_by_master_input_sign() {
        // 1e-9 は f16 で 0 へ丸まる（forward は 0）が、master 入力は正のため
        // 勾配は流れる（決定記録 §3.4）。
        let x = t(&[-1.0, 0.0, 1e-9, 2.0], &[2, 2]);
        for dtype in DTYPES {
            let tape = lp_tape();
            let vx = tape.var(&x);
            let y = relu_low_precision(&vx, dtype).unwrap();
            let loss = y.sum(None).unwrap();
            let dx = grad_of(&tape, &loss, &vx);
            assert_eq!(
                dx.host_slice().as_ref(),
                &[0.0f32, 0.0, 1.0, 1.0],
                "{dtype:?}"
            );
        }
        let tape = lp_tape();
        let vx = tape.var(&x);
        let y = relu_low_precision(&vx, ScalarDType::F16).unwrap();
        assert_eq!(
            value_of(&y).host_slice()[2],
            0.0,
            "f16 では 1e-9 が 0 へ丸まる"
        );
    }

    #[test]
    fn matmul_backward_reads_master_inputs() {
        let a = t(&[0.1, 0.2, -0.3, 0.4, 0.5, -0.7], &[2, 3]);
        let b = t(&[0.3, -0.2, 1.1, 0.05, -0.6, 0.9], &[3, 2]);
        for dtype in DTYPES {
            let tape = lp_tape();
            let va = tape.var(&a);
            let vb = tape.var(&b);
            let c = matmul_low_precision(&va, &vb, dtype).unwrap();
            let loss = c.sum(None).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let da = grads.get(&va).unwrap().unwrap().contiguous();
            let db = grads.get(&vb).unwrap().unwrap().contiguous();
            let bd = b.host_slice();
            let ad = a.host_slice();
            // upstream = 1: dA[i,k] = sum_j B[k,j]、dB[k,j] = sum_i A[i,k]（master 値）。
            let mut ea = Vec::new();
            for _i in 0..2 {
                for k in 0..3 {
                    ea.push(bd[k * 2] + bd[k * 2 + 1]);
                }
            }
            let mut eb = Vec::new();
            for k in 0..3 {
                for _j in 0..2 {
                    eb.push(ad[k] + ad[3 + k]);
                }
            }
            assert_eq!(bits(&da), bits(&t(&ea, &[2, 3])), "{dtype:?}");
            assert_eq!(bits(&db), bits(&t(&eb, &[3, 2])), "{dtype:?}");
        }
    }

    // --- P3: fail-closed・ノード属性 ---

    fn tape_len(tape: &Tape) -> usize {
        tape.nodes.borrow().len()
    }

    fn call_all(tape: &Tape, dtype: ScalarDType) -> Vec<Result<(), AutodiffError>> {
        let x = tape.var(&x23());
        let y = tape.var(&y3());
        let m = tape.var(&t(&[1.0; 6], &[3, 2]));
        vec![
            matmul_low_precision(&x, &m, dtype).map(drop),
            add_low_precision(&x, &y, dtype).map(drop),
            mul_low_precision(&x, &y, dtype).map(drop),
            relu_low_precision(&x, dtype).map(drop),
            exp_low_precision(&x, dtype).map(drop),
            tanh_low_precision(&x, dtype).map(drop),
        ]
    }

    #[test]
    fn backend_without_typed_ops_is_unsupported_and_pushes_no_node() {
        for dtype in DTYPES {
            let tape = Tape::new_with_ops(test_ops());
            let before = tape_len(&tape);
            let results = call_all(&tape, dtype);
            // 6 呼び出しとも Unsupported。葉 3 つ（x・y・m）以外のノードは増えない。
            for r in results {
                assert!(matches!(
                    r.unwrap_err(),
                    AutodiffError::Backend(BackendError::Unsupported(_))
                ));
            }
            assert_eq!(tape_len(&tape), before + 3);
        }
    }

    #[test]
    fn non_low_precision_dtype_is_rejected_as_invalid_argument() {
        for dtype in [ScalarDType::F32, ScalarDType::F64] {
            let tape = lp_tape();
            for r in call_all(&tape, dtype) {
                assert!(matches!(
                    r.unwrap_err(),
                    AutodiffError::Backend(BackendError::InvalidArgument(_))
                ));
            }
        }
    }

    #[test]
    fn cross_tape_operands_are_rejected() {
        let t1 = lp_tape();
        let t2 = lp_tape();
        let a = t1.var(&x23());
        let b = t2.var(&x23());
        assert!(matches!(
            add_low_precision(&a, &b, ScalarDType::F16).unwrap_err(),
            AutodiffError::TapeMismatch
        ));
        assert!(matches!(
            mul_low_precision(&a, &b, ScalarDType::F16).unwrap_err(),
            AutodiffError::TapeMismatch
        ));
    }

    #[test]
    fn result_nodes_are_eager_flagged_and_outside_fusion_chain() {
        let tape = lp_tape();
        let x = tape.var(&x23());
        let y = tape.var(&y3());
        let low = [
            add_low_precision(&x, &y, ScalarDType::F16).unwrap(),
            mul_low_precision(&x, &y, ScalarDType::F16).unwrap(),
            relu_low_precision(&x, ScalarDType::F16).unwrap(),
            exp_low_precision(&x, ScalarDType::F16).unwrap(),
            tanh_low_precision(&x, ScalarDType::F16).unwrap(),
        ];
        let normal = x.add(&y).unwrap();
        let nodes = tape.nodes.borrow();
        for v in &low {
            let n = &nodes[v.node_id().0];
            assert!(n.low_precision);
            assert!(n.value.get().is_some());
            assert_eq!(n.lazy_chain_size, 0);
        }
        assert!(!nodes[normal.node_id().0].low_precision);
    }

    #[test]
    fn lazy_f32_chain_input_is_materialized_in_f32_and_low_node_stays_out_of_fusion() {
        let tape = lp_tape();
        let x = tape.var(&x23());
        let lazy = x.add(&x).unwrap();
        let low = relu_low_precision(&lazy, ScalarDType::F16).unwrap();
        let doubled = crate::eval::add(&x23(), &x23());
        assert_eq!(
            bits(&value_of(&low)),
            bits(&oracle_unary(ScalarDType::F16, &doubled, crate::eval::relu))
        );
        // 低精度ノードを入力にした後続の遅延連鎖を実体化しても、低精度ノードの値は不変。
        let before = bits(&value_of(&low));
        let tail = low.exp().add(&low).unwrap();
        let _ = value_of(&tail);
        assert_eq!(before, bits(&value_of(&low)));
        assert!(tape.nodes.borrow()[low.node_id().0].low_precision);
    }

    #[test]
    fn checkpoint_region_keeps_low_precision_values() {
        let tape = lp_tape();
        let x = tape.var(&t(&[0.1, 0.2, -0.3, 0.4], &[2, 2]));
        let mut ids = Vec::new();
        let out = tape
            .checkpoint(|| {
                let a = add_low_precision(&x, &x, ScalarDType::F16)?;
                let b = relu_low_precision(&a, ScalarDType::F16)?;
                let c = exp_low_precision(&b, ScalarDType::F16)?;
                ids.push(a.node_id());
                ids.push(b.node_id());
                Ok(c)
            })
            .unwrap();
        let _ = out;
        let nodes = tape.nodes.borrow();
        for id in ids {
            assert!(
                nodes[id.0].value.get().is_some(),
                "値が解放されてはならない"
            );
            assert!(!nodes[id.0].recompute);
        }
    }

    // --- create_graph との関係（イシュー #2628） ---

    #[test]
    fn create_graph_rejects_every_low_precision_op_ancestor() {
        let x = x23();
        for build in 0..5 {
            let tape = lp_tape();
            let child = lp_tape();
            let vx = tape.var(&x);
            let vy = tape.var(&y3());
            let d = ScalarDType::F16;
            let node = match build {
                0 => add_low_precision(&vx, &vy, d),
                1 => mul_low_precision(&vx, &vy, d),
                2 => relu_low_precision(&vx, d),
                3 => exp_low_precision(&vx, d),
                _ => tanh_low_precision(&vx, d),
            }
            .unwrap();
            let loss = node.sum(None).unwrap();
            let err = tape.backward_create_graph(&loss, &child).unwrap_err();
            assert!(matches!(err, AutodiffError::Backward(_)), "case {build}");
            assert!(child.is_empty(), "case {build}");
        }
    }

    #[test]
    fn create_graph_accepts_low_precision_node_without_requires_grad() {
        let tape = lp_tape();
        let child = lp_tape();
        let c = tape.var_no_grad(&x23());
        let lp = exp_low_precision(&c, ScalarDType::F16).unwrap();
        let w = tape.var(&x23());
        let loss = lp.mul(&w).unwrap().sum(None).unwrap();
        let recorded = value_of(&lp);
        tape.backward_create_graph(&loss, &child)
            .expect("requires_grad == false の低精度ノードは定数葉として写せる");
        let found = child.nodes.borrow().iter().any(|n| {
            n.value
                .get()
                .is_some_and(|v| v.shape() == recorded.shape() && bits(v) == bits(&recorded))
        });
        assert!(
            found,
            "子テープに丸め済みの記録値が定数葉として存在するはず"
        );
    }
}
