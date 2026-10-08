//! `functional_ops` の double-VJP 版 `jvp`／`jacfwd` の受入テスト（イシュー #2940・親 #2939）。
//!
//! 契約の正は `docs/autodiff-functional-transforms-design.md` §8・§19・§23。比較先は
//! `jacobian_ops::jacobian`（facade の `Tape::jacobian` はこれへ 1 行委譲するだけで、autodiff から
//! facade へは依存できない）。判定は REQ-2 統一複合判定を `tests/common/mod.rs` の共有定数・
//! `req2_close` 経由でのみ行い、閾値は直書きしない。
//!
//! `pub(crate)` の内部実装なので統合テスト（`tests/`）からは呼べず、`#[cfg(test)]` の単体テストで
//! 検証する（#2880 の `double_vjp_feasibility_tests` と同じ形）。テスト関数・ヘルパーの名前は
//! `jvp`／`jacfwd` と完全一致させない（facade `api_surface` の宣言インベントリが
//! `crates/*/src` 全体を走査するため）。入力はキンク（0・clamp 境界・タイ）から離す。

use std::sync::Arc;

use fandhe_ai_tensor_core::{ScalarBinaryOp, ShapeError, Tensor};

// REQ-2 判定・共通ヘルパーは #2880 の単体テストが `#[path]` で取り込んだ単一の定義元を共用する
// （同一ファイルの二重取り込みは clippy::duplicate_mod になるため）。
use crate::double_vjp_feasibility_tests::common;
use crate::functional_ops::{jacfwd, jvp};
use crate::jacobian_ops::jacobian;
use crate::test_support::test_ops;
use crate::{AutodiffError, CustomFunction, Tape, Var};

fn new_tape() -> Tape {
    Tape::new_with_ops(test_ops())
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test: shape とデータ長は一致させている")
}

fn seq(n: usize, seed: f32) -> Vec<f32> {
    common::det_seq(n, seed)
}

fn pos(n: usize, seed: f32) -> Vec<f32> {
    common::det_pos(n, seed)
}

fn c<'t>(tape: &'t Tape, data: Vec<f32>, shape: &[usize]) -> Var<'t> {
    tape.var_no_grad(&t(data, shape))
}

type Built<'t> = Result<Var<'t>, AutodiffError>;

/// `build` で作った出力に対し、別テープの `jacobian` を基準に `jvp`（`J·v`）と `jacfwd`（`J`）を
/// 統一複合判定で全要素突合する。
fn check<F>(shape: &[usize], xv: Vec<f32>, ctx: &str, build: F)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let n: usize = shape.iter().product();
    let vv = seq(n, 5.0);

    let tb = new_tape();
    let xb = tb.var(&t(xv.clone(), shape));
    let yb = build(&tb, &xb).unwrap_or_else(|e| panic!("{ctx}: build 失敗: {e:?}"));
    let out_shape = yb.shape();
    let m: usize = out_shape.iter().product();
    let jac = jacobian(&tb, &yb, &xb).unwrap();
    let jh = jac.host_slice().into_owned();

    // jvp
    let ta = new_tape();
    let xa = ta.var(&t(xv.clone(), shape));
    let ya = build(&ta, &xa).unwrap();
    let child = new_tape();
    let got = jvp(&ta, &ya, &xa, &t(vv.clone(), shape), &child)
        .unwrap_or_else(|e| panic!("{ctx}: jvp 失敗: {e:?}"));
    assert_eq!(got.shape(), out_shape.as_slice(), "{ctx}: jvp shape");
    let expected = common::jacobian_times_vector_f64(&jh, m, n, &vv);
    common::assert_all_req2_close(&got.host_slice(), &expected, &format!("{ctx} jvp"));

    // jacfwd
    let tc = new_tape();
    let xc = tc.var(&t(xv, shape));
    let yc = build(&tc, &xc).unwrap();
    let child = new_tape();
    let full =
        jacfwd(&tc, &yc, &xc, &child).unwrap_or_else(|e| panic!("{ctx}: jacfwd 失敗: {e:?}"));
    assert_eq!(full.shape(), jac.shape(), "{ctx}: jacfwd shape");
    let jh64: Vec<f64> = jh.iter().map(|&v| f64::from(v)).collect();
    common::assert_all_req2_close(&full.host_slice(), &jh64, &format!("{ctx} jacfwd"));
}

// ------------------------------------------------------------ J1: 閉形式

#[test]
fn j1_closed_forms_have_no_second_order_term() {
    let xv = seq(5, 1.0);
    let vv = seq(5, 5.0);
    let run = |build: &dyn for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>| -> Vec<f32> {
        let tape = new_tape();
        let x = tape.var(&t(xv.clone(), &[5]));
        let y = build(&tape, &x).unwrap();
        let child = new_tape();
        jvp(&tape, &y, &x, &t(vv.clone(), &[5]), &child)
            .unwrap()
            .host_slice()
            .into_owned()
    };
    let got = run(&|_, x| x.mul(x));
    let e: Vec<f64> = (0..5)
        .map(|i| 2.0 * f64::from(xv[i]) * f64::from(vv[i]))
        .collect();
    common::assert_all_req2_close(&got, &e, "x*x");

    let got = run(&|_, x| Ok(x.tanh()));
    let e: Vec<f64> = (0..5)
        .map(|i| {
            let th = f64::from(xv[i]).tanh();
            (1.0 - th * th) * f64::from(vv[i])
        })
        .collect();
    common::assert_all_req2_close(&got, &e, "tanh");

    let got = run(&|_, x| Ok(x.relu()));
    let e: Vec<f64> = (0..5)
        .map(|i| if xv[i] > 0.0 { f64::from(vv[i]) } else { 0.0 })
        .collect();
    common::assert_all_req2_close(&got, &e, "relu");

    // スカラー出力 sum(x) → Σ v
    let got = run(&|_, x| x.sum(None));
    let e = [vv.iter().map(|&v| f64::from(v)).sum::<f64>()];
    common::assert_all_req2_close(&got, &e, "sum");

    // y = W x（線形）→ W v
    let wv = seq(12, 3.0);
    let tape = new_tape();
    let x = tape.var(&t(xv[..4].to_vec(), &[4, 1]));
    let w = tape.var_no_grad(&t(wv.clone(), &[3, 4]));
    let y = w.matmul(&x).unwrap();
    let child = new_tape();
    let got = jvp(&tape, &y, &x, &t(vv[..4].to_vec(), &[4, 1]), &child).unwrap();
    let e: Vec<f64> = (0..3)
        .map(|r| {
            (0..4)
                .map(|k| f64::from(wv[r * 4 + k]) * f64::from(vv[k]))
                .sum()
        })
        .collect();
    common::assert_all_req2_close(&got.host_slice(), &e, "Wx");
}

// ------------------------------------------------------------ J2・J3: jacobian との一致

#[test]
fn j2_binary_and_activations() {
    let xv = seq(6, 1.0);
    check(&[2, 3], xv.clone(), "add const", |tp, x| {
        x.add(&c(tp, seq(6, 2.0), &[2, 3]))
    });
    check(&[3], seq(3, 1.0), "add bias(x が [n])", |tp, x| {
        c(tp, seq(6, 2.0), &[2, 3]).add(x)
    });
    check(&[2, 3], xv.clone(), "add [m,n]+[n]", |tp, x| {
        x.add(&c(tp, seq(3, 4.0), &[3]))
    });
    check(&[6], xv.clone(), "x*x", |_, x| x.mul(x));
    check(&[6], xv.clone(), "relu", |_, x| Ok(x.relu()));
    check(&[6], xv.clone(), "exp", |_, x| Ok(x.exp()));
    check(&[6], xv.clone(), "tanh", |_, x| Ok(x.tanh()));
    check(&[6], xv.clone(), "sigmoid", |_, x| Ok(x.sigmoid()));
    check(&[6], xv.clone(), "relu(tanh)", |_, x| Ok(x.tanh().relu()));
    check(&[6], xv.clone(), "tanh(x)*x+exp(x)", |_, x| {
        x.tanh().mul(x)?.add(&x.exp())
    });
    check(&[2, 3], xv, "mlp", |tp, x| {
        let w1 = c(tp, seq(12, 4.0), &[3, 4]);
        let w2 = c(tp, seq(8, 6.0), &[4, 2]);
        x.matmul(&w1)?.relu().matmul(&w2)
    });
}

#[test]
fn j2_reductions_views_and_matmul() {
    let xv = seq(6, 1.0);
    check(&[2, 3], xv.clone(), "sum all", |_, x| x.sum(None));
    check(&[2, 3], xv.clone(), "sum dim", |_, x| x.sum(Some(1)));
    check(&[2, 3], xv.clone(), "mean all", |_, x| x.mean(None));
    check(&[2, 3], xv.clone(), "mean dim", |_, x| x.mean(Some(0)));
    check(&[2, 3], xv.clone(), "reshape", |_, x| x.reshape(&[3, 2]));
    check(&[3], seq(3, 2.0), "broadcast_to", |_, x| {
        x.broadcast_to(&[2, 3])
    });
    check(&[2, 3], xv.clone(), "transpose", |_, x| x.transpose(0, 1));
    check(&[2, 3], xv.clone(), "permute", |_, x| x.permute(&[1, 0]));
    check(&[2, 3], xv.clone(), "narrow", |_, x| x.narrow(1, 1, 2));
    check(&[2, 3], xv.clone(), "cat const", |tp, x| {
        Var::cat(&[*x, c(tp, seq(6, 2.0), &[2, 3])], 1)
    });
    check(&[2, 3], xv.clone(), "cat duplicate input", |_, x| {
        Var::cat(&[*x, *x], 0)
    });
    check(&[2, 3], xv.clone(), "permute→contiguous", |_, x| {
        x.permute(&[1, 0])?.contiguous()
    });
    check(&[2, 3], xv, "matmul x left", |tp, x| {
        x.matmul(&c(tp, seq(12, 2.0), &[3, 4]))
    });
    check(&[3, 4], seq(12, 1.0), "matmul x right", |tp, x| {
        c(tp, seq(6, 2.0), &[2, 3]).matmul(x)
    });
    check(&[3, 3], seq(9, 1.0), "matmul x·x", |_, x| x.matmul(x));
}

#[test]
fn j2_piecewise_linear_ops() {
    let xv = seq(6, 1.0);
    let mask = Tensor::new(vec![true, false, true, true, false, false], &[2, 3]).unwrap();
    check(&[2, 3], xv.clone(), "where a=x", |tp, x| {
        Var::where_cond(&mask, x, &c(tp, seq(6, 2.0), &[2, 3]))
    });
    check(&[2, 3], xv.clone(), "where b=x", |tp, x| {
        Var::where_cond(&mask, &c(tp, seq(6, 2.0), &[2, 3]), x)
    });
    check(&[2, 3], xv.clone(), "where a=x,b=x*x", |_, x| {
        Var::where_cond(&mask, x, &x.mul(x)?)
    });
    check(&[6], xv.clone(), "abs", |_, x| x.abs());
    check(&[6], xv.clone(), "leaky_relu", |_, x| x.leaky_relu(0.1));
    check(&[6], xv.clone(), "clamp", |_, x| x.clamp(-0.8, 0.8));
    for (op, name) in [
        (ScalarBinaryOp::Maximum, "maximum"),
        (ScalarBinaryOp::Minimum, "minimum"),
    ] {
        check(&[6], xv.clone(), &format!("{name} (x, c)"), |tp, x| {
            x.scalar_binary(&k(tp), op)
        });
        check(&[6], xv.clone(), &format!("{name} (x, x²)"), |_, x| {
            x.scalar_binary(&x.mul(x)?, op)
        });
    }
}

#[test]
fn j2_scalar_unary_and_binary() {
    let xv = seq(6, 1.0);
    let pv = pos(6, 1.0);
    check(&[6], xv.clone(), "neg", |_, x| x.neg());
    check(&[6], pv.clone(), "sqrt", |_, x| x.sqrt());
    check(&[6], pv.clone(), "log", |_, x| x.log());
    check(&[6], xv.clone(), "sin", |_, x| x.sin());
    check(&[6], xv.clone(), "silu", |_, x| x.silu());
    check(&[6], xv.clone(), "softplus", |_, x| x.softplus(1.0, 20.0));
    check(&[6], pv.clone(), "pow_scalar", |_, x| x.pow_scalar(3.0));
    check(&[6], xv.clone(), "sub x-c", |tp, x| x.sub(&k(tp)));
    check(&[6], xv.clone(), "sub c-x", |tp, x| k(tp).sub(x));
    check(&[6], xv, "div x/c", |tp, x| {
        x.div(&c(tp, pos(6, 2.0), &[6]))
    });
    check(&[6], pv, "pow base=x", |tp, x| {
        x.pow(&c(tp, vec![2.0, 1.5, 3.0, 0.5, 2.5, 1.0], &[6]))
    });
}

#[test]
fn j3_boundaries_rank0_nonsquare_and_zero_jacobian() {
    // rank 0 の入力・出力
    check(&[], vec![0.7], "scalar tanh", |_, x| Ok(x.tanh()));
    check(&[], vec![0.7], "scalar x*x", |_, x| x.mul(x));
    check(&[2, 3], seq(6, 1.0), "rank0 out (sum)", |_, x| x.sum(None));
    // m ≠ n の非正方形
    check(&[2, 3], seq(6, 1.0), "nonsquare [4]←[2,3]", |_, x| {
        x.narrow(1, 0, 2)?.contiguous()?.reshape(&[4])
    });
    check(&[5], seq(5, 2.0), "nonsquare [2,3]←[5]", |tp, x| {
        let w = c(tp, seq(30, 4.0), &[5, 6]);
        x.reshape(&[1, 5])?.matmul(&w)?.reshape(&[2, 3])
    });
    // J ≡ 0 の比較系は全ゼロ
    check(&[6], seq(6, 1.0), "gt", |tp, x| x.gt(&k(tp)));
    check(&[6], seq(6, 1.0), "ne", |tp, x| x.ne(&k(tp)));
}

fn k<'t>(tp: &'t Tape) -> Var<'t> {
    c(tp, seq(6, 2.0), &[6])
}

// ------------------------------------------------------------ J4: 入口検査（fail-closed）

#[test]
fn j4_entry_checks_leave_tapes_unchanged() {
    let tape = new_tape();
    let x = tape.var(&t(seq(3, 1.0), &[3]));
    let y = x.tanh();
    let tg = t(seq(3, 5.0), &[3]);
    let child = new_tape();
    let len = tape.len();
    let untouched = |tape: &Tape, child: &Tape, ctx: &str| {
        assert_eq!(tape.len(), len, "{ctx}: 親テープが変化した");
        assert!(child.is_empty(), "{ctx}: 子テープが変化した");
    };

    // 別テープの Var
    let other = new_tape();
    let ox = other.var(&t(seq(3, 1.0), &[3]));
    let e = jvp(&tape, &y, &ox, &tg, &child).unwrap_err();
    assert!(matches!(e, AutodiffError::TapeMismatch), "{e:?}");
    let e = jacfwd(&tape, &ox.tanh(), &x, &child).unwrap_err();
    assert!(matches!(e, AutodiffError::TapeMismatch), "{e:?}");
    untouched(&tape, &child, "別テープ");

    // 追跡なしの input
    let nx = tape.var_no_grad(&t(seq(3, 1.0), &[3]));
    let len_nx = tape.len();
    let e = jvp(&tape, &y, &nx, &tg, &child).unwrap_err();
    assert!(
        matches!(e, AutodiffError::GradientTrackingDisabled),
        "{e:?}"
    );
    let e = jacfwd(&tape, &y, &nx, &child).unwrap_err();
    assert!(
        matches!(e, AutodiffError::GradientTrackingDisabled),
        "{e:?}"
    );
    assert_eq!(tape.len(), len_nx);
    assert!(child.is_empty());

    // tangent の shape 不一致（ブロードキャスト不可）
    for bad in [
        t(vec![1.0], &[1]),
        t(seq(6, 1.0), &[6]),
        t(seq(3, 1.0), &[3, 1]),
    ] {
        let e = jvp(&tape, &y, &x, &bad, &child).unwrap_err();
        assert!(
            matches!(e, AutodiffError::Shape(ShapeError::ShapeMismatch { .. })),
            "{e:?}"
        );
    }
    assert_eq!(tape.len(), len_nx);
    assert!(child.is_empty());
}

#[test]
fn j4_untracked_output_and_empty_return_zeros_without_touching_tapes() {
    let tape = new_tape();
    let x = tape.var(&t(seq(3, 1.0), &[3]));
    let ny = tape.var_no_grad(&t(seq(4, 2.0), &[4]));
    let len = tape.len();
    let child = new_tape();
    let tg = t(seq(3, 5.0), &[3]);

    let r = jvp(&tape, &ny, &x, &tg, &child).unwrap();
    assert_eq!(r.shape(), &[4]);
    assert!(r.host_slice().iter().all(|&v| v == 0.0));
    let r = jacfwd(&tape, &ny, &x, &child).unwrap();
    assert_eq!(r.shape(), &[4, 3]);
    assert!(r.host_slice().iter().all(|&v| v == 0.0));
    assert_eq!(tape.len(), len);
    assert!(child.is_empty());
}

// ------------------------------------------------------------ J5: 非対象 Op の拒否

struct IdentityCustomFn;

impl CustomFunction for IdentityCustomFn {
    fn name(&self) -> &str {
        "identity_custom_fn"
    }

    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }

    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok(inputs[0].contiguous())
    }

    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        if !requires_grad[0] {
            return Ok(vec![None]);
        }
        Ok(vec![Some(upstream.contiguous())])
    }
}

/// `jvp`・`jacfwd` が既存の `Err(Backward)` で拒否し、子テープが空のまま、親テープが
/// ちょうど 2 ノード（`u`・`mul`）増えることを確かめる。
fn assert_rejected_both<F>(shape: &[usize], ctx: &str, build: F)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let n: usize = shape.iter().product();
    for which in 0..2 {
        let tape = new_tape();
        let x = tape.var(&t(seq(n, 1.0), shape));
        let y = build(&tape, &x).unwrap();
        let child = new_tape();
        let len = tape.len();
        let err = if which == 0 {
            jvp(&tape, &y, &x, &t(seq(n, 5.0), shape), &child).unwrap_err()
        } else {
            jacfwd(&tape, &y, &x, &child).unwrap_err()
        };
        assert!(matches!(err, AutodiffError::Backward(_)), "{ctx}: {err:?}");
        assert!(child.is_empty(), "{ctx}: 子テープが無変更でない");
        assert_eq!(tape.len(), len + 2, "{ctx}: 親テープの増分は u と mul の 2");
    }
}

#[test]
fn j5_unsupported_ops_are_rejected_fail_closed() {
    assert_rejected_both(&[2, 3], "max", |_, x| x.max(None));
    assert_rejected_both(&[2, 2, 2], "matmul rank3", |tp, x| {
        x.matmul(&c(tp, seq(8, 2.0), &[2, 2, 2]))
    });
    assert_rejected_both(&[3], "custom", |tp, x| {
        tp.custom(Arc::new(IdentityCustomFn), &[*x])
    });
    assert_rejected_both(&[4], "gelu", |_, x| x.gelu());
    assert_rejected_both(&[4], "selu", |_, x| x.selu());
}

#[test]
fn j5_non_empty_or_same_tape_child_is_rejected_unchanged() {
    // 非空の子テープ
    let tape = new_tape();
    let x = tape.var(&t(seq(3, 1.0), &[3]));
    let y = x.tanh();
    let child = new_tape();
    let _ = child.var(&t(vec![1.0], &[1]));
    let clen = child.len();
    let e = jvp(&tape, &y, &x, &t(seq(3, 5.0), &[3]), &child).unwrap_err();
    assert!(matches!(e, AutodiffError::Backward(_)), "{e:?}");
    let e = jacfwd(&tape, &y, &x, &child).unwrap_err();
    assert!(matches!(e, AutodiffError::Backward(_)), "{e:?}");
    assert_eq!(child.len(), clen);

    // 同一テープを子に渡す
    let e = jvp(&tape, &y, &x, &t(seq(3, 5.0), &[3]), &tape).unwrap_err();
    assert!(matches!(e, AutodiffError::Backward(_)), "{e:?}");
}

// ------------------------------------------------------------ J6: 機構と決定性

#[test]
fn j6_deterministic_and_does_not_mutate_existing_nodes() {
    let run = || {
        let tape = new_tape();
        let x = tape.var(&t(seq(6, 1.0), &[2, 3]));
        let y = x.tanh().mul(&x).unwrap();
        let yv = y.value().host_slice().into_owned();
        let xv = x.value().host_slice().into_owned();
        let child = new_tape();
        let a = jvp(&tape, &y, &x, &t(seq(6, 5.0), &[2, 3]), &child)
            .unwrap()
            .host_slice()
            .into_owned();
        // 既存ノードの値は不変。
        assert_eq!(y.value().host_slice().into_owned(), yv);
        assert_eq!(x.value().host_slice().into_owned(), xv);
        let child = new_tape();
        let b = jacfwd(&tape, &y, &x, &child)
            .unwrap()
            .host_slice()
            .into_owned();
        (a, b)
    };
    let (a1, b1) = run();
    let (a2, b2) = run();
    assert_eq!(a1, a2, "jvp が bit 一致しない");
    assert_eq!(b1, b2, "jacfwd が bit 一致しない");
}

#[test]
fn j6_jacfwd_columns_match_one_hot_tangent_products() {
    let shape = [2, 3];
    let n = 6;
    let xv = seq(n, 1.0);
    fn build<'t>(x: &Var<'t>) -> Var<'t> {
        x.tanh().mul(x).unwrap()
    }
    let tc = new_tape();
    let xc = tc.var(&t(xv.clone(), &shape));
    let yc = build(&xc);
    let child = new_tape();
    let full = jacfwd(&tc, &yc, &xc, &child)
        .unwrap()
        .host_slice()
        .into_owned();
    let m = yc.shape().iter().product::<usize>();
    for kk in 0..n {
        let mut e = vec![0.0f32; n];
        e[kk] = 1.0;
        let ta = new_tape();
        let xa = ta.var(&t(xv.clone(), &shape));
        let ya = build(&xa);
        let child = new_tape();
        let col = jvp(&ta, &ya, &xa, &t(e, &shape), &child)
            .unwrap()
            .host_slice()
            .into_owned();
        let expected: Vec<f64> = (0..m).map(|i| f64::from(full[i * n + kk])).collect();
        common::assert_all_req2_close(&col, &expected, &format!("column {kk}"));
    }
}
