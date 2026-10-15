//! `functional_ops` の double-VJP 版 `jvp`／`jacfwd` の受入テスト（イシュー #2940・親 #2939）。
//!
//! 契約の正は `docs/autodiff-functional-transforms-design.md` §8・§19・§24。比較先は
//! `jacobian_ops::jacobian`（facade の `Tape::jacobian` はこれへ 1 行委譲するだけで、autodiff から
//! facade へは依存できない）。判定は REQ-2 統一複合判定を `tests/common/mod.rs` の共有定数・
//! `req2_close` 経由でのみ行い、閾値は直書きしない。
//!
//! `jvp`／`jacfwd` は #2956 で facade `Tape::jvp`／`Tape::jacfwd` として公開済みだが、本ファイルは
//! 末尾の `c1`〜`c9` は微分可能な `jvp_create_graph`（イシュー #3067・設計記録 §31）の受入テスト。
//! 内部実装の `#[cfg(test)]` 単体テストとして維持する（#2880 の `double_vjp_feasibility_tests` と同じ形）。テスト関数・ヘルパーの名前は
//! `jvp`／`jacfwd` と完全一致させない（facade `api_surface` の宣言インベントリが
//! `crates/*/src` 全体を走査するため）。入力はキンク（0・clamp 境界・タイ）から離す。

use std::sync::Arc;

use fandhe_ai_tensor_core::{ScalarBinaryOp, ShapeError, Tensor};

// REQ-2 判定・共通ヘルパーは #2880 の単体テストが `#[path]` で取り込んだ単一の定義元を共用する
// （同一ファイルの二重取り込みは clippy::duplicate_mod になるため）。
use crate::double_vjp_feasibility_tests::common;
use crate::functional_ops::{JvpCreateGraphResult, hvp, jacfwd, jvp, jvp_create_graph};
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
    // #3081: 孫テープでの Erf 再生（閉包性）を通す。
    check(&[6], xv.clone(), "gelu", |_, x| x.gelu());
    check(&[6], xv.clone(), "gelu_tanh", |_, x| x.gelu_tanh());
    check(&[6], xv.clone(), "erf", |_, x| x.erf());
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

// ------------------------------------------------------------ M: 複数入力版（#3063）

/// 複数入力版の呼び出し（テスト名・ヘルパー名は宣言インベントリ対策で関数名と完全一致させない）。
fn multi_jv<'t>(
    tape: &'t Tape,
    y: &Var<'t>,
    xs: &[&Var<'t>],
    vs: &[&Tensor<f32>],
    child: &Tape,
) -> Result<Tensor<f32>, AutodiffError> {
    crate::functional_ops::jvp_multi(tape, y, xs, vs, child)
}

/// `build`（入力 k 個）で作った出力に対し、入力ごとの `jacobian` 由来の参照 `Σ J_i·v_i`（f64 合計）と
/// 統一複合判定で突合する。
fn check_multi<F>(shapes: &[Vec<usize>], xvs: &[Vec<f32>], ctx: &str, build: F)
where
    F: for<'t> Fn(&'t Tape, &[Var<'t>]) -> Built<'t>,
{
    let vvs: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| seq(s.iter().product(), 5.0 + i as f32))
        .collect();

    // 参照
    let tb = new_tape();
    let xb: Vec<Var<'_>> = shapes
        .iter()
        .zip(xvs)
        .map(|(s, d)| tb.var(&t(d.clone(), s)))
        .collect();
    let yb = build(&tb, &xb).unwrap();
    let m: usize = yb.shape().iter().product();
    let mut expected = vec![0.0f64; m];
    for (i, x) in xb.iter().enumerate() {
        let jh = jacobian(&tb, &yb, x).unwrap().host_slice().into_owned();
        let n: usize = shapes[i].iter().product();
        let part = common::jacobian_times_vector_f64(&jh, m, n, &vvs[i]);
        for (e, p) in expected.iter_mut().zip(part) {
            *e += p;
        }
    }

    let ta = new_tape();
    let xa: Vec<Var<'_>> = shapes
        .iter()
        .zip(xvs)
        .map(|(s, d)| ta.var(&t(d.clone(), s)))
        .collect();
    let ya = build(&ta, &xa).unwrap();
    let tangents: Vec<Tensor<f32>> = shapes
        .iter()
        .zip(&vvs)
        .map(|(s, d)| t(d.clone(), s))
        .collect();
    let xr: Vec<&Var<'_>> = xa.iter().collect();
    let tr: Vec<&Tensor<f32>> = tangents.iter().collect();
    let child = new_tape();
    let got = multi_jv(&ta, &ya, &xr, &tr, &child)
        .unwrap_or_else(|e| panic!("{ctx}: 複数入力 jvp 失敗: {e:?}"));
    assert_eq!(got.shape(), yb.shape().as_slice(), "{ctx}: shape");
    common::assert_all_req2_close(&got.host_slice(), &expected, ctx);
}

#[test]
fn m1_multi_input_matches_jacobian_sum() {
    // mul: x2⊙v1 + x1⊙v2
    check_multi(
        &[vec![5], vec![5]],
        &[seq(5, 1.0), seq(5, 2.0)],
        "mul",
        |_, x| x[0].mul(&x[1]),
    );
    // matmul→tanh（W・x とも追跡あり・非正方形）
    check_multi(
        &[vec![3, 4], vec![4, 2]],
        &[seq(12, 1.0), seq(8, 2.0)],
        "tanh(Wx)",
        |_, x| Ok(x[0].matmul(&x[1])?.tanh()),
    );
    // ブロードキャスト add
    check_multi(
        &[vec![2, 3], vec![3]],
        &[seq(6, 1.0), seq(3, 2.0)],
        "bcast add",
        |_, x| x[0].add(&x[1]),
    );
    // 3 入力の合成
    check_multi(
        &[vec![4], vec![4], vec![4]],
        &[seq(4, 1.0), seq(4, 2.0), seq(4, 3.0)],
        "tanh(a*b)+c",
        |_, x| x[0].mul(&x[1])?.tanh().add(&x[2]),
    );
    // rank 0 の入力を含む
    check_multi(
        &[vec![], vec![3]],
        &[vec![0.7], seq(3, 2.0)],
        "rank0",
        |_, x| x[1].mul(&x[0]),
    );
}

#[test]
fn m1_closed_form_mul() {
    let (a, b) = (seq(4, 1.0), seq(4, 2.0));
    let (va, vb) = (t(seq(4, 5.0), &[4]), t(seq(4, 6.0), &[4]));
    let tape = new_tape();
    let x1 = tape.var(&t(a.clone(), &[4]));
    let x2 = tape.var(&t(b.clone(), &[4]));
    let y = x1.mul(&x2).unwrap();
    let child = new_tape();
    let got = multi_jv(&tape, &y, &[&x1, &x2], &[&va, &vb], &child).unwrap();
    let (va, vb) = (va.host_slice(), vb.host_slice());
    let e: Vec<f64> = (0..4)
        .map(|i| f64::from(b[i]) * f64::from(va[i]) + f64::from(a[i]) * f64::from(vb[i]))
        .collect();
    common::assert_all_req2_close(&got.host_slice(), &e, "x2*v1 + x1*v2");
}

#[test]
fn m2_single_input_matches_single_and_zero_contribution_inputs_drop_out() {
    let tape = new_tape();
    let x1 = tape.var(&t(seq(4, 1.0), &[4]));
    let x2 = tape.var(&t(seq(4, 2.0), &[4]));
    let x3 = tape.var(&t(seq(0, 1.0), &[0]));
    let y = x1.tanh().mul(&x1).unwrap();
    let (v1, v2, v3) = (t(seq(4, 5.0), &[4]), t(seq(4, 6.0), &[4]), t(vec![], &[0]));
    let child = new_tape();
    let single = jvp(&tape, &y, &x1, &v1, &child).unwrap();
    // k=1
    let child = new_tape();
    let one = multi_jv(&tape, &y, &[&x1], &[&v1], &child).unwrap();
    common::assert_all_req2_close(
        &one.host_slice(),
        &single
            .host_slice()
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
        "k=1",
    );
    // y が x2 に依存しない（J_2 ≡ 0）・要素数 0 の入力混在
    let child = new_tape();
    let multi = multi_jv(&tape, &y, &[&x1, &x2, &x3], &[&v1, &v2, &v3], &child).unwrap();
    common::assert_all_req2_close(
        &multi.host_slice(),
        &single
            .host_slice()
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
        "J2=0 と空入力",
    );
}

#[test]
fn m3_entry_checks_and_ordering() {
    let tape = new_tape();
    let x1 = tape.var(&t(seq(3, 1.0), &[3]));
    let x2 = tape.var(&t(seq(3, 2.0), &[3]));
    let nx = tape.var_no_grad(&t(seq(3, 3.0), &[3]));
    let y = x1.mul(&x2).unwrap();
    let v = t(seq(3, 5.0), &[3]);
    let other = new_tape();
    let ox = other.var(&t(seq(3, 1.0), &[3]));
    let len = tape.len();
    let child = new_tape();
    let check = |e: AutodiffError, expect: &str, ctx: &str| {
        let ok = match expect {
            "invalid" => matches!(e, AutodiffError::InvalidArgument(_)),
            "mismatch" => matches!(e, AutodiffError::TapeMismatch),
            "notrack" => matches!(e, AutodiffError::GradientTrackingDisabled),
            "shape" => matches!(e, AutodiffError::Shape(ShapeError::ShapeMismatch { .. })),
            _ => false,
        };
        assert!(ok, "{ctx}: {e:?}");
        assert_eq!(tape.len(), len, "{ctx}: 親テープが変化した");
        assert!(child.is_empty(), "{ctx}: 子テープが変化した");
    };
    let bad = [t(vec![1.0], &[1]), t(seq(3, 1.0), &[3, 1])];

    check(
        multi_jv(&tape, &y, &[&x1, &x2], &[&v], &child).unwrap_err(),
        "invalid",
        "長さ不一致",
    );
    check(
        multi_jv(&tape, &y, &[], &[], &child).unwrap_err(),
        "invalid",
        "空",
    );
    check(
        multi_jv(&tape, &y, &[&x1, &ox], &[&v, &v], &child).unwrap_err(),
        "mismatch",
        "別テープ入力",
    );
    check(
        multi_jv(&tape, &ox.tanh(), &[&x1, &x2], &[&v, &v], &child).unwrap_err(),
        "mismatch",
        "別テープ output",
    );
    check(
        multi_jv(&tape, &y, &[&x1, &nx], &[&v, &v], &child).unwrap_err(),
        "notrack",
        "追跡なし",
    );
    check(
        multi_jv(&tape, &y, &[&x1, &x1], &[&v, &v], &child).unwrap_err(),
        "invalid",
        "重複",
    );
    for b in &bad {
        check(
            multi_jv(&tape, &y, &[&x1, &x2], &[&v, b], &child).unwrap_err(),
            "shape",
            "tangent shape",
        );
    }
    // 順序: 長さ不一致 > TapeMismatch > 追跡なし > 重複 > shape
    check(
        multi_jv(&tape, &y, &[&x1, &ox], &[&v], &child).unwrap_err(),
        "invalid",
        "長さ>mismatch",
    );
    check(
        multi_jv(&tape, &y, &[&nx, &ox], &[&v, &v], &child).unwrap_err(),
        "mismatch",
        "mismatch>追跡なし",
    );
    check(
        multi_jv(&tape, &y, &[&nx, &x2], &[&bad[0], &v], &child).unwrap_err(),
        "notrack",
        "追跡なし>shape",
    );
    check(
        multi_jv(&tape, &y, &[&x1, &x1], &[&bad[0], &v], &child).unwrap_err(),
        "invalid",
        "重複>shape",
    );
    // k=1 は単一版と同じエラー
    let e1 = jvp(&tape, &y, &nx, &v, &child).unwrap_err();
    let e2 = multi_jv(&tape, &y, &[&nx], &[&v], &child).unwrap_err();
    assert_eq!(format!("{e1:?}"), format!("{e2:?}"));
    let e1 = jvp(&tape, &y, &x1, &bad[0], &child).unwrap_err();
    let e2 = multi_jv(&tape, &y, &[&x1], &[&bad[0]], &child).unwrap_err();
    assert_eq!(format!("{e1:?}"), format!("{e2:?}"));
}

#[test]
fn m4_zero_return_paths_do_not_touch_tapes() {
    let tape = new_tape();
    let x1 = tape.var(&t(seq(3, 1.0), &[3]));
    let x2 = tape.var(&t(seq(3, 2.0), &[3]));
    let e1 = tape.var(&t(vec![], &[0]));
    let ny = tape.var_no_grad(&t(seq(4, 2.0), &[4]));
    let empty_y = tape.var(&t(vec![], &[0]));
    let y = x1.mul(&x2).unwrap();
    let v = t(seq(3, 5.0), &[3]);
    let ev = t(vec![], &[0]);
    let len = tape.len();
    let child = new_tape();
    let r = multi_jv(&tape, &ny, &[&x1, &x2], &[&v, &v], &child).unwrap();
    assert_eq!(r.shape(), &[4]);
    assert!(r.host_slice().iter().all(|&z| z == 0.0));
    let r = multi_jv(&tape, &empty_y, &[&x1, &x2], &[&v, &v], &child).unwrap();
    assert_eq!(r.shape(), &[0]);
    let r = multi_jv(&tape, &y, &[&e1], &[&ev], &child).unwrap();
    assert_eq!(r.shape(), &[3]);
    assert!(r.host_slice().iter().all(|&z| z == 0.0));
    assert_eq!(tape.len(), len);
    assert!(child.is_empty());
}

#[test]
fn m5_unsupported_ops_and_bad_child_are_rejected_with_two_parent_nodes() {
    let tape = new_tape();
    let x1 = tape.var(&t(seq(6, 1.0), &[2, 3]));
    let x2 = tape.var(&t(seq(6, 2.0), &[2, 3]));
    let y = x1.add(&x2).unwrap().max(None).unwrap();
    let v = t(seq(6, 5.0), &[2, 3]);
    let child = new_tape();
    let len = tape.len();
    let e = multi_jv(&tape, &y, &[&x1, &x2], &[&v, &v], &child).unwrap_err();
    assert!(matches!(e, AutodiffError::Backward(_)), "{e:?}");
    assert!(child.is_empty());
    assert_eq!(tape.len(), len + 2);

    // 非空の子・同一テープ
    let tape = new_tape();
    let x1 = tape.var(&t(seq(3, 1.0), &[3]));
    let x2 = tape.var(&t(seq(3, 2.0), &[3]));
    let y = x1.mul(&x2).unwrap();
    let v = t(seq(3, 5.0), &[3]);
    let child = new_tape();
    let _ = child.var(&t(vec![1.0], &[1]));
    let clen = child.len();
    let e = multi_jv(&tape, &y, &[&x1, &x2], &[&v, &v], &child).unwrap_err();
    assert!(matches!(e, AutodiffError::Backward(_)), "{e:?}");
    assert_eq!(child.len(), clen);
    let e = multi_jv(&tape, &y, &[&x1, &x2], &[&v, &v], &tape).unwrap_err();
    assert!(matches!(e, AutodiffError::Backward(_)), "{e:?}");
}

#[test]
fn m6_multi_is_deterministic_and_does_not_mutate_existing_nodes() {
    let run = || {
        let tape = new_tape();
        let x1 = tape.var(&t(seq(6, 1.0), &[2, 3]));
        let x2 = tape.var(&t(seq(3, 2.0), &[3]));
        let y = x1.tanh().mul(&x2).unwrap();
        let yv = y.value().host_slice().into_owned();
        let (v1, v2) = (t(seq(6, 5.0), &[2, 3]), t(seq(3, 6.0), &[3]));
        let child = new_tape();
        let a = multi_jv(&tape, &y, &[&x1, &x2], &[&v1, &v2], &child)
            .unwrap()
            .host_slice()
            .into_owned();
        assert_eq!(y.value().host_slice().into_owned(), yv);
        a
    };
    assert_eq!(run(), run(), "複数入力 jvp が bit 一致しない");
}
// ------------------------------------------------------------ C1〜C9: 微分可能な jvp（#3067・§31）

/// 別名ヘルパー（名前を `jvp_create_graph` と完全一致させない。api_surface のインベントリ対策）。
/// 親・child・grandchild を呼び出し側が持ち、結果を返す。
fn cg_jv<'g>(
    tape: &Tape,
    y: &Var<'_>,
    x: &Var<'_>,
    v: &Tensor<f32>,
    child: &Tape,
    grand: &'g Tape,
) -> Result<JvpCreateGraphResult<'g>, AutodiffError> {
    jvp_create_graph(tape, y, x, v, child, grand)
}

/// `build` の出力に対し、`jvp` と `jvp_create_graph` の値を REQ-2 判定で、2 段目の余接勾配を bit 同一で突合する。
fn check_cg<F>(shape: &[usize], xv: Vec<f32>, ctx: &str, build: F)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let n: usize = shape.iter().product();
    let vv = seq(n, 5.0);

    let ta = new_tape();
    let xa = ta.var(&t(xv.clone(), shape));
    let ya = build(&ta, &xa).unwrap();
    let child = new_tape();
    let want = jvp(&ta, &ya, &xa, &t(vv.clone(), shape), &child)
        .unwrap_or_else(|e| panic!("{ctx}: jvp 失敗: {e:?}"));

    let tb = new_tape();
    let xb = tb.var(&t(xv, shape));
    let yb = build(&tb, &xb).unwrap();
    let child = new_tape();
    let grand = new_tape();
    let res = cg_jv(&tb, &yb, &xb, &t(vv, shape), &child, &grand)
        .unwrap_or_else(|e| panic!("{ctx}: jvp_create_graph 失敗: {e:?}"));
    let out = res.output();
    assert_eq!(out.shape(), want.shape(), "{ctx}: shape");
    let want_h = want.host_slice().into_owned();
    let expected: Vec<f64> = want_h.iter().map(|&w| f64::from(w)).collect();
    common::assert_all_req2_close(&out.value().host_slice(), &expected, ctx);
    match res.stage2_cotangent_grad_host() {
        Some(h) => {
            let a: Vec<u32> = h.host_slice().iter().map(|f| f.to_bits()).collect();
            let b: Vec<u32> = want_h.iter().map(|f| f.to_bits()).collect();
            assert_eq!(a, b, "{ctx}: 2 段目の余接勾配が jvp と bit 同一でない");
        }
        None => assert!(want_h.iter().all(|&w| w == 0.0), "{ctx}: ゼロ経路"),
    }
}

#[test]
fn c1_closed_form_value_and_gradient() {
    let xv = seq(6, 1.0);
    let vv = seq(6, 5.0);
    let tape = new_tape();
    let x = tape.var(&t(xv.clone(), &[2, 3]));
    let y = x.mul(&x).unwrap();
    let (child, grand) = (new_tape(), new_tape());
    let res = cg_jv(&tape, &y, &x, &t(vv.clone(), &[2, 3]), &child, &grand).unwrap();
    let e: Vec<f64> = (0..6)
        .map(|i| 2.0 * f64::from(xv[i]) * f64::from(vv[i]))
        .collect();
    common::assert_all_req2_close(&res.output().value().host_slice(), &e, "value");
    let loss = res.output().sum(None).unwrap();
    let grads = grand.backward(&loss).unwrap();
    let gx = res.child_var(&x).unwrap().expect("x の写しが存在する");
    let g = grads.get(&gx).unwrap().expect("勾配が存在する");
    let e: Vec<f64> = (0..6).map(|i| 2.0 * f64::from(vv[i])).collect();
    common::assert_all_req2_close(&g.host_slice(), &e, "d/dx sum(J v) = 2v");
}

/// `F(W, x) = Σ_i (1 - tanh²(W x)_i) (W v)_i` の f64 値（`y = tanh(W x)` の `Σ J v`）。
fn f_tanh_wx(w: &[f64], x: &[f64], v: &[f64]) -> f64 {
    (0..2)
        .map(|i| {
            let z: f64 = (0..3).map(|j| w[i * 3 + j] * x[j]).sum();
            let wv: f64 = (0..3).map(|j| w[i * 3 + j] * v[j]).sum();
            (1.0 - z.tanh().powi(2)) * wv
        })
        .sum()
}

#[test]
fn c2_gradient_matches_f64_central_difference() {
    let wv = seq(6, 3.0);
    let xv = seq(3, 1.0);
    let vv = seq(3, 5.0);
    let tape = new_tape();
    let x = tape.var(&t(xv.clone(), &[3, 1]));
    let w = tape.var(&t(wv.clone(), &[2, 3]));
    let y = w.matmul(&x).unwrap().tanh();
    let (child, grand) = (new_tape(), new_tape());
    let res = cg_jv(&tape, &y, &x, &t(vv.clone(), &[3, 1]), &child, &grand).unwrap();
    let loss = res.output().sum(None).unwrap();
    let grads = grand.backward(&loss).unwrap();

    let to64 = |s: &[f32]| s.iter().map(|&a| f64::from(a)).collect::<Vec<f64>>();
    let (w64, x64, v64) = (to64(&wv), to64(&xv), to64(&vv));
    let h = 1e-4;
    let fd = |base: &[f64], which_w: bool| -> Vec<f64> {
        (0..base.len())
            .map(|k| {
                let eval = |d: f64| {
                    let mut b = base.to_vec();
                    b[k] += d;
                    if which_w {
                        f_tanh_wx(&b, &x64, &v64)
                    } else {
                        f_tanh_wx(&w64, &b, &v64)
                    }
                };
                (eval(h) - eval(-h)) / (2.0 * h)
            })
            .collect()
    };
    let gx = res.child_var(&x).unwrap().unwrap();
    let gw = res.child_var(&w).unwrap().unwrap();
    common::assert_all_req2_close(
        &grads.get(&gx).unwrap().unwrap().host_slice(),
        &fd(&x64, false),
        "tanh(Wx) d/dx",
    );
    common::assert_all_req2_close(
        &grads.get(&gw).unwrap().unwrap().host_slice(),
        &fd(&w64, true),
        "tanh(Wx) d/dW",
    );

    // y = tanh(x) ⊙ x: F(x) = Σ (tanh x + x sech² x) v_i
    let tape = new_tape();
    let x = tape.var(&t(xv.clone(), &[3]));
    let y = x.tanh().mul(&x).unwrap();
    let (child, grand) = (new_tape(), new_tape());
    let res = cg_jv(&tape, &y, &x, &t(vv.clone(), &[3]), &child, &grand).unwrap();
    let loss = res.output().sum(None).unwrap();
    let grads = grand.backward(&loss).unwrap();
    let f = |xs: &[f64]| -> f64 {
        xs.iter()
            .zip(&v64)
            .map(|(&a, &b)| (a.tanh() + a * (1.0 - a.tanh().powi(2))) * b)
            .sum()
    };
    let e: Vec<f64> = (0..3)
        .map(|k| {
            let eval = |d: f64| {
                let mut b = x64.clone();
                b[k] += d;
                f(&b)
            };
            (eval(h) - eval(-h)) / (2.0 * h)
        })
        .collect();
    let gx = res.child_var(&x).unwrap().unwrap();
    common::assert_all_req2_close(
        &grads.get(&gx).unwrap().unwrap().host_slice(),
        &e,
        "tanh(x)*x d/dx",
    );
}

#[test]
fn c3_scalar_gradient_matches_hvp() {
    let xv = seq(4, 1.0);
    let vv = seq(4, 5.0);
    let tape = new_tape();
    let x = tape.var(&t(xv.clone(), &[4]));
    let l = x.tanh().mul(&x).unwrap().sum(None).unwrap();
    let (child, grand) = (new_tape(), new_tape());
    let res = cg_jv(&tape, &l, &x, &t(vv.clone(), &[4]), &child, &grand).unwrap();
    let grads = grand.backward(&res.output()).unwrap();
    let gx = res.child_var(&x).unwrap().unwrap();
    let got = grads.get(&gx).unwrap().unwrap().host_slice().into_owned();

    let tb = new_tape();
    let xb = tb.var(&t(xv, &[4]));
    let lb = xb.tanh().mul(&xb).unwrap().sum(None).unwrap();
    let child = new_tape();
    let hv = hvp(&tb, &lb, &xb, &t(vv, &[4]), &child).unwrap();
    let e: Vec<f64> = hv.host_slice().iter().map(|&a| f64::from(a)).collect();
    common::assert_all_req2_close(&got, &e, "H v");
}

#[test]
fn c4_value_matches_jvp_and_stage2_is_bit_identical() {
    let xv = seq(6, 1.0);
    check_cg(&[6], xv.clone(), "x*x", |_, x| x.mul(x));
    check_cg(&[2, 3], xv.clone(), "mlp", |tp, x| {
        let w1 = c(tp, seq(12, 4.0), &[3, 4]);
        x.matmul(&w1)?.tanh().matmul(&c(tp, seq(8, 6.0), &[4, 2]))
    });
    check_cg(&[2, 3], xv.clone(), "broadcast add", |tp, x| {
        x.add(&c(tp, seq(3, 4.0), &[3]))
    });
    check_cg(&[], vec![0.7], "rank0 tanh", |_, x| Ok(x.tanh()));
    check_cg(&[2, 3], xv, "sum all", |_, x| x.sum(None));
}

#[test]
fn c5_mirror_none_some_and_child_var_misuse() {
    let tape = new_tape();
    let x = tape.var(&t(seq(4, 1.0), &[4, 1]));
    let w = tape.var(&t(seq(12, 3.0), &[3, 4]));
    let y = w.matmul(&x).unwrap();
    let (child, grand) = (new_tape(), new_tape());
    let res = cg_jv(&tape, &y, &x, &t(seq(4, 5.0), &[4, 1]), &child, &grand).unwrap();
    // 線形 y = W x: J·v = W v は x に依存しない（写し x は Ok(None)）、W には依存する。
    assert!(res.child_var(&x).unwrap().is_none());
    assert!(res.child_var(&w).unwrap().is_some());

    // 別テープ・追跡なしの Var の誤用。
    let other = new_tape();
    let ox = other.var(&t(seq(4, 1.0), &[4, 1]));
    assert!(matches!(
        res.child_var(&ox),
        Err(AutodiffError::TapeMismatch)
    ));
    let ng = tape.var_no_grad(&t(seq(4, 1.0), &[4, 1]));
    assert!(matches!(
        res.child_var(&ng),
        Err(AutodiffError::GradientTrackingDisabled)
    ));

    // 係数が追跡なしの y = A x では出力も追跡なしになる。
    let tape = new_tape();
    let x = tape.var(&t(seq(4, 1.0), &[4, 1]));
    let a = tape.var_no_grad(&t(seq(12, 3.0), &[3, 4]));
    let y = a.matmul(&x).unwrap();
    let (child, grand) = (new_tape(), new_tape());
    let res = cg_jv(&tape, &y, &x, &t(seq(4, 5.0), &[4, 1]), &child, &grand).unwrap();
    assert!(!res.output().requires_grad());
    assert!(grand.backward(&res.output()).is_err());
}

#[test]
fn c6_entry_checks_order_and_tapes_unchanged() {
    let tape = new_tape();
    let x = tape.var(&t(seq(3, 1.0), &[3]));
    let y = x.tanh();
    let tg = t(seq(3, 5.0), &[3]);
    let ng = tape.var_no_grad(&t(seq(3, 1.0), &[3]));
    let (child, grand) = (new_tape(), new_tape());
    let lens = |tape: &Tape, child: &Tape, grand: &Tape| (tape.len(), child.len(), grand.len());
    let before = lens(&tape, &child, &grand);
    let same = |ctx: &str, tape: &Tape, child: &Tape, grand: &Tape| {
        assert_eq!(lens(tape, child, grand), before, "{ctx}: テープが変化した");
    };

    let other = new_tape();
    let ox = other.var(&t(seq(3, 1.0), &[3]));
    let e = cg_jv(&tape, &y, &ox, &tg, &child, &grand).unwrap_err();
    assert!(matches!(e, AutodiffError::TapeMismatch), "{e:?}");
    same("tape mismatch", &tape, &child, &grand);

    let (child2, grand2) = (new_tape(), new_tape());
    let len2 = tape.len();
    // 追跡なし入力と shape 不一致が重なる場合は GradientTrackingDisabled が先。
    let e = cg_jv(&tape, &y, &ng, &t(seq(2, 5.0), &[2]), &child2, &grand2).unwrap_err();
    assert!(
        matches!(e, AutodiffError::GradientTrackingDisabled),
        "{e:?}"
    );
    assert_eq!(tape.len(), len2);

    // shape 不一致と非空 child が重なる場合は Shape が先。
    let busy = new_tape();
    let _ = busy.var(&t(vec![1.0], &[1]));
    let e = cg_jv(&tape, &y, &x, &t(seq(2, 5.0), &[2]), &busy, &grand).unwrap_err();
    assert!(matches!(e, AutodiffError::Shape(_)), "{e:?}");

    // テープ関係の違反は Backward。
    let e = cg_jv(&tape, &y, &x, &tg, &tape, &grand).unwrap_err();
    assert!(
        matches!(e, AutodiffError::Backward(_)),
        "child==tape: {e:?}"
    );
    let e = cg_jv(&tape, &y, &x, &tg, &child, &tape).unwrap_err();
    assert!(
        matches!(e, AutodiffError::Backward(_)),
        "grand==tape: {e:?}"
    );
    let e = cg_jv(&tape, &y, &x, &tg, &child, &child).unwrap_err();
    assert!(
        matches!(e, AutodiffError::Backward(_)),
        "grand==child: {e:?}"
    );
    let e = cg_jv(&tape, &y, &x, &tg, &busy, &grand).unwrap_err();
    assert!(matches!(e, AutodiffError::Backward(_)), "child 非空: {e:?}");
    let e = cg_jv(&tape, &y, &x, &tg, &child, &busy).unwrap_err();
    assert!(matches!(e, AutodiffError::Backward(_)), "grand 非空: {e:?}");
    same("tape relation", &tape, &child, &grand);
    assert_eq!(busy.len(), 1);
    // `DeviceMismatch` は別デバイスを返す BackendOps が test_support に無いため未検証
    // （backward_create_graph 側の同検査は create_graph のテストが担う。§31.9 に記録）。
}

type RejectedCase = (
    &'static str,
    Vec<usize>,
    Box<dyn for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>>,
);

#[test]
fn c7_zero_paths_and_node_counts() {
    // (a) 追跡なし output、(b) 出力要素数 0、(c) 入力要素数 0: 親・child 無変更、grand +1、写しなし。
    let tape = new_tape();
    let x = tape.var(&t(seq(3, 1.0), &[3]));
    let z = tape.var(&t(vec![], &[0]));
    let ng = tape.var_no_grad(&t(seq(3, 1.0), &[3]));
    let ez = Tensor::zeros(&[0]).unwrap();
    let cases: Vec<(&str, Var<'_>, Var<'_>, Tensor<f32>)> = vec![
        ("untracked out", ng.tanh(), x, t(seq(3, 5.0), &[3])),
        ("empty out", z.tanh(), x, t(seq(3, 5.0), &[3])),
        ("empty in", x.tanh(), z, ez),
    ];
    for (ctx, y, inp, tg) in &cases {
        let (child, grand) = (new_tape(), new_tape());
        let len = tape.len();
        let res = cg_jv(&tape, y, inp, tg, &child, &grand).unwrap();
        assert_eq!(tape.len(), len, "{ctx}");
        assert!(child.is_empty(), "{ctx}");
        assert_eq!(grand.len(), 1, "{ctx}");
        assert!(res.child_var(&x).unwrap().is_none(), "{ctx}");
        assert!(res.output().value().host_slice().iter().all(|&a| a == 0.0));
    }

    // (d) J ≡ 0: output が別の追跡葉だけに依存する。親 +2、child 非空、grand +1、全ゼロ。
    let zleaf = tape.var(&t(seq(3, 2.0), &[3]));
    let y = zleaf.tanh();
    let (child, grand) = (new_tape(), new_tape());
    let len = tape.len();
    let res = cg_jv(&tape, &y, &x, &t(seq(3, 5.0), &[3]), &child, &grand).unwrap();
    assert_eq!(tape.len(), len + 2);
    assert!(!child.is_empty());
    assert_eq!(grand.len(), 1);
    assert!(res.output().value().host_slice().iter().all(|&a| a == 0.0));

    // (e) 1 段目の拒否: Backward・親 +2・child と grand は空のまま。
    let rejected: Vec<RejectedCase> = vec![
        ("max", vec![2, 3], Box::new(|_, x| x.max(None))),
        (
            "matmul rank3",
            vec![2, 2, 2],
            Box::new(|tp, x| x.matmul(&c(tp, seq(8, 2.0), &[2, 2, 2]))),
        ),
        (
            "custom",
            vec![3],
            Box::new(|tp, x| tp.custom(Arc::new(IdentityCustomFn), &[*x])),
        ),
        ("selu", vec![4], Box::new(|_, x| x.selu())),
    ];
    for (ctx, shape, build) in &rejected {
        let n: usize = shape.iter().product();
        let tape = new_tape();
        let x = tape.var(&t(seq(n, 1.0), shape));
        let y = build(&tape, &x).unwrap();
        let (child, grand) = (new_tape(), new_tape());
        let len = tape.len();
        let e = cg_jv(&tape, &y, &x, &t(seq(n, 5.0), shape), &child, &grand).unwrap_err();
        assert!(matches!(e, AutodiffError::Backward(_)), "{ctx}: {e:?}");
        assert_eq!(tape.len(), len + 2, "{ctx}");
        assert!(child.is_empty() && grand.is_empty(), "{ctx}");
    }

    // (f) 成功時は child・grand ともに非空。
    let tape = new_tape();
    let x = tape.var(&t(seq(3, 1.0), &[3]));
    let y = x.tanh();
    let (child, grand) = (new_tape(), new_tape());
    cg_jv(&tape, &y, &x, &t(seq(3, 5.0), &[3]), &child, &grand).unwrap();
    assert!(!child.is_empty() && !grand.is_empty());
}

#[test]
fn c8_second_stage_mirrors_op_families() {
    let xv = seq(6, 1.0);
    let pv = pos(6, 1.0);
    let mask = Tensor::new(vec![true, false, true, true, false, false], &[2, 3]).unwrap();
    check_cg(&[2, 3], xv.clone(), "bias add", |tp, x| {
        x.add(&c(tp, seq(3, 4.0), &[3]))
    });
    check_cg(&[3], seq(3, 1.0), "bias add (x が [n])", |tp, x| {
        c(tp, seq(6, 2.0), &[2, 3]).add(x)
    });
    check_cg(&[2, 3], xv.clone(), "matmul", |tp, x| {
        x.matmul(&c(tp, seq(12, 2.0), &[3, 4]))
    });
    check_cg(&[3, 3], seq(9, 1.0), "matmul x·x", |_, x| x.matmul(x));
    check_cg(&[2, 3], xv.clone(), "where", |tp, x| {
        Var::where_cond(&mask, x, &c(tp, seq(6, 2.0), &[2, 3]))
    });
    check_cg(&[6], xv.clone(), "relu", |_, x| Ok(x.relu()));
    check_cg(&[6], xv.clone(), "clamp", |_, x| x.clamp(-0.8, 0.8));
    check_cg(&[6], xv.clone(), "abs", |_, x| x.abs());
    check_cg(&[6], pv.clone(), "log", |_, x| x.log());
    check_cg(&[6], xv.clone(), "sin", |_, x| x.sin());
    check_cg(&[6], xv.clone(), "exp", |_, x| Ok(x.exp()));
    check_cg(&[6], xv.clone(), "sigmoid", |_, x| Ok(x.sigmoid()));
    check_cg(&[6], pv.clone(), "pow_scalar", |_, x| x.pow_scalar(3.0));
    check_cg(&[6], xv.clone(), "sub", |tp, x| k(tp).sub(x));
    check_cg(&[6], xv.clone(), "div", |tp, x| {
        x.div(&c(tp, pos(6, 2.0), &[6]))
    });
    check_cg(&[6], pv, "pow", |tp, x| {
        x.pow(&c(tp, vec![2.0, 1.5, 3.0, 0.5, 2.5, 1.0], &[6]))
    });
    check_cg(&[2, 3], xv.clone(), "reshape", |_, x| x.reshape(&[3, 2]));
    check_cg(&[2, 3], xv.clone(), "transpose", |_, x| x.transpose(0, 1));
    check_cg(&[2, 3], xv.clone(), "narrow", |_, x| x.narrow(1, 1, 2));
    check_cg(&[2, 3], xv, "cat", |tp, x| {
        Var::cat(&[*x, c(tp, seq(6, 2.0), &[2, 3])], 1)
    });
}

#[test]
fn c9_deterministic_and_does_not_mutate_existing_nodes() {
    let run = || {
        let tape = new_tape();
        let x = tape.var(&t(seq(6, 1.0), &[2, 3]));
        let y = x.tanh().mul(&x).unwrap();
        let yv = y.value().host_slice().into_owned();
        let (child, grand) = (new_tape(), new_tape());
        let r = cg_jv(&tape, &y, &x, &t(seq(6, 5.0), &[2, 3]), &child, &grand).unwrap();
        assert_eq!(y.value().host_slice().into_owned(), yv);
        r.output()
            .value()
            .host_slice()
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<u32>>()
    };
    assert_eq!(run(), run());
}
