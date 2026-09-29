//! `fandhe_ai::TapeRef`（#2394）の統合テスト。facade だけに依存して使えること、
//! `TapeRef` 経由の葉と `Tape` 直の葉で backward が bit 一致することを固定する。

use fandhe_ai::{Tape, TapeRef, Tensor, Var, tape};

fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_slice(data, shape).expect("テスト入力の構築")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.as_slice()
        .expect("host tensor")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// 値渡しした `TapeRef` から作った `Var<'t>` を返せること（寿命が `'t` に結び付く）のコンパイル時保証。
fn leaf_from_by_value<'t>(r: TapeRef<'t>, x: &Tensor<f32>) -> Var<'t> {
    r.var(x)
}

fn loss_of<'t>(x: &Var<'t>, w: &Var<'t>) -> Var<'t> {
    let y = x.matmul(w).expect("matmul").tanh();
    y.mul(&y).expect("mul").sum(None).expect("sum")
}

const X: [f32; 6] = [0.5, -1.0, 2.0, 0.25, 1.5, -0.75];
const W: [f32; 6] = [0.1, 0.2, -0.3, 0.4, 0.5, -0.6];

#[test]
fn var_backward_is_bit_identical() {
    let ta = tape();
    let (xa, wa) = (ta.var(&t32(&X, &[2, 3])), ta.var(&t32(&W, &[3, 2])));
    let la = loss_of(&xa, &wa);
    let ga = ta.backward(&la).expect("backward A");

    let tb = tape();
    let r = TapeRef::from(&tb);
    let (xb, wb) = (r.var(&t32(&X, &[2, 3])), r.var(&t32(&W, &[3, 2])));
    let lb = loss_of(&xb, &wb);
    let gb = tb.backward(&lb).expect("backward B");

    assert_eq!(bits(&la.value()), bits(&lb.value()));
    for (a, b) in [(&xa, &xb), (&wa, &wb)] {
        let ga = ga.get(a).expect("get A").expect("勾配 A");
        let gb = gb.get(b).expect("get B").expect("勾配 B");
        assert_eq!(bits(ga), bits(gb));
    }
}

#[test]
fn var_from_backward_is_bit_identical() {
    let x64: Vec<f64> = X.iter().map(|&v| f64::from(v)).collect();
    let w64: Vec<f64> = W.iter().map(|&v| f64::from(v)).collect();
    let xt = Tensor::from_slice(&x64, &[2, 3]).expect("f64 入力");
    let wt = Tensor::from_slice(&w64, &[3, 2]).expect("f64 入力");

    let ta = tape();
    let (xa, wa) = (
        ta.var_from(&xt).expect("var_from A"),
        ta.var_from(&wt).expect("var_from A"),
    );
    let ga = ta.backward(&loss_of(&xa, &wa)).expect("backward A");

    let tb = tape();
    let r = TapeRef::from(&tb);
    let (xb, wb) = (
        r.var_from(&xt).expect("var_from B"),
        r.var_from(&wt).expect("var_from B"),
    );
    let gb = tb.backward(&loss_of(&xb, &wb)).expect("backward B");

    for (a, b) in [(&xa, &xb), (&wa, &wb)] {
        let ga = ga.get(a).expect("get A").expect("勾配 A");
        let gb = gb.get(b).expect("get B").expect("勾配 B");
        assert_eq!(bits(ga), bits(gb));
    }
}

#[test]
fn var_no_grad_has_no_gradient_and_grad_leaf_matches() {
    let ta = tape();
    let (xa, wa) = (ta.var_no_grad(&t32(&X, &[2, 3])), ta.var(&t32(&W, &[3, 2])));
    let ga = ta.backward(&loss_of(&xa, &wa)).expect("backward A");

    let tb = tape();
    let r = TapeRef::from(&tb);
    let (xb, wb) = (r.var_no_grad(&t32(&X, &[2, 3])), r.var(&t32(&W, &[3, 2])));
    let gb = tb.backward(&loss_of(&xb, &wb)).expect("backward B");

    // no_grad の葉は両経路とも同じ理由（勾配追跡なし）で get が Err になる。
    assert!(matches!(
        ga.get(&xa),
        Err(fandhe_ai::AutodiffError::GradientTrackingDisabled)
    ));
    assert!(matches!(
        gb.get(&xb),
        Err(fandhe_ai::AutodiffError::GradientTrackingDisabled)
    ));
    assert_eq!(
        bits(ga.get(&wa).expect("get").expect("勾配")),
        bits(gb.get(&wb).expect("get").expect("勾配"))
    );
}

#[test]
fn mixed_leaves_on_one_tape_share_backward() {
    let t = tape();
    let x = t.var(&t32(&X, &[2, 3]));
    let w = leaf_from_by_value(TapeRef::from(&t), &t32(&W, &[3, 2]));
    let g = t.backward(&loss_of(&x, &w)).expect("backward");
    assert!(g.get(&x).expect("get").is_some());
    assert!(g.get(&w).expect("get").is_some());
}

#[test]
fn traits_and_into() {
    fn assert_traits<T: Copy + Clone + std::fmt::Debug>() {}
    assert_traits::<TapeRef<'static>>();
    let t: Tape = tape();
    let r: TapeRef<'_> = (&t).into();
    let copy = r;
    assert!(format!("{r:?}").starts_with("TapeRef"));
    assert!(format!("{copy:?}").starts_with("TapeRef"));
}
