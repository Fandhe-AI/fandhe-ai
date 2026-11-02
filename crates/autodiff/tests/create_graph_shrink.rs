//! `hardshrink`・`softshrink`（イシュー #3817・親 #3813）の `create_graph`（二階微分）受け入れテスト。
//!
//! 2 演算は区分線形で、素の `sum(op(x))` の Hessian は恒等的に 0 になる。このため下流に非線形
//! （`sum(y ⊙ y ⊙ c)`）を合成した損失で、解析 Hessian を独立な有限差分（`Tape::backward` の中央差分）と
//! REQ-2 統一複合判定で突合する。入力は `|x| == lambd` の境界から離す。
//! 合成元の `MaskedFill`・`ScalarUnary(Clamp)`・`Sub` が既に `create_graph` 対象のため、追加実装は無い。

#![allow(clippy::needless_range_loop)]

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

const LAMBD: f32 = 0.5;
const X0: [f32; 8] = [-1.3, -0.8, -0.3, -0.1, 0.2, 0.7, 1.1, 1.9];

type Built<'t> = Result<Var<'t>, AutodiffError>;

/// `sum(y ⊙ y ⊙ c)`。
fn quad<'t>(tape: &'t Tape, y: &Var<'t>) -> Built<'t> {
    let c = tape.var_no_grad(&t(common::det_seq(8, 0.7), &[8]));
    y.mul(y)?.mul(&c)?.sum(None)
}

fn hard<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    quad(tape, &x.hardshrink(LAMBD)?)
}

fn soft<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    quad(tape, &x.softshrink(LAMBD)?)
}

fn finite_diff_hessian<F>(build: F, h: f64) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let n = X0.len();
    let grad_at = |x: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), &[n]));
        let loss = build(&tape, &xv).expect("forward");
        let grads = tape.backward(&loss).expect("backward");
        let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");
        (0..n).map(|i| dx.get(&[i]).unwrap() as f64).collect()
    };
    let mut hessian = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let (mut xp, mut xm) = (X0.to_vec(), X0.to_vec());
        xp[i] += h as f32;
        xm[i] -= h as f32;
        let (gp, gm) = (grad_at(&xp), grad_at(&xm));
        for j in 0..n {
            hessian[j][i] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    hessian
}

fn analytic_hessian<F>(build: F) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let n = X0.len();
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(X0.to_vec(), &[n]));
    let loss = build(&tape, &x).expect("forward");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph に失敗");
    let gx = cg.grad(&x).unwrap().expect("x は loss に到達する");
    let cx = cg.child_var(&x).unwrap().expect("子テープ上の写し");
    let mut hessian = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = gx.narrow(0, j, 1).unwrap().sum(None).unwrap();
        let row_grads = child.backward(&gj).expect("child.backward（2 階）に失敗");
        let row = row_grads
            .get(&cx)
            .unwrap()
            .expect("2 階勾配は x へ到達する");
        for i in 0..n {
            hessian[j][i] = row.get(&[i]).unwrap() as f64;
        }
    }
    hessian
}

fn assert_hessian_close(analytic: &[Vec<f64>], numeric: &[Vec<f64>]) {
    for (ra, rn) in analytic.iter().zip(numeric) {
        for (&a, &n) in ra.iter().zip(rn) {
            assert!(
                common::req2_close(a, n),
                "hessian mismatch: analytic={a} numeric={n}"
            );
        }
    }
}

#[test]
fn hardshrink_hessian_matches_finite_difference() {
    assert_hessian_close(&analytic_hessian(hard), &finite_diff_hessian(hard, 1e-2));
}

#[test]
fn softshrink_hessian_matches_finite_difference() {
    assert_hessian_close(&analytic_hessian(soft), &finite_diff_hessian(soft, 1e-2));
}

#[test]
fn first_order_is_bit_identical_to_plain_backward() {
    for build in [hard, soft] {
        let plain = {
            let tape = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(X0.to_vec(), &[8]));
            let loss = build(&tape, &x).unwrap();
            let g = tape.backward(&loss).unwrap();
            g.get(&x).unwrap().unwrap().host_slice().to_vec()
        };
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(X0.to_vec(), &[8]));
        let loss = build(&tape, &x).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let first = cg.first_order();
        let fo = first.get(&x).unwrap().unwrap().host_slice().to_vec();
        let a: Vec<u32> = plain.iter().map(|v| v.to_bits()).collect();
        let b: Vec<u32> = fo.iter().map(|v| v.to_bits()).collect();
        assert_eq!(a, b);
    }
}
