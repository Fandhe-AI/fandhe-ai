//! `bilinear`（イシュー #3818・親 #3813）の `create_graph`（二階微分）受け入れテスト。
//!
//! x1・x2・weight・bias を 1 本のベクトルにまとめ、下流に非線形（`sum(y ⊙ y ⊙ c)`）を合成した損失の
//! 解析 Hessian を、`Tape::backward` の中央差分（NaiveOps）と REQ-2 統一複合判定で突合する。
//! 双線形なので x1・x2 の交差項が非 0 になることも確かめる。合成元の `MatMul`（rank 2）・`Permute`・
//! `Contiguous`・`Reshape`・`Mul`・`Sum`・`Add` は既に `create_graph` 対象のため追加実装は無い。
//! 設計 §2.4 は `create_graph.rs` への節追加だが、並列 PR との競合を避けるため別ファイルにした。

#![allow(clippy::needless_range_loop)]

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

// N=2, in1=3, in2=2, out=2。ベクトル配置: x1(6) | x2(4) | weight(12) | bias(2)。
const N: usize = 2;
const IN1: usize = 3;
const IN2: usize = 2;
const OUT: usize = 2;
const LEN: usize = N * IN1 + N * IN2 + OUT * IN1 * IN2 + OUT;

type Built<'t> = Result<Var<'t>, AutodiffError>;

fn x0() -> Vec<f32> {
    (0..LEN)
        .map(|i| ((i as f32) * 0.77 + 0.3).sin() * 0.8)
        .collect()
}

fn build<'t>(tape: &'t Tape, v: &Var<'t>) -> Built<'t> {
    let mut at = 0usize;
    let mut take = |len: usize, shape: &[usize]| -> Built<'t> {
        let s = v.narrow(0, at, len)?.reshape(shape)?;
        at += len;
        Ok(s)
    };
    let x1 = take(N * IN1, &[N, IN1])?;
    let x2 = take(N * IN2, &[N, IN2])?;
    let w = take(OUT * IN1 * IN2, &[OUT, IN1, IN2])?;
    let b = take(OUT, &[OUT])?;
    let y = x1.bilinear(&x2, &w, Some(&b))?;
    let c = tape.var_no_grad(&t(common::det_seq(N * OUT, 0.7), &[N, OUT]));
    y.mul(&y)?.mul(&c)?.sum(None)
}

fn finite_diff_hessian(h: f64) -> Vec<Vec<f64>> {
    let base = x0();
    let grad_at = |x: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), &[LEN]));
        let loss = build(&tape, &xv).expect("forward");
        let grads = tape.backward(&loss).expect("backward");
        let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");
        (0..LEN).map(|i| dx.get(&[i]).unwrap() as f64).collect()
    };
    let mut hessian = vec![vec![0.0f64; LEN]; LEN];
    for i in 0..LEN {
        let (mut xp, mut xm) = (base.clone(), base.clone());
        xp[i] += h as f32;
        xm[i] -= h as f32;
        let (gp, gm) = (grad_at(&xp), grad_at(&xm));
        for j in 0..LEN {
            hessian[j][i] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    hessian
}

fn analytic_hessian() -> Vec<Vec<f64>> {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0(), &[LEN]));
    let loss = build(&tape, &x).expect("forward");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph に失敗");
    let gx = cg.grad(&x).unwrap().expect("x は loss に到達する");
    let cx = cg.child_var(&x).unwrap().expect("子テープ上の写し");
    let mut hessian = vec![vec![0.0f64; LEN]; LEN];
    for j in 0..LEN {
        let gj = gx.narrow(0, j, 1).unwrap().sum(None).unwrap();
        let row_grads = child.backward(&gj).expect("child.backward（2 階）に失敗");
        let row = row_grads
            .get(&cx)
            .unwrap()
            .expect("2 階勾配は x へ到達する");
        for i in 0..LEN {
            hessian[j][i] = row.get(&[i]).unwrap() as f64;
        }
    }
    hessian
}

#[test]
fn bilinear_hessian_matches_finite_difference() {
    let analytic = analytic_hessian();
    let numeric = finite_diff_hessian(1e-2);
    for (ra, rn) in analytic.iter().zip(&numeric) {
        for (&a, &n) in ra.iter().zip(rn) {
            assert!(
                common::req2_close(a, n),
                "hessian mismatch: analytic={a} numeric={n}"
            );
        }
    }
    // 交差項（x1 と x2 の混合 2 階微分）が非 0 になる。x1[0,0]=index 0、x2[0,0]=index 6。
    assert!(analytic[0][N * IN1].abs() > 1e-4, "x1-x2 交差項が 0");
}

#[test]
fn first_order_is_bit_identical_to_plain_backward() {
    let plain = {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0(), &[LEN]));
        let loss = build(&tape, &x).unwrap();
        let g = tape.backward(&loss).unwrap();
        g.get(&x).unwrap().unwrap().host_slice().to_vec()
    };
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0(), &[LEN]));
    let loss = build(&tape, &x).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let first = cg.first_order();
    let fo = first.get(&x).unwrap().unwrap().host_slice().to_vec();
    let a: Vec<u32> = plain.iter().map(|v| v.to_bits()).collect();
    let b: Vec<u32> = fo.iter().map(|v| v.to_bits()).collect();
    assert_eq!(a, b);
}
