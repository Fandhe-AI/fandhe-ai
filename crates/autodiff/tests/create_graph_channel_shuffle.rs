//! `Var::channel_shuffle`（イシュー #3822）の `create_graph`（二階微分）受け入れテスト。
//!
//! 合成元の `Contiguous`／reshape／permute が既に `create_graph` 対象のため追加実装は無い。
//! 線形写像なので素の損失では Hessian が 0 になる。下流に非線形（`sum(y ⊙ y ⊙ c)`）を置き、
//! 解析 Hessian を独立な中央差分と REQ-2 統一複合判定で突合する。
//! `create_graph.rs` に足さないのは並列 PR との衝突回避（設計記録 §3.13）。

#![allow(clippy::needless_range_loop)]

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

const N: usize = 8;
const SHAPE: [usize; 3] = [1, 4, 2];

fn x0() -> Vec<f32> {
    common::det_seq(N, 1.0)
}

type Built<'t> = Result<Var<'t>, AutodiffError>;

fn build<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    let c = tape.var_no_grad(&t(common::det_seq(N, 0.7), &SHAPE));
    let y = x.channel_shuffle(2)?;
    y.mul(&y)?.mul(&c)?.sum(None)
}

fn flat(tensor: &Tensor<f32>) -> Vec<f32> {
    tensor.contiguous().as_slice().unwrap().to_vec()
}

fn finite_diff_hessian(h: f64) -> Vec<Vec<f64>> {
    let base = x0();
    let grad_at = |x: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), &SHAPE));
        let loss = build(&tape, &xv).expect("forward");
        let grads = tape.backward(&loss).expect("backward");
        let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");
        flat(dx).iter().map(|&v| v as f64).collect()
    };
    let mut hessian = vec![vec![0.0f64; N]; N];
    for i in 0..N {
        let (mut xp, mut xm) = (base.clone(), base.clone());
        xp[i] += h as f32;
        xm[i] -= h as f32;
        let (gp, gm) = (grad_at(&xp), grad_at(&xm));
        for j in 0..N {
            hessian[j][i] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    hessian
}

fn analytic_hessian() -> Vec<Vec<f64>> {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0(), &SHAPE));
    let loss = build(&tape, &x).expect("forward");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph に失敗");
    let gx = cg.grad(&x).unwrap().expect("x は loss に到達する");
    let cx = cg.child_var(&x).unwrap().expect("子テープ上の写し");
    let gx = gx.reshape(&[N]).unwrap();
    let mut hessian = vec![vec![0.0f64; N]; N];
    for j in 0..N {
        let gj = gx.narrow(0, j, 1).unwrap().sum(None).unwrap();
        let row_grads = child.backward(&gj).expect("child.backward（2 階）に失敗");
        let row = row_grads
            .get(&cx)
            .unwrap()
            .expect("2 階勾配は x へ到達する");
        let row = flat(row);
        for i in 0..N {
            hessian[j][i] = row[i] as f64;
        }
    }
    hessian
}

#[test]
fn channel_shuffle_hessian_matches_finite_difference() {
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
}
