//! `functional_ops::vjp`（イシュー #2874・親 #2841）の統合テスト。
//! 契約の正は `docs/autodiff-functional-transforms-design.md` §3〜§7・§10。
//!
//! - V1: 閉形式の小関数との全要素突合（`common::req2_close`。tolerance 定数は新設しない）。
//! - V2: `jacobian_ops::jacobian` の転置積 `Jᵀu` との突合。
//! - V3: fail-closed（拒否時にテープ長が不変）。
//! - V4: ゼロ・空の分岐。
//! - V5: 副作用（補助ノード 2・既存値の bit 不変・決定性）。
//!
//! 実 CPU `BackendOps` との一致・CUDA／Metal は本 issue の対象外（設計 §10 の 4・11）。

mod common;

use fandhe_ai_autodiff::functional_ops::vjp;
use fandhe_ai_autodiff::jacobian_ops::jacobian;
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

fn new_tape() -> Tape {
    Tape::new_with_ops(common::naive_ops())
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test: shape とデータ長は一致させている")
}

fn host(x: &Tensor<f32>) -> Vec<f32> {
    x.host_slice().into_owned()
}

fn assert_close(actual: &Tensor<f32>, shape: &[usize], expected: &[f64], ctx: &str) {
    assert_eq!(actual.shape(), shape, "{ctx}: shape");
    let a = host(actual);
    assert_eq!(a.len(), expected.len(), "{ctx}: 要素数");
    for (i, (&x, &e)) in a.iter().zip(expected).enumerate() {
        assert!(
            common::req2_close(f64::from(x), e),
            "{ctx}[{i}]: {x} vs {e}"
        );
    }
}

/// 符号混在の決定的な値列。
fn seq(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.37).sin() * 1.5)
        .collect()
}

// ---------------------------------------------------------------- V1

#[test]
fn v1_elementwise_square() {
    let tape = new_tape();
    let xv = seq(5, 1.0);
    let uv = seq(5, 7.0);
    let x = tape.var(&t(xv.clone(), &[5]));
    let y = x.mul(&x).unwrap();
    let g = vjp(&tape, &y, &x, &t(uv.clone(), &[5])).unwrap();
    let e: Vec<f64> = xv
        .iter()
        .zip(&uv)
        .map(|(&a, &u)| 2.0 * f64::from(a) * f64::from(u))
        .collect();
    assert_close(&g, &[5], &e, "x*x");
}

#[test]
fn v1_tanh() {
    let tape = new_tape();
    let xv = seq(6, 2.0);
    let uv = seq(6, 3.0);
    let x = tape.var(&t(xv.clone(), &[2, 3]));
    let y = x.tanh();
    let g = vjp(&tape, &y, &x, &t(uv.clone(), &[2, 3])).unwrap();
    let e: Vec<f64> = xv
        .iter()
        .zip(&uv)
        .map(|(&a, &u)| {
            let th = f64::from(a).tanh();
            (1.0 - th * th) * f64::from(u)
        })
        .collect();
    assert_close(&g, &[2, 3], &e, "tanh");
}

#[test]
fn v1_matmul_gives_u_times_w_transpose() {
    let tape = new_tape();
    let wv = seq(12, 4.0);
    let uv = seq(8, 5.0);
    let x = tape.var(&t(seq(6, 6.0), &[2, 3]));
    let w = tape.var_no_grad(&t(wv.clone(), &[3, 4]));
    let y = x.matmul(&w).unwrap();
    let g = vjp(&tape, &y, &x, &t(uv.clone(), &[2, 4])).unwrap();
    let mut e = vec![0.0f64; 6];
    for i in 0..2 {
        for k in 0..3 {
            for j in 0..4 {
                e[i * 3 + k] += f64::from(uv[i * 4 + j]) * f64::from(wv[k * 4 + j]);
            }
        }
    }
    assert_close(&g, &[2, 3], &e, "matmul");
}

#[test]
fn v1_rank0_output() {
    let tape = new_tape();
    let xv = seq(4, 8.0);
    let x = tape.var(&t(xv.clone(), &[4]));
    let y = x.exp().sum(None).unwrap();
    let c = 1.75f32;
    let g = vjp(&tape, &y, &x, &t(vec![c], &[])).unwrap();
    let e: Vec<f64> = xv
        .iter()
        .map(|&a| f64::from(c) * f64::from(a).exp())
        .collect();
    assert_close(&g, &[4], &e, "sum(exp)");
}

#[test]
fn v1_non_contiguous_output() {
    let tape = new_tape();
    let x = tape.var(&t(seq(6, 9.0), &[2, 3]));
    let y = x.transpose(0, 1).unwrap(); // [3, 2]
    let uv = seq(6, 10.0);
    let g = vjp(&tape, &y, &x, &t(uv.clone(), &[3, 2])).unwrap();
    // y[j, i] = x[i, j] なので g[i, j] = u[j, i]。
    let mut e = vec![0.0f64; 6];
    for i in 0..2 {
        for j in 0..3 {
            e[i * 3 + j] = f64::from(uv[j * 2 + i]);
        }
    }
    assert_close(&g, &[2, 3], &e, "transpose");
}

// ---------------------------------------------------------------- V2

/// `vjp` と `jacobian` の転置積 `Jᵀu` を全要素突合する。
fn check_against_jacobian<'t>(tape: &'t Tape, y: &Var<'t>, x: &Var<'t>, seed: f32, ctx: &str) {
    let m: usize = y.to_tensor().shape().to_vec().iter().product();
    let n: usize = x.to_tensor().shape().to_vec().iter().product();
    let uv = seq(m, seed);
    let u = t(uv.clone(), y.to_tensor().shape());
    let j = host(&jacobian(tape, y, x).unwrap());
    assert_eq!(j.len(), m * n, "{ctx}: jacobian 要素数");
    let mut e = vec![0.0f64; n];
    for i in 0..m {
        for k in 0..n {
            e[k] += f64::from(uv[i]) * f64::from(j[i * n + k]);
        }
    }
    let g = vjp(tape, y, x, &u).unwrap();
    assert_close(&g, x.to_tensor().shape(), &e, ctx);
}

#[test]
fn v2_matches_jacobian_transpose_product() {
    let tape = new_tape();
    let x = tape.var(&t(seq(6, 1.0), &[2, 3]));
    let y = x.tanh().mul(&x).unwrap().add(&x.exp()).unwrap();
    check_against_jacobian(&tape, &y, &x, 2.0, "elementwise");

    let tape = new_tape();
    let x = tape.var(&t(seq(6, 3.0), &[2, 3]));
    let w = tape.var(&t(seq(12, 4.0), &[3, 4]));
    let y = x.matmul(&w).unwrap().tanh();
    check_against_jacobian(&tape, &y, &x, 5.0, "matmul_tanh");

    let tape = new_tape();
    let x = tape.var(&t(seq(3, 6.0), &[3]));
    let y = x.broadcast_to(&[2, 3]).unwrap();
    check_against_jacobian(&tape, &y, &x, 7.0, "broadcast_to");

    let tape = new_tape();
    let x = tape.var(&t(seq(6, 8.0), &[2, 3]));
    let y = x.mean(Some(1)).unwrap();
    check_against_jacobian(&tape, &y, &x, 9.0, "mean(dim)");

    let tape = new_tape();
    let x = tape.var(&t(seq(6, 10.0), &[2, 3]));
    let w = tape.var(&t(seq(12, 11.0), &[3, 4]));
    let h = x.matmul(&w).unwrap().relu();
    let y = h.matmul(&tape.var(&t(seq(8, 12.0), &[4, 2]))).unwrap();
    check_against_jacobian(&tape, &y, &x, 13.0, "mlp_relu");

    let tape = new_tape();
    let x = tape.var(&t(seq(4, 14.0), &[2, 2]));
    let c = tape.var_no_grad(&t(seq(2, 15.0), &[1, 2]));
    let y = Var::cat(&[x, c], 0).unwrap().tanh();
    check_against_jacobian(&tape, &y, &x, 16.0, "cat_const_row");
}

// ---------------------------------------------------------------- V3

#[test]
fn v3_rejections_leave_tape_unchanged() {
    let tape = new_tape();
    let other = new_tape();
    let x = tape.var(&t(seq(3, 1.0), &[3]));
    let y = x.tanh();
    let ox = other.var(&t(seq(3, 1.0), &[3]));
    let oy = ox.tanh();
    let nx = tape.var_no_grad(&t(seq(3, 1.0), &[3]));
    let s = y.sum(None).unwrap();
    let u = t(seq(3, 2.0), &[3]);
    let before = tape.len();

    assert!(matches!(
        vjp(&tape, &oy, &x, &u),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        vjp(&tape, &y, &ox, &u),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        vjp(&tape, &y, &nx, &u),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    for bad in [
        t(vec![0.0; 2], &[2]),
        t(vec![0.0], &[1]),
        t(vec![0.0; 3], &[3, 1]),
    ] {
        assert!(
            matches!(
                vjp(&tape, &y, &x, &bad),
                Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
            ),
            "shape {:?}",
            bad.shape()
        );
    }
    // rank 0 出力にブロードキャスト可能な [1] を渡しても拒否される。
    assert!(matches!(
        vjp(&tape, &s, &x, &t(vec![1.0], &[1])),
        Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
    ));
    assert_eq!(tape.len(), before);
}

// ---------------------------------------------------------------- V4

#[test]
fn v4_zero_and_empty_branches() {
    let tape = new_tape();
    let x = tape.var(&t(seq(3, 1.0), &[3]));
    // 追跡なし出力
    let c = tape.var_no_grad(&t(seq(3, 2.0), &[3]));
    let y_const = c.tanh();
    // 無関係な葉のみに依存
    let z = tape.var(&t(seq(3, 3.0), &[3]));
    let y_other = z.tanh();
    // 要素数 0 の input
    let e = tape.var(&t(vec![], &[0]));
    let ye = e.tanh();
    let u = t(seq(3, 4.0), &[3]);
    let before = tape.len();
    let g = vjp(&tape, &y_const, &x, &u).unwrap();
    assert_eq!(g.shape(), &[3]);
    assert!(host(&g).iter().all(|&v| v == 0.0));
    let g = vjp(&tape, &ye, &e, &t(vec![], &[0])).unwrap();
    assert_eq!(g.shape(), &[0]);
    assert!(host(&g).is_empty());
    assert_eq!(tape.len(), before);
    // 追跡ありだが input に届かない出力は backward 経由で全ゼロ（補助ノード 2 は足される）。
    let g = vjp(&tape, &y_other, &x, &u).unwrap();
    assert_eq!(g.shape(), &[3]);
    assert!(host(&g).iter().all(|&v| v == 0.0));
    assert_eq!(tape.len(), before + 2);
}

// ---------------------------------------------------------------- V5

#[test]
fn v5_adds_exactly_two_nodes_and_is_bit_invariant() {
    let tape = new_tape();
    let x = tape.var(&t(vec![0.5, -1.0, 2.0, 0.25], &[2, 2]));
    let y = x.tanh().mul(&x).unwrap();
    let loss = y.sum(None).unwrap();
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    let y_before = bits(&host(&y.to_tensor()));
    let g_before = bits(&host(
        tape.backward(&loss).unwrap().get(&x).unwrap().unwrap(),
    ));

    let u = t(seq(4, 1.0), &[2, 2]);
    let len0 = tape.len();
    let a = vjp(&tape, &y, &x, &u).unwrap();
    assert_eq!(tape.len(), len0 + 2);
    let b = vjp(&tape, &y, &x, &u).unwrap();
    assert_eq!(bits(&host(&a)), bits(&host(&b)));

    assert_eq!(y_before, bits(&host(&y.to_tensor())));
    let g_after = bits(&host(
        tape.backward(&loss).unwrap().get(&x).unwrap().unwrap(),
    ));
    assert_eq!(g_before, g_after);
}
