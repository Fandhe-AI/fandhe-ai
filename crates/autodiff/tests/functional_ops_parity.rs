//! `functional_ops::vjp`（イシュー #2874・親 #2841）の統合テスト。
//! 契約の正は `docs/autodiff-functional-transforms-design.md` §3〜§7・§10。
//!
//! - V1: 閉形式の小関数との全要素突合（`common::req2_close`。tolerance 定数は新設しない）。
//! - V2: `jacobian_ops::jacobian` の転置積 `Jᵀu` との突合。
//! - V3: fail-closed（拒否時にテープ長が不変）。
//! - V4: ゼロ・空の分岐。
//! - V5: 副作用（補助ノード 2・既存値の bit 不変・決定性）。
//! - H1〜H5: `hvp`（#2875）。閉形式・`hessian` との突合・fail-closed・ゼロ/空・副作用。
//!
//! - M1〜M6: `vmap`（#2876）。バッチなし一致（統一複合判定）・非 contiguous 出力・
//!   Phase A／B の fail-closed・クロージャ `Err` 時のノード残存・決定性と微分可能性。
//!
//! 実 CPU `BackendOps` との一致・CUDA／Metal は本 issue の対象外（設計 §10 の 4・11）。

mod common;

use fandhe_ai_autodiff::functional_ops::{hvp, vjp, vmap};
use fandhe_ai_autodiff::jacobian_ops::{hessian, jacobian};
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

// ---------------------------------------------------------------- H（hvp。#2875）

/// `hvp` と `jacobian_ops::hessian`（facade `Tape::hessian` の実体）の `H·v` を全要素
/// 突合する。facade 側は `jacobian_ops::hessian` への薄い委譲のため autodiff 層で受け入れ
/// 条件を満たす。
fn check_hvp_against_hessian(
    shape: &[usize],
    seed: f32,
    ctx: &str,
    build: impl for<'t> Fn(&'t Tape, &Var<'t>) -> Var<'t>,
) {
    let n: usize = shape.iter().product();
    let xv = seq(n, seed);
    let vv = seq(n, seed + 5.0);

    let tape = new_tape();
    let x = tape.var(&t(xv, shape));
    let loss = build(&tape, &x);
    let child_h = new_tape();
    let h = hessian(&tape, &loss, &x, &child_h).unwrap();
    let hd = host(&h);
    let expected: Vec<f64> = (0..n)
        .map(|j| {
            (0..n)
                .map(|k| f64::from(hd[j * n + k]) * f64::from(vv[k]))
                .sum()
        })
        .collect();

    let child = new_tape();
    let got = hvp(&tape, &loss, &x, &t(vv, shape), &child).unwrap();
    assert_close(&got, shape, &expected, ctx);
}

#[test]
fn h1_closed_forms() {
    // exp: H v = exp(x) ⊙ v
    let tape = new_tape();
    let xv = seq(4, 1.0);
    let vv = seq(4, 9.0);
    let x = tape.var(&t(xv.clone(), &[4]));
    let loss = x.exp().sum(None).unwrap();
    let child = new_tape();
    let g = hvp(&tape, &loss, &x, &t(vv.clone(), &[4]), &child).unwrap();
    let e: Vec<f64> = xv
        .iter()
        .zip(&vv)
        .map(|(&a, &v)| f64::from(a).exp() * f64::from(v))
        .collect();
    assert_close(&g, &[4], &e, "sum exp");

    // mean 版: / n
    let tape = new_tape();
    let x = tape.var(&t(xv.clone(), &[4]));
    let loss = x.exp().mean(None).unwrap();
    let child = new_tape();
    let g = hvp(&tape, &loss, &x, &t(vv.clone(), &[4]), &child).unwrap();
    let e: Vec<f64> = e.iter().map(|v| v / 4.0).collect();
    assert_close(&g, &[4], &e, "mean exp");

    // 二次形式 Σ x ⊙ (A x): (A + Aᵀ) v
    let a = [0.5f32, -1.0, 2.0, 0.25, 1.5, -0.75, 0.3, 0.9, -0.2];
    let tape = new_tape();
    let x = tape.var(&t(xv[..3].to_vec(), &[3, 1]));
    let am = tape.var_no_grad(&t(a.to_vec(), &[3, 3]));
    let ax = am.matmul(&x).unwrap();
    let loss = x.mul(&ax).unwrap().sum(None).unwrap();
    let child = new_tape();
    let v3 = t(vv[..3].to_vec(), &[3, 1]);
    let g = hvp(&tape, &loss, &x, &v3, &child).unwrap();
    let e: Vec<f64> = (0..3)
        .map(|j| {
            (0..3)
                .map(|k| f64::from(a[j * 3 + k] + a[k * 3 + j]) * f64::from(vv[k]))
                .sum()
        })
        .collect();
    assert_close(&g, &[3, 1], &e, "quadratic form");

    // tanh(x) ⊙ x: d²/dx² = 2 sech² - 2 x tanh sech²
    let tape = new_tape();
    let x = tape.var(&t(xv.clone(), &[4]));
    let loss = x.tanh().mul(&x).unwrap().sum(None).unwrap();
    let child = new_tape();
    let g = hvp(&tape, &loss, &x, &t(vv.clone(), &[4]), &child).unwrap();
    let e: Vec<f64> = xv
        .iter()
        .zip(&vv)
        .map(|(&a, &v)| {
            let a = f64::from(a);
            let th = a.tanh();
            let s2 = 1.0 - th * th;
            (2.0 * s2 - 2.0 * a * th * s2) * f64::from(v)
        })
        .collect();
    assert_close(&g, &[4], &e, "tanh*x");

    // relu(x) ⊙ x: 二階導関数 2·[x>0]
    let tape = new_tape();
    let x = tape.var(&t(xv.clone(), &[4]));
    let loss = x.relu().mul(&x).unwrap().sum(None).unwrap();
    let child = new_tape();
    let g = hvp(&tape, &loss, &x, &t(vv.clone(), &[4]), &child).unwrap();
    let e: Vec<f64> = xv
        .iter()
        .zip(&vv)
        .map(|(&a, &v)| if a > 0.0 { 2.0 * f64::from(v) } else { 0.0 })
        .collect();
    assert_close(&g, &[4], &e, "relu*x");
}

#[test]
fn h1_scalar_loss_shapes_accepted() {
    for shape in [vec![], vec![1usize], vec![1, 1]] {
        let tape = new_tape();
        let x = tape.var(&t(vec![0.5, -0.25], &[2]));
        let s = x.exp().sum(None).unwrap();
        let loss = if shape.is_empty() {
            s
        } else {
            s.reshape(&shape).unwrap()
        };
        let child = new_tape();
        let g = hvp(&tape, &loss, &x, &t(vec![1.0, 2.0], &[2]), &child).unwrap();
        assert_close(
            &g,
            &[2],
            &[0.5f64.exp(), 2.0 * (-0.25f64).exp()],
            &format!("loss shape {shape:?}"),
        );
    }
}

#[test]
fn h2_matches_hessian_times_vector() {
    check_hvp_against_hessian(&[5], 1.0, "tanh*x + exp", |_, x| {
        x.tanh()
            .mul(x)
            .unwrap()
            .add(&x.exp())
            .unwrap()
            .sum(None)
            .unwrap()
    });
    check_hvp_against_hessian(&[2, 3], 2.0, "matmul+tanh", |tape, x| {
        let w = tape.var_no_grad(&t(seq(6, 4.0), &[3, 2]));
        x.matmul(&w).unwrap().tanh().sum(None).unwrap()
    });
    check_hvp_against_hessian(&[2, 3], 3.0, "mlp", |tape, x| {
        let w1 = tape.var_no_grad(&t(seq(6, 5.0), &[3, 2]));
        let w2 = tape.var_no_grad(&t(seq(2, 6.0), &[2, 1]));
        x.matmul(&w1)
            .unwrap()
            .relu()
            .matmul(&w2)
            .unwrap()
            .tanh()
            .sum(None)
            .unwrap()
    });
    check_hvp_against_hessian(&[3], 4.0, "broadcast_to", |_, x| {
        x.exp()
            .broadcast_to(&[2, 3])
            .unwrap()
            .tanh()
            .sum(None)
            .unwrap()
    });
    check_hvp_against_hessian(&[2, 3], 5.0, "transpose", |_, x| {
        let xt = x.transpose(0, 1).unwrap();
        xt.tanh().mul(&xt).unwrap().sum(None).unwrap()
    });
    check_hvp_against_hessian(&[2, 3], 6.0, "mean(dim)", |_, x| {
        x.exp().mean(Some(1)).unwrap().tanh().sum(None).unwrap()
    });
    check_hvp_against_hessian(&[2, 2], 7.0, "cat+const", |tape, x| {
        let c = tape.var_no_grad(&t(seq(4, 8.0), &[2, 2]));
        let cat = Var::cat(&[*x, c], 0).unwrap();
        cat.tanh().mul(&cat).unwrap().sum(None).unwrap()
    });
}

#[test]
fn h3_rejections_leave_both_tapes_unchanged() {
    let tape = new_tape();
    let other = new_tape();
    let x = tape.var(&t(seq(3, 1.0), &[3]));
    let loss = x.exp().sum(None).unwrap();
    let ox = other.var(&t(seq(3, 1.0), &[3]));
    let oloss = ox.exp().sum(None).unwrap();
    let nx = tape.var_no_grad(&t(seq(3, 1.0), &[3]));
    let nonscalar = x.exp();
    let untracked_loss = nx.exp().sum(None).unwrap();
    let v = t(seq(3, 2.0), &[3]);
    let before = tape.len();
    let child = new_tape();

    assert!(matches!(
        hvp(&tape, &oloss, &x, &v, &child),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        hvp(&tape, &loss, &ox, &v, &child),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        hvp(&tape, &loss, &nx, &v, &child),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    assert!(matches!(
        hvp(&tape, &nonscalar, &x, &v, &child),
        Err(AutodiffError::InvalidArgument(_))
    ));
    for bad in [
        t(vec![0.0; 2], &[2]),
        t(vec![0.0], &[1]),
        t(vec![0.0; 3], &[3, 1]),
    ] {
        assert!(
            matches!(
                hvp(&tape, &loss, &x, &bad, &child),
                Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
            ),
            "shape {:?}",
            bad.shape()
        );
    }
    // 追跡なし loss は vjp の全ゼロと異なり hessian と同じく backward_create_graph の Err。
    assert!(hvp(&tape, &untracked_loss, &x, &v, &child).is_err());
    assert!(child.is_empty());
    assert_eq!(tape.len(), before);

    // 非対象 Op（max）
    let m = x.max(None).unwrap();
    let before = tape.len();
    assert!(matches!(
        hvp(&tape, &m, &x, &v, &child),
        Err(AutodiffError::Backward(_))
    ));
    assert!(child.is_empty());
    assert_eq!(tape.len(), before);

    // rank 3 matmul
    let a = tape.var(&t(seq(8, 1.0), &[2, 2, 2]));
    let b = tape.var_no_grad(&t(seq(8, 2.0), &[2, 2, 2]));
    let l3 = a.matmul(&b).unwrap().sum(None).unwrap();
    let before = tape.len();
    assert!(matches!(
        hvp(&tape, &l3, &a, &t(seq(8, 3.0), &[2, 2, 2]), &child),
        Err(AutodiffError::Backward(_))
    ));
    assert!(child.is_empty());
    assert_eq!(tape.len(), before);

    // 非空の子テープ
    let busy = new_tape();
    let _leaf = busy.var(&t(vec![1.0], &[1]));
    assert!(matches!(
        hvp(&tape, &loss, &x, &v, &busy),
        Err(AutodiffError::Backward(_))
    ));
    assert_eq!(busy.len(), 1);
}

#[test]
fn h4_zero_and_empty_branches() {
    let tape = new_tape();
    let x = tape.var(&t(seq(3, 1.0), &[3]));
    let z = tape.var(&t(seq(3, 2.0), &[3]));
    let v = t(seq(3, 3.0), &[3]);
    // input に届かない loss
    let l_other = z.exp().sum(None).unwrap();
    let child = new_tape();
    let g = hvp(&tape, &l_other, &x, &v, &child).unwrap();
    assert_eq!(g.shape(), &[3]);
    assert!(host(&g).iter().all(|&e| e == 0.0));
    // input に線形な loss
    let c = tape.var_no_grad(&t(seq(3, 4.0), &[3]));
    let l_lin = c.mul(&x).unwrap().sum(None).unwrap();
    let child = new_tape();
    let g = hvp(&tape, &l_lin, &x, &v, &child).unwrap();
    assert_eq!(g.shape(), &[3]);
    assert!(host(&g).iter().all(|&e| e == 0.0));
    // 要素数 0 の input
    let e = tape.var(&t(vec![], &[0]));
    let l = e.exp().sum(None).unwrap();
    let child = new_tape();
    let g = hvp(&tape, &l, &e, &t(vec![], &[0]), &child).unwrap();
    assert_eq!(g.shape(), &[0]);
    assert!(host(&g).is_empty());
    assert!(child.is_empty());
}

#[test]
fn h5_side_effects_and_determinism() {
    let tape = new_tape();
    let x = tape.var(&t(vec![0.5, -1.0, 2.0, 0.25], &[2, 2]));
    let y = x.tanh().mul(&x).unwrap();
    let loss = y.sum(None).unwrap();
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    let y_before = bits(&host(&y.to_tensor()));
    let g_before = bits(&host(
        tape.backward(&loss).unwrap().get(&x).unwrap().unwrap(),
    ));

    // backward_create_graph のみの子テープ長 L
    let probe = new_tape();
    let _cg = tape.backward_create_graph(&loss, &probe).unwrap();
    let l = probe.len();

    let v = t(seq(4, 1.0), &[2, 2]);
    let len0 = tape.len();
    let c1 = new_tape();
    let a = hvp(&tape, &loss, &x, &v, &c1).unwrap();
    assert_eq!(tape.len(), len0);
    assert_eq!(c1.len(), l + 2);
    let c2 = new_tape();
    let b = hvp(&tape, &loss, &x, &v, &c2).unwrap();
    assert_eq!(bits(&host(&a)), bits(&host(&b)));

    assert_eq!(y_before, bits(&host(&y.to_tensor())));
    let g_after = bits(&host(
        tape.backward(&loss).unwrap().get(&x).unwrap().unwrap(),
    ));
    assert_eq!(g_before, g_after);
}
// ---------------------------------------------------------------- M（vmap。#2876）

fn assert_vars_close(actual: &Var<'_>, expected: &Var<'_>, ctx: &str) {
    let e = expected.to_tensor();
    let a = actual.to_tensor();
    assert_eq!(a.shape(), e.shape(), "{ctx}: shape");
    for (i, (&x, &y)) in host(&a).iter().zip(host(&e).iter()).enumerate() {
        assert!(
            common::req2_close(f64::from(x), f64::from(y)),
            "{ctx}[{i}]: {x} vs {y}"
        );
    }
}

#[test]
fn m1_matches_unbatched() {
    // (a) 要素ごと
    let tape = new_tape();
    let x = tape.var(&t(seq(12, 1.0), &[4, 3]));
    let got = vmap(&tape, &x, 0, |s| s.tanh().mul(s)).unwrap();
    assert_vars_close(&got, &x.tanh().mul(&x).unwrap(), "elementwise");

    // (b) 行列積
    let w = tape.var_no_grad(&t(seq(15, 3.0), &[3, 5]));
    let got = vmap(&tape, &x, 0, |s| {
        s.reshape(&[1, 3])?.matmul(&w)?.tanh().reshape(&[5])
    })
    .unwrap();
    assert_vars_close(&got, &x.matmul(&w).unwrap().tanh(), "matmul");

    // (c) in_dim = 1（結果 [3, 2] は f(x) の転置と一致）
    let x2 = tape.var(&t(seq(6, 2.0), &[2, 3]));
    let got = vmap(&tape, &x2, 1, |s| s.tanh().mul(s)).unwrap();
    let full = x2.tanh().mul(&x2).unwrap().transpose(0, 1).unwrap();
    assert_vars_close(&got, &full, "in_dim=1");

    // (d) スカラー出力
    let got = vmap(&tape, &x, 0, |s| s.sum(None)).unwrap();
    assert_vars_close(&got, &x.sum(Some(1)).unwrap(), "scalar out");

    // (e) rank 1 入力（スライスは rank 0）
    let x1 = tape.var(&t(seq(5, 4.0), &[5]));
    let got = vmap(&tape, &x1, 0, |s| s.mul(s)).unwrap();
    assert_vars_close(&got, &x1.mul(&x1).unwrap(), "rank1");
}

#[test]
fn m2_non_contiguous_output_is_materialized() {
    let tape = new_tape();
    let x = tape.var(&t(seq(24, 1.0), &[2, 3, 4]));
    let got = vmap(&tape, &x, 0, |s| s.transpose(0, 1)).unwrap();
    assert_eq!(got.to_tensor().shape(), &[2, 4, 3]);
    let full = x.transpose(1, 2).unwrap();
    assert_vars_close(&got, &full, "transpose");

    // stack を直接使うと非 contiguous で拒否される（contiguous 化が必要な理由）
    let outs: Vec<Var<'_>> = x
        .unbind(0)
        .unwrap()
        .iter()
        .map(|s| s.transpose(0, 1).unwrap())
        .collect();
    assert!(matches!(
        Var::stack(&outs, 0),
        Err(AutodiffError::Shape(ShapeError::NonContiguousReshape))
    ));
}

/// 呼び出し回数を数える恒等クロージャ（Phase A でクロージャが呼ばれないことの確認用）。
fn counting<'t>(
    c: &std::cell::Cell<usize>,
) -> impl FnMut(&Var<'t>) -> Result<Var<'t>, AutodiffError> + '_ {
    move |s| {
        c.set(c.get() + 1);
        Ok(*s)
    }
}

#[test]
fn m3_phase_a_fail_closed_keeps_tape() {
    let tape = new_tape();
    let other = new_tape();
    let x = tape.var(&t(seq(6, 1.0), &[2, 3]));
    let xo = other.var(&t(seq(6, 1.0), &[2, 3]));
    let empty = tape.var(&t(vec![], &[0, 3]));
    let scalar = tape.var(&t(vec![1.0], &[]));
    let calls = std::cell::Cell::new(0usize);
    let len0 = tape.len();

    assert!(matches!(
        vmap(&tape, &xo, 0, counting(&calls)),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        vmap(&tape, &x, 2, counting(&calls)),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            axis: 2,
            rank: 2
        }))
    ));
    assert!(matches!(
        vmap(&tape, &scalar, 0, counting(&calls)),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
    ));
    assert!(matches!(
        vmap(&tape, &empty, 0, counting(&calls)),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // 表の順序: 別テープかつ in_dim 不正 → TapeMismatch、in_dim 不正かつ空軸 → AxisOutOfRange
    assert!(matches!(
        vmap(&tape, &xo, 9, counting(&calls)),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        vmap(&tape, &empty, 5, counting(&calls)),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
    ));
    assert_eq!(calls.get(), 0, "Phase A 失敗ではクロージャを呼ばない");
    assert_eq!(tape.len(), len0, "Phase A 失敗ではテープ無変更");
}

#[test]
fn m4_phase_b_fail_closed() {
    let tape = new_tape();
    let other = new_tape();
    let x = tape.var(&t(seq(6, 1.0), &[3, 2]));
    let foreign = other.var(&t(seq(2, 1.0), &[2]));
    let before = host(&x.to_tensor());

    // 別テープの出力（2 番目で検出 → 3 番目は呼ばれない）
    let calls = std::cell::Cell::new(0usize);
    let len0 = tape.len();
    let r = vmap(&tape, &x, 0, |s| {
        calls.set(calls.get() + 1);
        if calls.get() == 2 {
            Ok(foreign)
        } else {
            Ok(*s)
        }
    });
    assert!(matches!(r, Err(AutodiffError::TapeMismatch)));
    assert_eq!(calls.get(), 2);
    assert!(tape.len() > len0);

    // 形状不一致
    let calls = std::cell::Cell::new(0usize);
    let r = vmap(&tape, &x, 0, |s| {
        calls.set(calls.get() + 1);
        if calls.get() == 2 {
            s.reshape(&[1, 2])
        } else {
            Ok(*s)
        }
    });
    assert!(matches!(
        r,
        Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
    ));
    assert_eq!(calls.get(), 2);

    // 別テープかつ形状違い → TapeMismatch が先
    let foreign3 = other.var(&t(seq(3, 1.0), &[3]));
    let calls = std::cell::Cell::new(0usize);
    let r = vmap(&tape, &x, 0, |s| {
        calls.set(calls.get() + 1);
        if calls.get() == 2 {
            Ok(foreign3)
        } else {
            Ok(*s)
        }
    });
    assert!(matches!(r, Err(AutodiffError::TapeMismatch)));

    assert_eq!(before, host(&x.to_tensor()), "既存ノードの値は不変");
}

#[test]
fn m5_closure_err_propagates_and_leaves_nodes() {
    let tape = new_tape();
    let x = tape.var(&t(seq(8, 1.0), &[4, 2]));
    let before: Vec<u32> = host(&x.to_tensor()).iter().map(|f| f.to_bits()).collect();
    let len0 = tape.len();
    let calls = std::cell::Cell::new(0usize);
    let k = 2usize;
    let r = vmap(&tape, &x, 0, |s| {
        calls.set(calls.get() + 1);
        if calls.get() == k + 1 {
            Err(AutodiffError::InvalidArgument("test".into()))
        } else {
            Ok(s.tanh())
        }
    });
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(ref m)) if m == "test"));
    assert_eq!(calls.get(), k + 1);
    assert!(tape.len() > len0, "unbind 等のノードが残る");
    let after: Vec<u32> = host(&x.to_tensor()).iter().map(|f| f.to_bits()).collect();
    assert_eq!(before, after);
}

#[test]
fn m6_misc_and_differentiable() {
    let tape = new_tape();
    // 勾配追跡なし入力・B=1
    let c = tape.var_no_grad(&t(seq(3, 1.0), &[1, 3]));
    let got = vmap(&tape, &c, 0, |s| Ok(s.tanh())).unwrap();
    assert_vars_close(&got, &c.tanh(), "b=1 no grad");

    // 決定性（同一実行内の再実行比較。バッチなし実行との bit 一致ではない）
    let x = tape.var(&t(seq(12, 1.0), &[4, 3]));
    let a = vmap(&tape, &x, 0, |s| s.tanh().mul(s)).unwrap();
    let b = vmap(&tape, &x, 0, |s| s.tanh().mul(s)).unwrap();
    let bits = |v: &Var<'_>| {
        host(&v.to_tensor())
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&a), bits(&b));

    // 同じテープ上で微分可能
    let loss = a.sum(None).unwrap();
    let g = host(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
    let ref_loss = x.tanh().mul(&x).unwrap().sum(None).unwrap();
    let g_ref = host(tape.backward(&ref_loss).unwrap().get(&x).unwrap().unwrap());
    for (i, (&p, &q)) in g.iter().zip(g_ref.iter()).enumerate() {
        assert!(common::req2_close(f64::from(p), f64::from(q)), "grad[{i}]");
    }
}
