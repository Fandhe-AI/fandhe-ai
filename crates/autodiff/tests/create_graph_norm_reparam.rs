//! `Op::LocalResponseNorm`・`Op::WeightNorm`・`Op::SpectralNorm` の `create_graph`
//! （`Tape::backward_create_graph`）対応（イシュー #3794・親 #3727・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §36・§57）の受け入れ条件を直接検証する統合テスト。
//!
//! - Hessian を `Tape::backward` の 5 点中央差分（`create_graph` を経由しない独立経路）と REQ-2 統一
//!   複合判定（`common::req2_close`）で突合する（LRN は size 奇数・偶数・`size > C`・β・k・`α = 0`、
//!   WeightNorm は `dim` None／先頭／末尾で v と g の混合二階項を含む、SpectralNorm は `dim`・rank）。
//! - `cg.first_order()` が `Tape::backward` と bit 同一、`cg.grad(..)` が REQ-2 判定で一致する。
//! - 必須ベクトルで 1 階・`cg.grad`・孫テープ再生の 3 段が有限かつ一致する（または拒否されて `child` が空）。
//! - SpectralNorm の `create_graph`／`hvp` が power iteration の状態（`u`・`v`）を進めない。
//! - 拒否後の `child` が空で再試行でき、隣接する fail-closed Op（`Op::Custom`）が拒否のままである。
//! - 掃引で「受理なら全段が有限かつ一致、拒否なら `Err(Backward)` で `child` が空」を正のプローブで確かめ、
//!   受理と拒否の両方が現れることを固定する。
//!
//! 損失は `sum(tanh(y) ⊙ c)`（`c` は非一様な固定定数）を使う。`tanh` を挟むのは上流 `g` が入力に依存する
//! 経路（`g` を介した二階項）を検出するため。

#![allow(clippy::needless_range_loop)]

mod common;

use fandhe_ai_autodiff::weight_reparam_ops::SpectralNormState;
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn pattern(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + phase) * 0.9).sin() + 0.35 * (i as f32 + 1.0).cos() + 0.2)
        .collect()
}

fn dense(v: &Tensor<f32>) -> Vec<f32> {
    v.contiguous().as_slice().expect("contiguous").to_vec()
}

type Built<'t> = Result<Var<'t>, AutodiffError>;

const FD_STEP: f64 = 1e-2;

fn five_point_column(
    grad_at: &dyn Fn(&[f32]) -> Vec<f64>,
    x0: &[f32],
    i: usize,
    h: f64,
) -> Vec<f64> {
    let at = |d: f64| {
        let mut x = x0.to_vec();
        x[i] += d as f32;
        grad_at(&x)
    };
    let (p2, p1, m1, m2) = (at(2.0 * h), at(h), at(-h), at(-2.0 * h));
    (0..p1.len())
        .map(|j| (-p2[j] + 8.0 * p1[j] - 8.0 * m1[j] + m2[j]) / (12.0 * h))
        .collect()
}

fn finite_diff_hessian<F>(build: &F, x0: &[f32], shape: &[usize]) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let n = numel(shape);
    let grad_at = |x: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), shape));
        let loss = build(&tape, &xv).expect("forward 構築");
        let grads = tape.backward(&loss).expect("1 階 backward");
        let dx = grads.get(&xv).expect("get").expect("x は loss に到達する");
        dense(dx).into_iter().map(f64::from).collect()
    };
    let mut hessian = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let col = five_point_column(&grad_at, x0, i, FD_STEP);
        for j in 0..n {
            hessian[j][i] = col[j];
        }
    }
    hessian
}

fn extract_scalar<'c>(v: &Var<'c>, shape: &[usize], flat_idx: usize) -> Built<'c> {
    let mut rem = flat_idx;
    let mut idx = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        idx[d] = rem % shape[d];
        rem /= shape[d];
    }
    let mut cur = *v;
    for (dim, &i) in idx.iter().enumerate() {
        cur = cur.narrow(dim, i, 1)?;
    }
    cur.sum(None)
}

fn analytic_hessian<F>(build: &F, x0: &[f32], shape: &[usize]) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let n = numel(shape);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward 構築");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let gx = cg.grad(&x).expect("grad").expect("x は loss に到達する");
    let cx = cg.child_var(&x).expect("child_var").expect("写しあり");
    let mut hessian = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = extract_scalar(&gx, shape, j).expect("extract_scalar");
        let row_grads = child.backward(&gj).expect("child.backward");
        let row = dense(
            row_grads
                .get(&cx)
                .expect("get")
                .expect("2 階勾配は x へ到達する"),
        );
        for i in 0..n {
            hessian[j][i] = f64::from(row[i]);
        }
    }
    hessian
}

/// Hessian（有限差分）・1 階 bit 同一・`cg.grad` の REQ-2 一致を一括で検証する。
fn check_all<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let numeric = finite_diff_hessian(&build, x0, shape);
    let analytic = analytic_hessian(&build, x0, shape);
    for (j, (ra, rn)) in analytic.iter().zip(numeric.iter()).enumerate() {
        for (i, (&a, &n)) in ra.iter().zip(rn.iter()).enumerate() {
            assert!(
                common::req2_close(a, n),
                "{ctx}: hessian[{j}][{i}] analytic={a} numeric={n}"
            );
        }
    }
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward 構築");
    let plain = dense(
        tape.backward(&loss)
            .expect("backward")
            .get(&x)
            .expect("get")
            .expect("到達"),
    );
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let first = dense(cg.first_order().get(&x).expect("get").expect("到達"));
    assert_eq!(
        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{ctx}: first_order は Tape::backward と bit 同一"
    );
    let gx = dense(&cg.grad(&x).expect("grad").expect("到達").to_tensor());
    for (i, (&a, &b)) in gx.iter().zip(plain.iter()).enumerate() {
        assert!(
            common::req2_close(f64::from(a), f64::from(b)),
            "{ctx}: cg.grad[{i}]={a} vs backward={b}"
        );
        // V3 なので CPU（naive_ops）では 1 階と bit 一致する。
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{ctx}: cg.grad[{i}] は V3 で bit 一致"
        );
    }
}

fn weighted_tanh_sum<'t>(y: &Var<'t>, tape: &'t Tape, shape: &[usize]) -> Built<'t> {
    let c = tape.var_no_grad(&t(pattern(numel(shape), 1.0), shape));
    y.tanh().mul(&c)?.sum(None)
}

fn x_data(n: usize) -> Vec<f32> {
    pattern(n, 7.0)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v * 1.3 + (i % 3) as f32 * 0.4 - 0.5)
        .collect()
}

// --- 1. 1 階 bit 同一・cg.grad・Hessian: LocalResponseNorm ---------------------------------

#[test]
fn lrn_hessian_matches_finite_difference() {
    // C=3 に対し size=5（size > C）、奇数 3、偶数 2・4（非対称窓）。
    for shape in [vec![1usize, 3, 2], vec![2, 4, 2, 1]] {
        let c = shape[1];
        for size in [1usize, 2, 3, 4, 5] {
            for beta in [0.5f32, 0.75, 1.0, 2.0] {
                for k in [1.0f32, 2.0] {
                    let s = shape.clone();
                    check_all(
                        &format!("lrn/{shape:?}/size{size}/beta{beta}/k{k}/C{c}"),
                        |tape, x| {
                            let y = x.local_response_norm(size, 0.5, beta, k)?;
                            weighted_tanh_sum(&y, tape, &s)
                        },
                        &x_data(numel(&shape)),
                        &shape,
                    );
                }
            }
        }
    }
}

#[test]
fn lrn_alpha_zero_degenerate_branch_matches_finite_difference() {
    let shape = vec![1usize, 3, 2];
    let s = shape.clone();
    check_all(
        "lrn/alpha0",
        |tape, x| {
            let y = x.local_response_norm(3, 0.0, 0.75, 2.0)?;
            weighted_tanh_sum(&y, tape, &s)
        },
        &x_data(6),
        &shape,
    );
}

// --- 2. WeightNorm（v と g の混合二階項を含む） ----------------------------------------------

/// 平坦な入力 `x = [v..., g...]` を `v`・`g` に分けて `weight_norm` を作る。
fn weight_norm_loss<'t>(
    tape: &'t Tape,
    x: &Var<'t>,
    v_shape: &[usize],
    g_shape: &[usize],
    dim: Option<usize>,
) -> Built<'t> {
    let (nv, ng) = (numel(v_shape), numel(g_shape));
    let v = x.narrow(0, 0, nv)?.reshape(v_shape)?;
    let g = x.narrow(0, nv, ng)?.reshape(g_shape)?;
    let w = v.weight_norm(&g, dim)?;
    weighted_tanh_sum(&w, tape, v_shape)
}

#[test]
fn weight_norm_hessian_matches_finite_difference() {
    let v_shape = [2usize, 3, 2];
    let cases: [(Option<usize>, Vec<usize>); 4] = [
        (None, vec![]),
        (Some(0), vec![2, 1, 1]),
        (Some(1), vec![1, 3, 1]),
        (Some(2), vec![1, 1, 2]),
    ];
    for (dim, g_shape) in cases {
        let (nv, ng) = (numel(&v_shape), numel(&g_shape));
        let mut x0 = x_data(nv);
        x0.extend(pattern(ng, 4.0).into_iter().map(|g| g + 1.5));
        let (vs, gs) = (v_shape.to_vec(), g_shape.clone());
        check_all(
            &format!("weight_norm/dim{dim:?}"),
            |tape, x| weight_norm_loss(tape, x, &vs, &gs, dim),
            &x0,
            &[nv + ng],
        );
    }
}

// --- 3. SpectralNorm ---------------------------------------------------------------------

fn spectral_state(shape: &[usize], dim: usize) -> SpectralNormState {
    let h = shape[dim];
    let w = numel(shape) / h;
    SpectralNormState::from_vectors(shape, dim, &pattern(h, 2.0), &pattern(w, 6.0), 1, 1e-12)
        .expect("state")
}

#[test]
fn spectral_norm_hessian_matches_finite_difference() {
    for shape in [vec![2usize, 3], vec![3, 2, 2], vec![2, 2, 2, 2]] {
        for dim in 0..shape.len().min(2) {
            let s = shape.clone();
            let mut w0 = x_data(numel(&shape));
            // σ が 0 に近くならないよう u・v と正の相関を持つ成分を足す。
            let st = spectral_state(&shape, dim);
            let (h, w) = (shape[dim], numel(&shape) / shape[dim]);
            let inner: usize = shape[dim + 1..].iter().product();
            for i in 0..h {
                for j in 0..w {
                    w0[(j / inner * h + i) * inner + j % inner] += 2.0 * st.u()[i] * st.v()[j];
                }
            }
            check_all(
                &format!("spectral/{shape:?}/dim{dim}"),
                |tape, x| {
                    let mut st = spectral_state(&s, dim);
                    let y = x.spectral_norm(&mut st, false)?;
                    weighted_tanh_sum(&y, tape, &s)
                },
                &w0,
                &shape,
            );
        }
    }
}

#[test]
fn spectral_norm_create_graph_does_not_advance_power_iteration_state() {
    use fandhe_ai_autodiff::functional_ops::hvp;
    for training in [false, true] {
        let shape = [3usize, 2];
        let tape = Tape::new_with_ops(common::naive_ops());
        let mut st = spectral_state(&shape, 0);
        let mut w0 = x_data(6);
        w0[0] += 2.0;
        let w = tape.var(&t(w0, &shape));
        let y = w.spectral_norm(&mut st, training).expect("forward");
        let loss = weighted_tanh_sum(&y, &tape, &shape).expect("loss");
        let (u_before, v_before) = (st.u().to_vec(), st.v().to_vec());
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

        let child = Tape::new_with_ops(common::naive_ops());
        let _cg = tape
            .backward_create_graph(&loss, &child)
            .expect("create_graph");
        assert_eq!(bits(st.u()), bits(&u_before), "training={training}: u 不変");
        assert_eq!(bits(st.v()), bits(&v_before), "training={training}: v 不変");

        let child2 = Tape::new_with_ops(common::naive_ops());
        let dir = t(pattern(6, 3.0), &shape);
        hvp(&tape, &loss, &w, &dir, &child2).expect("hvp");
        assert_eq!(bits(st.u()), bits(&u_before));
        assert_eq!(bits(st.v()), bits(&v_before));
    }
}

// --- 4. 拒否・再試行・fail-closed -----------------------------------------------------------

fn expect_rejected(tape: &Tape, loss: &Var<'_>, child: &Tape, ctx: &str) {
    let err = tape
        .backward_create_graph(loss, child)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)), "{ctx}: {err:?}");
    assert!(child.is_empty(), "{ctx}: 拒否後の child は空");
}

#[test]
fn norm_reparam_unsafe_inputs_rejected_child_empty_and_retry() {
    // LRN: k=0 かつ窓内が全ゼロ（D=0）、α<0 で D<0、±inf。
    let shape = [1usize, 3, 2];
    let cases: [(&str, Vec<f32>, f32, f32); 4] = [
        ("lrn/k0-zero-window", vec![0.0; 6], 0.5, 0.0),
        ("lrn/alpha-neg", vec![3.0; 6], -1.0, 0.1),
        (
            "lrn/inf",
            vec![1.0, f32::INFINITY, 1.0, 1.0, 1.0, 1.0],
            0.5,
            1.0,
        ),
        ("lrn/k0-alpha0", vec![1.0; 6], 0.0, 0.0),
    ];
    for (ctx, x0, alpha, k) in cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0, &shape));
        let y = x.local_response_norm(3, alpha, 0.75, k).expect("forward");
        let loss = weighted_tanh_sum(&y, &tape, &shape).expect("loss");
        expect_rejected(&tape, &loss, &child, ctx);
    }
    // 拒否した child は再利用でき、安全な入力なら成功する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 6], &shape));
    let y = x.local_response_norm(3, 0.5, 0.75, 0.0).expect("forward");
    let loss = weighted_tanh_sum(&y, &tape, &shape).expect("loss");
    expect_rejected(&tape, &loss, &child, "retry/first");
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let x2 = tape2.var(&t(x_data(6), &shape));
    let y2 = x2.local_response_norm(3, 0.5, 0.75, 1.0).expect("forward");
    let loss2 = weighted_tanh_sum(&y2, &tape2, &shape).expect("loss");
    tape2
        .backward_create_graph(&loss2, &child)
        .expect("同じ child で安全な入力は成功");

    // WeightNorm: ゼロノルムのグループ。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let v = tape.var(&t(vec![0.0, 0.0, 1.0, 2.0], &[2, 2]));
    let g = tape.var(&t(vec![1.0, 1.0], &[2, 1]));
    let w = v.weight_norm(&g, Some(0)).expect("forward");
    let loss = weighted_tanh_sum(&w, &tape, &[2, 2]).expect("loss");
    expect_rejected(&tape, &loss, &child, "weight_norm/zero-group");

    // SpectralNorm: σ=0（W が u・v と直交）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let mut st = SpectralNormState::from_vectors(&[2, 2], 0, &[1.0, 0.0], &[1.0, 0.0], 1, 1e-12)
        .expect("state");
    let w = tape.var(&t(vec![0.0, 1.0, 1.0, 1.0], &[2, 2]));
    let y = w.spectral_norm(&mut st, false).expect("forward");
    let loss = weighted_tanh_sum(&y, &tape, &[2, 2]).expect("loss");
    expect_rejected(&tape, &loss, &child, "spectral/sigma0");
}

#[test]
fn norm_reparam_neighbouring_unsupported_op_stays_fail_closed() {
    // 隣接する fail-closed Op として `Op::Custom`（恒等のユーザー定義関数）を使う。
    use fandhe_ai_autodiff::CustomFunction;
    struct Identity;
    impl CustomFunction for Identity {
        fn name(&self) -> &str {
            "identity-for-fail-closed"
        }
        fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
            Ok(input_shapes[0].to_vec())
        }
        fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
            Ok(inputs[0].clone())
        }
        fn backward(
            &self,
            _inputs: &[&Tensor<f32>],
            _output: &Tensor<f32>,
            upstream: &Tensor<f32>,
            _requires_grad: &[bool],
        ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
            Ok(vec![Some(upstream.clone())])
        }
    }
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x_data(6), &[1, 3, 2]));
    let y = tape
        .custom(std::sync::Arc::new(Identity), &[x])
        .expect("custom");
    let z = y.local_response_norm(3, 0.5, 0.75, 1.0).expect("forward");
    let loss = weighted_tanh_sum(&z, &tape, &[1, 3, 2]).expect("loss");
    expect_rejected(&tape, &loss, &child, "custom upstream");
}

// --- 5. 必須ベクトルと掃引（受理なら全段が有限・一致、拒否なら child 空） -----------------------

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Outcome {
    Accepted,
    Rejected,
}

/// 1 階・`cg.grad`・二階（seed 全 1）・孫テープ再生が有限で一致するか、拒否されて child が空かを確かめる。
fn probe<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize]) -> Outcome
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward 構築");
    let first = match tape.backward(&loss) {
        Ok(g) => dense(g.get(&x).expect("get").expect("到達")),
        Err(_) => return Outcome::Rejected,
    };
    let cg = match tape.backward_create_graph(&loss, &child) {
        Ok(cg) => cg,
        Err(AutodiffError::Backward(_)) => {
            assert!(child.is_empty(), "{ctx}: 拒否後の child は空");
            return Outcome::Rejected;
        }
        Err(e) => panic!("{ctx}: 想定外のエラー {e:?}"),
    };
    let gx = cg.grad(&x).expect("grad").expect("到達");
    let gx_v = dense(&gx.to_tensor());
    assert!(first.iter().all(|v| v.is_finite()), "{ctx}: 1 階が有限");
    for (i, (&a, &b)) in gx_v.iter().zip(&first).enumerate() {
        assert!(
            common::req2_close(f64::from(a), f64::from(b)),
            "{ctx}: cg.grad[{i}]={a} vs 1 階={b}"
        );
    }
    // 二階: sum(cg.grad) を child 上で微分する。
    let s = gx.sum(None).expect("sum");
    let second = child.backward(&s).expect("child.backward");
    let cx = cg.child_var(&x).expect("child_var").expect("写し");
    let sec = dense(second.get(&cx).expect("get").expect("到達"));
    assert!(
        sec.iter().all(|v| v.is_finite()),
        "{ctx}: 二階が有限: {sec:?}"
    );
    // 孫テープ再生: child 上の s を更に create_graph し、その 1 階が二階と一致する。
    let grand = Tape::new_with_ops(common::naive_ops());
    match child.backward_create_graph(&s, &grand) {
        Ok(cg2) => {
            let g2 = dense(cg2.first_order().get(&cx).expect("get").expect("到達"));
            assert!(g2.iter().all(|v| v.is_finite()), "{ctx}: 孫の 1 階が有限");
            for (i, (&a, &b)) in g2.iter().zip(&sec).enumerate() {
                assert!(
                    common::req2_close(f64::from(a), f64::from(b)),
                    "{ctx}: 孫の 1 階[{i}]={a} vs 二階={b}"
                );
            }
        }
        Err(AutodiffError::Backward(_)) => {
            assert!(grand.is_empty(), "{ctx}: 孫の拒否後も空");
        }
        Err(e) => panic!("{ctx}: 孫で想定外のエラー {e:?}"),
    }
    Outcome::Accepted
}

#[test]
fn norm_reparam_mandatory_vectors_finite_in_three_stages() {
    let shape = [1usize, 3, 2];
    let lrn_probe = |ctx: &str, x0: Vec<f32>, size, alpha, beta, k| {
        let s = shape;
        probe(
            ctx,
            move |tape, x| {
                let y = x.local_response_norm(size, alpha, beta, k)?;
                weighted_tanh_sum(&y, tape, &s)
            },
            &x0,
            &shape,
        )
    };
    // 有限大振幅・偶数 size の非対称窓・小スケール。
    let _ = lrn_probe(
        "lrn/x=1e20",
        vec![1e20, -1e20, 0.5e20, 1e20, 2e20, 1e20],
        3,
        0.5,
        0.75,
        1.0,
    );
    let _ = lrn_probe("lrn/tiny", vec![1e-30; 6], 3, 0.5, 0.75, 1.0);
    assert_eq!(
        lrn_probe("lrn/even-asym", x_data(6), 2, 0.5, 1.0, 1.0),
        Outcome::Accepted
    );
    assert_eq!(
        lrn_probe("lrn/benign", x_data(6), 3, 0.5, 0.75, 1.0),
        Outcome::Accepted
    );
    // k=0 でも窓が非ゼロなら D>0 で受理。
    assert_eq!(
        lrn_probe("lrn/k0-nonzero", x_data(6), 3, 0.5, 0.75, 0.0),
        Outcome::Accepted
    );

    // WeightNorm: dim=None・v=1e20・小スケール・大きな g。
    let wn = |ctx: &str, v0: Vec<f32>, g0: f32, dim: Option<usize>, gshape: Vec<usize>| {
        let nv = v0.len();
        let ng = numel(&gshape);
        let mut x0 = v0;
        x0.extend(std::iter::repeat_n(g0, ng));
        let vs = [2usize, 2];
        probe(
            ctx,
            move |tape, x| weight_norm_loss(tape, x, &vs, &gshape, dim),
            &x0,
            &[nv + ng],
        )
    };
    assert_eq!(
        wn("wn/benign", vec![1.0, 2.0, 3.0, -1.0], 1.5, None, vec![]),
        Outcome::Accepted
    );
    let _ = wn(
        "wn/v=1e20",
        vec![1e20, -2e20, 1e20, 1e20],
        1.0,
        Some(0),
        vec![2, 1],
    );
    let _ = wn(
        "wn/small-v",
        vec![1e-20, 2e-20, 3e-20, -1e-20],
        1.0,
        None,
        vec![],
    );
    let _ = wn("wn/big-g", vec![1.0, 2.0, 3.0, -1.0], 3e38, None, vec![]);
}

#[test]
fn norm_reparam_sweep_every_structure_path_is_finite_or_rejected() {
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut tally = |o: Outcome| match o {
        Outcome::Accepted => accepted += 1,
        Outcome::Rejected => rejected += 1,
    };
    let lrn_shape = [1usize, 4, 2];
    for &scale in &[1e-30f32, 1e-3, 1.0, 1e3, 1e20, 3e38] {
        for &(size, alpha, beta, k) in &[
            (3usize, 0.5f32, 0.75f32, 1.0f32),
            (4, 1e-4, 2.0, 2.0),
            (2, 0.5, 1.0, 0.0),
            (5, 0.0, 0.75, 1.0),
        ] {
            let x0: Vec<f32> = x_data(8).iter().map(|v| v * scale).collect();
            tally(probe(
                &format!("sweep/lrn/{scale}/{size}/{alpha}/{beta}/{k}"),
                |tape, x| {
                    let y = x.local_response_norm(size, alpha, beta, k)?;
                    weighted_tanh_sum(&y, tape, &lrn_shape)
                },
                &x0,
                &lrn_shape,
            ));
        }
    }
    for &scale in &[1e-30f32, 1e-3, 1.0, 1e3, 1e20] {
        for &g0 in &[1e-3f32, 1.0, 1e30] {
            let v0: Vec<f32> = x_data(6).iter().map(|v| v * scale).collect();
            let mut x0 = v0;
            x0.extend([g0, g0 * 0.5]);
            tally(probe(
                &format!("sweep/wn/{scale}/{g0}"),
                |tape, x| weight_norm_loss(tape, x, &[2, 3], &[2, 1], Some(0)),
                &x0,
                &[8],
            ));
        }
        let shape = [2usize, 3];
        let mut w0: Vec<f32> = x_data(6).iter().map(|v| v * scale).collect();
        let st = spectral_state(&shape, 0);
        for i in 0..2 {
            for j in 0..3 {
                w0[i * 3 + j] += 2.0 * scale * st.u()[i] * st.v()[j];
            }
        }
        tally(probe(
            &format!("sweep/spectral/{scale}"),
            |tape, x| {
                let mut st = spectral_state(&shape, 0);
                let y = x.spectral_norm(&mut st, false)?;
                weighted_tanh_sum(&y, tape, &shape)
            },
            &w0,
            &shape,
        ));
    }
    assert!(
        accepted > 0,
        "受理側が 1 件も無い（掃引が拒否に偏っている）"
    );
    assert!(rejected > 0, "拒否側が 1 件も無い（判定が効いていない）");
}

/// PowScalar の微分係数 `e·d^(e-1)` 単体が f32 で overflow する入力（cotangent との f64 積の上界は
/// 通る）を事前に拒否する回帰テスト（PR #3950 codex 指摘 P1）。実装は係数を f32 で先に評価するため、
/// 受理すると二階微分が非有限になる。
#[test]
fn lrn_pow_scalar_coefficient_f32_overflow_is_rejected_child_empty() {
    let shape = [1usize, 1, 1];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1e20], &shape));
    let y = x.local_response_norm(1, 1e-30, -0.5, 0.0).expect("forward");
    let loss = y.sum(None).expect("loss");
    expect_rejected(&tape, &loss, &child, "lrn/pow-coefficient-overflow");
}

// --- 5. レビュー指摘の回帰（#3950） ----------------------------------------------------------

/// g が NaN でも入口・段検査は拒否せず、1 階と同じく NaN を伝播させて受理する。
#[test]
fn weight_norm_nan_g_propagates_instead_of_rejecting() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let v = tape.var(&t(vec![3.0, 4.0], &[2]));
    let g = tape.var(&t(vec![f32::NAN], &[]));
    let w = v.weight_norm(&g, None).expect("forward");
    let loss = w.sum(None).expect("loss");
    tape.backward_create_graph(&loss, &child)
        .expect("NaN の g は拒否せず伝播させる");
}

/// 上流勾配 up が全ゼロでも、σ の相殺で up への微分係数が REQ-2 を外れる SpectralNorm は拒否する。
#[test]
fn spectral_norm_zero_upstream_with_cancelling_sigma_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let mut st = SpectralNormState::from_vectors(&[2, 2], 0, &[0.6, 0.8], &[0.6, 0.8], 1, 1e-12)
        .expect("state");
    let w = tape.var(&t(vec![1.0, 1.0, 1.0, -2.06249], &[2, 2]));
    let a = tape.var(&t(vec![0.0], &[1]));
    let y = w.spectral_norm(&mut st, false).expect("forward");
    let y00 = y
        .narrow(0, 0, 1)
        .and_then(|r| r.narrow(1, 0, 1))
        .expect("narrow");
    let loss = y00
        .reshape(&[1])
        .and_then(|r| r.mul(&a))
        .and_then(|r| r.sum(None))
        .expect("loss");
    expect_rejected(&tape, &loss, &child, "spectral/zero-up-cancel");
}
/// PR #3950 codex 指摘 P1: WeightNorm の中間商 g/n_s の underflow（二階が 0 になる）を拒否する。
#[test]
fn weight_norm_div_partial_underflow_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let v = tape.var(&t(vec![1e-30; 16], &[16]));
    let g = tape.var(&t(vec![f32::from_bits(1)], &[]));
    let w = v.weight_norm(&g, None).expect("forward");
    let loss = w.narrow(0, 0, 1).and_then(|r| r.sum(None)).expect("loss");
    expect_rejected(&tape, &loss, &child, "weight_norm/div-underflow");
}

/// PR #3950 codex 指摘 P1: Div の微分係数単体の overflow（cotangent が 0 でも 0·inf=NaN）を拒否する。
#[test]
fn weight_norm_div_partial_overflow_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let v = tape.var(&t(vec![3.0 * 2f32.powi(65)], &[1]));
    let g = tape.var(&t(vec![2e38], &[]));
    let a = tape.var(&t(vec![0.0], &[1]));
    let w = v.weight_norm(&g, None).expect("forward");
    let loss = w.mul(&a).and_then(|r| r.sum(None)).expect("loss");
    expect_rejected(&tape, &loss, &child, "weight_norm/div-overflow");
}

/// PR #3950 codex 指摘 P1: SpectralNorm の σ_s 側 Div 係数の overflow を拒否する。
#[test]
fn spectral_norm_div_partial_overflow_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let mut st = SpectralNormState::from_vectors(&[1, 2], 0, &[1.0], &[4e-38, 1.0], 1, 1e-12)
        .expect("state");
    let w = tape.var(&t(vec![0.75 * 2f32.powi(127), 0.0], &[1, 2]));
    let a = tape.var(&t(vec![0.0], &[1]));
    let y = w.spectral_norm(&mut st, false).expect("forward");
    let loss = y
        .narrow(1, 0, 1)
        .and_then(|r| r.reshape(&[1]))
        .and_then(|r| r.mul(&a))
        .and_then(|r| r.sum(None))
        .expect("loss");
    expect_rejected(&tape, &loss, &child, "spectral/div-overflow");
}

/// PR #3950 codex 指摘 P1: 上流 up が全ゼロでも、v̂² が 1 に丸まり up への二階係数
/// （混合項 g·v₁²/‖v‖³ ≈ 100）が消える WeightNorm は拒否する（up の値に依存しない相殺検査）。
#[test]
fn weight_norm_zero_upstream_with_cancelling_vhat_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let v = tape.var(&t(vec![1.0, 1e-4], &[2]));
    let g = tape.var(&t(vec![1e10], &[]));
    let a = tape.var(&t(vec![0.0], &[1]));
    let w = v.weight_norm(&g, None).expect("forward");
    let loss = w
        .narrow(0, 0, 1)
        .and_then(|r| r.mul(&a))
        .and_then(|r| r.sum(None))
        .expect("loss");
    expect_rejected(&tape, &loss, &child, "weight_norm/zero-up-cancel");
}

/// PR #3950 codex 指摘 P1: LRN の対角係数 a - 2ρβ·x_s²·b が f32 の d_s=1+k/x² の丸めで 0 へ消える
/// （x=2^-33・k=1e-28。解析値 k/(x²+k)^1.5 ≈ 63）入力を、up=0 でも拒否する。
#[test]
fn lrn_zero_upstream_with_cancelling_diagonal_is_rejected() {
    let shape = [1usize, 1, 1];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![(-33f32).exp2()], &shape));
    let a = tape.var(&t(vec![0.0], &[1]));
    let y = x.local_response_norm(1, 1.0, 0.5, 1e-28).expect("forward");
    let loss = y
        .reshape(&[1])
        .and_then(|r| r.mul(&a))
        .and_then(|r| r.sum(None))
        .expect("loss");
    expect_rejected(&tape, &loss, &child, "lrn/zero-up-diag-cancel");
}

/// PR #3950 codex 指摘 P1: SpectralNorm の対角係数 1 - Ŵ_i·P_i（I - ŴP）が σ_s の精度検査を通っても
/// 相殺する入力（W=[2^-33,2^-33]・u=[1]・v=[1,1e-8]。解析値 ≈ 86）を、up=0 でも拒否する。
#[test]
fn spectral_norm_zero_upstream_with_cancelling_coefficient_matrix_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let mut st =
        SpectralNormState::from_vectors(&[1, 2], 0, &[1.0], &[1.0, 1e-8], 1, 1e-12).expect("state");
    let w2 = (-33f32).exp2();
    let w = tape.var(&t(vec![w2, w2], &[1, 2]));
    let a = tape.var(&t(vec![0.0], &[1]));
    let y = w.spectral_norm(&mut st, false).expect("forward");
    let y00 = y
        .narrow(0, 0, 1)
        .and_then(|r| r.narrow(1, 0, 1))
        .expect("narrow");
    let loss = y00
        .reshape(&[1])
        .and_then(|r| r.mul(&a))
        .and_then(|r| r.sum(None))
        .expect("loss");
    expect_rejected(&tape, &loss, &child, "spectral/zero-up-matrix-cancel");
}

/// PR #3950 codex 指摘 P1（スケール復元の逆伝播の underflow）: x=2^-80・size=1・β=−1・k=0 では
/// `s^{-2β} = 2^-160` を 2^-60・2^-100 のチャンクで戻すため、seed 1 の cotangent が f32 で 0 になり、
/// 構造の二階は 0 になる（解析値は `6·α·x`。順方向の値は正しいので forward の照合では見えない）。
/// α を振り、受理なら二階が解析値と REQ-2 で一致し、拒否なら `child` が空であること、Codex の入力
/// （α=1e36。解析値 ≈ 4.96e12）は拒否され、受理側（解析値が絶対許容内に収まる小さい α）も現れることを確かめる。
#[test]
fn lrn_scale_restoration_cotangent_underflow_is_rejected_or_exact() {
    let shape = [1usize, 1, 1];
    let x0 = (-80f32).exp2();
    let mut outcomes = Vec::new();
    for &alpha in &[1.0f32, 1e20, 1e30, 1e36] {
        let ctx = format!("lrn/scale-restoration-underflow/alpha={alpha:e}");
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![x0], &shape));
        let y = x.local_response_norm(1, alpha, -1.0, 0.0).expect("forward");
        let loss = y.sum(None).expect("loss");
        match tape.backward_create_graph(&loss, &child) {
            Err(AutodiffError::Backward(_)) => {
                assert!(child.is_empty(), "{ctx}: 拒否後の child は空");
                outcomes.push(Outcome::Rejected);
            }
            Err(e) => panic!("{ctx}: 想定外のエラー {e:?}"),
            Ok(cg) => {
                let gx = cg.grad(&x).expect("grad").expect("到達");
                let s = gx.sum(None).expect("sum");
                let second = child.backward(&s).expect("child.backward");
                let cx = cg.child_var(&x).expect("child_var").expect("写し");
                let h = dense(second.get(&cx).expect("get").expect("到達"))[0];
                let exact = 6.0 * f64::from(alpha) * f64::from(x0);
                assert!(
                    common::req2_close(f64::from(h), exact),
                    "{ctx}: 受理したなら二階 {h:e} は解析値 {exact:e} と REQ-2 で一致する"
                );
                outcomes.push(Outcome::Accepted);
            }
        }
    }
    assert_eq!(
        outcomes.last(),
        Some(&Outcome::Rejected),
        "Codex の入力（α=1e36）は拒否される"
    );
    assert!(
        outcomes.contains(&Outcome::Accepted),
        "受理側が無い（判定が過剰）: {outcomes:?}"
    );
}

/// LRN の総括判定の計算量の上限が窓内の二重走査（seed ごとに反転窓 × 前方窓）を数える（Codex P2）。
/// `C = 4096`・`size = 8191`（窓は `C` に打ち切って両窓とも幅 4096）・全ゼロ・`α = 0.5`・`β = 1`・
/// `k = 1e10`・損失は出力の和。数値の検査では拒否されない入力で、二重走査の合計は約 1374 億回になる。
/// 上限が出力要素数 × 帯幅だけを数えると判定へ入って長時間止まるので、走査量の上限で短時間に拒否される
/// こと（`child` が空）を確かめる。
#[test]
fn lrn_judge_budget_counts_window_double_scan() {
    let (c, size) = (4096usize, 8191usize);
    let shape = [1usize, c, 1];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; c], &shape));
    let y = x
        .local_response_norm(size, 0.5, 1.0, 1e10)
        .expect("forward");
    let loss = y.sum(None).expect("loss");
    let start = std::time::Instant::now();
    let err = tape
        .backward_create_graph(&loss, &child)
        .map(|_| ())
        .unwrap_err();
    let elapsed = start.elapsed();
    match &err {
        AutodiffError::Backward(msg) => {
            assert!(
                msg.contains("LocalResponseNorm") && msg.contains("2^27"),
                "{msg}"
            );
        }
        e => panic!("想定外のエラー {e:?}"),
    }
    assert!(child.is_empty(), "拒否後の child は空");
    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "走査量の上限で短時間に拒否される: {elapsed:?}"
    );
}

// --- NaN の到達範囲（Codex P1・PR #3950）---------------------------------------------------------
//
// NaN は伝播させる契約だが、NaN の届かない有限の要素（LRN は窓、WeightNorm は g が NaN のグループの
// g に依存しない要素）は通常どおり判定する。判定が外すのは f32 の再現と参照の両方が NaN の要素だけ。

/// `backward_create_graph` の結果（受理なら 1 階と、seed `e_j` ごとの二階の行）。
type SecondRows = Result<(Vec<f32>, Vec<Vec<f32>>), AutodiffError>;

/// 損失 `build(x)` の 1 階 `∂L/∂x` と、その各要素を seed にした `x` への二階の行を返す。拒否なら `child` が空。
fn second_rows<F>(x0: &[f32], shape: &[usize], mut build: F) -> SecondRows
where
    F: for<'t> FnMut(&Var<'t>) -> Built<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&x).expect("forward");
    let cg = match tape.backward_create_graph(&loss, &child) {
        Ok(cg) => cg,
        Err(e) => {
            assert!(matches!(e, AutodiffError::Backward(_)), "{e:?}");
            assert!(child.is_empty(), "拒否後の child は空");
            return Err(e);
        }
    };
    let gx = cg.grad(&x).expect("grad").expect("到達");
    let cx = cg.child_var(&x).expect("child_var").expect("写し");
    let first = dense(&gx.to_tensor());
    let rows = (0..numel(shape))
        .map(|j| {
            let s = extract_scalar(&gx, shape, j).expect("seed");
            let g = child.backward(&s).expect("child.backward");
            dense(g.get(&cx).expect("get").expect("到達"))
        })
        .collect();
    Ok((first, rows))
}

/// Codex の入力（`x = [NaN, 1]`・`size = 1`・`α = 1e-30`・`β = -0.5`・`k = 0`・損失は出力の和）。窓は互いに
/// 独立で、有限側の `b = d_s^{-0.5}` の微分係数 `-0.5·d_s^{-1.5}` が f32 で overflow する。列に NaN が
/// あっても有限の窓は判定されるので拒否される（以前は列ごと判定を外して受理し、有限側の二階が非有限だった）。
#[test]
fn lrn_nan_column_still_judges_independent_finite_window() {
    let shape = [1usize, 2, 1];
    let r = second_rows(&[f32::NAN, 1.0], &shape, |x| {
        x.local_response_norm(1, 1e-30, -0.5, 0.0)?.sum(None)
    });
    assert!(
        r.is_err(),
        "有限の窓の f32 overflow は NaN の列でも拒否される"
    );
}

fn lrn_sum<'t>(x: &Var<'t>, size: usize, beta: f32) -> Built<'t> {
    x.local_response_norm(size, 1.0, beta, 2.0)?.sum(None)
}

/// 正のプローブ: NaN の届かない窓の二階は有限で、NaN を有限値へ置き換えた入力（列の最大値は同じなので
/// スケールも同じ）の二階と REQ-2 で一致する。NaN の窓の 1 階は NaN。`size = 1`（窓が独立）と `size = 2`・
/// `size = 3`（NaN から最も遠いチャネルの自分への二階）。
#[test]
fn lrn_nan_column_finite_window_matches_reference() {
    // β = −1 は `−β−1 = 0` で `pow_grad` が定数 0 を返す分岐（f32・参照とも NaN を伝えない）。
    // 偶数 `size = 2`（forward 窓と反転窓がずれる）と、冪の指数 0 で NaN が消える β ∈ {0, −1, −2} も含む。
    let cases: [(Vec<f32>, usize, usize, f32); 8] = [
        (vec![f32::NAN, 1.0], 1, 1, 0.75),
        (vec![f32::NAN, 0.6, -0.3, 0.8, 0.4], 3, 4, 0.75),
        (vec![f32::NAN, 1.0, 1.0, 1.0, 1.0], 1, 4, -1.0),
        (vec![f32::NAN, 0.6, -0.3, 0.8, 0.4], 3, 4, -1.0),
        (vec![f32::NAN, 0.6, -0.3, 0.8, 0.4], 2, 4, 0.75),
        (vec![f32::NAN, 0.6, -0.3, 0.8, 0.4], 2, 4, -1.0),
        (vec![f32::NAN, 0.6, -0.3, 0.8, 0.4], 2, 4, 0.0),
        (vec![f32::NAN, 0.6, -0.3, 0.8, 0.4], 2, 4, -2.0),
    ];
    for (x0, size, far, beta) in cases {
        let shape = [1usize, x0.len(), 1];
        let (first, rows) =
            second_rows(&x0, &shape, |x| lrn_sum(x, size, beta)).expect("NaN の届かない窓は受理");
        let mut x_ref = x0.clone();
        x_ref[0] = 0.5;
        let (first_ref, rows_ref) =
            second_rows(&x_ref, &shape, |x| lrn_sum(x, size, beta)).expect("参照");
        assert!(first[0].is_nan(), "size{size}: NaN の窓の 1 階は NaN");
        assert!(
            common::req2_close(f64::from(first[far]), f64::from(first_ref[far])),
            "size{size}: 有限の窓の 1 階"
        );
        let (h, h_ref) = (rows[far][far], rows_ref[far][far]);
        assert!(h.is_finite(), "size{size}: 有限の窓の二階は有限: {h:e}");
        assert!(
            common::req2_close(f64::from(h), f64::from(h_ref)),
            "size{size}: 二階 {h:e} と参照 {h_ref:e}"
        );
    }
}

/// WeightNorm の g が NaN のグループも、g に依存しない二階（`dg` への seed の `v` への要素）は判定する。
/// グループ 0（g = NaN）の `v = [1, 1e-4]`・上流 `[1e4, 0]` は、f32 で `v̂₀² = 1` に丸まり `dg` の `v₀` への
/// 二階（解析値 `up₀·v̂₁²/n ≈ 1e-4`、項の大きさ約 1e4）が相殺で消えるので拒否される（以前は g が NaN の
/// グループを判定から外して受理していた）。グループ 1 は有限で無害。
#[test]
fn weight_norm_nan_g_group_still_judges_g_independent_entries() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let v = tape.var(&t(vec![1.0, 1e-4, 0.6, 0.8], &[2, 2]));
    let g = tape.var(&t(vec![f32::NAN, 0.5], &[2, 1]));
    let a = tape.var(&t(vec![1e4, 0.0, 1.0, 1.0], &[2, 2]));
    let w = v.weight_norm(&g, Some(0)).expect("forward");
    let loss = w.mul(&a).and_then(|r| r.sum(None)).expect("loss");
    expect_rejected(&tape, &loss, &child, "weight_norm/nan-g-cancel");
}

/// 正のプローブ: g が NaN のグループで、`dv` は NaN、`dg` と `dg` の `v` への二階は有限で、g を有限値へ
/// 置き換えた入力と REQ-2 で一致する（`dg = Σup⊙v̂` は g に依存しない）。ほかのグループは有限。
#[test]
fn weight_norm_nan_g_group_finite_entries_match_reference() {
    let run = |g0: f32| {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let v = tape.var(&t(vec![0.3, -0.7, 0.6, 0.8], &[2, 2]));
        let g = tape.var(&t(vec![g0, 0.5], &[2, 1]));
        let a = tape.var(&t(vec![0.9, -0.4, 1.0, 1.0], &[2, 2]));
        let w = v.weight_norm(&g, Some(0)).expect("forward");
        let loss = w.mul(&a).and_then(|r| r.sum(None)).expect("loss");
        let cg = tape
            .backward_create_graph(&loss, &child)
            .expect("g が NaN でも受理");
        let gv = cg.grad(&v).expect("grad").expect("到達");
        let gg = cg.grad(&g).expect("grad").expect("到達");
        let cv = cg.child_var(&v).expect("child_var").expect("写し");
        let dv = dense(&gv.to_tensor());
        let dg = dense(&gg.to_tensor());
        let s = extract_scalar(&gg, &[2, 1], 0).expect("seed");
        let h = dense(
            child
                .backward(&s)
                .expect("child.backward")
                .get(&cv)
                .expect("get")
                .expect("到達"),
        );
        (dv, dg, h)
    };
    let (dv, dg, h) = run(f32::NAN);
    let (dv_ref, dg_ref, h_ref) = run(1.3);
    assert!(
        dv[0].is_nan() && dv[1].is_nan(),
        "g が NaN のグループの dv は NaN"
    );
    assert!(
        dv[2].is_finite() && dv[3].is_finite(),
        "ほかのグループは有限"
    );
    assert!(common::req2_close(f64::from(dg[0]), f64::from(dg_ref[0])));
    for i in 0..2 {
        assert!(h[i].is_finite(), "dg の v への二階は有限: {:e}", h[i]);
        assert!(
            common::req2_close(f64::from(h[i]), f64::from(h_ref[i])),
            "dg の v への二階 {:e} と参照 {:e}",
            h[i],
            h_ref[i]
        );
        assert_eq!(h[2 + i], 0.0, "グループをまたぐ二階は 0");
    }
    let _ = dv_ref;
}

/// SpectralNorm の W に NaN が 1 つあると `σ_s = ΣW_s⊙P` を通ってテンソル全体へ届く（1 階・二階とも
/// 全要素 NaN）。テンソル単位で判定を外すのは到達範囲と一致し、有限の要素を検査なしで通さない。
#[test]
fn spectral_norm_single_nan_reaches_every_entry() {
    let mut st = SpectralNormState::from_vectors(&[2, 2], 0, &[0.6, 0.8], &[0.8, 0.6], 1, 1e-12)
        .expect("state");
    let r = second_rows(&[0.9, f32::NAN, 0.5, 0.6], &[2, 2], |w| {
        w.spectral_norm(&mut st, false)?.sum(None)
    })
    .expect("NaN は伝播させる");
    let (first, rows) = r;
    assert!(first.iter().all(|v| v.is_nan()), "1 階: {first:?}");
    assert!(rows.iter().flatten().all(|v| v.is_nan()), "二階: {rows:?}");
}

// --- ホストの 1 階の非有限・NaN を含む列の除外と順序（Codex P1／P2・Bugbot・PR #3950）-------------

/// `α = 0` の分岐でも、有限の入力からホストの 1 階に演算で生まれた NaN（`up·x` の f32 積が inf になり、
/// `cross = 0` との積で NaN）を拒否する（Codex P1。以前は構造だけを見て受理し、`cg.grad` に NaN が載った）。
#[test]
fn lrn_alpha_zero_rejects_generated_nonfinite_host_grad() {
    let shape = [1usize, 1, 1];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1e20], &shape));
    let a = tape.var(&t(vec![1e20], &shape));
    let y = x.local_response_norm(1, 0.0, 1.0, 1e20).expect("forward");
    let loss = y.mul(&a).and_then(|r| r.sum(None)).expect("loss");
    assert!(
        dense(&loss.to_tensor())[0].is_finite(),
        "forward と損失は有限"
    );
    let err = tape
        .backward_create_graph(&loss, &child)
        .map(|_| ())
        .unwrap_err();
    match &err {
        AutodiffError::Backward(msg) => assert!(msg.contains("演算で生まれた非有限"), "{msg}"),
        e => panic!("想定外のエラー {e:?}"),
    }
    assert!(child.is_empty(), "拒否後の child は空");
}

/// 正のプローブ（`α = 0`）: 積が f32 に収まる入力は受理され、`cg.grad` は `up·k^{-β}` で有限。x に NaN が
/// あっても（`size = 1`）NaN の位置だけが NaN で、ほかは同じ値。
#[test]
fn lrn_alpha_zero_accepts_finite_and_propagates_input_nan() {
    let shape = [1usize, 3, 1];
    let c = [0.5f32, -2.0, 1.5];
    for x0 in [vec![0.3f32, 1.0, -0.7], vec![f32::NAN, 1.0, -0.7]] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0.clone(), &shape));
        let a = tape.var(&t(c.to_vec(), &shape));
        let y = x.local_response_norm(1, 0.0, 0.75, 2.0).expect("forward");
        let loss = y.mul(&a).and_then(|r| r.sum(None)).expect("loss");
        let cg = tape
            .backward_create_graph(&loss, &child)
            .expect("α = 0 の有限な入力は受理");
        let gx = dense(&cg.grad(&x).expect("grad").expect("到達").to_tensor());
        for i in 0..3 {
            if x0[i].is_nan() {
                assert!(gx[i].is_nan(), "NaN の位置は NaN");
            } else {
                let exact = f64::from(c[i]) * 2f64.powf(-0.75);
                assert!(
                    common::req2_close(f64::from(gx[i]), exact),
                    "{i}: {} と {exact}",
                    gx[i]
                );
            }
        }
    }
}

/// 1 階・二階の全要素が入力の NaN で NaN になる列は、スケール上界の段検査（`s^{-2β}`）より前に外す
/// （Codex P2・Bugbot。`x = [NaN, 1e30]`・`size = 3`・`α = 0.5`・`β = -1`・`k = 1` で、以前は有限値の最大から
/// `s = 2^100` を選び `2^200` の段で拒否していた）。受理され、1 階は全要素 NaN、`child` には構造が積まれる。
/// 同じ入力で `size = 2` にすると有限の二階の経路が残るので外さない（次のテスト）。
#[test]
fn lrn_all_nan_window_column_is_skipped_before_stage_checks() {
    let shape = [1usize, 2, 1];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![f32::NAN, 1e30], &shape));
    let y = x.local_response_norm(3, 0.5, -1.0, 1.0).expect("forward");
    let loss = y.sum(None).expect("loss");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("守る有限の要素が無い列は段検査で拒否しない");
    assert!(!child.is_empty(), "構造が積まれる");
    let gx = dense(&cg.grad(&x).expect("grad").expect("到達").to_tensor());
    assert!(gx.iter().all(|v| v.is_nan()), "1 階は全要素 NaN: {gx:?}");
}

/// 全 forward 窓が NaN でも、二階が全要素 NaN とは限らない列は外さない（Codex P1・Bugbot。`x = [NaN, 1e30]`・
/// `size = 2`・`α = 0.5`・`β = -1`・`k = 1`）。偶数 size では反転窓がずれ、`β = −1` では `b = d^0 = 1` なので、
/// 子テープの二階に入力の NaN に依存しない経路が残る（その経路は `1e30` の 2 乗の overflow と 0 との積で
/// 演算から NaN を生む）。以前は全窓 NaN の列として外して受理していた。有限の `d_s` が無いのに有限の経路が
/// 残る形として fail-closed で拒否し、`child` は空に戻る。
#[test]
fn lrn_all_nan_window_column_with_finite_second_order_path_is_rejected() {
    let shape = [1usize, 2, 1];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![f32::NAN, 1e30], &shape));
    let y = x.local_response_norm(2, 0.5, -1.0, 1.0).expect("forward");
    let loss = y.sum(None).expect("loss");
    let err = tape
        .backward_create_graph(&loss, &child)
        .map(|_| ())
        .unwrap_err();
    match &err {
        AutodiffError::Backward(msg) => assert!(
            msg.contains("有限の二階の経路が残る"),
            "fail-closed の文言: {msg}"
        ),
        other => panic!("Backward で拒否: {other:?}"),
    }
    assert!(child.is_empty(), "拒否後の child は空");
}

/// 部分 NaN の列で、NaN の窓にしか入らない巨大値（`x_0 = 1e30`）にスケールを決めさせない（`size = 3` の
/// forward 窓は ch0〜2 が NaN を含み、ch3・ch4 は `{2,3,4}`・`{3,4}` で有限）。以前は列の全有限値の最大から
/// `s = 2^100` を選び、有限のチャネルの `d_s` が f32 の非正規数になって拒否していた。受理され、NaN の届かない
/// ch3・ch4 の 1 階と ch4 の自分への二階が、巨大値と NaN を小さい有限値へ置き換えた入力と REQ-2 で一致する。
#[test]
fn lrn_nan_column_scale_ignores_values_only_in_nan_windows() {
    fn build<'t>(x: &Var<'t>) -> Built<'t> {
        x.local_response_norm(3, 0.5, -1.0, 1.0)?.sum(None)
    }
    let shape = [1usize, 5, 1];
    let (first, rows) = second_rows(&[1e30, f32::NAN, 1.0, 1.0, 1.0], &shape, build)
        .expect("NaN の窓にしか入らない値で有限のチャネルを拒否しない");
    let (first_ref, rows_ref) =
        second_rows(&[1.0, 0.5, 1.0, 1.0, 1.0], &shape, build).expect("参照");
    // β = −1 は `d^{-β-1} = d^0 = 1`（`powf(NaN, 0) = 1`）なので、ch3 の 1 階も NaN にならない。
    assert!(first[..3].iter().all(|v| v.is_nan()), "{first:?}");
    for i in 3..5 {
        assert!(
            common::req2_close(f64::from(first[i]), f64::from(first_ref[i])),
            "{i}: {} と {}",
            first[i],
            first_ref[i]
        );
    }
    let (h, h_ref) = (rows[4][4], rows_ref[4][4]);
    assert!(h.is_finite(), "{h:e}");
    assert!(
        common::req2_close(f64::from(h), f64::from(h_ref)),
        "{h:e} と {h_ref:e}"
    );
}

// --- 5. 総括判定の拒否範囲の縮小（イシュー #3955・決定記録 §77）------------------------------

/// `(Σ tanh(g_k·v_k/‖v_k‖))²` の f64 の値。入力は `x = [v(groups·rest); g(groups)]`。
fn wn_sq_tanh_f64(x: &[f64], rest: usize) -> f64 {
    let groups = x.len() / (rest + 1);
    let mut sum = 0.0;
    for k in 0..groups {
        let v = &x[k * rest..(k + 1) * rest];
        let n = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        let g = x[groups * rest + k];
        sum += v.iter().map(|a| (g * a / n).tanh()).sum::<f64>();
    }
    sum * sum
}

/// f64 の損失の Hessian（二階中央差分。刻み 1e-4 で誤差は絶対 1e-7 程度）。
fn wn_sq_tanh_hessian_f64(x0: &[f32], rest: usize) -> Vec<Vec<f64>> {
    let n = x0.len();
    let h = 1e-4;
    let f = |d: &[(usize, f64)]| {
        let mut x: Vec<f64> = x0.iter().map(|&a| f64::from(a)).collect();
        for &(i, s) in d {
            x[i] += s;
        }
        wn_sq_tanh_f64(&x, rest)
    };
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = (f(&[(i, h), (j, h)]) - f(&[(i, h), (j, -h)]) - f(&[(i, -h), (j, h)])
                + f(&[(i, -h), (j, -h)]))
                / (4.0 * h * h);
        }
    }
    out
}

/// `(Σ tanh(weight_norm(v, g)))²`（`x = [v(2·rest); g(2)]`、`dim = 0`）。
fn sq_tanh_loss<'t>(_tape: &'t Tape, x: &Var<'t>, rest: usize) -> Built<'t> {
    let v = x.narrow(0, 0, 2 * rest)?.reshape(&[2, rest])?;
    let g = x.narrow(0, 2 * rest, 2)?.reshape(&[2, 1])?;
    let l = v.weight_norm(&g, Some(0))?.tanh().sum(None)?;
    l.mul(&l)
}

/// 正のプローブ（#3955）: PR #3950 が拒否を避けるため入力を下げた 2 つの普通の WeightNorm
/// （g≈0.5〜1.5・幅 3〜4・上流 約 2。Hessian 対角に値の数百倍の項が打ち消し合う要素がある）が
/// 受理され、二階が f64 参照と REQ-2 統一複合判定で一致する。tolerance は変えない。
#[test]
fn weight_norm_ordinary_cancelling_hessian_is_accepted_and_matches_f64() {
    let cases: [(&str, usize, Vec<f32>); 2] = [
        (
            "facade 入力（幅 3・g=[1.5, 0.9]）",
            3,
            vec![0.3, -0.8, 0.5, 1.0, 0.7, 0.4, 1.5, 0.9],
        ),
        (
            "GPU parity 入力（幅 4・g=[0.2, 0.5]）",
            4,
            vec![0.4, 0.55, 0.7, 0.85, 0.9, 0.1, 0.6, 0.3, 0.2, 0.5],
        ),
    ];
    for (ctx, rest, x0) in cases {
        let n = x0.len();
        let got = analytic_hessian(&|tape, x| sq_tanh_loss(tape, x, rest), &x0, &[n]);
        let want = wn_sq_tanh_hessian_f64(&x0, rest);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    common::req2_close(got[i][j], want[i][j]),
                    "{ctx}: H[{i}][{j}] = {:e}（f64 参照 {:e}）",
                    got[i][j],
                    want[i][j]
                );
            }
        }
    }
}
/// 幅 8 超の WeightNorm の入力（決定的な LCG。`v` は `[0.2, 1.0)`、`g = [g0, 0.8·g0]`）。
fn wn_wide_input(rest: usize, g0: f32, seed: u64) -> Vec<f32> {
    let mut st = seed;
    let mut x0: Vec<f32> = (0..2 * rest)
        .map(|_| {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (0.2 + 0.8 * ((st >> 33) as f64) / f64::from(1u32 << 31)) as f32
        })
        .collect();
    x0.push(g0);
    x0.push(g0 * 0.8);
    x0
}

/// 正のプローブ（#4082）: 幅 8 超（12・16）の普通の WeightNorm が受理され、二階が f64 参照と REQ-2 統一複合判定で
/// 一致する。変更前（総括判定の `Lin` が幅 8 以下だけだったとき）は従来の `Rf` + `coeff·M` で拒否されていた入力
/// （`v` は `[0.2, 1.0)` の乱数・g≈1〜1.5。決定記録 §80.5）。tolerance は変えない。
#[test]
fn weight_norm_wide_cancelling_hessian_is_accepted_and_matches_f64() {
    let cases: [(&str, usize, f32, u64); 2] = [
        ("幅 12・g0=1.0", 12, 1.0, 12345 + 12),
        ("幅 16・g0=1.5", 16, 1.5, 12345 + 16),
    ];
    for (ctx, rest, g0, seed) in cases {
        let x0 = wn_wide_input(rest, g0, seed);
        let n = x0.len();
        let got = analytic_hessian(&|tape, x| sq_tanh_loss(tape, x, rest), &x0, &[n]);
        let want = wn_sq_tanh_hessian_f64(&x0, rest);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    common::req2_close(got[i][j], want[i][j]),
                    "{ctx}: H[{i}][{j}] = {:e}（f64 参照 {:e}）",
                    got[i][j],
                    want[i][j]
                );
            }
        }
    }
}
/// `Rf` 拒否から `Lin` 受理に変わった入力 `(幅, g0, シード)`（#4118。決定記録 §80.7）。
/// `src/create_graph/norm_reparam.rs` の `tests::WN_FLIP_CASES` と同一に保つこと（そちらが「`Rf` 拒否・`Lin` 受理・
/// 全体受理」を assert し、本テストは受理後の二階を f64 参照と照合する。統合テストからクレート内は見えないため複製）。
const WN_FLIP_CASES: [(usize, f32, u64); 18] = [
    (24, 0.1, 12369),
    (24, 0.1, 15369),
    (24, 0.2, 13369),
    (32, 0.1, 12377),
    (32, 0.1, 13377),
    (32, 0.2, 12377),
    (48, 0.1, 12393),
    (48, 0.1, 13393),
    (48, 0.2, 12393),
    (64, 0.05, 12409),
    (64, 0.1, 12409),
    (64, 0.2, 12409),
    (96, 0.05, 12441),
    (96, 0.1, 12441),
    (96, 0.1, 13441),
    (128, 0.05, 12473),
    (128, 0.1, 12473),
    (128, 0.1, 15473),
];

/// 正のプローブ（#4118）: 幅 24〜128 で従来の `Rf` + `coeff·M` が拒否していた入力（`Lin` の適用範囲を広げて受理に
/// 変わったもの。拒否・受理の確認は `norm_reparam.rs` の `wn_sq_tanh_flip_inputs_reject_rf_accept_lin`）の二階が、
/// f64 参照と REQ-2 統一複合判定で全要素一致する（行の部分集合には絞らない。debug で約 7 秒）。tolerance は変えない。
/// 最大誤差（相対は絶対誤差 1e-5 以上の要素のみ）は決定記録 §80.7 の記録用に stderr へ出す（合否は REQ-2 判定だけ）。
#[test]
fn weight_norm_wide_flip_hessian_matches_f64_up_to_width_128() {
    let (mut max_rel, mut max_abs) = (0.0f64, 0.0f64);
    for (rest, g0, seed) in WN_FLIP_CASES {
        let ctx = format!("幅 {rest}・g0={g0}・seed={seed}");
        let x0 = wn_wide_input(rest, g0, seed);
        let n = x0.len();
        let got = analytic_hessian(&|tape, x| sq_tanh_loss(tape, x, rest), &x0, &[n]);
        let want = wn_sq_tanh_hessian_f64(&x0, rest);
        let (mut c_rel, mut c_abs) = (0.0f64, 0.0f64);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    common::req2_close(got[i][j], want[i][j]),
                    "{ctx}: H[{i}][{j}] = {:e}（f64 参照 {:e}）",
                    got[i][j],
                    want[i][j]
                );
                let d = (got[i][j] - want[i][j]).abs();
                c_abs = c_abs.max(d);
                // 絶対誤差 1e-5 未満の要素は絶対側で受理されるので、相対誤差は絶対側で救われない要素だけ集計する。
                if d >= 1e-5 {
                    c_rel = c_rel.max(d / want[i][j].abs());
                }
            }
        }
        eprintln!("WNFLIP {ctx} max_rel={c_rel:e} max_abs={c_abs:e}");
        max_rel = max_rel.max(c_rel);
        max_abs = max_abs.max(c_abs);
    }
    eprintln!("WNFLIP-ALL max_rel={max_rel:e} max_abs={max_abs:e}");
}
// --- 6. LRN の総括判定を Lin で追う（イシュー #4088・決定記録 §81.11）----------------------------

/// LRN の損失 `Σ w_i·y_i` の f64 の値（`y = x / (k + α/size·Σ_{窓} x²)^β`。窓は `[c − size/2, c + (size−1)/2]`
/// を `[0, C)` に打ち切る。`tensor-core` の `column_denominators` と同じ定義を独立に書いたもの）。
fn lrn_loss_f64(x: &[f64], w: &[f64], size: usize, alpha: f64, beta: f64, k: f64) -> f64 {
    let c = x.len();
    (0..c)
        .map(|i| {
            let lo = i.saturating_sub(size / 2);
            let hi = (i + (size - 1) / 2).min(c - 1);
            let sum: f64 = x[lo..=hi].iter().map(|v| v * v).sum();
            w[i] * x[i] / (k + alpha / size as f64 * sum).powf(beta)
        })
        .sum()
}

/// f64 の損失の Hessian（二階中央差分。刻み 1e-4 で誤差は絶対 1e-7 程度）。
fn lrn_hessian_f64(
    x0: &[f32],
    w: &[f32],
    size: usize,
    alpha: f32,
    beta: f32,
    k: f32,
) -> Vec<Vec<f64>> {
    let n = x0.len();
    let h = 1e-4;
    let wd: Vec<f64> = w.iter().map(|&v| f64::from(v)).collect();
    let f = |d: &[(usize, f64)]| {
        let mut x: Vec<f64> = x0.iter().map(|&a| f64::from(a)).collect();
        for &(i, s) in d {
            x[i] += s;
        }
        lrn_loss_f64(
            &x,
            &wd,
            size,
            f64::from(alpha),
            f64::from(beta),
            f64::from(k),
        )
    };
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = (f(&[(i, h), (j, h)]) - f(&[(i, h), (j, -h)]) - f(&[(i, -h), (j, h)])
                + f(&[(i, -h), (j, -h)]))
                / (4.0 * h * h);
        }
    }
    out
}

/// 正のプローブ（#4088）: 従来の `Rf` の総括判定が拒否していた REQ-2 内の LRN（`norm_reparam.rs` の
/// `LRN_LIN_ACCEPTED` と同じ入力。相殺の大きい二階の要素がある）が、丸めの一次伝播 `Lin` の上界で受理され、
/// 二階が f64 の参照（損失の Hessian の二階中央差分）と REQ-2 統一複合判定で一致する。tolerance は変えない。
#[test]
fn lrn_newly_accepted_by_lin_match_f64_second_order() {
    type Case = ([f32; 4], [f32; 4], usize, f32, f32, f32);
    let cases: [Case; 3] = [
        (
            [0.9, -0.6, 0.4, 1.1],
            [1.0, 1.0, 1.0, 1.0],
            3,
            1.0,
            2.0,
            1e-3,
        ),
        (
            [0.9, -0.6, 0.4, 1.1],
            [1.0, 0.5, -0.8, 0.3],
            2,
            1.0,
            0.5,
            1e-3,
        ),
        (
            [1.2, -0.4, 0.6, -0.9],
            [1.0, 0.5, -0.8, 0.3],
            5,
            1.0,
            2.0,
            1e-3,
        ),
    ];
    for (x0, w, size, alpha, beta, k) in cases {
        let ctx = format!("size{size}/alpha{alpha}/beta{beta}/k{k}");
        let shape = [1usize, 4, 1];
        let got = analytic_hessian(
            &|tape, x| {
                let y = x.local_response_norm(size, alpha, beta, k)?;
                let wc = tape.var_no_grad(&t(w.to_vec(), &shape));
                y.mul(&wc)?.sum(None)
            },
            &x0,
            &shape,
        );
        let want = lrn_hessian_f64(&x0, &w, size, alpha, beta, k);
        let mut worst = 0.0f64;
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    common::req2_close(got[i][j], want[i][j]),
                    "{ctx}: H[{i}][{j}] = {:e}（f64 参照 {:e}）",
                    got[i][j],
                    want[i][j]
                );
                let d = (got[i][j] - want[i][j]).abs();
                worst = worst.max((d / 1e-5).min(d / (1e-3 * want[i][j].abs())));
            }
        }
        eprintln!("LRNLIN {ctx} req2_ratio_max={worst:e}");
    }
}
