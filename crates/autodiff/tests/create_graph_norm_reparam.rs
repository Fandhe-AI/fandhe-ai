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
