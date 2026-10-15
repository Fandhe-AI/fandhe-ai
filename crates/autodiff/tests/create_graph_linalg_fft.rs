//! 線形代数 Op（`Inv`・`Solve`・`Det`・`SlogdetLogAbsDet`）と FFT 4 件（`Rfft`・`Irfft`・`Fft`・
//! `Ifft`）の `create_graph`（`Tape::backward_create_graph`）対応（イシュー #3086・親 #3076・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §21.2・§30）の受け入れ条件を直接検証する統合テスト。
//!
//! - Hessian を `Tape::backward` の 5 点中央差分（`create_graph` を経由しない独立経路）と REQ-2 統一
//!   複合判定（`common::req2_close`）で突合する（FFT は norm 3 種 × 長さ縮小／同長／拡大 × rank 1／2・
//!   dim 先頭／末尾寄り、線形代数は well-conditioned 行列。`Solve` は `A`・`B` を 1 本のベクトルに
//!   まとめて混合項も検査する）。
//! - `cg.first_order()` が `Tape::backward` と bit 同一、`cg.grad(..)` が REQ-2 判定で一致する。
//! - 閉包性: 子テープ上の勾配を孫テープへ再生（`child.backward_create_graph`）しても二階の値が一致する。
//! - 総括的な適用判定（決定記録 §30）: 拒否域の入力は `Err(Backward)` かつ `child.is_empty()`、
//!   拒否のあと同じ `child` で再試行できる。ほどほどの大きさの入力は受理され全段が有限で一致する。
//! - fail-closed に固定した 15 件（`Stft`・`Istft`・`Cholesky`・`Pinv`・`Lstsq`・`MatrixNorm`・
//!   `QrQ`／`QrR`・`SvdU`／`SvdS`／`SvdVh`・`EighValues`／`EighVectors`・`SlogdetSign`・`MatrixRank`）が
//!   型付きエラーで拒否されること。`requires_grad == false` の部分木にある場合は拒否されない。

#![allow(clippy::needless_range_loop)]

mod common;

use fandhe_ai_autodiff::{AutodiffError, IstftOptions, StftOptions, Tape, Var};
use fandhe_ai_tensor_core::{FftNorm, MatrixNormOrd, Tensor};

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
/// 二次形式の損失（勾配が入力に対し厳密に線形）は大きな刻みでも中央差分が厳密なため、f32 の勾配
/// 丸め雑音を刻みで割って抑えるのにこの値を使う。判定式・tolerance は変えない。
const FD_STEP_QUAD: f64 = 0.5;
const NORMS: [FftNorm; 3] = [FftNorm::Backward, FftNorm::Ortho, FftNorm::Forward];

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

fn finite_diff_hessian<F>(build: &F, x0: &[f32], shape: &[usize], h: f64) -> Vec<Vec<f64>>
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
        let col = five_point_column(&grad_at, x0, i, h);
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

fn assert_hessian_close(analytic: &[Vec<f64>], numeric: &[Vec<f64>], ctx: &str) {
    assert_eq!(analytic.len(), numeric.len());
    for (j, (ra, rn)) in analytic.iter().zip(numeric.iter()).enumerate() {
        for (i, (&a, &n)) in ra.iter().zip(rn.iter()).enumerate() {
            assert!(
                common::req2_close(a, n),
                "{ctx}: hessian[{j}][{i}] analytic={a} numeric={n}"
            );
        }
    }
}

/// 閉包性: 子テープ上の勾配（の重み付き和）をさらに `backward_create_graph` で孫テープへ再生し、
/// その 1 階（= 二階）が `child.backward` と REQ-2 で一致し、孫の勾配が有限であること。
fn check_grandchild<F>(ctx: &str, build: &F, x0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let grand = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward 構築");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let gx = cg.grad(&x).expect("grad").expect("到達");
    let cx = cg.child_var(&x).expect("child_var").expect("写しあり");
    let c = child.var_no_grad(&t(pattern(numel(shape), 2.0), shape));
    let scalar = gx.mul(&c).expect("mul").sum(None).expect("sum");
    let direct = dense(
        child
            .backward(&scalar)
            .expect("child.backward")
            .get(&cx)
            .expect("get")
            .expect("到達"),
    );
    let cg2 = child
        .backward_create_graph(&scalar, &grand)
        .expect("孫テープへの再生（閉包性）");
    let first = dense(cg2.first_order().get(&cx).expect("get").expect("到達"));
    let second = dense(&cg2.grad(&cx).expect("grad").expect("到達").to_tensor());
    for i in 0..direct.len() {
        assert!(
            first[i].is_finite() && second[i].is_finite(),
            "{ctx}: 非有限"
        );
        assert!(
            common::req2_close(f64::from(first[i]), f64::from(direct[i])),
            "{ctx}: 孫 first_order[{i}]={} vs child.backward={}",
            first[i],
            direct[i]
        );
        assert!(
            common::req2_close(f64::from(second[i]), f64::from(direct[i])),
            "{ctx}: 孫 grad[{i}]={} vs child.backward={}",
            second[i],
            direct[i]
        );
    }
}

/// Hessian（有限差分）・1 階（bit 同一・REQ-2）・孫テープ再生を一括で検証する。
fn check_all<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    check_all_h(ctx, build, x0, shape, FD_STEP);
}

fn check_all_h<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize], h: f64)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let numeric = finite_diff_hessian(&build, x0, shape, h);
    let analytic = analytic_hessian(&build, x0, shape);
    assert_hessian_close(&analytic, &numeric, ctx);

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
    }
    check_grandchild(ctx, &build, x0, shape);
}

/// 非線形な損失 `sum(tanh(y) ⊙ c)`（`c` は非一様な固定定数）。
fn tanh_sum<'t>(y: &Var<'t>, tape: &'t Tape) -> Built<'t> {
    let shape = y.to_tensor().shape().to_vec();
    let c = tape.var_no_grad(&t(pattern(numel(&shape), 1.0), &shape));
    y.tanh().mul(&c)?.sum(None)
}

/// 二次形式の損失 `sum(y² ⊙ c)`。
fn quad_sum<'t>(y: &Var<'t>, tape: &'t Tape) -> Built<'t> {
    let shape = y.to_tensor().shape().to_vec();
    let c = tape.var_no_grad(&t(pattern(numel(&shape), 4.0), &shape));
    y.mul(y)?.mul(&c)?.sum(None)
}

fn x_data(shape: &[usize]) -> Vec<f32> {
    pattern(numel(shape), 7.0)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v * 0.8 + (i % 3) as f32 * 0.3 - 0.4)
        .collect()
}

// --- 1. FFT 4 件の Hessian 突合 -------------------------------------------------------------

/// `(入力 shape, dim, n)` の掃引。`n` は変換軸の長さ `L` に対し縮小・同長・拡大（偶数・奇数）。
fn fft_cases(l_axis_shape: &[usize], dim: usize) -> Vec<(Vec<usize>, usize, usize)> {
    let l = l_axis_shape[dim];
    [l - 2, l, l + 3, l + 2]
        .into_iter()
        .map(|n| (l_axis_shape.to_vec(), dim, n))
        .collect()
}

#[test]
fn fft_ifft_hessian_matches_finite_difference() {
    // 複素入力: 末尾軸 2 が (re, im)。rank 2（dim 0）と rank 3（dim 1）。
    for (shape, dim) in [(vec![5usize, 2], 0usize), (vec![2, 5, 2], 1)] {
        for (s, d, n) in fft_cases(&shape, dim) {
            for norm in NORMS {
                for inverse in [false, true] {
                    check_all_h(
                        &format!("c2c inv={inverse} {s:?} dim={d} n={n} {norm:?}"),
                        move |tape, x| {
                            let y = if inverse {
                                x.ifft(Some(n), Some(d), norm)?
                            } else {
                                x.fft(Some(n), Some(d), norm)?
                            };
                            quad_sum(&y, tape)
                        },
                        &x_data(&s),
                        &s,
                        FD_STEP_QUAD,
                    );
                }
            }
        }
    }
}

#[test]
fn fft_nonlinear_composition_hessian_matches_finite_difference() {
    let shape = [4usize, 2];
    for norm in NORMS {
        check_all(
            &format!("fft tanh {norm:?}"),
            move |tape, x| tanh_sum(&x.fft(None, None, norm)?, tape),
            &x_data(&shape),
            &shape,
        );
        check_all(
            &format!("ifft tanh {norm:?}"),
            move |tape, x| tanh_sum(&x.ifft(Some(6), None, norm)?, tape),
            &x_data(&shape),
            &shape,
        );
    }
}

#[test]
fn rfft_hessian_matches_finite_difference() {
    // 実入力: rank 1（dim 0）と rank 2（dim 1／dim 0）。
    for (shape, dim) in [(vec![6usize], 0usize), (vec![2, 5], 1), (vec![5, 3], 0)] {
        for (s, d, n) in fft_cases(&shape, dim) {
            for norm in NORMS {
                check_all_h(
                    &format!("rfft {s:?} dim={d} n={n} {norm:?}"),
                    move |tape, x| quad_sum(&x.rfft(Some(n), Some(d), norm)?, tape),
                    &x_data(&s),
                    &s,
                    FD_STEP_QUAD,
                );
            }
        }
    }
}

#[test]
fn irfft_hessian_matches_finite_difference() {
    // 複素入力 `[.., m, .., 2]`。bin 数 m=4 に対し n/2+1 が m より小さい・等しい・大きい場合と奇数 n。
    for (shape, dim) in [
        (vec![4usize, 2], 0usize),
        (vec![2, 4, 2], 1),
        (vec![4, 3, 2], 0),
    ] {
        for n in [4usize, 6, 9, 10] {
            for norm in NORMS {
                check_all_h(
                    &format!("irfft {shape:?} dim={dim} n={n} {norm:?}"),
                    move |tape, x| {
                        let y = x.irfft(Some(n), Some(dim), norm)?;
                        quad_sum(&y, tape)
                    },
                    &x_data(&shape),
                    &shape,
                    FD_STEP_QUAD,
                );
            }
        }
    }
    let shape = [4usize, 2];
    for norm in NORMS {
        check_all(
            &format!("irfft tanh default-n {norm:?}"),
            move |tape, x| tanh_sum(&x.irfft(None, None, norm)?, tape),
            &x_data(&shape),
            &shape,
        );
    }
}

// --- 2. 線形代数 4 件の Hessian 突合 --------------------------------------------------------

/// well-conditioned な `n x n`（`I + 0.1 * pattern`）。
fn well_conditioned(n: usize) -> Vec<f32> {
    let p = pattern(n * n, 2.0);
    (0..n * n)
        .map(|i| {
            let diag = if i / n == i % n { 1.0 } else { 0.0 };
            diag + 0.1 * p[i]
        })
        .collect()
}

#[test]
fn inv_hessian_matches_finite_difference() {
    for n in [2usize, 3] {
        check_all(
            &format!("inv n={n}"),
            |tape, x| quad_sum(&x.inv()?, tape),
            &well_conditioned(n),
            &[n, n],
        );
        check_all(
            &format!("inv tanh n={n}"),
            |tape, x| tanh_sum(&x.inv()?, tape),
            &well_conditioned(n),
            &[n, n],
        );
    }
}

#[test]
fn solve_hessian_matches_finite_difference_including_mixed_terms() {
    // `A`（n x n）と `B`（n x k）を 1 本のベクトルにまとめ、混合項 d²/dA dB も検査する。
    for (n, k) in [(2usize, 1usize), (3, 2)] {
        let mut x0 = well_conditioned(n);
        x0.extend(pattern(n * k, 5.0));
        check_all(
            &format!("solve A+B n={n} k={k}"),
            move |tape, x| {
                let a = x.narrow(0, 0, n * n)?.reshape(&[n, n])?;
                let b = x.narrow(0, n * n, n * k)?.reshape(&[n, k])?;
                quad_sum(&a.solve(&b)?, tape)
            },
            &x0,
            &[n * n + n * k],
        );
    }
    // B のみ追跡（A は定数）と A のみ追跡（B は定数）。
    let (n, k) = (3usize, 2usize);
    let a_const = t(well_conditioned(n), &[n, n]);
    let b_const = t(pattern(n * k, 5.0), &[n, k]);
    check_all(
        "solve B only",
        |tape, b| quad_sum(&tape.var_no_grad(&a_const).solve(b)?, tape),
        &pattern(n * k, 5.0),
        &[n, k],
    );
    check_all(
        "solve A only",
        |tape, a| quad_sum(&a.solve(&tape.var_no_grad(&b_const))?, tape),
        &well_conditioned(n),
        &[n, n],
    );
}

#[test]
fn slogdet_logabsdet_hessian_matches_finite_difference() {
    for n in [2usize, 3] {
        check_all(
            &format!("logabsdet n={n}"),
            |_tape, x| Ok(x.slogdet()?.logabsdet),
            &well_conditioned(n),
            &[n, n],
        );
        check_all(
            &format!("logabsdet tanh n={n}"),
            |_tape, x| {
                let l = x.slogdet()?.logabsdet;
                Ok(l.mul(&l)?.tanh())
            },
            &well_conditioned(n),
            &[n, n],
        );
    }
}

#[test]
fn det_hessian_matches_finite_difference() {
    for n in [2usize, 3] {
        check_all(
            &format!("det n={n}"),
            |_tape, x| x.det(),
            &well_conditioned(n),
            &[n, n],
        );
        check_all(
            &format!("det squared n={n}"),
            |_tape, x| {
                let d = x.det()?;
                d.mul(&d)
            },
            &well_conditioned(n),
            &[n, n],
        );
    }
}

// --- 3. 総括的な適用判定（拒否域・正のプローブ） --------------------------------------------

fn is_backward_err<T>(r: &Result<T, AutodiffError>) -> bool {
    matches!(r, Err(AutodiffError::Backward(_)))
}

/// 拒否域の入力: `Err(Backward)` かつ `child.is_empty()`。さらに同じ `child` で別の損失を再試行できる。
fn assert_rejected_and_child_reusable<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward 構築");
    let r = tape.backward_create_graph(&loss, &child);
    assert!(
        is_backward_err(&r),
        "{ctx}: Err(Backward) のはず: {:?}",
        r.as_ref().err()
    );
    assert!(child.is_empty(), "{ctx}: 拒否後は child が空");
    // 同じ child で受理される損失（x の線形和）を再試行できる。
    let ok_loss = x.mul(&x).expect("mul").sum(None).expect("sum");
    assert!(
        tape.backward_create_graph(&ok_loss, &child).is_ok(),
        "{ctx}: 拒否後も同じ child を再利用できる"
    );
}

#[test]
fn inv_subnormal_input_is_rejected() {
    assert_rejected_and_child_reusable(
        "inv 1e-39",
        |_tape, x| x.inv()?.sum(None),
        &[1e-39],
        &[1, 1],
    );
}

#[test]
fn solve_overflowing_state_is_rejected() {
    assert_rejected_and_child_reusable(
        "solve A=1e-10 B=1e30",
        |tape, a| {
            let b = tape.var_no_grad(&t(vec![1e30], &[1, 1]));
            a.solve(&b)?.sum(None)
        },
        &[1e-10],
        &[1, 1],
    );
}

#[test]
fn det_large_input_is_rejected() {
    // diag(1e20, 1e20): det = 1e40 は f32 を超える（V3 が必要な域。本 issue では拒否。§30）。
    assert_rejected_and_child_reusable(
        "det diag(1e20)",
        |_tape, x| x.det(),
        &[1e20, 0.0, 0.0, 1e20],
        &[2, 2],
    );
}

/// 正のプローブ: ほどほどの大きさ・条件数の入力は受理され、全段が有限で 1 階と一致する。
#[test]
fn moderate_scale_inputs_are_accepted_and_finite() {
    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("diag(1e3,1e-3)", vec![1e3, 0.0, 0.0, 1e-3]),
        ("diag(30,0.05)", vec![30.0, 0.0, 0.0, 0.05]),
        ("dense", vec![2.0, 0.5, -0.3, 1.5]),
    ];
    for (name, a0) in cases {
        for kind in 0..4 {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let a = tape.var(&t(a0.clone(), &[2, 2]));
            let b = tape.var_no_grad(&t(vec![1.0, -2.0], &[2, 1]));
            let loss = match kind {
                0 => a.inv().and_then(|y| y.sum(None)),
                1 => a.solve(&b).and_then(|y| y.sum(None)),
                2 => a.det(),
                _ => a.slogdet().map(|s| s.logabsdet),
            }
            .expect("forward 構築");
            let cg = tape
                .backward_create_graph(&loss, &child)
                .unwrap_or_else(|e| panic!("{name}/kind{kind}: 受理されるはず: {e:?}"));
            let plain = dense(
                tape.backward(&loss)
                    .expect("backward")
                    .get(&a)
                    .expect("get")
                    .expect("到達"),
            );
            let g = dense(&cg.grad(&a).expect("grad").expect("到達").to_tensor());
            for i in 0..plain.len() {
                assert!(g[i].is_finite(), "{name}/kind{kind}: cg.grad が非有限");
                assert!(
                    common::req2_close(f64::from(g[i]), f64::from(plain[i])),
                    "{name}/kind{kind}: cg.grad[{i}]={} vs backward={}",
                    g[i],
                    plain[i]
                );
            }
        }
    }
}

// --- 4. fail-closed に固定した 15 件 --------------------------------------------------------

type Case = (
    &'static str,
    Vec<usize>,
    Box<dyn for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>>,
);

fn spd3() -> Vec<f32> {
    vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0]
}

fn fail_closed_cases() -> Vec<(Case, Vec<f32>)> {
    let sym3 = spd3();
    vec![
        (
            (
                "stft",
                vec![16],
                Box::new(|_t, x| {
                    x.stft(8, None, &StftOptions::default().with_center(false))?
                        .sum(None)
                }),
            ),
            x_data(&[16]),
        ),
        (
            (
                "istft",
                vec![5, 4, 2],
                Box::new(|_t, x| {
                    x.istft(8, None, &IstftOptions::default().with_center(false))?
                        .sum(None)
                }),
            ),
            x_data(&[5, 4, 2]),
        ),
        (
            (
                "cholesky",
                vec![3, 3],
                Box::new(|_t, x| x.cholesky()?.sum(None)),
            ),
            sym3.clone(),
        ),
        (
            (
                "pinv",
                vec![3, 2],
                Box::new(|_t, x| x.pinv(None)?.sum(None)),
            ),
            x_data(&[3, 2]),
        ),
        (
            (
                "lstsq",
                vec![3, 2],
                Box::new(|tape, x| {
                    let b = tape.var_no_grad(&t(vec![1.0, 2.0, 3.0], &[3, 1]));
                    x.lstsq(&b, None)?.sum(None)
                }),
            ),
            x_data(&[3, 2]),
        ),
        (
            (
                "matrix_norm",
                vec![3, 3],
                Box::new(|_t, x| x.matrix_norm(MatrixNormOrd::Fro)),
            ),
            sym3.clone(),
        ),
        (
            ("qr.q", vec![3, 2], Box::new(|_t, x| x.qr()?.q.sum(None))),
            x_data(&[3, 2]),
        ),
        (
            ("qr.r", vec![3, 2], Box::new(|_t, x| x.qr()?.r.sum(None))),
            x_data(&[3, 2]),
        ),
        (
            ("svd.u", vec![3, 2], Box::new(|_t, x| x.svd()?.u.sum(None))),
            x_data(&[3, 2]),
        ),
        (
            ("svd.s", vec![3, 2], Box::new(|_t, x| x.svd()?.s.sum(None))),
            x_data(&[3, 2]),
        ),
        (
            (
                "svd.vh",
                vec![3, 2],
                Box::new(|_t, x| x.svd()?.vh.sum(None)),
            ),
            x_data(&[3, 2]),
        ),
        (
            (
                "eigh.values",
                vec![3, 3],
                Box::new(|_t, x| x.eigh()?.eigenvalues.sum(None)),
            ),
            sym3.clone(),
        ),
        (
            (
                "eigh.vectors",
                vec![3, 3],
                Box::new(|_t, x| x.eigh()?.eigenvectors.sum(None)),
            ),
            sym3.clone(),
        ),
        (
            (
                "slogdet.sign",
                vec![3, 3],
                Box::new(|_t, x| Ok(x.slogdet()?.sign)),
            ),
            sym3.clone(),
        ),
        (
            (
                "matrix_rank",
                vec![3, 3],
                Box::new(|_t, x| x.matrix_rank(None)),
            ),
            sym3,
        ),
    ]
}

#[test]
fn fail_closed_ops_are_rejected_with_typed_error_and_leave_child_empty() {
    let cases = fail_closed_cases();
    assert_eq!(cases.len(), 15, "fail-closed 固定は 15 件");
    for ((name, shape, build), x0) in cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0, &shape));
        let loss = build(&tape, &x).unwrap_or_else(|e| panic!("{name}: forward 構築: {e:?}"));
        let r = tape.backward_create_graph(&loss, &child);
        assert!(
            is_backward_err(&r),
            "{name}: Err(Backward) のはず（Ok または別エラー）"
        );
        assert!(child.is_empty(), "{name}: 拒否後は child が空");
    }
}

#[test]
fn fail_closed_op_in_untracked_subtree_is_treated_as_constant() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let l = tape
        .var_no_grad(&t(spd3(), &[3, 3]))
        .cholesky()
        .expect("cholesky");
    let x = tape.var(&t(x_data(&[3, 3]), &[3, 3]));
    let loss = x.mul(&l).expect("mul").sum(None).expect("sum");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("requires_grad=false の部分木は定数として受理される");
    assert!(cg.grad(&x).expect("grad").is_some());
}

#[test]
fn rfft_irfft_roundtrip_second_order_matches_direct_composition() {
    // irfft(rfft(x)) の損失（恒等写像に近い）でも二階が有限差分と一致する。
    let shape = [8usize];
    for norm in NORMS {
        check_all(
            &format!("roundtrip {norm:?}"),
            move |tape, x| {
                let y = x.rfft(None, None, norm)?.irfft(Some(8), None, norm)?;
                tanh_sum(&y, tape)
            },
            &x_data(&shape),
            &shape,
        );
    }
}

/// 桁落ちする `Inv`（A=[[1,2],[3,6.0001]]・定数 G=A^T）は f32 構造の丸め誤差が REQ-2 を超えるため拒否する。
#[test]
fn inv_cancellation_input_is_rejected() {
    assert_rejected_and_child_reusable(
        "inv cancellation",
        |tape, x| {
            let g = tape.var_no_grad(&t(vec![1.0, 3.0, 2.0, 6.0001], &[2, 2]));
            x.inv()?.mul(&g)?.sum(None)
        },
        &[1.0, 2.0, 3.0, 6.0001],
        &[2, 2],
    );
}

/// DC 虚部のみ inf の上流勾配: 通常 backward と create_graph の `cg.grad` で同じく NaN が伝播する。
#[test]
fn rfft_dc_imag_nonfinite_propagates_like_first_order() {
    let n = 4usize;
    let mut w = vec![0.0f32; (n / 2 + 1) * 2];
    w[1] = f32::INFINITY;
    let run = |cg: bool| -> Vec<f32> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![0.5, -1.0, 2.0, 0.25], &[n]));
        let y = x.rfft(Some(n), Some(0), FftNorm::Backward).expect("rfft");
        let wv = tape.var_no_grad(&t(w.clone(), &[n / 2 + 1, 2]));
        let loss = y.mul(&wv).expect("mul").sum(None).expect("sum");
        if cg {
            let child = Tape::new_with_ops(common::naive_ops());
            let grads = tape.backward_create_graph(&loss, &child).expect("cg");
            dense(&grads.grad(&x).expect("grad").expect("到達").to_tensor())
        } else {
            let grads = tape.backward(&loss).expect("backward");
            dense(grads.get(&x).expect("get").expect("到達"))
        }
    };
    let a = run(false);
    let b = run(true);
    assert!(a.iter().all(|v| v.is_nan()), "通常 backward は NaN: {a:?}");
    assert!(b.iter().all(|v| v.is_nan()), "create_graph も NaN: {b:?}");
}

/// 通常 backward と create_graph の勾配を、NaN 位置・inf 符号・有限値（REQ-2 判定）で突合する。
fn assert_same_with_nonfinite(a: &[f32], b: &[f32], ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: 長さ");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x.is_nan() || y.is_nan() {
            assert!(
                x.is_nan() && y.is_nan(),
                "{ctx}[{i}]: NaN 不一致 {a:?} vs {b:?}"
            );
        } else if x.is_infinite() || y.is_infinite() {
            assert_eq!(x, y, "{ctx}[{i}]: inf 不一致 {a:?} vs {b:?}");
        } else {
            assert!(
                common::req2_close(f64::from(*x), f64::from(*y)),
                "{ctx}[{i}]: {x} vs {y}"
            );
        }
    }
}

/// rfft（`x: [lanes, len]`・`dim = 1`）の上流重み `w: [lanes, n/2+1, 2]` で両経路の勾配を返す。
fn rfft_grads(lanes: usize, len: usize, n: usize, w: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let run = |cg: bool| -> Vec<f32> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(pattern(lanes * len, 0.3), &[lanes, len]));
        let y = x.rfft(Some(n), Some(1), FftNorm::Backward).expect("rfft");
        let wv = tape.var_no_grad(&t(w.to_vec(), &[lanes, n / 2 + 1, 2]));
        let loss = y.mul(&wv).expect("mul").sum(None).expect("sum");
        if cg {
            let child = Tape::new_with_ops(common::naive_ops());
            let grads = tape.backward_create_graph(&loss, &child).expect("cg");
            dense(&grads.grad(&x).expect("grad").expect("到達").to_tensor())
        } else {
            let grads = tape.backward(&loss).expect("backward");
            dense(grads.get(&x).expect("get").expect("到達"))
        }
    };
    (run(false), run(true))
}

/// 1 レーンだけ DC 虚部が inf: 汚染は当該レーンの有効位置に限り、他レーンは有限（n < L・n = L・n > L）。
#[test]
fn rfft_nonfinite_poison_stays_in_lane() {
    for (len, n) in [(4usize, 4usize), (6, 4), (3, 4)] {
        let (lanes, bins) = (3usize, n / 2 + 1);
        let mut w = vec![0.0f32; lanes * bins * 2];
        w[bins * 2 + 1] = f32::INFINITY;
        let (a, b) = rfft_grads(lanes, len, n, &w);
        assert_same_with_nonfinite(&a, &b, "inf 1 レーン");
        let valid = len.min(n);
        for lane in 0..lanes {
            for j in 0..len {
                let expect_nan = lane == 1 && j < valid;
                assert_eq!(
                    b[lane * len + j].is_nan(),
                    expect_nan,
                    "lane={lane} j={j} {b:?}"
                );
            }
        }
    }
}

/// 有限の 3e38 が複数レーンの DC・Nyquist 虚部にある: 和の overflow による NaN を作らない。
#[test]
fn rfft_large_finite_imag_in_many_lanes_is_not_poisoned() {
    let (lanes, len, n) = (3usize, 4usize, 4usize);
    let bins = n / 2 + 1;
    let mut w = vec![0.0f32; lanes * bins * 2];
    for lane in 0..lanes {
        w[(lane * bins) * 2 + 1] = 3.0e38;
        w[(lane * bins + n / 2) * 2 + 1] = 3.0e38;
    }
    let (a, b) = rfft_grads(lanes, len, n, &w);
    assert_same_with_nonfinite(&a, &b, "3e38 複数レーン");
}

/// irfft に上流 `[inf, 0, 0, 0]`: DC・Nyquist の虚部勾配は両経路とも +0.0（NaN にしない）。
#[test]
fn irfft_real_bin_imag_grad_is_zero_with_inf_upstream() {
    let n = 4usize;
    let run = |cg: bool| -> Vec<f32> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(pattern((n / 2 + 1) * 2, 0.7), &[n / 2 + 1, 2]));
        let y = x.irfft(Some(n), Some(0), FftNorm::Backward).expect("irfft");
        let wv = tape.var_no_grad(&t(vec![f32::INFINITY, 0.0, 0.0, 0.0], &[n]));
        let loss = y.mul(&wv).expect("mul").sum(None).expect("sum");
        if cg {
            let child = Tape::new_with_ops(common::naive_ops());
            let grads = tape.backward_create_graph(&loss, &child).expect("cg");
            dense(&grads.grad(&x).expect("grad").expect("到達").to_tensor())
        } else {
            let grads = tape.backward(&loss).expect("backward");
            dense(grads.get(&x).expect("get").expect("到達"))
        }
    };
    let (a, b) = (run(false), run(true));
    assert_same_with_nonfinite(&a, &b, "irfft inf 上流");
    for bin in [0usize, n / 2] {
        assert_eq!(a[bin * 2 + 1], 0.0, "通常 backward の虚部: {a:?}");
        assert_eq!(b[bin * 2 + 1], 0.0, "create_graph の虚部: {b:?}");
    }
}
