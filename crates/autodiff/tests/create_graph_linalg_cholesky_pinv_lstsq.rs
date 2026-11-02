//! `Cholesky`・`Pinv`・`Lstsq` の `create_graph`（`Tape::backward_create_graph`）対応（イシュー #3805・
//! 親 #3734・決定記録 `docs/autodiff-higher-order-grad-decision.md` §43・§65）の受け入れ条件を公開 API
//! 経由で直接検証する統合テスト。
//!
//! - Hessian を `Tape::backward` の 5 点中央差分（`create_graph` を経由しない独立経路）と REQ-2 統一複合
//!   判定（`common::req2_close`）で突合する。`Cholesky` は `A = (X+Xᵀ)/2 + cI` で `X` をパラメータにする
//!   （VJP は `dA` を対称化し forward は下三角しか読まないため）。`Pinv` は縦長・横長・正方、`Lstsq` は
//!   `A`・`B` を 1 本のベクトルにまとめて混合項も検査する。いずれも打ち切りなしの入力。
//! - `cg.first_order()` は `Tape::backward` と bit 同一、`cg.grad(..)` は REQ-2 判定で一致し、V3 のため
//!   CPU では線形な上流なら bit 一致する。閉包性（孫テープ）と `jvp_create_graph` も確かめる。
//! - 打ち切りあり・rank 落ち・閾値付近の `Pinv`・`Lstsq`、悪条件の `Cholesky`、巨大な上流は
//!   `Err(Backward)` かつ `child.is_empty()` で、同じ `child` を再利用できる。縦長 `[100000,1]` は
//!   `m×m` を確保せず受理される。隣接する未対応 Op（`Op::Custom`）は fail-closed のまま。
//!
//! GPU 実機 parity は `crates/facade/tests/create_graph_gpu_parity.rs`（`#[ignore]`・未実測。Phase 10 の
//! #3857・#3859）で扱う。
#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;

use fandhe_ai_autodiff::functional_ops::jvp_create_graph;
use fandhe_ai_autodiff::{AutodiffError, CustomFunction, Tape, Var};
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
/// 擬似逆行列の Hessian は f32 の勾配丸め雑音を刻みで割った分が絶対許容に近づくため、やや大きい刻みを使う
/// （5 点中央差分の打ち切り誤差は十分小さい）。判定式・tolerance は変えない。
const PINV_FD_STEP: f64 = 5e-2;

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

// --- 1. 対象行列の組み立て -------------------------------------------------------------------

/// `A = (X + Xᵀ)/2 + c I`（対称正定値。`X` をパラメータにする）。
fn spd_from<'t>(tape: &'t Tape, x: &Var<'t>, c: f32) -> Built<'t> {
    let n = x.to_tensor().shape()[0];
    let half = tape.var_no_grad(&t(vec![0.5; n * n], &[n, n]));
    let mut eye = vec![0.0f32; n * n];
    for i in 0..n {
        eye[i * n + i] = c;
    }
    let eye = tape.var_no_grad(&t(eye, &[n, n]));
    x.add(&x.transpose(0, 1)?)?.mul(&half)?.add(&eye)
}

/// 条件の良い一般行列: `x_data` に対角寄りのオフセットを足す。
fn well_conditioned(shape: &[usize]) -> Vec<f32> {
    let cols = shape[1];
    x_data(shape)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v * 0.5 + if i / cols == i % cols { 2.0 } else { 0.0 })
        .collect()
}

fn build_y<'t>(name: &str, tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    match name {
        "cholesky" => spd_from(tape, x, 4.0)?.cholesky(),
        "lstsq" => {
            let (a, b) = split_ab(x, 3, 2, 1)?;
            a.lstsq(&b, None)
        }
        _ => x.pinv(None),
    }
}

// --- 2. Cholesky ---------------------------------------------------------------------------

#[test]
fn cholesky_hessian_matches_finite_difference() {
    let shape = [3usize, 3];
    check_all(
        "cholesky tanh",
        |tape, x| tanh_sum(&spd_from(tape, x, 4.0)?.cholesky()?, tape),
        &x_data(&shape),
        &shape,
    );
    check_all(
        "cholesky quad",
        |tape, x| quad_sum(&spd_from(tape, x, 4.0)?.cholesky()?, tape),
        &x_data(&shape),
        &shape,
    );
    let shape2 = [2usize, 2];
    check_all(
        "cholesky 2x2",
        |tape, x| tanh_sum(&spd_from(tape, x, 3.0)?.cholesky()?, tape),
        &x_data(&shape2),
        &shape2,
    );
}

/// 出力 `L` の上三角の位置にだけ置いた upstream の勾配は 0（`L` の上三角は forward に現れない）。
/// 入力 `X` の上三角方向の二階は 0 にならない（FD 側で検証する）ため、ここでは出力側だけを確かめる。
#[test]
fn cholesky_upper_triangle_upstream_has_zero_gradient() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x_data(&[3, 3]), &[3, 3]));
    let l = spd_from(&tape, &x, 4.0).unwrap().cholesky().unwrap();
    let mut mask = vec![0.0f32; 9];
    mask[1] = 1.0;
    mask[2] = 1.0;
    mask[5] = 1.0;
    let m = tape.var_no_grad(&t(mask, &[3, 3]));
    let loss = l.mul(&m).unwrap().sum(None).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let g = dense(&cg.grad(&x).unwrap().unwrap().to_tensor());
    assert!(g.iter().all(|&v| v == 0.0), "{g:?}");
}

// --- 3. Pinv -------------------------------------------------------------------------------

#[test]
fn pinv_hessian_matches_finite_difference() {
    for shape in [[3usize, 2], [2, 3], [3, 3], [4, 2], [2, 4]] {
        let x0 = well_conditioned(&shape);
        check_all_h(
            &format!("pinv tanh {shape:?}"),
            |tape, x| tanh_sum(&x.pinv(None)?, tape),
            &x0,
            &shape,
            PINV_FD_STEP,
        );
        check_all_h(
            &format!("pinv quad {shape:?}"),
            |tape, x| quad_sum(&x.pinv(None)?, tape),
            &x0,
            &shape,
            PINV_FD_STEP,
        );
    }
}

// --- 4. Lstsq ------------------------------------------------------------------------------

/// `x` の前半 `m*n` を `A: [m,n]`、後半 `m*k` を `B: [m,k]` とする。
fn split_ab<'t>(
    x: &Var<'t>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(Var<'t>, Var<'t>), AutodiffError> {
    let a = x.narrow(0, 0, m * n)?.reshape(&[m, n])?;
    let b = x.narrow(0, m * n, m * k)?.reshape(&[m, k])?;
    Ok((a, b))
}

fn lstsq_input(m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut v = well_conditioned(&[m, n]);
    v.extend(x_data(&[m, k]));
    v
}

#[test]
fn lstsq_mixed_second_derivatives() {
    for (m, n, k) in [(3usize, 2usize, 1usize), (4, 2, 2), (2, 3, 1), (3, 3, 2)] {
        let shape = [m * n + m * k];
        check_all(
            &format!("lstsq tanh m={m} n={n} k={k}"),
            move |tape, x| {
                let (a, b) = split_ab(x, m, n, k)?;
                tanh_sum(&a.lstsq(&b, None)?, tape)
            },
            &lstsq_input(m, n, k),
            &shape,
        );
    }
}

// --- 5. 孫テープ・jvp_create_graph ------------------------------------------------------------

#[test]
fn linalg_cpl_jvp_create_graph_matches_central_difference() {
    let cases: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
        ("cholesky", vec![3, 3], x_data(&[3, 3])),
        ("pinv", vec![3, 2], well_conditioned(&[3, 2])),
        ("lstsq", vec![9], lstsq_input(3, 2, 1)),
    ];
    for (name, shape, x0) in cases {
        let v = pattern(numel(&shape), 5.0);
        let loss_at = |x: &[f32]| -> f64 {
            let tape = Tape::new_with_ops(common::naive_ops());
            let xv = tape.var(&t(x.to_vec(), &shape));
            let y = build_y(name, &tape, &xv).unwrap();
            f64::from(dense(&quad_sum(&y, &tape).unwrap().to_tensor())[0])
        };
        let h = 0.01f32;
        let plus: Vec<f32> = x0.iter().zip(&v).map(|(a, b)| a + h * b).collect();
        let minus: Vec<f32> = x0.iter().zip(&v).map(|(a, b)| a - h * b).collect();
        let expected = (loss_at(&plus) - loss_at(&minus)) / (2.0 * f64::from(h));

        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0.clone(), &shape));
        let y = build_y(name, &tape, &x).unwrap();
        let loss = quad_sum(&y, &tape).unwrap();
        let res = jvp_create_graph(&tape, &loss, &x, &t(v.clone(), &shape), &child, &grand)
            .expect("jvp_create_graph");
        let got = f64::from(dense(&res.output().to_tensor())[0]);
        assert!(
            common::req2_close(got, expected),
            "{name}: jvp={got} vs central difference={expected}"
        );
    }
}

// --- 6. 1 階の bit 同一（V3） --------------------------------------------------------------

/// V3 の値は 1 階と同じホスト関数のため、`cg.grad` は CPU で 1 階と bit 一致する（線形な上流）。
#[test]
fn linalg_cpl_cg_grad_bit_identical_for_linear_upstream() {
    let cases: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
        ("cholesky", vec![3, 3], x_data(&[3, 3])),
        ("pinv", vec![3, 2], well_conditioned(&[3, 2])),
        ("pinv", vec![2, 3], well_conditioned(&[2, 3])),
        ("pinv", vec![3, 3], well_conditioned(&[3, 3])),
        ("lstsq", vec![9], lstsq_input(3, 2, 1)),
    ];
    for (name, shape, x0) in cases {
        let run = |cg: bool| -> Vec<f32> {
            let tape = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), &shape));
            let y = build_y(name, &tape, &x).unwrap();
            let c = tape.var_no_grad(&t(
                pattern(numel(y.to_tensor().shape()), 3.0),
                y.to_tensor().shape(),
            ));
            let loss = y.mul(&c).unwrap().sum(None).unwrap();
            if cg {
                let child = Tape::new_with_ops(common::naive_ops());
                let grads = tape.backward_create_graph(&loss, &child).expect("cg");
                dense(&grads.grad(&x).unwrap().unwrap().to_tensor())
            } else {
                dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap())
            }
        };
        let (a, b) = (run(false), run(true));
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{name} {shape:?}: cg.grad は 1 階と bit 一致"
        );
    }
}

// --- 7. 拒否域 ---------------------------------------------------------------------------

#[test]
fn pinv_lstsq_truncation_rejected_child_empty_and_retry() {
    // 打ち切りあり: A = diag(2,1)・rcond = 0.75（σ2 = 1 が捨てられる）。
    assert_rejected_and_child_reusable(
        "pinv truncated",
        |_t, x| x.pinv(Some(0.75))?.sum(None),
        &[2.0, 0.0, 0.0, 1.0],
        &[2, 2],
    );
    // rank 落ち（rank 1 の 3x2）。
    assert_rejected_and_child_reusable(
        "pinv rank-deficient",
        |_t, x| x.pinv(None)?.sum(None),
        &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0],
        &[3, 2],
    );
    // rcond の閾値付近: σ = [2, 1.5000001]・rcond = 0.75（tol = 1.5 の f32 隣接値の範囲内）。
    assert_rejected_and_child_reusable(
        "pinv near threshold",
        |_t, x| x.pinv(Some(0.75))?.sum(None),
        &[2.0, 0.0, 0.0, 1.500_000_1],
        &[2, 2],
    );
    // lstsq: 同じ 3 種。
    type TruncCase = (&'static str, Vec<f32>, [usize; 2], Option<f32>);
    let cases: [TruncCase; 3] = [
        ("truncated", vec![2.0, 0.0, 0.0, 1.0], [2, 2], Some(0.75)),
        (
            "rank-deficient",
            vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0],
            [3, 2],
            None,
        ),
        (
            "near threshold",
            vec![2.0, 0.0, 0.0, 1.500_000_1],
            [2, 2],
            Some(0.75),
        ),
    ];
    for (ctx, a, shape, rcond) in cases {
        let (m, n) = (shape[0], shape[1]);
        let mut x0 = a;
        x0.extend((0..m).map(|i| 1.0 + i as f32));
        assert_rejected_and_child_reusable(
            &format!("lstsq {ctx}"),
            move |_t, x| {
                let (a, b) = split_ab(x, m, n, 1)?;
                a.lstsq(&b, rcond)?.sum(None)
            },
            &x0,
            &[m * n + m],
        );
    }
}

#[test]
fn lstsq_cancelling_gram_product_rejected_child_empty_and_retry() {
    // A = [[1]]・B = [[1e8, 1, -1e8]]: G = gX Bᵀ は f64 で 1 だが f32 積和では 0（桁落ち）。
    assert_rejected_and_child_reusable(
        "lstsq cancelling G",
        |_t, x| {
            let (a, b) = split_ab(x, 1, 1, 3)?;
            a.lstsq(&b, None)?.sum(None)
        },
        &[1.0, 1e8, 1.0, -1e8],
        &[4],
    );
}

#[test]
fn lstsq_saturated_tanh_cotangent_overflow_rejected_child_empty_and_retry() {
    // A = [[1e-5]]・B = [[1e30]]: tanh 飽和で gX=G=gB=0 だが、二階の cotangent
    // （G から gX へ約 1e40）が f32 で overflow し NaN になるため構造構築前に拒否する。
    assert_rejected_and_child_reusable(
        "lstsq saturated tanh",
        |_t, x| {
            let (a, b) = split_ab(x, 1, 1, 1)?;
            a.lstsq(&b, None)?.tanh().sum(None)
        },
        &[1e-5, 1e30],
        &[2],
    );
}

#[test]
fn linalg_cpl_unsafe_inputs_rejected_child_empty_and_retry() {
    // Cholesky の悪条件（対角 1e-20 付近）。
    assert_rejected_and_child_reusable(
        "cholesky ill-conditioned",
        |tape, x| spd_from(tape, x, 0.0)?.cholesky()?.sum(None),
        &[1e-20, 0.0, 0.0, 1.0],
        &[2, 2],
    );
    // Cholesky・Pinv・Lstsq の g = ±3e38。
    assert_rejected_and_child_reusable(
        "cholesky g=3e38",
        |tape, x| {
            let c = tape.var_no_grad(&t(vec![3e38; 4], &[2, 2]));
            spd_from(tape, x, 3.0)?.cholesky()?.mul(&c)?.sum(None)
        },
        &[1.0, 0.1, 0.1, 1.0],
        &[2, 2],
    );
    assert_rejected_and_child_reusable(
        "pinv g=-3e38",
        |tape, x| {
            let c = tape.var_no_grad(&t(vec![-3e38; 6], &[2, 3]));
            x.pinv(None)?.mul(&c)?.sum(None)
        },
        &well_conditioned(&[3, 2]),
        &[3, 2],
    );
    assert_rejected_and_child_reusable(
        "lstsq g=3e38",
        |tape, x| {
            let (a, b) = split_ab(x, 3, 2, 1)?;
            let c = tape.var_no_grad(&t(vec![3e38; 2], &[2, 1]));
            a.lstsq(&b, None)?.mul(&c)?.sum(None)
        },
        &lstsq_input(3, 2, 1),
        &[9],
    );
    // Pinv の縦長で cond が大きい入力（rcond = 0 で打ち切りなしにしたうえで二階の上界が溢れる）。
    assert_rejected_and_child_reusable(
        "pinv ill-conditioned tall",
        |_t, x| x.pinv(Some(0.0))?.sum(None),
        &[1.0, 0.0, 0.0, 1e-10, 0.0, 0.0],
        &[3, 2],
    );
}

/// 縦長 `[100000,1]` の `Pinv` は `m×m`（10^10 要素）を確保せずに受理される。
#[test]
fn pinv_cgrad_does_not_materialize_m_by_m() {
    let m = 100_000usize;
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x0: Vec<f32> = (0..m).map(|i| 1.0 + (i % 7) as f32 * 0.1).collect();
    let x = tape.var(&t(x0, &[m, 1]));
    let loss = x.pinv(None).unwrap().sum(None).unwrap();
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("縦長 Pinv は m×m を作らず受理される");
    let g = cg.grad(&x).unwrap().unwrap();
    assert_eq!(g.to_tensor().shape(), &[m, 1]);
}

// --- 8. 隣接する未対応 Op は fail-closed のまま --------------------------------------------------

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

#[test]
fn linalg_cpl_neighbouring_unsupported_op_stays_fail_closed() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x_data(&[3, 3]), &[3, 3]));
    let y = spd_from(&tape, &x, 4.0).unwrap().cholesky().unwrap();
    let z = tape
        .custom(Arc::new(IdentityCustomFn), &[y])
        .expect("Tape::custom 登録成功");
    let loss = z.sum(None).unwrap();
    let r = tape.backward_create_graph(&loss, &child);
    assert!(is_backward_err(&r), "Op::Custom は拒否される");
    assert!(child.is_empty(), "拒否後は child が空");
    // 対象 Op だけの損失は受理される。
    let ok = y.sum(None).unwrap();
    tape.backward_create_graph(&ok, &child)
        .expect("拒否後の再試行は成功する");
}
