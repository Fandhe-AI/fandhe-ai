//! `QrQ`・`QrR` の `create_graph`（`Tape::backward_create_graph`）対応（イシュー #3807・親 #3734・
//! 決定記録 `docs/autodiff-higher-order-grad-decision.md` §44・§67）の受け入れ条件を公開 API 経由で直接
//! 検証する統合テスト。本ファイルは分解族の置き場で、#3808（`Svd*`・`Eigh*`）が追記する。
//!
//! - Hessian を `Tape::backward` の 5 点中央差分（`create_graph` を経由しない独立経路）と REQ-2 統一複合
//!   判定（`common::req2_close`）で突合する。縦長・正方、損失は Q だけ・R だけ・両方（兄弟が祖先に
//!   無い場合と有る場合の両方を通す）。
//! - `cg.first_order()` は `Tape::backward` と bit 同一、`cg.grad(..)` は V3 のため CPU で bit 一致する。
//!   閉包性（孫テープ）と `jvp_create_graph` も確かめる。
//! - 横長・rank 落ち・空行列・巨大な上流・悪条件は `Err(Backward)` かつ `child.is_empty()` で、同じ
//!   `child` を再利用できる。縦長 `[100000,1]` は `m×m` を確保せず受理される。隣接する未対応 Op
//!   （`Op::Custom`）は fail-closed のまま。
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

/// QR の Hessian は f32 の勾配丸め雑音を刻みで割った分が絶対許容に近づくため、やや大きい刻みを使う
/// （5 点中央差分の打ち切り誤差は十分小さい）。判定式・tolerance は変えない。
const FD_STEP: f64 = 5e-2;

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
    let cg2 = match child.backward_create_graph(&scalar, &grand) {
        Ok(cg2) => cg2,
        // SVD・Eigh の総括判定は保守的で、三階の機構（孫テープ）では子テープ側の cotangent が
        // 大きく桁落ちの見積りに掛かることがある。拒否なら型付きエラーで孫テープは空のままであること。
        Err(AutodiffError::Backward(msg)) if ctx.starts_with("svd") => {
            assert!(grand.is_empty(), "{ctx}: 拒否後は孫テープが空");
            eprintln!("NOTE {ctx}: 孫テープは保守的に拒否: {msg}");
            return;
        }
        Err(e) => panic!("{ctx}: 孫テープへの再生（閉包性）: {e:?}"),
    };
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

// --- 1. 損失と入力 ---------------------------------------------------------------------------

/// 条件の良い一般行列（対角寄りのオフセットを足してフルランクにする）。
fn well_conditioned(shape: &[usize]) -> Vec<f32> {
    let cols = shape[1];
    x_data(shape)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v * 0.5 + if i / cols == i % cols { 2.0 } else { 0.0 })
        .collect()
}

fn loss_q<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    tanh_sum(&x.qr()?.q, tape)
}

fn loss_r<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    quad_sum(&x.qr()?.r, tape)
}

fn loss_both<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    let f = x.qr()?;
    tanh_sum(&f.q, tape)?.add(&quad_sum(&f.r, tape)?)
}

type Loss = for<'t> fn(&'t Tape, &Var<'t>) -> Built<'t>;

const LOSSES: [(&str, Loss); 3] = [("q-only", loss_q), ("r-only", loss_r), ("q+r", loss_both)];

// --- 2. Hessian・1 階 bit 同一・孫テープ --------------------------------------------------------

#[test]
fn qr_hessian_matches_finite_difference() {
    for shape in [[4usize, 2], [3, 2], [3, 3]] {
        let x0 = well_conditioned(&shape);
        for (name, loss) in LOSSES {
            check_all_h(&format!("qr {name} {shape:?}"), loss, &x0, &shape, FD_STEP);
        }
    }
}

/// V3 の値は 1 階と同じホスト関数のため、`cg.grad` は CPU で 1 階と bit 一致する（線形な上流）。
#[test]
fn qr_cg_grad_bit_identical_for_linear_upstream() {
    for shape in [[4usize, 2], [3, 3]] {
        let x0 = well_conditioned(&shape);
        for which in ["q", "r", "both"] {
            let run = |cg: bool| -> Vec<f32> {
                let tape = Tape::new_with_ops(common::naive_ops());
                let x = tape.var(&t(x0.clone(), &shape));
                let f = x.qr().unwrap();
                let k = shape[0].min(shape[1]);
                let cq = tape.var_no_grad(&t(pattern(shape[0] * k, 3.0), &[shape[0], k]));
                let cr = tape.var_no_grad(&t(pattern(k * shape[1], 5.0), &[k, shape[1]]));
                let lq = f.q.mul(&cq).unwrap().sum(None).unwrap();
                let lr = f.r.mul(&cr).unwrap().sum(None).unwrap();
                let loss = match which {
                    "q" => lq,
                    "r" => lr,
                    _ => lq.add(&lr).unwrap(),
                };
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
                "qr {which} {shape:?}: cg.grad は 1 階と bit 一致"
            );
        }
    }
}

#[test]
fn qr_jvp_create_graph_matches_central_difference() {
    let shape = [3usize, 2];
    let x0 = well_conditioned(&shape);
    for (name, loss_fn) in LOSSES {
        let v = pattern(numel(&shape), 5.0);
        let loss_at = |x: &[f32]| -> f64 {
            let tape = Tape::new_with_ops(common::naive_ops());
            let xv = tape.var(&t(x.to_vec(), &shape));
            f64::from(dense(&loss_fn(&tape, &xv).unwrap().to_tensor())[0])
        };
        let h = 0.01f32;
        let plus: Vec<f32> = x0.iter().zip(&v).map(|(a, b)| a + h * b).collect();
        let minus: Vec<f32> = x0.iter().zip(&v).map(|(a, b)| a - h * b).collect();
        let expected = (loss_at(&plus) - loss_at(&minus)) / (2.0 * f64::from(h));

        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0.clone(), &shape));
        let loss = loss_fn(&tape, &x).unwrap();
        let res = jvp_create_graph(&tape, &loss, &x, &t(v.clone(), &shape), &child, &grand)
            .expect("jvp_create_graph");
        let got = f64::from(dense(&res.output().to_tensor())[0]);
        assert!(
            common::req2_close(got, expected),
            "{name}: jvp={got} vs central difference={expected}"
        );
    }
}

// --- 3. 拒否域 ---------------------------------------------------------------------------

fn sum_q_plus_r<'t>(_tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    let f = x.qr()?;
    f.q.sum(None)?.add(&f.r.sum(None)?)
}

#[test]
fn qr_degenerate_inputs_rejected_child_empty_and_retry() {
    let cases: Vec<(&str, Vec<f32>, Vec<usize>)> = vec![
        (
            "zero column",
            vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
            vec![3, 2],
        ),
        (
            "dependent columns",
            vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0],
            vec![3, 2],
        ),
        ("wide", vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0], vec![2, 3]),
        ("empty cols", vec![], vec![3, 0]),
        ("empty", vec![], vec![0, 0]),
    ];
    for (ctx, data, shape) in cases {
        assert_rejected_and_child_reusable(ctx, sum_q_plus_r, &data, &shape);
    }
}

/// 横長は 1 階の `backward` 単体では `InvalidArgument`（`Backward` ではない）だが、create_graph は
/// 入口で `Backward` として拒否する。
#[test]
fn qr_wide_first_order_errs_but_create_graph_is_backward_err() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0], &[2, 3]));
    let loss = x.qr().unwrap().r.sum(None).unwrap();
    assert!(tape.backward(&loss).is_err());
    assert!(is_backward_err(&tape.backward_create_graph(&loss, &child)));
    assert!(child.is_empty());
}

#[test]
fn qr_unsafe_inputs_rejected_child_empty() {
    let shape = [3usize, 2];
    // 巨大な上流（g = ±3e38）。
    for scale in [3.0e38f32, -3.0e38] {
        assert_rejected_and_child_reusable(
            "huge upstream",
            move |tape, x| {
                let c = tape.var_no_grad(&t(vec![scale], &[1]));
                x.qr()?.q.sum(None)?.reshape(&[1])?.mul(&c)?.sum(None)
            },
            &well_conditioned(&shape),
            &shape,
        );
    }
    // ほぼ平行な列（悪条件の R）。
    assert_rejected_and_child_reusable(
        "ill-conditioned",
        |_t, x| x.qr()?.q.sum(None),
        &[1.0, 1.0, 1.0, 1.000_001, 1.0, 1.000_002],
        &shape,
    );
}

#[test]
fn qr_moderate_scales_accepted_and_finite() {
    let shape = [3usize, 2];
    for scale in [1e-3f32, 1.0, 1e3] {
        let x0: Vec<f32> = well_conditioned(&shape).iter().map(|v| v * scale).collect();
        for (name, loss) in LOSSES {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), &shape));
            let l = loss(&tape, &x).unwrap();
            let cg = tape
                .backward_create_graph(&l, &child)
                .unwrap_or_else(|e| panic!("{name} scale={scale}: 受理されるはず: {e}"));
            let gx = cg.grad(&x).unwrap().unwrap();
            let cx = cg.child_var(&x).unwrap().unwrap();
            let s = gx.sum(None).unwrap();
            let h = child.backward(&s).unwrap();
            assert!(dense(&gx.to_tensor()).iter().all(|v| v.is_finite()));
            assert!(
                dense(h.get(&cx).unwrap().unwrap())
                    .iter()
                    .all(|v| v.is_finite()),
                "{name} scale={scale}: 二階が有限"
            );
        }
    }
}

// --- 4. m×m を確保しない ------------------------------------------------------------------

#[test]
fn qr_cgrad_does_not_materialize_m_by_m() {
    for (m, n) in [(100_000usize, 1usize), (5, 2)] {
        let data: Vec<f32> = (0..m * n)
            .map(|i| (1.0 + ((i * 7) % 11) as f32 * 0.37) * if m > 1000 { 100.0 } else { 1.0 })
            .collect();
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(data, &[m, n]));
        // 非一様な重み（`sum` だと g が Q と平行になり桁落ちの正当な拒否域に入る）。縦長は縮約長 m が
        // 大きく f32 積和の保守的な丸め上界が絶対許容を超えやすいため、値のスケールを上げて受理域に入れる。
        let loss = loss_both(&tape, &x).unwrap();
        let cg = tape
            .backward_create_graph(&loss, &child)
            .unwrap_or_else(|e| panic!("({m},{n}) は受理されるはず: {e}"));
        let g = dense(&cg.grad(&x).unwrap().unwrap().to_tensor());
        assert!(g.iter().all(|v| v.is_finite()), "({m},{n}) 有限");
    }
}

// --- 5. 隣接する未対応 Op は fail-closed のまま -----------------------------------------------

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
fn qr_neighbouring_unsupported_op_stays_fail_closed() {
    let shape = [3usize, 2];
    let x0 = well_conditioned(&shape);
    // QR と Op::Custom を同じ損失に混ぜる。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.clone(), &shape));
    let q = x.qr().unwrap().q;
    let z = tape
        .custom(Arc::new(IdentityCustomFn), &[q])
        .expect("Tape::custom 登録成功");
    let loss = z.sum(None).unwrap();
    assert!(is_backward_err(&tape.backward_create_graph(&loss, &child)));
    assert!(child.is_empty(), "拒否後は child が空");
}

/// 必須ベクトル（決定記録 §44.5）: 極端なスケール（`1e20`・`1e-20`）の入力は、実測ではどの損失でも
/// 二階の上界または段の overflow／ゼロ化域で拒否される（`Err(Backward)`・`child` は空）。受理域は
/// `qr_moderate_scales_accepted_and_finite`（`1e-3`〜`1e3`）。
#[test]
fn qr_extreme_scales_mandatory_vectors() {
    let shape = [3usize, 2];
    for scale in [1e20f32, 1e-20] {
        let x0: Vec<f32> = well_conditioned(&shape).iter().map(|v| v * scale).collect();
        for (name, loss) in LOSSES {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), &shape));
            let l = loss(&tape, &x).unwrap();
            let r = tape.backward_create_graph(&l, &child);
            assert!(is_backward_err(&r), "{name} scale={scale}: 拒否のはず");
            assert!(child.is_empty(), "{name} scale={scale}: child は空");
        }
    }
}
// =====================================================================================
// SvdU・SvdS・SvdVh・EighValues・EighVectors（イシュー #3808・決定記録 §44・§68）
// =====================================================================================

/// スペクトルがよく分離した（σ ≈ 3, 2, 1）一般行列。縦長・横長・正方で使う。
fn sv_separated(shape: &[usize]) -> Vec<f32> {
    let cols = shape[1];
    x_data(shape)
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let (r, c) = (i / cols, i % cols);
            let d = if r == c { 3.0 - r as f32 } else { 0.0 };
            v * 0.15 + d
        })
        .collect()
}

/// 下三角のみが読まれる対称入力。固有値が十分に分離している。
fn eigh_input(n: usize) -> Vec<f32> {
    (0..n * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            if r == c {
                1.0 + 1.7 * r as f32
            } else {
                0.3 / (1.0 + (r + c) as f32) * if r > c { 1.0 } else { 0.6 }
            }
        })
        .collect()
}

fn svd_u<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    tanh_sum(&x.svd()?.u, tape)
}

fn svd_s<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    quad_sum(&x.svd()?.s, tape)
}

fn svd_vh<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    tanh_sum(&x.svd()?.vh, tape)
}

fn svd_all<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    let f = x.svd()?;
    tanh_sum(&f.u, tape)?
        .add(&quad_sum(&f.s, tape)?)?
        .add(&tanh_sum(&f.vh, tape)?)
}

/// gauge 不変な損失: 再構成 `U diag(s) Vh` に対する非線形和。
fn svd_recon<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    let f = x.svd()?;
    let k = f.s.to_tensor().shape()[0];
    let recon = f.u.mul(&f.s.reshape(&[1, k])?)?.matmul(&f.vh)?;
    tanh_sum(&recon, tape)
}

/// gauge 不変な損失: `U_ij²` と `Vh_ij²` の重み付き和。
fn svd_sq<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    let f = x.svd()?;
    quad_sum(&f.u, tape)?.add(&quad_sum(&f.vh, tape)?)
}

const SVD_LOSSES: [(&str, Loss); 6] = [
    ("u", svd_u),
    ("s", svd_s),
    ("vh", svd_vh),
    ("u+s+vh", svd_all),
    ("recon", svd_recon),
    ("sq", svd_sq),
];

fn eigh_values<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    tanh_sum(&x.eigh()?.eigenvalues, tape)
}

fn eigh_vectors<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    quad_sum(&x.eigh()?.eigenvectors, tape)
}

fn eigh_both<'t>(tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    let f = x.eigh()?;
    tanh_sum(&f.eigenvalues, tape)?.add(&tanh_sum(&f.eigenvectors, tape)?)
}

const EIGH_LOSSES: [(&str, Loss); 3] = [
    ("values", eigh_values),
    ("vectors", eigh_vectors),
    ("values+vectors", eigh_both),
];

#[test]
fn svd_hessian_matches_finite_difference() {
    for shape in [[4usize, 3], [2, 3], [3, 3]] {
        let x0 = sv_separated(&shape);
        for (name, loss) in SVD_LOSSES {
            check_all_h(&format!("svd {name} {shape:?}"), loss, &x0, &shape, FD_STEP);
        }
    }
}

#[test]
fn eigh_hessian_matches_finite_difference() {
    for n in [2usize, 3, 4] {
        let shape = [n, n];
        let x0 = eigh_input(n);
        for (name, loss) in EIGH_LOSSES {
            check_all_h(&format!("eigh {name} n={n}"), loss, &x0, &shape, FD_STEP);
        }
        // forward は下三角だけを読む: 上三角方向の勾配は 0。
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0.clone(), &shape));
        let loss = eigh_both(&tape, &x).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let g = dense(&cg.grad(&x).unwrap().unwrap().to_tensor());
        for r in 0..n {
            for c in (r + 1)..n {
                assert_eq!(g[r * n + c], 0.0, "n={n}: 上三角の勾配は 0");
            }
        }
    }
}

/// V3 の値は 1 階と同じホスト関数のため、`cg.grad` は CPU で 1 階と bit 一致する（線形な上流）。
#[test]
fn svd_eigh_cg_grad_bit_identical_for_linear_upstream() {
    let run = |which: &str, cg: bool| -> Vec<f32> {
        let (shape, x0) = if which.starts_with("eigh") {
            (vec![3usize, 3], eigh_input(3))
        } else {
            (vec![4usize, 3], sv_separated(&[4, 3]))
        };
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0, &shape));
        let w = |len: usize, ph: f32, sh: &[usize]| tape.var_no_grad(&t(pattern(len, ph), sh));
        let loss = match which {
            "svd-u" => x.svd().unwrap().u.mul(&w(12, 3.0, &[4, 3])).unwrap(),
            "svd-s" => x.svd().unwrap().s.mul(&w(3, 5.0, &[3])).unwrap(),
            "svd-vh" => x.svd().unwrap().vh.mul(&w(9, 7.0, &[3, 3])).unwrap(),
            "eigh-values" => x.eigh().unwrap().eigenvalues.mul(&w(3, 3.0, &[3])).unwrap(),
            _ => x
                .eigh()
                .unwrap()
                .eigenvectors
                .mul(&w(9, 5.0, &[3, 3]))
                .unwrap(),
        }
        .sum(None)
        .unwrap();
        if cg {
            let child = Tape::new_with_ops(common::naive_ops());
            let grads = tape.backward_create_graph(&loss, &child).expect("cg");
            dense(&grads.grad(&x).unwrap().unwrap().to_tensor())
        } else {
            dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap())
        }
    };
    for which in ["svd-u", "svd-s", "svd-vh", "eigh-values", "eigh-vectors"] {
        let (a, b) = (run(which, false), run(which, true));
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{which}: cg.grad は 1 階と bit 一致"
        );
    }
}

#[test]
fn svd_eigh_jvp_create_graph_matches_central_difference() {
    let cases: Vec<(&str, Vec<usize>, Vec<f32>, Loss)> = vec![
        ("svd-all", vec![3, 2], sv_separated(&[3, 2]), svd_all),
        ("svd-recon", vec![3, 3], sv_separated(&[3, 3]), svd_recon),
        ("eigh-both", vec![3, 3], eigh_input(3), eigh_both),
        ("eigh-vectors", vec![3, 3], eigh_input(3), eigh_vectors),
    ];
    for (name, shape, x0, loss_fn) in cases {
        let v = pattern(numel(&shape), 5.0);
        let loss_at = |x: &[f32]| -> f64 {
            let tape = Tape::new_with_ops(common::naive_ops());
            let xv = tape.var(&t(x.to_vec(), &shape));
            f64::from(dense(&loss_fn(&tape, &xv).unwrap().to_tensor())[0])
        };
        let h = 0.01f32;
        let plus: Vec<f32> = x0.iter().zip(&v).map(|(a, b)| a + h * b).collect();
        let minus: Vec<f32> = x0.iter().zip(&v).map(|(a, b)| a - h * b).collect();
        let expected = (loss_at(&plus) - loss_at(&minus)) / (2.0 * f64::from(h));
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0.clone(), &shape));
        let loss = loss_fn(&tape, &x).unwrap();
        let res = jvp_create_graph(&tape, &loss, &x, &t(v.clone(), &shape), &child, &grand)
            .expect("jvp_create_graph");
        let got = f64::from(dense(&res.output().to_tensor())[0]);
        assert!(
            common::req2_close(got, expected),
            "{name}: jvp={got} vs central difference={expected}"
        );
    }
}

/// 兄弟が祖先に無い（U だけ・S だけ・Vh だけ・固有値だけ等）場合でも二階が正しい。
#[test]
fn svd_eigh_sibling_not_in_ancestors_second_order() {
    let shape = [4usize, 3];
    let x0 = sv_separated(&shape);
    for (name, loss) in [("s", svd_s as Loss), ("u", svd_u), ("vh", svd_vh)] {
        check_all_h(&format!("svd-only {name}"), loss, &x0, &shape, FD_STEP);
    }
    // U と Vh だけが祖先（S は祖先に無い）。
    let uvh: Loss = |tape, x| {
        let f = x.svd()?;
        tanh_sum(&f.u, tape)?.add(&tanh_sum(&f.vh, tape)?)
    };
    check_all_h("svd u+vh", uvh, &x0, &shape, FD_STEP);
}

#[test]
fn svd_eigh_degenerate_inputs_rejected_child_empty_and_retry() {
    // Eigh: 単位行列・重複固有値・しきい値の内側の近接スペクトル・空行列。
    let eigh_cases: Vec<(&str, Vec<f32>, Vec<usize>)> = vec![
        ("eigh(I_2)", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
        (
            "eigh(diag(2,2,1))",
            vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0],
            vec![3, 3],
        ),
        // 最大固有値 5 に対し 0 と 1e-12 の差は閾値 5e-9 の内側。
        (
            "eigh near-degenerate",
            vec![0.0, 0.0, 0.0, 0.0, 1.0e-12, 0.0, 0.0, 0.0, 5.0],
            vec![3, 3],
        ),
        ("eigh empty", vec![], vec![0, 0]),
    ];
    for (ctx, data, shape) in eigh_cases {
        assert_rejected_and_child_reusable(ctx, eigh_both, &data, &shape);
        assert_rejected_and_child_reusable(ctx, eigh_values, &data, &shape);
    }
    // SVD: 重複特異値・σ = 0・空行列。
    let svd_cases: Vec<(&str, Vec<f32>, Vec<usize>)> = vec![
        ("svd(diag(2,2))", vec![2.0, 0.0, 0.0, 2.0], vec![2, 2]),
        ("svd sigma=0", vec![1.0, 0.0, 0.0, 0.0], vec![2, 2]),
        ("svd empty", vec![], vec![0, 3]),
    ];
    for (ctx, data, shape) in svd_cases {
        assert_rejected_and_child_reusable(ctx, svd_all, &data, &shape);
        assert_rejected_and_child_reusable(ctx, svd_s, &data, &shape);
    }
}

/// 固有値だけの損失では、縮退入力でも 1 階の `backward` は Ok だが create_graph は拒否する
/// （二階は `1/E²` で破綻するため。1 階との意図的な違い）。
#[test]
fn eigh_degenerate_values_only_first_order_ok_but_create_graph_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let loss = x.eigh().unwrap().eigenvalues.sum(None).unwrap();
    assert!(tape.backward(&loss).is_ok());
    assert!(is_backward_err(&tape.backward_create_graph(&loss, &child)));
    assert!(child.is_empty());
}

#[test]
fn svd_eigh_unsafe_inputs_rejected_child_empty() {
    // 巨大な上流（g = ±3e38）。
    for scale in [3.0e38f32, -3.0e38] {
        assert_rejected_and_child_reusable(
            "svd huge upstream",
            move |tape, x| {
                let c = tape.var_no_grad(&t(vec![scale], &[1]));
                x.svd()?.s.sum(None)?.reshape(&[1])?.mul(&c)?.sum(None)
            },
            &sv_separated(&[3, 2]),
            &[3, 2],
        );
        assert_rejected_and_child_reusable(
            "eigh huge upstream",
            move |tape, x| {
                let c = tape.var_no_grad(&t(vec![scale], &[1]));
                x.eigh()?
                    .eigenvectors
                    .sum(None)?
                    .reshape(&[1])?
                    .mul(&c)?
                    .sum(None)
            },
            &eigh_input(3),
            &[3, 3],
        );
    }
    // 近接スペクトル（閾値 1e-12 の外だが 1/E² の上界が 2^125 を超える）。
    assert_rejected_and_child_reusable(
        "eigh close spectrum",
        eigh_vectors,
        &[0.0, 0.0, 0.0, 0.0, 2.0e-12, 0.0, 0.0, 0.0, 1.0e-3],
        &[3, 3],
    );
}

#[test]
fn svd_eigh_moderate_scales_accepted_and_finite() {
    for scale in [1e-3f32, 1.0, 1e3] {
        let svd_x: Vec<f32> = sv_separated(&[3, 2]).iter().map(|v| v * scale).collect();
        let eigh_x: Vec<f32> = eigh_input(3).iter().map(|v| v * scale).collect();
        let runs: Vec<(String, Loss, Vec<f32>, Vec<usize>)> = SVD_LOSSES
            .iter()
            .map(|(n, l)| (format!("svd {n}"), *l, svd_x.clone(), vec![3usize, 2]))
            .chain(
                EIGH_LOSSES
                    .iter()
                    .map(|(n, l)| (format!("eigh {n}"), *l, eigh_x.clone(), vec![3usize, 3])),
            )
            .collect();
        for (name, loss, x0, shape) in runs {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0, &shape));
            let l = loss(&tape, &x).unwrap();
            let cg = tape
                .backward_create_graph(&l, &child)
                .unwrap_or_else(|e| panic!("{name} scale={scale}: 受理されるはず: {e}"));
            let gx = cg.grad(&x).unwrap().unwrap();
            let cx = cg.child_var(&x).unwrap().unwrap();
            let h = child.backward(&gx.sum(None).unwrap()).unwrap();
            assert!(dense(&gx.to_tensor()).iter().all(|v| v.is_finite()));
            assert!(
                dense(h.get(&cx).unwrap().unwrap())
                    .iter()
                    .all(|v| v.is_finite()),
                "{name} scale={scale}: 二階が有限"
            );
        }
    }
}

/// 極端なスケールの必須ベクトル（§44.5）。どの損失でも、拒否（`Err(Backward)` かつ `child` は空）か、
/// 受理なら 1 階の勾配が有限のどちらかで、panic も非有限の受理も起きない。
/// SVD の `1e-20` は σ² の絶対判定（1e-9）で拒否される（1 階の `svd_vjp` と同じ挙動）。
#[test]
fn svd_eigh_extreme_scales_mandatory_vectors() {
    for scale in [1e20f32, 1e-20] {
        let mut runs: Vec<(String, Loss, Vec<f32>, Vec<usize>)> = Vec::new();
        for (n, l) in SVD_LOSSES {
            let x0 = sv_separated(&[3, 2]).iter().map(|v| v * scale).collect();
            runs.push((format!("svd {n}"), l, x0, vec![3, 2]));
        }
        for (n, l) in EIGH_LOSSES {
            let x0 = eigh_input(3).iter().map(|v| v * scale).collect();
            runs.push((format!("eigh {n}"), l, x0, vec![3, 3]));
        }
        for (name, loss, x0, shape) in runs {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0, &shape));
            let l = loss(&tape, &x).unwrap();
            match tape.backward_create_graph(&l, &child) {
                Ok(cg) => {
                    let g = dense(&cg.grad(&x).unwrap().unwrap().to_tensor());
                    assert!(
                        g.iter().all(|v| v.is_finite()),
                        "{name} scale={scale}: 受理なら有限"
                    );
                }
                Err(e) => {
                    assert!(matches!(e, AutodiffError::Backward(_)), "{name}: {e:?}");
                    assert!(child.is_empty(), "{name} scale={scale}: child は空");
                }
            }
        }
    }
}

/// `[100000,1]` の SVD は受理され、`m×m` を確保しない（確保すれば 10^10 要素でテストが落ちる）。
#[test]
fn svd_cgrad_does_not_materialize_m_by_m() {
    let (m, n) = (100_000usize, 1usize);
    let data: Vec<f32> = (0..m * n)
        .map(|i| (1.0 + ((i * 7) % 11) as f32 * 0.37) * 1000.0)
        .collect();
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(data, &[m, n]));
    let f = x.svd().unwrap();
    let w = tape.var_no_grad(&t(pattern(m, 3.0), &[m, 1]));
    let loss = f.u.mul(&w).unwrap().sum(None).unwrap();
    let cg = tape
        .backward_create_graph(&loss, &child)
        .unwrap_or_else(|e| panic!("[100000,1] は受理されるはず: {e}"));
    let g = dense(&cg.grad(&x).unwrap().unwrap().to_tensor());
    assert!(g.iter().all(|v| v.is_finite()));
}

/// 隣接する未対応 Op（`Op::Custom`）と組み合わせると拒否され、`child` は空。
#[test]
fn svd_eigh_neighbouring_unsupported_op_stays_fail_closed() {
    for which in ["svd", "eigh"] {
        let (shape, x0): (Vec<usize>, Vec<f32>) = if which == "svd" {
            (vec![3, 2], sv_separated(&[3, 2]))
        } else {
            (vec![3, 3], eigh_input(3))
        };
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0, &shape));
        let y = if which == "svd" {
            x.svd().unwrap().s
        } else {
            x.eigh().unwrap().eigenvalues
        };
        let z = tape
            .custom(Arc::new(IdentityCustomFn), &[y])
            .expect("Tape::custom 登録成功");
        let loss = z.sum(None).unwrap();
        assert!(is_backward_err(&tape.backward_create_graph(&loss, &child)));
        assert!(child.is_empty(), "{which}: 拒否後は child が空");
    }
}
/// 回帰（PR #3967 Codex P1）: 近接した特異値 `diag(1.400001, 1.4)` で、F の分母 `σj² − σi²` を f32 の
/// 二乗差で作ると桁落ちして二階が約 1.6% ずれた。分母を `(σj − σi)(σj + σi)` で作る安定構成にし、
/// 受理された場合は f64 の 2x2 閉形式（差分）と REQ-2 で一致、拒否（fail-closed）でも合格とする。
/// `SvdU`・`SvdVh` の両経路を通す。tolerance は変えない。
fn svd2_loss_ref(a: [f64; 4], w: [f64; 4], which: usize, sign: [f64; 2]) -> f64 {
    let (p, q, r) = if which == 0 {
        (
            a[0] * a[0] + a[1] * a[1],
            a[0] * a[2] + a[1] * a[3],
            a[2] * a[2] + a[3] * a[3],
        )
    } else {
        (
            a[0] * a[0] + a[2] * a[2],
            a[0] * a[1] + a[2] * a[3],
            a[1] * a[1] + a[3] * a[3],
        )
    };
    let th = 0.5 * (2.0 * q).atan2(p - r);
    let (c, s) = (th.cos(), th.sin());
    // 列ベクトル（which = 0: U の列、1: V の列。Vh は転置して重みを掛ける）。
    let m = [[c * sign[0], -s * sign[1]], [s * sign[0], c * sign[1]]];
    let mut acc = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            let v = if which == 0 { m[i][j] } else { m[j][i] };
            acc += w[i * 2 + j] * v;
        }
    }
    acc
}

#[test]
fn svd_close_singular_values_second_order_is_stable() {
    let x0 = vec![1.400001f32, 0.0, 0.0, 1.4];
    let w = [1e-5f64 * 0.3, 1e-5 * 0.9, 1e-5 * 0.6, 1e-5 * 0.1];
    for which in 0..2usize {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0.clone(), &[2, 2]));
        let f = x.svd().unwrap();
        let y = if which == 0 { f.u } else { f.vh };
        let u_val = dense(&f.u.value());
        let vh_val = dense(&f.vh.value());
        let wt = tape.var_no_grad(&t(w.iter().map(|&v| v as f32).collect(), &[2, 2]));
        let loss = y.mul(&wt).unwrap().sum(None).unwrap();
        let cg = match tape.backward_create_graph(&loss, &child) {
            Ok(cg) => cg,
            Err(AutodiffError::Backward(_)) => {
                assert!(child.is_empty());
                continue;
            }
            Err(e) => panic!("想定外のエラー: {e:?}"),
        };
        let sign = if which == 0 {
            [f64::from(u_val[0]).signum(), f64::from(u_val[3]).signum()]
        } else {
            [f64::from(vh_val[0]).signum(), f64::from(vh_val[3]).signum()]
        };
        let gx = cg.grad(&x).unwrap().unwrap();
        let cx = cg.child_var(&x).unwrap().unwrap();
        let h = 2e-8f64;
        let base: [f64; 4] = [x0[0], x0[1], x0[2], x0[3]].map(f64::from);
        for j in 0..4 {
            let gj = extract_scalar(&gx, &[2, 2], j).unwrap();
            let rg = child.backward(&gj).unwrap();
            let row = dense(rg.get(&cx).unwrap().unwrap());
            for i in 0..4 {
                let ev = |di: f64, dj: f64| {
                    let mut a = base;
                    a[i] += di;
                    a[j] += dj;
                    svd2_loss_ref(a, w, which, sign)
                };
                let num = if i == j {
                    (-ev(2.0 * h, 0.0) + 16.0 * ev(h, 0.0) - 30.0 * ev(0.0, 0.0)
                        + 16.0 * ev(-h, 0.0)
                        - ev(-2.0 * h, 0.0))
                        / (12.0 * h * h)
                } else {
                    (ev(h, h) - ev(h, -h) - ev(-h, h) + ev(-h, -h)) / (4.0 * h * h)
                };
                assert!(
                    common::req2_close(f64::from(row[i]), num),
                    "which={which} hessian[{j}][{i}] analytic={} ref={num}",
                    row[i]
                );
            }
        }
    }
}
