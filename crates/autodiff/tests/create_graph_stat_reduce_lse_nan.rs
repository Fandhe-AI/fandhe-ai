//! 統計縮約（`LogSumExp`・`Nansum`・`Nanmean`）の `create_graph` 対応（イシュー #3792・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §33、実装記録 §51）の受け入れ条件を確かめる統合テスト。
//!
//! - Hessian を独立な有限差分（1 階解析勾配の中央差分）と REQ-2 統一複合判定で突合し、
//!   `LogSumExp` は閉形式 `diag(s) − s sᵀ` でも確かめる。
//! - `first_order()` と `cg.grad` が 1 階 `Tape::backward` と bit 一致する（NaN・`±inf`・全 NaN レーンを含む）。
//! - 総括判定: 非有限・過大な上流は子テープへ書く前に拒否され `child` が空のまま再試行できる。
//! - 孫テープ（`jvp_create_graph`）の出力を独立参照と突合する。
//! - 隣接する未対応 Op（`selu`）は fail-closed のままで `child` が空に保たれる。
//!
//! GPU parity は `crates/facade/tests/create_graph_gpu_parity.rs`（実機待ち・`#[ignore]`）。

#![allow(clippy::needless_range_loop)]

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

type BuildResult<'t> = Result<Var<'t>, AutodiffError>;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn flat_len(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn unravel(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        out[d] = idx % shape[d];
        idx /= shape[d];
    }
    out
}

fn shp(v: &Var<'_>) -> Vec<usize> {
    v.value().shape().to_vec()
}

fn dense(v: impl std::borrow::Borrow<Tensor<f32>>) -> Vec<f32> {
    v.borrow()
        .contiguous()
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

fn extract_scalar<'c>(
    v: &Var<'c>,
    shape: &[usize],
    flat_idx: usize,
) -> Result<Var<'c>, AutodiffError> {
    let idx = unravel(flat_idx, shape);
    let mut cur = *v;
    for (dim, &i) in idx.iter().enumerate() {
        cur = cur.narrow(dim, i, 1)?;
    }
    cur.sum(None)
}

fn finite_diff_hessian<F>(build: F, x0: &[f32], shape: &[usize], h: f64) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let n = flat_len(shape);
    let grad_at = |x: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), shape));
        let loss = build(&tape, &xv).expect("forward");
        let grads = tape.backward(&loss).expect("backward");
        let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");
        (0..n)
            .map(|i| dx.get(&unravel(i, shape)).unwrap() as f64)
            .collect()
    };
    let mut hessian = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let mut xp = x0.to_vec();
        let mut xm = x0.to_vec();
        xp[i] += h as f32;
        xm[i] -= h as f32;
        let gp = grad_at(&xp);
        let gm = grad_at(&xm);
        for j in 0..n {
            hessian[j][i] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    hessian
}

fn analytic_hessian<F>(build: F, x0: &[f32], shape: &[usize]) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let n = flat_len(shape);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let gx = cg.grad(&x).unwrap().expect("1 階勾配");
    let cx = cg.child_var(&x).unwrap().expect("写し");
    let mut hessian = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = extract_scalar(&gx, shape, j).unwrap();
        let row_grads = child.backward(&gj).expect("2 階");
        let row = row_grads.get(&cx).unwrap().expect("2 階勾配は x へ到達");
        for i in 0..n {
            hessian[j][i] = row.get(&unravel(i, shape)).unwrap() as f64;
        }
    }
    hessian
}

fn assert_hessian_close(analytic: &[Vec<f64>], numeric: &[Vec<f64>], ctx: &str) {
    assert_eq!(analytic.len(), numeric.len());
    for (j, (ra, rn)) in analytic.iter().zip(numeric).enumerate() {
        for (i, (&a, &n)) in ra.iter().zip(rn).enumerate() {
            assert!(
                common::req2_close(a, n),
                "{ctx}: hessian[{j}][{i}] analytic={a} numeric={n}"
            );
        }
    }
}

fn hessian_case<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t> + Copy,
{
    hessian_case_h(ctx, build, x0, shape, 0.05);
}

/// 差分幅 `h` 指定版。`LogSumExp` の小さな非対角成分は `h = 0.05` だと打切り誤差（`f'''·h²/6`）が
/// REQ-2 の絶対側閾値を超えるため `h = 0.01` を使う（f32 の勾配ノイズは約 1e-7/h で許容内）。
fn hessian_case_h<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize], h: f64)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t> + Copy,
{
    let analytic = analytic_hessian(build, x0, shape);
    let numeric = finite_diff_hessian(build, x0, shape, h);
    assert_hessian_close(&analytic, &numeric, ctx);
    // 二階が自明に 0 の検査を弾く（H ≠ 0）。
    assert!(
        analytic.iter().flatten().any(|v| v.abs() > 1e-3),
        "{ctx}: Hessian が全ゼロ（検査が空振り）"
    );
}

fn distinct_vals(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + 5) % 29) as f32) * 0.21 - 2.0)
        .collect()
}

/// `sum(y ⊙ y ⊙ c)`（`c` は y と同 shape の非一様な定数）。
fn quad<'t>(tape: &'t Tape, y: &Var<'t>) -> BuildResult<'t> {
    let shape = shp(y);
    let c = tape.var_no_grad(&t(common::det_pos(flat_len(&shape).max(1), 0.7), &shape));
    y.mul(y)?.mul(&c)?.sum(None)
}

fn assert_first_order_bit_identical<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward");
    let reference = tape.backward(&loss).unwrap();
    let want = dense(reference.get(&x).unwrap().expect("x は到達"));
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let first = dense(cg.first_order().get(&x).unwrap().expect("x は到達"));
    let got = dense(cg.grad(&x).unwrap().expect("x は到達").to_tensor());
    // NaN は payload 非依存でクラス一致、それ以外は bit 一致。
    let norm = |v: &[f32]| -> Vec<u32> {
        v.iter()
            .map(|a| {
                if a.is_nan() {
                    f32::NAN.to_bits()
                } else {
                    a.to_bits()
                }
            })
            .collect()
    };
    assert_eq!(norm(&first), norm(&want), "{ctx}: first_order()");
    assert_eq!(norm(&got), norm(&want), "{ctx}: cg.grad");
}

type Reducer = for<'t> fn(&Var<'t>, Option<usize>) -> BuildResult<'t>;

fn reducers() -> Vec<(&'static str, Reducer)> {
    vec![
        ("logsumexp", |x, d| x.logsumexp(d)),
        ("nansum", |x, d| x.nansum(d)),
        ("nanmean", |x, d| x.nanmean(d)),
    ]
}

#[test]
fn first_order_and_cg_grad_are_bit_identical() {
    for shape in [vec![5usize], vec![2, 3], vec![2, 3, 4]] {
        let x0 = distinct_vals(flat_len(&shape));
        for (name, r) in reducers() {
            let mut dims: Vec<Option<usize>> = vec![None];
            dims.extend((0..shape.len()).map(Some));
            for d in dims {
                assert_first_order_bit_identical(
                    &format!("{name} {shape:?} {d:?}"),
                    |tp, x| quad(tp, &r(&x.mul(x)?, d)?),
                    &x0,
                    &shape,
                );
            }
        }
    }
}

#[test]
fn non_finite_inputs_keep_first_order_classes() {
    let inf = f32::INFINITY;
    let nan = f32::NAN;
    let cases: Vec<(&str, Vec<f32>, Vec<usize>)> = vec![
        ("[-inf,-inf]", vec![-inf, -inf], vec![2]),
        ("[+inf,1]", vec![inf, 1.0], vec![2]),
        ("[+inf,+inf]", vec![inf, inf], vec![2]),
        ("[nan,1]", vec![nan, 1.0], vec![2]),
        ("[nan,nan]", vec![nan, nan], vec![2]),
        ("lanes", vec![inf, 1.0, -inf, -inf, 0.5, nan], vec![3, 2]),
    ];
    for (name, x0, shape) in &cases {
        for (rname, r) in reducers() {
            for d in [None, Some(shape.len() - 1)] {
                assert_first_order_bit_identical(
                    &format!("{rname} {name} {d:?}"),
                    |_, x| r(x, d)?.sum(None),
                    x0,
                    shape,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// Hessian
// ---------------------------------------------------------------------------------

#[test]
fn stat_reduce_hessian_matches_finite_difference() {
    let shape = [2usize, 3, 4];
    let x0 = distinct_vals(flat_len(&shape));
    hessian_case_h("lse None", |_, x| x.logsumexp(None), &x0, &shape, 0.01);
    hessian_case_h(
        "lse axis1 quad",
        |tp, x| quad(tp, &x.logsumexp(Some(1))?),
        &x0,
        &shape,
        0.01,
    );
    hessian_case_h(
        "lse^2",
        |_, x| {
            let m = x.logsumexp(Some(2))?;
            m.mul(&m)?.sum(None)
        },
        &x0,
        &shape,
        0.01,
    );
    hessian_case(
        "nansum(x*x)^2",
        |_, x| {
            let m = x.mul(x)?.nansum(None)?;
            m.mul(&m)
        },
        &x0,
        &shape,
    );
    hessian_case(
        "nansum axis0 quad",
        |tp, x| quad(tp, &x.mul(x)?.nansum(Some(0))?),
        &x0,
        &shape,
    );
    hessian_case(
        "nanmean(x*x)^2",
        |_, x| {
            let m = x.mul(x)?.nanmean(None)?;
            m.mul(&m)
        },
        &x0,
        &shape,
    );
    hessian_case(
        "nanmean axis2 quad",
        |tp, x| quad(tp, &x.mul(x)?.nanmean(Some(2))?),
        &x0,
        &shape,
    );
}

#[test]
fn logsumexp_hessian_matches_closed_form() {
    let x0 = [0.3f32, -1.2, 0.8, 2.0];
    let h = analytic_hessian(|_, x| x.logsumexp(None), &x0, &[4]);
    let mx = x0.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let e: Vec<f64> = x0.iter().map(|&v| (v as f64 - mx).exp()).collect();
    let z: f64 = e.iter().sum();
    let s: Vec<f64> = e.iter().map(|v| v / z).collect();
    for j in 0..4 {
        for i in 0..4 {
            let want = if i == j { s[i] } else { 0.0 } - s[i] * s[j];
            assert!(
                common::req2_close(h[j][i], want),
                "H[{j}][{i}]={} want={want}",
                h[j][i]
            );
        }
    }
}

#[test]
fn mandatory_vectors_stay_finite_in_three_stages() {
    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("1e20", vec![1e20, 1e20]),
        ("pm1e20", vec![-1e20, 1e20]),
        ("tiny", vec![1e-30, 1e-30]),
        ("denormal", vec![1e-40, 2e-40]),
    ];
    for (name, x0) in &cases {
        for (rname, r) in reducers() {
            // LogSumExp の [1e20, 1e20] は一次 VJP と softmax 構造が乖離するため拒否（別テスト）。
            if rname == "logsumexp" && *name == "1e20" {
                continue;
            }
            // 1 階（Tape::backward）・cg.grad の bit 一致、孫テープ（jvp_create_graph）の有限性。
            assert_first_order_bit_identical(
                &format!("{rname} {name}"),
                |_, x| r(x, None),
                x0,
                &[2],
            );
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), &[2]));
            let loss = r(&x, None).unwrap();
            let cg = tape.backward_create_graph(&loss, &child).unwrap();
            let gx = cg.grad(&x).unwrap().unwrap();
            assert!(
                dense(gx.to_tensor()).iter().all(|v| v.is_finite()),
                "{rname} {name}"
            );
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let grand = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), &[2]));
            let loss = r(&x, None).unwrap();
            let res = fandhe_ai_autodiff::functional_ops::jvp_create_graph(
                &tape,
                &loss,
                &x,
                &t(vec![1.0, 0.5], &[2]),
                &child,
                &grand,
            )
            .unwrap();
            let jv = res.output().value().get(&[]).unwrap();
            assert!(jv.is_finite(), "{rname} {name}: jvp={jv}");
        }
    }
}

/// 一次 VJP は f64 の `ln(2) + 1e20` で `ln(2)` が落ち `[1, 1]` を返すが、構造は `[0.5, 0.5]` となり
/// 接ベクトル `[1, 0.5]` の二重 VJP が一次勾配と整合しない。型付きエラーで拒否し child は空のまま。
#[test]
fn logsumexp_large_shift_lane_is_rejected_not_silently_diverging() {
    assert_rejected(
        "logsumexp 1e20",
        |_, x| x.logsumexp(None),
        &[1e20, 1e20],
        &[2],
    );
    // 係数が一致する入力（相殺で ln(acc)=0 になる [-1e20, 1e20]）は受理され、独立参照と一致する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let grand = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-1e20, 1e20], &[2]));
    let loss = x.logsumexp(None).unwrap();
    let res = fandhe_ai_autodiff::functional_ops::jvp_create_graph(
        &tape,
        &loss,
        &x,
        &t(vec![1.0, 0.5], &[2]),
        &child,
        &grand,
    )
    .unwrap();
    // softmax = [0, 1]・ヘッシアン = 0 のため jvp は 0.5（一次 d = s·v）。
    let jv = res.output().value().get(&[]).unwrap() as f64;
    assert!(common::req2_close(jv, 0.5), "jvp={jv}");
}

#[test]
fn zero_denominator_lanes_have_finite_structure() {
    // LogSumExp の全 -inf レーンと nanmean の count=0（全 NaN）レーン。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![f32::NEG_INFINITY, f32::NEG_INFINITY], &[2]));
    let loss = x.logsumexp(None).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gx = cg.grad(&x).unwrap().unwrap();
    assert_eq!(dense(gx.to_tensor()), vec![0.0, 0.0]);
    let s = gx.sum(None).unwrap();
    let cx = cg.child_var(&x).unwrap().unwrap();
    let grads = child.backward(&s).unwrap();
    let row = grads.get(&cx).unwrap();
    assert!(row.is_none_or(|r| dense(r).iter().all(|v| v.is_finite())));

    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![f32::NAN, f32::NAN], &[2]));
    let loss = x.nanmean(None).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gx = cg.grad(&x).unwrap().unwrap();
    assert!(dense(gx.to_tensor()).iter().all(|v| v.is_nan()));
    // 構造は x に依存しない（二階は 0）ため、x への勾配追跡祖先が無く backward は型付きエラーになる。
    let s = gx.sum(None).unwrap();
    assert!(child.backward(&s).is_err());
}

// ---------------------------------------------------------------------------------
// 総括判定（非有限・過大な上流の拒否と再試行）
// ---------------------------------------------------------------------------------

#[test]
fn unsafe_upstream_is_rejected_child_stays_empty_and_retry_succeeds() {
    for (rname, r) in reducers() {
        for scale in [3e38f32, f32::INFINITY, f32::NAN] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(vec![0.5, 1.5, -0.5], &[3]));
            let c_bad = tape.var_no_grad(&t(vec![scale], &[1]));
            let c_ok = tape.var_no_grad(&t(vec![1e3], &[1]));
            let m = r(&x, None).unwrap();
            let bad = m.mul(&c_bad).unwrap().sum(None).unwrap();
            let ok = m.mul(&c_ok).unwrap().sum(None).unwrap();
            let err = tape.backward_create_graph(&bad, &child).unwrap_err();
            assert!(
                matches!(err, AutodiffError::Backward(_)),
                "{rname} {scale}: {err:?}"
            );
            assert!(child.is_empty(), "{rname} {scale}: child が空でない");
            // 同じ child で安全な損失を再試行できる。
            let cg = tape.backward_create_graph(&ok, &child).unwrap();
            let g = dense(cg.grad(&x).unwrap().unwrap().to_tensor());
            assert!(g.iter().all(|v| v.is_finite()), "{rname}: {g:?}");
        }
    }
}

#[test]
fn moderate_upstream_is_accepted_and_finite() {
    for (rname, r) in reducers() {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![0.5, 1.5, -0.5], &[3]));
        let c = tape.var_no_grad(&t(vec![1e30], &[1]));
        let loss = r(&x, None).unwrap().mul(&c).unwrap().sum(None).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let g = dense(cg.grad(&x).unwrap().unwrap().to_tensor());
        assert!(g.iter().all(|v| v.is_finite()), "{rname}: {g:?}");
    }
}

// ---------------------------------------------------------------------------------
// 孫テープ・fail-closed
// ---------------------------------------------------------------------------------

fn jvp_case<F>(ctx: &str, build: F, p0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let tangent: Vec<f32> = common::det_seq(p0.len(), 0.9);
    // 独立な参照: 1 階 `Tape::backward`（create_graph 非依存）の勾配と接ベクトルの f64 内積。
    let numeric = {
        let tape = Tape::new_with_ops(common::naive_ops());
        let p = tape.var(&t(p0.to_vec(), shape));
        let loss = build(&tape, &p).unwrap();
        let g = dense(tape.backward(&loss).unwrap().get(&p).unwrap().expect("p"));
        g.iter()
            .zip(&tangent)
            .map(|(a, v)| f64::from(*a) * f64::from(*v))
            .sum::<f64>()
    };
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let grand = Tape::new_with_ops(common::naive_ops());
    let p = tape.var(&t(p0.to_vec(), shape));
    let loss = build(&tape, &p).unwrap();
    let res = fandhe_ai_autodiff::functional_ops::jvp_create_graph(
        &tape,
        &loss,
        &p,
        &t(tangent, shape),
        &child,
        &grand,
    )
    .expect("統計縮約を含む関数でも jvp_create_graph が成立する");
    let jv = res.output().value().get(&[]).unwrap() as f64;
    assert!(
        common::req2_close(jv, numeric),
        "{ctx}: jvp={jv} numeric={numeric}"
    );
}

#[test]
fn jvp_create_graph_grandchild_matches_independent_reference() {
    let shape = [2usize, 3, 4];
    let x = distinct_vals(flat_len(&shape));
    for (name, r) in reducers() {
        jvp_case(
            &format!("{name} axis1"),
            |tp, x| quad(tp, &r(&x.mul(x)?, Some(1))?),
            &x,
            &shape,
        );
        jvp_case(
            &format!("{name} None"),
            |_, x| {
                let m = r(&x.tanh(), None)?;
                m.mul(&m)
            },
            &x,
            &shape,
        );
    }
}

fn assert_rejected<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).unwrap();
    let len = tape.len();
    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)), "{ctx}: {err:?}");
    assert!(child.is_empty(), "{ctx}: 子テープが無変更でない");
    assert_eq!(tape.len(), len, "{ctx}: 親テープが変化した");
}

#[test]
fn neighbouring_unsupported_op_stays_fail_closed() {
    let x0 = distinct_vals(6);
    assert_rejected("selu", |_, x| x.selu()?.sum(None), &x0, &[2, 3]);
    assert_rejected("lse + selu", |_, x| x.selu()?.logsumexp(None), &x0, &[2, 3]);
    assert_rejected(
        "nansum + selu",
        |_, x| x.selu()?.nansum(Some(1))?.sum(None),
        &x0,
        &[2, 3],
    );
}
