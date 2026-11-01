//! 選択縮約（`Max`・`Min`・`Amax`・`Amin`）の `create_graph` 対応（イシュー #3789・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §34）の受け入れ条件を確かめる統合テスト。
//!
//! - Hessian を独立な有限差分（1 階解析勾配の中央差分）と REQ-2 統一複合判定で突合する
//!   （タイ無しの入力。タイの点では摂動で勝者が入れ替わるため）。
//! - `Amax`／`Amin` のタイは解析的な閉形式（`(2/k²)·1_S 1_Sᵀ`）で確かめる。
//! - `first_order()` と `cg.grad` が 1 階 `Tape::backward` と bit 一致する（先勝ちタイ・符号付きゼロ・
//!   NaN・`k == 0`・大きな上流を含む）。
//! - 写しの Op タグが親と一致する（孫テープが正しい VJP を使う前提）。
//! - 孫テープ（`jvp_create_graph`）の出力を独立参照と突合する。
//! - 隣接する未対応 Op（`Op::Custom` を含むグラフ・`median`・`nansum`）は fail-closed のままで
//!   `child` が空に保たれる。
//!
//! 後続の #3790（`OrderSelect`・`MedianAll`・`Quantile`）が同じファイルへ Op ごとの節で追記する前提。

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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|a| a.to_bits()).collect()
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
    let analytic = analytic_hessian(build, x0, shape);
    let numeric = finite_diff_hessian(build, x0, shape, 0.05);
    assert_hessian_close(&analytic, &numeric, ctx);
    // 二階が自明に 0 の検査を弾く（H ≠ 0）。
    assert!(
        analytic.iter().flatten().any(|v| v.abs() > 1e-3),
        "{ctx}: Hessian が全ゼロ（検査が空振り）"
    );
}

/// 値が重複せず最小間隔が 0.3 の正の入力（`x⊙x` の argmax／argmin もタイにならない）。
fn distinct_pos(n: usize) -> Vec<f32> {
    let p = [29usize, 53, 101, 211, 401, 809]
        .into_iter()
        .find(|&p| p > n)
        .unwrap();
    (0..n)
        .map(|i| 1.0 + ((i * 37 + 5) % p) as f32 * 0.3)
        .collect()
}

/// `sum(y ⊙ y ⊙ c)`（`c` は y と同 shape の非一様な定数）。
fn quad<'t>(tape: &'t Tape, y: &Var<'t>) -> BuildResult<'t> {
    let shape = shp(y);
    let c = tape.var_no_grad(&t(common::det_pos(flat_len(&shape).max(1), 0.7), &shape));
    y.mul(y)?.mul(&c)?.sum(None)
}

// ---------------------------------------------------------------------------------
// Max／Min（V2）
// ---------------------------------------------------------------------------------

#[test]
fn max_min_hessian_matches_finite_difference() {
    let shape = [2usize, 3, 4];
    let x0 = distinct_pos(flat_len(&shape));
    hessian_case("max(x*x) None", |_, x| x.mul(x)?.max(None), &x0, &shape);
    hessian_case("min(x*x) None", |_, x| x.mul(x)?.min(None), &x0, &shape);
    hessian_case(
        "max(x)^2 None",
        |_, x| {
            let m = x.max(None)?;
            m.mul(&m)
        },
        &x0,
        &shape,
    );
    hessian_case(
        "min(x)^2 None",
        |_, x| {
            let m = x.min(None)?;
            m.mul(&m)
        },
        &x0,
        &shape,
    );
    hessian_case(
        "max axis1",
        |tp, x| quad(tp, &x.mul(x)?.max(Some(1))?),
        &x0,
        &shape,
    );
    hessian_case(
        "min axis1",
        |tp, x| quad(tp, &x.mul(x)?.min(Some(1))?),
        &x0,
        &shape,
    );
    hessian_case("max axis0", |tp, x| quad(tp, &x.max(Some(0))?), &x0, &shape);
}

#[test]
fn max_hessian_matches_closed_form() {
    // loss = max(x⊙x): 選ばれた位置 i* の対角だけが 2。
    let shape = [5usize];
    let x0 = [1.0f32, 4.0, 2.0, 3.0, 1.5];
    let h = analytic_hessian(|_, x| x.mul(x)?.max(None), &x0, &shape);
    for j in 0..5 {
        for i in 0..5 {
            let want = if i == 1 && j == 1 { 2.0 } else { 0.0 };
            assert!(common::req2_close(h[j][i], want), "h[{j}][{i}]={}", h[j][i]);
        }
    }
    // loss = max(x)^2: 2 e_i e_iᵀ（m の勾配が選択位置のみ）。
    let h = analytic_hessian(
        |_, x| {
            let m = x.max(None)?;
            m.mul(&m)
        },
        &x0,
        &shape,
    );
    for j in 0..5 {
        for i in 0..5 {
            let want = if i == 1 && j == 1 { 2.0 } else { 0.0 };
            assert!(common::req2_close(h[j][i], want), "h[{j}][{i}]={}", h[j][i]);
        }
    }
}

#[test]
fn max_dims_keepdim_is_accepted() {
    let shape = [2usize, 3, 4];
    let x0 = distinct_pos(flat_len(&shape));
    hessian_case(
        "max_dims [0,2] keepdim",
        |tp, x| quad(tp, &x.mul(x)?.max_dims(&[0, 2], true)?),
        &x0,
        &shape,
    );
}

// ---------------------------------------------------------------------------------
// Amax／Amin（V3）
// ---------------------------------------------------------------------------------

#[test]
fn amax_amin_hessian_matches_finite_difference() {
    let shape = [2usize, 3, 4];
    let x0 = distinct_pos(flat_len(&shape));
    hessian_case(
        "amax(x)^2 None",
        |_, x| {
            let m = x.amax(None)?;
            m.mul(&m)
        },
        &x0,
        &shape,
    );
    hessian_case("amin(x*x) None", |_, x| x.mul(x)?.amin(None), &x0, &shape);
    hessian_case(
        "amax axis2",
        |tp, x| quad(tp, &x.mul(x)?.amax(Some(2))?),
        &x0,
        &shape,
    );
    hessian_case(
        "amin axis1",
        |tp, x| quad(tp, &x.amin(Some(1))?),
        &x0,
        &shape,
    );
}

#[test]
fn amax_tie_second_order_matches_closed_form() {
    // x = [3, 3, 1, 3]、S = {0, 1, 3}、k = 3。loss = amax(x)^2 の Hessian は (2/k²)·1_S 1_Sᵀ。
    let shape = [4usize];
    let x0 = [3.0f32, 3.0, 1.0, 3.0];
    let h = analytic_hessian(
        |_, x| {
            let m = x.amax(None)?;
            m.mul(&m)
        },
        &x0,
        &shape,
    );
    let in_s = |i: usize| i != 2;
    for j in 0..4 {
        for i in 0..4 {
            let want = if in_s(i) && in_s(j) { 2.0 / 9.0 } else { 0.0 };
            assert!(common::req2_close(h[j][i], want), "h[{j}][{i}]={}", h[j][i]);
        }
    }
    // amin も同型（S = {0, 3}、k = 2）。
    let x1 = [1.0f32, 5.0, 4.0, 1.0];
    let h = analytic_hessian(
        |_, x| {
            let m = x.amin(None)?;
            m.mul(&m)
        },
        &x1,
        &shape,
    );
    let in_s = |i: usize| i == 0 || i == 3;
    for j in 0..4 {
        for i in 0..4 {
            let want = if in_s(i) && in_s(j) { 0.5 } else { 0.0 };
            assert!(
                common::req2_close(h[j][i], want),
                "amin h[{j}][{i}]={}",
                h[j][i]
            );
        }
    }
}

// ---------------------------------------------------------------------------------
// 1 階との bit 一致（既定経路の不変と cg.grad の契約）
// ---------------------------------------------------------------------------------

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
    assert_eq!(bits(&first), bits(&want), "{ctx}: first_order()");
    let got = dense(cg.grad(&x).unwrap().expect("x は到達").to_tensor());
    assert_eq!(bits(&got), bits(&want), "{ctx}: cg.grad");
}

type Reducer = for<'t> fn(&Var<'t>, Option<usize>) -> BuildResult<'t>;

fn reducers() -> Vec<(&'static str, Reducer)> {
    vec![
        ("max", |x, d| x.max(d)),
        ("min", |x, d| x.min(d)),
        ("amax", |x, d| x.amax(d)),
        ("amin", |x, d| x.amin(d)),
    ]
}

#[test]
fn first_order_and_cg_grad_are_bit_identical() {
    let shape = [2usize, 3, 4];
    let x0 = distinct_pos(flat_len(&shape));
    // タイあり入力（先勝ち／均等分配の差が出る）。
    let tied: Vec<f32> = (0..24).map(|i| ((i * 7) % 3) as f32).collect();
    for (name, r) in reducers() {
        for dim in [None, Some(0), Some(1), Some(2)] {
            for (label, input) in [("distinct", &x0), ("tied", &tied)] {
                assert_first_order_bit_identical(
                    &format!("{name} {dim:?} {label}"),
                    |tp, x| {
                        let y = r(x, dim)?;
                        let c = tp.var_no_grad(&t(
                            common::det_pos(flat_len(&shp(&y)).max(1), 0.3),
                            &shp(&y),
                        ));
                        y.mul(&c)?.sum(None)
                    },
                    input,
                    &shape,
                );
            }
        }
    }
}

#[test]
fn tie_nan_signed_zero_regressions() {
    // 符号付きゼロ: [0.0, -0.0] は `==` で一致し先頭が選ばれる。
    for (name, r) in reducers() {
        assert_first_order_bit_identical(
            &format!("{name} signed zero"),
            |_, x| r(x, None)?.mul(&r(x, None)?),
            &[0.0, -0.0, 0.0, -0.0],
            &[4],
        );
    }
    // NaN を含む lane: Max／Min の勾配はゼロ（全不一致）。cg.grad も 1 階と bit 一致で NaN 伝播しない。
    for (name, r) in reducers() {
        assert_first_order_bit_identical(
            &format!("{name} nan"),
            |_, x| r(x, Some(1))?.sum(None),
            &[f32::NAN, 1.0, 2.0, 2.0, 3.0, 1.0],
            &[2, 3],
        );
    }
    // 上流 3e38 でも 1 階・cg.grad が bit 一致し有限（where_cond は inf*0 を作らない）。
    for (name, r) in reducers() {
        assert_first_order_bit_identical(
            &format!("{name} huge upstream"),
            |tp, x| {
                let y = r(x, Some(1))?;
                let c = tp.var_no_grad(&t(vec![3.0e38, 3.0e38], &[2]));
                y.mul(&c)?.sum(None)
            },
            &[1.0, 2.0, 2.0, 5.0, 5.0, 0.0],
            &[2, 3],
        );
    }
}

#[test]
fn amax_k_zero_lane_second_order_is_not_nan() {
    // NaN lane は `out_value` が NaN で一致数 k = 0。二階が NaN にならない（k_safe = 1 で 0/1）。
    let shape = [2usize, 2];
    let x0 = [f32::NAN, 1.0, 2.0, 4.0];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), &shape));
    // x⊙x の項で gx を x 依存にする（amax 単独の gx は定数マスクで二階が追跡されない）。
    let loss = x
        .amax(Some(1))
        .unwrap()
        .sum(None)
        .unwrap()
        .add(&x.mul(&x).unwrap().sum(None).unwrap())
        .unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gx = cg.grad(&x).unwrap().unwrap();
    let s = gx.sum(None).unwrap();
    let grads = child.backward(&s).unwrap();
    let cx = cg.child_var(&x).unwrap().unwrap();
    let h = dense(grads.get(&cx).unwrap().expect("x へ到達"));
    // d/dx Σ(gx) = 2（x⊙x 由来）。amax 構造は寄与 0 で NaN を混ぜない。
    assert!(h.iter().all(|v| *v == 2.0), "二階は 2: {h:?}");
}

// ---------------------------------------------------------------------------------
// 写しの Op タグ・孫テープ
// ---------------------------------------------------------------------------------

#[test]
fn replayed_mirror_records_same_op_tags() {
    // タイ入力。amax の写しが `Max`（先勝ち VJP）で記録されると孫の 2 階が変わる。
    // loss = amax(x)^2 の 2 階 Hessian（閉形式 (2/k²)·1_S 1_Sᵀ）が、孫を経由しても一致する。
    let shape = [3usize];
    let x0 = [2.0f32, 2.0, 1.0];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let grand = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), &shape));
    let m = x.amax(None).unwrap();
    let loss = m.mul(&m).unwrap();
    let tangent = t(vec![1.0, 0.0, 0.0], &shape);
    let res = fandhe_ai_autodiff::functional_ops::jvp_create_graph(
        &tape, &loss, &x, &tangent, &child, &grand,
    )
    .unwrap();
    // jvp = d loss / dx · v = 2m · (1_S/k)·v = 2·2·(1/2)·1 = 2。
    let jv = res.output().value().get(&[]).unwrap() as f64;
    assert!(common::req2_close(jv, 2.0), "jvp={jv}");
}

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
    .expect("選択縮約を含む関数でも jvp_create_graph が成立する");
    let jv = res.output().value().get(&[]).unwrap() as f64;
    assert!(
        common::req2_close(jv, numeric),
        "{ctx}: jvp={jv} numeric={numeric}"
    );
}

#[test]
fn jvp_create_graph_grandchild_matches_independent_reference() {
    let shape = [2usize, 3, 4];
    let x = distinct_pos(flat_len(&shape));
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
                let m = r(x, None)?;
                m.mul(&m)
            },
            &x,
            &shape,
        );
    }
}

// ---------------------------------------------------------------------------------
// fail-closed: 隣接する未対応 Op は拒否のまま
// ---------------------------------------------------------------------------------

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
fn adjacent_fail_closed_ops_stay_rejected() {
    let x0 = distinct_pos(6);
    // 選択縮約を含んでいても、隣接する未対応 Op が 1 つでも祖先にあれば全体を拒否する。
    assert_rejected("max + selu", |_, x| x.selu()?.max(None), &x0, &[2, 3]);
    assert_rejected(
        "amax + selu",
        |_, x| x.selu()?.amax(Some(1))?.sum(None),
        &x0,
        &[2, 3],
    );
    // 現 base で `false` のままの隣接 Op（#3790 / #3791 の担当）。先にそれらが対象化された
    // 場合は、その時点で `false` のままの Op へ差し替える。
    assert_rejected("median", |_, x| x.median(None), &x0, &[2, 3]);
    assert_rejected("nansum", |_, x| x.nansum(None), &x0, &[2, 3]);
}
