//! 統計縮約（`Var`・`Std`・`VectorNorm`〈L1／L2〉・`PNorm`）の `create_graph` 対応（イシュー #3791・
//! 親 #3725・決定記録 `docs/autodiff-higher-order-grad-decision.md` §33・§52）の受け入れ条件を
//! 確かめる統合テスト。
//!
//! - 1 階が素の `Tape::backward` と bit 同一、`cg.grad` も CPU では bit 一致（V3）。
//! - Hessian を独立な 5 点中央差分（1 階解析勾配の差分。h=1e-2）と REQ-2 統一複合判定で突合する。
//! - 必須ベクトルが 3 段（1 階・`cg.grad`/HVP・孫テープ再生）で有限、または総括判定で拒否されて
//!   `child` が空に保たれる。
//! - 分母ゼロのレーン・NaN 伝播・L1 の ±inf 受理・入口と遅い拒否（再試行可）・全経路の掛け合わせ。
//! - 隣接する未対応 Op（`Op::Custom`）との合成が fail-closed のままであること。
//!
//! GPU 実機の parity は `crates/facade/tests/create_graph_gpu_parity.rs`（`#[ignore]`・未実測）。

#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;

use fandhe_ai_autodiff::{AutodiffError, CustomFunction, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

type Built<'t> = Result<Var<'t>, AutodiffError>;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn dense(v: &Tensor<f32>) -> Vec<f32> {
    v.contiguous().as_slice().expect("contiguous").to_vec()
}

fn dense_ref(v: &Tensor<f32>) -> Vec<f32> {
    dense(v)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|a| a.to_bits()).collect()
}

fn pattern(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + phase) * 0.9).sin() + 0.35 * (i as f32 + 1.0).cos() + 0.2)
        .collect()
}

/// 0 から離れた符号混在の決定的な入力（`p<1` の `|x|^p` が滑らかな範囲）。
fn away_from_zero(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let m = 0.6 + 0.37 * ((i * 7) % 5) as f32;
            if i % 3 == 1 { -m } else { m }
        })
        .collect()
}

/// 検査対象 Op の指定。
#[derive(Clone, Copy, Debug)]
enum Spec {
    Var(Option<usize>, usize),
    Std(Option<usize>, usize),
    L1(Option<usize>),
    L2(Option<usize>),
    PNorm(f32, Option<usize>),
}

impl Spec {
    fn apply<'t>(&self, x: &Var<'t>) -> Built<'t> {
        match *self {
            Spec::Var(d, c) => x.var(d, c),
            Spec::Std(d, c) => x.std(d, c),
            Spec::L1(d) => x.norm_l1(d),
            Spec::L2(d) => x.norm_l2(d),
            Spec::PNorm(p, d) => x.norm_p(p, d),
        }
    }
}

/// 縮約結果 `y` を `Σ y² ⊙ c`（`c` は非一様な定数）にして上流 `g = 2yc` を非自明にする。
fn quad<'t>(tape: &'t Tape, y: &Var<'t>) -> Built<'t> {
    let shape = y.value().shape().to_vec();
    // 係数を 0.1 倍にして 1 階勾配の絶対値を抑える（f32 の 1 階勾配の丸め雑音が、Hessian の
    // ゼロ成分を突合する REQ-2 の絶対許容 1e-5 を超えないようにする。判定式・許容は不変）。
    let c = tape.var_no_grad(&t(
        pattern(numel(&shape).max(1), 2.0)[..numel(&shape)]
            .iter()
            .map(|v| v * 0.1)
            .collect(),
        &shape,
    ));
    y.mul(y)?.mul(&c)?.sum(None)
}

fn specs_for(shape: &[usize]) -> Vec<Spec> {
    let mut dims: Vec<Option<usize>> = vec![None];
    dims.extend((0..shape.len()).map(Some));
    let mut out = Vec::new();
    for d in dims {
        for c in [0usize, 1] {
            out.push(Spec::Var(d, c));
            out.push(Spec::Std(d, c));
        }
        out.push(Spec::L1(d));
        out.push(Spec::L2(d));
        for p in [0.5f32, 1.0, 1.5, 2.0, 3.0] {
            out.push(Spec::PNorm(p, d));
        }
    }
    out
}

const SHAPES: [&[usize]; 3] = [&[5], &[3, 4], &[2, 3, 4]];

// ---------------------------------------------------------------------------------
// 1 階の bit 同一・cg.grad の bit 一致
// ---------------------------------------------------------------------------------

#[test]
fn stat_varnorm_first_order_bit_identical_and_cgrad_bit_exact() {
    for shape in SHAPES {
        let x0 = away_from_zero(numel(shape));
        for spec in specs_for(shape) {
            let ctx = format!("{spec:?} {shape:?}");
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), shape));
            let loss = quad(&tape, &spec.apply(&x).unwrap()).unwrap();
            let plain = dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
            let cg = tape
                .backward_create_graph(&loss, &child)
                .unwrap_or_else(|e| panic!("{ctx}: backward_create_graph 失敗 {e:?}"));
            let first = dense(cg.first_order().get(&x).unwrap().unwrap());
            assert_eq!(bits(&first), bits(&plain), "{ctx}: first_order");
            let got = dense(&cg.grad(&x).unwrap().unwrap().to_tensor());
            assert_eq!(
                bits(&got),
                bits(&plain),
                "{ctx}: cg.grad (V3 は CPU で bit 一致)"
            );
        }
    }
}

// ---------------------------------------------------------------------------------
// Hessian の有限差分突合
// ---------------------------------------------------------------------------------

const FD_STEP: f64 = 3e-2;

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

fn finite_diff_hessian(spec: Spec, x0: &[f32], shape: &[usize]) -> Vec<Vec<f64>> {
    let n = numel(shape);
    let grad_at = |x: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), shape));
        let loss = quad(&tape, &spec.apply(&xv).unwrap()).unwrap();
        let grads = tape.backward(&loss).unwrap();
        dense(grads.get(&xv).unwrap().unwrap())
            .into_iter()
            .map(f64::from)
            .collect()
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

fn analytic_hessian(spec: Spec, x0: &[f32], shape: &[usize]) -> Vec<Vec<f64>> {
    let n = numel(shape);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = quad(&tape, &spec.apply(&x).unwrap()).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gx = cg.grad(&x).unwrap().unwrap();
    let cx = cg.child_var(&x).unwrap().unwrap();
    let mut hessian = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = extract_scalar(&gx, shape, j).unwrap();
        let row_grads = child.backward(&gj).unwrap();
        let row = dense(row_grads.get(&cx).unwrap().expect("2 階勾配は x へ到達"));
        for i in 0..n {
            hessian[j][i] = f64::from(row[i]);
        }
    }
    hessian
}

#[test]
fn stat_varnorm_hessian_matches_finite_difference() {
    for shape in SHAPES {
        let x0 = away_from_zero(numel(shape));
        for spec in specs_for(shape) {
            let ctx = format!("{spec:?} {shape:?}");
            let analytic = analytic_hessian(spec, &x0, shape);
            let numeric = finite_diff_hessian(spec, &x0, shape);
            for (j, (ra, rn)) in analytic.iter().zip(&numeric).enumerate() {
                for (i, (&a, &b)) in ra.iter().zip(rn).enumerate() {
                    assert!(
                        common::req2_close(a, b),
                        "{ctx}: hessian[{j}][{i}] analytic={a} numeric={b}"
                    );
                }
            }
            assert!(
                analytic.iter().flatten().any(|v| v.abs() > 1e-4),
                "{ctx}: Hessian が全ゼロ（検査が空振り）"
            );
        }
    }
}

// ---------------------------------------------------------------------------------
// 3 段（1 階・cg.grad/HVP・孫テープ）で有限、または拒否で child が空
// ---------------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Accepted,
    Rejected,
    /// 1 階自体が非有限（対象外）
    FirstOrderNonFinite,
}

fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|a| a.is_finite())
}

/// `loss = Σ (spec(x) ⊙ gscale)`（上流 g = gscale）で 3 段を通す。`gscale` は `y` と同 shape。
fn three_stage(
    ctx: &str,
    spec: Spec,
    x0: &[f32],
    shape: &[usize],
    gscale: Option<&[f32]>,
) -> Outcome {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let y = spec.apply(&x).unwrap();
    let loss = match gscale {
        Some(gs) => {
            let ys = y.value().shape().to_vec();
            let c = tape.var_no_grad(&t(gs.to_vec(), &ys));
            y.mul(&c).unwrap().sum(None).unwrap()
        }
        None => y.sum(None).unwrap(),
    };
    let plain = dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
    if !all_finite(&plain) && x0.iter().all(|v| v.is_finite()) {
        return Outcome::FirstOrderNonFinite;
    }
    let len = tape.len();
    let cg = match tape.backward_create_graph(&loss, &child) {
        Ok(cg) => cg,
        Err(e) => {
            assert!(matches!(e, AutodiffError::Backward(_)), "{ctx}: {e:?}");
            assert!(child.is_empty(), "{ctx}: 拒否後に child が空でない");
            assert_eq!(tape.len(), len, "{ctx}: 親テープが変化した");
            return Outcome::Rejected;
        }
    };
    let first = dense(cg.first_order().get(&x).unwrap().unwrap());
    assert_eq!(bits(&first), bits(&plain), "{ctx}: first_order");
    let gx = cg.grad(&x).unwrap().unwrap();
    let gx_v = dense(&gx.to_tensor());
    assert_eq!(bits(&gx_v), bits(&plain), "{ctx}: cg.grad");
    let cx = cg.child_var(&x).unwrap().unwrap();
    let v = pattern(numel(shape), 9.0);
    let vc = child.var_no_grad(&t(v, shape));
    let prod = gx.mul(&vc).unwrap().sum(None).unwrap();
    // L1 の cg は x にも g にも依存しない定数（g が定数葉のとき）ので、2 階の起点が無い場合がある。
    let Ok(hv_grads) = child.backward(&prod) else {
        return Outcome::Accepted;
    };
    let hv = hv_grads
        .get(&cx)
        .unwrap()
        .map(dense_ref)
        .unwrap_or_else(|| vec![0.0; numel(shape)]);
    assert!(all_finite(&hv), "{ctx}: HVP が非有限 {hv:?}");
    let grand = Tape::new_with_ops(common::naive_ops());
    match child.backward_create_graph(&prod, &grand) {
        Ok(cg2) => {
            let hv_grand = dense(&cg2.grad(&cx).unwrap().unwrap().to_tensor());
            assert!(all_finite(&hv_grand), "{ctx}: 孫テープ再生が非有限");
            for (i, (&a, &b)) in hv_grand.iter().zip(&hv).enumerate() {
                assert!(
                    common::req2_close(f64::from(a), f64::from(b)),
                    "{ctx}: 孫 vs child HVP [{i}] {a} vs {b}"
                );
            }
        }
        Err(e) => {
            assert!(matches!(e, AutodiffError::Backward(_)), "{ctx}: 孫 {e:?}");
            assert!(grand.is_empty(), "{ctx}: 孫の拒否後に grand が空でない");
        }
    }
    Outcome::Accepted
}

fn all_specs_2d(lanes_dim: usize) -> Vec<Spec> {
    let d = Some(lanes_dim);
    vec![
        Spec::Var(d, 0),
        Spec::Var(d, 1),
        Spec::Std(d, 0),
        Spec::Std(d, 1),
        Spec::L1(d),
        Spec::L2(d),
        Spec::PNorm(0.5, d),
        Spec::PNorm(1.5, d),
        Spec::PNorm(3.0, d),
    ]
}

#[test]
fn stat_varnorm_mandatory_vectors_finite_in_three_stages() {
    let vectors: Vec<(&str, Vec<f32>)> = vec![
        ("x=1e20", vec![1e20, 1e20, 1e20]),
        ("[-1e20,1e20]", vec![-1e20, 1e20]),
        ("[16777216,16777218]", vec![16777216.0, 16777218.0]),
        ("[1e-30,1e-30]", vec![1e-30, 1e-30]),
        ("denormal", vec![1e-40, 3e-40, 2e-40]),
        ("constant lane", vec![2.0, 2.0, 2.0]),
        ("all zero", vec![0.0, 0.0, 0.0]),
        ("zero element p<1", vec![0.0, 1.5, -2.0]),
        ("moderate", vec![0.5, -1.25, 2.0, 0.75]),
    ];
    let mut accepted = 0usize;
    for (label, row) in &vectors {
        let shape = [1usize, row.len()];
        for spec in all_specs_2d(1) {
            for (gl, gs) in [
                ("g=1", None),
                ("g=3e38", Some(vec![3e38f32])),
                ("g=0.5", Some(vec![0.5f32])),
            ] {
                let ctx = format!("{label}/{spec:?}/{gl}");
                if three_stage(&ctx, spec, row, &shape, gs.as_deref()) == Outcome::Accepted {
                    accepted += 1;
                }
            }
        }
    }
    assert!(accepted > 0, "受理された経路が 1 つも無い（検査が空振り）");
    // g=[3e38,-3e38]（2 レーン）。
    let shape = [2usize, 3];
    let x0 = [1.0f32, 2.0, 4.0, -1.0, 0.5, 3.0];
    for spec in all_specs_2d(1) {
        let ctx = format!("two lanes/{spec:?}");
        three_stage(&ctx, spec, &x0, &shape, Some(&[3e38, -3e38]));
    }
}

// ---------------------------------------------------------------------------------
// 分母ゼロのレーン（std=0・‖x‖=0・p<1 で x_i=0）
// ---------------------------------------------------------------------------------

#[test]
fn stat_varnorm_zero_denominator_lanes() {
    // 2 レーン × 3 要素。レーン 0 は縮退（定数／全ゼロ）、レーン 1 は通常。
    let shape = [2usize, 3];
    let cases: Vec<(Spec, Vec<f32>)> = vec![
        (Spec::Std(Some(1), 0), vec![2.0, 2.0, 2.0, 1.0, 2.0, 4.0]),
        (Spec::Std(Some(1), 1), vec![-3.0, -3.0, -3.0, 1.0, 2.0, 4.0]),
        (Spec::L2(Some(1)), vec![0.0, 0.0, 0.0, 1.0, 2.0, 4.0]),
        (
            Spec::PNorm(0.5, Some(1)),
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 4.0],
        ),
        (
            Spec::PNorm(0.5, Some(1)),
            vec![0.0, 1.5, 2.5, 1.0, 2.0, 4.0],
        ),
        (
            Spec::PNorm(3.0, Some(1)),
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 4.0],
        ),
        (Spec::Var(Some(1), 0), vec![5.0, 5.0, 5.0, 1.0, 2.0, 4.0]),
    ];
    for (spec, x0) in cases {
        let ctx = format!("{spec:?} {x0:?}");
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0.clone(), &shape));
        let loss = quad(&tape, &spec.apply(&x).unwrap()).unwrap();
        let plain = dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
        let cg = tape
            .backward_create_graph(&loss, &child)
            .unwrap_or_else(|e| panic!("{ctx}: 拒否された {e:?}"));
        let gx = cg.grad(&x).unwrap().unwrap();
        assert_eq!(
            bits(&dense(&gx.to_tensor())),
            bits(&plain),
            "{ctx}: cg.grad"
        );
        let cx = cg.child_var(&x).unwrap().unwrap();
        let w = child.var_no_grad(&t(pattern(6, 3.0), &shape));
        let prod = gx.mul(&w).unwrap().sum(None).unwrap();
        let hv = dense(child.backward(&prod).unwrap().get(&cx).unwrap().unwrap());
        assert!(all_finite(&hv), "{ctx}: HVP に NaN／inf {hv:?}");
    }
}

// ---------------------------------------------------------------------------------
// 拒否（入口・遅い拒否）と再試行
// ---------------------------------------------------------------------------------

fn attempt(
    spec: Spec,
    x0: &[f32],
    shape: &[usize],
    child: &Tape,
    upstream: f32,
) -> Result<(), AutodiffError> {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let y = spec.apply(&x).unwrap();
    let ys = y.value().shape().to_vec();
    let c = tape.var_no_grad(&t(vec![upstream; numel(&ys)], &ys));
    let loss = y.mul(&c).unwrap().sum(None).unwrap();
    let len = tape.len();
    let res = tape.backward_create_graph(&loss, child).map(|_| ());
    assert_eq!(tape.len(), len, "親テープが変化した");
    res
}

#[test]
fn stat_varnorm_unsafe_inputs_rejected_child_empty_and_retry() {
    let safe = [0.5f32, -1.25, 2.0, 0.75, 1.5, -0.5];
    let shape = [2usize, 3];
    let mut with_inf = safe;
    with_inf[1] = f32::INFINITY;
    let mut with_neg_inf = safe;
    with_neg_inf[4] = f32::NEG_INFINITY;
    let long: Vec<f32> = (0..33).map(|i| 1.0 + i as f32 / 33.0).collect();
    type Case = (&'static str, Spec, Vec<f32>, Vec<usize>, f32);
    let mut cases: Vec<Case> = Vec::new();
    // 入口の拒否: ±inf の x（NaN は通す）。
    for spec in [
        Spec::Var(Some(1), 0),
        Spec::Std(Some(1), 0),
        Spec::L2(Some(1)),
        Spec::PNorm(3.0, Some(1)),
    ] {
        cases.push(("+inf", spec, with_inf.to_vec(), shape.to_vec(), 1.0));
        cases.push(("-inf", spec, with_neg_inf.to_vec(), shape.to_vec(), 1.0));
    }
    // 遅い拒否: 極小の p × 長いレーン、非有限の上流、Var の部分和 overflow 域。
    cases.push((
        "p=0.01 long lane",
        Spec::PNorm(0.01, Some(1)),
        long.clone(),
        vec![1, 33],
        1.0,
    ));
    cases.push((
        "g=inf",
        Spec::Std(Some(1), 0),
        safe.to_vec(),
        shape.to_vec(),
        f32::INFINITY,
    ));
    cases.push((
        "g=nan",
        Spec::L2(Some(1)),
        safe.to_vec(),
        shape.to_vec(),
        f32::NAN,
    ));
    cases.push((
        "var mean overflow",
        Spec::Var(Some(1), 0),
        vec![-1e37, 1e37],
        vec![1, 2],
        1.0,
    ));
    for (label, spec, x0, sh, up) in cases {
        let child = Tape::new_with_ops(common::naive_ops());
        let err = attempt(spec, &x0, &sh, &child, up).expect_err(label);
        assert!(
            matches!(err, AutodiffError::Backward(_)),
            "{label}: {err:?}"
        );
        assert!(child.is_empty(), "{label}: 拒否後に child が空でない");
        // 同じ child で安全な入力を再試行できる。
        attempt(Spec::Std(Some(1), 0), &safe, &shape, &child, 1.0)
            .unwrap_or_else(|e| panic!("{label}: 再試行失敗 {e:?}"));
    }
}

// ---------------------------------------------------------------------------------
// 全経路の掛け合わせ: 有限か拒否のどちらか（両方が出る）
// ---------------------------------------------------------------------------------

#[test]
fn stat_varnorm_sweep_every_structure_path_is_finite_or_rejected() {
    let lanes: Vec<Vec<f32>> = vec![
        vec![0.5, -1.25, 2.0, 0.75],
        vec![1e3, -2e3, 5e2, 1.0],
        vec![1e-3, 2e-3, -1e-3, 4e-3],
        vec![1e19, -1e19, 3e19, 2e19],
        vec![1e-20, 5e-21, -3e-20, 1e-20],
        vec![1e30, 1.0, -1e30, 2.0],
        vec![1e-38, 2e-38, 1e-39, 5e-39],
        vec![2.0, 2.0, 2.0, 2.0],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![16777216.0, 16777218.0, 16777216.0, 16777220.0],
        vec![3e38, 1.0, 2.0, 3.0],
    ];
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    for row in &lanes {
        let shape = [1usize, row.len()];
        for spec in all_specs_2d(1) {
            for gs in [
                None,
                Some(vec![1e-3f32]),
                Some(vec![1e6f32]),
                Some(vec![3e38f32]),
            ] {
                let ctx = format!("{row:?}/{spec:?}/{gs:?}");
                match three_stage(&ctx, spec, row, &shape, gs.as_deref()) {
                    Outcome::Accepted => accepted += 1,
                    Outcome::Rejected => rejected += 1,
                    Outcome::FirstOrderNonFinite => {}
                }
            }
        }
    }
    assert!(
        accepted > 0 && rejected > 0,
        "accepted={accepted} rejected={rejected}"
    );
}

// ---------------------------------------------------------------------------------
// NaN の伝播と L1 の ±inf 受理
// ---------------------------------------------------------------------------------

#[test]
fn stat_varnorm_nan_propagates_and_l1_accepts_inf() {
    let shape = [2usize, 3];
    // レーン 0 に NaN。NaN レーンだけが NaN になり、レーン 1 は有限で 1 階と bit 一致する。
    let x0 = [1.0f32, f32::NAN, 3.0, 1.0, 2.0, 4.0];
    for spec in [
        Spec::Var(Some(1), 1),
        Spec::Std(Some(1), 1),
        Spec::L2(Some(1)),
        Spec::PNorm(3.0, Some(1)),
    ] {
        let ctx = format!("{spec:?}");
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0.to_vec(), &shape));
        let loss = spec.apply(&x).unwrap().sum(None).unwrap();
        let plain = dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
        let cg = tape
            .backward_create_graph(&loss, &child)
            .unwrap_or_else(|e| panic!("{ctx}: NaN 入力が拒否された {e:?}"));
        let gx = cg.grad(&x).unwrap().unwrap();
        let got = dense(&gx.to_tensor());
        for i in 0..6 {
            assert!(
                got[i].to_bits() == plain[i].to_bits() || (got[i].is_nan() && plain[i].is_nan()),
                "{ctx}: cg.grad[{i}]={} plain={}",
                got[i],
                plain[i]
            );
        }
        assert!(
            got[..3].iter().any(|v| v.is_nan()),
            "{ctx}: NaN レーンが NaN にならない"
        );
        assert!(
            all_finite(&got[3..]),
            "{ctx}: 隣のレーンへ NaN が漏れた {got:?}"
        );
        // 2 階: 隣のレーンの行は NaN を含まない。
        let cx = cg.child_var(&x).unwrap().unwrap();
        let w = child.var_no_grad(&t(vec![0.0, 0.0, 0.0, 1.0, 0.5, -0.25], &shape));
        let prod = gx.mul(&w).unwrap().sum(None).unwrap();
        let hv = dense(child.backward(&prod).unwrap().get(&cx).unwrap().unwrap());
        assert!(
            all_finite(&hv[3..]),
            "{ctx}: 2 階で NaN が隣のレーンへ漏れた {hv:?}"
        );
    }
    // L1 は ±inf も NaN も受理する（構造が x に触れない 3 分岐）。
    let xi = [
        1.0f32,
        f32::INFINITY,
        -2.0,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0,
    ];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(xi.to_vec(), &shape));
    let loss = x.norm_l1(Some(1)).unwrap().sum(None).unwrap();
    let plain = dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("L1 は ±inf を受理する");
    let got = dense(&cg.grad(&x).unwrap().unwrap().to_tensor());
    assert_eq!(bits(&got), bits(&plain), "L1 cg.grad");
}

// ---------------------------------------------------------------------------------
// 孫テープ再生（写しの Op タグ）
// ---------------------------------------------------------------------------------

#[test]
fn stat_varnorm_grandchild_replay_matches_reference() {
    let shape = [3usize, 4];
    let x0 = away_from_zero(12);
    for spec in specs_for(&shape) {
        let ctx = format!("{spec:?}");
        let tangent = common::det_seq(12, 0.9);
        let reference = {
            let tape = Tape::new_with_ops(common::naive_ops());
            let p = tape.var(&t(x0.clone(), &shape));
            let loss = quad(&tape, &spec.apply(&p).unwrap()).unwrap();
            let g = dense(tape.backward(&loss).unwrap().get(&p).unwrap().unwrap());
            g.iter()
                .zip(&tangent)
                .map(|(a, v)| f64::from(*a) * f64::from(*v))
                .sum::<f64>()
        };
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let p = tape.var(&t(x0.clone(), &shape));
        let loss = quad(&tape, &spec.apply(&p).unwrap()).unwrap();
        let res = fandhe_ai_autodiff::functional_ops::jvp_create_graph(
            &tape,
            &loss,
            &p,
            &t(tangent, &shape),
            &child,
            &grand,
        )
        .unwrap_or_else(|e| panic!("{ctx}: jvp_create_graph 失敗 {e:?}"));
        let jv = res.output().value().get(&[]).unwrap() as f64;
        assert!(
            common::req2_close(jv, reference),
            "{ctx}: jvp={jv} reference={reference}"
        );
    }
}

// ---------------------------------------------------------------------------------
// 隣接する fail-closed（恒久対象外の Op::Custom）
// ---------------------------------------------------------------------------------

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

/// 回帰（PR #3942 Codex P1）: PNorm の f32 微分係数 overflow（x=[1e-30,1]・p=0.5）は拒否して
/// child を空に保ち、ゼロ要素を含む x=[0,0.75]・p=300 は死に枝が NaN を作らず二階が有限。
#[test]
fn pnorm_coefficient_overflow_rejected_and_dead_branch_finite() {
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![1e-30, 1.0], &[2]));
        let loss = x.norm_p(0.5, None).unwrap().sum(None).unwrap();
        let err = tape
            .backward_create_graph(&loss, &child)
            .expect_err("f32 微分係数 overflow は拒否されるはず");
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
        assert!(child.is_empty());
    }
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![0.0, 0.75], &[2]));
        let loss = x.norm_p(300.0, None).unwrap().sum(None).unwrap();
        match tape.backward_create_graph(&loss, &child) {
            Ok(cg) => {
                let g = cg.grad(&x).unwrap().unwrap().to_tensor();
                assert!(dense(&g).iter().all(|v| v.is_finite()));
            }
            Err(e) => {
                assert!(matches!(e, AutodiffError::Backward(_)), "{e:?}");
                assert!(child.is_empty());
            }
        }
    }
}

#[test]
fn stat_varnorm_neighbouring_custom_op_stays_fail_closed() {
    for spec in [
        Spec::Var(None, 0),
        Spec::Std(None, 1),
        Spec::L1(None),
        Spec::L2(None),
        Spec::PNorm(3.0, None),
    ] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(away_from_zero(6), &[6]));
        let y = spec.apply(&x).unwrap().reshape(&[1]).unwrap();
        let c = tape.custom(Arc::new(IdentityCustomFn), &[y]).unwrap();
        let loss = c.sum(None).unwrap();
        let len = tape.len();
        let err = tape
            .backward_create_graph(&loss, &child)
            .expect_err("Op::Custom は fail-closed で拒否されるはず");
        assert!(
            matches!(err, AutodiffError::Backward(_)),
            "{spec:?}: {err:?}"
        );
        assert!(child.is_empty(), "{spec:?}: child が空でない");
        assert_eq!(tape.len(), len, "{spec:?}: 親テープが変化した");
    }
}
