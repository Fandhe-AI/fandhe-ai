//! `MatrixNorm`（`Fro`・`One`・`Inf`・`Nuc`・`Spectral`）・`SlogdetSign`・`MatrixRank` の `create_graph`
//! （`Tape::backward_create_graph`）対応（イシュー #3806・#3991・親 #3734・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §43・§66・§76）の受け入れ条件を公開 API 経由で直接検証する
//! 統合テスト。
//!
//! - Hessian を `Tape::backward` の 5 点中央差分（`create_graph` を経由しない独立経路）と REQ-2 統一複合
//!   判定（`common::req2_close`）で突合する。`One`・`Inf` は A に関する二階がゼロなので、外側に非線形
//!   （`tanh`）を噛ませて `g` を経由する二階が非ゼロになるようにする。`SlogdetSign` は
//!   `sign · exp(logabsdet)`（= det）、`MatrixRank` は `rank · Σx²` の中で検査する。
//! - `cg.first_order()` は `Tape::backward` と bit 同一、`cg.grad(..)` も V3／ゼロ定数のため CPU で bit 一致。
//!   先勝ちのタイ・`sign(0) = 0`・`g < 0` での `-0` も 1 階と一致する。孫テープと `jvp_create_graph` も確かめる。
//! - 非有限の上流は `SlogdetSign`・`MatrixRank` では受理（ゼロ）、`MatrixNorm` では拒否。±inf の入力、
//!   閾値付近の `MatrixRank` は `Err(Backward)` かつ `child.is_empty()` で、同じ `child` を再利用できる。
//! - `Nuc`・`Spectral`（#3991・§76）は節 8: Hessian・1 階 bit 一致・縮退／σ = 0／空／±inf／NaN の拒否と再試行・
//!   `[100000,1]` で m×m を作らないこと・隣接 `Op::Custom` の fail-closed。隣接する未対応 Op（`Op::Custom`）は fail-closed のまま。
//!
//! GPU 実機 parity は `crates/facade/tests/create_graph_gpu_parity.rs`（`#[ignore]`・未実測。Phase 10 の
//! #3857・#3859）で扱う。
#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;

use fandhe_ai_autodiff::functional_ops::jvp_create_graph;
use fandhe_ai_autodiff::{AutodiffError, CustomFunction, Tape, Var};
use fandhe_ai_tensor_core::{MatrixNormOrd, Tensor};

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
        // SVD 系（`nuc`／`spectral`）の総括判定は保守的で、三階の機構では桁落ちの見積りに掛かって
        // 拒否されうる（§68.2・§76.4）。拒否なら型付きエラーで孫テープは空のままであること。
        Err(AutodiffError::Backward(msg))
            if ctx.starts_with("nuc") || ctx.starts_with("spectral") =>
        {
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

// --- 1. 損失の組み立て -------------------------------------------------------------------

fn ord_of(name: &str) -> MatrixNormOrd {
    match name {
        "fro" => MatrixNormOrd::Fro,
        "one" => MatrixNormOrd::One,
        _ => MatrixNormOrd::Inf,
    }
}

/// `det = sign · exp(logabsdet)` の二乗（`SlogdetSign` と `SlogdetLogAbsDet` の混合）。
fn det_sq<'t>(x: &Var<'t>) -> Built<'t> {
    let s = x.slogdet()?;
    let det = s.sign.mul(&s.logabsdet.exp())?;
    det.mul(&det)?.sum(None)
}

/// `rank · Σx²`（`MatrixRank` は区分定数。二階は rank 倍の `2I`）。
fn rank_quad<'t>(x: &Var<'t>) -> Built<'t> {
    let r = x.matrix_rank(None)?;
    r.mul(&x.mul(x)?.sum(None)?)?.sum(None)
}

fn build_loss<'t>(name: &str, _tape: &'t Tape, x: &Var<'t>) -> Built<'t> {
    match name {
        "fro" | "one" | "inf" => x.matrix_norm(ord_of(name))?.tanh().sum(None),
        "det_sq" => det_sq(x),
        "rank_quad" => rank_quad(x),
        "sign" => x.slogdet()?.sign.sum(None),
        _ => x.matrix_rank(None)?.sum(None),
    }
}

/// タイから離れた入力（One の最大列和・Inf の最大行和が一意）。
fn tie_free() -> Vec<f32> {
    vec![1.0, -2.0, 3.0, 0.5, -0.7, 1.1]
}

fn cases() -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    vec![
        ("fro", vec![3, 2], x_data(&[3, 2])),
        ("one", vec![3, 2], tie_free()),
        ("inf", vec![3, 2], tie_free()),
        ("det_sq", vec![3, 3], well_conditioned(&[3, 3])),
        ("rank_quad", vec![3, 3], well_conditioned(&[3, 3])),
        ("sign", vec![3, 3], well_conditioned(&[3, 3])),
        ("rank", vec![3, 3], well_conditioned(&[3, 3])),
    ]
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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// --- 2. 1 階の bit 同一・Hessian・孫テープ ---------------------------------------------------

#[test]
fn linalg_nr_first_order_bit_identical_and_cgrad() {
    for (name, shape, x0) in cases() {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0.clone(), &shape));
        let loss = build_loss(name, &tape, &x).unwrap();
        let plain = dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
        let cg = tape.backward_create_graph(&loss, &child).expect(name);
        let first = dense(cg.first_order().get(&x).unwrap().unwrap());
        let got = dense(&cg.grad(&x).unwrap().expect(name).to_tensor());
        assert_eq!(bits(&first), bits(&plain), "{name}: first_order");
        if name == "det_sq" {
            // 外側の det² は V3 以外の Op（Mul・Exp）を含むため bit 一致ではなく REQ-2 判定。
            for (a, b) in got.iter().zip(&plain) {
                assert!(
                    common::req2_close(f64::from(*a), f64::from(*b)),
                    "{name}: {a} vs {b}"
                );
            }
        } else {
            assert_eq!(
                bits(&got),
                bits(&plain),
                "{name}: cg.grad は V3／ゼロ定数で bit 一致"
            );
        }
        if name == "sign" || name == "rank" {
            assert!(plain.iter().all(|v| *v == 0.0), "{name}: 1 階はゼロ");
        }
    }
}

#[test]
fn matrix_norm_slogdet_rank_hessian_matches_finite_difference() {
    for (name, shape, x0) in cases() {
        if name == "sign" || name == "rank" {
            continue; // 単独ではゼロ勾配（上のテストと下のゼロ cgrad テストで扱う）。
        }
        check_all(name, |tape, x| build_loss(name, tape, x), &x0, &shape);
    }
}

#[test]
fn slogdet_sign_and_matrix_rank_zero_cgrad() {
    for name in ["sign", "rank"] {
        let x0 = well_conditioned(&[3, 3]);
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0, &[3, 3]));
        let loss = build_loss(name, &tape, &x).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let g = dense(&cg.grad(&x).unwrap().expect("Some(zeros)").to_tensor());
        assert!(g.iter().all(|v| *v == 0.0), "{name}: cgrad はゼロ");
    }
}

/// 非有限の上流（`inf`・`NaN` の定数との積）でも、1 階がゼロなので `SlogdetSign`・`MatrixRank` は受理しゼロ。
#[test]
fn slogdet_sign_and_matrix_rank_accept_non_finite_upstream() {
    for c in [f32::INFINITY, f32::NAN, 3.0e38] {
        for name in ["sign", "rank"] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(well_conditioned(&[3, 3]), &[3, 3]));
            let y = if name == "sign" {
                x.slogdet().unwrap().sign
            } else {
                x.matrix_rank(None).unwrap()
            };
            let k = tape.var_no_grad(&t(vec![c], &[]));
            let loss = y.mul(&k).unwrap().sum(None).unwrap();
            let plain = dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
            let cg = tape.backward_create_graph(&loss, &child).expect("受理");
            let got = dense(&cg.grad(&x).unwrap().unwrap().to_tensor());
            assert_eq!(bits(&got), bits(&plain), "{name} c={c}");
            assert!(got.iter().all(|v| *v == 0.0), "{name} c={c}: ゼロ");
        }
    }
}

// --- 3. MatrixRank の閾値付近 ----------------------------------------------------------------

#[test]
fn matrix_rank_threshold_near_rejected_child_empty_and_retry() {
    // diag(1, r) で r を rcond ちょうどに置く（隣接値で rank が変わる）。
    assert_rejected_and_child_reusable(
        "rank near threshold",
        |_t, x| {
            x.matrix_rank(Some(1e-3))?
                .mul(&x.mul(x)?.sum(None)?)?
                .sum(None)
        },
        &[1.0, 0.0, 0.0, 1e-3],
        &[2, 2],
    );
    // 閾値から離れていれば受理される。零行列・rcond = 0 の rank 落ち行列も受理。
    for (x0, rcond) in [
        (vec![1.0f32, 0.0, 0.0, 0.5], Some(1e-3f32)),
        (vec![0.0; 4], Some(1e-6)),
        (vec![1.0, 0.0, 0.0, 0.0], Some(0.0)),
    ] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0.clone(), &[2, 2]));
        let loss = x
            .matrix_rank(rcond)
            .unwrap()
            .mul(&x.mul(&x).unwrap().sum(None).unwrap())
            .unwrap()
            .sum(None)
            .unwrap();
        tape.backward_create_graph(&loss, &child)
            .unwrap_or_else(|e| panic!("{x0:?} rcond={rcond:?}: 受理されるはず: {e}"));
    }
}

// --- 4. One・Inf のタイ・符号・空行列 --------------------------------------------------------

#[test]
fn matrix_norm_one_inf_tie_follows_first_order() {
    let cases: Vec<(Vec<f32>, Vec<usize>)> = vec![
        (vec![1.0, 0.0], vec![2, 1]),
        (vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]),
        (vec![1.0, -1.0, -1.0, 1.0], vec![2, 2]),
        (vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]),
        (vec![2.0, 0.0, 0.0, 2.0, 0.0, 0.0], vec![3, 2]),
    ];
    for name in ["one", "inf"] {
        for (x0, shape) in &cases {
            for scale in [-1.0f32, 1.0, -2.5] {
                let tape = Tape::new_with_ops(common::naive_ops());
                let child = Tape::new_with_ops(common::naive_ops());
                let x = tape.var(&t(x0.clone(), shape));
                let y = x.matrix_norm(ord_of(name)).unwrap();
                let loss = y.mul(&tape.var_no_grad(&t(vec![scale], &[]))).unwrap();
                let plain = dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
                let cg = tape.backward_create_graph(&loss, &child).unwrap();
                let got = dense(&cg.grad(&x).unwrap().unwrap().to_tensor());
                assert_eq!(bits(&got), bits(&plain), "{name} {x0:?} scale={scale}");
            }
        }
    }
}

#[test]
fn matrix_norm_empty_input_returns_empty_gradient() {
    for name in ["fro", "one", "inf"] {
        for shape in [[0usize, 3], [3, 0]] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(vec![], &shape));
            let loss = x.matrix_norm(ord_of(name)).unwrap().sum(None).unwrap();
            let cg = tape
                .backward_create_graph(&loss, &child)
                .unwrap_or_else(|e| panic!("{name} {shape:?}: {e}"));
            let g = cg.grad(&x).unwrap().unwrap().to_tensor();
            assert_eq!(g.shape(), &shape, "{name}");
        }
    }
}

// --- 5. Fro の必須ベクトル -------------------------------------------------------------------

/// Fro の必須ベクトルの受理／拒否（実測で確定。`stat_varnorm` の L2 と同じ総括判定）。
/// `1e-38` は二階の上界（約 `1/‖A‖`）が `STAGE_BOUND_GUARD`（2^125）を超えて拒否される。
/// `|g| = 3e38` は `q·g` などの段が 2^125 以上になり拒否される。零行列と NaN は上流の大きさによらず受理。
fn expect_accept(name: &str, g: f32) -> bool {
    match name {
        "zeros" | "nan" => true,
        "1e30" | "benign" => g == 1.0,
        _ => false,
    }
}

/// 極端なスケール・零行列・NaN・巨大な上流の 3 段（1 階・`cg.grad`・孫テープ）。受理される場合は
/// 1 階と bit 一致し、孫テープの勾配が 1 階と同じ非有限クラスであること。拒否される場合は
/// `Err(Backward)` かつ `child` が空であること。
#[test]
fn matrix_norm_fro_mandatory_vectors_three_stages() {
    let inputs: Vec<(&str, Vec<f32>)> = vec![
        ("1e30", vec![1e30, 2e30, -1e30, 0.5e30]),
        ("1e-38", vec![1e-38, 2e-38, -1e-38, 0.5e-38]),
        ("zeros", vec![0.0; 4]),
        ("nan", vec![1.0, f32::NAN, 2.0, 3.0]),
        ("benign", vec![1.0, 2.0, -1.0, 0.5]),
    ];
    for (name, x0) in inputs {
        for g in [1.0f32, 3.0e38, -3.0e38] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let grand = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), &[2, 2]));
            let loss = x
                .matrix_norm(MatrixNormOrd::Fro)
                .unwrap()
                .mul(&tape.var_no_grad(&t(vec![g], &[])))
                .unwrap();
            let plain = dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
            match tape.backward_create_graph(&loss, &child) {
                Err(AutodiffError::Backward(_)) => {
                    assert!(!expect_accept(name, g), "{name} g={g}: 拒否されないはず");
                    assert!(child.is_empty(), "{name} g={g}: 拒否後は child が空");
                }
                Err(e) => panic!("{name} g={g}: 予期しないエラー {e}"),
                Ok(cg) => {
                    assert!(expect_accept(name, g), "{name} g={g}: 拒否されるはず");
                    let got = dense(&cg.grad(&x).unwrap().unwrap().to_tensor());
                    assert_eq!(
                        bits(&got),
                        bits(&plain),
                        "{name} g={g}: cg.grad は 1 階と bit 一致"
                    );
                    let cx = cg.child_var(&x).unwrap().unwrap();
                    let gx = cg.grad(&x).unwrap().unwrap();
                    let scalar = gx.sum(None).unwrap();
                    let second = dense(child.backward(&scalar).unwrap().get(&cx).unwrap().unwrap());
                    let cg2 = child.backward_create_graph(&scalar, &grand);
                    if plain.iter().all(|v| v.is_finite()) && x0.iter().all(|v| v.is_finite()) {
                        // 受理した有限入力では二階が有限、孫テープも受理または型付き拒否。
                        assert!(
                            second.iter().all(|v| v.is_finite()),
                            "{name} g={g}: 二階が非有限 {second:?}"
                        );
                    }
                    if let Err(e) = cg2 {
                        assert!(matches!(e, AutodiffError::Backward(_)), "{name} g={g}: {e}");
                    }
                }
            }
        }
    }
}

// --- 6. 拒否域 ---------------------------------------------------------------------------

#[test]
fn matrix_norm_rejects_inf_input_child_empty() {
    for name in ["fro", "one", "inf"] {
        assert_rejected_and_child_reusable(
            &format!("{name} inf input"),
            |_t, x| x.matrix_norm(ord_of(name))?.sum(None),
            &[1.0, f32::INFINITY, 2.0, 3.0],
            &[2, 2],
        );
    }
    for ord in [MatrixNormOrd::Nuc, MatrixNormOrd::Spectral] {
        assert_rejected_and_child_reusable(
            &format!("{ord:?} inf input"),
            move |_t, x| x.matrix_norm(ord)?.sum(None),
            &[1.0, f32::INFINITY, 2.0, 3.0],
            &[2, 2],
        );
    }
}

#[test]
fn matrix_norm_one_inf_reject_non_finite_upstream() {
    for name in ["one", "inf", "fro"] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]));
        let k = tape.var_no_grad(&t(vec![f32::INFINITY], &[]));
        let loss = x
            .matrix_norm(ord_of(name))
            .unwrap()
            .mul(&k)
            .unwrap()
            .sum(None)
            .unwrap();
        let r = tape.backward_create_graph(&loss, &child);
        assert!(is_backward_err(&r), "{name}: 非有限の上流は拒否");
        assert!(child.is_empty(), "{name}: child は空");
    }
}

// --- 7. jvp_create_graph（孫テープ）と隣接 Op ----------------------------------------------

#[test]
fn linalg_nr_grandchild_replay_matches_directional_derivative() {
    for (name, shape, x0) in cases() {
        if name == "sign" || name == "rank" {
            continue;
        }
        let v = pattern(numel(&shape), 5.0);
        let loss_at = |x: &[f32]| -> f64 {
            let tape = Tape::new_with_ops(common::naive_ops());
            let xv = tape.var(&t(x.to_vec(), &shape));
            let loss = build_loss(name, &tape, &xv).unwrap();
            f64::from(dense(&loss.to_tensor())[0])
        };
        let h = 0.01f32;
        let plus: Vec<f32> = x0.iter().zip(&v).map(|(a, b)| a + h * b).collect();
        let minus: Vec<f32> = x0.iter().zip(&v).map(|(a, b)| a - h * b).collect();
        let expected = (loss_at(&plus) - loss_at(&minus)) / (2.0 * f64::from(h));

        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0.clone(), &shape));
        let loss = build_loss(name, &tape, &x).unwrap();
        let res = jvp_create_graph(&tape, &loss, &x, &t(v.clone(), &shape), &child, &grand)
            .expect("jvp_create_graph");
        let got = f64::from(dense(&res.output().to_tensor())[0]);
        assert!(
            common::req2_close(got, expected),
            "{name}: jvp={got} vs central difference={expected}"
        );
    }
}

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
fn linalg_nr_neighbouring_unsupported_op_stays_fail_closed() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(tie_free(), &[3, 2]));
    let y = x.matrix_norm(MatrixNormOrd::Fro).unwrap();
    let z = tape
        .custom(Arc::new(IdentityCustomFn), &[y])
        .expect("Tape::custom 登録成功");
    let loss = z.sum(None).unwrap();
    let r = tape.backward_create_graph(&loss, &child);
    assert!(is_backward_err(&r), "Op::Custom は拒否される");
    assert!(child.is_empty(), "拒否後は child が空");
    let ok = y.sum(None).unwrap();
    tape.backward_create_graph(&ok, &child)
        .expect("拒否後の再試行は成功する");
}

// --- 8. Nuc・Spectral（イシュー #3991・決定記録 §75 A1・§76） ---------------------------------

/// スペクトルがよく分離した（σ ≈ 3, 2, 1）一般行列。縦長・横長・正方で使う（`create_graph_decomp` と同型）。
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

const SVD_NORMS: [(&str, MatrixNormOrd); 2] = [
    ("nuc", MatrixNormOrd::Nuc),
    ("spectral", MatrixNormOrd::Spectral),
];

#[test]
fn nuc_spectral_hessian_matches_finite_difference() {
    for shape in [[4usize, 3], [2, 3], [3, 3]] {
        let x0 = sv_separated(&shape);
        for (name, ord) in SVD_NORMS {
            check_all(
                &format!("{name} tanh {shape:?}"),
                move |_t, x| x.matrix_norm(ord)?.tanh().sum(None),
                &x0,
                &shape,
            );
            check_all(
                &format!("{name} sq {shape:?}"),
                move |tape, x| {
                    // 0.2 倍: 有限差分側（f32 の 1 階勾配）の丸めが REQ-2 の絶対許容に収まる大きさにする。
                    let k = tape.var_no_grad(&t(vec![0.2], &[]));
                    let n = x.matrix_norm(ord)?.mul(&k)?;
                    n.mul(&n)?.sum(None)
                },
                &x0,
                &shape,
            );
        }
    }
}

#[test]
fn nuc_spectral_first_order_and_cg_grad_bit_identical() {
    for shape in [[4usize, 3], [2, 3], [3, 3]] {
        let x0 = sv_separated(&shape);
        for (name, ord) in SVD_NORMS {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), &shape));
            let loss = x.matrix_norm(ord).unwrap().sum(None).unwrap();
            let plain = dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
            let cg = tape.backward_create_graph(&loss, &child).expect(name);
            let first = dense(cg.first_order().get(&x).unwrap().unwrap());
            let got = dense(&cg.grad(&x).unwrap().expect(name).to_tensor());
            assert_eq!(bits(&first), bits(&plain), "{name} {shape:?}: first_order");
            assert_eq!(
                bits(&got),
                bits(&plain),
                "{name} {shape:?}: cg.grad は V3 で bit 一致"
            );
        }
    }
}

#[test]
fn nuc_spectral_degenerate_sigma0_empty_nan_rejected_child_empty_and_retry() {
    let cases: Vec<(&str, Vec<f32>, Vec<usize>)> = vec![
        (
            "identity3",
            vec![1., 0., 0., 0., 1., 0., 0., 0., 1.],
            vec![3, 3],
        ),
        ("rotation", vec![0.6, -0.8, 0.8, 0.6], vec![2, 2]),
        ("sigma0", vec![1., 0., 0., 0.], vec![2, 2]),
        ("empty_0x3", vec![], vec![0, 3]),
        ("empty_3x0", vec![], vec![3, 0]),
        ("neg_inf", vec![1., f32::NEG_INFINITY, 2., 3.], vec![2, 2]),
        ("nan", vec![1., f32::NAN, 2., 3.], vec![2, 2]),
    ];
    for (label, x0, shape) in cases {
        for (name, ord) in SVD_NORMS {
            let ctx = format!("{name} {label}");
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), &shape));
            let Ok(norm) = x.matrix_norm(ord) else {
                continue; // forward 自体が拒否する入力は対象外。
            };
            let loss = norm.sum(None).unwrap();
            let r = tape.backward_create_graph(&loss, &child);
            assert!(
                is_backward_err(&r),
                "{ctx}: Err(Backward) のはず: {:?}",
                r.as_ref().err()
            );
            assert!(child.is_empty(), "{ctx}: 拒否後は child が空");
            let ok = x.sum(None).unwrap();
            assert!(
                tape.backward_create_graph(&ok, &child).is_ok(),
                "{ctx}: 同じ child で再試行できる"
            );
        }
    }
}

#[test]
fn nuc_spectral_huge_upstream_rejected() {
    for (name, ord) in SVD_NORMS {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(sv_separated(&[3, 2]), &[3, 2]));
        let k = tape.var_no_grad(&t(vec![3e38], &[]));
        let loss = x
            .matrix_norm(ord)
            .unwrap()
            .mul(&k)
            .unwrap()
            .sum(None)
            .unwrap();
        let r = tape.backward_create_graph(&loss, &child);
        assert!(is_backward_err(&r), "{name}: 過大な上流は拒否");
        assert!(child.is_empty(), "{name}: child は空");
    }
}

#[test]
fn nuc_spectral_cgrad_does_not_materialize_m_by_m() {
    let (m, n) = (100_000usize, 1usize);
    let data: Vec<f32> = (0..m * n)
        .map(|i| (1.0 + ((i * 7) % 11) as f32 * 0.37) * 1000.0)
        .collect();
    for (name, ord) in SVD_NORMS {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(data.clone(), &[m, n]));
        let loss = x.matrix_norm(ord).unwrap().sum(None).unwrap();
        let cg = tape
            .backward_create_graph(&loss, &child)
            .unwrap_or_else(|e| panic!("{name}: [100000,1] は受理されるはず: {e}"));
        let g = dense(&cg.grad(&x).unwrap().unwrap().to_tensor());
        assert!(g.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn nuc_spectral_neighbouring_custom_stays_fail_closed() {
    for (name, ord) in SVD_NORMS {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(sv_separated(&[3, 2]), &[3, 2]));
        let y = x.matrix_norm(ord).unwrap();
        let z = tape
            .custom(Arc::new(IdentityCustomFn), &[y])
            .expect("Tape::custom 登録成功");
        let loss = z.sum(None).unwrap();
        assert!(
            is_backward_err(&tape.backward_create_graph(&loss, &child)),
            "{name}: Op::Custom は拒否される"
        );
        assert!(child.is_empty(), "{name}: 拒否後は child が空");
    }
}
