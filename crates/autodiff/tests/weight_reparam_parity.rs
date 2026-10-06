//! `weight_reparam_ops`（イシュー #2646・`weight_norm`／`norm_except_dim`／`spectral_norm`）の
//! `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/lrn-weight-reparam-pytorch-reference/lrn_weight_reparam_reference.json`・
//!   生成条件は同ディレクトリの `README.md`）と突合する。f32 は u32 ビットパターンで保存。forward・
//!   勾配は REQ-2 統一複合判定（`common::req2_close`。tolerance 定数は新設しない。PyTorch は f32 累積・
//!   本実装は `f64` 累積のため bit 一致は求めない）。
//! - spectral_norm は fixture が保存した初期化直後の `u0`／`v0` から状態を作り、forward 1 回後の
//!   `u1`／`v1`・出力・`weight` 勾配が torch と一致することを確認する（初期化乱数は再現しない）。
//! - `common::naive_ops()` は新規 3 フックを override しないため、必ず共有ホストカーネルへの
//!   フォールバック経路を通る。CPU `BackendOps` 実装との一致は
//!   `crates/facade/tests/lrn_weight_reparam_backend_parity.rs` が担当する。
//! - フォールバックは `Unsupported` のときだけで、それ以外のエラーは握りつぶさず伝播する。
//!   エラー時に `SpectralNormState` が進まないこと・孤児ノード無し・中心差分オラクルも固定する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::weight_reparam_ops::{
    SPECTRAL_NORM_INIT_POWER_ITERATIONS, SpectralNormState, norm_except_dim, spectral_norm,
    weight_norm,
};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};
use serde::Deserialize;

const H: f64 = 1e-3;
const TAU: f64 = 1e-4;
const REL_TOL: f64 = 1e-2;
const ABS_TOL: f64 = 1e-3;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    weight_norm_cases: Vec<WnCase>,
    weight_norm_nonfinite_cases: Vec<WnCase>,
    weight_norm_error_cases: Vec<WnErr>,
    spectral_cases: Vec<SpCase>,
    spectral_error_cases: Vec<SpErr>,
}

#[derive(Deserialize)]
struct WnCase {
    name: String,
    v_shape: Vec<usize>,
    g_shape: Vec<usize>,
    dim: Option<usize>,
    v_bits: Vec<u32>,
    g_bits: Vec<u32>,
    up_bits: Vec<u32>,
    out_bits: Vec<u32>,
    dv_bits: Vec<u32>,
    dg_bits: Vec<u32>,
    norm_shape: Vec<usize>,
    norm_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct WnErr {
    name: String,
    v_shape: Vec<usize>,
    g_shape: Vec<usize>,
    dim: Option<usize>,
    torch_raises: bool,
}

#[derive(Deserialize)]
struct SpCase {
    name: String,
    shape: Vec<usize>,
    dim: usize,
    n_iter: usize,
    training: bool,
    eps_bits: u32,
    w_bits: Vec<u32>,
    u0_bits: Vec<u32>,
    v0_bits: Vec<u32>,
    u1_bits: Vec<u32>,
    v1_bits: Vec<u32>,
    up_bits: Vec<u32>,
    out_bits: Vec<u32>,
    grad_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct SpErr {
    name: String,
    shape: Vec<usize>,
    dim: usize,
    n_iter: usize,
    eps_bits: u32,
    torch_raises: bool,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/lrn-weight-reparam-pytorch-reference/lrn_weight_reparam_reference.json",
    );
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

/// NaN は NaN 同士、±inf は厳密、有限は REQ-2 統一複合判定。
fn class_close(a: f32, e: f32) -> bool {
    if e.is_nan() {
        a.is_nan()
    } else if e.is_infinite() {
        a == e
    } else {
        common::req2_close(f64::from(a), f64::from(e))
    }
}

fn assert_close_all(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(class_close(a, e), "{context}[{i}]: actual={a} expected={e}");
    }
}

fn dense(x: &Tensor<f32>) -> Vec<f32> {
    x.contiguous().host_slice().into_owned()
}

// --- weight_norm: PyTorch fixture 突合 ---

struct WnRun {
    out: Vec<f32>,
    dv: Vec<f32>,
    dg: Vec<f32>,
}

fn run_wn(ops: Box<dyn BackendOps + Send>, case: &WnCase) -> WnRun {
    let tape = Tape::new_with_ops(ops);
    let v = tape.var(&t(from_bits(&case.v_bits), &case.v_shape));
    let g = tape.var(&t(from_bits(&case.g_bits), &case.g_shape));
    let w = weight_norm(&v, &g, case.dim).expect("weight_norm");
    let out = w.to_tensor();
    let up = tape.var_no_grad(&t(from_bits(&case.up_bits), out.shape()));
    let loss = w.mul(&up).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    WnRun {
        out: dense(&out),
        dv: dense(grads.get(&v).unwrap().expect("v へ勾配が届く")),
        dg: dense(grads.get(&g).unwrap().expect("g へ勾配が届く")),
    }
}

fn check_wn(case: &WnCase) {
    let r = run_wn(common::naive_ops(), case);
    assert_close_all(
        &r.out,
        &from_bits(&case.out_bits),
        &format!("{} forward", case.name),
    );
    assert_close_all(
        &r.dv,
        &from_bits(&case.dv_bits),
        &format!("{} dv", case.name),
    );
    assert_close_all(
        &r.dg,
        &from_bits(&case.dg_bits),
        &format!("{} dg", case.name),
    );
}

#[test]
fn weight_norm_finite_cases_match_pytorch_reference() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    assert!(fixture.weight_norm_cases.len() >= 17);
    for case in &fixture.weight_norm_cases {
        check_wn(case);
    }
}

#[test]
fn weight_norm_zero_norm_and_nonfinite_cases_match_pytorch_reference() {
    let fixture = load_fixture();
    assert!(fixture.weight_norm_nonfinite_cases.len() >= 4);
    for case in &fixture.weight_norm_nonfinite_cases {
        check_wn(case);
    }
}

#[test]
fn norm_except_dim_matches_pytorch_values_and_shapes() {
    let fixture = load_fixture();
    for case in fixture
        .weight_norm_cases
        .iter()
        .chain(&fixture.weight_norm_nonfinite_cases)
    {
        let v = t(from_bits(&case.v_bits), &case.v_shape);
        let n = norm_except_dim(&v, case.dim).unwrap();
        assert_eq!(n.shape(), case.norm_shape.as_slice(), "{}", case.name);
        assert_close_all(
            &dense(&n),
            &from_bits(&case.norm_bits),
            &format!("{} norm", case.name),
        );
    }
}

/// torch と一致する点・意図的差分を表で固定する（決定記録 §5）。
#[test]
fn weight_norm_error_cases_agree_with_torch_except_documented_differences() {
    let fixture = load_fixture();
    for c in &fixture.weight_norm_error_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let vn: usize = c.v_shape.iter().product();
        let gn: usize = c.g_shape.iter().product();
        let v = tape.var(&t(vec![1.5; vn], &c.v_shape));
        let g = tape.var(&t(vec![1.0; gn], &c.g_shape));
        let before = tape.len();
        let r = weight_norm(&v, &g, c.dim);
        assert!(r.is_err(), "{}: 本実装は拒否する", c.name);
        assert_eq!(tape.len(), before, "{}: 孤児ノードを残さない", c.name);
        match c.name.as_str() {
            // torch も拒否。
            "dim_out_of_range" | "rank0" => assert!(c.torch_raises, "{}", c.name),
            // 意図的差分: torch は g の shape を（broadcast 可否すら）検査せず受理する。本実装は
            // `norm_except_dim` の出力 shape と完全一致のみ受理する。
            "g_flat_instead_of_keepdim"
            | "g_wrong_len"
            | "g_keepdim_on_wrong_axis"
            | "none_with_keepdim_g" => {
                assert!(!c.torch_raises, "{}: torch は受理するはず（差分）", c.name);
                assert!(matches!(r, Err(AutodiffError::Shape(_))), "{}", c.name);
            }
            other => panic!("未知の error case: {other}"),
        }
    }
}

// --- weight_norm: 独立オラクル ---

fn pseudo(n: usize, mul: usize, modulus: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * mul) % modulus) as f32 * 0.31 - 1.3)
        .collect()
}

fn assert_fd_close(label: &str, analytic: f32, numeric: f64) {
    let an = f64::from(analytic);
    let diff = (an - numeric).abs();
    let rel = diff / an.abs().max(numeric.abs()).max(TAU);
    assert!(
        rel <= REL_TOL || diff <= ABS_TOL,
        "{label}: analytic={an} numeric={numeric}"
    );
}

#[test]
fn weight_norm_gradients_match_central_difference() {
    let v_shape = [3_usize, 2, 4];
    let n: usize = v_shape.iter().product();
    for dim in [Some(0_usize), Some(1), Some(2), None] {
        let v = pseudo(n, 7, 23);
        let g_len = match dim {
            Some(d) => v_shape[d],
            None => 1,
        };
        let g: Vec<f32> = (0..g_len).map(|i| 0.6 + 0.4 * i as f32).collect();
        let g_shape: Vec<usize> = match dim {
            Some(d) => (0..3)
                .map(|a| if a == d { v_shape[d] } else { 1 })
                .collect(),
            None => vec![],
        };
        let up = pseudo(n, 5, 19);
        let loss = |vs: &[f32], gs: &[f32]| -> f64 {
            let tape = Tape::new_with_ops(common::naive_ops());
            let vv = tape.var(&t(vs.to_vec(), &v_shape));
            let gv = tape.var(&t(gs.to_vec(), &g_shape));
            let w = weight_norm(&vv, &gv, dim).unwrap();
            dense(&w.to_tensor())
                .iter()
                .zip(&up)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum()
        };
        let tape = Tape::new_with_ops(common::naive_ops());
        let vv = tape.var(&t(v.clone(), &v_shape));
        let gv = tape.var(&t(g.clone(), &g_shape));
        let w = weight_norm(&vv, &gv, dim).unwrap();
        let upv = tape.var_no_grad(&t(up.clone(), &v_shape));
        let l = w.mul(&upv).unwrap().sum(None).unwrap();
        let grads = tape.backward(&l).unwrap();
        let dv = dense(grads.get(&vv).unwrap().unwrap());
        let dg = dense(grads.get(&gv).unwrap().unwrap());
        for i in 0..n {
            let (mut p, mut m) = (v.clone(), v.clone());
            p[i] = (f64::from(v[i]) + H) as f32;
            m[i] = (f64::from(v[i]) - H) as f32;
            let num = (loss(&p, &g) - loss(&m, &g)) / (2.0 * H);
            assert_fd_close(&format!("{dim:?} dv[{i}]"), dv[i], num);
        }
        for i in 0..g_len {
            let (mut p, mut m) = (g.clone(), g.clone());
            p[i] = (f64::from(g[i]) + H) as f32;
            m[i] = (f64::from(g[i]) - H) as f32;
            let num = (loss(&v, &p) - loss(&v, &m)) / (2.0 * H);
            assert_fd_close(&format!("{dim:?} dg[{i}]"), dg[i], num);
        }
    }
}

#[test]
fn weight_norm_with_own_norm_reconstructs_the_weight() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let w0 = t(pseudo(24, 7, 23), &[2, 3, 4]);
    for dim in [Some(0_usize), Some(1), Some(2), None] {
        let g = tape.var(&norm_except_dim(&w0, dim).unwrap());
        let v = tape.var(&w0);
        let w = weight_norm(&v, &g, dim).unwrap();
        for (a, b) in dense(&w.to_tensor()).iter().zip(dense(&w0)) {
            assert!((a - b).abs() < 1e-5, "{dim:?}: {a} vs {b}");
        }
    }
}

#[test]
fn weight_norm_supports_no_grad_g_and_records_one_node() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let v = tape.var(&t(pseudo(6, 7, 23), &[2, 3]));
    let g = tape.var_no_grad(&t(vec![1.0, 2.0], &[2, 1]));
    let before = tape.len();
    let w = weight_norm(&v, &g, Some(0)).unwrap();
    assert_eq!(tape.len(), before + 1);
    let loss = w.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert!(grads.get(&v).unwrap().is_some());
    assert!(
        grads.get(&g).is_err(),
        "no-grad の g は勾配追跡対象外（GradientTrackingDisabled）"
    );
}

#[test]
fn weight_norm_invalid_arguments_leave_no_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let v = tape.var(&t(vec![1.0; 4], &[2, 2]));
    let g = tape.var(&t(vec![1.0; 2], &[2, 1]));
    let g_flat = tape.var(&t(vec![1.0; 2], &[2]));
    let other = Tape::new_with_ops(common::naive_ops());
    let g_other = other.var(&t(vec![1.0; 2], &[2, 1]));
    let before = tape.len();
    assert!(matches!(
        weight_norm(&v, &g, Some(2)),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
    ));
    assert!(matches!(
        weight_norm(&v, &g_flat, Some(0)),
        Err(AutodiffError::Shape(_))
    ));
    assert!(matches!(
        weight_norm(&v, &g_other, Some(0)),
        Err(AutodiffError::TapeMismatch)
    ));
    assert_eq!(tape.len(), before);
    assert!(norm_except_dim(&t(vec![1.0], &[]), Some(0)).is_err());
}

#[test]
fn weight_norm_huge_broadcast_view_is_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(vec![1.0], &[1, 1]));
    let huge = base.broadcast_to(&[1, 1usize << 61]).unwrap();
    let g = tape.var(&t(vec![1.0], &[1, 1]));
    assert!(matches!(
        weight_norm(&huge, &g, Some(0)),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
}

#[test]
fn create_graph_is_a_typed_error_not_a_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let v = tape.var(&t(pseudo(6, 7, 23), &[2, 3]));
    let g = tape.var(&t(vec![1.0, 2.0], &[2, 1]));
    let loss = weight_norm(&v, &g, Some(0)).unwrap().sum(None).unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
    let w = tape.var(&t(pseudo(6, 7, 23), &[2, 3]));
    let mut st =
        SpectralNormState::from_vectors(&[2, 3], 0, &[1.0, 1.0], &[1.0, 1.0, 1.0], 1, 1e-12)
            .unwrap();
    let loss = spectral_norm(&w, &mut st, false)
        .unwrap()
        .sum(None)
        .unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
}

// --- spectral_norm: PyTorch fixture 突合 ---

fn state_for(case: &SpCase) -> SpectralNormState {
    SpectralNormState::from_vectors(
        &case.shape,
        case.dim,
        &from_bits(&case.u0_bits),
        &from_bits(&case.v0_bits),
        case.n_iter,
        f32::from_bits(case.eps_bits),
    )
    .expect("from_vectors")
}

#[test]
fn spectral_norm_cases_match_pytorch_reference() {
    let fixture = load_fixture();
    assert!(fixture.spectral_cases.len() >= 18);
    let (mut train, mut eval) = (0, 0);
    for case in &fixture.spectral_cases {
        let mut st = state_for(case);
        let tape = Tape::new_with_ops(common::naive_ops());
        let w = tape.var(&t(from_bits(&case.w_bits), &case.shape));
        let out = spectral_norm(&w, &mut st, case.training).expect("spectral_norm");
        assert_close_all(
            st.u(),
            &from_bits(&case.u1_bits),
            &format!("{} u1", case.name),
        );
        assert_close_all(
            st.v(),
            &from_bits(&case.v1_bits),
            &format!("{} v1", case.name),
        );
        if case.training {
            train += 1;
        } else {
            eval += 1;
            assert_close_all(
                st.u(),
                &from_bits(&case.u0_bits),
                &format!("{} eval は状態不変", case.name),
            );
        }
        let out_t = out.to_tensor();
        assert_eq!(out_t.shape(), case.shape.as_slice(), "{}", case.name);
        assert_close_all(
            &dense(&out_t),
            &from_bits(&case.out_bits),
            &format!("{} forward", case.name),
        );
        let up = tape.var_no_grad(&t(from_bits(&case.up_bits), &case.shape));
        let loss = out.mul(&up).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        assert_close_all(
            &dense(grads.get(&w).unwrap().expect("weight へ勾配が届く")),
            &from_bits(&case.grad_bits),
            &format!("{} grad", case.name),
        );
    }
    assert!(train >= 12 && eval >= 3, "{train}/{eval}");
}

/// torch と一致する点・意図的差分を表で固定する（決定記録 §5）。
#[test]
fn spectral_error_cases_agree_with_torch_except_documented_differences() {
    let fixture = load_fixture();
    for c in &fixture.spectral_error_cases {
        let zeros_u = vec![1.0_f32; c.shape.first().copied().unwrap_or(0)];
        let cols = if c.shape.len() >= 2 { c.shape[1] } else { 0 };
        let r = SpectralNormState::from_vectors(
            &c.shape,
            c.dim,
            &zeros_u,
            &vec![1.0_f32; cols],
            c.n_iter,
            f32::from_bits(c.eps_bits),
        );
        assert!(r.is_err(), "{}: 本実装は拒否する", c.name);
        match c.name.as_str() {
            // torch も拒否。
            "n_iter0" | "dim_out_of_range" => assert!(c.torch_raises, "{}", c.name),
            // 意図的差分（torch は受理）: rank 1（`F.normalize` へ縮退）・要素数 0・負／NaN の eps。
            "rank1" | "zero_elements" | "eps_negative" | "eps_nan" => {
                assert!(!c.torch_raises, "{}: torch は受理するはず（差分）", c.name);
            }
            other => panic!("未知の error case: {other}"),
        }
    }
}

// --- spectral_norm: 独立オラクル・状態の不変条件 ---

fn weight_matrix_state(shape: &[usize], dim: usize, n: usize) -> SpectralNormState {
    let rows = shape[dim];
    let cols: usize = shape.iter().product::<usize>() / rows;
    let u0: Vec<f32> = (0..rows).map(|i| 1.0 + 0.3 * i as f32).collect();
    let v0: Vec<f32> = (0..cols).map(|j| 0.7 - 0.1 * j as f32).collect();
    SpectralNormState::from_vectors(shape, dim, &u0, &v0, n, 1e-12).unwrap()
}

#[test]
fn spectral_gradient_matches_central_difference_with_fixed_state() {
    for (shape, dim) in [
        (vec![3_usize, 4], 0_usize),
        (vec![3, 4], 1),
        (vec![2, 3, 2], 1),
    ] {
        let n: usize = shape.iter().product();
        let w = pseudo(n, 7, 23);
        let up = pseudo(n, 5, 19);
        // eval（状態固定）で `σ = uᵀ W v` の `W` 依存を含めた勾配を検証する。
        let st = weight_matrix_state(&shape, dim, 1);
        let loss = |ws: &[f32]| -> f64 {
            let mut s = st.clone();
            let tape = Tape::new_with_ops(common::naive_ops());
            let wv = tape.var(&t(ws.to_vec(), &shape));
            let y = spectral_norm(&wv, &mut s, false).unwrap();
            dense(&y.to_tensor())
                .iter()
                .zip(&up)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum()
        };
        let mut s = st.clone();
        let tape = Tape::new_with_ops(common::naive_ops());
        let wv = tape.var(&t(w.clone(), &shape));
        let y = spectral_norm(&wv, &mut s, false).unwrap();
        let upv = tape.var_no_grad(&t(up.clone(), &shape));
        let l = y.mul(&upv).unwrap().sum(None).unwrap();
        let grads = tape.backward(&l).unwrap();
        let dw = dense(grads.get(&wv).unwrap().unwrap());
        for i in 0..n {
            let (mut p, mut m) = (w.clone(), w.clone());
            p[i] = (f64::from(w[i]) + H) as f32;
            m[i] = (f64::from(w[i]) - H) as f32;
            let num = (loss(&p) - loss(&m)) / (2.0 * H);
            assert_fd_close(&format!("{shape:?} dim={dim} dw[{i}]"), dw[i], num);
        }
    }
}

#[test]
fn spectral_norm_converges_to_unit_largest_singular_value() {
    // diag(3, 1, 0.5)。十分な反復後の出力の最大特異値は 1（出力の対角成分 1・1/3・1/6）。
    let shape = [3_usize, 3];
    let w0 = t(vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], &shape);
    let mut st = weight_matrix_state(&shape, 0, 1);
    st.power_iterate(&w0, SPECTRAL_NORM_INIT_POWER_ITERATIONS * 4)
        .unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let w = tape.var(&w0);
    let out = dense(&spectral_norm(&w, &mut st, true).unwrap().to_tensor());
    for (got, want) in [(out[0], 1.0_f32), (out[4], 1.0 / 3.0), (out[8], 1.0 / 6.0)] {
        assert!((got - want).abs() < 1e-4, "{got} vs {want}");
    }
}

#[test]
fn eval_keeps_state_and_training_advances_it() {
    let shape = [3_usize, 4];
    let w0 = t(pseudo(12, 7, 23), &shape);
    let mut st = weight_matrix_state(&shape, 0, 2);
    let before = st.clone();
    let tape = Tape::new_with_ops(common::naive_ops());
    let w = tape.var(&w0);
    spectral_norm(&w, &mut st, false).unwrap();
    assert_eq!(st, before);
    spectral_norm(&w, &mut st, true).unwrap();
    assert_ne!(st, before);
    // training の 1 回 = n_power_iterations(2) 回の `power_iterate`。
    let mut manual = before.clone();
    manual.power_iterate(&w0, 2).unwrap();
    assert_eq!(st.u(), manual.u());
    assert_eq!(st.v(), manual.v());
}

#[test]
fn invalid_state_or_weight_leaves_state_and_tape_untouched() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let mut st = weight_matrix_state(&[3, 4], 0, 2);
    let before = st.clone();
    let wrong_shape = tape.var(&t(vec![1.0; 6], &[2, 3]));
    let rank1 = tape.var(&t(vec![1.0; 4], &[4]));
    let n = tape.len();
    assert!(matches!(
        spectral_norm(&wrong_shape, &mut st, true),
        Err(AutodiffError::Shape(_))
    ));
    assert!(spectral_norm(&rank1, &mut st, true).is_err());
    assert_eq!(st, before);
    assert_eq!(tape.len(), n, "孤児ノードを残さない");
    let w0 = t(vec![1.0; 6], &[2, 3]);
    assert!(st.power_iterate(&w0, 1).is_err());
    assert_eq!(st, before);
    // from_vectors の検査。
    assert!(SpectralNormState::from_vectors(&[3], 0, &[1.0; 3], &[1.0], 1, 1e-12).is_err());
    assert!(SpectralNormState::from_vectors(&[3, 4], 2, &[1.0; 3], &[1.0; 4], 1, 1e-12).is_err());
    assert!(SpectralNormState::from_vectors(&[0, 4], 0, &[], &[1.0; 4], 1, 1e-12).is_err());
    assert!(SpectralNormState::from_vectors(&[3, 4], 0, &[1.0; 3], &[1.0; 4], 0, 1e-12).is_err());
    assert!(SpectralNormState::from_vectors(&[3, 4], 0, &[1.0; 3], &[1.0; 4], 1, -1.0).is_err());
    assert!(
        SpectralNormState::from_vectors(&[3, 4], 0, &[1.0; 3], &[1.0; 4], 1, f32::NAN).is_err()
    );
    assert!(SpectralNormState::from_vectors(&[3, 4], 0, &[1.0; 2], &[1.0; 4], 1, 1e-12).is_err());
    assert!(
        SpectralNormState::from_vectors(
            &[3, 4],
            0,
            &[f32::INFINITY, 1.0, 1.0],
            &[1.0; 4],
            1,
            1e-12
        )
        .is_err()
    );
}

#[test]
fn state_accessors_report_the_normalized_vectors() {
    let st =
        SpectralNormState::from_vectors(&[2, 2], 1, &[3.0, 4.0], &[0.0, 2.0], 3, 1e-6).unwrap();
    assert_eq!(st.weight_shape(), &[2, 2]);
    assert_eq!((st.dim(), st.n_power_iterations()), (1, 3));
    assert_eq!(st.eps(), 1e-6);
    assert!((st.u()[0] - 0.6).abs() < 1e-6 && (st.u()[1] - 0.8).abs() < 1e-6);
    assert!((st.v()[1] - 1.0).abs() < 1e-6);
}

#[test]
fn spectral_results_are_bit_deterministic_run_to_run() {
    let fixture = load_fixture();
    for case in fixture.spectral_cases.iter().filter(|c| c.training) {
        let run = || {
            let mut st = state_for(case);
            let tape = Tape::new_with_ops(common::naive_ops());
            let w = tape.var(&t(from_bits(&case.w_bits), &case.shape));
            let out = spectral_norm(&w, &mut st, true).unwrap();
            (dense(&out.to_tensor()), st.u().to_vec(), st.v().to_vec())
        };
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        let (a, b) = (run(), run());
        assert_eq!(bits(&a.0), bits(&b.0), "{}", case.name);
        assert_eq!(bits(&a.1), bits(&b.1), "{}", case.name);
        assert_eq!(bits(&a.2), bits(&b.2), "{}", case.name);
    }
}

// --- フォールバックとエラー伝播（AC2） ---

/// 2 フックだけを差し替える `BackendOps`。それ以外は naive へ委譲する。
struct ReparamMock {
    inner: Box<dyn BackendOps + Send>,
    calls: Arc<AtomicUsize>,
    mode: Mode,
}

#[derive(Clone, Copy)]
enum Mode {
    Unsupported,
    LaunchFailed,
    WrongShape,
}

impl ReparamMock {
    fn respond(&self) -> Result<Tensor<f32>, BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongShape => Ok(t(vec![0.0; 2], &[2])),
        }
    }
}

impl BackendOps for ReparamMock {
    fn device(&self) -> Device {
        self.inner.device()
    }
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.gemm(a, b)
    }
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.add(a, b)
    }
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.mul(a, b)
    }
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.relu(a)
    }
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.exp(a)
    }
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.tanh(a)
    }
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.sum(a, dim)
    }
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.max(a, dim)
    }
    fn weight_norm_forward(
        &self,
        _v: &Tensor<f32>,
        _g: &Tensor<f32>,
        _dim: Option<usize>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.respond()
    }
    fn spectral_norm_forward(
        &self,
        _weight: &Tensor<f32>,
        _u: &Tensor<f32>,
        _v: &Tensor<f32>,
        _dim: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        self.respond()
    }
}

fn mock_tape(mode: Mode) -> (Tape, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(ReparamMock {
        inner: common::naive_ops(),
        calls: Arc::clone(&calls),
        mode,
    }));
    (tape, calls)
}

fn wn_inputs(tape: &Tape) -> (Var<'_>, Var<'_>) {
    (
        tape.var(&t(vec![3.0, 4.0, 0.0, 2.0], &[2, 2])),
        tape.var(&t(vec![10.0, 1.0], &[2, 1])),
    )
}

#[test]
fn unsupported_backend_falls_back_to_host_reference() {
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let (v, g) = wn_inputs(&tape);
    let w = weight_norm(&v, &g, Some(0)).unwrap();
    assert_eq!(dense(&w.to_tensor()), vec![6.0, 8.0, 0.0, 1.0]);
    let mut st = weight_matrix_state(&[2, 2], 0, 1);
    let sw = tape.var(&t(vec![3.0, 0.0, 0.0, 1.0], &[2, 2]));
    let before = st.clone();
    spectral_norm(&sw, &mut st, true).unwrap();
    assert_ne!(st, before, "フォールバック成功時は training で状態が進む");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "バックエンドを先に呼んでからフォールバックしているはず"
    );
}

#[test]
fn non_unsupported_backend_errors_are_propagated_and_state_does_not_advance() {
    let (tape, _) = mock_tape(Mode::LaunchFailed);
    let (v, g) = wn_inputs(&tape);
    assert!(matches!(
        weight_norm(&v, &g, Some(0)),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    let mut st = weight_matrix_state(&[2, 2], 0, 3);
    let before = st.clone();
    let sw = tape.var(&t(vec![3.0, 0.0, 0.0, 1.0], &[2, 2]));
    let n = tape.len();
    assert!(matches!(
        spectral_norm(&sw, &mut st, true),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    assert_eq!(st, before, "forward 失敗時に状態だけ進まない");
    assert_eq!(tape.len(), n);
}

#[test]
fn wrong_shape_from_backend_is_a_typed_error_and_state_does_not_advance() {
    let (tape, _) = mock_tape(Mode::WrongShape);
    let (v, g) = wn_inputs(&tape);
    let n = tape.len();
    assert!(matches!(
        weight_norm(&v, &g, Some(0)),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
    assert_eq!(tape.len(), n, "孤児ノードを残さない");
    let mut st = weight_matrix_state(&[2, 2], 0, 3);
    let before = st.clone();
    let sw = tape.var(&t(vec![3.0, 0.0, 0.0, 1.0], &[2, 2]));
    let n2 = tape.len();
    assert!(matches!(
        spectral_norm(&sw, &mut st, true),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
    assert_eq!(st, before);
    assert_eq!(tape.len(), n2);
}
