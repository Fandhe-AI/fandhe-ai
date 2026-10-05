//! `trig_ops`（イシュー #2634・`atan`／`asin`／`acos`／`sinh`／`cosh`／
//! `asinh`／`acosh`／`atanh`／`atan2`）の `Tape`／`Var` を経由する
//! end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/trig-ops-pytorch-reference/trig_ops_reference.json`・
//!   生成条件は同ディレクトリの `README.md`）の forward 出力と入力勾配を
//!   REQ-2 統一複合判定（`common::req2_close`）で突合する。tolerance 定数は
//!   新設しない。定義域の境界・外側・巨大値・符号付きゼロ（`edge_cases`）は
//!   値クラス（NaN／+inf／-inf／有限）で突合し、PyTorch と意図的に異なる
//!   項目（`atan2` の `1e-30`／`1e20`）は自前の挙動を明示 assert する。
//! - fixture とは独立に f64 中心差分で勾配を検証する。
//!
//! `common::naive_ops()` は `scalar_unary`／`scalar_binary` を持たない
//! （既定 `Unsupported`）ため、必ずホスト参照実装へのフォールバック経路を
//! 通る。CPU `BackendOps` 実装との一致は
//! `crates/facade/tests/trig_ops_backend_parity.rs` が担当する。

mod common;

use std::path::PathBuf;

use fandhe_ai_autodiff::trig_ops::{acos, acosh, asin, asinh, atan, atan2, atanh, cosh, sinh};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, ScalarBinaryOp, ScalarUnaryOp, Tensor};
use serde::Deserialize;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    cases: Vec<Case>,
    edge_cases: Vec<EdgeCase>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    op: String,
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    shape_a: Vec<usize>,
    #[serde(default)]
    shape_b: Vec<usize>,
    #[serde(default)]
    out_shape: Vec<usize>,
    #[serde(default)]
    x: Vec<f32>,
    #[serde(default)]
    a: Vec<f32>,
    #[serde(default)]
    b: Vec<f32>,
    g: Vec<f32>,
    out: Vec<f32>,
    #[serde(default)]
    grad_x: Vec<f32>,
    #[serde(default)]
    grad_a: Vec<f32>,
    #[serde(default)]
    grad_b: Vec<f32>,
}

#[derive(Deserialize, Clone)]
struct Klass {
    class: String,
    value: Option<f32>,
}

#[derive(Deserialize)]
struct EdgeCase {
    name: String,
    op: String,
    #[serde(default)]
    x: Vec<f32>,
    #[serde(default)]
    a: Vec<f32>,
    #[serde(default)]
    b: Vec<f32>,
    out: Vec<Klass>,
    #[serde(default)]
    grad_x: Vec<Klass>,
    #[serde(default)]
    grad_a: Vec<Klass>,
    #[serde(default)]
    grad_b: Vec<Klass>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/trig-ops-pytorch-reference/trig_ops_reference.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture を読めない: {}: {e}", path.display()));
    serde_json::from_str(&text).expect("fixture の JSON が不正")
}

type UnaryFn = for<'t> fn(&Var<'t>) -> Result<Var<'t>, AutodiffError>;

fn unary_fn(op: &str) -> UnaryFn {
    match op {
        "atan" => atan,
        "asin" => asin,
        "acos" => acos,
        "sinh" => sinh,
        "cosh" => cosh,
        "asinh" => asinh,
        "acosh" => acosh,
        "atanh" => atanh,
        other => panic!("未知の unary op: {other}"),
    }
}

fn unary_kind(op: &str) -> ScalarUnaryOp {
    match op {
        "atan" => ScalarUnaryOp::Atan,
        "asin" => ScalarUnaryOp::Asin,
        "acos" => ScalarUnaryOp::Acos,
        "sinh" => ScalarUnaryOp::Sinh,
        "cosh" => ScalarUnaryOp::Cosh,
        "asinh" => ScalarUnaryOp::Asinh,
        "acosh" => ScalarUnaryOp::Acosh,
        "atanh" => ScalarUnaryOp::Atanh,
        other => panic!("未知の unary op: {other}"),
    }
}

const UNARY_OPS: [&str; 8] = [
    "atan", "asin", "acos", "sinh", "cosh", "asinh", "acosh", "atanh",
];

fn assert_close_all(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            common::req2_close(f64::from(a), f64::from(e)),
            "{context}[{i}]: actual={a} expected={e}"
        );
    }
}

fn assert_class(actual: f32, expected: &Klass, context: &str) {
    match expected.class.as_str() {
        "nan" => assert!(actual.is_nan(), "{context}: NaN を期待, actual={actual}"),
        "pos_inf" => assert_eq!(actual, f32::INFINITY, "{context}"),
        "neg_inf" => assert_eq!(actual, f32::NEG_INFINITY, "{context}"),
        "finite" => {
            let e = expected.value.expect("finite には value がある");
            assert!(
                common::req2_close(f64::from(actual), f64::from(e)),
                "{context}: actual={actual} expected={e}"
            );
        }
        other => panic!("未知の class: {other}"),
    }
}

fn assert_classes(actual: &[f32], expected: &[Klass], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, e)) in actual.iter().zip(expected).enumerate() {
        assert_class(a, e, &format!("{context}[{i}]"));
    }
}

/// unary を実行して `(out, grad_x)` を返す。損失は `sum(y * g)`。
fn run_unary(op: &str, x: &[f32], shape: &[usize], g: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&t(x.to_vec(), shape));
    let y = unary_fn(op)(&xv).unwrap_or_else(|e| panic!("{op}: forward が失敗: {e}"));
    let out = y.to_tensor().host_slice().into_owned();
    let gv = tape.var_no_grad(&t(g.to_vec(), shape));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("入力へ勾配が届く").clone();
    (out, dx.host_slice().into_owned())
}

/// atan2 を実行して `(out, grad_a, grad_b)` を返す。
#[allow(clippy::type_complexity)]
fn run_atan2(
    a: &[f32],
    sa: &[usize],
    b: &[f32],
    sb: &[usize],
    g: &[f32],
    so: &[usize],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let av = tape.var(&t(a.to_vec(), sa));
    let bv = tape.var(&t(b.to_vec(), sb));
    let y = atan2(&av, &bv).expect("atan2 forward");
    assert_eq!(y.to_tensor().shape(), so, "atan2: 出力 shape");
    let out = y.to_tensor().host_slice().into_owned();
    let gv = tape.var_no_grad(&t(g.to_vec(), so));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&av).unwrap().expect("a へ勾配").clone();
    let db = grads.get(&bv).unwrap().expect("b へ勾配").clone();
    assert_eq!(da.shape(), sa, "grad_a は元の shape へ縮約される");
    assert_eq!(db.shape(), sb, "grad_b は元の shape へ縮約される");
    (
        out,
        da.host_slice().into_owned(),
        db.host_slice().into_owned(),
    )
}

#[test]
fn matches_pytorch_reference_forward_and_backward() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    for op in UNARY_OPS.iter().chain(["atan2"].iter()) {
        assert!(
            fixture.cases.iter().any(|c| c.op == *op),
            "fixture に {op} のケースが無い"
        );
    }
    for case in &fixture.cases {
        if case.op == "atan2" {
            let (out, da, db) = run_atan2(
                &case.a,
                &case.shape_a,
                &case.b,
                &case.shape_b,
                &case.g,
                &case.out_shape,
            );
            assert_close_all(&out, &case.out, &format!("{} forward", case.name));
            assert_close_all(&da, &case.grad_a, &format!("{} grad_a", case.name));
            assert_close_all(&db, &case.grad_b, &format!("{} grad_b", case.name));
        } else {
            let (out, dx) = run_unary(&case.op, &case.x, &case.shape, &case.g);
            assert_close_all(&out, &case.out, &format!("{} forward", case.name));
            assert_close_all(&dx, &case.grad_x, &format!("{} grad", case.name));
        }
    }
}

#[test]
fn edge_cases_match_pytorch_value_class() {
    let fixture = load_fixture();
    assert_eq!(fixture.edge_cases.len(), 9, "9 演算すべての edge_cases");
    for e in &fixture.edge_cases {
        if e.op == "atan2" {
            let n = e.a.len();
            let ones = vec![1.0; n];
            let (out, da, db) = run_atan2(&e.a, &[n], &e.b, &[n], &ones, &[n]);
            assert_classes(&out, &e.out, &format!("{} forward", e.name));
            // PyTorch は f32 のまま分母を計算し 1e-30／1e20 で 0 へ
            // underflow／overflow する。本実装は f64 昇格で数学的に正しい
            // 有限値を返す意図的な差分（決定記録 §5）。該当要素は自前の
            // 挙動を明示 assert し、それ以外は PyTorch と突合する。
            for i in 0..n {
                let ctx = format!("{} atan2 ({}, {})", e.name, e.a[i], e.b[i]);
                if e.a[i].abs() == 1e-30 || e.a[i].abs() == 1e20 {
                    let expected = f64::from(e.b[i]) / (2.0 * f64::from(e.a[i]).powi(2));
                    assert!(da[i].is_finite() && db[i].is_finite(), "{ctx}: 有限");
                    assert!(
                        common::req2_close(f64::from(da[i]), expected),
                        "{ctx}: da={}",
                        da[i]
                    );
                } else {
                    assert_class(da[i], &e.grad_a[i], &format!("{ctx} grad_a"));
                    assert_class(db[i], &e.grad_b[i], &format!("{ctx} grad_b"));
                }
            }
        } else {
            let n = e.x.len();
            let ones = vec![1.0; n];
            let (out, dx) = run_unary(&e.op, &e.x, &[n], &ones);
            assert_classes(&out, &e.out, &format!("{} forward", e.name));
            assert_classes(&dx, &e.grad_x, &format!("{} grad", e.name));
        }
    }
}

fn f64_unary(op: &str, x: f64) -> f64 {
    match op {
        "atan" => x.atan(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "sinh" => x.sinh(),
        "cosh" => x.cosh(),
        "asinh" => x.asinh(),
        "acosh" => x.acosh(),
        "atanh" => x.atanh(),
        other => panic!("未知の op: {other}"),
    }
}

#[test]
fn backward_matches_central_difference() {
    let h = 1e-5_f64;
    for op in UNARY_OPS {
        let xs: Vec<f32> = match op {
            "asin" | "acos" | "atanh" => vec![-0.7, -0.2, 0.3, 0.8],
            "acosh" => vec![1.3, 2.0, 3.5, 5.0],
            _ => vec![-1.7, -0.6, 0.9, 2.3],
        };
        let g = vec![1.0_f32; xs.len()];
        let (_, dx) = run_unary(op, &xs, &[xs.len()], &g);
        for (i, &x) in xs.iter().enumerate() {
            let x = f64::from(x);
            let numeric = (f64_unary(op, x + h) - f64_unary(op, x - h)) / (2.0 * h);
            assert!(
                common::req2_close(f64::from(dx[i]), numeric),
                "{op}[{i}] analytic={} numeric={numeric}",
                dx[i]
            );
        }
    }
    // atan2: broadcast（[2,3] 対 [3]）の縮約も含めて検証する。
    let a = [1.3_f32, 0.7, -1.1, -0.9, 1.6, -2.0];
    let b = [0.6_f32, -1.8, 1.2];
    let g = [1.0_f32; 6];
    let (_, da, db) = run_atan2(&a, &[2, 3], &b, &[3], &g, &[2, 3]);
    for j in 0..3 {
        let mut num_b = 0.0;
        for i in 0..2 {
            let (y, x) = (f64::from(a[i * 3 + j]), f64::from(b[j]));
            let na = ((y + h).atan2(x) - (y - h).atan2(x)) / (2.0 * h);
            assert!(
                common::req2_close(f64::from(da[i * 3 + j]), na),
                "atan2 da[{i},{j}]"
            );
            num_b += (y.atan2(x + h) - y.atan2(x - h)) / (2.0 * h);
        }
        assert!(common::req2_close(f64::from(db[j]), num_b), "atan2 db[{j}]");
    }
}

#[test]
fn forward_matches_apply_bit_for_bit() {
    let data = vec![-2.7_f32, -0.5, 0.0, -0.0, 1.5, 0.9, f32::NAN, f32::INFINITY];
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(data.clone(), &[data.len()]));
    for op in UNARY_OPS {
        let got = unary_fn(op)(&x).unwrap().to_tensor();
        let kind = unary_kind(op);
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(
                got.host_slice()[i].to_bits(),
                kind.apply(v).to_bits(),
                "{op}({v})"
            );
        }
    }
    let ys = [1.0_f32, -1.0, 0.0, -0.0, f32::NAN, 2.0];
    let xs = [-1.0_f32, 2.0, 0.0, -0.0, 1.0, f32::INFINITY];
    let y = tape.var(&t(ys.to_vec(), &[6]));
    let xv = tape.var(&t(xs.to_vec(), &[6]));
    let out = atan2(&y, &xv).unwrap().to_tensor();
    for i in 0..6 {
        assert_eq!(
            out.host_slice()[i].to_bits(),
            ScalarBinaryOp::Atan2.apply(ys[i], xs[i]).to_bits(),
            "atan2({}, {})",
            ys[i],
            xs[i]
        );
    }
}

#[test]
fn forward_preserves_shape_and_rejects_cross_tape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.1; 6], &[2, 3]));
    for op in UNARY_OPS {
        assert_eq!(unary_fn(op)(&x).unwrap().to_tensor().shape(), &[2, 3]);
    }
    // shape が broadcast 不能な atan2 は型付きエラー。
    let bad = tape.var(&t(vec![0.1; 4], &[4]));
    assert!(atan2(&x, &bad).is_err());
    // 別 tape の Var は拒否される。
    let other = Tape::new_with_ops(common::naive_ops());
    let o = other.var(&t(vec![0.1; 6], &[2, 3]));
    assert!(atan2(&x, &o).is_err());
}

// --- Unsupported 以外のバックエンドエラーはフォールバックしない ---

/// `scalar_unary`／`scalar_binary` が指定のエラーを返すスタブ。それ以外は
/// naive 実装へ委譲する。
struct ScalarErrOps {
    inner: Box<dyn BackendOps + Send>,
    error: fn() -> BackendError,
}

impl BackendOps for ScalarErrOps {
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
    fn scalar_unary(
        &self,
        _op: ScalarUnaryOp,
        _a: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err((self.error)())
    }
    fn scalar_binary(
        &self,
        _op: ScalarBinaryOp,
        _a: &Tensor<f32>,
        _b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err((self.error)())
    }
}

fn err_tape(error: fn() -> BackendError) -> Tape {
    Tape::new_with_ops(Box::new(ScalarErrOps {
        inner: common::naive_ops(),
        error,
    }))
}

#[test]
fn unsupported_backend_falls_back_to_host_forward_and_backward() {
    let tape = err_tape(|| BackendError::Unsupported("none".into()));
    for op in UNARY_OPS {
        let xs: Vec<f32> = match op {
            "asin" | "acos" | "atanh" => vec![-0.5, 0.25],
            "acosh" => vec![1.5, 2.5],
            _ => vec![-0.5, 0.25],
        };
        let x = tape.var(&t(xs.clone(), &[2]));
        let y = unary_fn(op)(&x).unwrap();
        let out = y.to_tensor().host_slice().into_owned();
        for (i, &v) in xs.iter().enumerate() {
            assert_eq!(out[i].to_bits(), unary_kind(op).apply(v).to_bits(), "{op}");
        }
        // backward が `unreachable!` を踏まず有限の勾配を返す。
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().expect("勾配").clone();
        assert!(
            dx.host_slice().iter().all(|v| v.is_finite()),
            "{op}: backward"
        );
    }
    let y = tape.var(&t(vec![1.0, -2.0], &[2]));
    let x = tape.var(&t(vec![0.5, 1.5], &[2]));
    let out = atan2(&y, &x).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert!(grads.get(&y).unwrap().is_some() && grads.get(&x).unwrap().is_some());
}

#[test]
fn non_unsupported_backend_error_is_propagated() {
    let tape = err_tape(|| BackendError::KernelLaunchFailed("boom".into()));
    let x = tape.var(&t(vec![0.1, 0.2], &[2]));
    for op in UNARY_OPS {
        assert!(
            matches!(
                unary_fn(op)(&x),
                Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
            ),
            "{op}: エラーを握りつぶさない"
        );
    }
    let y = tape.var(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        atan2(&y, &x),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
}
