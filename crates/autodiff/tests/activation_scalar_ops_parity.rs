//! `activation_scalar_ops`（イシュー #2649・`selu`／`celu`／`softsign`／
//! `hardsigmoid`／`log_sigmoid`）の `Tape`／`Var` を経由する end-to-end
//! 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/activation-scalar-ops-pytorch-reference/
//!   activation_scalar_ops_reference.json`・生成条件は同ディレクトリの
//!   `README.md`）の forward 出力と入力勾配を REQ-2 統一複合判定
//!   （`common::req2_close`）で突合する。tolerance 定数は新設しない。
//!   境界・巨大値・非有限（`edge_cases`）は値クラスで突合し、PyTorch と意図的に
//!   異なる項目は自前の挙動を明示 assert する。
//! - fixture とは独立に f64 中心差分で勾配を検証する。
//!
//! `common::naive_ops()` は `scalar_unary` を持たない（既定 `Unsupported`）ため、
//! 必ずホスト参照実装へのフォールバック経路を通る。CPU `BackendOps` 実装との
//! 一致は `crates/facade/tests/activation_scalar_ops_backend_parity.rs` が担当する。

mod common;

use std::path::PathBuf;

use fandhe_ai_autodiff::activation_scalar_ops::{celu, hardsigmoid, log_sigmoid, selu, softsign};
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
    error_cases: Vec<ErrorCase>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    op: String,
    shape: Vec<usize>,
    alpha: Option<f32>,
    x: Vec<f32>,
    g: Vec<f32>,
    out: Vec<f32>,
    grad_x: Vec<f32>,
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
    alpha: Option<f32>,
    x_bits: Vec<u32>,
    out: Vec<Klass>,
    grad_x: Vec<Klass>,
}

#[derive(Deserialize)]
struct ErrorCase {
    name: String,
    alpha_class: String,
    raised: bool,
    exception_type: Option<String>,
    out: Vec<Klass>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/activation-scalar-ops-pytorch-reference/activation_scalar_ops_reference.json",
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture を読めない: {}: {e}", path.display()));
    serde_json::from_str(&text).expect("fixture の JSON が不正")
}

const OPS: [&str; 5] = ["selu", "celu", "softsign", "hardsigmoid", "log_sigmoid"];

fn apply_op<'t>(op: &str, alpha: Option<f32>, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    match op {
        "selu" => selu(x),
        "celu" => celu(x, alpha.expect("celu には alpha がある")),
        "softsign" => softsign(x),
        "hardsigmoid" => hardsigmoid(x),
        "log_sigmoid" => log_sigmoid(x),
        other => panic!("未知の op: {other}"),
    }
}

fn kind(op: &str, alpha: Option<f32>) -> ScalarUnaryOp {
    match op {
        "selu" => ScalarUnaryOp::Selu,
        "celu" => ScalarUnaryOp::Celu {
            alpha: alpha.expect("celu には alpha がある"),
        },
        "softsign" => ScalarUnaryOp::Softsign,
        "hardsigmoid" => ScalarUnaryOp::Hardsigmoid,
        "log_sigmoid" => ScalarUnaryOp::LogSigmoid,
        other => panic!("未知の op: {other}"),
    }
}

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

/// `op` を実行して `(out, grad_x)` を返す。損失は `sum(y * g)`。
fn run(
    op: &str,
    alpha: Option<f32>,
    x: &[f32],
    shape: &[usize],
    g: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&t(x.to_vec(), shape));
    let y = apply_op(op, alpha, &xv).unwrap_or_else(|e| panic!("{op}: forward が失敗: {e}"));
    let out = y.to_tensor().host_slice().into_owned();
    let gv = tape.var_no_grad(&t(g.to_vec(), shape));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("入力へ勾配が届く").clone();
    (out, dx.host_slice().into_owned())
}

#[test]
fn matches_pytorch_reference_forward_and_backward() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    for op in OPS {
        assert!(
            fixture.cases.iter().any(|c| c.op == op),
            "fixture に {op} のケースが無い"
        );
    }
    for case in &fixture.cases {
        let (out, dx) = run(&case.op, case.alpha, &case.x, &case.shape, &case.g);
        assert_close_all(&out, &case.out, &format!("{} forward", case.name));
        assert_close_all(&dx, &case.grad_x, &format!("{} grad", case.name));
    }
}

#[test]
fn edge_cases_match_pytorch_value_class() {
    let fixture = load_fixture();
    assert_eq!(fixture.edge_cases.len(), 6, "5 演算（CELU は 2 alpha）");
    for e in &fixture.edge_cases {
        let x: Vec<f32> = e.x_bits.iter().map(|&b| f32::from_bits(b)).collect();
        let n = x.len();
        let ones = vec![1.0; n];
        let (out, dx) = run(&e.op, e.alpha, &x, &[n], &ones);
        for (i, &o) in out.iter().enumerate() {
            assert_class(o, &e.out[i], &format!("{} forward x={:?}", e.name, x[i]));
        }
        for (i, &d) in dx.iter().enumerate() {
            let ctx = format!("{} grad x={:?}", e.name, x[i]);
            // PyTorch と意図的に異なる 2 項目（決定記録 §5）は自前の挙動を明示 assert する。
            //  (1) SELU／CELU の NaN 入力の勾配: PyTorch は `x <= 0` が偽のため正側の
            //      係数を返すが、本実装は NaN を伝播する（forward が NaN の位置で
            //      勾配だけ有限値を返さない）。
            //  (2) Softsign の `±inf` 入力の勾配: PyTorch は合成 autograd で NaN だが、
            //      本実装は `f64` の係数 `1 / (1 + |x|)^2 = 0` を返す。
            if x[i].is_nan() && (e.op == "selu" || e.op == "celu") {
                assert!(d.is_nan(), "{ctx}: NaN 入力の勾配は NaN（意図的差分）");
            } else if x[i].is_infinite() && e.op == "softsign" {
                assert_eq!(d, 0.0, "{ctx}: 係数は 0（意図的差分）");
            } else {
                assert_class(d, &e.grad_x[i], &ctx);
            }
        }
    }
}

#[test]
fn celu_invalid_alpha_is_rejected_and_pytorch_behavior_is_recorded() {
    let fixture = load_fixture();
    assert_eq!(fixture.error_cases.len(), 4);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-1.0, 0.5], &[2]));
    for e in &fixture.error_cases {
        let alpha = match e.alpha_class.as_str() {
            "zero" => 0.0,
            "nan" => f32::NAN,
            "pos_inf" => f32::INFINITY,
            "neg_inf" => f32::NEG_INFINITY,
            other => panic!("未知の alpha_class: {other}"),
        };
        // 本実装は 4 種すべてを入口で拒否する。PyTorch は `alpha == 0` のみ例外
        // （RuntimeError）で、非有限 alpha は例外なしで NaN を返す（意図的差分）。
        assert!(
            matches!(celu(&x, alpha), Err(AutodiffError::InvalidArgument(_))),
            "{}: 拒否される",
            e.name
        );
        if e.alpha_class == "zero" {
            assert!(e.raised, "{}: PyTorch も拒否", e.name);
            assert_eq!(e.exception_type.as_deref(), Some("RuntimeError"));
        } else {
            assert!(!e.raised, "{}: PyTorch は例外なし", e.name);
            assert_class(f32::NAN, &e.out[0], &e.name);
        }
    }
}

fn f64_ref(op: &str, alpha: f64, x: f64) -> f64 {
    match op {
        "selu" => {
            let (a, s) = (1.673_263_242_354_377_2_f64, 1.050_700_987_355_480_5_f64);
            if x > 0.0 { s * x } else { s * a * x.exp_m1() }
        }
        "celu" => {
            if x > 0.0 {
                x
            } else {
                alpha * (x / alpha).exp_m1()
            }
        }
        "softsign" => x / (1.0 + x.abs()),
        "hardsigmoid" => ((x + 3.0).clamp(0.0, 6.0)) / 6.0,
        "log_sigmoid" => -(1.0 + (-x).exp()).ln(),
        other => panic!("未知の op: {other}"),
    }
}

#[test]
fn backward_matches_central_difference() {
    let h = 1e-5_f64;
    for op in OPS {
        // kink（`0`・`±3`）を避ける。
        let xs: Vec<f32> = if op == "hardsigmoid" {
            vec![-4.0, -1.0, 1.0, 4.0]
        } else {
            vec![-2.3, -0.6, 0.9, 1.7]
        };
        let alpha = 1.3_f32;
        let g = vec![1.0_f32; xs.len()];
        let (_, dx) = run(op, Some(alpha), &xs, &[xs.len()], &g);
        for (i, &x) in xs.iter().enumerate() {
            let x = f64::from(x);
            let a = f64::from(alpha);
            let numeric = (f64_ref(op, a, x + h) - f64_ref(op, a, x - h)) / (2.0 * h);
            assert!(
                common::req2_close(f64::from(dx[i]), numeric),
                "{op}[{i}] analytic={} numeric={numeric}",
                dx[i]
            );
        }
    }
}

#[test]
fn forward_matches_apply_bit_for_bit() {
    let data = vec![
        -9.0_f32,
        -3.0,
        -0.5,
        0.0,
        -0.0,
        0.5,
        3.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(data.clone(), &[data.len()]));
    for op in OPS {
        let got = apply_op(op, Some(1.7), &x).unwrap().to_tensor();
        let k = kind(op, Some(1.7));
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(
                got.host_slice()[i].to_bits(),
                k.apply(v).to_bits(),
                "{op}({v})"
            );
        }
    }
}

#[test]
fn hardsigmoid_backward_is_not_polluted_by_non_finite_upstream() {
    // 領域外・境界・NaN 入力は上流が inf でも勾配が厳密に 0（要素選択）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-5.0, -3.0, 0.5, 3.0, 5.0, f32::NAN], &[6]));
    let y = hardsigmoid(&x).unwrap();
    let inf = tape.var_no_grad(&t(vec![f32::INFINITY; 6], &[6]));
    let loss = y.mul(&inf).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("勾配").clone();
    let d = dx.host_slice();
    for i in [0usize, 1, 3, 4, 5] {
        assert_eq!(d[i], 0.0, "i={i}");
    }
    assert_eq!(d[2], f32::INFINITY);
}

#[test]
fn forward_preserves_shape_views_and_empty_tensors() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((0..6).map(|v| v as f32 - 2.5).collect(), &[2, 3]));
    for op in OPS {
        assert_eq!(
            apply_op(op, Some(1.0), &x).unwrap().to_tensor().shape(),
            &[2, 3]
        );
    }
    // 非連続 view（transpose／narrow）でも値は連続入力と一致する。
    let tr = x.transpose(0, 1).unwrap();
    let nr = x.narrow(1, 1, 2).unwrap();
    for op in OPS {
        let direct = |v: &Var<'_>| {
            let host = v.to_tensor().host_slice().into_owned();
            let k = kind(op, Some(1.0));
            let y = apply_op(op, Some(1.0), v).unwrap().to_tensor();
            let got = y.host_slice().into_owned();
            // `host_slice` は論理順の連続値を返す前提で、論理順に `apply` と一致する。
            for (a, h) in got.iter().zip(&host) {
                assert_eq!(a.to_bits(), k.apply(*h).to_bits(), "{op}");
            }
        };
        direct(&tr);
        direct(&nr);
    }
    // 空テンソル。
    let empty = tape.var(&t(vec![], &[0]));
    for op in OPS {
        assert_eq!(
            apply_op(op, Some(1.0), &empty).unwrap().to_tensor().shape(),
            &[0]
        );
    }
}

// --- Unsupported 以外のバックエンドエラーはフォールバックしない ---

/// `scalar_unary` が指定のエラーを返すスタブ。それ以外は naive 実装へ委譲する。
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
    for op in OPS {
        let xs = vec![-0.5_f32, 0.25];
        let x = tape.var(&t(xs.clone(), &[2]));
        let y = apply_op(op, Some(1.5), &x).unwrap();
        let out = y.to_tensor().host_slice().into_owned();
        for (i, &v) in xs.iter().enumerate() {
            assert_eq!(
                out[i].to_bits(),
                kind(op, Some(1.5)).apply(v).to_bits(),
                "{op}"
            );
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
}

#[test]
fn non_unsupported_backend_error_is_propagated() {
    let tape = err_tape(|| BackendError::KernelLaunchFailed("boom".into()));
    let x = tape.var(&t(vec![0.1, 0.2], &[2]));
    for op in OPS {
        assert!(
            matches!(
                apply_op(op, Some(1.0), &x),
                Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
            ),
            "{op}: エラーを握りつぶさない"
        );
    }
}
