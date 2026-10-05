//! `cumulative_ops`（イシュー #2636・`cummax`／`cummin`／`logcumsumexp`）の
//! `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/cumulative-pytorch-reference/cumulative_reference.json`・
//!   生成条件は同ディレクトリの `README.md`）と突合する。f32 は NaN／inf を運べる
//!   よう u32 ビットパターンで保存されている。`cummax`／`cummin` の値は選択のみ
//!   なので bit 一致（NaN はクラス一致）・索引は完全一致、勾配と `logcumsumexp`
//!   は REQ-2 統一複合判定（`common::req2_close`。tolerance 定数は新設しない）。
//! - `common::naive_ops()` は `scan_*` を override しないため、必ず共有ホスト
//!   カーネルへのフォールバック経路を通る。CPU `BackendOps` 実装との一致は
//!   `crates/facade/tests/cumulative_ops_backend_parity.rs` が担当する。
//! - フォールバックは `Unsupported` のときだけで、それ以外のバックエンドエラーは
//!   握りつぶさず伝播する。確保前サイズ検査・テープ記録数・VJP の独立オラクル
//!   （O(n²) 直接形・中心差分・手計算）・run-to-run 決定性も固定する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::cumulative_ops::{cummax, cummin, logcumsumexp};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};
use serde::Deserialize;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    finite_cases: Vec<Case>,
    nonfinite_cases: Vec<Case>,
    error_cases: Vec<ErrCase>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    op: String,
    shape: Vec<usize>,
    dim: usize,
    x_bits: Vec<u32>,
    g_bits: Vec<u32>,
    out_bits: Vec<u32>,
    grad_bits: Vec<u32>,
    index: Option<Vec<i32>>,
}

#[derive(Deserialize)]
struct ErrCase {
    name: String,
    shape: Vec<usize>,
    dim: i64,
    torch_raises: bool,
    out_shape: Vec<usize>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/cumulative-pytorch-reference/cumulative_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

/// NaN は NaN 同士（クラス一致）、それ以外は bit 完全一致。
fn assert_class_or_bits_eq(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{context}[{i}]: NaN のはず（actual={a}）");
        } else {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{context}[{i}]: actual={a} expected={e}"
            );
        }
    }
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

struct Run {
    values: Vec<f32>,
    index: Option<Vec<i32>>,
    grad: Vec<f32>,
}

fn apply<'t>(op: &str, x: &Var<'t>, dim: usize) -> (Var<'t>, Option<Tensor<i32>>) {
    match op {
        "cummax" => {
            let (v, i) = cummax(x, dim).expect("cummax");
            (v, Some(i))
        }
        "cummin" => {
            let (v, i) = cummin(x, dim).expect("cummin");
            (v, Some(i))
        }
        "logcumsumexp" => (logcumsumexp(x, dim).expect("logcumsumexp"), None),
        other => panic!("未知の op: {other}"),
    }
}

/// `(out * g).sum()` を損失とした forward 値・索引・入力勾配を返す。
fn run_case(
    ops: Box<dyn BackendOps + Send>,
    op: &str,
    x: Vec<f32>,
    shape: &[usize],
    dim: usize,
    g: Vec<f32>,
) -> Run {
    let tape = Tape::new_with_ops(ops);
    let xv = tape.var(&t(x, shape));
    let (y, idx) = apply(op, &xv, dim);
    let values = y.to_tensor().host_slice().into_owned();
    let gv = tape.var_no_grad(&t(g, shape));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("入力へ勾配が届く").clone();
    Run {
        values,
        index: idx.map(|i| i.contiguous().host_slice().into_owned()),
        grad: dx.host_slice().into_owned(),
    }
}

fn run_fixture_case(case: &Case) -> Run {
    run_case(
        common::naive_ops(),
        &case.op,
        from_bits(&case.x_bits),
        &case.shape,
        case.dim,
        from_bits(&case.g_bits),
    )
}

// --- PyTorch fixture 突合 ---

#[test]
fn finite_cases_match_pytorch_reference_forward_and_backward() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    assert!(fixture.finite_cases.len() >= 49);
    for case in &fixture.finite_cases {
        let r = run_fixture_case(case);
        let want = from_bits(&case.out_bits);
        if case.op == "logcumsumexp" {
            assert_close_all(&r.values, &want, &format!("{} forward", case.name));
        } else {
            // 選択のみの演算なので値は bit 一致（±0 の符号も含む）。
            assert_class_or_bits_eq(&r.values, &want, &format!("{} forward", case.name));
            assert_eq!(
                r.index.as_deref(),
                case.index.as_deref(),
                "{}: 索引",
                case.name
            );
        }
        assert_close_all(
            &r.grad,
            &from_bits(&case.grad_bits),
            &format!("{} grad", case.name),
        );
    }
}

#[test]
fn nonfinite_forward_matches_pytorch_reference() {
    let fixture = load_fixture();
    assert!(fixture.nonfinite_cases.len() >= 39);
    for case in &fixture.nonfinite_cases {
        let r = run_fixture_case(case);
        let want = from_bits(&case.out_bits);
        if case.op == "logcumsumexp" {
            assert_close_all(&r.values, &want, &format!("{} forward", case.name));
        } else {
            assert_class_or_bits_eq(&r.values, &want, &format!("{} forward", case.name));
            assert_eq!(
                r.index.as_deref(),
                case.index.as_deref(),
                "{}: 索引（タイ後勝ち・NaN 伝播規則）",
                case.name
            );
        }
    }
}

#[test]
fn nonfinite_extremum_gradients_match_pytorch_reference() {
    let fixture = load_fixture();
    for case in fixture
        .nonfinite_cases
        .iter()
        .filter(|c| c.op != "logcumsumexp")
    {
        let r = run_fixture_case(case);
        assert_close_all(
            &r.grad,
            &from_bits(&case.grad_bits),
            &format!("{} grad", case.name),
        );
    }
}

/// 非有限入力の `logcumsumexp` 勾配は式の IEEE 伝播に従う（拒否・panic しない）。
/// PyTorch との一致数は `docs/autodiff-cumulative-ops-decision.md` §5 に記録して
/// おり、ここでは「勾配が計算できる」ことと、有限 lane の一致だけを固定する。
#[test]
fn nonfinite_logcumsumexp_gradient_is_computed_without_panic() {
    let fixture = load_fixture();
    let mut compared = 0;
    let mut mismatched = 0;
    for case in fixture
        .nonfinite_cases
        .iter()
        .filter(|c| c.op == "logcumsumexp")
    {
        let r = run_fixture_case(case);
        assert_eq!(r.grad.len(), case.x_bits.len(), "{}", case.name);
        let want = from_bits(&case.grad_bits);
        if r.grad.iter().zip(&want).all(|(&a, &e)| class_close(a, e)) {
            compared += 1;
        } else {
            // 不一致は +inf を含む lane に限る（`x - out = inf - inf = NaN`。
            // 決定記録 §5 の実測差分）。それ以外の非有限入力は一致する。
            assert!(
                from_bits(&case.x_bits).contains(&f32::INFINITY),
                "{}: +inf を含まない lane は PyTorch と一致するはず",
                case.name
            );
            mismatched += 1;
        }
    }
    assert_eq!(compared, 9, "PyTorch と一致する非有限ケース数（実測）");
    assert_eq!(mismatched, 4, "+inf lane の実測差分ケース数");
}

#[test]
fn error_cases_agree_with_torch_where_in_scope() {
    let fixture = load_fixture();
    for case in &fixture.error_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let n: usize = case.shape.iter().product();
        let x = tape.var(&t(vec![0.0; n], &case.shape));
        let Ok(dim) = usize::try_from(case.dim) else {
            continue; // 負の dim は受けない（決定記録 §5）
        };
        let r = logcumsumexp(&x, dim);
        if case.torch_raises {
            assert!(
                matches!(
                    r,
                    Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
                ),
                "{}",
                case.name
            );
        } else if case.shape.is_empty() {
            // 0 次元入力は torch が受理するが本実装は rank 0 を拒否する（差分）。
            assert!(
                matches!(
                    r,
                    Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
                ),
                "{}",
                case.name
            );
        } else {
            assert_eq!(
                r.unwrap().to_tensor().shape(),
                case.out_shape.as_slice(),
                "{}",
                case.name
            );
        }
    }
}

// --- 独立オラクル（VJP） ---

fn lse_oracle_grad(x: &[f64], g: &[f64]) -> Vec<f64> {
    let n = x.len();
    let out: Vec<f64> = (0..n)
        .map(|k| x[..=k].iter().map(|v| v.exp()).sum::<f64>().ln())
        .collect();
    (0..n)
        .map(|j| (j..n).map(|i| g[i] * (x[j] - out[i]).exp()).sum())
        .collect()
}

#[test]
fn logcumsumexp_gradient_matches_quadratic_oracle_and_central_difference() {
    let x = [0.7_f32, -1.3, 2.2, 0.1, -0.4, 1.9];
    let g = [1.0_f32, -0.5, 0.25, 2.0, -1.5, 0.75];
    let r = run_case(
        common::naive_ops(),
        "logcumsumexp",
        x.to_vec(),
        &[6],
        0,
        g.to_vec(),
    );
    let xd: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let gd: Vec<f64> = g.iter().map(|&v| f64::from(v)).collect();
    let want = lse_oracle_grad(&xd, &gd);
    for (j, (&got, &w)) in r.grad.iter().zip(&want).enumerate() {
        assert!(
            common::req2_close(f64::from(got), w),
            "oracle x[{j}]: {got} vs {w}"
        );
    }
    let loss = |xs: &[f64]| -> f64 {
        (0..6)
            .map(|k| xs[..=k].iter().map(|v| v.exp()).sum::<f64>().ln() * gd[k])
            .sum()
    };
    let h = 1e-5;
    for j in 0..6 {
        let (mut hi, mut lo) = (xd.clone(), xd.clone());
        hi[j] += h;
        lo[j] -= h;
        let fd = (loss(&hi) - loss(&lo)) / (2.0 * h);
        assert!(
            common::req2_close(f64::from(r.grad[j]), fd),
            "fd x[{j}]: {} vs {fd}",
            r.grad[j]
        );
    }
}

#[test]
fn logcumsumexp_large_amplitude_gradient_stays_finite() {
    let x = vec![500.0_f32, -500.0, 480.0, 510.0, -30.0];
    let g = vec![1.0_f32; 5];
    let r = run_case(common::naive_ops(), "logcumsumexp", x, &[5], 0, g);
    assert!(r.values.iter().all(|v| v.is_finite()));
    assert!(r.grad.iter().all(|v| v.is_finite()));
}

#[test]
fn cummax_gradient_accumulates_on_duplicate_indices() {
    // 索引 [0,0,0]: 全上流が入力 0 番へ加算される（Overwrite だと 3.0 になり誤り）。
    let r = run_case(
        common::naive_ops(),
        "cummax",
        vec![3.0, 1.0, 2.0],
        &[3],
        0,
        vec![1.0, 2.0, 3.0],
    );
    assert_eq!(r.index.as_deref(), Some(&[0, 0, 0][..]));
    assert_eq!(r.grad, vec![6.0, 0.0, 0.0]);
    let r = run_case(
        common::naive_ops(),
        "cummin",
        vec![1.0, 3.0, 2.0, 0.5],
        &[4],
        0,
        vec![1.0, 2.0, 3.0, 4.0],
    );
    assert_eq!(r.index.as_deref(), Some(&[0, 0, 0, 3][..]));
    assert_eq!(r.grad, vec![6.0, 0.0, 0.0, 4.0]);
}

#[test]
fn extremum_gradient_matches_central_difference_without_ties() {
    let x0 = [0.3_f32, 1.7, -0.5, 2.4, 0.9, 3.1];
    let w = [1.0_f32, -0.5, 0.25, 2.0, -1.5, 0.75];
    for op in ["cummax", "cummin"] {
        let r = run_case(common::naive_ops(), op, x0.to_vec(), &[6], 0, w.to_vec());
        let loss = |xs: &[f64]| -> f64 {
            let mut cur = xs[0];
            let mut total = f64::from(w[0]) * cur;
            for k in 1..6 {
                cur = if op == "cummax" {
                    cur.max(xs[k])
                } else {
                    cur.min(xs[k])
                };
                total += f64::from(w[k]) * cur;
            }
            total
        };
        let base: Vec<f64> = x0.iter().map(|&v| f64::from(v)).collect();
        let h = 1e-4;
        for j in 0..6 {
            let (mut hi, mut lo) = (base.clone(), base.clone());
            hi[j] += h;
            lo[j] -= h;
            let fd = (loss(&hi) - loss(&lo)) / (2.0 * h);
            assert!(
                common::req2_close(f64::from(r.grad[j]), fd),
                "{op} x[{j}]: {} vs {fd}",
                r.grad[j]
            );
        }
    }
}

// --- フォールバックとエラー伝播 ---

/// `scan_*` だけを差し替える `BackendOps`。それ以外は naive へ委譲する。
struct ScanMock {
    inner: Box<dyn BackendOps + Send>,
    calls: Arc<AtomicUsize>,
    mode: Mode,
}

#[derive(Clone, Copy)]
enum Mode {
    Unsupported,
    LaunchFailed,
    WrongValueShape,
    WrongIndexShape,
}

impl ScanMock {
    fn bad(&self) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongValueShape => Ok((
                t(vec![0.0; 3], &[3]),
                Tensor::new(vec![0_i32; 4], &[4]).unwrap(),
            )),
            Mode::WrongIndexShape => Ok((
                t(vec![0.0; 4], &[4]),
                Tensor::new(vec![0_i32; 3], &[3]).unwrap(),
            )),
        }
    }
}

impl BackendOps for ScanMock {
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
    fn scan_cummax(
        &self,
        _x: &Tensor<f32>,
        _dim: usize,
    ) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        self.bad()
    }
    fn scan_cummin(
        &self,
        _x: &Tensor<f32>,
        _dim: usize,
    ) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        self.bad()
    }
    fn scan_logcumsumexp(
        &self,
        _x: &Tensor<f32>,
        _dim: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        self.bad().map(|(v, _)| v)
    }
}

fn mock_tape(mode: Mode) -> (Tape, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(ScanMock {
        inner: common::naive_ops(),
        calls: Arc::clone(&calls),
        mode,
    }));
    (tape, calls)
}

#[test]
fn unsupported_backend_falls_back_to_host_reference() {
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let x = tape.var(&t(vec![3.0, 1.0, 5.0, 2.0], &[4]));
    let (v, i) = cummax(&x, 0).unwrap();
    assert_eq!(
        v.to_tensor().host_slice().into_owned(),
        [3.0, 3.0, 5.0, 5.0]
    );
    assert_eq!(i.contiguous().host_slice().into_owned(), [0, 0, 2, 2]);
    let (v, i) = cummin(&x, 0).unwrap();
    assert_eq!(
        v.to_tensor().host_slice().into_owned(),
        [3.0, 1.0, 1.0, 1.0]
    );
    assert_eq!(i.contiguous().host_slice().into_owned(), [0, 1, 1, 1]);
    let l = logcumsumexp(&x, 0).unwrap();
    assert!((l.to_tensor().host_slice()[1] - (3.0_f32.exp() + 1.0_f32.exp()).ln()).abs() < 1e-6);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "バックエンドを先に呼んでからフォールバックしているはず"
    );
}

#[test]
fn non_unsupported_backend_errors_are_propagated() {
    let (tape, _) = mock_tape(Mode::LaunchFailed);
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let rs = [
        cummax(&x, 0).map(|_| ()),
        cummin(&x, 0).map(|_| ()),
        logcumsumexp(&x, 0).map(|_| ()),
    ];
    for r in rs {
        assert!(matches!(
            r,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ));
    }
}

#[test]
fn wrong_shape_from_backend_is_a_typed_error() {
    for mode in [Mode::WrongValueShape, Mode::WrongIndexShape] {
        let (tape, _) = mock_tape(mode);
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        assert!(matches!(
            cummax(&x, 0),
            Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
        ));
        assert!(matches!(
            cummin(&x, 0),
            Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
        ));
    }
    let (tape, _) = mock_tape(Mode::WrongValueShape);
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    assert!(matches!(
        logcumsumexp(&x, 0),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
}

// --- 境界・テープ・決定性 ---

#[test]
fn dim_out_of_range_and_rank0_are_typed_errors() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        cummax(&x, 1),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            axis: 1,
            rank: 1
        }))
    ));
    assert!(matches!(
        cummin(&x, 7),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
    ));
    let s = tape.var(&t(vec![1.0], &[]));
    assert!(matches!(
        logcumsumexp(&s, 0),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            rank: 0,
            ..
        }))
    ));
}

#[test]
fn empty_axis_yields_empty_outputs() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![], &[0]));
    let (v, i) = cummax(&x, 0).unwrap();
    assert_eq!(v.to_tensor().shape(), &[0]);
    assert_eq!(i.shape(), &[0]);
    assert_eq!(logcumsumexp(&x, 0).unwrap().to_tensor().shape(), &[0]);
}

#[test]
fn huge_broadcast_view_is_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(vec![1.0], &[1]));
    let huge = base.broadcast_to(&[1usize << 61]).unwrap();
    for r in [
        cummax(&huge, 0).map(|_| ()),
        cummin(&huge, 0).map(|_| ()),
        logcumsumexp(&huge, 0).map(|_| ()),
    ] {
        assert!(matches!(
            r,
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
    }
}

#[test]
fn each_op_records_exactly_one_node() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 3.0, 2.0], &[3]));
    let before = tape.len();
    cummax(&x, 0).unwrap();
    assert_eq!(tape.len(), before + 1);
    cummin(&x, 0).unwrap();
    assert_eq!(tape.len(), before + 2);
    logcumsumexp(&x, 0).unwrap();
    assert_eq!(tape.len(), before + 3);
}

#[test]
fn create_graph_is_a_typed_error_not_a_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let loss = logcumsumexp(&x, 0).unwrap().sum(None).unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
}

#[test]
fn results_are_bit_deterministic_run_to_run() {
    let fixture = load_fixture();
    for case in fixture
        .finite_cases
        .iter()
        .filter(|c| c.name.contains("rank3"))
    {
        let a = run_fixture_case(case);
        let b = run_fixture_case(case);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.values), bits(&b.values), "{}", case.name);
        assert_eq!(bits(&a.grad), bits(&b.grad), "{}", case.name);
        assert_eq!(a.index, b.index, "{}", case.name);
    }
}
