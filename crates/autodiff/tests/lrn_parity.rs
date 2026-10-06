//! `lrn_ops::local_response_norm`（イシュー #2646）の `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/lrn-weight-reparam-pytorch-reference/lrn_weight_reparam_reference.json`・
//!   生成条件は同ディレクトリの `README.md`）と突合する。f32 は NaN／inf を運べるよう u32 ビット
//!   パターンで保存されている。forward・入力勾配は REQ-2 統一複合判定（`common::req2_close`。
//!   tolerance 定数は新設しない。PyTorch は f32 累積・本実装は `f64` 累積のため bit 一致は求めない）。
//! - `common::naive_ops()` は `lrn_forward` を override しないため、必ず共有ホストカーネルへの
//!   フォールバック経路を通る。CPU `BackendOps` 実装との一致は
//!   `crates/facade/tests/lrn_weight_reparam_backend_parity.rs` が担当する。
//! - フォールバックは `Unsupported` のときだけで、それ以外のバックエンドエラーは握りつぶさず
//!   伝播する。確保前サイズ検査・テープ記録数・孤児ノード無し・VJP の独立オラクル（中心差分・
//!   手計算・偶数 `size` の非対称窓）・run-to-run 決定性も固定する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::lrn_ops::local_response_norm;
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::lrn::LrnParams;
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
    lrn_cases: Vec<Case>,
    lrn_nonfinite_cases: Vec<Case>,
    lrn_error_cases: Vec<ErrCase>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    shape: Vec<usize>,
    size: usize,
    alpha_bits: u32,
    beta_bits: u32,
    k_bits: u32,
    x_bits: Vec<u32>,
    g_bits: Vec<u32>,
    out_bits: Vec<u32>,
    grad_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct ErrCase {
    name: String,
    shape: Vec<usize>,
    size: usize,
    alpha_bits: u32,
    beta_bits: u32,
    k_bits: u32,
    torch_raises: bool,
    out_shape: Vec<usize>,
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

struct Run {
    values: Vec<f32>,
    out_shape: Vec<usize>,
    grad: Vec<f32>,
}

/// `(out * g).sum()` を損失とした forward 値・入力勾配を返す。
fn run_case(ops: Box<dyn BackendOps + Send>, case: &Case) -> Run {
    let tape = Tape::new_with_ops(ops);
    let xv = tape.var(&t(from_bits(&case.x_bits), &case.shape));
    let y = local_response_norm(
        &xv,
        case.size,
        f32::from_bits(case.alpha_bits),
        f32::from_bits(case.beta_bits),
        f32::from_bits(case.k_bits),
    )
    .expect("local_response_norm");
    let out = y.to_tensor();
    let gv = tape.var_no_grad(&t(from_bits(&case.g_bits), out.shape()));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("入力へ勾配が届く").clone();
    Run {
        values: out.host_slice().into_owned(),
        out_shape: out.shape().to_vec(),
        grad: dx.host_slice().into_owned(),
    }
}

fn run_fixture_case(case: &Case) -> Run {
    run_case(common::naive_ops(), case)
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
    assert!(fixture.lrn_cases.len() >= 20);
    for case in &fixture.lrn_cases {
        let r = run_fixture_case(case);
        assert_eq!(r.out_shape, case.shape, "{}: 出力 shape", case.name);
        assert_close_all(
            &r.values,
            &from_bits(&case.out_bits),
            &format!("{} forward", case.name),
        );
        assert_close_all(
            &r.grad,
            &from_bits(&case.grad_bits),
            &format!("{} grad", case.name),
        );
    }
}

#[test]
fn nonfinite_forward_and_gradients_match_pytorch_reference() {
    let fixture = load_fixture();
    assert!(fixture.lrn_nonfinite_cases.len() >= 5);
    for case in &fixture.lrn_nonfinite_cases {
        let r = run_fixture_case(case);
        assert_close_all(
            &r.values,
            &from_bits(&case.out_bits),
            &format!("{} forward", case.name),
        );
        assert_close_all(
            &r.grad,
            &from_bits(&case.grad_bits),
            &format!("{} grad", case.name),
        );
    }
}

/// torch と一致する点・意図的差分を表で固定する（決定記録 §5）。
#[test]
fn error_cases_agree_with_torch_except_documented_differences() {
    let fixture = load_fixture();
    // (name, 本実装が拒否するか)。
    let ours_rejects = |name: &str| -> bool {
        match name {
            // torch も拒否（rank < 3）。
            "rank2" | "rank1" => true,
            // torch も拒否（`size = 0` は avg_pool が実行時エラー）。
            "size0" => true,
            // torch は受理。本実装も受理（空出力）。
            "n0" | "c0" | "spatial0" => false,
            // 意図的差分: torch は無検査で受理するが、本実装は非有限パラメータを拒否する。
            "alpha_nan" | "beta_inf" | "k_nan" | "k_neg_inf" => true,
            other => panic!("未知の error case: {other}"),
        }
    };
    let differences = ["alpha_nan", "beta_inf", "k_nan", "k_neg_inf"];
    for c in &fixture.lrn_error_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let n: usize = c.shape.iter().product();
        let x = tape.var(&t(vec![0.5; n], &c.shape));
        let r = local_response_norm(
            &x,
            c.size,
            f32::from_bits(c.alpha_bits),
            f32::from_bits(c.beta_bits),
            f32::from_bits(c.k_bits),
        );
        let rejects = ours_rejects(&c.name);
        assert_eq!(r.is_err(), rejects, "{}: 拒否の有無", c.name);
        if differences.contains(&c.name.as_str()) {
            assert!(!c.torch_raises, "{}: torch は受理するはず（差分）", c.name);
            assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))));
        } else {
            assert_eq!(c.torch_raises, rejects, "{}: torch と一致するはず", c.name);
        }
        if !c.torch_raises && !rejects {
            assert_eq!(r.unwrap().to_tensor().shape(), c.out_shape.as_slice());
        }
    }
}

// --- 手計算・独立オラクル ---

fn pseudo(n: usize, mul: usize, modulus: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * mul) % modulus) as f32 * 0.21 - 1.1)
        .collect()
}

#[test]
fn hand_computed_odd_and_even_windows() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3, 1]));
    // size=3: d_c = S_c（alpha=3・k=0・beta=1）→ c=0: {0,1}=5 / c=1: 14 / c=2: {1,2}=13。
    let y = local_response_norm(&x, 3, 3.0, 1.0, 0.0).unwrap();
    let got = y.to_tensor().host_slice().into_owned();
    for (a, b) in got.iter().zip([1.0 / 5.0, 2.0 / 14.0, 3.0 / 13.0]) {
        assert!((a - b).abs() < 1e-6);
    }
    // size=2（偶数）: W(c) = [c-1, c]（前 1・後 0）→ c=0: 1 / c=1: 5 / c=2: 13。
    let y = local_response_norm(&x, 2, 2.0, 1.0, 0.0).unwrap();
    let got = y.to_tensor().host_slice().into_owned();
    for (a, b) in got.iter().zip([1.0, 2.0 / 5.0, 3.0 / 13.0]) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn even_size_gradient_uses_the_reversed_window() {
    // size=2・k=1・alpha=2・beta=1: y_c = x_c / (1 + S_c)、S_c = x_{c-1}² + x_c²。
    let xs = [0.5_f32, -1.0, 2.0];
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(xs.to_vec(), &[1, 3, 1]));
    let y = local_response_norm(&x, 2, 2.0, 1.0, 1.0).unwrap();
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
    // 解析式（forward 窓 W(c) = {c-1, c}）で dL/dx_m = Σ_c ∂y_c/∂x_m を直接組む。
    let d: Vec<f64> = (0..3)
        .map(|c: usize| {
            let s: f64 = (c.saturating_sub(1)..=c)
                .map(|j| f64::from(xs[j]).powi(2))
                .sum();
            1.0 + s
        })
        .collect();
    for m in 0..3 {
        let mut want = 1.0 / d[m];
        for c in m..(m + 2).min(3) {
            // m ∈ W(c) ⇔ c ∈ {m, m+1}。
            want -= f64::from(xs[c]) * 2.0 * f64::from(xs[m]) / d[c].powi(2);
        }
        assert!(
            (f64::from(dx[m]) - want).abs() < 1e-6,
            "dx[{m}]={} want={want}",
            dx[m]
        );
    }
}

#[test]
fn gradient_matches_central_difference_for_all_window_shapes() {
    for (size, shape) in [
        (1_usize, [2_usize, 4, 3]),
        (2, [2, 5, 3]),
        (3, [2, 5, 3]),
        (4, [1, 6, 4]),
        (5, [1, 5, 2]),
        (7, [1, 3, 2]),
    ] {
        let n: usize = shape.iter().product();
        let x = pseudo(n, 7, 23);
        let g = pseudo(n, 5, 19);
        let forward = |xs: &[f32]| -> f64 {
            let tape = Tape::new_with_ops(common::naive_ops());
            let xv = tape.var(&t(xs.to_vec(), &shape));
            let y = local_response_norm(&xv, size, 0.5, 0.75, 2.0).unwrap();
            y.to_tensor()
                .host_slice()
                .iter()
                .zip(&g)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum()
        };
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.clone(), &shape));
        let y = local_response_norm(&xv, size, 0.5, 0.75, 2.0).unwrap();
        let gv = tape.var_no_grad(&t(g.clone(), &shape));
        let loss = y.mul(&gv).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&xv).unwrap().unwrap().host_slice().into_owned();
        for i in 0..n {
            let (mut p, mut m) = (x.clone(), x.clone());
            p[i] = (f64::from(x[i]) + H) as f32;
            m[i] = (f64::from(x[i]) - H) as f32;
            let num = (forward(&p) - forward(&m)) / (2.0 * H);
            let an = f64::from(dx[i]);
            let diff = (an - num).abs();
            let rel = diff / an.abs().max(num.abs()).max(TAU);
            assert!(
                rel <= REL_TOL || diff <= ABS_TOL,
                "size={size} x[{i}]: analytic={an} numeric={num}"
            );
        }
    }
}

#[test]
fn non_contiguous_input_is_normalized_in_logical_order() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let data = pseudo(2 * 3 * 4, 7, 23);
    let base = tape.var(&t(data.clone(), &[2, 3, 4]));
    let permuted = base.permute(&[0, 2, 1]).unwrap();
    let a = local_response_norm(&permuted, 3, 0.5, 0.75, 1.0).unwrap();
    let dense = tape.var(&permuted.to_tensor().contiguous());
    let b = local_response_norm(&dense, 3, 0.5, 0.75, 1.0).unwrap();
    assert_eq!(
        a.to_tensor().host_slice().into_owned(),
        b.to_tensor().host_slice().into_owned()
    );
}

#[test]
fn squares_do_not_overflow_in_f32() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![2e20, 1.0], &[1, 2, 1]));
    let y = local_response_norm(&x, 2, 1.0, 0.75, 1.0).unwrap();
    assert!(y.to_tensor().host_slice().iter().all(|v| v.is_finite()));
}

// --- フォールバックとエラー伝播 ---

/// `lrn_forward` だけを差し替える `BackendOps`。それ以外は naive へ委譲する。
struct LrnMock {
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

impl BackendOps for LrnMock {
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
    fn lrn_forward(
        &self,
        _input: &Tensor<f32>,
        _params: &LrnParams,
    ) -> Result<Tensor<f32>, BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongShape => Ok(t(vec![0.0; 2], &[2])),
        }
    }
}

fn mock_tape(mode: Mode) -> (Tape, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(LrnMock {
        inner: common::naive_ops(),
        calls: Arc::clone(&calls),
        mode,
    }));
    (tape, calls)
}

fn small(tape: &Tape) -> Var<'_> {
    tape.var(&t((0..6).map(|i| i as f32).collect(), &[1, 3, 2]))
}

#[test]
fn unsupported_backend_falls_back_to_host_reference() {
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let x = small(&tape);
    let y = local_response_norm(&x, 3, 0.5, 0.75, 1.0).unwrap();
    let (naive_tape, _) = (Tape::new_with_ops(common::naive_ops()), ());
    let nx = small(&naive_tape);
    let ny = local_response_norm(&nx, 3, 0.5, 0.75, 1.0).unwrap();
    assert_eq!(
        y.to_tensor().host_slice().into_owned(),
        ny.to_tensor().host_slice().into_owned()
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "バックエンドを先に呼んでからフォールバックしているはず"
    );
}

#[test]
fn non_unsupported_backend_errors_are_propagated() {
    let (tape, _) = mock_tape(Mode::LaunchFailed);
    let x = small(&tape);
    assert!(matches!(
        local_response_norm(&x, 3, 0.5, 0.75, 1.0),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
}

#[test]
fn wrong_shape_from_backend_is_a_typed_error() {
    let (tape, _) = mock_tape(Mode::WrongShape);
    let x = small(&tape);
    let before = tape.len();
    assert!(matches!(
        local_response_norm(&x, 3, 0.5, 0.75, 1.0),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
    assert_eq!(tape.len(), before, "孤児ノードを残さない");
}

// --- 境界・テープ・決定性 ---

#[test]
fn invalid_arguments_are_typed_errors_and_leave_no_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = small(&tape);
    let rank2 = tape.var(&t(vec![0.0; 6], &[2, 3]));
    let before = tape.len();
    assert!(matches!(
        local_response_norm(&x, 0, 0.5, 0.75, 1.0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    for (a, b, k) in [
        (f32::NAN, 0.75, 1.0),
        (0.5, f32::INFINITY, 1.0),
        (0.5, 0.75, f32::NEG_INFINITY),
    ] {
        assert!(matches!(
            local_response_norm(&x, 3, a, b, k),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
    assert!(matches!(
        local_response_norm(&rank2, 3, 0.5, 0.75, 1.0),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert_eq!(tape.len(), before);
}

#[test]
fn empty_tensors_yield_empty_outputs() {
    let tape = Tape::new_with_ops(common::naive_ops());
    for shape in [[0_usize, 3, 2], [2, 0, 2], [2, 3, 0]] {
        let x = tape.var(&t(vec![], &shape));
        let y = local_response_norm(&x, 3, 0.5, 0.75, 1.0).unwrap();
        assert_eq!(y.to_tensor().shape(), &shape);
    }
}

#[test]
fn huge_broadcast_view_is_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(vec![1.0], &[1, 1, 1]));
    let huge = base.broadcast_to(&[1, 1, 1usize << 61]).unwrap();
    assert!(matches!(
        local_response_norm(&huge, 3, 0.5, 0.75, 1.0),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
}

#[test]
fn each_call_records_exactly_one_node_and_no_grad_input_is_supported() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = small(&tape);
    let before = tape.len();
    local_response_norm(&x, 3, 0.5, 0.75, 1.0).unwrap();
    assert_eq!(tape.len(), before + 1);
    let ng = tape.var_no_grad(&t(vec![1.0; 6], &[1, 3, 2]));
    let y = local_response_norm(&ng, 3, 0.5, 0.75, 1.0).unwrap();
    assert_eq!(y.to_tensor().shape(), &[1, 3, 2]);
}

#[test]
fn create_graph_is_a_typed_error_not_a_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = small(&tape);
    let loss = local_response_norm(&x, 3, 0.5, 0.75, 1.0)
        .unwrap()
        .sum(None)
        .unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
}

#[test]
fn results_are_bit_deterministic_run_to_run() {
    let fixture = load_fixture();
    for case in fixture
        .lrn_cases
        .iter()
        .filter(|c| c.name.contains("even") || c.name.contains("r4"))
    {
        let a = run_fixture_case(case);
        let b = run_fixture_case(case);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.values), bits(&b.values), "{}", case.name);
        assert_eq!(bits(&a.grad), bits(&b.grad), "{}", case.name);
    }
}
