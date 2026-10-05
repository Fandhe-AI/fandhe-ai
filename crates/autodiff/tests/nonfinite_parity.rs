//! `nonfinite_ops`（イシュー #2635・`isnan`／`isinf`／`isfinite`／`nan_to_num`）の
//! `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/nonfinite-pytorch-reference/nonfinite_reference.json`・
//!   生成条件は同ディレクトリの `README.md`）と突合する。f32 は NaN／inf を
//!   運べるよう u32 ビットパターンで保存されている。判定 3 種は bool 完全一致、
//!   `nan_to_num` の forward は要素選択のみの演算なので bit 一致（NaN はクラス
//!   一致）、勾配は REQ-2 統一複合判定（`common::req2_close`）で突合する
//!   （tolerance 定数は新設しない。PyTorch は `g * 0` で `-0.0` を出しうるため
//!   勾配の bit 一致は要求しない）。
//! - `common::naive_ops()` は `scalar_unary` が既定 `Unsupported` のため、必ず
//!   ホスト参照実装へのフォールバック経路を通る。CPU `BackendOps` 実装との
//!   一致は `crates/facade/tests/nonfinite_ops_backend_parity.rs` が担当する。
//! - フォールバックは `Unsupported` のときだけで、それ以外のバックエンド
//!   エラーは握りつぶさず伝播する。確保前サイズ検査・view 入力・非微分・
//!   `create_graph` 非対応・run-to-run 決定性も固定する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::nonfinite_ops::{isfinite, isinf, isnan, nan_to_num};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, ScalarUnaryOp, ShapeError, Tensor};
use serde::Deserialize;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn bools(x: &Tensor<bool>) -> Vec<bool> {
    x.contiguous().host_slice().into_owned()
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    predicate_cases: Vec<PredCase>,
    nan_to_num_cases: Vec<N2nCase>,
    upstream_nonfinite_observations: Vec<ObsCase>,
}

#[derive(Deserialize)]
struct PredCase {
    name: String,
    shape: Vec<usize>,
    x_bits: Vec<u32>,
    isnan: Vec<bool>,
    isinf: Vec<bool>,
    isfinite: Vec<bool>,
}

#[derive(Deserialize)]
struct N2nCase {
    name: String,
    shape: Vec<usize>,
    x_bits: Vec<u32>,
    nan_bits: Option<u32>,
    posinf_bits: Option<u32>,
    neginf_bits: Option<u32>,
    out_bits: Vec<u32>,
    g_bits: Vec<u32>,
    grad_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct ObsCase {
    name: String,
    shape: Vec<usize>,
    x_bits: Vec<u32>,
    g_bits: Vec<u32>,
    grad_bits: Vec<u32>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/nonfinite-pytorch-reference/nonfinite_reference.json");
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

fn assert_close_all(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            common::req2_close(f64::from(a), f64::from(e)),
            "{context}[{i}]: actual={a} expected={e}"
        );
    }
}

fn args_of(case: &N2nCase) -> (Option<f32>, Option<f32>, Option<f32>) {
    (
        case.nan_bits.map(f32::from_bits),
        case.posinf_bits.map(f32::from_bits),
        case.neginf_bits.map(f32::from_bits),
    )
}

/// `(out * g).sum()` を損失とした forward 値と入力勾配を返す。
fn run_n2n(
    x_data: Vec<f32>,
    shape: &[usize],
    g_data: Vec<f32>,
    args: (Option<f32>, Option<f32>, Option<f32>),
) -> (Vec<f32>, Vec<f32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x_data, shape));
    let y = nan_to_num(&x, args.0, args.1, args.2).expect("nan_to_num forward");
    let out = y.to_tensor().host_slice().into_owned();
    let g = tape.var_no_grad(&t(g_data, shape));
    let loss = y.mul(&g).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("入力へ勾配が届く").clone();
    (out, dx.host_slice().into_owned())
}

// --- PyTorch fixture 突合 ---

#[test]
fn predicates_match_pytorch_reference() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    assert!(!fixture.predicate_cases.is_empty());
    for case in &fixture.predicate_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(from_bits(&case.x_bits), &case.shape));
        let n = isnan(&x).unwrap();
        assert_eq!(n.shape(), case.shape.as_slice(), "{}: shape", case.name);
        assert_eq!(bools(&n), case.isnan, "{}: isnan", case.name);
        assert_eq!(
            bools(&isinf(&x).unwrap()),
            case.isinf,
            "{}: isinf",
            case.name
        );
        assert_eq!(
            bools(&isfinite(&x).unwrap()),
            case.isfinite,
            "{}: isfinite",
            case.name
        );
    }
}

#[test]
fn nan_to_num_matches_pytorch_reference_forward_and_backward() {
    let fixture = load_fixture();
    assert!(fixture.torch_version.starts_with("2.14.0"));
    assert!(fixture.nan_to_num_cases.len() >= 30);
    for case in &fixture.nan_to_num_cases {
        let (out, dx) = run_n2n(
            from_bits(&case.x_bits),
            &case.shape,
            from_bits(&case.g_bits),
            args_of(case),
        );
        assert_class_or_bits_eq(
            &out,
            &from_bits(&case.out_bits),
            &format!("{} forward", case.name),
        );
        // 有限の期待値は統一複合判定でも確認する（bit 一致の補強）。
        for (i, (&a, &e)) in out.iter().zip(&from_bits(&case.out_bits)).enumerate() {
            if e.is_finite() {
                assert!(
                    common::req2_close(f64::from(a), f64::from(e)),
                    "{} forward[{i}]",
                    case.name
                );
            }
        }
        assert_close_all(
            &dx,
            &from_bits(&case.grad_bits),
            &format!("{} grad", case.name),
        );
    }
}

/// 上流勾配が非有限のケースの観測記録（PyTorch は `grad * isfinite(x)` のため
/// 入力が非有限かつ上流も非有限の位置で `0 * inf = NaN` になる）。本実装は
/// 要素選択のため非有限入力位置は常に 0、有限入力位置は PyTorch と一致する。
#[test]
fn upstream_nonfinite_observation_differs_only_at_nonfinite_input_positions() {
    let fixture = load_fixture();
    for case in &fixture.upstream_nonfinite_observations {
        let x = from_bits(&case.x_bits);
        let (_, dx) = run_n2n(
            x.clone(),
            &case.shape,
            from_bits(&case.g_bits),
            (None, None, None),
        );
        let torch_grad = from_bits(&case.grad_bits);
        let mut diff_positions = 0;
        for i in 0..x.len() {
            if x[i].is_finite() {
                assert_class_or_bits_eq(&dx[i..=i], &torch_grad[i..=i], &case.name);
            } else {
                assert_eq!(dx[i], 0.0, "{}[{i}]: 非有限入力位置は 0", case.name);
                if torch_grad[i] != 0.0 {
                    diff_positions += 1;
                }
            }
        }
        assert!(
            diff_positions > 0,
            "{}: PyTorch が非有限入力位置で 0 以外（NaN）を返す観測が fixture に残っているはず",
            case.name
        );
    }
}

// --- フォールバックとエラー伝播 ---

/// `scalar_unary` だけを差し替える `BackendOps`。それ以外は naive へ委譲する。
struct ScalarUnaryMock {
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

impl BackendOps for ScalarUnaryMock {
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongShape => Ok(t(vec![0.0; 3], &[3])),
        }
    }
}

fn mock_tape(mode: Mode) -> (Tape, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(ScalarUnaryMock {
        inner: common::naive_ops(),
        calls: Arc::clone(&calls),
        mode,
    }));
    (tape, calls)
}

#[test]
fn unsupported_backend_falls_back_to_host_reference() {
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let x = tape.var(&t(
        vec![f32::NAN, f32::INFINITY, 2.0, f32::NEG_INFINITY],
        &[4],
    ));
    assert_eq!(bools(&isnan(&x).unwrap()), [true, false, false, false]);
    assert_eq!(bools(&isinf(&x).unwrap()), [false, true, false, true]);
    assert_eq!(bools(&isfinite(&x).unwrap()), [false, false, true, false]);
    let y = nan_to_num(&x, None, None, None).unwrap();
    assert_eq!(
        y.to_tensor().host_slice().into_owned(),
        [0.0, f32::MAX, 2.0, f32::MIN]
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        4,
        "バックエンドを先に呼んでからフォールバックしているはず"
    );
}

#[test]
fn non_unsupported_backend_errors_are_propagated() {
    let (tape, _) = mock_tape(Mode::LaunchFailed);
    let x = tape.var(&t(vec![1.0, f32::NAN], &[2]));
    for r in [
        isnan(&x).map(|_| ()),
        isinf(&x).map(|_| ()),
        isfinite(&x).map(|_| ()),
        nan_to_num(&x, None, None, None).map(|_| ()),
    ] {
        assert!(matches!(
            r,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ));
    }
}

#[test]
fn wrong_shape_from_backend_is_a_typed_error() {
    let (tape, _) = mock_tape(Mode::WrongShape);
    let x = tape.var(&t(vec![1.0, f32::NAN], &[2]));
    assert!(matches!(
        isnan(&x),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
    assert!(matches!(
        nan_to_num(&x, None, None, None),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
}

// --- 非微分・VJP ---

#[test]
fn predicates_do_not_record_tape_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![f32::NAN, 1.0], &[2]));
    let before = tape.len();
    isnan(&x).unwrap();
    isinf(&x).unwrap();
    isfinite(&x).unwrap();
    assert_eq!(tape.len(), before);
}

fn n2n_loss_grad(x0: &[f32], shape: &[usize], w: &[f32]) -> Vec<f32> {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let y = nan_to_num(&x, Some(1.0), Some(2.0), Some(-3.0)).unwrap();
    // 非線形な後段（y * y * w）へ流して連鎖を確かめる。
    let wv = tape.var_no_grad(&t(w.to_vec(), shape));
    let loss = y.mul(&y).unwrap().mul(&wv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    grads
        .get(&x)
        .unwrap()
        .expect("勾配が届く")
        .host_slice()
        .into_owned()
}

fn n2n_loss(x0: &[f64], w: &[f32]) -> f64 {
    let op = ScalarUnaryOp::NanToNum {
        nan: 1.0,
        posinf: 2.0,
        neginf: -3.0,
    };
    x0.iter()
        .zip(w)
        .map(|(&x, &w)| {
            let y = f64::from(op.apply(x as f32));
            y * y * f64::from(w)
        })
        .sum()
}

#[test]
fn nan_to_num_vjp_matches_central_difference_at_finite_points() {
    let x0 = [0.7f32, -1.3, 2.2, 0.1];
    let w = [1.0f32, -0.5, 0.25, 2.0];
    let g = n2n_loss_grad(&x0, &[4], &w);
    let h = 1e-3f64;
    for i in 0..4 {
        let mut hi: Vec<f64> = x0.iter().map(|&v| f64::from(v)).collect();
        let mut lo = hi.clone();
        hi[i] += h;
        lo[i] -= h;
        let fd = (n2n_loss(&hi, &w) - n2n_loss(&lo, &w)) / (2.0 * h);
        assert!(
            common::req2_close(f64::from(g[i]), fd),
            "x[{i}]: analytic={} fd={fd}",
            g[i]
        );
    }
}

#[test]
fn nan_to_num_gradient_is_zero_at_nonfinite_inputs_even_with_inf_upstream() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(
        vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.0],
        &[4],
    ));
    let y = nan_to_num(&x, None, None, None).unwrap();
    let inf = tape.var_no_grad(&t(vec![f32::INFINITY; 4], &[4]));
    let loss = y.mul(&inf).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
    assert_eq!(dx[..3], [0.0, 0.0, 0.0], "非有限入力位置は NaN にならず 0");
    assert_eq!(dx[3], f32::INFINITY, "有限入力位置は上流をそのまま通す");
}

// --- 入力形態 ---

#[test]
fn view_inputs_are_handled() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(
        vec![f32::NAN, 1.0, f32::INFINITY, 2.0, f32::NEG_INFINITY, 3.0],
        &[2, 3],
    ));
    let tr = base.transpose(0, 1).unwrap();
    assert_eq!(tr.to_tensor().shape(), &[3, 2]);
    assert_eq!(
        bools(&isnan(&tr).unwrap()),
        [true, false, false, false, false, false]
    );
    assert_eq!(
        bools(&isinf(&tr).unwrap()),
        [false, false, false, true, true, false]
    );
    let nr = base.narrow(1, 1, 2).unwrap();
    assert_eq!(bools(&isfinite(&nr).unwrap()), [true, false, false, true]);
    let one = tape.var(&t(vec![f32::NAN], &[1]));
    let bc = one.broadcast_to(&[2, 3]).unwrap();
    assert_eq!(bools(&isnan(&bc).unwrap()), [true; 6]);
    let y = nan_to_num(&bc, Some(5.0), None, None).unwrap();
    assert_eq!(y.to_tensor().host_slice().into_owned(), [5.0; 6]);
}

#[test]
fn empty_and_scalar_shapes_are_handled() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let empty = tape.var(&t(vec![], &[0]));
    let n = isnan(&empty).unwrap();
    assert_eq!(n.shape(), &[0]);
    assert!(bools(&n).is_empty());
    let scalar = tape.var(&t(vec![f32::INFINITY], &[]));
    assert_eq!(isinf(&scalar).unwrap().shape(), &[] as &[usize]);
    assert_eq!(bools(&isinf(&scalar).unwrap()), [true]);
}

// --- 境界・対象外 ---

#[test]
fn huge_broadcast_view_is_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(vec![1.0], &[1]));
    let huge = base.broadcast_to(&[1usize << 61]).unwrap();
    for r in [
        isnan(&huge).map(|_| ()),
        isinf(&huge).map(|_| ()),
        isfinite(&huge).map(|_| ()),
        nan_to_num(&huge, None, None, None).map(|_| ()),
    ] {
        assert!(matches!(
            r,
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
    }
}

#[test]
fn nan_to_num_create_graph_is_a_typed_error_not_a_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, f32::NAN], &[2]));
    let loss = nan_to_num(&x, None, None, None).unwrap().sum(None).unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
}

#[test]
fn results_are_bit_deterministic_run_to_run() {
    let fixture = load_fixture();
    let case = &fixture.nan_to_num_cases[fixture.nan_to_num_cases.len() - 1];
    let a = run_n2n(
        from_bits(&case.x_bits),
        &case.shape,
        from_bits(&case.g_bits),
        args_of(case),
    );
    let b = run_n2n(
        from_bits(&case.x_bits),
        &case.shape,
        from_bits(&case.g_bits),
        args_of(case),
    );
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.0), bits(&b.0));
    assert_eq!(bits(&a.1), bits(&b.1));
}
