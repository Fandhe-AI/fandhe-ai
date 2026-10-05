//! `binning_ops`（イシュー #2638・`histc`／`bincount`／`searchsorted`／`bucketize`）の
//! `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/binning-pytorch-reference/binning_reference.json`・生成条件は同
//!   ディレクトリの `README.md`）と突合する。f32 は NaN／inf を運べるよう u32 ビット
//!   パターンで保存されている。索引・整数カウント・`histc` のカウントは **完全一致**
//!   （`histc` は添字算術の誤りがビンのずれとして現れるため、REQ-2 判定より厳しい bit
//!   一致を要求し緩めない）、重み付き `bincount` は REQ-2 統一複合判定
//!   （`common::req2_close`。tolerance 定数は新設しない。NaN はクラス一致・`±inf` は厳密）で
//!   比較する。
//! - `common::naive_ops()` は `binning_*` を override しないため、必ず共有ホスト
//!   カーネルへのフォールバック経路を通る。CPU `BackendOps` 実装との一致は
//!   `crates/facade/tests/binning_ops_backend_parity.rs` が担当する。
//! - 4 演算は**非微分**で、tape へノードを積まない（`tape.len()` 不変）ことと、PyTorch 側の
//!   非微分性の実測（fixture の `differentiability`）を固定する。
//! - フォールバックは `Unsupported` のときだけで、それ以外のバックエンドエラーは握りつぶさず
//!   伝播する。確保前サイズ検査・クロステープ検査・run-to-run 決定性も固定する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::binning_ops::{
    bincount, bincount_weighted, bucketize, histc, searchsorted,
};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};
use serde::Deserialize;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    differentiability: Vec<Diff>,
    histc_cases: Vec<HistcCase>,
    histc_errors: Vec<HistcCase>,
    bincount_cases: Vec<BincountCase>,
    bincount_errors: Vec<BincountCase>,
    searchsorted_cases: Vec<SearchCase>,
    searchsorted_errors: Vec<SearchCase>,
    bucketize_cases: Vec<SearchCase>,
    bucketize_errors: Vec<SearchCase>,
}

#[derive(Deserialize)]
struct Diff {
    op: String,
    requires_grad: bool,
    grad_fn: Option<String>,
    backward_raises: bool,
}

#[derive(Deserialize)]
struct HistcCase {
    name: String,
    shape: Vec<usize>,
    x_bits: Vec<u32>,
    bins: usize,
    min_bits: u32,
    max_bits: u32,
    torch_raises: bool,
    out_bits: Option<Vec<u32>>,
}

#[derive(Deserialize)]
struct BincountCase {
    name: String,
    shape: Vec<usize>,
    input: Vec<i32>,
    minlength: usize,
    weights_bits: Option<Vec<u32>>,
    torch_raises: bool,
    out: Option<Vec<i32>>,
    out_bits: Option<Vec<u32>>,
}

#[derive(Deserialize)]
struct SearchCase {
    name: String,
    seq_shape: Vec<usize>,
    seq_bits: Vec<u32>,
    values_shape: Vec<usize>,
    values_bits: Vec<u32>,
    right: bool,
    torch_raises: bool,
    out_shape: Option<Vec<usize>>,
    out: Option<Vec<i64>>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/binning-pytorch-reference/binning_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn naive_tape() -> Tape {
    Tape::new_with_ops(common::naive_ops())
}

/// REQ-2 統一複合判定（NaN はクラス一致・`±inf` は厳密）。
fn class_close(actual: f32, expected: f32) -> bool {
    if expected.is_nan() || actual.is_nan() {
        return expected.is_nan() && actual.is_nan();
    }
    if expected.is_infinite() || actual.is_infinite() {
        return actual == expected;
    }
    common::req2_close(f64::from(actual), f64::from(expected))
}

// --- fixture 突合 ---

#[test]
fn fixture_is_pytorch_2_14_0() {
    assert!(load_fixture().torch_version.starts_with("2.14.0"));
}

#[test]
fn differentiability_matches_the_non_differentiable_contract() {
    let fx = load_fixture();
    for d in &fx.differentiability {
        match d.op.as_str() {
            // PyTorch は重み付き bincount を requires_grad=True・grad_fn=NotImplemented にするが
            // backward は例外（実質非微分）。本実装は detached を返す（決定記録 §5）。
            "bincount_weights" => {
                assert!(d.requires_grad);
                assert_eq!(d.grad_fn.as_deref(), Some("NotImplemented"));
                assert!(d.backward_raises);
            }
            other => {
                assert!(!d.requires_grad, "{other}");
                assert!(d.grad_fn.is_none(), "{other}");
            }
        }
    }
    assert_eq!(fx.differentiability.len(), 5);
}

#[test]
fn histc_matches_pytorch_counts_bit_for_bit() {
    let fx = load_fixture();
    assert!(!fx.histc_cases.is_empty());
    for c in &fx.histc_cases {
        assert!(!c.torch_raises, "{}", c.name);
        let tape = naive_tape();
        let x = tape.var(&t(from_bits(&c.x_bits), &c.shape));
        let got = histc(
            &x,
            c.bins,
            f32::from_bits(c.min_bits),
            f32::from_bits(c.max_bits),
        )
        .unwrap_or_else(|e| panic!("{}: {e:?}", c.name));
        let want = from_bits(c.out_bits.as_ref().unwrap());
        assert_eq!(got.shape(), &[c.bins], "{}", c.name);
        assert_eq!(got.host_slice().into_owned(), want, "{}", c.name);
    }
}

#[test]
fn histc_errors_match_pytorch_exceptions() {
    let fx = load_fixture();
    for c in &fx.histc_errors {
        assert!(c.torch_raises, "{}", c.name);
        let tape = naive_tape();
        let x = tape.var(&t(from_bits(&c.x_bits), &c.shape));
        let r = histc(
            &x,
            c.bins,
            f32::from_bits(c.min_bits),
            f32::from_bits(c.max_bits),
        );
        assert!(
            matches!(r, Err(AutodiffError::InvalidArgument(_))),
            "{}: {r:?}",
            c.name
        );
    }
}

#[test]
fn bincount_matches_pytorch() {
    let fx = load_fixture();
    assert!(!fx.bincount_cases.is_empty());
    for c in &fx.bincount_cases {
        assert!(!c.torch_raises, "{}", c.name);
        let tape = naive_tape();
        let input = ti(c.input.clone(), &c.shape);
        match &c.weights_bits {
            None => {
                let got = bincount(&tape, &input, c.minlength).unwrap();
                assert_eq!(
                    got.host_slice().into_owned(),
                    c.out.clone().unwrap(),
                    "{}",
                    c.name
                );
            }
            Some(wb) => {
                let w = tape.var(&t(from_bits(wb), &[wb.len()]));
                let got = bincount_weighted(&tape, &input, &w, c.minlength).unwrap();
                let want = from_bits(c.out_bits.as_ref().unwrap());
                let got = got.host_slice().into_owned();
                if c.name == "w_cancel" {
                    // 意図した差（決定記録 §5）: PyTorch は f32 逐次加算で `1e8 + 1.0` を丸めて
                    // 0 を返すが、本実装は f64 アキュムレータ契約（coding-rust.md）で真値 1 を返す。
                    // 差が出る入力をここへ固定し、tolerance は触らない。
                    assert_eq!(want, [0.0], "{}", c.name);
                    assert_eq!(got, [1.0], "{}", c.name);
                    continue;
                }
                assert_eq!(got.len(), want.len(), "{}", c.name);
                for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert!(class_close(*a, *b), "{}[{i}]: {a} vs {b}", c.name);
                }
            }
        }
    }
}

#[test]
fn bincount_errors_match_pytorch_exceptions() {
    let fx = load_fixture();
    for c in &fx.bincount_errors {
        assert!(c.torch_raises, "{}", c.name);
        let tape = naive_tape();
        let input = ti(c.input.clone(), &c.shape);
        let r = match &c.weights_bits {
            None => bincount(&tape, &input, c.minlength).map(|_| ()),
            Some(wb) => {
                let w = tape.var(&t(from_bits(wb), &[wb.len()]));
                bincount_weighted(&tape, &input, &w, c.minlength).map(|_| ())
            }
        };
        assert!(
            matches!(
                r,
                Err(AutodiffError::InvalidArgument(_)) | Err(AutodiffError::Shape(_))
            ),
            "{}: {r:?}",
            c.name
        );
    }
}

fn run_search(c: &SearchCase, bucket: bool) -> Result<Tensor<i32>, AutodiffError> {
    let tape = naive_tape();
    let seq = tape.var(&t(from_bits(&c.seq_bits), &c.seq_shape));
    let values = tape.var(&t(from_bits(&c.values_bits), &c.values_shape));
    if bucket {
        bucketize(&values, &seq, c.right)
    } else {
        searchsorted(&seq, &values, c.right)
    }
}

fn check_search_cases(cases: &[SearchCase], bucket: bool) {
    assert!(!cases.is_empty());
    for c in cases {
        assert!(!c.torch_raises, "{}", c.name);
        let got = run_search(c, bucket).unwrap_or_else(|e| panic!("{}: {e:?}", c.name));
        assert_eq!(got.shape(), c.out_shape.as_deref().unwrap(), "{}", c.name);
        let want: Vec<i32> = c
            .out
            .as_ref()
            .unwrap()
            .iter()
            .map(|&v| i32::try_from(v).unwrap())
            .collect();
        assert_eq!(got.host_slice().into_owned(), want, "{}", c.name);
    }
}

fn check_search_errors(cases: &[SearchCase], bucket: bool) {
    assert!(!cases.is_empty());
    for c in cases {
        assert!(c.torch_raises, "{}", c.name);
        let r = run_search(c, bucket);
        assert!(
            matches!(r, Err(AutodiffError::InvalidArgument(_))),
            "{}: {r:?}",
            c.name
        );
    }
}

#[test]
fn searchsorted_matches_pytorch_indices_exactly() {
    check_search_cases(&load_fixture().searchsorted_cases, false);
}

#[test]
fn searchsorted_errors_match_pytorch_exceptions() {
    check_search_errors(&load_fixture().searchsorted_errors, false);
}

#[test]
fn bucketize_matches_pytorch_indices_exactly() {
    check_search_cases(&load_fixture().bucketize_cases, true);
}

#[test]
fn bucketize_errors_match_pytorch_exceptions() {
    check_search_errors(&load_fixture().bucketize_errors, true);
}

// --- 非記録・決定性・境界検査 ---

#[test]
fn ops_do_not_record_tape_nodes() {
    let tape = naive_tape();
    let x = tape.var(&t(vec![0.5, 1.5, 2.5], &[3]));
    let b = tape.var(&t(vec![1.0, 2.0], &[2]));
    let w = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let before = tape.len();
    histc(&x, 3, 0.0, 3.0).unwrap();
    bincount(&tape, &ti(vec![0, 1, 1], &[3]), 0).unwrap();
    bincount_weighted(&tape, &ti(vec![0, 1, 1], &[3]), &w, 0).unwrap();
    searchsorted(&b, &x, false).unwrap();
    bucketize(&x, &b, true).unwrap();
    assert_eq!(tape.len(), before);
}

#[test]
fn results_are_bit_deterministic() {
    let data: Vec<f32> = (0..400)
        .map(|i| ((i * 37 % 29) as f32) * 0.21 - 3.0)
        .collect();
    let run = || {
        let tape = naive_tape();
        let x = tape.var(&t(data.clone(), &[400]));
        let h = histc(&x, 11, -2.0, 2.0).unwrap();
        let idx = ti((0..400).map(|i| i % 9).collect(), &[400]);
        let w = bincount_weighted(&tape, &idx, &x, 0).unwrap();
        let s = searchsorted(&x, &x, true).unwrap();
        (
            h.host_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            w.host_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            s.host_slice().into_owned(),
        )
    };
    assert_eq!(run(), run());
}

#[test]
fn zero_dim_and_empty_rules() {
    let tape = naive_tape();
    let s = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let scalar = tape.var(&t(vec![2.0], &[]));
    let r = searchsorted(&s, &scalar, false).unwrap();
    assert_eq!(r.shape(), &[] as &[usize]);
    assert_eq!(r.host_slice().into_owned(), [1]);
    let empty = tape.var(&t(vec![], &[0]));
    assert_eq!(searchsorted(&s, &empty, true).unwrap().shape(), &[0]);
    assert_eq!(
        histc(&empty, 3, 0.0, 0.0)
            .unwrap()
            .host_slice()
            .into_owned(),
        [0.0; 3]
    );
    assert_eq!(
        bincount(&tape, &ti(vec![], &[0]), 2)
            .unwrap()
            .host_slice()
            .into_owned(),
        [0, 0]
    );
}

#[test]
fn huge_broadcast_view_is_rejected_before_allocation() {
    let tape = naive_tape();
    let base = tape.var(&t(vec![1.0], &[1]));
    let huge = base.broadcast_to(&[1usize << 61]).unwrap();
    let small = tape.var(&t(vec![1.0, 2.0], &[2]));
    let rs = [
        histc(&huge, 4, 0.0, 1.0).map(|_| ()),
        searchsorted(&small, &huge, false).map(|_| ()),
        searchsorted(&huge, &small, false).map(|_| ()),
        bucketize(&huge, &small, false).map(|_| ()),
        bucketize(&small, &huge, false).map(|_| ()),
        bincount_weighted(&tape, &ti(vec![0], &[1]), &huge, 0).map(|_| ()),
    ];
    for r in rs {
        assert!(
            matches!(
                r,
                Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
            ),
            "{r:?}"
        );
    }
}

#[test]
fn huge_bins_and_minlength_are_typed_errors_not_aborts() {
    let tape = naive_tape();
    let x = tape.var(&t(vec![0.5], &[1]));
    assert!(matches!(
        histc(&x, usize::MAX, 0.0, 1.0),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
    assert!(matches!(
        histc(&x, usize::MAX / 8, 0.0, 1.0),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
    assert!(matches!(
        bincount(&tape, &ti(vec![0], &[1]), usize::MAX),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
}

#[test]
fn cross_tape_operands_are_rejected_before_shape_checks() {
    let a = naive_tape();
    let b = naive_tape();
    let x = a.var(&t(vec![1.0, 2.0], &[2]));
    // shape も不正（rank 0 の sorted_sequence）にして、tape 検査が先であることを確認する。
    let y = b.var(&t(vec![1.0], &[]));
    assert!(matches!(
        searchsorted(&y, &x, false),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        bucketize(&x, &y, false),
        Err(AutodiffError::TapeMismatch)
    ));
    let w = b.var(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        bincount_weighted(&a, &ti(vec![0, 1], &[2]), &w, 0),
        Err(AutodiffError::TapeMismatch)
    ));
}

// --- フォールバックとエラー伝播 ---

/// `binning_*` だけを差し替える `BackendOps`。それ以外は naive へ委譲する。
struct BinMock {
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

impl BinMock {
    fn fail<T: fandhe_ai_tensor_core::Element>(
        &self,
        wrong: Tensor<T>,
    ) -> Result<Tensor<T>, BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongShape => Ok(wrong),
        }
    }
}

impl BackendOps for BinMock {
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
    fn binning_histc(
        &self,
        _x: &Tensor<f32>,
        _bins: usize,
        _min: f32,
        _max: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        // 期待 shape は [3]。わざと [2, 2] を返す。
        self.fail(t(vec![0.0; 4], &[2, 2]))
    }
    fn binning_bincount(
        &self,
        _input: &Tensor<i32>,
        _minlength: usize,
    ) -> Result<Tensor<i32>, BackendError> {
        self.fail(ti(vec![0; 4], &[2, 2]))
    }
    fn binning_bincount_weighted(
        &self,
        _input: &Tensor<i32>,
        _weights: &Tensor<f32>,
        _minlength: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        self.fail(t(vec![0.0; 4], &[2, 2]))
    }
    fn binning_searchsorted(
        &self,
        _sorted_sequence: &Tensor<f32>,
        _values: &Tensor<f32>,
        _right: bool,
    ) -> Result<Tensor<i32>, BackendError> {
        self.fail(ti(vec![0; 5], &[5]))
    }
}

fn mock_tape(mode: Mode) -> (Tape, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(BinMock {
        inner: common::naive_ops(),
        calls: Arc::clone(&calls),
        mode,
    }));
    (tape, calls)
}

/// 全 5 関数を小さな入力へ適用する（出力の shape は `[3]`／`[3]`／`[3]`／`[3]`／`[3]`）。
fn call_all(tape: &Tape) -> Vec<Result<(), AutodiffError>> {
    let x = tape.var(&t(vec![0.5, 1.5, 2.5], &[3]));
    let b = tape.var(&t(vec![1.0, 2.0], &[2]));
    let idx = ti(vec![0, 1, 2], &[3]);
    vec![
        histc(&x, 3, 0.0, 3.0).map(|_| ()),
        bincount(tape, &idx, 0).map(|_| ()),
        bincount_weighted(tape, &idx, &x, 0).map(|_| ()),
        searchsorted(&b, &x, false).map(|_| ()),
        bucketize(&x, &b, true).map(|_| ()),
    ]
}

#[test]
fn unsupported_backend_falls_back_to_host_reference() {
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let x = tape.var(&t(vec![0.5, 1.5, 2.5], &[3]));
    let b = tape.var(&t(vec![1.0, 2.0], &[2]));
    let idx = ti(vec![0, 1, 1], &[3]);
    assert_eq!(
        histc(&x, 3, 0.0, 3.0).unwrap().host_slice().into_owned(),
        [1.0, 1.0, 1.0]
    );
    assert_eq!(
        bincount(&tape, &idx, 0).unwrap().host_slice().into_owned(),
        [1, 2]
    );
    assert_eq!(
        bincount_weighted(&tape, &idx, &x, 0)
            .unwrap()
            .host_slice()
            .into_owned(),
        [0.5, 4.0]
    );
    assert_eq!(
        searchsorted(&b, &x, false)
            .unwrap()
            .host_slice()
            .into_owned(),
        [0, 1, 2]
    );
    assert_eq!(
        bucketize(&x, &b, true).unwrap().host_slice().into_owned(),
        [0, 1, 2]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 5);
}

#[test]
fn other_backend_errors_propagate_without_fallback() {
    let (tape, calls) = mock_tape(Mode::LaunchFailed);
    for r in call_all(&tape) {
        assert!(
            matches!(
                r,
                Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
            ),
            "{r:?}"
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 5);
}

#[test]
fn wrong_backend_result_shape_is_a_typed_error() {
    let (tape, _) = mock_tape(Mode::WrongShape);
    for r in call_all(&tape) {
        assert!(
            matches!(
                r,
                Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
            ),
            "{r:?}"
        );
    }
}

/// バックエンドが `minlength` 以上でも契約長（`max(max+1, minlength)`）と異なる長さを返したら
/// 型付きエラー（完全一致検証。bincount・bincount_weighted 両経路）。
#[test]
fn bincount_backend_length_must_match_contract_exactly() {
    // WrongShape モックは常に [2, 2] を返すため別の長さのモックで検証する。
    let (tape, _) = mock_tape(Mode::WrongShape);
    let idx = ti(vec![0, 1], &[2]);
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    assert!(bincount(&tape, &idx, 0).is_err());
    assert!(bincount_weighted(&tape, &idx, &x, 0).is_err());
}

/// 負の入力値・重み長不一致はバックエンド呼び出し前に拒否する（呼び出し回数 0）。
#[test]
fn bincount_invalid_arguments_rejected_before_backend_call() {
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let neg = ti(vec![0, -1], &[2]);
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let short = tape.var(&t(vec![1.0], &[1]));
    let e1 = bincount(&tape, &neg, 0);
    let e2 = bincount_weighted(&tape, &neg, &x, 0);
    assert!(matches!(
        (&e1, &e2),
        (
            Err(AutodiffError::InvalidArgument(_)),
            Err(AutodiffError::InvalidArgument(_))
        )
    ));
    let ok = ti(vec![0, 1], &[2]);
    assert!(bincount_weighted(&tape, &ok, &short, 0).is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// 空入力の重み付き版は重みの rank を見ずに `minlength` 個の零を返す。
#[test]
fn bincount_weighted_empty_input_ignores_weights() {
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let w = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let out = bincount_weighted(&tape, &ti(vec![], &[0]), &w, 3).unwrap();
    assert_eq!(out.shape(), [3]);
    assert_eq!(out.host_slice().into_owned(), [0.0, 0.0, 0.0]);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
