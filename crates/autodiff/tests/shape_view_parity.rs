//! `shape_view_ops`（イシュー #2639・`unbind`／`movedim`／`swapaxes`／
//! `tensor_split`／`meshgrid`／`rot90`）の `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/shape-view-pytorch-reference/shape_view_reference.json`・生成条件は
//!   同ディレクトリの `README.md`）と突合する。f32 は NaN／inf を運べるよう u32
//!   ビットパターンで保存されている。forward はコピーのみのため NaN の payload まで
//!   bit 一致を要求し、勾配は REQ-2 統一複合判定（`common::req2_close`。tolerance 定数は
//!   新設しない）で比較する。
//! - `error_cases` は torch が例外を出すケースが型付きエラーになることを確認する。
//!   意図的な差分（`INTENDED_DIFFS`）は決定記録 §5 と一対一に対応させる。
//! - 独立オラクルとして、損失が入力の線形関数になる性質を使った中心差分を併用する。
//! - バックエンド到達性: `gather`／`scatter`／`concat` を差し替えるモックで、
//!   `Unsupported` のときだけホスト参照実装へフォールバックすること（呼び出し回数が
//!   1 以上）、他のエラーは握りつぶさず伝播すること、view だけの演算の forward では
//!   これらが 1 回も呼ばれないことを固定する。CPU `BackendOps` 実装との一致は
//!   `crates/facade/tests/shape_view_ops_backend_parity.rs` が担当する。
//! - テープ記録数・run-to-run の bit 決定性・クロステープ拒否も固定する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::shape_view_ops::{
    MeshgridIndexing, meshgrid, movedim, rot90, swapaxes, tensor_split, tensor_split_indices,
    unbind,
};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, ScatterReduce, ShapeError, Tensor};
use serde::Deserialize;
use serde_json::Value;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    cases: Vec<Case>,
    error_cases: Vec<ErrCase>,
}

#[derive(Deserialize)]
struct Input {
    shape: Vec<usize>,
    x_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Out {
    shape: Vec<usize>,
    bits: Vec<u32>,
    g_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    op: String,
    params: Value,
    pre_transpose: Option<Vec<usize>>,
    inputs: Vec<Input>,
    outs: Vec<Out>,
    grads: Vec<Vec<u32>>,
}

#[derive(Deserialize)]
struct ErrCase {
    name: String,
    op: String,
    params: Value,
    input_shapes: Vec<Vec<usize>>,
    torch_raises: bool,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/shape-view-pytorch-reference/shape_view_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn usize_of(p: &Value, key: &str) -> usize {
    p[key]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or_else(|| panic!("params.{key} が usize でない"))
}

fn usize_list(p: &Value, key: &str) -> Vec<usize> {
    p[key]
        .as_array()
        .unwrap_or_else(|| panic!("params.{key} が配列でない"))
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .expect("usize")
        })
        .collect()
}

/// fixture の演算指定を Rust 実装へ適用する（出力は常に `Vec`）。
fn apply_op<'t>(op: &str, p: &Value, xs: &[Var<'t>]) -> Result<Vec<Var<'t>>, AutodiffError> {
    match op {
        "unbind" => unbind(&xs[0], usize_of(p, "dim")),
        "movedim" => Ok(vec![movedim(
            &xs[0],
            &usize_list(p, "source"),
            &usize_list(p, "destination"),
        )?]),
        "swapaxes" => Ok(vec![swapaxes(&xs[0], usize_of(p, "a"), usize_of(p, "b"))?]),
        "tensor_split" => tensor_split(&xs[0], usize_of(p, "sections"), usize_of(p, "dim")),
        "tensor_split_indices" => {
            tensor_split_indices(&xs[0], &usize_list(p, "indices"), usize_of(p, "dim"))
        }
        "meshgrid" => {
            let indexing = match p["indexing"].as_str() {
                Some("ij") => MeshgridIndexing::Ij,
                Some("xy") => MeshgridIndexing::Xy,
                other => panic!("未知の indexing: {other:?}"),
            };
            meshgrid(xs, indexing)
        }
        "rot90" => {
            let k = isize::try_from(p["k"].as_i64().expect("k")).expect("k は isize");
            let dims = usize_list(p, "dims");
            Ok(vec![rot90(&xs[0], k, [dims[0], dims[1]])?])
        }
        other => panic!("未知の op: {other}"),
    }
}

/// NaN の payload まで含めて bit 完全一致。
fn assert_bits_eq(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{context}[{i}]: actual={a} expected={e}"
        );
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

struct Run {
    out_shapes: Vec<Vec<usize>>,
    out_values: Vec<Vec<f32>>,
    grads: Vec<Vec<f32>>,
}

/// `Σ_k (y_k * g_k).sum()` を損失とした forward 値・入力勾配を返す。
fn run_case(ops: Box<dyn BackendOps + Send>, c: &Case) -> Run {
    let tape = Tape::new_with_ops(ops);
    let leaves: Vec<Var<'_>> = c
        .inputs
        .iter()
        .map(|i| tape.var(&t(from_bits(&i.x_bits), &i.shape)))
        .collect();
    let mut views = leaves.clone();
    if let Some(pt) = &c.pre_transpose {
        views[0] = leaves[0].transpose(pt[0], pt[1]).expect("pre_transpose");
    }
    let outs = apply_op(&c.op, &c.params, &views).expect("forward");
    let out_shapes: Vec<Vec<usize>> = outs
        .iter()
        .map(|o| o.to_tensor().shape().to_vec())
        .collect();
    let out_values: Vec<Vec<f32>> = outs
        .iter()
        .map(|o| o.to_tensor().host_slice().into_owned())
        .collect();
    let grads = if outs.is_empty() {
        leaves
            .iter()
            .map(|l| vec![0.0; l.to_tensor().numel()])
            .collect()
    } else {
        let mut loss: Option<Var<'_>> = None;
        for (y, o) in outs.iter().zip(&c.outs) {
            let g = tape.var_no_grad(&t(from_bits(&o.g_bits), &o.shape));
            let term = y.mul(&g).expect("mul").sum(None).expect("sum");
            loss = Some(match loss {
                None => term,
                Some(acc) => acc.add(&term).expect("add"),
            });
        }
        let loss = loss.expect("非空");
        let gs = tape.backward(&loss).expect("backward");
        leaves
            .iter()
            .map(|l| match gs.get(l).expect("get") {
                Some(g) => g.host_slice().into_owned(),
                None => vec![0.0; l.to_tensor().numel()],
            })
            .collect()
    };
    Run {
        out_shapes,
        out_values,
        grads,
    }
}

fn check_case(c: &Case, r: &Run) {
    assert_eq!(r.out_shapes.len(), c.outs.len(), "{}: 出力本数", c.name);
    for (k, o) in c.outs.iter().enumerate() {
        assert_eq!(r.out_shapes[k], o.shape, "{}: 出力 {k} の shape", c.name);
        assert_bits_eq(
            &r.out_values[k],
            &from_bits(&o.bits),
            &format!("{} forward[{k}]", c.name),
        );
    }
    for (i, g) in c.grads.iter().enumerate() {
        assert_close_all(&r.grads[i], &from_bits(g), &format!("{} grad[{i}]", c.name));
    }
}

#[test]
fn fixture_matches_pytorch_2_14_0() {
    let fx = load_fixture();
    assert!(
        fx.torch_version.starts_with("2.14.0"),
        "fixture の torch バージョン: {}",
        fx.torch_version
    );
    assert!(
        fx.cases.len() >= 150,
        "fixture ケース数: {}",
        fx.cases.len()
    );
    for op in [
        "unbind",
        "movedim",
        "swapaxes",
        "tensor_split",
        "tensor_split_indices",
        "meshgrid",
        "rot90",
    ] {
        assert!(
            fx.cases.iter().any(|c| c.op == op),
            "fixture に {op} のケースがない"
        );
    }
    for c in &fx.cases {
        let r = run_case(common::naive_ops(), c);
        check_case(c, &r);
    }
}

#[test]
fn fixture_includes_nonfinite_inputs_with_payloads() {
    let fx = load_fixture();
    let nonfinite: Vec<&Case> = fx
        .cases
        .iter()
        .filter(|c| c.name.contains("nonfinite"))
        .collect();
    assert!(
        nonfinite.len() >= 12,
        "nonfinite ケース数: {}",
        nonfinite.len()
    );
    let mut payload_nan = false;
    let mut neg_zero = false;
    let mut inf = false;
    for c in &nonfinite {
        for i in &c.inputs {
            for &b in &i.x_bits {
                payload_nan |= b == 0x7FC0_0001;
                neg_zero |= b == 0x8000_0000;
                inf |= b == 0x7F80_0000 || b == 0xFF80_0000;
            }
        }
    }
    assert!(payload_nan && neg_zero && inf);
}

// --- error_cases ---

/// torch が例外を出さないのに本実装が型付きエラーにするケース（決定記録 §5）。
const INTENDED_DIFFS: [&str; 1] = ["swapaxes_rank0"];

#[test]
fn error_cases_are_typed_errors_and_diffs_are_intended() {
    let fx = load_fixture();
    let mut diffs: Vec<String> = Vec::new();
    for e in &fx.error_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xs: Vec<Var<'_>> = e
            .input_shapes
            .iter()
            .map(|s| tape.var(&t(vec![0.0; s.iter().product()], s)))
            .collect();
        let before = tape.len();
        let r = apply_op(&e.op, &e.params, &xs);
        assert_eq!(tape.len(), before, "{}: エラー時にノードを残さない", e.name);
        match (&r, e.torch_raises) {
            (Err(_), true) => {}
            (Err(_), false) => diffs.push(e.name.clone()),
            (Ok(_), true) => panic!("{}: torch は例外だが本実装は成功した", e.name),
            (Ok(_), false) => {}
        }
        if let Err(err) = &r {
            let kind_ok = match e.name.as_str() {
                "movedim_len_mismatch"
                | "tensor_split_sections0"
                | "meshgrid_empty"
                | "meshgrid_rank2_input" => matches!(err, AutodiffError::InvalidArgument(_)),
                "movedim_dup_source" | "movedim_dup_dest" | "rot90_dims_same" => {
                    matches!(err, AutodiffError::Shape(ShapeError::DuplicateAxis { .. }))
                }
                _ => matches!(err, AutodiffError::Shape(ShapeError::AxisOutOfRange { .. })),
            };
            assert!(kind_ok, "{}: 想定外のエラー種別 {err:?}", e.name);
        }
    }
    let mut expected: Vec<String> = INTENDED_DIFFS.iter().map(|s| s.to_string()).collect();
    diffs.sort();
    expected.sort();
    assert_eq!(diffs, expected, "意図的な差分の集合が決定記録 §5 と不一致");
}

// --- 独立オラクル（中心差分） ---

/// 損失 `Σ (y_k * g_k)` を f64 でホスト計算する。
fn host_loss(outs: &[Var<'_>], gs: &[Vec<f32>]) -> f64 {
    let mut s = 0.0f64;
    for (y, g) in outs.iter().zip(gs) {
        let v = y.to_tensor().host_slice().into_owned();
        assert_eq!(v.len(), g.len());
        for (a, b) in v.iter().zip(g) {
            s += f64::from(*a) * f64::from(*b);
        }
    }
    s
}

fn make_vars<'t>(tape: &'t Tape, shapes: &[Vec<usize>], xs: &[Vec<f32>]) -> Vec<Var<'t>> {
    shapes
        .iter()
        .zip(xs)
        .map(|(s, x)| tape.var(&t(x.clone(), s)))
        .collect()
}

/// 全演算は入力の線形関数なので、刻み h の中心差分は丸め誤差を除いて厳密。
fn central_diff_check(op: &str, params: &Value, shapes: &[Vec<usize>]) {
    let h = 0.5f32;
    let base: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(k, s)| {
            let n: usize = s.iter().product();
            (0..n)
                .map(|i| ((i * 7 + k * 3 + 1) % 11) as f32 * 0.25 - 1.0)
                .collect()
        })
        .collect();
    // 出力形状と上流勾配を決める。
    let tape0 = Tape::new_with_ops(common::naive_ops());
    let xs0 = make_vars(&tape0, shapes, &base);
    let outs0 = apply_op(op, params, &xs0).expect("forward");
    let gs: Vec<Vec<f32>> = outs0
        .iter()
        .enumerate()
        .map(|(k, y)| {
            let n = y.to_tensor().numel();
            (0..n)
                .map(|i| ((i * 5 + k * 2 + 3) % 13) as f32 * 0.125 - 0.75)
                .collect()
        })
        .collect();
    // 解析勾配。
    let tape1 = Tape::new_with_ops(common::naive_ops());
    let xs1 = make_vars(&tape1, shapes, &base);
    let outs1 = apply_op(op, params, &xs1).expect("forward");
    let mut loss: Option<Var<'_>> = None;
    for (y, g) in outs1.iter().zip(&gs) {
        let gv = tape1.var_no_grad(&t(g.clone(), y.to_tensor().shape()));
        let term = y.mul(&gv).unwrap().sum(None).unwrap();
        loss = Some(match loss {
            None => term,
            Some(a) => a.add(&term).unwrap(),
        });
    }
    let grads = tape1.backward(&loss.expect("非空")).unwrap();
    for (k, x1) in xs1.iter().enumerate() {
        let analytic = grads
            .get(x1)
            .unwrap()
            .expect("勾配")
            .host_slice()
            .into_owned();
        for i in 0..base[k].len() {
            let eval = |delta: f32| -> f64 {
                let mut pert = base.clone();
                pert[k][i] += delta;
                let tape = Tape::new_with_ops(common::naive_ops());
                let xs = make_vars(&tape, shapes, &pert);
                let outs = apply_op(op, params, &xs).expect("forward");
                host_loss(&outs, &gs)
            };
            let numeric = (eval(h) - eval(-h)) / (2.0 * f64::from(h));
            assert!(
                common::req2_close(f64::from(analytic[i]), numeric),
                "{op} 入力 {k}[{i}]: analytic={} numeric={numeric}",
                analytic[i]
            );
        }
    }
}

#[test]
fn central_difference_agrees_for_all_seven_functions() {
    let j = |s: &str| -> Value { serde_json::from_str(s).expect("json") };
    central_diff_check("unbind", &j(r#"{"dim":1}"#), &[vec![2, 3, 2]]);
    central_diff_check("unbind", &j(r#"{"dim":0}"#), &[vec![3, 2]]);
    central_diff_check(
        "movedim",
        &j(r#"{"source":[0,1],"destination":[2,0]}"#),
        &[vec![2, 3, 2]],
    );
    central_diff_check("swapaxes", &j(r#"{"a":0,"b":2}"#), &[vec![2, 3, 2]]);
    central_diff_check(
        "tensor_split",
        &j(r#"{"sections":3,"dim":1}"#),
        &[vec![2, 7]],
    );
    central_diff_check(
        "tensor_split_indices",
        &j(r#"{"indices":[5,2,9],"dim":1}"#),
        &[vec![2, 7]],
    );
    central_diff_check(
        "meshgrid",
        &j(r#"{"indexing":"xy"}"#),
        &[vec![3], vec![2], vec![2]],
    );
    central_diff_check("meshgrid", &j(r#"{"indexing":"ij"}"#), &[vec![3], vec![2]]);
    central_diff_check("rot90", &j(r#"{"k":1,"dims":[0,1]}"#), &[vec![2, 3]]);
    central_diff_check("rot90", &j(r#"{"k":3,"dims":[0,2]}"#), &[vec![2, 3, 2]]);
}

// --- バックエンド到達性（モック） ---

/// `gather`／`scatter`／`concat` だけを差し替える `BackendOps`。他は naive へ委譲する。
struct ReachMock {
    inner: Box<dyn BackendOps + Send>,
    gather_calls: Arc<AtomicUsize>,
    scatter_calls: Arc<AtomicUsize>,
    concat_calls: Arc<AtomicUsize>,
    launch_failed: bool,
}

impl ReachMock {
    fn answer(&self) -> BackendError {
        if self.launch_failed {
            BackendError::KernelLaunchFailed("simulated".into())
        } else {
            BackendError::Unsupported("mock".into())
        }
    }
}

impl BackendOps for ReachMock {
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
    fn gather(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _index: &Tensor<i32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.gather_calls.fetch_add(1, Ordering::SeqCst);
        Err(self.answer())
    }
    fn scatter(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _index: &Tensor<i32>,
        _src: &Tensor<f32>,
        _reduce: ScatterReduce,
    ) -> Result<Tensor<f32>, BackendError> {
        self.scatter_calls.fetch_add(1, Ordering::SeqCst);
        Err(self.answer())
    }
    fn concat(&self, _inputs: &[&Tensor<f32>], _dim: usize) -> Result<Tensor<f32>, BackendError> {
        self.concat_calls.fetch_add(1, Ordering::SeqCst);
        Err(self.answer())
    }
}

struct Calls {
    gather: Arc<AtomicUsize>,
    scatter: Arc<AtomicUsize>,
    concat: Arc<AtomicUsize>,
}

impl Calls {
    fn get(&self) -> (usize, usize, usize) {
        (
            self.gather.load(Ordering::SeqCst),
            self.scatter.load(Ordering::SeqCst),
            self.concat.load(Ordering::SeqCst),
        )
    }
}

fn mock_tape(launch_failed: bool) -> (Tape, Calls) {
    let calls = Calls {
        gather: Arc::new(AtomicUsize::new(0)),
        scatter: Arc::new(AtomicUsize::new(0)),
        concat: Arc::new(AtomicUsize::new(0)),
    };
    let tape = Tape::new_with_ops(Box::new(ReachMock {
        inner: common::naive_ops(),
        gather_calls: Arc::clone(&calls.gather),
        scatter_calls: Arc::clone(&calls.scatter),
        concat_calls: Arc::clone(&calls.concat),
        launch_failed,
    }));
    (tape, calls)
}

fn seq(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    t((0..n).map(|i| i as f32 + 1.0).collect(), shape)
}

/// `Σ (y * y)` を損失にして逆伝播する（出力全体へ非自明な上流勾配を流す）。
fn backward_of<'t>(tape: &'t Tape, outs: &[Var<'t>]) -> Result<(), AutodiffError> {
    let mut loss: Option<Var<'t>> = None;
    for y in outs {
        let term = y.mul(y)?.sum(None)?;
        loss = Some(match loss {
            None => term,
            Some(a) => a.add(&term)?,
        });
    }
    tape.backward(&loss.expect("非空")).map(|_| ())
}

#[test]
fn unsupported_backend_falls_back_to_host_reference() {
    // rot90: forward は gather、backward は scatter を先に呼んでからフォールバックする。
    let (tape, calls) = mock_tape(false);
    let x = tape.var(&seq(&[2, 3]));
    let y = rot90(&x, 1, [0, 1]).unwrap();
    assert_eq!(
        y.to_tensor().host_slice().into_owned(),
        vec![3.0, 6.0, 2.0, 5.0, 1.0, 4.0]
    );
    let (g, s, _) = calls.get();
    assert!(g >= 1, "gather を先に呼んでいるはず");
    assert_eq!(s, 0);
    backward_of(&tape, &[y]).unwrap();
    assert!(calls.get().1 >= 1, "scatter を先に呼んでいるはず");

    // tensor_split／unbind: backward は Narrow の VJP が concat を呼ぶ。
    let (tape, calls) = mock_tape(false);
    let x = tape.var(&seq(&[2, 6]));
    let parts = tensor_split(&x, 3, 1).unwrap();
    assert_eq!(calls.get(), (0, 0, 0), "view の forward は呼ばない");
    backward_of(&tape, &parts).unwrap();
    assert!(calls.get().2 >= 1, "concat を先に呼んでいるはず");

    let (tape, calls) = mock_tape(false);
    let x = tape.var(&seq(&[3, 2]));
    let parts = unbind(&x, 0).unwrap();
    assert_eq!(calls.get(), (0, 0, 0));
    backward_of(&tape, &parts).unwrap();
    assert!(calls.get().2 >= 1);
}

#[test]
fn view_only_functions_never_call_gather_scatter_or_concat() {
    let (tape, calls) = mock_tape(false);
    let x = tape.var(&seq(&[2, 3, 4]));
    let a = swapaxes(&x, 0, 2).unwrap();
    let b = movedim(&x, &[0, 1], &[2, 0]).unwrap();
    let m1 = tape.var(&seq(&[3]));
    let m2 = tape.var(&seq(&[2]));
    let mg = meshgrid(&[m1, m2], MeshgridIndexing::Xy).unwrap();
    assert_eq!(
        calls.get(),
        (0, 0, 0),
        "view の forward はバックエンドを呼ばない"
    );
    let mut outs = vec![a, b];
    outs.extend(mg);
    backward_of(&tape, &outs).unwrap();
    assert_eq!(
        calls.get(),
        (0, 0, 0),
        "swapaxes／movedim／meshgrid の backward も gather／scatter／concat を呼ばない"
    );
}

#[test]
fn non_unsupported_backend_errors_are_propagated() {
    let (tape, _) = mock_tape(true);
    let x = tape.var(&seq(&[2, 3]));
    assert!(matches!(
        rot90(&x, 1, [0, 1]),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    let parts = tensor_split(&x, 3, 1).unwrap();
    assert!(matches!(
        backward_of(&tape, &parts),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    let (tape, _) = mock_tape(true);
    let x = tape.var(&seq(&[3, 2]));
    let parts = unbind(&x, 0).unwrap();
    assert!(matches!(
        backward_of(&tape, &parts),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
}

// --- テープ記録数・決定性・クロステープ ---

#[test]
fn tape_node_counts_are_fixed() {
    let count = |f: &dyn Fn(&Tape, Var<'_>) -> usize, shape: &[usize]| -> usize {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&seq(shape));
        let before = tape.len();
        let produced = f(&tape, x);
        assert!(produced > 0);
        tape.len() - before
    };
    assert_eq!(
        count(&|_, x| swapaxes(&x, 0, 1).unwrap().shape_len(), &[2, 3]),
        1
    );
    assert_eq!(
        count(
            &|_, x| movedim(&x, &[0], &[1]).unwrap().shape_len(),
            &[2, 3]
        ),
        1
    );
    assert_eq!(
        count(&|_, x| tensor_split(&x, 3, 0).unwrap().len(), &[7]),
        3
    );
    assert_eq!(
        count(
            &|_, x| tensor_split_indices(&x, &[2, 5], 0).unwrap().len(),
            &[7]
        ),
        3
    );
    // 先頭軸の unbind は narrow + reshape の 2 ノード／本、先頭軸以外は contiguous が加わる。
    assert_eq!(count(&|_, x| unbind(&x, 0).unwrap().len(), &[3, 4]), 6);
    assert_eq!(count(&|_, x| unbind(&x, 1).unwrap().len(), &[3, 4]), 12);
}

trait ShapeLen {
    fn shape_len(&self) -> usize;
}

impl ShapeLen for Var<'_> {
    fn shape_len(&self) -> usize {
        self.to_tensor().shape().len()
    }
}

#[test]
fn results_are_bit_deterministic_across_runs() {
    let fx = load_fixture();
    for c in fx.cases.iter().filter(|c| c.name.contains("nonfinite")) {
        let a = run_case(common::naive_ops(), c);
        let b = run_case(common::naive_ops(), c);
        for (va, vb) in a.out_values.iter().zip(&b.out_values) {
            assert_bits_eq(va, vb, &format!("{} forward", c.name));
        }
        for (ga, gb) in a.grads.iter().zip(&b.grads) {
            assert_bits_eq(ga, gb, &format!("{} grad", c.name));
        }
    }
}

#[test]
fn meshgrid_across_tapes_is_rejected() {
    let a_tape = Tape::new_with_ops(common::naive_ops());
    let b_tape = Tape::new_with_ops(common::naive_ops());
    let a = a_tape.var(&seq(&[2]));
    let b = b_tape.var(&seq(&[3]));
    assert!(matches!(
        meshgrid(&[a, b], MeshgridIndexing::Ij),
        Err(AutodiffError::TapeMismatch)
    ));
}
