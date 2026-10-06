//! `tensor_product_ops`（イシュー #2640・`kron`／`tensordot`／`cdist`／`cross`）の
//! `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/tensor-product-pytorch-reference/tensor_product_reference.json`・
//!   生成条件は同ディレクトリの `README.md`）と突合する。forward・勾配とも REQ-2 統一
//!   複合判定（`common::req2_close`。tolerance 定数は新設しない）。非有限値入力の
//!   ケース（名前に `nonfinite`）は forward の値クラス（NaN 同士・同符号の inf・有限値は
//!   REQ-2）だけを突合し、勾配の一致は受入条件外とする。
//! - `error_cases` は torch が例外を出すケースが型付きエラーになることを確認する。
//!   本実装だけがエラーにする／しないケース（`INTENDED_DIFFS`）は決定記録 §5 と一対一。
//! - 独立オラクル: テスト内の `f64` 総当たり実装と forward を突合し、その `f64` 実装の
//!   中心差分と backward を突合する。
//! - バックエンド到達性: `scalar_binary`／`vector_norm`／`vector_norm_p`／`gather`／
//!   `scatter` を差し替えるモックで、`Unsupported` のときだけホスト参照実装へ落ちること、
//!   他のエラーは握りつぶさず伝播すること、`kron`／`tensordot` がそれらを呼ばないことを
//!   固定する。CPU `BackendOps` との一致は
//!   `crates/facade/tests/tensor_product_ops_backend_parity.rs` が担当する。
//! - テープ記録数・run-to-run の bit 決定性・クロステープ拒否も固定する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::tensor_product_ops::{cdist, cross, kron, tensordot, tensordot_axes};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{
    BackendOps, ScalarBinaryOp, ScatterReduce, ShapeError, Tensor, VectorNormOrd,
};
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
    out: Out,
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
        .join("tests/fixtures/tensor-product-pytorch-reference/tensor_product_reference.json");
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

/// fixture の `p`。`inf` は JSON の数値で運べないため文字列 `"inf"` で保存される。
fn p_of(p: &Value) -> f32 {
    match &p["p"] {
        Value::Number(n) => n.as_f64().expect("p") as f32,
        Value::String(s) if s == "inf" => f32::INFINITY,
        other => panic!("未知の p: {other:?}"),
    }
}

fn apply_op<'t>(op: &str, p: &Value, xs: &[Var<'t>]) -> Result<Var<'t>, AutodiffError> {
    match op {
        "kron" => kron(&xs[0], &xs[1]),
        "tensordot" => tensordot(&xs[0], &xs[1], usize_of(p, "n")),
        "tensordot_axes" => tensordot_axes(
            &xs[0],
            &xs[1],
            &usize_list(p, "dims_a"),
            &usize_list(p, "dims_b"),
        ),
        "cdist" => cdist(&xs[0], &xs[1], p_of(p)),
        "cross" => cross(&xs[0], &xs[1], usize_of(p, "dim")),
        other => panic!("未知の op: {other}"),
    }
}

/// NaN 同士・同符号の inf・有限値は REQ-2 で突合する（値クラス一致）。
fn assert_close_class(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() || a.is_nan() {
            assert!(
                a.is_nan() && e.is_nan(),
                "{context}[{i}]: actual={a} expected={e}"
            );
        } else if e.is_infinite() || a.is_infinite() {
            assert_eq!(a, e, "{context}[{i}]");
        } else {
            assert!(
                common::req2_close(f64::from(a), f64::from(e)),
                "{context}[{i}]: actual={a} expected={e}"
            );
        }
    }
}

struct Run {
    out_shape: Vec<usize>,
    out_values: Vec<f32>,
    grads: Vec<Vec<f32>>,
}

/// `(y * g).sum()` を損失とした forward 値・入力勾配を返す。
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
    let y = apply_op(&c.op, &c.params, &views).expect("forward");
    let out_shape = y.to_tensor().shape().to_vec();
    let out_values = y.to_tensor().host_slice().into_owned();
    let g = tape.var_no_grad(&t(from_bits(&c.out.g_bits), &c.out.shape));
    let loss = y.mul(&g).expect("mul").sum(None).expect("sum");
    let gs = tape.backward(&loss).expect("backward");
    let grads = leaves
        .iter()
        .map(|l| match gs.get(l).expect("get") {
            Some(g) => g.host_slice().into_owned(),
            None => vec![0.0; l.to_tensor().numel()],
        })
        .collect();
    Run {
        out_shape,
        out_values,
        grads,
    }
}

fn check_case(c: &Case, r: &Run) {
    assert_eq!(r.out_shape, c.out.shape, "{}: 出力 shape", c.name);
    assert_close_class(
        &r.out_values,
        &from_bits(&c.out.bits),
        &format!("{} forward", c.name),
    );
    if c.name.contains("nonfinite") {
        return;
    }
    for (i, g) in c.grads.iter().enumerate() {
        assert_close_class(&r.grads[i], &from_bits(g), &format!("{} grad[{i}]", c.name));
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
    assert!(fx.cases.len() >= 90, "fixture ケース数: {}", fx.cases.len());
    for op in ["kron", "tensordot", "tensordot_axes", "cdist", "cross"] {
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
        nonfinite.len() >= 5,
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

/// torch と成否が食い違う（torch は成功／本実装はエラー、またはその逆）ケース。
/// 決定記録 §5 と一対一。
const INTENDED_DIFFS: [&str; 4] = [
    "cross_broadcastable",
    "cdist_p_zero",
    "cdist_p_inf",
    "tensordot_size1_vs_n",
];

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
                "cross_axis_oor" | "cross_rank0" | "tensordot_axis_oor" => {
                    matches!(err, AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
                }
                "tensordot_dup_axes" => {
                    matches!(err, AutodiffError::Shape(ShapeError::DuplicateAxis { .. }))
                }
                "cdist_rank1" => {
                    matches!(err, AutodiffError::Shape(ShapeError::RankMismatch { .. }))
                }
                "cdist_batch_incompatible" => matches!(
                    err,
                    AutodiffError::Shape(ShapeError::BroadcastIncompatible { .. })
                ),
                _ => matches!(err, AutodiffError::InvalidArgument(_)),
            };
            assert!(kind_ok, "{}: 想定外のエラー種別 {err:?}", e.name);
        }
    }
    let mut expected: Vec<String> = INTENDED_DIFFS.iter().map(|s| s.to_string()).collect();
    diffs.sort();
    expected.sort();
    assert_eq!(diffs, expected, "意図的な差分の集合が決定記録 §5 と不一致");
}

// --- 独立オラクル（f64 総当たり + 中心差分） ---

/// 行優先 2 次元の f64 行列。
#[derive(Clone)]
struct M {
    r: usize,
    c: usize,
    d: Vec<f64>,
}

impl M {
    fn at(&self, i: usize, j: usize) -> f64 {
        self.d[i * self.c + j]
    }
}

fn oracle_kron(a: &M, b: &M) -> M {
    let (r, c) = (a.r * b.r, a.c * b.c);
    let mut d = vec![0.0; r * c];
    for i in 0..a.r {
        for j in 0..a.c {
            for k in 0..b.r {
                for l in 0..b.c {
                    d[(i * b.r + k) * c + (j * b.c + l)] = a.at(i, j) * b.at(k, l);
                }
            }
        }
    }
    M { r, c, d }
}

fn oracle_matmul(a: &M, b: &M) -> M {
    let mut d = vec![0.0; a.r * b.c];
    for i in 0..a.r {
        for j in 0..b.c {
            for k in 0..a.c {
                d[i * b.c + j] += a.at(i, k) * b.at(k, j);
            }
        }
    }
    M { r: a.r, c: b.c, d }
}

fn oracle_cdist(a: &M, b: &M, p: f64) -> M {
    let mut d = vec![0.0; a.r * b.r];
    for i in 0..a.r {
        for j in 0..b.r {
            let s: f64 = (0..a.c)
                .map(|k| (a.at(i, k) - b.at(j, k)).abs().powf(p))
                .sum();
            d[i * b.r + j] = s.powf(1.0 / p);
        }
    }
    M { r: a.r, c: b.r, d }
}

/// 行ごとの 3 次元外積（`dim = 1`）。
fn oracle_cross(a: &M, b: &M) -> M {
    let mut d = vec![0.0; a.r * 3];
    for i in 0..a.r {
        for k in 0..3 {
            let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
            d[i * 3 + k] = a.at(i, k1) * b.at(i, k2) - a.at(i, k2) * b.at(i, k1);
        }
    }
    M { r: a.r, c: 3, d }
}

fn sample(r: usize, c: usize, salt: usize) -> M {
    M {
        r,
        c,
        d: (0..r * c)
            .map(|i| f64::from(((i * 7 + salt * 5 + 3) % 17) as i32) * 0.3 - 2.0)
            .collect(),
    }
}

fn weights(r: usize, c: usize) -> M {
    M {
        r,
        c,
        d: (0..r * c)
            .map(|i| f64::from(((i * 5 + 2) % 13) as i32) * 0.125 - 0.75)
            .collect(),
    }
}

fn to_f32_tensor(m: &M) -> Tensor<f32> {
    t(m.d.iter().map(|&v| v as f32).collect(), &[m.r, m.c])
}

type OracleFn = dyn Fn(&M, &M) -> M;

/// forward を f64 総当たりと突合し、`Σ(y*w)` の解析勾配を f64 オラクルの中心差分と突合する。
fn check_against_oracle<F>(name: &str, a0: &M, b0: &M, ours: F, oracle: &OracleFn)
where
    F: for<'t> Fn(&Var<'t>, &Var<'t>) -> Result<Var<'t>, AutodiffError>,
{
    let expect = oracle(a0, b0);
    let w = weights(expect.r, expect.c);
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&to_f32_tensor(a0));
    let b = tape.var(&to_f32_tensor(b0));
    let y = ours(&a, &b).expect("forward");
    assert_eq!(
        y.to_tensor().shape(),
        &[expect.r, expect.c],
        "{name}: shape"
    );
    for (i, (g, e)) in y.to_tensor().host_slice().iter().zip(&expect.d).enumerate() {
        assert!(
            common::req2_close(f64::from(*g), *e),
            "{name} forward[{i}]: {g} vs {e}"
        );
    }
    let wv = tape.var_no_grad(&to_f32_tensor(&w));
    let loss = y.mul(&wv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let loss_of = |a: &M, b: &M| -> f64 {
        let o = oracle(a, b);
        o.d.iter().zip(&w.d).map(|(x, y)| x * y).sum()
    };
    let h = 1e-4;
    for (which, (v, base)) in [(&a, a0), (&b, b0)].into_iter().enumerate() {
        let analytic = grads
            .get(v)
            .unwrap()
            .expect("勾配")
            .host_slice()
            .into_owned();
        assert_eq!(analytic.len(), base.d.len());
        for (i, &analytic_i) in analytic.iter().enumerate() {
            let eval = |delta: f64| -> f64 {
                let (mut pa, mut pb) = (a0.clone(), b0.clone());
                if which == 0 {
                    pa.d[i] += delta;
                } else {
                    pb.d[i] += delta;
                }
                loss_of(&pa, &pb)
            };
            let numeric = (eval(h) - eval(-h)) / (2.0 * h);
            assert!(
                common::req2_close(f64::from(analytic_i), numeric),
                "{name} 入力 {which}[{i}]: analytic={analytic_i} numeric={numeric}"
            );
        }
    }
}

#[test]
fn f64_oracle_agrees_for_forward_and_gradient() {
    let (a, b) = (sample(2, 3, 1), sample(3, 2, 2));
    check_against_oracle("kron", &a, &b, kron, &oracle_kron);
    check_against_oracle(
        "tensordot_n1",
        &a,
        &b,
        |x, y| tensordot(x, y, 1),
        &oracle_matmul,
    );
    // 縮約軸を a の軸 0・b の軸 0 にする（転置を伴う）。
    let (a2, b2) = (sample(3, 2, 3), sample(3, 4, 4));
    check_against_oracle(
        "tensordot_axes_transposed",
        &a2,
        &b2,
        |x, y| tensordot_axes(x, y, &[0], &[0]),
        &|a, b| {
            let at = M {
                r: a.c,
                c: a.r,
                d: (0..a.c)
                    .flat_map(|j| (0..a.r).map(move |i| (i, j)))
                    .map(|(i, j)| a.at(i, j))
                    .collect(),
            };
            oracle_matmul(&at, b)
        },
    );
    // 距離 0 や |差| = 0 の非可微分点を避けた入力。
    let (x1, x2) = (sample(3, 4, 5), sample(2, 4, 6));
    for p in [1.0f32, 2.0, 3.0] {
        check_against_oracle(
            &format!("cdist_p{p}"),
            &x1,
            &x2,
            move |x, y| cdist(x, y, p),
            &move |a, b| oracle_cdist(a, b, f64::from(p)),
        );
    }
    let (c1, c2) = (sample(3, 3, 7), sample(3, 3, 8));
    check_against_oracle("cross", &c1, &c2, |x, y| cross(x, y, 1), &oracle_cross);
}

// --- バックエンド到達性（モック） ---

/// `scalar_binary`／`vector_norm`／`vector_norm_p`／`gather`／`scatter` だけを差し替える
/// `BackendOps`。他は naive へ委譲する。
struct ReachMock {
    inner: Box<dyn BackendOps + Send>,
    counters: Arc<[AtomicUsize; 5]>,
    launch_failed: bool,
}

const SCALAR_BINARY: usize = 0;
const VECTOR_NORM: usize = 1;
const VECTOR_NORM_P: usize = 2;
const GATHER: usize = 3;
const SCATTER: usize = 4;

impl ReachMock {
    fn answer(&self, which: usize) -> BackendError {
        self.counters[which].fetch_add(1, Ordering::SeqCst);
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
    fn scalar_binary(
        &self,
        _op: ScalarBinaryOp,
        _a: &Tensor<f32>,
        _b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(self.answer(SCALAR_BINARY))
    }
    fn vector_norm(
        &self,
        _a: &Tensor<f32>,
        _ord: VectorNormOrd,
        _dim: Option<usize>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(self.answer(VECTOR_NORM))
    }
    fn vector_norm_p(
        &self,
        _a: &Tensor<f32>,
        _p: f32,
        _dim: Option<usize>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(self.answer(VECTOR_NORM_P))
    }
    fn gather(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _index: &Tensor<i32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(self.answer(GATHER))
    }
    fn scatter(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _index: &Tensor<i32>,
        _src: &Tensor<f32>,
        _reduce: ScatterReduce,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(self.answer(SCATTER))
    }
}

struct Calls(Arc<[AtomicUsize; 5]>);

impl Calls {
    fn get(&self, which: usize) -> usize {
        self.0[which].load(Ordering::SeqCst)
    }
    fn all(&self) -> [usize; 5] {
        [0, 1, 2, 3, 4].map(|i| self.get(i))
    }
}

fn mock_tape(launch_failed: bool) -> (Tape, Calls) {
    let counters = Arc::new([const { AtomicUsize::new(0) }; 5]);
    let tape = Tape::new_with_ops(Box::new(ReachMock {
        inner: common::naive_ops(),
        counters: Arc::clone(&counters),
        launch_failed,
    }));
    (tape, Calls(counters))
}

fn seq(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    t((0..n).map(|i| (i as f32) * 0.5 + 1.0).collect(), shape)
}

fn backward_of<'t>(tape: &'t Tape, y: &Var<'t>) -> Result<(), AutodiffError> {
    let loss = y.mul(y)?.sum(None)?;
    tape.backward(&loss).map(|_| ())
}

fn assert_values_match_naive(
    mock: &Var<'_>,
    op: impl for<'t> Fn(&Var<'t>, &Var<'t>) -> Result<Var<'t>, AutodiffError>,
    shapes: (&[usize], &[usize]),
) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&seq(shapes.0));
    let b = tape.var(&seq(shapes.1));
    let want = op(&a, &b).unwrap();
    let (g, w) = (
        mock.to_tensor().host_slice().into_owned(),
        want.to_tensor().host_slice().into_owned(),
    );
    assert_eq!(g.len(), w.len());
    for (x, y) in g.iter().zip(&w) {
        assert!(common::req2_close(f64::from(*x), f64::from(*y)));
    }
}

#[test]
fn unsupported_backend_falls_back_to_host_reference() {
    for p in [2.0f32, 3.0, 1.0] {
        let (tape, calls) = mock_tape(false);
        let a = tape.var(&seq(&[3, 4]));
        let b = tape.var(&seq(&[2, 4]));
        let y = cdist(&a, &b, p).expect("Unsupported はホスト参照実装へ落ちる");
        assert!(
            calls.get(SCALAR_BINARY) >= 1,
            "減算で scalar_binary を先に呼ぶ"
        );
        let norm_calls = calls.get(VECTOR_NORM) + calls.get(VECTOR_NORM_P);
        assert!(
            norm_calls >= 1,
            "ノルムで vector_norm(_p) を先に呼ぶ（p={p}）"
        );
        assert_values_match_naive(&y, |x, z| cdist(x, z, p), (&[3, 4], &[2, 4]));
        backward_of(&tape, &y).expect("backward");
    }

    let (tape, calls) = mock_tape(false);
    let a = tape.var(&seq(&[2, 3]));
    let b = tape.var(&seq(&[2, 3]));
    let y = cross(&a, &b, 1).unwrap();
    assert!(calls.get(GATHER) >= 1, "roll は gather を先に呼ぶ");
    assert!(calls.get(SCALAR_BINARY) >= 1);
    assert_values_match_naive(&y, |x, z| cross(x, z, 1), (&[2, 3], &[2, 3]));
    assert_eq!(calls.get(SCATTER), 0);
    backward_of(&tape, &y).unwrap();
    assert!(
        calls.get(SCATTER) >= 1,
        "roll の backward は scatter を先に呼ぶ"
    );
}

#[test]
fn non_unsupported_backend_errors_are_propagated() {
    let (tape, _) = mock_tape(true);
    let a = tape.var(&seq(&[3, 4]));
    let b = tape.var(&seq(&[2, 4]));
    assert!(matches!(
        cdist(&a, &b, 2.0),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    let (tape, _) = mock_tape(true);
    let a = tape.var(&seq(&[2, 3]));
    let b = tape.var(&seq(&[2, 3]));
    assert!(matches!(
        cross(&a, &b, 1),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
}

#[test]
fn kron_and_tensordot_never_call_fallback_capable_methods() {
    let (tape, calls) = mock_tape(false);
    let a = tape.var(&seq(&[2, 3]));
    let b = tape.var(&seq(&[3, 2]));
    let k = kron(&a, &b).unwrap();
    let d = tensordot(&a, &b, 1).unwrap();
    let e = tensordot_axes(&a, &b, &[0], &[1]).unwrap();
    assert_eq!(
        calls.all(),
        [0; 5],
        "forward は必須メソッド（mul／gemm）のみ"
    );
    backward_of(&tape, &k).unwrap();
    backward_of(&tape, &d).unwrap();
    backward_of(&tape, &e).unwrap();
    assert_eq!(calls.all(), [0; 5], "backward も同様");
}

// --- テープ記録数・決定性・クロステープ ---

#[test]
fn tape_node_counts_are_fixed() {
    let count =
        |f: &dyn for<'a> Fn(Var<'a>, Var<'a>) -> Var<'a>, sa: &[usize], sb: &[usize]| -> usize {
            let tape = Tape::new_with_ops(common::naive_ops());
            let a = tape.var(&seq(sa));
            let b = tape.var(&seq(sb));
            let before = tape.len();
            let _ = f(a, b);
            tape.len() - before
        };
    // rank 2 同士の tensordot(a, b, 1) は matmul と同じ 1 ノード。
    assert_eq!(
        count(&|a, b| tensordot(&a, &b, 1).unwrap(), &[2, 3], &[3, 4]),
        1
    );
    assert_eq!(count(&|a, b| a.matmul(&b).unwrap(), &[2, 3], &[3, 4]), 1);
    // kron: reshape 2 + mul 1 + reshape 1。
    assert_eq!(count(&|a, b| kron(&a, &b).unwrap(), &[2, 3], &[3, 2]), 4);
    // cross: roll 4 回 + mul 2 + sub 1。
    let cross_nodes = count(&|a, b| cross(&a, &b, 1).unwrap(), &[2, 3], &[2, 3]);
    let roll_nodes = {
        let tape = Tape::new_with_ops(common::naive_ops());
        let a = tape.var(&seq(&[2, 3]));
        let before = tape.len();
        let _ = fandhe_ai_autodiff::rearrange_ops::roll(&a, &[-1], &[1]).unwrap();
        tape.len() - before
    };
    assert_eq!(cross_nodes, roll_nodes * 4 + 3);
}

#[test]
fn results_are_bit_deterministic_across_runs() {
    let fx = load_fixture();
    for c in fx.cases.iter().filter(|c| c.name.contains("nonfinite")) {
        let a = run_case(common::naive_ops(), c);
        let b = run_case(common::naive_ops(), c);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&a.out_values),
            bits(&b.out_values),
            "{} forward",
            c.name
        );
        for (ga, gb) in a.grads.iter().zip(&b.grads) {
            assert_eq!(bits(ga), bits(gb), "{} grad", c.name);
        }
    }
}

#[test]
fn kron_forward_is_single_multiplication_bit_exact() {
    let fx = load_fixture();
    for c in fx.cases.iter().filter(|c| c.op == "kron") {
        if c.name.contains("nonfinite") {
            continue;
        }
        let r = run_case(common::naive_ops(), c);
        let want = from_bits(&c.out.bits);
        let same = r
            .out_values
            .iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(
            same,
            "{}: kron の forward は PyTorch と bit 一致する想定",
            c.name
        );
    }
}

#[test]
fn ops_across_tapes_are_rejected() {
    let a_tape = Tape::new_with_ops(common::naive_ops());
    let b_tape = Tape::new_with_ops(common::naive_ops());
    let a = a_tape.var(&seq(&[3, 3]));
    let b = b_tape.var(&seq(&[3, 3]));
    assert!(matches!(kron(&a, &b), Err(AutodiffError::TapeMismatch)));
    assert!(matches!(
        tensordot(&a, &b, 1),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        cdist(&a, &b, 2.0),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(cross(&a, &b, 1), Err(AutodiffError::TapeMismatch)));
}
