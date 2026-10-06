//! `softmin_threshold_ops`・`nn::softmin_threshold`（イシュー #2650・
//! `softmin`／`tanhshrink`／`threshold`／`rrelu`／`rrelu_with_noise`）の
//! `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/softmin-threshold-pytorch-reference/
//!   softmin_threshold_reference.json`・生成条件は同ディレクトリの
//!   `README.md`）の forward 出力と入力勾配を REQ-2 統一複合判定
//!   （`common::req2_close`）で突合する。tolerance 定数は新設しない。
//!   選択と乗算のみの演算（`threshold`・固定 noise の `rrelu_with_noise`）は
//!   実測で bit 一致が成立したため（`rrelu` 推論は傾きの `f32` 丸めで 1 ulp
//!   ずれうるため REQ-2 判定のみ。理由は `is_bit_exact_op` doc）、追加で bit 一致も
//!   固定する。`edge_cases` は値クラス（NaN／+inf／-inf／有限）で突合する。
//! - RReLU 学習時の乱数列は PyTorch と一致しないため、fixture の noise を
//!   `rrelu_with_noise` へ渡して検証する。乱数経路そのもの（グローバル RNG
//!   との対応・再現性・推論時の非消費）は `manual_seed` と `rand` で別途
//!   固定する（グローバル RNG を使うためファイル局所 `Mutex` で直列化）。
//!
//! `common::naive_ops()` は `scalar_unary` 等を持たない（既定 `Unsupported`）
//! ため、ホスト参照実装へのフォールバック経路を通る。CPU `BackendOps` 実装
//! との一致は `crates/facade/tests/softmin_threshold_ops_backend_parity.rs`
//! が担当する。

mod common;

use std::path::PathBuf;
use std::sync::Mutex;

use fandhe_ai_autodiff::nn::Module;
use fandhe_ai_autodiff::nn::softmin_threshold::{RRelu, Softmin, Tanhshrink, Threshold};
use fandhe_ai_autodiff::softmin_threshold_ops::{
    rrelu, rrelu_with_noise, softmin, tanhshrink, threshold,
};
use fandhe_ai_autodiff::{Tape, Var, manual_seed, rand};
use fandhe_ai_tensor_core::Tensor;
use serde::Deserialize;

/// グローバル RNG（`manual_seed`／`rand`）を触るテストの直列化用。
static RNG_LOCK: Mutex<()> = Mutex::new(());

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

#[derive(Deserialize, Default, Clone)]
struct Params {
    #[serde(default)]
    dim: usize,
    #[serde(default)]
    threshold: f32,
    #[serde(default)]
    value: f32,
    #[serde(default)]
    lower: f32,
    #[serde(default)]
    upper: f32,
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
    shape: Vec<usize>,
    params: Params,
    x: Vec<f32>,
    g: Vec<f32>,
    out: Vec<f32>,
    grad_x: Vec<f32>,
    #[serde(default)]
    noise: Vec<f32>,
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
    params: Params,
    x: Vec<Klass>,
    out: Vec<Klass>,
    grad_x: Vec<Klass>,
    #[serde(default)]
    noise: Vec<f32>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/softmin-threshold-pytorch-reference/softmin_threshold_reference.json",
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture を読めない: {}: {e}", path.display()));
    serde_json::from_str(&text).expect("fixture の JSON が不正")
}

fn klass_to_f32(k: &Klass) -> f32 {
    match k.class.as_str() {
        "nan" => f32::NAN,
        "pos_inf" => f32::INFINITY,
        "neg_inf" => f32::NEG_INFINITY,
        "finite" => k.value.expect("finite には value がある"),
        other => panic!("未知の class: {other}"),
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

fn assert_bits_all(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
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

/// op 名に応じて演算を適用する。`rrelu_train` は fixture の noise を渡す。
fn apply<'t>(
    op: &str,
    p: &Params,
    noise: &[f32],
    x: &Var<'t>,
    shape: &[usize],
) -> Result<Var<'t>, fandhe_ai_autodiff::AutodiffError> {
    match op {
        "softmin" => softmin(x, p.dim),
        "tanhshrink" => tanhshrink(x),
        "threshold" => threshold(x, p.threshold, p.value),
        "rrelu_eval" => rrelu(x, p.lower, p.upper, false),
        "rrelu_train" => rrelu_with_noise(x, &t(noise.to_vec(), shape)),
        other => panic!("未知の op: {other}"),
    }
}

/// 演算を実行して `(out, grad_x)` を返す。損失は `sum(y * g)`。
fn run(
    op: &str,
    p: &Params,
    noise: &[f32],
    x: &[f32],
    shape: &[usize],
    g: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&t(x.to_vec(), shape));
    let y = apply(op, p, noise, &xv, shape).unwrap_or_else(|e| panic!("{op}: forward 失敗: {e}"));
    let out = y.to_tensor().host_slice().into_owned();
    let gv = tape.var_no_grad(&t(g.to_vec(), shape));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("入力へ勾配が届く").clone();
    (out, dx.host_slice().into_owned())
}

/// 選択と IEEE 乗算 1 回のみで bit 一致が成立する演算。`rrelu_eval` は
/// 含めない: PyTorch は傾き `(lower + upper) / 2` を `f64` の `Scalar` から
/// 計算するが、本実装の `lower`／`upper` は `f32` 引数（`0.05` の `f32` 丸めが
/// `f64` の `0.05` と異なる）のため、傾きが 1 ulp ずれうる（実測: lower=0.05・
/// upper=0.6 で出力が 1 ulp 差）。REQ-2 判定では一致する。
fn is_bit_exact_op(op: &str) -> bool {
    matches!(op, "threshold" | "rrelu_train")
}

#[test]
fn matches_pytorch_reference_forward_and_backward() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    for op in [
        "softmin",
        "tanhshrink",
        "threshold",
        "rrelu_eval",
        "rrelu_train",
    ] {
        assert!(
            fixture.cases.iter().any(|c| c.op == op),
            "fixture に {op} のケースが無い"
        );
    }
    for case in &fixture.cases {
        let (out, dx) = run(
            &case.op,
            &case.params,
            &case.noise,
            &case.x,
            &case.shape,
            &case.g,
        );
        assert_close_all(&out, &case.out, &format!("{} forward", case.name));
        assert_close_all(&dx, &case.grad_x, &format!("{} grad", case.name));
        if is_bit_exact_op(&case.op) {
            assert_bits_all(&out, &case.out, &format!("{} forward(bit)", case.name));
            assert_bits_all(&dx, &case.grad_x, &format!("{} grad(bit)", case.name));
        }
    }
}

#[test]
fn edge_cases_match_pytorch_value_class() {
    let fixture = load_fixture();
    assert_eq!(fixture.edge_cases.len(), 8, "edge_cases の件数");
    for e in &fixture.edge_cases {
        let x: Vec<f32> = e.x.iter().map(klass_to_f32).collect();
        let n = x.len();
        let ones = vec![1.0; n];
        let (out, dx) = run(&e.op, &e.params, &e.noise, &x, &[n], &ones);
        assert_classes(&out, &e.out, &format!("{} forward", e.name));
        assert_classes(&dx, &e.grad_x, &format!("{} grad", e.name));
    }
}

#[test]
fn layers_match_free_functions_through_module_forward() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-1.5, -0.2, 0.7, 2.0, 0.4, -3.0], &[2, 3]));
    let same = |a: &Var<'_>, b: &Var<'_>| {
        assert_bits_all(
            a.to_tensor().host_slice().as_ref(),
            b.to_tensor().host_slice().as_ref(),
            "layer vs fn",
        );
    };
    same(
        &Module::forward(&Softmin::new(1), &tape, &x).unwrap(),
        &softmin(&x, 1).unwrap(),
    );
    same(
        &Module::forward(&Tanhshrink, &tape, &x).unwrap(),
        &tanhshrink(&x).unwrap(),
    );
    same(
        &Module::forward(&Threshold::new(0.3, -2.0), &tape, &x).unwrap(),
        &threshold(&x, 0.3, -2.0).unwrap(),
    );
    let mut r = RRelu::new(0.05, 0.6).unwrap();
    r.set_training(false);
    same(
        &Module::forward(&r, &tape, &x).unwrap(),
        &rrelu(&x, 0.05, 0.6, false).unwrap(),
    );
}

// --- 乱数経路（グローバル RNG） ---

/// `rrelu(training = true)` が `rand(shape)` を全要素分 1 回引いた一様列から
/// 期待 noise（`x <= 0` は `lower + (upper - lower) * u`、他は 1）を組み立てた
/// 結果と一致する。
#[test]
fn rrelu_training_consumes_rand_stream_as_documented() {
    let _guard = RNG_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let xs = vec![1.0, -1.0, 2.0, -0.5, 0.0, -3.0];
    let shape = [2usize, 3];
    let (lo, hi) = (0.1f32, 0.4f32);

    manual_seed(2650);
    let u = rand(&shape).unwrap();
    let expected_noise: Vec<f32> = xs
        .iter()
        .zip(u.host_slice().iter())
        .map(|(&v, &u)| {
            if v <= 0.0 {
                let r = f64::from(lo) + (f64::from(hi) - f64::from(lo)) * f64::from(u);
                (r as f32).clamp(lo, hi)
            } else {
                1.0
            }
        })
        .collect();

    manual_seed(2650);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(xs.clone(), &shape));
    let y = rrelu(&x, lo, hi, true).unwrap();
    let out = y.to_tensor().host_slice().into_owned();
    let expected_out: Vec<f32> = xs.iter().zip(&expected_noise).map(|(a, b)| a * b).collect();
    assert_bits_all(&out, &expected_out, "rrelu train forward");

    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
    assert_bits_all(&dx, &expected_noise, "rrelu train grad == noise");
}

#[test]
fn rrelu_training_is_reproducible_with_same_seed() {
    let _guard = RNG_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let xs = vec![-1.0, -2.0, -3.0, 1.0];
    let call = || {
        manual_seed(77);
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(xs.clone(), &[4]));
        rrelu(&x, 0.1, 0.9, true)
            .unwrap()
            .to_tensor()
            .host_slice()
            .into_owned()
    };
    assert_bits_all(&call(), &call(), "same seed");
}

#[test]
fn rrelu_eval_does_not_advance_rng_state() {
    let _guard = RNG_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let shape = [3usize];
    manual_seed(5);
    let baseline = rand(&shape).unwrap().host_slice().into_owned();

    manual_seed(5);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-1.0, 0.5, -2.0], &shape));
    let _ = rrelu(&x, 0.1, 0.3, false).unwrap();
    let mut layer = RRelu::default();
    layer.set_training(false);
    let _ = Module::forward(&layer, &tape, &x).unwrap();
    let after = rand(&shape).unwrap().host_slice().into_owned();
    assert_bits_all(&after, &baseline, "推論は RNG を消費しない");
}
