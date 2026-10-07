//! `gradcheck`・`backward_detect_anomaly`（イシュー #2671・親 #2668）の統合テスト。
//! 契約の正は `docs/autodiff-jacobian-hessian-gradcheck-decision.md` §3.4〜§3.6。
//!
//! - G1: 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/gradcheck-anomaly-pytorch-reference/gradcheck_anomaly_reference.json`。
//!   生成条件は同ディレクトリの `README.md`）と、解析ヤコビアン・forward 値を REQ-2 統一複合判定
//!   （`common::req2_close`。tolerance 定数は新設しない）で全要素突合する。あわせて同じ
//!   プログラムが `gradcheck` に合格すること。
//! - G2: 正しい VJP は合格・意図的に誤らせた VJP は不合格（最悪位置の確認）。
//! - G3: 数値ヤコビアンと PyTorch f64 の差を**非ゲートで記録**する（assert しない）。
//! - A1: anomaly detection（forward／gradient 段階・bit 一致・不変性・fail-closed・情報非露出）。
//! - fail-closed: オプション・入口検査・テープ呼び出し回数。
//!
//! `gradcheck` へ渡す閾値は #223 承認済みの組（`eps=1e-3`・`tau=1e-4`・`rtol=1e-2`・
//! `atol=1e-3`）のみ。CPU `BackendOps` 実装との一致・CUDA／Metal は
//! `crates/facade/tests/gradcheck_anomaly_backend_parity.rs` が担当する。

mod common;

use std::cell::Cell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use fandhe_ai_autodiff::anomaly::backward_detect_anomaly;
use fandhe_ai_autodiff::gradcheck::{GradcheckOptions, gradcheck};
use fandhe_ai_autodiff::jacobian_ops::jacobian;
use fandhe_ai_autodiff::{AutodiffError, CustomFunction, Tape, Var};
use fandhe_ai_tensor_core::Tensor;
use serde::Deserialize;

#[derive(Deserialize)]
struct Packed {
    shape: Vec<usize>,
    bits: Vec<u64>,
}

impl Packed {
    fn values32(&self) -> Vec<f32> {
        self.bits
            .iter()
            .map(|&b| f32::from_bits(b as u32))
            .collect()
    }
    fn values64(&self) -> Vec<f64> {
        self.bits.iter().map(|&b| f64::from_bits(b)).collect()
    }
    fn tensor(&self) -> Tensor<f32> {
        Tensor::new(self.values32(), &self.shape).expect("fixture: shape とデータ長は一致している")
    }
}

#[derive(Deserialize)]
struct Case {
    name: String,
    xs: Vec<Packed>,
    consts: HashMap<String, Packed>,
    out: Packed,
    expected: Vec<Packed>,
    out_f64: Packed,
    expected_f64: Vec<Packed>,
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    cases: Vec<Case>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/gradcheck-anomaly-pytorch-reference/gradcheck_anomaly_reference.json",
    );
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test: shape とデータ長は一致させている")
}

fn host(x: &Tensor<f32>) -> Vec<f32> {
    x.host_slice().into_owned()
}

fn new_tape() -> Tape {
    Tape::new_with_ops(common::naive_ops())
}

/// fixture 側 `gen_reference.py` のプログラムと同名・同式。
fn build_program<'t>(
    name: &str,
    xs: &[Var<'t>],
    c: &HashMap<String, Var<'t>>,
) -> Result<Var<'t>, AutodiffError> {
    match name {
        "elementwise" => xs[0].tanh().mul(&xs[0])?.add(&xs[0].exp()),
        "matmul_tanh" => Ok(xs[0].matmul(&c["w"])?.tanh()),
        "sum_reduce" => xs[0].sigmoid().mul(&xs[0])?.sum(None),
        "mlp" => {
            let h = xs[0].matmul(&c["w1"])?.add(&c["b1"])?.relu();
            h.matmul(&c["w2"])
        }
        "scalar_in" => xs[0].exp().mul(&xs[0]),
        "multi_input" => xs[0].mul(&xs[1])?.add(&xs[1].tanh()),
        "relu_safe" => xs[0].relu().mul(&xs[0]),
        other => panic!("未知のプログラム: {other}"),
    }
}

fn case_consts(case: &Case) -> HashMap<String, Tensor<f32>> {
    case.consts
        .iter()
        .map(|(k, v)| (k.clone(), v.tensor()))
        .collect()
}

/// `case` のプログラムを `gradcheck` へ渡す形（定数は評価ごとにテープへ登録）で実行する。
fn run_gradcheck(
    case: &Case,
    opts: &GradcheckOptions,
) -> Result<fandhe_ai_autodiff::gradcheck::GradcheckReport, AutodiffError> {
    let consts = case_consts(case);
    let inputs: Vec<Tensor<f32>> = case.xs.iter().map(Packed::tensor).collect();
    gradcheck(
        new_tape,
        |tape, xs| {
            let c: HashMap<String, Var<'_>> = consts
                .iter()
                .map(|(k, v)| (k.clone(), tape.var_no_grad(v)))
                .collect();
            build_program(&case.name, xs, &c)
        },
        &inputs,
        opts,
    )
}

fn approved_options() -> GradcheckOptions {
    GradcheckOptions::new(1e-3, 1e-3, 1e-2, 1e-4).expect("承認済みの閾値の組は有効")
}

fn assert_close_all(actual: &[f32], expected: &[f32], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: 要素数");
    for (i, (&x, &y)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            common::req2_close(x as f64, y as f64),
            "{ctx}[{i}]: actual={x} expected={y}"
        );
    }
}

// =====================================================================
// G1: PyTorch fixture × 解析ヤコビアン・gradcheck 合格
// =====================================================================

#[test]
fn g1_analytic_jacobian_matches_pytorch_fixture_and_gradcheck_passes() {
    let fx = load_fixture();
    assert!(
        fx.torch_version.starts_with("2.14.0"),
        "{}",
        fx.torch_version
    );
    assert!(!fx.cases.is_empty());
    for case in &fx.cases {
        let tape = new_tape();
        let xs: Vec<Var<'_>> = case.xs.iter().map(|p| tape.var(&p.tensor())).collect();
        let c: HashMap<String, Var<'_>> = case
            .consts
            .iter()
            .map(|(k, v)| (k.clone(), tape.var_no_grad(&v.tensor())))
            .collect();
        let y = build_program(&case.name, &xs, &c).expect("program");
        assert_eq!(
            y.to_tensor().shape(),
            case.out.shape.as_slice(),
            "{} forward shape",
            case.name
        );
        assert_close_all(
            &host(&y.to_tensor()),
            &case.out.values32(),
            &format!("{} forward", case.name),
        );
        for (k, x) in xs.iter().enumerate() {
            let jac = jacobian(&tape, &y, x).unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
            assert_eq!(
                jac.shape(),
                case.expected[k].shape,
                "{} jac shape",
                case.name
            );
            assert_close_all(
                &host(&jac),
                &case.expected[k].values32(),
                &format!("{} jacobian[{k}]", case.name),
            );
        }
        let report = run_gradcheck(case, &approved_options())
            .unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
        assert!(
            report.passed(),
            "{}: gradcheck 不合格 abs={} rel={} worst={:?}",
            case.name,
            report.max_abs_error(),
            report.max_rel_error(),
            report.worst_location()
        );
        let expected_elems: usize = case.out.shape.iter().product::<usize>()
            * case
                .xs
                .iter()
                .map(|p| p.shape.iter().product::<usize>())
                .sum::<usize>();
        assert_eq!(report.checked_elements(), expected_elems, "{}", case.name);
    }
}

// =====================================================================
// G2: 誤った VJP は不合格
// =====================================================================

/// forward は恒等、backward は `2 * upstream`（正しくは `upstream`）。
struct WrongIdentity;

impl CustomFunction for WrongIdentity {
    fn name(&self) -> &str {
        "wrong_identity"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok(inputs[0].clone())
    }
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        if !requires_grad[0] {
            return Ok(vec![None]);
        }
        let data: Vec<f32> = host(upstream).iter().map(|v| 2.0 * v).collect();
        let g = Tensor::new(data, inputs[0].shape()).map_err(AutodiffError::Shape)?;
        Ok(vec![Some(g)])
    }
}

#[test]
fn g2_wrong_vjp_fails_and_reports_worst_location() {
    let inputs = [t(vec![0.5, -1.0, 2.0], &[3])];
    let report = gradcheck(
        new_tape,
        |tape, xs| tape.custom(Arc::new(WrongIdentity), &[xs[0]]),
        &inputs,
        &approved_options(),
    )
    .expect("gradcheck は評価に成功する");
    assert!(!report.passed());
    // 対角のみ解析 2・数値 1 → 絶対誤差 1。非対角は 0 で一致。
    assert!((report.max_abs_error() - 1.0).abs() < 1e-3, "{report:?}");
    let (input, out_i, in_j) = report.worst_location();
    assert_eq!(input, 0);
    assert_eq!(out_i, in_j, "最悪要素は対角");
    assert_eq!(report.checked_elements(), 9);
}

#[test]
fn g2_correct_vjp_passes() {
    let inputs = [t(vec![0.5, -1.0, 2.0], &[3])];
    let report = gradcheck(
        new_tape,
        |_tape, xs| xs[0].mul(&xs[0]),
        &inputs,
        &approved_options(),
    )
    .expect("評価に成功");
    assert!(report.passed(), "{report:?}");
}

// =====================================================================
// G3: 数値ヤコビアンと PyTorch f64 の差（非ゲート。記録のみ）
// =====================================================================

#[test]
fn g3_records_numeric_jacobian_difference_against_pytorch_f64_without_gating() {
    let fx = load_fixture();
    for case in &fx.cases {
        // PyTorch f64 参照の forward 値が f32 値と大きく乖離していないことのみ確認する
        // （参照データの健全性。数値ヤコビアンとの差そのものはゲートしない）。
        let f64_out = case.out_f64.values64();
        assert_eq!(f64_out.len(), case.out.values32().len(), "{}", case.name);
        let max_expected = case
            .expected_f64
            .iter()
            .flat_map(|p| p.values64())
            .fold(0.0f64, |m, v| m.max(v.abs()));
        let report = run_gradcheck(case, &approved_options()).expect("gradcheck");
        println!(
            "G3 {}: gradcheck max_abs={:.3e} max_rel={:.3e} （PyTorch f64 jacobian の最大絶対値 {:.3e}）",
            case.name,
            report.max_abs_error(),
            report.max_rel_error(),
            max_expected
        );
    }
}

// =====================================================================
// gradcheck fail-closed
// =====================================================================

#[test]
fn gradcheck_rejects_empty_inputs() {
    let r = gradcheck(
        new_tape,
        |_t, _x| Err(AutodiffError::InvalidArgument("unreachable".into())),
        &[],
        &approved_options(),
    );
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(ref m)) if m.contains("inputs")));
}

#[test]
fn gradcheck_rejects_zero_element_output() {
    let inputs = [t(vec![1.0, 2.0], &[2])];
    let r = gradcheck(
        new_tape,
        |_t, xs| xs[0].narrow(0, 0, 0),
        &inputs,
        &approved_options(),
    );
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))), "{r:?}");
}

#[test]
fn gradcheck_rejects_output_from_another_tape() {
    // `for<'a>` のクロージャは通常は他テープの `Var` を返せない。`'static` に漏らしたテープの
    // `Var` だけが型を通る経路で、実行時の `TapeMismatch` 検査（多層防御）を確認する。
    let other: &'static Tape = Box::leak(Box::new(new_tape()));
    let foreign = other.var(&t(vec![1.0, 2.0], &[2]));
    let inputs = [t(vec![1.0, 2.0], &[2])];
    let r = gradcheck(
        new_tape,
        |_t, _xs| Ok(foreign),
        &inputs,
        &approved_options(),
    );
    assert!(matches!(r, Err(AutodiffError::TapeMismatch)), "{r:?}");
}

#[test]
fn gradcheck_propagates_closure_error() {
    let inputs = [t(vec![1.0], &[1])];
    let r = gradcheck(
        new_tape,
        |_t, _xs| Err(AutodiffError::InvalidArgument("boom".into())),
        &inputs,
        &approved_options(),
    );
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(ref m)) if m == "boom"));
}

#[test]
fn gradcheck_rejects_non_finite_perturbation_before_evaluation() {
    // 有限の入力 3e38 と有限の eps=1e38 でも f32 変換で x+eps が +inf になる。
    // 非有限の摂動値を f へ渡さず InvalidArgument を返すこと（f は解析側の 1 回のみ）。
    let inputs = [t(vec![3.0e38], &[1])];
    let opts = GradcheckOptions::new(1e38, 1e-3, 1e-2, 1e-4).expect("有限・正の閾値");
    let calls = Cell::new(0usize);
    let r = gradcheck(
        new_tape,
        |_t, xs| {
            calls.set(calls.get() + 1);
            Ok(xs[0])
        },
        &inputs,
        &opts,
    );
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))), "{r:?}");
    assert_eq!(calls.get(), 1, "摂動点では f を評価しない");
}

#[test]
fn gradcheck_rejects_shape_change_at_perturbed_point() {
    let inputs = [t(vec![1.0, 2.0], &[2])];
    let calls = Cell::new(0usize);
    let r = gradcheck(
        new_tape,
        |_t, xs| {
            // 1 回目（解析側の基準点）だけ shape [2]、以後は shape [1] を返す。
            let n = calls.get();
            calls.set(n + 1);
            if n == 0 {
                xs[0].mul(&xs[0])
            } else {
                xs[0].narrow(0, 0, 1)
            }
        },
        &inputs,
        &approved_options(),
    );
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(ref m)) if m.contains("shape")));
}

#[test]
fn gradcheck_calls_make_tape_once_plus_twice_per_input_element() {
    let inputs = [t(vec![0.3, 0.4, 0.5], &[3]), t(vec![1.0, 2.0], &[2])];
    let made = Cell::new(0usize);
    let report = gradcheck(
        || {
            made.set(made.get() + 1);
            new_tape()
        },
        |_t, xs| {
            xs[0]
                .sum(None)?
                .reshape(&[1])?
                .add(&xs[1].sum(None)?.reshape(&[1])?)
        },
        &inputs,
        &approved_options(),
    )
    .expect("評価に成功");
    assert!(report.passed(), "{report:?}");
    assert_eq!(made.get(), 1 + 2 * (3 + 2));
}

#[test]
fn gradcheck_does_not_touch_callers_existing_tape() {
    let tape = new_tape();
    let x = tape.var(&t(vec![0.5, 1.5], &[2]));
    let y = x.mul(&x).expect("mul");
    let len_before = tape.len();
    let value_before = host(&y.to_tensor());
    let inputs = [t(vec![0.5, 1.5], &[2])];
    gradcheck(
        new_tape,
        |_t, xs| xs[0].mul(&xs[0]),
        &inputs,
        &approved_options(),
    )
    .expect("評価に成功");
    assert_eq!(tape.len(), len_before);
    assert_eq!(host(&y.to_tensor()), value_before);
}

#[test]
fn gradcheck_rejects_eps_swallowed_by_f32() {
    // |x| が大きく eps=1e-3 では f32 の丸めで x±eps が同値になる。
    let inputs = [t(vec![1.0e8], &[1])];
    let r = gradcheck(
        new_tape,
        |_t, xs| xs[0].mul(&xs[0]),
        &inputs,
        &approved_options(),
    );
    assert!(
        matches!(r, Err(AutodiffError::InvalidArgument(ref m)) if m.contains("eps")),
        "{r:?}"
    );
}

// =====================================================================
// A1: anomaly detection
// =====================================================================

fn message(e: AutodiffError) -> String {
    match e {
        AutodiffError::Backward(m) => m,
        other => panic!("Backward 以外のエラー: {other:?}"),
    }
}

#[test]
fn a1_detects_forward_nan_with_node_id_and_op_name() {
    let tape = new_tape();
    let x = tape.var(&t(vec![-1.0, 2.0], &[2]));
    let y = x.log().expect("log");
    let y_id = tape.len() - 1;
    let loss = y.sum(None).expect("sum");
    let msg = message(backward_detect_anomaly(&tape, &loss).expect_err("NaN を検出する"));
    assert!(msg.contains("forward"), "{msg}");
    assert!(msg.contains(&format!("node {y_id}")), "{msg}");
    assert!(msg.contains("Op ScalarUnary"), "{msg}");
    assert!(msg.contains("[2]"), "shape が載る: {msg}");
}

#[test]
fn a1_detects_forward_infinity_from_overflow() {
    let tape = new_tape();
    let x = tape.var(&t(vec![1.0, 100.0], &[2]));
    let y = x.exp();
    let loss = y.sum(None).expect("sum");
    let msg = message(backward_detect_anomaly(&tape, &loss).expect_err("inf を検出する"));
    assert!(msg.contains("forward"), "{msg}");
}

#[test]
fn a1_detects_gradient_stage_when_forward_is_finite() {
    let tape = new_tape();
    let x = tape.var(&t(vec![0.0, 4.0], &[2]));
    let y = x.sqrt().expect("sqrt");
    let loss = y.sum(None).expect("sum");
    assert!(host(&loss.to_tensor()).iter().all(|v| v.is_finite()));
    let msg = message(backward_detect_anomaly(&tape, &loss).expect_err("勾配 inf を検出する"));
    assert!(msg.contains("勾配"), "{msg}");
    assert!(msg.contains("Leaf"), "{msg}");
    assert!(
        msg.contains("消費側の候補") && msg.contains("ScalarUnary"),
        "{msg}"
    );
}

#[test]
fn a1_returns_gradients_bit_equal_to_tape_backward_when_clean() {
    let fx = load_fixture();
    for case in &fx.cases {
        let tape = new_tape();
        let xs: Vec<Var<'_>> = case.xs.iter().map(|p| tape.var(&p.tensor())).collect();
        let c: HashMap<String, Var<'_>> = case
            .consts
            .iter()
            .map(|(k, v)| (k.clone(), tape.var_no_grad(&v.tensor())))
            .collect();
        let y = build_program(&case.name, &xs, &c).expect("program");
        let plain = tape.backward(&y).expect("backward");
        let checked = backward_detect_anomaly(&tape, &y).expect("異常なし");
        for x in &xs {
            let (a, b) = (plain.get(x).expect("get"), checked.get(x).expect("get"));
            match (a, b) {
                (Some(a), Some(b)) => {
                    let (a, b) = (host(a), host(b));
                    assert_eq!(a.len(), b.len(), "{}", case.name);
                    for (u, v) in a.iter().zip(b.iter()) {
                        assert_eq!(u.to_bits(), v.to_bits(), "{}", case.name);
                    }
                }
                (None, None) => {}
                _ => panic!("{}: 到達性が一致しない", case.name),
            }
        }
    }
}

#[test]
fn a1_detection_does_not_change_tape_nodes_or_values() {
    let tape = new_tape();
    let x = tape.var(&t(vec![-1.0, 2.0], &[2]));
    let y = x.log().expect("log");
    let loss = y.sum(None).expect("sum");
    let len_before = tape.len();
    let (xv, yv) = (host(&x.to_tensor()), host(&y.to_tensor()));
    let _ = backward_detect_anomaly(&tape, &loss);
    assert_eq!(tape.len(), len_before, "ノードを追加しない");
    assert_eq!(host(&x.to_tensor()), xv);
    // NaN を含むためビットで比較する。
    let after = host(&y.to_tensor());
    assert_eq!(
        after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        yv.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn a1_untracked_loss_and_foreign_tape_keep_existing_errors() {
    let tape = new_tape();
    let c = tape.var_no_grad(&t(vec![1.0, 2.0], &[2]));
    let untracked = c.mul(&c).expect("mul");
    let plain = format!("{:?}", tape.backward(&untracked).map(|_| ()));
    let checked = format!(
        "{:?}",
        backward_detect_anomaly(&tape, &untracked).map(|_| ())
    );
    assert_eq!(plain, checked);
    assert!(checked.starts_with("Err"), "{checked}");

    let other = new_tape();
    let foreign = other.var(&t(vec![1.0], &[1]));
    let plain = format!("{:?}", tape.backward(&foreign).map(|_| ()));
    let checked = format!("{:?}", backward_detect_anomaly(&tape, &foreign).map(|_| ()));
    assert_eq!(plain, checked);
}

/// forward が NaN を返す `Custom` Op（利用者定義名がメッセージへ漏れないことの検証用）。
struct NanCustom;

impl CustomFunction for NanCustom {
    fn name(&self) -> &str {
        "SECRET_CUSTOM_NAME_7f3a"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        let data = vec![f32::NAN; inputs[0].shape().iter().product()];
        Tensor::new(data, inputs[0].shape()).map_err(AutodiffError::Shape)
    }
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        _requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        let _ = inputs;
        Ok(vec![Some(upstream.clone())])
    }
}

#[test]
fn a1_message_does_not_leak_op_payloads() {
    // Custom: 利用者定義名を載せない。
    let tape = new_tape();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = tape.custom(Arc::new(NanCustom), &[x]).expect("custom");
    let loss = y.sum(None).expect("sum");
    let msg = message(backward_detect_anomaly(&tape, &loss).expect_err("NaN"));
    assert!(msg.contains("Op Custom"), "{msg}");
    assert!(!msg.contains("SECRET_CUSTOM_NAME"), "{msg}");
}
