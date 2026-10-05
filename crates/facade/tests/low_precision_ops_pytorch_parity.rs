//! `fandhe_ai_autodiff::low_precision_ops`（イシュー #2628。MatMul と
//! elementwise 5 演算の opt-in 低精度 forward。facade 非公開のため
//! `fandhe_ai_autodiff::low_precision_ops::*` を直接 use する）の PyTorch
//! 参照値突合テスト（`docs/autodiff-low-precision-op-extension-decision.md`
//! §3.5 の P4・P5）。
//!
//! 参照値は実 PyTorch 2.14.0 の実行値 fixture
//! （`crates/autodiff/tests/fixtures/low-precision-ops-pytorch-reference/
//! low_precision_ops_reference.json`。生成条件は同ディレクトリの
//! `README.md`）。CI は Python／PyTorch に依存せずコミット済み JSON のみを読む。
//!
//! - **P4（forward）**: `fandhe_ai::tape()`（実 `CpuBackendOps`）上の低精度
//!   forward を PyTorch の `op(x.to(dtype), ..).float()` と REQ-2 統一複合判定
//!   （`fandhe_ai_backend_cpu::parity::compare`）で突合する。F16 が主判定・
//!   Bf16 が副判定。tolerance 定数は新設・変更しない。
//! - **P5（勾配）**: `MatMul`／`Add`／`Mul`／`Relu` の入力勾配を PyTorch の
//!   f32 autograd 勾配と突合する（案 C: backward は f32）。`Exp`／`Tanh` は
//!   観測値の出力のみで非ゲート（丸め済み forward 値を読む straight-through
//!   のため f32 勾配とは構造的に差が出る）。
//!
//! **判定不能の扱い（事前登録。決定記録 §3.5）**: 第三者比較対象（PyTorch）
//! 側の内部計算差で丸め先が 1 ulp ずれて統一複合判定を外れる組合せは、
//! [`INDETERMINATE`] に理由付きで列挙し、PyTorch 比較のゲートから外す代わりに
//! 丸めオラクルとの bit 一致を assert する。リストの各項目が fixture に実在
//! することも assert する（陳腐化防止）。

use std::path::PathBuf;

use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::low_precision_ops::{
    add_low_precision, exp_low_precision, matmul_low_precision, mul_low_precision,
    relu_low_precision, tanh_low_precision,
};
use fandhe_ai_backend_cpu::parity::compare;
use fandhe_ai_tensor_core::{BackendOps, ScalarDType, Tensor};
use serde_json::Value;

struct Fixture {
    torch_version: String,
    cases: Vec<Case>,
}

struct Tj {
    shape: Vec<usize>,
    data: Vec<f32>,
}

struct Forward {
    supported: bool,
    out: Option<Tj>,
}

struct Fwd {
    f16: Forward,
    bf16: Forward,
}

struct Case {
    name: String,
    op: String,
    inputs: Vec<Tj>,
    upstream: Tj,
    forward: Fwd,
    grad_inputs: Vec<Tj>,
}

// `facade` の dev-dependency は `serde_json` のみ（derive なし）のため、
// `serde_json::Value` から手で読む（`interop_onnx_external_data.rs` と同型）。
fn parse_tj(v: &Value) -> Tj {
    Tj {
        shape: v["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|x| x.as_u64().expect("shape 要素") as usize)
            .collect(),
        data: v["data"]
            .as_array()
            .expect("data")
            .iter()
            .map(|x| x.as_f64().expect("data 要素") as f32)
            .collect(),
    }
}

fn parse_forward(v: &Value) -> Forward {
    let supported = v["supported"].as_bool().expect("supported");
    Forward {
        supported,
        out: supported.then(|| parse_tj(&v["out"])),
    }
}

fn parse_fixture(root: &Value) -> Fixture {
    Fixture {
        torch_version: root["torch_version"]
            .as_str()
            .expect("torch_version")
            .to_string(),
        cases: root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .map(|c| Case {
                name: c["name"].as_str().expect("name").to_string(),
                op: c["op"].as_str().expect("op").to_string(),
                inputs: c["inputs"]
                    .as_array()
                    .expect("inputs")
                    .iter()
                    .map(parse_tj)
                    .collect(),
                upstream: parse_tj(&c["upstream"]),
                forward: Fwd {
                    f16: parse_forward(&c["forward"]["f16"]),
                    bf16: parse_forward(&c["forward"]["bf16"]),
                },
                grad_inputs: c["grad_inputs"]
                    .as_array()
                    .expect("grad_inputs")
                    .iter()
                    .map(parse_tj)
                    .collect(),
            })
            .collect(),
    }
}

/// fixture に含まれるべき全ケース名（実行された (op, dtype) 集合が
/// 「全ケース × {f16, bf16}」と完全一致することを固定する。fixture が
/// 「全部 unsupported」へ劣化しても通ってしまう事態を防ぐ）。
const EXPECTED_CASES: &[&str] = &[
    "matmul_2x3_3x2",
    "matmul_4x5_5x3",
    "matmul_batched",
    "matmul_exact",
    "add_same",
    "add_bias",
    "add_broadcast",
    "add_exact",
    "mul_same",
    "mul_bias",
    "mul_broadcast",
    "mul_exact",
    "relu_random",
    "relu_zero_mix",
    "relu_exact",
    "exp_random",
    "exp_exact",
    "tanh_random",
    "tanh_exact",
];

/// PyTorch 比較で判定不能と分類した (ケース名, dtype, 理由)。空でなければ
/// PR 本文で要ユーザー確認として報告する。
const INDETERMINATE: &[(&str, &str, &str)] = &[];

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../autodiff/tests/fixtures/low-precision-ops-pytorch-reference/low_precision_ops_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    let root: Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"));
    parse_fixture(&root)
}

fn tensor(t: &Tj) -> Tensor<f32> {
    let numel: usize = t.shape.iter().product();
    assert_eq!(numel, t.data.len(), "fixture: shape とデータ長が一致しない");
    Tensor::new(t.data.clone(), &t.shape).expect("fixture: shape とデータ長は一致")
}

fn run<'t>(
    op: &str,
    xs: &[Var<'t>],
    dtype: ScalarDType,
) -> Result<Var<'t>, fandhe_ai_autodiff::AutodiffError> {
    match op {
        "matmul" => matmul_low_precision(&xs[0], &xs[1], dtype),
        "add" => add_low_precision(&xs[0], &xs[1], dtype),
        "mul" => mul_low_precision(&xs[0], &xs[1], dtype),
        "relu" => relu_low_precision(&xs[0], dtype),
        "exp" => exp_low_precision(&xs[0], dtype),
        "tanh" => tanh_low_precision(&xs[0], dtype),
        other => panic!("未知の op: {other}"),
    }
}

fn dtype_of(key: &str) -> ScalarDType {
    match key {
        "f16" => ScalarDType::F16,
        "bf16" => ScalarDType::Bf16,
        other => panic!("未知の dtype: {other}"),
    }
}

fn round(dtype: ScalarDType, v: f32) -> f32 {
    match dtype {
        ScalarDType::F16 => fandhe_ai_tensor_core::f16::from_f32(v).to_f32(),
        _ => fandhe_ai_tensor_core::bf16::from_f32(v).to_f32(),
    }
}

fn round_t(dtype: ScalarDType, t: &Tensor<f32>) -> Tensor<f32> {
    let data: Vec<f32> = t.host_slice().iter().map(|&v| round(dtype, v)).collect();
    Tensor::new(data, t.shape()).unwrap()
}

/// 丸めオラクル: `round(f32_op(round(x...)))`（実 `CpuBackendOps` の f32 演算）。
fn oracle(op: &str, dtype: ScalarDType, xs: &[Tensor<f32>]) -> Tensor<f32> {
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let r: Vec<Tensor<f32>> = xs.iter().map(|x| round_t(dtype, x)).collect();
    let y = match op {
        "matmul" => cpu.gemm_batched(&r[0], &r[1]),
        "add" => cpu.add(&r[0], &r[1]),
        "mul" => cpu.mul(&r[0], &r[1]),
        "relu" => cpu.relu(&r[0]),
        "exp" => cpu.exp(&r[0]),
        "tanh" => cpu.tanh(&r[0]),
        other => panic!("未知の op: {other}"),
    }
    .expect("oracle の f32 演算は成功するはず");
    round_t(dtype, &y)
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.host_slice().iter().map(|v| v.to_bits()).collect()
}

fn is_indeterminate(name: &str, key: &str) -> bool {
    INDETERMINATE
        .iter()
        .any(|(n, k, _)| *n == name && *k == key)
}

fn forward_of<'a>(case: &'a Case, key: &str) -> &'a Forward {
    match key {
        "f16" => &case.forward.f16,
        _ => &case.forward.bf16,
    }
}

#[test]
fn fixture_covers_expected_op_dtype_set() {
    let fx = load_fixture();
    assert!(
        fx.torch_version.starts_with("2.14.0"),
        "torch_version: {}",
        fx.torch_version
    );
    let names: Vec<&str> = fx.cases.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, EXPECTED_CASES);
    for c in &fx.cases {
        assert!(
            c.forward.f16.supported,
            "{}: f16 が実行できていない",
            c.name
        );
        assert!(
            c.forward.bf16.supported,
            "{}: bf16 が実行できていない",
            c.name
        );
    }
    // 判定不能リストの項目が fixture に実在すること。
    for (n, k, why) in INDETERMINATE {
        assert!(
            fx.cases.iter().any(|c| c.name == *n) && (*k == "f16" || *k == "bf16"),
            "判定不能リストの項目が fixture に存在しない: {n} {k} ({why})"
        );
    }
}

/// P4: forward を PyTorch の低精度実行値と統一複合判定で突合する。
#[test]
fn forward_matches_pytorch_low_precision_execution() {
    let fx = load_fixture();
    for case in &fx.cases {
        for key in ["f16", "bf16"] {
            let dtype = dtype_of(key);
            let tape = fandhe_ai::tape();
            let xs_t: Vec<Tensor<f32>> = case.inputs.iter().map(tensor).collect();
            let xs: Vec<Var<'_>> = xs_t.iter().map(|t| tape.var(t)).collect();
            let y = run(&case.op, &xs, dtype).unwrap().to_tensor();
            let expected = tensor(forward_of(case, key).out.as_ref().unwrap());
            assert_eq!(y.shape(), expected.shape(), "{} {key}: shape", case.name);
            if is_indeterminate(&case.name, key) {
                // 判定不能: PyTorch 比較のゲートから外す代わりに丸めオラクルと
                // bit 一致を要求する。
                assert_eq!(
                    bits(&y),
                    bits(&oracle(&case.op, dtype, &xs_t)),
                    "{} {key}: 判定不能組合せは丸めオラクルと bit 一致が必要",
                    case.name
                );
                continue;
            }
            let report = compare(y.host_slice().as_ref(), expected.host_slice().as_ref()).unwrap();
            if !report.passes() {
                eprintln!(
                    "P4 FAIL {} {key}: fail_count={} max_abs={:e} max_rel={:e}",
                    case.name, report.fail_count, report.max_abs_diff, report.max_rel_err
                );
            }
            assert!(report.passes(), "{} {key}: {report:?}", case.name);
        }
    }
}

/// P5: 入力勾配を PyTorch の f32 autograd 勾配と突合する（Exp／Tanh は非ゲート）。
#[test]
fn input_gradients_match_pytorch_f32_autograd() {
    let fx = load_fixture();
    for case in &fx.cases {
        for key in ["f16", "bf16"] {
            let dtype = dtype_of(key);
            let tape = fandhe_ai::tape();
            let xs_t: Vec<Tensor<f32>> = case.inputs.iter().map(tensor).collect();
            let xs: Vec<Var<'_>> = xs_t.iter().map(|t| tape.var(t)).collect();
            let y = run(&case.op, &xs, dtype).unwrap();
            let g = tape.var_no_grad(&tensor(&case.upstream));
            let loss = y.mul(&g).unwrap().sum(None).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let gated = matches!(case.op.as_str(), "matmul" | "add" | "mul" | "relu");
            for (i, x) in xs.iter().enumerate() {
                let dx = grads
                    .get(x)
                    .unwrap()
                    .expect("勾配が到達するはず")
                    .contiguous();
                let expected = tensor(&case.grad_inputs[i]);
                assert_eq!(
                    dx.shape(),
                    expected.shape(),
                    "{} {key} in{i}: shape",
                    case.name
                );
                let report =
                    compare(dx.host_slice().as_ref(), expected.host_slice().as_ref()).unwrap();
                if gated {
                    assert!(report.passes(), "{} {key} in{i}: {report:?}", case.name);
                } else {
                    eprintln!(
                        "P5 observe (非ゲート) {} {key} in{i}: fail_count={} max_rel={:e}",
                        case.name, report.fail_count, report.max_rel_err
                    );
                }
            }
        }
    }
}
