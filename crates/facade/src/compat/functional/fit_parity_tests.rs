//! `compat::functional::train`（イシュー #2667）と PyTorch 2.14.0 実行値 fixture の照合
//! （`fandhe_ai_backend_cpu::parity::assert_parity`＝統一複合判定「相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満」。tolerance 定数は変更しない）。
//!
//! fixture は `tests/fixtures/functional-fit-pytorch-reference/`（出自・sha256・再生成手順は同 README）。
//! 照合するもの: (1) 多入力・fan-out・結合・多出力グラフの**パラメータ勾配**（`bind` 経路。
//! #2666 の fixture は入力勾配のみ）・(2) `fit` の epoch 損失と最終パラメータ（SGD〈momentum〉・
//! Adam・CrossEntropy）。実機 parity（CUDA／Metal）は `#[ignore]` で分離し、
//! `docs/perf/logs/functional-fit-2667/README.md` へ申し送る。
//!
//! 結合テスト（`tests/`）からは `pub(crate)` に届かないためクレート内に置く。

use std::collections::HashMap;

use fandhe_ai_backend_cpu::parity::assert_parity;
use serde_json::Value;

use super::{FunctionalBuilder, FunctionalModel, Node};
use crate::compat::{FitConfig, Loss, Optimizer, Sequential};
use crate::optim::{AdamConfig, SgdConfig};
use crate::{Tensor, Var};

fn bits_tensor(v: &Value) -> Tensor<f32> {
    let shape: Vec<usize> = v["shape"]
        .as_array()
        .expect("shape")
        .iter()
        .map(|d| d.as_u64().expect("dim") as usize)
        .collect();
    let data: Vec<f32> = v["bits"]
        .as_array()
        .expect("bits")
        .iter()
        .map(|b| f32::from_bits(b.as_u64().expect("bit") as u32))
        .collect();
    Tensor::new(data, &shape).expect("fixture tensor")
}

fn class_tensor(v: &Value) -> Tensor<i32> {
    let shape: Vec<usize> = v["shape"]
        .as_array()
        .expect("shape")
        .iter()
        .map(|d| d.as_u64().expect("dim") as usize)
        .collect();
    let data: Vec<i32> = v["values"]
        .as_array()
        .expect("values")
        .iter()
        .map(|b| b.as_i64().expect("class") as i32)
        .collect();
    Tensor::new(data, &shape).expect("fixture class tensor")
}

fn values(t: &Tensor<f32>) -> Vec<f32> {
    t.host_slice().into_owned()
}

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/functional-fit-pytorch-reference/functional_fit_reference.json"
    );
    let text = std::fs::read_to_string(path).expect("fixture を読めない");
    serde_json::from_str(&text).expect("fixture の JSON")
}

fn block_from_json(layers: &Value) -> Sequential {
    let mut seq = Sequential::new();
    for layer in layers.as_array().expect("layers") {
        seq = match layer["kind"].as_str().expect("kind") {
            "linear" => seq
                .add_linear(
                    layer["in"].as_u64().expect("in") as usize,
                    layer["out"].as_u64().expect("out") as usize,
                    1,
                )
                .expect("linear"),
            "relu" => seq.add_relu(),
            "tanh" => seq.add_tanh(),
            "sigmoid" => seq.add_sigmoid(),
            other => panic!("未対応の層 {other}"),
        };
    }
    seq
}

/// fixture のグラフ定義から `FunctionalModel` を組み、初期パラメータを読み込む。
fn model_from_case(case: &Value) -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let mut handles: Vec<Node> = Vec::new();
    for node in case["nodes"].as_array().expect("nodes") {
        let handle = match node["kind"].as_str().expect("kind") {
            "input" => b.input().expect("input"),
            "block" => {
                let src = node["input"].as_u64().expect("input") as usize;
                b.apply(block_from_json(&node["layers"]), handles[src])
                    .expect("apply")
            }
            "merge" => {
                let srcs: Vec<Node> = node["inputs"]
                    .as_array()
                    .expect("inputs")
                    .iter()
                    .map(|i| handles[i.as_u64().expect("index") as usize])
                    .collect();
                let dim = node["dim"].as_u64().expect("dim") as usize;
                match node["op"].as_str().expect("op") {
                    "concatenate" => b.concatenate(&srcs, dim).expect("concatenate"),
                    "add" => b.add(&srcs).expect("add"),
                    "multiply" => b.multiply(&srcs).expect("multiply"),
                    "average" => b.average(&srcs).expect("average"),
                    other => panic!("未対応の結合 {other}"),
                }
            }
            other => panic!("未対応のノード {other}"),
        };
        handles.push(handle);
    }
    let pick = |key: &str| -> Vec<Node> {
        case[key]
            .as_array()
            .expect(key)
            .iter()
            .map(|i| handles[i.as_u64().expect("index") as usize])
            .collect()
    };
    let mut model = b.build(&pick("inputs"), &pick("outputs")).expect("build");
    let params: HashMap<String, Tensor<f32>> = case["params"]
        .as_object()
        .expect("params")
        .iter()
        .map(|(k, v)| (k.clone(), bits_tensor(v)))
        .collect();
    model.load_state_dict(params).expect("重みの読み込み");
    model
}

fn inputs_of(case: &Value) -> Vec<Tensor<f32>> {
    case["xs"]
        .as_array()
        .expect("xs")
        .iter()
        .map(bits_tensor)
        .collect()
}

/// 損失 `Σ_k mse(out_k, y_k)`（出力の指定順の左畳み込み）を作り、パラメータ勾配を
/// （通し番号キー, 値）の列で返す。`tape` は CPU でも実機でもよい。
fn param_grads_on(
    model: &FunctionalModel,
    case: &Value,
    tape: &crate::Tape,
) -> (f32, Vec<(String, Vec<f32>)>) {
    let xs = inputs_of(case);
    let ys: Vec<Tensor<f32>> = case["ys"]
        .as_array()
        .expect("ys")
        .iter()
        .map(bits_tensor)
        .collect();
    let bound = model.bind(tape);
    let x_vars: Vec<Var<'_>> = xs.iter().map(|x| tape.var(x)).collect();
    let outs = bound.forward(tape, &x_vars).expect("forward");
    let mut loss: Option<Var<'_>> = None;
    for (out, y) in outs.iter().zip(&ys) {
        let yv = tape.var_no_grad(y);
        let term = out.mse_loss(&yv).expect("mse");
        loss = Some(match loss {
            Some(acc) => acc.add(&term).expect("add"),
            None => term,
        });
    }
    let loss = loss.expect("loss");
    let loss_value = loss.to_tensor().get(&[]).expect("scalar");
    let grads = tape.backward(&loss).expect("backward");
    let grad_refs = bound.trainable_grads(&grads).expect("trainable_grads");
    let names = model.named_parameters().expect("named");
    assert_eq!(names.len(), grad_refs.len());
    let out = names
        .iter()
        .zip(&grad_refs)
        .map(|((k, _), g)| (k.clone(), values(g)))
        .collect();
    (loss_value, out)
}

fn kind_cases(doc: &Value, kind: &str) -> Vec<Value> {
    doc["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .filter(|c| c["kind"] == kind)
        .cloned()
        .collect()
}

#[test]
fn parameter_gradients_match_pytorch() {
    let doc = fixture();
    assert!(
        doc["torch_version"]
            .as_str()
            .expect("v")
            .starts_with("2.14.0")
    );
    let cases = kind_cases(&doc, "grad");
    assert_eq!(cases.len(), 2);
    let tape = crate::tape();
    for case in &cases {
        let name = case["name"].as_str().expect("name");
        let model = model_from_case(case);
        let (loss, grads) = param_grads_on(&model, case, &tape);
        let want_loss = values(
            &Tensor::new(
                vec![f32::from_bits(
                    case["loss_bits"][0].as_u64().expect("bit") as u32
                )],
                &[],
            )
            .expect("loss"),
        );
        assert_parity(&format!("{name} loss"), &[loss], &want_loss);
        let expected = case["param_grads"].as_object().expect("param_grads");
        assert_eq!(grads.len(), expected.len(), "{name}: パラメータ数");
        for (key, got) in &grads {
            let want = expected.get(key).unwrap_or_else(|| panic!("{name}: {key}"));
            assert_parity(&format!("{name} d{key}"), got, &values(&bits_tensor(want)));
        }
    }
}

fn optimizer_of(case: &Value) -> Optimizer {
    let o = &case["optimizer"];
    let f = |k: &str| o[k].as_f64().expect(k) as f32;
    match o["kind"].as_str().expect("kind") {
        "sgd" => Optimizer::Sgd(SgdConfig::new(f("lr")).with_momentum(f("momentum"))),
        "adam" => Optimizer::Adam(AdamConfig {
            lr: f("lr"),
            beta1: f("beta1"),
            beta2: f("beta2"),
            eps: f("eps"),
            weight_decay: 0.0,
        }),
        other => panic!("未対応の optimizer {other}"),
    }
}

#[test]
fn fit_trajectories_match_pytorch() {
    let doc = fixture();
    let cases = kind_cases(&doc, "fit");
    assert_eq!(cases.len(), 3);
    for case in &cases {
        let name = case["name"].as_str().expect("name");
        let mut model = model_from_case(case);
        let xs = inputs_of(case);
        let x_refs: Vec<&Tensor<f32>> = xs.iter().collect();
        let config = FitConfig::new(
            case["epochs"].as_u64().expect("epochs") as usize,
            case["batch_size"].as_u64().expect("batch") as usize,
        );
        let history = match case["target_dtype"].as_str().expect("dtype") {
            "f32" => {
                model
                    .compile(optimizer_of(case), Loss::Mse)
                    .expect("compile");
                let ys: Vec<Tensor<f32>> = case["ys"]
                    .as_array()
                    .expect("ys")
                    .iter()
                    .map(bits_tensor)
                    .collect();
                let y_refs: Vec<&Tensor<f32>> = ys.iter().collect();
                model.fit(&x_refs, &y_refs, config).expect("fit")
            }
            "i32" => {
                model
                    .compile(optimizer_of(case), Loss::CrossEntropy)
                    .expect("compile");
                let ys: Vec<Tensor<i32>> = case["ys"]
                    .as_array()
                    .expect("ys")
                    .iter()
                    .map(class_tensor)
                    .collect();
                let y_refs: Vec<&Tensor<i32>> = ys.iter().collect();
                model.fit(&x_refs, &y_refs, config).expect("fit")
            }
            other => panic!("未対応の目標 dtype {other}"),
        };
        let want_losses: Vec<f32> = case["epoch_losses_bits"]
            .as_array()
            .expect("epoch_losses_bits")
            .iter()
            .map(|b| f32::from_bits(b[0].as_u64().expect("bit") as u32))
            .collect();
        assert_eq!(history.loss.len(), want_losses.len(), "{name}: epoch 数");
        assert_parity(&format!("{name} epoch loss"), &history.loss, &want_losses);
        let expected = case["final_params"].as_object().expect("final_params");
        let named = model.named_parameters().expect("named");
        assert_eq!(named.len(), expected.len(), "{name}: パラメータ数");
        for (key, got) in &named {
            let want = expected.get(key).unwrap_or_else(|| panic!("{name}: {key}"));
            assert_parity(
                &format!("{name} final {key}"),
                &values(got),
                &values(&bits_tensor(want)),
            );
        }
    }
}

// ------------------------------------------------- 実機 parity（#[ignore]）

/// 同じ初期重みで CPU tape と実機 tape の `bind → forward → backward` のパラメータ勾配を比較する
/// （`fit` 自体は `Sequential::fit` と同じく `crate::tape()` の CPU 経路のため対象外）。
fn device_matches_cpu(device: crate::Device) {
    let doc = fixture();
    for case in kind_cases(&doc, "grad") {
        let name = case["name"].as_str().expect("name").to_string();
        let model = model_from_case(&case);
        let cpu_tape = crate::tape();
        let (cpu_loss, cpu_grads) = param_grads_on(&model, &case, &cpu_tape);
        let device_tape = crate::tape_for(device).expect("実機 tape");
        let (dev_loss, dev_grads) = param_grads_on(&model, &case, &device_tape);
        assert_parity(&format!("{device:?} {name} loss"), &[dev_loss], &[cpu_loss]);
        for ((k, got), (_, want)) in dev_grads.iter().zip(&cpu_grads) {
            assert_parity(&format!("{device:?} {name} d{k}"), got, want);
        }
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。docs/perf/logs/functional-fit-2667/README.md"]
fn cuda_parameter_gradients_match_cpu_reference() {
    device_matches_cpu(crate::Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機依存。docs/perf/logs/functional-fit-2667/README.md"]
fn metal_parameter_gradients_match_cpu_reference() {
    device_matches_cpu(crate::Device::Metal);
}
