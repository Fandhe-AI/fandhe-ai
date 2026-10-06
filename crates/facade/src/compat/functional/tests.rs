//! `compat::functional`（Functional API 内部実装。イシュー #2665）のクレート内ユニットテスト。
//!
//! 実装が `#[cfg(test)]` 限定の `pub(crate)` のため、テストは結合テスト（`tests/`）ではなく
//! ここに置く。構成: 検証規則（fail-closed）・単一連鎖の同値性・fan-out の評価回数・モード伝播・
//! 通し番号キー・strict `load_state_dict`・PyTorch 2.14.0 実行値 fixture との統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）・実機 parity（`#[ignore]`）。
//! tolerance 定数は変更しない。
//!
//! テスト関数・ヘルパーの名前は `tests/api_surface.rs` の workspace 走査が数える名前
//! （`state_dict` 等）と衝突させない。

use std::collections::HashMap;

use fandhe_ai_backend_cpu::parity::assert_parity;
use serde_json::Value;

use super::{FunctionalBuilder, FunctionalModel, Node, NodeDef, to_global_key};
use crate::compat::{Loss, Optimizer, Sequential};
use crate::optim::SgdConfig;
use crate::{AutodiffError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture tensor")
}

fn values(t: &Tensor<f32>) -> Vec<f32> {
    t.host_slice().into_owned()
}

fn is_invalid<T>(r: &Result<T, AutodiffError>) -> bool {
    matches!(r, Err(AutodiffError::InvalidArgument(_)))
}

/// Linear→ReLU の 1 ブロック。
fn lin_relu(i: usize, o: usize, seed: u64) -> Sequential {
    Sequential::new()
        .add_linear(i, o, seed)
        .expect("linear")
        .add_relu()
}

/// 入力 1・ブロック 1 の最小グラフ。
fn single_chain(block: Sequential) -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.apply(block, x).expect("apply");
    b.build(&[x], &[y]).expect("build")
}

fn sample_input(rows: usize, cols: usize, offset: f32) -> Tensor<f32> {
    let data = (0..rows * cols)
        .map(|k| ((k as f32) * 0.37 + offset).sin())
        .collect();
    t(data, &[rows, cols])
}

// ---------------------------------------------------------------- 検証規則

#[test]
fn build_rejects_empty_inputs_or_outputs() {
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.apply(lin_relu(2, 2, 1), x).expect("apply");
    assert!(is_invalid(&b.build(&[], &[y]).map(|_| ())));

    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let _ = b.apply(lin_relu(2, 2, 1), x).expect("apply");
    assert!(is_invalid(&b.build(&[x], &[]).map(|_| ())));
}

#[test]
fn handles_from_other_builders_are_rejected_everywhere() {
    let mut other = FunctionalBuilder::new();
    let foreign = other.input().expect("input");

    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    // apply の入力
    assert!(is_invalid(&b.apply(lin_relu(2, 2, 1), foreign).map(|_| ())));
    let y = b.apply(lin_relu(2, 2, 1), x).expect("apply");
    // inputs・outputs
    let mut b2 = FunctionalBuilder::new();
    let x2 = b2.input().expect("input");
    let y2 = b2.apply(lin_relu(2, 2, 1), x2).expect("apply");
    assert!(is_invalid(&b2.build(&[foreign], &[y2]).map(|_| ())));
    assert!(is_invalid(&b.build(&[x], &[foreign]).map(|_| ())));
    let _ = y;
}

#[test]
fn out_of_range_handle_is_rejected() {
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let bogus = Node {
        builder_id: x.builder_id,
        index: 99,
    };
    assert!(is_invalid(&b.apply(lin_relu(2, 2, 1), bogus).map(|_| ())));
}

#[test]
fn apply_rejects_compiled_and_empty_blocks() {
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let mut compiled = lin_relu(2, 2, 1);
    compiled
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .expect("compile");
    assert!(is_invalid(&b.apply(compiled, x).map(|_| ())));
    assert!(is_invalid(&b.apply(Sequential::new(), x).map(|_| ())));
}

#[test]
fn build_rejects_duplicates_non_inputs_unbound_and_dead_nodes() {
    // inputs の重複
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.apply(lin_relu(2, 2, 1), x).expect("apply");
    assert!(is_invalid(&b.build(&[x, x], &[y]).map(|_| ())));

    // outputs の重複
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.apply(lin_relu(2, 2, 1), x).expect("apply");
    assert!(is_invalid(&b.build(&[x], &[y, y]).map(|_| ())));

    // inputs にブロックノード
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.apply(lin_relu(2, 2, 1), x).expect("apply");
    assert!(is_invalid(&b.build(&[y], &[y]).map(|_| ())));

    // 未束縛の入力ノード（x2 が inputs に載らない）
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let x2 = b.input().expect("input");
    let y = b.apply(lin_relu(2, 2, 1), x).expect("apply");
    let _ = x2;
    assert!(is_invalid(&b.build(&[x], &[y]).map(|_| ())));

    // どの出力にも寄与しないブロック
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.apply(lin_relu(2, 2, 1), x).expect("apply");
    let _dead = b.apply(lin_relu(2, 2, 2), x).expect("apply");
    assert!(is_invalid(&b.build(&[x], &[y]).map(|_| ())));

    // 出力へ到達しない入力（入力 2 は列挙されているが誰にも使われない）
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let unused = b.input().expect("input");
    let y = b.apply(lin_relu(2, 2, 1), x).expect("apply");
    assert!(is_invalid(&b.build(&[x, unused], &[y]).map(|_| ())));
}

#[test]
fn passthrough_output_is_allowed() {
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.apply(lin_relu(2, 2, 1), x).expect("apply");
    let model = b.build(&[x], &[y, x]).expect("passthrough は許可");
    let input = sample_input(2, 2, 0.1);
    let outs = model.predict(&[&input]).expect("predict");
    assert_eq!(outs.len(), 2);
    assert_eq!(values(&outs[1]), values(&input));
}

#[test]
fn forward_rejects_wrong_input_count() {
    let model = single_chain(lin_relu(2, 2, 1));
    let input = sample_input(2, 2, 0.1);
    assert!(is_invalid(&model.predict(&[]).map(|_| ())));
    assert!(is_invalid(&model.predict(&[&input, &input]).map(|_| ())));
}

// ------------------------------------------------------------------ 同値性

#[test]
fn single_block_matches_sequential_bit_exact_and_predict_parity() {
    let seq = Sequential::new()
        .add_linear(4, 6, 7)
        .expect("linear")
        .add_tanh()
        .add_linear(6, 2, 8)
        .expect("linear");
    let model = single_chain(
        Sequential::new()
            .add_linear(4, 6, 7)
            .expect("linear")
            .add_tanh()
            .add_linear(6, 2, 8)
            .expect("linear"),
    );
    let input = sample_input(3, 4, 0.2);

    let tape = crate::tape();
    let x = tape.var(&input);
    let expected = seq.forward(&tape, &x).expect("seq forward");
    let got = model.forward(&tape, &[x]).expect("forward");
    assert_eq!(got.len(), 1);
    assert_eq!(values(&got[0].to_tensor()), values(&expected.to_tensor()));

    let predicted = model.predict(&[&input]).expect("predict");
    let seq_predicted = seq.predict(&input).expect("seq predict");
    assert_parity("predict", &values(&predicted[0]), &values(&seq_predicted));
}

#[test]
fn shared_upstream_is_evaluated_once_under_fan_out() {
    // 上流ブロックに BatchNorm（train で running stats を更新する副作用）を置き、
    // 2 ブロックが消費する。上流が 1 回だけ評価されれば、eval 後の出力は
    // 単独 Sequential を 1 回 forward した場合と一致する。
    let upstream = || {
        Sequential::new()
            .add_linear(3, 3, 11)
            .expect("linear")
            .add_batch_norm1d(3, 1e-5, 0.5)
            .expect("bn")
    };
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let shared = b.apply(upstream(), x).expect("apply");
    let left = b.apply(lin_relu(3, 2, 12), shared).expect("apply");
    let right = b.apply(lin_relu(3, 2, 13), shared).expect("apply");
    let mut model = b.build(&[x], &[shared, left, right]).expect("build");

    let mut reference = upstream();
    let input = sample_input(4, 3, 1.3);

    let tape = crate::tape();
    let xv = tape.var(&input);
    model.forward(&tape, &[xv]).expect("model forward");
    let rv = tape.var(&input);
    reference.forward(&tape, &rv).expect("reference forward");

    model.eval();
    reference.eval();
    let probe = sample_input(4, 3, -0.7);
    let outs = model.predict(&[&probe]).expect("predict");
    let expected = reference.predict(&probe).expect("reference predict");
    assert_eq!(values(&outs[0]), values(&expected));
}

#[test]
fn deep_chain_does_not_exhaust_the_stack() {
    const DEPTH: usize = 3000;
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let mut cur = x;
    for _ in 0..DEPTH {
        cur = b
            .apply(Sequential::new().add_identity(), cur)
            .expect("apply");
    }
    let model = b.build(&[x], &[cur]).expect("build");
    let input = sample_input(1, 2, 0.5);
    let outs = model.predict(&[&input]).expect("predict");
    assert_eq!(values(&outs[0]), values(&input));
}

// -------------------------------------------------------------------- モード

#[test]
fn mode_propagates_to_every_block_and_training_is_derived() {
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let drop = b
        .apply(Sequential::new().add_dropout(0.5).expect("dropout"), x)
        .expect("apply");
    let lin = b.apply(lin_relu(2, 2, 3), drop).expect("apply");
    let mut model = b.build(&[x], &[lin]).expect("build");

    assert!(model.training());
    model.eval();
    assert!(!model.training());
    let input = sample_input(4, 2, 0.9);
    let first = model.predict(&[&input]).expect("predict");
    let second = model.predict(&[&input]).expect("predict");
    assert_eq!(
        values(&first[0]),
        values(&second[0]),
        "eval の Dropout は決定的"
    );

    model.train();
    assert!(model.training());
    model.set_training(false);
    assert!(!model.training());

    // ブロック側のモードを直接変えた場合（build はモードを同期しない）の導出値。
    for def in &mut model.nodes {
        if let NodeDef::Block { block, .. } = def {
            block.set_training(true);
            break;
        }
    }
    assert!(!model.training(), "全ブロックが train でなければ false");
}

// ------------------------------------------------------- 通し番号キーと state

#[test]
fn global_key_splits_only_at_the_first_dot() {
    assert_eq!(to_global_key(4, "2.weight").expect("key"), "6.weight");
    assert_eq!(to_global_key(0, "3.a.b").expect("key"), "3.a.b");
    assert_eq!(to_global_key(10, "0.q.proj").expect("key"), "10.q.proj");
    assert!(to_global_key(0, "weight").is_err());
    assert!(to_global_key(0, "x.weight").is_err());
    assert!(to_global_key(usize::MAX, "1.weight").is_err());
}

fn two_block_model() -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let first = b
        .apply(
            Sequential::new()
                .add_linear(2, 3, 21)
                .expect("linear")
                .add_relu()
                .add_linear(3, 3, 22)
                .expect("linear"),
            x,
        )
        .expect("apply");
    let second = b.apply(lin_relu(3, 2, 23), first).expect("apply");
    b.build(&[x], &[second]).expect("build")
}

#[test]
fn parameter_keys_use_global_layer_numbering() {
    let model = two_block_model();
    let named = model.named_parameters().expect("named");
    let keys: Vec<&str> = named.iter().map(|(k, _)| k.as_str()).collect();
    // ブロック 1: 層 0(Linear)・1(ReLU)・2(Linear)。ブロック 2 は通し番号 3 から。
    assert_eq!(
        keys,
        [
            "0.weight", "0.bias", "2.weight", "2.bias", "3.weight", "3.bias"
        ]
    );
    let snapshot = model.state_dict().expect("snapshot");
    let mut a: Vec<&String> = snapshot.keys().collect();
    let mut b: Vec<&str> = keys.clone();
    a.sort();
    b.sort();
    assert_eq!(a, b);
}

#[test]
fn strict_load_roundtrips_and_rejects_without_mutation() {
    let mut model = two_block_model();
    let original = model.state_dict().expect("snapshot");

    // 往復: 別シードのモデルへ読み込むと出力が一致する。
    let mut other = {
        let mut b = FunctionalBuilder::new();
        let x = b.input().expect("input");
        let first = b
            .apply(
                Sequential::new()
                    .add_linear(2, 3, 91)
                    .expect("linear")
                    .add_relu()
                    .add_linear(3, 3, 92)
                    .expect("linear"),
                x,
            )
            .expect("apply");
        let second = b.apply(lin_relu(3, 2, 93), first).expect("apply");
        b.build(&[x], &[second]).expect("build")
    };
    other.load_state_dict(original.clone()).expect("load");
    let input = sample_input(3, 2, 0.4);
    assert_eq!(
        values(&model.predict(&[&input]).expect("p")[0]),
        values(&other.predict(&[&input]).expect("p")[0])
    );

    let before = model.state_dict().expect("before");
    let unchanged = |m: &FunctionalModel| {
        let now = m.state_dict().expect("now");
        before.len() == now.len()
            && before
                .iter()
                .all(|(k, v)| now.get(k).map(values) == Some(values(v)))
    };

    // 未知キー
    let mut with_extra = original.clone();
    with_extra.insert("9.weight".to_string(), t(vec![0.0], &[1]));
    assert!(is_invalid(&model.load_state_dict(with_extra)));
    assert!(unchanged(&model));

    // 欠落キー
    let mut missing: HashMap<String, Tensor<f32>> = original.clone();
    missing.remove("3.bias");
    assert!(is_invalid(&model.load_state_dict(missing)));
    assert!(unchanged(&model));

    // shape 不一致（最後のキーだけ壊しても先行ブロックが変わらない）
    let mut bad_shape = original.clone();
    bad_shape.insert("3.weight".to_string(), t(vec![0.0; 4], &[2, 2]));
    let mut poisoned = HashMap::new();
    for (k, v) in bad_shape {
        let v = if k == "0.weight" {
            t(vec![7.0; v.host_slice().len()], v.shape())
        } else {
            v
        };
        poisoned.insert(k, v);
    }
    assert!(is_invalid(&model.load_state_dict(poisoned)));
    assert!(unchanged(&model));
}

// ----------------------------------------------------- PyTorch fixture 照合

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
    t(data, &shape)
}

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/functional-graph-pytorch-reference/functional_graph_reference.json"
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

/// fixture のトポロジから `FunctionalModel` を組み、通し番号キーの重みを読み込む。
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

/// 1 ケースの出力と入力勾配を、与えた tape 上で fixture と照合する（CPU tape 用）。
fn check_case(case: &Value, tape: &crate::Tape) {
    let name = case["name"].as_str().expect("name");
    let model = model_from_case(case);
    let inputs: Vec<Tensor<f32>> = case["input_values"]
        .as_array()
        .expect("input_values")
        .iter()
        .map(bits_tensor)
        .collect();
    let vars: Vec<_> = inputs.iter().map(|i| tape.var(i)).collect();
    let outs = model.forward(tape, &vars).expect("forward");
    let expected_outs = case["outputs_value"].as_array().expect("outputs_value");
    assert_eq!(outs.len(), expected_outs.len(), "{name}: 出力数");
    for (k, (got, want)) in outs.iter().zip(expected_outs).enumerate() {
        assert_parity(
            &format!("{name} out[{k}]"),
            &values(&got.to_tensor()),
            &values(&bits_tensor(want)),
        );
    }
    // 損失 = Σ_k Σ(out_k ⊙ g_k)
    let mut loss: Option<crate::Var<'_>> = None;
    for (out, g) in outs
        .iter()
        .zip(case["upstream"].as_array().expect("upstream"))
    {
        let gv = tape.var_no_grad(&bits_tensor(g));
        let term = out.mul(&gv).expect("mul").sum(None).expect("sum");
        loss = Some(match loss {
            Some(acc) => acc.add(&term).expect("add"),
            None => term,
        });
    }
    let grads = tape.backward(&loss.expect("loss")).expect("backward");
    let expected_grads = case["input_grads"].as_array().expect("input_grads");
    for (k, (var, want)) in vars.iter().zip(expected_grads).enumerate() {
        let got = grads.get(var).expect("grad").expect("勾配が存在する");
        assert_parity(
            &format!("{name} dx[{k}]"),
            &values(got),
            &values(&bits_tensor(want)),
        );
    }
}

#[test]
fn matches_pytorch_reference_outputs_and_input_grads() {
    let doc = fixture();
    assert!(
        doc["torch_version"]
            .as_str()
            .expect("v")
            .starts_with("2.14.0")
    );
    let cases = doc["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 5);
    let tape = crate::tape();
    for case in cases {
        check_case(case, &tape);
    }
}

// ------------------------------------------------- 実機 parity（#[ignore]）

/// fan-out ケースを CPU tape と実機 tape で同じ重みのまま forward／backward して比較する。
fn device_matches_cpu(device: crate::Device) {
    let doc = fixture();
    let case = doc["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"] == "fan_out")
        .expect("fan_out ケース");
    let model = model_from_case(case);
    let input = bits_tensor(&case["input_values"][0]);

    let run = |tape: &crate::Tape| -> (Vec<Vec<f32>>, Vec<f32>) {
        let x = tape.var(&input);
        let outs = model.forward(tape, &[x]).expect("forward");
        let mut loss: Option<crate::Var<'_>> = None;
        for (out, g) in outs
            .iter()
            .zip(case["upstream"].as_array().expect("upstream"))
        {
            let gv = tape.var_no_grad(&bits_tensor(g));
            let term = out.mul(&gv).expect("mul").sum(None).expect("sum");
            loss = Some(match loss {
                Some(acc) => acc.add(&term).expect("add"),
                None => term,
            });
        }
        let grads = tape.backward(&loss.expect("loss")).expect("backward");
        let dx = grads.get(&x).expect("grad").expect("勾配が存在する");
        (
            outs.iter().map(|o| values(&o.to_tensor())).collect(),
            values(dx),
        )
    };
    let (cpu_outs, cpu_dx) = run(&crate::tape());
    let device_tape = crate::tape_for(device).expect("実機 tape");
    let (dev_outs, dev_dx) = run(&device_tape);
    for (k, (a, b)) in dev_outs.iter().zip(&cpu_outs).enumerate() {
        assert_parity(&format!("{device:?} out[{k}]"), a, b);
    }
    assert_parity(&format!("{device:?} dx"), &dev_dx, &cpu_dx);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。docs/perf/logs/functional-graph-2665/README.md"]
fn cuda_functional_graph_matches_cpu_reference() {
    device_matches_cpu(crate::Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機依存。docs/perf/logs/functional-graph-2665/README.md"]
fn metal_functional_graph_matches_cpu_reference() {
    device_matches_cpu(crate::Device::Metal);
}
