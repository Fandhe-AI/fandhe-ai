//! `compat::functional` の結合ノード（Concatenate／Add／Multiply／Average。イシュー #2666）の
//! クレート内ユニットテスト。
//!
//! 実装が `#[cfg(test)]` 限定の `pub(crate)` のため、結合テスト（`tests/`）ではなくここに置く
//! （`tests.rs` が #2665 の範囲、本ファイルが #2666 の範囲）。構成: 構築時の検証規則・
//! 到達性伝播の回帰・`merge_ops` 直接呼び出しとの bit 一致・forward 時の型付きエラー・
//! 通し番号キーの不変性・PyTorch 2.14.0 実行値 fixture との統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）・実機 parity（`#[ignore]`）。
//! tolerance 定数は変更しない。
//!
//! テスト関数・ヘルパーの名前は `tests/api_surface.rs` の workspace 走査が数える名前
//! （`merge_*` の 4 関数名・`concatenate`／`multiply`／`average` 等）と衝突させない。

use std::collections::HashMap;

use fandhe_ai_autodiff::merge_ops;
use fandhe_ai_backend_cpu::parity::assert_parity;
use serde_json::Value;

use super::{FunctionalBuilder, FunctionalModel, Node, NodeDef};
use crate::compat::Sequential;
use crate::{AutodiffError, Tensor, Var};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture tensor")
}

fn values(t: &Tensor<f32>) -> Vec<f32> {
    t.host_slice().into_owned()
}

fn bit_pattern(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn is_invalid<T>(r: &Result<T, AutodiffError>) -> bool {
    matches!(r, Err(AutodiffError::InvalidArgument(_)))
}

fn lin_relu(i: usize, o: usize, seed: u64) -> Sequential {
    Sequential::new()
        .add_linear(i, o, seed)
        .expect("linear")
        .add_relu()
}

fn sample_input(rows: usize, cols: usize, offset: f32) -> Tensor<f32> {
    let data = (0..rows * cols)
        .map(|k| ((k as f32) * 0.37 + offset).sin())
        .collect();
    t(data, &[rows, cols])
}

/// 種別を文字列で選ぶ薄いディスパッチ（4 種の検証規則を同じ本文で回すため）。
fn build_merge(
    b: &mut FunctionalBuilder,
    kind: &str,
    inputs: &[Node],
) -> Result<Node, AutodiffError> {
    match kind {
        "concatenate" => b.concatenate(inputs, 1),
        "add" => b.add(inputs),
        "multiply" => b.multiply(inputs),
        "average" => b.average(inputs),
        other => panic!("未知の種別 {other}"),
    }
}

const KINDS: [&str; 4] = ["concatenate", "add", "multiply", "average"];

// ---------------------------------------------------------------- 構築時の検証規則

#[test]
fn every_kind_rejects_fewer_than_two_inputs() {
    for kind in KINDS {
        let mut b = FunctionalBuilder::new();
        let x = b.input().expect("input");
        let before = b.nodes.len();
        assert!(is_invalid(&build_merge(&mut b, kind, &[])), "{kind}: 0 件");
        assert!(is_invalid(&build_merge(&mut b, kind, &[x])), "{kind}: 1 件");
        assert_eq!(b.nodes.len(), before, "{kind}: 拒否でノードを増やさない");
    }
}

#[test]
fn every_kind_rejects_foreign_out_of_range_and_duplicate_handles() {
    let mut other = FunctionalBuilder::new();
    let foreign = other.input().expect("input");
    for kind in KINDS {
        let mut b = FunctionalBuilder::new();
        let x = b.input().expect("input");
        let y = b.input().expect("input");
        let bogus = Node {
            builder_id: x.builder_id,
            index: 99,
        };
        let before = b.nodes.len();
        assert!(
            is_invalid(&build_merge(&mut b, kind, &[x, foreign])),
            "{kind}: 他ビルダー"
        );
        assert!(
            is_invalid(&build_merge(&mut b, kind, &[x, bogus])),
            "{kind}: 範囲外"
        );
        assert!(
            is_invalid(&build_merge(&mut b, kind, &[x, y, x])),
            "{kind}: 重複"
        );
        assert_eq!(b.nodes.len(), before, "{kind}: 拒否でノードを増やさない");
        assert!(build_merge(&mut b, kind, &[x, y]).is_ok(), "{kind}: 正常");
    }
}

#[test]
fn build_rejects_dead_merge_and_merge_in_inputs() {
    // どの出力にも寄与しない結合ノード
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let a = b.apply(lin_relu(2, 2, 1), x).expect("apply");
    let c = b.apply(lin_relu(2, 2, 2), x).expect("apply");
    let _dead = b.add(&[a, c]).expect("add");
    assert!(is_invalid(&b.build(&[x], &[a]).map(|_| ())));

    // inputs に結合ノードは指定できない
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.input().expect("input");
    let m = b.add(&[x, y]).expect("add");
    assert!(is_invalid(&b.build(&[x, m], &[m]).map(|_| ())));
}

#[test]
fn reachability_propagates_through_merge_inputs() {
    // 入力 y とブロック a は結合ノード経由でのみ出力へ寄与する（到達性伝播の回帰）。
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.input().expect("input");
    let a = b.apply(lin_relu(2, 2, 1), x).expect("apply");
    let m = b.add(&[a, y]).expect("add");
    let model = b.build(&[x, y], &[m]).expect("結合経由の寄与は到達扱い");
    let out = model
        .predict(&[&sample_input(2, 2, 0.1), &sample_input(2, 2, 0.2)])
        .expect("predict");
    assert_eq!(out[0].shape(), &[2, 2]);
}

// ------------------------------------------------- merge_ops 直接呼び出しとの bit 一致

/// 入力 `x`（`[rows, 3]`）・ブロック A・ブロック B から `kind` で結合するグラフを組み、
/// 同じ tape 上で `merge_ops` を直接呼んだ結果と forward 値・勾配が bit 一致することを確かめる。
fn merge_graph_matches_direct_call(kind: &str) {
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let a = b.apply(lin_relu(3, 3, 41), x).expect("apply");
    let c = b.apply(lin_relu(3, 3, 42), x).expect("apply");
    let m = build_merge(&mut b, kind, &[a, c]).expect("merge");
    let model = b.build(&[x], &[m]).expect("build");

    let input = sample_input(4, 3, 0.4);
    let tape = crate::tape();

    let xv = tape.var(&input);
    let got = model.forward(&tape, &[xv]).expect("forward");
    let got_out = values(&got[0].to_tensor());
    let got_loss = got[0].sum(None).expect("sum");
    let got_grads = tape.backward(&got_loss).expect("backward");
    let got_dx = values(got_grads.get(&xv).expect("grad").expect("勾配が存在する"));

    let ra = lin_relu(3, 3, 41);
    let rc = lin_relu(3, 3, 42);
    let xr = tape.var(&input);
    let ya = ra.forward(&tape, &xr).expect("a");
    let yc = rc.forward(&tape, &xr).expect("c");
    let direct = match kind {
        "concatenate" => merge_ops::merge_concatenate(&[ya, yc], 1),
        "add" => merge_ops::merge_add(&[ya, yc]),
        "multiply" => merge_ops::merge_multiply(&[ya, yc]),
        "average" => merge_ops::merge_average(&[ya, yc]),
        other => panic!("未知の種別 {other}"),
    }
    .expect("direct");
    let want_out = values(&direct.to_tensor());
    let want_loss = direct.sum(None).expect("sum");
    let want_grads = tape.backward(&want_loss).expect("backward");
    let want_dx = values(want_grads.get(&xr).expect("grad").expect("勾配が存在する"));

    assert_eq!(bit_pattern(&got_out), bit_pattern(&want_out), "{kind}");
    assert_eq!(bit_pattern(&got_dx), bit_pattern(&want_dx), "{kind} dx");
}

#[test]
fn every_kind_matches_direct_merge_call_bit_exact() {
    for kind in KINDS {
        merge_graph_matches_direct_call(kind);
    }
}

// ------------------------------------------------------------ forward 時の型付きエラー

#[test]
fn forward_reports_shape_errors_as_typed_errors() {
    // 幅の異なる 2 入力（add 系は shape 不一致・concatenate は dim 1 で幅が違っても連結可能）。
    for kind in ["add", "multiply", "average"] {
        let mut b = FunctionalBuilder::new();
        let x = b.input().expect("input");
        let y = b.input().expect("input");
        let m = build_merge(&mut b, kind, &[x, y]).expect("merge");
        let model = b.build(&[x, y], &[m]).expect("build");
        let tape = crate::tape();
        let wide = tape.var(&sample_input(2, 3, 0.0));
        let narrow = tape.var(&sample_input(2, 2, 0.0));
        let row = tape.var(&sample_input(1, 3, 0.0));
        let before = tape.0.len();
        assert!(
            matches!(
                model.forward(&tape, &[wide, narrow]),
                Err(AutodiffError::Shape(_))
            ),
            "{kind}: shape 不一致"
        );
        // broadcast 可能な shape も黙って通さない。
        assert!(
            matches!(
                model.forward(&tape, &[wide, row]),
                Err(AutodiffError::Shape(_))
            ),
            "{kind}: broadcast 可能 shape"
        );
        assert_eq!(tape.0.len(), before, "{kind}: 拒否でノードを増やさない");
    }

    // concatenate: dim 範囲外・非連結軸の不一致
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.input().expect("input");
    let oob = b.concatenate(&[x, y], 2).expect("構築時は shape を見ない");
    let model = b.build(&[x, y], &[oob]).expect("build");
    let tape = crate::tape();
    let p = tape.var(&sample_input(2, 3, 0.0));
    let q = tape.var(&sample_input(2, 3, 0.5));
    assert!(matches!(
        model.forward(&tape, &[p, q]),
        Err(AutodiffError::Shape(_))
    ));

    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.input().expect("input");
    let cat = b.concatenate(&[x, y], 1).expect("concatenate");
    let model = b.build(&[x, y], &[cat]).expect("build");
    let tape = crate::tape();
    let p = tape.var(&sample_input(2, 3, 0.0));
    let q = tape.var(&sample_input(3, 3, 0.5));
    assert!(matches!(
        model.forward(&tape, &[p, q]),
        Err(AutodiffError::Shape(_))
    ));
}

// ------------------------------------------------------------ 残差・fan-out・結合のみ

#[test]
fn residual_with_fan_out_evaluates_shared_upstream_once() {
    // BatchNorm（train で running stats を更新する副作用）を持つ共有上流を、結合ノードと
    // ブロックの双方が消費する。上流が 1 回だけ評価されれば、eval 後の出力は単独 Sequential を
    // 1 回 forward した場合と一致する。
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
    let branch = b.apply(lin_relu(3, 3, 12), shared).expect("apply");
    let sum = b.add(&[shared, branch]).expect("add");
    let mut model = b.build(&[x], &[sum]).expect("build");

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
    let shared_expected = reference.predict(&probe).expect("reference predict");
    let mut branch_ref = lin_relu(3, 3, 12);
    branch_ref.eval();
    let branch_expected = branch_ref.predict(&shared_expected).expect("branch");
    let want: Vec<f32> = values(&shared_expected)
        .iter()
        .zip(values(&branch_expected))
        .map(|(a, c)| a + c)
        .collect();
    let got = model.predict(&[&probe]).expect("predict");
    assert_eq!(bit_pattern(&values(&got[0])), bit_pattern(&want));
}

#[test]
fn merge_only_graph_has_no_parameters_and_is_in_train_mode() {
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.input().expect("input");
    let s = b.add(&[x, y]).expect("add");
    let model = b.build(&[x, y], &[s]).expect("build");
    assert!(model.training(), "ブロック 0 個なら training() は true");
    assert!(model.state_dict().expect("state_dict").is_empty());
    assert!(model.named_parameters().expect("named").is_empty());
    let a = sample_input(2, 2, 0.0);
    let c = sample_input(2, 2, 1.0);
    let out = model.predict(&[&a, &c]).expect("predict");
    let want: Vec<f32> = values(&a)
        .iter()
        .zip(values(&c))
        .map(|(p, q)| p + q)
        .collect();
    assert_eq!(values(&out[0]), want);
}

// --------------------------------------------- 通し番号キー・strict load・モードの不変性

fn chain_with_merge_between_blocks() -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let first = b.apply(lin_relu(2, 3, 21), x).expect("apply");
    let side = b.apply(lin_relu(2, 3, 22), x).expect("apply");
    let merged = b.add(&[first, side]).expect("add");
    let second = b.apply(lin_relu(3, 2, 23), merged).expect("apply");
    b.build(&[x], &[second]).expect("build")
}

#[test]
fn merge_nodes_do_not_shift_global_layer_keys() {
    let model = chain_with_merge_between_blocks();
    let mut keys: Vec<String> = model
        .state_dict()
        .expect("state_dict")
        .into_keys()
        .collect();
    keys.sort();
    // Block 3 個（各 Linear+ReLU の 2 層）の通し番号。結合ノードは層を持たない。
    assert_eq!(
        keys,
        vec![
            "0.bias", "0.weight", "2.bias", "2.weight", "4.bias", "4.weight"
        ]
    );
}

#[test]
fn strict_load_roundtrips_and_mode_propagates_across_merge() {
    let mut model = chain_with_merge_between_blocks();
    let snapshot: HashMap<String, Tensor<f32>> = model.state_dict().expect("state_dict");
    let input = sample_input(3, 2, 0.8);
    model.eval();
    let before = values(&model.predict(&[&input]).expect("predict")[0]);

    let mut fresh = chain_with_merge_between_blocks();
    fresh.load_state_dict(snapshot).expect("strict load");
    fresh.eval();
    let after = values(&fresh.predict(&[&input]).expect("predict")[0]);
    assert_eq!(bit_pattern(&before), bit_pattern(&after));

    let mut missing = model.state_dict().expect("state_dict");
    missing.remove("0.weight");
    assert!(is_invalid(&fresh.load_state_dict(missing)));

    assert!(!model.training());
    model.train();
    assert!(model.training());
    for def in &model.nodes {
        if let NodeDef::Block { block, .. } = def {
            assert!(block.training());
        }
    }
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
        "/tests/fixtures/functional-merge-pytorch-reference/functional_merge_reference.json"
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

/// 損失 `Σ_k Σ(out_k ⊙ g_k)` を作り、全入力の勾配とともに返す。
fn run_with_loss<'t>(
    model: &FunctionalModel,
    case: &Value,
    tape: &'t crate::Tape,
    inputs: &[Tensor<f32>],
) -> (Vec<Var<'t>>, Vec<Var<'t>>, Vec<Vec<f32>>) {
    let vars: Vec<Var<'t>> = inputs.iter().map(|i| tape.var(i)).collect();
    let outs = model.forward(tape, &vars).expect("forward");
    let mut loss: Option<Var<'t>> = None;
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
    let dxs = vars
        .iter()
        .map(|v| values(grads.get(v).expect("grad").expect("勾配が存在する")))
        .collect();
    (vars, outs, dxs)
}

fn check_case(case: &Value, tape: &crate::Tape) {
    let name = case["name"].as_str().expect("name");
    let model = model_from_case(case);
    let inputs: Vec<Tensor<f32>> = case["input_values"]
        .as_array()
        .expect("input_values")
        .iter()
        .map(bits_tensor)
        .collect();
    let (_, outs, dxs) = run_with_loss(&model, case, tape, &inputs);
    let expected_outs = case["outputs_value"].as_array().expect("outputs_value");
    assert_eq!(outs.len(), expected_outs.len(), "{name}: 出力数");
    for (k, (got, want)) in outs.iter().zip(expected_outs).enumerate() {
        assert_parity(
            &format!("{name} out[{k}]"),
            &values(&got.to_tensor()),
            &values(&bits_tensor(want)),
        );
    }
    let expected_grads = case["input_grads"].as_array().expect("input_grads");
    for (k, (got, want)) in dxs.iter().zip(expected_grads).enumerate() {
        assert_parity(&format!("{name} dx[{k}]"), got, &values(&bits_tensor(want)));
    }
}

#[test]
fn matches_pytorch_reference_for_merge_graphs() {
    let doc = fixture();
    assert!(
        doc["torch_version"]
            .as_str()
            .expect("v")
            .starts_with("2.14.0")
    );
    let cases = doc["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 6);
    let tape = crate::tape();
    for case in cases {
        check_case(case, &tape);
    }
}

// ------------------------------------------------- 実機 parity（#[ignore]）

/// 連鎖結合ケースを CPU tape と実機 tape で同じ重みのまま forward／backward して比較する。
fn device_matches_cpu(device: crate::Device) {
    let doc = fixture();
    let case = doc["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"] == "chained_merges_two_inputs")
        .expect("連鎖結合ケース");
    let model = model_from_case(case);
    let inputs: Vec<Tensor<f32>> = case["input_values"]
        .as_array()
        .expect("input_values")
        .iter()
        .map(bits_tensor)
        .collect();
    let cpu_tape = crate::tape();
    let (_, cpu_outs, cpu_dxs) = run_with_loss(&model, case, &cpu_tape, &inputs);
    let cpu_outs: Vec<Vec<f32>> = cpu_outs.iter().map(|o| values(&o.to_tensor())).collect();
    let device_tape = crate::tape_for(device).expect("実機 tape");
    let (_, dev_outs, dev_dxs) = run_with_loss(&model, case, &device_tape, &inputs);
    for (k, (a, b)) in dev_outs.iter().zip(&cpu_outs).enumerate() {
        assert_parity(&format!("{device:?} out[{k}]"), &values(&a.to_tensor()), b);
    }
    for (k, (a, b)) in dev_dxs.iter().zip(&cpu_dxs).enumerate() {
        assert_parity(&format!("{device:?} dx[{k}]"), a, b);
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。docs/perf/logs/merge-ops-2666/README.md"]
fn cuda_merge_graph_matches_cpu_reference() {
    device_matches_cpu(crate::Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機依存。docs/perf/logs/merge-ops-2666/README.md"]
fn metal_merge_graph_matches_cpu_reference() {
    device_matches_cpu(crate::Device::Metal);
}
