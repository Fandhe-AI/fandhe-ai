//! ONNX の学習（autograd）公開面の受入テスト（イシュー #3843・親 #3759。
//! 設計記録は `docs/onnx-autograd-decision.md` §8.4・§8.13、承認は #3770）。
//!
//! 利用者が facade だけで書く形で、次の 4 点を CI テストとして固定する。
//! facade と内部 `BoundGraph` の bit 一致・エラー写像は #3842 の
//! `interop_onnx_internal_parity.rs` が担い、本ファイルは重ねない。
//!
//! - 学習が進むこと（固定データ・決定的な重み・SGD。重みの更新は initializer と
//!   同名の feed で上書きする。書き戻し API はない）
//! - 未対応 op の fail-closed 拒否（16 op 側・CNN 6 op 側）
//! - `Cast` は F32 から F32 以外へ変換するときだけ勾配を切る
//! - `trainable` の既定（全 F32 initializer）と絞り込み・typo の拒否
//!
//! import 方針: 実行経路はすべて `fandhe_ai` だけを使う。例外は 1 つで、`Cast`
//! ノードを持つ ONNX bytes を作るために `fandhe_ai_onnx_interop::onnx::proto`
//! （`ModelProto` 等と `encode_model`）だけを import する（`Cast` を持つ fixture
//! はコミットされていないため）。内部の autograd・interp・autodiff・backend は
//! 参照しない。この制限は `api_surface.rs` の
//! `onnx_autograd_acceptance_coverage_map_matches` が機械的に検査する。
//!
//! 数値比較は bit 一致か型・`Option` の判定だけで、tolerance は持ち込まない。
//! 実機（GPU）の parity は `train_gpu_parity.rs` の `onnx` 族（#3864・#3865）が担当する。

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::path::PathBuf;

use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::onnx::{
    OnnxAutogradOptions, OnnxError, OnnxModel, OnnxTapeValue, OnnxValue,
};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{Tensor, Var};

use fandhe_ai_onnx_interop::onnx::proto::{
    self, AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto,
    attribute_type, data_type,
};

type TestResult = Result<(), Box<dyn Error>>;

fn onnx_interop_fixture(rel: &str) -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../onnx-interop/tests/fixtures"
    ))
    .join(rel)
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("連続化済み")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.contiguous().as_slice().expect("連続化済み")[0]
}

// --- 1. 学習が進む ---------------------------------------------------------

struct TrainLog {
    losses: Vec<f32>,
    /// 各ステップの出力（bit 列）。feed による重み上書きが効いている証拠に使う。
    outputs: Vec<Vec<u32>>,
    /// bind 時の葉（`g.param`）の勾配が全ステップで `None` だったか。
    bind_leaf_grad_always_none: bool,
}

/// 利用者の書き方: 毎ステップ新しい tape へ bind し、重みは initializer と同名の
/// feed（`tape.var(&w)`）で渡す。更新は feed 側 `Var` の勾配で SGD を回す。
fn train_sgd(
    model: &OnnxModel,
    x: &Tensor<f32>,
    target: &Tensor<f32>,
    lr: f32,
    steps: usize,
) -> Result<TrainLog, Box<dyn Error>> {
    let mut sgd = Sgd::new(SgdConfig::new(lr))?;
    let mut weights: Vec<(String, Tensor<f32>)> = Vec::new();
    let mut log = TrainLog {
        losses: Vec::new(),
        outputs: Vec::new(),
        bind_leaf_grad_always_none: true,
    };
    for _ in 0..steps {
        let tape = fandhe_ai::tape();
        let g = model.bind_autograd(&tape, &OnnxAutogradOptions::default())?;
        if weights.is_empty() {
            weights = g
                .params()
                .iter()
                .map(|(n, v)| (n.clone(), v.value().clone()))
                .collect();
        }
        let xv = tape.var(x);
        let wvars: Vec<(String, Var)> = weights
            .iter()
            .map(|(n, t)| (n.clone(), tape.var(t)))
            .collect();
        let mut feeds = HashMap::new();
        feeds.insert("input".to_string(), OnnxTapeValue::Var(xv));
        for (n, v) in &wvars {
            feeds.insert(n.clone(), OnnxTapeValue::Var(*v));
        }
        let outs = g.run(feeds)?;
        assert_eq!(outs.len(), 1, "出力は 1 個のはず");
        let y = match outs.into_values().next().expect("出力 1 個") {
            OnnxTapeValue::Var(y) => y,
            other => panic!("Var を期待したが {other:?}"),
        };
        let loss = y.mse_loss(&tape.var(target))?;
        let grads = tape.backward(&loss)?;

        log.losses.push(scalar(&loss.value()));
        log.outputs.push(bits(&y.value()));
        for (n, _) in &weights {
            let leaf = g.param(n).expect("bind 時の葉が存在する");
            if grads.get(&leaf)?.is_some() {
                log.bind_leaf_grad_always_none = false;
            }
        }

        let gs: Vec<Tensor<f32>> = wvars
            .iter()
            .map(|(n, v)| {
                grads
                    .get(v)
                    .expect("get 成功")
                    .unwrap_or_else(|| panic!("feed 側の勾配が存在する: {n}"))
                    .clone()
            })
            .collect();
        let params: Vec<&Tensor<f32>> = weights.iter().map(|(_, t)| t).collect();
        let grad_refs: Vec<&Tensor<f32>> = gs.iter().collect();
        let updated = sgd.step(&params, &grad_refs)?;
        weights = weights
            .iter()
            .map(|(n, _)| n.clone())
            .zip(updated)
            .collect();
    }
    Ok(log)
}

const MONOTONE_STEPS: usize = 3;

fn assert_training_progressed(label: &str, log: &TrainLog) {
    let first = log.losses[0];
    let last = *log.losses.last().expect("非空");
    assert!(
        log.losses.iter().all(|l| l.is_finite()),
        "{label}: 損失が有限でない: {:?}",
        log.losses
    );
    assert!(
        last < first,
        "{label}: 最終損失が初回より小さくない: first={first} last={last}"
    );
    // 小さい lr の序盤は単調に下がる（終盤は固定データで頭打ちになりうるため
    // 全区間の単調性は要求しない。係数・比率の閾値も置かない）。
    assert!(
        log.losses[..MONOTONE_STEPS].windows(2).all(|w| w[1] < w[0]),
        "{label}: 序盤の損失が単調減少でない: {:?}",
        log.losses
    );
    assert_ne!(
        log.outputs[0], log.outputs[1],
        "{label}: feed による重み上書きが効いていない（出力が変わらない）"
    );
    assert!(
        log.bind_leaf_grad_always_none,
        "{label}: 同名 feed で上書きしたのに bind 時の葉へ勾配が付いた"
    );
}

#[test]
fn onnx_autograd_sgd_loop_decreases_loss() -> TestResult {
    let x = Tensor::<f32>::new(vec![0.5, -1.0, -0.3, 0.8, 1.2, 0.4, -0.9, -0.6], &[4, 2])?;
    let target = Tensor::<f32>::new(vec![0.3, 0.6, 0.5, 0.7], &[4, 1])?;

    // (a) PyTorch が export した model.onnx を fine-tuning する想定。
    let pytorch = OnnxModel::from_path(onnx_interop_fixture("model.onnx"))?;
    let log_a = train_sgd(&pytorch, &x, &target, 0.1, 12)?;
    assert_training_progressed("model.onnx", &log_a);

    // (b) compat::Sequential から作ったモデル。
    let seq = Sequential::new()
        .add_linear(2, 4, 7)?
        .add_relu()
        .add_linear(4, 1, 8)?;
    let from_seq = OnnxModel::from_sequential(&seq)?;
    let target_b = Tensor::<f32>::new(vec![0.2, 0.8, 0.5, -0.3], &[4, 1])?;
    let log_b = train_sgd(&from_seq, &x, &target_b, 0.05, 12)?;
    assert_training_progressed("from_sequential", &log_b);

    // 決定性: 同じ設定で回し直すと損失列が bit 一致する。
    let again = train_sgd(&pytorch, &x, &target, 0.1, 12)?;
    let a: Vec<u32> = log_a.losses.iter().map(|l| l.to_bits()).collect();
    let b: Vec<u32> = again.losses.iter().map(|l| l.to_bits()).collect();
    assert_eq!(a, b, "同条件の再実行で損失列が bit 不一致");
    Ok(())
}

// --- 2. 未対応 op の fail-closed ------------------------------------------

/// `(fixture 相対パス, 入力 shape, 最初に止まる op)`。実際の PyTorch export を使う。
const UNSUPPORTED_CASES: &[(&str, &[usize], &str)] = &[
    (
        "pytorch-onnx-cnn-ops/conv2d_basic_ts.onnx",
        &[1, 3, 8, 8],
        "Conv",
    ),
    (
        "pytorch-onnx-cnn-ops/maxpool2d_basic_ts.onnx",
        &[1, 3, 8, 8],
        "MaxPool",
    ),
    (
        "pytorch-onnx-cnn-ops/avgpool2d_include_pad_ts.onnx",
        &[1, 3, 8, 8],
        "AveragePool",
    ),
    (
        "pytorch-onnx-cnn-ops/bn2d_eval_ts.onnx",
        &[2, 3, 5, 5],
        "BatchNormalization",
    ),
    (
        "pytorch-onnx-cnn-ops/gap2d_ts.onnx",
        &[1, 3, 7, 9],
        "GlobalAveragePool",
    ),
    (
        "pytorch-onnx-cnn-ops/flatten_default_ts.onnx",
        &[2, 3, 4, 5],
        "Flatten",
    ),
    (
        "pytorch-onnx-cnn-ops/flatten_default_dynamo.onnx",
        &[2, 3, 4, 5],
        "Reshape",
    ),
    (
        "pytorch-onnx-cnn-ops/gap2d_dynamo.onnx",
        &[1, 3, 7, 9],
        "ReduceMean",
    ),
    ("slice_repro.onnx", &[5, 6], "Gather"),
];

#[test]
fn unsupported_op_in_autograd_is_rejected_fail_closed() -> TestResult {
    for (rel, shape, expected_op) in UNSUPPORTED_CASES {
        let model = OnnxModel::from_path(onnx_interop_fixture(rel))?;
        let n: usize = shape.iter().product();
        let x = Tensor::<f32>::new((0..n).map(|i| i as f32 * 0.01 - 0.5).collect(), shape)?;

        let tape = fandhe_ai::tape();
        // 拒否は bind 時ではなく実行時（未対応 op のノードに到達した時点）。
        let g = model.bind_autograd(&tape, &OnnxAutogradOptions::default())?;
        let err = g
            .forward(&tape.var(&x))
            .expect_err(&format!("{rel}: forward が成功してはならない"));
        match &err {
            OnnxError::UnsupportedInAutograd { op_type, node } => {
                assert_eq!(op_type, expected_op, "{rel}: 止まる op が違う");
                assert!(!node.is_empty(), "{rel}: node 名が空");
            }
            other => panic!("{rel}: UnsupportedInAutograd を期待したが {other:?}"),
        }
        assert!(
            err.to_string().contains(expected_op),
            "{rel}: Display に op 名がない: {err}"
        );

        // run 経路でも同じ（部分的な Ok を返さない）。入力名は MissingFeed から得る。
        let input_name = match g.run(HashMap::new()).expect_err("feed 無しは失敗") {
            OnnxError::MissingFeed { input } => input,
            other => panic!("{rel}: MissingFeed を期待したが {other:?}"),
        };
        let mut feeds = HashMap::new();
        feeds.insert(input_name.clone(), OnnxTapeValue::Var(tape.var(&x)));
        let run_err = g.run(feeds).expect_err("run も失敗する");
        assert!(
            matches!(&run_err, OnnxError::UnsupportedInAutograd { op_type, .. } if op_type == expected_op),
            "{rel}: run の写像が違う: {run_err:?}"
        );

        // 同じ bytes の推論は成功する（既定経路は変わっていない）。
        let mut feeds = HashMap::new();
        feeds.insert(input_name, OnnxValue::F32(x));
        model
            .run(feeds)
            .unwrap_or_else(|e| panic!("{rel}: 推論 run が失敗: {e}"));
    }
    Ok(())
}

// --- 3. Cast の勾配切断 ----------------------------------------------------

fn node(
    op: &str,
    name: &str,
    inputs: &[&str],
    output: &str,
    attrs: Vec<AttributeProto>,
) -> NodeProto {
    NodeProto {
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: vec![output.to_string()],
        name: name.to_string(),
        op_type: op.to_string(),
        attribute: attrs,
        domain: String::new(),
    }
}

fn cast_node(name: &str, input: &str, output: &str, to: i64) -> NodeProto {
    node(
        "Cast",
        name,
        &[input],
        output,
        vec![AttributeProto {
            name: "to".to_string(),
            f: 0.0,
            i: to,
            s: vec![],
            t: None,
            floats: vec![],
            ints: vec![],
            r#type: attribute_type::INT,
        }],
    )
}

fn model_of(nodes: Vec<NodeProto>, inits: Vec<TensorProto>, output: &str) -> OnnxModel {
    let m = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "facade-acceptance".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: nodes,
            name: "g".to_string(),
            initializer: inits,
            input: vec![ValueInfoProto {
                name: "x".to_string(),
            }],
            output: vec![ValueInfoProto {
                name: output.to_string(),
            }],
        }),
    };
    OnnxModel::from_bytes(&proto::encode_model(&m)).expect("構築は妥当")
}

fn f32_init(name: &str, values: &[f32]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: data_type::FLOAT,
        dims: vec![values.len() as i64],
        float_data: values.to_vec(),
        int64_data: vec![],
        raw_data: vec![],
        external_data: Vec::new(),
        data_location: 0,
    }
}

const W: [f32; 4] = [0.5, -1.5, 2.0, 3.25];
const X: [f32; 4] = [1.0, 0.0, 1.0, 1.0];
const ONNX_FLOAT: i64 = 1;
const ONNX_INT64: i64 = 7;
const ONNX_BOOL: i64 = 9;
const ONNX_FLOAT16: i64 = 10;

/// `x -> [Cast(to=mid) -> Cast(to=FLOAT)]? -> Mul(., w) -> y`。
fn cast_chain_model(mid: Option<i64>) -> OnnxModel {
    let mut nodes = Vec::new();
    let mul_in = match mid {
        Some(t) => {
            nodes.push(cast_node("c1", "x", "x_mid", t));
            nodes.push(cast_node("c2", "x_mid", "x_back", ONNX_FLOAT));
            "x_back"
        }
        None => {
            nodes.push(cast_node("c1", "x", "x_back", ONNX_FLOAT));
            "x_back"
        }
    };
    nodes.push(node("Mul", "m", &[mul_in, "w"], "y", vec![]));
    model_of(nodes, vec![f32_init("w", &W)], "y")
}

#[test]
fn cast_severs_gradient_only_for_non_f32_target() -> TestResult {
    let x = Tensor::<f32>::new(X.to_vec(), &[4])?;
    let w_bits: Vec<u32> = W.iter().map(|v| v.to_bits()).collect();

    for (label, mid, severs) in [
        ("F32->INT64->F32", Some(ONNX_INT64), true),
        ("F32->BOOL->F32", Some(ONNX_BOOL), true),
        ("F32->FLOAT16->F32", Some(ONNX_FLOAT16), true),
        ("F32->F32（透過）", None, false),
    ] {
        let model = cast_chain_model(mid);
        let tape = fandhe_ai::tape();
        let g = model.bind_autograd(&tape, &OnnxAutogradOptions::default())?;
        let xv = tape.var(&x);
        let y = g.forward(&xv)?;
        let loss = y.sum(None)?;
        let grads = tape.backward(&loss)?;

        // forward の値は全系で同じ（x は往復で丸めが出ない値）。
        let expected: Vec<u32> = X.iter().zip(W).map(|(a, b)| (a * b).to_bits()).collect();
        assert_eq!(bits(&y.value()), expected, "{label}: forward の値");

        let wv = g.param("w").expect("w は学習対象");
        assert!(grads.get(&wv)?.is_some(), "{label}: w の勾配がない");
        if severs {
            assert!(
                grads.get(&xv)?.is_none(),
                "{label}: F32 以外への Cast で勾配が切れていない"
            );
        } else {
            let gx = grads.get(&xv)?.expect("F32->F32 は透過で x に勾配が届く");
            assert_eq!(bits(gx), w_bits, "{label}: dΣ(x*w)/dx = w");
        }

        // 同じモデルの推論 run と forward が bit 一致する。
        let mut feeds = HashMap::new();
        feeds.insert("x".to_string(), OnnxValue::F32(x.clone()));
        match &model.run(feeds)?["y"] {
            OnnxValue::F32(t) => assert_eq!(bits(t), bits(&y.value()), "{label}: run と不一致"),
            other => panic!("{label}: F32 を期待したが {other:?}"),
        }
    }

    // 出力が Cast(to=INT64) のままのグラフ: run は Const(I64) を返し、forward は拒否する。
    let model = model_of(vec![cast_node("c1", "x", "y", ONNX_INT64)], vec![], "y");
    let tape = fandhe_ai::tape();
    let g = model.bind_autograd(&tape, &OnnxAutogradOptions::default())?;
    let mut feeds = HashMap::new();
    feeds.insert("x".to_string(), OnnxTapeValue::Var(tape.var(&x)));
    let outs = g.run(feeds)?;
    assert!(
        matches!(&outs["y"], OnnxTapeValue::Const(OnnxValue::I64(_))),
        "Const(I64) を期待したが {:?}",
        outs["y"]
    );
    let err = g.forward(&tape.var(&x)).expect_err("forward は拒否");
    assert!(matches!(err, OnnxError::FeedTypeMismatch { .. }), "{err:?}");
    Ok(())
}

// --- 4. trainable の既定・絞り込み・typo ------------------------------------

#[test]
fn trainable_default_and_filter_semantics() -> TestResult {
    let model = OnnxModel::from_path(onnx_interop_fixture("model.onnx"))?;
    let x = Tensor::<f32>::new(vec![0.3, -0.7], &[1, 2])?;
    let all = [
        "fc1.bias",
        "fc1.weight",
        "fc2.bias",
        "fc2.weight",
        "fc3.bias",
        "fc3.weight",
    ];

    // 既定: 全 F32 initializer（名前順）。
    let tape = fandhe_ai::tape();
    let g = model.bind_autograd(&tape, &OnnxAutogradOptions::default())?;
    let names: Vec<String> = g.params().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, all);

    // 絞り込み: 凍結層を通しても入力へ勾配が届く（凍結は勾配の切断ではない）。
    let tape = fandhe_ai::tape();
    let opts = OnnxAutogradOptions::with_trainable(HashSet::from([
        "fc3.weight".to_string(),
        "fc3.bias".to_string(),
    ]));
    let g = model.bind_autograd(&tape, &opts)?;
    let names: Vec<String> = g.params().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["fc3.bias", "fc3.weight"]);
    assert!(g.param("fc1.weight").is_none(), "凍結層は param に出ない");
    let xv = tape.var(&x);
    let y = g.forward(&xv)?;
    let loss = y.sum(None)?;
    let grads = tape.backward(&loss)?;
    for (n, v) in g.params() {
        assert!(grads.get(&v)?.is_some(), "{n}: 学習対象に勾配がない");
    }
    assert!(
        grads.get(&xv)?.is_some(),
        "凍結層を通しても入力に勾配が届く"
    );

    // typo と非 F32 initializer 名は bind 時に拒否される。
    let slice = OnnxModel::from_path(onnx_interop_fixture("slice_repro.onnx"))?;
    for (m, bad) in [(&model, "fc3.weigth"), (&slice, "const_axes")] {
        let tape = fandhe_ai::tape();
        let err = m
            .bind_autograd(
                &tape,
                &OnnxAutogradOptions::with_trainable(HashSet::from([bad.to_string()])),
            )
            .expect_err("bind は拒否");
        assert!(
            matches!(&err, OnnxError::UnknownTrainable { name } if name == bad),
            "{bad}: {err:?}"
        );
    }
    Ok(())
}

// --- 5. 別 tape の Var は Autodiff variant へ写像される ---------------------

#[test]
fn onnx_autograd_cross_tape_feed_maps_to_autodiff_variant() -> TestResult {
    let model = OnnxModel::from_path(onnx_interop_fixture("model.onnx"))?;
    let x = Tensor::<f32>::new(vec![0.3, -0.7], &[1, 2])?;
    let tape_a = fandhe_ai::tape();
    let tape_b = fandhe_ai::tape();
    let g = model.bind_autograd(&tape_a, &OnnxAutogradOptions::default())?;
    let err = g
        .forward(&tape_b.var(&x))
        .expect_err("別 tape の Var は拒否");
    assert!(matches!(&err, OnnxError::Autodiff(_)), "{err:?}");
    assert!(
        std::error::Error::source(&err).is_some(),
        "Autodiff は source を持つ"
    );
    Ok(())
}
