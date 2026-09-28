//! ONNX `Conv`（1D 拡張）・`MaxPool`／`AveragePool`（イシュー #2199・親
//! #2185）の統合テスト。
//!
//! `ops::conv`／`ops::max_pool`／`ops::average_pool`（ops 直接呼び出し。
//! decode 層を経由しない経路）を、`fandhe_ai_autodiff::nn::{Conv1d,
//! Conv2d, MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d}::forward_host`
//! （`CpuBackendOps`。tape 不要経路）と突合する。
//!
//! ## 「PyTorch export との bit 同一」の再定義（実装計画 §3.3。受け入れ
//! 文言からの逸脱を本ファイルに明記する）
//!
//! 手元に `torch` が無いため、`torch.onnx.export` が実際に出力する
//! ONNX グラフ由来の fixture は使わない。代わりに `ops::*` を直接呼び、
//! 同じハイパーパラメータで構築した `nn::*` 層の `forward_host` と
//! 突合する。bit 一致を主張する組合せ:
//!
//! 1. MaxPool1d／2d・AvgPool1d／2d（`ceil_mode=0`・対称 `pads <=
//!    kernel/2`・avg の `dilation=1`）と `nn::{MaxPool1d,MaxPool2d,
//!    AvgPool1d,AvgPool2d}::forward_host`
//! 2. Conv1d と、同じ重みを 2D へ持ち上げた Conv2d（`ops::conv` 内部の
//!    1D 持ち上げと同じ主張。`ops/conv.rs` の unit テストと同型だが、
//!    ここでは決定的乱数入力で再確認する）
//!
//! REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）
//! に留める組合せ: `Conv` と `nn::Conv1d`／`Conv2d::forward_host`
//! （直接ループと im2col+GEMM で結合順序が異なるため。#2076 の既存契約）。
//!
//! ceil_mode=1・非対称 pads は `nn::*` 側に対応が無いため、手計算した
//! 期待値との bit 一致で確認する。

// 決定的シード PRNG は `bench_harness::rng::Xorshift64Star` ではなく
// `support::Xorshift64Star`（同一アルゴリズムの複製）を使う。理由は
// `tests/support/mod.rs` 冒頭ドキュメント参照（`bench-harness` の
// `backend-cuda` 依存が非 unix ターゲットで `compile_error!` になるため、
// 本クレートの CPU のみで完結するテストを Windows でも動かせるよう依存を
// 切り離した。codex-review 指摘 `PRRT_kwDOTuUCJc6mzcKf`・PR #2351）。
mod support;

use std::collections::HashMap;

use fandhe_ai_autodiff::nn::{AvgPool1d, AvgPool2d, Conv1d, Conv2d, MaxPool1d, MaxPool2d, Module};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_onnx_interop::onnx::graph::build_graph;
use fandhe_ai_onnx_interop::onnx::interp::{Value, run};
use fandhe_ai_onnx_interop::onnx::proto::{
    self, AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto,
    attribute_type, data_type,
};
use fandhe_ai_onnx_interop::ops::{ConvAttrs, PoolAttrs, average_pool, conv, max_pool};
use fandhe_ai_tensor_core::Tensor;
use support::Xorshift64Star;

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().unwrap().to_vec()
}

fn rand_tensor(shape: &[usize], seed: u64) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let mut rng = Xorshift64Star::new(seed);
    let data: Vec<f32> = (0..numel).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    Tensor::new(data, shape).unwrap()
}

// --- MaxPool2d／AvgPool2d: `ops::*` と `nn::*::forward_host` の bit 一致 ---

#[test]
fn max_pool2d_matches_nn_forward_host_bit_exact() {
    let x = rand_tensor(&[2, 3, 7, 7], 1);
    let attrs = PoolAttrs {
        kernel_shape: vec![3, 3],
        strides: vec![2, 2],
        pads: vec![1, 1, 1, 1],
        ..PoolAttrs::default()
    };
    let via_ops = max_pool(&x, &attrs).unwrap();

    let nn_pool = MaxPool2d::new([3, 3], Some([2, 2]), [1, 1], [1, 1]).unwrap();
    let ops = CpuBackendOps::new();
    let via_nn = nn_pool.forward_host(&ops, &x).unwrap();

    assert_eq!(dense(&via_ops), dense(&via_nn));
}

#[test]
fn max_pool1d_matches_nn_forward_host_bit_exact() {
    let x = rand_tensor(&[2, 3, 11], 2);
    let attrs = PoolAttrs {
        kernel_shape: vec![3],
        strides: vec![2],
        pads: vec![1, 1],
        ..PoolAttrs::default()
    };
    let via_ops = max_pool(&x, &attrs).unwrap();

    let nn_pool = MaxPool1d::new(3, Some(2), 1, 1).unwrap();
    let ops = CpuBackendOps::new();
    let via_nn = nn_pool.forward_host(&ops, &x).unwrap();

    assert_eq!(dense(&via_ops), dense(&via_nn));
}

#[test]
fn avg_pool2d_count_include_pad_true_matches_nn_forward_host_bit_exact() {
    let x = rand_tensor(&[1, 2, 6, 6], 3);
    let attrs = PoolAttrs {
        kernel_shape: vec![2, 2],
        strides: vec![2, 2],
        pads: vec![0, 0, 0, 0],
        count_include_pad: 1,
        ..PoolAttrs::default()
    };
    let via_ops = average_pool(&x, &attrs).unwrap();

    let nn_pool = AvgPool2d::new([2, 2], Some([2, 2]), [0, 0], true).unwrap();
    let ops = CpuBackendOps::new();
    let via_nn = nn_pool.forward_host(&ops, &x).unwrap();

    assert_eq!(dense(&via_ops), dense(&via_nn));
}

#[test]
fn avg_pool1d_count_include_pad_false_matches_nn_forward_host_bit_exact() {
    let x = rand_tensor(&[1, 2, 9], 4);
    let attrs = PoolAttrs {
        kernel_shape: vec![3],
        strides: vec![1],
        pads: vec![1, 1],
        count_include_pad: 0,
        ..PoolAttrs::default()
    };
    let via_ops = average_pool(&x, &attrs).unwrap();

    let nn_pool = AvgPool1d::new(3, Some(1), 1, false).unwrap();
    let ops = CpuBackendOps::new();
    let via_nn = nn_pool.forward_host(&ops, &x).unwrap();

    assert_eq!(dense(&via_ops), dense(&via_nn));
}

// --- Conv1d／Conv2d: REQ-2 統一複合判定（im2col+GEMM vs 直接ループ） ---

#[test]
fn conv2d_matches_nn_forward_host_within_req2() {
    let layer = Conv2d::new(3, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, true, 42).unwrap();
    let x = rand_tensor(&[2, 3, 8, 8], 5);

    let attrs = ConvAttrs {
        kernel_shape: vec![3, 3],
        strides: vec![1, 1],
        pads: vec![1, 1, 1, 1],
        dilations: vec![1, 1],
        group: 1,
        auto_pad: String::new(),
    };
    let via_ops = conv(&x, layer.weight(), layer.bias(), &attrs).unwrap();

    let ops = CpuBackendOps::new();
    let via_nn = layer.forward_host(&ops, &x).unwrap();

    assert_parity(
        "Conv2d: ops::conv vs nn::Conv2d::forward_host",
        &dense(&via_ops),
        &dense(&via_nn),
    );
}

#[test]
fn conv1d_matches_nn_forward_host_within_req2() {
    let layer = Conv1d::new(2, 3, 3, 2, 1, 1, 1, true, 43).unwrap();
    let x = rand_tensor(&[2, 2, 9], 6);

    let attrs = ConvAttrs {
        kernel_shape: vec![3],
        strides: vec![2],
        pads: vec![1, 1],
        dilations: vec![1],
        group: 1,
        auto_pad: String::new(),
    };
    let via_ops = conv(&x, layer.weight(), layer.bias(), &attrs).unwrap();

    let ops = CpuBackendOps::new();
    let via_nn = layer.forward_host(&ops, &x).unwrap();

    assert_parity(
        "Conv1d: ops::conv vs nn::Conv1d::forward_host",
        &dense(&via_ops),
        &dense(&via_nn),
    );
}

// --- ceil_mode=1・非対称 pads: 手計算値との bit 一致 ---

#[test]
fn max_pool_ceil_mode_matches_hand_computed_value() {
    // X: [1,1,1,5] = [1,3,2,5,4]. kernel=2, stride=2, pad=0, ceil_mode=1.
    // floor: (5-2)/2+1=2. ceil: ceil(3/2)+1=3、調整規則 (3-1)*2=4 >= 5+0 は
    // 偽 -> out=3 のまま。
    // windows: [1,3]->3, [2,5]->5, [4]->4 (最後の窓は入力内で 1 要素のみ)。
    let x = Tensor::<f32>::new(vec![1.0, 3.0, 2.0, 5.0, 4.0], &[1, 1, 1, 5]).unwrap();
    let attrs = PoolAttrs {
        kernel_shape: vec![1, 2],
        strides: vec![1, 2],
        ceil_mode: 1,
        ..PoolAttrs::default()
    };
    let y = max_pool(&x, &attrs).unwrap();
    assert_eq!(y.shape(), &[1, 1, 1, 3]);
    let out = dense(&y);
    assert_eq!(out, vec![3.0, 5.0, 4.0]);
}

#[test]
fn average_pool_asymmetric_pads_matches_hand_computed_value() {
    // X: [1,1,1,3] = [1,2,3]. kernel=2, stride=1, pads=[0,1] (begin=0,end=1).
    // count_include_pad=1 -> divisor は padded 座標でクリップ。
    // padded_w = 3+0+1 = 4. windows starts: 0,1,2.
    // w=0: [1,2] divisor=min(0+2,4)-0=2 -> 1.5
    // w=1: [2,3] divisor=min(1+2,4)-1=2 -> 2.5
    // w=2: [3,pad] divisor=min(2+2,4)-2=2 -> (3+0)/2=1.5
    let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[1, 1, 1, 3]).unwrap();
    let attrs = PoolAttrs {
        kernel_shape: vec![1, 2],
        strides: vec![1, 1],
        pads: vec![0, 0, 0, 1],
        count_include_pad: 1,
        ..PoolAttrs::default()
    };
    let y = average_pool(&x, &attrs).unwrap();
    assert_eq!(y.shape(), &[1, 1, 1, 3]);
    let out = dense(&y);
    assert_eq!(out, vec![1.5, 2.5, 1.5]);
}

// --- 複合 CNN: Conv -> MaxPool -> Conv -> AveragePool（ops 手動連結 vs nn 連結） ---

#[test]
fn composite_cnn_chain_matches_nn_within_req2() {
    // Conv(3->4,k3,s1,p1) -> MaxPool(k2,s2) -> Conv(4->2,k3,s1,p1) ->
    // AveragePool(k2,s2)。`ops::*` を手動連結した経路と、`nn::*` を
    // `Module::forward_host` で連結した経路を突合する。Conv を含むため
    // REQ-2 統一複合判定（`assert_parity`）を用いる。
    let conv1 = Conv2d::new(3, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, true, 100).unwrap();
    let conv2 = Conv2d::new(4, 2, [3, 3], [1, 1], [1, 1], [1, 1], 1, true, 200).unwrap();
    let x = rand_tensor(&[1, 3, 8, 8], 7);

    // ops 手動連結。
    let conv_attrs = |kh: i64, kw: i64| ConvAttrs {
        kernel_shape: vec![kh, kw],
        strides: vec![1, 1],
        pads: vec![1, 1, 1, 1],
        dilations: vec![1, 1],
        group: 1,
        auto_pad: String::new(),
    };
    let y1 = conv(&x, conv1.weight(), conv1.bias(), &conv_attrs(3, 3)).unwrap();
    let pool_attrs = PoolAttrs {
        kernel_shape: vec![2, 2],
        strides: vec![2, 2],
        ..PoolAttrs::default()
    };
    let y2 = max_pool(&y1, &pool_attrs).unwrap();
    let y3 = conv(&y2, conv2.weight(), conv2.bias(), &conv_attrs(3, 3)).unwrap();
    let avg_attrs = PoolAttrs {
        kernel_shape: vec![2, 2],
        strides: vec![2, 2],
        count_include_pad: 1,
        ..PoolAttrs::default()
    };
    let y4 = average_pool(&y3, &avg_attrs).unwrap();

    // nn 連結（forward_host）。
    let ops = CpuBackendOps::new();
    let n1 = conv1.forward_host(&ops, &x).unwrap();
    let nn_max = MaxPool2d::new([2, 2], Some([2, 2]), [0, 0], [1, 1]).unwrap();
    let n2 = nn_max.forward_host(&ops, &n1).unwrap();
    let n3 = conv2.forward_host(&ops, &n2).unwrap();
    let nn_avg = AvgPool2d::new([2, 2], Some([2, 2]), [0, 0], true).unwrap();
    let n4 = nn_avg.forward_host(&ops, &n3).unwrap();

    assert_eq!(y4.shape(), n4.shape());
    assert_parity(
        "複合 CNN（Conv->MaxPool->Conv->AveragePool）: ops 連結 vs nn 連結",
        &dense(&y4),
        &dense(&n4),
    );
}

// --- decode_model -> build_graph -> interp::run の連結経路（codex-review 指摘。
//     PR #2314 レビュー。`composite_cnn_chain_matches_nn_within_req2` は
//     `ops::*` を手動連結するのみで decode 層を経由しないため、Conv の出力を
//     Pool へ渡す結線が「シリアライズ済みグラフ（protobuf バイト列）→ decode
//     → build_graph → run」の経路上で機能することを別途固定する ---

/// [`Tensor<f32>`] を dense（contiguous）データのまま ONNX `TensorProto`
/// initializer へ変換する（`Conv2d::weight`／`bias` の shape・レイアウトは
/// ONNX `Conv` の `W`／`B` 入力とそのまま一致するため転置は不要。
/// `crates/facade/tests/interop_onnx_conv_pool_import.rs` の initializer
/// 構築と同型）。
fn tensor_to_initializer(name: &str, t: &Tensor<f32>) -> TensorProto {
    TensorProto {
        dims: t.shape().iter().map(|&d| d as i64).collect(),
        data_type: data_type::FLOAT,
        float_data: dense(t),
        name: name.to_string(),
        ..Default::default()
    }
}

fn attr_ints(name: &str, ints: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ints,
        r#type: attribute_type::INTS,
        ..Default::default()
    }
}

fn attr_int(name: &str, i: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i,
        r#type: attribute_type::INT,
        ..Default::default()
    }
}

fn value_info(name: &str) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
    }
}

/// `Conv(x, w, b) -> <pool_op_type>(conv_out)` の 2 ノード `ModelProto` を
/// 組み立てる（`w`／`b` は initializer として埋め込み、`x` のみ feed で
/// 与える）。`pool_attrs` は `MaxPool`／`AveragePool` 共通の属性列。
#[allow(clippy::too_many_arguments)]
fn build_conv_then_pool_model(
    conv_weight: &Tensor<f32>,
    conv_bias: &Tensor<f32>,
    conv_attrs: &ConvAttrs,
    pool_op_type: &str,
    pool_attrs_attribute: Vec<AttributeProto>,
) -> ModelProto {
    let conv_node = NodeProto {
        input: vec!["x".to_string(), "w".to_string(), "b".to_string()],
        output: vec!["conv_out".to_string()],
        name: "n_conv".to_string(),
        op_type: "Conv".to_string(),
        attribute: vec![
            attr_ints("kernel_shape", conv_attrs.kernel_shape.clone()),
            attr_ints("strides", conv_attrs.strides.clone()),
            attr_ints("pads", conv_attrs.pads.clone()),
            attr_ints("dilations", conv_attrs.dilations.clone()),
            attr_int("group", conv_attrs.group),
        ],
        domain: String::new(),
    };
    let pool_node = NodeProto {
        input: vec!["conv_out".to_string()],
        output: vec!["y".to_string()],
        name: "n_pool".to_string(),
        op_type: pool_op_type.to_string(),
        attribute: pool_attrs_attribute,
        domain: String::new(),
    };
    ModelProto {
        graph: Some(GraphProto {
            node: vec![conv_node, pool_node],
            name: "conv_then_pool_test".to_string(),
            initializer: vec![
                tensor_to_initializer("w", conv_weight),
                tensor_to_initializer("b", conv_bias),
            ],
            input: vec![value_info("x")],
            output: vec![value_info("y")],
            value_info: vec![],
            sparse_initializer: vec![],
        }),
        ..Default::default()
    }
}

#[test]
fn conv_maxpool_serialized_graph_matches_nn_within_req2() {
    // Conv(2->4,k3,s1,p1) -> MaxPool(k2,s2)。ModelProto を組み立てて
    // `proto::encode_model` でバイト列化し、`proto::decode_model` から
    // `build_graph`・`interp::run` へ渡す（実際の `.onnx` バイト列取り込みと
    // 同じ経路）。期待値は `ops::*` の手動連結ではなく、`nn::Conv2d`／
    // `nn::MaxPool2d::forward_host`（独立実装。backend-cpu 参照経路）の
    // 連結結果と突合する。
    let conv = Conv2d::new(2, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, true, 300).unwrap();
    let x = rand_tensor(&[1, 2, 8, 8], 10);

    let conv_attrs = ConvAttrs {
        kernel_shape: vec![3, 3],
        strides: vec![1, 1],
        pads: vec![1, 1, 1, 1],
        dilations: vec![1, 1],
        group: 1,
        auto_pad: String::new(),
    };
    let model = build_conv_then_pool_model(
        conv.weight(),
        conv.bias().unwrap(),
        &conv_attrs,
        "MaxPool",
        vec![
            attr_ints("kernel_shape", vec![2, 2]),
            attr_ints("strides", vec![2, 2]),
        ],
    );
    let bytes = proto::encode_model(&model);
    let decoded = proto::decode_model(&bytes).expect("decode_model は成功するはず");
    let graph = build_graph(&decoded).expect("build_graph は成功するはず");

    let mut feeds = HashMap::new();
    feeds.insert("x".to_string(), Value::F32(x.clone()));
    let result = run(&graph, feeds).expect("run は成功するはず");
    let interp_out = match &result["y"] {
        Value::F32(t) => t.clone(),
        other => panic!("Value::F32 を期待したが {other:?}"),
    };

    let ops = CpuBackendOps::new();
    let conv_out = conv.forward_host(&ops, &x).unwrap();
    let nn_max = MaxPool2d::new([2, 2], Some([2, 2]), [0, 0], [1, 1]).unwrap();
    let reference = nn_max.forward_host(&ops, &conv_out).unwrap();

    assert_eq!(interp_out.shape(), reference.shape());
    assert_parity(
        "decode_model->build_graph->run（Conv->MaxPool） vs nn::Conv2d->MaxPool2d::forward_host",
        &dense(&interp_out),
        &dense(&reference),
    );
}

#[test]
fn conv_averagepool_serialized_graph_matches_nn_within_req2() {
    // Conv(3->2,k3,s1,p1) -> AveragePool(k2,s2,count_include_pad=1)。
    // `conv_maxpool_serialized_graph_matches_nn_within_req2` と同型だが
    // pool 種別を AveragePool に差し替え、`count_include_pad` 属性（INT 型）
    // の decode 経由での結線も併せて固定化する。
    let conv = Conv2d::new(3, 2, [3, 3], [1, 1], [1, 1], [1, 1], 1, true, 400).unwrap();
    let x = rand_tensor(&[2, 3, 8, 8], 11);

    let conv_attrs = ConvAttrs {
        kernel_shape: vec![3, 3],
        strides: vec![1, 1],
        pads: vec![1, 1, 1, 1],
        dilations: vec![1, 1],
        group: 1,
        auto_pad: String::new(),
    };
    let model = build_conv_then_pool_model(
        conv.weight(),
        conv.bias().unwrap(),
        &conv_attrs,
        "AveragePool",
        vec![
            attr_ints("kernel_shape", vec![2, 2]),
            attr_ints("strides", vec![2, 2]),
            attr_int("count_include_pad", 1),
        ],
    );
    let bytes = proto::encode_model(&model);
    let decoded = proto::decode_model(&bytes).expect("decode_model は成功するはず");
    let graph = build_graph(&decoded).expect("build_graph は成功するはず");

    let mut feeds = HashMap::new();
    feeds.insert("x".to_string(), Value::F32(x.clone()));
    let result = run(&graph, feeds).expect("run は成功するはず");
    let interp_out = match &result["y"] {
        Value::F32(t) => t.clone(),
        other => panic!("Value::F32 を期待したが {other:?}"),
    };

    let ops = CpuBackendOps::new();
    let conv_out = conv.forward_host(&ops, &x).unwrap();
    let nn_avg = AvgPool2d::new([2, 2], Some([2, 2]), [0, 0], true).unwrap();
    let reference = nn_avg.forward_host(&ops, &conv_out).unwrap();

    assert_eq!(interp_out.shape(), reference.shape());
    assert_parity(
        "decode_model->build_graph->run（Conv->AveragePool） vs nn::Conv2d->AvgPool2d::forward_host",
        &dense(&interp_out),
        &dense(&reference),
    );
}
