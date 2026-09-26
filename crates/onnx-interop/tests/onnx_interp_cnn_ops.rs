//! `onnx::interp::run` の `GlobalAveragePool`／`BatchNormalization`／
//! `Flatten` 結線（イシュー #2200・親 #2185）統合テスト。
//!
//! `tests/onnx_interp.rs`（decode → build_graph → run の全経路）とは異なり、
//! 本ファイルは `onnx::graph::Graph` を直接組み立てて `interp::run` を呼ぶ
//! （`tests/onnx_interp_backend_dispatch.rs` と同じ方式。ONNX proto decode
//! 層は本イシューのスコープ外——decode 層は #2200 実装計画で変更していない
//! ため、decode → build_graph 経路の固定化は #2076（Conv 導入時）の
//! `tests/onnx_interp.rs` 拡張が既に担う）。
//!
//! 数値契約: `BatchNorm`／`GlobalAveragePool`／`Flatten` はいずれも
//! `fandhe_ai_autodiff::nn` の対応する層（`CpuBackendOps` 経由の
//! `Module::forward_host`。tape 不要のホスト常駐 forward）と**同一の逐語式**
//! （`ops/batch_norm.rs`／`ops/global_average_pool.rs` モジュール doc の
//! 数値契約節が backend-cpu 参照実装との対応を明記）を通るため、**bit 完全
//! 一致**（`to_bits()` 比較）で検証する。PyTorch 実機で生成した fixture
//! との突合（REQ-7 判定）は torch のない本環境では実施できないため、PR 本文
//! に申し送りとして記録する（イシュー #2200 実装計画 §7）。

use std::collections::HashMap;

use fandhe_ai_autodiff::nn::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, BatchNorm1d, BatchNorm2d, Flatten, Module,
};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_onnx_interop::onnx::graph::Graph;
use fandhe_ai_onnx_interop::onnx::interp::{InterpError, Value, run};
use fandhe_ai_onnx_interop::onnx::proto::{AttributeProto, NodeProto};
use fandhe_ai_tensor_core::Tensor;

// ---- グラフ構築ヘルパ（`interp.rs`／`onnx_interp_backend_dispatch.rs` と同型） ----

fn node(op_type: &str, name: &str, input: Vec<&str>, output: Vec<&str>) -> NodeProto {
    NodeProto {
        input: input.into_iter().map(String::from).collect(),
        output: output.into_iter().map(String::from).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        attribute: vec![],
        domain: String::new(),
    }
}

fn node_with_attrs(
    op_type: &str,
    name: &str,
    input: Vec<&str>,
    output: Vec<&str>,
    attribute: Vec<AttributeProto>,
) -> NodeProto {
    let mut n = node(op_type, name, input, output);
    n.attribute = attribute;
    n
}

fn attr_i64_typed(name: &str, i: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i,
        r#type: fandhe_ai_onnx_interop::onnx::proto::attribute_type::INT,
        ..Default::default()
    }
}

/// 複数ノードのグラフ（initializer なし。全入力を feed で渡す）。
fn graph(nodes: Vec<NodeProto>, inputs: &[&str], outputs: &[&str]) -> Graph {
    Graph {
        nodes,
        initializers: HashMap::new(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
    }
}

fn feed_f32(name: &str, data: Vec<f32>, shape: &[usize]) -> (String, Value) {
    (
        name.to_string(),
        Value::F32(Tensor::<f32>::new(data, shape).unwrap()),
    )
}

fn as_f32_tensor(v: &Value) -> Tensor<f32> {
    match v {
        Value::F32(t) => t.clone(),
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

fn assert_bit_identical(a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(
        a.shape(),
        b.shape(),
        "shape 不一致: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );
    let ac = a.contiguous();
    let bc = b.contiguous();
    let a_slice = ac.as_slice().unwrap();
    let b_slice = bc.as_slice().unwrap();
    assert_eq!(a_slice.len(), b_slice.len());
    for (i, (x, y)) in a_slice.iter().zip(b_slice.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "bit 不一致 index={i}: interp={x} (bits={:x}) reference={y} (bits={:x})",
            x.to_bits(),
            y.to_bits()
        );
    }
}

// ---- BatchNormalization（rank 4。BatchNorm2d eval と bit 一致） ----

#[test]
fn batch_normalization_rank4_matches_autodiff_batch_norm2d_eval() {
    let n = 2usize;
    let c = 3usize;
    let (h, w) = (4usize, 4usize);
    let numel = n * c * h * w;
    let x_data: Vec<f32> = (0..numel).map(|v| (v as f32) * 0.1 - 3.0).collect();

    let weight = vec![1.1f32, 0.9, 1.3];
    let bias = vec![0.2f32, -0.1, 0.05];
    let mean = vec![-1.5f32, 0.0, 2.0];
    let var = vec![0.8f32, 1.5, 0.5];
    let eps = 1e-5f32;

    let node_bn = node(
        "BatchNormalization",
        "bn",
        vec!["x", "scale", "bias", "mean", "var"],
        vec!["y"],
    );
    let g = graph(
        vec![node_bn],
        &["x", "scale", "bias", "mean", "var"],
        &["y"],
    );
    let feeds = HashMap::from([
        feed_f32("x", x_data.clone(), &[n, c, h, w]),
        feed_f32("scale", weight.clone(), &[c]),
        feed_f32("bias", bias.clone(), &[c]),
        feed_f32("mean", mean.clone(), &[c]),
        feed_f32("var", var.clone(), &[c]),
    ]);
    let result = run(&g, feeds).unwrap();
    let interp_out = as_f32_tensor(result.get("y").unwrap());

    // 参照実装: `BatchNorm2d::from_parameters` + eval モード + `forward_host`
    // （tape 不要のホスト常駐 forward。`Module::forward_host` デフォルト
    // オーバーライド実装が backend-cpu の `run_batch_norm_infer_f32` を
    // 呼ぶ。`ops/batch_norm.rs` モジュール doc の数値契約参照）。
    let mut bn2d = BatchNorm2d::from_parameters(
        Some(Tensor::<f32>::new(weight, &[c]).unwrap()),
        Some(Tensor::<f32>::new(bias, &[c]).unwrap()),
        Tensor::<f32>::new(mean, &[c]).unwrap(),
        Tensor::<f32>::new(var, &[c]).unwrap(),
        eps,
        0.1,
    )
    .unwrap();
    bn2d.set_training(false);
    let x = Tensor::<f32>::new(x_data, &[n, c, h, w]).unwrap();
    let ops = CpuBackendOps::new();
    let reference = bn2d.forward_host(&ops, &x).unwrap();

    assert_bit_identical(&interp_out, &reference);
}

// ---- BatchNormalization（rank 2・rank 3。BatchNorm1d eval と bit 一致） ----

#[test]
fn batch_normalization_rank2_matches_autodiff_batch_norm1d_eval() {
    let n = 4usize;
    let c = 3usize;
    let x_data: Vec<f32> = (0..(n * c)).map(|v| (v as f32) * 0.3 - 1.0).collect();
    let weight = vec![1.0f32, 2.0, 0.5];
    let bias = vec![0.0f32, 0.1, -0.2];
    let mean = vec![0.5f32, -0.3, 1.0];
    let var = vec![1.2f32, 0.7, 2.0];
    let eps = 1e-5f32;

    let node_bn = node(
        "BatchNormalization",
        "bn",
        vec!["x", "scale", "bias", "mean", "var"],
        vec!["y"],
    );
    let g = graph(
        vec![node_bn],
        &["x", "scale", "bias", "mean", "var"],
        &["y"],
    );
    let feeds = HashMap::from([
        feed_f32("x", x_data.clone(), &[n, c]),
        feed_f32("scale", weight.clone(), &[c]),
        feed_f32("bias", bias.clone(), &[c]),
        feed_f32("mean", mean.clone(), &[c]),
        feed_f32("var", var.clone(), &[c]),
    ]);
    let result = run(&g, feeds).unwrap();
    let interp_out = as_f32_tensor(result.get("y").unwrap());

    let mut bn1d = BatchNorm1d::from_parameters(
        Some(Tensor::<f32>::new(weight, &[c]).unwrap()),
        Some(Tensor::<f32>::new(bias, &[c]).unwrap()),
        Tensor::<f32>::new(mean, &[c]).unwrap(),
        Tensor::<f32>::new(var, &[c]).unwrap(),
        eps,
        0.1,
    )
    .unwrap();
    bn1d.set_training(false);
    let x = Tensor::<f32>::new(x_data, &[n, c]).unwrap();
    let ops = CpuBackendOps::new();
    let reference = bn1d.forward_host(&ops, &x).unwrap();

    assert_bit_identical(&interp_out, &reference);
}

#[test]
fn batch_normalization_rank3_matches_autodiff_batch_norm1d_eval() {
    let n = 2usize;
    let c = 2usize;
    let l = 5usize;
    let x_data: Vec<f32> = (0..(n * c * l)).map(|v| (v as f32) * 0.2 - 2.0).collect();
    let weight = vec![1.5f32, 0.8];
    let bias = vec![0.1f32, -0.2];
    let mean = vec![0.0f32, 1.0];
    let var = vec![1.0f32, 0.5];
    let eps = 1e-5f32;

    let node_bn = node(
        "BatchNormalization",
        "bn",
        vec!["x", "scale", "bias", "mean", "var"],
        vec!["y"],
    );
    let g = graph(
        vec![node_bn],
        &["x", "scale", "bias", "mean", "var"],
        &["y"],
    );
    let feeds = HashMap::from([
        feed_f32("x", x_data.clone(), &[n, c, l]),
        feed_f32("scale", weight.clone(), &[c]),
        feed_f32("bias", bias.clone(), &[c]),
        feed_f32("mean", mean.clone(), &[c]),
        feed_f32("var", var.clone(), &[c]),
    ]);
    let result = run(&g, feeds).unwrap();
    let interp_out = as_f32_tensor(result.get("y").unwrap());

    let mut bn1d = BatchNorm1d::from_parameters(
        Some(Tensor::<f32>::new(weight, &[c]).unwrap()),
        Some(Tensor::<f32>::new(bias, &[c]).unwrap()),
        Tensor::<f32>::new(mean, &[c]).unwrap(),
        Tensor::<f32>::new(var, &[c]).unwrap(),
        eps,
        0.1,
    )
    .unwrap();
    bn1d.set_training(false);
    let x = Tensor::<f32>::new(x_data, &[n, c, l]).unwrap();
    let ops = CpuBackendOps::new();
    let reference = bn1d.forward_host(&ops, &x).unwrap();

    assert_bit_identical(&interp_out, &reference);
}

// ---- GlobalAveragePool（rank 4・rank 3。AdaptiveAvgPool2d/1d([1]) と bit 一致） ----

#[test]
fn global_average_pool_rank4_matches_adaptive_avg_pool2d_1x1() {
    let (n, c, h, w) = (2usize, 3usize, 5usize, 7usize);
    let x_data: Vec<f32> = (0..(n * c * h * w))
        .map(|v| (v as f32) * 0.37 - 10.0)
        .collect();

    let node_gap = node("GlobalAveragePool", "gap", vec!["x"], vec!["y"]);
    let g = graph(vec![node_gap], &["x"], &["y"]);
    let feeds = HashMap::from([feed_f32("x", x_data.clone(), &[n, c, h, w])]);
    let result = run(&g, feeds).unwrap();
    let interp_out = as_f32_tensor(result.get("y").unwrap());
    assert_eq!(interp_out.shape(), &[n, c, 1, 1]);

    let pool = AdaptiveAvgPool2d::new([1, 1]).unwrap();
    let x = Tensor::<f32>::new(x_data, &[n, c, h, w]).unwrap();
    let ops = CpuBackendOps::new();
    let reference = pool.forward_host(&ops, &x).unwrap();

    assert_bit_identical(&interp_out, &reference);
}

#[test]
fn global_average_pool_rank3_matches_adaptive_avg_pool1d_1() {
    let (n, c, l) = (2usize, 3usize, 9usize);
    let x_data: Vec<f32> = (0..(n * c * l)).map(|v| (v as f32) * 0.5 - 4.0).collect();

    let node_gap = node("GlobalAveragePool", "gap", vec!["x"], vec!["y"]);
    let g = graph(vec![node_gap], &["x"], &["y"]);
    let feeds = HashMap::from([feed_f32("x", x_data.clone(), &[n, c, l])]);
    let result = run(&g, feeds).unwrap();
    let interp_out = as_f32_tensor(result.get("y").unwrap());
    assert_eq!(interp_out.shape(), &[n, c, 1]);

    let pool = AdaptiveAvgPool1d::new(1).unwrap();
    let x = Tensor::<f32>::new(x_data, &[n, c, l]).unwrap();
    let ops = CpuBackendOps::new();
    let reference = pool.forward_host(&ops, &x).unwrap();

    assert_bit_identical(&interp_out, &reference);
}

// ---- Flatten（axis=1。autodiff Flatten(1, rank-1) と bit 一致・shape 一致） ----

#[test]
fn flatten_axis1_matches_autodiff_flatten() {
    let (n, c, h, w) = (2usize, 3usize, 4usize, 4usize);
    let x_data: Vec<f32> = (0..(n * c * h * w)).map(|v| v as f32).collect();

    let node_flatten = node("Flatten", "flatten", vec!["x"], vec!["y"]);
    let g = graph(vec![node_flatten], &["x"], &["y"]);
    let feeds = HashMap::from([feed_f32("x", x_data.clone(), &[n, c, h, w])]);
    let result = run(&g, feeds).unwrap();
    let interp_out = as_f32_tensor(result.get("y").unwrap());
    assert_eq!(interp_out.shape(), &[n, c * h * w]);

    // `Var::flatten(1, rank-1)`（`start_dim=1, end_dim=3`）は入力の先頭軸
    // （N のみ）を残しそれ以降を 1 軸へ潰す。ONNX `Flatten(axis=1)`
    // （outer=[..axis) の積・inner=[axis..) の積）は、outer が単一軸 N の
    // ときこれと同値になる（本テストの入力形状はその条件を満たす）。
    let flatten_layer = Flatten::new(1, 3);
    let x = Tensor::<f32>::new(x_data, &[n, c, h, w]).unwrap();
    let ops = CpuBackendOps::new();
    let reference = flatten_layer.forward_host(&ops, &x).unwrap();

    assert_bit_identical(&interp_out, &reference);
}

#[test]
fn flatten_axis_out_of_range_rejected() {
    let node_flatten = node_with_attrs(
        "Flatten",
        "flatten",
        vec!["x"],
        vec!["y"],
        vec![attr_i64_typed("axis", 5)],
    );
    let g = graph(vec![node_flatten], &["x"], &["y"]);
    let feeds = HashMap::from([feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[2, 2])]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::Op(_)));
}

// ---- PyTorch export 構造の連鎖: X -> BN -> Relu -> GAP -> Flatten -> Gemm ----

#[test]
fn pytorch_style_bn_relu_gap_flatten_gemm_chain_matches_autodiff() {
    // `torch.onnx.export` された eval モード CNN は BN を Conv に畳み込む
    // ため、BN 単体グラフは BN の直前に Conv を置かない（イシュー #2200
    // 実装計画 §3.6）。小さい batch=2 の連鎖で検証する。
    let (n, c, h, w) = (2usize, 2usize, 3usize, 3usize);
    let x_data: Vec<f32> = (0..(n * c * h * w))
        .map(|v| (v as f32) * 0.2 - 1.0)
        .collect();

    let bn_weight = vec![1.0f32, 1.2];
    let bn_bias = vec![0.0f32, 0.1];
    let bn_mean = vec![0.0f32, 0.0];
    let bn_var = vec![1.0f32, 1.0];
    let eps = 1e-5f32;

    // Gemm: [N, C] x [C, out_features] + bias。gemm.rs は A×B（trans なし）で
    // B の shape は [C, out_features]。
    let out_features = 2usize;
    let gemm_w: Vec<f32> = (0..(c * out_features)).map(|v| v as f32 * 0.1).collect();
    let gemm_b = vec![0.5f32, -0.5];

    let nodes = vec![
        node(
            "BatchNormalization",
            "bn",
            vec!["x", "bn_w", "bn_b", "bn_mean", "bn_var"],
            vec!["bn_out"],
        ),
        node("Relu", "relu", vec!["bn_out"], vec!["relu_out"]),
        node(
            "GlobalAveragePool",
            "gap",
            vec!["relu_out"],
            vec!["gap_out"],
        ),
        node("Flatten", "flatten", vec!["gap_out"], vec!["flat_out"]),
        node(
            "Gemm",
            "gemm",
            vec!["flat_out", "gemm_w", "gemm_b"],
            vec!["y"],
        ),
    ];
    let g = graph(
        nodes,
        &["x", "bn_w", "bn_b", "bn_mean", "bn_var", "gemm_w", "gemm_b"],
        &["y"],
    );
    let feeds = HashMap::from([
        feed_f32("x", x_data.clone(), &[n, c, h, w]),
        feed_f32("bn_w", bn_weight.clone(), &[c]),
        feed_f32("bn_b", bn_bias.clone(), &[c]),
        feed_f32("bn_mean", bn_mean.clone(), &[c]),
        feed_f32("bn_var", bn_var.clone(), &[c]),
        feed_f32("gemm_w", gemm_w.clone(), &[c, out_features]),
        feed_f32("gemm_b", gemm_b.clone(), &[out_features]),
    ]);
    let result = run(&g, feeds).unwrap();
    let interp_out = as_f32_tensor(result.get("y").unwrap());
    assert_eq!(interp_out.shape(), &[n, out_features]);

    // 参照実装: autodiff の同構成層を `forward_host` で順に適用する。
    let ops = CpuBackendOps::new();
    let x = Tensor::<f32>::new(x_data, &[n, c, h, w]).unwrap();
    let mut bn2d = BatchNorm2d::from_parameters(
        Some(Tensor::<f32>::new(bn_weight, &[c]).unwrap()),
        Some(Tensor::<f32>::new(bn_bias, &[c]).unwrap()),
        Tensor::<f32>::new(bn_mean, &[c]).unwrap(),
        Tensor::<f32>::new(bn_var, &[c]).unwrap(),
        eps,
        0.1,
    )
    .unwrap();
    bn2d.set_training(false);
    let bn_out = bn2d.forward_host(&ops, &x).unwrap();
    // `Relu`／`Gemm` は interp が既に `ops::relu`／`ops::gemm`（本クレート
    // 純粋関数）へディスパッチしているため、本テストの焦点である
    // `BatchNormalization`／`GlobalAveragePool`／`Flatten` の 3 op と違い
    // 二重参照実装を用意する意味がない。同じ関数を直接呼んで連鎖の残り
    // （Relu・Gemm）を再現し、全体としての shape・値の連鎖を固定化する。
    let relu_out = fandhe_ai_onnx_interop::ops::relu(&bn_out).unwrap();
    let gap = AdaptiveAvgPool2d::new([1, 1]).unwrap();
    let gap_out = gap.forward_host(&ops, &relu_out).unwrap();
    let flatten_layer = Flatten::new(1, 3);
    let flat_out = flatten_layer.forward_host(&ops, &gap_out).unwrap();
    let gemm_w_t = Tensor::<f32>::new(gemm_w, &[c, out_features]).unwrap();
    let gemm_b_t = Tensor::<f32>::new(gemm_b, &[out_features]).unwrap();
    let reference = fandhe_ai_onnx_interop::ops::gemm(
        &flat_out,
        &gemm_w_t,
        Some(&gemm_b_t),
        &fandhe_ai_onnx_interop::ops::GemmAttrs {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b: false,
        },
    )
    .unwrap();

    assert_bit_identical(&interp_out, &reference);
}
