//! `fandhe_ai_autodiff::nn::Module` の層列（`&[Box<dyn Module>]`。通常
//! `nn::Sequential::layers()` から取得）から [`super::export::build_model_proto`]
//! が受け取れる [`Graph`] を組み立てる橋渡しモジュール（イシュー #2036・
//! 親 #2034・設計 `docs/facade-onnx-export-exposure-decision.md` §15）。
//!
//! 本モジュール自体は非公開クレート `onnx-interop` の内部限定であり、
//! facade（`fandhe_ai`）へは再エクスポートしない。facade
//! `OnnxModel::from_sequential`（イシュー #2037・`crates/facade/src/
//! interop/onnx.rs`）が [`graph_from_layers`] を薄く委譲して呼ぶ。
//! 呼び出し元は本クレートの統合テストと上記 facade メソッド。
//!
//! ## 対応層（初期範囲）
//!
//! [`fandhe_ai_autodiff::nn::Module`] は種類の異なる層を統一シグネチャで
//! 扱う trait object（`Box<dyn Module>`）だが、実際の型を判別する公開手段は
//! `as_linear()`／`as_relu()` 等の閉じたダウンキャストフック集合のみで
//! `Any` は使わない（`docs/compat-api-scope.md` §1 の閉集合方針）。本モジュールは
//! そのうち `as_linear`（`Some` なら [`ExportOp::Gemm`]）・`as_relu`（`true` なら
//! [`ExportOp::Relu`]）の 2 種のみに対応する。`docs/facade-onnx-export-exposure-decision.md`
//! §15.7 は「Sigmoid」も初期範囲候補に挙げていたが、`Module` trait に
//! `as_sigmoid` フックが存在せず（承認事項 2「Sigmoid 対応」・承認事項 5
//! 「`as_sigmoid` フック新設」がいずれも未承認のまま）、フック不在のまま
//! `Sigmoid` を判別する手段が無いため本 issue では対象外とする（同 issue
//! コメントで「Linear／ReLU の 2 種へ縮小」する代替案 γ が示されている）。
//!
//! それ以外の層（Sigmoid・Tanh・Dropout・Conv2d・LayerNorm 等）は
//! [`ExportError::UnsupportedLayer`] で fail-closed に拒否する。`layer_kind`
//! フィールドは `Module` trait が公開する既存ダウンキャストフック 8 種
//! （`as_conv2d`／`as_conv1d`／`as_layer_norm`／`as_rms_norm`／
//! `as_batch_norm1d`／`as_batch_norm2d`／`as_embedding`／
//! `as_multihead_attention`）で判別できる範囲のみ具体名を報告し、
//! それ以外は `"unknown"` とする。
//!
//! ## `autodiff` API バージョン制約（重要）
//!
//! `crates/onnx-interop/Cargo.toml` は `fandhe-ai-autodiff` を
//! `version = "=0.9.0"` 併記の通常依存として宣言している
//! （crates.io 公開クレート間 path 依存の公開要件。
//! `docs/crates-io-publishing-order.md`）。`cargo publish --dry-run` は
//! 単一クレートの検証ビルド時にこの `version` 制約に従い registry から
//! `fandhe-ai-autodiff =0.9.0` を取得してビルドするため、本モジュールが
//! 使う `autodiff` の公開 API は **crates.io `fandhe-ai-autodiff =0.9.0`
//! （タグ `v0.9.0`）に存在するものへ限定する**。本モジュールが使うのは
//! `nn::Module`（`as_linear`／`as_relu`／`as_conv2d`／`as_conv1d`／
//! `as_layer_norm`／`as_rms_norm`／`as_batch_norm1d`／`as_batch_norm2d`／
//! `as_embedding`／`as_multihead_attention`）・`nn::Linear`
//! （`weight`／`bias`）のみで、いずれも `v0.9.0` 時点で存在することを
//! 確認済み（`is_pooling` は #1957 で追加された後発 API のため意図的に
//! 使わない）。
//!
//! ## 名前規約（決定的）
//!
//! - graph input: [`GRAPH_INPUT_NAME`]（常に `"input"`）
//! - graph output: [`GRAPH_OUTPUT_NAME`]（常に `"output"`。最終層の出力）
//! - 中間テンソル名: `layer{i}_out`（`i` は層の位置。0 始まり）
//! - ノード名: `layer{i}`
//! - initializer 名: `{i}.weight`／`{i}.bias`（`i` は層の位置。
//!   `nn::Sequential::named_parameters()`／`state_dict()` が使う
//!   `"{index}.{name}"` 接頭辞契約〈`nn::container::ModuleList::named_parameters`〉
//!   と同一形式——`Linear::named_parameters()` が返す `"weight"`／`"bias"`
//!   をそのまま層位置で前置する）
//!
//! ## 検証順序（fail-closed。`security.md` A08）
//!
//! [`export_parts_from_layers`] は `ExportNode`／`RawTensor` を 1 つも
//! 構築する前にすべての層を検証し、1 件でも失敗した時点で即座に `Err`
//! を返す（部分的に構築されたグラフを返さない）。検証対象:
//!
//! 1. 層列が空でないこと（[`ExportError::EmptyModel`]）
//! 2. 各層が `as_linear`／`as_relu` のいずれかに該当すること
//!    （[`ExportError::UnsupportedLayer`]）
//! 3. `Linear` の `weight` が rank 2・`bias`（`Some` の場合）が rank 1 かつ
//!    `bias.len() == weight.shape()[1]` であること
//!    （[`ExportError::InvalidLayerParameter`]）
//! 4. shape の各次元が `usize -> i64` へ変換可能であること（同上）
//! 5. 生成するテンソル名すべてが一意であること
//!    （[`ExportError::DuplicateTensorName`]。現行の命名規約では構造上
//!    発生し得ないが、設計要件として機械的に検査する）
//!
//! ## bit 一致契約の前提（テスト側の注記）
//!
//! 本モジュール自体は算術を行わない（既存 `Linear::weight()`／`bias()`
//! の値をそのままコピーするのみ）ため、export 単体は決定的である。
//! roundtrip（export → `interp::run`）の出力が手動 forward
//! （`Module::forward_host`）と bit 完全一致することを主張するテストは、
//! GEMM 出力に厳密な `±0.0` が現れない入力を前提とする（CPU `Relu` は
//! `x.max(0.0)`・`interp::ops::relu` は `nan_propagating_max` を使い、
//! ±0.0 同士の `max` は符号ビットが実装依存になりうるため）。

use std::collections::HashSet;

use fandhe_ai_autodiff::nn::Module;
use fandhe_ai_tensor_core::Tensor;

use super::export::ExportError;
use super::export_ops::{ExportNode, ExportOp};
use super::graph::{Graph, RawTensor};
use crate::ops::GemmAttrs;

/// graph input の固定名（`"input"`）。facade `OnnxModel::from_sequential`
/// の公開契約（イシュー #2037）が参照する定数（モジュール冒頭
/// ドキュメント「名前規約」参照）。
pub const GRAPH_INPUT_NAME: &str = "input";
/// graph output の固定名（`"output"`）。同上。
pub const GRAPH_OUTPUT_NAME: &str = "output";

/// 層列を [`super::export::build_model_proto`] へ渡す前の中間結果
/// （[`ExportNode`] 列・initializer 列・graph input／output 名）。
/// テスト・[`graph_from_layers`] から検査・利用できるよう公開する。
#[derive(Debug)]
pub struct NnExportParts {
    /// 層順に並んだノード列（`to_node_proto` で `NodeProto` へ変換前）。
    pub nodes: Vec<ExportNode>,
    /// 層順に並んだ initializer（`(名前, RawTensor)`）。`Graph.initializers`
    /// は `HashMap` のため、ここでは決定的な層順の `Vec` として保持する。
    pub initializers: Vec<(String, RawTensor)>,
    /// graph input 名（常に [`GRAPH_INPUT_NAME`]）。
    pub input: String,
    /// graph output 名（常に [`GRAPH_OUTPUT_NAME`]）。
    pub output: String,
}

/// `usize` の shape 次元を `i64`（`TensorProto.dims`／`RawTensor` の shape
/// 表現）へ変換する。失敗（`i64::MAX` 超）は
/// [`ExportError::InvalidLayerParameter`] として層 index 付きで報告する
/// （`export.rs::element_count` と異なり、ここでは export 元テンソルの
/// 次元それぞれの範囲検査のみを担う）。
fn shape_to_i64_dims(index: usize, shape: &[usize]) -> Result<Vec<i64>, ExportError> {
    shape
        .iter()
        .map(|&d| {
            i64::try_from(d).map_err(|_| ExportError::InvalidLayerParameter {
                index,
                reason: format!("shape 次元 {d} が i64 の範囲に収まらない"),
            })
        })
        .collect()
}

/// `Tensor<f32>` を平坦な `Vec<f32>`（行優先・連続領域）へ複製する。
/// `contiguous()` は既に連続な場合はコピーを避ける実装だが、ここでは
/// 呼び出し元が `RawTensor::F32` として所有権を持つ独立バッファが必要な
/// ため `as_slice().to_vec()` で複製する（`Linear::weight()`／`bias()`
/// が返す `&Tensor<f32>` の借用寿命を export 後まで延ばさないため）。
fn tensor_to_f32_vec(index: usize, tensor: &Tensor<f32>) -> Result<Vec<f32>, ExportError> {
    let contiguous = tensor.contiguous();
    contiguous
        .as_slice()
        .map(<[f32]>::to_vec)
        .ok_or_else(|| ExportError::InvalidLayerParameter {
            index,
            reason: "contiguous() 後のテンソルが as_slice() で読み取れない（内部不変条件違反）"
                .to_string(),
        })
}

/// この層が判別可能な `Module` 実装のうち [`export_parts_from_layers`] が
/// 未対応のものであれば、報告用の種別名を返す（モジュール冒頭「対応層」
/// 節参照）。判別できない場合は `"unknown"`。
fn layer_kind_name(layer: &dyn Module) -> &'static str {
    if layer.as_conv2d().is_some() {
        "Conv2d"
    } else if layer.as_conv1d().is_some() {
        "Conv1d"
    } else if layer.as_layer_norm().is_some() {
        "LayerNorm"
    } else if layer.as_rms_norm().is_some() {
        "RmsNorm"
    } else if layer.as_batch_norm1d().is_some() {
        "BatchNorm1d"
    } else if layer.as_batch_norm2d().is_some() {
        "BatchNorm2d"
    } else if layer.as_embedding().is_some() {
        "Embedding"
    } else if layer.as_multihead_attention().is_some() {
        "MultiheadAttention"
    } else {
        "unknown"
    }
}

/// 層 1 個分の export 計画（検証済み。まだ `ExportNode`／`RawTensor` の
/// 所有権を持つ最終形ではなく、全層の検証完了後にまとめてコミットする
/// ための中間状態）。
struct PlannedLayer {
    node: ExportNode,
    initializers: Vec<(String, RawTensor)>,
}

/// `nn::Module` の層列から [`NnExportParts`] を組み立てる。
///
/// モジュール冒頭ドキュメントの「検証順序」節に従い、全層の検証が完了
/// してから初めて戻り値を構築する（1 層でも拒否されれば `Err` を返し、
/// 部分的な結果は一切返さない）。
pub fn export_parts_from_layers(layers: &[Box<dyn Module>]) -> Result<NnExportParts, ExportError> {
    if layers.is_empty() {
        return Err(ExportError::EmptyModel);
    }

    let mut seen_names: HashSet<String> = HashSet::new();
    seen_names.insert(GRAPH_INPUT_NAME.to_string());

    let mut insert_name = |name: String| -> Result<(), ExportError> {
        if seen_names.insert(name.clone()) {
            Ok(())
        } else {
            Err(ExportError::DuplicateTensorName { name })
        }
    };

    let mut planned: Vec<PlannedLayer> = Vec::with_capacity(layers.len());
    let mut current_output = GRAPH_INPUT_NAME.to_string();

    for (index, layer) in layers.iter().enumerate() {
        let is_last = index + 1 == layers.len();
        let out_name = if is_last {
            GRAPH_OUTPUT_NAME.to_string()
        } else {
            format!("layer{index}_out")
        };

        if let Some(linear) = layer.as_linear() {
            let weight = linear.weight();
            if weight.rank() != 2 {
                return Err(ExportError::InvalidLayerParameter {
                    index,
                    reason: format!(
                        "Linear の weight は rank 2 でなければならない（実際 rank {}）",
                        weight.rank()
                    ),
                });
            }
            let weight_shape = weight.shape();
            let out_features = weight_shape[1];

            let weight_name = format!("{index}.weight");
            insert_name(weight_name.clone())?;
            let weight_data = tensor_to_f32_vec(index, weight)?;
            let weight_dims = shape_to_i64_dims(index, weight_shape)?;

            let mut inputs = vec![current_output.clone(), weight_name.clone()];
            let mut initializers = vec![(
                weight_name,
                RawTensor::F32 {
                    data: weight_data,
                    shape: weight_dims,
                },
            )];

            if let Some(bias) = linear.bias() {
                if bias.rank() != 1 {
                    return Err(ExportError::InvalidLayerParameter {
                        index,
                        reason: format!(
                            "Linear の bias は rank 1 でなければならない（実際 rank {}）",
                            bias.rank()
                        ),
                    });
                }
                let bias_shape = bias.shape();
                if bias_shape[0] != out_features {
                    return Err(ExportError::InvalidLayerParameter {
                        index,
                        reason: format!(
                            "Linear の bias 長 {} が weight の out_features {out_features} と一致しない",
                            bias_shape[0]
                        ),
                    });
                }
                let bias_name = format!("{index}.bias");
                insert_name(bias_name.clone())?;
                let bias_data = tensor_to_f32_vec(index, bias)?;
                let bias_dims = shape_to_i64_dims(index, bias_shape)?;
                inputs.push(bias_name.clone());
                initializers.push((
                    bias_name,
                    RawTensor::F32 {
                        data: bias_data,
                        shape: bias_dims,
                    },
                ));
            }

            insert_name(out_name.clone())?;

            planned.push(PlannedLayer {
                node: ExportNode {
                    name: format!("layer{index}"),
                    op: ExportOp::Gemm(GemmAttrs {
                        alpha: 1.0,
                        beta: 1.0,
                        trans_a: false,
                        trans_b: false,
                    }),
                    inputs,
                    outputs: vec![out_name.clone()],
                },
                initializers,
            });
        } else if layer.as_relu() {
            insert_name(out_name.clone())?;
            planned.push(PlannedLayer {
                node: ExportNode {
                    name: format!("layer{index}"),
                    op: ExportOp::Relu,
                    inputs: vec![current_output.clone()],
                    outputs: vec![out_name.clone()],
                },
                initializers: Vec::new(),
            });
        } else {
            return Err(ExportError::UnsupportedLayer {
                index,
                layer_kind: layer_kind_name(layer.as_ref()),
            });
        }

        current_output = out_name;
    }

    let mut nodes = Vec::with_capacity(planned.len());
    let mut initializers = Vec::new();
    for layer in planned {
        nodes.push(layer.node);
        initializers.extend(layer.initializers);
    }

    Ok(NnExportParts {
        nodes,
        initializers,
        input: GRAPH_INPUT_NAME.to_string(),
        output: GRAPH_OUTPUT_NAME.to_string(),
    })
}

/// [`export_parts_from_layers`] の結果を [`super::export::build_model_proto`]
/// へ渡せる [`Graph`] へ組み立てる（`ExportNode` -> `NodeProto` の変換
/// 〈`super::export_ops::to_node_proto`〉を含む）。
///
/// 呼び出し元は本クレートの統合テストと facade
/// `OnnxModel::from_sequential`（イシュー #2037）。`build_model_proto` は
/// `Graph.nodes`
/// （`NodeProto` 列）に対して `export_ops::check_exportable` の allowlist
/// 検査を再度行うため、本関数はその前段で `ExportNode` -> `NodeProto`
/// の変換のみを担う。
pub fn graph_from_layers(layers: &[Box<dyn Module>]) -> Result<Graph, ExportError> {
    let parts = export_parts_from_layers(layers)?;
    let mut nodes = Vec::with_capacity(parts.nodes.len());
    for node in &parts.nodes {
        nodes.push(super::export_ops::to_node_proto(node)?);
    }
    Ok(Graph {
        nodes,
        initializers: parts.initializers.into_iter().collect(),
        inputs: vec![parts.input],
        outputs: vec![parts.output],
    })
}
