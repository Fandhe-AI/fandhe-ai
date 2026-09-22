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
//! ## 対応層（イシュー #2076・親 #2034 で Sigmoid・Softmax・LayerNorm・
//! GELU（erf 版）・Conv2d へ拡大）
//!
//! [`fandhe_ai_autodiff::nn::Module`] は種類の異なる層を統一シグネチャで
//! 扱う trait object（`Box<dyn Module>`）だが、実際の型を判別する公開手段は
//! `as_linear()`／`as_relu()` 等の閉じたダウンキャストフック集合のみで
//! `Any` は使わない（`docs/compat-api-scope.md` §1 の閉集合方針）。本モジュールは
//! そのうち以下 7 種に対応する:
//!
//! | フック | 対応 [`ExportOp`] |
//! |---|---|
//! | `as_linear` が `Some` | [`ExportOp::Gemm`] |
//! | `as_relu` が `true` | [`ExportOp::Relu`] |
//! | `as_sigmoid` が `true` | [`ExportOp::Sigmoid`] |
//! | `as_softmax` が `Some` | [`ExportOp::Softmax`]（`axis = Softmax::dim() as i64`） |
//! | `as_layer_norm` が `Some` | [`ExportOp::LayerNormalization`]（`axis = -1`） |
//! | `as_gelu` が `true` | `Mul → Erf → Add → Mul → Mul` の 5 演算ノード（下記） |
//! | `as_conv2d` が `Some` | [`ExportOp::Conv`] |
//!
//! `LayerNorm::weight() == None`（`without_affine`）は ONNX
//! `LayerNormalization` の `Scale` 必須制約と両立しないため
//! [`ExportError::InvalidLayerParameter`] で fail-closed に拒否する
//! （ones initializer を合成すると「initializer 名 = `state_dict` キー」
//! 契約を破るため不採用。`docs/onnx-export-op-mapping.md` §7 参照）。
//! `GELU` は ONNX opset 17 に単体演算が無いため `gelu(x) = 0.5 · x ·
//! (1 + erf(x / √2))` を `Mul(x, 1/√2) → Erf → Add(1.0) → Mul(x, ·) →
//! Mul(0.5)` の 5 ノード（+ `Constant` 3 ノード）へ合成する（`GeluTanh`
//! は対応する ONNX 演算が無いため対象外のまま）。`Softmax::dim()` は
//! [`ExportOp::Softmax::axis`] を組み立てるため `pub` へ変更した
//! （`crates/autodiff/src/nn/activation.rs`。facade は `nn::activation::
//! Softmax` を再エクスポートしないため facade の公開面は拡張しない）。
//! `LogSoftmax` は対応する ONNX 演算が無いため対象外のまま。
//!
//! Sigmoid の数値契約は `docs/facade-onnx-export-exposure-decision.md`
//! §15.7 項 5（承認保留）の対象で、本 issue（#2076）では REQ-2 統一
//! 複合判定（推奨案 (α)）を前提に実装する（`docs/onnx-export-op-mapping.md`
//! §7 参照。既存 tolerance 定数は変更しない）。
//!
//! それ以外の層（Tanh・GeluTanh・LogSoftmax・Dropout・Conv1d・RmsNorm・
//! BatchNorm・Embedding・MultiheadAttention 等）は
//! [`ExportError::UnsupportedLayer`] で fail-closed に拒否する。`layer_kind`
//! フィールドは `Module` trait が公開する既存ダウンキャストフックのうち
//! 本モジュールが未対応のもの（`as_conv1d`／`as_rms_norm`／
//! `as_batch_norm1d`／`as_batch_norm2d`／`as_embedding`／
//! `as_multihead_attention`）で判別できる範囲のみ具体名を報告し、
//! それ以外は `"unknown"` とする。
//!
//! ## `autodiff` API バージョン制約について
//!
//! `crates/onnx-interop/Cargo.toml` は `fandhe-ai-autodiff` を
//! `version = "=0.9.0"` 併記の通常依存として宣言している
//! （crates.io 公開クレート間 path 依存の公開要件。
//! `docs/crates-io-publishing-order.md`）。単一クレート
//! `cargo publish --dry-run` はこの `version` 制約に従い registry から
//! `fandhe-ai-autodiff =0.9.0` を取得してビルド検証するため、本 issue
//! で追加した `Module::as_sigmoid`／`as_gelu`／`as_softmax`（crates.io
//! 未公開の新 API）はその検証を通らない。ただし
//! `docs/facade-onnx-export-exposure-decision.md` §15.2 項 5・§17.3 が
//! 整理するとおり単一クレート dry-run の失敗は既知の非ブロッカーで
//! あり、7 パッケージ一括 dry-run（`docs/crates-io-publishing-order.md`
//! §8.1）はワークスペース内のローカル解決で成立するため、本 issue の
//! ブロッカーにはならない。
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
//! - GELU（複数ノードに展開される唯一の層）の中間名: `layer{i}_gelu_{k}`
//!   （`k` は演算段。0 始まり・最終段は上記 `layer{i}_out` へ合流）・
//!   ノード名 `layer{i}_{k}`・定数ノード名／出力名 `layer{i}_c_inv_sqrt2`／
//!   `layer{i}_c_one`／`layer{i}_c_half`。
//!
//! ## 検証順序（fail-closed。`security.md` A08）
//!
//! [`export_parts_from_layers`] は `ExportNode`／`RawTensor` を 1 つも
//! 構築する前にすべての層を検証し、1 件でも失敗した時点で即座に `Err`
//! を返す（部分的に構築されたグラフを返さない）。検証対象:
//!
//! 1. 層列が空でないこと（[`ExportError::EmptyModel`]）
//! 2. 各層が対応 7 種のいずれかに該当すること
//!    （[`ExportError::UnsupportedLayer`]）
//! 3. `Linear`／`Conv2d` の `weight`／`bias` の rank・長さ整合、
//!    `LayerNorm` の `weight`（`Scale`）が `Some` であること
//!    （[`ExportError::InvalidLayerParameter`]）
//! 4. shape の各次元・`Softmax::dim()` が `usize -> i64` へ変換可能で
//!    あること（同上）
//! 5. 生成するテンソル名すべてが一意であること
//!    （[`ExportError::DuplicateTensorName`]。現行の命名規約では構造上
//!    発生し得ないが、設計要件として機械的に検査する）
//!
//! ## bit 一致契約の前提（テスト側の注記）
//!
//! 本モジュール自体は算術を行わない（既存 `Linear::weight()`／`bias()`
//! 等の値をそのままコピーするのみ）ため、export 単体は決定的である。
//! roundtrip（export → `interp::run`）の出力が手動 forward
//! （`Module::forward_host`）と bit 完全一致することを主張できるのは
//! `Linear`／`Relu` 層のみのモデルに限る（CPU `Relu` は `x.max(0.0)`・
//! `interp::ops::relu` は `nan_propagating_max` を使い、±0.0 同士の
//! `max` は符号ビットが実装依存になりうるため、GEMM 出力に厳密な
//! `±0.0` が現れない入力を前提とする）。Sigmoid・Softmax・LayerNorm・
//! GELU・Conv2d を含むモデルは結合順序・実装経路が異なる
//! （`.claude/rules/coding-rust.md` FMA 契約）ため REQ-2 統一複合判定
//! （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で検証する。

use std::collections::HashSet;

use fandhe_ai_autodiff::nn::Module;
use fandhe_ai_tensor_core::Tensor;

use super::export::ExportError;
use super::export_ops::{ConstantAttr, ExportNode, ExportOp};
use super::graph::{Graph, RawTensor};
use crate::ops::{ConvAttrs, GemmAttrs, LayerNormAttrs};

/// `gelu(x) = 0.5 · x · (1 + erf(x / √2))` の `1/√2` 定数（モジュール冒頭
/// 「対応層」節の GELU 合成 1 段目）。
const GELU_INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

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
    if layer.as_conv1d().is_some() {
        "Conv1d"
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
    /// 層 1 個分のノード列（GELU のみ複数ノードへ展開される。それ以外の
    /// 対応層は要素数 1。モジュール冒頭「対応層」節参照）。
    nodes: Vec<ExportNode>,
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
                nodes: vec![ExportNode {
                    name: format!("layer{index}"),
                    op: ExportOp::Gemm(GemmAttrs {
                        alpha: 1.0,
                        beta: 1.0,
                        trans_a: false,
                        trans_b: false,
                    }),
                    inputs,
                    outputs: vec![out_name.clone()],
                }],
                initializers,
            });
        } else if layer.as_relu() {
            insert_name(out_name.clone())?;
            planned.push(PlannedLayer {
                nodes: vec![ExportNode {
                    name: format!("layer{index}"),
                    op: ExportOp::Relu,
                    inputs: vec![current_output.clone()],
                    outputs: vec![out_name.clone()],
                }],
                initializers: Vec::new(),
            });
        } else if layer.as_sigmoid() {
            insert_name(out_name.clone())?;
            planned.push(PlannedLayer {
                nodes: vec![ExportNode {
                    name: format!("layer{index}"),
                    op: ExportOp::Sigmoid,
                    inputs: vec![current_output.clone()],
                    outputs: vec![out_name.clone()],
                }],
                initializers: Vec::new(),
            });
        } else if let Some(softmax) = layer.as_softmax() {
            let axis =
                i64::try_from(softmax.dim()).map_err(|_| ExportError::InvalidLayerParameter {
                    index,
                    reason: format!("Softmax の dim {} が i64 の範囲に収まらない", softmax.dim()),
                })?;
            insert_name(out_name.clone())?;
            planned.push(PlannedLayer {
                nodes: vec![ExportNode {
                    name: format!("layer{index}"),
                    op: ExportOp::Softmax { axis },
                    inputs: vec![current_output.clone()],
                    outputs: vec![out_name.clone()],
                }],
                initializers: Vec::new(),
            });
        } else if let Some(layer_norm) = layer.as_layer_norm() {
            let weight = layer_norm
                .weight()
                .ok_or_else(|| ExportError::InvalidLayerParameter {
                    index,
                    reason: "ONNX LayerNormalization は Scale（weight）必須のため、\
                         without_affine（weight == None）の LayerNorm は export できない"
                        .to_string(),
                })?;
            let weight_shape = weight.shape();

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

            if let Some(bias) = layer_norm.bias() {
                let bias_name = format!("{index}.bias");
                insert_name(bias_name.clone())?;
                let bias_data = tensor_to_f32_vec(index, bias)?;
                let bias_dims = shape_to_i64_dims(index, bias.shape())?;
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
                nodes: vec![ExportNode {
                    name: format!("layer{index}"),
                    op: ExportOp::LayerNormalization(LayerNormAttrs {
                        axis: -1,
                        epsilon: layer_norm.eps(),
                    }),
                    inputs,
                    outputs: vec![out_name.clone()],
                }],
                initializers,
            });
        } else if layer.as_gelu() {
            // `gelu(x) = 0.5 · x · (1 + erf(x / √2))` を 5 演算ノード
            // （+ `Constant` 3 ノード）へ合成する（モジュール冒頭「対応層」
            // 節・`docs/onnx-export-op-mapping.md` §7 参照）。ONNX opset 17
            // に GELU 単体演算が無いための代替。
            let c_inv_sqrt2 = format!("layer{index}_c_inv_sqrt2");
            let c_one = format!("layer{index}_c_one");
            let c_half = format!("layer{index}_c_half");
            insert_name(c_inv_sqrt2.clone())?;
            insert_name(c_one.clone())?;
            insert_name(c_half.clone())?;

            let g0 = format!("layer{index}_gelu_0");
            let g1 = format!("layer{index}_gelu_1");
            let g2 = format!("layer{index}_gelu_2");
            insert_name(g0.clone())?;
            insert_name(g1.clone())?;
            insert_name(g2.clone())?;
            insert_name(out_name.clone())?;

            let x = current_output.clone();
            let nodes = vec![
                ExportNode {
                    name: c_inv_sqrt2.clone(),
                    op: ExportOp::Constant(ConstantAttr::Float(GELU_INV_SQRT2)),
                    inputs: Vec::new(),
                    outputs: vec![c_inv_sqrt2.clone()],
                },
                ExportNode {
                    name: c_one.clone(),
                    op: ExportOp::Constant(ConstantAttr::Float(1.0)),
                    inputs: Vec::new(),
                    outputs: vec![c_one.clone()],
                },
                ExportNode {
                    name: c_half.clone(),
                    op: ExportOp::Constant(ConstantAttr::Float(0.5)),
                    inputs: Vec::new(),
                    outputs: vec![c_half.clone()],
                },
                ExportNode {
                    name: format!("layer{index}_0"),
                    op: ExportOp::Mul,
                    inputs: vec![x.clone(), c_inv_sqrt2],
                    outputs: vec![g0.clone()],
                },
                ExportNode {
                    name: format!("layer{index}_1"),
                    op: ExportOp::Erf,
                    inputs: vec![g0],
                    outputs: vec![g1.clone()],
                },
                ExportNode {
                    name: format!("layer{index}_2"),
                    op: ExportOp::Add,
                    inputs: vec![g1, c_one],
                    outputs: vec![g2.clone()],
                },
                ExportNode {
                    name: format!("layer{index}_3"),
                    op: ExportOp::Mul,
                    inputs: vec![x, g2],
                    outputs: vec![format!("layer{index}_gelu_3")],
                },
                ExportNode {
                    name: format!("layer{index}_4"),
                    op: ExportOp::Mul,
                    inputs: vec![format!("layer{index}_gelu_3"), c_half],
                    outputs: vec![out_name.clone()],
                },
            ];
            insert_name(format!("layer{index}_gelu_3"))?;
            planned.push(PlannedLayer {
                nodes,
                initializers: Vec::new(),
            });
        } else if let Some(conv) = layer.as_conv2d() {
            let weight = conv.weight();
            let weight_shape = weight.shape();
            let (cout, _cin_g, kh, kw) = (
                weight_shape[0],
                weight_shape[1],
                weight_shape[2],
                weight_shape[3],
            );

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

            if let Some(bias) = conv.bias() {
                if bias.rank() != 1 || bias.shape()[0] != cout {
                    return Err(ExportError::InvalidLayerParameter {
                        index,
                        reason: format!(
                            "Conv2d の bias は shape [{cout}]（rank 1）でなければならない\
                             （実際 {:?}）",
                            bias.shape()
                        ),
                    });
                }
                let bias_name = format!("{index}.bias");
                insert_name(bias_name.clone())?;
                let bias_data = tensor_to_f32_vec(index, bias)?;
                let bias_dims = shape_to_i64_dims(index, bias.shape())?;
                inputs.push(bias_name.clone());
                initializers.push((
                    bias_name,
                    RawTensor::F32 {
                        data: bias_data,
                        shape: bias_dims,
                    },
                ));
            }

            let [sh, sw] = conv.stride();
            let [ph, pw] = conv.padding();
            let [dh, dw] = conv.dilation();
            let group =
                i64::try_from(conv.groups()).map_err(|_| ExportError::InvalidLayerParameter {
                    index,
                    reason: format!(
                        "Conv2d の groups {} が i64 の範囲に収まらない",
                        conv.groups()
                    ),
                })?;
            let to_i64_pair = |name: &str, pair: [usize; 2]| -> Result<Vec<i64>, ExportError> {
                pair.iter()
                    .map(|&v| {
                        i64::try_from(v).map_err(|_| ExportError::InvalidLayerParameter {
                            index,
                            reason: format!("Conv2d の {name} 値 {v} が i64 の範囲に収まらない"),
                        })
                    })
                    .collect()
            };
            let strides = to_i64_pair("stride", [sh, sw])?;
            let dilations = to_i64_pair("dilation", [dh, dw])?;
            let pads_hw = to_i64_pair("padding", [ph, pw])?;
            let pads = vec![pads_hw[0], pads_hw[1], pads_hw[0], pads_hw[1]];
            let kernel_shape = to_i64_pair("kernel_size", [kh, kw])?;

            insert_name(out_name.clone())?;
            planned.push(PlannedLayer {
                nodes: vec![ExportNode {
                    name: format!("layer{index}"),
                    op: ExportOp::Conv(ConvAttrs {
                        kernel_shape,
                        strides,
                        pads,
                        dilations,
                        group,
                        auto_pad: "NOTSET".to_string(),
                    }),
                    inputs,
                    outputs: vec![out_name.clone()],
                }],
                initializers,
            });
        } else {
            return Err(ExportError::UnsupportedLayer {
                index,
                layer_kind: layer_kind_name(layer.as_ref()),
            });
        }

        current_output = out_name;
    }

    let mut nodes = Vec::new();
    let mut initializers = Vec::new();
    for layer in planned {
        nodes.extend(layer.nodes);
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
