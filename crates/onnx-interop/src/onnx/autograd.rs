//! ONNX import グラフ（[`super::graph::Graph`]）を `Var` 入出力の学習可能な
//! 計算グラフとして実行するための autograd 接続層（イシュー #2078・REQ-7／
//! REQ-9）。
//!
//! `docs/facade-onnx-import-exposure-decision.md` §6.3 (c)・§12.6 (c) が
//! 「学習可能化（`Tape`／`Var` への変換層）は未実装・別 issue」と記録していた
//! ギャップを埋める。本モジュールは `fandhe-ai-autodiff`（#2036 で通常依存へ
//! 昇格済み）を利用し、import した ONNX グラフを fine-tuning／transfer
//! learning のベースモデルとして扱えるようにする。
//!
//! **内部クレート限定の公開範囲**: `onnx-interop` は crates.io 公開クレート
//! （公開名 `fandhe-ai-onnx-interop`）のため本モジュールも `pub mod` として
//! 公開されるが、facade（唯一のサポート対象公開面。`docs/compat-api-scope.md`
//! §0）は再エクスポートしない——`fandhe_ai_autodiff::Tape::custom`
//! （#1946・`docs/autodiff-custom-function-decision.md` §12.5 (a)）と同じ
//! 位置づけ。facade 公開の可否・API 形式はユーザー承認事項（PR 本文参照）。
//!
//! ## スコープ（実装時に確定。計画からの縮小）
//!
//! 22 op のうち以下 11 op を **勾配追跡対象（`Var` 経路）** として実装する
//! （forward は必ず [`crate::ops`] の同一関数を呼ぶため `interp::run` と
//! bit 完全一致し、backward は解析的勾配を手書きする。REQ-2 の統一複合判定
//! で検証する）:
//! `Gemm`・`MatMul`・`Add`・`Mul`・`Div`・`Sqrt`・`Relu`・`Sigmoid`・`Erf`・
//! `Softmax`・`LayerNormalization`。
//!
//! 残り 11 op のうち `Shape`・`Cast`・`Constant` は非勾配（メタデータ／定数）
//! 経路として実装する（`Cast` のみ `Var` 入力を許し、実体化して非 F32 へ
//! 変換する——ここでのみ意図的に勾配を切断する。承認事項 4）。
//!
//! `Gather`・`Unsqueeze`・`Concat`・`Slice`・`Mod`・`Reshape`・`Squeeze`・
//! `Transpose`（8 op）は本スコープでは未実装で、これらのノードに到達すると
//! [`AutogradError::UnsupportedInAutograd`] で fail-closed に拒否する
//! （no-silent-skip 契約。`.claude/rules/coding-rust.md`）。view／shape 系
//! 演算の勾配対応は後続スコープとして PR 本文に記録する
//! （`.claude/rules/out-of-scope-tracking.md`）。

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use fandhe_ai_autodiff::{AutodiffError, CustomFunction, Tape, Var};
use fandhe_ai_tensor_core::{ShapeError, Tensor, broadcast_shape};

use super::graph::{Graph, RawTensor};
use super::interp::{InterpError, Value};
use super::proto::NodeProto;
use crate::ops::{self, ConstantValue, GemmAttrs, LayerNormAttrs, OpError};

// ================= エラー型 =================

/// `BoundGraph` の実行時エラー。`#[non_exhaustive]`: [`InterpError`]／
/// [`OpError`] と同じ理由（公開 API 非破壊。後続 op 追加に備える）。
#[non_exhaustive]
#[derive(Debug)]
pub enum AutogradError {
    /// 非勾配経路（`Shape`／`Cast`／`Constant`）の実行が
    /// [`super::interp::InterpError`] 相当のエラーで失敗した。
    Interp(InterpError),
    /// `fandhe_ai_autodiff`（`Tape::custom`／`Var::*`）が返した型付きエラー。
    Autodiff(AutodiffError),
    /// `Tensor::*` の shape 不整合（`ops::*` を経由しない本モジュール直書きの
    /// 勾配計算コード自身が起こしたもの）。
    Shape(ShapeError),
    /// `ops::*`（forward 計算）が返したオペ固有エラー。
    Op(OpError),
    /// 本スコープでは勾配追跡経路を実装していない op へ到達した
    /// （`node`／`op_type` に加え、勾配追跡なし〈`Const`〉入力のみなら
    /// 実行できる可能性がある旨を `reason` で示す）。
    UnsupportedInAutograd {
        node: String,
        op_type: String,
        reason: &'static str,
    },
    /// [`BoundGraph::forward`] は「initializer を除く非追跡入力がちょうど
    /// 1 個・グラフ出力が 1 個」のグラフにのみ使える便宜 API。
    NotSingleInputOutput { inputs: usize, outputs: usize },
    /// feed／出力の [`AutogradValue`] variant が期待と異なる（例: `Var` を
    /// 期待する演算に `Const` 非 F32 値が渡された）。
    FeedTypeMismatch {
        name: String,
        expected: &'static str,
    },
    /// `graph.inputs` のうち initializer を持たない入力に feed が渡されな
    /// かった（[`super::interp::run`] と同じ検証順序・同じ契約）。
    MissingFeed { input: String },
    /// `run` に渡された feed 名がグラフ入力にも initializer にも属さない。
    UnknownFeed { name: String },
}

impl fmt::Display for AutogradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutogradError::Interp(e) => write!(f, "{e}"),
            AutogradError::Autodiff(e) => write!(f, "{e}"),
            AutogradError::Shape(e) => write!(f, "{e}"),
            AutogradError::Op(e) => write!(f, "{e}"),
            AutogradError::UnsupportedInAutograd {
                node,
                op_type,
                reason,
            } => write!(
                f,
                "ノード '{node}'（op_type={op_type}）は autograd 経路未対応: {reason}"
            ),
            AutogradError::NotSingleInputOutput { inputs, outputs } => write!(
                f,
                "BoundGraph::forward は単一入出力グラフ専用（非 initializer 入力 {inputs} 個・出力 {outputs} 個）"
            ),
            AutogradError::FeedTypeMismatch { name, expected } => {
                write!(f, "'{name}': 型不一致（期待: {expected}）")
            }
            AutogradError::MissingFeed { input } => {
                write!(f, "グラフ入力 '{input}' に対応する feed がありません")
            }
            AutogradError::UnknownFeed { name } => {
                write!(
                    f,
                    "feed '{name}' はグラフ入力にも initializer にも属しません"
                )
            }
        }
    }
}

impl std::error::Error for AutogradError {}

impl From<InterpError> for AutogradError {
    fn from(e: InterpError) -> Self {
        AutogradError::Interp(e)
    }
}

impl From<AutodiffError> for AutogradError {
    fn from(e: AutodiffError) -> Self {
        AutogradError::Autodiff(e)
    }
}

impl From<ShapeError> for AutogradError {
    fn from(e: ShapeError) -> Self {
        AutogradError::Shape(e)
    }
}

impl From<OpError> for AutogradError {
    fn from(e: OpError) -> Self {
        AutogradError::Op(e)
    }
}

impl From<super::graph::GraphError> for AutogradError {
    fn from(e: super::graph::GraphError) -> Self {
        AutogradError::Interp(InterpError::from(e))
    }
}

/// [`CustomFunction::forward`]／`backward` が返す [`AutodiffError`] は
/// `OpError` を運べないため、文字列化して `InvalidArgument` へ詰める。
fn op_err_to_autodiff(e: OpError) -> AutodiffError {
    AutodiffError::InvalidArgument(format!("{e}"))
}

// ================= 値モデル =================

/// `BoundGraph` の feed／出力／中間値が取りうる 2 態。`Var` は勾配追跡対象の
/// F32 テンソル、`Const` は非追跡値（[`super::interp::Value`] をそのまま
/// 再利用。任意 dtype）。
#[derive(Clone, Debug)]
pub enum AutogradValue<'t> {
    Var(Var<'t>),
    Const(Value),
}

/// [`BoundGraph::bind`] のオプション。`trainable` が `None` の場合は
/// 全 F32 initializer を勾配追跡対象（`requires_grad = true`）として葉化する
/// （PyTorch の `requires_grad=True` 既定と同じ「明示的に凍結したものだけ
/// 除外する」設計）。`Some(names)` の場合は `names` に含まれる initializer
/// のみ勾配追跡し、それ以外は [`fandhe_ai_autodiff::Tape::var_no_grad`] で
/// 凍結する。
#[non_exhaustive]
#[derive(Default)]
pub struct BindOptions {
    pub trainable: Option<HashSet<String>>,
}

impl BindOptions {
    /// `trainable` を指定して構築する（`#[non_exhaustive]` のため、クレート外から
    /// フィールドを直接列挙するリテラル構築ができない。この構築子経由で作る）。
    pub fn with_trainable(trainable: HashSet<String>) -> Self {
        BindOptions {
            trainable: Some(trainable),
        }
    }
}

/// import した ONNX [`Graph`] を `tape` 上へ束縛した実行単位。
///
/// `bind` 時点で全 F32 initializer を一度だけ `Tape` の葉ノードへ変換する
/// （[`Tape::var`]／[`Tape::var_no_grad`]。二重に葉化しない）。非 F32
/// initializer（shape 定数等）はそのまま [`Value`] として保持する。
pub struct BoundGraph<'g, 't> {
    graph: &'g Graph,
    tape: &'t Tape,
    init: HashMap<String, AutogradValue<'t>>,
    params: HashMap<String, Var<'t>>,
}

impl<'g, 't> BoundGraph<'g, 't> {
    /// `graph` の initializer を `tape` へ葉化して束縛する。
    pub fn bind(
        graph: &'g Graph,
        tape: &'t Tape,
        options: &BindOptions,
    ) -> Result<Self, AutogradError> {
        let mut init = HashMap::with_capacity(graph.initializers.len());
        let mut params = HashMap::new();
        for (name, raw) in &graph.initializers {
            let value = raw_to_value(raw)?;
            match value {
                Value::F32(t) => {
                    let trainable = options
                        .trainable
                        .as_ref()
                        .map(|set| set.contains(name))
                        .unwrap_or(true);
                    let v = if trainable {
                        tape.var(&t)
                    } else {
                        tape.var_no_grad(&t)
                    };
                    if trainable {
                        params.insert(name.clone(), v);
                    }
                    init.insert(name.clone(), AutogradValue::Var(v));
                }
                other => {
                    init.insert(name.clone(), AutogradValue::Const(other));
                }
            }
        }
        Ok(BoundGraph {
            graph,
            tape,
            init,
            params,
        })
    }

    /// 勾配追跡対象（`trainable`）の initializer 一覧。
    pub fn params(&self) -> &HashMap<String, Var<'t>> {
        &self.params
    }

    /// 名前で 1 個だけ取り出す便宜アクセサ。
    pub fn param(&self, name: &str) -> Option<Var<'t>> {
        self.params.get(name).copied()
    }

    /// グラフを実行する。`feeds` の検証順序は [`super::interp::run`] と同一
    /// （1. `MissingFeed` → 2. `UnknownFeed`）。initializer と同名の feed は
    /// initializer を上書きする（ONNX のデフォルト値セマンティクス。
    /// `interp::run` doc 参照）。
    pub fn run(
        &self,
        feeds: HashMap<String, AutogradValue<'t>>,
    ) -> Result<HashMap<String, AutogradValue<'t>>, AutogradError> {
        for input in &self.graph.inputs {
            if !self.graph.initializers.contains_key(input) && !feeds.contains_key(input) {
                return Err(AutogradError::MissingFeed {
                    input: input.clone(),
                });
            }
        }
        let input_set: HashSet<&str> = self.graph.inputs.iter().map(String::as_str).collect();
        for name in feeds.keys() {
            if !input_set.contains(name.as_str()) && !self.graph.initializers.contains_key(name) {
                return Err(AutogradError::UnknownFeed { name: name.clone() });
            }
        }

        let mut env: HashMap<String, AutogradValue<'t>> =
            HashMap::with_capacity(self.init.len() + feeds.len());
        for (k, v) in &self.init {
            env.insert(k.clone(), v.clone());
        }
        for (k, v) in feeds {
            env.insert(k, v);
        }

        for node in &self.graph.nodes {
            let out_value = dispatch_node(self.tape, &env, node)?;
            let out_name = require_single_output(node)?.to_string();
            env.insert(out_name, out_value);
        }

        let mut result = HashMap::with_capacity(self.graph.outputs.len());
        for name in &self.graph.outputs {
            let v = env
                .get(name)
                .cloned()
                .ok_or_else(|| InterpError::GraphOutputNotProduced { name: name.clone() })?;
            result.insert(name.clone(), v);
        }
        Ok(result)
    }

    /// 「initializer を除く非追跡入力がちょうど 1 個・出力が 1 個」の
    /// グラフに限り使える便宜 API（`nn::Linear::bind` 的な単純さのための
    /// ショートカット）。入出力が単一でないグラフは
    /// [`AutogradError::NotSingleInputOutput`] で拒否する。
    pub fn forward(&self, input: &Var<'t>) -> Result<Var<'t>, AutogradError> {
        let non_init_inputs: Vec<&String> = self
            .graph
            .inputs
            .iter()
            .filter(|n| !self.graph.initializers.contains_key(n.as_str()))
            .collect();
        if non_init_inputs.len() != 1 || self.graph.outputs.len() != 1 {
            return Err(AutogradError::NotSingleInputOutput {
                inputs: non_init_inputs.len(),
                outputs: self.graph.outputs.len(),
            });
        }
        let mut feeds = HashMap::new();
        feeds.insert(non_init_inputs[0].clone(), AutogradValue::Var(*input));
        let out_name = self.graph.outputs[0].clone();
        let mut out = self.run(feeds)?;
        match out.remove(&out_name) {
            Some(AutogradValue::Var(v)) => Ok(v),
            Some(AutogradValue::Const(_)) => Err(AutogradError::FeedTypeMismatch {
                name: out_name,
                expected: "Var（F32・勾配追跡対象）出力",
            }),
            None => Err(InterpError::GraphOutputNotProduced { name: out_name }.into()),
        }
    }
}

/// [`super::graph::RawTensor`]（initializer の復号結果）を [`Value`] へ変換
/// する。`interp::raw_to_value`（module-private）とロジックは同一だが、
/// 本モジュールから private 関数を呼べないため小さく複製する（interp.rs 自体は
/// 変更しない方針。#2078 実装計画からの縮小: `ValueLookup` 汎用化リファクタは
/// 見送り、既存の安定した interp.rs に触れず新モジュールのみで完結させた）。
fn raw_to_value(raw: &RawTensor) -> Result<Value, AutogradError> {
    let (shape_i64, value): (&[i64], Value) = match raw {
        RawTensor::F32 { data, shape } => (
            shape,
            Value::F32(Tensor::new(data.clone(), &to_usize_shape(shape))?),
        ),
        RawTensor::I64 { data, shape } => (
            shape,
            Value::I64(Tensor::new(data.clone(), &to_usize_shape(shape))?),
        ),
        RawTensor::Bool { data, shape } => (
            shape,
            Value::Bool(Tensor::new(data.clone(), &to_usize_shape(shape))?),
        ),
        RawTensor::F16 { data, shape } => (
            shape,
            Value::F16(Tensor::new(data.clone(), &to_usize_shape(shape))?),
        ),
    };
    let _ = shape_i64; // shape は Tensor::new 内で再利用済み（可読性のため変数だけ残す）
    Ok(value)
}

fn to_usize_shape(shape: &[i64]) -> Vec<usize> {
    shape.iter().map(|&d| d as usize).collect()
}

fn require_single_output(node: &NodeProto) -> Result<&str, InterpError> {
    if node.output.len() != 1 {
        return Err(InterpError::OutputArityMismatch {
            node: node.name.clone(),
            expected: 1,
            actual: node.output.len(),
        });
    }
    Ok(node.output[0].as_str())
}

fn input_name(node: &NodeProto, idx: usize) -> Result<&str, InterpError> {
    node.input
        .get(idx)
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| InterpError::MissingInput {
            node: node.name.clone(),
            input: format!("<input[{idx}]>"),
        })
}

fn get_env<'a, 't>(
    env: &'a HashMap<String, AutogradValue<'t>>,
    node: &NodeProto,
    name: &str,
) -> Result<&'a AutogradValue<'t>, InterpError> {
    env.get(name).ok_or_else(|| InterpError::MissingInput {
        node: node.name.clone(),
        input: name.to_string(),
    })
}

fn attr_f32(node: &NodeProto, name: &str, default: f32) -> f32 {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.f)
        .unwrap_or(default)
}

fn attr_i64(node: &NodeProto, name: &str, default: i64) -> i64 {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.i)
        .unwrap_or(default)
}

// ================= 共通ヘルパー（backward で使う） =================

/// 行優先（row-major）ストライドを計算する。
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// broadcast 前提で拡大された勾配 `grad` を `target_shape` へ縮約する
/// （broadcast VJP の標準パターン: 追加された先頭軸・サイズ 1 へ縮小された軸を
/// 総和する）。長軸縮約は `f64` アキュムレータで統一する
/// （`.claude/rules/coding-rust.md`。最終 1 回のみ `f32` へ downcast）。
fn reduce_to_shape(
    grad: &Tensor<f32>,
    target_shape: &[usize],
) -> Result<Tensor<f32>, AutogradError> {
    let g = grad.contiguous();
    if g.shape() == target_shape {
        return Ok(g);
    }
    let g_shape = g.shape().to_vec();
    let rank_diff = g_shape.len().saturating_sub(target_shape.len());
    let mut padded_target = vec![1usize; rank_diff];
    padded_target.extend_from_slice(target_shape);

    let out_numel: usize = padded_target.iter().product();
    let mut out_f64 = vec![0f64; out_numel];
    let g_strides = row_major_strides(&g_shape);
    let out_strides = row_major_strides(&padded_target);
    let g_slice = g
        .as_slice()
        .ok_or_else(|| AutogradError::from(OpError::NonContiguousInternal("reduce_to_shape")))?;

    for (flat_idx, &v) in g_slice.iter().enumerate() {
        let mut rem = flat_idx;
        let mut out_offset = 0usize;
        for axis in 0..g_shape.len() {
            let stride = g_strides[axis];
            let coord = rem.checked_div(stride).unwrap_or(0);
            if let Some(r) = rem.checked_rem(stride) {
                rem = r;
            }
            let out_coord = if padded_target[axis] == 1 { 0 } else { coord };
            out_offset += out_coord * out_strides[axis];
        }
        out_f64[out_offset] += v as f64;
    }
    let out_f32: Vec<f32> = out_f64.iter().map(|&d| d as f32).collect();
    let out_tensor = Tensor::new(out_f32, &padded_target)?;
    Ok(out_tensor.reshape(target_shape)?)
}

fn scale(t: &Tensor<f32>, s: f32) -> Result<Tensor<f32>, AutogradError> {
    let tc = t.contiguous();
    let slice = tc
        .as_slice()
        .ok_or_else(|| AutogradError::from(OpError::NonContiguousInternal("scale")))?;
    let data: Vec<f32> = slice.iter().map(|&v| v * s).collect();
    Ok(Tensor::new(data, tc.shape())?)
}

/// `env` 上の値を「勾配追跡対象の f32 `Var`」として取り出す。`Const(Value::F32)`
/// が渡された場合は `Tape::var_no_grad` でその場だけ葉化する（この co-input
/// からは勾配が流れない。値そのものは forward に正しく参加する）。非 F32
/// `Const` が渡された場合は型不一致として拒否する。
fn as_var<'t>(
    tape: &'t Tape,
    env: &HashMap<String, AutogradValue<'t>>,
    node: &NodeProto,
    name: &str,
) -> Result<Var<'t>, AutogradError> {
    match get_env(env, node, name)? {
        AutogradValue::Var(v) => Ok(*v),
        AutogradValue::Const(Value::F32(t)) => Ok(tape.var_no_grad(t)),
        AutogradValue::Const(_) => Err(AutogradError::FeedTypeMismatch {
            name: name.to_string(),
            expected: "f32",
        }),
    }
}

// ================= CustomFunction 実装（勾配追跡対象 11 op） =================

/// 1 入力要素ごと演算（`ops::sqrt`／`ops::relu`／`ops::sigmoid`／`ops::erf`）の
/// forward 関数ポインタ型。dispatch 側の match 式が複雑な型リテラルを持たない
/// よう `UnaryElemFn`／`BinaryElemFn` のフィールド・タプル型として共有する。
type UnaryForwardFn = fn(&Tensor<f32>) -> Result<Tensor<f32>, OpError>;
/// 2 入力要素ごと演算（`ops::add`／`ops::mul`／`ops::div`）の forward 関数
/// ポインタ型。
type BinaryForwardFn = fn(&Tensor<f32>, &Tensor<f32>) -> Result<Tensor<f32>, OpError>;

/// `Relu`／`Sigmoid`／`Sqrt`／`Erf` 共通の要素ごと 1 入力演算。forward は
/// `ops::*` をそのまま呼ぶため `interp::run` と bit 完全一致する。backward は
/// `grad_fn(x, y)`（`x`: 入力要素・`y`: forward 出力要素）が返す局所導関数値に
/// 上流勾配を乗じる。
struct UnaryElemFn {
    op_name: &'static str,
    forward_fn: UnaryForwardFn,
    grad_fn: fn(f32, f32) -> f32,
}

impl CustomFunction for UnaryElemFn {
    fn name(&self) -> &str {
        self.op_name
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        (self.forward_fn)(inputs[0]).map_err(op_err_to_autodiff)
    }
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        if !requires_grad[0] {
            return Ok(vec![None]);
        }
        let x = inputs[0].contiguous();
        let y = out_value.contiguous();
        let g = upstream.contiguous();
        let (xs, ys, gs) = (
            x.as_slice()
                .ok_or_else(|| op_err_to_autodiff(OpError::NonContiguousInternal(self.op_name)))?,
            y.as_slice()
                .ok_or_else(|| op_err_to_autodiff(OpError::NonContiguousInternal(self.op_name)))?,
            g.as_slice()
                .ok_or_else(|| op_err_to_autodiff(OpError::NonContiguousInternal(self.op_name)))?,
        );
        let data: Vec<f32> = xs
            .iter()
            .zip(ys.iter())
            .zip(gs.iter())
            .map(|((&xv, &yv), &gv)| gv * (self.grad_fn)(xv, yv))
            .collect();
        let dx = Tensor::new(data, x.shape()).map_err(AutodiffError::Shape)?;
        Ok(vec![Some(dx)])
    }
}

fn relu_grad(x: f32, _y: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}
fn sigmoid_grad(_x: f32, y: f32) -> f32 {
    y * (1.0 - y)
}
fn sqrt_grad(_x: f32, y: f32) -> f32 {
    0.5 / y
}
fn erf_grad(x: f32, _y: f32) -> f32 {
    (2.0 / std::f32::consts::PI.sqrt()) * (-x * x).exp()
}

/// `Add`／`Mul`／`Div` 共通の broadcast 対応 2 項演算。forward は `ops::*` を
/// そのまま呼ぶため bit 完全一致。backward は各要素の局所勾配を broadcast 後の
/// shape で計算してから [`reduce_to_shape`] で入力 shape へ縮約する。
struct BinaryElemFn {
    op_name: &'static str,
    forward_fn: BinaryForwardFn,
    kind: BinaryKind,
}

#[derive(Clone, Copy)]
enum BinaryKind {
    Add,
    Mul,
    Div,
}

impl CustomFunction for BinaryElemFn {
    fn name(&self) -> &str {
        self.op_name
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        broadcast_shape(input_shapes[0], input_shapes[1]).map_err(AutodiffError::Shape)
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        (self.forward_fn)(inputs[0], inputs[1]).map_err(op_err_to_autodiff)
    }
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        let a = inputs[0];
        let b = inputs[1];
        let out_shape = upstream.shape().to_vec();
        let a_full = a
            .broadcast_to(&out_shape)
            .map_err(AutodiffError::Shape)?
            .contiguous();
        let b_full = b
            .broadcast_to(&out_shape)
            .map_err(AutodiffError::Shape)?
            .contiguous();
        let g = upstream.contiguous();
        let (as_, bs_, gs) = (
            a_full
                .as_slice()
                .ok_or_else(|| op_err_to_autodiff(OpError::NonContiguousInternal(self.op_name)))?,
            b_full
                .as_slice()
                .ok_or_else(|| op_err_to_autodiff(OpError::NonContiguousInternal(self.op_name)))?,
            g.as_slice()
                .ok_or_else(|| op_err_to_autodiff(OpError::NonContiguousInternal(self.op_name)))?,
        );
        let mut da_local: Option<Vec<f32>> = if requires_grad[0] {
            Some(vec![0.0; gs.len()])
        } else {
            None
        };
        let mut db_local: Option<Vec<f32>> = if requires_grad[1] {
            Some(vec![0.0; gs.len()])
        } else {
            None
        };
        for i in 0..gs.len() {
            let (av, bv, gv) = (as_[i], bs_[i], gs[i]);
            match self.kind {
                BinaryKind::Add => {
                    if let Some(d) = da_local.as_mut() {
                        d[i] = gv;
                    }
                    if let Some(d) = db_local.as_mut() {
                        d[i] = gv;
                    }
                }
                BinaryKind::Mul => {
                    if let Some(d) = da_local.as_mut() {
                        d[i] = gv * bv;
                    }
                    if let Some(d) = db_local.as_mut() {
                        d[i] = gv * av;
                    }
                }
                BinaryKind::Div => {
                    if let Some(d) = da_local.as_mut() {
                        d[i] = gv / bv;
                    }
                    if let Some(d) = db_local.as_mut() {
                        d[i] = -gv * av / (bv * bv);
                    }
                }
            }
        }
        let da = match da_local {
            Some(data) => {
                let full = Tensor::new(data, &out_shape).map_err(AutodiffError::Shape)?;
                Some(
                    reduce_to_shape(&full, a.shape())
                        .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))?,
                )
            }
            None => None,
        };
        let db = match db_local {
            Some(data) => {
                let full = Tensor::new(data, &out_shape).map_err(AutodiffError::Shape)?;
                Some(
                    reduce_to_shape(&full, b.shape())
                        .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))?,
                )
            }
            None => None,
        };
        Ok(vec![da, db])
    }
}

/// `Gemm(A,B,C?) = alpha * A' @ B' + beta * C`（`A'`/`B'` は `trans_a`/`trans_b`
/// 適用後）。forward は `ops::gemm` をそのまま呼ぶため bit 完全一致。backward は
/// 標準的な行列積 VJP を `ops::matmul`／`Tensor::permute` で組み立てる（REQ-2
/// 統一複合判定で検証・bit 一致は要求しない）。
struct GemmFn {
    attrs: GemmAttrs,
    has_c: bool,
}

impl CustomFunction for GemmFn {
    fn name(&self) -> &str {
        "Gemm"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        let a = input_shapes[0];
        let b = input_shapes[1];
        if a.len() != 2 || b.len() != 2 {
            return Err(AutodiffError::InvalidArgument(
                "Gemm(autograd): A/B は 2 次元のみ対応".into(),
            ));
        }
        let m = if self.attrs.trans_a { a[1] } else { a[0] };
        let n = if self.attrs.trans_b { b[0] } else { b[1] };
        Ok(vec![m, n])
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        let c = if self.has_c { Some(inputs[2]) } else { None };
        ops::gemm(inputs[0], inputs[1], c, &self.attrs).map_err(op_err_to_autodiff)
    }
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        let a = inputs[0];
        let b = inputs[1];
        let a_eff = if self.attrs.trans_a {
            a.permute(&[1, 0]).map_err(AutodiffError::Shape)?
        } else {
            a.clone()
        };
        let b_eff = if self.attrs.trans_b {
            b.permute(&[1, 0]).map_err(AutodiffError::Shape)?
        } else {
            b.clone()
        };
        let dy = upstream;

        let mut out = Vec::with_capacity(inputs.len());
        if requires_grad[0] {
            let b_eff_t = b_eff.permute(&[1, 0]).map_err(AutodiffError::Shape)?;
            let d_a_eff = ops::matmul(dy, &b_eff_t).map_err(op_err_to_autodiff)?;
            let d_a_eff = scale(&d_a_eff, self.attrs.alpha)
                .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))?;
            let da = if self.attrs.trans_a {
                d_a_eff.permute(&[1, 0]).map_err(AutodiffError::Shape)?
            } else {
                d_a_eff
            };
            out.push(Some(da));
        } else {
            out.push(None);
        }
        if requires_grad[1] {
            let a_eff_t = a_eff.permute(&[1, 0]).map_err(AutodiffError::Shape)?;
            let d_b_eff = ops::matmul(&a_eff_t, dy).map_err(op_err_to_autodiff)?;
            let d_b_eff = scale(&d_b_eff, self.attrs.alpha)
                .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))?;
            let db = if self.attrs.trans_b {
                d_b_eff.permute(&[1, 0]).map_err(AutodiffError::Shape)?
            } else {
                d_b_eff
            };
            out.push(Some(db));
        } else {
            out.push(None);
        }
        if self.has_c {
            if requires_grad.get(2).copied().unwrap_or(false) {
                let dy_beta = scale(dy, self.attrs.beta)
                    .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))?;
                let dc = reduce_to_shape(&dy_beta, inputs[2].shape())
                    .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))?;
                out.push(Some(dc));
            } else {
                out.push(None);
            }
        }
        Ok(out)
    }
}

/// `MatMul`（`numpy.matmul` バッチ行列積）。rank 2 以上のみ対応する（1 次元
/// ベクトル入力は本スコープ対象外。`output_shape` で拒否）。forward は
/// `ops::matmul` をそのまま呼ぶため bit 完全一致。
struct MatMulFn;

fn matmul_batch_out_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, AutodiffError> {
    if a.len() < 2 || b.len() < 2 {
        return Err(AutodiffError::InvalidArgument(
            "MatMul(autograd): rank 1 のベクトル入力は本スコープ未対応".into(),
        ));
    }
    let a_batch = &a[..a.len() - 2];
    let b_batch = &b[..b.len() - 2];
    let mut batch = broadcast_shape(a_batch, b_batch).map_err(AutodiffError::Shape)?;
    let (m, k) = (a[a.len() - 2], a[a.len() - 1]);
    let (k2, n) = (b[b.len() - 2], b[b.len() - 1]);
    if k != k2 {
        return Err(AutodiffError::InvalidArgument(format!(
            "MatMul(autograd): 内部次元不一致 k={k} k2={k2}"
        )));
    }
    batch.push(m);
    batch.push(n);
    Ok(batch)
}

impl CustomFunction for MatMulFn {
    fn name(&self) -> &str {
        "MatMul"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        matmul_batch_out_shape(input_shapes[0], input_shapes[1])
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        ops::matmul(inputs[0], inputs[1]).map_err(op_err_to_autodiff)
    }
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        let a = inputs[0];
        let b = inputs[1];
        let rank_a = a.rank();
        let rank_b = b.rank();
        let mut perm_a: Vec<usize> = (0..rank_a).collect();
        perm_a.swap(rank_a - 2, rank_a - 1);
        let mut perm_b: Vec<usize> = (0..rank_b).collect();
        perm_b.swap(rank_b - 2, rank_b - 1);

        let mut out = Vec::with_capacity(2);
        if requires_grad[0] {
            let b_t = b.permute(&perm_b).map_err(AutodiffError::Shape)?;
            let da_full = ops::matmul(upstream, &b_t).map_err(op_err_to_autodiff)?;
            let da = reduce_to_shape(&da_full, a.shape())
                .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))?;
            out.push(Some(da));
        } else {
            out.push(None);
        }
        if requires_grad[1] {
            let a_t = a.permute(&perm_a).map_err(AutodiffError::Shape)?;
            let db_full = ops::matmul(&a_t, upstream).map_err(op_err_to_autodiff)?;
            let db = reduce_to_shape(&db_full, b.shape())
                .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))?;
            out.push(Some(db));
        } else {
            out.push(None);
        }
        Ok(out)
    }
}

/// `Softmax(x, axis)`。`dx = y ⊙ (g − sum_axis(g ⊙ y))`。
struct SoftmaxFn {
    axis: i64,
}

impl CustomFunction for SoftmaxFn {
    fn name(&self) -> &str {
        "Softmax"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        ops::softmax(inputs[0], self.axis).map_err(op_err_to_autodiff)
    }
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        if !requires_grad[0] {
            return Ok(vec![None]);
        }
        let rank = inputs[0].rank();
        let axis = ops::normalize_axis(self.axis, rank).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "Softmax(autograd): axis={} が rank={rank} の範囲外",
                self.axis
            ))
        })?;
        let shape = out_value.shape().to_vec();
        let inner = shape[axis];
        let outer: usize = shape[..axis].iter().product();
        let trailing: usize = shape[axis + 1..].iter().product();

        let yc = out_value.contiguous();
        let gc = upstream.contiguous();
        let ys = yc
            .as_slice()
            .ok_or_else(|| op_err_to_autodiff(OpError::NonContiguousInternal("Softmax")))?;
        let gs = gc
            .as_slice()
            .ok_or_else(|| op_err_to_autodiff(OpError::NonContiguousInternal("Softmax")))?;

        let mut dx = vec![0f32; ys.len()];
        // `outer`／`trailing` は shape の積であり、`axis` 自体のサイズ（`inner`）が
        // 0 でも 0 にならない（例: shape=[usize::MAX, 0], axis=1 は
        // outer=usize::MAX・inner=0）。forward（`ops::softmax`）は
        // `slice.is_empty()` で早期リターンして巨大 outer 反復のハングを防いで
        // おり（`ops/softmax.rs` の PR #276 Bugbot 指摘コメント参照）、backward
        // にも同じガードが必要（レビュー指摘: PR #2223 Cursor Bugbot
        // discussion_r4072661705）。`ys`/`gs` は forward 出力・upstream と同じ
        // 要素数のため `ys.is_empty()` で判定できる。
        if ys.is_empty() {
            let dx_t = Tensor::new(dx, &shape).map_err(AutodiffError::Shape)?;
            return Ok(vec![Some(dx_t)]);
        }
        for o in 0..outer {
            for t in 0..trailing {
                // `sum_axis(g ⊙ y)` は f64 で蓄積する（長軸縮約の一般方針。
                // `.claude/rules/coding-rust.md`）。要素積 `g[idx] * y[idx]` は
                // 規定の丸め契約（要素積は f32 で確定してから f64 へ昇格して
                // 蓄積する）に従い、まず f32 で積を確定してから f64 へ昇格する
                // （レビュー指摘: PR #2223 codex-review discussion_r4072652847）。
                let mut dot: f64 = 0.0;
                for a in 0..inner {
                    let idx = (o * inner + a) * trailing + t;
                    let prod = gs[idx] * ys[idx];
                    dot += prod as f64;
                }
                // `dot`（f64）を乗算前に `f32` へ downcast すると、
                // `y * (g - dot)` が有限の `f32` 入力でも overflow しうる
                // （`grad::softmax_vjp_along` と同じ懸念。レビュー指摘:
                // PR #2223 Cursor Bugbot discussion_r4072661711）。
                // `autodiff::grad::softmax_vjp_along` と同じ f64 契約
                // （`.claude/rules/coding-rust.md`）に従い、`g` からの
                // 減算・`y` との最終乗算まで f64 で保持し、最終書き出しで
                // のみ `f32` へ downcast する。
                for a in 0..inner {
                    let idx = (o * inner + a) * trailing + t;
                    let d = (ys[idx] as f64) * (gs[idx] as f64 - dot);
                    dx[idx] = d as f32;
                }
            }
        }
        let dx_t = Tensor::new(dx, &shape).map_err(AutodiffError::Shape)?;
        Ok(vec![Some(dx_t)])
    }
}

/// `LayerNormalization(x, scale, bias?)`（trailing `axis..` を正規化集合とする）。
///
/// **正規化統計（平均・分散・逆標準偏差）は forward（[`ops::layer_normalization`]）
/// と同じ `f32` 演算（`mean = sum * inv_n`・分散の二乗差累積は `f32::mul_add`）を
/// bit 完全一致で再現する**。forward 自体は `f32` 単一精度で統計を計算する実装
/// （`ops/layer_norm.rs`。PR #277 で確定・本 PR〈#2078〉のスコープ外）のため、
/// backward が独自に `f64` で統計を再計算すると forward が実際に計算した関数とは
/// 異なる関数を微分してしまい丸め差のある入力で `dx`／`dscale` が不正確になる
/// （レビュー指摘: PR #2223 codex-review discussion_r4072652835）。forward 自体の
/// 統計を `f64` 契約へ揃える変更は既存関数（`ops/layer_norm.rs`）への影響が
/// 本 PR のスコープを超えるため採らない。
///
/// 一方、`dxhat` の行方向縮約（`mean_dxhat`／`mean_dxhat_xhat`）は「勾配の長軸
/// 縮約」（`.claude/rules/coding-rust.md`）に該当し、要素積を `f32` で確定して
/// から `f64` へ昇格して蓄積する契約に従う。縮約結果を使う `dx` の最終合成
/// （`rstd * (dxhat − mean_dxhat − xhat ⊙ mean_dxhat_xhat)`）も
/// `autodiff::grad::softmax_vjp_along` と同じ f64 契約に従い、`f32` への
/// downcast は最終書き出しの 1 回のみに限る（早期 downcast は有限の `f32`
/// 入力でも overflow しうる。レビュー指摘: PR #2223 Cursor Bugbot
/// discussion_r4072661711）。
struct LayerNormFn {
    axis: i64,
    epsilon: f32,
    has_bias: bool,
}

impl CustomFunction for LayerNormFn {
    fn name(&self) -> &str {
        "LayerNormalization"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        let bias = if self.has_bias { Some(inputs[2]) } else { None };
        ops::layer_normalization(
            inputs[0],
            inputs[1],
            bias,
            &LayerNormAttrs {
                axis: self.axis,
                epsilon: self.epsilon,
            },
        )
        .map_err(op_err_to_autodiff)
    }
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        let x = inputs[0];
        let scale_t = inputs[1];
        let rank = x.rank();
        let axis = ops::normalize_axis(self.axis, rank).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "LayerNorm(autograd): axis={} が rank={rank} の範囲外",
                self.axis
            ))
        })?;
        let normalized_shape = &x.shape()[axis..];
        let inner: usize = normalized_shape.iter().product();
        let outer: usize = x.shape()[..axis].iter().product();

        let xc = x.contiguous();
        let gc = upstream.contiguous();
        let scale_b = scale_t
            .broadcast_to(normalized_shape)
            .map_err(AutodiffError::Shape)?
            .contiguous();
        let xs = xc.as_slice().ok_or_else(|| {
            op_err_to_autodiff(OpError::NonContiguousInternal("LayerNormalization"))
        })?;
        let gs = gc.as_slice().ok_or_else(|| {
            op_err_to_autodiff(OpError::NonContiguousInternal("LayerNormalization"))
        })?;
        let scale_s = scale_b.as_slice().ok_or_else(|| {
            op_err_to_autodiff(OpError::NonContiguousInternal("LayerNormalization"))
        })?;

        // 他の CustomFunction 実装（BinaryElemFn／GemmFn／MatMulFn）と同じ
        // 「requires_grad[i] が false の入力は計算自体を行わない」方針に揃える。
        // dx は dxhat／mean_dxhat／mean_dxhat_xhat（dscale とは独立の中間値）を要し、
        // dscale は xhat のみを要するため、必要フラグごとに計算を分岐する。
        let need_dx = requires_grad[0];
        let need_dscale = requires_grad[1];
        let need_dbias = self.has_bias && requires_grad.get(2).copied().unwrap_or(false);

        let mut dx = if need_dx {
            vec![0f32; xs.len()]
        } else {
            Vec::new()
        };
        let mut dscale_full = if need_dscale {
            vec![0f32; xs.len()]
        } else {
            Vec::new()
        };
        let mut dbias_full = if need_dbias {
            vec![0f32; xs.len()]
        } else {
            Vec::new()
        };

        if need_dx || need_dscale || need_dbias {
            let inv_n = 1.0f32 / inner as f32;
            for o in 0..outer {
                let row = &xs[o * inner..(o + 1) * inner];
                let grow = &gs[o * inner..(o + 1) * inner];
                // forward（`ops::layer_normalization`）と bit 完全一致する `f32`
                // 統計再計算（mean = sum * inv_n・分散の二乗差累積は
                // `f32::mul_add`）。上の struct doc コメント参照。
                let mean: f32 = row.iter().sum::<f32>() * inv_n;
                let mut sq_acc = 0f32;
                for &v in row {
                    let diff = v - mean;
                    sq_acc = diff.mul_add(diff, sq_acc);
                }
                let var = sq_acc * inv_n;
                let rstd = 1.0f32 / (var + self.epsilon).sqrt();

                // xhat は dx・dscale の双方が使うため need_dx || need_dscale のときのみ計算する。
                // forward の `normalized = (block[i] - mean) * inv_std` と同じ f32 演算。
                let xhat: Vec<f32> = if need_dx || need_dscale {
                    row.iter().map(|&v| (v - mean) * rstd).collect()
                } else {
                    Vec::new()
                };

                // dxhat・mean_dxhat・mean_dxhat_xhat は dx 専用の中間値。
                // 要素積（dy・scale／dxhat・xhat）はまず f32 で確定してから f64 の
                // 縮約アキュムレータへ昇格する（勾配の長軸縮約の丸め契約。
                // レビュー指摘: PR #2223 codex-review discussion_r4072652847）。
                let (mean_dxhat, mean_dxhat_xhat, dxhat) = if need_dx {
                    let dxhat: Vec<f32> = (0..inner).map(|i| grow[i] * scale_s[i]).collect();
                    let mean_dxhat: f64 =
                        dxhat.iter().map(|&d| d as f64).sum::<f64>() / inner as f64;
                    let mean_dxhat_xhat: f64 = dxhat
                        .iter()
                        .zip(xhat.iter())
                        .map(|(&d, &xh)| (d * xh) as f64)
                        .sum::<f64>()
                        / inner as f64;
                    (mean_dxhat, mean_dxhat_xhat, dxhat)
                } else {
                    (0.0, 0.0, Vec::new())
                };
                // 縮約結果（f64）を要素ごとの最終合成の前に `f32` へ downcast
                // すると、`rstd * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat)`
                // が有限の `f32` 入力でも overflow しうる（`SoftmaxFn::backward`
                // と同じ懸念。レビュー指摘: PR #2223 Cursor Bugbot
                // discussion_r4072661711）。`autodiff::grad::softmax_vjp_along`
                // と同じ f64 契約（`.claude/rules/coding-rust.md`）に従い、
                // `dxhat`・`xhat`・`rstd`（bit 完全一致契約の統計値そのものは
                // 不変。乗算のためだけに f64 へ widen する）との合成まで f64 で
                // 保持し、最終書き出しで 1 回だけ `f32` へ downcast する。

                for i in 0..inner {
                    let idx = o * inner + i;
                    if need_dx {
                        let d = (rstd as f64)
                            * (dxhat[i] as f64 - mean_dxhat - (xhat[i] as f64) * mean_dxhat_xhat);
                        dx[idx] = d as f32;
                    }
                    if need_dscale {
                        dscale_full[idx] = grow[i] * xhat[i];
                    }
                    if need_dbias {
                        dbias_full[idx] = grow[i];
                    }
                }
            }
        }

        let mut out: Vec<Option<Tensor<f32>>> = Vec::with_capacity(inputs.len());
        if need_dx {
            out.push(Some(
                Tensor::new(dx, x.shape()).map_err(AutodiffError::Shape)?,
            ));
        } else {
            out.push(None);
        }
        if need_dscale {
            let full = Tensor::new(dscale_full, x.shape()).map_err(AutodiffError::Shape)?;
            let d = reduce_to_shape(&full, scale_t.shape())
                .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))?;
            out.push(Some(d));
        } else {
            out.push(None);
        }
        if self.has_bias {
            if need_dbias {
                let full = Tensor::new(dbias_full, x.shape()).map_err(AutodiffError::Shape)?;
                let d = reduce_to_shape(&full, inputs[2].shape())
                    .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))?;
                out.push(Some(d));
            } else {
                out.push(None);
            }
        }
        Ok(out)
    }
}

// ================= ノードディスパッチ =================

fn unsupported(node: &NodeProto, reason: &'static str) -> AutogradError {
    AutogradError::UnsupportedInAutograd {
        node: node.name.clone(),
        op_type: node.op_type.clone(),
        reason,
    }
}

/// `env` から `Const` を要求する（`Var` が渡された場合は fail-closed）。
/// `Cast` 以外の非勾配経路 op（`Shape`／`Constant` を除く）が使う。
fn require_const<'a, 't>(
    env: &'a HashMap<String, AutogradValue<'t>>,
    node: &NodeProto,
    name: &str,
) -> Result<&'a Value, AutogradError> {
    match get_env(env, node, name)? {
        AutogradValue::Const(v) => Ok(v),
        AutogradValue::Var(_) => Err(unsupported(
            node,
            "この演算は Var（勾配追跡対象）入力に対応していない（#2078 スコープ外。\
             view／shape 系演算の勾配対応は後続イシューで追跡）",
        )),
    }
}

fn dispatch_node<'t>(
    tape: &'t Tape,
    env: &HashMap<String, AutogradValue<'t>>,
    node: &NodeProto,
) -> Result<AutogradValue<'t>, AutogradError> {
    match node.op_type.as_str() {
        "Gemm" => {
            let a = as_var(tape, env, node, input_name(node, 0)?)?;
            let b = as_var(tape, env, node, input_name(node, 1)?)?;
            let has_c = matches!(node.input.get(2), Some(n) if !n.is_empty());
            let attrs = GemmAttrs {
                alpha: attr_f32(node, "alpha", 1.0),
                beta: attr_f32(node, "beta", 1.0),
                trans_a: attr_i64(node, "transA", 0) != 0,
                trans_b: attr_i64(node, "transB", 0) != 0,
            };
            let mut inputs = vec![a, b];
            if has_c {
                inputs.push(as_var(tape, env, node, node.input[2].as_str())?);
            }
            let out = tape.custom(Arc::new(GemmFn { attrs, has_c }), &inputs)?;
            Ok(AutogradValue::Var(out))
        }
        "MatMul" => {
            let a = as_var(tape, env, node, input_name(node, 0)?)?;
            let b = as_var(tape, env, node, input_name(node, 1)?)?;
            let out = tape.custom(Arc::new(MatMulFn), &[a, b])?;
            Ok(AutogradValue::Var(out))
        }
        "Add" | "Mul" | "Div" => {
            let a = as_var(tape, env, node, input_name(node, 0)?)?;
            let b = as_var(tape, env, node, input_name(node, 1)?)?;
            let (op, forward_fn, kind): (&'static str, BinaryForwardFn, BinaryKind) =
                match node.op_type.as_str() {
                    "Add" => ("Add", ops::add, BinaryKind::Add),
                    "Mul" => ("Mul", ops::mul, BinaryKind::Mul),
                    "Div" => ("Div", ops::div, BinaryKind::Div),
                    _ => unreachable!(),
                };
            let out = tape.custom(
                Arc::new(BinaryElemFn {
                    op_name: op,
                    forward_fn,
                    kind,
                }),
                &[a, b],
            )?;
            Ok(AutogradValue::Var(out))
        }
        "Sqrt" | "Relu" | "Sigmoid" | "Erf" => {
            let x = as_var(tape, env, node, input_name(node, 0)?)?;
            let (op, forward_fn, grad_fn): (&'static str, UnaryForwardFn, fn(f32, f32) -> f32) =
                match node.op_type.as_str() {
                    "Sqrt" => ("Sqrt", ops::sqrt, sqrt_grad),
                    "Relu" => ("Relu", ops::relu, relu_grad),
                    "Sigmoid" => ("Sigmoid", ops::sigmoid, sigmoid_grad),
                    "Erf" => ("Erf", ops::erf, erf_grad),
                    _ => unreachable!(),
                };
            let out = tape.custom(
                Arc::new(UnaryElemFn {
                    op_name: op,
                    forward_fn,
                    grad_fn,
                }),
                &[x],
            )?;
            Ok(AutogradValue::Var(out))
        }
        "Softmax" => {
            let x = as_var(tape, env, node, input_name(node, 0)?)?;
            let axis = attr_i64(node, "axis", -1);
            let out = tape.custom(Arc::new(SoftmaxFn { axis }), &[x])?;
            Ok(AutogradValue::Var(out))
        }
        "LayerNormalization" => {
            let x = as_var(tape, env, node, input_name(node, 0)?)?;
            let s = as_var(tape, env, node, input_name(node, 1)?)?;
            let has_bias = matches!(node.input.get(2), Some(n) if !n.is_empty());
            let axis = attr_i64(node, "axis", -1);
            let epsilon = attr_f32(node, "epsilon", 1e-5);
            let mut inputs = vec![x, s];
            if has_bias {
                inputs.push(as_var(tape, env, node, node.input[2].as_str())?);
            }
            let out = tape.custom(
                Arc::new(LayerNormFn {
                    axis,
                    epsilon,
                    has_bias,
                }),
                &inputs,
            )?;
            Ok(AutogradValue::Var(out))
        }
        "Shape" => {
            let name = input_name(node, 0)?;
            let dims: Vec<i64> = match get_env(env, node, name)? {
                AutogradValue::Var(v) => ops::shape(&*v.value()),
                AutogradValue::Const(Value::F32(t)) => ops::shape(t),
                AutogradValue::Const(Value::I64(t)) => ops::shape(t),
                AutogradValue::Const(Value::Bool(t)) => ops::shape(t),
                AutogradValue::Const(Value::F16(t)) => ops::shape(t),
            };
            let len = dims.len();
            Ok(AutogradValue::Const(Value::I64(Tensor::new(dims, &[len])?)))
        }
        "Cast" => {
            const ONNX_DATA_TYPE_FLOAT: i64 = 1;
            const ONNX_DATA_TYPE_INT64: i64 = 7;
            const ONNX_DATA_TYPE_BOOL: i64 = 9;
            const ONNX_DATA_TYPE_FLOAT16: i64 = 10;
            let name = input_name(node, 0)?;
            let to = node
                .attribute
                .iter()
                .find(|a| a.name == "to")
                .map(|a| a.i)
                .ok_or_else(|| InterpError::MissingAttribute {
                    node: node.name.clone(),
                    attr: "to".to_string(),
                })?;
            ops::check_supported_cast_target(to)?;
            match get_env(env, node, name)? {
                AutogradValue::Var(v) if to == ONNX_DATA_TYPE_FLOAT => Ok(AutogradValue::Var(*v)),
                AutogradValue::Var(v) => {
                    // F32 -> 非 F32: ここでのみ勾配を意図的に切断する（承認事項 4）。
                    let t = v.value();
                    match to {
                        ONNX_DATA_TYPE_INT64 => {
                            Ok(AutogradValue::Const(Value::I64(ops::cast_to_int64(&t)?)))
                        }
                        ONNX_DATA_TYPE_BOOL => {
                            Ok(AutogradValue::Const(Value::Bool(ops::cast_to_bool(&t)?)))
                        }
                        ONNX_DATA_TYPE_FLOAT16 => {
                            Ok(AutogradValue::Const(Value::F16(ops::cast_to_f16(&t)?)))
                        }
                        _ => Err(InterpError::TypeMismatch {
                            node: node.name.clone(),
                            expected: "supported Cast source/target dtype combination",
                        }
                        .into()),
                    }
                }
                AutogradValue::Const(value) => match (value, to) {
                    (Value::F32(t), ONNX_DATA_TYPE_FLOAT) => {
                        Ok(AutogradValue::Const(Value::F32(t.clone())))
                    }
                    (Value::F32(t), ONNX_DATA_TYPE_INT64) => {
                        Ok(AutogradValue::Const(Value::I64(ops::cast_to_int64(t)?)))
                    }
                    (Value::F32(t), ONNX_DATA_TYPE_BOOL) => {
                        Ok(AutogradValue::Const(Value::Bool(ops::cast_to_bool(t)?)))
                    }
                    (Value::F32(t), ONNX_DATA_TYPE_FLOAT16) => {
                        Ok(AutogradValue::Const(Value::F16(ops::cast_to_f16(t)?)))
                    }
                    (Value::I64(t), ONNX_DATA_TYPE_INT64) => {
                        Ok(AutogradValue::Const(Value::I64(t.clone())))
                    }
                    (Value::I64(t), ONNX_DATA_TYPE_FLOAT) => {
                        Ok(AutogradValue::Const(Value::F32(ops::cast_to_float(t)?)))
                    }
                    (Value::Bool(t), ONNX_DATA_TYPE_BOOL) => {
                        Ok(AutogradValue::Const(Value::Bool(t.clone())))
                    }
                    (Value::Bool(t), ONNX_DATA_TYPE_FLOAT) => Ok(AutogradValue::Const(Value::F32(
                        ops::cast_bool_to_float(t)?,
                    ))),
                    (Value::F16(t), ONNX_DATA_TYPE_FLOAT16) => {
                        Ok(AutogradValue::Const(Value::F16(t.clone())))
                    }
                    (Value::F16(t), ONNX_DATA_TYPE_FLOAT) => {
                        Ok(AutogradValue::Const(Value::F32(ops::cast_f16_to_float(t)?)))
                    }
                    _ => Err(InterpError::TypeMismatch {
                        node: node.name.clone(),
                        expected: "supported Cast source/target dtype combination",
                    }
                    .into()),
                },
            }
        }
        "Constant" => {
            if let Some(attr) = node.attribute.iter().find(|a| a.name == "value") {
                let t = attr
                    .t
                    .as_ref()
                    .ok_or_else(|| InterpError::MissingAttribute {
                        node: node.name.clone(),
                        attr: "value".to_string(),
                    })?;
                let raw = super::graph::decode_tensor(t)?;
                Ok(AutogradValue::Const(raw_to_value(&raw)?))
            } else if let Some(attr) = node.attribute.iter().find(|a| a.name == "value_float") {
                Ok(AutogradValue::Const(Value::F32(ops::constant(
                    &ConstantValue::Float(attr.f),
                )?)))
            } else if let Some(attr) = node.attribute.iter().find(|a| a.name == "value_floats") {
                Ok(AutogradValue::Const(Value::F32(ops::constant(
                    &ConstantValue::Floats(attr.floats.clone()),
                )?)))
            } else if let Some(attr) = node.attribute.iter().find(|a| a.name == "value_int") {
                Ok(AutogradValue::Const(Value::I64(Tensor::new(
                    vec![attr.i],
                    &[],
                )?)))
            } else if let Some(attr) = node.attribute.iter().find(|a| a.name == "value_ints") {
                let len = attr.ints.len();
                Ok(AutogradValue::Const(Value::I64(Tensor::new(
                    attr.ints.clone(),
                    &[len],
                )?)))
            } else {
                Err(InterpError::MissingAttribute {
                    node: node.name.clone(),
                    attr: "value".to_string(),
                }
                .into())
            }
        }
        // Gather / Unsqueeze / Concat / Slice / Mod / Reshape / Squeeze / Transpose:
        // 本スコープ未実装（モジュール冒頭コメント参照）。定数専用の
        // "存在確認のみ" のショートカットも設けない（no-silent-skip 契約）。
        "Gather" | "Unsqueeze" | "Concat" | "Slice" | "Mod" | "Reshape" | "Squeeze"
        | "Transpose" => {
            // 参照する入力が Const のみであっても、本 PR のスコープでは
            // 常に fail-closed とする（Const-only 最適化は将来の拡張余地として
            // 残すが、いま実装すると Var/Const 判定漏れが no-silent-skip
            // 契約を壊しうるため見送る。#2078 実装計画からの縮小）。
            let _ = require_const; // 将来の Const-only 経路実装で使用予定
            Err(unsupported(
                node,
                "本スコープ（#2078）では autograd 経路未実装。interp::run（非勾配）を使うか、\
                 後続イシューでの拡張を待つ",
            ))
        }
        other => Err(InterpError::UnsupportedOp(other.to_string()).into()),
    }
}
