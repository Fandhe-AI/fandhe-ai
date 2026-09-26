//! `interp` のディスパッチ表が委譲する追加 8 op（`Clip`／`Tanh`／`Gelu`／
//! `Where`／`Expand`／`ReduceMean`／`Pad`／`Resize`。イシュー #2186）。
//!
//! `interp.rs`（TASK-7.2b・#78）と同じ「ONNX という信頼できない外部
//! フォーマットを実行する」前提（OWASP A03）に立つが、実装の委譲方式が
//! 異なるため別ファイルへ分離した（sibling イシュー #2199／#2200 が
//! `interp.rs` 本体を同時に編集するための衝突緩和という実務上の理由も
//! 兼ねる）。
//!
//! ## 委譲の形（既存 8 op ディスパッチ表との違い）
//!
//! 既存の `compute_*`（`interp.rs`）は `crate::ops::*`（`Tensor<f32>` を
//! 直接受け取るホスト関数）へ委譲する。本モジュールの `compute_*` は
//! 代わりに `fandhe_ai_autodiff::Var` の同名演算（`tanh`／`gelu`／
//! `gelu_tanh`／`clamp`／`where_cond`／`expand`／`mean_dims`／`pad`／
//! `interpolate`）へ委譲する。理由は `crate::ops` 側に等価な演算が
//! 存在しないため新規実装が必要になるが、`autodiff::Var` が既に
//! 数値契約込みで提供している実装（正規化統計・勾配の長軸縮約の
//! `f64` アキュムレータ規律等。`.claude/rules/coding-rust.md`）を
//! 二重実装しない判断（実装計画 §2.1）。
//!
//! 各 `compute_*` は毎回 `Tape::new()`（`default_ops::naive_ops()`。
//! naive CPU 参照実装）でテープを新規に作り、入力を
//! `tape.var_no_grad(&t)` で無勾配 `Var` として載せてから演算を呼び、
//! `.to_tensor()` で結果を取り出す。`interp::run_impl` が受け取る
//! `dev_ops: Option<&dyn BackendOps>`（CUDA／Metal 実行 opt-in。
//! イシュー #2077）はここでは一切参照しない——`Tape::new_with_ops` は
//! `Box<dyn BackendOps + Send>` の所有値を要求するため、呼び出し元が
//! 持つ `Option<&dyn BackendOps>`（借用）からは構築できない。このため
//! 本モジュールの 8 op は **常にホスト（CPU）実行**であり、
//! `run_impl` のディスパッチ表では常に `used_device = false` を返す
//! （CUDA／Metal 実行への到達経路は既定の未対応フォールバック——
//! ホスト計算——で担保する。実装計画「基盤方針」節）。
//!
//! ## エラー方針
//!
//! `fandhe_ai_autodiff::AutodiffError` は `super::interp::autodiff_err`
//! で `InterpError`（`AutodiffError::Shape` は既存の
//! `InterpError::Shape` へ、それ以外は `InterpError::Autodiff` へ）に
//! 変換する。属性・入力の検証（型・個数・opset 形式の混在検出）は
//! `interp.rs` と同じ型検証ヘルパ（`attr_*_typed`／`attr_string`／
//! `find_attr_unique`）を再利用し、無検証の `attr_i64s` 等は使わない
//! （OWASP A03。`.claude/rules/security.md`）。

use std::collections::{HashMap, HashSet};

use fandhe_ai_autodiff::{Tape, Var};
use fandhe_ai_tensor_core::{InterpolateMode, Tensor, broadcast_shape};

use super::interp::{
    InterpError, Value, attr_f32_typed, attr_i64_typed, attr_ints_typed, attr_string, autodiff_err,
    find_attr_unique, get_bool, get_f32, get_value, i64_vec_and_shape, input_name,
};
use super::proto::NodeProto;
use crate::ops::{OpError, normalize_axis};

/// 単一入力（`node.input[0]`）を要求する op 共通の入力数検査。
fn require_single_input(node: &NodeProto) -> Result<(), InterpError> {
    if node.input.len() != 1 {
        return Err(InterpError::InputArityMismatch {
            node: node.name.clone(),
            min: 1,
            max: 1,
            actual: node.input.len(),
        });
    }
    Ok(())
}

/// rank 0（スカラー）の `f32` テンソルから値を取り出す（`Clip`／`Pad` の
/// テンソル形式属性〈`min`／`max`／`constant_value`〉。ONNX 仕様上これらは
/// 0 次元テンソルであり、rank ≥ 1（例: `[1]`）は非対応として拒否する
/// （no-silent-skip 契約。呼び出し元が「rank 0 のみ受理する」と明示した
/// 契約を守る）。
fn scalar_f32(node: &NodeProto, label: &str, t: &Tensor<f32>) -> Result<f32, InterpError> {
    if t.rank() != 0 {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: label.to_string(),
            reason: format!(
                "rank 0 のスカラーである必要があります（実際: rank {}）",
                t.rank()
            ),
        });
    }
    t.get(&[]).ok_or_else(|| InterpError::InvalidAttribute {
        node: node.name.clone(),
        attr: label.to_string(),
        reason: "スカラー値の取得に失敗しました".to_string(),
    })
}

/// `Tanh(x) -> y`（ONNX Tanh-13。イシュー #2186）。`Var::tanh` は
/// `Result` を返さない（入力 shape によらず常に成功する要素ごと演算）
/// ため、本関数もエラーを返す経路は入力数検査のみ。
pub(super) fn compute_tanh(
    env: &HashMap<String, Value>,
    node: &NodeProto,
) -> Result<Value, InterpError> {
    require_single_input(node)?;
    let x = get_f32(env, node, input_name(node, 0)?)?;
    let tape = Tape::new();
    let out = tape.var_no_grad(x).tanh();
    Ok(Value::F32(out.to_tensor()))
}

/// `Gelu(x, approximate) -> y`（ONNX Gelu-20。イシュー #2186）。
/// `approximate` は STRING 属性で `"none"`（既定・erf 版）か `"tanh"`
/// （tanh 近似版）のいずれかのみを受理する。それ以外の値（空文字列を
/// 含む）は [`InterpError::InvalidAttribute`] で拒否する（無言で
/// `"none"` へ fallback しない。OWASP A03）。
pub(super) fn compute_gelu(
    env: &HashMap<String, Value>,
    node: &NodeProto,
) -> Result<Value, InterpError> {
    require_single_input(node)?;
    let x = get_f32(env, node, input_name(node, 0)?)?;
    let approximate = attr_string(node, "approximate", "none")?;
    let tape = Tape::new();
    let v = tape.var_no_grad(x);
    let out = match approximate.as_str() {
        "none" => v.gelu().map_err(|e| autodiff_err(&node.name, e))?,
        "tanh" => v.gelu_tanh().map_err(|e| autodiff_err(&node.name, e))?,
        other => {
            return Err(InterpError::InvalidAttribute {
                node: node.name.clone(),
                attr: "approximate".to_string(),
                reason: format!("未対応の approximate 値です（'none'／'tanh' のみ対応）: {other}"),
            });
        }
    };
    Ok(Value::F32(out.to_tensor()))
}

/// `Clip(x, min?, max?) -> y`（ONNX Clip-6〈attr 形〉／Clip-11+〈入力
/// 形〉。イシュー #2186）。opset ごとに形式が異なる（`Graph` は opset を
/// 保持しないため、`min`／`max` 属性の有無と入力数から構造的に判定する。
/// 実装計画 §2.3）:
///
/// - attr 形: `min`／`max`（FLOAT 属性。いずれも省略可・省略時は
///   `-inf`／`+inf`）
/// - 入力形: 第 2・第 3 入力（rank 0 の `f32`。省略可・空文字列も
///   省略として扱う既定の `input_name` 規約とは異なり `node.input.get`
///   で直接判定する——`Clip` の `min`／`max` は「渡さない」ことが
///   `-inf`／`+inf` という意味を持つ ONNX 仕様上の optional input のため）
///
/// 両形式が同時に検出された場合（属性が存在し、かつ第 2 入力以降が
/// ある）は [`InterpError::InvalidAttribute`] で fail-closed に拒否する
/// （opset 判定を無言で片方優先しない）。`min > max` の場合は ONNX
/// 仕様どおり常に `max` を返し、NaN は伝播する（`Var::clamp` の契約。
/// `.claude/rules/coding-rust.md` 参照コメント不要——`clamp` 自身の
/// doc が正）。
pub(super) fn compute_clip(
    env: &HashMap<String, Value>,
    node: &NodeProto,
) -> Result<Value, InterpError> {
    if node.input.is_empty() || node.input.len() > 3 {
        return Err(InterpError::InputArityMismatch {
            node: node.name.clone(),
            min: 1,
            max: 3,
            actual: node.input.len(),
        });
    }
    let has_min_attr = find_attr_unique(node, "min")?.is_some();
    let has_max_attr = find_attr_unique(node, "max")?.is_some();
    let has_extra_input = node.input.len() > 1;
    if (has_min_attr || has_max_attr) && has_extra_input {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: "min/max".to_string(),
            reason: "attr 形（min/max 属性）と入力形（第 2・第 3 入力）が混在しています"
                .to_string(),
        });
    }

    let x = get_f32(env, node, input_name(node, 0)?)?;
    let (min, max) = if has_min_attr || has_max_attr {
        (
            attr_f32_typed(node, "min", f32::NEG_INFINITY)?,
            attr_f32_typed(node, "max", f32::INFINITY)?,
        )
    } else {
        let min = match node.input.get(1) {
            Some(name) if !name.is_empty() => scalar_f32(node, "min", get_f32(env, node, name)?)?,
            _ => f32::NEG_INFINITY,
        };
        let max = match node.input.get(2) {
            Some(name) if !name.is_empty() => scalar_f32(node, "max", get_f32(env, node, name)?)?,
            _ => f32::INFINITY,
        };
        (min, max)
    };

    let tape = Tape::new();
    let v = tape.var_no_grad(x);
    let out = v.clamp(min, max).map_err(|e| autodiff_err(&node.name, e))?;
    Ok(Value::F32(out.to_tensor()))
}

/// `Where(cond, X, Y) -> output`（ONNX Where-16。イシュー #2186）。
/// `cond` は bool、`X`／`Y` は `f32`（双方向ブロードキャスト。3 入力
/// 全体のブロードキャストは `Var::where_cond` 自身が担う）。
pub(super) fn compute_where(
    env: &HashMap<String, Value>,
    node: &NodeProto,
) -> Result<Value, InterpError> {
    if node.input.len() != 3 {
        return Err(InterpError::InputArityMismatch {
            node: node.name.clone(),
            min: 3,
            max: 3,
            actual: node.input.len(),
        });
    }
    let cond = get_bool(env, node, input_name(node, 0)?)?;
    let x = get_f32(env, node, input_name(node, 1)?)?;
    let y = get_f32(env, node, input_name(node, 2)?)?;
    let tape = Tape::new();
    let a = tape.var_no_grad(x);
    let b = tape.var_no_grad(y);
    let out = Var::where_cond(cond, &a, &b).map_err(|e| autodiff_err(&node.name, e))?;
    Ok(Value::F32(out.to_tensor()))
}

/// `Expand(data, shape) -> output`（ONNX Expand-13。イシュー #2186）。
/// `shape`（1 次元 `i64`）と `data` の shape を双方向ブロードキャストで
/// 合成した出力 shape へ拡張する（`fandhe_ai_tensor_core::broadcast_shape`
/// が単一情報源。`Var::expand`／`Tensor::broadcast_to` は共に
/// 「`shape.len() >= 入力 rank`・各軸が一致または入力側が 1」という
/// 単方向契約のみを持つため、真の双方向ブロードキャスト先 shape を
/// 事前に確定してから渡す）。`f32` は `Var::expand`（勾配追跡不要の
/// `var_no_grad` 経由）、`i64`／`bool`／`f16` は算術を伴わない純コピー
/// のため `Tensor::broadcast_to` + `contiguous()` で直接処理する
/// （実装計画 §2.1「非 F32 の dtype」節）。
pub(super) fn compute_expand(
    env: &HashMap<String, Value>,
    node: &NodeProto,
) -> Result<Value, InterpError> {
    if node.input.len() != 2 {
        return Err(InterpError::InputArityMismatch {
            node: node.name.clone(),
            min: 2,
            max: 2,
            actual: node.input.len(),
        });
    }
    let data_name = input_name(node, 0)?;
    let shape_name = input_name(node, 1)?;
    let (shape_i64, shape_shape) = i64_vec_and_shape(env, node, shape_name)?;
    if shape_shape.len() != 1 {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: "shape".to_string(),
            reason: format!(
                "shape 入力は 1 次元でなければなりません（実際: rank {}）",
                shape_shape.len()
            ),
        });
    }
    let mut target_shape = Vec::with_capacity(shape_i64.len());
    for &d in &shape_i64 {
        if d < 0 {
            return Err(InterpError::InvalidAttribute {
                node: node.name.clone(),
                attr: "shape".to_string(),
                reason: format!("shape の要素は非負である必要があります（実際: {d}）"),
            });
        }
        target_shape.push(d as usize);
    }

    match get_value(env, node, data_name)? {
        Value::F32(t) => {
            let out_shape = broadcast_shape(t.shape(), &target_shape)?;
            let tape = Tape::new();
            let out = tape
                .var_no_grad(t)
                .expand(&out_shape)
                .map_err(|e| autodiff_err(&node.name, e))?;
            Ok(Value::F32(out.to_tensor()))
        }
        Value::I64(t) => {
            let out_shape = broadcast_shape(t.shape(), &target_shape)?;
            Ok(Value::I64(t.broadcast_to(&out_shape)?.contiguous()))
        }
        Value::Bool(t) => {
            let out_shape = broadcast_shape(t.shape(), &target_shape)?;
            Ok(Value::Bool(t.broadcast_to(&out_shape)?.contiguous()))
        }
        Value::F16(t) => {
            let out_shape = broadcast_shape(t.shape(), &target_shape)?;
            Ok(Value::F16(t.broadcast_to(&out_shape)?.contiguous()))
        }
    }
}

/// `ReduceMean(data, axes?) -> reduced`（ONNX ReduceMean-13〈`axes`
/// INTS 属性〉／ReduceMean-18〈`axes` 第 2 入力・`noop_with_empty_axes`〉。
/// イシュー #2186）。`f32` 専用（`Var::mean`／`mean_dims` が `f32`
/// 専用のため）。
///
/// `axes` は属性・入力のどちらか一方のみ許容し（両方存在すれば
/// [`InterpError::InvalidAttribute`]）、省略時は全軸縮約
/// （`Var::mean(None)`。空リストが明示された場合も `noop_with_empty_axes`
/// が `0`〈既定〉なら同じ全軸縮約、`1` なら恒等）。負軸は
/// [`normalize_axis`] で正規化し、範囲外は
/// [`OpError::AxisOutOfRange`] で拒否する。重複軸の検出は
/// `Var::mean_dims`（`autodiff::reduce_dims::plan_reduce_dims`）に
/// 委譲する（同じ検査ロジックの二重実装を避ける）。
pub(super) fn compute_reduce_mean(
    env: &HashMap<String, Value>,
    node: &NodeProto,
) -> Result<Value, InterpError> {
    if node.input.is_empty() || node.input.len() > 2 {
        return Err(InterpError::InputArityMismatch {
            node: node.name.clone(),
            min: 1,
            max: 2,
            actual: node.input.len(),
        });
    }
    let has_axes_input = matches!(node.input.get(1), Some(n) if !n.is_empty());
    let has_axes_attr = find_attr_unique(node, "axes")?.is_some();
    if has_axes_input && has_axes_attr {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: "axes".to_string(),
            reason: "axes の attr 形（INTS 属性）と入力形（第 2 入力）が混在しています".to_string(),
        });
    }

    let x = get_f32(env, node, input_name(node, 0)?)?;
    let rank = x.rank();
    let keepdims = attr_i64_typed(node, "keepdims", 1)? != 0;
    let noop_with_empty_axes = attr_i64_typed(node, "noop_with_empty_axes", 0)? != 0;

    let axes: Option<Vec<i64>> = if has_axes_input {
        Some(i64_vec_and_shape(env, node, node.input[1].as_str())?.0)
    } else {
        attr_ints_typed(node, "axes")?.map(<[i64]>::to_vec)
    };

    let tape = Tape::new();
    let v = tape.var_no_grad(x);

    // 全軸縮約（属性・入力とも省略、または空リストかつ noop=0）:
    // `Var::mean_dims` は空 dims を拒否する（実装計画注記）ため
    // `Var::mean(None)` を使う（単軸 API・全軸縮約は元々の等価経路）。
    let full_reduce = |v: &Var<'_>| -> Result<Value, InterpError> {
        let reduced = v.mean(None).map_err(|e| autodiff_err(&node.name, e))?;
        let out = if keepdims {
            reduced
                .reshape(&vec![1usize; rank])
                .map_err(|e| autodiff_err(&node.name, e))?
        } else {
            reduced
        };
        Ok(Value::F32(out.to_tensor()))
    };

    match axes {
        None => full_reduce(&v),
        Some(axes) if axes.is_empty() => {
            if noop_with_empty_axes {
                Ok(Value::F32(x.clone()))
            } else {
                full_reduce(&v)
            }
        }
        Some(axes) => {
            let mut dims = Vec::with_capacity(axes.len());
            for a in axes {
                let n = normalize_axis(a, rank).ok_or(OpError::AxisOutOfRange {
                    op: "ReduceMean",
                    axis: a,
                    rank,
                })?;
                dims.push(n);
            }
            let out = v
                .mean_dims(&dims, keepdims)
                .map_err(|e| autodiff_err(&node.name, e))?;
            Ok(Value::F32(out.to_tensor()))
        }
    }
}

/// `Pad(data, pads?, constant_value?, axes?) -> output`（ONNX Pad-2
/// 〈attr 形: `pads`／`value`／`mode`〉／Pad-11+〈入力形: `pads`（必須
/// 第 2 入力・`i64` `[2r]`）／`constant_value`（第 3 入力・省略可）／
/// `axes`（第 4 入力・opset 18・省略可）〉。イシュー #2186）。
/// `mode` は `"constant"`（既定）のみ対応（`reflect`／`edge`／`wrap`
/// は非対応として拒否）。負の pads（クロップ）も非対応（[`Var::pad`]
/// の契約）。
pub(super) fn compute_pad(
    env: &HashMap<String, Value>,
    node: &NodeProto,
) -> Result<Value, InterpError> {
    if node.input.is_empty() || node.input.len() > 4 {
        return Err(InterpError::InputArityMismatch {
            node: node.name.clone(),
            min: 1,
            max: 4,
            actual: node.input.len(),
        });
    }
    let has_pads_attr = find_attr_unique(node, "pads")?.is_some();
    let has_extra_input = node.input.len() > 1;
    if has_pads_attr && has_extra_input {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: "pads".to_string(),
            reason: "pads の attr 形と入力形が混在しています".to_string(),
        });
    }

    let x = get_f32(env, node, input_name(node, 0)?)?;
    let rank = x.rank();

    let mode = attr_string(node, "mode", "constant")?;
    if mode != "constant" {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: "mode".to_string(),
            reason: format!("mode='{mode}' は非対応です（'constant' のみ対応）"),
        });
    }

    let (pads_i64, value, axes): (Vec<i64>, f32, Option<Vec<i64>>) = if has_pads_attr {
        // `find_attr_unique` で存在確認済みのため `attr_ints_typed` は
        // 必ず `Some` を返す（`unwrap_or_default` は到達しない防御）。
        let pads = attr_ints_typed(node, "pads")?.unwrap_or_default().to_vec();
        let v = attr_f32_typed(node, "value", 0.0)?;
        (pads, v, None)
    } else {
        let (pads, _) = i64_vec_and_shape(env, node, input_name(node, 1)?)?;
        let v = match node.input.get(2) {
            Some(name) if !name.is_empty() => {
                scalar_f32(node, "constant_value", get_f32(env, node, name)?)?
            }
            _ => 0.0,
        };
        let axes = match node.input.get(3) {
            Some(name) if !name.is_empty() => Some(i64_vec_and_shape(env, node, name)?.0),
            _ => None,
        };
        (pads, v, axes)
    };

    let pairs: Vec<(usize, usize)> = if let Some(axes_raw) = axes {
        if pads_i64.len() != 2 * axes_raw.len() {
            return Err(InterpError::InvalidAttribute {
                node: node.name.clone(),
                attr: "pads".to_string(),
                reason: format!(
                    "axes 指定時、pads の長さは 2*axes.len()({}) である必要があります（実際: {}）",
                    2 * axes_raw.len(),
                    pads_i64.len()
                ),
            });
        }
        let mut result = vec![(0usize, 0usize); rank];
        // `axes` 入力で同一軸が重複指定される（例: [0, 0]）と、後勝ちで
        // `result[n]` が無言上書きされ入力の一部意図が消える。本ファイルの
        // 他の曖昧入力検出（`find_attr_unique` の複数出現・ReduceMean の
        // attr/input 混在等）と同じ fail-closed 方針に揃え、重複軸は
        // InvalidAttribute で拒否する（レビュー指摘対応）。
        let mut seen_axes: HashSet<usize> = HashSet::with_capacity(axes_raw.len());
        for (i, &ax) in axes_raw.iter().enumerate() {
            let n = normalize_axis(ax, rank).ok_or(OpError::AxisOutOfRange {
                op: "Pad",
                axis: ax,
                rank,
            })?;
            if !seen_axes.insert(n) {
                return Err(InterpError::InvalidAttribute {
                    node: node.name.clone(),
                    attr: "axes".to_string(),
                    reason: format!("axes に重複した軸指定があります（軸: {n}）"),
                });
            }
            let before = pads_i64[i];
            let after = pads_i64[axes_raw.len() + i];
            if before < 0 || after < 0 {
                return Err(InterpError::InvalidAttribute {
                    node: node.name.clone(),
                    attr: "pads".to_string(),
                    reason: "負の pads（クロップ）は非対応です".to_string(),
                });
            }
            result[n] = (before as usize, after as usize);
        }
        result
    } else {
        if pads_i64.len() != 2 * rank {
            return Err(InterpError::InvalidAttribute {
                node: node.name.clone(),
                attr: "pads".to_string(),
                reason: format!(
                    "pads の長さは 2*rank({}) である必要があります（実際: {}）",
                    2 * rank,
                    pads_i64.len()
                ),
            });
        }
        let mut result = Vec::with_capacity(rank);
        for i in 0..rank {
            let before = pads_i64[i];
            let after = pads_i64[rank + i];
            if before < 0 || after < 0 {
                return Err(InterpError::InvalidAttribute {
                    node: node.name.clone(),
                    attr: "pads".to_string(),
                    reason: "負の pads（クロップ）は非対応です".to_string(),
                });
            }
            result.push((before as usize, after as usize));
        }
        result
    };

    let tape = Tape::new();
    let out = tape
        .var_no_grad(x)
        .pad(&pairs, value)
        .map_err(|e| autodiff_err(&node.name, e))?;
    Ok(Value::F32(out.to_tensor()))
}

/// `Resize(X, roi?, scales?, sizes?) -> Y`（ONNX Resize-10〈2 入力:
/// `X`／`scales`〉・Resize-11+〈3〜4 入力: `X`／`roi`／`scales`／
/// `sizes`〉。イシュー #2186）。`X` は rank 4（NCHW）限定・N／C 軸の
/// 倍率は 1 固定。受理する `mode`／`coordinate_transformation_mode`／
/// `nearest_mode` の組合せは実装計画 §2.3 の表を正とする（それ以外は
/// [`InterpError::InvalidAttribute`] で fail-closed に拒否し、
/// `reason` に代替手段〈`sizes` 入力を使う等〉の手掛かりを含める）。
pub(super) fn compute_resize(
    env: &HashMap<String, Value>,
    node: &NodeProto,
) -> Result<Value, InterpError> {
    if node.input.len() < 2 || node.input.len() > 4 {
        return Err(InterpError::InputArityMismatch {
            node: node.name.clone(),
            min: 2,
            max: 4,
            actual: node.input.len(),
        });
    }
    let x = get_f32(env, node, input_name(node, 0)?)?;
    if x.rank() != 4 {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: "X".to_string(),
            reason: format!("rank 4（NCHW）のみ対応です（実際: rank {}）", x.rank()),
        });
    }
    let in_shape = x.shape().to_vec();

    if let Some(name) = node.input.get(1).filter(|n| !n.is_empty()) {
        let roi = get_f32(env, node, name)?;
        if roi.numel() != 0 {
            return Err(InterpError::InvalidAttribute {
                node: node.name.clone(),
                attr: "roi".to_string(),
                reason: "roi は空である必要があります（tf_crop_and_resize は非対応）".to_string(),
            });
        }
    }

    let scales_name = node.input.get(2).filter(|n| !n.is_empty());
    let sizes_name = node.input.get(3).filter(|n| !n.is_empty());
    // `(Some, Some)`（両方指定）・`(None, None)`（どちらも省略）は fail-closed
    // に拒否する。`match` の各腕が `name` を直接束縛するため、以降
    // `expect`／`unwrap`／`unreachable!` を使わずに分岐できる
    // （coding-rust.md「本番経路で unwrap/expect 禁止」）。
    let (out_h, out_w) = match (scales_name, sizes_name) {
        (Some(name), None) => {
            let scales_t = get_f32(env, node, name)?.contiguous();
            let s = scales_t
                .as_slice()
                .ok_or(OpError::NonContiguousInternal("Resize(scales)"))?;
            if s.len() != 4 {
                return Err(InterpError::InvalidAttribute {
                    node: node.name.clone(),
                    attr: "scales".to_string(),
                    reason: format!("scales は長さ 4（NCHW）が必要です（実際: {}）", s.len()),
                });
            }
            for (i, &sv) in s.iter().take(2).enumerate() {
                if (sv - 1.0).abs() > 1e-6 {
                    return Err(InterpError::InvalidAttribute {
                        node: node.name.clone(),
                        attr: "scales".to_string(),
                        reason: format!(
                            "N/C 軸（index {i}）の倍率は 1 でなければなりません（実際: {sv}）"
                        ),
                    });
                }
            }
            let out_h = resize_scale_to_out_size(node, "scales", s[2], in_shape[2])?;
            let out_w = resize_scale_to_out_size(node, "scales", s[3], in_shape[3])?;
            (out_h, out_w)
        }
        (None, Some(name)) => {
            let (sizes, _) = i64_vec_and_shape(env, node, name)?;
            if sizes.len() != 4 {
                return Err(InterpError::InvalidAttribute {
                    node: node.name.clone(),
                    attr: "sizes".to_string(),
                    reason: format!("sizes は長さ 4（NCHW）が必要です（実際: {}）", sizes.len()),
                });
            }
            if sizes[0] as usize != in_shape[0] || sizes[1] as usize != in_shape[1] {
                return Err(InterpError::InvalidAttribute {
                    node: node.name.clone(),
                    attr: "sizes".to_string(),
                    reason: "N/C 軸の sizes は入力の N/C と一致する必要があります".to_string(),
                });
            }
            if sizes[2] < 0 || sizes[3] < 0 {
                return Err(InterpError::InvalidAttribute {
                    node: node.name.clone(),
                    attr: "sizes".to_string(),
                    reason: "H/W 軸の sizes は非負である必要があります".to_string(),
                });
            }
            (sizes[2] as usize, sizes[3] as usize)
        }
        (Some(_), Some(_)) | (None, None) => {
            return Err(InterpError::InvalidAttribute {
                node: node.name.clone(),
                attr: "scales/sizes".to_string(),
                reason: "scales と sizes はどちらか一方のみ指定してください".to_string(),
            });
        }
    };

    let coord = attr_string(node, "coordinate_transformation_mode", "half_pixel")?;
    let mode_attr = attr_string(node, "mode", "nearest")?;
    let nearest_mode = attr_string(node, "nearest_mode", "round_prefer_floor")?;
    let antialias = attr_i64_typed(node, "antialias", 0)?;
    let exclude_outside = attr_i64_typed(node, "exclude_outside", 0)?;
    let keep_aspect_ratio_policy = attr_string(node, "keep_aspect_ratio_policy", "stretch")?;

    if antialias != 0 {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: "antialias".to_string(),
            reason: "antialias != 0 は非対応です".to_string(),
        });
    }
    if exclude_outside != 0 {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: "exclude_outside".to_string(),
            reason: "exclude_outside != 0 は非対応です".to_string(),
        });
    }
    if keep_aspect_ratio_policy != "stretch" {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: "keep_aspect_ratio_policy".to_string(),
            reason: format!("'{keep_aspect_ratio_policy}' は非対応です（'stretch' のみ対応）"),
        });
    }
    if coord == "tf_crop_and_resize" || coord == "tf_half_pixel_for_nn" {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: "coordinate_transformation_mode".to_string(),
            reason: format!("'{coord}' は非対応です"),
        });
    }
    if mode_attr == "cubic" {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: "mode".to_string(),
            reason: "'cubic' は非対応です".to_string(),
        });
    }

    let interp_mode = match (mode_attr.as_str(), coord.as_str(), nearest_mode.as_str()) {
        ("nearest", "asymmetric", "floor") => InterpolateMode::Nearest,
        ("nearest", "half_pixel", "round_prefer_ceil") => InterpolateMode::NearestExact,
        ("linear", "half_pixel", _) => InterpolateMode::Bilinear {
            align_corners: false,
        },
        ("linear", "pytorch_half_pixel", _) => {
            if out_h == 1 || out_w == 1 {
                return Err(InterpError::InvalidAttribute {
                    node: node.name.clone(),
                    attr: "coordinate_transformation_mode".to_string(),
                    reason: "'pytorch_half_pixel' は出力サイズ 1 の特例が PyTorch と食い違うため \
                             非対応です（sizes 入力を使うか coordinate_transformation_mode を \
                             変更してください）"
                        .to_string(),
                });
            }
            InterpolateMode::Bilinear {
                align_corners: false,
            }
        }
        ("linear", "align_corners", _) => InterpolateMode::Bilinear {
            align_corners: true,
        },
        (m, c, n) => {
            return Err(InterpError::InvalidAttribute {
                node: node.name.clone(),
                attr: "mode/coordinate_transformation_mode/nearest_mode".to_string(),
                reason: format!(
                    "組合せ mode='{m}' coordinate_transformation_mode='{c}' nearest_mode='{n}' \
                     は非対応です（sizes 入力を使うか、受理される組合せ〈実装計画 §2.3〉へ \
                     変更してください）"
                ),
            });
        }
    };

    let tape = Tape::new();
    let out = tape
        .var_no_grad(x)
        .interpolate(&[out_h, out_w], interp_mode)
        .map_err(|e| autodiff_err(&node.name, e))?;
    Ok(Value::F32(out.to_tensor()))
}

/// `Resize` の `scales`（H／W いずれか 1 軸分）から出力サイズを導出する。
/// ONNX は `scales` をそのまま座標変換式に使うため、`in_size` に対して
/// 整数倍（`scale >= 1`）または整数分の 1（`scale < 1` かつ `in_size` が
/// 割り切れる）のときのみ `in/out` の対応が一意に定まる。それ以外
/// （非整数倍率・割り切れない縮小率）は写像できないため拒否する
/// （実装計画 §2.3「Resize の受理と拒否」節）。
fn resize_scale_to_out_size(
    node: &NodeProto,
    attr: &str,
    scale: f32,
    in_size: usize,
) -> Result<usize, InterpError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: attr.to_string(),
            reason: format!("scales は正の有限値である必要があります（実際: {scale}）"),
        });
    }
    if scale >= 1.0 {
        if scale.fract() != 0.0 {
            return Err(InterpError::InvalidAttribute {
                node: node.name.clone(),
                attr: attr.to_string(),
                reason: format!("整数倍率のみ対応です（実際: {scale}）"),
            });
        }
        // `scale` は非信頼な ONNX モデルの `scales` 入力に由来するため、
        // `usize::MAX` を超える巨大な整数値浮動小数点数（例: 1e30）でも
        // `as usize` は素通しでキャストしてしまう（Rust の `as` は
        // 飽和変換だが、直後の `in_size * scale_usize` が usize の乗算
        // オーバーフローを起こしうる）。`checked_mul` で乗算そのものを
        // fail-closed に拒否し、debug/release いずれのビルドでも
        // panic・誤った出力 shape を生まないようにする（レビュー指摘対応）。
        // `usize::MAX as f32` は丸めで 2^64 になり得るため `>` だと
        // ちょうど 2^64 の scale を素通ししてしまう（`as usize` が
        // `usize::MAX` へ飽和し、後続の `checked_mul` が偶然 in_size==1 等で
        // 成立してしまう）。`>=` にして境界値も確実に拒否する。
        if scale >= usize::MAX as f32 {
            return Err(InterpError::InvalidAttribute {
                node: node.name.clone(),
                attr: attr.to_string(),
                reason: format!("scales が大きすぎます（実際: {scale}）"),
            });
        }
        let scale_usize = scale as usize;
        in_size
            .checked_mul(scale_usize)
            .ok_or_else(|| InterpError::InvalidAttribute {
                node: node.name.clone(),
                attr: attr.to_string(),
                reason: format!(
                    "in_size({in_size}) と scales({scale}) の積が usize の範囲を超えます"
                ),
            })
    } else {
        let inv = 1.0 / scale;
        if inv.fract() != 0.0 {
            return Err(InterpError::InvalidAttribute {
                node: node.name.clone(),
                attr: attr.to_string(),
                reason: format!("整数分の 1 の縮小率のみ対応です（実際: {scale}）"),
            });
        }
        let divisor = inv as usize;
        if divisor == 0 || !in_size.is_multiple_of(divisor) {
            return Err(InterpError::InvalidAttribute {
                node: node.name.clone(),
                attr: attr.to_string(),
                reason: format!("in_size({in_size}) が縮小率 1/{divisor} で割り切れません"),
            });
        }
        Ok(in_size / divisor)
    }
}
