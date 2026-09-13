//! `ScalarUnaryOp`／`ScalarBinaryOp`（`tensor-core::scalar_op`。イシュー
//! #1634）の CPU 参照実装。
//!
//! `elementwise.rs`（TASK-1.6b・#22）の 2 層構成（スライスカーネル層／
//! Tensor 入口層）・並列化方針（[`crate::elementwise::PARALLEL_THRESHOLD`]
//! 未満は逐次実行）・非 contiguous 入力の読み出し経路
//! （[`crate::elementwise::ElementwiseReadOperand`]・`Tensor::get` への
//! 二段フォールバック）をそのまま踏襲するが、演算本体を `fn` ポインタ
//! ではなく `ScalarUnaryOp`／`ScalarBinaryOp`（`Copy` な enum 値）で
//! 受け取る汎用ループにする（演算を 1 つ足すごとに専用スライス関数を
//! 増やさずに済むようにするのが本モジュールの動機。`crate::scalar_op`
//! モジュール doc「動機」参照）。
//!
//! # 既存 5 演算（`add`／`mul`／`relu`／`exp`／`tanh`）との bit 同一性
//!
//! elementwise 演算は縮約を含まない要素ごとの map 演算であり、演算順序
//! （逐次／`rayon` 並列のどちらで要素を処理するか）が結果に影響しない
//! （`elementwise.rs` モジュール doc「並列化」と同じ理由）。`ScalarUnaryOp::
//! Add`/`Mul`/`Exp`/`Tanh`〈のうち `Add`/`Mul` は `ScalarBinaryOp`〉の
//! `apply` は既存 `add_slice`/`mul_slice`/`exp_slice`/`tanh_slice` と同一の
//! 1 回の IEEE 754 演算（`+`/`*`/`f32::exp`/`f32::tanh`）を行うため、本
//! モジュールの汎用ループは既存カーネルと **bit 同一** になる
//! （`tests/scalar_op_parity.rs` で機械確認）。
//!
//! **`ScalarUnaryOp::Relu` のみ例外**: `tensor-core::scalar_op` モジュール
//! doc が明記するとおり、本 variant は既存 `BackendOps::relu`
//! （`elementwise::relu`。`f32::max` で `NaN` を伝播しない）とは異なり
//! `NaN` を明示的に伝播する。非 `NaN` 入力では両者とも `x.max(0.0)` と
//! 同値のため bit 同一だが、`NaN` 入力では結果が異なる（意図的な差異。
//! `docs/scalar-op-dispatch-design.md` §3.6 参照）。

use fandhe_ai_tensor_core::{
    Element, ScalarBinaryOp, ScalarUnaryOp, ShapeError, Tensor, elementwise_out_shape,
};
use rayon::prelude::*;

use crate::elementwise::{ElementwiseReadOperand, PARALLEL_THRESHOLD, increment_index};

// --- スライスカーネル層 ---

/// 単項スカラー演算のスライスカーネル（`out[i] = op.apply(a[i])`）。
/// `a`/`out` の長さ不一致は `assert_eq!`（release ビルドでも有効。
/// `elementwise.rs` の `*_slice` 関数群と同じ契約）で検査する。
fn scalar_unary_slice(a: &[f32], op: ScalarUnaryOp, out: &mut [f32]) {
    assert_eq!(a.len(), out.len(), "scalar_unary_slice: length mismatch");
    if a.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(a.par_iter())
            .for_each(|(o, &x)| *o = op.apply(x));
    } else {
        for (o, &x) in out.iter_mut().zip(a) {
            *o = op.apply(x);
        }
    }
}

/// 2 項スカラー演算のスライスカーネル（`out[i] = op.apply(a[i], b[i])`）。
fn scalar_binary_slice(a: &[f32], b: &[f32], op: ScalarBinaryOp, out: &mut [f32]) {
    assert_eq!(
        a.len(),
        b.len(),
        "scalar_binary_slice: length mismatch (a vs b)"
    );
    assert_eq!(
        a.len(),
        out.len(),
        "scalar_binary_slice: length mismatch (a vs out)"
    );
    if a.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(a.par_iter())
            .zip(b.par_iter())
            .for_each(|((o, &x), &y)| *o = op.apply(x, y));
    } else {
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = op.apply(x, y);
        }
    }
}

// --- Tensor 入口層 ---

/// 単項スカラー演算の Tensor 入口（shape 不変。`elementwise::
/// unary_elementwise` と同型の fast path／general path 分岐）。
pub fn scalar_unary(a: &Tensor<f32>, op: ScalarUnaryOp) -> Result<Tensor<f32>, ShapeError> {
    if let Some(sa) = a.as_slice() {
        let mut out = vec![0.0f32; sa.len()];
        scalar_unary_slice(sa, op, &mut out);
        return Tensor::new(out, a.shape());
    }

    let shape = a.shape();
    let numel = a.numel();
    let mut out = Vec::with_capacity(numel);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..numel {
        let x = a.get(&index);
        debug_assert!(
            x.is_some(),
            "scalar_unary: shape 走査ロジックのバグにより index {index:?} が範囲外になった"
        );
        out.push(op.apply(x.unwrap_or_else(Element::zero)));
        increment_index(&mut index, shape);
    }
    Tensor::new(out, shape)
}

/// 2 項スカラー演算の Tensor 入口（ブロードキャスト対応。`elementwise::
/// binary_elementwise` と同型の 3 段フォールバック: fast path
/// （両 view が contiguous）→ stride 読み（[`ElementwiseReadOperand`]。
/// 両オペランドとも非負 stride）→ `Tensor::get` 経由の走査）。
pub fn scalar_binary(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    op: ScalarBinaryOp,
) -> Result<Tensor<f32>, ShapeError> {
    let out_shape = elementwise_out_shape(a.shape(), b.shape())?;
    let ba = a.broadcast_to(&out_shape)?;
    let bb = b.broadcast_to(&out_shape)?;

    if let (Some(sa), Some(sb)) = (ba.as_slice(), bb.as_slice()) {
        let mut out = vec![0.0f32; sa.len()];
        scalar_binary_slice(sa, sb, op, &mut out);
        return Tensor::new(out, &out_shape);
    }

    let numel = out_shape.iter().product::<usize>();
    let mut index = vec![0usize; out_shape.len()];

    if let (Some(a_op), Some(b_op)) = (
        ElementwiseReadOperand::classify(&ba),
        ElementwiseReadOperand::classify(&bb),
    ) {
        let mut out = Vec::with_capacity(numel);
        for flat in 0..numel {
            let x = a_op.read(&index, flat);
            let y = b_op.read(&index, flat);
            debug_assert!(
                x.is_some() && y.is_some(),
                "scalar_binary: as_view_slice の span 保証が破れ、境界外アクセスを検知した (index {index:?})"
            );
            out.push(op.apply(
                x.unwrap_or_else(Element::zero),
                y.unwrap_or_else(Element::zero),
            ));
            increment_index(&mut index, &out_shape);
        }
        return Tensor::new(out, &out_shape);
    }

    let mut out = Vec::with_capacity(numel);
    for _ in 0..numel {
        let x = ba.get(&index);
        let y = bb.get(&index);
        debug_assert!(
            x.is_some() && y.is_some(),
            "scalar_binary: shape 走査ロジックのバグにより index {index:?} が範囲外になった"
        );
        out.push(op.apply(
            x.unwrap_or_else(Element::zero),
            y.unwrap_or_else(Element::zero),
        ));
        increment_index(&mut index, &out_shape);
    }
    Tensor::new(out, &out_shape)
}
