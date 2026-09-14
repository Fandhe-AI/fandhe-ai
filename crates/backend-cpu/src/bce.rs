//! 二値交差エントロピー損失（BCE）の融合カーネル（イシュー #1737・
//! 親イシュー #1609「損失関数の拡張」）。
//!
//! [`mse`](crate::mse) と同じ「CPU バックエンド固有のカーネル本体を
//! 独立モジュールへ切り出し、`ops.rs` の `CpuBackendOps` からは薄く
//! 委譲するだけ」という既存構成を踏襲し、
//! `fandhe_ai_tensor_core::BackendOps::bce_loss`／`bce_loss_backward` の
//! CPU 実装を提供する。`fandhe_ai_autodiff::var::Var::bce_loss`／
//! `bce_with_logits_loss`／`grad::vjp` の `Op::BceLoss` 分岐から
//! `ops.rs::CpuBackendOps` 経由で呼ばれる。
//!
//! 要素式（forward の損失・backward の `dInput`）の意味論の正は
//! `fandhe_ai_autodiff::eval::bce_elem_loss`／`bce_elem_grad_input`
//! （`crates/autodiff/src/eval.rs`）であり、本モジュールは同じ数式を
//! 意図的に複製する（`autodiff` → `backend-cpu` の逆依存を作れないため。
//! `mse.rs`・`docs/autodiff-linalg-design.md` の「eval と CPU 実装の
//! 意図的複製」と同型の整理）。`bce_parity.rs`（integration test）が
//! 両者の一致を突き合わせる。
//!
//! # 決定性契約
//!
//! forward の要素損失和は [`mse::CHUNK`](crate::mse) 相当の固定チャンク
//! （[`CHUNK`]）で分割し、チャンク内は逐次 `f32` 加算、チャンク間は
//! rayon `par_chunks`（`IndexedParallelIterator` の順序保持契約）で
//! 並列化したのちチャンク番号順に逐次結合する（`mse.rs::mse_sum_sq_f32`
//! と同型）。`Σ` 演算は `diff*diff` のような単純な積和ではなく
//! `ln`／`ln_1p`／`exp` を含む合成式のため `f32::mul_add` は使わない
//! （`.claude/rules/coding-rust.md` の FMA 契約は積和演算〈GEMM〉限定）。
//!
//! backward（`dInput`）は要素独立（アキュムレータなし）の map 演算の
//! ため `par_iter_mut` 並列化で数値へ影響しない（`mse.rs` と同じ理由。
//! MSE backward が導入した要素数しきい値フォールバック
//! ［`MSE_BACKWARD_PARALLEL_MIN_ELEMS`］は #1578 で REJECT 確定〈既定
//! 常に並列〉のため、本モジュールでは同型の機構を新設しない）。

use fandhe_ai_tensor_core::{BackendError, BceKind, ShapeError};
use rayon::prelude::*;

/// [`mse::CHUNK`](crate::mse) と同値の固定チャンクサイズ（forward の
/// 決定的縮約に使う。由来は同モジュール参照）。
const CHUNK: usize = 4096;

/// `bce.rs`／`mse.rs` 共通の要素式（`fandhe_ai_autodiff::eval::
/// bce_elem_loss` と意図的に複製。モジュール doc 参照）。
fn bce_elem_loss(input: f32, target: f32, kind: BceKind) -> f32 {
    match kind {
        BceKind::Probabilities => {
            let log_p = input.ln().max(-100.0);
            let log_1mp = (1.0 - input).ln().max(-100.0);
            -(target * log_p + (1.0 - target) * log_1mp)
        }
        // `Logits`、および `BceKind`（`#[non_exhaustive]`）の未知
        // variant は同じ `Logits` 意味論へ安全側フォールバックする
        // （`fandhe_ai_autodiff::eval::bce_elem_loss` と同型の規律。
        // 本関数は infallible 契約のため `Result` を返せない）。
        kind => {
            debug_assert!(
                matches!(kind, BceKind::Logits),
                "bce::bce_elem_loss: 未知の BceKind variant へフォールバックした（契約違反）"
            );
            input.max(0.0) - input * target + (-input.abs()).exp().ln_1p()
        }
    }
}

/// [`bce_elem_loss`] の `dInput`（`scale` を乗じる前。`fandhe_ai_autodiff::
/// eval::bce_elem_grad_input` と意図的に複製）。
fn bce_elem_grad_input(input: f32, target: f32, kind: BceKind) -> f32 {
    match kind {
        BceKind::Probabilities => {
            let denom = (input * (1.0 - input)).max(1e-12);
            (input - target) / denom
        }
        kind => {
            debug_assert!(
                matches!(kind, BceKind::Logits),
                "bce::bce_elem_grad_input: 未知の BceKind variant へフォールバックした（契約違反）"
            );
            let sigmoid = if input >= 0.0 {
                1.0 / (1.0 + (-input).exp())
            } else {
                let e = input.exp();
                e / (1.0 + e)
            };
            sigmoid - target
        }
    }
}

/// 2 つの長さの一致を検証する（`mse.rs::validate_mse_len` と同じ構成）。
fn validate_bce_len(expected: usize, actual: usize) -> Result<(), BackendError> {
    if expected != actual {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch { expected, actual },
        ));
    }
    Ok(())
}

/// forward: `Σ bce_elem_loss(input[i], target[i], kind)`（縮約前の和の
/// み。`Mean`/`Sum` への変換は呼び出し元 [`crate::ops::CpuBackendOps::
/// bce_loss`] が行う。`mse_sum_sq_f32` と同型）。
///
/// `input`/`target` は呼び出し元が長さ一致を検証済みの contiguous
/// スライスである契約だが、[`validate_bce_len`] で改めて検証する
/// （`mse.rs` と同じ理由。`numel == 0` は `0.0`）。
pub(crate) fn bce_sum_f32(
    input: &[f32],
    target: &[f32],
    kind: BceKind,
) -> Result<f32, BackendError> {
    validate_bce_len(input.len(), target.len())?;
    if input.is_empty() {
        return Ok(0.0);
    }
    let sum = input
        .par_chunks(CHUNK)
        .zip(target.par_chunks(CHUNK))
        .map(|(i_chunk, t_chunk)| {
            i_chunk
                .iter()
                .zip(t_chunk.iter())
                .fold(0.0f32, |acc, (&p, &y)| acc + bce_elem_loss(p, y, kind))
        })
        .collect::<Vec<f32>>()
        .into_iter()
        .fold(0.0f32, |acc, v| acc + v);
    Ok(sum)
}

/// backward: `dInput[i] = scale·bce_elem_grad_input(input[i], target[i],
/// kind)`。`scale` は呼び出し元（`fandhe_ai_autodiff::grad::vjp`）が
/// 上流勾配・`reduction` から事前計算済み（`backend_ops.rs::
/// BackendOps::bce_loss_backward` doc 参照）。`dTarget` は呼び出し元が
/// ホスト側で別途計算する契約（本関数は `dInput` のみを計算する）。
pub(crate) fn bce_loss_backward_f32(
    input: &[f32],
    target: &[f32],
    kind: BceKind,
    scale: f32,
    dinput: &mut [f32],
) -> Result<(), BackendError> {
    validate_bce_len(input.len(), target.len())?;
    validate_bce_len(input.len(), dinput.len())?;
    dinput
        .par_iter_mut()
        .zip(input.par_iter())
        .zip(target.par_iter())
        .for_each(|((o, &p), &y)| *o = scale * bce_elem_grad_input(p, y, kind));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_sum(input: &[f32], target: &[f32], kind: BceKind) -> f32 {
        input
            .iter()
            .zip(target.iter())
            .map(|(&p, &y)| bce_elem_loss(p, y, kind))
            .sum()
    }

    #[test]
    fn bce_sum_f32_probabilities_matches_naive() {
        let input = vec![0.2, 0.8, 0.5, 0.9];
        let target = vec![0.0, 1.0, 1.0, 0.0];
        let got = bce_sum_f32(&input, &target, BceKind::Probabilities).unwrap();
        let want = naive_sum(&input, &target, BceKind::Probabilities);
        assert!((got - want).abs() < 1e-6, "got={got} want={want}");
    }

    #[test]
    fn bce_sum_f32_logits_matches_naive() {
        let input = vec![-3.0, 2.0, 0.0, 5.5];
        let target = vec![0.0, 1.0, 1.0, 0.0];
        let got = bce_sum_f32(&input, &target, BceKind::Logits).unwrap();
        let want = naive_sum(&input, &target, BceKind::Logits);
        assert!((got - want).abs() < 1e-4, "got={got} want={want}");
    }

    #[test]
    fn bce_sum_f32_empty_is_zero() {
        assert_eq!(bce_sum_f32(&[], &[], BceKind::Probabilities).unwrap(), 0.0);
    }

    #[test]
    fn bce_sum_f32_length_mismatch_is_typed_error() {
        let err = bce_sum_f32(&[1.0, 2.0], &[1.0], BceKind::Logits).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn bce_sum_f32_chunk_boundary_is_deterministic() {
        for n in [CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK + 1] {
            let input: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) / 100.0 + 0.001).collect();
            let target: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
            let a = bce_sum_f32(&input, &target, BceKind::Probabilities).unwrap();
            let b = bce_sum_f32(&input, &target, BceKind::Probabilities).unwrap();
            assert_eq!(a.to_bits(), b.to_bits(), "n={n}");
        }
    }

    #[test]
    fn bce_loss_backward_f32_matches_naive() {
        let input = vec![0.2, 0.8, 0.5];
        let target = vec![0.0, 1.0, 1.0];
        let mut dinput = vec![0.0; 3];
        bce_loss_backward_f32(&input, &target, BceKind::Probabilities, 2.0, &mut dinput).unwrap();
        for i in 0..3 {
            let want = 2.0 * bce_elem_grad_input(input[i], target[i], BceKind::Probabilities);
            assert!((dinput[i] - want).abs() < 1e-6, "i={i}");
        }
    }

    #[test]
    fn bce_loss_backward_f32_length_mismatch_is_typed_error() {
        let input = vec![1.0, 2.0];
        let target = vec![0.0];
        let mut dinput = vec![0.0; 2];
        let err =
            bce_loss_backward_f32(&input, &target, BceKind::Logits, 1.0, &mut dinput).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }
}
