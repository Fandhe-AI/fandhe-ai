//! Kullback-Leibler ダイバージェンス損失（`KLDivLoss`）の融合カーネル
//! （イシュー #1738・親イシュー #1609「損失関数の拡張」）。
//!
//! `fandhe_ai_tensor_core::BackendOps::kl_div_loss`／
//! `kl_div_loss_backward` の CPU 実装として提供する（`mse.rs`／`nll.rs`
//! と同じ「`ops.rs` の薄い委譲層に徹する」既存方針）。
//! `fandhe_ai_autodiff::var::Var::kl_div_loss`／
//! `kl_div_loss_with_log_target`・`grad::vjp` の `Op::KlDivLoss` 分岐から
//! `ops.rs::CpuBackendOps` 経由で呼ばれる。
//!
//! forward の要素損失和は [`mse::CHUNK`](crate::mse) 相当の固定チャンク
//! （[`CHUNK`]）で分割し、チャンク内は逐次 `f32` 加算、チャンク間は
//! rayon `par_chunks`（`IndexedParallelIterator` の順序保持契約）で
//! 並列化したのちチャンク番号順に逐次結合する（`mse.rs::mse_sum_sq_f32`
//! と同型）。`ln`／`exp` を含む合成式のため `f32::mul_add` は使わない
//! （`.claude/rules/coding-rust.md` の FMA 契約は積和演算〈GEMM〉限定）。
//!
//! backward（`dInput`）は要素独立（アキュムレータなし）の map 演算の
//! ため `par_iter_mut` 並列化で数値へ影響しない（`mse.rs`／`bce.rs` と
//! 同じ理由）。

use fandhe_ai_tensor_core::{BackendError, KlDivTarget, ShapeError};
use rayon::prelude::*;

/// [`mse::CHUNK`](crate::mse) と同値の固定チャンクサイズ。
const CHUNK: usize = 4096;

/// `mse.rs`／`bce.rs` 共通の要素式（`fandhe_ai_autodiff::eval::
/// kl_div_elem_loss` と意図的に複製。モジュール doc 参照）。
fn kl_div_elem_loss(input: f32, target: f32, kind: KlDivTarget) -> f32 {
    match kind {
        KlDivTarget::Probabilities => {
            if target == 0.0 {
                0.0
            } else {
                target * (target.ln() - input)
            }
        }
        // `LogProbabilities`、および `KlDivTarget`（`#[non_exhaustive]`）
        // の未知 variant は同じ `LogProbabilities` 意味論へ安全側
        // フォールバックする（`fandhe_ai_autodiff::eval::
        // kl_div_elem_loss` と同型の規律。本関数は infallible 契約の
        // ため `Result` を返せない）。
        kind => {
            debug_assert!(
                matches!(kind, KlDivTarget::LogProbabilities),
                "kl_div::kl_div_elem_loss: 未知の KlDivTarget variant へフォールバックした\
                 （契約違反）"
            );
            target.exp() * (target - input)
        }
    }
}

/// [`kl_div_elem_loss`] の `dInput`（`scale` を乗じる前。
/// `fandhe_ai_autodiff::eval::kl_div_elem_grad_input` と意図的に複製）。
fn kl_div_elem_grad_input(_input: f32, target: f32, kind: KlDivTarget) -> f32 {
    match kind {
        KlDivTarget::Probabilities => -target,
        kind => {
            debug_assert!(
                matches!(kind, KlDivTarget::LogProbabilities),
                "kl_div::kl_div_elem_grad_input: 未知の KlDivTarget variant へフォールバック\
                 した（契約違反）"
            );
            -target.exp()
        }
    }
}

/// 2 つの長さの一致を検証する（`mse.rs::validate_mse_len` と同じ構成）。
fn validate_kl_div_len(expected: usize, actual: usize) -> Result<(), BackendError> {
    if expected != actual {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch { expected, actual },
        ));
    }
    Ok(())
}

/// forward: `Σ kl_div_elem_loss(input[i], target[i], kind)`（縮約前の
/// 和のみ。`Mean`/`Sum` への変換は呼び出し元 [`crate::ops::
/// CpuBackendOps::kl_div_loss`] が行う。`mse_sum_sq_f32` と同型）。
///
/// `input`/`target` は呼び出し元が長さ一致を検証済みの contiguous
/// スライスである契約だが、[`validate_kl_div_len`] で改めて検証する
/// （`mse.rs` と同じ理由。`numel == 0` は `0.0`）。
pub(crate) fn kl_div_sum_f32(
    input: &[f32],
    target: &[f32],
    kind: KlDivTarget,
) -> Result<f32, BackendError> {
    validate_kl_div_len(input.len(), target.len())?;
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
                .fold(0.0f32, |acc, (&x, &tv)| acc + kl_div_elem_loss(x, tv, kind))
        })
        .collect::<Vec<f32>>()
        .into_iter()
        .fold(0.0f32, |acc, v| acc + v);
    Ok(sum)
}

/// backward: `dInput[i] = scale·kl_div_elem_grad_input(input[i],
/// target[i], kind)`。`scale` は呼び出し元（`fandhe_ai_autodiff::
/// grad::vjp`）が上流勾配・`reduction` から事前計算済み
/// （`backend_ops.rs::BackendOps::kl_div_loss_backward` doc 参照）。
/// `dTarget` は呼び出し元がホスト側で別途計算する契約（本関数は
/// `dInput` のみを計算する）。
pub(crate) fn kl_div_loss_backward_f32(
    input: &[f32],
    target: &[f32],
    kind: KlDivTarget,
    scale: f32,
    dinput: &mut [f32],
) -> Result<(), BackendError> {
    validate_kl_div_len(input.len(), target.len())?;
    validate_kl_div_len(input.len(), dinput.len())?;
    dinput
        .par_iter_mut()
        .zip(input.par_iter())
        .zip(target.par_iter())
        .for_each(|((o, &x), &tv)| *o = scale * kl_div_elem_grad_input(x, tv, kind));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_sum(input: &[f32], target: &[f32], kind: KlDivTarget) -> f32 {
        input
            .iter()
            .zip(target.iter())
            .map(|(&x, &tv)| kl_div_elem_loss(x, tv, kind))
            .sum()
    }

    #[test]
    fn kl_div_sum_f32_probabilities_matches_naive() {
        let input = vec![-2.0, -0.5, -1.2, -0.1];
        let target = vec![0.2, 0.8, 0.5, 0.5];
        let got = kl_div_sum_f32(&input, &target, KlDivTarget::Probabilities).unwrap();
        let want = naive_sum(&input, &target, KlDivTarget::Probabilities);
        assert!((got - want).abs() < 1e-6, "got={got} want={want}");
    }

    #[test]
    fn kl_div_sum_f32_log_target_matches_naive() {
        let input = vec![-2.0, -0.5];
        let target = vec![-1.6, -0.2];
        let got = kl_div_sum_f32(&input, &target, KlDivTarget::LogProbabilities).unwrap();
        let want = naive_sum(&input, &target, KlDivTarget::LogProbabilities);
        assert!((got - want).abs() < 1e-4, "got={got} want={want}");
    }

    #[test]
    fn kl_div_sum_f32_target_zero_contributes_zero() {
        let input = vec![-2.0];
        let target = vec![0.0];
        let got = kl_div_sum_f32(&input, &target, KlDivTarget::Probabilities).unwrap();
        assert_eq!(got, 0.0);
    }

    #[test]
    fn kl_div_sum_f32_empty_is_zero() {
        assert_eq!(
            kl_div_sum_f32(&[], &[], KlDivTarget::Probabilities).unwrap(),
            0.0
        );
    }

    #[test]
    fn kl_div_sum_f32_length_mismatch_is_typed_error() {
        let err = kl_div_sum_f32(&[1.0, 2.0], &[1.0], KlDivTarget::Probabilities).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }
}
