//! 平均二乗誤差（MSE）の融合カーネル（イシュー #1045・親イシュー #1043
//! 「カーネル融合・autodiff 実行モデルの強化」）。
//!
//! `docs/kernel-fusion.md` 限界表が「reduction 融合はバックエンド実行
//! レベルで未実装」と記録していた対象のうち、MSE（forward の elementwise
//! `(pred−target)²` + 全要素 reduction、backward の解析形勾配）を
//! `fandhe_ai_tensor_core::BackendOps::mse_loss`／`mse_loss_backward` の
//! CPU 実装として提供する。`fandhe_ai_autodiff::var::Var::mse_loss_with`／
//! `grad::vjp` の `Op::MseLoss` 分岐から `ops.rs::CpuBackendOps` 経由で
//! 呼ばれる（`ops.rs` の薄い委譲層に徹する既存方針・モジュール冒頭
//! コメント「CPU バックエンドの `BackendOps` 実装」を踏襲）。
//!
//! # 決定性契約
//!
//! [`reduction`](crate::reduction) モジュールと同じ [`CHUNK`] 固定チャンク
//! （`reduction::CHUNK` と同値。TASK-2.2 の数値一致回帰テストが前提とする
//! 「演算順序を固定した決定的な reduction」契約を forward の 2 乗和にも
//! 適用する）で分割し、チャンク内は逐次 `f32::mul_add` 累積（FMA 契約統一。
//! `.claude/rules/coding-rust.md`）、チャンク間は rayon
//! `par_chunks`（`IndexedParallelIterator` の順序保持契約により
//! スレッド数に依らず bit 決定的。`reduction.rs` モジュール doc 参照）で
//! 並列化したのちチャンク番号順に逐次結合する。
//!
//! backward は要素独立（アキュムレータなし）のため
//! `elementwise` モジュールと同じ `par_iter_mut` 並列化でよい。

use fandhe_ai_tensor_core::{BackendError, ShapeError};
use rayon::prelude::*;

/// [`reduction::CHUNK`](crate::reduction) と同値の固定チャンクサイズ
/// （由来は同モジュール参照。forward の 2 乗和も同じ決定性契約に従う
/// ため、別の値を使う理由がない）。
const CHUNK: usize = 4096;

/// 2 つの長さの一致を検証する（`backend-cuda::mse::
/// validate_mse_binary_len` と同じ構成）。
///
/// [`mse_sum_sq_f32`]／[`mse_loss_backward_f32`] は現状 `pub(crate)` で
/// `ops.rs` の事前検証（`require_same_shape`）を経てのみ呼ばれるが、
/// 将来クレート内の別経路から直接呼ばれた場合や `ops.rs` 側の検証条件が
/// 変更された場合に備え、`assert_eq!`（release ビルドでも消えない panic）
/// ではなく型付きエラーとして長さ不一致を伝播する契約とする（AGENTS.md
/// 「本番経路の panic 禁止」。`backend-cuda`／`backend-metal` の公開
/// MSE API を同じ理由で型付きエラー化した変更〈#1045〉と揃える）。
fn validate_mse_len(expected: usize, actual: usize) -> Result<(), BackendError> {
    if expected != actual {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch { expected, actual },
        ));
    }
    Ok(())
}

/// forward: `Σ(pred[i]−target[i])²`（2 乗和のみ。`Mean`/`Sum` への変換は
/// 呼び出し元 [`crate::ops::CpuBackendOps::mse_loss`] が行う）。
///
/// `reduction` 分岐をここに置かない理由: `MseReduction` は
/// `#[non_exhaustive]`（`backend_ops.rs`。将来 variant 追加時に呼び出し
/// 側の網羅的 match を破壊しない設計）であり、本関数のように `f32` を
/// 返す関数では未知 variant の wildcard 分岐に「安全な既定値」が
/// 存在しない（`Sum` へフォールバックすると誤った値を静かに返す）。
/// `BackendError::Unsupported` を返せる `ops.rs` 側で reduction を
/// 解決することで、未知 variant は型付きエラーとして拒否できる
/// （`.claude/rules/coding-rust.md`「本番経路で unwrap/expect を使わ
/// ない」と同じ「黙って誤った値を返さない」規律）。
///
/// `pred`/`target` は呼び出し元（`ops.rs`）が長さ一致を検証済みの
/// contiguous スライスである契約だが、[`validate_mse_len`] で改めて
/// 検証し不一致は `BackendError::ShapeMismatch` として返す（`assert_eq!`
/// による release ビルドでも消えない panic を `BackendOps` 境界外へ
/// 漏らさないため。rayon `zip` が短い方へ黙って切り詰めて誤った結果を
/// 返す事態も同時に避ける）。`numel == 0` は `0.0`（`Mean`/`Sum` いずれ
/// も空和は数学的に 0。`fandhe_ai_autodiff::eval::mse_loss` と同じ
/// 契約）。
pub(crate) fn mse_sum_sq_f32(pred: &[f32], target: &[f32]) -> Result<f32, BackendError> {
    validate_mse_len(pred.len(), target.len())?;
    if pred.is_empty() {
        return Ok(0.0);
    }
    let sum_sq = pred
        .par_chunks(CHUNK)
        .zip(target.par_chunks(CHUNK))
        .map(|(p_chunk, t_chunk)| {
            p_chunk
                .iter()
                .zip(t_chunk.iter())
                .fold(0.0f32, |acc, (&p, &t)| {
                    let diff = p - t;
                    // FMA 契約統一（`.claude/rules/coding-rust.md`）:
                    // `diff * diff + acc` を 1 回の丸めで計算する。
                    diff.mul_add(diff, acc)
                })
        })
        .collect::<Vec<f32>>()
        .into_iter()
        .fold(0.0f32, |acc, v| acc + v);
    Ok(sum_sq)
}

/// backward: `dPred[i] = scale·(pred[i]−target[i])`。
///
/// `scale` は呼び出し元（`fandhe_ai_autodiff::grad::vjp`）が上流勾配・
/// `reduction` から事前計算済み（`backend_ops.rs::BackendOps::
/// mse_loss_backward` doc 参照）。要素独立のため `elementwise` モジュール
/// と同じ `par_iter_mut` 並列化（順序に依存しない map 演算）でよい。
/// `dTarget = −dPred` は呼び出し元がホスト側で符号反転して得る契約
/// （本関数は `dPred` のみを計算する）。
///
/// 長さ不一致は [`validate_mse_len`]（[`mse_sum_sq_f32`] と同じ理由。
/// `BackendError::ShapeMismatch` として返し、release ビルドでも消えない
/// panic を境界外へ漏らさない。rayon `zip` の黙示切り詰めも同時に防ぐ）
/// で検出する。
pub(crate) fn mse_loss_backward_f32(
    pred: &[f32],
    target: &[f32],
    scale: f32,
    dpred: &mut [f32],
) -> Result<(), BackendError> {
    validate_mse_len(pred.len(), target.len())?;
    validate_mse_len(pred.len(), dpred.len())?;
    dpred
        .par_iter_mut()
        .zip(pred.par_iter())
        .zip(target.par_iter())
        .for_each(|((o, &p), &t)| *o = scale * (p - t));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_sum_sq_f32_matches_naive() {
        let pred = vec![1.0, 2.0, 3.0, 4.0];
        let target = vec![0.0, 0.0, 0.0, 0.0];
        // Σ p² = 1+4+9+16 = 30
        let got = mse_sum_sq_f32(&pred, &target).unwrap();
        assert!((got - 30.0).abs() < 1e-6, "got={got}");
    }

    #[test]
    fn mse_sum_sq_f32_empty_is_zero() {
        assert_eq!(mse_sum_sq_f32(&[], &[]).unwrap(), 0.0);
    }

    #[test]
    fn mse_sum_sq_f32_length_mismatch_is_typed_error() {
        // 契約違反（長さ不一致）は panic ではなく `BackendError` として
        // 返る（AGENTS.md「本番経路の panic 禁止」。イシュー #1045
        // codex-review P1 指摘の再発防止テスト）。
        let err = mse_sum_sq_f32(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn mse_sum_sq_f32_chunk_boundary_is_deterministic() {
        // CHUNK 境界を跨ぐサイズ（4096±1・8193）で複数回計算しても
        // 決定的（bit 一致）であることを固定する（reduction.rs と同じ
        // 決定性契約の再確認）。
        for n in [CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK + 1] {
            let pred: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
            let target: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0005).collect();
            let a = mse_sum_sq_f32(&pred, &target).unwrap();
            let b = mse_sum_sq_f32(&pred, &target).unwrap();
            assert_eq!(a.to_bits(), b.to_bits(), "n={n}");
        }
    }

    #[test]
    fn mse_loss_backward_f32_matches_naive() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![0.0, 1.0, 1.0];
        let mut dpred = vec![0.0; 3];
        mse_loss_backward_f32(&pred, &target, 2.0, &mut dpred).unwrap();
        // scale=2.0 * (pred - target) = [2.0, 2.0, 4.0]
        assert_eq!(dpred, vec![2.0, 2.0, 4.0]);
    }

    #[test]
    fn mse_loss_backward_f32_length_mismatch_is_typed_error() {
        let pred = vec![1.0, 2.0];
        let target = vec![0.0];
        let mut dpred = vec![0.0; 2];
        let err = mse_loss_backward_f32(&pred, &target, 1.0, &mut dpred).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }
}
