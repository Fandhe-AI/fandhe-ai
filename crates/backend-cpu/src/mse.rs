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

use rayon::prelude::*;

/// [`reduction::CHUNK`](crate::reduction) と同値の固定チャンクサイズ
/// （由来は同モジュール参照。forward の 2 乗和も同じ決定性契約に従う
/// ため、別の値を使う理由がない）。
const CHUNK: usize = 4096;

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
/// contiguous スライス（`assert_eq!` で契約違反を検出する。
/// `elementwise.rs` の `add_slice` 等と同方針: release ビルドでも
/// 消えない検査とし、rayon `zip` が短い方へ黙って切り詰めて誤った
/// 結果を返す事態を避ける）。`numel == 0` は `0.0`（`Mean`/`Sum` いずれ
/// も空和は数学的に 0。`fandhe_ai_autodiff::eval::mse_loss` と同じ
/// 契約）。
pub(crate) fn mse_sum_sq_f32(pred: &[f32], target: &[f32]) -> f32 {
    assert_eq!(
        pred.len(),
        target.len(),
        "mse_sum_sq_f32: length mismatch（呼び出し元 ops.rs が事前検証する契約）"
    );
    if pred.is_empty() {
        return 0.0;
    }
    pred.par_chunks(CHUNK)
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
        .fold(0.0f32, |acc, v| acc + v)
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
/// 長さ不一致は `assert_eq!`（[`mse_sum_sq_f32`] と同じ理由。release
/// ビルドでも rayon `zip` の黙示切り詰めを起こさせない）で検出する。
pub(crate) fn mse_loss_backward_f32(pred: &[f32], target: &[f32], scale: f32, dpred: &mut [f32]) {
    assert_eq!(
        pred.len(),
        target.len(),
        "mse_loss_backward_f32: length mismatch（pred vs target）"
    );
    assert_eq!(
        pred.len(),
        dpred.len(),
        "mse_loss_backward_f32: length mismatch（pred vs dpred）"
    );
    dpred
        .par_iter_mut()
        .zip(pred.par_iter())
        .zip(target.par_iter())
        .for_each(|((o, &p), &t)| *o = scale * (p - t));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_sum_sq_f32_matches_naive() {
        let pred = vec![1.0, 2.0, 3.0, 4.0];
        let target = vec![0.0, 0.0, 0.0, 0.0];
        // Σ p² = 1+4+9+16 = 30
        let got = mse_sum_sq_f32(&pred, &target);
        assert!((got - 30.0).abs() < 1e-6, "got={got}");
    }

    #[test]
    fn mse_sum_sq_f32_empty_is_zero() {
        assert_eq!(mse_sum_sq_f32(&[], &[]), 0.0);
    }

    #[test]
    fn mse_sum_sq_f32_chunk_boundary_is_deterministic() {
        // CHUNK 境界を跨ぐサイズ（4096±1・8193）で複数回計算しても
        // 決定的（bit 一致）であることを固定する（reduction.rs と同じ
        // 決定性契約の再確認）。
        for n in [CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK + 1] {
            let pred: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
            let target: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0005).collect();
            let a = mse_sum_sq_f32(&pred, &target);
            let b = mse_sum_sq_f32(&pred, &target);
            assert_eq!(a.to_bits(), b.to_bits(), "n={n}");
        }
    }

    #[test]
    fn mse_loss_backward_f32_matches_naive() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![0.0, 1.0, 1.0];
        let mut dpred = vec![0.0; 3];
        mse_loss_backward_f32(&pred, &target, 2.0, &mut dpred);
        // scale=2.0 * (pred - target) = [2.0, 2.0, 4.0]
        assert_eq!(dpred, vec![2.0, 2.0, 4.0]);
    }
}
