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
//! backward は要素独立（アキュムレータなし）の map 演算であり、
//! `elementwise` モジュールと同じ理由で `par_iter_mut` 並列化が数値へ
//! 影響しない（結合則の影響を受ける加算・乗算の跨りがないため）。
//! ただし framework-compare の `train`（`BATCH=64 × D_OUT=10 = 640`
//! 要素。`scripts/bench/framework-compare/bench-fandhe/src/main.rs`）
//! 規模では rayon の fork-join 固定費が支配的になることを低レイヤー
//! 診断（`docs/perf/lowlayer-diagnosis-2026-09-12.md` §4・§7「A-11」・
//! イシュー #1574）が実測しており、[`elementwise::PARALLEL_THRESHOLD`]
//! と同型の要素数しきい値フォールバックを [`MSE_BACKWARD_PARALLEL_MIN_ELEMS`]
//! として導入する（イシュー #1578。実測記録・事前登録規則は
//! `docs/perf/cpu-mse-backward-sequential-threshold.md`）。

use fandhe_ai_tensor_core::{BackendError, ShapeError};
use rayon::prelude::*;

/// [`reduction::CHUNK`](crate::reduction) と同値の固定チャンクサイズ
/// （由来は同モジュール参照。forward の 2 乗和も同じ決定性契約に従う
/// ため、別の値を使う理由がない）。
const CHUNK: usize = 4096;

/// [`mse_loss_backward_f32`] が逐次ループへフォールバックする要素数の
/// しきい値（この値**未満**は逐次）。`crate::elementwise::PARALLEL_THRESHOLD`
/// と同じ「rayon fork-join 固定費 対 実作業」のトレードオフだが、MSE
/// backward 特有の実測（イシュー #1578・`docs/perf/
/// cpu-mse-backward-sequential-threshold.md` の Phase 0）で個別に決定
/// した値のため別定数として持つ（elementwise 側の値と揃うとは限らない）。
///
/// 出荷時の既定は Phase 0／Phase 1 の実測判定（ADOPT／REJECT）に従う。
/// REJECT の場合は `0`（常に並列＝変更前と bit 同一の挙動）とし、
/// 機構自体は残す（`ops.rs::GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS` を
/// `usize::MAX` で無効化する慣行〈#1299/#1482〉の逆向き）。
///
/// Phase 0 実測（M4 Max・GB10（DGX Spark）各 5 プロセス起動。
/// `docs/perf/cpu-mse-backward-sequential-threshold.md` §5）では、
/// スイープ上限 `1 << 18`（262144 要素）までの全サイズ・全 run で
/// 逐次が並列を一貫して上回った（`r(n) = seq/par` が一度も 1.00 を
/// 超えなかった）ため、事前登録規則の「見つからなければ `1 << 18`」を
/// 適用し候補 `T = 1 << 18` を得た。しかし Phase 1（framework-compare
/// train A/B・事前登録規則）では GB10 の reuse セルが `ratio = 1.0167`
/// （> 1.00）となり **REJECT** と確定した（M4 Max は `ratio = 0.9163`
/// で非後退・両機体とも checksum 完全一致）。事前登録規則は事後緩和
/// しない契約のため、GB10 の後退 1 件で全体を REJECT とし、既定値は
/// `0`（常に並列＝変更前と bit 同一の挙動）へ確定する（機構自体は
/// 残す。詳細・原因分析は同 doc §6・§7）。
pub(crate) const MSE_BACKWARD_PARALLEL_MIN_ELEMS: usize = 0;

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
///
/// 本番経路は [`MSE_BACKWARD_PARALLEL_MIN_ELEMS`] を既定しきい値として
/// 使う薄いラッパー（[`mse_loss_backward_f32`]）。計測・テストからは
/// [`mse_loss_backward_f32_with_threshold`] を直接呼び、しきい値を差し
/// 替えて逐次／並列の両腕を比較できるようにする。
pub(crate) fn mse_loss_backward_f32(
    pred: &[f32],
    target: &[f32],
    scale: f32,
    dpred: &mut [f32],
) -> Result<(), BackendError> {
    mse_loss_backward_f32_with_threshold(
        pred,
        target,
        scale,
        dpred,
        MSE_BACKWARD_PARALLEL_MIN_ELEMS,
    )
}

/// [`mse_loss_backward_f32`] の本体。`min_elems` を明示的に受け取り、
/// `pred.len() < min_elems` なら逐次ループへフォールバックする
/// （rayon の fork-join 固定費が実作業を上回る小規模入力向け。
/// イシュー #1578）。
///
/// # bit 同一契約
///
/// 逐次分岐は並列分岐と**同一の式**（`scale * (p - t)`）を使う。両分岐
/// とも要素独立（アキュムレータなし）の map 演算のため丸めの発生源が
/// 存在せず、構成上 bit 同一になる（`.claude/rules/coding-rust.md` の
/// FMA 契約は積和演算〈GEMM〉限定で本関数の elementwise 差分には
/// 適用外。`elementwise.rs` モジュール doc と同じ判断）。
/// `#[cfg(test)]` の `mse_loss_backward_threshold_bit_exact` で
/// forced-seq／forced-par／素朴ループの 3 者一致を検証する。
pub(crate) fn mse_loss_backward_f32_with_threshold(
    pred: &[f32],
    target: &[f32],
    scale: f32,
    dpred: &mut [f32],
    min_elems: usize,
) -> Result<(), BackendError> {
    validate_mse_len(pred.len(), target.len())?;
    validate_mse_len(pred.len(), dpred.len())?;
    if pred.len() < min_elems {
        for ((o, &p), &t) in dpred.iter_mut().zip(pred.iter()).zip(target.iter()) {
            *o = scale * (p - t);
        }
    } else {
        dpred
            .par_iter_mut()
            .zip(pred.par_iter())
            .zip(target.par_iter())
            .for_each(|((o, &p), &t)| *o = scale * (p - t));
    }
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

    #[test]
    fn mse_loss_backward_with_threshold_length_mismatch_both_arms() {
        // 長さ不一致検証は逐次・並列いずれの分岐に入る前でも行われる
        // ことを固定する（forced-seq / forced-par 双方）。
        let pred = vec![1.0, 2.0];
        let target = vec![0.0];
        let mut dpred = vec![0.0; 2];
        for min_elems in [0usize, usize::MAX] {
            let err =
                mse_loss_backward_f32_with_threshold(&pred, &target, 1.0, &mut dpred, min_elems)
                    .unwrap_err();
            assert!(
                matches!(err, BackendError::ShapeMismatch(_)),
                "min_elems={min_elems}"
            );
        }
    }

    /// 素朴な逐次リファレンス実装（rayon 非使用）。forced-seq / forced-par
    /// と bit 単位で突き合わせる正の根拠とする。
    fn naive_backward(pred: &[f32], target: &[f32], scale: f32) -> Vec<f32> {
        pred.iter()
            .zip(target.iter())
            .map(|(&p, &t)| scale * (p - t))
            .collect()
    }

    #[test]
    fn mse_loss_backward_threshold_bit_exact() {
        // forced-seq（min_elems = usize::MAX）・forced-par（min_elems = 0）・
        // 素朴ループの 3 者が bit 単位で完全一致することを、NaN・-0.0・
        // subnormal・±inf・scale 負値/0 を含む入力で検証する（イシュー
        // #1578 の bit 同一契約）。
        let sizes = [0usize, 1, 2, 639, 640, 641, 32767, 32768, 32769];
        for &n in &sizes {
            let pred: Vec<f32> = (0..n)
                .map(|i| match i % 7 {
                    0 => f32::NAN,
                    1 => -0.0,
                    2 => f32::MIN_POSITIVE * 0.5, // subnormal
                    3 => f32::INFINITY,
                    4 => f32::NEG_INFINITY,
                    _ => (i as f32) * 0.0001 - 3.0,
                })
                .collect();
            let target: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0002 - 1.5).collect();

            for &scale in &[1.0f32, -2.5, 0.0] {
                let mut seq = vec![0.0f32; n];
                let mut par = vec![0.0f32; n];
                mse_loss_backward_f32_with_threshold(&pred, &target, scale, &mut seq, usize::MAX)
                    .unwrap();
                mse_loss_backward_f32_with_threshold(&pred, &target, scale, &mut par, 0).unwrap();
                let naive = naive_backward(&pred, &target, scale);

                for i in 0..n {
                    assert_eq!(
                        seq[i].to_bits(),
                        par[i].to_bits(),
                        "n={n} scale={scale} i={i} seq vs par mismatch (NaN 込みのため to_bits 比較)"
                    );
                    assert_eq!(
                        seq[i].to_bits(),
                        naive[i].to_bits(),
                        "n={n} scale={scale} i={i} seq vs naive mismatch"
                    );
                }
            }
        }
    }

    #[test]
    fn mse_loss_backward_default_threshold_is_expected() {
        // 既定ラッパー（mse_loss_backward_f32）が MSE_BACKWARD_PARALLEL_MIN_ELEMS
        // を使う `_with_threshold` 呼び出しと bit 一致することを固定し、
        // ラッパー結線のドリフトを検出する。
        let n = 4096;
        let pred: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let target: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0005).collect();
        let mut via_wrapper = vec![0.0f32; n];
        let mut via_explicit = vec![0.0f32; n];
        mse_loss_backward_f32(&pred, &target, 3.0, &mut via_wrapper).unwrap();
        mse_loss_backward_f32_with_threshold(
            &pred,
            &target,
            3.0,
            &mut via_explicit,
            MSE_BACKWARD_PARALLEL_MIN_ELEMS,
        )
        .unwrap();
        for i in 0..n {
            assert_eq!(via_wrapper[i].to_bits(), via_explicit[i].to_bits(), "i={i}");
        }
    }

    /// Phase 0 マイクロベンチ（イシュー #1578・事前登録規則。
    /// `docs/perf/cpu-mse-backward-sequential-threshold.md` §5 参照）。
    /// forced-seq / forced-par の中央値（ns）をサイズごとに出力する。
    /// 実機（CI 非対象）での手動計測用のため `#[ignore]`。
    #[test]
    #[ignore]
    fn mse_backward_threshold_sweep() {
        use std::time::Instant;

        const SIZES: &[usize] = &[640, 2560, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
        const WARMUP: usize = 50;
        const ITERS: usize = 1000;

        eprintln!("threads={}", rayon::current_num_threads());

        for &n in SIZES {
            let pred: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0001).collect();
            let target: Vec<f32> = (0..n).map(|i| (i as f32) * 0.00005).collect();
            let mut dpred = vec![0.0f32; n];

            for (arm, min_elems) in [("seq", usize::MAX), ("par", 0usize)] {
                for _ in 0..WARMUP {
                    mse_loss_backward_f32_with_threshold(
                        &pred, &target, 1.0, &mut dpred, min_elems,
                    )
                    .unwrap();
                }
                let mut samples = Vec::with_capacity(ITERS);
                let first_start = Instant::now();
                mse_loss_backward_f32_with_threshold(&pred, &target, 1.0, &mut dpred, min_elems)
                    .unwrap();
                let first_call_ns = first_start.elapsed().as_nanos();
                for _ in 0..ITERS {
                    let start = Instant::now();
                    mse_loss_backward_f32_with_threshold(
                        &pred, &target, 1.0, &mut dpred, min_elems,
                    )
                    .unwrap();
                    samples.push(start.elapsed().as_nanos());
                }
                samples.sort_unstable();
                let median_ns = samples[samples.len() / 2];
                let checksum: u64 = dpred.iter().map(|v| v.to_bits() as u64).sum();
                eprintln!(
                    "n={n} arm={arm} median_ns={median_ns} first_call_ns={first_call_ns} threads={} checksum={checksum:#x}",
                    rayon::current_num_threads()
                );
            }
        }
    }
}
