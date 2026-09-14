//! Huber／SmoothL1 損失の融合カーネル（イシュー #1739）。
//!
//! `docs/kernel-fusion.md` 限界表が「reduction 融合はバックエンド実行
//! レベルで未実装」と記録していた対象のうち、Huber／SmoothL1
//! （forward の elementwise 区分損失 `l(pred−target)` + 全要素
//! reduction、backward の解析形勾配）を
//! `fandhe_ai_tensor_core::BackendOps::huber_loss`／`huber_loss_backward`
//! の CPU 実装として提供する（`mse.rs`〈イシュー #1045〉と同型の融合
//! パターン）。`fandhe_ai_autodiff::var::Var::huber_loss_impl`／
//! `grad::vjp` の `Op::HuberLoss` 分岐から `ops.rs::CpuBackendOps` 経由で
//! 呼ばれる（`ops.rs` の薄い委譲層に徹する既存方針）。
//!
//! # 決定性契約
//!
//! [`CHUNK`] 固定チャンク（`mse::CHUNK`／`reduction::CHUNK` と同値）で
//! 分割し、チャンク内は逐次加算、チャンク間は rayon
//! `par_chunks`（`IndexedParallelIterator` の順序保持契約により
//! スレッド数に依らず bit 決定的）で並列化したのちチャンク番号順に
//! 逐次結合する（`mse::mse_sum_sq_f32` と同じ構成。積和ではない区分
//! 損失のため `mul_add` によるチャンク内 FMA 契約は適用しない——
//! `.claude/rules/coding-rust.md` の FMA 契約は積和演算〈GEMM〉限定）。
//!
//! backward は要素独立（アキュムレータなし）の map 演算であり、
//! `mse::mse_loss_backward_f32` と同じ理由で `par_iter_mut` 並列化が
//! 数値へ影響しない。`mse` の [`crate::mse::MSE_BACKWARD_PARALLEL_MIN_ELEMS`]
//! と同型の要素数しきい値フォールバック機構は導入しない（イシュー
//! #1578 は MSE backward 固有の実測に基づく判断であり、本イシューの
//! スコープでは個別実測を行わない。実装計画 §2.2）。

use fandhe_ai_tensor_core::{BackendError, HuberKind, ShapeError};
use rayon::prelude::*;

/// [`crate::mse::CHUNK`] と同値の固定チャンクサイズ（由来は同モジュール
/// 参照。forward の区分損失和も同じ決定性契約に従うため、別の値を使う
/// 理由がない）。
const CHUNK: usize = 4096;

/// 要素損失 `l(d)`（`d = pred − target`）。
/// `fandhe_ai_autodiff::eval::huber_elem_loss` と同一の意味論（意味論の
/// 正はホスト参照実装側。本関数はその CPU 融合カーネル側の複製——
/// `mse.rs`／`eval::mse_loss` が独立複製する関係と同型）。
///
/// | kind | `\|d\| < delta` | それ以外 |
/// |---|---|---|
/// | `Huber` | `0.5·d²` | `delta·(\|d\| − 0.5·delta)` |
/// | `SmoothL1` | `0.5·d²/delta` | `\|d\| − 0.5·delta` |
///
/// `#[non_exhaustive]` な `HuberKind` の未知 variant は `Huber` 意味論へ
/// 安全側フォールバックする（`eval::huber_elem_loss` と同型の判断）。
#[inline]
fn elem_loss(d: f32, kind: HuberKind, delta: f32) -> f32 {
    let abs_d = d.abs();
    match kind {
        HuberKind::SmoothL1 => {
            if abs_d < delta {
                // `d*d` を先に計算すると delta・d が巨大な有限値の
                // ときに中間積が overflow しうる（例: d=1e20,
                // delta=2e20 で `d*d`=1e40 は f32 の表現範囲外）。
                // `abs_d < delta` 分岐内では `|d/delta| < 1` が
                // 保証されるため、先に delta で割ってから d を
                // 掛けることで中間値を `|d|` 以下に抑える
                // （`eval::huber_elem_loss`・CUDA `kernels_huber.rs`・
                // Metal `shaders/huber.metal` と同じ演算順序で揃える）。
                0.5 * (d / delta) * d
            } else {
                abs_d - 0.5 * delta
            }
        }
        _ => {
            if abs_d < delta {
                0.5 * d * d
            } else {
                delta * (abs_d - 0.5 * delta)
            }
        }
    }
}

/// [`elem_loss`] の `pred` に対する要素勾配 `∂l/∂pred`（`scale` 乗算前）。
/// `fandhe_ai_autodiff::eval::huber_elem_grad` と同一の意味論。
///
/// | kind | `\|d\| < delta` | それ以外 |
/// |---|---|---|
/// | `Huber` | `d` | `copysign(delta, d)` |
/// | `SmoothL1` | `d/delta` | `copysign(1, d)` |
#[inline]
fn elem_grad(d: f32, kind: HuberKind, delta: f32) -> f32 {
    let abs_d = d.abs();
    match kind {
        HuberKind::SmoothL1 => {
            if abs_d < delta {
                d / delta
            } else {
                1.0f32.copysign(d)
            }
        }
        _ => {
            if abs_d < delta {
                d
            } else {
                delta.copysign(d)
            }
        }
    }
}

/// 2 つの長さの一致を検証する（`crate::mse::validate_mse_len` と同じ
/// 構成・同じ理由: `assert_eq!` ではなく型付きエラーとして長さ不一致を
/// 伝播する）。
fn validate_huber_len(expected: usize, actual: usize) -> Result<(), BackendError> {
    if expected != actual {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch { expected, actual },
        ));
    }
    Ok(())
}

/// forward: `Σ l(pred[i]−target[i])`（区分損失和のみ。`Mean`/`Sum` への
/// 変換は呼び出し元 [`crate::ops::CpuBackendOps::huber_loss`] が行う。
/// `crate::mse::mse_sum_sq_f32` と同型の構成・同じ理由で `reduction`
/// 分岐をここに置かない）。
///
/// `pred`/`target` は呼び出し元（`ops.rs`）が長さ一致を検証済みの
/// contiguous スライスである契約だが、[`validate_huber_len`] で改めて
/// 検証し不一致は `BackendError::ShapeMismatch` として返す（`mse.rs` と
/// 同じ理由）。`numel == 0` は `0.0`（`Mean`/`Sum` いずれも空和は数学的
/// に 0。`fandhe_ai_autodiff::eval::huber_loss` と同じ契約）。
pub(crate) fn huber_sum_f32(
    pred: &[f32],
    target: &[f32],
    kind: HuberKind,
    delta: f32,
) -> Result<f32, BackendError> {
    validate_huber_len(pred.len(), target.len())?;
    if pred.is_empty() {
        return Ok(0.0);
    }
    let sum = pred
        .par_chunks(CHUNK)
        .zip(target.par_chunks(CHUNK))
        .map(|(p_chunk, t_chunk)| {
            p_chunk
                .iter()
                .zip(t_chunk.iter())
                .fold(0.0f32, |acc, (&p, &t)| acc + elem_loss(p - t, kind, delta))
        })
        .collect::<Vec<f32>>()
        .into_iter()
        .fold(0.0f32, |acc, v| acc + v);
    Ok(sum)
}

/// backward: `dPred[i] = scale·grad_elem(pred[i]−target[i])`。
///
/// `scale` は呼び出し元（`fandhe_ai_autodiff::grad::vjp`）が上流勾配・
/// `reduction` から事前計算済み（`backend_ops.rs::BackendOps::
/// huber_loss_backward` doc 参照）。要素独立のため `par_iter_mut`
/// 並列化（順序に依存しない map 演算。`mse::mse_loss_backward_f32` と
/// 同じ理由）でよい。`dTarget = −dPred` は呼び出し元がホスト側で符号
/// 反転して得る契約（本関数は `dPred` のみを計算する）。
///
/// 長さ不一致は [`validate_huber_len`] で検出する（`mse.rs` と同じ
/// 理由）。
pub(crate) fn huber_loss_backward_f32(
    pred: &[f32],
    target: &[f32],
    kind: HuberKind,
    delta: f32,
    scale: f32,
    dpred: &mut [f32],
) -> Result<(), BackendError> {
    validate_huber_len(pred.len(), target.len())?;
    validate_huber_len(pred.len(), dpred.len())?;
    dpred
        .par_iter_mut()
        .zip(pred.par_iter())
        .zip(target.par_iter())
        .for_each(|((o, &p), &t)| *o = scale * elem_grad(p - t, kind, delta));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_sum(pred: &[f32], target: &[f32], kind: HuberKind, delta: f32) -> f32 {
        pred.iter()
            .zip(target.iter())
            .map(|(&p, &t)| elem_loss(p - t, kind, delta))
            .sum()
    }

    #[test]
    fn huber_sum_f32_matches_naive_huber() {
        let pred = vec![0.0, 1.5, -3.0, 0.25];
        let target = vec![0.0, 0.0, 0.0, 0.0];
        let got = huber_sum_f32(&pred, &target, HuberKind::Huber, 1.0).unwrap();
        let expect = naive_sum(&pred, &target, HuberKind::Huber, 1.0);
        assert!((got - expect).abs() < 1e-6, "got={got} expect={expect}");
        // 実装計画 §5.2 手計算参照値。
        assert!((got - 3.53125).abs() < 1e-6, "got={got}");
    }

    #[test]
    fn huber_sum_f32_matches_naive_smooth_l1() {
        let pred = vec![0.0, 1.5, -3.0, 0.25];
        let target = vec![0.0, 0.0, 0.0, 0.0];
        let got = huber_sum_f32(&pred, &target, HuberKind::SmoothL1, 2.0).unwrap();
        let expect = naive_sum(&pred, &target, HuberKind::SmoothL1, 2.0);
        assert!((got - expect).abs() < 1e-6, "got={got} expect={expect}");
        // 実装計画 §5.2 手計算参照値。
        assert!((got - 2.578125).abs() < 1e-6, "got={got}");
    }

    #[test]
    fn huber_sum_f32_empty_is_zero() {
        assert_eq!(huber_sum_f32(&[], &[], HuberKind::Huber, 1.0).unwrap(), 0.0);
    }

    #[test]
    fn huber_sum_f32_length_mismatch_is_typed_error() {
        let err = huber_sum_f32(&[1.0, 2.0], &[1.0], HuberKind::Huber, 1.0).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn huber_sum_f32_chunk_boundary_is_deterministic() {
        // `CHUNK` 境界をまたぐ大規模入力でチャンク分割の有無に依らず
        // 同一値になることを確認する（`mse.rs` の同種テストと同型）。
        let n = CHUNK * 2 + 137;
        let pred: Vec<f32> = (0..n).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let target: Vec<f32> = (0..n).map(|i| (i as f32 % 5.0) - 2.0).collect();
        let a = huber_sum_f32(&pred, &target, HuberKind::Huber, 1.0).unwrap();
        let b = huber_sum_f32(&pred, &target, HuberKind::Huber, 1.0).unwrap();
        assert_eq!(a.to_bits(), b.to_bits(), "run-to-run bit 同一");
    }

    #[test]
    fn huber_loss_backward_f32_matches_naive() {
        let pred = vec![1.0, -1.0, 0.5, -0.75];
        let target = vec![0.0, 0.0, 0.0, 0.0];
        let scale = 0.25f32;
        let mut dpred = vec![0.0f32; 4];
        huber_loss_backward_f32(&pred, &target, HuberKind::Huber, 1.0, scale, &mut dpred).unwrap();
        let expected: Vec<f32> = pred
            .iter()
            .zip(target.iter())
            .map(|(&p, &t)| scale * elem_grad(p - t, HuberKind::Huber, 1.0))
            .collect();
        assert_eq!(dpred, expected);
    }

    #[test]
    fn huber_loss_backward_f32_length_mismatch_is_typed_error() {
        let mut dpred = vec![0.0f32; 1];
        let err =
            huber_loss_backward_f32(&[1.0, 2.0], &[1.0], HuberKind::Huber, 1.0, 1.0, &mut dpred)
                .unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn huber_loss_backward_f32_bit_exact_across_runs() {
        // 要素独立の map 演算のため丸めの発生源がなく run-to-run bit
        // 同一になる（`mse.rs` の bit 同一契約テストと同型）。
        let n = CHUNK * 2 + 33;
        let pred: Vec<f32> = (0..n).map(|i| (i as f32 % 11.0) - 5.0).collect();
        let target: Vec<f32> = (0..n).map(|i| (i as f32 % 9.0) - 4.0).collect();
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        huber_loss_backward_f32(&pred, &target, HuberKind::SmoothL1, 1.5, 0.5, &mut a).unwrap();
        huber_loss_backward_f32(&pred, &target, HuberKind::SmoothL1, 1.5, 0.5, &mut b).unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
}
