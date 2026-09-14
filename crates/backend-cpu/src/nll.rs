//! 負対数尤度損失（`NLLLoss`）の融合カーネル（イシュー #1738・親イシュー
//! #1609「損失関数の拡張」）。
//!
//! `fandhe_ai_tensor_core::BackendOps::nll_loss`／`nll_loss_backward` の
//! CPU 実装として提供する（`mse.rs` と同じ「`ops.rs` の薄い委譲層に
//! 徹する」既存方針）。`fandhe_ai_autodiff::var::Var::nll_loss`／
//! `grad::vjp` の `Op::NllLoss` 分岐から `ops.rs::CpuBackendOps` 経由で
//! 呼ばれる。
//!
//! # 決定性契約
//!
//! [`mse.rs`](crate::mse) と同じ [`CHUNK`] 固定チャンク・rayon
//! `par_chunks`（`IndexedParallelIterator` の順序保持契約）＋チャンク
//! 番号順の逐次結合で forward の総和を計算する。backward は要素独立
//! （アキュムレータなし）の書き込みのみ（ターゲット位置のみ非ゼロ）の
//! ため並列化しても数値へ影響しない。

use fandhe_ai_tensor_core::{BackendError, ShapeError};
use rayon::prelude::*;

/// [`mse::CHUNK`](crate::mse) と同値の固定チャンクサイズ。
const CHUNK: usize = 4096;

/// `input[(o·C+c)·inner+i]` の形状を表す軽量レイアウト
/// （`outer = shape[..class_dim].product()`・`num_classes =
/// shape[class_dim]`・`inner = shape[class_dim+1..].product()`。
/// 呼び出し元 [`crate::ops::CpuBackendOps::nll_loss`]／`nll_loss_backward`
/// が `reduce_out_shape` 相当の検証済み shape から導出する）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct NllLayout {
    pub(crate) outer: usize,
    pub(crate) num_classes: usize,
    pub(crate) inner: usize,
}

impl NllLayout {
    /// サンプル数（`outer * inner`。`targets.numel()` と一致する契約）。
    fn n(&self) -> usize {
        self.outer * self.inner
    }
}

/// forward: `Σ_s −input[(o·C+t_s)·inner+i]`（サンプル `s = o·inner+i`）。
/// `Mean`／`Sum` への変換は呼び出し元 [`crate::ops::CpuBackendOps::
/// nll_loss`] が行う（`mse_sum_sq_f32` と同じ理由。`MseReduction` の
/// `#[non_exhaustive]` を踏まえ `ops.rs` 側で未知 variant を拒否する）。
///
/// `targets` は `0 <= t < num_classes` を呼び出し元
/// （`fandhe_ai_autodiff::var::Var::nll_loss`）が実体化前に検証済みの
/// 契約だが、本関数自身も `debug_assert!` で契約違反を検知しつつ
/// 安全側（寄与 0）へフォールバックする（`.claude/rules/coding-rust.md`
/// 本番経路 panic 禁止方針。`fandhe_ai_autodiff::eval::nll_loss` と同型
/// の縦深防御）。
pub(crate) fn nll_sum_f32(
    input: &[f32],
    targets: &[i32],
    layout: NllLayout,
) -> Result<f32, BackendError> {
    let expected_input_len = layout.outer * layout.num_classes * layout.inner;
    if input.len() != expected_input_len {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: expected_input_len,
                actual: input.len(),
            },
        ));
    }
    let n = layout.n();
    if targets.len() != n {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: n,
                actual: targets.len(),
            },
        ));
    }
    if n == 0 {
        return Ok(0.0);
    }
    let NllLayout {
        num_classes, inner, ..
    } = layout;
    let sum = targets
        .par_chunks(CHUNK)
        .enumerate()
        .map(|(chunk_idx, t_chunk)| {
            let base = chunk_idx * CHUNK;
            let mut acc = 0f32;
            for (offset, &t) in t_chunk.iter().enumerate() {
                let flat = base + offset;
                let o = flat / inner;
                let i = flat % inner;
                if t >= 0 && (t as usize) < num_classes {
                    let idx = (o * num_classes + t as usize) * inner + i;
                    acc -= input[idx];
                } else {
                    debug_assert!(false, "nll_sum_f32: target 添字が範囲外（契約違反）");
                }
            }
            acc
        })
        .collect::<Vec<f32>>()
        .into_iter()
        .fold(0.0f32, |acc, v| acc + v);
    Ok(sum)
}

/// backward: `dInput[(o·C+c)·inner+i] = −scale·1{c == t_s}`。
///
/// `scale` は呼び出し元（`fandhe_ai_autodiff::grad::vjp`）が上流勾配・
/// `reduction` から事前計算済み（`backend_ops.rs::BackendOps::
/// nll_loss_backward` doc 参照）。`dinput` は呼び出し元がゼロ初期化
/// 済みのバッファ（`ops.rs` 参照。ターゲット位置以外は書き込まれず
/// ゼロのまま残る契約）。
pub(crate) fn nll_backward_f32(
    targets: &[i32],
    layout: NllLayout,
    scale: f32,
    dinput: &mut [f32],
) -> Result<(), BackendError> {
    let expected_input_len = layout.outer * layout.num_classes * layout.inner;
    if dinput.len() != expected_input_len {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: expected_input_len,
                actual: dinput.len(),
            },
        ));
    }
    let n = layout.n();
    if targets.len() != n {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: n,
                actual: targets.len(),
            },
        ));
    }
    let NllLayout {
        num_classes, inner, ..
    } = layout;
    // `dinput` は呼び出し元（`ops.rs::CpuBackendOps::nll_loss_backward`）
    // がゼロ初期化して渡す契約（ターゲット位置以外は書き込まず `0.0`
    // のまま）。書き込み対象はサンプル数 `n = outer*inner` 個（高々
    // `numel` の `1/num_classes`）と少なく、各書き込みは独立した
    // ランダムアクセスのため rayon 並列化の恩恵が小さい（`mse.rs` の
    // 全要素 map と異なり疎な書き込み）。`&mut [f32]` を複数スレッドで
    // 安全に分割するには `idx` ごとの排他区間が非自明（`class_dim` が
    // 中間軸のとき `inner` 間隔での分割が必要）なため、`unsafe` を
    // 増やさず素直な逐次ループとする（`.claude/rules/coding-rust.md`
    // 「`unsafe` は FFI 境界等の必要最小限に限る」方針）。
    for (flat, &t) in targets.iter().enumerate() {
        let o = flat / inner;
        let i = flat % inner;
        if t >= 0 && (t as usize) < num_classes {
            let idx = (o * num_classes + t as usize) * inner + i;
            if let Some(slot) = dinput.get_mut(idx) {
                *slot = -scale;
            } else {
                debug_assert!(
                    false,
                    "nll_backward_f32: idx が dinput の範囲外（契約違反）"
                );
            }
        } else {
            debug_assert!(false, "nll_backward_f32: target 添字が範囲外（契約違反）");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nll_sum_f32_matches_naive() {
        // input: [2,3] (outer=2? actually class_dim=1 so layout depends)
        let input = vec![-0.1, -2.0, -1.5, -0.3, -0.2, -3.0];
        let targets = vec![0i32, 1];
        let layout = NllLayout {
            outer: 2,
            num_classes: 3,
            inner: 1,
        };
        let got = nll_sum_f32(&input, &targets, layout).unwrap();
        // l0 = -input[0] = 0.1, l1 = -input[4] = 0.2, sum=0.3
        assert!((got - 0.3).abs() < 1e-6, "got={got}");
    }

    #[test]
    fn nll_sum_f32_empty_is_zero() {
        let layout = NllLayout {
            outer: 0,
            num_classes: 3,
            inner: 1,
        };
        assert_eq!(nll_sum_f32(&[], &[], layout).unwrap(), 0.0);
    }
}
