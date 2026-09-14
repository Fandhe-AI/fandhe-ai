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
    /// サンプル数（`outer * inner`。`targets.numel()` と一致する契約）を
    /// `checked_mul` で求める（`backend-cuda::nll::NllLayout::
    /// checked_n_samples` と同型。`outer`／`inner` は呼び出し元
    /// （`ops.rs`）が実shapeから導出するため通常オーバーフローしないが、
    /// 公開 `BackendOps` を直接呼ぶ経路〈`Var::nll_loss` の事前検証を
    /// 経由しない〉に備え、本関数自身も `usize` オーバーフロー時は
    /// `None`（呼び出し元が型付きエラーへ変換）を返す。PR #1850
    /// codex-review P1 是正）。
    fn checked_n(&self) -> Option<usize> {
        self.outer.checked_mul(self.inner)
    }

    /// `outer * num_classes * inner`（`backend-cuda::nll::NllLayout::
    /// checked_numel` と同型。[`checked_n`](Self::checked_n) と同じ
    /// 理由で `checked_mul` を使う）。
    fn checked_numel(&self) -> Option<usize> {
        self.outer
            .checked_mul(self.num_classes)
            .and_then(|v| v.checked_mul(self.inner))
    }
}

/// 寸法積のオーバーフロー（`usize` 範囲超過）を型付きエラーへ変換する
/// （`ops.rs::checked_shape_product` と同じ `ShapeError::
/// ElementCountOverflow` を使い、エラー種別を統一する）。
fn overflow_err() -> BackendError {
    BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
}

/// `targets` の全添字が `0 <= t < num_classes` を満たすことを検証する
/// （`backend-cuda::nll::validate_nll_buffers`／`backend-metal::nll` の
/// 同型関数と同じ理由: `Var::nll_loss` の事前検証は公開 `BackendOps`
/// を直接呼ぶ経路には及ばないため、この委譲層自身が検証しない限り
/// 範囲外添字による境界外アクセスを防げない。範囲外検出時は寄与を
/// 無視する安全側フォールバックではなく型付きエラーで拒否する
/// — `.claude/rules/coding-rust.md` 本番経路 panic 禁止方針。
/// PR #1850 codex-review P1 是正: 従来の `debug_assert!` は release
/// ビルドで契約違反を黙って無視し、debug ビルドでは本番経路の
/// panic を招いていた）。
fn validate_targets(targets: &[i32], num_classes: usize) -> Result<(), BackendError> {
    if let Some(&bad) = targets
        .iter()
        .find(|&&t| t < 0 || (t as usize) >= num_classes)
    {
        return Err(BackendError::InvalidArgument(format!(
            "nll_loss: target index {bad} out of range [0, {num_classes})"
        )));
    }
    Ok(())
}

/// forward: `Σ_s −input[(o·C+t_s)·inner+i]`（サンプル `s = o·inner+i`）。
/// `Mean`／`Sum` への変換は呼び出し元 [`crate::ops::CpuBackendOps::
/// nll_loss`] が行う（`mse_sum_sq_f32` と同じ理由。`MseReduction` の
/// `#[non_exhaustive]` を踏まえ `ops.rs` 側で未知 variant を拒否する）。
///
/// `targets` は `0 <= t < num_classes` を呼び出し元
/// （`fandhe_ai_autodiff::var::Var::nll_loss`）が実体化前に検証済みの
/// 契約だが、公開 `BackendOps` を直接呼ぶ経路にはその事前検証が
/// 及ばないため、本関数自身も [`validate_targets`] で契約違反を
/// 型付きエラーとして拒否する（`.claude/rules/coding-rust.md` 本番
/// 経路 panic 禁止方針。`backend-cuda`／`backend-metal` の同型検証と
/// 同じ縦深防御。PR #1850 codex-review P1 是正: 従来の
/// `debug_assert!` を安全側フォールバックとして残す設計は release
/// ビルドで契約違反を黙って無視し、debug ビルドでは本番経路の panic
/// を招いていた）。
pub(crate) fn nll_sum_f32(
    input: &[f32],
    targets: &[i32],
    layout: NllLayout,
) -> Result<f32, BackendError> {
    let expected_input_len = layout.checked_numel().ok_or_else(overflow_err)?;
    if input.len() != expected_input_len {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: expected_input_len,
                actual: input.len(),
            },
        ));
    }
    let n = layout.checked_n().ok_or_else(overflow_err)?;
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
    validate_targets(targets, layout.num_classes)?;
    let NllLayout {
        num_classes, inner, ..
    } = layout;
    // `validate_targets` により全 `t` が `0 <= t < num_classes` を満たす
    // ことが検証済みのため、以下のループは `debug_assert!` 相当の
    // else 分岐を持たない（`idx` は `expected_input_len` の検証を経た
    // `checked_numel` により `input` の範囲内であることが構造的に
    // 保証される）。
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
                let idx = (o * num_classes + t as usize) * inner + i;
                acc -= input[idx];
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
    let expected_input_len = layout.checked_numel().ok_or_else(overflow_err)?;
    if dinput.len() != expected_input_len {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: expected_input_len,
                actual: dinput.len(),
            },
        ));
    }
    let n = layout.checked_n().ok_or_else(overflow_err)?;
    if targets.len() != n {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: n,
                actual: targets.len(),
            },
        ));
    }
    // `nll_sum_f32` と同じ理由（PR #1850 codex-review P1 是正）:
    // 公開 `BackendOps` を直接呼ぶ経路には `Var::nll_loss` の事前検証が
    // 及ばないため、本関数自身も型付きエラーで範囲外添字を拒否する。
    validate_targets(targets, layout.num_classes)?;
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
    // `validate_targets` により全 `t` が範囲内であることが検証済みの
    // ため、`idx` は `expected_input_len` の検証を経た `checked_numel`
    // により `dinput` の範囲内であることが構造的に保証される
    // （`debug_assert!` else 分岐は不要）。
    for (flat, &t) in targets.iter().enumerate() {
        let o = flat / inner;
        let i = flat % inner;
        let idx = (o * num_classes + t as usize) * inner + i;
        dinput[idx] = -scale;
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

    /// PR #1850 codex-review P1 是正の回帰: 範囲外 `target` を
    /// `debug_assert!` による本番経路 panic ではなく型付きエラー
    /// （`BackendError::InvalidArgument`）で拒否することを確認する
    /// （公開 `BackendOps` を直接呼ぶ経路には `Var::nll_loss` の事前
    /// 検証が及ばないため、この委譲層自身の検証が必須）。
    #[test]
    fn nll_sum_f32_rejects_out_of_range_target() {
        let input = vec![-0.1, -2.0];
        let targets = vec![-1i32];
        let layout = NllLayout {
            outer: 1,
            num_classes: 2,
            inner: 1,
        };
        let err = nll_sum_f32(&input, &targets, layout).unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn nll_backward_f32_rejects_out_of_range_target() {
        let targets = vec![5i32];
        let layout = NllLayout {
            outer: 1,
            num_classes: 2,
            inner: 1,
        };
        let mut dinput = vec![0.0f32; 2];
        let err = nll_backward_f32(&targets, layout, 1.0, &mut dinput).unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)), "{err:?}");
    }

    /// PR #1850 codex-review P1 是正の回帰: `input_shape=[usize::MAX,
    /// 0, 2]` 相当（`outer=usize::MAX`・`num_classes=0`・`inner=2`）の
    /// `layout.checked_n()`（`outer * inner`）オーバーフローを、debug
    /// ビルドの乗算 panic ではなく `ShapeError::ElementCountOverflow`
    /// として返すことを確認する（元の再現条件: `nll_loss_backward` の
    /// `layout.n()` が `usize::MAX * 2` を計算していた）。
    #[test]
    fn nll_backward_f32_rejects_n_overflow() {
        let layout = NllLayout {
            outer: usize::MAX,
            num_classes: 0,
            inner: 2,
        };
        let mut dinput: Vec<f32> = Vec::new();
        let err = nll_backward_f32(&[], layout, 1.0, &mut dinput).unwrap_err();
        assert!(
            matches!(
                err,
                BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
            ),
            "{err:?}"
        );
    }

    #[test]
    fn nll_sum_f32_rejects_numel_overflow() {
        let layout = NllLayout {
            outer: usize::MAX,
            num_classes: 2,
            inner: 2,
        };
        let err = nll_sum_f32(&[], &[], layout).unwrap_err();
        assert!(
            matches!(
                err,
                BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
            ),
            "{err:?}"
        );
    }
}
