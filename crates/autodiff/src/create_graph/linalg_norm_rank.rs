//! `MatrixNorm`（`Fro`・`One`・`Inf`）・`SlogdetSign`・`MatrixRank` の `create_graph` 対応
//! （イシュー #3806・親 #3734・決定記録 `docs/autodiff-higher-order-grad-decision.md` §43・§66）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `validate_ancestors`（入口検査）・`replay_op`
//! （公開メソッドへの自己再生）・`build_cgrads`（1 階 VJP の子テープ記録）から呼ばれる。
//! 構造に積む Op は `supports_create_graph() == true` な Op（`Reshape`・`Contiguous`・`Div`・`Mul`・
//! `Sum`・`Unsqueeze`・`BroadcastTo`・`Where`・`Sqrt`・定数葉）に限り、閉包性（§21.1(4)）を保つ。
//!
//! **方式**
//! - `SlogdetSign`・`MatrixRank`: 1 階 VJP が恒等ゼロ（`grad.rs`）なので、cgrad は入力 shape の
//!   ゼロ定数（[`zero_cgrad`]）。`accumulate` を省かない（省くと `cg.grad` が `None` になり 1 階の
//!   `Some(zeros)` と食い違う）。上流 `g` は非有限でも受理する（1 階が `g` によらずゼロなので bit 一致。
//!   他族が非有限 `g` を拒否するのと逆の扱い）。`g.mul(0)` は `inf·0 = NaN` になるため使わない。
//! - `MatrixNorm`（`Fro`）: 全要素を 1 レーンとする L2 と同型。`stat_varnorm` の L2 構造
//!   （2 のべき乗スケール + `Sqrt` の入力側保護。`StatKind::MatrixFro`）を再利用し、値だけ 1 階と同じ
//!   `eval::linalg::matrix_norm_vjp(Fro)` にする（V3 + S1）。
//! - `MatrixNorm`（`One`・`Inf`）: 値は `matrix_norm_vjp` の結果（先勝ちのタイ・`sign(0) = 0`・
//!   `g < 0` での `-0` まで 1 階と bit 一致）。構造は `g_b ⊙ S`（`S = matrix_norm_vjp(a, ord, 1)` の
//!   符号定数。選択位置では ±1／0、他は 0）で、`A` に関する二階は 0、`g` を経由する分だけが残る
//!   （[`one_inf_cgrad`]）。上流 `g` に非有限要素があれば拒否する（`g·0` を NaN にしないため）。
//! - `Nuc`・`Spectral`: SVD の多出力に依存するため fail-closed のまま（`supports_create_graph` が false）。
//!
//! **拒否条件**: `MatrixNorm` の入力に ±inf がある（NaN は伝播する。`stat_varnorm` の入口検査と共有）、
//! `MatrixRank` で特異値が `rcond` の閾値付近にあり rank が f32 隣接値の範囲で一定でない
//! （[`validate_matrix_rank`]。再生で rank が反転すると子テープの値が親と食い違うため）。
//! 拒否は呼び出し側のロールバックで `child` を空へ戻す。
//!
//! 保証範囲: 受理した入力で `cg.grad` は 1 階と bit 一致（V3／ゼロ定数）し、二階と孫テープは有限差分と
//! REQ-2 判定で一致する。GPU 実機（CUDA・Metal）はホストの `eval::linalg` へフォールバックするため
//! CPU と同じ値になる想定だが、実機の値は未実測（Phase 10。#3857・#3859）。

use fandhe_ai_tensor_core::{MatrixNormOrd, Tensor};

use super::linalg_cholesky_pinv::{next_f32, prev_f32};
use super::linalg_fft::empty_grad;
use super::select_reduce::bcast_upstream;
use super::{child_value, push_v3};
use crate::error::AutodiffError;
use crate::eval::dense_vec;
use crate::eval::linalg::{matrix_norm_vjp, resolve_rcond, svd, truncated_svd_rank};
use crate::tape::Tape;
use crate::var::Var;

fn reject(what: &str) -> AutodiffError {
    AutodiffError::Backward(format!(
        "create_graph: {what}（MatrixNorm／SlogdetSign／MatrixRank の対象外入力。決定記録 §43・§66）"
    ))
}

/// `SlogdetSign`・`MatrixRank` の d_input（1 階 VJP は恒等ゼロ）。入力 shape のゼロ定数を返す。
pub(super) fn zero_cgrad<'c>(
    child: &'c Tape,
    in_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    empty_grad(child, in_shape)
}

/// `MatrixRank` の入口検査（`validate_ancestors` から、子テープへ書く前に呼ばれる）。
///
/// 親の記録 rank と、payload の `rcond` およびその f32 隣接値で数え直した rank がすべて一致する
/// ことを確かめる（特異値が閾値付近にあると、payload の f32 丸めや再生で rank が反転しうる）。
/// 零行列・`rcond = 0` の rank 落ち行列など、区間内で rank が一定な入力は受理する。
pub(super) fn validate_matrix_rank(
    a: &Tensor<f32>,
    recorded_rank: &Tensor<f32>,
    rcond: f32,
) -> Result<(), AutodiffError> {
    let shape = a.shape();
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Ok(());
    }
    let (m, n) = (shape[0], shape[1]);
    let (_u, s, _vh) = svd(a)?;
    let s64: Vec<f64> = dense_vec(&s).into_iter().map(f64::from).collect();
    let rank_at = |r: f32| -> Result<usize, AutodiffError> {
        Ok(truncated_svd_rank(&s64, resolve_rcond(Some(r), m, n)?))
    };
    let mid = rank_at(rcond)?;
    let recorded = dense_vec(recorded_rank).first().copied().unwrap_or(0.0);
    if rank_at(prev_f32(rcond))? != mid
        || rank_at(next_f32(rcond))? != mid
        || recorded != mid as f32
    {
        return Err(reject(
            "MatrixRank: 特異値が rcond の閾値付近にある（再生で rank が反転しうる）",
        ));
    }
    Ok(())
}

/// `MatrixNorm`（`One`・`Inf`）の d_input（V3）。`a_val` は入力値、`g` は上流（shape `[]`）。
/// 入力に ±inf がある場合は呼び出し前の入口検査（`validate_stat_varnorm`）が拒否済み。
pub(super) fn one_inf_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    a_val: &Tensor<f32>,
    ord: MatrixNormOrd,
) -> Result<Var<'c>, AutodiffError> {
    if !matches!(ord, MatrixNormOrd::One | MatrixNormOrd::Inf) {
        return Err(reject("MatrixNorm: One／Inf 以外の ord は対象外"));
    }
    let in_shape = a_val.shape().to_vec();
    if in_shape.contains(&0) {
        return empty_grad(child, &in_shape);
    }
    let g_val = child_value(child, g)?;
    let g_data = dense_vec(&g_val);
    if g_data.len() != 1 || !g_data[0].is_finite() {
        return Err(reject("MatrixNorm: 上流 g が非有限、またはスカラーでない"));
    }
    let host = matrix_norm_vjp(a_val, ord, g_data[0])?;
    // 符号定数 S: g = 1 の 1 階 VJP（選択位置で sign(a)、他は 0。NaN 入力は全要素 NaN）。
    let sign = matrix_norm_vjp(a_val, ord, 1.0)?;
    let s_const = child.var_no_grad(&sign);
    let g_b = bcast_upstream(g, &in_shape, None)?;
    let composed = g_b.mul(&s_const)?;
    push_v3(child, &composed, host)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).unwrap()
    }

    #[test]
    fn rank_threshold_accepts_zero_matrix_and_rcond_zero_rank_drop() {
        let rank0 = mat(vec![0.0], &[]);
        // 零行列: すべての rcond で rank 0。
        validate_matrix_rank(&mat(vec![0.0; 4], &[2, 2]), &rank0, 1e-6).unwrap();
        // rcond = 0・特異値 0 を持つ入力: rank は 1 で一定。
        let rank1 = mat(vec![1.0], &[]);
        validate_matrix_rank(&mat(vec![1.0, 0.0, 0.0, 0.0], &[2, 2]), &rank1, 0.0).unwrap();
    }

    #[test]
    fn rank_threshold_rejects_singular_value_at_threshold() {
        // diag(1, r) で r = rcond（閾値ちょうど）: 隣接値で rank が変わる。
        let rcond = 1e-3f32;
        let a = mat(vec![1.0, 0.0, 0.0, rcond], &[2, 2]);
        let rank = mat(vec![1.0], &[]);
        assert!(validate_matrix_rank(&a, &rank, rcond).is_err());
        // 閾値から十分離れていれば受理。
        let b = mat(vec![1.0, 0.0, 0.0, 0.5], &[2, 2]);
        validate_matrix_rank(&b, &mat(vec![2.0], &[]), rcond).unwrap();
    }
}
