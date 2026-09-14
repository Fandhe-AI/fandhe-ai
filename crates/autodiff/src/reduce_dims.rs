//! 複数軸・`keepdim` 対応の縮約（`Var::sum_dims`／`max_dims`／
//! `mean_dims`。イシュー #1719・親 #1601「Phase 2（Tier 1）」）が
//! 使う、軸検証と「複数軸を 1 軸へ併合してから単一軸縮約へ委譲する」
//! 計画の純関数部（`Var` を一切操作しない。`crate::einsum` の
//! 「検証と Var 操作の分離」規律と同じ）。
//!
//! **併合方式を選ぶ理由（逐次〈軸ごと〉縮約にしない理由）**:
//! - `f64` アキュムレータ（`backend-cpu::reduction::axis_reduce_sum`）
//!   による蓄積が軸をまたいで 1 パス・丸め 1 回に閉じる
//!   （`.claude/rules/coding-rust.md` の長軸縮約契約と整合）。
//! - `max_dims` の同値タイが縮約対象全要素を 1 回の `max_vjp` 呼び出し
//!   で見るため、`grad.rs::max_vjp` の「先勝ち決定的」規約
//!   （イシュー #1718 は本イシュー時点で未確定のため変更しない）が
//!   軸をまたいでもそのまま成立する。タイの順序は「kept 軸（元の順序）
//!   → reduced 軸（昇順）」の併合順で最初に現れる要素。
//!
//! **実現方式**: 縮約対象軸が 1 個・全軸のいずれかなら、既存
//! `Var::sum(Option<usize>)`／`max(Option<usize>)` の単一軸／全軸経路へ
//! そのまま委譲する（`sum_dims(&[d], false)` が `sum(Some(d))` と
//! **bit 同一**になる契約はこの直接委譲で自動的に満たされる）。
//! それ以外（複数軸・非全軸）は `Var::permute`（kept 軸 → reduced 軸の
//! 順）→ `Var::contiguous`（非 contiguous のときのみ実体化コピー。
//! 既に contiguous な並びなら no-op）→ `Var::reshape`（reduced 軸を
//! 1 軸へ併合）で 1 軸縮約へ帰着させる。

use std::collections::HashSet;

use fandhe_ai_tensor_core::ShapeError;

use crate::error::AutodiffError;
use crate::var::Var;

/// [`plan_reduce_dims`] が算出する、複数軸縮約の検証済み計画（純データ。
/// `Var` を一切保持しない）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReducePlan {
    /// 温存される軸（元 shape での軸番号・元の順序のまま昇順）。
    pub(crate) kept_axes: Vec<usize>,
    /// 縮約対象軸（昇順）。
    pub(crate) reduced_axes: Vec<usize>,
    /// `kept_axes ++ reduced_axes` が恒等順列（`[0, 1, ..., rank-1]`）
    /// と一致するか。一致するなら `Var::permute` の呼び出し自体を
    /// 省略できる（`merge_for_reduction` が参照）。
    pub(crate) perm_needed: bool,
    /// 併合後（kept 軸を温存し reduced 軸を 1 軸へ併合した）shape。
    /// 縮約対象軸が単一・全軸のいずれかの場合は使用しない
    /// （`merge_for_reduction` がその場合は生成しない）ため空のまま。
    pub(crate) merged_shape: Vec<usize>,
    /// `keepdim=true` のときの最終 shape（元の rank を保持し、縮約対象
    /// 軸のみサイズ 1 に置き換えたもの）。
    pub(crate) keepdim_shape: Vec<usize>,
}

/// `dims` を検証・正規化し、[`ReducePlan`] を算出する（純関数）。
///
/// 検査順序（fail-closed。`.claude/rules/security.md` A03）:
/// ①`dims` が空 → [`AutodiffError::InvalidArgument`]（PyTorch
/// `dim=()` の歴史的曖昧さを持ち込まない。全軸縮約は全軸を明示列挙
/// するか `sum(None)`／`max(None)`／`mean(None)` を使う）。②各軸が
/// range 内か → [`ShapeError::AxisOutOfRange`]。③重複軸 →
/// [`AutodiffError::InvalidArgument`]。④縮約対象要素数の積を
/// `checked_mul` で検査 → [`ShapeError::ElementCountOverflow`]
/// （REQ-8 趣旨の境界検査）。
pub(crate) fn plan_reduce_dims(
    shape: &[usize],
    dims: &[usize],
) -> Result<ReducePlan, AutodiffError> {
    let rank = shape.len();
    if dims.is_empty() {
        return Err(AutodiffError::InvalidArgument(
            "reduce_dims: dims が空です（全軸縮約は全軸を明示列挙するか \
             dim=None の単軸 API を使ってください）"
                .to_string(),
        ));
    }

    let mut sorted_dims = dims.to_vec();
    sorted_dims.sort_unstable();
    for axis in &sorted_dims {
        if *axis >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: *axis,
                rank,
            }));
        }
    }
    for w in sorted_dims.windows(2) {
        if w[0] == w[1] {
            return Err(AutodiffError::InvalidArgument(format!(
                "reduce_dims: dims に重複した軸 {} があります",
                w[0]
            )));
        }
    }

    let reduced_set: HashSet<usize> = sorted_dims.iter().copied().collect();
    let kept_axes: Vec<usize> = (0..rank).filter(|a| !reduced_set.contains(a)).collect();
    let reduced_axes = sorted_dims;

    // 縮約対象要素数の積（`Var::reshape` へ渡す併合後 shape の最終軸に
    // なる値でもある）。オーバーフロー検査は `Var::reshape` 自身も行う
    // が、こちら側でも事前検査して契約を明示する（`crate::var::Var::
    // reshape` の `checked_mul` 検査と同じ規律）。
    let count = reduced_axes
        .iter()
        .try_fold(1usize, |acc, &a| acc.checked_mul(shape[a]))
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;

    let perm: Vec<usize> = kept_axes
        .iter()
        .copied()
        .chain(reduced_axes.iter().copied())
        .collect();
    let perm_needed = perm != (0..rank).collect::<Vec<usize>>();

    let kept_sizes: Vec<usize> = kept_axes.iter().map(|&a| shape[a]).collect();
    let merged_shape: Vec<usize> = if reduced_axes.len() == rank || reduced_axes.len() == 1 {
        // 全軸・単一軸は `merge_for_reduction` が既存 `sum(None)`／
        // `sum(Some(d))` へ直接委譲するため併合 shape を使わない。
        Vec::new()
    } else {
        let mut m = kept_sizes;
        m.push(count);
        m
    };

    let mut keepdim_shape = shape.to_vec();
    for &axis in &reduced_axes {
        keepdim_shape[axis] = 1;
    }

    Ok(ReducePlan {
        kept_axes,
        reduced_axes,
        perm_needed,
        merged_shape,
        keepdim_shape,
    })
}

/// [`ReducePlan`] に従って `v` を「単一軸縮約（`Var::sum`／`max`／
/// `mean` の `Option<usize>` 引数）へそのまま渡せる形」へ整形する
/// （副作用あり。tape へノードを push しうる）。戻り値の第 2 要素が
/// そのまま `sum`/`max`/`mean` の `dim` 引数になる。
///
/// - 縮約対象軸が全軸 → `(v, None)`（`v` は無加工。呼び出し側が
///   `v.sum(None)` 等を呼べば全軸縮約になる）。
/// - 縮約対象軸が単一 → `(v, Some(axis))`（無加工。
///   `sum_dims(&[d], false)` が `sum(Some(d))` と bit 同一になる契約は
///   この分岐が担う）。
/// - それ以外 → `kept_axes ++ reduced_axes` の順に `permute`
///   （`perm_needed` が `false` なら省略）→ `contiguous`（非
///   contiguous のときのみ実体化）→ `reshape`（reduced 軸を 1 軸へ
///   併合）した `v'` と `Some(kept_axes.len())` を返す。
pub(crate) fn merge_for_reduction<'t>(
    v: Var<'t>,
    plan: &ReducePlan,
) -> Result<(Var<'t>, Option<usize>), AutodiffError> {
    let rank = plan.kept_axes.len() + plan.reduced_axes.len();
    // 単一軸判定を全軸判定より先に評価する（rank=1 のとき両条件が
    // 同時に成立しうるため）。公開 API 契約「`sum_dims(&[d], false)`
    // は `sum(Some(d))` と bit 同一」を守るには単一軸経路を優先する
    // 必要がある（全軸経路 `sum(None)` は CPU 実装上 4096 要素単位の
    // 部分和結合を行うため、逐次加算の単一軸経路と数値的に異なりうる）。
    if plan.reduced_axes.len() == 1 {
        return Ok((v, Some(plan.reduced_axes[0])));
    }
    if plan.reduced_axes.len() == rank {
        return Ok((v, None));
    }
    let perm: Vec<usize> = plan
        .kept_axes
        .iter()
        .copied()
        .chain(plan.reduced_axes.iter().copied())
        .collect();
    let v = if plan.perm_needed {
        v.permute(&perm)?
    } else {
        v
    };
    let v = v.contiguous()?;
    let v = v.reshape(&plan.merged_shape)?;
    Ok((v, Some(plan.kept_axes.len())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_rejects_empty_dims() {
        let err = plan_reduce_dims(&[2, 3, 4], &[]).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn plan_rejects_out_of_range_axis() {
        let err = plan_reduce_dims(&[2, 3, 4], &[3]).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::AxisOutOfRange { axis: 3, rank: 3 })
        ));
    }

    #[test]
    fn plan_rejects_duplicate_axis() {
        let err = plan_reduce_dims(&[2, 3, 4], &[1, 1]).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn plan_rejects_overflowing_count() {
        let err = plan_reduce_dims(&[usize::MAX, 2], &[0, 1]).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn plan_single_axis_has_no_merged_shape() {
        let plan = plan_reduce_dims(&[2, 3, 4], &[1]).unwrap();
        assert_eq!(plan.reduced_axes, vec![1]);
        assert_eq!(plan.kept_axes, vec![0, 2]);
        assert!(plan.merged_shape.is_empty());
        assert_eq!(plan.keepdim_shape, vec![2, 1, 4]);
    }

    #[test]
    fn plan_full_axes_has_no_merged_shape() {
        let plan = plan_reduce_dims(&[2, 3], &[0, 1]).unwrap();
        assert_eq!(plan.reduced_axes, vec![0, 1]);
        assert!(plan.kept_axes.is_empty());
        assert!(plan.merged_shape.is_empty());
        assert_eq!(plan.keepdim_shape, vec![1, 1]);
    }

    #[test]
    fn plan_non_trailing_dims_need_perm() {
        // dims=[1,2] は shape=[2,3,4,5] の kept=[0,3] が縮約軸の前後を
        // 挟む典型ケースで、kept ++ reduced = [0,3,1,2] は恒等順列
        // ではないため perm が必要。
        let plan = plan_reduce_dims(&[2, 3, 4, 5], &[1, 2]).unwrap();
        assert_eq!(plan.kept_axes, vec![0, 3]);
        assert_eq!(plan.reduced_axes, vec![1, 2]);
        assert!(plan.perm_needed);
        assert_eq!(plan.merged_shape, vec![2, 5, 12]);
    }

    #[test]
    fn plan_true_trailing_dims_no_perm_needed() {
        // dims が本当に末尾（kept=[0,1] がそのまま先頭）なら
        // kept ++ reduced == 恒等順列で perm 不要。
        let plan = plan_reduce_dims(&[2, 3, 4, 5], &[2, 3]).unwrap();
        assert_eq!(plan.kept_axes, vec![0, 1]);
        assert_eq!(plan.reduced_axes, vec![2, 3]);
        assert!(!plan.perm_needed);
        assert_eq!(plan.merged_shape, vec![2, 3, 20]);
    }

    #[test]
    fn plan_keepdim_shape_preserves_rank_with_ones() {
        let plan = plan_reduce_dims(&[2, 3, 4], &[0, 2]).unwrap();
        assert_eq!(plan.keepdim_shape, vec![1, 3, 1]);
        assert_eq!(plan.kept_axes, vec![1]);
        assert_eq!(plan.reduced_axes, vec![0, 2]);
    }

    #[test]
    fn plan_dims_order_does_not_matter() {
        let a = plan_reduce_dims(&[2, 3, 4], &[2, 0]).unwrap();
        let b = plan_reduce_dims(&[2, 3, 4], &[0, 2]).unwrap();
        assert_eq!(a, b);
    }
}
