//! unique カーネル（`torch.unique(input, sorted=True)` の values のみ。
//! イシュー #1734）。
//!
//! [`fandhe_ai_tensor_core::BackendOps::unique`]（`ops.rs`）の CPU
//! 実装本体。契約（totalOrder ソート・`==` による重複判定・
//! `-0.0`／`+0.0` の集約・NaN 全保持）は
//! [`fandhe_ai_tensor_core::BackendOps::unique`] doc を正とする。
//! `autodiff::eval::unique`（`autodiff` クレート非公開のため本クレート
//! から直接は呼べない）と**意図的に同一アルゴリズムを複製**する
//! （gather／scatter の先例 `gather_scatter.rs` 冒頭コメントと同じ
//! 方針）。
//!
//! `Tensor::host_slice()` で stride 対応の稠密化を行ってから読むため
//! 非 contiguous な入力（transpose 済み view 等）も正しく扱える。
//! 並列化しない（決定的・単純。将来の性能最適化は
//! `.claude/rules/out-of-scope-tracking.md` 対象）。`unsafe` は
//! 使わず、`Tensor::new` の失敗は `ShapeError` として型付きで返す
//! （本番経路 `unwrap()`/`expect()` 禁止。`.claude/rules/
//! coding-rust.md`）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

/// shape の要素数積を `checked_mul` の畳み込みで検査する
/// （`gather_scatter.rs::checked_numel` と同型の独立実装。
/// `pub(crate)` でクレートを跨いで共有できないため複製する。
/// PR #1828 codex-review P1 是正: `transpose` 済みの非 contiguous
/// view は `x.shape()` の各軸積が `usize` 範囲を超えないことを
/// `Tensor::new`/`transpose` 単体では保証しないため（`shape.swap`
/// は要素数積を再検査しない）、`x.numel()`（内部で `.product()`
/// を使い overflow-checks 有効時に panic しうる）を呼ぶ前に本関数で
/// 事前検査し、型付きエラーとして返す。
fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// [`fandhe_ai_tensor_core::BackendOps::unique`] の CPU 実装本体。
///
/// 空入力（`numel == 0`）は shape `[0]` を返す。それ以外は
/// `host_slice()` で稠密化 → `f32::total_cmp`（IEEE 754 totalOrder）
/// でソート → `==`（IEEE 比較）で隣接重複を除去する。
pub fn unique(x: &Tensor<f32>) -> Result<Tensor<f32>, ShapeError> {
    // `x.numel()`／`x.host_slice()`（非 contiguous 時は `contiguous()`
    // 経由で `numel()` を再度使う）を呼ぶ前に要素数積のオーバーフロー
    // を検査する（PR #1828 codex-review P1 是正）。
    checked_numel(x.shape())?;
    if x.numel() == 0 {
        return Tensor::new(Vec::new(), &[0]);
    }
    let mut v: Vec<f32> = x.host_slice().into_owned();
    v.sort_unstable_by(f32::total_cmp);
    v.dedup_by(|cur, prev| *cur == *prev);
    let m = v.len();
    Tensor::new(v, &[m])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
    }

    #[test]
    fn sorts_and_dedups() {
        let x = t(vec![3.0, 1.0, 2.0, 1.0, 3.0], &[5]);
        let out = unique(&x).unwrap();
        assert_eq!(out.shape(), &[3]);
        assert_eq!(out.host_slice().into_owned(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn negative_and_positive_zero_collapse_to_negative_zero() {
        let x = t(vec![0.0, -0.0, 1.0], &[3]);
        let out = unique(&x).unwrap();
        assert_eq!(out.shape(), &[2]);
        let data = out.host_slice().into_owned();
        assert_eq!(data[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(data[1], 1.0);
    }

    #[test]
    fn nan_values_are_all_preserved() {
        let nan1 = f32::NAN;
        let nan2 = f32::from_bits(f32::NAN.to_bits() | 1);
        let x = t(vec![nan1, 1.0, nan2], &[3]);
        let out = unique(&x).unwrap();
        assert_eq!(out.shape(), &[3]);
    }

    #[test]
    fn empty_input_returns_shape_zero() {
        let x = t(Vec::new(), &[2, 0]);
        let out = unique(&x).unwrap();
        assert_eq!(out.shape(), &[0]);
    }

    #[test]
    fn non_contiguous_input_matches_contiguous() {
        let base = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let transposed = base
            .permute(&[1, 0])
            .expect("test fixture: rank 2 の permute は常に妥当");
        let out = unique(&transposed).unwrap();
        assert_eq!(out.shape(), &[6]);
        assert_eq!(
            out.host_slice().into_owned(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn single_element_input() {
        let x = t(vec![7.0], &[1]);
        let out = unique(&x).unwrap();
        assert_eq!(out.shape(), &[1]);
        assert_eq!(out.host_slice().into_owned(), vec![7.0]);
    }

    /// PR #1828 codex-review P1 是正の回帰テスト:
    /// `Tensor::new(Vec::new(), &[0, 2, usize::MAX])` は要素数積が
    /// `0` のため構築に成功するが、`transpose(0, 2)` で軸順を
    /// `[usize::MAX, 2, 0]` へ入れ替えると `x.numel()` 内部の
    /// `.product()` の評価順序が変わり `usize::MAX * 2` が先に評価
    /// される。`unique` が `x.numel()`/`contiguous()` を呼ぶ前に
    /// 要素数積のオーバーフローを検査し、`panic!` ではなく
    /// `ShapeError::ElementCountOverflow` を返すことを確認する。
    #[test]
    fn transposed_zero_element_shape_does_not_overflow_panic() {
        let x = Tensor::new(Vec::<f32>::new(), &[0, 2, usize::MAX])
            .expect("要素数積は 0 のため構築は成功する契約");
        let transposed = x
            .transpose(0, 2)
            .expect("rank 3 の transpose(0, 2) は常に妥当");
        assert_eq!(transposed.shape(), &[usize::MAX, 2, 0]);
        let err = unique(&transposed).unwrap_err();
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }
}
