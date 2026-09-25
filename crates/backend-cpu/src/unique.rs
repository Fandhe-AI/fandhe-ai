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

use fandhe_ai_tensor_core::{ShapeError, Tensor, UniqueExtOutput};

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

/// [`fandhe_ai_tensor_core::BackendOps::unique_ext`] の CPU 実装本体
/// （イシュー #2153・親 #2131）。`fandhe_ai_autodiff::eval::
/// unique_ext`（`autodiff` クレート非公開のため本クレートから直接は
/// 呼べない）と**意図的に同一アルゴリズムを複製**する（[`unique`] と
/// 同じ理由による複製方針）。契約（群化キー・±0／NaN の扱い・
/// 退化ケース）は [`fandhe_ai_tensor_core::BackendOps::unique_ext`]
/// doc を正とする。
pub fn unique_ext(
    x: &Tensor<f32>,
    dim: Option<usize>,
    consecutive: bool,
) -> Result<UniqueExtOutput, ShapeError> {
    checked_numel(x.shape())?;
    match dim {
        None => {
            let data: Vec<f32> = x.host_slice().into_owned();
            let (values, inverse, counts) = unique_ext_flat(&data, consecutive);
            let in_shape = x.shape().to_vec();
            Ok(UniqueExtOutput {
                values: Tensor::new(values.clone(), &[values.len()])?,
                inverse: Tensor::new(inverse, &in_shape)?,
                counts: Tensor::new(counts.clone(), &[counts.len()])?,
            })
        }
        Some(d) => {
            if d >= x.shape().len() {
                return Err(ShapeError::AxisOutOfRange {
                    axis: d,
                    rank: x.shape().len(),
                });
            }
            let shape = x.shape().to_vec();
            // codex-review P0 是正（PR #2270・イシュー #2153）: shape に
            // 0 長軸が含まれる（numel == 0）場合は `row_major_strides`
            // へ進む前に打ち切る。`unique_ext_slices` は shape 全体の
            // suffix stride 積（`row_major_strides`）と
            // `other_shape.iter().product()` を無条件に計算するため、
            // 例えば `[0, usize::MAX, 2]` のように総積は 0 でも
            // 部分積（`usize::MAX * 2`）が usize を溢れる形状で
            // overflow panic／wrap を起こしうる（Cursor Bugbot 指摘）。
            // さらに `d` 自身が非 0 長軸で他の軸が 0 長のケース
            // （例: `[大軸長, 0]`）では、各スライスが等しく空である
            // ことが自明なので、`axis_len` に比例した `Vec<Vec<f32>>`
            // （`unique_ext_slices`／`unique_ext_group_sorted` が
            // 確保する行配列・ソート添字配列）を構築せず「1 群」へ
            // 直接畳み込むことで過大確保を避ける（codex 指摘）。
            // `axis_len`（`= shape[d]`）自体は呼び出し元
            // `topk_unique_ops::ensure_target_len_fits_i32` が
            // `i32::MAX` 以下であることを事前検査済みの契約
            // （`fandhe_ai_autodiff::grad::unique_ext_with_fallback`
            // doc 参照）のため、`inverse`（`i32`）の確保自体は
            // 契約どおりの出力サイズに収まる。
            if shape.contains(&0) {
                let axis_len = shape[d];
                let m = usize::from(axis_len != 0);
                let mut out_shape = shape.clone();
                out_shape[d] = m;
                // `out_shape` は shape 全体に 0 長軸を含んだまま
                // （`d` 自身が 0 長なら `out_shape[d] == 0`、そうで
                // なければ他の軸に残る 0 長がそのまま残る）なので
                // 要素数積は常に 0 になる。検査は `.product()` では
                // なく `.any()` で行う（`.product()` は左から順に
                // 素朴な乗算で畳み込むため、0 の手前に巨大な値が
                // 複数並ぶ shape では検査対象の debug_assert 自体が
                // overflow しうる——まさに本 P0 是正が避けたい種類の
                // 計算のため、検査側にも持ち込まない）。
                debug_assert!(out_shape.contains(&0));
                let inverse = vec![0i32; axis_len];
                let counts = if m == 1 {
                    vec![axis_len as i32]
                } else {
                    Vec::new()
                };
                return Ok(UniqueExtOutput {
                    values: Tensor::new(Vec::new(), &out_shape)?,
                    inverse: Tensor::new(inverse, &[axis_len])?,
                    counts: Tensor::new(counts, &[m])?,
                });
            }
            let data: Vec<f32> = x.host_slice().into_owned();
            let (rows, other_shape) = unique_ext_slices(&shape, &data, d);
            let (group_of, reps) = if consecutive {
                unique_ext_group_consecutive(&rows)
            } else {
                unique_ext_group_sorted(&rows)
            };
            let m = reps.len();
            let mut counts = vec![0i32; m];
            for &g in &group_of {
                counts[g] += 1;
            }
            let mut out_shape = shape.clone();
            out_shape[d] = m;
            let out_strides = row_major_strides(&out_shape);
            let other_axes: Vec<usize> = (0..shape.len()).filter(|&a| a != d).collect();
            let mut values_data = vec![0f32; out_shape.iter().product()];
            for (g, row) in reps.iter().enumerate() {
                for (o, &val) in row.iter().enumerate() {
                    let other_coords = unravel(o, &other_shape);
                    let mut flat = g * out_strides[d];
                    for (k, &axis) in other_axes.iter().enumerate() {
                        flat += other_coords[k] * out_strides[axis];
                    }
                    values_data[flat] = val;
                }
            }
            let inverse: Vec<i32> = group_of.iter().map(|&g| g as i32).collect();
            Ok(UniqueExtOutput {
                values: Tensor::new(values_data, &out_shape)?,
                inverse: Tensor::new(inverse, &[shape[d]])?,
                counts: Tensor::new(counts, &[m])?,
            })
        }
    }
}

/// [`unique_ext`] の `dim = None` 経路が使うヘルパー
/// （`fandhe_ai_autodiff::eval::unique_ext_flat` の複製）。
fn unique_ext_flat(data: &[f32], consecutive: bool) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let n = data.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let mut values = Vec::new();
    let mut inverse = vec![0i32; n];
    let mut counts: Vec<i32> = Vec::new();
    if consecutive {
        let mut cur = data[0];
        values.push(cur);
        counts.push(1);
        let mut cur_group = 0usize;
        for i in 1..n {
            if data[i] == cur {
                counts[cur_group] += 1;
            } else {
                cur = data[i];
                values.push(cur);
                counts.push(1);
                cur_group += 1;
            }
            inverse[i] = (values.len() - 1) as i32;
        }
    } else {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| data[a].total_cmp(&data[b]).then(a.cmp(&b)));
        let mut prev: Option<f32> = None;
        for &idx in &order {
            let v = data[idx];
            if prev != Some(v) {
                values.push(v);
                counts.push(0);
                prev = Some(v);
            }
            let g = values.len() - 1;
            inverse[idx] = g as i32;
            counts[g] += 1;
        }
    }
    (values, inverse, counts)
}

/// row-major strides を求める（`fandhe_ai_autodiff::eval::
/// row_major_strides` の複製。`pub(crate)` でクレートを跨いで共有
/// できないため複製する）。
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// フラット添字を多次元添字へ展開する（`fandhe_ai_autodiff::eval::
/// unravel` の複製）。
fn unravel(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for i in (0..shape.len()).rev() {
        let dim = shape[i].max(1);
        coords[i] = idx % dim;
        idx /= dim;
    }
    coords
}

/// [`unique_ext`] の `dim = Some(d)` 経路が使うヘルパー
/// （`fandhe_ai_autodiff::eval::unique_ext_slices` の複製）。
fn unique_ext_slices(shape: &[usize], data: &[f32], dim: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let strides = row_major_strides(shape);
    let axis_len = shape[dim];
    let other_axes: Vec<usize> = (0..shape.len()).filter(|&a| a != dim).collect();
    let other_shape: Vec<usize> = other_axes.iter().map(|&a| shape[a]).collect();
    let slice_len: usize = other_shape.iter().product();
    let mut rows = Vec::with_capacity(axis_len);
    for idx in 0..axis_len {
        let mut row = Vec::with_capacity(slice_len);
        for o in 0..slice_len {
            let other_coords = unravel(o, &other_shape);
            let mut flat = idx * strides[dim];
            for (k, &axis) in other_axes.iter().enumerate() {
                flat += other_coords[k] * strides[axis];
            }
            row.push(data[flat]);
        }
        rows.push(row);
    }
    (rows, other_shape)
}

/// 2 行が IEEE `==`（要素ごと）で一致するかを判定する
/// （`fandhe_ai_autodiff::eval::unique_ext_row_eq` の複製）。
fn unique_ext_row_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| x == y)
}

/// 行同士を比較する全順序（`fandhe_ai_autodiff::eval::
/// unique_ext_row_cmp` の複製。2 キー方式は同関数 doc を参照）。
fn unique_ext_row_cmp(a: &[f32], b: &[f32]) -> std::cmp::Ordering {
    let normalize = |v: f32| if v == 0.0 { 0.0 } else { v };
    for (&x, &y) in a.iter().zip(b) {
        let c = normalize(x).total_cmp(&normalize(y));
        if c != std::cmp::Ordering::Equal {
            return c;
        }
    }
    for (&x, &y) in a.iter().zip(b) {
        let c = x.total_cmp(&y);
        if c != std::cmp::Ordering::Equal {
            return c;
        }
    }
    std::cmp::Ordering::Equal
}

/// ソートベースの群化（`fandhe_ai_autodiff::eval::
/// unique_ext_group_sorted` の複製）。
fn unique_ext_group_sorted(rows: &[Vec<f32>]) -> (Vec<usize>, Vec<Vec<f32>>) {
    let n = rows.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| unique_ext_row_cmp(&rows[a], &rows[b]).then(a.cmp(&b)));
    let mut group_of = vec![0usize; n];
    let mut reps: Vec<Vec<f32>> = Vec::new();
    for &idx in &order {
        let is_new = reps
            .last()
            .is_none_or(|last: &Vec<f32>| !unique_ext_row_eq(last, &rows[idx]));
        if is_new {
            reps.push(rows[idx].clone());
        }
        group_of[idx] = reps.len() - 1;
    }
    (group_of, reps)
}

/// 元の並び順のまま隣接行のみ群化する（`fandhe_ai_autodiff::eval::
/// unique_ext_group_consecutive` の複製）。
fn unique_ext_group_consecutive(rows: &[Vec<f32>]) -> (Vec<usize>, Vec<Vec<f32>>) {
    let n = rows.len();
    let mut group_of = vec![0usize; n];
    let mut reps: Vec<Vec<f32>> = Vec::new();
    for idx in 0..n {
        let is_new = idx == 0 || !unique_ext_row_eq(&rows[idx - 1], &rows[idx]);
        if is_new {
            reps.push(rows[idx].clone());
        }
        group_of[idx] = reps.len() - 1;
    }
    (group_of, reps)
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

    /// [`unique_ext`] の `dim=None` 経路が [`unique`] と bit 一致する
    /// `values` を返すこと・`inverse`/`counts` の不変条件を確認する
    /// （イシュー #2153）。
    #[test]
    fn unique_ext_flat_matches_unique_and_reconstructs() {
        let x = t(vec![3.0, 1.0, 2.0, 1.0, 3.0], &[5]);
        let plain = unique(&x).unwrap();
        let out = unique_ext(&x, None, false).unwrap();
        assert_eq!(
            out.values.host_slice().into_owned(),
            plain.host_slice().into_owned()
        );
        assert_eq!(out.inverse.shape(), &[5]);
        assert_eq!(out.counts.shape(), &[3]);
        let values = out.values.host_slice().into_owned();
        let inverse = out.inverse.host_slice().into_owned();
        for (i, &g) in inverse.iter().enumerate() {
            assert_eq!(values[g as usize], x.host_slice()[i]);
        }
        let counts = out.counts.host_slice().into_owned();
        assert_eq!(counts.iter().sum::<i32>(), 5);
    }

    /// `consecutive=true` は元順序のまま隣接要素のみ群化し、代表は
    /// 各ランの先頭出現であることを確認する（イシュー #2153）。
    #[test]
    fn unique_ext_consecutive_groups_adjacent_only() {
        let x = t(vec![1.0, 1.0, 2.0, 1.0, 1.0], &[5]);
        let out = unique_ext(&x, None, true).unwrap();
        assert_eq!(out.values.host_slice().into_owned(), vec![1.0, 2.0, 1.0]);
        assert_eq!(out.inverse.host_slice().into_owned(), vec![0, 0, 1, 2, 2]);
        assert_eq!(out.counts.host_slice().into_owned(), vec![2, 1, 2]);
    }

    /// `dim` 指定時、`[-0.0, 5.0]`／`[+0.0, 5.0]` のような IEEE 等価
    /// スライスが正しく 1 群に畳み込まれることを確認する
    /// （2 キー方式の回帰テスト。イシュー #2153）。
    #[test]
    fn unique_ext_dim_collapses_ieee_equal_rows_with_signed_zero() {
        let neg_zero = -0.0f32;
        let pos_zero = 0.0f32;
        // shape [3, 2]: row0=[-0,5], row1=[-0,7], row2=[+0,5]
        let x = t(vec![neg_zero, 5.0, neg_zero, 7.0, pos_zero, 5.0], &[3, 2]);
        let out = unique_ext(&x, Some(0), false).unwrap();
        assert_eq!(out.values.shape(), &[2, 2]);
        assert_eq!(out.counts.host_slice().into_owned().iter().sum::<i32>(), 3);
        let inverse = out.inverse.host_slice().into_owned();
        assert_eq!(
            inverse[0], inverse[2],
            "row0([-0,5]) と row2([+0,5]) は同じ群"
        );
        assert_ne!(inverse[0], inverse[1]);
    }

    /// `dim` 指定・degenerate ケース（`shape[d]==0`）は `m=0` を返す
    /// ことを確認する（イシュー #2153）。
    #[test]
    fn unique_ext_dim_zero_length_axis_returns_empty() {
        let x = t(Vec::new(), &[0, 3]);
        let out = unique_ext(&x, Some(0), false).unwrap();
        assert_eq!(out.values.shape(), &[0, 3]);
        assert_eq!(out.inverse.shape(), &[0]);
        assert_eq!(out.counts.shape(), &[0]);
    }

    /// PR #2270 codex-review P0 是正の回帰テスト: `d` 自身は非 0 長軸
    /// だが他軸が 0 長（`slice_len == 0`）の場合、全スライスが等しく
    /// 空であるため「1 群」に畳み込まれ、`axis_len` に比例した行配列
    /// を構築せず `inverse`／`counts` が必要量だけ生成されることを
    /// 確認する（`shape = [大軸長, 0]` 型。codex 指摘）。
    #[test]
    fn unique_ext_dim_nonzero_axis_with_other_zero_axis_collapses_to_one_group() {
        let axis_len = 100_000usize;
        let x = t(Vec::new(), &[axis_len, 0]);
        let out = unique_ext(&x, Some(0), false).unwrap();
        assert_eq!(out.values.shape(), &[1, 0]);
        assert_eq!(out.inverse.shape(), &[axis_len]);
        assert!(out.inverse.host_slice().iter().all(|&g| g == 0));
        assert_eq!(out.counts.host_slice().into_owned(), vec![axis_len as i32]);
    }

    /// PR #2270 codex-review Medium 是正の回帰テスト: `shape` の先頭が
    /// 0 長軸で、他軸が `row_major_strides` の suffix 積で usize を
    /// 溢れさせるほど巨大（`[0, usize::MAX, 2]` 型。総積は 0 だが
    /// `usize::MAX * 2` の部分積は overflow する）でも panic せず、
    /// `d` を 0 長軸自身に取れば早期 return で strides 計算自体を
    /// 回避できることを確認する（Cursor Bugbot 指摘）。
    #[test]
    fn unique_ext_dim_leading_zero_axis_with_overflow_prone_suffix_does_not_panic() {
        let x = t(Vec::new(), &[0, usize::MAX, 2]);
        let out = unique_ext(&x, Some(0), false).unwrap();
        assert_eq!(out.values.shape(), &[0, usize::MAX, 2]);
        assert_eq!(out.inverse.shape(), &[0]);
        assert_eq!(out.counts.shape(), &[0]);
    }
}
