//! MaxUnpool1d／2d／3d（`F.max_unpool1d/2d/3d` 相当）の形状・サイズ検査（イシュー #2644・親
//! #2625「Phase 4」・ルート #2499）。
//!
//! 役割: `autodiff::max_unpool_ops::max_unpool{1,2,3}d` の入口が、実体化・確保・tape 操作より前に
//! 「rank・索引 shape・kernel／stride の 0・既定出力長・`output_size` の許容範囲・確保サイズ」を
//! 検査し、`(n, c)` 平面へ平坦化した scatter 用のレイアウト（[`MaxUnpoolLayout`]）を得るための
//! 純関数を提供する。**形状・サイズ検査のみ**を担い、索引値の範囲検査（負値・`>= 出力平面長`）は
//! autodiff 入口が行う（`AutodiffError::InvalidArgument`）。1d／2d／3d の差は空間 rank だけで、
//! 本体は共通。
//!
//! **PyTorch 相当**: 既定出力長 `(in − 1)·stride − 2·padding + kernel`。`output_size`（空間軸のみ）は
//! PyTorch 2.14.0 の実測どおり各軸 `default − stride < size < default + stride`（開区間）を許容する
//! （`docs/autodiff-conv-transpose3d-max-unpool-decision.md` §5）。`size == 0` は拒否する。

use crate::error::ShapeError;
use crate::tensor::checked_numel_for;

/// MaxUnpool の入出力レイアウト。`(n, c)` 平面ごとに独立な scatter（入力平面長 `L_in` →
/// 出力平面長 `L_out`）として扱うための寸法を保持する。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaxUnpoolLayout {
    n: usize,
    c: usize,
    in_spatial: Vec<usize>,
    out_spatial: Vec<usize>,
    nc: usize,
    in_plane: usize,
    out_plane: usize,
}

impl MaxUnpoolLayout {
    /// 入力 shape `[N, C, 空間...]`。
    pub fn in_shape(&self) -> Vec<usize> {
        let mut s = vec![self.n, self.c];
        s.extend_from_slice(&self.in_spatial);
        s
    }

    /// 出力 shape `[N, C, 空間...]`。
    pub fn out_shape(&self) -> Vec<usize> {
        let mut s = vec![self.n, self.c];
        s.extend_from_slice(&self.out_spatial);
        s
    }

    /// 入力平面長 `L_in`（空間軸の積）。
    pub fn in_plane(&self) -> usize {
        self.in_plane
    }

    /// 出力平面長 `L_out`（空間軸の積）。索引値の有効範囲は `[0, L_out)`。
    pub fn out_plane(&self) -> usize {
        self.out_plane
    }

    /// 入力を `(n, c)` 平面へ平坦化した shape `[N·C, L_in]`。
    pub fn flat_in(&self) -> [usize; 2] {
        [self.nc, self.in_plane]
    }

    /// 出力を `(n, c)` 平面へ平坦化した shape `[N·C, L_out]`。
    pub fn flat_out(&self) -> [usize; 2] {
        [self.nc, self.out_plane]
    }
}

fn mismatch(lhs: &[usize], rhs: &[usize]) -> ShapeError {
    ShapeError::ShapeMismatch {
        lhs: lhs.to_vec(),
        rhs: rhs.to_vec(),
    }
}

fn checked_product(dims: &[usize]) -> Result<usize, ShapeError> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// MaxUnpool の入出力レイアウトを検査・導出する。
///
/// 空間 rank は `kernel_size.len()`（1〜3）。`stride = None` は `kernel_size`。検査順序: 空間 rank
/// と引数長の一致 → 入力 rank（空間 rank + 2）→ `index_shape == input_shape` → kernel／stride の 0
/// 拒否 → 入力の空間軸 0 拒否（`N`／`C == 0` は受理して空出力）→ 軸ごとの既定出力長
/// （`checked_*`・結果 1 以上）→ `output_size` の範囲検査 → `N·C`・平面長の `checked_mul` →
/// 入出力の確保サイズ検査。
pub fn max_unpool_layout(
    input_shape: &[usize],
    index_shape: &[usize],
    kernel_size: &[usize],
    stride: Option<&[usize]>,
    padding: &[usize],
    output_size: Option<&[usize]>,
) -> Result<MaxUnpoolLayout, ShapeError> {
    let rank = kernel_size.len();
    if !(1..=3).contains(&rank) {
        return Err(ShapeError::RankMismatch {
            expected: 3,
            actual: rank,
        });
    }
    let stride_v: Vec<usize> = match stride {
        Some(s) => s.to_vec(),
        None => kernel_size.to_vec(),
    };
    if stride_v.len() != rank {
        return Err(mismatch(&[stride_v.len()], &[rank]));
    }
    if padding.len() != rank {
        return Err(mismatch(&[padding.len()], &[rank]));
    }
    if let Some(os) = output_size
        && os.len() != rank
    {
        return Err(mismatch(&[os.len()], &[rank]));
    }
    if input_shape.len() != rank + 2 {
        return Err(ShapeError::RankMismatch {
            expected: rank + 2,
            actual: input_shape.len(),
        });
    }
    if index_shape != input_shape {
        return Err(mismatch(index_shape, input_shape));
    }
    if kernel_size.contains(&0) {
        return Err(mismatch(kernel_size, &vec![1; rank]));
    }
    if stride_v.contains(&0) {
        return Err(mismatch(&stride_v, &vec![1; rank]));
    }
    let in_spatial = input_shape[2..].to_vec();
    if in_spatial.contains(&0) {
        return Err(mismatch(&in_spatial, &vec![1; rank]));
    }
    let mut out_spatial = Vec::with_capacity(rank);
    for a in 0..rank {
        // 既定出力長 (in − 1)·stride − 2·padding + kernel（checked。結果 1 以上）。
        let base = (in_spatial[a] - 1)
            .checked_mul(stride_v[a])
            .and_then(|v| v.checked_add(kernel_size[a]))
            .ok_or(ShapeError::ElementCountOverflow)?;
        let two_p = padding[a]
            .checked_mul(2)
            .ok_or(ShapeError::ElementCountOverflow)?;
        let default_len = match base.checked_sub(two_p) {
            Some(v) if v >= 1 => v,
            _ => return Err(mismatch(&[base], &[two_p])),
        };
        let len = match output_size {
            None => default_len,
            Some(os) => {
                let size = os[a];
                // PyTorch 2.14.0 実測: default − stride < size < default + stride（開区間）。
                let upper = default_len
                    .checked_add(stride_v[a])
                    .ok_or(ShapeError::ElementCountOverflow)?;
                let lower_ok = size
                    .checked_add(stride_v[a])
                    .ok_or(ShapeError::ElementCountOverflow)?
                    > default_len;
                if size == 0 || !lower_ok || size >= upper {
                    return Err(mismatch(&[size], &[default_len]));
                }
                size
            }
        };
        out_spatial.push(len);
    }
    let n = input_shape[0];
    let c = input_shape[1];
    let nc = n.checked_mul(c).ok_or(ShapeError::ElementCountOverflow)?;
    let in_plane = checked_product(&in_spatial)?;
    let out_plane = checked_product(&out_spatial)?;
    let layout = MaxUnpoolLayout {
        n,
        c,
        in_spatial,
        out_spatial,
        nc,
        in_plane,
        out_plane,
    };
    checked_numel_for::<f32>(&layout.in_shape())?;
    checked_numel_for::<f32>(&layout.out_shape())?;
    checked_numel_for::<i32>(&layout.in_shape())?;
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_length_matches_pytorch_formula() {
        // 1d: in=3, k=2, s=2, p=0 -> 6（PyTorch 2.14.0 実測）。
        let l = max_unpool_layout(&[1, 1, 3], &[1, 1, 3], &[2], None, &[0], None).unwrap();
        assert_eq!(l.out_shape(), vec![1, 1, 6]);
        assert_eq!((l.in_plane(), l.out_plane()), (3, 6));
        assert_eq!(l.flat_in(), [1, 3]);
        assert_eq!(l.flat_out(), [1, 6]);
        // 1d: in=3, k=2, s=2, p=1 -> 4。
        let l = max_unpool_layout(&[1, 1, 3], &[1, 1, 3], &[2], None, &[1], None).unwrap();
        assert_eq!(l.out_shape(), vec![1, 1, 4]);
        // 2d: stride 明示・非等方。
        let l = max_unpool_layout(
            &[2, 3, 3, 4],
            &[2, 3, 3, 4],
            &[2, 3],
            Some(&[2, 2]),
            &[0, 1],
            None,
        )
        .unwrap();
        assert_eq!(l.out_shape(), vec![2, 3, 6, 7]);
        assert_eq!(l.flat_in(), [6, 12]);
        assert_eq!(l.flat_out(), [6, 42]);
        // 3d。
        let l = max_unpool_layout(
            &[1, 2, 2, 2, 2],
            &[1, 2, 2, 2, 2],
            &[2, 2, 2],
            None,
            &[0, 0, 0],
            None,
        )
        .unwrap();
        assert_eq!(l.out_shape(), vec![1, 2, 4, 4, 4]);
    }

    #[test]
    fn output_size_is_open_interval_around_default() {
        // default=6・stride=2 -> 許容は 5..=7（PyTorch 2.14.0 実測: 4 と 8 は拒否）。
        let f = |size: usize| {
            max_unpool_layout(&[1, 1, 3], &[1, 1, 3], &[2], None, &[0], Some(&[size]))
        };
        for bad in [0usize, 3, 4, 8, 9] {
            assert!(
                matches!(f(bad), Err(ShapeError::ShapeMismatch { .. })),
                "{bad}"
            );
        }
        for ok in [5usize, 6, 7] {
            assert_eq!(f(ok).unwrap().out_shape(), vec![1, 1, ok]);
        }
    }

    #[test]
    fn rejects_rank_shape_and_argument_length_mismatch() {
        assert!(matches!(
            max_unpool_layout(&[1, 3], &[1, 3], &[2], None, &[0], None),
            Err(ShapeError::RankMismatch {
                expected: 3,
                actual: 2
            })
        ));
        assert!(matches!(
            max_unpool_layout(&[1, 1, 3], &[1, 1, 4], &[2], None, &[0], None),
            Err(ShapeError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            max_unpool_layout(&[1, 1, 3], &[1, 1, 3], &[2, 2], None, &[0, 0], None),
            Err(ShapeError::RankMismatch {
                expected: 4,
                actual: 3
            })
        ));
        assert!(matches!(
            max_unpool_layout(&[1, 1, 3], &[1, 1, 3], &[2], Some(&[2, 2]), &[0], None),
            Err(ShapeError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            max_unpool_layout(&[1, 1, 3], &[1, 1, 3], &[2], None, &[0, 0], None),
            Err(ShapeError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            max_unpool_layout(&[1, 1, 3], &[1, 1, 3], &[2], None, &[0], Some(&[6, 6])),
            Err(ShapeError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            max_unpool_layout(&[1, 1, 3], &[1, 1, 3], &[], None, &[], None),
            Err(ShapeError::RankMismatch { .. })
        ));
    }

    #[test]
    fn rejects_zero_kernel_stride_and_spatial_axis() {
        assert!(max_unpool_layout(&[1, 1, 3], &[1, 1, 3], &[0], None, &[0], None).is_err());
        assert!(max_unpool_layout(&[1, 1, 3], &[1, 1, 3], &[2], Some(&[0]), &[0], None).is_err());
        assert!(max_unpool_layout(&[1, 1, 0], &[1, 1, 0], &[2], None, &[0], None).is_err());
    }

    #[test]
    fn accepts_zero_batch_and_channel_as_empty_output() {
        let l = max_unpool_layout(&[0, 1, 3], &[0, 1, 3], &[2], None, &[0], None).unwrap();
        assert_eq!(l.out_shape(), vec![0, 1, 6]);
        assert_eq!(l.flat_in(), [0, 3]);
        let l = max_unpool_layout(&[1, 0, 3], &[1, 0, 3], &[2], None, &[0], None).unwrap();
        assert_eq!(l.out_shape(), vec![1, 0, 6]);
    }

    #[test]
    fn rejects_non_positive_default_length() {
        // in=1, k=1, s=1, p=1: 1 − 2 < 1。
        assert!(matches!(
            max_unpool_layout(&[1, 1, 1], &[1, 1, 1], &[1], None, &[1], None),
            Err(ShapeError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_overflowing_shapes_before_allocation() {
        let big = usize::MAX / 2;
        // (in − 1)·stride のオーバーフロー。
        assert!(matches!(
            max_unpool_layout(&[1, 1, big], &[1, 1, big], &[2], Some(&[4]), &[0], None),
            Err(ShapeError::ElementCountOverflow)
        ));
        // N·C のオーバーフロー。
        assert!(matches!(
            max_unpool_layout(&[big, 3, 2], &[big, 3, 2], &[2], None, &[0], None),
            Err(ShapeError::ElementCountOverflow)
        ));
        // 出力バイト数が確保上限を超える。
        let huge = usize::MAX / 4;
        assert!(matches!(
            max_unpool_layout(&[1, 1, huge], &[1, 1, huge], &[1], Some(&[1]), &[0], None),
            Err(ShapeError::ElementCountOverflow)
        ));
        // 2p のオーバーフロー。
        assert!(matches!(
            max_unpool_layout(&[1, 1, 3], &[1, 1, 3], &[2], None, &[usize::MAX], None),
            Err(ShapeError::ElementCountOverflow)
        ));
    }
}
