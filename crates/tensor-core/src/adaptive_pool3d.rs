//! 3D Adaptive プーリング（`adaptive_avg_pool3d`／`adaptive_max_pool3d`）のホスト参照カーネルの
//! **単一情報源**（イシュー #3823・親 #3747。設計・実装記録は
//! `docs/autodiff-pool3d-ops-decision.md` §18・§21）。
//!
//! # 役割と呼び出し元
//!
//! - `autodiff::adaptive_pool3d_ops`（`BackendOps` が `Unsupported` のときのホストフォールバック）・
//!   `autodiff::grad`（`Op::AdaptiveAvgPool3d` の VJP）・`backend-cpu` の
//!   `CpuBackendOps::adaptive_pool3d_avg`／`adaptive_pool3d_max` がいずれも本モジュールの関数を
//!   直接呼ぶ。窓の走査順・タイ／NaN 規則・アキュムレータ契約をクレート間で複製しない
//!   （[`crate::pool3d`] と同じ「共有可能な層に一本化する」方式）。
//! - 窓は軸ごとに [`crate::adaptive_window`]（2D 版と共有する単一情報源）で求める。
//! - CUDA／Metal の専用カーネルは本イシューの対象外で、`BackendOps` 既定の `Unsupported` から
//!   autodiff 側が本ホスト実装へフォールバックする（GPU 実機 parity は #3860／#3861）。
//!
//! # 数値契約
//!
//! - **Max**: 窓内を `kd` 外側 → `kh` → `kw` 内側の row-major で走査し、更新条件は
//!   `v > best || (v.is_nan() && !best.is_nan())`（タイ先勝ち・窓内で最初の NaN の索引が残る）。
//!   値は入力要素と bit 一致。索引は `(n, c)` 平面内 flat 添字 `d·H·W + h·W + w`（`i32`）。
//! - **Avg**: 窓内を row-major で `f64` へ昇格して逐次加算し、`f64` の窓要素数で除算して
//!   1 回だけ `f32` へ downcast する。[`adaptive_avg_pool3d_vjp_host`] は出力 major 走査で
//!   `f64` 配列へ加算し 1 回 downcast する。`mul_add` は使わない（FMA 契約は不変）。
//!
//! # 境界検査（REQ-8・OWASP A03）
//!
//! 形状は [`adaptive_pool3d_layout`] に集約し、確保より前に rank・空間軸 0・`output_size` 0・
//! 要素数／バイト数を検査する。カーネルは入力スライス長を再検査し、座標を手動で範囲検査する。
//! 索引は `i32::try_from`。`unsafe`／`get_unchecked` は使わない。

use crate::error::ShapeError;
use crate::ops_shape::{adaptive_pool3d_out_shape, adaptive_window};

/// 検査済みの 3D Adaptive プーリングレイアウト（[`adaptive_pool3d_layout`] の戻り値）。
/// フィールドは非公開で、検査を通った値だけがカーネルに渡る。
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptivePool3dLayout {
    in_shape: [usize; 5],
    out_shape: [usize; 5],
}

impl AdaptivePool3dLayout {
    /// 入力 shape `[N, C, D, H, W]`。
    pub fn in_shape(&self) -> [usize; 5] {
        self.in_shape
    }

    /// 出力 shape `[N, C, Dout, Hout, Wout]`。
    pub fn out_shape(&self) -> [usize; 5] {
        self.out_shape
    }

    /// 入力の総要素数（[`adaptive_pool3d_layout`] が検査済みのため桁あふれしない）。
    pub fn in_numel(&self) -> usize {
        self.in_shape.iter().product()
    }

    /// 出力の総要素数。
    pub fn out_numel(&self) -> usize {
        self.out_shape.iter().product()
    }

    /// Max の索引（平面内 flat 添字の最大値 `D·H·W - 1`）が `i32` で表現できることを検査する。
    /// `N`／`C` が 0 の空バッチでも検査する（`D·H·W` の `checked_mul` を含む）。
    pub fn check_i32_indices(&self) -> Result<(), ShapeError> {
        let plane = self.in_shape[2]
            .checked_mul(self.in_shape[3])
            .and_then(|p| p.checked_mul(self.in_shape[4]))
            .ok_or(ShapeError::ElementCountOverflow)?;
        if plane > i32::MAX as usize {
            return Err(ShapeError::IndexRangeOverflow { index: plane });
        }
        Ok(())
    }

    fn in_plane(&self) -> usize {
        self.in_shape[2] * self.in_shape[3] * self.in_shape[4]
    }

    fn out_plane(&self) -> usize {
        self.out_shape[2] * self.out_shape[3] * self.out_shape[4]
    }
}

/// 入力 shape と `output_size` を検査して [`AdaptivePool3dLayout`] を返す。
/// 検査は [`adaptive_pool3d_out_shape`] に従う。入力の総要素数の桁あふれも検査する。
pub fn adaptive_pool3d_layout(
    in_shape: &[usize],
    output_size: [usize; 3],
) -> Result<AdaptivePool3dLayout, ShapeError> {
    let out = adaptive_pool3d_out_shape(in_shape, output_size)?;
    crate::tensor::checked_numel_for::<f32>(in_shape)?;
    let mut i = [0usize; 5];
    let mut o = [0usize; 5];
    i.copy_from_slice(in_shape);
    o.copy_from_slice(&out);
    Ok(AdaptivePool3dLayout {
        in_shape: i,
        out_shape: o,
    })
}

fn check_input_len(layout: &AdaptivePool3dLayout, actual: usize) -> Result<(), ShapeError> {
    let expected = layout.in_numel();
    if actual == expected {
        Ok(())
    } else {
        Err(ShapeError::ElementCountMismatch { expected, actual })
    }
}

/// 出力位置 `o` の軸ごとの窓 `[start, end)` を `adaptive_window` で求める。
/// 窓が空（`start >= end`）または入力を超える場合は fail-closed で拒否する。
fn windows(
    layout: &AdaptivePool3dLayout,
    o: [usize; 3],
) -> Result<[(usize, usize); 3], ShapeError> {
    let mut w = [(0usize, 0usize); 3];
    for a in 0..3 {
        let in_len = layout.in_shape[2 + a];
        let out_len = layout.out_shape[2 + a];
        let (s, e) =
            adaptive_window(o[a], in_len, out_len).ok_or(ShapeError::ElementCountOverflow)?;
        if s >= e || e > in_len {
            return Err(ShapeError::ShapeMismatch {
                lhs: vec![s, e],
                rhs: vec![0, in_len],
            });
        }
        w[a] = (s, e);
    }
    Ok(w)
}

/// 3D Adaptive max pooling（値と索引）。`F.adaptive_max_pool3d(..., return_indices=True)` 相当。
///
/// 戻り値は `([N,C,Dout,Hout,Wout] の値, 同 shape の索引)`。タイ先勝ち・最初の NaN の索引が残る
/// （PyTorch は最後の NaN。差は設計記録に記録済み）。
pub fn adaptive_max_pool3d_host(
    input: &[f32],
    layout: &AdaptivePool3dLayout,
) -> Result<(Vec<f32>, Vec<i32>), ShapeError> {
    check_input_len(layout, input.len())?;
    layout.check_i32_indices()?;
    let out_numel = layout.out_numel();
    if out_numel == 0 || input.is_empty() {
        return Ok((vec![0.0; out_numel], vec![0; out_numel]));
    }
    let [n, c] = [layout.in_shape[0], layout.in_shape[1]];
    let [h_in, w_in] = [layout.in_shape[3], layout.in_shape[4]];
    let [d_out, h_out, w_out] = [
        layout.out_shape[2],
        layout.out_shape[3],
        layout.out_shape[4],
    ];
    let (in_plane, out_plane) = (layout.in_plane(), layout.out_plane());
    let mut values = Vec::with_capacity(out_numel);
    let mut indices = Vec::with_capacity(out_numel);
    for nc in 0..n * c {
        let plane = &input[nc * in_plane..(nc + 1) * in_plane];
        debug_assert_eq!(values.len(), nc * out_plane);
        for od in 0..d_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let [(ds, de), (hs, he), (ws, we)] = windows(layout, [od, oh, ow])?;
                    let mut best: Option<(f32, usize)> = None;
                    for d in ds..de {
                        for h in hs..he {
                            for w in ws..we {
                                let flat = (d * h_in + h) * w_in + w;
                                let v = *plane
                                    .get(flat)
                                    .ok_or(ShapeError::IndexRangeOverflow { index: flat })?;
                                let take = match best {
                                    None => true,
                                    Some((b, _)) => v > b || (v.is_nan() && !b.is_nan()),
                                };
                                if take {
                                    best = Some((v, flat));
                                }
                            }
                        }
                    }
                    let (v, flat) = best.ok_or(ShapeError::ShapeMismatch {
                        lhs: vec![od, oh, ow],
                        rhs: vec![0],
                    })?;
                    values.push(v);
                    indices.push(
                        i32::try_from(flat)
                            .map_err(|_| ShapeError::IndexRangeOverflow { index: flat })?,
                    );
                }
            }
        }
    }
    Ok((values, indices))
}

/// 3D Adaptive average pooling（`F.adaptive_avg_pool3d` 相当）。窓内 `f64` 逐次加算・
/// `f64` の窓要素数で除算・1 回だけ `f32` へ downcast。
pub fn adaptive_avg_pool3d_host(
    input: &[f32],
    layout: &AdaptivePool3dLayout,
) -> Result<Vec<f32>, ShapeError> {
    check_input_len(layout, input.len())?;
    let out_numel = layout.out_numel();
    if out_numel == 0 || input.is_empty() {
        return Ok(vec![0.0; out_numel]);
    }
    let [n, c] = [layout.in_shape[0], layout.in_shape[1]];
    let [h_in, w_in] = [layout.in_shape[3], layout.in_shape[4]];
    let [d_out, h_out, w_out] = [
        layout.out_shape[2],
        layout.out_shape[3],
        layout.out_shape[4],
    ];
    let in_plane = layout.in_plane();
    let mut out = Vec::with_capacity(out_numel);
    for nc in 0..n * c {
        let plane = &input[nc * in_plane..(nc + 1) * in_plane];
        for od in 0..d_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let [(ds, de), (hs, he), (ws, we)] = windows(layout, [od, oh, ow])?;
                    let mut acc = 0.0_f64;
                    let mut count = 0usize;
                    for d in ds..de {
                        for h in hs..he {
                            for w in ws..we {
                                let flat = (d * h_in + h) * w_in + w;
                                let v = *plane
                                    .get(flat)
                                    .ok_or(ShapeError::IndexRangeOverflow { index: flat })?;
                                acc += f64::from(v);
                                count += 1;
                            }
                        }
                    }
                    if count == 0 {
                        return Err(ShapeError::ShapeMismatch {
                            lhs: vec![od, oh, ow],
                            rhs: vec![0],
                        });
                    }
                    out.push((acc / count as f64) as f32);
                }
            }
        }
    }
    Ok(out)
}

/// [`adaptive_avg_pool3d_host`] の VJP（入力勾配）。出力 major（`N`・`C`・`od`・`oh`・`ow`）で
/// 走査し、`upstream / 窓要素数` を `f64` 配列（入力要素数）へ加算して最後に 1 回 `f32` へ戻す。
/// `upstream` は出力 shape の連続配置スライス。
pub fn adaptive_avg_pool3d_vjp_host(
    upstream: &[f32],
    layout: &AdaptivePool3dLayout,
) -> Result<Vec<f32>, ShapeError> {
    let out_numel = layout.out_numel();
    if upstream.len() != out_numel {
        return Err(ShapeError::ElementCountMismatch {
            expected: out_numel,
            actual: upstream.len(),
        });
    }
    let in_numel = layout.in_numel();
    let acc_bytes = in_numel
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(ShapeError::ElementCountOverflow)?;
    if acc_bytes > isize::MAX as usize {
        return Err(ShapeError::ElementCountOverflow);
    }
    if in_numel == 0 || out_numel == 0 {
        return Ok(vec![0.0; in_numel]);
    }
    let [n, c] = [layout.in_shape[0], layout.in_shape[1]];
    let [h_in, w_in] = [layout.in_shape[3], layout.in_shape[4]];
    let [d_out, h_out, w_out] = [
        layout.out_shape[2],
        layout.out_shape[3],
        layout.out_shape[4],
    ];
    let (in_plane, out_plane) = (layout.in_plane(), layout.out_plane());
    let mut acc = vec![0.0_f64; in_numel];
    for nc in 0..n * c {
        let acc_plane = &mut acc[nc * in_plane..(nc + 1) * in_plane];
        let up_plane = &upstream[nc * out_plane..(nc + 1) * out_plane];
        for od in 0..d_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let [(ds, de), (hs, he), (ws, we)] = windows(layout, [od, oh, ow])?;
                    let count = (de - ds)
                        .checked_mul(he - hs)
                        .and_then(|v| v.checked_mul(we - ws))
                        .ok_or(ShapeError::ElementCountOverflow)?;
                    if count == 0 {
                        return Err(ShapeError::ShapeMismatch {
                            lhs: vec![od, oh, ow],
                            rhs: vec![0],
                        });
                    }
                    let g = up_plane[(od * h_out + oh) * w_out + ow];
                    let contrib = f64::from(g) / count as f64;
                    for d in ds..de {
                        for h in hs..he {
                            for w in ws..we {
                                let flat = (d * h_in + h) * w_in + w;
                                let slot = acc_plane
                                    .get_mut(flat)
                                    .ok_or(ShapeError::IndexRangeOverflow { index: flat })?;
                                *slot += contrib;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(acc.into_iter().map(|v| v as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    #[test]
    fn max_divisible_values_and_indices() {
        let l = adaptive_pool3d_layout(&[1, 1, 2, 2, 2], [1, 1, 1]).unwrap();
        let (v, i) = adaptive_max_pool3d_host(&seq(8), &l).unwrap();
        assert_eq!((v, i), (vec![7.0], vec![7]));
        let l = adaptive_pool3d_layout(&[1, 1, 2, 2, 2], [2, 2, 2]).unwrap();
        let (v, i) = adaptive_max_pool3d_host(&seq(8), &l).unwrap();
        assert_eq!(v, seq(8));
        assert_eq!(i, (0..8).collect::<Vec<i32>>());
    }

    #[test]
    fn max_non_divisible_overlapping_windows() {
        // D 軸 in=3,out=2: 窓 [0,2),[1,3)（重なる）。入力 [0,1,2] → 最大 1,2 / 索引 1,2。
        let l = adaptive_pool3d_layout(&[1, 1, 3, 1, 1], [2, 1, 1]).unwrap();
        let (v, i) = adaptive_max_pool3d_host(&[0.0, 1.0, 2.0], &l).unwrap();
        assert_eq!((v, i), (vec![1.0, 2.0], vec![1, 2]));
    }

    #[test]
    fn max_output_larger_than_input_repeats() {
        let l = adaptive_pool3d_layout(&[1, 1, 1, 1, 1], [3, 1, 1]).unwrap();
        let (v, i) = adaptive_max_pool3d_host(&[5.0], &l).unwrap();
        assert_eq!((v, i), (vec![5.0; 3], vec![0; 3]));
    }

    #[test]
    fn max_tie_first_wins_and_first_nan_kept() {
        let l = adaptive_pool3d_layout(&[1, 1, 1, 1, 4], [1, 1, 1]).unwrap();
        let (v, i) = adaptive_max_pool3d_host(&[3.0, 3.0, 1.0, 3.0], &l).unwrap();
        assert_eq!((v, i), (vec![3.0], vec![0]));
        let (v, i) = adaptive_max_pool3d_host(&[1.0, f32::NAN, 5.0, f32::NAN], &l).unwrap();
        assert!(v[0].is_nan());
        assert_eq!(i, vec![1]);
        let (v, i) = adaptive_max_pool3d_host(&[f32::NEG_INFINITY; 4], &l).unwrap();
        assert_eq!((v[0], i[0]), (f32::NEG_INFINITY, 0));
    }

    #[test]
    fn avg_closed_form_and_overlap() {
        let l = adaptive_pool3d_layout(&[1, 1, 2, 2, 2], [1, 1, 1]).unwrap();
        assert_eq!(adaptive_avg_pool3d_host(&seq(8), &l).unwrap(), vec![3.5]);
        let l = adaptive_pool3d_layout(&[1, 1, 3, 1, 1], [2, 1, 1]).unwrap();
        assert_eq!(
            adaptive_avg_pool3d_host(&[0.0, 1.0, 2.0], &l).unwrap(),
            vec![0.5, 1.5]
        );
    }

    #[test]
    fn vjp_accumulates_overlapping_windows() {
        let l = adaptive_pool3d_layout(&[1, 1, 3, 1, 1], [2, 1, 1]).unwrap();
        let g = adaptive_avg_pool3d_vjp_host(&[1.0, 1.0], &l).unwrap();
        assert_eq!(g, vec![0.5, 1.0, 0.5]);
    }

    #[test]
    fn rejections() {
        assert!(adaptive_pool3d_layout(&[1, 1, 4, 4], [1, 1, 1]).is_err());
        assert!(adaptive_pool3d_layout(&[1, 1, 0, 4, 4], [1, 1, 1]).is_err());
        assert!(adaptive_pool3d_layout(&[1, 1, 4, 4, 4], [1, 0, 1]).is_err());
        let l = adaptive_pool3d_layout(&[1, 1, 2, 2, 2], [1, 1, 1]).unwrap();
        assert!(matches!(
            adaptive_max_pool3d_host(&[0.0; 7], &l),
            Err(ShapeError::ElementCountMismatch { .. })
        ));
        assert!(matches!(
            adaptive_avg_pool3d_host(&[0.0; 9], &l),
            Err(ShapeError::ElementCountMismatch { .. })
        ));
        assert!(matches!(
            adaptive_avg_pool3d_vjp_host(&[0.0; 2], &l),
            Err(ShapeError::ElementCountMismatch { .. })
        ));
        // N=0 でも D·H·W が i32 を超える形状は索引検査で拒否する（shape だけ作る）。
        let big = adaptive_pool3d_layout(&[0, 1, 2048, 2048, 1024], [1, 1, 1]).unwrap();
        assert!(big.check_i32_indices().is_err());
        assert!(adaptive_max_pool3d_host(&[], &big).is_err());
    }

    #[test]
    fn empty_batch_gives_empty_output() {
        let l = adaptive_pool3d_layout(&[0, 3, 2, 2, 2], [1, 1, 1]).unwrap();
        let (v, i) = adaptive_max_pool3d_host(&[], &l).unwrap();
        assert!(v.is_empty() && i.is_empty());
        assert!(adaptive_avg_pool3d_host(&[], &l).unwrap().is_empty());
    }

    #[test]
    fn deterministic() {
        let l = adaptive_pool3d_layout(&[2, 2, 5, 4, 3], [3, 2, 2]).unwrap();
        let x: Vec<f32> = (0..l.in_numel())
            .map(|i| ((i * 7) % 13) as f32 - 6.0)
            .collect();
        assert_eq!(
            adaptive_max_pool3d_host(&x, &l).unwrap(),
            adaptive_max_pool3d_host(&x, &l).unwrap()
        );
        assert_eq!(
            adaptive_avg_pool3d_host(&x, &l).unwrap(),
            adaptive_avg_pool3d_host(&x, &l).unwrap()
        );
    }
}
