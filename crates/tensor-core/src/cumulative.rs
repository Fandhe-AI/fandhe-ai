//! `cummax`／`cummin`／`logcumsumexp` のホスト参照カーネルの**単一情報源**
//! （イシュー #2636・親 #2625。実装記録は
//! `docs/autodiff-cumulative-ops-decision.md`）。
//!
//! # 役割と呼び出し元
//!
//! - `autodiff::cumulative_ops`（`BackendOps` が `Unsupported` のときの
//!   ホストフォールバック）・`autodiff::grad`（`logcumsumexp` の VJP）・
//!   `backend-cpu` の `CpuBackendOps::scan_cummax`／`scan_cummin`／
//!   `scan_logcumsumexp` がいずれも本モジュールの関数を直接呼ぶ。走査順・
//!   タイ／NaN 規則・アキュムレータ契約をクレート間で複製せず、乖離を構造的に
//!   排除する（`fft.rs` と同じ「共有可能な層に一本化する」方式）。
//! - CUDA／Metal の専用カーネルは本イシューの対象外で、`BackendOps` 既定の
//!   `Unsupported` から autodiff 側がこのホスト実装へフォールバックする。
//!
//! # 数値契約
//!
//! - `cummax`／`cummin` は算術を含まない比較・選択のみ（値は入力の要素そのもの
//!   で bit 一致）。lane（`dim` 以外の軸の組）ごとに `dim` 昇順の逐次走査。
//!   更新規則は PyTorch 2.14.0 の実測（`tests/fixtures/cumulative-pytorch-
//!   reference/`）に合わせる: 現在値 `cur` が NaN でなければ `x >= cur`
//!   （`cummin` は `x <= cur`）または `x` が NaN のとき `cur = x`・索引を更新
//!   する（**タイは後勝ち**）。`cur` が NaN になった後は NaN 入力でのみ索引が
//!   更新される（NaN は伝播）。
//! - `logcumsumexp` は lane ごとに `f64` アキュムレータを持ち、
//!   `acc = log_add_exp(acc, x)` の各時点を `f32` へ 1 回だけ downcast して
//!   書き出す（downcast 値は読み戻さない。cumsum の `f64` アキュムレータ契約の
//!   拡張）。`exp`／`ln_1p` は libm 依存のため、クレート間 bit 同一は受入条件に
//!   しない（REQ-2 統一複合判定で比較）。`mul_add` は使わず、matmul 系 FMA
//!   契約には触れない。
//! - 非有限入力は事前に拒否せず伝播する（`cumsum`・FFT と同じ）。
//!
//! # 境界検査（REQ-8・OWASP A03）
//!
//! 形状・`dim` の検査は [`cumulative_layout`] に集約し、確保より前に `dim` 範囲
//! と要素数／バイト数の `checked_mul` を検査して型付きエラー（[`ShapeError`]）で
//! 拒否する。カーネルは入力スライス長も再検査し、`cummax`／`cummin` の索引は
//! `i32::try_from`（無検査 `as i32` を使わない）で変換する。`unsafe`／
//! `get_unchecked` は使わない。

use crate::error::ShapeError;
use crate::ops_shape::reduce_out_shape;

/// 解決済みの累積走査レイアウト（[`cumulative_layout`] の戻り値）。
///
/// 入出力はいずれも連続配置の行優先で同一 shape。`dim` 軸の前を `outer`・後ろ
/// を `inner` として、1 lane = `(outer 添字, inner 添字)` を独立に走査する。
///
/// フィールドは非公開で、[`cumulative_layout`] だけが生成する。生成時の検査で
/// 整合性が保証され、公開カーネルが外部から改変された値で範囲外アクセス
/// しないようにする（REQ-8）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CumulativeLayout {
    outer: usize,
    axis_len: usize,
    inner: usize,
    numel: usize,
    shape: Vec<usize>,
}

impl CumulativeLayout {
    /// `dim` 軸の長さ。
    pub fn axis_len(&self) -> usize {
        self.axis_len
    }

    /// 入出力共通の shape。
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 総要素数。
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// `cummax`／`cummin` の索引を `i32` で表現できることを検査する。
    ///
    /// 軸長 0 は走査対象が無いため常に成功する。最大索引 `axis_len - 1` が
    /// `i32::MAX` を超える場合は [`ShapeError::IndexRangeOverflow`]（`sort`／
    /// `topk` と同じ契約）。autodiff 入口が実体化より前に呼ぶ。
    pub fn check_i32_indices(&self) -> Result<(), ShapeError> {
        if self.axis_len == 0 {
            return Ok(());
        }
        let max_index = self.axis_len - 1;
        i32::try_from(max_index)
            .map(|_| ())
            .map_err(|_| ShapeError::IndexRangeOverflow { index: max_index })
    }

    fn check_len(&self, actual: usize) -> Result<(), ShapeError> {
        if actual == self.numel {
            Ok(())
        } else {
            Err(ShapeError::ElementCountMismatch {
                expected: self.numel,
                actual,
            })
        }
    }

    #[inline]
    fn pos(&self, o: usize, a: usize, i: usize) -> usize {
        (o * self.axis_len + a) * self.inner + i
    }
}

/// `dim` と shape を検査して [`CumulativeLayout`] を返す。
///
/// `dim >= rank`（rank 0 を含む）は [`ShapeError::AxisOutOfRange`]、要素数・
/// バイト数の `checked_mul` オーバーフローや `isize::MAX` 超過は
/// [`ShapeError::ElementCountOverflow`]。要素数 0 の入力は部分積を計算する前に
/// 空レイアウト（走査 0 回）へ倒す。
pub fn cumulative_layout(shape: &[usize], dim: usize) -> Result<CumulativeLayout, ShapeError> {
    reduce_out_shape(shape, Some(dim))?;
    let axis_len = shape[dim];
    if shape.contains(&0) {
        return Ok(CumulativeLayout {
            outer: 0,
            axis_len,
            inner: 0,
            numel: 0,
            shape: shape.to_vec(),
        });
    }
    let numel = checked_numel(shape)?;
    let bytes = numel
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(ShapeError::ElementCountOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(ShapeError::ElementCountOverflow);
    }
    Ok(CumulativeLayout {
        outer: checked_numel(&shape[..dim])?,
        axis_len,
        inner: checked_numel(&shape[dim + 1..])?,
        numel,
        shape: shape.to_vec(),
    })
}

fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// 累積最大値（値と索引）。`torch.cummax` 相当。タイは後勝ち・NaN は伝播。
pub fn cummax_host(
    x: &[f32],
    layout: &CumulativeLayout,
) -> Result<(Vec<f32>, Vec<i32>), ShapeError> {
    scan_extremum(x, layout, true)
}

/// 累積最小値（値と索引）。`torch.cummin` 相当。タイは後勝ち・NaN は伝播。
pub fn cummin_host(
    x: &[f32],
    layout: &CumulativeLayout,
) -> Result<(Vec<f32>, Vec<i32>), ShapeError> {
    scan_extremum(x, layout, false)
}

fn scan_extremum(
    x: &[f32],
    layout: &CumulativeLayout,
    is_max: bool,
) -> Result<(Vec<f32>, Vec<i32>), ShapeError> {
    layout.check_len(x.len())?;
    layout.check_i32_indices()?;
    let mut values = vec![0.0_f32; layout.numel];
    let mut indices = vec![0_i32; layout.numel];
    for o in 0..layout.outer {
        for i in 0..layout.inner {
            let mut cur = 0.0_f32;
            let mut cur_idx = 0_i32;
            for a in 0..layout.axis_len {
                let p = layout.pos(o, a, i);
                let v = x[p];
                let idx =
                    i32::try_from(a).map_err(|_| ShapeError::IndexRangeOverflow { index: a })?;
                let take = if a == 0 {
                    true
                } else if cur.is_nan() {
                    v.is_nan()
                } else if v.is_nan() {
                    true
                } else if is_max {
                    v >= cur
                } else {
                    v <= cur
                };
                if take {
                    cur = v;
                    cur_idx = idx;
                }
                values[p] = cur;
                indices[p] = cur_idx;
            }
        }
    }
    Ok((values, indices))
}

/// `log(exp(a) + exp(b))` を `f64` で安定に求める。
///
/// 分岐順: どちらかが NaN なら NaN → `a == b` かつ非有限なら `a`
/// （`(+inf, +inf)`・`(-inf, -inf)` で `inf - inf = NaN` になるのを避ける）→
/// それ以外は `max + ln_1p(exp(min - max))`。
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b && !a.is_finite() {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// 累積 `logsumexp`。`torch.logcumsumexp` 相当。lane ごとに `f64` で蓄積し、
/// 各時点を 1 回だけ `f32` へ downcast して書き出す。
pub fn logcumsumexp_host(x: &[f32], layout: &CumulativeLayout) -> Result<Vec<f32>, ShapeError> {
    layout.check_len(x.len())?;
    let mut out = vec![0.0_f32; layout.numel];
    for o in 0..layout.outer {
        for i in 0..layout.inner {
            let mut acc = 0.0_f64;
            for a in 0..layout.axis_len {
                let p = layout.pos(o, a, i);
                let v = f64::from(x[p]);
                acc = if a == 0 { v } else { log_add_exp(acc, v) };
                out[p] = acc as f32;
            }
        }
    }
    Ok(out)
}

/// [`logcumsumexp_host`] の VJP（入力勾配）。
///
/// `d_x[j] = Σ_{i>=j} g[i]·exp(x[j] - out[i])` を O(n) の逆向き再帰で求める:
/// `T[n-1] = g[n-1]`・`T[j] = g[j] + exp(out[j] - out[j+1])·T[j+1]`・
/// `d_x[j] = exp(x[j] - out[j])·T[j]`。`out` は forward と同じ `log_add_exp`
/// で `f64` のまま再計算する（`f32` へ丸めた記録値は使わない）。`out` は単調
/// 非減少かつ `x[j] <= out[j]` のため指数の引数は常に 0 以下で、係数は
/// `[0, 1]` に収まり overflow しない。最後に 1 回だけ `f32` へ downcast する。
/// 非有限を含む lane の勾配は式の IEEE 伝播に従う（拒否・panic しない）。
/// 作業バッファは lane 長の `Vec<f64>` 1 本のみ。
pub fn logcumsumexp_vjp_host(
    x: &[f32],
    upstream: &[f32],
    layout: &CumulativeLayout,
) -> Result<Vec<f32>, ShapeError> {
    layout.check_len(x.len())?;
    layout.check_len(upstream.len())?;
    let mut d_x = vec![0.0_f32; layout.numel];
    let mut prefix = vec![0.0_f64; layout.axis_len];
    for o in 0..layout.outer {
        for i in 0..layout.inner {
            for a in 0..layout.axis_len {
                let v = f64::from(x[layout.pos(o, a, i)]);
                prefix[a] = if a == 0 {
                    v
                } else {
                    log_add_exp(prefix[a - 1], v)
                };
            }
            let mut t = 0.0_f64;
            for a in (0..layout.axis_len).rev() {
                let p = layout.pos(o, a, i);
                let g = f64::from(upstream[p]);
                t = if a + 1 == layout.axis_len {
                    g
                } else {
                    g + (prefix[a] - prefix[a + 1]).exp() * t
                };
                d_x[p] = ((f64::from(x[p]) - prefix[a]).exp() * t) as f32;
            }
        }
    }
    Ok(d_x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(shape: &[usize], dim: usize) -> CumulativeLayout {
        cumulative_layout(shape, dim).expect("test layout")
    }

    #[test]
    fn max_and_min_follow_ties_later_wins() {
        let x = [3.0_f32, 1.0, 3.0, 2.0, 5.0, 5.0];
        let l = layout(&[6], 0);
        let (v, i) = cummax_host(&x, &l).unwrap();
        assert_eq!(v, vec![3.0, 3.0, 3.0, 3.0, 5.0, 5.0]);
        assert_eq!(i, vec![0, 0, 2, 2, 4, 5]);
        let (v, i) = cummin_host(&x, &l).unwrap();
        assert_eq!(v, vec![3.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(i, vec![0, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn nan_propagates_and_updates_only_on_nan() {
        let x = [1.0_f32, f32::NAN, 2.0, f32::NAN, 0.0];
        let (v, i) = cummax_host(&x, &layout(&[5], 0)).unwrap();
        assert_eq!(v[0], 1.0);
        assert!(v[1..].iter().all(|a| a.is_nan()));
        assert_eq!(i, vec![0, 1, 1, 3, 3]);
        let (_, i) = cummin_host(&x, &layout(&[5], 0)).unwrap();
        assert_eq!(i, vec![0, 1, 1, 3, 3]);
    }

    #[test]
    fn signed_zero_ties_pick_later_bits() {
        let x = [0.0_f32, -0.0, 0.0];
        let (v, i) = cummax_host(&x, &layout(&[3], 0)).unwrap();
        assert_eq!(i, vec![0, 1, 2]);
        assert_eq!(v[1].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn non_last_dim_scans_each_lane() {
        // shape [2,3], dim 0: 列ごとに走査。
        let x = [1.0_f32, 5.0, 2.0, 4.0, 3.0, 9.0];
        let (v, i) = cummax_host(&x, &layout(&[2, 3], 0)).unwrap();
        assert_eq!(v, vec![1.0, 5.0, 2.0, 4.0, 5.0, 9.0]);
        assert_eq!(i, vec![0, 0, 0, 1, 0, 1]);
    }

    #[test]
    fn empty_and_single_axis() {
        let l = layout(&[0], 0);
        let (v, i) = cummax_host(&[], &l).unwrap();
        assert!(v.is_empty() && i.is_empty());
        assert!(logcumsumexp_host(&[], &l).unwrap().is_empty());
        // 0 を含む多次元でも部分積を計算せず空に倒れる。
        let l = layout(&[usize::MAX, usize::MAX, 0], 2);
        assert_eq!(l.numel(), 0);
        let (v, _) = cummin_host(&[], &l).unwrap();
        assert!(v.is_empty());
        let l = layout(&[1], 0);
        let (v, i) = cummax_host(&[7.0], &l).unwrap();
        assert_eq!((v, i), (vec![7.0], vec![0]));
        assert_eq!(
            logcumsumexp_host(&[7.0], &l).unwrap()[0].to_bits(),
            7.0_f32.to_bits()
        );
    }

    #[test]
    fn layout_rejects_bad_dim_and_huge_shapes() {
        assert!(matches!(
            cumulative_layout(&[], 0),
            Err(ShapeError::AxisOutOfRange { axis: 0, rank: 0 })
        ));
        assert!(matches!(
            cumulative_layout(&[4], 1),
            Err(ShapeError::AxisOutOfRange { axis: 1, rank: 1 })
        ));
        assert!(matches!(
            cumulative_layout(&[usize::MAX, 2], 0),
            Err(ShapeError::ElementCountOverflow)
        ));
        assert!(matches!(
            cumulative_layout(&[usize::MAX / 2], 0),
            Err(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn i32_index_axis_limit_is_checked() {
        let l = CumulativeLayout {
            outer: 1,
            axis_len: i32::MAX as usize + 2,
            inner: 1,
            numel: 0,
            shape: vec![],
        };
        assert!(matches!(
            l.check_i32_indices(),
            Err(ShapeError::IndexRangeOverflow { .. })
        ));
        let ok = CumulativeLayout {
            axis_len: i32::MAX as usize + 1,
            ..l
        };
        assert!(ok.check_i32_indices().is_ok());
    }

    #[test]
    fn kernels_reject_mismatched_slice_len() {
        let l = layout(&[3], 0);
        assert!(matches!(
            cummax_host(&[1.0, 2.0], &l),
            Err(ShapeError::ElementCountMismatch {
                expected: 3,
                actual: 2
            })
        ));
        assert!(logcumsumexp_host(&[1.0], &l).is_err());
        assert!(logcumsumexp_vjp_host(&[1.0; 3], &[1.0; 2], &l).is_err());
    }

    #[test]
    fn logcumsumexp_matches_direct_and_resists_overflow() {
        let x = [1.0_f32, 2.0, 3.0];
        let out = logcumsumexp_host(&x, &layout(&[3], 0)).unwrap();
        let e = [1.0_f64.exp(), 2.0_f64.exp(), 3.0_f64.exp()];
        assert!((f64::from(out[1]) - (e[0] + e[1]).ln()).abs() < 1e-6);
        assert!((f64::from(out[2]) - (e[0] + e[1] + e[2]).ln()).abs() < 1e-6);
        let big = logcumsumexp_host(&[500.0, 500.0, -500.0], &layout(&[3], 0)).unwrap();
        assert!(big.iter().all(|v| v.is_finite()));
        assert!((f64::from(big[1]) - (500.0 + 2.0_f64.ln())).abs() < 1e-4);
    }

    #[test]
    fn log_add_exp_non_finite_branches() {
        assert!(log_add_exp(f64::NAN, 1.0).is_nan());
        assert_eq!(log_add_exp(f64::INFINITY, f64::INFINITY), f64::INFINITY);
        assert_eq!(
            log_add_exp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
        assert_eq!(log_add_exp(f64::NEG_INFINITY, 2.0), 2.0);
        assert_eq!(log_add_exp(f64::INFINITY, f64::NEG_INFINITY), f64::INFINITY);
    }

    #[test]
    fn vjp_matches_quadratic_oracle() {
        let x = [0.3_f32, -1.2, 2.5, 0.0, 1.1];
        let g = [0.5_f32, -0.7, 1.3, 0.2, -0.9];
        let l = layout(&[5], 0);
        let d = logcumsumexp_vjp_host(&x, &g, &l).unwrap();
        let xs: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let mut out = [0.0_f64; 5];
        for k in 0..5 {
            out[k] = xs[..=k].iter().map(|v| v.exp()).sum::<f64>().ln();
        }
        for j in 0..5 {
            let want: f64 = (j..5)
                .map(|i| f64::from(g[i]) * (xs[j] - out[i]).exp())
                .sum();
            assert!((f64::from(d[j]) - want).abs() < 1e-6, "j={j}");
        }
        // run-to-run bit 決定性。
        let d2 = logcumsumexp_vjp_host(&x, &g, &l).unwrap();
        assert!(d.iter().zip(&d2).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}
