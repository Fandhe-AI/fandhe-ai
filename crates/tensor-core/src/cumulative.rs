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

/// [`logcumsumexp_host`] の VJP（入力勾配）。`autodiff::grad`（`Op::Logcumsumexp` の 1 階）と
/// create_graph の V3 値・二階の再現（`autodiff::create_graph::cumulative_b`）から呼ばれる。
/// GPU（CUDA・Metal・ROCm）に本 VJP のカーネルは無く、全バックエンドがこのホスト関数を共用する。
///
/// `d_x[j] = Σ_{i>=j} g[i]·exp(x[j] - out[i])` を O(n) の逆向き再帰で求める:
/// `T[n-1] = g[n-1]`・`T[j] = g[j] + c[j]·T[j+1]`・`d_x[j] = w[j]·T[j]`。係数 `c`・`w` は
/// 出力 `out`（prefix の logsumexp）を経由せず入力どうしの差だけから作る（イシュー #3965。
/// `docs/autodiff-higher-order-grad-decision.md` §78）。`out` の差から作ると、大きな共通オフセット
/// の上の小さな差（`[1e30, 1e30]` の `ln 2`）が `f64` でも `x + ln 2 == x` で消え、係数が 1 に
/// 潰れて勾配が誤る。prefix 最大値 `m[k] = max_{i<=k} x[i]` と正規化和
/// `Z[k] = Σ_{i<=k} exp(x[i] - m[k])`（`[1, k+1]`）を
/// `lead[k] = Z[k-1]·exp(m[k-1] - m[k])`・`Z[k] = lead[k] + exp(x[k] - m[k])` で求め、
/// `c[j] = lead[j+1] / Z[j+1]`・`w[j] = exp(x[j] - m[j]) / Z[j]` とする。指数の引数はすべて
/// 入力どうしの差（0 以下）なので共通オフセット不変で overflow せず、係数は `[0, 1]` に収まる。
/// 同型の式を PR #3959 の create_graph 参照 `lcse_ref_coeffs` も使うが、参照の独立性（§64.2）の
/// ため実装は共有しない。`f64` アキュムレータで最後に 1 回だけ `f32` へ downcast する（REQ-2・
/// FMA 契約は不変）。非有限（NaN・±inf）を含む lane だけは従来どおり `f64` の prefix の差から
/// 係数を作る（共通オフセットの消失は有限入力の問題で、非有限 lane の NaN／有限のクラスと
/// PyTorch との一致数〈`cumulative_parity`〉を変えないため）。勾配は式の IEEE 伝播に従う
/// （拒否・panic しない）。作業バッファは lane 長の `Vec<f64>` 3 本。
pub fn logcumsumexp_vjp_host(
    x: &[f32],
    upstream: &[f32],
    layout: &CumulativeLayout,
) -> Result<Vec<f32>, ShapeError> {
    layout.check_len(x.len())?;
    layout.check_len(upstream.len())?;
    // 空レイアウト（numel == 0）は走査対象の要素が無い。`axis_len` は
    // `[usize::MAX, 0]` のように巨大でありうるため、作業バッファ確保前に空勾配を返す
    // （巨大確保による OOM・panic を避ける。境界検査の契約）。
    if layout.numel == 0 {
        return Ok(Vec::new());
    }
    let mut d_x = vec![0.0_f32; layout.numel];
    let n = layout.axis_len;
    let mut m = vec![0.0_f64; n];
    let mut z = vec![0.0_f64; n];
    let mut lead = vec![0.0_f64; n];
    for o in 0..layout.outer {
        for i in 0..layout.inner {
            // 非有限（NaN・±inf）を含む lane は従来どおり f64 の prefix の差から係数を作る。
            // 共通オフセットで係数が潰れる問題は有限入力の話で、非有限 lane は式の IEEE 伝播の
            // クラス（PyTorch との一致数を含む。`cumulative_parity`・決定記録 §5）を変えないため。
            let finite = (0..n).all(|a| x[layout.pos(o, a, i)].is_finite());
            if finite {
                for a in 0..n {
                    let v = f64::from(x[layout.pos(o, a, i)]);
                    if a == 0 {
                        m[0] = v;
                        z[0] = 1.0;
                        continue;
                    }
                    let mk = m[a - 1].max(v);
                    lead[a] = z[a - 1] * (m[a - 1] - mk).exp();
                    z[a] = lead[a] + (v - mk).exp();
                    m[a] = mk;
                }
            } else {
                for a in 0..n {
                    let v = f64::from(x[layout.pos(o, a, i)]);
                    m[a] = if a == 0 { v } else { log_add_exp(m[a - 1], v) };
                }
            }
            let mut t = 0.0_f64;
            for a in (0..n).rev() {
                let p = layout.pos(o, a, i);
                let g = f64::from(upstream[p]);
                let xv = f64::from(x[p]);
                let (c, w) = match (finite, a + 1 < n) {
                    (true, true) => (lead[a + 1] / z[a + 1], (xv - m[a]).exp() / z[a]),
                    (true, false) => (0.0, (xv - m[a]).exp() / z[a]),
                    (false, true) => ((m[a] - m[a + 1]).exp(), (xv - m[a]).exp()),
                    (false, false) => (0.0, (xv - m[a]).exp()),
                };
                t = if a + 1 == n { g } else { g + c * t };
                d_x[p] = (w * t) as f32;
            }
        }
    }
    Ok(d_x)
}

/// 修正前（#3965 以前）の実装。`out` の f64 prefix の差から係数を作る。回帰テストで新方式との
/// NaN／inf のクラス一致と通常入力での差を実測するためだけに `cfg(test)` で残す。
#[cfg(test)]
fn logcumsumexp_vjp_legacy(x: &[f32], upstream: &[f32], layout: &CumulativeLayout) -> Vec<f32> {
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
    d_x
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
    fn vjp_on_empty_layout_does_not_allocate_axis_prefix() {
        // axis_len が巨大でも numel == 0 なら確保せず空勾配を返す。
        for shape in [[usize::MAX, 0], [1 << 40, 0]] {
            let l = layout(&shape, 0);
            assert_eq!(l.numel(), 0);
            assert_eq!(
                logcumsumexp_vjp_host(&[], &[], &l).unwrap(),
                Vec::<f32>::new()
            );
            assert!(logcumsumexp_host(&[], &l).unwrap().is_empty());
        }
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

    /// f64 の二次オラクル。オフセット `m_k` を引いた差から `p_{jk} = exp(x_j - m_k)/Z_k` を作る
    /// （`out` 経由の素朴な式ではない）。`d_x[j] = Σ_{k>=j} g[k]·p_{jk}`。
    fn offset_free_oracle(x: &[f32], g: &[f32]) -> Vec<f64> {
        let xs: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let n = xs.len();
        let mut d = vec![0.0_f64; n];
        for k in 0..n {
            let mk = xs[..=k].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let zk: f64 = xs[..=k].iter().map(|v| (v - mk).exp()).sum();
            for j in 0..=k {
                d[j] += f64::from(g[k]) * (xs[j] - mk).exp() / zk;
            }
        }
        d
    }

    #[test]
    fn vjp_resists_common_offset() {
        let l2 = layout(&[2], 0);
        let d = logcumsumexp_vjp_host(&[1e30, 1e30], &[0.0, 1.0], &l2).unwrap();
        assert_eq!(d, vec![0.5, 0.5]);
        let d = logcumsumexp_vjp_host(&[1e30, 1e30], &[1.0, 1.0], &l2).unwrap();
        assert_eq!(d, vec![1.5, 0.5]);
        let cases: [&[f32]; 3] = [
            &[3e20, 3e20, 3e20],
            &[0.0, 2e20, 2e20, 1.0],
            &[-3.0, 1.0, 5e19, 5e19, 5e19],
        ];
        for x in cases {
            let g: Vec<f32> = (0..x.len()).map(|i| 0.5 + 0.25 * i as f32).collect();
            let d = logcumsumexp_vjp_host(x, &g, &layout(&[x.len()], 0)).unwrap();
            let want = offset_free_oracle(x, &g);
            for j in 0..x.len() {
                assert!(
                    (f64::from(d[j]) - want[j]).abs() <= 1e-6 * want[j].abs().max(1.0),
                    "x={x:?} j={j} d={} want={}",
                    d[j],
                    want[j]
                );
            }
        }
        // 旧実装は誤る（[1,1]）ことを固定し、回帰テストが検出力を持つことを示す。
        assert_eq!(
            logcumsumexp_vjp_legacy(&[1e30, 1e30], &[0.0, 1.0], &l2),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn vjp_wide_range_and_translation_invariance() {
        let l = layout(&[2], 0);
        let d = logcumsumexp_vjp_host(&[-1000.0, 1000.0], &[1.0, 1.0], &l).unwrap();
        assert!(d.iter().all(|v| v.is_finite()));
        // f32 で厳密な +128 シフトでも同じ値（共通オフセット不変）。
        let x = [0.5_f32, -1.25, 2.0, 0.75, -3.0];
        let g = [0.3_f32, -0.7, 1.1, 0.2, -0.4];
        let xs: Vec<f32> = x.iter().map(|v| v + 128.0).collect();
        let l = layout(&[5], 0);
        let a = logcumsumexp_vjp_host(&x, &g, &l).unwrap();
        let b = logcumsumexp_vjp_host(&xs, &g, &l).unwrap();
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() <= 1e-6 * p.abs().max(1.0), "{p} vs {q}");
        }
    }

    #[test]
    fn vjp_multi_lane_all_dims_match_legacy_and_are_deterministic() {
        // rank 3・全 dim で lane の取り違えが無いこと。
        let shape = [2_usize, 3, 4];
        let n: usize = shape.iter().product();
        let x: Vec<f32> = (0..n).map(|i| ((i * 7 % 11) as f32) * 0.4 - 2.0).collect();
        let g: Vec<f32> = (0..n).map(|i| ((i * 5 % 13) as f32) * 0.2 - 1.0).collect();
        for dim in 0..3 {
            let l = layout(&shape, dim);
            let d = logcumsumexp_vjp_host(&x, &g, &l).unwrap();
            let legacy = logcumsumexp_vjp_legacy(&x, &g, &l);
            for (p, q) in d.iter().zip(&legacy) {
                assert!((p - q).abs() <= 1e-5 * q.abs().max(1.0), "dim={dim}");
            }
            let again = logcumsumexp_vjp_host(&x, &g, &l).unwrap();
            assert!(
                d.iter()
                    .zip(&again)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }

    #[test]
    fn vjp_matches_legacy_on_ordinary_inputs_and_nonfinite_classes() {
        // 決定的な LCG。通常入力では旧実装と REQ-2 複合判定内で一致する。
        let mut s = 0x1234_5678_u64;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / 16777216.0) * 2.0 - 1.0
        };
        let mut bit_diff = 0usize;
        let mut total = 0usize;
        let mut max_ulp = 0u32;
        for n in 1..=64usize {
            for scale in [0.5_f32, 8.0, 40.0] {
                let x: Vec<f32> = (0..n).map(|_| rnd() * scale).collect();
                let g: Vec<f32> = (0..n).map(|_| rnd()).collect();
                let l = layout(&[n], 0);
                let d = logcumsumexp_vjp_host(&x, &g, &l).unwrap();
                let o = logcumsumexp_vjp_legacy(&x, &g, &l);
                for (p, q) in d.iter().zip(&o) {
                    total += 1;
                    if p.to_bits() != q.to_bits() {
                        bit_diff += 1;
                        if p.is_sign_negative() == q.is_sign_negative() {
                            max_ulp = max_ulp.max(p.to_bits().abs_diff(q.to_bits()));
                        }
                    }
                    let diff = (p - q).abs();
                    assert!(diff < 1e-5 || diff < 1e-3 * q.abs(), "n={n} {p} vs {q}");
                }
            }
        }
        eprintln!("legacy bit diff: {bit_diff}/{total}, max ulp {max_ulp}");
        // 非有限: NaN／有限のクラスが旧実装と一致する。
        let nan = f32::NAN;
        for x in [
            vec![1.0_f32, nan, 2.0],
            vec![nan, 1.0, 2.0],
            vec![0.0, 1.0, nan],
        ] {
            let g = vec![1.0_f32; x.len()];
            let l = layout(&[x.len()], 0);
            let d = logcumsumexp_vjp_host(&x, &g, &l).unwrap();
            let o = logcumsumexp_vjp_legacy(&x, &g, &l);
            let cls = |v: &[f32]| v.iter().map(|a| a.is_nan()).collect::<Vec<_>>();
            assert_eq!(cls(&d), cls(&o), "x={x:?}");
        }
        for x in [
            [f32::INFINITY, 0.0],
            [f32::NEG_INFINITY, 0.0],
            [0.0, f32::INFINITY],
        ] {
            let l = layout(&[2], 0);
            let d = logcumsumexp_vjp_host(&x, &[1.0, 1.0], &l).unwrap();
            let o = logcumsumexp_vjp_legacy(&x, &[1.0, 1.0], &l);
            assert!(
                d.iter()
                    .zip(&o)
                    .all(|(p, q)| p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan())),
                "inf case {x:?}: new={d:?} legacy={o:?}"
            );
        }
    }
}
