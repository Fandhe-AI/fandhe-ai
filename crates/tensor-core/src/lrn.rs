//! LocalResponseNorm（`F.local_response_norm`）のホスト参照カーネルの**単一情報源**
//! （イシュー #2646・親 #2625「Phase 4」・ルート #2499。実装記録は
//! `docs/autodiff-lrn-weight-reparam-decision.md`）。
//!
//! # 役割と呼び出し元
//!
//! - `autodiff::lrn_ops`（`BackendOps::lrn_forward` が `Unsupported` のときのホストフォール
//!   バック）・`autodiff::grad`（`Op::LocalResponseNorm` の VJP）・`backend-cpu` の
//!   `CpuBackendOps::lrn_forward` がいずれも本モジュールの関数を直接呼ぶ。窓定義・アキュムレータ
//!   契約をクレート間で複製せず、乖離を構造的に排除する。
//! - CUDA／Metal の専用カーネルは本イシューの対象外で、`BackendOps` 既定の `Unsupported` から
//!   autodiff 側が本ホスト実装へフォールバックする。
//!
//! # 定義（PyTorch の実装に一致）
//!
//! 入力 `[N, C, *S]`（rank 3 以上。チャネル軸は dim 1）に対し
//! `y_c = x_c / d_c^β`、`d_c = k + (α / size) · S_c`、`S_c = Σ_{j ∈ W(c)} x_j²`、
//! `W(c) = [c − ⌊size/2⌋, c + ⌊(size−1)/2⌋] ∩ [0, C)`。**除数は境界でも常に `size`**（ゼロ
//! padding を数える）。**偶数 `size` は前後非対称**（前 `size/2`・後 `(size−1)/2`）。
//!
//! # 数値契約
//!
//! 窓内の二乗和は要素を**先に `f64` へ昇格してから二乗**し index 順に `f64` で蓄積する
//! （`f32` のまま二乗すると有限入力でも溢れうるため。`.claude/rules/coding-rust.md`）。`d`・
//! `d^β`（`f64::powf`）・除算まで `f64` のまま、最後に 1 回だけ `f32` へ downcast する。窓和は
//! スライディング差分にせず窓ごとに直接加算する（非負和の引き算による桁落ちを避ける）。
//! `mul_add` は使わない（matmul 系 FMA 契約には触れない）。
//! VJP は `dx_m = g_m·d_m^{−β} − (2αβ/size)·x_m·Σ_{c ∈ W'(m)} g_c·x_c·d_c^{−β−1}`、
//! `W'(m) = [m − ⌊(size−1)/2⌋, m + ⌊size/2⌋] ∩ [0, C)`（forward 窓の反転。偶数 `size` で
//! forward と異なる）。要素積 `g_c·x_c` は `f32` で確定してから `f64` へ昇格する
//! （`grad.rs::rmsnorm_vjp_rows` と同じ契約形）。非有限入力は拒否せず伝播する。
//!
//! # 境界検査（REQ-8・OWASP A03）
//!
//! パラメータは [`LrnParams::new`]、形状は [`lrn_layout`]（rank・要素数／バイト数の
//! `checked_*`）で確保より前に検査する。カーネルは入力スライス長を再検査し、窓端を
//! `saturating_sub`／`min` で手動クリップする。`unsafe`／`get_unchecked` は使わない。

use crate::device::BackendError;
use crate::error::ShapeError;
use crate::tensor::checked_numel_for;

/// LocalResponseNorm のパラメータ記述子。フィールドは [`LrnParams::new`] 経由でのみ設定する
/// （`#[non_exhaustive]` で将来のフィールド追加を非破壊にする）。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct LrnParams {
    size: usize,
    alpha: f32,
    beta: f32,
    k: f32,
}

impl LrnParams {
    /// `size == 0` と非有限の `alpha`／`beta`／`k` を [`BackendError::InvalidArgument`] で拒否する
    /// （PyTorch は無検査。差分は決定記録 §5）。`k <= 0`・負の `alpha` は拒否せず結果へ伝播する。
    pub fn new(size: usize, alpha: f32, beta: f32, k: f32) -> Result<Self, BackendError> {
        if size == 0 {
            return Err(BackendError::InvalidArgument(
                "LrnParams::new: size は 1 以上である必要がある".into(),
            ));
        }
        if !(alpha.is_finite() && beta.is_finite() && k.is_finite()) {
            return Err(BackendError::InvalidArgument(format!(
                "LrnParams::new: alpha／beta／k は有限である必要がある（alpha={alpha}, beta={beta}, k={k}）"
            )));
        }
        Ok(Self {
            size,
            alpha,
            beta,
            k,
        })
    }

    /// 窓幅（チャネル軸）。
    pub fn size(&self) -> usize {
        self.size
    }

    /// `alpha`。
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// `beta`。
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// `k`。
    pub fn k(&self) -> f32 {
        self.k
    }
}

/// 検査済みの入力レイアウト（`[N, C, *S]` を `(n, c, inner)` へ畳んだもの）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LrnLayout {
    shape: Vec<usize>,
    n: usize,
    c: usize,
    inner: usize,
}

impl LrnLayout {
    /// 入力（＝出力）shape。
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 総要素数。
    pub fn numel(&self) -> usize {
        self.n * self.c * self.inner
    }
}

/// rank 3 以上・要素数／バイト数の確保前検査を行いレイアウトを返す。
pub fn lrn_layout(shape: &[usize]) -> Result<LrnLayout, ShapeError> {
    if shape.len() < 3 {
        return Err(ShapeError::RankMismatch {
            expected: 3,
            actual: shape.len(),
        });
    }
    checked_numel_for::<f32>(shape)?;
    // `checked_numel_for` 通過後なので以下の積は溢れない。
    let inner = shape[2..].iter().product();
    Ok(LrnLayout {
        shape: shape.to_vec(),
        n: shape[0],
        c: shape[1],
        inner,
    })
}

fn check_len(len: usize, layout: &LrnLayout) -> Result<(), ShapeError> {
    if len != layout.numel() {
        return Err(ShapeError::ElementCountMismatch {
            expected: layout.numel(),
            actual: len,
        });
    }
    Ok(())
}

/// 1 列（`(n, s)` 固定のチャネル方向）の `d_c`（`f64`）を `out` へ書く。
fn column_denominators(col: &[f32], size: usize, alpha: f32, k: f32, out: &mut [f64]) {
    let c = col.len();
    let lo_span = size / 2;
    let hi_span = (size - 1) / 2;
    let coeff = f64::from(alpha) / size as f64;
    for (ci, slot) in out.iter_mut().enumerate().take(c) {
        let lo = ci.saturating_sub(lo_span);
        let hi = ci.saturating_add(hi_span).min(c - 1);
        let mut sum = 0.0_f64;
        for &x in &col[lo..=hi] {
            let xd = f64::from(x);
            sum += xd * xd;
        }
        *slot = f64::from(k) + coeff * sum;
    }
}

/// forward。出力は入力と同 shape の論理順（row-major）`Vec`。
pub fn local_response_norm_host(
    x: &[f32],
    layout: &LrnLayout,
    params: &LrnParams,
) -> Result<Vec<f32>, ShapeError> {
    check_len(x.len(), layout)?;
    let (n, c, inner) = (layout.n, layout.c, layout.inner);
    let mut out = vec![0.0_f32; layout.numel()];
    if out.is_empty() {
        return Ok(out);
    }
    let beta = f64::from(params.beta);
    let mut col = vec![0.0_f32; c];
    let mut d = vec![0.0_f64; c];
    for ni in 0..n {
        for s in 0..inner {
            for (ci, slot) in col.iter_mut().enumerate() {
                *slot = x[(ni * c + ci) * inner + s];
            }
            column_denominators(&col, params.size, params.alpha, params.k, &mut d);
            for ci in 0..c {
                let y = f64::from(col[ci]) / d[ci].powf(beta);
                out[(ni * c + ci) * inner + s] = y as f32;
            }
        }
    }
    Ok(out)
}

/// VJP。`x`（forward 入力）と `upstream` から `dx` を返す。統計 `d` は forward と同じ規則で再計算する。
pub fn local_response_norm_vjp_host(
    x: &[f32],
    upstream: &[f32],
    layout: &LrnLayout,
    params: &LrnParams,
) -> Result<Vec<f32>, ShapeError> {
    check_len(x.len(), layout)?;
    check_len(upstream.len(), layout)?;
    let (n, c, inner) = (layout.n, layout.c, layout.inner);
    let mut out = vec![0.0_f32; layout.numel()];
    if out.is_empty() {
        return Ok(out);
    }
    let beta = f64::from(params.beta);
    let size = params.size;
    let cross = 2.0 * f64::from(params.alpha) * beta / size as f64;
    // forward 窓の反転。
    let lo_span = (size - 1) / 2;
    let hi_span = size / 2;
    let mut col = vec![0.0_f32; c];
    let mut gcol = vec![0.0_f32; c];
    let mut d = vec![0.0_f64; c];
    let mut t = vec![0.0_f64; c];
    for ni in 0..n {
        for s in 0..inner {
            for ci in 0..c {
                let idx = (ni * c + ci) * inner + s;
                col[ci] = x[idx];
                gcol[ci] = upstream[idx];
            }
            column_denominators(&col, size, params.alpha, params.k, &mut d);
            for ci in 0..c {
                // 要素積は f32 で確定してから f64 へ昇格する（rmsnorm_vjp_rows と同じ契約形）。
                let prod = gcol[ci] * col[ci];
                t[ci] = f64::from(prod) * d[ci].powf(-beta - 1.0);
            }
            for m in 0..c {
                let lo = m.saturating_sub(lo_span);
                let hi = m.saturating_add(hi_span).min(c - 1);
                let mut acc = 0.0_f64;
                for &tv in &t[lo..=hi] {
                    acc += tv;
                }
                let dx = f64::from(gcol[m]) * d[m].powf(-beta) - cross * f64::from(col[m]) * acc;
                out[(ni * c + m) * inner + s] = dx as f32;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(x: &[f32], shape: &[usize], size: usize, a: f32, b: f32, k: f32) -> Vec<f32> {
        let l = lrn_layout(shape).unwrap();
        let p = LrnParams::new(size, a, b, k).unwrap();
        local_response_norm_host(x, &l, &p).unwrap()
    }

    #[test]
    fn size_one_is_pointwise() {
        // d = k + alpha * x^2、beta = 1 → y = x / (1 + x^2)（size = 1 は除数 1）。
        let y = run(&[2.0, 3.0], &[1, 2, 1], 1, 1.0, 1.0, 1.0);
        assert!((y[0] - 2.0 / 5.0).abs() < 1e-6);
        assert!((y[1] - 3.0 / 10.0).abs() < 1e-6);
    }

    #[test]
    fn odd_window_hand_computed_with_zero_padding_divisor() {
        // C=3・size=3・alpha=3・beta=1・k=0: d_c = S_c（除数 size=3 と alpha=3 が相殺）。
        // c=0: W={0,1}→1+4=5 / c=1: W={0,1,2}→14 / c=2: W={1,2}→13。
        let y = run(&[1.0, 2.0, 3.0], &[1, 3, 1], 3, 3.0, 1.0, 0.0);
        let want = [1.0 / 5.0, 2.0 / 14.0, 3.0 / 13.0];
        for (a, b) in y.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn even_window_is_asymmetric_front_heavy() {
        // size=2: W(c) = [c-1, c]。alpha=2・beta=1・k=0 → d_c = S_c。
        // c=0: {0}→1 / c=1: {0,1}→5 / c=2: {1,2}→13。
        let y = run(&[1.0, 2.0, 3.0], &[1, 3, 1], 2, 2.0, 1.0, 0.0);
        let want = [1.0 / 1.0, 2.0 / 5.0, 3.0 / 13.0];
        for (a, b) in y.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn size_larger_than_channels_clips_window() {
        let y = run(&[1.0, 2.0], &[1, 2, 1], 7, 7.0, 1.0, 0.0);
        // 全チャネルが窓内: d = 1 + 4 = 5。
        assert!((y[0] - 1.0 / 5.0).abs() < 1e-6);
        assert!((y[1] - 2.0 / 5.0).abs() < 1e-6);
    }

    #[test]
    fn squares_do_not_overflow_in_f32() {
        // 2e20 の二乗は f32 では inf だが f64 昇格後に二乗するので有限（y は 0 に近い有限値）。
        let y = run(&[2e20, 1.0], &[1, 2, 1], 2, 1.0, 0.75, 1.0);
        assert!(y.iter().all(|v| v.is_finite()), "{y:?}");
    }

    #[test]
    fn nonfinite_input_propagates() {
        let y = run(&[f32::NAN, 1.0], &[1, 2, 1], 2, 1.0, 0.75, 1.0);
        assert!(y[0].is_nan() && y[1].is_nan());
    }

    #[test]
    fn empty_tensors_yield_empty_output() {
        assert!(run(&[], &[0, 3, 2], 3, 1.0, 0.75, 1.0).is_empty());
        assert!(run(&[], &[2, 0, 2], 3, 1.0, 0.75, 1.0).is_empty());
        assert!(run(&[], &[2, 3, 0], 3, 1.0, 0.75, 1.0).is_empty());
    }

    #[test]
    fn invalid_params_and_layouts_are_rejected() {
        assert!(LrnParams::new(0, 1.0, 1.0, 1.0).is_err());
        assert!(LrnParams::new(1, f32::NAN, 1.0, 1.0).is_err());
        assert!(LrnParams::new(1, 1.0, f32::INFINITY, 1.0).is_err());
        assert!(LrnParams::new(1, 1.0, 1.0, f32::NEG_INFINITY).is_err());
        assert!(lrn_layout(&[2, 3]).is_err());
        assert!(lrn_layout(&[usize::MAX, 2, 2]).is_err());
        let l = lrn_layout(&[1, 2, 2]).unwrap();
        let p = LrnParams::new(2, 1.0, 1.0, 1.0).unwrap();
        assert!(local_response_norm_host(&[0.0; 3], &l, &p).is_err());
        assert!(local_response_norm_vjp_host(&[0.0; 4], &[0.0; 3], &l, &p).is_err());
    }

    #[test]
    fn vjp_matches_central_difference_for_even_and_odd_sizes() {
        for size in [1_usize, 2, 3, 4, 5] {
            let shape = [2, 4, 2];
            let n: usize = shape.iter().product();
            let x: Vec<f32> = (0..n).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3).collect();
            let g: Vec<f32> = (0..n).map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.2).collect();
            let l = lrn_layout(&shape).unwrap();
            let p = LrnParams::new(size, 0.5, 0.75, 2.0).unwrap();
            let dx = local_response_norm_vjp_host(&x, &g, &l, &p).unwrap();
            let loss = |xs: &[f32]| -> f64 {
                local_response_norm_host(xs, &l, &p)
                    .unwrap()
                    .iter()
                    .zip(&g)
                    .map(|(&y, &gv)| f64::from(y) * f64::from(gv))
                    .sum()
            };
            for i in 0..n {
                let h = 1e-2_f32;
                let mut xp = x.clone();
                xp[i] += h;
                let mut xm = x.clone();
                xm[i] -= h;
                let num = (loss(&xp) - loss(&xm)) / (2.0 * f64::from(h));
                assert!(
                    (num - f64::from(dx[i])).abs() < 5e-3,
                    "size={size} i={i} num={num} an={}",
                    dx[i]
                );
            }
        }
    }
}
