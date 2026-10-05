//! `rfft`／`irfft` のホスト参照カーネルと正規化種別 [`FftNorm`] の
//! **単一情報源**（イシュー #2631・親 #2630。設計は
//! `docs/autodiff-fft-design.md` 案 B・実装記録は
//! `docs/autodiff-fft-ops-decision.md`）。
//!
//! # 役割と呼び出し元
//!
//! - `autodiff::fft_ops`（`BackendOps` が `Unsupported` のときのホスト
//!   フォールバック）・`autodiff::grad`（VJP）・`backend-cpu` の
//!   `CpuBackendOps::fft_rfft`／`fft_irfft` がいずれも本モジュールの関数を
//!   直接呼ぶ。座標・twiddle・総和順序をクレート間で複製せず乖離を構造的に
//!   排除する（`interpolate.rs` と同じ「共有可能な層に一本化する」方式）。
//! - 複素数は末尾次元 2 の `f32` 実テンソル対 `(re, im)`
//!   （`torch.view_as_real` と同レイアウト）で表す。complex dtype は
//!   非目標のまま（`docs/tensor-core-sparse-complex-decision.md` §6）。
//!
//! # 数値契約（`docs/autodiff-fft-design.md` §4・§6）
//!
//! - 内部は `f64` で**逐次・固定順序**に蓄積し、出力時に 1 回だけ `f32` へ
//!   downcast する。蓄積は素の `acc += a * b`（`mul_add` を使わない。`f64`
//!   内部精度のため FMA の有無は REQ-2 統一複合判定の範囲内。matmul 系の
//!   FMA 契約には触れない）。同一入力は run-to-run で bit 決定的。
//! - twiddle は長さ `n` の `(cos, sin)` 表を 1 変換につき 1 回だけ作り、
//!   `(k·j) mod n` の添字で引く（添字は増分更新で、乗算オーバーフローを
//!   構造的に避ける）。整数個の 1/4 turn と 1/2 turn は `sin_cos` を使わず
//!   厳密値（`0`／`±1`）を格納する。`sin_cos` は `libm` 依存のため、
//!   クレート間 bit 同一は受入条件にしない。
//! - DC と（`n` 偶数の）Nyquist の虚部は**蓄積結果ではなくリテラル
//!   `+0.0`** を書き込む（rfft forward・irfft VJP）。厳密 twiddle に加えた
//!   実装上の補強で、`-0.0` と非有限入力での `inf·0 = NaN` を避ける。
//! - 非有限入力は事前に拒否せず伝播する（設計 §6）。
//! - 計算量は 1 レーンあたり O(n²)（直接 DFT。O(n log n) 化は対象外）。
//!
//! # 境界検査（REQ-8・OWASP A03）
//!
//! 形状・引数の検査は [`rfft_layout`]／[`irfft_layout`] に集約し、確保より
//! 前に rank・`dim`・`n`・要素数／バイト数の `checked_mul` を検査して型付き
//! エラー（[`FftError`]）で拒否する。カーネルはスライス長も再検査し、
//! `unsafe`／`get_unchecked` を使わない。

use crate::device::BackendError;
use crate::error::ShapeError;

/// FFT の正規化種別（`torch.fft` の `norm` 引数相当）。
///
/// 順変換（`rfft`）のスケール `s` と逆変換（`irfft`）のスケール `t` は
/// `n` を変換長として次の値になる。
///
/// | 種別 | `s`（順） | `t`（逆） |
/// |---|---|---|
/// | [`Self::Backward`]（既定） | 1 | 1/n |
/// | [`Self::Ortho`] | 1/√n | 1/√n |
/// | [`Self::Forward`] | 1/n | 1 |
///
/// 下流クレートが `match` せずに済むよう、係数は [`Self::forward_scale`]・
/// [`Self::inverse_scale`] で提供する。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum FftNorm {
    /// 順変換スケールなし・逆変換 `1/n`（PyTorch 既定の `"backward"`）。
    #[default]
    Backward,
    /// 両方向とも `1/√n`（`"ortho"`）。
    Ortho,
    /// 順変換 `1/n`・逆変換スケールなし（`"forward"`）。
    Forward,
}

impl FftNorm {
    /// 順変換（`rfft`）に掛けるスケール `s`（`n >= 1`）。
    pub fn forward_scale(self, n: usize) -> f64 {
        match self {
            FftNorm::Backward => 1.0,
            FftNorm::Ortho => 1.0 / (n as f64).sqrt(),
            FftNorm::Forward => 1.0 / n as f64,
        }
    }

    /// 逆変換（`irfft`）に掛けるスケール `t`（`n >= 1`）。
    pub fn inverse_scale(self, n: usize) -> f64 {
        match self {
            FftNorm::Backward => 1.0 / n as f64,
            FftNorm::Ortho => 1.0 / (n as f64).sqrt(),
            FftNorm::Forward => 1.0,
        }
    }
}

/// 形状・引数検査の失敗（型付き）。`autodiff` は `AutodiffError` へ、
/// バックエンドは [`BackendError`] へ写像する。
#[derive(Debug, Clone, PartialEq)]
pub enum FftError {
    /// rank 不足・末尾次元 ≠ 2・要素数／バイト数オーバーフロー等の形状違反。
    Shape(ShapeError),
    /// `n`・`dim`・既定 `n` が 0 になる入力など、引数の不正。
    InvalidArgument(String),
}

impl std::fmt::Display for FftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FftError::Shape(e) => write!(f, "fft: shape error: {e:?}"),
            FftError::InvalidArgument(msg) => write!(f, "fft: invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for FftError {}

impl From<FftError> for BackendError {
    fn from(err: FftError) -> Self {
        match err {
            FftError::Shape(e) => BackendError::ShapeMismatch(e),
            FftError::InvalidArgument(msg) => BackendError::InvalidArgument(msg),
        }
    }
}

/// 解決済みの変換レイアウト（[`rfft_layout`]／[`irfft_layout`] の戻り値）。
///
/// 入出力はいずれも連続配置の行優先。`dim` 軸の前を `outer`・後ろ（複素
/// 軸を除く）を `inner` として 1 レーン = `(outer 添字, inner 添字)` を
/// 独立に変換する。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftLayout {
    /// 解決済みの変換長 `n`（`>= 1`）。
    pub n: usize,
    /// 変換軸（実軸の添字）。
    pub dim: usize,
    /// `dim` より前の次元の積。
    pub outer: usize,
    /// `dim` より後ろ（複素軸を除く）の次元の積。
    pub inner: usize,
    /// 入力の `dim` 軸長（rfft は実長 `L`・irfft は bin 数 `m`）。
    pub in_len: usize,
    /// 順変換の出力形状（rfft は複素・irfft は実）。VJP では上流勾配の形状。
    pub out_shape: Vec<usize>,
}

impl FftLayout {
    /// 片側スペクトルの bin 数 `n/2 + 1`。
    pub fn bins(&self) -> usize {
        self.n / 2 + 1
    }
}

fn checked_numel(shape: &[usize]) -> Result<usize, FftError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(FftError::Shape(ShapeError::ElementCountOverflow))
}

/// 要素数が `f32` で確保可能（バイト数 `<= isize::MAX`）であることを検査する。
fn check_f32_alloc(shape: &[usize]) -> Result<usize, FftError> {
    let numel = checked_numel(shape)?;
    let bytes = numel
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(FftError::Shape(ShapeError::ElementCountOverflow))?;
    if bytes > isize::MAX as usize {
        return Err(FftError::Shape(ShapeError::ElementCountOverflow));
    }
    Ok(numel)
}

/// `rfft`（実入力 `[..., L, ...]` → `[..., n/2+1, ..., 2]`）の引数を解決・検査する。
///
/// `n` 省略時は `L`、`dim` 省略時は末尾軸。`n >= 1` を要求する（`n` 省略かつ
/// `L == 0` は `n = 0` として拒否。`torch.fft.rfft` と同じ。`L == 0` でも `n` を
/// 明示すればゼロ詰めとして受理し、これも torch 2.14.0 の実測と一致する）。
pub fn rfft_layout(
    in_shape: &[usize],
    n: Option<usize>,
    dim: Option<usize>,
) -> Result<FftLayout, FftError> {
    let rank = in_shape.len();
    if rank < 1 {
        return Err(FftError::Shape(ShapeError::RankMismatch {
            expected: 1,
            actual: rank,
        }));
    }
    let dim = dim.unwrap_or(rank - 1);
    if dim >= rank {
        return Err(FftError::InvalidArgument(format!(
            "rfft: dim {dim} は rank {rank} の範囲外"
        )));
    }
    let len = in_shape[dim];
    let n = n.unwrap_or(len);
    if n == 0 {
        return Err(FftError::InvalidArgument(
            "rfft: 変換長 n は 1 以上である必要がある".into(),
        ));
    }
    let mut out_shape = in_shape.to_vec();
    out_shape[dim] = n / 2 + 1;
    out_shape.push(2);
    check_f32_alloc(in_shape)?;
    check_f32_alloc(&out_shape)?;
    Ok(FftLayout {
        n,
        dim,
        outer: checked_numel(&in_shape[..dim])?,
        inner: checked_numel(&in_shape[dim + 1..])?,
        in_len: len,
        out_shape,
    })
}

/// `irfft`（複素入力 `[..., m, ..., 2]` → 実 `[..., n, ...]`）の引数を解決・検査する。
///
/// `dim` は末尾の複素軸を除いた実軸の添字（既定は rank-2）。`n` 省略時は
/// `2*(m-1)`（`m == 1` で省略すると 0 になるため拒否する）。
pub fn irfft_layout(
    in_shape: &[usize],
    n: Option<usize>,
    dim: Option<usize>,
) -> Result<FftLayout, FftError> {
    let rank = in_shape.len();
    if rank < 2 {
        return Err(FftError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: rank,
        }));
    }
    if in_shape[rank - 1] != 2 {
        let mut expected = in_shape.to_vec();
        expected[rank - 1] = 2;
        return Err(FftError::Shape(ShapeError::ShapeMismatch {
            lhs: in_shape.to_vec(),
            rhs: expected,
        }));
    }
    let dim = dim.unwrap_or(rank - 2);
    if dim >= rank - 1 {
        return Err(FftError::InvalidArgument(format!(
            "irfft: dim {dim} は複素軸を除いた rank {} の範囲外",
            rank - 1
        )));
    }
    let m = in_shape[dim];
    if m == 0 {
        return Err(FftError::InvalidArgument(
            "irfft: 変換軸の入力 bin 数は 1 以上である必要がある".into(),
        ));
    }
    let n = match n {
        Some(n) => n,
        None => 2 * (m - 1),
    };
    if n == 0 {
        return Err(FftError::InvalidArgument(
            "irfft: 変換長 n は 1 以上である必要がある（bin 数 1 で n を省略すると 0 になる）"
                .into(),
        ));
    }
    let mut out_shape = in_shape[..rank - 1].to_vec();
    out_shape[dim] = n;
    check_f32_alloc(in_shape)?;
    check_f32_alloc(&out_shape)?;
    Ok(FftLayout {
        n,
        dim,
        outer: checked_numel(&in_shape[..dim])?,
        inner: checked_numel(&in_shape[dim + 1..rank - 1])?,
        in_len: m,
        out_shape,
    })
}

/// 長さ `n` の `(cos, sin)` twiddle 表（`θ_t = 2π·t/n`）。
struct Twiddle {
    cos: Vec<f64>,
    sin: Vec<f64>,
}

impl Twiddle {
    /// 1/4 turn・1/2 turn は厳密値、それ以外は `sin_cos`。
    fn new(n: usize) -> Self {
        let mut cos = Vec::with_capacity(n);
        let mut sin = Vec::with_capacity(n);
        let quarter = if n % 4 == 0 { n / 4 } else { 0 };
        for t in 0..n {
            let (s, c) = if quarter != 0 && t % quarter == 0 {
                match t / quarter {
                    0 => (0.0, 1.0),
                    1 => (1.0, 0.0),
                    2 => (0.0, -1.0),
                    _ => (-1.0, 0.0),
                }
            } else if n % 2 == 0 && t == n / 2 {
                (0.0, -1.0)
            } else {
                (2.0 * std::f64::consts::PI * (t as f64) / (n as f64)).sin_cos()
            };
            cos.push(c);
            sin.push(s);
        }
        Twiddle { cos, sin }
    }
}

fn len_error(what: &str, expected: usize, actual: usize) -> FftError {
    FftError::InvalidArgument(format!(
        "{what}: スライス長 {actual} が形状の要素数 {expected} と一致しない"
    ))
}

/// bin `k` が DC または（`n` 偶数の）Nyquist か。
fn is_real_bin(k: usize, n: usize) -> bool {
    k == 0 || (n % 2 == 0 && k == n / 2)
}

/// `rfft` forward（実入力 → 片側スペクトル。出力は `layout.out_shape` の連続配置）。
///
/// `x` は入力形状の連続データ。変換前に `dim` 軸を長さ `n` へ切り詰め／
/// ゼロ詰めする。`Y_k = s·Σ_j x_j·exp(-2πi·kj/n)`（`k = 0..=n/2`）。
pub fn rfft_host(x: &[f32], layout: &FftLayout, norm: FftNorm) -> Result<Vec<f32>, FftError> {
    let (n, bins, len, inner) = (layout.n, layout.bins(), layout.in_len, layout.inner);
    let in_numel = checked_numel(&[layout.outer, len, inner])?;
    if x.len() != in_numel {
        return Err(len_error("rfft_host", in_numel, x.len()));
    }
    let out_numel = checked_numel(&layout.out_shape)?;
    let mut out = vec![0.0f32; out_numel];
    if out_numel == 0 {
        return Ok(out);
    }
    let tw = Twiddle::new(n);
    let scale = norm.forward_scale(n);
    let jmax = len.min(n);
    let mut xs: Vec<f64> = Vec::with_capacity(jmax);
    for o in 0..layout.outer {
        for i in 0..inner {
            xs.clear();
            xs.extend((0..jmax).map(|t| f64::from(x[(o * len + t) * inner + i])));
            for k in 0..bins {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                let mut idx = 0usize;
                for &v in &xs {
                    re += v * tw.cos[idx];
                    im += v * tw.sin[idx];
                    idx += k;
                    if idx >= n {
                        idx -= n;
                    }
                }
                let base = ((o * bins + k) * inner + i) * 2;
                out[base] = (scale * re) as f32;
                out[base + 1] = if is_real_bin(k, n) {
                    0.0
                } else {
                    (-(scale * im)) as f32
                };
            }
        }
    }
    Ok(out)
}

/// `rfft` の VJP。`g` は `layout.out_shape` の上流勾配、戻り値は入力形状
/// （`dim` 軸長 `L = layout.in_len`）の勾配。
///
/// `dx_j = s·Σ_{k=0}^{n/2} (g_k.re·cos θ − g_k.im·sin θ)`。`n` 倍は掛けない
/// （設計 §5）。`n < L` の余り位置は 0、`n > L` の超過分は破棄する。
pub fn rfft_vjp_host(g: &[f32], layout: &FftLayout, norm: FftNorm) -> Result<Vec<f32>, FftError> {
    let (n, bins, len, inner) = (layout.n, layout.bins(), layout.in_len, layout.inner);
    let g_numel = checked_numel(&layout.out_shape)?;
    if g.len() != g_numel {
        return Err(len_error("rfft_vjp_host", g_numel, g.len()));
    }
    let in_numel = checked_numel(&[layout.outer, len, inner])?;
    let mut out = vec![0.0f32; in_numel];
    if in_numel == 0 {
        return Ok(out);
    }
    let tw = Twiddle::new(n);
    let scale = norm.forward_scale(n);
    let jmax = len.min(n);
    let mut gs: Vec<(f64, f64)> = Vec::with_capacity(bins);
    for o in 0..layout.outer {
        for i in 0..inner {
            gs.clear();
            gs.extend((0..bins).map(|k| {
                let base = ((o * bins + k) * inner + i) * 2;
                (f64::from(g[base]), f64::from(g[base + 1]))
            }));
            for j in 0..jmax {
                let mut acc = 0.0f64;
                let mut idx = 0usize;
                for &(gr, gi) in &gs {
                    acc += gr * tw.cos[idx] - gi * tw.sin[idx];
                    idx += j;
                    if idx >= n {
                        idx -= n;
                    }
                }
                out[(o * len + j) * inner + i] = (scale * acc) as f32;
            }
        }
    }
    Ok(out)
}

/// `irfft` forward（片側スペクトル → 実出力。出力は `layout.out_shape`）。
///
/// 入力 bin を `n/2+1` 個へ切り詰め／ゼロ詰めし、Hermitian 折り畳み形
/// `x_j = t·[X_0.re + Σ_{k=1}^{⌈n/2⌉-1} 2(X_k.re·cos θ − X_k.im·sin θ)
/// + (n 偶数なら X_{n/2}.re·(−1)^j)]` を評価する。DC・Nyquist の虚部は
/// 読まない（PyTorch の C2R と同じ）。
pub fn irfft_host(x: &[f32], layout: &FftLayout, norm: FftNorm) -> Result<Vec<f32>, FftError> {
    let (n, m, inner) = (layout.n, layout.in_len, layout.inner);
    let in_numel = checked_numel(&[layout.outer, m, inner, 2])?;
    if x.len() != in_numel {
        return Err(len_error("irfft_host", in_numel, x.len()));
    }
    let out_numel = checked_numel(&layout.out_shape)?;
    let mut out = vec![0.0f32; out_numel];
    if out_numel == 0 {
        return Ok(out);
    }
    let tw = Twiddle::new(n);
    let scale = norm.inverse_scale(n);
    let used = m.min(layout.bins());
    let mut bins_buf: Vec<(f64, f64)> = Vec::with_capacity(used);
    for o in 0..layout.outer {
        for i in 0..inner {
            bins_buf.clear();
            bins_buf.extend((0..used).map(|k| {
                let base = ((o * m + k) * inner + i) * 2;
                (f64::from(x[base]), f64::from(x[base + 1]))
            }));
            for j in 0..n {
                let mut acc = bins_buf[0].0;
                let mut idx = 0usize;
                for (k, &(re, im)) in bins_buf.iter().enumerate().skip(1) {
                    idx += j;
                    if idx >= n {
                        idx -= n;
                    }
                    if n % 2 == 0 && k == n / 2 {
                        acc += re * tw.cos[idx];
                    } else {
                        acc += 2.0 * (re * tw.cos[idx] - im * tw.sin[idx]);
                    }
                }
                out[(o * n + j) * inner + i] = (scale * acc) as f32;
            }
        }
    }
    Ok(out)
}

/// `irfft` の VJP。`g` は `layout.out_shape`（実）の上流勾配、戻り値は入力形状
/// `[..., m, ..., 2]` の勾配。
///
/// `dX_k.re = t·c_k·Σ_j g_j·cos θ`・`dX_k.im = −t·c_k·Σ_j g_j·sin θ`。`c_k` は
/// DC と（`n` 偶数の）Nyquist で 1、それ以外で 2。使うのは**正規化なしの
/// 生の r2c** で、正規化済み `rfft` は再利用しない（`t` の二重適用を避ける。
/// 設計 §5）。DC・Nyquist の虚部勾配はリテラル `+0.0`。入力 bin が
/// `n/2+1` より多ければ余剰 bin の勾配は 0、少なければ切り詰める。
pub fn irfft_vjp_host(g: &[f32], layout: &FftLayout, norm: FftNorm) -> Result<Vec<f32>, FftError> {
    let (n, m, inner) = (layout.n, layout.in_len, layout.inner);
    let g_numel = checked_numel(&layout.out_shape)?;
    if g.len() != g_numel {
        return Err(len_error("irfft_vjp_host", g_numel, g.len()));
    }
    let in_numel = checked_numel(&[layout.outer, m, inner, 2])?;
    let mut out = vec![0.0f32; in_numel];
    if in_numel == 0 {
        return Ok(out);
    }
    let tw = Twiddle::new(n);
    let scale = norm.inverse_scale(n);
    let used = m.min(layout.bins());
    let mut gs: Vec<f64> = Vec::with_capacity(n);
    for o in 0..layout.outer {
        for i in 0..inner {
            gs.clear();
            gs.extend((0..n).map(|j| f64::from(g[(o * n + j) * inner + i])));
            for k in 0..used {
                let (mut sr, mut si) = (0.0f64, 0.0f64);
                let mut idx = 0usize;
                for &gv in &gs {
                    sr += gv * tw.cos[idx];
                    si += gv * tw.sin[idx];
                    idx += k;
                    if idx >= n {
                        idx -= n;
                    }
                }
                let ck = if is_real_bin(k, n) { 1.0 } else { 2.0 };
                let base = ((o * m + k) * inner + i) * 2;
                out[base] = (scale * ck * sr) as f32;
                out[base + 1] = if is_real_bin(k, n) {
                    0.0
                } else {
                    (-(scale * ck * si)) as f32
                };
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn rand_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..len).map(|_| lcg(&mut s)).collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    }

    #[test]
    fn twiddle_exact_quarter_turns() {
        for n in [1usize, 2, 4, 8, 6, 12] {
            let tw = Twiddle::new(n);
            assert_eq!((tw.cos[0], tw.sin[0]), (1.0, 0.0));
            if n % 2 == 0 {
                assert_eq!((tw.cos[n / 2], tw.sin[n / 2]), (-1.0, 0.0), "n={n}");
            }
            if n % 4 == 0 {
                assert_eq!((tw.cos[n / 4], tw.sin[n / 4]), (0.0, 1.0), "n={n}");
                assert_eq!((tw.cos[3 * n / 4], tw.sin[3 * n / 4]), (0.0, -1.0), "n={n}");
            }
        }
    }

    #[test]
    fn rfft_delta_is_constant() {
        let layout = rfft_layout(&[4], None, None).expect("layout");
        let y = rfft_host(&[1.0, 0.0, 0.0, 0.0], &layout, FftNorm::Backward).expect("rfft");
        assert_eq!(y, vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn rfft_cos_is_spectral_line() {
        let x: Vec<f32> = (0..8)
            .map(|j| (2.0 * std::f64::consts::PI * j as f64 / 8.0).cos() as f32)
            .collect();
        let layout = rfft_layout(&[8], None, None).expect("layout");
        let y = rfft_host(&x, &layout, FftNorm::Backward).expect("rfft");
        assert!((y[2] - 4.0).abs() < 1e-5);
        for k in [0usize, 2, 3, 4] {
            assert!(y[2 * k].abs() < 1e-5 && y[2 * k + 1].abs() < 1e-5, "k={k}");
        }
        assert!(y[3].abs() < 1e-5);
    }

    #[test]
    fn rfft_odd_n_matches_manual() {
        let layout = rfft_layout(&[3], None, None).expect("layout");
        let y = rfft_host(&[1.0, 2.0, 3.0], &layout, FftNorm::Backward).expect("rfft");
        assert_eq!(y.len(), 4);
        assert!((y[0] - 6.0).abs() < 1e-6 && y[1] == 0.0);
        assert!((y[2] + 1.5).abs() < 1e-6);
        assert!((y[3] - 0.866_025_4).abs() < 1e-6);
    }

    #[test]
    fn rfft_n1_and_n2() {
        let l1 = rfft_layout(&[1], None, None).expect("layout");
        assert_eq!(
            rfft_host(&[5.0], &l1, FftNorm::Backward).expect("rfft"),
            vec![5.0, 0.0]
        );
        let l2 = rfft_layout(&[2], None, None).expect("layout");
        let y = rfft_host(&[3.0, 1.0], &l2, FftNorm::Backward).expect("rfft");
        assert_eq!(y, vec![4.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn rfft_pad_and_truncate() {
        let padded = rfft_layout(&[2], Some(4), None).expect("layout");
        let a = rfft_host(&[1.0, 2.0], &padded, FftNorm::Backward).expect("rfft");
        let full = rfft_layout(&[4], None, None).expect("layout");
        let b = rfft_host(&[1.0, 2.0, 0.0, 0.0], &full, FftNorm::Backward).expect("rfft");
        assert_eq!(a, b);
        let trunc = rfft_layout(&[4], Some(2), None).expect("layout");
        let c = rfft_host(&[1.0, 2.0, 9.0, 9.0], &trunc, FftNorm::Backward).expect("rfft");
        let d = rfft_layout(&[2], None, None).expect("layout");
        let e = rfft_host(&[1.0, 2.0], &d, FftNorm::Backward).expect("rfft");
        assert_eq!(c, e);
    }

    #[test]
    fn nyquist_and_dc_imag_are_positive_zero_bits() {
        let layout = rfft_layout(&[6], None, None).expect("layout");
        let y = rfft_host(&rand_vec(6, 1), &layout, FftNorm::Backward).expect("rfft");
        assert_eq!(y[1].to_bits(), 0);
        assert_eq!(y[2 * 3 + 1].to_bits(), 0);
        let il = irfft_layout(&[4, 2], Some(6), None).expect("layout");
        let g = rand_vec(6, 2);
        let dx = irfft_vjp_host(&g, &il, FftNorm::Ortho).expect("vjp");
        assert_eq!(dx[1].to_bits(), 0);
        assert_eq!(dx[2 * 3 + 1].to_bits(), 0);
    }

    #[test]
    fn nonfinite_input_does_not_poison_dc_imag() {
        let layout = rfft_layout(&[4], None, None).expect("layout");
        let y =
            rfft_host(&[f32::INFINITY, 0.0, 0.0, 0.0], &layout, FftNorm::Backward).expect("rfft");
        assert_eq!(y[1].to_bits(), 0);
        assert_eq!(y[5].to_bits(), 0);
    }

    #[test]
    fn roundtrip_rfft_irfft() {
        for n in [1usize, 2, 3, 4, 5, 8, 9] {
            for norm in [FftNorm::Backward, FftNorm::Ortho, FftNorm::Forward] {
                let x = rand_vec(n, n as u64 + 7);
                let fl = rfft_layout(&[n], None, None).expect("layout");
                let y = rfft_host(&x, &fl, norm).expect("rfft");
                let il = irfft_layout(&fl.out_shape, Some(n), None).expect("layout");
                let z = irfft_host(&y, &il, norm).expect("irfft");
                for (a, b) in x.iter().zip(&z) {
                    assert!((a - b).abs() < 1e-5, "n={n} norm={norm:?}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn adjoint_identity_rfft() {
        for (len, n) in [(5usize, None), (4, Some(7)), (7, Some(4)), (6, Some(6))] {
            for norm in [FftNorm::Backward, FftNorm::Ortho, FftNorm::Forward] {
                let layout = rfft_layout(&[2, len], n, None).expect("layout");
                let x = rand_vec(2 * len, 3);
                let y = rfft_host(&x, &layout, norm).expect("rfft");
                let g = rand_vec(y.len(), 4);
                let dx = rfft_vjp_host(&g, &layout, norm).expect("vjp");
                let lhs = dot(&y, &g);
                let rhs = dot(&x, &dx);
                assert!(
                    (lhs - rhs).abs() < 1e-4,
                    "len={len} n={n:?} {norm:?}: {lhs} {rhs}"
                );
            }
        }
    }

    #[test]
    fn adjoint_identity_irfft() {
        for (m, n) in [
            (3usize, None),
            (3, Some(5)),
            (5, Some(4)),
            (2, Some(7)),
            (4, Some(6)),
        ] {
            for norm in [FftNorm::Backward, FftNorm::Ortho, FftNorm::Forward] {
                let layout = irfft_layout(&[2, m, 2], n, None).expect("layout");
                // 構造的に DC／Nyquist の虚部は読まれないため随伴恒等式は成立する。
                let x = rand_vec(2 * m * 2, 5);
                let y = irfft_host(&x, &layout, norm).expect("irfft");
                let g = rand_vec(y.len(), 6);
                let dx = irfft_vjp_host(&g, &layout, norm).expect("vjp");
                let lhs = dot(&y, &g);
                let rhs = dot(&x, &dx);
                assert!(
                    (lhs - rhs).abs() < 1e-4,
                    "m={m} n={n:?} {norm:?}: {lhs} {rhs}"
                );
            }
        }
    }

    #[test]
    fn non_last_dim_matches_transposed_lane() {
        // [3, 4] の dim=0 を、転置した [4, 3] の dim=1 と比較する。
        let a = rand_vec(12, 9);
        let mut t = vec![0.0f32; 12];
        for r in 0..3 {
            for c in 0..4 {
                t[c * 3 + r] = a[r * 4 + c];
            }
        }
        let l0 = rfft_layout(&[3, 4], None, Some(0)).expect("layout"); // out [2,4,2]
        let y0 = rfft_host(&a, &l0, FftNorm::Backward).expect("rfft");
        let l1 = rfft_layout(&[4, 3], None, Some(1)).expect("layout"); // out [4,2,2]
        let y1 = rfft_host(&t, &l1, FftNorm::Backward).expect("rfft");
        for k in 0..2 {
            for c in 0..4 {
                for p in 0..2 {
                    assert_eq!(y0[(k * 4 + c) * 2 + p], y1[(c * 2 + k) * 2 + p]);
                }
            }
        }
    }

    #[test]
    fn deterministic_bits() {
        let layout = rfft_layout(&[3, 9], None, None).expect("layout");
        let x = rand_vec(27, 11);
        let a = rfft_host(&x, &layout, FftNorm::Ortho).expect("rfft");
        let b = rfft_host(&x, &layout, FftNorm::Ortho).expect("rfft");
        assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
    }

    #[test]
    fn layout_errors_are_typed() {
        assert!(matches!(
            rfft_layout(&[], None, None),
            Err(FftError::Shape(ShapeError::RankMismatch { .. }))
        ));
        assert!(matches!(
            rfft_layout(&[4], Some(0), None),
            Err(FftError::InvalidArgument(_))
        ));
        assert!(matches!(
            rfft_layout(&[4], None, Some(1)),
            Err(FftError::InvalidArgument(_))
        ));
        assert!(matches!(
            rfft_layout(&[0], None, None),
            Err(FftError::InvalidArgument(_))
        ));
        // `L == 0` でも `n` 明示ならゼロ詰めとして受理する（torch と同じ）。
        let l = rfft_layout(&[0], Some(4), None).expect("layout");
        let y = rfft_host(&[], &l, FftNorm::Backward).expect("rfft");
        assert_eq!(y, vec![0.0; 6]);
        assert_eq!(
            rfft_vjp_host(&y, &l, FftNorm::Backward).expect("vjp"),
            Vec::<f32>::new()
        );
        assert!(matches!(
            irfft_layout(&[4], None, None),
            Err(FftError::Shape(ShapeError::RankMismatch { .. }))
        ));
        assert!(matches!(
            irfft_layout(&[4, 3], None, None),
            Err(FftError::Shape(ShapeError::ShapeMismatch { .. }))
        ));
        assert!(matches!(
            irfft_layout(&[1, 2], None, None),
            Err(FftError::InvalidArgument(_))
        ));
        assert!(matches!(
            irfft_layout(&[3, 2], None, Some(1)),
            Err(FftError::InvalidArgument(_))
        ));
        assert!(matches!(
            rfft_layout(&[4], Some(usize::MAX / 2), None),
            Err(FftError::Shape(ShapeError::ElementCountOverflow))
        ));
        assert!(matches!(
            irfft_layout(&[3, 2], Some(usize::MAX), None),
            Err(FftError::Shape(ShapeError::ElementCountOverflow))
        ));
    }

    #[test]
    fn kernel_rejects_wrong_slice_length() {
        let layout = rfft_layout(&[4], None, None).expect("layout");
        assert!(rfft_host(&[1.0; 3], &layout, FftNorm::Backward).is_err());
        assert!(rfft_vjp_host(&[1.0; 3], &layout, FftNorm::Backward).is_err());
        let il = irfft_layout(&[3, 2], None, None).expect("layout");
        assert!(irfft_host(&[1.0; 5], &il, FftNorm::Backward).is_err());
        assert!(irfft_vjp_host(&[1.0; 3], &il, FftNorm::Backward).is_err());
    }

    #[test]
    fn norm_scales() {
        assert_eq!(FftNorm::default(), FftNorm::Backward);
        assert_eq!(FftNorm::Backward.forward_scale(4), 1.0);
        assert_eq!(FftNorm::Backward.inverse_scale(4), 0.25);
        assert_eq!(FftNorm::Ortho.forward_scale(4), 0.5);
        assert_eq!(FftNorm::Forward.forward_scale(4), 0.25);
        assert_eq!(FftNorm::Forward.inverse_scale(4), 1.0);
    }
}
