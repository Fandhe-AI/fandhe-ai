//! `stft`／`istft`（短時間フーリエ変換とその逆変換）のホスト参照カーネルと
//! 検査付きパラメータ型の**単一情報源**（イシュー #2633・親 #2630。実装記録は
//! `docs/autodiff-fft-ops-decision.md` §13）。
//!
//! # 役割と呼び出し元
//!
//! - `fft.rs` の子モジュール（`fft::stft`）。フレームごとの変換は親モジュールの
//!   既存カーネル（[`rfft_host`]／[`rfft_vjp_host`]／[`irfft_host`]／
//!   [`irfft_vjp_host`]）を `[B·T, n_fft]` レイアウトで呼ぶだけで、マージ済み
//!   カーネルの演算順序は変えない。本モジュールが持つのは「フレーム切り出し・
//!   端パディング・窓掛け・重畳加算・窓包絡による除算」の添字計算だけである。
//! - `autodiff::fft_ops`（`BackendOps::fft_stft`／`fft_istft` が `Unsupported`
//!   のときのホストフォールバック）・`autodiff::grad`（VJP）・`backend-cpu` の
//!   `CpuBackendOps::fft_stft`／`fft_istft` がいずれも本モジュールの関数を直接
//!   呼ぶ（座標・総和順序をクレート間で複製せず乖離を構造的に排除する）。
//! - 複素数は末尾次元 2 の `f32` 実テンソル `(re, im)`（`torch.view_as_real` と
//!   同レイアウト）。出力軸順は PyTorch と同じく**周波数軸がフレーム軸より前**
//!   （`[B, N, T, 2]`）。
//!
//! # 数値契約
//!
//! - フレームの窓掛けは `f32` 積（PyTorch と同じ丸め位置）→ 親モジュールの
//!   `f64` 逐次・固定順序の変換。
//! - `istft` の重畳加算・窓包絡・除算と、`stft` VJP の散布加算（反射パディング・
//!   重畳の随伴）は **`f64` アキュムレータに index 昇順（`t` 昇順 → `j` 昇順）で
//!   蓄積し、最後に 1 回だけ `f32` へ downcast** する。`mul_add` は使わない
//!   （matmul 系 FMA 契約には触れない）。同一入力は run-to-run で bit 決定的。
//! - 非有限入力は事前に拒否せず伝播する（`rfft` と同じ）。
//! - 計算量はフレームあたり O(`n_fft`²)（直接 DFT。O(n log n) 化は対象外）。
//!
//! # 境界検査（REQ-8・OWASP A03）
//!
//! 引数・形状の検査は [`StftParams::new`]／[`IstftParams::new`]／
//! [`stft_window`]／[`stft_layout`]／[`istft_layout`] に集約し、確保より前に
//! `checked_add`／`checked_mul` とバイト数（`f64` 幅で保守的に評価）の検査を行って
//! 型付きエラー（[`FftError`]）で拒否する。カーネルはスライス長も再検査し、
//! 添字写像の結果は境界検査付きアクセスで読む（`unsafe`／`get_unchecked` 不使用）。
//! `istft` の NOLA（窓二乗の重畳和）違反はゼロ除算へ進む前に
//! [`istft_check_nola`] が拒否する。

use super::{
    FftError, FftNorm, irfft_host, irfft_layout, irfft_vjp_host, len_error, rfft_host, rfft_layout,
    rfft_vjp_host,
};
use crate::error::ShapeError;

/// `istft` の NOLA 検査しきい値: 出力に使う区間での窓二乗の重畳和（包絡）の
/// 最小絶対値がこの値**未満**なら拒否する。
///
/// 演算の意味論を PyTorch 2.14.0 の実測（境界ペア。`docs/autodiff-fft-ops-
/// decision.md` §13）に合わせて定めた定数であり、REQ-2 の tolerance・
/// ガードレール閾値・テスト許容誤差ではない。
pub const ISTFT_NOLA_MIN_ENVELOPE: f64 = 1e-11;

/// `stft` の `center = true` 時の端パディング種別（`pad_mode` 相当）。
///
/// `replicate`／`circular` は非対応（後から足せるよう `#[non_exhaustive]`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum StftPadMode {
    /// 端点を含めず折り返す反射パディング（PyTorch 既定の `"reflect"`）。
    /// 片側パディング幅 `n_fft/2` は信号長 `L` 未満である必要がある。
    #[default]
    Reflect,
    /// 0 で拡張する定数パディング（`"constant"`）。
    Constant,
}

/// 加算・乗算を `checked_*` で行い、`f64` 幅（保守的）のバイト数が `isize::MAX`
/// 以下であることを検査して要素数を返す。
fn check_alloc(shape: &[usize]) -> Result<usize, FftError> {
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(FftError::Shape(ShapeError::ElementCountOverflow))?;
    let bytes = numel
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(FftError::Shape(ShapeError::ElementCountOverflow))?;
    if bytes > isize::MAX as usize {
        return Err(FftError::Shape(ShapeError::ElementCountOverflow));
    }
    Ok(numel)
}

fn invalid(msg: impl Into<String>) -> FftError {
    FftError::InvalidArgument(msg.into())
}

fn norm_of(normalized: bool) -> FftNorm {
    if normalized {
        FftNorm::Ortho
    } else {
        FftNorm::Backward
    }
}

// ---------------------------------------------------------------------
// 窓
// ---------------------------------------------------------------------

/// 実効窓（長さ `n_fft`）を解決する。
///
/// `win_length` 省略時は `n_fft`。`1 <= win_length <= n_fft` と
/// `window.len() == win_length` を要求し、`win_length < n_fft` のときは左に
/// `(n_fft - win_length)/2` 個のゼロを置いて中央寄せする（PyTorch と同じ）。
/// `window` が `None` のときは矩形窓（`win_length` 個の 1.0 を同様に中央寄せ）。
pub fn stft_window(
    window: Option<&[f32]>,
    win_length: Option<usize>,
    n_fft: usize,
) -> Result<Vec<f32>, FftError> {
    if n_fft == 0 {
        return Err(invalid("stft: n_fft は 1 以上である必要がある"));
    }
    let wl = win_length.unwrap_or(n_fft);
    if wl == 0 || wl > n_fft {
        return Err(invalid(format!(
            "stft: win_length {wl} は 1 以上 n_fft {n_fft} 以下である必要がある"
        )));
    }
    if let Some(w) = window
        && w.len() != wl
    {
        return Err(invalid(format!(
            "stft: 窓の長さ {} が win_length {wl} と一致しない",
            w.len()
        )));
    }
    check_alloc(&[n_fft])?;
    let left = (n_fft - wl) / 2;
    let mut eff = vec![0.0f32; n_fft];
    for i in 0..wl {
        eff[left + i] = window.map_or(1.0, |w| w[i]);
    }
    Ok(eff)
}

// ---------------------------------------------------------------------
// stft
// ---------------------------------------------------------------------

/// `stft` の解決済み引数（[`Self::new`] だけが生成する。フィールド非公開）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StftParams {
    n_fft: usize,
    hop_length: usize,
    center: bool,
    pad_mode: StftPadMode,
    normalized: bool,
    onesided: bool,
}

impl StftParams {
    /// 引数を検査して生成する。`hop_length` 省略時は `n_fft/4`（0 になる
    /// `n_fft < 4` で省略した場合は拒否。PyTorch と同じ）。
    pub fn new(
        n_fft: usize,
        hop_length: Option<usize>,
        center: bool,
        pad_mode: StftPadMode,
        normalized: bool,
        onesided: bool,
    ) -> Result<Self, FftError> {
        if n_fft == 0 {
            return Err(invalid("stft: n_fft は 1 以上である必要がある"));
        }
        let hop_length = hop_length.unwrap_or(n_fft / 4);
        if hop_length == 0 {
            return Err(invalid(
                "stft: hop_length は 1 以上である必要がある（省略時の既定 n_fft/4 が 0 になる場合を含む）",
            ));
        }
        Ok(Self {
            n_fft,
            hop_length,
            center,
            pad_mode,
            normalized,
            onesided,
        })
    }

    /// FFT 長 `n_fft`。
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }
    /// フレーム間隔。
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }
    /// 両端を `n_fft/2` 拡張するか。
    pub fn center(&self) -> bool {
        self.center
    }
    /// 端パディング種別。
    pub fn pad_mode(&self) -> StftPadMode {
        self.pad_mode
    }
    /// `1/√n_fft` 正規化か。
    pub fn normalized(&self) -> bool {
        self.normalized
    }
    /// 片側スペクトル（`n_fft/2+1` bin）か。
    pub fn onesided(&self) -> bool {
        self.onesided
    }
}

/// `stft` の解決済みレイアウト（[`stft_layout`] だけが生成する。フィールド非公開）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StftLayout {
    params: StftParams,
    batch: usize,
    len: usize,
    pad: usize,
    frames: usize,
    out_bins: usize,
    out_shape: Vec<usize>,
}

impl StftLayout {
    /// 解決済み引数。
    pub fn params(&self) -> &StftParams {
        &self.params
    }
    /// 出力形状 `[B?, N, T, 2]`。VJP では上流勾配の形状。
    pub fn out_shape(&self) -> &[usize] {
        &self.out_shape
    }
    /// フレーム数 `T`。
    pub fn frames(&self) -> usize {
        self.frames
    }
    /// 出力の周波数 bin 数 `N`。
    pub fn out_bins(&self) -> usize {
        self.out_bins
    }

    /// パディング後座標 `pos` に対応する元信号の添字（範囲外の定数パディングは
    /// `None`）。反射は `pad < len` の検査済みのため 1 回の折り返しで範囲内に収まる。
    fn src_index(&self, pos: usize) -> Option<usize> {
        let orig = pos as isize - self.pad as isize;
        let len = self.len as isize;
        let mapped = if orig < 0 || orig >= len {
            match self.params.pad_mode {
                StftPadMode::Reflect => {
                    if orig < 0 {
                        -orig
                    } else {
                        2 * (len - 1) - orig
                    }
                }
                StftPadMode::Constant => return None,
            }
        } else {
            orig
        };
        if (0..len).contains(&mapped) {
            Some(mapped as usize)
        } else {
            None
        }
    }
}

/// `stft`（実 `[L]`／`[B, L]` → `[N, T, 2]`／`[B, N, T, 2]`）の形状・確保サイズを
/// 検査してレイアウトを解決する（確保・実体化より前）。
pub fn stft_layout(in_shape: &[usize], params: &StftParams) -> Result<StftLayout, FftError> {
    let rank = in_shape.len();
    if !(1..=2).contains(&rank) {
        return Err(FftError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: rank,
        }));
    }
    if in_shape.contains(&0) {
        return Err(invalid("stft: 入力に 0 要素の次元は許容されない"));
    }
    let (batch, len) = if rank == 2 {
        (in_shape[0], in_shape[1])
    } else {
        (1, in_shape[0])
    };
    let n = params.n_fft;
    let pad = if params.center { n / 2 } else { 0 };
    if params.center && params.pad_mode == StftPadMode::Reflect && pad >= len {
        return Err(invalid(format!(
            "stft: 反射パディング幅 {pad} は信号長 {len} 未満である必要がある"
        )));
    }
    let padded = pad
        .checked_mul(2)
        .and_then(|p2| len.checked_add(p2))
        .ok_or(FftError::Shape(ShapeError::ElementCountOverflow))?;
    if n > padded {
        return Err(invalid(format!(
            "stft: n_fft {n} がパディング後の信号長 {padded} を超える"
        )));
    }
    let frames = 1 + (padded - n) / params.hop_length;
    let bins = n / 2 + 1;
    let out_bins = if params.onesided { bins } else { n };
    let bt = batch
        .checked_mul(frames)
        .ok_or(FftError::Shape(ShapeError::ElementCountOverflow))?;
    check_alloc(&[bt, n])?;
    check_alloc(&[bt, bins, 2])?;
    check_alloc(in_shape)?;
    let mut out_shape = Vec::with_capacity(4);
    if rank == 2 {
        out_shape.push(batch);
    }
    out_shape.extend([out_bins, frames, 2]);
    check_alloc(&out_shape)?;
    Ok(StftLayout {
        params: params.clone(),
        batch,
        len,
        pad,
        frames,
        out_bins,
        out_shape,
    })
}

/// `stft` forward（出力は `layout.out_shape()` の連続配置）。
///
/// `window` は [`stft_window`] で解決済みの長さ `n_fft` の実効窓。
pub fn stft_host(x: &[f32], window: &[f32], layout: &StftLayout) -> Result<Vec<f32>, FftError> {
    let p = &layout.params;
    let (n, hop, t_n, b_n) = (p.n_fft, p.hop_length, layout.frames, layout.batch);
    let bins = n / 2 + 1;
    let in_numel = b_n * layout.len;
    if x.len() != in_numel {
        return Err(len_error("stft_host", in_numel, x.len()));
    }
    if window.len() != n {
        return Err(len_error("stft_host(window)", n, window.len()));
    }
    let bt = b_n * t_n;
    let mut frames = vec![0.0f32; bt * n];
    for b in 0..b_n {
        for t in 0..t_n {
            let row = (b * t_n + t) * n;
            for j in 0..n {
                // 定数パディングの範囲外サンプルは 0 として取得し、全 j に同じ窓掛けを
                // 適用する（window[j] が NaN/inf のとき 0*window[j]=NaN となり、
                // 非有限値の伝播契約を範囲内サンプルと揃える）。
                let v = layout
                    .src_index(t * hop + j)
                    .map_or(0.0f32, |s| x[b * layout.len + s]);
                frames[row + j] = v * window[j];
            }
        }
    }
    let rl = rfft_layout(&[bt, n], Some(n), Some(1))?;
    let spec = rfft_host(&frames, &rl, norm_of(p.normalized))?;
    let out_bins = layout.out_bins;
    let mut out = vec![0.0f32; b_n * out_bins * t_n * 2];
    for b in 0..b_n {
        for t in 0..t_n {
            for k in 0..bins {
                let src = ((b * t_n + t) * bins + k) * 2;
                let dst = ((b * out_bins + k) * t_n + t) * 2;
                out[dst] = spec[src];
                out[dst + 1] = spec[src + 1];
            }
            // 全 bin 出力は実入力の共役対称 `Y[n-k] = conj(Y[k])` で埋める。
            for k in bins..out_bins {
                let src = ((b * t_n + t) * bins + (n - k)) * 2;
                let dst = ((b * out_bins + k) * t_n + t) * 2;
                out[dst] = spec[src];
                out[dst + 1] = -spec[src + 1];
            }
        }
    }
    Ok(out)
}

/// `stft` の VJP。`g` は `layout.out_shape()` の上流勾配、戻り値は入力形状の勾配。
///
/// forward の随伴を逆順にたどる: 共役対称ミラーの折り畳み（実部は加算・虚部は
/// 減算）→ `rfft_vjp_host` → 窓掛け → 反射／定数パディング写像先への散布加算
/// （`f64` アキュムレータ・index 昇順）。
pub fn stft_vjp_host(g: &[f32], window: &[f32], layout: &StftLayout) -> Result<Vec<f32>, FftError> {
    let p = &layout.params;
    let (n, hop, t_n, b_n) = (p.n_fft, p.hop_length, layout.frames, layout.batch);
    let bins = n / 2 + 1;
    let out_bins = layout.out_bins;
    let g_numel = b_n * out_bins * t_n * 2;
    if g.len() != g_numel {
        return Err(len_error("stft_vjp_host", g_numel, g.len()));
    }
    if window.len() != n {
        return Err(len_error("stft_vjp_host(window)", n, window.len()));
    }
    let bt = b_n * t_n;
    let mut spec_g = vec![0.0f32; bt * bins * 2];
    for b in 0..b_n {
        for t in 0..t_n {
            for k in 0..bins {
                let dst = ((b * t_n + t) * bins + k) * 2;
                let src = ((b * out_bins + k) * t_n + t) * 2;
                spec_g[dst] = g[src];
                spec_g[dst + 1] = g[src + 1];
            }
            for k in bins..out_bins {
                let dst = ((b * t_n + t) * bins + (n - k)) * 2;
                let src = ((b * out_bins + k) * t_n + t) * 2;
                spec_g[dst] = (f64::from(spec_g[dst]) + f64::from(g[src])) as f32;
                spec_g[dst + 1] = (f64::from(spec_g[dst + 1]) - f64::from(g[src + 1])) as f32;
            }
        }
    }
    let rl = rfft_layout(&[bt, n], Some(n), Some(1))?;
    let dframes = rfft_vjp_host(&spec_g, &rl, norm_of(p.normalized))?;
    let mut acc = vec![0.0f64; b_n * layout.len];
    for b in 0..b_n {
        for t in 0..t_n {
            let row = (b * t_n + t) * n;
            for j in 0..n {
                if let Some(s) = layout.src_index(t * hop + j) {
                    acc[b * layout.len + s] += f64::from(dframes[row + j]) * f64::from(window[j]);
                }
            }
        }
    }
    Ok(acc.into_iter().map(|v| v as f32).collect())
}

// ---------------------------------------------------------------------
// istft
// ---------------------------------------------------------------------

/// `istft` の解決済み引数（[`Self::new`] だけが生成する。フィールド非公開）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IstftParams {
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    center: bool,
    normalized: bool,
    onesided: Option<bool>,
    length: Option<usize>,
}

impl IstftParams {
    /// 引数を検査して生成する。`hop_length` 省略時は `n_fft/4`・`win_length`
    /// 省略時は `n_fft`。`1 <= hop_length <= win_length <= n_fft` を要求する。
    /// `onesided = None` は入力 bin 数から推定（`bin 数 != n_fft` なら片側。
    /// [`istft_layout`] で解決）。`length = Some(0)` は拒否する。
    pub fn new(
        n_fft: usize,
        hop_length: Option<usize>,
        win_length: Option<usize>,
        center: bool,
        normalized: bool,
        onesided: Option<bool>,
        length: Option<usize>,
    ) -> Result<Self, FftError> {
        if n_fft == 0 {
            return Err(invalid("istft: n_fft は 1 以上である必要がある"));
        }
        let win_length = win_length.unwrap_or(n_fft);
        if win_length == 0 || win_length > n_fft {
            return Err(invalid(format!(
                "istft: win_length {win_length} は 1 以上 n_fft {n_fft} 以下である必要がある"
            )));
        }
        let hop_length = hop_length.unwrap_or(n_fft / 4);
        if hop_length == 0 || hop_length > win_length {
            return Err(invalid(format!(
                "istft: hop_length {hop_length} は 1 以上 win_length {win_length} 以下である必要がある"
            )));
        }
        if length == Some(0) {
            return Err(invalid("istft: length は 1 以上である必要がある"));
        }
        Ok(Self {
            n_fft,
            hop_length,
            win_length,
            center,
            normalized,
            onesided,
            length,
        })
    }

    /// FFT 長 `n_fft`。
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }
    /// フレーム間隔。
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }
    /// 解決済みの窓長（`window` の長さと一致する必要がある）。
    pub fn win_length(&self) -> usize {
        self.win_length
    }
    /// 両端の `n_fft/2` を切り落とすか。
    pub fn center(&self) -> bool {
        self.center
    }
    /// `1/√n_fft` 正規化か。
    pub fn normalized(&self) -> bool {
        self.normalized
    }
    /// 指定された `onesided`（`None` は入力 bin 数から推定）。
    pub fn onesided(&self) -> Option<bool> {
        self.onesided
    }
    /// 出力長の指定。
    pub fn length(&self) -> Option<usize> {
        self.length
    }
}

/// `istft` の解決済みレイアウト（[`istft_layout`] だけが生成する。フィールド非公開）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IstftLayout {
    params: IstftParams,
    batch: usize,
    in_bins: usize,
    frames: usize,
    expected: usize,
    start: usize,
    valid: usize,
    out_len: usize,
    out_shape: Vec<usize>,
}

impl IstftLayout {
    /// 解決済み引数。
    pub fn params(&self) -> &IstftParams {
        &self.params
    }
    /// 出力形状 `[L_out]`／`[B, L_out]`。VJP では上流勾配の形状。
    pub fn out_shape(&self) -> &[usize] {
        &self.out_shape
    }
    /// 出力長 `L_out`。
    pub fn out_len(&self) -> usize {
        self.out_len
    }
}

/// `istft`（`[N, T, 2]`／`[B, N, T, 2]` → 実 `[L_out]`／`[B, L_out]`）の形状・
/// 確保サイズを検査してレイアウトを解決する（確保・実体化より前）。
pub fn istft_layout(in_shape: &[usize], params: &IstftParams) -> Result<IstftLayout, FftError> {
    let rank = in_shape.len();
    if !(3..=4).contains(&rank) {
        return Err(FftError::Shape(ShapeError::RankMismatch {
            expected: 4,
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
    if in_shape.contains(&0) {
        return Err(invalid("istft: 入力に 0 要素の次元は許容されない"));
    }
    let (batch, in_bins, frames) = if rank == 4 {
        (in_shape[0], in_shape[1], in_shape[2])
    } else {
        (1, in_shape[0], in_shape[1])
    };
    let n = params.n_fft;
    let onesided = params.onesided.unwrap_or(in_bins != n);
    let want_bins = if onesided { n / 2 + 1 } else { n };
    if in_bins != want_bins {
        return Err(invalid(format!(
            "istft: 入力 bin 数 {in_bins} が onesided={onesided} の期待値 {want_bins} と一致しない"
        )));
    }
    let overflow = || FftError::Shape(ShapeError::ElementCountOverflow);
    let expected = (frames - 1)
        .checked_mul(params.hop_length)
        .and_then(|v| v.checked_add(n))
        .ok_or_else(overflow)?;
    let start = if params.center { n / 2 } else { 0 };
    let end = match params.length {
        Some(l) => start.checked_add(l).ok_or_else(overflow)?,
        None if params.center => expected - n / 2,
        None => expected,
    };
    if end <= start {
        return Err(invalid("istft: 出力長が 0 以下になる引数の組合せ"));
    }
    let out_len = end - start;
    let valid = end.min(expected) - start;
    let bt = batch.checked_mul(frames).ok_or_else(overflow)?;
    let nb = n / 2 + 1;
    check_alloc(&[bt, nb, 2])?;
    check_alloc(&[bt, n])?;
    check_alloc(&[batch, expected])?;
    check_alloc(in_shape)?;
    let mut out_shape = Vec::with_capacity(2);
    if rank == 4 {
        out_shape.push(batch);
    }
    out_shape.push(out_len);
    check_alloc(&out_shape)?;
    Ok(IstftLayout {
        params: IstftParams {
            onesided: Some(onesided),
            ..params.clone()
        },
        batch,
        in_bins,
        frames,
        expected,
        start,
        valid,
        out_len,
        out_shape,
    })
}

/// 窓二乗の重畳和（包絡）。`f64` で `t` 昇順 → `j` 昇順に蓄積する。
fn istft_envelope(window: &[f32], layout: &IstftLayout) -> Vec<f64> {
    let (n, hop) = (layout.params.n_fft, layout.params.hop_length);
    let mut env = vec![0.0f64; layout.expected];
    for t in 0..layout.frames {
        for (j, &w) in window.iter().enumerate().take(n) {
            env[t * hop + j] += f64::from(w) * f64::from(w);
        }
    }
    env
}

/// NOLA（窓二乗の重畳和が 0 にならないこと）を検査する。
///
/// 出力に使う区間 `[start, min(end, expected))` の包絡の最小絶対値が
/// [`ISTFT_NOLA_MIN_ENVELOPE`] 未満なら `InvalidArgument`（ゼロ除算へ進ませない）。
/// カーネル内と `autodiff::fft_ops::istft`（バックエンド呼び出し前）の両方から
/// 呼ばれ、バックエンド実装が検査を迂回できないようにする。
pub fn istft_check_nola(window: &[f32], layout: &IstftLayout) -> Result<(), FftError> {
    if window.len() != layout.params.n_fft {
        return Err(len_error(
            "istft_check_nola(window)",
            layout.params.n_fft,
            window.len(),
        ));
    }
    let env = istft_envelope(window, layout);
    let lo = layout.start;
    let hi = lo + layout.valid;
    let min = env[lo..hi]
        .iter()
        .fold(f64::INFINITY, |m, &v| m.min(v.abs()));
    if min < ISTFT_NOLA_MIN_ENVELOPE {
        return Err(invalid(format!(
            "istft: 窓包絡の最小値 {min:e} が {ISTFT_NOLA_MIN_ENVELOPE:e} 未満（NOLA 違反）"
        )));
    }
    Ok(())
}

/// 入力 `[B, N_in, T, 2]` から各フレームの先頭 `n/2+1` bin だけを
/// `[B·T, n/2+1, 2]` へ並べ替える（c2r が読む範囲。`onesided = false` の上位 bin は
/// 読まない）。
fn gather_istft_spectra(x: &[f32], layout: &IstftLayout) -> Vec<f32> {
    let (b_n, t_n, in_bins) = (layout.batch, layout.frames, layout.in_bins);
    let nb = layout.params.n_fft / 2 + 1;
    let mut spec = vec![0.0f32; b_n * t_n * nb * 2];
    for b in 0..b_n {
        for t in 0..t_n {
            for k in 0..nb {
                let src = ((b * in_bins + k) * t_n + t) * 2;
                let dst = ((b * t_n + t) * nb + k) * 2;
                spec[dst] = x[src];
                spec[dst + 1] = x[src + 1];
            }
        }
    }
    spec
}

/// `istft` forward（出力は `layout.out_shape()` の連続配置）。
///
/// 各フレームを c2r（`onesided = false` でも先頭 `n_fft/2+1` bin のみを読む。
/// PyTorch 2.14.0 の実測）→ `f32` の窓掛け → `f64` で重畳加算 → NOLA 検査 →
/// 包絡で除算。`length` が期待長を超える分は 0 詰め。
pub fn istft_host(x: &[f32], window: &[f32], layout: &IstftLayout) -> Result<Vec<f32>, FftError> {
    let p = &layout.params;
    let (n, hop, t_n, b_n) = (p.n_fft, p.hop_length, layout.frames, layout.batch);
    let in_numel = b_n * layout.in_bins * t_n * 2;
    if x.len() != in_numel {
        return Err(len_error("istft_host", in_numel, x.len()));
    }
    if window.len() != n {
        return Err(len_error("istft_host(window)", n, window.len()));
    }
    let env = istft_envelope(window, layout);
    istft_check_nola(window, layout)?;
    let nb = n / 2 + 1;
    let bt = b_n * t_n;
    let spec = gather_istft_spectra(x, layout);
    let rl = irfft_layout(&[bt, nb, 2], Some(n), Some(1))?;
    let frames = irfft_host(&spec, &rl, norm_of(p.normalized))?;
    let mut y = vec![0.0f64; b_n * layout.expected];
    for b in 0..b_n {
        for t in 0..t_n {
            let row = (b * t_n + t) * n;
            for j in 0..n {
                y[b * layout.expected + t * hop + j] += f64::from(frames[row + j] * window[j]);
            }
        }
    }
    let mut out = vec![0.0f32; b_n * layout.out_len];
    for b in 0..b_n {
        for i in 0..layout.valid {
            let pos = layout.start + i;
            out[b * layout.out_len + i] = (y[b * layout.expected + pos] / env[pos]) as f32;
        }
    }
    Ok(out)
}

/// `istft` の VJP。`g` は `layout.out_shape()` の上流勾配、戻り値は入力形状
/// `[B?, N_in, T, 2]` の勾配。
///
/// forward の随伴: 包絡除算 → フレーム位置への読み出し×窓 → `irfft_vjp_host` →
/// 入力 bin への並べ替え。forward が読まなかった bin（`onesided = false` の
/// 上位 bin）と期待長を超える 0 詰め部分の勾配は 0。
pub fn istft_vjp_host(
    g: &[f32],
    window: &[f32],
    layout: &IstftLayout,
) -> Result<Vec<f32>, FftError> {
    let p = &layout.params;
    let (n, hop, t_n, b_n) = (p.n_fft, p.hop_length, layout.frames, layout.batch);
    let g_numel = b_n * layout.out_len;
    if g.len() != g_numel {
        return Err(len_error("istft_vjp_host", g_numel, g.len()));
    }
    if window.len() != n {
        return Err(len_error("istft_vjp_host(window)", n, window.len()));
    }
    istft_check_nola(window, layout)?;
    let env = istft_envelope(window, layout);
    let nb = n / 2 + 1;
    let bt = b_n * t_n;
    let mut dframes = vec![0.0f32; bt * n];
    for b in 0..b_n {
        for t in 0..t_n {
            let row = (b * t_n + t) * n;
            for j in 0..n {
                let pos = t * hop + j;
                if pos >= layout.start && pos - layout.start < layout.valid {
                    let gv = f64::from(g[b * layout.out_len + (pos - layout.start)]);
                    dframes[row + j] = (gv / env[pos] * f64::from(window[j])) as f32;
                }
            }
        }
    }
    let rl = irfft_layout(&[bt, nb, 2], Some(n), Some(1))?;
    let dspec = irfft_vjp_host(&dframes, &rl, norm_of(p.normalized))?;
    let mut out = vec![0.0f32; b_n * layout.in_bins * t_n * 2];
    for b in 0..b_n {
        for t in 0..t_n {
            for k in 0..nb {
                let src = ((b * t_n + t) * nb + k) * 2;
                let dst = ((b * layout.in_bins + k) * t_n + t) * 2;
                out[dst] = dspec[src];
                out[dst + 1] = dspec[src + 1];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn inner_product(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    }

    fn hann(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()) as f32)
            .collect()
    }

    fn sp(
        n: usize,
        hop: Option<usize>,
        center: bool,
        pad: StftPadMode,
        norm: bool,
        one: bool,
    ) -> StftParams {
        StftParams::new(n, hop, center, pad, norm, one).expect("params")
    }

    #[test]
    fn stft_params_reject_invalid_arguments() {
        assert!(StftParams::new(0, None, true, StftPadMode::Reflect, false, true).is_err());
        assert!(StftParams::new(3, None, true, StftPadMode::Reflect, false, true).is_err());
        assert!(StftParams::new(8, Some(0), true, StftPadMode::Reflect, false, true).is_err());
        assert!(StftParams::new(4, None, true, StftPadMode::Reflect, false, true).is_ok());
    }

    #[test]
    fn stft_window_resolution() {
        assert_eq!(stft_window(None, None, 4).unwrap(), vec![1.0; 4]);
        assert_eq!(
            stft_window(None, Some(2), 5).unwrap(),
            vec![0.0, 1.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            stft_window(Some(&[2.0, 3.0]), Some(2), 6).unwrap(),
            vec![0.0, 0.0, 2.0, 3.0, 0.0, 0.0]
        );
        assert!(stft_window(Some(&[1.0; 3]), None, 4).is_err());
        assert!(stft_window(None, Some(0), 4).is_err());
        assert!(stft_window(None, Some(5), 4).is_err());
        assert!(stft_window(None, None, 0).is_err());
    }

    #[test]
    fn stft_layout_checks() {
        let p = sp(8, Some(2), true, StftPadMode::Reflect, false, true);
        let l = stft_layout(&[32], &p).unwrap();
        assert_eq!(l.out_shape(), &[5, 17, 2]);
        let l = stft_layout(&[3, 32], &p).unwrap();
        assert_eq!(l.out_shape(), &[3, 5, 17, 2]);
        // 反射パディング幅は信号長未満。
        assert!(stft_layout(&[4], &p).is_err());
        assert!(stft_layout(&[5], &p).is_ok());
        // 定数パディングは短信号も可。
        let pc = sp(8, Some(2), true, StftPadMode::Constant, false, true);
        assert_eq!(stft_layout(&[3], &pc).unwrap().out_shape(), &[5, 2, 2]);
        // n_fft > パディング後長・rank・空次元。
        let pn = sp(8, Some(2), false, StftPadMode::Reflect, false, true);
        assert!(stft_layout(&[7], &pn).is_err());
        assert!(stft_layout(&[], &p).is_err());
        assert!(stft_layout(&[1, 2, 32], &p).is_err());
        assert!(stft_layout(&[0], &p).is_err());
        assert!(stft_layout(&[0, 32], &p).is_err());
        // オーバーフロー・巨大確保は確保前に拒否。
        assert!(stft_layout(&[usize::MAX], &pc).is_err());
        assert!(stft_layout(&[usize::MAX / 2, 4], &pc).is_err());
        let ph = sp(1 << 20, Some(1), false, StftPadMode::Reflect, false, true);
        assert!(stft_layout(&[1 << 40], &ph).is_err());
    }

    #[test]
    fn stft_constant_pad_propagates_non_finite_window() {
        // 範囲外（定数パディング）サンプルでも 0*NaN=NaN が窓掛けで伝播する。
        let pc = sp(4, Some(1), true, StftPadMode::Constant, false, true);
        // L=1 のとき窓タップ 0 は全フレームで padded 座標 0・1（範囲外）にだけ当たる
        // ため、NaN の出所は定数パディング由来の 0*NaN に限られる。
        let layout = stft_layout(&[1], &pc).unwrap();
        assert_eq!(layout.src_index(0), None);
        let x = vec![1.0f32; 1];
        let mut w = vec![1.0f32; 4];
        w[0] = f32::NAN;
        let out = stft_host(&x, &w, &layout).unwrap();
        assert!(out.iter().any(|v| v.is_nan()));
    }

    #[test]
    fn src_index_reflect_and_constant() {
        let p = sp(4, Some(1), true, StftPadMode::Reflect, false, true);
        let l = stft_layout(&[5], &p).unwrap();
        // pad = 2。padded 座標 0,1 → 元 2,1。末尾 7,8 → 元 3,2。
        let got: Vec<_> = (0..9).map(|i| l.src_index(i)).collect();
        assert_eq!(
            got,
            vec![
                Some(2),
                Some(1),
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(3),
                Some(2)
            ]
        );
        let pc = sp(4, Some(1), true, StftPadMode::Constant, false, true);
        let l = stft_layout(&[5], &pc).unwrap();
        assert_eq!(l.src_index(1), None);
        assert_eq!(l.src_index(2), Some(0));
        assert_eq!(l.src_index(7), None);
    }

    #[test]
    fn stft_matches_blockwise_rfft_without_center() {
        // hop = n_fft・矩形窓・center=false はブロックごとの rfft と一致する。
        let x = rand_vec(12, 7);
        let p = sp(4, Some(4), false, StftPadMode::Reflect, false, true);
        let l = stft_layout(&[12], &p).unwrap();
        let y = stft_host(&x, &[1.0; 4], &l).unwrap();
        for t in 0..3 {
            let rl = rfft_layout(&[4], None, None).unwrap();
            let want = rfft_host(&x[t * 4..t * 4 + 4], &rl, FftNorm::Backward).unwrap();
            for k in 0..3 {
                assert_eq!(y[(k * 3 + t) * 2], want[k * 2], "re t={t} k={k}");
                assert_eq!(y[(k * 3 + t) * 2 + 1], want[k * 2 + 1], "im t={t} k={k}");
            }
        }
    }

    #[test]
    fn stft_dc_and_nyquist_imag_are_positive_zero() {
        let x = rand_vec(32, 3);
        let p = sp(8, None, true, StftPadMode::Reflect, false, true);
        let l = stft_layout(&[32], &p).unwrap();
        let y = stft_host(&x, &hann(8), &l).unwrap();
        let t_n = l.frames();
        for t in 0..t_n {
            assert_eq!(y[(t) * 2 + 1].to_bits(), 0);
            assert_eq!(y[((4 * t_n) + t) * 2 + 1].to_bits(), 0);
        }
    }

    #[test]
    fn stft_two_sided_is_conjugate_symmetric() {
        let x = rand_vec(20, 11);
        for n in [6usize, 7] {
            let p = sp(n, Some(2), true, StftPadMode::Reflect, false, false);
            let l = stft_layout(&[20], &p).unwrap();
            let y = stft_host(&x, &vec![1.0; n], &l).unwrap();
            let t_n = l.frames();
            for t in 0..t_n {
                for k in 1..n {
                    let a = ((k * t_n) + t) * 2;
                    let b = (((n - k) * t_n) + t) * 2;
                    assert_eq!(y[a], y[b]);
                    assert_eq!(y[a + 1], -y[b + 1]);
                }
            }
        }
    }

    fn adjoint_stft(shape: &[usize], p: &StftParams, wl: Option<usize>) {
        let win = stft_window(None, wl, p.n_fft()).unwrap();
        let win: Vec<f32> = win
            .iter()
            .zip(hann(p.n_fft()))
            .map(|(a, b)| a * b)
            .collect();
        let l = stft_layout(shape, p).unwrap();
        let numel: usize = shape.iter().product();
        let out_numel: usize = l.out_shape().iter().product();
        let x = rand_vec(numel, 21);
        let g = rand_vec(out_numel, 22);
        let ax = stft_host(&x, &win, &l).unwrap();
        let atg = stft_vjp_host(&g, &win, &l).unwrap();
        let lhs = inner_product(&ax, &g);
        let rhs = inner_product(&x, &atg);
        assert!(
            (lhs - rhs).abs() <= 1e-4 * lhs.abs().max(1.0),
            "adjoint {p:?} {shape:?}: {lhs} vs {rhs}"
        );
    }

    #[test]
    fn stft_vjp_satisfies_adjoint_identity_over_parameter_grid() {
        for n in [8usize, 7] {
            for center in [true, false] {
                for pad in [StftPadMode::Reflect, StftPadMode::Constant] {
                    for norm in [false, true] {
                        for one in [true, false] {
                            for hop in [2usize, 3, n] {
                                let p = sp(n, Some(hop), center, pad, norm, one);
                                adjoint_stft(&[40], &p, None);
                                adjoint_stft(&[2, 40], &p, Some(n - 2));
                            }
                        }
                    }
                }
            }
        }
    }

    fn ip(
        n: usize,
        hop: Option<usize>,
        wl: Option<usize>,
        center: bool,
        norm: bool,
        one: Option<bool>,
        len: Option<usize>,
    ) -> IstftParams {
        IstftParams::new(n, hop, wl, center, norm, one, len).expect("params")
    }

    #[test]
    fn istft_params_reject_invalid_arguments() {
        assert!(IstftParams::new(0, None, None, true, false, None, None).is_err());
        assert!(IstftParams::new(8, Some(9), None, true, false, None, None).is_err());
        assert!(IstftParams::new(8, Some(0), None, true, false, None, None).is_err());
        assert!(IstftParams::new(8, Some(5), Some(4), true, false, None, None).is_err());
        assert!(IstftParams::new(8, None, Some(9), true, false, None, None).is_err());
        assert!(IstftParams::new(8, None, None, true, false, None, Some(0)).is_err());
        assert!(IstftParams::new(3, None, None, true, false, None, None).is_err());
    }

    #[test]
    fn istft_layout_checks() {
        let p = ip(8, Some(2), None, true, false, None, None);
        let l = istft_layout(&[5, 17, 2], &p).unwrap();
        assert_eq!(l.out_shape(), &[32]);
        assert_eq!(l.params().onesided(), Some(true));
        let l = istft_layout(&[3, 5, 17, 2], &p).unwrap();
        assert_eq!(l.out_shape(), &[3, 32]);
        // 両側（bin 数 == n_fft）。
        let l = istft_layout(&[8, 17, 2], &p).unwrap();
        assert_eq!(l.params().onesided(), Some(false));
        // bin 数不一致・末尾次元・rank・T=0。
        assert!(istft_layout(&[3, 17, 2], &p).is_err());
        assert!(istft_layout(&[5, 17, 3], &p).is_err());
        assert!(istft_layout(&[5, 17], &p).is_err());
        assert!(istft_layout(&[1, 1, 5, 17, 2], &p).is_err());
        assert!(istft_layout(&[5, 0, 2], &p).is_err());
        // onesided 明示との不一致。
        let po = ip(8, Some(2), None, true, false, Some(false), None);
        assert!(istft_layout(&[5, 17, 2], &po).is_err());
        // length: 切り詰め・超過。
        let pl = ip(8, Some(2), None, true, false, None, Some(100));
        assert_eq!(istft_layout(&[5, 17, 2], &pl).unwrap().out_len(), 100);
        // 出力長 0（center かつ T=1・n_fft 偶数）。
        assert!(istft_layout(&[5, 1, 2], &p).is_err());
        // オーバーフロー。
        assert!(istft_layout(&[5, usize::MAX / 2, 2], &p).is_err());
        let pl = ip(8, Some(2), None, true, false, None, Some(usize::MAX));
        assert!(istft_layout(&[5, 17, 2], &pl).is_err());
    }

    #[test]
    fn istft_nola_rejects_zero_envelope() {
        // Hann・center=false は端の包絡が 0。
        let p = ip(8, Some(2), None, false, false, None, None);
        let l = istft_layout(&[5, 17, 2], &p).unwrap();
        assert!(istft_check_nola(&hann(8), &l).is_err());
        let x = vec![0.0f32; 5 * 17 * 2];
        assert!(istft_host(&x, &hann(8), &l).is_err());
        // center=true なら端が切り落とされ通る。
        let pc = ip(8, Some(2), None, true, false, None, None);
        let lc = istft_layout(&[5, 17, 2], &pc).unwrap();
        assert!(istft_check_nola(&hann(8), &lc).is_ok());
    }

    #[test]
    fn istft_inverts_stft_for_hann() {
        let x = rand_vec(64, 5);
        for hop in [2usize, 4] {
            let sp_ = sp(8, Some(hop), true, StftPadMode::Reflect, false, true);
            let win = hann(8);
            let l = stft_layout(&[64], &sp_).unwrap();
            let s = stft_host(&x, &win, &l).unwrap();
            let ip_ = ip(8, Some(hop), None, true, false, None, Some(64));
            let il = istft_layout(l.out_shape(), &ip_).unwrap();
            let y = istft_host(&s, &win, &il).unwrap();
            for (a, b) in x.iter().zip(&y) {
                assert!((a - b).abs() < 1e-4, "hop={hop}: {a} vs {b}");
            }
        }
    }

    fn adjoint_istft(shape: &[usize], p: &IstftParams, win: &[f32]) {
        let l = istft_layout(shape, p).unwrap();
        let numel: usize = shape.iter().product();
        let out_numel: usize = l.out_shape().iter().product();
        let x = rand_vec(numel, 31);
        let g = rand_vec(out_numel, 32);
        let ax = istft_host(&x, win, &l).unwrap();
        let atg = istft_vjp_host(&g, win, &l).unwrap();
        let lhs = inner_product(&ax, &g);
        let rhs = inner_product(&x, &atg);
        assert!(
            (lhs - rhs).abs() <= 1e-4 * lhs.abs().max(1.0),
            "adjoint {p:?} {shape:?}: {lhs} vs {rhs}"
        );
    }

    #[test]
    fn istft_vjp_satisfies_adjoint_identity_over_parameter_grid() {
        for n in [8usize, 7] {
            for center in [true, false] {
                for norm in [false, true] {
                    for hop in [1usize, 2, 3] {
                        for len in [None, Some(10), Some(60)] {
                            let t = 9usize;
                            // 窓は矩形（NOLA は hop <= n で成立。center=false の端も包絡 >= 1）。
                            let win = stft_window(None, Some(n - 1), n).unwrap();
                            for one in [Some(true), Some(false)] {
                                let bins = if one == Some(true) { n / 2 + 1 } else { n };
                                let p = ip(n, Some(hop), Some(n - 1), center, norm, one, len);
                                // 中央寄せ窓は端のサンプルが 0 のため NOLA は hop と範囲次第。
                                let probe = istft_layout(&[bins, t, 2], &p).unwrap();
                                if istft_check_nola(&win, &probe).is_err() {
                                    continue;
                                }
                                adjoint_istft(&[bins, t, 2], &p, &win);
                                adjoint_istft(&[2, bins, t, 2], &p, &win);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn istft_two_sided_ignores_upper_bins_and_gives_zero_grad() {
        let n = 8;
        let p = ip(n, Some(4), None, false, false, Some(false), None);
        let l = istft_layout(&[n, 3, 2], &p).unwrap();
        let x = rand_vec(n * 3 * 2, 9);
        let win = vec![1.0f32; n];
        let y = istft_host(&x, &win, &l).unwrap();
        let mut x2 = x.clone();
        for k in 5..n {
            for t in 0..3 {
                x2[(k * 3 + t) * 2] += 7.0;
                x2[(k * 3 + t) * 2 + 1] -= 3.0;
            }
        }
        assert_eq!(y, istft_host(&x2, &win, &l).unwrap());
        let g = rand_vec(y.len(), 10);
        let dx = istft_vjp_host(&g, &win, &l).unwrap();
        for k in 5..n {
            for t in 0..3 {
                assert_eq!(dx[(k * 3 + t) * 2], 0.0);
                assert_eq!(dx[(k * 3 + t) * 2 + 1], 0.0);
            }
        }
    }

    #[test]
    fn kernels_are_deterministic_and_check_slice_lengths() {
        let x = rand_vec(40, 1);
        let p = sp(8, None, true, StftPadMode::Reflect, true, true);
        let l = stft_layout(&[40], &p).unwrap();
        let w = hann(8);
        let a = stft_host(&x, &w, &l).unwrap();
        let b = stft_host(&x, &w, &l).unwrap();
        assert_eq!(a, b);
        assert!(stft_host(&x[..39], &w, &l).is_err());
        assert!(stft_host(&x, &w[..7], &l).is_err());
        assert!(stft_vjp_host(&a[..1], &w, &l).is_err());
    }
}
