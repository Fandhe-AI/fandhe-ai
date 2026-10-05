//! `rfft`・`irfft`・`fft`・`ifft`・`stft`・`istft` の自由関数（イシュー #2631・
//! #2632・#2633・親 #2630「FFT」。ルート #2499 Phase 4）。
//!
//! **facade 非公開（保留）**: 公開形（`Var::rfft`／`Var::irfft`／`Var::fft`／
//! `Var::ifft`／`Var::stft`／`Var::istft` の委譲メソッドと `FftNorm`・
//! `StftOptions`・`IstftOptions`・`StftPadMode` の再エクスポート）は未承認で、
//! 承認依頼は #2677
//! （公開自体は承認後の #2678）。本モジュールは内部クレート限定の入口で、
//! `Var` に inherent メソッドを足さない。保留は `crates/facade/src/lib.rs`
//! の `FftOpsHoldDoctestGuard` と `crates/facade/tests/api_surface.rs` の
//! 否定ガードが機械的に固定する（`docs/autodiff-fft-ops-decision.md`）。
//!
//! **PyTorch 相当**（`docs/autodiff-fft-ops-decision.md` §1）:
//!
//! | 演算 | PyTorch 相当 | 入力 → 出力 |
//! |---|---|---|
//! | [`rfft`] | `torch.view_as_real(torch.fft.rfft(x, n, dim, norm))` | 実 `[..., L, ...]` → `[..., n/2+1, ..., 2]` |
//! | [`irfft`] | `torch.fft.irfft(torch.view_as_complex(x), n, dim, norm)` | `[..., m, ..., 2]` → 実 `[..., n, ...]` |
//! | `fft` | `torch.view_as_real(torch.fft.fft(torch.view_as_complex(x), n, dim, norm))` | `[..., L, ..., 2]` → `[..., n, ..., 2]` |
//! | `ifft` | `torch.view_as_real(torch.fft.ifft(torch.view_as_complex(x), n, dim, norm))` | 同上 |
//! | [`stft`] | `torch.view_as_real(torch.stft(x, n_fft, hop, win_length, window, center, pad_mode, normalized, onesided, return_complex=True))` | 実 `[L]`／`[B, L]` → `[N, T, 2]`／`[B, N, T, 2]`（**周波数軸がフレーム軸より前**） |
//! | [`istft`] | `torch.istft(torch.view_as_complex(x), n_fft, hop, win_length, window, center, normalized, onesided, length)` | `[N, T, 2]`／`[B, N, T, 2]` → 実 `[L_out]`／`[B, L_out]` |
//!
//! 複素数は末尾次元 2 の `f32` 実テンソル `(re, im)` で表す（complex dtype
//! は非目標）。`dim` は `usize`（負の添字は受けない）。`irfft` の `dim` は
//! 複素軸を除いた実軸の添字（既定は rank-2）。`fft`／`ifft` も同じ `dim` 規約で、
//! 実入力の自動昇格（虚部 0 の付加）はしない（呼び出し側の責務）。
//!
//! **経路**: ① 引数解決・形状・確保サイズの検査
//! （`fandhe_ai_tensor_core::fft::{rfft_layout, irfft_layout, fft_layout}`。
//! 確保・実体化より前）→ ② 入力の実体化 → ③ `BackendOps::fft_rfft`／
//! `fft_irfft`／`fft_fft`／`fft_ifft`（`Unsupported` のときだけ共有ホスト
//! カーネル `fft::rfft_host`／`irfft_host`／`fft_host`／`ifft_host` へ
//! フォールバックし、他のエラーは伝播する）→ ④ 専用 `Op`（`Op::Rfft`／
//! `Op::Irfft`／`Op::Fft`／`Op::Ifft`）を積む。VJP は `grad.rs`（共有カーネルの
//! `*_vjp_host`）。`stft`／`istft` も同じ経路（`fft_stft`／`fft_istft`・
//! `stft_host`／`istft_host`・`Op::Stft`／`Op::Istft`）で、`istft` は NOLA 検査
//! （`fft::istft_check_nola`）をバックエンド呼び出し前にも行い迂回させない。
//! 窓は非追跡の `Tensor<f32>` で、**窓への勾配は流れない**（PyTorch との差分。
//! 入力信号／スペクトルへの勾配のみ。`docs/autodiff-fft-ops-decision.md` §13）。
//!
//! **数値契約**: 内部は `f64` 逐次・固定順序で、最後に 1 回だけ `f32` へ
//! downcast（`fandhe_ai_tensor_core::fft` のモジュール doc が正。直接 DFT
//! のため 1 レーンあたり O(n²)）。非有限入力は拒否せず伝播する
//! （PyTorch と同じ。linalg 系の「非有限は `InvalidArgument`」とは異なる）。
//! 高階微分（`create_graph`）・activation checkpoint・f64 自動微分経路は
//! 対象外。

use fandhe_ai_tensor_core::fft::{self, FftLayout, IstftParams, StftParams};
use fandhe_ai_tensor_core::{BackendError, FftNorm, ShapeError, StftPadMode, Tensor};

use crate::error::AutodiffError;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

/// バックエンド実装の戻り値 shape を検証する（`linalg_ops::verify_shape` と
/// 同型。`var.rs` 側は非公開のため複製）。
fn verify_shape(actual: &[usize], expected: &[usize]) -> Result<(), AutodiffError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: actual.to_vec(),
                rhs: expected.to_vec(),
            },
        )))
    }
}

/// `Unsupported` 以外のバックエンドエラーを `AutodiffError` へ写像する
/// （`linalg_ops::unify_backend_error` と同型）。
fn unify_backend_error(err: BackendError) -> AutodiffError {
    match err {
        BackendError::InvalidArgument(msg) => AutodiffError::InvalidArgument(msg),
        other => AutodiffError::Backend(other),
    }
}

/// `x` を層 1 で実体化した `Tensor<f32>` を返す（`linalg_ops::materialize_one`
/// と同型）。
fn materialize_one<'t>(x: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = x.tape().nodes.borrow();
    let ops = x.tape().ops();
    Ok(materialize_fallible(&nodes, ops, x.node_id())?.clone())
}

/// ホスト参照カーネルの結果を `Tensor` へ包む。
fn wrap(data: Vec<f32>, layout: &FftLayout) -> Result<Tensor<f32>, AutodiffError> {
    Tensor::new(data, layout.out_shape()).map_err(AutodiffError::Shape)
}

/// 実数 FFT（実入力 `[..., L, ...]` → `Var`（`[..., n/2+1, ..., 2]`））。
///
/// `n` 省略時は変換軸の長さ `L`、`dim` 省略時は末尾軸。変換前に入力を長さ
/// `n` へ切り詰め／ゼロ詰めする。`norm` の既定は [`FftNorm::Backward`]。
///
/// **確保前の検査**: rank・`dim`・`n >= 1`・入出力の要素数／バイト数
/// （`checked_mul`。`isize::MAX` 超過は型付きエラー）を実体化より前に行う。
/// **非有限入力**: 拒否せず伝播する（DC・Nyquist の虚部はリテラル `0`）。
pub fn rfft<'t>(
    x: &Var<'t>,
    n: Option<usize>,
    dim: Option<usize>,
    norm: FftNorm,
) -> Result<Var<'t>, AutodiffError> {
    let layout = fft::rfft_layout(&x.shape(), n, dim)?;
    let input = materialize_one(x)?;
    let value = match x
        .tape()
        .ops()
        .fft_rfft(&input, layout.n(), layout.dim(), norm)
    {
        Ok(v) => {
            verify_shape(v.shape(), layout.out_shape())?;
            v
        }
        Err(BackendError::Unsupported(_)) => {
            wrap(fft::rfft_host(&input.host_slice(), &layout, norm)?, &layout)?
        }
        Err(other) => return Err(unify_backend_error(other)),
    };
    let id = x.tape().push_eager(
        Op::Rfft {
            input: x.node_id(),
            n: layout.n(),
            dim: layout.dim(),
            norm,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}

/// 実数逆 FFT（複素入力 `[..., m, ..., 2]` → `Var`（実 `[..., n, ...]`））。
///
/// `n` 省略時は `2*(m-1)`（`m == 1` で省略すると 0 になるため
/// `InvalidArgument`）。`dim` は複素軸を除いた実軸の添字（既定は rank-2）。
/// 入力 bin は `n/2+1` 個へ切り詰め／ゼロ詰めされ、DC と（`n` 偶数の）
/// Nyquist の虚部は無視される（PyTorch の C2R と同じ）。
///
/// 確保前の検査・非有限入力の扱いは [`rfft`] と同じ。
pub fn irfft<'t>(
    x: &Var<'t>,
    n: Option<usize>,
    dim: Option<usize>,
    norm: FftNorm,
) -> Result<Var<'t>, AutodiffError> {
    let layout = fft::irfft_layout(&x.shape(), n, dim)?;
    let input = materialize_one(x)?;
    let value = match x
        .tape()
        .ops()
        .fft_irfft(&input, layout.n(), layout.dim(), norm)
    {
        Ok(v) => {
            verify_shape(v.shape(), layout.out_shape())?;
            v
        }
        Err(BackendError::Unsupported(_)) => wrap(
            fft::irfft_host(&input.host_slice(), &layout, norm)?,
            &layout,
        )?,
        Err(other) => return Err(unify_backend_error(other)),
    };
    let id = x.tape().push_eager(
        Op::Irfft {
            input: x.node_id(),
            n: layout.n(),
            dim: layout.dim(),
            norm,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}

/// 複素 FFT（複素入力 `[..., L, ..., 2]` → `Var`（`[..., n, ..., 2]`））。
///
/// 入出力とも末尾次元 2 は `(re, im)`（`torch.view_as_real` と同レイアウト）。
/// `n` 省略時は変換軸の長さ `L`、`dim` は複素軸を除いた実軸の添字（既定は
/// rank-2）。変換前に入力を長さ `n` へ切り詰め／ゼロ詰めする。`norm` の既定は
/// [`FftNorm::Backward`]。実入力は受けない（rank・末尾次元 2 を検査して拒否）。
///
/// 確保前の検査・非有限入力の扱いは [`rfft`] と同じ（c2c には DC／Nyquist の
/// 構造的ゼロが無く、非有限入力はそのまま伝播する）。
pub fn fft<'t>(
    x: &Var<'t>,
    n: Option<usize>,
    dim: Option<usize>,
    norm: FftNorm,
) -> Result<Var<'t>, AutodiffError> {
    c2c(x, n, dim, norm, C2c::Forward)
}

/// 複素逆 FFT。`fft` と引数・検査・経路が同じで、符号とスケールだけが逆。
pub fn ifft<'t>(
    x: &Var<'t>,
    n: Option<usize>,
    dim: Option<usize>,
    norm: FftNorm,
) -> Result<Var<'t>, AutodiffError> {
    c2c(x, n, dim, norm, C2c::Inverse)
}

/// c2c の方向（`fft`／`ifft` の共通経路 [`c2c`] の分岐用）。
#[derive(Clone, Copy)]
enum C2c {
    Forward,
    Inverse,
}

/// `fft`／`ifft` の共通経路（検査 → 実体化 → バックエンド／ホストカーネル →
/// `Op` 記録）。
fn c2c<'t>(
    x: &Var<'t>,
    n: Option<usize>,
    dim: Option<usize>,
    norm: FftNorm,
    dir: C2c,
) -> Result<Var<'t>, AutodiffError> {
    let layout = fft::fft_layout(&x.shape(), n, dim)?;
    let input = materialize_one(x)?;
    let ops = x.tape().ops();
    let backend = match dir {
        C2c::Forward => ops.fft_fft(&input, layout.n(), layout.dim(), norm),
        C2c::Inverse => ops.fft_ifft(&input, layout.n(), layout.dim(), norm),
    };
    let value = match backend {
        Ok(v) => {
            verify_shape(v.shape(), layout.out_shape())?;
            v
        }
        Err(BackendError::Unsupported(_)) => {
            let data = match dir {
                C2c::Forward => fft::fft_host(&input.host_slice(), &layout, norm)?,
                C2c::Inverse => fft::ifft_host(&input.host_slice(), &layout, norm)?,
            };
            wrap(data, &layout)?
        }
        Err(other) => return Err(unify_backend_error(other)),
    };
    let (input_id, n, dim) = (x.node_id(), layout.n(), layout.dim());
    let op = match dir {
        C2c::Forward => Op::Fft {
            input: input_id,
            n,
            dim,
            norm,
        },
        C2c::Inverse => Op::Ifft {
            input: input_id,
            n,
            dim,
            norm,
        },
    };
    let id = x.tape().push_eager(op, value);
    Ok(Var::from_raw(x.tape(), id))
}

/// [`stft`] のオプション（`torch.stft` の `hop_length`／`win_length`／`center`／
/// `pad_mode`／`normalized`／`onesided` 引数相当）。PyTorch 既定を [`Default`] と
/// する。非破壊拡張に備え `#[non_exhaustive]`・`with_*` ビルダ方式を採る。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct StftOptions {
    /// フレーム間隔。`None` は `n_fft/4`（0 になる `n_fft < 4` では拒否）。
    pub hop_length: Option<usize>,
    /// 窓長。`None` は `n_fft`。`window` の長さと一致する必要がある。
    pub win_length: Option<usize>,
    /// `true` なら両端を `n_fft/2` 拡張してからフレーム化する。
    pub center: bool,
    /// `center = true` 時の端パディング種別。
    pub pad_mode: StftPadMode,
    /// `true` なら `1/√n_fft` で正規化する（`false` はスケールなし）。
    pub normalized: bool,
    /// `true` なら片側スペクトル（`n_fft/2+1` bin）、`false` なら全 bin。
    pub onesided: bool,
}

impl Default for StftOptions {
    fn default() -> Self {
        Self {
            hop_length: None,
            win_length: None,
            center: true,
            pad_mode: StftPadMode::Reflect,
            normalized: false,
            onesided: true,
        }
    }
}

impl StftOptions {
    /// フレーム間隔を設定する。
    pub fn with_hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = Some(hop_length);
        self
    }
    /// 窓長を設定する。
    pub fn with_win_length(mut self, win_length: usize) -> Self {
        self.win_length = Some(win_length);
        self
    }
    /// 両端パディングの有無を設定する。
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }
    /// 端パディング種別を設定する。
    pub fn with_pad_mode(mut self, pad_mode: StftPadMode) -> Self {
        self.pad_mode = pad_mode;
        self
    }
    /// 正規化の有無を設定する。
    pub fn with_normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }
    /// 片側スペクトルか否かを設定する。
    pub fn with_onesided(mut self, onesided: bool) -> Self {
        self.onesided = onesided;
        self
    }
}

/// [`istft`] のオプション（`torch.istft` の `hop_length`／`win_length`／`center`／
/// `normalized`／`onesided`／`length` 引数相当）。PyTorch 既定を [`Default`] と
/// する。[`StftOptions`] と同じく `#[non_exhaustive]`・`with_*` ビルダ方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct IstftOptions {
    /// フレーム間隔。`None` は `n_fft/4`。`win_length` 以下である必要がある。
    pub hop_length: Option<usize>,
    /// 窓長。`None` は `n_fft`。
    pub win_length: Option<usize>,
    /// `true` なら両端の `n_fft/2` を切り落とす。
    pub center: bool,
    /// `true` なら `1/√n_fft` 正規化（`stft` の `normalized` と対）。
    pub normalized: bool,
    /// `None` は入力 bin 数 ≠ `n_fft` なら片側と推定。`Some(false)` は先頭
    /// `n_fft/2+1` bin だけを読む（PyTorch 2.14.0 の実測。上位 bin の勾配は 0）。
    pub onesided: Option<bool>,
    /// 出力長。`None` は `center` に応じた既定長。期待長を超える分は 0 詰め。
    pub length: Option<usize>,
}

impl Default for IstftOptions {
    fn default() -> Self {
        Self {
            hop_length: None,
            win_length: None,
            center: true,
            normalized: false,
            onesided: None,
            length: None,
        }
    }
}

impl IstftOptions {
    /// フレーム間隔を設定する。
    pub fn with_hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = Some(hop_length);
        self
    }
    /// 窓長を設定する。
    pub fn with_win_length(mut self, win_length: usize) -> Self {
        self.win_length = Some(win_length);
        self
    }
    /// 両端の切り落としの有無を設定する。
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }
    /// 正規化の有無を設定する。
    pub fn with_normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }
    /// 片側スペクトルか否かを明示する。
    pub fn with_onesided(mut self, onesided: bool) -> Self {
        self.onesided = Some(onesided);
        self
    }
    /// 出力長を指定する。
    pub fn with_length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }
}

/// 窓テンソル（rank 1）を実効窓（長さ `n_fft`・中央寄せゼロ詰め）へ解決する。
fn resolve_window(
    window: Option<&Tensor<f32>>,
    win_length: usize,
    n_fft: usize,
) -> Result<Tensor<f32>, AutodiffError> {
    let data = match window {
        Some(w) => {
            if w.shape().len() != 1 {
                return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                    expected: 1,
                    actual: w.shape().len(),
                }));
            }
            fft::stft_window(Some(&w.host_slice()), Some(win_length), n_fft)?
        }
        None => fft::stft_window(None, Some(win_length), n_fft)?,
    };
    Tensor::new(data, &[n_fft]).map_err(AutodiffError::Shape)
}

/// 短時間フーリエ変換（実 `[L]`／`[B, L]` → `Var`（`[N, T, 2]`／`[B, N, T, 2]`））。
///
/// `N = n_fft/2+1`（`onesided`）または `n_fft`、`T = 1 + (L_pad − n_fft)/hop`。
/// `window` は rank 1・長さ `win_length`（`None` は矩形窓）で、`win_length <
/// n_fft` なら中央寄せでゼロ詰めされる。窓は非追跡で、窓への勾配は流れない。
///
/// **確保前の検査**: `n_fft`・`hop`・窓長・rank・反射パディング幅・フレーム数・
/// 作業／出力バッファの要素数とバイト数を実体化より前に検査し、違反は型付き
/// エラーで拒否する。**非有限入力**は拒否せず伝播する。
pub fn stft<'t>(
    x: &Var<'t>,
    n_fft: usize,
    window: Option<&Tensor<f32>>,
    options: &StftOptions,
) -> Result<Var<'t>, AutodiffError> {
    let params = StftParams::new(
        n_fft,
        options.hop_length,
        options.center,
        options.pad_mode,
        options.normalized,
        options.onesided,
    )?;
    let win = resolve_window(window, options.win_length.unwrap_or(n_fft), n_fft)?;
    let layout = fft::stft_layout(&x.shape(), &params)?;
    let input = materialize_one(x)?;
    let value = match x.tape().ops().fft_stft(&input, &win, &params) {
        Ok(v) => {
            verify_shape(v.shape(), layout.out_shape())?;
            v
        }
        Err(BackendError::Unsupported(_)) => Tensor::new(
            fft::stft_host(&input.host_slice(), &win.host_slice(), &layout)?,
            layout.out_shape(),
        )
        .map_err(AutodiffError::Shape)?,
        Err(other) => return Err(unify_backend_error(other)),
    };
    let id = x.tape().push_eager(
        Op::Stft {
            input: x.node_id(),
            window: win,
            params,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}

/// 逆短時間フーリエ変換（`[N, T, 2]`／`[B, N, T, 2]` → `Var`（実 `[L_out]`／
/// `[B, L_out]`））。
///
/// 各フレームを逆変換して窓を掛け、重畳加算した結果を窓二乗の重畳和（包絡）で
/// 割る。包絡の最小値が `fft::ISTFT_NOLA_MIN_ENVELOPE` 未満（NOLA 違反）なら
/// バックエンド呼び出し前に `InvalidArgument` で拒否する。`onesided = false` は
/// 先頭 `n_fft/2+1` bin だけを読む。確保前の検査・窓・非有限入力の扱いは
/// [`stft`] と同じ。
pub fn istft<'t>(
    x: &Var<'t>,
    n_fft: usize,
    window: Option<&Tensor<f32>>,
    options: &IstftOptions,
) -> Result<Var<'t>, AutodiffError> {
    let params = IstftParams::new(
        n_fft,
        options.hop_length,
        options.win_length,
        options.center,
        options.normalized,
        options.onesided,
        options.length,
    )?;
    let win = resolve_window(window, params.win_length(), n_fft)?;
    let layout = fft::istft_layout(&x.shape(), &params)?;
    fft::istft_check_nola(&win.host_slice(), &layout)?;
    let input = materialize_one(x)?;
    let value = match x.tape().ops().fft_istft(&input, &win, &params) {
        Ok(v) => {
            verify_shape(v.shape(), layout.out_shape())?;
            v
        }
        Err(BackendError::Unsupported(_)) => Tensor::new(
            fft::istft_host(&input.host_slice(), &win.host_slice(), &layout)?,
            layout.out_shape(),
        )
        .map_err(AutodiffError::Shape)?,
        Err(other) => return Err(unify_backend_error(other)),
    };
    let id = x.tape().push_eager(
        Op::Istft {
            input: x.node_id(),
            window: win,
            params,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}
