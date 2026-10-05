//! `rfft`・`irfft` の自由関数（イシュー #2631・親 #2630「FFT」。ルート
//! #2499 Phase 4）。
//!
//! **facade 非公開（保留）**: 公開形（`Var::rfft`／`Var::irfft` の委譲
//! メソッドと `FftNorm` の再エクスポート）は未承認で、承認依頼は #2677
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
//!
//! 複素数は末尾次元 2 の `f32` 実テンソル `(re, im)` で表す（complex dtype
//! は非目標）。`dim` は `usize`（負の添字は受けない）。`irfft` の `dim` は
//! 複素軸を除いた実軸の添字（既定は rank-2）。
//!
//! **経路**: ① 引数解決・形状・確保サイズの検査
//! （`fandhe_ai_tensor_core::fft::{rfft_layout, irfft_layout}`。確保・
//! 実体化より前）→ ② 入力の実体化 → ③ `BackendOps::fft_rfft`／`fft_irfft`
//! （`Unsupported` のときだけ共有ホストカーネル `fft::rfft_host`／
//! `irfft_host` へフォールバックし、他のエラーは伝播する）→ ④ 専用 `Op`
//! （`Op::Rfft`／`Op::Irfft`）を積む。VJP は `grad.rs`（共有カーネルの
//! `*_vjp_host`）。
//!
//! **数値契約**: 内部は `f64` 逐次・固定順序で、最後に 1 回だけ `f32` へ
//! downcast（`fandhe_ai_tensor_core::fft` のモジュール doc が正。直接 DFT
//! のため 1 レーンあたり O(n²)）。非有限入力は拒否せず伝播する
//! （PyTorch と同じ。linalg 系の「非有限は `InvalidArgument`」とは異なる）。
//! 高階微分（`create_graph`）・activation checkpoint・f64 自動微分経路は
//! 対象外。

use fandhe_ai_tensor_core::fft::{self, FftLayout};
use fandhe_ai_tensor_core::{BackendError, FftNorm, ShapeError, Tensor};

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
    Tensor::new(data, &layout.out_shape).map_err(AutodiffError::Shape)
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
    let value = match x.tape().ops().fft_rfft(&input, layout.n, layout.dim, norm) {
        Ok(v) => {
            verify_shape(v.shape(), &layout.out_shape)?;
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
            n: layout.n,
            dim: layout.dim,
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
    let value = match x.tape().ops().fft_irfft(&input, layout.n, layout.dim, norm) {
        Ok(v) => {
            verify_shape(v.shape(), &layout.out_shape)?;
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
            n: layout.n,
            dim: layout.dim,
            norm,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}
