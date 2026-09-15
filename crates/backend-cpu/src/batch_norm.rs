//! BatchNorm1d／2d 順伝播カーネル（CPU 参照実装。`rayon` チャネル方向
//! 並列。イシュー #1732・親 #1608）。
//!
//! `layer_norm.rs`（#1596）と同じモジュール構成方針を踏襲する:
//! [`run_batch_norm_train_f32`]／[`run_batch_norm_infer_f32`] を公開
//! エントリとし、`ops.rs::CpuBackendOps::batch_norm_train`／
//! `batch_norm_infer` から直接呼ばれる。
//!
//! # チャネル軸・レイアウト契約
//!
//! `x` は `[n, c, spatial]` 相当の行優先平坦化済みスライス（NCHW／NCL
//! 固定。`fandhe_ai_tensor_core::batch_norm_layout` が導出。チャネル
//! `ch` の要素は `batch*(c*spatial) + ch*spatial + sp`〈`batch in
//! 0..n`・`sp in 0..spatial`〉に位置する非連続ストライドアクセス）。
//!
//! # 縮約精度契約（`.claude/rules/coding-rust.md`）
//!
//! `layer_norm.rs::warp_reduce_f64`（GPU warp／simdgroup butterfly 縮約
//! と同一の加算順序）をチャネル方向の `M = n*spatial` 要素へ適用する
//! （`fandhe_ai_autodiff::eval::channel_bn_stats` のホスト参照実装と
//! 同一の縮約順序・同一の演算列——両者は bit 完全一致する）。分散は
//! 「二パス」（`Σ(x−μ)²/M`）。`mean`／`rstd` を `f64` のまま偏差計算
//! まで保持し、出力書き込み直前の 1 回だけ `f32` へ downcast する
//! （`layer_norm_row` と同じ理由）。
//!
//! # 並列化方式（scatter 書き込みの安全性）
//!
//! チャネルごとの出力要素は元データ上で非連続（ストライド `spatial`・
//! `c*spatial`）であるため、`layer_norm.rs` の `par_chunks_mut` の
//! ように出力バッファを連続チャンクへ直接分割する並列化はできない。
//! 本実装はチャネルごとに独立な結果 `Vec<f32>`（長さ `M`）を
//! `rayon::par_iter` で計算し（各クロージャは他チャネルの状態に触れ
//! ないため `unsafe` を要しない）、最後に単一スレッドでスキャッタ
//! 書き込みする 2 段構成を取る（`unsafe` 非導入。`.claude/rules/
//! coding-rust.md` の REQ-8 手動境界検査規約はスライス添字アクセスの
//! みのため該当しない）。
//!
//! # 境界検査（REQ-8）
//!
//! `unsafe` intrinsics を使わない（スライス添字アクセスはすべて
//! Rust の境界検査を経る）ため、REQ-8 の手動境界検査規約は該当しない。

use rayon::prelude::*;

use crate::elementwise::PARALLEL_THRESHOLD;

/// [`run_batch_norm_train_f32`]／[`run_batch_norm_infer_f32`] の
/// 型付きエラー（`layer_norm::LayerNormError` と同じ「小さな enum」
/// 方針）。
#[non_exhaustive]
#[derive(Debug)]
pub enum BatchNormError {
    /// `n*c*spatial` が overflow するか、`x.len()` と一致しない。
    InvalidShape { detail: String },
    /// `w` が指定されているが `w.len() != c`。
    WeightLenMismatch { c: usize, w_len: usize },
    /// `b` が指定されているが `b.len() != c`。
    BiasLenMismatch { c: usize, b_len: usize },
    /// `mean`／`var`（eval モード）の長さが `c` と不一致。
    StatsLenMismatch { c: usize, len: usize },
    /// `eps` が有限でないか負。
    InvalidEps { eps: f32 },
}

impl std::fmt::Display for BatchNormError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchNormError::InvalidShape { detail } => {
                write!(f, "batch_norm invalid shape: {detail}")
            }
            BatchNormError::WeightLenMismatch { c, w_len } => {
                write!(
                    f,
                    "batch_norm weight length mismatch: c={c}, w.len()={w_len}"
                )
            }
            BatchNormError::BiasLenMismatch { c, b_len } => {
                write!(f, "batch_norm bias length mismatch: c={c}, b.len()={b_len}")
            }
            BatchNormError::StatsLenMismatch { c, len } => {
                write!(f, "batch_norm mean/var length mismatch: c={c}, len={len}")
            }
            BatchNormError::InvalidEps { eps } => {
                write!(
                    f,
                    "batch_norm eps must be finite and non-negative: eps={eps}"
                )
            }
        }
    }
}

impl std::error::Error for BatchNormError {}

/// 起動前 fail-closed 検証（`layer_norm::validate_layer_norm_launch`
/// と同型。OWASP A03・`.claude/rules/security.md`）: `eps` が有限かつ
/// 非負・`n*c*spatial == x.len()`（checked 乗算）・`w.len() == c`
/// （`w` 指定時）・`b.len() == c`（`b` 指定時）・`mean_len`／`var_len`
/// が `Some` の場合は `c` と一致。
#[allow(clippy::too_many_arguments)]
fn validate_batch_norm_launch(
    n: usize,
    c: usize,
    spatial: usize,
    x_len: usize,
    w_len: Option<usize>,
    b_len: Option<usize>,
    mean_len: Option<usize>,
    var_len: Option<usize>,
    eps: f32,
) -> Result<(), BatchNormError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(BatchNormError::InvalidEps { eps });
    }
    // `run_batch_norm_train_f32`／`run_batch_norm_eval_f32` はいずれも
    // 本関数の検査を通過したあとチャネル数 `c` だけに依存する `Vec<f32>`
    // （`mean`／`var`。n==0 早期 return 経路も含む）を確保する。
    // `n*c*spatial` の `usize` 積オーバーフローを検査するだけでは、
    // 例えば `n=0, c=usize::MAX, spatial=1` のように積自体は 0 へ
    // 潰れて検査を通過するが `vec![0.0f32; c]` 単体が `isize::MAX` バイト
    // 上限で capacity overflow panic する経路を防げない
    // （`fandhe_ai_tensor_core::tensor::checked_numel_for` と同じ理由。
    // 本番経路 panic 禁止規約 `.claude/rules/coding-rust.md` に反する。
    // イシュー #1732・PR #1874 codex-review P1 是正）。確保前に
    // `c * size_of::<f32>()` を `isize::MAX` 上限まで検査する。
    if c.checked_mul(std::mem::size_of::<f32>())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(BatchNormError::InvalidShape {
            detail: format!("channel count too large to allocate mean/var: c={c}"),
        });
    }
    let numel = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(spatial))
        .ok_or_else(|| BatchNormError::InvalidShape {
            detail: format!("n*c*spatial overflowed usize: n={n}, c={c}, spatial={spatial}"),
        })?;
    if numel != x_len {
        return Err(BatchNormError::InvalidShape {
            detail: format!("x length mismatch: n*c*spatial={numel}, x.len()={x_len}"),
        });
    }
    if let Some(wl) = w_len
        && wl != c
    {
        return Err(BatchNormError::WeightLenMismatch { c, w_len: wl });
    }
    if let Some(bl) = b_len
        && bl != c
    {
        return Err(BatchNormError::BiasLenMismatch { c, b_len: bl });
    }
    if let Some(ml) = mean_len
        && ml != c
    {
        return Err(BatchNormError::StatsLenMismatch { c, len: ml });
    }
    if let Some(vl) = var_len
        && vl != c
    {
        return Err(BatchNormError::StatsLenMismatch { c, len: vl });
    }
    Ok(())
}

/// GPU の warp／simdgroup butterfly 縮約と同一の加算順序（`layer_norm.
/// rs::warp_reduce_f64` の逐語複製。モジュール間で共有する共通クレート
/// が無いための重複——`fandhe_ai_autodiff::eval::warp_reduce_f64`
/// （同型）と合わせて 3 箇所目の複製になるが、いずれも `.claude/rules/
/// coding-rust.md` の縮約契約を単一の関数本体として固定するための
/// 意図的な重複であり、将来の共通化は別イシューの対象とする）。
fn warp_reduce_f64(m: usize, mut contribute: impl FnMut(usize, f64) -> f64) -> f64 {
    const LANES: usize = 32;
    let mut lanes = [0.0f64; LANES];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut idx = lane;
        while idx < m {
            *slot = contribute(idx, *slot);
            idx += LANES;
        }
    }
    let mut offset = 16usize;
    while offset > 0 {
        let snapshot = lanes;
        for (lane, slot) in lanes.iter_mut().enumerate() {
            *slot = snapshot[lane] + snapshot[lane ^ offset];
        }
        offset >>= 1;
    }
    lanes[0]
}

/// チャネル `ch` に属する `M = n*spatial` 要素の局所添字 `i` を実
/// データ添字へ写像する（モジュール doc comment「チャネル軸・
/// レイアウト契約」）。
#[inline]
fn channel_index(i: usize, ch: usize, c: usize, spatial: usize) -> usize {
    let batch = i / spatial;
    let sp = i % spatial;
    batch * (c * spatial) + ch * spatial + sp
}

/// affine（`w`／`b`）適用の共通 4 分岐（`layer_norm_row` と同じ FMA
/// 契約統一）。
#[inline]
fn apply_affine(xhat: f32, w: Option<f32>, b: Option<f32>) -> f32 {
    match (w, b) {
        (Some(w), Some(b)) => xhat.mul_add(w, b),
        (Some(w), None) => xhat * w,
        (None, Some(b)) => xhat + b,
        (None, None) => xhat,
    }
}

/// [`run_batch_norm_train_f32`] の戻り値。`out` は入力と同一 shape の
/// 平坦化データ、`mean`／`var`（biased ÷M）は長さ `c`。
#[derive(Debug)]
pub struct BatchNormTrainRaw {
    pub out: Vec<f32>,
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
}

/// チャネル `ch` の `M = n*spatial` 要素から `(mean, rstd, var)`
/// （すべて `f64`。`var` は biased ÷M）を計算する（1 チャネル分の
/// 縮約本体。並列化の単位）。
fn channel_stats(
    x: &[f32],
    ch: usize,
    c: usize,
    spatial: usize,
    m: usize,
    eps: f32,
) -> (f64, f64, f64) {
    let mm = m as f64;
    let sum = warp_reduce_f64(m, |i, acc| acc + x[channel_index(i, ch, c, spatial)] as f64);
    let mean = sum / mm;
    let sq_acc = warp_reduce_f64(m, |i, acc| {
        let d = x[channel_index(i, ch, c, spatial)] as f64 - mean;
        d.mul_add(d, acc)
    });
    let var = sq_acc / mm;
    let rstd = 1.0f64 / (var + eps as f64).sqrt();
    (mean, rstd, var)
}

/// BatchNorm1d／2d train モード（バッチ統計）。
///
/// `x` は `[n, c, spatial]` 相当の行優先平坦化済みスライス。
/// `n == 0 || c == 0 || spatial == 0` は全ゼロ出力・全ゼロ統計を返す
/// （`layer_norm::run_layer_norm_f32` の早期 return 契約と同型）。
pub fn run_batch_norm_train_f32(
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    n: usize,
    c: usize,
    spatial: usize,
) -> Result<BatchNormTrainRaw, BatchNormError> {
    validate_batch_norm_launch(
        n,
        c,
        spatial,
        x.len(),
        w.map(|s| s.len()),
        b.map(|s| s.len()),
        None,
        None,
        eps,
    )?;

    if n == 0 || c == 0 || spatial == 0 {
        return Ok(BatchNormTrainRaw {
            out: Vec::new(),
            mean: vec![0.0f32; c],
            var: vec![0.0f32; c],
        });
    }

    let m = n * spatial;
    let numel = x.len();

    // チャネルごとに独立に計算する（モジュール doc comment「並列化
    // 方式」参照）。`numel`／`c` がともに閾値を跨ぐ場合のみ並列化する
    // （`layer_norm.rs` の `numel >= PARALLEL_THRESHOLD && rows >= 2`
    // と同型の判定軸をチャネル数へ適用）。
    let compute_channel = |ch: usize| -> (f64, f64, Vec<f32>) {
        let (mean, rstd, var) = channel_stats(x, ch, c, spatial, m, eps);
        let wv = w.map(|w| w[ch]);
        let bv = b.map(|b| b[ch]);
        let mut channel_out = vec![0.0f32; m];
        for (i, o) in channel_out.iter_mut().enumerate() {
            let idx = channel_index(i, ch, c, spatial);
            let xhat = ((x[idx] as f64 - mean) * rstd) as f32;
            *o = apply_affine(xhat, wv, bv);
        }
        (mean, var, channel_out)
    };

    let per_channel: Vec<(f64, f64, Vec<f32>)> = if numel >= PARALLEL_THRESHOLD && c >= 2 {
        (0..c).into_par_iter().map(compute_channel).collect()
    } else {
        (0..c).map(compute_channel).collect()
    };

    let mut out = vec![0.0f32; numel];
    let mut mean = vec![0.0f32; c];
    let mut var = vec![0.0f32; c];
    for (ch, (mean_ch, var_ch, channel_out)) in per_channel.into_iter().enumerate() {
        mean[ch] = mean_ch as f32;
        var[ch] = var_ch as f32;
        for (i, v) in channel_out.into_iter().enumerate() {
            out[channel_index(i, ch, c, spatial)] = v;
        }
    }

    Ok(BatchNormTrainRaw { out, mean, var })
}

/// BatchNorm1d／2d eval モード（固定統計）。`mean`／`var`（長さ `c`）
/// はバッチから計算し直さずそのまま使う。
#[allow(clippy::too_many_arguments)]
pub fn run_batch_norm_infer_f32(
    x: &[f32],
    mean: &[f32],
    var: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    n: usize,
    c: usize,
    spatial: usize,
) -> Result<Vec<f32>, BatchNormError> {
    validate_batch_norm_launch(
        n,
        c,
        spatial,
        x.len(),
        w.map(|s| s.len()),
        b.map(|s| s.len()),
        Some(mean.len()),
        Some(var.len()),
        eps,
    )?;

    if n == 0 || c == 0 || spatial == 0 {
        return Ok(Vec::new());
    }

    let m = n * spatial;
    let numel = x.len();

    let compute_channel = |ch: usize| -> Vec<f32> {
        let mean_c = mean[ch] as f64;
        let rstd = 1.0f64 / (var[ch] as f64 + eps as f64).sqrt();
        let wv = w.map(|w| w[ch]);
        let bv = b.map(|b| b[ch]);
        let mut channel_out = vec![0.0f32; m];
        for (i, o) in channel_out.iter_mut().enumerate() {
            let idx = channel_index(i, ch, c, spatial);
            let xhat = ((x[idx] as f64 - mean_c) * rstd) as f32;
            *o = apply_affine(xhat, wv, bv);
        }
        channel_out
    };

    let per_channel: Vec<Vec<f32>> = if numel >= PARALLEL_THRESHOLD && c >= 2 {
        (0..c).into_par_iter().map(compute_channel).collect()
    } else {
        (0..c).map(compute_channel).collect()
    };

    let mut out = vec![0.0f32; numel];
    for (ch, channel_out) in per_channel.into_iter().enumerate() {
        for (i, v) in channel_out.into_iter().enumerate() {
            out[channel_index(i, ch, c, spatial)] = v;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_batch_norm_launch_accepts_matching_dims() {
        assert!(
            validate_batch_norm_launch(2, 3, 4, 24, Some(3), Some(3), None, None, 1e-5).is_ok()
        );
        assert!(validate_batch_norm_launch(2, 3, 4, 24, None, None, None, None, 1e-5).is_ok());
    }

    #[test]
    fn validate_batch_norm_launch_rejects_x_len_mismatch() {
        let err =
            validate_batch_norm_launch(2, 3, 4, 23, None, None, None, None, 1e-5).unwrap_err();
        assert!(matches!(err, BatchNormError::InvalidShape { .. }));
    }

    #[test]
    fn validate_batch_norm_launch_rejects_w_len_mismatch() {
        let err =
            validate_batch_norm_launch(2, 3, 4, 24, Some(2), None, None, None, 1e-5).unwrap_err();
        assert!(matches!(err, BatchNormError::WeightLenMismatch { .. }));
    }

    #[test]
    fn validate_batch_norm_launch_rejects_stats_len_mismatch() {
        let err = validate_batch_norm_launch(2, 3, 4, 24, None, None, Some(2), Some(3), 1e-5)
            .unwrap_err();
        assert!(matches!(err, BatchNormError::StatsLenMismatch { .. }));
    }

    #[test]
    fn validate_batch_norm_launch_rejects_non_finite_eps() {
        for eps in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err =
                validate_batch_norm_launch(2, 3, 4, 24, None, None, None, None, eps).unwrap_err();
            assert!(matches!(err, BatchNormError::InvalidEps { .. }));
        }
    }

    #[test]
    fn run_batch_norm_train_f32_rejects_invalid_eps() {
        let x = vec![1.0f32; 4];
        let err = run_batch_norm_train_f32(&x, None, None, -1.0, 2, 1, 2).unwrap_err();
        assert!(matches!(err, BatchNormError::InvalidEps { .. }));
    }

    #[test]
    fn run_batch_norm_train_f32_matches_manual_computation() {
        // n=1,c=2,spatial=2: ch0=[1,2] mean=1.5 var=0.25; ch1=[3,4]
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let raw = run_batch_norm_train_f32(&x, None, None, 0.0, 1, 2, 2).unwrap();
        assert!((raw.mean[0] as f64 - 1.5).abs() < 1e-9);
        assert!((raw.mean[1] as f64 - 3.5).abs() < 1e-9);
        assert!((raw.var[0] as f64 - 0.25).abs() < 1e-9);
        let rstd = 1.0f32 / 0.25f32.sqrt();
        let expected = [
            (1.0 - 1.5) * rstd,
            (2.0 - 1.5) * rstd,
            (3.0 - 3.5) * rstd,
            (4.0 - 3.5) * rstd,
        ];
        for (o, e) in raw.out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "o={o} e={e}");
        }
    }

    #[test]
    fn run_batch_norm_train_f32_reduces_over_batch_and_spatial() {
        // n=2,c=1,spatial=2: 全 4 要素が 1 チャネルへ縮約される。
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let raw = run_batch_norm_train_f32(&x, None, None, 0.0, 2, 1, 2).unwrap();
        assert!((raw.mean[0] as f64 - 2.5).abs() < 1e-9);
        assert!((raw.var[0] as f64 - 1.25).abs() < 1e-9);
    }

    #[test]
    fn run_batch_norm_train_f32_applies_weight_and_bias() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = [2.0f32, 0.5];
        let b = [1.0f32, -1.0];
        let raw = run_batch_norm_train_f32(&x, Some(&w), Some(&b), 0.0, 1, 2, 2).unwrap();
        let rstd = 1.0f32 / 0.25f32.sqrt();
        let expected = [
            (1.0 - 1.5) * rstd * 2.0 + 1.0,
            (2.0 - 1.5) * rstd * 2.0 + 1.0,
            (3.0 - 3.5) * rstd * 0.5 - 1.0,
            (4.0 - 3.5) * rstd * 0.5 - 1.0,
        ];
        for (o, e) in raw.out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "o={o} e={e}");
        }
    }

    #[test]
    fn run_batch_norm_train_f32_empty_axes_are_empty_output() {
        let raw = run_batch_norm_train_f32(&[], None, None, 1e-5, 0, 2, 2).unwrap();
        assert_eq!(raw.out, Vec::<f32>::new());
        assert_eq!(raw.mean, vec![0.0f32; 2]);
        assert_eq!(raw.var, vec![0.0f32; 2]);
    }

    #[test]
    fn run_batch_norm_train_f32_nan_propagates() {
        let x = vec![f32::NAN, 1.0, 1.0, 1.0];
        let raw = run_batch_norm_train_f32(&x, None, None, 1e-5, 1, 2, 2).unwrap();
        assert!(raw.out[0].is_nan() && raw.out[1].is_nan());
        assert!(raw.mean[0].is_nan());
        assert!(!raw.mean[1].is_nan());
    }

    /// `n=0, c=usize::MAX, spatial=1` は `n*c*spatial` が `usize` 積として
    /// は `0` に潰れて `validate_batch_norm_launch` の旧実装（オーバー
    /// フロー検査のみ）を通過し、`n==0` 早期 return 経路の
    /// `vec![0.0f32; c]`（`mean`／`var`）が `isize::MAX` バイト上限で
    /// capacity overflow panic していた（本番経路 panic 禁止規約
    /// `.claude/rules/coding-rust.md` 違反。PR #1874 codex-review P1）。
    /// `c * size_of::<f32>()` の確保前上限検査後は型付きエラーへ収束
    /// することを確認する（panic しないこと自体が本テストの主目的）。
    #[test]
    fn run_batch_norm_train_f32_rejects_huge_channel_count_without_panicking() {
        let err = run_batch_norm_train_f32(&[], None, None, 1e-5, 0, usize::MAX, 1).unwrap_err();
        assert!(matches!(err, BatchNormError::InvalidShape { .. }));
    }

    #[test]
    fn validate_batch_norm_launch_rejects_huge_channel_count() {
        let err = validate_batch_norm_launch(0, usize::MAX, 1, 0, None, None, None, None, 1e-5)
            .unwrap_err();
        assert!(matches!(err, BatchNormError::InvalidShape { .. }));
    }

    #[test]
    fn run_batch_norm_train_f32_extreme_scale_does_not_overflow() {
        let x = vec![2e20f32, -2e20, 2e20, -2e20];
        let raw = run_batch_norm_train_f32(&x, None, None, 1e-5, 1, 2, 2).unwrap();
        assert!(raw.out.iter().all(|v| v.is_finite()), "{:?}", raw.out);
    }

    #[test]
    fn run_batch_norm_infer_f32_uses_fixed_stats() {
        let x = vec![10.0f32, 20.0, 30.0, 40.0];
        let mean = [0.0f32, 0.0];
        let var = [1.0f32, 1.0];
        let out = run_batch_norm_infer_f32(&x, &mean, &var, None, None, 0.0, 1, 2, 2).unwrap();
        assert_eq!(out, vec![10.0f32, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn run_batch_norm_infer_f32_empty_axes_are_empty_output() {
        let mean = [0.0f32, 0.0];
        let var = [1.0f32, 1.0];
        let out = run_batch_norm_infer_f32(&[], &mean, &var, None, None, 1e-5, 0, 2, 2).unwrap();
        assert_eq!(out, Vec::<f32>::new());
    }

    /// rayon 並列閾値を跨いだ場合（`c` 方向並列）でも、各チャネルを
    /// 単独（`c=1`。`numel` が小さく並列閾値未満のため逐次経路を
    /// 通る）で計算した結果と bit 完全一致することを確認する
    /// （`layer_norm.rs` の
    /// `run_layer_norm_f32_multi_row_parallel_path_matches_serial` と
    /// 同型の「チャネル間の独立性」回帰。並列版は他チャネルの状態を
    /// 一切参照しない設計〈モジュール doc comment「並列化方式」〉の
    /// ため、単独計算と一致するはずである）。
    #[test]
    fn run_batch_norm_train_f32_parallel_path_matches_per_channel_serial() {
        let c = 8usize;
        let n = 2usize;
        let spatial = 1usize << 12; // numel = 2*8*4096 = 65536 >= PARALLEL_THRESHOLD
        let numel = n * c * spatial;
        let x: Vec<f32> = (0..numel).map(|i| (i % 97) as f32 * 0.1).collect();
        let raw_par = run_batch_norm_train_f32(&x, None, None, 1e-5, n, c, spatial).unwrap();

        for ch in 0..c {
            // チャネル ch のみを抜き出した [n, 1, spatial] 入力を構築
            // する（`numel` が小さいため PARALLEL_THRESHOLD 未満の
            // 逐次経路を通る）。
            let mut x_ch = vec![0.0f32; n * spatial];
            for i in 0..(n * spatial) {
                x_ch[i] = x[channel_index(i, ch, c, spatial)];
            }
            let raw_single =
                run_batch_norm_train_f32(&x_ch, None, None, 1e-5, n, 1, spatial).unwrap();

            assert_eq!(raw_par.mean[ch], raw_single.mean[0], "ch={ch}");
            assert_eq!(raw_par.var[ch], raw_single.var[0], "ch={ch}");
            for i in 0..(n * spatial) {
                let idx = channel_index(i, ch, c, spatial);
                assert_eq!(raw_par.out[idx], raw_single.out[i], "ch={ch} i={i}");
            }
        }
    }
}
