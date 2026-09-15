//! BatchNorm1d／2d 順伝播カーネル（イシュー #1736・親 #1608・設計
//! `docs/batch-norm-ops-design.md`）のホスト側検証・逐語モデル。
//! `crate::soft_f64` の binary64 ソフトウェアエミュレーション
//! （[`crate::soft_f64::widen_f32_bits`]／[`crate::soft_f64::
//! add_f64_bits`]／[`crate::soft_f64::sub_f64_bits`]／
//! [`crate::soft_f64::mul_f64_bits`]／[`crate::soft_f64::div_f64_bits`]／
//! [`crate::soft_f64::add_f64_bits_round_to_odd`]／[`crate::soft_f64::
//! narrow_f64_bits`]／[`crate::soft_f64::rsqrt_newton_f64_bits`]）を
//! `shaders/batch_norm.metal::bn_f64_*` と同じ演算列で呼び出すことで、
//! GPU 側カーネルが正しい soft-f64 演算列を実行することを Apple
//! Silicon 実機に到達できない環境（本実装環境。Linux・CI）でも機械的に
//! 裏付ける（`im2col_model.rs`・`scan_model.rs` と同じ設計判断:
//! `objc2` 系 FFI に触れないため `cfg(target_os = "macos")` を付けず、
//! `lib.rs` で cfg なしブロックへ配置する）。
//!
//! # なぜこれが Metal 側の正しさの根拠になるか
//!
//! [`batch_norm_train_host_model`]／[`batch_norm_infer_host_model`] は
//! `shaders/batch_norm.metal::batch_norm_train_f32`／
//! `batch_norm_infer_f32` の逐語移植（縮約順序・soft-f64 演算列・
//! round-to-odd affine とも本ファイルの `crate::soft_f64` 呼び出しと
//! 1 対 1 対応する）である。本モジュールの単体テストはこれらを
//! `fandhe_ai_backend_cpu::{run_batch_norm_train_f32, run_batch_norm_infer_f32}`
//! （dev-dependency）と REQ-2 統一複合判定（`fandhe_ai_backend_cpu::
//! parity::assert_parity`）で突き合わせる（CPU との bit 一致は
//! 主張しない。`shaders/batch_norm.metal` 冒頭コメント「REQ-2 判定
//! 契約」参照）。
//!
//! [`validate_batch_norm_launch`]・[`channel_index`]・[`m_f64_bits`]
//! は `crate::batch_norm`（macOS 限定の起動 API）からも呼ばれる純関数
//! （`backend-cpu::batch_norm::{validate_batch_norm_launch,
//! channel_index}` の GPU 側複製。カーネル引数の `u32` 上限検査のみ
//! CPU 側に無い追加検査）。

use crate::soft_f64::{
    add_f64_bits, add_f64_bits_round_to_odd, div_f64_bits, mul_f64_bits, narrow_f64_bits,
    rsqrt_newton_f64_bits, sub_f64_bits, widen_f32_bits,
};

/// [`validate_batch_norm_launch`] の型付きエラー（`backend-cpu::
/// batch_norm::BatchNormError`・`crate::im2col_model::
/// Im2colPrepareError` と同型の「小さな enum」方針）。
#[non_exhaustive]
#[derive(Debug)]
pub enum BatchNormPrepareError {
    /// `n*c*spatial` が overflow するか `x_len` と一致しない・
    /// `w`／`b`／`mean`／`var` の長さが `c` と不一致・`eps` が有限
    /// でないか負・チャネル数の確保サイズが `isize::MAX` を超える
    /// （`backend-cpu::batch_norm::BatchNormError` の各 variant を
    /// 1 つの enum へ集約。`detail` に具体的な検査項目を記す）。
    InvalidShape { detail: String },
    /// `n`／`c`／`spatial`／`m`／`numel` のいずれかがカーネル引数の
    /// `uint`（`u32`）上限を超える（`crate::im2col_model::
    /// Im2colPrepareError::SizeLimitExceeded` と同型の設計判断。
    /// `ops.rs` 側でこの variant のみ `BackendError::Unsupported`
    /// へ写像しホストフォールバックへ委ねる）。
    SizeLimitExceeded { detail: String },
}

impl std::fmt::Display for BatchNormPrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchNormPrepareError::InvalidShape { detail } => {
                write!(f, "batch_norm invalid shape: {detail}")
            }
            BatchNormPrepareError::SizeLimitExceeded { detail } => {
                write!(f, "batch_norm size limit exceeded: {detail}")
            }
        }
    }
}

impl std::error::Error for BatchNormPrepareError {}

/// `n`／`c`／`spatial`／`m`／`numel` がカーネル引数の `uint`（`u32`）
/// 契約に収まる上限（`crate::im2col_model::IM2COL_KERNEL_ARG_LIMIT`
/// と同型）。
pub const BATCH_NORM_KERNEL_ARG_LIMIT: usize = u32::MAX as usize;

/// 起動前 fail-closed 検証（`backend-cpu::batch_norm::
/// validate_batch_norm_launch` と同型 + `u32` 上限検査を追加。
/// OWASP A03・`.claude/rules/security.md`）: `eps` が有限かつ非負・
/// `n*c*spatial == x_len`（checked 乗算）・`w_len`／`b_len`／
/// `mean_len`／`var_len` が `Some` の場合は `c` と一致・
/// `c * size_of::<f32>() <= isize::MAX`（`n=0, c=usize::MAX` のような
/// 退化入力での `mean`／`var` 確保時 capacity overflow panic を防ぐ。
/// CPU 側 `batch_norm::validate_batch_norm_launch`〈PR #1874
/// codex-review P1〉と同じ対策）・`n`／`c`／`spatial`／`m`
/// （`n*spatial`）／`numel` が [`BATCH_NORM_KERNEL_ARG_LIMIT`]
/// （`u32::MAX`）以下であること。
#[allow(clippy::too_many_arguments)]
pub fn validate_batch_norm_launch(
    n: usize,
    c: usize,
    spatial: usize,
    x_len: usize,
    w_len: Option<usize>,
    b_len: Option<usize>,
    mean_len: Option<usize>,
    var_len: Option<usize>,
    eps: f32,
) -> Result<(), BatchNormPrepareError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(BatchNormPrepareError::InvalidShape {
            detail: format!("batch_norm eps must be finite and non-negative: eps={eps}"),
        });
    }
    if c.checked_mul(std::mem::size_of::<f32>())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(BatchNormPrepareError::InvalidShape {
            detail: format!("channel count too large to allocate mean/var: c={c}"),
        });
    }
    let numel = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(spatial))
        .ok_or_else(|| BatchNormPrepareError::InvalidShape {
            detail: format!("n*c*spatial overflowed usize: n={n}, c={c}, spatial={spatial}"),
        })?;
    if numel != x_len {
        return Err(BatchNormPrepareError::InvalidShape {
            detail: format!("x length mismatch: n*c*spatial={numel}, x.len()={x_len}"),
        });
    }
    if let Some(wl) = w_len
        && wl != c
    {
        return Err(BatchNormPrepareError::InvalidShape {
            detail: format!("weight length mismatch: c={c}, w.len()={wl}"),
        });
    }
    if let Some(bl) = b_len
        && bl != c
    {
        return Err(BatchNormPrepareError::InvalidShape {
            detail: format!("bias length mismatch: c={c}, b.len()={bl}"),
        });
    }
    if let Some(ml) = mean_len
        && ml != c
    {
        return Err(BatchNormPrepareError::InvalidShape {
            detail: format!("mean length mismatch: c={c}, mean.len()={ml}"),
        });
    }
    if let Some(vl) = var_len
        && vl != c
    {
        return Err(BatchNormPrepareError::InvalidShape {
            detail: format!("var length mismatch: c={c}, var.len()={vl}"),
        });
    }

    let m = n
        .checked_mul(spatial)
        .ok_or_else(|| BatchNormPrepareError::InvalidShape {
            detail: format!("n*spatial overflowed usize: n={n}, spatial={spatial}"),
        })?;
    if n > BATCH_NORM_KERNEL_ARG_LIMIT
        || c > BATCH_NORM_KERNEL_ARG_LIMIT
        || spatial > BATCH_NORM_KERNEL_ARG_LIMIT
        || m > BATCH_NORM_KERNEL_ARG_LIMIT
        || numel > BATCH_NORM_KERNEL_ARG_LIMIT
    {
        return Err(BatchNormPrepareError::SizeLimitExceeded {
            detail: format!(
                "batch_norm dims must fit in u32 (kernel argument type): n={n}, c={c}, \
                 spatial={spatial}, m={m}, numel={numel}"
            ),
        });
    }
    Ok(())
}

/// チャネル `ch` に属する `M = n*spatial` 要素の局所添字 `i` を実
/// データ添字へ写像する（`backend-cpu::batch_norm::channel_index`・
/// `shaders/batch_norm.metal::BN_IDX` の GPU 側複製）。
#[inline]
pub fn channel_index(i: usize, ch: usize, c: usize, spatial: usize) -> usize {
    let batch = i / spatial;
    let sp = i % spatial;
    batch * (c * spatial) + ch * spatial + sp
}

/// `M`（`n*spatial`）の `f64` 表現を厳密なビットパターンへ変換し、
/// カーネル引数 `m_f64_hi`／`m_f64_lo`（上位／下位 32bit）へ渡す
/// 形へ分割する（`shaders/batch_norm.metal` 冒頭コメント「`M` の f64
/// 表現」参照。`m` はカーネル引数の `uint` 契約〈[`BATCH_NORM_KERNEL_
/// ARG_LIMIT`]〉に収まる前提のため `as f64` は丸めなしの厳密変換）。
#[inline]
pub fn m_f64_bits(m: usize) -> (u32, u32) {
    let bits = (m as f64).to_bits();
    ((bits >> 32) as u32, bits as u32)
}

/// affine（`w`／`b`）適用を soft-f64 の round-to-odd 経由
/// （`shaders/batch_norm.metal` の `bn_f64_add_ro` 経由 affine と
/// 同一の演算列）で計算する（`has_weight`／`has_bias == 0` のとき
/// カーネル側は `wv=1.0`／`bv=0.0` を使うが、それでも `mul`／
/// `add_ro` を経由するため本モデルも同じ経路を通す）。
#[inline]
fn apply_affine_soft_f64(xhat_bits: u32, w: f32, b: f32) -> u32 {
    let w64 = widen_f32_bits(w.to_bits());
    let b64 = widen_f32_bits(b.to_bits());
    let affine64 = add_f64_bits_round_to_odd(mul_f64_bits(widen_f32_bits(xhat_bits), w64), b64);
    narrow_f64_bits(affine64)
}

/// [`batch_norm_train_host_model`] の戻り値（`crate::batch_norm::
/// BatchNormTrainRaw`〈macOS 限定〉と同型。本モジュールは
/// `cfg(target_os = "macos")` を付けないため独立に定義する）。
#[derive(Debug, Clone, PartialEq)]
pub struct BatchNormTrainHostModel {
    pub out: Vec<f32>,
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
}

/// `shaders/batch_norm.metal::batch_norm_train_f32` の逐語モデル
/// （縮約順序は `crate::soft_f64` の `add_f64_bits` を index 順に
/// チャネルごと `M` 要素へ適用する——カーネル側の 32 レーン butterfly
/// とは異なる順序だが、soft-f64 加算の非結合性の影響は
/// REQ-2 統一複合判定の範囲内で無視できる〈`shaders/batch_norm.metal`
/// 冒頭コメント「総和の順序」と同じ判断〉ため、本モデルは索引順の
/// 逐次加算で代用する。二重丸め回避・round-to-odd affine の構造は
/// カーネルと 1 対 1 対応する）。
///
/// 呼び出し前提: [`validate_batch_norm_launch`] を通過済みの
/// `n`／`c`／`spatial`・`x.len() == n*c*spatial`。`n == 0 || c == 0
/// || spatial == 0` は空出力・ゼロ統計を返す（`crate::batch_norm::
/// run_batch_norm_train_f32` と同じ早期 return 契約）。
pub fn batch_norm_train_host_model(
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    n: usize,
    c: usize,
    spatial: usize,
) -> BatchNormTrainHostModel {
    if n == 0 || c == 0 || spatial == 0 {
        return BatchNormTrainHostModel {
            out: Vec::new(),
            mean: vec![0.0f32; c],
            var: vec![0.0f32; c],
        };
    }

    let m = n * spatial;
    // `m_f64_bits` の厳密ビット渡しをそのまま再構成する（カーネル側
    // `((ulong)m_f64_hi << 32) | (ulong)m_f64_lo` と同じ手順）。
    let m64 = {
        let (hi, lo) = m_f64_bits(m);
        (u64::from(hi) << 32) | u64::from(lo)
    };

    let mut out = vec![0.0f32; x.len()];
    let mut mean_out = vec![0.0f32; c];
    let mut var_out = vec![0.0f32; c];

    for ch in 0..c {
        let mut sum64 = 0u64; // +0.0（f64）。
        for i in 0..m {
            let xv = widen_f32_bits(x[channel_index(i, ch, c, spatial)].to_bits());
            sum64 = add_f64_bits(sum64, xv);
        }
        let mean64 = div_f64_bits(sum64, m64);

        let mut sq64 = 0u64;
        for i in 0..m {
            let xv = widen_f32_bits(x[channel_index(i, ch, c, spatial)].to_bits());
            let dev = sub_f64_bits(xv, mean64);
            let devsq = mul_f64_bits(dev, dev);
            sq64 = add_f64_bits(sq64, devsq);
        }
        let var64 = div_f64_bits(sq64, m64);
        let eps64 = widen_f32_bits(eps.to_bits());
        let rstd64 = rsqrt_newton_f64_bits(add_f64_bits(var64, eps64));

        mean_out[ch] = f32::from_bits(narrow_f64_bits(mean64));
        var_out[ch] = f32::from_bits(narrow_f64_bits(var64));

        let wv = w.map_or(1.0f32, |w| w[ch]);
        let bv = b.map_or(0.0f32, |b| b[ch]);
        for i in 0..m {
            let idx = channel_index(i, ch, c, spatial);
            let xv = widen_f32_bits(x[idx].to_bits());
            let dev = sub_f64_bits(xv, mean64);
            let xhat64 = mul_f64_bits(dev, rstd64);
            let xhat_bits = narrow_f64_bits(xhat64);
            out[idx] = f32::from_bits(apply_affine_soft_f64(xhat_bits, wv, bv));
        }
    }

    BatchNormTrainHostModel {
        out,
        mean: mean_out,
        var: var_out,
    }
}

/// `shaders/batch_norm.metal::batch_norm_infer_f32` の逐語モデル
/// （統計を再計算しないため縮約なし。[`batch_norm_train_host_model`]
/// の書き出しパスと同じ soft-f64 演算列を要素ごとに適用する）。
///
/// 呼び出し前提: [`validate_batch_norm_launch`] を通過済みの
/// `n`／`c`／`spatial`・`x.len() == n*c*spatial`・
/// `mean.len() == var.len() == c`。`n == 0 || c == 0 || spatial == 0`
/// は空出力を返す。
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_infer_host_model(
    x: &[f32],
    mean: &[f32],
    var: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    n: usize,
    c: usize,
    spatial: usize,
) -> Vec<f32> {
    if n == 0 || c == 0 || spatial == 0 {
        return Vec::new();
    }

    let mut out = vec![0.0f32; x.len()];
    for ch in 0..c {
        let mean64 = widen_f32_bits(mean[ch].to_bits());
        let var64 = widen_f32_bits(var[ch].to_bits());
        let eps64 = widen_f32_bits(eps.to_bits());
        let rstd64 = rsqrt_newton_f64_bits(add_f64_bits(var64, eps64));
        let wv = w.map_or(1.0f32, |w| w[ch]);
        let bv = b.map_or(0.0f32, |b| b[ch]);
        for batch in 0..n {
            for sp in 0..spatial {
                let i = batch * spatial + sp;
                let idx = channel_index(i, ch, c, spatial);
                let xv = widen_f32_bits(x[idx].to_bits());
                let dev = sub_f64_bits(xv, mean64);
                let xhat64 = mul_f64_bits(dev, rstd64);
                let xhat_bits = narrow_f64_bits(xhat64);
                out[idx] = f32::from_bits(apply_affine_soft_f64(xhat_bits, wv, bv));
            }
        }
    }
    out
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
        assert!(matches!(err, BatchNormPrepareError::InvalidShape { .. }));
    }

    #[test]
    fn validate_batch_norm_launch_rejects_weight_len_mismatch() {
        let err =
            validate_batch_norm_launch(2, 3, 4, 24, Some(2), None, None, None, 1e-5).unwrap_err();
        assert!(matches!(err, BatchNormPrepareError::InvalidShape { .. }));
    }

    #[test]
    fn validate_batch_norm_launch_rejects_stats_len_mismatch() {
        let err = validate_batch_norm_launch(2, 3, 4, 24, None, None, Some(2), Some(3), 1e-5)
            .unwrap_err();
        assert!(matches!(err, BatchNormPrepareError::InvalidShape { .. }));
    }

    #[test]
    fn validate_batch_norm_launch_rejects_non_finite_eps() {
        for eps in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0f32] {
            let err =
                validate_batch_norm_launch(2, 3, 4, 24, None, None, None, None, eps).unwrap_err();
            assert!(matches!(err, BatchNormPrepareError::InvalidShape { .. }));
        }
    }

    #[test]
    fn validate_batch_norm_launch_rejects_degenerate_channel_count_allocation() {
        // `n=0, c=usize::MAX` は `n*c*spatial` の積自体は 0 へ潰れるが、
        // `mean`／`var`（長さ `c`）確保時の capacity overflow panic を
        // 防ぐため事前に拒否する（PR #1874 codex-review P1 是正と同じ
        // 対策の Metal 側複製）。
        let err = validate_batch_norm_launch(0, usize::MAX, 1, 0, None, None, None, None, 1e-5)
            .unwrap_err();
        assert!(matches!(err, BatchNormPrepareError::InvalidShape { .. }));
    }

    #[test]
    fn validate_batch_norm_launch_rejects_u32_overflow() {
        let big = (u32::MAX as usize) + 1;
        let err =
            validate_batch_norm_launch(big, 1, 1, big, None, None, None, None, 1e-5).unwrap_err();
        assert!(matches!(
            err,
            BatchNormPrepareError::SizeLimitExceeded { .. }
        ));
    }

    #[test]
    fn channel_index_matches_cpu_reference_layout() {
        // `backend-cpu::batch_norm::channel_index`（非公開）と同じ
        // レイアウト式であることを手計算で固定する: n=2, c=3, spatial=4
        // の x[batch, ch, sp] は batch*(c*spatial) + ch*spatial + sp。
        assert_eq!(channel_index(0, 0, 3, 4), 0);
        assert_eq!(channel_index(4, 1, 3, 4), 12 + 4); // batch=1,sp=0,ch=1
        assert_eq!(channel_index(5, 2, 3, 4), 12 + 8 + 1); // batch=1,sp=1,ch=2
    }

    #[test]
    fn m_f64_bits_roundtrips_to_exact_f64() {
        for m in [0usize, 1, 32, 1_000_000, u32::MAX as usize] {
            let (hi, lo) = m_f64_bits(m);
            let bits = (u64::from(hi) << 32) | u64::from(lo);
            assert_eq!(f64::from_bits(bits), m as f64);
        }
    }

    /// [`batch_norm_train_host_model`]／[`batch_norm_infer_host_model`]
    /// を `fandhe_ai_backend_cpu::{run_batch_norm_train_f32,
    /// run_batch_norm_infer_f32}`（CPU 参照実装。dev-dependency）と
    /// REQ-2 統一複合判定で突き合わせる（モジュール doc comment
    /// 「なぜこれが Metal 側の正しさの根拠になるか」参照）。
    #[test]
    fn train_host_model_matches_cpu_reference_req2() {
        use fandhe_ai_backend_cpu::parity::assert_parity;

        let n = 4usize;
        let c = 3usize;
        let spatial = 2usize;
        let x: Vec<f32> = (0..(n * c * spatial))
            .map(|i| (i as f32 - 12.0) * 0.37)
            .collect();
        let w = vec![1.5f32, -0.8, 1.0];
        let b = vec![0.1f32, -0.2, 0.3];
        let eps = 1e-5f32;

        let model = batch_norm_train_host_model(&x, Some(&w), Some(&b), eps, n, c, spatial);
        let cpu = fandhe_ai_backend_cpu::run_batch_norm_train_f32(
            &x,
            Some(&w),
            Some(&b),
            eps,
            n,
            c,
            spatial,
        )
        .expect("valid shape");

        assert_parity(
            "batch_norm_train_host_model (soft-f64) vs CPU reference",
            &model.out,
            &cpu.out,
        );
        assert_parity(
            "batch_norm_train_host_model mean (soft-f64) vs CPU reference",
            &model.mean,
            &cpu.mean,
        );
        assert_parity(
            "batch_norm_train_host_model var (soft-f64) vs CPU reference",
            &model.var,
            &cpu.var,
        );
    }

    #[test]
    fn infer_host_model_matches_cpu_reference_req2() {
        use fandhe_ai_backend_cpu::parity::assert_parity;

        let n = 4usize;
        let c = 3usize;
        let spatial = 2usize;
        let x: Vec<f32> = (0..(n * c * spatial))
            .map(|i| (i as f32 - 12.0) * 0.37)
            .collect();
        let mean = vec![0.2f32, -0.1, 0.5];
        let var = vec![1.2f32, 0.8, 2.0];
        let w = vec![1.5f32, -0.8, 1.0];
        let b = vec![0.1f32, -0.2, 0.3];
        let eps = 1e-5f32;

        let model_out =
            batch_norm_infer_host_model(&x, &mean, &var, Some(&w), Some(&b), eps, n, c, spatial);
        let cpu_out = fandhe_ai_backend_cpu::run_batch_norm_infer_f32(
            &x,
            &mean,
            &var,
            Some(&w),
            Some(&b),
            eps,
            n,
            c,
            spatial,
        )
        .expect("valid shape");

        assert_parity(
            "batch_norm_infer_host_model (soft-f64) vs CPU reference",
            &model_out,
            &cpu_out,
        );
    }

    #[test]
    fn train_host_model_handles_zero_axis_early_return() {
        let model = batch_norm_train_host_model(&[], None, None, 1e-5, 0, 3, 4);
        assert!(model.out.is_empty());
        assert_eq!(model.mean, vec![0.0f32; 3]);
        assert_eq!(model.var, vec![0.0f32; 3]);
    }

    #[test]
    fn infer_host_model_handles_zero_axis_early_return() {
        let out = batch_norm_infer_host_model(&[], &[], &[], None, None, 1e-5, 0, 3, 4);
        assert!(out.is_empty());
    }
}
