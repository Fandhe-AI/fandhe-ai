//! LayerNorm 順伝播カーネル（CPU 参照実装。`rayon` 行方向並列。
//! イシュー #1596）。
//!
//! `rmsnorm.rs`（#607）と同じモジュール構成方針を踏襲する:
//! [`run_layer_norm_f32`] を公開エントリとし、`ops.rs::CpuBackendOps::
//! layer_norm` から直接呼ばれる。RMSNorm と異なり `run_fused`（canonical
//! 融合プラン一致経路）への LayerNorm 一致経路は追加しない
//! （`docs/norm-ops-design.md`「LayerNorm は本エントリ経由でのみ到達
//! する」）ため、`rmsnorm.rs::run_rmsnorm_f32_raw` のような `_raw`
//! 二重入口は持たない。
//!
//! # 縮約精度契約（`.claude/rules/coding-rust.md`）
//!
//! 平均は要素を `f64` へ昇格してから逐次和し、分散は「二パス」
//! （`Σ(x−μ)²/N`。`E[x²]−μ²` は使わない）で `(x−μ)` を `f64` へ昇格
//! してから二乗・蓄積する。`rstd` へ代入する 1 回だけ `f32` へ
//! downcast する（`rmsnorm.rs::rmsnorm_row_scalar` と同じ縮約方式。
//! イシュー #1102 の一般契約をそのまま適用する）。
//!
//! # NEON 化について
//!
//! 本イシュー時点ではスカラー経路のみを実装する（`rmsnorm.rs` の
//! NEON `float64x2_t` 二乗和ベクトル化と同型の最適化は後続の性能
//! 課題として `docs/norm-ops-design.md` に記録し、本 PR のスコープ外
//! とする）。
//!
//! # 境界検査（REQ-8）
//!
//! 本ファイルは境界チェックを省略する `unsafe` intrinsics を使わない
//! （スライス添字アクセスはすべて Rust の境界検査を経る）ため、
//! REQ-8 の手動境界検査規約は該当しない。

use rayon::prelude::*;

use crate::elementwise::PARALLEL_THRESHOLD;

/// [`run_layer_norm_f32`] の型付きエラー（`rmsnorm::RmsNormError` と
/// 同じ「小さな enum」方針）。
#[non_exhaustive]
#[derive(Debug)]
pub enum LayerNormError {
    /// `rows * hidden` が overflow するか、`x.len()` と一致しない。
    InvalidShape { detail: String },
    /// `w` が指定されているが `w.len() != hidden`。
    WeightLenMismatch { hidden: usize, w_len: usize },
    /// `b` が指定されているが `b.len() != hidden`。
    BiasLenMismatch { hidden: usize, b_len: usize },
}

impl std::fmt::Display for LayerNormError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerNormError::InvalidShape { detail } => {
                write!(f, "layer_norm invalid shape: {detail}")
            }
            LayerNormError::WeightLenMismatch { hidden, w_len } => write!(
                f,
                "layer_norm weight length mismatch: hidden={hidden}, w.len()={w_len}"
            ),
            LayerNormError::BiasLenMismatch { hidden, b_len } => write!(
                f,
                "layer_norm bias length mismatch: hidden={hidden}, b.len()={b_len}"
            ),
        }
    }
}

impl std::error::Error for LayerNormError {}

/// 起動前 fail-closed 検証（`rmsnorm::validate_rmsnorm_launch` と同型。
/// OWASP A03・`.claude/rules/security.md`）: `rows * hidden ==
/// x.len()`（checked 乗算）・`w.len() == hidden`（`w` 指定時）・
/// `b.len() == hidden`（`b` 指定時）。
fn validate_layer_norm_launch(
    rows: usize,
    hidden: usize,
    x_len: usize,
    w_len: Option<usize>,
    b_len: Option<usize>,
) -> Result<(), LayerNormError> {
    let numel = rows
        .checked_mul(hidden)
        .ok_or_else(|| LayerNormError::InvalidShape {
            detail: format!("rows*hidden overflowed usize: rows={rows}, hidden={hidden}"),
        })?;
    if numel != x_len {
        return Err(LayerNormError::InvalidShape {
            detail: format!("x length mismatch: rows*hidden={numel}, x.len()={x_len}"),
        });
    }
    if let Some(wl) = w_len
        && wl != hidden
    {
        return Err(LayerNormError::WeightLenMismatch { hidden, w_len: wl });
    }
    if let Some(bl) = b_len
        && bl != hidden
    {
        return Err(LayerNormError::BiasLenMismatch { hidden, b_len: bl });
    }
    Ok(())
}

/// LayerNorm（`out = (x − mean(x)) · rsqrt(var(x) + eps) · w + b`。
/// 分散は biased ÷N。`w`／`b` はそれぞれ `None` の場合は対応する演算を
/// スキップ）。
///
/// `x` は `[rows, hidden]` の行優先 1 次元化済みスライス。
/// `rows == 0 || hidden == 0` は空出力を返す（`rmsnorm::
/// run_rmsnorm_f32_raw` と同じ早期 return 契約）。
pub fn run_layer_norm_f32(
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Result<Vec<f32>, LayerNormError> {
    validate_layer_norm_launch(
        rows,
        hidden,
        x.len(),
        w.map(|s| s.len()),
        b.map(|s| s.len()),
    )?;

    if rows == 0 || hidden == 0 {
        return Ok(Vec::new());
    }

    let mut out = vec![0.0f32; x.len()];
    let numel = rows * hidden;
    let inv_n = 1.0f64 / hidden as f64;

    if numel >= PARALLEL_THRESHOLD && rows >= 2 {
        out.par_chunks_mut(hidden)
            .zip(x.par_chunks(hidden))
            .for_each(|(out_row, in_row)| {
                layer_norm_row(in_row, w, b, eps, inv_n, out_row);
            });
    } else {
        for (out_row, in_row) in out.chunks_mut(hidden).zip(x.chunks(hidden)) {
            layer_norm_row(in_row, w, b, eps, inv_n, out_row);
        }
    }

    Ok(out)
}

/// 1 行分の LayerNorm を計算する（スカラーのみ。冒頭コメント参照）。
///
/// 平均は `f64` 逐次和、分散は「二パス」（`(x−μ)` を `f64` へ昇格して
/// から二乗し `f64::mul_add` で蓄積）。`rstd` へ代入する 1 回だけ
/// `f32` へ downcast する（`rmsnorm_row_scalar` と同じ縮約方式）。
fn layer_norm_row(
    row: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    inv_n: f64,
    out_row: &mut [f32],
) {
    let mut sum = 0.0f64;
    for &v in row {
        sum += v as f64;
    }
    let mean = sum * inv_n;
    let mut sq_acc = 0.0f64;
    for &v in row {
        let d = v as f64 - mean;
        sq_acc = d.mul_add(d, sq_acc);
    }
    let var = sq_acc * inv_n;
    let rstd = (1.0f64 / (var + eps as f64).sqrt()) as f32;
    let mean = mean as f32;

    match (w, b) {
        (Some(w), Some(b)) => {
            for (((o, &v), &wv), &bv) in out_row
                .iter_mut()
                .zip(row.iter())
                .zip(w.iter())
                .zip(b.iter())
            {
                *o = (v - mean) * rstd * wv + bv;
            }
        }
        (Some(w), None) => {
            for ((o, &v), &wv) in out_row.iter_mut().zip(row.iter()).zip(w.iter()) {
                *o = (v - mean) * rstd * wv;
            }
        }
        (None, Some(b)) => {
            for ((o, &v), &bv) in out_row.iter_mut().zip(row.iter()).zip(b.iter()) {
                *o = (v - mean) * rstd + bv;
            }
        }
        (None, None) => {
            for (o, &v) in out_row.iter_mut().zip(row.iter()) {
                *o = (v - mean) * rstd;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_layer_norm_launch_accepts_matching_dims() {
        assert!(validate_layer_norm_launch(3, 8, 24, Some(8), Some(8)).is_ok());
        assert!(validate_layer_norm_launch(3, 8, 24, None, None).is_ok());
    }

    #[test]
    fn validate_layer_norm_launch_rejects_x_len_mismatch() {
        let err = validate_layer_norm_launch(3, 8, 23, None, None).unwrap_err();
        assert!(matches!(err, LayerNormError::InvalidShape { .. }));
    }

    #[test]
    fn validate_layer_norm_launch_rejects_w_len_mismatch() {
        let err = validate_layer_norm_launch(3, 8, 24, Some(7), None).unwrap_err();
        assert!(matches!(err, LayerNormError::WeightLenMismatch { .. }));
    }

    #[test]
    fn validate_layer_norm_launch_rejects_b_len_mismatch() {
        let err = validate_layer_norm_launch(3, 8, 24, None, Some(7)).unwrap_err();
        assert!(matches!(err, LayerNormError::BiasLenMismatch { .. }));
    }

    #[test]
    fn run_layer_norm_f32_empty_rows_or_hidden_returns_empty() {
        assert_eq!(
            run_layer_norm_f32(&[], None, None, 1e-5, 0, 8).unwrap(),
            Vec::<f32>::new()
        );
        assert_eq!(
            run_layer_norm_f32(&[], None, None, 1e-5, 3, 0).unwrap(),
            Vec::<f32>::new()
        );
    }

    #[test]
    fn run_layer_norm_f32_basic_no_affine() {
        // x=[1,2,3,4] -> mean=2.5, var=Sigma(x-2.5)^2/4=1.25
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = run_layer_norm_f32(&x, None, None, 0.0, 1, 4).unwrap();
        let rstd = 1.0f32 / 1.25f32.sqrt();
        let expected = [-1.5 * rstd, -0.5 * rstd, 0.5 * rstd, 1.5 * rstd];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "o={o} e={e}");
        }
    }

    #[test]
    fn run_layer_norm_f32_applies_weight_and_bias() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = [2.0f32, 1.0, 1.0, 0.5];
        let b = [1.0f32, 0.0, -1.0, 2.0];
        let out = run_layer_norm_f32(&x, Some(&w), Some(&b), 0.0, 1, 4).unwrap();
        let rstd = 1.0f32 / 1.25f32.sqrt();
        let xhat = [-1.5 * rstd, -0.5 * rstd, 0.5 * rstd, 1.5 * rstd];
        let expected = [
            xhat[0] * 2.0 + 1.0,
            xhat[1] * 1.0,
            xhat[2] * 1.0 - 1.0,
            xhat[3] * 0.5 + 2.0,
        ];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "o={o} e={e}");
        }
    }

    #[test]
    fn run_layer_norm_f32_weight_only() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = [2.0f32, 1.0, 1.0, 0.5];
        let out_wb = run_layer_norm_f32(&x, Some(&w), None, 0.0, 1, 4).unwrap();
        let zero_bias = [0.0f32; 4];
        let out_ref = run_layer_norm_f32(&x, Some(&w), Some(&zero_bias), 0.0, 1, 4).unwrap();
        assert_eq!(out_wb, out_ref);
    }

    #[test]
    fn run_layer_norm_f32_bias_only() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, -1.0, 0.5, 2.0];
        let out_b = run_layer_norm_f32(&x, None, Some(&b), 0.0, 1, 4).unwrap();
        let ones = [1.0f32; 4];
        let out_ref = run_layer_norm_f32(&x, Some(&ones), Some(&b), 0.0, 1, 4).unwrap();
        assert_eq!(out_b, out_ref);
    }

    /// NaN 伝播（`rmsnorm.rs` の同名テストと同じ意味論・理由）。
    #[test]
    fn run_layer_norm_f32_propagates_nan_for_row_with_nan_element() {
        for hidden in [4usize, 5] {
            let mut x = vec![1.0f32; hidden];
            x[0] = f32::NAN;
            let out = run_layer_norm_f32(&x, None, None, 1e-5, 1, hidden).unwrap();
            assert!(
                out.iter().all(|v| v.is_nan()),
                "hidden={hidden}: NaN 要素を含む行の出力が NaN へ伝播していない: {out:?}"
            );
        }
    }

    /// 極端な大きさ（`f64` 昇格前に二乗すると overflow しうる規模）でも
    /// 出力が有限であることを確認する（イシュー #1102 契約の回帰）。
    #[test]
    fn run_layer_norm_f32_extreme_scale_does_not_overflow() {
        let x = vec![2e20f32, -2e20, 2e20, -2e20];
        let out = run_layer_norm_f32(&x, None, None, 1e-5, 1, 4).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
    }

    #[test]
    fn run_layer_norm_f32_multi_row_parallel_path_matches_serial() {
        // rayon 並列閾値（PARALLEL_THRESHOLD）を跨いだ場合の行間独立性を
        // 確認する（rows>=2 のみ並列化されるため大きめの hidden で
        // numel >= PARALLEL_THRESHOLD を満たす）。
        let hidden = 1 << 14; // 16384
        let rows = 4;
        let x: Vec<f32> = (0..rows * hidden).map(|i| (i % 97) as f32 * 0.1).collect();
        let out_par = run_layer_norm_f32(&x, None, None, 1e-5, rows, hidden).unwrap();
        // 1 行ずつ処理した結果と突合（`chunks` 経由の逐次実装を直接呼ぶ
        // 代わりに rows=1 を rows 回呼んで連結する）。
        let mut out_serial = Vec::with_capacity(x.len());
        for chunk in x.chunks(hidden) {
            out_serial.extend(run_layer_norm_f32(chunk, None, None, 1e-5, 1, hidden).unwrap());
        }
        assert_eq!(out_par, out_serial);
    }
}
