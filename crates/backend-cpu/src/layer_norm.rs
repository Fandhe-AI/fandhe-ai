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
//! 平均・分散とも要素を `f64` へ昇格し、**GPU（CUDA
//! `kernels_layer_norm.rs`／Metal `layer_norm.metal`）の warp／
//! simdgroup butterfly 縮約と同一の加算順序**（`warp_reduce_f64`。
//! 32 レーンのストライドアクセス + offset 16→8→4→2→1 の木構造縮約）
//! で蓄積する（単純な先頭からの逐次和ではない。PR #1671 codex-review
//! P1 是正・イシュー #1596）。分散は「二パス」（`Σ(x−μ)²/N`。
//! `E[x²]−μ²` は使わない）で `(x−μ)` を `f64` へ昇格してから
//! `f64::mul_add` で融合二乗・蓄積する。`rstd` へ代入する 1 回だけ
//! `f32` へ downcast する。
//!
//! **加算順序を GPU 側へ揃える理由**: 相殺を含む入力
//! （例 `x=[1e30, 1, -1e30, 0]`）では加算順序で結果が変わりうる
//! （単純な逐次和は `0`、butterfly 縮約は数学的に正しい `1` を
//! 返す）。GPU 側を逐次和へ合わせるには縮約を直列化する必要があり
//! 性能面で著しく不利なため、CPU 参照実装
//! （`autodiff::eval::row_ln_stats` も同型で追従）側を GPU の
//! 決定的な縮約順序へ合わせた（`warp_reduce_f64` doc comment 参照）。
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
    /// `eps` が有限でないか負（`backend-cuda::layer_norm::
    /// validate_layer_norm_launch`／`row_kernel::validate_row_kernel_launch`
    /// と同型の検証。CPU 側は本 PR まで欠落していた〈codex-review 指摘〉）。
    InvalidEps { eps: f32 },
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
            LayerNormError::InvalidEps { eps } => {
                write!(
                    f,
                    "layer_norm eps must be finite and non-negative: eps={eps}"
                )
            }
        }
    }
}

impl std::error::Error for LayerNormError {}

/// 起動前 fail-closed 検証（`rmsnorm::validate_rmsnorm_launch`・
/// `backend-cuda::layer_norm::validate_layer_norm_launch` と同型。
/// OWASP A03・`.claude/rules/security.md`）: `eps` が有限かつ非負
/// （`is_finite() && eps >= 0.0`）・`rows * hidden == x.len()`
/// （checked 乗算）・`w.len() == hidden`（`w` 指定時）・
/// `b.len() == hidden`（`b` 指定時）。
fn validate_layer_norm_launch(
    rows: usize,
    hidden: usize,
    x_len: usize,
    w_len: Option<usize>,
    b_len: Option<usize>,
    eps: f32,
) -> Result<(), LayerNormError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(LayerNormError::InvalidEps { eps });
    }
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
        eps,
    )?;

    if rows == 0 || hidden == 0 {
        return Ok(Vec::new());
    }

    let mut out = vec![0.0f32; x.len()];
    let numel = rows * hidden;

    if numel >= PARALLEL_THRESHOLD && rows >= 2 {
        out.par_chunks_mut(hidden)
            .zip(x.par_chunks(hidden))
            .for_each(|(out_row, in_row)| {
                layer_norm_row(in_row, w, b, eps, hidden, out_row);
            });
    } else {
        for (out_row, in_row) in out.chunks_mut(hidden).zip(x.chunks(hidden)) {
            layer_norm_row(in_row, w, b, eps, hidden, out_row);
        }
    }

    Ok(out)
}

/// GPU（CUDA `kernels_layer_norm.rs`::`__shfl_xor_sync`／Metal
/// `layer_norm.metal`::`simd_shuffle_xor`）の warp／simdgroup 縮約と
/// **同一の演算順序**を CPU 上で再現する（PR #1671 codex-review P1
/// 指摘・イシュー #1596 是正）。
///
/// # 背景（なぜ縮約順序を揃える必要があるか）
///
/// GPU 側は 1 CTA/threadgroup = 32 レーンが 1 行を担当し、各レーンが
/// `idx = lane, lane+32, lane+64, …`（ストライド 32）の要素を**逐次**
/// 蓄積したのち、offset `16→8→4→2→1` の 5 段 butterfly（xor shuffle）
/// で 32 レーン分を木構造に縮約する。これは CPU の単純な「先頭から
/// 末尾まで 1 個ずつ逐次加算」とは**加算順序が異なる**。加算は
/// 結合則を厳密には満たさないため、相殺を含む入力（例
/// `x=[1e30, 1, -1e30, 0]`）では両者の結果が乖離しうる（実測:
/// 逐次和は 0／butterfly 縮約は 1 になり、`mean` が 0 と 0.25 という
/// REQ-2 統一複合判定〈相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満〉
/// を超える差になる）。
///
/// **本関数は CPU 側を GPU の縮約順序へ合わせる**（逆ではない。
/// butterfly 縮約は数学的に正しい値〈上記例では 1〉を返す一方、
/// 単純な逐次和は `1e30` に埋もれて `+1` を取りこぼす／`0` へ丸め
/// 落ちる側であり、GPU 側を「正しさが劣る」逐次和へ合わせると
/// GPU 側の縮約全体を直列化する必要が生じ〈1 レーンが `hidden` 回の
/// soft-f64 演算を担う〉性能面で著しく不利になる。既存の GPU
/// カーネル実装〈CUDA／Metal〉は変更せず、CPU 参照実装
/// 〈`autodiff::eval::row_ln_stats` も同型で追従〉側をこの決定的な
/// 縮約順序へ合わせることで両者を一致させる）。
///
/// `contribute(idx, lane_acc)` はレーンの部分和へ要素 `idx` の寄与を
/// 足し込んだ新しい部分和を返す（平均パスは単純加算、分散パスは
/// `(x-mean)^2` を `f64::mul_add` で融合加算——GPU 側 CUDA の
/// `fma(d, d, sq_acc)` と同じ単一丸め）。レーン間の合流
/// （butterfly 段）は GPU 側と同じ単純加算で行う（CUDA
/// `sq_acc += __shfl_xor_sync(...)`／Metal `ln_f64_add` と同型）。
fn warp_reduce_f64(hidden: usize, mut contribute: impl FnMut(usize, f64) -> f64) -> f64 {
    const LANES: usize = 32;
    let mut lanes = [0.0f64; LANES];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut idx = lane;
        while idx < hidden {
            *slot = contribute(idx, *slot);
            idx += LANES;
        }
    }
    // butterfly 縮約（offset 16→8→4→2→1）。各段は「更新前」の全レーン値
    // を参照する必要がある（GPU の `simd_shuffle_xor`／`__shfl_xor_sync`
    // が段内で全レーン同時にシャッフルするのと同じ意味論）ため、段ごとに
    // スナップショットを取ってから書き戻す（in-place 更新は不可）。
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

/// 1 行分の LayerNorm を計算する（スカラーのみ。冒頭コメント参照）。
///
/// 平均・分散とも GPU の warp/simdgroup butterfly 縮約と同一順序
/// （`warp_reduce_f64` 参照）で `f64` 蓄積する。**`mean`／`rstd` は
/// いずれも `f64` のまま `x̂ = (x − mean) · rstd` の偏差計算まで
/// 保持し**、`x̂` を出力へ書き込む直前の 1 回だけ `f32` へ downcast
/// する（codex-review 指摘: `mean` を偏差計算前に `f32` へ丸めると
/// `mean` の丸め誤差がそのまま `x̂` へ伝播し、`x` の値域が `f32` の
/// 仮数精度限界〈例: 2^24 付近〉に達する入力で顕著な誤差を生む。
/// `rmsnorm_row_scalar` の `rstd` 単体丸めとは異なり、LayerNorm は
/// `mean` 減算があるため両方を高精度に保つ必要がある）。`mean`／`var`
/// は事前丸めした逆数 `1/hidden` との積ではなく `hidden` による直接
/// 除算で求める（codex-review 指摘: `x=[1e30f32;49]` のような一様行で
/// `sum * (1/hidden)` は 2 回の丸めが複合し、本来 0 の偏差が巨大な
/// 非ゼロ値になる。`eval::row_ln_stats` と同じ契約）。
///
/// affine（`x̂·w+b`）は CUDA カーネル（`kernels_layer_norm.rs`。
/// `xhat * wv + bv` が nvcc の既定 FMA contraction で `fmaf` 相当に
/// 融合される）と揃えるため、`f32::mul_add` で明示的に融合する
/// （`.claude/rules/coding-rust.md` の FMA 契約統一。codex-review
/// 指摘）。
fn layer_norm_row(
    row: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    hidden: usize,
    out_row: &mut [f32],
) {
    let n = hidden as f64;
    let sum = warp_reduce_f64(hidden, |idx, acc| acc + row[idx] as f64);
    let mean = sum / n;
    let sq_acc = warp_reduce_f64(hidden, |idx, acc| {
        let d = row[idx] as f64 - mean;
        d.mul_add(d, acc)
    });
    let var = sq_acc / n;
    let rstd = 1.0f64 / (var + eps as f64).sqrt();

    match (w, b) {
        (Some(w), Some(b)) => {
            for (((o, &v), &wv), &bv) in out_row
                .iter_mut()
                .zip(row.iter())
                .zip(w.iter())
                .zip(b.iter())
            {
                let xhat = ((v as f64 - mean) * rstd) as f32;
                *o = xhat.mul_add(wv, bv);
            }
        }
        (Some(w), None) => {
            for ((o, &v), &wv) in out_row.iter_mut().zip(row.iter()).zip(w.iter()) {
                let xhat = ((v as f64 - mean) * rstd) as f32;
                *o = xhat * wv;
            }
        }
        (None, Some(b)) => {
            for ((o, &v), &bv) in out_row.iter_mut().zip(row.iter()).zip(b.iter()) {
                let xhat = ((v as f64 - mean) * rstd) as f32;
                *o = xhat + bv;
            }
        }
        (None, None) => {
            for (o, &v) in out_row.iter_mut().zip(row.iter()) {
                *o = ((v as f64 - mean) * rstd) as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_layer_norm_launch_accepts_matching_dims() {
        assert!(validate_layer_norm_launch(3, 8, 24, Some(8), Some(8), 1e-5).is_ok());
        assert!(validate_layer_norm_launch(3, 8, 24, None, None, 1e-5).is_ok());
    }

    #[test]
    fn validate_layer_norm_launch_rejects_x_len_mismatch() {
        let err = validate_layer_norm_launch(3, 8, 23, None, None, 1e-5).unwrap_err();
        assert!(matches!(err, LayerNormError::InvalidShape { .. }));
    }

    #[test]
    fn validate_layer_norm_launch_rejects_w_len_mismatch() {
        let err = validate_layer_norm_launch(3, 8, 24, Some(7), None, 1e-5).unwrap_err();
        assert!(matches!(err, LayerNormError::WeightLenMismatch { .. }));
    }

    #[test]
    fn validate_layer_norm_launch_rejects_b_len_mismatch() {
        let err = validate_layer_norm_launch(3, 8, 24, None, Some(7), 1e-5).unwrap_err();
        assert!(matches!(err, LayerNormError::BiasLenMismatch { .. }));
    }

    #[test]
    fn validate_layer_norm_launch_rejects_negative_eps() {
        let err = validate_layer_norm_launch(3, 8, 24, None, None, -1.0).unwrap_err();
        assert!(matches!(err, LayerNormError::InvalidEps { .. }));
    }

    #[test]
    fn validate_layer_norm_launch_rejects_non_finite_eps() {
        for eps in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = validate_layer_norm_launch(3, 8, 24, None, None, eps).unwrap_err();
            assert!(matches!(err, LayerNormError::InvalidEps { .. }));
        }
    }

    #[test]
    fn validate_layer_norm_launch_accepts_zero_eps() {
        assert!(validate_layer_norm_launch(3, 8, 24, None, None, 0.0).is_ok());
    }

    /// CPU の `run_layer_norm_f32` は起動前検証を経由するため、不正な
    /// `eps` は `NaN` を返さず `Err` になる（codex-review 指摘: CUDA／
    /// Metal は既に拒否するが CPU は本 PR まで受理し `NaN` 出力を
    /// 生んでいた）。
    #[test]
    fn run_layer_norm_f32_rejects_invalid_eps() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let err = run_layer_norm_f32(&x, None, None, -1.0, 1, 4).unwrap_err();
        assert!(matches!(err, LayerNormError::InvalidEps { .. }));
        let err = run_layer_norm_f32(&x, None, None, f32::NAN, 1, 4).unwrap_err();
        assert!(matches!(err, LayerNormError::InvalidEps { .. }));
    }

    /// codex-review 指摘の再現ケース: `mean` を偏差計算前に `f32` へ
    /// 丸めると `x=[16777216, 16777218]`（`2^24` 近傍で `f32` の ULP が
    /// `2` になる領域）で `mean=16777217.0` が `16777216.0` へ丸められ、
    /// 出力が期待値 `[-1, 1]`（`eps` 無視できる規模）から大きく乖離する
    /// （偏差計算前丸めの場合 `[0, 1.99999]` になる）。`f64` のまま偏差
    /// 計算まで保持すれば `[-1, 1]` に一致する。
    #[test]
    fn run_layer_norm_f32_preserves_mean_precision_near_f32_epsilon_boundary() {
        let x = vec![16777216.0f32, 16777218.0];
        let out = run_layer_norm_f32(&x, None, None, 1e-5, 1, 2).unwrap();
        let rstd = 1.0f64 / (1.0f64 + 1e-5f64).sqrt();
        let expected = [(-rstd) as f32, rstd as f32];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-4, "o={o} e={e}");
        }
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

    /// PR #1671 codex-review P1 指摘（イシュー #1596）の反例そのものを
    /// 再現する回帰テスト: `x=[1e30, 1, -1e30, 0]`（相殺を含む行）で
    /// GPU の warp/simdgroup butterfly 縮約は `mean=0.25`（数学的に
    /// 正しい値）を返すが、単純な先頭からの逐次和は `mean=0`（`+1` を
    /// 取りこぼす）を返す。`warp_reduce_f64` 導入前は本関数が後者
    /// （`mean=0`）を返しており、`out[3] == 0.0` になっていた。
    /// 是正後は GPU 側の縮約順序に一致し `out[3] ≈ -sqrt(2)/4` になる
    /// （指摘コメント記載の実測値 `-0.353553` と一致）。
    #[test]
    fn run_layer_norm_f32_matches_gpu_butterfly_order_on_cancelling_row() {
        let x = vec![1e30f32, 1.0, -1e30, 0.0];
        let w = vec![1.0f32, 1.0, 1.0, 1e30];
        let out = run_layer_norm_f32(&x, Some(&w), None, 1e-5, 1, 4).unwrap();

        let expected_out3 = -std::f64::consts::SQRT_2 / 4.0;
        assert!(
            (out[3] as f64 - expected_out3).abs() < 1e-3,
            "out[3]={} expected~={expected_out3} (逐次和のままなら 0.0 になる回帰)",
            out[3]
        );
        assert_ne!(
            out[3], 0.0,
            "単純な逐次和（是正前の実装）へ後退している可能性がある"
        );
    }

    /// PR #1671 codex-review 2 回目の P1 指摘（イシュー #1596）の反例を
    /// CPU 参照実装（本関数。`affine` は行末の `xhat.mul_add(wv, bv)`。
    /// ハードウェア FMA）で再現する Linux 実行可能な回帰テスト
    /// （Metal 実機依存版は `crates/backend-metal/tests/
    /// layer_norm_parity.rs::
    /// layer_norm_affine_overflow_boundary_matches_hardware_fma_not_double_rounding`）。
    ///
    /// `x` は平均 0・分散 256（`rstd=1/16`）となるよう構成した行で、
    /// `xhat[0] = 31/16` が丸め無しで厳密に成立する。
    /// `weight[0]=f32::from_bits(0x7f042108)`・`bias[0]=-1` との単一丸め
    /// FMA（`f32::mul_add`）は有限の `f32::MAX` を返す（`f64` へ丸めて
    /// から `f32` へ narrow する素朴な二段階丸めだと `+inf` になる、
    /// という指摘の反例そのもの。Metal 側の是正〈round-to-odd 丸め加算
    /// `ln_f64_add_ro`〉が本 CPU 参照実装〈ハードウェア FMA〉と同じ
    /// 値を返すことを期待する契約の CPU 側担保）。
    #[test]
    fn affine_overflow_boundary_row_returns_f32_max_not_inf() {
        let x = [
            31.0f32, -31.0, 31.0, -31.0, 11.0, -11.0, 2.0, -2.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
        ];
        let mut w = [1.0f32; 16];
        w[0] = f32::from_bits(0x7f042108);
        let mut b = [0.0f32; 16];
        b[0] = -1.0;

        let out = run_layer_norm_f32(&x, Some(&w), Some(&b), 0.0, 1, 16)
            .expect("run_layer_norm_f32 must succeed");
        assert!(
            out[0].is_finite(),
            "expected out[0] to be finite (f32::MAX), got {:?}",
            out[0]
        );
        assert_eq!(
            out[0].to_bits(),
            f32::MAX.to_bits(),
            "out[0] は f32::MAX と bit 完全一致するはず（ハードウェア FMA \
             f32::mul_add の結果）: got={:?}",
            out[0]
        );
    }
}
