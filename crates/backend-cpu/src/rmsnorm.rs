//! 融合 RMSNorm 順伝播カーネル（CPU 参照実装。NEON + `rayon` 行方向並列。
//! イシュー #607）。
//!
//! `backend-cuda::rmsnorm`（#592）・`backend-metal::rmsnorm`（#604）と同じ
//! モジュール構成方針を踏襲する: [`run_rmsnorm_f32`]（標準 RMSNorm。`mean`
//! 化 + `eps` + 任意 `weight`）を公開エントリとし、`ops.rs::CpuBackendOps::
//! run_fused` は canonical 融合プラン（`mean` 化・`eps`・`weight` を含まない
//! `x * rsqrt(sum(x^2))`）専用の内部エントリ [`run_rmsnorm_f32_raw`] を
//! `inv_n = 1.0` で直接呼ぶ（[`match_rmsnorm_plan`] が一致判定するプラン
//! 形状は CUDA `rmsnorm.rs::match_rmsnorm_plan` と 1:1 対応）。
//!
//! # NEON / スカラー二重経路
//!
//! aarch64（Metal 実機の Apple Silicon・DGX Spark GB10 Grace CPU いずれでも
//! 常時利用可能なベースライン ISA）では `std::arch::aarch64` intrinsics を
//! 使い、他アーキテクチャ（x86_64 開発環境・CI）はスカラー経路を使う
//! （`gemm_blis/microkernel/neon.rs` 冒頭コメントの既存整理を踏襲。
//! `target_feature` によるコンパイル時分岐は不要）。スカラー経路は
//! `pub(crate)` とし、非 aarch64 では常時コンパイル、aarch64 では本ファイル
//! 内 `#[cfg(test)] mod tests` の NEON との A/B 同値テストからのみ参照する
//! （`cfg(any(not(aarch64), test))`。dead_code 判定はコンパイル単位ごとの
//! ため、テスト専用参照のみでは aarch64 の非テストビルドで dead_code に
//! なる点に注意）。
//!
//! # FMA 契約（REQ-2）
//!
//! 二乗和の累積は `f32::mul_add(v, v, acc)`（NEON は `vfmaq_f32`）を用いる
//! （GPU 側 `fmaf`／`simdgroup_multiply_accumulate` の既定 FMA 契約と揃える。
//! `.claude/rules/coding-rust.md`）。
//!
//! # 境界検査（REQ-8）
//!
//! NEON 経路は `chunks_exact(4)` で検証済みの full チャンクのみを `unsafe`
//! ロード対象とし、端数（`hidden % 4 != 0`）はスカラーで処理する
//! （`get_unchecked` は使わない）。

use rayon::prelude::*;

use crate::elementwise::PARALLEL_THRESHOLD;

/// [`run_rmsnorm_f32`]／[`run_rmsnorm_f32_raw`] の型付きエラー
/// （`reduction::ReduceError` と同じ「小さな enum」方針）。
#[non_exhaustive]
#[derive(Debug)]
pub enum RmsNormError {
    /// `rows * hidden` が overflow するか、`x.len()` と一致しない。
    InvalidShape { detail: String },
    /// `w` が指定されているが `w.len() != hidden`。
    WeightLenMismatch { hidden: usize, w_len: usize },
}

impl std::fmt::Display for RmsNormError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RmsNormError::InvalidShape { detail } => write!(f, "rmsnorm invalid shape: {detail}"),
            RmsNormError::WeightLenMismatch { hidden, w_len } => write!(
                f,
                "rmsnorm weight length mismatch: hidden={hidden}, w.len()={w_len}"
            ),
        }
    }
}

impl std::error::Error for RmsNormError {}

/// 起動前 fail-closed 検証（`backend-cuda::rmsnorm::validate_rmsnorm_launch`
/// と同型。OWASP A03・`.claude/rules/security.md`）: `rows * hidden ==
/// x.len()`（checked 乗算）・`w.len() == hidden`（`w` 指定時のみ）。
fn validate_rmsnorm_launch(
    rows: usize,
    hidden: usize,
    x_len: usize,
    w_len: Option<usize>,
) -> Result<(), RmsNormError> {
    let numel = rows
        .checked_mul(hidden)
        .ok_or_else(|| RmsNormError::InvalidShape {
            detail: format!("rows*hidden overflowed usize: rows={rows}, hidden={hidden}"),
        })?;
    if numel != x_len {
        return Err(RmsNormError::InvalidShape {
            detail: format!("x length mismatch: rows*hidden={numel}, x.len()={x_len}"),
        });
    }
    if let Some(wl) = w_len
        && wl != hidden
    {
        return Err(RmsNormError::WeightLenMismatch { hidden, w_len: wl });
    }
    Ok(())
}

/// 標準 RMSNorm（`mean` 正規化あり）: `out = x * rsqrt(mean(x^2, axis=-1) +
/// eps) * w`（`w` が `None` の場合は乗算をスキップ）。
///
/// `x` は `[rows, hidden]` の行優先 1 次元化済みスライス。`inv_n = 1/hidden`
/// を内部導出し [`run_rmsnorm_f32_raw`] へ委譲する（`hidden == 0` の場合は
/// `inv_n = 1.0`〈使われない〉として渡し、ゼロ除算を避ける）。
pub fn run_rmsnorm_f32(
    x: &[f32],
    w: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Result<Vec<f32>, RmsNormError> {
    let inv_n = if hidden == 0 {
        1.0f32
    } else {
        1.0f32 / hidden as f32
    };
    run_rmsnorm_f32_raw(x, w, eps, inv_n, rows, hidden)
}

/// `out = x * rsqrt(sum(x^2, axis=-1) * inv_n + eps) * w`（`w` が `None`
/// の場合は乗算をスキップ）を実行する内部エントリ。`inv_n` を呼び出し元が
/// 明示するため、標準 RMSNorm（[`run_rmsnorm_f32`]・`inv_n = 1/hidden`）と
/// canonical 融合プラン（`ops.rs::CpuBackendOps::run_fused`・`inv_n = 1.0`。
/// `mean` 化しない）の両方の起動元になれる
/// （`backend-cuda::rmsnorm::CudaRmsNorm::run_rmsnorm_f32_raw` と同型の
/// 二重入口構成）。
pub(crate) fn run_rmsnorm_f32_raw(
    x: &[f32],
    w: Option<&[f32]>,
    eps: f32,
    inv_n: f32,
    rows: usize,
    hidden: usize,
) -> Result<Vec<f32>, RmsNormError> {
    validate_rmsnorm_launch(rows, hidden, x.len(), w.map(|s| s.len()))?;

    if rows == 0 || hidden == 0 {
        return Ok(Vec::new());
    }

    let mut out = vec![0.0f32; x.len()];
    let numel = rows * hidden;

    if numel >= PARALLEL_THRESHOLD && rows >= 2 {
        out.par_chunks_mut(hidden)
            .zip(x.par_chunks(hidden))
            .for_each(|(out_row, in_row)| {
                rmsnorm_row(in_row, w, eps, inv_n, out_row);
            });
    } else {
        for (out_row, in_row) in out.chunks_mut(hidden).zip(x.chunks(hidden)) {
            rmsnorm_row(in_row, w, eps, inv_n, out_row);
        }
    }

    Ok(out)
}

/// 1 行分の RMSNorm を計算する（NEON / スカラーの経路選択点）。
fn rmsnorm_row(row: &[f32], w: Option<&[f32]>, eps: f32, inv_n: f32, out_row: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        rmsnorm_row_neon(row, w, eps, inv_n, out_row);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        rmsnorm_row_scalar(row, w, eps, inv_n, out_row);
    }
}

/// スカラー経路（`pub(crate)`: 非 aarch64 では [`rmsnorm_row`] の通常経路
/// として使う。aarch64 では本体からは呼ばれないが、本ファイル内
/// `#[cfg(test)] mod tests` の `neon_matches_scalar_various_hidden`
/// （NEON/スカラー A/B 同値テスト）が参照実装として直接呼ぶため、
/// `cfg(any(not(aarch64), test))` で当該テスト構成時のみ aarch64 でも
/// 残す（さもないと aarch64 の非テストビルドで dead_code になる。
/// dead_code 判定はコンパイル単位ごとであり、テスト専用の呼び出しは
/// 通常の `lib` ターゲットビルドには含まれないため）。
///
/// `f32::mul_add` で二乗和を累積し FMA 契約（REQ-2）を GPU 側と揃える。
#[cfg(any(not(target_arch = "aarch64"), test))]
pub(crate) fn rmsnorm_row_scalar(
    row: &[f32],
    w: Option<&[f32]>,
    eps: f32,
    inv_n: f32,
    out_row: &mut [f32],
) {
    let mut acc = 0.0f32;
    for &v in row {
        acc = v.mul_add(v, acc);
    }
    let rstd = 1.0f32 / acc.mul_add(inv_n, eps).sqrt();
    match w {
        Some(w) => {
            for ((o, &v), &wv) in out_row.iter_mut().zip(row.iter()).zip(w.iter()) {
                *o = v * rstd * wv;
            }
        }
        None => {
            for (o, &v) in out_row.iter_mut().zip(row.iter()) {
                *o = v * rstd;
            }
        }
    }
}

/// NEON 経路（`cfg(target_arch = "aarch64")` 限定）。
///
/// 二乗和は `chunks_exact(4)` の full チャンクを `vfmaq_f32` で累積し、
/// 端数（`hidden % 4 != 0`）はスカラー（[`rmsnorm_row_scalar`] と同一定義の
/// `f32::mul_add`）で処理する（REQ-8 境界検査。ベクトル化ロードは検証済み
/// full チャンクのみを対象とし `get_unchecked` は使わない）。出力パスも
/// 同じ分割で `vmulq_f32`（+ 任意 `w` の `vmulq_f32`）を適用する。
#[cfg(target_arch = "aarch64")]
fn rmsnorm_row_neon(row: &[f32], w: Option<&[f32]>, eps: f32, inv_n: f32, out_row: &mut [f32]) {
    use std::arch::aarch64::{vaddvq_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vmulq_n_f32, vst1q_f32};

    let chunks = row.chunks_exact(4);
    let remainder = chunks.remainder();
    // SAFETY: `chunks_exact(4)` が返す各チャンクは長さ 4 を境界検査済み
    // （REQ-8）。`vld1q_f32` はチャンク先頭ポインタから 4 要素（16 バイト）
    // 読み出すが、チャンク自体が 4 要素ちょうどのスライスであるため範囲外
    // 読み出しは起きない。
    let acc_vec = unsafe {
        let mut acc = std::arch::aarch64::vdupq_n_f32(0.0);
        for c in chunks {
            let v = vld1q_f32(c.as_ptr());
            acc = vfmaq_f32(acc, v, v);
        }
        acc
    };
    // 端数はスカラーで `f32::mul_add`（NEON `vfmaq_f32` と同一の FMA
    // 契約・演算順序: ベクタ部分を先に、端数を後に累積する）。
    let mut acc_scalar = 0.0f32;
    for &v in remainder {
        acc_scalar = v.mul_add(v, acc_scalar);
    }
    // `vaddvq_f32` はレーン間水平和（NEON 標準の水平縮約命令）。
    let acc = unsafe { vaddvq_f32(acc_vec) } + acc_scalar;
    let rstd = 1.0f32 / acc.mul_add(inv_n, eps).sqrt();

    let out_chunks = out_row.chunks_exact_mut(4);
    let row_chunks = row.chunks_exact(4);
    match w {
        Some(w) => {
            let w_chunks = w.chunks_exact(4);
            for ((oc, rc), wc) in out_chunks.zip(row_chunks).zip(w_chunks) {
                // SAFETY: `chunks_exact(4)` により `oc`/`rc`/`wc` はいずれも
                // 長さ 4（境界検査済み）。`vld1q_f32`/`vst1q_f32` は
                // 4 要素のみを読み書きする。
                unsafe {
                    let rv = vld1q_f32(rc.as_ptr());
                    let wv = vld1q_f32(wc.as_ptr());
                    let normed = vmulq_n_f32(rv, rstd);
                    let out_v = vmulq_f32(normed, wv);
                    vst1q_f32(oc.as_mut_ptr(), out_v);
                }
            }
            let rem_start = row.len() - row.chunks_exact(4).remainder().len();
            for i in rem_start..row.len() {
                out_row[i] = row[i] * rstd * w[i];
            }
        }
        None => {
            for (oc, rc) in out_chunks.zip(row_chunks) {
                // SAFETY: 上記と同じ（`w` 分岐が無いのみ）。
                unsafe {
                    let rv = vld1q_f32(rc.as_ptr());
                    let out_v = vmulq_n_f32(rv, rstd);
                    vst1q_f32(oc.as_mut_ptr(), out_v);
                }
            }
            let rem_start = row.len() - row.chunks_exact(4).remainder().len();
            for i in rem_start..row.len() {
                out_row[i] = row[i] * rstd;
            }
        }
    }
}

/// canonical RMSNorm 融合プラン（`x * rsqrt(sum(x^2))`。`mean` 化・`eps`・
/// `weight` を含まない）に厳密一致する `plan` から、起動に必要な行長
/// （`row_fusion().row_len()`）を取り出す。
///
/// `backend-cuda::rmsnorm::match_rmsnorm_plan` と同一の 6 op 列・leaf 1
/// 個・`row_fusion().axis() == None` 厳密一致契約を持つ（重複実装ではなく
/// 同一契約の CPU 側ミラー。プランは `fandhe_ai_tensor_core::FusionPlan` のバックエンド
/// 非依存 DTO であり、`tensor-core` 内部の `pub(crate)` 型には依存しない）。
pub(crate) fn match_rmsnorm_plan(plan: &fandhe_ai_tensor_core::FusionPlan) -> Option<usize> {
    use fandhe_ai_tensor_core::{FusedOpKind, RowFusionMeta};

    if plan.leaf_count() != 1 {
        return None;
    }
    let ops: Vec<FusedOpKind> = plan.ops().collect();
    if ops.len() != 6 {
        return None;
    }
    let expect = [
        matches!(ops[0], FusedOpKind::Input { leaf_index: 0 }),
        matches!(ops[1], FusedOpKind::Mul { lhs: 0, rhs: 0 }),
        matches!(
            ops[2],
            FusedOpKind::Sum {
                input: 1,
                axis: None
            }
        ),
        matches!(ops[3], FusedOpKind::Rsqrt { input: 2 }),
        matches!(
            ops[4],
            FusedOpKind::Broadcast {
                input: 3,
                axis: None
            }
        ),
        matches!(ops[5], FusedOpKind::Mul { lhs: 4, rhs: 0 }),
    ];
    if expect.iter().any(|ok| !ok) {
        return None;
    }

    let row_fusion: &RowFusionMeta = plan.row_fusion()?;
    if row_fusion.axis().is_some() {
        return None;
    }
    Some(row_fusion.row_len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rmsnorm_launch_accepts_matching_dims() {
        assert!(validate_rmsnorm_launch(3, 8, 24, Some(8)).is_ok());
        assert!(validate_rmsnorm_launch(3, 8, 24, None).is_ok());
    }

    #[test]
    fn validate_rmsnorm_launch_rejects_x_len_mismatch() {
        let err = validate_rmsnorm_launch(3, 8, 23, None).unwrap_err();
        assert!(matches!(err, RmsNormError::InvalidShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_launch_rejects_w_len_mismatch() {
        let err = validate_rmsnorm_launch(3, 8, 24, Some(7)).unwrap_err();
        assert!(matches!(err, RmsNormError::WeightLenMismatch { .. }));
    }

    #[test]
    fn run_rmsnorm_f32_empty_rows_or_hidden_returns_empty() {
        assert_eq!(
            run_rmsnorm_f32(&[], None, 1e-5, 0, 8).unwrap(),
            Vec::<f32>::new()
        );
        assert_eq!(
            run_rmsnorm_f32(&[], None, 1e-5, 3, 0).unwrap(),
            Vec::<f32>::new()
        );
    }

    #[test]
    fn run_rmsnorm_f32_basic_no_weight() {
        // hidden=4, x = [1,2,3,4] -> mean(x^2) = (1+4+9+16)/4 = 7.5
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = run_rmsnorm_f32(&x, None, 0.0, 1, 4).unwrap();
        let rstd = 1.0f32 / 7.5f32.sqrt();
        for (o, v) in out.iter().zip(x.iter()) {
            assert!((o - v * rstd).abs() < 1e-5);
        }
    }

    #[test]
    fn match_rmsnorm_plan_accepts_canonical_plan() {
        use fandhe_ai_tensor_core::{DType, FusedOpKind, FusionPlan};
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Mul { lhs: 0, rhs: 0 },
            FusedOpKind::Sum {
                input: 1,
                axis: None,
            },
            FusedOpKind::Rsqrt { input: 2 },
            FusedOpKind::Broadcast {
                input: 3,
                axis: None,
            },
            FusedOpKind::Mul { lhs: 4, rhs: 0 },
        ];
        let plan = FusionPlan::from_ops(ops, vec![8], DType::F32, 1).unwrap();
        assert_eq!(match_rmsnorm_plan(&plan), Some(8));
    }

    #[test]
    fn match_rmsnorm_plan_rejects_elementwise_only_plan() {
        use fandhe_ai_tensor_core::{DType, FusedOpKind, FusionPlan};
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Input { leaf_index: 1 },
            FusedOpKind::Add { lhs: 0, rhs: 1 },
        ];
        let plan = FusionPlan::from_ops(ops, vec![4], DType::F32, 2).unwrap();
        assert_eq!(match_rmsnorm_plan(&plan), None);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_various_hidden() {
        for hidden in [1usize, 3, 4, 5, 7, 8, 17, 33, 128] {
            let row: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.37 - 1.5).collect();
            let w: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.01).collect();
            for use_w in [false, true] {
                let mut out_neon = vec![0.0f32; hidden];
                let mut out_scalar = vec![0.0f32; hidden];
                let wref = if use_w { Some(w.as_slice()) } else { None };
                rmsnorm_row_neon(&row, wref, 1e-5, 1.0 / hidden as f32, &mut out_neon);
                rmsnorm_row_scalar(&row, wref, 1e-5, 1.0 / hidden as f32, &mut out_scalar);
                for (a, b) in out_neon.iter().zip(out_scalar.iter()) {
                    assert!((a - b).abs() < 1e-5, "hidden={hidden} a={a} b={b}");
                }
            }
        }
    }
}
