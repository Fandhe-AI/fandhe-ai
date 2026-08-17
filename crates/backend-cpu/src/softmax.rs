//! 融合 softmax 順伝播カーネル（CPU 参照実装。NEON + `rayon` 行方向並列。
//! イシュー #607）。
//!
//! `backend-cuda::softmax`（#594）・`backend-metal::softmax`（#604）と同じ
//! モジュール構成方針を踏襲する: [`run_softmax_f32`] を公開エントリとし、
//! `ops.rs::CpuBackendOps::run_fused` は canonical 融合プラン
//! （`exp(x - max(x)) / sum(exp(x - max(x)))`）検出時（[`match_softmax_plan`]
//! 参照）に本モジュールへルーティングする。
//!
//! # exp 実装方式の採用判断（受入基準 2。tolerance 緩和なし）
//!
//! 標準 `f32::exp` を採用する。理由:
//! 1. `onnx-interop` 素朴実装（`ops/softmax.rs`）・GPU parity テストの CPU
//!    参照実装がいずれも `f32::exp`／`expf` を使うため、REQ-2 一致が構成上
//!    保証される。
//! 2. `exp2` 変換（GPU カーネルが実際に計算する `exp2(x * log2(e))`）・NEON
//!    多項式近似は、本実装セッション・CI が x86_64 のため aarch64 実機での
//!    実測 A/B 検証ができない。未検証の近似実装を出荷経路として採用すると
//!    REQ-2 複合判定（相対誤差 1e-3 未満 または絶対誤差 1e-5 未満）に対し
//!    安全側でないため、標準 `f32::exp` に留める。
//! 3. GPU 側 `exp2` カーネルとの一致は既存 CUDA/Metal parity テスト
//!    （`f32::exp` ベース CPU 参照で green）が実証済みであり、本実装が
//!    同じ参照点を使うことで一致根拠を引き継ぐ。
//!
//! NEON 近似 exp の実機 A/B による将来採用はフォローアップ（実機検証環境
//! `docs/real-hardware-verification-env.md` でのみ実行可能）。
//!
//! # NEON / スカラー二重経路・FMA 契約・境界検査
//!
//! [`rmsnorm`](crate::rmsnorm) モジュールと同じ方針（`cfg(target_arch =
//! "aarch64")` の `std::arch::aarch64` intrinsics + 常時コンパイルの
//! スカラー参照実装。REQ-8 境界検査は `chunks_exact(4)` の full チャンクの
//! みを `unsafe` ロード対象とする）。

use rayon::prelude::*;

use crate::elementwise::PARALLEL_THRESHOLD;

/// [`run_softmax_f32`] の型付きエラー（`rmsnorm::RmsNormError` と同じ
/// 「小さな enum」方針）。
#[non_exhaustive]
#[derive(Debug)]
pub enum SoftmaxError {
    /// `rows * cols` が overflow するか、`x.len()` と一致しない。
    InvalidShape { detail: String },
}

impl std::fmt::Display for SoftmaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SoftmaxError::InvalidShape { detail } => write!(f, "softmax invalid shape: {detail}"),
        }
    }
}

impl std::error::Error for SoftmaxError {}

/// 起動前 fail-closed 検証（`backend-cuda::softmax::validate_softmax_launch`
/// と同型。OWASP A03）: `rows * cols == x.len()`（checked 乗算）。
fn validate_softmax_launch(rows: usize, cols: usize, x_len: usize) -> Result<(), SoftmaxError> {
    let numel = rows
        .checked_mul(cols)
        .ok_or_else(|| SoftmaxError::InvalidShape {
            detail: format!("rows*cols overflowed usize: rows={rows}, cols={cols}"),
        })?;
    if numel != x_len {
        return Err(SoftmaxError::InvalidShape {
            detail: format!("x length mismatch: rows*cols={numel}, x.len()={x_len}"),
        });
    }
    Ok(())
}

/// `x`（`[rows, cols]` の行優先 1 次元化済みスライス）へ行方向 softmax を
/// 適用する: 行ごとに `max` 減算 → `exp` → `sum` → 正規化（3 パス構成。
/// 同一ロードパスでの online 化は本イシューのスコープ外——実装計画
/// §2.2「online 化は行わず、まず素朴参照と丸め挙動が近い多パス構成で
/// parity を確立する」）。
pub fn run_softmax_f32(x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, SoftmaxError> {
    validate_softmax_launch(rows, cols, x.len())?;

    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }

    let mut out = vec![0.0f32; x.len()];
    let numel = rows * cols;

    if numel >= PARALLEL_THRESHOLD && rows >= 2 {
        out.par_chunks_mut(cols)
            .zip(x.par_chunks(cols))
            .for_each(|(out_row, in_row)| {
                softmax_row(in_row, out_row);
            });
    } else {
        for (out_row, in_row) in out.chunks_mut(cols).zip(x.chunks(cols)) {
            softmax_row(in_row, out_row);
        }
    }

    Ok(out)
}

/// 1 行分の softmax を計算する（NEON / スカラーの経路選択点）。
fn softmax_row(row: &[f32], out_row: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        softmax_row_neon(row, out_row);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        softmax_row_scalar(row, out_row);
    }
}

/// スカラー経路（`pub(crate)`: aarch64 での NEON/スカラー A/B 同値テスト・
/// exp 方式 A/B ハーネスの参照実装として `tests/softmax_parity.rs` から
/// 直接呼ばれる）。
///
/// pass1: 行 max（`onnx-interop::ops::softmax` と同一セマンティクス。
/// 初期値 `f32::NEG_INFINITY`）。pass2: `exp(v - max)` を書き込みつつ
/// `sum` を集約。pass3: `1/sum` を乗算。
pub(crate) fn softmax_row_scalar(row: &[f32], out_row: &mut [f32]) {
    let mut max_v = f32::NEG_INFINITY;
    for &v in row {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = 0.0f32;
    for (o, &v) in out_row.iter_mut().zip(row.iter()) {
        let e = (v - max_v).exp();
        *o = e;
        sum += e;
    }
    let inv_sum = 1.0f32 / sum;
    for o in out_row.iter_mut() {
        *o *= inv_sum;
    }
}

/// NEON 経路（`cfg(target_arch = "aarch64")` 限定）。
///
/// pass1（行 max）は `chunks_exact(4)` の full チャンクを `vmaxq_f32` で
/// 縮約し、端数はスカラー `f32::max` を適用する（REQ-8）。pass2
/// （`exp(v - max)` の書き込み + `sum` 集約）は `exp` 自体をベクトル化
/// せず（NEON に `exp` intrinsic はなく、多項式近似は本イシューで未検証
/// のため不採用——モジュール冒頭「exp 実装方式の採用判断」参照）
/// スカラーで行うが、`v - max` の減算・`sum` への加算はスカラーループの
/// 中で完結させる（`softmax_row_scalar` と同一の演算順序・同一結果）。
#[cfg(target_arch = "aarch64")]
fn softmax_row_neon(row: &[f32], out_row: &mut [f32]) {
    use std::arch::aarch64::{vld1q_f32, vmaxq_f32, vmaxvq_f32};

    let chunks = row.chunks_exact(4);
    let remainder = chunks.remainder();
    let mut max_v = f32::NEG_INFINITY;
    // SAFETY: `chunks_exact(4)` が返す各チャンクは長さ 4 を境界検査済み
    // （REQ-8）。`vld1q_f32` はチャンク先頭ポインタから 4 要素のみ読む。
    if row.len() >= 4 {
        let max_vec = unsafe {
            let mut acc = vld1q_f32(row.as_ptr());
            for c in chunks {
                let v = vld1q_f32(c.as_ptr());
                acc = vmaxq_f32(acc, v);
            }
            acc
        };
        max_v = unsafe { vmaxvq_f32(max_vec) };
    }
    for &v in remainder {
        if v > max_v {
            max_v = v;
        }
    }

    // pass2/pass3 はスカラー経路と同一定義（`exp` は libm 経由。上記
    // モジュールドキュメンテーションコメント参照）。
    let mut sum = 0.0f32;
    for (o, &v) in out_row.iter_mut().zip(row.iter()) {
        let e = (v - max_v).exp();
        *o = e;
        sum += e;
    }
    let inv_sum = 1.0f32 / sum;
    for o in out_row.iter_mut() {
        *o *= inv_sum;
    }
}

/// canonical softmax 融合プラン（`exp(x - max(x)) / sum(exp(x - max(x)))`）
/// に厳密一致する `plan` から、起動に必要な `(rows, cols)` を取り出す。
///
/// `backend-cuda::softmax::match_softmax_plan` と同一の 8 op 列・leaf 1
/// 個・軸は最終次元または `None` のみ受理する契約を持つ（重複実装では
/// なく同一契約の CPU 側ミラー。モジュール冒頭コメント参照）。
pub(crate) fn match_softmax_plan(plan: &tensor_core::FusionPlan) -> Option<(usize, usize)> {
    use tensor_core::{FusedOpKind, RowFusionMeta};

    if plan.leaf_count() != 1 {
        return None;
    }
    let ops: Vec<FusedOpKind> = plan.ops().collect();
    if ops.len() != 8 {
        return None;
    }
    if !matches!(ops[0], FusedOpKind::Input { leaf_index: 0 }) {
        return None;
    }
    let axis = match ops[1] {
        FusedOpKind::Max { input: 0, axis } => axis,
        _ => return None,
    };
    let expect = [
        matches!(ops[2], FusedOpKind::Broadcast { input: 1, axis: a } if a == axis),
        matches!(ops[3], FusedOpKind::Sub { lhs: 0, rhs: 2 }),
        matches!(ops[4], FusedOpKind::Exp { input: 3 }),
        matches!(ops[5], FusedOpKind::Sum { input: 4, axis: a } if a == axis),
        matches!(ops[6], FusedOpKind::Broadcast { input: 5, axis: a } if a == axis),
        matches!(ops[7], FusedOpKind::Div { lhs: 4, rhs: 6 }),
    ];
    if expect.iter().any(|ok| !ok) {
        return None;
    }

    let output_shape = plan.output_shape();
    let rank = output_shape.len();
    if let Some(a) = axis
        && (rank == 0 || a != rank - 1)
    {
        // 最終軸以外（中間軸 softmax）は対象外。
        return None;
    }

    let row_fusion: &RowFusionMeta = plan.row_fusion()?;
    if row_fusion.axis() != axis {
        return None;
    }
    let cols = row_fusion.row_len();

    let rows = match axis {
        None => 1,
        Some(a) => {
            if output_shape[a] != cols {
                return None;
            }
            output_shape[..a].iter().product()
        }
    };
    Some((rows, cols))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_softmax_launch_accepts_matching_dims() {
        assert!(validate_softmax_launch(3, 8, 24).is_ok());
    }

    #[test]
    fn validate_softmax_launch_rejects_x_len_mismatch() {
        let err = validate_softmax_launch(3, 8, 23).unwrap_err();
        assert!(matches!(err, SoftmaxError::InvalidShape { .. }));
    }

    #[test]
    fn run_softmax_f32_empty_rows_or_cols_returns_empty() {
        assert_eq!(run_softmax_f32(&[], 0, 8).unwrap(), Vec::<f32>::new());
        assert_eq!(run_softmax_f32(&[], 3, 0).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn run_softmax_f32_row_sums_to_one() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let out = run_softmax_f32(&x, 2, 4).unwrap();
        for row in out.chunks(4) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row sum={sum}");
        }
    }

    #[test]
    fn run_softmax_f32_extreme_values_no_nan_inf() {
        let x = vec![1e30f32, -1e30, 1e30, -1e30];
        let out = run_softmax_f32(&x, 1, 4).unwrap();
        for &v in &out {
            assert!(v.is_finite(), "expected finite, got {v}");
        }
    }

    #[test]
    fn match_softmax_plan_accepts_canonical_plan_rank1() {
        use tensor_core::{DType, FusedOpKind, FusionPlan};
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Max {
                input: 0,
                axis: None,
            },
            FusedOpKind::Broadcast {
                input: 1,
                axis: None,
            },
            FusedOpKind::Sub { lhs: 0, rhs: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Sum {
                input: 4,
                axis: None,
            },
            FusedOpKind::Broadcast {
                input: 5,
                axis: None,
            },
            FusedOpKind::Div { lhs: 4, rhs: 6 },
        ];
        let plan = FusionPlan::from_ops(ops, vec![8], DType::F32, 1).unwrap();
        assert_eq!(match_softmax_plan(&plan), Some((1, 8)));
    }

    #[test]
    fn match_softmax_plan_accepts_last_axis_2d() {
        use tensor_core::{DType, FusedOpKind, FusionPlan};
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Max {
                input: 0,
                axis: Some(1),
            },
            FusedOpKind::Broadcast {
                input: 1,
                axis: Some(1),
            },
            FusedOpKind::Sub { lhs: 0, rhs: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Sum {
                input: 4,
                axis: Some(1),
            },
            FusedOpKind::Broadcast {
                input: 5,
                axis: Some(1),
            },
            FusedOpKind::Div { lhs: 4, rhs: 6 },
        ];
        let plan = FusionPlan::from_ops(ops, vec![2, 8], DType::F32, 1).unwrap();
        assert_eq!(match_softmax_plan(&plan), Some((2, 8)));
    }

    #[test]
    fn match_softmax_plan_rejects_rmsnorm_shaped_plan() {
        use tensor_core::{DType, FusedOpKind, FusionPlan};
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
        assert_eq!(match_softmax_plan(&plan), None);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_various_cols() {
        for cols in [1usize, 2, 3, 4, 5, 7, 8, 17, 33, 128] {
            let row: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.53 - 2.0).collect();
            let mut out_neon = vec![0.0f32; cols];
            let mut out_scalar = vec![0.0f32; cols];
            softmax_row_neon(&row, &mut out_neon);
            softmax_row_scalar(&row, &mut out_scalar);
            for (a, b) in out_neon.iter().zip(out_scalar.iter()) {
                assert!((a - b).abs() < 1e-6, "cols={cols} a={a} b={b}");
            }
        }
    }
}
