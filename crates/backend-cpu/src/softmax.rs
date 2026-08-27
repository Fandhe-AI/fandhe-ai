//! 融合 softmax 順伝播カーネル（CPU 参照実装。NEON + `rayon` 行方向並列。
//! イシュー #607）。
//!
//! `backend-cuda::softmax`（#594）・`backend-metal::softmax`（#604）と同じ
//! モジュール構成方針を踏襲する: [`run_softmax_f32`] を公開エントリとし、
//! `ops.rs::CpuBackendOps::run_fused` は canonical 融合プラン
//! （`exp(x - max(x)) / sum(exp(x - max(x)))`）検出時（`match_softmax_plan`
//! 参照）に本モジュールへルーティングする。
//!
//! # exp 実装方式の採用判断（受入基準 2。tolerance 緩和なし）
//!
//! 標準 `f32::exp` を採用する。理由:
//! 1. `onnx-interop` 素朴実装（`ops/softmax.rs`）・GPU parity テストの CPU
//!    参照実装がいずれも `f32::exp`／`expf` を使うため、REQ-2 一致が構成上
//!    保証される。
//! 2. `exp2` 変換（GPU カーネルが実際に計算する `exp2(x * log2(e))`）との
//!    A/B 比較はアーキテクチャ非依存の算術（`f32::exp2` と `f32::exp` の
//!    比較のみで NEON intrinsic を要さない）であり x86_64 でも実行可能
//!    なため、`tests::exp2_matches_exp_within_parity_tolerance`（本モジュール
//!    末尾）で実施済み。softmax の実引数域（`v - max <= 0`。極値
//!    `[-1e30, 0]` を含む）で REQ-2 統一複合判定
//!    （[`crate::parity::compare`]）を満たすことを確認したうえで、桁数
//!    増（追加の `log2(e)` 乗算・`exp2` 呼び出し）に見合う性能上の利点が
//!    softmax 単体では小さいため、実装を分岐させず `f32::exp` に統一する
//!    判断とした（GPU 側との一致根拠は 3. を参照）。
//! 3. NEON 多項式近似 exp は、本実装セッション・CI が x86_64 のため
//!    aarch64 実機での実測 A/B 検証ができない。未検証の近似実装を出荷
//!    経路として採用すると REQ-2 複合判定に対し安全側でないため不採用。
//!    実機 A/B による将来採用はフォローアップ（実機検証環境
//!    `docs/real-hardware-verification-env.md` でのみ実行可能）。
//! 4. GPU 側 `exp2` カーネルとの一致は既存 CUDA/Metal parity テスト
//!    （`f32::exp` ベース CPU 参照で green）が実証済みであり、本実装が
//!    同じ参照点を使うことで一致根拠を引き継ぐ。
//!
//! # NEON / スカラー二重経路・FMA 契約・境界検査
//!
//! [`rmsnorm`](crate::rmsnorm) モジュールと同じ方針（`cfg(target_arch =
//! "aarch64")` の `std::arch::aarch64` intrinsics + 常時コンパイルの
//! スカラー参照実装。REQ-8 境界検査は `as_chunks::<4>()` の full チャンクの
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

/// スカラー経路（`pub(crate)`: 非 aarch64 では [`softmax_row`] の通常経路
/// として使う。aarch64 では本体からは呼ばれないが、本ファイル内
/// `#[cfg(test)] mod tests` の NEON/スカラー A/B 同値テストが参照実装として
/// 直接呼ぶため、`cfg(any(not(aarch64), test))` で当該テスト構成時のみ
/// aarch64 でも残す（[`crate::rmsnorm::rmsnorm_row_scalar`] と同じ理由。
/// dead_code 判定はコンパイル単位ごとであり、テスト専用の呼び出しは通常の
/// `lib` ターゲットビルドには含まれないため）。
///
/// pass1: 行 max（`onnx-interop::ops::softmax` と同一セマンティクス。
/// 初期値 `f32::NEG_INFINITY`）。pass2: `exp(v - max)` を書き込みつつ
/// `sum` を集約。pass3: `1/sum` を乗算。
#[cfg(any(not(target_arch = "aarch64"), test))]
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
/// pass1（行 max）は `as_chunks::<4>()` の full チャンクを `vmaxq_f32` で
/// 縮約し、端数はスカラー `f32::max` を適用する（REQ-8）。pass2
/// （`exp(v - max)` の書き込み + `sum` 集約）は `exp` 自体をベクトル化
/// せず（NEON に `exp` intrinsic はなく、多項式近似は本イシューで未検証
/// のため不採用——モジュール冒頭「exp 実装方式の採用判断」参照）
/// スカラーで行うが、`v - max` の減算・`sum` への加算はスカラーループの
/// 中で完結させる（`softmax_row_scalar` と同一の演算順序・同一結果）。
#[cfg(target_arch = "aarch64")]
fn softmax_row_neon(row: &[f32], out_row: &mut [f32]) {
    use std::arch::aarch64::{vld1q_f32, vmaxq_f32, vmaxvq_f32};

    let (chunks, remainder) = row.as_chunks::<4>();
    let mut max_v = f32::NEG_INFINITY;
    // SAFETY: `as_chunks::<4>()` が返す各チャンクは長さ 4 の固定長配列
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
pub(crate) fn match_softmax_plan(
    plan: &fandhe_ai_tensor_core::FusionPlan,
) -> Option<(usize, usize)> {
    use fandhe_ai_tensor_core::{FusedOpKind, RowFusionMeta};

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
        use fandhe_ai_tensor_core::{DType, FusedOpKind, FusionPlan};
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
        use fandhe_ai_tensor_core::{DType, FusedOpKind, FusionPlan};
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
        assert_eq!(match_softmax_plan(&plan), None);
    }

    /// exp 実装方式 A/B 比較（受入基準 2）: 標準 `f32::exp(x)` と GPU
    /// カーネルが実際に計算する `exp2(x * log2(e))`（`f32::exp2` ベース。
    /// アーキテクチャ非依存の算術のため x86_64 CI でも実行可能——モジュール
    /// 冒頭「exp 実装方式の採用判断」2. 参照）を、softmax の実引数域
    /// （`v - max <= 0`。最大値 0・極値 -1e30 を含む）で突き合わせる。
    /// REQ-2 統一複合判定（[`crate::parity::compare`]。tolerance 緩和なし）
    /// を満たすことを確認する。
    #[test]
    fn exp2_matches_exp_within_parity_tolerance() {
        use crate::parity::assert_parity;

        let xs: Vec<f32> = vec![
            0.0, -1e-6, -0.001, -0.1, -0.5, -1.0, -2.0, -5.0, -10.0, -20.0, -50.0,
            -87.0, // f32::exp(-87) はアンダーフロー境界付近
            -1e3, -1e6, -1e30,
        ];
        let via_exp: Vec<f32> = xs.iter().map(|&v| v.exp()).collect();
        let via_exp2: Vec<f32> = xs
            .iter()
            .map(|&v| (v * std::f32::consts::LOG2_E).exp2())
            .collect();
        assert_parity("softmax exp vs exp2(x*log2(e)) A/B", &via_exp2, &via_exp);
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
