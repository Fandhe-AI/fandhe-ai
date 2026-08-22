//! 融合 RMSNorm 順伝播カーネルの起動 API（NVRTC コンパイル・保持・実行。
//! イシュー #592）。
//!
//! `elementwise.rs::CudaElementwise` と同じ構成方針を踏襲する:
//! [`CudaRmsNorm::new`] が `CudaDevice` から 2 カーネル（1 パス／2 パス。
//! `kernels_rmsnorm.rs`）を一括 NVRTC コンパイルして保持し、同時にデバイス
//! 属性（SMEM 予算・SM 数）を 1 回だけ取得してキャッシュする
//! （`gemm_auto.rs::CudaGemmAuto` の `DeviceCaps` キャッシュと同じ設計
//! 判断: 呼び出しごとに driver API 照会を繰り返さない）。以降は
//! [`CudaRmsNorm::run_rmsnorm_f32`] へホスト側スライスを渡すだけで
//! 経路選択（[`rmsnorm_route`]）・persistent grid 導出
//! （[`derive_persistent_grid_one_pass`]／[`derive_persistent_grid_two_pass`]）・
//! H2D → 起動 → 同期 → D2H を内部で完結できる。
//!
//! `ops.rs::CudaBackendOps::run_fused` から canonical RMSNorm プラン
//! （`x * rsqrt(sum(x^2))`。`mean`／`eps`／`weight` を含まない）検出時に
//! 呼ばれる（[`match_rmsnorm_plan`] 参照）。

use std::sync::Arc;

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use fandhe_ai_tensor_core::{FusedOpKind, FusionPlan, RowFusionMeta};

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm_auto::read_clamped_smem_budget_bytes;
use crate::kernels_rmsnorm::{
    self, RMSNORM_BLOCK_DIM, RMSNORM_BWD_BLOCK_DIM, RMSNORM_BWD_DW_PARTIAL_BLOCK_DIM,
    RMSNORM_DW_REDUCE_BLOCK_DIM,
};
use crate::nvrtc::compile_ptx;

/// [`rmsnorm_route`] が返す経路選択。行長（`RowFusionMeta::row_len`／
/// 直接 API の `hidden`）が 1 パス経路（動的 SMEM 常駐）の予算に収まるか
/// どうかで分岐する（実装計画 §4.4）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RmsNormRoute {
    /// [`kernels_rmsnorm::RMSNORM_F32_ONEPASS`]（動的 SMEM 常駐・`x` の
    /// HBM 読みは 1 回のみ）。
    OnePassSmem,
    /// [`kernels_rmsnorm::RMSNORM_F32_TWOPASS`]（global 再読・SMEM 非使用）。
    TwoPass,
}

/// `rows`／`hidden` の対を 1 引数へまとめる（`clippy::too_many_arguments`
/// 対策。[`CudaRmsNorm::run_rmsnorm_bwd_f32`]（`pub`）の引数に使うため
/// `pub` とする。内部専用の `CudaRmsNorm::run_rmsnorm_f32_inner` も同じ
/// 2 値を使い回すため共有する）。
#[derive(Debug, Clone, Copy)]
pub struct RmsNormShape {
    /// 正規化対象の行数（バッチ次元。`x`／`dy`／`dx` の長さは
    /// `rows * hidden` に一致する契約）。
    pub rows: usize,
    /// 1 行あたりの要素数（正規化軸の長さ。`w`／`dw` の長さおよび
    /// [`CudaRmsNorm::run_rmsnorm_f32_train`]・
    /// [`CudaRmsNorm::run_rmsnorm_bwd_f32`] が内部導出する
    /// `inv_n = 1.0 / hidden`（`hidden == 0` は 1.0）の分母に一致する契約）。
    pub hidden: usize,
}

/// `row_len`（行長）が `per_block_smem_budget_bytes`（[`crate::gemm_auto::
/// read_clamped_smem_budget_bytes`] でクランプ済みの per-block SMEM 予算）
/// に `f32` 換算で収まるかを純関数で判定する（実機なしで単体テスト可能）。
///
/// **1 パス／2 パス判定は本関数（バックエンド側）の責務**である
/// （`fandhe_ai_tensor_core::fusion::RowFusionMeta` ドキュメンテーションコメント
/// 「1 パス／2 パス判定は各バックエンドの責務」参照。#588 codex-review
/// P2 是正で `tensor-core` 側から閾値を切り離した設計を、本関数が CUDA
/// バックエンド側の具体的な予算判定として実装する）。opt-in SMEM
/// （`cuFuncSetAttribute`）は使わないため、予算は常に既定 per-block 上限
/// （[`crate::gemm_auto::STATIC_SMEM_BUDGET_CAP_BYTES`] でクランプ済み）
/// を超えない（`read_clamped_smem_budget_bytes` 呼び出し元が保証する）。
pub(crate) fn rmsnorm_route(row_len: usize, per_block_smem_budget_bytes: u64) -> RmsNormRoute {
    let needed_bytes = (row_len as u64).saturating_mul(4);
    if needed_bytes <= per_block_smem_budget_bytes {
        RmsNormRoute::OnePassSmem
    } else {
        RmsNormRoute::TwoPass
    }
}

/// 1 パス経路（動的 SMEM 常駐）の persistent grid（block 数）を導出する
/// 純関数（実装計画 §4.4）。
///
/// `blocks_per_sm = min(smem_per_sm_bytes / smem_bytes_per_block, 16)`
/// （16 は参照実装〈TileKernels engram gate カーネル〉由来のレジスタ圧
/// キャップであり、実測値ではなくコード上の定数である。`smem_bytes_per_block
/// == 0` は `hidden == 0` の縮退ケースを表し、SMEM 制約が事実上存在しない
/// ため 2 パス経路と同じ上限 16 を用いる）。`grid = clamp(sm_count *
/// blocks_per_sm, 1, rows)`。`rows == 0` は呼び出し元（[`CudaRmsNorm::
/// run_rmsnorm_f32`]）が早期 return するため本関数へは渡らない契約
/// （`rows == 0` で呼ばれた場合は `clamp` パニックを避けるため 1 を返す
/// フェイルセーフのみ持つ）。
pub(crate) fn derive_persistent_grid_one_pass(
    smem_per_sm_bytes: u64,
    sm_count: u32,
    smem_bytes_per_block: u64,
    rows: u32,
) -> u32 {
    if rows == 0 {
        return 1;
    }
    let blocks_per_sm: u64 = smem_per_sm_bytes
        .checked_div(smem_bytes_per_block)
        .map_or(16, |v| v.clamp(1, 16));
    let grid = (sm_count as u64).saturating_mul(blocks_per_sm);
    grid.clamp(1, rows as u64) as u32
}

/// 2 パス経路（SMEM 制約なし）の persistent grid を導出する純関数。
/// [`derive_persistent_grid_one_pass`] と同じ `rows == 0` フェイルセーフを
/// 持つ。
pub(crate) fn derive_persistent_grid_two_pass(sm_count: u32, rows: u32) -> u32 {
    if rows == 0 {
        return 1;
    }
    let grid = (sm_count as u64).saturating_mul(16);
    grid.clamp(1, rows as u64) as u32
}

/// `rmsnorm_bwd_dw_f32`（`kernels_rmsnorm.rs`）専用のブロックあたり
/// スレッド数。dx カーネル（[`RMSNORM_BWD_BLOCK_DIM`]・行方向 persistent
/// grid）とは軸が異なる列方向 grid-stride のため、共有 SM 予算計算に
/// 依存しない単純な 256 スレッドを用いる（reduction を伴わないため
/// warp 数の制約はない）。
pub(crate) const RMSNORM_BWD_DW_BLOCK_DIM: u32 = 256;

/// dw カーネル（列＝`hidden` 方向 grid-stride）の persistent grid を
/// 導出する純関数。dx／順伝播の `derive_persistent_grid_*`（行方向）とは
/// 軸が異なるため独立した関数として持つ（`kernels_rmsnorm.rs::
/// RMSNORM_BWD_DW_F32` ドキュメンテーションコメント参照）。
/// `grid = clamp(sm_count * 16, 1, ceil(hidden / RMSNORM_BWD_DW_BLOCK_DIM))`。
/// `hidden == 0` は呼び出し元が早期 return するため本関数へは渡らない
/// 契約だが、`derive_persistent_grid_two_pass` と同じフェイルセーフ
/// （1 を返す）を持つ。
pub(crate) fn derive_persistent_grid_dw(sm_count: u32, hidden: u32) -> u32 {
    if hidden == 0 {
        return 1;
    }
    let blocks_needed = hidden.div_ceil(RMSNORM_BWD_DW_BLOCK_DIM);
    let grid = (sm_count as u64).saturating_mul(16);
    grid.clamp(1, blocks_needed as u64) as u32
}

/// dw split-K（イシュー #597）: 部分和バッファ（`num_blocks * hidden * 4`
/// bytes）が超えてはならない上限。`derive_dw_split` の fail-closed
/// フォールバック判定にのみ使う（`.claude/rules/security.md` A03:
/// `checked_mul` によるホスト側 usize/u64 オーバーフロー防止）。
pub(crate) const RMSNORM_DW_PARTIAL_BUFFER_CAP_BYTES: u64 = 64 * 1024 * 1024;

/// dw split-K の 1 CTA（`blockIdx.y`）が担当する行数の下限。これを下回る
/// 細分化は、部分和の書き出しコスト（追加の縮約カーネル起動・HBM
/// トラフィック）が並列化の利得を上回るため split-K を適用しない基準に
/// 使う（実装計画 §3.3。実測ではなくコード上の初期値）。
pub(crate) const RMSNORM_DW_MIN_ROWS_PER_BLOCK: u32 = 32;

/// dw split-K の `num_blocks`（gridDim.y）上限。部分和バッファサイズと
/// 縮約カーネルのバッチ数（[`RMSNORM_DW_REDUCE_BATCH`]〈`kernels_rmsnorm`〉
/// 単位）を抑える（実装計画 §3.3。実測ではなくコード上の初期値）。
pub(crate) const RMSNORM_DW_MAX_SPLIT: u32 = 64;

/// dw split-K の段数 `num_blocks` を導出する純関数（実装計画 §3.3。
/// `CudaRmsNorm::run_rmsnorm_bwd_f32` から呼ばれる）。
///
/// 目標 occupancy は既存 [`derive_persistent_grid_dw`] と同じ
/// `sm_count * 16` を「列タイル数 × num_blocks」の積で狙う: `hidden` が
/// 広く列タイル数だけで occupancy を確保できる形状では `num_blocks` を
/// 増やす余地がないため `1`（単段固定。呼び出し元はこの場合
/// [`kernels_rmsnorm::RMSNORM_BWD_DW_F32`] へフォールバックする）を返す。
/// `hidden` が狭く `rows` が大きい形状でのみ split-K の並列度が効く
/// （advisor 指摘: 「rows が大きければ必ず num_blocks >= 2」という単純な
/// 主張は `hidden` 依存で偽になるため採らない）。
///
/// クランプ順序: `sm_count * 16 / col_tiles` を `[1, rows/
/// RMSNORM_DW_MIN_ROWS_PER_BLOCK, RMSNORM_DW_MAX_SPLIT, rows]` の下限へ
/// クランプした後、部分和バッファが [`RMSNORM_DW_PARTIAL_BUFFER_CAP_BYTES`]
/// を超える場合は `num_blocks` を単調減少させ、`2` 未満になったら `1`
/// （単段）へフォールバックする（`checked_mul` で usize/u64
/// オーバーフローを検査。上限側から線形に下げるだけで十分（`num_blocks`
/// は高々 [`RMSNORM_DW_MAX_SPLIT`] であり性能上のホットパスでもない）。
pub(crate) fn derive_dw_split(sm_count: u32, rows: u32, hidden: u32) -> u32 {
    if rows == 0 || hidden == 0 {
        return 1;
    }

    let col_tiles = hidden.div_ceil(RMSNORM_BWD_DW_PARTIAL_BLOCK_DIM).max(1);
    let target = (sm_count as u64).saturating_mul(16);
    let raw_num_blocks = (target / col_tiles as u64).max(1);

    // 行数下限（`RMSNORM_DW_MIN_ROWS_PER_BLOCK` 未満の細分化を避ける）・
    // `RMSNORM_DW_MAX_SPLIT`・`rows` 自体（1 block = 最低 1 行）の 3 つで
    // 上限をクランプする。
    let rows_cap = (rows / RMSNORM_DW_MIN_ROWS_PER_BLOCK).max(1);
    let max_num_blocks = rows_cap.min(RMSNORM_DW_MAX_SPLIT).min(rows);
    if max_num_blocks < 2 {
        return 1;
    }

    let mut num_blocks = raw_num_blocks.clamp(1, max_num_blocks as u64) as u32;
    if num_blocks < 2 {
        return 1;
    }

    // 部分和バッファ上限検査。境界超過時は `num_blocks` を単調減少させる
    // （`num_blocks` は高々 `RMSNORM_DW_MAX_SPLIT` = 64 のため最大 63 回の
    // ループで確実に停止する）。
    let fits_budget = |n: u32| -> bool {
        (n as u64)
            .checked_mul(hidden as u64)
            .and_then(|v| v.checked_mul(4))
            .is_some_and(|bytes| bytes <= RMSNORM_DW_PARTIAL_BUFFER_CAP_BYTES)
    };
    while num_blocks >= 2 && !fits_budget(num_blocks) {
        num_blocks -= 1;
    }

    if num_blocks < 2 { 1 } else { num_blocks }
}

/// [`kernels_rmsnorm::RMSNORM_BWD_DW_PARTIAL_F32`] の行範囲計算
/// （`rows_per_block = ceil(rows / num_blocks)`・`row_start = b *
/// rows_per_block`・`row_end = min(row_start + rows_per_block, rows)`）を
/// 純粋 Rust でミラーする（advisor 指摘: 実機なしでカーネルの行分割契約
/// 〈ギャップなし・重複なしで `0..rows` を分割する〉を検証できる唯一の
/// CI 到達可能な保証点）。`u64` を使うのはカーネル側の `long long`
/// 算術と同じ理由（本ファイル・`kernels_rmsnorm.rs` 冒頭コメント
/// 「ループ添字のオーバーフロー安全性」参照）。`num_blocks == 0` は
/// カーネルへは渡らない契約（[`derive_dw_split`] は常に `>= 1` を返す）
/// だが、フェイルセーフとして空範囲 `(0, 0)` を返す。
///
/// `#[allow(dead_code)]` について: 本関数は実行経路（カーネル起動）から
/// 呼ばれず、`mod tests::dw_split_row_range_partitions_rows_without_gaps_
/// or_overlap` の検証専用（`kernels_rmsnorm.rs::RMSNORM_DW_REDUCE_BATCH`
/// の同アノテーションと同じ理由）。
#[allow(dead_code)]
pub(crate) fn dw_split_row_range(rows: u64, num_blocks: u32, b: u32) -> (u64, u64) {
    if num_blocks == 0 {
        return (0, 0);
    }
    let rows_per_block = rows.div_ceil(num_blocks as u64);
    let row_start = (b as u64).saturating_mul(rows_per_block);
    let row_end = row_start.saturating_add(rows_per_block).min(rows);
    (row_start, row_end)
}

/// split-K dw 経路（[`kernels_rmsnorm::RMSNORM_BWD_DW_PARTIAL_F32`]／
/// [`kernels_rmsnorm::RMSNORM_BWD_DW_REDUCE_F32`]）専用のホスト側検証
/// （起動前・fail-closed）。[`validate_rmsnorm_backward_launch`]（`x`/
/// `dy`/`rstd`/`w` の形状検証で 7 個の既存テストを持つ）は拡張せず、
/// `num_blocks` 固有の検査を本関数へ分離する（advisor 指摘:
/// `#[doc(hidden)] pub` の split 強制テスト API が任意の `num_blocks` を
/// 受け取るため fail-closed で検証する必要があり、既存関数の確立した
/// 挙動を壊さないため）。呼び出し元は `rows > 0 && hidden > 0`
/// （早期 return 済み）の文脈でのみ本関数を呼ぶ契約。
///
/// `num_blocks` 自体の検査（`>= 1`・`<= rows`）は分岐（単段／split-K）に
/// 依らず常に行うが、部分和バッファの [`RMSNORM_DW_PARTIAL_BUFFER_CAP_
/// BYTES`] cap 検査は `num_blocks > 1`（split-K 経路が実際にバッファを
/// 確保する場合）のみに限定する（cursor[bot] 指摘・PR #716）。
pub(crate) fn validate_dw_split_launch(
    rows: usize,
    hidden: usize,
    num_blocks: u32,
) -> Result<(), CudaError> {
    if num_blocks < 1 {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("rmsnorm dw split num_blocks must be >= 1: num_blocks={num_blocks}"),
        });
    }
    if num_blocks as usize > rows {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!(
                "rmsnorm dw split num_blocks must not exceed rows: num_blocks={num_blocks}, \
                 rows={rows}"
            ),
        });
    }
    // 部分和バッファの cap 検査は split-K 経路（`num_blocks > 1`）にのみ
    // 適用する。単段フォールバック（`num_blocks <= 1`）はこのバッファを
    // 一切確保しないため、cap 超過を理由に拒否すると旧実装（単段カーネル
    // のみ）で成功していた大きな `hidden` の形状を退行させてしまう
    // （cursor[bot] 指摘・PR #716: `hidden * 4` だけで cap を超える形状は
    // `derive_dw_split` が正しく `1` を返すにもかかわらず、分岐前の検証で
    // 一律に拒否されていた）。
    if num_blocks > 1 {
        let partial_bytes = (num_blocks as u64)
            .checked_mul(hidden as u64)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| CudaError::InvalidRmsNormShape {
                detail: format!(
                    "rmsnorm dw split partial buffer size overflowed u64: \
                     num_blocks={num_blocks}, hidden={hidden}"
                ),
            })?;
        if partial_bytes > RMSNORM_DW_PARTIAL_BUFFER_CAP_BYTES {
            return Err(CudaError::InvalidRmsNormShape {
                detail: format!(
                    "rmsnorm dw split partial buffer exceeds cap: bytes={partial_bytes}, \
                     cap={RMSNORM_DW_PARTIAL_BUFFER_CAP_BYTES}"
                ),
            });
        }
    }
    Ok(())
}

/// ホスト側検証（逆伝播起動前・fail-closed。イシュー #596）:
/// `rows * hidden == x_len == dy_len`（checked 乗算）・`rstd_len == rows`・
/// `w_len == Some(hidden)`（指定時のみ）・`rows`／`hidden`／`numel` が
/// `i32::MAX`（カーネル引数 `int rows`／`int hidden` 契約）に収まること
/// を検証する（[`validate_rmsnorm_launch`] と同型・`CudaError::
/// InvalidRmsNormShape` を再利用。`.claude/rules/security.md` A03 対策）。
/// `eps` は逆伝播カーネルの引数に含まれない（保存済み `rstd` から
/// 再計算するため）ため検証対象外。
pub(crate) fn validate_rmsnorm_backward_launch(
    rows: usize,
    hidden: usize,
    x_len: usize,
    dy_len: usize,
    rstd_len: usize,
    w_len: Option<usize>,
) -> Result<(), CudaError> {
    let numel = rows
        .checked_mul(hidden)
        .ok_or_else(|| CudaError::InvalidRmsNormShape {
            detail: format!(
                "rmsnorm backward rows*hidden overflowed usize: rows={rows}, hidden={hidden}"
            ),
        })?;
    if numel != x_len {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!(
                "rmsnorm backward x length mismatch: rows*hidden={numel}, x.len()={x_len}"
            ),
        });
    }
    if numel != dy_len {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!(
                "rmsnorm backward dy length mismatch: rows*hidden={numel}, dy.len()={dy_len}"
            ),
        });
    }
    if rstd_len != rows {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!(
                "rmsnorm backward rstd length mismatch: rows={rows}, rstd.len()={rstd_len}"
            ),
        });
    }
    if let Some(wl) = w_len
        && wl != hidden
    {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("rmsnorm backward w length mismatch: hidden={hidden}, w.len()={wl}"),
        });
    }
    if rows > i32::MAX as usize || hidden > i32::MAX as usize || numel > i32::MAX as usize {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!(
                "rmsnorm backward dims must fit in i32 (kernel argument type): rows={rows}, \
                 hidden={hidden}, numel={numel}"
            ),
        });
    }
    Ok(())
}

/// ホスト側検証（起動前・fail-closed）: `rows * hidden == x_len`（checked
/// 乗算）・`w_len == Some(hidden)`（`w` 指定時のみ）・`rows`／`hidden`／
/// `numel` が `i32::MAX`（カーネル引数 `int rows`／`int hidden` 契約）に
/// 収まること・`eps` が有限かつ非負（`is_finite() && eps >= 0.0`。`0.0`
/// は許容する——`run_fused` 経由の canonical プラン起動は `eps = 0.0` を
/// 渡す契約——`ops.rs` ドキュメンテーションコメント参照。負の `eps` を
/// 許すと `sum(x^2) * inv_n + eps` が負化しうる〈有限入力から `sqrt` が
/// NaN を生成する経路。codex-review 指摘・PR #706 レビュー
/// r3793473250〉ため fail-closed で拒否する）を検証する
/// （`elementwise.rs::validate_elementwise_len` と同じ OWASP A03 対策。
/// `.claude/rules/security.md`）。
pub(crate) fn validate_rmsnorm_launch(
    rows: usize,
    hidden: usize,
    x_len: usize,
    w_len: Option<usize>,
    eps: f32,
) -> Result<(), CudaError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("rmsnorm eps must be finite and non-negative: eps={eps}"),
        });
    }

    let numel = rows
        .checked_mul(hidden)
        .ok_or_else(|| CudaError::InvalidRmsNormShape {
            detail: format!("rmsnorm rows*hidden overflowed usize: rows={rows}, hidden={hidden}"),
        })?;
    if numel != x_len {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("rmsnorm x length mismatch: rows*hidden={numel}, x.len()={x_len}"),
        });
    }
    if let Some(wl) = w_len
        && wl != hidden
    {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("rmsnorm w length mismatch: hidden={hidden}, w.len()={wl}"),
        });
    }
    if rows > i32::MAX as usize || hidden > i32::MAX as usize || numel > i32::MAX as usize {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!(
                "rmsnorm dims must fit in i32 (kernel argument type): rows={rows}, \
                 hidden={hidden}, numel={numel}"
            ),
        });
    }
    Ok(())
}

/// canonical RMSNorm 融合プラン（`x * rsqrt(sum(x^2))`。mean 化・eps・
/// weight を含まない）に厳密一致する `plan` から、起動に必要な行長
/// （`row_fusion().row_len()`）を取り出す。
///
/// プラン形状は `plan.rs::from_segment_builds_rmsnorm_plan_with_row_fusion_metadata`
/// が構築する 6 op 列（leaf 1 個・`Mul(0,0) → Sum{axis:None} → Rsqrt →
/// Broadcast{axis:None} → Mul(bc, x)`）に厳密一致する場合のみ `Some` を
/// 返す。一致しない場合（softmax 型・elementwise-only 等）は `None` を
/// 返し、呼び出し元（`ops.rs::CudaBackendOps::run_fused`）はデフォルトの
/// `Unsupported` へフォールバックする（`backend-cpu::fused_elementwise::
/// run_fused_elementwise` の allowlist 拒否方針と同じ fail-closed。
/// 実装計画 §5「Step 4」）。
///
/// `axis: None`（全軸縮約）のみを受理する: canonical プランは 1 次元
/// テンソル入力（`row_fusion().axis() == None`・`rows == 1` 相当）を
/// 対象とし、行方向（`axis: Some(a)`）の RMSNorm 融合プランは対象外
/// （現時点の融合 IR〈#588〉が生成する形状と一致させる。実装計画 §5
/// 「Step 4」の「厳密一致」方針）。
pub(crate) fn match_rmsnorm_plan(plan: &FusionPlan) -> Option<usize> {
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

/// 融合 RMSNorm 順伝播カーネル（1 パス／2 パスの 2 エントリ）のコンパイル
/// 済みハンドルと、経路選択・persistent grid 導出に使うデバイス属性
/// （SMEM 予算・SM 数）を保持する。
pub struct CudaRmsNorm {
    stream: Arc<CudaStream>,
    onepass_f32: CudaFunction,
    twopass_f32: CudaFunction,
    /// 逆伝播 dx カーネル（イシュー #596。recompute-in-backward）。
    bwd_dx_f32: CudaFunction,
    /// 逆伝播 dw カーネル（`w.is_some()` の場合のみ起動元が呼ぶ）。
    /// `derive_dw_split` が `num_blocks <= 1` を返す小規模形状のみで使う
    /// フォールバック経路（イシュー #597。`kernels_rmsnorm::
    /// RMSNORM_BWD_DW_F32` ドキュメンテーションコメント参照）。
    bwd_dw_f32: CudaFunction,
    /// dw split-K 第 1 カーネル（部分和生成。イシュー #597）。
    bwd_dw_partial_f32: CudaFunction,
    /// dw split-K 第 2 カーネル（縮約＋epilogue。イシュー #597）。
    bwd_dw_reduce_f32: CudaFunction,
    /// per-block SMEM 予算（[`crate::gemm_auto::STATIC_SMEM_BUDGET_CAP_BYTES`]
    /// でクランプ済み。[`rmsnorm_route`] の分岐に使う）。
    smem_per_block_budget_bytes: u64,
    /// per-SM SMEM 予算（[`derive_persistent_grid_one_pass`] の
    /// `blocks_per_sm` 導出に使う）。
    smem_per_sm_budget_bytes: u64,
    /// SM（マルチプロセッサ）数（`derive_persistent_grid_*` の共通入力）。
    sm_count: u32,
}

impl CudaRmsNorm {
    /// `device` 上で RMSNorm 2 カーネルを NVRTC コンパイルし、経路選択・
    /// persistent grid 導出に使うデバイス属性を 1 回だけ取得して保持する
    /// ハンドルを構築する。
    ///
    /// 属性取得の失敗（driver API エラー・負値応答）は fail-closed で
    /// `CudaError` を返す（推定値へフォールバックしない。`gemm_auto.rs::
    /// read_clamped_smem_budget_bytes`／`select_tile_config_for_device`
    /// と同じ方針）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let onepass_ptx = compile_ptx(kernels_rmsnorm::RMSNORM_F32_ONEPASS, arch)?;
        let twopass_ptx = compile_ptx(kernels_rmsnorm::RMSNORM_F32_TWOPASS, arch)?;

        let onepass_f32 = device
            .context()
            .load_module(onepass_ptx)?
            .load_function("rmsnorm_f32_onepass")?;
        let twopass_f32 = device
            .context()
            .load_module(twopass_ptx)?
            .load_function("rmsnorm_f32_twopass")?;

        let bwd_dx_ptx = compile_ptx(kernels_rmsnorm::RMSNORM_BWD_DX_F32, arch)?;
        let bwd_dw_ptx = compile_ptx(kernels_rmsnorm::RMSNORM_BWD_DW_F32, arch)?;
        let bwd_dx_f32 = device
            .context()
            .load_module(bwd_dx_ptx)?
            .load_function("rmsnorm_bwd_dx_f32")?;
        let bwd_dw_f32 = device
            .context()
            .load_module(bwd_dw_ptx)?
            .load_function("rmsnorm_bwd_dw_f32")?;

        // dw split-K 二段リダクション（イシュー #597）: 部分和生成・縮約
        // の 2 カーネルを追加コンパイル・ロードする（`bwd_dw_f32` と同じ
        // 「NVRTC を事前コンパイルせず文字列のまま埋め込む」契約。
        // `kernels_rmsnorm.rs` 冒頭コメント参照）。
        let bwd_dw_partial_ptx = compile_ptx(kernels_rmsnorm::RMSNORM_BWD_DW_PARTIAL_F32, arch)?;
        let bwd_dw_reduce_ptx = compile_ptx(kernels_rmsnorm::RMSNORM_BWD_DW_REDUCE_F32, arch)?;
        let bwd_dw_partial_f32 = device
            .context()
            .load_module(bwd_dw_partial_ptx)?
            .load_function("rmsnorm_bwd_dw_partial_f32")?;
        let bwd_dw_reduce_f32 = device
            .context()
            .load_module(bwd_dw_reduce_ptx)?
            .load_function("rmsnorm_bwd_dw_reduce_f32")?;

        let smem_per_block_budget_bytes = read_clamped_smem_budget_bytes(device)?;

        let raw_smem_per_sm = device.context().attribute(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        )?;
        // 属性取得失敗（`TryFrom` の負値検出）は `InvalidRmsNormShape`
        // （入力テンソルの shape/eps 起因のホスト側検証エラー）ではなく
        // `InvalidKernelDescriptor` を使う。`gemm_auto.rs::
        // read_clamped_smem_budget_bytes` が同種のデバイス属性負値検出に
        // 使う variant と揃え、`CudaError::Display` の意味論的な誤表示を
        // 避ける（cursor[bot] 指摘・PR #706 レビュー r3793478993）。
        let smem_per_sm_budget_bytes =
            u64::try_from(raw_smem_per_sm).map_err(|_| CudaError::InvalidKernelDescriptor {
                detail: format!(
                    "CudaRmsNorm::new: CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR \
                     returned a negative value ({raw_smem_per_sm}), which cannot be a valid \
                     SMEM budget"
                ),
            })?;

        let raw_sm_count = device
            .context()
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        let sm_count =
            u32::try_from(raw_sm_count).map_err(|_| CudaError::InvalidKernelDescriptor {
                detail: format!(
                    "CudaRmsNorm::new: CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT returned a \
                     negative value ({raw_sm_count}), which cannot be a valid SM count"
                ),
            })?;

        Ok(Self {
            stream: device.stream().clone(),
            onepass_f32,
            twopass_f32,
            bwd_dx_f32,
            bwd_dw_f32,
            bwd_dw_partial_f32,
            bwd_dw_reduce_f32,
            smem_per_block_budget_bytes,
            smem_per_sm_budget_bytes,
            sm_count,
        })
    }

    /// 標準 RMSNorm（mean 正規化あり）: `out = x * rsqrt(mean(x^2, axis=-1)
    /// + eps) * w`（`w` が `None` の場合は乗算をスキップ）を実行する。
    ///
    /// `inv_n = 1/hidden` を内部で導出し `Self::run_rmsnorm_f32_raw`
    /// （`hidden == 0` の早期 return より後に呼ぶため `1/hidden` のゼロ
    /// 除算は起きない）へ委譲する。`ops.rs::CudaBackendOps::run_fused` は
    /// canonical プランの意味論（mean 化しない・`x * rsqrt(sum(x^2))`）に
    /// 厳密一致させるため本メソッドを経由せず `Self::run_rmsnorm_f32_raw`
    /// を `inv_n = 1.0` で直接呼ぶ（`ops.rs` ドキュメンテーションコメント
    /// 参照）。
    pub fn run_rmsnorm_f32(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, CudaError> {
        if rows == 0 || hidden == 0 {
            // `validate_rmsnorm_launch` 経由の検証は `run_rmsnorm_f32_raw`
            // 側で行う（0 要素契約の早期 return も含め、二重の入口を
            // 一本化する）。
            return self.run_rmsnorm_f32_raw(x, w, eps, 1.0, rows, hidden);
        }
        let inv_n = 1.0f32 / hidden as f32;
        self.run_rmsnorm_f32_raw(x, w, eps, inv_n, rows, hidden)
    }

    /// `out = x * rsqrt(sum(x^2, axis=-1) * inv_n + eps) * w`（`w` が
    /// `None` の場合は `has_weight = 0` で乗算をスキップ）を実行する
    /// 内部エントリ。`inv_n` を呼び出し元が明示するため、標準 RMSNorm
    /// （[`Self::run_rmsnorm_f32`]・`inv_n = 1/hidden`）と canonical
    /// 融合プラン（`ops.rs::CudaBackendOps::run_fused`・`inv_n = 1.0`。
    /// mean 化しない）の両方の起動元になれる。
    ///
    /// `x` は `[rows, hidden]` の行優先 1 次元化済みバッファ。`rows == 0 ||
    /// hidden == 0` は空結果の早期 return（`elementwise.rs::run_binary` の
    /// 0 要素契約と同じ）。経路選択は [`rmsnorm_route`]（構築時に保持した
    /// `smem_per_block_budget_bytes` を使用）、persistent grid は
    /// [`derive_persistent_grid_one_pass`]／[`derive_persistent_grid_two_pass`]
    /// が導出する。
    pub(crate) fn run_rmsnorm_f32_raw(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        eps: f32,
        inv_n: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let shape = RmsNormShape { rows, hidden };
        let (out, _rstd) = self.run_rmsnorm_f32_inner(x, w, eps, inv_n, shape, false)?;
        Ok(out)
    }

    /// 学習用エントリ（イシュー #596）: 順伝播 `out` に加え、逆伝播が
    /// 再計算に必要とする行あたりスカラー `rstd`（`len == rows`）を
    /// 併せて返す。`ops.rs::run_fused` 経由の推論専用プランは
    /// `Self::run_rmsnorm_f32_raw`（`save_rstd = 0`・スカラーすら
    /// 書かない）を使い続けるため、本メソッドは学習ループ（autodiff の
    /// `Var::rmsnorm` 相当。#596 スコープでは backend-cuda 単体 API として
    /// 提供する）の呼び出し元にのみ関わる。
    pub fn run_rmsnorm_f32_train(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
        let inv_n = if hidden == 0 {
            1.0f32
        } else {
            1.0f32 / hidden as f32
        };
        let shape = RmsNormShape { rows, hidden };
        let (out, rstd) = self.run_rmsnorm_f32_inner(x, w, eps, inv_n, shape, true)?;
        // `save_rstd = true` 経路は `rows == 0 || hidden == 0` の早期
        // return でも `Some(Vec::new())` を返す契約（`run_rmsnorm_f32_inner`
        // 参照）ため `unwrap_or_default` で空 Vec に正規化する。
        Ok((out, rstd.unwrap_or_default()))
    }

    /// [`Self::run_rmsnorm_f32_raw`]／[`Self::run_rmsnorm_f32_train`] の
    /// 共通実装。`save_rstd` により学習経路（`rstd_out` へ行あたり 1
    /// スカラーを書く）と推論経路（一切書かない）を切り替える
    /// （`kernels_rmsnorm.rs` の `save_rstd` 契約と 1:1 対応）。
    fn run_rmsnorm_f32_inner(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        eps: f32,
        inv_n: f32,
        shape: RmsNormShape,
        save_rstd: bool,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), CudaError> {
        let RmsNormShape { rows, hidden } = shape;
        validate_rmsnorm_launch(rows, hidden, x.len(), w.map(|s| s.len()), eps)?;

        if rows == 0 || hidden == 0 {
            // `save_rstd == true`（学習経路）の契約は「`rstd.len() ==
            // rows`」（[`Self::run_rmsnorm_f32_train`] ドキュメンテーション
            // コメント）であり、`hidden == 0` でもこれを維持しなければ
            // ならない。`hidden == 0` の行は要素を持たないため
            // `sum(x^2) == 0` が数学的に確定し、`inv_n` の値に関わらず
            // `rstd = rsqrt(0 * inv_n + eps) = rsqrt(eps)`（全行同一値）が
            // 一意に定まる（`eps` は `validate_rmsnorm_launch` で有限・
            // 非負を検証済みのため `eps.sqrt()` は NaN 化しない。`eps ==
            // 0.0` は `rsqrt(0) == inf` になるが有限性検証の対象外＝許容
            // 済みの退化値）。`rows == 0` の場合は `vec![_; 0]` が自然に
            // 空になるため分岐不要。Cursor Bugbot 指摘（PR #711 レビュー
            // r3794159146）: この早期 return が常に空 Vec を返すと、
            // `run_rmsnorm_f32_train` 経由で `hidden == 0 && rows > 0` を
            // 逆伝播へ渡した際 `validate_rmsnorm_backward_launch` の
            // `rstd.len() == rows` 検証が `hidden == 0` の空判定より先に
            // 失敗してしまう。
            let rstd = if save_rstd {
                let degenerate_rstd = 1.0f32 / eps.sqrt();
                Some(vec![degenerate_rstd; rows])
            } else {
                None
            };
            return Ok((Vec::new(), rstd));
        }

        let route = rmsnorm_route(hidden, self.smem_per_block_budget_bytes);

        let x_dev = self.stream.clone_htod(x)?;
        // `w` が `None` の場合もカーネル引数としてポインタは必要だが
        // `has_weight == 0` により決してデリファレンスされない
        // （`kernels_rmsnorm.rs` 参照）。ダミーとして 1 要素のゼロ初期化
        // バッファを渡す（`0` 要素バッファの確保を一部 CUDA driver が
        // 拒否しうる問題を避ける。`elementwise.rs` の `numel == 0` 早期
        // return と同じ理由）。
        let (w_dev, has_weight) = match w {
            Some(w_slice) => (self.stream.clone_htod(w_slice)?, 1i32),
            None => (self.stream.alloc_zeros::<f32>(1)?, 0i32),
        };
        let mut out_dev = self.stream.alloc_zeros::<f32>(x.len())?;
        // `save_rstd == false`（推論経路）でもカーネル引数としてポインタは
        // 必要だが `save_rstd == 0` により決してデリファレンスされない
        // （`w_dev` ダミーと同じイディオム）。学習経路は `rows` 要素を
        // 確保する。
        let mut rstd_dev = if save_rstd {
            self.stream.alloc_zeros::<f32>(rows)?
        } else {
            self.stream.alloc_zeros::<f32>(1)?
        };
        let save_rstd_i = save_rstd as i32;

        let rows_i = rows as i32;
        let hidden_i = hidden as i32;

        let (func, cfg): (&CudaFunction, LaunchConfig) = match route {
            RmsNormRoute::OnePassSmem => {
                // `derive_persistent_grid_one_pass` の `smem_bytes_per_block`
                // 引数は「1 ブロックが実際に確保する SMEM バイト数」を
                // 受け取る契約（同関数のドキュメンテーションコメント
                // 参照）。予算上限（`self.smem_per_block_budget_bytes`。
                // 通常 48KiB）をそのまま渡すと、行サイズが予算より小さい
                // 一般的な hidden で `blocks_per_sm` が過小評価され、
                // 意図した persistent occupancy 上限（16）を大きく下回る
                // （cursor[bot] 指摘・PR #706 レビュー r3793478990）。
                // 経路判定〈`rmsnorm_route`〉は予算上限との比較のままで
                // よいが、grid 導出とカーネル起動の `shared_mem_bytes` は
                // 実際に確保する `hidden * 4` を単一の真実源として揃える。
                let smem_bytes_per_block = (hidden as u64).saturating_mul(4);
                let grid = derive_persistent_grid_one_pass(
                    self.smem_per_sm_budget_bytes,
                    self.sm_count,
                    smem_bytes_per_block,
                    rows_i as u32,
                );
                let shared_mem_bytes = smem_bytes_per_block.min(u32::MAX as u64) as u32;
                (
                    &self.onepass_f32,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (RMSNORM_BLOCK_DIM, 1, 1),
                        shared_mem_bytes,
                    },
                )
            }
            RmsNormRoute::TwoPass => {
                let grid = derive_persistent_grid_two_pass(self.sm_count, rows_i as u32);
                (
                    &self.twopass_f32,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (RMSNORM_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    },
                )
            }
        };

        // SAFETY: カーネル引数（x_dev/w_dev/out_dev/rstd_dev・rows_i/
        // hidden_i・eps/inv_n/has_weight/save_rstd_i）は
        // `validate_rmsnorm_launch` で検証済みの形状と 1:1 対応する
        // デバイスバッファ長・値であり、カーネル内の手動境界チェック
        // （`if (base+3 < hidden)`／グリッドストライド `row < rows`・
        // REQ-8）と合わせて OOB 読み書きが起きない根拠とする。1 パス
        // 経路の `shared_mem_bytes` は `hidden * 4`（実際に確保する SMEM
        // バイト数）であり、`rmsnorm_route` が判定した
        // `smem_per_block_budget_bytes` 以下であることを既に確認済み
        // （経路判定は予算上限との比較、起動は実バイト数という異なる
        // 量を扱うが、`hidden * 4 <= 予算上限` の不変条件により smem
        // 予算超過による起動失敗は起きない）。`rstd_dev` は
        // `save_rstd == true` なら `rows` 要素（カーネル内 `rstd_out[row]`
        // が `row < rows` の範囲でのみ書く）、`false` ならダミー 1 要素
        // （`save_rstd == 0` によりカーネルが一切デリファレンスしない。
        // `w_dev` ダミーと同じ根拠）。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&x_dev)
                .arg(&w_dev)
                .arg(&mut out_dev)
                .arg(&mut rstd_dev)
                .arg(&rows_i)
                .arg(&hidden_i)
                .arg(&eps)
                .arg(&inv_n)
                .arg(&has_weight)
                .arg(&save_rstd_i)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;

        let out = self.stream.clone_dtoh(&out_dev)?;
        let rstd = if save_rstd {
            Some(self.stream.clone_dtoh(&rstd_dev)?)
        } else {
            None
        };
        Ok((out, rstd))
    }

    /// 逆伝播（イシュー #596。recompute-in-backward）: 保存 `rstd`
    /// （[`Self::run_rmsnorm_f32_train`] が返す行あたりスカラー）と生の
    /// `x`／`dy` から `dx`（常に）・`dw`（`w.is_some()` の場合のみ）を
    /// 再計算する。中間の正規化済みテンソルは一切受け取らない
    /// （`kernels_rmsnorm.rs::RMSNORM_BWD_DX_F32`／`RMSNORM_BWD_DW_F32`
    /// ドキュメンテーションコメント参照）。
    ///
    /// `inv_n` は公開引数に含めない。現状の唯一の学習用順伝播
    /// （[`Self::run_rmsnorm_f32_train`]）が `rstd` を常に `inv_n =
    /// 1/hidden`（`hidden == 0` は `1.0` の退化値。ゼロ除算回避）で導出する
    /// ため、本関数も `shape.hidden` から同一式で内部導出する（呼び出し元
    /// が任意の `inv_n` を渡せると、`rstd` の導出時前提と再計算式が食い違い
    /// `validate_rmsnorm_backward_launch` をすり抜けたまま数学的に誤った
    /// `dx` を「正常結果」として返しうる。codex-review P1 指摘・PR #711
    /// レビュー r3794149870）。将来 `ops.rs::run_fused`〈`inv_n = 1.0`〉
    /// 経路の逆伝播を追加する場合は、`inv_n` を復活させるのではなく
    /// 順伝播側の係数選択（`raw` か `train` か）を型で分ける設計とし、
    /// 本関数の暗黙導出を安易に上書きしないこと。
    pub fn run_rmsnorm_bwd_f32(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        dy: &[f32],
        rstd: &[f32],
        shape: RmsNormShape,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), CudaError> {
        self.run_rmsnorm_bwd_f32_inner(x, w, dy, rstd, shape, None)
    }

    /// [`Self::run_rmsnorm_bwd_f32`] のテスト専用フック（イシュー #597）:
    /// dw の split-K 段数（`num_blocks`）を [`derive_dw_split`] の
    /// ヒューリスティクス（`sm_count` 依存で実機ごとに変わりうる）に
    /// 頼らず明示指定し、単段（`num_blocks == 1`）・split-K
    /// （`num_blocks >= 2`）の両経路を決定的に検証できるようにする
    /// （実装計画 §3.3「テスト用に num_blocks を明示指定できる経路」）。
    ///
    /// `pub`（`#[doc(hidden)]`）である都合上、`num_blocks` はホスト側
    /// 呼び出し元が任意に指定できてしまうため、内部の
    /// [`validate_dw_split_launch`]（`num_blocks >= 1`・`<= rows`・部分和
    /// バッファ上限）による fail-closed 検証を必ず経由する
    /// （`.claude/rules/security.md` A03: テストコードだからといって
    /// 検証を省略しない）。
    #[doc(hidden)]
    pub fn run_rmsnorm_bwd_f32_with_forced_dw_split(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        dy: &[f32],
        rstd: &[f32],
        shape: RmsNormShape,
        num_blocks: u32,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), CudaError> {
        self.run_rmsnorm_bwd_f32_inner(x, w, dy, rstd, shape, Some(num_blocks))
    }

    /// [`Self::run_rmsnorm_bwd_f32`]／[`Self::
    /// run_rmsnorm_bwd_f32_with_forced_dw_split`] の共通実装。
    /// `force_dw_num_blocks` が `None` の場合は [`derive_dw_split`] の
    /// ヒューリスティクスへ委ね、`Some(n)` の場合は `n` を
    /// [`validate_dw_split_launch`] で検証したうえで使う。
    fn run_rmsnorm_bwd_f32_inner(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        dy: &[f32],
        rstd: &[f32],
        shape: RmsNormShape,
        force_dw_num_blocks: Option<u32>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), CudaError> {
        let RmsNormShape { rows, hidden } = shape;
        validate_rmsnorm_backward_launch(
            rows,
            hidden,
            x.len(),
            dy.len(),
            rstd.len(),
            w.map(|s| s.len()),
        )?;

        // `run_rmsnorm_f32_train` と同一式（呼び出し元が指定する余地を
        // なくすことで数値契約を強制する。上記ドキュメンテーションコメント
        // 参照）。
        let inv_n = if hidden == 0 {
            1.0f32
        } else {
            1.0f32 / hidden as f32
        };

        if rows == 0 || hidden == 0 {
            // `dx` は `x` と同じ形状（`numel = rows*hidden`）のため常に
            // 空。`dw` は `w` と同じ形状（`len == hidden`）を維持する契約
            // （`validate_rmsnorm_backward_launch` の `w_len == hidden`
            // 検証・CPU 参照実装と揃える）。`rows == 0` かつ `hidden > 0`
            // では「0 行分の勾配和」= 全要素 0 の `hidden` 長ベクトルが
            // 正しい形状（`w` の長さは検証済みで `hidden` と一致）。
            // Cursor Bugbot 指摘（PR #711 レビュー r3794159146）: 従来は
            // 常に空 Vec を返し `w.len() == hidden` の検証・CPU 参照実装と
            // 形状が食い違っていた。
            let dw = w.map(|w_slice| vec![0.0f32; w_slice.len()]);
            return Ok((Vec::new(), dw));
        }

        let x_dev = self.stream.clone_htod(x)?;
        let dy_dev = self.stream.clone_htod(dy)?;
        let rstd_dev = self.stream.clone_htod(rstd)?;
        // `w` ダミーは順伝播と同じイディオム（`has_weight == 0` で
        // カーネルが一切デリファレンスしない）。
        let (w_dev, has_weight) = match w {
            Some(w_slice) => (self.stream.clone_htod(w_slice)?, 1i32),
            None => (self.stream.alloc_zeros::<f32>(1)?, 0i32),
        };
        let mut dx_dev = self.stream.alloc_zeros::<f32>(x.len())?;

        let rows_i = rows as i32;
        let hidden_i = hidden as i32;
        let grid = derive_persistent_grid_two_pass(self.sm_count, rows_i as u32);

        // SAFETY: `validate_rmsnorm_backward_launch` で `x`/`dy`/`rstd`/
        // `w`（指定時）の長さと `rows`/`hidden`（i32 上限）を検証済み。
        // カーネル内の手動境界チェック（グリッドストライド `row < rows`・
        // `base + 3 < hidden`・REQ-8）と合わせて OOB 読み書きが起きない
        // 根拠とする。`RMSNORM_BWD_BLOCK_DIM`（256）は `RMSNORM_BLOCK_DIM`
        // （32・順伝播）と独立のブロック幅であり、grid 導出
        // （`derive_persistent_grid_two_pass`。行数ベースでブロック幅に
        // 依存しない）はそのまま再利用できる（`kernels_rmsnorm.rs::
        // RMSNORM_BWD_BLOCK_DIM` ドキュメンテーションコメント参照）。
        unsafe {
            self.stream
                .launch_builder(&self.bwd_dx_f32)
                .arg(&x_dev)
                .arg(&w_dev)
                .arg(&dy_dev)
                .arg(&rstd_dev)
                .arg(&mut dx_dev)
                .arg(&rows_i)
                .arg(&hidden_i)
                .arg(&inv_n)
                .arg(&has_weight)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (RMSNORM_BWD_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        let dw = if let Some(w_slice) = w {
            let mut dw_dev = self.stream.alloc_zeros::<f32>(w_slice.len())?;
            let auto_num_blocks = derive_dw_split(self.sm_count, rows_i as u32, hidden_i as u32);
            let num_blocks = force_dw_num_blocks.unwrap_or(auto_num_blocks);
            // `force_dw_num_blocks`（テストフック経由）はホスト境界から任意値
            // （`0` を含む）が来うるため、分岐（単段 `num_blocks <= 1` か
            // split-K か）で検証有無が変わらないよう、分岐前に必ず検証する
            // （fail-closed。security.md A03。codex-review P2 指摘・PR #716:
            // 旧実装は `num_blocks <= 1` 分岐内で検証しておらず
            // `Some(0)` が禁止値のまま単段カーネルへ通っていた）。
            // `derive_dw_split` によるヒューリスティクス由来（`force_dw_
            // num_blocks == None`）は常に `>= 1` を返す契約のため、検証は
            // `force_dw_num_blocks.is_some()` の場合のみで十分だが、
            // 検証コスト自体が軽量なため分岐なく常に通す（境界条件の
            // 実装ドリフトを避ける）。
            validate_dw_split_launch(rows, hidden, num_blocks)?;
            let col_grid = derive_persistent_grid_dw(self.sm_count, hidden_i as u32);

            if num_blocks <= 1 {
                // 単段フォールバック（小規模形状で余分なカーネル起動・
                // 部分和バッファを避ける。`derive_dw_split` ドキュメン
                // テーションコメント参照。実装計画 §3.3）。
                //
                // SAFETY: dw カーネルは列方向 grid-stride（`i < hidden`）で
                // 起動する。`x`/`dy` の長さは `rows*hidden` と検証済み
                // （`validate_rmsnorm_backward_launch`）であり、行方向は
                // カーネル内ループ条件 `row < rows` で境界保証される
                // （REQ-8）。
                unsafe {
                    self.stream
                        .launch_builder(&self.bwd_dw_f32)
                        .arg(&x_dev)
                        .arg(&dy_dev)
                        .arg(&rstd_dev)
                        .arg(&mut dw_dev)
                        .arg(&rows_i)
                        .arg(&hidden_i)
                        .launch(LaunchConfig {
                            grid_dim: (col_grid, 1, 1),
                            block_dim: (RMSNORM_BWD_DW_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
            } else {
                // split-K 二段リダクション（イシュー #597）。`num_blocks` の
                // fail-closed 検証は分岐前（上記 `validate_dw_split_launch`
                // 呼び出し）で完了済み。
                let num_blocks_i = num_blocks as i32;

                let partial_len = (num_blocks as usize).checked_mul(hidden).ok_or_else(|| {
                    CudaError::InvalidRmsNormShape {
                        detail: format!(
                            "rmsnorm dw split partial buffer length overflowed usize: \
                             num_blocks={num_blocks}, hidden={hidden}"
                        ),
                    }
                })?;
                let mut dw_partial_dev = self.stream.alloc_zeros::<f32>(partial_len)?;

                // SAFETY（第 1 カーネル）: `validate_dw_split_launch` で
                // `num_blocks` が `[1, rows]` かつ部分和バッファが上限内で
                // あることを検証済み。`x`/`dy`/`rstd` の長さは
                // `validate_rmsnorm_backward_launch` で検証済み。カーネル内
                // の手動境界チェック（`i < hidden`・`row < row_end` かつ
                // `row_end <= rows`・REQ-8）と合わせて OOB 読み書きが
                // 起きない根拠とする。`dw_partial` は `num_blocks * hidden`
                // 要素で `blockIdx.y = b` が一意に `[b*hidden, (b+1)*hidden)`
                // 範囲へのみ書くため、CTA 間の書き込み競合は起きない
                // （atomics 不使用でも決定的）。
                unsafe {
                    self.stream
                        .launch_builder(&self.bwd_dw_partial_f32)
                        .arg(&x_dev)
                        .arg(&dy_dev)
                        .arg(&rstd_dev)
                        .arg(&mut dw_partial_dev)
                        .arg(&rows_i)
                        .arg(&hidden_i)
                        .arg(&num_blocks_i)
                        .launch(LaunchConfig {
                            grid_dim: (col_grid, num_blocks, 1),
                            block_dim: (RMSNORM_BWD_DW_PARTIAL_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }

                // SAFETY（第 2 カーネル）: `dw_partial_dev` は上記カーネルが
                // `num_blocks * hidden` 要素を無条件に埋める契約
                // （`kernels_rmsnorm::RMSNORM_BWD_DW_PARTIAL_F32` ドキュメン
                // テーションコメント「末尾要素ブロックの扱い」参照）。両
                // カーネルは同一 `self.stream` 上へ順に enqueue されるため、
                // stream 順序保証により第 2 カーネルは第 1 カーネルの完了後
                // にのみ `dw_partial_dev` を読む（明示的な追加同期は不要）。
                // `dw_partial_dev`（デバイスバッファのバインディング）は
                // この `unsafe` ブロックと下の `synchronize` の両方を
                // 包む本スコープの終わりまで生存するため、カーネル実行中に
                // Rust 側で先に drop され解放されることはない。
                // `RMSNORM_DW_REDUCE_BLOCK_DIM` は縮約カーネルの静的 smem
                // 配列サイズ（`kernels_rmsnorm.rs` 参照）と一致する契約の
                // ためブロック幅を固定で渡す（`RMSNORM_BWD_DW_BLOCK_DIM`
                // 〈単段〉とは独立の値だが現状同じ 256）。
                unsafe {
                    self.stream
                        .launch_builder(&self.bwd_dw_reduce_f32)
                        .arg(&dw_partial_dev)
                        .arg(&mut dw_dev)
                        .arg(&hidden_i)
                        .arg(&num_blocks_i)
                        .launch(LaunchConfig {
                            grid_dim: (col_grid, 1, 1),
                            block_dim: (RMSNORM_DW_REDUCE_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
            }

            self.stream.synchronize()?;
            Some(self.stream.clone_dtoh(&dw_dev)?)
        } else {
            self.stream.synchronize()?;
            None
        };

        let dx = self.stream.clone_dtoh(&dx_dev)?;
        Ok((dx, dw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- derive_persistent_grid_dw ---

    #[test]
    fn derive_persistent_grid_dw_uses_16_per_sm_clamped_by_blocks_needed() {
        // hidden=100000, block=256 -> blocks_needed = ceil(100000/256) = 391。
        // sm_count=4 -> grid候補 = 64 < 391 なので 64。
        assert_eq!(derive_persistent_grid_dw(4, 100_000), 64);
    }

    #[test]
    fn derive_persistent_grid_dw_clamps_at_blocks_needed_for_small_hidden() {
        // hidden=8, block=256 -> blocks_needed = 1。
        assert_eq!(derive_persistent_grid_dw(100, 8), 1);
    }

    #[test]
    fn derive_persistent_grid_dw_hidden_zero_is_failsafe_one() {
        assert_eq!(derive_persistent_grid_dw(4, 0), 1);
    }

    // --- validate_rmsnorm_backward_launch ---

    #[test]
    fn validate_rmsnorm_backward_launch_accepts_matching_dims() {
        assert!(validate_rmsnorm_backward_launch(3, 8, 24, 24, 3, Some(8)).is_ok());
        assert!(validate_rmsnorm_backward_launch(3, 8, 24, 24, 3, None).is_ok());
    }

    #[test]
    fn validate_rmsnorm_backward_launch_rejects_x_len_mismatch() {
        let err = validate_rmsnorm_backward_launch(3, 8, 23, 24, 3, None).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_backward_launch_rejects_dy_len_mismatch() {
        let err = validate_rmsnorm_backward_launch(3, 8, 24, 23, 3, None).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_backward_launch_rejects_rstd_len_mismatch() {
        let err = validate_rmsnorm_backward_launch(3, 8, 24, 24, 2, None).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_backward_launch_rejects_w_len_mismatch() {
        let err = validate_rmsnorm_backward_launch(3, 8, 24, 24, 3, Some(7)).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_backward_launch_rejects_dims_exceeding_i32_max() {
        let err = validate_rmsnorm_backward_launch(
            i32::MAX as usize + 1,
            1,
            i32::MAX as usize + 1,
            i32::MAX as usize + 1,
            i32::MAX as usize + 1,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    // --- rmsnorm_route ---

    #[test]
    fn rmsnorm_route_selects_onepass_when_exactly_at_budget() {
        // hidden=1024 の f32 行 = 4096 バイト。予算ちょうど。
        assert_eq!(rmsnorm_route(1024, 4096), RmsNormRoute::OnePassSmem);
    }

    #[test]
    fn rmsnorm_route_selects_twopass_when_one_byte_over_budget() {
        assert_eq!(rmsnorm_route(1024, 4095), RmsNormRoute::TwoPass);
    }

    #[test]
    fn rmsnorm_route_selects_onepass_when_one_byte_under_budget() {
        assert_eq!(rmsnorm_route(1023, 4096), RmsNormRoute::OnePassSmem);
    }

    #[test]
    fn rmsnorm_route_selects_twopass_for_large_hidden() {
        // 16384 * 4 = 65536 > 49152（既定 per-block 上限）。
        assert_eq!(rmsnorm_route(16384, 49152), RmsNormRoute::TwoPass);
    }

    #[test]
    fn rmsnorm_route_selects_onepass_for_zero_row_len() {
        assert_eq!(rmsnorm_route(0, 0), RmsNormRoute::OnePassSmem);
    }

    // --- derive_persistent_grid_one_pass ---

    #[test]
    fn derive_persistent_grid_one_pass_clamps_blocks_per_sm_at_16() {
        // smem_per_sm=1MiB, smem_per_block=1B → blocks_per_sm は
        // 極端に大きい値になるはずだが 16 でクランプされる。
        let grid = derive_persistent_grid_one_pass(1024 * 1024, 4, 1, 1_000_000);
        assert_eq!(grid, 4 * 16);
    }

    #[test]
    fn derive_persistent_grid_one_pass_clamps_grid_at_rows() {
        let grid = derive_persistent_grid_one_pass(1024 * 1024, 100, 1, 10);
        assert_eq!(grid, 10);
    }

    #[test]
    fn derive_persistent_grid_one_pass_returns_at_least_one() {
        let grid = derive_persistent_grid_one_pass(0, 0, 4096, 5);
        assert_eq!(grid, 1);
    }

    #[test]
    fn derive_persistent_grid_one_pass_zero_smem_per_block_uses_cap() {
        // hidden==0（縮退）で smem_bytes_per_block==0 の場合、SMEM
        // 制約が事実上存在しないため 2 パスと同じ上限 16 を使う。
        let grid = derive_persistent_grid_one_pass(0, 4, 0, 1000);
        assert_eq!(grid, 4 * 16);
    }

    #[test]
    fn derive_persistent_grid_one_pass_rows_zero_is_failsafe_one() {
        assert_eq!(derive_persistent_grid_one_pass(1024, 4, 4, 0), 1);
    }

    // --- derive_persistent_grid_two_pass ---

    #[test]
    fn derive_persistent_grid_two_pass_uses_16_per_sm() {
        assert_eq!(derive_persistent_grid_two_pass(4, 1_000_000), 4 * 16);
    }

    #[test]
    fn derive_persistent_grid_two_pass_clamps_at_rows() {
        assert_eq!(derive_persistent_grid_two_pass(100, 3), 3);
    }

    #[test]
    fn derive_persistent_grid_two_pass_rows_zero_is_failsafe_one() {
        assert_eq!(derive_persistent_grid_two_pass(4, 0), 1);
    }

    // --- validate_rmsnorm_launch ---

    #[test]
    fn validate_rmsnorm_launch_accepts_matching_dims() {
        assert!(validate_rmsnorm_launch(3, 8, 24, Some(8), 1e-5).is_ok());
        assert!(validate_rmsnorm_launch(3, 8, 24, None, 0.0).is_ok());
    }

    #[test]
    fn validate_rmsnorm_launch_rejects_x_len_mismatch() {
        let err = validate_rmsnorm_launch(3, 8, 23, None, 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_launch_rejects_w_len_mismatch() {
        let err = validate_rmsnorm_launch(3, 8, 24, Some(7), 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_launch_rejects_non_finite_eps() {
        let err = validate_rmsnorm_launch(3, 8, 24, None, f32::NAN).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
        let err = validate_rmsnorm_launch(3, 8, 24, None, f32::INFINITY).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_launch_rejects_negative_eps() {
        // 負の eps は `sum(x^2) * inv_n + eps` を負化しうるため fail-closed
        // で拒否する（codex-review 指摘・PR #706 レビュー r3793473250）。
        let err = validate_rmsnorm_launch(3, 8, 24, None, -1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
        let err = validate_rmsnorm_launch(3, 8, 24, None, -0.0001).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_launch_accepts_zero_eps() {
        // `run_fused` 経由の canonical プラン起動は eps=0.0 を渡す契約
        // （`ops.rs` ドキュメンテーションコメント参照）。0.0 は有限値。
        assert!(validate_rmsnorm_launch(3, 8, 24, None, 0.0).is_ok());
    }

    #[test]
    fn validate_rmsnorm_launch_rejects_dims_exceeding_i32_max() {
        let err =
            validate_rmsnorm_launch(i32::MAX as usize + 1, 1, i32::MAX as usize + 1, None, 1e-5)
                .unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_launch_accepts_hidden_at_i32_max_boundary() {
        // `hidden == i32::MAX`（許容上限ちょうど）は受理される契約
        // （`kernels_rmsnorm.rs` 冒頭コメント「ループ添字のオーバーフロー
        // 安全性」参照）。実機でこの `hidden` を実行するには行あたり
        // 約 8 GiB のバッファが要るため実行までは検証しないが、ホスト側
        // 検証がこの境界を「拒否しすぎていない」ことのみ回帰確認する
        // （`validate_rmsnorm_launch_rejects_dims_exceeding_i32_max` の
        // 対となる境界値ケース）。
        assert!(
            validate_rmsnorm_launch(1, i32::MAX as usize, i32::MAX as usize, None, 1e-5).is_ok()
        );
    }

    // --- match_rmsnorm_plan ---
    //
    // `fandhe_ai_tensor_core::fusion::graph`／`detect` は `tensor-core` 内部限定の
    // `pub(crate)`（`fusion/mod.rs` 冒頭コメント「配置は `tensor-core` の
    // 1 か所に閉じる」参照）で `backend-cuda` からは参照できないため、
    // ここでは `autodiff` と同じ構築経路（[`FusionPlan::from_ops`]。`pub` +
    // `#[doc(hidden)]`）で直接 canonical プランを組み立てる
    // （`plan.rs` モジュール冒頭コメント「`FusionPlan::from_ops`
    // （`pub` + `#[doc(hidden)]`）: `autodiff` クレート専用の構築経路」
    // 参照）。

    fn build_canonical_rmsnorm_plan(hidden: usize) -> FusionPlan {
        // leaf 0=x, 1=sq(Mul(0,0)), 2=sum(Sum{axis:None}(1)), 3=rsqrt(2),
        // 4=bc(Broadcast{axis:None}(3)), 5=out(Mul(4,0))
        // （`plan.rs::from_segment_builds_rmsnorm_plan_with_row_fusion_metadata`
        // と同型の op 列）。
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
        FusionPlan::from_ops(ops, vec![hidden], fandhe_ai_tensor_core::DType::F32, 1).unwrap()
    }

    #[test]
    fn match_rmsnorm_plan_accepts_canonical_plan() {
        let plan = build_canonical_rmsnorm_plan(8);
        assert_eq!(match_rmsnorm_plan(&plan), Some(8));
    }

    #[test]
    fn match_rmsnorm_plan_rejects_softmax_shaped_plan() {
        // softmax: max → sub(broadcast) → exp → sum → div(broadcast)
        // （RMSNorm と異なる op 列・leaf_count）。
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
        let plan =
            FusionPlan::from_ops(ops, vec![2, 8], fandhe_ai_tensor_core::DType::F32, 1).unwrap();
        assert_eq!(match_rmsnorm_plan(&plan), None);
    }

    #[test]
    fn match_rmsnorm_plan_rejects_elementwise_only_plan() {
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Input { leaf_index: 1 },
            FusedOpKind::Add { lhs: 0, rhs: 1 },
        ];
        let plan =
            FusionPlan::from_ops(ops, vec![4], fandhe_ai_tensor_core::DType::F32, 2).unwrap();
        assert_eq!(match_rmsnorm_plan(&plan), None);
    }

    #[test]
    fn match_rmsnorm_plan_rejects_row_axis_variant() {
        // 行方向（axis: Some(a)）の RMSNorm 型プランは対象外
        // （`match_rmsnorm_plan` ドキュメンテーションコメント参照）。
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Mul { lhs: 0, rhs: 0 },
            FusedOpKind::Sum {
                input: 1,
                axis: Some(1),
            },
            FusedOpKind::Rsqrt { input: 2 },
            FusedOpKind::Broadcast {
                input: 3,
                axis: Some(1),
            },
            FusedOpKind::Mul { lhs: 4, rhs: 0 },
        ];
        let plan =
            FusionPlan::from_ops(ops, vec![2, 8], fandhe_ai_tensor_core::DType::F32, 1).unwrap();
        assert_eq!(match_rmsnorm_plan(&plan), None);
    }

    // --- derive_dw_split（イシュー #597） ---

    #[test]
    fn derive_dw_split_returns_one_for_zero_rows_or_hidden() {
        assert_eq!(derive_dw_split(4, 0, 8), 1);
        assert_eq!(derive_dw_split(4, 8, 0), 1);
    }

    #[test]
    fn derive_dw_split_splits_when_hidden_small_and_rows_large() {
        // hidden=64（列タイル 1）・rows=10000（十分な行数）・sm_count=4 →
        // target=64・col_tiles=1・raw=64。rows_cap=10000/32=312、
        // MAX_SPLIT=64 でクランプされ num_blocks=64（部分和バッファ
        // 64*64*4=16KiB は上限内）。
        assert_eq!(derive_dw_split(4, 10_000, 64), 64);
    }

    #[test]
    fn derive_dw_split_stays_single_stage_when_hidden_saturates_occupancy() {
        // advisor 指摘: 「rows が大きければ必ず num_blocks >= 2」は
        // hidden 依存で偽になる（列タイル数だけで occupancy が確保できる
        // 形状では split-K の余地がない）。hidden=100000 → col_tiles=391 >
        // sm_count*16=64 のため raw_num_blocks は 1 未満に切り捨たり
        // `.max(1)` で 1 に floor される → 単段（1）を返す。
        assert_eq!(derive_dw_split(4, 10_000_000, 100_000), 1);
    }

    #[test]
    fn derive_dw_split_stays_single_stage_when_rows_below_min_rows_per_block() {
        // rows=10 < RMSNORM_DW_MIN_ROWS_PER_BLOCK（32）→ rows_cap が 1 に
        // floor され split-K の余地がない。
        assert_eq!(derive_dw_split(4, 10, 8), 1);
    }

    #[test]
    fn derive_dw_split_never_exceeds_max_split_const() {
        // sm_count を極端に大きくしても RMSNORM_DW_MAX_SPLIT（64）で
        // クランプされる。
        assert_eq!(
            derive_dw_split(1_000_000, 1_000_000, 8),
            RMSNORM_DW_MAX_SPLIT
        );
    }

    #[test]
    fn derive_dw_split_falls_back_when_partial_buffer_exceeds_cap() {
        // sm_count=20000・rows=100000・hidden=1000000 では
        // col_tiles=ceil(1000000/256)=3907・target=320000・
        // raw_num_blocks=320000/3907=81 → rows_cap（100000/32=3125）・
        // MAX_SPLIT（64）でクランプされ num_blocks 候補は 64。しかし
        // 64*1000000*4=256,000,000 bytes（約 244 MiB）は
        // RMSNORM_DW_PARTIAL_BUFFER_CAP_BYTES（64 MiB）を超えるため、
        // 上限に収まる最大値（floor(64 MiB / (hidden*4)) = 16）まで
        // 単調減少する。
        let num_blocks = derive_dw_split(20_000, 100_000, 1_000_000);
        assert_eq!(num_blocks, 16);
        let bytes = (num_blocks as u64) * 1_000_000 * 4;
        assert!(bytes <= RMSNORM_DW_PARTIAL_BUFFER_CAP_BYTES);
        // 17 block では上限を超える（= 16 が「収まる最大値」であることの
        // 根拠。`clippy::assertions_on_constants` を避けるため、両辺の
        // 定数を変数へ束縛してから比較する）。
        let bytes_at_17 = 17u64 * 1_000_000 * 4;
        assert!(bytes_at_17 > RMSNORM_DW_PARTIAL_BUFFER_CAP_BYTES);
    }

    // --- dw_split_row_range（イシュー #597） ---

    #[test]
    fn dw_split_row_range_zero_num_blocks_is_empty() {
        assert_eq!(dw_split_row_range(100, 0, 0), (0, 0));
    }

    /// カーネル [`kernels_rmsnorm::RMSNORM_BWD_DW_PARTIAL_F32`] の行分割
    /// 契約（ギャップなし・重複なしで `0..rows` を分割する）を、複数の
    /// `(rows, num_blocks)` 組み合わせで検証する（advisor 指摘: 実機
    /// なしで検証できる唯一の CI 到達可能な保証点）。
    #[test]
    fn dw_split_row_range_partitions_rows_without_gaps_or_overlap() {
        let cases: &[(u64, u32)] = &[
            (100, 1),
            (100, 4),
            (10, 6), // rows_per_block=2, 末尾 block(5) は空範囲になる
            (7, 3),  // rows_per_block=3, 末尾 block(2) は 1 行のみ
            (1, 5),  // rows < num_blocks（呼び出し元は起こさない契約だが
            // 分割ロジック自体はここでも矛盾しないことを確認する）
            (1_000_003, 64),
        ];
        for &(rows, num_blocks) in cases {
            let mut prev_end = 0u64;
            let mut covered_all = false;
            for b in 0..num_blocks {
                let (start, end) = dw_split_row_range(rows, num_blocks, b);
                if start >= end {
                    // 空範囲に達したら、それ以前の非空範囲の合計が
                    // 既に rows 全体を覆っている必要がある。
                    assert_eq!(
                        prev_end, rows,
                        "空範囲に達する前に 0..rows を覆い切れていない: \
                         rows={rows}, num_blocks={num_blocks}, b={b}"
                    );
                    covered_all = true;
                } else {
                    assert_eq!(
                        start, prev_end,
                        "ギャップまたは重複を検出: rows={rows}, num_blocks={num_blocks}, b={b}"
                    );
                    assert!(end <= rows);
                    prev_end = end;
                }
            }
            assert!(
                covered_all || prev_end == rows,
                "0..rows を覆い切れていない: rows={rows}, num_blocks={num_blocks}, \
                 prev_end={prev_end}"
            );
        }
    }

    // --- validate_dw_split_launch（イシュー #597） ---

    #[test]
    fn validate_dw_split_launch_accepts_valid_split() {
        assert!(validate_dw_split_launch(1000, 64, 8).is_ok());
    }

    #[test]
    fn validate_dw_split_launch_rejects_zero_num_blocks() {
        let err = validate_dw_split_launch(1000, 64, 0).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_dw_split_launch_rejects_num_blocks_exceeding_rows() {
        let err = validate_dw_split_launch(4, 64, 5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_dw_split_launch_accepts_num_blocks_equal_to_rows() {
        assert!(validate_dw_split_launch(4, 64, 4).is_ok());
    }

    #[test]
    fn validate_dw_split_launch_rejects_partial_buffer_exceeding_cap() {
        // 64 * 300_000_000 * 4 bytes は明らかに 64 MiB を超える
        // （テストフック `run_rmsnorm_bwd_f32_with_forced_dw_split` が
        // 任意の `num_blocks` を渡せることに対する fail-closed 検証。
        // advisor 指摘・security.md A03）。
        let err = validate_dw_split_launch(1_000_000_000, 300_000_000, 64).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_dw_split_launch_accepts_single_stage_even_if_hidden_alone_exceeds_cap() {
        // cursor[bot] 指摘（PR #716）の回帰テスト: `num_blocks == 1`
        // （単段フォールバック）は部分和バッファを一切確保しないため、
        // `hidden * 4` 単体が cap（64 MiB = 16_777_216 要素）を超える
        // 大きな `hidden` でも拒否してはならない。`derive_dw_split` は
        // このような形状では正しく `1` を返す契約（本テストは呼び出し元
        // の分岐に依らず `validate_dw_split_launch` 単体で保証する）。
        let hidden = 20_000_000; // hidden * 4 bytes ≈ 76 MiB > 64 MiB cap
        assert!(validate_dw_split_launch(1, hidden, 1).is_ok());
    }
}
