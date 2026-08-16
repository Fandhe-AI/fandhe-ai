//! naive／tiled／WMMA(TF32) GEMM の起動 API（NVRTC コンパイル・保持・実行）。
//!
//! `CudaGemm` は `crates/tensor-core` の演算グラフ実行（TASK-1.9・#43 で
//! `BackendOps` へ結線予定）から見て、「`CudaDevice` を渡すと naive／tiled／
//! WMMA(TF32) GEMM カーネル（naive/tiled は f32/f16 各 2 種、WMMA(TF32) は
//! f32 1 種、計 5 カーネル）をコンパイル・保持し、以降はホスト側スライスを
//! 渡すだけで GPU 実行できる」境界を担う。カーネルソース自体は
//! `kernels.rs`（NVRTC 文字列埋め込み）に閉じ込め、本モジュールは
//! コンパイル結果（`CudaFunction`）の保持とメモリ転送・起動手続きのみを扱う。
//! WMMA(TF32) 経路（TASK-11.1c・#62）は naive/tiled の 4 カーネルと異なる
//! ブロック次元契約（[`WMMA_TF32_BLOCK_DIM`]・[`wmma_tf32_launch_config`]）を
//! 持つため、専用の起動手続き（[`CudaGemm::run_wmma_f32_kernel`]）に分離する。
//!
//! **移植元**: `docs/spec/03-poc/poc-v2-3-cuda-gemm/code/rust/src/cuda/mod.rs`
//! の `CudaGemm::new`／`run_naive_f32`／`run_naive_f16`／`run_tiled_f32`／
//! `run_tiled_f16`。productize にあたり PoC から変更した点:
//!
//! 1. **型付きエラー化**（`.claude/rules/coding-rust.md`）: PoC の
//!    `CudaGemmError(String)` を廃し、`CudaError`（`error.rs`）に統一した。
//! 2. **ホスト側形状検証を追加**: PoC は形状検証を持たず、不整合な
//!    スライス長・オーバーフローする m/n/k をそのままカーネル引数へ渡す
//!    経路が存在した。本実装は GPU 起動前に [`validate_gemm_dims`] で
//!    拒否し `CudaError::InvalidShape` を返す（OWASP A03 対応。
//!    `.claude/rules/security.md`）。tiled 経路はさらに
//!    [`validate_tiled_k_bound`] で `k` の追加上限を検証する（後述）。
//! 3. **`Duration` 非返却**: PoC の `run_*` はカーネル実行時間を計測して
//!    返していたが、計測は `bench-harness` 側の責務であり本クレートの
//!    責務境界外と判断した（TASK-8.x・`bench-harness` の同期方式節参照）。
//! 4. **naive/tiled 共通ヘルパーへの整理**: PoC の `run_f32`/`run_f16`
//!    （`func`・`block_dim` を引数化した private ヘルパー）の構造を踏襲し、
//!    naive/tiled 双方の起動手続き（転送・起動・同期・回収）を
//!    [`CudaGemm::run_f32_kernel`]／[`CudaGemm::run_f16_kernel`] に集約する
//!    （#34 で naive 専用だった手続きを共通化）。

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels;
use crate::kernels_wmma_opt;
use crate::nvrtc::compile_ptx;

/// naive GEMM カーネル起動 1 回あたりのブロック次元（16x16 = 256 スレッド）。
///
/// PoC-v2-3（`cuda/mod.rs:174`）と同じ値を踏襲する。tiled 版の
/// `kernels::TILE`（32x32。[`TILED_BLOCK_DIM`]）とは独立したパラメータであり、
/// 共有メモリを使わない naive カーネルの `__shared__` 配列サイズ制約は受けない。
const NAIVE_BLOCK_DIM: (u32, u32, u32) = (16, 16, 1);

/// tiled GEMM カーネル起動 1 回あたりのブロック次元。
///
/// `kernels::TILE` x `kernels::TILE` に固定する必要がある（カーネル内
/// `__shared__ float as_tile[TILE][TILE]` 等はブロック内スレッド数と
/// 1:1 対応するコンパイル時定数のため、ここがずれるとタイル境界外の
/// スレッドが共有メモリを書かない一方でロード先が欠落し誤った積和になる）。
const TILED_BLOCK_DIM: (u32, u32, u32) = (kernels::TILE, kernels::TILE, 1);

/// WMMA TF32 GEMM カーネル起動 1 回あたりのブロック次元（128 スレッド = 4 warp、
/// `kernels::WMMA_TF32_THREADS` を 1 次元ブロックとして起動する）。
///
/// `kernels::WMMA_TF32_F32` は `blockDim.x`（線形スレッド ID）から warp を
/// `warp_id = tid / 32` で導出し、2x2 warp グリッド（`warp_id / 2`／`warp_id % 2`）
/// へマップする実装のため、ここが `kernels::WMMA_TF32_THREADS` とずれると
/// 一部 warp が欠落し誤った積和・共有メモリロード漏れが起きる（`TILED_BLOCK_DIM`
/// と同じ「ホスト側ブロック次元とカーネル内定数の 1:1 対応」契約）。
const WMMA_TF32_BLOCK_DIM: (u32, u32, u32) = (kernels::WMMA_TF32_THREADS, 1, 1);

/// WMMA TF32 opt（共有メモリ・タイル最適化版。TASK-11.1d・#63）カーネル
/// 起動 1 回あたりのブロック次元（128 スレッド = 4 warp、
/// `kernels_wmma_opt::WMMA_TF32_OPT_THREADS` を 1 次元ブロックとして
/// 起動する）。[`WMMA_TF32_BLOCK_DIM`] と偶然同じ値（128）だが、opt 側は
/// warp あたり fragment 2×2 個（レジスタブロッキング）を担当する点が
/// 基本版と異なる独立した契約であるため、値を共有せず専用定数として
/// 分離する（`kernels_wmma_opt.rs` 冒頭ドキュメントコメント「タイル構成」
/// 参照）。
const WMMA_TF32_OPT_BLOCK_DIM: (u32, u32, u32) = (kernels_wmma_opt::WMMA_TF32_OPT_THREADS, 1, 1);

/// WMMA TF32 opt-staged（cp.async 多段パイプライン・fragment 先読み。
/// イシュー #500）カーネル起動 1 回あたりのブロック次元。
/// [`WMMA_TF32_OPT_BLOCK_DIM`] と同じ理由（`kernels_wmma_opt.rs` 冒頭
/// ドキュメンテーションコメント参照）で専用定数として分離する
/// （`kernels_wmma_opt::WMMA_TF32_STAGED_THREADS` と偶然同じ値 128）。
const WMMA_TF32_STAGED_BLOCK_DIM: (u32, u32, u32) =
    (kernels_wmma_opt::WMMA_TF32_STAGED_THREADS, 1, 1);

/// [`CudaGemm::run_tiled_bias_act_f32`] が実際に GPU カーネルを起動した
/// 回数（イシュー #599）。
///
/// `ops.rs::CudaBackendOps::gemm_bias_act` の経路選択（融合 vs
/// `tensor_core::backend_ops::BackendOps::gemm_bias_act` デフォルト実装の
/// 非融合 3 段合成）が実際に融合カーネルへ到達しているかを、実機なしの
/// 単体テスト（`ops.rs` 内 `#[cfg(test)]`）が検証するための可観測点。
/// テスト専用の計測であり公開 API の意味論・数値契約には一切影響しない
/// （`Relaxed` オーダリングで十分。厳密な happens-before 関係は不要で、
/// 単に「呼ばれたか」を数える）。`m == 0 || n == 0`（no-op）・`k == 0`
/// （ホスト側で直接 epilogue のみ計算し GPU 起動を回避する分岐。
/// [`CudaGemm::run_tiled_bias_act_f32`] 参照）の場合はカーネルを起動しない
/// ためカウントしない。
pub(crate) static BIAS_ACT_FUSED_LAUNCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// naive／tiled GEMM カーネル（f32/f16 各 2 種）のコンパイル済みハンドルを保持する。
///
/// `stream` は [`CudaDevice`] から `Arc` クローンで受け取る（`device.rs` の
/// 共有契約どおり）。`new` 時に 4 カーネルを一括コンパイルするのは、
/// `nvrtc::compile_ptx` の呼び出し契約「`Box::leak` によるアーキテクチャ
/// 文字列リークはデバイスあたり定数回に限る」を守るためであり、
/// `run_naive_*`／`run_tiled_*` 呼び出しのたびに再コンパイルしない。
pub struct CudaGemm {
    stream: Arc<CudaStream>,
    naive_f32: CudaFunction,
    naive_f16: CudaFunction,
    tiled_f32: CudaFunction,
    tiled_f16: CudaFunction,
    /// イシュー #599 で追加。GEMM epilogue（bias 加算・activation）を融合
    /// した tiled GEMM（`kernels::TILED_BIAS_ACT_F32`）のコンパイル済み
    /// ハンドル。`tiled_f32` と異なり `#include` を使わない・compute
    /// capability を問わない点は同じ（naive/tiled 4 カーネルと同様、失敗
    /// しうる環境が広い WMMA(TF32) 系より狭いため `Option` 化しない）。
    tiled_bias_act_f32: CudaFunction,
    /// TASK-11.1c（#62）で追加。WMMA（Tensor Core）を用いた TF32 GEMM
    /// （`kernels::WMMA_TF32_F32`）のコンパイル済みハンドル。#61（f16 WMMA
    /// GEMM）が本 PR 時点で未マージのため、WMMA 共通基盤（NVRTC コンパイル・
    /// 起動 API の骨格）はこのフィールドと共に本イシューで最小実装した
    /// （イシュー #62 実装計画 2 節「安全側の判断」）。
    ///
    /// `Option` にする理由（レビュー指摘 #62）: `WMMA_TF32_F32` は
    /// `#include <mma.h>`（NVRTC の include パス解決が必要）と compute
    /// capability 8.0 以降を要求し、naive/tiled の 4 カーネル（`#include`
    /// を使わず全 compute capability で成立）より失敗しうる環境が広い。
    /// `new` 内でこのコンパイルを `?` により早期 return させると、
    /// WMMA 経路のコンパイル失敗だけで naive/tiled 4 カーネルまで
    /// `CudaGemm::new` ごと使用不能になる回帰を招く。`None` は
    /// コンパイル・ロード失敗を表し、`wmma_tf32_error` に detail を保持
    /// して `run_wmma_tf32` 呼び出し時にのみ `CudaError::WmmaUnavailable`
    /// として表面化させる。
    wmma_tf32: Option<CudaFunction>,
    /// `wmma_tf32` が `None` の場合の失敗理由（`Display` 済み文字列）。
    /// `CudaError`（`Compile`/`Driver` 等）は `Clone` を実装しないため、
    /// `run_wmma_tf32` から再送出できるよう文字列化して保持する。
    wmma_tf32_error: Option<String>,
    /// TASK-11.1d（#63）で追加。共有メモリ・タイル最適化版 WMMA(TF32)
    /// カーネル（`kernels_wmma_opt::wmma_tf32_f32_opt_source()`）のコンパイル済み
    /// ハンドル。`wmma_tf32`（基本版）と同じ理由（`#include <mma.h>` の
    /// include パス解決・compute capability 8.0 以降を要求し、失敗しうる
    /// 環境が広い）で `Option` にし、コンパイル失敗を `new` の早期 return
    /// に合流させない。`run_wmma_tf32` はこちらが `Some` なら優先的に使い、
    /// `None` なら `wmma_tf32`（基本版）へ自動フォールバックする
    /// （`kernels_wmma_opt.rs` 冒頭ドキュメントコメント「公開 API への
    /// 影響」参照）。
    wmma_tf32_opt: Option<CudaFunction>,
    /// `wmma_tf32_opt` が `None` の場合の失敗理由。`wmma_tf32_error` と
    /// 同じ理由で文字列化して保持する（`run_wmma_tf32` は基本版が利用可能な
    /// 限りこの detail を表面化させないが、[`Self::wmma_tf32_opt_error`]
    /// 経由でテスト・呼び出し側が参照できる。基本版も失敗している場合は
    /// `wmma_tf32_error` の detail が `CudaError::WmmaUnavailable` として
    /// 表面化する）。
    wmma_tf32_opt_error: Option<String>,
    /// イシュー #500 で追加。`kernels_wmma_opt::wmma_tf32_f32_staged_source()`
    /// （cp.async 多段パイプライン・fragment 先読みを WMMA TF32 経路へ
    /// 横展開したカーネル）のコンパイル済みハンドル。`wmma_tf32_opt` と
    /// 同じ理由（`#include <mma.h>` の解決・compute capability 8.0 以降
    /// 要求）で `Option` にし、コンパイル失敗を `new` の早期 return に
    /// 合流させない。`run_wmma_tf32` はこちらが `Some` かつ cp.async 16
    /// バイト整列条件（`n % 4 == 0 && k % 4 == 0`）を満たす場合に最優先で
    /// 使い、いずれかが成立しなければ `wmma_tf32_opt`（既存 opt 版）→
    /// `wmma_tf32`（基本版）の順にフォールバックする（3 段選択。
    /// `kernels_wmma_opt.rs` 冒頭ドキュメントコメント「TF32 opt-staged」
    /// 節参照）。
    wmma_tf32_staged: Option<CudaFunction>,
    /// `wmma_tf32_staged` が `None` の場合の失敗理由。`wmma_tf32_opt_error`
    /// と同じ理由で文字列化して保持する。
    wmma_tf32_staged_error: Option<String>,
}

/// GEMM 呼び出しの `m`/`n`/`k` とホスト側スライス長の整合性を検証する。
///
/// GPU 起動前に呼ぶことで、不整合な形状値がカーネル引数（`int m, n, k`）や
/// デバイスバッファ確保サイズへそのまま渡る経路を断つ（OWASP A03。
/// `.claude/rules/security.md`「外部フォーマットパースは長さ・形状の検証を
/// 先に行う」と同じ思想を GEMM 起動入口に適用）。`backend-cpu::gemm::
/// validate_dims`（`crates/backend-cpu/src/gemm.rs:146`）と同種の検証だが、
/// 本関数はさらに「カーネル引数が C の `int`（`i32`）であること」を理由に
/// `i32::MAX` 上限チェックを追加で行う点が異なる（PoC には存在しなかった
/// 検証。上記モジュールコメント参照）。
///
/// `pub(crate)`: `tests/gemm_naive.rs` が実機非依存の単体テストとして
/// 直接呼べるよう `#[cfg(test)]` 外の通常関数として公開範囲をクレート内に
/// 限定する。
pub(crate) fn validate_gemm_dims(
    a_len: usize,
    b_len: usize,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), CudaError> {
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;

    let mk = m_usize
        .checked_mul(k_usize)
        .ok_or_else(|| CudaError::InvalidShape {
            detail: format!("m*k overflows usize: m={m}, k={k}"),
        })?;
    let kn = k_usize
        .checked_mul(n_usize)
        .ok_or_else(|| CudaError::InvalidShape {
            detail: format!("k*n overflows usize: k={k}, n={n}"),
        })?;
    // m*n はカーネル引数には現れないが、`alloc_zeros::<f32>((m*n) as usize)`
    // の確保サイズ計算（`gemm.rs::run_f32`/`run_f16`）で使うため、こちらも
    // 起動前に検証する。
    let mn = m_usize
        .checked_mul(n_usize)
        .ok_or_else(|| CudaError::InvalidShape {
            detail: format!("m*n overflows usize: m={m}, n={n}"),
        })?;

    if a_len != mk {
        return Err(CudaError::InvalidShape {
            detail: format!("a length mismatch: expected {mk} (m*k), actual {a_len}"),
        });
    }
    if b_len != kn {
        return Err(CudaError::InvalidShape {
            detail: format!("b length mismatch: expected {kn} (k*n), actual {b_len}"),
        });
    }

    // カーネル引数（`int m, int n, int k`）は C の 32bit 符号付き整数のため、
    // i32::MAX を超える値を渡すと未定義の切り詰め・符号反転が起こりうる。
    // `u32` 引数の型レベル上限（4294967295）より厳しいこの制約をここで拒否する。
    if m > i32::MAX as u32 || n > i32::MAX as u32 || k > i32::MAX as u32 {
        return Err(CudaError::InvalidShape {
            detail: format!("m/n/k must fit in i32 (kernel argument type): m={m}, n={n}, k={k}"),
        });
    }

    // Cursor Bugbot 指摘（PR #240）: `kernels.rs` の naive カーネルは
    // `row * k + p`／`p * n + col`／`row * n + col` を C の `int`（i32）
    // 算術で計算する。m/n/k 各々が i32::MAX に収まっていても、その積
    // （mk・kn・mn）が i32::MAX を超えるとインデックス計算そのものが
    // 32bit 符号付き整数の範囲でラップし、範囲外読み書きを引き起こしうる。
    // ここでは実際にカーネルが触れる最大インデックス（各積 - 1）が
    // i32 に収まることを起動前に検証する。
    if mk > i32::MAX as usize || kn > i32::MAX as usize || mn > i32::MAX as usize {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "m*k, k*n, m*n must fit in i32 (kernel index arithmetic is 32bit int): \
                 m={m}, n={n}, k={k}, m*k={mk}, k*n={kn}, m*n={mn}"
            ),
        });
    }

    Ok(())
}

/// 出力バッファ `c_dev` の長さが `m*n` と一致することを検証する。
///
/// `launch_tiled_f32`／`launch_wmma_tf32`（`gemm_wmma.rs::launch_f16` も
/// 同様）は `run_*` 系と異なり出力バッファをカーネル起動側で確保せず
/// 呼び出し元から受け取るため、[`validate_gemm_dims`] が検証する
/// 「`a_len`/`b_len` が `m*k`/`k*n` と一致する」だけでは C 側の OOB
/// 書き込みを防げない。PR #349 codex-review 指摘 P0（safe な公開起動 API
/// がバッファ境界検証を省略していた）を受けて追加した、C バッファ専用の
/// 検証。`pub(crate)`: `gemm_wmma.rs::launch_f16` からも呼べるよう
/// クレート内に公開範囲を限定する。
pub(crate) fn validate_output_len(c_len: usize, m: u32, n: u32) -> Result<(), CudaError> {
    let mn = (m as usize)
        .checked_mul(n as usize)
        .ok_or_else(|| CudaError::InvalidShape {
            detail: format!("m*n overflows usize: m={m}, n={n}"),
        })?;
    if c_len != mn {
        return Err(CudaError::InvalidShape {
            detail: format!("c length mismatch: expected {mn} (m*n), actual {c_len}"),
        });
    }
    Ok(())
}

/// tiled カーネル専用の `k` 追加上限検証。
///
/// tiled カーネル（`kernels::TILED_F32`/`TILED_F16`）は各タイル反復で
/// `t * TILE + threadIdx.x`（`threadIdx.x` は最大 `TILE - 1`）を C の `int`
/// 算術で計算し `a_col`／`b_row` を得る。この値は `k` に近い最終タイルで
/// 最大 `k + TILE - 2` 程度に達しうるため、`k` が `i32::MAX - (TILE - 1)`
/// を超えると当該算術が i32 の範囲でオーバーフローしうる（実行前ガード。
/// `validate_gemm_dims` の i32 積ガードとは独立に、tiled 固有のタイル
/// インデックス算術を保護する）。`run_tiled_f32`／`run_tiled_f16` からのみ
/// 呼ばれ、naive 経路の契約（`validate_gemm_dims` のみ）は変更しない。
pub(crate) fn validate_tiled_k_bound(k: u32) -> Result<(), CudaError> {
    let limit = i32::MAX as u32 - (kernels::TILE - 1);
    if k > limit {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "k must not exceed i32::MAX - (TILE - 1) for tiled kernel tile-index \
                 arithmetic: k={k}, limit={limit}, TILE={}",
                kernels::TILE
            ),
        });
    }
    Ok(())
}

/// WMMA TF32 カーネル固有の `k` 追加上限検証。
///
/// `kernels::WMMA_TF32_F32` は各 K タイル反復で `t * WMMA_TF32_K_TILE +
/// local_col`（`local_col` は最大 `WMMA_TF32_K_TILE - 1`）を C の `int` 算術で
/// 計算するため、`validate_tiled_k_bound`（`kernels::TILE` 基準）と同じ理由で
/// `k` が `i32::MAX - (WMMA_TF32_K_TILE - 1)` を超えると当該算術が i32 の
/// 範囲でオーバーフローしうる。`WMMA_TF32_K_TILE`（8）は `TILE`（32）より小さく
/// 上限自体は緩いが、独立したガードとして分離する（tiled 経路の契約を変更
/// しないため。`run_wmma_tf32` からのみ呼ばれる）。
pub(crate) fn validate_wmma_tf32_k_bound(k: u32) -> Result<(), CudaError> {
    let limit = i32::MAX as u32 - (kernels::WMMA_TF32_K_TILE - 1);
    if k > limit {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "k must not exceed i32::MAX - (WMMA_TF32_K_TILE - 1) for WMMA TF32 \
                 kernel tile-index arithmetic: k={k}, limit={limit}, WMMA_TF32_K_TILE={}",
                kernels::WMMA_TF32_K_TILE
            ),
        });
    }
    Ok(())
}

/// WMMA TF32 opt カーネル固有の `k` 追加上限検証（TASK-11.1d・#63）。
///
/// `kernels_wmma_opt::wmma_tf32_f32_opt_source()` は各 K タイル反復で
/// `t * WMMA_TF32_OPT_K_TILE + local_col`（`local_col` は最大
/// `WMMA_TF32_OPT_K_TILE - 1`）を C の `int` 算術で計算する。実際にカーネルが
/// 計算しうる最大インデックスは `ceil(k / WMMA_TF32_OPT_K_TILE) *
/// WMMA_TF32_OPT_K_TILE - 1`（`k == 0` のときは計算自体が発生しないため 0）
/// であり、これが `i32::MAX` を超えると当該算術が i32 の範囲でオーバー
/// フローしうる。
///
/// レビュー指摘（PR #256・chatgpt-codex-connector）: 当初は `i32::MAX -
/// (WMMA_TF32_OPT_K_TILE - 1)` という定数近似の上限を用いていたが、これは
/// 「あらゆる余り（`k mod WMMA_TF32_OPT_K_TILE`）のうち最悪ケース（余り 1）」
/// を仮定した安全側すぎる近似であり、実際には安全な `k`（例:
/// `k = 2_147_483_633..=2_147_483_640`。最終タイル開始位置が
/// `2_147_483_632` で最大インデックスはちょうど `i32::MAX` に収まり
/// オーバーフローしない）まで `InvalidShape` として拒否していた。ここでは
/// 上記の式をそのまま `u64` 算術で計算し `i32::MAX` と比較することで、
/// 個々の `k` について厳密に安全性を判定する（`WMMA_TF32_OPT_K_TILE` を
/// 将来変更しても定数近似のずれが再発しない）。
///
/// なお `WMMA_TF32_OPT_K_TILE`（16）は `i32::MAX + 1`（`2^31`）の約数の
/// ため、[`validate_gemm_dims`] が既に保証する `k <= i32::MAX` の範囲内では
/// 本関数は理論上常に `Ok` を返す（`ceil(k/16)*16 <= 2^31` が任意の
/// `k <= i32::MAX` で成立するため）。それでも実行時に厳密計算を残すのは、
/// この事実が `WMMA_TF32_OPT_K_TILE` の具体値（2 の冪であること）に依存する
/// 暗黙の前提であり、値の変更時に静かに破綻させないため（`run_wmma_tf32`
/// が opt カーネル選択時にのみ呼ぶ。基本版へフォールバックした場合は
/// 引き続き [`validate_wmma_tf32_k_bound`] を適用する）。
pub(crate) fn validate_wmma_tf32_opt_k_bound(k: u32) -> Result<(), CudaError> {
    let tile = kernels_wmma_opt::WMMA_TF32_OPT_K_TILE as u64;
    let max_computed_index = if k == 0 {
        0
    } else {
        (k as u64).div_ceil(tile) * tile - 1
    };
    if max_computed_index > i32::MAX as u64 {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "k tile-index arithmetic for WMMA TF32 opt kernel would overflow i32: k={k}, \
                 max_computed_index={max_computed_index}, WMMA_TF32_OPT_K_TILE={}",
                kernels_wmma_opt::WMMA_TF32_OPT_K_TILE
            ),
        });
    }
    Ok(())
}

/// WMMA TF32 opt-staged カーネル固有の `k` 追加上限検証（イシュー #500）。
/// [`validate_wmma_tf32_opt_k_bound`] と同型・同じ厳密算術（`ceil(k /
/// WMMA_TF32_STAGED_K_TILE) * WMMA_TF32_STAGED_K_TILE - 1` を `u64` で
/// 計算し `i32::MAX` と比較する）だが、`WMMA_TF32_STAGED_K_TILE` 基準で
/// 独立して検証する（両定数は現状同値〈16〉だが、staged 側の K タイル
/// 拡張時に opt 側の検証と無言で食い違わないよう独立関数として保つ）。
pub(crate) fn validate_wmma_tf32_staged_k_bound(k: u32) -> Result<(), CudaError> {
    let tile = kernels_wmma_opt::WMMA_TF32_STAGED_K_TILE as u64;
    let max_computed_index = if k == 0 {
        0
    } else {
        (k as u64).div_ceil(tile) * tile - 1
    };
    if max_computed_index > i32::MAX as u64 {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "k tile-index arithmetic for WMMA TF32 staged kernel would overflow i32: k={k}, \
                 max_computed_index={max_computed_index}, WMMA_TF32_STAGED_K_TILE={}",
                kernels_wmma_opt::WMMA_TF32_STAGED_K_TILE
            ),
        });
    }
    Ok(())
}

/// WMMA TF32 opt-staged カーネルの cp.async 16 バイト転送粒度が要求する
/// グローバル側整列制約を検証する（`gemm_mma.rs::validate_mma_alignment`
/// の f32 版。f16 は 8 要素/16B、f32 は 4 要素/16B である点のみ異なる）。
///
/// A の行ストライドは `k`、B の行ストライドは `n` であり、共有メモリ側の
/// タイル幅（`WMMA_TF32_STAGED_K_TILE`/`WMMA_TF32_STAGED_BLOCK_N`）が共に
/// 4 の倍数であることと合わせて `k % 4 == 0 && n % 4 == 0` を満たさない
/// 限り、行境界をまたぐ列オフセットが 16 バイト境界からずれうる
/// （`kernels_wmma_opt.rs::WMMA_TF32_F32_STAGED_BODY` 内
/// `LOAD_A_STAGE_GROUP`/`LOAD_B_STAGE_GROUP` コメント「REQ-8」参照）。
/// 満たさない形状は `run_wmma_tf32` が staged 経路を選ばず opt 版へ
/// フォールバックするため、`InvalidShape` ではなく `bool` を返す
/// （`validate_mma_alignment` は独立経路〈`CudaMmaGemm`〉の唯一のゲート
/// のため fail-closed に拒否するが、本関数は 3 段フォールバックの経路
/// 選択条件であり拒否ではなくフォールバックが正しい契約のため）。
pub(crate) fn wmma_tf32_staged_alignment_ok(n: u32, k: u32) -> bool {
    n.is_multiple_of(4) && k.is_multiple_of(4)
}

/// `kernels::WMMA_TF32_BLOCK_M`／`WMMA_TF32_BLOCK_N`（ブロックタイル一辺）を
/// 単位に `m`/`n` を `div_ceil` で包含するグリッド次元を構築する。
///
/// naive/tiled 版の [`launch_config`] は「ブロック次元（スレッド形状）＝
/// タイル一辺」の 1:1 対応を前提にグリッドを導出するが、WMMA カーネルは
/// スレッド形状（[`WMMA_TF32_BLOCK_DIM`]、4 warp を 1 次元 128 スレッドに
/// 束ねた形）とタイル一辺（32×32、2×2 warp グリッド）が異なるため、
/// 専用のグリッド計算関数として分離する。末尾ブロックの余剰スレッドは
/// カーネル内の手動境界チェック（REQ-8）に委ねる契約は共通。
/// WMMA(TF32) カーネル（`kernels::WMMA_TF32_F32`）を単独でコンパイル・
/// ロードする。`CudaGemm::new` から呼ばれ、戻り値の `Err` は naive/tiled
/// 4 カーネルの `?` 早期 return には合流させず、呼び出し元で
/// `wmma_tf32_error` として退避する（[`CudaGemm::wmma_tf32`] フィールドの
/// ドキュメンテーションコメント参照。レビュー指摘 #62）。
fn compile_wmma_tf32(device: &CudaDevice, arch: &str) -> Result<CudaFunction, CudaError> {
    let ptx = compile_ptx(kernels::WMMA_TF32_F32, arch)?;
    let func = device
        .context()
        .load_module(ptx)?
        .load_function("gemm_wmma_tf32")?;
    Ok(func)
}

fn wmma_tf32_launch_config(m: u32, n: u32) -> LaunchConfig {
    let grid_dim = (
        n.div_ceil(kernels::WMMA_TF32_BLOCK_N),
        m.div_ceil(kernels::WMMA_TF32_BLOCK_M),
        1,
    );
    LaunchConfig {
        grid_dim,
        block_dim: WMMA_TF32_BLOCK_DIM,
        shared_mem_bytes: 0,
    }
}

/// WMMA(TF32) opt カーネル（`kernels_wmma_opt::wmma_tf32_f32_opt_source()`）を単独で
/// コンパイル・ロードする。[`compile_wmma_tf32`] と同じ理由（レビュー指摘
/// #62 の踏襲）で `CudaGemm::new` の早期 return には合流させず、呼び出し元で
/// `wmma_tf32_opt_error` として退避する。
fn compile_wmma_tf32_opt(device: &CudaDevice, arch: &str) -> Result<CudaFunction, CudaError> {
    let ptx = compile_ptx(kernels_wmma_opt::wmma_tf32_f32_opt_source(), arch)?;
    let func = device
        .context()
        .load_module(ptx)?
        .load_function("gemm_wmma_tf32_opt")?;
    Ok(func)
}

/// [`wmma_tf32_launch_config`] の opt 版。ブロックタイル
/// `kernels_wmma_opt::WMMA_TF32_OPT_BLOCK_M/N`（64×64）を単位に `div_ceil`
/// でグリッドを構築する。末尾ブロックの余剰は opt カーネル内の手動境界
/// チェック（REQ-8）に委ねる契約は基本版と共通。
fn wmma_tf32_opt_launch_config(m: u32, n: u32) -> LaunchConfig {
    let grid_dim = (
        n.div_ceil(kernels_wmma_opt::WMMA_TF32_OPT_BLOCK_N),
        m.div_ceil(kernels_wmma_opt::WMMA_TF32_OPT_BLOCK_M),
        1,
    );
    LaunchConfig {
        grid_dim,
        block_dim: WMMA_TF32_OPT_BLOCK_DIM,
        shared_mem_bytes: 0,
    }
}

/// WMMA(TF32) opt-staged カーネル（イシュー #500・
/// `kernels_wmma_opt::wmma_tf32_f32_staged_source()`）を単独でコンパイル・
/// ロードする。[`compile_wmma_tf32_opt`] と同じ理由で `CudaGemm::new` の
/// 早期 return には合流させず、呼び出し元で `wmma_tf32_staged_error` として
/// 退避する。
fn compile_wmma_tf32_staged(device: &CudaDevice, arch: &str) -> Result<CudaFunction, CudaError> {
    let ptx = compile_ptx(kernels_wmma_opt::wmma_tf32_f32_staged_source(), arch)?;
    let func = device
        .context()
        .load_module(ptx)?
        .load_function("gemm_wmma_tf32_staged")?;
    Ok(func)
}

/// [`wmma_tf32_opt_launch_config`] の staged 版。ブロックタイル
/// `kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_M/N`（64×64。既存 opt 版と
/// 同一）を単位に `div_ceil` でグリッドを構築する。
fn wmma_tf32_staged_launch_config(m: u32, n: u32) -> LaunchConfig {
    let grid_dim = (
        n.div_ceil(kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_N),
        m.div_ceil(kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_M),
        1,
    );
    LaunchConfig {
        grid_dim,
        block_dim: WMMA_TF32_STAGED_BLOCK_DIM,
        shared_mem_bytes: 0,
    }
}

impl CudaGemm {
    /// `device` 上で naive／tiled GEMM カーネル（f32/f16 各 2 種）を NVRTC
    /// コンパイルし保持するハンドルを構築する。
    ///
    /// 手順: `kernels::{NAIVE_F32,NAIVE_F16,TILED_F32,TILED_F16}` を
    /// `device.arch()` 向けに `nvrtc::compile_ptx` でコンパイル →
    /// `device.context().load_module()` →
    /// `load_function("gemm_naive_f32"/"gemm_naive_f16"/"gemm_tiled_f32"/
    /// "gemm_tiled_f16")`。カーネルコンパイル自体は `CudaDevice::new` と
    /// 同じく `libnvrtc` 不在時に `CudaError::NvrtcUnavailable` を返す
    /// （`compile_ptx` のプローブゲートを経由。panic しない）。
    ///
    /// WMMA(TF32) カーネル（`kernels::WMMA_TF32_F32`）のコンパイル・ロードは
    /// 上記 4 カーネルとは独立に扱う（レビュー指摘 #62）。`#include <mma.h>`
    /// の解決失敗や compute capability 8.0 未満といった、naive/tiled より
    /// 広い環境で失敗しうるため、失敗しても `new` 全体を `Err` にせず
    /// `wmma_tf32` を `None`・detail を `wmma_tf32_error` に保持する
    /// （[`Self::wmma_tf32`] フィールドのドキュメンテーションコメント
    /// 参照）。これにより NVRTC が `<mma.h>` を解決できない・compute
    /// capability 8.0 未満の環境でも naive/tiled 4 カーネルは引き続き
    /// 使用可能なままになる。f16 WMMA 経路（`gemm_wmma.rs::CudaWmmaGemm::new`。
    /// #61）は NVRTC コンパイルを試みる前に `device.compute_capability()` で
    /// cc≥7.0 を検査し `CudaError::TensorCoreUnsupported` を返す事前ゲート
    /// 方式だが、本経路は事前ゲートを設けず NVRTC コンパイル結果（成功／
    /// 失敗）のみで可否を判定する事後判定方式を採る。TF32 WMMA が要求する
    /// cc≥8.0 は `m16n16k8` TF32 fragment 命令が非対応アーキテクチャ向け
    /// PTX を生成しようとした際に NVRTC が拒否することで間接的に検査される
    /// （実機での網羅検証は未実施。`docs/cuda-tensor-core-design.md` 参照）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let naive_f32_ptx = compile_ptx(kernels::NAIVE_F32, arch)?;
        let naive_f16_ptx = compile_ptx(kernels::NAIVE_F16, arch)?;
        let tiled_f32_ptx = compile_ptx(kernels::TILED_F32, arch)?;
        let tiled_f16_ptx = compile_ptx(kernels::TILED_F16, arch)?;

        let naive_f32 = device
            .context()
            .load_module(naive_f32_ptx)?
            .load_function("gemm_naive_f32")?;
        let naive_f16 = device
            .context()
            .load_module(naive_f16_ptx)?
            .load_function("gemm_naive_f16")?;
        let tiled_f32 = device
            .context()
            .load_module(tiled_f32_ptx)?
            .load_function("gemm_tiled_f32")?;
        let tiled_f16 = device
            .context()
            .load_module(tiled_f16_ptx)?
            .load_function("gemm_tiled_f16")?;

        // イシュー #599: epilogue 融合カーネルは naive/tiled と同様
        // `#include` を使わず全 compute capability で成立するため、
        // WMMA(TF32) 系のような Option 化・失敗の退避は行わず、`new` の
        // 早期 return（`?`）に合流させる（上記 4 カーネルと同じ扱い）。
        let tiled_bias_act_f32_ptx = compile_ptx(kernels::TILED_BIAS_ACT_F32, arch)?;
        let tiled_bias_act_f32 = device
            .context()
            .load_module(tiled_bias_act_f32_ptx)?
            .load_function("gemm_tiled_bias_act_f32")?;

        // `kernels::WMMA_TF32_F32` はブロックタイル（M/N=32）を warp タイル
        // （WMMA_TF32_FRAG=16）の 2x2 グリッドに割ることを前提にしており、
        // `WMMA_TF32_THREADS`（128 = 4 warp）ともこの分割数と対応する。この
        // 不変条件はカーネルソースの定数変更で壊れうるため、実行時条件に
        // 依存しないコンパイル時 const アサーションで機械検査する
        // （レビュー指摘 #62: `debug_assert_eq!` は release ビルドで消え、
        // かつ CUDA 非搭載の通常 CI ではこの `new` 自体が実行されないため、
        // 従来の位置では実質的に検査されていなかった）。
        const _: () = assert!(kernels::WMMA_TF32_BLOCK_M.is_multiple_of(kernels::WMMA_TF32_FRAG));
        const _: () = assert!(kernels::WMMA_TF32_BLOCK_N.is_multiple_of(kernels::WMMA_TF32_FRAG));
        const _: () = assert!(
            (kernels::WMMA_TF32_BLOCK_M / kernels::WMMA_TF32_FRAG)
                * (kernels::WMMA_TF32_BLOCK_N / kernels::WMMA_TF32_FRAG)
                * 32
                == kernels::WMMA_TF32_THREADS
        );

        // `<mma.h>` は NVRTC 組み込みで解決できないため、include_paths なしの
        // 初回試行は失敗し `nvrtc::compile_ptx` の 2 段目フォールバック
        // （`CUDA_INCLUDE_PATH` または既知候補パス）を経由する契約
        // （`nvrtc.rs` ドキュメンテーションコメント・設計メモ 3.2 節「NVRTC
        // ヘッダ問題」参照。本モジュールは `compile_ptx` に手を加えない）。
        // 上記 4 カーネルと異なり `?` で早期 return せず、失敗を
        // `wmma_tf32_error` に退避して naive/tiled の可用性から切り離す
        // （`Self::wmma_tf32` フィールドのドキュメンテーションコメント参照）。
        let (wmma_tf32, wmma_tf32_error) = match compile_wmma_tf32(device, arch) {
            Ok(func) => (Some(func), None),
            Err(e) => (None, Some(e.to_string())),
        };

        // TASK-11.1d（#63）: `kernels_wmma_opt::wmma_tf32_f32_opt_source()` はブロック
        // タイル（M/N=64）を warp タイル（WARP_TILE=32）の 2x2 グリッドに
        // 割ることを前提にしており、`WMMA_TF32_OPT_THREADS`（128 = 4 warp）
        // ともこの分割数と対応する。上記 `WMMA_TF32_BLOCK_M/N` const アサー
        // ションと同じ理由（レビュー指摘 #62 の踏襲）でコンパイル時に検査
        // する。
        const _: () = assert!(
            kernels_wmma_opt::WMMA_TF32_OPT_BLOCK_M
                .is_multiple_of(kernels_wmma_opt::WMMA_TF32_OPT_WARP_TILE)
        );
        const _: () = assert!(
            kernels_wmma_opt::WMMA_TF32_OPT_BLOCK_N
                .is_multiple_of(kernels_wmma_opt::WMMA_TF32_OPT_WARP_TILE)
        );
        const _: () = assert!(
            (kernels_wmma_opt::WMMA_TF32_OPT_BLOCK_M / kernels_wmma_opt::WMMA_TF32_OPT_WARP_TILE)
                * (kernels_wmma_opt::WMMA_TF32_OPT_BLOCK_N
                    / kernels_wmma_opt::WMMA_TF32_OPT_WARP_TILE)
                * 32
                == kernels_wmma_opt::WMMA_TF32_OPT_THREADS
        );
        // warp タイル（32）は fragment 辺（16）のちょうど 2 倍でなければ
        // ならない（レジスタブロッキング 2×2 の前提。`WMMA_TF32_OPT_FRAG_ROWS`
        // ／`WMMA_TF32_OPT_FRAG_COLS`＝2 とカーネルソース側の固定値に対応）。
        const _: () = assert!(
            kernels_wmma_opt::WMMA_TF32_OPT_WARP_TILE == kernels_wmma_opt::WMMA_TF32_OPT_FRAG * 2
        );
        // 共有メモリ K タイル（16）は fragment K（8）のちょうど 2 倍でなければ
        // ならない（カーネルソース側の `WMMA_TF32_OPT_K_SUBSTEPS`＝2 固定値
        // に対応）。
        const _: () = assert!(
            kernels_wmma_opt::WMMA_TF32_OPT_K_TILE == kernels_wmma_opt::WMMA_TF32_OPT_FRAG_K * 2
        );
        // パディング後の行幅は f32 の `ldm` 制約（4 の倍数）を満たさなければ
        // ならない（`kernels_wmma_opt.rs` 冒頭ドキュメントコメント
        // 「アライメント」参照）。
        const _: () = assert!(kernels_wmma_opt::WMMA_TF32_OPT_A_PAD.is_multiple_of(4));
        const _: () = assert!(kernels_wmma_opt::WMMA_TF32_OPT_B_PAD.is_multiple_of(4));

        let (wmma_tf32_opt, wmma_tf32_opt_error) = match compile_wmma_tf32_opt(device, arch) {
            Ok(func) => (Some(func), None),
            Err(e) => (None, Some(e.to_string())),
        };

        // イシュー #500: `kernels_wmma_opt::wmma_tf32_f32_staged_source()`
        // は既存 TF32 opt と同一のブロックタイル（64）・warp タイル（32）
        // 分割・レジスタブロッキング（2×2）前提を持つ。上記
        // `WMMA_TF32_OPT_BLOCK_M/N` 系 const アサーションと同じ理由
        // （レビュー指摘 #62 の踏襲）でコンパイル時に検査する。
        const _: () = assert!(
            kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_M
                .is_multiple_of(kernels_wmma_opt::WMMA_TF32_STAGED_WARP_TILE)
        );
        const _: () = assert!(
            kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_N
                .is_multiple_of(kernels_wmma_opt::WMMA_TF32_STAGED_WARP_TILE)
        );
        const _: () = assert!(
            (kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_M
                / kernels_wmma_opt::WMMA_TF32_STAGED_WARP_TILE)
                * (kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_N
                    / kernels_wmma_opt::WMMA_TF32_STAGED_WARP_TILE)
                * 32
                == kernels_wmma_opt::WMMA_TF32_STAGED_THREADS
        );
        const _: () = assert!(
            kernels_wmma_opt::WMMA_TF32_STAGED_WARP_TILE
                == kernels_wmma_opt::WMMA_TF32_STAGED_FRAG * 2
        );
        const _: () = assert!(
            kernels_wmma_opt::WMMA_TF32_STAGED_K_TILE
                == kernels_wmma_opt::WMMA_TF32_STAGED_FRAG_K * 2
        );
        const _: () = assert!(kernels_wmma_opt::WMMA_TF32_STAGED_A_PAD.is_multiple_of(4));
        const _: () = assert!(kernels_wmma_opt::WMMA_TF32_STAGED_B_PAD.is_multiple_of(4));
        const _: () = assert!(kernels_wmma_opt::WMMA_TF32_STAGED_STAGES >= 2);

        let (wmma_tf32_staged, wmma_tf32_staged_error) =
            match compile_wmma_tf32_staged(device, arch) {
                Ok(func) => (Some(func), None),
                Err(e) => (None, Some(e.to_string())),
            };

        Ok(Self {
            stream: device.stream().clone(),
            naive_f32,
            naive_f16,
            tiled_f32,
            tiled_f16,
            tiled_bias_act_f32,
            wmma_tf32,
            wmma_tf32_error,
            wmma_tf32_opt,
            wmma_tf32_opt_error,
            wmma_tf32_staged,
            wmma_tf32_staged_error,
        })
    }

    /// naive f32 GEMM を実行する。C = A @ B（`m x k` @ `k x n`）。
    ///
    /// ホスト側形状検証（[`validate_gemm_dims`]）を先行させた後、
    /// 16x16 ブロック・`div_ceil` グリッドで [`Self::run_f32_kernel`] を呼ぶ
    /// （PoC-v2-3 `run_f32` を踏襲。計測用 `Duration` は返さない。
    /// モジュールコメント「PoC からの変更点」参照）。
    pub fn run_naive_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        self.run_f32_kernel(&self.naive_f32, a, b, m, n, k, NAIVE_BLOCK_DIM)
    }

    /// naive f16 GEMM を実行する。入出力は `half::f16`、GPU 内部アキュムレート
    /// は f32（`kernels::NAIVE_F16` 参照）。手順は [`Self::run_naive_f32`] と同一。
    pub fn run_naive_f16(
        &self,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f16>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        self.run_f16_kernel(&self.naive_f16, a, b, m, n, k, NAIVE_BLOCK_DIM)
    }

    /// tiled f32 GEMM を実行する。C = A @ B（`m x k` @ `k x n`）。
    ///
    /// ホスト側形状検証は naive 版と同じ [`validate_gemm_dims`] に加え、
    /// tiled カーネル固有のタイルインデックス算術を保護する
    /// [`validate_tiled_k_bound`] を経由する（モジュールコメント
    /// 「PoC からの変更点」3 参照）。ブロック次元は
    /// [`TILED_BLOCK_DIM`]（`kernels::TILE` x `kernels::TILE`）で固定する。
    pub fn run_tiled_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        self.run_f32_kernel(&self.tiled_f32, a, b, m, n, k, TILED_BLOCK_DIM)
    }

    /// tiled f16 GEMM を実行する。手順は [`Self::run_tiled_f32`] と同一。
    pub fn run_tiled_f16(
        &self,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f16>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        self.run_f16_kernel(&self.tiled_f16, a, b, m, n, k, TILED_BLOCK_DIM)
    }

    /// GEMM epilogue（bias 加算・activation）を融合した tiled GEMM を実行
    /// する。`act(A @ B + bias)`（イシュー #599・TASK-12.1f）。
    ///
    /// `bias` は `Some` の場合 `n` 要素の 1 次元スライス（`B` の列数への
    /// 行方向複製）でなければならない（それ以外の shape・ブロードキャスト
    /// はサポートしない。`ops.rs::CudaBackendOps::gemm_bias_act` が
    /// `bias.shape() == [n]` の場合にのみ本関数へ委譲する契約。呼び出し元
    /// ドキュメント参照）。`act_relu` が `true` なら epilogue で
    /// `max(v, 0)` を追加適用する。
    ///
    /// ホスト側形状検証は [`Self::run_tiled_f32`] と同一
    /// （[`validate_gemm_dims`]・[`validate_tiled_k_bound`]）に加え、
    /// カーネル本体へ触れる前に `bias` の長さが `n` と一致することを検証
    /// する（CPU 参照実装 `gemm_blis_bias_act_parallel` の
    /// `GemmError::BiasLenMismatch` と同じ「カーネル本体アクセス前に検証」
    /// の順序契約。REQ-8・OWASP A03）。
    ///
    /// **`m == 0 || n == 0`**: [`Self::run_f32_kernel`] と同じ理由（no-op
    /// 形状）で空の結果を返す。**`k == 0`**: `run_f32_kernel`／
    /// `run_tiled_f32` は「K 方向の累積対象が存在しない = C は全 0」という
    /// GEMM の数学的定義どおり無条件で全 0 を返すが、CPU 参照実装
    /// （`gemm_blis_bias_act_parallel`）は `k == 0` でも epilogue（bias 加算・
    /// activation）を適用する契約であるため、本関数はその契約に合わせ
    /// `acc == 0` に対する epilogue をホスト側で直接計算する（GPU 起動を
    /// 回避しつつ CPU と同一の意味論を保つ。0 バイトデバイス確保を一部
    /// CUDA driver が拒否しうる問題〈`run_f32_kernel` の該当コメント参照〉
    /// も同時に回避する）。
    ///
    /// [`gemm::BIAS_ACT_FUSED_LAUNCH_COUNT`](BIAS_ACT_FUSED_LAUNCH_COUNT)
    /// は実際に GPU カーネルが起動した場合（`m/n > 0` かつ `k > 0`）にのみ
    /// 増加する（上記フィールドのドキュメンテーションコメント参照）。
    #[allow(clippy::too_many_arguments)]
    pub fn run_tiled_bias_act_f32(
        &self,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        act_relu: bool,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        if let Some(bias) = bias
            && bias.len() != n as usize
        {
            return Err(CudaError::InvalidShape {
                detail: format!(
                    "bias length mismatch: expected {n} (n), actual {}",
                    bias.len()
                ),
            });
        }

        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }

        if k == 0 {
            // 上記ドキュメンテーションコメント「k == 0」参照: CPU 参照
            // 実装と同じく epilogue はホスト側で直接適用し、GPU 起動は
            // 行わない（BIAS_ACT_FUSED_LAUNCH_COUNT は増加させない）。
            let mut out = vec![0.0f32; (m as usize) * (n as usize)];
            if let Some(bias) = bias {
                for row in out.chunks_mut(n as usize) {
                    for (x, bv) in row.iter_mut().zip(bias.iter()) {
                        *x += *bv;
                    }
                }
            }
            if act_relu {
                for x in out.iter_mut() {
                    *x = x.max(0.0);
                }
            }
            return Ok(out);
        }

        BIAS_ACT_FUSED_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed);

        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        // `bias` が `None` の場合はダミーの 1 要素バッファを渡す（null
        // ポインタをカーネル引数へ渡す経路を作らない。`has_bias == 0` の
        // ガードによりカーネル側は実際にはこのバッファを参照しない。
        // `kernels::TILED_BIAS_ACT_F32` ドキュメンテーションコメント参照）。
        let (bias_dev, has_bias): (CudaSlice<f32>, i32) = match bias {
            Some(bias) => (self.stream.clone_htod(bias)?, 1),
            None => (self.stream.alloc_zeros::<f32>(1)?, 0),
        };
        let mut c_dev = self
            .stream
            .alloc_zeros::<f32>((m as usize) * (n as usize))?;

        let cfg = launch_config(m, n, TILED_BLOCK_DIM);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
        let act_i: i32 = if act_relu { 1 } else { 0 };

        // SAFETY: run_f32_kernel と同一の根拠（該当コメント参照）。
        // 追加引数（bias_dev・has_bias・act_i）は上記で検証済みの `n`／
        // `bias` の有無と 1:1 対応し、カーネル内 epilogue は書き込み
        // ガード（`row < m && col < n`）の内側でのみ `bias[col]` を
        // 参照するため OOB は発生しない（REQ-8）。
        unsafe {
            self.stream
                .launch_builder(&self.tiled_bias_act_f32)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&bias_dev)
                .arg(&mut c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&has_bias)
                .arg(&act_i)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;

        let c_host = self.stream.clone_dtoh(&c_dev)?;
        Ok(c_host)
    }

    /// WMMA（Tensor Core）を用いた TF32 GEMM を実行する。C = A @ B（`m x k` @
    /// `k x n`）、入出力は f32（内部で TF32 に丸めて Tensor Core へ投入する。
    /// `kernels::WMMA_TF32_F32` 参照）。
    ///
    /// ホスト側形状検証は naive/tiled 版と同じ [`validate_gemm_dims`] に加え、
    /// WMMA カーネル固有のタイルインデックス算術を保護する
    /// [`validate_wmma_tf32_k_bound`] を経由する（`validate_tiled_k_bound`
    /// と同じ考え方だが `kernels::WMMA_TF32_K_TILE`（8）基準で独立して検証する）。
    /// TASK-11.1c（#62）・REQ-11。
    ///
    /// `CudaGemm::new` 時点で WMMA(TF32) カーネルのコンパイル・ロードが
    /// 失敗していた場合（[`Self::wmma_tf32`] フィールド参照）は
    /// `CudaError::WmmaUnavailable` を返す。この場合でも naive/tiled 版の
    /// `run_naive_*`／`run_tiled_*` は道連れにならず引き続き使用できる
    /// （レビュー指摘 #62）。
    ///
    /// **TASK-11.1d（#63）フォールバック方針**: 共有メモリ・タイル最適化版
    /// （[`Self::wmma_tf32_opt`]）が `new` 時点でコンパイル・ロードに成功
    /// していれば、そちらを優先的に使用する（#63 の受け入れ条件「tiled
    /// 実装を上回る実測」を満たす経路）。opt 版が `None`（コンパイル失敗
    /// または未対応環境）の場合は基本版（[`Self::wmma_tf32`]）へ自動
    /// フォールバックし、公開シグネチャ・呼び出し側の挙動は変えない
    /// （REQ-11 は明示切替 API を提供しない方針。`kernels_wmma_opt.rs`
    /// 冒頭ドキュメントコメント「公開 API への影響」参照）。
    /// 共有メモリ・タイル最適化版 WMMA(TF32) カーネル（[`Self::wmma_tf32_opt`]）
    /// が `new` 時点でコンパイル・ロードに成功しているかを返す（TASK-11.1d・
    /// #63。PR #256 レビュー指摘: chatgpt-codex-connector「Require the
    /// optimized kernel in the optimized benchmark」対応）。
    ///
    /// `run_wmma_tf32` は opt カーネルが `None` の場合に基本版
    /// （[`Self::wmma_tf32`]）へ自動フォールバックする（公開 API の挙動は
    /// 変えない設計判断。上記ドキュメンテーションコメント参照）ため、
    /// `run_wmma_tf32` の戻り値の成否だけでは opt カーネルが実際に実行
    /// されたかを判定できない。opt カーネル固有の性能・タイル境界を検証
    /// する受け入れテスト（`tests/gemm_wmma_tf32_opt.rs`）はこの関数で
    /// 事前に可用性を確認し、フォールバックが起きていないことを保証した
    /// うえで計測・検証する。
    pub fn wmma_tf32_opt_available(&self) -> bool {
        self.wmma_tf32_opt.is_some()
    }

    /// [`Self::wmma_tf32_opt_available`] が `false` の場合の失敗理由
    /// （[`Self::wmma_tf32_opt_error`] の公開読み取り口）。opt カーネルが
    /// 利用可能な場合は `None` を返す。テストが「opt カーネルが使用不能
    /// だった具体的な理由」をパニックメッセージへ含められるようにする。
    pub fn wmma_tf32_opt_unavailable_reason(&self) -> Option<&str> {
        self.wmma_tf32_opt_error.as_deref()
    }

    /// TF32 opt-staged カーネル（イシュー #500）が `CudaGemm::new` 時点で
    /// コンパイル・ロードに成功しているかを返す（[`Self::wmma_tf32_opt_available`]
    /// と同じ理由。`run_wmma_tf32` は staged カーネルが利用可能かつ
    /// cp.async 16 バイト整列条件を満たす形状でのみ staged 経路を選ぶため、
    /// 戻り値の成否だけでは staged 経路が実際に実行されたかを判定
    /// できない。`tests/gemm_wmma_tf32_staged.rs` はこの関数で可用性を
    /// 確認したうえで計測・検証する）。
    pub fn wmma_tf32_staged_available(&self) -> bool {
        self.wmma_tf32_staged.is_some()
    }

    /// [`Self::wmma_tf32_staged_available`] が `false` の場合の失敗理由
    /// （[`Self::wmma_tf32_opt_unavailable_reason`] と同じ理由）。
    pub fn wmma_tf32_staged_unavailable_reason(&self) -> Option<&str> {
        self.wmma_tf32_staged_error.as_deref()
    }

    /// 指定した `(n, k)` の形状で `run_wmma_tf32` が実際に WMMA 経路
    /// （staged または opt のいずれか）を選び、basic へフォールバック
    /// しないことを判定する（PR #678 codex-review Medium 指摘対応
    /// 「Weak routed-path availability gate」）。
    ///
    /// [`Self::wmma_tf32_staged_available`] と [`Self::wmma_tf32_opt_available`]
    /// を `||` で束ねただけのゲートは形状非依存であり、staged が利用可能
    /// でも整列非対応形状（`wmma_tf32_staged_alignment_ok` が `false` を
    /// 返す形状。例: 63×65×33・1×1×1）では staged 経路を選ばない。opt が
    /// 未対応な環境ではその形状だけ静かに basic へフォールバックし、
    /// 「WMMA 経路の parity を検証している」というテストの意図を裏切って
    /// も検査は素通りしてしまう。本関数は `run_wmma_tf32` の 3 段選択
    /// ロジック（`wmma_tf32_staged_alignment_ok` を含む）を形状ごとに
    /// 再現し、その形状で実際に WMMA 経路が選ばれるかを判定する。
    pub fn wmma_tf32_routed_path_available(&self, n: u32, k: u32) -> bool {
        (self.wmma_tf32_staged_available() && wmma_tf32_staged_alignment_ok(n, k))
            || self.wmma_tf32_opt_available()
    }

    pub fn run_wmma_tf32(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        // イシュー #500: 3 段選択。staged（cp.async 多段パイプライン・
        // fragment 先読み。cp.async 16 バイト整列条件 `n%4==0 && k%4==0`
        // を満たす形状限定）→ opt（既存 `__syncthreads()` ベース。整列
        // 非対応形状のフォールバック先）→ basic の順。整列非対応形状
        // （63×65×33 等）は従来どおり opt カーネルが処理するため、
        // 既存 `#[ignore]` テストの parity 特性は非後退（本ファイル
        // `wmma_tf32_staged_alignment_ok` ドキュメンテーションコメント
        // 参照）。
        if let Some(func) = self.wmma_tf32_staged.as_ref()
            && wmma_tf32_staged_alignment_ok(n, k)
        {
            validate_gemm_dims(a.len(), b.len(), m, n, k)?;
            validate_wmma_tf32_staged_k_bound(k)?;
            return self.run_wmma_tf32_staged_kernel(func, a, b, m, n, k);
        }

        if let Some(func) = self.wmma_tf32_opt.as_ref() {
            validate_gemm_dims(a.len(), b.len(), m, n, k)?;
            validate_wmma_tf32_opt_k_bound(k)?;
            return self.run_wmma_tf32_opt_kernel(func, a, b, m, n, k);
        }

        let func = self
            .wmma_tf32
            .as_ref()
            .ok_or_else(|| CudaError::WmmaUnavailable {
                // opt/基本の両方が使用不能な場合、両方の失敗理由を connect
                // して返す（`wmma_tf32_opt_error` は opt 版が使用不能な場合
                // のみ意味を持つ detail であり、opt 版が `Some` の場合は
                // この分岐に到達しないためここでのみ参照される）。
                detail: match (&self.wmma_tf32_error, &self.wmma_tf32_opt_error) {
                    (Some(basic), Some(opt)) => {
                        format!("opt kernel unavailable: {opt}; basic kernel unavailable: {basic}")
                    }
                    (Some(basic), None) => basic.clone(),
                    (None, _) => "WMMA(TF32) kernel unavailable for an unknown reason".to_string(),
                },
            })?;
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        validate_wmma_tf32_k_bound(k)?;
        self.run_wmma_f32_kernel(func, a, b, m, n, k)
    }

    /// WMMA TF32 カーネル専用の起動手続き。[`Self::run_f32_kernel`] と
    /// 転送・同期・回収の構造は同一だが、グリッド計算に
    /// [`wmma_tf32_launch_config`]（ブロックタイル 32×32 基準）を使う点が
    /// 異なる（`WMMA_TF32_BLOCK_DIM` は 4 warp を束ねた 128 スレッドの
    /// 1 次元ブロックであり、`launch_config` が前提とする「ブロック次元＝
    /// タイル一辺」の対応が成立しないため。上記 `wmma_tf32_launch_config`
    /// ドキュメンテーションコメント参照）。
    fn run_wmma_f32_kernel(
        &self,
        func: &CudaFunction,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        // run_f32_kernel と同一の根拠（下記コメント参照。Cursor Bugbot 指摘
        // PR #240／#244）。
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if k == 0 {
            return Ok(vec![0.0f32; (m as usize) * (n as usize)]);
        }

        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        let mut c_dev = self
            .stream
            .alloc_zeros::<f32>((m as usize) * (n as usize))?;

        let cfg = wmma_tf32_launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: run_f32_kernel と同一の根拠。カーネル引数
        // （a_dev/b_dev/c_dev・m_i/n_i/k_i）はホスト側検証
        // （validate_gemm_dims・validate_wmma_tf32_k_bound）済みの m/n/k から
        // 導出しており、カーネル内の手動境界チェック（guarded load・
        // エピローグ store のガード付きコピー。kernels.rs の
        // WMMA_TF32_F32 ドキュメンテーションコメント参照、REQ-8）と
        // 合わせて OOB 読み書きが起きない根拠とする。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;

        let c_host = self.stream.clone_dtoh(&c_dev)?;
        Ok(c_host)
    }

    /// WMMA TF32 opt カーネル専用の起動手続き（TASK-11.1d・#63）。
    /// [`Self::run_wmma_f32_kernel`] と転送・同期・回収の構造は同一だが、
    /// グリッド計算に [`wmma_tf32_opt_launch_config`]（ブロックタイル
    /// 64×64 基準）を使う点が異なる。
    fn run_wmma_tf32_opt_kernel(
        &self,
        func: &CudaFunction,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        // run_wmma_f32_kernel と同一の根拠（Cursor Bugbot 指摘 PR #240／#244）。
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if k == 0 {
            return Ok(vec![0.0f32; (m as usize) * (n as usize)]);
        }

        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        let mut c_dev = self
            .stream
            .alloc_zeros::<f32>((m as usize) * (n as usize))?;

        let cfg = wmma_tf32_opt_launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: run_wmma_f32_kernel と同一の根拠。カーネル引数
        // （a_dev/b_dev/c_dev・m_i/n_i/k_i）はホスト側検証
        // （validate_gemm_dims・validate_wmma_tf32_opt_k_bound）済みの
        // m/n/k から導出しており、opt カーネル内の手動境界チェック
        // （guarded load・エピローグ store のガード付きコピー。
        // kernels_wmma_opt.rs 参照、REQ-8）と合わせて OOB 読み書きが
        // 起きない根拠とする。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;

        let c_host = self.stream.clone_dtoh(&c_dev)?;
        Ok(c_host)
    }

    /// WMMA TF32 opt-staged カーネル専用の起動手続き（イシュー #500）。
    /// [`Self::run_wmma_tf32_opt_kernel`] と転送・同期・回収の構造は
    /// 同一だが、グリッド計算に [`wmma_tf32_staged_launch_config`]
    /// （ブロックタイルは既存 opt 版と同じ 64×64）を使う点が異なる。
    fn run_wmma_tf32_staged_kernel(
        &self,
        func: &CudaFunction,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        // run_wmma_tf32_opt_kernel と同一の根拠（Cursor Bugbot 指摘 PR #240／#244）。
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if k == 0 {
            return Ok(vec![0.0f32; (m as usize) * (n as usize)]);
        }

        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        let mut c_dev = self
            .stream
            .alloc_zeros::<f32>((m as usize) * (n as usize))?;

        let cfg = wmma_tf32_staged_launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: run_wmma_tf32_opt_kernel と同一の根拠。カーネル引数
        // （a_dev/b_dev/c_dev・m_i/n_i/k_i）はホスト側検証
        // （validate_gemm_dims・validate_wmma_tf32_staged_k_bound）済みの
        // m/n/k から導出しており、staged カーネル内の手動境界チェック
        // （cp.async src-size ゼロ充填・エピローグ store のガード付き
        // コピー。kernels_wmma_opt.rs::WMMA_TF32_F32_STAGED_BODY 参照、
        // REQ-8）と合わせて OOB 読み書きが起きない根拠とする。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;

        let c_host = self.stream.clone_dtoh(&c_dev)?;
        Ok(c_host)
    }

    /// f32 カーネル共通の起動手続き（naive/tiled 双方から呼ばれる）。
    ///
    /// 呼び出し元がホスト側形状検証（`validate_gemm_dims`／
    /// `validate_tiled_k_bound`）を終えている前提で、`clone_htod` で A・B
    /// を転送し `block_dim` に応じたグリッドでカーネルを起動、
    /// `synchronize` の後 `clone_dtoh` で C を回収する（PoC-v2-3 の
    /// `run_f32` と同じ構造）。
    #[allow(clippy::too_many_arguments)]
    fn run_f32_kernel(
        &self,
        func: &CudaFunction,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
        block_dim: (u32, u32, u32),
    ) -> Result<Vec<f32>, CudaError> {
        // Cursor Bugbot 指摘（PR #240）: `validate_gemm_dims` は
        // `backend-cpu::gemm_naive` と同様 m==0／n==0（a/c が空）を no-op
        // として許容するが、その形状のまま `launch_config` を呼ぶと
        // grid_dim の x（n 由来）または y（m 由来）が 0 になり、CUDA
        // ドライバは 0 次元の起動を拒否する。CPU 側の no-op 契約
        // （`backend-cpu/src/gemm.rs` の `n == 0` 早期 return コメント参照）
        // に揃え、カーネル起動自体を行わず空の結果を返す（naive/tiled 共通）。
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }

        // Cursor Bugbot 指摘（PR #244）: `k == 0` の場合 `a`（m*k 要素）・
        // `b`（k*n 要素）は共に空スライスになり、そのまま `clone_htod` を
        // 呼ぶと 0 バイトのデバイスバッファ確保を driver に要求する。一部
        // 環境の CUDA driver は 0 バイト確保を拒否するため（`cudarc` 経由の
        // `cuMemAlloc` は 0 バイトで `CUDA_ERROR_INVALID_VALUE` を返しうる）、
        // カーネル起動自体を回避し `m*n` 要素の全 0 ベクタを返す（K 方向の
        // 累積対象が存在しない = C は全 0 という GEMM の数学的定義どおりの
        // 契約。`tests/gemm_tiled.rs` の `tiled_f32_zero_k_returns_all_zero`
        // 参照）。
        if k == 0 {
            return Ok(vec![0.0f32; (m as usize) * (n as usize)]);
        }

        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        let mut c_dev = self
            .stream
            .alloc_zeros::<f32>((m as usize) * (n as usize))?;

        let cfg = launch_config(m, n, block_dim);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: カーネル引数は a_dev/b_dev/c_dev（それぞれ a.len()/b.len()/
        // (m*n) 要素の確保済みデバイスバッファ）と m_i/n_i/k_i の 5 個・型・
        // 個数がホスト側検証（validate_gemm_dims、tiled はさらに
        // validate_tiled_k_bound）済みの m/n/k と 1:1 対応し、カーネル内の
        // 手動境界チェック（naive: `if (row < m && col < n)`。tiled: タイル
        // ロード時の三項ガード＋書き込み時の同条件。kernels.rs 参照、
        // REQ-8）と合わせて OOB 読み書きが起きない根拠とする。グリッド
        // 次元は `div_ceil` で m/n を包含するよう構築しており
        // （launch_config）、末尾ブロックの余剰スレッドはカーネル内境界
        // チェックで弾かれる。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;

        let c_host = self.stream.clone_dtoh(&c_dev)?;
        Ok(c_host)
    }

    /// f16 カーネル共通の起動手続き。[`Self::run_f32_kernel`] と同一構造
    /// （naive/tiled 双方の `run_*_f16` から呼ばれる）。
    #[allow(clippy::too_many_arguments)]
    fn run_f16_kernel(
        &self,
        func: &CudaFunction,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
        block_dim: (u32, u32, u32),
    ) -> Result<Vec<f16>, CudaError> {
        // run_f32_kernel と同一の根拠（上記コメント参照）。
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }

        // run_f32_kernel の k==0 早期 return と同一の根拠（上記コメント
        // 参照）。
        if k == 0 {
            return Ok(vec![f16::ZERO; (m as usize) * (n as usize)]);
        }

        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        let mut c_dev = self
            .stream
            .alloc_zeros::<f16>((m as usize) * (n as usize))?;

        let cfg = launch_config(m, n, block_dim);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: run_f32_kernel と同一の根拠（上記コメント参照）。カーネル
        // 引数の型（__half*/int）・個数・デバイスバッファ長は
        // validate_gemm_dims（tiled はさらに validate_tiled_k_bound）で
        // 検証済みの m/n/k から導出しており、カーネル内手動境界チェック
        // （REQ-8）と合わせて OOB を防ぐ。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;

        let c_host = self.stream.clone_dtoh(&c_dev)?;
        Ok(c_host)
    }

    /// A・B（f32）をホスト→デバイスへ転送する（tiled f32／WMMA(TF32) の
    /// H2D 部分の切り出し。`gemm_mma.rs::CudaMmaGemm::upload_f16` と同じ
    /// 理由でベンチマークが転送とカーネル実行を分離できるよう公開する。
    /// PR #349 codex-review 指摘 P1「PyTorch 参照計測（`torch.matmul`+
    /// `torch.cuda.synchronize()` のみ。入力テンソルは計測ループ開始前に
    /// 生成済みで反復ごとの H2D 転送・出力バッファ確保を含まない。
    /// `docs/spec/03-poc/poc-v2-3-cuda-gemm/code/pytorch/gemm_bench_torch_cuda.py`
    /// 実測確認済み）と計測境界を揃える」対応。tiled f32・WMMA(TF32) は
    /// いずれも入力が f32 のため 1 メソッドで共有する）。
    pub fn upload_f32(
        &self,
        a: &[f32],
        b: &[f32],
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), CudaError> {
        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        Ok((a_dev, b_dev))
    }

    /// C 用のゼロ初期化デバイスバッファを確保する（[`Self::upload_f32`] と
    /// 同じ理由で公開する。tiled f32・WMMA(TF32) で共有）。
    pub fn alloc_output_f32(&self, m: u32, n: u32) -> Result<CudaSlice<f32>, CudaError> {
        Ok(self
            .stream
            .alloc_zeros::<f32>((m as usize) * (n as usize))?)
    }

    /// デバイス常駐済みの A/B/C バッファに対して tiled f32 カーネルを
    /// 起動し、完了を待つ（H2D/D2H を含まない「GPU 実行のみ」の区間。
    /// [`Self::upload_f32`]・[`Self::alloc_output_f32`] と組み合わせて
    /// ベンチマークの計測対象を絞るために公開する。
    ///
    /// PR #349 codex-review 指摘 P0（`launch_*` 系は `pub fn` でありながら
    /// `CudaSlice` 長・`m/n/k` の整合・`i32` 変換上限・tiled 固有の `k` 上限
    /// を検証せずに `unsafe` launch へ渡していた）を受け、`run_tiled_f32`
    /// と同じ検証（[`validate_gemm_dims`]・[`validate_tiled_k_bound`]）に
    /// 加え、デバイスバッファ長（`a_dev`/`b_dev`/`c_dev`）が `m/n/k` と
    /// 1:1 対応することを起動前に検証する（safe な公開 API である以上、
    /// 呼び出し元の契約違反〈短い A/B/C と大きな次元の組み合わせ〉から
    /// 独立して GPU 側 OOB を防ぐ必要があるため、呼び出し元が事前検証済みで
    /// あることを前提にした検証省略はしない）。
    pub fn launch_tiled_f32(
        &self,
        a_dev: &CudaSlice<f32>,
        b_dev: &CudaSlice<f32>,
        c_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        validate_output_len(c_dev.len(), m, n)?;

        let cfg = launch_config(m, n, TILED_BLOCK_DIM);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: run_f32_kernel と同一の根拠。カーネル引数
        // （a_dev/b_dev/c_dev・m_i/n_i/k_i）は上記で検証済みの m/n/k
        // と 1:1 対応し、カーネル内の手動境界チェック（REQ-8）と合わせて
        // OOB 読み書きが起きない根拠とする。
        unsafe {
            self.stream
                .launch_builder(&self.tiled_f32)
                .arg(a_dev)
                .arg(b_dev)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// デバイス常駐済みの A/B/C バッファに対して WMMA(TF32) カーネルを
    /// 起動し、完了を待つ（[`Self::launch_tiled_f32`] と同じ「GPU 実行
    /// のみ」契約）。`run_wmma_tf32` と同一の 3 段選択ロジック（staged →
    /// opt → 基本）を用いる（イシュー #500 で staged 選択を追加）。
    /// 呼び出し元は事前に `run_wmma_tf32` を 1 回 probe 実行して可用性を
    /// 確認している前提だが（`cuda_floor_bench.rs::measure_wmma_tf32`
    /// 参照）、本関数自体は safe な公開 API のため、選択される経路
    /// （staged／opt／基本）ごとに必要な `k` 境界検証
    /// （[`validate_wmma_tf32_staged_k_bound`]／
    /// [`validate_wmma_tf32_opt_k_bound`]／[`validate_wmma_tf32_k_bound`]）
    /// とデバイスバッファ長検証を呼び出し元の事前検証に依存せず自前で行う
    /// （PR #349 codex-review 指摘 P0。[`Self::launch_tiled_f32`] の
    /// ドキュメンテーションコメント参照）。
    pub fn launch_wmma_tf32(
        &self,
        a_dev: &CudaSlice<f32>,
        b_dev: &CudaSlice<f32>,
        c_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        validate_output_len(c_dev.len(), m, n)?;
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: run_wmma_f32_kernel／run_wmma_tf32_opt_kernel／
        // run_wmma_tf32_staged_kernel と同一の根拠。カーネル引数は
        // 上記・各分岐内で検証済みの m/n/k と 1:1 対応し、カーネル内の
        // 手動境界チェック（REQ-8）と合わせて OOB 読み書きが起きない
        // 根拠とする。
        if let Some(func) = self
            .wmma_tf32_staged
            .as_ref()
            .filter(|_| wmma_tf32_staged_alignment_ok(n, k))
        {
            validate_wmma_tf32_staged_k_bound(k)?;
            let cfg = wmma_tf32_staged_launch_config(m, n);
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(a_dev)
                    .arg(b_dev)
                    .arg(c_dev)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i)
                    .launch(cfg)?;
            }
        } else if let Some(func) = self.wmma_tf32_opt.as_ref() {
            validate_wmma_tf32_opt_k_bound(k)?;
            let cfg = wmma_tf32_opt_launch_config(m, n);
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(a_dev)
                    .arg(b_dev)
                    .arg(c_dev)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i)
                    .launch(cfg)?;
            }
        } else if let Some(func) = self.wmma_tf32.as_ref() {
            validate_wmma_tf32_k_bound(k)?;
            let cfg = wmma_tf32_launch_config(m, n);
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(a_dev)
                    .arg(b_dev)
                    .arg(c_dev)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i)
                    .launch(cfg)?;
            }
        } else {
            return Err(CudaError::WmmaUnavailable {
                detail: "WMMA(TF32) kernel unavailable (neither opt nor basic kernel loaded); \
                         launch_wmma_tf32 called without a prior successful run_wmma_tf32 probe"
                    .to_string(),
            });
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// C（f32）をデバイス→ホストへ転送する（[`Self::upload_f32`] と同じ
    /// 理由で公開する。tiled f32・WMMA(TF32) で共有）。
    pub fn download_f32(&self, c_dev: &CudaSlice<f32>) -> Result<Vec<f32>, CudaError> {
        Ok(self.stream.clone_dtoh(c_dev)?)
    }
}

/// `block_dim` に対し `m`/`n` を切り上げ（`div_ceil`）で包含するグリッド
/// 次元を構築する。末尾ブロックが `m`/`n` を超える分はカーネル内の手動
/// 境界チェック（REQ-8）に委ねる契約（`kernels.rs` 参照）。
fn launch_config(m: u32, n: u32, block_dim: (u32, u32, u32)) -> LaunchConfig {
    let grid_dim = (n.div_ceil(block_dim.0), m.div_ceil(block_dim.1), 1);
    LaunchConfig {
        grid_dim,
        block_dim,
        shared_mem_bytes: 0,
    }
}

/// parity 非後退契約（イシュー #491・#500）のベースライン fixture・検査
/// ユーティリティ（`ParityBaseline`・`BASELINES`・
/// `assert_no_parity_regression` 等）。
///
/// **PR #640 codex-review P1 指摘対応（テスト専用切替 API の公開 feature
/// 露出）**: 以前は基本版 WMMA(TF32) カーネル（opt 可用性に関わらず
/// `self.wmma_tf32` を強制実行するエントリポイント）を `internal-testing`
/// という通常の Cargo feature で `pub` 化し、`tests/parity_nonregression.rs`
/// （独立クレート扱いの統合テスト）から呼んでいた。Cargo feature は依存
/// グラフ全体で単一に統合されるため、downstream が自分の `Cargo.toml` で
/// `backend-cuda = { ..., features = ["internal-testing"] }` と明示すれば
/// この feature を有効化でき、`[dev-dependencies]` の自己参照だけでは
/// 外部からの明示的な有効化を防げない（REQ-11「明示切替 API を提供しない」
/// 方針・内部表現の非漏出に抵触）。
///
/// 本モジュール以下の検査は代わりにライブラリ自身の単体テスト
/// （`#[cfg(test)]`。downstream のビルドには一切コンパイルされず、feature
/// でも到達不能）として実装し、基本版カーネルには [`CudaGemm::wmma_tf32`]
/// （private field）・[`CudaGemm::run_wmma_f32_kernel`]（private fn）へ
/// 同一モジュール内から直接アクセスする。新規の公開 API・feature は
/// 増やさない。イシュー #500 で TF32 opt-staged カーネルが追加され、公開
/// `run_wmma_tf32` が整列形状で staged 経路を最優先するようになったため
/// （3 段選択。`run_wmma_tf32` ドキュメンテーションコメント参照）、opt
/// カーネル単独の非後退検査（`wmma_tf32_opt_kernel_parity_does_not_regress`）
/// も同じ理由で同一モジュール内から `self.wmma_tf32_opt`（private field）・
/// `run_wmma_tf32_opt_kernel`（private fn）へ直接アクセスする（PR #678
/// codex-review P1 指摘対応: 公開 API 経由の旧検査が staged 経路へ黙って
/// すり替わっていた欠陥の是正）。
///
/// fixture 自体は `tests/common/parity_baseline.rs` を `#[path]` で直接
/// 取り込み、統合テスト側（`tests/parity_nonregression.rs::common`）と
/// 単一のソースを共有する（値を複製すると `docs/perf/cuda-parity-baseline.md`
/// との二重管理・記録漏れの温床になるため避ける）。
#[cfg(test)]
#[path = "../tests/common/parity_baseline.rs"]
mod parity_baseline_fixture;

#[cfg(test)]
mod tests {
    use super::*;

    /// parity 非後退契約（イシュー #491）: 基本版 WMMA(TF32) カーネル
    /// （[`CudaGemm::wmma_tf32`] フィールド）専用の非後退ゲート。
    ///
    /// `run_wmma_tf32`（本番唯一の公開 API）は opt カーネルが利用可能な
    /// 環境では常に opt を優先するため（`run_wmma_tf32` ドキュメンテーション
    /// コメント参照）、公開 API 経由では基本版カーネル単独を検査できない。
    /// 本テストは同一モジュール内から `self.wmma_tf32`／
    /// `run_wmma_f32_kernel`（いずれも private）へ直接アクセスすることで、
    /// 公開 API・feature を一切増やさずにこの検査を実現する
    /// （上記 `parity_baseline_fixture` モジュールコメント・PR #640
    /// codex-review 指摘対応参照）。
    ///
    /// fail-closed 契約: 記録済みベースライン行の provenance が未確定
    /// （`ParityBaseline::baseline_provenance_unconfirmed == true`）の
    /// 場合、`assert_no_parity_regression` 側が必ず panic する（黙って
    /// skip しない。`tests/common/parity_baseline.rs` 参照）。実機再測定で
    /// 確定値を記録し `false` へ更新するまで、本テストは実機実行のたびに
    /// fail し続ける契約であり、これは意図した挙動である（実機テストの
    /// 恒常 fail は本リポで既知の受け入れ済み状態。
    /// `docs/backend-cuda-real-device-testing.md` §5.3・§7 参照）。
    #[test]
    #[ignore = "CUDA 実機（compute capability 8.0 以降）必須"]
    fn wmma_tf32_basic_kernel_parity_does_not_regress() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
        let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

        let func = gemm.wmma_tf32.as_ref().expect(
            "basic WMMA(TF32) kernel must be available on this ignored test runner (reason: \
             see wmma_tf32_error)",
        );

        let mut failures: Vec<String> = Vec::new();

        for baseline in super::parity_baseline_fixture::BASELINES
            .iter()
            .filter(|b| b.path == super::parity_baseline_fixture::ParityPath::WmmaTf32)
        {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut rng = bench_harness::rng::Xorshift64Star::new(baseline.seed);
                let a = rng.fill_vec((baseline.m as usize) * (baseline.k as usize));
                let b = rng.fill_vec((baseline.k as usize) * (baseline.n as usize));

                let mut c_ref = vec![0.0f32; (baseline.m as usize) * (baseline.n as usize)];
                backend_cpu::matmul_reference_fma(
                    &a,
                    &b,
                    &mut c_ref,
                    baseline.m as usize,
                    baseline.n as usize,
                    baseline.k as usize,
                )
                .expect(
                    "matmul_reference_fma shape validation must pass for well-formed baseline \
                     input",
                );

                validate_gemm_dims(a.len(), b.len(), baseline.m, baseline.n, baseline.k)
                    .expect("baseline fixture shapes must be valid GEMM dimensions");
                validate_wmma_tf32_k_bound(baseline.k)
                    .expect("baseline fixture k must satisfy WMMA(TF32) k bound");

                let c_gpu = gemm
                    .run_wmma_f32_kernel(func, &a, &b, baseline.m, baseline.n, baseline.k)
                    .expect(
                        "basic WMMA(TF32) kernel execution must succeed on this ignored test \
                         runner",
                    );

                let report = backend_cpu::compare(&c_gpu, &c_ref)
                    .expect("shape must match baseline fixture");

                super::parity_baseline_fixture::assert_no_parity_regression(
                    baseline.context,
                    &report,
                    baseline,
                );
            }));
            if let Err(err) = result {
                let msg = err
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "panic (詳細不明)".to_string());
                failures.push(format!("{}: {msg}", baseline.context));
            }
        }

        assert!(
            failures.is_empty(),
            "parity 非後退契約 FAIL（{} 件）:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// parity 非後退契約（イシュー #491・#500）: opt カーネル
    /// （[`CudaGemm::wmma_tf32_opt`] フィールド）専用の非後退ゲート。
    ///
    /// **PR #678 codex-review P1 指摘対応**: `run_wmma_tf32`（公開唯一の
    /// API）はイシュー #500 で追加された TF32 opt-staged カーネルが利用
    /// 可能かつ cp.async 16 バイト整列条件を満たす形状では staged 経路を
    /// 最優先で選ぶため（`run_wmma_tf32` ドキュメンテーションコメント
    /// 参照）、公開 API 経由では opt カーネル単独を検査できない
    /// （`tests/parity_nonregression.rs` の旧実装はこの分岐を知らずに
    /// `wmma_tf32_opt_available()` の確認だけで opt 専用と誤認しており、
    /// staged 経路の回帰を opt の非後退として黙って見逃す・opt 自体の
    /// 回帰を検出できない、という「opt 非後退テストが staged 経路へ
    /// すり替わる」欠陥があった）。本テストは
    /// [`wmma_tf32_basic_kernel_parity_does_not_regress`] と同型のパターンで
    /// 同一モジュール内から `self.wmma_tf32_opt`／`run_wmma_tf32_opt_kernel`
    /// （いずれも private）へ直接アクセスし、3 段選択を経由せず opt
    /// カーネルを強制実行することで、公開 API・feature を一切増やさずに
    /// この検査を実現する。
    #[test]
    #[ignore = "CUDA 実機（compute capability 8.0 以降）必須"]
    fn wmma_tf32_opt_kernel_parity_does_not_regress() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
        let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

        let func = gemm.wmma_tf32_opt.as_ref().expect(
            "opt WMMA(TF32) kernel must be available on this ignored test runner (reason: see \
             wmma_tf32_opt_error)",
        );

        let mut failures: Vec<String> = Vec::new();

        for baseline in super::parity_baseline_fixture::BASELINES
            .iter()
            .filter(|b| b.path == super::parity_baseline_fixture::ParityPath::WmmaTf32Opt)
        {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut rng = bench_harness::rng::Xorshift64Star::new(baseline.seed);
                let a = rng.fill_vec((baseline.m as usize) * (baseline.k as usize));
                let b = rng.fill_vec((baseline.k as usize) * (baseline.n as usize));

                let mut c_ref = vec![0.0f32; (baseline.m as usize) * (baseline.n as usize)];
                backend_cpu::matmul_reference_fma(
                    &a,
                    &b,
                    &mut c_ref,
                    baseline.m as usize,
                    baseline.n as usize,
                    baseline.k as usize,
                )
                .expect(
                    "matmul_reference_fma shape validation must pass for well-formed baseline \
                     input",
                );

                validate_gemm_dims(a.len(), b.len(), baseline.m, baseline.n, baseline.k)
                    .expect("baseline fixture shapes must be valid GEMM dimensions");
                validate_wmma_tf32_opt_k_bound(baseline.k)
                    .expect("baseline fixture k must satisfy WMMA(TF32) opt k bound");

                let c_gpu = gemm
                    .run_wmma_tf32_opt_kernel(func, &a, &b, baseline.m, baseline.n, baseline.k)
                    .expect(
                        "opt WMMA(TF32) kernel execution must succeed on this ignored test \
                         runner",
                    );

                let report = backend_cpu::compare(&c_gpu, &c_ref)
                    .expect("shape must match baseline fixture");

                super::parity_baseline_fixture::assert_no_parity_regression(
                    baseline.context,
                    &report,
                    baseline,
                );
            }));
            if let Err(err) = result {
                let msg = err
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "panic (詳細不明)".to_string());
                failures.push(format!("{}: {msg}", baseline.context));
            }
        }

        assert!(
            failures.is_empty(),
            "parity 非後退契約 FAIL（{} 件）:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// [`wmma_tf32_opt_kernel_parity_does_not_regress`] と同一の private
    /// field 経由アクセス（`self.wmma_tf32_opt`／`run_wmma_tf32_opt_kernel`）
    /// で `run_wmma_tf32_opt_kernel` を CPU 参照実装と直接照合するヘルパー。
    /// 3 段選択（`run_wmma_tf32`）を経由しないため、cp.async 整列形状でも
    /// opt-staged カーネルへ横取りされず opt カーネル固有のタイル境界
    /// （ブロックタイル 64、共有メモリ K タイル 16）を確実に踏む。
    ///
    /// `assert_no_parity_regression`（記録済みベースラインとの比較）ではなく
    /// [`backend_cpu::assert_parity`]（複合判定そのものでの合否）を使う点が
    /// [`wmma_tf32_opt_kernel_parity_does_not_regress`] との違いである:
    /// 本ヘルパーが検査するのは `tests/gemm_wmma_tf32_opt.rs` から移設した
    /// 任意形状の網羅（`docs/perf/cuda-parity-baseline.md` に記録された
    /// 実機実測値を持たない形状を含む）であり、`ParityBaseline::BASELINES`
    /// への未計測行追加（`baseline_provenance_unconfirmed: true` の
    /// プレースホルダ）は fail-closed 契約により無条件 panic になってしまう
    /// （`tests/common/parity_baseline.rs::assert_no_parity_regression`
    /// ドキュメンテーションコメント参照）。実測値を持つ 3 形状（64×64×64・
    /// 512×512×512・512×512×4096）の非後退検査は
    /// [`wmma_tf32_opt_kernel_parity_does_not_regress`] に委ね、本ヘルパーは
    /// 二重管理しない。
    fn assert_wmma_tf32_opt_kernel_parity(
        gemm: &CudaGemm,
        func: &CudaFunction,
        context: &str,
        seed: u64,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
        let a = rng.fill_vec((m as usize) * (k as usize));
        let b = rng.fill_vec((k as usize) * (n as usize));

        let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
        backend_cpu::matmul_reference_fma(&a, &b, &mut c_ref, m as usize, n as usize, k as usize)
            .expect("matmul_reference_fma shape validation must pass for well-formed test input");

        validate_gemm_dims(a.len(), b.len(), m, n, k)
            .expect("test shape must be a valid GEMM dimension");
        validate_wmma_tf32_opt_k_bound(k).expect("test k must satisfy WMMA(TF32) opt k bound");

        let c_gpu = gemm
            .run_wmma_tf32_opt_kernel(func, &a, &b, m, n, k)
            .expect("opt WMMA(TF32) kernel execution must succeed on this ignored test runner");

        backend_cpu::assert_parity(context, &c_gpu, &c_ref);
    }

    /// opt カーネル**単独**の形状網羅テスト（PR #678 codex-review P1
    /// 再指摘対応）。
    ///
    /// イシュー #500 のルーティング変更で `tests/gemm_wmma_tf32_opt.rs::
    /// wmma_tf32_opt_matches_reference_across_shapes`（公開 API
    /// `run_wmma_tf32` 経由）が整列形状（`n%4==0 && k%4==0`）では
    /// opt-staged カーネルへ横取りされるようになり、opt カーネル固有の
    /// タイル境界カバレッジを失っていた。本テストは
    /// [`assert_wmma_tf32_opt_kernel_parity`] で 3 段選択を経由せず opt
    /// カーネルを強制実行することで、旧 `wmma_tf32_opt_matches_reference_across_shapes`
    /// が検査していた全形状（ブロックタイル倍数・非倍数境界・非正方・
    /// 極小）のカバレッジを復元する（シード起点 `3000 + idx` は移設元と
    /// 同一。64×64×64・seed=3000 の 1 行目は `ParityBaseline` 記録値
    /// 〈`wmma_tf32_opt 64x64x64 seed=3000`〉と同一入力になる）。
    #[test]
    #[ignore = "CUDA 実機（compute capability 8.0 以降）必須"]
    fn wmma_tf32_opt_kernel_matches_reference_across_shapes() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
        let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

        let func = gemm.wmma_tf32_opt.as_ref().expect(
            "opt WMMA(TF32) kernel must be available on this ignored test runner (reason: see \
             wmma_tf32_opt_error)",
        );

        let cases: &[(u32, u32, u32)] = &[
            (64, 64, 64),
            (128, 128, 128),
            (512, 512, 512),
            (63, 65, 33),
            (65, 63, 17),
            (64, 96, 256),
            (1, 1, 1),
        ];
        for (idx, &(m, n, k)) in cases.iter().enumerate() {
            let context = format!("opt kernel shape m={m} n={n} k={k}");
            assert_wmma_tf32_opt_kernel_parity(&gemm, func, &context, 3000 + idx as u64, m, n, k);
        }
    }

    /// opt カーネル**単独**の K 大ストレスケース（PR #678 codex-review P1
    /// 再指摘対応）。旧 `tests/gemm_wmma_tf32_opt.rs::wmma_tf32_opt_k4096_stress`
    /// からの移設（シード・形状とも同一。上記
    /// [`wmma_tf32_opt_kernel_matches_reference_across_shapes`]
    /// ドキュメンテーションコメント参照）。
    #[test]
    #[ignore = "CUDA 実機（compute capability 8.0 以降）必須"]
    fn wmma_tf32_opt_kernel_k4096_stress() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
        let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

        let func = gemm.wmma_tf32_opt.as_ref().expect(
            "opt WMMA(TF32) kernel must be available on this ignored test runner (reason: see \
             wmma_tf32_opt_error)",
        );

        assert_wmma_tf32_opt_kernel_parity(
            &gemm,
            func,
            "opt kernel K4096 stress 512x512x4096",
            0xC0FFEE,
            512,
            512,
            4096,
        );
        assert_wmma_tf32_opt_kernel_parity(
            &gemm,
            func,
            "opt kernel K4096 stress 4096x4096x4096",
            0xBEEF,
            4096,
            4096,
            4096,
        );
    }

    /// #500 の目的（cp.async 多段化・fragment 先読みによる TF32 経路の性能
    /// 改善）の実測本体: staged カーネルが opt カーネルを上回ることを、
    /// 同一実行内で 5 回計測した中央値で確認する（`.claude/rules/
    /// coding-rust.md`「ベンチは 5 回計測の中央値」）。
    ///
    /// **PR #678 codex-review P2 指摘対応**: `tests/gemm_wmma_tf32_staged.rs`
    /// の旧実装は独立クレート扱いのため公開 API 経由でしか計測できず、
    /// `run_wmma_tf32`（公開 API）は staged 選択後は opt 単体を分離計測
    /// できないため、doc comment・関数名が主張する「opt を上回る」ことを
    /// 実際には検証していなかった（比較対象が `run_tiled_f32` にすり替わる。
    /// codex-review 指摘）。本テストは `wmma_tf32_opt_kernel_parity_does_not_regress`
    /// と同じ手段（同一モジュール内から `self.wmma_tf32_staged`／
    /// `self.wmma_tf32_opt`（いずれも private）へ直接アクセス）で、staged・
    /// opt 双方を 3 段選択を経由せず強制実行し、同一実行内で TFLOPS を
    /// 直接比較する。`tests/gemm_wmma_tf32_staged.rs::
    /// wmma_tf32_staged_exceeds_tiled_f32_tflops_at_4096`（公開 API 経由の
    /// tiled 比較。受け入れ基準「4096 の対 PyTorch 比が 25.64% を上回る」の
    /// 実測本体）とは別軸の検査である。
    #[test]
    #[ignore = "CUDA 実機（compute capability 8.0 以降）必須。実測記録は \
                docs/perf/cuda-gemm-wmma-tf32-phase-b.md"]
    fn wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096() {
        use std::time::Instant;

        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
        let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

        let staged_func = gemm.wmma_tf32_staged.as_ref().expect(
            "staged WMMA(TF32) kernel must be available on this ignored test runner so that the \
             TFLOPS comparison actually exercises the staged kernel (reason: see \
             wmma_tf32_staged_error)",
        );
        let opt_func = gemm.wmma_tf32_opt.as_ref().expect(
            "opt WMMA(TF32) kernel must be available on this ignored test runner so that the \
             comparison baseline is the actual optimized kernel (reason: see wmma_tf32_opt_error)",
        );

        let (m, n, k) = (4096u32, 4096u32, 4096u32);
        let mut rng = bench_harness::rng::Xorshift64Star::new(0xACE1);
        let a = rng.fill_vec((m as usize) * (k as usize));
        let b = rng.fill_vec((k as usize) * (n as usize));

        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);

        let median_tflops = |run: &dyn Fn() -> Vec<f32>| -> f64 {
            // warmup（NVRTC JIT・クロック遷移の影響を計測から除外する）。
            let _ = run();
            let mut samples = Vec::with_capacity(5);
            for _ in 0..5 {
                let start = Instant::now();
                let _ = run();
                samples.push(start.elapsed().as_secs_f64());
            }
            samples.sort_by(|x, y| x.partial_cmp(y).expect("elapsed seconds must not be NaN"));
            let median = samples[samples.len() / 2];
            (flops / median) / 1e12
        };

        let opt_tflops = median_tflops(&|| {
            gemm.run_wmma_tf32_opt_kernel(opt_func, &a, &b, m, n, k)
                .expect("opt WMMA(TF32) kernel must succeed on CUDA-equipped test runner")
        });
        let staged_tflops = median_tflops(&|| {
            gemm.run_wmma_tf32_staged_kernel(staged_func, &a, &b, m, n, k)
                .expect("staged WMMA(TF32) kernel must succeed on CUDA-equipped test runner")
        });

        assert!(
            staged_tflops > opt_tflops,
            "staged 経路（{staged_tflops:.3} TFLOPS）が opt 経路（{opt_tflops:.3} TFLOPS）を \
             上回りませんでした（M=N=K=4096）"
        );
    }

    #[test]
    fn validate_gemm_dims_accepts_matching_lengths() {
        assert!(validate_gemm_dims(2 * 3, 3 * 4, 2, 4, 3).is_ok());
    }

    #[test]
    fn validate_gemm_dims_rejects_a_len_mismatch() {
        let err = validate_gemm_dims(5, 12, 2, 4, 3).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_gemm_dims_rejects_b_len_mismatch() {
        let err = validate_gemm_dims(6, 11, 2, 4, 3).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_gemm_dims_rejects_mk_overflow() {
        let err = validate_gemm_dims(0, 0, u32::MAX, 1, u32::MAX).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_gemm_dims_rejects_i32_max_exceeding_dims() {
        // usize (64bit) では m*k はオーバーフローしないが、カーネル引数
        // が i32 のため i32::MAX 超過は別途拒否される必要がある。
        let m = (i32::MAX as u32) + 1;
        let a_len = m as usize;
        let err = validate_gemm_dims(a_len, 1, m, 1, 1).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_gemm_dims_accepts_zero_m_or_n_as_noop_shape() {
        // m==0／n==0 は `backend-cpu::gemm_naive` と同じ no-op 形状として
        // 許容する（Cursor Bugbot 指摘 #240。カーネル起動自体は `run_naive_*`
        // 側の早期 return で回避するため、検証自体は拒否しない）。
        assert!(validate_gemm_dims(0, 3 * 4, 0, 4, 3).is_ok());
        assert!(validate_gemm_dims(4 * 3, 0, 4, 0, 3).is_ok());
    }

    #[test]
    fn validate_gemm_dims_rejects_mk_product_exceeding_i32_max() {
        // m*k は usize（64bit）に収まるが、カーネル側の `row * k + p` は
        // i32 算術のためインデックスがラップしうる（Cursor Bugbot 指摘
        // #240）。m/n/k 個々は i32::MAX 以下でも積が超過するケースを拒否する。
        //
        // Cursor Bugbot 指摘（PR #240 再指摘）: a_len/b_len を
        // validate_gemm_dims の長さチェック（mk/kn との一致検査）を
        // 通過する値にしないと、意図した i32 積ガードに到達する前に
        // 長さ不一致エラーで reject されてしまい、このガードを削除しても
        // テストが pass し続ける（意図した経路をロックできない）。
        // よって b_len は k*n（このケースは n=1 なので k）に正しく合わせる。
        let m: u32 = 1 << 16; // 65536
        let n: u32 = 1;
        let k: u32 = 1 << 16; // 65536 → m*k = 2^32 > i32::MAX
        let a_len = (m as usize) * (k as usize);
        let b_len = (k as usize) * (n as usize);
        let err = validate_gemm_dims(a_len, b_len, m, n, k).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_gemm_dims_rejects_kn_product_exceeding_i32_max() {
        // 上記と同じ理由で a_len を m*k（このケースは m=1 なので k）に
        // 正しく合わせ、長さチェックを通過させたうえで k*n の積ガードへ
        // 到達させる。
        let m: u32 = 1;
        let k: u32 = 1 << 16;
        let n: u32 = 1 << 16;
        let a_len = (m as usize) * (k as usize);
        let b_len = (k as usize) * (n as usize);
        let err = validate_gemm_dims(a_len, b_len, m, n, k).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_gemm_dims_rejects_mn_product_exceeding_i32_max() {
        // 上記と同じ理由で b_len を k*n（このケースは k=1 なので n）に
        // 正しく合わせ、長さチェックを通過させたうえで m*n の積ガードへ
        // 到達させる。
        let m: u32 = 1 << 16;
        let n: u32 = 1 << 16;
        let k: u32 = 1;
        let a_len = (m as usize) * (k as usize);
        let b_len = (k as usize) * (n as usize);
        let err = validate_gemm_dims(a_len, b_len, m, n, k).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_tiled_k_bound_accepts_ordinary_k() {
        assert!(validate_tiled_k_bound(4096).is_ok());
        assert!(validate_tiled_k_bound(0).is_ok());
    }

    #[test]
    fn validate_tiled_k_bound_rejects_k_exceeding_limit() {
        let limit = i32::MAX as u32 - (kernels::TILE - 1);
        assert!(validate_tiled_k_bound(limit).is_ok());
        let err = validate_tiled_k_bound(limit + 1).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn launch_config_grid_dim_covers_m_and_n_via_div_ceil() {
        // 17x19 を 16x16 ブロックで覆うには grid (2, 2) が必要
        // （div_ceil(17,16)=2, div_ceil(19,16)=2）。
        let cfg = launch_config(17, 19, (16, 16, 1));
        assert_eq!(cfg.grid_dim, (2, 2, 1));
        assert_eq!(cfg.block_dim, (16, 16, 1));
    }

    #[test]
    fn validate_wmma_tf32_k_bound_accepts_ordinary_k() {
        assert!(validate_wmma_tf32_k_bound(4096).is_ok());
        assert!(validate_wmma_tf32_k_bound(0).is_ok());
    }

    #[test]
    fn validate_wmma_tf32_k_bound_rejects_k_exceeding_limit() {
        let limit = i32::MAX as u32 - (kernels::WMMA_TF32_K_TILE - 1);
        assert!(validate_wmma_tf32_k_bound(limit).is_ok());
        let err = validate_wmma_tf32_k_bound(limit + 1).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn wmma_tf32_launch_config_grid_dim_covers_m_and_n_via_block_tile_div_ceil() {
        // 33x31 を 32x32 ブロックタイルで覆うには grid (1, 2) が必要
        // （div_ceil(31,32)=1 が n=31 に対応する x、div_ceil(33,32)=2 が
        // m=33 に対応する y。`wmma_tf32_launch_config(m, n)` の呼び出し順は
        // launch_config と同じく (m, n) だが、grid_dim は (n 由来, m 由来)
        // の順で構築される）。
        let cfg = wmma_tf32_launch_config(33, 31);
        assert_eq!(cfg.grid_dim, (1, 2, 1));
        // ブロック次元は「タイル一辺」ではなく WMMA_TF32_THREADS（128 スレッド
        // 1 次元）である点が naive/tiled 版の launch_config と異なる
        // （wmma_tf32_launch_config ドキュメンテーションコメント参照）。
        assert_eq!(cfg.block_dim, WMMA_TF32_BLOCK_DIM);
        assert_eq!(cfg.block_dim, (kernels::WMMA_TF32_THREADS, 1, 1));
    }

    #[test]
    fn validate_wmma_tf32_opt_k_bound_accepts_ordinary_k() {
        assert!(validate_wmma_tf32_opt_k_bound(4096).is_ok());
        assert!(validate_wmma_tf32_opt_k_bound(0).is_ok());
    }

    /// PR #256 レビュー指摘（chatgpt-codex-connector）の回帰テスト:
    /// 旧実装（`i32::MAX - (WMMA_TF32_OPT_K_TILE - 1)` の定数近似）は
    /// `2_147_483_633..=2_147_483_640` を安全にもかかわらず `InvalidShape`
    /// として拒否していた。厳密計算版はこの範囲全体を受理し、かつ
    /// [`validate_gemm_dims`] が許容する上限そのもの（`i32::MAX`）まで
    /// 一貫して受理する（`WMMA_TF32_OPT_K_TILE`（16）が `i32::MAX + 1`
    /// （`2^31`）の約数であるため、`k <= i32::MAX` の範囲内で本関数は理論上
    /// 常に `Ok` になる。関数側ドキュメンテーションコメント参照）。
    #[test]
    fn validate_wmma_tf32_opt_k_bound_accepts_full_range_up_to_i32_max() {
        for k in 2_147_483_633u32..=2_147_483_640u32 {
            assert!(
                validate_wmma_tf32_opt_k_bound(k).is_ok(),
                "k={k} must be accepted (largest computed index is exactly i32::MAX, not an overflow)"
            );
        }
        assert!(validate_wmma_tf32_opt_k_bound(i32::MAX as u32).is_ok());
    }

    #[test]
    fn wmma_tf32_opt_launch_config_grid_dim_covers_m_and_n_via_block_tile_div_ceil() {
        // 65x63 を 64x64 ブロックタイルで覆うには grid (1, 2) が必要
        // （div_ceil(63,64)=1 が n=63 に対応する x、div_ceil(65,64)=2 が
        // m=65 に対応する y。wmma_tf32_launch_config のテストと同じ
        // 呼び出し順・grid_dim 順の契約）。
        let cfg = wmma_tf32_opt_launch_config(65, 63);
        assert_eq!(cfg.grid_dim, (1, 2, 1));
        assert_eq!(cfg.block_dim, WMMA_TF32_OPT_BLOCK_DIM);
        assert_eq!(
            cfg.block_dim,
            (kernels_wmma_opt::WMMA_TF32_OPT_THREADS, 1, 1)
        );
    }

    #[test]
    fn wmma_tf32_opt_launch_config_exact_multiple_shape_has_no_extra_tile() {
        let cfg = wmma_tf32_opt_launch_config(128, 192);
        assert_eq!(cfg.grid_dim, (3, 2, 1));
    }

    // ========================================================================
    // TF32 opt-staged（イシュー #500）
    // ========================================================================

    #[test]
    fn validate_wmma_tf32_staged_k_bound_accepts_ordinary_k() {
        assert!(validate_wmma_tf32_staged_k_bound(4096).is_ok());
        assert!(validate_wmma_tf32_staged_k_bound(0).is_ok());
    }

    /// [`validate_wmma_tf32_opt_k_bound_accepts_full_range_up_to_i32_max`]
    /// と同じ根拠。`WMMA_TF32_STAGED_K_TILE`（16）は
    /// `WMMA_TF32_OPT_K_TILE` と同値のため同一の受理範囲を持つ。
    #[test]
    fn validate_wmma_tf32_staged_k_bound_accepts_full_range_up_to_i32_max() {
        for k in 2_147_483_633u32..=2_147_483_640u32 {
            assert!(
                validate_wmma_tf32_staged_k_bound(k).is_ok(),
                "k={k} must be accepted (largest computed index is exactly i32::MAX, not an overflow)"
            );
        }
        assert!(validate_wmma_tf32_staged_k_bound(i32::MAX as u32).is_ok());
    }

    /// [`wmma_tf32_staged_alignment_ok`] が cp.async 16 バイト転送粒度
    /// （f32 4 要素）の整列条件を正しく判定することを検証する
    /// （`gemm_mma.rs::validate_mma_alignment_accepts_multiples_of_eight`
    /// と同方針だが、こちらは 3 段フォールバックの経路選択条件のため
    /// `Result` ではなく `bool` を返す）。
    #[test]
    fn wmma_tf32_staged_alignment_ok_accepts_multiples_of_four_and_rejects_others() {
        assert!(wmma_tf32_staged_alignment_ok(64, 32));
        assert!(wmma_tf32_staged_alignment_ok(4, 4));
        assert!(wmma_tf32_staged_alignment_ok(0, 0));
        assert!(!wmma_tf32_staged_alignment_ok(65, 32)); // n が 4 の倍数でない
        assert!(!wmma_tf32_staged_alignment_ok(64, 33)); // k が 4 の倍数でない
        assert!(!wmma_tf32_staged_alignment_ok(65, 33)); // 両方とも 4 の倍数でない
    }

    #[test]
    fn wmma_tf32_staged_launch_config_grid_dim_covers_m_and_n_via_block_tile_div_ceil() {
        // wmma_tf32_opt_launch_config_grid_dim_covers_m_and_n_via_block_tile_div_ceil
        // と同一形状（ブロックタイルは既存 opt 版と同じ 64×64）。
        let cfg = wmma_tf32_staged_launch_config(65, 63);
        assert_eq!(cfg.grid_dim, (1, 2, 1));
        assert_eq!(cfg.block_dim, WMMA_TF32_STAGED_BLOCK_DIM);
        assert_eq!(
            cfg.block_dim,
            (kernels_wmma_opt::WMMA_TF32_STAGED_THREADS, 1, 1)
        );
    }

    #[test]
    fn wmma_tf32_staged_launch_config_exact_multiple_shape_has_no_extra_tile() {
        let cfg = wmma_tf32_staged_launch_config(128, 192);
        assert_eq!(cfg.grid_dim, (3, 2, 1));
    }
}
