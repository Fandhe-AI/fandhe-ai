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

use std::cell::Cell;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg};
use half::f16;

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels;
use crate::kernels_tiled_pipeline;
use crate::kernels_tiled_pipeline_128x64;
use crate::kernels_transpose;
use crate::kernels_wmma_opt;
use crate::module_cache::load_function_cached;
use crate::nvrtc::{CompiledDims, CudaKernelDescriptor};
use crate::pool::CudaAllocator;
// イシュー #1214: VJP 専用 NT/TN 転置入口（`transpose_to_pooled`）が
// 転置中間バッファをプールから確保するために本番経路でも使う
// （従来は `#[cfg(test)]` 限定のテストヘルパー専用だった）。
use crate::pool::PooledCudaHandle;
use crate::transpose::{
    tiled_launch_config, validate_transpose_dims, validate_transpose_output_len,
};
// `compile_ptx` は本番経路（`CudaGemm::new`。イシュー #1024 で
// `load_function_cached` 経由へ結線済み）ではなく、`internal-diagnostics`
// feature（既定 off）限定の診断用コンストラクタ
// （`new_with_tf32_staged_swizzle`／`new_with_tf32_staged_pads`）専用の
// 直コンパイル経路でのみ使う。既定ビルドでは未参照のため import 自体も
// 同 feature でゲートし、未使用 import 警告を避ける。
#[cfg(feature = "internal-diagnostics")]
use crate::nvrtc::compile_ptx;

/// naive GEMM カーネル起動 1 回あたりのブロック次元（16x16 = 256 スレッド）。
///
/// PoC-v2-3（`cuda/mod.rs:174`）と同じ値を踏襲する。tiled 版の
/// `kernels::TILE`（32x32。[`TILED_BLOCK_DIM`]）とは独立したパラメータであり、
/// 共有メモリを使わない naive カーネルの `__shared__` 配列サイズ制約は受けない。
const NAIVE_BLOCK_DIM: (u32, u32, u32) = (16, 16, 1);

/// tiled f16 GEMM（[`CudaGemm::run_tiled_f16`]）カーネル起動 1 回あたりの
/// ブロック次元。**イシュー #1032 以降 f16 専用**（f32 tiled 系は
/// [`TILED_F32_BLOCK_DIM`] へ分離。f16 は本イシューのスコープ外）。
///
/// `kernels::TILE` x `kernels::TILE` に固定する必要がある（カーネル内
/// `__shared__ __half as_tile[TILE][TILE]` 等はブロック内スレッド数と
/// 1:1 対応するコンパイル時定数のため、ここがずれるとタイル境界外の
/// スレッドが共有メモリを書かない一方でロード先が欠落し誤った積和になる）。
const TILED_BLOCK_DIM: (u32, u32, u32) = (kernels::TILE, kernels::TILE, 1);

/// tiled f32 GEMM（[`CudaGemm::run_tiled_f32`]／
/// [`CudaGemm::run_tiled_bias_act_f32`]。イシュー #1032 レジスタブロッキング
/// 刷新版）カーネル起動 1 回あたりのブロック次元（16x16 = 256 スレッド）。
///
/// `kernels::TILED_F32_THREADS_X` x `kernels::TILED_F32_THREADS_Y` に
/// 固定する必要がある（各スレッドが `TILED_F32_TM` x `TILED_F32_TN` 出力
/// を担当するレジスタブロッキング構成のため、旧 [`TILED_BLOCK_DIM`] の
/// ような「ブロック次元＝タイル一辺」の 1:1 対応ではなく、
/// [`tiled_f32_launch_config`] がタイル一辺（`TILED_F32_BM`/`BN`）と
/// ブロック次元を別々に扱う。`kernels.rs` の
/// `tiled_f32_constants_satisfy_thread_and_tile_invariants` テストが
/// この整合を検査する）。
const TILED_F32_BLOCK_DIM: (u32, u32, u32) = (
    kernels::TILED_F32_THREADS_X,
    kernels::TILED_F32_THREADS_Y,
    1,
);

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

/// `swizzle::SWIZZLE_APPLY_MIN_M_BLOCKS`/`SWIZZLE_APPLY_MIN_N_BLOCKS` は
/// f16 `mma.sync` 経路のブロックタイル（64×128）から導出された定数
/// （`gemm_mma.rs` 冒頭の同型 `const _: () = assert!(...)` 参照）であり、
/// TF32 opt-staged のブロックタイル（64×64。`WMMA_TF32_STAGED_BLOCK_M/N`）
/// からの再導出値とは一致しない（N 方向: `4096/64=64` に対し定数は 32）。
/// `should_apply_swizzle`（`swizzle.rs`）の実効判定は正方形条件
/// （`m == n && m >= SWIZZLE_APPLY_MIN_SQUARE_DIM`）が成立すれば軸別
/// ブロック数下限・総タイル数下限は自動的に成立する冗長条件（`swizzle.rs::
/// should_apply_swizzle` ドキュメンテーションコメント「呼び出し元が
/// `mma_launch_config` と同一の `div_ceil`〜」節参照）ため、TF32 staged の
/// より細かいブロックタイル（64×64）から導出される軸別ブロック数は f16 の
/// 共有定数を**必ず上回る**（`4096/64=64 >= 64`・`4096/64=64 >= 32`）。
/// 厳密な等式ではなく不等式で検証するのはこの理由による（イシュー #856。
/// `gemm_mma.rs` の等式 assert と異なる形になる根拠を明記する）。
const _: () = assert!(
    4096 / kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_M >= crate::swizzle::SWIZZLE_APPLY_MIN_M_BLOCKS
        && 4096 / kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_N
            >= crate::swizzle::SWIZZLE_APPLY_MIN_N_BLOCKS
        && crate::swizzle::SWIZZLE_APPLY_MIN_K == 4096
        && crate::swizzle::SWIZZLE_APPLY_MIN_SQUARE_DIM == 4096,
    "swizzle::SWIZZLE_APPLY_MIN_M_BLOCKS/SWIZZLE_APPLY_MIN_N_BLOCKS が WMMA_TF32_STAGED_BLOCK_M/N \
     から導出される M=N=K=4096 相当のブロック数を上回ってしまっています（TF32 staged 側の \
     should_apply_swizzle 判定が想定より狭くなる）。swizzle.rs／WMMA_TF32_STAGED_BLOCK_M/N を \
     確認してください"
);

thread_local! {
    /// [`CudaGemm::run_tiled_bias_act_f32`] が実際に GPU カーネルを起動した
    /// 回数（イシュー #599）。
    ///
    /// `ops.rs::CudaBackendOps::gemm_bias_act` の経路選択（融合 vs
    /// `fandhe_ai_tensor_core::backend_ops::BackendOps::gemm_bias_act` デフォルト実装の
    /// 非融合 3 段合成）が実際に融合カーネルへ到達しているかを、実機なしの
    /// 単体テスト（`ops.rs` 内 `#[cfg(test)]`）が検証するための可観測点。
    /// テスト専用の計測であり公開 API の意味論・数値契約には一切影響しない。
    /// `m == 0 || n == 0`（no-op）・`k == 0`（ホスト側で直接 epilogue のみ
    /// 計算し GPU 起動を回避する分岐。[`CudaGemm::run_tiled_bias_act_f32`]
    /// 参照）の場合はカーネルを起動しないためカウントしない。
    ///
    /// **スレッドローカルにする理由（codex-review 指摘・PR #688）**: `static
    /// AtomicU64`（プロセス全体共有）だと `cargo test` 既定の並列実行下で
    /// 「`before` 読み取り〜`gemm_bias_act` 呼び出し〜`after` 読み取り」の間に
    /// 別スレッドで実行中の別テストが同じ融合カーネルを起動すると、当該呼び
    /// 出しが実際には融合経路を通らなくても `after > before` が偶然成立し
    /// うる（偽陽性）。Rust の既定テストハーネストは各テスト関数の実行を
    /// 単一スレッド内で完結させ、同一スレッド上で複数テストが同時に走る
    /// ことはない（スレッドプールがテストをスレッド間で使い回すのは逐次的）
    /// ため、カウンタをスレッドローカルにすれば「呼び出し元スレッドが実際に
    /// 起動した回数」だけを観測でき、他スレッドで並走する別テストの起動が
    /// 混入しない（直列化やプロセス全体 Mutex を要さない、呼び出し単位の
    /// 観測フック）。
    pub(crate) static BIAS_ACT_FUSED_LAUNCH_COUNT: Cell<u64> = const { Cell::new(0) };

    /// `ops.rs::CudaBackendOps::gemm` が `crate::precision::tf32_gemm_enabled()`
    /// opt-in 時に [`CudaGemm::run_wmma_tf32`] へ実際にルーティングした回数
    /// （イシュー #1042）。`BIAS_ACT_FUSED_LAUNCH_COUNT` と同型の可観測点で、
    /// 実機なしの単体テストが opt-in フラグの分岐（TF32 経路 vs 既定の
    /// `run_tiled_f32`）を検証するために使う。スレッドローカルにする理由も
    /// 同上（並列テスト間の偽陽性混入を避ける）。
    pub(crate) static TF32_OPTIN_GEMM_LAUNCH_COUNT: Cell<u64> = const { Cell::new(0) };

    /// `run_tiled_f32`／`launch_tiled_f32`／`launch_tiled_f32_resident` の
    /// 3 入口が [`tiled_f32_kernel_kind`] の判定で
    /// [`TiledF32Kernel::Pipeline`] 側（cp.async 3 stage パイプライン。
    /// #1033・#1137 で本番結線）へ実際にルーティングした回数。
    /// `TF32_OPTIN_GEMM_LAUNCH_COUNT` と同型の可観測点で、実機なしの単体
    /// テストが形状条件付き分岐（整列形状 → pipeline／非整列形状 →
    /// classic）を検証するために使う。スレッドローカルにする理由も同上
    /// （並列テスト間の偽陽性混入を避ける）。
    pub(crate) static TILED_PIPELINE_LAUNCH_COUNT: Cell<u64> = const { Cell::new(0) };

    /// `TILED_PIPELINE_LAUNCH_COUNT` のうち、実際に 128×64 側
    /// （[`TiledPipelineTile::Bm128Bn64`]。[`tiled_pipeline_tile_kind`]
    /// の形状条件付き選択）へルーティングした回数（イシュー #1344）。
    /// `TILED_PIPELINE_128X64_PRODUCTION_ENABLED` が `false` の既定状態
    /// では常に 0 のまま増加しない（第 2 スロットが常に `None` のため）。
    pub(crate) static TILED_PIPELINE_128X64_LAUNCH_COUNT: Cell<u64> = const { Cell::new(0) };

    /// VJP 専用 NT/TN 転置入口（`run_tiled_f32_nt`／`run_tiled_f32_tn`／
    /// `launch_tiled_f32_resident_nt`。イシュー #1214）が実際に GPU 側
    /// smem 転置カーネル（`transpose_smem_f32`）へ起動した回数。
    /// `TILED_PIPELINE_LAUNCH_COUNT` と同型の可観測点で、実機なしの単体
    /// テストが「NT/TN 判定が効いて `ops.rs::GEMM_HOST_REPACK_COUNT` の
    /// `contiguous()` フォールバックを通っていないこと」を検証するために
    /// 使う。スレッドローカルにする理由も同上（並列テスト間の偽陽性混入
    /// を避ける）。
    pub(crate) static GEMM_TRANSPOSED_ENTRY_LAUNCH_COUNT: Cell<u64> = const { Cell::new(0) };
}

/// naive／tiled GEMM カーネル（f32/f16 各 2 種）のコンパイル済みハンドルを保持する。
///
/// `stream` は [`CudaDevice`] から `Arc` クローンで受け取る（`device.rs` の
/// 共有契約どおり）。`new` 時に 4 カーネルを一括コンパイルするのは、
/// `nvrtc::compile_ptx` の呼び出し契約「`Box::leak` によるアーキテクチャ
/// 文字列リークはデバイスあたり定数回に限る」を守るためであり、
/// `run_naive_*`／`run_tiled_*` 呼び出しのたびに再コンパイルしない。
pub struct CudaGemm {
    stream: Arc<CudaStream>,
    /// 出力バッファのサイズクラス別プール（イシュー #1020・REQ-14）。
    /// `run_f32_kernel`／`run_tiled_bias_act_f32` の `alloc_zeros::<f32>`
    /// 直接呼び出しを置換する（`crate::pool` モジュール冒頭参照）。
    /// `context_cache::cached_allocator` 経由で `device` の
    /// `CudaContext` 単位のプロセスワイドプールを共有するため、
    /// `CudaGemm` の複数インスタンス（`new_with_tf32_staged_swizzle`
    /// 等の診断用変種を含む）が同一 `device`（＝同一 ordinal かつ
    /// 同一 `CudaContext`）から構築されれば同じプール状態を共有する。
    /// ordinal だけでなく context の同一性も一致条件に含めるのは、
    /// `CudaDevice::new` が公開関数のため同一 ordinal でも異なる
    /// `CudaContext` を持つ複数インスタンスが作られうるため
    /// （`context_cache.rs::ContextKey` のドキュメントコメント参照。
    /// codex-review 指摘。イシュー #1020 PR #1061）。
    allocator: Arc<CudaAllocator>,
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
    /// イシュー #856。GB10 実機 A/B（2026-08-22・§7.4.1 サイズ条件付き
    /// 新基準で採用: 4096 で ×1.54・512〜2048 劣化 5% 以内。
    /// `docs/perf/cuda-gemm-swizzle-ab.md` §7.6/§7.7.6 参照）を根拠に
    /// `gemm_mma.rs::CudaMmaGemm::mma_f16_swizzle`（イシュー #782）と
    /// 同型のサイズ条件付き適用機構を `wmma_tf32_staged` へ追加結線した
    /// swizzle 変種ハンドル。`Some(_)` は [`new`](Self::new) が
    /// `wmma_tf32_staged` のコンパイルに成功し、かつ `device.
    /// multiprocessor_count()` の実測に成功した場合（`run_wmma_tf32`／
    /// `launch_wmma_tf32` の staged 分岐は形状ごとに
    /// [`Self::should_launch_wmma_tf32_staged_swizzle`] で本フィールドと
    /// `wmma_tf32_staged`（base）のいずれを起動するか判定する）。`None` は
    /// `wmma_tf32_staged` 自体が `None`（staged 経路が使用不能）、SM 数が
    /// 取得できなかった場合、または変種コンパイルが失敗した場合
    /// （fail-soft。理由は [`Self::wmma_tf32_staged_swizzle_unavailable_reason`]
    /// 参照）のいずれか。
    wmma_tf32_staged_swizzle: Option<CudaFunction>,
    /// [`Self::wmma_tf32_staged_swizzle`] に適用したグルーピング幅
    /// （`swizzle::select_swizzle_group_width`。`examples/cuda_floor_bench.rs`
    /// の起動時診断が可観測にするために使う。`gemm_mma.rs::
    /// CudaMmaGemm::swizzle_group_width` と同型）。
    wmma_tf32_staged_swizzle_group_width: Option<u32>,
    /// [`new`](Self::new) が `wmma_tf32_staged`（base）のコンパイルに成功し
    /// SM 数実測にも成功したにもかかわらず、swizzle 変種のソース生成・
    /// NVRTC コンパイルに失敗した場合の理由文字列（`wmma_tf32_staged_error`
    /// と同型の fail-soft 方針。base の可用性へは波及させない）。
    wmma_tf32_staged_swizzle_error: Option<String>,
    /// イシュー #1033。`kernels_tiled_pipeline::tiled_pipeline_f32_source()`
    /// （cp.async 多段パイプラインを Tensor Core 不使用の FP32 SIMT 経路へ
    /// 移植した変種カーネル。既定 3 stage）のコンパイル済みハンドル。
    /// `wmma_tf32` 系と同じ理由（cp.async は Ampere〈sm_80〉以降限定で
    /// naive/tiled の 5 カーネルより失敗しうる環境が広い）で `Option` に
    /// し、コンパイル失敗を `new` の早期 return に合流させない
    /// （`Self::wmma_tf32` フィールドのドキュメンテーションコメントと同型
    /// の fail-soft 方針）。本イシューのスコープでは `run_tiled_f32`
    /// （既定本番経路）を置き換えず、[`CudaGemm::run_tiled_pipeline_f32`]
    /// で明示的に呼べる選択可能な変種として追加するに留める
    /// （`kernels_tiled_pipeline.rs` 冒頭コメント「位置づけ・非結線」）。
    tiled_pipeline: Option<TiledPipelineFunction>,
    /// `tiled_pipeline` が `None` の場合の失敗理由。`wmma_tf32_error` と
    /// 同じ理由で文字列化して保持する。
    tiled_pipeline_error: Option<String>,
    /// イシュー #1344。128×64×16 pipeline カーネル（`TiledPipelineTile::
    /// Bm128Bn64`）の第 2 スロット。`tiled_pipeline`（64×64。常に
    /// コンパイルされる）に加えて[`TILED_PIPELINE_128X64_PRODUCTION_ENABLED`]
    /// が `true` の場合のみ追加でコンパイルする（「置換」ではなく
    /// 「追加」意味論。#1343 が導入した単一スロット置換方式〈const `true`
    /// で 64×64 の代わりに 128×64 のみをコンパイル〉から、GB10 実機の
    /// 純カーネル時間比較（#1344）に基づく形状条件付き選択（
    /// [`tiled_pipeline_tile_kind`]）へ拡張するために追加した）。既定
    /// `false` では一切コンパイルされず常に `None`（JIT コスト・
    /// `select_tiled_f32_kernel` の分岐先とも完全不変）。診断専用の
    /// `new_with_tiled_pipeline_128x64`（`internal-diagnostics`）は本
    /// フィールドではなく従来どおり `tiled_pipeline` を直接差し替える
    /// （既存 bit 一致テスト T7〜T11 の契約を変えないため）。
    tiled_pipeline_128x64: Option<TiledPipelineFunction>,
    /// `tiled_pipeline_128x64` が `None` の場合の失敗理由。
    /// `tiled_pipeline_error` と同じ理由で文字列化して保持する。
    tiled_pipeline_128x64_error: Option<String>,
    /// イシュー #1214。VJP 専用 NT/TN 転置入口（`run_tiled_f32_nt`／
    /// `run_tiled_f32_tn`／`launch_tiled_f32_resident_nt`）が使う GPU 側
    /// smem 転置カーネル（`kernels_transpose::transpose_smem_source_f32(false)`
    /// パディングのみ変種。`transpose.rs::CudaTranspose` とは独立の
    /// コンパイル単位・ハンドルとして本構造体に直接保持する。同一
    /// カーネルを `CudaTranspose::new`〈7 カーネル eager コンパイル・
    /// `load_function_cached` を経由しない〉から借用すると、fresh モード
    /// 初回 backward に大きな固定費が乗るため採用しない
    /// （`docs/matmul-vjp-zero-copy-decision.md` §4.3「不採用: `CudaTranspose`
    /// キャッシュ」）。転置カーネル自体は `#include` を使わず全 compute
    /// capability で成立するため通常は失敗しないが、`wmma_tf32` 系と同じ
    /// fail-soft 方針（`CudaGemm::new` の早期 return には合流させず
    /// naive/tiled 系の可用性を道連れにしない）を踏襲する。
    transpose_smem_f32: Option<CudaFunction>,
    /// `transpose_smem_f32` が `None` の場合の失敗理由。`wmma_tf32_error`
    /// と同じ理由で文字列化して保持する。
    transpose_smem_f32_error: Option<String>,
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
/// `kernels::TILED_F16`（`TILE`=32 基準）は各タイル反復で `t * TILE +
/// threadIdx.x`（`threadIdx.x` は最大 `TILE - 1`）を C の `int` 算術で計算
/// し `a_col`／`b_row` を得る。この値は `k` に近い最終タイルで最大
/// `k + TILE - 2` 程度に達しうるため、`k` が `i32::MAX - (TILE - 1)` を
/// 超えると当該算術が i32 の範囲でオーバーフローしうる（実行前ガード。
/// `validate_gemm_dims` の i32 積ガードとは独立に、tiled 固有のタイル
/// インデックス算術を保護する）。
///
/// **`kernels::TILED_F32`（イシュー #1032・`TILED_F32_BK`=16 基準）との
/// 関係**: `TILED_F32` の同種算術（`t * BK + kk`。`kk` は最大 `BK - 1`＝
/// 15）が要求する上限は `i32::MAX - (BK - 1)` であり、`TILE`（32）> `BK`
/// （16）のため本関数（`TILE` 基準）が計算する上限は `TILED_F32` の実際の
/// 必要上限より**厳しい**（より小さい `k` で弾く安全側）。したがって
/// `TILED_F32` に対しても本関数をそのまま流用してよく、`BK` 専用の別関数
/// を新設する必要はない（実装計画 §3.4）。
///
/// `run_tiled_f32`／`run_tiled_f16` からのみ呼ばれ、naive 経路の契約
/// （`validate_gemm_dims` のみ）は変更しない。
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

/// tiled pipeline カーネル（イシュー #1033）の cp.async 16 バイト転送粒度が
/// 要求するグローバル側整列制約を検証する。[`wmma_tf32_staged_alignment_ok`]
/// と同一の根拠（A の行ストライドは `k`、B の行ストライドは `n` であり、
/// 共有メモリ側のタイル幅が 4 の倍数であることと合わせて `k % 4 == 0 &&
/// n % 4 == 0` を満たさない限り 16 バイト境界からずれうる。
/// `kernels_tiled_pipeline.rs` 冒頭コメント「整列制約」参照）。満たさない
/// 形状は `run_tiled_pipeline_f32` が `CudaError::InvalidShape` を返す
/// （`wmma_tf32_staged_alignment_ok` はフォールバック経路選択の条件だが、
/// tiled pipeline は他経路へのフォールバックを持たない単独の選択可能
/// 変種のため fail-closed に拒否する）。
pub(crate) fn tiled_pipeline_alignment_ok(n: u32, k: u32) -> bool {
    n.is_multiple_of(4) && k.is_multiple_of(4)
}

/// tiled pipeline カーネルの cp.async 16 バイト転送粒度が要求する、A
/// バッファ**先頭ポインタ自体**の整列制約を検証する（[`tiled_pipeline_alignment_ok`]
/// が保証するのは行内・行間ストライドの整列のみで、ビュー自体の開始位置は
/// 別問題。codex-review P0／Cursor Bugbot High 指摘・PR #1164）。
///
/// `CudaGemm::launch_tiled_f32_resident` が受け取る `a_dev: &CudaView<'_,
/// f32>` は `fandhe_ai_autodiff::optim::device_store::DeviceParamStore`
/// が保持する 1 本の連結バッファ（cudaMalloc 由来。CUDA ドライバは
/// `cudaMalloc`/`cuMemAlloc` の返すポインタを少なくとも 256 バイト境界に
/// 整列することを保証する）内の任意要素オフセット `a_offset`（要素単位）
/// から切り出した部分ビューであり、そのオフセットが 4 要素（f32 4 個 =
/// 16 バイト）の倍数でない限り、切り出したビューの先頭ポインタ自体が
/// cp.async の 16 バイト境界要求を満たさない（基底バッファが 256 バイト
/// 境界に整列している前提の下で `base + a_offset * 4` が 16 バイト整列
/// となるのは `a_offset % 4 == 0` のときのみ）。`run_tiled_f32`／
/// `launch_tiled_f32` は毎回新規 upload/確保した全体バッファ（`a_offset`
/// は常に 0）のみを渡すため本関数は常に真を返し実質無効化されるが、
/// [`CudaGemm::launch_tiled_f32_resident`] 経由の呼び出しでは実際に非 0
/// オフセットが起こりうるため、[`tiled_f32_kernel_kind`] の判定に
/// `n`/`k` の行整列チェックと独立に組み込む（非整列オフセットでは常に
/// classic へ fail-closed にフォールバックする契約）。
pub(crate) fn tiled_pipeline_offset_aligned(a_offset: usize) -> bool {
    a_offset.is_multiple_of(4)
}

/// tiled pipeline カーネル固有の `k` 追加上限検証（イシュー #1033）。
/// [`validate_tiled_k_bound`] と同一の理由（各 K タイル反復で `t * TP_BK +
/// col0`〈`col0` は最大 `TP_BK - 4`〉相当を C の `int` 算術で計算するため、
/// `k` が `i32::MAX - (TP_BK - 1)` を超えると当該算術が i32 の範囲で
/// オーバーフローしうる）で、`TP_BK` 基準に独立して検証する。
pub(crate) fn validate_tiled_pipeline_k_bound(k: u32) -> Result<(), CudaError> {
    let limit = i32::MAX as u32 - (kernels_tiled_pipeline::TP_BK - 1);
    if k > limit {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "k must not exceed i32::MAX - (TP_BK - 1) for tiled pipeline kernel \
                 tile-index arithmetic: k={k}, limit={limit}, TP_BK={}",
                kernels_tiled_pipeline::TP_BK
            ),
        });
    }
    Ok(())
}

/// tiled f32 経路（[`CudaGemm::run_tiled_f32`]／[`CudaGemm::launch_tiled_f32`]／
/// [`CudaGemm::launch_tiled_f32_resident`] の 3 入口）が実際にどちらの
/// カーネルへ分岐したかを表す（観測用の可観測 API。
/// `wmma_tf32_staged_swizzle_group_width` と同型。イシュー #1137）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiledF32Kernel {
    /// `kernels::TILED_F32`（32×32 共有メモリタイル・同期ロード。
    /// #1032 レジスタブロッキング版）。
    Classic,
    /// `kernels_tiled_pipeline::gemm_tiled_pipeline_f32`（cp.async 3 stage
    /// パイプライン。#1033・GB10 実測に基づき #1137 で本番結線）。
    Pipeline,
}

/// [`CudaGemm::launch_tiled_f32_pooled`]（イシュー #1182 診断専用）の
/// カーネル選択トグル。用途・`Classic` 固定が必要な理由は同メソッドの
/// doc コメントを参照。[`TiledF32Kernel`]（`select_tiled_f32_kernel` の
/// **結果**を表す可観測型）とは別物で、こちらは**呼び出し前**の選択
/// 指定である。`gemm_reuse_phase_diag_tests`（`#[cfg(test)]` 兄弟
/// モジュール）専用のため `#[cfg(test)]` で本体ビルドから除外する
/// （通常ビルドでの dead-code 化を避ける。`launch_tiled_f32_pooled` も
/// 同様）。
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiagTiledF32Kernel {
    /// 本番と同じ形状条件付き自動選択（[`CudaGemm::select_tiled_f32_kernel`]
    /// をそのまま呼ぶ）。
    Select,
    /// 常に classic（`tiled_f32`）へ固定する。
    Classic,
}

/// 形状・パイプラインカーネル可用性から tiled f32 経路 3 入口が使う
/// カーネルを決定する純粋関数（GPU 不要。`gemm.rs` 内 `#[cfg(test)]` で
/// 単体検査する。イシュー #1137）。
///
/// パイプラインは cp.async 16 バイト整列制約（[`tiled_pipeline_alignment_ok`]
/// による行内・行間ストライドの整列、および [`tiled_pipeline_offset_aligned`]
/// による A バッファ先頭ポインタ自体の整列。codex-review P0／Cursor
/// Bugbot High 指摘・PR #1164 で `a_offset` 引数を追加）を満たし、かつ
/// `new` 時のコンパイルに成功している（`pipeline_available` が真）場合に
/// のみ選び、それ以外は常に classic へ fail-closed にフォールバックする
/// （非対応環境・非整列形状・非整列オフセットのいずれでも本番既定経路が
/// 壊れないことを保証する契約。`run_tiled_pipeline_f32`
/// の単独選択可能変種としての fail-closed 拒否契約とは異なり、こちらは
/// 常に成功するフォールバックを持つ）。
///
/// `a_offset`（要素単位）は [`CudaGemm::launch_tiled_f32_resident`]
/// 経由でのみ非 0 になりうる（`run_tiled_f32`／`launch_tiled_f32` は
/// 常に全体バッファ＝オフセット 0 を渡す）。
///
/// `k` の追加境界検証は呼び出し元（`run_tiled_f32` 等）が
/// [`validate_tiled_k_bound`]（`TILE`=32 基準）を先に通しており、これは
/// パイプライン側の [`validate_tiled_pipeline_k_bound`]（`TP_BK`=16 基準）
/// より厳しい（同関数のドキュメンテーションコメント参照）ため、本関数は
/// `k` の境界を再検証しない。
pub(crate) fn tiled_f32_kernel_kind(
    pipeline_available: bool,
    a_offset: usize,
    n: u32,
    k: u32,
) -> TiledF32Kernel {
    if pipeline_available
        && tiled_pipeline_offset_aligned(a_offset)
        && tiled_pipeline_alignment_ok(n, k)
    {
        TiledF32Kernel::Pipeline
    } else {
        TiledF32Kernel::Classic
    }
}

/// tiled pipeline カーネル（イシュー #1033・既定 stage 数固定）の
/// [`crate::nvrtc::CudaKernelDescriptor`] を構築する。`kernel_specs` の
/// 固定配列には含めない（本モジュールは #1032 との並行実装衝突回避のため
/// 独立ファイルに切り出した選択可能変種であり、本番必須 5 カーネル・
/// WMMA(TF32) 系 3 カーネルの一覧管理とは別枠で扱う。`kernels_tiled_pipeline.rs`
/// 冒頭コメント「位置づけ・非結線」参照）。
fn tiled_pipeline_descriptor() -> Result<CudaKernelDescriptor, CudaError> {
    CudaKernelDescriptor::new_with_compiled_dims(
        "tiled_pipeline_f32",
        fandhe_ai_tensor_core::dispatch::GemmShape::new(0, 0, 0),
        kernels_tiled_pipeline::TP_BM,
        kernels_tiled_pipeline::TP_BN,
        kernels_tiled_pipeline::TP_BK,
        kernels_tiled_pipeline::TP_DEFAULT_STAGES,
        fandhe_ai_tensor_core::dispatch::DType::F32,
        CompiledDims::DYNAMIC_ALL,
    )
}

/// [`TiledPipelineFunction`] が保持するタイル構成タグ（イシュー #1343）。
///
/// [`CudaGemm::select_tiled_f32_kernel`]（本番既定経路）・
/// [`tiled_pipeline_launch_config`]（grid/block 構成の導出）がこのタグを
/// 見て起動 config を決める。64×64（既存・#1033）と 128×64（本イシュー・
/// #1343）はブロックタイル寸法・スレッド当たり担当要素数が異なり、
/// `LaunchConfig` の grid_dim が別式になるため、`TiledPipelineFunction` に
/// 埋め込むことで「別タイル構成のハンドルへ誤って旧タイル用の launch
/// config を適用する」クラスの不整合を型で防ぐ
/// （`kernels_tiled_pipeline_128x64.rs` 冒頭コメント「位置づけ」参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiledPipelineTile {
    /// `kernels_tiled_pipeline::gemm_tiled_pipeline_f32`（64×64×16・4×4
    /// レジスタブロック。イシュー #1033・#1137 で本番結線済み）。
    Bm64Bn64,
    /// `kernels_tiled_pipeline_128x64::gemm_tiled_pipeline_128x64_f32`
    /// （128×64×16・8×4 レジスタブロック・A フラグメント XOR スウィズル。
    /// イシュー #1343。本番結線は
    /// [`TILED_PIPELINE_128X64_PRODUCTION_ENABLED`] で opt-in ゲート）。
    Bm128Bn64,
}

/// tiled pipeline カーネルのコンパイル済みハンドル（イシュー #1033・
/// codex-review P0〈PR #1071〉対応）。
///
/// [`CudaGemm::launch_tiled_pipeline_f32`] は safe 公開 API でありながら
/// `unsafe { launch(..) }` を行うため、渡された関数ハンドルが
/// `gemm_tiled_pipeline_f32`（`TP_BM`/`TP_BN`/`TP_BK` 固定・
/// `tiled_pipeline_launch_config` が仮定する grid/block 構成・
/// カーネル引数シグネチャ）と一致することを型で保証する必要がある。
/// 生の [`CudaFunction`] を直接引数に取ると、呼び出し元が全く無関係な
/// （別シグネチャ・別 launch config 前提の）関数ハンドルを safe に渡せて
/// しまい、`unsafe` launch 側の GPU 範囲外アクセス等の前提が崩れる。
///
/// 本型はフィールドを非公開にし、[`CudaGemm::compile_tiled_pipeline_variant`]
/// （64×64・公開コンストラクタ）・[`CudaGemm::compile_tiled_pipeline_128x64_variant`]
/// （128×64・イシュー #1343 で追加した公開コンストラクタ）と本モジュール内の
/// `compile_tiled_pipeline`／`compile_tiled_pipeline_128x64`
/// （`CudaGemm::new`／`CudaGemm::new_with_tiled_pipeline_128x64` 経由）以外に
/// 生成手段を持たない。いずれも `tiled_pipeline_descriptor` 系の記述子生成
/// 内で `TP_BM`/`TP_BN`/`TP_BK`（または 128×64 側の `TP128_BM`/`TP128_BN`/
/// `TP128_BK`）固定・対応する `func_name`（`"gemm_tiled_pipeline_f32"` また
/// は `"gemm_tiled_pipeline_128x64_f32"`）を経由してのみ [`CudaFunction`] を
/// 得るため、この型の値は必ず期待するシグネチャ・タイル構成（保持する
/// [`TiledPipelineTile`] タグと整合する側）のカーネルを指すことが構築時点
/// で保証される（検証済みハンドルへの封じ込め）。
///
/// `context_ptr`（codex-review P1 指摘・PR #1071）: 上記のシグネチャ・
/// タイル構成の一致だけでは、`compile_tiled_pipeline_variant` に**別の
/// `CudaDevice`（＝別 GPU・別 `CudaContext`）** を渡して得たハンドルを
/// `CudaGemm::launch_tiled_pipeline_f32`（別インスタンス。別 context の
/// `stream` を保持）へ safe Rust から渡せてしまう不変条件の穴が残る。
/// 複数 GPU・複数 context 利用時にこれを行うと、context 固有の
/// `CUfunction` と、別 context の `stream`／デバイスバッファを混在させた
/// `unsafe` launch に到達し、CUDA driver レベルの未定義動作
/// （invalid device context・実質的な OOB リスク）を招きうる。
/// `Arc<CudaContext>` のポインタ同一性（`context_cache.rs::ContextKey` と
/// 同じ識別方式）をハンドルへ焼き込み、`launch_tiled_pipeline_f32` が
/// 起動直前に `self.stream.context()` と fail-closed に一致検証すること
/// で、非公開フィールドの型保証をシグネチャだけでなく生成元 context にも
/// 拡張する。`CudaFunction` は内部で `Arc<CudaModule>`→`Arc<CudaContext>`
/// を強参照し続けるため（cudarc 0.19.8 `driver::safe::core::CudaFunction`
/// の `module` フィールド）、本型が生存する間はポインタが指す
/// `CudaContext` の再利用（ABA）は起こらない。
pub struct TiledPipelineFunction(CudaFunction, usize, TiledPipelineTile);

impl TiledPipelineFunction {
    /// 起動に使う内部の [`CudaFunction`] を返す。本モジュール限定
    /// （呼び出し元が生ハンドルを取り出して検証を迂回できないようにする
    /// ため `pub(crate)` に留める）。
    fn as_cuda_function(&self) -> &CudaFunction {
        &self.0
    }

    /// 生成元 `CudaDevice` の `Arc<CudaContext>` ポインタ同一性識別子。
    /// [`CudaGemm::launch_tiled_pipeline_f32`] が起動直前の context 一致
    /// 検証に使う（型ドキュメントコメント参照）。
    fn context_ptr(&self) -> usize {
        self.1
    }

    /// このハンドルが保持するタイル構成タグ（[`TiledPipelineTile`]）。
    /// [`Self::launch_config`] が grid/block 構成を導出するために使う。
    fn tile(&self) -> TiledPipelineTile {
        self.2
    }

    /// このハンドルのタイル構成に対応する [`LaunchConfig`] を導出する
    /// （[`tiled_pipeline_launch_config`] へ委譲。イシュー #1343 で
    /// 64×64 固定だった呼び出し元〈`run_tiled_pipeline_f32`・
    /// `launch_tiled_pipeline_f32`・`select_tiled_f32_kernel`〉をこの
    /// メソッド経由へ一般化し、128×64 ハンドルでも正しい grid_dim を
    /// 得られるようにする）。
    fn launch_config(&self, m: u32, n: u32) -> LaunchConfig {
        tiled_pipeline_launch_config(self.tile(), m, n)
    }
}

/// tiled pipeline カーネル（既定 stage 数固定）を単独で
/// [`load_function_cached`] 経由でロードする。`compile_wmma_tf32` と同じ
/// 理由（cp.async は sm_80 以降限定で失敗しうる環境が広い）で `CudaGemm::new`
/// の早期 return には合流させず、呼び出し元で `tiled_pipeline_error` として
/// 退避する。
fn compile_tiled_pipeline(device: &CudaDevice) -> Result<TiledPipelineFunction, CudaError> {
    let descriptor = tiled_pipeline_descriptor()?;
    let func = load_function_cached(
        device,
        descriptor,
        kernels_tiled_pipeline::tiled_pipeline_f32_source(),
        "gemm_tiled_pipeline_f32",
    )?;
    let context_ptr = Arc::as_ptr(device.context()) as usize;
    Ok(TiledPipelineFunction(
        func,
        context_ptr,
        TiledPipelineTile::Bm64Bn64,
    ))
}

/// 128×64×16 pipeline カーネル（イシュー #1343）の
/// [`crate::nvrtc::CudaKernelDescriptor`] を構築する。[`tiled_pipeline_descriptor`]
/// の 128×64 版。`kernel_specs` の固定配列には含めない（64×64 版と同じ
/// 理由。`kernels_tiled_pipeline_128x64.rs` 冒頭コメント「位置づけ」参照）。
fn tiled_pipeline_128x64_descriptor() -> Result<CudaKernelDescriptor, CudaError> {
    CudaKernelDescriptor::new_with_compiled_dims(
        "tiled_pipeline_f32_128x64",
        fandhe_ai_tensor_core::dispatch::GemmShape::new(0, 0, 0),
        kernels_tiled_pipeline_128x64::TP128_BM,
        kernels_tiled_pipeline_128x64::TP128_BN,
        kernels_tiled_pipeline_128x64::TP128_BK,
        kernels_tiled_pipeline_128x64::TP128_DEFAULT_STAGES,
        fandhe_ai_tensor_core::dispatch::DType::F32,
        CompiledDims::DYNAMIC_ALL,
    )
}

/// 128×64×16 pipeline カーネル（既定 stage 数固定）を単独でロードする
/// （[`compile_tiled_pipeline`] の 128×64 版）。[`Self::new`](CudaGemm::new)
/// からは [`TILED_PIPELINE_128X64_PRODUCTION_ENABLED`] が `true` の場合に
/// のみ呼ばれ（既定 `false` のため通常は呼ばれない＝コンパイルコスト
/// なし）、それ以外では [`CudaGemm::new_with_tiled_pipeline_128x64`]
/// （`internal-diagnostics` feature 限定の診断入口）からのみ呼ばれる。
fn compile_tiled_pipeline_128x64(device: &CudaDevice) -> Result<TiledPipelineFunction, CudaError> {
    let descriptor = tiled_pipeline_128x64_descriptor()?;
    let func = load_function_cached(
        device,
        descriptor,
        kernels_tiled_pipeline_128x64::tiled_pipeline_128x64_f32_source(),
        "gemm_tiled_pipeline_128x64_f32",
    )?;
    let context_ptr = Arc::as_ptr(device.context()) as usize;
    Ok(TiledPipelineFunction(
        func,
        context_ptr,
        TiledPipelineTile::Bm128Bn64,
    ))
}

/// 128×64×16 pipeline カーネル（イシュー #1343・#1344）の本番結線スイッチ。
///
/// `false`（既定）: [`CudaGemm::new`] は 64×64 版（[`compile_tiled_pipeline`]。
/// `tiled_pipeline` スロット）のみをコンパイルし、128×64 版
/// （`tiled_pipeline_128x64` 第 2 スロット）は一切コンパイルしない（JIT
/// コスト・`run_tiled_f32` 系 3 入口の挙動とも完全に不変）。128×64 版へは
/// `internal-diagnostics` feature 限定の診断入口
/// （[`CudaGemm::new_with_tiled_pipeline_128x64`]・
/// [`CudaGemm::compile_tiled_pipeline_128x64_variant`]）からのみ到達する。
///
/// `true`: 64×64 版に**加えて**128×64 版も第 2 スロットへ追加コンパイル
/// する（「置換」ではなく「追加」意味論。#1343 時点の単一スロット置換
/// 方式から #1344 で拡張）。`select_tiled_f32_kernel` は形状が
/// [`tiled_pipeline_tile_kind`] の閾値（`TILED_PIPELINE_128X64_MIN_M`/
/// `_MIN_N`/`_MIN_K`）を満たす場合のみ 128×64 側へ分岐し、それ以外は
/// 64×64 側を使う。
///
/// GB10 実機での純カーネル時間比較・形状条件付き結線の可否判断はイシュー
/// #1344 で実施した（`docs/perf/cuda-gemm-tiled-pipeline.md`「#1344」節。
/// 実測記録・採否・閾値の根拠を記載）。`true` への切り替えは同節のゲート
/// C（純カーネル時間の優位性）・ゲート D（本番ディスパッチ非後退）の
/// 両方が成立した場合のみ行う（`gemm_auto.rs::
/// MMA_PRIORITY_PRODUCTION_ENABLED` と同型の opt-in ゲート運用）。
pub(crate) const TILED_PIPELINE_128X64_PRODUCTION_ENABLED: bool = true;

/// [`tiled_pipeline_tile_kind`] が 128×64 側を選ぶ N の下限（要素数）。
/// GB10 実機実測（イシュー #1344。`docs/perf/cuda-gemm-tiled-pipeline.md`
/// 「#1344」節ゲート C。N=256/512/1024/2048/4096 の正方形状 5 回中央値
/// 比較）で確定した値。GPU-only 純カーネル時間比（128×64 ÷ 64×64。
/// `pipeline128x64_over_pipeline3_gpu_only`）の中央値は N=1024: 1.050・
/// N=2048: 1.116・N=4096: 1.353（いずれもゲート C の採用基準 ≥1.05 を
/// 満たす）に対し、N=512: 0.992・N=256: 0.678（採用基準未達。とくに
/// N=256 は 128×64 が明確に劣後）だったため、N=1024 を下限とする
/// （`Option<u32>` にする理由: `u32` の下限値 0 を閾値として使うと
/// `n >= 0` が常に真になり `clippy::absurd_extreme_comparisons` に抵触
/// するため、「常に適用」を型で表現する）。受け入れ条件が正方形状
/// （M=N）の比較のみを求めているため、`tiled_pipeline_tile_kind` は M 軸
/// の閾値を持たない（N/K のみで判定。将来 M 軸の非正方形状での優位性
/// 境界が別途実測された場合は M 軸の閾値を追加する）。
pub(crate) const TILED_PIPELINE_128X64_MIN_N: Option<u32> = Some(1024);

/// [`tiled_pipeline_tile_kind`] が 128×64 側を選ぶ K の下限（要素数）。
/// `TILED_PIPELINE_128X64_MIN_N` と同じ理由・実測根拠（実測は M=N=K の
/// 正方形状のみのため N と同値を採用。非正方 K の独立した優位性境界は
/// 未実測）。
pub(crate) const TILED_PIPELINE_128X64_MIN_K: Option<u32> = Some(1024);

/// `tiled_f32_kernel_kind` が [`TiledF32Kernel::Pipeline`] を返した場合に、
/// [`CudaGemm::select_tiled_f32_kernel`] がさらにどちらのタイル構成
/// （[`TiledPipelineTile`]）を使うかを決める GPU 不要の純粋関数
/// （イシュー #1344。`tiled_f32_kernel_kind` と同型の設計: 形状条件を
/// 満たさない、または 128×64 ハンドル自体が未コンパイル
/// （`has_128x64` が偽。[`TILED_PIPELINE_128X64_PRODUCTION_ENABLED`] が
/// `false` の既定状態を含む）場合は常に安全側の 64×64 へ fail-closed
/// フォールバックする）。
///
/// 閾値（`TILED_PIPELINE_128X64_MIN_N`/`_MIN_K`）は 2 軸ともを満たす場合
/// にのみ 128×64 を選ぶ（AND 条件。`None` は無条件で満たす扱い。M 軸を
/// 持たない理由は `TILED_PIPELINE_128X64_MIN_N` のドキュメンテーション
/// コメント参照）。
pub(crate) fn tiled_pipeline_tile_kind(has_128x64: bool, n: u32, k: u32) -> TiledPipelineTile {
    let n_ok = TILED_PIPELINE_128X64_MIN_N.is_none_or(|min| n >= min);
    let k_ok = TILED_PIPELINE_128X64_MIN_K.is_none_or(|min| k >= min);
    if has_128x64 && n_ok && k_ok {
        TiledPipelineTile::Bm128Bn64
    } else {
        TiledPipelineTile::Bm64Bn64
    }
}

/// VJP 専用 NT/TN 転置入口（イシュー #1214）の GPU 側 smem 転置カーネル
/// （`kernels_transpose::transpose_smem_source_f32(false)`。パディングの
/// み変種。swizzle 変種は #601 での A/B が未計測のため選ばない。
/// `docs/perf/cuda-gemm-transpose-ab.md` §2）を [`load_function_cached`]
/// 経由でロードする。`compile_tiled_pipeline` と同じ理由（`transpose.rs::
/// CudaTranspose::new` の 7 カーネル一括コンパイルとは独立に、`CudaGemm::
/// new` の早期 return には合流させず呼び出し元で `transpose_smem_f32_error`
/// として退避する）。
fn compile_transpose_smem_f32(device: &CudaDevice) -> Result<CudaFunction, CudaError> {
    let descriptor = CudaKernelDescriptor::new_with_compiled_dims(
        "transpose_smem_f32",
        fandhe_ai_tensor_core::dispatch::GemmShape::new(0, 0, 0),
        kernels_transpose::TRANSPOSE_TILE,
        kernels_transpose::TRANSPOSE_TILE,
        1,
        1,
        fandhe_ai_tensor_core::dispatch::DType::F32,
        CompiledDims::DYNAMIC_ALL,
    )?;
    load_function_cached(
        device,
        descriptor,
        &kernels_transpose::transpose_smem_source_f32(false),
        "transpose_smem_f32",
    )
}

/// [`wmma_tf32_launch_config`] の tiled pipeline 版。ブロックタイル
/// `kernels_tiled_pipeline::TP_BM/TP_BN`（64×64）を単位に `div_ceil` で
/// グリッドを構築する。ブロック次元は `TP_BLOCK_THREADS`（256）の 1 次元
/// （カーネル内で `tx = tid % TP_THREADS_X`・`ty = tid / TP_THREADS_X` へ
/// 分解する契約。`kernels_tiled_pipeline.rs` 参照）。末尾ブロックの余剰は
/// カーネル内の手動境界チェック（REQ-8）に委ねる契約は他 GEMM カーネルと
/// 共通。
/// イシュー #1343: `tile` タグ（[`TiledPipelineTile`]）に応じてブロック
/// タイル寸法・スレッド数を切り替える（従来は 64×64 固定だったが、128×64
/// 版〈#1343〉の追加に伴い一般化した。`TiledPipelineFunction::launch_config`
/// 経由でのみ呼ぶ契約とし、呼び出し元がタグと無関係な寸法を誤って選べない
/// ようにする）。末尾ブロックの余剰はカーネル内の手動境界チェック
/// （REQ-8）に委ねる契約はいずれのタイルでも共通。
fn tiled_pipeline_launch_config(tile: TiledPipelineTile, m: u32, n: u32) -> LaunchConfig {
    let (block_m, block_n, block_threads) = match tile {
        TiledPipelineTile::Bm64Bn64 => (
            kernels_tiled_pipeline::TP_BM,
            kernels_tiled_pipeline::TP_BN,
            kernels_tiled_pipeline::TP_BLOCK_THREADS,
        ),
        TiledPipelineTile::Bm128Bn64 => (
            kernels_tiled_pipeline_128x64::TP128_BM,
            kernels_tiled_pipeline_128x64::TP128_BN,
            kernels_tiled_pipeline_128x64::TP128_BLOCK_THREADS,
        ),
    };
    let grid_dim = (n.div_ceil(block_n), m.div_ceil(block_m), 1);
    LaunchConfig {
        grid_dim,
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// [`kernel_specs`] の要素型。診断・ログ用ラベル（`label`）・NVRTC ソース
/// （`source`）・ロードする関数名（`func_name`）に加え、イシュー #1024 で
/// `module_cache`／NVRTC ディスクキャッシュへ結線するためのキー識別
/// メタデータ（`block_m`/`block_n`/`block_k`/`stages`/`dtype`）を持つ。
///
/// `block_*`/`stages` は実際の起動パラメータ（[`launch_config`] 等が別途
/// 保持する）ではなく、[`crate::nvrtc::CudaKernelDescriptor`] のキャッシュ
/// キーを一意にするための識別メタデータに過ぎない（一意性の本体は
/// [`Self::descriptor`] が保持する `source` 全文＋`kernel_name`。
/// `CudaKernelDescriptor` ドキュメンテーションコメント「フィールドは
/// private + getter とし」節参照）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct GemmKernelSpec {
    pub(crate) label: &'static str,
    pub(crate) source: &'static str,
    pub(crate) func_name: &'static str,
    block_m: u32,
    block_n: u32,
    block_k: u32,
    stages: u32,
    dtype: fandhe_ai_tensor_core::dispatch::DType,
}

impl GemmKernelSpec {
    /// `self` から [`crate::nvrtc::CudaKernelDescriptor`] を導出する
    /// （イシュー #1024）。`CudaGemm::new` が構築する各カーネルは shape
    /// 特化を行わない（同一 device 上では常に同一ソース・同一パラメータ
    /// でコンパイルする）ため、`shape` は全次元 sentinel `0`・
    /// `compiled_dims` は [`CompiledDims::DYNAMIC_ALL`] を渡す
    /// （`kernels_mma.rs::RenderedMmaKernel::cache_descriptor` が
    /// shape 特化ありの場合に個別次元を `Static`/`Dynamic` へ振り分けるのと
    /// 対照的に、本経路は形状非依存の固定カーネルのみを扱うため常に
    /// 全次元動的固定でよい）。
    pub(crate) fn descriptor(&self) -> Result<CudaKernelDescriptor, CudaError> {
        CudaKernelDescriptor::new_with_compiled_dims(
            self.label,
            fandhe_ai_tensor_core::dispatch::GemmShape::new(0, 0, 0),
            self.block_m,
            self.block_n,
            self.block_k,
            self.stages,
            self.dtype,
            CompiledDims::DYNAMIC_ALL,
        )
    }
}

/// `CudaGemm::new` が 1 回の構築で NVRTC コンパイル・ロードする 8 カーネル
/// の [`GemmKernelSpec`] 組の**唯一の真実源**。
///
/// 順序は naive f32/f16・tiled f32/f16・tiled_bias_act_f32（先頭 5 件、
/// `?` 早期 return に合流する必須カーネル）・wmma_tf32・wmma_tf32_opt・
/// wmma_tf32_staged（末尾 3 件、失敗を `Option` へ退避するフォールバック
/// カーネル）の順に固定する。先頭 5 件は [`CudaGemm::new`] が本関数の
/// スライスを直接ループして [`crate::module_cache::load_function_cached`]
/// （イシュー #1024。module_cache／NVRTC ディスクキャッシュ経由）を呼ぶ
/// ため実装上ずれようがなく、末尾 3 件は [`compile_wmma_tf32`]／
/// [`compile_wmma_tf32_opt`]／[`compile_wmma_tf32_staged`] がそれぞれ本関数
/// の対応要素を取得する（これらは失敗を `Option` へ退避するフォールバック
/// 方式のため、先頭 5 件と同じ単純ループへは合流させない。個別関数化の
/// 理由は各関数のドキュメンテーションコメント参照）。
///
/// `crates/backend-cuda/src/init_cost_diag_tests.rs`（イシュー #926 の
/// フェーズ分解診断テスト）も本関数をそのまま呼び出し、8 カーネルの
/// 内訳を個別計測する。以前は同テストファイル側に本関数と同一内容を
/// 手作業で複製していたが、本番側でカーネルの追加・削除・ソース差し替え
/// が起きても診断側の一覧が追従せず乖離しうるため、単一の `pub(crate)`
/// 関数へ統合した（Review #945 P2 指摘）。
pub(crate) fn kernel_specs() -> [GemmKernelSpec; 8] {
    use fandhe_ai_tensor_core::dispatch::DType;
    [
        GemmKernelSpec {
            label: "naive_f32",
            source: kernels::NAIVE_F32,
            func_name: "gemm_naive_f32",
            block_m: NAIVE_BLOCK_DIM.0,
            block_n: NAIVE_BLOCK_DIM.1,
            block_k: 1,
            stages: 1,
            dtype: DType::F32,
        },
        GemmKernelSpec {
            label: "naive_f16",
            source: kernels::NAIVE_F16,
            func_name: "gemm_naive_f16",
            block_m: NAIVE_BLOCK_DIM.0,
            block_n: NAIVE_BLOCK_DIM.1,
            block_k: 1,
            stages: 1,
            dtype: DType::F16,
        },
        GemmKernelSpec {
            label: "tiled_f32",
            source: kernels::TILED_F32,
            func_name: "gemm_tiled_f32",
            block_m: kernels::TILED_F32_BM,
            block_n: kernels::TILED_F32_BN,
            block_k: kernels::TILED_F32_BK,
            stages: 1,
            dtype: DType::F32,
        },
        GemmKernelSpec {
            label: "tiled_f16",
            source: kernels::TILED_F16,
            func_name: "gemm_tiled_f16",
            block_m: kernels::TILE,
            block_n: kernels::TILE,
            block_k: kernels::TILE,
            stages: 1,
            dtype: DType::F16,
        },
        GemmKernelSpec {
            label: "tiled_bias_act_f32",
            source: kernels::TILED_BIAS_ACT_F32,
            func_name: "gemm_tiled_bias_act_f32",
            block_m: kernels::TILED_F32_BM,
            block_n: kernels::TILED_F32_BN,
            block_k: kernels::TILED_F32_BK,
            stages: 1,
            dtype: DType::F32,
        },
        GemmKernelSpec {
            label: "wmma_tf32",
            source: kernels::WMMA_TF32_F32,
            func_name: "gemm_wmma_tf32",
            block_m: kernels::WMMA_TF32_BLOCK_M,
            block_n: kernels::WMMA_TF32_BLOCK_N,
            block_k: kernels::WMMA_TF32_K_TILE,
            stages: 1,
            dtype: DType::F32,
        },
        GemmKernelSpec {
            label: "wmma_tf32_opt",
            source: kernels_wmma_opt::wmma_tf32_f32_opt_source(),
            func_name: "gemm_wmma_tf32_opt",
            block_m: kernels_wmma_opt::WMMA_TF32_OPT_BLOCK_M,
            block_n: kernels_wmma_opt::WMMA_TF32_OPT_BLOCK_N,
            block_k: kernels_wmma_opt::WMMA_TF32_OPT_K_TILE,
            stages: 1,
            dtype: DType::F32,
        },
        GemmKernelSpec {
            label: "wmma_tf32_staged",
            source: kernels_wmma_opt::wmma_tf32_f32_staged_source(),
            func_name: "gemm_wmma_tf32_staged",
            block_m: kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_M,
            block_n: kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_N,
            block_k: kernels_wmma_opt::WMMA_TF32_STAGED_K_TILE,
            stages: kernels_wmma_opt::WMMA_TF32_STAGED_STAGES,
            dtype: DType::F32,
        },
    ]
}

/// WMMA(TF32) カーネル（[`kernel_specs`] index 5）を単独で
/// [`load_function_cached`]（module_cache／NVRTC ディスクキャッシュ経由。
/// イシュー #1024）経由でロードする。`CudaGemm::new` から呼ばれ、戻り値の
/// `Err` は naive/tiled 4 カーネルの `?` 早期 return には合流させず、
/// 呼び出し元で `wmma_tf32_error` として退避する（[`CudaGemm::wmma_tf32`]
/// フィールドのドキュメンテーションコメント参照。レビュー指摘 #62）。
fn compile_wmma_tf32(device: &CudaDevice) -> Result<CudaFunction, CudaError> {
    let spec = kernel_specs()[5];
    load_function_cached(device, spec.descriptor()?, spec.source, spec.func_name)
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

/// `kernels::TILED_F32_BM`／`TILED_F32_BN`（ブロックタイル一辺）を単位に
/// `m`/`n` を `div_ceil` で包含するグリッド次元を構築する（イシュー
/// #1032）。
///
/// [`wmma_tf32_launch_config`] と同じ理由（該当コメント参照）: レジスタ
/// ブロッキング版 tiled f32 カーネルはスレッド形状（[`TILED_F32_BLOCK_DIM`]、
/// 16x16=256 スレッド）とタイル一辺（64×64）が「ブロック次元＝タイル
/// 一辺」の旧 [`launch_config`] 前提から外れるため、専用のグリッド計算
/// 関数として分離する。末尾ブロックの余剰はカーネル内の手動境界チェック
/// （REQ-8）に委ねる契約は共通。
fn tiled_f32_launch_config(m: u32, n: u32) -> LaunchConfig {
    let grid_dim = (
        n.div_ceil(kernels::TILED_F32_BN),
        m.div_ceil(kernels::TILED_F32_BM),
        1,
    );
    LaunchConfig {
        grid_dim,
        block_dim: TILED_F32_BLOCK_DIM,
        shared_mem_bytes: 0,
    }
}

/// WMMA(TF32) opt カーネル（[`kernel_specs`] index 6）を単独で
/// [`load_function_cached`] 経由でロードする。[`compile_wmma_tf32`] と
/// 同じ理由（レビュー指摘 #62 の踏襲）で `CudaGemm::new` の早期 return には
/// 合流させず、呼び出し元で `wmma_tf32_opt_error` として退避する。
fn compile_wmma_tf32_opt(device: &CudaDevice) -> Result<CudaFunction, CudaError> {
    let spec = kernel_specs()[6];
    load_function_cached(device, spec.descriptor()?, spec.source, spec.func_name)
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

/// WMMA(TF32) opt-staged カーネル（イシュー #500・[`kernel_specs`] index 7）
/// を単独で [`load_function_cached`] 経由でロードする。
/// [`compile_wmma_tf32_opt`] と同じ理由で `CudaGemm::new` の早期 return には
/// 合流させず、呼び出し元で `wmma_tf32_staged_error` として退避する。
fn compile_wmma_tf32_staged(device: &CudaDevice) -> Result<CudaFunction, CudaError> {
    let spec = kernel_specs()[7];
    load_function_cached(device, spec.descriptor()?, spec.source, spec.func_name)
}

/// [`compile_wmma_tf32_staged`]（[`kernel_specs`] index 7）と同一のブロック
/// タイル・段数メタデータを流用しつつ `kernel_name` のみ
/// `"wmma_tf32_staged_swizzle"` へ差し替えた
/// [`crate::nvrtc::CudaKernelDescriptor`] を構築する。
///
/// swizzle 変種は `group_width`（実行時に `device.multiprocessor_count()`
/// から導出。[`CudaGemm::new`] 参照）によってソース文字列が変わるため
/// [`kernel_specs`] の固定配列には含めない。`kernel_name` を base
/// （`"wmma_tf32_staged"`）と分けるのは、同一 `kernel_name`・異なる
/// `source` のキーが `CudaKernelCacheKey` の `Hash`/`Eq`（`source` を含む）
/// 上は別エントリになる（ソース全文比較のため誤ヒットはしない）ものの、
/// ディスクキャッシュのディレクトリ命名（`kernel.<name>.<hash>`）が
/// base と衝突しないようにするため（イシュー #1024）。
fn wmma_tf32_staged_swizzle_descriptor() -> Result<CudaKernelDescriptor, CudaError> {
    let base = kernel_specs()[7];
    CudaKernelDescriptor::new_with_compiled_dims(
        "wmma_tf32_staged_swizzle",
        fandhe_ai_tensor_core::dispatch::GemmShape::new(0, 0, 0),
        base.block_m,
        base.block_n,
        base.block_k,
        base.stages,
        base.dtype,
        CompiledDims::DYNAMIC_ALL,
    )
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
    /// （`Self::wmma_tf32` フィールドのドキュメンテーションコメント
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
        // naive f32/f16・tiled f32/f16・tiled_bias_act_f32（イシュー #599。
        // epilogue 融合カーネルも naive/tiled と同様 `#include` を使わず全
        // compute capability で成立するため、WMMA(TF32) 系のような Option
        // 化・失敗の退避は行わず、`new` の早期 return（`?`）に合流させる）の
        // 5 カーネルは [`kernel_specs`]（本番・イシュー #926 診断テストの
        // 双方が参照する単一の真実源。Review #945 P2 指摘）の先頭 5 要素を
        // [`load_function_cached`]（module_cache／NVRTC ディスクキャッシュ
        // 経由。イシュー #1024）でロードする（固定長配列の直接インデックス
        // 参照のため、本番経路で禁止される `unwrap`/`expect` を要素取得に
        // 必要としない。`.claude/rules/coding-rust.md`）。
        let specs = kernel_specs();

        let spec = specs[0];
        let naive_f32 =
            load_function_cached(device, spec.descriptor()?, spec.source, spec.func_name)?;

        let spec = specs[1];
        let naive_f16 =
            load_function_cached(device, spec.descriptor()?, spec.source, spec.func_name)?;

        let spec = specs[2];
        let tiled_f32 =
            load_function_cached(device, spec.descriptor()?, spec.source, spec.func_name)?;

        let spec = specs[3];
        let tiled_f16 =
            load_function_cached(device, spec.descriptor()?, spec.source, spec.func_name)?;

        let spec = specs[4];
        let tiled_bias_act_f32 =
            load_function_cached(device, spec.descriptor()?, spec.source, spec.func_name)?;

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
        let (wmma_tf32, wmma_tf32_error) = match compile_wmma_tf32(device) {
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

        let (wmma_tf32_opt, wmma_tf32_opt_error) = match compile_wmma_tf32_opt(device) {
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

        let (wmma_tf32_staged, wmma_tf32_staged_error) = match compile_wmma_tf32_staged(device) {
            Ok(func) => (Some(func), None),
            Err(e) => (None, Some(e.to_string())),
        };

        // イシュー #856: `wmma_tf32_staged`（base）のコンパイルに成功し、
        // かつ SM 数が実測できた場合のみ swizzle 変種を追加コンパイルする
        // （`gemm_mma.rs::CudaMmaGemm::new` の `mma_f16_swizzle` 分岐と同型の
        // fail-soft 判断。swizzle はあくまで L2 再利用の性能最適化であり
        // base の可用性とは独立であるべきため、ソース生成・NVRTC
        // コンパイルいずれの失敗も `new` 全体の `Err` へ波及させない）。
        let (
            wmma_tf32_staged_swizzle,
            wmma_tf32_staged_swizzle_group_width,
            wmma_tf32_staged_swizzle_error,
        ) = match (&wmma_tf32_staged, device.multiprocessor_count()) {
            (Some(_), Some(num_sms)) => {
                let group_width = crate::swizzle::select_swizzle_group_width(
                    num_sms,
                    kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_M,
                    kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_N,
                );
                match kernels_wmma_opt::wmma_tf32_f32_staged_source_with_swizzle(group_width)
                    .and_then(|src| {
                        let descriptor = wmma_tf32_staged_swizzle_descriptor()?;
                        load_function_cached(device, descriptor, &src, "gemm_wmma_tf32_staged")
                    }) {
                    Ok(func) => (Some(func), Some(group_width), None),
                    Err(e) => (None, None, Some(e.to_string())),
                }
            }
            _ => (None, None, None),
        };

        // イシュー #1033: `kernels_tiled_pipeline::tiled_pipeline_f32_source()`
        // の cp.async パイプライン契約（`TP_STAGES >= 2` 等）は同モジュール側の
        // コンパイル時 const assert が保証するため、ここでは追加の const
        // アサーションを重複させない（`kernels_tiled_pipeline.rs` 冒頭の
        // 契約検査群を単一の真実源とする）。
        //
        // イシュー #1344: 64×64 版は常にコンパイルする（第 2 スロット方式
        // への変更後は「置換」ではなく「追加」意味論のため、const の値に
        // 依らず本番既定経路の土台として不変）。
        let (tiled_pipeline, tiled_pipeline_error) = match compile_tiled_pipeline(device) {
            Ok(func) => (Some(func), None),
            Err(e) => (None, Some(e.to_string())),
        };

        // イシュー #1344: `TILED_PIPELINE_128X64_PRODUCTION_ENABLED`
        // （既定 `false`）が有効化された場合のみ、128×64 版を
        // `tiled_pipeline` とは独立の第 2 スロットへ追加でコンパイルする
        // （既定 `false` では一切コンパイルされない＝JIT コスト・
        // `select_tiled_f32_kernel` の分岐先とも完全不変。GB10 実機での
        // 純カーネル時間比較・形状条件付き結線の可否判断は
        // `docs/perf/cuda-gemm-tiled-pipeline.md`「#1344」節参照）。
        let (tiled_pipeline_128x64, tiled_pipeline_128x64_error) =
            if TILED_PIPELINE_128X64_PRODUCTION_ENABLED {
                match compile_tiled_pipeline_128x64(device) {
                    Ok(func) => (Some(func), None),
                    Err(e) => (None, Some(e.to_string())),
                }
            } else {
                (None, None)
            };

        // イシュー #1214: VJP 専用 NT/TN 転置入口の smem 転置カーネル。
        // 上記 `wmma_tf32`／`tiled_pipeline` 系と同じ fail-soft 方針
        // （`transpose_smem_f32_error` フィールドのドキュメンテーション
        // コメント参照）。
        let (transpose_smem_f32, transpose_smem_f32_error) =
            match compile_transpose_smem_f32(device) {
                Ok(func) => (Some(func), None),
                Err(e) => (None, Some(e.to_string())),
            };

        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            stream: device.stream().clone(),
            allocator,
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
            wmma_tf32_staged_swizzle,
            wmma_tf32_staged_swizzle_group_width,
            wmma_tf32_staged_swizzle_error,
            tiled_pipeline,
            tiled_pipeline_error,
            tiled_pipeline_128x64,
            tiled_pipeline_128x64_error,
            transpose_smem_f32,
            transpose_smem_f32_error,
        })
    }

    /// `device` 上で、L2 再利用のためのタイル→SM 割り当てスウィズル
    /// （イシュー #741。f16 経路の #499・`gemm_mma.rs::
    /// CudaMmaGemm::new_with_swizzle` と同一設計）を明示指定の
    /// `group_width` で TF32 opt-staged カーネルへ**強制適用**した変種を
    /// NVRTC コンパイルし保持するハンドルを構築する（**診断用・明示幅
    /// 指定の入口**。[`new`](Self::new)（本番既定コンストラクタ。イシュー
    /// #856 で GB10 実機 A/B（§7.4.1 サイズ条件付き新基準で採用）を根拠に
    /// サイズ条件付き適用を結線済み）とは異なり、本コンストラクタは
    /// 形状・SM 数の判定を経ずに指定幅を全サイズへ強制適用するため、A/B
    /// 計測・bit 一致検証で候補幅 `{8, 16}` を個別に指定・強制適用したい
    /// 場合の用途に限定される。`gemm_mma.rs::CudaMmaGemm::new_with_swizzle`
    /// と同型の位置づけ）。
    ///
    /// 手順: [`new`](Self::new) で通常構築した後、`wmma_tf32_staged`
    /// スロットのみを `kernels_wmma_opt::
    /// wmma_tf32_f32_staged_source_with_swizzle(group_width)` のコンパイル
    /// 結果へ差し替える。naive/tiled・WMMA 基本版・opt 版の各スロットは
    /// [`new`](Self::new) の構築結果をそのまま保持し、swizzle 変種の
    /// コンパイル失敗の影響を受けない。**`new` が独立に構築したサイズ
    /// 条件付き動的変種（`wmma_tf32_staged_swizzle`）は破棄し
    /// `wmma_tf32_staged_swizzle_group_width` のみへ明示指定の
    /// `group_width` を保持する**（`should_launch_wmma_tf32_staged_swizzle`
    /// の `None` 分岐が形状に関わらず常に `true` を返すようにするため。
    /// 破棄しないと、サイズ条件付き適用条件を満たす形状で `new` 由来の
    /// 動的幅（本関数の明示指定と異なりうる）へ黙ってすり替わり、A/B
    /// 計測が意図した固定幅を計測できなくなる）。
    ///
    /// **変種のコンパイル失敗は base へ黙ってフォールバックせず `Err` を
    /// 返す**（`run_wmma_tf32` の 3 段選択のように `None` へ退避すると、
    /// 実験経路のつもりで呼んだ A/B ベンチが気づかず base（無 swizzle）を
    /// 計測してしまう A/A 誤認を招くため。fail-closed な安全側判断）。
    ///
    /// 返す [`CudaGemm`] は [`new`](Self::new) が返すものと同一の型・API
    /// （`run_wmma_tf32` 含む）を持ち、grid/block 構成・形状検証・整列
    /// 判定（[`wmma_tf32_staged_alignment_ok`]）・K 上限検証
    /// （[`validate_wmma_tf32_staged_k_bound`]）はブロックタイル定数
    /// （`WMMA_TF32_STAGED_BLOCK_M`/`_N`）を変更しないため共有できる
    /// （swizzle はブロックがどの `(m_block, n_block)` を担当するかの
    /// 割り当てのみを変え、各出力要素のアキュムレート順序・ブロックあたり
    /// の計算内容は変えない）。
    ///
    /// 任意ソースを受ける公開 API（`new_with_source` 型）は意図的に作らず、
    /// `group_width: u32` のみを受けてカプセル化する
    /// （`kernels_wmma_opt.rs` 側で固定文字列アンカーの `replacen` と数値
    /// `format!` 埋め込みのみを行う契約を維持し、外部入力を直接カーネル
    /// ソースへ流し込む経路を作らない。`.claude/rules/security.md` A03
    /// インジェクション対策）。`group_width < 2` は
    /// `kernels_wmma_opt::wmma_tf32_f32_staged_source_with_swizzle` が
    /// `CudaError::InvalidShape` で拒否する。
    ///
    /// **`internal-diagnostics` feature（既定 off）でのみコンパイルされる**
    /// （`gemm_mma.rs::CudaMmaGemm::new_with_swizzle` と同じ feature ゲート
    /// 方針。`Cargo.toml` の `[features]` 参照）。
    /// `examples/gemm_wmma_tf32_swizzle_bench.rs`（`Cargo.toml` の
    /// `required-features` で同 feature を要求）・bit 一致テスト
    /// （本ファイル下部 `wmma_tf32_staged_swizzle_variant_matches_base_
    /// bit_exact_output`）専用の入口（`docs/perf/cuda-gemm-swizzle-ab.md`
    /// §7 参照）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn new_with_tf32_staged_swizzle(
        device: &CudaDevice,
        group_width: u32,
    ) -> Result<Self, CudaError> {
        let mut gemm = Self::new(device)?;

        let arch = device.arch();
        let src = kernels_wmma_opt::wmma_tf32_f32_staged_source_with_swizzle(group_width)?;
        let ptx = compile_ptx(&src, arch)?;
        let wmma_tf32_staged = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_wmma_tf32_staged")?;

        gemm.wmma_tf32_staged = Some(wmma_tf32_staged);
        gemm.wmma_tf32_staged_error = None;
        // イシュー #856: `wmma_tf32_staged` 自体を明示指定の group_width で
        // 強制変種へ差し替えたため、`new`（本番既定コンストラクタ）由来の
        // サイズ条件付き動的変種（`wmma_tf32_staged_swizzle`）は破棄し
        // `wmma_tf32_staged_swizzle_group_width` のみへ group_width を保持
        // する（`should_launch_wmma_tf32_staged_swizzle` の `None` 分岐が
        // これを見て形状に関わらず常に `true` を返し、`func`〈= 上記で
        // 差し替えた強制変種〉が選ばれる。`gemm_mma.rs::CudaMmaGemm::
        // new_with_swizzle` の `mma_f16_swizzle: None`／
        // `swizzle_group_width: Some(_)` と同型の設計）。これを怠ると
        // `new` が偶然この device で選んだ動的幅（例: 8）が、本関数が
        // 明示指定した幅（例: 16）を形状によっては黙って上書きしてしまい、
        // A/B 計測が意図した固定幅を計測できなくなる（本関数ドキュメント
        // コメント「A/A 誤認」節と同じ懸念の裏返し）。
        gemm.wmma_tf32_staged_swizzle = None;
        gemm.wmma_tf32_staged_swizzle_group_width = Some(group_width);
        gemm.wmma_tf32_staged_swizzle_error = None;
        Ok(gemm)
    }

    /// `device` 上で、L2 再利用のためのタイル→SM 割り当てスウィズル
    /// （イシュー #1034。TF32 opt-staged 経路の #741・f16 mma.sync 経路の
    /// #499 と同一設計）を明示指定の `group_width` で **本番既定 f32 経路
    /// （`TILED_F32`）** へ**強制適用**した変種を NVRTC コンパイルし保持
    /// するハンドルを構築する（**診断用・明示幅指定の入口**。
    /// [`new_with_tf32_staged_swizzle`](Self::new_with_tf32_staged_swizzle)
    /// と同型の位置づけ）。
    ///
    /// **本番既定コンストラクタ（[`new`](Self::new)）への結線は本イシュー
    /// のスコープ外**（実装計画 §2「本番結線を本ランで行わない判断」・
    /// §7「スコープ外」(a)）。`new`・`run_tiled_f32`／`launch_tiled_f32`
    /// は本コンストラクタが返す変種を経由しない限り一切影響を受けない。
    /// ncu による L2 ヒット率実測・N=4096 改善値・サイズ条件付き適用への
    /// ユーザー承認を DGX Spark 実機セッションで得てから、`swizzle.rs::
    /// should_apply_swizzle` 相当のサイズ条件付き適用を `new` へ結線する
    /// 判断へ進める（f16 経路の先例〈#740 差し戻し→#775/#782 実機ゲート
    /// 後に `new` へ昇格〉と同じ安全側の順序）。
    ///
    /// 手順: [`new`](Self::new) で通常構築した後、`tiled_f32` スロット
    /// のみを `kernels::tiled_f32_source_with_swizzle(group_width)` の
    /// コンパイル結果へ差し替える。naive・tiled_f16・tiled_bias_act_f32・
    /// WMMA 系の各スロットは [`new`](Self::new) の構築結果をそのまま
    /// 保持し、swizzle 変種のコンパイル失敗の影響を受けない。
    ///
    /// `tiled_f32` は（`wmma_tf32_staged` と異なり）`Option` ではなく
    /// 必須 `CudaFunction` フィールドのため、`wmma_tf32_staged_swizzle`
    /// のような「動的変種を破棄する」後始末は不要（`new` はそもそも
    /// `tiled_f32` へのサイズ条件付き動的変種を構築しない。上記「スコープ
    /// 外」節参照）。
    ///
    /// **変種のコンパイル失敗は base へ黙ってフォールバックせず `Err` を
    /// 返す**（`new_with_tf32_staged_swizzle` と同じ理由。A/B ベンチが
    /// 気づかず base〈無 swizzle〉を計測してしまう A/A 誤認を防ぐ
    /// fail-closed な安全側判断）。
    ///
    /// 返す [`CudaGemm`] は [`new`](Self::new) が返すものと同一の型・API
    /// （`run_tiled_f32`／`launch_tiled_f32` 含む）を持つ。grid/block
    /// 構成（`TILED_BLOCK_DIM`）・形状検証（`validate_gemm_dims`／
    /// `validate_tiled_k_bound`）はブロックタイル定数（`kernels::TILE`）を
    /// 変更しないため共有できる（swizzle はブロックがどの `(m_block,
    /// n_block)` を担当するかの割り当てのみを変え、各出力要素の積和順序・
    /// 手動境界チェックの要否は変えない。`kernels::
    /// tiled_f32_source_with_swizzle` ドキュメンテーションコメント参照）。
    ///
    /// 任意ソースを受ける公開 API（`new_with_source` 型）は意図的に作らず
    /// `group_width: u32` のみを受ける（`new_with_tf32_staged_swizzle` と
    /// 同じ理由。`.claude/rules/security.md` A03 インジェクション対策）。
    /// `group_width < 2` は `kernels::tiled_f32_source_with_swizzle` が
    /// `CudaError::InvalidShape` で拒否する。
    ///
    /// **`internal-diagnostics` feature（既定 off）でのみコンパイルされる**
    /// （`new_with_tf32_staged_swizzle` と同じ feature ゲート方針）。
    /// `examples/gemm_tiled_f32_swizzle_bench.rs`（`Cargo.toml` の
    /// `required-features` で同 feature を要求）・bit 一致テスト（本ファイル
    /// 下部 `tiled_f32_swizzle_variant_matches_base_bit_exact_output`）専用の
    /// 入口。
    #[cfg(feature = "internal-diagnostics")]
    pub fn new_with_tiled_f32_swizzle(
        device: &CudaDevice,
        group_width: u32,
    ) -> Result<Self, CudaError> {
        let mut gemm = Self::new(device)?;

        let arch = device.arch();
        let src = kernels::tiled_f32_source_with_swizzle(group_width)?;
        let ptx = compile_ptx(&src, arch)?;
        let tiled_f32 = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_tiled_f32")?;

        gemm.tiled_f32 = tiled_f32;
        // イシュー #1137: `run_tiled_f32` 系 3 入口は #1137 以降 cp.async
        // パイプライン版へ形状条件付きで分岐しうるため（`select_tiled_f32_kernel`）、
        // swizzle 変種を差し替えたままだと、整列形状の呼び出しが swizzle
        // 適用前の classic パイプライン相当のパイプラインカーネルへ流れて
        // しまい、A/B ベンチ・bit 一致テストが「swizzle が効いていない」
        // ことに気づかず base 扱いしてしまう A/A 誤認が起きる
        // （`new_with_tf32_staged_pads` と同じ fail-closed 判断）。
        // パイプラインスロットを強制的に無効化し、swizzle 変種は必ず
        // classic 側（差し替え済み `tiled_f32`）で起動されるようにする。
        gemm.tiled_pipeline = None;
        gemm.tiled_pipeline_error = Some(
            "disabled by tiled_f32 swizzle variant (issue #1137 A/A confound guard)".to_string(),
        );
        // イシュー #1344: 第 2 スロット（128×64）も同じ A/A 混同ガードで
        // 無効化する（`TILED_PIPELINE_128X64_PRODUCTION_ENABLED` が
        // 有効化された将来の状態でも、本診断コンストラクタは classic 側
        // 固定を保証する契約を保つ）。
        gemm.tiled_pipeline_128x64 = None;
        gemm.tiled_pipeline_128x64_error = Some(
            "disabled by tiled_f32 swizzle variant (issue #1137 A/A confound guard)".to_string(),
        );
        Ok(gemm)
    }

    /// `device` 上で、`wmma_tf32_staged` の SMEM パディング幅（`a_pad`/
    /// `b_pad`）のみを差し替えた変種を NVRTC コンパイルし保持するハンドルを
    /// 構築する（イシュー #743・`kernels_wmma_opt::
    /// wmma_tf32_f32_staged_source_with_pads` 参照。**opt-in・未計測の
    /// 実験実装**）。[`new_with_tf32_staged_swizzle`](Self::new_with_tf32_staged_swizzle)
    /// と同じ設計（`new` で通常構築した後 `wmma_tf32_staged` スロットのみ
    /// 差し替え、変種のコンパイル失敗は base へフォールバックせず `Err` を
    /// 返す。理由も同じ: A/B ベンチが気づかず base を計測してしまう
    /// A/A 誤認を防ぐ fail-closed 判断）。
    ///
    /// `a_pad`/`b_pad` の妥当性検査（4 要素倍数・下限・SMEM 予算内）は
    /// [`kernels_wmma_opt::wmma_tf32_f32_staged_source_with_pads`] が
    /// 経由する `validate_wmma_tf32_staged_config` に委譲する。
    ///
    /// **`internal-diagnostics` feature（既定 off）でのみコンパイルされる**
    /// （swizzle 版と同じ feature ゲート方針）。唯一の呼び出し元は
    /// `gemm.rs` 内の `#[ignore]` 付き bit 一致テスト（本ファイル下部
    /// 参照）であり、ncu 計測経路は `gemm_profile_target --b-pad`
    /// （`diagnostics::render_wmma_tf32_staged` 経由。本関数は不使用）。
    /// 実機 A/B 計測後に採用確定した段階で `WMMA_TF32_STAGED_B_PAD`
    /// （本番既定値）を書き換える判断へつなげる（`docs/perf/
    /// cuda-gemm-wmma-tf32-staged-bank-conflict.md` 参照）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn new_with_tf32_staged_pads(
        device: &CudaDevice,
        a_pad: u32,
        b_pad: u32,
    ) -> Result<Self, CudaError> {
        let mut gemm = Self::new(device)?;

        let arch = device.arch();
        let src = kernels_wmma_opt::wmma_tf32_f32_staged_source_with_pads(a_pad, b_pad)?;
        let ptx = compile_ptx(&src, arch)?;
        let wmma_tf32_staged = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_wmma_tf32_staged")?;

        gemm.wmma_tf32_staged = Some(wmma_tf32_staged);
        gemm.wmma_tf32_staged_error = None;
        // イシュー #856: `Self::new(device)?` が独立に構築した swizzle 変種
        // （パディング無変更のソースから導出。本関数が差し替えた pad 変種
        // とは異なるカーネル）を破棄する。破棄しない場合、サイズ条件付き
        // 適用条件を満たす形状（M=N=K>=4096 の正方形）では
        // `should_launch_wmma_tf32_staged_swizzle` が `wmma_tf32_staged_
        // swizzle`（pad 無変更の swizzle 変種）を選んでしまい、この関数が
        // 意図する「指定した a_pad/b_pad の効果を計測する」契約を裏切る
        // （`new_with_tf32_staged_swizzle` の同型是正コメント参照）。
        gemm.wmma_tf32_staged_swizzle = None;
        gemm.wmma_tf32_staged_swizzle_group_width = None;
        gemm.wmma_tf32_staged_swizzle_error = None;
        Ok(gemm)
    }

    /// `device` 上で、`wmma_tf32_staged` の swizzle 変種を**常に**保持しない
    /// ハンドルを構築する（`gemm_mma.rs::CudaMmaGemm::new_without_swizzle`
    /// の TF32 staged 版。イシュー #856）。
    ///
    /// [`new`](Self::new)（本番既定コンストラクタ。SM 数実測に成功した
    /// device では `wmma_tf32_staged_swizzle` を `Some` に持ちうる）とは
    /// 独立に、**常に**swizzle 無適用の base（`wmma_tf32_staged`）へアクセス
    /// するための明示的な入口。手順: [`new`](Self::new) で通常構築した後、
    /// `wmma_tf32_staged_swizzle`/`wmma_tf32_staged_swizzle_group_width`/
    /// `wmma_tf32_staged_swizzle_error` を明示的に `None` へ破棄する
    /// （`should_launch_wmma_tf32_staged_swizzle` の `None` 分岐は
    /// `wmma_tf32_staged_swizzle_group_width.is_some()` を見るため、これで
    /// 形状に関わらず常に base が選ばれる）。
    ///
    /// **導入理由（A/A 誤認の回避）**: `examples/gemm_wmma_tf32_swizzle_bench.rs`
    /// の base 計測腕は元々 `CudaGemm::new`（本番既定コンストラクタ）を
    /// 使っていたが、イシュー #856 の本番結線後は `new` 自体が SM 数実測
    /// 成功時に swizzle 変種を追加コンパイルし、`launch_wmma_tf32` が
    /// M=N=K=4096 級正方形では自動的にその変種を選ぶようになった。base 腕を
    /// `new` のままにすると、4096（採否判定の根拠となった唯一のサイズ）で
    /// base 腕自身が swizzle 変種を計測してしまい、`swizzle_g8_over_base`
    /// が ~1.0 に潰れる A/A 誤認が生じる（`new_with_tf32_staged_swizzle`
    /// ドキュメンテーションコメント「A/A 誤認」節と同型の懸念が、結線後は
    /// base 腕側にも生じるようになったための追加対応）。本コンストラクタを
    /// base 腕に使うことで、`new` の内部状態（SM 数実測の成否）に関わらず
    /// 常に swizzle 無適用の base を計測できるようにする。
    ///
    /// **`internal-diagnostics` feature（既定 off）でのみコンパイルされる**
    /// （`gemm_mma.rs::CudaMmaGemm::new_without_swizzle` と同じ feature
    /// ゲート方針）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn new_without_tf32_staged_swizzle(device: &CudaDevice) -> Result<Self, CudaError> {
        let mut gemm = Self::new(device)?;
        gemm.wmma_tf32_staged_swizzle = None;
        gemm.wmma_tf32_staged_swizzle_group_width = None;
        gemm.wmma_tf32_staged_swizzle_error = None;
        Ok(gemm)
    }

    /// `device` 上で、`run_wmma_tf32` の 3 段選択（staged→opt→basic）を
    /// **opt 版に強制**したハンドルを構築する（イシュー #994）。
    ///
    /// 呼び出し元は `examples/wmma_tolerance_probe.rs` の
    /// `--tf32-kernel opt`（`internal-diagnostics` feature 限定）のみ。
    /// `docs/perf/cuda-tensor-core-tolerance-evaluation.md` §2.1 の TF32
    /// 実測は TASK-11.1c 時点の基本版カーネルを対象としており、
    /// TASK-11.1d（#63）で追加された opt 版（`Self::wmma_tf32_opt`）は
    /// 未計測だった。本コンストラクタは `run_wmma_tf32` の 3 段選択から
    /// staged 経路を除外することで、公開 API を経由しつつ常に opt 版が
    /// 選ばれる状態を作る（[`new_without_tf32_staged_swizzle`]・
    /// [`new_with_tf32_staged_swizzle`] と同じ「`new` で通常構築した後に
    /// スロットを差し替える」設計）。
    ///
    /// 手順: [`new`](Self::new) で通常構築した後、`wmma_tf32_staged`／
    /// `wmma_tf32_staged_swizzle`／`wmma_tf32_staged_swizzle_group_width`
    /// を `None` に、`wmma_tf32_staged_error`／
    /// `wmma_tf32_staged_swizzle_error` を診断専用の理由文字列に差し替える
    /// （`tf32_kernel_availability_header` が理由付きで `staged=no (…)` を
    /// 表示できるようにする）。`run_wmma_tf32` は staged が `None` のため
    /// 常に opt 分岐（`self.wmma_tf32_opt.as_ref()`）へ進む。
    ///
    /// **fail-closed（A/A 誤認防止）**: `wmma_tf32_opt.is_none()`
    /// （opt 版がこの device でコンパイル・ロードに失敗している）場合は
    /// `basic` へ黙ってフォールバックせず `CudaError::WmmaUnavailable` を
    /// 返す。黙って `Ok` を返すと「opt 版を計測したつもりが実際は basic を
    /// 計測していた」誤認を招く（`new_with_tf32_staged_swizzle`
    /// ドキュメンテーションコメント「A/A 誤認」節と同じ判断）。
    ///
    /// **`internal-diagnostics` feature（既定 off）でのみコンパイルされる**
    /// （REQ-11「明示切替 API を公開面に置かない」を維持するため。既定
    /// ビルドの公開 API 面は変更しない）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn new_tf32_opt_only(device: &CudaDevice) -> Result<Self, CudaError> {
        let mut gemm = Self::new(device)?;

        if gemm.wmma_tf32_opt.is_none() {
            return Err(CudaError::WmmaUnavailable {
                detail: format!(
                    "new_tf32_opt_only: opt kernel unavailable, cannot force opt path \
                     (reason: {})",
                    gemm.wmma_tf32_opt_error
                        .as_deref()
                        .unwrap_or("unknown reason")
                ),
            });
        }

        gemm.wmma_tf32_staged = None;
        gemm.wmma_tf32_staged_error =
            Some("disabled by CudaGemm::new_tf32_opt_only (diagnostic)".to_string());
        gemm.wmma_tf32_staged_swizzle = None;
        gemm.wmma_tf32_staged_swizzle_group_width = None;
        gemm.wmma_tf32_staged_swizzle_error =
            Some("disabled by CudaGemm::new_tf32_opt_only (diagnostic)".to_string());
        Ok(gemm)
    }

    /// `device` 上で、`run_wmma_tf32` の 3 段選択を**基本版（`wmma_tf32`）に
    /// 強制**したハンドルを構築する（イシュー #994。
    /// [`new_tf32_opt_only`](Self::new_tf32_opt_only) の basic 版）。
    ///
    /// 呼び出し元は `examples/wmma_tolerance_probe.rs` の
    /// `--tf32-kernel basic`（`internal-diagnostics` feature 限定）のみ。
    ///
    /// 手順: staged 系スロット（`wmma_tf32_staged`／
    /// `wmma_tf32_staged_swizzle`／`wmma_tf32_staged_swizzle_group_width`）
    /// を無効化したうえで、さらに `wmma_tf32_opt` を `None`・
    /// `wmma_tf32_opt_error` を診断専用の理由文字列に差し替える。
    /// `run_wmma_tf32` は staged・opt がいずれも `None` のため常に basic
    /// 分岐（`self.wmma_tf32.as_ref()`）へ進む。
    ///
    /// **fail-closed（A/A 誤認防止）**: `wmma_tf32.is_none()`（基本版
    /// カーネル自体がこの device で使用不能）の場合は `CudaError::
    /// WmmaUnavailable` を返す（[`new_tf32_opt_only`](Self::new_tf32_opt_only)
    /// と同じ判断。basic を計測するつもりで実は何も実行できない状態を
    /// 静かに通さない）。
    ///
    /// **`internal-diagnostics` feature（既定 off）でのみコンパイルされる**
    /// （[`new_tf32_opt_only`](Self::new_tf32_opt_only) と同じ理由）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn new_tf32_basic_only(device: &CudaDevice) -> Result<Self, CudaError> {
        let mut gemm = Self::new(device)?;

        gemm.wmma_tf32_staged = None;
        gemm.wmma_tf32_staged_error =
            Some("disabled by CudaGemm::new_tf32_basic_only (diagnostic)".to_string());
        gemm.wmma_tf32_staged_swizzle = None;
        gemm.wmma_tf32_staged_swizzle_group_width = None;
        gemm.wmma_tf32_staged_swizzle_error =
            Some("disabled by CudaGemm::new_tf32_basic_only (diagnostic)".to_string());

        if gemm.wmma_tf32.is_none() {
            return Err(CudaError::WmmaUnavailable {
                detail: format!(
                    "new_tf32_basic_only: basic kernel unavailable, cannot force basic path \
                     (reason: {})",
                    gemm.wmma_tf32_error.as_deref().unwrap_or("unknown reason")
                ),
            });
        }

        gemm.wmma_tf32_opt = None;
        gemm.wmma_tf32_opt_error =
            Some("disabled by CudaGemm::new_tf32_basic_only (diagnostic)".to_string());
        Ok(gemm)
    }

    /// naive f32 GEMM を実行する。C = A @ B（`m x k` @ `k x n`）。
    ///
    /// ホスト側形状検証（`validate_gemm_dims`）を先行させた後、
    /// 16x16 ブロック・`div_ceil` グリッドで `Self::run_f32_kernel` を呼ぶ
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
        self.run_f32_kernel(
            &self.naive_f32,
            a,
            b,
            m,
            n,
            k,
            launch_config(m, n, NAIVE_BLOCK_DIM),
        )
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

    /// tiled f32 経路 3 入口（[`Self::run_tiled_f32`]／[`Self::launch_tiled_f32`]／
    /// [`Self::launch_tiled_f32_resident`]）が共通で使うカーネル選択
    /// （イシュー #1137）。[`tiled_f32_kernel_kind`]（純粋関数）の判定を
    /// `self.tiled_pipeline` の実体（`Option<TiledPipelineFunction>`）と
    /// 突き合わせ、`&CudaFunction` と対応する `LaunchConfig` の組を返す。
    ///
    /// `self.tiled_pipeline` は `Self::new` で `self.stream` と同じ
    /// `CudaDevice` に対してコンパイルされる（[`compile_tiled_pipeline`]
    /// 参照）ため、[`Self::launch_tiled_pipeline_f32`]（外部から任意の
    /// `TiledPipelineFunction` を受け取る公開 API）と異なり、本メソッドは
    /// context 一致検証を行わない（常に self 由来で自己無矛盾）。
    ///
    /// `tiled_f32_kernel_kind` が `Pipeline` を返す場合は
    /// `pipeline_available` 引数（`self.tiled_pipeline.is_some()`）が
    /// 真であることが判定条件に含まれるため、通常経路では
    /// `self.tiled_pipeline` は必ず `Some` になる。ただし本番経路で
    /// `unwrap`/`expect` によるパニックを避ける規約
    /// （`.claude/rules/coding-rust.md`「エラーは型付きエラーとし、
    /// 本番経路で `unwrap()` / `expect()` を使わない」）に従い、
    /// `None` の場合も `if let` で fail-closed に classic 版へ
    /// フォールバックする（codex-review P1 指摘・PR #1164。理論上
    /// 到達しない防御的分岐であり、`tiled_f32_kernel_kind` 自体の契約は
    /// 変えない）。
    ///
    /// `a_offset`（要素単位。[`tiled_f32_kernel_kind`] 参照）は
    /// [`Self::launch_tiled_f32_resident`] のみ非 0 を渡しうる。
    fn select_tiled_f32_kernel(
        &self,
        a_offset: usize,
        m: u32,
        n: u32,
        k: u32,
    ) -> (&CudaFunction, LaunchConfig) {
        let kind = tiled_f32_kernel_kind(self.tiled_pipeline.is_some(), a_offset, n, k);
        if let (TiledF32Kernel::Pipeline, Some(func64)) = (kind, self.tiled_pipeline.as_ref()) {
            // イシュー #1344: `tiled_pipeline` が既に Pipeline 側と判定
            // された形状に対して、第 2 スロット（128×64）が利用可能かつ
            // 閾値条件を満たす場合のみそちらへ分岐する。第 2 スロットが
            // `None` になるのは `TILED_PIPELINE_128X64_PRODUCTION_ENABLED`
            // が `false`（既定でない特殊ビルド）の場合、または
            // `compile_tiled_pipeline_128x64` が実行時に失敗した場合のみ
            // （`TILED_PIPELINE_128X64_PRODUCTION_ENABLED = true` の現行
            // 既定では `Self::new` が `tiled_pipeline_128x64` を
            // `tiled_pipeline` とは独立にコンパイルするため、
            // `new_with_tiled_pipeline_128x64` 診断経由〈内部で
            // `Self::new` を呼んだうえで `tiled_pipeline` フィールドのみ
            // 128×64 へ差し替える〉でも `tiled_pipeline_128x64` は
            // 通常どおり `Some`（＝第 1 スロットと同じ 128×64 ハンドル）
            // になりうる）。`None` の場合は常に `func64`（＝実際に
            // `self.tiled_pipeline` に入っているハンドル。既定は 64×64、
            // 診断経由では 128×64）をそのまま使う fail-closed
            // フォールバックで、`func.tile()` は選択されたハンドル自身の
            // タグを見るため launch config は常に整合し、既存の T7〜T11
            // （`new_with_tiled_pipeline_128x64` 経由）の契約も変えない。
            let tile_kind = tiled_pipeline_tile_kind(self.tiled_pipeline_128x64.is_some(), n, k);
            let func = if tile_kind == TiledPipelineTile::Bm128Bn64 {
                self.tiled_pipeline_128x64.as_ref().unwrap_or(func64)
            } else {
                func64
            };
            TILED_PIPELINE_LAUNCH_COUNT.with(|c| c.set(c.get() + 1));
            if func.tile() == TiledPipelineTile::Bm128Bn64 {
                TILED_PIPELINE_128X64_LAUNCH_COUNT.with(|c| c.set(c.get() + 1));
            }
            (func.as_cuda_function(), func.launch_config(m, n))
        } else {
            (&self.tiled_f32, tiled_f32_launch_config(m, n))
        }
    }

    /// `select_tiled_f32_kernel` の判定結果のみを、実際の起動を
    /// 伴わずに照会する（テスト・ベンチ・診断が事前に分岐先を確認できる
    /// ようにするための公開 API。`wmma_tf32_opt_available` と同じ理由）。
    /// `run_tiled_f32`／`launch_tiled_f32`（常にオフセット 0 の全体
    /// バッファ）の分岐先照会用のため `a_offset` 引数は取らない
    /// （`internal-diagnostics` feature 限定。`TiledF32Kernel` 冒頭
    /// ドキュメンテーションコメント「公開範囲」参照。codex-review P1
    /// 指摘・PR #1164）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn tiled_f32_kernel_for(&self, n: u32, k: u32) -> TiledF32Kernel {
        tiled_f32_kernel_kind(self.tiled_pipeline.is_some(), 0, n, k)
    }

    /// [`Self::select_tiled_f32_kernel`] が実際に選ぶタイル構成
    /// （[`TiledPipelineTile`]）を、起動を伴わずに照会する（イシュー
    /// #1344。`tiled_f32_kernel_for` の一段詳細版——`Pipeline` と判定
    /// された場合に 64×64／128×64 のどちらへ分岐するかまで返す）。
    /// `TiledF32Kernel::Classic` と判定される形状では `None` を返す
    /// （`a_offset` は常に 0 の入口〈`run_tiled_f32`／`launch_tiled_f32`〉
    /// 向けの照会のため引数に取らない。`tiled_f32_kernel_for` と同じ
    /// 理由。`tiled_pipeline_tile_kind` が M 軸の閾値を持たないため本
    /// メソッドも `m` を取らない）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn tiled_pipeline_tile_for(&self, n: u32, k: u32) -> Option<TiledPipelineTile> {
        match self.tiled_f32_kernel_for(n, k) {
            TiledF32Kernel::Classic => None,
            TiledF32Kernel::Pipeline => Some(tiled_pipeline_tile_kind(
                self.tiled_pipeline_128x64.is_some(),
                n,
                k,
            )),
        }
    }

    /// tiled f32 GEMM を実行する。C = A @ B（`m x k` @ `k x n`）。
    ///
    /// ホスト側形状検証は naive 版と同じ `validate_gemm_dims` に加え、
    /// tiled カーネル固有のタイルインデックス算術を保護する
    /// `validate_tiled_k_bound` を経由する（モジュールコメント
    /// 「PoC からの変更点」3 参照）。イシュー #1032 のレジスタブロッキング
    /// 刷新版カーネル（`kernels::TILED_F32`）を既定とし、cp.async 16
    /// バイト整列形状（`n % 4 == 0 && k % 4 == 0`）かつ `new` 時のコン
    /// パイルに成功している場合は cp.async 3 stage パイプライン版
    /// （`kernels_tiled_pipeline::gemm_tiled_pipeline_f32`）へ形状条件付き
    /// で分岐する（`select_tiled_f32_kernel`。GB10 実測に基づく
    /// 本番結線判断はイシュー #1137・`docs/perf/cuda-gemm-tiled-pipeline.md`
    /// 参照）。非整列形状・コンパイル失敗環境では常に classic 版へ
    /// fail-closed にフォールバックするため、呼び出し元から見た挙動・
    /// 検証契約（`validate_gemm_dims`・`validate_tiled_k_bound`）は変わらない。
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
        let (func, cfg) = self.select_tiled_f32_kernel(0, m, n, k);
        self.run_f32_kernel(func, a, b, m, n, k, cfg)
    }

    /// [`Self::run_tiled_f32`] と同じ選択（`select_tiled_f32_kernel`）
    /// を、`internal-diagnostics` feature 限定で常に classic 版
    /// （`kernels::TILED_F32`）へ強制した版。診断・A/B ベンチ
    /// （`examples/gemm_tiled_pipeline_bench.rs`）・bit 一致テスト
    /// （`tests/cpu_cuda_tiled_pipeline_parity.rs`）の base 側入口として使う
    /// （イシュー #1137）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn run_tiled_f32_classic(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        self.run_f32_kernel(
            &self.tiled_f32,
            a,
            b,
            m,
            n,
            k,
            tiled_f32_launch_config(m, n),
        )
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

    /// イシュー #1033: `kernels_tiled_pipeline::tiled_pipeline_f32_source()`
    /// を実行する。[`Self::run_tiled_f32`] の**選択可能な変種**（本番既定
    /// 経路は置き換えない。`kernels_tiled_pipeline.rs` 冒頭コメント
    /// 「位置づけ・非結線」参照）。
    ///
    /// ホスト側形状検証は `validate_gemm_dims` に加え、
    /// `validate_tiled_pipeline_k_bound`・cp.async 16 バイト整列検証
    /// （`tiled_pipeline_alignment_ok`。満たさない形状は
    /// `CudaError::InvalidShape` を返す。フォールバック経路を持たない
    /// 単独の選択可能変種のため fail-closed に拒否する契約）を経由する。
    /// カーネル自体が使用不能（`new` 時のコンパイル失敗。cp.async は
    /// sm_80 以降限定）な場合は `CudaError::TiledPipelineUnavailable` を
    /// 返す。
    pub fn run_tiled_pipeline_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        validate_tiled_pipeline_k_bound(k)?;
        if !tiled_pipeline_alignment_ok(n, k) {
            return Err(CudaError::InvalidShape {
                detail: format!(
                    "tiled pipeline kernel requires n % 4 == 0 && k % 4 == 0 \
                     (cp.async 16-byte transfer granularity): n={n}, k={k}"
                ),
            });
        }
        // codex-review P2 指摘（PR #1071）: 形状検証（`validate_gemm_dims`・
        // `validate_tiled_pipeline_k_bound`・整列検証）を先に完了させ、
        // `m == 0 || n == 0`（他の `run_*_f32` 系と同じ正当な no-op 形状。
        // 本関数コメント「ホスト側形状検証」参照）を `self.tiled_pipeline`
        // ハンドル参照より前に判定する。従来は逆順だったため、cp.async
        // 非対応環境（sm_80 未満・NVRTC コンパイル失敗等で `tiled_pipeline`
        // が `None`）では正当なゼロ次元入力まで
        // `CudaError::TiledPipelineUnavailable` として拒否されていた。
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        let func =
            self.tiled_pipeline
                .as_ref()
                .ok_or_else(|| CudaError::TiledPipelineUnavailable {
                    detail: self.tiled_pipeline_error.clone().unwrap_or_else(|| {
                        "tiled pipeline kernel unavailable for an unknown reason".to_string()
                    }),
                })?;
        self.run_f32_kernel(
            func.as_cuda_function(),
            a,
            b,
            m,
            n,
            k,
            func.launch_config(m, n),
        )
    }

    /// [`Self::run_tiled_pipeline_f32`] が使用可能かを返す（`new` 時の
    /// cp.async カーネルコンパイルに成功しているか）。`wmma_tf32_opt_available`
    /// と同じ理由（形状に依らない静的な可用性照会 API。テスト・bench
    /// example が実行前に判定できるようにするため）で公開する。
    pub fn tiled_pipeline_available(&self) -> bool {
        self.tiled_pipeline.is_some()
    }

    /// [`Self::tiled_pipeline_available`] が `false` の場合の失敗理由
    /// （`wmma_tf32_opt_unavailable_reason` と同じ理由で公開する）。
    pub fn tiled_pipeline_unavailable_reason(&self) -> Option<&str> {
        self.tiled_pipeline_error.as_deref()
    }

    /// `device` 上で任意のステージ数（2〜4）の tiled pipeline カーネルを
    /// オンデマンドでコンパイルする（イシュー #1033・
    /// `examples/gemm_tiled_pipeline_bench.rs` の 3 vs 4 stage 比較専用。
    /// `kernels_tiled_pipeline.rs` 冒頭コメント「stages=4 版はベンチ用途に
    /// 限りオンデマンドでコンパイルする」参照）。本番オブジェクト
    /// （[`new`](Self::new) が保持する既定ステージ数固定の
    /// `Self::tiled_pipeline`）の初期化コストには影響しない独立した
    /// コンパイル経路であり、`&self` を取らない（`CudaGemm` の状態に
    /// 触れず `device` のみから完結するため）。
    ///
    /// **公開面ゲート（codex-review P1 指摘・PR #1071）**: 本関数はベンチ
    /// 専用（`examples/gemm_tiled_pipeline_bench.rs`）であり本番ディスパッチ
    /// 経路（`run_tiled_pipeline_f32`・既定 `run_tiled_f32`）からは呼ばれ
    /// ない。`SpecializedMmaKernelHandle::compile`（`gemm_auto.rs`）と同じ
    /// `internal-diagnostics` feature（既定 off）でゲートし、通常ビルドの
    /// 安定した公開 API 面から除外する（`lib.rs` の `TiledPipelineFunction`
    /// re-export コメント参照）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn compile_tiled_pipeline_variant(
        device: &CudaDevice,
        stages: u32,
    ) -> Result<TiledPipelineFunction, CudaError> {
        let source = kernels_tiled_pipeline::tiled_pipeline_f32_source_with_stages(stages)?;
        let descriptor = CudaKernelDescriptor::new_with_compiled_dims(
            "tiled_pipeline_f32_variant",
            fandhe_ai_tensor_core::dispatch::GemmShape::new(0, 0, 0),
            kernels_tiled_pipeline::TP_BM,
            kernels_tiled_pipeline::TP_BN,
            kernels_tiled_pipeline::TP_BK,
            stages,
            fandhe_ai_tensor_core::dispatch::DType::F32,
            CompiledDims::DYNAMIC_ALL,
        )?;
        let func = load_function_cached(device, descriptor, &source, "gemm_tiled_pipeline_f32")?;
        let context_ptr = Arc::as_ptr(device.context()) as usize;
        Ok(TiledPipelineFunction(
            func,
            context_ptr,
            TiledPipelineTile::Bm64Bn64,
        ))
    }

    /// 128×64×16 pipeline カーネル（イシュー #1343）の任意ステージ数
    /// （[`kernels_tiled_pipeline_128x64::TP128_MIN_STAGES`]..=
    /// [`kernels_tiled_pipeline_128x64::TP128_MAX_STAGES`]）変種を
    /// オンデマンドでコンパイルする（[`Self::compile_tiled_pipeline_variant`]
    /// の 128×64 版。`examples/gemm_tiled_pipeline_bench.rs` の段数比較・
    /// A/B 計測用途。`&self` を取らない理由・公開面ゲートの理由は同メソッド
    /// と同一）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn compile_tiled_pipeline_128x64_variant(
        device: &CudaDevice,
        stages: u32,
    ) -> Result<TiledPipelineFunction, CudaError> {
        let source =
            kernels_tiled_pipeline_128x64::tiled_pipeline_128x64_f32_source_with_stages(stages)?;
        let descriptor = CudaKernelDescriptor::new_with_compiled_dims(
            "tiled_pipeline_f32_128x64_variant",
            fandhe_ai_tensor_core::dispatch::GemmShape::new(0, 0, 0),
            kernels_tiled_pipeline_128x64::TP128_BM,
            kernels_tiled_pipeline_128x64::TP128_BN,
            kernels_tiled_pipeline_128x64::TP128_BK,
            stages,
            fandhe_ai_tensor_core::dispatch::DType::F32,
            CompiledDims::DYNAMIC_ALL,
        )?;
        let func = load_function_cached(
            device,
            descriptor,
            &source,
            "gemm_tiled_pipeline_128x64_f32",
        )?;
        let context_ptr = Arc::as_ptr(device.context()) as usize;
        Ok(TiledPipelineFunction(
            func,
            context_ptr,
            TiledPipelineTile::Bm128Bn64,
        ))
    }

    /// [`new`](Self::new) が保持する既定 3 stage の `Self::tiled_pipeline`
    /// が実際にどちらのタイル構成（[`TiledPipelineTile`]）でコンパイル
    /// されているかを返す（既定は常に `Some(Bm64Bn64)`。
    /// [`TILED_PIPELINE_128X64_PRODUCTION_ENABLED`] が有効化されない限り
    /// `Bm128Bn64` にはならない。[`Self::new_with_tiled_pipeline_128x64`]
    /// 経由で構築したインスタンスでは `Some(Bm128Bn64)` になる。cp.async
    /// 非対応環境等でコンパイル自体に失敗している場合は `None`。診断・
    /// テスト専用の可観測点のため `internal-diagnostics` feature 限定）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn tiled_pipeline_tile(&self) -> Option<TiledPipelineTile> {
        self.tiled_pipeline
            .as_ref()
            .map(TiledPipelineFunction::tile)
    }

    /// [`Self::new`] と同じ手順で構築したうえで、`tiled_pipeline` スロット
    /// を 128×64 版（イシュー #1343。[`compile_tiled_pipeline_128x64`]）へ
    /// 差し替えた診断専用インスタンスを返す。
    ///
    /// `run_tiled_f32` 系 3 入口（[`Self::select_tiled_f32_kernel`]）は
    /// `self.tiled_pipeline` を直接参照するため、本コンストラクタで得た
    /// インスタンスに対しては cp.async 16 バイト整列形状（`n % 4 == 0 &&
    /// k % 4 == 0`）で自動的に 128×64 カーネルへ分岐する（#1344 の GB10
    /// 実機比較・`tests/cpu_cuda_tiled_pipeline_parity.rs` の bit 一致
    /// 自己検証が使う経路）。[`TILED_PIPELINE_128X64_PRODUCTION_ENABLED`]
    /// の値には依存しない（常に 128×64 をコンパイルする明示的な opt-in
    /// 経路であり、本番既定 [`Self::new`] の挙動は変えない）。
    ///
    /// 128×64 カーネルのコンパイルに失敗した場合は `CudaError`（NVRTC・
    /// module ロード起因のエラー種別。`compile_tiled_pipeline_128x64` が
    /// 返すものをそのまま伝播）を返す（fail-closed。`Self::new` 自体は
    /// 64×64 版のコンパイル失敗を早期 return に合流させない fail-soft
    /// 方針だが、本コンストラクタは診断用途で「呼び出し元が明示的に
    /// 128×64 を要求した」ことが前提のため、失敗を握りつぶさず即座に
    /// 伝える）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn new_with_tiled_pipeline_128x64(device: &CudaDevice) -> Result<Self, CudaError> {
        let mut gemm = Self::new(device)?;
        let func = compile_tiled_pipeline_128x64(device)?;
        gemm.tiled_pipeline = Some(func);
        gemm.tiled_pipeline_error = None;
        Ok(gemm)
    }

    /// デバイス常駐済みの A/B/C バッファに対して、任意のコンパイル済み
    /// tiled pipeline カーネル（既定 3 stage の `Self::tiled_pipeline`、
    /// または [`Self::compile_tiled_pipeline_variant`] が返す任意段数の
    /// 変種）を起動し、完了を待たずに投入する（`#1013` の非同期投入契約。
    /// [`Self::launch_tiled_f32`] と同じ「GPU 実行のみ」区間をベンチ計測
    /// できるよう公開する）。
    ///
    /// ホスト側形状検証は [`Self::run_tiled_pipeline_f32`] と同一
    /// （`validate_gemm_dims`・`validate_tiled_pipeline_k_bound`・
    /// cp.async 16 バイト整列検証）に加え、デバイスバッファ長
    /// （`a_dev`/`b_dev`/`c_dev`）が `m/n/k` と 1:1 対応することを起動前に
    /// 検証する（`launch_tiled_f32` と同じ「safe な公開 API である以上、
    /// 呼び出し元の契約違反から独立して GPU 側 OOB を防ぐ」判断）。
    ///
    /// **context 一致検証（codex-review P1 指摘・PR #1071）**: `func` が
    /// `self.stream`（＝この `CudaGemm` を構築した `CudaDevice`）と同じ
    /// `CudaContext` から生成されたことを [`TiledPipelineFunction`] の
    /// `context_ptr`（型ドキュメントコメント参照）で fail-closed に検証
    /// する。複数 GPU・複数 `CudaContext` 利用時に、別 context で
    /// `compile_tiled_pipeline_variant` して得たハンドルをこのインスタンス
    /// へ渡すと、context 固有の `CUfunction` と本インスタンスの
    /// `stream`／デバイスバッファを混在させた `unsafe` launch に到達し
    /// うる（invalid device context・実質的な UB／OOB リスク）ため、
    /// 起動前に一致しなければ `CudaError::TiledPipelineContextMismatch`
    /// を返し `unsafe` launch へ到達させない。
    ///
    /// **バッファ生成元 context の検証（codex-review P0 指摘・PR #1071）**:
    /// 上記の `func` 検証だけでは、公開引数 `a_dev`/`b_dev`/`c_dev`
    /// （いずれも safe な [`CudaSlice`]）が同じ context 由来かどうかを
    /// 保証できない。`CudaSlice` はどの `CudaDevice`／`CudaContext` で
    /// 確保したものでも safe Rust から本関数へ渡せてしまうため、
    /// 呼び出し元が別の `CudaDevice`（＝別 `CudaGemm`）で確保した
    /// `CudaSlice` を（長さだけ `m/n/k` と一致させて）この
    /// `CudaGemm`（別 context の `stream`）へ渡すと、context 固有の
    /// デバイスポインタと本インスタンスの `stream` を混在させた
    /// `unsafe` launch に到達し、invalid device context・実質的な
    /// UB／OOB リスクを招く。`CudaSlice::context()`
    /// （cudarc 0.19.8 `driver::safe::core::CudaSlice::context`）が
    /// 返す `Arc<CudaContext>` のポインタ同一性を `self.stream.context()`
    /// と `func` 側と同じ fail-closed 方式で検証し、3 バッファのいずれか
    /// が不一致なら `CudaError::TiledPipelineContextMismatch` を返して
    /// `unsafe` launch へ到達させない。
    ///
    /// **`m == 0 || n == 0`（codex-review P1 指摘・PR #1071）**:
    /// `validate_gemm_dims` は naive/tiled 系（`run_f32_kernel` 冒頭コメント
    /// 参照）と同じくこの形状を正当な no-op として受理するが、そのまま
    /// `tiled_pipeline_launch_config` を呼ぶと grid_dim の x（n 由来）また
    /// は y（m 由来）が 0 になり driver launch へ進んでしまう。
    /// `run_tiled_pipeline_f32` は早期 return で空の結果を返す一方、本関数
    /// （常駐 API）はこれまで検証後に無条件で launch config を構築して
    /// いたため、ゼロ次元形状で `CUDA_ERROR_INVALID_VALUE` 等になり得た。
    /// `validate_output_len` の直後（launch config 構築前）で no-op を
    /// `Ok(())` として返し、`c_dev`（論理長 0）に触れず・カーネルを起動
    /// せずに `run_tiled_pipeline_f32` と同じ契約へ揃える。
    ///
    /// **公開面ゲート（codex-review P1 指摘・PR #1071）**: 本関数は
    /// `compile_tiled_pipeline_variant` と同じくベンチ専用の常駐 API
    /// （`examples/gemm_tiled_pipeline_bench.rs`）であり本番ディスパッチ
    /// 経路からは呼ばれない。同じ `internal-diagnostics` feature（既定
    /// off）でゲートする（`lib.rs` の `TiledPipelineFunction` re-export
    /// コメント参照）。
    #[cfg(feature = "internal-diagnostics")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_tiled_pipeline_f32(
        &self,
        func: &TiledPipelineFunction,
        a_dev: &CudaSlice<f32>,
        b_dev: &CudaSlice<f32>,
        c_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        let self_context_ptr = Arc::as_ptr(self.stream.context()) as usize;
        if func.context_ptr() != self_context_ptr {
            return Err(CudaError::TiledPipelineContextMismatch {
                detail: "TiledPipelineFunction was compiled against a different CudaContext \
                         (different CudaDevice/GPU) than this CudaGemm instance's stream; \
                         refusing to launch across mismatched CUDA contexts"
                    .to_string(),
            });
        }
        // codex-review P0 指摘（PR #1071）: `func` の context 一致検証だけ
        // では、safe な `CudaSlice` 引数が別 `CudaDevice`／`CudaContext`
        // 由来である可能性を排除できない（長さの一致のみでは検出不能）。
        // `CudaSlice::context()` のポインタ同一性を `self_context_ptr` と
        // 個別に fail-closed 検証し、混在した `unsafe` launch を防ぐ
        // （関数ドキュメントコメント「バッファ生成元 context の検証」参照）。
        for (name, buf_context_ptr) in [
            ("a_dev", Arc::as_ptr(a_dev.context()) as usize),
            ("b_dev", Arc::as_ptr(b_dev.context()) as usize),
            ("c_dev", Arc::as_ptr(c_dev.context()) as usize),
        ] {
            if buf_context_ptr != self_context_ptr {
                return Err(CudaError::TiledPipelineContextMismatch {
                    detail: format!(
                        "{name} was allocated on a different CudaContext (different \
                         CudaDevice/GPU) than this CudaGemm instance's stream; refusing \
                         to launch across mismatched CUDA contexts"
                    ),
                });
            }
        }
        validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        validate_tiled_pipeline_k_bound(k)?;
        if !tiled_pipeline_alignment_ok(n, k) {
            return Err(CudaError::InvalidShape {
                detail: format!(
                    "tiled pipeline kernel requires n % 4 == 0 && k % 4 == 0 \
                     (cp.async 16-byte transfer granularity): n={n}, k={k}"
                ),
            });
        }
        validate_output_len(c_dev.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(());
        }

        let cfg = func.launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: `run_f32_kernel`／`launch_tiled_f32` と同一の根拠。
        // カーネル引数（a_dev/b_dev/c_dev・m_i/n_i/k_i）は上記で検証済みの
        // m/n/k と 1:1 対応し、カーネル内の手動境界チェック（cp.async
        // src_size ゼロ充填・エピローグ guarded store。
        // `kernels_tiled_pipeline.rs` 冒頭コメント「REQ-8」参照）と合わせて
        // OOB 読み書きが起きない根拠とする。
        unsafe {
            self.stream
                .launch_builder(func.as_cuda_function())
                .arg(a_dev)
                .arg(b_dev)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点
        // （`download_f32`／明示 `synchronize`）へ委ねる。
        Ok(())
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
    /// （`validate_gemm_dims`・`validate_tiled_k_bound`）に加え、
    /// カーネル本体へ触れる前に `bias` の長さが `n` と一致することを検証
    /// する（CPU 参照実装 `gemm_blis_bias_act_parallel` の
    /// `GemmError::BiasLenMismatch` と同じ「カーネル本体アクセス前に検証」
    /// の順序契約。REQ-8・OWASP A03）。
    ///
    /// **`m == 0 || n == 0`**: `Self::run_f32_kernel` と同じ理由（no-op
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
    /// `gemm::BIAS_ACT_FUSED_LAUNCH_COUNT`
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
            return Err(CudaError::InvalidElementwiseShape {
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

        BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.set(c.get() + 1));

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
        // イシュー #1020: `run_f32_kernel` と同じ理由（epilogue も
        // `row < m && col < n` ガード内で全 `m*n` 要素を必ず埋める。
        // `kernels::TILED_BIAS_ACT_F32` 参照）でプール経由 `alloc_uninit_f32`
        // を使う。
        let mut c_dev = self
            .allocator
            .alloc_uninit_f32((m as usize) * (n as usize))?;

        let cfg = tiled_f32_launch_config(m, n);
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
                .arg(&mut c_dev.as_view_mut())
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&has_bias)
                .arg(&act_i)
                .launch(cfg)?;
        }
        // 同期点は readback ヘルパーへ集約（#1013）。プール割当ハンドル
        // （`PooledCudaHandle`。イシュー #1020）は `DevicePtr` を直接実装しない
        // ため、論理長ビュー（`as_view()`）を渡す。
        let c_host = crate::memory::readback(&self.stream, &c_dev.as_view())?;
        Ok(c_host)
    }

    /// WMMA（Tensor Core）を用いた TF32 GEMM を実行する。C = A @ B（`m x k` @
    /// `k x n`）、入出力は f32（内部で TF32 に丸めて Tensor Core へ投入する。
    /// `kernels::WMMA_TF32_F32` 参照）。
    ///
    /// ホスト側形状検証は naive/tiled 版と同じ `validate_gemm_dims` に加え、
    /// WMMA カーネル固有のタイルインデックス算術を保護する
    /// `validate_wmma_tf32_k_bound` を経由する（`validate_tiled_k_bound`
    /// と同じ考え方だが `kernels::WMMA_TF32_K_TILE`（8）基準で独立して検証する）。
    /// TASK-11.1c（#62）・REQ-11。
    ///
    /// `CudaGemm::new` 時点で WMMA(TF32) カーネルのコンパイル・ロードが
    /// 失敗していた場合（`Self::wmma_tf32` フィールド参照）は
    /// `CudaError::WmmaUnavailable` を返す。この場合でも naive/tiled 版の
    /// `run_naive_*`／`run_tiled_*` は道連れにならず引き続き使用できる
    /// （レビュー指摘 #62）。
    ///
    /// **TASK-11.1d（#63）フォールバック方針**: 共有メモリ・タイル最適化版
    /// （`Self::wmma_tf32_opt`）が `new` 時点でコンパイル・ロードに成功
    /// していれば、そちらを優先的に使用する（#63 の受け入れ条件「tiled
    /// 実装を上回る実測」を満たす経路）。opt 版が `None`（コンパイル失敗
    /// または未対応環境）の場合は基本版（`Self::wmma_tf32`）へ自動
    /// フォールバックし、公開シグネチャ・呼び出し側の挙動は変えない
    /// （REQ-11 は明示切替 API を提供しない方針。`kernels_wmma_opt.rs`
    /// 冒頭ドキュメントコメント「公開 API への影響」参照）。
    /// 共有メモリ・タイル最適化版 WMMA(TF32) カーネル（`Self::wmma_tf32_opt`）
    /// が `new` 時点でコンパイル・ロードに成功しているかを返す（TASK-11.1d・
    /// #63。PR #256 レビュー指摘: chatgpt-codex-connector「Require the
    /// optimized kernel in the optimized benchmark」対応）。
    ///
    /// `run_wmma_tf32` は opt カーネルが `None` の場合に基本版
    /// （`Self::wmma_tf32`）へ自動フォールバックする（公開 API の挙動は
    /// 変えない設計判断。上記ドキュメンテーションコメント参照）ため、
    /// `run_wmma_tf32` の戻り値の成否だけでは opt カーネルが実際に実行
    /// されたかを判定できない。opt カーネル固有の性能・タイル境界を検証
    /// する受け入れテスト（`tests/gemm_wmma_tf32_opt.rs`）はこの関数で
    /// 事前に可用性を確認し、フォールバックが起きていないことを保証した
    /// うえで計測・検証する。
    /// 基本版 WMMA(TF32) カーネル（`wmma_tf32` フィールド。非公開）が
    /// `new` 時点でコンパイル・ロードに成功しているかを返す（イシュー #1024）。
    /// [`Self::wmma_tf32_opt_available`]・[`Self::wmma_tf32_staged_available`]
    /// と同型の可用性 getter（`run_wmma_tf32` の内部フォールバック先を
    /// 強制実行する API ではなく、単に `new` 時点の成否を読み取るだけの
    /// もので REQ-11「明示切替 API を提供しない」方針に抵触しない）。
    /// module_cache／NVRTC ディスクキャッシュ結線の診断テスト
    /// （`module_cache_wiring_tests.rs`）が、基本版が NVRTC に拒否される
    /// 環境（naive/tiled はビルドできるが TF32 WMMA は未対応の CUDA
    /// デバイス）で `kernel_specs` の必須件数を過大に見積もらないよう
    /// 実際にコンパイルへ成功した件数を算出するために使う。
    ///
    /// `module_cache_wiring_tests.rs` は同一クレート内の `#[cfg(test)]`
    /// モジュール（`crate::CudaGemm` へ直接アクセス。統合テスト
    /// クレートではない）からのみ呼ばれる内部診断用アクセサのため
    /// `pub(crate)` に限定する（PR #1060 codex-review P1 指摘対応:
    /// 安定した公開 API 面〈`wmma_tf32_opt_available`・
    /// `wmma_tf32_staged_available` 等〉と異なり `tests/`・`examples/`
    /// から参照されない内部表現であり、`pub fn` として公開 API へ
    /// 露出させない）。
    ///
    /// 非 test ビルドでは呼び出し元が存在しないため `#[cfg(test)]` を
    /// 付与し `dead_code` lint（`-D warnings`）を回避する。
    #[cfg(test)]
    pub(crate) fn wmma_tf32_available(&self) -> bool {
        self.wmma_tf32.is_some()
    }

    /// [`Self::wmma_tf32_available`] が `false` の場合の失敗理由
    /// （[`Self::wmma_tf32_opt_unavailable_reason`] と同じ理由）。
    /// [`Self::wmma_tf32_available`] と同じ理由で `pub(crate)` かつ
    /// `#[cfg(test)]` に限定する（PR #1060 codex-review P1 指摘対応）。
    #[cfg(test)]
    pub(crate) fn wmma_tf32_unavailable_reason(&self) -> Option<&str> {
        self.wmma_tf32_error.as_deref()
    }

    pub fn wmma_tf32_opt_available(&self) -> bool {
        self.wmma_tf32_opt.is_some()
    }

    /// [`Self::wmma_tf32_opt_available`] が `false` の場合の失敗理由
    /// （`Self::wmma_tf32_opt_error` の公開読み取り口）。opt カーネルが
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

    /// 指定した `(n, k)` の形状で `run_wmma_tf32` が実際に **staged 経路**
    /// を選ぶかを判定する（[`Self::wmma_tf32_routed_path_available`] は
    /// staged／opt いずれかへのルーティングを束ねて判定するのに対し、本
    /// 関数は staged 経路そのものが選ばれたかに限定する）。
    ///
    /// `mma_tf32` vs `wmma_tf32` の A/B 比較（`cuda_floor_bench.rs`
    /// `measure_wmma_tf32` 呼び出し元。イシュー #802・
    /// `docs/perf/cuda-gemm-mma-tf32-ab.md` §5 採否条件）は比較対象を
    /// 明示的に `wmma_tf32`（staged）としているため、staged 以外
    /// （opt／basic）へフォールバックした実測値を staged との A/B 結果と
    /// して扱うと採否判断を誤らせる（codex-review 指摘。PR #826）。呼び
    /// 出し側はこの関数で staged 選択を確認したうえで比率を出力する。
    pub fn wmma_tf32_routed_path_is_staged(&self, n: u32, k: u32) -> bool {
        self.wmma_tf32_staged_available() && wmma_tf32_staged_alignment_ok(n, k)
    }

    /// イシュー #856。`Self::wmma_tf32_staged_swizzle` に適用された
    /// グルーピング幅（`gemm_mma.rs::CudaMmaGemm::swizzle_group_width` と
    /// 同型のアクセサ）。`examples/cuda_floor_bench.rs` の起動時診断が
    /// 現在選択されている値を可観測にするために呼ぶ。
    pub fn wmma_tf32_staged_swizzle_group_width(&self) -> Option<u32> {
        self.wmma_tf32_staged_swizzle_group_width
    }

    /// [`new`](Self::new) が `wmma_tf32_staged`（base）のコンパイルに成功し
    /// SM 数実測にも成功した（＝ swizzle 変種を試みた）にもかかわらず、
    /// ソース生成・NVRTC コンパイルに失敗し swizzle 変種を保持できなかった
    /// 場合の理由文字列（`gemm_mma.rs::CudaMmaGemm::swizzle_unavailable_reason`
    /// と同型）。swizzle 変種を保持している場合・base 自体が使用不能・
    /// SM 数が取得できず試みなかった場合は `None`。
    pub fn wmma_tf32_staged_swizzle_unavailable_reason(&self) -> Option<&str> {
        self.wmma_tf32_staged_swizzle_error.as_deref()
    }

    /// `run_wmma_tf32`/`launch_wmma_tf32` が形状 `(m, n, k)` に対して実際に
    /// staged swizzle 変種を起動するかを返す（`gemm_mma.rs::CudaMmaGemm::
    /// swizzle_applies` と同型。イシュー #856）。
    pub fn wmma_tf32_staged_swizzle_applies(&self, m: u32, n: u32, k: u32) -> bool {
        self.should_launch_wmma_tf32_staged_swizzle(m, n, k)
    }

    /// raw 次元 `(m, n)` と `k` から、staged 経路が選ばれた場合に起動すべき
    /// カーネルが swizzle 変種か base かを判定する共通ロジック
    /// （[`Self::wmma_tf32_staged_swizzle_applies`] と `run_wmma_tf32`／
    /// `launch_wmma_tf32` の staged 分岐の両方が参照する単一の真実源。
    /// `gemm_mma.rs::CudaMmaGemm::should_launch_swizzle_kernel` と同型）。
    ///
    /// `wmma_tf32_staged_swizzle` が `Some`（[`new`](Self::new) が SM 数
    /// 実測に成功しサイズ条件付き変種を保持している）場合は `(m, n)` から
    /// ブロックタイル数（`wmma_tf32_staged_launch_config` の grid 次元と
    /// 同じ `WMMA_TF32_STAGED_BLOCK_M`/`_N` 単位の `div_ceil`）を導出し
    /// [`crate::swizzle::should_apply_swizzle`] で判定する。`None`
    /// （SM 数未取得・変種コンパイル失敗）の場合は常に `false`（base
    /// フォールバック。強制適用の診断入口は
    /// [`new_with_tf32_staged_swizzle`](Self::new_with_tf32_staged_swizzle)
    /// が別途担う——同関数は `wmma_tf32_staged` スロット自体を変種へ
    /// 差し替えるため、この判定関数を経由せず常に変種を起動する）。
    fn should_launch_wmma_tf32_staged_swizzle(&self, m: u32, n: u32, k: u32) -> bool {
        match &self.wmma_tf32_staged_swizzle {
            Some(_) => {
                let num_m_blocks = m.div_ceil(kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_M);
                let num_n_blocks = n.div_ceil(kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_N);
                crate::swizzle::should_apply_swizzle(m, n, num_m_blocks, num_n_blocks, k)
            }
            // `wmma_tf32_staged_swizzle` が `None` の場合の意味は 2 通り
            // ある: (a) `new`（本番既定コンストラクタ）が SM 数を取得
            // できず fail-soft に縮退した場合（`wmma_tf32_staged_swizzle_
            // group_width` も `None`）、(b)
            // [`new_with_tf32_staged_swizzle`](Self::new_with_tf32_staged_swizzle)
            // 経由（強制適用の診断入口。`wmma_tf32_staged` スロット自体を
            // 変種へ差し替え済みのため `wmma_tf32_staged_swizzle` は使わず
            // `wmma_tf32_staged_swizzle_group_width` のみを `Some` にする。
            // `gemm_mma.rs::CudaMmaGemm::new_with_swizzle` と同型）。
            // (a) は常に false（base フォールバック）、(b) は形状に関わらず
            // 常に true（`func` 自体が既に強制変種のため `unwrap_or(func)`
            // で自然に選択される）で区別する。
            None => self.wmma_tf32_staged_swizzle_group_width.is_some(),
        }
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
            // イシュー #856: staged 経路が選ばれた形状について、さらに
            // サイズ条件付き swizzle 変種（`wmma_tf32_staged_swizzle`）を
            // 起動すべきかを判定する（`gemm_mma.rs::CudaMmaGemm::launch_f16`
            // の `should_launch_swizzle_kernel` 分岐と同型）。判定 false
            // または変種未保持の場合は base（`func`）へフォールバックする。
            let kernel = if self.should_launch_wmma_tf32_staged_swizzle(m, n, k) {
                self.wmma_tf32_staged_swizzle.as_ref().unwrap_or(func)
            } else {
                func
            };
            return self.run_wmma_tf32_staged_kernel(kernel, a, b, m, n, k);
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
        // 同期点は readback ヘルパーへ集約（#1013）。
        let c_host = crate::memory::readback(&self.stream, &c_dev)?;
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
        // 同期点は readback ヘルパーへ集約（#1013）。
        let c_host = crate::memory::readback(&self.stream, &c_dev)?;
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
        // 同期点は readback ヘルパーへ集約（#1013）。
        let c_host = crate::memory::readback(&self.stream, &c_dev)?;
        Ok(c_host)
    }

    /// f32 カーネル共通の起動手続き（naive/tiled/tiled pipeline 共通から
    /// 呼ばれる）。
    ///
    /// 呼び出し元がホスト側形状検証（`validate_gemm_dims`／
    /// `validate_tiled_k_bound`）を終えている前提で、`clone_htod` で A・B
    /// を転送し、呼び出し元が構築済みの `cfg`（`LaunchConfig`）でカーネルを
    /// 起動、`synchronize` の後 `clone_dtoh` で C を回収する（PoC-v2-3 の
    /// `run_f32` と同じ構造）。
    ///
    /// **イシュー #1033 Review 指摘対応**: 以前は `block_dim` を受け取り
    /// 内部で `launch_config(m, n, block_dim)`（`block_dim` = 出力タイル
    /// サイズという 1 スレッド = 1 出力要素モデルの契約）を呼んでいたが、
    /// tiled pipeline カーネル（256 スレッド 1 次元ブロックで `TP_BM x
    /// TP_BN`＝64×64 タイルをレジスタブロッキング担当）はこの契約を満た
    /// さず、`run_tiled_pipeline_f32` が `block_dim = (TP_BLOCK_THREADS,
    /// 1, 1) = (256, 1, 1)` を渡すと `grid.x = n.div_ceil(256)` となって
    /// 正しい `n.div_ceil(TP_BN=64)` より過小になり、`n > 256` の形状で C
    /// の広範囲が未計算（未初期化メモリ）のまま返っていた。呼び出し元が
    /// 自身のカーネルの実際のタイル形状に応じた `LaunchConfig` を構築して
    /// 渡す方式へ変更し、`run_tiled_pipeline_f32` は
    /// `launch_tiled_pipeline_f32` と同じ `tiled_pipeline_launch_config`
    /// を使うようにした。naive/tiled は従来どおり `launch_config(m, n,
    /// NAIVE_BLOCK_DIM/TILED_BLOCK_DIM)` を呼び出し元で構築する。
    #[allow(clippy::too_many_arguments)]
    fn run_f32_kernel(
        &self,
        func: &CudaFunction,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
        cfg: LaunchConfig,
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
        // イシュー #1020: 出力バッファはサイズクラス別プール
        // （`crate::pool::CudaAllocator`）経由で確保する（都度
        // `alloc_zeros`／解放していた固定費の削減。#1008 実測が主因の
        // 1 つとして指摘）。naive/tiled いずれのカーネルも `if (row < m
        // && col < n)` の書き込みガード内で全 `m*n` 要素を必ず埋める
        // （`kernels.rs` 参照）ため `alloc_uninit_f32` を使う（前利用
        // データの残留は起動直後に全要素上書きされ露出しない。OWASP A02
        // ではなく `docs/backend-cuda-pool-allocator-decision.md` §「`
        // alloc_uninit` の適用」の確認済みケース）。
        let mut c_dev = self
            .allocator
            .alloc_uninit_f32((m as usize) * (n as usize))?;

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
        // チェックで弾かれる。`c_dev.as_view_mut()` は論理長 `m*n` の
        // ビュー（サイズクラス丸めによる余剰容量は含まない）。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev.as_view_mut())
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        // 同期点は readback ヘルパーへ集約（#1013）。プール割当ハンドル
        // （`PooledCudaHandle`。イシュー #1020）は `DevicePtr` を直接実装しない
        // ため、論理長ビュー（`as_view()`）を渡す。
        let c_host = crate::memory::readback(&self.stream, &c_dev.as_view())?;
        Ok(c_host)
    }

    /// VJP 専用 NT/TN 転置入口（イシュー #1214）の内部ヘルパー: `src`
    /// （論理形状 `rows`×`cols` 行優先のデバイス常駐バッファ。転置格納
    /// オペランドの元 storage をそのまま H2D 転送したもの。呼び出し元は
    /// `run_tiled_f32_nt`／`run_tiled_f32_tn`）を GPU 側 smem 転置カーネル
    /// （`transpose_smem_f32`）で転置し、結果（論理形状 `cols`×`rows`
    /// 行優先＝標準 storage）をプール確保した中間バッファへ書き込む。
    ///
    /// 転置先へは既存 NN GEMM カーネル（`select_tiled_f32_kernel` が選ぶ
    /// classic／cp.async パイプライン）をそのまま渡せるため、GEMM カーネル
    /// 側の変更は一切不要（`docs/matmul-vjp-zero-copy-decision.md` §4.3
    /// 「採用: GPU 側 smem 転置 → 既存 NN GEMM カーネル」）。
    ///
    /// # alloc_uninit の適用根拠
    ///
    /// 転置カーネル（`kernels_transpose::transpose_smem_source_f32`）の
    /// epilogue ストアガード `if (out_row < n && out_col < m)`（`n`/`m`
    /// はカーネル引数。呼び出し規約は本関数の `cols`/`rows` に対応）は、
    /// 出力グリッド全体（`rows*cols` 要素）を標準の行列転置としてちょうど
    /// 1 回ずつ書き切る（重複書き込み・欠落のいずれも生じない）。その
    /// ため確保直後の未初期化領域は起動完了までに全要素が上書きされ
    /// 呼び出し元へ露出しない（`docs/backend-cuda-pool-allocator-
    /// decision.md` §「`alloc_uninit` の適用」の確認済みケースに準じる）。
    ///
    /// 呼び出し元は `rows > 0 && cols > 0` を保証する契約（`run_tiled_f32_nt`／
    /// `run_tiled_f32_tn` は `m == 0 || n == 0`／`k == 0` を本関数呼び出し
    /// より前に早期 return する。`run_f32_kernel` と同型の契約）。
    fn transpose_to_pooled(
        &self,
        src: &CudaSlice<f32>,
        rows: u32,
        cols: u32,
    ) -> Result<PooledCudaHandle<f32>, CudaError> {
        validate_transpose_dims(src.len(), rows, cols)?;
        let out_len = (rows as usize) * (cols as usize);
        let mut dst = self.allocator.alloc_uninit_f32(out_len)?;
        validate_transpose_output_len(dst.as_view().len(), cols, rows)?;

        if rows == 0 || cols == 0 {
            // 0 次元グリッドの起動を CUDA driver が拒否するため
            // （`transpose.rs::launch_naive_f32` の `m == 0 || n == 0`
            // 早期 return と同じ理由）。上記契約上は到達しないはずだが、
            // safe な内部ヘルパーとして防御的に no-op で返す。
            return Ok(dst);
        }

        let func = self.transpose_smem_f32.as_ref().ok_or_else(|| {
            CudaError::TransposeEntryUnavailable {
                detail: self.transpose_smem_f32_error.clone().unwrap_or_else(|| {
                    "transpose_smem_f32 kernel unavailable for an unknown reason".to_string()
                }),
            }
        })?;

        let cfg = tiled_launch_config(rows, cols);
        let (m_i, n_i) = (rows as i32, cols as i32);

        // SAFETY: `src`（呼び出し元検証済み rows*cols 要素）・`dst`
        // （同じく rows*cols 要素。プール確保直後は上記 alloc_uninit 根拠
        // により起動完了までに全要素が上書きされる）はカーネル引数
        // （`const float* src, float* dst, int m, int n`）と型・個数が
        // 1:1 対応し、カーネル内の手動境界チェック（`transpose.rs::
        // launch_f32` と同一の根拠。REQ-8）と合わせて OOB を防ぐ。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(&mut dst.as_view_mut())
                .arg(&m_i)
                .arg(&n_i)
                .launch(cfg)?;
        }
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点
        // （`download_*`／`MemoryOps::download`／明示 `synchronize`）へ
        // 委ねる（設計文書 §3〜§4）。
        GEMM_TRANSPOSED_ENTRY_LAUNCH_COUNT.with(|c| c.set(c.get() + 1));
        Ok(dst)
    }

    /// VJP 専用 NT 転置入口（イシュー #1214）: `matmul_vjp` の d_weight
    /// （`Aᵀ @ g`。CUDA では `b`＝`g` が転置格納ではなく `a`＝`Aᵀ` を渡す
    /// 側が該当）や、より一般に「B が転置格納」なパターンを、`b`（正確
    /// には転置元 storage を表す `bt`。論理形状 `[n,k]` 行優先）を
    /// `Tensor::contiguous()` の再パックコピーを経由せず GPU 側 smem
    /// 転置カーネルで標準 `[k,n]` へ変換してから既存 NN GEMM カーネル
    /// （`select_tiled_f32_kernel`）へ渡す。呼び出し元は `ops.rs::
    /// CudaBackendOps::gemm_fp32_strict_impl`（`dense_transposed_view(b)`
    /// が `Some` を返す場合のみ）。
    ///
    /// ホスト側形状検証は [`Self::run_tiled_f32`] と同一
    /// （`validate_gemm_dims`・`validate_tiled_k_bound`）。`bt` の長さは
    /// `n*k`（`a.len() == m*k` と対で `validate_gemm_dims` が検証する
    /// `b.len() == k*n` と同じ要素数）。
    pub(crate) fn run_tiled_f32_nt(
        &self,
        a: &[f32],
        bt: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_gemm_dims(a.len(), bt.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if k == 0 {
            return Ok(vec![0.0f32; (m as usize) * (n as usize)]);
        }

        let a_dev = self.stream.clone_htod(a)?;
        let bt_dev = self.stream.clone_htod(bt)?;
        // `bt`（論理形状 [n,k] 行優先）を転置して標準 `b`（[k,n] 行優先）
        // を得る（`Self::transpose_to_pooled` ドキュメンテーションコメント
        // 参照）。
        let b_std = self.transpose_to_pooled(&bt_dev, n, k)?;

        let (func, cfg) = self.select_tiled_f32_kernel(0, m, n, k);
        let mut c_dev = self
            .allocator
            .alloc_uninit_f32((m as usize) * (n as usize))?;
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: `run_f32_kernel` と同一の根拠。`a_dev`（m*k 要素）・
        // `b_std`（k*n 要素。転置カーネルの出力として上記で構築済み）・
        // `c_dev`（m*n 要素）はホスト側検証（`validate_gemm_dims`・
        // `validate_tiled_k_bound`）済みの m/n/k と 1:1 対応し、GEMM
        // カーネル内の手動境界チェック（REQ-8）と合わせて OOB を防ぐ。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&a_dev)
                .arg(&b_std.as_view())
                .arg(&mut c_dev.as_view_mut())
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        let c_host = crate::memory::readback(&self.stream, &c_dev.as_view())?;
        Ok(c_host)
    }

    /// VJP 専用 TN 転置入口（イシュー #1214）: [`Self::run_tiled_f32_nt`]
    /// と対称の「A が転置格納」パターン。`at`（転置元 storage。論理形状
    /// `[k,m]` 行優先）を GPU 側 smem 転置カーネルで標準 `[m,k]` へ変換
    /// してから既存 NN GEMM カーネルへ渡す。呼び出し元は `ops.rs::
    /// CudaBackendOps::gemm_fp32_strict_impl`（`dense_transposed_view(a)`
    /// が `Some` を返す場合のみ。`matmul_vjp` の d_input `g @ Bᵀ` が
    /// 該当しうる）。
    pub(crate) fn run_tiled_f32_tn(
        &self,
        at: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_gemm_dims(at.len(), b.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if k == 0 {
            return Ok(vec![0.0f32; (m as usize) * (n as usize)]);
        }

        let at_dev = self.stream.clone_htod(at)?;
        let b_dev = self.stream.clone_htod(b)?;
        // `at`（論理形状 [k,m] 行優先）を転置して標準 `a`（[m,k] 行優先）
        // を得る。
        let a_std = self.transpose_to_pooled(&at_dev, k, m)?;

        let (func, cfg) = self.select_tiled_f32_kernel(0, m, n, k);
        let mut c_dev = self
            .allocator
            .alloc_uninit_f32((m as usize) * (n as usize))?;
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: run_tiled_f32_nt と同一の根拠。`a_std`（m*k 要素。転置
        // カーネルの出力として上記で構築済み）・`b_dev`（k*n 要素）・
        // `c_dev`（m*n 要素）はホスト側検証済みの m/n/k と 1:1 対応する。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&a_std.as_view())
                .arg(&b_dev)
                .arg(&mut c_dev.as_view_mut())
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        let c_host = crate::memory::readback(&self.stream, &c_dev.as_view())?;
        Ok(c_host)
    }

    /// `transpose_smem_f32` フィールドが `Some` かどうか（可用性照会
    /// API。`tiled_pipeline_available` と同型）。`ops.rs` が NT/TN 転置
    /// 入口へ分岐する前に呼ぶことは必須ではない（`run_tiled_f32_nt`／
    /// `run_tiled_f32_tn`／`launch_tiled_f32_resident_nt` 自身が
    /// `CudaError::TransposeEntryUnavailable` を返すため）が、診断・
    /// テストが事前に判定できるよう公開する。
    pub(crate) fn transpose_smem_f32_available(&self) -> bool {
        self.transpose_smem_f32.is_some()
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
        // 同期点は readback ヘルパーへ集約（#1013）。
        let c_host = crate::memory::readback(&self.stream, &c_dev)?;
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
    /// と同じ検証（`validate_gemm_dims`・`validate_tiled_k_bound`）に
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

        // イシュー #1137: `run_tiled_f32` と同じ形状条件付き選択
        // （`select_tiled_f32_kernel`）を GPU 実行のみの入口にも適用する。
        // `self.tiled_pipeline` は self と同じ context 由来のため
        // `launch_tiled_pipeline_f32`（外部ハンドル受け取り版）と異なり
        // context 一致検証は不要（メソッドコメント参照）。
        let (func, cfg) = self.select_tiled_f32_kernel(0, m, n, k);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: run_f32_kernel と同一の根拠。カーネル引数
        // （a_dev/b_dev/c_dev・m_i/n_i/k_i）は上記で検証済みの m/n/k
        // と 1:1 対応し、カーネル内の手動境界チェック（REQ-8。classic 版は
        // `kernels.rs`、pipeline 版は `kernels_tiled_pipeline.rs` の
        // cp.async src_size ゼロ充填・エピローグ guarded store）と合わせて
        // OOB 読み書きが起きない根拠とする。
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
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点
        // （`download_f32`／`MemoryOps::download`／明示 `synchronize`）へ
        // 委ねる（設計文書 §3〜§4）。
        Ok(())
    }

    /// [`Self::launch_tiled_f32`] のプール確保出力版（イシュー #1182
    /// `gemm_reuse_phase_diag_tests` 専用）。本番経路
    /// （`run_f32_kernel`）は出力バッファをサイズクラス別プール
    /// （[`PooledCudaHandle`]）経由で確保する一方、`launch_tiled_f32`
    /// は `CudaSlice<f32>`（`alloc_output_f32`＝`alloc_zeros`。memset
    /// を伴う）を受け取るため「本番と同じ確保方式での GPU 実行のみの
    /// 時間」を計測できない。本メソッドは `PooledCudaHandle` を受け取り
    /// `launch_tiled_f32` と同じ検証・SAFETY 根拠のまま起動する。
    /// `pub(crate)` で公開 API 面には出さない（診断専用。§7 スコープ外）。
    ///
    /// `kernel`（[`DiagTiledF32Kernel`]）: `Select` は本番
    /// [`Self::select_tiled_f32_kernel`] と同じ形状条件付き自動選択
    /// （pipeline／classic）、`Classic` は
    /// [`Self::launch_tiled_f32_classic`]（`internal-diagnostics`
    /// feature 限定）と同じ意味で常に classic（`tiled_f32`）へ固定
    /// する。`Classic` 固定が必要な理由: crates.io 公開版 `fandhe-ai
    /// =0.6.0` の `kernels.rs`（TILED_F32 ソース）は本 HEAD と差分が
    /// あり（`select_tiled_f32_kernel` の pipeline 分岐自体は 0.6.0 に
    /// 無い）、`bench-fandhe`（0.6.0 固定）の `gemm --mode reuse
    /// --phases`（イシュー #1182）が観測する `matmul` 区間と本診断を
    /// 突合する際、pipeline 分岐が有効な形状（N=1024/2048/4096 は整列
    /// のため pipeline 経由。`docs/perf/cuda-gemm-tiled-pipeline.md`
    /// 参照）では `Select` が 0.6.0 の実測と乖離しうる。`Classic` は
    /// その乖離を避けた近似比較用（`docs/perf/
    /// cuda-gemm-reuse-phase-breakdown.md` §5 参照）。
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_tiled_f32_pooled(
        &self,
        a_dev: &CudaSlice<f32>,
        b_dev: &CudaSlice<f32>,
        c_dev: &mut PooledCudaHandle<f32>,
        m: u32,
        n: u32,
        k: u32,
        kernel: DiagTiledF32Kernel,
    ) -> Result<(), CudaError> {
        validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        validate_output_len(c_dev.as_view().len(), m, n)?;

        let (func, cfg) = match kernel {
            DiagTiledF32Kernel::Select => self.select_tiled_f32_kernel(0, m, n, k),
            DiagTiledF32Kernel::Classic => (&self.tiled_f32, tiled_f32_launch_config(m, n)),
        };
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: launch_tiled_f32 と同一の根拠（カーネル引数の型・個数・
        // デバイスバッファ長は上記で検証済みの m/n/k と 1:1 対応し、
        // カーネル内の手動境界チェック〈REQ-8〉と合わせて OOB を防ぐ）。
        // `c_dev.as_view_mut()` は論理長 `m*n` のビュー（`run_f32_kernel`
        // と同じ）。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(a_dev)
                .arg(b_dev)
                .arg(&mut c_dev.as_view_mut())
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        // 非同期投入契約（#1013）。診断テスト側が明示 `stream.synchronize()`
        // でカーネル専有時間を分離する（`gemm_reuse_phase_diag_tests` の
        // `kernel_wait` 区間）。
        Ok(())
    }

    /// [`Self::launch_tiled_f32`] と同じ選択を、`internal-diagnostics`
    /// feature 限定で常に classic 版へ強制した版（[`Self::run_tiled_f32_classic`]
    /// と同じ理由。イシュー #1137）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn launch_tiled_f32_classic(
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

        let cfg = tiled_f32_launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: launch_tiled_f32 と同一の根拠。
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
        Ok(())
    }

    /// デバイス常駐済みの A/B/C バッファに対して bias 加算・activation
    /// 融合 tiled f32 カーネルを起動し、完了を待つ（イシュー #1022。
    /// [`Self::launch_tiled_f32`] と同じ「GPU 実行のみ」契約を
    /// [`Self::run_tiled_bias_act_f32`] の epilogue 融合版へ拡張したもの）。
    ///
    /// `ops::CudaBackendOps::gemm_resident_rhs` が `w`（デバイス常駐
    /// weight）をホストへ download せずに forward するために使う——
    /// `run_tiled_bias_act_f32` は `a`／`b` をホストスライスから
    /// `clone_htod` する契約のため、既にデバイス上にある `w` に対して
    /// 呼ぶと download→upload の往復（本イシューが排除する対象）が
    /// 発生してしまう。本関数は `a_dev`／`b_dev`（`w` はここに渡す）が
    /// 既にデバイス上にあることを前提にする。
    ///
    /// `bias_dev` が `None` の場合は `has_bias = 0` を渡し、カーネル側は
    /// このバッファを実際には参照しない（`run_tiled_bias_act_f32` と
    /// 同じ契約。呼び出し元はダミーバッファを用意する必要はなく、代わりに
    /// `a_dev` を再利用してよい〈`has_bias=0` のため参照されない〉）。
    ///
    /// ホスト側形状検証は [`Self::launch_tiled_f32`] と同一
    /// （`validate_gemm_dims`・`validate_tiled_k_bound`・
    /// `validate_output_len`）に加え、`bias_dev` の長さが `n` と一致する
    /// ことをカーネル本体アクセス前に検証する（REQ-8・OWASP A03。
    /// `run_tiled_bias_act_f32` と同じ順序契約）。
    #[allow(clippy::too_many_arguments)]
    pub fn launch_tiled_bias_act_f32(
        &self,
        a_dev: &CudaSlice<f32>,
        b_dev: &CudaSlice<f32>,
        bias_dev: Option<&CudaSlice<f32>>,
        act_relu: bool,
        c_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        validate_output_len(c_dev.len(), m, n)?;
        if let Some(bias_dev) = bias_dev
            && bias_dev.len() != n as usize
        {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!(
                    "bias length mismatch: expected {n} (n), actual {}",
                    bias_dev.len()
                ),
            });
        }

        let (has_bias, bias_arg): (i32, &CudaSlice<f32>) = match bias_dev {
            Some(bias_dev) => (1, bias_dev),
            // `has_bias == 0` のガードによりカーネル側はこのバッファを
            // 実際には参照しない（`run_tiled_bias_act_f32` ドキュメント
            // コメント「`bias` が `None` の場合はダミーの 1 要素バッファ」
            // と同じ設計。ここでは既存の `a_dev` を安全なダミーとして
            // 再利用し、新規デバイス確保を避ける）。
            None => (0, a_dev),
        };

        let cfg = tiled_f32_launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
        let act_i: i32 = if act_relu { 1 } else { 0 };

        BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.set(c.get() + 1));

        // SAFETY: run_f32_kernel と同一の根拠（`launch_tiled_f32` の
        // 該当コメント参照）。追加引数（bias_arg・has_bias・act_i）は
        // 上記で検証済みの `n`／`bias_dev` の有無と 1:1 対応し、カーネル
        // 内 epilogue は書き込みガード（`row < m && col < n`）の内側でのみ
        // `bias[col]` を参照するため OOB は発生しない（REQ-8）。
        unsafe {
            self.stream
                .launch_builder(&self.tiled_bias_act_f32)
                .arg(a_dev)
                .arg(b_dev)
                .arg(bias_arg)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&has_bias)
                .arg(&act_i)
                .launch(cfg)?;
        }
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点
        // （`download_f32`／`MemoryOps::download`／明示 `synchronize`）へ
        // 委ねる（設計文書 §3〜§4）。
        Ok(())
    }

    /// [`Self::launch_tiled_bias_act_f32`] の常駐ビュー版（イシュー
    /// #1023「R3: 要素オフセット付き常駐ビュー」設計。`docs/
    /// device-resident-update-design.md` 追補参照）。
    ///
    /// `#1023`（パラメータ横断の単一連結バッファ化）後、
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore` は
    /// `weight`／`bias` を含む全パラメータを 1 本の連結
    /// `DeviceBuffer<f32>` として保持するため、個々のパラメータは
    /// 連結バッファの部分範囲（`cudarc::driver::CudaView`）としてしか
    /// 表現できない。本メソッドは [`Self::launch_tiled_bias_act_f32`]
    /// と同一のカーネル・同一の境界検証ロジックを、`w_dev`／`bias_dev`
    /// のみ `CudaView`（オフセット付き部分ビュー）で受け取る形へ
    /// 変えたオーバーロードであり、`a_dev`／`c_dev`（毎ステップ新規
    /// upload/確保するため常に全体バッファ）は従来どおり
    /// `&CudaSlice<f32>` のまま。
    #[allow(clippy::too_many_arguments)]
    pub fn launch_tiled_bias_act_f32_resident(
        &self,
        a_dev: &CudaSlice<f32>,
        w_dev: &CudaView<'_, f32>,
        bias_dev: Option<&CudaView<'_, f32>>,
        act_relu: bool,
        c_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        validate_gemm_dims(a_dev.len(), w_dev.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        validate_output_len(c_dev.len(), m, n)?;
        if let Some(bias_dev) = bias_dev
            && bias_dev.len() != n as usize
        {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!(
                    "bias length mismatch: expected {n} (n), actual {}",
                    bias_dev.len()
                ),
            });
        }

        let cfg = tiled_f32_launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
        let act_i: i32 = if act_relu { 1 } else { 0 };

        BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.set(c.get() + 1));

        // SAFETY: `launch_tiled_bias_act_f32` と同一の根拠。`w_dev`／
        // `bias_dev` は `CudaView`（`PushKernelArg` は `CudaSlice` と
        // 同様 `&CudaView` にも実装されており、渡すポインタは元バッファ
        // 先頭 + オフセットバイトを指す。`DeviceBufferView::new`
        // （`tensor-core`）が構築時に offset+numel の範囲検査を済ませて
        // いるため、ここでの追加のオフセット検証は不要）。has_bias == 0
        // の場合のダミー引数には `a_dev` を再利用する（`CudaView` 型引数
        // を要求されるため `a_dev.slice(..)` でその場の一時ビューを渡す）。
        let dummy_view;
        let bias_arg: (i32, &CudaView<'_, f32>) = match bias_dev {
            Some(bias_dev) => (1, bias_dev),
            None => {
                dummy_view = a_dev.slice(..);
                (0, &dummy_view)
            }
        };
        let (has_bias, bias_arg) = bias_arg;

        unsafe {
            self.stream
                .launch_builder(&self.tiled_bias_act_f32)
                .arg(a_dev)
                .arg(w_dev)
                .arg(bias_arg)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&has_bias)
                .arg(&act_i)
                .launch(cfg)?;
        }
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点
        // （`download_f32`／`MemoryOps::download`／明示 `synchronize`）へ
        // 委ねる（設計文書 §3〜§4）。
        Ok(())
    }

    /// [`Self::launch_tiled_f32`] の常駐ビュー版（イシュー #1023「R3」）。
    /// `Op::LinearResident` の VJP（`fandhe_ai_autodiff::grad`）が
    /// `d_input^T = w @ g^T` を計算する際、`w`（連結バッファ内の部分
    /// ビュー）を `a_dev` 位置へ渡すために使う。[`Self::
    /// launch_tiled_bias_act_f32_resident`] と同じ理由で `a_dev` のみ
    /// `CudaView`、`b_dev`／`c_dev` は毎回新規 upload/確保される全体
    /// バッファのまま `&CudaSlice<f32>`。
    ///
    /// `a_offset`（`a_dev` の連結バッファ先頭からの要素オフセット。
    /// 呼び出し元が `CudaView` 構築に使った値をそのまま渡す）は
    /// `select_tiled_f32_kernel` の cp.async 経路選択にのみ使う
    /// （codex-review P0／Cursor Bugbot High 指摘・PR #1164:
    /// `CudaView` は cudarc 側でポインタフィールドが非公開のため、この
    /// メソッド内から `a_dev` の整列を検査できず、呼び出し元がすでに
    /// 把握しているオフセットを明示的に受け取る形にした。オフセットが
    /// 4 要素の倍数でない場合は cp.async 16 バイト整列制約
    /// （`tiled_pipeline_offset_aligned`）を満たさないため常に classic
    /// 版へ fail-closed にフォールバックする。`validate_gemm_dims`
    /// 等の既存境界検証には影響しない）。
    ///
    /// **可視性は `pub(crate)` に限定する**（codex-review P0 再指摘・
    /// PR #1164: `a_offset` を `a_dev` から独立した引数として受け取る
    /// 設計上、この関数を crate 外の safe 公開 API のままにすると、
    /// 呼び出し元が `a_dev` の実際の開始位置と無関係な `a_offset`
    /// （例えば非整列ビューに対して偽の `a_offset = 0`）を渡すことで
    /// 上記の cp.async 整列 fail-closed フォールバックを迂回しうる。
    /// 現状の唯一の呼び出し元は `ops.rs::CudaBackendOps::gemm_resident_lhs`
    /// で、`w.offset()`（`DeviceParamStore` が管理する `CudaView` 自身の
    /// オフセット）をそのまま渡す信頼できる構築経路のみを通るため、
    /// `pub(crate)` へ絞ることで「検証不能な外部入力では選択できない」
    /// を型システムで保証する（REQ-8 の手動境界チェック省略禁止）。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_tiled_f32_resident(
        &self,
        a_dev: &CudaView<'_, f32>,
        a_offset: usize,
        b_dev: &CudaSlice<f32>,
        c_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        validate_output_len(c_dev.len(), m, n)?;

        // イシュー #1137・#1164: `launch_tiled_f32` と同じ選択
        // （`CudaView` 引数も両カーネルとも `(a, b, c, m, n, k)` シグ
        // ネチャで同一のため launch_builder の差し替えのみで済む）に
        // `a_offset` の整列検査を追加する。
        let (func, cfg) = self.select_tiled_f32_kernel(a_offset, m, n, k);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: `launch_tiled_f32` と同一の根拠。
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
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点
        // （`download_f32`／`MemoryOps::download`／明示 `synchronize`）へ
        // 委ねる（設計文書 §3〜§4）。
        Ok(())
    }

    /// [`Self::launch_tiled_f32_resident`] の NT 版（イシュー #1214）:
    /// `b`（`bt_dev`）が転置格納（`ops.rs::CudaBackendOps::gemm_resident_lhs`
    /// が呼ぶ `dense_transposed_view(b)` が `Some` を返す場合）である
    /// `Op::LinearResident` の d_input（`w @ gᵀ`）を、`b` を
    /// `Tensor::contiguous()` の再パックコピー（`MemoryOps::upload`）を
    /// 経由せず、`bt_dev`（転置元 storage を直接 H2D 転送した生バッファ。
    /// 論理形状 `[n,k]` 行優先）を GPU 側 smem 転置カーネルで標準
    /// `[k,n]` へ変換してから起動する。`a_dev`（`w`）は
    /// [`Self::launch_tiled_f32_resident`] と同じくデバイス常駐 `CudaView`
    /// のまま渡す（この呼び出し元は `w` を毎回 upload し直さないため）。
    ///
    /// `a_offset`（`select_tiled_f32_kernel` の cp.async 整列判定用）の
    /// 意味・可視性制約（`pub(crate)` に限定する理由）は
    /// [`Self::launch_tiled_f32_resident`] のドキュメンテーションコメント
    /// と同一。
    #[allow(clippy::too_many_arguments)]
    /// 戻り値の `PooledCudaHandle<f32>`（`b_std`。転置カーネルの出力
    /// バッファ）は**呼び出し元が保持し続けなければならない**（advisor
    /// 指摘）: `PooledCudaHandle::Drop` はプールへの返却（`release_cached`
    /// 経由の実解放時は `cuMemFree` 相当）を伴い、本メソッド自体は
    /// 非同期投入のみで完了を待たないため、`b_std` をこの関数内で drop
    /// すると GEMM カーネルの実行完了前にバッファが再利用・解放されうる
    /// （単一ストリームの FIFO 順序保証により実際には安全な可能性が高い
    /// が、これは検証されていない暗黙の前提であり明示的に握る）。呼び出し
    /// 元は本ハンドルを次の同期点（`readback`／`MemoryOps::download`／
    /// 明示 `synchronize`）まで生存させること（`run_tiled_f32_nt`／
    /// `run_tiled_f32_tn` は関数内で readback まで完結するため同様の問題
    /// はない）。
    pub(crate) fn launch_tiled_f32_resident_nt(
        &self,
        a_dev: &CudaView<'_, f32>,
        a_offset: usize,
        bt_dev: &CudaSlice<f32>,
        c_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<PooledCudaHandle<f32>, CudaError> {
        validate_gemm_dims(a_dev.len(), bt_dev.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        validate_output_len(c_dev.len(), m, n)?;

        // `bt_dev`（論理形状 [n,k] 行優先）を転置して標準 `b`（[k,n]
        // 行優先）を得る（`Self::transpose_to_pooled` ドキュメンテーション
        // コメント参照）。
        let b_std = self.transpose_to_pooled(bt_dev, n, k)?;

        let (func, cfg) = self.select_tiled_f32_kernel(a_offset, m, n, k);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: `launch_tiled_f32_resident` と同一の根拠。`b_std` は
        // 転置カーネルの出力として上記で構築済みの k*n 要素バッファ。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(a_dev)
                .arg(&b_std.as_view())
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点
        // （`download_f32`／`MemoryOps::download`／明示 `synchronize`）へ
        // 委ねる（設計文書 §3〜§4）。`b_std` は上記 SAFETY コメントの
        // とおり呼び出し元が同期点まで保持する契約のため、ここでは
        // drop せずそのまま返す。
        Ok(b_std)
    }

    /// デバイス常駐済みの A/B/C バッファに対して WMMA(TF32) カーネルを
    /// 起動し、完了を待つ（[`Self::launch_tiled_f32`] と同じ「GPU 実行
    /// のみ」契約）。`run_wmma_tf32` と同一の 3 段選択ロジック（staged →
    /// opt → 基本）を用いる（イシュー #500 で staged 選択を追加）。
    /// 呼び出し元は事前に `run_wmma_tf32` を 1 回 probe 実行して可用性を
    /// 確認している前提だが（`cuda_floor_bench.rs::measure_wmma_tf32`
    /// 参照）、本関数自体は safe な公開 API のため、選択される経路
    /// （staged／opt／基本）ごとに必要な `k` 境界検証
    /// （`validate_wmma_tf32_staged_k_bound`／
    /// `validate_wmma_tf32_opt_k_bound`／`validate_wmma_tf32_k_bound`）
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
            // イシュー #856: run_wmma_tf32 と同じサイズ条件付き swizzle
            // 選択（`should_launch_wmma_tf32_staged_swizzle`）を適用する。
            let kernel = if self.should_launch_wmma_tf32_staged_swizzle(m, n, k) {
                self.wmma_tf32_staged_swizzle.as_ref().unwrap_or(func)
            } else {
                func
            };
            let cfg = wmma_tf32_staged_launch_config(m, n);
            unsafe {
                self.stream
                    .launch_builder(kernel)
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
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点
        // （`download_f32`／`MemoryOps::download`／明示 `synchronize`）へ
        // 委ねる（設計文書 §3〜§4）。
        Ok(())
    }

    /// C（f32）をデバイス→ホストへ転送する（[`Self::upload_f32`] と同じ
    /// 理由で公開する。tiled f32・WMMA(TF32) で共有）。
    ///
    /// 同期点（#1013）: 常駐 `launch_*` は非同期投入のみで完了を待たない
    /// ため、本関数が readback ヘルパー経由で完了を確定する。
    pub fn download_f32(&self, c_dev: &CudaSlice<f32>) -> Result<Vec<f32>, CudaError> {
        crate::memory::readback(&self.stream, c_dev)
    }

    /// ストリームの完了を明示的に待つ（イシュー #1013）。
    ///
    /// `launch_*`（常駐 API）が非同期投入のみに契約変更されたことで、
    /// D2H を伴わない計測区間（`examples/cuda_floor_bench.rs` 等の
    /// 「GPU 実行 + 完了待ち」を PyTorch 参照計測と同一境界で比較する
    /// マイクロベンチ）が完了保証を失った。本関数はその計測境界を
    /// 明示的に復元するための公開 API であり、本番ディスパッチ
    /// （`ops.rs`）からは呼ばれない。
    pub fn synchronize(&self) -> Result<(), CudaError> {
        Ok(self.stream.synchronize()?)
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

    /// イシュー #1343: [`tiled_pipeline_launch_config`] が
    /// [`TiledPipelineTile::Bm128Bn64`] に対して 128×64 のブロックタイル
    /// 寸法（`kernels_tiled_pipeline_128x64::TP128_BM`/`TP128_BN`）で
    /// grid_dim を導出し、[`TiledPipelineTile::Bm64Bn64`] は従来どおり
    /// 64×64 の値のままであることを検査する（GPU 不要の純粋関数）。
    /// **64×64 の launch config で 128×64 を起動すると grid_y が本来より
    /// 2 倍多くなり、末尾ブロックの手動境界チェックにより数値上は誤りが
    /// 隠蔽されつつも冗長な block 起動が生じる**ことへの回帰防御
    /// （実装計画 §4 T6）。
    #[test]
    fn tiled_pipeline_launch_config_matches_tile_dimensions() {
        let (m, n) = (300u32, 200u32);

        let cfg64 = tiled_pipeline_launch_config(TiledPipelineTile::Bm64Bn64, m, n);
        assert_eq!(
            cfg64.grid_dim,
            (
                n.div_ceil(kernels_tiled_pipeline::TP_BN),
                m.div_ceil(kernels_tiled_pipeline::TP_BM),
                1,
            ),
            "Bm64Bn64 grid_dim must use the 64x64 block tile dimensions"
        );
        assert_eq!(
            cfg64.block_dim,
            (kernels_tiled_pipeline::TP_BLOCK_THREADS, 1, 1)
        );

        let cfg128 = tiled_pipeline_launch_config(TiledPipelineTile::Bm128Bn64, m, n);
        assert_eq!(
            cfg128.grid_dim,
            (
                n.div_ceil(kernels_tiled_pipeline_128x64::TP128_BN),
                m.div_ceil(kernels_tiled_pipeline_128x64::TP128_BM),
                1,
            ),
            "Bm128Bn64 grid_dim must use the 128x64 block tile dimensions"
        );
        assert_eq!(
            cfg128.block_dim,
            (kernels_tiled_pipeline_128x64::TP128_BLOCK_THREADS, 1, 1)
        );

        // 128 のブロック行タイルは 64 の 2 倍のため、同じ m に対する
        // grid_dim.1（行方向のブロック数）は 128×64 の方が小さいか等しい
        // （末尾ブロックの端数処理により厳密に半分にはならない場合が
        // ある）。誤って 64×64 用の grid を 128×64 カーネルへ適用すると
        // grid_dim.1 が本来必要な行ブロック数の約 2 倍になる、という
        // 回帰の性質を明示的に固定する。
        assert!(
            cfg128.grid_dim.1 <= cfg64.grid_dim.1,
            "128x64 block tile must not require more row-blocks than 64x64 for the same m"
        );
    }

    /// イシュー #1137: `tiled_f32_kernel_kind`（GPU 不要の純粋関数）が
    /// 整列形状・非整列形状・パイプライン非可用（`pipeline_available =
    /// false`）の全組み合わせで fail-closed に classic へフォールバック
    /// することを検査する。`n=0`／`k=0` は `run_tiled_f32` 側の
    /// `validate_gemm_dims`／`run_f32_kernel` の早期 return で本関数まで
    /// 到達しない形状だが、本関数自体は純粋な整列判定のため
    /// `tiled_pipeline_alignment_ok(0, 0)` の契約どおり「4 の倍数」を
    /// 満たし `Pipeline` を返す（呼び出し元の早期 return が実際の起動を
    /// 防ぐため、本関数の判定だけを見て起動されるわけではない）。
    #[test]
    fn tiled_f32_kernel_kind_falls_back_to_classic_when_unavailable_or_unaligned() {
        // 整列形状（n%4==0 && k%4==0）+ オフセット 0（整列）+ パイプライン
        // 可用 → Pipeline。
        assert_eq!(
            tiled_f32_kernel_kind(true, 0, 256, 256),
            TiledF32Kernel::Pipeline,
            "aligned shape with pipeline available must select Pipeline"
        );
        assert_eq!(
            tiled_f32_kernel_kind(true, 0, 4096, 4096),
            TiledF32Kernel::Pipeline
        );

        // 非整列形状（n%4!=0 または k%4!=0）→ パイプライン可用でも
        // classic へフォールバック（fail-closed）。
        assert_eq!(
            tiled_f32_kernel_kind(true, 0, 257, 256),
            TiledF32Kernel::Classic,
            "n not a multiple of 4 must fall back to Classic even when pipeline is available"
        );
        assert_eq!(
            tiled_f32_kernel_kind(true, 0, 256, 257),
            TiledF32Kernel::Classic,
            "k not a multiple of 4 must fall back to Classic even when pipeline is available"
        );
        assert_eq!(
            tiled_f32_kernel_kind(true, 0, 61, 61),
            TiledF32Kernel::Classic
        );

        // パイプライン非可用（`new` 時のコンパイル失敗・sm_80 未満・
        // swizzle 変種による強制無効化）→ 整列形状でも常に classic。
        assert_eq!(
            tiled_f32_kernel_kind(false, 0, 256, 256),
            TiledF32Kernel::Classic,
            "pipeline unavailable must always select Classic regardless of alignment"
        );
        assert_eq!(
            tiled_f32_kernel_kind(false, 0, 4096, 4096),
            TiledF32Kernel::Classic
        );

        // 境界形状: n=0/k=0 は tiled_pipeline_alignment_ok の定義上
        // 「4 の倍数」を満たすため Pipeline 判定になる（本関数自体の契約。
        // 実際の起動は呼び出し元の m==0||n==0／k==0 早期 return で
        // カーネルへ到達しない。関数ドキュメントコメント参照）。
        assert_eq!(
            tiled_f32_kernel_kind(true, 0, 0, 0),
            TiledF32Kernel::Pipeline
        );
    }

    /// codex-review P0／Cursor Bugbot High 指摘（PR #1164）の回帰テスト:
    /// `CudaGemm::launch_tiled_f32_resident` が渡す `a_offset`（`CudaView`
    /// の連結バッファ先頭からの要素オフセット）が 4 要素（f32 4 個 = 16
    /// バイト）の倍数でない場合、`n`/`k` が cp.async 整列形状を満たし
    /// パイプラインが可用であっても常に classic へ fail-closed に
    /// フォールバックすることを検査する（`tiled_pipeline_offset_aligned`
    /// の契約）。整列オフセット（4 の倍数）では従来どおり Pipeline を
    /// 選ぶことも合わせて確認する。
    #[test]
    fn tiled_f32_kernel_kind_falls_back_to_classic_for_unaligned_a_offset() {
        // 整列形状 + 整列オフセット（4 の倍数）→ Pipeline。
        assert_eq!(
            tiled_f32_kernel_kind(true, 0, 256, 256),
            TiledF32Kernel::Pipeline
        );
        assert_eq!(
            tiled_f32_kernel_kind(true, 4, 256, 256),
            TiledF32Kernel::Pipeline,
            "a_offset that is a multiple of 4 elements (16 bytes) must not block Pipeline"
        );
        assert_eq!(
            tiled_f32_kernel_kind(true, 1024, 256, 256),
            TiledF32Kernel::Pipeline
        );

        // 整列形状だが非整列オフセット（4 の倍数でない）→ n/k・
        // pipeline_available が全て条件を満たしていても classic へ
        // fail-closed にフォールバックする（PR #1164 の主眼）。
        for offset in [1usize, 2, 3, 5, 1023] {
            assert_eq!(
                tiled_f32_kernel_kind(true, offset, 256, 256),
                TiledF32Kernel::Classic,
                "a_offset={offset} not a multiple of 4 must fall back to Classic even when \
                 shape is aligned and pipeline is available"
            );
        }
    }

    /// イシュー #1344: `tiled_pipeline_tile_kind` の閾値判定（GPU 不要の
    /// 純粋関数）を、プレースホルダ閾値（`u32::MAX`）の下で検査する。
    /// GB10 実機実測で閾値定数を確定したら、本テストの期待値も実測結果に
    /// 合わせて更新する（`tiled_f32_kernel_kind_falls_back_to_classic_*`
    /// と同型の表テスト方針）。
    #[test]
    fn tiled_pipeline_tile_kind_falls_back_to_64x64_when_unavailable_or_below_threshold() {
        // 第 2 スロット自体が未コンパイル（`has_128x64 = false`）の場合は
        // 形状に依らず常に 64×64（既定 `TILED_PIPELINE_128X64_PRODUCTION_
        // ENABLED = false` の本番状態と同じ）。
        for (n, k) in [(64u32, 64u32), (4096, 4096), (0, 0)] {
            assert_eq!(
                tiled_pipeline_tile_kind(false, n, k),
                TiledPipelineTile::Bm64Bn64,
                "has_128x64=false must always fall back to Bm64Bn64 regardless of shape \
                 (n={n}, k={k})"
            );
        }

        // 第 2 スロットが利用可能でも、いずれかの軸が確定済み閾値
        // （`TILED_PIPELINE_128X64_MIN_N`/`_MIN_K`。GB10 実機実測根拠は
        // `docs/perf/cuda-gemm-tiled-pipeline.md`「#1344」節）を下回る
        // 形状では 64×64 側になる（AND 条件の fail-closed 契約。`None`
        // 閾値の軸は常に満たすため、その軸単体では検証しない）。
        if let Some(min_n) = TILED_PIPELINE_128X64_MIN_N {
            let k_floor = TILED_PIPELINE_128X64_MIN_K.unwrap_or(0);
            assert_eq!(
                tiled_pipeline_tile_kind(true, min_n - 1, k_floor),
                TiledPipelineTile::Bm64Bn64,
                "n below TILED_PIPELINE_128X64_MIN_N must fall back to Bm64Bn64"
            );
        }
        if let Some(min_k) = TILED_PIPELINE_128X64_MIN_K {
            let n_floor = TILED_PIPELINE_128X64_MIN_N.unwrap_or(0);
            assert_eq!(
                tiled_pipeline_tile_kind(true, n_floor, min_k - 1),
                TiledPipelineTile::Bm64Bn64,
                "k below TILED_PIPELINE_128X64_MIN_K must fall back to Bm64Bn64"
            );
        }

        // 両軸とも確定済み閾値以上（`None` は無条件で満たす扱い）を
        // 満たす場合は 128×64 が選ばれる（AND 条件・境界含む契約の確認。
        // 閾値未確定〈両方 `None`〉の間は任意形状で検証する）。
        let n_at_threshold = TILED_PIPELINE_128X64_MIN_N.unwrap_or(1024);
        let k_at_threshold = TILED_PIPELINE_128X64_MIN_K.unwrap_or(1024);
        assert_eq!(
            tiled_pipeline_tile_kind(true, n_at_threshold, k_at_threshold),
            TiledPipelineTile::Bm128Bn64,
            "both axes meeting the confirmed threshold (or unconstrained) must select \
             Bm128Bn64"
        );
    }

    /// イシュー #856。`CudaGemm::should_launch_wmma_tf32_staged_swizzle`
    /// が実際に `self` へ委譲する `swizzle::should_apply_swizzle` を、
    /// `WMMA_TF32_STAGED_BLOCK_M`/`_N`（64×64）由来のブロック数導出込みで
    /// 検査する CPU 側単体テスト（実機不要。`CudaGemm` インスタンスの
    /// 構築自体は実 GPU コンパイルを要するため `#[ignore]` にせざるを
    /// 得ないが、判定ロジックの入力（raw 次元→ブロック数の導出式）は
    /// 実機非依存で検証できる。`gemm_mma.rs::CudaMmaGemm::
    /// should_launch_swizzle_kernel` は private だが同じ導出式
    /// （`div_ceil` + `should_apply_swizzle`）を使うため、本テストは
    /// `CudaGemm::should_launch_wmma_tf32_staged_swizzle` 自身と同じ式を
    /// 直接呼び出す形で退化させず、公開関数 `swizzle::should_apply_swizzle`
    /// へブロック数を渡す形で再現する）。
    #[test]
    fn wmma_tf32_staged_swizzle_size_condition_matches_expected_shapes() {
        let derive_and_check = |m: u32, n: u32, k: u32| -> bool {
            let num_m_blocks = m.div_ceil(kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_M);
            let num_n_blocks = n.div_ceil(kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_N);
            crate::swizzle::should_apply_swizzle(m, n, num_m_blocks, num_n_blocks, k)
        };

        // 実測承認点 M=N=K=4096（正方形）: 採用（§7.4.1・docs/perf/
        // cuda-gemm-swizzle-ab.md §7.7.6）。
        assert!(
            derive_and_check(4096, 4096, 4096),
            "M=N=K=4096（実測承認点）は適用対象のはずです"
        );
        // 2048 正方形は実測承認点（4096）未満のため非適用（§7.4.1 の
        // サイズ条件付きガードレールが 4096 級正方形限定であることの検査。
        // 512/1024/2048 は劣化 5% 以内を条件に base のまま扱う契約）。
        assert!(
            !derive_and_check(2048, 2048, 2048),
            "M=N=K=2048 は実測承認点（4096）未満のため非適用のはずです"
        );
        // 非正方形（M=8192, N=4096, K=4096）は PR #784 codex-review P1
        // 是正が導入した正方形ガードにより非適用（未検証の非正方形形状への
        // 外挿を防ぐ。`swizzle.rs::SWIZZLE_APPLY_MIN_SQUARE_DIM` 参照）。
        assert!(
            !derive_and_check(8192, 4096, 4096),
            "非正方形（M != N）は正方形ガードにより非適用のはずです"
        );
        // M=N=4096・K=8 は K ガード未達のため非適用（メモリアクセス量・
        // L2 再利用特性が実測承認点と大きく異なる形状への外挿を防ぐ。
        // `swizzle.rs::SWIZZLE_APPLY_MIN_K` 参照）。
        assert!(
            !derive_and_check(4096, 4096, 8),
            "K=8（実測承認点 4096 未満）は K ガードにより非適用のはずです"
        );
    }

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
                fandhe_ai_backend_cpu::matmul_reference_fma(
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

                let report = fandhe_ai_backend_cpu::compare(&c_gpu, &c_ref)
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
                fandhe_ai_backend_cpu::matmul_reference_fma(
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

                let report = fandhe_ai_backend_cpu::compare(&c_gpu, &c_ref)
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
    /// **イシュー #1106（案 A）**: 本ヘルパーが使う
    /// [`fandhe_ai_backend_cpu::assert_parity`]（複合判定そのものでの厳密
    /// ゼロ fail 判定）は、GB10 実機実測（#1106 reopen コメント・診断ダンプ
    /// `wmma_tf32_opt_kernel_parity_diagnostic_dump_issue_1106`〈修正確定
    /// につき削除済み〉）により、opt カーネルの大半の形状（TF32 丸めの
    /// K 方向蓄積により非ゼロ fail_count を持つのが既知の恒常特性。
    /// opt/basic bit-identical・sm_86/GB10 世代間差分なし。
    /// `docs/perf/cuda-tensor-core-tolerance-opt-remeasurement.md` §5〜§7）
    /// では原理的に成立しないことが判明した。ゼロ fail が実際に成立する
    /// のは 1x1x1（sub-K-tile）のみであり、[`wmma_tf32_opt_kernel_matches_reference_across_shapes`]
    /// はこの 1 形状のみを検査する。非ゼロ fail が判明した残り 8 形状
    /// （64×64×64・128×128×128・512×512×512〈2 シード〉・63×65×33・
    /// 65×63×17・64×96×256・512×512×4096・4096×4096×4096）は
    /// `ParityBaseline::BASELINES` へ実測値付きで移し、
    /// [`wmma_tf32_opt_kernel_parity_does_not_regress`]（baseline 非後退
    /// 方式）が検査する（旧 `wmma_tf32_opt_kernel_k4096_stress` は全ケースが
    /// この移管対象になったため削除済み）。tolerance 定数
    /// （`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）は変更していない
    /// （ユーザー承認 2026-09-02）。
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
        fandhe_ai_backend_cpu::matmul_reference_fma(
            &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
        )
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");

        validate_gemm_dims(a.len(), b.len(), m, n, k)
            .expect("test shape must be a valid GEMM dimension");
        validate_wmma_tf32_opt_k_bound(k).expect("test k must satisfy WMMA(TF32) opt k bound");

        let c_gpu = gemm
            .run_wmma_tf32_opt_kernel(func, &a, &b, m, n, k)
            .expect("opt WMMA(TF32) kernel execution must succeed on this ignored test runner");

        fandhe_ai_backend_cpu::assert_parity(context, &c_gpu, &c_ref);
    }

    /// opt カーネル**単独**の厳密ゼロ fail 判定テスト（PR #678
    /// codex-review P1 再指摘対応・イシュー #1106 案 A で対象形状を縮小）。
    ///
    /// GB10 実機実測（[`assert_wmma_tf32_opt_kernel_parity`] ドキュメン
    /// テーションコメント参照）により、`assert_parity`（厳密ゼロ fail
    /// 判定）が実際に成立するのは 1x1x1（sub-K-tile。K 方向蓄積が発生
    /// せず TF32 丸め誤差が蓄積しない）のみと判明した。他の 8 形状
    /// （旧 `cases` に含まれていた 64×64×64・128×128×128・512×512×512・
    /// 63×65×33・65×63×17・64×96×256 と、旧 `wmma_tf32_opt_kernel_k4096_stress`
    /// の 512×512×4096・4096×4096×4096）は `ParityBaseline::BASELINES`
    /// （`wmma_tf32_opt_kernel_parity_does_not_regress` が検査）へ移管した。
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

        // seed は 3006（元の 7 ケース配列での 1x1x1 の位置 idx=6 に対応する
        // 3000+6）を直接指定する。案 A の縮小前は `3000 + idx` で自動算出
        // していたが、配列を 1 要素へ縮小すると `idx` が 0 に戻り、実測して
        // いない seed=3000（元は 64x64x64 用）が (1,1,1) 形状に誤って
        // 適用されてしまう（イシュー #1106 GB10 全件洗い出しで発覚したバグ。
        // 実測は seed=3006〈0xbbe〉のみ。GB10 実機で fail_count=0/1 を確認
        // 済み。`docs/perf/cuda-parity-baseline.md` §10.5 参照）。
        assert_wmma_tf32_opt_kernel_parity(
            &gemm,
            func,
            "opt kernel shape m=1 n=1 k=1",
            3006,
            1,
            1,
            1,
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

    // `validate_tiled_k_bound`（`kernels::TILE`=32 基準）を `TILED_F32`
    // （`TILED_F32_BK`=16 基準）にもそのまま流用してよいという判断
    // （同関数のドキュメンテーションコメント参照）は「`TILE` >
    // `TILED_F32_BK`（同関数が計算する上限のほうが厳しい＝安全側）」
    // という前提に依存する。この前提はコメントに文章として書かれて
    // いるのみで機械検証されていなかったため、コンパイル時定数検査で
    // 固定する（レビュー指摘。イシュー #1032）。将来どちらかの定数の
    // 変更で前提が崩れた場合はビルドが失敗し、`TILED_F32` 専用の
    // `k` 上限検証関数の新設が必要になったことに気づける。
    const _: () = assert!(
        kernels::TILE > kernels::TILED_F32_BK,
        "TILE が TILED_F32_BK 以下になったため、\
         validate_tiled_k_bound を TILED_F32 経路へ流用する前提が崩れている"
    );

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

    /// イシュー #856 受け入れ基準（実機検証）: [`CudaGemm::new`]（本番既定
    /// コンストラクタ）が実際に `wmma_tf32_staged_swizzle`（サイズ条件付き
    /// swizzle 変種）を結線していることを確認する
    /// （`gemm_mma.rs::CudaMmaGemm::
    /// mma_f16_new_wires_size_conditional_swizzle_into_production_constructor`
    /// と同型の 3 分岐検査）。
    ///
    /// `#[ignore]`: `CudaDevice::new` が CUDA 実機を要求するため
    /// （本ファイル冒頭コメント「検証状態」）。DGX Spark GB10 等の実機で
    /// `cargo test -p fandhe-ai-backend-cuda --lib --release -- --ignored --nocapture
    /// wmma_tf32_staged_new_wires_size_conditional_swizzle_into_production_constructor`
    /// から実行する。feature 非依存（`wmma_tf32_staged_swizzle`
    /// フィールド・`CudaGemm::new` はいずれも feature ゲートされていない
    /// ため、`internal-diagnostics` を要求しない）。
    #[test]
    #[ignore = "CUDA 実機（compute capability 8.0 以降）必須"]
    fn wmma_tf32_staged_new_wires_size_conditional_swizzle_into_production_constructor() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
        let production =
            CudaGemm::new(&device).expect("CudaGemm::new must succeed on ignored test runner");

        if !production.wmma_tf32_staged_available() {
            // staged base 自体がこの device で使用不能な場合、swizzle 変種
            // も試みられない契約（`new` 実装参照）。この分岐は
            // `wmma_tf32_staged_available` を主張する他テストで既に
            // カバーされているため、本テストは何もしない。
            return;
        }

        let expected_group_width_if_compile_succeeds =
            device.multiprocessor_count().map(|num_sms| {
                crate::swizzle::select_swizzle_group_width(
                    num_sms,
                    kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_M,
                    kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_N,
                )
            });

        match (
            device.multiprocessor_count(),
            production.wmma_tf32_staged_swizzle_group_width(),
            production.wmma_tf32_staged_swizzle_unavailable_reason(),
        ) {
            (None, actual_group_width, reason) => {
                // 分岐 (a): SM 数取得失敗。
                assert_eq!(
                    actual_group_width, None,
                    "分岐 (a)（SM 数取得失敗）では wmma_tf32_staged_swizzle_group_width は \
                     None のはずです"
                );
                assert_eq!(
                    reason, None,
                    "分岐 (a)（SM 数取得失敗）では \
                     wmma_tf32_staged_swizzle_unavailable_reason は None のはずです \
                     （コンパイル自体を試みていないため）"
                );
            }
            (Some(_), Some(actual_group_width), reason) => {
                // 分岐 (b): SM 数取得成功・コンパイル成功。
                assert_eq!(
                    Some(actual_group_width),
                    expected_group_width_if_compile_succeeds,
                    "分岐 (b)（SM 数取得成功・コンパイル成功）では \
                     wmma_tf32_staged_swizzle_group_width が select_swizzle_group_width の \
                     動的選択幅と一致するはずです"
                );
                assert_eq!(
                    reason, None,
                    "分岐 (b)（コンパイル成功）では \
                     wmma_tf32_staged_swizzle_unavailable_reason は None のはずです"
                );
                // 分岐 (b) では、実測承認点 M=N=K=4096（正方形）で
                // should_launch_wmma_tf32_staged_swizzle が true を返す
                // （＝ launch_wmma_tf32/run_wmma_tf32 が実際に swizzle
                // 変種を起動する）ことも合わせて確認する。
                assert!(
                    production.wmma_tf32_staged_swizzle_applies(4096, 4096, 4096),
                    "分岐 (b) では M=N=K=4096 で swizzle 変種が適用されるはずです"
                );
            }
            (Some(_), None, reason) => {
                // 分岐 (c): SM 数取得成功・コンパイル失敗（fail-soft 縮退）。
                assert!(
                    reason.is_some(),
                    "分岐 (c)（SM 数取得成功・コンパイル失敗）では \
                     wmma_tf32_staged_swizzle_unavailable_reason に失敗理由が記録されている \
                     はずです"
                );
                assert!(
                    !production.wmma_tf32_staged_swizzle_applies(4096, 4096, 4096),
                    "分岐 (c)（変種未保持）では swizzle 変種は適用されないはずです"
                );
            }
        }
    }

    /// イシュー #741 受け入れ基準（実機検証）:
    /// [`CudaGemm::new_with_tf32_staged_swizzle`] が生成する各
    /// `group_width` の変種が、[`CudaGemm::new_without_tf32_staged_swizzle`]
    /// （base。TF32 opt-staged 経路。イシュー #856 で `CudaGemm::new` から
    /// 切替——本番結線後は `new` 自体が形状によって swizzle 変種を選び
    /// うるため、恒久的に swizzle 無適用を保証する本コンストラクタを base
    /// に使う）と**ビット一致**の出力を返すことを確認する。
    ///
    /// swizzle はブロックがどの `(m_block, n_block)` を担当するかの割り当て
    /// のみを変え、各ブロック内部の計算（wmma フラグメントロード・
    /// アキュムレート順序）は変えないため（`kernels_wmma_opt.rs::
    /// wmma_tf32_f32_staged_source_with_swizzle` ドキュメンテーション
    /// コメント参照）、`gemm_mma.rs::
    /// mma_f16_swizzle_variant_matches_base_bit_exact_output` と同じ論法
    /// で tolerance を使わない bit 等値で主張できる（`.claude/rules/
    /// coding-rust.md` の「バックエンド間数値一致テストの許容誤差を単独で
    /// 緩和しない」契約に抵触しない。swizzle 変種間比較はバックエンド間
    /// 比較ではなく同一バックエンド内の実装詳細比較のため tolerance の
    /// 対象外）。
    ///
    /// `group_width` は動的選択結果（`device.multiprocessor_count()`
    /// 実測値ベース）に加え、参考として固定候補 `8`/`16` も検査する
    /// （実装計画 4 節「ステップ 5」）。
    ///
    /// fail-closed（実装計画 4 節「ステップ 5」）: staged カーネルが
    /// `CudaGemm::new` 時点でコンパイル・ロードに失敗する環境では
    /// `wmma_tf32_staged_available()` の assert が先に落ち、本テストが
    /// 静かに basic 経路同士の空比較へ退化しない。
    ///
    /// 形状: 全形状とも staged 整列条件（`n % 4 == 0 && k % 4 == 0`）を
    /// 満たす:
    /// - `(512, 512, 512)`: ブロックタイル（`WMMA_TF32_STAGED_BLOCK_M/N`
    ///   = 64）の整数倍（`num_m_blocks=8`）。group_width=8 では
    ///   `full_groups=1・remainder=0`（remainder 分岐は経由しない）、
    ///   group_width=16 では `full_groups=0・remainder=8`（remainder
    ///   分岐のみ経由）となり、両分岐を候補幅間でカバーする。
    /// - `(80, 136, 160)`: タイル端形状（`m` がブロックタイル非整数倍）。
    /// - `(1088, 256, 2048)`: `gemm_mma.rs` の f16 側 bit 一致テストと
    ///   同じ `full_groups` 分岐形状（`num_m_blocks=17`）を再利用し、
    ///   TF32 側でも full_groups・remainder の両分岐を group_width∈
    ///   {8,16} 双方で経由させる。
    ///
    /// `#[ignore]`: 本セッションは NVRTC 非搭載のため実行できない。DGX
    /// Spark GB10 等の実機で `cargo test -p fandhe-ai-backend-cuda --lib --release
    /// --features internal-diagnostics -- --ignored --nocapture
    /// wmma_tf32_staged_swizzle_variant_matches_base_bit_exact_output` から
    /// 実行する。`internal-diagnostics` feature（既定 off）でのみコンパイル
    /// される（[`CudaGemm::new_with_tf32_staged_swizzle`] 自体が同 feature
    /// でゲートされているため）。
    #[cfg(feature = "internal-diagnostics")]
    #[test]
    #[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
    fn wmma_tf32_staged_swizzle_variant_matches_base_bit_exact_output() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
        // イシュー #856: `CudaGemm::new`（本番既定コンストラクタ）ではなく
        // `new_without_tf32_staged_swizzle` を base に使う。本番結線後は
        // `new` 自体が SM 数実測成功時に swizzle 変種を追加コンパイルする
        // ため、`shapes` に将来 M=N=K>=4096 の正方形を追加した場合に `new`
        // のままだと base 自身が swizzle 変種を計測してしまい、本テストが
        // 自明に pass する A/A 誤認へ退化する
        // （`new_without_tf32_staged_swizzle` ドキュメンテーションコメント
        // 「導入理由」節・`examples/gemm_wmma_tf32_swizzle_bench.rs` の同型
        // 是正参照）。
        let base = CudaGemm::new_without_tf32_staged_swizzle(&device)
            .expect("base new_without_tf32_staged_swizzle must succeed on ignored test runner");
        assert!(
            base.wmma_tf32_staged_available(),
            "TF32 opt-staged kernel must be available on ignored test runner \
             (reason: {:?}); a fallback-to-basic comparison would degenerate \
             into a no-op",
            base.wmma_tf32_staged_unavailable_reason()
        );

        let num_sms = device.multiprocessor_count().unwrap_or(1).max(1);
        let dynamic_group_width = crate::swizzle::select_swizzle_group_width(
            num_sms,
            kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_M,
            kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_N,
        );

        // 3 形状のうち (512,512,512) は M=N だが実測承認点 4096 未満のため
        // 現行の should_apply_swizzle では非適用（base に `new` を使っても
        // 現状は A/A 化しない）。上記コメントのとおり将来 4096 級正方形を
        // 追加する場合に備え、base は恒久的に `new_without_tf32_staged_
        // swizzle` を使う契約とする。
        let shapes: [(u32, u32, u32); 3] = [(512, 512, 512), (80, 136, 160), (1088, 256, 2048)];
        let seed: u64 = 424_243;

        for group_width in [dynamic_group_width, 8, 16] {
            let variant = CudaGemm::new_with_tf32_staged_swizzle(&device, group_width)
                .unwrap_or_else(|err| {
                    panic!("group_width={group_width}: new_with_tf32_staged_swizzle failed: {err}")
                });

            for &(m, n, k) in &shapes {
                let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
                let a: Vec<f32> = rng.fill_vec((m as usize) * (k as usize));
                let b: Vec<f32> = rng.fill_vec((k as usize) * (n as usize));

                let base_c = base.run_wmma_tf32(&a, &b, m, n, k).unwrap_or_else(|err| {
                    panic!("base run_wmma_tf32 failed for shape (m={m}, n={n}, k={k}): {err}")
                });
                let variant_c = variant
                    .run_wmma_tf32(&a, &b, m, n, k)
                    .unwrap_or_else(|err| {
                        panic!(
                            "group_width={group_width} run_wmma_tf32 failed for shape \
                         (m={m}, n={n}, k={k}): {err}"
                        )
                    });

                assert_eq!(
                    variant_c, base_c,
                    "shape (m={m}, n={n}, k={k}) group_width={group_width}: swizzle \
                     変種の出力が base と bit 一致しません（remap がブロック内部の \
                     計算・アキュムレート順序に影響していないか確認すること）"
                );
            }
        }
    }

    /// イシュー #1034 受け入れ基準（実機検証）: [`CudaGemm::
    /// new_with_tiled_f32_swizzle`] が生成する本番既定 f32 経路
    /// （`TILED_F32`）の swizzle 変種が、`CudaGemm::new`（base）と
    /// **ビット一致**の出力を返すことを確認する。
    ///
    /// swizzle はブロック割り当て（どの `(m_block, n_block)` をどの物理
    /// ブロックが担当するか）のみを変え、各ブロック内部の積和順序・
    /// アキュムレート方式は変えないため、`wmma_tf32_staged_swizzle_
    /// variant_matches_base_bit_exact_output` と同じ論法で tolerance を
    /// 使わない bit 等値で主張できる（同一バックエンド内の実装詳細比較の
    /// ため `.claude/rules/coding-rust.md` の許容誤差緩和禁止契約の対象
    /// 外）。**本番既定コンストラクタ `CudaGemm::new` は `tiled_f32` へ
    /// サイズ条件付き動的変種を一切構築しない**（`new_with_tiled_f32_
    /// swizzle` ドキュメンテーションコメント「本番結線は本イシューの
    /// スコープ外」節）ため、`wmma_tf32_staged_swizzle_variant_matches_
    /// base_bit_exact_output` と異なり base に専用の `new_without_*`
    /// バリアントは不要（`CudaGemm::new` をそのまま base として使っても
    /// A/A 誤認は生じない）。
    ///
    /// `group_width` は動的選択結果（`device.multiprocessor_count()`
    /// 実測値ベース。`kernels::TILE` x `kernels::TILE` ブロックタイル）に
    /// 加え、参考として固定候補 `8`/`16` も検査する。
    ///
    /// 形状（`kernels::TILE` = 32 を基準に `wmma_tf32_staged_swizzle_
    /// variant_matches_base_bit_exact_output`〈ブロックタイル 64〉と同じ
    /// ブロック数比になるよう縮尺した）:
    /// - `(256, 256, 256)`: `num_m_blocks=8`。group_width=8 では
    ///   `full_groups=1・remainder=0`（remainder 分岐は経由しない）、
    ///   group_width=16 では `full_groups=0・remainder=8`（remainder
    ///   分岐のみ経由）となり、両分岐を候補幅間でカバーする。
    /// - `(80, 136, 160)`: タイル端形状（いずれの次元も `TILE`〈32〉の
    ///   非整数倍。手動境界チェック分岐も併せて踏む）。
    /// - `(544, 256, 2048)`: `num_m_blocks=17`。group_width∈{8,16} 双方で
    ///   full_groups・remainder の両分岐を経由させる（`wmma_tf32_staged_
    ///   swizzle_variant_matches_base_bit_exact_output` の `(1088, 256,
    ///   2048)`〈`num_m_blocks=17`〉と同じブロック数比）。
    ///
    /// `#[ignore]`: 本セッションは NVRTC 非搭載のため実行できない。DGX
    /// Spark GB10 等の実機で `cargo test -p fandhe-ai-backend-cuda --lib --release
    /// --features internal-diagnostics -- --ignored --nocapture
    /// tiled_f32_swizzle_variant_matches_base_bit_exact_output` から実行
    /// する。`internal-diagnostics` feature（既定 off）でのみコンパイル
    /// される（[`CudaGemm::new_with_tiled_f32_swizzle`] 自体が同 feature
    /// でゲートされているため）。
    #[cfg(feature = "internal-diagnostics")]
    #[test]
    #[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
    fn tiled_f32_swizzle_variant_matches_base_bit_exact_output() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
        let base =
            CudaGemm::new(&device).expect("base CudaGemm::new must succeed on ignored test runner");

        let num_sms = device.multiprocessor_count().unwrap_or(1).max(1);
        // イシュー #1139: #1032（PR #1072）のレジスタブロッキング導入で
        // `TILED_F32` のブロックタイルは `TILE`（32）ではなく `TILED_F32_BM`/
        // `BN`（64）になったため、単位を揃えて選択する（`lib.rs::
        // Diagnostics::tiled_f32_swizzle_group_width` と同じ是正）。
        let dynamic_group_width = crate::swizzle::select_swizzle_group_width(
            num_sms,
            kernels::TILED_F32_BM,
            kernels::TILED_F32_BN,
        );

        let shapes: [(u32, u32, u32); 3] = [(256, 256, 256), (80, 136, 160), (544, 256, 2048)];
        let seed: u64 = 424_244;

        for group_width in [dynamic_group_width, 8, 16] {
            let variant = CudaGemm::new_with_tiled_f32_swizzle(&device, group_width)
                .unwrap_or_else(|err| {
                    panic!("group_width={group_width}: new_with_tiled_f32_swizzle failed: {err}")
                });

            for &(m, n, k) in &shapes {
                let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
                let a: Vec<f32> = rng.fill_vec((m as usize) * (k as usize));
                let b: Vec<f32> = rng.fill_vec((k as usize) * (n as usize));

                // イシュー #1137: base 側は常に classic 版（swizzle 非適用の
                // `kernels::TILED_F32`）で固定する。`run_tiled_f32`（無印）
                // は #1137 以降 cp.async パイプライン版へ形状条件付きで
                // 分岐しうるため、整列形状では base と variant
                // （`new_with_tiled_f32_swizzle` がパイプラインを強制無効化
                // 済み）とでカーネル系統そのものが異なってしまい、swizzle
                // の remap 差分のみを検査するという本テストの前提が崩れる。
                let base_c = base
                    .run_tiled_f32_classic(&a, &b, m, n, k)
                    .unwrap_or_else(|err| {
                        panic!(
                            "base run_tiled_f32_classic failed for shape (m={m}, n={n}, k={k}): {err}"
                        )
                    });
                let variant_c = variant
                    .run_tiled_f32(&a, &b, m, n, k)
                    .unwrap_or_else(|err| {
                        panic!(
                            "group_width={group_width} run_tiled_f32 failed for shape \
                         (m={m}, n={n}, k={k}): {err}"
                        )
                    });

                assert_eq!(
                    variant_c, base_c,
                    "shape (m={m}, n={n}, k={k}) group_width={group_width}: swizzle \
                     変種の出力が base と bit 一致しません（remap がブロック内部の \
                     計算・アキュムレート順序に影響していないか確認すること）"
                );
            }
        }
    }

    /// イシュー #1034 受け入れ基準（実機検証）: [`CudaGemm::
    /// new_with_tiled_f32_swizzle`] の swizzle 変種が CPU 参照実装
    /// （[`fandhe_ai_backend_cpu::matmul_reference_fma`]）と統一複合判定
    /// （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
    /// `.claude/rules/coding-rust.md`）で一致することを確認する。
    ///
    /// `tiled_f32_swizzle_variant_matches_base_bit_exact_output` は変種と
    /// base の**相互**一致のみを検査するため、base 自体が CPU 参照実装と
    /// parity を保っているか（tolerance を用いた真の数値正当性）は別途
    /// 検査する必要がある（`tests/gemm_tiled.rs::
    /// assert_tiled_f32_matches_cpu_reference` が base 側で既に検査
    /// 済みだが、swizzle 変種側も独立して同じ主張を成立させておく）。
    ///
    /// `#[ignore]`: 本セッションは NVRTC 非搭載のため実行できない。DGX
    /// Spark GB10 等の実機で `cargo test -p fandhe-ai-backend-cuda --lib --release
    /// --features internal-diagnostics -- --ignored --nocapture
    /// tiled_f32_swizzle_variant_matches_cpu_reference` から実行する。
    #[cfg(feature = "internal-diagnostics")]
    #[test]
    #[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
    fn tiled_f32_swizzle_variant_matches_cpu_reference() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");

        let num_sms = device.multiprocessor_count().unwrap_or(1).max(1);
        // イシュー #1139: #1032（PR #1072）のレジスタブロッキング導入で
        // `TILED_F32` のブロックタイルは `TILE`（32）ではなく `TILED_F32_BM`/
        // `BN`（64）になったため、単位を揃えて選択する（`lib.rs::
        // Diagnostics::tiled_f32_swizzle_group_width` と同じ是正）。
        let dynamic_group_width = crate::swizzle::select_swizzle_group_width(
            num_sms,
            kernels::TILED_F32_BM,
            kernels::TILED_F32_BN,
        );

        let shapes: [(u32, u32, u32); 3] = [(256, 256, 256), (80, 136, 160), (544, 256, 2048)];
        let seed: u64 = 424_245;

        for group_width in [dynamic_group_width, 8, 16] {
            let variant = CudaGemm::new_with_tiled_f32_swizzle(&device, group_width)
                .unwrap_or_else(|err| {
                    panic!("group_width={group_width}: new_with_tiled_f32_swizzle failed: {err}")
                });

            for &(m, n, k) in &shapes {
                let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
                let a: Vec<f32> = rng.fill_vec((m as usize) * (k as usize));
                let b: Vec<f32> = rng.fill_vec((k as usize) * (n as usize));

                let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
                fandhe_ai_backend_cpu::matmul_reference_fma(
                    &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
                )
                .expect("matmul_reference_fma shape validation must pass for test input");

                let c_gpu = variant
                    .run_tiled_f32(&a, &b, m, n, k)
                    .unwrap_or_else(|err| {
                        panic!(
                            "group_width={group_width} run_tiled_f32 failed for shape \
                         (m={m}, n={n}, k={k}): {err}"
                        )
                    });

                fandhe_ai_backend_cpu::assert_parity(
                    &format!(
                        "tiled f32 swizzle (group_width={group_width}) CPU/GPU parity \
                         (shape m={m} n={n} k={k})"
                    ),
                    &c_gpu,
                    &c_ref,
                );
            }
        }
    }

    /// イシュー #743 受け入れ基準（実機検証）: [`CudaGemm::
    /// new_with_tf32_staged_pads`] が生成する SMEM パディング変種が、
    /// [`CudaGemm::new`]（base）と**ビット一致**の出力を返すことを確認
    /// する。
    ///
    /// パディングはタイル行ストライドのみを変え、各ブロック内部の計算・
    /// アキュムレート順序は変えない（`kernels_wmma_opt.rs::
    /// wmma_tf32_f32_staged_source_with_pads` ドキュメンテーションコメント
    /// 参照）ため、
    /// `wmma_tf32_staged_swizzle_variant_matches_base_bit_exact_output` と
    /// 同じ論法で tolerance を使わない bit 等値で主張できる（同変種間比較は
    /// バックエンド間比較ではなく同一バックエンド内の実装詳細比較のため
    /// `.claude/rules/coding-rust.md` の許容誤差緩和禁止契約の対象外）。
    ///
    /// `(a_pad, b_pad)` 候補: `(WMMA_TF32_STAGED_A_PAD, WMMA_TF32_STAGED_B_PAD)`
    /// （本番既定値と同一構成。恒等変換として byte 完全一致の回帰も兼ねる）
    /// と `(WMMA_TF32_STAGED_A_PAD, WMMA_TF32_STAGED_B_PAD + 4)`（72。
    /// バンクコンフリクト解消候補。本ファイル `WMMA_TF32_STAGED_B_PAD`
    /// 直下コメント参照）。
    ///
    /// `#[ignore]`: 本セッションは NVRTC 非搭載のため実行できない。DGX
    /// Spark GB10 等の実機で `cargo test -p fandhe-ai-backend-cuda --lib --release
    /// --features internal-diagnostics -- --ignored --nocapture
    /// wmma_tf32_staged_pad_variant_matches_base_bit_exact_output` から
    /// 実行する。`internal-diagnostics` feature（既定 off）でのみ
    /// コンパイルされる（[`CudaGemm::new_with_tf32_staged_pads`] 自体が
    /// 同 feature でゲートされているため）。
    #[cfg(feature = "internal-diagnostics")]
    #[test]
    #[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
    fn wmma_tf32_staged_pad_variant_matches_base_bit_exact_output() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
        let base =
            CudaGemm::new(&device).expect("base CudaGemm::new must succeed on ignored test runner");
        assert!(
            base.wmma_tf32_staged_available(),
            "TF32 opt-staged kernel must be available on ignored test runner \
             (reason: {:?}); a fallback-to-basic comparison would degenerate \
             into a no-op",
            base.wmma_tf32_staged_unavailable_reason()
        );

        let shapes: [(u32, u32, u32); 3] = [(512, 512, 512), (80, 136, 160), (1088, 256, 2048)];
        let seed: u64 = 743_001;

        let candidates: [(u32, u32); 2] = [
            (
                kernels_wmma_opt::WMMA_TF32_STAGED_A_PAD,
                kernels_wmma_opt::WMMA_TF32_STAGED_B_PAD,
            ),
            (
                kernels_wmma_opt::WMMA_TF32_STAGED_A_PAD,
                kernels_wmma_opt::WMMA_TF32_STAGED_B_PAD + 4,
            ),
        ];

        for (a_pad, b_pad) in candidates {
            let variant = CudaGemm::new_with_tf32_staged_pads(&device, a_pad, b_pad)
                .unwrap_or_else(|err| {
                    panic!("a_pad={a_pad} b_pad={b_pad}: new_with_tf32_staged_pads failed: {err}")
                });

            for &(m, n, k) in &shapes {
                let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
                let a: Vec<f32> = rng.fill_vec((m as usize) * (k as usize));
                let b: Vec<f32> = rng.fill_vec((k as usize) * (n as usize));

                let base_c = base.run_wmma_tf32(&a, &b, m, n, k).unwrap_or_else(|err| {
                    panic!("base run_wmma_tf32 failed for shape (m={m}, n={n}, k={k}): {err}")
                });
                let variant_c = variant
                    .run_wmma_tf32(&a, &b, m, n, k)
                    .unwrap_or_else(|err| {
                        panic!(
                            "a_pad={a_pad} b_pad={b_pad} run_wmma_tf32 failed for shape \
                         (m={m}, n={n}, k={k}): {err}"
                        )
                    });

                assert_eq!(
                    variant_c, base_c,
                    "shape (m={m}, n={n}, k={k}) a_pad={a_pad} b_pad={b_pad}: pad \
                     変種の出力が base と bit 一致しません（パディング変更がタイル \
                     行ストライド以外に影響していないか確認すること）"
                );
            }
        }
    }

    /// イシュー #994 受け入れ基準（実機検証）: [`CudaGemm::
    /// new_tf32_opt_only`]／[`CudaGemm::new_tf32_basic_only`] が、それぞれ
    /// 意図したカーネル種別（可用性フラグ）を強制することを確認する。
    /// 数値一致（誤差分布）は `examples/wmma_tolerance_probe.rs
    /// --tf32-kernel opt/basic` の実機計測（`docs/perf/
    /// cuda-tensor-core-tolerance-opt-remeasurement.md`）側の責務であり、
    /// 本テストは「この診断入口が正しくルーティングを固定するか」のみを
    /// 検証する（数値判定はしない）。
    ///
    /// - `new_tf32_opt_only`: `wmma_tf32_staged_available() == false` かつ
    ///   `wmma_tf32_opt_available() == true` であることを assert する。
    /// - `new_tf32_basic_only`: 両方 `false`（staged・opt とも不能）かつ
    ///   `run_wmma_tf32` が 64×64×64 で `Ok` を返す（basic 経路が実際に
    ///   起動できる）ことを assert する。
    ///
    /// `#[ignore]`: 本セッションは NVRTC 非搭載のため実行できない。DGX
    /// Spark GB10 等の実機で `cargo test -p fandhe-ai-backend-cuda --lib
    /// --release --features internal-diagnostics -- --ignored --nocapture
    /// wmma_tf32_diagnostic_constructors_force_expected_kernel` から実行
    /// する。`internal-diagnostics` feature（既定 off）でのみコンパイル
    /// される（診断コンストラクタ自体が同 feature でゲートされているため）。
    #[cfg(feature = "internal-diagnostics")]
    #[test]
    #[ignore = "CUDA 実機（compute capability 8.0 以降）必須"]
    fn wmma_tf32_diagnostic_constructors_force_expected_kernel() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");

        let opt_only = CudaGemm::new_tf32_opt_only(&device)
            .expect("new_tf32_opt_only must succeed when the opt kernel is available");
        assert!(
            !opt_only.wmma_tf32_staged_available(),
            "new_tf32_opt_only must disable the staged path (routing must fall through to opt)"
        );
        assert!(
            opt_only.wmma_tf32_opt_available(),
            "new_tf32_opt_only must keep the opt kernel available"
        );

        let basic_only = CudaGemm::new_tf32_basic_only(&device)
            .expect("new_tf32_basic_only must succeed when the basic kernel is available");
        assert!(
            !basic_only.wmma_tf32_staged_available(),
            "new_tf32_basic_only must disable the staged path"
        );
        assert!(
            !basic_only.wmma_tf32_opt_available(),
            "new_tf32_basic_only must disable the opt path (routing must fall through to basic)"
        );
        basic_only
            .run_wmma_tf32(&[1.0f32; 64 * 64], &[1.0f32; 64 * 64], 64, 64, 64)
            .expect("run_wmma_tf32 must succeed via the forced basic path for a 64x64x64 shape");
    }
}
