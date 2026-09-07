//! `cp.async` 多段パイプライン（3〜4 stage）を導入した FP32 SIMT（Tensor
//! Core 不使用・通常の FMA 積和）GEMM の CUDA カーネルソース（イシュー
//! #1033・親イシュー #1031「FP32 SIMT GEMM 強化」）。
//!
//! # 位置づけ・本番結線（#1137）
//!
//! 本モジュールが生成するカーネルは、GB10 実測（bit 一致確認・parity 0
//! fail・`cuda_floor_bench` 性能 A/B）に基づき、イシュー #1137 で本番
//! 既定経路（`ops.rs::CudaBackendOps::gemm` → `context_cache::cached_gemm`
//! → `CudaGemm::run_tiled_f32`）へ**形状条件付きで結線済み**である
//! （`gemm.rs::CudaGemm::select_tiled_f32_kernel`・
//! `gemm.rs::tiled_f32_kernel_kind`）: cp.async 16 バイト整列制約
//! （`n % 4 == 0 && k % 4 == 0`）を満たし、かつ `CudaGemm::new` 時の
//! コンパイルに成功している場合にのみ本モジュールのカーネルへ分岐し、
//! それ以外（非整列形状・sm_80 未満・NVRTC コンパイル失敗環境）は常に
//! `kernels.rs::TILED_F32`（classic 版）へ fail-closed にフォールバック
//! する。詳細な実測記録・採否判断根拠は
//! `docs/perf/cuda-gemm-tiled-pipeline.md`「#1137 本番結線判断」節を正とする。
//!
//! 明示的に classic／pipeline いずれかを強制する入口
//! （[`CudaGemm::run_tiled_f32_classic`]／[`CudaGemm::run_tiled_pipeline_f32`]）
//! は診断・A/B ベンチ用途として引き続き残す。
//!
//! # `kernels::TILED_F32` との違い
//!
//! `TILED_F32`（`kernels.rs`）は 32×32 共有メモリタイル・1 スレッド 1
//! 要素・同期ロード（`__syncthreads()` を挟んでロード→計算を毎タイル
//! 直列化する）のみで、グローバルロードと演算がオーバーラップしない。
//! 本モジュールは `kernels_mma_tf32.rs`（TF32 `mma.sync` 経路。cp.async
//! 多段パイプライン・prologue 先行発行・`cp.async.wait_group STAGES-2`
//! の steady-state・境界外チャンクの `src_size=0` ゼロ充填）と同一の
//! パイプライン骨格を、Tensor Core を使わない通常の FMA 積和へ移植した
//! もの。差分は次の 2 点のみ:
//!
//! 1. **A/B フラグメントロードが `ldmatrix`/PTX レジスタロードではなく
//!    共有メモリへの直接インデックスアクセス**（`as_tile[stage][row][kk]`
//!    ／`bs_tile[stage][kk][col]`）である。
//! 2. **積和が `mma.sync` PTX 命令ではなく `fmaf()`**（NVRTC 組み込み。
//!    CPU 参照実装 `f32::mul_add` と同じ「明示的な融合積和」契約。
//!    `.claude/rules/coding-rust.md`「バックエンド構成（REQ-2）」の
//!    FMA 契約統一節）である。
//!
//! cp.async 段数管理（prologue の `STAGES-1` 先行発行・本体ループの
//! `wait_group (STAGES-2)`・ループ末尾の無条件 `commit_group` +
//! `__syncthreads()`・ループ外 `wait_group 0` の drain）は
//! `kernels_mma_tf32.rs::MMA_TF32_BODY` と同一の正しさ論証を踏襲する
//! （同ファイル該当コメント参照。本ファイルでは重複記載しない）。
//!
//! # タイル構成・レジスタブロッキング
//!
//! ブロックタイル `TP_BM x TP_BN`（64×64）・K タイル `TP_BK`（16）を
//! `TP_BLOCK_THREADS`（256 = 16×16 スレッド）で分担し、各スレッドが C の
//! `TP_THREAD_M x TP_THREAD_N`（4×4 = 16 要素）を担当する（外積型
//! レジスタブロッキング。CUTLASS の基本パターンと同型）。共有メモリは
//! A・B とも「元の行主導レイアウトのまま」保持する（`TILED_F32` と同じ
//! 転置なし方針）: `as_tile[STAGES][BM][A_PAD]`（各行が K 方向に連続）・
//! `bs_tile[STAGES][BK][B_PAD]`（各行が N 方向に連続）。A を転置格納
//! しないのは、`cp.async.cg.shared.global` の 1 回のコピー粒度（16 バイト
//! = f32 4 要素）が共有メモリ側でも連続であることを要求するため
//! （転置格納には A の 16 バイトチャンクを 4 つの異なる K 添字へ分散
//! 書き込みする必要があり cp.async の 1 命令では表現できない）。
//!
//! # 整列制約（cp.async 16 バイト境界）
//!
//! `gemm.rs` の起動前検証（`validate_tiled_pipeline_alignment` 相当。
//! `wmma_tf32_staged_alignment_ok` と同じ理由）が `n % 4 == 0 && k % 4 ==
//! 0` を要求する。A の行ストライドは `k`、B の行ストライドは `n` であり、
//! いずれも 4 の倍数でなければ 16 バイト境界をまたぐチャンクが生じうる
//! （`kernels_mma_tf32.rs` 冒頭コメント「整列制約」節と同じ論拠）。
//!
//! # REQ-8（カーネル境界検査規約。省略しない）
//!
//! 1. A/B タイルの `cp.async` ロードは範囲外チャンクで `src_size = 0` を
//!    渡しゼロ充填する（`LOAD_A_STAGE`/`LOAD_B_STAGE` マクロ）。列方向は
//!    f32 4 要素（16 バイト）境界へ切り下げてクランプする
//!    （`kernels_mma_tf32.rs::LOAD_A_STAGE_GROUP` と同一方式）。
//! 2. エピローグ store は要素ごとに `if (r < m && c < n)` の手動ガードを
//!    維持する（`#pragma unroll` によるレジスタブロッキング展開は演算・
//!    分岐命令数を削減する最適化であり、境界チェックそのものは無効化
//!    しないため REQ-8 の許容範囲内。`kernels.rs` 冒頭コメントの実例と
//!    同じ判断）。
//!
//! # Stream-K（最終 wave 限定・固定順序 fixup。イシュー #1358）
//!
//! 親イシュー #1347 が persistent タイルキュー版（K 分割なし）を REJECT
//! と確定したのを受け、末尾（最終 wave）の残タイルだけ K 反復を全 CTA へ
//! 平坦配布する Stream-K 版（`TP_SK_KERNEL_PREFIX`／`TP_SK_TILE_CORE`／
//! `TP_SK_FIXUP_KERNEL`）を opt-in で追加する。#812
//! （`docs/cuda-streamk-decision.md`）が保留した fixup 加算順序の非決定性
//! 懸念を「部分和は一意なスロットへ書く・fixup は寄与者昇順の固定順序で
//! 直列加算する」設計で解消する。本番既定経路（`CudaGemm::new`）は本
//! カーネル群を一切生成しない（`internal-diagnostics` feature 限定の
//! opt-in API。詳細は `TP_SK_FIXUP_KERNEL` ドキュメンテーションコメント
//! および `crate::gemm::streamk_plan`／`crate::gemm::
//! compile_tiled_pipeline_streamk_variant` を参照）。

use std::sync::LazyLock;

use crate::error::CudaError;

/// ブロックタイル M（C の行方向。64）。
pub const TP_BM: u32 = 64;
/// ブロックタイル N（C の列方向。64）。
pub const TP_BN: u32 = 64;
/// K タイル幅（16）。
pub const TP_BK: u32 = 16;

/// 1 スレッドが担当する C タイルの行数（4）。
pub const TP_THREAD_M: u32 = 4;
/// 1 スレッドが担当する C タイルの列数（4）。
pub const TP_THREAD_N: u32 = 4;

/// ブロック内スレッドグリッドの x 方向本数（`TP_BN / TP_THREAD_N` = 16）。
pub const TP_THREADS_X: u32 = TP_BN / TP_THREAD_N;
/// ブロック内スレッドグリッドの y 方向本数（`TP_BM / TP_THREAD_M` = 16）。
pub const TP_THREADS_Y: u32 = TP_BM / TP_THREAD_M;
/// ブロックあたりスレッド総数（256。カーネルは 1 次元ブロックとして
/// 起動し、`tx = tid % TP_THREADS_X`・`ty = tid / TP_THREADS_X` で 2 次元
/// 添字へ分解する。`kernels_mma_tf32.rs::MMA_TF32_BODY` の `tid`/
/// `num_threads` パターンを踏襲）。
pub const TP_BLOCK_THREADS: u32 = TP_THREADS_X * TP_THREADS_Y;

/// `cp.async` 多段パイプラインの既定ステージ数（3。`kernels_mma_tf32.rs::
/// MMA_TF32_STAGES` と同一値。実装計画 4 節）。
pub const TP_DEFAULT_STAGES: u32 = 3;

/// パイプラインステージ数として受理する最小値（`cp.async.wait_group
/// STAGES-2` の u32 アンダーフロー防止。`kernels_mma_tf32.rs` の
/// `MMA_TF32_STAGES >= 2` 契約と同一）。
pub const TP_MIN_STAGES: u32 = 2;

/// PTX ISA の `cp.async.wait_group` 即値オペランドの上限（0〜7）。
const MAX_WAIT_GROUP_IMMEDIATE: u32 = 7;

/// パイプラインステージ数として受理する最大値（実装計画のスコープ:
/// 3〜4 stage の比較。PTX 即値上限からは最大 9 まで許容できるが、本
/// イシューの受け入れ条件・ベンチ対象を 2〜4 に絞る）。
pub const TP_MAX_STAGES: u32 = 4;

/// A タイル（`as_tile[STAGES][BM][A_PAD]`）の行幅（パディング後）。
/// `BK + 4` 要素（cp.async 16B = f32 4 要素粒度の整列を保つ最小加算。
/// `kernels_mma_tf32.rs::MMA_TF32_A_PAD` と同一パディング方針）。
pub const TP_A_PAD: u32 = TP_BK + 4;
/// B タイル（`bs_tile[STAGES][BK][B_PAD]`）の行幅（パディング後）。
/// `BN + 4` 要素（同上方針）。
pub const TP_B_PAD: u32 = TP_BN + 4;

/// 1 ステージあたりの A タイルロードチャンク数（16 バイト = f32 4 要素
/// 単位。`(BM * BK) / 4`）。
pub const TP_A_CHUNKS: u32 = (TP_BM * TP_BK) / 4;
/// 1 ステージあたりの B タイルロードチャンク数（同上。`(BK * BN) / 4`）。
pub const TP_B_CHUNKS: u32 = (TP_BK * TP_BN) / 4;

/// ステージあたりの静的共有メモリ使用量（バイト）。
/// `(BM*A_PAD + BK*B_PAD) * 4B`。
pub const TP_SMEM_BYTES_PER_STAGE: u32 = (TP_BM * TP_A_PAD + TP_BK * TP_B_PAD) * 4;

// コンパイル時契約検査（`kernels_mma_tf32.rs` 冒頭の const assert 群と
// 同型。実機コンパイルできない環境でも `cargo build` の時点で機械検出
// できる代替チェック）。
const _: () = assert!(
    TP_BM.is_multiple_of(TP_THREAD_M),
    "TP_BM must be a multiple of TP_THREAD_M (per-thread register-blocked \
     output tile must evenly divide the block tile)"
);
const _: () = assert!(
    TP_BN.is_multiple_of(TP_THREAD_N),
    "TP_BN must be a multiple of TP_THREAD_N (per-thread register-blocked \
     output tile must evenly divide the block tile)"
);
const _: () = assert!(
    TP_BLOCK_THREADS <= 1024,
    "TP_BLOCK_THREADS must not exceed CUDA's per-block thread limit (1024)"
);
const _: () = assert!(
    TP_BM.is_multiple_of(4) && TP_BN.is_multiple_of(4) && TP_BK.is_multiple_of(4),
    "TP_BM/TP_BN/TP_BK must be multiples of 4 (cp.async 16-byte / f32 \
     4-element transfer granularity)"
);
const _: () = assert!(
    TP_A_PAD.is_multiple_of(4) && TP_B_PAD.is_multiple_of(4),
    "TP_A_PAD/TP_B_PAD must be multiples of 4 (cp.async 16-byte transfer \
     granularity / f32 element alignment)"
);
const _: () = assert!(
    !(TP_A_PAD * 4).is_multiple_of(128) && !(TP_B_PAD * 4).is_multiple_of(128),
    "TP_A_PAD/TP_B_PAD row stride in bytes must not be a multiple of 128B \
     (32 banks x 4B) or bank-phase padding degenerates to no-op"
);
const _: () = assert!(
    TP_A_CHUNKS * 4 == TP_BM * TP_BK && TP_B_CHUNKS * 4 == TP_BK * TP_BN,
    "TP_BM*TP_BK / TP_BK*TP_BN must be exact multiples of 4 (each cp.async \
     chunk transfers exactly 4 f32 elements; TP_A_CHUNKS/TP_B_CHUNKS must \
     not truncate)"
);
const _: () = assert!(
    TP_MIN_STAGES >= 2,
    "kernels_tiled_pipeline の cp.async パイプラインは STAGES >= 2 を前提と \
     する（カーネルソース側の `STAGES - 2` 計算が u32 でアンダーフロー \
     しないため）"
);
const _: () = assert!(
    TP_MAX_STAGES >= TP_MIN_STAGES && TP_MAX_STAGES <= MAX_WAIT_GROUP_IMMEDIATE + 2,
    "TP_MAX_STAGES must fit the cp.async.wait_group immediate operand range \
     (STAGES - 2 must be in [0, 7])"
);
const _: () = assert!(
    TP_DEFAULT_STAGES >= TP_MIN_STAGES && TP_DEFAULT_STAGES <= TP_MAX_STAGES,
    "TP_DEFAULT_STAGES must lie within [TP_MIN_STAGES, TP_MAX_STAGES]"
);
// 静的共有メモリ予算（全 compute capability 共通の per-block 48KiB）は
// 最悪ケース（TP_MAX_STAGES=4）でも超過しないことをコンパイル時に検査
// する。段数を増やすほど所要量は単調増加するため、この 1 点の検査で
// TP_MIN_STAGES..=TP_MAX_STAGES の全段数を保証できる。
const _: () = assert!(
    TP_SMEM_BYTES_PER_STAGE * TP_MAX_STAGES <= crate::kernels_mma::MMA_STATIC_SMEM_LIMIT_BYTES,
    "kernels_tiled_pipeline static shared memory (at TP_MAX_STAGES) exceeds \
     the 48KiB per-block limit shared by every compute capability"
);

/// 本番結線（[`crate::gemm::CudaGemm::new`]）が既定でコンパイルする
/// ステージ数（[`TP_DEFAULT_STAGES`]）固定のカーネルソース。
///
/// カーネルソースはコンパイル時定数のみから `format!` で組み立て、外部
/// 入力文字列を連結しない（`nvrtc.rs` A03 節と同じ契約。
/// `.claude/rules/security.md` A03）。
pub fn tiled_pipeline_f32_source() -> &'static str {
    &TILED_PIPELINE_F32_SOURCE
}

static TILED_PIPELINE_F32_SOURCE: LazyLock<String> =
    LazyLock::new(|| render_source(TP_DEFAULT_STAGES));

/// 任意のステージ数（[`TP_MIN_STAGES`]..=[`TP_MAX_STAGES`]）のカーネル
/// ソースを生成する。本番結線は既定ステージ数固定の
/// [`tiled_pipeline_f32_source`] のみを使い、本関数はベンチ example
/// （`examples/gemm_tiled_pipeline_bench.rs`）が 3 vs 4 stage を比較する
/// ためにオンデマンドで呼ぶ（実装計画 §5「stages=4 版はベンチ用途に限り
/// オンデマンドでコンパイルする」）。
pub fn tiled_pipeline_f32_source_with_stages(stages: u32) -> Result<String, CudaError> {
    if !(TP_MIN_STAGES..=TP_MAX_STAGES).contains(&stages) {
        return Err(CudaError::InvalidKernelConfig {
            detail: format!(
                "tiled_pipeline_f32_source_with_stages stages ({stages}) must lie within \
                 [{TP_MIN_STAGES}, {TP_MAX_STAGES}]"
            ),
        });
    }
    Ok(render_source(stages))
}

/// [`render_source`]／[`render_persistent_source`] 共通の `#define` 群を
/// 生成する（イシュー #1346 でカーネル本体テンプレートを非 persistent／
/// persistent の 2 系統へ分離したのに伴い、従来 `render_source` 内に
/// あった `format!` 冒頭部分を共有ヘルパーへ切り出した）。
fn render_defines(stages: u32) -> String {
    format!(
        "\n#define TP_BM {bm}\n\
         #define TP_BN {bn}\n\
         #define TP_BK {bk}\n\
         #define TP_THREAD_M {thread_m}\n\
         #define TP_THREAD_N {thread_n}\n\
         #define TP_THREADS_X {threads_x}\n\
         #define TP_A_PAD {a_pad}\n\
         #define TP_B_PAD {b_pad}\n\
         #define TP_STAGES {stages}\n\
         \n",
        bm = TP_BM,
        bn = TP_BN,
        bk = TP_BK,
        thread_m = TP_THREAD_M,
        thread_n = TP_THREAD_N,
        threads_x = TP_THREADS_X,
        a_pad = TP_A_PAD,
        b_pad = TP_B_PAD,
        stages = stages,
    )
}

/// 非 persistent 版（既存 `gemm_tiled_pipeline_f32`）のソース全文を生成
/// する。[`TP_CP_ASYNC_HELPER`]・[`TP_NON_PERSISTENT_PREFIX`]・
/// [`TP_TILE_CORE`]・[`TP_KERNEL_SUFFIX`] の連結が、分割前の
/// `TILED_PIPELINE_F32_BODY`（旧単一定数）とバイト同一であることは実装
/// 時に機械検証済み（イシュー #1346 AC-3。分割は文字列の切り出し・
/// 再結合のみで内容を変更していない）。
fn render_source(stages: u32) -> String {
    format!(
        "{defines}{helper}{prefix}{core}{suffix}",
        defines = render_defines(stages),
        helper = TP_CP_ASYNC_HELPER,
        prefix = TP_NON_PERSISTENT_PREFIX,
        core = TP_TILE_CORE,
        suffix = TP_KERNEL_SUFFIX,
    )
}

/// persistent 版（`gemm_tiled_pipeline_persistent_f32`。イシュー #1346）
/// のソース全文を生成する。[`render_source`] と同じ `#define` 群・同じ
/// [`TP_CP_ASYNC_HELPER`]／[`TP_TILE_CORE`] を共有し、CTA→出力タイルの
/// 割り当て部分（[`TP_KERNEL_PERSISTENT_PREFIX`]／
/// [`TP_KERNEL_PERSISTENT_SUFFIX`]）のみが異なる。
fn render_persistent_source(stages: u32) -> String {
    format!(
        "{defines}{helper}{prefix}{core}{suffix}",
        defines = render_defines(stages),
        helper = TP_CP_ASYNC_HELPER,
        prefix = TP_KERNEL_PERSISTENT_PREFIX,
        core = TP_TILE_CORE,
        suffix = TP_KERNEL_PERSISTENT_SUFFIX,
    )
}

/// 本番結線（[`crate::gemm::CudaGemm::new`]）が既定でコンパイルする
/// ステージ数（[`TP_DEFAULT_STAGES`]）固定の persistent 版カーネルソース。
/// `internal-diagnostics` feature 限定の opt-in API
/// （[`crate::gemm::CudaGemm::compile_tiled_pipeline_persistent_variant`]）
/// からのみ呼ばれる（`new` 自体は本番既定経路のためコンパイルしない。
/// [`TP_KERNEL_PERSISTENT_PREFIX`] ドキュメンテーションコメント「位置
/// づけ・非結線」参照）。
pub fn tiled_pipeline_persistent_f32_source() -> &'static str {
    &TILED_PIPELINE_PERSISTENT_F32_SOURCE
}

static TILED_PIPELINE_PERSISTENT_F32_SOURCE: LazyLock<String> =
    LazyLock::new(|| render_persistent_source(TP_DEFAULT_STAGES));

/// 任意のステージ数（[`TP_MIN_STAGES`]..=[`TP_MAX_STAGES`]）の persistent
/// 版カーネルソースを生成する（[`tiled_pipeline_f32_source_with_stages`]
/// の persistent 版。`examples/gemm_tiled_pipeline_persistent_bench.rs`
/// が段数比較のためオンデマンドで呼ぶ）。
pub fn tiled_pipeline_persistent_f32_source_with_stages(stages: u32) -> Result<String, CudaError> {
    if !(TP_MIN_STAGES..=TP_MAX_STAGES).contains(&stages) {
        return Err(CudaError::InvalidKernelConfig {
            detail: format!(
                "tiled_pipeline_persistent_f32_source_with_stages stages ({stages}) must lie \
                 within [{TP_MIN_STAGES}, {TP_MAX_STAGES}]"
            ),
        });
    }
    Ok(render_persistent_source(stages))
}

/// [`render_source`] が結合するカーネル本体テンプレート。
///
/// `TP_STAGES` は `format!` で埋め込まれる `#define` のみに依存し、本体
/// 文字列自体はステージ数に非依存（配列サイズ・`STAGES - 2` 等の算術は
/// すべて `TP_STAGES` マクロ経由）。
/// 非 persistent 版・persistent 版共有の `cp.async` 16 バイト転送
/// ヘルパー（`tp_cp_async16`）。NVRTC は 1 コンパイル単位に両カーネル
/// を同時に含めうるため関数定義は 1 箇所のみ生成する（
/// [`render_source`]・[`render_persistent_source`] のどちらも本定数を
/// 連結する）。
const TP_CP_ASYNC_HELPER: &str = r#"
// REQ-8: グローバル→共有メモリの 16 バイト単位（f32 4 要素）非同期
// コピー。src_size==16 で実データをコピーし、src_size==0 で共有メモリ側を
// ゼロ充填する（kernels_mma_tf32.rs::mma_tf32_cp_async16 と同じ契約・
// 同じ PTX 命令。関数名は同一 NVRTC コンパイル単位内での衝突を避けるため
// 本カーネル専用の接頭辞を付す）。
__device__ __forceinline__ void tp_cp_async16(void* smem_ptr, const void* gmem_ptr, int src_size)
{
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr), "r"(src_size)
    );
}

"#;

/// 非 persistent 版（既存 `gemm_tiled_pipeline_f32`）の関数シグネ
/// チャ・共有メモリ宣言・`blockIdx` 由来の `block_row0`/`block_col0`
/// 計算。直後（[`TP_TILE_CORE`]）は `block_row0`/`block_col0` が
/// スコープに存在することのみを前提にする（persistent 版は本断片を
/// 使わず、[`TP_KERNEL_PERSISTENT_PREFIX`] が同じ 2 変数を別の方法
/// 〈タイルキュー〉で計算してから同じ [`TP_TILE_CORE`] を合流させる）。
const TP_NON_PERSISTENT_PREFIX: &str = r#"extern "C" __global__ void gemm_tiled_pipeline_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k)
{
    // __align__(16): cp.async の 16 バイト転送先整列要件。A_PAD/B_PAD が
    // 4 要素の倍数のため各行の先頭は常に 16 バイト整列する（転置なし
    // レイアウト。本ファイル冒頭コメント「タイル構成」参照）。
    __shared__ __align__(16) float as_tile[TP_STAGES][TP_BM][TP_A_PAD];
    __shared__ __align__(16) float bs_tile[TP_STAGES][TP_BK][TP_B_PAD];

    int block_row0 = blockIdx.y * TP_BM;
    int block_col0 = blockIdx.x * TP_BN;

"#;

/// [`render_source`]／[`render_persistent_source`] が共有するタイル内
/// 計算本体（プロローグ→K ループ→drain→エピローグ guarded store）。
/// `block_row0`/`block_col0`（呼び出し元プレフィックスが計算済み）・
/// `m`/`n`/`k`/`a`/`b`/`c` のみに依存し、CTA→出力タイルの割り当て方法
/// には依存しない。非 persistent 版・persistent 版のいずれでも同一の
/// 命令列になることが bit 同一の根拠（[`TP_KERNEL_PERSISTENT_PREFIX`]
/// ドキュメンテーションコメント「bit 同一の根拠」節参照）。
const TP_TILE_CORE: &str = r#"    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int tx = tid % TP_THREADS_X;
    int ty = tid / TP_THREADS_X;

    int thread_row0 = block_row0 + ty * TP_THREAD_M;
    int thread_col0 = block_col0 + tx * TP_THREAD_N;

    float acc[TP_THREAD_M][TP_THREAD_N] = {};

    int num_k_tiles = (k > 0) ? (k - 1) / TP_BK + 1 : 0;

    #define A_CHUNKS ((TP_BM * TP_BK) / 4)
    #define B_CHUNKS ((TP_BK * TP_BN) / 4)

    // REQ-8: 境界外チャンクでも 16 バイト整列を保ったままクランプする
    // （列方向は f32 4 要素境界へ切り下げ。`gemm.rs` 側の起動前整列検証
    // 〈n%4==0 && k%4==0〉と合わせて行ストライドの 4 要素倍数性を保証
    // する。`kernels_mma_tf32.rs::LOAD_A_STAGE_GROUP` と同一式）。
    #define LOAD_A_STAGE(stage, k0) \
        for (int idx = tid; idx < A_CHUNKS; idx += num_threads) { \
            int row = idx / (TP_BK / 4); \
            int col0 = (idx % (TP_BK / 4)) * 4; \
            int gr = block_row0 + row; \
            int gc = (k0) + col0; \
            int gr_c = gr < m ? gr : (m > 0 ? m - 1 : 0); \
            int gc_c = gc < k ? gc : (k > 0 ? ((k - 1) / 4) * 4 : 0); \
            int valid = (gr < m && gc < k) ? 16 : 0; \
            tp_cp_async16(&as_tile[stage][row][col0], &a[(size_t)gr_c * k + gc_c], valid); \
        }

    #define LOAD_B_STAGE(stage, k0) \
        for (int idx = tid; idx < B_CHUNKS; idx += num_threads) { \
            int row = idx / (TP_BN / 4); \
            int col0 = (idx % (TP_BN / 4)) * 4; \
            int gr = (k0) + row; \
            int gc = block_col0 + col0; \
            int gr_c = gr < k ? gr : (k > 0 ? k - 1 : 0); \
            int gc_c = gc < n ? gc : (n > 0 ? ((n - 1) / 4) * 4 : 0); \
            int valid = (gr < k && gc < n) ? 16 : 0; \
            tp_cp_async16(&bs_tile[stage][row][col0], &b[(size_t)gr_c * n + gc_c], valid); \
        }

    // プロローグ: kernels_mma_tf32.rs::MMA_TF32_BODY プロローグと同一の
    // 「1 イテレーション = 必ず 1 commit」不変条件。
    for (int s = 0; s < TP_STAGES - 1; ++s) {
        if (s < num_k_tiles) {
            LOAD_A_STAGE(s, s * TP_BK);
            LOAD_B_STAGE(s, s * TP_BK);
        }
        asm volatile("cp.async.commit_group;\n");
    }

    for (int t = 0; t < num_k_tiles; ++t) {
        int compute_stage = t % TP_STAGES;
        int next_tile = t + TP_STAGES - 1;
        int load_stage = next_tile % TP_STAGES;

        // kernels_mma_tf32.rs::MMA_TF32_BODY と同一の段数一般形固定即値
        // （`STAGES - 2`）・同一の正しさ論証（非負性は上記
        // `TP_MIN_STAGES >= 2` のコンパイル時 assert が担保する）。
        asm volatile("cp.async.wait_group %0;\n" ::"n"(TP_STAGES - 2));
        __syncthreads();

        // compute_stage の共有メモリタイルを使い、TP_THREAD_M x
        // TP_THREAD_N の外積型レジスタブロッキングで積和する。CPU 参照
        // 実装（f32::mul_add）と同じ「明示的な融合積和」契約を保つため
        // fmaf() を使う（`.claude/rules/coding-rust.md`「バックエンド
        // 構成（REQ-2）」の FMA 契約統一節）。
#pragma unroll
        for (int kk = 0; kk < TP_BK; ++kk) {
            float a_reg[TP_THREAD_M];
#pragma unroll
            for (int i = 0; i < TP_THREAD_M; ++i) {
                a_reg[i] = as_tile[compute_stage][ty * TP_THREAD_M + i][kk];
            }
            float b_reg[TP_THREAD_N];
#pragma unroll
            for (int j = 0; j < TP_THREAD_N; ++j) {
                b_reg[j] = bs_tile[compute_stage][kk][tx * TP_THREAD_N + j];
            }
#pragma unroll
            for (int i = 0; i < TP_THREAD_M; ++i) {
#pragma unroll
                for (int j = 0; j < TP_THREAD_N; ++j) {
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }

        // 次タイル（load_stage）の cp.async 発行は本イテレーションの
        // compute_stage 読み取り（上記ループ、mma.sync ではなく共有メモリ
        // 直接アクセス）の後に置く。load_stage != compute_stage
        // （TP_STAGES >= 2 のため next_tile = t + STAGES - 1 は t と
        // mod STAGES で一致しない）であり、異なる物理バッファへの書き込み
        // のため上記読み取りとは競合しない。
        if (next_tile < num_k_tiles) {
            LOAD_A_STAGE(load_stage, next_tile * TP_BK);
            LOAD_B_STAGE(load_stage, next_tile * TP_BK);
        }

        // kernels_mma_tf32.rs::MMA_TF32_BODY と同一の「1 イテレーション =
        // 必ず 1 commit」不変条件、および同一の syncthreads 配置（次
        // イテレーションが同じ物理ステージバッファを再利用する前に、
        // 全スレッドが本イテレーションの compute_stage 読み取りを終えた
        // ことを保証する WAR 安全性の論証。`kernels_mma_tf32.rs` 該当
        // コメント参照）。
        asm volatile("cp.async.commit_group;\n");
        __syncthreads();
    }

    // ループ外 drain（kernels_mma_tf32.rs::MMA_TF32_BODY と同一の正しさ
    // 論証）。
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    #undef LOAD_A_STAGE
    #undef LOAD_B_STAGE
    #undef A_CHUNKS
    #undef B_CHUNKS

    // REQ-8: エピローグの guarded store。`#pragma unroll` によるループ
    // 展開は演算・分岐命令数を削減する最適化であり、境界チェックそのもの
    // は無効化しない（`kernels.rs` 冒頭コメントの実例と同じ判断）。
#pragma unroll
    for (int i = 0; i < TP_THREAD_M; ++i) {
#pragma unroll
        for (int j = 0; j < TP_THREAD_N; ++j) {
            int r = thread_row0 + i;
            int cc = thread_col0 + j;
            if (r < m && cc < n) {
                c[(size_t)r * n + cc] = acc[i][j];
            }
        }
    }
"#;

/// 非 persistent 版カーネル関数の閉じ括弧（`}}`）。
const TP_KERNEL_SUFFIX: &str = "}\n";

/// persistent 版（イシュー #1346）カーネルの関数シグネチャ・共有
/// メモリ宣言・グローバル atomic タイルキューからの `block_row0`/
/// `block_col0` 取得ループ。
///
/// # 位置づけ・非結線
///
/// 本カーネルは grid を [`crate::device::CudaDevice::multiprocessor_count`]
/// 実測 SM 数 × 占有可能 block 数（またはホスト指定値）で固定起動し、
/// 各 CTA が完了するたびにグローバル `unsigned int` カウンタ
/// （`tile_counter`）へ `atomicAdd` して次の未処理出力タイルを取得する
/// （K 分割なし・1 タイル = 1 CTA が K 全体を担当）。GB10（sm_121・
/// SM 48 基）でタイル数が SM 数の倍数から外れる形状（wave
/// quantization。`docs/cuda-streamk-decision.md` §2）において、末尾
/// wave の CTA 数不足による GPU 遊休時間を緩和しうる候補として追加
/// する（親イシュー #1345・本イシュー #1346）。**本番既定経路
/// （[`crate::gemm::CudaGemm::new`]）は不変**であり、本カーネルは
/// `internal-diagnostics` feature 限定の opt-in API
/// （[`crate::gemm::CudaGemm::compile_tiled_pipeline_persistent_variant`]）
/// からのみコンパイルされる。実機性能比較・本番結線可否判断は兄弟
/// イシュー #1347 が担う。
///
/// # bit 同一の根拠
///
/// タイル内の計算（[`TP_TILE_CORE`]）は非 persistent 版
/// （[`TP_NON_PERSISTENT_PREFIX`]）と完全に共有する文字列であり、
/// smem レイアウト・cp.async ロード順・K ループの `fmaf` 蓄積順序・
/// エピローグの guarded store はいずれも同一の命令列になる。
/// 「出力タイル→CTA」の割り当て方法（`blockIdx` 直接 対 atomic タイル
/// キュー）のみが異なり、これは各出力タイルの**どの CTA が計算するか**
/// を変えるだけで**各要素がどう計算されるか**は変えない。タイルキュー
/// 用の `atomicAdd` は `unsigned int` カウンタ（スケジューリング専用）
/// にのみ作用し、GEMM の数値蓄積（`float` の `acc[][]`）には一切
/// 触れない（`.claude/rules/coding-rust.md` の FMA 契約・
/// `kernels_mse.rs`/`kernels_rmsnorm.rs` が禁止する「float `atomicAdd`
/// による非決定的な結合順序」とは別種の atomic であり、決定性契約を
/// 破らない）。よって同一入力に対し persistent 版と非 persistent 版は
/// 出力 bit 同一になる（`tests/cpu_cuda_tiled_pipeline_persistent_
/// parity.rs` が実機でこれを検証する）。
///
/// # 不変条件
///
/// - `break` はブロック一様: 全スレッドが `__syncthreads()` の後に同じ
///   `s_tile` を読んでから判定する（一部スレッドだけが抜けるダイバー
///   ジェンスは起きない）。
/// - `s_tile` への WAR（次イテレーションの thread-0 書き込みと前イテ
///   レーションの全スレッド読み出し）は [`TP_TILE_CORE`] 末尾の drain
///   `__syncthreads()`（cp.async 完了待ちと共用）が前イテレーション
///   の読み出しを完了させてから thread 0 が次の `atomicAdd` を発行
///   する順序で安全になる。
/// - `acc[][]` は [`TP_TILE_CORE`] 内で毎回宣言される自動変数のため
///   タイルごとに再初期化される。
/// - cp.async の `wait_group`/`commit_group` 会計は各タイルの
///   [`TP_TILE_CORE`] 内で完結する（drain `wait_group 0` により次
///   タイル開始時点で未完了の cp.async グループが残らない）ため、
///   タイル境界をまたいでも会計が破綻しない。
const TP_KERNEL_PERSISTENT_PREFIX: &str = r#"extern "C" __global__ void gemm_tiled_pipeline_persistent_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k,
    unsigned int* tile_counter)
{
    // __align__(16): 非 persistent 版（TP_NON_PERSISTENT_PREFIX）と
    // 同一の共有メモリ整列要件。
    __shared__ __align__(16) float as_tile[TP_STAGES][TP_BM][TP_A_PAD];
    __shared__ __align__(16) float bs_tile[TP_STAGES][TP_BK][TP_B_PAD];
    // タイル取得カウンタの読み出し結果を全スレッドへ配る 1 スロット
    // （thread 0 のみが atomicAdd を発行し、__syncthreads() の後に
    // 全スレッドが同じ値を読む。ブロック一様な break 判定のため）。
    __shared__ unsigned int s_tile;

    // 出力タイル総数（行主導: tile = row_tile * tiles_x + col_tile）。
    // 非 persistent 版の blockIdx.y/blockIdx.x と同じ走査順
    // （grid x = n 方向）に対応させ、tiled_pipeline_launch_config が
    // 仮定するタイル→座標写像と一致させる。
    int tiles_x = (n - 1) / TP_BN + 1;
    int tiles_y = (m - 1) / TP_BM + 1;
    int num_tiles = tiles_x * tiles_y;

    // persistent CTA ループ: grid をブロックタイル数より少なく起動し
    // （ホスト側 persistent_grid_blocks）、各 CTA が完了するたびに
    // 次の未処理タイルをグローバル atomic カウンタから取得する
    // （wave quantization 緩和が目的。docs/cuda-streamk-decision.md
    // §2）。K 分割は行わない（1 タイル = 1 CTA が K 全体を担当）ため
    // 蓄積順序・fmaf 命令列は非 persistent 版と同一であり、出力は
    // bit 同一になる（本ファイル冒頭コメント「persistent 版」節）。
    for (;;) {
        // カウンタの発行はスレッド 0 のみ（複数スレッドが同時に
        // atomicAdd すると同一 CTA 内で異なるタイルを取り合い、後続
        // の共有メモリロード・計算がタイル不整合になるため）。
        if (threadIdx.x == 0) {
            s_tile = atomicAdd(tile_counter, 1u);
        }
        // ブロック全体が同じ s_tile を読むまで待つ（thread 0 の書き
        // 込みと他スレッドの読み出しの happens-before を保証。この
        // barrier は各タイル末尾の drain barrier（TP_TILE_CORE）と
        // 対になり、s_tile への WAR（次イテレーションの書き込み対
        // 前イテレーションの読み出し）を安全にする）。
        __syncthreads();
        unsigned int tile_u = s_tile;
        // tile_u は unsigned のまま num_tiles と比較する（int→unsigned
        // 昇格）。カウンタは全 CTA 分の余剰 atomicAdd により num_tiles
        // を超えて進み続けるため、先に int へキャストすると理論上の
        // 桁あふれで負値化し比較をすり抜けうる（REQ-8・fail-closed。
        // 比較後にのみ int へ変換する）。
        if (tile_u >= (unsigned int)num_tiles) {
            break;
        }
        int tile = (int)tile_u;
        // 非 persistent 版の block_row0 = blockIdx.y * TP_BM・
        // block_col0 = blockIdx.x * TP_BN（grid x = n 方向）と同じ
        // 走査順写像（tiled_pipeline_launch_config 参照）。
        int block_row0 = (tile / tiles_x) * TP_BM;
        int block_col0 = (tile % tiles_x) * TP_BN;
"#;

/// persistent 版の `for (;;)` ループ・関数の閉じ括弧（[`TP_TILE_CORE`]
/// の drain barrier 直後にタイル取得ループの `}` を、続けて関数の `}`
/// を閉じる。非 persistent 版の [`TP_KERNEL_SUFFIX`] とは異なる文字列
/// になる）。
const TP_KERNEL_PERSISTENT_SUFFIX: &str = "    }\n}\n";

// =====================================================================
// Stream-K（最終 wave 限定・固定順序 fixup。イシュー #1358）
// =====================================================================
//
// 親イシュー #1357 の実測（K 分割なし persistent 化のみでは末尾 wave の
// 遊休を縮められない。#1347 REJECT）を受け、末尾（最終 wave）の残タイル
// （[`persistent_grid_blocks`] の容量 `G` に満たない端数タイル `R`）だけ
// K 反復を全 CTA へ平坦配布する opt-in 版。#812
// （`docs/cuda-streamk-decision.md`）が保留した「fixup の加算順序が
// 非決定的になりうる」懸念を、(1) 部分和は `(タイル, 寄与者)` ごとに
// 一意なスロットへ書く、(2) fixup は寄与者昇順の固定順序で直列加算する、
// という設計で解消する（詳細な決定性論証は下記 [`TP_SK_FIXUP_KERNEL`]
// ドキュメンテーションコメント末尾を参照）。
//
// 配布計画（GPU 不要のホスト側純関数 [`crate::gemm::streamk_plan`]）が
// 出力タイル総数 `T`・grid 容量 `G`・K タイル数 `nk` から
// `full_tiles`（先頭 `F` 個。1 CTA が K 全体を担当する従来どおりの
// タイル）・`remainder_tiles`（末尾 `R` 個。K 反復を `q` 幅ずつ `U` 個の
// 「SK 単位」へ平坦配布するタイル）・`q`・`max_contributors` を導出する。
// 本モジュールのカーネルはこの計画をホストから受け取って実行するのみで、
// 計画そのものの正しさ・網羅性は `gemm.rs` 側のホストシミュレータ
// テストが担保する（GPU 不要のため通常 CI で検証可能）。

/// `crate::gemm::CudaGemm::compile_tiled_pipeline_streamk_variant`
/// （`internal-diagnostics` feature 限定）からコンパイルされる Stream-K
/// 版 GEMM カーネルの関数シグネチャ・共有メモリ宣言・タイル取得ループ・
/// 単位復号（最大 2 個のサブレンジへの分解。本ファイル冒頭コメント
/// 参照）を生成する。
///
/// # 引数の位置づけ
///
/// - `unit_counter`: persistent 版の `tile_counter` と同じ役割（起動
///   ごとにホスト側でゼロ化する `unsigned int` カウンタ 1 個。
///   スケジューリング専用で GEMM の数値蓄積には触れない）。
/// - `partials`: 部分和バッファ（`[slot][TP_BM*TP_BN]` 形状。ホスト側
///   確保サイズは `3 * grid_capacity * TP_BM * TP_BN` floats固定。
///   `crate::gemm::StreamKPlan` ドキュメンテーションコメント参照）。
/// - `full_tiles`／`total_units`／`q`／`max_contributors`／
///   `remainder_tiles`: いずれも [`crate::gemm::streamk_plan`] が算出する
///   計画値をそのままホストから渡す（本カーネル自身は計画のロジックを
///   持たず、計画に従って復号するのみ）。
/// - `partials_capacity`: `partials` バッファの要素数（呼び出し元が
///   確保したハンドルの `CudaSlice::len()`。REQ-8・fail-closed）。ホスト側
///   （`streamk_plan`・起動前検査）がスロット添字は容量内であることを
///   証明済みだが、性能下限・最適化を理由に手動境界チェックを省略しない
///   （`.claude/rules/coding-rust.md`）ためカーネル側でも `partials` への
///   書き込み直前に `slot * (TP_BM*TP_BN) + local` を本値と突き合わせる
///   （[`TP_SK_TILE_CORE`] 参照）。
const TP_SK_KERNEL_PREFIX: &str = r#"extern "C" __global__ void gemm_tiled_pipeline_streamk_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k,
    unsigned int* unit_counter,
    float* __restrict__ partials,
    int full_tiles, int total_units, int q, int max_contributors,
    int remainder_tiles, int partials_capacity)
{
    __shared__ __align__(16) float as_tile[TP_STAGES][TP_BM][TP_A_PAD];
    __shared__ __align__(16) float bs_tile[TP_STAGES][TP_BK][TP_B_PAD];
    __shared__ unsigned int s_unit;

    int tiles_x = (n - 1) / TP_BN + 1;
    int num_k_tiles = (k > 0) ? (k - 1) / TP_BK + 1 : 0;

    for (;;) {
        if (threadIdx.x == 0) {
            s_unit = atomicAdd(unit_counter, 1u);
        }
        __syncthreads();
        unsigned int unit_u = s_unit;
        if (unit_u >= (unsigned int)total_units) {
            break;
        }
        int unit = (int)unit_u;

        // 1 単位は最大 2 個の「サブレンジ」（タイル境界をまたぐ場合の
        // 前半・後半）に分解される（`q < nk` の active な構成では単位長
        // `q` がタイル幅 `nk` を超えないため。本ファイル冒頭コメント
        // 参照）。full タイル単位（`unit < full_tiles`）は常に 1 個の
        // サブレンジ（K 全体）。
        int sub_count = 0;
        int sub_tile[2];
        int sub_kt_begin[2];
        int sub_kt_end[2];
        int sub_store_to_c[2];
        int sub_slot[2];

        if (unit < full_tiles) {
            sub_count = 1;
            sub_tile[0] = unit;
            sub_kt_begin[0] = 0;
            sub_kt_end[0] = num_k_tiles;
            sub_store_to_c[0] = 1;
            sub_slot[0] = 0;
        } else {
            // 平坦添字空間 [0, remainder_tiles*num_k_tiles) を幅 q の
            // 連続範囲へ分割した SK 単位の復号（`crate::gemm::
            // streamk_plan` ドキュメンテーションコメント §3.1 と同じ
            // 整数演算。ホストシミュレータテストが同じ式で全単位の
            // 網羅性・一意性を検証する）。
            long long u_prime = (long long)(unit - full_tiles);
            long long flat_begin = u_prime * (long long)q;
            long long total_flat = (long long)remainder_tiles * (long long)num_k_tiles;
            long long flat_end = flat_begin + (long long)q;
            if (flat_end > total_flat) {
                flat_end = total_flat;
            }
            long long flat_pos = flat_begin;
            while (flat_pos < flat_end && sub_count < 2) {
                int r = (int)(flat_pos / (long long)num_k_tiles);
                long long tile_flat_end = (long long)(r + 1) * (long long)num_k_tiles;
                long long sub_end = flat_end < tile_flat_end ? flat_end : tile_flat_end;
                long long tile_flat_begin = (long long)r * (long long)num_k_tiles;
                int kt_begin = (int)(flat_pos - tile_flat_begin);
                int kt_end = (int)(sub_end - tile_flat_begin);
                long long u_first = tile_flat_begin / (long long)q;
                int c = (int)(u_prime - u_first);

                sub_tile[sub_count] = full_tiles + r;
                sub_kt_begin[sub_count] = kt_begin;
                sub_kt_end[sub_count] = kt_end;
                sub_store_to_c[sub_count] = 0;
                sub_slot[sub_count] = r * max_contributors + c;
                sub_count += 1;
                flat_pos = sub_end;
            }
        }

        for (int sub = 0; sub < sub_count; ++sub) {
            int tile = sub_tile[sub];
            int kt_begin = sub_kt_begin[sub];
            int kt_end = sub_kt_end[sub];
            int store_to_c = sub_store_to_c[sub];
            int slot = sub_slot[sub];
            int block_row0 = (tile / tiles_x) * TP_BM;
            int block_col0 = (tile % tiles_x) * TP_BN;
"#;

/// [`TP_SK_KERNEL_PREFIX`] の `for (int sub ...)` ループ本体（1 サブ
/// レンジ分の K 範囲 `[kt_begin, kt_end)` に対するタイル内計算）。
///
/// [`crate::kernels_tiled_pipeline::TP_TILE_CORE`]
/// （非 persistent・persistent 版共有のタイル内計算）から派生した K
/// 範囲付き版。プロローグ・本体ループの段数会計（`commit_group`／
/// `wait_group` の配置・`__syncthreads()` の位置）は `TP_TILE_CORE` と
/// 同一の正しさ論証をそのまま踏襲するが、インデックスを絶対タイル番号
/// `t` ではなく `kt_begin` からのローカルオフセット `tl` で管理する
/// （K 全体を担当する full タイル単位では `kt_begin == 0` のため
/// `tl == t` と一致し、`TP_TILE_CORE` と完全に同一の命令列になる——これが
/// 「full タイル領域は非 Stream-K 版と bit 同一」という受け入れ条件の
/// 根拠。`LOAD_A_STAGE`／`LOAD_B_STAGE` マクロ・`fmaf` 内積ループは
/// `TP_TILE_CORE` と文字列として同一——`kt_begin + s`／`kt_begin + tl`
/// を通じて絶対 K タイル番号を渡すのみで、マクロ自体の本文は変更しない）。
///
/// エピローグは `store_to_c` により分岐する:
/// - `store_to_c != 0`（full タイル単位）: `TP_TILE_CORE` と同じ guarded
///   store（`if (r < m && cc < n)`）で `c` へ直接書く。
/// - `store_to_c == 0`（残タイルのサブレンジ単位）: `partials[slot *
///   (TP_BM*TP_BN) + local]` へタイル内ローカル添字で**無条件**に書く
///   （スロットは `crate::gemm::streamk_plan` が一意に割り当てるため、
///   同一スロットへの書き手は本サブレンジのみ。境界外要素は cp.async の
///   ゼロ充填ロード由来の 0 が書かれるだけで、[`TP_SK_FIXUP_KERNEL`] の
///   guarded store が読み捨てる）。
const TP_SK_TILE_CORE: &str = r#"            int tid = threadIdx.x;
            int num_threads = blockDim.x;
            int tx = tid % TP_THREADS_X;
            int ty = tid / TP_THREADS_X;

            int thread_row0 = block_row0 + ty * TP_THREAD_M;
            int thread_col0 = block_col0 + tx * TP_THREAD_N;

            float acc[TP_THREAD_M][TP_THREAD_N] = {};

            int local_k_tiles = kt_end - kt_begin;

            #define A_CHUNKS ((TP_BM * TP_BK) / 4)
            #define B_CHUNKS ((TP_BK * TP_BN) / 4)

            // REQ-8: 境界外チャンクでも 16 バイト整列を保ったままクランプ
            // する（`TP_TILE_CORE` と同一のマクロ本文。`k0` に絶対 K
            // タイル位置〈`(kt_begin + s) * TP_BK` 等〉を渡す点のみが
            // 呼び出し側の違い）。
            #define LOAD_A_STAGE(stage, k0) \
                for (int idx = tid; idx < A_CHUNKS; idx += num_threads) { \
                    int row = idx / (TP_BK / 4); \
                    int col0 = (idx % (TP_BK / 4)) * 4; \
                    int gr = block_row0 + row; \
                    int gc = (k0) + col0; \
                    int gr_c = gr < m ? gr : (m > 0 ? m - 1 : 0); \
                    int gc_c = gc < k ? gc : (k > 0 ? ((k - 1) / 4) * 4 : 0); \
                    int valid = (gr < m && gc < k) ? 16 : 0; \
                    tp_cp_async16(&as_tile[stage][row][col0], &a[(size_t)gr_c * k + gc_c], valid); \
                }

            #define LOAD_B_STAGE(stage, k0) \
                for (int idx = tid; idx < B_CHUNKS; idx += num_threads) { \
                    int row = idx / (TP_BN / 4); \
                    int col0 = (idx % (TP_BN / 4)) * 4; \
                    int gr = (k0) + row; \
                    int gc = block_col0 + col0; \
                    int gr_c = gr < k ? gr : (k > 0 ? k - 1 : 0); \
                    int gc_c = gc < n ? gc : (n > 0 ? ((n - 1) / 4) * 4 : 0); \
                    int valid = (gr < k && gc < n) ? 16 : 0; \
                    tp_cp_async16(&bs_tile[stage][row][col0], &b[(size_t)gr_c * n + gc_c], valid); \
                }

            for (int s = 0; s < TP_STAGES - 1; ++s) {
                if (s < local_k_tiles) {
                    LOAD_A_STAGE(s, (kt_begin + s) * TP_BK);
                    LOAD_B_STAGE(s, (kt_begin + s) * TP_BK);
                }
                asm volatile("cp.async.commit_group;\n");
            }

            for (int tl = 0; tl < local_k_tiles; ++tl) {
                int compute_stage = tl % TP_STAGES;
                int next_tl = tl + TP_STAGES - 1;
                int load_stage = next_tl % TP_STAGES;

                asm volatile("cp.async.wait_group %0;\n" ::"n"(TP_STAGES - 2));
                __syncthreads();

#pragma unroll
                for (int kk = 0; kk < TP_BK; ++kk) {
                    float a_reg[TP_THREAD_M];
#pragma unroll
                    for (int i = 0; i < TP_THREAD_M; ++i) {
                        a_reg[i] = as_tile[compute_stage][ty * TP_THREAD_M + i][kk];
                    }
                    float b_reg[TP_THREAD_N];
#pragma unroll
                    for (int j = 0; j < TP_THREAD_N; ++j) {
                        b_reg[j] = bs_tile[compute_stage][kk][tx * TP_THREAD_N + j];
                    }
#pragma unroll
                    for (int i = 0; i < TP_THREAD_M; ++i) {
#pragma unroll
                        for (int j = 0; j < TP_THREAD_N; ++j) {
                            acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
                        }
                    }
                }

                if (next_tl < local_k_tiles) {
                    LOAD_A_STAGE(load_stage, (kt_begin + next_tl) * TP_BK);
                    LOAD_B_STAGE(load_stage, (kt_begin + next_tl) * TP_BK);
                }

                asm volatile("cp.async.commit_group;\n");
                __syncthreads();
            }

            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();

            #undef LOAD_A_STAGE
            #undef LOAD_B_STAGE
            #undef A_CHUNKS
            #undef B_CHUNKS

            // REQ-8: エピローグの guarded store（full タイル単位）／
            // 無条件だが一意なスロットへの書き出し（残タイルのサブ
            // レンジ単位。上記ドキュメンテーションコメント参照）。
            if (store_to_c) {
#pragma unroll
                for (int i = 0; i < TP_THREAD_M; ++i) {
#pragma unroll
                    for (int j = 0; j < TP_THREAD_N; ++j) {
                        int r = thread_row0 + i;
                        int cc = thread_col0 + j;
                        if (r < m && cc < n) {
                            c[(size_t)r * n + cc] = acc[i][j];
                        }
                    }
                }
            } else {
#pragma unroll
                for (int i = 0; i < TP_THREAD_M; ++i) {
#pragma unroll
                    for (int j = 0; j < TP_THREAD_N; ++j) {
                        int local_row = ty * TP_THREAD_M + i;
                        int local_col = tx * TP_THREAD_N + j;
                        int local = local_row * TP_BN + local_col;
                        // REQ-8: スロット添字はホスト側 streamk_plan／起動前
                        // 検査（`crate::gemm::CudaGemm::
                        // launch_tiled_pipeline_streamk_f32`）で partials
                        // 容量内であることを証明済みだが、性能下限・最適化を
                        // 理由に手動境界チェックを省略しない
                        // （`.claude/rules/coding-rust.md`）ためカーネル側でも
                        // 明示的に検査する。
                        long long p_idx = (long long)slot * (TP_BM * TP_BN) + local;
                        if (p_idx >= 0 && p_idx < (long long)partials_capacity) {
                            partials[(size_t)slot * (TP_BM * TP_BN) + local] = acc[i][j];
                        }
                    }
                }
            }
"#;

/// [`TP_SK_KERNEL_PREFIX`]／[`TP_SK_TILE_CORE`] の `for (int sub ...)`
/// ループ・タイル取得 `for (;;)` ループ・関数を閉じる（`}}}\n`）。
const TP_SK_KERNEL_SUFFIX: &str = "        }\n    }\n}\n";

/// [`TP_SK_FIXUP_KERNEL`] のブロックあたりスレッド数（1 タイル分の
/// `TP_BM*TP_BN` 要素を `blockIdx.y` 方向へ分担する固定値。
/// `SPLITK_REDUCE_BLOCK_DIM` と同じ「smem を使わずレジスタのみで完結する
/// ため大きな値は不要」という判断）。カーネルソース中のリテラル `256`
/// （`TP_SK_FIXUP_KERNEL` の `blockIdx.y * 256 + threadIdx.x`）と
/// `crate::gemm::CudaGemm::launch_tiled_pipeline_streamk_f32` の起動
/// `block_dim` の双方が本定数を単一の真実源として使う（起動側は本定数を
/// 直接参照するため乖離しないが、カーネルソース側のリテラルは NVRTC
/// コンパイル時定数のため本定数の値を変更する場合は
/// `TP_SK_FIXUP_KERNEL` 内のリテラルも合わせて変更すること。
/// `tiled_pipeline_streamk_fixup_block_dim_matches_kernel_source_literal`
/// が両者の食い違いを検出する）。
pub const TP_SK_FIXUP_BLOCK_THREADS: u32 = 256;

/// Stream-K fixup カーネル（部分和の固定順序直列還元。イシュー #1358）。
///
/// [`TP_SK_KERNEL_PREFIX`]／[`TP_SK_TILE_CORE`] が書いた部分和バッファ
/// `partials` を、寄与者 `c` 昇順（`c = 0..contributors`）の**固定順序**
/// で加算し、最終 `c`（出力）へ 1 回だけ書く。`gemm_splitk_reduce_f32`
/// （[`crate::kernels_gemm_variants::SPLITK_REDUCE_F32`]）と同じ「atomics
/// 不使用・順序固定により決定的」という設計を踏襲する。
///
/// # 決定性の根拠（イシュー #1358 の受け入れ条件本体）
///
/// 1. full タイルの `c` 書き込みは [`TP_SK_TILE_CORE`] の `store_to_c`
///    分岐から 1 要素につき 1 単位（1 CTA）のみが書く。
/// 2. 部分和スロット `(r, c)` は `crate::gemm::streamk_plan` が一意に
///    割り当てるため、各スロットの書き手はちょうど 1 サブレンジ
///    （[`TP_SK_TILE_CORE`] の `else` 分岐）。
/// 3. 本カーネルは寄与者 `c` について **`for (int c = 0; c < contributors;
///    ++c) acc += partials[...]`** の固定昇順で逐次加算し、`tile =
///    full_tiles + r` の座標へ 1 要素 1 スレッドが 1 回だけ書く。
/// 4. `unit_counter` への `atomicAdd`（[`TP_SK_KERNEL_PREFIX`]）は
///    `unsigned int` カウンタ 1 箇所のみに作用し、GEMM の数値蓄積
///    （`float` の `acc[][]`・`partials[]`）には一切触れない
///    （`.claude/rules/coding-rust.md` の FMA 契約・`kernels_mse.rs`／
///    `kernels_rmsnorm.rs` が禁止する float atomicAdd による非決定的な
///    結合順序とは別種の atomic）。
/// 5. Stream-K カーネルの `c` 書き込み領域（full タイル）と本カーネルの
///    書き込み領域（残タイル）は互いに素であり、同一ストリーム順序
///    （Stream-K カーネル → 本カーネル。[`crate::gemm::
///    CudaGemm::launch_tiled_pipeline_streamk_f32`] が同一ストリームへ
///    投入する）で実行される。
///
/// よって同一入力・同一 `crate::gemm::StreamKPlan` に対し出力は実行の
/// たびに bit 同一になる。**残タイルの値は非 Stream-K 版（`TP_TILE_CORE`
/// が K 全体を 1 パスで蓄積する版）とは bit 同一ではない**（K 連鎖の
/// 分割による丸め差。#1100 `splitk_reorder_error_host_model.rs` が示す
/// とおり真値ゼロ近傍で複合判定 fail が出うる。合否判定は兄弟イシュー
/// #1359 が担う。tolerance は変更しない）。
///
/// # 手動境界チェック（REQ-8）
///
/// `r`（`blockIdx.x`）は `remainder_tiles` 未満で起動するため範囲内。
/// `local`（`blockIdx.y*256+threadIdx.x`）が `TP_BM*TP_BN` 以上のスレッド
/// は早期 return する（smem を使わないため `SPLITK_REDUCE_F32` と同じく
/// ブロック同期プリミティブへの到達義務がない）。出力座標は
/// `if (row < m && col < n)` の guarded store。`contributors`（ホスト側
/// `streamk_plan` が `max_contributors` 以下であることを証明済み）は
/// カーネル側でも `max_contributors` へクランプし、`partials` の読み取り
/// 添字 `slot * (TP_BM*TP_BN) + local` は呼び出し元から渡される
/// `partials_capacity`（`CudaSlice::len()`）と突き合わせて範囲内のときのみ
/// 読む（[`TP_SK_KERNEL_PREFIX`] の書き込み側検査と対をなす。性能下限・
/// 最適化を理由に手動境界チェックを省略しない。
/// `.claude/rules/coding-rust.md`）。
///
/// ブロックあたりスレッド数は [`TP_SK_FIXUP_BLOCK_THREADS`] 固定
/// （`crate::gemm::CudaGemm::launch_tiled_pipeline_streamk_f32` の
/// `block_dim` と単一の真実源を共有する）。
const TP_SK_FIXUP_KERNEL: &str = r#"
extern "C" __global__ void gemm_tiled_pipeline_streamk_fixup_f32(
    const float* __restrict__ partials,
    float* __restrict__ c,
    int m, int n, int k,
    int full_tiles, int remainder_tiles, int q, int max_contributors,
    int partials_capacity)
{
    int r = blockIdx.x;
    if (r >= remainder_tiles) {
        return;
    }
    int local = blockIdx.y * 256 + threadIdx.x;
    if (local >= TP_BM * TP_BN) {
        return;
    }

    int num_k_tiles = (k > 0) ? (k - 1) / TP_BK + 1 : 0;
    // 寄与者数 C(r)（`crate::gemm::streamk_plan` ドキュメンテーション
    // コメント §3.1 と同一の整数演算）。
    long long tile_flat_begin = (long long)r * (long long)num_k_tiles;
    long long tile_flat_last = tile_flat_begin + (long long)num_k_tiles - 1;
    int contributors =
        (int)(tile_flat_last / (long long)q - tile_flat_begin / (long long)q + 1);
    // REQ-8: contributors はホスト側 streamk_plan が max_contributors 以下
    // であることを証明済みだが、性能下限・最適化を理由に手動境界チェックを
    // 省略しない（`.claude/rules/coding-rust.md`）ためカーネル側でも
    // クランプする。
    if (contributors > max_contributors) {
        contributors = max_contributors;
    }
    if (contributors < 0) {
        contributors = 0;
    }

    // 固定順序（c 昇順）の逐次加算。decompose せずレジスタのみで完結する
    // （`SPLITK_REDUCE_F32` と同じ判断）。
    float acc = 0.0f;
    for (int c_idx = 0; c_idx < contributors; ++c_idx) {
        long long slot = (long long)r * (long long)max_contributors + (long long)c_idx;
        long long p_idx = slot * (long long)(TP_BM * TP_BN) + local;
        // REQ-8: partials 読み取り前の手動境界チェック（書き込み側の
        // p_idx 検査と対をなす）。
        if (p_idx >= 0 && p_idx < (long long)partials_capacity) {
            acc += partials[p_idx];
        }
    }

    int tile = full_tiles + r;
    int tiles_x = (n - 1) / TP_BN + 1;
    int block_row0 = (tile / tiles_x) * TP_BM;
    int block_col0 = (tile % tiles_x) * TP_BN;
    int row = block_row0 + local / TP_BN;
    int col = block_col0 + local % TP_BN;
    if (row < m && col < n) {
        c[(size_t)row * n + col] = acc;
    }
}
"#;

/// [`render_source`]／[`render_persistent_source`] と同じ形で Stream-K
/// 版ソース全文を組み立てる（`render_defines`・[`TP_CP_ASYNC_HELPER`]・
/// [`TP_SK_KERNEL_PREFIX`]・[`TP_SK_TILE_CORE`]・[`TP_SK_KERNEL_SUFFIX`]・
/// [`TP_SK_FIXUP_KERNEL`] の連結）。両カーネル（Stream-K 本体・fixup）を
/// 同一コンパイル単位に含めるため、`crate::module_cache::
/// load_function_cached` は同一ソース・同一記述子で 2 回呼ばれ、2 回目
/// は（LRU がヒットしていれば）モジュール再コンパイルなしで
/// `func_name` 違いの関数ロードのみになる
/// （`crate::gemm::CudaGemm::compile_tiled_pipeline_streamk_variant` 参照）。
fn render_streamk_source(stages: u32) -> String {
    format!(
        "{defines}{helper}{prefix}{core}{suffix}{fixup}",
        defines = render_defines(stages),
        helper = TP_CP_ASYNC_HELPER,
        prefix = TP_SK_KERNEL_PREFIX,
        core = TP_SK_TILE_CORE,
        suffix = TP_SK_KERNEL_SUFFIX,
        fixup = TP_SK_FIXUP_KERNEL,
    )
}

/// ステージ数（[`TP_DEFAULT_STAGES`]）固定の Stream-K 版カーネルソース。
/// 初回アクセス時に 1 回だけレンダーし、以降はキャッシュ済み文字列参照を
/// 返す（[`tiled_pipeline_persistent_f32_source`] と同じ判断）。
pub fn tiled_pipeline_streamk_f32_source() -> &'static str {
    &TILED_PIPELINE_STREAMK_F32_SOURCE
}

static TILED_PIPELINE_STREAMK_F32_SOURCE: LazyLock<String> =
    LazyLock::new(|| render_streamk_source(TP_DEFAULT_STAGES));

/// 任意のステージ数（[`TP_MIN_STAGES`]..=[`TP_MAX_STAGES`]）の Stream-K
/// 版ソースをオンデマンド生成する（[`tiled_pipeline_persistent_f32_source_with_stages`]
/// と同じ範囲検証）。
pub fn tiled_pipeline_streamk_f32_source_with_stages(stages: u32) -> Result<String, CudaError> {
    if !(TP_MIN_STAGES..=TP_MAX_STAGES).contains(&stages) {
        return Err(CudaError::InvalidKernelConfig {
            detail: format!(
                "tiled_pipeline_streamk_f32_source_with_stages stages ({stages}) must lie \
                 within [{TP_MIN_STAGES}, {TP_MAX_STAGES}]"
            ),
        });
    }
    Ok(render_streamk_source(stages))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// カーネルソースが `cp.async` の主要命令を含むことを検査する
    /// （`kernels_mma_tf32.rs::mma_tf32_source_uses_mma_sync_ldmatrix_cp_async_instructions`
    /// と同型の静的検査。実機コンパイルできない環境でも `cargo test` で
    /// パイプライン機構の存在を機械検出する）。
    #[test]
    fn tiled_pipeline_source_uses_cp_async_instructions() {
        let source = tiled_pipeline_f32_source();
        for needle in [
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "cp.async.wait_group",
            "fmaf(",
        ] {
            assert!(
                source.contains(needle),
                "tiled_pipeline_f32_source() が `{needle}` を含みません"
            );
        }
    }

    /// REQ-8 の手動境界検査（cp.async src_size ゼロ充填・エピローグ
    /// guarded store）がソースから省略されていないことを検査する。
    #[test]
    fn tiled_pipeline_source_retains_manual_bounds_checks() {
        let source = tiled_pipeline_f32_source();
        assert!(
            source.contains("int valid = (gr < m && gc < k) ? 16 : 0;"),
            "A タイルロードの guarded cp.async（src_size ゼロ充填）が見当たりません"
        );
        assert!(
            source.contains("int valid = (gr < k && gc < n) ? 16 : 0;"),
            "B タイルロードの guarded cp.async（src_size ゼロ充填）が見当たりません"
        );
        assert!(
            source.contains("if (r < m && cc < n) {"),
            "エピローグの guarded store が見当たりません"
        );
    }

    /// Rust 側の唯一の真実源（`TP_BM`/`TP_BN`/`TP_BK`/`TP_THREAD_M`/
    /// `TP_THREAD_N`/`TP_THREADS_X`/`TP_A_PAD`/`TP_B_PAD`/既定
    /// `TP_STAGES`）が生成済みカーネルソース内の `#define` と食い違わない
    /// ことを検査する（`kernels.rs::tile_constant_matches_kernel_source_define`
    /// と同型）。
    #[test]
    fn tiled_pipeline_constants_match_kernel_source_defines() {
        let source = tiled_pipeline_f32_source();
        let checks: [(&str, u32); 9] = [
            ("TP_BM", TP_BM),
            ("TP_BN", TP_BN),
            ("TP_BK", TP_BK),
            ("TP_THREAD_M", TP_THREAD_M),
            ("TP_THREAD_N", TP_THREAD_N),
            ("TP_THREADS_X", TP_THREADS_X),
            ("TP_A_PAD", TP_A_PAD),
            ("TP_B_PAD", TP_B_PAD),
            ("TP_STAGES", TP_DEFAULT_STAGES),
        ];
        for (name, value) in checks {
            let expected = format!("#define {name} {value}");
            assert!(
                source.contains(&expected),
                "tiled_pipeline_f32_source() の `#define {name}` が Rust 側の \
                 定数（{value}）と一致しません"
            );
        }
    }

    /// [`tiled_pipeline_f32_source_with_stages`] の範囲検証（2〜4）を
    /// 検査する。
    #[test]
    fn tiled_pipeline_source_with_stages_validates_range() {
        assert!(tiled_pipeline_f32_source_with_stages(1).is_err());
        assert!(tiled_pipeline_f32_source_with_stages(TP_MAX_STAGES + 1).is_err());
        for stages in TP_MIN_STAGES..=TP_MAX_STAGES {
            let src = tiled_pipeline_f32_source_with_stages(stages)
                .unwrap_or_else(|e| panic!("stages={stages} must be accepted: {e}"));
            assert!(src.contains(&format!("#define TP_STAGES {stages}")));
        }
    }

    /// `cp.async.commit_group` がループ内で 1 箇所（プロローグ・本体
    /// ループ末尾）のみから発行され、`wait_group` がプロローグ後・ループ内
    /// にのみ現れることを検査する（段数を変えても不変条件が崩れていない
    /// ことの粗い機械検査。`kernels_mma_tf32.rs` 同種テストと同じ動機）。
    #[test]
    fn tiled_pipeline_commit_wait_group_counts() {
        let source = tiled_pipeline_f32_source();
        let commit_count = source.matches("cp.async.commit_group;").count();
        let wait_count = source.matches("cp.async.wait_group").count();
        // プロローグ 1 箇所 + 本体ループ末尾 1 箇所 = 2 箇所。
        assert_eq!(
            commit_count, 2,
            "commit_group は 2 箇所（prologue・本体末尾）"
        );
        // 本体ループ内 wait_group（固定即値） + drain（即値 0）= 2 箇所。
        assert_eq!(wait_count, 2, "wait_group は 2 箇所（本体ループ・drain）");
    }

    /// [`fnv1a64`] で使う FNV-1a 64bit オフセットベーシス（FNV-1a 仕様
    /// 定数。<http://www.isthe.com/chongo/tech/comp/fnv/> 準拠）。
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    /// FNV-1a 64bit 素数（同上）。
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    /// [`tiled_pipeline_fragments_reconstruct_non_persistent_source`] が
    /// 断片連結の内容一致を検査するための決定的ハッシュ関数
    /// （FNV-1a 64bit。暗号学的強度は不要で、断片への偶発的変更を検出
    /// できる決定性だけを要件とする）。
    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut hash = FNV_OFFSET_BASIS;
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// [`TP_CP_ASYNC_HELPER`]・[`TP_NON_PERSISTENT_PREFIX`]・
    /// [`TP_TILE_CORE`]・[`TP_KERNEL_SUFFIX`] の連結内容が、分割時点で
    /// 記録した固定ハッシュ（[`EXPECTED_NON_PERSISTENT_FRAGMENTS_FNV1A64`]）
    /// と一致することを検査する（イシュー #1346 AC-3・PR #1383
    /// codex-review 指摘への対応）。
    ///
    /// 旧実装は `expected`（[`tiled_pipeline_f32_source`]。内部で
    /// [`render_source`] が同じ 4 断片から `format!` する）と
    /// `reconstructed`（同じ 4 断片を直接連結）を比較する自己参照的な
    /// 構成だった。この構成では [`TP_TILE_CORE`] 等の断片自体に偶発的な
    /// 変更が入っても `expected`／`reconstructed` の双方に同じ変更が
    /// 伝播するため検出できない。本テストは分割時点の連結内容から一度
    /// だけ計算し固定リテラルとして埋め込んだハッシュ
    /// （断片の実体に依存しない独立した期待値）と比較することで、断片
    /// への偶発的変更を検出可能にする。
    #[test]
    fn tiled_pipeline_fragments_reconstruct_non_persistent_source() {
        /// 分割時点（イシュー #1346 実装時）の
        /// `TP_CP_ASYNC_HELPER`＋`TP_NON_PERSISTENT_PREFIX`＋`TP_TILE_CORE`＋
        /// `TP_KERNEL_SUFFIX` 連結内容から算出した FNV-1a 64bit ハッシュ
        /// （[`fnv1a64`]）。断片群のいずれかが意図せず変更されると
        /// このハッシュと一致しなくなる。意図した変更（カーネル改修）の
        /// 際は、変更後の連結内容から再計算した値へこの定数を更新する
        /// （`git show` 等で分割前ソース fixture との突合も併せて行う）。
        const EXPECTED_NON_PERSISTENT_FRAGMENTS_FNV1A64: u64 = 7_730_393_218_813_914_207;

        let expected = tiled_pipeline_f32_source();
        let reconstructed = format!(
            "{TP_CP_ASYNC_HELPER}{TP_NON_PERSISTENT_PREFIX}{TP_TILE_CORE}{TP_KERNEL_SUFFIX}"
        );
        assert!(
            expected.ends_with(&reconstructed),
            "TP_CP_ASYNC_HELPER/TP_NON_PERSISTENT_PREFIX/TP_TILE_CORE/TP_KERNEL_SUFFIX の \
             連結が tiled_pipeline_f32_source() の本体と一致しません"
        );
        let actual_hash = fnv1a64(reconstructed.as_bytes());
        assert_eq!(
            actual_hash, EXPECTED_NON_PERSISTENT_FRAGMENTS_FNV1A64,
            "TP_CP_ASYNC_HELPER/TP_NON_PERSISTENT_PREFIX/TP_TILE_CORE/TP_KERNEL_SUFFIX の \
             連結内容が分割時点の固定ハッシュと一致しません（断片への意図しない変更の \
             可能性。意図した変更であれば EXPECTED_NON_PERSISTENT_FRAGMENTS_FNV1A64 を \
             再計算して更新する）"
        );
    }

    /// persistent 版ソースが `cp.async`／`fmaf` 命令を含むこと（タイル内
    /// 計算は非 persistent 版と共有のため同じ命令が現れる契約）を検査
    /// する。
    #[test]
    fn tiled_pipeline_persistent_source_uses_cp_async_instructions() {
        let source = tiled_pipeline_persistent_f32_source();
        for needle in [
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "cp.async.wait_group",
            "fmaf(",
        ] {
            assert!(
                source.contains(needle),
                "tiled_pipeline_persistent_f32_source() が `{needle}` を含みません"
            );
        }
    }

    /// REQ-8 の手動境界検査が persistent 版でも省略されていないことを
    /// 検査する（[`tiled_pipeline_source_retains_manual_bounds_checks`]
    /// の persistent 版。[`TP_TILE_CORE`] 共有のため同一文字列が現れる）。
    #[test]
    fn tiled_pipeline_persistent_source_retains_manual_bounds_checks() {
        let source = tiled_pipeline_persistent_f32_source();
        assert!(
            source.contains("int valid = (gr < m && gc < k) ? 16 : 0;"),
            "A タイルロードの guarded cp.async（src_size ゼロ充填）が見当たりません"
        );
        assert!(
            source.contains("int valid = (gr < k && gc < n) ? 16 : 0;"),
            "B タイルロードの guarded cp.async（src_size ゼロ充填）が見当たりません"
        );
        assert!(
            source.contains("if (r < m && cc < n) {"),
            "エピローグの guarded store が見当たりません"
        );
    }

    /// 決定性契約（`.claude/rules/coding-rust.md`・`atomicAdd` を使わない
    /// 数値蓄積の原則）の機械検査: persistent 版ソースに現れる
    /// `atomicAdd` はタイルキューカウンタ用の**ちょうど 1 箇所**であり、
    /// 対象が `unsigned int* tile_counter` であること（float 蓄積用の
    /// `atomicAdd` ではないこと）を検査する（`kernels_mse.rs`／
    /// `kernels_rmsnorm.rs` の同種検査と同じ動機。本ファイル冒頭の
    /// `TP_KERNEL_PERSISTENT_PREFIX` ドキュメンテーションコメント
    /// 「bit 同一の根拠」参照）。非 persistent 版ソースには `atomicAdd`
    /// が一切現れないことも合わせて検査する。
    #[test]
    fn tiled_pipeline_persistent_source_uses_single_integer_atomic_add() {
        let persistent_source = tiled_pipeline_persistent_f32_source();
        let atomic_add_count = persistent_source.matches("atomicAdd(").count();
        assert_eq!(
            atomic_add_count, 1,
            "persistent 版ソースの atomicAdd はタイルキューカウンタ用の 1 箇所のみのはず"
        );
        assert!(
            persistent_source.contains("s_tile = atomicAdd(tile_counter, 1u);"),
            "atomicAdd の対象が unsigned int* tile_counter（スケジューリング専用）と \
             確認できません"
        );
        assert!(
            !persistent_source.contains("atomicAdd(&acc")
                && !persistent_source.contains("atomicAdd(&c["),
            "GEMM の数値蓄積（acc[][]／出力 c）へ atomicAdd が使われていないことを確認できません"
        );

        let non_persistent_source = tiled_pipeline_f32_source();
        assert!(
            !non_persistent_source.contains("atomicAdd"),
            "非 persistent 版ソースに atomicAdd は現れないはず"
        );
    }

    /// persistent 版カーネルのシグネチャ（`tile_counter` 引数）・タイル
    /// キュー機構（`num_tiles`・`s_tile`・`break`）がソースに存在すること
    /// を検査する（関数名・シグネチャの取り違えを防ぐ粗い機械検査）。
    #[test]
    fn tiled_pipeline_persistent_source_has_tile_queue_scaffolding() {
        let source = tiled_pipeline_persistent_f32_source();
        for needle in [
            "gemm_tiled_pipeline_persistent_f32(",
            "unsigned int* tile_counter",
            "__shared__ unsigned int s_tile;",
            "int num_tiles = tiles_x * tiles_y;",
            "for (;;) {",
            "if (tile_u >= (unsigned int)num_tiles) {",
            "break;",
        ] {
            assert!(
                source.contains(needle),
                "tiled_pipeline_persistent_f32_source() が `{needle}` を含みません"
            );
        }
    }

    /// [`tiled_pipeline_persistent_f32_source_with_stages`] の範囲検証
    /// （[`tiled_pipeline_source_with_stages_validates_range`] の
    /// persistent 版）を検査する。
    #[test]
    fn tiled_pipeline_persistent_source_with_stages_validates_range() {
        assert!(tiled_pipeline_persistent_f32_source_with_stages(1).is_err());
        assert!(tiled_pipeline_persistent_f32_source_with_stages(TP_MAX_STAGES + 1).is_err());
        for stages in TP_MIN_STAGES..=TP_MAX_STAGES {
            let src = tiled_pipeline_persistent_f32_source_with_stages(stages)
                .unwrap_or_else(|e| panic!("stages={stages} must be accepted: {e}"));
            assert!(src.contains(&format!("#define TP_STAGES {stages}")));
        }
    }

    /// persistent 版の `commit_group`／`wait_group` 回数が非 persistent 版
    /// と同じ「1 タイル = 2 commit・2 wait」であることを検査する
    /// （[`TP_TILE_CORE`] 共有のため回数自体は不変。持続ループ自体は追加
    /// の commit/wait を発行しない契約の粗い機械検査）。
    #[test]
    fn tiled_pipeline_persistent_commit_wait_group_counts() {
        let source = tiled_pipeline_persistent_f32_source();
        let commit_count = source.matches("cp.async.commit_group;").count();
        let wait_count = source.matches("cp.async.wait_group").count();
        assert_eq!(
            commit_count, 2,
            "persistent 版でも commit_group は 2 箇所（prologue・本体末尾）"
        );
        assert_eq!(
            wait_count, 2,
            "persistent 版でも wait_group は 2 箇所（本体ループ・drain）"
        );
    }

    // -------------------------------------------------------------------
    // Stream-K（最終 wave 限定・固定順序 fixup。イシュー #1358）。
    // -------------------------------------------------------------------

    /// Stream-K 版ソースが `cp.async`／`fmaf`（タイル内計算は
    /// [`TP_TILE_CORE`] 由来）を含むことを検査する
    /// （[`tiled_pipeline_persistent_source_uses_cp_async_instructions`]
    /// と同型）。
    #[test]
    fn tiled_pipeline_streamk_source_uses_cp_async_instructions() {
        let source = tiled_pipeline_streamk_f32_source();
        for needle in [
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "cp.async.wait_group",
            "fmaf(",
        ] {
            assert!(
                source.contains(needle),
                "tiled_pipeline_streamk_f32_source() が `{needle}` を含みません"
            );
        }
    }

    /// REQ-8 の手動境界検査（cp.async src_size ゼロ充填・エピローグ
    /// guarded store）が Stream-K 版でも省略されていないことを検査する。
    #[test]
    fn tiled_pipeline_streamk_source_retains_manual_bounds_checks() {
        let source = tiled_pipeline_streamk_f32_source();
        assert!(
            source.contains("int valid = (gr < m && gc < k) ? 16 : 0;"),
            "A タイルロードの guarded cp.async（src_size ゼロ充填）が見当たりません"
        );
        assert!(
            source.contains("int valid = (gr < k && gc < n) ? 16 : 0;"),
            "B タイルロードの guarded cp.async（src_size ゼロ充填）が見当たりません"
        );
        assert!(
            source.contains("if (r < m && cc < n) {"),
            "full タイル単位のエピローグ guarded store が見当たりません"
        );
        assert!(
            source.contains("if (row < m && col < n) {"),
            "fixup カーネルの guarded store が見当たりません"
        );
    }

    /// 空白（インデント幅）のみを除去する（`TP_TILE_CORE`／`TP_SK_TILE_CORE`
    /// はネスト深さが異なるためインデント幅だけが違う。以下のテストは
    /// この差を無視して命令列自体の同一性を検証する）。
    fn strip_ws(s: &str) -> String {
        s.chars().filter(|c| !c.is_whitespace()).collect()
    }

    /// [`TP_SK_TILE_CORE`] が [`TP_TILE_CORE`] と（インデントを除き）同一の
    /// `LOAD_A_STAGE`／`LOAD_B_STAGE` マクロ本文・K 内積 `fmaf` ループ本文を
    /// 含むことを検査する（決定性論証「full タイル単位は非 Stream-K 版と
    /// 同一命令列」の機械的裏付け。実装計画 §3.2 参照。`TP_TILE_CORE` を
    /// 単一の真実源として動的に切り出すことで、インデント幅の変更に対して
    /// 脆くならないようにする）。
    #[test]
    fn tiled_pipeline_streamk_tile_core_shares_load_and_fma_fragments_with_tile_core() {
        let core_ws = strip_ws(TP_TILE_CORE);
        let sk_core_ws = strip_ws(TP_SK_TILE_CORE);

        // LOAD_A_STAGE マクロ本文（`#define LOAD_A_STAGE(stage, k0)` から
        // `#define LOAD_B_STAGE(stage, k0)` 直前まで）を `TP_TILE_CORE` から
        // 動的に切り出す。
        let load_a_marker = "#defineLOAD_A_STAGE(stage,k0)";
        let load_b_marker = "#defineLOAD_B_STAGE(stage,k0)";
        let load_a_start = core_ws
            .find(load_a_marker)
            .expect("テスト前提が崩れています: TP_TILE_CORE に LOAD_A_STAGE 定義が見当たりません");
        let load_b_start = core_ws
            .find(load_b_marker)
            .expect("テスト前提が崩れています: TP_TILE_CORE に LOAD_B_STAGE 定義が見当たりません");
        assert!(load_a_start < load_b_start);
        let load_a_fragment = &core_ws[load_a_start..load_b_start];
        assert!(
            sk_core_ws.contains(load_a_fragment),
            "TP_SK_TILE_CORE が TP_TILE_CORE と同一の LOAD_A_STAGE マクロ本文を（空白を \
             除いて）含みません"
        );

        // LOAD_B_STAGE マクロ本文（`#define LOAD_B_STAGE` からマクロ本体の
        // 最終行〈`tp_cp_async16(...);` を含む閉じ `}`〉まで。この直後に
        // `TP_TILE_CORE` 側だけに存在するプロローグ導入コメントが続くため、
        // そのコメント文字列を巻き込まないよう終端をマクロ本体の既知の末尾
        // 文字列で明示的に区切る（コメントの有無自体は `TP_SK_TILE_CORE` と
        // 異なりうる非本質的差異であり、本テストが検証したいのは命令列その
        // ものの同一性）。
        let load_b_end_marker = strip_ws(
            "tp_cp_async16(&bs_tile[stage][row][col0], &b[(size_t)gr_c * n + gc_c], valid); \\\n        }",
        );
        let load_b_end_rel = core_ws[load_b_start..].find(&load_b_end_marker).expect(
            "テスト前提が崩れています: TP_TILE_CORE に LOAD_B_STAGE 本体の末尾が見当たりません",
        );
        let load_b_end = load_b_start + load_b_end_rel + load_b_end_marker.len();
        let load_b_fragment = &core_ws[load_b_start..load_b_end];
        assert!(
            sk_core_ws.contains(load_b_fragment),
            "TP_SK_TILE_CORE が TP_TILE_CORE と同一の LOAD_B_STAGE マクロ本文を（空白を \
             除いて）含みません"
        );

        let kk_fma_fragment = strip_ws("acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);");
        assert!(core_ws.contains(&kk_fma_fragment));
        assert!(
            sk_core_ws.contains(&kk_fma_fragment),
            "TP_SK_TILE_CORE が TP_TILE_CORE と同一の fmaf 内積ループ本文を含みません"
        );
    }

    /// 決定性契約の機械検査: Stream-K 版ソースの `atomicAdd` はタイル
    /// キューカウンタ用の**ちょうど 1 箇所**（`unit_counter`）であり、
    /// GEMM の数値蓄積（`acc[][]`・`c[`・`partials[`）へは一切使われない
    /// ことを検査する（[`tiled_pipeline_persistent_source_uses_single_
    /// integer_atomic_add`] と同型）。
    #[test]
    fn tiled_pipeline_streamk_source_uses_single_integer_atomic_add() {
        let source = tiled_pipeline_streamk_f32_source();
        let atomic_add_count = source.matches("atomicAdd(").count();
        assert_eq!(
            atomic_add_count, 1,
            "Stream-K 版ソースの atomicAdd はタイルキューカウンタ用の 1 箇所のみのはず"
        );
        assert!(
            source.contains("s_unit = atomicAdd(unit_counter, 1u);"),
            "atomicAdd の対象が unsigned int* unit_counter（スケジューリング専用）と \
                 確認できません"
        );
        assert!(
            !source.contains("atomicAdd(&acc")
                && !source.contains("atomicAdd(&c[")
                && !source.contains("atomicAdd(&partials["),
            "GEMM の数値蓄積（acc[][]／c／partials）へ atomicAdd が使われていないことを \
                 確認できません"
        );
    }

    /// fixup カーネルが寄与者 `c_idx` について固定昇順（`for (int c_idx =
    /// 0; c_idx < contributors; ++c_idx)`）で逐次加算し、`c[` への書き
    /// 込みが 1 箇所のみであることを検査する（`SPLITK_REDUCE_F32` と同型
    /// の決定性契約。イシュー #1358 の受け入れ条件本体）。
    #[test]
    fn tiled_pipeline_streamk_fixup_uses_fixed_ascending_order_reduction() {
        assert!(
            TP_SK_FIXUP_KERNEL.contains("for (int c_idx = 0; c_idx < contributors; ++c_idx) {"),
            "fixup カーネルが寄与者昇順の固定順序ループを持ちません"
        );
        assert_eq!(
            TP_SK_FIXUP_KERNEL
                .matches("c[(size_t)row * n + col] = acc;")
                .count(),
            1,
            "fixup カーネルの c への書き込みは 1 箇所のはずです"
        );
        assert!(!TP_SK_FIXUP_KERNEL.contains("atomicAdd"));
    }

    /// Stream-K 版・fixup カーネルいずれも SK 側の `c[` 書き込みが full
    /// タイル単位（`store_to_c` 分岐）の 1 箇所のみであること、残タイル
    /// は `partials[` への書き込みのみであることを検査する（full タイル
    /// と残タイルの書き込み領域が互いに素であることの静的裏付け。
    /// [`TP_SK_FIXUP_KERNEL`] ドキュメンテーションコメント「決定性の
    /// 根拠」点 5 参照）。
    #[test]
    fn tiled_pipeline_streamk_c_and_partials_writes_are_disjoint_by_construction() {
        let source = tiled_pipeline_streamk_f32_source();
        assert_eq!(
            source.matches("c[(size_t)r * n + cc] = acc[i][j];").count(),
            1,
            "Stream-K 版カーネルの c への書き込み（full タイル単位）は 1 箇所のはずです"
        );
        assert_eq!(
            source
                .matches("partials[(size_t)slot * (TP_BM * TP_BN) + local] = acc[i][j];")
                .count(),
            1,
            "Stream-K 版カーネルの partials への書き込み（残タイルのサブレンジ単位）は \
                 1 箇所のはずです"
        );
    }

    /// Stream-K 版カーネルのシグネチャ（`unit_counter`／`partials`／
    /// `full_tiles`／`total_units`／`q`／`max_contributors`／
    /// `remainder_tiles` 引数）・単位取得ループの scaffolding
    /// （[`tiled_pipeline_persistent_source_has_tile_queue_scaffolding`]
    /// の Stream-K 版）を検査する。
    #[test]
    fn tiled_pipeline_streamk_source_has_unit_queue_scaffolding() {
        let source = tiled_pipeline_streamk_f32_source();
        for needle in [
            "gemm_tiled_pipeline_streamk_f32(",
            "unsigned int* unit_counter",
            "float* __restrict__ partials",
            "int full_tiles, int total_units, int q, int max_contributors,",
            "int remainder_tiles",
            "__shared__ unsigned int s_unit;",
            "for (;;) {",
            "if (unit_u >= (unsigned int)total_units) {",
            "gemm_tiled_pipeline_streamk_fixup_f32(",
        ] {
            assert!(
                source.contains(needle),
                "tiled_pipeline_streamk_f32_source() が `{needle}` を含みません"
            );
        }
    }

    /// [`tiled_pipeline_streamk_f32_source_with_stages`] の範囲検証
    /// （[`tiled_pipeline_source_with_stages_validates_range`] と同型）。
    #[test]
    fn tiled_pipeline_streamk_source_with_stages_validates_range() {
        assert!(tiled_pipeline_streamk_f32_source_with_stages(1).is_err());
        assert!(tiled_pipeline_streamk_f32_source_with_stages(TP_MAX_STAGES + 1).is_err());
        for stages in TP_MIN_STAGES..=TP_MAX_STAGES {
            let src = tiled_pipeline_streamk_f32_source_with_stages(stages)
                .unwrap_or_else(|e| panic!("stages={stages} must be accepted: {e}"));
            assert!(src.contains(&format!("#define TP_STAGES {stages}")));
        }
    }

    /// Stream-K 版ソースの `commit_group`／`wait_group` 回数が非 Stream-K
    /// 版と同じ「1 サブレンジ = 2 commit・2 wait」であることを検査する
    /// （[`TP_SK_TILE_CORE`] が [`TP_TILE_CORE`] と同一命令列のため回数
    /// 自体は不変。fixup カーネルは cp.async を一切使わない）。
    #[test]
    fn tiled_pipeline_streamk_commit_wait_group_counts() {
        let source = tiled_pipeline_streamk_f32_source();
        let commit_count = source.matches("cp.async.commit_group;").count();
        let wait_count = source.matches("cp.async.wait_group").count();
        assert_eq!(
            commit_count, 2,
            "Stream-K 版でも commit_group は 2 箇所（prologue・本体末尾）"
        );
        assert_eq!(
            wait_count, 2,
            "Stream-K 版でも wait_group は 2 箇所（本体ループ・drain）"
        );
    }

    /// レンダー済み Stream-K 版ソースの構造的整合性（波括弧・`#define`／
    /// `#undef` の対応数・`extern "C" __global__` 関数が 2 個）を検査する
    /// 粗い機械検査（advisor 指摘: NVRTC 非搭載環境ではブレース崩れが
    /// 実機でしか顕在化しないため、部分文字列検査に加えて構造カウントで
    /// 保険をかける）。
    #[test]
    fn tiled_pipeline_streamk_source_is_structurally_balanced() {
        let source = tiled_pipeline_streamk_f32_source();
        let open_braces = source.matches('{').count();
        let close_braces = source.matches('}').count();
        assert_eq!(
            open_braces, close_braces,
            "Stream-K 版ソースの `{{`/`}}` 個数が一致しません（open={open_braces}, \
                 close={close_braces}）"
        );
        // `render_defines` が生成する `TP_BM` 等の恒常マクロには対応する
        // `#undef` が存在しない契約（他フラグメントと共通の前提）のため、
        // `#define`/`#undef` の総数一致ではなく `TP_SK_TILE_CORE` がスコープ
        // 内でのみ使う 4 マクロ（`A_CHUNKS`／`B_CHUNKS`／`LOAD_A_STAGE`／
        // `LOAD_B_STAGE`）に限定して対応する `#undef` の存在を検査する。
        for macro_name in ["A_CHUNKS", "B_CHUNKS", "LOAD_A_STAGE", "LOAD_B_STAGE"] {
            assert_eq!(
                source.matches(&format!("#define {macro_name}")).count(),
                1,
                "`#define {macro_name}` は 1 箇所のはずです"
            );
            assert_eq!(
                source.matches(&format!("#undef {macro_name}")).count(),
                1,
                "`#undef {macro_name}` は 1 箇所のはずです（TP_SK_TILE_CORE スコープ内で \
                 定義・破棄される契約）"
            );
        }
        assert_eq!(
            source.matches("extern \"C\" __global__ void").count(),
            2,
            "Stream-K 版ソースは Stream-K 本体・fixup の 2 関数のみを含むはずです"
        );
    }

    /// [`TP_SK_FIXUP_BLOCK_THREADS`] と [`TP_SK_FIXUP_KERNEL`] 内のリテラル
    /// （`blockIdx.y * 256 + threadIdx.x`）が食い違っていないことを検査
    /// する（[`TP_SK_FIXUP_KERNEL`] ドキュメンテーションコメント参照。
    /// `crate::gemm::CudaGemm::launch_tiled_pipeline_streamk_f32` の起動
    /// `block_dim` は本定数を直接参照するため、ここではカーネルソース側の
    /// リテラルとの一致のみを検査すれば足りる）。
    #[test]
    fn tiled_pipeline_streamk_fixup_block_dim_matches_kernel_source_literal() {
        let expected = format!("blockIdx.y * {TP_SK_FIXUP_BLOCK_THREADS} + threadIdx.x");
        assert!(
            TP_SK_FIXUP_KERNEL.contains(&expected),
            "TP_SK_FIXUP_KERNEL のリテラルが TP_SK_FIXUP_BLOCK_THREADS（{TP_SK_FIXUP_BLOCK_THREADS}）\
             と一致しません"
        );
    }
}
