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

fn render_source(stages: u32) -> String {
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
         \n{body}",
        bm = TP_BM,
        bn = TP_BN,
        bk = TP_BK,
        thread_m = TP_THREAD_M,
        thread_n = TP_THREAD_N,
        threads_x = TP_THREADS_X,
        a_pad = TP_A_PAD,
        b_pad = TP_B_PAD,
        stages = stages,
        body = TILED_PIPELINE_F32_BODY,
    )
}

/// [`render_source`] が結合するカーネル本体テンプレート。
///
/// `TP_STAGES` は `format!` で埋め込まれる `#define` のみに依存し、本体
/// 文字列自体はステージ数に非依存（配列サイズ・`STAGES - 2` 等の算術は
/// すべて `TP_STAGES` マクロ経由）。
const TILED_PIPELINE_F32_BODY: &str = r#"
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

extern "C" __global__ void gemm_tiled_pipeline_f32(
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

    int tid = threadIdx.x;
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
}
"#;

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
}
