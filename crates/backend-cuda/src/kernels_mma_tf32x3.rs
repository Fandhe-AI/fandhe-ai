//! 3×TF32（split-single 法）`mma.sync`(m16n8k8) GEMM の CUDA カーネル
//! ソース（イシュー #1355。親ツリー #1354・承認元 #1338）。
//!
//! # 位置づけ
//!
//! `kernels_mma_tf32.rs::MMA_TF32_BODY`（単発 TF32 経路。#801）を基点に、
//! A・B オペランドを hi/lo の 2 語（各々 TF32 丸め済み）へレジスタ段で
//! 分割し、`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` を
//! `(a_hi, b_lo) → (a_lo, b_hi) → (a_hi, b_hi)` の順で 3 回累積すること
//! で、単発 TF32（仮数 10bit 丸め）より高い実効精度を Tensor Core 上で
//! 得る（CUTLASS `mma_tensor_op_fast_f32` と同型の split-single 法。
//! `lo·lo` 項は相対 ~2^-22 のため省略する）。
//!
//! `Fp32Strict`（既定）・`Tf32`（単発。#1042）に続く `precision.rs`
//! （`CudaGemmPrecision`）の第 3 モードとして `ops.rs::gemm` へ opt-in
//! 結線する（本ファイル自体は結線を持たない。結線は `ops.rs`）。
//!
//! # タイル構成のエイリアス
//!
//! 本カーネルはブロックタイル・ステージ数・warp 構成のいずれも
//! `kernels_mma_tf32.rs` の定数を `pub const … = kernels_mma_tf32::…;`
//! でそのままエイリアスし独自の値を持たない（#1356 の A/B 比較条件を
//! 単発 TF32 と揃えるため）。共有メモリは hi/lo 分割を **レジスタ段**
//! （フラグメントロード直後）で行うため smem 上には生の f32 をそのまま
//! 保持すればよく、単発 TF32 経路と静的 SMEM サイズは完全に同一
//! （[`MMA_TF32X3_SHARED_MEM_BYTES`] は
//! `kernels_mma_tf32::MMA_TF32_SHARED_MEM_BYTES` とコンパイル時 assert で
//! 一致を固定する）。hi/lo 2 面の smem 分離案（各面 28,416B・合計
//! 56,832B）は静的 SMEM 上限（[`crate::kernels_mma::
//! MMA_STATIC_SMEM_LIMIT_BYTES`]・48KiB）を超えるため不採用（設計判断の
//! 詳細は `docs/cuda-tf32x3-split-single-decision.md`）。
//!
//! # 分割式（hi/lo 分割。レジスタ段）
//!
//! `float v = __uint_as_float(raw); float hi = wmma::__float_to_tf32(v);
//! float lo = wmma::__float_to_tf32(v - hi);`（`v - hi` は f32 減算で
//! 計算し、その残差を再度 TF32 丸めして `lo` を得る）。A オペランドは
//! `kernels_mma_tf32.rs::LDSM_A_FRAG` と同じ `ldmatrix.x4` の b16 流用
//! ロードで smem 上の生ビットをレジスタへ運んだ直後にこの分割を適用する
//! （ldmatrix はビットパターンをそのまま転置なしで再配置するのみのため、
//! smem に生 f32 を置いても正しく `raw` を得られる）。B オペランドは
//! `.trans` ldmatrix を使わない直接 smem ロード（`kernels_mma_tf32.rs::
//! LDS_B_FRAG` と同型）の直後に同じ分割を適用する。
//!
//! # 3 回の累積順序
//!
//! `(a_hi, b_lo) → (a_lo, b_hi) → (a_hi, b_hi)`（小さい項を先に累積し
//! `hi·hi` を最後に累積する。CUTLASS `mma_tensor_op_fast_f32`
//! （`include/cutlass/gemm/warp/mma_tensor_op_fast_f32.h`）の
//! `mma(d, a[1], b[0], c); mma(d, a[0], b[1], d); mma(d, a[0], b[0],
//! d);`（`a[0]`/`b[0]` が hi、`a[1]`/`b[1]` が lo の命名規約）と同一の
//! 累積順序）。`mma.sync` 命令文字列自体は
//! [`MMA_TF32X3_ISSUE`]（本ファイル内マクロ。ソース上 1 箇所）から 3 回
//! 呼び出す（コピペ増殖の回帰検出テスト参照）。
//!
//! # FMA 契約の例外（重要）
//!
//! 本経路の結果は f32 SIMT 参照実装と **bit 一致しない**
//! （`.claude/rules/coding-rust.md` の FMA 契約統一節の明示的な例外。
//! ユーザー承認 2026-09-06・#1338 コメント）。数値一致は
//! `.claude/rules/coding-rust.md` の統一複合判定（相対誤差 1e-3 未満
//! または 絶対誤差 1e-5 未満）の範囲内で扱う。
//!
//! # 境界検査（REQ-8。省略禁止）・整列制約
//!
//! `kernels_mma_tf32.rs` 冒頭コメント「境界検査」「整列制約」と同一
//! （cp.async ゼロ充填・16B/f32 4 要素境界クランプ・エピローグ guarded
//! store。ホスト側整列検証は `gemm_mma_tf32x3.rs` が
//! `gemm_mma_tf32::validate_mma_tf32_alignment` 等を再利用する）。
//!
//! # 非対象（スコープ外）
//!
//! `mma_tf32_source_with_block_tile`（診断専用・候補タイル生成）相当の
//! x3 版は本イシューでは実装しない（`.claude/rules/out-of-scope-
//! tracking.md`。実測・性能評価は #1356 が引き継ぐ）。

use std::sync::LazyLock;

use crate::kernels_mma_tf32;

// タイル構成の完全エイリアス（本ファイル冒頭コメント「タイル構成の
// エイリアス」参照）。x3 経路は独自のタイル値を持たない。
pub const MMA_TF32X3_M: u32 = kernels_mma_tf32::MMA_TF32_M;
pub const MMA_TF32X3_N: u32 = kernels_mma_tf32::MMA_TF32_N;
pub const MMA_TF32X3_K: u32 = kernels_mma_tf32::MMA_TF32_K;
pub const MMA_TF32X3_BM: u32 = kernels_mma_tf32::MMA_TF32_BM;
pub const MMA_TF32X3_BN: u32 = kernels_mma_tf32::MMA_TF32_BN;
pub const MMA_TF32X3_BK: u32 = kernels_mma_tf32::MMA_TF32_BK;
pub const MMA_TF32X3_STAGES: u32 = kernels_mma_tf32::MMA_TF32_STAGES;
pub const MMA_TF32X3_WARP_TILES_M: u32 = kernels_mma_tf32::MMA_TF32_WARP_TILES_M;
pub const MMA_TF32X3_WARP_TILES_N: u32 = kernels_mma_tf32::MMA_TF32_WARP_TILES_N;
pub const MMA_TF32X3_WARP_M: u32 = kernels_mma_tf32::MMA_TF32_WARP_M;
pub const MMA_TF32X3_WARP_N: u32 = kernels_mma_tf32::MMA_TF32_WARP_N;
pub const MMA_TF32X3_WARPS_M: u32 = kernels_mma_tf32::MMA_TF32_WARPS_M;
pub const MMA_TF32X3_WARPS_N: u32 = kernels_mma_tf32::MMA_TF32_WARPS_N;
pub const MMA_TF32X3_BLOCK_THREADS: u32 = kernels_mma_tf32::MMA_TF32_BLOCK_THREADS;
pub const MMA_TF32X3_K_STEPS_PER_STAGE: u32 = kernels_mma_tf32::MMA_TF32_K_STEPS_PER_STAGE;
pub const MMA_TF32X3_A_PAD: u32 = kernels_mma_tf32::MMA_TF32_A_PAD;
pub const MMA_TF32X3_B_PAD: u32 = kernels_mma_tf32::MMA_TF32_B_PAD;

/// 静的共有メモリ使用量（バイト）。hi/lo 分割はレジスタ段のため smem
/// レイアウトは単発 TF32 経路と完全に同一（本ファイル冒頭コメント
/// 「タイル構成のエイリアス」参照）。
pub const MMA_TF32X3_SHARED_MEM_BYTES: u32 = kernels_mma_tf32::MMA_TF32_SHARED_MEM_BYTES;

// コンパイル時契約検査。タイル定数を完全エイリアスしているため、
// 単発 TF32 側で既に検査済みの制約（BK の K 整除性・BM/BN の 4 の
// 倍数性・パディングの 4 の倍数性・バンク衝突回避・スレッド数上限・
// warp タイルの整除性・STAGES>=2・K_STEPS_PER_STAGE>=2 等）は
// `kernels_mma_tf32.rs` の const assert が既に固定している値をそのまま
// 継承するため本ファイルで再定義しない。本ファイル固有の契約のみ検査
// する。
const _: () = assert!(
    MMA_TF32X3_SHARED_MEM_BYTES == kernels_mma_tf32::MMA_TF32_SHARED_MEM_BYTES,
    "kernels_mma_tf32x3 の静的共有メモリ使用量は hi/lo 分割をレジスタ段で \
     行う設計（本ファイル冒頭コメント参照）のため単発 TF32 経路と \
     完全に一致するはずです"
);
const _: () = assert!(
    MMA_TF32X3_BLOCK_THREADS == 128,
    "kernels_mma_tf32x3 のブロックスレッド数は単発 TF32 経路と同じ 128 \
     （4 warp）を前提とします"
);
// warp 構成のエイリアスが崩れていないことも実行時に再確認する（値の
// 定義はいずれも `kernels_mma_tf32` 側の const 演算をそのまま参照して
// いるため恒真だが、本ファイル冒頭コメント「タイル構成のエイリアス」
// との整合をコンパイル時に固定する）。
const _: () = assert!(MMA_TF32X3_WARP_M == kernels_mma_tf32::MMA_TF32_WARP_M);
const _: () = assert!(MMA_TF32X3_WARP_N == kernels_mma_tf32::MMA_TF32_WARP_N);
const _: () = assert!(MMA_TF32X3_WARPS_M == kernels_mma_tf32::MMA_TF32_WARPS_M);
const _: () = assert!(MMA_TF32X3_K_STEPS_PER_STAGE == kernels_mma_tf32::MMA_TF32_K_STEPS_PER_STAGE);

/// 3×TF32 `mma.sync`(m16n8k8) GEMM（f32 入出力・f32 内部アキュムレート・
/// hi/lo 分割 3 回累積）の既定構成カーネルソース。
/// `gemm_mma_tf32x3.rs::CudaMmaTf32x3Gemm::new` はこの文字列を
/// `nvrtc::compile_ptx` に渡して `CudaFunction` を得る。カーネルソースは
/// コンパイル時定数のみから `format!` で組み立て、外部入力文字列を
/// 連結しない（`nvrtc.rs` A03 節と同じ契約。`.claude/rules/security.md`
/// A03）。
pub fn mma_tf32x3_source() -> &'static str {
    &MMA_TF32X3_SOURCE
}

static MMA_TF32X3_SOURCE: LazyLock<String> = LazyLock::new(|| {
    format!(
        "\n#include <mma.h>\n\n\
         using namespace nvcuda;\n\n\
         #define MMA_TF32X3_M {m}\n\
         #define MMA_TF32X3_N {n}\n\
         #define MMA_TF32X3_K {k}\n\
         #define MMA_TF32X3_BM {bm}\n\
         #define MMA_TF32X3_BN {bn}\n\
         #define MMA_TF32X3_BK {bk}\n\
         #define MMA_TF32X3_STAGES {stages}\n\
         #define MMA_TF32X3_WARP_TILES_M {warp_tiles_m}\n\
         #define MMA_TF32X3_WARP_TILES_N {warp_tiles_n}\n\
         #define MMA_TF32X3_WARPS_N {warps_n}\n\
         #define MMA_TF32X3_A_PAD {a_pad}\n\
         #define MMA_TF32X3_B_PAD {b_pad}\n\
         \n{body}",
        m = MMA_TF32X3_M,
        n = MMA_TF32X3_N,
        k = MMA_TF32X3_K,
        bm = MMA_TF32X3_BM,
        bn = MMA_TF32X3_BN,
        bk = MMA_TF32X3_BK,
        stages = MMA_TF32X3_STAGES,
        warp_tiles_m = MMA_TF32X3_WARP_TILES_M,
        warp_tiles_n = MMA_TF32X3_WARP_TILES_N,
        warps_n = MMA_TF32X3_WARPS_N,
        a_pad = MMA_TF32X3_A_PAD,
        b_pad = MMA_TF32X3_B_PAD,
        body = MMA_TF32X3_BODY,
    )
});

/// [`mma_tf32x3_source`] が結合するカーネル本体テンプレート。
/// `kernels_mma_tf32.rs::MMA_TF32_BODY` の cp.async 3 ステージ骨格を
/// そのまま踏襲し、差分は (1) smem ステージング直後の TF32 丸め
/// （`CONVERT_A_STAGE_GROUP`/`CONVERT_B_STAGE_GROUP`）を行わず smem には
/// 生の f32 を保持する点、(2) フラグメントロード直後（レジスタ段）で
/// hi/lo 2 語へ分割する点、(3) `mma.sync` を 1 回ではなく `(hi,lo) →
/// (lo,hi) → (hi,hi)` の順で 3 回発行しアキュムレートする点のみ（本
/// ファイル冒頭コメント参照）。
const MMA_TF32X3_BODY: &str = r#"
// REQ-8: グローバル→共有メモリの 16 バイト単位（f32 4 要素）非同期
// コピー。src_size==16 で実データをコピーし、src_size==0 で共有メモリ側を
// ゼロ充填する（kernels_mma_tf32.rs::mma_tf32_cp_async16 と同じ契約・
// 同じ PTX 命令。関数名は同一 NVRTC コンパイル単位内での衝突を避ける
// ため本カーネル専用の接頭辞を付す）。
__device__ __forceinline__ void mma_tf32x3_cp_async16(void* smem_ptr, const void* gmem_ptr, int src_size)
{
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr), "r"(src_size)
    );
}

extern "C" __global__ void gemm_mma_tf32x3(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k)
{
    // __align__(16): cp.async の 16 バイト転送先整列要件。A_PAD/B_PAD が
    // 4 要素の倍数のため各行の先頭は常に 16 バイト整列する。smem には
    // 生の f32（TF32 未丸め）をそのまま保持する（hi/lo 分割はレジスタ段。
    // 本ファイル冒頭コメント「分割式」参照）。
    __shared__ __align__(16) float as_tile[MMA_TF32X3_STAGES][MMA_TF32X3_BM][MMA_TF32X3_A_PAD];
    __shared__ __align__(16) float bs_tile[MMA_TF32X3_STAGES][MMA_TF32X3_BK][MMA_TF32X3_B_PAD];

    int block_row0 = blockIdx.y * MMA_TF32X3_BM;
    int block_col0 = blockIdx.x * MMA_TF32X3_BN;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int warp_row = warp_id / MMA_TF32X3_WARPS_N;
    int warp_col = warp_id % MMA_TF32X3_WARPS_N;
    int row0_warp = block_row0 + warp_row * (MMA_TF32X3_M * MMA_TF32X3_WARP_TILES_M);
    int col0_warp = block_col0 + warp_col * (MMA_TF32X3_N * MMA_TF32X3_WARP_TILES_N);

    // C/D・B オペランドが共有する groupID/tid_in_group（PTX ISA 標準
    // m16n8k8 分解。`kernels_mma_tf32.rs` 冒頭コメント「命令選定」参照）。
    int group_id = lane / 4;
    int tid_in_group = lane % 4;

    // C アキュムレータ（f32 x4 を WARP_TILES_M x WARP_TILES_N 個）。全ゼロ
    // 初期化。`kernels_mma_tf32.rs::MMA_TF32_BODY` と同じレジスタ
    // ブロッキング方式。3 回の mma.sync 累積はいずれもこの同じ d[][][]
    // を読み書きする（本ファイル冒頭コメント「3 回の累積順序」参照）。
    float d[MMA_TF32X3_WARP_TILES_M][MMA_TF32X3_WARP_TILES_N][4] = {};

    int num_k_tiles = (k > 0) ? (k - 1) / MMA_TF32X3_BK + 1 : 0;

    // #496 相当: 1 K タイル分の cp.async 発行を warp 内 kstep ループへ
    // 分散するための添字空間分割（`kernels_mma_tf32.rs::MMA_TF32_BODY`
    // 「#496」節と同一設計）。
    #define K_GROUPS (MMA_TF32X3_BK / MMA_TF32X3_K)
    #define A_CHUNKS ((MMA_TF32X3_BM * MMA_TF32X3_BK) / 4)
    #define B_CHUNKS ((MMA_TF32X3_BK * MMA_TF32X3_BN) / 4)
    #define A_GROUP_CHUNKS ((A_CHUNKS + K_GROUPS - 1) / K_GROUPS)
    #define B_GROUP_CHUNKS ((B_CHUNKS + K_GROUPS - 1) / K_GROUPS)

    // REQ-8: 境界外チャンクでも 16 バイト整列を保ったままクランプする
    // （`kernels_mma_tf32.rs::LOAD_A_STAGE_GROUP`/`LOAD_B_STAGE_GROUP` と
    // 同一式）。
    #define LOAD_A_STAGE_GROUP(stage, k0, g) \
        for (int idx = (g) * A_GROUP_CHUNKS + tid; \
             idx < A_CHUNKS && idx < ((g) + 1) * A_GROUP_CHUNKS; \
             idx += num_threads) { \
            int row = idx / (MMA_TF32X3_BK / 4); \
            int col0 = (idx % (MMA_TF32X3_BK / 4)) * 4; \
            int gr = block_row0 + row; \
            int gc = (k0) + col0; \
            int gr_c = gr < m ? gr : (m > 0 ? m - 1 : 0); \
            int gc_c = gc < k ? gc : (k > 0 ? ((k - 1) / 4) * 4 : 0); \
            int valid = (gr < m && gc < k) ? 16 : 0; \
            mma_tf32x3_cp_async16(&as_tile[stage][row][col0], &a[(size_t)gr_c * k + gc_c], valid); \
        }

    #define LOAD_B_STAGE_GROUP(stage, k0, g) \
        for (int idx = (g) * B_GROUP_CHUNKS + tid; \
             idx < B_CHUNKS && idx < ((g) + 1) * B_GROUP_CHUNKS; \
             idx += num_threads) { \
            int row = idx / (MMA_TF32X3_BN / 4); \
            int col0 = (idx % (MMA_TF32X3_BN / 4)) * 4; \
            int gr = (k0) + row; \
            int gc = block_col0 + col0; \
            int gr_c = gr < k ? gr : (k > 0 ? k - 1 : 0); \
            int gc_c = gc < n ? gc : (n > 0 ? ((n - 1) / 4) * 4 : 0); \
            int valid = (gr < k && gc < n) ? 16 : 0; \
            mma_tf32x3_cp_async16(&bs_tile[stage][row][col0], &b[(size_t)gr_c * n + gc_c], valid); \
        }

    #define LOAD_A_STAGE(stage, k0) \
        for (int g_ = 0; g_ < K_GROUPS; ++g_) { \
            LOAD_A_STAGE_GROUP(stage, k0, g_); \
        }

    #define LOAD_B_STAGE(stage, k0) \
        for (int g_ = 0; g_ < K_GROUPS; ++g_) { \
            LOAD_B_STAGE_GROUP(stage, k0, g_); \
        }

    // プロローグ: `kernels_mma_tf32.rs::MMA_TF32_BODY` プロローグと同一の
    // 「1 イテレーション = 必ず 1 commit」不変条件（#492）。
    for (int s = 0; s < MMA_TF32X3_STAGES - 1; ++s) {
        if (s < num_k_tiles) {
            LOAD_A_STAGE(s, s * MMA_TF32X3_BK);
            LOAD_B_STAGE(s, s * MMA_TF32X3_BK);
        }
        asm volatile("cp.async.commit_group;\n");
    }

    // hi/lo 分割ヘルパー（レジスタ段。フラグメントロード直後に 1 回だけ
    // 適用。本ファイル冒頭コメント「分割式」参照）: raw のビット
    // パターンを f32 として解釈し、TF32 丸め（hi）→ 残差の再丸め（lo）で
    // 2 語へ分解する。CUTLASS `mma_tensor_op_fast_f32` と同型の
    // split-single 法。
    #define MMA_TF32X3_SPLIT(RAW, HI, LO) \
        do { \
            float mma_tf32x3_v_ = __uint_as_float(RAW); \
            float mma_tf32x3_hi_ = wmma::__float_to_tf32(mma_tf32x3_v_); \
            float mma_tf32x3_lo_ = wmma::__float_to_tf32(mma_tf32x3_v_ - mma_tf32x3_hi_); \
            HI = __float_as_uint(mma_tf32x3_hi_); \
            LO = __float_as_uint(mma_tf32x3_lo_); \
        } while (0)

    for (int t = 0; t < num_k_tiles; ++t) {
        int compute_stage = t % MMA_TF32X3_STAGES;
        int next_tile = t + MMA_TF32X3_STAGES - 1;
        int load_stage = next_tile % MMA_TF32X3_STAGES;

        // `kernels_mma_tf32.rs::MMA_TF32_BODY` 「#492」節と同一の段数
        // 一般形固定即値（`STAGES - 2`）。
        asm volatile("cp.async.wait_group %0;\n" ::"n"(MMA_TF32X3_STAGES - 2));

        // 単発 TF32 経路と異なり smem 上での丸め（CONVERT_*_STAGE_GROUP
        // 相当）は行わない（本ファイル冒頭コメント「位置づけ」参照:
        // 分割はレジスタ段で行うため smem は生の f32 のまま）。この
        // __syncthreads() は cp.async で書き込まれた生 f32 を全 warp の
        // ldmatrix/直接ロードへ公開する（`kernels_mma_tf32.rs` の同型
        // バリアと同じ役割）。
        __syncthreads();

        // A: hi/lo 2 面 x 2 バッファ（先読みダブルバッファ。#495 相当）。
        // B: 同様に hi/lo 2 面 x 2 バッファ。
        unsigned a_hi_frag[2][MMA_TF32X3_WARP_TILES_M][4];
        unsigned a_lo_frag[2][MMA_TF32X3_WARP_TILES_M][4];
        unsigned b_hi_frag[2][MMA_TF32X3_WARP_TILES_N][2];
        unsigned b_lo_frag[2][MMA_TF32X3_WARP_TILES_N][2];

        // A フラグメントロード（`kernels_mma_tf32.rs::LDSM_A_FRAG` と
        // 同一の ldmatrix b16 流用アドレッシング・同一の象限順序
        // TL, BL, TR, BR）。ldmatrix はビットパターンを転置なしで
        // 再配置するのみのため、smem の生 f32 ビットをそのまま
        // `a_raw0..3` へ運ぶ。直後に [`MMA_TF32X3_SPLIT`] で hi/lo へ
        // 分割する。
        #define LDSM_A_FRAG(buf, stage, kstep, mi) \
            do { \
                int a_col_ = (kstep) * MMA_TF32X3_K; \
                int a_row = warp_row * (MMA_TF32X3_M * MMA_TF32X3_WARP_TILES_M) + (mi) * MMA_TF32X3_M; \
                int a_quad_group = lane / 8; \
                int a_quad_row = a_quad_group % 2; \
                int a_quad_col = a_quad_group / 2; \
                int a_row_in_tile = lane % 8; \
                float* a_addr = &as_tile[stage] \
                                          [a_row + a_quad_row * 8 + a_row_in_tile] \
                                          [a_col_ + a_quad_col * 4]; \
                unsigned a_smem = (unsigned)__cvta_generic_to_shared(a_addr); \
                unsigned a_raw0, a_raw1, a_raw2, a_raw3; \
                asm volatile( \
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
                    : "=r"(a_raw0), "=r"(a_raw1), "=r"(a_raw2), "=r"(a_raw3) \
                    : "r"(a_smem) \
                ); \
                MMA_TF32X3_SPLIT(a_raw0, a_hi_frag[buf][mi][0], a_lo_frag[buf][mi][0]); \
                MMA_TF32X3_SPLIT(a_raw1, a_hi_frag[buf][mi][1], a_lo_frag[buf][mi][1]); \
                MMA_TF32X3_SPLIT(a_raw2, a_hi_frag[buf][mi][2], a_lo_frag[buf][mi][2]); \
                MMA_TF32X3_SPLIT(a_raw3, a_hi_frag[buf][mi][3], a_lo_frag[buf][mi][3]); \
            } while (0)

        // B フラグメントロード（`kernels_mma_tf32.rs::LDS_B_FRAG` と
        // 同一の素の共有メモリロード。転置版 ldmatrix 修飾子は不使用）。
        // 直後に hi/lo へ分割する。
        #define LDS_B_FRAG(buf, stage, kstep, nj) \
            do { \
                int b_row0 = (kstep) * MMA_TF32X3_K; \
                int b_col = warp_col * (MMA_TF32X3_N * MMA_TF32X3_WARP_TILES_N) + (nj) * MMA_TF32X3_N; \
                unsigned b_raw0 = __float_as_uint(bs_tile[stage][b_row0 + tid_in_group][b_col + group_id]); \
                unsigned b_raw1 = __float_as_uint(bs_tile[stage][b_row0 + tid_in_group + 4][b_col + group_id]); \
                MMA_TF32X3_SPLIT(b_raw0, b_hi_frag[buf][nj][0], b_lo_frag[buf][nj][0]); \
                MMA_TF32X3_SPLIT(b_raw1, b_hi_frag[buf][nj][1], b_lo_frag[buf][nj][1]); \
            } while (0)

        // #495 warp プロローグ: kstep=0 のフラグメントをバッファ 0 へ
        // ロードしてから kstep ループへ入る。
#pragma unroll
        for (int mi = 0; mi < MMA_TF32X3_WARP_TILES_M; ++mi) {
            LDSM_A_FRAG(0, compute_stage, 0, mi);
        }
#pragma unroll
        for (int nj = 0; nj < MMA_TF32X3_WARP_TILES_N; ++nj) {
            LDS_B_FRAG(0, compute_stage, 0, nj);
        }

        // mma.sync 命令文字列は本マクロの定義サイト 1 箇所のみに閉じ、
        // kstep x mi x nj ループから 3 回ずつ呼び出す（本ファイル冒頭
        // コメント「3 回の累積順序」参照。コピペ増殖の回帰検出テスト
        // 対象）。
        #define MMA_TF32X3_ISSUE(A0, A1, A2, A3, B0, B1, DST) \
            asm volatile( \
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 " \
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n" \
                : "=f"(DST[0]), "=f"(DST[1]), "=f"(DST[2]), "=f"(DST[3]) \
                : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), \
                  "f"(DST[0]), "f"(DST[1]), "f"(DST[2]), "f"(DST[3]) \
            )

#pragma unroll
        for (int kstep = 0; kstep < MMA_TF32X3_BK / MMA_TF32X3_K; ++kstep) {
            int cur = kstep % 2;
            int nxt = (kstep + 1) % 2;

            // #495 相当: 次段（kstep+1）のフラグメントを先読みする
            // （`kernels_mma_tf32.rs::MMA_TF32_BODY` 「#495」節と同一
            // 設計）。
            if (kstep + 1 < MMA_TF32X3_BK / MMA_TF32X3_K) {
#pragma unroll
                for (int mi = 0; mi < MMA_TF32X3_WARP_TILES_M; ++mi) {
                    LDSM_A_FRAG(nxt, compute_stage, kstep + 1, mi);
                }
#pragma unroll
                for (int nj = 0; nj < MMA_TF32X3_WARP_TILES_N; ++nj) {
                    LDS_B_FRAG(nxt, compute_stage, kstep + 1, nj);
                }
            }

            // #496 相当: cp.async issue interleaving（`kernels_mma_tf32.rs
            // ::MMA_TF32_BODY` 「#496」節と同一設計）。
            if (next_tile < num_k_tiles) {
                LOAD_A_STAGE_GROUP(load_stage, next_tile * MMA_TF32X3_BK, kstep);
                LOAD_B_STAGE_GROUP(load_stage, next_tile * MMA_TF32X3_BK, kstep);
            }

            // mi x nj の通りで mma.sync を 3 回（hi·lo → lo·hi → hi·hi）
            // 発行し d[mi][nj] へ累積する（本ファイル冒頭コメント「3 回
            // の累積順序」参照。`lo·lo` は省略）。
#pragma unroll
            for (int mi = 0; mi < MMA_TF32X3_WARP_TILES_M; ++mi) {
#pragma unroll
                for (int nj = 0; nj < MMA_TF32X3_WARP_TILES_N; ++nj) {
                    MMA_TF32X3_ISSUE(
                        a_hi_frag[cur][mi][0], a_hi_frag[cur][mi][1],
                        a_hi_frag[cur][mi][2], a_hi_frag[cur][mi][3],
                        b_lo_frag[cur][nj][0], b_lo_frag[cur][nj][1],
                        d[mi][nj]);
                    MMA_TF32X3_ISSUE(
                        a_lo_frag[cur][mi][0], a_lo_frag[cur][mi][1],
                        a_lo_frag[cur][mi][2], a_lo_frag[cur][mi][3],
                        b_hi_frag[cur][nj][0], b_hi_frag[cur][nj][1],
                        d[mi][nj]);
                    MMA_TF32X3_ISSUE(
                        a_hi_frag[cur][mi][0], a_hi_frag[cur][mi][1],
                        a_hi_frag[cur][mi][2], a_hi_frag[cur][mi][3],
                        b_hi_frag[cur][nj][0], b_hi_frag[cur][nj][1],
                        d[mi][nj]);
                }
            }
        }

        #undef LDSM_A_FRAG
        #undef LDS_B_FRAG
        #undef MMA_TF32X3_ISSUE

        // `kernels_mma_tf32.rs::MMA_TF32_BODY` 「#492/#496」節と同一の
        // 「1 イテレーション = 必ず 1 commit」不変条件。
        asm volatile("cp.async.commit_group;\n");
        __syncthreads();
    }

    #undef MMA_TF32X3_SPLIT

    // ループ外 drain（`kernels_mma_tf32.rs::MMA_TF32_BODY` と同一の
    // 正しさ論証）。
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    #undef LOAD_A_STAGE
    #undef LOAD_B_STAGE
    #undef LOAD_A_STAGE_GROUP
    #undef LOAD_B_STAGE_GROUP
    #undef A_GROUP_CHUNKS
    #undef B_GROUP_CHUNKS
    #undef A_CHUNKS
    #undef B_CHUNKS
    #undef K_GROUPS

    // REQ-8: エピローグの guarded store。mma.m16n8k8 の C/D フラグメント
    // レーン対応は f16 m16n8k16 版・単発 TF32 版と同一形（本ファイル
    // 冒頭コメント「命令選定」参照）: d0/d1 は行 groupID、d2/d3 は行
    // groupID+8。
#pragma unroll
    for (int mi = 0; mi < MMA_TF32X3_WARP_TILES_M; ++mi) {
#pragma unroll
        for (int nj = 0; nj < MMA_TF32X3_WARP_TILES_N; ++nj) {
            int r0 = row0_warp + mi * MMA_TF32X3_M + group_id;
            int r1 = row0_warp + mi * MMA_TF32X3_M + group_id + 8;
            int c0 = col0_warp + nj * MMA_TF32X3_N + tid_in_group * 2;
            int c1 = c0 + 1;

            if (r0 < m && c0 < n) c[(size_t)r0 * n + c0] = d[mi][nj][0];
            if (r0 < m && c1 < n) c[(size_t)r0 * n + c1] = d[mi][nj][1];
            if (r1 < m && c0 < n) c[(size_t)r1 * n + c0] = d[mi][nj][2];
            if (r1 < m && c1 < n) c[(size_t)r1 * n + c1] = d[mi][nj][3];
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// カーネルソースが `mma.sync`・`ldmatrix`・`cp.async` の主要命令を
    /// 実在させることを検査する（`kernels_mma_tf32.rs::
    /// mma_tf32_source_uses_mma_sync_ldmatrix_cp_async_instructions` と
    /// 同型）。
    #[test]
    fn mma_tf32x3_source_uses_mma_sync_ldmatrix_cp_async_instructions() {
        let src = mma_tf32x3_source();
        for needle in [
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "cp.async.wait_group",
        ] {
            assert!(
                src.contains(needle),
                "expected kernel source to contain {needle:?}"
            );
        }
        assert!(
            !src.contains(".trans"),
            "TF32x3 mma.sync kernel must not use `.trans` ldmatrix for the B \
             operand (32bit tf32 elements would be split by b16-granularity \
             transpose)"
        );
    }

    /// mma.sync の命令文字列が単一マクロ定義サイトのみに存在し
    /// （コピペ増殖の回帰検出）、その定義サイトが 3 回呼び出されている
    /// ことを検査する（本ファイル冒頭コメント「3 回の累積順序」参照）。
    #[test]
    fn mma_tf32x3_source_issues_mma_sync_from_single_macro_site_called_three_times() {
        let src = mma_tf32x3_source();
        let instr_count = src
            .matches("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32")
            .count();
        assert_eq!(
            instr_count, 1,
            "expected exactly 1 occurrence of the mma.sync instruction text \
             (single macro definition site), found {instr_count}"
        );

        // `MMA_TF32X3_ISSUE(` の総出現数には `#define MMA_TF32X3_ISSUE(...)`
        // の定義サイト自体も 1 件含まれるため、呼び出しサイト数は
        // 総数から定義サイト数（常に 1）を差し引いた値になる。
        let total_occurrences = src.matches("MMA_TF32X3_ISSUE(").count();
        let definition_sites = src.matches("#define MMA_TF32X3_ISSUE(").count();
        assert_eq!(
            definition_sites, 1,
            "expected exactly 1 MMA_TF32X3_ISSUE macro definition site, found {definition_sites}"
        );
        let call_count = total_occurrences - definition_sites;
        assert_eq!(
            call_count, 3,
            "expected exactly 3 call sites of MMA_TF32X3_ISSUE (hi·lo, lo·hi, \
             hi·hi accumulation), found {call_count}"
        );
    }

    /// hi/lo 分割ヘルパー（`wmma::__float_to_tf32`）が
    /// `MMA_TF32X3_SPLIT` マクロ定義内のみに存在することを検査する
    /// （smem ステージング時の丸めを行わない設計。本ファイル冒頭コメント
    /// 「位置づけ」参照）。
    #[test]
    fn mma_tf32x3_source_rounds_only_in_split_macro() {
        let src = mma_tf32x3_source();
        let round_count = src.matches("wmma::__float_to_tf32(").count();
        assert_eq!(
            round_count, 2,
            "expected exactly 2 wmma::__float_to_tf32 call sites (hi, lo) \
             inside MMA_TF32X3_SPLIT, found {round_count}"
        );

        // 単発 TF32 経路が持つ smem ステージング時丸めマクロ
        // （CONVERT_A_STAGE_GROUP/CONVERT_B_STAGE_GROUP）が本ファイルには
        // 存在しないことを確認する（本ファイル冒頭コメント「位置づけ」
        // の設計差分）。
        assert!(!src.contains("CONVERT_A_STAGE_GROUP"));
        assert!(!src.contains("CONVERT_B_STAGE_GROUP"));
    }

    /// REQ-8 手動境界チェック（guarded load のゼロ充填分岐・guarded
    /// store）がソース中に存置されていることを検査する（`kernels_mma_
    /// tf32.rs::mma_tf32_source_retains_req8_boundary_checks` と同型）。
    #[test]
    fn mma_tf32x3_source_retains_req8_boundary_checks() {
        let src = mma_tf32x3_source();
        for needle in [
            "int valid = (gr < m && gc < k) ? 16 : 0;",
            "int valid = (gr < k && gc < n) ? 16 : 0;",
            "if (r0 < m && c0 < n)",
            "if (r1 < m && c1 < n)",
        ] {
            assert!(
                src.contains(needle),
                "expected REQ-8 boundary check text {needle:?} to remain in the kernel source"
            );
        }
    }

    /// `cp.async.commit_group`/`cp.async.wait_group` の発行サイト数が
    /// 単発 TF32 経路と同一の構造的不変条件を保っていることを検査する
    /// （`kernels_mma_tf32.rs::mma_tf32_source_commit_wait_group_invariant`
    /// と同型）。
    #[test]
    fn mma_tf32x3_source_commit_wait_group_invariant() {
        let src = mma_tf32x3_source();
        let commit_count = src.matches("cp.async.commit_group;").count();
        assert_eq!(commit_count, 2);
        let wait_count = src.matches("cp.async.wait_group").count();
        assert_eq!(wait_count, 2);
    }

    /// タイル定数が単発 TF32 経路と完全に一致する（エイリアス）ことを
    /// 実行時にも再確認する（本ファイル冒頭コメント「タイル構成の
    /// エイリアス」参照）。
    #[test]
    fn mma_tf32x3_tile_constants_alias_mma_tf32() {
        assert_eq!(MMA_TF32X3_BM, kernels_mma_tf32::MMA_TF32_BM);
        assert_eq!(MMA_TF32X3_BN, kernels_mma_tf32::MMA_TF32_BN);
        assert_eq!(MMA_TF32X3_BK, kernels_mma_tf32::MMA_TF32_BK);
        assert_eq!(MMA_TF32X3_STAGES, kernels_mma_tf32::MMA_TF32_STAGES);
        assert_eq!(
            MMA_TF32X3_SHARED_MEM_BYTES,
            kernels_mma_tf32::MMA_TF32_SHARED_MEM_BYTES
        );
        assert_eq!(MMA_TF32X3_SHARED_MEM_BYTES, 28_416);
        assert_eq!(MMA_TF32X3_BLOCK_THREADS, 128);
        assert_eq!(MMA_TF32X3_WARP_M, kernels_mma_tf32::MMA_TF32_WARP_M);
        assert_eq!(MMA_TF32X3_WARP_N, kernels_mma_tf32::MMA_TF32_WARP_N);
        assert_eq!(MMA_TF32X3_WARPS_M, kernels_mma_tf32::MMA_TF32_WARPS_M);
        assert_eq!(
            MMA_TF32X3_K_STEPS_PER_STAGE,
            kernels_mma_tf32::MMA_TF32_K_STEPS_PER_STAGE
        );
    }
}
