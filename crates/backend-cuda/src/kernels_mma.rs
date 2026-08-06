//! f16 `mma.sync`/`ldmatrix`/`cp.async` GEMM の CUDA カーネルソース
//! （NVRTC 実行時コンパイル用の静的文字列。TASK-11.1h・#187）。
//!
//! `kernels_wmma.rs`（#61）が WMMA C++ API（`<mma.h>`）の 1 ブロック =
//! 1 warp = fragment 1 個という最小構成だったのに対し、本モジュールは
//! `docs/cuda-tensor-core-design.md` 3.2 節が「方式 B（PTX）への段階移行」
//! と位置づける低レベル経路（`mma.sync` PTX 直叩き・`ldmatrix` によるレジスタ
//! ロード・`cp.async` によるグローバル→共有メモリの非同期多段パイプライン）
//! を実装する。`kernels.rs`（naive／tiled）・`kernels_wmma.rs`・`gemm.rs`・
//! `gemm_wmma.rs` とは意図的に別ファイルへ分離しており（並行 issue #62/#63
//! が上記ファイルを編集中のため。実装計画 4 節）、本クレートのディスパッチ
//! （どの経路をいつ選ぶか）は TASK-11.2（#66）のスコープで未実装のまま残す。
//!
//! # 検証状態（重要）
//!
//! 実装セッションの環境には CUDA **driver**（`libcuda`。compute capability
//! 8.6・RTX 3060 実機）は存在するが、NVRTC（`libnvrtc`）は存在せず
//! `nvrtc::compile_ptx` は `CudaError::NvrtcUnavailable` を返す
//! （`crates/backend-cuda/tests/gemm_mma.rs` の環境適応テストがこの分岐を
//! green として扱う）。したがって本ファイルの CUDA C++ ／インライン PTX
//! ソースは **NVRTC による構文検証を一度も通過していない**。sm_121（DGX
//! Spark GB10）はおろか、この実機（sm_86）上でも未検証である。実機での
//! 最初の実行が構文検証を兼ねる（`docs/perf/metal-gemm-dynamic-tile.md`
//! の先例と同じ位置づけ）。詳細は `docs/perf/cuda-gemm-mma-pipeline.md`。
//!
//! # タイル構成（実装計画 3.2 節からの意図的な縮小）
//!
//! 実装計画の候補値（ブロックタイル 128x128・BK=32・3 ステージ）は
//! 静的共有メモリの上限（全 compute capability 共通の per-block 48KiB。
//! 動的共有メモリ opt-in `cudaFuncSetAttribute` を追加で呼ばない限り
//! 超過するとコンパイル・起動が失敗する）に対して余裕がなく
//! （128x32+32x128）x2Bx3 ≈ 49152B = ちょうど 48KiB）、コンパイル検証が
//! できない本セッションでは危険側に倒れる。よって本実装は
//! `BM=32`・`BN=64`・`BK=32`・3 ステージ（共有メモリ 18432B ≈ 18KiB）に
//! 縮小し、さらに 1 warp = C の `MMA_M x MMA_N`（`16x8`）タイル 1 個のみを
//! 担当する構成（warp 内での M/N 方向の追加タイルループを持たない）とする。
//! `kernels_wmma.rs` 冒頭コメントの「実機未接続・コンパイル未検証による
//! リスク最小化」判断をそのまま踏襲する（実装計画 8 節「リスク」・
//! アドバイザレビューで確認済みの判断）。ブロックタイル拡大・warp あたり
//! 複数 mma タイル化・レジスタブロッキングは後続（#65 へ知見を記録し、
//! 拡張は #63 と同種のスコープとして引き継ぐ）。
//!
//! XOR swizzle（実装計画「段階 3」）は不採用とする。索引演算が最も
//! 複雑でありながらコンパイル未検証環境では誤りを検出できないため、
//! バンクコンフリクト低減は将来の性能最適化（実測可能な環境）へ明示的に
//! 先送りする（out-of-scope-tracking.md に従い記録。本ファイル末尾
//! 参照）。
//!
//! # 命令選定・sm_80+ ゲート
//!
//! `cp.async`・`ldmatrix` は compute capability 8.0 以降を要求する
//! （`nvidia-cuda` スキル `references/advanced/features/async-copies.md`
//! 「LDGSTS (CC 8.0+)」）。`gemm_mma.rs::CudaMmaGemm::new` は
//! `MIN_COMPUTE_CAPABILITY_MAJOR = 8` で NVRTC コンパイル前にこれを検査する
//! （`kernels_wmma.rs` の cc>=7.0 ゲートと同じ設計。WMMA 経路（cc>=7.0）とは
//! 独立した下限）。sm_121 は Ampere 系譜の `mma.sync`/`cp.async` プログラミング
//! モデルを維持する（設計メモ 2 節・3.3 節）。
//!
//! `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` は f16 tensor core
//! 経路の標準 mma shape（sm_80+）である。A フラグメントは
//! `ldmatrix.sync.aligned.m8n8.x4.shared.b16`、B フラグメントは
//! `.trans` 修飾子付きの `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16`
//! で共有メモリからレジスタへロードする（B は mma 側で `.col` 配置を
//! 要求するため、共有メモリ上は自然な row-major（k x n）のまま `.trans`
//! ロードで整合させる。CUTLASS・公開の mma.sync チュートリアル群で
//! 標準的に使われる組み合わせ）。
//!
//! # 整列制約（cp.async 16 バイト境界。ホスト側 `gemm_mma.rs` が検証）
//!
//! `cp.async.cg.shared.global` は 1 回のコピー粒度を 16 バイト（f16 8 要素）
//! に固定し、転送元・転送先双方が 16 バイト境界に整列している必要がある。
//! 本カーネルの共有メモリ側は `BK`/`BN` が共に 8 の倍数であるため常に整列
//! するが、グローバル側は行ストライド（A は `k`、B は `n`）が 8 の倍数で
//! ない場合、行境界をまたいだ列オフセットが 16 バイト整列しない可能性が
//! ある。よって `gemm_mma.rs::CudaMmaGemm::run_f16` はホスト側で
//! `k % 8 == 0 && n % 8 == 0` を追加検証し（`gemm.rs::validate_tiled_k_bound`
//! と同種の経路固有追加検証パターン）、満たさない形状は
//! `CudaError::InvalidShape` で拒否する。この制約下では K/N 方向のタイル
//! 境界チェックが「8 要素チャンク全体が有効か無効か」の二値になり
//! （`k`/`n` 自体が 8 の倍数のため、チャンク途中で境界を跨がない）、
//! 境界検査の実装を単純化できる（本ファイル冒頭コメント「境界検査」参照）。
//! `m` 方向には整列制約を課さない（行方向の可変長はゼロ充填のみで済む）。
//!
//! # 境界検査（REQ-8。省略禁止）
//!
//! 1. **A/B タイルの `cp.async` ロード**: グローバル→共有メモリのコピーは
//!    `cp.async.cg.shared.global [dst], [src], 16, src_size;` の
//!    src-size オペランドを使う。範囲外チャンクは `src_size = 0` を渡し
//!    （実際のグローバル読み出しを発生させず）、共有メモリ側を丸ごと
//!    ゼロ充填する。アドレス計算自体は `m-1`/`k-1` にクランプした添字を
//!    使い、ポインタが確保済み範囲外を指さないようにする（範囲外
//!    メモリへの実読み出し・境界外ポインタ生成のいずれも避ける）。
//! 2. **エピローグの guarded store**: mma アキュムレータ（`d0..d3`）を
//!    グローバル C へ書き戻す際、`(gr < m && gc < n)` を満たす要素のみ
//!    書き込む（範囲外書き込みを発生させない）。
//! 3. ホスト側 `gemm_mma.rs::CudaMmaGemm::run_f16` は起動前に
//!    `gemm::validate_gemm_dims`（i32 積ガード含む）と上記整列検証の
//!    両方を必ず先行させる。
//!
//! # 数値契約
//!
//! f16 入出力・f32 内部アキュムレートは `kernels_wmma.rs::WMMA_F16` と
//! 同一方針（`.claude/rules/coding-rust.md` FMA 契約統一節）。

/// mma 命令 1 回あたりの行列形状（`m16n8k16`。sm_80+ の f16 標準 shape）。
pub const MMA_M: u32 = 16;
pub const MMA_N: u32 = 8;
pub const MMA_K: u32 = 16;

/// ブロックタイル（本ファイル冒頭コメント「タイル構成」参照。実装計画
/// 3.2 節候補値からの意図的な縮小）。
pub const MMA_BM: u32 = 32;
pub const MMA_BN: u32 = 64;
pub const MMA_BK: u32 = 32;

/// `cp.async` multi-stage pipelining のステージ数。共有メモリ使用量
/// `(MMA_BM*MMA_BK + MMA_BK*MMA_BN) * 2B * MMA_STAGES` = 18432B（18KiB）
/// で per-block 48KiB 上限に対し十分な余裕を持つ（本ファイル冒頭コメント
/// 参照）。
pub const MMA_STAGES: u32 = 3;

/// 1 ブロックあたりの warp 構成（M 方向 2・N 方向 8 = 16 warp = 512 スレッド）。
/// 1 warp が C の `MMA_M x MMA_N` タイル 1 個のみを担当する
/// （本ファイル冒頭コメント「タイル構成」参照）。
pub const MMA_WARPS_M: u32 = MMA_BM / MMA_M; // 2
pub const MMA_WARPS_N: u32 = MMA_BN / MMA_N; // 8

/// ブロック内スレッド総数（32 スレッド/warp x warp 数）。
pub const MMA_BLOCK_THREADS: u32 = MMA_WARPS_M * MMA_WARPS_N * 32;

/// `cp.async.wait_group` の非最終タイル向け即値。`MMA_STAGES - 2` に
/// 一致する必要がある（プロローグで `MMA_STAGES - 1` グループを commit
/// した後、最古のグループの完了を待つには「直近 `MMA_STAGES - 2`
/// グループの未完了を許容する」`wait_group` 即値が必要。標準的な
/// ソフトウェアパイプラインの式）。最終 K タイルでは新規 commit が発生
/// しないため、この即値では最後のグループの完了を保証できず
/// `wait_group 0` による drain が別途必要（カーネルソース `if (t ==
/// num_k_tiles - 1)` 分岐。PR #255 レビュー指摘）。`MMA_STAGES` を
/// 変更する場合、カーネルソース中の `cp.async.wait_group 1;`／`0;` の
/// 即値と分岐条件もあわせて見直すこと。`gemm_mma.rs` が起動前の
/// `debug_assert` で参照し、`MMA_STAGES` の実利用を兼ねる。
pub const MMA_WAIT_GROUP_IMMEDIATE: u32 = MMA_STAGES - 2;

/// 1 ステージあたりの `mma.sync` 呼び出し回数（`BK / MMA_K`。カーネル内
/// `for (int kstep = 0; kstep < BK / MMA_K; ++kstep)` に対応する Rust 側の
/// 唯一の真実源）。`gemm_mma.rs` が起動前の `debug_assert` で参照する。
pub const MMA_K_STEPS_PER_STAGE: u32 = MMA_BK / MMA_K;

/// 静的共有メモリ使用量（バイト）。`(MMA_BM*MMA_BK + MMA_BK*MMA_BN) * 2B
/// (f16) * MMA_STAGES`。全 compute capability 共通の per-block 静的共有
/// メモリ上限（49152 バイト = 48KiB）に対する実使用量を下記
/// `const _: () = assert!(...)` でコンパイル時に検査する（本ファイル冒頭
/// コメント「タイル構成」参照。タイル定数変更時に即座にビルドエラーで
/// 検出できるよう、実行時 `debug_assert` ではなくコンパイル時定数
/// アサーションとする）。
pub const MMA_SHARED_MEM_BYTES: u32 = (MMA_BM * MMA_BK + MMA_BK * MMA_BN) * 2 * MMA_STAGES;

// コンパイル時契約検査（タイル定数の内部整合性。実機コンパイルできない
// 環境でも `cargo build` の時点で機械検出できる代替チェック。本ファイル
// 冒頭コメント「タイル構成」参照）。
const _: () = assert!(
    MMA_SHARED_MEM_BYTES <= 49_152,
    "kernels_mma::MMA_F16 static shared memory exceeds the 48KiB per-block \
     limit shared by every compute capability"
);
const _: () = assert!(
    MMA_BK.is_multiple_of(MMA_K),
    "MMA_BK must be a multiple of MMA_K (kernel-side kstep loop divisibility)"
);
const _: () = assert!(
    MMA_BN.is_multiple_of(8) && MMA_BM.is_multiple_of(8),
    "MMA_BM/MMA_BN must be multiples of 8 (cp.async 16-byte transfer granularity)"
);
const _: () = assert!(
    MMA_BLOCK_THREADS <= 1024,
    "MMA_BLOCK_THREADS must not exceed CUDA's per-block thread limit (1024)"
);
// カーネルソース内 `if (t == num_k_tiles - 1) { wait_group 0 } else { wait_group 1 }`
// の二値分岐（本ファイル冒頭「命令選定」・`MMA_WAIT_GROUP_IMMEDIATE`
// ドキュメンテーションコメント参照）は `MMA_STAGES = 3` の下でのみ正しい
// （一般には `wait_group` の必要値は `min(MMA_STAGES-2, num_k_tiles-t-1)`
// であり、`MMA_STAGES > 3` では末尾の中間値をこの二値分岐では表現でき
// ない）。`debug_assert_eq!`（`gemm_mma.rs::CudaMmaGemm::new`）はデバッグ
// ビルドでのみ検査するのに対し、こちらはリリースビルドでも即座に
// ビルドエラーとして検出する（PR #255 レビュー指摘。実機コンパイル
// できないセッションでの安全側の追加ガード）。
const _: () = assert!(
    MMA_STAGES == 3,
    "kernels_mma::MMA_F16 の cp.async drain 分岐は MMA_STAGES=3 前提の \
     二値分岐（if (t == num_k_tiles - 1)）のため、MMA_STAGES を変更する \
     場合はカーネルソース側の wait_group 分岐ロジックも合わせて見直すこと"
);

/// f16 `mma.sync`/`ldmatrix`/`cp.async` GEMM（f16 入出力・f32 アキュムレート）。
///
/// ホスト側（`gemm_mma.rs::CudaMmaGemm`）はこの文字列を `nvrtc::compile_ptx`
/// に渡して `CudaFunction` を得る。カーネルソースはコンパイル時定数の
/// まま埋め込み、ビルド時に nvcc／CUDA ヘッダを要求しない契約
/// （`.claude/rules/deps-policy.md`）を維持する（`kernels_wmma.rs` と同じ
/// 方針）。
pub const MMA_F16: &str = r#"
#include <cuda_fp16.h>

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define BM 32
#define BN 64
#define BK 32
#define WARPS_N 8
#define STAGES 3

// REQ-8: グローバル→共有メモリの 16 バイト単位コピー。src_size==16 で
// 実データをコピーし、src_size==0 で実際のグローバル読み出しを発生させず
// 共有メモリ側を丸ごとゼロ充填する（本ファイル冒頭コメント「境界検査」参照）。
__device__ __forceinline__ void mma_cp_async16(void* smem_ptr, const void* gmem_ptr, int src_size)
{
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr), "r"(src_size)
    );
}

extern "C" __global__ void gemm_mma_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m, int n, int k)
{
    // __align__(16): cp.async の 16 バイト転送先整列要件（本ファイル冒頭
    // コメント「整列制約」）。BK/BN が 8 の倍数のため各行の先頭は常に
    // 16 バイト整列する。
    __shared__ __align__(16) __half as_tile[STAGES][BM][BK];
    __shared__ __align__(16) __half bs_tile[STAGES][BK][BN];

    int block_row0 = blockIdx.y * BM;
    int block_col0 = blockIdx.x * BN;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int warp_row = warp_id / WARPS_N;
    int warp_col = warp_id % WARPS_N;
    int row0_warp = block_row0 + warp_row * MMA_M;
    int col0_warp = block_col0 + warp_col * MMA_N;

    // mma.m16n8k16 のレーン→フラグメント要素対応（PTX ISA の標準
    // groupID/threadID_in_group 分解。本ファイル冒頭コメント「命令選定」）。
    int group_id = lane / 4;
    int tid_in_group = lane % 4;

    // C アキュムレータ（f32 x4。1 warp = 1 mma タイルのみ担当のため
    // 単一フラグメントで足りる。本ファイル冒頭コメント「タイル構成」）。
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    int num_k_tiles = (k > 0) ? (k - 1) / BK + 1 : 0;

    // REQ-8: A/B タイルを stage へ非同期ロードする。gr/gc は呼び出し側で
    // クランプ済みの添字（境界外ポインタを作らないため）。valid は実際の
    // コピーサイズ（16 or 0）を選ぶだけで、アドレス自体は常に確保済み
    // 範囲内を指す。
    // REQ-8 追補（PR #255 レビュー指摘）: 範囲外チャンク（valid=0）でも
    // `cp.async.cg.shared.global` のソースアドレスは常に 16 バイト境界に
    // 整列している必要がある（size=0 でもアドレス自体の整列制約は緩和
    // されない。cp.async の未定義動作を避けるための PTX 側の要件）。行
    // ストライド（A は k、B は n）はホスト側 `gemm_mma.rs::run_f16` が
    // カーネル起動前に必ず `validate_mma_alignment` を経由させることで
    // 8 の倍数であることを保証するため行方向のクランプ（gr_c）は整列に
    // 影響しないが、列方向のクランプ（gc_c）を単純に `k-1`/`n-1` にすると
    // 8 要素境界からずれる。よって直近の 8 要素境界（`((k-1)/8)*8` など）
    // に切り下げてクランプする。この gr_c 側の整列不問という前提は
    // `validate_mma_alignment` が起動前に必ず通ることに依存しており、
    // `run_f16` 側でこの検証呼び出しを外す・順序を変える場合は本コメント
    // ごと見直すこと。
    #define LOAD_A_STAGE(stage, k0) \
        for (int idx = tid; idx < (BM * BK) / 8; idx += blockDim.x) { \
            int row = idx / (BK / 8); \
            int col0 = (idx % (BK / 8)) * 8; \
            int gr = block_row0 + row; \
            int gc = (k0) + col0; \
            int gr_c = gr < m ? gr : (m > 0 ? m - 1 : 0); \
            int gc_c = gc < k ? gc : (k > 0 ? ((k - 1) / 8) * 8 : 0); \
            int valid = (gr < m && gc < k) ? 16 : 0; \
            mma_cp_async16(&as_tile[stage][row][col0], &a[(size_t)gr_c * k + gc_c], valid); \
        }

    #define LOAD_B_STAGE(stage, k0) \
        for (int idx = tid; idx < (BK * BN) / 8; idx += blockDim.x) { \
            int row = idx / (BN / 8); \
            int col0 = (idx % (BN / 8)) * 8; \
            int gr = (k0) + row; \
            int gc = block_col0 + col0; \
            int gr_c = gr < k ? gr : (k > 0 ? k - 1 : 0); \
            int gc_c = gc < n ? gc : (n > 0 ? ((n - 1) / 8) * 8 : 0); \
            int valid = (gr < k && gc < n) ? 16 : 0; \
            mma_cp_async16(&bs_tile[stage][row][col0], &b[(size_t)gr_c * n + gc_c], valid); \
        }

    // プロローグ: 最初の STAGES-1 タイルをロードし、それぞれ独立した
    // cp.async グループとして commit する（標準的なソフトウェア
    // パイプライン初期化。本ファイル冒頭コメント「命令選定」参照）。
    for (int s = 0; s < STAGES - 1 && s < num_k_tiles; ++s) {
        LOAD_A_STAGE(s, s * BK);
        LOAD_B_STAGE(s, s * BK);
        asm volatile("cp.async.commit_group;\n");
    }

    for (int t = 0; t < num_k_tiles; ++t) {
        int compute_stage = t % STAGES;
        int next_tile = t + STAGES - 1;
        int load_stage = next_tile % STAGES;

        // MMA_STAGES=3 前提の即値（Rust 側 `kernels_mma::MMA_WAIT_GROUP_IMMEDIATE`
        // 参照。`gemm_mma.rs::CudaMmaGemm::new` の `debug_assert_eq!` が
        // この即値との対応を検査する）。最古の commit 済みグループ
        // （compute_stage に対応）の完了を保証する。
        //
        // 最終 K タイル（t == num_k_tiles - 1）では下の
        // `if (next_tile < num_k_tiles)` が false のまま新規 commit が
        // 発生しないため、`wait_group 1` のままだと最後の cp.async
        // グループの完了を待たずに ldmatrix/mma.sync が共有メモリを読み
        // うる（PR #255 レビュー指摘。k<=BK の小 K・16x8x16 smoke test で
        // 即座に発生しうるレースコンディション）。最終タイルのみ
        // `wait_group 0`（全 outstanding グループの完了待ち）で drain する。
        // `MMA_WAIT_GROUP_IMMEDIATE`（`MMA_STAGES - 2` = 1）は
        // `MMA_STAGES = 3` 固定の下でのみ「最終タイル以外は 1」が正しい値
        // になる関係にあり、`MMA_STAGES` を変える場合はこの二値分岐自体を
        // 見直す必要がある。
        if (t == num_k_tiles - 1) {
            asm volatile("cp.async.wait_group 0;\n");
        } else {
            asm volatile("cp.async.wait_group 1;\n");
        }
        __syncthreads();

        for (int kstep = 0; kstep < BK / MMA_K; ++kstep) {
            int a_row = warp_row * MMA_M;
            int a_col = kstep * MMA_K;
            int b_row = kstep * MMA_K;
            int b_col = warp_col * MMA_N;

            // A フラグメント（16x16）: ldmatrix.x4（4 個の 8x8 b16 サブ
            // タイルを 1 命令でロード。本ファイル冒頭コメント「命令選定」）。
            // ldmatrix.x4 はレーン群 0-7/8-15/16-23/24-31 の順で出力
            // レジスタ a0/a1/a2/a3 を埋めるが、mma.m16n8k16 が要求する
            // A フラグメントの象限順序は TL/BL/TR/BR（PTX ISA
            // mma.m16n8k16 A フラグメントレイアウト）である。行を
            // レーン群の下位ビット、列を上位ビットへ割り当てることで
            // a0=TL, a1=BL, a2=TR, a3=BR の順を作る（PR #255 レビュー
            // 指摘。逆に取ると a1/a2 に TR/BL が入れ替わって載り、
            // K/M ハーフが入れ替わった不正な結果になる）。
            int a_quad_row = (lane / 8) % 2; // 0,1,0,1 -> TL,BL,TR,BR の行
            int a_quad_col = (lane / 8) / 2; // 0,0,1,1 -> TL,BL,TR,BR の列
            int a_row_in_tile = lane % 8;
            __half* a_addr = &as_tile[compute_stage]
                                      [a_row + a_quad_row * 8 + a_row_in_tile]
                                      [a_col + a_quad_col * 8];
            unsigned a_smem = (unsigned)__cvta_generic_to_shared(a_addr);
            unsigned a0, a1, a2, a3;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                : "r"(a_smem)
            );

            // B フラグメント（16x8。k x n の row-major 格納から `.trans`
            // ロードで mma の `.col` 要求配置へ変換。本ファイル冒頭
            // コメント「命令選定」）。
            int b_row_in_tile = lane % 8;
            int b_quad = lane / 8; // 0..1 のみ使用（x2）
            __half* b_addr = &bs_tile[compute_stage]
                                      [b_row + (b_quad % 2) * 8 + b_row_in_tile]
                                      [b_col];
            unsigned b_smem = (unsigned)__cvta_generic_to_shared(b_addr);
            unsigned b0, b1;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b0), "=r"(b1)
                : "r"(b_smem)
            );

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(d0), "f"(d1), "f"(d2), "f"(d3)
            );
        }

        if (next_tile < num_k_tiles) {
            LOAD_A_STAGE(load_stage, next_tile * BK);
            LOAD_B_STAGE(load_stage, next_tile * BK);
            asm volatile("cp.async.commit_group;\n");
        }
        __syncthreads();
    }

    #undef LOAD_A_STAGE
    #undef LOAD_B_STAGE

    // REQ-8: エピローグの guarded store。mma.m16n8k16 の C/D フラグメント
    // レーン対応（groupID/threadID_in_group。本ファイル冒頭コメント
    // 「命令選定」）: d0/d1 は行 groupID、d2/d3 は行 groupID+8。
    int r0 = row0_warp + group_id;
    int r1 = row0_warp + group_id + 8;
    int c0 = col0_warp + tid_in_group * 2;
    int c1 = c0 + 1;

    if (r0 < m && c0 < n) c[(size_t)r0 * n + c0] = __float2half(d0);
    if (r0 < m && c1 < n) c[(size_t)r0 * n + c1] = __float2half(d1);
    if (r1 < m && c0 < n) c[(size_t)r1 * n + c0] = __float2half(d2);
    if (r1 < m && c1 < n) c[(size_t)r1 * n + c1] = __float2half(d3);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// Rust 側タイル定数が CUDA ソース内 `#define` と食い違わないことを
    /// 検査する（`kernels_wmma.rs::wmma_tile_constant_matches_kernel_source_defines`
    /// と同じ方針。値の不一致はコンパイルエラーにならず誤った積和結果を
    /// 静かに生成しうるため CI 上で機械検出する）。
    #[test]
    fn mma_tile_constants_match_kernel_source_defines() {
        for (name, value) in [
            ("MMA_M", MMA_M),
            ("MMA_N", MMA_N),
            ("MMA_K", MMA_K),
            ("BM", MMA_BM),
            ("BN", MMA_BN),
            ("BK", MMA_BK),
            ("STAGES", MMA_STAGES),
            ("WARPS_N", MMA_WARPS_N),
        ] {
            let expected = format!("#define {name} {value}");
            assert!(
                MMA_F16.contains(&expected),
                "MMA_F16 の `#define {name}` が Rust 側定数（{value}）と一致しません"
            );
        }
    }

    /// TASK-11.3（tensor core 命令使用の証跡）を兼ねる: `mma.sync`・
    /// `ldmatrix`・`cp.async` の主要命令がソース文字列内に実在することを
    /// ロックする（`kernels_wmma.rs::wmma_f16_source_uses_wmma_instructions`
    /// と同じ方針）。
    #[test]
    fn mma_f16_source_uses_mma_sync_ldmatrix_cp_async_instructions() {
        for needle in [
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "cp.async.wait_group",
        ] {
            assert!(
                MMA_F16.contains(needle),
                "MMA_F16 に tensor core 命令 `{needle}` が見つかりません"
            );
        }
    }

    /// REQ-8: A/B タイルの `cp.async` src-size ゼロ充填・エピローグ
    /// guarded store の手動境界チェックが除去されていないことをロックする
    /// （`kernels_wmma.rs` の REQ-8 テスト方針と同様、性能最適化を理由に
    /// 境界検査が省略される回帰を防ぐ）。
    #[test]
    fn mma_f16_source_retains_req8_boundary_guards() {
        for needle in [
            "gr < m && gc < k",
            "gr < k && gc < n",
            "r0 < m && c0 < n",
            "r0 < m && c1 < n",
            "r1 < m && c0 < n",
            "r1 < m && c1 < n",
        ] {
            assert!(
                MMA_F16.contains(needle),
                "MMA_F16 に REQ-8 境界チェック `{needle}` が見つかりません"
            );
        }
    }

    /// `MMA_BLOCK_THREADS` が CUDA の 1 ブロックあたり最大スレッド数
    /// （1024）を超えないことは本ファイル冒頭の `const _: () =
    /// assert!(...)` でコンパイル時に検査済み。本テストは
    /// `MMA_WARPS_M`/`MMA_WARPS_N` からの導出式が崩れていないことのみ
    /// 検査する。
    #[test]
    fn mma_block_threads_matches_warp_layout() {
        assert_eq!(MMA_BLOCK_THREADS, MMA_WARPS_M * MMA_WARPS_N * 32);
    }

    /// cp.async 16 バイト整列制約の前提（`BK`/`BN` が 8 の倍数）を検査する
    /// （本ファイル冒頭コメント「整列制約」の共有メモリ側前提。崩れると
    /// `cp.async.cg.shared.global` の宛先アドレスが 16 バイト整列しなく
    /// なる）。
    #[test]
    fn mma_tile_dims_satisfy_cp_async_alignment_granularity() {
        assert_eq!(MMA_BK % 8, 0);
        assert_eq!(MMA_BN % 8, 0);
    }

    /// PR #255 レビュー指摘の回帰防止: 最終 K タイルで `cp.async.wait_group 0`
    /// による drain 分岐（`if (t == num_k_tiles - 1)`）が存在することを
    /// ロックする。`wait_group 1` のみだと最終タイルの cp.async 完了を
    /// 待たずに ldmatrix/mma.sync が共有メモリを読みうる（本ファイル
    /// `MMA_WAIT_GROUP_IMMEDIATE` ドキュメンテーションコメント参照）。
    #[test]
    fn mma_f16_source_drains_final_async_copy_group_before_compute() {
        assert!(
            MMA_F16.contains("if (t == num_k_tiles - 1)")
                && MMA_F16.contains("cp.async.wait_group 0;"),
            "MMA_F16 に最終 K タイルの cp.async drain 分岐（wait_group 0）が見つかりません"
        );
    }

    /// PR #255 レビュー指摘の回帰防止: A/B タイルロードの範囲外チャンク
    /// （`valid=0` のゼロ充填）でも `cp.async` ソースアドレスの列オフセット
    /// クランプが 16 バイト（8 要素）境界に切り下げられていることを
    /// ロックする（`k-1`/`n-1` への素朴なクランプはアラインを崩し
    /// 未定義動作になりうる。本ファイル `LOAD_A_STAGE`/`LOAD_B_STAGE`
    /// マクロ直前のコメント参照）。
    #[test]
    fn mma_f16_source_zero_fill_clamp_stays_16_byte_aligned() {
        for needle in ["((k - 1) / 8) * 8", "((n - 1) / 8) * 8"] {
            assert!(
                MMA_F16.contains(needle),
                "MMA_F16 に 16 バイト整列クランプ `{needle}` が見つかりません"
            );
        }
    }

    /// PR #255 レビュー指摘の回帰防止: A フラグメントの ldmatrix.x4
    /// 4 象限が mma.m16n8k16 要求順序（TL/BL/TR/BR）どおりに割り当て
    /// られていることをロックする（レーン群の下位ビットを行、上位ビット
    /// を列に対応させる式。誤って `(lane/8)/2` を行、`(lane/8)%2` を列に
    /// すると TL/TR/BL/BR の順になり不正な結果を招く）。
    #[test]
    fn mma_f16_source_uses_mma_fragment_quadrant_order_for_a() {
        assert!(
            MMA_F16.contains("int a_quad_row = (lane / 8) % 2;")
                && MMA_F16.contains("int a_quad_col = (lane / 8) / 2;"),
            "MMA_F16 の A フラグメント象限順序（TL/BL/TR/BR）が見つかりません"
        );
    }
}
