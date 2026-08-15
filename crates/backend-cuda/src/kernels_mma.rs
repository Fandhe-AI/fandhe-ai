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
//! 縮小した（`kernels_wmma.rs` 冒頭コメントの「実機未接続・コンパイル
//! 未検証によるリスク最小化」判断をそのまま踏襲。実装計画 8 節「リスク」・
//! アドバイザレビューで確認済みの判断）。
//!
//! 当初は 1 warp = C の `MMA_M x MMA_N`（`16x8`）タイル 1 個のみを担当する
//! 構成だったが、warp あたりの算術強度（FLOPs / SMEM ロードバイト）が低く
//! `kernels_wmma_opt.rs`（TF32/f16 opt 経路。冒頭コメント「タイル構成」節）
//! と技法が非対称だったため、イシュー #493（GEMM 性能改善ツリー #479 →
//! Phase 2 親 #490 の B-2）で warp あたり `2x2` の mma タイル（出力
//! `MMA_WARP_M x MMA_WARP_N` = `32x16`）を担当するレジスタブロッキングへ
//! 拡張した。CUTLASS の `MmaIterations` 2 重ループ・metal-flash-attention の
//! レジスタタイル配列・MLX の `MMATile` と同型の標準技法であり、ロード済み
//! A フラグメント 2 個・B フラグメント 2 個を 4 通りの `mma.sync` で総当たり
//! 再利用することで、同一出力あたりの `ldmatrix` 発行回数を半減させる
//! （1 K タイル・16x8 出力あたり: 旧 4 発行〈A x4 + B x2 を warp ごとに
//! 個別発行〉→ 新 2 発行〈A x4 + B x2 を 4 タイル分で共有〉。本ファイル
//! `MMA_WARP_TILES_M`/`MMA_WARP_TILES_N` 定数直下のドキュメンテーション
//! コメント参照）。
//!
//! **B-2 単体でのスループット改善は受け入れ条件ではない**（#493 本文が
//! 明示）。warp 数が 16→4 へ減る（スレッド数 512→128）ことで cp.async
//! 協調ロードの反復回数が増え、B-2 単体では悪化しうる。改善判定はブロック
//! タイル拡大（B-3・#494）完了時点でペアに対して行う（本ファイル
//! `docs/perf/cuda-gemm-mma-pipeline.md` 参照）。数値順序への影響はない:
//! 各出力要素のアキュムレートは同一 warp 内の同一 `mma.sync` 系列
//! （kstep 順）のままであり、B-1（#492）時点と bit 一致の出力になる
//! （parity 非後退契約は `tests/parity_nonregression.rs` が機械検査）。
//!
//! ブロックタイル拡大・ldmatrix 先読みダブルバッファ・cp.async issue
//! interleaving は後続（知見は `docs/cuda-tensor-core-knowledge.md`
//! 〈#65・TASK-11.1f〉に集約済み。#494〜#496 のスコープとして引き継ぐ）。
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

/// warp あたりのレジスタブロッキング係数（#493）。1 warp が C の
/// `MMA_M x MMA_N` タイルを `MMA_WARP_TILES_M x MMA_WARP_TILES_N`（2x2）
/// 個担当し、ロード済み A/B フラグメントを 4 通りの `mma.sync` で再利用
/// する（本ファイル冒頭コメント「タイル構成」参照）。
pub const MMA_WARP_TILES_M: u32 = 2;
pub const MMA_WARP_TILES_N: u32 = 2;

/// 1 warp が担当する C タイルの実寸（`MMA_M`/`MMA_N` を warp タイル数で
/// 拡大したもの）。カーネルソース内 warp 座標計算（`row0_warp`/
/// `col0_warp`）の単位になる。
pub const MMA_WARP_M: u32 = MMA_M * MMA_WARP_TILES_M; // 32
pub const MMA_WARP_N: u32 = MMA_N * MMA_WARP_TILES_N; // 16

/// 1 ブロックあたりの warp 構成（M 方向 1・N 方向 4 = 4 warp = 128
/// スレッド。#493 でレジスタブロッキング導入に伴い warp タイル実寸
/// （`MMA_WARP_M x MMA_WARP_N` = `32x16`）基準へ再導出した。旧構成
/// （M 方向 2・N 方向 8 = 16 warp = 512 スレッド。1 warp = 16x8 タイル
/// 1 個のみ）からスレッド数が 1/4 に減るため、cp.async 協調ロードの
/// 反復回数が増える（A タイル: 512 スレッド中 128 スレッドのみ稼働
/// していた分岐が解消され全 128 スレッドが 1 反復／B タイル: 0.5 反復
/// 相当 → 2 反復）。B-2 単体でのスループット改善を主張しない理由の一つ
/// （本ファイル冒頭コメント「タイル構成」参照。ブロックタイル拡大に
/// よるスレッド数回復は #494 のスコープ）。
pub const MMA_WARPS_M: u32 = MMA_BM / MMA_WARP_M; // 1
pub const MMA_WARPS_N: u32 = MMA_BN / MMA_WARP_N; // 4

/// ブロック内スレッド総数（32 スレッド/warp x warp 数）。
pub const MMA_BLOCK_THREADS: u32 = MMA_WARPS_M * MMA_WARPS_N * 32;

// `cp.async.wait_group` のループ内固定即値は `MMA_STAGES - 2` に一致する
// 必要がある（プロローグで `MMA_STAGES - 1` グループを commit した後、
// 最古のグループの完了を待つには「直近 `MMA_STAGES - 2` グループの
// 未完了を許容する」`wait_group` 即値が必要。標準的なソフトウェア
// パイプラインの式。CUTLASS `mma_multistage.h` と同型）。
//
// #492 で「ループ内固定即値＋ループ外 drain」構造へ整理し、旧来の
// 「最終タイルのみ `wait_group 0`・それ以外は `wait_group 1`」という
// `MMA_STAGES == 3` 専用の 2 値分岐（PR #255 由来）を撤去した。カーネル
// ソース側は毎イテレーション必ず 1 commit を発行する不変条件（範囲外
// タイルでは空グループを commit する）により、ループ内の wait は
// `"n"(STAGES - 2)` という段数非依存の単一即値で常に正しくなる
// （イテレーション t の時点の commit 総数は `(STAGES-1) + t` であり、
// `wait_group (STAGES-2)` は完了数 `>= t+1` を保証するため、タイル t
// 自身のグループの完了が全 t で保証される）。最終タイルの空グループは
// 即完了するため、ループ外の `wait_group 0;`（drain）は残存グループの
// 掃き出しのみを担う。
//
// `STAGES - 2` はカーネルソース側のコンパイル時 `"n"` 制約
// （`asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2))`）で
// 直接計算されるため、Rust 側で対応する定数を別途持つ必要はない
// （非負性は下記 `MMA_STAGES >= 2` のコンパイル時 assert が担保する）。
// かつて `MMA_WAIT_GROUP_IMMEDIATE` という Rust 側定数を持っていたが、
// その定義式自身と比較するだけの debug_assert しか利用箇所がなく実質的な
// 検査価値がなかったため撤去した（#492 レビュー指摘）。

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
///
/// 上記の通りコンパイル時 const アサーションのみからの参照であり、実行時
/// `debug_assert` は意図的に用いない（このコメント自身の設計判断）。その
/// ため rustc 1.88 系の dead-code 解析はこの `pub const` を誤って未使用と
/// 判定する（1.92 以降では解消済み。`cargo +1.88.0 clippy` と
/// `cargo +1.92.0 clippy` の実測差分で確認済み。#149 PR CI 指摘対応）。
#[allow(dead_code)]
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
// #493: warp タイル（`MMA_WARP_M x MMA_WARP_N`）がブロックタイル
// （`MMA_BM`/`MMA_BN`）を割り切ることを検査する（`MMA_WARPS_M`/
// `MMA_WARPS_N` の整数除算が余りを切り捨てて誤った warp グリッドを
// 静かに生成しないための機械検査。本ファイル `MMA_WARPS_M`/
// `MMA_WARPS_N` 定数直下のドキュメンテーションコメント参照）。
const _: () = assert!(
    MMA_BM.is_multiple_of(MMA_WARP_M) && MMA_BN.is_multiple_of(MMA_WARP_N),
    "MMA_BM/MMA_BN must be exact multiples of MMA_WARP_M/MMA_WARP_N \
     (warp register-blocking tile must evenly divide the block tile)"
);
// #492 で「ループ内固定即値＋ループ外 drain」構造へ整理したことにより、
// カーネルソース内の wait_group はもはや `MMA_STAGES == 3` に依存しない
// 段数一般形（`"n"(STAGES - 2)` の固定即値。本ファイル `MMA_STAGES` 定数
// 直下のドキュメンテーションコメント参照）になった。残る制約は
// `STAGES - 2` が非負であること（`u32` の減算アンダーフローを避ける）
// のみであり、`MMA_STAGES == 3` 固定ガードは撤去し `MMA_STAGES >= 2` の
// 下限検査に一般化する。上限は既存の共有メモリ 48KiB assert・
// `MMA_BLOCK_THREADS` assert が引き続き機械検査する。
const _: () = assert!(
    MMA_STAGES >= 2,
    "kernels_mma::MMA_F16 の cp.async パイプラインは MMA_STAGES >= 2 を \
     前提とする（カーネルソース側の `STAGES - 2` 計算が u32 で \
     アンダーフローしないため）"
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
#define WARPS_N 4
#define WARP_TILES_M 2
#define WARP_TILES_N 2
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
    // #493: warp が担当する C タイルの原点。warp タイル実寸
    // （MMA_M*WARP_TILES_M x MMA_N*WARP_TILES_N = 32x16）単位で配置する
    // （本ファイル冒頭コメント「タイル構成」）。
    int row0_warp = block_row0 + warp_row * (MMA_M * WARP_TILES_M);
    int col0_warp = block_col0 + warp_col * (MMA_N * WARP_TILES_N);

    // mma.m16n8k16 のレーン→フラグメント要素対応（PTX ISA の標準
    // groupID/threadID_in_group 分解。本ファイル冒頭コメント「命令選定」）。
    int group_id = lane / 4;
    int tid_in_group = lane % 4;

    // #493: C アキュムレータ（f32 x4 を WARP_TILES_M x WARP_TILES_N 個。
    // レジスタブロッキングにより 1 warp が 2x2 の mma タイルを担当する
    // ため、mi/nj でインデックスされる出力タイルごとに独立したアキュム
    // レータが要る。全ゼロ初期化。本ファイル冒頭コメント「タイル構成」）。
    float d[WARP_TILES_M][WARP_TILES_N][4] = {};

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
    //
    // #492: commit を無条件化する（CUTLASS `mma_multistage.h` と同じ
    // 「1 イテレーション = 必ず 1 commit」不変条件。範囲外ステージ
    // （s >= num_k_tiles、K が浅い形状で発生しうる）はロードを飛ばし
    // 空グループを commit する。PTX ISA 上、未 commit の cp.async が無い
    // 状態の `cp.async.commit_group` は空グループを作り即完了するため、
    // 後述のループ内固定即値 wait が全 num_k_tiles で成立するための
    // 前提となる）。
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < num_k_tiles) {
            LOAD_A_STAGE(s, s * BK);
            LOAD_B_STAGE(s, s * BK);
        }
        asm volatile("cp.async.commit_group;\n");
    }

    for (int t = 0; t < num_k_tiles; ++t) {
        int compute_stage = t % STAGES;
        int next_tile = t + STAGES - 1;
        int load_stage = next_tile % STAGES;

        // #492: 段数一般形の固定即値（`STAGES - 2`。非負性は Rust 側
        // `const _: () = assert!(MMA_STAGES >= 2, ...)` がコンパイル時に
        // 担保する。本ファイル `MMA_STAGES` 定数直下のドキュメンテーション
        // コメント参照）。`"n"` 制約はコンパイル時整数即値を要求する PTX
        // インラインアセンブリの制約（CUTLASS が同様に `cp_async_wait<N>()`
        // をテンプレート非型パラメータで即値化するのと同じ理由）。
        //
        // 正しさ: 上記プロローグの無条件 commit により、イテレーション t
        // の時点での commit 総数は常に `(STAGES-1) + t`。`wait_group
        // (STAGES-2)` は「未完了グループ数 <= STAGES-2」を保証するため、
        // 完了数 >= `(STAGES-1) + t - (STAGES-2)` = `t + 1`、すなわちタイル
        // t のグループ（(t+1) 番目）の完了が全 t で保証される。最終タイル
        // 直前までのみ wait すればよく、最終タイル自身の drain（旧来の
        // `wait_group 0` 分岐。PR #255 レビュー指摘の起点）はループ外へ
        // 切り出した（本関数末尾参照）。よってループ内は段数分岐を持たず、
        // `MMA_STAGES` を 2 や 4 に変えてもカーネルソース側の書き換えが
        // 不要になる。
        asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));
        __syncthreads();

        for (int kstep = 0; kstep < BK / MMA_K; ++kstep) {
            int a_col = kstep * MMA_K;
            int b_row = kstep * MMA_K;

            // #493: A フラグメントを WARP_TILES_M 個（mi = 0..1、半開区間で
            // 値は 0 と 1 の 2 個）先に
            // ロードしてレジスタへ保持する。B フラグメント（nj）と合わせて
            // 4 通りの mma.sync で再利用するレジスタブロッキング（CUTLASS
            // の `MmaIterations` 2 重ループ・MLX の `MMATile` と同型。本
            // ファイル冒頭コメント「タイル構成」）。ldmatrix.x4 1 発行が
            // 16x16（TL/BL/TR/BR の 4 x 8x8）をまとめてロードする点は
            // 単一タイル構成と同じで、mi ごとに独立した 16x16 領域を読む。
            unsigned a_frag[WARP_TILES_M][4];
#pragma unroll
            for (int mi = 0; mi < WARP_TILES_M; ++mi) {
                int a_row = warp_row * (MMA_M * WARP_TILES_M) + mi * MMA_M;

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
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[mi][0]), "=r"(a_frag[mi][1]),
                      "=r"(a_frag[mi][2]), "=r"(a_frag[mi][3])
                    : "r"(a_smem)
                );
            }

            // #493: B フラグメントを WARP_TILES_N 個（nj = 0..1、半開区間で
            // 値は 0 と 1 の 2 個）先に
            // ロードする（A と同じレジスタブロッキング方針。B 2 個を
            // x4.trans 1 発行へ融合する余地は残るが、受け入れ基準の
            // 「A 2 個・B 2 個を 4 通りで再利用」という最小差分に合わせ
            // ここでは x2.trans を nj ごとに個別発行する。融合は B-3 以降の
            // 最適化余地としてここに記録する）。
            unsigned b_frag[WARP_TILES_N][2];
#pragma unroll
            for (int nj = 0; nj < WARP_TILES_N; ++nj) {
                int b_col = warp_col * (MMA_N * WARP_TILES_N) + nj * MMA_N;

                // B フラグメント（16x8。k x n の row-major 格納から `.trans`
                // ロードで mma の `.col` 要求配置へ変換。本ファイル冒頭
                // コメント「命令選定」）。
                int b_row_in_tile = lane % 8;
                int b_quad = lane / 8; // 0..1 のみ使用（x2）
                __half* b_addr = &bs_tile[compute_stage]
                                          [b_row + (b_quad % 2) * 8 + b_row_in_tile]
                                          [b_col];
                unsigned b_smem = (unsigned)__cvta_generic_to_shared(b_addr);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[nj][0]), "=r"(b_frag[nj][1])
                    : "r"(b_smem)
                );
            }

            // #493: mi x nj の 4 通りで mma.sync を発行し d[mi][nj] へ
            // アキュムレートする。1 K タイル（kstep 2 回）あたりの
            // ldmatrix 発行回数は、ブロック全体（C の 32x64 = 16x8
            // タイル 16 個）で旧構成 16 warp x (A x4 + B x2) x 2 kstep =
            // 64 発行 → 新構成 4 warp x (A x4 x2mi + B x2 x2nj) x 2 kstep
            // = 32 発行と半減する（同一出力タイルあたり 4 発行 → 2
            // 発行。ロード済みフラグメントを 4 通り再利用するため）。
            //
            // #497: 内側 nj ループを蛇行（serpentine）走査順にする。
            // 外側 mi が奇数のとき nj を逆順（WARP_TILES_N-1 .. 0）に
            // 辿ることで、mi が進んだ直後に直前と同じ b_frag[nj] を
            // 再利用する形になり、レジスタ局所性・命令スケジューリングの
            // 改善を狙う（CUTLASS `mma_tensor_op.h` の serpentine 走査、
            // MLX `mma.h` `tile_matmad` の `n_serp = (m % 2) ? (N - 1 - n)
            // : n` と同型。Metal 側の先例は #536・PR #652）。
            // 数値不変性: 各アキュムレータ d[mi][x] は 1 kstep あたり
            // 必ず 1 回だけ mma を受け、K 方向の加算順序（kstep ループの
            // 反復順）は変えていない。nj_s は「その kstep 内でどの mi/nj
            // 組を先に発行するか」という命令発行順のみを並べ替えるもので
            // あり、mi x nj の集合（4 通り）自体は不変のため、走査順の
            // 並べ替えはビット単位で同一の結果を生む（§1.2 非後退契約の
            // 前提。tolerance 定数・parity_baseline.rs は変更しない）。
#pragma unroll
            for (int mi = 0; mi < WARP_TILES_M; ++mi) {
#pragma unroll
                for (int nj = 0; nj < WARP_TILES_N; ++nj) {
                    int nj_s = (mi % 2) ? (WARP_TILES_N - 1 - nj) : nj;
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=f"(d[mi][nj_s][0]), "=f"(d[mi][nj_s][1]),
                          "=f"(d[mi][nj_s][2]), "=f"(d[mi][nj_s][3])
                        : "r"(a_frag[mi][0]), "r"(a_frag[mi][1]),
                          "r"(a_frag[mi][2]), "r"(a_frag[mi][3]),
                          "r"(b_frag[nj_s][0]), "r"(b_frag[nj_s][1]),
                          "f"(d[mi][nj_s][0]), "f"(d[mi][nj_s][1]),
                          "f"(d[mi][nj_s][2]), "f"(d[mi][nj_s][3])
                    );
                }
            }
        }

        // #492: commit をループ末尾で無条件発行する（プロローグと同じ
        // 「1 イテレーション = 必ず 1 commit」不変条件。`next_tile` が
        // 範囲外のタイル、つまりループ末尾に近い反復では、ロードのみを
        // `if` で抑止し空グループを commit する）。
        if (next_tile < num_k_tiles) {
            LOAD_A_STAGE(load_stage, next_tile * BK);
            LOAD_B_STAGE(load_stage, next_tile * BK);
        }
        asm volatile("cp.async.commit_group;\n");
        __syncthreads();
    }

    // #492: ループ外 drain。ループ内固定即値 wait（`wait_group
    // (STAGES-2)`）はタイル t 自身のグループの完了までしか保証しない
    // （本ファイル上記コメント「正しさ」参照）ため、ループを抜けた直後に
    // 残存する outstanding グループ（プロローグ以降に commit された空
    // グループを含む）を `wait_group 0` で掃き出してから mma の結果を
    // 読み出す（旧来の「最終タイルのみ wait_group 0」分岐が担っていた
    // 安全性を、段数非依存な形でループ外へ移設したもの。PR #255 レビュー
    // 指摘の趣旨を引き継ぐ）。
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    #undef LOAD_A_STAGE
    #undef LOAD_B_STAGE

    // REQ-8: エピローグの guarded store。mma.m16n8k16 の C/D フラグメント
    // レーン対応（groupID/threadID_in_group。本ファイル冒頭コメント
    // 「命令選定」）: d0/d1 は行 groupID、d2/d3 は行 groupID+8。#493 で
    // mi/nj の 2 重ループへ拡張し、warp が担当する WARP_TILES_M x
    // WARP_TILES_N 個の出力タイルを順に書き戻す（各タイルの guarded 条件
    // 式の形は単一タイル構成から変えていない。REQ-8 ソーステスト
    // needle との互換のため）。
#pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; ++mi) {
#pragma unroll
        for (int nj = 0; nj < WARP_TILES_N; ++nj) {
            int r0 = row0_warp + mi * MMA_M + group_id;
            int r1 = row0_warp + mi * MMA_M + group_id + 8;
            int c0 = col0_warp + nj * MMA_N + tid_in_group * 2;
            int c1 = c0 + 1;

            if (r0 < m && c0 < n) c[(size_t)r0 * n + c0] = __float2half(d[mi][nj][0]);
            if (r0 < m && c1 < n) c[(size_t)r0 * n + c1] = __float2half(d[mi][nj][1]);
            if (r1 < m && c0 < n) c[(size_t)r1 * n + c0] = __float2half(d[mi][nj][2]);
            if (r1 < m && c1 < n) c[(size_t)r1 * n + c1] = __float2half(d[mi][nj][3]);
        }
    }
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
            ("WARP_TILES_M", MMA_WARP_TILES_M),
            ("WARP_TILES_N", MMA_WARP_TILES_N),
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
    /// `MMA_WARPS_M`/`MMA_WARPS_N` からの導出式が崩れていないことに加え、
    /// #493 のレジスタブロッキング導入で再導出された値（4 warp = 128
    /// スレッド）をロックする（本ファイル `MMA_WARPS_M`/`MMA_WARPS_N`
    /// 定数直下のドキュメンテーションコメント参照）。
    #[test]
    fn mma_block_threads_matches_warp_layout() {
        assert_eq!(MMA_BLOCK_THREADS, MMA_WARPS_M * MMA_WARPS_N * 32);
        assert_eq!(MMA_WARPS_M, 1, "#493 再導出値: MMA_BM / MMA_WARP_M");
        assert_eq!(MMA_WARPS_N, 4, "#493 再導出値: MMA_BN / MMA_WARP_N");
        assert_eq!(MMA_BLOCK_THREADS, 128);
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

    /// #492 の回帰防止（旧テスト
    /// `mma_f16_source_drains_final_async_copy_group_before_compute` の
    /// 改名・主張反転）: ループ内の `if (t == num_k_tiles - 1)` 二値分岐が
    /// **存在しない**こと、ループ内 wait が段数一般形の即値制約
    /// （`"n"(STAGES - 2)`）であること、ループ外（`#undef` の直前）に
    /// 無条件の `cp.async.wait_group 0;` drain が存在することを検査する
    /// （本ファイル冒頭コメント「命令選定」・`MMA_STAGES` 定数直下の
    /// ドキュメンテーションコメント参照）。
    #[test]
    fn mma_f16_source_uses_fixed_immediate_wait_with_loop_exit_drain() {
        assert!(
            !MMA_F16.contains("if (t == num_k_tiles - 1)"),
            "MMA_F16 に MMA_STAGES=3 専用の wait_group 二値分岐が残っています \
             （#492 でループ内固定即値＋ループ外 drain へ整理したはず）"
        );
        assert!(
            MMA_F16.contains(r#"asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));"#),
            "MMA_F16 のループ内 wait が段数一般形の即値制約（\"n\"(STAGES - 2)）ではありません"
        );
        // 数字即値 `wait_group 0;` の出現位置が `#undef LOAD_A_STAGE`
        // （ループ本体終了直後の目印）より手前にある、すなわちループの
        // 外側（ループ末尾 `}` の後）に置かれていることを位置関係で検査
        // する。カーネルソースの PTX asm 文字列内部は `\n`（バックスラッシュ
        // + n の 2 文字）が実際の改行ではなくリテラルとして現れるため、
        // 改行を含む固定文字列一致ではなく `find` によるインデックス比較
        // を用いる。
        let undef_pos = MMA_F16
            .find("#undef LOAD_A_STAGE")
            .expect("MMA_F16 に #undef LOAD_A_STAGE が見つかりません");
        let drain_pos = MMA_F16
            .rfind("cp.async.wait_group 0;")
            .expect("MMA_F16 に cp.async.wait_group 0; が見つかりません");
        assert!(
            drain_pos < undef_pos,
            "MMA_F16 のループ外 drain（wait_group 0）が #undef LOAD_A_STAGE より \
             後ろにあります（ループ内へ紛れ込んでいないか確認すること）"
        );
        // ループ本体の閉じ `}`（`for (int t = ...)` ループ末尾。直前の
        // `__syncthreads();` を目印にする）より drain が後ろにあること。
        let loop_syncthreads_pos = MMA_F16
            .rfind("asm volatile(\"cp.async.commit_group;")
            .expect("MMA_F16 にループ末尾の cp.async.commit_group が見つかりません");
        assert!(
            drain_pos > loop_syncthreads_pos,
            "MMA_F16 のループ外 drain（wait_group 0）がループ末尾の commit_group より \
             前にあります（ループ外へ切り出されていない可能性）"
        );
    }

    /// #492 の回帰防止: `cp.async.wait_group <数字リテラル>` 形の出現が
    /// ループ外 drain の `0` の 1 箇所のみであることを検査する。段数
    /// （`MMA_STAGES`）由来のリテラル（旧来の `wait_group 1` 等）が
    /// ループ内へ再導入されると、`MMA_STAGES` を変えた際に無音で誤った
    /// 同期になるため、出現数を機械的に固定する。
    #[test]
    fn mma_f16_source_has_single_numeric_wait_group_literal() {
        let count = MMA_F16.matches("cp.async.wait_group 0;").count();
        assert_eq!(
            count, 1,
            "MMA_F16 中の `cp.async.wait_group 0;`（数字即値）出現数が 1 ではありません \
             （ループ外 drain の 1 箇所のみが正。段数由来の数字リテラルがループ内へ \
             再導入されていないか確認すること）"
        );
        // ループ内 wait は数字即値ではなく `%0` プレースホルダ＋`"n"` 制約
        // 経由の段数一般形でなければならない。
        assert!(
            !MMA_F16.contains("cp.async.wait_group 1;"),
            "MMA_F16 に MMA_STAGES=3 専用の数字即値 `wait_group 1;` が残っています"
        );
    }

    /// #492 受け入れ基準（段数可変化）の CI 側担保: `#define STAGES 3` を
    /// `stages` へ書き換えたソースについて、段数依存の即値リテラル・分岐
    /// が残らないこと、および Rust 側で導出される整合条件（共有メモリ
    /// 48KiB 上限）が成立することを `stages ∈ {2, 4}` について検査する。
    /// `stages >= 2`（`STAGES - 2` の非負性）は下記ループの値が固定
    /// リテラル配列由来のため実行時検査の対象にならず、本ファイル冒頭の
    /// `const _: () = assert!(MMA_STAGES >= 2, ...)` が担う。実機 NVRTC
    /// コンパイル自体は `#[ignore]` 分離（`gemm_mma.rs` 側。本ファイル
    /// 冒頭コメント「検証状態」参照）だが、ソース文字列レベルの整合は
    /// ここで通常 CI 下でも検査できる。
    #[test]
    fn mma_f16_source_stages_are_swappable_without_kernel_source_edits() {
        for stages in [2u32, 4u32] {
            let src = mma_f16_source_with_stages(stages);

            // 段数依存の旧来リテラル・分岐が残っていないこと。
            assert!(
                !src.contains("if (t == num_k_tiles - 1)"),
                "stages={stages}: wait_group 二値分岐が残っています"
            );
            for wrong in ["cp.async.wait_group 1;", "cp.async.wait_group 2;"] {
                assert!(
                    !src.contains(wrong),
                    "stages={stages}: 段数由来の数字即値 `{wrong}` が残っています"
                );
            }

            // Rust 側導出値の整合（本ファイル冒頭 `const _: () = assert!(...)`
            // と同じ式を stages 可変で検査する）。
            let shared_mem_bytes = (MMA_BM * MMA_BK + MMA_BK * MMA_BN) * 2 * stages;
            assert!(
                shared_mem_bytes <= 49_152,
                "stages={stages}: 共有メモリ使用量 {shared_mem_bytes}B が 48KiB を超えています"
            );
        }
    }

    /// `mma_f16_source_stages_are_swappable_without_kernel_source_edits` が
    /// 使うヘルパー: `#define STAGES <MMA_STAGES>` 行のみを `stages` へ
    /// 置換したソース文字列を返す。置換が正確に 1 回起きたことを assert
    /// することで、ヘルパー自体の壊れ（`#define STAGES` 行の文言変化で
    /// マッチしなくなる等）を検出する。
    fn mma_f16_source_with_stages(stages: u32) -> String {
        let from = format!("#define STAGES {MMA_STAGES}\n");
        let to = format!("#define STAGES {stages}\n");
        let count = MMA_F16.matches(&from).count();
        assert_eq!(
            count, 1,
            "MMA_F16 中の `{from:?}` の出現数が 1 ではありません（ヘルパーの \
             前提が崩れています）"
        );
        MMA_F16.replacen(&from, &to, 1)
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

    /// #493 受け入れ基準（回帰防止）: warp あたり 2x2 レジスタブロッキング
    /// 構造（アキュムレータ配列・A/B フラグメント配列・mi/nj 2 重ループの
    /// mma.sync 発行）がソース文字列から失われていないことをロックする。
    #[test]
    fn mma_f16_source_uses_2x2_register_blocking_structure() {
        for needle in [
            "float d[WARP_TILES_M][WARP_TILES_N][4] = {};",
            "unsigned a_frag[WARP_TILES_M][4];",
            "unsigned b_frag[WARP_TILES_N][2];",
            "for (int mi = 0; mi < WARP_TILES_M; ++mi) {",
            "for (int nj = 0; nj < WARP_TILES_N; ++nj) {",
        ] {
            assert!(
                MMA_F16.contains(needle),
                "MMA_F16 に #493 の 2x2 レジスタブロッキング構造 `{needle}` が見つかりません"
            );
        }
    }

    /// #497 受け入れ基準: mma 発行の内側 nj ループが蛇行（serpentine）
    /// 走査順であることをソース証跡でロックする（Metal 側
    /// `shader_source_evidence.rs::gemm_simdgroup_tiled_source_uses_serpentine_scan_order`
    /// 〈PR #652〉と同方針）。蛇行式 `int nj_s = (mi % 2) ? (WARP_TILES_N -
    /// 1 - nj) : nj;` の実在に加え、mma.sync 発行部のオペランドが
    /// `nj_s`（`nj` の生値ではなく）を参照していることを検査し、走査順の
    /// 並べ替えが実際に mma 発行へ反映されていることを保証する。位置検査
    /// （蛇行式が mma.sync asm リテラルより手前・エピローグより前の乗算
    /// ループ内にあること）も合わせて検査する。
    #[test]
    fn mma_f16_source_uses_serpentine_scan_order() {
        let serpentine_needle = "int nj_s = (mi % 2) ? (WARP_TILES_N - 1 - nj) : nj;";
        let serpentine_pos = MMA_F16
            .find(serpentine_needle)
            .expect("MMA_F16 に #497 の蛇行走査式 `nj_s` が見つかりません");

        let mma_sync_pos = MMA_F16
            .find("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
            .expect("MMA_F16 に mma.sync 発行箇所が見つかりません");
        assert!(
            serpentine_pos < mma_sync_pos,
            "蛇行走査式は mma.sync 発行（asm volatile）より手前の乗算ループ内にある必要があります"
        );

        let epilogue_loop_pos = MMA_F16
            .find("for (int mi = 0; mi < WARP_TILES_M; ++mi) {\n#pragma unroll\n        for (int nj = 0; nj < WARP_TILES_N; ++nj) {\n            int r0 = row0_warp")
            .expect("MMA_F16 にエピローグの mi/nj 2 重ループが見つかりません");
        assert!(
            serpentine_pos < epilogue_loop_pos,
            "蛇行走査式はエピローグの guarded store ループより前（mma 発行ループ内）にある必要があります"
        );

        for needle in [
            "\"=f\"(d[mi][nj_s][0]), \"=f\"(d[mi][nj_s][1]),",
            "\"=f\"(d[mi][nj_s][2]), \"=f\"(d[mi][nj_s][3])",
            "\"r\"(b_frag[nj_s][0]), \"r\"(b_frag[nj_s][1]),",
            "\"f\"(d[mi][nj_s][0]), \"f\"(d[mi][nj_s][1]),",
            "\"f\"(d[mi][nj_s][2]), \"f\"(d[mi][nj_s][3])",
        ] {
            assert!(
                MMA_F16.contains(needle),
                "MMA_F16 の mma.sync 発行オペランドが `nj_s` を参照していません（`{needle}` が見つかりません）"
            );
        }
    }

    /// #493 受け入れ基準: `mma.sync` 命令の発行箇所（アセンブリ文字列
    /// リテラル）が mi/nj 2 重ループ内の 1 箇所のみであること（kstep
    /// ループ内で 4 回実行はされるが、ソース上の発行箇所自体は 1 箇所に
    /// 集約されている＝レジスタブロッキングがコピペではなくループ化で
    /// 実装されていることを検査する）。
    #[test]
    fn mma_f16_source_issues_mma_sync_from_single_loop_site() {
        let count = MMA_F16
            .matches("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
            .count();
        assert_eq!(
            count, 1,
            "MMA_F16 中の mma.sync 発行箇所（アセンブリ文字列リテラル）が \
             1 箇所ではありません（mi/nj 2 重ループへ集約されているはず）"
        );
    }

    /// #493 受け入れ基準: エピローグの guarded store が mi/nj 2 重ループの
    /// 内側にあり、4 条件の guarded 式自体は単一タイル構成から変わって
    /// いないことをロックする（`mma_f16_source_retains_req8_boundary_guards`
    /// の REQ-8 needle と合わせて、2x2 化後も境界チェックが希釈されて
    /// いないことを検査する）。
    #[test]
    fn mma_f16_source_epilogue_store_is_inside_warp_tile_loop() {
        let loop_pos = MMA_F16
            .find("for (int mi = 0; mi < WARP_TILES_M; ++mi) {\n#pragma unroll\n        for (int nj = 0; nj < WARP_TILES_N; ++nj) {\n            int r0 = row0_warp")
            .expect("MMA_F16 にエピローグの mi/nj 2 重ループが見つかりません");
        let store_pos = MMA_F16
            .find("c[(size_t)r0 * n + c0] = __float2half(d[mi][nj][0]);")
            .expect("MMA_F16 にエピローグの guarded store（d[mi][nj]）が見つかりません");
        assert!(
            store_pos > loop_pos,
            "MMA_F16 のエピローグ guarded store が mi/nj ループの外側にあります"
        );
    }
}
