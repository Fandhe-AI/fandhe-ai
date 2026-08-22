//! TF32 `mma.sync`(m16n8k8)/`ldmatrix`/`cp.async` GEMM の CUDA カーネルソース
//! （イシュー #801。TF32 経路を WMMA C++ API〈`kernels_wmma_opt.rs` の
//! `WMMA_TF32_F32_STAGED_BODY`〉から生 PTX 直叩きへ移行する refactor）。
//!
//! # 位置づけ・非結線（重要）
//!
//! 本モジュールは `kernels_mma.rs::MMA_F16_BODY`（f16 `mma.sync`/`ldmatrix`/
//! `cp.async` 経路。TASK-11.1h・#187）の構造を TF32 精度へ移植したもので
//! あり、fragment が opaque な WMMA C++ API とは異なり ldmatrix 運用・
//! レジスタダブルバッファ・タイル形状を直接制御できる（実装計画 1 節）。
//! **本イシューでは本番ディスパッチ経路（`gemm.rs`／`ops.rs`／
//! `gemm_auto.rs` の 3 段選択・JIT 特化基盤）へは一切結線しない**。数値
//! 一致回帰・parity 非後退契約の実機確定は #838 が引き継ぎ、その実測
//! （数値一致 6 本中 4 本 FAIL＝機能欠陥の疑い）を根拠に本番採否判断は
//! **#839 で不採用（凍結）と確定した**。#852 で原因（A フラグメント
//! ldmatrix 象限マッピングの PTX ISA 誤読）を特定・修正し、実機
//! （DGX Spark GB10）再実測で FAIL 件数を大幅に縮小した（詳細・残存差分
//! は `docs/perf/cuda-gemm-mma-tf32-ab.md` §8）。**ただし数値一致 6 本
//! 全 pass には至っておらず、凍結は継続する**。凍結解除の判断・性能
//! 再評価は引き続き #835 系の後続に委ねる。再評価条件は
//! `docs/perf/cuda-gemm-mma-tf32-ab.md` §5.1 を参照
//! （`.claude/rules/out-of-scope-tracking.md`）。
//!
//! # 検証状態（#852 で実機検証済み）
//!
//! 本ファイルのカーネルソースは実機（DGX Spark GB10・driver 580.159.03・
//! CUDA 13.0 V13.0.88）の NVRTC 構文検証・実行を通過済み（#852。詳細は
//! `docs/perf/cuda-gemm-mma-tf32-ab.md` §8）。数値一致は同梱テスト（
//! `tests/gemm_mma_tf32.rs`／`tests/mma_tf32_vs_wmma_tf32_staged.rs`）で
//! 実機確認済みだが、残存 FAIL がある。この残存 FAIL の原因は TF32 丸め
//! 誤差・機能欠陥のいずれとも確定していない（`wmma_tf32` との GPU-GPU
//! 相互一致誤差が CPU 参照比較より小さいことは、両経路が共有する TF32
//! 丸め誤差成分の相殺でも説明でき、TF32 丸め誤差説への反証にはならない。
//! §8.4 参照）。
//!
//! # 命令選定
//!
//! `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` は TF32 tensor core
//! 経路の `mma.sync` 標準 shape（sm_80+。PTX ISA 9.7.15.5.3 相当）。cc>=8.0
//! ゲートは `kernels_mma.rs::MIN_COMPUTE_CAPABILITY_MAJOR`（f16 経路）と
//! 同一根拠（`cp.async`/`ldmatrix` は LDGSTS・sm_80+ 限定）を
//! `gemm_mma.rs::check_min_compute_capability` から再利用する
//! （`gemm_mma_tf32.rs` 参照）。
//!
//! ## A フラグメント: ldmatrix の b16 流用
//!
//! TF32 の A タイル（16 行 x 8 列・f32 4B/要素）を、メモリレイアウトが
//! ビット等価な 16 行 x 16 列の b16（2B/要素）タイルとして再解釈すると、
//! `ldmatrix.sync.aligned.m8n8.x4.shared.b16` が読む 4 個の 8x8 b16
//! 象限（各象限 = 8 行 x 4 f32 列）にちょうど一致する（CUTLASS が用いる
//! 既知の技法）。
//!
//! PTX ISA の m16n8k8 tf32 A オペランド分解（groupID = lane/4,
//! tid_in_group = lane%4 という標準表記とは別に、ldmatrix.x4 が実際に
//! 各出力レジスタへどの物理象限を配るかは「アドレスを供給したレーン
//! グループ（lane/8 の 0..3）」で決まる。本カーネルの `a0..a3` 各々が
//! 保持すべき象限は PTX ISA の当該命令の A オペランドテーブルから
//! 以下のとおり導出される（`row = groupID` は a0/a2、`row = groupID+8`
//! は a1/a3、`col = tid_in_group` は a0/a1、`col = tid_in_group+4` は
//! a2/a3。ここでの groupID/tid_in_group は「オペランド論理位置」の表記
//! であり後述のレーン→ldmatrix 対応とは独立）:
//!
//! - a0 = 左上象限（行 0-7・列 0-3）
//! - a1 = 左下象限（行 8-15・列 0-3）
//! - a2 = 右上象限（行 0-7・列 4-7）
//! - a3 = 右下象限（行 8-15・列 4-7）
//!
//! すなわち象限順序は **TL, BL, TR, BR**（`kernels_mma.rs::LDSM_A_FRAG` の
//! f16 版と同一。#852 是正: 旧実装は「TF32 m16n8k8 の A オペランドは
//! f16 m16n8k16 と行/列分解が異なるため象限順序も異なる」と主張し
//! `a_quad_row = g/2`・`a_quad_col = g%2`〈{TL,TR,BL,BR}〉としていたが、
//! これは PTX ISA「Matrix Fragments for mma.m16n8k8（.tf32）」の A
//! オペランド表の誤読だった。同表の a1 は `row = groupID+8`（行8ずれ＝
//! 下段）・a2 は `col = tid_in_group+4`（列4ずれ＝右列）であり、
//! a1=左下・a2=右上という f16 版と同一の対応になる（CUTLASS/CuTe の
//! `SM80_16x8x8_F32TF32TF32F32_TN` の `ALayout`〈value stride (8, 64):
//! v0 が m+8 方向＝BL、v1 が k+4 方向＝TR〉からも同一結論が導出できる）。
//! 実機（GB10）で数値一致 6 本中 4 本 FAIL・GPU-GPU 相互一致 FAIL
//! （最小形状 m16n8k8 で 128/128 要素 FAIL）という #839 の実測は、
//! a1↔a2 のレジスタ入れ替えに相当する本誤りで説明できる。
//! `ldmatrix.x4` はアドレスを供給したレーングループ `g = lane/8`
//! （0..3）の象限データを出力レジスタ `r_g` へ配る（f16 版と共通の
//! ldmatrix 仕様）ため、`LDSM_A_FRAG` は `g` から上記 4 象限への写像を
//! `a_quad_row = g%2`・`a_quad_col = g/2`（f16 版〈`kernels_mma.rs`
//! 1588-1589 行〉と同一の式）とすることで a_frag[0..3] = {TL,BL,TR,BR}
//! を得る。
//!
//! ## B フラグメント: 素の共有メモリロード（`.trans` ldmatrix 不使用）
//!
//! `ldmatrix .trans` は b16 粒度の転置命令であり、32bit 要素（tf32）を
//! 2 個の b16 に分断してしまうため使用できない。B オペランド分解
//! （`row = tid_in_group(+4)`・`col = groupID`。groupID=lane/4,
//! tid_in_group=lane%4）に従い、row-major の共有メモリ（`cp.async` の
//! 生バイトコピーの帰結）から `bs_tile[stage][k0+tid_in_group][col+group_id]`・
//! `bs_tile[stage][k0+tid_in_group+4][col+group_id]` を `__float_as_uint`
//! でレジスタへ直接ロードする（`LDS_B_FRAG` マクロ）。
//!
//! ## TF32 丸め（イシュー #800 の設計を踏襲）
//!
//! `mma.sync` の tf32 オペランドは明示変換済みビットを要求する。cp.async
//! は生バイトコピーのため転送「中」に丸めを挟めない。よって本ファイル
//! `CONVERT_A_STAGE_GROUP`/`CONVERT_B_STAGE_GROUP`（走査添字が
//! `LOAD_A_STAGE_GROUP`/`LOAD_B_STAGE_GROUP` と完全一致することに依存する
//! 3 点論証。定義直上コメント参照）により、各 compute イテレーション t の
//! 先頭・`cp.async.wait_group` 直後・`__syncthreads()` 前に、その stage の
//! smem チャンクを 1 回だけ丸める。#800 で `kernels_wmma_opt.rs`（wmma
//! 経路）へ同型の丸め位置変更を導入したが、512〜2048 で性能回帰したため
//! #851 で revert された（記録: `docs/perf/cuda-gemm-wmma-tf32-smem-
//! staging-rounding.md`）。本ファイル（mma 経路）は bisect 対象外で回帰が
//! 確認されていないため本設計を維持する（正しさ論証は本ファイル内で完結
//! し、他ファイルの残存構造には依存しない）。
//! 変換関数は `wmma::__float_to_tf32`（`#include <mma.h>` 経由。インライン
//! PTX `cvt.rna.tf32.f32` と同一命令でビット一致・NVRTC 構文検証不能環境
//! での asm 構文リスクを避けるため既存カーネルで実証済みのこちらを採用）。
//!
//! # パイプライン骨格
//!
//! `kernels_mma.rs::MMA_F16_BODY` を忠実に踏襲する: プロローグ無条件
//! commit（1 イテレーション = 1 commit 不変条件・#492）、ループ内
//! `cp.async.wait_group (STAGES-2)` 固定即値、cp.async issue interleaving
//! （`K_GROUPS = BK / MMA_K` を kstep ループへ分散・#496）、ldmatrix/lds
//! 先読み 2 面レジスタダブルバッファ（タイル内限定・クロスタイル先読み
//! 不採用・#495）、ループ外 `wait_group 0` drain。各所の正しさ論証は
//! f16 版と同一のためここでは繰り返さず、`kernels_mma.rs` 該当コメントを
//! 参照する形にする。
//!
//! # 整列制約（cp.async 16 バイト境界。`gemm_mma_tf32.rs` が起動前検証）
//!
//! `cp.async.cg.shared.global` の 1 回のコピー粒度は 16 バイト（f32 4
//! 要素）に固定される（`kernels_wmma_opt.rs::WMMA_TF32_F32_STAGED_BODY`
//! と同じ粒度・同じ理由。f16 版の 8 要素/16B とは粒度が異なる）。
//! `gemm_mma_tf32.rs::CudaMmaTf32Gemm::run_tf32` はホスト側で
//! `n % 4 == 0 && k % 4 == 0` を検証する。
//!
//! # 境界検査（REQ-8。省略禁止）
//!
//! `kernels_mma.rs` 冒頭コメント「境界検査」と同一方針:
//! 1. A/B タイルの `cp.async` ロードは範囲外チャンクで `src_size = 0`
//!    を渡しゼロ充填する。アドレスクランプは f32 4 要素（16B）境界へ
//!    切り下げる（`kernels_wmma_opt.rs` の staged tf32 版と同一式。f16 版
//!    の 8 要素境界とは異なる点に注意）。
//! 2. エピローグの guarded store は `(r < m && c < n)` を満たす要素のみ
//!    書き込む。
//! 3. ホスト側 `gemm_mma_tf32.rs::CudaMmaTf32Gemm::run_tf32` は起動前に
//!    `gemm::validate_gemm_dims`・整列検証・グリッド上限検証・K タイル
//!    添字オーバーフロー検証を必ず先行させる。
//!
//! # 数値契約
//!
//! f32 入出力・f32 内部アキュムレート、TF32 丸めは `mma.sync` 投入直前に
//! smem 上で 1 回適用する（`.claude/rules/coding-rust.md` FMA 契約統一節。
//! 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」は
//! TF32 前提の複合指標として適用する）。

use std::sync::LazyLock;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::error::CudaError;

/// mma 命令 1 回あたりの行列形状（`m16n8k8`。sm_80+ の TF32 標準 shape）。
pub const MMA_TF32_M: u32 = 16;
pub const MMA_TF32_N: u32 = 8;
pub const MMA_TF32_K: u32 = 8;

/// ブロックタイル。既存 TF32 opt-staged（`kernels_wmma_opt.rs::
/// WMMA_TF32_STAGED_BLOCK_M/_N` = 64x64・K タイル 16・3 ステージ）と同一の
/// ブロック形状を採用し、A/B 比較（#802）を同条件にする（実装計画 3.2 節）。
pub const MMA_TF32_BM: u32 = 64;
pub const MMA_TF32_BN: u32 = 64;
pub const MMA_TF32_BK: u32 = 16;

/// `cp.async` multi-stage pipelining のステージ数。
pub const MMA_TF32_STAGES: u32 = 3;

/// warp あたりのレジスタブロッキング係数。1 warp が C の `MMA_TF32_M x
/// MMA_TF32_N` タイルを `WARP_TILES_M x WARP_TILES_N`（2x4）個担当する
/// （warp タイル実寸 32x32。実装計画 3.2 節）。
pub const MMA_TF32_WARP_TILES_M: u32 = 2;
pub const MMA_TF32_WARP_TILES_N: u32 = 4;

/// 1 warp が担当する C タイルの実寸。
pub const MMA_TF32_WARP_M: u32 = MMA_TF32_M * MMA_TF32_WARP_TILES_M; // 32
pub const MMA_TF32_WARP_N: u32 = MMA_TF32_N * MMA_TF32_WARP_TILES_N; // 32

/// 1 ブロックあたりの warp 構成（M 方向 2・N 方向 2 = 4 warp = 128
/// スレッド。実装計画 3.2 節）。
pub const MMA_TF32_WARPS_M: u32 = MMA_TF32_BM / MMA_TF32_WARP_M; // 2
pub const MMA_TF32_WARPS_N: u32 = MMA_TF32_BN / MMA_TF32_WARP_N; // 2

/// ブロック内スレッド総数（32 スレッド/warp x warp 数）。
pub const MMA_TF32_BLOCK_THREADS: u32 = MMA_TF32_WARPS_M * MMA_TF32_WARPS_N * 32;

/// 1 ステージあたりの `mma.sync` 呼び出し回数（`BK / MMA_K`）。カーネル内
/// `for (int kstep = 0; kstep < MMA_TF32_BK / MMA_TF32_K; ++kstep)` に対応
/// する Rust 側の唯一の真実源。
pub const MMA_TF32_K_STEPS_PER_STAGE: u32 = MMA_TF32_BK / MMA_TF32_K;

/// A タイル（`as_tile[STAGES][BM][A_PAD]`）の行幅（パディング後）。
/// `kernels_wmma_opt.rs::WMMA_TF32_STAGED_A_PAD` と同一パディング方針
/// （`BK + 4` 要素。cp.async 16B = f32 4 要素粒度の整列を保つ最小加算）。
pub const MMA_TF32_A_PAD: u32 = MMA_TF32_BK + 4;

/// B タイル（`bs_tile[STAGES][BK][B_PAD]`）の行幅（パディング後）。
/// `kernels_wmma_opt.rs::WMMA_TF32_STAGED_B_PAD` と同一パディング方針
/// （`BN + 4` 要素）。
pub const MMA_TF32_B_PAD: u32 = MMA_TF32_BN + 4;

/// 静的共有メモリ使用量（バイト）。`(BM*A_PAD + BK*B_PAD) * 4B * STAGES`。
pub const MMA_TF32_SHARED_MEM_BYTES: u32 =
    (MMA_TF32_BM * MMA_TF32_A_PAD + MMA_TF32_BK * MMA_TF32_B_PAD) * 4 * MMA_TF32_STAGES;

// コンパイル時契約検査（`kernels_mma.rs` 冒頭の f16 版 const assert 群と
// 同型。実機コンパイルできない環境でも `cargo build` の時点で機械検出
// できる代替チェック）。
const _: () = assert!(
    MMA_TF32_SHARED_MEM_BYTES <= crate::kernels_mma::MMA_STATIC_SMEM_LIMIT_BYTES,
    "kernels_mma_tf32 static shared memory exceeds the 48KiB per-block limit \
     shared by every compute capability"
);
const _: () = assert!(
    MMA_TF32_BK.is_multiple_of(MMA_TF32_K),
    "MMA_TF32_BK must be a multiple of MMA_TF32_K (kernel-side kstep loop divisibility)"
);
const _: () = assert!(
    MMA_TF32_BM.is_multiple_of(4) && MMA_TF32_BN.is_multiple_of(4),
    "MMA_TF32_BM/MMA_TF32_BN must be multiples of 4 (cp.async 16-byte / f32 \
     4-element transfer granularity)"
);
const _: () = assert!(
    MMA_TF32_A_PAD.is_multiple_of(4) && MMA_TF32_B_PAD.is_multiple_of(4),
    "MMA_TF32_A_PAD/MMA_TF32_B_PAD must be multiples of 4 (cp.async 16-byte \
     transfer granularity / f32 element alignment)"
);
const _: () = assert!(
    !(MMA_TF32_A_PAD * 4).is_multiple_of(128) && !(MMA_TF32_B_PAD * 4).is_multiple_of(128),
    "MMA_TF32_A_PAD/MMA_TF32_B_PAD row stride in bytes must not be a multiple \
     of 128B (32 banks x 4B) or bank-phase padding degenerates to no-op"
);
const _: () = assert!(
    MMA_TF32_BLOCK_THREADS <= 1024,
    "MMA_TF32_BLOCK_THREADS must not exceed CUDA's per-block thread limit (1024)"
);
const _: () = assert!(
    MMA_TF32_BM.is_multiple_of(MMA_TF32_WARP_M) && MMA_TF32_BN.is_multiple_of(MMA_TF32_WARP_N),
    "MMA_TF32_BM/MMA_TF32_BN must be exact multiples of MMA_TF32_WARP_M/MMA_TF32_WARP_N \
     (warp register-blocking tile must evenly divide the block tile)"
);
const _: () = assert!(
    MMA_TF32_STAGES >= 2,
    "kernels_mma_tf32 の cp.async パイプラインは MMA_TF32_STAGES >= 2 を \
     前提とする（カーネルソース側の `STAGES - 2` 計算が u32 で \
     アンダーフローしないため）"
);
const _: () = assert!(
    MMA_TF32_K_STEPS_PER_STAGE >= 2,
    "kernels_mma_tf32 は CUTLASS mma_base.h の kWarpGemmIterations >= 2 相当 \
     （MMA_TF32_BK / MMA_TF32_K >= 2）を要求する（ソフトウェアパイプライン化の \
     前提。kernels_mma.rs::MMA_K_STEPS_PER_STAGE 直下コメントと同じ理由）"
);

/// TF32 `mma.sync`(m16n8k8) GEMM（f32 入出力・f32 内部アキュムレート）の
/// 既定構成カーネルソース。`gemm_mma_tf32.rs::CudaMmaTf32Gemm::new` は
/// この文字列を `nvrtc::compile_ptx` に渡して `CudaFunction` を得る。
/// カーネルソースはコンパイル時定数のみから `format!` で組み立て、外部
/// 入力文字列を連結しない（`nvrtc.rs` A03 節と同じ契約。
/// `.claude/rules/security.md` A03）。
pub fn mma_tf32_source() -> &'static str {
    &MMA_TF32_SOURCE
}

static MMA_TF32_SOURCE: LazyLock<String> = LazyLock::new(|| {
    format!(
        "\n#include <mma.h>\n\n\
         using namespace nvcuda;\n\n\
         #define MMA_TF32_M {m}\n\
         #define MMA_TF32_N {n}\n\
         #define MMA_TF32_K {k}\n\
         #define MMA_TF32_BM {bm}\n\
         #define MMA_TF32_BN {bn}\n\
         #define MMA_TF32_BK {bk}\n\
         #define MMA_TF32_STAGES {stages}\n\
         #define MMA_TF32_WARP_TILES_M {warp_tiles_m}\n\
         #define MMA_TF32_WARP_TILES_N {warp_tiles_n}\n\
         #define MMA_TF32_WARPS_N {warps_n}\n\
         #define MMA_TF32_A_PAD {a_pad}\n\
         #define MMA_TF32_B_PAD {b_pad}\n\
         \n{body}",
        m = MMA_TF32_M,
        n = MMA_TF32_N,
        k = MMA_TF32_K,
        bm = MMA_TF32_BM,
        bn = MMA_TF32_BN,
        bk = MMA_TF32_BK,
        stages = MMA_TF32_STAGES,
        warp_tiles_m = MMA_TF32_WARP_TILES_M,
        warp_tiles_n = MMA_TF32_WARP_TILES_N,
        warps_n = MMA_TF32_WARPS_N,
        a_pad = MMA_TF32_A_PAD,
        b_pad = MMA_TF32_B_PAD,
        body = MMA_TF32_BODY,
    )
});

/// [`mma_tf32_source`] が結合するカーネル本体テンプレート。
/// `kernels_mma.rs::MMA_F16_BODY` の構造（cp.async 3 ステージ・issue
/// interleaving・レジスタ先読みダブルバッファ）を、TF32 `mma.sync`
/// (m16n8k8) 経路へ移植したもの（本ファイル冒頭コメント「命令選定」
/// 「パイプライン骨格」参照）。差分は (1) グローバル→共有メモリの
/// ロード粒度が f16 8 要素/16B ではなく f32 4 要素/16B である点、
/// (2) A フラグメントは ldmatrix の b16 流用・B フラグメントは素の
/// 共有メモリロード（`.trans` ldmatrix 不使用）である点、(3) 各 stage
/// 到着直後に TF32 丸め（`CONVERT_A_STAGE_GROUP`/`CONVERT_B_STAGE_GROUP`。
/// イシュー #800 の設計踏襲）を挟む点のみで、cp.async 段数管理・
/// commit/wait 配置・issue interleaving の骨格は f16 版と同一の t/stage
/// 添字算術を使う。
const MMA_TF32_BODY: &str = r#"
// REQ-8: グローバル→共有メモリの 16 バイト単位（f32 4 要素）非同期
// コピー。src_size==16 で実データをコピーし、src_size==0 で共有メモリ側を
// ゼロ充填する（kernels_mma.rs::mma_cp_async16 と同じ契約・同じ PTX
// 命令。関数名は同一 NVRTC コンパイル単位内での衝突を避けるため本カーネル
// 専用の接頭辞を付す）。
__device__ __forceinline__ void mma_tf32_cp_async16(void* smem_ptr, const void* gmem_ptr, int src_size)
{
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr), "r"(src_size)
    );
}

extern "C" __global__ void gemm_mma_tf32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k)
{
    // __align__(16): cp.async の 16 バイト転送先整列要件。A_PAD/B_PAD が
    // 4 要素の倍数のため各行の先頭は常に 16 バイト整列する。
    __shared__ __align__(16) float as_tile[MMA_TF32_STAGES][MMA_TF32_BM][MMA_TF32_A_PAD];
    __shared__ __align__(16) float bs_tile[MMA_TF32_STAGES][MMA_TF32_BK][MMA_TF32_B_PAD];

    int block_row0 = blockIdx.y * MMA_TF32_BM;
    int block_col0 = blockIdx.x * MMA_TF32_BN;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int warp_row = warp_id / MMA_TF32_WARPS_N;
    int warp_col = warp_id % MMA_TF32_WARPS_N;
    int row0_warp = block_row0 + warp_row * (MMA_TF32_M * MMA_TF32_WARP_TILES_M);
    int col0_warp = block_col0 + warp_col * (MMA_TF32_N * MMA_TF32_WARP_TILES_N);

    // C/D・B オペランドが共有する groupID/tid_in_group（PTX ISA 標準
    // m16n8k8 分解。本ファイル冒頭コメント「命令選定」参照。C/D のレーン
    // 対応は f16 m16n8k16 と同一形〈m16n8 出力形状が共通のため〉）。
    int group_id = lane / 4;
    int tid_in_group = lane % 4;

    // C アキュムレータ（f32 x4 を WARP_TILES_M x WARP_TILES_N 個）。全ゼロ
    // 初期化。kernels_mma.rs::MMA_F16_BODY と同じレジスタブロッキング方式。
    float d[MMA_TF32_WARP_TILES_M][MMA_TF32_WARP_TILES_N][4] = {};

    int num_k_tiles = (k > 0) ? (k - 1) / MMA_TF32_BK + 1 : 0;

    // #496 相当: 1 K タイル分の cp.async 発行を warp 内 kstep ループへ
    // 分散するための添字空間分割（kernels_mma.rs::MMA_F16_BODY 「#496」
    // 節と同一設計）。K_GROUPS（= BK/MMA_K）は下記 kstep ループの反復回数
    // と必ず一致する。
    #define K_GROUPS (MMA_TF32_BK / MMA_TF32_K)
    #define A_CHUNKS ((MMA_TF32_BM * MMA_TF32_BK) / 4)
    #define B_CHUNKS ((MMA_TF32_BK * MMA_TF32_BN) / 4)
    #define A_GROUP_CHUNKS ((A_CHUNKS + K_GROUPS - 1) / K_GROUPS)
    #define B_GROUP_CHUNKS ((B_CHUNKS + K_GROUPS - 1) / K_GROUPS)

    // REQ-8: 境界外チャンクでも 16 バイト整列を保ったままクランプする
    // （列方向は f32 4 要素境界へ切り下げ。kernels_wmma_opt.rs::
    // LOAD_A_STAGE_GROUP〈TF32 staged 版〉と同一式。行ストライド〈A は
    // k・B は n〉が 4 の倍数であることは gemm_mma_tf32.rs 側の起動前整列
    // 検証が保証する）。
    #define LOAD_A_STAGE_GROUP(stage, k0, g) \
        for (int idx = (g) * A_GROUP_CHUNKS + tid; \
             idx < A_CHUNKS && idx < ((g) + 1) * A_GROUP_CHUNKS; \
             idx += num_threads) { \
            int row = idx / (MMA_TF32_BK / 4); \
            int col0 = (idx % (MMA_TF32_BK / 4)) * 4; \
            int gr = block_row0 + row; \
            int gc = (k0) + col0; \
            int gr_c = gr < m ? gr : (m > 0 ? m - 1 : 0); \
            int gc_c = gc < k ? gc : (k > 0 ? ((k - 1) / 4) * 4 : 0); \
            int valid = (gr < m && gc < k) ? 16 : 0; \
            mma_tf32_cp_async16(&as_tile[stage][row][col0], &a[(size_t)gr_c * k + gc_c], valid); \
        }

    #define LOAD_B_STAGE_GROUP(stage, k0, g) \
        for (int idx = (g) * B_GROUP_CHUNKS + tid; \
             idx < B_CHUNKS && idx < ((g) + 1) * B_GROUP_CHUNKS; \
             idx += num_threads) { \
            int row = idx / (MMA_TF32_BN / 4); \
            int col0 = (idx % (MMA_TF32_BN / 4)) * 4; \
            int gr = (k0) + row; \
            int gc = block_col0 + col0; \
            int gr_c = gr < k ? gr : (k > 0 ? k - 1 : 0); \
            int gc_c = gc < n ? gc : (n > 0 ? ((n - 1) / 4) * 4 : 0); \
            int valid = (gr < k && gc < n) ? 16 : 0; \
            mma_tf32_cp_async16(&bs_tile[stage][row][col0], &b[(size_t)gr_c * n + gc_c], valid); \
        }

    // イシュー #800 の設計踏襲: smem 到着直後の TF32 丸め（1 回化）。走査
    // する idx 範囲・row/col0 算術は上記 LOAD_A_STAGE_GROUP/
    // LOAD_B_STAGE_GROUP と完全に同一（同じ (stage, g) に対し同じ tid が
    // 同じチャンクを担当する）。
    //
    // 正しさの論証（#800 で `kernels_wmma_opt.rs` に導入後 #851 で
    // revert された同型構造と同一の 3 点論証。呼び出し側 = 下記 t
    // ループ先頭）:
    // 1. 自スレッドの読み取り安全性: wait_group は当該スレッド
    //    自身が発行した cp.async の完了を保証する（PTX 契約）。上記の
    //    chunk 一致設計により、本マクロが読む要素は必ず同一スレッドが
    //    直前に cp.async で書き込んだ要素なので、wait_group のみで安全に
    //    読める。
    // 2. 他スレッドへの公開: 変換結果を全 warp の ldmatrix/直接ロードへ
    //    公開するのは wait_group ではなく、呼び出し直後に保持している
    //    `__syncthreads()` である（本マクロの呼び出しはその
    //    `__syncthreads()` より前に置く。順序を入れ替えると本論証は成立
    //    しない）。
    // 3. stage バッファ再利用時の WAR 安全性: 同一物理 stage バッファは
    //    STAGES イテレーションごとに再利用されるが、直前の利用
    //    （ldmatrix/直接ロード・mma.sync によるフラグメント読み出し）は
    //    ループ末尾の無条件 `__syncthreads()`（t ループ末尾）で必ず
    //    完了してから次の cp.async 上書き・本マクロの変換が走る
    //    （`wmma::__float_to_tf32` は冪等なので万一の重複も数値影響なし）。
    #define CONVERT_A_STAGE_GROUP(stage, g) \
        for (int idx = (g) * A_GROUP_CHUNKS + tid; \
             idx < A_CHUNKS && idx < ((g) + 1) * A_GROUP_CHUNKS; \
             idx += num_threads) { \
            int row = idx / (MMA_TF32_BK / 4); \
            int col0 = (idx % (MMA_TF32_BK / 4)) * 4; \
            for (int e = 0; e < 4; ++e) { \
                as_tile[stage][row][col0 + e] = wmma::__float_to_tf32(as_tile[stage][row][col0 + e]); \
            } \
        }

    #define CONVERT_B_STAGE_GROUP(stage, g) \
        for (int idx = (g) * B_GROUP_CHUNKS + tid; \
             idx < B_CHUNKS && idx < ((g) + 1) * B_GROUP_CHUNKS; \
             idx += num_threads) { \
            int row = idx / (MMA_TF32_BN / 4); \
            int col0 = (idx % (MMA_TF32_BN / 4)) * 4; \
            for (int e = 0; e < 4; ++e) { \
                bs_tile[stage][row][col0 + e] = wmma::__float_to_tf32(bs_tile[stage][row][col0 + e]); \
            } \
        }

    #define LOAD_A_STAGE(stage, k0) \
        for (int g_ = 0; g_ < K_GROUPS; ++g_) { \
            LOAD_A_STAGE_GROUP(stage, k0, g_); \
        }

    #define LOAD_B_STAGE(stage, k0) \
        for (int g_ = 0; g_ < K_GROUPS; ++g_) { \
            LOAD_B_STAGE_GROUP(stage, k0, g_); \
        }

    // プロローグ: kernels_mma.rs::MMA_F16_BODY プロローグと同一の
    // 「1 イテレーション = 必ず 1 commit」不変条件（#492）。
    for (int s = 0; s < MMA_TF32_STAGES - 1; ++s) {
        if (s < num_k_tiles) {
            LOAD_A_STAGE(s, s * MMA_TF32_BK);
            LOAD_B_STAGE(s, s * MMA_TF32_BK);
        }
        asm volatile("cp.async.commit_group;\n");
    }

    for (int t = 0; t < num_k_tiles; ++t) {
        int compute_stage = t % MMA_TF32_STAGES;
        int next_tile = t + MMA_TF32_STAGES - 1;
        int load_stage = next_tile % MMA_TF32_STAGES;

        // kernels_mma.rs::MMA_F16_BODY 「#492」節と同一の段数一般形
        // 固定即値（`STAGES - 2`）・同一の正しさ論証（非負性は上記
        // `MMA_TF32_STAGES >= 2` のコンパイル時 assert が担保する）。
        asm volatile("cp.async.wait_group %0;\n" ::"n"(MMA_TF32_STAGES - 2));

        // イシュー #800 の設計踏襲: 丸めは smem 到着直後・全 warp への
        // 公開（下記 __syncthreads）より前にここで 1 回だけ適用する
        // （正しさの論証は CONVERT_A_STAGE_GROUP/CONVERT_B_STAGE_GROUP
        // 定義直上コメント参照。本ループと __syncthreads() の順序を
        // 入れ替えないこと）。
        for (int g_ = 0; g_ < K_GROUPS; ++g_) {
            CONVERT_A_STAGE_GROUP(compute_stage, g_);
            CONVERT_B_STAGE_GROUP(compute_stage, g_);
        }

        __syncthreads();

        // #495 相当: A/B フラグメントを 2 面バッファ化する
        // （kernels_mma.rs::MMA_F16_BODY 「#495」節と同一設計。タイル内
        // 限定・クロスタイル先読み不採用も同一理由）。
        unsigned a_frag[2][MMA_TF32_WARP_TILES_M][4];
        unsigned b_frag[2][MMA_TF32_WARP_TILES_N][2];

        // A フラグメントロード（本ファイル冒頭コメント「命令選定」節
        // 「A フラグメント: ldmatrix の b16 流用」参照。#852 是正後は
        // f16 版〈`kernels_mma.rs::LDSM_A_FRAG`〉と同一の象限順序
        // TL, BL, TR, BR を用いる）。
        #define LDSM_A_FRAG(buf, stage, kstep, mi) \
            do { \
                int a_col_ = (kstep) * MMA_TF32_K; \
                int a_row = warp_row * (MMA_TF32_M * MMA_TF32_WARP_TILES_M) + (mi) * MMA_TF32_M; \
                int a_quad_group = lane / 8; \
                int a_quad_row = a_quad_group % 2; \
                int a_quad_col = a_quad_group / 2; \
                int a_row_in_tile = lane % 8; \
                float* a_addr = &as_tile[stage] \
                                          [a_row + a_quad_row * 8 + a_row_in_tile] \
                                          [a_col_ + a_quad_col * 4]; \
                unsigned a_smem = (unsigned)__cvta_generic_to_shared(a_addr); \
                asm volatile( \
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
                    : "=r"(a_frag[buf][mi][0]), "=r"(a_frag[buf][mi][1]), \
                      "=r"(a_frag[buf][mi][2]), "=r"(a_frag[buf][mi][3]) \
                    : "r"(a_smem) \
                ); \
            } while (0)

        // B フラグメントロード（本ファイル冒頭コメント「命令選定」節
        // 「B フラグメント: 素の共有メモリロード」参照。転置 ldmatrix
        // 修飾子は使わない）。
        #define LDS_B_FRAG(buf, stage, kstep, nj) \
            do { \
                int b_row0 = (kstep) * MMA_TF32_K; \
                int b_col = warp_col * (MMA_TF32_N * MMA_TF32_WARP_TILES_N) + (nj) * MMA_TF32_N; \
                b_frag[buf][nj][0] = __float_as_uint(bs_tile[stage][b_row0 + tid_in_group][b_col + group_id]); \
                b_frag[buf][nj][1] = __float_as_uint(bs_tile[stage][b_row0 + tid_in_group + 4][b_col + group_id]); \
            } while (0)

        // #495 warp プロローグ: kstep=0 のフラグメントをバッファ 0 へ
        // ロードしてから kstep ループへ入る。
#pragma unroll
        for (int mi = 0; mi < MMA_TF32_WARP_TILES_M; ++mi) {
            LDSM_A_FRAG(0, compute_stage, 0, mi);
        }
#pragma unroll
        for (int nj = 0; nj < MMA_TF32_WARP_TILES_N; ++nj) {
            LDS_B_FRAG(0, compute_stage, 0, nj);
        }

#pragma unroll
        for (int kstep = 0; kstep < MMA_TF32_BK / MMA_TF32_K; ++kstep) {
            int cur = kstep % 2;
            int nxt = (kstep + 1) % 2;

            // #495 相当: 次段（kstep+1）のフラグメントを先読みする
            // （kernels_mma.rs::MMA_F16_BODY 「#495」節と同一設計・同一
            // 「タイル内先読みに限定」判断）。
            if (kstep + 1 < MMA_TF32_BK / MMA_TF32_K) {
#pragma unroll
                for (int mi = 0; mi < MMA_TF32_WARP_TILES_M; ++mi) {
                    LDSM_A_FRAG(nxt, compute_stage, kstep + 1, mi);
                }
#pragma unroll
                for (int nj = 0; nj < MMA_TF32_WARP_TILES_N; ++nj) {
                    LDS_B_FRAG(nxt, compute_stage, kstep + 1, nj);
                }
            }

            // #496 相当: cp.async issue interleaving（kernels_mma.rs::
            // MMA_F16_BODY 「#496」節と同一設計・同一の同期の正しさ
            // 論証）。
            if (next_tile < num_k_tiles) {
                LOAD_A_STAGE_GROUP(load_stage, next_tile * MMA_TF32_BK, kstep);
                LOAD_B_STAGE_GROUP(load_stage, next_tile * MMA_TF32_BK, kstep);
            }

            // mi x nj の通りで mma.sync を発行し d[mi][nj] へアキュムレート
            // する（kernels_mma.rs::MMA_F16_BODY 「タイル構成」と同型の
            // レジスタブロッキング再利用）。
#pragma unroll
            for (int mi = 0; mi < MMA_TF32_WARP_TILES_M; ++mi) {
#pragma unroll
                for (int nj = 0; nj < MMA_TF32_WARP_TILES_N; ++nj) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=f"(d[mi][nj][0]), "=f"(d[mi][nj][1]),
                          "=f"(d[mi][nj][2]), "=f"(d[mi][nj][3])
                        : "r"(a_frag[cur][mi][0]), "r"(a_frag[cur][mi][1]),
                          "r"(a_frag[cur][mi][2]), "r"(a_frag[cur][mi][3]),
                          "r"(b_frag[cur][nj][0]), "r"(b_frag[cur][nj][1]),
                          "f"(d[mi][nj][0]), "f"(d[mi][nj][1]),
                          "f"(d[mi][nj][2]), "f"(d[mi][nj][3])
                    );
                }
            }
        }

        #undef LDSM_A_FRAG
        #undef LDS_B_FRAG

        // kernels_mma.rs::MMA_F16_BODY 「#492/#496」節と同一の「1
        // イテレーション = 必ず 1 commit」不変条件。
        asm volatile("cp.async.commit_group;\n");
        __syncthreads();
    }

    // ループ外 drain（kernels_mma.rs::MMA_F16_BODY と同一の正しさ論証）。
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    #undef LOAD_A_STAGE
    #undef LOAD_B_STAGE
    #undef LOAD_A_STAGE_GROUP
    #undef LOAD_B_STAGE_GROUP
    #undef CONVERT_A_STAGE_GROUP
    #undef CONVERT_B_STAGE_GROUP
    #undef A_GROUP_CHUNKS
    #undef B_GROUP_CHUNKS
    #undef A_CHUNKS
    #undef B_CHUNKS
    #undef K_GROUPS

    // REQ-8: エピローグの guarded store。mma.m16n8k8 の C/D フラグメント
    // レーン対応は f16 m16n8k16 版と同一形（本ファイル冒頭コメント
    // 「命令選定」参照）: d0/d1 は行 groupID、d2/d3 は行 groupID+8。
#pragma unroll
    for (int mi = 0; mi < MMA_TF32_WARP_TILES_M; ++mi) {
#pragma unroll
        for (int nj = 0; nj < MMA_TF32_WARP_TILES_N; ++nj) {
            int r0 = row0_warp + mi * MMA_TF32_M + group_id;
            int r1 = row0_warp + mi * MMA_TF32_M + group_id + 8;
            int c0 = col0_warp + nj * MMA_TF32_N + tid_in_group * 2;
            int c1 = c0 + 1;

            if (r0 < m && c0 < n) c[(size_t)r0 * n + c0] = d[mi][nj][0];
            if (r0 < m && c1 < n) c[(size_t)r0 * n + c1] = d[mi][nj][1];
            if (r1 < m && c0 < n) c[(size_t)r1 * n + c0] = d[mi][nj][2];
            if (r1 < m && c1 < n) c[(size_t)r1 * n + c1] = d[mi][nj][3];
        }
    }
}
"#;

/// 診断専用（`internal-diagnostics` feature 限定。イシュー #806）:
/// ブロックタイル（`bm`/`bn`/`bk`）・`cp.async` パイプライン段数
/// （`stages`）・warp タイル形状（`warp_tiles_m`/`_n`）・
/// `__launch_bounds__` を任意に組み合わせた候補 TF32 カーネルソースを
/// 生成する。`kernels_mma.rs::mma_f16_source_with_block_tile`（#804）と
/// 同型の設計（アンカー完全一致置換・2 段階 SMEM 予算判定・fail-closed
/// 検証）を TF32 `mma.sync`(m16n8k8) 経路へ移植したもの。
///
/// # f16 版との差分
///
/// - cp.async 転送粒度は f32 4 要素（16B）であり、f16 版の 8 要素/16B
///   とは異なる（本ファイル冒頭コメント「整列制約」参照）。よって
///   `bm`/`bn` の倍数制約は 4（f16 版は 8）。
/// - `A_PAD`/`B_PAD` は `BK+4`/`BN+4`（f16 版は `BK+8`/`BN+8`。要素サイズが
///   4B〈f32〉であるためパディング加算量も異なる。本ファイル
///   [`MMA_TF32_A_PAD`]/[`MMA_TF32_B_PAD`] 定数直下コメント参照）。
/// - 共有メモリ 1 要素あたり 4B（f32。f16 版は 2B）のため SMEM 予算式の
///   乗数が異なる（[`MMA_TF32_SHARED_MEM_BYTES`] 定数と同じ式）。
/// - `#define` 名前空間は `MMA_TF32_*` 接頭辞（f16 版は無接頭辞の
///   `BM`/`BN`/`STAGES` 等）。
///
/// # 本番結線との違い
///
/// 本関数は本番ディスパッチ（`gemm_mma_tf32.rs::CudaMmaTf32Gemm`。本
/// ファイル冒頭コメント「位置づけ・非結線」参照）から一切呼ばれない。
/// `examples/mma_tf32_ptx_dump.rs`（`internal-diagnostics` feature 限定・
/// `lib.rs::diagnostics` 経由）専用。
///
/// # 共有メモリ予算（2 段階判定。`kernels_mma.rs::
/// mma_f16_source_with_block_tile` と同じ方針）
///
/// `optin_budget_bytes` は呼び出し元がデバイス実測値
/// （`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`）から渡す。
///
/// - 静的共有メモリ予算（[`crate::kernels_mma::MMA_STATIC_SMEM_LIMIT_BYTES`]・
///   48KiB）以下: 本番と同じ静的 `__shared__` 配列宣言のまま候補ソース
///   を返す。
/// - 静的予算超・`optin_budget_bytes` 以下: `as_tile`/`bs_tile` の静的
///   宣言を `extern __shared__` バッファ上のポインタへ変換した候補
///   ソースを返す（宣言 2 行のみの置換。**この経路は `nvrtc`/`ptxas`
///   実機での構文検証を経ていない**）。本番起動側の動的 SMEM opt-in
///   結線は本関数のスコープ外。
/// - `optin_budget_bytes` 超: 机上除外として `CudaError::InvalidKernelConfig`
///   を返す。
#[allow(dead_code)] // 理由は mma_f16_source_with_block_tile と同じ（非公開モジュール）
#[allow(clippy::too_many_arguments)] // 診断専用の候補パラメータを 1 関数に集約する設計上の要求
pub fn mma_tf32_source_with_block_tile(
    bm: u32,
    bn: u32,
    bk: u32,
    stages: u32,
    warp_tiles_m: u32,
    warp_tiles_n: u32,
    launch_bounds: Option<u32>,
    optin_budget_bytes: u32,
) -> Result<String, CudaError> {
    let invalid = |detail: String| CudaError::InvalidKernelConfig { detail };

    if bm == 0 || bn == 0 || bk == 0 || stages == 0 || warp_tiles_m == 0 || warp_tiles_n == 0 {
        return Err(invalid(
            "mma_tf32_source_with_block_tile requires bm/bn/bk/stages/warp_tiles_m/n >= 1"
                .to_string(),
        ));
    }
    if stages < 2 {
        return Err(invalid(format!(
            "mma_tf32_source_with_block_tile stages ({stages}) must be >= 2 (cp.async \
             software pipeline invariant; see mma_tf32_source_commit_wait_group_invariant)"
        )));
    }
    // `kernels_mma.rs::mma_f16_source_with_block_tile` と同じ即値範囲検査
    // （PTX ISA `cp.async.wait_group` の `"n"` オペランドは 0〜7）。
    const MAX_WAIT_GROUP_IMMEDIATE: u32 = 7;
    const MAX_STAGES: u32 = MAX_WAIT_GROUP_IMMEDIATE + 2;
    if stages > MAX_STAGES {
        return Err(invalid(format!(
            "mma_tf32_source_with_block_tile stages ({stages}) must be <= {MAX_STAGES} \
             (cp.async.wait_group immediate operand STAGES - 2 must fit in \
             [0, {MAX_WAIT_GROUP_IMMEDIATE}])"
        )));
    }
    if !bk.is_multiple_of(MMA_TF32_K) {
        return Err(invalid(format!(
            "mma_tf32_source_with_block_tile bk ({bk}) must be a multiple of MMA_TF32_K \
             ({MMA_TF32_K})"
        )));
    }
    let k_steps_per_stage = bk / MMA_TF32_K;
    if k_steps_per_stage < 2 {
        return Err(invalid(format!(
            "mma_tf32_source_with_block_tile bk / MMA_TF32_K ({k_steps_per_stage}) must be >= 2"
        )));
    }
    // cp.async 16 バイト転送粒度の前提（f32 4 要素。本ファイル冒頭コメント
    // 「整列制約」参照。f16 版の 8 要素とは異なる）。
    if !bm.is_multiple_of(4) || !bn.is_multiple_of(4) {
        return Err(invalid(format!(
            "mma_tf32_source_with_block_tile bm ({bm}) and bn ({bn}) must both be multiples \
             of 4 (cp.async 16-byte / f32 4-element transfer granularity)"
        )));
    }

    let warp_m = warp_tiles_m.checked_mul(MMA_TF32_M).ok_or_else(|| {
        invalid(format!(
            "mma_tf32_source_with_block_tile warp_tiles_m={warp_tiles_m} overflows u32 when \
             multiplied by MMA_TF32_M={MMA_TF32_M}"
        ))
    })?;
    let warp_n = warp_tiles_n.checked_mul(MMA_TF32_N).ok_or_else(|| {
        invalid(format!(
            "mma_tf32_source_with_block_tile warp_tiles_n={warp_tiles_n} overflows u32 when \
             multiplied by MMA_TF32_N={MMA_TF32_N}"
        ))
    })?;
    if !bm.is_multiple_of(warp_m) || !bn.is_multiple_of(warp_n) {
        return Err(invalid(format!(
            "mma_tf32_source_with_block_tile candidate warp tile {warp_m}x{warp_n} \
             (warp_tiles_m={warp_tiles_m}, warp_tiles_n={warp_tiles_n}) does not evenly divide \
             the candidate block tile bm={bm}x bn={bn}"
        )));
    }
    let warps_m = bm / warp_m;
    let warps_n = bn / warp_n;
    let threads = warps_m
        .checked_mul(warps_n)
        .and_then(|w| w.checked_mul(32))
        .ok_or_else(|| {
            invalid("mma_tf32_source_with_block_tile block thread count overflow".to_string())
        })?;
    if threads > 1024 {
        return Err(invalid(format!(
            "mma_tf32_source_with_block_tile candidate derives {threads} threads/block, \
             exceeding CUDA's per-block limit (1024)"
        )));
    }
    if let Some(v) = launch_bounds
        && v != threads
    {
        return Err(invalid(format!(
            "mma_tf32_source_with_block_tile launch_bounds ({v}) must equal the derived thread \
             count ({threads})"
        )));
    }

    // 共有メモリ予算（1 要素 4B〈f32〉。`MMA_TF32_SHARED_MEM_BYTES` と
    // 同じ式）。
    let a_pad = bk.checked_add(4).ok_or_else(|| {
        invalid("mma_tf32_source_with_block_tile A tile padded row width overflow".to_string())
    })?;
    let b_pad = bn.checked_add(4).ok_or_else(|| {
        invalid("mma_tf32_source_with_block_tile B tile padded row width overflow".to_string())
    })?;
    let smem_bytes = bm
        .checked_mul(a_pad)
        .and_then(|a| bk.checked_mul(b_pad).and_then(|b| a.checked_add(b)))
        .and_then(|sum| sum.checked_mul(4))
        .and_then(|v| v.checked_mul(stages))
        .ok_or_else(|| {
            invalid("mma_tf32_source_with_block_tile shared memory byte count overflow".to_string())
        })?;
    if smem_bytes > optin_budget_bytes {
        return Err(invalid(format!(
            "mma_tf32_source_with_block_tile candidate bm={bm} bn={bn} bk={bk} stages={stages} \
             requires {smem_bytes} bytes of shared memory, exceeding the opt-in budget \
             ({optin_budget_bytes} bytes; CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)"
        )));
    }
    let needs_dynamic_smem = smem_bytes > crate::kernels_mma::MMA_STATIC_SMEM_LIMIT_BYTES;

    // ブロックタイル・段数の #define をアンカー置換する（`mma_tf32_source()`
    // 実体は既定 config の bm=64/bn=64/bk=16/stages=3/a_pad=20/b_pad=68 が
    // 焼き込み済みのため、これらを候補値へ差し替える）。
    let source = replace_source_anchor(
        mma_tf32_source().to_owned(),
        &format!("#define MMA_TF32_BM {MMA_TF32_BM}\n"),
        &format!("#define MMA_TF32_BM {bm}\n"),
        "mma_tf32_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define MMA_TF32_BN {MMA_TF32_BN}\n"),
        &format!("#define MMA_TF32_BN {bn}\n"),
        "mma_tf32_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define MMA_TF32_BK {MMA_TF32_BK}\n"),
        &format!("#define MMA_TF32_BK {bk}\n"),
        "mma_tf32_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define MMA_TF32_WARPS_N {MMA_TF32_WARPS_N}\n"),
        &format!("#define MMA_TF32_WARPS_N {warps_n}\n"),
        "mma_tf32_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define MMA_TF32_WARP_TILES_M {MMA_TF32_WARP_TILES_M}\n"),
        &format!("#define MMA_TF32_WARP_TILES_M {warp_tiles_m}\n"),
        "mma_tf32_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define MMA_TF32_WARP_TILES_N {MMA_TF32_WARP_TILES_N}\n"),
        &format!("#define MMA_TF32_WARP_TILES_N {warp_tiles_n}\n"),
        "mma_tf32_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define MMA_TF32_STAGES {MMA_TF32_STAGES}\n"),
        &format!("#define MMA_TF32_STAGES {stages}\n"),
        "mma_tf32_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define MMA_TF32_A_PAD {MMA_TF32_A_PAD}\n"),
        &format!("#define MMA_TF32_A_PAD {a_pad}\n"),
        "mma_tf32_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define MMA_TF32_B_PAD {MMA_TF32_B_PAD}\n"),
        &format!("#define MMA_TF32_B_PAD {b_pad}\n"),
        "mma_tf32_source_with_block_tile",
    )?;

    // 動的 SMEM 化（`kernels_mma.rs::mma_f16_source_with_block_tile` と
    // 同じ「宣言 2 行のみ置換」方式。多次元添字構文
    // `as_tile[stage][row][col]`/`bs_tile[stage][row][col]` は本体側の
    // 参照を無変更のまま流用する）。
    const STATIC_SMEM_ANCHOR: &str = "    __shared__ __align__(16) float as_tile[MMA_TF32_STAGES][MMA_TF32_BM][MMA_TF32_A_PAD];\n    __shared__ __align__(16) float bs_tile[MMA_TF32_STAGES][MMA_TF32_BK][MMA_TF32_B_PAD];\n";
    const DYNAMIC_SMEM_REPLACEMENT: &str = "    extern __shared__ __align__(16) unsigned char mma_tf32_dyn_smem[];\n    typedef float MmaTf32AsTileT[MMA_TF32_BM][MMA_TF32_A_PAD];\n    typedef float MmaTf32BsTileT[MMA_TF32_BK][MMA_TF32_B_PAD];\n    MmaTf32AsTileT* as_tile = reinterpret_cast<MmaTf32AsTileT*>(mma_tf32_dyn_smem);\n    MmaTf32BsTileT* bs_tile = reinterpret_cast<MmaTf32BsTileT*>(mma_tf32_dyn_smem + sizeof(MmaTf32AsTileT) * MMA_TF32_STAGES);\n";
    let source = if needs_dynamic_smem {
        replace_source_anchor(
            source,
            STATIC_SMEM_ANCHOR,
            DYNAMIC_SMEM_REPLACEMENT,
            "mma_tf32_source_with_block_tile",
        )?
    } else {
        source
    };

    let source = if let Some(v) = launch_bounds {
        const SIG_ANCHOR: &str = "extern \"C\" __global__ void gemm_mma_tf32(";
        let sig_replacement =
            format!("extern \"C\" __global__ void __launch_bounds__({v}) gemm_mma_tf32(");
        replace_source_anchor(
            source,
            SIG_ANCHOR,
            &sig_replacement,
            "mma_tf32_source_with_block_tile",
        )?
    } else {
        source
    };

    Ok(source)
}

/// [`mma_tf32_source_with_block_tile`] 専用のアンカー完全一致置換
/// ヘルパー（`kernels_mma.rs::replace_source_anchor` と同型・同一契約。
/// `kernels_mma` の当該関数は非公開のためモジュールをまたいで再利用
/// できず、本ファイルへ同型のものを複製する）。`anchor` の出現回数が
/// 1 でない場合は定数変更等で置換前提が崩れたとみなし fail-closed で
/// 拒否する。
fn replace_source_anchor(
    src: String,
    anchor: &str,
    replacement: &str,
    caller: &str,
) -> Result<String, CudaError> {
    let occurrences = src.matches(anchor).count();
    if occurrences != 1 {
        return Err(CudaError::InvalidKernelConfig {
            detail: format!(
                "{caller}: anchor {anchor:?} occurs {occurrences} times in mma_tf32_source() \
                 (expected exactly 1); the define replacement assumption no longer holds"
            ),
        });
    }
    Ok(src.replacen(anchor, replacement, 1))
}

/// [`derive_mma_tf32_block_tile_layout`] が返す、候補ブロックタイル・段数・
/// warp タイル構成から導出したカーネル起動パラメータの束（イシュー #841。
/// `kernels_mma.rs::MmaBlockTileLayout` の TF32 版）。
///
/// [`render_mma_tf32_block_tile`] のカーネルソース展開と、
/// `examples/gemm_mma_tf32_block_tile_bench.rs`（診断専用 A/B ランナー。
/// `internal-diagnostics` feature 限定）のカーネル起動（`threads`・
/// `smem_bytes`・opt-in 動的 SMEM 要否判定）の両方が本構造体を経由する
/// ことで、ブロックスレッド数・共有メモリバイト数の算出式が 1 箇所
/// （[`derive_mma_tf32_block_tile_layout`]）にのみ存在する状態を保つ
/// （`kernels_mma.rs::MmaBlockTileLayout` と同じ「単一の真実源」方針）。
#[allow(dead_code)] // 理由は mma_tf32_source_with_block_tile と同じ（非公開モジュール）
#[derive(Debug, Clone, Copy)]
pub struct MmaTf32BlockTileLayout {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub stages: u32,
    pub warp_tiles_m: u32,
    pub warp_tiles_n: u32,
    /// warp グリッド（`bm`/`warp_m` 行 × `bn`/`warp_n` 列）の行数。
    pub warps_m: u32,
    /// warp グリッドの列数。カーネルソース側 `#define MMA_TF32_WARPS_N` に
    /// 対応。
    pub warps_n: u32,
    /// 導出ブロックスレッド数（`warps_m * warps_n * 32`）。
    /// `LaunchConfig.block_dim.x`・`__launch_bounds__` 一致検査の両方に
    /// 使う。
    pub threads: u32,
    /// A タイル 1 行あたりのパディング済み要素数（`bk + 4`。f16 版は `+8`。
    /// 要素サイズが 4B〈f32〉であるためパディング加算量が異なる。
    /// [`mma_tf32_source_with_block_tile`] ドキュメンテーションコメント
    /// 「f16 版との差分」参照）。
    pub a_pad: u32,
    /// B タイル 1 行あたりのパディング済み要素数（`bn + 4`）。
    pub b_pad: u32,
    /// 共有メモリ総使用量（バイト）。`(bm*a_pad + bk*b_pad) * 4B(f32) *
    /// stages`（[`mma_tf32_source_with_block_tile`] 内 `smem_bytes` 算出と
    /// 同一式）。
    pub smem_bytes: u32,
}

impl MmaTf32BlockTileLayout {
    /// `smem_bytes` が静的 48KiB 上限
    /// （[`crate::kernels_mma::MMA_STATIC_SMEM_LIMIT_BYTES`]。f16 版・TF32
    /// 版で共通の CUDA 既定上限のため定数を共有する）を超え、
    /// `extern __shared__`（opt-in 動的 SMEM）変種を要求するか。
    #[allow(dead_code)] // 理由は Self と同じ（非公開モジュール）
    pub fn needs_dynamic_smem(&self) -> bool {
        self.smem_bytes > crate::kernels_mma::MMA_STATIC_SMEM_LIMIT_BYTES
    }
}

/// 候補 `bm`/`bn`/`bk`/`stages`/`warp_tiles_m`/`_n` から
/// [`MmaTf32BlockTileLayout`] を導出する（イシュー #841。
/// `kernels_mma.rs::derive_mma_block_tile_layout` の TF32 版）。
///
/// 検査する不変条件は [`mma_tf32_source_with_block_tile`] と同一（零値
/// 拒否・段数範囲・`bk`/`MMA_TF32_K` 倍数関係・`bm`/`bn` の 4 の倍数・warp
/// タイルの整数除算・スレッド数上限）。`optin_budget_bytes` との比較・
/// `launch_bounds` 一致検査は行わない（呼び出し元ごとに許容判断が異なる。
/// f16 版 `derive_mma_block_tile_layout` と同じ責務分割）。
pub(crate) fn derive_mma_tf32_block_tile_layout(
    bm: u32,
    bn: u32,
    bk: u32,
    stages: u32,
    warp_tiles_m: u32,
    warp_tiles_n: u32,
) -> Result<MmaTf32BlockTileLayout, CudaError> {
    let invalid = |detail: String| CudaError::InvalidKernelConfig { detail };

    if bm == 0 || bn == 0 || bk == 0 || stages == 0 || warp_tiles_m == 0 || warp_tiles_n == 0 {
        return Err(invalid(
            "derive_mma_tf32_block_tile_layout requires bm/bn/bk/stages/warp_tiles_m/n >= 1"
                .to_string(),
        ));
    }
    if stages < 2 {
        return Err(invalid(format!(
            "derive_mma_tf32_block_tile_layout stages ({stages}) must be >= 2 (cp.async \
             software pipeline invariant)"
        )));
    }
    const MAX_WAIT_GROUP_IMMEDIATE: u32 = 7;
    const MAX_STAGES: u32 = MAX_WAIT_GROUP_IMMEDIATE + 2;
    if stages > MAX_STAGES {
        return Err(invalid(format!(
            "derive_mma_tf32_block_tile_layout stages ({stages}) must be <= {MAX_STAGES} \
             (cp.async.wait_group immediate operand STAGES - 2 must fit in \
             [0, {MAX_WAIT_GROUP_IMMEDIATE}])"
        )));
    }
    if !bk.is_multiple_of(MMA_TF32_K) {
        return Err(invalid(format!(
            "derive_mma_tf32_block_tile_layout bk ({bk}) must be a multiple of MMA_TF32_K \
             ({MMA_TF32_K})"
        )));
    }
    let k_steps_per_stage = bk / MMA_TF32_K;
    if k_steps_per_stage < 2 {
        return Err(invalid(format!(
            "derive_mma_tf32_block_tile_layout bk / MMA_TF32_K ({k_steps_per_stage}) must be \
             >= 2"
        )));
    }
    if !bm.is_multiple_of(4) || !bn.is_multiple_of(4) {
        return Err(invalid(format!(
            "derive_mma_tf32_block_tile_layout bm ({bm}) and bn ({bn}) must both be multiples \
             of 4 (cp.async 16-byte / f32 4-element transfer granularity)"
        )));
    }

    let warp_m = warp_tiles_m.checked_mul(MMA_TF32_M).ok_or_else(|| {
        invalid(format!(
            "derive_mma_tf32_block_tile_layout warp_tiles_m={warp_tiles_m} overflows u32 when \
             multiplied by MMA_TF32_M={MMA_TF32_M}"
        ))
    })?;
    let warp_n = warp_tiles_n.checked_mul(MMA_TF32_N).ok_or_else(|| {
        invalid(format!(
            "derive_mma_tf32_block_tile_layout warp_tiles_n={warp_tiles_n} overflows u32 when \
             multiplied by MMA_TF32_N={MMA_TF32_N}"
        ))
    })?;
    if !bm.is_multiple_of(warp_m) || !bn.is_multiple_of(warp_n) {
        return Err(invalid(format!(
            "derive_mma_tf32_block_tile_layout candidate warp tile {warp_m}x{warp_n} \
             (warp_tiles_m={warp_tiles_m}, warp_tiles_n={warp_tiles_n}) does not evenly divide \
             the candidate block tile bm={bm}x bn={bn}"
        )));
    }
    let warps_m = bm / warp_m;
    let warps_n = bn / warp_n;
    let threads = warps_m
        .checked_mul(warps_n)
        .and_then(|w| w.checked_mul(32))
        .ok_or_else(|| {
            invalid("derive_mma_tf32_block_tile_layout block thread count overflow".to_string())
        })?;
    if threads > 1024 {
        return Err(invalid(format!(
            "derive_mma_tf32_block_tile_layout candidate derives {threads} threads/block, \
             exceeding CUDA's per-block limit (1024)"
        )));
    }

    let a_pad = bk.checked_add(4).ok_or_else(|| {
        invalid("derive_mma_tf32_block_tile_layout A tile padded row width overflow".to_string())
    })?;
    let b_pad = bn.checked_add(4).ok_or_else(|| {
        invalid("derive_mma_tf32_block_tile_layout B tile padded row width overflow".to_string())
    })?;
    let smem_bytes = bm
        .checked_mul(a_pad)
        .and_then(|a| bk.checked_mul(b_pad).and_then(|b| a.checked_add(b)))
        .and_then(|sum| sum.checked_mul(4))
        .and_then(|v| v.checked_mul(stages))
        .ok_or_else(|| {
            invalid(
                "derive_mma_tf32_block_tile_layout shared memory byte count overflow".to_string(),
            )
        })?;

    Ok(MmaTf32BlockTileLayout {
        bm,
        bn,
        bk,
        stages,
        warp_tiles_m,
        warp_tiles_n,
        warps_m,
        warps_n,
        threads,
        a_pad,
        b_pad,
        smem_bytes,
    })
}

/// [`render_mma_tf32_block_tile`] が返す、展開済み候補ソース・展開元
/// [`MmaTf32BlockTileLayout`] を 1 個にまとめた descriptor（イシュー #841。
/// `kernels_mma.rs::RenderedMmaF16BlockTileKernel` の TF32 版）。
///
/// フィールドは非公開。生ソースを外部へ返す公開メソッドは持たない
/// （f16 版と同じ「検査を経ずに `CudaFunction` へ到達する経路を作らない」
/// 契約）。診断専用（`internal-diagnostics` feature 限定）:
/// `examples/gemm_mma_tf32_block_tile_bench.rs` が唯一の呼び出し元。
/// 本番経路（`gemm_mma_tf32.rs::CudaMmaTf32Gemm`）はこの型に一切依存
/// しない（`MMA_TF32_BM`/`MMA_TF32_BN`/`MMA_TF32_STAGES` 等の本番定数は
/// 本イシューでは無変更。`CudaMmaTf32Gemm` 自体が #839 で不採用〈凍結〉
/// 判断済みであることに変わりはない。本ファイル冒頭コメント「位置づけ・
/// 非結線」参照）。
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RenderedMmaTf32BlockTileKernel {
    source: String,
    layout: MmaTf32BlockTileLayout,
}

impl RenderedMmaTf32BlockTileKernel {
    /// カーネルソースを NVRTC コンパイル → 固定エントリポイント
    /// `"gemm_mma_tf32"`（[`mma_tf32_source_with_block_tile`] は `#define`
    /// 群のみを置換しシグネチャ名自体は変えないため、本番既定コンストラクタ
    /// `CudaMmaTf32Gemm::new` と同じエントリポイント名になる）のロードまで
    /// 完結させる。`layout.needs_dynamic_smem()` が真の候補（静的 48KiB
    /// 超）のみ `CudaFunction::set_attribute`
    /// （`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`。cudarc 0.19.8
    /// の安全 API・`unsafe` を要求しない）で opt-in 予算を設定する
    /// （`kernels_mma.rs::RenderedMmaF16BlockTileKernel::compile` と同じ
    /// 「必要時のみ opt-in する」方針）。
    ///
    /// プロセス内 LRU／ディスクキャッシュは使わない（f16 版と同じ理由:
    /// 本 A/B ランナーは候補ごとに 1 回だけコンパイルすればよく、計測
    /// 対象は起動後のカーネル実行時間のみ）。
    #[allow(dead_code)]
    pub fn compile(
        &self,
        device: &crate::device::CudaDevice,
    ) -> Result<CompiledMmaTf32BlockTileKernel, CudaError> {
        let ptx = crate::nvrtc::compile_ptx(&self.source, device.arch())?;
        let func = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_mma_tf32")?;
        if self.layout.needs_dynamic_smem() {
            let bytes_i32 = i32::try_from(self.layout.smem_bytes).map_err(|_| {
                CudaError::InvalidKernelConfig {
                    detail: format!(
                        "dynamic shared memory byte count {} exceeds i32 range required by \
                         cuFuncSetAttribute",
                        self.layout.smem_bytes
                    ),
                }
            })?;
            func.set_attribute(
                cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes_i32,
            )
            .map_err(CudaError::from)?;
        }
        Ok(CompiledMmaTf32BlockTileKernel {
            func,
            layout: self.layout,
        })
    }

    /// テスト専用ソースアクセサ（`RenderedMmaF16BlockTileKernel::source`
    /// と同じ理由・同じ「本番公開 API には現れない」契約）。
    #[cfg(test)]
    fn source(&self) -> &str {
        &self.source
    }
}

/// 候補ブロックタイル・段数・warp タイル構成からカーネルソースを展開し、
/// [`RenderedMmaTf32BlockTileKernel`] を返す（イシュー #841）。
///
/// [`mma_tf32_source_with_block_tile`]（ソース文字列のみを返す既存 API。
/// #806）の結果と、その展開に使った [`derive_mma_tf32_block_tile_layout`]
/// の結果を 1 個の descriptor へ束ねる薄いラッパー（`kernels_mma.rs::
/// render_mma_f16_block_tile` と同型）。`optin_budget_bytes` 超過時は
/// [`mma_tf32_source_with_block_tile`] と同じ理由で
/// `CudaError::InvalidKernelConfig` を返す（呼び出し元
/// `examples/gemm_mma_tf32_block_tile_bench.rs` はこれを「机上除外」として
/// 非致命的に扱い、除外理由をログへ残してスイープを継続する）。
#[allow(dead_code, clippy::too_many_arguments)]
pub fn render_mma_tf32_block_tile(
    bm: u32,
    bn: u32,
    bk: u32,
    stages: u32,
    warp_tiles_m: u32,
    warp_tiles_n: u32,
    launch_bounds: Option<u32>,
    optin_budget_bytes: u32,
) -> Result<RenderedMmaTf32BlockTileKernel, CudaError> {
    let layout = derive_mma_tf32_block_tile_layout(bm, bn, bk, stages, warp_tiles_m, warp_tiles_n)?;
    if layout.smem_bytes > optin_budget_bytes {
        return Err(CudaError::InvalidKernelConfig {
            detail: format!(
                "render_mma_tf32_block_tile candidate bm={bm} bn={bn} bk={bk} stages={stages} \
                 requires {} bytes of shared memory, exceeding the opt-in budget \
                 ({optin_budget_bytes} bytes; CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)",
                layout.smem_bytes
            ),
        });
    }
    let source = mma_tf32_source_with_block_tile(
        bm,
        bn,
        bk,
        stages,
        warp_tiles_m,
        warp_tiles_n,
        launch_bounds,
        optin_budget_bytes,
    )?;
    Ok(RenderedMmaTf32BlockTileKernel { source, layout })
}

/// [`RenderedMmaTf32BlockTileKernel::compile`] が返す、コンパイル済み
/// `CudaFunction`・展開元 [`MmaTf32BlockTileLayout`] を不可分に束ねた
/// descriptor（イシュー #841。`CompiledMmaF16BlockTileKernel` と同型）。
#[allow(dead_code)]
pub struct CompiledMmaTf32BlockTileKernel {
    func: cudarc::driver::CudaFunction,
    layout: MmaTf32BlockTileLayout,
}

/// K タイル添字（`t * bk`。カーネル内 `int` 算術）が `i32` をオーバー
/// フローしないことを検証する（`kernels_mma.rs::validate_mma_k_tile_bound`
/// と同型。`gemm_mma_tf32::validate_mma_tf32_k_bound` は本番固定定数
/// `MMA_TF32_BK` のみを検査対象にする関数のため、候補ごとに異なる `bk` を
/// 検査する本用途には使えず、`bk` を引数に取る本関数を別途用意する）。
fn validate_mma_tf32_k_tile_bound(k: u32, bk: u32) -> Result<(), CudaError> {
    let tile = bk as u64;
    let max_computed_index = if k == 0 {
        0
    } else {
        (k as u64).div_ceil(tile) * tile - 1
    };
    if max_computed_index > i32::MAX as u64 {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "k tile-index arithmetic for TF32 mma.sync(m16n8k8) block-tile candidate \
                 would overflow i32: k={k}, max_computed_index={max_computed_index}, bk={bk}"
            ),
        });
    }
    Ok(())
}

impl CompiledMmaTf32BlockTileKernel {
    /// [`CudaMmaTf32Gemm::launch_tf32`]（`crate::gemm_mma_tf32`）と同じ
    /// 検証手順（`validate_gemm_dims`／`validate_output_len`／no-op 早期
    /// return／`validate_mma_tf32_alignment`／grid y 上限検査／K タイル
    /// 境界検査）に加え、`LaunchConfig.shared_mem_bytes` へ
    /// `self.layout.smem_bytes`（動的変種のみ非零。静的変種は 0）を設定
    /// する。
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn launch_tf32(
        &self,
        stream: &CudaStream,
        a_dev: &CudaSlice<f32>,
        b_dev: &CudaSlice<f32>,
        c_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        crate::gemm::validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        crate::gemm::validate_output_len(c_dev.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        crate::gemm_mma_tf32::validate_mma_tf32_alignment(n, k)?;

        const MAX_GRID_DIM_Y: u32 = 65_535;
        let grid_y = m.div_ceil(self.layout.bm);
        if grid_y > MAX_GRID_DIM_Y {
            return Err(CudaError::InvalidShape {
                detail: format!(
                    "mma_tf32 block-tile candidate grid_dim.y ({grid_y}) exceeds CUDA's \
                     {MAX_GRID_DIM_Y} limit for grid dimensions y/z (bm={}); m={m} is too large",
                    self.layout.bm
                ),
            });
        }
        validate_mma_tf32_k_tile_bound(k, self.layout.bk)?;

        let smem_bytes_u32 = if self.layout.needs_dynamic_smem() {
            self.layout.smem_bytes
        } else {
            0
        };
        let launch_config = LaunchConfig {
            grid_dim: (n.div_ceil(self.layout.bn), m.div_ceil(self.layout.bm), 1),
            block_dim: (self.layout.threads, 1, 1),
            shared_mem_bytes: smem_bytes_u32,
        };
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: `CudaMmaTf32Gemm::launch_tf32` と同一の根拠。カーネル
        // 引数は上記で検証済みの m/n/k から導出しており、カーネル内の
        // 手動境界チェック（cp.async src-size ゼロ充填・エピローグ
        // guarded store。REQ-8）と合わせて OOB 読み書きが起きない根拠と
        // する。`shared_mem_bytes` は
        // `RenderedMmaTf32BlockTileKernel::compile` が算出・opt-in 設定
        // した値（`self.layout.smem_bytes`）と同一であり、カーネル側
        // `extern __shared__` の実際の使用量を過不足なく満たす（静的
        // 変種は 0 のまま。static `__shared__` 宣言は起動時設定を要求
        // しない）。
        unsafe {
            stream
                .launch_builder(&self.func)
                .arg(a_dev)
                .arg(b_dev)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(launch_config)?;
        }
        stream.synchronize()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// カーネルソースが `mma.sync`・`ldmatrix`・`cp.async` の主要命令を
    /// 実在させることを検査する（`kernels_mma.rs::
    /// mma_f16_source_uses_mma_sync_ldmatrix_cp_async_instructions` と
    /// 同型）。
    #[test]
    fn mma_tf32_source_uses_mma_sync_ldmatrix_cp_async_instructions() {
        let src = mma_tf32_source();
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
        // B フラグメントは `.trans` ldmatrix を使わない契約（本ファイル
        // 冒頭コメント「B フラグメント」参照）。
        assert!(
            !src.contains(".trans"),
            "TF32 mma.sync kernel must not use `.trans` ldmatrix for the B \
             operand (32bit tf32 elements would be split by b16-granularity \
             transpose)"
        );
    }

    /// mma.sync 発行が単一ループサイトのみであることを検査する
    /// （`kernels_mma.rs::mma_f16_source_issues_mma_sync_from_single_loop_site`
    /// と同型。コピペ増殖の回帰検出）。
    #[test]
    fn mma_tf32_source_issues_mma_sync_from_single_loop_site() {
        let src = mma_tf32_source();
        let count = src
            .matches("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32")
            .count();
        assert_eq!(
            count, 1,
            "expected exactly 1 occurrence of the mma.sync instruction text \
             (single loop site emitting it at compile time), found {count}"
        );
    }

    /// REQ-8 手動境界チェック（guarded load のゼロ充填分岐・guarded
    /// store）がソース中に存置されていることを検査する。
    #[test]
    fn mma_tf32_source_retains_req8_boundary_checks() {
        let src = mma_tf32_source();
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

    /// #852 の回帰防止: A フラグメントの ldmatrix.x4 4 象限が
    /// mma.m16n8k8（.tf32）要求順序（TL/BL/TR/BR。`kernels_mma.rs::
    /// LDSM_A_FRAG` の f16 版・`mma_f16_source_uses_mma_fragment_
    /// quadrant_order_for_a` と同一契約）どおりに割り当てられていることを
    /// ロックする。誤って `a_quad_group / 2` を行、`a_quad_group % 2` を
    /// 列に対応させると TL/TR/BL/BR の順になり（#839 で実測した数値一致
    /// 6 本中 4 本 FAIL・GPU-GPU 相互不一致の原因）不正な結果を招く。
    #[test]
    fn mma_tf32_source_uses_mma_fragment_quadrant_order_for_a() {
        let src = mma_tf32_source();
        assert!(
            src.contains("int a_quad_row = a_quad_group % 2;")
                && src.contains("int a_quad_col = a_quad_group / 2;"),
            "mma_tf32_source() の A フラグメント象限順序（TL/BL/TR/BR）が見つかりません"
        );
    }

    /// TF32 丸め（`wmma::__float_to_tf32`）の出現箇所が
    /// `CONVERT_A_STAGE_GROUP`/`CONVERT_B_STAGE_GROUP` マクロ定義内のみ
    /// であり、kstep ループ内（mma.sync 発行帯）には存在しないことを検査
    /// する（イシュー #800 の設計「smem ステージング時 1 回化」を本移植
    /// でも壊していないことの回帰検査。`kernels_wmma_opt.rs`〈wmma 経路〉
    /// は #851 で同型構造を revert 済みのため、本テストは mma_tf32 経路
    /// 単独の契約として維持する）。
    #[test]
    fn mma_tf32_source_rounds_only_in_convert_stage_group_macros() {
        let src = mma_tf32_source();
        let convert_count = src.matches("wmma::__float_to_tf32(").count();
        assert_eq!(
            convert_count, 2,
            "expected exactly 2 wmma::__float_to_tf32 call sites (one each in \
             CONVERT_A_STAGE_GROUP/CONVERT_B_STAGE_GROUP), found {convert_count}"
        );

        let kstep_loop_pos = src
            .find("for (int kstep = 0; kstep < MMA_TF32_BK / MMA_TF32_K; ++kstep)")
            .expect("kstep loop must exist in the kernel source");
        assert!(
            !src[kstep_loop_pos..].contains("wmma::__float_to_tf32("),
            "found wmma::__float_to_tf32 at/after the kstep loop (position \
             {kstep_loop_pos}); rounding must stay confined to \
             CONVERT_A_STAGE_GROUP/CONVERT_B_STAGE_GROUP before the loop"
        );
    }

    /// `cp.async.commit_group` がループ内で 1 箇所（無条件・ループ末尾）
    /// のみから発行され、`cp.async.wait_group` がプロローグ後・ループ内・
    /// drain の計 3 箇所から発行されることを検査する（#492 の「1
    /// イテレーション = 必ず 1 commit」不変条件・段数一般形固定即値の
    /// 構造回帰検査）。
    #[test]
    fn mma_tf32_source_commit_wait_group_invariant() {
        let src = mma_tf32_source();
        let commit_count = src.matches("cp.async.commit_group;").count();
        // プロローグ（STAGES-1 回のループ本体で 1 箇所発行）+ t ループ末尾
        // （1 箇所発行）= ソース上の commit 発行サイトは 2 箇所（両者とも
        // 実行時に複数回評価されるループ内の単一サイト）。
        assert_eq!(
            commit_count, 2,
            "expected exactly 2 cp.async.commit_group call sites (prologue \
             loop body, t loop tail), found {commit_count}"
        );

        let wait_count = src.matches("cp.async.wait_group").count();
        // ループ内固定即値 wait（1 サイト）+ ループ外 drain（1 サイト）
        // = 2 箇所。
        assert_eq!(
            wait_count, 2,
            "expected exactly 2 cp.async.wait_group call sites (fixed-immediate \
             in-loop wait, out-of-loop drain), found {wait_count}"
        );
    }

    /// タイル定数の内部整合性（f16 版 `kernels_mma.rs` の同型 const assert
    /// と同じ不変条件）を実行時にも再確認する（コンパイル時 assert の
    /// 二重化ではなく、`cargo test` 実行時に値そのものを表示できるようにする
    /// ための可読性目的の重複）。
    #[test]
    fn mma_tf32_tile_constants_are_internally_consistent() {
        // smem 上限内であることはコンパイル時 `const _: () = assert!(...)`
        // で既に検査済み（本ファイル冒頭）。ここでは値そのものを表示
        // できる形で再確認する（可読性目的の重複）。
        assert_eq!(MMA_TF32_SHARED_MEM_BYTES, 28_416);
        assert_eq!(MMA_TF32_BLOCK_THREADS, 128);
        assert_eq!(MMA_TF32_K_STEPS_PER_STAGE, 2);
        assert_eq!(MMA_TF32_WARP_M, 32);
        assert_eq!(MMA_TF32_WARP_N, 32);
        assert_eq!(MMA_TF32_WARPS_M, 2);
        assert_eq!(MMA_TF32_WARPS_N, 2);
    }

    /// イシュー #806: 既定値 `(MMA_TF32_BM, MMA_TF32_BN, MMA_TF32_BK,
    /// MMA_TF32_STAGES, MMA_TF32_WARP_TILES_M, MMA_TF32_WARP_TILES_N, None,
    /// MMA_TF32_SHARED_MEM_BYTES)` を渡すと `mma_tf32_source()` とバイト
    /// 一致することをロックする（`kernels_mma.rs::
    /// mma_f16_source_with_block_tile_default_matches_mma_f16_source` と
    /// 同じ回帰方針。既定引数では本番ソースへの影響がないことを機械的に
    /// 担保する）。
    #[test]
    fn mma_tf32_source_with_block_tile_default_matches_mma_tf32_source() {
        let src = mma_tf32_source_with_block_tile(
            MMA_TF32_BM,
            MMA_TF32_BN,
            MMA_TF32_BK,
            MMA_TF32_STAGES,
            MMA_TF32_WARP_TILES_M,
            MMA_TF32_WARP_TILES_N,
            None,
            MMA_TF32_SHARED_MEM_BYTES,
        )
        .expect("default block tile config must succeed");
        assert_eq!(src, mma_tf32_source());
    }

    /// 実装計画（イシュー #806）候補表の「両拡大」候補
    /// （128x128x16・S3・warp2x4）が静的 48KiB を超え opt-in 予算内に
    /// 収まること、`extern __shared__` 変種になることを確認する
    /// （`(128*20 + 16*132) * 4B * 3 stages = 56,064B`。48KiB=49,152B を
    /// 超え GB10 実測 opt-in 上限 101,376B 以下）。
    #[test]
    fn mma_tf32_source_with_block_tile_expanded_tile_uses_dynamic_smem() {
        let src = mma_tf32_source_with_block_tile(128, 128, 16, 3, 2, 4, None, 101_376)
            .expect("128x128x16 S3 warp2x4 must fit within the opt-in budget");
        assert!(
            src.contains("extern __shared__ __align__(16) unsigned char mma_tf32_dyn_smem[];"),
            "56,064B (> 48KiB static limit) candidate must use the extern __shared__ variant"
        );
        for needle in [
            "#define MMA_TF32_BM 128\n",
            "#define MMA_TF32_BN 128\n",
            "#define MMA_TF32_BK 16\n",
            "#define MMA_TF32_WARP_TILES_M 2\n",
            "#define MMA_TF32_WARP_TILES_N 4\n",
            "#define MMA_TF32_WARPS_N 4\n",
            "#define MMA_TF32_A_PAD 20\n",
            "#define MMA_TF32_B_PAD 132\n",
        ] {
            assert!(
                src.contains(needle),
                "missing {needle:?} in generated source"
            );
        }
        assert!(
            !src.contains(
                "__shared__ __align__(16) float as_tile[MMA_TF32_STAGES][MMA_TF32_BM][MMA_TF32_A_PAD];"
            ),
            "the static declaration must be fully replaced, not left alongside the dynamic one"
        );
    }

    /// `launch_bounds` を導出スレッド数（512）で明示付与した場合に
    /// シグネチャへ反映されることを確認する。
    #[test]
    fn mma_tf32_source_with_block_tile_applies_launch_bounds() {
        let src = mma_tf32_source_with_block_tile(128, 128, 16, 3, 2, 4, Some(512), 101_376)
            .expect("128x128x16 S3 warp2x4 with explicit launch_bounds must succeed");
        assert!(src.contains("__launch_bounds__(512)"));
    }

    /// 実装計画候補表の「BK 拡大」候補（64x64x32・S3・warp2x2）は机上
    /// 見積もり 53,760B。GB10 実測 opt-in 上限 101,376B より小さい仮想
    /// 予算（50,000B）を渡すと opt-in 予算超過として拒否されることを
    /// ロックする（実機到達前でも判定できることの確認。
    /// `mma_f16_source_with_block_tile_rejects_over_optin_budget` と同型）。
    #[test]
    fn mma_tf32_source_with_block_tile_rejects_over_optin_budget() {
        let err = mma_tf32_source_with_block_tile(64, 64, 32, 3, 2, 2, None, 50_000)
            .expect_err("64x64x32 S3 (53,760B) must exceed the 50,000B opt-in budget");
        assert!(matches!(err, CudaError::InvalidKernelConfig { .. }));
    }

    /// `stages < 2` は cp.async ソフトウェアパイプラインの不変条件違反
    /// として拒否される。
    #[test]
    fn mma_tf32_source_with_block_tile_rejects_stages_below_two() {
        let err = mma_tf32_source_with_block_tile(64, 64, 16, 1, 2, 4, None, u32::MAX)
            .expect_err("stages=1 must be rejected");
        assert!(matches!(err, CudaError::InvalidKernelConfig { .. }));
    }

    /// `cp.async.wait_group "n"(STAGES - 2)` の即値オペランドは 0〜7 の
    /// 範囲でなければならない（PTX ISA 仕様）ため、`stages` は 9 以下で
    /// なければならない。`stages=10`（`STAGES - 2 = 8`）は即値範囲超過と
    /// して拒否される必要がある。`optin_budget_bytes` は `u32::MAX` を
    /// 渡し、拒否理由が共有メモリ予算超過ではなく段数上限であることを
    /// 切り分ける。
    #[test]
    fn mma_tf32_source_with_block_tile_rejects_stages_above_nine() {
        let err = mma_tf32_source_with_block_tile(64, 64, 16, 10, 2, 4, None, u32::MAX)
            .expect_err("stages=10 must be rejected (cp.async.wait_group immediate overflow)");
        assert!(matches!(err, CudaError::InvalidKernelConfig { .. }));
    }

    /// `stages=9`（`STAGES - 2 = 7`）は即値範囲の境界値であり受理される
    /// ことを確認する（上限検証が過剰に厳しくないことの回帰防止）。
    #[test]
    fn mma_tf32_source_with_block_tile_accepts_stages_at_nine() {
        mma_tf32_source_with_block_tile(64, 64, 16, 9, 2, 4, None, u32::MAX)
            .expect("stages=9 is the maximum allowed by the cp.async.wait_group immediate range");
    }

    /// launch_bounds の値が導出スレッド数と食い違う場合は拒否される
    /// （fail-closed 契約）。
    #[test]
    fn mma_tf32_source_with_block_tile_rejects_launch_bounds_mismatch() {
        let err = mma_tf32_source_with_block_tile(64, 64, 16, 3, 2, 4, Some(256), u32::MAX)
            .expect_err("launch_bounds mismatch (actual threads=128) must be rejected");
        assert!(matches!(err, CudaError::InvalidKernelConfig { .. }));
    }

    /// 0 引数（`bm`/`bn`/`bk`/`stages`/`warp_tiles_m`/`warp_tiles_n`）は
    /// すべて fail-closed で拒否される。
    #[test]
    fn mma_tf32_source_with_block_tile_rejects_zero_arguments() {
        for (bm, bn, bk, stages, wtm, wtn) in [
            (0, 64, 16, 3, 2, 4),
            (64, 0, 16, 3, 2, 4),
            (64, 64, 0, 3, 2, 4),
            (64, 64, 16, 0, 2, 4),
            (64, 64, 16, 3, 0, 4),
            (64, 64, 16, 3, 2, 0),
        ] {
            let err = mma_tf32_source_with_block_tile(bm, bn, bk, stages, wtm, wtn, None, 101_376)
                .expect_err("zero-valued argument must be rejected");
            assert!(matches!(err, CudaError::InvalidKernelConfig { .. }));
        }
    }

    /// `bm`/`bn` が cp.async 転送粒度（f32 4 要素）の倍数でない場合は
    /// 拒否される（f16 版の 8 要素倍数制約とは異なる TF32 固有の粒度。
    /// 本ファイル冒頭コメント「整列制約」参照）。
    #[test]
    fn mma_tf32_source_with_block_tile_rejects_non_multiple_of_four_bm_bn() {
        for (bm, bn) in [(65, 64), (64, 65)] {
            let err = mma_tf32_source_with_block_tile(bm, bn, 16, 3, 2, 4, None, u32::MAX)
                .expect_err("bm/bn not a multiple of 4 must be rejected");
            assert!(matches!(err, CudaError::InvalidKernelConfig { .. }));
        }
    }

    /// イシュー #841: `derive_mma_tf32_block_tile_layout` の既定値
    /// （`MMA_TF32_BM`/`_BN`/`_BK`/`_STAGES`/`_WARP_TILES_M`/`_N`）が現行の
    /// 本番タイル定数と一致することをロックする（`kernels_mma.rs::
    /// derive_mma_block_tile_layout_default_matches_production_constants`
    /// と同じ回帰方針）。
    #[test]
    fn derive_mma_tf32_block_tile_layout_default_matches_production_constants() {
        let layout = derive_mma_tf32_block_tile_layout(
            MMA_TF32_BM,
            MMA_TF32_BN,
            MMA_TF32_BK,
            MMA_TF32_STAGES,
            MMA_TF32_WARP_TILES_M,
            MMA_TF32_WARP_TILES_N,
        )
        .expect("default block tile config must succeed");
        assert_eq!(layout.warps_m, MMA_TF32_WARPS_M);
        assert_eq!(layout.warps_n, MMA_TF32_WARPS_N);
        assert_eq!(layout.threads, MMA_TF32_BLOCK_THREADS);
        assert_eq!(layout.a_pad, MMA_TF32_A_PAD);
        assert_eq!(layout.b_pad, MMA_TF32_B_PAD);
        assert_eq!(layout.smem_bytes, MMA_TF32_SHARED_MEM_BYTES);
        assert!(
            !layout.needs_dynamic_smem(),
            "default (static-fit) config must not require the opt-in dynamic SMEM path"
        );
    }

    /// イシュー #841 実装計画候補表（`docs/perf/
    /// cuda-gemm-mma-tf32-block-tile.md` §4）の「BK 拡大」候補
    /// （64/64/32・S3・warp2x2）について、`derive_mma_tf32_block_tile_layout`
    /// が候補表記載の SMEM 実測要求量 53,760B と一致する値を導出することを
    /// ロックする（`(64*36 + 32*68) * 4B * 3stages = 53,760B`）。
    #[test]
    fn derive_mma_tf32_block_tile_layout_matches_candidate_table_smem_bytes() {
        let layout = derive_mma_tf32_block_tile_layout(64, 64, 32, 3, 2, 2)
            .expect("bk-expansion candidate layout derivation must succeed");
        assert_eq!(layout.smem_bytes, 53_760);
        assert!(
            layout.needs_dynamic_smem(),
            "53,760B exceeds the 48KiB static limit"
        );
    }

    /// `render_mma_tf32_block_tile` は `optin_budget_bytes` 超過候補を
    /// 非致命的な `CudaError::InvalidKernelConfig` として拒否する
    /// （`examples/gemm_mma_tf32_block_tile_bench.rs` が「机上除外」として
    /// ログへ残しスイープを継続するための契約。両拡大+ステージ増候補
    /// 〈128/128/16・S4〉机上見積もり 74,752B を GB10 実測上限より小さい
    /// 予算〈50,000B〉と比較し拒否されることを確認する）。
    #[test]
    fn render_mma_tf32_block_tile_rejects_over_optin_budget() {
        let err = render_mma_tf32_block_tile(128, 128, 16, 4, 2, 4, None, 50_000)
            .expect_err("74,752B candidate must exceed the 50,000B opt-in budget");
        assert!(matches!(err, CudaError::InvalidKernelConfig { .. }));
    }

    /// `render_mma_tf32_block_tile` が返す descriptor のソースが
    /// `mma_tf32_source_with_block_tile` 単体呼び出しと同一バイト列で
    /// あることを確認する（`RenderedMmaTf32BlockTileKernel` が独自の
    /// ソース組み立て経路を持たず、公開済み関数へ委譲するだけであることの
    /// 回帰防止。`kernels_mma.rs::render_mma_f16_block_tile_source_
    /// matches_mma_f16_source_with_block_tile` と同型）。
    #[test]
    fn render_mma_tf32_block_tile_source_matches_mma_tf32_source_with_block_tile() {
        let rendered = render_mma_tf32_block_tile(64, 64, 32, 3, 2, 2, None, 101_376)
            .expect("bk-expansion candidate must fit within the opt-in budget");
        let direct = mma_tf32_source_with_block_tile(64, 64, 32, 3, 2, 2, None, 101_376)
            .expect("bk-expansion candidate must fit within the opt-in budget");
        assert_eq!(rendered.source(), direct);
    }
}
