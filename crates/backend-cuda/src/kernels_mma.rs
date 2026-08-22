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
//! アドバイザレビューで確認済みの判断）。**この `BM=32`・`BN=64` は当初
//! 判断の記録であり、現行値は下記 B-3（#494）節が示す `BM=64`・`BN=128`
//! である**（`MMA_BM`/`MMA_BN` 定数が単一の真実源）。
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
//! B-3（#494・GEMM 性能改善ツリー #479 → Phase 2 親 #490）は上記の
//! スレッド数減少を回復させるため、warp あたりのレジスタブロッキング
//! （`MMA_WARP_TILES_M`/`MMA_WARP_TILES_N`）自体は変えず、ブロックタイル
//! （`MMA_BM`/`MMA_BN`）を `32x64` から `64x128` へ拡大した（warp 構成
//! `MMA_WARPS_M x MMA_WARPS_N` = `2x8` = 16 warp = 512 スレッド。B-1 時点
//! の 512 スレッドへ回復）。共有メモリ使用量は 18432B（18KiB）から
//! 36864B（36KiB）へ増えるが per-block 48KiB 上限に対し余裕を残す。
//! `MMA_BK=32` は不変のため、K タイル・kstep 単位のアキュムレート順序は
//! BM/BN の値に依存せず B-1/B-2 時点と bit 一致の出力を維持する（parity
//! 非後退契約は変更不要。候補算出の詳細・SMEM/レジスタ予算・段階的計測
//! 手順は `docs/perf/cuda-gemm-mma-block-tile.md` を参照。sm_121（DGX
//! Spark GB10）実機属性は未実測のため候補判断は全 compute capability
//! 共通の保証値ベースであり、実機での再確認は #502 へ引き継ぐ）。
//!
//! cp.async issue interleaving は #496（GEMM 性能改善ツリー #479 →
//! Phase 2 親 #490 の B-5）で実装済み。CUTLASS `mma_multistage.h` の
//! `kAccessesPerGroupA/B`（`AsyncCopyIterationsPerStage` を
//! `kWarpGemmIterations` 個へ分割発行する方式）と同型の技法を、本
//! カーネルのチャンク添字空間分割として翻案する: 1 K タイル分の
//! `cp.async` 発行を `K_GROUPS`（= `BK / MMA_K`。kstep ループの反復回数と
//! 一致）個の連続チャンクレンジへ分割し、各 kstep の ldmatrix 先読み後・
//! mma.sync 発行前にレンジ 1 個分を発行する（`LOAD_A_STAGE_GROUP`/
//! `LOAD_B_STAGE_GROUP` マクロ直前のコメント参照）。従来はループ末尾で
//! 次段タイルの `cp.async` を一括発行していたため発行コストが K タイル
//! 境界の 1 点に集中していたが、本変更により warp 内 mma ループへ発行が
//! 分散され Tensor Core 演算とオーバーラップする。`cp.async.commit_group`
//! の位置（ループ末尾・無条件）・「1 イテレーション = 必ず 1 commit」
//! 不変条件（#492）は変更しない（分割後も `wait_group (STAGES-2)` の
//! 正しさ論証がそのまま成立する。本ファイル `MMA_STAGES` 定数直下の
//! ドキュメンテーションコメント参照）。発行タイミングの変更はコピー
//! されるデータ・`mma.sync` の発行順序・オペランド値を変えないため、
//! 出力は #495 時点と bit 一致（parity 非後退契約は
//! `tests/parity_nonregression.rs` が機械検査。tolerance・fixture は
//! 変更なし）。未実測（実機 NVRTC 非搭載環境のため。本ファイル冒頭
//! コメント「検証状態」参照）: 実測記録は
//! `docs/perf/cuda-gemm-mma-cp-async-interleaving.md` を参照。
//!
//! 共有メモリのバンクコンフリクト対策（下記「XOR swizzle」節参照）は
//! #498 で非 2 冪パディングを適用済み。
//!
//! ldmatrix 先読みダブルバッファは #495（GEMM 性能改善ツリー #479 →
//! Phase 2 親 #490 の B-4）で実装済み。warp レベルの kstep ループ内で
//! A/B フラグメントを 2 面バッファ化し（`a_frag[2][...]`/`b_frag[2][...]`）、
//! kstep+1 段のフラグメントを kstep 段の `mma.sync` 発行前に先読みする
//! ことで、SMEM→レジスタのロードレイテンシと Tensor Core 演算をオーバー
//! ラップさせる（CUTLASS `mma_multistage.h` の `PipeState`/`mac_loop_iter`
//! と同型。カーネルソース `LDSM_A_FRAG`/`LDSM_B_FRAG` マクロ直前の
//! コメント参照）。CUTLASS の `mac_loop_iter` はタイル境界を跨いで次
//! タイルの kstep=0 まで先読みするが、本実装は**タイル内先読みに限定**
//! する（クロスタイル先読みは不採用。理由は下記マクロ直前のコメント）。
//! 数値順序（アキュムレート順序）は ldmatrix 発行タイミングの変更では
//! 変わらないため、B-3（#494）時点と bit 一致の出力を維持する（parity
//! 非後退契約は `tests/parity_nonregression.rs` が機械検査。tolerance・
//! fixture は変更なし）。未実測（実機 NVRTC 非搭載環境のため。本ファイル
//! 冒頭コメント「検証状態」参照）: 実測記録・レジスタ予算リスクは
//! `docs/perf/cuda-gemm-mma-ldmatrix-double-buffer.md` を参照。イシュー
//! #812 は実機到達不能のまま机上定量化で再評価し、**保留**（`K_STEPS=2`
//! で非先読み kstep が全体の 50% を占めるが、wait/sync 再構成の同期バグ
//! リスクが依然許容できないためのリスク起点判断。より安価な代替案
//! `MMA_BK` 拡大による `K_STEPS` 増を提示）と結論した。詳細・再評価条件は
//! `docs/perf/cuda-gemm-mma-ldmatrix-double-buffer.md`「#812 追加判断」節・
//! `docs/cuda-tensor-core-design.md` §17 を参照。
//!
//! 共有メモリのバンクコンフリクト対策（#498。`docs/perf/cuda-gemm-mma-bank-conflict.md`）:
//! 非 2 冪パディング（`MMA_A_PAD`/`MMA_B_PAD` 定数参照）を適用済み。
//! `kernels_wmma_opt.rs`（`WMMA_TF32_OPT_A_PAD`/`WMMA_TF32_OPT_B_PAD`）と
//! 同型の技法だが、f16（2B/要素）＋ `cp.async` 16B 転送粒度のため
//! パディング幅は 8 要素（16B）単位（f32 opt 経路の +4 要素とは粒度が
//! 異なる）。XOR swizzle（実装計画「段階 3」）は本 PR ではコードとして
//! 実装せず、実機 nsight-compute でバンクコンフリクトの残存が実測された
//! 場合のみ検討する（採否判断基準は上記 docs/perf ファイル参照。
//! 先送り理由だった「コンパイル未検証環境では誤り検出不能」は #486 の
//! プロファイル手段整備で解消済みという位置づけ。
//! out-of-scope-tracking.md に従い記録）。イシュー #812 は実機到達不能の
//! まま**不採用（保留）**を維持しつつ、SMEM フットプリント差分（パディング
//! で `STAGES=4` が静的上限ぴったり適合から動的 SMEM opt-in 必須へ後退）を
//! 第 2 の再評価トリガーとして追加した。詳細は
//! `docs/perf/cuda-gemm-mma-bank-conflict.md`「#812 追加判断」節・
//! `docs/cuda-tensor-core-design.md` §17 を参照。
//!
//! エピローグの `__half2` ベクトル store 化は #805 で実装済み（CUTLASS
//! `predicated_tile_iterator.h` の AlignedArray 方式を簡易化した形）。
//! 隣接列ペア（c0/c1 = c0+1）の C 書き戻しを `__float2half` スカラー
//! store 4 回から `__floats2half2_rn` の `__half2` ベクトル store 2 回へ
//! まとめ、エピローグの store 命令数を半減させる。整列（要素添字が常に
//! 偶数）・ペアの全有効/全無効・丸めの同一性（RN → bit 一致）の根拠は
//! 本ファイル「境界検査」節・エピローグ実装直前のコメントを参照。丸めが
//! 変わらないため出力は #782 時点と bit 一致（parity 非後退契約は
//! `tests/parity_nonregression.rs` が機械検査。tolerance・fixture は
//! 変更なし）。
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
//!    書き込む（範囲外書き込みを発生させない）。#805 でペア単位の
//!    `__half2` ベクトル store 化を行った後も、隣接列ペア（c0/c1）が
//!    「両方有効か両方無効」の二値になる不変条件（`n % 8 == 0` かつ
//!    `c0` 偶数。本ファイル「整列制約」節）を根拠にペア判定
//!    （`c1 < DIM_N`）へ切り替えており、範囲外書き込みを許容していない。
//!    不変条件が到達不能な defensive fallback（`c0 < DIM_N` 単独判定の
//!    スカラー store）も残し、境界検査を弱めない（下記エピローグ実装
//!    直前のコメント参照）。
//! 3. ホスト側 `gemm_mma.rs::CudaMmaGemm::run_f16` は起動前に
//!    `gemm::validate_gemm_dims`（i32 積ガード含む）と上記整列検証の
//!    両方を必ず先行させる。
//!
//! # 数値契約
//!
//! f16 入出力・f32 内部アキュムレートは `kernels_wmma.rs::WMMA_F16` と
//! 同一方針（`.claude/rules/coding-rust.md` FMA 契約統一節）。

use std::sync::LazyLock;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;

use crate::error::CudaError;

/// mma 命令 1 回あたりの行列形状（`m16n8k16`。sm_80+ の f16 標準 shape）。
pub const MMA_M: u32 = 16;
pub const MMA_N: u32 = 8;
pub const MMA_K: u32 = 16;

/// ブロックタイル（本ファイル冒頭コメント「タイル構成」参照。#494（B-3・
/// GEMM 性能改善ツリー #479 → Phase 2 親 #490）で B-2（#493）後の
/// `32x64` から `64x128` へ拡大した候補 B〈`docs/perf/cuda-gemm-mma-block-tile.md`
/// §3 候補表〉。`MMA_BK=32` を維持しているため、アキュムレート順序
/// （K タイル t 順 → kstep 順）は BM/BN の値に依存せず、B-1/B-2 時点と
/// bit 一致の出力を保つ（`tests/parity_nonregression.rs` の parity 非後退
/// 契約はこの論拠に基づき変更不要）。
pub const MMA_BM: u32 = 64;
pub const MMA_BN: u32 = 128;
pub const MMA_BK: u32 = 32;

/// `cp.async` multi-stage pipelining のステージ数。共有メモリ使用量
/// `(MMA_BM*MMA_BK + MMA_BK*MMA_BN) * 2B * MMA_STAGES` = 36864B（36KiB）
/// で per-block 48KiB 上限に対し十分な余裕を持つ（本ファイル冒頭コメント
/// 参照。#494 のブロックタイル拡大後の値）。
pub const MMA_STAGES: u32 = 3;

/// warp あたりのレジスタブロッキング係数（#493）。1 warp が C の
/// `MMA_M x MMA_N` タイルを `MMA_WARP_TILES_M x MMA_WARP_TILES_N`（2x2）
/// 個担当し、ロード済み A/B フラグメントを 4 通りの `mma.sync` で再利用
/// する（本ファイル冒頭コメント「タイル構成」参照）。
///
/// warp タイル拡大候補（2x4／4x2／4x4）のレジスタ収支・`__launch_bounds__`
/// 設計は #803（`mma_f16_source_with_warp_tiles`・
/// `docs/perf/cuda-gemm-mma-warp-tile-register-budget.md`・
/// `docs/cuda-tensor-core-design.md` §14）で診断機構と検証方針を整備済み。
/// レジスタ収支の実機 `ptxas -v` 実測は未了（採用形状・
/// `__launch_bounds__` 付与値は未確定。詳細は上記 perf ドキュメント §5）。
/// 本番結線（この定数自体の変更・`gemm_mma.rs` からの呼び出し）は #804 の
/// スコープであり、本定数は本イシューでは変更しない。
pub const MMA_WARP_TILES_M: u32 = 2;
pub const MMA_WARP_TILES_N: u32 = 2;

/// 1 warp が担当する C タイルの実寸（`MMA_M`/`MMA_N` を warp タイル数で
/// 拡大したもの）。カーネルソース内 warp 座標計算（`row0_warp`/
/// `col0_warp`）の単位になる。
pub const MMA_WARP_M: u32 = MMA_M * MMA_WARP_TILES_M; // 32
pub const MMA_WARP_N: u32 = MMA_N * MMA_WARP_TILES_N; // 16

/// 1 ブロックあたりの warp 構成（M 方向 2・N 方向 8 = 16 warp = 512
/// スレッド。#494 でブロックタイルを `32x64`→`64x128` へ拡大したことで
/// warp タイル実寸（`MMA_WARP_M x MMA_WARP_N` = `32x16`）基準の再導出値が
/// `MMA_WARPS_M=2`/`MMA_WARPS_N=8` になった。#493 が導入したブロックタイル
/// 縮小構成（M 方向 1・N 方向 4 = 4 warp = 128 スレッド）からスレッド数を
/// 4 倍（B-1 時点の 512 スレッド相当）へ回復させ、cp.async 協調ロードの
/// 並列度低下（#493 冒頭コメント参照）を解消する（本ファイル冒頭コメント
/// 「タイル構成」参照。実装計画候補表 B の採用）。
pub const MMA_WARPS_M: u32 = MMA_BM / MMA_WARP_M; // 2
pub const MMA_WARPS_N: u32 = MMA_BN / MMA_WARP_N; // 8

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

/// 全 compute capability 共通の per-block 静的共有メモリ上限（49,152 バイト
/// = 48KiB。動的共有メモリ opt-in `cudaFuncSetAttribute` を追加で呼ばない
/// 限りの静的 `__shared__` 上限）。[`MMA_SHARED_MEM_BYTES`] の下記
/// `const _: () = assert!(...)` と `gemm_auto.rs::STATIC_SMEM_BUDGET_CAP_BYTES`
/// が同じ 49,152 の値を独立にハードコードして重複管理していた（#521
/// レビュー指摘）ため、本定数を唯一の真実源として両所から参照する。
/// `kernels_wmma_opt.rs` の config 検証器（MMA/WMMA opt 双方の静的
/// 共有メモリ上限検査）も本定数を参照する（イシュー #516 レビュー
/// 指摘対応。48KiB はハードウェア側の固定上限であり MMA 固有の値
/// ではないため、モジュールをまたいだ共有が妥当）。
pub const MMA_STATIC_SMEM_LIMIT_BYTES: u32 = 49_152;

/// A タイル（`as_tile[STAGES][BM][MMA_A_PAD]`）の行幅（パディング後。
/// #498「共有メモリのバンクコンフリクト対策」）。パディングなしの
/// `MMA_BK=32`（f16 2B/要素で行ストライド 64B = 16 バンク）は 2 冪の
/// ため、`ldmatrix.x4`（A フラグメント。本ファイル冒頭コメント「命令
/// 選定」）が読む 8 行の開始バンクが全て同一位相へ収束し 4-way バンク
/// コンフリクトが理論上発生しうる。`+8` 要素（16B）を加えると行ストライド
/// 80B = 20 バンクとなり、8 行の開始バンクは `0,20,8,28,16,4,24,12` と
/// 全て相異なる（`gcd(20,32)=4` だが 8 行分の巡回で 32 バンクを完全被覆。
/// `docs/perf/cuda-gemm-mma-bank-conflict.md` §2 参照）。パディング幅を
/// 8 要素単位に限定する理由は `cp.async` の 16B（f16 8 要素）転送粒度・
/// 整列要件（本ファイル冒頭コメント「整列制約」）を崩さないため。
///
/// [`MMA_SHARED_MEM_BYTES`] と同じ理由（コンパイル時 const アサーション
/// のみからの参照）で rustc 1.88 系 dead-code 誤検知の対象になるため
/// `#[allow(dead_code)]` を付す。
#[allow(dead_code)]
pub const MMA_A_PAD: u32 = MMA_BK + 8;

/// B タイル（`bs_tile[STAGES][BK][MMA_B_PAD]`）の行幅（パディング後。
/// #498）。パディングなしの `MMA_BN=128`（行ストライド 256B）は
/// バンク位相が全行で 0 固定のため、`ldmatrix.x2.trans`（B フラグメント）
/// が読む 8 行で 8-way バンクコンフリクトが理論上発生しうる（[`MMA_A_PAD`]
/// より深刻）。`+8` 要素を加えると行ストライド 272B = バンク位相 +4/行の
/// 等差数列となり、8 行の開始バンクが `0,4,8,...,28` と分散する
/// （`docs/perf/cuda-gemm-mma-bank-conflict.md` §2 参照）。
///
/// [`MMA_SHARED_MEM_BYTES`] と同じ理由でコンパイル時 const アサーション
/// のみからの参照のため `#[allow(dead_code)]` を付す。
#[allow(dead_code)]
pub const MMA_B_PAD: u32 = MMA_BN + 8;

/// 静的共有メモリ使用量（バイト）。`(MMA_BM*MMA_A_PAD + MMA_BK*MMA_B_PAD) *
/// 2B (f16) * MMA_STAGES`（#498 のバンクコンフリクト対策パディング込み。
/// パディング前は `(MMA_BM*MMA_BK + MMA_BK*MMA_BN) * 2B * MMA_STAGES` =
/// 36,864B だったが、パディング後は 41,472B）。[`MMA_STATIC_SMEM_LIMIT_BYTES`]
/// に対する実使用量を下記 `const _: () = assert!(...)` でコンパイル時に
/// 検査する（本ファイル冒頭コメント「タイル構成」参照。タイル定数変更時に
/// 即座にビルドエラーで検出できるよう、実行時 `debug_assert` ではなく
/// コンパイル時定数アサーションとする）。
///
/// `gemm_auto.rs::derive_stages_for_device` からも参照されるようになったが
/// （デバイス実測 SMEM 予算のクランプ上限として。`gemm_auto.rs` 冒頭の
/// コンパイル時契約検査コメント参照）、rustc 1.88 系の dead-code 解析が
/// 誤って未使用と判定する既知の quirk（1.92 以降では解消済み。`#149` PR CI
/// 指摘対応）への対策として `#[allow(dead_code)]` は保守的に残す（この
/// crate 内参照追加により quirk 自体が解消したかは 1.88 系での再実測なし
/// には確認できないため、断定しない）。
#[allow(dead_code)]
pub const MMA_SHARED_MEM_BYTES: u32 = (MMA_BM * MMA_A_PAD + MMA_BK * MMA_B_PAD) * 2 * MMA_STAGES;

// コンパイル時契約検査（タイル定数の内部整合性。実機コンパイルできない
// 環境でも `cargo build` の時点で機械検出できる代替チェック。本ファイル
// 冒頭コメント「タイル構成」参照）。
const _: () = assert!(
    MMA_SHARED_MEM_BYTES <= MMA_STATIC_SMEM_LIMIT_BYTES,
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
// #498: パディング幅の 16B（f16 8 要素）整列前提を機械検査する。
// `cp.async.cg.shared.global` は転送先アドレスが 16B 境界に整列している
// 必要があり（本ファイル冒頭コメント「整列制約」）、`ldmatrix` の行
// アドレス計算（`as_tile[stage][row][col]`/`bs_tile[stage][row][col]`）も
// 行幅の変化がこの整列を崩さないことに依存する（[`MMA_A_PAD`]/
// [`MMA_B_PAD`] 定数直下のドキュメンテーションコメント参照）。
const _: () = assert!(
    MMA_A_PAD.is_multiple_of(8) && MMA_B_PAD.is_multiple_of(8),
    "MMA_A_PAD/MMA_B_PAD must be multiples of 8 (cp.async 16-byte transfer \
     granularity / ldmatrix row alignment)"
);
// #498: パディング後の行ストライド（バイト数）が 128B（= 32 バンク x 4B/
// バンクの 1 巡回長）の倍数でないことを機械検査する。128B の倍数だと
// 全行が同一バンク位相に収束し、パディングを追加した意味（バンク位相の
// 分散によるコンフリクト低減）が失われる（[`MMA_A_PAD`]/[`MMA_B_PAD`]
// 定数直下のドキュメンテーションコメント・
// `docs/perf/cuda-gemm-mma-bank-conflict.md` §2 参照）。
const _: () = assert!(
    !(MMA_A_PAD * 2).is_multiple_of(128) && !(MMA_B_PAD * 2).is_multiple_of(128),
    "MMA_A_PAD/MMA_B_PAD row stride in bytes must not be a multiple of 128B \
     (32 banks x 4B) or bank-phase padding degenerates to no-op"
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
// #494 受け入れ基準 3 項: CUTLASS `mma_base.h` の
// `kWarpGemmIterations >= 2`（`static_assert`）相当の機械検査。
// `MMA_K_STEPS_PER_STAGE`（= `BK / MMA_K`）が 1 だと 1 ステージあたりの
// mma.sync 発行が 1 回のみになり、ldmatrix ロード直後に mma を 1 回しか
// 発行しないため命令レイテンシを隠蔽できず、ソフトウェアパイプライン化の
// 効果が実質失われる。候補表（`docs/perf/cuda-gemm-mma-block-tile.md`
// §3）の全候補は `MMA_BK=32` を維持するため `MMA_K_STEPS_PER_STAGE=2` で
// 本条件を満たす。
const _: () = assert!(
    MMA_K_STEPS_PER_STAGE >= 2,
    "kernels_mma::MMA_F16 は CUTLASS mma_base.h の kWarpGemmIterations >= 2 \
     相当（MMA_BK / MMA_K >= 2）を要求する（#494 受け入れ基準 3 項）"
);

/// 次元（M／N／K）1 つぶんの焼き込み方式（イシュー #516）。
///
/// `Dynamic` はカーネル引数（`m`/`n`/`k`）をそのまま `#define DIM_* <param>`
/// として間接参照する既定形（現行カーネルとプリプロセス後等価）。
/// `Static(value)` はコンパイル時定数として焼き込み、当該次元の境界比較・
/// 添字計算を NVRTC のコンパイル時定数畳み込みへ委ねる（受け入れ基準 2
/// 「コンパイル時展開」）。どの次元をいつ静的化するかの選択ポリシーは
/// 後続 #519（C-7）のスコープであり、本モジュールは機構のみを提供する。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DimSpec {
    /// カーネル引数をそのまま使う（既定）。
    Dynamic,
    /// この次元をコンパイル時定数として焼き込む。
    ///
    /// `mod kernels_mma` は非公開モジュール（`lib.rs` の `mod kernels_mma;`）
    /// のため、本モジュール内から `Static` を構築する経路が現状テスト
    /// コードのみだと rustc の dead-code 解析が誤検知する。本 variant は
    /// 後続 #519（次元別静的化選択ポリシー）・#521（段数逆算）が実利用
    /// する機構の一部であるため `#[allow(dead_code)]` を付す（本ファイル
    /// 冒頭の他の定数に対する同種の判断と同じ方針）。
    #[allow(dead_code)]
    Static(u32),
}

impl DimSpec {
    /// この次元が実際の起動時形状 `actual`（カーネル引数として渡す
    /// `m`/`n`/`k`）と整合するかを fail-closed で検査する（PR #643
    /// codex-review P1 指摘への対応）。
    ///
    /// `validate_mma_kernel_config`／`validate_wmma_tf32_opt_config`／
    /// `validate_wmma_f16_opt_config` 等は `DimSpec::Static` が非ゼロで
    /// あることしか検査しない。生成ソースは `Static(value)` を境界比較・
    /// A/B/C ストライドへ直接コンパイル時定数として焼き込むため、コンパイル
    /// 済み関数を実際に起動する `m`/`n`/`k` が `value` と食い違うと、REQ-8
    /// （`.claude/rules/coding-rust.md`「カーネル実装の境界検査」）のガード
    /// が実バッファ境界ではなく静的値を基準に通過し、境界外アクセスに
    /// なりうる。[`MmaKernelConfig::validate_launch_shape`]／
    /// `WmmaOptKernelConfig::validate_launch_shape`（`kernels_wmma_opt.rs`）
    /// はこのメソッドへ委譲し、[`RenderedMmaKernel::validate_launch_shape`]
    /// が config を保持したまま起動前検査を強制する構造的な契約になって
    /// いる（doc comment のみに依存する「must call」注意書きでは呼び忘れを
    /// 防げないため）。`Dynamic` はカーネル引数をそのまま使うため常に整合する。
    ///
    /// `mod kernels_mma` が非公開モジュールのため、`render_mma_f16` 同様
    /// 現状の `gemm_mma.rs`（既定＝全 `Dynamic` config のみ消費）からは
    /// 呼ばれず dead-code 解析が誤検知する。`#[allow(dead_code)]` の理由は
    /// [`render_mma_f16`] と同じ。
    #[allow(dead_code)]
    pub fn matches_launch_dim(self, actual: u32) -> Result<(), CudaError> {
        match self {
            DimSpec::Dynamic => Ok(()),
            DimSpec::Static(value) if value == actual => Ok(()),
            DimSpec::Static(value) => Err(CudaError::InvalidKernelConfig {
                detail: format!(
                    "static dim value ({value}) does not match actual launch shape \
                     ({actual}); launching a kernel compiled with a mismatched static \
                     dimension would let REQ-8 boundary checks pass against the wrong bound"
                ),
            }),
        }
    }
}

/// mma f16 カーネルの入出力 dtype 識別子。
///
/// 現状 f16 経路のみのため variant は 1 つだが、`MmaKernelConfig` に
/// フィールドとして持たせておくことで、後続 #504（キャッシュキー構成
/// 要素としての `Hash + Eq` 導出）が dtype を区別できる形にする（実装計画
/// 4.1 節「dtype の差し替え」）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MmaDtype {
    #[default]
    F16,
}

/// `render_mma_f16` に渡す構成値（shape／タイル／段数／dtype）。
///
/// `Hash + Eq` を導出可能な単純型に留めているのは、後続 #504（descriptor・
/// コンパイルキャッシュキー）がそのまま鍵の構成要素として使えるように
/// するため（実装計画 10 節「スコープ外・引き継ぎ」参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MmaKernelConfig {
    /// ブロックタイル M（`MMA_M` の倍数・8 の倍数必須）。
    pub bm: u32,
    /// ブロックタイル N（`MMA_N` の倍数・8 の倍数必須）。
    pub bn: u32,
    /// ブロックタイル K（`MMA_K` の倍数必須）。
    pub bk: u32,
    /// `cp.async` パイプラインの段数。`wait_group` drain 分岐が
    /// `STAGES == 3` 前提の二値分岐のため、現状 3 以外は
    /// [`validate_mma_kernel_config`] が拒否する（段数可変化は #492）。
    pub stages: u32,
    /// M 次元の焼き込み方式。
    pub dim_m: DimSpec,
    /// N 次元の焼き込み方式。
    pub dim_n: DimSpec,
    /// K 次元の焼き込み方式。
    pub dim_k: DimSpec,
    /// 入出力 dtype 識別子。
    pub dtype: MmaDtype,
}

impl MmaKernelConfig {
    /// [`RenderedMmaKernel::validate_launch_shape`] の実体（起動前検査）。
    /// `dim_m`/`dim_n`/`dim_k` それぞれについて [`DimSpec::matches_launch_dim`]
    /// を実引数 `m`/`n`/`k` に対して検査し、`Static` 次元の食い違いを
    /// fail-closed で拒否する（PR #643 codex-review P1 指摘への対応）。
    /// `Dynamic` のみの既定 config では常に `Ok`。
    #[allow(dead_code)] // 理由は matches_launch_dim と同じ（非公開モジュール）
    pub fn validate_launch_shape(&self, m: u32, n: u32, k: u32) -> Result<(), CudaError> {
        self.dim_m.matches_launch_dim(m)?;
        self.dim_n.matches_launch_dim(n)?;
        self.dim_k.matches_launch_dim(k)?;
        Ok(())
    }

    /// `m`/`n` から実際の起動グリッドを構築する（`gemm_mma.rs::mma_launch_config`
    /// と同じ「1 warp = C の `MMA_M x MMA_N` タイル 1 個」設計を、既定タイル
    /// 定数〈`MMA_BM`/`MMA_BN`〉ではなく本 `cfg` のタイル値〈`self.bm`/`self.bn`〉
    /// に一般化したもの。[`CompiledMmaKernel::launch_f16`] が本メソッドの
    /// 戻り値を内部でのみ使い、呼び出し元へは一切公開しないことで、
    /// 検証済み `m`/`n`/`k` とは無関係な grid/block 設定を持ち込んで起動
    /// する経路を型で塞ぐ（`CompiledMmaKernel` ドキュメンテーションコメント
    /// 参照）。
    ///
    /// 静的共有メモリ（`__shared__` 配列としてカーネルソースへ焼き込み
    /// 済み）のみを使う設計のため `shared_mem_bytes` は常に 0
    /// （`gemm_mma.rs::mma_launch_config`／`gemm_wmma.rs::wmma_launch_config`
    /// と同じ契約）。
    #[allow(dead_code)] // 理由は validate_launch_shape と同じ（非公開モジュール）
    pub fn launch_config(&self, m: u32, n: u32) -> LaunchConfig {
        // #493 の warp あたり 2x2 レジスタブロッキング導入後は、1 warp が
        // 担当する C タイル実寸は `MMA_WARP_M x MMA_WARP_N`（`MMA_M`/`MMA_N`
        // ではない）。カーネルソース側 `WARPS_N` マクロ（本ファイル
        // `render_mma_f16_unchecked`）と同じ式に揃える必要がある
        // （不一致は warp_row/warp_col の導出がブロック内の実際の warp
        // 配置と食い違い、境界外書き込み・誤結果につながる）。
        let warps_m = self.bm / MMA_WARP_M;
        let warps_n = self.bn / MMA_WARP_N;
        LaunchConfig {
            grid_dim: (n.div_ceil(self.bn), m.div_ceil(self.bm), 1),
            block_dim: (warps_m * warps_n * 32, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

impl Default for MmaKernelConfig {
    /// 現行の Rust 側タイル定数（唯一の真実源。`MMA_BM` 等）をそのまま
    /// 初期値とする。全次元 `Dynamic`（既定は静的次元焼き込みなし。
    /// 実装計画 4.1 節）。
    fn default() -> Self {
        Self {
            bm: MMA_BM,
            bn: MMA_BN,
            bk: MMA_BK,
            stages: MMA_STAGES,
            dim_m: DimSpec::Dynamic,
            dim_n: DimSpec::Dynamic,
            dim_k: DimSpec::Dynamic,
            dtype: MmaDtype::F16,
        }
    }
}

/// [`MmaKernelConfig`] の不変条件を、実際にカーネルソースへ展開する前に
/// fail-closed で検査する（A03 対策。実装計画 4.2 節）。既存 const
/// アサーション（本ファイル冒頭）の非既定構成向け一般化にあたる。
fn validate_mma_kernel_config(cfg: &MmaKernelConfig) -> Result<(), CudaError> {
    let invalid = |detail: String| CudaError::InvalidKernelConfig { detail };

    if cfg.bm == 0 || cfg.bn == 0 || cfg.bk == 0 || cfg.stages == 0 {
        return Err(invalid("bm/bn/bk/stages must all be non-zero".to_string()));
    }
    // #493 の warp あたり 2x2 レジスタブロッキング後、1 warp が担当する C
    // タイル実寸は `MMA_WARP_M x MMA_WARP_N`（`MMA_M`/`MMA_N` ではない）。
    // `launch_config`／`render_mma_f16_unchecked` の `WARPS_N` マクロは
    // いずれも `bm`/`bn` を `MMA_WARP_M`/`MMA_WARP_N` で除算するため、
    // ここでの倍数検査も同じ除数で行わなければ `bm`/`bn` が `MMA_M`/
    // `MMA_N` の倍数だが `MMA_WARP_M`/`MMA_WARP_N` の倍数ではない構成
    // （例: bm=16・bn=8）が render/compile まで通過し、`warps_m`/
    // `warps_n` が 0 になって `block_dim.x=0` の無効な起動設定・生成
    // ソース内 `WARPS_N=0` を生じる（PR #643 codex-review P1 指摘への
    // 対応。REQ-8 の非既定構成 fail-closed 検査契約）。`MMA_WARP_M`/
    // `MMA_WARP_N` は `MMA_M`/`MMA_N` の倍数（本ファイル `MMA_WARP_M`/
    // `MMA_WARP_N` 定数直下参照）のため、この検査は旧 `MMA_M`/`MMA_N`
    // 基準の検査を包含する。
    if !cfg.bm.is_multiple_of(MMA_WARP_M) {
        return Err(invalid(format!(
            "bm ({}) must be a multiple of MMA_WARP_M ({MMA_WARP_M})",
            cfg.bm
        )));
    }
    if !cfg.bn.is_multiple_of(MMA_WARP_N) {
        return Err(invalid(format!(
            "bn ({}) must be a multiple of MMA_WARP_N ({MMA_WARP_N})",
            cfg.bn
        )));
    }
    if !cfg.bk.is_multiple_of(MMA_K) {
        return Err(invalid(format!(
            "bk ({}) must be a multiple of MMA_K ({MMA_K})",
            cfg.bk
        )));
    }
    // 冒頭 const アサーション `MMA_K_STEPS_PER_STAGE >= 2`（本ファイル
    // §297-303）の非既定構成向け一般化。1 ステージあたりの `mma.sync`
    // 呼び出し回数（`bk / MMA_K`）が 1 だと cp.async ソフトウェア
    // パイプライン（現ステージのロード完了を待つ間に次ステージを issue
    // する設計）が成立せず、`bk=16`（`MMA_K` の倍数ではあるが 1 ステップ
    // 分しかない）のような構成が render/compile を素通りしてしまう
    // （PR #643 codex-review Medium 指摘への対応。`bk % MMA_K == 0` だけ
    // では不変条件を再現できていなかった）。
    let k_steps_per_stage = cfg.bk / MMA_K;
    if k_steps_per_stage < 2 {
        return Err(invalid(format!(
            "bk / MMA_K ({k_steps_per_stage}) must be >= 2 (MMA_K_STEPS_PER_STAGE \
             invariant; the cp.async software pipeline needs at least 2 mma.sync \
             steps per stage, bk={} MMA_K={MMA_K})",
            cfg.bk
        )));
    }
    // cp.async 16 バイト転送粒度の前提（本ファイル冒頭コメント「整列制約」）。
    if !cfg.bm.is_multiple_of(8) || !cfg.bn.is_multiple_of(8) {
        return Err(invalid(format!(
            "bm ({}) and bn ({}) must both be multiples of 8 (cp.async 16-byte transfer granularity)",
            cfg.bm, cfg.bn
        )));
    }

    // `launch_config`／生成ソース `WARPS_N` と同じ除数（`MMA_WARP_M`/
    // `MMA_WARP_N`）で warp 数を導出する（上記倍数検査のコメント参照）。
    let warps_m = cfg.bm / MMA_WARP_M;
    let warps_n = cfg.bn / MMA_WARP_N;
    let threads = warps_m
        .checked_mul(warps_n)
        .and_then(|w| w.checked_mul(32))
        .ok_or_else(|| invalid("block thread count overflow".to_string()))?;
    if threads > 1024 {
        return Err(invalid(format!(
            "block thread count {threads} exceeds CUDA's per-block limit (1024)"
        )));
    }

    // 静的共有メモリ予算（本ファイル冒頭コメント「タイル構成」・
    // `MMA_SHARED_MEM_BYTES` ドキュメンテーションコメント参照）。
    // #498: `MMA_A_PAD`/`MMA_B_PAD`（既定 config 専用の定数）と同じ式
    // （`bk + 8`/`bn + 8`）を非既定 config 向けに一般化して使う。非既定
    // config でもカーネルソース側は常に `A_PAD`/`B_PAD` マクロ（下記
    // `render_mma_f16_unchecked` 参照）で `as_tile`/`bs_tile` を確保する
    // ため、パディング前の `bm*bk + bk*bn` のまま検査すると実際の SMEM
    // 使用量より小さく見積もり、48KiB 上限超過の構成を誤って通過させて
    // しまう（`docs/perf/cuda-gemm-mma-bank-conflict.md` §2 参照）。
    let a_pad = cfg
        .bk
        .checked_add(8)
        .ok_or_else(|| invalid("A tile padded row width overflow".to_string()))?;
    let b_pad = cfg
        .bn
        .checked_add(8)
        .ok_or_else(|| invalid("B tile padded row width overflow".to_string()))?;
    let smem_bytes = cfg
        .bm
        .checked_mul(a_pad)
        .and_then(|a| cfg.bk.checked_mul(b_pad).and_then(|b| a.checked_add(b)))
        .and_then(|sum| sum.checked_mul(2))
        .and_then(|v| v.checked_mul(cfg.stages))
        .ok_or_else(|| invalid("shared memory byte count overflow".to_string()))?;
    if smem_bytes > MMA_STATIC_SMEM_LIMIT_BYTES {
        return Err(invalid(format!(
            "static shared memory usage {smem_bytes} bytes exceeds the 48KiB per-block limit"
        )));
    }

    // wait_group drain 分岐（`if (t == num_k_tiles - 1)`）は STAGES=3 固定
    // 前提の二値分岐（本ファイル冒頭コメント「命令選定」）。段数可変化は
    // #492（Phase B）のスコープであり、それまで render 側で拒否する。
    if cfg.stages != 3 {
        return Err(invalid(format!(
            "stages ({}) must be 3; the cp.async wait_group drain branch \
             (if (t == num_k_tiles - 1)) is hard-coded for STAGES=3. \
             variable stage counts are tracked by #492",
            cfg.stages
        )));
    }

    for (name, spec) in [
        ("dim_m", cfg.dim_m),
        ("dim_n", cfg.dim_n),
        ("dim_k", cfg.dim_k),
    ] {
        if let DimSpec::Static(0) = spec {
            return Err(invalid(format!(
                "{name} static value must not be zero (degenerate dimension)"
            )));
        }
    }

    // PR #643 codex-review P0 指摘への対応: `DIM_K`/`DIM_N` は
    // `LOAD_A_STAGE`/`LOAD_B_STAGE`（本ファイル `MMA_F16_BODY`）で A/B の
    // 行ストライドとして直接使われ、`mma_cp_async16` の 16 バイト転送先
    // アドレス計算（`&a[gr_c * DIM_K + gc_c]`／`&b[gr_c * DIM_N + gc_c]`）に
    // 畳み込まれる。`Static` で焼き込む場合、値が 8 要素（f16 で 16 バイト）
    // の倍数でないと非既定行の開始アドレスが 16 バイト境界からずれ、
    // `cp.async.cg.shared.global` が未定義動作になる（REQ-8 境界検査の
    // 前提が崩れる）。`dim_m` はストライドに使われず境界クランプにのみ
    // 使われるため対象外（`LOAD_A_STAGE` の `gr_c` は行方向で整列を問わない
    // 旨のコメント参照）。ゼロ拒否（上記ループ）とは独立の検査。
    for (name, spec) in [("dim_n", cfg.dim_n), ("dim_k", cfg.dim_k)] {
        if let DimSpec::Static(value) = spec
            && !value.is_multiple_of(8)
        {
            return Err(invalid(format!(
                "{name} static value ({value}) must be a multiple of 8 to preserve \
                 cp.async's 16-byte transfer alignment for A/B row strides"
            )));
        }
    }

    Ok(())
}

/// [`CompiledMmaKernel::launch_f16`] の起動前検査の一部として呼ばれる、
/// K タイル反復のインデックス算術（カーネル内 `LOAD_A_STAGE`/
/// `LOAD_B_STAGE` マクロが計算する `s * bk + col0`。`s` はステージ番号
/// 〈最大 `num_k_tiles - 1`〉、`col0` はタイル内オフセット〈最大
/// `bk - 1`〉）が `int`（i32）算術でオーバーフローしないことを検証する
/// 純粋関数（`self` を要求しないため device 実機なしで単体テスト可能。
/// codex-review 指摘・PR #643 再レビュー）。
///
/// `gemm.rs::validate_wmma_tf32_opt_k_bound` と同型だが、こちらは
/// `kernels_mma::MMA_BK` 固定ではなく引数 `bk`〈テンプレート展開元の
/// タイル値。イシュー #516 で `bk` が可変になったため一般化が必須〉で
/// 計算する。実際にカーネルが計算しうる最大インデックスは `ceil(k / bk) *
/// bk - 1`（`k == 0` のときは計算自体が発生しないため 0）であり、これが
/// `i32::MAX` を超えると当該算術が i32 の範囲でオーバーフローしうる
/// （符号付きオーバーフロー後に境界ガード式 `gk < DIM_K` が誤って成立し
/// REQ-8 の境界チェックを迂回しうるため P0）。`validate_mma_kernel_config`
/// は `bk` が 8/`MMA_K` の倍数であること等は検査するが、`k` との組合せに
/// よる算術オーバーフローは起動時の `k` に依存するためここで検査する。
fn validate_mma_k_tile_bound(k: u32, bk: u32) -> Result<(), CudaError> {
    let tile = bk as u64;
    let max_computed_index = if k == 0 {
        0
    } else {
        (k as u64).div_ceil(tile) * tile - 1
    };
    if max_computed_index > i32::MAX as u64 {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "k tile-index arithmetic for mma.sync kernel would overflow i32: k={k}, \
                 max_computed_index={max_computed_index}, bk={bk}"
            ),
        });
    }
    Ok(())
}

/// `#define {macro_name} <param_name または static 値>` を 1 行生成する。
///
/// `kernels_wmma_opt.rs` からも呼ばれる（同モジュールは既に本ファイルの
/// [`DimSpec`] を import 済みのため `pub(crate)` として re-export し、
/// 重複定義を避ける。レビュー指摘: 「用途がモジュールをまたぐためここでも
/// 定義する」という理由は import 経路が既に存在するため根拠として弱い）。
pub(crate) fn render_dim_define(macro_name: &str, param_name: &str, spec: DimSpec) -> String {
    match spec {
        DimSpec::Dynamic => format!("#define {macro_name} {param_name}"),
        DimSpec::Static(value) => format!("#define {macro_name} {value}"),
    }
}

/// 検証済み [`MmaKernelConfig`] からカーネルソース文字列を組み立てる
/// 内部関数（`validate_mma_kernel_config` を経ない infallible 経路。
/// 呼び出し元は必ず検証済みの `cfg` を渡すこと）。
fn render_mma_f16_unchecked(cfg: &MmaKernelConfig) -> String {
    // #493 の warp あたり 2x2 レジスタブロッキング導入後、1 warp が担当する
    // C タイル実寸は `MMA_WARP_N`（`MMA_N` ではない）。カーネルソース側
    // `warp_col = warp_id % WARPS_N` はこの値を前提に warp グリッドへの
    // 分配を計算するため、[`MmaKernelConfig::launch_config`] と同じ式へ
    // 揃える（本ファイル同メソッドのコメント参照）。
    let warps_n = cfg.bn / MMA_WARP_N;
    // #498: 共有メモリのバンクコンフリクト対策パディング（本ファイル冒頭
    // コメント「バンクコンフリクト対策」・`MMA_A_PAD`/`MMA_B_PAD` 定数
    // 直下のドキュメンテーションコメント参照）。既定 config 専用の
    // `MMA_A_PAD`/`MMA_B_PAD`（`MMA_BK + 8`/`MMA_BN + 8`）と同じ式を
    // 非既定 `cfg.bk`/`cfg.bn` へ一般化する（`validate_mma_kernel_config`
    // の smem_bytes 計算と同じ式。両者は独立に定義されているため、
    // 式を変更する場合は両方を揃えて更新すること）。
    let a_pad = cfg.bk + 8;
    let b_pad = cfg.bn + 8;
    let dim_m_define = render_dim_define("DIM_M", "m", cfg.dim_m);
    let dim_n_define = render_dim_define("DIM_N", "n", cfg.dim_n);
    let dim_k_define = render_dim_define("DIM_K", "k", cfg.dim_k);
    format!(
        "\n#include <cuda_fp16.h>\n\n\
         #define MMA_M {MMA_M}\n\
         #define MMA_N {MMA_N}\n\
         #define MMA_K {MMA_K}\n\
         #define BM {bm}\n\
         #define BN {bn}\n\
         #define BK {bk}\n\
         #define WARPS_N {warps_n}\n\
         #define WARP_TILES_M {MMA_WARP_TILES_M}\n\
         #define WARP_TILES_N {MMA_WARP_TILES_N}\n\
         #define STAGES {stages}\n\
         #define A_PAD {a_pad}\n\
         #define B_PAD {b_pad}\n\
         {dim_m_define}\n\
         {dim_n_define}\n\
         {dim_k_define}\n\
         \n{MMA_F16_BODY}",
        bm = cfg.bm,
        bn = cfg.bn,
        bk = cfg.bk,
        stages = cfg.stages,
    )
}

/// [`render_mma_f16`] が返す、展開済みカーネルソースと展開元
/// [`MmaKernelConfig`] を 1 個にまとめた descriptor（PR #643 codex-review
/// P1 指摘への対応）。
///
/// フィールドは非公開。生ソースを `&str`/`String` として外へ返す公開
/// メソッドは一切持たない（PR #643 codex-review 再々指摘への対応:
/// 従来の `source_for_launch(m, n, k) -> Result<&str, _>` は「検査を通ら
/// ないとソースを取得できない」構造だったが、取得した `&str` を NVRTC へ
/// 渡して得た `CudaFunction` は呼び出し元がその後どこにでも保持でき、
/// 2 回目以降の起動が `validate_launch_shape` を経由するかは呼び出し元の
/// 実装規律に委ねられてしまっていた（「独立した public メソッドを都度
/// 呼ぶ契約」は型レベルで強制されていなかった）。本構造体はこの穴を
/// ふさぐため、ソースの受け渡し先を [`Self::compile`] 内部（NVRTC
/// コンパイル・固定エントリポイントのロード）に限定する。ソース文字列が
/// 外部（呼び出し元）へ渡らないため、コンパイル自体が本 descriptor の
/// 管理下でのみ、かつ必ず `self.source` に対して起こる（PR #643
/// codex-review 再々々々々指摘〈P0〉への対応。`Self::compile`
/// ドキュメンテーションコメント参照。旧来クロージャ方式では呼び出し元が
/// `self.source` を無視した `CudaFunction` を返せてしまっていた）。
///
/// 後続 #504（コンパイルキャッシュキー）もこの descriptor をそのまま
/// 鍵の構成要素として使える。
///
/// `mod kernels_mma` が非公開モジュールのため、既定構成のみを使う現状の
/// `gemm_mma.rs` からは本構造体・以下の全メソッドが呼ばれず dead-code
/// 解析が誤検知する。`#[allow(dead_code)]` の理由は [`render_mma_f16`]
/// と同じ。
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RenderedMmaKernel {
    source: String,
    cfg: MmaKernelConfig,
}

impl RenderedMmaKernel {
    /// `self.cfg`（`MmaKernelConfig` の `dim_m`/`dim_n`/`dim_k`）から
    /// コンパイルキャッシュキーの構成要素 [`crate::nvrtc::CudaKernelDescriptor`]
    /// を内部導出する（イシュー #511・C-4。実装計画 §3.4 点 1）。
    ///
    /// `gemm_auto.rs::specialized_mma_descriptor` と同じ「config とキーの
    /// 構造的一致」原則を、`RenderedMmaKernel` が保持する `cfg` のみから
    /// 満たす: `DimSpec::Static(v)` な次元はコンパイル時定数化された次元
    /// （`compiled_dims` の当該フラグを立て、shape 実値は `v`）、
    /// `DimSpec::Dynamic` な次元は非定数化（フラグを立てず、shape 実値は
    /// `0`）として扱う。`CudaKernelDescriptor::cache_key_shape` が非
    /// 定数化次元を sentinel `0` へ正規化する契約（同型ドキュメンテー
    /// ションコメント参照）のため、`Dynamic` 次元へ渡す実値は結果に影響
    /// しない。これにより、`specialized_mma_config(shape, compiled)` →
    /// `render_mma_f16` を経て得た `RenderedMmaKernel` に対して本メソッドを
    /// 呼んだ場合、`gemm_auto.rs::specialized_mma_descriptor(shape,
    /// compiled)` が直接返す descriptor と（`cache_key_shape` を介して）
    /// 同一のキャッシュキーになる。外部から任意の descriptor を注入させず
    /// `cfg` から構造的に導出することで、key と source（`self.source`）の
    /// 乖離が生じない（`CudaKernelCacheKey` ドキュメンテーションコメント
    /// 「C-5」節と同じ意図）。
    ///
    /// 唯一の呼び出し元 [`Self::cache_key`]（延いては [`Self::compile`]）が
    /// `internal-diagnostics` feature（既定 off）ゲート経由の
    /// `gemm_auto.rs::SpecializedMmaKernelHandle::compile` からのみ呼ばれる
    /// ため、既定ビルドでは crate 内呼び出し元が存在せず dead-code 解析が
    /// 誤検知する。`#[allow(dead_code)]` の理由は [`Self::compile`] と同じ。
    #[allow(dead_code)]
    fn cache_descriptor(&self) -> Result<crate::nvrtc::CudaKernelDescriptor, CudaError> {
        let split = |dim: DimSpec| -> (u32, bool) {
            match dim {
                DimSpec::Dynamic => (0, false),
                DimSpec::Static(value) => (value, true),
            }
        };
        let (m, is_m_compiled) = split(self.cfg.dim_m);
        let (n, is_n_compiled) = split(self.cfg.dim_n);
        let (k, is_k_compiled) = split(self.cfg.dim_k);
        crate::nvrtc::CudaKernelDescriptor::new_with_compiled_dims(
            "mma_f16",
            tensor_core::dispatch::GemmShape::new(m, n, k),
            self.cfg.bm,
            self.cfg.bn,
            self.cfg.bk,
            self.cfg.stages,
            match self.cfg.dtype {
                MmaDtype::F16 => tensor_core::dispatch::DType::F16,
            },
            crate::nvrtc::CompiledDims::new(is_m_compiled, is_n_compiled, is_k_compiled),
        )
    }

    /// [`Self::cache_descriptor`]・`device`（compute capability・arch）・
    /// [`crate::nvrtc::nvrtc_version`]・`self.source`（最終レンダー済み
    /// ソース全体。C-5・#514 契約）から [`crate::nvrtc::CudaKernelCacheKey`]
    /// を構築する（イシュー #511・C-4）。
    ///
    /// `compile_flags` は `--gpu-architecture=<arch>`（`compile_ptx` が
    /// 実際に `CompileOptions::arch` へ渡す値と同一の `device.arch()`。
    /// `nvrtc.rs` のキャッシュキーテスト群と同じ表記規約）のみを含める。
    /// include path フォールバック（`compile_ptx` の `CUDA_INCLUDE_PATH`／
    /// 既定パス探索）はキーへ含めない: toolkit ヘッダの ABI 差は
    /// `nvrtc_version` フィールドが追従するため（`CudaKernelCacheKey`
    /// ドキュメンテーションコメント「ソースコード断片」節と同じ判断）。
    ///
    /// `#[allow(dead_code)]` の理由は [`Self::cache_descriptor`] と同じ
    /// （唯一の呼び出し元 [`Self::compile`] が `internal-diagnostics`
    /// feature ゲート経由でのみ到達するため、既定ビルドで dead-code 解析
    /// が誤検知する）。
    #[allow(dead_code)]
    fn cache_key(
        &self,
        device: &crate::device::CudaDevice,
    ) -> Result<crate::nvrtc::CudaKernelCacheKey, CudaError> {
        let descriptor = self.cache_descriptor()?;
        let compile_flags = vec![format!("--gpu-architecture={}", device.arch())];
        crate::nvrtc::CudaKernelCacheKey::from_device(
            descriptor,
            device,
            compile_flags,
            self.source.clone(),
        )
    }

    /// カーネルソースを NVRTC コンパイル → 固定エントリポイント
    /// `"gemm_mma_f16"` のロードまで descriptor 内部で完結させ、結果
    /// （`CudaFunction`）と展開元 `cfg` を不可分に束ねた
    /// [`CompiledMmaKernel`] を返す唯一の公開経路（PR #643 codex-review
    /// 再々々々々指摘〈P0〉への対応: 従来はコンパイルをクロージャ
    /// `impl FnOnce(&str) -> Result<CudaFunction, CudaError>` へ委ねて
    /// いたが、クロージャは受け取ったソース文字列を無視して無関係な
    /// `CudaFunction`（別カーネル・別モジュールの関数）を返しても型上
    /// 検出できず、以降 `launch_f16` が前提とする「検証済み `cfg` と
    /// `self.func` が対応している」契約がホスト側からは検証不能なまま
    /// 崩れうる欠陥があった。`compile_ptx`・`load_module`・
    /// `load_function` をこのメソッド内で直接呼ぶことで、`self.source`
    /// 以外のソースから得た `CudaFunction` を `CompiledMmaKernel` へ
    /// 格納する経路自体を型・構造の両面で消す。
    ///
    /// # 3 段フォールバック（イシュー #511・C-4。実装計画 §3.4）
    ///
    /// 1. **プロセス内 LRU**（[`crate::module_cache::KernelModuleCache`]）:
    ///    同一 `CudaContext`・同一キャッシュキーでロード済みの
    ///    `Arc<CudaModule>` があれば `cuModuleGetFunction`（軽量）のみで
    ///    済ませ、NVRTC 再コンパイル・`load_module` 再ロードを回避する。
    /// 2. **ディスクキャッシュ**（[`crate::nvrtc::load_cache_entry`]。
    ///    C-3・#509）: ソース全文のバイト単位照合込みでヒット判定のみ
    ///    行う。**ヒットしてもディスク上の `kernel.ptx` は実行入力として
    ///    使わない**（イシュー #511 PR #703 codex-review P0 再指摘対応。
    ///    詳細理由は下記実装のコメント参照）。ヒット／ミスいずれの場合も
    ///    3 段目の NVRTC 直コンパイルへ進み、このプロセス内で生成した
    ///    PTX のみを `load_module` する。
    /// 3. **NVRTC 直コンパイル**（2 段目のヒット／ミスを問わず必ず実行):
    ///    `compile_ptx` 実行後、ディスクに当該キーのエントリがまだ
    ///    なければ [`crate::nvrtc::store_cache_entry`] でディスクへ保存
    ///    する（C-2〜C-3 が用意した導線への結線。設計文書
    ///    `docs/cuda-jit-cache-design.md` C-4 節）。
    ///
    /// いずれの段でロードした `Arc<CudaModule>` も、最終的に
    /// [`crate::module_cache::KernelModuleCache::insert`] へ登録し、次回
    /// 以降のプロセス内再利用に備える。
    ///
    /// # 縮退方針（fail-safe。実装計画 §3.5）
    ///
    /// プロセス内 LRU（容量設定不正・`Mutex` poison 等）・ディスク
    /// キャッシュ（`workspace_root` 解決不能・fs I/O 失敗）いずれの失敗も
    /// コンパイル失敗にせず、直後の段（最終的には NVRTC 直コンパイル）へ
    /// 静かにフォールバックする。両キャッシュはあくまで最適化であり、
    /// 数値正しさは NVRTC 直コンパイル経路・ディスクキャッシュのソース
    /// 全文照合（[`crate::nvrtc::load_cache_entry`]。ハッシュ衝突安全弁）の
    /// いずれでも独立に保たれるため、キャッシュ層の可用性低下が誤った
    /// PTX の実行につながることはない（`module_cache.rs` 冒頭ドキュメン
    /// テーションコメント「縮退方針」・`crate::nvrtc::runtime_workspace_root`
    /// ドキュメンテーションコメントと同じ判断）。
    #[allow(dead_code)]
    pub fn compile(
        &self,
        device: &crate::device::CudaDevice,
    ) -> Result<CompiledMmaKernel, CudaError> {
        let ctx = device.context();
        let key = self.cache_key(device)?;

        // 1 段目: プロセス内 LRU。キャッシュ自体が利用不能（容量設定
        // 不正・poison）でもフォールバックし続ける（縮退方針）。
        let module_cache = crate::module_cache::KernelModuleCache::global().ok();
        if let Some(cache) = module_cache
            && let Ok(Some(module)) = cache.get(ctx, &key)
        {
            let func = module.load_function("gemm_mma_f16")?;
            return Ok(CompiledMmaKernel {
                func,
                cfg: self.cfg,
            });
        }

        // ディスクキャッシュの読み書きに使う `workspace_root`。解決失敗も
        // 縮退運転（ディスクキャッシュなし）へ倒す（`runtime_workspace_root`
        // ドキュメンテーションコメント参照）。
        let workspace_root = crate::nvrtc::runtime_workspace_root().ok();

        // 2 段目: ディスクキャッシュ（ソース全文のバイト照合込み）。
        //
        // **ヒットしてもディスク上の `kernel.ptx` バイト列を実行入力として
        // 使わない（イシュー #511 PR #703 codex-review P0 再指摘対応）**:
        // `load_cache_entry` がここで検証するのは「保存済み `kernel.cu` が
        // `self.source` とバイト単位で一致すること」のみであり、同一
        // ディレクトリの `kernel.ptx` がそのソースから実際に生成された
        // 成果物であることまでは検証しない。追加した権限検査
        // （`nvrtc::is_cache_entry_permission_untrusted`。同一 uid のみ
        // 信頼）を経てもなお、同一 uid の別プロセス・侵害プロセスが
        // `kernel.cu` を保ったまま `kernel.ptx` だけを任意の有効な PTX へ
        // 差し替える攻撃は防げない（ファイルを書き換えられる主体は同じ
        // uid で新たな正当エントリも作れてしまうため、暗号学的ダイジェスト
        // をディスク上へ同居させても認証にはならない。許容依存 8 区分
        // `.claude/rules/deps-policy.md` に署名検証用クレートは含まれず
        // 新規追加はユーザー承認が要るため本 PR のスコープでは導入
        // しない）。よって disk hit は「ソース一致が確認できた」という
        // シグナルとしてのみ扱い、実際にロードする PTX は常にこのプロセス
        // 内で NVRTC が生成したものに限る（hit／miss いずれの分岐でも
        // `compile_ptx` を経由する。下記参照）。`load_cache_entry` 呼び
        // 出しそのもの（#509 の fs 配線・権限検査）は将来、認証済み検証
        // 手段を導入した際に実行入力として再有効化できるよう維持する。
        let disk_hit = workspace_root.as_ref().and_then(|root| {
            crate::nvrtc::load_cache_entry(root, &key, &self.source)
                .ok()
                .flatten()
        });

        // 3 段目: NVRTC 直コンパイル。上記の理由によりディスクキャッシュが
        // 保持する PTX バイト列は信頼せず、hit／miss いずれの場合も
        // このプロセス内で NVRTC を実行して得た PTX のみをロードする
        // （ディスク PTX を実行しない、という codex-review 指摘の対応案を
        // 採用。安全な認証手段〈署名／暗号学的バインディング〉を導入する
        // までの間の恒常方針とする）。
        let ptx = crate::nvrtc::compile_ptx(&self.source, device.arch())?;
        if disk_hit.is_none()
            && let Some(root) = workspace_root.as_ref()
        {
            // C-4 導線: コンパイル成功後、ディスクに当該キーのエントリが
            // まだない場合のみ store_cache_entry を呼ぶ（設計文書
            // `docs/cuda-jit-cache-design.md` C-4 節）。保存失敗はコン
            // パイル結果自体には影響しない（縮退方針）ため戻り値は無視
            // する。上記のとおりこのエントリは将来の認証済み検証手段
            // 導入まで実行入力としては使われないが、#509 のディスク
            // 永続化自体は今後の再有効化に備えて維持する。
            let _ = crate::nvrtc::store_cache_entry(root, &key, &self.source, &ptx.to_src());
        }
        let module = ctx.load_module(ptx)?;

        // ロード済みモジュールをプロセス内 LRU へ登録する（挿入失敗＝
        // poison も縮退方針でコンパイル結果自体は返す）。
        if let Some(cache) = module_cache {
            let _ = cache.insert(ctx, key, std::sync::Arc::clone(&module));
        }

        let func = module.load_function("gemm_mma_f16")?;
        Ok(CompiledMmaKernel {
            func,
            cfg: self.cfg,
        })
    }

    /// テスト専用のソース内容検査アクセサ（`#[cfg(test)]` のためリリース
    /// ビルドには存在しない）。生成された `#define`／REQ-8 境界チェックの
    /// 文字列内容を検査するテストのみが使い、[`Self::compile`] が公開する
    /// 「検査を経ないと `CudaFunction` に到達できない」という契約には
    /// 影響しない（本番経路の公開 API には現れない）。
    #[cfg(test)]
    fn source(&self) -> &str {
        &self.source
    }
}

/// [`RenderedMmaKernel::compile`] が返す、コンパイル済み `CudaFunction`
/// と展開元 [`MmaKernelConfig`] を不可分に束ねた descriptor（PR #643
/// codex-review 再々指摘への対応）。
///
/// フィールドは非公開。`func` を取り出す・あるいは検証を経ずに起動できる
/// 公開経路は一切存在しない。唯一の起動経路 [`Self::launch_f16`] は、
/// PR #643 codex-review 再々々指摘（P0。`with_validated_function` が
/// `&CudaFunction` と `LaunchConfig` をクロージャへ渡していたため、
/// クロージャが渡された `LaunchConfig` を無視して独自の grid/block を
/// 構築し、検証済み `m`/`n`/`k` と無関係なバッファ・引数で起動できて
/// しまう欠陥があった）への対応として、クロージャ経由の起動 API を廃し、
/// 起動そのものをこのメソッド内で完結させる設計に変更した
/// （`gemm_mma.rs::CudaMmaGemm::launch_f16` と同型。呼び出し元は
/// `CudaFunction`／`LaunchConfig` のいずれにも触れられないため、検証済み
/// shape 由来の grid/block・引数以外での起動が構造的に不可能になる）。
#[allow(dead_code)]
pub struct CompiledMmaKernel {
    func: cudarc::driver::CudaFunction,
    cfg: MmaKernelConfig,
}

impl CompiledMmaKernel {
    /// 検証済み shape でのみ起動できる、`CudaFunction` へアクセスする
    /// 唯一の公開経路（PR #643 codex-review 再々指摘・再々々指摘・
    /// 再々々々指摘〈いずれも P0〉への対応。`CompiledMmaKernel`
    /// ドキュメンテーションコメント参照）。
    ///
    /// `Dynamic` 次元は起動ごとに異なる `m`/`n`/`k` を許容しうるため、
    /// 実際の起動直前に毎回このメソッドを経由する想定。手順は
    /// (1) [`MmaKernelConfig::validate_launch_shape`] で `Static` 次元と
    /// 実引数の不一致を fail-closed で拒否、(2)
    /// `gemm.rs::validate_gemm_dims`／`validate_output_len` で `a_dev`/
    /// `b_dev`/`c_dev` の長さが `m`/`n`/`k` と一致することを検証、(3)
    /// `m==0 || n==0` の no-op 形状（`gemm_mma.rs::CudaMmaGemm::run_f16` と
    /// 同じ根拠。0 次元 grid はドライバが拒否するためカーネル起動自体を
    /// 回避する。PR #643 codex-review P2 指摘への対応: 本メソッドは
    /// `CudaMmaGemm::run_f16` の no-op ガードを経由せず直接呼ばれうる
    /// テンプレート展開 API の唯一の起動経路であるため、このメソッド自身が
    /// no-op を吸収する必要がある）の早期 return、(4)
    /// `gemm_mma.rs::validate_mma_alignment` で `cp.async` の 16 バイト転送
    /// 粒度制約（`n`/`k` が 8 の倍数）を検証、(5) `self.cfg.bm` を単位に
    /// した grid_dim.y（`m.div_ceil(self.cfg.bm)`）が CUDA の grid y/z 上限
    /// （65,535）を超えないことを検証、してから初めて
    /// [`MmaKernelConfig::launch_config`] が導出した `LaunchConfig` で
    /// `self.func` を起動する。呼び出し元は `CudaFunction`／`LaunchConfig`
    /// のいずれも受け取らないため、検証済み shape・検証済みバッファ長と
    /// 無関係な grid/block・引数で起動する経路が型で塞がれる。
    ///
    /// no-op 判定をアライメント・grid 上限検証より前に置く（`gemm_mma.rs::
    /// CudaMmaGemm::run_f16` ドキュメンテーションコメントと同じ理由）:
    /// 例えば `(m,n,k)=(0,7,0)` のような有効な no-op 形状は `n=7` が
    /// 8 の倍数でないため、アライメント検証を先に行うと実際にはカーネルを
    /// 起動しない形状まで誤って `CudaError::InvalidShape` で拒否して
    /// しまう。
    ///
    /// `m==0 || n==0` の no-op 経路では `c_dev` を一切書き換えない
    /// （`validate_output_len` により `c_dev.len() == m*n == 0` が既に
    /// 保証されているため要素自体が存在せず、書き込み対象がない。
    /// `gemm_mma.rs::CudaMmaGemm::run_f16` の `m==0 || n==0` 早期 return が
    /// 新規 `Vec::new()` を返すのと異なり、本メソッドは呼び出し元所有の
    /// 既存バッファに対する操作のため「ゼロ初期化して返す」責務を持たない
    /// 契約である）。`k==0`（`m`/`n` は非ゼロ）は本 no-op 判定の対象外
    /// （thread fVS7 は `m`/`n` の no-op のみを指摘しており、`k==0` は
    /// `num_k_tiles=0` としてカーネル自体は起動するが計算を行わない別経路。
    /// 呼び出し元がこのケースで `c_dev` の初期値をどう扱うかは
    /// `CudaMmaGemm::run_f16` 側の `k==0` 早期 return が担う責務であり、
    /// 本メソッド単体の契約には含めない）。
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn launch_f16(
        &self,
        stream: &CudaStream,
        a_dev: &CudaSlice<f16>,
        b_dev: &CudaSlice<f16>,
        c_dev: &mut CudaSlice<f16>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        self.cfg.validate_launch_shape(m, n, k)?;
        crate::gemm::validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        crate::gemm::validate_output_len(c_dev.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        crate::gemm_mma::validate_mma_alignment(n, k)?;
        self.validate_grid_bounds(m)?;
        self.validate_k_tile_bound(k)?;

        let launch_config = self.cfg.launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: カーネル引数は a_dev/b_dev/c_dev（それぞれ上記
        // validate_gemm_dims/validate_output_len で m*k/k*n/m*n 要素の
        // 確保済みデバイスバッファであることを検証済み）と m_i/n_i/k_i の
        // 5 個・型・個数が、検証済みの m/n/k と 1:1 対応する。grid/block は
        // 同じく検証済みの m/n から launch_config が導出したもののみを
        // 使い、呼び出し元が独自に構築した LaunchConfig を持ち込む経路は
        // 存在しない。カーネル内の手動境界チェック（cp.async src-size
        // ゼロ充填・エピローグ guarded store。REQ-8）と合わせて OOB
        // 読み書きが起きない根拠とする。共有メモリは静的 `__shared__`
        // 配列のみを使用するため shared_mem_bytes は 0 のままでよい。
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

    /// grid_dim.y（`m.div_ceil(self.cfg.bm)`）が CUDA の grid y/z 上限
    /// （65,535。全 compute capability 共通）を超えないことを検証する
    /// （`gemm_mma.rs::validate_mma_grid_bounds` と同じ理由だが、既定タイル
    /// 定数〈`MMA_BM`〉固定ではなく本 `cfg.bm`〈テンプレート展開元のタイル
    /// 値〉で計算する点が異なる。超過するとホスト側の他の検証はすべて
    /// 通過した上で、ドライバのカーネル起動が失敗する）。
    fn validate_grid_bounds(&self, m: u32) -> Result<(), CudaError> {
        const MAX_GRID_DIM_Y: u32 = 65_535;
        let grid_y = m.div_ceil(self.cfg.bm);
        if grid_y > MAX_GRID_DIM_Y {
            return Err(CudaError::InvalidShape {
                detail: format!(
                    "mma.sync path grid_dim.y (m.div_ceil(bm)={grid_y}) exceeds CUDA's \
                     {MAX_GRID_DIM_Y} limit for grid dimensions y/z (bm={}); m={m} is too \
                     large",
                    self.cfg.bm
                ),
            });
        }
        Ok(())
    }

    /// K タイル反復のインデックス算術（カーネル内 `LOAD_A_STAGE`/
    /// `LOAD_B_STAGE` マクロが計算する `s * BK + col0`。`s` はステージ番号
    /// 〈最大 `num_k_tiles - 1`〉、`col0` はタイル内オフセット〈最大
    /// `BK - 1`〉）が `int`（i32）算術でオーバーフローしないことを検証する
    /// （codex-review 指摘・PR #643 再レビュー。`gemm.rs::
    /// validate_wmma_tf32_opt_k_bound` と同型だが、こちらは
    /// `kernels_mma::MMA_BK` 固定ではなく本 `cfg.bk`〈テンプレート展開元の
    /// タイル値。イシュー #516 で `bk` が可変になったため一般化が必須〉で
    /// 計算する）。実際にカーネルが計算しうる最大インデックスは
    /// `ceil(k / bk) * bk - 1`（`k == 0` のときは計算自体が発生しないため
    /// 0）であり、これが `i32::MAX` を超えると当該算術が i32 の範囲で
    /// オーバーフローしうる（符号付きオーバーフロー後に境界ガード式
    /// `gk < DIM_K` が誤って成立し REQ-8 の境界チェックを迂回しうるため
    /// P0）。`validate_mma_kernel_config` は `bk` が 8/`MMA_K` の倍数
    /// であること等は検査するが、`k` との組合せによる算術オーバーフローは
    /// 起動時の `k` に依存するためここで検査する。
    fn validate_k_tile_bound(&self, k: u32) -> Result<(), CudaError> {
        validate_mma_k_tile_bound(k, self.cfg.bk)
    }
}

/// f16 `mma.sync`/`ldmatrix`/`cp.async` GEMM のテンプレート化されたカーネル
/// ソースを、`cfg` の shape／タイル／段数で展開する（イシュー #516）。
///
/// 展開前に [`validate_mma_kernel_config`] で不変条件を fail-closed 検査
/// する（SMEM 予算・倍数関係・スレッド数上限・`stages == 3` 等）。返す
/// [`RenderedMmaKernel`] はソースと展開元 `cfg` を保持し、ホスト側
/// （`gemm_mma.rs::CudaMmaGemm` 相当の将来の非既定構成起動 API）は
/// `.compile(|src| ...)` で `nvrtc::compile_ptx` 等を実行して
/// [`CompiledMmaKernel`] を得て、以降の起動は
/// `CompiledMmaKernel::launch_f16(stream, a_dev, b_dev, c_dev, m, n, k)`
/// 経由でのみ行うこと（`RenderedMmaKernel`／`CompiledMmaKernel`
/// ドキュメンテーションコメント参照。生ソース・コンパイル済み
/// `CudaFunction` のいずれも検査を経ない形で外部へ返す公開メソッドは
/// 存在しない）。カーネルソースは型付き数値・enum のみから組み立てられ、
/// 外部入力文字列をソースへ連結しない契約を維持する（`nvrtc.rs` A03
/// 節参照）。
///
/// `mod kernels_mma` が非公開モジュールのため、既定構成のみを使う現状の
/// `gemm_mma.rs`（[`mma_f16_source`] 経由）からは呼ばれず、rustc の
/// dead-code 解析が誤検知する。非既定 config を渡す呼び出し元は後続
/// #504（descriptor・コンパイルキャッシュキー）・#519（次元別静的化
/// 選択ポリシー）・#521（段数逆算）で追加される想定のため
/// `#[allow(dead_code)]` を付す。
#[allow(dead_code)]
pub fn render_mma_f16(cfg: &MmaKernelConfig) -> Result<RenderedMmaKernel, CudaError> {
    validate_mma_kernel_config(cfg)?;
    Ok(RenderedMmaKernel {
        source: render_mma_f16_unchecked(cfg),
        cfg: *cfg,
    })
}

/// 既定 [`MmaKernelConfig`]（現行 Rust 定数と同一値）で展開したカーネル
/// ソースを 1 回だけ生成しキャッシュする（`gemm_mma.rs` からの毎呼び出し
/// での再フォーマットを避ける）。既定 config はコンパイル時 const
/// アサーション（本ファイル冒頭）で不変条件を保証済みのため、ここでは
/// `validate_mma_kernel_config` を経ない `_unchecked` 経路を使い、本番
/// 経路に `unwrap()`/`expect()` を置かない（`.claude/rules/coding-rust.md`）。
static MMA_F16_SOURCE: LazyLock<String> =
    LazyLock::new(|| render_mma_f16_unchecked(&MmaKernelConfig::default()));

/// f16 `mma.sync`/`ldmatrix`/`cp.async` GEMM（f16 入出力・f32 アキュムレート）の
/// 既定構成カーネルソース。`gemm_mma.rs::CudaMmaGemm::new` はこの文字列を
/// `nvrtc::compile_ptx` に渡して `CudaFunction` を得る。カーネルソースは
/// コンパイル時定数のまま埋め込み、ビルド時に nvcc／CUDA ヘッダを要求
/// しない契約（`.claude/rules/deps-policy.md`）を維持する（`kernels_wmma.rs`
/// と同じ方針）。
pub fn mma_f16_source() -> &'static str {
    &MMA_F16_SOURCE
}

/// [`render_mma_f16_unchecked`]／[`mma_f16_source`] が結合するカーネル本体
/// テンプレート。`m`/`n`/`k`（カーネル引数）への直接参照は `DIM_M`/`DIM_N`/
/// `DIM_K` マクロ経由に置き換えてある（実装計画 4.1 節「shape の焼き込み
/// 機構」）。カーネルシグネチャ自体は `int m, int n, int k` のまま変更
/// しない（起動側 ABI 不変。`DimSpec::Static` 選択時は当該引数が未使用に
/// なるだけで安全）。
const MMA_F16_BODY: &str = r#"
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
    // コメント「整列制約」）。A_PAD/B_PAD が 8 の倍数のため各行の先頭は
    // 常に 16 バイト整列する（#498。パディングされた行幅
    // `BK`->`A_PAD`・`BN`->`B_PAD` を使うことでバンクコンフリクトを
    // 低減する。本ファイル冒頭コメント「バンクコンフリクト対策」参照。
    // 索引式（LOAD_A/B_STAGE・ldmatrix アドレス計算）は配列次元経由で
    // このパディングを自動的に反映するため、パディング領域そのものへの
    // 明示的な書き込み・読み出しは発生しない）。
    __shared__ __align__(16) __half as_tile[STAGES][BM][A_PAD];
    __shared__ __align__(16) __half bs_tile[STAGES][BK][B_PAD];

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

    int num_k_tiles = (DIM_K > 0) ? (DIM_K - 1) / BK + 1 : 0;

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
    // #496: 1 K タイル分の cp.async 発行を warp 内 kstep ループへ分散する
    // ための添字空間分割。`K_GROUPS`（= `BK / MMA_K`）は下記 kstep ループの
    // 反復回数（`for (int kstep = 0; kstep < BK / MMA_K; ...)`）と必ず
    // 一致する（両者とも同一の `#define` 式由来のため取り零しなく整合する。
    // `MMA_K_STEPS_PER_STAGE >= 2` の Rust 側コンパイル時 assert が
    // `K_GROUPS >= 2` を実質的に保証する。本ファイル `MMA_K_STEPS_PER_STAGE`
    // 定数直下のドキュメンテーションコメント参照）。`A_GROUP_CHUNKS`/
    // `B_GROUP_CHUNKS` は ceil 分割（`(総数 + K_GROUPS - 1) / K_GROUPS`）
    // のため、`A_CHUNKS`/`B_CHUNKS` が `K_GROUPS` で割り切れない構成
    // （将来の `BM`/`BN`/`BK` 変更）でも全チャンクが必ずいずれかの
    // グループに含まれる（最終グループの範囲が `LOAD_*_STAGE_GROUP` 内の
    // `idx < A_CHUNKS`/`idx < B_CHUNKS` 判定でクランプされるため取り零し・
    // 二重発行のいずれも生じない）。
    #define K_GROUPS (BK / MMA_K)
    #define A_CHUNKS ((BM * BK) / 8)
    #define B_CHUNKS ((BK * BN) / 8)
    #define A_GROUP_CHUNKS ((A_CHUNKS + K_GROUPS - 1) / K_GROUPS)
    #define B_GROUP_CHUNKS ((B_CHUNKS + K_GROUPS - 1) / K_GROUPS)

    // #496: グループ `g`（0 <= g < K_GROUPS）が担当するチャンクレンジ
    // `[g * A_GROUP_CHUNKS, min((g+1) * A_GROUP_CHUNKS, A_CHUNKS))` のみを
    // 発行する（CUTLASS `copy_tiles_and_advance` の per-group 発行と同旨）。
    // 境界クランプ（`gr_c`/`gc_c`・16 バイト整列切り下げ）・`valid`（範囲外
    // チャンクの src-size 0 ゼロ充填）の REQ-8 境界検査ロジックは
    // `LOAD_A_STAGE`/`LOAD_B_STAGE`（旧・一括発行版）と一切変更していない
    // （本ファイル冒頭コメント「境界検査」・下記マクロ本体参照）。
    #define LOAD_A_STAGE_GROUP(stage, k0, g) \
        for (int idx = (g) * A_GROUP_CHUNKS + tid; \
             idx < A_CHUNKS && idx < ((g) + 1) * A_GROUP_CHUNKS; \
             idx += blockDim.x) { \
            int row = idx / (BK / 8); \
            int col0 = (idx % (BK / 8)) * 8; \
            int gr = block_row0 + row; \
            int gc = (k0) + col0; \
            int gr_c = gr < DIM_M ? gr : (DIM_M > 0 ? DIM_M - 1 : 0); \
            int gc_c = gc < DIM_K ? gc : (DIM_K > 0 ? ((DIM_K - 1) / 8) * 8 : 0); \
            int valid = (gr < DIM_M && gc < DIM_K) ? 16 : 0; \
            mma_cp_async16(&as_tile[stage][row][col0], &a[(size_t)gr_c * DIM_K + gc_c], valid); \
        }

    #define LOAD_B_STAGE_GROUP(stage, k0, g) \
        for (int idx = (g) * B_GROUP_CHUNKS + tid; \
             idx < B_CHUNKS && idx < ((g) + 1) * B_GROUP_CHUNKS; \
             idx += blockDim.x) { \
            int row = idx / (BN / 8); \
            int col0 = (idx % (BN / 8)) * 8; \
            int gr = (k0) + row; \
            int gc = block_col0 + col0; \
            int gr_c = gr < DIM_K ? gr : (DIM_K > 0 ? DIM_K - 1 : 0); \
            int gc_c = gc < DIM_N ? gc : (DIM_N > 0 ? ((DIM_N - 1) / 8) * 8 : 0); \
            int valid = (gr < DIM_K && gc < DIM_N) ? 16 : 0; \
            mma_cp_async16(&bs_tile[stage][row][col0], &b[(size_t)gr_c * DIM_N + gc_c], valid); \
        }

    // #496: プロローグ（下記）は K タイル 1 段分をまとめてロードする必要が
    // あるため、`LOAD_A_STAGE_GROUP`/`LOAD_B_STAGE_GROUP` を全グループに
    // ついて呼ぶ薄いラッパーとして再定義する（発行本体は上記 2 マクロの
    // 単一サイトに集約されたまま。`K_GROUPS` は 2 相当の小さいコンパイル
    // 時定数のためコンパイラが自動的に展開する）。
    #define LOAD_A_STAGE(stage, k0) \
        for (int g_ = 0; g_ < K_GROUPS; ++g_) { \
            LOAD_A_STAGE_GROUP(stage, k0, g_); \
        }

    #define LOAD_B_STAGE(stage, k0) \
        for (int g_ = 0; g_ < K_GROUPS; ++g_) { \
            LOAD_B_STAGE_GROUP(stage, k0, g_); \
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

        // #495: A/B フラグメントを 2 面バッファ化する（`a_frag[2][...]`/
        // `b_frag[2][...]`）。kstep ループの外側（K タイル t ごと）で
        // 宣言し、下記 warp プロローグ＋ kstep ループ内の先読みが両バッファを
        // 交互に埋める。CUTLASS `mma_multistage.h` の `PipeState`
        // （`warp_loaded_frag_A_[2]` 等）と同型（本ファイル冒頭コメント
        // 「ldmatrix 先読みダブルバッファ」参照）。
        unsigned a_frag[2][WARP_TILES_M][4];
        unsigned b_frag[2][WARP_TILES_N][2];

        // #495: A/B フラグメント 1 個（`mi`/`nj` で指定）を `buf` 面へ
        // ロードするマクロ（`stage` は cp.async ステージ・`kstep` は K
        // タイル内の mma ステップ）。#493 時点のループ本体の 1 反復分を
        // マクロ化したもので、`ldmatrix` の発行箇所自体は A/B 各 1 箇所の
        // まま（`buf`/`mi`/`nj` 違いで呼び分けるだけ。既存テスト
        // `mma_f16_source_issues_mma_sync_from_single_loop_site` と同じ
        // 「ループ化・非コピペ」方針を先読みロードにも適用する）。
        // ldmatrix.x4 1 発行が 16x16（TL/BL/TR/BR の 4 x 8x8）をまとめて
        // ロードする点・象限順序（TL/BL/TR/BR。PR #255 レビュー指摘）は
        // #493 時点と不変。
        //
        // マクロを「1 フラグメント単位」に留め、`WARP_TILES_M`/
        // `WARP_TILES_N` 回のループと `#pragma unroll` を呼び出し側に置く
        // 設計にしている理由（マクロ内で `for` + `_Pragma("unroll")` を
        // 完結させない理由）: 本ファイルは NVRTC 構文検証不能環境で書いて
        // いる（本ファイル冒頭コメント「検証状態」参照）。マクロ内 `for`
        // ループ＋バックスラッシュ継続＋インライン asm の組み合わせ自体は
        // 既存 `LOAD_A_STAGE`/`LOAD_B_STAGE` に前例があるが、`_Pragma`
        // 演算子は本ファイルに前例がなく、NVRTC 上での挙動を実機なしで
        // 確認できない。一方、呼び出し側の `#pragma unroll` はプリプロ
        // セッサ済みの実際の文出現位置に置かれるため、既存の
        // mi/nj 二重ループ（mma.sync 発行側）と同じ形（前例あり・確実）。
        // このマクロは `unroll` を要求する: `buf`（`cur`/`nxt`）・
        // `mi`/`nj` はインラインアセンブリの出力オペランド
        // （`a_frag[buf][mi][...]` 等）の添字であり、レジスタ割り当ては
        // コンパイル時に確定した添字を要求する（実行時添字だとレジスタを
        // 選べず local memory へ溢れ、ロード先読み最適化が逆に SMEM
        // 律速を招く）。呼び出し側で `#pragma unroll` を伴わずにこの
        // マクロを呼ばないこと。
        #define LDSM_A_FRAG(buf, stage, kstep, mi) \
            do { \
                int a_col_ = (kstep) * MMA_K; \
                int a_row = warp_row * (MMA_M * WARP_TILES_M) + (mi) * MMA_M; \
                int a_quad_row = (lane / 8) % 2; \
                int a_quad_col = (lane / 8) / 2; \
                int a_row_in_tile = lane % 8; \
                __half* a_addr = &as_tile[stage] \
                                          [a_row + a_quad_row * 8 + a_row_in_tile] \
                                          [a_col_ + a_quad_col * 8]; \
                unsigned a_smem = (unsigned)__cvta_generic_to_shared(a_addr); \
                asm volatile( \
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
                    : "=r"(a_frag[buf][mi][0]), "=r"(a_frag[buf][mi][1]), \
                      "=r"(a_frag[buf][mi][2]), "=r"(a_frag[buf][mi][3]) \
                    : "r"(a_smem) \
                ); \
            } while (0)

        // #493: B フラグメント（16x8。k x n の row-major 格納から `.trans`
        // ロードで mma の `.col` 要求配置へ変換。本ファイル冒頭コメント
        // 「命令選定」）。`b_quad`（`lane / 8`）は 0..1 のみ使用（x2）。
        // B 2 個（`nj = 0..1`）を x4.trans 1 発行へ融合する余地は残るが、
        // #493 受け入れ基準の「A 2 個・B 2 個を 4 通りで再利用」という
        // 最小差分に合わせ、ここでは x2.trans を nj ごとに個別発行する
        // （融合は #495 以降・#496（cp.async issue interleaving）以降の
        // 最適化余地として引き継ぐ）。
        #define LDSM_B_FRAG(buf, stage, kstep, nj) \
            do { \
                int b_row_ = (kstep) * MMA_K; \
                int b_col = warp_col * (MMA_N * WARP_TILES_N) + (nj) * MMA_N; \
                int b_row_in_tile = lane % 8; \
                int b_quad = lane / 8; \
                __half* b_addr = &bs_tile[stage] \
                                          [b_row_ + (b_quad % 2) * 8 + b_row_in_tile] \
                                          [b_col]; \
                unsigned b_smem = (unsigned)__cvta_generic_to_shared(b_addr); \
                asm volatile( \
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" \
                    : "=r"(b_frag[buf][nj][0]), "=r"(b_frag[buf][nj][1]) \
                    : "r"(b_smem) \
                ); \
            } while (0)

        // #495 warp プロローグ: kstep=0 のフラグメントをバッファ 0 へ
        // ロードしてから kstep ループへ入る（CUTLASS `mac_loop_iter` の
        // ウォームアップ相当）。
#pragma unroll
        for (int mi = 0; mi < WARP_TILES_M; ++mi) {
            LDSM_A_FRAG(0, compute_stage, 0, mi);
        }
#pragma unroll
        for (int nj = 0; nj < WARP_TILES_N; ++nj) {
            LDSM_B_FRAG(0, compute_stage, 0, nj);
        }

        // #495: kstep ループ自体を `#pragma unroll` する。`BK / MMA_K` は
        // コンパイル時定数式（`#define` 由来）のためトリップ回数は既知
        // であり、これにより `cur`/`nxt`（`kstep % 2`/`(kstep+1) % 2`）が
        // コンパイル時定数へ畳み込まれる。上記 `LDSM_A_FRAG`/`LDSM_B_FRAG`
        // マクロ直前のコメントの通り、これはフラグメント配列のインライン
        // asm 出力オペランド添字（`a_frag[cur][mi][...]` 等）がコンパイル
        // 時定数であることを要求するための必須の pragma であり、
        // 省略すると local memory 溢れによる性能後退を招く（cosmetic な
        // pragma ではない）。
#pragma unroll
        for (int kstep = 0; kstep < BK / MMA_K; ++kstep) {
            int cur = kstep % 2;
            int nxt = (kstep + 1) % 2;

            // #495: 次段（kstep+1）のフラグメントを、現在段（kstep）の
            // mma.sync 発行前に先読みしてバッファ `nxt` へロードする。
            // これにより `ldmatrix` の SMEM→レジスタロードレイテンシが
            // 直後の mma.sync 系列とオーバーラップする（ソフトウェア
            // パイプライン化。本ファイル冒頭コメント「ldmatrix 先読み
            // ダブルバッファ」参照）。
            //
            // タイル境界を跨ぐ先読み（CUTLASS の `(warp_mma_k+1) % K`
            // wrap-around で次タイルの kstep=0 まで読む方式）は**採用
            // しない**（意図的な縮小。#495 実装計画 §3.3）。理由:
            // 本カーネルのループ内 `cp.async.wait_group (STAGES-2)` は
            // イテレーション t の時点でタイル t のグループ完了までしか
            // 保証しない（本ファイル `MMA_STAGES` 定数直下コメント
            // 「正しさ」参照）。タイル t+1 の SMEM 完了保証前に読み出す
            // クロスタイル先読みは wait/sync 配置の大規模再構成を要し、
            // NVRTC 構文検証不能な本環境ではリスクが高いため、タイル内
            // 先読みに限定した（残余の最適化余地として
            // `docs/perf/cuda-gemm-mma-ldmatrix-double-buffer.md` に記録）。
            if (kstep + 1 < BK / MMA_K) {
#pragma unroll
                for (int mi = 0; mi < WARP_TILES_M; ++mi) {
                    LDSM_A_FRAG(nxt, compute_stage, kstep + 1, mi);
                }
#pragma unroll
                for (int nj = 0; nj < WARP_TILES_N; ++nj) {
                    LDSM_B_FRAG(nxt, compute_stage, kstep + 1, nj);
                }
            }

            // #496: cp.async issue interleaving。1 K タイル分の次段ロード
            // 発行を一括せず、kstep ごとにグループ 1 個分（`g = kstep`。
            // `K_GROUPS == BK / MMA_K` が kstep ループの反復回数と一致する
            // ため全グループが過不足なく発行される）だけ発行し、直後の
            // mma.sync 系列とオーバーラップさせる（CUTLASS
            // `mma_multistage.h` が `warp_mma` 呼び出しと
            // `copy_tiles_and_advance` を交互に置く配置と同旨。本ファイル
            // 冒頭コメント「cp.async issue interleaving」参照）。
            //
            // 同期の正しさ: 発行先 `load_stage` の SMEM ステージは、
            // 前イテレーション（K タイル t-1）末尾の `__syncthreads()`
            // 通過後は本イテレーションの ldmatrix（`compute_stage` を読む。
            // `load_stage != compute_stage` は STAGES>=2 で常に成立）から
            // 一切読まれない。したがって発行位置を旧来のループ末尾一括
            // 発行から kstep ループ内へ前倒ししても、書き込みと読み出しの
            // ハザードは生じない（追加の `__syncthreads()` は不要）。
            // `next_tile < num_k_tiles` ガードは分割前と同じ意味（K が
            // 浅い形状での範囲外ステージロード抑止。#492 不変条件）を
            // 各グループ発行に対して個別に適用する。`cp.async.commit_group`
            // 自体はループ末尾・無条件のまま動かさない（本関数末尾の
            // コメント参照。「1 イテレーション = 必ず 1 commit」不変条件・
            // `wait_group (STAGES-2)` の正しさ論証を無傷で維持するため）。
            if (next_tile < num_k_tiles) {
                LOAD_A_STAGE_GROUP(load_stage, next_tile * BK, kstep);
                LOAD_B_STAGE_GROUP(load_stage, next_tile * BK, kstep);
            }

            // #493: mi x nj の 4 通りで mma.sync を発行し d[mi][nj] へ
            // アキュムレートする。同一出力タイルあたりの ldmatrix 発行
            // 回数は、単一タイル構成（1 warp = 16x8 タイル 1 個）の
            // 4 発行（A x4 + B x2 を warp ごとに個別発行）から本構成
            // （2x2 レジスタブロッキング）の 2 発行（A x4 + B x2 を
            // 4 タイル分で共有）へ半減する（本ファイル冒頭コメント
            // 「タイル構成」参照。#494 のブロックタイル拡大〈`32x64`→
            // `64x128`〉は warp あたりの担当範囲・レジスタブロッキング
            // 係数〈`WARP_TILES_M`/`WARP_TILES_N`〉自体には影響せず、
            // ブロック全体の warp 数〈`MMA_WARPS_M`/`MMA_WARPS_N`〉のみを
            // 変える）。#495 でフラグメントソースを `a_frag[cur]`/
            // `b_frag[cur]` へ切り替えた以外、発行箇所・オペランド順は
            // #493/#494 時点と不変（先読みタイミングの変更は mma.sync の
            // 発行順序・オペランド値を変えないため、出力は bit 一致。
            // 本ファイル冒頭コメント「ldmatrix 先読みダブルバッファ」参照）。
#pragma unroll
            for (int mi = 0; mi < WARP_TILES_M; ++mi) {
#pragma unroll
                for (int nj = 0; nj < WARP_TILES_N; ++nj) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
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
        #undef LDSM_B_FRAG

        // #492/#496: commit をループ末尾で無条件発行する（プロローグと
        // 同じ「1 イテレーション = 必ず 1 commit」不変条件）。#496 で
        // 次段タイルのロード発行自体は kstep ループ内へ分散したため
        // （上記「cp.async issue interleaving」コメント参照）、ここには
        // commit のみが残る。分割後も全グループの発行が同一イテレーション
        // 内で完了してから commit されることに変わりはないため、
        // `wait_group (STAGES-2)` の正しさ論証（本ファイル `MMA_STAGES`
        // 定数直下のドキュメンテーションコメント「正しさ」参照）は無傷で
        // 成立する。
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
    #undef LOAD_A_STAGE_GROUP
    #undef LOAD_B_STAGE_GROUP
    #undef A_GROUP_CHUNKS
    #undef B_GROUP_CHUNKS
    #undef A_CHUNKS
    #undef B_CHUNKS
    #undef K_GROUPS

    // REQ-8: エピローグの guarded store。mma.m16n8k16 の C/D フラグメント
    // レーン対応（groupID/threadID_in_group。本ファイル冒頭コメント
    // 「命令選定」）: d0/d1 は行 groupID、d2/d3 は行 groupID+8。#493 で
    // mi/nj の 2 重ループへ拡張し、warp が担当する WARP_TILES_M x
    // WARP_TILES_N 個の出力タイルを順に書き戻す。
    //
    // #805: 隣接列ペア（c0/c1 = c0+1）を `__half2`（4 バイト）のベクトル
    // store 2 回（行 r0 分・行 r1 分）へまとめ、旧来の 4 スカラー store を
    // 半減させる。正当性の根拠:
    //   - 整列: `validate_mma_alignment`（ホスト側 `gemm_mma.rs`）が起動前に
    //     `n % 8 == 0`（`MMA_N = 8`）を fail-closed で検査するため DIM_N は
    //     常に偶数。`c0 = col0_warp + nj * MMA_N + tid_in_group * 2` も
    //     （`col0_warp` が 8 の倍数のため）常に偶数。ゆえに要素添字
    //     `r * DIM_N + c0` は常に偶数 → バイトオフセットは 4 の倍数となり、
    //     C バッファ（cudarc デバイス確保・256B 整列）基点からの
    //     `__half2`（4B 整列要求）store は常に整列する。
    //   - ペアの全有効/全無効: DIM_N 偶数・c0 偶数のため `c0 < DIM_N` ⇔
    //     `c1 < DIM_N`。列ペアは「両方有効」か「両方無効」の二値になり、
    //     ペア単位の境界判定（`c1 < DIM_N`）が REQ-8 の境界検査を弱めずに
    //     成立する（cp.async 側「8 要素チャンクが境界を跨がない」設計と
    //     同じ論法。本ファイル冒頭「整列制約」節）。
    //   - 丸めの同一性: `__floats2half2_rn` は `__float2half` と同じ
    //     round-to-nearest-even のため、書き込まれる値は変わらない
    //     （bit 一致。`tests/parity_nonregression.rs` の fixture・
    //     tolerance は無変更で成立）。
    //   - defensive fallback: 上記不変条件下では `c0 < DIM_N` かつ
    //     `c1 >= DIM_N` は成立し得ず到達不能だが、将来 `DIM_N` 偶数制約が
    //     緩んだ場合に列 c0 側の書き落としが起きないよう、ペア判定が
    //     不成立のときは c0 単独のスカラー store へフォールバックする
    //     （fail-closed。REQ-8「境界検査を無効化する最適化はシェーダ側で
    //     手動境界チェックを維持したうえで行う」に準拠）。
#pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; ++mi) {
#pragma unroll
        for (int nj = 0; nj < WARP_TILES_N; ++nj) {
            int r0 = row0_warp + mi * MMA_M + group_id;
            int r1 = row0_warp + mi * MMA_M + group_id + 8;
            int c0 = col0_warp + nj * MMA_N + tid_in_group * 2;
            int c1 = c0 + 1;

            if (r0 < DIM_M && c1 < DIM_N) {
                *reinterpret_cast<__half2*>(&c[(size_t)r0 * DIM_N + c0]) =
                    __floats2half2_rn(d[mi][nj][0], d[mi][nj][1]);
            } else if (r0 < DIM_M && c0 < DIM_N) {
                c[(size_t)r0 * DIM_N + c0] = __float2half(d[mi][nj][0]);
            }

            if (r1 < DIM_M && c1 < DIM_N) {
                *reinterpret_cast<__half2*>(&c[(size_t)r1 * DIM_N + c0]) =
                    __floats2half2_rn(d[mi][nj][2], d[mi][nj][3]);
            } else if (r1 < DIM_M && c0 < DIM_N) {
                c[(size_t)r1 * DIM_N + c0] = __float2half(d[mi][nj][2]);
            }
        }
    }
}
"#;

/// [`mma_f16_source`] のアンカー 2 行（`block_row0`/`block_col0` のブロック
/// 原点計算）を、`swizzle.rs::swizzled_block_idx` と同一の整数式（グループ幅
/// `group_width` の M 方向グルーピング remap）へ差し替えた変種ソースを
/// 生成する（イシュー #499 受け入れ基準 1〜2 項。イシュー #775 で
/// サイズ条件付き適用ロジックとして実装し、イシュー #782 で GB10 実機
/// ゲート通過（2026-08-21）を根拠に本番既定コンストラクタへ結線した。
/// 下記「呼び出し元」節参照）。
///
/// **`mma_f16_source()`（既定 config の render 結果。イシュー #516 で
/// `MMA_F16` 定数からテンプレート展開へ移行済み）自体は変更しない**
/// （`replacen` で新規 `String` を都度構築するのみ）。
///
/// # 呼び出し元
///
/// `gemm_mma.rs::CudaMmaGemm::new`（本番既定コンストラクタ・feature 非
/// 依存。`device.multiprocessor_count()` の実測に成功した場合のみ
/// swizzle 変種を追加コンパイルする。`launch_f16` が呼び出し形状ごとに
/// `swizzle::should_apply_swizzle` で base／swizzle 変種いずれを起動する
/// か判定する）と `new_with_swizzle`（`internal-diagnostics` feature
/// 限定・診断用・明示幅指定・強制適用）の両方から呼ばれる。`ops.rs`／
/// `gemm_auto.rs` は mma_f16 経路自体を参照しないため無変更のまま
/// （`lib.rs` 冒頭コメント「#775」「#782」節参照）。
///
/// # remap の整数式（`swizzle.rs::swizzled_block_idx` と単一の設計を共有）
///
/// `linear_idx = blockIdx.y * gridDim.x + blockIdx.x`（CUDA が
/// `blockIdx.x` を先に増分する既定の 2 次元グリッド線形化。`gridDim.x` =
/// `num_n_blocks`・`gridDim.y` = `num_m_blocks`）を、`group_width` 個の M
/// ブロックごとにグルーピングした順序へ remap する。式自体は
/// `swizzle.rs::swizzled_block_idx` のホスト側参照実装と同一だが、両者は
/// 独立した文字列（Rust 式 vs CUDA C++ 文字列）であり単一の共有ソースを
/// 持たないため、不一致は `mma_f16_source_with_swizzle_matches_group_width_define`
/// （本ファイル）と `gemm_mma.rs` の実機 bit 一致テストの二段で検出する
/// （`swizzle.rs` 冒頭コメント参照）。
///
/// # エラー契約
///
/// `group_width < 2` は `CudaError::InvalidShape` で拒否する
/// （`group_width == 1` は各グループが M ブロック 1 個のみを持つ退化
/// ケースで、恒等写像に等しく L2 再利用効果を持たないため。実装計画
/// 3 節「(c)」。安全側の実装単純化として `swizzle.rs::
/// select_swizzle_group_width` の候補 `{8, 16}` 以外の値も受理するが
/// `1` 未満・`1` そのものは拒否する）。
///
/// イシュー #782: `gemm_mma.rs::CudaMmaGemm::new`（本番既定コンストラクタ・
/// feature 非依存）が上記「呼び出し元」節のとおりこの関数を呼ぶため、通常
/// ビルド（feature 指定なし）でも到達する。
pub fn mma_f16_source_with_swizzle(group_width: u32) -> Result<String, crate::error::CudaError> {
    if group_width < 2 {
        return Err(crate::error::CudaError::InvalidShape {
            detail: format!(
                "mma_f16_source_with_swizzle requires group_width >= 2 (got {group_width}); \
                 group_width == 1 degenerates to the identity block mapping \
                 (swizzle.rs::swizzled_block_idx_group_width_one_is_identity_mapping) \
                 and offers no L2 reuse benefit"
            ),
        });
    }

    const ANCHOR: &str =
        "    int block_row0 = blockIdx.y * BM;\n    int block_col0 = blockIdx.x * BN;\n";
    // イシュー #516 でカーネルソースが `MMA_F16` 定数からテンプレート展開
    // （`render_mma_f16_unchecked`／`MMA_F16_SOURCE`）へ移行したため、
    // 定数を直接参照せず `mma_f16_source()`（既定 config の render 結果。
    // 他の回帰テストと同じ参照経路）を対象にする。
    let source = mma_f16_source();
    let occurrences = source.matches(ANCHOR).count();
    // `unwrap()`/`expect()`・panic 系マクロを本番経路で使わない方針
    // （coding-rust.md「エラーは型付きエラーとし、本番経路で unwrap()
    // / expect() を使わない」）に合わせ、`assert_eq!` ではなく型付き
    // エラーで返す。`mma_f16_source()` 側の不変条件は
    // `mma_f16_source_with_swizzle_does_not_mutate_mma_f16_source` が
    // 別途 CI 上で回帰検査するため、ここで通常到達しない前提だが、
    // `new_with_swizzle` から到達しうる公開関数として panic を避ける。
    if occurrences != 1 {
        return Err(crate::error::CudaError::InvalidShape {
            detail: format!(
                "MMA_F16 中のブロック原点アンカー（block_row0/block_col0）の \
                 出現数が 1 ではありません（{occurrences} 件検出。 \
                 mma_f16_source_with_swizzle の前提が崩れています）"
            ),
        });
    }

    let remap = format!(
        "    // イシュー #499: L2 再利用のためのタイル→SM 割り当てスウィズル\n\
         \x20   // remap（swizzle.rs::swizzled_block_idx と同一式。本ファイル\n\
         \x20   // mma_f16_source_with_swizzle ドキュメンテーションコメント参照）。\n\
         \x20   // PR #667 codex-review P0 是正: 線形 index・ブロック数・積は\n\
         \x20   // `long long`（64 bit）で計算する。`gridDim.y` は\n\
         \x20   // `gemm_mma.rs::MAX_GRID_DIM_Y`（65,535）で上限検証済みだが\n\
         \x20   // `gridDim.x` は上限検証していないため（同ファイル冒頭コメント\n\
         \x20   // 「x 成分の上限は 2^31-1 と大きく実用的に問題にならないため\n\
         \x20   // x は検証しない」）、`blockIdx.y * gridDim.x` や\n\
         \x20   // `SWIZZLE_GROUP * num_n_blocks` は `int`（32 bit 符号付き）\n\
         \x20   // のままでは容易にオーバーフローしうる（REQ-8 「境界検査の\n\
         \x20   // 省略禁止」）。最終座標は `gridDim` 内であることを明示的に\n\
         \x20   // 検査してから `int` へ縮小する（`m_block < num_m_blocks <=\n\
         \x20   // 65,535`・`n_block < num_n_blocks <= 2^31-1` を上記の\n\
         \x20   // 境界検査が保証するため、`m_block * BM`/`n_block * BN` は\n\
         \x20   // 元の `m`/`n`（呼び出し元 `gemm_mma.rs::mma_launch_config`\n\
         \x20   // が `int` として渡す形状）を超えず、64→32 bit への縮小は\n\
         \x20   // 安全。オーバーフロー元は 64 bit 側で解消済みであり、この\n\
         \x20   // 縮小自体は新たな符号なし/符号付きオーバーフロー経路を\n\
         \x20   // 導入しない）。\n\
         \x20   #define SWIZZLE_GROUP {group_width}\n\
         \x20   long long num_m_blocks = gridDim.y;\n\
         \x20   long long num_n_blocks = gridDim.x;\n\
         \x20   long long linear_idx = (long long)blockIdx.y * gridDim.x + blockIdx.x;\n\
         \x20   long long full_groups = num_m_blocks / SWIZZLE_GROUP;\n\
         \x20   long long remainder = num_m_blocks % SWIZZLE_GROUP;\n\
         \x20   long long full_group_blocks = (long long)SWIZZLE_GROUP * num_n_blocks;\n\
         \x20   long long full_groups_total_blocks = full_groups * full_group_blocks;\n\
         \x20   long long m_block, n_block;\n\
         \x20   if (linear_idx < full_groups_total_blocks) {{\n\
         \x20       long long group_idx = linear_idx / full_group_blocks;\n\
         \x20       long long idx_in_group = linear_idx % full_group_blocks;\n\
         \x20       m_block = group_idx * SWIZZLE_GROUP + (idx_in_group % SWIZZLE_GROUP);\n\
         \x20       n_block = idx_in_group / SWIZZLE_GROUP;\n\
         \x20   }} else {{\n\
         \x20       long long idx_in_group = linear_idx - full_groups_total_blocks;\n\
         \x20       m_block = full_groups * SWIZZLE_GROUP + (idx_in_group % remainder);\n\
         \x20       n_block = idx_in_group / remainder;\n\
         \x20   }}\n\
         \x20   if (m_block < 0 || m_block >= num_m_blocks || n_block < 0 ||\n\
         \x20       n_block >= num_n_blocks) {{\n\
         \x20       return;\n\
         \x20   }}\n\
         \x20   int block_row0 = (int)(m_block * BM);\n\
         \x20   int block_col0 = (int)(n_block * BN);\n"
    );

    Ok(source.replacen(ANCHOR, &remap, 1))
}

/// warp タイル拡大候補（イシュー #803）のレジスタ収支を実機 `ptxas -v` で
/// 比較するための診断用ソース生成器。`mma_f16_source_with_swizzle` と同型の
/// 方式（`mma_f16_source()`〈既定 config の render 結果〉に対するアンカー
/// 完全一致置換・アンカー不在／複数出現は fail-closed エラー）で、本番
/// カーネル定数（`MMA_WARP_TILES_M`/`_N`・`MMA_WARPS_N` 等）は一切変更せず
/// カーネルソース文字列だけを差し替える。**本番結線（`MMA_WARP_TILES_M`/
/// `_N` 定数の変更・`gemm_mma.rs` からの呼び出し）は本イシューのスコープ外
/// であり後続 #804 が担う**（本ファイル `MMA_WARP_TILES_M`/`_N` 定数直下の
/// ドキュメンテーションコメント参照）。
///
/// # 引数
///
/// - `warp_tiles_m`/`warp_tiles_n`: 候補の warp あたりレジスタブロッキング
///   係数（`MMA_WARP_TILES_M`/`_N` 相当）。
/// - `launch_bounds`: `Some(v)` の場合、カーネルシグネチャへ
///   `__launch_bounds__(v)` を付与する（CUTLASS `device_kernel.h` 方式。
///   `docs/cuda-tensor-core-design.md` §14 参照）。`v` は本関数が導出する
///   ブロックスレッド数と完全一致する必要がある（不一致は誤った
///   `.maxntid` での計測を招くため拒否する）。`None` は付与しない。
///
/// # エラー契約（fail-closed。REQ-8 の境界検査省略禁止と同方針）
///
/// - `warp_tiles_m == 0 || warp_tiles_n == 0`
/// - `MMA_BM`/`MMA_BN` が `MMA_M * warp_tiles_m`/`MMA_N * warp_tiles_n` の
///   倍数でない（warp グリッドを割り切らない構成は `MMA_WARPS_M`/`_N` の
///   整数除算が余りを切り捨て、誤ったブロックタイル被覆を静かに生成する。
///   `MMA_WARPS_M`/`_N` 定数直下のコンパイル時契約検査と同じ理由）
/// - 導出ブロックスレッド数が `1024`（CUDA の per-block 上限）を超える
/// - `launch_bounds == Some(v)` かつ `v` が導出ブロックスレッド数と不一致
///
/// 既定値 `(MMA_WARP_TILES_M, MMA_WARP_TILES_N, None)` を渡すと
/// `mma_f16_source()` と完全に同一のバイト列を返す（本ファイル `tests`
/// モジュールの `mma_f16_source_with_warp_tiles_default_matches_mma_f16_source`
/// が回帰検査する）。
///
/// # 呼び出し元
///
/// `examples/mma_ptx_dump.rs`（`internal-diagnostics` feature 限定・
/// `lib.rs::diagnostics` 経由）専用。本番経路（`gemm_mma.rs`）はこの関数を
/// 呼ばない。
///
/// `mma_f16_source_with_swizzle` と異なり本番経路から無条件に呼ばれる
/// 消費者を持たない（本番結線は #804 のスコープ）ため、`internal-
/// diagnostics` feature 未指定の既定ビルドでは `lib.rs::diagnostics` の
/// 再公開のみが唯一の参照元になり、rustc の dead-code 解析が未使用と
/// 誤検知する（`kernels_wmma_opt.rs` の同種定数群と同じ理由）。
#[allow(dead_code)]
pub fn mma_f16_source_with_warp_tiles(
    warp_tiles_m: u32,
    warp_tiles_n: u32,
    launch_bounds: Option<u32>,
) -> Result<String, CudaError> {
    if warp_tiles_m == 0 || warp_tiles_n == 0 {
        return Err(CudaError::InvalidKernelConfig {
            detail: format!(
                "mma_f16_source_with_warp_tiles requires warp_tiles_m/n >= 1 \
                 (got warp_tiles_m={warp_tiles_m}, warp_tiles_n={warp_tiles_n})"
            ),
        });
    }
    // `MMA_M`/`MMA_N` は定数だが `warp_tiles_m`/`warp_tiles_n` は任意の
    // `u32`（呼び出し元検証なしの公開 API）のため、乗算は `checked_mul`
    // で行い、オーバーフローする入力は境界検査（0 除算・wrap 混入）より
    // 前に fail-closed で `CudaError::InvalidKernelConfig` として拒否する
    // （本番経路 panic 禁止規約・関数自身の fail-closed 契約。#822 codex-review 指摘）。
    let warp_m = warp_tiles_m
        .checked_mul(MMA_M)
        .ok_or_else(|| CudaError::InvalidKernelConfig {
            detail: format!(
                "mma_f16_source_with_warp_tiles warp_tiles_m={warp_tiles_m} overflows u32 when \
             multiplied by MMA_M={MMA_M}"
            ),
        })?;
    let warp_n = warp_tiles_n
        .checked_mul(MMA_N)
        .ok_or_else(|| CudaError::InvalidKernelConfig {
            detail: format!(
                "mma_f16_source_with_warp_tiles warp_tiles_n={warp_tiles_n} overflows u32 when \
             multiplied by MMA_N={MMA_N}"
            ),
        })?;
    if !MMA_BM.is_multiple_of(warp_m) || !MMA_BN.is_multiple_of(warp_n) {
        return Err(CudaError::InvalidKernelConfig {
            detail: format!(
                "mma_f16_source_with_warp_tiles candidate warp tile {warp_m}x{warp_n} \
                 (warp_tiles_m={warp_tiles_m}, warp_tiles_n={warp_tiles_n}) does not evenly \
                 divide the block tile MMA_BM={MMA_BM}x MMA_BN={MMA_BN}"
            ),
        });
    }
    let warps_m = MMA_BM / warp_m;
    let warps_n = MMA_BN / warp_n;
    // `warps_m`/`warps_n` は上記倍数検査により常に `>= 1`（0 除算・0 warp
    // は生じない）。
    let threads = warps_m * warps_n * 32;
    if threads > 1024 {
        return Err(CudaError::InvalidKernelConfig {
            detail: format!(
                "mma_f16_source_with_warp_tiles candidate warp tile {warp_m}x{warp_n} derives \
                 {threads} threads/block, exceeding CUDA's per-block limit (1024)"
            ),
        });
    }
    if let Some(v) = launch_bounds
        && v != threads
    {
        return Err(CudaError::InvalidKernelConfig {
            detail: format!(
                "mma_f16_source_with_warp_tiles launch_bounds ({v}) must equal the derived \
                 thread count ({threads}) for warp tile {warp_m}x{warp_n}; a mismatched value \
                 would measure ptxas register allocation under a `.maxntid` that does not \
                 match the actual launch configuration"
            ),
        });
    }

    // #define アンカーは `mma_f16_source()`（既定 config の render 結果）に
    // 実際に現れる値（`MMA_WARPS_N`/`MMA_WARP_TILES_M`/`_N` 定数）から
    // 組み立てる（`mma_f16_source_with_swizzle` と同じ、定数変更に追従する
    // 方式）。出現回数 1 を検査してから置換する（fail-closed）。
    // Bugbot 指摘是正（PR #831）: 以前は本関数内にローカルクロージャとして
    // 同じ置換ロジックを重複定義していた。`mma_f16_source_with_block_tile`
    // 向けに切り出した [`replace_source_anchor`] へ一本化する（同一契約が
    // 2 箇所に存在する状態を解消）。
    let source = replace_source_anchor(
        mma_f16_source().to_owned(),
        &format!("#define WARPS_N {MMA_WARPS_N}\n"),
        &format!("#define WARPS_N {warps_n}\n"),
        "mma_f16_source_with_warp_tiles",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define WARP_TILES_M {MMA_WARP_TILES_M}\n"),
        &format!("#define WARP_TILES_M {warp_tiles_m}\n"),
        "mma_f16_source_with_warp_tiles",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define WARP_TILES_N {MMA_WARP_TILES_N}\n"),
        &format!("#define WARP_TILES_N {warp_tiles_n}\n"),
        "mma_f16_source_with_warp_tiles",
    )?;

    let source = if let Some(v) = launch_bounds {
        const SIG_ANCHOR: &str = "extern \"C\" __global__ void gemm_mma_f16(";
        let sig_replacement =
            format!("extern \"C\" __global__ void __launch_bounds__({v}) gemm_mma_f16(");
        replace_source_anchor(
            source,
            SIG_ANCHOR,
            &sig_replacement,
            "mma_f16_source_with_warp_tiles",
        )?
    } else {
        source
    };

    Ok(source)
}

/// [`mma_f16_source_with_warp_tiles`]／[`mma_f16_source_with_block_tile`]
/// 共通のアンカー完全一致置換ヘルパー（イシュー #804 でモジュール関数へ
/// 括り出した。以前は `mma_f16_source_with_warp_tiles` 内のローカル
/// クロージャとして重複定義されていた）。`anchor` の出現回数が 1 でない
/// 場合は定数変更等で置換前提が崩れたとみなし fail-closed で拒否する
/// （PR #822・#804 と同じ契約）。
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
                "{caller}: anchor {anchor:?} occurs {occurrences} times in mma_f16_source() \
                 (expected exactly 1); the define replacement assumption no longer holds"
            ),
        });
    }
    Ok(src.replacen(anchor, replacement, 1))
}

/// [`derive_mma_block_tile_layout`] が返す、候補ブロックタイル・段数・warp
/// タイル構成から導出したカーネル起動パラメータの束（イシュー #840）。
///
/// `mma_f16_source_with_block_tile`（下記）のカーネルソース展開と、
/// `examples/gemm_mma_block_tile_bench.rs`（診断専用 A/B ランナー。
/// `internal-diagnostics` feature 限定）のカーネル起動（`threads`・
/// `smem_bytes`・opt-in 動的 SMEM 要否判定）の両方が本構造体を経由する
/// ことで、ブロックスレッド数・共有メモリバイト数の算出式が 1 箇所
/// （[`derive_mma_block_tile_layout`]）にのみ存在する状態を保つ
/// （実装計画「レイアウト導出ヘルパー」節: 「既存
/// `mma_f16_source_with_block_tile` 内部の SMEM 式を共有化し二重定義を
/// 作らない」）。
#[allow(dead_code)] // 理由は mma_f16_source_with_warp_tiles と同じ（非公開モジュール）
#[derive(Debug, Clone, Copy)]
pub struct MmaBlockTileLayout {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub stages: u32,
    pub warp_tiles_m: u32,
    pub warp_tiles_n: u32,
    /// warp グリッド（`bm`/`warp_m` 行 × `bn`/`warp_n` 列）の行数。
    pub warps_m: u32,
    /// warp グリッドの列数。カーネルソース側 `#define WARPS_N` に対応
    /// （`render_mma_f16_unchecked`・`mma_f16_source_with_warp_tiles` と
    /// 同じ「1 warp = C の `warp_m x warp_n` タイル 1 個」設計）。
    pub warps_n: u32,
    /// 導出ブロックスレッド数（`warps_m * warps_n * 32`）。
    /// `LaunchConfig.block_dim.x`・`__launch_bounds__` 一致検査の両方に
    /// 使う。
    pub threads: u32,
    /// A タイル 1 行あたりのパディング済み要素数（`bk + 8`。#498 バンク
    /// コンフリクト対策）。
    pub a_pad: u32,
    /// B タイル 1 行あたりのパディング済み要素数（`bn + 8`）。
    pub b_pad: u32,
    /// 共有メモリ総使用量（バイト）。`(bm*a_pad + bk*b_pad) * 2B(f16) *
    /// stages`（`validate_mma_kernel_config`・従来の
    /// `mma_f16_source_with_block_tile` 内 `smem_bytes` 算出と同一式）。
    pub smem_bytes: u32,
}

impl MmaBlockTileLayout {
    /// `smem_bytes` が静的 48KiB 上限（[`MMA_STATIC_SMEM_LIMIT_BYTES`]）を
    /// 超え、`extern __shared__`（opt-in 動的 SMEM）変種を要求するか。
    #[allow(dead_code)] // 理由は Self と同じ（非公開モジュール）
    pub fn needs_dynamic_smem(&self) -> bool {
        self.smem_bytes > MMA_STATIC_SMEM_LIMIT_BYTES
    }
}

/// 候補 `bm`/`bn`/`bk`/`stages`/`warp_tiles_m`/`_n` から
/// [`MmaBlockTileLayout`] を導出する（イシュー #840。従来
/// `mma_f16_source_with_block_tile` 内にインライン展開されていた検証・
/// 算出ロジックの切り出し）。
///
/// 検査する不変条件は元の `mma_f16_source_with_block_tile` と同一
/// （零値拒否・段数範囲・`bk`/`MMA_K` 倍数関係・`bm`/`bn` の 8 の倍数・
/// warp タイルの整数除算・スレッド数上限）。`optin_budget_bytes` との
/// 比較・`launch_bounds` 一致検査は行わない（呼び出し元ごとに許容判断が
/// 異なる: `mma_f16_source_with_block_tile` は超過を拒否してソースを
/// 生成しないが、`gemm_mma_block_tile_bench.rs` は除外理由をログへ残し
/// つつスイープを継続する。比較・検査は各呼び出し元へ委ねる）。
pub(crate) fn derive_mma_block_tile_layout(
    bm: u32,
    bn: u32,
    bk: u32,
    stages: u32,
    warp_tiles_m: u32,
    warp_tiles_n: u32,
) -> Result<MmaBlockTileLayout, CudaError> {
    let invalid = |detail: String| CudaError::InvalidKernelConfig { detail };

    if bm == 0 || bn == 0 || bk == 0 || stages == 0 || warp_tiles_m == 0 || warp_tiles_n == 0 {
        return Err(invalid(
            "mma_f16_source_with_block_tile requires bm/bn/bk/stages/warp_tiles_m/n >= 1"
                .to_string(),
        ));
    }
    // 段数一般形の前提（`mma_f16_source_with_block_tile` ドキュメンテー
    // ションコメント「本番結線の validate_mma_kernel_config との違い」節）。
    if stages < 2 {
        return Err(invalid(format!(
            "mma_f16_source_with_block_tile stages ({stages}) must be >= 2 (cp.async \
             software pipeline invariant; see mma_f16_source_uses_fixed_immediate_wait_with_loop_exit_drain)"
        )));
    }
    // codex-review P2 是正（PR #831）: ループ内固定即値
    // `cp.async.wait_group "n"(STAGES - 2)` の `"n"` オペランドは
    // `cp.async.wait_group` の即値オペランド（0〜7 の範囲。PTX ISA
    // `cp.async.wait_group` 命令仕様）を要求する。`STAGES - 2` がこの範囲を
    // 超える（`stages >= 10`）候補を受理すると、NVRTC/ptxas 側で不正な
    // 即値としてコンパイル失敗する（後段での失敗であり fail-closed
    // 契約違反）。ここで `stages <= 9`（`STAGES - 2 <= 7`）を検査し、
    // 実機・NVRTC 到達前に机上で拒否する。
    const MAX_WAIT_GROUP_IMMEDIATE: u32 = 7;
    const MAX_STAGES: u32 = MAX_WAIT_GROUP_IMMEDIATE + 2;
    if stages > MAX_STAGES {
        return Err(invalid(format!(
            "mma_f16_source_with_block_tile stages ({stages}) must be <= {MAX_STAGES} \
             (cp.async.wait_group immediate operand STAGES - 2 must fit in \
             [0, {MAX_WAIT_GROUP_IMMEDIATE}])"
        )));
    }
    // `validate_mma_kernel_config` と同じ kWarpGemmIterations 相当条件。
    if !bk.is_multiple_of(MMA_K) {
        return Err(invalid(format!(
            "mma_f16_source_with_block_tile bk ({bk}) must be a multiple of MMA_K ({MMA_K})"
        )));
    }
    let k_steps_per_stage = bk / MMA_K;
    if k_steps_per_stage < 2 {
        return Err(invalid(format!(
            "mma_f16_source_with_block_tile bk / MMA_K ({k_steps_per_stage}) must be >= 2"
        )));
    }
    // cp.async 16 バイト転送粒度の前提（`validate_mma_kernel_config` と同じ）。
    if !bm.is_multiple_of(8) || !bn.is_multiple_of(8) {
        return Err(invalid(format!(
            "mma_f16_source_with_block_tile bm ({bm}) and bn ({bn}) must both be multiples \
             of 8 (cp.async 16-byte transfer granularity)"
        )));
    }

    // warp タイル形状の検証（`mma_f16_source_with_warp_tiles` と同じ式だが
    // `MMA_BM`/`MMA_BN` 固定ではなく候補の `bm`/`bn` に対して行う）。
    let warp_m = warp_tiles_m
        .checked_mul(MMA_M)
        .ok_or_else(|| invalid(format!(
            "mma_f16_source_with_block_tile warp_tiles_m={warp_tiles_m} overflows u32 when multiplied by MMA_M={MMA_M}"
        )))?;
    let warp_n = warp_tiles_n
        .checked_mul(MMA_N)
        .ok_or_else(|| invalid(format!(
            "mma_f16_source_with_block_tile warp_tiles_n={warp_tiles_n} overflows u32 when multiplied by MMA_N={MMA_N}"
        )))?;
    if !bm.is_multiple_of(warp_m) || !bn.is_multiple_of(warp_n) {
        return Err(invalid(format!(
            "mma_f16_source_with_block_tile candidate warp tile {warp_m}x{warp_n} \
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
            invalid("mma_f16_source_with_block_tile block thread count overflow".to_string())
        })?;
    if threads > 1024 {
        return Err(invalid(format!(
            "mma_f16_source_with_block_tile candidate derives {threads} threads/block, \
             exceeding CUDA's per-block limit (1024)"
        )));
    }

    // 共有メモリ予算（`validate_mma_kernel_config` と同じ式。
    // `mma_f16_source_with_block_tile` ドキュメンテーションコメント
    // 「共有メモリ予算」参照）。
    let a_pad = bk.checked_add(8).ok_or_else(|| {
        invalid("mma_f16_source_with_block_tile A tile padded row width overflow".to_string())
    })?;
    let b_pad = bn.checked_add(8).ok_or_else(|| {
        invalid("mma_f16_source_with_block_tile B tile padded row width overflow".to_string())
    })?;
    let smem_bytes = bm
        .checked_mul(a_pad)
        .and_then(|a| bk.checked_mul(b_pad).and_then(|b| a.checked_add(b)))
        .and_then(|sum| sum.checked_mul(2))
        .and_then(|v| v.checked_mul(stages))
        .ok_or_else(|| {
            invalid("mma_f16_source_with_block_tile shared memory byte count overflow".to_string())
        })?;

    Ok(MmaBlockTileLayout {
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

/// 診断専用（`internal-diagnostics` feature 限定。イシュー #804）:
/// ブロックタイル（`bm`/`bn`/`bk`）・`cp.async` パイプライン段数
/// （`stages`）・warp タイル形状（`warp_tiles_m`/`_n`）・
/// `__launch_bounds__` を任意に組み合わせた候補カーネルソースを生成する。
///
/// `mma_f16_source_with_warp_tiles`（#803・#822）が warp タイル形状のみを
/// `mma_f16_source()` の `#define` へアンカー置換していたのに対し、本関数は
/// ブロックタイル・段数の `#define`（`BM`/`BN`/`BK`/`STAGES`/`A_PAD`/
/// `B_PAD`）も同じ方式で置換し、warp タイル置換と合成する。
///
/// # 本番結線の `validate_mma_kernel_config`（`render_mma_f16` 用）との違い
///
/// 本番検査は `cfg.stages != 3` を拒否する（cp.async 二値分岐撤去〈#492〉
/// 後もステージ可変化の本番結線〈#492 Phase B〉が未完のための暫定制約。
/// 本ファイル `validate_mma_kernel_config` 該当コメント参照）。本関数は
/// `mma_f16_source_uses_fixed_immediate_wait_with_loop_exit_drain`
/// テストが担保する段数一般形（ループ内 `wait_group %0;` の即値制約が
/// `STAGES - 2`）が既に成立していることを前提に `stages >= 2` のみを
/// 要求し、3 以外の段数も候補として生成できるようにする（#804 の
/// ステージ数増候補向け）。
///
/// # 共有メモリ予算（2 段階判定。#804 実装計画 Step 1）
///
/// `optin_budget_bytes` は呼び出し元がデバイス実測値
/// （`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`。GB10 実測
/// 101,376B。出典 `docs/perf/sm121-device-attributes.md`）から渡す
/// （`kernels_wmma_opt.rs::validate_wmma_tf32_staged_dyn_config` と同じ
/// 「ハードコード定数ではなく呼び出し元供給」の方針）。
///
/// - 静的共有メモリ予算（[`MMA_STATIC_SMEM_LIMIT_BYTES`]・48KiB）以下:
///   本番と同じ静的 `__shared__` 配列宣言のまま候補ソースを返す
///   （`needs_dynamic_smem=false`。呼び出し元は `mma_ptx_dump` の
///   `dump_ptx` に渡すだけで `ptxas -v` 計測できる）。
/// - 静的予算超・`optin_budget_bytes` 以下: 静的 `__shared__` 配列宣言
///   （`as_tile`/`bs_tile`）を `extern __shared__` バッファ上の
///   ポインタへ変換した候補ソースを返す（`needs_dynamic_smem=true`）。
///   多次元添字構文（`as_tile[stage][row][col]`）は配列本体
///   （`AsTileT`/`BsTileT` の `typedef`）をそのまま流用し、宣言 2 行の
///   置換のみでインデックス計算・バンク位相設計（`docs/perf/
///   cuda-gemm-mma-bank-conflict.md` §2）を不変に保つ。**この経路は
///   #804 実装セッション時点では `nvrtc`/`ptxas` 実機での構文検証を
///   経ていなかった**（当時は本 worktree・DGX Spark GB10 実機のいずれにも
///   CUDA toolkit へ到達できなかったため。計画 Step F 参照）が、
///   **イシュー #840 で GB10 実機の NVRTC/ptxas 構文検証・起動検証を
///   通過済み**（`examples/gemm_mma_block_tile_bench.rs` 経由の A/B 計測。
///   詳細は `docs/perf/cuda-gemm-mma-block-tile-stages.md` §2・§4）。本番
///   起動側の動的 SMEM opt-in 結線（`CudaFunction::set_attribute`・
///   `shared_mem_bytes`）は本関数のスコープ外のままであり、採否判断は
///   #842 へ委譲されている。
/// - `optin_budget_bytes` 超: 机上除外として `CudaError::InvalidKernelConfig`
///   を返す（実機到達を待たず判定できる）。
///
/// # 呼び出し元
///
/// `examples/mma_ptx_dump.rs`・`examples/gemm_mma_block_tile_bench.rs`
/// （いずれも `internal-diagnostics` feature 限定・`lib.rs::diagnostics`
/// 経由）専用。本番経路（`gemm_mma.rs::CudaMmaGemm`）はこの関数を呼ばない
/// （`mma_f16_source_with_warp_tiles` と同じ非消費契約）。
#[allow(dead_code)] // 理由は mma_f16_source_with_warp_tiles と同じ（非公開モジュール）
#[allow(clippy::too_many_arguments)] // 診断専用の候補パラメータを 1 関数に集約する設計上の要求
pub fn mma_f16_source_with_block_tile(
    bm: u32,
    bn: u32,
    bk: u32,
    stages: u32,
    warp_tiles_m: u32,
    warp_tiles_n: u32,
    launch_bounds: Option<u32>,
    optin_budget_bytes: u32,
) -> Result<String, CudaError> {
    mma_f16_source_with_block_tile_impl(
        bm,
        bn,
        bk,
        stages,
        warp_tiles_m,
        warp_tiles_n,
        launch_bounds,
        optin_budget_bytes,
        false,
    )
}

/// 診断専用（`internal-diagnostics` feature 限定。イシュー #855）:
/// [`mma_f16_source_with_block_tile`] と同じ候補パラメータで、静的
/// 48KiB 予算以下でも常に `extern __shared__` 動的 SMEM 変換
/// （[`mma_f16_source_with_block_tile_impl`] の `force_dynamic_smem`）を
/// 強制適用したソースを返す。
///
/// # 目的（イシュー #842 からの引き継ぎ・実行時観測の第一段）
///
/// #840 の GB10 実機 A/B で、動的 SMEM 変換を通る候補（`bt64x128_s4`・
/// `bt128x128_s3_wt2x4`）のみが CPU `f32::mul_add` 参照との数値一致に
/// fail した。#842 の机上調査では変換のアドレス計算・アライメントに
/// 欠陥を特定できず、「動的変換そのものが原因か・48KiB 超で初めて
/// 顕在化する候補定数側の潜在バグか」を実機の対照実験で切り分ける
/// 必要があるとして本イシューへ引き継がれた
/// （`docs/perf/cuda-gemm-mma-block-tile-stages.md` §7.1）。
///
/// 基準構成（BM=64/BN=128/BK=32/STAGES=3。静的 41,472B）を本関数で
/// 強制動的化して起動し、parity が pass すれば「変換は無罪、候補定数
/// 側の潜在バグ」、fail すれば「変換そのものに欠陥」と判定できる
/// （`docs/perf/cuda-gemm-mma-block-tile-stages.md` §8 参照）。
///
/// 呼び出し元は [`render_mma_f16_block_tile_forced_dynamic_smem`]
/// （起動まで完結させる A/B ランナー側のラッパー）のみを想定する
/// （`examples/gemm_mma_block_tile_bench.rs` 経由）。本番経路
/// （`gemm_mma.rs::CudaMmaGemm`）はこの関数に一切依存しない。
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn mma_f16_source_with_block_tile_forced_dynamic_smem(
    bm: u32,
    bn: u32,
    bk: u32,
    stages: u32,
    warp_tiles_m: u32,
    warp_tiles_n: u32,
    launch_bounds: Option<u32>,
    optin_budget_bytes: u32,
) -> Result<String, CudaError> {
    mma_f16_source_with_block_tile_impl(
        bm,
        bn,
        bk,
        stages,
        warp_tiles_m,
        warp_tiles_n,
        launch_bounds,
        optin_budget_bytes,
        true,
    )
}

/// [`mma_f16_source_with_block_tile`]／
/// [`mma_f16_source_with_block_tile_forced_dynamic_smem`] の共通実体。
/// `force_dynamic_smem` は後者専用の診断フラグ（イシュー #855）で、
/// 真の場合は静的予算以下の候補でも `extern __shared__` 変換
/// （[`DYNAMIC_SMEM_REPLACEMENT`]）を適用する。既存 8 引数の公開シグ
/// ネチャ（呼び出し元・テスト多数）を変えずに強制フラグを追加するため
/// 本関数を切り出した。
#[allow(clippy::too_many_arguments)]
fn mma_f16_source_with_block_tile_impl(
    bm: u32,
    bn: u32,
    bk: u32,
    stages: u32,
    warp_tiles_m: u32,
    warp_tiles_n: u32,
    launch_bounds: Option<u32>,
    optin_budget_bytes: u32,
    force_dynamic_smem: bool,
) -> Result<String, CudaError> {
    let invalid = |detail: String| CudaError::InvalidKernelConfig { detail };

    // レイアウト導出（不変条件検査込み）は [`derive_mma_block_tile_layout`]
    // へ切り出し済み（イシュー #840 実装計画「レイアウト導出ヘルパー」節。
    // `examples/gemm_mma_block_tile_bench.rs`（診断専用 A/B ランナー）が
    // 同じ式を再定義せず参照できるようにするため、`smem_bytes`/`threads`
    // 算出ロジックの単一の真実源をここへ集約した）。
    let layout = derive_mma_block_tile_layout(bm, bn, bk, stages, warp_tiles_m, warp_tiles_n)?;
    let (warps_n, threads, a_pad, b_pad, smem_bytes) = (
        layout.warps_n,
        layout.threads,
        layout.a_pad,
        layout.b_pad,
        layout.smem_bytes,
    );

    if let Some(v) = launch_bounds
        && v != threads
    {
        return Err(invalid(format!(
            "mma_f16_source_with_block_tile launch_bounds ({v}) must equal the derived thread \
             count ({threads})"
        )));
    }

    if smem_bytes > optin_budget_bytes {
        return Err(invalid(format!(
            "mma_f16_source_with_block_tile candidate bm={bm} bn={bn} bk={bk} stages={stages} \
             requires {smem_bytes} bytes of shared memory, exceeding the opt-in budget \
             ({optin_budget_bytes} bytes; CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)"
        )));
    }
    let needs_dynamic_smem = smem_bytes > MMA_STATIC_SMEM_LIMIT_BYTES || force_dynamic_smem;

    // ブロックタイル・段数の #define をアンカー置換する（`mma_f16_source()`
    // 実体は既定 config の bm=64/bn=128/bk=32/stages=3/a_pad=40/b_pad=136 が
    // 焼き込み済みのため、これらを候補値へ差し替える）。
    let source = replace_source_anchor(
        mma_f16_source().to_owned(),
        &format!("#define BM {MMA_BM}\n"),
        &format!("#define BM {bm}\n"),
        "mma_f16_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define BN {MMA_BN}\n"),
        &format!("#define BN {bn}\n"),
        "mma_f16_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define BK {MMA_BK}\n"),
        &format!("#define BK {bk}\n"),
        "mma_f16_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define WARPS_N {MMA_WARPS_N}\n"),
        &format!("#define WARPS_N {warps_n}\n"),
        "mma_f16_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define WARP_TILES_M {MMA_WARP_TILES_M}\n"),
        &format!("#define WARP_TILES_M {warp_tiles_m}\n"),
        "mma_f16_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define WARP_TILES_N {MMA_WARP_TILES_N}\n"),
        &format!("#define WARP_TILES_N {warp_tiles_n}\n"),
        "mma_f16_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define STAGES {MMA_STAGES}\n"),
        &format!("#define STAGES {stages}\n"),
        "mma_f16_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define A_PAD {MMA_A_PAD}\n"),
        &format!("#define A_PAD {a_pad}\n"),
        "mma_f16_source_with_block_tile",
    )?;
    let source = replace_source_anchor(
        source,
        &format!("#define B_PAD {MMA_B_PAD}\n"),
        &format!("#define B_PAD {b_pad}\n"),
        "mma_f16_source_with_block_tile",
    )?;

    // 動的 SMEM 化（本関数ドキュメンテーションコメント「共有メモリ予算」
    // 参照）。多次元添字構文をそのまま使えるよう `typedef` 配列型への
    // ポインタへ変換する（宣言 2 行のみの置換。以降のカーネル本体の
    // `as_tile[stage][row][col]`/`bs_tile[stage][row][col]` アクセスは
    // 無変更で成立する）。
    const STATIC_SMEM_ANCHOR: &str = "    __shared__ __align__(16) __half as_tile[STAGES][BM][A_PAD];\n    __shared__ __align__(16) __half bs_tile[STAGES][BK][B_PAD];\n";
    const DYNAMIC_SMEM_REPLACEMENT: &str = "    extern __shared__ __align__(16) unsigned char mma_dyn_smem[];\n    typedef __half MmaAsTileT[BM][A_PAD];\n    typedef __half MmaBsTileT[BK][B_PAD];\n    MmaAsTileT* as_tile = reinterpret_cast<MmaAsTileT*>(mma_dyn_smem);\n    MmaBsTileT* bs_tile = reinterpret_cast<MmaBsTileT*>(mma_dyn_smem + sizeof(MmaAsTileT) * STAGES);\n";
    let source = if needs_dynamic_smem {
        replace_source_anchor(
            source,
            STATIC_SMEM_ANCHOR,
            DYNAMIC_SMEM_REPLACEMENT,
            "mma_f16_source_with_block_tile",
        )?
    } else {
        source
    };

    let source = if let Some(v) = launch_bounds {
        const SIG_ANCHOR: &str = "extern \"C\" __global__ void gemm_mma_f16(";
        let sig_replacement =
            format!("extern \"C\" __global__ void __launch_bounds__({v}) gemm_mma_f16(");
        replace_source_anchor(
            source,
            SIG_ANCHOR,
            &sig_replacement,
            "mma_f16_source_with_block_tile",
        )?
    } else {
        source
    };

    Ok(source)
}

/// [`render_mma_f16_block_tile`] が返す、展開済み候補ソース・展開元
/// [`MmaBlockTileLayout`] を 1 個にまとめた descriptor（イシュー #840。
/// `kernels_wmma_opt.rs::RenderedWmmaTf32StagedDynKernel` と同型）。
///
/// フィールドは非公開。生ソースを外部へ返す公開メソッドは持たない
/// （`RenderedMmaKernel` と同じ「検査を経ずに `CudaFunction` へ到達する
/// 経路を作らない」契約）。診断専用（`internal-diagnostics` feature
/// 限定）: `examples/gemm_mma_block_tile_bench.rs` が唯一の呼び出し元。
/// 本番経路（`gemm_mma.rs::CudaMmaGemm`）はこの型に一切依存しない
/// （`MMA_BM`/`MMA_BN`/`MMA_STAGES` 等の本番定数は本イシューでは無変更）。
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RenderedMmaF16BlockTileKernel {
    source: String,
    layout: MmaBlockTileLayout,
    /// 実際に動的 SMEM 変種を使うか（イシュー #855）。既定の
    /// `render_mma_f16_block_tile` は `layout.needs_dynamic_smem()`
    /// （静的 48KiB 超）と一致するが、
    /// `render_mma_f16_block_tile_forced_dynamic_smem`（診断専用の対照
    /// 実験）は静的予算以下の候補でも真になる。`compile`/`launch_f16`
    /// はこのフィールドのみを見て動的 SMEM の起動側設定
    /// （`set_attribute`／`shared_mem_bytes`）を決める必要がある
    /// （`layout.needs_dynamic_smem()` を直接見ると、強制動的候補で
    /// `shared_mem_bytes=0` のまま起動してしまい「変換の欠陥」と
    /// 「起動側設定漏れ」が交絡する）。
    uses_dynamic_smem: bool,
}

impl RenderedMmaF16BlockTileKernel {
    /// カーネルソースを NVRTC コンパイル → 固定エントリポイント
    /// `"gemm_mma_f16"`（`mma_f16_source_with_block_tile` は `#define` 群の
    /// みを置換しシグネチャ名自体は変えないため、本番既定コンストラクタ
    /// `CompiledMmaKernel::compile` と同じエントリポイント名になる）の
    /// ロードまで完結させる。`layout.needs_dynamic_smem()` が真の候補
    /// （静的 48KiB 超）のみ `CudaFunction::set_attribute`
    /// （`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`。cudarc 0.19.8
    /// の安全 API・`unsafe` を要求しない）で opt-in 予算を設定する
    /// （`kernels_wmma_opt.rs::RenderedWmmaTf32StagedDynKernel::compile` と
    /// 同じ「必要時のみ opt-in する」方針）。
    ///
    /// プロセス内 LRU／ディスクキャッシュ（[`RenderedMmaKernel::compile`]
    /// の 3 段フォールバック）は使わない: 本 A/B ランナーは候補ごとに
    /// 1 回だけコンパイルすればよく（計測対象は起動後のカーネル実行時間の
    /// み）、キャッシュ層を経由する複雑さを避ける（`gemm_wmma_tf32_staged_
    /// stages_bench.rs` の `RenderedWmmaTf32StagedDynKernel::compile` も
    /// 同じ判断）。
    #[allow(dead_code)]
    pub fn compile(
        &self,
        device: &crate::device::CudaDevice,
    ) -> Result<CompiledMmaF16BlockTileKernel, CudaError> {
        let ptx = crate::nvrtc::compile_ptx(&self.source, device.arch())?;
        let func = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_mma_f16")?;
        if self.uses_dynamic_smem {
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
        Ok(CompiledMmaF16BlockTileKernel {
            func,
            layout: self.layout,
            uses_dynamic_smem: self.uses_dynamic_smem,
        })
    }

    /// テスト専用ソースアクセサ（`RenderedMmaKernel::source` と同じ理由・
    /// 同じ「本番公開 API には現れない」契約）。
    #[cfg(test)]
    fn source(&self) -> &str {
        &self.source
    }
}

/// 候補ブロックタイル・段数・warp タイル構成からカーネルソースを展開し、
/// [`RenderedMmaF16BlockTileKernel`] を返す（イシュー #840）。
///
/// `mma_f16_source_with_block_tile`（ソース文字列のみを返す既存 API。
/// #804）の結果と、その展開に使った [`derive_mma_block_tile_layout`] の
/// 結果を 1 個の descriptor へ束ねる薄いラッパー。`optin_budget_bytes`
/// 超過時は `mma_f16_source_with_block_tile` と同じ理由で
/// `CudaError::InvalidKernelConfig` を返す（呼び出し元
/// `examples/gemm_mma_block_tile_bench.rs` はこれを「机上除外」として
/// 非致命的に扱い、除外理由をログへ残してスイープを継続する）。
#[allow(dead_code, clippy::too_many_arguments)]
pub fn render_mma_f16_block_tile(
    bm: u32,
    bn: u32,
    bk: u32,
    stages: u32,
    warp_tiles_m: u32,
    warp_tiles_n: u32,
    launch_bounds: Option<u32>,
    optin_budget_bytes: u32,
) -> Result<RenderedMmaF16BlockTileKernel, CudaError> {
    render_mma_f16_block_tile_impl(
        bm,
        bn,
        bk,
        stages,
        warp_tiles_m,
        warp_tiles_n,
        launch_bounds,
        optin_budget_bytes,
        false,
    )
}

/// 診断専用（`internal-diagnostics` feature 限定。イシュー #855）:
/// [`render_mma_f16_block_tile`] と同じ候補パラメータで、
/// [`mma_f16_source_with_block_tile_forced_dynamic_smem`] を使い
/// `extern __shared__` 動的 SMEM 変換を強制適用した
/// [`RenderedMmaF16BlockTileKernel`] を返す（`uses_dynamic_smem=true`
/// 固定。`compile`/`launch_f16` が起動側 opt-in 設定・
/// `shared_mem_bytes` を実際に動的変種として扱うことを保証する）。
///
/// 呼び出し元は `examples/gemm_mma_block_tile_bench.rs` の対照実験行
/// （基準構成を強制動的化した control。`mma_f16_source_with_block_tile_
/// forced_dynamic_smem` ドキュメンテーションコメント「目的」節参照）。
#[allow(dead_code, clippy::too_many_arguments)]
pub fn render_mma_f16_block_tile_forced_dynamic_smem(
    bm: u32,
    bn: u32,
    bk: u32,
    stages: u32,
    warp_tiles_m: u32,
    warp_tiles_n: u32,
    launch_bounds: Option<u32>,
    optin_budget_bytes: u32,
) -> Result<RenderedMmaF16BlockTileKernel, CudaError> {
    render_mma_f16_block_tile_impl(
        bm,
        bn,
        bk,
        stages,
        warp_tiles_m,
        warp_tiles_n,
        launch_bounds,
        optin_budget_bytes,
        true,
    )
}

/// [`render_mma_f16_block_tile`]／
/// [`render_mma_f16_block_tile_forced_dynamic_smem`] の共通実体
/// （イシュー #855。`mma_f16_source_with_block_tile_impl` と同じ
/// 「既存 8 引数シグネチャを変えずに強制フラグを追加する」設計）。
#[allow(clippy::too_many_arguments)]
fn render_mma_f16_block_tile_impl(
    bm: u32,
    bn: u32,
    bk: u32,
    stages: u32,
    warp_tiles_m: u32,
    warp_tiles_n: u32,
    launch_bounds: Option<u32>,
    optin_budget_bytes: u32,
    force_dynamic_smem: bool,
) -> Result<RenderedMmaF16BlockTileKernel, CudaError> {
    let layout = derive_mma_block_tile_layout(bm, bn, bk, stages, warp_tiles_m, warp_tiles_n)?;
    if layout.smem_bytes > optin_budget_bytes {
        return Err(CudaError::InvalidKernelConfig {
            detail: format!(
                "render_mma_f16_block_tile candidate bm={bm} bn={bn} bk={bk} stages={stages} \
                 requires {} bytes of shared memory, exceeding the opt-in budget \
                 ({optin_budget_bytes} bytes; CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)",
                layout.smem_bytes
            ),
        });
    }
    let source = mma_f16_source_with_block_tile_impl(
        bm,
        bn,
        bk,
        stages,
        warp_tiles_m,
        warp_tiles_n,
        launch_bounds,
        optin_budget_bytes,
        force_dynamic_smem,
    )?;
    let uses_dynamic_smem = layout.needs_dynamic_smem() || force_dynamic_smem;
    Ok(RenderedMmaF16BlockTileKernel {
        source,
        layout,
        uses_dynamic_smem,
    })
}

/// [`RenderedMmaF16BlockTileKernel::compile`] が返す、コンパイル済み
/// `CudaFunction`・展開元 [`MmaBlockTileLayout`] を不可分に束ねた
/// descriptor（イシュー #840。`CompiledMmaKernel`・
/// `CompiledWmmaTf32StagedDynKernel` と同型）。
#[allow(dead_code)]
pub struct CompiledMmaF16BlockTileKernel {
    func: cudarc::driver::CudaFunction,
    layout: MmaBlockTileLayout,
    /// [`RenderedMmaF16BlockTileKernel::uses_dynamic_smem`] を `compile`
    /// が引き継いだもの（イシュー #855）。`launch_f16` は
    /// `layout.needs_dynamic_smem()` ではなく本フィールドを見て
    /// `shared_mem_bytes` を決める（強制動的候補で `layout` 自体は
    /// 静的判定のままのため、`layout` 側だけを見ると起動時
    /// `shared_mem_bytes=0` のまま `extern __shared__` カーネルを
    /// 起動してしまう）。
    uses_dynamic_smem: bool,
}

impl CompiledMmaF16BlockTileKernel {
    /// [`CompiledMmaKernel::launch_f16`] と同じ検証手順（`validate_launch_
    /// shape` 相当は候補生成時に固定済みのため不要・`validate_gemm_dims`／
    /// `validate_output_len`／no-op 早期 return／`validate_mma_alignment`／
    /// grid y 上限検査／K タイル境界検査）に加え、`LaunchConfig.
    /// shared_mem_bytes` へ `self.layout.smem_bytes`（動的変種のみ非零。
    /// 静的変種は 0）を設定する。
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn launch_f16(
        &self,
        stream: &CudaStream,
        a_dev: &CudaSlice<f16>,
        b_dev: &CudaSlice<f16>,
        c_dev: &mut CudaSlice<f16>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        crate::gemm::validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        crate::gemm::validate_output_len(c_dev.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        crate::gemm_mma::validate_mma_alignment(n, k)?;

        const MAX_GRID_DIM_Y: u32 = 65_535;
        let grid_y = m.div_ceil(self.layout.bm);
        if grid_y > MAX_GRID_DIM_Y {
            return Err(CudaError::InvalidShape {
                detail: format!(
                    "mma_f16 block-tile candidate grid_dim.y ({grid_y}) exceeds CUDA's \
                     {MAX_GRID_DIM_Y} limit for grid dimensions y/z (bm={}); m={m} is too large",
                    self.layout.bm
                ),
            });
        }
        validate_mma_k_tile_bound(k, self.layout.bk)?;

        let smem_bytes_u32 = if self.uses_dynamic_smem {
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

        // SAFETY: `CompiledMmaKernel::launch_f16` と同一の根拠。カーネル
        // 引数は上記で検証済みの m/n/k から導出しており、カーネル内の
        // 手動境界チェック（REQ-8）と合わせて OOB 読み書きが起きない
        // 根拠とする。`shared_mem_bytes` は `RenderedMmaF16BlockTileKernel::
        // compile` が算出・opt-in 設定した値（`self.layout.smem_bytes`）と
        // 同一であり、カーネル側 `extern __shared__` の実際の使用量を
        // 過不足なく満たす（静的変種は 0 のまま。static `__shared__`
        // 宣言は起動時設定を要求しない）。
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

    /// Rust 側タイル定数が既定構成の生成ソース内 `#define` と食い違わない
    /// ことを検査する（`kernels_wmma.rs::wmma_tile_constant_matches_kernel_source_defines`
    /// と同じ方針。値の不一致はコンパイルエラーにならず誤った積和結果を
    /// 静かに生成しうるため CI 上で機械検出する）。イシュー #516 でテンプレート
    /// 展開へ移行したため、静的リテラルではなく `mma_f16_source()`（既定
    /// config の render 結果）を対象にする。
    #[test]
    fn mma_tile_constants_match_kernel_source_defines() {
        let src = mma_f16_source();
        for (name, value) in [
            ("MMA_M", MMA_M),
            ("MMA_N", MMA_N),
            ("MMA_K", MMA_K),
            ("BM", MMA_BM),
            ("BN", MMA_BN),
            ("BK", MMA_BK),
            ("A_PAD", MMA_A_PAD),
            ("B_PAD", MMA_B_PAD),
            ("STAGES", MMA_STAGES),
            ("WARPS_N", MMA_WARPS_N),
            ("WARP_TILES_M", MMA_WARP_TILES_M),
            ("WARP_TILES_N", MMA_WARP_TILES_N),
        ] {
            let expected = format!("#define {name} {value}");
            assert!(
                src.contains(&expected),
                "mma_f16_source() の `#define {name}` が Rust 側定数（{value}）と一致しません"
            );
        }
    }

    /// 既定構成（全次元 `Dynamic`）では `#define DIM_* <カーネル引数>`
    /// 形式でカーネル引数へ間接するのみで、既存カーネルとプリプロセス後
    /// 等価であることをロックする（実装計画 4.4 節「境界検査・数値契約の
    /// 非後退」）。
    #[test]
    fn mma_default_config_dim_defines_alias_kernel_parameters() {
        let src = mma_f16_source();
        for expected in ["#define DIM_M m", "#define DIM_N n", "#define DIM_K k"] {
            assert!(
                src.contains(expected),
                "mma_f16_source() に既定次元マクロ `{expected}` が見つかりません"
            );
        }
    }

    /// TASK-11.3（tensor core 命令使用の証跡）を兼ねる: `mma.sync`・
    /// `ldmatrix`・`cp.async` の主要命令がソース文字列内に実在することを
    /// ロックする（`kernels_wmma.rs::wmma_f16_source_uses_wmma_instructions`
    /// と同じ方針）。
    #[test]
    fn mma_f16_source_uses_mma_sync_ldmatrix_cp_async_instructions() {
        let src = mma_f16_source();
        for needle in [
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "cp.async.wait_group",
        ] {
            assert!(
                src.contains(needle),
                "mma_f16_source() に tensor core 命令 `{needle}` が見つかりません"
            );
        }
    }

    /// REQ-8: A/B タイルの `cp.async` src-size ゼロ充填・エピローグ
    /// guarded store の手動境界チェックが除去されていないことをロックする
    /// （`kernels_wmma.rs` の REQ-8 テスト方針と同様、性能最適化を理由に
    /// 境界検査が省略される回帰を防ぐ）。マクロ化（イシュー #516）に伴い
    /// needle は `DIM_M`/`DIM_N`/`DIM_K` 形式へ更新している。
    #[test]
    fn mma_f16_source_retains_req8_boundary_guards() {
        let src = mma_f16_source();
        for needle in [
            "gr < DIM_M && gc < DIM_K",
            "gr < DIM_K && gc < DIM_N",
            "r0 < DIM_M && c0 < DIM_N",
            "r0 < DIM_M && c1 < DIM_N",
            "r1 < DIM_M && c0 < DIM_N",
            "r1 < DIM_M && c1 < DIM_N",
        ] {
            assert!(
                src.contains(needle),
                "mma_f16_source() に REQ-8 境界チェック `{needle}` が見つかりません"
            );
        }
    }

    /// #805 受け入れ基準 1 の機械検査: エピローグの C 書き戻しが
    /// `__half2` ペア store（隣接列 c0/c1 をまとめて 1 回で書く）へ
    /// 置き換わっており、旧来の 4 連スカラー store パターン（4 要素とも
    /// `__float2half` の直書き）が residual していないことをロックする。
    /// 旧パターンが残ると、境界検査（`mma_f16_source_retains_req8_boundary_guards`）
    /// は通過するが store 命令数の半減という本イシューの効果が失われるため、
    /// 実装がベクトル化 store へ確実に切り替わっていることを別軸で保証する。
    #[test]
    fn mma_f16_source_epilogue_uses_half2_pair_store() {
        let src = mma_f16_source();
        // (a) __half2 ペア store が存在する（r0 側・r1 側の 2 箇所）。
        for needle in [
            "*reinterpret_cast<__half2*>(&c[(size_t)r0 * DIM_N + c0]) =",
            "__floats2half2_rn(d[mi][nj][0], d[mi][nj][1]);",
            "*reinterpret_cast<__half2*>(&c[(size_t)r1 * DIM_N + c0]) =",
            "__floats2half2_rn(d[mi][nj][2], d[mi][nj][3]);",
        ] {
            assert!(
                src.contains(needle),
                "mma_f16_source() に #805 の __half2 ペア store `{needle}` が見つかりません"
            );
        }
        // (b) 旧来の 4 連スカラー store（c1 要素への直書き）が残っていない。
        // c0 側のスカラー store（defensive fallback）は不変条件下で到達不能
        // ながら残置するため、c1 側の旧パターンのみを不在検査する。
        for stale_needle in [
            "c[(size_t)r0 * DIM_N + c1] = __float2half(d[mi][nj][1]);",
            "c[(size_t)r1 * DIM_N + c1] = __float2half(d[mi][nj][3]);",
        ] {
            assert!(
                !src.contains(stale_needle),
                "mma_f16_source() に #805 で置き換えたはずの旧スカラー store \
                 `{stale_needle}` が残存しています"
            );
        }

        // (c) swizzle 変種（`mma_f16_source_with_swizzle`）はブロック原点
        // アンカー（`block_row0`/`block_col0`）のみを `replacen` し
        // エピローグには触れない（`mma_f16_source_with_swizzle` 実装の
        // `ANCHOR` 定数参照）。#782 で本番既定コンストラクタへ結線済みの
        // ため GB10 実機で実際に実行される経路であり、その出力にも
        // `__half2` ペア store が引き継がれていることを検査する。
        let swizzled =
            mma_f16_source_with_swizzle(4).expect("mma_f16_source_with_swizzle(4) が失敗しました");
        assert!(
            swizzled.contains("__floats2half2_rn(d[mi][nj][0], d[mi][nj][1]);"),
            "mma_f16_source_with_swizzle() に #805 の __half2 ペア store が \
             見つかりません（swizzle 変種でエピローグが欠落・変質した疑い）"
        );
    }

    /// `MMA_BLOCK_THREADS` が CUDA の 1 ブロックあたり最大スレッド数
    /// （1024）を超えないことは本ファイル冒頭の `const _: () =
    /// assert!(...)` でコンパイル時に検査済み。本テストは
    /// `MMA_WARPS_M`/`MMA_WARPS_N` からの導出式が崩れていないことに加え、
    /// #494 のブロックタイル拡大（`32x64`→`64x128`）で再導出された値
    /// （16 warp = 512 スレッド。B-1 時点のスレッド数への回復）をロックする
    /// （本ファイル `MMA_WARPS_M`/`MMA_WARPS_N` 定数直下のドキュメンテーション
    /// コメント参照）。
    #[test]
    fn mma_block_threads_matches_warp_layout() {
        assert_eq!(MMA_BLOCK_THREADS, MMA_WARPS_M * MMA_WARPS_N * 32);
        assert_eq!(MMA_WARPS_M, 2, "#494 再導出値: MMA_BM / MMA_WARP_M");
        assert_eq!(MMA_WARPS_N, 8, "#494 再導出値: MMA_BN / MMA_WARP_N");
        assert_eq!(MMA_BLOCK_THREADS, 512);
    }

    /// #494 受け入れ基準の回帰防止（#498 でパディング込みの値へ更新）:
    /// ブロックタイル拡大後・バンクコンフリクト対策パディング適用後の
    /// 静的共有メモリ使用量（`(64*40 + 32*136) * 2B * 3 stages` =
    /// 41,472B。パディング前は 36,864B だった）と `kWarpGemmIterations`
    /// 相当条件（`MMA_K_STEPS_PER_STAGE >= 2`）をロックする。前者は本
    /// ファイル冒頭の `const _: () = assert!(...)`（48KiB 上限検査）とは
    /// 独立に、候補表（`docs/perf/cuda-gemm-mma-block-tile.md` §3）が
    /// 採用した候補 B へ #498 のパディングを適用した具体値そのものが
    /// 崩れていないことを検査する。
    #[test]
    fn mma_shared_mem_and_k_steps_match_block_tile_expansion() {
        assert_eq!(MMA_SHARED_MEM_BYTES, 41_472);
        assert_eq!(MMA_K_STEPS_PER_STAGE, 2);
    }

    /// #498 の回帰防止: パディング後の行ストライド（バイト）が共有メモリ
    /// 32 バンク（各 4B）を 8 行（`ldmatrix.x4`/`.x2` が読む行数）で完全に
    /// 分散させる非 2 冪バンク位相であることをロックする（[`MMA_A_PAD`]/
    /// [`MMA_B_PAD`] 定数直下のドキュメンテーションコメント・
    /// `docs/perf/cuda-gemm-mma-bank-conflict.md` §2 参照）。バンク番号は
    /// `(行バイトオフセット / 4) % 32`（CUDA 共有メモリの標準 32 バンク・
    /// 4B/バンク・128B 周期モデル）。
    #[test]
    fn mma_tile_padding_distributes_bank_phase_across_rows() {
        assert_eq!(MMA_A_PAD, 40, "MMA_A_PAD は #498 で BK+8 = 40 に固定");
        assert_eq!(MMA_B_PAD, 136, "MMA_B_PAD は #498 で BN+8 = 136 に固定");

        let a_stride_bytes = MMA_A_PAD * 2;
        let b_stride_bytes = MMA_B_PAD * 2;

        let bank_of = |stride_bytes: u32, row: u32| -> u32 { (row * stride_bytes / 4) % 32 };

        for (name, stride_bytes) in [("A", a_stride_bytes), ("B", b_stride_bytes)] {
            let mut banks: Vec<u32> = (0..8).map(|row| bank_of(stride_bytes, row)).collect();
            banks.sort_unstable();
            banks.dedup();
            assert_eq!(
                banks.len(),
                8,
                "{name} タイル: パディング後も 8 行の開始バンクが分散していません \
                 （非 2 冪パディングによるバンク位相分散が崩れている可能性）"
            );
        }
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
        let src = mma_f16_source();
        assert!(
            !src.contains("if (t == num_k_tiles - 1)"),
            "mma_f16_source() に MMA_STAGES=3 専用の wait_group 二値分岐が残っています \
             （#492 でループ内固定即値＋ループ外 drain へ整理したはず）"
        );
        assert!(
            src.contains(r#"asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));"#),
            "mma_f16_source() のループ内 wait が段数一般形の即値制約（\"n\"(STAGES - 2)）ではありません"
        );
        // 数字即値 `wait_group 0;` の出現位置が `#undef LOAD_A_STAGE`
        // （ループ本体終了直後の目印）より手前にある、すなわちループの
        // 外側（ループ末尾 `}` の後）に置かれていることを位置関係で検査
        // する。カーネルソースの PTX asm 文字列内部は `\n`（バックスラッシュ
        // + n の 2 文字）が実際の改行ではなくリテラルとして現れるため、
        // 改行を含む固定文字列一致ではなく `find` によるインデックス比較
        // を用いる。
        let undef_pos = src
            .find("#undef LOAD_A_STAGE")
            .expect("mma_f16_source() に #undef LOAD_A_STAGE が見つかりません");
        let drain_pos = src
            .rfind("cp.async.wait_group 0;")
            .expect("mma_f16_source() に cp.async.wait_group 0; が見つかりません");
        assert!(
            drain_pos < undef_pos,
            "mma_f16_source() のループ外 drain（wait_group 0）が #undef LOAD_A_STAGE より \
             後ろにあります（ループ内へ紛れ込んでいないか確認すること）"
        );
        // ループ本体の閉じ `}`（`for (int t = ...)` ループ末尾。直前の
        // `__syncthreads();` を目印にする）より drain が後ろにあること。
        let loop_syncthreads_pos = src
            .rfind("asm volatile(\"cp.async.commit_group;")
            .expect("mma_f16_source() にループ末尾の cp.async.commit_group が見つかりません");
        assert!(
            drain_pos > loop_syncthreads_pos,
            "mma_f16_source() のループ外 drain（wait_group 0）がループ末尾の commit_group より \
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
        let src = mma_f16_source();
        let count = src.matches("cp.async.wait_group 0;").count();
        assert_eq!(
            count, 1,
            "mma_f16_source() 中の `cp.async.wait_group 0;`（数字即値）出現数が 1 ではありません \
             （ループ外 drain の 1 箇所のみが正。段数由来の数字リテラルがループ内へ \
             再導入されていないか確認すること）"
        );
        // ループ内 wait は数字即値ではなく `%0` プレースホルダ＋`"n"` 制約
        // 経由の段数一般形でなければならない。
        assert!(
            !src.contains("cp.async.wait_group 1;"),
            "mma_f16_source() に MMA_STAGES=3 専用の数字即値 `wait_group 1;` が残っています"
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
    /// 冒頭コメント「検証状態」）だが、ソース文字列レベルの整合は
    /// ここで通常 CI 下でも検査できる。
    ///
    /// #498 追補: パディング（[`MMA_A_PAD`]/[`MMA_B_PAD`]）適用後は共有
    /// メモリ使用量が増える（36,864B→41,472B。本ファイル
    /// [`MMA_SHARED_MEM_BYTES`] 定数直下参照）ため、`stages=4` はもはや
    /// 48KiB 上限に収まらない（`(64*40+32*136)*2*4` = 55,296B）。
    /// `docs/perf/cuda-gemm-mma-block-tile.md` §3 の「STAGES=4 は
    /// パディング前で余裕ゼロ」という記述はパディング後は「不成立」へ
    /// 更新済み（同ファイル参照）。本テストは段数依存リテラルが残らない
    /// ことは両段数について維持しつつ、SMEM 検査は「stages=2 は上限内・
    /// stages=4 はパディング後は上限超過（BM/BN 縮小なしでは不成立）」を
    /// 明示的に assert する形へ改訂した。
    #[test]
    fn mma_f16_source_stages_are_swappable_without_kernel_source_edits() {
        for stages in [2u32, 4u32] {
            let src = mma_f16_source_with_stages(stages);

            // 段数依存の旧来リテラル・分岐が残っていないこと（この不変条件
            // 自体は段数にもパディングにも依存しないため両段数で維持）。
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
            // と同じ式を stages 可変で検査する。#498 でパディング込みの
            // 式へ更新）。
            let shared_mem_bytes = (MMA_BM * MMA_A_PAD + MMA_BK * MMA_B_PAD) * 2 * stages;
            if stages == 2 {
                assert!(
                    shared_mem_bytes <= MMA_STATIC_SMEM_LIMIT_BYTES,
                    "stages={stages}: 共有メモリ使用量 {shared_mem_bytes}B が 48KiB を \
                     超えています（stages=2 はパディング後も上限内であるはず）"
                );
            } else {
                assert!(
                    shared_mem_bytes > MMA_STATIC_SMEM_LIMIT_BYTES,
                    "stages={stages}: 共有メモリ使用量 {shared_mem_bytes}B が 48KiB \
                     以下です（#498 のパディング後は stages=4 は BM/BN 縮小なしでは \
                     不成立になるはずだが、上限超過が検出されなかった）"
                );
            }
        }
    }

    /// `mma_f16_source_stages_are_swappable_without_kernel_source_edits` が
    /// 使うヘルパー: `#define STAGES <MMA_STAGES>` 行のみを `stages` へ
    /// 置換したソース文字列を返す。置換が正確に 1 回起きたことを assert
    /// することで、ヘルパー自体の壊れ（`#define STAGES` 行の文言変化で
    /// マッチしなくなる等）を検出する。
    fn mma_f16_source_with_stages(stages: u32) -> String {
        let src = mma_f16_source();
        let from = format!("#define STAGES {MMA_STAGES}\n");
        let to = format!("#define STAGES {stages}\n");
        let count = src.matches(&from).count();
        assert_eq!(
            count, 1,
            "mma_f16_source() 中の `{from:?}` の出現数が 1 ではありません（ヘルパーの \
             前提が崩れています）"
        );
        src.replacen(&from, &to, 1)
    }

    /// PR #255 レビュー指摘の回帰防止: A/B タイルロードの範囲外チャンク
    /// （`valid=0` のゼロ充填）でも `cp.async` ソースアドレスの列オフセット
    /// クランプが 16 バイト（8 要素）境界に切り下げられていることを
    /// ロックする（`k-1`/`n-1` への素朴なクランプはアラインを崩し
    /// 未定義動作になりうる。本ファイル `LOAD_A_STAGE`/`LOAD_B_STAGE`
    /// マクロ直前のコメント参照）。needle は `DIM_K`/`DIM_N` マクロ化後の
    /// 形式（イシュー #516）。
    #[test]
    fn mma_f16_source_zero_fill_clamp_stays_16_byte_aligned() {
        let src = mma_f16_source();
        for needle in ["((DIM_K - 1) / 8) * 8", "((DIM_N - 1) / 8) * 8"] {
            assert!(
                src.contains(needle),
                "mma_f16_source() に 16 バイト整列クランプ `{needle}` が見つかりません"
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
        let src = mma_f16_source();
        assert!(
            src.contains("int a_quad_row = (lane / 8) % 2;")
                && src.contains("int a_quad_col = (lane / 8) / 2;"),
            "mma_f16_source() の A フラグメント象限順序（TL/BL/TR/BR）が見つかりません"
        );
    }

    /// 受け入れ基準 2（コンパイル時展開）: kstep ループ直前に `#pragma unroll`
    /// が付与されていることをロックする（実装計画 4.3 節）。
    #[test]
    fn mma_f16_source_has_pragma_unroll_before_kstep_loop() {
        let src = mma_f16_source();
        let idx = src
            .find("for (int kstep = 0; kstep < BK / MMA_K; ++kstep)")
            .expect("kstep ループが見つかりません");
        let before = &src[..idx];
        let pragma_idx = before
            .rfind("#pragma unroll")
            .expect("kstep ループ直前に #pragma unroll が見つかりません");
        // #pragma unroll と for の間に他のステートメントが挟まっていない
        // ことを確認する（空白・コメント行のみ許容）。
        let between = &src[pragma_idx + "#pragma unroll".len()..idx];
        assert!(
            between.trim().starts_with("//") || between.trim().is_empty(),
            "#pragma unroll と kstep ループの間に余計な文があります: {between:?}"
        );
    }

    /// 非既定 config（`bm=64, bn=64, bk=32`・M 次元を `Static(4096)` で
    /// 焼き込み）での特化 render: `#define` 実値・派生定数（`WARPS_N` 等）・
    /// `#define DIM_M 4096` 形式の焼き込みが正しく展開され、REQ-8 ガードが
    /// 引き続き存在することを検査する（実装計画 7 節「特化 render」）。
    #[test]
    fn render_mma_f16_specializes_tile_and_static_dim() {
        let cfg = MmaKernelConfig {
            bm: 64,
            bn: 64,
            bk: 32,
            stages: 3,
            dim_m: DimSpec::Static(4096),
            dim_n: DimSpec::Dynamic,
            dim_k: DimSpec::Dynamic,
            dtype: MmaDtype::F16,
        };
        let rendered = render_mma_f16(&cfg).expect("有効な構成が拒否されました");
        // dim_m=Static(4096)・dim_n/dim_k=Dynamic のため、実起動形状は
        // m=4096 固定・n/k は任意（ここでは後段の validate_launch_shape
        // 呼び出しと揃えて n=64・k=32 を使う）。テスト専用アクセサ
        // `source()`（`#[cfg(test)]`。本番経路には存在しない）で生成
        // 内容のみを検査する。
        let src = rendered.source();

        for expected in [
            "#define BM 64",
            "#define BN 64",
            "#define BK 32",
            "#define WARPS_N 4", // 64 / MMA_WARP_N(16)（#493 レジスタブロッキング後の式）
            "#define DIM_M 4096",
            "#define DIM_N n",
            "#define DIM_K k",
        ] {
            assert!(
                src.contains(expected),
                "特化 render に `{expected}` が見つかりません: config={cfg:?}"
            );
        }
        for needle in ["gr < DIM_M && gc < DIM_K", "r0 < DIM_M && c0 < DIM_N"] {
            assert!(
                src.contains(needle),
                "特化 render に REQ-8 境界チェック `{needle}` が見つかりません"
            );
        }

        // dim_m=Static(4096) のため、実際の起動形状 m=16（コンパイル時に
        // 焼き込んだ値と食い違う）は fail-closed に拒否されなければ
        // ならない。`CompiledMmaKernel::launch_f16` は実機依存の
        // `CudaFunction`／`CudaStream` なしに単体テストできないため
        // （cudarc はテスト用コンストラクタを持たない）、同じ検査を内部で
        // 実行する `MmaKernelConfig::validate_launch_shape` を直接検査する
        // （PR #643 codex-review 再々指摘への対応。ロジックの単一の真実源は
        // `cfg.validate_launch_shape` であり、`CompiledMmaKernel::launch_f16`
        // はこれへ委譲するだけのため、ここでの検査で契約全体をカバーする）。
        assert!(cfg.validate_launch_shape(4096, 64, 32).is_ok());
        assert!(matches!(
            cfg.validate_launch_shape(16, 64, 32),
            Err(CudaError::InvalidKernelConfig { .. })
        ));
    }

    /// イシュー #531 受け入れ基準 3 項（境界検査の非省略・REQ-8）:
    /// `render_mma_f16_specializes_tile_and_static_dim` は `dim_m=Static`
    /// のみを検査しており、`gemm_auto::CompiledDims::STATIC_NK`（N/K
    /// 静的化・M 動的）・`STATIC_MNK`（全次元静的化）相当の config でも
    /// REQ-8 手動境界チェック（`kernels_mma.rs` 冒頭コメント「境界検査」）
    /// が render 後のソースに残存していることを検査する（イシュー #531
    /// 実装計画 §3.4）。テンプレート展開（`#define` によるコンパイル時
    /// 定数の焼き込み。イシュー #516）は演算命令列・境界チェック文自体を
    /// 変更しない設計のため、いずれの `DimSpec` 組合せでも同一の needle
    /// が残ることを機械的にロックする。
    #[test]
    fn render_mma_f16_retains_req8_boundary_guards_for_static_nk_and_mnk() {
        // `STATIC_NK` 相当: M=Dynamic・N/K=Static（`nvrtc::CompiledDims::STATIC_NK`
        // と同じ選択。本モジュールは `nvrtc` に依存しないため値は直接
        // 構築する）。
        let static_nk = MmaKernelConfig {
            dim_m: DimSpec::Dynamic,
            dim_n: DimSpec::Static(128),
            dim_k: DimSpec::Static(64),
            ..MmaKernelConfig::default()
        };
        // `STATIC_MNK` 相当: 全次元 Static。
        let static_mnk = MmaKernelConfig {
            dim_m: DimSpec::Static(4096),
            dim_n: DimSpec::Static(128),
            dim_k: DimSpec::Static(64),
            ..MmaKernelConfig::default()
        };

        for cfg in [static_nk, static_mnk] {
            let rendered = render_mma_f16(&cfg).expect("有効な構成が拒否されました");
            let src = rendered.source();
            for needle in [
                "gr < DIM_M && gc < DIM_K",
                "gr < DIM_K && gc < DIM_N",
                "r0 < DIM_M && c0 < DIM_N",
                "r0 < DIM_M && c1 < DIM_N",
                "r1 < DIM_M && c0 < DIM_N",
                "r1 < DIM_M && c1 < DIM_N",
            ] {
                assert!(
                    src.contains(needle),
                    "特化 render（config={cfg:?}）に REQ-8 境界チェック \
                     `{needle}` が見つかりません"
                );
            }
        }
    }

    /// 決定性の機械検査（#516 実装計画 4 節・§8「スコープ外」の C-5/C-2
    /// キャッシュ系タスクが本 render の出力をハッシュ材料として使う前提の
    /// 検査）。`render_mma_f16_unchecked` は `format!` のみで構成される
    /// 純関数（`HashMap` 走査・乱数・時刻等の非決定要素を持たない）だが、
    /// 同一 `MmaKernelConfig` から 2 回 render して byte 単位一致することを
    /// 明示的にロックし、将来の実装変更（例: 走査順が意味を持つデータ構造
    /// への置き換え）が非決定性を持ち込む回帰を検出できるようにする。
    #[test]
    fn render_mma_f16_is_deterministic_for_same_config() {
        let cfg = MmaKernelConfig {
            bm: 64,
            bn: 128,
            bk: 32,
            stages: 3,
            dim_m: DimSpec::Static(4096),
            dim_n: DimSpec::Static(4096),
            dim_k: DimSpec::Static(4096),
            dtype: MmaDtype::F16,
        };
        let first = render_mma_f16(&cfg)
            .expect("有効な構成が拒否されました")
            .source()
            .to_owned();
        let second = render_mma_f16(&cfg)
            .expect("有効な構成が拒否されました")
            .source()
            .to_owned();
        assert_eq!(
            first, second,
            "同一 MmaKernelConfig からの render_mma_f16 が byte 一致しません \
             （キャッシュキー材料としての決定性契約が崩れています）"
        );
    }

    /// イシュー #519（C-7）受け入れ基準 2 の直接検証: `CompiledDims::
    /// STATIC_NK`（N/K 定数化・M 動的。`gemm_auto.rs::dim_specs_for` の
    /// 既定ポリシー相当）で `dim_m=Dynamic` の config は、`nvrtc.rs`
    /// 側でキャッシュキーが縮退する（`CompiledDims::cache_shape` が M を
    /// sentinel 0 へ正規化する）だけでなく、**実際に render される
    /// カーネルソース自体に M の実行時値を表す定数が一切焼き込まれない**
    /// ことを直接検査する（`#define DIM_M m` の `m` はカーネル引数名で
    /// あり数値ではない）。対照として `dim_m=Static(m)` は焼き込み値
    /// ごとに異なるソースを render することも確認し、「定数化した次元
    /// だけがキャッシュキー・ソース両面で分岐する」契約を固定する。
    /// `gemm_auto.rs` 側のキャッシュエントリ数テスト
    /// （`static_nk_collapses_varying_m_into_a_single_cache_entry` 等）は
    /// `CudaKernelDescriptor`（正規化済み shape・非公開ソース）のみを
    /// 比較材料にしており、「同一キーだが異なるソースが縮退する」という
    /// キャッシュ汚染（A08）の否定的証拠にはならない。`RenderedMmaKernel::
    /// source()` は `#[cfg(test)]` のため本テストは `kernels_mma.rs`
    /// 内に置く必要がある（`gemm_auto.rs` からは参照できない）。
    #[test]
    fn dynamic_m_dim_render_never_bakes_in_a_concrete_m_value() {
        let cfg_for = |dim_m: DimSpec| MmaKernelConfig {
            bm: 64,
            bn: 128,
            bk: 32,
            stages: 3,
            dim_m,
            dim_n: DimSpec::Static(128),
            dim_k: DimSpec::Static(256),
            dtype: MmaDtype::F16,
        };

        let dynamic_src = render_mma_f16(&cfg_for(DimSpec::Dynamic))
            .expect("Dynamic dim_m の既定タイル構成は有効なはず")
            .source()
            .to_owned();
        assert!(
            dynamic_src.contains("#define DIM_M m"),
            "dim_m=Dynamic は DIM_M をカーネル引数 `m` の間接参照へ \
             展開しなければならない（数値の焼き込みなし）"
        );

        // 定数化した M ごとに異なるソースが render されること（縮退させて
        // はならない）。
        let static_8 = render_mma_f16(&cfg_for(DimSpec::Static(8)))
            .expect("Static(8) は有効な構成のはず")
            .source()
            .to_owned();
        let static_64 = render_mma_f16(&cfg_for(DimSpec::Static(64)))
            .expect("Static(64) は有効な構成のはず")
            .source()
            .to_owned();
        assert_ne!(
            static_8, static_64,
            "dim_m=Static(m) は焼き込み値ごとに異なるソースを render しなければならない"
        );
        assert_ne!(
            dynamic_src, static_8,
            "dim_m=Dynamic と dim_m=Static(8) は異なるソースを render しなければならない"
        );
    }

    /// フェイルクローズド検証（実装計画 7 節・4.2 節）: SMEM 予算超過・
    /// 倍数違反・スレッド数超過・`stages != 3`・ゼロ次元の各構成が全て
    /// `Err(CudaError::InvalidKernelConfig)` になることを検査する。
    #[test]
    fn render_mma_f16_rejects_invalid_configs() {
        let base = MmaKernelConfig::default();

        let cases: [(&str, MmaKernelConfig); 9] = [
            (
                "bm not multiple of MMA_WARP_M",
                MmaKernelConfig { bm: 17, ..base },
            ),
            (
                // PR #643 codex-review P1 再指摘への対応: bm/bn は
                // `MMA_WARP_M`/`MMA_WARP_N`（`MMA_M`/`MMA_N` ではない）の
                // 倍数を要求するようになったため、`MMA_M`/`MMA_N` の倍数
                // だが `MMA_WARP_M`/`MMA_WARP_N` の倍数ではない構成
                // （旧: bm=16 は MMA_M(16) の倍数だが MMA_WARP_M(32) の
                // 倍数ではない）を独立ケースとして検査する。
                // `warps_m`/`warps_n` が 0 になる境界値（bm=16 は
                // MMA_WARP_M(32) 未満のため warps_m=0）で block_dim.x=0
                // の無効な起動設定を防ぐ検査が効くことを確認する。
                "bn not multiple of MMA_WARP_N",
                MmaKernelConfig { bn: 8, ..base },
            ),
            (
                // PR #643 codex-review P2 指摘への対応: 旧ケース
                // （bm=128・bn=128・bk=32）は warps_m(8)*warps_n(16)*32=4096
                // threads となり、smem 予算検査より前のスレッド数上限
                // （1024）で拒否されてしまい SMEM の fail-closed 分岐を
                // 検査できていなかった。bm/bn は `MMA_WARP_M`/`MMA_WARP_N`
                // の倍数（PR #643 P1 再指摘）が前提のため
                // bm=32（warps_m=1）・bn=496（bn/MMA_WARP_N(16)=31。
                // threads=1*31*32=992。上限 1024 内で拒否されない）×
                // bk=32 なら smem_bytes=(32*32+32*496)*2*3=101376 > 49152
                // のみが拒否理由になる（thread count は上限内で通過）。
                "smem budget exceeded",
                MmaKernelConfig {
                    bm: 32,
                    bn: 496,
                    bk: 32,
                    ..base
                },
            ),
            (
                // PR #643 Bugbot 指摘への対応: bk=16 だと bk/MMA_K=1 で
                // 上位の `MMA_K_STEPS_PER_STAGE >= 2` 検査（本ファイル
                // §511-527）に先に拒否されてしまい、このケースが検証
                // したいスレッド数上限（1024）の fail-closed 分岐を
                // 実際には通過できていなかった（「bk / MMA_K below
                // MMA_K_STEPS_PER_STAGE」ケースと拒否理由が重複していた）。
                // bk=32（MMA_BK・bk/MMA_K=2 で同検査を通過）に変更し、
                // bm=512・bn=512 由来の threads=16*32*32=16384 が
                // k_steps 検査より後段のスレッド数上限検査（本ファイル
                // §544）で拒否されることを専用に検査する。
                "thread count exceeds 1024",
                MmaKernelConfig {
                    bm: 512,
                    bn: 512,
                    bk: 32,
                    stages: 1,
                    ..base
                },
            ),
            ("stages != 3", MmaKernelConfig { stages: 2, ..base }),
            // PR #643 codex-review Medium 指摘への対応: bk=16 は
            // MMA_K(16) の倍数だが bk/MMA_K=1 で
            // MMA_K_STEPS_PER_STAGE(>=2) 不変条件（本ファイル冒頭 const
            // アサーション §297-303）を満たさない。cp.async ソフトウェア
            // パイプラインが 1 ステップでは成立しないため fail-closed で
            // 拒否されなければならない（`bk % MMA_K == 0` 検査だけでは
            // 見逃されていた構成）。
            (
                "bk / MMA_K below MMA_K_STEPS_PER_STAGE (bk=MMA_K)",
                MmaKernelConfig { bk: MMA_K, ..base },
            ),
            (
                "static dim zero",
                MmaKernelConfig {
                    dim_m: DimSpec::Static(0),
                    ..base
                },
            ),
            // PR #643 codex-review P0 指摘への対応（再指摘）: dim_k=Static(7)
            // は 8 の倍数でないため cp.async 16 バイト転送のアドレス整列
            // 契約を破る。fail-closed に拒否されなければならない。
            (
                "static dim_k not a multiple of 8",
                MmaKernelConfig {
                    dim_k: DimSpec::Static(7),
                    ..base
                },
            ),
            // 同上・dim_n=Static(9) のケース（B の行ストライドへ直接畳み
            // 込まれる）。
            (
                "static dim_n not a multiple of 8",
                MmaKernelConfig {
                    dim_n: DimSpec::Static(9),
                    ..base
                },
            ),
        ];

        for (label, cfg) in cases {
            let result = render_mma_f16(&cfg);
            match &result {
                Err(CudaError::InvalidKernelConfig { detail }) => {
                    // PR #643 codex-review P2 指摘への対応: 拒否されたこと
                    // だけでなく、"smem budget exceeded" ケースが実際に
                    // SMEM 予算超過分岐（スレッド数上限分岐ではなく）で
                    // 拒否されたことを detail 文字列で確認する。
                    if label == "smem budget exceeded" {
                        assert!(
                            detail.contains("shared memory"),
                            "{label} は SMEM 予算超過として拒否されるべきです（実際の detail: {detail}）"
                        );
                    }
                    // PR #643 Bugbot 指摘への対応: "smem budget exceeded"
                    // 同様に、このケースが実際にスレッド数上限分岐
                    // （k_steps_per_stage 分岐ではなく）で拒否されたことを
                    // detail 文字列で確認する。
                    if label == "thread count exceeds 1024" {
                        assert!(
                            detail.contains("thread count") && detail.contains("1024"),
                            "{label} はスレッド数上限超過として拒否されるべきです（実際の detail: {detail}）"
                        );
                    }
                }
                other => panic!(
                    "{label} は InvalidKernelConfig で拒否されるべきです: config={cfg:?}, result={other:?}"
                ),
            }
        }
    }

    /// PR #643 codex-review P0 指摘への対応（再指摘）の受理側検査:
    /// `dim_k`/`dim_n` が 8 の倍数の `Static` 値であれば許容され、`dim_m`
    /// は 8 の倍数制約の対象外（cp.async のアドレス計算に使われるのは
    /// `DIM_K`/`DIM_N` のみで `DIM_M` は境界クランプにのみ使われるため）
    /// であることを検査する（`validate_mma_kernel_config` ドキュメンテー
    /// ションコメント参照）。
    #[test]
    fn render_mma_f16_accepts_static_dims_aligned_to_eight_and_exempts_dim_m() {
        let base = MmaKernelConfig::default();

        assert!(
            render_mma_f16(&MmaKernelConfig {
                dim_k: DimSpec::Static(4096),
                dim_n: DimSpec::Static(4096),
                ..base
            })
            .is_ok()
        );

        // dim_m は 8 の倍数でない値（5）でも拒否されない。
        assert!(
            render_mma_f16(&MmaKernelConfig {
                dim_m: DimSpec::Static(5),
                ..base
            })
            .is_ok()
        );
    }

    /// [`DimSpec::matches_launch_dim`] が `Dynamic` を常に許容し、`Static`
    /// は実引数と完全一致する場合のみ許容することを検査する（PR #643
    /// codex-review P1 指摘への対応）。
    #[test]
    fn dim_spec_matches_launch_dim_accepts_dynamic_and_exact_static_match() {
        assert!(DimSpec::Dynamic.matches_launch_dim(0).is_ok());
        assert!(DimSpec::Dynamic.matches_launch_dim(4096).is_ok());
        assert!(DimSpec::Static(4096).matches_launch_dim(4096).is_ok());
    }

    /// `DimSpec::Static(value)` は実引数 `value` と食い違う場合
    /// `InvalidKernelConfig` で fail-closed に拒否されることを検査する
    /// （静的値を実バッファ境界と誤認して境界外アクセスへ繋がる REQ-8
    /// 違反を防ぐ契約）。
    #[test]
    fn dim_spec_matches_launch_dim_rejects_mismatched_static() {
        let result = DimSpec::Static(4096).matches_launch_dim(16);
        assert!(
            matches!(result, Err(CudaError::InvalidKernelConfig { .. })),
            "static dim と実 shape の不一致は InvalidKernelConfig で拒否されるべきです: {result:?}"
        );
    }

    /// [`MmaKernelConfig::validate_launch_shape`] が dim_m/dim_n/dim_k の
    /// いずれか 1 つでも実 shape と食い違えば拒否することを検査する
    /// （codex-review 指摘の具体例: `dim_k=Static(4096)` の関数を
    /// 実際は K=16 の入力で起動しようとするケース）。
    #[test]
    fn mma_kernel_config_validate_launch_shape_rejects_k_mismatch() {
        let cfg = MmaKernelConfig {
            dim_m: DimSpec::Dynamic,
            dim_n: DimSpec::Dynamic,
            dim_k: DimSpec::Static(4096),
            ..MmaKernelConfig::default()
        };

        assert!(cfg.validate_launch_shape(128, 128, 4096).is_ok());

        let result = cfg.validate_launch_shape(128, 128, 16);
        assert!(
            matches!(result, Err(CudaError::InvalidKernelConfig { .. })),
            "dim_k=Static(4096) の関数を実際は K=16 で起動しようとする場合は拒否されるべきです: {result:?}"
        );
    }

    /// [`MmaKernelConfig::launch_config`] が `bm`/`bn` を単位に
    /// `div_ceil` でグリッドを構築し、`shared_mem_bytes` が常に 0
    /// （静的共有メモリのみの契約）であることを検査する（PR #643
    /// codex-review P0 再指摘への対応: `CompiledMmaKernel::launch_f16` が
    /// 本メソッドの戻り値を内部起動にのみ使う設計の土台）。
    #[test]
    fn mma_kernel_config_launch_config_grid_dim_covers_m_and_n_via_div_ceil() {
        let cfg = MmaKernelConfig {
            bm: 32,
            bn: 64,
            ..MmaKernelConfig::default()
        };

        // m=65（bm=32 の 2 タイル分 +1 端数）・n=64（bn=64 のちょうど 1
        // タイル分）。
        let launch_config = cfg.launch_config(65, 64);
        assert_eq!(launch_config.grid_dim, (1, 3, 1));
        // #493 の warp あたり 2x2 レジスタブロッキング後は、1 warp が担当する
        // C タイル実寸が `MMA_WARP_M x MMA_WARP_N`（`MMA_M x MMA_N` ではない）
        // ため、warp 数の導出式も `MMA_WARP_M`/`MMA_WARP_N` 基準に揃える
        // （`launch_config` 本体のコメント参照）。
        assert_eq!(
            launch_config.block_dim,
            ((cfg.bm / MMA_WARP_M) * (cfg.bn / MMA_WARP_N) * 32, 1, 1)
        );
        assert_eq!(launch_config.shared_mem_bytes, 0);
    }

    /// 受け入れ基準 2（PTX/SASS ダンプによるコンパイル時展開の実確認）は
    /// CI・本環境に NVRTC／実機がないため通常 CI では実行しない
    /// （`kernels_mma.rs` 冒頭コメント「検証状態」・`tests/gemm_mma.rs` の
    /// 環境適応方式を踏襲）。NVRTC が使える環境では、既定＋特化 render 済み
    /// ソースが `nvrtc::compile_ptx` を通ることを確認する（実測記録は
    /// #531/#534/#539 へ引き継ぐ。実装計画 10 節）。
    #[test]
    #[ignore = "requires NVRTC (libnvrtc); run manually on a CUDA-enabled host"]
    fn mma_f16_sources_compile_with_nvrtc_when_available() {
        use crate::nvrtc::compile_ptx;

        // 実機の compute capability に依存しないよう sm_80（本モジュールの
        // 最小要求。`gemm_mma.rs::MIN_COMPUTE_CAPABILITY_MAJOR`）を使う。
        let arch = "compute_80";

        let default_ptx = match compile_ptx(mma_f16_source(), arch) {
            Ok(ptx) => ptx,
            Err(CudaError::NvrtcUnavailable { .. }) => return,
            Err(e) => panic!("既定構成カーネルソースの NVRTC コンパイルに失敗しました: {e}"),
        };
        // 受け入れ基準 2 の本体: `#pragma unroll` を付与した `kstep` ループ
        // （`MMA_K_STEPS_PER_STAGE` 回展開・kernels_mma.rs 冒頭 §設計）が
        // NVRTC のコンパイル時展開でループ制御なしに `mma.sync` 命令列へ
        // 落ちていることを PTX テキストで確認する（compile 成功のみでは
        // ループが残っていても検出できないため出現数を数える）。
        let default_mma_count = default_ptx.to_src().matches("mma.sync.aligned").count();
        assert!(
            default_mma_count >= MMA_K_STEPS_PER_STAGE as usize,
            "既定構成 PTX の mma.sync.aligned 出現数（{default_mma_count}）が \
             MMA_K_STEPS_PER_STAGE（{MMA_K_STEPS_PER_STAGE}）未満です \
             （#pragma unroll によるコンパイル時展開の証跡が見つかりません）"
        );

        let specialized_cfg = MmaKernelConfig {
            bm: 64,
            bn: 64,
            bk: 32,
            stages: 3,
            dim_m: DimSpec::Static(4096),
            dim_n: DimSpec::Static(4096),
            dim_k: DimSpec::Static(4096),
            dtype: MmaDtype::F16,
        };
        let specialized = render_mma_f16(&specialized_cfg).expect("有効な構成が拒否されました");
        // 本テストは NVRTC の構文検証のみが目的で `CudaFunction`（実機の
        // `CudaModule` が必要）を作らないため、テスト専用アクセサ
        // `source()`（`#[cfg(test)]`）でソース文字列へ直接アクセスする
        // （`Self::compile` 経由の契約は `launch_f16` を使う実機依存の
        // 別テストが必要になるため、本テストのスコープ外）。
        let specialized_ptx = compile_ptx(specialized.source(), arch)
            .expect("特化構成カーネルソースの NVRTC コンパイルに失敗しました");
        let specialized_expected_steps = specialized_cfg.bk / MMA_K;
        let specialized_mma_count = specialized_ptx.to_src().matches("mma.sync.aligned").count();
        assert!(
            specialized_mma_count >= specialized_expected_steps as usize,
            "特化構成 PTX の mma.sync.aligned 出現数（{specialized_mma_count}）が \
             bk/MMA_K（{specialized_expected_steps}）未満です \
             （shape/タイル特化構成でもコンパイル時展開が維持されることの証跡が見つかりません）"
        );
    }

    /// [`validate_mma_k_tile_bound`] が通常サイズの `k`/`bk` を受理する
    /// ことを確認する（回帰防止の基本ケース）。
    #[test]
    fn validate_mma_k_tile_bound_accepts_ordinary_k() {
        assert!(validate_mma_k_tile_bound(4096, MMA_BK).is_ok());
        assert!(validate_mma_k_tile_bound(0, MMA_BK).is_ok());
    }

    /// codex-review 指摘（PR #643 再レビュー）の再現ケース: 既定より大きい
    /// `bk`（16 の倍数だが `MMA_BK` とは異なる非既定値）と、最終タイルの
    /// `s * bk + col0` が `i32::MAX` を超える `k` の組合せを `InvalidShape`
    /// として fail-closed に拒否することを検証する。
    #[test]
    fn validate_mma_k_tile_bound_rejects_i32_overflow_for_non_default_bk() {
        let bk: u32 = 48; // MMA_K(16) の倍数・8 の倍数だが既定 MMA_BK(32) とは異なる
        // k = i32::MAX + 1 のとき ceil(k/bk)*bk - 1 は i32::MAX を超える。
        let k = i32::MAX as u32; // u32 範囲内で確実に超過させるため i32::MAX を使う
        // ceil(i32::MAX / 48) * 48 - 1 を手計算すると i32::MAX を超えることを
        // 事前に確認済み（i32::MAX=2_147_483_647 は 48 の倍数でないため
        // 切り上げ後の最大インデックスが i32::MAX を上回る）。
        let tile = bk as u64;
        let expected_max_index = (k as u64).div_ceil(tile) * tile - 1;
        assert!(
            expected_max_index > i32::MAX as u64,
            "テスト前提が崩れています: expected_max_index={expected_max_index} は i32::MAX 以下です"
        );

        let result = validate_mma_k_tile_bound(k, bk);
        assert!(
            matches!(result, Err(CudaError::InvalidShape { .. })),
            "i32 オーバーフローが起こりうる k/bk の組合せが拒否されませんでした: {result:?}"
        );
    }

    /// `k == 0` は算術自体が発生しない no-op 形状のため、`bk` の値に
    /// 関わらず常に受理されることを確認する（境界条件）。
    #[test]
    fn validate_mma_k_tile_bound_accepts_zero_k_regardless_of_bk() {
        assert!(validate_mma_k_tile_bound(0, u32::MAX).is_ok());
    }

    /// #493 受け入れ基準（回帰防止）: warp あたり 2x2 レジスタブロッキング
    /// 構造（アキュムレータ配列・A/B フラグメント配列・mi/nj 2 重ループの
    /// mma.sync 発行）がソース文字列から失われていないことをロックする。
    /// イシュー #516 でテンプレート展開へ移行したため、`MMA_F16` 定数では
    /// なく `mma_f16_source()`（既定 config の render 結果）を対象にする
    /// （`mma_tile_constants_match_kernel_source_defines` と同じ方針）。
    /// フラグメント配列の宣言 needle は #495（ldmatrix 先読みダブルバッファ）
    /// で 2 面バッファ化した宣言形（`a_frag[2][...]`/`b_frag[2][...]`）へ
    /// 更新済み（本ファイル冒頭コメント「ldmatrix 先読みダブルバッファ」
    /// 参照）。
    #[test]
    fn mma_f16_source_uses_2x2_register_blocking_structure() {
        let src = mma_f16_source();
        for needle in [
            "float d[WARP_TILES_M][WARP_TILES_N][4] = {};",
            "unsigned a_frag[2][WARP_TILES_M][4];",
            "unsigned b_frag[2][WARP_TILES_N][2];",
            "for (int mi = 0; mi < WARP_TILES_M; ++mi) {",
            "for (int nj = 0; nj < WARP_TILES_N; ++nj) {",
        ] {
            assert!(
                src.contains(needle),
                "mma_f16_source() に #493 の 2x2 レジスタブロッキング構造 `{needle}` が見つかりません"
            );
        }
    }

    /// #495 受け入れ基準（回帰防止）: A/B フラグメントの 2 面バッファ化
    /// 宣言・warp プロローグ（kstep=0 の先読みロードが kstep ループの
    /// 手前にある位置関係）・kstep+1 段の先読みガード（`kstep + 1 < BK /
    /// MMA_K`。範囲外 kstep の SMEM 読み出しを発行しないための境界検査）・
    /// kstep ループ自体の `#pragma unroll`（`cur`/`nxt` をコンパイル時
    /// 定数へ畳み込むために必須。`LDSM_A_FRAG`/`LDSM_B_FRAG` マクロ直前の
    /// コメント参照）がソース文字列から失われていないことをロックする
    /// （本ファイル冒頭コメント「ldmatrix 先読みダブルバッファ」参照）。
    ///
    /// イシュー #516 でテンプレート展開へ移行したため、`MMA_F16` 定数では
    /// なく `mma_f16_source()`（既定 config の render 結果）を対象にする
    /// （`mma_f16_source_uses_2x2_register_blocking_structure` と同じ方針）。
    #[test]
    fn mma_f16_source_uses_ldmatrix_double_buffer_structure() {
        let src = mma_f16_source();
        for needle in [
            "unsigned a_frag[2][WARP_TILES_M][4];",
            "unsigned b_frag[2][WARP_TILES_N][2];",
            "#define LDSM_A_FRAG(buf, stage, kstep, mi)",
            "#define LDSM_B_FRAG(buf, stage, kstep, nj)",
            "LDSM_A_FRAG(0, compute_stage, 0, mi);",
            "LDSM_B_FRAG(0, compute_stage, 0, nj);",
            "if (kstep + 1 < BK / MMA_K) {",
            "LDSM_A_FRAG(nxt, compute_stage, kstep + 1, mi);",
            "LDSM_B_FRAG(nxt, compute_stage, kstep + 1, nj);",
        ] {
            assert!(
                src.contains(needle),
                "mma_f16_source() に #495 のダブルバッファ構造 `{needle}` が見つかりません"
            );
        }

        // warp プロローグ（kstep=0 のロード）は kstep ループの手前に
        // 位置しなければならない（`mma_f16_source_uses_fixed_immediate_wait_with_loop_exit_drain`
        // と同じ `find` インデックス比較方式）。
        let prologue_pos = src
            .find("LDSM_A_FRAG(0, compute_stage, 0, mi);")
            .expect("mma_f16_source() に warp プロローグの LDSM_A_FRAG(0, ...) が見つかりません");
        let kstep_loop_pos = src
            .find("for (int kstep = 0; kstep < BK / MMA_K; ++kstep) {")
            .expect("mma_f16_source() に kstep ループが見つかりません");
        assert!(
            prologue_pos < kstep_loop_pos,
            "mma_f16_source() の warp プロローグ（kstep=0 先読み）が kstep ループより \
             後ろにあります（プロローグは kstep ループの手前で発行される必要がある）"
        );

        // 先読みガードは kstep ループの内側（プロローグより後ろ）にある
        // こと。
        let guard_pos = src
            .find("if (kstep + 1 < BK / MMA_K) {")
            .expect("mma_f16_source() に先読みガードが見つかりません");
        assert!(
            guard_pos > kstep_loop_pos,
            "mma_f16_source() の先読みガード（kstep + 1 < BK / MMA_K）が kstep ループより \
             前にあります（kstep ループ内で発行される必要がある）"
        );

        // kstep ループ直前に `#pragma unroll` があること（`cur`/`nxt` の
        // コンパイル時定数畳み込みに必須。省略は local memory 溢れに
        // よる性能後退へ直結するため、cosmetic な pragma 除去として
        // 見逃さないよう隣接した 1 文字列として検査する）。
        assert!(
            src.contains(
                "#pragma unroll\n        for (int kstep = 0; kstep < BK / MMA_K; ++kstep) {"
            ),
            "mma_f16_source() の kstep ループ直前に #pragma unroll が見つかりません \
             （cur/nxt のコンパイル時定数畳み込みに必須）"
        );
    }

    /// #496 受け入れ基準（回帰防止）: cp.async issue interleaving のグループ
    /// 版マクロ（`LOAD_A_STAGE_GROUP`/`LOAD_B_STAGE_GROUP`）・添字空間分割
    /// の `#define`（`K_GROUPS`/`A_CHUNKS`/`B_CHUNKS`/`A_GROUP_CHUNKS`/
    /// `B_GROUP_CHUNKS`）がソース文字列から失われていないこと、グループ
    /// 発行が kstep ループの内側（ldmatrix 先読みガードより後ろ・mma.sync
    /// 発行より前）にあること、`cp.async.commit_group` がループ末尾・
    /// 無条件のまま動いていないこと（#492 不変条件の維持）をロックする
    /// （本ファイル冒頭コメント「cp.async issue interleaving」参照）。
    ///
    /// イシュー #516 でテンプレート展開へ移行したため、`MMA_F16` 定数では
    /// なく `mma_f16_source()`（既定 config の render 結果）を対象にする
    /// （`mma_f16_source_uses_2x2_register_blocking_structure` と同じ方針）。
    /// `contains` ではなく `matches().count() == 1` で検査するのは、main
    /// 追従マージが `#define` を二重に持ち込んでいないか（テンプレート側の
    /// パラメータ化定義と main 側の旧リテラル定義が両方残る等）を機械検出
    /// するため（PR #643 マージ・イシュー #516 のレビュー指摘対応）。
    #[test]
    fn mma_f16_source_interleaves_cp_async_issue_into_kstep_loop() {
        let src = mma_f16_source();
        for needle in [
            "#define K_GROUPS (BK / MMA_K)",
            "#define A_CHUNKS ((BM * BK) / 8)",
            "#define B_CHUNKS ((BK * BN) / 8)",
            "#define A_GROUP_CHUNKS ((A_CHUNKS + K_GROUPS - 1) / K_GROUPS)",
            "#define B_GROUP_CHUNKS ((B_CHUNKS + K_GROUPS - 1) / K_GROUPS)",
            "#define LOAD_A_STAGE_GROUP(stage, k0, g)",
            "#define LOAD_B_STAGE_GROUP(stage, k0, g)",
            "LOAD_A_STAGE_GROUP(load_stage, next_tile * BK, kstep);",
            "LOAD_B_STAGE_GROUP(load_stage, next_tile * BK, kstep);",
        ] {
            assert_eq!(
                src.matches(needle).count(),
                1,
                "mma_f16_source() に #496 の cp.async issue interleaving 構造 `{needle}` が \
                 ちょうど 1 回出現しません（main 追従マージによる定義の重複・欠落の疑い）"
            );
        }

        // `BK` は `K_GROUPS` 等の派生 `#define` より前に定義されている
        // こと（プリプロセッサのマクロ展開順序契約。マージで定義順序が
        // 入れ替わると NVRTC コンパイル不能になる回帰を機械検出する）。
        let bk_define_pos = src
            .find("#define BK")
            .expect("mma_f16_source() に #define BK が見つかりません");
        let k_groups_define_pos = src
            .find("#define K_GROUPS")
            .expect("mma_f16_source() に #define K_GROUPS が見つかりません");
        assert!(
            bk_define_pos < k_groups_define_pos,
            "mma_f16_source() の #define BK が #define K_GROUPS より後ろにあります \
             （K_GROUPS は BK に依存するため BK が先に定義されている必要がある）"
        );

        // グループ発行（分散発行サイト）は kstep ループの内側にあること
        // （kstep ループ開始より後ろ）。
        let kstep_loop_pos = src
            .find("for (int kstep = 0; kstep < BK / MMA_K; ++kstep) {")
            .expect("mma_f16_source() に kstep ループが見つかりません");
        let group_issue_pos = src
            .find("LOAD_A_STAGE_GROUP(load_stage, next_tile * BK, kstep);")
            .expect(
                "mma_f16_source() に分散発行サイト（LOAD_A_STAGE_GROUP 呼び出し）が見つかりません",
            );
        assert!(
            group_issue_pos > kstep_loop_pos,
            "mma_f16_source() の cp.async 分散発行サイトが kstep ループより前にあります \
             （kstep ループ内で発行される必要がある）"
        );

        // グループ発行は ldmatrix 先読みガード（kstep+1 段の先読み）より
        // 後ろ・mma.sync 発行（アセンブリ文字列リテラル）より前にあること
        // （ldmatrix 先読みの直後・Tensor Core 演算の直前という配置。
        // 本ファイル冒頭コメント「cp.async issue interleaving」参照）。
        let prefetch_guard_pos = src
            .find("if (kstep + 1 < BK / MMA_K) {")
            .expect("mma_f16_source() に先読みガードが見つかりません");
        let mma_sync_pos = src
            .find("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
            .expect("mma_f16_source() に mma.sync 発行が見つかりません");
        assert!(
            group_issue_pos > prefetch_guard_pos && group_issue_pos < mma_sync_pos,
            "mma_f16_source() の cp.async 分散発行サイトが「ldmatrix 先読みガードの後・\
             mma.sync 発行の前」という配置になっていません"
        );

        // ループ末尾の commit_group は無条件のまま（#492 不変条件）:
        // 直前に旧来の一括ロード呼び出し
        // `LOAD_A_STAGE(load_stage, next_tile * BK);` が存在しないこと
        // （分割前はここにあったが #496 で kstep ループ内へ移設済み）。
        assert!(
            !src.contains("LOAD_A_STAGE(load_stage, next_tile * BK);"),
            "mma_f16_source() にループ末尾の旧・一括ロード呼び出しが残っています \
             （#496 で kstep ループ内の分散発行へ置き換わっているはず）"
        );
    }

    /// #493 受け入れ基準: `mma.sync` 命令の発行箇所（アセンブリ文字列
    /// リテラル）が mi/nj 2 重ループ内の 1 箇所のみであること（kstep
    /// ループ内で 4 回実行はされるが、ソース上の発行箇所自体は 1 箇所に
    /// 集約されている＝レジスタブロッキングがコピペではなくループ化で
    /// 実装されていることを検査する）。
    #[test]
    fn mma_f16_source_issues_mma_sync_from_single_loop_site() {
        let src = mma_f16_source();
        let count = src
            .matches("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
            .count();
        assert_eq!(
            count, 1,
            "mma_f16_source() 中の mma.sync 発行箇所（アセンブリ文字列リテラル）が \
             1 箇所ではありません（mi/nj 2 重ループへ集約されているはず）"
        );
    }

    /// #493 受け入れ基準: エピローグの guarded store が mi/nj 2 重ループの
    /// 内側にあり、4 条件の guarded 式自体は単一タイル構成から変わって
    /// いないことをロックする（`mma_f16_source_retains_req8_boundary_guards`
    /// の REQ-8 needle と合わせて、2x2 化後も境界チェックが希釈されて
    /// いないことを検査する）。イシュー #516 のテンプレート展開に伴い
    /// エピローグの境界比較は `m`/`n` ではなく `DIM_M`/`DIM_N` マクロ経由
    /// になっている（本ファイル `render_mma_f16_unchecked` 参照）。
    #[test]
    fn mma_f16_source_epilogue_store_is_inside_warp_tile_loop() {
        let src = mma_f16_source();
        let loop_pos = src
            .find("for (int mi = 0; mi < WARP_TILES_M; ++mi) {\n#pragma unroll\n        for (int nj = 0; nj < WARP_TILES_N; ++nj) {\n            int r0 = row0_warp")
            .expect("mma_f16_source() にエピローグの mi/nj 2 重ループが見つかりません");
        // #805: エピローグ主 store は `__half2` ペア store（下記 needle）へ
        // 置き換わっているため、本テストは新 needle で検査する。
        let store_pos = src
            .find("__floats2half2_rn(d[mi][nj][0], d[mi][nj][1]);")
            .expect("mma_f16_source() にエピローグの guarded store（d[mi][nj]）が見つかりません");
        assert!(
            store_pos > loop_pos,
            "mma_f16_source() のエピローグ guarded store が mi/nj ループの外側にあります"
        );
    }

    /// #501 形状横断回帰テストの前提固定: `crates/backend-cuda/tests/
    /// cpu_cuda_mma_parity.rs::mma_f16_matches_reference_across_shapes` の
    /// 形状表（タイル倍数・タイル±端数の分類コメント）は
    /// `MMA_BM=64`/`MMA_BN=128`/`MMA_BK=32`（#494 時点の値）を前提に
    /// 選定されている。本テストはこの前提値を機械的にロックし、将来の
    /// ブロックタイル変更（`docs/perf/cuda-gemm-mma-block-tile.md` の
    /// 再候補選定等）で `MMA_BM`/`MMA_BN`/`MMA_BK` が変わった際に、
    /// 形状表の分類コメントが陳腐化する（例: 65 が「BM+1」でなくなる）
    /// ことをこのテストの failure で気付けるようにする。
    /// `mma_tile_constants_match_kernel_source_defines`（本ファイル）が
    /// CUDA ソース側 `#define` との一致を検査するのに対し、本テストは
    /// Rust 側定数の具体値そのものを固定する点で異なる。
    #[test]
    fn mma_tile_constants_pinned_for_shape_table_cross_reference() {
        assert_eq!(
            MMA_BM, 64,
            "MMA_BM が変更された場合は cpu_cuda_mma_parity.rs の形状表分類コメントを見直すこと"
        );
        assert_eq!(
            MMA_BN, 128,
            "MMA_BN が変更された場合は cpu_cuda_mma_parity.rs の形状表分類コメントを見直すこと"
        );
        assert_eq!(
            MMA_BK, 32,
            "MMA_BK が変更された場合は cpu_cuda_mma_parity.rs の形状表分類コメントを見直すこと"
        );
    }

    /// #499 受け入れ基準: `group_width < 2` を拒否する（本ファイル
    /// `mma_f16_source_with_swizzle` ドキュメンテーションコメント
    /// 「エラー契約」参照）。
    #[test]
    fn mma_f16_source_with_swizzle_rejects_group_width_below_two() {
        let err = mma_f16_source_with_swizzle(1).expect_err("group_width=1 must be rejected");
        assert!(matches!(err, crate::error::CudaError::InvalidShape { .. }));
        let err = mma_f16_source_with_swizzle(0).expect_err("group_width=0 must be rejected");
        assert!(matches!(err, crate::error::CudaError::InvalidShape { .. }));
    }

    /// #499 受け入れ基準: `group_width >= 2` では生成ソースに
    /// `#define SWIZZLE_GROUP <group_width>` と remap 断片が含まれ、かつ
    /// 元のアンカー（`blockIdx.y * BM`/`blockIdx.x * BN` 直書き）は
    /// 除去されていることを検査する（アンカー出現数 1 の pin を兼ねる。
    /// `mma_f16_source_with_swizzle` 内部の `assert_eq!` が既に検査する
    /// 前提だが、ここでは戻り値側から独立に確認する）。
    #[test]
    fn mma_f16_source_with_swizzle_contains_group_define_and_remap_fragment() {
        for group_width in [2u32, 8, 16] {
            let src = mma_f16_source_with_swizzle(group_width)
                .unwrap_or_else(|err| panic!("group_width={group_width}: {err}"));

            let expected_define = format!("#define SWIZZLE_GROUP {group_width}");
            assert!(
                src.contains(&expected_define),
                "group_width={group_width}: 生成ソースに `{expected_define}` が \
                 見つかりません"
            );
            for needle in [
                "long long linear_idx = (long long)blockIdx.y * gridDim.x + blockIdx.x;",
                "long long full_groups = num_m_blocks / SWIZZLE_GROUP;",
                "long long remainder = num_m_blocks % SWIZZLE_GROUP;",
                "int block_row0 = (int)(m_block * BM);",
                "int block_col0 = (int)(n_block * BN);",
            ] {
                assert!(
                    src.contains(needle),
                    "group_width={group_width}: 生成ソースに remap 断片 `{needle}` \
                     が見つかりません"
                );
            }
            assert!(
                !src.contains("int block_row0 = blockIdx.y * BM;"),
                "group_width={group_width}: 元のアンカー（blockIdx.y 直書き）が \
                 remap 後も残っています"
            );
        }
    }

    /// `mma_f16_source_with_swizzle` はアンカー置換のみを行い、
    /// `mma_f16_source()`（既定 config の render 結果）自体は不変で
    /// あることをロックする（本ファイル `mma_f16_source_with_swizzle`
    /// ドキュメンテーションコメント「**`mma_f16_source()` 自体は変更
    /// しない**」・実装計画 2 節の安全側判断の回帰防止）。イシュー #516
    /// でカーネルソースが `MMA_F16` 定数からテンプレート展開へ移行した
    /// ため、対象を `mma_f16_source()` に合わせてある（他の回帰テストと
    /// 同じ参照経路）。
    #[test]
    fn mma_f16_source_with_swizzle_does_not_mutate_mma_f16_source() {
        let before = mma_f16_source();
        let _ = mma_f16_source_with_swizzle(8).expect("group_width=8 must be accepted");
        assert_eq!(
            mma_f16_source(),
            before,
            "mma_f16_source_with_swizzle 呼び出し後に mma_f16_source() が変化しています"
        );
        assert!(
            mma_f16_source().contains("int block_row0 = blockIdx.y * BM;"),
            "mma_f16_source() の元のアンカー行が失われています（本番カーネルは無変更のはず）"
        );
    }

    /// #803 受け入れ基準: 既定値
    /// `(MMA_WARP_TILES_M, MMA_WARP_TILES_N, None)` は `mma_f16_source()` と
    /// バイト一致する（`mma_f16_source_with_warp_tiles` ドキュメンテーション
    /// コメント「既定値」節参照。parity 非後退契約への無影響を機械確認する）。
    #[test]
    fn mma_f16_source_with_warp_tiles_default_matches_mma_f16_source() {
        let src = mma_f16_source_with_warp_tiles(MMA_WARP_TILES_M, MMA_WARP_TILES_N, None)
            .expect("default warp tile config must be accepted");
        assert_eq!(
            src,
            mma_f16_source(),
            "既定値 (MMA_WARP_TILES_M, MMA_WARP_TILES_N, None) は mma_f16_source() と \
             バイト一致するはず"
        );
    }

    /// #803 実装計画 §3.1 の候補 A/B/C を含む複数の warp タイルで
    /// `#define WARP_TILES_M`/`WARP_TILES_N`/`WARPS_N` が期待どおり置換
    /// されることを検査する（候補表: 2x2 現行・2x4 案 A・4x2 案 B・4x4 案 C）。
    #[test]
    fn mma_f16_source_with_warp_tiles_replaces_defines_for_each_candidate() {
        // (warp_tiles_m, warp_tiles_n, expected_warps_n, expected_threads)
        let candidates = [
            (2u32, 2u32, 8u32, 512u32),
            (2, 4, 4, 256),
            (4, 2, 8, 256),
            (4, 4, 4, 128),
        ];
        for (wtm, wtn, expected_warps_n, expected_threads) in candidates {
            let src = mma_f16_source_with_warp_tiles(wtm, wtn, None)
                .unwrap_or_else(|err| panic!("wt={wtm}x{wtn}: {err}"));
            assert!(
                src.contains(&format!("#define WARP_TILES_M {wtm}\n")),
                "wt={wtm}x{wtn}: WARP_TILES_M define が期待値になっていません"
            );
            assert!(
                src.contains(&format!("#define WARP_TILES_N {wtn}\n")),
                "wt={wtm}x{wtn}: WARP_TILES_N define が期待値になっていません"
            );
            assert!(
                src.contains(&format!("#define WARPS_N {expected_warps_n}\n")),
                "wt={wtm}x{wtn}: WARPS_N define が期待値（{expected_warps_n}）になっていません"
            );
            // ブロックスレッド数と launch_bounds 導出値との整合を launch_bounds
            // 経路でも独立検査する（下記テスト参照）ため、ここでは導出値
            // そのもの（expected_threads）が候補表の設計どおりであることを
            // pin する。
            let with_lb = mma_f16_source_with_warp_tiles(wtm, wtn, Some(expected_threads))
                .unwrap_or_else(|err| panic!("wt={wtm}x{wtn} lb={expected_threads}: {err}"));
            assert!(
                with_lb.contains(&format!(
                    "extern \"C\" __global__ void __launch_bounds__({expected_threads}) gemm_mma_f16("
                )),
                "wt={wtm}x{wtn}: __launch_bounds__({expected_threads}) がシグネチャに \
                 見つかりません"
            );
        }
    }

    /// launch_bounds 指定時にシグネチャへ `__launch_bounds__` が 1 箇所だけ
    /// 入り、非指定時（`None`）は元のシグネチャのまま（`__launch_bounds__`
    /// を含まない）ことを検査する。
    #[test]
    fn mma_f16_source_with_warp_tiles_launch_bounds_none_omits_attribute() {
        let src = mma_f16_source_with_warp_tiles(2, 2, None)
            .expect("warp_tiles=2x2 with launch_bounds=None must be accepted");
        assert!(
            !src.contains("__launch_bounds__"),
            "launch_bounds=None のとき __launch_bounds__ を含んではならない"
        );
        assert_eq!(
            src.matches("extern \"C\" __global__ void gemm_mma_f16(")
                .count(),
            1,
            "元のシグネチャがちょうど 1 回だけ残っているはず"
        );
    }

    /// 不正入力（0・`MMA_BM`/`MMA_BN` を割り切らない値・`launch_bounds` 値
    /// 不一致）が `Err(CudaError::InvalidKernelConfig)` になることを検査
    /// する（`mma_f16_source_with_warp_tiles` ドキュメンテーションコメント
    /// 「エラー契約」節参照）。
    #[test]
    fn mma_f16_source_with_warp_tiles_rejects_invalid_inputs() {
        // warp_tiles_m/n == 0
        let err = mma_f16_source_with_warp_tiles(0, 2, None).expect_err("warp_tiles_m=0");
        assert!(matches!(
            err,
            crate::error::CudaError::InvalidKernelConfig { .. }
        ));
        let err = mma_f16_source_with_warp_tiles(2, 0, None).expect_err("warp_tiles_n=0");
        assert!(matches!(
            err,
            crate::error::CudaError::InvalidKernelConfig { .. }
        ));

        // MMA_BM=64/MMA_BN=128 を割り切らない値（MMA_M=16・MMA_N=8 なので
        // warp_tiles_m=3 → warp_m=48 は 64 を割り切らない）。
        let err = mma_f16_source_with_warp_tiles(3, 2, None)
            .expect_err("warp_tiles_m=3 does not evenly divide MMA_BM");
        assert!(matches!(
            err,
            crate::error::CudaError::InvalidKernelConfig { .. }
        ));

        // launch_bounds 値の不一致（wt=2x2 の正しい導出値は 512）。
        let err = mma_f16_source_with_warp_tiles(2, 2, Some(256))
            .expect_err("launch_bounds mismatch must be rejected");
        assert!(matches!(
            err,
            crate::error::CudaError::InvalidKernelConfig { .. }
        ));
    }

    #[test]
    fn mma_f16_source_with_warp_tiles_rejects_multiplication_overflow() {
        // `warp_tiles_m * MMA_M` / `warp_tiles_n * MMA_N` が `u32` を
        // オーバーフローする境界値。境界検査（0 除算・wrap 混入）より前に
        // `checked_mul` で fail-closed に拒否されることを確認する回帰
        // テスト（#822 codex-review 指摘）。デバッグビルドの panic・
        // リリースビルドの wrap 混入いずれも防ぐ。
        let err = mma_f16_source_with_warp_tiles(u32::MAX, 2, None)
            .expect_err("warp_tiles_m * MMA_M must not overflow silently");
        assert!(matches!(
            err,
            crate::error::CudaError::InvalidKernelConfig { .. }
        ));

        let err = mma_f16_source_with_warp_tiles(2, u32::MAX, None)
            .expect_err("warp_tiles_n * MMA_N must not overflow silently");
        assert!(matches!(
            err,
            crate::error::CudaError::InvalidKernelConfig { .. }
        ));
    }

    /// イシュー #804: 既定値 `(MMA_BM, MMA_BN, MMA_BK, MMA_STAGES,
    /// MMA_WARP_TILES_M, MMA_WARP_TILES_N, None, MMA_SHARED_MEM_BYTES)` を
    /// 渡すと `mma_f16_source()` とバイト一致することをロックする
    /// （`mma_f16_source_with_warp_tiles_default_matches_mma_f16_source` と
    /// 同じ回帰方針。全アンカー置換が「同じ値への置換」に潰れるため
    /// 既定引数では本番ソースへの影響がないことを機械的に担保する）。
    #[test]
    fn mma_f16_source_with_block_tile_default_matches_mma_f16_source() {
        let src = mma_f16_source_with_block_tile(
            MMA_BM,
            MMA_BN,
            MMA_BK,
            MMA_STAGES,
            MMA_WARP_TILES_M,
            MMA_WARP_TILES_N,
            None,
            MMA_SHARED_MEM_BYTES,
        )
        .expect("default block tile config must succeed");
        assert_eq!(src, mma_f16_source());
    }

    /// #804 実装計画 Step 1 の候補表（`docs/perf/
    /// cuda-gemm-mma-warp-tile-register-budget.md` 系の先例）にある
    /// 「ステージ増のみ」候補（64x128x32・S4・warp2x2）が静的 48KiB を
    /// 超え opt-in 予算内に収まること、`needs_dynamic_smem` 相当の判定が
    /// `extern __shared__` 変換を伴うことを確認する（`(64*40 + 32*136) *
    /// 2B * 4 stages = 55,296B`。48KiB=49,152B を超え 101,376B 以下）。
    #[test]
    fn mma_f16_source_with_block_tile_stage_increase_uses_dynamic_smem() {
        let src = mma_f16_source_with_block_tile(64, 128, 32, 4, 2, 2, None, 101_376)
            .expect("64x128x32 S4 must fit within the opt-in budget");
        assert!(
            src.contains("extern __shared__ __align__(16) unsigned char mma_dyn_smem[];"),
            "55,296B (> 48KiB static limit) candidate must use the extern __shared__ variant"
        );
        assert!(src.contains("#define STAGES 4\n"));
        assert!(
            !src.contains("__shared__ __align__(16) __half as_tile[STAGES][BM][A_PAD];"),
            "the static declaration must be fully replaced, not left alongside the dynamic one"
        );
    }

    /// 診断専用の強制動的 SMEM 変種（イシュー #855）: 静的予算以下（基準
    /// 構成 41,472B）でも `extern __shared__` 変換が適用されることを
    /// 検査する。`mma_f16_source_with_block_tile`（非強制）は同じ引数で
    /// 静的宣言のままであることも合わせて確認し、強制フラグが「静的
    /// 予算以下では無変換」という既定挙動を壊さないことを担保する。
    #[test]
    fn mma_f16_source_with_block_tile_forced_dynamic_smem_applies_transform_below_static_budget() {
        let forced =
            mma_f16_source_with_block_tile_forced_dynamic_smem(64, 128, 32, 3, 2, 2, None, 101_376)
                .expect("41,472B base config must fit within the opt-in budget");
        assert!(
            forced.contains("extern __shared__ __align__(16) unsigned char mma_dyn_smem[];"),
            "force_dynamic_smem=true must apply the extern __shared__ transform even when \
             smem_bytes (41,472B) is within the static 48KiB limit"
        );
        assert!(
            !forced.contains("__shared__ __align__(16) __half as_tile[STAGES][BM][A_PAD];"),
            "the static declaration must be fully replaced under force_dynamic_smem"
        );

        let unforced = mma_f16_source_with_block_tile(64, 128, 32, 3, 2, 2, None, 101_376)
            .expect("41,472B base config must fit within the opt-in budget");
        assert!(
            unforced.contains("__shared__ __align__(16) __half as_tile[STAGES][BM][A_PAD];"),
            "the non-forced entry point must keep the static declaration for a candidate at \
             or below the static budget (regression guard for the force flag threading)"
        );
    }

    /// [`render_mma_f16_block_tile_forced_dynamic_smem`] が返す
    /// descriptor の `uses_dynamic_smem` が強制的に真になり、
    /// `layout.needs_dynamic_smem()`（静的判定のまま）とは独立している
    /// ことを検査する（イシュー #855。`CompiledMmaF16BlockTileKernel::
    /// launch_f16` が `uses_dynamic_smem` を見て `shared_mem_bytes` を
    /// 決める契約の前提）。
    #[test]
    fn render_mma_f16_block_tile_forced_dynamic_smem_marks_uses_dynamic_smem() {
        let rendered =
            render_mma_f16_block_tile_forced_dynamic_smem(64, 128, 32, 3, 2, 2, None, 101_376)
                .expect("41,472B base config must fit within the opt-in budget");
        assert!(
            rendered.uses_dynamic_smem,
            "forced-dynamic rendering must set uses_dynamic_smem=true regardless of \
             layout.needs_dynamic_smem()"
        );
        assert!(
            !rendered.layout.needs_dynamic_smem(),
            "the underlying layout for the 41,472B base config must still report the static \
             judgement (force is a rendering-time override, not a layout property)"
        );
        assert!(
            rendered
                .source
                .contains("extern __shared__ __align__(16) unsigned char mma_dyn_smem[];")
        );

        let unforced = render_mma_f16_block_tile(64, 128, 32, 3, 2, 2, None, 101_376)
            .expect("41,472B base config must fit within the opt-in budget");
        assert!(
            !unforced.uses_dynamic_smem,
            "the non-forced entry point must derive uses_dynamic_smem from \
             layout.needs_dynamic_smem() (false for the static base config)"
        );
    }

    /// タイル拡大候補（128x256x32・S3・warp4x4）が opt-in 予算 101,376B 内
    /// （実測 81,408B）に収まり、`WARP_TILES_M`/`_N`/`WARPS_N`/`BM`/`BN` が
    /// 候補値へ置換されることを確認する。
    #[test]
    fn mma_f16_source_with_block_tile_expanded_tile_replaces_all_defines() {
        let src = mma_f16_source_with_block_tile(128, 256, 32, 3, 4, 4, Some(512), 101_376)
            .expect("128x256x32 S3 warp4x4 must fit within the opt-in budget");
        for needle in [
            "#define BM 128\n",
            "#define BN 256\n",
            "#define BK 32\n",
            "#define WARP_TILES_M 4\n",
            "#define WARP_TILES_N 4\n",
            "#define WARPS_N 8\n",
            "#define A_PAD 40\n",
            "#define B_PAD 264\n",
            "extern __shared__ __align__(16) unsigned char mma_dyn_smem[];",
            "__launch_bounds__(512)",
        ] {
            assert!(
                src.contains(needle),
                "missing {needle:?} in generated source"
            );
        }
    }

    /// 128x256x32・S4 は机上見積もり 108,544B で opt-in 上限 101,376B を
    /// 超える（#804 実装計画 Step 1 候補表の机上除外根拠）。実機到達前でも
    /// 判定できることをロックする。
    #[test]
    fn mma_f16_source_with_block_tile_rejects_over_optin_budget() {
        let err = mma_f16_source_with_block_tile(128, 256, 32, 4, 4, 4, None, 101_376)
            .expect_err("128x256x32 S4 (108,544B) must exceed the 101,376B opt-in budget");
        assert!(matches!(
            err,
            crate::error::CudaError::InvalidKernelConfig { .. }
        ));
    }

    /// `stages < 2` は cp.async ソフトウェアパイプラインの不変条件違反
    /// として拒否される（本関数ドキュメンテーションコメント参照）。
    #[test]
    fn mma_f16_source_with_block_tile_rejects_stages_below_two() {
        let err = mma_f16_source_with_block_tile(64, 128, 32, 1, 2, 2, None, 101_376)
            .expect_err("stages=1 must be rejected");
        assert!(matches!(
            err,
            crate::error::CudaError::InvalidKernelConfig { .. }
        ));
    }

    /// codex-review P2 是正（PR #831）: `cp.async.wait_group "n"(STAGES - 2)`
    /// の即値オペランドは 0〜7 の範囲でなければならない（PTX ISA
    /// `cp.async.wait_group` 命令仕様）ため、`stages` は 9 以下でなければ
    /// ならない。`stages=10`（`STAGES - 2 = 8`）は即値範囲超過として
    /// 拒否される必要がある。`optin_budget_bytes` は
    /// `u32::MAX` を渡し、拒否理由が共有メモリ予算超過ではなく段数上限
    /// であることを切り分ける。
    #[test]
    fn mma_f16_source_with_block_tile_rejects_stages_above_nine() {
        let err = mma_f16_source_with_block_tile(64, 128, 32, 10, 2, 2, None, u32::MAX)
            .expect_err("stages=10 must be rejected (cp.async.wait_group immediate overflow)");
        assert!(matches!(
            err,
            crate::error::CudaError::InvalidKernelConfig { .. }
        ));
    }

    /// `stages=9`（`STAGES - 2 = 7`）は即値範囲の境界値であり受理される
    /// ことを確認する（上限検証が過剰に厳しくないことの回帰防止）。
    #[test]
    fn mma_f16_source_with_block_tile_accepts_stages_at_nine() {
        mma_f16_source_with_block_tile(64, 128, 32, 9, 2, 2, None, u32::MAX)
            .expect("stages=9 is the maximum allowed by the cp.async.wait_group immediate range");
    }

    /// launch_bounds の値が導出スレッド数と食い違う場合は拒否される
    /// （`mma_f16_source_with_warp_tiles` と同じ fail-closed 契約）。
    #[test]
    fn mma_f16_source_with_block_tile_rejects_launch_bounds_mismatch() {
        let err = mma_f16_source_with_block_tile(64, 128, 32, 3, 2, 2, Some(256), 101_376)
            .expect_err("launch_bounds mismatch (actual threads=512) must be rejected");
        assert!(matches!(
            err,
            crate::error::CudaError::InvalidKernelConfig { .. }
        ));
    }

    /// 0 引数（`bm`/`bn`/`bk`/`stages`/`warp_tiles_m`/`warp_tiles_n`）は
    /// すべて fail-closed で拒否される。
    #[test]
    fn mma_f16_source_with_block_tile_rejects_zero_arguments() {
        for (bm, bn, bk, stages, wtm, wtn) in [
            (0, 128, 32, 3, 2, 2),
            (64, 0, 32, 3, 2, 2),
            (64, 128, 0, 3, 2, 2),
            (64, 128, 32, 0, 2, 2),
            (64, 128, 32, 3, 0, 2),
            (64, 128, 32, 3, 2, 0),
        ] {
            let err = mma_f16_source_with_block_tile(bm, bn, bk, stages, wtm, wtn, None, 101_376)
                .expect_err("zero-valued argument must be rejected");
            assert!(matches!(
                err,
                crate::error::CudaError::InvalidKernelConfig { .. }
            ));
        }
    }

    /// イシュー #840: `derive_mma_block_tile_layout` の既定値
    /// （`MMA_BM`/`MMA_BN`/`MMA_BK`/`MMA_STAGES`/`MMA_WARP_TILES_M`/`_N`）が
    /// 現行の本番タイル定数と一致することをロックする（`mma_f16_source_
    /// with_block_tile_default_matches_mma_f16_source` と同じ回帰方針。
    /// レイアウト導出ヘルパーの切り出し〈#840〉が既定構成の算出値を
    /// 変えていないことを機械的に担保する）。
    #[test]
    fn derive_mma_block_tile_layout_default_matches_production_constants() {
        let layout = derive_mma_block_tile_layout(
            MMA_BM,
            MMA_BN,
            MMA_BK,
            MMA_STAGES,
            MMA_WARP_TILES_M,
            MMA_WARP_TILES_N,
        )
        .expect("default block tile config must succeed");
        assert_eq!(layout.warps_m, MMA_WARPS_M);
        assert_eq!(layout.warps_n, MMA_WARPS_N);
        assert_eq!(layout.threads, MMA_WARPS_M * MMA_WARPS_N * 32);
        assert_eq!(layout.a_pad, MMA_A_PAD);
        assert_eq!(layout.b_pad, MMA_B_PAD);
        assert_eq!(layout.smem_bytes, MMA_SHARED_MEM_BYTES);
        assert!(
            !layout.needs_dynamic_smem(),
            "default (static-fit) config must not require the opt-in dynamic SMEM path"
        );
    }

    /// イシュー #840 実装計画候補表（`docs/perf/
    /// cuda-gemm-mma-block-tile-stages.md` §3.1）の 4 候補について、
    /// `derive_mma_block_tile_layout` が候補表記載の SMEM 実測要求量と
    /// 一致する値を導出することをロックする（実測値の出典は
    /// `examples/mma_ptx_dump.rs` 冒頭コメント「ブロックタイル拡大・
    /// ステージ数増候補」節・PR #831 と同一の机上見積もり）。
    #[test]
    fn derive_mma_block_tile_layout_matches_candidate_table_smem_bytes() {
        for (label, bm, bn, bk, stages, wtm, wtn, expected_smem_bytes) in [
            (
                "bt64x128_s4",
                64u32,
                128u32,
                32u32,
                4u32,
                2u32,
                2u32,
                55_296u32,
            ),
            ("bt128x128_s3_wt2x4", 128, 128, 32, 3, 2, 4, 56_832),
            ("bt128x256_s3_wt4x4", 128, 256, 32, 3, 4, 4, 81_408),
            ("bt128x256_s4", 128, 256, 32, 4, 4, 4, 108_544),
        ] {
            let layout = derive_mma_block_tile_layout(bm, bn, bk, stages, wtm, wtn)
                .unwrap_or_else(|e| panic!("{label}: layout derivation must succeed ({e:?})"));
            assert_eq!(
                layout.smem_bytes, expected_smem_bytes,
                "{label}: smem_bytes mismatch"
            );
            assert!(
                layout.needs_dynamic_smem(),
                "{label}: all four §3.1 candidates exceed the 48KiB static limit"
            );
        }
    }

    /// `render_mma_f16_block_tile` は `optin_budget_bytes` 超過候補を
    /// 非致命的な `CudaError::InvalidKernelConfig` として拒否する
    /// （`gemm_mma_block_tile_bench.rs` が「机上除外」としてログへ残し
    /// スイープを継続するための契約。`mma_f16_source_with_block_tile_
    /// rejects_over_optin_budget` と同じ 128x256x32 S4 候補〈机上見積もり
    /// 108,544B〉で GB10 実測上限 101,376B との比較を確認する）。
    #[test]
    fn render_mma_f16_block_tile_rejects_over_optin_budget() {
        let err = render_mma_f16_block_tile(128, 256, 32, 4, 4, 4, None, 101_376)
            .expect_err("108,544B candidate must exceed the 101,376B opt-in budget");
        assert!(matches!(
            err,
            crate::error::CudaError::InvalidKernelConfig { .. }
        ));
    }

    /// `render_mma_f16_block_tile` が返す descriptor のソースが
    /// `mma_f16_source_with_block_tile` 単体呼び出しと同一バイト列で
    /// あることを確認する（`RenderedMmaF16BlockTileKernel` が独自の
    /// ソース組み立て経路を持たず、公開済み関数へ委譲するだけであることの
    /// 回帰防止）。
    #[test]
    fn render_mma_f16_block_tile_source_matches_mma_f16_source_with_block_tile() {
        let rendered = render_mma_f16_block_tile(64, 128, 32, 4, 2, 2, None, 101_376)
            .expect("64x128x32 S4 must fit within the opt-in budget");
        let direct = mma_f16_source_with_block_tile(64, 128, 32, 4, 2, 2, None, 101_376)
            .expect("64x128x32 S4 must fit within the opt-in budget");
        assert_eq!(rendered.source(), direct);
    }
}
