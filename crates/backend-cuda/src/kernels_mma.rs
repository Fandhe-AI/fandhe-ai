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
//! XOR swizzle は引き続き不採用（下記「XOR swizzle」節参照）。
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
//! `docs/perf/cuda-gemm-mma-ldmatrix-double-buffer.md` を参照。
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
//! out-of-scope-tracking.md に従い記録）。
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

/// A タイル（`as_tile[STAGES][BM][MMA_A_PAD]`）の行幅（パディング後。
/// #498「共有メモリのバンクコンフリクト対策」）。パディングなしの
/// `MMA_BK=32`（f16 2B/要素で行ストライド 64B = 16 バンク）は 2 冪の
/// ため、`ldmatrix.x4`（A フラグメント。本ファイル冒頭コメント「命令
/// 選定」）が読む 8 行の開始バンクが全て同一位相へ収束し 4-way バンク
/// コンフリクトが理論上発生しうる。`+8` 要素（16B）を加えると行ストライド
/// 80B = 20 バンクとなり、8 行の開始バンクは `0,20,8,28,16,4,24,12` と
/// 全て相異なる（`gcd(20,32)=4` だが 8 行分の巡回で 32 バンクを完全被覆。
/// 本ファイル冒頭コメント「バンクコンフリクト対策」節・
/// `docs/perf/cuda-gemm-mma-bank-conflict.md` §2 参照）。パディング幅を
/// 8 要素単位に限定する理由は `cp.async` の 16B（f16 8 要素）転送粒度・
/// 整列要件（本ファイル冒頭コメント「整列制約」）を崩さないため（f32 opt
/// 経路の `kernels_wmma_opt.rs::WMMA_TF32_OPT_A_PAD` は `+4` 要素=8B だが、
/// f32 は元々 4B/要素で `cp.async` 粒度単位が異なるため同じ +4 要素は
/// f16 では 8B にしかならず 16B 整列が崩れる）。
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
pub const MMA_STATIC_SMEM_LIMIT_BYTES: u32 = 49_152;

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
#define BM 64
#define BN 128
#define BK 32
#define WARPS_N 8
#define WARP_TILES_M 2
#define WARP_TILES_N 2
#define STAGES 3
// #498: 共有メモリのバンクコンフリクト対策パディング（本ファイル冒頭
// コメント「バンクコンフリクト対策」・Rust 側 MMA_A_PAD/MMA_B_PAD 定数
// 直下のドキュメンテーションコメント参照）。索引式（LOAD_A/B_STAGE・
// ldmatrix アドレス計算）は配列次元経由でこのパディングを自動的に
// 反映するため、パディング領域そのものへの明示的な書き込み・読み出しは
// 発生しない。
#define A_PAD 40
#define B_PAD 136

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
    // 低減する。本ファイル冒頭コメント「バンクコンフリクト対策」参照）。
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
            int gr_c = gr < m ? gr : (m > 0 ? m - 1 : 0); \
            int gc_c = gc < k ? gc : (k > 0 ? ((k - 1) / 8) * 8 : 0); \
            int valid = (gr < m && gc < k) ? 16 : 0; \
            mma_cp_async16(&as_tile[stage][row][col0], &a[(size_t)gr_c * k + gc_c], valid); \
        }

    #define LOAD_B_STAGE_GROUP(stage, k0, g) \
        for (int idx = (g) * B_GROUP_CHUNKS + tid; \
             idx < B_CHUNKS && idx < ((g) + 1) * B_GROUP_CHUNKS; \
             idx += blockDim.x) { \
            int row = idx / (BN / 8); \
            int col0 = (idx % (BN / 8)) * 8; \
            int gr = (k0) + row; \
            int gc = block_col0 + col0; \
            int gr_c = gr < k ? gr : (k > 0 ? k - 1 : 0); \
            int gc_c = gc < n ? gc : (n > 0 ? ((n - 1) / 8) * 8 : 0); \
            int valid = (gr < k && gc < n) ? 16 : 0; \
            mma_cp_async16(&bs_tile[stage][row][col0], &b[(size_t)gr_c * n + gc_c], valid); \
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
            ("A_PAD", MMA_A_PAD),
            ("B_PAD", MMA_B_PAD),
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
    /// フラグメント配列の宣言 needle は #495（ldmatrix 先読みダブルバッファ）
    /// で 2 面バッファ化した宣言形（`a_frag[2][...]`/`b_frag[2][...]`）へ
    /// 更新済み（本ファイル冒頭コメント「ldmatrix 先読みダブルバッファ」
    /// 参照）。
    #[test]
    fn mma_f16_source_uses_2x2_register_blocking_structure() {
        for needle in [
            "float d[WARP_TILES_M][WARP_TILES_N][4] = {};",
            "unsigned a_frag[2][WARP_TILES_M][4];",
            "unsigned b_frag[2][WARP_TILES_N][2];",
            "for (int mi = 0; mi < WARP_TILES_M; ++mi) {",
            "for (int nj = 0; nj < WARP_TILES_N; ++nj) {",
        ] {
            assert!(
                MMA_F16.contains(needle),
                "MMA_F16 に #493 の 2x2 レジスタブロッキング構造 `{needle}` が見つかりません"
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
    #[test]
    fn mma_f16_source_uses_ldmatrix_double_buffer_structure() {
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
                MMA_F16.contains(needle),
                "MMA_F16 に #495 のダブルバッファ構造 `{needle}` が見つかりません"
            );
        }

        // warp プロローグ（kstep=0 のロード）は kstep ループの手前に
        // 位置しなければならない（`mma_f16_source_uses_fixed_immediate_wait_with_loop_exit_drain`
        // と同じ `find` インデックス比較方式）。
        let prologue_pos = MMA_F16
            .find("LDSM_A_FRAG(0, compute_stage, 0, mi);")
            .expect("MMA_F16 に warp プロローグの LDSM_A_FRAG(0, ...) が見つかりません");
        let kstep_loop_pos = MMA_F16
            .find("for (int kstep = 0; kstep < BK / MMA_K; ++kstep) {")
            .expect("MMA_F16 に kstep ループが見つかりません");
        assert!(
            prologue_pos < kstep_loop_pos,
            "MMA_F16 の warp プロローグ（kstep=0 先読み）が kstep ループより \
             後ろにあります（プロローグは kstep ループの手前で発行される必要がある）"
        );

        // 先読みガードは kstep ループの内側（プロローグより後ろ）にある
        // こと。
        let guard_pos = MMA_F16
            .find("if (kstep + 1 < BK / MMA_K) {")
            .expect("MMA_F16 に先読みガードが見つかりません");
        assert!(
            guard_pos > kstep_loop_pos,
            "MMA_F16 の先読みガード（kstep + 1 < BK / MMA_K）が kstep ループより \
             前にあります（kstep ループ内で発行される必要がある）"
        );

        // kstep ループ直前に `#pragma unroll` があること（`cur`/`nxt` の
        // コンパイル時定数畳み込みに必須。省略は local memory 溢れに
        // よる性能後退へ直結するため、cosmetic な pragma 除去として
        // 見逃さないよう隣接した 1 文字列として検査する）。
        assert!(
            MMA_F16.contains(
                "#pragma unroll\n        for (int kstep = 0; kstep < BK / MMA_K; ++kstep) {"
            ),
            "MMA_F16 の kstep ループ直前に #pragma unroll が見つかりません \
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
    #[test]
    fn mma_f16_source_interleaves_cp_async_issue_into_kstep_loop() {
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
            assert!(
                MMA_F16.contains(needle),
                "MMA_F16 に #496 の cp.async issue interleaving 構造 `{needle}` が見つかりません"
            );
        }

        // グループ発行（分散発行サイト）は kstep ループの内側にあること
        // （kstep ループ開始より後ろ）。
        let kstep_loop_pos = MMA_F16
            .find("for (int kstep = 0; kstep < BK / MMA_K; ++kstep) {")
            .expect("MMA_F16 に kstep ループが見つかりません");
        let group_issue_pos = MMA_F16
            .find("LOAD_A_STAGE_GROUP(load_stage, next_tile * BK, kstep);")
            .expect("MMA_F16 に分散発行サイト（LOAD_A_STAGE_GROUP 呼び出し）が見つかりません");
        assert!(
            group_issue_pos > kstep_loop_pos,
            "MMA_F16 の cp.async 分散発行サイトが kstep ループより前にあります \
             （kstep ループ内で発行される必要がある）"
        );

        // グループ発行は ldmatrix 先読みガード（kstep+1 段の先読み）より
        // 後ろ・mma.sync 発行（アセンブリ文字列リテラル）より前にあること
        // （ldmatrix 先読みの直後・Tensor Core 演算の直前という配置。
        // 本ファイル冒頭コメント「cp.async issue interleaving」参照）。
        let prefetch_guard_pos = MMA_F16
            .find("if (kstep + 1 < BK / MMA_K) {")
            .expect("MMA_F16 に先読みガードが見つかりません");
        let mma_sync_pos = MMA_F16
            .find("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
            .expect("MMA_F16 に mma.sync 発行が見つかりません");
        assert!(
            group_issue_pos > prefetch_guard_pos && group_issue_pos < mma_sync_pos,
            "MMA_F16 の cp.async 分散発行サイトが「ldmatrix 先読みガードの後・\
             mma.sync 発行の前」という配置になっていません"
        );

        // ループ末尾の commit_group は無条件のまま（#492 不変条件）:
        // 直前に旧来の一括ロード呼び出し
        // `LOAD_A_STAGE(load_stage, next_tile * BK);` が存在しないこと
        // （分割前はここにあったが #496 で kstep ループ内へ移設済み）。
        assert!(
            !MMA_F16.contains("LOAD_A_STAGE(load_stage, next_tile * BK);"),
            "MMA_F16 にループ末尾の旧・一括ロード呼び出しが残っています \
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
}
