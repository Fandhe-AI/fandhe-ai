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
//! 複数 mma タイル化・レジスタブロッキングは後続（知見は
//! `docs/cuda-tensor-core-knowledge.md`〈#65・TASK-11.1f〉に集約済み。
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

use std::sync::LazyLock;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;

use crate::error::CudaError;

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
        let warps_m = self.bm / MMA_M;
        let warps_n = self.bn / MMA_N;
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
    if !cfg.bm.is_multiple_of(MMA_M) {
        return Err(invalid(format!(
            "bm ({}) must be a multiple of MMA_M ({MMA_M})",
            cfg.bm
        )));
    }
    if !cfg.bn.is_multiple_of(MMA_N) {
        return Err(invalid(format!(
            "bn ({}) must be a multiple of MMA_N ({MMA_N})",
            cfg.bn
        )));
    }
    if !cfg.bk.is_multiple_of(MMA_K) {
        return Err(invalid(format!(
            "bk ({}) must be a multiple of MMA_K ({MMA_K})",
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

    let warps_m = cfg.bm / MMA_M;
    let warps_n = cfg.bn / MMA_N;
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
    let smem_bytes = cfg
        .bm
        .checked_mul(cfg.bk)
        .and_then(|a| cfg.bk.checked_mul(cfg.bn).and_then(|b| a.checked_add(b)))
        .and_then(|sum| sum.checked_mul(2))
        .and_then(|v| v.checked_mul(cfg.stages))
        .ok_or_else(|| invalid("shared memory byte count overflow".to_string()))?;
    if smem_bytes > 49_152 {
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
fn render_dim_define(macro_name: &str, param_name: &str, spec: DimSpec) -> String {
    match spec {
        DimSpec::Dynamic => format!("#define {macro_name} {param_name}"),
        DimSpec::Static(value) => format!("#define {macro_name} {value}"),
    }
}

/// 検証済み [`MmaKernelConfig`] からカーネルソース文字列を組み立てる
/// 内部関数（`validate_mma_kernel_config` を経ない infallible 経路。
/// 呼び出し元は必ず検証済みの `cfg` を渡すこと）。
fn render_mma_f16_unchecked(cfg: &MmaKernelConfig) -> String {
    let warps_n = cfg.bn / MMA_N;
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
         #define STAGES {stages}\n\
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
    #[allow(dead_code)]
    pub fn compile(
        &self,
        device: &crate::device::CudaDevice,
    ) -> Result<CompiledMmaKernel, CudaError> {
        let ptx = crate::nvrtc::compile_ptx(&self.source, device.arch())?;
        let func = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_mma_f16")?;
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
    #define LOAD_A_STAGE(stage, k0) \
        for (int idx = tid; idx < (BM * BK) / 8; idx += blockDim.x) { \
            int row = idx / (BK / 8); \
            int col0 = (idx % (BK / 8)) * 8; \
            int gr = block_row0 + row; \
            int gc = (k0) + col0; \
            int gr_c = gr < DIM_M ? gr : (DIM_M > 0 ? DIM_M - 1 : 0); \
            int gc_c = gc < DIM_K ? gc : (DIM_K > 0 ? ((DIM_K - 1) / 8) * 8 : 0); \
            int valid = (gr < DIM_M && gc < DIM_K) ? 16 : 0; \
            mma_cp_async16(&as_tile[stage][row][col0], &a[(size_t)gr_c * DIM_K + gc_c], valid); \
        }

    #define LOAD_B_STAGE(stage, k0) \
        for (int idx = tid; idx < (BK * BN) / 8; idx += blockDim.x) { \
            int row = idx / (BN / 8); \
            int col0 = (idx % (BN / 8)) * 8; \
            int gr = (k0) + row; \
            int gc = block_col0 + col0; \
            int gr_c = gr < DIM_K ? gr : (DIM_K > 0 ? DIM_K - 1 : 0); \
            int gc_c = gc < DIM_N ? gc : (DIM_N > 0 ? ((DIM_N - 1) / 8) * 8 : 0); \
            int valid = (gr < DIM_K && gc < DIM_N) ? 16 : 0; \
            mma_cp_async16(&bs_tile[stage][row][col0], &b[(size_t)gr_c * DIM_N + gc_c], valid); \
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

        // 受け入れ基準 2（コンパイル時展開）: BK/MMA_K は #define 定数の
        // ため NVRTC がコンパイル時に反復回数を確定でき、`#pragma unroll`
        // でループ展開する（wmma_opt 系は既に付与済み。実装計画 4.3 節）。
        #pragma unroll
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

    if (r0 < DIM_M && c0 < DIM_N) c[(size_t)r0 * DIM_N + c0] = __float2half(d0);
    if (r0 < DIM_M && c1 < DIM_N) c[(size_t)r0 * DIM_N + c1] = __float2half(d1);
    if (r1 < DIM_M && c0 < DIM_N) c[(size_t)r1 * DIM_N + c0] = __float2half(d2);
    if (r1 < DIM_M && c1 < DIM_N) c[(size_t)r1 * DIM_N + c1] = __float2half(d3);
}
"#;

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
            ("STAGES", MMA_STAGES),
            ("WARPS_N", MMA_WARPS_N),
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
        let src = mma_f16_source();
        assert!(
            src.contains("if (t == num_k_tiles - 1)") && src.contains("cp.async.wait_group 0;"),
            "mma_f16_source() に最終 K タイルの cp.async drain 分岐（wait_group 0）が見つかりません"
        );
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
            "#define WARPS_N 8", // 64 / MMA_N(8)
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

    /// フェイルクローズド検証（実装計画 7 節・4.2 節）: SMEM 予算超過・
    /// 倍数違反・スレッド数超過・`stages != 3`・ゼロ次元の各構成が全て
    /// `Err(CudaError::InvalidKernelConfig)` になることを検査する。
    #[test]
    fn render_mma_f16_rejects_invalid_configs() {
        let base = MmaKernelConfig::default();

        let cases: [(&str, MmaKernelConfig); 7] = [
            (
                "bm not multiple of MMA_M",
                MmaKernelConfig { bm: 17, ..base },
            ),
            (
                // PR #643 codex-review P2 指摘への対応: 旧ケース
                // （bm=128・bn=128・bk=32）は warps_m(8)*warps_n(16)*32=4096
                // threads となり、smem 予算検査より前のスレッド数上限
                // （1024）で拒否されてしまい SMEM の fail-closed 分岐を
                // 検査できていなかった。bm=16・bn=256（bm/MMA_M=1・
                // bn/MMA_N=32・threads=1*32*32=1024。ちょうど上限内で
                // 拒否されない）× bk=32 なら
                // smem_bytes=(16*32+32*256)*2*3=52224 > 49152 のみが
                // 拒否理由になる（thread count は境界の 1024 で通過）。
                "smem budget exceeded",
                MmaKernelConfig {
                    bm: 16,
                    bn: 256,
                    bk: 32,
                    ..base
                },
            ),
            (
                "thread count exceeds 1024",
                MmaKernelConfig {
                    bm: 512,
                    bn: 512,
                    bk: 16,
                    stages: 1,
                    ..base
                },
            ),
            ("stages != 3", MmaKernelConfig { stages: 2, ..base }),
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
        assert_eq!(
            launch_config.block_dim,
            ((cfg.bm / MMA_M) * (cfg.bn / MMA_N) * 32, 1, 1)
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
}
