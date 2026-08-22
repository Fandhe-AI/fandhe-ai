//! f16 `mma.sync` ブロックタイル・ステージ数候補の実機 A/B 計測バイナリ
//! （イシュー #840）。
//!
//! #804（PR #831）は `mma_f16` のブロックタイル拡大・ステージ数増候補を
//! 「候補ソース生成（[`diagnostics::mma_f16_source_with_block_tile`]）＋
//! PTX ダンプ（`examples/mma_ptx_dump.rs`）」までの**診断専用パス**として
//! 整備したが、実機・CUDA toolkit のいずれにも到達できず「候補を実際に
//! 起動して計測する経路」は未実装のまま残った
//! （`docs/perf/cuda-gemm-mma-block-tile-stages.md` §4 実測表が全欄
//! 「未実測」）。本バイナリはその欠落を埋め、
//! [`diagnostics::render_mma_f16_block_tile`]（本イシューで追加した A/B
//! ランナー型。`kernels_wmma_opt.rs::RenderedWmmaTf32StagedDynKernel`と
//! 同型設計）を使って候補を NVRTC コンパイル・起動し、比較基準行
//! （現行本番定数の [`CudaMmaGemm`]）と同条件で TFLOPS を計測する。
//!
//! **本番経路（`gemm_mma.rs::CudaMmaGemm`・`MMA_BM`/`MMA_BN`/`MMA_STAGES`
//! 等の本番定数）は一切変更しない**。採否判断・本番結線は後続イシュー
//! #842 のスコープ。
//!
//! ## 実行時観測（イシュー #855）
//!
//! #840 の GB10 実機 A/B で `bt64x128_s4`／`bt128x128_s3_wt2x4`
//! （いずれも `extern __shared__` 動的 SMEM 変換を通る候補）のみが CPU
//! `f32::mul_add` 参照との数値一致に fail し、#842 の机上調査ではアドレス
//! 計算・アライメントに欠陥を特定できなかった。イシュー #855 は
//! `CONTROL_CANDIDATES`（対照実験行）と `compute-sanitizer` による実機
//! 実行時観測でこれを切り分けた。
//!
//! **結論: 不一致の原因は `extern __shared__` 変換にもタイル候補定数にも
//! ない**。`CONTROL_CANDIDATES::debug_default_via_diagnostics_path`
//! （production とバイト一致のソース・静的 SMEM のまま、診断コンパイル・
//! 起動経路のみを経由）が production 自身（本ファイルが標準出力へ書く
//! `production_direct(no diagnostics path):` 行）と全く同一の座標・同一値
//! （`mismatch_count=2/266240`・`max_abs_diff=1.562e-2`・
//! `max_rel_err=6.818e-2`・`first_mismatch=(168, 2)`）で不一致を出すことを
//! 確認した。すなわち production の base カーネル自身が、本バイナリの
//! 正しさ検査データ（`CORRECTNESS_M=520`・`SEED=0xC0FFEE`）に対して
//! この不一致を既に持っており、#840/#842 はこれを「動的 SMEM 変換の
//! 欠陥」と誤って帰属していた。`compute-sanitizer --tool memcheck` は
//! 全候補（`extern __shared__` 変種を含む）でメモリ安全性エラーを検出
//! しなかった（起動不能な `bt128x256_s3_wt4x4` の
//! `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` を除く。これは #854 で既知の
//! レジスタ不足）。既存の `#[ignore]` テスト
//! `tests/cpu_cuda_mma_parity.rs::mma_f16_k4096_stress` も本パッチと無関係
//! に（`main` ブランチ単体でも）同系統の小規模不一致で fail することを
//! 確認しており、これは本イシューのスコープ外の別課題である（詳細・
//! 追跡方針は `docs/perf/cuda-gemm-mma-block-tile-stages.md` §9.4）。
//!
//! ## 計測対象候補（`docs/perf/cuda-gemm-mma-block-tile-stages.md` §3.1）
//!
//! | 識別子 | BM/BN/BK | STAGES | warp タイル | SMEM（机上見積もり） |
//! |--------|----------|--------|------------|----------------------|
//! | `mma_f16_base`（現行・比較基準） | 64/128/32 | 3 | 2x2 | 41,472B（静的） |
//! | `bt64x128_s4` | 64/128/32 | 4 | 2x2 | 55,296B |
//! | `bt128x128_s3_wt2x4` | 128/128/32 | 3 | 2x4 | 56,832B |
//! | `bt128x256_s3_wt4x4` | 128/256/32 | 3 | 4x4 | 81,408B |
//! | `bt128x256_s4` | 128/256/32 | 4 | 4x4 | 108,544B（GB10 実測 opt-in 上限
//! 超のため除外候補。固定除外ではなく実測上限との動的比較で判定する。
//! `mma_ptx_dump.rs` の同種コメント・PR #831 codex-review P1 是正と同じ
//! 方針） |
//! | `bt128x256_s3_wt4x4_lb512`（イシュー #854 で追加） | 128/256/32 | 3 |
//! 4x4 | 81,408B（`bt128x256_s3_wt4x4` と同一形状・`__launch_bounds__(512)`
//! のみが異なる） |
//!
//! 最初の 4 候補は threads/block=512（`launch_bounds` は付与しない。占有率
//! ヒント無しでのレジスタ割り当てを比較基準行と揃えるため）。
//! `bt128x256_s3_wt4x4_lb512` のみ `launch_bounds: Some(512)` を付与する
//! （イシュー #854。命名は `mma_ptx_dump.rs` の `_lb{v}` サフィックス規約に
//! 一致）。`bt128x256_s3_wt4x4`〈`launch_bounds` なし〉は 130
//! registers/thread × 512 threads = 66,560 > 65,536（GB10 per-SM レジスタ
//! 上限）で 1 ブロックも起動できないと実測済み（`docs/perf/
//! cuda-gemm-mma-block-tile-stages.md` §4・#840）。一方 `ptxas -v` 実測では
//! `__launch_bounds__(512)` 付き変種が 128 registers/thread（128×512=65,536
//! の境界値ちょうど）となり、机上では 1 block/SM で起動可能と見積もられる
//! が、#840 の A/B 計測は `launch_bounds` を付与しない構成のみを対象とした
//! ため未計測のまま #842 へ引き継がれた（同 doc §4「未計測」・§6 引き継ぎ
//! 事項）。**本イシュー（#854）はこの境界値変種の実起動可否を実測する**
//! （実起動不能なら #847 の不採用判断を確定、起動できれば性能比較へ進む）。
//! `MMA_BK=32` は全候補で不変。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p backend-cuda --example gemm_mma_block_tile_bench --release \
//!     --features internal-diagnostics
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載・cc<8.0（`mma.sync` の下限）・opt-in 予算
//! 未取得環境では、理由を表示して正常終了する
//! （`gemm_wmma_tf32_staged_stages_bench.rs` と同じ環境適応分岐）。
//! 候補ごとのコンパイル失敗・opt-in 予算超過（机上除外）・数値一致 fail・
//! 計測中の CUDA 起動失敗は理由付きで SKIP／FAIL／desk-excluded 表示し、
//! 残りの候補の計測は継続する（fail-closed だがスイープ全体は止めない
//! 設計。実装計画 §7「リスクと安全側の倒し方」）。
//!
//! 「5 回計測の中央値」（`.claude/rules/coding-rust.md`）は**本バイナリを
//! 5 回プロセス起動**し、候補×形状ごとに 5 run の出力（本バイナリ自体は
//! `bench_harness::protocol::run` の warmup 20/計測 20 の中央値を 1 run
//! として出力する）から中央値を取ることで満たす契約とする
//! （`gemm_wmma_tf32_staged_stages_bench.rs` と同じ「1 プロセス起動 = 1
//! run」設計。本バイナリ自体は 1 回の起動につき候補×形状ごとに 1 行の
//! CSV を出力するのみで、5 run 分の集計は呼び出し側〈実機セッションの
//! 記録手順。`docs/perf/cuda-gemm-mma-block-tile-stages.md` §4〉が担う）。
//!
//! 実測値・対現行比・数値一致結果は
//! `docs/perf/cuda-gemm-mma-block-tile-stages.md` §4 へ記録する。

use backend_cuda::diagnostics::{self, MmaBlockTileLayout};
use backend_cuda::{CudaDevice, CudaError, CudaMmaGemm};
use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use half::f16;

/// 決定的シード（`gemm_mma_bench.rs`・`gemm_mma_swizzle_bench.rs` と同一値）。
const SEED: u64 = 0xC0FFEE;

/// スイープ対象の計測形状（実装計画 §3 の 512/1024/2048/4096）。
const BENCH_SIZES: [usize; 4] = [512, 1024, 2048, 4096];

/// 正しさ検査用の小形状。M を全候補の `bm`（64/128）いずれの倍数でも
/// ないよう選び、エピローグ guarded store（REQ-8）の境界分岐を実際に
/// 踏ませる（実装計画 §3「非整列端を踏む小形状」）。N/K は `mma.sync`
/// 経路の整列制約（8 の倍数。`kernels_mma.rs` 冒頭コメント「整列制約」）
/// を満たす必要があるため崩さない。
const CORRECTNESS_M: u32 = 520;
const CORRECTNESS_N: u32 = 512;
const CORRECTNESS_K: u32 = 512;

/// 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」の要素単位
/// 判定を `backend_cpu::compare`（`RELATIVE_TOLERANCE`/
/// `ABSOLUTE_RESCUE_THRESHOLD`。`.claude/rules/coding-rust.md`「バックエンド
/// 構成」）と**同一式**で再現する（`docs/perf/
/// cuda-gemm-mma-tf32-ab.md`／`parity_baseline.rs` が引用する
/// `fail_count`/`max_abs_diff`/`max_rel_err` はいずれも `compare` 由来の
/// ため、初回不一致座標の特定にも同じ分母規約〈`max(|a|,|b|,1e-12)`〉を
/// 使い、集計値〈`ParityDiagnostics`〉との整合を保つ。#842 codex-review
/// 想定是正: 独自の許容誤差式を再実装すると `expected.abs()` のみを分母に
/// 使う旧実装との間で判定境界がわずかに乖離しうるため、既存の唯一の正
/// である `compare` の式へ委譲する）。
fn is_mismatch(actual: f32, expected: f32) -> bool {
    let diff = (actual as f64 - expected as f64).abs();
    let scale = (actual as f64)
        .abs()
        .max((expected as f64).abs())
        .max(1e-12);
    let rel = diff / scale;
    let pass =
        rel < backend_cpu::RELATIVE_TOLERANCE || diff < backend_cpu::ABSOLUTE_RESCUE_THRESHOLD;
    !pass
}

fn tflops(size: usize, secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / secs / 1e12
}

/// `bench_harness::Measurement`（所要時間の中央値/Q1/Q3）を TFLOPS へ
/// 変換する（`gemm_wmma_tf32_staged_stages_bench.rs::TflopsMeasurement` と
/// 同じ理由・同じ変換式: 時間の Q1〈速い側〉が TFLOPS の上限、Q3〈遅い側〉
/// が TFLOPS の下限になる）。
struct TflopsMeasurement {
    median: f64,
    q1: f64,
    q3: f64,
}

impl TflopsMeasurement {
    fn from_secs(size: usize, measurement: &bench_harness::Measurement) -> Self {
        Self {
            median: tflops(size, measurement.median_secs),
            q1: tflops(size, measurement.q3_secs),
            q3: tflops(size, measurement.q1_secs),
        }
    }
}

/// #840 実装計画候補表（本ファイル冒頭コメント参照）1 行分の静的定義。
///
/// `launch_bounds`（イシュー #854 で追加）: `None` は「占有率ヒント無し」
/// （既存 4 候補・比較基準行と揃える）、`Some(v)` はシグネチャへ
/// `__launch_bounds__(v)` を付与する（`diagnostics::render_mma_f16_
/// block_tile` の `launch_bounds: Option<u32>` 引数へそのまま渡す）。
///
/// `force_dynamic_smem`（イシュー #855 で追加）: 真の場合
/// `diagnostics::render_mma_f16_block_tile_forced_dynamic_smem` を使い、
/// 静的 48KiB 予算以下の候補でも `extern __shared__` 動的 SMEM 変換を
/// 強制適用する（対照実験専用。本ファイル冒頭コメント「実行時観測」
/// 節・`CONTROL_CANDIDATES` 参照）。既存候補はすべて `false`（本番と
/// 同じ「48KiB 超のみ動的化」の既定挙動）。
struct Candidate {
    label: &'static str,
    bm: u32,
    bn: u32,
    bk: u32,
    stages: u32,
    warp_tiles_m: u32,
    warp_tiles_n: u32,
    launch_bounds: Option<u32>,
    force_dynamic_smem: bool,
}

const CANDIDATES: [Candidate; 5] = [
    Candidate {
        label: "bt64x128_s4",
        bm: 64,
        bn: 128,
        bk: 32,
        stages: 4,
        warp_tiles_m: 2,
        warp_tiles_n: 2,
        launch_bounds: None,
        force_dynamic_smem: false,
    },
    Candidate {
        label: "bt128x128_s3_wt2x4",
        bm: 128,
        bn: 128,
        bk: 32,
        stages: 3,
        warp_tiles_m: 2,
        warp_tiles_n: 4,
        launch_bounds: None,
        force_dynamic_smem: false,
    },
    Candidate {
        label: "bt128x256_s3_wt4x4",
        bm: 128,
        bn: 256,
        bk: 32,
        stages: 3,
        warp_tiles_m: 4,
        warp_tiles_n: 4,
        launch_bounds: None,
        force_dynamic_smem: false,
    },
    Candidate {
        label: "bt128x256_s4",
        bm: 128,
        bn: 256,
        bk: 32,
        stages: 4,
        warp_tiles_m: 4,
        warp_tiles_n: 4,
        launch_bounds: None,
        force_dynamic_smem: false,
    },
    // イシュー #854: `bt128x256_s3_wt4x4`（`launch_bounds` なし）は
    // 130 registers/thread × 512 threads = 66,560 > 65,536（GB10
    // per-SM レジスタ上限）で起動不能と実測済み（`docs/perf/
    // cuda-gemm-mma-block-tile-stages.md` §4・#840）。`__launch_bounds__
    // (512)` を付与すると `ptxas -v` 実測で 128 registers/thread
    // （128×512=65,536 の境界値ちょうど）となり、机上では 1 block/SM で
    // 起動可能と見積もられるが実起動は未計測だった（同 doc §6）。本候補
    // でその実起動可否を実測する（本ファイル冒頭コメント参照）。
    Candidate {
        label: "bt128x256_s3_wt4x4_lb512",
        bm: 128,
        bn: 256,
        bk: 32,
        stages: 3,
        warp_tiles_m: 4,
        warp_tiles_n: 4,
        launch_bounds: Some(512),
        force_dynamic_smem: false,
    },
];

/// イシュー #855 対照実験行（本ファイル冒頭コメント「実行時観測（イシュー
/// #855）」節）。`CANDIDATES` とは独立の配列にし、既存 5 候補の意味
/// （#840/#842 実測の対象）を変えずに切り分け専用行を追加する。
///
/// **実機観測の結論（#855。本ファイル冒頭コメント参照）**: 以下 4 行の
/// うち `debug_default_via_diagnostics_path`（production とバイト一致
/// ソース・静的 SMEM のまま、診断コンパイル・起動経路のみを経由）が
/// production 自身と同じ座標（`(row=168, col=2)`・
/// `mismatch_count=2/266240`）で fail したことにより、**#840/#842 が
/// 「動的 SMEM 変換の欠陥」と推定した不一致は、実際には extern
/// __shared__ 変換にもタイル候補定数にも起因しない**ことが判明した。
/// 詳細は `docs/perf/cuda-gemm-mma-block-tile-stages.md` §9.3 を参照。
///
/// - `debug_default_via_diagnostics_path`: 現行本番定数（BM=64/BN=128/
///   BK=32/STAGES=3/warp2x2）を非強制（`mma_f16_source()` とバイト一致・
///   静的 `__shared__` のまま）で診断コンパイル・起動経路
///   （`RenderedMmaF16BlockTileKernel::compile`／
///   `CompiledMmaF16BlockTileKernel::launch_f16`）のみを経由して起動する。
///   production（`CudaMmaGemm`）との差はコンパイル・起動コードパスの
///   みであり、これが fail することで「診断ハーネスの compile/launch
///   コードパス自体のバグではなく、カーネル・参照値比較そのものに起因
///   する不一致」（実際には後述の通り production 自身も同じ不一致を
///   出す）と判明した。
/// - `mma_f16_base_dynsmem`: 現行本番定数を [`Candidate::
///   force_dynamic_smem`] で強制動的化したもの（動的 41,472B）。
///   `debug_default_via_diagnostics_path` と同一箇所・同一値で fail し、
///   動的 SMEM 変換の有無が結果に影響しないことを追加確認する。
/// - `bt64x64_s4_static`: `bt64x128_s4`（BM=64/BN=128/BK=32/STAGES=4。
///   動的 55,296B）から BN のみ 64 へ縮小し、STAGES=4 のパイプライン
///   段数を静的予算内（38,912B。`extern __shared__` 変換を経ない）で
///   単独検証する。これも同一箇所・同一値で fail し、STAGES=4 側にも
///   欠陥がないことを確認した。
/// - `bt128x64_s3_wt2x4_static`: `bt128x128_s3_wt2x4`（BM=128/BN=128/
///   BK=32/STAGES=3・warp2x4。動的 56,832B）から BN のみ 64 へ縮小し、
///   BM=128・warp2x4 のタイル写像を静的予算内（44,544B。`extern
///   __shared__` 変換を経ない）で単独検証する。同じく同一箇所・同一値で
///   fail し、BM=128/warp2x4 タイル写像側にも欠陥がないことを確認した。
///
/// いずれも `derive_mma_block_tile_layout`（`(bm*a_pad + bk*b_pad) * 2 *
/// stages`、`a_pad=bk+8`・`b_pad=bn+8`）から机上算出した値
/// （下記コメント）で `MMA_STATIC_SMEM_LIMIT_BYTES`＝49,152B 以下に収まる
/// ことを確認済み。
const CONTROL_CANDIDATES: [Candidate; 4] = [
    Candidate {
        label: "debug_default_via_diagnostics_path",
        bm: 64,
        bn: 128,
        bk: 32,
        stages: 3,
        warp_tiles_m: 2,
        warp_tiles_n: 2,
        launch_bounds: None,
        force_dynamic_smem: false,
    },
    Candidate {
        label: "mma_f16_base_dynsmem",
        bm: 64,
        bn: 128,
        bk: 32,
        stages: 3,
        warp_tiles_m: 2,
        warp_tiles_n: 2,
        launch_bounds: None,
        force_dynamic_smem: true,
    },
    Candidate {
        // (64*40 + 32*72) * 2 * 4 = 38,912B（静的）。
        label: "bt64x64_s4_static",
        bm: 64,
        bn: 64,
        bk: 32,
        stages: 4,
        warp_tiles_m: 2,
        warp_tiles_n: 2,
        launch_bounds: None,
        force_dynamic_smem: false,
    },
    Candidate {
        // (128*40 + 32*72) * 2 * 3 = 44,544B（静的）。
        label: "bt128x64_s3_wt2x4_static",
        bm: 128,
        bn: 64,
        bk: 32,
        stages: 3,
        warp_tiles_m: 2,
        warp_tiles_n: 4,
        launch_bounds: None,
        force_dynamic_smem: false,
    },
];

/// 比較基準行（現行本番定数の [`CudaMmaGemm`]。base カーネルの GPU 実行
/// のみを計測する。`gemm_mma_bench.rs::measure_mma_f16` と同じ計測方針:
/// H2D/D2H・出力確保は計測区間の外）。
fn measure_production(
    gemm: &CudaMmaGemm,
    size: usize,
    config: &MeasurementConfig,
) -> Result<TflopsMeasurement, CudaError> {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f16> = rng.fill_vec_f16(size * size);
    let b: Vec<f16> = rng.fill_vec_f16(size * size);

    let (a_dev, b_dev) = gemm.upload_f16(&a, &b)?;
    let mut c_dev = gemm.alloc_output_f16(size as u32, size as u32)?;

    // `bench_run` のクロージャは `FnMut()`（非 fallible）契約のため、計測中
    // （ウォームアップ／反復）の起動失敗はここで最初の `CudaError` を捕捉し
    // `Result` として呼び出し元へ返す（`gemm_wmma_tf32_staged_stages_
    // bench.rs::measure_dyn_staged` と同じ理由・同じ契約）。
    let mut first_err: Option<CudaError> = None;
    let measurement = bench_run(config, || {
        if first_err.is_some() {
            return;
        }
        if let Err(e) = gemm.launch_f16(
            &a_dev,
            &b_dev,
            &mut c_dev,
            size as u32,
            size as u32,
            size as u32,
        ) {
            first_err = Some(e);
        }
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    if let Some(e) = first_err {
        return Err(e);
    }
    Ok(TflopsMeasurement::from_secs(size, &measurement))
}

/// 候補カーネル（[`diagnostics::CompiledMmaF16BlockTileKernel`]）の GPU
/// 実行のみを計測する（[`measure_production`] と同じ計測方針・同じ
/// エラー捕捉契約）。
fn measure_candidate(
    compiled: &diagnostics::CompiledMmaF16BlockTileKernel,
    gemm: &CudaMmaGemm,
    device: &CudaDevice,
    size: usize,
    config: &MeasurementConfig,
) -> Result<TflopsMeasurement, CudaError> {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f16> = rng.fill_vec_f16(size * size);
    let b: Vec<f16> = rng.fill_vec_f16(size * size);

    // アップロード・出力バッファ確保は比較基準行と同じ `CudaMmaGemm`
    // ヘルパー（`upload_f16`/`alloc_output_f16`）を再利用する（候補
    // カーネルもバッファレイアウト・要素型は本番経路と同一の f16
    // 行優先ため、専用のアップロード経路を新設する必要がない）。
    let (a_dev, b_dev) = gemm.upload_f16(&a, &b)?;
    let mut c_dev = gemm.alloc_output_f16(size as u32, size as u32)?;

    let stream = device.stream();
    let mut first_err: Option<CudaError> = None;
    let measurement = bench_run(config, || {
        if first_err.is_some() {
            return;
        }
        if let Err(e) = compiled.launch_f16(
            stream,
            &a_dev,
            &b_dev,
            &mut c_dev,
            size as u32,
            size as u32,
            size as u32,
        ) {
            first_err = Some(e);
        }
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    if let Some(e) = first_err {
        return Err(e);
    }
    Ok(TflopsMeasurement::from_secs(size, &measurement))
}

/// 数値一致検査の診断出力（#842 引き継ぎ事項。`docs/perf/
/// cuda-gemm-mma-block-tile-stages.md` §6「まず `within_tolerance` 判定を
/// ミスマッチ件数・最大誤差付きで出力するよう拡張し、再現・切り分けを
/// 行う」を受けた拡張。#840 時点の `candidate_parity_ok` は bool のみを
/// 返しており、`bt64x128_s4`／`bt128x128_s3_wt2x4` の FAIL がどの座標・
/// どの規模の不一致かを追加ログなしには特定できなかった）。
///
/// `mismatch_count`/`max_abs_diff`/`max_rel_err` は [`backend_cpu::
/// CompareReport`]（`fail_count`/`max_abs_diff`/`max_rel_err`。全セル
/// 対象の集計）をそのまま転記する。`docs/perf/cuda-gemm-mma-tf32-ab.md`・
/// `tests/common/parity_baseline.rs` が引用する同名統計はいずれもこの
/// `CompareReport` 由来のため、本診断出力も同じ集計方式に揃えることで
/// 実測記録との比較可能性を保つ（独自の集計方式を再実装しない）。
struct ParityDiagnostics {
    mismatch_count: usize,
    max_abs_diff: f64,
    max_rel_err: f64,
    /// 最初に不一致となった要素の行優先フラットインデックス
    /// （`row = idx / CORRECTNESS_N`・`col = idx % CORRECTNESS_N`）。
    /// `is_mismatch`（`CompareReport` と同一の判定式）で `mismatch_count`
    /// と独立に再走査して求める（`CompareReport` 自体は座標を保持しない
    /// ため）。
    first_mismatch_index: Option<usize>,
}

impl ParityDiagnostics {
    fn is_pass(&self) -> bool {
        self.mismatch_count == 0
    }
}

/// 候補カーネルの数値一致を検査する（計測の前に必ず実施。fail 時は
/// 当該候補を計測から除外し、残候補の計測は継続する。実装計画「計測
/// 前へ数値一致検査」節）。CPU 参照実装は `backend_cpu::matmul_reference_
/// fma`（`f32::mul_add` FMA 契約。`tests/cpu_cuda_mma_parity.rs` と同一
/// 手順: f16→f32→参照 FMA→f16 丸め→f32 の経路で得た参照値と、カーネル
/// 出力（f16→f32）を統一複合判定で照合する）。判定・集計は
/// `backend_cpu::compare`（REQ-2 統一複合判定の唯一の正）へ委譲する。
fn candidate_parity_ok(
    compiled: &diagnostics::CompiledMmaF16BlockTileKernel,
    gemm: &CudaMmaGemm,
    device: &CudaDevice,
    a_f16: &[f16],
    b_f16: &[f16],
    expected_f32: &[f32],
) -> Result<ParityDiagnostics, CudaError> {
    let (a_dev, b_dev) = gemm.upload_f16(a_f16, b_f16)?;
    let mut c_dev = gemm.alloc_output_f16(CORRECTNESS_M, CORRECTNESS_N)?;
    compiled.launch_f16(
        device.stream(),
        &a_dev,
        &b_dev,
        &mut c_dev,
        CORRECTNESS_M,
        CORRECTNESS_N,
        CORRECTNESS_K,
    )?;
    let actual_f16 = gemm.download_f16(&c_dev)?;
    let actual_f32: Vec<f32> = actual_f16.iter().map(|x| x.to_f32()).collect();

    let report = backend_cpu::compare(&actual_f32, expected_f32).map_err(|e| {
        CudaError::InvalidKernelConfig {
            detail: format!("candidate_parity_ok: length mismatch in backend_cpu::compare: {e}"),
        }
    })?;
    let first_mismatch_index = actual_f32
        .iter()
        .zip(expected_f32.iter())
        .position(|(a, e)| is_mismatch(*a, *e));
    Ok(ParityDiagnostics {
        mismatch_count: report.fail_count,
        max_abs_diff: report.max_abs_diff,
        max_rel_err: report.max_rel_err,
        first_mismatch_index,
    })
}

/// 候補が opt-in 予算内かを実測レイアウトから判定する（固定除外を
/// 避け、接続中の実デバイスの opt-in 上限との動的比較で行う。
/// `mma_ptx_dump.rs` の `desk-excluded` 分岐・PR #831 codex-review P1
/// 是正と同じ方針）。
fn layout_or_print_excluded(
    candidate: &Candidate,
    optin_budget_bytes: u32,
) -> Option<MmaBlockTileLayout> {
    match diagnostics::mma_f16_block_tile_layout(
        candidate.bm,
        candidate.bn,
        candidate.bk,
        candidate.stages,
        candidate.warp_tiles_m,
        candidate.warp_tiles_n,
    ) {
        Ok(layout) if layout.smem_bytes > optin_budget_bytes => {
            println!(
                "desk-excluded: {} ({}x{}x{} S{}) requires {} bytes, exceeding opt-in budget \
                 ({} bytes)",
                candidate.label,
                candidate.bm,
                candidate.bn,
                candidate.bk,
                candidate.stages,
                layout.smem_bytes,
                optin_budget_bytes
            );
            None
        }
        Ok(layout) => Some(layout),
        Err(e) => {
            println!("{}: SKIP (layout derivation failed: {e})", candidate.label);
            None
        }
    }
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_mma_block_tile_bench: CUDA driver unavailable ({detail}); \
                 skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_mma_block_tile_bench: CudaDevice::new failed ({other}); skipping."
            );
            return;
        }
    };

    let optin_budget_bytes = match device.shared_memory_per_block_optin() {
        Some(v) => v,
        None => {
            println!(
                "backend-cuda gemm_mma_block_tile_bench: \
                 CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN unavailable; skipping."
            );
            return;
        }
    };
    println!("device: optin_budget_bytes={optin_budget_bytes}");

    let gemm = match CudaMmaGemm::new(&device) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "backend-cuda gemm_mma_block_tile_bench: CudaMmaGemm::new failed ({e}); \
                 nothing to measure. See docs/perf/cuda-gemm-mma-block-tile-stages.md."
            );
            return;
        }
    };

    // 正しさ検査用の CPU 参照値（統一複合判定。緩和しない。本ファイル
    // 冒頭コメント「候補カーネルの数値一致を検査する」参照）。
    let mut rng = Xorshift64Star::new(SEED);
    let a_ref: Vec<f16> = rng.fill_vec_f16((CORRECTNESS_M * CORRECTNESS_K) as usize);
    let b_ref: Vec<f16> = rng.fill_vec_f16((CORRECTNESS_K * CORRECTNESS_N) as usize);
    let a_ref_f32: Vec<f32> = a_ref.iter().map(|x| x.to_f32()).collect();
    let b_ref_f32: Vec<f32> = b_ref.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; (CORRECTNESS_M * CORRECTNESS_N) as usize];
    backend_cpu::matmul_reference_fma(
        &a_ref_f32,
        &b_ref_f32,
        &mut c_ref_f32,
        CORRECTNESS_M as usize,
        CORRECTNESS_N as usize,
        CORRECTNESS_K as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed bench input");
    let expected_f32: Vec<f32> = c_ref_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    // production 直接検査行（イシュー #855）: production 経路
    // （`CudaMmaGemm::launch_f16`。診断ハーネスの
    // `CompiledMmaF16BlockTileKernel` を一切経由しない）が同じ
    // CORRECTNESS_M/N/K・同じ a_ref/b_ref・同じ expected_f32 に対しても
    // 同一箇所で不一致を出すかを検査する行。実機観測の結果、production
    // 自身も `CONTROL_CANDIDATES::debug_default_via_diagnostics_path` と
    // 同一箇所・同一値（`mismatch_count=2/266240`・
    // `first_mismatch=(168, 2)`）で不一致を出すことを確認した（イシュー
    // #855。`docs/perf/cuda-gemm-mma-block-tile-stages.md` §9.3）。これにより
    // #840/#842 が観測した不一致は診断ハーネス（extern __shared__ 変換・
    // タイル候補定数・compile/launch コードパス）に起因せず、production の
    // base カーネル自体が本テストデータ（M=520 非整列端・SEED=0xC0FFEE）
    // に対して持つ既存の狭い数値差（`docs/perf/
    // cuda-gemm-mma-block-tile-stages.md` §9.4 参照。既存 `#[ignore]` テスト
    // `cpu_cuda_mma_parity.rs::mma_f16_k4096_stress` の失敗と同系統）に
    // 起因すると確定できた。以降の A/B 実行でも production 自身の実測値を
    // 常に記録できるよう、この検査は一時デバッグではなく本バイナリの
    // 標準出力の一部として残す。
    // `production_direct_fail_count` はブロック外（下記 fail-closed 分岐）
    // でも参照するため、ブロック式の値として持ち出す（codex-review 是正:
    // PR #862 review。以前は `report.fail_count` を println するだけで
    // ブロック外へ持ち出さず、直後の性能計測（`production_medians`
    // 構築・全候補の ratio 分母採用）が production 自身の parity 結果を
    // 一切見ずに進んでいた。本ファイル冒頭コメントおよび PR 本文が掲げる
    // 「parity ゲートが性能値採用に先立つ」契約・AGENTS.md の数値契約に
    // 反するため、production も候補〈`candidate_parity_ok` 呼び出し側の
    // fail-closed 分岐〉と同じ fail-closed ゲートを通す）。
    let production_direct_fail_count: usize = {
        let (a_dev, b_dev) = gemm
            .upload_f16(&a_ref, &b_ref)
            .expect("production direct-check upload must succeed");
        let mut c_dev = gemm
            .alloc_output_f16(CORRECTNESS_M, CORRECTNESS_N)
            .expect("production direct-check alloc must succeed");
        gemm.launch_f16(
            &a_dev,
            &b_dev,
            &mut c_dev,
            CORRECTNESS_M,
            CORRECTNESS_N,
            CORRECTNESS_K,
        )
        .expect("production direct-check launch must succeed");
        let actual_f16 = gemm
            .download_f16(&c_dev)
            .expect("production direct-check download must succeed");
        let actual_f32: Vec<f32> = actual_f16.iter().map(|x| x.to_f32()).collect();
        let report = backend_cpu::compare(&actual_f32, &expected_f32)
            .expect("production direct-check compare shape must match");
        let first_mismatch = actual_f32
            .iter()
            .zip(expected_f32.iter())
            .position(|(a, e)| is_mismatch(*a, *e));
        println!(
            "production_direct(no diagnostics path): mismatch_count={}/{}, \
             max_abs_diff={:.3e}, max_rel_err={:.3e}, first_mismatch={:?}",
            report.fail_count,
            (CORRECTNESS_M * CORRECTNESS_N) as usize,
            report.max_abs_diff,
            report.max_rel_err,
            first_mismatch.map(|idx| (idx / CORRECTNESS_N as usize, idx % CORRECTNESS_N as usize)),
        );
        report.fail_count
    };

    // 二段階ゲート（codex-review P1 是正・PR #862 review 追補。イシュー
    // #855）: production 自身が本テストデータに対して統一複合判定に
    // 不合格（`mismatch_count != 0`）の場合でも、**性能計測のみ**を
    // スキップし、候補（`CANDIDATES`・`CONTROL_CANDIDATES`）の
    // render/compile/parity 診断（`candidate_parity_ok`）までは実行する。
    // GB10 実機では `CORRECTNESS_M=520`・固定シードにより production の
    // parity が常に不合格になるため（`docs/perf/
    // cuda-gemm-mma-block-tile-stages.md` §9.3）、旧実装（不合格時に即
    // `return`）だと `CONTROL_CANDIDATES`（強制 dynamic SMEM・静的対照
    // 候補・diagnostics 経路）の compile/parity 検査自体が実機で恒常的に
    // 到達不能になっていた（PR #862 codex-review P2 指摘）。
    // `ratio_vs_production` 等の性能値は不正な基準実装との比率になり
    // 得るため採用禁止のまま維持し（本ファイル冒頭コメント・PR 本文の
    // 契約）、`skip_performance_measurement` で production・候補いずれの
    // 実測ループも perf 計測部分だけを SKIP させる。既知の狭い数値差
    // 自体は上記 `production_direct` ログ・
    // docs/perf/cuda-gemm-mma-block-tile-stages.md §9.3 に記録済み。
    let skip_performance_measurement = production_direct_fail_count != 0;
    if skip_performance_measurement {
        println!(
            "mma_f16_base(production): FAIL (parity mismatch vs CPU f32::mul_add reference; \
             mismatch_count={production_direct_fail_count}; skipping performance measurement \
             for production and all candidates — parity ゲートが性能値採用に先立つ契約のため、\
             production 自身が数値不一致の間は性能比較を行わない。ただし候補の \
             render/compile/parity 診断（CONTROL_CANDIDATES を含む）は打ち切らず継続する \
             （PR #862 codex-review P2 是正）。詳細は上記 production_direct ログ・\
             docs/perf/cuda-gemm-mma-block-tile-stages.md §9.3 を参照)"
        );
    }

    println!(
        "candidate,bm,bn,bk,stages,warp_tiles_m,warp_tiles_n,launch_bounds,threads,smem_bytes,\
         dynamic_smem,{}",
        BENCH_SIZES
            .iter()
            .map(|s| format!(
                "tflops_median_{s},tflops_q1_{s},tflops_q3_{s},ratio_vs_production_{s}"
            ))
            .collect::<Vec<_>>()
            .join(",")
    );

    // 比較基準行（現行本番定数）。候補と同じ CSV スキーマで先頭行として
    // 出力する。`bm`/`bn`/`bk`/`stages`/warp タイル・`threads`/
    // `smem_bytes`/`dynamic_smem` は候補行と同じ「単一の真実源」
    // （`diagnostics::mma_f16_block_tile_layout_production`
    // → `derive_mma_block_tile_layout`）から導出する（codex-review 是正:
    // 以前は `MMA_STAGES` 等をリテラルで再記述しており、定数変更時に
    // 追従しない不整合の余地があった）。
    let production_medians: std::collections::HashMap<usize, f64> = {
        let layout = diagnostics::mma_f16_block_tile_layout_production()
            .expect("production MMA_BM/BN/BK/STAGES/WARP_TILES must derive a valid layout");
        let (bm, bn, bk, stages, warp_tiles_m, warp_tiles_n, threads, smem_bytes, dynamic_smem) = (
            layout.bm,
            layout.bn,
            layout.bk,
            layout.stages,
            layout.warp_tiles_m,
            layout.warp_tiles_n,
            layout.threads,
            layout.smem_bytes,
            layout.needs_dynamic_smem(),
        );
        // 比較基準行（本番経路）は `launch_bounds` を付与しない構成のため
        // 候補行と同じ表記規約で "none" を記す（イシュー #854 で CSV へ
        // 追加した列。候補行は `Candidate.launch_bounds` を同じ表記で出力
        // する。下記候補行の該当箇所参照）。
        let mut row = format!(
            "mma_f16_base(production),{bm},{bn},{bk},{stages},{warp_tiles_m},{warp_tiles_n},none,\
             {threads},{smem_bytes},{dynamic_smem}"
        );
        // 比較基準行の実測値は `measure_production` を size ごとに 1 回だけ
        // 呼び、以下の `production_medians`（ratio 分母）にも同じ結果を
        // 使い回す（codex-review 是正: 以前は本ブロックと下の
        // `production_medians` 構築ループが `measure_production` を size
        // ごとに独立計測しており、直下コメント「単一のベースライン計測
        // 結果を全候補で共有する」の意図に反して base 行の
        // `tflops_median_*` と各候補の ratio 分母が別計測値になっていた。
        // GPU 計測は試行間でばらつくため、この不一致は base 行の
        // `ratio_vs_production=1.0000` が実際の分母と一致しない・
        // 計測時間が 2 倍になる、の 2 点の実害を生む）。
        let mut production_medians: std::collections::HashMap<usize, f64> =
            std::collections::HashMap::new();
        // `skip_performance_measurement` が真の場合（production 自身が
        // parity FAIL）は上記二段階ゲートの契約により性能計測を行わず
        // 全 size を n/a で埋める。`production_medians` は空のままとなり、
        // 後続の候補ループでも ratio 分母が見つからず ratio は n/a になる
        // （`ratio.filter(base != 0.0)` の分母探索が空 HashMap で必ず
        // `None` を返すため。#855）。
        for &size in &BENCH_SIZES {
            if skip_performance_measurement {
                row.push_str(",n/a,n/a,n/a,n/a");
                continue;
            }
            let config = MeasurementConfig::default();
            match measure_production(&gemm, size, &config) {
                Ok(m) => {
                    row.push_str(&format!(",{:.4},{:.4},{:.4},1.0000", m.median, m.q1, m.q3));
                    production_medians.insert(size, m.median);
                }
                Err(e) => {
                    println!(
                        "mma_f16_base size={size}: SKIP measurement (production launch failed: {e})"
                    );
                    row.push_str(",n/a,n/a,n/a,n/a");
                }
            }
        }
        println!("{row}");
        production_medians
    };

    // イシュー #855: 既存 5 候補（本番の「48KiB 超のみ動的化」既定挙動）
    // に続けて対照実験行（`CONTROL_CANDIDATES`）も同じループ・同じ CSV
    // スキーマで計測する。`.chain()` で 1 本のイテレータへ合成すること
    // で、計測手順（layout 導出 → render → compile → parity → 計測）を
    // 重複実装しない（本ファイル冒頭コメント「実行時観測」節参照）。
    for candidate in CANDIDATES.iter().chain(CONTROL_CANDIDATES.iter()) {
        let Some(layout) = layout_or_print_excluded(candidate, optin_budget_bytes) else {
            continue;
        };

        // `force_dynamic_smem` は対照実験専用（`CONTROL_CANDIDATES` の
        // `mma_f16_base_dynsmem` のみ真）。真の場合は静的予算以下でも
        // `extern __shared__` 変換を強制する診断専用エントリポイントへ
        // 分岐する（`diagnostics::render_mma_f16_block_tile_forced_
        // dynamic_smem` ドキュメンテーションコメント「目的」節参照）。
        let rendered = if candidate.force_dynamic_smem {
            diagnostics::render_mma_f16_block_tile_forced_dynamic_smem(
                candidate.bm,
                candidate.bn,
                candidate.bk,
                candidate.stages,
                candidate.warp_tiles_m,
                candidate.warp_tiles_n,
                candidate.launch_bounds,
                optin_budget_bytes,
            )
        } else {
            diagnostics::render_mma_f16_block_tile(
                candidate.bm,
                candidate.bn,
                candidate.bk,
                candidate.stages,
                candidate.warp_tiles_m,
                candidate.warp_tiles_n,
                candidate.launch_bounds,
                optin_budget_bytes,
            )
        };
        let rendered = match rendered {
            Ok(r) => r,
            Err(e) => {
                println!("{}: SKIP (render failed: {e})", candidate.label);
                continue;
            }
        };
        let compiled = match rendered.compile(&device) {
            Ok(c) => c,
            Err(e) => {
                println!(
                    "{}: SKIP (NVRTC compile / opt-in attribute failed: {e})",
                    candidate.label
                );
                continue;
            }
        };

        // 数値一致検査（計測より先に実施。実装計画「計測前へ数値一致
        // 検査」節）。
        match candidate_parity_ok(&compiled, &gemm, &device, &a_ref, &b_ref, &expected_f32) {
            Ok(diag) if diag.is_pass() => {}
            Ok(diag) => {
                let (row, col) = diag
                    .first_mismatch_index
                    .map(|idx| (idx / CORRECTNESS_N as usize, idx % CORRECTNESS_N as usize))
                    .expect("mismatch_count > 0 implies first_mismatch_index is Some");
                println!(
                    "{}: FAIL (parity mismatch vs CPU f32::mul_add reference; not measuring; \
                     mismatch_count={}/{}, max_abs_diff={:.3e}, max_rel_err={:.3e}, \
                     first_mismatch=(row={row}, col={col}))",
                    candidate.label,
                    diag.mismatch_count,
                    (CORRECTNESS_M * CORRECTNESS_N) as usize,
                    diag.max_abs_diff,
                    diag.max_rel_err,
                );
                continue;
            }
            Err(e) => {
                println!("{}: SKIP (parity launch failed: {e})", candidate.label);
                continue;
            }
        }

        // `launch_bounds` 列（イシュー #854 で追加）は `None` を "none"
        // （比較基準行と同じ表記）、`Some(v)` を数値そのものへ変換する。
        let launch_bounds_field = candidate
            .launch_bounds
            .map_or_else(|| "none".to_string(), |v| v.to_string());
        // `dynamic_smem` 列は実際に起動側で使う動的 SMEM 設定
        // （`RenderedMmaF16BlockTileKernel::uses_dynamic_smem` と同じ式。
        // イシュー #855）を反映する。`layout.needs_dynamic_smem()`
        // （静的判定のみ）をそのまま使うと `force_dynamic_smem=true` の
        // 対照実験行で「静的扱いのまま動的起動している」という CSV 上の
        // 矛盾が生じる。
        let uses_dynamic_smem = layout.needs_dynamic_smem() || candidate.force_dynamic_smem;
        let mut row = format!(
            "{},{},{},{},{},{},{},{},{},{},{}",
            candidate.label,
            candidate.bm,
            candidate.bn,
            candidate.bk,
            candidate.stages,
            candidate.warp_tiles_m,
            candidate.warp_tiles_n,
            launch_bounds_field,
            layout.threads,
            layout.smem_bytes,
            uses_dynamic_smem,
        );
        // `skip_performance_measurement` が真の場合、この候補の
        // render/compile/parity 診断は上で完了済みだが（#855 二段階
        // ゲート）、production 自身が parity FAIL のため性能値の採用は
        // 禁止のまま維持し、全 size を n/a で埋めて `measure_candidate`
        // を呼ばない。
        for &size in &BENCH_SIZES {
            if skip_performance_measurement {
                row.push_str(",n/a,n/a,n/a,n/a");
                continue;
            }
            let config = MeasurementConfig::default();
            match measure_candidate(&compiled, &gemm, &device, size, &config) {
                Ok(m) => {
                    let ratio = production_medians
                        .get(&size)
                        .filter(|&&base| base != 0.0)
                        .map(|&base| m.median / base);
                    row.push_str(&format!(
                        ",{:.4},{:.4},{:.4},{}",
                        m.median,
                        m.q1,
                        m.q3,
                        ratio.map_or("n/a".to_string(), |r| format!("{r:.4}"))
                    ));
                }
                Err(e) => {
                    println!(
                        "{} size={size}: SKIP measurement (candidate launch failed: {e})",
                        candidate.label
                    );
                    row.push_str(",n/a,n/a,n/a,n/a");
                }
            }
        }
        println!("{row}");
    }
}
