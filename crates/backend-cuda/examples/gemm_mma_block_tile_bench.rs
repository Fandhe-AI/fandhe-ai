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
//!
//! 全候補 threads/block=512（`launch_bounds` は付与しない。占有率ヒント
//! 無しでのレジスタ割り当てを比較基準行と揃えるため）・`MMA_BK=32` 不変。
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
struct Candidate {
    label: &'static str,
    bm: u32,
    bn: u32,
    bk: u32,
    stages: u32,
    warp_tiles_m: u32,
    warp_tiles_n: u32,
}

const CANDIDATES: [Candidate; 4] = [
    Candidate {
        label: "bt64x128_s4",
        bm: 64,
        bn: 128,
        bk: 32,
        stages: 4,
        warp_tiles_m: 2,
        warp_tiles_n: 2,
    },
    Candidate {
        label: "bt128x128_s3_wt2x4",
        bm: 128,
        bn: 128,
        bk: 32,
        stages: 3,
        warp_tiles_m: 2,
        warp_tiles_n: 4,
    },
    Candidate {
        label: "bt128x256_s3_wt4x4",
        bm: 128,
        bn: 256,
        bk: 32,
        stages: 3,
        warp_tiles_m: 4,
        warp_tiles_n: 4,
    },
    Candidate {
        label: "bt128x256_s4",
        bm: 128,
        bn: 256,
        bk: 32,
        stages: 4,
        warp_tiles_m: 4,
        warp_tiles_n: 4,
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

    println!(
        "candidate,bm,bn,bk,stages,warp_tiles_m,warp_tiles_n,threads,smem_bytes,dynamic_smem,{}",
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
        let mut row = format!(
            "mma_f16_base(production),{bm},{bn},{bk},{stages},{warp_tiles_m},{warp_tiles_n},\
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
        for &size in &BENCH_SIZES {
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

    for candidate in &CANDIDATES {
        let Some(layout) = layout_or_print_excluded(candidate, optin_budget_bytes) else {
            continue;
        };

        let rendered = match diagnostics::render_mma_f16_block_tile(
            candidate.bm,
            candidate.bn,
            candidate.bk,
            candidate.stages,
            candidate.warp_tiles_m,
            candidate.warp_tiles_n,
            None,
            optin_budget_bytes,
        ) {
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

        let mut row = format!(
            "{},{},{},{},{},{},{},{},{},{}",
            candidate.label,
            candidate.bm,
            candidate.bn,
            candidate.bk,
            candidate.stages,
            candidate.warp_tiles_m,
            candidate.warp_tiles_n,
            layout.threads,
            layout.smem_bytes,
            layout.needs_dynamic_smem(),
        );
        for &size in &BENCH_SIZES {
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
