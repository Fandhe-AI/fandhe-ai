//! TF32 生 `mma.sync`(m16n8k8) ブロックタイル・ステージ数候補の実機 A/B
//! 計測バイナリ（イシュー #841）。
//!
//! #806（PR #832）は `mma_tf32` のブロックタイル拡大・ステージ数増候補を
//! 「候補ソース生成（[`diagnostics::mma_tf32_source_with_block_tile`]）＋
//! PTX ダンプ（`examples/mma_tf32_ptx_dump.rs`）」までの**診断専用パス**
//! として整備したが、実機到達不能のため「候補を実際に起動して計測する
//! 経路」は未実装のまま残った（`docs/perf/cuda-gemm-mma-tf32-block-tile.md`
//! §7 実測表が全欄「未実測」）。本バイナリはその欠落を埋め、
//! [`diagnostics::render_mma_tf32_block_tile`]（本イシューで追加した A/B
//! ランナー型。`gemm_mma_block_tile_bench.rs`〈#840・f16 版〉と同型設計）を
//! 使って候補を NVRTC コンパイル・起動し、比較基準行（現行本番定数の
//! [`CudaMmaTf32Gemm`]）と同条件で TFLOPS を計測する。
//!
//! # `CudaMmaTf32Gemm` の既知 correctness bug（重要）
//!
//! **`CudaMmaTf32Gemm`（生 TF32 `mma.sync` 経路）は #839 で不採用
//! （凍結）と確定済み**: #838 の DGX Spark GB10 実機実測で数値一致 6 本中
//! 4 本 FAIL（`docs/perf/cuda-gemm-mma-tf32-ab.md` §2〜§3）となった機能
//! 欠陥が未修正のまま残っている。本バイナリの数値一致検査（下記）は
//! その事実を隠さず**候補ごとの結果（pass/FAIL）を正直に記録する**ことを
//! 目的とし、pass を強制しない。数値一致 FAIL の候補でも launch-only の
//! スループット計測自体は実施し、CSV へ `parity_cpu`/`parity_vs_base`
//! 列で有効性区分を機械可読に残す（既知欠陥修正前の値は「参考値（採否
//! 判断に使用不可）」として扱う。`docs/perf/cuda-phase34-remeasurement.md`
//! §4 の運用を踏襲）。
//!
//! **本番経路（`gemm_mma_tf32.rs::CudaMmaTf32Gemm`・
//! `MMA_TF32_BM`/`MMA_TF32_BN`/`MMA_TF32_STAGES` 等の本番定数）は一切
//! 変更しない**。採否判断・本番結線は後続イシュー #842 のスコープ。
//!
//! ## 計測対象候補（`docs/perf/cuda-gemm-mma-tf32-block-tile.md` §4）
//!
//! | 識別子 | BM/BN/BK | STAGES | warp タイル | SMEM（机上見積もり） |
//! |--------|----------|--------|------------|----------------------|
//! | `mma_tf32_base`（現行・比較基準） | 64/64/16 | 3 | 2x4 | 28,416B（静的） |
//! | `stage_increase` | 64/64/16 | 4 | 2x4 | 37,888B（静的） |
//! | `m_expand` | 128/64/16 | 3 | 4x2 | 43,776B（静的） |
//! | `n_expand` | 64/128/16 | 3 | 2x4 | 40,704B（静的） |
//! | `both_expand` | 128/128/16 | 3 | 2x4 | 56,064B（opt-in） |
//! | `both_expand_stage_increase` | 128/128/16 | 4 | 2x4 | 74,752B（opt-in） |
//! | `bk_expand` | 64/64/32 | 3 | 2x2 | 53,760B（opt-in） |
//!
//! 全候補 `launch_bounds` は付与しない（占有率ヒント無しでのレジスタ
//! 割り当てを比較基準行と揃えるため。実装計画「A/B 行は `launch_bounds`
//! なし」）。全候補が GB10 opt-in 上限（101,376B。
//! `docs/perf/sm121-device-attributes.md`）以下だが、除外判定は固定値
//! ではなく実機取得した opt-in 予算との動的比較で行う（f16 版・
//! `mma_tf32_ptx_dump.rs` と同じ方針。#831 codex-review P1 是正）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p backend-cuda --example gemm_mma_tf32_block_tile_bench --release \
//!     --features internal-diagnostics
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載・cc<8.0（`mma.sync` の下限）・opt-in 予算
//! 未取得環境では、理由を表示して正常終了する
//! （`gemm_mma_block_tile_bench.rs` と同じ環境適応分岐）。候補ごとの
//! コンパイル失敗・opt-in 予算超過（机上除外）・起動失敗は理由付きで
//! SKIP／desk-excluded 表示し、残りの候補の計測は継続する（fail-closed
//! だがスイープ全体は止めない設計）。数値一致 FAIL は測定を止めず、CSV の
//! `parity_cpu`/`parity_vs_base` 列へ記録した上で計測を続行する（本ファイル
//! 冒頭コメント「`CudaMmaTf32Gemm` の既知 correctness bug」参照）。
//!
//! 「5 回計測の中央値」（`.claude/rules/coding-rust.md`）は**本バイナリを
//! 5 回プロセス起動**し、候補×形状ごとに 5 run の出力から中央値を取る
//! ことで満たす契約とする（`gemm_mma_block_tile_bench.rs` と同じ「1
//! プロセス起動 = 1 run」設計）。
//!
//! 実測値・対現行比・数値一致結果は
//! `docs/perf/cuda-gemm-mma-tf32-block-tile.md` §7 へ記録する。

use backend_cuda::diagnostics::{self, MmaTf32BlockTileLayout};
use backend_cuda::{CudaDevice, CudaError, CudaMmaTf32Gemm};
use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};

/// 決定的シード（`gemm_mma_block_tile_bench.rs`・`gemm_mma_tf32_bench.rs`
/// 系と同一値）。
const SEED: u64 = 0xC0FFEE;

/// スイープ対象の計測形状（実装計画 §3 の 512/1024/2048/4096）。
const BENCH_SIZES: [usize; 4] = [512, 1024, 2048, 4096];

/// 正しさ検査用の小形状。M を全候補の `bm`（64/128）いずれの倍数でもない
/// よう選び、エピローグ guarded store（REQ-8）の境界分岐を実際に踏ませる
/// （実装計画 §3「非整列端を踏む小形状」）。N/K は TF32 `mma.sync` 経路の
/// 整列制約（`k % 4 == 0 && n % 4 == 0`。`gemm_mma_tf32.rs::
/// validate_mma_tf32_alignment`）を満たす必要があるため崩さない。
const CORRECTNESS_M: u32 = 520;
const CORRECTNESS_N: u32 = 512;
const CORRECTNESS_K: u32 = 512;

/// 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」
/// （`.claude/rules/coding-rust.md`「バックエンド構成」。緩和しない）。
fn within_tolerance(actual: f32, expected: f32) -> bool {
    let abs_diff = (actual - expected).abs();
    abs_diff < 1e-5 || abs_diff < 1e-3 * expected.abs()
}

fn tflops(size: usize, secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / secs / 1e12
}

/// `bench_harness::Measurement`（所要時間の中央値/Q1/Q3）を TFLOPS へ
/// 変換する（`gemm_mma_block_tile_bench.rs::TflopsMeasurement` と同じ理由・
/// 同じ変換式）。
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

/// #841 実装計画候補表（本ファイル冒頭コメント参照）1 行分の静的定義。
struct Candidate {
    label: &'static str,
    bm: u32,
    bn: u32,
    bk: u32,
    stages: u32,
    warp_tiles_m: u32,
    warp_tiles_n: u32,
}

const CANDIDATES: [Candidate; 6] = [
    Candidate {
        label: "stage_increase",
        bm: 64,
        bn: 64,
        bk: 16,
        stages: 4,
        warp_tiles_m: 2,
        warp_tiles_n: 4,
    },
    Candidate {
        label: "m_expand",
        bm: 128,
        bn: 64,
        bk: 16,
        stages: 3,
        warp_tiles_m: 4,
        warp_tiles_n: 2,
    },
    Candidate {
        label: "n_expand",
        bm: 64,
        bn: 128,
        bk: 16,
        stages: 3,
        warp_tiles_m: 2,
        warp_tiles_n: 4,
    },
    Candidate {
        label: "both_expand",
        bm: 128,
        bn: 128,
        bk: 16,
        stages: 3,
        warp_tiles_m: 2,
        warp_tiles_n: 4,
    },
    Candidate {
        label: "both_expand_stage_increase",
        bm: 128,
        bn: 128,
        bk: 16,
        stages: 4,
        warp_tiles_m: 2,
        warp_tiles_n: 4,
    },
    Candidate {
        label: "bk_expand",
        bm: 64,
        bn: 64,
        bk: 32,
        stages: 3,
        warp_tiles_m: 2,
        warp_tiles_n: 2,
    },
];

/// 比較基準行（現行本番定数の [`CudaMmaTf32Gemm`]。base カーネルの GPU
/// 実行のみを計測する。`gemm_mma_block_tile_bench.rs::measure_production`
/// と同じ計測方針: H2D/D2H・出力確保は計測区間の外）。
fn measure_production(
    gemm: &CudaMmaTf32Gemm,
    size: usize,
    config: &MeasurementConfig,
) -> Result<TflopsMeasurement, CudaError> {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);

    let (a_dev, b_dev) = gemm.upload_f32(&a, &b)?;
    let mut c_dev = gemm.alloc_output_f32(size as u32, size as u32)?;

    // `bench_run` のクロージャは `FnMut()`（非 fallible）契約のため、計測中
    // の起動失敗はここで最初の `CudaError` を捕捉し `Result` として呼び
    // 出し元へ返す（`gemm_mma_block_tile_bench.rs::measure_production` と
    // 同じ理由・同じ契約）。
    let mut first_err: Option<CudaError> = None;
    let measurement = bench_run(config, || {
        if first_err.is_some() {
            return;
        }
        if let Err(e) = gemm.launch_tf32(
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

/// 候補カーネル（[`diagnostics::CompiledMmaTf32BlockTileKernel`]）の GPU
/// 実行のみを計測する（[`measure_production`] と同じ計測方針・同じエラー
/// 捕捉契約）。
fn measure_candidate(
    compiled: &diagnostics::CompiledMmaTf32BlockTileKernel,
    gemm: &CudaMmaTf32Gemm,
    device: &CudaDevice,
    size: usize,
    config: &MeasurementConfig,
) -> Result<TflopsMeasurement, CudaError> {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);

    // アップロード・出力バッファ確保は比較基準行と同じ `CudaMmaTf32Gemm`
    // ヘルパー（`upload_f32`/`alloc_output_f32`）を再利用する（候補
    // カーネルもバッファレイアウト・要素型は本番経路と同一の f32 行優先
    // ため、専用のアップロード経路を新設する必要がない）。
    let (a_dev, b_dev) = gemm.upload_f32(&a, &b)?;
    let mut c_dev = gemm.alloc_output_f32(size as u32, size as u32)?;

    let stream = device.stream();
    let mut first_err: Option<CudaError> = None;
    let measurement = bench_run(config, || {
        if first_err.is_some() {
            return;
        }
        if let Err(e) = compiled.launch_tf32(
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

/// 候補カーネルを [`CORRECTNESS_M`]/[`CORRECTNESS_N`]/[`CORRECTNESS_K`]
/// 形状で 1 回起動し、f32 出力を得る（数値一致検査 (a)/(b) 共通の実行
/// ヘルパー）。
fn run_correctness_shape(
    compiled: &diagnostics::CompiledMmaTf32BlockTileKernel,
    gemm: &CudaMmaTf32Gemm,
    device: &CudaDevice,
    a_f32: &[f32],
    b_f32: &[f32],
) -> Result<Vec<f32>, CudaError> {
    let (a_dev, b_dev) = gemm.upload_f32(a_f32, b_f32)?;
    let mut c_dev = gemm.alloc_output_f32(CORRECTNESS_M, CORRECTNESS_N)?;
    compiled.launch_tf32(
        device.stream(),
        &a_dev,
        &b_dev,
        &mut c_dev,
        CORRECTNESS_M,
        CORRECTNESS_N,
        CORRECTNESS_K,
    )?;
    gemm.download_f32(&c_dev)
}

/// 候補出力と参照出力（要素ごと）を統一複合判定で比較し、mismatch が
/// 0 件かを返す。
fn matches_reference(actual: &[f32], expected: &[f32]) -> bool {
    actual
        .iter()
        .zip(expected.iter())
        .filter(|(a, e)| !within_tolerance(**a, **e))
        .count()
        == 0
}

/// 候補が opt-in 予算内かを実測レイアウトから判定する（固定除外を避け、
/// 接続中の実デバイスの opt-in 上限との動的比較で行う。
/// `gemm_mma_block_tile_bench.rs::layout_or_print_excluded` と同じ方針）。
fn layout_or_print_excluded(
    candidate: &Candidate,
    optin_budget_bytes: u32,
) -> Option<MmaTf32BlockTileLayout> {
    match diagnostics::mma_tf32_block_tile_layout(
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
                "backend-cuda gemm_mma_tf32_block_tile_bench: CUDA driver unavailable \
                 ({detail}); skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_mma_tf32_block_tile_bench: CudaDevice::new failed ({other}); \
                 skipping."
            );
            return;
        }
    };

    let optin_budget_bytes = match device.shared_memory_per_block_optin() {
        Some(v) => v,
        None => {
            println!(
                "backend-cuda gemm_mma_tf32_block_tile_bench: \
                 CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN unavailable; skipping."
            );
            return;
        }
    };
    println!("device: optin_budget_bytes={optin_budget_bytes}");

    let gemm = match CudaMmaTf32Gemm::new(&device) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "backend-cuda gemm_mma_tf32_block_tile_bench: CudaMmaTf32Gemm::new failed \
                 ({e}); nothing to measure. See docs/perf/cuda-gemm-mma-tf32-block-tile.md."
            );
            return;
        }
    };

    // 正しさ検査用の CPU 参照値（統一複合判定。緩和しない。`f32::mul_add`
    // FMA 契約。本ファイル冒頭コメント「数値一致検査」参照）。TF32 経路は
    // ホスト側が既に f32 のため f16 版のような丸め往復は不要。
    let mut rng = Xorshift64Star::new(SEED);
    let a_ref: Vec<f32> = rng.fill_vec((CORRECTNESS_M * CORRECTNESS_K) as usize);
    let b_ref: Vec<f32> = rng.fill_vec((CORRECTNESS_K * CORRECTNESS_N) as usize);
    let mut expected_cpu = vec![0.0f32; (CORRECTNESS_M * CORRECTNESS_N) as usize];
    backend_cpu::matmul_reference_fma(
        &a_ref,
        &b_ref,
        &mut expected_cpu,
        CORRECTNESS_M as usize,
        CORRECTNESS_N as usize,
        CORRECTNESS_K as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed bench input");

    // base カーネル（現行本番定数の `CudaMmaTf32Gemm`）の同一入力に対する
    // 実際の GPU 出力（数値一致検査 (b): 候補との GPU-GPU 相互一致）。
    // #839 の既知 correctness bug のため、この base 出力自体が CPU 参照値
    // と一致しない可能性がある（本ファイル冒頭コメント「既知
    // correctness bug」参照）。(b) は「ブロックタイル書き換え自体が
    // 演算内容を変えていないか」を (a) から分離して検証するための独立
    // 判定であり、base の pass/FAIL とは無関係に実施する。
    let base_actual: Result<Vec<f32>, CudaError> = (|| {
        let (a_dev, b_dev) = gemm.upload_f32(&a_ref, &b_ref)?;
        let mut c_dev = gemm.alloc_output_f32(CORRECTNESS_M, CORRECTNESS_N)?;
        gemm.launch_tf32(
            &a_dev,
            &b_dev,
            &mut c_dev,
            CORRECTNESS_M,
            CORRECTNESS_N,
            CORRECTNESS_K,
        )?;
        gemm.download_f32(&c_dev)
    })();
    let base_vs_cpu_ok = base_actual
        .as_ref()
        .map(|actual| matches_reference(actual, &expected_cpu))
        .unwrap_or(false);
    println!(
        "mma_tf32_base parity_cpu={} (known correctness bug tracked in \
         docs/perf/cuda-gemm-mma-tf32-ab.md #839; see file header comment)",
        if base_actual.is_ok() {
            base_vs_cpu_ok.to_string()
        } else {
            "SKIP".to_string()
        }
    );

    println!(
        "candidate,bm,bn,bk,stages,warp_tiles_m,warp_tiles_n,threads,smem_bytes,dynamic_smem,\
         parity_cpu,parity_vs_base,{}",
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
    // （`diagnostics::mma_tf32_block_tile_layout_production` →
    // `derive_mma_tf32_block_tile_layout`）から導出する
    // （`gemm_mma_block_tile_bench.rs` codex-review 是正済みの方針を最初
    // から反映）。
    let production_medians: std::collections::HashMap<usize, f64> = {
        let layout = diagnostics::mma_tf32_block_tile_layout_production()
            .expect("production MMA_TF32_BM/BN/BK/STAGES/WARP_TILES must derive a valid layout");
        let mut row = format!(
            "mma_tf32_base(production),{},{},{},{},{},{},{},{},{},{},n/a",
            layout.bm,
            layout.bn,
            layout.bk,
            layout.stages,
            layout.warp_tiles_m,
            layout.warp_tiles_n,
            layout.threads,
            layout.smem_bytes,
            layout.needs_dynamic_smem(),
            base_actual
                .is_ok()
                .then_some(base_vs_cpu_ok)
                .map_or("SKIP".to_string(), |ok| ok.to_string()),
        );
        // 比較基準行の実測値は `measure_production` を size ごとに 1 回
        // だけ呼び、`production_medians`（ratio 分母）にも同じ結果を
        // 使い回す（`gemm_mma_block_tile_bench.rs` codex-review 是正済み
        // の方針: base 行の実測値と各候補の ratio 分母を同一計測にする）。
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
                        "mma_tf32_base size={size}: SKIP measurement (production launch \
                         failed: {e})"
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

        let rendered = match diagnostics::render_mma_tf32_block_tile(
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

        // 数値一致検査（計測より先に実施。FAIL でも計測は止めず「参考値」
        // として続行する。本ファイル冒頭コメント「既知 correctness bug」
        // 参照）。(a) CPU f32::mul_add 参照との統一複合判定、(b) base
        // カーネル出力との GPU-GPU 相互一致（統一複合判定）。
        let candidate_actual = run_correctness_shape(&compiled, &gemm, &device, &a_ref, &b_ref);
        let (parity_cpu, parity_vs_base) = match &candidate_actual {
            Ok(actual) => {
                let vs_cpu = matches_reference(actual, &expected_cpu);
                let vs_base = base_actual
                    .as_ref()
                    .map(|base| matches_reference(actual, base))
                    .unwrap_or(false);
                (Some(vs_cpu), Some(vs_base))
            }
            Err(e) => {
                println!("{}: SKIP (parity launch failed: {e})", candidate.label);
                (None, None)
            }
        };
        if parity_cpu.is_none() {
            continue;
        }
        if parity_cpu == Some(false) {
            println!(
                "{}: FAIL (parity mismatch vs CPU f32::mul_add reference; measuring anyway as \
                 reference-only value — not usable for adoption decisions until #839's known \
                 correctness bug is fixed)",
                candidate.label
            );
        }

        let mut row = format!(
            "{},{},{},{},{},{},{},{},{},{},{},{}",
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
            parity_cpu.map_or("n/a".to_string(), |v| v.to_string()),
            parity_vs_base.map_or("n/a".to_string(), |v| v.to_string()),
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
