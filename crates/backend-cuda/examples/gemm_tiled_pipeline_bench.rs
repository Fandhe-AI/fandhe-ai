//! tiled f32（本番既定経路）vs tiled pipeline（cp.async 3 stage・既定）vs
//! tiled pipeline（4 stage・オンデマンドコンパイル）の 3 経路比較ベンチ
//! 実測バイナリ（イシュー #1033）。
//!
//! 受け入れ条件「N=4096 での改善値の記録（5 回計測中央値）」の実測手段。
//! 計測コアは `bench-harness::run`（warmup 20 回以上・計測 20 回以上・
//! 中央値/Q1/Q3。TASK-8.1）を使う（`gemm_mma_bench.rs` と同じ計測コア。
//! `.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値」の上位互換
//! として使う）。
//!
//! `examples/` に置くのは、通常の `cargo test`／CI では実行されず、
//! ビルド検証（`cargo build --workspace --all-targets`）のみが CI で走る
//! ようにするため（self-hosted runner をベンチ実行で占有しない。`ci.md`）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-cuda --example gemm_tiled_pipeline_bench --release
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載・cp.async 非対応（sm_80 未満）環境では、各
//! 経路の初期化失敗を検出した時点でその経路をスキップし理由を表示する
//! （`gemm_mma_bench.rs` の環境適応分岐と同じ判断。本実装セッションの
//! 実行環境が実際にこの経路を通る）。実測値は
//! `docs/perf/cuda-gemm-tiled-pipeline.md` の記録テンプレへ転記する。
//!
//! tiled pipeline 経路は `n`/`k` が 4 の倍数の形状のみ対応する
//! （`kernels_tiled_pipeline.rs` 冒頭コメント「整列制約」）。本ベンチの
//! 形状はすべてこの制約を満たす正方形状のみを使う。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm};

/// 決定的シード（`gemm_mma_bench.rs` と同一値。過去 PoC・他ベンチと同じ
/// 入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

/// 4 stage 変種の比較対象ステージ数。
const STAGE_4: u32 = 4;

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

/// tiled f32（本番既定経路。転送込み）を計測する。
fn measure_tiled_f32(gemm: &CudaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);

    let measurement = bench_run(config, || {
        gemm.run_tiled_f32(&a, &b, size as u32, size as u32, size as u32)
            .expect("tiled f32 GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

/// tiled pipeline（既定 3 stage。転送込み）を計測する。
fn measure_tiled_pipeline_default(gemm: &CudaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);

    let measurement = bench_run(config, || {
        gemm.run_tiled_pipeline_f32(&a, &b, size as u32, size as u32, size as u32)
            .expect("tiled pipeline GEMM must succeed on cp.async-capable runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

/// tiled pipeline（4 stage。オンデマンドコンパイル済みハンドル経由。
/// GPU 実行のみを計測——H2D/D2H・出力バッファ確保は計測区間外。
/// `gemm_mma_bench.rs::measure_mma_f16` と同じ計測方針）を計測する。
fn measure_tiled_pipeline_stage4(
    gemm: &CudaGemm,
    func: &cudarc::driver::CudaFunction,
    size: usize,
    config: &MeasurementConfig,
) -> Result<f64, CudaError> {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);

    let (a_dev, b_dev) = gemm.upload_f32(&a, &b)?;
    let mut c_dev = gemm.alloc_output_f32(size as u32, size as u32)?;

    // `bench_run` のクロージャは `FnMut()` （非 fallible）契約のため、
    // 計測中の CUDA 起動失敗をここで捕捉し、計測終了後に `Err` として
    // 返す（`gemm_wmma_tf32_staged_stages_bench.rs::measure_dyn_staged`
    // と同じ理由・同じ契約）。
    let mut first_err: Option<CudaError> = None;
    let measurement = bench_run(config, || {
        if first_err.is_some() {
            return;
        }
        if let Err(e) = gemm.launch_tiled_pipeline_f32(
            func,
            &a_dev,
            &b_dev,
            &mut c_dev,
            size as u32,
            size as u32,
            size as u32,
        ) {
            first_err = Some(e);
            return;
        }
        if let Err(e) = gemm.synchronize() {
            first_err = Some(e);
        }
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    if let Some(e) = first_err {
        return Err(e);
    }
    Ok(tflops(size, measurement.median_secs))
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_tiled_pipeline_bench: CUDA driver unavailable ({detail}); \
                 skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_tiled_pipeline_bench: CudaDevice::new failed ({other}); \
                 skipping."
            );
            return;
        }
    };

    let gemm = match CudaGemm::new(&device) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "backend-cuda gemm_tiled_pipeline_bench: CudaGemm::new failed ({e}); nothing \
                 to measure. See docs/perf/cuda-gemm-tiled-pipeline.md."
            );
            return;
        }
    };

    if !gemm.tiled_pipeline_available() {
        println!(
            "tiled pipeline kernel unavailable ({:?}); pipeline columns will be skipped. \
             tiled_f32 column is still measured below for reference.",
            gemm.tiled_pipeline_unavailable_reason()
        );
    }

    // 4 stage 変種はオンデマンドコンパイル（本番オブジェクトの初期化コスト
    // には影響しない独立経路。`kernels_tiled_pipeline.rs` 冒頭コメント
    // 「stages=4 版はベンチ用途に限りオンデマンドでコンパイルする」参照）。
    let stage4_func = match CudaGemm::compile_tiled_pipeline_variant(&device, STAGE_4) {
        Ok(f) => Some(f),
        Err(e) => {
            println!(
                "tiled pipeline (stages={STAGE_4}) compilation failed ({e}); stage4 column \
                 will be skipped."
            );
            None
        }
    };

    for size in [1024usize, 2048, 4096] {
        let config = MeasurementConfig::default();

        let tiled = measure_tiled_f32(&gemm, size, &config);
        let pipeline3 = gemm
            .tiled_pipeline_available()
            .then(|| measure_tiled_pipeline_default(&gemm, size, &config));
        let pipeline4 = stage4_func.as_ref().and_then(|func| {
            match measure_tiled_pipeline_stage4(&gemm, func, size, &config) {
                Ok(v) => Some(v),
                Err(e) => {
                    println!("size={size}: stage4 measurement failed ({e}); skipping.");
                    None
                }
            }
        });

        let fmt = |v: Option<f64>| v.map_or("n/a".to_string(), |x| format!("{x:.4}"));
        let ratio = |num: Option<f64>| match num {
            Some(n) if tiled != 0.0 => format!("{:.4}", n / tiled),
            _ => "n/a".to_string(),
        };

        println!(
            "size={size} tiled_f32_tflops={:.4} pipeline3_tflops={} pipeline4_tflops={} \
             pipeline3_over_tiled={} pipeline4_over_tiled={}",
            tiled,
            fmt(pipeline3),
            fmt(pipeline4),
            ratio(pipeline3),
            ratio(pipeline4),
        );
    }
}
