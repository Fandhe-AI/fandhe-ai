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
//!
//! ## 計測区間の統一（codex-review P2／Cursor Bugbot 指摘。PR #1071 対応）
//!
//! 3 stage（既定）・4 stage（オンデマンドコンパイル）の両方を
//! `launch_tiled_pipeline_f32`（GPU 実行のみ。H2D/D2H・出力バッファ確保は
//! 計測区間外）で揃えて計測する。転送込みの `run_tiled_pipeline_f32`
//! （`tiled_f32` 本番経路と同じ計測区間）と GPU-only 経路
//! （`launch_tiled_pipeline_f32`）は測る対象が異なり、異なる区間の
//! TFLOPS を同じ比率へ混ぜると「転送有無の違い」が「stage 数増加による
//! 改善」として誤計上される（以前の実装の不具合）。そのため本ベンチは
//! 比較を 2 段に分ける:
//!
//! 1. 転送込み同士: `tiled_f32`（本番既定経路） vs `pipeline3`（既定
//!    3 stage・転送込み）。`pipeline3_over_tiled` はこの区間の比率。
//! 2. GPU-only 同士: `pipeline3_gpu_only`（既定 3 stage） vs
//!    `pipeline4_gpu_only`（4 stage）。`pipeline4_over_pipeline3_gpu_only`
//!    はこの区間の比率で、cp.async ステージ数の増加そのものの効果を表す。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm, TiledPipelineFunction};

/// 決定的シード（`gemm_mma_bench.rs` と同一値。過去 PoC・他ベンチと同じ
/// 入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

/// 既定（本番オブジェクトが保持する）ステージ数。GPU-only 比較用に
/// `CudaGemm::compile_tiled_pipeline_variant` で同一段数のハンドルを
/// 別途オンデマンドコンパイルする（`kernels_tiled_pipeline.rs::
/// TP_DEFAULT_STAGES` と同値。本クレート内部定数は非公開のためベンチ側で
/// 値を複製する）。
const STAGE_3: u32 = 3;

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

/// tiled pipeline（既定 3 stage。転送込み）を計測する。`tiled_f32` と
/// 同じ計測区間（H2D/D2H 込み）のため `pipeline3_over_tiled` の分子として
/// 使ってよい。
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

/// tiled pipeline（任意ステージ数。オンデマンドコンパイル済みハンドル経由。
/// GPU 実行のみを計測——H2D/D2H・出力バッファ確保は計測区間外。
/// `gemm_mma_bench.rs::measure_mma_f16` と同じ計測方針）を計測する。
/// 3 stage・4 stage いずれの比較にもこの関数を使い、計測区間を揃える
/// （モジュールコメント「計測区間の統一」参照）。
fn measure_tiled_pipeline_gpu_only(
    gemm: &CudaGemm,
    func: &TiledPipelineFunction,
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

    // 3 stage・4 stage いずれも GPU-only 比較用にオンデマンドコンパイルする
    // （本番オブジェクトの初期化コストには影響しない独立経路。
    // `kernels_tiled_pipeline.rs` 冒頭コメント「stages=4 版はベンチ用途に
    // 限りオンデマンドでコンパイルする」参照。3 stage 側も同一経路で
    // コンパイルし直すのは、GPU-only 計測区間を 4 stage 側と厳密に揃える
    // ため——`CudaGemm::new` が保持する既定ハンドルへは `&self` 経由でしか
    // 到達できず、safe API の型で GPU-only 用途に流用できないため）。
    let stage3_func = match CudaGemm::compile_tiled_pipeline_variant(&device, STAGE_3) {
        Ok(f) => Some(f),
        Err(e) => {
            println!(
                "tiled pipeline (stages={STAGE_3}) on-demand compilation failed ({e}); \
                 GPU-only stage3 column will be skipped."
            );
            None
        }
    };
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
        let pipeline3_gpu_only = stage3_func.as_ref().and_then(|func| {
            match measure_tiled_pipeline_gpu_only(&gemm, func, size, &config) {
                Ok(v) => Some(v),
                Err(e) => {
                    println!("size={size}: stage3 GPU-only measurement failed ({e}); skipping.");
                    None
                }
            }
        });
        let pipeline4_gpu_only = stage4_func.as_ref().and_then(|func| {
            match measure_tiled_pipeline_gpu_only(&gemm, func, size, &config) {
                Ok(v) => Some(v),
                Err(e) => {
                    println!("size={size}: stage4 GPU-only measurement failed ({e}); skipping.");
                    None
                }
            }
        });

        let fmt = |v: Option<f64>| v.map_or("n/a".to_string(), |x| format!("{x:.4}"));
        // 転送込み同士（`tiled_f32` vs `pipeline3`）の比率。
        let ratio_over_tiled = |num: Option<f64>| match num {
            Some(n) if tiled != 0.0 => format!("{:.4}", n / tiled),
            _ => "n/a".to_string(),
        };
        // GPU-only 同士（`pipeline3_gpu_only` vs `pipeline4_gpu_only`）の
        // 比率。stage 数増加そのものの効果を転送有無の差から切り離して表す
        // （codex-review P2／Cursor Bugbot 指摘の是正対象）。
        let pipeline4_over_pipeline3_gpu_only = match (pipeline3_gpu_only, pipeline4_gpu_only) {
            (Some(p3), Some(p4)) if p3 != 0.0 => format!("{:.4}", p4 / p3),
            _ => "n/a".to_string(),
        };

        println!(
            "size={size} tiled_f32_tflops={:.4} pipeline3_tflops={} \
             pipeline3_over_tiled={} | pipeline3_gpu_only_tflops={} \
             pipeline4_gpu_only_tflops={} pipeline4_over_pipeline3_gpu_only={}",
            tiled,
            fmt(pipeline3),
            ratio_over_tiled(pipeline3),
            fmt(pipeline3_gpu_only),
            fmt(pipeline4_gpu_only),
            pipeline4_over_pipeline3_gpu_only,
        );
    }
}
