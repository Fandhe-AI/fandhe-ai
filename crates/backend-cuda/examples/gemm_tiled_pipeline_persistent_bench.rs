//! persistent タイルキュー版 pipeline カーネル（grid=SM 数・atomic タイル
//! 取得。イシュー #1346）vs 既定（非 persistent）3 stage pipeline カーネル
//! の GPU-only 性能比較ベンチ実測バイナリ。
//!
//! 兄弟イシュー #1347（GB10 実機での N=1024/2048/4096 × タイル 2 種・
//! 5 回計測中央値・採否判断）の実行手段。本イシュー（#1346）自体は疎通
//! 確認のみを行う（`docs/perf/cuda-gemm-tiled-pipeline-persistent.md`
//! 「実測」節に未実測の旨を明記する）。
//!
//! `examples/` に置く理由・計測コア（`bench-harness::run`）は
//! `gemm_tiled_pipeline_bench.rs` と同一（同モジュールコメント参照）。
//!
//! ## 計測区間の統一
//!
//! `gemm_tiled_pipeline_bench.rs::measure_tiled_pipeline_gpu_only` と同じ
//! 「H2D/D2H・出力バッファ確保を計測区間外に置く GPU-only」区間で、既定
//! （非 persistent）3 stage 版と persistent 版（`blocks_per_sm` 引数で
//! `--blocks-per-sm auto`〈占有率実測既定〉／整数指定を選べる）を比較する。
//! persistent 版はタイルキューカウンタのゼロ化（`memset_zeros`）も
//! カーネル起動と同じストリーム順序で計測区間に含まれる（実際の起動
//! コストの一部という設計判断。`gemm.rs::CudaGemm::
//! launch_tiled_pipeline_persistent_f32` ドキュメンテーションコメント
//! 参照）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-cuda --example gemm_tiled_pipeline_persistent_bench \
//!   --release --features internal-diagnostics -- --blocks-per-sm auto
//! ```
//!
//! `--sizes 1024,2048,4096`（既定）・`--stages 3`（既定）・
//! `--blocks-per-sm auto|<n>`（既定 `auto`）を指定できる。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cuda::{
    CudaDevice, CudaError, CudaGemm, PersistentTiledPipelineFunction, TiledPipelineFunction,
};

/// 決定的シード（`gemm_tiled_pipeline_bench.rs::SEED` と同一値。過去
/// PoC・他ベンチと同じ入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

/// 既定（非 persistent）3 stage pipeline カーネルの GPU-only 計測
/// （`gemm_tiled_pipeline_bench.rs::measure_tiled_pipeline_gpu_only` と
/// 同一実装。本ファイル単体で自己完結させるため複製する）。
fn measure_pipeline_gpu_only(
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

/// persistent タイルキュー版カーネルの GPU-only 計測（`&mut` ハンドル
/// 経由。タイルキューカウンタのゼロ化はモジュールコメント「計測区間の
/// 統一」参照）。
fn measure_persistent_gpu_only(
    gemm: &CudaGemm,
    func: &mut PersistentTiledPipelineFunction,
    size: usize,
    config: &MeasurementConfig,
) -> Result<f64, CudaError> {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);

    let (a_dev, b_dev) = gemm.upload_f32(&a, &b)?;
    let mut c_dev = gemm.alloc_output_f32(size as u32, size as u32)?;

    let mut first_err: Option<CudaError> = None;
    let measurement = bench_run(config, || {
        if first_err.is_some() {
            return;
        }
        if let Err(e) = gemm.launch_tiled_pipeline_persistent_f32(
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

/// `--sizes`／`--stages`／`--blocks-per-sm` の 3 引数のみを扱う最小限の
/// パーサ（外部依存を増やさない。値は整数パース・範囲検証のみ行い、
/// シェル呼び出し等へは渡さない。`.claude/rules/security.md` A03）。
struct Args {
    sizes: Vec<usize>,
    stages: u32,
    blocks_per_sm: Option<u32>,
}

fn parse_args() -> Args {
    let mut sizes = vec![1024usize, 2048, 4096];
    let mut stages = 3u32;
    let mut blocks_per_sm: Option<u32> = None;

    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--sizes" if i + 1 < raw.len() => {
                sizes = raw[i + 1]
                    .split(',')
                    .map(|s| {
                        s.trim().parse::<usize>().unwrap_or_else(|_| {
                            panic!("--sizes に整数以外の値が含まれています: {s}")
                        })
                    })
                    .collect();
                i += 2;
            }
            "--stages" if i + 1 < raw.len() => {
                stages = raw[i + 1]
                    .trim()
                    .parse::<u32>()
                    .unwrap_or_else(|_| panic!("--stages に整数以外の値です: {}", raw[i + 1]));
                i += 2;
            }
            "--blocks-per-sm" if i + 1 < raw.len() => {
                let v = raw[i + 1].trim();
                blocks_per_sm = if v.eq_ignore_ascii_case("auto") {
                    None
                } else {
                    Some(v.parse::<u32>().unwrap_or_else(|_| {
                        panic!("--blocks-per-sm に auto/整数以外の値です: {v}")
                    }))
                };
                i += 2;
            }
            other => {
                println!("unknown argument `{other}`; ignoring.");
                i += 1;
            }
        }
    }

    Args {
        sizes,
        stages,
        blocks_per_sm,
    }
}

fn main() {
    let args = parse_args();

    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_tiled_pipeline_persistent_bench: CUDA driver unavailable \
                 ({detail}); skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_tiled_pipeline_persistent_bench: CudaDevice::new failed \
                 ({other}); skipping."
            );
            return;
        }
    };

    let gemm = match CudaGemm::new(&device) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "backend-cuda gemm_tiled_pipeline_persistent_bench: CudaGemm::new failed ({e}); \
                 nothing to measure."
            );
            return;
        }
    };

    let non_persistent_func = match CudaGemm::compile_tiled_pipeline_variant(&device, args.stages) {
        Ok(f) => f,
        Err(e) => {
            println!(
                "tiled pipeline (stages={}) compilation failed ({e}); nothing to measure.",
                args.stages
            );
            return;
        }
    };
    let mut persistent_func = match CudaGemm::compile_tiled_pipeline_persistent_variant(
        &device,
        args.stages,
        args.blocks_per_sm,
    ) {
        Ok(f) => f,
        Err(e) => {
            println!(
                "persistent tiled pipeline (stages={}, blocks_per_sm={:?}) compilation failed \
                 ({e}); persistent column will be skipped for all sizes.",
                args.stages, args.blocks_per_sm
            );
            for size in &args.sizes {
                let config = MeasurementConfig::default();
                match measure_pipeline_gpu_only(&gemm, &non_persistent_func, *size, &config) {
                    Ok(v) => println!(
                        "size={size} pipeline3_gpu_only_tflops={v:.4} persistent_gpu_only_tflops=n/a"
                    ),
                    Err(e) => println!("size={size}: non-persistent measurement failed ({e})"),
                }
            }
            return;
        }
    };

    println!(
        "num_sms/blocks_per_sm 実測は PersistentTiledPipelineFunction 内部に封じ込め済み \
         （診断表示 API は本ベンチのスコープ外。stages={} blocks_per_sm={:?}）",
        args.stages, args.blocks_per_sm
    );

    for size in &args.sizes {
        let config = MeasurementConfig::default();
        let non_persistent = measure_pipeline_gpu_only(&gemm, &non_persistent_func, *size, &config);
        let persistent = measure_persistent_gpu_only(&gemm, &mut persistent_func, *size, &config);

        let fmt = |v: &Result<f64, CudaError>| match v {
            Ok(x) => format!("{x:.4}"),
            Err(e) => {
                println!("size={size}: measurement failed ({e})");
                "n/a".to_string()
            }
        };
        let ratio = match (&non_persistent, &persistent) {
            (Ok(np), Ok(p)) if *np != 0.0 => format!("{:.4}", p / np),
            _ => "n/a".to_string(),
        };

        println!(
            "size={size} pipeline3_gpu_only_tflops={} persistent_gpu_only_tflops={} \
             persistent_over_pipeline3={}",
            fmt(&non_persistent),
            fmt(&persistent),
            ratio,
        );
    }
}
