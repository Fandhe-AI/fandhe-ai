//! persistent タイルキュー版 pipeline カーネル（grid=SM 数・atomic タイル
//! 取得。イシュー #1346）vs 既定（非 persistent）3 stage pipeline カーネル
//! の GPU-only 性能比較ベンチ実測バイナリ。64×64（#1346）・128×64
//! （イシュー #1347）の両タイル構成を `--tile` で選べる。
//!
//! 本イシュー（#1347）の GB10 実機実測（N=1024/2048/4096 × タイル 2 種・
//! 5 回計測中央値・採否判断）の実行手段
//! （`docs/perf/cuda-gemm-tiled-pipeline-persistent.md`「#1347」節）。
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
//! 参照。`docs/perf/cuda-gemm-tiled-pipeline.md`「#1347」節「ゲート C の
//! 解釈注記」で、この固定費を区間外へ出して数値を救済しない方針を
//! 明記する）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-cuda --example gemm_tiled_pipeline_persistent_bench \
//!   --release --features internal-diagnostics -- --tile both --blocks-per-sm auto
//! ```
//!
//! `--sizes 1024,2048,4096`（既定）・`--stages 3`（既定。128×64 は 2〜3
//! の範囲外だとそのタイルをスキップして理由を表示する）・
//! `--blocks-per-sm auto|<n>`（既定 `auto`）・`--tile 64x64|128x64|both`
//! （既定 `both`）を指定できる。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cuda::{
    CudaDevice, CudaError, CudaGemm, PersistentTiledPipelineFunction, TiledPipelineFunction,
};

/// 決定的シード（`gemm_tiled_pipeline_bench.rs::SEED` と同一値。過去
/// PoC・他ベンチと同じ入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

/// 本ベンチが扱うタイル構成の選択（`--tile` 引数。イシュー #1347）。
/// `gemm.rs::TiledPipelineTile` と 1:1 対応するが、`--tile both` を表現
/// するため列挙を分ける（起動側は選択されたタイルごとに `TiledPipelineTile`
/// へ変換して各 API を呼ぶ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TileSelect {
    Bm64Bn64,
    Bm128Bn64,
}

impl TileSelect {
    fn label(self) -> &'static str {
        match self {
            TileSelect::Bm64Bn64 => "64x64",
            TileSelect::Bm128Bn64 => "128x64",
        }
    }
}

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

/// 既定（非 persistent）pipeline カーネルの GPU-only 計測
/// （`gemm_tiled_pipeline_bench.rs::measure_tiled_pipeline_gpu_only` と
/// 同一実装。本ファイル単体で自己完結させるため複製する。64×64・128×64
/// どちらのタイルでコンパイルされた `func` でも `TiledPipelineFunction`
/// が内部でタグ経由に起動 config を導出するため呼び出し側は変わらない）。
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

/// `--sizes`／`--stages`／`--blocks-per-sm`／`--tile` の 4 引数のみを扱う
/// 最小限のパーサ（外部依存を増やさない。値は整数パース・固定語彙照合
/// のみ行い、シェル呼び出し等へは渡さない。`.claude/rules/security.md`
/// A03）。
struct Args {
    sizes: Vec<usize>,
    stages: u32,
    blocks_per_sm: Option<u32>,
    tiles: Vec<TileSelect>,
}

fn parse_args() -> Args {
    let mut sizes = vec![1024usize, 2048, 4096];
    let mut stages = 3u32;
    let mut blocks_per_sm: Option<u32> = None;
    let mut tiles = vec![TileSelect::Bm64Bn64, TileSelect::Bm128Bn64];

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
            "--tile" if i + 1 < raw.len() => {
                let v = raw[i + 1].trim();
                tiles = match v {
                    "64x64" => vec![TileSelect::Bm64Bn64],
                    "128x64" => vec![TileSelect::Bm128Bn64],
                    "both" => vec![TileSelect::Bm64Bn64, TileSelect::Bm128Bn64],
                    other => panic!(
                        "--tile は 64x64|128x64|both のいずれかである必要があります: {other}"
                    ),
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
        tiles,
    }
}

/// 1 タイル構成分の非 persistent／persistent ハンドルをコンパイルし、
/// `args.sizes` 全形状を計測して 1 行ずつ出力する。`--tile` で選ばれた
/// タイルごとに `main` から呼ばれる（イシュー #1347）。
fn run_tile(gemm: &CudaGemm, device: &CudaDevice, args: &Args, tile: TileSelect) {
    let label = tile.label();

    let non_persistent_func = match tile {
        TileSelect::Bm64Bn64 => CudaGemm::compile_tiled_pipeline_variant(device, args.stages),
        TileSelect::Bm128Bn64 => {
            CudaGemm::compile_tiled_pipeline_128x64_variant(device, args.stages)
        }
    };
    let non_persistent_func = match non_persistent_func {
        Ok(f) => f,
        Err(e) => {
            println!(
                "tile={label} tiled pipeline (stages={}) compilation failed ({e}); nothing to \
                 measure for this tile.",
                args.stages
            );
            return;
        }
    };

    let persistent_func = match tile {
        TileSelect::Bm64Bn64 => CudaGemm::compile_tiled_pipeline_persistent_variant(
            device,
            args.stages,
            args.blocks_per_sm,
        ),
        TileSelect::Bm128Bn64 => CudaGemm::compile_tiled_pipeline_persistent_128x64_variant(
            device,
            args.stages,
            args.blocks_per_sm,
        ),
    };
    let mut persistent_func = match persistent_func {
        Ok(f) => f,
        Err(e) => {
            println!(
                "tile={label} persistent tiled pipeline (stages={}, blocks_per_sm={:?}) \
                 compilation failed ({e}); persistent column will be skipped for this tile.",
                args.stages, args.blocks_per_sm
            );
            for size in &args.sizes {
                let config = MeasurementConfig::default();
                match measure_pipeline_gpu_only(gemm, &non_persistent_func, *size, &config) {
                    Ok(v) => println!(
                        "size={size} tile={label} pipeline3_gpu_only_tflops={v:.4} \
                         persistent_gpu_only_tflops=n/a"
                    ),
                    Err(e) => {
                        println!(
                            "size={size} tile={label}: non-persistent measurement failed ({e})"
                        )
                    }
                }
            }
            return;
        }
    };

    // 解決済み grid 構成（`--blocks-per-sm auto` の実測値・128×64 が
    // 「smem 制約で 2 block/SM」と主張するモジュールコメントの実測確認・
    // 反証に使う。§3.2(e) 判定基準の分母。イシュー #1347）。
    let num_sms = persistent_func.num_sms();
    let blocks_per_sm = persistent_func.blocks_per_sm();
    println!(
        "tile={label} num_sms={num_sms} blocks_per_sm={blocks_per_sm} \
         grid_capacity={}",
        num_sms.saturating_mul(blocks_per_sm)
    );

    for size in &args.sizes {
        let config = MeasurementConfig::default();
        let non_persistent = measure_pipeline_gpu_only(gemm, &non_persistent_func, *size, &config);
        let persistent = measure_persistent_gpu_only(gemm, &mut persistent_func, *size, &config);

        let fmt = |v: &Result<f64, CudaError>| match v {
            Ok(x) => format!("{x:.4}"),
            Err(e) => {
                println!("size={size} tile={label}: measurement failed ({e})");
                "n/a".to_string()
            }
        };
        let ratio = match (&non_persistent, &persistent) {
            (Ok(np), Ok(p)) if *np != 0.0 => format!("{:.4}", p / np),
            _ => "n/a".to_string(),
        };

        println!(
            "size={size} tile={label} pipeline3_gpu_only_tflops={} \
             persistent_gpu_only_tflops={} persistent_over_pipeline3={}",
            fmt(&non_persistent),
            fmt(&persistent),
            ratio,
        );
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

    for tile in &args.tiles {
        run_tile(&gemm, &device, &args, *tile);
    }
}
