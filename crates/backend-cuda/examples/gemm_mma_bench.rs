//! tiled f32／WMMA f16／`mma.sync` f16 の 3 経路比較ベンチ実測バイナリ
//! （TASK-11.1h・#187）。
//!
//! 受け入れ条件「tiled 実装比の性能向上と対 PyTorch 比の実測記録
//! （5 回中央値）」の実測手段。計測コアは `bench-harness::protocol::run`
//! （warmup 20 回以上・計測 20 回以上・中央値/Q1/Q3。TASK-8.1）を使い、
//! `.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を採用」を
//! 満たす（`crates/backend-metal/examples/gemm_bench.rs` と同じ計測コアを
//! 再利用する判断。`MeasurementConfig::default` の warmup/計測 20 回下限は
//! REQ-8 確定実測用の下限プロトコルであり、そのまま「5 回計測中央値」の
//! 上位互換として使う。両者の使い分けは同ファイルの先例どおり）。
//!
//! `examples/` に置くのは、通常の `cargo test`／CI では実行されず、
//! ビルド検証（`cargo build --workspace --all-targets`）のみが CI で走る
//! ようにするため（self-hosted runner をベンチ実行で占有しない。`ci.md`）。
//! `backend-cpu`／`bench-harness` は既に `backend-cuda` の
//! `dev-dependencies`（`tests/cpu_cuda_wmma_parity.rs` が使用）であり、
//! 本ファイルの追加に伴う `Cargo.toml` の変更は不要（`deps-policy.md`
//! ユーザー承認事項に該当しない）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p backend-cuda --example gemm_mma_bench --release
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載・cc<8.0（mma 経路の下限）環境では、各経路の
//! 初期化失敗を検出した時点でその経路をスキップし理由を表示する
//! （`tests/gemm_mma.rs` の環境適応分岐と同じ判断。本実装セッションの
//! 実行環境が実際にこの経路——CUDA driver はあるが NVRTC はない——を
//! 通る。`kernels_mma.rs` 冒頭「検証状態」参照）。実測値は
//! `docs/perf/cuda-gemm-mma-pipeline.md` の記録テンプレへ転記する。
//!
//! `mma.sync` 経路は `n`/`k` が 8 の倍数の形状のみ対応する
//! （`kernels_mma.rs` 冒頭コメント「整列制約」）。本ベンチの形状は
//! すべてこの制約を満たす正方形状のみを使う。

use backend_cuda::{CudaDevice, CudaError, CudaGemm, CudaMmaGemm, CudaWmmaGemm};
use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use half::f16;

/// 決定的シード（`crates/backend-metal/examples/gemm_bench.rs` と同一値。
/// 過去 PoC・他バックエンドベンチと同じ入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

fn measure_tiled_f32(gemm: &CudaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a = rng.fill_vec(size * size);
    let b = rng.fill_vec(size * size);

    let measurement = bench_run(config, || {
        gemm.run_tiled_f32(&a, &b, size as u32, size as u32, size as u32)
            .expect("tiled f32 GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

fn measure_wmma_f16(gemm: &CudaWmmaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f16> = rng.fill_vec_f16(size * size);
    let b: Vec<f16> = rng.fill_vec_f16(size * size);

    let measurement = bench_run(config, || {
        gemm.run_f16(&a, &b, size as u32, size as u32, size as u32)
            .expect("WMMA f16 GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

fn measure_mma_f16(gemm: &CudaMmaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f16> = rng.fill_vec_f16(size * size);
    let b: Vec<f16> = rng.fill_vec_f16(size * size);

    let measurement = bench_run(config, || {
        gemm.run_f16(&a, &b, size as u32, size as u32, size as u32)
            .expect("mma.sync f16 GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

/// PoC-v2-3 実測の PyTorch f16 実効値（TFLOPS。size=4096 での参照値）。
/// `docs/spec/03-poc/` の実測記録を根拠とする（実装計画 1 節「背景・目的」）。
const PYTORCH_F16_TFLOPS_AT_4096: f64 = 97.6;

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!("backend-cuda gemm_mma_bench: CUDA driver unavailable ({detail}); skipping.");
            return;
        }
        Err(other) => {
            println!("backend-cuda gemm_mma_bench: CudaDevice::new failed ({other}); skipping.");
            return;
        }
    };

    let tiled_gemm = match CudaGemm::new(&device) {
        Ok(g) => Some(g),
        Err(e) => {
            println!("tiled f32 kernel unavailable ({e}); tiled column will be skipped.");
            None
        }
    };
    let wmma_gemm = match CudaWmmaGemm::new(&device) {
        Ok(g) => Some(g),
        Err(e) => {
            println!("WMMA f16 kernel unavailable ({e}); wmma column will be skipped.");
            None
        }
    };
    let mma_gemm = match CudaMmaGemm::new(&device) {
        Ok(g) => Some(g),
        Err(e) => {
            println!("mma.sync f16 kernel unavailable ({e}); mma column will be skipped.");
            None
        }
    };

    if tiled_gemm.is_none() && wmma_gemm.is_none() && mma_gemm.is_none() {
        println!(
            "backend-cuda gemm_mma_bench: no kernel path available in this environment \
             (NVRTC unavailable or device unsupported); nothing to measure. \
             See docs/perf/cuda-gemm-mma-pipeline.md."
        );
        return;
    }

    for size in [512usize, 1024, 2048, 4096] {
        let config = MeasurementConfig::default();

        let tiled = tiled_gemm
            .as_ref()
            .map(|g| measure_tiled_f32(g, size, &config));
        let wmma = wmma_gemm
            .as_ref()
            .map(|g| measure_wmma_f16(g, size, &config));
        let mma = mma_gemm.as_ref().map(|g| measure_mma_f16(g, size, &config));

        let fmt = |v: Option<f64>| v.map_or("n/a".to_string(), |x| format!("{x:.4}"));
        let ratio = |num: Option<f64>, den: Option<f64>| match (num, den) {
            (Some(n), Some(d)) if d != 0.0 => format!("{:.4}", n / d),
            _ => "n/a".to_string(),
        };

        println!(
            "size={size} tiled_f32_tflops={} wmma_f16_tflops={} mma_f16_tflops={} \
             mma_over_tiled={} mma_over_wmma={}",
            fmt(tiled),
            fmt(wmma),
            fmt(mma),
            ratio(mma, tiled),
            ratio(mma, wmma),
        );

        if size == 4096
            && let Some(mma_tflops) = mma
        {
            println!(
                "size=4096 mma_over_pytorch_f16={:.4} (PyTorch f16 reference: {PYTORCH_F16_TFLOPS_AT_4096} TFLOPS, PoC-v2-3)",
                mma_tflops / PYTORCH_F16_TFLOPS_AT_4096
            );
        }
    }
}
