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
//! cargo run -p fandhe-ai-backend-cuda --example gemm_mma_bench --release
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

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm, CudaMmaGemm, CudaWmmaGemm};
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

/// `mma.sync` 経路のみ H2D/D2H 転送・出力バッファ確保を計測区間の外へ
/// 出し、GPU 実行（カーネル起動 + 同期）のみを計測する（PR #255 レビュー
/// 指摘: 転送込みの時間で TFLOPS を算出すると PyTorch の compute-only
/// 基準〔`PYTORCH_F16_TFLOPS_AT_4096` の比較対象〕と不整合になる）。
/// tiled f32／WMMA f16 の 2 経路は本 PR のスコープ外（`CudaGemm`・
/// `CudaWmmaGemm` は既存 API のままとし、本イシューでは変更しない）
/// のため引き続き転送込みで計測する。
fn measure_mma_f16(gemm: &CudaMmaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f16> = rng.fill_vec_f16(size * size);
    let b: Vec<f16> = rng.fill_vec_f16(size * size);

    let (a_dev, b_dev) = gemm
        .upload_f16(&a, &b)
        .expect("mma.sync f16 upload must succeed on CUDA-equipped runner");
    let mut c_dev = gemm
        .alloc_output_f16(size as u32, size as u32)
        .expect("mma.sync f16 output allocation must succeed on CUDA-equipped runner");

    let measurement = bench_run(config, || {
        gemm.launch_f16(
            &a_dev,
            &b_dev,
            &mut c_dev,
            size as u32,
            size as u32,
            size as u32,
        )
        .expect("mma.sync f16 GEMM must succeed on CUDA-equipped runner");
        // #1013: `launch_f16` は非同期投入のみに契約変更されたため、
        // 明示的な同期で「GPU 実行 + 同期」の計測境界を維持する。
        gemm.synchronize()
            .expect("stream synchronize must succeed on CUDA-equipped runner");
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

    // PR #255 レビュー指摘への対処（`measure_mma_f16` 冒頭コメント参照）:
    // mma_f16_tflops のみ H2D/D2H・出力バッファ確保を計測区間から除外した
    // 「GPU 実行のみ」の値であり、tiled_f32_tflops／wmma_f16_tflops は
    // 引き続き転送込みで計測する（tiled/WMMA 経路は本 PR のスコープ外）。
    // よって mma_over_tiled／mma_over_wmma の比は厳密な apples-to-apples
    // 比較ではなく mma 側に有利な方向へ偏る。docs/perf/cuda-gemm-mma-pipeline.md
    // へ転記する際はこの注記も一緒に残すこと。
    println!(
        "NOTE: mma_f16_tflops excludes H2D/D2H transfer and output buffer \
         allocation from the timed region (GPU execution only); \
         tiled_f32_tflops/wmma_f16_tflops still include them. \
         mma_over_tiled/mma_over_wmma are therefore NOT apples-to-apples \
         (biased in mma's favor). See PR #255 review."
    );

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
