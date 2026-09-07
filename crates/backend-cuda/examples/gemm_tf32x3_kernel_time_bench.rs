//! f32 SIMT／TF32 mma.sync／3×TF32／（参考）WMMA TF32 の純カーネル時間
//! 比較ハーネス（イシュー #1356）。
//!
//! `docs/perf/cuda-tensor-core-tolerance-tf32x3-gb10.md` §9「純カーネル
//! 時間」が要求する「単発 TF32・FP32 厳密（`run_tiled_f32`）との比較」を
//! 実施する。計測境界は `gemm_mma_tf32_block_tile_bench.rs::
//! measure_production`（H2D／D2H を計測区間外に置き、`launch_* +
//! synchronize()` のみを計測する契約）と同一にする。
//!
//! # 計測対象
//!
//! - `f32_simt`（[`CudaGemm::launch_tiled_f32`]。`select_tiled_f32_kernel`
//!   の形状条件付き選択〈classic／pipeline〉を本番と同じロジックで通す）
//! - `mma_tf32`（[`CudaMmaTf32Gemm::launch_tf32`]。単発 TF32 mma.sync）
//! - `mma_tf32x3`（[`CudaMmaTf32x3Gemm::launch_tf32x3`]。3×TF32
//!   split-single 法。#1355・#1356 本題）
//! - `wmma_tf32`（参考。[`CudaGemm::launch_wmma_tf32`]。WMMA(TF32) 経路）
//!
//! TF32 mma.sync／3×TF32 の 2 経路は cp.async 16 バイト整列制約
//! （`n % 4 == 0 && k % 4 == 0`）を持つため、対象形状は全て整列形状のみ
//! とする（`SHAPES` 参照。`wmma_tolerance_probe.rs::is_mma_aligned` と
//! 同じ判定基準）。
//!
//! # 実行手順
//!
//! ```sh
//! cargo build -p fandhe-ai-backend-cuda --example gemm_tf32x3_kernel_time_bench --release
//! ./target/release/examples/gemm_tf32x3_kernel_time_bench
//! ```
//!
//! 「5 回計測の中央値」（`.claude/rules/coding-rust.md`）は本バイナリを
//! 5 回プロセス起動することで満たす契約とする（`gemm_mma_tf32_block_tile_bench.rs`
//! と同じ「1 プロセス起動 = 1 run」設計）。出力は `route,m,n,k,kernel,
//! median_ms,q1_ms,q3_ms,tflops` の CSV 風テキスト行（1 行 1
//! (route, shape)）で、集計スクリプトは
//! `docs/perf/cuda-tensor-core-tolerance-tf32x3-gb10.md` に記載する。
//!
//! CUDA driver／NVRTC 非搭載・compute capability 8.0 未満の環境では、
//! 経路ごとに理由を表示して当該経路のみスキップし、残りの経路の計測は
//! 継続する（`f32_simt` は cc 制約がないため必ず計測する）。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm, CudaMmaTf32Gemm, CudaMmaTf32x3Gemm};

/// 決定的シード（他の CUDA GEMM ベンチと同じ `Xorshift64Star::new(0xC0FFEE)`。
/// `docs/perf/cuda-gemm-tiled-pipeline.md` 等の既存 GEMM ベンチと同一値）。
const SEED: u64 = 0xC0FFEE;

/// 計測対象形状（実装計画 §3.3）。正方 4 形状（512〜4096）と K 支配的な
/// 非正方 1 形状（256x256x4096）。全形状とも `n % 4 == 0 && k % 4 == 0`
/// を満たす（mma 系 2 経路の cp.async 整列制約を満たすため）。
const SHAPES: &[(u32, u32, u32)] = &[
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (256, 256, 4096),
];

fn tflops(m: u32, n: u32, k: u32, secs: f64) -> f64 {
    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    flops / secs / 1e12
}

/// [`bench_harness::Measurement`]（median/q1/q3 秒）を CSV 出力用の
/// ミリ秒・TFLOPS へ変換する（`gemm_mma_tf32_block_tile_bench.rs::
/// TflopsMeasurement` と同じ「q1/q3 は所要時間の分布であり、TFLOPS
/// 換算では速い方＝時間が短い方が高 TFLOPS になるため q1/q3 が入れ替わる」
/// 契約は本ハーネスの出力に持ち込まない — 生の秒単位分布をそのまま
/// ミリ秒へ変換して出力し、TFLOPS は中央値のみ併記する（集計スクリプト
/// 側の混乱を避けるため）。
struct KernelTimeRow {
    median_ms: f64,
    q1_ms: f64,
    q3_ms: f64,
    tflops_median: f64,
}

impl KernelTimeRow {
    fn from_measurement(m: u32, n: u32, k: u32, measurement: &bench_harness::Measurement) -> Self {
        Self {
            median_ms: measurement.median_secs * 1e3,
            q1_ms: measurement.q1_secs * 1e3,
            q3_ms: measurement.q3_secs * 1e3,
            tflops_median: tflops(m, n, k, measurement.median_secs),
        }
    }
}

fn print_row(route: &str, m: u32, n: u32, k: u32, kernel: &str, row: &KernelTimeRow) {
    println!(
        "{route},{m},{n},{k},{kernel},{:.6},{:.6},{:.6},{:.6}",
        row.median_ms, row.q1_ms, row.q3_ms, row.tflops_median
    );
}

fn print_header() {
    println!("route,m,n,k,kernel,median_ms,q1_ms,q3_ms,tflops");
}

/// `gemm` が形状 `(n, k)` で実際に選ぶ tiled f32 カーネル種別
/// （`Classic`／`Pipeline`）を文字列で返す（`wmma_tolerance_probe.rs::
/// f32_simt_kernel_kind` と同じ理由・同じ `internal-diagnostics` feature
/// 限定の可用性〈非 feature ビルドでは "n/a" を返す〉）。
fn f32_simt_kernel_kind(_gemm: &CudaGemm, _n: u32, _k: u32) -> &'static str {
    #[cfg(feature = "internal-diagnostics")]
    {
        use fandhe_ai_backend_cuda::TiledF32Kernel;
        match _gemm.tiled_f32_kernel_for(_n, _k) {
            TiledF32Kernel::Classic => "Classic",
            TiledF32Kernel::Pipeline => "Pipeline",
        }
    }
    #[cfg(not(feature = "internal-diagnostics"))]
    {
        "n/a (internal-diagnostics)"
    }
}

/// f32 SIMT 経路（[`CudaGemm::launch_tiled_f32`]）の純カーネル時間を
/// 計測する。計測境界は H2D／D2H を含まない「GPU 実行のみ」（`upload_f32`／
/// `alloc_output_f32` を計測区間外に置き、`launch_tiled_f32` +
/// `synchronize()` のみを計測する。冒頭コメント「計測境界」参照）。
fn measure_f32_simt(
    gemm: &CudaGemm,
    m: u32,
    n: u32,
    k: u32,
    config: &MeasurementConfig,
) -> Result<KernelTimeRow, CudaError> {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec((m as usize) * (k as usize));
    let b: Vec<f32> = rng.fill_vec((k as usize) * (n as usize));

    let (a_dev, b_dev) = gemm.upload_f32(&a, &b)?;
    let mut c_dev = gemm.alloc_output_f32(m, n)?;

    let mut first_err: Option<CudaError> = None;
    let measurement = bench_run(config, || {
        if first_err.is_some() {
            return;
        }
        if let Err(e) = gemm.launch_tiled_f32(&a_dev, &b_dev, &mut c_dev, m, n, k) {
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
    Ok(KernelTimeRow::from_measurement(m, n, k, &measurement))
}

/// TF32 mma.sync 経路（[`CudaMmaTf32Gemm::launch_tf32`]）の純カーネル
/// 時間を計測する（境界は [`measure_f32_simt`] と同じ）。
fn measure_mma_tf32(
    gemm: &CudaMmaTf32Gemm,
    m: u32,
    n: u32,
    k: u32,
    config: &MeasurementConfig,
) -> Result<KernelTimeRow, CudaError> {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec((m as usize) * (k as usize));
    let b: Vec<f32> = rng.fill_vec((k as usize) * (n as usize));

    let (a_dev, b_dev) = gemm.upload_f32(&a, &b)?;
    let mut c_dev = gemm.alloc_output_f32(m, n)?;

    let mut first_err: Option<CudaError> = None;
    let measurement = bench_run(config, || {
        if first_err.is_some() {
            return;
        }
        if let Err(e) = gemm.launch_tf32(&a_dev, &b_dev, &mut c_dev, m, n, k) {
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
    Ok(KernelTimeRow::from_measurement(m, n, k, &measurement))
}

/// 3×TF32 経路（[`CudaMmaTf32x3Gemm::launch_tf32x3`]）の純カーネル時間を
/// 計測する（境界は [`measure_f32_simt`] と同じ）。本ハーネスの入力は
/// `Xorshift64Star` の `[-1, 1)` のため `validate_tf32x3_finite_input`
/// の非有限・TF32 丸めオーバーフロー拒否は発火しない。
fn measure_mma_tf32x3(
    gemm: &CudaMmaTf32x3Gemm,
    m: u32,
    n: u32,
    k: u32,
    config: &MeasurementConfig,
) -> Result<KernelTimeRow, CudaError> {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec((m as usize) * (k as usize));
    let b: Vec<f32> = rng.fill_vec((k as usize) * (n as usize));

    let inputs = gemm.upload_f32(&a, &b)?;
    let mut c_dev = gemm.alloc_output_f32(m, n)?;

    let mut first_err: Option<CudaError> = None;
    let measurement = bench_run(config, || {
        if first_err.is_some() {
            return;
        }
        if let Err(e) = gemm.launch_tf32x3(&inputs, &mut c_dev, m, n, k) {
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
    Ok(KernelTimeRow::from_measurement(m, n, k, &measurement))
}

/// WMMA(TF32) 経路（[`CudaGemm::launch_wmma_tf32`]。参考比較用）の純
/// カーネル時間を計測する（境界は [`measure_f32_simt`] と同じ）。事前に
/// `run_wmma_tf32` を 1 回 probe しておく契約（`cuda_floor_bench.rs::
/// measure_wmma_tf32` と同じ理由。`launch_wmma_tf32` ドキュメンテーション
/// コメント参照）。
fn measure_wmma_tf32(
    gemm: &CudaGemm,
    m: u32,
    n: u32,
    k: u32,
    config: &MeasurementConfig,
) -> Result<KernelTimeRow, CudaError> {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec((m as usize) * (k as usize));
    let b: Vec<f32> = rng.fill_vec((k as usize) * (n as usize));

    // probe: `launch_wmma_tf32` は `run_wmma_tf32` の事前成功を前提とする
    // safe API（呼び出し元の事前検証に依存せず自前でも検証するが、
    // 3 段選択カーネルのいずれもロードされていない場合は
    // `WmmaUnavailable` を返す）。
    gemm.run_wmma_tf32(&a, &b, m, n, k)?;

    let (a_dev, b_dev) = gemm.upload_f32(&a, &b)?;
    let mut c_dev = gemm.alloc_output_f32(m, n)?;

    let mut first_err: Option<CudaError> = None;
    let measurement = bench_run(config, || {
        if first_err.is_some() {
            return;
        }
        if let Err(e) = gemm.launch_wmma_tf32(&a_dev, &b_dev, &mut c_dev, m, n, k) {
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
    Ok(KernelTimeRow::from_measurement(m, n, k, &measurement))
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_tf32x3_kernel_time_bench: CUDA driver unavailable \
                 ({detail}); skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_tf32x3_kernel_time_bench: CudaDevice::new failed ({other}); \
                 skipping."
            );
            return;
        }
    };
    println!("device compute capability: {}", device.arch());

    let config = MeasurementConfig::default();
    print_header();

    // f32_simt（cc 制約なし。必ず構築を試みる）。
    match CudaGemm::new(&device) {
        Ok(gemm) => {
            for &(m, n, k) in SHAPES {
                let kernel = f32_simt_kernel_kind(&gemm, n, k);
                match measure_f32_simt(&gemm, m, n, k, &config) {
                    Ok(row) => print_row("f32_simt", m, n, k, kernel, &row),
                    Err(e) => println!(
                        "# f32_simt m={m} n={n} k={k}: launch failed ({e}); skipping this shape."
                    ),
                }
            }

            // wmma_tf32（参考。同じ `CudaGemm` ハンドルを再利用）。
            for &(m, n, k) in SHAPES {
                match measure_wmma_tf32(&gemm, m, n, k, &config) {
                    Ok(row) => print_row("wmma_tf32", m, n, k, "wmma_tf32", &row),
                    Err(CudaError::WmmaUnavailable { detail }) => {
                        println!(
                            "# wmma_tf32 m={m} n={n} k={k}: unavailable ({detail}); skipping."
                        );
                    }
                    Err(e) => println!(
                        "# wmma_tf32 m={m} n={n} k={k}: launch failed ({e}); skipping this shape."
                    ),
                }
            }
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            println!("# f32_simt/wmma_tf32: NVRTC unavailable ({detail}); skipping both routes.");
        }
        Err(other) => {
            println!("# f32_simt/wmma_tf32: CudaGemm::new failed ({other}); skipping both routes.");
        }
    }

    // mma_tf32（単発 TF32。cc>=8.0 要求）。
    match CudaMmaTf32Gemm::new(&device) {
        Ok(gemm) => {
            for &(m, n, k) in SHAPES {
                match measure_mma_tf32(&gemm, m, n, k, &config) {
                    Ok(row) => print_row("mma_tf32", m, n, k, "mma_tf32", &row),
                    Err(e) => println!(
                        "# mma_tf32 m={m} n={n} k={k}: launch failed ({e}); skipping this shape."
                    ),
                }
            }
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            println!("# mma_tf32: NVRTC unavailable ({detail}); skipping route.");
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            println!("# mma_tf32: compute capability < 8.0 ({detail}); skipping route.");
        }
        Err(other) => {
            println!("# mma_tf32: CudaMmaTf32Gemm::new failed ({other}); skipping route.");
        }
    }

    // mma_tf32x3（3×TF32。cc>=8.0 要求。イシュー #1356 本題）。
    match CudaMmaTf32x3Gemm::new(&device) {
        Ok(gemm) => {
            for &(m, n, k) in SHAPES {
                match measure_mma_tf32x3(&gemm, m, n, k, &config) {
                    Ok(row) => print_row("mma_tf32x3", m, n, k, "mma_tf32x3", &row),
                    Err(e) => println!(
                        "# mma_tf32x3 m={m} n={n} k={k}: launch failed ({e}); skipping this \
                         shape."
                    ),
                }
            }
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            println!("# mma_tf32x3: NVRTC unavailable ({detail}); skipping route.");
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            println!("# mma_tf32x3: compute capability < 8.0 ({detail}); skipping route.");
        }
        Err(other) => {
            println!("# mma_tf32x3: CudaMmaTf32x3Gemm::new failed ({other}); skipping route.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tflops_matches_square_formula_for_equal_dims() {
        // m=n=k のとき `gemm_mma_tf32_block_tile_bench.rs::tflops` の
        // `2*size^3` と一致することを確認する（本ハーネス独自の一般形
        // `2*m*n*k` の縮退確認）。
        let size = 512.0f64;
        let secs = 0.01;
        let expected = 2.0 * size.powi(3) / secs / 1e12;
        assert!((tflops(512, 512, 512, secs) - expected).abs() < 1e-9);
    }

    #[test]
    fn tflops_handles_non_square_shape() {
        let secs = 0.02;
        let expected = 2.0 * 256.0 * 256.0 * 4096.0 / secs / 1e12;
        assert!((tflops(256, 256, 4096, secs) - expected).abs() < 1e-9);
    }

    #[test]
    fn shapes_are_all_mma_aligned() {
        // mma_tf32／mma_tf32x3 の cp.async 16 バイト整列制約
        // （`wmma_tolerance_probe.rs::is_mma_aligned` と同じ判定基準）を
        // 全形状が満たすことを確認する。
        for &(_, n, k) in SHAPES {
            assert!(n.is_multiple_of(4) && k.is_multiple_of(4), "n={n} k={k}");
        }
    }
}
