//! `CudaMmaGemm::run_f16`（転送込み）の生 API／プール API 比較計測
//! バイナリ（イシュー #1153。`gemm_mma.rs` の per-call 確保〈`clone_htod`
//! ／`alloc_zeros::<f16>` 直呼び〉を `pool.rs::CudaAllocator` 経由へ
//! 結線する Phase 2 の判定手段）。
//!
//! `run_f16`（既定）は raw API（`upload_f16`／`alloc_output_f16`／
//! `launch_f16`／`download_f16`）を呼ぶ。`--pooled` 指定時は同一の
//! 呼び出し列をプール API（`upload_f16_pooled`／`alloc_output_f16_pooled`
//! ／`launch_f16_pooled`／`download_f16_pooled`。`internal-diagnostics`
//! feature 限定）で再現する。**同一バイナリ・同一プロセス内で両経路を
//! 切り替えられる**ため、`--pooled` の有無だけを変えた 2 回の実行を
//! 比較すれば、別コミット間のバイナリ差分（コードサイズ・静的データ
//! レイアウト）を before/after の交絡因子から排除できる（GB10 実機の
//! unified memory 環境では、単なるバイナリの違いが glibc アロケータの
//! 挙動〈mmap しきい値の動的調整〉に影響しうることが判明したため。
//! `docs/perf/cuda-gemm-mma-f16-pool-wiring.md` §6〜§7 参照）。
//!
//! 実測の結果、`--pooled` は dim4096 で明確に後退したため、`run_f16`
//! への本番結線は見送った（`gemm_mma.rs::CudaMmaGemm::run_f16` の
//! ドキュメンテーションコメント参照）。
//!
//! ## 実行手順
//!
//! ```sh
//! # 生 API（run_f16 の本番経路と同一）
//! cargo run -p fandhe-ai-backend-cuda --example gemm_mma_f16_pool_wiring_bench --release
//! # プール API（internal-diagnostics feature 必須）
//! cargo run -p fandhe-ai-backend-cuda --example gemm_mma_f16_pool_wiring_bench \
//!     --release --features internal-diagnostics -- --pooled
//! ```
//!
//! `examples/gemm_mma_bench.rs` と同じ理由（CI ではビルド検証のみ・
//! 実行しない）で `examples/` に置く。決定的シード・計測コア
//! （`bench_harness::protocol::run`。warmup/計測 20 回下限）も
//! `gemm_mma_bench.rs` と同じ方針を踏襲する。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaMmaGemm};
use half::f16;

/// `gemm_mma_bench.rs::SEED` と同一値（過去 PoC・他バックエンドベンチと
/// 同じ入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

/// 計測対象の形状（イシュー #1153 実装計画の受入基準: 512/1024/2048/4096）。
const SIZES: [usize; 4] = [512, 1024, 2048, 4096];

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

/// `--pooled` 引数の有無を判定する（`internal-diagnostics` feature 非
/// 有効時に `--pooled` が渡された場合は起動時に明示エラーで停止する。
/// 黙って raw 経路にフォールバックすると比較対象を取り違える事故に
/// つながるため fail-closed）。
fn parse_pooled_flag() -> bool {
    let pooled = std::env::args().any(|a| a == "--pooled");
    if pooled && cfg!(not(feature = "internal-diagnostics")) {
        eprintln!(
            "gemm_mma_f16_pool_wiring_bench: --pooled には \
             --features internal-diagnostics が必要です"
        );
        std::process::exit(1);
    }
    pooled
}

/// 生 API（`run_f16` の本番経路と同一の呼び出し列）で 1 回計測する。
fn run_raw(gemm: &CudaMmaGemm, a: &[f16], b: &[f16], m: u32, n: u32, k: u32) {
    let (a_dev, b_dev) = gemm
        .upload_f16(a, b)
        .expect("upload_f16 must succeed on CUDA-equipped runner");
    let mut c_dev = gemm
        .alloc_output_f16(m, n)
        .expect("alloc_output_f16 must succeed on CUDA-equipped runner");
    gemm.launch_f16(&a_dev, &b_dev, &mut c_dev, m, n, k)
        .expect("launch_f16 must succeed on CUDA-equipped runner");
    let _ = gemm
        .download_f16(&c_dev)
        .expect("download_f16 must succeed on CUDA-equipped runner");
}

/// プール API（`internal-diagnostics` feature 限定）で 1 回計測する。
#[cfg(feature = "internal-diagnostics")]
fn run_pooled(gemm: &CudaMmaGemm, a: &[f16], b: &[f16], m: u32, n: u32, k: u32) {
    let (a_dev, b_dev) = gemm
        .upload_f16_pooled(a, b)
        .expect("upload_f16_pooled must succeed on CUDA-equipped runner");
    let mut c_dev = gemm
        .alloc_output_f16_pooled(m, n)
        .expect("alloc_output_f16_pooled must succeed on CUDA-equipped runner");
    gemm.launch_f16_pooled(&a_dev, &b_dev, &mut c_dev, m, n, k)
        .expect("launch_f16_pooled must succeed on CUDA-equipped runner");
    let _ = gemm
        .download_f16_pooled(&c_dev)
        .expect("download_f16_pooled must succeed on CUDA-equipped runner");
}

#[cfg(not(feature = "internal-diagnostics"))]
fn run_pooled(_gemm: &CudaMmaGemm, _a: &[f16], _b: &[f16], _m: u32, _n: u32, _k: u32) {
    unreachable!("parse_pooled_flag が internal-diagnostics 非有効時の --pooled を弾く");
}

fn main() {
    let pooled = parse_pooled_flag();

    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_mma_f16_pool_wiring_bench: CUDA driver unavailable \
                 ({detail}); skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_mma_f16_pool_wiring_bench: CudaDevice::new failed \
                 ({other}); skipping."
            );
            return;
        }
    };

    let gemm = match CudaMmaGemm::new(&device) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "backend-cuda gemm_mma_f16_pool_wiring_bench: mma.sync f16 kernel \
                 unavailable ({e}); nothing to measure."
            );
            return;
        }
    };

    let config = MeasurementConfig::default();

    println!("mode={}", if pooled { "pooled" } else { "raw" });
    println!("size,median_ms,q1_ms,q3_ms,tflops");
    for &size in &SIZES {
        let mut rng = Xorshift64Star::new(SEED);
        let a: Vec<f16> = rng.fill_vec_f16(size * size);
        let b: Vec<f16> = rng.fill_vec_f16(size * size);
        let (m, n, k) = (size as u32, size as u32, size as u32);

        let measurement = bench_run(&config, || {
            if pooled {
                run_pooled(&gemm, &a, &b, m, n, k);
            } else {
                run_raw(&gemm, &a, &b, m, n, k);
            }
        })
        .expect("MeasurementConfig::default satisfies the 20/20 lower bound");

        println!(
            "{size},{:.4},{:.4},{:.4},{:.4}",
            measurement.median_secs * 1e3,
            measurement.q1_secs * 1e3,
            measurement.q3_secs * 1e3,
            tflops(size, measurement.median_secs),
        );
    }
}
