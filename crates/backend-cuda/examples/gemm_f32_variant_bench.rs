//! f32 GEMM の形状別カーネル選択（イシュー #1035。simple / double-buffer /
//! split-K）の A/B 計測バイナリ。
//!
//! `CudaGemm::new`（本番既定コンストラクタ。Simple 経路の base）と
//! `gemm_variant_selection::CudaGemmF32VariantSelection`
//! （`gemm_variant::select_f32_gemm_variant` の判定に従って実際に変種を
//! 起動するハンドル）を、`gemm_tiled_f32_swizzle_bench.rs` と同じ計測コア
//! （`bench-harness::protocol::run`。warmup/計測込みで 20 回以上・
//! 中央値/Q1/Q3）で比較する。
//!
//! `internal-diagnostics` feature を要求する（`Cargo.toml`
//! `required-features`。`gemm_variant_selection` モジュール自体が同 feature
//! 限定のため）。
//!
//! ## 計測境界（`gemm_tiled_f32_swizzle_bench.rs` との差分）
//!
//! `gemm_tiled_f32_swizzle_bench.rs` は `upload_f32`/`alloc_output_f32`/
//! `launch_tiled_f32` の低水準 API で H2D/D2H を計測区間外へ出すが、
//! `CudaGemmF32VariantSelection::run_f32`（本 example が使う唯一の公開
//! API）は H2D→カーネル起動→D2H を一括で行う高水準 API のみを提供する
//! （実装計画のスコープを「選択ロジック＋起動」に絞り、低水準分割 API の
//! 追加はスコープ外とした判断。実測値は H2D/D2H 込みのエンドツーエンド
//! 時間である点に注意——本ランは CUDA 非搭載環境のため実測値そのものを
//! ここに書かない）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-cuda --example gemm_f32_variant_bench --release \
//!     --features internal-diagnostics
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載環境では `CudaDevice::new` 失敗時に理由を
//! 表示してスキップする（既存ベンチと同じ環境適応分岐）。
//!
//! 実測値の記録・採否判断は `docs/perf/`（実機セッションで追記予定）を
//! 参照。本番既定経路への結線判断は実測後にユーザー承認を得て行う
//! （実装計画 §8「スコープ外・追跡事項」）。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cuda::gemm_variant_selection::CudaGemmF32VariantSelection;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm};

/// 決定的シード（既存 GEMM ベンチと同一値。過去 PoC・他ベンチと同じ入力
/// 分布に揃える）。
const SEED: u64 = 0xC0FFEE;

fn tflops(m: usize, n: usize, k: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    flops / median_secs / 1e12
}

/// base（本番既定 `TILED_F32`）を `CudaGemm::run_tiled_f32` で計測する。
fn measure_base(gemm: &CudaGemm, m: usize, n: usize, k: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(m * k);
    let b: Vec<f32> = rng.fill_vec(k * n);
    let (m_u, n_u, k_u) = (m as u32, n as u32, k as u32);

    let measurement = bench_run(config, || {
        gemm.run_tiled_f32(&a, &b, m_u, n_u, k_u)
            .expect("base tiled f32 GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(m, n, k, measurement.median_secs)
}

/// `select_f32_gemm_variant` が選ぶ変種を `CudaGemmF32VariantSelection::
/// run_f32` で計測する。
fn measure_selected(
    gemm: &CudaGemmF32VariantSelection,
    m: usize,
    n: usize,
    k: usize,
    config: &MeasurementConfig,
) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(m * k);
    let b: Vec<f32> = rng.fill_vec(k * n);
    let (m_u, n_u, k_u) = (m as u32, n as u32, k as u32);

    let measurement = bench_run(config, || {
        gemm.run_f32(&a, &b, m_u, n_u, k_u)
            .expect("variant-selected f32 GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(m, n, k, measurement.median_secs)
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_f32_variant_bench: CUDA driver unavailable ({detail}); skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_f32_variant_bench: CudaDevice::new failed ({other}); skipping."
            );
            return;
        }
    };

    let base = match CudaGemm::new(&device) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "backend-cuda gemm_f32_variant_bench: CudaGemm::new failed ({e}); nothing to measure."
            );
            return;
        }
    };

    let selection = match CudaGemmF32VariantSelection::new(&device) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "backend-cuda gemm_f32_variant_bench: \
                 CudaGemmF32VariantSelection::new failed ({e}); nothing to measure."
            );
            return;
        }
    };

    println!(
        "num_sms={:?} double_buffer_available={} split_k_partial_error={:?} \
         split_k_reduce_error={:?}",
        selection.num_sms(),
        selection.double_buffer_available(),
        selection.split_k_partial_error(),
        selection.split_k_reduce_error(),
    );

    // N=256〜512（小サイズ。SM が遊ぶ candle 実測レンジ）・4096（大サイズ）に
    // 加え、K 支配的非正方形状（split-K 対象。実装計画 §3 のヒューリスティック
    // が意図する形状）を計測する。
    let shapes: [(usize, usize, usize); 6] = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (128, 128, 8192),
        (256, 256, 16384),
    ];

    for (m, n, k) in shapes {
        let config = MeasurementConfig::default();
        let base_tflops = measure_base(&base, m, n, k, &config);
        let variant = selection.selected_variant(m as u32, n as u32, k as u32);
        let selected_tflops = measure_selected(&selection, m, n, k, &config);
        let ratio = if base_tflops != 0.0 {
            selected_tflops / base_tflops
        } else {
            f64::NAN
        };
        println!(
            "m={m} n={n} k={k} variant={variant:?} base_tflops={base_tflops:.4} \
             selected_tflops={selected_tflops:.4} ratio={ratio:.4}"
        );
    }
}
