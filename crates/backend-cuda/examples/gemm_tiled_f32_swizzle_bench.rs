//! tiled f32 経路（本番既定 f32 カーネル `TILED_F32`）への threadblock
//! swizzle 横展開（イシュー #1034）の A/B 計測バイナリ。
//!
//! `CudaGemm::new`（本番既定コンストラクタ。base。swizzle 変種を一切
//! 構築しないため `gemm_wmma_tf32_swizzle_bench.rs` が必要とする
//! `new_without_*` 相当は不要）と、swizzle remap を適用した head 変種
//! （[`CudaGemm::new_with_tiled_f32_swizzle`]。動的選択幅 + 参考として
//! 固定候補 `{8, 16}`）を、`gemm_wmma_tf32_swizzle_bench.rs` と同じ計測
//! コア（`bench-harness::protocol::run`。warmup/計測 20 回以上・
//! 中央値/Q1/Q3）で比較する（実装計画 3 節「ステップ 5」）。
//!
//! `internal-diagnostics` feature を要求する（`Cargo.toml`
//! `required-features`）。動的グルーピング幅選択
//! （`swizzle::select_swizzle_group_width`）は非公開 `mod swizzle` の
//! 関数であり、crate 外部（本 example）からは
//! `fandhe_ai_backend_cuda::diagnostics::tiled_f32_swizzle_group_width`
//! 経由でのみ到達できる（`gemm_wmma_tf32_swizzle_bench.rs` と同じ
//! feature ゲート方針）。
//!
//! ## 計測境界
//!
//! `gemm_wmma_tf32_swizzle_bench.rs`（イシュー #776 の是正経緯参照）と
//! 同じ低水準 API 方式に最初から揃える: `upload_f32`/`alloc_output_f32`
//! を計測ループ外へ出し、`launch_tiled_f32`（GPU 実行＋synchronize の
//! みを計測対象に限定する。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-cuda --example gemm_tiled_f32_swizzle_bench --release \
//!     --features internal-diagnostics
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載環境では [`CudaDevice::new`] 失敗時に理由を
//! 表示してスキップする（`gemm_wmma_tf32_swizzle_bench.rs` と同じ環境
//! 適応分岐）。`TILED_F32` は naive/tiled 4 カーネルの一つで `CudaGemm::
//! new` の早期 return に合流する必須カーネルのため（`kernels::TILED_F32`
//! ドキュメンテーションコメント参照）、`CudaGemm::new` が成功した環境
//! では常に利用可能（`wmma_tf32_staged_available()` 相当の可用性チェック
//! は不要）。
//!
//! 実測値の記録・採否判断は `docs/perf/`（実機セッションで追記予定。
//! 実装計画 5 節「実測値の捏造は行わない」方針。#713）を参照。本ランは
//! CUDA 非搭載環境のため実測値をここに書かない。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm, diagnostics};

/// 決定的シード（`gemm_wmma_tf32_swizzle_bench.rs` と同一値。過去 PoC・
/// 他ベンチと同じ入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

/// H2D/D2H 転送・出力バッファ確保を計測区間の外へ出し、GPU 実行
/// （カーネル起動＋同期）のみを計測する（`gemm_wmma_tf32_swizzle_bench.rs::
/// measure_wmma_tf32` と同じ計測方針。base／head 双方に同一方針を適用
/// するため apples-to-apples の比較になる）。
///
/// **イシュー #1137 対応**: `launch_tiled_f32`（無印）ではなく
/// `launch_tiled_f32_classic`（診断専用・常に classic 版を強制する入口）
/// を使う。#1137 以降 `launch_tiled_f32` は整列形状（本ベンチの形状は
/// すべて該当）で cp.async パイプライン版へ分岐しうるが、`base`
/// （`CudaGemm::new`）はパイプライン利用可能・`variant`
/// （`new_with_tiled_f32_swizzle`）はパイプライン強制無効化（swizzle
/// 変種の A/A 誤認防止。`gemm.rs::CudaGemm::new_with_tiled_f32_swizzle`
/// 参照）済みのため、無印のままでは base がパイプライン・variant が
/// classic+swizzle という異なるカーネル系統同士を比較してしまい、本
/// ベンチが測りたい「swizzle remap の効果のみ」から外れる。
fn measure_tiled_f32(gemm: &CudaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);
    let (m, n, k) = (size as u32, size as u32, size as u32);

    let (a_dev, b_dev) = gemm
        .upload_f32(&a, &b)
        .expect("tiled f32 upload must succeed on CUDA-equipped runner");
    let mut c_dev = gemm
        .alloc_output_f32(m, n)
        .expect("tiled f32 output allocation must succeed on CUDA-equipped runner");

    let measurement = bench_run(config, || {
        gemm.launch_tiled_f32_classic(&a_dev, &b_dev, &mut c_dev, m, n, k)
            .expect("tiled f32 classic GEMM must succeed on CUDA-equipped runner");
        // `launch_tiled_f32_classic` は非同期投入のみの契約（#1013 と同型）
        // のため、明示的な同期で計測境界を維持する。
        gemm.synchronize()
            .expect("stream synchronize must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_tiled_f32_swizzle_bench: CUDA driver unavailable ({detail}); \
                 skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_tiled_f32_swizzle_bench: CudaDevice::new failed ({other}); \
                 skipping."
            );
            return;
        }
    };

    // `TILED_F32` は必須 4 カーネルの一つ（`CudaGemm::new` の早期 return
    // に合流。`kernels::TILED_F32` ドキュメンテーションコメント参照）
    // のため、`new` が成功した時点で base 腕は常に利用可能。
    let base = match CudaGemm::new(&device) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "backend-cuda gemm_tiled_f32_swizzle_bench: CudaGemm::new failed ({e}); \
                 nothing to measure."
            );
            return;
        }
    };

    let num_sms = device.multiprocessor_count().unwrap_or(1).max(1);
    let dynamic_group_width = diagnostics::tiled_f32_swizzle_group_width(num_sms);
    println!(
        "num_sms={num_sms} dynamic_group_width={dynamic_group_width} \
         (candidates: {{8, 16}}; see swizzle.rs::select_swizzle_group_width)"
    );

    // 動的選択幅 + 参考として固定候補 8/16（`gemm_wmma_tf32_swizzle_bench.rs`
    // と同じ方針）。動的選択が候補と一致する場合は重複計測になるが、出力の
    // 単純さを優先し de-dup はしない。
    let group_widths = [dynamic_group_width, 8u32, 16u32];

    for size in [512usize, 1024, 2048, 4096] {
        let config = MeasurementConfig::default();
        let base_tflops = measure_tiled_f32(&base, size, &config);

        let mut variants: Vec<(u32, Option<f64>)> = Vec::new();
        for &group_width in &group_widths {
            match CudaGemm::new_with_tiled_f32_swizzle(&device, group_width) {
                Ok(variant) => {
                    let tflops = measure_tiled_f32(&variant, size, &config);
                    variants.push((group_width, Some(tflops)));
                }
                Err(e) => {
                    println!(
                        "size={size} group_width={group_width}: \
                         new_with_tiled_f32_swizzle failed ({e}); skipping this variant."
                    );
                    variants.push((group_width, None));
                }
            }
        }

        let fmt = |v: Option<f64>| v.map_or("n/a".to_string(), |x| format!("{x:.4}"));
        let ratio = |num: Option<f64>, den: f64| match num {
            Some(n) if den != 0.0 => format!("{:.4}", n / den),
            _ => "n/a".to_string(),
        };

        print!("size={size} base_tflops={base_tflops:.4}");
        for &(group_width, tflops) in &variants {
            print!(
                " swizzle_g{group_width}_tflops={} swizzle_g{group_width}_over_base={}",
                fmt(tflops),
                ratio(tflops, base_tflops),
            );
        }
        println!();
    }
}
