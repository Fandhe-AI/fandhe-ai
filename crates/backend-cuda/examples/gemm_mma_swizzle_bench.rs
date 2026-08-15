//! L2 再利用のためのタイル→SM 割り当てスウィズル（イシュー #499）の
//! A/B 計測バイナリ。
//!
//! `kernels_mma::MMA_F16`（本番カーネル・無変更）を使う base
//! （[`CudaMmaGemm::new`]）と、swizzle remap を適用した head 変種
//! （[`CudaMmaGemm::new_with_swizzle`]。動的選択幅 + 参考として固定候補
//! `{8, 16}`）を、`gemm_mma_bench.rs` と同じ計測コア
//! （`bench-harness::protocol::run`。warmup/計測 20 回以上・中央値/Q1/Q3。
//! `.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値」の上位互換）
//! で比較する（実装計画 3 節「example」）。
//!
//! `internal-diagnostics` feature を要求する（`Cargo.toml`
//! `required-features`）。動的グルーピング幅選択
//! （`swizzle::select_swizzle_group_width`）は非公開 `mod swizzle` の
//! 関数であり、crate 外部（本 example）からは
//! `backend_cuda::diagnostics::mma_swizzle_group_width` 経由でのみ到達
//! できるため（`lib.rs::diagnostics` モジュール冒頭コメント参照。
//! `examples/gemm_profile_target.rs` と同じ feature ゲート方針）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p backend-cuda --example gemm_mma_swizzle_bench --release \
//!     --features internal-diagnostics
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載・cc<8.0（mma 経路の下限）環境では、初期化
//! 失敗を検出した時点でスキップし理由を表示する（`gemm_mma_bench.rs` と
//! 同じ環境適応分岐）。実測値の記録・採否判断は
//! `docs/perf/cuda-gemm-swizzle-ab.md` を参照。
//!
//! `mma.sync` 経路は `n`/`k` が 8 の倍数の形状のみ対応する
//! （`kernels_mma.rs` 冒頭コメント「整列制約」）。本ベンチの形状はすべて
//! この制約を満たす正方形状のみを使う。

use backend_cuda::{CudaDevice, CudaError, CudaMmaGemm, diagnostics};
use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use half::f16;

/// 決定的シード（`gemm_mma_bench.rs` と同一値。過去 PoC・他ベンチと同じ
/// 入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

/// H2D/D2H 転送・出力バッファ確保を計測区間の外へ出し、GPU 実行
/// （カーネル起動 + 同期）のみを計測する（`gemm_mma_bench.rs::
/// measure_mma_f16` と同じ計測方針。base／head 双方に同一方針を適用する
/// ため、`gemm_mma_bench.rs` と異なり本ベンチは apples-to-apples の比較に
/// なる）。
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
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_mma_swizzle_bench: CUDA driver unavailable ({detail}); skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_mma_swizzle_bench: CudaDevice::new failed ({other}); skipping."
            );
            return;
        }
    };

    let base = match CudaMmaGemm::new(&device) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "backend-cuda gemm_mma_swizzle_bench: base mma.sync kernel unavailable ({e}); \
                 nothing to measure. See docs/perf/cuda-gemm-swizzle-ab.md."
            );
            return;
        }
    };

    let num_sms = device.multiprocessor_count().unwrap_or(1).max(1);
    let dynamic_group_width = diagnostics::mma_swizzle_group_width(num_sms);
    println!(
        "num_sms={num_sms} dynamic_group_width={dynamic_group_width} \
         (candidates: {{8, 16}}; see swizzle.rs::select_swizzle_group_width)"
    );

    // 動的選択幅 + 参考として固定候補 8/16（実装計画 3 節）。動的選択が
    // 候補と一致する場合は重複計測になるが、出力の単純さを優先し
    // de-dup はしない（`gemm_mma.rs` の同種の ignored テストと同じ判断）。
    let group_widths = [dynamic_group_width, 8u32, 16u32];

    for size in [512usize, 1024, 2048, 4096] {
        let config = MeasurementConfig::default();
        let base_tflops = measure_mma_f16(&base, size, &config);

        let mut variants: Vec<(u32, Option<f64>)> = Vec::new();
        for &group_width in &group_widths {
            match CudaMmaGemm::new_with_swizzle(&device, group_width) {
                Ok(variant) => {
                    let tflops = measure_mma_f16(&variant, size, &config);
                    variants.push((group_width, Some(tflops)));
                }
                Err(e) => {
                    println!(
                        "size={size} group_width={group_width}: new_with_swizzle failed ({e}); \
                         skipping this variant."
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
