//! TF32 opt-staged 経路への threadblock swizzle 横展開（イシュー #741）の
//! A/B 計測バイナリ。
//!
//! `kernels_wmma_opt::wmma_tf32_f32_staged_source()`（本番カーネル・
//! 無変更）を使う base（[`CudaGemm::new`]）と、swizzle remap を適用した
//! head 変種（[`CudaGemm::new_with_tf32_staged_swizzle`]。動的選択幅 +
//! 参考として固定候補 `{8, 16}`）を、`gemm_mma_swizzle_bench.rs` と同じ
//! 計測コア（`bench-harness::protocol::run`。warmup/計測 20 回以上・
//! 中央値/Q1/Q3）で比較する（実装計画 3 節「ステップ 6」）。
//!
//! `internal-diagnostics` feature を要求する（`Cargo.toml`
//! `required-features`）。動的グルーピング幅選択
//! （`swizzle::select_swizzle_group_width`）は非公開 `mod swizzle` の
//! 関数であり、crate 外部（本 example）からは
//! `backend_cuda::diagnostics::wmma_tf32_staged_swizzle_group_width`
//! 経由でのみ到達できる（`gemm_mma_swizzle_bench.rs` と同じ feature
//! ゲート方針）。
//!
//! ## 計測境界の是正経緯（イシュー #776）
//!
//! 当初の実装は `run_wmma_tf32`（H2D×2＋`alloc_zeros`＋launch＋sync＋D2H
//! を毎 iteration 行う高水準 API）を計測クロージャに直接使っていた。
//! M=N=K=4096 では毎 iteration 約 192MB のデバイス確保・解放が計測区間に
//! 混入しており、run 間で最大約 16 倍の乖離が観測された（同一カーネルの
//! launch-only 計測である `cuda_floor_bench.rs` は 4096 で安定していた）。
//! **一次仮説（未検証）**: GB10 の unified memory 環境でのアロケータ・
//! ページマッピングの状態依存揺らぎが計測値を支配していた、というのが
//! 有力な原因仮説である。原因はカーネル自体ではなく本ベンチの計測境界
//! 設計にあるとみているが、クロック・温度等の交絡要因は切り分けられて
//! おらず、実機再計測での比較が揃うまで確定原因としては扱わない。是正
//! として `upload_f32`/
//! `alloc_output_f32` を計測ループ外へ出し、`launch_wmma_tf32`
//! （GPU 実行＋synchronize のみ）を計測対象に限定した
//! （`cuda_floor_bench.rs::measure_wmma_tf32`・`gemm_mma_swizzle_bench.rs::
//! measure_mma_f16` と同じ低水準 API 方式へ統一）。この境界変更により
//! 本ベンチの絶対値は是正前（H2D 込み）の記録と比較不能になる。
//! 切り分け過程・是正後の全サイズ実測は
//! `docs/perf/cuda-gemm-swizzle-ab.md` §7.7 を参照。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p backend-cuda --example gemm_wmma_tf32_swizzle_bench --release \
//!     --features internal-diagnostics
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載・cc<8.0（opt-staged 経路の下限）環境、または
//! staged カーネルが `CudaGemm::new` 時点でコンパイル・ロードに失敗した
//! 環境では、[`CudaGemm::wmma_tf32_staged_available`] で可用性を確認して
//! からスキップし理由を表示する（`gemm_mma_swizzle_bench.rs` と同じ環境
//! 適応分岐）。実測値の記録・採否判断は
//! `docs/perf/cuda-gemm-swizzle-ab.md`（#741 節）を参照。
//!
//! staged カーネルは cp.async 16 バイト整列条件（`n % 4 == 0 && k % 4 ==
//! 0`）を満たす形状のみ対応する（`gemm.rs::wmma_tf32_staged_alignment_ok`）。
//! 本ベンチの形状はすべてこの制約を満たす正方形状のみを使う。

use backend_cuda::{CudaDevice, CudaError, CudaGemm, diagnostics};
use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};

/// 決定的シード（`gemm_mma_swizzle_bench.rs` と同一値。過去 PoC・他ベンチと
/// 同じ入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

/// H2D/D2H 転送・出力バッファ確保を計測区間の外へ出し、GPU 実行
/// （カーネル起動＋同期）のみを計測する（`cuda_floor_bench.rs::
/// measure_wmma_tf32`・`gemm_mma_swizzle_bench.rs::measure_mma_f16` と
/// 同じ計測方針。base／head 双方に同一方針を適用するため
/// apples-to-apples の比較になる）。イシュー #776: 是正前は高水準 API
/// `run_wmma_tf32`（毎 iteration H2D×2＋`alloc_zeros`＋D2H を含む）を
/// 計測クロージャに直接使っており、M=N=K=4096 で毎 iteration 約 192MB の
/// デバイス確保・解放が計測値を支配し run 間で最大約 16 倍の乖離が生じて
/// いた（本ファイル冒頭コメント参照）。`run_wmma_tf32` は 1 回だけ probe
/// 実行して可用性を確認する（`launch_wmma_tf32` の前提契約。`gemm.rs`
/// 該当ドキュメンテーションコメント参照）。
fn measure_wmma_tf32(gemm: &CudaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);
    let (m, n, k) = (size as u32, size as u32, size as u32);

    gemm.run_wmma_tf32(&a, &b, m, n, k)
        .expect("wmma_tf32 probe must succeed on CUDA-equipped runner");

    let (a_dev, b_dev) = gemm
        .upload_f32(&a, &b)
        .expect("WMMA(TF32) upload must succeed on CUDA-equipped runner (probed above)");
    let mut c_dev = gemm
        .alloc_output_f32(m, n)
        .expect("WMMA(TF32) output allocation must succeed on CUDA-equipped runner (probed above)");

    let measurement = bench_run(config, || {
        gemm.launch_wmma_tf32(&a_dev, &b_dev, &mut c_dev, m, n, k)
            .expect("WMMA(TF32) GEMM must succeed on CUDA-equipped runner (probed above)");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_wmma_tf32_swizzle_bench: CUDA driver unavailable ({detail}); skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_wmma_tf32_swizzle_bench: CudaDevice::new failed ({other}); skipping."
            );
            return;
        }
    };

    let base = match CudaGemm::new(&device) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "backend-cuda gemm_wmma_tf32_swizzle_bench: CudaGemm::new failed ({e}); \
                 nothing to measure. See docs/perf/cuda-gemm-swizzle-ab.md."
            );
            return;
        }
    };

    if !base.wmma_tf32_staged_available() {
        let reason = base
            .wmma_tf32_staged_unavailable_reason()
            .unwrap_or("unknown");
        println!(
            "backend-cuda gemm_wmma_tf32_swizzle_bench: TF32 opt-staged kernel unavailable \
             ({reason}); nothing to measure. See docs/perf/cuda-gemm-swizzle-ab.md."
        );
        return;
    }

    let num_sms = device.multiprocessor_count().unwrap_or(1).max(1);
    let dynamic_group_width = diagnostics::wmma_tf32_staged_swizzle_group_width(num_sms);
    println!(
        "num_sms={num_sms} dynamic_group_width={dynamic_group_width} \
         (candidates: {{8, 16}}; see swizzle.rs::select_swizzle_group_width)"
    );

    // 動的選択幅 + 参考として固定候補 8/16（`gemm_mma_swizzle_bench.rs` と
    // 同じ方針）。動的選択が候補と一致する場合は重複計測になるが、出力の
    // 単純さを優先し de-dup はしない。
    let group_widths = [dynamic_group_width, 8u32, 16u32];

    for size in [512usize, 1024, 2048, 4096] {
        let config = MeasurementConfig::default();
        let base_tflops = measure_wmma_tf32(&base, size, &config);

        let mut variants: Vec<(u32, Option<f64>)> = Vec::new();
        for &group_width in &group_widths {
            match CudaGemm::new_with_tf32_staged_swizzle(&device, group_width) {
                Ok(variant) => {
                    let tflops = measure_wmma_tf32(&variant, size, &config);
                    variants.push((group_width, Some(tflops)));
                }
                Err(e) => {
                    println!(
                        "size={size} group_width={group_width}: \
                         new_with_tf32_staged_swizzle failed ({e}); skipping this variant."
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
