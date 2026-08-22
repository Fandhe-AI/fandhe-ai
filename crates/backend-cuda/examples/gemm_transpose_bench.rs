//! smem パディング + スウィズルによる転置カーネル（イシュー #601）の A/B
//! 計測バイナリ。
//!
//! **advisor 指摘の是正**: 当初 `run_*`（H2D/D2H 転送込み）で 3 経路を
//! 直接比較していたが、size=4096 f32 では 1 回あたり 64MB 分の PCIe
//! 転送（数 ms オーダー）がカーネル本体の実行時間（数百 us オーダー）を
//! 支配し、naive/smem パディング/スウィズルの差がすべて転送時間へ埋もれて
//! 比が常に約 1.00 に近い値しか観測できない欠陥があった（受け入れ基準
//! 「改善が実測で確認できなければ不採用と記録して完了」に対し、この計測
//! は偽の「不採用」判定を導きうる）。`gemm_mma_swizzle_bench.rs` と同じ
//! 「H2D/D2H を計測区間の外へ出す」方針に揃え、以下 2 系統で比較する:
//!
//! - **A（transpose 単体）**: 同一デバイス常駐バッファ `c_dev` に対する
//!   `launch_naive_f32`／`launch_smem_f32`（pad／pad+swizzle）を比較する
//!   （[`CudaTranspose::upload_f32`]／[`CudaTranspose::alloc_output_f32`]／
//!   [`CudaTranspose::launch_naive_f32`]／[`CudaTranspose::launch_smem_f32`]）。
//!   GEMM 分を含まないため、パディング・スウィズルそのものの効果を分離
//!   して観測できる。
//! - **B（tiled+transpose 分離 vs 融合）**: `launch_tiled_f32` →
//!   `launch_naive_f32`（分離。両方デバイス常駐入出力）と
//!   `launch_tiled_transposed_f32`（融合）を比較する。両者とも入力は
//!   デバイス常駐済みの同一 `a_dev`/`b_dev` を使うため、差分は「中間
//!   バッファ C の HBM 書き込み・再読み出しの有無」のみに絞られる。
//!
//! CUDA 非搭載・NVRTC 非搭載環境では、初期化失敗を検出した時点でスキップし
//! 理由を表示する（`gemm_mma_swizzle_bench.rs` と同じ環境適応分岐）。
//! nsight-compute によるバンクコンフリクト実測・採否判断は
//! `docs/perf/cuda-gemm-transpose-ab.md` を参照（同ファイルに「GB/s は
//! transpose 単体（系統 A）／tiled+transpose 合成（系統 B）」である旨を
//! 明記する）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-cuda --example gemm_transpose_bench --release
//! ```

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm, CudaTranspose};

/// 決定的シード（`gemm_mma_bench.rs`/`gemm_mma_swizzle_bench.rs` と同じ
/// 値域の入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

fn median_secs<F: FnMut()>(config: &MeasurementConfig, workload: F) -> f64 {
    bench_run(config, workload)
        .expect("MeasurementConfig::default satisfies the 20/20 lower bound")
        .median_secs
}

/// 転置のメモリ帯域（GB/s）。読み込み + 書き込みの合計バイト数を計測時間
/// で割る（転置は演算を伴わない純メモリ移動のため、GEMM の TFLOPS 相当の
/// 指標としてバンド幅を使う）。
fn bandwidth_gbps(m: usize, n: usize, elem_bytes: usize, median_secs: f64) -> f64 {
    let bytes = 2.0 * (m * n * elem_bytes) as f64; // 読み込み + 書き込み
    bytes / median_secs / 1e9
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_transpose_bench: CUDA driver unavailable ({detail}); skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_transpose_bench: CudaDevice::new failed ({other}); skipping."
            );
            return;
        }
    };

    let gemm = match CudaGemm::new(&device) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "backend-cuda gemm_transpose_bench: tiled GEMM kernel unavailable ({e}); \
                 nothing to measure."
            );
            return;
        }
    };
    let transpose = match CudaTranspose::new(&device) {
        Ok(t) => t,
        Err(e) => {
            println!(
                "backend-cuda gemm_transpose_bench: transpose kernels unavailable ({e}); \
                 nothing to measure. See docs/perf/cuda-gemm-transpose-ab.md."
            );
            return;
        }
    };

    let config = MeasurementConfig::default();

    println!("=== 系統 A: transpose 単体（デバイス常駐 c_dev に対する launch_*） ===");
    for size in [512usize, 1024, 2048, 4096] {
        let mut rng_c = Xorshift64Star::new(SEED ^ (size as u64));
        let c: Vec<f32> = rng_c.fill_vec(size * size);
        let c_dev = transpose
            .upload_f32(&c)
            .expect("upload_f32 must succeed on CUDA-equipped runner");

        let mut dst_dev = transpose
            .alloc_output_f32(size as u32, size as u32)
            .expect("alloc_output_f32 must succeed on CUDA-equipped runner");
        let secs_naive = median_secs(&config, || {
            transpose
                .launch_naive_f32(&c_dev, &mut dst_dev, size as u32, size as u32)
                .expect("launch_naive_f32 must succeed on CUDA-equipped runner");
        });

        let secs_smem_pad = median_secs(&config, || {
            transpose
                .launch_smem_f32(&c_dev, &mut dst_dev, size as u32, size as u32, false)
                .expect("launch_smem_f32(pad) must succeed on CUDA-equipped runner");
        });

        let secs_smem_swizzle = median_secs(&config, || {
            transpose
                .launch_smem_f32(&c_dev, &mut dst_dev, size as u32, size as u32, true)
                .expect("launch_smem_f32(pad+swizzle) must succeed on CUDA-equipped runner");
        });

        let bw_naive = bandwidth_gbps(size, size, 4, secs_naive);
        let bw_smem_pad = bandwidth_gbps(size, size, 4, secs_smem_pad);
        let bw_smem_swizzle = bandwidth_gbps(size, size, 4, secs_smem_swizzle);

        println!(
            "size={size} naive_secs={secs_naive:.6} naive_gbps={bw_naive:.2} \
             smem_pad_secs={secs_smem_pad:.6} smem_pad_gbps={bw_smem_pad:.2} \
             smem_swizzle_secs={secs_smem_swizzle:.6} smem_swizzle_gbps={bw_smem_swizzle:.2} \
             smem_pad_over_naive={:.4} smem_swizzle_over_naive={:.4}",
            secs_naive / secs_smem_pad,
            secs_naive / secs_smem_swizzle,
        );
    }

    println!(
        "\n=== 系統 B: tiled+transpose 分離 vs 融合（デバイス常駐 a_dev/b_dev、\
         差分は中間バッファ C の HBM 書き込み・再読み出しの有無のみ） ==="
    );
    for size in [512usize, 1024, 2048, 4096] {
        let mut rng_a = Xorshift64Star::new(SEED ^ (size as u64) ^ 0xA);
        let a: Vec<f32> = rng_a.fill_vec(size * size);
        let mut rng_b = Xorshift64Star::new(SEED ^ (size as u64) ^ 0xB);
        let b: Vec<f32> = rng_b.fill_vec(size * size);

        let (a_dev, b_dev) = gemm
            .upload_f32(&a, &b)
            .expect("upload_f32 must succeed on CUDA-equipped runner");
        let mut c_dev = gemm
            .alloc_output_f32(size as u32, size as u32)
            .expect("alloc_output_f32 must succeed on CUDA-equipped runner");
        let mut c_t_dev = transpose
            .alloc_output_f32(size as u32, size as u32)
            .expect("alloc_output_f32 must succeed on CUDA-equipped runner");

        // 分離経路: launch_tiled_f32（C を書く）→ launch_naive_f32
        // （C を読み C^T を書く）。中間バッファ C は smem 転置基準点
        // （naive）を使い、パディング/スウィズルの効果は系統 A で
        // 既に分離計測しているためここでは再計測しない。
        let secs_separate = median_secs(&config, || {
            gemm.launch_tiled_f32(
                &a_dev,
                &b_dev,
                &mut c_dev,
                size as u32,
                size as u32,
                size as u32,
            )
            .expect("launch_tiled_f32 must succeed on CUDA-equipped runner");
            transpose
                .launch_naive_f32(&c_dev, &mut c_t_dev, size as u32, size as u32)
                .expect("launch_naive_f32 must succeed on CUDA-equipped runner");
        });

        // 融合経路: launch_tiled_transposed_f32（中間バッファ C を HBM へ
        // 書かず、epilogue で smem 経由の転置ストアまで完結させる）。
        let secs_fused = median_secs(&config, || {
            transpose
                .launch_tiled_transposed_f32(
                    &a_dev,
                    &b_dev,
                    &mut c_t_dev,
                    size as u32,
                    size as u32,
                    size as u32,
                )
                .expect("launch_tiled_transposed_f32 must succeed on CUDA-equipped runner");
        });

        println!(
            "size={size} separate_secs={secs_separate:.6} fused_secs={secs_fused:.6} \
             fused_over_separate={:.4}",
            secs_separate / secs_fused,
        );
    }
}
