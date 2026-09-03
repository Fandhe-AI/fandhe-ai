//! `CudaGemmAuto::run_f16` の `MatrixUnit` 分岐 mma 優先化（#1156）の
//! 切替前後比較ベンチ実測バイナリ。
//!
//! `docs/dispatch-rules-design.md` §5.6 と #1156 のユーザー承認条件
//! （「切替前後を同一プロトコル・5 回計測中央値で比較し、後退時は結線
//! しない」）に対応する実測手段。`docs/perf/cuda-gemm-auto-f16-mma-
//! switch.md`（未実測明記だった前バージョン。本実測で更新する）が
//! 求めていた「auto 経路（転送込み）」の計測に限定し、`examples/
//! gemm_mma_bench.rs::measure_mma_f16`（GPU 実行のみ計測）とは異なり
//! `CudaGemmAuto::run_f16` の H2D／カーネル起動／D2H を丸ごと計測する
//! （切替後にユーザーが体感する経路そのものを計測する狙い。本ファイル
//! 単体では前後比較を行わない: 同一バイナリを base（結線前コミット
//! `0c91218`）／HEAD（結線後コミット。`CudaGemmAuto::run_f16` の
//! `MatrixUnit` 分岐を mma 優先へ切替済み）でそれぞれビルド・実行し、
//! 出力される TFLOPS 中央値を手動で突き合わせる）。
//!
//! 計測コアは `bench-harness::protocol::run`（`MeasurementConfig::default`。
//! warmup 20 回・計測 20 回。REQ-8 確定実測用下限プロトコルで
//! 「5 回計測中央値」の上位互換。`gemm_mma_bench.rs` と同じ判断）。
//!
//! `examples/` に置くのは通常の `cargo test`／CI では実行されず
//! ビルド検証のみが走るようにするため（self-hosted runner をベンチ
//! 実行で占有しない。`.claude/rules/ci.md`）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-cuda --example gemm_auto_f16_mma_switch_bench --release
//! ```
//!
//! CUDA 非搭載環境・`CudaGemmAuto::new` 失敗環境では理由を表示して終了する
//! （`gemm_mma_bench.rs` と同じ環境適応分岐）。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemmAuto};
use half::f16;

/// 決定的シード（`gemm_mma_bench.rs`・`backend-metal/examples/gemm_bench.rs`
/// と同一値。過去 PoC・他バックエンドベンチと同じ入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

/// `CudaGemmAuto::run_f16` を転送込み（H2D + カーネル起動 + D2H）で計測する。
/// base（結線前）／HEAD（結線後）のどちらでビルドしても同じ計測境界になる
/// （`run_f16` のシグネチャ・呼び出し規約は本切替で変わらないため）。
fn measure_auto_f16(auto: &CudaGemmAuto, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f16> = rng.fill_vec_f16(size * size);
    let b: Vec<f16> = rng.fill_vec_f16(size * size);

    let measurement = bench_run(config, || {
        auto.run_f16(&a, &b, size as u32, size as u32, size as u32)
            .expect("CudaGemmAuto::run_f16 must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_auto_f16_mma_switch_bench: CUDA driver unavailable \
                 ({detail}); skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_auto_f16_mma_switch_bench: CudaDevice::new failed \
                 ({other}); skipping."
            );
            return;
        }
    };

    let auto = match CudaGemmAuto::new(&device) {
        Ok(a) => a,
        Err(e) => {
            println!(
                "backend-cuda gemm_auto_f16_mma_switch_bench: CudaGemmAuto::new failed \
                 ({e}); skipping."
            );
            return;
        }
    };

    println!(
        "NOTE: auto_f16_tflops は H2D/D2H・カーネル起動を含む転送込み計測 \
         （CudaGemmAuto::run_f16 の実利用経路そのもの）。本バイナリを base \
         （0c91218・結線前 = wmma 優先）／HEAD（結線後 = mma 優先）で個別に \
         ビルド・実行し、出力を docs/perf/cuda-gemm-auto-f16-mma-switch.md \
         の表へ手動転記して比較する。"
    );

    for size in [512usize, 1024, 2048, 4096] {
        let config = MeasurementConfig::default();
        let tflops = measure_auto_f16(&auto, size, &config);
        println!("size={size} auto_f16_tflops={tflops:.4}");
    }
}
