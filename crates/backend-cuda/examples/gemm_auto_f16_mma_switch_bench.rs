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
//! `0c91218`）／after（`CudaGemmAuto::run_f16` が実際に mma 優先経路を
//! 通る構成）でそれぞれビルド・実行し、出力される run-median TFLOPS を
//! 手動で突き合わせる）。
//!
//! **本番既定は wmma 優先のまま（codex-review PR #1177 P1 是正）**:
//! `select_f16_matrix_unit_impl` の判定ロジック自体は §5.6 の mma 優先
//! 設計どおり実装済みだが、`gemm_auto::MMA_PRIORITY_PRODUCTION_ENABLED`
//! （既定 `false`）がこの優先順位を無効化しているため、本 PR（#1177）
//! 時点の HEAD で本バイナリを素朴にビルドしても base と同じ wmma 優先
//! 経路が計測されるだけで、mma 優先経路の A/B 比較にならない。#1160 が
//! 「after」を計測する際は、`crates/backend-cuda/src/gemm_auto.rs` の
//! `MMA_PRIORITY_PRODUCTION_ENABLED` を一時的に `true` へ書き換えた
//! ワークツリーでビルドし、非後退確認後にこの記録（`docs/perf/
//! cuda-gemm-auto-f16-mma-switch.md`）へ実測値を残したうえで、承認を
//! 経てから同定数の恒久的な `true` 化を別途行う。
//!
//! 計測プロトコル（codex-review PR #1177 指摘の是正）: `docs/dispatch-
//! rules-design.md` §5.6 が求める「5 回計測中央値」は**独立した 5 回の
//! 計測それぞれの中央値**を指す（`.claude/rules/coding-rust.md`「ベンチ
//! は 5 回計測の中央値を採用し」も同義）。単一 `bench_run` 呼び出し内の
//! 20 サンプルから求めた中央値は「1 回の計測」に過ぎず、これを「5 回
//! 計測中央値の上位互換」とみなしていた前バージョンは要求を満たさない
//! （20 サンプルは同一プロセス内の反復であり、独立実行間のばらつき
//! （プロセス起動・クロック挙動・熱条件等）を捕捉できないため）。
//! 本バイナリは各形状で `bench_run`（`MeasurementConfig::default`。
//! warmup 20 回・計測 20 回。REQ-8 確定実測用下限プロトコル）を
//! **独立に 5 回**呼び出し、5 個の run 中央値 TFLOPS を個別に出力した
//! うえで、`bench_harness::median_q1_q3` でその 5 値自体の中央値
//! （run-median）を算出して報告する。前後比較にはこの run-median を
//! 用いる。
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
use bench_harness::{MeasurementConfig, median_q1_q3, run as bench_run};
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemmAuto};
use half::f16;

/// 決定的シード（`gemm_mma_bench.rs`・`backend-metal/examples/gemm_bench.rs`
/// と同一値。過去 PoC・他バックエンドベンチと同じ入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

/// 独立計測の回数（`.claude/rules/coding-rust.md`「ベンチは 5 回計測の
/// 中央値を採用し」・`docs/dispatch-rules-design.md` §5.6 の承認条件に
/// 対応。各回は `bench_run`〈warmup 20・計測 20〉を独立に 1 回実行する）。
const INDEPENDENT_RUNS: usize = 5;

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

/// `CudaGemmAuto::run_f16` を転送込み（H2D + カーネル起動 + D2H）で 1 回分
/// 計測する（`bench_run` 1 呼び出し＝ウォームアップ 20 回・計測 20 回で
/// 求めた中央値 1 個）。base（結線前）／after（`MMA_PRIORITY_PRODUCTION_
/// ENABLED` を一時的に `true` にしたワークツリー）のどちらでビルドしても
/// 同じ計測境界になる（`run_f16` のシグネチャ・呼び出し規約は変わらない
/// ため）。
fn measure_auto_f16_once(auto: &CudaGemmAuto, size: usize, config: &MeasurementConfig) -> f64 {
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

/// `measure_auto_f16_once` を [`INDEPENDENT_RUNS`] 回独立に実行し、各回の
/// TFLOPS 一覧と、その一覧自体の中央値（run-median）を返す。
///
/// 各回内部（`bench_run`）で既に中央値化されているため、ここで求める
/// 中央値は「中央値の中央値」ではなく「独立試行間ばらつきに対する
/// 中央値」であり、`docs/dispatch-rules-design.md` §5.6 が求める
/// 「5 回計測中央値」そのものに対応する。
fn measure_auto_f16_runs(
    auto: &CudaGemmAuto,
    size: usize,
    config: &MeasurementConfig,
) -> (Vec<f64>, f64) {
    let per_run: Vec<f64> = (0..INDEPENDENT_RUNS)
        .map(|_| measure_auto_f16_once(auto, size, config))
        .collect();
    let run_median = median_q1_q3(&per_run)
        .expect("INDEPENDENT_RUNS 個の TFLOPS 値は非空かつ NaN を含まない")
        .median;
    (per_run, run_median)
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
        "NOTE: auto_f16_tflops_run_median は H2D/D2H・カーネル起動を含む \
         転送込み計測（CudaGemmAuto::run_f16 の実利用経路そのもの）を \
         独立に {INDEPENDENT_RUNS} 回実行した run 中央値。本バイナリを \
         base（0c91218・結線前 = wmma 優先）／after（HEAD の \
         gemm_auto::MMA_PRIORITY_PRODUCTION_ENABLED を一時的に true へ \
         書き換えたワークツリー = mma 優先）で個別にビルド・実行し、出力を \
         docs/perf/cuda-gemm-auto-f16-mma-switch.md の表へ手動転記して \
         比較する（HEAD をそのままビルドした場合は本番既定〈false〉のため \
         base と同じ wmma 優先経路が計測される点に注意）。"
    );

    for size in [512usize, 1024, 2048, 4096] {
        let config = MeasurementConfig::default();
        let (per_run, run_median) = measure_auto_f16_runs(&auto, size, &config);
        let per_run_str = per_run
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "size={size} auto_f16_tflops_runs=[{per_run_str}] \
             auto_f16_tflops_run_median={run_median:.4}"
        );
    }
}
