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
//! （切替後にユーザーが体感する経路そのものを計測する狙い）。
//!
//! **本番結線は有効化済み（イシュー #1191。`MMA_PRIORITY_PRODUCTION_
//! ENABLED = true`）**:
//! `select_f16_matrix_unit_impl` の判定ロジックは §5.6 の mma 優先設計
//! どおり実装済みで、これを有効化する `gemm_auto::
//! MMA_PRIORITY_PRODUCTION_ENABLED`（既定 `true`）は #1160 の A/B 実測
//! ・#1190 の K=4096 非後退ゲート `MmaF16` baseline ceiling 承認を経て
//! #1191 で mma 優先を本番有効化した。したがって現在の HEAD で本
//! バイナリをそのままビルドすれば mma 優先経路（after 相当）が計測
//! される。#1191 の再計測は「after」= HEAD（既定 `true`）、「base」=
//! 同定数を一時的に `false` へ書き換えたノード側コピーでそれぞれ
//! ビルドして得る（`docs/perf/cuda-gemm-auto-f16-mma-switch.md` に
//! #1160 時点・#1191 再計測の両方の実測値を記録している）。
//!
//! 計測プロトコル（codex-review PR #1177 指摘の是正。2 回目）: `docs/
//! dispatch-rules-design.md` §5.6・`.claude/rules/coding-rust.md`
//! 「ベンチは 5 回計測の中央値を採用し」が求める「5 回計測中央値」は
//! **独立した 5 回のプロセス起動それぞれの計測値**を指す。前バージョン
//! は本バイナリ内で `measure_auto_f16_once`（`bench_run` 1 回・warmup
//! 20・計測 20）を同一プロセス内ループで 5 回呼ぶ方式を「独立 5 回」と
//! 称していたが、これは同一プロセス内の反復に過ぎずプロセス起動・
//! クロック・熱条件の独立実行間ばらつきを捕捉できない（本 PR 自身が
//! 前段で取り下げた「20 サンプル内 bench_run 1 回」方式と同じ欠陥を
//! プロセス境界で再発させていた、との codex-review 指摘）。
//!
//! 本バイナリは**プロセス起動ごとに各形状 1 回だけ**
//! `measure_auto_f16_once`（`bench_run`。warmup 20・計測 20）を実行し、
//! `size=<N> auto_f16_tflops=<value>` を 1 行ずつ標準出力へ書いて終了
//! する（同一プロセス内でのループ・中央値集約は行わない）。独立 5 回
//! 起動と run-median の集約は、本バイナリを**外側から 5 回起動する**
//! `scripts/bench/run_gemm_auto_f16_mma_switch_bench.sh` が担う（真に
//! 独立したプロセス実行間で TFLOPS を比較するための分離）。
//!
//! `examples/` に置くのは通常の `cargo test`／CI では実行されず
//! ビルド検証のみが走るようにするため（self-hosted runner をベンチ
//! 実行で占有しない。`.claude/rules/ci.md`）。
//!
//! ## 実行手順
//!
//! ```sh
//! # 単発（デバッグ用。5 回計測中央値の対象外）:
//! cargo run -p fandhe-ai-backend-cuda --example gemm_auto_f16_mma_switch_bench --release
//!
//! # 独立 5 回起動・run-median 集約（正式な計測プロトコル）:
//! bash scripts/bench/run_gemm_auto_f16_mma_switch_bench.sh
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

/// `CudaGemmAuto::run_f16` を転送込み（H2D + カーネル起動 + D2H）で 1 回分
/// 計測する（`bench_run` 1 呼び出し＝ウォームアップ 20 回・計測 20 回で
/// 求めた中央値 1 個）。base（結線前）／after（`MMA_PRIORITY_PRODUCTION_
/// ENABLED` を一時的に `true` にしたワークツリー）のどちらでビルドしても
/// 同じ計測境界になる（`run_f16` のシグネチャ・呼び出し規約は変わらない
/// ため）。独立 5 回計測は本関数をプロセスごとに 1 回だけ呼ぶ形（外側の
/// `run_gemm_auto_f16_mma_switch_bench.sh` によるプロセス分離）で担う。
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
         （CudaGemmAuto::run_f16 の実利用経路そのもの）をこのプロセス内で \
         各形状 1 回だけ実行した値（独立 5 回起動・run-median の集約は \
         scripts/bench/run_gemm_auto_f16_mma_switch_bench.sh が担う）。 \
         after（HEAD。既定 true = mma 優先。#1191 で本番有効化済み）／ \
         base（同定数を一時的に false へ書き換えたノード側コピー = \
         wmma 優先）で個別にビルド・実行し、出力を docs/perf/ \
         cuda-gemm-auto-f16-mma-switch.md の表へ手動転記して比較した \
         記録が同ドキュメントに残っている。"
    );

    for size in [512usize, 1024, 2048, 4096] {
        let config = MeasurementConfig::default();
        let value = measure_auto_f16_once(&auto, size, &config);
        println!("size={size} auto_f16_tflops={value:.4}");
    }
}
