//! `gemm_tiled.rs::tiled_f32_outperforms_naive_at_4096` の新規 FAIL
//! （speedup=0.235x。#1162 の GB10 実機 sweep・`--test-threads=1` 直列条件）
//! を切り分けるための診断専用テスト群（イシュー #1203）。
//!
//! ## 位置づけ・引き継ぎ元
//!
//! `docs/backend-cuda-real-device-testing.md` §5.1（2026-08-10）は同テストの
//! FAIL を「同一バイナリ内並列実行による GPU 時間分割」と結論づけていたが、
//! #1162 は `--test-threads=1` 直列条件で FAIL しており、その説明では今回の
//! 事象を説明できない。`run_naive_f32`／`run_tiled_f32` は
//! `crates/backend-cuda/src/gemm.rs::run_f32_kernel` を共有ホスト経路として
//! 使う（差分はカーネルハンドルと `LaunchConfig` のみ）ため、4.2 倍の差の
//! 発生源は次の 2 つに限られる:
//!
//! - **(a)** tiled カーネル自体（cp.async パイプライン化・#1137/#1164/#1344
//!   の 128×64 結線）が 4096 で遅い
//! - **(b)** 共有ホスト経路（H2D／プール確保／readback）が「64 MiB D2H
//!   二峰性の slow モード」（`docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`
//!   §11。M=N=K=4096・f32 は 1 行列 64 MiB で該当範囲）に tiled 側の計測
//!   中だけ入った（環境要因）
//!
//! 本ファイルはこの 2 つを分離するための計測のみを行う（**結論を先取り
//! しない**）。切り分け結果・判定は `docs/perf/cuda-gemm-tiled-naive-speedup-4096-triage.md`
//! にまとめる。
//!
//! ## 対策コードを含まない・本番非変更
//!
//! 本ファイルは**計測・記録専用**。`crates/backend-cuda/src/**` の
//! プロダクションコードは一切変更しない。assert は「実行成功・形状」の
//! みに限り、性能アサーションは持たない（判定は診断出力を人間・後続
//! セッションが読んで行う）。
//!
//! ## 実機前提・feature ゲート
//!
//! `run_tiled_f32_classic`（`internal-diagnostics` feature 限定。
//! `gemm.rs` 参照）を使う診断 (3) を含むため、本ファイル全体を
//! `internal-diagnostics` feature 限定にする（`Cargo.toml` の
//! `required-features` エントリ参照。他の `*_1203` 系ファイルと同様、
//! `cargo test --workspace`（feature 未指定）では対象外としてスキップ
//! され、`cargo test --all-features` でのみビルド・実行される）。
//! 通常 CI（GitHub ホステッド・CUDA 実機なし）では `#[ignore]` により
//! 実行されない。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --locked \
//!     --features internal-diagnostics \
//!     --test gemm_tiled_naive_speedup_triage_1203 \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--release`・`--test-threads=1` 必須（#1162 と同じ計測条件を揃える）。

#![cfg(feature = "internal-diagnostics")]

use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm};

const M: u32 = 4096;
const N: u32 = 4096;
const K: u32 = 4096;
const WARMUP: usize = 2;
const SAMPLES: usize = 5;

type GemmRunFn<'a> = dyn Fn(&[f32], &[f32], u32, u32, u32) -> Result<Vec<f32>, CudaError> + 'a;

/// naive/tiled 双方に共通の計測ヘルパー。`gemm_tiled.rs` の `measure` と
/// 同一プロトコル（warmup 2 回・計測 5 回・`run_*` 呼び出し全体を計測）
/// だが、判定を持たず生サンプル列をそのまま返す（呼び出し側で
/// 順序・組合せを変えて診断するための共通部品）。
fn measure_samples(a: &[f32], b: &[f32], run: &GemmRunFn<'_>) -> Vec<f64> {
    for _ in 0..WARMUP {
        run(a, b, M, N, K).expect("warmup run must succeed on CUDA-equipped test runner");
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = std::time::Instant::now();
        run(a, b, M, N, K).expect("measured run must succeed on CUDA-equipped test runner");
        samples.push(start.elapsed().as_secs_f64());
    }
    samples
}

fn eprint_samples(label: &str, samples: &[f64]) {
    let q = bench_harness::median_q1_q3(samples).expect("5 non-NaN samples must yield quartiles");
    eprintln!(
        "[#1203 triage] {label}: samples={samples:?}s median={:.6}s q1={:.6}s q3={:.6}s",
        q.median, q.q1, q.q3
    );
}

fn make_inputs(seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec((M as usize) * (K as usize));
    let b = rng.fill_vec((K as usize) * (N as usize));
    (a, b)
}

/// 診断 (1): 呼び出し順序を tiled→naive へ反転する。#1146 §4.4 が記録した
/// 「順序依存スパイク」（先行呼び出しが後続の計測に影響する可能性）の
/// 有無を、naive 先行を前提にした本番テストと比較して見る。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須（イシュー #1203 診断専用）"]
fn triage_reversed_order_tiled_then_naive() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive/tiled kernel compilation must succeed");
    let (a, b) = make_inputs(31415);

    let tiled = measure_samples(&a, &b, &|a, b, m, n, k| gemm.run_tiled_f32(a, b, m, n, k));
    eprint_samples("tiled(1st)", &tiled);
    let naive = measure_samples(&a, &b, &|a, b, m, n, k| gemm.run_naive_f32(a, b, m, n, k));
    eprint_samples("naive(2nd)", &naive);

    let tiled_median = bench_harness::median_q1_q3(&tiled).unwrap().median;
    let naive_median = bench_harness::median_q1_q3(&naive).unwrap().median;
    eprintln!(
        "[#1203 triage] reversed_order speedup(naive/tiled)={:.6}x",
        naive_median / tiled_median
    );

    assert_eq!(tiled.len(), SAMPLES);
    assert_eq!(naive.len(), SAMPLES);
}

/// 診断 (2): naive/tiled をペアごとにインターリーブして交互計測する。
/// 両者を同じ確率で「slow モード」に曝すことで、tiled 側だけが不利な
/// タイミングに当たっているのか（(b) 環境要因の傍証）を見る。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須（イシュー #1203 診断専用）"]
fn triage_interleaved_naive_tiled() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive/tiled kernel compilation must succeed");
    let (a, b) = make_inputs(31415);

    // warmup（各カーネル 2 回ずつ、本番テストと同じ回数）。
    for _ in 0..WARMUP {
        gemm.run_naive_f32(&a, &b, M, N, K)
            .expect("naive warmup must succeed");
        gemm.run_tiled_f32(&a, &b, M, N, K)
            .expect("tiled warmup must succeed");
    }

    let mut naive_samples = Vec::with_capacity(SAMPLES);
    let mut tiled_samples = Vec::with_capacity(SAMPLES);
    for i in 0..SAMPLES {
        let start = std::time::Instant::now();
        gemm.run_naive_f32(&a, &b, M, N, K)
            .expect("naive measured run must succeed");
        let naive_elapsed = start.elapsed().as_secs_f64();

        let start = std::time::Instant::now();
        gemm.run_tiled_f32(&a, &b, M, N, K)
            .expect("tiled measured run must succeed");
        let tiled_elapsed = start.elapsed().as_secs_f64();

        eprintln!(
            "[#1203 triage] interleaved pair {i}: naive={naive_elapsed:.6}s tiled={tiled_elapsed:.6}s"
        );
        naive_samples.push(naive_elapsed);
        tiled_samples.push(tiled_elapsed);
    }
    eprint_samples("naive(interleaved)", &naive_samples);
    eprint_samples("tiled(interleaved)", &tiled_samples);

    let naive_median = bench_harness::median_q1_q3(&naive_samples).unwrap().median;
    let tiled_median = bench_harness::median_q1_q3(&tiled_samples).unwrap().median;
    eprintln!(
        "[#1203 triage] interleaved speedup(naive/tiled)={:.6}x",
        naive_median / tiled_median
    );

    assert_eq!(naive_samples.len(), SAMPLES);
    assert_eq!(tiled_samples.len(), SAMPLES);
}

/// 診断 (3): 本番ディスパッチ（`run_tiled_f32`。#1385 以降 4096 では
/// 128×64 pipeline カーネルを選択）と、`internal-diagnostics` feature
/// 限定で常に classic 版（`kernels::TILED_F32`。cp.async パイプライン
/// 非経由）へ固定した `run_tiled_f32_classic` を同一プロトコルで比較
/// する。classic 側にも同程度の slow モードが出れば、pipeline 結線
/// （#1137/#1164/#1344）はこの FAIL の原因から除外できる（(b) 環境要因
/// 説を補強）。逆に classic が速く本番 pipeline だけが遅ければ (a) 結線
/// 劣化の疑いが残る。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須（イシュー #1203 診断専用）"]
fn triage_classic_vs_pipeline_tiled() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive/tiled kernel compilation must succeed");
    let (a, b) = make_inputs(31415);

    let pipeline = measure_samples(&a, &b, &|a, b, m, n, k| gemm.run_tiled_f32(a, b, m, n, k));
    eprint_samples("tiled_pipeline(production)", &pipeline);
    let classic = measure_samples(&a, &b, &|a, b, m, n, k| {
        gemm.run_tiled_f32_classic(a, b, m, n, k)
    });
    eprint_samples("tiled_classic(diagnostics)", &classic);

    let pipeline_median = bench_harness::median_q1_q3(&pipeline).unwrap().median;
    let classic_median = bench_harness::median_q1_q3(&classic).unwrap().median;
    eprintln!(
        "[#1203 triage] classic_vs_pipeline ratio(pipeline/classic)={:.6}x \
         (>1 means production pipeline is slower than classic)",
        pipeline_median / classic_median
    );

    assert_eq!(pipeline.len(), SAMPLES);
    assert_eq!(classic.len(), SAMPLES);
}
