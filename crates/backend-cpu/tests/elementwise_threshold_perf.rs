//! elementwise（`add`／`mul`／`exp`）の rayon 並列化閾値
//! （[`fandhe_ai_backend_cpu::elementwise`] の `PARALLEL_THRESHOLD =
//! 1 << 15`。`pub(crate)` のため本ハーネスからは直接参照できず、要素数
//! 2^12〜2^18 のスイープで実測から交差点を確認する）のスイープ実測
//! ハーネス（イシュー #1027）。
//!
//! 既存の `PARALLEL_THRESHOLD` は softmax／rmsnorm／`fused_elementwise`
//! （`crate::elementwise::PARALLEL_THRESHOLD` を再利用する既存パターン）
//! にも共有されるため値の変更は影響範囲が広い。本ハーネスは値そのものの
//! 変更判断ではなく、既存値の妥当性を実測で裏付ける・乖離があれば
//! 記録するための計測専用ツールとして追加する（`gemm_small_shape_perf.rs`
//! の出力形式〈`key=value` 列挙〉を踏襲）。
//!
//! # 本ハーネスの範囲（公開 API のみ・PR #1066 codex-review P1 対応）
//!
//! 本ファイルは integration test（外部クレート視点）のため、公開 API の
//! 自動判定版（`add_slice`／`mul_slice`／`exp_slice`。`PARALLEL_THRESHOLD`
//! で経路が切り替わる）のスイープのみを扱う。要素数増加の影響と並列化
//! オーバーヘッドの影響を切り分ける同一サイズ直列 vs 並列比較
//! （`PARALLEL_THRESHOLD` 判定を経由しない逐次／並列強制版による計測。
//! codex-review 指摘・イシュー #1027）は、強制版を公開面に出さないため
//! `crates/backend-cpu/src/elementwise.rs` の `#[cfg(test)] mod
//! bench_internal` 単体テスト（`elementwise_serial_vs_parallel_sweep`）へ
//! 移設した。両スイープの結果は `docs/perf/cpu-parallel-threshold-sweep.md`
//! で突合する。
//!
//! **計測環境の注記（重要）**: 本ハーネスの実行環境はローカル QEMU
//! x86_64 であり、REQ-8 の正式対象実機（Apple M4 Max。`docs/perf/
//! gemm-optimization-baseline.md` §3）とは異なる。閾値の最終確定には
//! M4 Max 実機での再スイープを要する（`docs/perf/
//! cpu-parallel-threshold-sweep.md` に記録する残課題）。
//!
//! 実行例:
//! ```text
//! cargo test -p fandhe-ai-backend-cpu --release -- --ignored elementwise_threshold_sweep
//! cargo test -p fandhe-ai-backend-cpu --release -- --ignored elementwise_serial_vs_parallel_sweep
//! ```

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run};
use fandhe_ai_backend_cpu::{add_slice, exp_slice, mul_slice};
use std::hint::black_box;

fn random_vec(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

/// 公開 API の自動判定版（`PARALLEL_THRESHOLD` 判定を内蔵する
/// `add_slice`／`mul_slice`／`exp_slice`）を要素数ごとに計測する。
/// 異なる要素数間の比較であり要素数増加の影響と並列化オーバーヘッドの
/// 影響が混在する点に注意（切り分けはモジュール doc「本ハーネスの範囲」の
/// とおり `elementwise.rs` 側の単体テストスイープを参照）。
fn measure_size(label: &str, n: usize) {
    let a = random_vec(7000 + n as u64, n);
    let b = random_vec(8000 + n as u64, n);
    let config = MeasurementConfig::default(); // warmup 20・iters 20（TASK-8.1 下限）

    // `run` はクロージャ呼び出し自体を `black_box` するのみで、クロージャ内部の
    // 入出力までは保護しない（`bench_harness::protocol::run` ドキュメンテーション
    // コメント参照）。release/LTO で `a`／`b`／出力バッファが定数畳み込み・
    // dead code 除去の対象にならないよう、計測対象の入出力を明示的に
    // `black_box` で保護する（codex-review 指摘・イシュー #1027）。
    let mut out_add = vec![0.0f32; n];
    let add = run(&config, || {
        add_slice(black_box(&a), black_box(&b), black_box(&mut out_add));
    })
    .expect("add_slice の計測に失敗");
    black_box(&out_add);

    let mut out_mul = vec![0.0f32; n];
    let mul = run(&config, || {
        mul_slice(black_box(&a), black_box(&b), black_box(&mut out_mul));
    })
    .expect("mul_slice の計測に失敗");
    black_box(&out_mul);

    let mut out_exp = vec![0.0f32; n];
    let exp = run(&config, || {
        exp_slice(black_box(&a), black_box(&mut out_exp));
    })
    .expect("exp_slice（libm 経由）の計測に失敗");
    black_box(&out_exp);

    println!(
        "label={label} n={n} add_median_ns={add_ns:.1} mul_median_ns={mul_ns:.1} \
         exp_median_ns={exp_ns:.1}",
        add_ns = add.median_secs * 1e9,
        mul_ns = mul.median_secs * 1e9,
        exp_ns = exp.median_secs * 1e9,
    );
}

/// `PARALLEL_THRESHOLD = 1 << 15 = 32,768` を挟む要素数 2^12〜2^18 で
/// `add`／`mul`／`exp` を計測し、標準出力へ `key=value` 形式で記録する
/// （docs 側〈`docs/perf/cpu-parallel-threshold-sweep.md`〉での突合を
/// 容易にするため）。
///
/// 自動判定版（`PARALLEL_THRESHOLD` で経路が切り替わる）のスイープであり、
/// 要素数増加の影響と並列化オーバーヘッドの影響が混在する。切り分けた
/// 実測は `elementwise.rs` の `#[cfg(test)] mod bench_internal` にある
/// `elementwise_serial_vs_parallel_sweep`（単体テスト）を参照。
#[test]
#[ignore = "実測ハーネス（--release 推奨）。通常 CI では実行しない"]
fn elementwise_threshold_sweep() {
    for exp2 in 12..=18u32 {
        let n = 1usize << exp2;
        measure_size(&format!("2^{exp2}"), n);
    }
}
