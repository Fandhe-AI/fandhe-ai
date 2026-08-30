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
//! **計測環境の注記（重要）**: 本ハーネスの実行環境はローカル QEMU
//! x86_64 であり、REQ-8 の正式対象実機（Apple M4 Max。`docs/perf/
//! gemm-optimization-baseline.md` §3）とは異なる。閾値の最終確定には
//! M4 Max 実機での再スイープを要する（`docs/perf/
//! cpu-parallel-threshold-sweep.md` に記録する残課題）。
//!
//! 実行例:
//! ```text
//! cargo test -p fandhe-ai-backend-cpu --release -- --ignored elementwise_threshold_sweep
//! ```

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run};
use fandhe_ai_backend_cpu::{add_slice, exp_slice, mul_slice};

fn random_vec(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

/// `add_slice`／`mul_slice`／`exp_slice` は `PARALLEL_THRESHOLD` の判定を
/// 内部で行う公開関数のため、逐次専用・並列専用の版を用意せず「小さい
/// 要素数（逐次側）」と「大きい要素数（並列側）」の実測値を並べて比較する
/// （`gemm_small_shape_perf.rs` が `gemm_blis`〈直列専用入口〉と
/// `gemm_blis_parallel`〈自動判定入口〉を並べて比較するのと同じ発想だが、
/// elementwise 層には直列専用の公開関数が無いため、要素数のスイープ
/// そのもので閾値前後の挙動差を捉える）。
fn measure_size(label: &str, n: usize) {
    let a = random_vec(7000 + n as u64, n);
    let b = random_vec(8000 + n as u64, n);
    let config = MeasurementConfig::default(); // warmup 20・iters 20（TASK-8.1 下限）

    let mut out_add = vec![0.0f32; n];
    let add = run(&config, || {
        add_slice(&a, &b, &mut out_add);
    })
    .expect("add_slice の計測に失敗");

    let mut out_mul = vec![0.0f32; n];
    let mul = run(&config, || {
        mul_slice(&a, &b, &mut out_mul);
    })
    .expect("mul_slice の計測に失敗");

    let mut out_exp = vec![0.0f32; n];
    let exp = run(&config, || {
        exp_slice(&a, &mut out_exp);
    })
    .expect("exp_slice（libm 経由）の計測に失敗");

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
#[test]
#[ignore = "実測ハーネス（--release 推奨）。通常 CI では実行しない"]
fn elementwise_threshold_sweep() {
    for exp2 in 12..=18u32 {
        let n = 1usize << exp2;
        measure_size(&format!("2^{exp2}"), n);
    }
}
