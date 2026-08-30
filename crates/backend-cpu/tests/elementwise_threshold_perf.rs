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
//! # 同一サイズ直列 vs 並列比較（codex-review 指摘・イシュー #1027）
//!
//! 当初の実装は `add_slice`／`mul_slice`／`exp_slice`（`PARALLEL_THRESHOLD`
//! による自動判定を内蔵する公開関数）を要素数 2^12〜2^18 でスイープする
//! だけで、閾値未満（逐次）・閾値以上（並列）を異なる要素数間で比較して
//! いた。これでは 2^14→2^15 の所要時間増加が「並列化オーバーヘッド」に
//! よるものか「要素数が単純に倍増した」ことによるものかを切り分けられず、
//! `docs/perf/cpu-parallel-threshold-sweep.md` の「並列化が明確に不利」等の
//! 判断を実測として裏付けられていなかった（P2 指摘）。
//!
//! この切り分けのため、`elementwise` モジュールへ `PARALLEL_THRESHOLD`
//! 判定を経由しない逐次強制版・並列強制版
//! （`add_slice_force_serial`／`add_slice_force_parallel` 等。本体の
//! `*_slice` 関数と同一ロジックから分岐のみを除いたもの。`gemm_blis`
//! 〈直列専用入口〉／`gemm_blis_parallel`〈並列専用入口〉と同じ発想）を
//! 追加し（公開 API 契約外の `#[doc(hidden)]` な `bench_internal`
//! モジュール。PR #1066 codex-review P1 対応）、本ハーネスは各候補サイズで
//! 両経路を**同一サイズ**で計測して
//! 中央値比（並列 / 逐次）を報告する。1.0 に近い・下回るサイズが
//! 「並列化が逐次に追いつく／上回る」交差点であり、これと
//! `PARALLEL_THRESHOLD` の関係を突合できる。
//!
//! 自動判定版（`add_slice` 等）のスイープも従来どおり残し、本番の閾値
//! 分岐が実際にどちらの経路を選ぶかの参考値として併記する。
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
// 逐次／並列強制版は `#[doc(hidden)]` の `bench_internal` モジュール経由で
// 参照する（公開 API 契約外のベンチ専用面。PR #1066 codex-review P1 対応）。
use fandhe_ai_backend_cpu::bench_internal::{
    add_slice_force_parallel, add_slice_force_serial, exp_slice_force_parallel,
    exp_slice_force_serial, mul_slice_force_parallel, mul_slice_force_serial,
};
use fandhe_ai_backend_cpu::{add_slice, exp_slice, mul_slice};
use std::hint::black_box;

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
///
/// 異なる要素数間の比較であり要素数増加の影響と並列化オーバーヘッドの
/// 影響が混在する点に注意（切り分けは
/// [`measure_size_serial_vs_parallel`] を参照）。
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

/// 同一要素数で逐次強制版・並列強制版を測り、中央値比（並列 / 逐次。
/// 1.0 未満＝並列が有利）を報告する（モジュール doc「同一サイズ直列 vs
/// 並列比較」参照）。`measure_size` と異なりサイズを固定するため、
/// 要素数増加の影響を排除して並列化オーバーヘッドそのものを捉える。
fn measure_size_serial_vs_parallel(label: &str, n: usize) {
    let a = random_vec(7000 + n as u64, n);
    let b = random_vec(8000 + n as u64, n);
    let config = MeasurementConfig::default();

    macro_rules! measure_pair {
        ($name:literal, $serial_fn:expr, $parallel_fn:expr, $out_len:expr) => {{
            let mut out_s = vec![0.0f32; $out_len];
            let serial = run(&config, || {
                $serial_fn(black_box(&a), black_box(&b), black_box(&mut out_s));
            })
            .expect(concat!($name, " 逐次強制版の計測に失敗"));
            black_box(&out_s);

            let mut out_p = vec![0.0f32; $out_len];
            let parallel = run(&config, || {
                $parallel_fn(black_box(&a), black_box(&b), black_box(&mut out_p));
            })
            .expect(concat!($name, " 並列強制版の計測に失敗"));
            black_box(&out_p);

            (serial.median_secs, parallel.median_secs)
        }};
    }

    let (add_serial_secs, add_parallel_secs) = measure_pair!(
        "add_slice",
        add_slice_force_serial,
        add_slice_force_parallel,
        n
    );
    let (mul_serial_secs, mul_parallel_secs) = measure_pair!(
        "mul_slice",
        mul_slice_force_serial,
        mul_slice_force_parallel,
        n
    );

    // `exp_slice_force_{serial,parallel}` は単項（`b` を使わない）のため
    // 上記マクロの二項シグネチャに合わせられず個別に計測する。
    let mut out_exp_s = vec![0.0f32; n];
    let exp_serial = run(&config, || {
        exp_slice_force_serial(black_box(&a), black_box(&mut out_exp_s));
    })
    .expect("exp_slice 逐次強制版の計測に失敗");
    black_box(&out_exp_s);

    let mut out_exp_p = vec![0.0f32; n];
    let exp_parallel = run(&config, || {
        exp_slice_force_parallel(black_box(&a), black_box(&mut out_exp_p));
    })
    .expect("exp_slice 並列強制版の計測に失敗");
    black_box(&out_exp_p);

    println!(
        "label={label} n={n} \
         add_serial_ns={as_ns:.1} add_parallel_ns={ap_ns:.1} add_parallel_ratio={ar:.3} \
         mul_serial_ns={ms_ns:.1} mul_parallel_ns={mp_ns:.1} mul_parallel_ratio={mr:.3} \
         exp_serial_ns={es_ns:.1} exp_parallel_ns={ep_ns:.1} exp_parallel_ratio={er:.3}",
        as_ns = add_serial_secs * 1e9,
        ap_ns = add_parallel_secs * 1e9,
        ar = add_parallel_secs / add_serial_secs,
        ms_ns = mul_serial_secs * 1e9,
        mp_ns = mul_parallel_secs * 1e9,
        mr = mul_parallel_secs / mul_serial_secs,
        es_ns = exp_serial.median_secs * 1e9,
        ep_ns = exp_parallel.median_secs * 1e9,
        er = exp_parallel.median_secs / exp_serial.median_secs,
    );
}

/// `PARALLEL_THRESHOLD = 1 << 15 = 32,768` を挟む要素数 2^12〜2^18 で
/// `add`／`mul`／`exp` を計測し、標準出力へ `key=value` 形式で記録する
/// （docs 側〈`docs/perf/cpu-parallel-threshold-sweep.md`〉での突合を
/// 容易にするため）。
///
/// 自動判定版（`PARALLEL_THRESHOLD` で経路が切り替わる）のスイープであり、
/// 要素数増加の影響と並列化オーバーヘッドの影響が混在する。切り分けた
/// 実測は [`elementwise_serial_vs_parallel_sweep`] を参照。
#[test]
#[ignore = "実測ハーネス（--release 推奨）。通常 CI では実行しない"]
fn elementwise_threshold_sweep() {
    for exp2 in 12..=18u32 {
        let n = 1usize << exp2;
        measure_size(&format!("2^{exp2}"), n);
    }
}

/// [`measure_size_serial_vs_parallel`] による同一サイズ直列 vs 並列比較の
/// スイープ（codex-review 指摘・イシュー #1027）。`PARALLEL_THRESHOLD` を
/// 挟む要素数 2^12〜2^18 で、要素数増加の影響を排除した並列化オーバー
/// ヘッドそのものの交差点を実測する。
#[test]
#[ignore = "実測ハーネス（--release 推奨）。通常 CI では実行しない"]
fn elementwise_serial_vs_parallel_sweep() {
    for exp2 in 12..=18u32 {
        let n = 1usize << exp2;
        measure_size_serial_vs_parallel(&format!("2^{exp2}"), n);
    }
}
