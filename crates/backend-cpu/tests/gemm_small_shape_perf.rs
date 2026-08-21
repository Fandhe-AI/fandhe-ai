//! 小形状・細長形状（gemv/gevv 相当）における `gemm_blis_parallel`
//! （並列。TASK-1.6f・#184）・`gemm_blis`（直列）・`gemm_naive`（TASK-1.6a
//! 参照実装）の性能比較ハーネス（イシュー #811）。
//!
//! `gemm` crate（OSS 比較対象。`docs/oss-comparison-harness-decision.md`）
//! は (jc,pc) ブロック単位の `total_work` が閾値未満なら 1 スレッド実行へ
//! 落とす（`DEFAULT_THREADING_THRESHOLD = 48*48*256 = 589,824`。
//! ローカル cargo cache の `gemm-common-0.19.0/src/gemm.rs` で実体確認）。
//! 本ハーネスは、自作 BLIS 実装（`crates/backend-cpu/src/gemm_blis/mod.rs`）
//! に同種の閾値直列フォールバックを導入する根拠となる、小形状・細長
//! 形状での rayon 並列化オーバーヘッドを定量化する（`#[ignore]` 通常計測
//! ハーネスの前例は `tests/gemm_blis_perf.rs` を踏襲。`bench_harness::run`
//! は warmup 20・iters 20・中央値/Q1/Q3 記録。TASK-8.1 準拠）。
//!
//! **計測環境の注記（重要）**: 本ハーネスの実行環境はローカル QEMU
//! x86_64 12 コア（AVX2+FMA あり）であり、REQ-8 の正式対象実機（Apple
//! M4 Max。`docs/perf/gemm-optimization-baseline.md` §3）とは異なる。
//! rayon タスク分割・スレッド同期オーバーヘッド（µs オーダーの固定費）
//! の帯域特定はローカル環境でも有意に計測できるが、閾値の最終確定は
//! M4 Max 実機での再スイープを要する（`docs/perf/
//! cpu-gemm-small-shape-serial-fallback.md` の残課題として記録）。
//!
//! 実行例（AVX2+FMA を有効化してビルド）:
//! ```text
//! RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p backend-cpu \
//!     --release -- --ignored gemm_small_shape
//! ```

use backend_cpu::{gemm_blis, gemm_blis_parallel, gemm_naive};
use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

/// 1 形状ぶんの `gemm_naive`・`gemm_blis`（直列）・`gemm_blis_parallel`
/// （並列）を計測し、中央値・比を標準出力へ 1 行で記録する。
///
/// 出力形式は `tests/gemm_blis_perf.rs` の
/// `gemm_blis_baseline_pytorch_square_512_to_4096` に倣い、
/// `key=value` 列挙で機械可読にする（docs 側での突合を容易にするため）。
fn measure_shape(label: &str, m: usize, n: usize, k: usize) {
    let a = random_matrix(5000 + m as u64 * 31 + k as u64, m * k);
    let b = random_matrix(6000 + k as u64 * 31 + n as u64, k * n);

    let config = MeasurementConfig::default(); // warmup 20・iters 20（TASK-8.1 下限）

    let mut c_naive = vec![0.0f32; m * n];
    let naive = run(&config, || {
        c_naive.iter_mut().for_each(|v| *v = 0.0);
        gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();
    })
    .expect("gemm_naive の計測に失敗");

    let mut c_serial = vec![0.0f32; m * n];
    let serial = run(&config, || {
        c_serial.iter_mut().for_each(|v| *v = 0.0);
        gemm_blis(&a, &b, &mut c_serial, m, n, k).unwrap();
    })
    .expect("gemm_blis の計測に失敗");

    let mut c_parallel = vec![0.0f32; m * n];
    let parallel = run(&config, || {
        c_parallel.iter_mut().for_each(|v| *v = 0.0);
        gemm_blis_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap();
    })
    .expect("gemm_blis_parallel の計測に失敗");

    // 3 実装の bit 完全一致契約（REQ-2）を計測に使った具体的な入力で
    // 保険的に確認する（網羅検証は `tests/gemm_blis_parity.rs` の責務）。
    assert_eq!(
        c_naive, c_serial,
        "gemm_naive と gemm_blis が bit 一致しない（{label} m={m} n={n} k={k}）"
    );
    assert_eq!(
        c_naive, c_parallel,
        "gemm_naive と gemm_blis_parallel が bit 一致しない（{label} m={m} n={n} k={k}）"
    );

    let parallel_vs_serial = serial.median_secs / parallel.median_secs;
    let serial_vs_naive = naive.median_secs / serial.median_secs;

    println!(
        "shape={label} m={m} n={n} k={k} \
         naive_median_secs={:.9} \
         serial_median_secs={:.9} (q1={:.9}, q3={:.9}) \
         parallel_median_secs={:.9} (q1={:.9}, q3={:.9}) \
         parallel_vs_serial={parallel_vs_serial:.4}x \
         serial_vs_naive={serial_vs_naive:.4}x",
        naive.median_secs,
        serial.median_secs,
        serial.q1_secs,
        serial.q3_secs,
        parallel.median_secs,
        parallel.q1_secs,
        parallel.q3_secs,
    );
}

/// 小正方形状（閾値直列フォールバックの主対象帯）。
#[test]
#[ignore = "性能計測ハーネス。--release かつ RUSTFLAGS で AVX2+FMA を有効化して個別実行する想定"]
fn gemm_small_shape_perf_square() {
    for size in [16usize, 32, 64, 128, 256] {
        measure_shape("square", size, size, size);
    }
}

/// 細長形状（gemv 相当: n<=1 または m<=1、gevv 相当: k<=2）。
/// §3.2（gemv/gevv 専用経路の採否判断）の実測根拠。
#[test]
#[ignore = "性能計測ハーネス。--release かつ RUSTFLAGS で AVX2+FMA を有効化して個別実行する想定"]
fn gemm_small_shape_perf_elongated() {
    // gemv 相当: m×k @ k×1（列ベクトル）・1×k @ k×n（行ベクトル）
    for n in [512usize, 1024, 2048, 4096] {
        measure_shape("gemv_n1", n, 1, n);
        measure_shape("gemv_m1", 1, n, n);
    }
    // gevv 相当: k が極小（外積に近い形状）
    for size in [256usize, 512, 1024, 2048] {
        measure_shape("gevv_k1", size, size, 1);
        measure_shape("gevv_k2", size, size, 2);
    }
}
