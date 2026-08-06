//! `gemm_blis_parallel`（TASK-1.6f・#184）と `gemm_parallel`（TASK-1.6a）の
//! 性能比較ハーネス。`bench_harness::protocol::run`（warmup 20 回以上・
//! 計測 20 回以上・中央値／Q1/Q3 記録。TASK-8.1 準拠）を用いる。
//!
//! `#[ignore]` として通常 CI から除外する（`.claude/rules/coding-rust.md`
//! 「ベンチは 5 回計測の中央値を採用し」の趣旨と、TASK-8.1 の 20/20
//! プロトコルとの整合は `bench-harness` クレートドキュメント参照。本
//! ハーネスは受け入れ条件「TASK-1.6a 比の相対改善を実測・記録する」を
//! 満たすための性能計測であり、pass/fail の自動判定は行わない
//! （性能下限〈REQ-8 の 20%〉達成の可否判定自体は #24 のスコープ）。
//!
//! 実行例（AVX2+FMA を有効化してビルド）:
//! ```text
//! RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p backend-cpu \
//!     --release -- --ignored gemm_blis_perf
//! ```

use backend_cpu::{gemm_blis_parallel, gemm_parallel};
use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

/// M=N=K の正方形状で `gemm_parallel`（TASK-1.6a）・`gemm_blis_parallel`
/// （TASK-1.6f）を計測し、中央値・改善比を標準出力へ記録する。
fn measure_square(size: usize) {
    let m = size;
    let n = size;
    let k = size;
    let a = random_matrix(1000 + size as u64, m * k);
    let b = random_matrix(2000 + size as u64, k * n);

    let config = MeasurementConfig::default(); // warmup 20・iters 20（TASK-8.1 下限）

    let mut c_blocked = vec![0.0f32; m * n];
    let blocked = run(&config, || {
        c_blocked.iter_mut().for_each(|v| *v = 0.0);
        gemm_parallel(&a, &b, &mut c_blocked, m, n, k).unwrap();
    })
    .expect("gemm_parallel の計測に失敗");

    let mut c_blis = vec![0.0f32; m * n];
    let blis = run(&config, || {
        c_blis.iter_mut().for_each(|v| *v = 0.0);
        gemm_blis_parallel(&a, &b, &mut c_blis, m, n, k).unwrap();
    })
    .expect("gemm_blis_parallel の計測に失敗");

    let speedup = blocked.median_secs / blis.median_secs;

    println!(
        "M=N=K={size}: gemm_parallel median={:.6}s (q1={:.6}, q3={:.6}) / \
         gemm_blis_parallel median={:.6}s (q1={:.6}, q3={:.6}) / speedup={speedup:.3}x",
        blocked.median_secs,
        blocked.q1_secs,
        blocked.q3_secs,
        blis.median_secs,
        blis.q1_secs,
        blis.q3_secs,
    );

    // 数値も一致することを併せて確認する（性能計測が誤った実装を比較して
    // いないことの保険。bit 完全一致契約は `tests/gemm_blis_parity.rs` で
    // 別途網羅的に検証済みのため、ここでは計測に使った具体的な入力に
    // 限定した簡易チェックに留める）。
    assert_eq!(
        c_blocked, c_blis,
        "計測対象の 2 実装が bit 一致しない（M=N=K={size}）"
    );
}

#[test]
#[ignore = "性能計測ハーネス。--release かつ RUSTFLAGS で AVX2+FMA を有効化して個別実行する想定"]
fn gemm_blis_perf_square_512_1024_2048() {
    for size in [512usize, 1024, 2048] {
        measure_square(size);
    }
}
