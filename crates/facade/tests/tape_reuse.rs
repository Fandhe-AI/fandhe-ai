//! `facade::Tape::reset`/`leaf_count`/`leaf`（イシュー #1048）の
//! newtype 経由の結線検証。
//!
//! 内部の受け入れ条件（葉プレフィックス保持・世代不一致の fail-closed
//! 拒否・bit-exact 一致）は `crates/autodiff/tests/tape_reset.rs` が
//! 直接検証済みのため、本ファイルは facade の薄い委譲経路
//! （`fandhe_ai::Tape::reset`/`leaf_count`/`leaf`）が正しく機能すること
//! と、reuse（`reset` で同一 `Tape` を再利用）が fresh（毎回新規
//! `tape()`）と CPU 上で bit-exact に一致すること、参考として reuse が
//! fresh に対し極端な性能劣化（アロケーション蓄積の再発）を起こして
//! いないことを確認する。

use fandhe_ai::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("shape 一致")
}

/// `facade::Tape` の newtype 経由で `reset`/`leaf_count`/`leaf` が呼べ、
/// reset 後も葉プレフィックスが保持されることを検証する。
#[test]
fn facade_tape_reset_keeps_leaf_prefix() {
    let mut tape = fandhe_ai::tape();
    let a = tape.var(&t(vec![1.0, 2.0], &[2]));
    let b = tape.var(&t(vec![3.0, 4.0], &[2]));
    assert_eq!(tape.leaf_count(), 2);

    let _sum = a.add(&b).expect("add は shape 一致で成功する");

    tape.reset();
    assert_eq!(tape.leaf_count(), 2);
    let a2 = tape.leaf(0).expect("葉 0 は reset 後も保持される");
    let b2 = tape.leaf(1).expect("葉 1 は reset 後も保持される");
    assert_eq!(a2.to_tensor().contiguous().as_slice().unwrap(), &[1.0, 2.0]);
    assert_eq!(b2.to_tensor().contiguous().as_slice().unwrap(), &[3.0, 4.0]);
    assert!(tape.leaf(2).is_none());
}

/// CPU 上で N=256 の GEMM を reuse（`reset` で同一 `Tape` を使い回す）
/// した結果が、fresh（毎回 `fandhe_ai::tape()` を新規生成する従来運用）
/// と bit-exact に一致することを検証する（#1048 受け入れ基準の中核:
/// tape 再利用が数値結果を変えないこと）。
#[test]
fn cpu_gemm_reuse_matches_fresh_bit_exact_n256() {
    const N: usize = 256;
    let a_data: Vec<f32> = (0..N * N).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..N * N).map(|i| (i % 5) as f32 * 0.2 - 0.3).collect();

    // fresh: 毎回新規 tape。
    let fresh_tape = fandhe_ai::tape();
    let fa = fresh_tape.var(&t(a_data.clone(), &[N, N]));
    let fb = fresh_tape.var(&t(b_data.clone(), &[N, N]));
    let fresh_result = fa
        .matmul(&fb)
        .expect("matmul は shape 一致で成功する")
        .to_tensor();

    // reuse: 1 回目は fresh と同じ演算を行い、reset して同じ葉で
    // もう一度 matmul する（2 回目の結果を fresh と突き合わせる）。
    let mut reuse_tape = fandhe_ai::tape();
    let ra = reuse_tape.var(&t(a_data.clone(), &[N, N]));
    let rb = reuse_tape.var(&t(b_data.clone(), &[N, N]));
    let _first = ra.matmul(&rb).expect("matmul は shape 一致で成功する");

    reuse_tape.reset();
    let ra2 = reuse_tape.leaf(0).expect("葉 0 は reset 後も保持される");
    let rb2 = reuse_tape.leaf(1).expect("葉 1 は reset 後も保持される");
    let reuse_result = ra2
        .matmul(&rb2)
        .expect("matmul は shape 一致で成功する")
        .to_tensor();

    assert_eq!(
        fresh_result.contiguous().as_slice(),
        reuse_result.contiguous().as_slice(),
        "reuse（reset 経由の tape 再利用）は fresh（毎回新規 tape）と \
         bit-exact に一致するはず（同じ BackendOps・同じ演算順のため）"
    );
}

/// **参考計測（非 ignore・小サイズ・緩い閾値）**: reuse が fresh に
/// 対して極端な性能劣化を起こしていない（#1048 発端のバッファ蓄積
/// 問題が再発していない）ことを大まかに確認する。CI 実行環境のノイズ
/// 耐性のため閾値は意図的に緩く取り（`reuse 中央値 <= fresh 中央値 ×
/// 4.0`）、失敗時は実測値をログ出力する。厳密な性能判定・N=512〜2048
/// でのフル計測は `#[ignore]` の実機ベンチ（CUDA/Metal）側で行う
/// （framework-compare 側の reuse 再計測は crates.io 次回公開後に
/// 別途行う。PR 本文の残課題参照）。
#[test]
fn cpu_gemm_reuse_is_not_drastically_slower_than_fresh_n256() {
    use bench_harness::median_q1_q3;
    use std::time::Instant;

    const N: usize = 256;
    const ITERS: usize = 5;
    let a_data: Vec<f32> = (0..N * N).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..N * N).map(|i| (i % 5) as f32 * 0.2 - 0.3).collect();

    let mut fresh_samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let tape = fandhe_ai::tape();
        let a = tape.var(&t(a_data.clone(), &[N, N]));
        let b = tape.var(&t(b_data.clone(), &[N, N]));
        let _ = a.matmul(&b).unwrap().to_tensor();
        fresh_samples.push(start.elapsed().as_secs_f64());
    }

    let mut reuse_tape = fandhe_ai::tape();
    let _seed_a = reuse_tape.var(&t(a_data.clone(), &[N, N]));
    let _seed_b = reuse_tape.var(&t(b_data.clone(), &[N, N]));
    reuse_tape.reset(); // ウォームアップ演算を切り詰めてから計測開始。

    let mut reuse_samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let a = reuse_tape.leaf(0).unwrap();
        let b = reuse_tape.leaf(1).unwrap();
        let _ = a.matmul(&b).unwrap().to_tensor();
        reuse_tape.reset();
        reuse_samples.push(start.elapsed().as_secs_f64());
    }

    let fresh_q = median_q1_q3(&fresh_samples).expect("5 サンプルは非空");
    let reuse_q = median_q1_q3(&reuse_samples).expect("5 サンプルは非空");

    eprintln!(
        "cpu_gemm_reuse_is_not_drastically_slower_than_fresh_n256: \
         fresh median={:.6}s reuse median={:.6}s",
        fresh_q.median, reuse_q.median
    );
    assert!(
        reuse_q.median <= fresh_q.median * 4.0,
        "reuse（{:.6}s）が fresh（{:.6}s）の 4 倍を超えて遅い \
         （バッファ蓄積の再発を疑う）",
        reuse_q.median,
        fresh_q.median
    );
}

/// **実機ベンチ（`#[ignore]`。CUDA/Metal）**: N=512/1024/2048 で
/// fresh／reuse-with-reset の中央値と `memory_pool_stats`（デバイス側
/// プール統計）を出力する（#1048 の受け入れ条件 R2「reuse GEMM で
/// N=512〜1024 が fresh 以上の性能になること」の実機側確認用）。
///
/// `tape_cuda_cache_bench.rs` と同じ「record only（hard assert しない）」
/// 方針を踏襲する（GPU クロック挙動・他プロセス競合の環境揺らぎを
/// hard assert に持ち込むと flaky 化するため）。本ランでは実機
/// （DGX Spark GB10・Metal 実機）へのアクセスがないため値は未計測の
/// まま構造のみ用意し、実機セッションで `cargo test -p fandhe-ai --test \
/// tape_reuse -- --ignored --nocapture` を実行して記録することを
/// 前提とする（`docs/perf/` への記入は別途対応）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機必須。イシュー #1048 受け入れ条件 R2 の実機側計測"]
fn metal_gemm_reuse_vs_fresh_n512_1024_2048() {
    run_device_reuse_vs_fresh_bench(fandhe_ai::Device::Metal);
}

#[test]
#[ignore = "CUDA 実機必須。イシュー #1048 受け入れ条件 R2 の実機側計測"]
fn cuda_gemm_reuse_vs_fresh_n512_1024_2048() {
    run_device_reuse_vs_fresh_bench(fandhe_ai::Device::Cuda(0));
}

/// `cuda_gemm_reuse_vs_fresh_n512_1024_2048`／
/// `metal_gemm_reuse_vs_fresh_n512_1024_2048` の共通本体。`device` の
/// 実機が利用できない場合は `tape_for` が `Err` を返すため、ガード
/// なしで `expect` する（`#[ignore]` により通常 CI では実行されず、
/// 実機セッションでのみ意図的に実行するため。`tape_cuda_cache_bench.rs`
/// の `measure_tape_for_cuda_matmul` と異なりデバイスプローブを別途
/// 挟まない——後続の `tape_for` 自体が実機なしを検出して `Err` を返す）。
fn run_device_reuse_vs_fresh_bench(device: fandhe_ai::Device) {
    use bench_harness::median_q1_q3;
    use std::time::Instant;

    const ITERS: usize = 5;
    for &n in &[512usize, 1024, 2048] {
        let a_data: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..n * n).map(|i| (i % 5) as f32 * 0.2 - 0.3).collect();

        let mut fresh_samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let start = Instant::now();
            let tape = fandhe_ai::tape_for(device).expect("実機セッションでのみ実行する");
            let a = tape.var(&t(a_data.clone(), &[n, n]));
            let b = tape.var(&t(b_data.clone(), &[n, n]));
            let _ = a.matmul(&b).unwrap().to_tensor();
            fresh_samples.push(start.elapsed().as_secs_f64());
        }

        let mut reuse_tape = fandhe_ai::tape_for(device).expect("実機セッションでのみ実行する");
        let _seed_a = reuse_tape.var(&t(a_data.clone(), &[n, n]));
        let _seed_b = reuse_tape.var(&t(b_data.clone(), &[n, n]));
        reuse_tape.reset();

        let mut reuse_samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let start = Instant::now();
            let a = reuse_tape.leaf(0).unwrap();
            let b = reuse_tape.leaf(1).unwrap();
            let _ = a.matmul(&b).unwrap().to_tensor();
            reuse_tape.reset();
            reuse_samples.push(start.elapsed().as_secs_f64());
        }

        let fresh_q = median_q1_q3(&fresh_samples).expect("5 サンプルは非空");
        let reuse_q = median_q1_q3(&reuse_samples).expect("5 サンプルは非空");
        let pool_stats = fandhe_ai::memory_pool_stats(device).ok().flatten();

        eprintln!(
            "run_device_reuse_vs_fresh_bench: device={device:?} N={n} \
             fresh median={:.6}s reuse median={:.6}s pool_stats={pool_stats:?}",
            fresh_q.median, reuse_q.median
        );
    }
}
