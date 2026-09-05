//! VJP 専用 NT/TN strided 入口（イシュー #1215）の補助 A/B 計測。
//!
//! `MetalBackendOps::gemm` に転置 view を渡した場合の before（明示
//! `contiguous()` 経路。本イシュー導入前の挙動と等価——`contiguous()`
//! 済みの入力は NN 判定のため常に従来 `dispatch_auto` 経路を通る）／
//! after（NT/TN strided 入口。本イシューで追加）を、VJP で実際に現れる
//! 形状（`docs/perf/train-step-phase-breakdown.md` の size 64 相当の層
//! 形状、および 1024²・2048² 程度の大形状）について 5 回計測中央値で
//! 比較する。`#[ignore]`（実機計測専用。`.claude/rules/coding-rust.md`
//! 「ベンチは 5 回計測の中央値」）。
//!
//! `backend-cpu::tests::gemm_transposed_perf`（#1213）と同型の構成だが、
//! Metal は NT/TN 経路が `dispatch_auto` とは異なるカーネル（classic
//! strided）を通るため、大形状で性能が後退しうる（`docs/matmul-vjp-
//! zero-copy-decision.md` §4.4「判定ルール」参照）。本ファイルは計測のみ
//! 行い、採否判断は同 doc に記録する。
//!
//! 実行: `cargo test -p fandhe-ai-backend-metal --release --test
//! gemm_transposed_perf -- --ignored --nocapture`

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::BackendOps;
use fandhe_ai_tensor_core::Tensor;
use std::time::Instant;

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// before: `w_t` を毎回明示的に `contiguous()` してから `gemm(g,
/// w_t_owned)` を呼ぶ（NT/TN 導入前の `backend-metal::ops::gemm` 相当。
/// 両オペランドとも NN 判定になるため常に `dispatch_auto` を通る）。
fn run_nt_before(ops: &MetalBackendOps, g: &Tensor<f32>, w_t: &Tensor<f32>) -> f64 {
    let start = Instant::now();
    let w_owned = w_t.contiguous();
    ops.gemm(g, &w_owned).unwrap();
    start.elapsed().as_secs_f64()
}

/// after: `w_t`（転置 view）をそのまま渡す（NT strided 入口。#1215）。
fn run_nt_after(ops: &MetalBackendOps, g: &Tensor<f32>, w_t: &Tensor<f32>) -> f64 {
    let start = Instant::now();
    ops.gemm(g, w_t).unwrap();
    start.elapsed().as_secs_f64()
}

fn bench_nt_shape(m: usize, k: usize, n: usize, runs: usize) -> (f64, f64) {
    let ops = MetalBackendOps::new();
    let g = tensor(random_matrix(0x1000 + m as u64, m * k), &[m, k]);
    let w = tensor(random_matrix(0x2000 + n as u64, n * k), &[n, k]);
    let w_t = w.transpose_2d().unwrap();

    // warmup
    for _ in 0..2 {
        run_nt_before(&ops, &g, &w_t);
        run_nt_after(&ops, &g, &w_t);
    }

    let before: Vec<f64> = (0..runs).map(|_| run_nt_before(&ops, &g, &w_t)).collect();
    let after: Vec<f64> = (0..runs).map(|_| run_nt_after(&ops, &g, &w_t)).collect();
    (median(before), median(after))
}

#[test]
#[ignore = "実機（Apple Silicon）計測専用（#1215 補助 A/B）。cargo test \
            -p fandhe-ai-backend-metal --release --test gemm_transposed_perf \
            -- --ignored --nocapture"]
fn nt_transposed_entry_vs_contiguous_across_shapes() {
    // (m, k, n): VJP で実際に現れる層形状（size 64 相当）・大形状の順。
    let shapes = [
        (64usize, 784usize, 256usize),
        (64, 256, 10),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ];
    for (m, k, n) in shapes {
        let (before, after) = bench_nt_shape(m, k, n, 5);
        println!(
            "NT m={m} k={k} n={n} before_median_s={before:.6} after_median_s={after:.6} speedup={:.3}x",
            before / after
        );
    }
}

/// before: `x_t` を毎回明示的に `contiguous()` してから `gemm(x_t_owned,
/// g)` を呼ぶ（TN 導入前の `backend-metal::ops::gemm` 相当）。
fn run_tn_before(ops: &MetalBackendOps, x_t: &Tensor<f32>, g: &Tensor<f32>) -> f64 {
    let start = Instant::now();
    let x_owned = x_t.contiguous();
    ops.gemm(&x_owned, g).unwrap();
    start.elapsed().as_secs_f64()
}

/// after: `x_t`（転置 view）をそのまま渡す（TN strided 入口。#1215）。
fn run_tn_after(ops: &MetalBackendOps, x_t: &Tensor<f32>, g: &Tensor<f32>) -> f64 {
    let start = Instant::now();
    ops.gemm(x_t, g).unwrap();
    start.elapsed().as_secs_f64()
}

fn bench_tn_shape(m: usize, k: usize, n: usize, runs: usize) -> (f64, f64) {
    let ops = MetalBackendOps::new();
    let x = tensor(random_matrix(0x3000 + m as u64, m * k), &[m, k]);
    let x_t = x.transpose_2d().unwrap();
    let g = tensor(random_matrix(0x4000 + n as u64, m * n), &[m, n]);

    for _ in 0..2 {
        run_tn_before(&ops, &x_t, &g);
        run_tn_after(&ops, &x_t, &g);
    }

    let before: Vec<f64> = (0..runs).map(|_| run_tn_before(&ops, &x_t, &g)).collect();
    let after: Vec<f64> = (0..runs).map(|_| run_tn_after(&ops, &x_t, &g)).collect();
    (median(before), median(after))
}

/// TN（`matmul_vjp`／`Op::LinearResident` の d_weight `Aᵀ @ g` 相当）版。
#[test]
#[ignore = "実機（Apple Silicon）計測専用（#1215 補助 A/B）。cargo test \
            -p fandhe-ai-backend-metal --release --test gemm_transposed_perf \
            -- --ignored --nocapture"]
fn tn_transposed_entry_vs_contiguous_across_shapes() {
    let shapes = [
        (64usize, 784usize, 256usize),
        (64, 256, 10),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ];
    for (m, k, n) in shapes {
        let (before, after) = bench_tn_shape(m, k, n, 5);
        println!(
            "TN m={m} k={k} n={n} before_median_s={before:.6} after_median_s={after:.6} speedup={:.3}x",
            before / after
        );
    }
}
