//! イシュー #1045: MSE 損失の融合カーネル（forward 2 段 reduction・
//! backward 1 段。`simd_sum` + threadgroup 間結合）の CPU-Metal 数値一致
//! 検証。
//!
//! `softmax_parity.rs`（#604）と同じ構成方針: Metal 実機（Apple Silicon）
//! 依存のため `#![cfg(target_os = "macos")]` でファイル全体を macOS 限定に
//! し、各テストに `#[ignore]` を付けて通常 CI では実行しない。判定式・
//! 許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の参照と
//! する（`.claude/rules/coding-rust.md`）。
//!
//! 実行コマンド（Mac 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test mse_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::{MetalContext, MetalMse};

/// テスト専用 CPU 参照実装（素朴な逐次実装。`backend-cpu::mse` の融合
/// カーネルとは独立に丸め手順を分離する）。
fn cpu_mse_forward_reference(pred: &[f32], target: &[f32], mean: bool) -> f32 {
    let numel = pred.len();
    if numel == 0 {
        return 0.0;
    }
    let sum_sq: f32 = pred
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| {
            let d = p - t;
            d * d
        })
        .sum();
    if mean { sum_sq / numel as f32 } else { sum_sq }
}

fn cpu_mse_backward_reference(pred: &[f32], target: &[f32], scale: f32) -> Vec<f32> {
    pred.iter()
        .zip(target.iter())
        .map(|(&p, &t)| scale * (p - t))
        .collect()
}

fn assert_mse_forward_parity(
    ctx: &MetalContext,
    mse: &MetalMse,
    seed: u64,
    numel: usize,
    mean: bool,
) {
    let pred = Xorshift64Star::new(seed).fill_vec(numel);
    let target = Xorshift64Star::new(seed.wrapping_add(1)).fill_vec(numel);
    let factor = if mean {
        if numel == 0 { 1.0 } else { 1.0 / numel as f32 }
    } else {
        1.0
    };

    let gpu_value = mse
        .run_mse_loss_f32(ctx, &pred, &target, factor)
        .expect("MetalMse::run_mse_loss_f32 must succeed on Metal-equipped test runner");
    let cpu_value = cpu_mse_forward_reference(&pred, &target, mean);

    assert_parity(
        &format!("mse forward cpu-metal parity numel={numel} mean={mean}"),
        &[gpu_value],
        &[cpu_value],
    );
}

fn assert_mse_backward_parity(ctx: &MetalContext, mse: &MetalMse, seed: u64, numel: usize) {
    let pred = Xorshift64Star::new(seed).fill_vec(numel);
    let target = Xorshift64Star::new(seed.wrapping_add(1)).fill_vec(numel);
    let scale = 1.7f32;

    let gpu_out = mse
        .run_mse_backward_f32(ctx, &pred, &target, scale)
        .expect("MetalMse::run_mse_backward_f32 must succeed on Metal-equipped test runner");
    let cpu_out = cpu_mse_backward_reference(&pred, &target, scale);

    assert_eq!(gpu_out.len(), cpu_out.len());
    assert_parity(
        &format!("mse backward cpu-metal parity numel={numel}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`mse_num_threadgroups` の
/// 分岐点（`MSE_THREADGROUP_WIDTH=256` 単位・`MSE_MAX_THREADGROUPS=1024`
/// 上限跨ぎ）を含む形状を網羅する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn mse_matches_cpu_across_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let mse = MetalMse::new(&ctx).expect("mse パイプラインの構築に失敗した");

    let mut seed = 3000u64;
    for &numel in &[0usize, 1, 100, 256, 257, 10_000, 300_000] {
        for mean in [true, false] {
            seed += 1;
            assert_mse_forward_parity(&ctx, &mse, seed, numel, mean);
        }
        seed += 1;
        assert_mse_backward_parity(&ctx, &mse, seed, numel);
    }
}
