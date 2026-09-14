//! イシュー #1737: BCE 損失の融合カーネル（forward 2 段 reduction・
//! backward 1 段。`simd_sum` + threadgroup 間結合）の CPU-Metal 数値
//! 一致検証（`mse_parity.rs` と同型構成）。
//!
//! Metal 実機（Apple Silicon）依存のため `#![cfg(target_os = "macos")]`
//! でファイル全体を macOS 限定にし、各テストに `#[ignore]` を付けて
//! 通常 CI では実行しない。判定式・許容誤差は再定義せず
//! `fandhe_ai_backend_cpu::parity` を唯一の参照とする
//! （`.claude/rules/coding-rust.md`）。
//!
//! 実行コマンド（Mac 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test bce_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::{MetalBce, MetalContext};
use fandhe_ai_tensor_core::BceKind;

/// テスト専用 CPU 参照実装（素朴な逐次実装。`backend-cpu::bce` の融合
/// カーネルとは独立に丸め手順を分離する）。
fn cpu_bce_elem_loss(input: f32, target: f32, kind: BceKind) -> f32 {
    match kind {
        BceKind::Probabilities => {
            let log_p = input.ln().max(-100.0);
            let log_1mp = (1.0 - input).ln().max(-100.0);
            -(target * log_p + (1.0 - target) * log_1mp)
        }
        _ => input.max(0.0) - input * target + (-input.abs()).exp().ln_1p(),
    }
}

fn cpu_bce_elem_grad_input(input: f32, target: f32, kind: BceKind) -> f32 {
    match kind {
        BceKind::Probabilities => {
            let denom = (input * (1.0 - input)).max(1e-12);
            (input - target) / denom
        }
        _ => {
            let sigmoid = if input >= 0.0 {
                1.0 / (1.0 + (-input).exp())
            } else {
                let e = input.exp();
                e / (1.0 + e)
            };
            sigmoid - target
        }
    }
}

fn cpu_bce_forward_reference(input: &[f32], target: &[f32], kind: BceKind, mean: bool) -> f32 {
    let numel = input.len();
    if numel == 0 {
        return 0.0;
    }
    let sum: f32 = input
        .iter()
        .zip(target.iter())
        .map(|(&p, &y)| cpu_bce_elem_loss(p, y, kind))
        .sum();
    if mean { sum / numel as f32 } else { sum }
}

fn cpu_bce_backward_reference(
    input: &[f32],
    target: &[f32],
    kind: BceKind,
    scale: f32,
) -> Vec<f32> {
    input
        .iter()
        .zip(target.iter())
        .map(|(&p, &y)| scale * cpu_bce_elem_grad_input(p, y, kind))
        .collect()
}

fn to_probability(raw: f32) -> f32 {
    raw.clamp(1e-4, 1.0 - 1e-4)
}

fn make_inputs(seed: u64, numel: usize, kind: BceKind) -> (Vec<f32>, Vec<f32>) {
    let raw_input = Xorshift64Star::new(seed).fill_vec(numel);
    let raw_target = Xorshift64Star::new(seed.wrapping_add(1)).fill_vec(numel);
    match kind {
        BceKind::Probabilities => (
            raw_input.iter().map(|&v| to_probability(v)).collect(),
            raw_target
                .iter()
                .map(|&v| if v < 0.5 { 0.0 } else { 1.0 })
                .collect(),
        ),
        _ => (
            raw_input.iter().map(|&v| v * 8.0 - 4.0).collect(),
            raw_target
                .iter()
                .map(|&v| if v < 0.5 { 0.0 } else { 1.0 })
                .collect(),
        ),
    }
}

fn assert_bce_forward_parity(
    ctx: &MetalContext,
    bce: &MetalBce,
    seed: u64,
    numel: usize,
    kind: BceKind,
    mean: bool,
) {
    let (input, target) = make_inputs(seed, numel, kind);
    let factor = if mean {
        if numel == 0 { 1.0 } else { 1.0 / numel as f32 }
    } else {
        1.0
    };

    let gpu_value = bce
        .run_bce_loss_f32(ctx, &input, &target, kind, factor)
        .expect("MetalBce::run_bce_loss_f32 must succeed on Metal-equipped test runner");
    let cpu_value = cpu_bce_forward_reference(&input, &target, kind, mean);

    assert_parity(
        &format!("bce forward cpu-metal parity numel={numel} kind={kind:?} mean={mean}"),
        &[gpu_value],
        &[cpu_value],
    );
}

fn assert_bce_backward_parity(
    ctx: &MetalContext,
    bce: &MetalBce,
    seed: u64,
    numel: usize,
    kind: BceKind,
) {
    let (input, target) = make_inputs(seed, numel, kind);
    let scale = 1.7f32;

    let gpu_out = bce
        .run_bce_backward_f32(ctx, &input, &target, kind, scale)
        .expect("MetalBce::run_bce_backward_f32 must succeed on Metal-equipped test runner");
    let cpu_out = cpu_bce_backward_reference(&input, &target, kind, scale);

    assert_eq!(gpu_out.len(), cpu_out.len());
    assert_parity(
        &format!("bce backward cpu-metal parity numel={numel} kind={kind:?}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`bce_num_threadgroups` の
/// 分岐点（`BCE_THREADGROUP_WIDTH=256` 単位・`BCE_MAX_THREADGROUPS=1024`
/// 上限跨ぎ）を含む形状を網羅する。`mse_matches_cpu_across_shapes` と
/// 同型。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn bce_matches_cpu_across_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let bce = MetalBce::new(&ctx).expect("bce パイプラインの構築に失敗した");

    let mut seed = 4000u64;
    for &numel in &[0usize, 1, 100, 256, 257, 10_000, 300_000] {
        for kind in [BceKind::Probabilities, BceKind::Logits] {
            for mean in [true, false] {
                seed += 1;
                assert_bce_forward_parity(&ctx, &bce, seed, numel, kind, mean);
            }
            seed += 1;
            assert_bce_backward_parity(&ctx, &bce, seed, numel, kind);
        }
    }
}
