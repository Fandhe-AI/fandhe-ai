//! イシュー #1738: KLDivLoss 損失の融合カーネル（forward 2 段
//! reduction・backward 1 段。`simd_sum` + threadgroup 間結合）の
//! CPU-Metal 数値一致検証（`bce_parity.rs`〈#1737〉と同型構成）。
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
//! cargo test -p fandhe-ai-backend-metal --release --test kl_div_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::{MetalContext, MetalKlDiv};
use fandhe_ai_tensor_core::KlDivTarget;

fn cpu_kl_div_elem_loss(input: f32, target: f32, kind: KlDivTarget) -> f32 {
    match kind {
        KlDivTarget::Probabilities => {
            if target == 0.0 {
                0.0
            } else {
                target * (target.ln() - input)
            }
        }
        _ => target.exp() * (target - input),
    }
}

fn cpu_kl_div_elem_grad_input(_input: f32, target: f32, kind: KlDivTarget) -> f32 {
    match kind {
        KlDivTarget::Probabilities => -target,
        _ => -target.exp(),
    }
}

fn cpu_kl_div_forward_reference(
    input: &[f32],
    target: &[f32],
    kind: KlDivTarget,
    mean: bool,
) -> f32 {
    let numel = input.len();
    if numel == 0 {
        return 0.0;
    }
    let sum: f32 = input
        .iter()
        .zip(target.iter())
        .map(|(&x, &tv)| cpu_kl_div_elem_loss(x, tv, kind))
        .sum();
    if mean { sum / numel as f32 } else { sum }
}

fn cpu_kl_div_backward_reference(
    input: &[f32],
    target: &[f32],
    kind: KlDivTarget,
    scale: f32,
) -> Vec<f32> {
    input
        .iter()
        .zip(target.iter())
        .map(|(&x, &tv)| scale * cpu_kl_div_elem_grad_input(x, tv, kind))
        .collect()
}

fn to_probability(raw: f32) -> f32 {
    raw.clamp(1e-4, 1.0 - 1e-4)
}

fn make_inputs(seed: u64, numel: usize, kind: KlDivTarget) -> (Vec<f32>, Vec<f32>) {
    let raw_input = Xorshift64Star::new(seed).fill_vec(numel);
    let raw_target = Xorshift64Star::new(seed.wrapping_add(1)).fill_vec(numel);
    let input: Vec<f32> = raw_input.iter().map(|&v| -(v * 3.0 + 0.01)).collect();
    let target: Vec<f32> = match kind {
        KlDivTarget::Probabilities => raw_target.iter().map(|&v| to_probability(v)).collect(),
        _ => raw_target.iter().map(|&v| -(v * 3.0 + 0.01)).collect(),
    };
    (input, target)
}

fn assert_kl_div_forward_parity(
    ctx: &MetalContext,
    kl_div: &MetalKlDiv,
    seed: u64,
    numel: usize,
    kind: KlDivTarget,
    mean: bool,
) {
    let (input, target) = make_inputs(seed, numel, kind);
    let factor = if mean {
        if numel == 0 { 1.0 } else { 1.0 / numel as f32 }
    } else {
        1.0
    };

    let gpu_value = kl_div
        .run_kl_div_loss_f32(ctx, &input, &target, kind, factor)
        .expect("MetalKlDiv::run_kl_div_loss_f32 must succeed on Metal-equipped test runner");
    let cpu_value = cpu_kl_div_forward_reference(&input, &target, kind, mean);

    assert_parity(
        &format!("kl_div forward cpu-metal parity numel={numel} kind={kind:?} mean={mean}"),
        &[gpu_value],
        &[cpu_value],
    );
}

fn assert_kl_div_backward_parity(
    ctx: &MetalContext,
    kl_div: &MetalKlDiv,
    seed: u64,
    numel: usize,
    kind: KlDivTarget,
) {
    let (input, target) = make_inputs(seed, numel, kind);
    let scale = 1.7f32;

    let gpu_out = kl_div
        .run_kl_div_backward_f32(ctx, &input, &target, kind, scale)
        .expect("MetalKlDiv::run_kl_div_backward_f32 must succeed on Metal-equipped test runner");
    let cpu_out = cpu_kl_div_backward_reference(&input, &target, kind, scale);

    assert_eq!(gpu_out.len(), cpu_out.len());
    assert_parity(
        &format!("kl_div backward cpu-metal parity numel={numel} kind={kind:?}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`kl_div_num_threadgroups`
/// の分岐点（`KL_DIV_THREADGROUP_WIDTH=256` 単位・
/// `KL_DIV_MAX_THREADGROUPS=1024` 上限跨ぎ）を含む形状を網羅する。
/// `bce_matches_cpu_across_shapes` と同型。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn kl_div_matches_cpu_across_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let kl_div = MetalKlDiv::new(&ctx).expect("kl_div パイプラインの構築に失敗した");

    let mut seed = 6000u64;
    for &numel in &[0usize, 1, 100, 256, 257, 10_000, 300_000] {
        for kind in [KlDivTarget::Probabilities, KlDivTarget::LogProbabilities] {
            for mean in [true, false] {
                seed += 1;
                assert_kl_div_forward_parity(&ctx, &kl_div, seed, numel, kind, mean);
            }
            seed += 1;
            assert_kl_div_backward_parity(&ctx, &kl_div, seed, numel, kind);
        }
    }
}
