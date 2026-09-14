//! イシュー #1739: Huber／SmoothL1 損失の融合カーネル（forward 2 段
//! reduction・backward 1 段。`simd_sum` + threadgroup 間結合）の
//! CPU-Metal 数値一致検証（`mse_parity.rs` と同型の構成）。
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
//! cargo test -p fandhe-ai-backend-metal --release --test huber_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::{MetalContext, MetalHuber};
use fandhe_ai_tensor_core::HuberKind;

/// テスト専用 CPU 参照実装（素朴な逐次実装。`backend-cpu::huber` の
/// 融合カーネルとは独立に丸め手順を分離する）。
fn cpu_elem_loss(d: f32, kind: HuberKind, delta: f32) -> f32 {
    let abs_d = d.abs();
    match kind {
        HuberKind::Huber => {
            if abs_d < delta {
                0.5 * d * d
            } else {
                delta * (abs_d - 0.5 * delta)
            }
        }
        HuberKind::SmoothL1 => {
            if abs_d < delta {
                0.5 * d * d / delta
            } else {
                abs_d - 0.5 * delta
            }
        }
        _ => 0.0,
    }
}

fn cpu_elem_grad(d: f32, kind: HuberKind, delta: f32) -> f32 {
    let abs_d = d.abs();
    match kind {
        HuberKind::Huber => {
            if abs_d < delta {
                d
            } else {
                delta.copysign(d)
            }
        }
        HuberKind::SmoothL1 => {
            if abs_d < delta {
                d / delta
            } else {
                1.0f32.copysign(d)
            }
        }
        _ => 0.0,
    }
}

fn cpu_huber_forward_reference(
    pred: &[f32],
    target: &[f32],
    kind: HuberKind,
    delta: f32,
    mean: bool,
) -> f32 {
    let numel = pred.len();
    if numel == 0 {
        return 0.0;
    }
    let sum: f32 = pred
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| cpu_elem_loss(p - t, kind, delta))
        .sum();
    if mean { sum / numel as f32 } else { sum }
}

fn cpu_huber_backward_reference(
    pred: &[f32],
    target: &[f32],
    kind: HuberKind,
    delta: f32,
    scale: f32,
) -> Vec<f32> {
    pred.iter()
        .zip(target.iter())
        .map(|(&p, &t)| scale * cpu_elem_grad(p - t, kind, delta))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn assert_huber_forward_parity(
    ctx: &MetalContext,
    huber: &MetalHuber,
    seed: u64,
    numel: usize,
    kind: HuberKind,
    delta: f32,
    mean: bool,
) {
    let pred = Xorshift64Star::new(seed).fill_vec(numel);
    let target = Xorshift64Star::new(seed.wrapping_add(1)).fill_vec(numel);
    let factor = if mean {
        if numel == 0 { 1.0 } else { 1.0 / numel as f32 }
    } else {
        1.0
    };

    let gpu_value = huber
        .run_huber_loss_f32(ctx, &pred, &target, kind, delta, factor)
        .expect("MetalHuber::run_huber_loss_f32 must succeed on Metal-equipped test runner");
    let cpu_value = cpu_huber_forward_reference(&pred, &target, kind, delta, mean);

    assert_parity(
        &format!(
            "huber forward cpu-metal parity numel={numel} kind={kind:?} delta={delta} mean={mean}"
        ),
        &[gpu_value],
        &[cpu_value],
    );
}

fn assert_huber_backward_parity(
    ctx: &MetalContext,
    huber: &MetalHuber,
    seed: u64,
    numel: usize,
    kind: HuberKind,
    delta: f32,
) {
    let pred = Xorshift64Star::new(seed).fill_vec(numel);
    let target = Xorshift64Star::new(seed.wrapping_add(1)).fill_vec(numel);
    let scale = 1.7f32;

    let gpu_out = huber
        .run_huber_backward_f32(ctx, &pred, &target, kind, delta, scale)
        .expect("MetalHuber::run_huber_backward_f32 must succeed on Metal-equipped test runner");
    let cpu_out = cpu_huber_backward_reference(&pred, &target, kind, delta, scale);

    assert_eq!(gpu_out.len(), cpu_out.len());
    assert_parity(
        &format!("huber backward cpu-metal parity numel={numel} kind={kind:?} delta={delta}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`huber_num_threadgroups`
/// の分岐点（`HUBER_THREADGROUP_WIDTH=256` 単位・
/// `HUBER_MAX_THREADGROUPS=1024` 上限跨ぎ）に加え、kind×delta の組合せ
/// を網羅する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn huber_matches_cpu_across_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let huber = MetalHuber::new(&ctx).expect("huber パイプラインの構築に失敗した");

    let mut seed = 17390u64;
    for &numel in &[0usize, 1, 100, 256, 257, 10_000, 300_000] {
        for (kind, delta) in [
            (HuberKind::Huber, 0.5f32),
            (HuberKind::Huber, 1.0),
            (HuberKind::Huber, 2.0),
            (HuberKind::SmoothL1, 0.5),
            (HuberKind::SmoothL1, 1.0),
            (HuberKind::SmoothL1, 2.0),
        ] {
            for mean in [true, false] {
                seed += 1;
                assert_huber_forward_parity(&ctx, &huber, seed, numel, kind, delta, mean);
            }
            seed += 1;
            assert_huber_backward_parity(&ctx, &huber, seed, numel, kind, delta);
        }
    }
}
