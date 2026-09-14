//! イシュー #1738: NLLLoss 損失の融合カーネル（forward 2 段
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
//! cargo test -p fandhe-ai-backend-metal --release --test nll_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::{MetalContext, MetalNll, NllLayout};

fn cpu_nll_forward_reference(input: &[f32], targets: &[i32], layout: NllLayout, mean: bool) -> f32 {
    let n = layout.outer * layout.inner;
    let mut total = 0f32;
    for o in 0..layout.outer {
        for i in 0..layout.inner {
            let t = targets[o * layout.inner + i] as usize;
            total -= input[(o * layout.num_classes + t) * layout.inner + i];
        }
    }
    if mean {
        if n == 0 { 0.0 } else { total / n as f32 }
    } else {
        total
    }
}

fn cpu_nll_backward_reference(targets: &[i32], layout: NllLayout, scale: f32) -> Vec<f32> {
    let mut grad = vec![0f32; layout.outer * layout.num_classes * layout.inner];
    for o in 0..layout.outer {
        for i in 0..layout.inner {
            let t = targets[o * layout.inner + i] as usize;
            grad[(o * layout.num_classes + t) * layout.inner + i] = -scale;
        }
    }
    grad
}

fn make_targets(seed: u64, outer: usize, num_classes: usize, inner: usize) -> Vec<i32> {
    let n = outer * inner;
    Xorshift64Star::new(seed)
        .fill_vec(n)
        .into_iter()
        .map(|v| ((v * num_classes as f32) as usize).min(num_classes.saturating_sub(1)) as i32)
        .collect()
}

fn assert_nll_forward_parity(
    ctx: &MetalContext,
    nll: &MetalNll,
    seed: u64,
    layout: NllLayout,
    mean: bool,
) {
    let input =
        Xorshift64Star::new(seed).fill_vec(layout.outer * layout.num_classes * layout.inner);
    let targets = make_targets(
        seed.wrapping_add(1),
        layout.outer,
        layout.num_classes,
        layout.inner,
    );
    let n = layout.outer * layout.inner;
    let factor = if mean {
        if n == 0 { 1.0 } else { 1.0 / n as f32 }
    } else {
        1.0
    };

    let gpu_value = nll
        .run_nll_loss_f32(ctx, &input, &targets, layout, factor)
        .expect("MetalNll::run_nll_loss_f32 must succeed on Metal-equipped test runner");
    let cpu_value = cpu_nll_forward_reference(&input, &targets, layout, mean);

    assert_parity(
        &format!("nll forward cpu-metal parity layout={layout:?} mean={mean}"),
        &[gpu_value],
        &[cpu_value],
    );
}

fn assert_nll_backward_parity(ctx: &MetalContext, nll: &MetalNll, seed: u64, layout: NllLayout) {
    let targets = make_targets(seed, layout.outer, layout.num_classes, layout.inner);
    let scale = 1.7f32;

    let gpu_out = nll
        .run_nll_backward_f32(ctx, &targets, layout, scale)
        .expect("MetalNll::run_nll_backward_f32 must succeed on Metal-equipped test runner");
    let cpu_out = cpu_nll_backward_reference(&targets, layout, scale);

    assert_eq!(gpu_out.len(), cpu_out.len());
    assert_parity(
        &format!("nll backward cpu-metal parity layout={layout:?}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`nll_num_threadgroups` の
/// 分岐点（`NLL_THREADGROUP_WIDTH=256` 単位・`NLL_MAX_THREADGROUPS=1024`
/// 上限跨ぎ）・`class_dim` の先頭・末尾・中間軸を含む形状を網羅する。
/// `bce_matches_cpu_across_shapes` と同型。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn nll_matches_cpu_across_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let nll = MetalNll::new(&ctx).expect("nll パイプラインの構築に失敗した");

    let layouts = [
        NllLayout {
            outer: 0,
            num_classes: 3,
            inner: 1,
        },
        NllLayout {
            outer: 1,
            num_classes: 3,
            inner: 1,
        },
        NllLayout {
            outer: 100,
            num_classes: 5,
            inner: 1,
        },
        NllLayout {
            outer: 1,
            num_classes: 300,
            inner: 1,
        },
        NllLayout {
            outer: 1,
            num_classes: 3,
            inner: 100,
        },
        NllLayout {
            outer: 20,
            num_classes: 10,
            inner: 30,
        },
        NllLayout {
            outer: 300,
            num_classes: 1000,
            inner: 1,
        },
        NllLayout {
            outer: 300_000,
            num_classes: 4,
            inner: 1,
        }, // n_samples=300_000 > NLL_THREADGROUP_WIDTH*NLL_MAX_THREADGROUPS
           // =262_144（grid-stride ループが `n_samples` 全域を確実に走査
           // することの実機確認。PR #1850 codex-review P2 是正）
    ];

    let mut seed = 5000u64;
    for &layout in &layouts {
        for mean in [true, false] {
            seed += 1;
            assert_nll_forward_parity(&ctx, &nll, seed, layout, mean);
        }
        seed += 1;
        assert_nll_backward_parity(&ctx, &nll, seed, layout);
    }
}
