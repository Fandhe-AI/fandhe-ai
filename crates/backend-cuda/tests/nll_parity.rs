//! イシュー #1738: NLLLoss 損失の融合カーネル（forward 2 段 reduction・
//! backward 1 段）の CPU-CUDA 数値一致検証（`mse_parity.rs`・
//! `bce_parity.rs` と同型構成）。
//!
//! `softmax_parity.rs`（#594）・`mse_parity.rs`（#1045）と同じ構成方針を
//! 踏襲する: 環境適応スモーク（属性なし。通常 CI で実行し、CUDA 非搭載
//! 環境では `fandhe_ai_backend_cuda::CudaError::DriverUnavailable`／
//! `NvrtcUnavailable` を確認して panic しないことのみ検証）と、実機必須の
//! 形状網羅（`#[ignore]`。DGX Spark GB10 等）を分離する。判定式・許容
//! 誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の参照とする
//! （`.claude/rules/coding-rust.md`）。
//!
//! CPU 参照実装は本ファイル内の素朴な逐次実装（`fandhe_ai_autodiff::
//! eval::nll_loss` と数式的に同一。`mse_parity.rs::
//! cpu_mse_forward_reference` と同型）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test nll_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaNll, NllLayout};

mod common;

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

fn assert_nll_forward_parity(nll: &CudaNll, seed: u64, layout: NllLayout, mean: bool) {
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
        .run_nll_loss_f32(&input, &targets, layout, factor)
        .expect("CudaNll::run_nll_loss_f32 must succeed on CUDA-equipped test runner");
    let cpu_value = cpu_nll_forward_reference(&input, &targets, layout, mean);

    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("nll forward cpu-cuda parity layout={layout:?} mean={mean}"),
        &[gpu_value],
        &[cpu_value],
    );
}

fn assert_nll_backward_parity(nll: &CudaNll, seed: u64, layout: NllLayout) {
    let targets = make_targets(seed, layout.outer, layout.num_classes, layout.inner);
    let scale = 1.7f32;

    let gpu_out = nll
        .run_nll_backward_f32(&targets, layout, scale)
        .expect("CudaNll::run_nll_backward_f32 must succeed on CUDA-equipped test runner");
    let cpu_out = cpu_nll_backward_reference(&targets, layout, scale);

    assert_eq!(gpu_out.len(), cpu_out.len());
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("nll backward cpu-cuda parity layout={layout:?}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。`mse_parity.rs::
/// mse_parity_smoke_env_adaptive` と同じ分岐パターン。
#[test]
fn nll_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaNll::new(&device) {
        Ok(nll) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
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
                    outer: 50,
                    num_classes: 10,
                    inner: 1,
                },
                NllLayout {
                    outer: 5,
                    num_classes: 4,
                    inner: 6,
                },
            ];
            for (idx, &layout) in layouts.iter().enumerate() {
                let seed = 2001 + idx as u64 * 10;
                assert_nll_forward_parity(&nll, seed, layout, true);
                assert_nll_forward_parity(&nll, seed + 1, layout, false);
                assert_nll_backward_parity(&nll, seed + 2, layout);
            }
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {
            // NVRTC 非搭載環境。panic しないことのみ確認する
            // （`mse_parity.rs` と同じ理由）。
        }
        Err(other) => panic!("unexpected error variant for CudaNll::new: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`nll_num_blocks` の分岐点
/// （`NLL_BLOCK_DIM=256` 単位・`NLL_MAX_BLOCKS=1024` 上限跨ぎ）・
/// `class_dim` の先頭・末尾・中間軸を含む形状を網羅する。
/// `mse_matches_cpu_across_shapes` と同型。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn nll_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let nll = CudaNll::new(&device).expect("nll kernel compile must succeed");

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
        }, // class_dim = 先頭軸相当
        NllLayout {
            outer: 20,
            num_classes: 10,
            inner: 30,
        }, // class_dim = 中間軸相当
        NllLayout {
            outer: 300,
            num_classes: 1000,
            inner: 1,
        }, // NLL_MAX_BLOCKS 跨ぎ
        NllLayout {
            outer: 300_000,
            num_classes: 4,
            inner: 1,
        }, // n_samples=300_000 > NLL_BLOCK_DIM*NLL_MAX_BLOCKS=262_144
           // （grid-stride ループが `n_samples` 全域を確実に走査することの
           // 実機確認。PR #1850 codex-review P2 是正）
    ];

    for (idx, &layout) in layouts.iter().enumerate() {
        for mean in [true, false] {
            assert_nll_forward_parity(&nll, 9201 + idx as u64, layout, mean);
        }
        assert_nll_backward_parity(&nll, 9301 + idx as u64, layout);
    }
}
