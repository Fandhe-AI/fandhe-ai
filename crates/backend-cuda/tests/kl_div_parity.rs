//! イシュー #1738: KLDivLoss 損失の融合カーネル（forward 2 段
//! reduction・backward 1 段）の CPU-CUDA 数値一致検証（`bce_parity.rs`
//! と同型構成）。
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
//! eval::kl_div_elem_loss`／`kl_div_elem_grad_input` と数式的に同一。
//! `bce_parity.rs::cpu_bce_forward_reference` と同型）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test kl_div_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaKlDiv};
use fandhe_ai_tensor_core::KlDivTarget;

mod common;

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

/// `(1e-4, 1 - 1e-4)` 開区間に収まる決定的な確率入力（`Probabilities`
/// 用。`Xorshift64Star` は `[0, 1)` を返すため微小 epsilon で両端を
/// 避ける。`bce_parity.rs::to_probability` と同型）。
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
    kl_div: &CudaKlDiv,
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
        .run_kl_div_loss_f32(&input, &target, kind, factor)
        .expect("CudaKlDiv::run_kl_div_loss_f32 must succeed on CUDA-equipped test runner");
    let cpu_value = cpu_kl_div_forward_reference(&input, &target, kind, mean);

    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("kl_div forward cpu-cuda parity numel={numel} kind={kind:?} mean={mean}"),
        &[gpu_value],
        &[cpu_value],
    );
}

fn assert_kl_div_backward_parity(kl_div: &CudaKlDiv, seed: u64, numel: usize, kind: KlDivTarget) {
    let (input, target) = make_inputs(seed, numel, kind);
    let scale = 1.7f32;

    let gpu_out = kl_div
        .run_kl_div_backward_f32(&input, &target, kind, scale)
        .expect("CudaKlDiv::run_kl_div_backward_f32 must succeed on CUDA-equipped test runner");
    let cpu_out = cpu_kl_div_backward_reference(&input, &target, kind, scale);

    assert_eq!(gpu_out.len(), cpu_out.len());
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("kl_div backward cpu-cuda parity numel={numel} kind={kind:?}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。`mse_parity.rs::
/// mse_parity_smoke_env_adaptive` と同じ分岐パターン。
#[test]
fn kl_div_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaKlDiv::new(&device) {
        Ok(kl_div) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            for kind in [KlDivTarget::Probabilities, KlDivTarget::LogProbabilities] {
                assert_kl_div_forward_parity(&kl_div, 2101, 0, kind, true);
                assert_kl_div_forward_parity(&kl_div, 2103, 1, kind, true);
                assert_kl_div_forward_parity(&kl_div, 2105, 300, kind, true);
                assert_kl_div_forward_parity(&kl_div, 2107, 300, kind, false);
                assert_kl_div_backward_parity(&kl_div, 2109, 0, kind);
                assert_kl_div_backward_parity(&kl_div, 2111, 300, kind);
            }
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {
            // NVRTC 非搭載環境。panic しないことのみ確認する
            // （`mse_parity.rs` と同じ理由）。
        }
        Err(other) => panic!("unexpected error variant for CudaKlDiv::new: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`kl_div_num_blocks` の
/// 分岐点（`KL_DIV_BLOCK_DIM=256` 単位・`KL_DIV_MAX_BLOCKS=1024` 上限
/// 跨ぎ）を含む形状を網羅する。`mse_matches_cpu_across_shapes` と同型。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn kl_div_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let kl_div = CudaKlDiv::new(&device).expect("kl_div kernel compile must succeed");

    for &numel in &[0usize, 1, 100, 256, 257, 10_000, 300_000] {
        for kind in [KlDivTarget::Probabilities, KlDivTarget::LogProbabilities] {
            for mean in [true, false] {
                assert_kl_div_forward_parity(&kl_div, 9401 + numel as u64, numel, kind, mean);
            }
            assert_kl_div_backward_parity(&kl_div, 9501 + numel as u64, numel, kind);
        }
    }
}
