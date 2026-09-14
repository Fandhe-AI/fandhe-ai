//! イシュー #1737: BCE 損失の融合カーネル（forward 2 段 reduction・
//! backward 1 段）の CPU-CUDA 数値一致検証（`mse_parity.rs` と同型
//! 構成）。
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
//! eval::bce_elem_loss`／`bce_elem_grad_input` と数式的に同一。
//! `mse_parity.rs::cpu_mse_forward_reference` と同型）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test bce_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaBce, CudaDevice, CudaError};
use fandhe_ai_tensor_core::BceKind;

mod common;

/// テスト専用 CPU 参照実装（素朴な逐次実装。`backend-cpu::bce` の融合
/// カーネルとは独立に丸め手順を分離する。一致判定は REQ-2 複合判定
/// 〈`fandhe_ai_backend_cpu::parity::assert_parity`〉に依るため問題ない）。
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

/// `[0, 1]` 開区間に収まる決定的な確率入力（`Probabilities` 用。実機
/// テストで乱数系列から確率を作るための正規化。`Xorshift64Star` は
/// `[0, 1)` を返すため微小 epsilon で両端を避ける）。
fn to_probability(raw: f32) -> f32 {
    raw.clamp(1e-4, 1.0 - 1e-4)
}

fn assert_bce_forward_parity(bce: &CudaBce, seed: u64, numel: usize, kind: BceKind, mean: bool) {
    let raw_input = Xorshift64Star::new(seed).fill_vec(numel);
    let raw_target = Xorshift64Star::new(seed.wrapping_add(1)).fill_vec(numel);
    let (input, target): (Vec<f32>, Vec<f32>) = match kind {
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
    };
    let factor = if mean {
        if numel == 0 { 1.0 } else { 1.0 / numel as f32 }
    } else {
        1.0
    };

    let gpu_value = bce
        .run_bce_loss_f32(&input, &target, kind, factor)
        .expect("CudaBce::run_bce_loss_f32 must succeed on CUDA-equipped test runner");
    let cpu_value = cpu_bce_forward_reference(&input, &target, kind, mean);

    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("bce forward cpu-cuda parity numel={numel} kind={kind:?} mean={mean}"),
        &[gpu_value],
        &[cpu_value],
    );
}

fn assert_bce_backward_parity(bce: &CudaBce, seed: u64, numel: usize, kind: BceKind) {
    let raw_input = Xorshift64Star::new(seed).fill_vec(numel);
    let raw_target = Xorshift64Star::new(seed.wrapping_add(1)).fill_vec(numel);
    let (input, target): (Vec<f32>, Vec<f32>) = match kind {
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
    };
    let scale = 1.7f32;

    let gpu_out = bce
        .run_bce_backward_f32(&input, &target, kind, scale)
        .expect("CudaBce::run_bce_backward_f32 must succeed on CUDA-equipped test runner");
    let cpu_out = cpu_bce_backward_reference(&input, &target, kind, scale);

    assert_eq!(gpu_out.len(), cpu_out.len());
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("bce backward cpu-cuda parity numel={numel} kind={kind:?}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。`mse_parity.rs::
/// mse_parity_smoke_env_adaptive` と同じ分岐パターン。
#[test]
fn bce_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaBce::new(&device) {
        Ok(bce) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            for kind in [BceKind::Probabilities, BceKind::Logits] {
                assert_bce_forward_parity(&bce, 1801, 0, kind, true);
                assert_bce_forward_parity(&bce, 1803, 1, kind, true);
                assert_bce_forward_parity(&bce, 1805, 300, kind, true);
                assert_bce_forward_parity(&bce, 1807, 300, kind, false);
                assert_bce_backward_parity(&bce, 1809, 0, kind);
                assert_bce_backward_parity(&bce, 1811, 300, kind);
            }
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {
            // NVRTC 非搭載環境。panic しないことのみ確認する
            // （`mse_parity.rs` と同じ理由）。
        }
        Err(other) => panic!("unexpected error variant for CudaBce::new: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`bce_num_blocks` の分岐点
/// （`BCE_BLOCK_DIM=256` 単位・`BCE_MAX_BLOCKS=1024` 上限跨ぎ）を含む
/// 形状を網羅する。`mse_matches_cpu_across_shapes` と同型。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn bce_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let bce = CudaBce::new(&device).expect("bce kernel compile must succeed");

    for &numel in &[0usize, 1, 100, 256, 257, 10_000, 300_000] {
        for kind in [BceKind::Probabilities, BceKind::Logits] {
            for mean in [true, false] {
                assert_bce_forward_parity(&bce, 9201 + numel as u64, numel, kind, mean);
            }
            assert_bce_backward_parity(&bce, 9301 + numel as u64, numel, kind);
        }
    }
}
