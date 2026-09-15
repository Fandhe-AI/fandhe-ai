//! イシュー #1739: Huber／SmoothL1 損失の融合カーネル（forward 2 段
//! reduction・backward 1 段）の CPU-CUDA 数値一致検証（`mse_parity.rs`
//! と同型の構成）。
//!
//! 環境適応スモーク（属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `DriverUnavailable`／`NvrtcUnavailable` を確認して panic しないことの
//! み検証）と、実機必須の形状網羅（`#[ignore]`。DGX Spark GB10 等）を
//! 分離する。判定式・許容誤差は再定義せず `fandhe_ai_backend_cpu::parity`
//! を唯一の参照とする（`.claude/rules/coding-rust.md`）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test huber_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaHuber};
use fandhe_ai_tensor_core::HuberKind;

mod common;

/// テスト専用 CPU 参照実装（素朴な逐次実装。`huber.rs` の融合カーネルと
/// は独立に丸め手順を分離する。一致判定は REQ-2 複合判定
/// 〈`fandhe_ai_backend_cpu::parity::assert_parity`〉に依るため問題ない）。
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

fn assert_huber_forward_parity(
    huber: &CudaHuber,
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
        .run_huber_loss_f32(&pred, &target, kind, delta, factor)
        .expect("CudaHuber::run_huber_loss_f32 must succeed on CUDA-equipped test runner");
    let cpu_value = cpu_huber_forward_reference(&pred, &target, kind, delta, mean);

    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!(
            "huber forward cpu-cuda parity numel={numel} kind={kind:?} delta={delta} mean={mean}"
        ),
        &[gpu_value],
        &[cpu_value],
    );
}

fn assert_huber_backward_parity(
    huber: &CudaHuber,
    seed: u64,
    numel: usize,
    kind: HuberKind,
    delta: f32,
) {
    let pred = Xorshift64Star::new(seed).fill_vec(numel);
    let target = Xorshift64Star::new(seed.wrapping_add(1)).fill_vec(numel);
    let scale = 1.7f32;

    let gpu_out = huber
        .run_huber_backward_f32(&pred, &target, kind, delta, scale)
        .expect("CudaHuber::run_huber_backward_f32 must succeed on CUDA-equipped test runner");
    let cpu_out = cpu_huber_backward_reference(&pred, &target, kind, delta, scale);

    assert_eq!(gpu_out.len(), cpu_out.len());
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("huber backward cpu-cuda parity numel={numel} kind={kind:?} delta={delta}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。`mse_parity.rs::
/// mse_parity_smoke_env_adaptive` と同じ分岐パターン。
#[test]
fn huber_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaHuber::new(&device) {
        Ok(huber) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            assert_huber_forward_parity(&huber, 1739, 0, HuberKind::Huber, 1.0, true);
            assert_huber_forward_parity(&huber, 1741, 1, HuberKind::Huber, 1.0, true);
            assert_huber_forward_parity(&huber, 1743, 300, HuberKind::Huber, 1.0, true);
            assert_huber_forward_parity(&huber, 1745, 300, HuberKind::SmoothL1, 2.0, false);
            assert_huber_backward_parity(&huber, 1747, 0, HuberKind::Huber, 1.0);
            assert_huber_backward_parity(&huber, 1749, 300, HuberKind::SmoothL1, 0.5);
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {
            // NVRTC 非搭載環境（`mse_parity.rs` と同じ分岐。panic しない
            // ことのみ確認する）。
        }
        Err(other) => panic!("unexpected error variant for CudaHuber::new: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`huber_num_blocks` の分岐点
/// （`HUBER_BLOCK_DIM=256` 単位・`HUBER_MAX_BLOCKS=1024` 上限跨ぎ）に
/// 加え、kind×delta の組合せを網羅する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn huber_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let huber = CudaHuber::new(&device).expect("huber kernel compile must succeed");

    // 0・1・単一ブロック未満・単一ブロックちょうど（256）・複数ブロック・
    // HUBER_MAX_BLOCKS（1024）*256=262144 を跨ぐ大サイズ。
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
                assert_huber_forward_parity(&huber, 9739 + numel as u64, numel, kind, delta, mean);
            }
            assert_huber_backward_parity(&huber, 9839 + numel as u64, numel, kind, delta);
        }
    }
}
