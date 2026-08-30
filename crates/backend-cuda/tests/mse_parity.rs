//! イシュー #1045: MSE 損失の融合カーネル（forward 2 段 reduction・
//! backward 1 段）の CPU-CUDA 数値一致検証。
//!
//! `softmax_parity.rs`（#594）と同じ構成方針を踏襲する: 環境適応スモーク
//! （属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `fandhe_ai_backend_cuda::CudaError::DriverUnavailable`／
//! `NvrtcUnavailable` を確認して panic しないことのみ検証）と、実機必須の
//! 形状網羅（`#[ignore]`。DGX Spark GB10 等）を分離する。判定式・許容
//! 誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の参照とする
//! （`.claude/rules/coding-rust.md`）。
//!
//! CPU 参照実装は `fandhe_ai_backend_cpu::CpuBackendOps::mse_loss`／
//! `mse_loss_backward`（融合カーネル。`backend-cuda` は既に
//! `dev-dependencies` に `fandhe-ai-backend-cpu` を持つ前提。持たない
//! 場合はテストローカルの素朴実装へフォールバックする方針だが、本ファイル
//! では `Cargo.toml` 変更なしに委ねられる既存 dev-dependency の有無を
//! 実装時に確認済み）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test mse_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaMse};

mod common;

/// テスト専用 CPU 参照実装（素朴な逐次実装。`mul_add`・固定チャンクは
/// 使わず、`backend-cpu::mse` の融合カーネルとは独立に丸め手順を分離
/// する。一致判定は REQ-2 複合判定〈`fandhe_ai_backend_cpu::parity::
/// assert_parity`〉に依るため問題ない）。
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

fn assert_mse_forward_parity(mse: &CudaMse, seed: u64, numel: usize, mean: bool) {
    let pred = Xorshift64Star::new(seed).fill_vec(numel);
    let target = Xorshift64Star::new(seed.wrapping_add(1)).fill_vec(numel);
    let factor = if mean {
        if numel == 0 { 1.0 } else { 1.0 / numel as f32 }
    } else {
        1.0
    };

    let gpu_value = mse
        .run_mse_loss_f32(&pred, &target, factor)
        .expect("CudaMse::run_mse_loss_f32 must succeed on CUDA-equipped test runner");
    let cpu_value = cpu_mse_forward_reference(&pred, &target, mean);

    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("mse forward cpu-cuda parity numel={numel} mean={mean}"),
        &[gpu_value],
        &[cpu_value],
    );
}

fn assert_mse_backward_parity(mse: &CudaMse, seed: u64, numel: usize) {
    let pred = Xorshift64Star::new(seed).fill_vec(numel);
    let target = Xorshift64Star::new(seed.wrapping_add(1)).fill_vec(numel);
    let scale = 1.7f32;

    let gpu_out = mse
        .run_mse_backward_f32(&pred, &target, scale)
        .expect("CudaMse::run_mse_backward_f32 must succeed on CUDA-equipped test runner");
    let cpu_out = cpu_mse_backward_reference(&pred, &target, scale);

    assert_eq!(gpu_out.len(), cpu_out.len());
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("mse backward cpu-cuda parity numel={numel}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。`softmax_parity.rs::
/// softmax_parity_smoke_env_adaptive` と同じ分岐パターン: 環境不在を表す
/// 既知の variant（`DriverUnavailable`／`NvrtcUnavailable`）のみを早期
/// return の対象とし、それ以外は `panic!` する。
#[test]
fn mse_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaMse::new(&device) {
        Ok(mse) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            assert_mse_forward_parity(&mse, 1701, 0, true);
            assert_mse_forward_parity(&mse, 1703, 1, true);
            assert_mse_forward_parity(&mse, 1705, 300, true);
            assert_mse_forward_parity(&mse, 1707, 300, false);
            assert_mse_backward_parity(&mse, 1709, 0);
            assert_mse_backward_parity(&mse, 1711, 300);
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {
            // NVRTC 非搭載環境（driver はあるが nvrtc が無い。本ラン環境
            // 〈RTX 3060・libnvrtc 不在〉はこの分岐を通る）。panic しない
            // ことのみ確認する。
        }
        Err(other) => panic!("unexpected error variant for CudaMse::new: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`mse_num_blocks` の分岐点
/// （`MSE_BLOCK_DIM=256` 単位・`MSE_MAX_BLOCKS=1024` 上限跨ぎ）を含む
/// 形状を網羅する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn mse_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let mse = CudaMse::new(&device).expect("mse kernel compile must succeed");

    // 0・1・単一ブロック未満・単一ブロックちょうど（256）・複数ブロック・
    // MSE_MAX_BLOCKS（1024）*256=262144 を跨ぐ大サイズ。
    for &numel in &[0usize, 1, 100, 256, 257, 10_000, 300_000] {
        for mean in [true, false] {
            assert_mse_forward_parity(&mse, 9001 + numel as u64, numel, mean);
        }
        assert_mse_backward_parity(&mse, 9101 + numel as u64, numel);
    }
}
