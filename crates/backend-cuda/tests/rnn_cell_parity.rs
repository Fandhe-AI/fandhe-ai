//! イシュー #1647: RNN／LSTM／GRU セル演算（pointwise 段・backward 段。
//! `lstm_pointwise`／`lstm_hidden_backward`／`lstm_cell_backward`／
//! `gru_pointwise`／`gru_backward` の全 5 種。codex-review P2 指摘を
//! 受け `lstm_cell_backward`／`gru_backward` も網羅した）の
//! CPU-CUDA 数値一致検証。
//!
//! `mse_parity.rs` と同じ構成方針を踏襲する: 環境適応スモーク（属性
//! なし。通常 CI で実行し、CUDA 非搭載環境では
//! `fandhe_ai_backend_cuda::CudaError::DriverUnavailable`／
//! `NvrtcUnavailable` を確認して panic しないことのみ検証）と、実機必須
//! の形状網羅（`#[ignore]`。DGX Spark GB10 等）を分離する。判定式・
//! 許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の参照と
//! する（`.claude/rules/coding-rust.md`）。
//!
//! CPU 参照実装は `fandhe_ai_backend_cpu::CpuBackendOps::{lstm_pointwise,
//! lstm_hidden_backward, lstm_cell_backward, gru_pointwise, gru_backward}`
//! （既に融合カーネルであり、`fandhe_ai_autodiff::eval` のホスト参照
//! 実装と一致検証済み。`crates/backend-cpu/tests/rnn_cell_parity.rs`）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test rnn_cell_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaRnnCell};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

mod common;

fn rand_vec(seed: u64, n: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(n)
}

fn assert_lstm_pointwise_parity(rnn: &CudaRnnCell, seed: u64, b: usize, h: usize) {
    let pre = rand_vec(seed, b * 4 * h);
    let c_prev = rand_vec(seed.wrapping_add(1), b * h);

    let (gpu_gates, gpu_c, gpu_h) = rnn
        .run_lstm_pointwise_f32(&pre, &c_prev, h)
        .expect("CudaRnnCell::run_lstm_pointwise_f32 must succeed on CUDA-equipped runner");

    let cpu = CpuBackendOps::new();
    let pre_t = Tensor::new(pre, &[b, 4 * h]).unwrap();
    let c_prev_t = Tensor::new(c_prev, &[b, h]).unwrap();
    let cpu_out = cpu.lstm_pointwise(&pre_t, &c_prev_t).unwrap();

    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("lstm_pointwise gates cpu-cuda parity b={b} h={h}"),
        &gpu_gates,
        cpu_out.gates.as_slice().unwrap(),
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("lstm_pointwise c cpu-cuda parity b={b} h={h}"),
        &gpu_c,
        cpu_out.c.as_slice().unwrap(),
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("lstm_pointwise h cpu-cuda parity b={b} h={h}"),
        &gpu_h,
        cpu_out.h.as_slice().unwrap(),
    );
}

fn assert_lstm_hidden_backward_parity(rnn: &CudaRnnCell, seed: u64, b: usize, h: usize) {
    let c = rand_vec(seed, b * h);
    let gate_o: Vec<f32> = rand_vec(seed.wrapping_add(1), b * h)
        .into_iter()
        .map(|v| (v.abs() * 0.4 + 0.3).min(0.95))
        .collect();
    let dh = rand_vec(seed.wrapping_add(2), b * h);

    let (gpu_d_pre_o, gpu_dc) = rnn
        .run_lstm_hidden_backward_f32(&c, &gate_o, &dh)
        .expect("CudaRnnCell::run_lstm_hidden_backward_f32 must succeed");

    let cpu = CpuBackendOps::new();
    let shape = [b, h];
    let (cpu_d_pre_o, cpu_dc) = cpu
        .lstm_hidden_backward(
            &Tensor::new(c, &shape).unwrap(),
            &Tensor::new(gate_o, &shape).unwrap(),
            &Tensor::new(dh, &shape).unwrap(),
        )
        .unwrap();

    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("lstm_hidden_backward d_pre_o cpu-cuda parity b={b} h={h}"),
        &gpu_d_pre_o,
        cpu_d_pre_o.as_slice().unwrap(),
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("lstm_hidden_backward dc cpu-cuda parity b={b} h={h}"),
        &gpu_dc,
        cpu_dc.as_slice().unwrap(),
    );
}

fn assert_gru_pointwise_parity(rnn: &CudaRnnCell, seed: u64, b: usize, h: usize) {
    let pre_i = rand_vec(seed, b * 3 * h);
    let pre_h = rand_vec(seed.wrapping_add(1), b * 3 * h);
    let h_prev = rand_vec(seed.wrapping_add(2), b * h);

    let (gpu_gates, gpu_q, gpu_h) = rnn
        .run_gru_pointwise_f32(&pre_i, &pre_h, &h_prev, h)
        .expect("CudaRnnCell::run_gru_pointwise_f32 must succeed");

    let cpu = CpuBackendOps::new();
    let cpu_out = cpu
        .gru_pointwise(
            &Tensor::new(pre_i, &[b, 3 * h]).unwrap(),
            &Tensor::new(pre_h, &[b, 3 * h]).unwrap(),
            &Tensor::new(h_prev, &[b, h]).unwrap(),
        )
        .unwrap();

    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("gru_pointwise gates cpu-cuda parity b={b} h={h}"),
        &gpu_gates,
        cpu_out.gates.as_slice().unwrap(),
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("gru_pointwise q cpu-cuda parity b={b} h={h}"),
        &gpu_q,
        cpu_out.q.as_slice().unwrap(),
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("gru_pointwise h cpu-cuda parity b={b} h={h}"),
        &gpu_h,
        cpu_out.h.as_slice().unwrap(),
    );
}

fn assert_lstm_cell_backward_parity(rnn: &CudaRnnCell, seed: u64, b: usize, h: usize) {
    let gates_ifg = rand_vec(seed, b * 3 * h)
        .into_iter()
        .map(|v| (v.abs() * 0.4 + 0.3).min(0.95))
        .collect::<Vec<f32>>();
    let c_prev = rand_vec(seed.wrapping_add(1), b * h);
    let dc = rand_vec(seed.wrapping_add(2), b * h);

    let (gpu_d_pre_ifg, gpu_dc_prev) = rnn
        .run_lstm_cell_backward_f32(&gates_ifg, &c_prev, &dc, h)
        .expect("CudaRnnCell::run_lstm_cell_backward_f32 must succeed");

    let cpu = CpuBackendOps::new();
    let (cpu_d_pre_ifg, cpu_dc_prev) = cpu
        .lstm_cell_backward(
            &Tensor::new(gates_ifg, &[b, 3 * h]).unwrap(),
            &Tensor::new(c_prev, &[b, h]).unwrap(),
            &Tensor::new(dc, &[b, h]).unwrap(),
        )
        .unwrap();

    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("lstm_cell_backward d_pre_ifg cpu-cuda parity b={b} h={h}"),
        &gpu_d_pre_ifg,
        cpu_d_pre_ifg.as_slice().unwrap(),
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("lstm_cell_backward dc_prev cpu-cuda parity b={b} h={h}"),
        &gpu_dc_prev,
        cpu_dc_prev.as_slice().unwrap(),
    );
}

fn assert_gru_backward_parity(rnn: &CudaRnnCell, seed: u64, b: usize, h: usize) {
    let gates_rzn = rand_vec(seed, b * 3 * h)
        .into_iter()
        .map(|v| (v.abs() * 0.4 + 0.3).min(0.95))
        .collect::<Vec<f32>>();
    let q = rand_vec(seed.wrapping_add(1), b * h);
    let h_prev = rand_vec(seed.wrapping_add(2), b * h);
    let dh = rand_vec(seed.wrapping_add(3), b * h);

    let (gpu_d_pre_i, gpu_d_pre_h, gpu_dh_prev_direct) = rnn
        .run_gru_backward_f32(&gates_rzn, &q, &h_prev, &dh, h)
        .expect("CudaRnnCell::run_gru_backward_f32 must succeed");

    let cpu = CpuBackendOps::new();
    let (cpu_d_pre_i, cpu_d_pre_h, cpu_dh_prev_direct) = cpu
        .gru_backward(
            &Tensor::new(gates_rzn, &[b, 3 * h]).unwrap(),
            &Tensor::new(q, &[b, h]).unwrap(),
            &Tensor::new(h_prev, &[b, h]).unwrap(),
            &Tensor::new(dh, &[b, h]).unwrap(),
        )
        .unwrap();

    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("gru_backward d_pre_i cpu-cuda parity b={b} h={h}"),
        &gpu_d_pre_i,
        cpu_d_pre_i.as_slice().unwrap(),
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("gru_backward d_pre_h cpu-cuda parity b={b} h={h}"),
        &gpu_d_pre_h,
        cpu_d_pre_h.as_slice().unwrap(),
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("gru_backward dh_prev_direct cpu-cuda parity b={b} h={h}"),
        &gpu_dh_prev_direct,
        cpu_dh_prev_direct.as_slice().unwrap(),
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。
#[test]
fn rnn_cell_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaRnnCell::new(&device) {
        Ok(rnn) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            assert_lstm_pointwise_parity(&rnn, 2101, 3, 7);
            assert_lstm_hidden_backward_parity(&rnn, 2103, 3, 7);
            assert_gru_pointwise_parity(&rnn, 2105, 3, 7);
            assert_lstm_cell_backward_parity(&rnn, 2107, 3, 7);
            assert_gru_backward_parity(&rnn, 2109, 3, 7);
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {
            // NVRTC 非搭載環境。panic しないことのみ確認する
            // （`mse_parity.rs` と同じ分岐）。
        }
        Err(other) => panic!("unexpected error variant for CudaRnnCell::new: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn rnn_cell_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let rnn = CudaRnnCell::new(&device).expect("rnn cell kernel compile must succeed");

    for &(b, h) in &[(1usize, 1usize), (3, 7), (64, 64), (257, 300)] {
        assert_lstm_pointwise_parity(&rnn, 9001 + (b * h) as u64, b, h);
        assert_lstm_hidden_backward_parity(&rnn, 9101 + (b * h) as u64, b, h);
        assert_gru_pointwise_parity(&rnn, 9201 + (b * h) as u64, b, h);
        assert_lstm_cell_backward_parity(&rnn, 9301 + (b * h) as u64, b, h);
        assert_gru_backward_parity(&rnn, 9401 + (b * h) as u64, b, h);
    }
}
