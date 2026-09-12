//! イシュー #1647: RNN／LSTM／GRU セル演算（pointwise 段・backward 段）
//! の CPU-Metal 数値一致検証。
//!
//! `mse_parity.rs` と同じ構成方針: Metal 実機（Apple Silicon）依存の
//! ため `#![cfg(target_os = "macos")]` でファイル全体を macOS 限定に
//! し、各テストに `#[ignore]` を付けて通常 CI では実行しない。判定式・
//! 許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の参照と
//! する（`.claude/rules/coding-rust.md`）。
//!
//! CPU 参照実装は `fandhe_ai_backend_cpu::CpuBackendOps::{lstm_pointwise,
//! lstm_hidden_backward, lstm_cell_backward, gru_pointwise, gru_backward}`
//! （既に融合カーネルであり、`fandhe_ai_autodiff::eval` のホスト参照
//! 実装と一致検証済み。`crates/backend-cpu/tests/rnn_cell_parity.rs`）。
//!
//! 実行コマンド（Mac 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test rnn_cell_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::{MetalContext, MetalRnnCell};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn rand_vec(seed: u64, n: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(n)
}

fn assert_lstm_pointwise_parity(
    ctx: &MetalContext,
    rnn: &MetalRnnCell,
    seed: u64,
    b: usize,
    h: usize,
) {
    let pre = rand_vec(seed, b * 4 * h);
    let c_prev = rand_vec(seed.wrapping_add(1), b * h);

    let (gpu_gates, gpu_c, gpu_h) = rnn
        .run_lstm_pointwise_f32(ctx, &pre, &c_prev, h)
        .expect("MetalRnnCell::run_lstm_pointwise_f32 must succeed on Metal-equipped runner");

    let cpu = CpuBackendOps::new();
    let pre_t = Tensor::new(pre, &[b, 4 * h]).unwrap();
    let c_prev_t = Tensor::new(c_prev, &[b, h]).unwrap();
    let cpu_out = cpu.lstm_pointwise(&pre_t, &c_prev_t).unwrap();

    assert_parity(
        &format!("lstm_pointwise gates cpu-metal parity b={b} h={h}"),
        &gpu_gates,
        cpu_out.gates.as_slice().unwrap(),
    );
    assert_parity(
        &format!("lstm_pointwise c cpu-metal parity b={b} h={h}"),
        &gpu_c,
        cpu_out.c.as_slice().unwrap(),
    );
    assert_parity(
        &format!("lstm_pointwise h cpu-metal parity b={b} h={h}"),
        &gpu_h,
        cpu_out.h.as_slice().unwrap(),
    );
}

fn assert_lstm_hidden_backward_parity(
    ctx: &MetalContext,
    rnn: &MetalRnnCell,
    seed: u64,
    b: usize,
    h: usize,
) {
    let c = rand_vec(seed, b * h);
    let gate_o: Vec<f32> = rand_vec(seed.wrapping_add(1), b * h)
        .into_iter()
        .map(|v| (v.abs() * 0.4 + 0.3).min(0.95))
        .collect();
    let dh = rand_vec(seed.wrapping_add(2), b * h);

    let (gpu_d_pre_o, gpu_dc) = rnn
        .run_lstm_hidden_backward_f32(ctx, &c, &gate_o, &dh)
        .expect("MetalRnnCell::run_lstm_hidden_backward_f32 must succeed");

    let cpu = CpuBackendOps::new();
    let shape = [b, h];
    let (cpu_d_pre_o, cpu_dc) = cpu
        .lstm_hidden_backward(
            &Tensor::new(c, &shape).unwrap(),
            &Tensor::new(gate_o, &shape).unwrap(),
            &Tensor::new(dh, &shape).unwrap(),
        )
        .unwrap();

    assert_parity(
        &format!("lstm_hidden_backward d_pre_o cpu-metal parity b={b} h={h}"),
        &gpu_d_pre_o,
        cpu_d_pre_o.as_slice().unwrap(),
    );
    assert_parity(
        &format!("lstm_hidden_backward dc cpu-metal parity b={b} h={h}"),
        &gpu_dc,
        cpu_dc.as_slice().unwrap(),
    );
}

fn assert_lstm_cell_backward_parity(
    ctx: &MetalContext,
    rnn: &MetalRnnCell,
    seed: u64,
    b: usize,
    h: usize,
) {
    let pre = rand_vec(seed, b * 4 * h);
    let c_prev = rand_vec(seed.wrapping_add(1), b * h);
    let cpu = CpuBackendOps::new();
    let pre_t = Tensor::new(pre, &[b, 4 * h]).unwrap();
    let c_prev_t = Tensor::new(c_prev.clone(), &[b, h]).unwrap();
    let fwd = cpu.lstm_pointwise(&pre_t, &c_prev_t).unwrap();
    let gates_ifg = fwd.gates.narrow(1, 0, 3 * h).unwrap().contiguous();
    let dc = rand_vec(seed.wrapping_add(2), b * h);

    let gates_slice = gates_ifg.as_slice().unwrap();
    let (gpu_d_pre_ifg, gpu_dc_prev) = rnn
        .run_lstm_cell_backward_f32(ctx, gates_slice, &c_prev, &dc, h)
        .expect("MetalRnnCell::run_lstm_cell_backward_f32 must succeed");

    let (cpu_d_pre_ifg, cpu_dc_prev) = cpu
        .lstm_cell_backward(
            &gates_ifg,
            &Tensor::new(c_prev, &[b, h]).unwrap(),
            &Tensor::new(dc, &[b, h]).unwrap(),
        )
        .unwrap();

    assert_parity(
        &format!("lstm_cell_backward d_pre_ifg cpu-metal parity b={b} h={h}"),
        &gpu_d_pre_ifg,
        cpu_d_pre_ifg.as_slice().unwrap(),
    );
    assert_parity(
        &format!("lstm_cell_backward dc_prev cpu-metal parity b={b} h={h}"),
        &gpu_dc_prev,
        cpu_dc_prev.as_slice().unwrap(),
    );
}

fn assert_gru_pointwise_parity(
    ctx: &MetalContext,
    rnn: &MetalRnnCell,
    seed: u64,
    b: usize,
    h: usize,
) {
    let pre_i = rand_vec(seed, b * 3 * h);
    let pre_h = rand_vec(seed.wrapping_add(1), b * 3 * h);
    let h_prev = rand_vec(seed.wrapping_add(2), b * h);

    let (gpu_gates, gpu_q, gpu_h) = rnn
        .run_gru_pointwise_f32(ctx, &pre_i, &pre_h, &h_prev, h)
        .expect("MetalRnnCell::run_gru_pointwise_f32 must succeed");

    let cpu = CpuBackendOps::new();
    let cpu_out = cpu
        .gru_pointwise(
            &Tensor::new(pre_i, &[b, 3 * h]).unwrap(),
            &Tensor::new(pre_h, &[b, 3 * h]).unwrap(),
            &Tensor::new(h_prev, &[b, h]).unwrap(),
        )
        .unwrap();

    assert_parity(
        &format!("gru_pointwise gates cpu-metal parity b={b} h={h}"),
        &gpu_gates,
        cpu_out.gates.as_slice().unwrap(),
    );
    assert_parity(
        &format!("gru_pointwise q cpu-metal parity b={b} h={h}"),
        &gpu_q,
        cpu_out.q.as_slice().unwrap(),
    );
    assert_parity(
        &format!("gru_pointwise h cpu-metal parity b={b} h={h}"),
        &gpu_h,
        cpu_out.h.as_slice().unwrap(),
    );
}

fn assert_gru_backward_parity(
    ctx: &MetalContext,
    rnn: &MetalRnnCell,
    seed: u64,
    b: usize,
    h: usize,
) {
    let pre_i = rand_vec(seed, b * 3 * h);
    let pre_h = rand_vec(seed.wrapping_add(1), b * 3 * h);
    let h_prev = rand_vec(seed.wrapping_add(2), b * h);
    let cpu = CpuBackendOps::new();
    let fwd = cpu
        .gru_pointwise(
            &Tensor::new(pre_i, &[b, 3 * h]).unwrap(),
            &Tensor::new(pre_h, &[b, 3 * h]).unwrap(),
            &Tensor::new(h_prev.clone(), &[b, h]).unwrap(),
        )
        .unwrap();
    let dh = rand_vec(seed.wrapping_add(3), b * h);

    let gates_slice = fwd.gates.as_slice().unwrap();
    let q_slice = fwd.q.as_slice().unwrap();
    let (gpu_d_pre_i, gpu_d_pre_h, gpu_dh_prev_direct) = rnn
        .run_gru_backward_f32(ctx, gates_slice, q_slice, &h_prev, &dh, h)
        .expect("MetalRnnCell::run_gru_backward_f32 must succeed");

    let (cpu_d_pre_i, cpu_d_pre_h, cpu_dh_prev_direct) = cpu
        .gru_backward(
            &fwd.gates,
            &fwd.q,
            &Tensor::new(h_prev, &[b, h]).unwrap(),
            &Tensor::new(dh, &[b, h]).unwrap(),
        )
        .unwrap();

    assert_parity(
        &format!("gru_backward d_pre_i cpu-metal parity b={b} h={h}"),
        &gpu_d_pre_i,
        cpu_d_pre_i.as_slice().unwrap(),
    );
    assert_parity(
        &format!("gru_backward d_pre_h cpu-metal parity b={b} h={h}"),
        &gpu_d_pre_h,
        cpu_d_pre_h.as_slice().unwrap(),
    );
    assert_parity(
        &format!("gru_backward dh_prev_direct cpu-metal parity b={b} h={h}"),
        &gpu_dh_prev_direct,
        cpu_dh_prev_direct.as_slice().unwrap(),
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn rnn_cell_matches_cpu_across_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let rnn = MetalRnnCell::new(&ctx).expect("rnn cell パイプラインの構築に失敗した");

    let mut seed = 4000u64;
    for &(b, h) in &[(1usize, 1usize), (3, 7), (64, 64), (257, 300)] {
        seed += 1;
        assert_lstm_pointwise_parity(&ctx, &rnn, seed, b, h);
        seed += 1;
        assert_lstm_hidden_backward_parity(&ctx, &rnn, seed, b, h);
        seed += 1;
        assert_lstm_cell_backward_parity(&ctx, &rnn, seed, b, h);
        seed += 1;
        assert_gru_pointwise_parity(&ctx, &rnn, seed, b, h);
        seed += 1;
        assert_gru_backward_parity(&ctx, &rnn, seed, b, h);
    }
}
