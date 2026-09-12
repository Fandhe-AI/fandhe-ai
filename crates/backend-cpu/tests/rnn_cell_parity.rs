//! `CpuBackendOps::{lstm_pointwise,lstm_hidden_backward,
//! lstm_cell_backward,gru_pointwise,gru_backward}`（イシュー #1647）と
//! 素朴な参照実装（本ファイル内。`mul_add` 不使用・逐次スカラー計算）の
//! 数値一致検証。
//!
//! `rnn_cell.rs` 側は `mul_add`（FMA）・rayon 並列化を使うため丸め手順が
//! 異なりうる。突合は `mse_parity.rs` と同じ統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）で行う。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn naive_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// ゲート配置 `i,f,g,o`。`pre: [B,4H]`・`c_prev: [B,H]`。
fn naive_lstm_pointwise(
    pre: &[f32],
    c_prev: &[f32],
    hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let b_dim = c_prev.len() / hidden;
    let mut gates = vec![0f32; b_dim * 4 * hidden];
    let mut c = vec![0f32; b_dim * hidden];
    let mut h = vec![0f32; b_dim * hidden];
    for b in 0..b_dim {
        for j in 0..hidden {
            let base = b * 4 * hidden;
            let i_val = naive_sigmoid(pre[base + j]);
            let f_val = naive_sigmoid(pre[base + hidden + j]);
            let g_val = pre[base + 2 * hidden + j].tanh();
            let o_val = naive_sigmoid(pre[base + 3 * hidden + j]);
            gates[base + j] = i_val;
            gates[base + hidden + j] = f_val;
            gates[base + 2 * hidden + j] = g_val;
            gates[base + 3 * hidden + j] = o_val;
            let c_val = f_val * c_prev[b * hidden + j] + i_val * g_val;
            c[b * hidden + j] = c_val;
            h[b * hidden + j] = o_val * c_val.tanh();
        }
    }
    (gates, c, h)
}

fn naive_lstm_hidden_backward(c: &[f32], gate_o: &[f32], dh: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut d_pre_o = vec![0f32; c.len()];
    let mut dc = vec![0f32; c.len()];
    for idx in 0..c.len() {
        let tanh_c = c[idx].tanh();
        d_pre_o[idx] = dh[idx] * tanh_c * gate_o[idx] * (1.0 - gate_o[idx]);
        dc[idx] = dh[idx] * gate_o[idx] * (1.0 - tanh_c * tanh_c);
    }
    (d_pre_o, dc)
}

fn naive_lstm_cell_backward(
    gates_ifg: &[f32],
    c_prev: &[f32],
    dc: &[f32],
    hidden: usize,
) -> (Vec<f32>, Vec<f32>) {
    let b_dim = c_prev.len() / hidden;
    let mut d_pre_ifg = vec![0f32; b_dim * 3 * hidden];
    let mut dc_prev = vec![0f32; b_dim * hidden];
    for b in 0..b_dim {
        for j in 0..hidden {
            let base3 = b * 3 * hidden;
            let i_val = gates_ifg[base3 + j];
            let f_val = gates_ifg[base3 + hidden + j];
            let g_val = gates_ifg[base3 + 2 * hidden + j];
            let dc_val = dc[b * hidden + j];
            d_pre_ifg[base3 + j] = dc_val * g_val * i_val * (1.0 - i_val);
            d_pre_ifg[base3 + hidden + j] = dc_val * c_prev[b * hidden + j] * f_val * (1.0 - f_val);
            d_pre_ifg[base3 + 2 * hidden + j] = dc_val * i_val * (1.0 - g_val * g_val);
            dc_prev[b * hidden + j] = dc_val * f_val;
        }
    }
    (d_pre_ifg, dc_prev)
}

fn naive_gru_pointwise(
    pre_i: &[f32],
    pre_h: &[f32],
    h_prev: &[f32],
    hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let b_dim = h_prev.len() / hidden;
    let mut gates = vec![0f32; b_dim * 3 * hidden];
    let mut q = vec![0f32; b_dim * hidden];
    let mut h = vec![0f32; b_dim * hidden];
    for b in 0..b_dim {
        for j in 0..hidden {
            let base3 = b * 3 * hidden;
            let r_pre = pre_i[base3 + j] + pre_h[base3 + j];
            let z_pre = pre_i[base3 + hidden + j] + pre_h[base3 + hidden + j];
            let q_val = pre_h[base3 + 2 * hidden + j];
            let pre_i_n = pre_i[base3 + 2 * hidden + j];
            let r_val = naive_sigmoid(r_pre);
            let z_val = naive_sigmoid(z_pre);
            let n_val = (r_val * q_val + pre_i_n).tanh();
            gates[base3 + j] = r_val;
            gates[base3 + hidden + j] = z_val;
            gates[base3 + 2 * hidden + j] = n_val;
            q[b * hidden + j] = q_val;
            h[b * hidden + j] = z_val * h_prev[b * hidden + j] + (1.0 - z_val) * n_val;
        }
    }
    (gates, q, h)
}

fn naive_gru_backward(
    gates_rzn: &[f32],
    q: &[f32],
    h_prev: &[f32],
    dh: &[f32],
    hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let b_dim = h_prev.len() / hidden;
    let mut d_pre_i = vec![0f32; b_dim * 3 * hidden];
    let mut d_pre_h = vec![0f32; b_dim * 3 * hidden];
    let mut dh_prev_direct = vec![0f32; b_dim * hidden];
    for b in 0..b_dim {
        for j in 0..hidden {
            let base3 = b * 3 * hidden;
            let r_val = gates_rzn[base3 + j];
            let z_val = gates_rzn[base3 + hidden + j];
            let n_val = gates_rzn[base3 + 2 * hidden + j];
            let q_val = q[b * hidden + j];
            let h_prev_val = h_prev[b * hidden + j];
            let dh_val = dh[b * hidden + j];
            let dn = dh_val * (1.0 - z_val);
            let dz = dh_val * (h_prev_val - n_val);
            let d_pre_n = dn * (1.0 - n_val * n_val);
            let dr = d_pre_n * q_val;
            let d_pre_r = dr * r_val * (1.0 - r_val);
            let d_pre_z = dz * z_val * (1.0 - z_val);
            d_pre_i[base3 + j] = d_pre_r;
            d_pre_i[base3 + hidden + j] = d_pre_z;
            d_pre_i[base3 + 2 * hidden + j] = d_pre_n;
            d_pre_h[base3 + j] = d_pre_r;
            d_pre_h[base3 + hidden + j] = d_pre_z;
            d_pre_h[base3 + 2 * hidden + j] = d_pre_n * r_val;
            dh_prev_direct[b * hidden + j] = dh_val * z_val;
        }
    }
    (d_pre_i, d_pre_h, dh_prev_direct)
}

fn shapes() -> Vec<(usize, usize)> {
    vec![(1, 1), (3, 7), (64, 64), (257, 300)]
}

fn deterministic_data(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.017).sin() * scale + offset)
        .collect()
}

#[test]
fn lstm_pointwise_matches_naive() {
    let ops = CpuBackendOps::new();
    for (b, h) in shapes() {
        let pre = deterministic_data(b * 4 * h, 3.0, 0.1);
        let c_prev = deterministic_data(b * h, 1.5, -0.2);

        let pre_t = Tensor::new(pre.clone(), &[b, 4 * h]).unwrap();
        let c_prev_t = Tensor::new(c_prev.clone(), &[b, h]).unwrap();

        let got = ops.lstm_pointwise(&pre_t, &c_prev_t).unwrap();
        let (exp_gates, exp_c, exp_h) = naive_lstm_pointwise(&pre, &c_prev, h);

        assert_eq!(got.gates.shape(), &[b, 4 * h]);
        assert_eq!(got.c.shape(), &[b, h]);
        assert_eq!(got.h.shape(), &[b, h]);
        assert_parity(
            &format!("lstm_pointwise gates b={b} h={h}"),
            got.gates.as_slice().unwrap(),
            &exp_gates,
        );
        assert_parity(
            &format!("lstm_pointwise c b={b} h={h}"),
            got.c.as_slice().unwrap(),
            &exp_c,
        );
        assert_parity(
            &format!("lstm_pointwise h b={b} h={h}"),
            got.h.as_slice().unwrap(),
            &exp_h,
        );
    }
}

#[test]
fn lstm_pointwise_all_zero_input_is_finite() {
    let ops = CpuBackendOps::new();
    let b = 2;
    let h = 4;
    let pre = Tensor::new(vec![0.0f32; b * 4 * h], &[b, 4 * h]).unwrap();
    let c_prev = Tensor::new(vec![0.0f32; b * h], &[b, h]).unwrap();
    let got = ops.lstm_pointwise(&pre, &c_prev).unwrap();
    for v in got.h.as_slice().unwrap() {
        assert!(v.is_finite());
    }
}

#[test]
fn lstm_pointwise_large_negative_input_is_finite_stable_sigmoid() {
    // 数値安定形 sigmoid の検証（大きな負値でオーバーフローしない）。
    let ops = CpuBackendOps::new();
    let b = 1;
    let h = 2;
    let pre = Tensor::new(vec![-1000.0f32; b * 4 * h], &[b, 4 * h]).unwrap();
    let c_prev = Tensor::new(vec![0.0f32; b * h], &[b, h]).unwrap();
    let got = ops.lstm_pointwise(&pre, &c_prev).unwrap();
    for v in got.gates.as_slice().unwrap() {
        assert!(v.is_finite(), "gate value should be finite, got {v}");
    }
}

#[test]
fn lstm_pointwise_propagates_nan() {
    let ops = CpuBackendOps::new();
    let b = 1;
    let h = 2;
    let mut pre_data = vec![0.1f32; b * 4 * h];
    pre_data[0] = f32::NAN;
    let pre = Tensor::new(pre_data, &[b, 4 * h]).unwrap();
    let c_prev = Tensor::new(vec![0.0f32; b * h], &[b, h]).unwrap();
    let got = ops.lstm_pointwise(&pre, &c_prev).unwrap();
    assert!(got.gates.as_slice().unwrap()[0].is_nan());
    assert!(got.h.as_slice().unwrap()[0].is_nan());
}

#[test]
fn lstm_hidden_backward_matches_naive() {
    let ops = CpuBackendOps::new();
    for (b, h) in shapes() {
        let c = deterministic_data(b * h, 1.2, 0.05);
        let gate_o = deterministic_data(b * h, 0.4, 0.5)
            .into_iter()
            .map(|v| v.clamp(0.01, 0.99))
            .collect::<Vec<_>>();
        let dh = deterministic_data(b * h, 0.8, 0.0);

        let c_t = Tensor::new(c.clone(), &[b, h]).unwrap();
        let gate_o_t = Tensor::new(gate_o.clone(), &[b, h]).unwrap();
        let dh_t = Tensor::new(dh.clone(), &[b, h]).unwrap();

        let (got_d_pre_o, got_dc) = ops.lstm_hidden_backward(&c_t, &gate_o_t, &dh_t).unwrap();
        let (exp_d_pre_o, exp_dc) = naive_lstm_hidden_backward(&c, &gate_o, &dh);

        assert_parity(
            &format!("lstm_hidden_backward d_pre_o b={b} h={h}"),
            got_d_pre_o.as_slice().unwrap(),
            &exp_d_pre_o,
        );
        assert_parity(
            &format!("lstm_hidden_backward dc b={b} h={h}"),
            got_dc.as_slice().unwrap(),
            &exp_dc,
        );
    }
}

#[test]
fn lstm_cell_backward_matches_naive() {
    let ops = CpuBackendOps::new();
    for (b, h) in shapes() {
        let pre = deterministic_data(b * 4 * h, 2.0, 0.0);
        let c_prev = deterministic_data(b * h, 1.0, 0.0);
        let (gates, _c, _h) = naive_lstm_pointwise(&pre, &c_prev, h);
        let gates_ifg: Vec<f32> = (0..b)
            .flat_map(|bi| gates[bi * 4 * h..bi * 4 * h + 3 * h].to_vec())
            .collect();
        let dc = deterministic_data(b * h, 0.6, 0.1);

        let gates_t = Tensor::new(gates_ifg.clone(), &[b, 3 * h]).unwrap();
        let c_prev_t = Tensor::new(c_prev.clone(), &[b, h]).unwrap();
        let dc_t = Tensor::new(dc.clone(), &[b, h]).unwrap();

        let (got_d_pre_ifg, got_dc_prev) =
            ops.lstm_cell_backward(&gates_t, &c_prev_t, &dc_t).unwrap();
        let (exp_d_pre_ifg, exp_dc_prev) = naive_lstm_cell_backward(&gates_ifg, &c_prev, &dc, h);

        assert_parity(
            &format!("lstm_cell_backward d_pre_ifg b={b} h={h}"),
            got_d_pre_ifg.as_slice().unwrap(),
            &exp_d_pre_ifg,
        );
        assert_parity(
            &format!("lstm_cell_backward dc_prev b={b} h={h}"),
            got_dc_prev.as_slice().unwrap(),
            &exp_dc_prev,
        );
    }
}

#[test]
fn gru_pointwise_matches_naive() {
    let ops = CpuBackendOps::new();
    for (b, h) in shapes() {
        let pre_i = deterministic_data(b * 3 * h, 2.5, 0.0);
        let pre_h = deterministic_data(b * 3 * h, 1.8, 0.2);
        let h_prev = deterministic_data(b * h, 1.0, -0.1);

        let pre_i_t = Tensor::new(pre_i.clone(), &[b, 3 * h]).unwrap();
        let pre_h_t = Tensor::new(pre_h.clone(), &[b, 3 * h]).unwrap();
        let h_prev_t = Tensor::new(h_prev.clone(), &[b, h]).unwrap();

        let got = ops.gru_pointwise(&pre_i_t, &pre_h_t, &h_prev_t).unwrap();
        let (exp_gates, exp_q, exp_h) = naive_gru_pointwise(&pre_i, &pre_h, &h_prev, h);

        assert_parity(
            &format!("gru_pointwise gates b={b} h={h}"),
            got.gates.as_slice().unwrap(),
            &exp_gates,
        );
        assert_parity(
            &format!("gru_pointwise q b={b} h={h}"),
            got.q.as_slice().unwrap(),
            &exp_q,
        );
        assert_parity(
            &format!("gru_pointwise h b={b} h={h}"),
            got.h.as_slice().unwrap(),
            &exp_h,
        );
    }
}

#[test]
fn gru_backward_matches_naive() {
    let ops = CpuBackendOps::new();
    for (b, h) in shapes() {
        let pre_i = deterministic_data(b * 3 * h, 2.0, 0.1);
        let pre_h = deterministic_data(b * 3 * h, 1.5, -0.1);
        let h_prev = deterministic_data(b * h, 1.0, 0.0);
        let (gates, q, _h) = naive_gru_pointwise(&pre_i, &pre_h, &h_prev, h);
        let dh = deterministic_data(b * h, 0.7, 0.0);

        let gates_t = Tensor::new(gates.clone(), &[b, 3 * h]).unwrap();
        let q_t = Tensor::new(q.clone(), &[b, h]).unwrap();
        let h_prev_t = Tensor::new(h_prev.clone(), &[b, h]).unwrap();
        let dh_t = Tensor::new(dh.clone(), &[b, h]).unwrap();

        let (got_d_pre_i, got_d_pre_h, got_dh_prev_direct) =
            ops.gru_backward(&gates_t, &q_t, &h_prev_t, &dh_t).unwrap();
        let (exp_d_pre_i, exp_d_pre_h, exp_dh_prev_direct) =
            naive_gru_backward(&gates, &q, &h_prev, &dh, h);

        assert_parity(
            &format!("gru_backward d_pre_i b={b} h={h}"),
            got_d_pre_i.as_slice().unwrap(),
            &exp_d_pre_i,
        );
        assert_parity(
            &format!("gru_backward d_pre_h b={b} h={h}"),
            got_d_pre_h.as_slice().unwrap(),
            &exp_d_pre_h,
        );
        assert_parity(
            &format!("gru_backward dh_prev_direct b={b} h={h}"),
            got_dh_prev_direct.as_slice().unwrap(),
            &exp_dh_prev_direct,
        );
    }
}
