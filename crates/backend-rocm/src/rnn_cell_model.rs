//! `kernels_rnn_cell.rs` の 5 カーネルを CPU 上で (block, thread) ごとに再現するテスト専用モデル
//! （イシュー #3441・R10）。
//!
//! 役割: AMD 実機がない環境で、1 次元グリッド分解・`gid >= numel` ガード・**全出力配列の全要素が
//! ちょうど 1 回書かれること**（1 スレッドが gates の 3〜4 ブロック・c・h など複数配列へ書くため、
//! 配列ごとに書き込み回数を数える）・カーネル式（2 分岐シグモイド・FMA 3 か所）の Rust 訳を構造的に
//! 検証する。`RocmBackendOps::with_host_model` の経路（`ops/rnn_cell.rs`）が呼ぶ。
//!
//! 限界: 超越関数（`exp`・`tanh`）はホスト libm で、**GPU の libm を模したものではない**。したがって
//! 一致は構造の検証にとどまり、実機での CPU との parity は #3159 で測る。HIPRTC のコンパイル結果
//! そのものの検証ではない。FMA は CPU 参照と同じ 3 か所だけ `f32::mul_add`、それ以外は字面どおり。

use crate::launch::for_each_thread_gid;

/// 1 出力配列と、要素ごとの書き込み回数。
struct Out {
    data: Vec<f32>,
    writes: Vec<u8>,
}

impl Out {
    fn new(len: usize) -> Self {
        Self {
            data: vec![f32::NAN; len],
            writes: vec![0; len],
        }
    }

    fn put(&mut self, idx: usize, v: f32) {
        self.data[idx] = v;
        self.writes[idx] += 1;
    }

    /// 全要素がちょうど 1 回書かれたことを確認して値を返す。
    fn finish(self) -> Vec<f32> {
        assert!(
            self.writes.iter().all(|&w| w == 1),
            "各出力要素はちょうど 1 回書かれる"
        );
        self.data
    }
}

/// カーネルの `rnn_sigmoid`（2 分岐の数値安定形）。
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        x.exp() / (1.0 + x.exp())
    }
}

/// 全スレッドを回し、`gid < numel` のものだけ `f(gid, row, j)` を呼ぶ（範囲外 gid は素通り）。
fn grid(numel: usize, hidden: usize, mut f: impl FnMut(usize, usize, usize)) {
    for_each_thread_gid(numel, |gid| {
        if gid >= numel as i64 {
            return;
        }
        let g = gid as usize;
        f(g, g / hidden, g % hidden);
    });
}

/// `lstm_pointwise_f32`。`(gates, c, h)`。
pub(crate) fn lstm_pointwise(
    pre: &[f32],
    c_prev: &[f32],
    hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let numel = c_prev.len();
    let (mut gates, mut c_out, mut h_out) = (Out::new(numel * 4), Out::new(numel), Out::new(numel));
    grid(numel, hidden, |g, row, j| {
        let base = row * 4 * hidden;
        let i_val = sigmoid(pre[base + j]);
        let f_val = sigmoid(pre[base + hidden + j]);
        let g_val = pre[base + 2 * hidden + j].tanh();
        let o_val = sigmoid(pre[base + 3 * hidden + j]);
        gates.put(base + j, i_val);
        gates.put(base + hidden + j, f_val);
        gates.put(base + 2 * hidden + j, g_val);
        gates.put(base + 3 * hidden + j, o_val);
        let c_val = f_val.mul_add(c_prev[g], i_val * g_val);
        c_out.put(g, c_val);
        h_out.put(g, o_val * c_val.tanh());
    });
    (gates.finish(), c_out.finish(), h_out.finish())
}

/// `lstm_hidden_backward_f32`。`(d_pre_o, dc)`。
pub(crate) fn lstm_hidden_backward(c: &[f32], gate_o: &[f32], dh: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let numel = c.len();
    let (mut d_pre_o, mut dc) = (Out::new(numel), Out::new(numel));
    // hidden は使わないので 1 を渡す（row = gid, j = 0 になるだけ）。
    grid(numel, 1, |g, _, _| {
        let tanh_c = c[g].tanh();
        let o_val = gate_o[g];
        let dh_val = dh[g];
        d_pre_o.put(g, dh_val * tanh_c * o_val * (1.0 - o_val));
        dc.put(g, dh_val * o_val * (1.0 - tanh_c * tanh_c));
    });
    (d_pre_o.finish(), dc.finish())
}

/// `lstm_cell_backward_f32`。`(d_pre_ifg, dc_prev)`。
pub(crate) fn lstm_cell_backward(
    gates_ifg: &[f32],
    c_prev: &[f32],
    dc: &[f32],
    hidden: usize,
) -> (Vec<f32>, Vec<f32>) {
    let numel = c_prev.len();
    let (mut d_pre, mut dc_prev) = (Out::new(numel * 3), Out::new(numel));
    grid(numel, hidden, |g, row, j| {
        let base = row * 3 * hidden;
        let i_val = gates_ifg[base + j];
        let f_val = gates_ifg[base + hidden + j];
        let g_val = gates_ifg[base + 2 * hidden + j];
        let dc_val = dc[g];
        let c_prev_val = c_prev[g];
        d_pre.put(base + j, dc_val * g_val * i_val * (1.0 - i_val));
        d_pre.put(
            base + hidden + j,
            dc_val * c_prev_val * f_val * (1.0 - f_val),
        );
        d_pre.put(
            base + 2 * hidden + j,
            dc_val * i_val * (1.0 - g_val * g_val),
        );
        dc_prev.put(g, dc_val * f_val);
    });
    (d_pre.finish(), dc_prev.finish())
}

/// `gru_pointwise_f32`。`(gates, q, h)`。
pub(crate) fn gru_pointwise(
    pre_i: &[f32],
    pre_h: &[f32],
    h_prev: &[f32],
    hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let numel = h_prev.len();
    let (mut gates, mut q_out, mut h_out) = (Out::new(numel * 3), Out::new(numel), Out::new(numel));
    grid(numel, hidden, |g, row, j| {
        let base = row * 3 * hidden;
        let r_pre = pre_i[base + j] + pre_h[base + j];
        let z_pre = pre_i[base + hidden + j] + pre_h[base + hidden + j];
        let q_val = pre_h[base + 2 * hidden + j];
        let pre_i_n = pre_i[base + 2 * hidden + j];
        let r_val = sigmoid(r_pre);
        let z_val = sigmoid(z_pre);
        let n_val = r_val.mul_add(q_val, pre_i_n).tanh();
        gates.put(base + j, r_val);
        gates.put(base + hidden + j, z_val);
        gates.put(base + 2 * hidden + j, n_val);
        q_out.put(g, q_val);
        h_out.put(g, z_val.mul_add(h_prev[g], (1.0 - z_val) * n_val));
    });
    (gates.finish(), q_out.finish(), h_out.finish())
}

/// `gru_backward_f32`。`(d_pre_i, d_pre_h, dh_prev_direct)`。
pub(crate) fn gru_backward(
    gates_rzn: &[f32],
    q: &[f32],
    h_prev: &[f32],
    dh: &[f32],
    hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let numel = h_prev.len();
    let (mut d_pre_i, mut d_pre_h, mut direct) =
        (Out::new(numel * 3), Out::new(numel * 3), Out::new(numel));
    grid(numel, hidden, |g, row, j| {
        let base = row * 3 * hidden;
        let r_val = gates_rzn[base + j];
        let z_val = gates_rzn[base + hidden + j];
        let n_val = gates_rzn[base + 2 * hidden + j];
        let q_val = q[g];
        let h_prev_val = h_prev[g];
        let dh_val = dh[g];
        let dn = dh_val * (1.0 - z_val);
        let dz = dh_val * (h_prev_val - n_val);
        let d_pre_n = dn * (1.0 - n_val * n_val);
        let dr = d_pre_n * q_val;
        let d_pre_r = dr * r_val * (1.0 - r_val);
        let d_pre_z = dz * z_val * (1.0 - z_val);
        d_pre_i.put(base + j, d_pre_r);
        d_pre_i.put(base + hidden + j, d_pre_z);
        d_pre_i.put(base + 2 * hidden + j, d_pre_n);
        d_pre_h.put(base + j, d_pre_r);
        d_pre_h.put(base + hidden + j, d_pre_z);
        d_pre_h.put(base + 2 * hidden + j, d_pre_n * r_val);
        direct.put(g, dh_val * z_val);
    });
    (d_pre_i.finish(), d_pre_h.finish(), direct.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 2001) as f32
                    / 250.0
                    - 4.0
            })
            .collect()
    }

    /// 端数ブロック（255・257 など）を含む numel で全出力が 1 回だけ書かれる（`finish` が assert）。
    #[test]
    fn every_output_is_written_exactly_once_across_partial_blocks() {
        for (b, hidden) in [(1usize, 1usize), (1, 255), (1, 256), (1, 257), (4, 250)] {
            let n = b * hidden;
            let g4 = data(n * 4, 1);
            let g3 = data(n * 3, 2);
            let s = data(n, 3);
            let _ = lstm_pointwise(&g4, &s, hidden);
            let _ = lstm_hidden_backward(&s, &s, &s);
            let _ = lstm_cell_backward(&g3, &s, &s, hidden);
            let _ = gru_pointwise(&g3, &g3, &s, hidden);
            let _ = gru_backward(&g3, &s, &s, &s, hidden);
        }
    }

    #[test]
    fn sigmoid_negative_branch_is_finite_for_large_negative_input() {
        assert!(sigmoid(-100.0).is_finite());
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        assert!(sigmoid(100.0) > 0.99);
    }
}
