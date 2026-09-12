// RNN／LSTM／GRU セル演算（イシュー #1647）の MSL カーネルソース
// （CUDA 側 `backend-cuda::kernels_rnn_cell`〈同イシュー〉の Metal
// 対応版）。
//
// `crate::rnn_cell` が `include_str!` で本ファイルを取り込み、
// `MTLCompileOptions`（Safe/Precise。`crate::pipeline::compile_options`。
// `shaders/mse.metal`・`shaders/elementwise.metal` と同一設定）で
// 実行時コンパイルする。1 スレッド = 1 `(b, j)`（`j` は `0..hidden`）の
// 1 次元グリッドで、全カーネルが `if (idx < numel)` の手動境界チェック
// を維持する（REQ-8。`.claude/rules/coding-rust.md`）。
//
// # 意味論の正
//
// `backend-cpu::rnn_cell`（`crates/backend-cpu/src/rnn_cell.rs`）が
// 意味論の正。`exp`／`tanh` はリポジトリの precise math 方針
// （`shaders/elementwise.metal` 冒頭コメント参照）に従い
// `metal::precise::exp`／`metal::precise::tanh` を明示使用する。GPU 側の
// 丸めが CPU の libm と厳密一致する保証はないため、backend 間の数値
// 突合は統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
// `.claude/rules/coding-rust.md`）で検証する。
//
// # ゲート配置（決定 5。`docs/autodiff-rnn-cell-tape-design.md`）
//
// LSTM は列ブロック順 `i,f,g,o`、GRU は `r,z,n`。`hidden` パラメータで
// 各ブロックの列オフセットを求める。`row = idx / hidden`・
// `col = idx % hidden` でバッチ・列番号を復元する（CUDA 側
// `kernels_rnn_cell.rs` と同じレイアウト）。

#include <metal_stdlib>
using namespace metal;

// 数値安定形のシグモイド（`backend-cpu::rnn_cell::sigmoid_scalar` と
// 同じ 2 分岐形）。
inline float rnn_sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + metal::precise::exp(-x));
    }
    float e = metal::precise::exp(x);
    return e / (1.0f + e);
}

// LSTM セルの pointwise 段（決定 1・1b）。`pre: [B, 4H]`（列ブロック順
// `i,f,g,o`）・`c_prev: [B, H]` を読み、`gates: [B, 4H]`（活性化後）・
// `c: [B, H]`・`h: [B, H]` を書く。
kernel void lstm_pointwise_f32(
    device const float* pre [[buffer(0)]],
    device const float* c_prev [[buffer(1)]],
    device float* gates [[buffer(2)]],
    device float* c_out [[buffer(3)]],
    device float* h_out [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& numel [[buffer(6)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        uint row = idx / hidden;
        uint j = idx % hidden;
        uint base = row * 4 * hidden;

        float i_val = rnn_sigmoid(pre[base + j]);
        float f_val = rnn_sigmoid(pre[base + hidden + j]);
        float g_val = metal::precise::tanh(pre[base + 2 * hidden + j]);
        float o_val = rnn_sigmoid(pre[base + 3 * hidden + j]);

        gates[base + j] = i_val;
        gates[base + hidden + j] = f_val;
        gates[base + 2 * hidden + j] = g_val;
        gates[base + 3 * hidden + j] = o_val;

        float c_val = fma(f_val, c_prev[idx], i_val * g_val);
        c_out[idx] = c_val;
        h_out[idx] = o_val * metal::precise::tanh(c_val);
    }
}

// `Op::LstmHidden` の VJP 補助。`c`・`gate_o`・`dh` はいずれも `[B, H]`。
kernel void lstm_hidden_backward_f32(
    device const float* c [[buffer(0)]],
    device const float* gate_o [[buffer(1)]],
    device const float* dh [[buffer(2)]],
    device float* d_pre_o [[buffer(3)]],
    device float* dc [[buffer(4)]],
    constant uint& numel [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        float tanh_c = metal::precise::tanh(c[idx]);
        float o_val = gate_o[idx];
        float dh_val = dh[idx];
        d_pre_o[idx] = dh_val * tanh_c * o_val * (1.0f - o_val);
        dc[idx] = dh_val * o_val * (1.0f - tanh_c * tanh_c);
    }
}

// `Op::LstmCell` の VJP 補助。`gates_ifg: [B, 3H]`（活性化後 `i,f,g`）・
// `c_prev: [B, H]`・`dc: [B, H]` から `d_pre_ifg: [B, 3H]`・
// `dc_prev: [B, H]` を計算する。
kernel void lstm_cell_backward_f32(
    device const float* gates_ifg [[buffer(0)]],
    device const float* c_prev [[buffer(1)]],
    device const float* dc [[buffer(2)]],
    device float* d_pre_ifg [[buffer(3)]],
    device float* dc_prev [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& numel [[buffer(6)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        uint row = idx / hidden;
        uint j = idx % hidden;
        uint base = row * 3 * hidden;

        float i_val = gates_ifg[base + j];
        float f_val = gates_ifg[base + hidden + j];
        float g_val = gates_ifg[base + 2 * hidden + j];
        float dc_val = dc[idx];
        float c_prev_val = c_prev[idx];

        d_pre_ifg[base + j] = dc_val * g_val * i_val * (1.0f - i_val);
        d_pre_ifg[base + hidden + j] = dc_val * c_prev_val * f_val * (1.0f - f_val);
        d_pre_ifg[base + 2 * hidden + j] = dc_val * i_val * (1.0f - g_val * g_val);
        dc_prev[idx] = dc_val * f_val;
    }
}

// GRU セルの pointwise 段（決定 1c・5。`reset_after=True` 規約）。
// `pre_i`／`pre_h: [B, 3H]`（列ブロック順 `r,z,n`）・`h_prev: [B, H]`
// から `gates: [B, 3H]`（活性化後）・`q: [B, H]`（`pre_h` の n 列
// ブロック）・`h: [B, H]` を計算する。
kernel void gru_pointwise_f32(
    device const float* pre_i [[buffer(0)]],
    device const float* pre_h [[buffer(1)]],
    device const float* h_prev [[buffer(2)]],
    device float* gates [[buffer(3)]],
    device float* q_out [[buffer(4)]],
    device float* h_out [[buffer(5)]],
    constant uint& hidden [[buffer(6)]],
    constant uint& numel [[buffer(7)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        uint row = idx / hidden;
        uint j = idx % hidden;
        uint base = row * 3 * hidden;

        float r_pre = pre_i[base + j] + pre_h[base + j];
        float z_pre = pre_i[base + hidden + j] + pre_h[base + hidden + j];
        float q_val = pre_h[base + 2 * hidden + j];
        float pre_i_n = pre_i[base + 2 * hidden + j];

        float r_val = rnn_sigmoid(r_pre);
        float z_val = rnn_sigmoid(z_pre);
        float n_val = metal::precise::tanh(fma(r_val, q_val, pre_i_n));

        gates[base + j] = r_val;
        gates[base + hidden + j] = z_val;
        gates[base + 2 * hidden + j] = n_val;
        q_out[idx] = q_val;

        h_out[idx] = fma(z_val, h_prev[idx], (1.0f - z_val) * n_val);
    }
}

// `Op::GruCell` の VJP 補助。`gates_rzn: [B, 3H]`（活性化後 `r,z,n`）・
// `q`／`h_prev`／`dh: [B, H]` から `d_pre_i: [B, 3H]`・
// `d_pre_h: [B, 3H]`・`dh_prev_direct: [B, H]` を計算する。
kernel void gru_backward_f32(
    device const float* gates_rzn [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* h_prev [[buffer(2)]],
    device const float* dh [[buffer(3)]],
    device float* d_pre_i [[buffer(4)]],
    device float* d_pre_h [[buffer(5)]],
    device float* dh_prev_direct [[buffer(6)]],
    constant uint& hidden [[buffer(7)]],
    constant uint& numel [[buffer(8)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        uint row = idx / hidden;
        uint j = idx % hidden;
        uint base = row * 3 * hidden;

        float r_val = gates_rzn[base + j];
        float z_val = gates_rzn[base + hidden + j];
        float n_val = gates_rzn[base + 2 * hidden + j];
        float q_val = q[idx];
        float h_prev_val = h_prev[idx];
        float dh_val = dh[idx];

        float dn = dh_val * (1.0f - z_val);
        float dz = dh_val * (h_prev_val - n_val);
        float d_pre_n = dn * (1.0f - n_val * n_val);
        float dr = d_pre_n * q_val;
        float d_pre_r = dr * r_val * (1.0f - r_val);
        float d_pre_z = dz * z_val * (1.0f - z_val);

        d_pre_i[base + j] = d_pre_r;
        d_pre_i[base + hidden + j] = d_pre_z;
        d_pre_i[base + 2 * hidden + j] = d_pre_n;

        d_pre_h[base + j] = d_pre_r;
        d_pre_h[base + hidden + j] = d_pre_z;
        d_pre_h[base + 2 * hidden + j] = d_pre_n * r_val;

        dh_prev_direct[idx] = dh_val * z_val;
    }
}
