//! RNN／LSTM／GRU セル演算（イシュー #1647）の CUDA C カーネルソース
//! （NVRTC 実行時コンパイル用の静的文字列）。
//!
//! `rnn_cell.rs`（呼び出し元）は本モジュールの定数を `nvrtc::compile_ptx`
//! に渡し `CudaFunction` を得る（`kernels_elementwise.rs` と同じ理由で
//! ソースを `nvcc` 事前コンパイルせず文字列のまま埋め込む。「CUDA
//! toolkit 非搭載環境でも `cargo build --workspace` が成立する」契約を
//! 維持する。`.claude/rules/deps-policy.md`）。
//!
//! # 意味論の正
//!
//! `backend-cpu::rnn_cell`（`crates/backend-cpu/src/rnn_cell.rs`）が
//! 意味論の正。`diff*diff` 系の積を和へ蓄積する箇所は単精度 `fmaf`
//! （CPU 側 `f32::mul_add` と同じ FMA 契約統一方針。`.claude/rules/
//! coding-rust.md`）を用いる。シグモイドは CPU 側 `sigmoid_scalar`
//! と同じ 2 分岐の数値安定形（`x >= 0 ? 1/(1+exp(-x)) : e/(1+e)`）を
//! 使う。GPU 側の丸めが CPU の libm と厳密一致する保証はないため、
//! backend 間の数値突合は統一複合判定（相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満。`.claude/rules/coding-rust.md`）で検証する。
//!
//! # ゲート配置（決定 5。`docs/autodiff-rnn-cell-tape-design.md`）
//!
//! LSTM は列ブロック順 `i,f,g,o`、GRU は `r,z,n`。`hidden` パラメータ
//! （カーネル引数）で各ブロックの列オフセットを求める。
//!
//! # スレッド割当・REQ-8（カーネル境界検査規約）
//!
//! いずれも 1 スレッド = 1 `(b, j)`（`j` は `0..hidden`）で
//! `numel = B * hidden`。全カーネルが `if (idx < numel)` の手動境界
//! チェックを維持する（`.claude/rules/coding-rust.md` の REQ-8 規約）。
//! `row = idx / hidden`・`col = idx % hidden` でバッチ・列番号を復元する。

/// 1 スレッドブロックあたりのスレッド数（1 次元）。
/// `kernels_elementwise::EW_BLOCK_DIM` と同じ値・同じ理由。
pub const RNN_BLOCK_DIM: u32 = 256;

/// LSTM セルの pointwise 段（決定 1・1b）。
/// `pre: [B, 4H]`（列ブロック順 `i,f,g,o`）・`c_prev: [B, H]` を読み、
/// `gates: [B, 4H]`（活性化後）・`c: [B, H]`・`h: [B, H]` を書く。
pub const LSTM_POINTWISE_F32: &str = r#"
extern "C" __global__ void lstm_pointwise_f32(
    const float* __restrict__ pre,
    const float* __restrict__ c_prev,
    float* __restrict__ gates,
    float* __restrict__ c_out,
    float* __restrict__ h_out,
    int hidden,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int row = idx / hidden;
        int j = idx % hidden;
        int base = row * 4 * hidden;

        float i_pre = pre[base + j];
        float f_pre = pre[base + hidden + j];
        float g_pre = pre[base + 2 * hidden + j];
        float o_pre = pre[base + 3 * hidden + j];

        float i_val = i_pre >= 0.0f ? 1.0f / (1.0f + expf(-i_pre)) : expf(i_pre) / (1.0f + expf(i_pre));
        float f_val = f_pre >= 0.0f ? 1.0f / (1.0f + expf(-f_pre)) : expf(f_pre) / (1.0f + expf(f_pre));
        float g_val = tanhf(g_pre);
        float o_val = o_pre >= 0.0f ? 1.0f / (1.0f + expf(-o_pre)) : expf(o_pre) / (1.0f + expf(o_pre));

        gates[base + j] = i_val;
        gates[base + hidden + j] = f_val;
        gates[base + 2 * hidden + j] = g_val;
        gates[base + 3 * hidden + j] = o_val;

        float c_prev_val = c_prev[idx];
        float c_val = fmaf(f_val, c_prev_val, i_val * g_val);
        c_out[idx] = c_val;
        h_out[idx] = o_val * tanhf(c_val);
    }
}
"#;

/// [`Op::LstmHidden`] の VJP 補助。`c`・`gate_o`・`dh` はいずれも
/// `[B, H]`。`d_pre_o = dh*tanh(c)*o*(1-o)`、`dc = dh*o*(1-tanh(c)^2)`。
pub const LSTM_HIDDEN_BACKWARD_F32: &str = r#"
extern "C" __global__ void lstm_hidden_backward_f32(
    const float* __restrict__ c,
    const float* __restrict__ gate_o,
    const float* __restrict__ dh,
    float* __restrict__ d_pre_o,
    float* __restrict__ dc,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float tanh_c = tanhf(c[idx]);
        float o_val = gate_o[idx];
        float dh_val = dh[idx];
        d_pre_o[idx] = dh_val * tanh_c * o_val * (1.0f - o_val);
        dc[idx] = dh_val * o_val * (1.0f - tanh_c * tanh_c);
    }
}
"#;

/// [`Op::LstmCell`] の VJP 補助。`gates_ifg: [B, 3H]`（活性化後
/// `i,f,g`）・`c_prev: [B, H]`・`dc: [B, H]` から `d_pre_ifg: [B, 3H]`・
/// `dc_prev: [B, H]` を計算する。
pub const LSTM_CELL_BACKWARD_F32: &str = r#"
extern "C" __global__ void lstm_cell_backward_f32(
    const float* __restrict__ gates_ifg,
    const float* __restrict__ c_prev,
    const float* __restrict__ dc,
    float* __restrict__ d_pre_ifg,
    float* __restrict__ dc_prev,
    int hidden,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int row = idx / hidden;
        int j = idx % hidden;
        int base = row * 3 * hidden;

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
"#;

/// GRU セルの pointwise 段（決定 1c・5。`reset_after=True` 規約）。
/// `pre_i`／`pre_h: [B, 3H]`（列ブロック順 `r,z,n`）・`h_prev: [B, H]`
/// から `gates: [B, 3H]`（活性化後）・`q: [B, H]`（`pre_h` の n 列
/// ブロック）・`h: [B, H]` を計算する。
pub const GRU_POINTWISE_F32: &str = r#"
extern "C" __global__ void gru_pointwise_f32(
    const float* __restrict__ pre_i,
    const float* __restrict__ pre_h,
    const float* __restrict__ h_prev,
    float* __restrict__ gates,
    float* __restrict__ q_out,
    float* __restrict__ h_out,
    int hidden,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int row = idx / hidden;
        int j = idx % hidden;
        int base = row * 3 * hidden;

        float r_pre = pre_i[base + j] + pre_h[base + j];
        float z_pre = pre_i[base + hidden + j] + pre_h[base + hidden + j];
        float q_val = pre_h[base + 2 * hidden + j];
        float pre_i_n = pre_i[base + 2 * hidden + j];

        float r_val = r_pre >= 0.0f ? 1.0f / (1.0f + expf(-r_pre)) : expf(r_pre) / (1.0f + expf(r_pre));
        float z_val = z_pre >= 0.0f ? 1.0f / (1.0f + expf(-z_pre)) : expf(z_pre) / (1.0f + expf(z_pre));
        float n_val = tanhf(fmaf(r_val, q_val, pre_i_n));

        gates[base + j] = r_val;
        gates[base + hidden + j] = z_val;
        gates[base + 2 * hidden + j] = n_val;
        q_out[idx] = q_val;

        float h_prev_val = h_prev[idx];
        h_out[idx] = fmaf(z_val, h_prev_val, (1.0f - z_val) * n_val);
    }
}
"#;

/// [`Op::GruCell`] の VJP 補助。`gates_rzn: [B, 3H]`（活性化後
/// `r,z,n`）・`q: [B, H]`（決定 1c）・`h_prev: [B, H]`・`dh: [B, H]` から
/// `d_pre_i: [B, 3H]`・`d_pre_h: [B, 3H]`・`dh_prev_direct: [B, H]` を
/// 計算する。
pub const GRU_BACKWARD_F32: &str = r#"
extern "C" __global__ void gru_backward_f32(
    const float* __restrict__ gates_rzn,
    const float* __restrict__ q,
    const float* __restrict__ h_prev,
    const float* __restrict__ dh,
    float* __restrict__ d_pre_i,
    float* __restrict__ d_pre_h,
    float* __restrict__ dh_prev_direct,
    int hidden,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int row = idx / hidden;
        int j = idx % hidden;
        int base = row * 3 * hidden;

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
"#;

#[cfg(test)]
mod tests {
    //! REQ-8（境界検査規約）・決定性（`atomicAdd` 不使用）のソース検査。
    //! GPU 非依存（文字列検査のみ）。
    use super::*;

    #[test]
    fn all_kernels_have_manual_bounds_check() {
        for src in [
            LSTM_POINTWISE_F32,
            LSTM_HIDDEN_BACKWARD_F32,
            LSTM_CELL_BACKWARD_F32,
            GRU_POINTWISE_F32,
            GRU_BACKWARD_F32,
        ] {
            assert!(
                src.contains("if (idx < numel)"),
                "kernel source must contain a manual REQ-8 bounds check: {src}"
            );
        }
    }

    #[test]
    fn no_kernel_uses_atomic_add() {
        // pointwise カーネルはブロック間の縮約を持たないため、決定性の
        // ためにも `atomicAdd` を必要としない（`kernels_mse.rs` の
        // reduction カーネルとは異なる設計）。
        for src in [
            LSTM_POINTWISE_F32,
            LSTM_HIDDEN_BACKWARD_F32,
            LSTM_CELL_BACKWARD_F32,
            GRU_POINTWISE_F32,
            GRU_BACKWARD_F32,
        ] {
            assert!(!src.contains("atomicAdd"), "unexpected atomicAdd: {src}");
        }
    }
}
