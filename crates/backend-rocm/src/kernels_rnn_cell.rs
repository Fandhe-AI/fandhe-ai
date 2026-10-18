//! LSTM／GRU セルの pointwise 段と backward の HIP ソース（`backend-cuda::kernels_rnn_cell` の移植。
//! イシュー #3441・分解案 R10）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_rnn_cell_source`]）。呼び出し元は `rnn_cell.rs` の
//! `RocmRnnCell`。5 本のカーネルを 1 ソース（1 module）にまとめ、`#define BLOCK_DIM` だけを前置する
//! （A03: 静的テンプレート＋検証済み数値のみ。`hidden`・`numel` は引数で渡しソースに入れない）。
//!
//! # 数値契約
//! - 意味論の正は CPU 参照（`backend-cpu::rnn_cell`）。ゲート列順は LSTM が `i,f,g,o`、GRU が `r,z,n`
//!   （`reset_after=True`）。1 スレッド = 1 `(b, j)`、`numel = B * hidden`。
//! - ソース先頭に `#pragma clang fp contract(off)` を置く。HIP/clang の既定は FMA への暗黙縮約を許し、
//!   縮約しない CPU 参照と丸めがずれるため。意図した FMA は CPU の `f32::mul_add` と同じ **3 か所**
//!   （LSTM の `c`、GRU の `n` と `h`）だけで、`__builtin_fmaf` で明示する。backward 3 本に FMA はなく、
//!   積は字面どおりの順に評価する。
//! - シグモイドは CPU と同じ 2 分岐の数値安定形。`__expf` 等の高速近似は使わない。`expf`／`tanhf` は GPU の
//!   libm で、CPU との bit 一致は主張せず REQ-2 複合判定のみで検証する（実機 parity は #3159）。
//! - 縮約を持たない pointwise 演算のため、`f64` アキュムレータ契約（正規化統計・勾配の長軸縮約）は該当しない。
//!
//! # REQ-8
//! 全カーネルが `gid >= numel` を検査してから読み書きする。添字（`row`・`j`・`base`）は `long long`。
//! ブロックは `INDEX_OPS_BLOCK_DIM`（256 = 64 の倍数）で、wave 幅に依存する命令は使わない。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};

const P: ArgKind = ArgKind::Ptr;
const I: ArgKind = ArgKind::I32;

/// `(pre, c_prev, gates, c, h, hidden, numel)`。
const LSTM_POINTWISE_ARGS: [ArgKind; 7] = [P, P, P, P, P, I, I];
/// `(c, gate_o, dh, d_pre_o, dc, numel)`。
const LSTM_HIDDEN_BACKWARD_ARGS: [ArgKind; 6] = [P, P, P, P, P, I];
/// `(gates_ifg, c_prev, dc, d_pre_ifg, dc_prev, hidden, numel)`。
const LSTM_CELL_BACKWARD_ARGS: [ArgKind; 7] = [P, P, P, P, P, I, I];
/// `(pre_i, pre_h, h_prev, gates, q, h, hidden, numel)`。
const GRU_POINTWISE_ARGS: [ArgKind; 8] = [P, P, P, P, P, P, I, I];
/// `(gates_rzn, q, h_prev, dh, d_pre_i, d_pre_h, dh_prev_direct, hidden, numel)`。
const GRU_BACKWARD_ARGS: [ArgKind; 9] = [P, P, P, P, P, P, P, I, I];

pub(crate) const LSTM_POINTWISE_SPEC: KernelSpec = KernelSpec {
    name: "lstm_pointwise_f32",
    args: &LSTM_POINTWISE_ARGS,
};
pub(crate) const LSTM_HIDDEN_BACKWARD_SPEC: KernelSpec = KernelSpec {
    name: "lstm_hidden_backward_f32",
    args: &LSTM_HIDDEN_BACKWARD_ARGS,
};
pub(crate) const LSTM_CELL_BACKWARD_SPEC: KernelSpec = KernelSpec {
    name: "lstm_cell_backward_f32",
    args: &LSTM_CELL_BACKWARD_ARGS,
};
pub(crate) const GRU_POINTWISE_SPEC: KernelSpec = KernelSpec {
    name: "gru_pointwise_f32",
    args: &GRU_POINTWISE_ARGS,
};
pub(crate) const GRU_BACKWARD_SPEC: KernelSpec = KernelSpec {
    name: "gru_backward_f32",
    args: &GRU_BACKWARD_ARGS,
};

/// 静的テンプレート本体。`BLOCK_DIM` は前置する `#define` で与える。
pub(crate) const RNN_CELL_BODY: &str = r#"#pragma clang fp contract(off)

__device__ __forceinline__ float rnn_sigmoid(float x)
{
    return x >= 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void lstm_pointwise_f32(
    const float* __restrict__ pre,
    const float* __restrict__ c_prev,
    float* __restrict__ gates,
    float* __restrict__ c_out,
    float* __restrict__ h_out,
    int hidden,
    int numel)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    long long hid = (long long)hidden;
    long long row = gid / hid;
    long long j = gid % hid;
    long long base = row * 4 * hid;

    float i_val = rnn_sigmoid(pre[base + j]);
    float f_val = rnn_sigmoid(pre[base + hid + j]);
    float g_val = tanhf(pre[base + 2 * hid + j]);
    float o_val = rnn_sigmoid(pre[base + 3 * hid + j]);

    gates[base + j] = i_val;
    gates[base + hid + j] = f_val;
    gates[base + 2 * hid + j] = g_val;
    gates[base + 3 * hid + j] = o_val;

    float c_val = __builtin_fmaf(f_val, c_prev[gid], i_val * g_val);
    c_out[gid] = c_val;
    h_out[gid] = o_val * tanhf(c_val);
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void lstm_hidden_backward_f32(
    const float* __restrict__ c,
    const float* __restrict__ gate_o,
    const float* __restrict__ dh,
    float* __restrict__ d_pre_o,
    float* __restrict__ dc,
    int numel)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    float tanh_c = tanhf(c[gid]);
    float o_val = gate_o[gid];
    float dh_val = dh[gid];
    d_pre_o[gid] = dh_val * tanh_c * o_val * (1.0f - o_val);
    dc[gid] = dh_val * o_val * (1.0f - tanh_c * tanh_c);
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void lstm_cell_backward_f32(
    const float* __restrict__ gates_ifg,
    const float* __restrict__ c_prev,
    const float* __restrict__ dc,
    float* __restrict__ d_pre_ifg,
    float* __restrict__ dc_prev,
    int hidden,
    int numel)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    long long hid = (long long)hidden;
    long long row = gid / hid;
    long long j = gid % hid;
    long long base = row * 3 * hid;

    float i_val = gates_ifg[base + j];
    float f_val = gates_ifg[base + hid + j];
    float g_val = gates_ifg[base + 2 * hid + j];
    float dc_val = dc[gid];
    float c_prev_val = c_prev[gid];

    d_pre_ifg[base + j] = dc_val * g_val * i_val * (1.0f - i_val);
    d_pre_ifg[base + hid + j] = dc_val * c_prev_val * f_val * (1.0f - f_val);
    d_pre_ifg[base + 2 * hid + j] = dc_val * i_val * (1.0f - g_val * g_val);
    dc_prev[gid] = dc_val * f_val;
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void gru_pointwise_f32(
    const float* __restrict__ pre_i,
    const float* __restrict__ pre_h,
    const float* __restrict__ h_prev,
    float* __restrict__ gates,
    float* __restrict__ q_out,
    float* __restrict__ h_out,
    int hidden,
    int numel)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    long long hid = (long long)hidden;
    long long row = gid / hid;
    long long j = gid % hid;
    long long base = row * 3 * hid;

    float r_pre = pre_i[base + j] + pre_h[base + j];
    float z_pre = pre_i[base + hid + j] + pre_h[base + hid + j];
    float q_val = pre_h[base + 2 * hid + j];
    float pre_i_n = pre_i[base + 2 * hid + j];

    float r_val = rnn_sigmoid(r_pre);
    float z_val = rnn_sigmoid(z_pre);
    float n_val = tanhf(__builtin_fmaf(r_val, q_val, pre_i_n));

    gates[base + j] = r_val;
    gates[base + hid + j] = z_val;
    gates[base + 2 * hid + j] = n_val;
    q_out[gid] = q_val;

    h_out[gid] = __builtin_fmaf(z_val, h_prev[gid], (1.0f - z_val) * n_val);
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void gru_backward_f32(
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
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    long long hid = (long long)hidden;
    long long row = gid / hid;
    long long j = gid % hid;
    long long base = row * 3 * hid;

    float r_val = gates_rzn[base + j];
    float z_val = gates_rzn[base + hid + j];
    float n_val = gates_rzn[base + 2 * hid + j];
    float q_val = q[gid];
    float h_prev_val = h_prev[gid];
    float dh_val = dh[gid];

    float dn = dh_val * (1.0f - z_val);
    float dz = dh_val * (h_prev_val - n_val);
    float d_pre_n = dn * (1.0f - n_val * n_val);
    float dr = d_pre_n * q_val;
    float d_pre_r = dr * r_val * (1.0f - r_val);
    float d_pre_z = dz * z_val * (1.0f - z_val);

    d_pre_i[base + j] = d_pre_r;
    d_pre_i[base + hid + j] = d_pre_z;
    d_pre_i[base + 2 * hid + j] = d_pre_n;

    d_pre_h[base + j] = d_pre_r;
    d_pre_h[base + hid + j] = d_pre_z;
    d_pre_h[base + 2 * hid + j] = d_pre_n * r_val;

    dh_prev_direct[gid] = dh_val * z_val;
}
"#;

/// RNN セル 5 カーネル用ソースを生成する（`BLOCK_DIM` のみ前置）。
pub(crate) fn render_rnn_cell_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(&[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], RNN_CELL_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    const ALL: [KernelSpec; 5] = [
        LSTM_POINTWISE_SPEC,
        LSTM_HIDDEN_BACKWARD_SPEC,
        LSTM_CELL_BACKWARD_SPEC,
        GRU_POINTWISE_SPEC,
        GRU_BACKWARD_SPEC,
    ];

    #[test]
    fn spec_matches_source_signature() {
        for spec in ALL {
            assert_eq!(kernel_param_kinds(RNN_CELL_BODY, spec.name), spec.args);
        }
    }

    #[test]
    fn source_defines_only_block_dim_and_starts_with_contract_off() {
        let s = render_rnn_cell_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define BLOCK_DIM 256"]);
        assert!(RNN_CELL_BODY.starts_with("#pragma clang fp contract(off)\n"));
        assert_eq!(s, render_rnn_cell_source().unwrap());
    }

    #[test]
    fn req8_guard_in_every_kernel_and_fma_only_at_three_sites() {
        assert_eq!(
            RNN_CELL_BODY
                .matches("if (gid >= (long long)numel)")
                .count(),
            ALL.len()
        );
        // FMA は CPU の `mul_add` と同じ 3 か所だけ。残りに素の `fma(`／`fmaf(` が無い。
        assert_eq!(RNN_CELL_BODY.matches("__builtin_fmaf(").count(), 3);
        let rest = RNN_CELL_BODY.replace("__builtin_fmaf(", "");
        for banned in [
            "fmaf(",
            "fma(",
            "atomicAdd",
            "__shfl",
            "warpSize",
            "__ballot",
            "__expf",
            "__tanhf",
        ] {
            assert!(!rest.contains(banned), "{banned}");
        }
    }
}
