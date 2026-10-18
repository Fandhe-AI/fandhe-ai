//! elementwise 基本カーネル（`add`・`mul`・`relu`・`exp`・`tanh`・`where_cond`・`masked_fill`）の
//! HIP ソース（`backend-cuda::kernels_elementwise` の移植。イシュー #3433・分解案 R2）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_elementwise_source`]）。呼び出し元は
//! `elementwise.rs` の `RocmElementwise`。7 本のカーネルを 1 ソース（1 module）にまとめ、
//! `#define BLOCK_DIM` だけを前置する（A03: 静的テンプレート＋検証済み数値のみ）。
//!
//! # 数値契約
//! - ソース先頭に `#pragma clang fp contract(off)` を置く。HIP/clang の既定は FMA への暗黙縮約を
//!   許すため、置かないと「`a * b` の後の `+ c`」が静かに FMA 化し、縮約しない CPU 参照と丸めが
//!   ずれる。本系統に意図した FMA はなく、字面どおりの演算が意味論になる（matmul 系の FMA 契約
//!   〈`__builtin_fmaf` 明示〉は GEMM 側で別軸）。縮約すべき式を足すときは `__builtin_fmaf` で明示する。
//! - `add`・`mul` は IEEE 754 単精度の 1 演算で CPU と bit 一致。`relu` は CUDA 版と同じ
//!   `x > 0 ? x : 0`（NaN は 0。CPU の `f32::max(x, 0.0)` と `-0.0` の符号だけが異なりうる）。
//!   `expf`／`tanhf` は GPU の libm で、CPU との bit 一致は主張せず REQ-2 複合判定のみで検証する。
//! - elementwise に縮約はなく、`f64` アキュムレータ契約（正規化統計・勾配の長軸縮約）は該当しない。
//!
//! # REQ-8
//! 全カーネルが `gid >= numel` を検査してから読み書きする。添字は `long long`。ブロックは
//! `INDEX_OPS_BLOCK_DIM`（256 = 64 の倍数）で、wave 幅に依存する命令は使わない。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};

const BINARY_ARGS: [ArgKind; 4] = [ArgKind::Ptr, ArgKind::Ptr, ArgKind::Ptr, ArgKind::I32];
const UNARY_ARGS: [ArgKind; 3] = [ArgKind::Ptr, ArgKind::Ptr, ArgKind::I32];
const WHERE_ARGS: [ArgKind; 5] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
];
/// `(x, mask, value, out, numel)`。
const MASKED_FILL_ARGS: [ArgKind; 5] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::F32,
    ArgKind::Ptr,
    ArgKind::I32,
];

pub(crate) const ADD_SPEC: KernelSpec = KernelSpec {
    name: "ew_add_f32",
    args: &BINARY_ARGS,
};
pub(crate) const MUL_SPEC: KernelSpec = KernelSpec {
    name: "ew_mul_f32",
    args: &BINARY_ARGS,
};
pub(crate) const RELU_SPEC: KernelSpec = KernelSpec {
    name: "ew_relu_f32",
    args: &UNARY_ARGS,
};
pub(crate) const EXP_SPEC: KernelSpec = KernelSpec {
    name: "ew_exp_f32",
    args: &UNARY_ARGS,
};
pub(crate) const TANH_SPEC: KernelSpec = KernelSpec {
    name: "ew_tanh_f32",
    args: &UNARY_ARGS,
};
pub(crate) const WHERE_SPEC: KernelSpec = KernelSpec {
    name: "ew_where_f32",
    args: &WHERE_ARGS,
};
pub(crate) const MASKED_FILL_SPEC: KernelSpec = KernelSpec {
    name: "ew_masked_fill_f32",
    args: &MASKED_FILL_ARGS,
};

/// 静的テンプレート本体。`BLOCK_DIM` は前置する `#define` で与える。
pub(crate) const ELEMENTWISE_BODY: &str = r#"#pragma clang fp contract(off)

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void ew_add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int numel)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    out[gid] = a[gid] + b[gid];
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void ew_mul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int numel)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    out[gid] = a[gid] * b[gid];
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void ew_relu_f32(
    const float* __restrict__ a,
    float* __restrict__ out,
    int numel)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    float x = a[gid];
    out[gid] = x > 0.0f ? x : 0.0f;
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void ew_exp_f32(
    const float* __restrict__ a,
    float* __restrict__ out,
    int numel)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    out[gid] = expf(a[gid]);
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void ew_tanh_f32(
    const float* __restrict__ a,
    float* __restrict__ out,
    int numel)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    out[gid] = tanhf(a[gid]);
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void ew_where_f32(
    const float* __restrict__ cond,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int numel)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    out[gid] = (cond[gid] != 0.0f) ? a[gid] : b[gid];
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void ew_masked_fill_f32(
    const float* __restrict__ x,
    const float* __restrict__ mask,
    float value,
    float* __restrict__ out,
    int numel)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    out[gid] = (mask[gid] != 0.0f) ? value : x[gid];
}
"#;

/// elementwise 基本カーネル用ソースを生成する（`BLOCK_DIM` のみ前置）。
pub(crate) fn render_elementwise_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(&[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], ELEMENTWISE_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    const ALL: [KernelSpec; 7] = [
        ADD_SPEC,
        MUL_SPEC,
        RELU_SPEC,
        EXP_SPEC,
        TANH_SPEC,
        WHERE_SPEC,
        MASKED_FILL_SPEC,
    ];

    #[test]
    fn spec_matches_source_signature() {
        for spec in ALL {
            assert_eq!(kernel_param_kinds(ELEMENTWISE_BODY, spec.name), spec.args);
        }
    }

    #[test]
    fn source_defines_only_block_dim_and_starts_with_contract_off() {
        let s = render_elementwise_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define BLOCK_DIM 256"]);
        assert!(ELEMENTWISE_BODY.starts_with("#pragma clang fp contract(off)\n"));
        assert_eq!(s, render_elementwise_source().unwrap());
    }

    #[test]
    fn req8_guard_in_every_kernel_and_no_fma_or_warp_constructs() {
        assert_eq!(
            ELEMENTWISE_BODY
                .matches("if (gid >= (long long)numel)")
                .count(),
            ALL.len()
        );
        for banned in [
            "fma(",
            "__builtin_fma",
            "__shfl",
            "warpSize",
            "__ballot",
            "__expf",
            "__tanhf",
        ] {
            assert!(!ELEMENTWISE_BODY.contains(banned), "{banned}");
        }
        assert!(ELEMENTWISE_BODY.contains("x > 0.0f ? x : 0.0f"));
    }
}
