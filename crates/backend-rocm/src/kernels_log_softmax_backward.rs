//! `log_softmax_backward`（VJP）の HIP カーネルソース（`backend-cuda::kernels_log_softmax_backward`
//! の移植。イシュー #3435・分解案 R4）。
//!
//! 役割: HIPRTC へ渡すソースの生成元（[`render_log_softmax_backward_source`]）。呼び出し元は
//! `softmax.rs`（`RocmLogSoftmaxBackward`）。意味論は `dx = g - exp(y) * Σ_dim(g)`
//! （`autodiff::grad::log_softmax_vjp_along` と同式）。
//!
//! # 構造（1 block = 1 wave = 1 行。grid = rows）
//! 被覆計画 §6 は本演算を「shfl のみ」と分類するが、現行 CUDA 版は PR #1994 の指摘で butterfly を
//! 撤回済みで **shfl を使わない**。lane 0 だけが `i` 昇順に `double` 逐次和を取り（ホスト参照と
//! 完全同順。相殺入力 `g = [1e20, -1e20, 1, 0]` でも桁落ちの位置が一致する）、`__shared__ double
//! row_sum` と `__syncthreads()` で残り lane へ公開する。wave 幅依存は列ループの
//! `i += WARP_SIZE`（CUDA 版はリテラル 32）だけで、ここだけを注入値に置換した。
//!
//! `if (row >= rows) return;` は block 単位で一様（`row = blockIdx.x`）なので、`__syncthreads()` へ
//! 一部の lane だけが到達しない事態は起きない。
//!
//! # 数値・FMA 契約
//! `Σ_dim(g)` は要素を直接 `double` へ昇格する（coding-rust.md「勾配の長軸縮約」）。`exp(y)` は
//! `f32` の `expf` で確定してから `double` へ昇格し、乗算と減算は `double` のまま行って最後に 1 回
//! `float` へ downcast する。先頭の `#pragma clang fp contract(off)` で `g - exp_y * sum` の暗黙 FMA
//! 縮約を止め、ホスト参照（非縮約）と同じ丸めにする。`expf` の丸めは CPU の `f32::exp` と一致を
//! 保証できないため bit 一致は主張せず、REQ-2 統一複合判定で検証する。
//!
//! # REQ-8
//! CTA ガード `row >= rows`、列ループ `i < cols`。添字は `long long`。`INFINITY`／`__FLT_MAX__`／
//! `#include` は使わない。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, KernelSpec};
use crate::warp_geometry::WarpGeometry;

/// `(y, g, dx, rows, cols)`。
const BACKWARD_ARGS: [ArgKind; 5] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
];

/// backward カーネルの spec。
pub(crate) const LOG_SOFTMAX_BACKWARD_SPEC: KernelSpec = KernelSpec {
    name: "log_softmax_backward_f32",
    args: &BACKWARD_ARGS,
};

/// 静的テンプレート本体（wave 幅は前置する `WARP_SIZE` で与える）。
pub(crate) const LOG_SOFTMAX_BACKWARD_BODY: &str = r#"#pragma clang fp contract(off)

extern "C" __global__ __launch_bounds__(WARP_SIZE) void log_softmax_backward_f32(
    const float* __restrict__ y,
    const float* __restrict__ g,
    float* __restrict__ dx,
    int rows,
    int cols)
{
    int lane = threadIdx.x;
    long long row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    long long row_base = row * (long long)cols;
    const float* y_row = y + row_base;
    const float* g_row = g + row_base;
    float* dx_row = dx + row_base;

    __shared__ double row_sum;
    if (lane == 0) {
        double acc = 0.0;
        for (long long i = 0; i < cols; ++i) {
            acc += (double)g_row[i];
        }
        row_sum = acc;
    }
    __syncthreads();
    double sum = row_sum;

    for (long long i = lane; i < cols; i += WARP_SIZE) {
        double exp_y = (double)expf(y_row[i]);
        dx_row[i] = (float)((double)g_row[i] - exp_y * sum);
    }
}
"#;

/// backward 用ソースを生成する（wave 幅の `#define` だけを前置）。
pub(crate) fn render_log_softmax_backward_source(
    geom: WarpGeometry,
) -> Result<KernelSource, RocmError> {
    KernelSource::render_with_warp(
        &geom.render_hip_defines(None)?,
        &[],
        LOG_SOFTMAX_BACKWARD_BODY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    #[test]
    fn spec_matches_source_signature() {
        assert_eq!(
            kernel_param_kinds(LOG_SOFTMAX_BACKWARD_BODY, LOG_SOFTMAX_BACKWARD_SPEC.name),
            LOG_SOFTMAX_BACKWARD_SPEC.args
        );
    }

    #[test]
    fn contract_fp_off_double_accumulator_and_guards() {
        let b = LOG_SOFTMAX_BACKWARD_BODY;
        assert!(b.starts_with("#pragma clang fp contract(off)\n"));
        assert!(b.contains("double acc = 0.0;"));
        assert!(b.contains("acc += (double)g_row[i];"));
        assert!(!b.contains("float acc"));
        assert!(b.contains("if (row >= rows)"));
        assert!(b.contains("for (long long i = lane; i < cols; i += WARP_SIZE)"));
        assert!(b.contains("__syncthreads();"));
        assert!(b.contains("long long row = blockIdx.x;"));
        // 最終書き出しの 1 回だけ float へ downcast する。
        assert_eq!(b.matches("(float)").count(), 1);
    }

    #[test]
    fn no_shuffle_32bit_masks_or_literal_stride() {
        let b = LOG_SOFTMAX_BACKWARD_BODY;
        for banned in [
            "__shfl",
            "WARP_SHFL_XOR",
            "__shfl_xor_sync",
            "0xffffffff",
            "__syncwarp",
            "WARP_SYNC",
            "INFINITY",
            "__FLT_MAX__",
            "#include",
            "i += 32",
        ] {
            assert!(!b.contains(banned), "{banned}");
        }
    }

    #[test]
    fn render_is_deterministic_for_both_wave_widths() {
        for w in [32u32, 64] {
            let g = WarpGeometry::new(w).unwrap();
            let a = render_log_softmax_backward_source(g).unwrap();
            assert_eq!(a, render_log_softmax_backward_source(g).unwrap());
            assert!(a.as_str().starts_with(&format!("#define WARP_SIZE {w}\n")));
            assert!(a.as_str().ends_with(LOG_SOFTMAX_BACKWARD_BODY));
        }
    }
}
