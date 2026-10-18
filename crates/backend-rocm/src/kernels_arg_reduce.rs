//! `argmax`／`argmin`（全軸・単一軸）の HIP カーネルソース（`backend-cuda::kernels_arg_reduce` の
//! 逐語移植。イシュー #3434・分解案 R3）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_arg_reduce_source`]）。呼び出し元は
//! `arg_reduce.rs`。6 本のカーネル（{argmax, argmin} × {axis, all_partial, all_finalize}）を 1 ソース
//! （1 module）にまとめ、`#define BLOCK_DIM` だけを前置する。
//!
//! # wave 幅非依存
//! 本カーネル群は shfl・warp 内通信・shared memory を使わず、スレッドごとに独立して逐次走査する。
//! wave 幅マクロ（`WarpGeometry`）は注入せず、CUDA 版を逐語移植してよい（wave32／wave64 の両方で
//! そのまま正しい）。
//!
//! # 走査契約（`backend-cpu::reduction::{arg_slice, axis_arg_reduce}` と同一）
//! 厳密比較のみ（`v > best` / `v < best`）で更新するためタイは最初の添字が勝つ。NaN は無視し
//! （NaN 判定は `__float_as_uint` の bit 検査。`isnan` に依存しない）、全要素 NaN なら添字 0。
//!
//! # 全軸縮約の 2 段構成
//! 1 段目: スレッド `t` が連続チャンク `[t*chunk_len, min((t+1)*chunk_len, numel))` を昇順走査し
//! `pval[t]`／`pidx[t]`（非 NaN 要素が無ければ `pidx[t] = -1`）を書く。2 段目: 単一スレッドが
//! `0..num_chunks` を昇順・厳密比較のみで結合する（チャンクが連続・昇順のためタイは必ず最小添字が
//! 勝つ。結合律により単一走査と同じ結果）。チャンク計画 `arg_all_plan` はホスト側
//! （`arg_reduce.rs`）で決める。
//!
//! 丸めを伴わない添字選択のため `f64` アキュムレータ契約・FMA の対象外（`double` を使わない）。
//!
//! # REQ-8
//! 手動境界チェック（`idx < total`／`t < num_chunks`／`blockIdx.x == 0 && threadIdx.x == 0`）を
//! 全カーネルで維持する。添字計算は `long long`。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};

/// `(in, out, outer, axis_len, inner)`。
const AXIS_ARGS: [ArgKind; 5] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
];
/// `(in, pval, pidx, numel, chunk_len, num_chunks)`。
const ALL_PARTIAL_ARGS: [ArgKind; 6] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
];
/// `(pval, pidx, out, num_chunks)`。
const ALL_FINALIZE_ARGS: [ArgKind; 4] = [ArgKind::Ptr, ArgKind::Ptr, ArgKind::Ptr, ArgKind::I32];

/// argmax 単一軸。
pub(crate) const ARGMAX_AXIS_SPEC: KernelSpec = KernelSpec {
    name: "reduce_argmax_axis_f32",
    args: &AXIS_ARGS,
};
/// argmin 単一軸。
pub(crate) const ARGMIN_AXIS_SPEC: KernelSpec = KernelSpec {
    name: "reduce_argmin_axis_f32",
    args: &AXIS_ARGS,
};
/// argmax 全軸 1 段目。
pub(crate) const ARGMAX_ALL_PARTIAL_SPEC: KernelSpec = KernelSpec {
    name: "reduce_argmax_all_partial_f32",
    args: &ALL_PARTIAL_ARGS,
};
/// argmin 全軸 1 段目。
pub(crate) const ARGMIN_ALL_PARTIAL_SPEC: KernelSpec = KernelSpec {
    name: "reduce_argmin_all_partial_f32",
    args: &ALL_PARTIAL_ARGS,
};
/// argmax 全軸 2 段目。
pub(crate) const ARGMAX_ALL_FINALIZE_SPEC: KernelSpec = KernelSpec {
    name: "reduce_argmax_all_finalize_f32",
    args: &ALL_FINALIZE_ARGS,
};
/// argmin 全軸 2 段目。
pub(crate) const ARGMIN_ALL_FINALIZE_SPEC: KernelSpec = KernelSpec {
    name: "reduce_argmin_all_finalize_f32",
    args: &ALL_FINALIZE_ARGS,
};

/// 6 本の spec（テスト用の列挙）。
#[cfg(test)]
pub(crate) const ALL_SPECS: [KernelSpec; 6] = [
    ARGMAX_AXIS_SPEC,
    ARGMIN_AXIS_SPEC,
    ARGMAX_ALL_PARTIAL_SPEC,
    ARGMIN_ALL_PARTIAL_SPEC,
    ARGMAX_ALL_FINALIZE_SPEC,
    ARGMIN_ALL_FINALIZE_SPEC,
];

/// 静的テンプレート本体。`BLOCK_DIM` は前置する `#define` で与える。
pub(crate) const ARG_REDUCE_BODY: &str = r#"
#pragma clang fp contract(off)

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_argmax_axis_f32(
    const float* __restrict__ in,
    int* __restrict__ out,
    int outer,
    int axis_len,
    int inner)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)outer * (long long)inner;
    if (idx < total) {
        long long o = idx / inner;
        long long i = idx % inner;
        int has_best = 0;
        float best_val = 0.0f;
        int best_idx = 0;
        for (long long a = 0; a < axis_len; a++) {
            long long src = (o * (long long)axis_len + a) * (long long)inner + i;
            float v = in[src];
            if ((__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u) {
                continue;
            }
            if (!has_best || v > best_val) {
                has_best = 1;
                best_val = v;
                best_idx = (int)a;
            }
        }
        out[idx] = best_idx;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_argmin_axis_f32(
    const float* __restrict__ in,
    int* __restrict__ out,
    int outer,
    int axis_len,
    int inner)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)outer * (long long)inner;
    if (idx < total) {
        long long o = idx / inner;
        long long i = idx % inner;
        int has_best = 0;
        float best_val = 0.0f;
        int best_idx = 0;
        for (long long a = 0; a < axis_len; a++) {
            long long src = (o * (long long)axis_len + a) * (long long)inner + i;
            float v = in[src];
            if ((__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u) {
                continue;
            }
            if (!has_best || v < best_val) {
                has_best = 1;
                best_val = v;
                best_idx = (int)a;
            }
        }
        out[idx] = best_idx;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_argmax_all_partial_f32(
    const float* __restrict__ in,
    float* __restrict__ pval,
    int* __restrict__ pidx,
    int numel,
    int chunk_len,
    int num_chunks)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < num_chunks) {
        long long start = (long long)t * (long long)chunk_len;
        long long end = start + (long long)chunk_len;
        if (end > numel) {
            end = numel;
        }
        int has_best = 0;
        float best_val = 0.0f;
        int best_idx = -1;
        for (long long i = start; i < end; i++) {
            float v = in[i];
            if ((__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u) {
                continue;
            }
            if (!has_best || v > best_val) {
                has_best = 1;
                best_val = v;
                best_idx = (int)i;
            }
        }
        pval[t] = best_val;
        pidx[t] = best_idx;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_argmin_all_partial_f32(
    const float* __restrict__ in,
    float* __restrict__ pval,
    int* __restrict__ pidx,
    int numel,
    int chunk_len,
    int num_chunks)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < num_chunks) {
        long long start = (long long)t * (long long)chunk_len;
        long long end = start + (long long)chunk_len;
        if (end > numel) {
            end = numel;
        }
        int has_best = 0;
        float best_val = 0.0f;
        int best_idx = -1;
        for (long long i = start; i < end; i++) {
            float v = in[i];
            if ((__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u) {
                continue;
            }
            if (!has_best || v < best_val) {
                has_best = 1;
                best_val = v;
                best_idx = (int)i;
            }
        }
        pval[t] = best_val;
        pidx[t] = best_idx;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_argmax_all_finalize_f32(
    const float* __restrict__ pval,
    const int* __restrict__ pidx,
    int* __restrict__ out,
    int num_chunks)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int has_best = 0;
        float best_val = 0.0f;
        int best_idx = 0;
        for (int c = 0; c < num_chunks; c++) {
            int idx = pidx[c];
            if (idx < 0) {
                continue;
            }
            float v = pval[c];
            if (!has_best || v > best_val) {
                has_best = 1;
                best_val = v;
                best_idx = idx;
            }
        }
        out[0] = best_idx;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_argmin_all_finalize_f32(
    const float* __restrict__ pval,
    const int* __restrict__ pidx,
    int* __restrict__ out,
    int num_chunks)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int has_best = 0;
        float best_val = 0.0f;
        int best_idx = 0;
        for (int c = 0; c < num_chunks; c++) {
            int idx = pidx[c];
            if (idx < 0) {
                continue;
            }
            float v = pval[c];
            if (!has_best || v < best_val) {
                has_best = 1;
                best_val = v;
                best_idx = idx;
            }
        }
        out[0] = best_idx;
    }
}
"#;

/// argmax／argmin 用ソースを生成する（`BLOCK_DIM` のみ前置。wave 幅マクロは入れない）。
pub(crate) fn render_arg_reduce_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(&[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], ARG_REDUCE_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    #[test]
    fn spec_matches_source_signature() {
        for spec in ALL_SPECS {
            assert_eq!(
                kernel_param_kinds(ARG_REDUCE_BODY, spec.name),
                spec.args,
                "{}",
                spec.name
            );
        }
    }

    #[test]
    fn source_defines_only_block_dim_and_is_deterministic() {
        let s = render_arg_reduce_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define BLOCK_DIM 256"]);
        assert_eq!(s, render_arg_reduce_source().unwrap());
    }

    #[test]
    fn req8_guards_are_kept() {
        assert_eq!(ARG_REDUCE_BODY.matches("if (idx < total)").count(), 2);
        assert_eq!(ARG_REDUCE_BODY.matches("if (t < num_chunks)").count(), 2);
        assert_eq!(
            ARG_REDUCE_BODY
                .matches("blockIdx.x == 0 && threadIdx.x == 0")
                .count(),
            2
        );
        assert_eq!(
            ARG_REDUCE_BODY
                .matches("__launch_bounds__(BLOCK_DIM)")
                .count(),
            6
        );
        assert_eq!(ARG_REDUCE_BODY.matches("if (end > numel)").count(), 2);
    }

    #[test]
    fn comparators_and_nan_guard_are_correct() {
        assert_eq!(ARG_REDUCE_BODY.matches("v > best_val").count(), 3);
        assert_eq!(ARG_REDUCE_BODY.matches("v < best_val").count(), 3);
        // NaN は bit 検査で無視する（4 カーネル。finalize は pidx < 0 を skip）。
        assert_eq!(
            ARG_REDUCE_BODY
                .matches("(__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u")
                .count(),
            4
        );
        assert_eq!(ARG_REDUCE_BODY.matches("if (idx < 0)").count(), 2);
        assert_eq!(ARG_REDUCE_BODY.matches("int best_idx = -1;").count(), 2);
    }

    #[test]
    fn no_warp_width_dependence_or_undefined_symbols() {
        for banned in [
            "__shfl",
            "WARP_",
            "warpSize",
            "__shared__",
            "__syncthreads",
            "isnan",
            "INFINITY",
            "#include",
            "double",
            "atomic",
        ] {
            assert!(!ARG_REDUCE_BODY.contains(banned), "{banned}");
        }
    }
}
