//! interpolate（nearest・bilinear）の HIP カーネルソース（`backend-cuda::kernels_interpolate` の
//! 移植。イシュー #3439・分解案 R8）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_interpolate_source`]）。呼び出し元は
//! `interpolate.rs`。2 本のカーネルを 1 ソース（1 module）にまとめ `#define BLOCK_DIM` のみ前置する。
//! shape・stride・scale は実行時にデバイスバッファ／カーネル引数で渡し、ソースへは入れない（A03）。
//!
//! # nearest
//! 空間軸（`a >= spatial_start`）は `src = min((c*in)/out, in-1)`、それ以外の軸は素通し。整数演算のみ。
//!
//! # bilinear と FMA 契約
//! 座標・重み・ブレンドの式は tensor-core の `bilinear_src_coord`／`bilinear_blend`（CPU 参照実装が
//! 共有する単一情報源）と同一の演算順序で、積和 5 箇所（src 座標 2・ブレンド 3）を CUDA 版の `fmaf` から
//! **`__builtin_fmaf` の明示 FMA** へ置き換えた（`f32::mul_add` と同じ丸め。HIPRTC の
//! fp-contract 既定に依存しない。docs/backend-rocm-hiprtc-gemm-decision.md §2 と同方針）。
//! `l0x * v00` のような積は `fmaf` の第 3 引数で、これ以上の縮約余地はない。
//! 空間軸は末尾 2 軸ちょうど（H, W）。`rank` 可変の先頭軸は素通しで `base_flat` へ畳み込む。
//!
//! # REQ-8
//! `idx >= numel` で早期リターン。src 添字は整数除算・floor の後、`min(i0, in-1)`／
//! `min(i0+1, in-1)` で再度クランプする。ホスト側が空間軸サイズ 0 を拒否する前提だが、
//! warp 幅依存の命令は使わない。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};

/// `(in, out, out_shape, in_shape, in_strides, rank, spatial_start, numel)`。
const NEAREST_ARGS: [ArgKind; 8] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
];

/// `(in, out, out_shape, in_shape, in_strides, rank, scale_h, scale_w, align_corners, numel)`。
const BILINEAR_ARGS: [ArgKind; 10] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::F32,
    ArgKind::F32,
    ArgKind::I32,
    ArgKind::I32,
];

/// nearest の spec。
pub(crate) const NEAREST_SPEC: KernelSpec = KernelSpec {
    name: "interpolate_nearest_f32",
    args: &NEAREST_ARGS,
};
/// bilinear の spec。
pub(crate) const BILINEAR_SPEC: KernelSpec = KernelSpec {
    name: "interpolate_bilinear_f32",
    args: &BILINEAR_ARGS,
};

/// 静的テンプレート本体。`BLOCK_DIM` は前置する `#define` で与える。
pub(crate) const INTERPOLATE_BODY: &str = r#"
extern "C" __global__ __launch_bounds__(BLOCK_DIM) void interpolate_nearest_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int* __restrict__ out_shape,
    const int* __restrict__ in_shape,
    const int* __restrict__ in_strides,
    int rank,
    int spatial_start,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long in_flat = 0;
        for (int a = rank - 1; a >= 0; a--) {
            long long axis_size = (long long)out_shape[a];
            long long c = rem % axis_size;
            rem /= axis_size;
            long long src_c;
            if (a >= spatial_start) {
                long long in_size = (long long)in_shape[a];
                src_c = (c * in_size) / (long long)out_shape[a];
                long long max_c = in_size - 1;
                if (src_c > max_c) {
                    src_c = max_c;
                }
            } else {
                src_c = c;
            }
            in_flat += src_c * (long long)in_strides[a];
        }
        out[idx] = in[in_flat];
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void interpolate_bilinear_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int* __restrict__ out_shape,
    const int* __restrict__ in_shape,
    const int* __restrict__ in_strides,
    int rank,
    float scale_h,
    float scale_w,
    int align_corners,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    long long rem = idx;
    long long base_flat = 0;
    long long cy = 0;
    long long cx = 0;
    for (int a = rank - 1; a >= 0; a--) {
        long long axis_size = (long long)out_shape[a];
        long long c = rem % axis_size;
        rem /= axis_size;
        if (a == rank - 1) {
            cx = c;
        } else if (a == rank - 2) {
            cy = c;
        } else {
            base_flat += c * (long long)in_strides[a];
        }
    }

    long long in_h = (long long)in_shape[rank - 2];
    long long in_w = (long long)in_shape[rank - 1];
    long long stride_h = (long long)in_strides[rank - 2];
    long long stride_w = (long long)in_strides[rank - 1];

    float srcy;
    float srcx;
    if (align_corners) {
        srcy = (float)cy * scale_h;
        srcx = (float)cx * scale_w;
    } else {
        srcy = __builtin_fmaf((float)cy + 0.5f, scale_h, -0.5f);
        if (srcy < 0.0f) {
            srcy = 0.0f;
        }
        srcx = __builtin_fmaf((float)cx + 0.5f, scale_w, -0.5f);
        if (srcx < 0.0f) {
            srcx = 0.0f;
        }
    }

    long long i0y = (long long)floorf(srcy);
    if (i0y > in_h - 1) {
        i0y = in_h - 1;
    }
    long long i1y = i0y + 1;
    if (i1y > in_h - 1) {
        i1y = in_h - 1;
    }
    float l1y = srcy - (float)i0y;
    float l0y = 1.0f - l1y;

    long long i0x = (long long)floorf(srcx);
    if (i0x > in_w - 1) {
        i0x = in_w - 1;
    }
    long long i1x = i0x + 1;
    if (i1x > in_w - 1) {
        i1x = in_w - 1;
    }
    float l1x = srcx - (float)i0x;
    float l0x = 1.0f - l1x;

    float v00 = in[base_flat + i0y * stride_h + i0x * stride_w];
    float v01 = in[base_flat + i0y * stride_h + i1x * stride_w];
    float v10 = in[base_flat + i1y * stride_h + i0x * stride_w];
    float v11 = in[base_flat + i1y * stride_h + i1x * stride_w];

    float row0 = __builtin_fmaf(l1x, v01, l0x * v00);
    float row1 = __builtin_fmaf(l1x, v11, l0x * v10);
    out[idx] = __builtin_fmaf(l1y, row1, l0y * row0);
}
"#;

/// interpolate 用ソースを生成する（`BLOCK_DIM` のみ前置）。
pub(crate) fn render_interpolate_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(&[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], INTERPOLATE_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    /// ソース中の `fmaf(` のうち `__builtin_fmaf(` でないもの（素の `fmaf(`）の個数。
    fn bare_fmaf_count(s: &str) -> usize {
        s.matches("fmaf(").count() - s.matches("__builtin_fmaf(").count()
    }

    #[test]
    fn spec_matches_source_signature() {
        for spec in [NEAREST_SPEC, BILINEAR_SPEC] {
            assert_eq!(
                kernel_param_kinds(INTERPOLATE_BODY, spec.name),
                spec.args,
                "{}",
                spec.name
            );
        }
    }

    #[test]
    fn source_defines_only_block_dim_and_is_deterministic() {
        let s = render_interpolate_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define BLOCK_DIM 256"]);
        assert_eq!(s, render_interpolate_source().unwrap());
    }

    #[test]
    fn req8_guards_and_clamps_are_kept() {
        assert!(INTERPOLATE_BODY.contains("if (idx < numel)"));
        assert!(INTERPOLATE_BODY.contains("if (idx >= numel)"));
        // nearest の src クランプ 1 + bilinear の i0/i1 クランプ 4。
        assert!(INTERPOLATE_BODY.contains("if (src_c > max_c) {"));
        for clamp in [
            "if (i0y > in_h - 1)",
            "if (i1y > in_h - 1)",
            "if (i0x > in_w - 1)",
            "if (i1x > in_w - 1)",
        ] {
            assert!(INTERPOLATE_BODY.contains(clamp), "{clamp}");
        }
    }

    #[test]
    fn bilinear_uses_explicit_builtin_fmaf_only() {
        assert_eq!(bare_fmaf_count(INTERPOLATE_BODY), 0);
        assert_eq!(INTERPOLATE_BODY.matches("__builtin_fmaf(").count(), 5);
        for call in [
            "__builtin_fmaf((float)cy + 0.5f, scale_h, -0.5f)",
            "__builtin_fmaf((float)cx + 0.5f, scale_w, -0.5f)",
            "__builtin_fmaf(l1x, v01, l0x * v00)",
            "__builtin_fmaf(l1x, v11, l0x * v10)",
            "__builtin_fmaf(l1y, row1, l0y * row0)",
        ] {
            assert!(INTERPOLATE_BODY.contains(call), "{call}");
        }
    }

    #[test]
    fn bare_fmaf_counter_detects_regressions() {
        assert_eq!(bare_fmaf_count("x = fmaf(a, b, c);"), 1);
        assert_eq!(bare_fmaf_count("x = __builtin_fmaf(a, b, c);"), 0);
    }

    #[test]
    fn no_warp_width_dependent_constructs() {
        for banned in ["__shfl", "warpSize", "WARP_", "__ballot", "__activemask"] {
            assert!(!INTERPOLATE_BODY.contains(banned), "{banned}");
        }
    }
}
