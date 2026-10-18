//! 定数 pad（`torch.nn.functional.pad(mode='constant')` 相当）の HIP カーネルソース
//! （`backend-cuda::kernels_constant_pad` の移植。イシュー #3438・分解案 R7）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_constant_pad_source`]）。呼び出し元は
//! `constant_pad.rs`。shape・before・stride は `int` 配列のデバイスバッファ、`value` は `float` 引数で
//! 渡し、ソースへは入れない（A03）。前置する `#define` は `BLOCK_DIM` のみ。
//!
//! 数値契約: 「内部ならコピー・外側なら `value`」だけの純コピーで算術を含まず、CPU 参照と bit 一致する
//! （`value` の NaN／inf もそのまま書く）。FMA の余地はない。REQ-8: `idx < numel` を維持し、各軸の
//! `src_c` が `[0, in_shape[a])` に入るときだけ `in` を読む。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};

/// `(in, out, out_shape, in_shape, before, in_strides, rank, numel, value)`。
const PAD_ARGS: [ArgKind; 9] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::F32,
];

/// pad の spec。
pub(crate) const PAD_SPEC: KernelSpec = KernelSpec {
    name: "constant_pad_f32",
    args: &PAD_ARGS,
};

/// 静的テンプレート本体。
pub(crate) const CONSTANT_PAD_BODY: &str = r#"#pragma clang fp contract(off)

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void constant_pad_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int* __restrict__ out_shape,
    const int* __restrict__ in_shape,
    const int* __restrict__ before,
    const int* __restrict__ in_strides,
    int rank,
    int numel,
    float value)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long in_flat = 0;
        bool inside = true;
        for (int a = rank - 1; a >= 0; a--) {
            long long axis_size = (long long)out_shape[a];
            long long c = rem % axis_size;
            rem /= axis_size;
            long long src_c = c - (long long)before[a];
            if (src_c < 0 || src_c >= (long long)in_shape[a]) {
                inside = false;
                break;
            }
            in_flat += src_c * (long long)in_strides[a];
        }
        out[idx] = inside ? in[in_flat] : value;
    }
}
"#;

/// pad 用ソースを生成する（`BLOCK_DIM` のみ前置）。
pub(crate) fn render_constant_pad_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(&[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], CONSTANT_PAD_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    #[test]
    fn spec_matches_source_signature() {
        assert_eq!(
            kernel_param_kinds(CONSTANT_PAD_BODY, PAD_SPEC.name),
            PAD_SPEC.args
        );
    }

    #[test]
    fn source_defines_only_block_dim_and_is_deterministic() {
        let s = render_constant_pad_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define BLOCK_DIM 256"]);
        assert_eq!(s, render_constant_pad_source().unwrap());
    }

    #[test]
    fn req8_guards_and_pure_copy() {
        assert!(CONSTANT_PAD_BODY.contains("if (idx < numel)"));
        assert!(CONSTANT_PAD_BODY.contains("long long idx"));
        assert!(CONSTANT_PAD_BODY.contains("if (src_c < 0 || src_c >= (long long)in_shape[a])"));
        assert!(CONSTANT_PAD_BODY.contains("out[idx] = inside ? in[in_flat] : value;"));
        assert!(!CONSTANT_PAD_BODY.contains("double"));
        assert_eq!(CONSTANT_PAD_BODY.matches("fmaf(").count(), 0);
        assert!(CONSTANT_PAD_BODY.starts_with("#pragma clang fp contract(off)\n"));
    }

    #[test]
    fn no_warp_width_dependent_constructs() {
        for banned in ["__shfl", "warpSize", "WARP_", "__ballot", "__activemask"] {
            assert!(!CONSTANT_PAD_BODY.contains(banned), "{banned}");
        }
    }
}
