//! gather／scatter／one_hot の HIP カーネルソース（`backend-cuda::kernels_gather_scatter` の
//! 移植。イシュー #3439・分解案 R8）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_gather_scatter_source`]）。呼び出し元は
//! `gather_scatter.rs`。4 本のカーネルを 1 ソース（1 module）にまとめ `#define BLOCK_DIM` のみ前置する。
//! shape・stride・dim はソースへ入れず、デバイスバッファ／カーネル引数で渡す（A03）。
//!
//! 意味論は CPU 参照実装（`backend-cpu::gather_scatter`）と同じ。rank 可変の shape は、出力
//! flat 添字を末尾軸から `%`／`/=` で剥がしながら使う一過性のスカラー変数だけで処理し、座標
//! 配列を保持しない（rank に上限を設けない）。`scatter` は出力位置ごとに index を `j` 昇順に
//! 走査し、Overwrite は最後に一致した値が残り、Add は `double` アキュムレータへ逐次加算して最後に
//! 1 回だけ `(float)` へ downcast する（`ScatterReduce` の決定的集約契約・coding-rust.md の
//! 長軸縮約方針）。加算のみの連鎖で FMA 縮約の余地はない。warp 幅依存の命令は使わない。
//!
//! # REQ-8
//! 全カーネルで `idx < numel` を維持する。範囲外 index はカーネル内でも検査し（gather は `0.0f`、
//! one_hot は全 0 行）、ホスト側（`gather_scatter.rs`）が起動前に値域を全走査して拒否するため
//! 通常は到達しない縦深防御。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};

/// `(in, index, out, out_shape, in_strides, rank, dim, numel, in_dim_size)`。
const GATHER_ARGS: [ArgKind; 9] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
];

/// `(input, index, src, out, out_shape, index_shape, index_strides, rank, dim, numel)`。
const SCATTER_ARGS: [ArgKind; 10] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
];

/// `(index, out, num_classes, numel)`。
const ONE_HOT_ARGS: [ArgKind; 4] = [ArgKind::Ptr, ArgKind::Ptr, ArgKind::I32, ArgKind::I32];

/// gather の spec。
pub(crate) const GATHER_SPEC: KernelSpec = KernelSpec {
    name: "gather_f32",
    args: &GATHER_ARGS,
};
/// scatter（上書き）の spec。
pub(crate) const SCATTER_OVERWRITE_SPEC: KernelSpec = KernelSpec {
    name: "scatter_overwrite_f32",
    args: &SCATTER_ARGS,
};
/// scatter（加算）の spec。
pub(crate) const SCATTER_ADD_SPEC: KernelSpec = KernelSpec {
    name: "scatter_add_f32",
    args: &SCATTER_ARGS,
};
/// one_hot の spec。
pub(crate) const ONE_HOT_SPEC: KernelSpec = KernelSpec {
    name: "one_hot_f32",
    args: &ONE_HOT_ARGS,
};

/// 静的テンプレート本体。`BLOCK_DIM` は前置する `#define` で与える。
pub(crate) const GATHER_SCATTER_BODY: &str = r#"
extern "C" __global__ __launch_bounds__(BLOCK_DIM) void gather_f32(
    const float* __restrict__ in,
    const int* __restrict__ index,
    float* __restrict__ out,
    const int* __restrict__ out_shape,
    const int* __restrict__ in_strides,
    int rank,
    int dim,
    int numel,
    int in_dim_size)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int index_val = index[idx];
        if (index_val >= 0 && index_val < in_dim_size) {
            long long rem = idx;
            long long in_flat = 0;
            for (int a = rank - 1; a >= 0; a--) {
                long long axis_size = (long long)out_shape[a];
                long long c = rem % axis_size;
                rem /= axis_size;
                long long coord = (a == dim) ? (long long)index_val : c;
                in_flat += coord * (long long)in_strides[a];
            }
            out[idx] = in[in_flat];
        } else {
            out[idx] = 0.0f;
        }
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void scatter_overwrite_f32(
    const float* __restrict__ input,
    const int* __restrict__ index,
    const float* __restrict__ src,
    float* __restrict__ out,
    const int* __restrict__ out_shape,
    const int* __restrict__ index_shape,
    const int* __restrict__ index_strides,
    int rank,
    int dim,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long index_base = 0;
        long long dim_coord = 0;
        bool in_range = true;
        for (int a = rank - 1; a >= 0; a--) {
            long long axis_size = (long long)out_shape[a];
            long long c = rem % axis_size;
            rem /= axis_size;
            if (a == dim) {
                dim_coord = c;
            } else if (c >= (long long)index_shape[a]) {
                in_range = false;
                break;
            } else {
                index_base += c * (long long)index_strides[a];
            }
        }

        float result = input[idx];
        if (in_range) {
            long long dim_len = (long long)index_shape[dim];
            long long dim_stride = (long long)index_strides[dim];
            for (long long j = 0; j < dim_len; j++) {
                long long p = index_base + j * dim_stride;
                if ((long long)index[p] == dim_coord) {
                    result = src[p];
                }
            }
        }
        out[idx] = result;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void scatter_add_f32(
    const float* __restrict__ input,
    const int* __restrict__ index,
    const float* __restrict__ src,
    float* __restrict__ out,
    const int* __restrict__ out_shape,
    const int* __restrict__ index_shape,
    const int* __restrict__ index_strides,
    int rank,
    int dim,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long index_base = 0;
        long long dim_coord = 0;
        bool in_range = true;
        for (int a = rank - 1; a >= 0; a--) {
            long long axis_size = (long long)out_shape[a];
            long long c = rem % axis_size;
            rem /= axis_size;
            if (a == dim) {
                dim_coord = c;
            } else if (c >= (long long)index_shape[a]) {
                in_range = false;
                break;
            } else {
                index_base += c * (long long)index_strides[a];
            }
        }

        double acc = (double)input[idx];
        if (in_range) {
            long long dim_len = (long long)index_shape[dim];
            long long dim_stride = (long long)index_strides[dim];
            for (long long j = 0; j < dim_len; j++) {
                long long p = index_base + j * dim_stride;
                if ((long long)index[p] == dim_coord) {
                    acc += (double)src[p];
                }
            }
        }
        out[idx] = (float)acc;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void one_hot_f32(
    const int* __restrict__ index,
    float* __restrict__ out,
    int num_classes,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long row = idx / num_classes;
        long long c = idx % num_classes;
        int index_val = index[row];
        out[idx] = (index_val >= 0 && index_val < num_classes && (long long)index_val == c)
            ? 1.0f
            : 0.0f;
    }
}
"#;

/// gather/scatter/one_hot 用ソースを生成する（`BLOCK_DIM` のみ前置）。
pub(crate) fn render_gather_scatter_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(&[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], GATHER_SCATTER_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    const SPECS: [KernelSpec; 4] = [
        GATHER_SPEC,
        SCATTER_OVERWRITE_SPEC,
        SCATTER_ADD_SPEC,
        ONE_HOT_SPEC,
    ];

    /// 1 カーネル分の本体テキスト（次の `extern "C"` 直前まで）。
    fn kernel_text(name: &str) -> &'static str {
        let start = GATHER_SCATTER_BODY
            .find(&format!("void {name}("))
            .expect("kernel");
        let rest = &GATHER_SCATTER_BODY[start..];
        let end = rest[1..].find("extern \"C\"").map_or(rest.len(), |e| e + 1);
        &rest[..end]
    }

    #[test]
    fn spec_matches_source_signature() {
        for spec in SPECS {
            assert_eq!(
                kernel_param_kinds(GATHER_SCATTER_BODY, spec.name),
                spec.args,
                "{}",
                spec.name
            );
        }
    }

    #[test]
    fn source_defines_only_block_dim_and_is_deterministic() {
        let s = render_gather_scatter_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define BLOCK_DIM 256"]);
        assert_eq!(s, render_gather_scatter_source().unwrap());
    }

    #[test]
    fn every_kernel_keeps_req8_guard_and_long_long_index() {
        for spec in SPECS {
            let t = kernel_text(spec.name);
            assert!(t.contains("if (idx < numel)"), "{}", spec.name);
            assert!(t.contains("long long idx"), "{}", spec.name);
        }
    }

    #[test]
    fn scatter_add_uses_double_accumulator_and_single_downcast() {
        let t = kernel_text("scatter_add_f32");
        assert!(t.contains("double acc = (double)input[idx];"));
        assert!(t.contains("acc += (double)src[p];"));
        assert_eq!(t.matches("(float)acc").count(), 1);
        assert!(!t.contains("float acc"));
        for name in ["scatter_overwrite_f32", "gather_f32", "one_hot_f32"] {
            assert!(!kernel_text(name).contains("double"), "{name}");
        }
    }

    #[test]
    fn fallbacks_and_non_dim_range_checks_exist() {
        assert!(kernel_text("gather_f32").contains("out[idx] = 0.0f;"));
        for name in ["scatter_overwrite_f32", "scatter_add_f32"] {
            assert!(kernel_text(name).contains("in_range = false"), "{name}");
        }
    }

    #[test]
    fn no_warp_width_dependent_constructs() {
        for banned in ["__shfl", "warpSize", "WARP_", "__ballot", "__activemask"] {
            assert!(!GATHER_SCATTER_BODY.contains(banned), "{banned}");
        }
    }
}
