//! 汎用縮約（`sum`／`max`／`min`。全軸・単一軸）の HIP カーネルソース
//! （`backend-cuda::kernels_reduce` の wave 幅対応移植。イシュー #3434・分解案 R3）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_reduce_source`]）。呼び出し元は `reduce.rs`。
//! 9 本のカーネル（{sum, max, min} × {all_partial, all_finalize, axis}）を 1 ソース（1 module）に
//! まとめ、`WarpGeometry::render_hip_defines(Some(REDUCE_BLOCK_DIM))` の wave 幅マクロ
//! （`WARP_SIZE`・`WARP_HALF`・`WARP_LANE`・`WARP_ID`・`WARP_SHFL_XOR`・`WARPS_PER_BLOCK`）と
//! 数値の `#define BLOCK_DIM` だけを前置する（`KernelSource::render_with_warp`。外部入力の文字列は
//! ソースに入れない。A03）。
//!
//! # CUDA 版との差分（wave 幅対応。`docs/backend-rocm-ops-coverage-plan.md` §6）
//! CUDA 版は「8 warp × 32 lane」固定（`warp_sums[8]`・`lane < 8`・`% 32`・`offset = 16`・
//! `__shfl_xor_sync(0xffffffff, …)`）で、これを wave64（gfx942）へそのまま持ち込むと
//! `warp_sums` の書き込みは 4 要素・読み出しは 8 要素になり未初期化読みになる
//! （`warp_model.rs::cuda_fixed_eight_warps_breaks_on_wave64` で固定済み）。本移植は
//! `WARPS_PER_BLOCK = BLOCK_DIM / wave 幅`（wave64 で 4・wave32 で 8）から
//! バッファ長と 2 段目の読み出し境界を導く。`__shfl_xor` はマスク引数を取らない非 sync 版
//! （`WARP_SHFL_XOR`）。`double` オーバーロードの HIPRTC コンパイルは実機未確認（#3159）。
//!
//! CUDA の `*_lastaxis_f32`（1 warp = 1 行の coalesced 版）は移植しない。`inner == 1` も汎用の
//! `*_axis_f32`（1 スレッド = 1 出力・軸を昇順に逐次）で処理する。block = wave 幅の起動には明示
//! geometry の新しい起動経路が要り unsafe 監査の対象が増えるため。汎用版は CPU の
//! `axis_reduce`／`axis_reduce_sum` と同じ逐次順序なので CPU と bit 一致する。
//!
//! # 数値契約
//! - sum: `double` アキュムレータで累積し、書き出し箇所（`finalize` の `out[0]`・`axis` の
//!   `out[idx]`）でのみ 1 回 `(float)` へ downcast する（`.claude/rules/coding-rust.md`）。
//!   全軸の partial バッファは `double`。
//! - max／min: `float acc` と `fmaxf`／`fminf` の厳密選択（丸めなし・NaN を伝播しない。
//!   `f32::max`／`f32::min` と同じ）。±inf は `__uint_as_float` の bit パターンで作り
//!   `INFINITY` マクロに依存しない（CUDA #1893 と同じ理由）。
//! - FMA: 対象カーネルに乗算と加算の連鎖はなく（加算・選択のみ）、`__builtin_fmaf` の適用対象は
//!   ない。縮約の暗黙融合を避けるためソース先頭で `#pragma clang fp contract(off)` を置く。
//! - `atomic*` は使わず結合順序は決定的（`blockIdx.x` 昇順・lane の butterfly 固定）。
//!
//! # REQ-8
//! 全カーネルで手動境界チェック（`idx < numel`／`idx < num_partials`／`idx < total`）を維持する。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};
use crate::target::GfxTarget;
use crate::warp_geometry::WarpGeometry;

/// 縮約カーネルのブロックあたりスレッド数（`INDEX_OPS_BLOCK_DIM` と同値。`run_blocks`／
/// `run` が起動するブロック寸法と一致させる）。
pub(crate) const REDUCE_BLOCK_DIM: u32 = 256;

const _: () = assert!(REDUCE_BLOCK_DIM == INDEX_OPS_BLOCK_DIM);

/// 全軸縮約 2 段目が単一ブロックで読み切る partial の最大長（1 段目の起動ブロック数の上限）。
pub(crate) const REDUCE_MAX_BLOCKS: u32 = 1024;

/// `(in, partial, numel)`。
const ALL_PARTIAL_ARGS: [ArgKind; 3] = [ArgKind::Ptr, ArgKind::Ptr, ArgKind::I32];
/// `(partial, out, num_partials)`。
const ALL_FINALIZE_ARGS: [ArgKind; 3] = [ArgKind::Ptr, ArgKind::Ptr, ArgKind::I32];
/// `(in, out, outer, axis_len, inner)`。
const AXIS_ARGS: [ArgKind; 5] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
];

macro_rules! spec {
    ($name:ident, $kernel:literal, $args:expr) => {
        pub(crate) const $name: KernelSpec = KernelSpec {
            name: $kernel,
            args: &$args,
        };
    };
}

spec!(
    SUM_ALL_PARTIAL_SPEC,
    "reduce_sum_all_partial_f32",
    ALL_PARTIAL_ARGS
);
spec!(
    SUM_ALL_FINALIZE_SPEC,
    "reduce_sum_all_finalize_f32",
    ALL_FINALIZE_ARGS
);
spec!(SUM_AXIS_SPEC, "reduce_sum_axis_f32", AXIS_ARGS);
spec!(
    MAX_ALL_PARTIAL_SPEC,
    "reduce_max_all_partial_f32",
    ALL_PARTIAL_ARGS
);
spec!(
    MAX_ALL_FINALIZE_SPEC,
    "reduce_max_all_finalize_f32",
    ALL_FINALIZE_ARGS
);
spec!(MAX_AXIS_SPEC, "reduce_max_axis_f32", AXIS_ARGS);
spec!(
    MIN_ALL_PARTIAL_SPEC,
    "reduce_min_all_partial_f32",
    ALL_PARTIAL_ARGS
);
spec!(
    MIN_ALL_FINALIZE_SPEC,
    "reduce_min_all_finalize_f32",
    ALL_FINALIZE_ARGS
);
spec!(MIN_AXIS_SPEC, "reduce_min_axis_f32", AXIS_ARGS);

/// 9 本の spec（テスト用の列挙）。
#[cfg(test)]
pub(crate) const ALL_SPECS: [KernelSpec; 9] = [
    SUM_ALL_PARTIAL_SPEC,
    SUM_ALL_FINALIZE_SPEC,
    SUM_AXIS_SPEC,
    MAX_ALL_PARTIAL_SPEC,
    MAX_ALL_FINALIZE_SPEC,
    MAX_AXIS_SPEC,
    MIN_ALL_PARTIAL_SPEC,
    MIN_ALL_FINALIZE_SPEC,
    MIN_AXIS_SPEC,
];

/// 静的テンプレート本体。wave 幅マクロと `BLOCK_DIM` は前置する `#define` で与える。
pub(crate) const REDUCE_BODY: &str = r#"
#pragma clang fp contract(off)
#define NEG_INF_F32 (__uint_as_float(0xff800000u))
#define POS_INF_F32 (__uint_as_float(0x7f800000u))

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_sum_all_partial_f32(
    const float* __restrict__ in,
    double* __restrict__ partial,
    int numel)
{
    __shared__ double warp_sums[WARPS_PER_BLOCK];
    int lane = WARP_LANE(threadIdx.x);
    int warp_id = WARP_ID(threadIdx.x);

    double acc = 0.0;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        acc += (double)in[idx];
    }

    for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
        acc += WARP_SHFL_XOR(acc, offset);
    }
    if (lane == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        double block_sum = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0;
        for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
            block_sum += WARP_SHFL_XOR(block_sum, offset);
        }
        if (lane == 0) {
            partial[blockIdx.x] = block_sum;
        }
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_sum_all_finalize_f32(
    const double* __restrict__ partial,
    float* __restrict__ out,
    int num_partials)
{
    __shared__ double warp_sums[WARPS_PER_BLOCK];
    int lane = WARP_LANE(threadIdx.x);
    int warp_id = WARP_ID(threadIdx.x);

    double acc = 0.0;
    for (int idx = threadIdx.x; idx < num_partials; idx += blockDim.x) {
        acc += partial[idx];
    }

    for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
        acc += WARP_SHFL_XOR(acc, offset);
    }
    if (lane == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        double block_sum = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0;
        for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
            block_sum += WARP_SHFL_XOR(block_sum, offset);
        }
        if (lane == 0) {
            out[0] = (float)block_sum;
        }
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_sum_axis_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int outer,
    int axis_len,
    int inner)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)outer * (long long)inner;
    if (idx < total) {
        long long o = idx / inner;
        long long i = idx % inner;
        double acc = 0.0;
        for (long long a = 0; a < axis_len; a++) {
            long long src = (o * (long long)axis_len + a) * (long long)inner + i;
            acc += (double)in[src];
        }
        out[idx] = (float)acc;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_max_all_partial_f32(
    const float* __restrict__ in,
    float* __restrict__ partial,
    int numel)
{
    __shared__ float warp_maxes[WARPS_PER_BLOCK];
    int lane = WARP_LANE(threadIdx.x);
    int warp_id = WARP_ID(threadIdx.x);

    float acc = NEG_INF_F32;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        acc = fmaxf(acc, in[idx]);
    }

    for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
        acc = fmaxf(acc, WARP_SHFL_XOR(acc, offset));
    }
    if (lane == 0) {
        warp_maxes[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_max = (lane < WARPS_PER_BLOCK) ? warp_maxes[lane] : NEG_INF_F32;
        for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, WARP_SHFL_XOR(block_max, offset));
        }
        if (lane == 0) {
            partial[blockIdx.x] = block_max;
        }
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_max_all_finalize_f32(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int num_partials)
{
    __shared__ float warp_maxes[WARPS_PER_BLOCK];
    int lane = WARP_LANE(threadIdx.x);
    int warp_id = WARP_ID(threadIdx.x);

    float acc = NEG_INF_F32;
    for (int idx = threadIdx.x; idx < num_partials; idx += blockDim.x) {
        acc = fmaxf(acc, partial[idx]);
    }

    for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
        acc = fmaxf(acc, WARP_SHFL_XOR(acc, offset));
    }
    if (lane == 0) {
        warp_maxes[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_max = (lane < WARPS_PER_BLOCK) ? warp_maxes[lane] : NEG_INF_F32;
        for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, WARP_SHFL_XOR(block_max, offset));
        }
        if (lane == 0) {
            out[0] = block_max;
        }
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_max_axis_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int outer,
    int axis_len,
    int inner)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)outer * (long long)inner;
    if (idx < total) {
        long long o = idx / inner;
        long long i = idx % inner;
        float acc = NEG_INF_F32;
        for (long long a = 0; a < axis_len; a++) {
            long long src = (o * (long long)axis_len + a) * (long long)inner + i;
            acc = fmaxf(acc, in[src]);
        }
        out[idx] = acc;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_min_all_partial_f32(
    const float* __restrict__ in,
    float* __restrict__ partial,
    int numel)
{
    __shared__ float warp_mins[WARPS_PER_BLOCK];
    int lane = WARP_LANE(threadIdx.x);
    int warp_id = WARP_ID(threadIdx.x);

    float acc = POS_INF_F32;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        acc = fminf(acc, in[idx]);
    }

    for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
        acc = fminf(acc, WARP_SHFL_XOR(acc, offset));
    }
    if (lane == 0) {
        warp_mins[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_min = (lane < WARPS_PER_BLOCK) ? warp_mins[lane] : POS_INF_F32;
        for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
            block_min = fminf(block_min, WARP_SHFL_XOR(block_min, offset));
        }
        if (lane == 0) {
            partial[blockIdx.x] = block_min;
        }
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_min_all_finalize_f32(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int num_partials)
{
    __shared__ float warp_mins[WARPS_PER_BLOCK];
    int lane = WARP_LANE(threadIdx.x);
    int warp_id = WARP_ID(threadIdx.x);

    float acc = POS_INF_F32;
    for (int idx = threadIdx.x; idx < num_partials; idx += blockDim.x) {
        acc = fminf(acc, partial[idx]);
    }

    for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
        acc = fminf(acc, WARP_SHFL_XOR(acc, offset));
    }
    if (lane == 0) {
        warp_mins[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_min = (lane < WARPS_PER_BLOCK) ? warp_mins[lane] : POS_INF_F32;
        for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
            block_min = fminf(block_min, WARP_SHFL_XOR(block_min, offset));
        }
        if (lane == 0) {
            out[0] = block_min;
        }
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void reduce_min_axis_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int outer,
    int axis_len,
    int inner)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)outer * (long long)inner;
    if (idx < total) {
        long long o = idx / inner;
        long long i = idx % inner;
        float acc = POS_INF_F32;
        for (long long a = 0; a < axis_len; a++) {
            long long src = (o * (long long)axis_len + a) * (long long)inner + i;
            acc = fminf(acc, in[src]);
        }
        out[idx] = acc;
    }
}
"#;

/// `target` の wave 幅を `WarpGeometry` 経由で注入した縮約ソースを生成する。
///
/// `BLOCK_DIM`（= [`REDUCE_BLOCK_DIM`]）が wave 幅の倍数で warp 数が wave 幅以下であることは
/// `WarpGeometry::warps_per_block` が検証する（fail-closed）。
pub(crate) fn render_reduce_source(target: GfxTarget) -> Result<KernelSource, RocmError> {
    let warp = WarpGeometry::for_target(target).render_hip_defines(Some(REDUCE_BLOCK_DIM))?;
    KernelSource::render_with_warp(&warp, &[("BLOCK_DIM", REDUCE_BLOCK_DIM)], REDUCE_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    #[test]
    fn spec_matches_source_signature() {
        for spec in ALL_SPECS {
            assert_eq!(
                kernel_param_kinds(REDUCE_BODY, spec.name),
                spec.args,
                "{}",
                spec.name
            );
        }
        // `double*` もポインタとして分類される。
        assert_eq!(
            kernel_param_kinds(REDUCE_BODY, "reduce_sum_all_partial_f32")[1],
            ArgKind::Ptr
        );
    }

    #[test]
    fn cuda_fixed_warp_literals_are_absent() {
        for banned in [
            "warp_sums[8]",
            "warp_maxes[8]",
            "warp_mins[8]",
            "[8]",
            "lane < 8",
            "% 32",
            "/ 32",
            "offset = 16",
            "_sync(",
            "0xffffffff,",
            "__syncwarp",
            "atomicAdd",
            "atomicMax",
            "INFINITY",
            "warpSize",
            "lastaxis",
        ] {
            assert!(!REDUCE_BODY.contains(banned), "{banned}");
        }
        // 前置した固定 0xffffffffu 系の sync マスクも含まない。
        assert!(!REDUCE_BODY.contains("__shfl_xor_sync"));
    }

    #[test]
    fn req8_guards_are_kept() {
        assert_eq!(REDUCE_BODY.matches("idx < numel").count(), 3);
        assert_eq!(REDUCE_BODY.matches("idx < num_partials").count(), 3);
        assert_eq!(REDUCE_BODY.matches("if (idx < total)").count(), 3);
        assert_eq!(
            REDUCE_BODY.matches("__launch_bounds__(BLOCK_DIM)").count(),
            9
        );
        assert_eq!(REDUCE_BODY.matches("__syncthreads();").count(), 6);
    }

    #[test]
    fn sum_uses_double_accumulator_and_single_downcast() {
        assert!(REDUCE_BODY.contains("double acc = 0.0;"));
        assert!(!REDUCE_BODY.contains("float acc = 0.0"));
        // sum 系の float acc は無い（max/min は NEG/POS_INF 初期化の float acc のみ）。
        assert_eq!(REDUCE_BODY.matches("float acc = ").count(), 6);
        // downcast は finalize の out[0] と axis の out[idx] の 2 か所だけ。
        assert_eq!(REDUCE_BODY.matches("(float)").count(), 2);
        assert!(REDUCE_BODY.contains("out[0] = (float)block_sum;"));
        assert!(REDUCE_BODY.contains("out[idx] = (float)acc;"));
    }

    #[test]
    fn fma_contract_pragma_and_no_explicit_fma_chain() {
        assert!(REDUCE_BODY.contains("#pragma clang fp contract(off)"));
        // 乗算と加算の連鎖がないため __builtin_fmaf は不要（PR 本文に「FMA 該当なし」と記載）。
        assert!(!REDUCE_BODY.contains("fmaf"));
    }

    #[test]
    fn warp_geometry_is_injected_for_gfx942() {
        let s = render_reduce_source(GfxTarget::Gfx942).unwrap();
        let src = s.as_str();
        assert!(src.contains("#define WARP_SIZE 64\n"));
        assert!(src.contains("#define WARPS_PER_BLOCK 4\n"));
        assert!(src.contains("#define BLOCK_DIM 256\n"));
        // 決定的。
        assert_eq!(s, render_reduce_source(GfxTarget::Gfx942).unwrap());
        // 本体の前置部には wave 幅マクロと BLOCK_DIM 以外の #define を足していない。
        let head = &src[..src.find("#pragma clang").unwrap()];
        assert_eq!(head.lines().filter(|l| l.starts_with("#define")).count(), 8);
    }
}
