//! 正規化 6 メソッド（rmsnorm・layer_norm・両 backward・batch_norm の train/infer）の HIP カーネル
//! ソース（`backend-cuda` の `kernels_rmsnorm.rs`〈`RMSNORM_F32_TWOPASS`〉・`kernels_layer_norm.rs`・
//! `kernels_norm_backward.rs`・`kernels_batch_norm.rs` の移植。イシュー #3437・分解案 R6・親 #3150）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_norm_source`]）。呼び出し元は `norm.rs` の
//! `RocmNorm::new`。8 本のカーネルを 1 ソース（1 module）にまとめ、`WarpGeometry` の `#define`
//! （`WARP_SIZE`・`WARP_HALF`・`WARP_SHFL_XOR`）と `BLOCK_DIM` だけを前置する（A03。shape・eps・
//! has_weight など実行時の値はカーネル引数で渡し、ソースへは入れない）。
//!
//! # 構成（CUDA 版との差分。性能最適化は #3159 以降へ送る）
//! - 行・チャネル単位のカーネルは「1 block = 1 wave（`WARP_SIZE` スレッド）= 1 行（1 チャネル）」の
//!   1 対 1 マッピング（grid = `rows`／`c`）で、縮約は wave 内 butterfly だけ。共有メモリ・
//!   `__syncthreads()` は使わない。CUDA の persistent grid・`float4` ベクトル化・SMEM 常駐
//!   （`RMSNORM_F32_TWOPASS` の最適化）は持ち込まない（正しさを優先。性能は保証しない）。
//! - wave 幅依存の定数（`i += 32`・`offset = 16`・`0xffffffffu`・`__shfl_xor_sync`・`__syncwarp`）は
//!   `WARP_SIZE`・`WARP_HALF`・`WARP_SHFL_XOR` に置き換える（CDNA = 64・RDNA = 32。
//!   `docs/backend-rocm-ops-coverage-plan.md` §6）。早期 return は `row >= rows`／`ch >= c` だけで
//!   `blockIdx` に依存するため wave 内で一様であり、shfl の全 lane 参加は崩れない。
//! - dw／dwdb／infer は 1 次元（1 スレッド = 1 列／1 要素。`launch::KernelHost::run`、block = 256）。
//!   infer は CUDA の grid-stride をやめ gid 方式（他の添字系カーネルと同形）にした。
//!
//! # 数値契約（`.claude/rules/coding-rust.md`・変更しない）
//! - 先頭の `#pragma clang fp contract(off)` で暗黙の積和縮約を止め、積和は `__builtin_fmaf`（f32）・
//!   `__builtin_fma`（f64）で明示する（`kernels_linalg.rs` と同方針。ホスト側モデルが同じ丸めを再現できる）。
//! - 正規化統計の二乗和は要素を先に f64 へ昇格してから二乗（`__builtin_fma((double)v,(double)v,acc)`）。
//!   勾配の長軸縮約（dw・db・`dot`・`sum_dxhat`）は要素積を f32 で確定してから f64 へ昇格して蓄積し、
//!   最終書き出しで f32 へ戻すのは 1 回だけ。
//! - affine（`x̂·w+b`）は CPU の 4 分岐（`apply_affine`）と同じ分岐にする: (w,b) は `__builtin_fmaf`、
//!   w のみは `*`、b のみは `+`、なしはそのまま。CUDA 版は常に `fmaf(xhat, wv, bv)` のため -0 の符号だけ
//!   CPU と異なる。ROCm は CPU と bit 一致する側（signed zero を含む）へ寄せた。
//! - rmsnorm の `rstd` は CPU の `rmsnorm_row_scalar`／`eval::row_rms_stats` と同じ
//!   `(float)(1.0 / sqrt(fma(acc, inv_n, eps)))`。
//!
//! # REQ-8・A03
//! 手動境界チェック（`row >= rows`・`ch >= c`・`i < hidden`／`i < m`・`col >= hidden`・`idx >= numel`）を
//! 残す。添字・ループカウンタは `long long`（符号付きオーバーフロー回避）。`rsqrtf`・`INFINITY`・
//! `#include` は使わない。カーネル内コメントは ASCII のみ（HIPRTC への非 ASCII 入力を避ける）。
//! `__shfl_xor` の double オーバーロードの HIPRTC 実コンパイルと `__builtin_fma` の f64 命令化は
//! 実機で未確認（#3159）。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};
use crate::warp_geometry::WarpGeometry;

const P: ArgKind = ArgKind::Ptr;
const I: ArgKind = ArgKind::I32;
const F: ArgKind = ArgKind::F32;

/// `(x, w, out, rows, hidden, eps, inv_n, has_weight)`。
const RMSNORM_ARGS: [ArgKind; 8] = [P, P, P, I, I, F, F, I];
/// `(x, w, b, out, rows, hidden, eps, has_weight, has_bias)`。
const LAYER_NORM_ARGS: [ArgKind; 9] = [P, P, P, P, I, I, F, I, I];
/// `(x, w, dy, dx, rstd_out, rows, hidden, eps, has_weight)`。
const RMSNORM_BWD_DX_ARGS: [ArgKind; 9] = [P, P, P, P, P, I, I, F, I];
/// `(x, dy, rstd_in, dw, rows, hidden)`。
const RMSNORM_BWD_DW_ARGS: [ArgKind; 6] = [P, P, P, P, I, I];
/// `(x, w, dy, dx, mean_out, rstd_out, rows, hidden, eps, has_weight)`。
const LAYER_NORM_BWD_DX_ARGS: [ArgKind; 10] = [P, P, P, P, P, P, I, I, F, I];
/// `(x, dy, mean_in, rstd_in, dw, db, rows, hidden, has_weight, has_bias)`。
const LAYER_NORM_BWD_DWDB_ARGS: [ArgKind; 10] = [P, P, P, P, P, P, I, I, I, I];
/// `(x, w, b, out, mean_out, var_out, c, spatial, m, eps, has_weight, has_bias)`。
const BATCH_NORM_TRAIN_ARGS: [ArgKind; 12] = [P, P, P, P, P, P, I, I, I, F, I, I];
/// `(x, mean, var, w, b, out, c, spatial, numel, eps, has_weight, has_bias)`。
const BATCH_NORM_INFER_ARGS: [ArgKind; 12] = [P, P, P, P, P, P, I, I, ArgKind::I64, F, I, I];

/// rmsnorm forward の spec（1 block = 1 wave = 1 行）。
pub(crate) const RMSNORM_SPEC: KernelSpec = KernelSpec {
    name: "rmsnorm_f32",
    args: &RMSNORM_ARGS,
};
/// layer_norm forward の spec（1 block = 1 wave = 1 行）。
pub(crate) const LAYER_NORM_SPEC: KernelSpec = KernelSpec {
    name: "layer_norm_f32",
    args: &LAYER_NORM_ARGS,
};
/// rmsnorm backward の dx の spec（1 block = 1 wave = 1 行）。
pub(crate) const RMSNORM_BWD_DX_SPEC: KernelSpec = KernelSpec {
    name: "rmsnorm_bwd_dx_f32",
    args: &RMSNORM_BWD_DX_ARGS,
};
/// rmsnorm backward の dw の spec（1 スレッド = 1 列）。
pub(crate) const RMSNORM_BWD_DW_SPEC: KernelSpec = KernelSpec {
    name: "rmsnorm_bwd_dw_f32",
    args: &RMSNORM_BWD_DW_ARGS,
};
/// layer_norm backward の dx の spec（1 block = 1 wave = 1 行）。
pub(crate) const LAYER_NORM_BWD_DX_SPEC: KernelSpec = KernelSpec {
    name: "layer_norm_bwd_dx_f32",
    args: &LAYER_NORM_BWD_DX_ARGS,
};
/// layer_norm backward の dw／db の spec（1 スレッド = 1 列）。
pub(crate) const LAYER_NORM_BWD_DWDB_SPEC: KernelSpec = KernelSpec {
    name: "layer_norm_bwd_dwdb_f32",
    args: &LAYER_NORM_BWD_DWDB_ARGS,
};
/// batch_norm train の spec（1 block = 1 wave = 1 チャネル）。
pub(crate) const BATCH_NORM_TRAIN_SPEC: KernelSpec = KernelSpec {
    name: "batch_norm_train_f32",
    args: &BATCH_NORM_TRAIN_ARGS,
};
/// batch_norm infer の spec（1 スレッド = 1 要素）。
pub(crate) const BATCH_NORM_INFER_SPEC: KernelSpec = KernelSpec {
    name: "batch_norm_infer_f32",
    args: &BATCH_NORM_INFER_ARGS,
};

/// 静的テンプレート本体。`WARP_SIZE`・`WARP_HALF`・`WARP_SHFL_XOR`・`BLOCK_DIM` は前置する
/// `#define` で与える。
pub(crate) const NORM_BODY: &str = r#"#pragma clang fp contract(off)

// Sum over one wave with an xor butterfly (offset WARP_HALF .. 1). Every lane ends with the same value.
__device__ __forceinline__ double wave_sum_f64(double v)
{
    for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
        v += WARP_SHFL_XOR(v, offset);
    }
    return v;
}

// Affine step shared by layer_norm and batch_norm: same 4 branches as the CPU reference.
__device__ __forceinline__ float apply_affine(float xhat, int has_weight, float wv, int has_bias, float bv)
{
    if (has_weight != 0 && has_bias != 0) {
        return __builtin_fmaf(xhat, wv, bv);
    }
    if (has_weight != 0) {
        return xhat * wv;
    }
    if (has_bias != 0) {
        return xhat + bv;
    }
    return xhat;
}

// Channel-local index i (0..m) to the flat NCS index.
__device__ __forceinline__ long long bn_index(long long i, long long ch, long long c, long long spatial)
{
    return (i / spatial) * c * spatial + ch * spatial + (i % spatial);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE) void rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int rows,
    int hidden,
    float eps,
    float inv_n,
    int has_weight)
{
    int lane = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    long long row_base = (long long)row * (long long)hidden;
    const float* x_row = x + row_base;
    float* out_row = out + row_base;

    double acc = 0.0;
    for (long long i = lane; i < hidden; i += WARP_SIZE) {
        double v = (double)x_row[i];
        acc = __builtin_fma(v, v, acc);
    }
    acc = wave_sum_f64(acc);
    float rstd = (float)(1.0 / sqrt(__builtin_fma(acc, (double)inv_n, (double)eps)));

    for (long long i = lane; i < hidden; i += WARP_SIZE) {
        float v = x_row[i] * rstd;
        out_row[i] = (has_weight != 0) ? v * w[i] : v;
    }
}

extern "C" __global__ __launch_bounds__(WARP_SIZE) void layer_norm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int rows,
    int hidden,
    float eps,
    int has_weight,
    int has_bias)
{
    int lane = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    long long row_base = (long long)row * (long long)hidden;
    const float* x_row = x + row_base;
    float* out_row = out + row_base;

    double sum = 0.0;
    for (long long i = lane; i < hidden; i += WARP_SIZE) {
        sum += (double)x_row[i];
    }
    sum = wave_sum_f64(sum);
    double mean = sum / (double)hidden;

    double sq_acc = 0.0;
    for (long long i = lane; i < hidden; i += WARP_SIZE) {
        double d = (double)x_row[i] - mean;
        sq_acc = __builtin_fma(d, d, sq_acc);
    }
    sq_acc = wave_sum_f64(sq_acc);
    double var = sq_acc / (double)hidden;
    double rstd = 1.0 / sqrt(var + (double)eps);

    for (long long i = lane; i < hidden; i += WARP_SIZE) {
        float xhat = (float)(((double)x_row[i] - mean) * rstd);
        float wv = (has_weight != 0) ? w[i] : 1.0f;
        float bv = (has_bias != 0) ? b[i] : 0.0f;
        out_row[i] = apply_affine(xhat, has_weight, wv, has_bias, bv);
    }
}

extern "C" __global__ __launch_bounds__(WARP_SIZE) void rmsnorm_bwd_dx_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ dy,
    float* __restrict__ dx,
    double* __restrict__ rstd_out,
    int rows,
    int hidden,
    float eps,
    int has_weight)
{
    int lane = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    long long row_base = (long long)row * (long long)hidden;
    const float* x_row = x + row_base;
    const float* dy_row = dy + row_base;
    float* dx_row = dx + row_base;

    double acc = 0.0;
    for (long long i = lane; i < hidden; i += WARP_SIZE) {
        double v = (double)x_row[i];
        acc = __builtin_fma(v, v, acc);
    }
    acc = wave_sum_f64(acc);
    double inv_n = 1.0 / (double)hidden;
    float rstd = (float)(1.0 / sqrt(__builtin_fma(acc, inv_n, (double)eps)));
    if (lane == 0) {
        rstd_out[row] = (double)rstd;
    }

    double dot = 0.0;
    for (long long i = lane; i < hidden; i += WARP_SIZE) {
        float xhat = x_row[i] * rstd;
        float dyv = dy_row[i];
        float dxhat = (has_weight != 0) ? dyv * w[i] : dyv;
        float term = dxhat * xhat;
        dot += (double)term;
    }
    dot = wave_sum_f64(dot);
    double mean_dot = dot * inv_n;

    for (long long i = lane; i < hidden; i += WARP_SIZE) {
        float xhat = x_row[i] * rstd;
        float dyv = dy_row[i];
        float dxhat = (has_weight != 0) ? dyv * w[i] : dyv;
        double d = (double)rstd * ((double)dxhat - (double)xhat * mean_dot);
        dx_row[i] = (float)d;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void rmsnorm_bwd_dw_f32(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const double* __restrict__ rstd_in,
    float* __restrict__ dw,
    int rows,
    int hidden)
{
    long long col = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (col >= (long long)hidden) {
        return;
    }
    double acc_w = 0.0;
    for (long long r = 0; r < (long long)rows; r++) {
        long long idx = r * (long long)hidden + col;
        float rstd = (float)rstd_in[r];
        float xhat = x[idx] * rstd;
        float dyv = dy[idx];
        float term = dyv * xhat;
        acc_w = (double)term + acc_w;
    }
    dw[col] = (float)acc_w;
}

extern "C" __global__ __launch_bounds__(WARP_SIZE) void layer_norm_bwd_dx_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ dy,
    float* __restrict__ dx,
    double* __restrict__ mean_out,
    double* __restrict__ rstd_out,
    int rows,
    int hidden,
    float eps,
    int has_weight)
{
    int lane = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    long long row_base = (long long)row * (long long)hidden;
    const float* x_row = x + row_base;
    const float* dy_row = dy + row_base;
    float* dx_row = dx + row_base;

    double sum = 0.0;
    for (long long i = lane; i < hidden; i += WARP_SIZE) {
        sum += (double)x_row[i];
    }
    sum = wave_sum_f64(sum);
    double mean = sum / (double)hidden;

    double sq_acc = 0.0;
    for (long long i = lane; i < hidden; i += WARP_SIZE) {
        double d = (double)x_row[i] - mean;
        sq_acc = __builtin_fma(d, d, sq_acc);
    }
    sq_acc = wave_sum_f64(sq_acc);
    double var = sq_acc / (double)hidden;
    double rstd = 1.0 / sqrt(var + (double)eps);
    if (lane == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    double sum_dxhat = 0.0;
    double dot = 0.0;
    for (long long i = lane; i < hidden; i += WARP_SIZE) {
        float xhat = (float)(((double)x_row[i] - mean) * rstd);
        float dyv = dy_row[i];
        float dxhat = (has_weight != 0) ? dyv * w[i] : dyv;
        sum_dxhat += (double)dxhat;
        float term = dxhat * xhat;
        dot += (double)term;
    }
    sum_dxhat = wave_sum_f64(sum_dxhat);
    dot = wave_sum_f64(dot);
    double mean_dxhat = sum_dxhat / (double)hidden;
    double mean_dot = dot / (double)hidden;

    for (long long i = lane; i < hidden; i += WARP_SIZE) {
        float xhat = (float)(((double)x_row[i] - mean) * rstd);
        float dyv = dy_row[i];
        float dxhat = (has_weight != 0) ? dyv * w[i] : dyv;
        double d = rstd * ((double)dxhat - mean_dxhat - (double)xhat * mean_dot);
        dx_row[i] = (float)d;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void layer_norm_bwd_dwdb_f32(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const double* __restrict__ mean_in,
    const double* __restrict__ rstd_in,
    float* __restrict__ dw,
    float* __restrict__ db,
    int rows,
    int hidden,
    int has_weight,
    int has_bias)
{
    long long col = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (col >= (long long)hidden) {
        return;
    }
    double acc_w = 0.0;
    double acc_b = 0.0;
    for (long long r = 0; r < (long long)rows; r++) {
        long long idx = r * (long long)hidden + col;
        double mean = mean_in[r];
        double rstd = rstd_in[r];
        float dyv = dy[idx];
        if (has_weight != 0) {
            float xhat = (float)(((double)x[idx] - mean) * rstd);
            float term = dyv * xhat;
            acc_w = (double)term + acc_w;
        }
        if (has_bias != 0) {
            acc_b = (double)dyv + acc_b;
        }
    }
    if (has_weight != 0) {
        dw[col] = (float)acc_w;
    }
    if (has_bias != 0) {
        db[col] = (float)acc_b;
    }
}

extern "C" __global__ __launch_bounds__(WARP_SIZE) void batch_norm_train_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    float* __restrict__ mean_out,
    float* __restrict__ var_out,
    int c,
    int spatial,
    int m,
    float eps,
    int has_weight,
    int has_bias)
{
    int lane = threadIdx.x;
    int ch = blockIdx.x;
    if (ch >= c) {
        return;
    }

    double sum = 0.0;
    for (long long i = lane; i < (long long)m; i += WARP_SIZE) {
        sum += (double)x[bn_index(i, ch, c, spatial)];
    }
    sum = wave_sum_f64(sum);
    double mean = sum / (double)m;

    double sq_acc = 0.0;
    for (long long i = lane; i < (long long)m; i += WARP_SIZE) {
        double d = (double)x[bn_index(i, ch, c, spatial)] - mean;
        sq_acc = __builtin_fma(d, d, sq_acc);
    }
    sq_acc = wave_sum_f64(sq_acc);
    double var = sq_acc / (double)m;
    double rstd = 1.0 / sqrt(var + (double)eps);

    if (lane == 0) {
        mean_out[ch] = (float)mean;
        var_out[ch] = (float)var;
    }

    float wv = (has_weight != 0) ? w[ch] : 1.0f;
    float bv = (has_bias != 0) ? b[ch] : 0.0f;
    for (long long i = lane; i < (long long)m; i += WARP_SIZE) {
        long long idx = bn_index(i, ch, c, spatial);
        float xhat = (float)(((double)x[idx] - mean) * rstd);
        out[idx] = apply_affine(xhat, has_weight, wv, has_bias, bv);
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void batch_norm_infer_f32(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int c,
    int spatial,
    long long numel,
    float eps,
    int has_weight,
    int has_bias)
{
    long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }
    int ch = (int)((idx / (long long)spatial) % (long long)c);
    double rstd = 1.0 / sqrt((double)var[ch] + (double)eps);
    float xhat = (float)(((double)x[idx] - (double)mean[ch]) * rstd);
    float wv = (has_weight != 0) ? w[ch] : 1.0f;
    float bv = (has_bias != 0) ? b[ch] : 0.0f;
    out[idx] = apply_affine(xhat, has_weight, wv, has_bias, bv);
}
"#;

/// 正規化用ソースを生成する（`WarpGeometry` の `#define` と `BLOCK_DIM` のみ前置。A03）。
pub(crate) fn render_norm_source(geom: WarpGeometry) -> Result<KernelSource, RocmError> {
    KernelSource::render_with_warp(geom, &[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], NORM_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    const ALL_SPECS: [KernelSpec; 8] = [
        RMSNORM_SPEC,
        LAYER_NORM_SPEC,
        RMSNORM_BWD_DX_SPEC,
        RMSNORM_BWD_DW_SPEC,
        LAYER_NORM_BWD_DX_SPEC,
        LAYER_NORM_BWD_DWDB_SPEC,
        BATCH_NORM_TRAIN_SPEC,
        BATCH_NORM_INFER_SPEC,
    ];

    #[test]
    fn spec_matches_source_signature() {
        for spec in ALL_SPECS {
            assert_eq!(
                kernel_param_kinds(NORM_BODY, spec.name),
                spec.args,
                "{}",
                spec.name
            );
        }
    }

    #[test]
    fn double_pointers_are_pointer_kind_and_numel_is_i64() {
        assert_eq!(BATCH_NORM_INFER_SPEC.args[8], ArgKind::I64);
        assert!(NORM_BODY.contains("double* __restrict__ rstd_out"));
        assert!(NORM_BODY.contains("long long numel"));
    }

    #[test]
    fn source_starts_contract_off_and_defines_are_validated_only() {
        assert!(NORM_BODY.starts_with("#pragma clang fp contract(off)"));
        for w in [32, 64] {
            let g = WarpGeometry::new(w).unwrap();
            let s = render_norm_source(g).unwrap();
            assert_eq!(s, render_norm_source(g).unwrap());
            assert!(!s.as_str().contains("WARPS_PER_BLOCK"));
            assert!(s.as_str().contains(&format!("#define WARP_SIZE {w}\n")));
            assert!(s.as_str().contains("#define BLOCK_DIM 256\n"));
            assert!(s.as_str().ends_with(NORM_BODY));
            assert!(!s.as_str().contains('\0'));
        }
    }

    #[test]
    fn req8_manual_bounds_guards_remain() {
        for guard in [
            "if (row >= rows)",
            "if (ch >= c)",
            "if (col >= (long long)hidden)",
            "if (idx >= numel)",
            "i < hidden",
            "i < (long long)m",
        ] {
            assert!(NORM_BODY.contains(guard), "{guard}");
        }
        assert_eq!(NORM_BODY.matches("if (row >= rows)").count(), 4);
        assert_eq!(
            NORM_BODY.matches("if (col >= (long long)hidden)").count(),
            2
        );
        // ループカウンタは long long（符号付きオーバーフロー回避）。
        assert!(!NORM_BODY.contains("for (int i"));
        assert!(NORM_BODY.contains("for (long long i = lane;"));
    }

    #[test]
    fn no_cuda_wave32_constructs_or_banned_builtins() {
        for banned in [
            "__shfl_xor_sync",
            "__syncwarp",
            "__syncthreads",
            "__shared__",
            "0xffffffff",
            "i += 32",
            "offset = 16",
            "warp_sums[8]",
            "rsqrtf",
            "INFINITY",
            "#include",
            "WARPS_PER_BLOCK",
            "warpSize",
        ] {
            assert!(!NORM_BODY.contains(banned), "{banned}");
        }
        assert!(NORM_BODY.is_ascii(), "カーネルソースは ASCII のみ");
    }

    #[test]
    fn fma_is_always_explicit_builtin() {
        // 素の `fma(`／`fmaf(` を使わない（`__builtin_fma(`／`__builtin_fmaf(` のみ）。
        assert_eq!(
            NORM_BODY.matches("fma(").count(),
            NORM_BODY.matches("__builtin_fma(").count()
        );
        assert_eq!(
            NORM_BODY.matches("fmaf(").count(),
            NORM_BODY.matches("__builtin_fmaf(").count()
        );
        assert!(NORM_BODY.contains("__builtin_fmaf(xhat, wv, bv)"));
    }

    #[test]
    fn sum_of_squares_promotes_to_f64_before_squaring() {
        assert_eq!(
            NORM_BODY
                .matches("double v = (double)x_row[i];\n        acc = __builtin_fma(v, v, acc);")
                .count(),
            2
        );
        assert!(!NORM_BODY.contains("float acc"));
        assert!(
            NORM_BODY
                .contains("(float)(1.0 / sqrt(__builtin_fma(acc, (double)inv_n, (double)eps)))")
        );
        assert!(NORM_BODY.contains("(float)(1.0 / sqrt(__builtin_fma(acc, inv_n, (double)eps)))"));
    }

    #[test]
    fn gradient_reductions_fix_f32_product_before_f64_accumulation() {
        assert!(
            NORM_BODY.contains("float term = dyv * xhat;\n        acc_w = (double)term + acc_w;")
        );
        assert_eq!(
            NORM_BODY.matches("float term = dyv * xhat;").count(),
            2,
            "rmsnorm dw と layer_norm dw"
        );
        assert!(NORM_BODY.contains("acc_b = (double)dyv + acc_b;"));
        assert_eq!(NORM_BODY.matches("float term = dxhat * xhat;").count(), 2);
        assert_eq!(NORM_BODY.matches("dot += (double)term;").count(), 2);
        // 書き出しは f64 から 1 回だけ f32 へ。
        assert!(NORM_BODY.contains("dw[col] = (float)acc_w;"));
        assert!(NORM_BODY.contains("db[col] = (float)acc_b;"));
    }

    #[test]
    fn butterfly_uses_injected_geometry_macros() {
        assert!(NORM_BODY.contains("for (int offset = WARP_HALF; offset > 0; offset >>= 1)"));
        assert!(NORM_BODY.contains("WARP_SHFL_XOR(v, offset)"));
        assert_eq!(NORM_BODY.matches("i += WARP_SIZE").count(), 15);
        assert_eq!(NORM_BODY.matches("__launch_bounds__(WARP_SIZE)").count(), 5);
        assert_eq!(NORM_BODY.matches("__launch_bounds__(BLOCK_DIM)").count(), 3);
    }
}
