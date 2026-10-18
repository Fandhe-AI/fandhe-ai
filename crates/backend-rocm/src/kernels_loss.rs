//! loss 5 種（mse・huber/smooth_l1・bce・nll・kl_div）の HIP カーネルソース（`backend-cuda` の
//! `kernels_{mse,huber,bce,nll,kl_div}.rs` の移植。イシュー #3436・分解案 R5・親 #3150）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_loss_source`]）。呼び出し元は `loss.rs`
//! （`RocmLoss`）。11 本のカーネル（partial 5・finalize 1・backward 5）を 1 ソース（1 module）に
//! まとめ、HIPRTC のコンパイルと module のロードを 1 回で済ませる。意味論の正は CPU 参照
//! （`backend-cpu` の `mse.rs`・`huber.rs`・`bce.rs`・`nll.rs`・`kl_div.rs`）で、要素関数の式と
//! 演算順序は CPU に揃える。被覆計画は `docs/backend-rocm-ops-coverage-plan.md` §4・§6・§7。
//!
//! # forward の 2 段構成
//! 1. `*_partial_f32`: grid-stride ループで各ブロックが担当区間の要素損失を部分和にし、wave 内
//!    butterfly → `warp_sums[WARPS_PER_BLOCK]` → wave 0 が再縮約して `partial[blockIdx.x]` へ書く。
//! 2. `loss_finalize_f32`: 1 ブロックだけ起動し、`partial` を同じ縮約で総和して `factor`
//!    （`Mean` は `1/n`・`Sum` は `1.0`。ホストが決める）を掛け `out[0]` へ書く。5 種で数式が同じなので
//!    1 本を共有する（結合順序は CUDA の 5 本と同じ: `blockIdx.x` 昇順 → butterfly）。
//!
//! 起動ブロック数は `loss.rs` が `min(ceil_div(n, LOSS_BLOCK_DIM), LOSS_MAX_BLOCKS)` で決め、
//! `partial` の確保長と必ず一致させる（カーネル内では検査できない呼び出し元契約）。
//! 浮動小数点 `atomicAdd` は使わない（決定性。CUDA と同じ）。
//!
//! # wave 幅の注入（`docs/backend-rocm-ops-coverage-plan.md` §6）
//! CUDA 版は 8 warp 固定（`warp_sums[8]`・`lane < 8`）だったが、wave64（CDNA）では
//! `256 / 64 = 4` wave しかなく、固定長のままだと未初期化の `warp_sums` を読む。本ソースは
//! `WarpGeometry::render_hip_defines(Some(LOSS_BLOCK_DIM))` が前置する `WARP_SIZE`・`WARP_HALF`・
//! `WARPS_PER_BLOCK`・`WARP_SHFL_XOR` から配列長と読み出し境界を導く（本体にリテラルを残さない。
//! テストで固定）。`WARPS_PER_BLOCK <= WARP_SIZE` は `warps_per_block` が検査済みなので、
//! wave 0 の 1 回の再縮約で足りる。
//!
//! # FMA と浮動小数点の契約
//! ソース先頭の `#pragma clang fp contract(off)` で暗黙の積和縮約を止める（CPU 参照は演算を
//! 別々に丸める）。FMA を使うのは mse の `diff * diff + acc` だけで、`__builtin_fmaf` で明示する
//! （CPU の `f32::mul_add` と同じ契約）。huber・bce・kl_div・nll は積和ではない合成式なので
//! FMA を使わない（CPU も `mul_add` を使わない）。
//!
//! # f64 アキュムレータ契約の適用判断
//! `coding-rust.md` の対象は「正規化統計の二乗和」と「勾配の長軸縮約」で、loss の forward は
//! 損失値の全要素縮約でそのどちらにも当たらない（CPU は固定チャンク内 `f32` 逐次・CUDA は
//! `f32` butterfly）。backward は要素ごとの演算で縮約を含まない。よって CPU・CUDA と同じ `f32`
//! 累積にする。契約の解釈は変えない。
//!
//! # REQ-8（手動境界検査）と A03
//! partial は `idx < numel`／`idx < n_samples`、finalize は `idx < num_partials`、backward は
//! `idx < numel`／`idx < n_samples`、nll は `0 <= t < num_classes` を維持する。添字は
//! `long long`。外部入力（shape・delta・kind・factor・scale）はソースへ入れず、カーネル引数で
//! 渡す。ソースは `&'static str` の本体と検証済みの `#define` だけから組み立てる。
//!
//! 実機（gfx942）での HIPRTC コンパイル・数学関数の解決と精度・wave64 での縮約順序の差の確認は
//! #3159（段 B）で行う。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, KernelSpec};
use crate::warp_geometry::WarpGeometry;

/// 1 ブロックあたりのスレッド数（`backend-cuda` の `*_BLOCK_DIM` と同値）。
///
/// 64 の倍数にして wave64・wave32 のどちらでも端数 wave を作らない。
pub(crate) const LOSS_BLOCK_DIM: u32 = 256;

/// forward 1 段目の起動ブロック数の上限（＝ `loss_finalize_f32` が単一ブロックで処理する
/// `partial` の最大長。`backend-cuda` の `*_MAX_BLOCKS` と同値）。
pub(crate) const LOSS_MAX_BLOCKS: u32 = 1024;

const _: () = assert!(LOSS_BLOCK_DIM.is_multiple_of(64));

const P: ArgKind = ArgKind::Ptr;
const I: ArgKind = ArgKind::I32;
const F: ArgKind = ArgKind::F32;

const MSE_PARTIAL_ARGS: [ArgKind; 4] = [P, P, P, I];
const HUBER_PARTIAL_ARGS: [ArgKind; 6] = [P, P, P, I, I, F];
const KIND_PARTIAL_ARGS: [ArgKind; 5] = [P, P, P, I, I];
const NLL_PARTIAL_ARGS: [ArgKind; 7] = [P, P, P, I, I, I, I];
const FINALIZE_ARGS: [ArgKind; 4] = [P, P, I, F];
const MSE_BACKWARD_ARGS: [ArgKind; 5] = [P, P, P, I, F];
const HUBER_BACKWARD_ARGS: [ArgKind; 7] = [P, P, P, I, I, F, F];
const KIND_BACKWARD_ARGS: [ArgKind; 6] = [P, P, P, I, I, F];
const NLL_BACKWARD_ARGS: [ArgKind; 7] = [P, P, I, I, I, I, F];

/// `(pred, target, partial, numel)`。
pub(crate) const MSE_PARTIAL_SPEC: KernelSpec = KernelSpec {
    name: "mse_partial_f32",
    args: &MSE_PARTIAL_ARGS,
};
/// `(pred, target, partial, numel, kind, delta)`。
pub(crate) const HUBER_PARTIAL_SPEC: KernelSpec = KernelSpec {
    name: "huber_partial_f32",
    args: &HUBER_PARTIAL_ARGS,
};
/// `(input, target, partial, numel, kind)`。
pub(crate) const BCE_PARTIAL_SPEC: KernelSpec = KernelSpec {
    name: "bce_partial_f32",
    args: &KIND_PARTIAL_ARGS,
};
/// `(input, target, partial, numel, kind)`。
pub(crate) const KL_DIV_PARTIAL_SPEC: KernelSpec = KernelSpec {
    name: "kl_div_partial_f32",
    args: &KIND_PARTIAL_ARGS,
};
/// `(input, targets, partial, outer, num_classes, inner, n_samples)`。
pub(crate) const NLL_PARTIAL_SPEC: KernelSpec = KernelSpec {
    name: "nll_partial_f32",
    args: &NLL_PARTIAL_ARGS,
};
/// `(partial, out, num_partials, factor)`。
pub(crate) const FINALIZE_SPEC: KernelSpec = KernelSpec {
    name: "loss_finalize_f32",
    args: &FINALIZE_ARGS,
};
/// `(pred, target, dpred, numel, scale)`。
pub(crate) const MSE_BACKWARD_SPEC: KernelSpec = KernelSpec {
    name: "mse_backward_f32",
    args: &MSE_BACKWARD_ARGS,
};
/// `(pred, target, dpred, numel, kind, delta, scale)`。
pub(crate) const HUBER_BACKWARD_SPEC: KernelSpec = KernelSpec {
    name: "huber_backward_f32",
    args: &HUBER_BACKWARD_ARGS,
};
/// `(input, target, dinput, numel, kind, scale)`。
pub(crate) const BCE_BACKWARD_SPEC: KernelSpec = KernelSpec {
    name: "bce_backward_f32",
    args: &KIND_BACKWARD_ARGS,
};
/// `(input, target, dinput, numel, kind, scale)`。
pub(crate) const KL_DIV_BACKWARD_SPEC: KernelSpec = KernelSpec {
    name: "kl_div_backward_f32",
    args: &KIND_BACKWARD_ARGS,
};
/// `(targets, dinput, outer, num_classes, inner, n_samples, scale)`。
pub(crate) const NLL_BACKWARD_SPEC: KernelSpec = KernelSpec {
    name: "nll_backward_f32",
    args: &NLL_BACKWARD_ARGS,
};

/// 全カーネルの spec（ソース署名との照合テスト用）。
#[cfg(test)]
pub(crate) const ALL_SPECS: [KernelSpec; 11] = [
    MSE_PARTIAL_SPEC,
    HUBER_PARTIAL_SPEC,
    BCE_PARTIAL_SPEC,
    KL_DIV_PARTIAL_SPEC,
    NLL_PARTIAL_SPEC,
    FINALIZE_SPEC,
    MSE_BACKWARD_SPEC,
    HUBER_BACKWARD_SPEC,
    BCE_BACKWARD_SPEC,
    KL_DIV_BACKWARD_SPEC,
    NLL_BACKWARD_SPEC,
];

/// 静的テンプレート本体。`BLOCK_DIM`・`WARP_*`・`WARPS_PER_BLOCK`・`WARP_SHFL_XOR` は
/// 前置する `#define` で与える。kind は `int`（Huber: 0=Huber・1=SmoothL1、Bce: 0=Probabilities・
/// 1=Logits、KlDiv: 0=Probabilities・1=LogProbabilities）で、ホスト側が allowlist で写す。
pub(crate) const LOSS_BODY: &str = r#"
#pragma clang fp contract(off)

__device__ __forceinline__ float block_reduce_sum(float acc)
{
    __shared__ float warp_sums[WARPS_PER_BLOCK];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
        acc += WARP_SHFL_XOR(acc, offset);
    }
    if (lane == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0f;
        for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
            block_sum += WARP_SHFL_XOR(block_sum, offset);
        }
    }
    return block_sum;
}

__device__ __forceinline__ float huber_elem_loss(float d, int kind, float delta)
{
    float abs_d = fabsf(d);
    if (kind == 1) {
        if (abs_d < delta) {
            return 0.5f * (d / delta) * d;
        }
        return abs_d - 0.5f * delta;
    }
    if (abs_d < delta) {
        return 0.5f * d * d;
    }
    return delta * (abs_d - 0.5f * delta);
}

__device__ __forceinline__ float huber_elem_grad(float d, int kind, float delta)
{
    if (isnan(d)) {
        return d;
    }
    float abs_d = fabsf(d);
    if (kind == 1) {
        if (abs_d < delta) {
            return d / delta;
        }
        return copysignf(1.0f, d);
    }
    if (abs_d < delta) {
        return d;
    }
    return copysignf(delta, d);
}

__device__ __forceinline__ float bce_elem_loss(float input, float target, int kind)
{
    if (kind == 0) {
        float log_p = fmaxf(logf(input), -100.0f);
        float log_1mp = fmaxf(logf(1.0f - input), -100.0f);
        return -(target * log_p + (1.0f - target) * log_1mp);
    }
    if (input >= 0.0f) {
        return (1.0f - target) * input + log1pf(expf(-input));
    }
    return -target * input + log1pf(expf(input));
}

__device__ __forceinline__ float bce_elem_grad_input(float input, float target, int kind)
{
    if (kind == 0) {
        float denom = fmaxf(input * (1.0f - input), 1e-12f);
        return (input - target) / denom;
    }
    float sigmoid;
    if (input >= 0.0f) {
        sigmoid = 1.0f / (1.0f + expf(-input));
    } else {
        float e = expf(input);
        sigmoid = e / (1.0f + e);
    }
    return sigmoid - target;
}

__device__ __forceinline__ float kl_div_elem_loss(float input, float target, int kind)
{
    if (kind == 0) {
        if (target == 0.0f) {
            return 0.0f;
        }
        return target * (logf(target) - input);
    }
    return expf(target) * (target - input);
}

__device__ __forceinline__ float kl_div_elem_grad_input(float target, int kind)
{
    if (kind == 0) {
        return -target;
    }
    return -expf(target);
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void mse_partial_f32(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ partial,
    int numel)
{
    float acc = 0.0f;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        float diff = pred[idx] - target[idx];
        acc = __builtin_fmaf(diff, diff, acc);
    }
    float block_sum = block_reduce_sum(acc);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = block_sum;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void huber_partial_f32(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ partial,
    int numel,
    int kind,
    float delta)
{
    float acc = 0.0f;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        acc += huber_elem_loss(pred[idx] - target[idx], kind, delta);
    }
    float block_sum = block_reduce_sum(acc);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = block_sum;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void bce_partial_f32(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ partial,
    int numel,
    int kind)
{
    float acc = 0.0f;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        acc += bce_elem_loss(input[idx], target[idx], kind);
    }
    float block_sum = block_reduce_sum(acc);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = block_sum;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void kl_div_partial_f32(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ partial,
    int numel,
    int kind)
{
    float acc = 0.0f;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        acc += kl_div_elem_loss(input[idx], target[idx], kind);
    }
    float block_sum = block_reduce_sum(acc);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = block_sum;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void nll_partial_f32(
    const float* __restrict__ input,
    const int* __restrict__ targets,
    float* __restrict__ partial,
    int outer,
    int num_classes,
    int inner,
    int n_samples)
{
    float acc = 0.0f;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < n_samples; idx += stride) {
        long long o = idx / (long long)inner;
        long long i = idx % (long long)inner;
        int t = targets[idx];
        if (t < 0 || t >= num_classes) {
            continue;
        }
        long long input_idx = (o * (long long)num_classes + (long long)t) * (long long)inner + i;
        acc -= input[input_idx];
    }
    float block_sum = block_reduce_sum(acc);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = block_sum;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void loss_finalize_f32(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int num_partials,
    float factor)
{
    float acc = 0.0f;
    for (int idx = threadIdx.x; idx < num_partials; idx += blockDim.x) {
        acc += partial[idx];
    }
    float block_sum = block_reduce_sum(acc);
    if (threadIdx.x == 0) {
        out[0] = block_sum * factor;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void mse_backward_f32(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ dpred,
    int numel,
    float scale)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        dpred[idx] = scale * (pred[idx] - target[idx]);
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void huber_backward_f32(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ dpred,
    int numel,
    int kind,
    float delta,
    float scale)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        dpred[idx] = scale * huber_elem_grad(pred[idx] - target[idx], kind, delta);
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void bce_backward_f32(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ dinput,
    int numel,
    int kind,
    float scale)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        dinput[idx] = scale * bce_elem_grad_input(input[idx], target[idx], kind);
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void kl_div_backward_f32(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ dinput,
    int numel,
    int kind,
    float scale)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        dinput[idx] = scale * kl_div_elem_grad_input(target[idx], kind);
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void nll_backward_f32(
    const int* __restrict__ targets,
    float* __restrict__ dinput,
    int outer,
    int num_classes,
    int inner,
    int n_samples,
    float scale)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        long long o = idx / (long long)inner;
        long long i = idx % (long long)inner;
        int t = targets[idx];
        if (t >= 0 && t < num_classes) {
            long long input_idx = (o * (long long)num_classes + (long long)t) * (long long)inner + i;
            dinput[input_idx] = -scale;
        }
    }
}
"#;

/// loss 用ソースを生成する（wave 幅の `#define` と `BLOCK_DIM` を前置）。
///
/// `geom` は検証済みの wave 幅（32／64）。`warps_per_block(LOSS_BLOCK_DIM)` が失敗する幅は
/// ここで `InvalidKernelConfig` になり、文字列を組み立てない。
pub(crate) fn render_loss_source(geom: WarpGeometry) -> Result<KernelSource, RocmError> {
    let warp = geom.render_hip_defines(Some(LOSS_BLOCK_DIM))?;
    KernelSource::render_with_warp(&warp, &[("BLOCK_DIM", LOSS_BLOCK_DIM)], LOSS_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    fn geoms() -> [WarpGeometry; 2] {
        [
            WarpGeometry::new(32).unwrap(),
            WarpGeometry::new(64).unwrap(),
        ]
    }

    #[test]
    fn spec_matches_source_signature() {
        for spec in ALL_SPECS {
            assert_eq!(
                kernel_param_kinds(LOSS_BODY, spec.name),
                spec.args,
                "{}",
                spec.name
            );
        }
        // 11 本すべてが LOSS_BODY に存在し、余分なカーネルがない。
        assert_eq!(LOSS_BODY.matches("extern \"C\" __global__").count(), 11);
    }

    #[test]
    fn rendered_source_injects_wave_defines_for_both_widths() {
        for (g, half, warps) in [(geoms()[0], 16, 8), (geoms()[1], 32, 4)] {
            let w = g.width();
            let src = render_loss_source(g).unwrap();
            let text = src.as_str();
            assert!(
                text.starts_with(&format!(
                    "#define WARP_SIZE {w}\n#define WARP_HALF {half}\n"
                )),
                "w{w}"
            );
            assert!(text.contains(&format!("#define WARPS_PER_BLOCK {warps}\n")));
            assert!(text.contains("#define BLOCK_DIM 256\n"));
            assert_eq!(text, render_loss_source(g).unwrap().as_str());
        }
    }

    #[test]
    fn body_has_no_cuda_warp_constants() {
        for banned in [
            "warp_sums[8]",
            "lane < 8",
            "% 32",
            "/ 32",
            "offset = 16",
            "0xffffffff",
            "__shfl_xor_sync",
            "__syncwarp",
            "atomicAdd",
        ] {
            assert!(!LOSS_BODY.contains(banned), "{banned}");
        }
        // 素の fmaf（暗黙の関数名解決・縮約に頼る形）は使わず、__builtin_fmaf だけを使う。
        assert!(!LOSS_BODY.replace("__builtin_fmaf(", "").contains("fmaf("));
        assert_eq!(
            LOSS_BODY.matches("__builtin_fmaf(diff, diff, acc)").count(),
            1
        );
        assert!(LOSS_BODY.contains("#pragma clang fp contract(off)"));
        assert!(LOSS_BODY.contains("__shared__ float warp_sums[WARPS_PER_BLOCK];"));
        assert!(LOSS_BODY.contains("(lane < WARPS_PER_BLOCK)"));
    }

    #[test]
    fn req8_bounds_and_long_long_indices() {
        for name in [
            "mse_partial_f32",
            "huber_partial_f32",
            "bce_partial_f32",
            "kl_div_partial_f32",
        ] {
            let body = kernel_body(name);
            assert!(body.contains("idx < numel; idx += stride"), "{name}");
            assert!(body.contains("for (long long idx"), "{name}");
            assert!(body.contains("long long stride"), "{name}");
        }
        let nll = kernel_body("nll_partial_f32");
        assert!(nll.contains("idx < n_samples; idx += stride"));
        assert!(nll.contains("t < 0 || t >= num_classes"));
        assert!(kernel_body("loss_finalize_f32").contains("idx < num_partials"));
        for name in [
            "mse_backward_f32",
            "huber_backward_f32",
            "bce_backward_f32",
            "kl_div_backward_f32",
        ] {
            let body = kernel_body(name);
            assert!(body.contains("long long idx"), "{name}");
            assert!(body.contains("if (idx < numel)"), "{name}");
        }
        let nb = kernel_body("nll_backward_f32");
        assert!(nb.contains("long long idx"));
        assert!(nb.contains("if (idx < n_samples)"));
        assert!(nb.contains("t >= 0 && t < num_classes"));
        // 全カーネルに __launch_bounds__(BLOCK_DIM) が付く。
        assert_eq!(
            LOSS_BODY
                .matches("extern \"C\" __global__ __launch_bounds__(BLOCK_DIM) void")
                .count(),
            11
        );
    }

    #[test]
    fn unsupported_widths_are_rejected_before_render() {
        for w in [0, 16, 48, 128] {
            assert!(WarpGeometry::new(w).is_err(), "{w}");
        }
    }

    #[test]
    fn no_user_controlled_text_in_defines() {
        for g in geoms() {
            let src = render_loss_source(g).unwrap();
            let defines: Vec<_> = src
                .as_str()
                .lines()
                .filter(|l| l.starts_with("#define"))
                .collect();
            assert!(defines.contains(&"#define BLOCK_DIM 256"));
            assert!(!src.as_str().contains('\0'));
        }
    }

    /// `name` のカーネル定義 1 本分の本文（次の `extern "C"` の直前まで）。
    fn kernel_body(name: &str) -> &'static str {
        let marker = format!("void {name}(");
        let start = LOSS_BODY.find(&marker).unwrap();
        let rest = &LOSS_BODY[start..];
        let end = rest[1..].find("extern \"C\"").map_or(rest.len(), |p| p + 1);
        &rest[..end]
    }
}
