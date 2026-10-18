//! MaxPool2d／AvgPool2d／AdaptiveAvgPool2d の HIP カーネルソース（`backend-cuda::kernels_pooling` の
//! 移植。イシュー #3438・分解案 R7）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_pooling_source`]）。呼び出し元は `pooling.rs`。
//! 3 本のカーネルを 1 ソース（1 module）にまとめ、前置する `#define` は `BLOCK_DIM` のみ。
//! 形状パラメータはすべてカーネル引数（`int`）で渡し、ソースへは入れない（A03）。
//! 1 出力要素 = 1 スレッド。wave 幅に依存する命令は使わない（`WarpGeometry` は不要）。
//!
//! # 数値契約（`docs/pooling-ops-design.md` §5／§7・coding-rust.md）
//! - `max_pool2d_f32`: 窓内を `kh` 外側・`kw` 内側で走査し `v > best || (isnan(v) && !isnan(best))` のときだけ
//!   更新する（同値は先勝ち・最初に現れた NaN の索引を保持）。`-INFINITY` 番兵は使わず `found` フラグで
//!   最初の有効タップから始める。算術を含まない選択演算なので値・索引とも CPU 参照と bit 一致する。
//!   索引は (n, c) 平面内の `h * w_in + w`（`int`。ホストが `h_in * w_in <= i32::MAX` を保証）。
//! - `avg_pool2d_f32`／`adaptive_avg_pool2d_f32`: 窓内を row-major 固定順で `double` へ逐次加算し、`double`
//!   で除算して最後に 1 回だけ `float` へ落とす（勾配の長軸縮約は f64 の方針）。CPU 参照の `f64` 逐次和と
//!   演算順まで同じで bit 一致する。avg は dilation を引数に持たない（1 固定。ホストが dilation ≠ 1 を
//!   `Unsupported` にする）。
//! - float の積和はどのカーネルにも無く、`__builtin_fmaf` を使う箇所はない。本体先頭の
//!   `#pragma clang fp contract(off)` は、将来の編集で暗黙 FMA が入るのを止める予防策（R2 と同じ）。
//!
//! # REQ-8
//! `idx < numel` を維持し、添字は `long long` で計算する。`in` は `0 <= h < h_in && 0 <= w < w_in` を確かめて
//! から読む（padding 位置はスキップ）。adaptive は CUDA が窓の式の性質に頼る窓端を、`h_end`／`w_end` の手動
//! クランプで明示的に守る（有効な入力では値は変わらない）。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};

/// `(in, out, idx_out, n_batch, c, h_in, w_in, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw, numel)`。
const MAX_ARGS: [ArgKind; 18] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
];

/// `(in, out, n_batch, c, h_in, w_in, h_out, w_out, kh, kw, sh, sw, ph, pw, count_include_pad, numel)`。
const AVG_ARGS: [ArgKind; 16] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
];

/// `(in, out, n_batch, c, h_in, w_in, h_out, w_out, numel)`。
const ADAPTIVE_ARGS: [ArgKind; 9] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
];

/// max_pool2d の spec。
pub(crate) const MAX_POOL_SPEC: KernelSpec = KernelSpec {
    name: "max_pool2d_f32",
    args: &MAX_ARGS,
};
/// avg_pool2d の spec。
pub(crate) const AVG_POOL_SPEC: KernelSpec = KernelSpec {
    name: "avg_pool2d_f32",
    args: &AVG_ARGS,
};
/// adaptive_avg_pool2d の spec。
pub(crate) const ADAPTIVE_POOL_SPEC: KernelSpec = KernelSpec {
    name: "adaptive_avg_pool2d_f32",
    args: &ADAPTIVE_ARGS,
};

/// 静的テンプレート本体。`BLOCK_DIM` は前置する `#define` で与える。
pub(crate) const POOLING_BODY: &str = r#"#pragma clang fp contract(off)

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void max_pool2d_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int* __restrict__ idx_out,
    int n_batch, int c, int h_in, int w_in, int h_out, int w_out,
    int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long ow = rem % w_out;
        rem /= w_out;
        long long oh = rem % h_out;
        rem /= h_out;
        long long cc = rem % c;
        long long n = rem / c;

        float best = 0.0f;
        int best_idx = 0;
        int found = 0;

        for (int kh_ = 0; kh_ < kh; kh_++) {
            long long h = oh * (long long)sh + (long long)kh_ * dh - (long long)ph;
            if (h < 0 || h >= h_in) continue;
            for (int kw_ = 0; kw_ < kw; kw_++) {
                long long w = ow * (long long)sw + (long long)kw_ * dw - (long long)pw;
                if (w < 0 || w >= w_in) continue;
                long long in_idx = ((n * (long long)c + cc) * h_in + h) * w_in + w;
                float v = in[in_idx];
                int cur_idx = (int)(h * w_in + w);
                if (!found) {
                    best = v;
                    best_idx = cur_idx;
                    found = 1;
                } else if (v > best || (isnan(v) && !isnan(best))) {
                    best = v;
                    best_idx = cur_idx;
                }
            }
        }
        out[idx] = best;
        idx_out[idx] = best_idx;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void avg_pool2d_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n_batch, int c, int h_in, int w_in, int h_out, int w_out,
    int kh, int kw, int sh, int sw, int ph, int pw,
    int count_include_pad, int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long ow = rem % w_out;
        rem /= w_out;
        long long oh = rem % h_out;
        rem /= h_out;
        long long cc = rem % c;
        long long n = rem / c;

        double acc = 0.0;
        long long valid_count = 0;
        for (int kh_ = 0; kh_ < kh; kh_++) {
            long long h = oh * (long long)sh + (long long)kh_ - (long long)ph;
            if (h < 0 || h >= h_in) continue;
            for (int kw_ = 0; kw_ < kw; kw_++) {
                long long w = ow * (long long)sw + (long long)kw_ - (long long)pw;
                if (w < 0 || w >= w_in) continue;
                long long in_idx = ((n * (long long)c + cc) * h_in + h) * w_in + w;
                acc += (double)in[in_idx];
                valid_count++;
            }
        }
        long long divisor = count_include_pad ? ((long long)kh * (long long)kw) : valid_count;
        out[idx] = (float)(acc / (double)divisor);
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void adaptive_avg_pool2d_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n_batch, int c, int h_in, int w_in, int h_out, int w_out,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long ow = rem % w_out;
        rem /= w_out;
        long long oh = rem % h_out;
        rem /= h_out;
        long long cc = rem % c;
        long long n = rem / c;

        long long h_start = (oh * (long long)h_in) / h_out;
        long long h_end = ((oh + 1) * (long long)h_in + h_out - 1) / h_out;
        long long w_start = (ow * (long long)w_in) / w_out;
        long long w_end = ((ow + 1) * (long long)w_in + w_out - 1) / w_out;
        if (h_end > h_in) h_end = h_in;
        if (w_end > w_in) w_end = w_in;

        double acc = 0.0;
        for (long long h = h_start; h < h_end; h++) {
            for (long long w = w_start; w < w_end; w++) {
                long long in_idx = ((n * (long long)c + cc) * h_in + h) * w_in + w;
                acc += (double)in[in_idx];
            }
        }
        long long divisor = (h_end - h_start) * (w_end - w_start);
        out[idx] = (float)(acc / (double)divisor);
    }
}
"#;

/// pooling 3 種用ソースを生成する（`BLOCK_DIM` のみ前置）。
pub(crate) fn render_pooling_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(&[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], POOLING_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    /// 関数 `name` の本体（次の `extern "C"` まで）。
    fn kernel_text(name: &str) -> &'static str {
        let start = POOLING_BODY
            .find(&format!("void {name}("))
            .expect("kernel exists");
        let rest = &POOLING_BODY[start..];
        let end = rest[1..].find("extern \"C\"").map_or(rest.len(), |e| e + 1);
        &rest[..end]
    }

    #[test]
    fn spec_matches_source_signature() {
        for spec in [MAX_POOL_SPEC, AVG_POOL_SPEC, ADAPTIVE_POOL_SPEC] {
            assert_eq!(
                kernel_param_kinds(POOLING_BODY, spec.name),
                spec.args,
                "{}",
                spec.name
            );
        }
    }

    #[test]
    fn source_defines_only_block_dim_and_is_deterministic() {
        let s = render_pooling_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define BLOCK_DIM 256"]);
        assert_eq!(s, render_pooling_source().unwrap());
    }

    #[test]
    fn req8_guards_are_kept() {
        assert_eq!(POOLING_BODY.matches("if (idx < numel)").count(), 3);
        assert_eq!(POOLING_BODY.matches("long long idx").count(), 3);
        assert_eq!(
            POOLING_BODY
                .matches("if (h < 0 || h >= h_in) continue;")
                .count(),
            2
        );
        assert_eq!(
            POOLING_BODY
                .matches("if (w < 0 || w >= w_in) continue;")
                .count(),
            2
        );
    }

    #[test]
    fn max_pool_is_pure_selection() {
        let k = kernel_text("max_pool2d_f32");
        assert!(!k.contains("double"));
        assert!(k.contains("} else if (v > best || (isnan(v) && !isnan(best))) {"));
        assert!(k.contains("if (!found) {"));
        assert!(!k.contains("INFINITY"));
        assert!(k.contains("int cur_idx = (int)(h * w_in + w);"));
    }

    #[test]
    fn avg_kernels_use_f64_accumulator_with_single_downcast() {
        for name in ["avg_pool2d_f32", "adaptive_avg_pool2d_f32"] {
            let k = kernel_text(name);
            assert_eq!(k.matches("double acc = 0.0;").count(), 1, "{name}");
            assert_eq!(
                k.matches("(float)(acc / (double)divisor)").count(),
                1,
                "{name}"
            );
            assert_eq!(k.matches("(float)").count(), 1, "{name}");
        }
        // avg は dilation を引数に持たない。
        let avg = kernel_text("avg_pool2d_f32");
        assert!(!avg.contains("int dh") && !avg.contains("int dw"));
    }

    #[test]
    fn adaptive_clamps_window_ends() {
        let k = kernel_text("adaptive_avg_pool2d_f32");
        assert!(k.contains("if (h_end > h_in) h_end = h_in;"));
        assert!(k.contains("if (w_end > w_in) w_end = w_in;"));
    }

    #[test]
    fn no_fma_and_contraction_is_disabled() {
        assert_eq!(POOLING_BODY.matches("fmaf(").count(), 0);
        assert!(POOLING_BODY.starts_with("#pragma clang fp contract(off)\n"));
    }

    #[test]
    fn no_warp_width_dependent_constructs() {
        for banned in ["__shfl", "warpSize", "WARP_", "__ballot", "__activemask"] {
            assert!(!POOLING_BODY.contains(banned), "{banned}");
        }
    }
}
