//! Conv2d の im2col／col2im の HIP カーネルソース（`backend-cuda::kernels_im2col` の移植。
//! イシュー #3438・分解案 R7）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_im2col_source`]）。呼び出し元は `im2col.rs`。
//! 2 本のカーネルを 1 ソース（1 module）にまとめ、前置する `#define` は `BLOCK_DIM` のみ。
//! 形状パラメータはすべてカーネル引数（`int` 19 個）で渡し、ソースへは入れない（A03）。
//! `Conv2dParams` は rank 固定 4D（NCHW）なので shape 配列のデバイス転送は要らない。
//!
//! # 出力レイアウト（`tensor-core::im2col_out_shape` と同一）
//! `im2col_f32`: `[N, Cin, H, W]` を `[N, G, K_g, P]`（`K_g` は `(c_in_g, kh, kw)`・`P` は `(oh, ow)` の
//! row-major）へ展開する。`col2im_f32` はその随伴で `[N, G, K_g, P]` を `[N, Cin, H, W]` へ畳み戻す。
//!
//! # 数値契約
//! - `im2col_f32` は「内部ならコピー・padding 位置なら `0.0`」だけの純コピーで算術を含まず、
//!   CPU 参照（`backend-cpu::im2col`）と bit 一致する。
//! - `col2im_f32` は重なり窓の加算順を `kh` 外側・`kw` 内側の row-major に固定し、`double` アキュムレータへ
//!   逐次加算して最後に 1 回だけ `float` へ落とす（coding-rust.md「勾配の長軸縮約は f64」）。HIP は
//!   `double` をネイティブに扱うので Metal の soft-f64 は要らない。加算のみで乗算がないため FMA 縮約の
//!   余地はなく、`__builtin_fmaf` を使う箇所はない。本体先頭の `#pragma clang fp contract(off)` は、
//!   将来の編集で暗黙 FMA が入るのを止める予防策（R2 と同じ）。
//! - 座標計算は `long long` で行い、剰余を取る**前**に符号を判定する（C の負数 `%` の符号曖昧性を避ける）。
//!
//! # REQ-8
//! `idx < numel` のガードを維持する。im2col は `0 <= h < h_in && 0 <= w < w_in`、col2im は `oh < h_out`・
//! `ow < w_out` を確かめてから読む。wave 幅に依存する命令は使わない（`WarpGeometry` は不要）。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};

/// `(in|d_col, out, n_batch, cin, groups, cin_g, h_in, w_in, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw, k_g, p, numel)`。
const IM2COL_ARGS: [ArgKind; 21] = [
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
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
];

/// im2col の spec。
pub(crate) const IM2COL_SPEC: KernelSpec = KernelSpec {
    name: "im2col_f32",
    args: &IM2COL_ARGS,
};
/// col2im の spec（引数署名は im2col と同一）。
pub(crate) const COL2IM_SPEC: KernelSpec = KernelSpec {
    name: "col2im_f32",
    args: &IM2COL_ARGS,
};

/// 静的テンプレート本体。`BLOCK_DIM` は前置する `#define` で与える。
pub(crate) const IM2COL_BODY: &str = r#"#pragma clang fp contract(off)

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void im2col_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n_batch, int cin, int groups, int cin_g,
    int h_in, int w_in, int h_out, int w_out,
    int kh, int kw, int sh, int sw,
    int ph, int pw, int dh, int dw,
    int k_g, int p, int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long p_idx = rem % p;
        rem /= p;
        long long k_idx = rem % k_g;
        rem /= k_g;
        long long g = rem % groups;
        rem /= groups;
        long long n = rem;

        long long kw_ = k_idx % kw;
        long long rest = k_idx / kw;
        long long kh_ = rest % kh;
        long long c_g = rest / kh;
        long long c = g * (long long)cin_g + c_g;

        long long ow = p_idx % w_out;
        long long oh = p_idx / w_out;

        long long h = oh * (long long)sh + kh_ * (long long)dh - (long long)ph;
        long long w = ow * (long long)sw + kw_ * (long long)dw - (long long)pw;

        float value = 0.0f;
        if (h >= 0 && h < h_in && w >= 0 && w < w_in) {
            long long in_idx = ((n * (long long)cin + c) * h_in + h) * w_in + w;
            value = in[in_idx];
        }
        out[idx] = value;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void col2im_f32(
    const float* __restrict__ d_col,
    float* __restrict__ out,
    int n_batch, int cin, int groups, int cin_g,
    int h_in, int w_in, int h_out, int w_out,
    int kh, int kw, int sh, int sw,
    int ph, int pw, int dh, int dw,
    int k_g, int p, int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long w = rem % w_in;
        rem /= w_in;
        long long h = rem % h_in;
        rem /= h_in;
        long long c = rem % cin;
        rem /= cin;
        long long n = rem;

        long long g = c / (long long)cin_g;
        long long c_g = c % (long long)cin_g;

        double acc = 0.0;
        for (int kh_ = 0; kh_ < kh; kh_++) {
            long long num_h = h + (long long)ph - (long long)kh_ * dh;
            if (num_h < 0) continue;
            if (num_h % sh != 0) continue;
            long long oh = num_h / sh;
            if (oh >= h_out) continue;
            for (int kw_ = 0; kw_ < kw; kw_++) {
                long long num_w = w + (long long)pw - (long long)kw_ * dw;
                if (num_w < 0) continue;
                if (num_w % sw != 0) continue;
                long long ow = num_w / sw;
                if (ow >= w_out) continue;

                long long k_idx = (c_g * kh + kh_) * kw + kw_;
                long long p_idx = oh * w_out + ow;
                long long col_idx = ((n * (long long)groups + g) * k_g + k_idx) * p + p_idx;
                acc += (double)d_col[col_idx];
            }
        }
        out[idx] = (float)acc;
    }
}
"#;

/// im2col／col2im 用ソースを生成する（`BLOCK_DIM` のみ前置）。
pub(crate) fn render_im2col_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(&[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], IM2COL_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    #[test]
    fn spec_matches_source_signature() {
        for spec in [IM2COL_SPEC, COL2IM_SPEC] {
            assert_eq!(
                kernel_param_kinds(IM2COL_BODY, spec.name),
                spec.args,
                "{}",
                spec.name
            );
        }
    }

    #[test]
    fn source_defines_only_block_dim_and_is_deterministic() {
        let s = render_im2col_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define BLOCK_DIM 256"]);
        assert_eq!(s, render_im2col_source().unwrap());
    }

    #[test]
    fn req8_guards_are_kept() {
        assert_eq!(IM2COL_BODY.matches("if (idx < numel)").count(), 2);
        assert_eq!(IM2COL_BODY.matches("long long idx").count(), 2);
        assert!(IM2COL_BODY.contains("if (h >= 0 && h < h_in && w >= 0 && w < w_in)"));
        assert!(IM2COL_BODY.contains("if (oh >= h_out) continue;"));
        assert!(IM2COL_BODY.contains("if (ow >= w_out) continue;"));
    }

    #[test]
    fn col2im_checks_sign_before_remainder() {
        for (neg, rem) in [
            ("if (num_h < 0) continue;", "if (num_h % sh != 0) continue;"),
            ("if (num_w < 0) continue;", "if (num_w % sw != 0) continue;"),
        ] {
            let a = IM2COL_BODY.find(neg).expect(neg);
            let b = IM2COL_BODY.find(rem).expect(rem);
            assert!(a < b, "{neg} must precede {rem}");
        }
    }

    #[test]
    fn col2im_uses_f64_accumulator_with_single_downcast() {
        assert_eq!(IM2COL_BODY.matches("double acc = 0.0;").count(), 1);
        assert_eq!(IM2COL_BODY.matches("(float)acc").count(), 1);
        assert!(IM2COL_BODY.contains("acc += (double)d_col[col_idx];"));
        // im2col 側（col2im より前）に double はない。
        let col2im_at = IM2COL_BODY.find("void col2im_f32").unwrap();
        assert!(!IM2COL_BODY[..col2im_at].contains("double"));
    }

    #[test]
    fn no_fma_and_contraction_is_disabled() {
        assert_eq!(IM2COL_BODY.matches("fmaf(").count(), 0);
        assert!(IM2COL_BODY.starts_with("#pragma clang fp contract(off)\n"));
    }

    #[test]
    fn no_warp_width_dependent_constructs() {
        for banned in ["__shfl", "warpSize", "WARP_", "__ballot", "__activemask"] {
            assert!(!IM2COL_BODY.contains(banned), "{banned}");
        }
    }
}
