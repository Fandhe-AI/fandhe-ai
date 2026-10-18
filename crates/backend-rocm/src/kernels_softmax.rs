//! `softmax`・`log_softmax` の HIP カーネルソース（`backend-cuda::kernels_softmax` の 2 パス経路の
//! 移植。イシュー #3435・分解案 R4）。
//!
//! 役割: HIPRTC へ渡すソースの生成元（[`render_softmax_source`]）。呼び出し元は `softmax.rs`
//! （`RocmSoftmax`）。1 ソース（1 module）に 2 カーネルを置き、wave 幅は [`WarpGeometry`] の
//! `#define`（`WARP_SIZE`・`WARP_HALF`・`WARP_SHFL_XOR`）で注入する。
//!
//! # CUDA 版との差分
//! - wave 幅は 32 固定ではなく注入（wave64＝gfx942 でそのまま正しい）。`__shfl_xor_sync` と 32 bit
//!   マスク、`__syncwarp`（`WARP_SYNC`）は使わない。2 パス経路は SMEM を使わず同期も要らない。
//! - **onepass（動的 SMEM）経路は移植しない**: `launch.rs` は `sharedMemBytes` を 0 固定で渡し、
//!   動的共有メモリの口がない。2 パス経路は Pass 1 とビット同一の `raw = x[i]` を再読するため
//!   数値契約・REQ-2 判定は onepass と同じ（性能のみの差。範囲外として PR に記録）。
//! - persistent grid の導出（SM 数）はデバイス属性の照会口がないため行わず、`grid = min(rows,
//!   SOFTMAX_MAX_GRID)` ＋カーネル内 grid-stride ループ（`row += gridDim.x`）で全行を覆う。
//!
//! # 数値契約（CUDA 版を踏襲。詳細は `backend-cuda/src/kernels_softmax.rs` 冒頭）
//! - 差分 `raw - m`（`m` は生ドメインの走査済み最大値）を取ってから `scale`（= log2 e）を掛け、
//!   `exp2f` のみを使う（`raw * scale` を先に作ると `±f32::MAX` 付近で overflow して NaN になる）。
//! - `m_new > m`（厳密不等号）のときだけ補正係数を掛ける。lane 初期値は有限の最小値リテラル
//!   `SOFTMAX_MASK_E2`（`INFINITY`／`__FLT_MAX__` は未定義マクロの教訓〈#1101〉で使わない）。
//!   非空行では最大値要素が `exp2f(0) = 1` を加えるため `l >= 1` で `1.0f / l` は有限。
//! - `log_softmax` は `(raw - m) - logf(l)`（CPU 参照と同じ Sterbenz 順序）。
//! - bit 一致は主張せず REQ-2 統一複合判定で検証する（`exp2f`／`logf` の丸めと wave 内結合順が
//!   CPU と異なる）。
//!
//! # FMA 契約
//! 先頭に `#pragma clang fp contract(off)` を置く（hiprtc は `-ffp-contract` を受け付けないため、
//! 暗黙の mul+add 縮約を止める手段はこの pragma のみ。`kernels_linalg.rs` と同方針）。`l *= corr;
//! l += e;`・butterfly 結合 `l_self + l_peer` は縮約させず、ホストモデルと CPU 参照が非縮約で
//! あることに合わせる。FMA を意図する箇所はない（必要なら `__builtin_fmaf` で明示する）。
//!
//! # REQ-8
//! `float4` 経路は `cols % 4 == 0` のときだけ使い、`if (base + 3 < cols)` を維持する。スカラー経路は
//! `i < cols`。`hipMalloc` の整列と `cols % 4 == 0`・`base % 4 == 0` で 16 byte 整列が成立する。
//! 行・列添字は `long long`。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, KernelSpec};
use crate::warp_geometry::WarpGeometry;

/// grid の固定上限（行数がこれを超えてもカーネル内 grid-stride ループで覆う）。
pub(crate) const SOFTMAX_MAX_GRID: u32 = 65535;

/// `(x, out, rows, cols, scale)`。
const SOFTMAX_ARGS: [ArgKind; 5] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::F32,
];

/// softmax カーネルの spec。
pub(crate) const SOFTMAX_SPEC: KernelSpec = KernelSpec {
    name: "softmax_f32_twopass",
    args: &SOFTMAX_ARGS,
};

/// log_softmax カーネルの spec。
pub(crate) const LOG_SOFTMAX_SPEC: KernelSpec = KernelSpec {
    name: "log_softmax_f32_twopass",
    args: &SOFTMAX_ARGS,
};

/// 静的テンプレート本体（wave 幅は前置する `WARP_*` の `#define` で与える）。
pub(crate) const SOFTMAX_BODY: &str = r#"#pragma clang fp contract(off)
#define SOFTMAX_MASK_E2 (-3.402823466e+38f)

extern "C" __global__ __launch_bounds__(WARP_SIZE) void softmax_f32_twopass(
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int cols,
    float scale)
{
    int lane = threadIdx.x;
    int vec_cols = (cols % 4 == 0) ? cols : 0;

    for (long long row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* x_row = x + row * (long long)cols;
        float* out_row = out + row * (long long)cols;

        float m = SOFTMAX_MASK_E2;
        float l = 0.0f;
        for (long long base = (long long)lane * 4; base < vec_cols; base += WARP_SIZE * 4) {
            if (base + 3 < cols) {
                float4 v4 = *reinterpret_cast<const float4*>(x_row + base);
                float vals[4] = { v4.x, v4.y, v4.z, v4.w };
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float raw = vals[k];
                    float m_new = fmaxf(m, raw);
                    if (m_new > m) {
                        l *= exp2f((m - m_new) * scale);
                    }
                    l += exp2f((raw - m_new) * scale);
                    m = m_new;
                }
            }
        }
        for (long long i = (long long)vec_cols + lane; i < cols; i += WARP_SIZE) {
            float raw = x_row[i];
            float m_new = fmaxf(m, raw);
            if (m_new > m) {
                l *= exp2f((m - m_new) * scale);
            }
            l += exp2f((raw - m_new) * scale);
            m = m_new;
        }

        #pragma unroll
        for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
            float m_o = WARP_SHFL_XOR(m, offset);
            float l_o = WARP_SHFL_XOR(l, offset);
            float m_t = fmaxf(m, m_o);
            float l_self = (m_t > m) ? (l * exp2f((m - m_t) * scale)) : l;
            float l_peer = (m_t > m_o) ? (l_o * exp2f((m_o - m_t) * scale)) : l_o;
            l = l_self + l_peer;
            m = m_t;
        }

        float inv_l = 1.0f / l;

        for (long long base = (long long)lane * 4; base < vec_cols; base += WARP_SIZE * 4) {
            if (base + 3 < cols) {
                float4 v4 = *reinterpret_cast<const float4*>(x_row + base);
                float4 o;
                o.x = exp2f((v4.x - m) * scale) * inv_l;
                o.y = exp2f((v4.y - m) * scale) * inv_l;
                o.z = exp2f((v4.z - m) * scale) * inv_l;
                o.w = exp2f((v4.w - m) * scale) * inv_l;
                *reinterpret_cast<float4*>(out_row + base) = o;
            }
        }
        for (long long i = (long long)vec_cols + lane; i < cols; i += WARP_SIZE) {
            out_row[i] = exp2f((x_row[i] - m) * scale) * inv_l;
        }
    }
}

extern "C" __global__ __launch_bounds__(WARP_SIZE) void log_softmax_f32_twopass(
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int cols,
    float scale)
{
    int lane = threadIdx.x;
    int vec_cols = (cols % 4 == 0) ? cols : 0;

    for (long long row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* x_row = x + row * (long long)cols;
        float* out_row = out + row * (long long)cols;

        float m = SOFTMAX_MASK_E2;
        float l = 0.0f;
        for (long long base = (long long)lane * 4; base < vec_cols; base += WARP_SIZE * 4) {
            if (base + 3 < cols) {
                float4 v4 = *reinterpret_cast<const float4*>(x_row + base);
                float vals[4] = { v4.x, v4.y, v4.z, v4.w };
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float raw = vals[k];
                    float m_new = fmaxf(m, raw);
                    if (m_new > m) {
                        l *= exp2f((m - m_new) * scale);
                    }
                    l += exp2f((raw - m_new) * scale);
                    m = m_new;
                }
            }
        }
        for (long long i = (long long)vec_cols + lane; i < cols; i += WARP_SIZE) {
            float raw = x_row[i];
            float m_new = fmaxf(m, raw);
            if (m_new > m) {
                l *= exp2f((m - m_new) * scale);
            }
            l += exp2f((raw - m_new) * scale);
            m = m_new;
        }

        #pragma unroll
        for (int offset = WARP_HALF; offset > 0; offset >>= 1) {
            float m_o = WARP_SHFL_XOR(m, offset);
            float l_o = WARP_SHFL_XOR(l, offset);
            float m_t = fmaxf(m, m_o);
            float l_self = (m_t > m) ? (l * exp2f((m - m_t) * scale)) : l;
            float l_peer = (m_t > m_o) ? (l_o * exp2f((m_o - m_t) * scale)) : l_o;
            l = l_self + l_peer;
            m = m_t;
        }

        float log_l = logf(l);

        for (long long base = (long long)lane * 4; base < vec_cols; base += WARP_SIZE * 4) {
            if (base + 3 < cols) {
                float4 v4 = *reinterpret_cast<const float4*>(x_row + base);
                float4 o;
                o.x = (v4.x - m) - log_l;
                o.y = (v4.y - m) - log_l;
                o.z = (v4.z - m) - log_l;
                o.w = (v4.w - m) - log_l;
                *reinterpret_cast<float4*>(out_row + base) = o;
            }
        }
        for (long long i = (long long)vec_cols + lane; i < cols; i += WARP_SIZE) {
            out_row[i] = (x_row[i] - m) - log_l;
        }
    }
}
"#;

/// softmax／log_softmax 用ソースを生成する（wave 幅の `#define` だけを前置）。
pub(crate) fn render_softmax_source(geom: WarpGeometry) -> Result<KernelSource, RocmError> {
    KernelSource::render_with_warp(&geom.render_hip_defines(None)?, &[], SOFTMAX_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    #[test]
    fn spec_matches_source_signature() {
        for spec in [SOFTMAX_SPEC, LOG_SOFTMAX_SPEC] {
            assert_eq!(kernel_param_kinds(SOFTMAX_BODY, spec.name), spec.args);
        }
    }

    #[test]
    fn fp_contract_is_off_and_first() {
        assert!(SOFTMAX_BODY.starts_with("#pragma clang fp contract(off)\n"));
    }

    #[test]
    fn req8_guards_and_long_long_indices() {
        // 2 カーネル × (vec 経路 2 箇所)。
        assert_eq!(SOFTMAX_BODY.matches("if (base + 3 < cols)").count(), 4);
        assert_eq!(SOFTMAX_BODY.matches("i < cols").count(), 4);
        assert_eq!(SOFTMAX_BODY.matches("row < rows").count(), 2);
        assert!(!SOFTMAX_BODY.contains("int row ="));
        assert!(!SOFTMAX_BODY.contains("int base ="));
        assert!(SOFTMAX_BODY.contains("long long row = blockIdx.x"));
    }

    #[test]
    fn no_wave32_assumptions_or_undefined_macros() {
        for banned in [
            "__shfl_xor_sync",
            "0xffffffff",
            "__syncwarp",
            "WARP_SYNC",
            "INFINITY",
            "__FLT_MAX__",
            "#include",
            "expf(",
            "+= 32",
            "__shared__",
            "extern __shared__",
        ] {
            assert!(!SOFTMAX_BODY.contains(banned), "{banned}");
        }
        assert!(SOFTMAX_BODY.contains("WARP_SHFL_XOR(m, offset)"));
        assert!(SOFTMAX_BODY.contains("float log_l = logf(l);"));
        assert!(SOFTMAX_BODY.contains("m_new > m"));
    }

    #[test]
    fn render_is_deterministic_for_both_wave_widths() {
        for w in [32u32, 64] {
            let g = WarpGeometry::new(w).unwrap();
            let a = render_softmax_source(g).unwrap();
            assert_eq!(a, render_softmax_source(g).unwrap());
            assert!(a.as_str().starts_with(&format!("#define WARP_SIZE {w}\n")));
            assert!(a.as_str().ends_with(SOFTMAX_BODY));
            // 数値 #define は WarpGeometry 由来の固定行のみ。
            assert!(!a.as_str().contains("WARPS_PER_BLOCK"));
        }
        assert_ne!(
            render_softmax_source(WarpGeometry::new(32).unwrap()).unwrap(),
            render_softmax_source(WarpGeometry::new(64).unwrap()).unwrap()
        );
    }
}
