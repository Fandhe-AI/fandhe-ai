//! 累積和・累積積（`cumsum`／`cumprod`）の HIP カーネルソース（`backend-cuda::kernels_scan` の
//! 移植。イシュー #3439・分解案 R8）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_scan_source`]）。呼び出し元は `scan.rs`。
//! 2 本のカーネルを 1 ソース（1 module）にまとめ、`#define BLOCK_DIM` だけを前置する。
//!
//! # lane 逐次 scan
//! `BackendOps::cumsum`／`cumprod` の数値契約（`dim` 以外の軸の組＝lane ごとに `f64`
//! アキュムレータを保持し `dim` 昇順に逐次計算。各ステップの出力はアキュムレータを `f32` へ
//! downcast したスナップショットで、次ステップは読み戻さない）は `backend-cpu::scan` と bit
//! 完全一致を要求する。ブロック内並列 scan は結合順序が変わるため使えず、「1 スレッド = 1 lane」
//! の逐次構造のみを取る（lane 数が少ない形状では並列度が出ない。CUDA 版と同じ既知の制約）。
//! 加算のみ・乗算のみの連鎖なので FMA 縮約の余地はなく、HIPRTC 既定オプションのままで CPU と
//! 一致する。`double` は gfx942 でネイティブ対応。warp 幅依存の命令は使わない（wave64 で
//! そのまま正しい）。
//!
//! # REQ-8
//! `gid >= lanes` を検査してから処理する。添字計算は `long long`。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};

/// 累積和カーネルの spec。
pub(crate) const CUMSUM_SPEC: KernelSpec = KernelSpec {
    name: "cumsum_f32",
    args: &SCAN_ARGS,
};
/// 累積積カーネルの spec。
pub(crate) const CUMPROD_SPEC: KernelSpec = KernelSpec {
    name: "cumprod_f32",
    args: &SCAN_ARGS,
};

/// `(x, out, lanes, axis_len, inner)`。
const SCAN_ARGS: [ArgKind; 5] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
];

/// 静的テンプレート本体。`BLOCK_DIM` は前置する `#define` で与える。
pub(crate) const SCAN_BODY: &str = r#"
extern "C" __global__ __launch_bounds__(BLOCK_DIM) void cumsum_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int lanes,
    int axis_len,
    int inner)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)lanes) {
        return;
    }
    long long o = gid / (long long)inner;
    long long i = gid % (long long)inner;
    double acc = 0.0;
    for (int a = 0; a < axis_len; a++) {
        long long idx = (o * (long long)axis_len + (long long)a) * (long long)inner + i;
        acc = acc + (double)x[idx];
        out[idx] = (float)acc;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void cumprod_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int lanes,
    int axis_len,
    int inner)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)lanes) {
        return;
    }
    long long o = gid / (long long)inner;
    long long i = gid % (long long)inner;
    double acc = 1.0;
    for (int a = 0; a < axis_len; a++) {
        long long idx = (o * (long long)axis_len + (long long)a) * (long long)inner + i;
        acc = acc * (double)x[idx];
        out[idx] = (float)acc;
    }
}
"#;

/// scan 用ソースを生成する（`BLOCK_DIM` のみ前置）。
pub(crate) fn render_scan_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(&[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], SCAN_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    #[test]
    fn spec_matches_source_signature() {
        for spec in [CUMSUM_SPEC, CUMPROD_SPEC] {
            assert_eq!(kernel_param_kinds(SCAN_BODY, spec.name), spec.args);
        }
    }

    #[test]
    fn source_defines_only_block_dim() {
        let s = render_scan_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define BLOCK_DIM 256"]);
        assert_eq!(s, render_scan_source().unwrap());
    }

    #[test]
    fn req8_guard_and_double_accumulator() {
        assert_eq!(SCAN_BODY.matches("if (gid >= (long long)lanes)").count(), 2);
        assert!(SCAN_BODY.contains("double acc = 0.0;"));
        assert!(SCAN_BODY.contains("double acc = 1.0;"));
        assert!(!SCAN_BODY.contains("float acc"));
        assert_eq!(SCAN_BODY.matches("out[idx] = (float)acc;").count(), 2);
    }

    #[test]
    fn no_warp_width_dependent_constructs() {
        for banned in ["__shfl", "warpSize", "WARP_", "__ballot", "__activemask"] {
            assert!(!SCAN_BODY.contains(banned), "{banned}");
        }
    }
}
