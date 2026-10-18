//! sort／topk の HIP カーネルソース（`backend-cuda::kernels_sort` の移植。イシュー #3439・
//! 分解案 R8）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_sort_source`]）。呼び出し元は `sort.rs`。
//! 3 本のカーネルを 1 ソース（1 module）にまとめ `#define BLOCK_DIM` のみ前置する。
//! shape 由来の値（lines・dim_size・inner・padded・out_len 等）はソースへ入れず、カーネル引数で渡す（A03）。
//!
//! # アルゴリズム（CUDA 版と同一）
//! 1. `sort_build_keys_u64`: 各ライン（`dim` 軸に沿った 1 本）を `padded`（2 のべき乗）長の 64bit
//!    合成キー配列へ展開する。上位 32bit は値キー（NaN は `0xFFFFFFFF`＝最大、±0 は同値、それ以外は
//!    符号に応じたビット反転で全順序化。降順は `~vk`）、下位 32bit は元添字（安定化）。パディング域は
//!    `0xFFFFFFFFFFFFFFFF`。
//! 2. `bitonic_step_u64`: ビットニックソートの 1 比較ステップ。`j`・`k` のループはホスト側。整数
//!    compare/swap のみで決定的。
//! 3. `sort_finalize_f32`: 先頭 `out_len`（sort は `dim_size`、topk は `k`）個のキー下位 32bit で
//!    `input` を gather して `values`／`index` へ書く（無演算の読み出しのみ）。
//!
//! 浮動小数点演算を含まず（比較は `isnan`・`==` のみ）、FMA 契約・fp-contract の影響を受けない。
//! warp 幅依存の命令は使わない。
//!
//! # REQ-8
//! 全カーネルで `gid < total` と、`in_pos`／`out_pos`／`o < padded` が確保長に収まることを検査する。
//! 添字演算は `long long`。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};

/// `(input, keys, lines, dim_size, inner, padded, descending, numel_in)`。
const BUILD_KEYS_ARGS: [ArgKind; 8] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I64,
    ArgKind::I64,
    ArgKind::I64,
    ArgKind::I64,
    ArgKind::I32,
    ArgKind::I64,
];

/// `(keys, j, k, padded, lines)`。
const BITONIC_ARGS: [ArgKind; 5] = [
    ArgKind::Ptr,
    ArgKind::I64,
    ArgKind::I64,
    ArgKind::I64,
    ArgKind::I64,
];

/// `(input, keys, values, index, lines, dim_size, inner, padded, out_len, numel_in, numel_out)`。
const FINALIZE_ARGS: [ArgKind; 11] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I64,
    ArgKind::I64,
    ArgKind::I64,
    ArgKind::I64,
    ArgKind::I64,
    ArgKind::I64,
    ArgKind::I64,
];

/// キー構築カーネルの spec。
pub(crate) const BUILD_KEYS_SPEC: KernelSpec = KernelSpec {
    name: "sort_build_keys_u64",
    args: &BUILD_KEYS_ARGS,
};
/// ビットニックステップの spec。
pub(crate) const BITONIC_STEP_SPEC: KernelSpec = KernelSpec {
    name: "bitonic_step_u64",
    args: &BITONIC_ARGS,
};
/// 仕上げカーネルの spec。
pub(crate) const FINALIZE_SPEC: KernelSpec = KernelSpec {
    name: "sort_finalize_f32",
    args: &FINALIZE_ARGS,
};

/// 静的テンプレート本体。`BLOCK_DIM` は前置する `#define` で与える。
pub(crate) const SORT_BODY: &str = r#"
extern "C" __global__ __launch_bounds__(BLOCK_DIM) void sort_build_keys_u64(
    const float* __restrict__ input,
    unsigned long long* __restrict__ keys,
    long long lines,
    long long dim_size,
    long long inner,
    long long padded,
    int descending,
    long long numel_in)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = lines * padded;
    if (gid >= total) {
        return;
    }
    long long line = gid / padded;
    long long i = gid % padded;
    long long out_pos = line * padded + i;
    if (i < dim_size) {
        long long in_pos = (line / inner) * dim_size * inner + (line % inner) + i * inner;
        if (in_pos < 0 || in_pos >= numel_in) {
            return;
        }
        float v = input[in_pos];
        unsigned int vk;
        if (isnan(v)) {
            vk = 0xFFFFFFFFu;
        } else {
            float vv = (v == 0.0f) ? 0.0f : v;
            unsigned int bits = __float_as_uint(vv);
            vk = (bits >> 31) ? (~bits) : (bits | 0x80000000u);
        }
        unsigned int hi = descending ? (~vk) : vk;
        unsigned int lo = (unsigned int)i;
        keys[out_pos] = (((unsigned long long)hi) << 32) | (unsigned long long)lo;
    } else {
        keys[out_pos] = 0xFFFFFFFFFFFFFFFFULL;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void bitonic_step_u64(
    unsigned long long* keys,
    long long j,
    long long k,
    long long padded,
    long long lines)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = lines * padded;
    if (gid >= total) {
        return;
    }
    long long line = gid / padded;
    long long i = gid % padded;
    long long ixj = i ^ j;
    if (ixj <= i || ixj >= padded) {
        return;
    }
    long long base = line * padded;
    unsigned long long a = keys[base + i];
    unsigned long long b = keys[base + ixj];
    bool ascending = (i & k) == 0;
    bool should_swap = ascending ? (a > b) : (a < b);
    if (should_swap) {
        keys[base + i] = b;
        keys[base + ixj] = a;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void sort_finalize_f32(
    const float* __restrict__ input,
    const unsigned long long* __restrict__ keys,
    float* __restrict__ values,
    int* __restrict__ index,
    long long lines,
    long long dim_size,
    long long inner,
    long long padded,
    long long out_len,
    long long numel_in,
    long long numel_out)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = lines * out_len;
    if (gid >= total) {
        return;
    }
    long long line = gid / out_len;
    long long o = gid % out_len;
    if (o >= padded) {
        return;
    }
    unsigned long long key = keys[line * padded + o];
    unsigned long long idxu = key & 0xFFFFFFFFULL;
    if (idxu >= (unsigned long long)dim_size) {
        return;
    }
    long long idx = (long long)idxu;
    long long in_pos = (line / inner) * dim_size * inner + (line % inner) + idx * inner;
    long long out_pos = (line / inner) * out_len * inner + (line % inner) + o * inner;
    if (in_pos < 0 || in_pos >= numel_in || out_pos < 0 || out_pos >= numel_out) {
        return;
    }
    values[out_pos] = input[in_pos];
    index[out_pos] = (int)idx;
}
"#;

/// sort/topk 用ソースを生成する（`BLOCK_DIM` のみ前置）。
pub(crate) fn render_sort_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(&[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], SORT_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    fn kernel_text(name: &str) -> &'static str {
        let start = SORT_BODY.find(&format!("void {name}(")).expect("kernel");
        let rest = &SORT_BODY[start..];
        let end = rest[1..].find("extern \"C\"").map_or(rest.len(), |e| e + 1);
        &rest[..end]
    }

    #[test]
    fn spec_matches_source_signature() {
        for spec in [BUILD_KEYS_SPEC, BITONIC_STEP_SPEC, FINALIZE_SPEC] {
            assert_eq!(
                kernel_param_kinds(SORT_BODY, spec.name),
                spec.args,
                "{}",
                spec.name
            );
        }
    }

    #[test]
    fn source_defines_only_block_dim_and_is_deterministic() {
        let s = render_sort_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define BLOCK_DIM 256"]);
        assert_eq!(s, render_sort_source().unwrap());
    }

    #[test]
    fn req8_guards_are_kept() {
        for name in [
            "sort_build_keys_u64",
            "bitonic_step_u64",
            "sort_finalize_f32",
        ] {
            assert!(kernel_text(name).contains("if (gid >= total)"), "{name}");
            assert!(kernel_text(name).contains("long long gid"), "{name}");
        }
        assert!(kernel_text("sort_build_keys_u64").contains("in_pos >= numel_in"));
        assert!(kernel_text("bitonic_step_u64").contains("ixj >= padded"));
        let fin = kernel_text("sort_finalize_f32");
        assert!(fin.contains("if (o >= padded)"));
        assert!(fin.contains("idxu >= (unsigned long long)dim_size"));
        assert!(fin.contains("in_pos >= numel_in"));
        assert!(fin.contains("out_pos >= numel_out"));
    }

    #[test]
    fn no_float_arithmetic_and_no_warp_dependence() {
        // 浮動小数点の算術（積和）を含まない。比較・ビット変換のみ。
        for banned in ["fmaf", "__fmaf", " * v", "double"] {
            assert!(!SORT_BODY.contains(banned), "{banned}");
        }
        for banned in ["__shfl", "warpSize", "WARP_", "__ballot", "__activemask"] {
            assert!(!SORT_BODY.contains(banned), "{banned}");
        }
    }

    #[test]
    fn key_encoding_constants_are_present() {
        let b = kernel_text("sort_build_keys_u64");
        assert!(b.contains("vk = 0xFFFFFFFFu;"));
        assert!(b.contains("(v == 0.0f) ? 0.0f : v"));
        assert!(b.contains("descending ? (~vk) : vk"));
        assert!(b.contains("keys[out_pos] = 0xFFFFFFFFFFFFFFFFULL;"));
    }
}
