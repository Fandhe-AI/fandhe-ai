//! unique 用ビットニックソートの HIP カーネルソース（`backend-cuda::kernels_unique` の移植。
//! イシュー #3439・分解案 R8）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_unique_source`]）。呼び出し元は `unique.rs`。
//! ホストが `f32` を全順序 `u32` キー（`total_order_key`）へ変換し 2 のべき乗長へ `u32::MAX` で
//! パディングして渡し、デバイスは整数 compare/swap のみで昇順整列する（浮動小数点演算なし・
//! NaN の payload もキーに保存されるため bit 保存）。`j`・`k` のループはホスト側。
//!
//! CUDA 版との差分: スレッド添字を `int` ではなく `long long` で計算する（grid が `padded` を
//! 256 の倍数へ切り上げるため `int` だと `i32::MAX` 近傍でスレッド添字が溢れうる）。引数 `j`・`k`・`n`
//! も `long long`。REQ-8: `i >= n` と `ixj <= i || ixj >= n` で早期リターン。warp 幅依存の命令は使わない。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};

/// `(keys, j, k, n)`。
const BITONIC_U32_ARGS: [ArgKind; 4] = [ArgKind::Ptr, ArgKind::I64, ArgKind::I64, ArgKind::I64];

/// ビットニックステップ（u32）の spec。
pub(crate) const BITONIC_STEP_U32_SPEC: KernelSpec = KernelSpec {
    name: "bitonic_step_u32",
    args: &BITONIC_U32_ARGS,
};

/// 静的テンプレート本体。`BLOCK_DIM` は前置する `#define` で与える。
pub(crate) const UNIQUE_BODY: &str = r#"
extern "C" __global__ __launch_bounds__(BLOCK_DIM) void bitonic_step_u32(
    unsigned int* keys,
    long long j,
    long long k,
    long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    long long ixj = i ^ j;
    if (ixj <= i || ixj >= n) {
        return;
    }
    unsigned int a = keys[i];
    unsigned int b = keys[ixj];
    bool ascending = (i & k) == 0;
    bool should_swap = ascending ? (a > b) : (a < b);
    if (should_swap) {
        keys[i] = b;
        keys[ixj] = a;
    }
}
"#;

/// unique 用ソースを生成する（`BLOCK_DIM` のみ前置）。
pub(crate) fn render_unique_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(&[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], UNIQUE_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    #[test]
    fn spec_matches_source_signature() {
        assert_eq!(
            kernel_param_kinds(UNIQUE_BODY, BITONIC_STEP_U32_SPEC.name),
            BITONIC_STEP_U32_SPEC.args
        );
    }

    #[test]
    fn source_defines_only_block_dim_and_is_deterministic() {
        let s = render_unique_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define BLOCK_DIM 256"]);
        assert_eq!(s, render_unique_source().unwrap());
    }

    #[test]
    fn guards_use_long_long_thread_index() {
        assert!(UNIQUE_BODY.contains("long long i = (long long)blockIdx.x"));
        assert!(UNIQUE_BODY.contains("if (i >= n)"));
        assert!(UNIQUE_BODY.contains("ixj <= i || ixj >= n"));
        assert!(!UNIQUE_BODY.contains("int i ="));
    }

    #[test]
    fn no_float_and_no_warp_dependence() {
        for banned in ["float", "double", "__shfl", "warpSize", "WARP_", "__ballot"] {
            assert!(!UNIQUE_BODY.contains(banned), "{banned}");
        }
    }
}
