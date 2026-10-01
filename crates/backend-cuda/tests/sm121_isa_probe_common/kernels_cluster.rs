//! cluster（thread block cluster）・分散共有メモリ（DSMEM）のカーネルソース
//! （RULE.txt R-CLU）。
//!
//! cluster 次元は `__cluster_dims__(N,1,1)`（コンパイル時指定）で与え、
//! 起動は cudarc の safe な `launch`（グリッドは cluster 次元の倍数）で行う。
//! 起動時に cluster 次元を与える `cuLaunchKernelEx` 経路は、TMA プローブ
//! （PR-B）と同時に追加する（本 PR では unsafe を増やさない）。

use super::kernels_mma::pre;

/// `clu.dims<N>` のカーネル。各 block の thread 0 が自身の cluster 内 rank と
/// cluster の block 数を out へ書く（out[2*b], out[2*b+1]）。
macro_rules! clu_dims_src {
    ($n:literal, $sym:literal) => {
        concat!(
            pre!(),
            "extern \"C\" __global__ void __cluster_dims__(",
            $n,
            ",1,1) __launch_bounds__(32) ",
            $sym,
            "(\n",
            "    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)\n",
            "{\n",
            "    unsigned rank, nrank;\n",
            "    asm volatile(\"mov.u32 %0, %%cluster_ctarank;\" : \"=r\"(rank));\n",
            "    asm volatile(\"mov.u32 %0, %%cluster_nctarank;\" : \"=r\"(nrank));\n",
            "    if (threadIdx.x == 0) {\n",
            "        ST(2u * blockIdx.x, rank);\n",
            "        ST(2u * blockIdx.x + 1u, nrank);\n",
            "    }\n",
            "}\n",
        )
    };
}

pub const CLU_DIMS1: &str = clu_dims_src!(1, "clu_dims1");
pub const CLU_DIMS2: &str = clu_dims_src!(2, "clu_dims2");
pub const CLU_DIMS4: &str = clu_dims_src!(4, "clu_dims4");
pub const CLU_DIMS8: &str = clu_dims_src!(8, "clu_dims8");
pub const CLU_DIMS16: &str = clu_dims_src!(16, "clu_dims16");

/// DSMEM の往復（cluster 2）。各 block が自身の smem へ `0xA000 + rank` を書き、
/// `barrier.cluster` で同期した後、`mapa.shared::cluster` で隣接 block の smem
/// アドレスを得て `ld.shared::cluster` で読み、out[blockIdx.x] へ書く。
/// 隣接 block が先に終了して smem が無効になるのを防ぐため、読んだ後に
/// もう一度 `barrier.cluster` を通す。`barrier.cluster` のデッドロックは
/// 外部 `timeout` で記録される（orchestrate.sh）。
pub const CLU_DSMEM: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __cluster_dims__(2,1,1) __launch_bounds__(32) clu_dsmem(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    __shared__ unsigned buf[1];
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    if (threadIdx.x == 0) { buf[0] = 0xA000u + rank; }
    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    unsigned local_addr;
    asm volatile("{ .reg .u64 t_; cvta.to.shared.u64 t_, %1; cvt.u32.u64 %0, t_; }" : "=r"(local_addr) : "l"(&buf[0]));
    unsigned peer = rank ^ 1u;
    unsigned remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_addr) : "r"(local_addr), "r"(peer));
    unsigned v;
    asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(remote_addr) : "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (threadIdx.x == 0) { ST(blockIdx.x, v); }
}
"#
);
