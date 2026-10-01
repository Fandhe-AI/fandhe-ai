//! AC3（`cp.async.bulk.tensor` の意味論。RULE.txt R-TMA-BASE／R-TMA-SEM／R-TMA-XFER）の
//! カーネルソース。イシュー #2122 の PR-B。
//!
//! すべて `&'static str` のコンパイル時定数（A03）。tensor map 系のカーネルは
//! `(CUtensorMap tm, unsigned* out, int n, int cx, int cy, unsigned expect_tx,
//! unsigned dump_words)`、tensor を使わない bulk コピーは PR-A と同じ
//! `(in, n_in, out, n)` の ABI。起動は `cuLaunchKernelEx`（`runner.rs` の raw 経路。
//! tensor map を値渡しするため）。出力は境界チェック付きの `ST` で書く（REQ-8）。
//!
//! 共通の出力レイアウト（tensor 系 load・multicast）: スロットごとに
//! `[状態語（0=mbarrier 完了・1=ポーリング上限に到達）, ポーリング回数, smem の生ダンプ…]`。
//! 待ちには上限回数（`TMA_POLL_LIMIT`）と状態語を持たせ、ハングさせない
//! （`tests/tma_probe_real_device.rs` と同じ方針。上限は実測チューニング値ではない）。
//! smem は転送前に番兵（`0xFEEDFACE`）で埋め、書かれなかった語をダンプで判別できるようにする。
//! 期待値はカーネル側に持たず、ホスト側の候補モデル（`model_tma.rs`）と突き合わせる。

use super::kernels_mma::pre;

/// 共通ヘルパ（`CUtensorMap` の typedef は `tests/tma_probe_real_device.rs` と同じ
/// `align(128)`・`cuda-13000` feature 下の `cudarc::driver::sys::CUtensorMap` と一致）。
macro_rules! tma_pre {
    () => {
        concat!(
            pre!(),
            r#"
typedef struct __align__(128) { unsigned long long opaque[16]; } CUtensorMap;
#define TMA_POLL_LIMIT 1000000u
#define TMA_SENTINEL 0xFEEDFACEu
__device__ __forceinline__ unsigned smem_u32(const void* p) {
    unsigned a;
    asm volatile("{ .reg .u64 t_; cvta.to.shared.u64 t_, %1; cvt.u32.u64 %0, t_; }" : "=r"(a) : "l"(p));
    return a;
}
__device__ __forceinline__ unsigned tma_wait(unsigned mb, unsigned* polls_out) {
    unsigned complete = 0u, polls = 0u;
    while (!complete && polls < TMA_POLL_LIMIT) {
        asm volatile("{ .reg .pred p_; mbarrier.try_wait.parity.shared::cta.b64 p_, [%1], 0; selp.u32 %0, 1, 0, p_; }"
            : "=r"(complete) : "r"(mb));
        polls++;
    }
    *polls_out = polls;
    return complete;
}
"#
        )
    };
}

/// tensor map を使う load。`$space` は宛先の状態空間（`shared::cta`／`shared::cluster`）。
macro_rules! tma_load_src {
    ($sym:literal, $space:literal) => {
        concat!(
            tma_pre!(),
            "extern \"C\" __global__ void __launch_bounds__(128) ",
            $sym,
            "(\n",
            "    const __grid_constant__ CUtensorMap tm, unsigned* __restrict__ out, int n,\n",
            "    int cx, int cy, unsigned expect_tx, unsigned dump_words)\n",
            r#"{
    __shared__ __align__(1024) unsigned smem[512];
    __shared__ __align__(8) unsigned long long mbar;
    __shared__ unsigned st_timeout;
    __shared__ unsigned st_polls;
    unsigned tid = threadIdx.x;
    for (unsigned i = tid; i < 512u; i += blockDim.x) { smem[i] = TMA_SENTINEL; }
    __syncthreads();
    if (tid == 0) {
        unsigned mb = smem_u32(&mbar);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb));
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        unsigned sm = smem_u32(smem);
        unsigned long long map = (unsigned long long)&tm;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(expect_tx));
        asm volatile("cp.async.bulk.tensor.2d."#,
            $space,
            r#".global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
            :: "r"(sm), "l"(map), "r"(cx), "r"(cy), "r"(mb) : "memory");
        unsigned polls;
        unsigned c = tma_wait(mb, &polls);
        st_timeout = c ? 0u : 1u;
        st_polls = polls;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    if (tid == 0) { ST(0u, st_timeout); ST(1u, st_polls); }
    for (unsigned i = tid; i < dump_words && i < 512u; i += blockDim.x) { ST(2u + i, smem[i]); }
}
"#
        )
    };
}

pub const TMA_LOAD_CTA: &str = tma_load_src!("tma_load_cta", "shared::cta");
pub const TMA_LOAD_CLUSTER: &str = tma_load_src!("tma_load_cluster", "shared::cluster");

/// tensor map 経由の store（`cp.async.bulk.tensor.2d.global.shared::cta.bulk_group`＋
/// `commit_group`／`wait_group`）。smem を既知パターン（`0xC0DE0000 | 添字`）で埋めて
/// box 1 枚を global へ書き、ホストが global を読み戻して検証する。out[0] は完走の目印。
pub const TMA_STORE_CTA: &str = concat!(
    tma_pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(128) tma_store_cta(
    const __grid_constant__ CUtensorMap tm, unsigned* __restrict__ out, int n,
    int cx, int cy, unsigned expect_tx, unsigned dump_words)
{
    __shared__ __align__(1024) unsigned smem[512];
    unsigned tid = threadIdx.x;
    for (unsigned i = tid; i < 512u; i += blockDim.x) { smem[i] = 0xC0DE0000u | i; }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        unsigned sm = smem_u32(smem);
        unsigned long long map = (unsigned long long)&tm;
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
            :: "l"(map), "r"(cx), "r"(cy), "r"(sm) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        ST(0u, 0x57025E5Eu);
    }
}
"#
);

/// tensor でない bulk コピー（`cp.async.bulk`。global → smem 256 バイト）。宛先の状態空間
/// （`shared::cta`／`shared::cluster`）の 2 変種。出力は `[状態語, ポーリング回数, smem 64 語]`。
macro_rules! tma_bulk_src {
    ($sym:literal, $space:literal) => {
        concat!(
            tma_pre!(),
            "extern \"C\" __global__ void __launch_bounds__(128) ",
            $sym,
            "(\n",
            "    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)\n",
            r#"{
    __shared__ __align__(128) unsigned smem[64];
    __shared__ __align__(8) unsigned long long mbar;
    __shared__ unsigned st_timeout;
    __shared__ unsigned st_polls;
    unsigned tid = threadIdx.x;
    for (unsigned i = tid; i < 64u; i += blockDim.x) { smem[i] = TMA_SENTINEL; }
    __syncthreads();
    if (tid == 0) {
        unsigned mb = smem_u32(&mbar);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb));
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        unsigned sm = smem_u32(smem);
        unsigned long long src = (unsigned long long)in;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 256;" :: "r"(mb));
        asm volatile("cp.async.bulk."#,
            $space,
            r#".global.mbarrier::complete_tx::bytes [%0], [%1], 256, [%2];"
            :: "r"(sm), "l"(src), "r"(mb) : "memory");
        unsigned polls;
        unsigned c = tma_wait(mb, &polls);
        st_timeout = c ? 0u : 1u;
        st_polls = polls;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    if (tid == 0) { ST(0u, st_timeout); ST(1u, st_polls); }
    for (unsigned i = tid; i < 64u; i += blockDim.x) { ST(2u + i, smem[i]); }
}
"#
        )
    };
}

pub const TMA_BULK_CTA: &str = tma_bulk_src!("tma_bulk_cta", "shared::cta");
pub const TMA_BULK_CLUSTER: &str = tma_bulk_src!("tma_bulk_cluster", "shared::cluster");

/// prefetch（`prefetch.tensormap`＋`cp.async.bulk.prefetch.tensor.2d.L2`）。意味論は
/// 観測できないため完走の目印（out[0]）のみ（#2130 §3.3 (c) の前提となる受理・実行の確認）。
pub const TMA_PREFETCH: &str = concat!(
    tma_pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(128) tma_prefetch(
    const __grid_constant__ CUtensorMap tm, unsigned* __restrict__ out, int n,
    int cx, int cy, unsigned expect_tx, unsigned dump_words)
{
    if (threadIdx.x == 0) {
        unsigned long long map = (unsigned long long)&tm;
        asm volatile("prefetch.tensormap [%0];" :: "l"(map) : "memory");
        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];"
            :: "l"(map), "r"(cx), "r"(cy) : "memory");
        ST(0u, 0x9E7F3C11u);
    }
}
"#
);

/// multicast（`.multicast::cluster`。cluster 2・runtime の cluster 次元で起動）。rank 0 の
/// thread 0 が mask 0b11 で発行し、両 CTA の同一 smem オフセットへ転送される。各 CTA が自分の
/// mbarrier を待ち、スロット（`blockIdx.x * (2 + dump_words)`）へ `[状態語, ポーリング回数, ダンプ]`
/// を書く。peer の mbarrier 初期化前に発行しないよう `barrier.cluster` で同期し、peer が先に
/// 終了して smem が無効になるのを防ぐため最後にもう一度同期する。
pub const TMA_MULTICAST: &str = concat!(
    tma_pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(128) tma_multicast(
    const __grid_constant__ CUtensorMap tm, unsigned* __restrict__ out, int n,
    int cx, int cy, unsigned expect_tx, unsigned dump_words)
{
    __shared__ __align__(1024) unsigned smem[512];
    __shared__ __align__(8) unsigned long long mbar;
    __shared__ unsigned st_timeout;
    __shared__ unsigned st_polls;
    unsigned tid = threadIdx.x;
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    unsigned slot = blockIdx.x * (2u + dump_words);
    for (unsigned i = tid; i < 512u; i += blockDim.x) { smem[i] = TMA_SENTINEL; }
    __syncthreads();
    unsigned mb = smem_u32(&mbar);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb));
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (tid == 0) {
        unsigned sm = smem_u32(smem);
        unsigned long long map = (unsigned long long)&tm;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(expect_tx));
        if (rank == 0u) {
            unsigned short mask = 3;
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
                "[%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(sm), "l"(map), "r"(cx), "r"(cy), "r"(mb), "h"(mask) : "memory");
        }
        unsigned polls;
        unsigned c = tma_wait(mb, &polls);
        st_timeout = c ? 0u : 1u;
        st_polls = polls;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    if (tid == 0) { ST(slot, st_timeout); ST(slot + 1u, st_polls); }
    for (unsigned i = tid; i < dump_words && i < 512u; i += blockDim.x) { ST(slot + 2u + i, smem[i]); }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
"#
);

/// runtime で cluster 次元を与える起動（`cuLaunchKernelEx` の `CLUSTER_DIMENSION`）の補助カーネル。
/// `clu.dims*`（`__cluster_dims__` のコンパイル時指定）と同じ出力
/// （各 block の cluster 内 rank と cluster の block 数）を、属性なしのカーネルで観測する。
macro_rules! clu_rt_src {
    ($sym:literal) => {
        concat!(
            pre!(),
            "extern \"C\" __global__ void __launch_bounds__(32) ",
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

pub const CLU_RT2: &str = clu_rt_src!("clu_rt2");
pub const CLU_RT4: &str = clu_rt_src!("clu_rt4");

/// raw 起動経路（`cuLaunchKernelEx`）の健全性を確かめる対照（R-CTL。PR-A の `ctl.copy` と同じ内容を
/// raw 経路で実行する）。tensor map を使わないので sm_80 以上のどのアーキでも動き、開発機でも
/// 起動・同期・読み戻しの機構を検証できる。
pub const CTL_RAW: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(64) ctl_raw(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned i = threadIdx.x;
    ST(i, LD(i));
}
"#
);

/// tensor map 系の引数 ABI（`CUtensorMap` 値渡し＝128 バイト整列・後続引数のオフセット）の対照。
/// TMA 命令は使わず、後続引数の値と、tensor map がゼロ初期化でない（encode 済み）ことだけを
/// out へ書く（`[cx, cy, expect_tx, dump_words, tm が非ゼロなら 1]`）。TMA 非対応のアーキ
/// （開発機の sm_86 を含む）でも動くため、raw 起動の引数渡しを開発機で検証できる。
pub const CTL_RAWMAP: &str = concat!(
    tma_pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) ctl_rawmap(
    const __grid_constant__ CUtensorMap tm, unsigned* __restrict__ out, int n,
    int cx, int cy, unsigned expect_tx, unsigned dump_words)
{
    if (threadIdx.x == 0) {
        unsigned long long any = 0ull;
        for (int i = 0; i < 16; ++i) { any |= tm.opaque[i]; }
        ST(0u, (unsigned)cx); ST(1u, (unsigned)cy); ST(2u, expect_tx); ST(3u, dump_words);
        ST(4u, any != 0ull ? 1u : 0u);
    }
}
"#
);
