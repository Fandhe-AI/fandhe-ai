//! アーキ固有機能のカーネルソース（AC1 tcgen05・macro、AC5 wgmma・setmaxnreg・
//! Hopper 由来命令、対照 `ctl.copy`）。
//!
//! すべて `&'static str` のコンパイル時定数（A03）。ABI は `kernels_mma.rs` と
//! 同じ `(in, n_in, out, n)`。PTX ISA の対象アーキ要件は記憶で書かず、
//! 証拠は S2 の陽性対照（R-HOME）に持たせる。

use super::kernels_mma::pre;

/// 対照カーネル。全プロセスが最初に同じ target で通し、toolchain・ドライバ・
/// 実行基盤の健全性を確認する（R-CTL）。
pub const CTL_COPY: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(64) ctl_copy(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned i = threadIdx.x;
    ST(i, LD(i));
}
"#
);

/// `macro.arch` が記録する候補マクロ名（値の記録順。カーネルの `v[]` の添字と
/// 一致させる）。存在は「記録のみ」で、registry テストはマクロ名の存在を
/// 主張しない（名前を記憶で断定しない。CUDA 13.0 文書での確認は実機実測側）。
pub const MACRO_CANDIDATES: [&str; 11] = [
    "__CUDA_ARCH__",
    "__CUDA_ARCH_SPECIFIC__",
    "__CUDA_ARCH_FAMILY_SPECIFIC__",
    "__CUDA_ARCH_FEAT_SM121_ALL",
    "__CUDA_ARCH_FEAT_SM120_ALL",
    "__CUDA_ARCH_FEAT_SM100_ALL",
    "__CUDA_ARCH_FEAT_SM90_ALL",
    "__CUDACC_RTC__",
    "__CUDACC_VER_MAJOR__",
    "__CUDACC_VER_MINOR__",
    "__NVCC__",
];

/// 候補マクロの値（`__CUDA_ARCH__`・`__CUDACC_VER_*`）または存在（1/0）を
/// out へ書く。存在判定は `#if defined(...)` のみで値を前提にしない。
pub const MACRO_ARCH: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) macro_arch(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    if (threadIdx.x != 0) { return; }
#if defined(__CUDA_ARCH__)
    ST(0, (unsigned)__CUDA_ARCH__);
#else
    ST(0, 0u);
#endif
#if defined(__CUDA_ARCH_SPECIFIC__)
    ST(1, 1u);
#else
    ST(1, 0u);
#endif
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
    ST(2, 1u);
#else
    ST(2, 0u);
#endif
#if defined(__CUDA_ARCH_FEAT_SM121_ALL)
    ST(3, 1u);
#else
    ST(3, 0u);
#endif
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
    ST(4, 1u);
#else
    ST(4, 0u);
#endif
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
    ST(5, 1u);
#else
    ST(5, 0u);
#endif
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    ST(6, 1u);
#else
    ST(6, 0u);
#endif
#if defined(__CUDACC_RTC__)
    ST(7, 1u);
#else
    ST(7, 0u);
#endif
#if defined(__CUDACC_VER_MAJOR__)
    ST(8, (unsigned)__CUDACC_VER_MAJOR__);
#else
    ST(8, 0u);
#endif
#if defined(__CUDACC_VER_MINOR__)
    ST(9, (unsigned)__CUDACC_VER_MINOR__);
#else
    ST(9, 0u);
#endif
#if defined(__NVCC__)
    ST(10, 1u);
#else
    ST(10, 0u);
#endif
}
"#
);

/// tcgen05 の TMEM 割り当て・解放（1 warp）。TMEM アドレスを smem 経由で
/// out[0] へ、完走の目印 [`TC5_MAGIC`] を out[1] へ書く（dealloc 完了後）。
/// 対応アーキは `sm_100a` 想定（S2 の陽性対照 R-HOME で実測する）。
pub const TC5_ALLOC: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) tc5_alloc(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    __shared__ unsigned taddr_smem;
    unsigned sa;
    asm volatile("{ .reg .u64 t_; cvta.to.shared.u64 t_, %1; cvt.u32.u64 %0, t_; }" : "=r"(sa) : "l"(&taddr_smem));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" :: "r"(sa) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    __syncwarp();
    unsigned taddr = taddr_smem;
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" :: "r"(taddr) : "memory");
    if (threadIdx.x == 0) {
        ST(0, taddr);
        ST(1, 0x7C5A110Cu);
    }
}
"#
);

/// [`TC5_ALLOC`] が完走したときに out[1] へ書く目印。
pub const TC5_MAGIC: u32 = 0x7C5A_110C;

/// tcgen05 の TMEM ロード（`tcgen05.ld`＋`tcgen05.wait::ld`）。初期化前の
/// TMEM を読むため値は不定で、記録のみ（out[0]）。完走の目印は out[1]。
pub const TC5_LD: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) tc5_ld(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    __shared__ unsigned taddr_smem;
    unsigned sa;
    asm volatile("{ .reg .u64 t_; cvta.to.shared.u64 t_, %1; cvt.u32.u64 %0, t_; }" : "=r"(sa) : "l"(&taddr_smem));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" :: "r"(sa) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    __syncwarp();
    unsigned taddr = taddr_smem;
    unsigned v;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(v) : "r"(taddr) : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" :: "r"(taddr) : "memory");
    if (threadIdx.x == 0) {
        ST(0, v);
        ST(1, 0x7C5A110Cu);
    }
}
"#
);

/// `wgmma.m64n8k16`（f16 入力・f32 累積。`wgmma.fence`／`mma_async`／
/// `commit_group`／`wait_group`）。受理段のみ（accept_only）で起動しない。
/// smem 記述子は `in` 由来のダミー値で、実行意味論は検証対象外。
pub const WGMMA_M64N8K16: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(128) wgmma_m64n8k16(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned long long da = ((unsigned long long)LD(1u) << 32) | (unsigned long long)LD(0u);
    unsigned long long db = ((unsigned long long)LD(3u) << 32) | (unsigned long long)LD(2u);
    unsigned scale_d = LD(4u);
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %6, 0; "
                 "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0,%1,%2,%3}, %4, %5, p_, 1, 1, 0, 0; }"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(da), "l"(db), "r"(scale_d));
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    unsigned i = threadIdx.x;
    ST(i * 4u, __float_as_uint(d0)); ST(i * 4u + 1u, __float_as_uint(d1));
    ST(i * 4u + 2u, __float_as_uint(d2)); ST(i * 4u + 3u, __float_as_uint(d3));
}
"#
);

/// `setmaxnreg.dec` のみ（1 warpgroup）。既存 `tests/setmaxnreg_common` の
/// カーネルと同じ命令列（R-LEGACY で結論の整合を突き合わせる）。
pub const SNR_DEC: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(128) snr_dec(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned i = threadIdx.x;
    asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    ST(i, LD(i) + 1u);
}
"#
);

/// producer（`setmaxnreg.dec`）／consumer（`setmaxnreg.inc`）非対称版
/// （2 warpgroup）。既存 `setmaxnreg_common` と同じ値（24／232）。
pub const SNR_INCDEC: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(256) snr_incdec(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned i = threadIdx.x;
    if ((i / 128u) == 0u) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
    }
    ST(i, LD(i) * 2u);
}
"#
);

/// `griddepcontrol`（Hopper 由来。PDL 属性なしの通常起動では実質 no-op）。
pub const HOP_GRIDDEPCONTROL: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(64) hop_griddepcontrol(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned i = threadIdx.x;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    ST(i, LD(i) + 1u);
}
"#
);

/// `fence.proxy.async`（Hopper 由来）。
pub const HOP_FENCE_PROXY_ASYNC: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(64) hop_fence_proxy_async(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned i = threadIdx.x;
    asm volatile("fence.proxy.async;" ::: "memory");
    ST(i, LD(i) + 1u);
}
"#
);
