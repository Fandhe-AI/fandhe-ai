//! AC4（mma 形状の一覧。RULE.txt R-MMA）のカーネルソース: Tensor Core の
//! `mma.sync`・`ldmatrix`／`stmatrix` と SIMT 命令。
//!
//! すべて `&'static str` のコンパイル時定数（外部入力・環境変数を連結しない。
//! `nvrtc.rs` の A03 契約を踏襲）。`#include` を使わず、f16／bf16 の operand は
//! ホストが `u32` のビット列へ詰めて `in` で渡し、カーネルは生のレジスタとして
//! `mma` へ渡す（ASCII のみ・ヘッダ不要）。
//!
//! カーネル ABI は全プローブ共通: `(const unsigned* in, int n_in, unsigned* out, int n)`。
//! 読み書きは `LD`／`ST` マクロで必ず境界チェックする（REQ-8。
//! `.claude/rules/coding-rust.md`「カーネル実装の境界検査」）。
//!
//! fragment レイアウトの出典は PTX ISA の warp-level MMA 章（9.7.15。個別の
//! 小節番号は要確認）。ホスト側の参照モデルは `model.rs`。開発機（sm_86）で
//! 実行できる形状は RTX 3060 でビット一致を確認し、確認できない形状は
//! レジストリの `layout=unverified` として RULE.txt に事前登録する。

/// 共通プリセット（`LD`／`ST`／`DBL`）。各カーネル先頭へ連結する。
macro_rules! pre {
    () => {
        concat!(
            "#define LD(i) (((unsigned)(i) < (unsigned)n_in) ? in[(i)] : 0u)\n",
            "#define ST(i, v) do { unsigned k_ = (unsigned)(i); if (k_ < (unsigned)n) { out[k_] = (v); } } while (0)\n",
            "#define ST_F64(i, v) do { unsigned long long u_ = (unsigned long long)__double_as_longlong(v); ST((i), (unsigned)u_); ST((i) + 1u, (unsigned)(u_ >> 32)); } while (0)\n",
            "#define DBL(i) __longlong_as_double((long long)(((unsigned long long)LD(2u * (i) + 1u) << 32) | (unsigned long long)LD(2u * (i))))\n",
        )
    };
}
pub(crate) use pre;

pub const MMA_TF32_M16N8K8: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_tf32_m16n8k8(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned b = l * 10u;
    unsigned a0 = LD(b), a1 = LD(b + 1u), a2 = LD(b + 2u), a3 = LD(b + 3u);
    unsigned b0 = LD(b + 4u), b1 = LD(b + 5u);
    float c0 = __uint_as_float(LD(b + 6u)), c1 = __uint_as_float(LD(b + 7u));
    float c2 = __uint_as_float(LD(b + 8u)), c3 = __uint_as_float(LD(b + 9u));
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    ST(l * 4u, __float_as_uint(d0)); ST(l * 4u + 1u, __float_as_uint(d1));
    ST(l * 4u + 2u, __float_as_uint(d2)); ST(l * 4u + 3u, __float_as_uint(d3));
}
"#
);

pub const MMA_TF32_M16N8K4: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_tf32_m16n8k4(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned b = l * 7u;
    unsigned a0 = LD(b), a1 = LD(b + 1u);
    unsigned b0 = LD(b + 2u);
    float c0 = __uint_as_float(LD(b + 3u)), c1 = __uint_as_float(LD(b + 4u));
    float c2 = __uint_as_float(LD(b + 5u)), c3 = __uint_as_float(LD(b + 6u));
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    ST(l * 4u, __float_as_uint(d0)); ST(l * 4u + 1u, __float_as_uint(d1));
    ST(l * 4u + 2u, __float_as_uint(d2)); ST(l * 4u + 3u, __float_as_uint(d3));
}
"#
);

pub const MMA_F16_M16N8K16_F32: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_f16_m16n8k16_f32(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned b = l * 10u;
    unsigned a0 = LD(b), a1 = LD(b + 1u), a2 = LD(b + 2u), a3 = LD(b + 3u);
    unsigned b0 = LD(b + 4u), b1 = LD(b + 5u);
    float c0 = __uint_as_float(LD(b + 6u)), c1 = __uint_as_float(LD(b + 7u));
    float c2 = __uint_as_float(LD(b + 8u)), c3 = __uint_as_float(LD(b + 9u));
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    ST(l * 4u, __float_as_uint(d0)); ST(l * 4u + 1u, __float_as_uint(d1));
    ST(l * 4u + 2u, __float_as_uint(d2)); ST(l * 4u + 3u, __float_as_uint(d3));
}
"#
);

pub const MMA_F16_M16N8K8_F32: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_f16_m16n8k8_f32(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned b = l * 7u;
    unsigned a0 = LD(b), a1 = LD(b + 1u);
    unsigned b0 = LD(b + 2u);
    float c0 = __uint_as_float(LD(b + 3u)), c1 = __uint_as_float(LD(b + 4u));
    float c2 = __uint_as_float(LD(b + 5u)), c3 = __uint_as_float(LD(b + 6u));
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    ST(l * 4u, __float_as_uint(d0)); ST(l * 4u + 1u, __float_as_uint(d1));
    ST(l * 4u + 2u, __float_as_uint(d2)); ST(l * 4u + 3u, __float_as_uint(d3));
}
"#
);

pub const MMA_F16_M16N8K16_F16: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_f16_m16n8k16_f16(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned b = l * 8u;
    unsigned a0 = LD(b), a1 = LD(b + 1u), a2 = LD(b + 2u), a3 = LD(b + 3u);
    unsigned b0 = LD(b + 4u), b1 = LD(b + 5u);
    unsigned c0 = LD(b + 6u), c1 = LD(b + 7u);
    unsigned d0, d1;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
    ST(l * 2u, d0); ST(l * 2u + 1u, d1);
}
"#
);

pub const MMA_BF16_M16N8K16_F32: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_bf16_m16n8k16_f32(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned b = l * 10u;
    unsigned a0 = LD(b), a1 = LD(b + 1u), a2 = LD(b + 2u), a3 = LD(b + 3u);
    unsigned b0 = LD(b + 4u), b1 = LD(b + 5u);
    float c0 = __uint_as_float(LD(b + 6u)), c1 = __uint_as_float(LD(b + 7u));
    float c2 = __uint_as_float(LD(b + 8u)), c3 = __uint_as_float(LD(b + 9u));
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    ST(l * 4u, __float_as_uint(d0)); ST(l * 4u + 1u, __float_as_uint(d1));
    ST(l * 4u + 2u, __float_as_uint(d2)); ST(l * 4u + 3u, __float_as_uint(d3));
}
"#
);

pub const MMA_BF16_M16N8K8_F32: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_bf16_m16n8k8_f32(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned b = l * 7u;
    unsigned a0 = LD(b), a1 = LD(b + 1u);
    unsigned b0 = LD(b + 2u);
    float c0 = __uint_as_float(LD(b + 3u)), c1 = __uint_as_float(LD(b + 4u));
    float c2 = __uint_as_float(LD(b + 5u)), c3 = __uint_as_float(LD(b + 6u));
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    ST(l * 4u, __float_as_uint(d0)); ST(l * 4u + 1u, __float_as_uint(d1));
    ST(l * 4u + 2u, __float_as_uint(d2)); ST(l * 4u + 3u, __float_as_uint(d3));
}
"#
);

pub const MMA_F64_M8N8K4: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_f64_m8n8k4(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    double a0 = DBL(l * 4u), b0 = DBL(l * 4u + 1u);
    double c0 = DBL(l * 4u + 2u), c1 = DBL(l * 4u + 3u);
    double d0, d1;
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};"
        : "=d"(d0), "=d"(d1) : "d"(a0), "d"(b0), "d"(c0), "d"(c1));
    ST_F64(l * 4u, d0);
    ST_F64(l * 4u + 2u, d1);
}
"#
);

/// 旧世代（sm_70）形の `m8n8k4 .f16`。quad-pair 単位の特殊レイアウトで、
/// 参照モデルを誤りなく記述する根拠が手元にないため受理段のみ（accept_only）。
pub const MMA_F16_M8N8K4: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_f16_m8n8k4(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned b = l * 12u;
    unsigned a0 = LD(b), a1 = LD(b + 1u), b0 = LD(b + 2u), b1 = LD(b + 3u);
    float c0 = __uint_as_float(LD(b + 4u)), c1 = __uint_as_float(LD(b + 5u));
    float c2 = __uint_as_float(LD(b + 6u)), c3 = __uint_as_float(LD(b + 7u));
    float c4 = __uint_as_float(LD(b + 8u)), c5 = __uint_as_float(LD(b + 9u));
    float c6 = __uint_as_float(LD(b + 10u)), c7 = __uint_as_float(LD(b + 11u));
    float d0, d1, d2, d3, d4, d5, d6, d7;
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
    ST(l * 8u, __float_as_uint(d0)); ST(l * 8u + 1u, __float_as_uint(d1));
    ST(l * 8u + 2u, __float_as_uint(d2)); ST(l * 8u + 3u, __float_as_uint(d3));
    ST(l * 8u + 4u, __float_as_uint(d4)); ST(l * 8u + 5u, __float_as_uint(d5));
    ST(l * 8u + 6u, __float_as_uint(d6)); ST(l * 8u + 7u, __float_as_uint(d7));
}
"#
);

pub const MMA_F64_M16N8K4: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_f64_m16n8k4(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    double a0 = DBL(l * 7u), a1 = DBL(l * 7u + 1u), b0 = DBL(l * 7u + 2u);
    double c0 = DBL(l * 7u + 3u), c1 = DBL(l * 7u + 4u), c2 = DBL(l * 7u + 5u), c3 = DBL(l * 7u + 6u);
    double d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=d"(d0), "=d"(d1), "=d"(d2), "=d"(d3)
        : "d"(a0), "d"(a1), "d"(b0), "d"(c0), "d"(c1), "d"(c2), "d"(c3));
    ST_F64(l * 8u, d0); ST_F64(l * 8u + 2u, d1);
    ST_F64(l * 8u + 4u, d2); ST_F64(l * 8u + 6u, d3);
}
"#
);

pub const MMA_F64_M16N8K8: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_f64_m16n8k8(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    double a0 = DBL(l * 10u), a1 = DBL(l * 10u + 1u), a2 = DBL(l * 10u + 2u), a3 = DBL(l * 10u + 3u);
    double b0 = DBL(l * 10u + 4u), b1 = DBL(l * 10u + 5u);
    double c0 = DBL(l * 10u + 6u), c1 = DBL(l * 10u + 7u), c2 = DBL(l * 10u + 8u), c3 = DBL(l * 10u + 9u);
    double d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=d"(d0), "=d"(d1), "=d"(d2), "=d"(d3)
        : "d"(a0), "d"(a1), "d"(a2), "d"(a3), "d"(b0), "d"(b1), "d"(c0), "d"(c1), "d"(c2), "d"(c3));
    ST_F64(l * 8u, d0); ST_F64(l * 8u + 2u, d1);
    ST_F64(l * 8u + 4u, d2); ST_F64(l * 8u + 6u, d3);
}
"#
);

pub const MMA_F64_M16N8K16: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_f64_m16n8k16(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    double a0 = DBL(l * 16u), a1 = DBL(l * 16u + 1u), a2 = DBL(l * 16u + 2u), a3 = DBL(l * 16u + 3u);
    double a4 = DBL(l * 16u + 4u), a5 = DBL(l * 16u + 5u), a6 = DBL(l * 16u + 6u), a7 = DBL(l * 16u + 7u);
    double b0 = DBL(l * 16u + 8u), b1 = DBL(l * 16u + 9u), b2 = DBL(l * 16u + 10u), b3 = DBL(l * 16u + 11u);
    double c0 = DBL(l * 16u + 12u), c1 = DBL(l * 16u + 13u), c2 = DBL(l * 16u + 14u), c3 = DBL(l * 16u + 15u);
    double d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64 {%0,%1,%2,%3}, {%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15}, {%16,%17,%18,%19};"
        : "=d"(d0), "=d"(d1), "=d"(d2), "=d"(d3)
        : "d"(a0), "d"(a1), "d"(a2), "d"(a3), "d"(a4), "d"(a5), "d"(a6), "d"(a7),
          "d"(b0), "d"(b1), "d"(b2), "d"(b3), "d"(c0), "d"(c1), "d"(c2), "d"(c3));
    ST_F64(l * 8u, d0); ST_F64(l * 8u + 2u, d1);
    ST_F64(l * 8u + 4u, d2); ST_F64(l * 8u + 6u, d3);
}
"#
);

/// INT8 `m16n8k32 .s8`（受理段のみ。実行・数値は int8 プローブ
/// `docs/int8-quant-grade-up-verification-plan.md` に委ねる。計画 8.1）。
pub const MMA_S8_M16N8K32: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_s8_m16n8k32(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned b = l * 10u;
    unsigned a0 = LD(b), a1 = LD(b + 1u), a2 = LD(b + 2u), a3 = LD(b + 3u);
    unsigned b0 = LD(b + 4u), b1 = LD(b + 5u);
    int c0 = (int)LD(b + 6u), c1 = (int)LD(b + 7u), c2 = (int)LD(b + 8u), c3 = (int)LD(b + 9u);
    int d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    ST(l * 4u, (unsigned)d0); ST(l * 4u + 1u, (unsigned)d1);
    ST(l * 4u + 2u, (unsigned)d2); ST(l * 4u + 3u, (unsigned)d3);
}
"#
);

/// FP8 `m16n8k32 .e4m3`（受理段のみ。計画 8.1）。
pub const MMA_E4M3_M16N8K32: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_e4m3_m16n8k32(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned b = l * 10u;
    unsigned a0 = LD(b), a1 = LD(b + 1u), a2 = LD(b + 2u), a3 = LD(b + 3u);
    unsigned b0 = LD(b + 4u), b1 = LD(b + 5u);
    float c0 = __uint_as_float(LD(b + 6u)), c1 = __uint_as_float(LD(b + 7u));
    float c2 = __uint_as_float(LD(b + 8u)), c3 = __uint_as_float(LD(b + 9u));
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    ST(l * 4u, __float_as_uint(d0)); ST(l * 4u + 1u, __float_as_uint(d1));
    ST(l * 4u + 2u, __float_as_uint(d2)); ST(l * 4u + 3u, __float_as_uint(d3));
}
"#
);

/// FP8 `m16n8k32 .e5m2`（受理段のみ。計画 8.1）。
pub const MMA_E5M2_M16N8K32: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_e5m2_m16n8k32(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned b = l * 10u;
    unsigned a0 = LD(b), a1 = LD(b + 1u), a2 = LD(b + 2u), a3 = LD(b + 3u);
    unsigned b0 = LD(b + 4u), b1 = LD(b + 5u);
    float c0 = __uint_as_float(LD(b + 6u)), c1 = __uint_as_float(LD(b + 7u));
    float c2 = __uint_as_float(LD(b + 8u)), c3 = __uint_as_float(LD(b + 9u));
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    ST(l * 4u, __float_as_uint(d0)); ST(l * 4u + 1u, __float_as_uint(d1));
    ST(l * 4u + 2u, __float_as_uint(d2)); ST(l * 4u + 3u, __float_as_uint(d3));
}
"#
);

/// `kind::f8f6f4` 形（受理段のみ。計画 8.1・8.5）。
pub const MMA_F8F6F4_M16N8K32: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_f8f6f4_m16n8k32(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned b = l * 10u;
    unsigned a0 = LD(b), a1 = LD(b + 1u), a2 = LD(b + 2u), a3 = LD(b + 3u);
    unsigned b0 = LD(b + 4u), b1 = LD(b + 5u);
    float c0 = __uint_as_float(LD(b + 6u)), c1 = __uint_as_float(LD(b + 7u));
    float c2 = __uint_as_float(LD(b + 8u)), c3 = __uint_as_float(LD(b + 9u));
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    ST(l * 4u, __float_as_uint(d0)); ST(l * 4u + 1u, __float_as_uint(d1));
    ST(l * 4u + 2u, __float_as_uint(d2)); ST(l * 4u + 3u, __float_as_uint(d3));
}
"#
);

/// block_scale 形（`kind::mxf4`・`scale_vec::2X`。受理段のみ。計画 8.5）。
pub const MMA_BLOCK_SCALE_M16N8K64: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_block_scale_m16n8k64(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned b = l * 12u;
    unsigned a0 = LD(b), a1 = LD(b + 1u), a2 = LD(b + 2u), a3 = LD(b + 3u);
    unsigned b0 = LD(b + 4u), b1 = LD(b + 5u);
    float c0 = __uint_as_float(LD(b + 6u)), c1 = __uint_as_float(LD(b + 7u));
    float c2 = __uint_as_float(LD(b + 8u)), c3 = __uint_as_float(LD(b + 9u));
    unsigned sa = LD(b + 10u), sb = LD(b + 11u);
    unsigned short bid = 0, tid = 0;
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {%16, %17}, %15, {%16, %17};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sa), "r"(sb), "h"(bid), "h"(tid));
    ST(l * 4u, __float_as_uint(d0)); ST(l * 4u + 1u, __float_as_uint(d1));
    ST(l * 4u + 2u, __float_as_uint(d2)); ST(l * 4u + 3u, __float_as_uint(d3));
}
"#
);

/// `ldmatrix` の 4 変種。smem（8x8 b16 行列を最大 4 枚連続配置）をホストの
/// 入力で埋め、行アドレスを lane*16 バイトで与える（x1 は lane%8・x2 は lane%16）。
pub const MMA_LDMATRIX_X1: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_ldmatrix_x1(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    __shared__ __align__(16) unsigned sm[128];
    unsigned l = threadIdx.x;
    for (unsigned i = l; i < 128u; i += 32u) { sm[i] = LD(i); }
    __syncwarp();
    unsigned base;
    asm volatile("{ .reg .u64 t_; cvta.to.shared.u64 t_, %1; cvt.u32.u64 %0, t_; }" : "=r"(base) : "l"(&sm[0]));
    unsigned addr = base + (l & 7u) * 16u;
    unsigned r0;
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r0) : "r"(addr) : "memory");
    ST(l * 4u, r0); ST(l * 4u + 1u, 0u); ST(l * 4u + 2u, 0u); ST(l * 4u + 3u, 0u);
}
"#
);

pub const MMA_LDMATRIX_X2: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_ldmatrix_x2(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    __shared__ __align__(16) unsigned sm[128];
    unsigned l = threadIdx.x;
    for (unsigned i = l; i < 128u; i += 32u) { sm[i] = LD(i); }
    __syncwarp();
    unsigned base;
    asm volatile("{ .reg .u64 t_; cvta.to.shared.u64 t_, %1; cvt.u32.u64 %0, t_; }" : "=r"(base) : "l"(&sm[0]));
    unsigned addr = base + (l & 15u) * 16u;
    unsigned r0, r1;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr) : "memory");
    ST(l * 4u, r0); ST(l * 4u + 1u, r1); ST(l * 4u + 2u, 0u); ST(l * 4u + 3u, 0u);
}
"#
);

pub const MMA_LDMATRIX_X4: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_ldmatrix_x4(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    __shared__ __align__(16) unsigned sm[128];
    unsigned l = threadIdx.x;
    for (unsigned i = l; i < 128u; i += 32u) { sm[i] = LD(i); }
    __syncwarp();
    unsigned base;
    asm volatile("{ .reg .u64 t_; cvta.to.shared.u64 t_, %1; cvt.u32.u64 %0, t_; }" : "=r"(base) : "l"(&sm[0]));
    unsigned addr = base + l * 16u;
    unsigned r0, r1, r2, r3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
    ST(l * 4u, r0); ST(l * 4u + 1u, r1); ST(l * 4u + 2u, r2); ST(l * 4u + 3u, r3);
}
"#
);

pub const MMA_LDMATRIX_X4_TRANS: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_ldmatrix_x4_trans(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    __shared__ __align__(16) unsigned sm[128];
    unsigned l = threadIdx.x;
    for (unsigned i = l; i < 128u; i += 32u) { sm[i] = LD(i); }
    __syncwarp();
    unsigned base;
    asm volatile("{ .reg .u64 t_; cvta.to.shared.u64 t_, %1; cvt.u32.u64 %0, t_; }" : "=r"(base) : "l"(&sm[0]));
    unsigned addr = base + l * 16u;
    unsigned r0, r1, r2, r3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
    ST(l * 4u, r0); ST(l * 4u + 1u, r1); ST(l * 4u + 2u, r2); ST(l * 4u + 3u, r3);
}
"#
);

/// `stmatrix x4`（sm_90 以降。開発機で実行できないためレイアウトは未検証）。
pub const MMA_STMATRIX_X4: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_stmatrix_x4(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    __shared__ __align__(16) unsigned sm[128];
    unsigned l = threadIdx.x;
    for (unsigned i = l; i < 128u; i += 32u) { sm[i] = 0u; }
    __syncwarp();
    unsigned base;
    asm volatile("{ .reg .u64 t_; cvta.to.shared.u64 t_, %1; cvt.u32.u64 %0, t_; }" : "=r"(base) : "l"(&sm[0]));
    unsigned addr = base + l * 16u;
    unsigned r0 = LD(l * 4u), r1 = LD(l * 4u + 1u), r2 = LD(l * 4u + 2u), r3 = LD(l * 4u + 3u);
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
        :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    __syncwarp();
    for (unsigned i = l; i < 128u; i += 32u) { ST(i, sm[i]); }
}
"#
);

// ---------------------------------------------------------------- SIMT

pub const SIMT_FMA_F32: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) simt_fma_f32(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    float a = __uint_as_float(LD(l * 3u)), b = __uint_as_float(LD(l * 3u + 1u)), c = __uint_as_float(LD(l * 3u + 2u));
    float d;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    ST(l, __float_as_uint(d));
}
"#
);

pub const SIMT_FMA_F64: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) simt_fma_f64(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    double a = DBL(l * 3u), b = DBL(l * 3u + 1u), c = DBL(l * 3u + 2u);
    double d;
    asm volatile("fma.rn.f64 %0, %1, %2, %3;" : "=d"(d) : "d"(a), "d"(b), "d"(c));
    unsigned long long u = (unsigned long long)__double_as_longlong(d);
    ST(l * 2u, (unsigned)u); ST(l * 2u + 1u, (unsigned)(u >> 32));
}
"#
);

pub const SIMT_FMA_F16X2: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) simt_fma_f16x2(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned a = LD(l * 3u), b = LD(l * 3u + 1u), c = LD(l * 3u + 2u);
    unsigned d;
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    ST(l, d);
}
"#
);

pub const SIMT_FMA_BF16X2: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) simt_fma_bf16x2(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned a = LD(l * 3u), b = LD(l * 3u + 1u), c = LD(l * 3u + 2u);
    unsigned d;
    asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    ST(l, d);
}
"#
);

pub const SIMT_F32X2_ADD: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) simt_f32x2_add(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned long long a = ((unsigned long long)LD(l * 4u + 1u) << 32) | (unsigned long long)LD(l * 4u);
    unsigned long long b = ((unsigned long long)LD(l * 4u + 3u) << 32) | (unsigned long long)LD(l * 4u + 2u);
    unsigned long long d;
    asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
    ST(l * 2u, (unsigned)d); ST(l * 2u + 1u, (unsigned)(d >> 32));
}
"#
);

pub const SIMT_F32X2_MUL: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) simt_f32x2_mul(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned long long a = ((unsigned long long)LD(l * 4u + 1u) << 32) | (unsigned long long)LD(l * 4u);
    unsigned long long b = ((unsigned long long)LD(l * 4u + 3u) << 32) | (unsigned long long)LD(l * 4u + 2u);
    unsigned long long d;
    asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
    ST(l * 2u, (unsigned)d); ST(l * 2u + 1u, (unsigned)(d >> 32));
}
"#
);

pub const SIMT_F32X2_FMA: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) simt_f32x2_fma(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned long long a = ((unsigned long long)LD(l * 6u + 1u) << 32) | (unsigned long long)LD(l * 6u);
    unsigned long long b = ((unsigned long long)LD(l * 6u + 3u) << 32) | (unsigned long long)LD(l * 6u + 2u);
    unsigned long long c = ((unsigned long long)LD(l * 6u + 5u) << 32) | (unsigned long long)LD(l * 6u + 4u);
    unsigned long long d;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
    ST(l * 2u, (unsigned)d); ST(l * 2u + 1u, (unsigned)(d >> 32));
}
"#
);

pub const SIMT_CVT_TF32: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) simt_cvt_tf32(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    float a = __uint_as_float(LD(l));
    unsigned d;
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(d) : "f"(a));
    ST(l, d);
}
"#
);

/// `elect.sync`（sm_90 以降）。どの lane が選ばれるかは仕様上未規定のため、
/// 検証は「ちょうど 1 lane が選ばれる」ことのみ（`registry.rs` の check）。
pub const SIMT_ELECT_SYNC: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) simt_elect_sync(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned pred;
    asm volatile("{ .reg .pred p_; elect.sync _|p_, 0xffffffff; selp.u32 %0, 1, 0, p_; }" : "=r"(pred));
    ST(l, pred);
}
"#
);

pub const SIMT_REDUX_U32: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) simt_redux_u32(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned v = LD(l);
    unsigned s, mn, mx, o;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(s) : "r"(v));
    asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(mn) : "r"(v));
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(mx) : "r"(v));
    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(o) : "r"(v));
    ST(l * 4u, s); ST(l * 4u + 1u, mn); ST(l * 4u + 2u, mx); ST(l * 4u + 3u, o);
}
"#
);

/// `redux.sync` の f32 版（`.min`/`.max`。受理段のみ。対応アーキは要確認）。
pub const SIMT_REDUX_F32: &str = concat!(
    pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) simt_redux_f32(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    float v = __uint_as_float(LD(l));
    float r;
    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(r) : "f"(v));
    ST(l, __float_as_uint(r));
}
"#
);
