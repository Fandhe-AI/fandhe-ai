//! `ScalarUnaryOp`／`ScalarBinaryOp` の HIP カーネルソース（`backend-cuda::kernels_scalar_op` の
//! 移植。イシュー #3433・分解案 R2）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_scalar_source`]）と、kind から
//! カーネル仕様（[`KernelSpec`]）・起動ペイロードを引く表（[`unary_spec`]・[`binary_spec`]・
//! [`unary_payload`]）。呼び出し元は `elementwise.rs` の `RocmScalarOp` と `ops.rs`。
//!
//! # 構造（CUDA 版との違い）
//! CUDA 版は kind ごとに式を `format!` でソースへ埋めて 1 kind = 1 コンパイル単位にするが、
//! A03（静的テンプレートのみ）に合わせ、本移植は **対応する全 kind のカーネルを 1 本の静的ソース
//! （1 module）に並べる**。式は `concat!` で定数畳み込みされ実行時に組み立てない。カーネル名は
//! `scalar_unary_<kind_name>`／`scalar_binary_<kind_name>`（`ScalarOpKind::kind_name`）。
//! `f32` ペイロード（Clamp の min/max、LeakyRelu・Elu の係数、Softplus の beta/threshold）は
//! ソースへ入れず、`float p0, float p1` のカーネル引数で渡す。
//!
//! # 対応範囲
//! CUDA の実装集合と同一。unary は Sqrt・Neg・Abs・Log・Log2・Log10・Sin・Cos・Tan・Clamp・
//! LeakyRelu・Hardswish・Silu・Elu・Gelu・GeluTanh・Softplus、binary は Sub・Div・Pow・
//! Gt・Ge・Lt・Le・Eq・Ne。それ以外は `None`（呼び出し側が `Unsupported` を返し、autodiff が
//! ホスト参照実装へフォールバックする既存契約）。
//!
//! # 数値契約
//! - ソース先頭に `#pragma clang fp contract(off)`（`kernels_elementwise.rs` と同じ理由。特に
//!   `GeluTanh` の `x + 0.044715f * x * x * x` や `Hardswish` の `x + 3.0f` 周りが暗黙に FMA へ
//!   縮約されると CPU 参照と丸めがずれる）。FMA は使わない。
//! - `Sub`・`Div`・`Sqrt`・`Neg`・`Abs`・比較・`Clamp`・`LeakyRelu` は CPU と bit 一致の想定。
//!   超越関数（`logf`・`sinf`・`powf`・`expf`・`erff` 等）は GPU の単精度 libm で、bit 一致は
//!   主張せず REQ-2 複合判定のみで検証する。近似 intrinsic（`__logf` 等）・`double` 版・
//!   `fminf`／`fmaxf`（NaN の扱いが明示分岐と異なる）は使わない。
//! - 縮約を持たないため `f64` アキュムレータ契約は該当しない。
//!
//! # REQ-8
//! 全カーネルが `gid >= numel` を検査してから読み書きする。添字は `long long`。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};
use fandhe_ai_tensor_core::{ScalarBinaryOp, ScalarUnaryOp};

/// unary カーネル 1 本分の静的テキストを `concat!` で組む（`$params` は追加の `float` 引数宣言）。
macro_rules! unary_kernel {
    ($name:literal, $params:literal, $expr:literal) => {
        concat!(
            "extern \"C\" __global__ __launch_bounds__(BLOCK_DIM) void scalar_unary_",
            $name,
            "(\n    const float* __restrict__ a,\n    float* __restrict__ out,\n    int numel",
            $params,
            ")\n{\n    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n",
            "    if (gid >= (long long)numel) {\n        return;\n    }\n",
            "    float x = a[gid];\n    out[gid] = ",
            $expr,
            ";\n}\n\n"
        )
    };
}

/// binary カーネル 1 本分（変数名は `a_v`／`b_v`。同 shape 前提でブロードキャストはホスト側で展開済み）。
macro_rules! binary_kernel {
    ($name:literal, $expr:literal) => {
        concat!(
            "extern \"C\" __global__ __launch_bounds__(BLOCK_DIM) void scalar_binary_",
            $name,
            "(\n    const float* __restrict__ a,\n    const float* __restrict__ b,\n",
            "    float* __restrict__ out,\n    int numel)\n{\n",
            "    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n",
            "    if (gid >= (long long)numel) {\n        return;\n    }\n",
            "    float a_v = a[gid];\n    float b_v = b[gid];\n    out[gid] = ",
            $expr,
            ";\n}\n\n"
        )
    };
}

/// 静的テンプレート本体（全対応 kind のカーネル）。式の正は `ScalarUnaryOp::apply`／
/// `ScalarBinaryOp::apply`（`tensor-core::scalar_op`）で、本文は HIP C++ 版の意図的な複製
/// （言語境界のため。`backend-cuda::kernels_scalar_op` と同じ事情）。
pub(crate) const SCALAR_BODY: &str = concat!(
    "#pragma clang fp contract(off)\n\n",
    unary_kernel!("sqrt", "", "sqrtf(x)"),
    unary_kernel!("neg", "", "-x"),
    unary_kernel!("abs", "", "fabsf(x)"),
    unary_kernel!("log", "", "logf(x)"),
    unary_kernel!("log2", "", "log2f(x)"),
    unary_kernel!("log10", "", "log10f(x)"),
    unary_kernel!("sin", "", "sinf(x)"),
    unary_kernel!("cos", "", "cosf(x)"),
    unary_kernel!("tan", "", "tanf(x)"),
    // NaN は伝播し、min > max は常に max（`ScalarUnaryOp::apply` の Clamp 分岐の逐語）。
    unary_kernel!(
        "clamp",
        ",\n    float p0,\n    float p1",
        "isnan(x) ? x : (p0 > p1 ? p1 : (x < p0 ? p0 : (x > p1 ? p1 : x)))"
    ),
    unary_kernel!(
        "leaky_relu",
        ",\n    float p0",
        "(x >= 0.0f) ? x : (p0 * x)"
    ),
    // relu6(x + 3) / 6 を先に評価してから x に掛ける順序を保つ（x が大きいときの overflow 回避）。
    unary_kernel!(
        "hardswish",
        "",
        "x * ((x + 3.0f < 0.0f) ? 0.0f : (((x + 3.0f > 6.0f) ? 6.0f : (x + 3.0f)) / 6.0f))"
    ),
    unary_kernel!(
        "silu",
        "",
        "(x >= 0.0f) ? (x * (1.0f / (1.0f + expf(-x)))) : (x * (expf(x) / (1.0f + expf(x))))"
    ),
    unary_kernel!(
        "elu",
        ",\n    float p0",
        "(x > 0.0f) ? x : (p0 * expm1f(x))"
    ),
    unary_kernel!(
        "gelu",
        "",
        "0.5f * x * (1.0f + erff(x * 0.70710678118654752f))"
    ),
    unary_kernel!(
        "gelu_tanh",
        "",
        "0.5f * x * (1.0f + tanhf(0.7978846f * (x + 0.044715f * x * x * x)))"
    ),
    // `p0` = beta、`p1` = threshold。`log1pf` で桁落ちを避ける（`f32::ln_1p` と同じ意図）。
    unary_kernel!(
        "softplus",
        ",\n    float p0,\n    float p1",
        "(x * p0 > p1) ? x : (log1pf(expf(p0 * x)) / p0)"
    ),
    binary_kernel!("sub", "a_v - b_v"),
    binary_kernel!("div", "a_v / b_v"),
    binary_kernel!("pow", "powf(a_v, b_v)"),
    binary_kernel!("gt", "(a_v > b_v) ? 1.0f : 0.0f"),
    binary_kernel!("ge", "(a_v >= b_v) ? 1.0f : 0.0f"),
    binary_kernel!("lt", "(a_v < b_v) ? 1.0f : 0.0f"),
    binary_kernel!("le", "(a_v <= b_v) ? 1.0f : 0.0f"),
    binary_kernel!("eq", "(a_v == b_v) ? 1.0f : 0.0f"),
    binary_kernel!("ne", "(a_v != b_v) ? 1.0f : 0.0f"),
);

/// `(a, out, numel)`。
const UNARY_ARGS_0: [ArgKind; 3] = [ArgKind::Ptr, ArgKind::Ptr, ArgKind::I32];
/// `(a, out, numel, p0)`。
const UNARY_ARGS_1: [ArgKind; 4] = [ArgKind::Ptr, ArgKind::Ptr, ArgKind::I32, ArgKind::F32];
/// `(a, out, numel, p0, p1)`。
const UNARY_ARGS_2: [ArgKind; 5] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::F32,
    ArgKind::F32,
];
/// `(a, b, out, numel)`。
const BINARY_ARGS: [ArgKind; 4] = [ArgKind::Ptr, ArgKind::Ptr, ArgKind::Ptr, ArgKind::I32];

/// `&'static KernelSpec` を match の腕ごとに 1 個ずつ作る（`static` を式ブロック内に置く）。
macro_rules! spec {
    ($name:literal, $args:expr) => {{
        static S: KernelSpec = KernelSpec {
            name: $name,
            args: $args,
        };
        Some(&S)
    }};
}

/// unary kind のカーネル仕様。対象外（`Relu`・`Floor`・`Atan` など）は `None`。
pub(crate) fn unary_spec(op: ScalarUnaryOp) -> Option<&'static KernelSpec> {
    match op {
        ScalarUnaryOp::Sqrt => spec!("scalar_unary_sqrt", &UNARY_ARGS_0),
        ScalarUnaryOp::Neg => spec!("scalar_unary_neg", &UNARY_ARGS_0),
        ScalarUnaryOp::Abs => spec!("scalar_unary_abs", &UNARY_ARGS_0),
        ScalarUnaryOp::Log => spec!("scalar_unary_log", &UNARY_ARGS_0),
        ScalarUnaryOp::Log2 => spec!("scalar_unary_log2", &UNARY_ARGS_0),
        ScalarUnaryOp::Log10 => spec!("scalar_unary_log10", &UNARY_ARGS_0),
        ScalarUnaryOp::Sin => spec!("scalar_unary_sin", &UNARY_ARGS_0),
        ScalarUnaryOp::Cos => spec!("scalar_unary_cos", &UNARY_ARGS_0),
        ScalarUnaryOp::Tan => spec!("scalar_unary_tan", &UNARY_ARGS_0),
        ScalarUnaryOp::Clamp { .. } => spec!("scalar_unary_clamp", &UNARY_ARGS_2),
        ScalarUnaryOp::LeakyRelu { .. } => spec!("scalar_unary_leaky_relu", &UNARY_ARGS_1),
        ScalarUnaryOp::Hardswish => spec!("scalar_unary_hardswish", &UNARY_ARGS_0),
        ScalarUnaryOp::Silu => spec!("scalar_unary_silu", &UNARY_ARGS_0),
        ScalarUnaryOp::Elu { .. } => spec!("scalar_unary_elu", &UNARY_ARGS_1),
        ScalarUnaryOp::Gelu => spec!("scalar_unary_gelu", &UNARY_ARGS_0),
        ScalarUnaryOp::GeluTanh => spec!("scalar_unary_gelu_tanh", &UNARY_ARGS_0),
        ScalarUnaryOp::Softplus { .. } => spec!("scalar_unary_softplus", &UNARY_ARGS_2),
        // 対象外 kind（`ScalarUnaryOp` は `#[non_exhaustive]`）。CUDA と同じく HIP に触れず
        // `Unsupported` を返し、ホスト参照実装へのフォールバックに委ねる。
        _ => None,
    }
}

/// binary kind のカーネル仕様。対象外（`Atan2` など）は `None`。
pub(crate) fn binary_spec(op: ScalarBinaryOp) -> Option<&'static KernelSpec> {
    match op {
        ScalarBinaryOp::Sub => spec!("scalar_binary_sub", &BINARY_ARGS),
        ScalarBinaryOp::Div => spec!("scalar_binary_div", &BINARY_ARGS),
        ScalarBinaryOp::Pow => spec!("scalar_binary_pow", &BINARY_ARGS),
        ScalarBinaryOp::Gt => spec!("scalar_binary_gt", &BINARY_ARGS),
        ScalarBinaryOp::Ge => spec!("scalar_binary_ge", &BINARY_ARGS),
        ScalarBinaryOp::Lt => spec!("scalar_binary_lt", &BINARY_ARGS),
        ScalarBinaryOp::Le => spec!("scalar_binary_le", &BINARY_ARGS),
        ScalarBinaryOp::Eq => spec!("scalar_binary_eq", &BINARY_ARGS),
        ScalarBinaryOp::Ne => spec!("scalar_binary_ne", &BINARY_ARGS),
        _ => None,
    }
}

/// 起動時に `numel` の後ろへ渡す `f32` ペイロード（`p0`・`p1` の順。ソースには入れない）。
/// 戻り値は `(個数, 値)`。個数は [`unary_spec`] の引数列（`F32` の数）と一致する。
pub(crate) fn unary_payload(op: ScalarUnaryOp) -> (usize, [f32; 2]) {
    match op {
        ScalarUnaryOp::Clamp { min, max } => (2, [min, max]),
        ScalarUnaryOp::LeakyRelu { negative_slope } => (1, [negative_slope, 0.0]),
        ScalarUnaryOp::Elu { alpha } => (1, [alpha, 0.0]),
        ScalarUnaryOp::Softplus { beta, threshold } => (2, [beta, threshold]),
        _ => (0, [0.0, 0.0]),
    }
}

/// scalar 用ソースを生成する（`BLOCK_DIM` のみ前置）。
pub(crate) fn render_scalar_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(&[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], SCALAR_BODY)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};
    use fandhe_ai_tensor_core::ScalarOpKind;

    /// 対応する全 unary kind（ペイロード値は任意）。
    pub(crate) fn supported_unary() -> Vec<ScalarUnaryOp> {
        use ScalarUnaryOp::*;
        vec![
            Sqrt,
            Neg,
            Abs,
            Log,
            Log2,
            Log10,
            Sin,
            Cos,
            Tan,
            Clamp {
                min: -0.5,
                max: 0.75,
            },
            LeakyRelu {
                negative_slope: 0.1,
            },
            Hardswish,
            Silu,
            Elu { alpha: 1.5 },
            Gelu,
            GeluTanh,
            Softplus {
                beta: 1.5,
                threshold: 20.0,
            },
        ]
    }

    pub(crate) fn supported_binary() -> Vec<ScalarBinaryOp> {
        use ScalarBinaryOp::*;
        vec![Sub, Div, Pow, Gt, Ge, Lt, Le, Eq, Ne]
    }

    #[test]
    fn every_spec_matches_source_signature_and_kind_name() {
        for op in supported_unary() {
            let spec = unary_spec(op).expect("対応 kind");
            assert_eq!(spec.name, format!("scalar_unary_{}", op.kind_name()));
            assert_eq!(kernel_param_kinds(SCALAR_BODY, spec.name), spec.args);
            // F32 引数の数 = ペイロード個数。
            let floats = spec.args.iter().filter(|k| **k == ArgKind::F32).count();
            assert_eq!(floats, unary_payload(op).0, "{op:?}");
        }
        for op in supported_binary() {
            let spec = binary_spec(op).expect("対応 kind");
            assert_eq!(spec.name, format!("scalar_binary_{}", op.kind_name()));
            assert_eq!(kernel_param_kinds(SCALAR_BODY, spec.name), spec.args);
        }
    }

    #[test]
    fn every_kernel_in_source_has_a_spec() {
        let in_source = SCALAR_BODY.matches("__global__").count();
        assert_eq!(
            in_source,
            supported_unary().len() + supported_binary().len()
        );
    }

    #[test]
    fn unsupported_kinds_are_none() {
        use ScalarUnaryOp as U;
        for op in [
            U::Relu,
            U::Exp,
            U::Tanh,
            U::Sigmoid,
            U::Floor,
            U::Ceil,
            U::Round,
            U::Sign,
            U::Reciprocal,
            U::Rsqrt,
            U::Erf,
            U::Atan,
            U::IsNan,
            U::Selu,
            U::PowScalar { exponent: 2.0 },
        ] {
            assert!(unary_spec(op).is_none(), "{op:?}");
        }
        for op in [
            ScalarBinaryOp::Atan2,
            ScalarBinaryOp::Add,
            ScalarBinaryOp::Mul,
        ] {
            assert!(binary_spec(op).is_none(), "{op:?}");
        }
    }

    #[test]
    fn payload_values_never_enter_source() {
        // ソースは静的で、ペイロードの値（`f32`）に依存しない。
        let a = render_scalar_source().unwrap();
        let _ = unary_payload(ScalarUnaryOp::Clamp {
            min: 123.456,
            max: 789.0,
        });
        assert_eq!(a, render_scalar_source().unwrap());
        assert!(!a.as_str().contains("123.456"));
        assert_eq!(
            unary_payload(ScalarUnaryOp::Softplus {
                beta: 2.0,
                threshold: 9.0
            }),
            (2, [2.0, 9.0])
        );
    }

    #[test]
    fn source_defines_only_block_dim_and_starts_with_contract_off() {
        let s = render_scalar_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define BLOCK_DIM 256"]);
        assert!(SCALAR_BODY.starts_with("#pragma clang fp contract(off)\n"));
    }

    #[test]
    fn req8_guard_and_numeric_contract_in_source() {
        let kernels = supported_unary().len() + supported_binary().len();
        assert_eq!(
            SCALAR_BODY.matches("if (gid >= (long long)numel)").count(),
            kernels
        );
        for banned in [
            "fma(",
            "__builtin_fma",
            "fminf",
            "fmaxf",
            "__logf",
            "__expf",
            "__sinf",
            "__cosf",
            "__tanf",
            "__powf",
            "__shfl",
            "warpSize",
            "double",
        ] {
            assert!(!SCALAR_BODY.contains(banned), "{banned}");
        }
        // 倍精度の libm 名（`exp(`・`log(` 等）へ暗黙昇格する書き方をしていない。
        for banned in [" exp(", " log(", " sin(", " pow("] {
            assert!(!SCALAR_BODY.contains(banned), "{banned}");
        }
        assert_eq!(SCALAR_BODY.matches("? 1.0f : 0.0f").count(), 6);
    }
}
