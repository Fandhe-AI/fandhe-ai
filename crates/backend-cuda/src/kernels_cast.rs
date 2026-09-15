//! dtype 変換（`fandhe_ai_tensor_core::cast::CastOps`。イシュー #1751・
//! 親 #1613・依存 #1750）の CUDA C カーネルソース（NVRTC 実行時
//! コンパイル用の静的文字列）。
//!
//! `cast.rs`（呼び出し元）は本モジュールの 8 定数を `nvrtc::
//! compile_ptx` に渡し `CudaFunction` を得る。`kernels_unique.rs`・
//! `kernels_gather_scatter.rs` と同じ理由でソースを `nvcc` 事前
//! コンパイルせず文字列のまま埋め込む（ビルド時に nvcc/CUDA ヘッダを
//! 一切要求しない。「CUDA toolkit 非搭載環境でも `cargo build
//! --workspace` が成立する」契約を維持する。`.claude/rules/
//! deps-policy.md`）。
//!
//! cast は算術を含まない変換のため、ホスト参照実装
//! （`fandhe_ai_tensor_core::cast::{cast_from_f32, cast_to_f32}`）と
//! **bit 完全一致**する契約（NaN のみクラス一致）を負う
//! （`tensor-core::cast` モジュール doc の数値契約表参照）。この契約を
//! 満たすため、以下の記述規則を全カーネル共通で守る:
//!
//! 1. **NaN／非ゼロ判定は bit パターンで行う**
//!    （`__float_as_uint(v) & 0x7fffffffu`）。`isnan()`／通常の浮動小数点
//!    比較に依存すると、NVRTC の math モード（`compile_ptx` 既定
//!    オプション自体は fast-math／ftz を指定しないが、将来の変更に
//!    対して脆弱）に暗黙依存してしまうため避ける。`abits > 0x7f800000u`
//!    が NaN、`abits != 0u` が非ゼロ（`v != 0.0` と同じ真偽値。−0.0 は
//!    `abits == 0` で false、NaN は `abits` が非ゼロのため `abits != 0u`
//!    だけで「NaN→true」も同時に成立し、ホスト `v != 0.0`〈IEEE 754:
//!    NaN はどんな値とも `!=` で真〉と一致する）。
//! 2. **f32→i32／i64 は単一の明示式**（CUDA 組み込みの飽和変換
//!    intrinsic には依存しない）。上限・下限はヘッダ非依存のリテラル
//!    定数で書く（`INT_MAX`／`LLONG_MAX` は使わない。NVRTC
//!    `compile_ptx` は include path なしのコンパイルを先に試みるため
//!    `<climits>` に依存できない）。Rust `as`（ゼロ方向切り捨て・飽和・
//!    NaN→0）の意味論と一致させる。
//! 3. **整数→f32 は最近接偶数丸め**（`__int2float_rn`／`__ll2float_rn`
//!    intrinsic を明示指定する。暗黙の `(float)` キャストの丸めモードに
//!    依存しない）。
//! 4. **f32↔f64 は単純なキャスト**（`(double)v`／`(float)d`）。
//!
//! # bool の実体化
//!
//! `bool` は `unsigned char`（0／1）としてホスト⇔デバイス間を転送する
//! （`tensor-core::cast` モジュール doc「GPU 実装への注記」・`cast.rs`
//! モジュール doc参照。ホスト側で `Vec<bool>` を生バイトから
//! transmute しない）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! 全カーネルで `idx < numel` を維持する（グリッドがちょうど割り切れ
//! ない場合の末尾ブロック対策）。添字演算は `long long`（`i32::MAX`
//! 近傍でのオーバーフロー回避。`kernels_gather_scatter.rs` と同じ
//! イシュー #1675 の教訓）。

/// 1 スレッドブロックあたりのスレッド数（8 カーネル共通。
/// `kernels_gather_scatter::GATHER_SCATTER_BLOCK_DIM` と同じ値・同じ
/// 理由）。
pub const CAST_BLOCK_DIM: u32 = 256;

/// f32→f64（完全表現。`(double)` への単純キャスト）。
pub const CAST_F32_TO_F64: &str = r#"
extern "C" __global__ void cast_f32_to_f64(
    const float* __restrict__ in,
    double* __restrict__ out,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = (double)in[idx];
    }
}
"#;

/// f32→i32（ゼロ方向切り捨て・範囲外は飽和・NaN→0）。
pub const CAST_F32_TO_I32: &str = r#"
extern "C" __global__ void cast_f32_to_i32(
    const float* __restrict__ in,
    int* __restrict__ out,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float v = in[idx];
        unsigned int bits = __float_as_uint(v);
        unsigned int abits = bits & 0x7fffffffu;
        bool is_nan = abits > 0x7f800000u;
        int result;
        if (is_nan) {
            result = 0;
        } else if (v >= 2147483648.0f) {
            result = 2147483647;
        } else if (v < -2147483648.0f) {
            result = (-2147483647 - 1);
        } else {
            result = (int)v;
        }
        out[idx] = result;
    }
}
"#;

/// f32→i64（同上。`long long` へ拡張した飽和境界を使う）。
pub const CAST_F32_TO_I64: &str = r#"
extern "C" __global__ void cast_f32_to_i64(
    const float* __restrict__ in,
    long long* __restrict__ out,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float v = in[idx];
        unsigned int bits = __float_as_uint(v);
        unsigned int abits = bits & 0x7fffffffu;
        bool is_nan = abits > 0x7f800000u;
        long long result;
        if (is_nan) {
            result = 0;
        } else if (v >= 9223372036854775808.0f) {
            result = 9223372036854775807LL;
        } else if (v < -9223372036854775808.0f) {
            result = (-9223372036854775807LL - 1);
        } else {
            result = (long long)v;
        }
        out[idx] = result;
    }
}
"#;

/// f32→bool（`unsigned char` 0／1。`v != 0.0` 相当。−0.0→0・NaN→1）。
pub const CAST_F32_TO_BOOL: &str = r#"
extern "C" __global__ void cast_f32_to_bool(
    const float* __restrict__ in,
    unsigned char* __restrict__ out,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        unsigned int abits = __float_as_uint(in[idx]) & 0x7fffffffu;
        out[idx] = (abits != 0u) ? 1 : 0;
    }
}
"#;

/// f64→f32（最近接偶数丸め・範囲超過は ±inf。`(float)` への単純キャスト）。
pub const CAST_F64_TO_F32: &str = r#"
extern "C" __global__ void cast_f64_to_f32(
    const double* __restrict__ in,
    float* __restrict__ out,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = (float)in[idx];
    }
}
"#;

/// i32→f32（最近接偶数丸め。`__int2float_rn` intrinsic を明示指定）。
pub const CAST_I32_TO_F32: &str = r#"
extern "C" __global__ void cast_i32_to_f32(
    const int* __restrict__ in,
    float* __restrict__ out,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __int2float_rn(in[idx]);
    }
}
"#;

/// i64→f32（同上。`__ll2float_rn` intrinsic を明示指定）。
pub const CAST_I64_TO_F32: &str = r#"
extern "C" __global__ void cast_i64_to_f32(
    const long long* __restrict__ in,
    float* __restrict__ out,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __ll2float_rn(in[idx]);
    }
}
"#;

/// bool→f32（`true→1.0`・`false→0.0`。`unsigned char` 入力）。
pub const CAST_BOOL_TO_F32: &str = r#"
extern "C" __global__ void cast_bool_to_f32(
    const unsigned char* __restrict__ in,
    float* __restrict__ out,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = (in[idx] != 0) ? 1.0f : 0.0f;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_SOURCES: [&str; 8] = [
        CAST_F32_TO_F64,
        CAST_F32_TO_I32,
        CAST_F32_TO_I64,
        CAST_F32_TO_BOOL,
        CAST_F64_TO_F32,
        CAST_I32_TO_F32,
        CAST_I64_TO_F32,
        CAST_BOOL_TO_F32,
    ];

    /// 全カーネルが REQ-8 の境界検査（`idx < numel`）を維持している
    /// ことの機械検証（`kernels_gather_scatter.rs`／`kernels_unique.rs`
    /// と同型の文字列テスト）。
    #[test]
    fn all_kernels_include_bounds_check() {
        for src in ALL_SOURCES {
            assert!(
                src.contains("if (idx < numel)"),
                "kernel source must retain the idx < numel bounds check: {src}"
            );
        }
    }

    /// 添字演算は `long long`（`i32::MAX` 近傍でのオーバーフロー回避。
    /// イシュー #1675 の教訓を踏襲）。
    #[test]
    fn all_kernels_use_long_long_for_flat_index_arithmetic() {
        for src in ALL_SOURCES {
            assert!(
                src.contains("long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;"),
                "kernel source must compute idx via long long arithmetic: {src}"
            );
        }
    }

    /// NaN／非ゼロ判定は bit パターン（`__float_as_uint`）で行い、
    /// `isnan()` には依存しない（モジュール doc 規則 1）。
    #[test]
    fn nan_sensitive_kernels_use_bit_pattern_checks_not_isnan() {
        for src in [CAST_F32_TO_I32, CAST_F32_TO_I64, CAST_F32_TO_BOOL] {
            assert!(src.contains("__float_as_uint"));
            assert!(!src.contains("isnan"));
        }
    }

    /// f32→i32／i64 の飽和境界はリテラル定数（`INT_MAX`／`LLONG_MAX`
    /// 等のヘッダマクロに依存しない。モジュール doc 規則 2）。
    #[test]
    fn saturating_int_casts_use_literal_bounds_not_header_macros() {
        assert!(CAST_F32_TO_I32.contains("2147483647"));
        assert!(CAST_F32_TO_I32.contains("(-2147483647 - 1)"));
        assert!(!CAST_F32_TO_I32.contains("INT_MAX"));
        assert!(!CAST_F32_TO_I32.contains("INT_MIN"));

        assert!(CAST_F32_TO_I64.contains("9223372036854775807LL"));
        assert!(CAST_F32_TO_I64.contains("(-9223372036854775807LL - 1)"));
        assert!(!CAST_F32_TO_I64.contains("LLONG_MAX"));
        assert!(!CAST_F32_TO_I64.contains("LLONG_MIN"));
    }

    /// 整数→f32 は丸めモードを明示する `_rn` intrinsic を使う
    /// （モジュール doc 規則 3）。
    #[test]
    fn int_to_float_casts_use_round_to_nearest_intrinsics() {
        assert!(CAST_I32_TO_F32.contains("__int2float_rn"));
        assert!(CAST_I64_TO_F32.contains("__ll2float_rn"));
    }

    /// bool 方向は `unsigned char`（0／1）で転送する（モジュール doc
    /// 「bool の実体化」節）。
    #[test]
    fn bool_directions_use_unsigned_char() {
        assert!(CAST_F32_TO_BOOL.contains("unsigned char* __restrict__ out"));
        assert!(CAST_BOOL_TO_F32.contains("const unsigned char* __restrict__ in"));
    }

    /// f32↔f64 は単純なキャストのみ（算術・丸め補正を追加しない）。
    #[test]
    fn f32_f64_directions_use_plain_cast() {
        assert!(CAST_F32_TO_F64.contains("out[idx] = (double)in[idx];"));
        assert!(CAST_F64_TO_F32.contains("out[idx] = (float)in[idx];"));
    }
}
