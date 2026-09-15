// dtype 変換（`fandhe_ai_tensor_core::cast::CastOps`。イシュー #1751・
// 親 #1613・依存 #1750）の Metal カーネル。
//
// `crate::cast::MetalCast`（`cast.rs`）から実行時コンパイルされ、
// `ops.rs::MetalBackendOps::cast_*` から呼ばれる。MSL は `double`
// 非対応のため f64 の 2 方向（`cast_f32_to_f64`／`cast_f64_to_f32`）は
// 実装せず、呼び出し元（`ops.rs`）が恒久 `Unsupported` を返しホスト
// フォールバックへ委ねる（`crates/backend-cuda/src/kernels_cast.rs`
// との差分。CUDA 側は 8 方向すべて実装済み）。
//
// cast は算術を含まない変換のため、CPU 参照実装（`fandhe_ai_tensor_core::
// cast::{cast_from_f32, cast_to_f32}`）と **bit 完全一致**する契約
// （NaN のみクラス一致）を負う。`kernels_cast.rs`（CUDA 版）と同じ
// 記述規則を MSL でも踏襲する:
//
// 1. NaN／非ゼロ判定は bit パターン（`as_type<uint>`）で行う。
//    `isnan()`／通常の浮動小数点比較には依存しない。
// 2. f32→i32／i64 の飽和境界はリテラル定数で書く（MSL に `INT_MAX`
//    等のマクロは無いためこの点は当初から問題にならないが、CUDA 側
//    との記述対称性のため明示リテラルで揃える）。
// 3. 整数→f32 は暗黙の `float(int)`／`float(long)` キャストを使う
//    （MSL には `_rn` 系 intrinsic が無いため、コンパイラの既定丸め
//    〈最近接偶数丸め。IEEE 754 準拠〉に委ねる。実機での bit 一致は
//    `docs/tensor-core-cast-design.md` §11 の事前登録どおり実測で
//    確認し、不一致なら Metal 側で当該方向のみ `Unsupported` へ戻す）。
//
// `long`（64bit 符号付き整数）は MSL がネイティブ対応する型
// （`sort.metal` の `ulong` と同じ扱い。冒頭コメント参照）。bool は
// `uchar`（0／1）としてホスト⇔デバイス間を転送する（`tensor-core::
// cast` モジュール doc「GPU 実装への注記」参照。ホスト側で生バイトを
// transmute しない）。
//
// ---- REQ-8（カーネル境界検査規約）----
//
// 全カーネルで `if (gid >= numel) return;` を維持する（グリッドが
// ちょうど割り切れない場合の末尾ブロック対策）。

#include <metal_stdlib>
using namespace metal;

// f32→i32（ゼロ方向切り捨て・範囲外は飽和・NaN→0）。
kernel void cast_f32_to_i32(
    device const float* in [[buffer(0)]],
    device int* out [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numel) {
        return;
    }
    float v = in[gid];
    uint abits = as_type<uint>(v) & 0x7fffffffu;
    bool is_nan = abits > 0x7f800000u;
    int result;
    if (is_nan) {
        result = 0;
    } else if (v >= 2147483648.0f) {
        result = 2147483647;
    } else if (v < -2147483648.0f) {
        result = (-2147483647 - 1);
    } else {
        result = int(v);
    }
    out[gid] = result;
}

// f32→i64（同上。`long` へ拡張した飽和境界を使う）。
kernel void cast_f32_to_i64(
    device const float* in [[buffer(0)]],
    device long* out [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numel) {
        return;
    }
    float v = in[gid];
    uint abits = as_type<uint>(v) & 0x7fffffffu;
    bool is_nan = abits > 0x7f800000u;
    long result;
    if (is_nan) {
        result = 0;
    } else if (v >= 9223372036854775808.0f) {
        result = 9223372036854775807L;
    } else if (v < -9223372036854775808.0f) {
        result = (-9223372036854775807L - 1);
    } else {
        result = long(v);
    }
    out[gid] = result;
}

// f32→bool（`uchar` 0／1。`v != 0.0` 相当。−0.0→0・NaN→1）。
kernel void cast_f32_to_bool(
    device const float* in [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numel) {
        return;
    }
    uint abits = as_type<uint>(in[gid]) & 0x7fffffffu;
    out[gid] = (abits != 0u) ? 1 : 0;
}

// i32→f32（最近接偶数丸め。`|v| > 2^24` は非可逆）。
kernel void cast_i32_to_f32(
    device const int* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numel) {
        return;
    }
    out[gid] = float(in[gid]);
}

// i64→f32（同上）。
kernel void cast_i64_to_f32(
    device const long* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numel) {
        return;
    }
    out[gid] = float(in[gid]);
}

// bool→f32（`true→1.0`・`false→0.0`。`uchar` 入力）。
kernel void cast_bool_to_f32(
    device const uchar* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numel) {
        return;
    }
    out[gid] = (in[gid] != 0) ? 1.0f : 0.0f;
}
