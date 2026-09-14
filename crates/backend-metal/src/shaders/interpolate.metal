// 最近傍リサンプリング（`torch.nn.functional.interpolate
// (mode='nearest')` 相当。イシュー #1757。CPU 参照実装
// `crates/backend-cpu/src/interpolate.rs` の Metal 対応版）。
//
// `crate::interpolate::MetalInterpolate`（`interpolate.rs`）から実行時
// コンパイルされ、`ops.rs::MetalBackendOps::interpolate` から呼ばれる。
//
// 1 スレッド = 1 出力位置（`gid`）。出力の各要素は対応する入力の単一
// 要素をそのままコピーする添字演算のみで決まる（算術を含まない）ため、
// CPU 参照実装（`crates/backend-cpu/src/interpolate.rs::
// interpolate_nearest`）と bit 完全一致する（NaN のみ payload が
// ハードウェア依存のためクラス一致で比較する）。
//
// `crates/backend-cuda/src/kernels_interpolate.rs::
// INTERPOLATE_NEAREST_F32` と同じ「座標配列を保持しない末尾軸剥がし
// 方式」を採る（`shaders/gather_scatter.metal` の `GS_MAX_RANK` 固定長
// スタック配列方式とは異なる。interpolate は gather／scatter と違い
// `dim` 軸のみの差し替えではなく全軸が対象のため、固定長配列を避ける
// ことで rank 上限を設けずに済む）。
//
// **ホスト側の逐語モデル**: `crates/backend-metal/src/
// interpolate_model.rs`（`interpolate_model`）が本ファイルの
// アルゴリズムのホスト側逐語再現であり、CPU 参照実装
// （`fandhe_ai_backend_cpu::CpuBackendOps`）との bit 一致をユニット
// テスト（Linux 実行可能）で網羅検証する。本ファイルを変更した場合は
// 同モジュールも追従させること。実機でのカーネル出力 bit 一致は
// `tests/interpolate_parity.rs`（`#[ignore]`）で確認する。
//
// **添字計算・境界検査（REQ-8）**: shape 配列（`out_shape`／
// `in_shape`／`in_strides`）はすべて `constant uint*` で渡し、添字
// 計算は `ulong`（64bit 符号なし）で行う。`gid >= numel` の早期 return
// （REQ-8「手動境界チェックを省略しない」）を必ず持つ。
//
// **符号なし採用の理由（イシュー #1834 codex-review P0 是正）**:
// 起動前検証（`interpolate_model.rs::validate_interpolate_launch`）は
// 各軸を `u32::MAX` まで許容するため、空間軸の座標積 `c * in_shape[a]`
// は最大 `(2^32-1)^2 ≈ 1.8447e19` に達しうる。これは `i64::MAX
// ≈ 9.2234e18` を超えるが `u64::MAX ≈ 1.8447e19` には収まるため、
// 符号付き `long` では中間積が負値へオーバーフロー（UB）し、下限
// クランプを持たないと `input[in_flat]` への範囲外読み出しにつながる
// （上限のみの `min(src_c, in_shape[a]-1)` クランプは負値化した
// `src_c` を防げない）。`ulong` 採用によりこの積は自然に安全域へ収まる。
// 空間軸の src 添字は整数除算 `(c * in_shape[a]) / out_shape[a]` の後、
// 数学的に `[0, in_shape[a])` の範囲内が保証されるが縦深防御として
// `min(src_c, in_shape[a]-1)` へ明示的にクランプする。
//
// ホスト側逐語モデル（`interpolate_model.rs::interpolate_nearest_model`）
// も本ファイルと同じ `u64` 整数契約へ揃えている（`usize`〈通常 64bit〉
// はこの符号付きオーバーフローをそもそも再現しないため、モデル側の
// 蓄積変数を明示的に `u64` とし kernel の演算意味論と一致させる）。

#include <metal_stdlib>
using namespace metal;

// `input(0)`／`out(1)`／`shapes(2, uint: out_shape[rank] ++
// in_shape[rank] ++ in_strides[rank])`／`constant uint& rank(3)`／
// `constant uint& spatial_start(4)`／`numel(5)`。
kernel void interpolate_nearest_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint* shapes [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant uint& spatial_start [[buffer(4)]],
    constant uint& numel [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // REQ-8: grid は `ceil(numel/W)` threadgroup のため端で `numel` を
    // はみ出しうる（手動境界チェックを省略しない）。
    if (gid >= numel) {
        return;
    }
    constant uint* out_shape = shapes;
    constant uint* in_shape = shapes + rank;

    // `in_strides`（行優先ストライド）はカーネル起動側で事前計算して
    // 渡す（`crates/backend-cuda/src/kernels_interpolate.rs` と同じ
    // 設計。カーネル内で毎スレッド再計算しない）。
    constant uint* in_strides = shapes + 2u * rank;

    ulong rem = (ulong)gid;
    ulong in_flat = 0;
    for (int a = (int)rank - 1; a >= 0; a--) {
        ulong axis_size = (ulong)out_shape[a];
        ulong c = rem % axis_size;
        rem /= axis_size;
        ulong src_c;
        if ((uint)a >= spatial_start) {
            ulong in_size = (ulong)in_shape[a];
            // `c` と `in_size` はいずれも最大 `u32::MAX` のため積は
            // 最大 `(2^32-1)^2 ≈ 1.8447e19` で `ulong`（`u64::MAX
            // ≈ 1.8447e19`）に収まる（上記モジュール冒頭コメント
            // 参照。符号付き `long` ではここが `i64::MAX` を超え
            // オーバーフローする）。
            src_c = (c * in_size) / axis_size;
            ulong max_c = in_size - 1;
            if (src_c > max_c) {
                src_c = max_c;
            }
        } else {
            src_c = c;
        }
        in_flat += src_c * (ulong)in_strides[a];
    }
    out[gid] = input[in_flat];
}

// バイリニアリサンプリング（`torch.nn.functional.interpolate
// (mode='bilinear', align_corners=…)` 相当。イシュー #1762。CPU
// 参照実装 `crates/backend-cpu/src/interpolate.rs::
// interpolate_bilinear` の Metal 対応版）。
//
// `crate::interpolate::MetalInterpolate`（`interpolate.rs`）から実行時
// コンパイルされ、`ops.rs::MetalBackendOps::interpolate` から呼ばれる。
//
// `interpolate_nearest_f32` と異なり空間軸は**ちょうど 2 軸**
// （末尾 2 軸 = `rank-2`〈H〉・`rank-1`〈W〉限定。
// `InterpolateMode::Bilinear` doc・`interpolate_out_shape_for_mode`
// 参照）で算術（4 近傍の線形重み付け合成）を含むため、受入契約は
// REQ-2 統一複合判定（`assert_parity`）——`Nearest` のような bit
// 完全一致は前提としない。
//
// **ホスト側の逐語モデル**: `crates/backend-metal/src/
// interpolate_model.rs::interpolate_bilinear_model` が本カーネルの
// 添字・ブレンド計算のホスト側逐語再現（`fma` を `f32::mul_add` で
// 再現）。`scale_h`／`scale_w` はホスト側（`fandhe_ai_tensor_core::
// bilinear_scale`。forward の他バックエンド・ホスト参照実装と共有
// する単一情報源）で 1 回だけ計算しカーネル引数として渡す。
//
// `shapes`（buffer 2）は `interpolate_nearest_f32` と同一レイアウト
// （`out_shape[rank] ++ in_shape[rank] ++ in_strides[rank]`）を共有
// する。

// `input(0)`／`out(1)`／`shapes(2)`／`constant uint& rank(3)`／
// `constant uint& align_corners(4, 0 または 1)`／
// `constant uint& numel(5)`／`constant float& scale_h(6)`／
// `constant float& scale_w(7)`。
kernel void interpolate_bilinear_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint* shapes [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant uint& align_corners [[buffer(4)]],
    constant uint& numel [[buffer(5)]],
    constant float& scale_h [[buffer(6)]],
    constant float& scale_w [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    // REQ-8: grid は `ceil(numel/W)` threadgroup のため端で `numel` を
    // はみ出しうる（手動境界チェックを省略しない）。
    if (gid >= numel) {
        return;
    }
    constant uint* out_shape = shapes;
    constant uint* in_shape = shapes + rank;
    constant uint* in_strides = shapes + 2u * rank;

    ulong rem = (ulong)gid;
    ulong base_flat = 0;
    ulong cy = 0;
    ulong cx = 0;
    for (int a = (int)rank - 1; a >= 0; a--) {
        ulong axis_size = (ulong)out_shape[a];
        ulong c = rem % axis_size;
        rem /= axis_size;
        if ((uint)a == rank - 1) {
            cx = c;
        } else if ((uint)a == rank - 2) {
            cy = c;
        } else {
            base_flat += c * (ulong)in_strides[a];
        }
    }

    ulong in_h = (ulong)in_shape[rank - 2];
    ulong in_w = (ulong)in_shape[rank - 1];
    ulong stride_h = (ulong)in_strides[rank - 2];
    ulong stride_w = (ulong)in_strides[rank - 1];

    float srcy;
    float srcx;
    if (align_corners != 0u) {
        srcy = (float)cy * scale_h;
        srcx = (float)cx * scale_w;
    } else {
        srcy = fma((float)cy + 0.5f, scale_h, -0.5f);
        if (srcy < 0.0f) {
            srcy = 0.0f;
        }
        srcx = fma((float)cx + 0.5f, scale_w, -0.5f);
        if (srcx < 0.0f) {
            srcx = 0.0f;
        }
    }

    long i0y = (long)floor(srcy);
    if (i0y > (long)in_h - 1) {
        i0y = (long)in_h - 1;
    }
    long i1y = i0y + 1;
    if (i1y > (long)in_h - 1) {
        i1y = (long)in_h - 1;
    }
    float l1y = srcy - (float)i0y;
    float l0y = 1.0f - l1y;

    long i0x = (long)floor(srcx);
    if (i0x > (long)in_w - 1) {
        i0x = (long)in_w - 1;
    }
    long i1x = i0x + 1;
    if (i1x > (long)in_w - 1) {
        i1x = (long)in_w - 1;
    }
    float l1x = srcx - (float)i0x;
    float l0x = 1.0f - l1x;

    ulong u_i0y = (ulong)i0y;
    ulong u_i1y = (ulong)i1y;
    ulong u_i0x = (ulong)i0x;
    ulong u_i1x = (ulong)i1x;

    float v00 = input[base_flat + u_i0y * stride_h + u_i0x * stride_w];
    float v01 = input[base_flat + u_i0y * stride_h + u_i1x * stride_w];
    float v10 = input[base_flat + u_i1y * stride_h + u_i0x * stride_w];
    float v11 = input[base_flat + u_i1y * stride_h + u_i1x * stride_w];

    float row0 = fma(l1x, v01, l0x * v00);
    float row1 = fma(l1x, v11, l0x * v10);
    out[gid] = fma(l1y, row1, l0y * row0);
}
