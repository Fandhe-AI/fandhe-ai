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
