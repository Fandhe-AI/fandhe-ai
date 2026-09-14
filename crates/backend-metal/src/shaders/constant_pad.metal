// 定数パディング（`torch.nn.functional.pad(mode='constant')` 相当。
// イシュー #1756。CPU 参照実装 `crates/backend-cpu/src/constant_pad.rs`
// の Metal 対応版）。
//
// `crate::constant_pad::MetalConstantPad`（`constant_pad.rs`）から実行時
// コンパイルされ、`ops.rs::MetalBackendOps::pad` から呼ばれる。
//
// 1 スレッド = 1 出力位置（`gid`）。出力の各要素は「`input` 内部位置
// ならそのままコピー・パディング領域なら `value`」の 2 分岐のみで
// 決まる純粋なコピー演算（算術を含まない）のため、CPU 参照実装
// （`crates/backend-cpu/src/constant_pad.rs::pad`）と bit 完全一致する
// （`value` が NaN の場合のみ payload がハードウェア依存のためクラス
// 一致で比較する）。
//
// `crates/backend-cuda/src/kernels_constant_pad.rs::CONSTANT_PAD_F32` と
// 同じ「座標配列を保持しない末尾軸剥がし方式」を採る（`shaders/
// gather_scatter.metal` の `GS_MAX_RANK` 固定長スタック配列方式とは
// 異なる。pad は gather／scatter と違い `dim` 軸のみの差し替えではなく
// 全軸が対象のため、固定長配列を避けることで rank 上限を設けずに済む）。
//
// **ホスト側の逐語モデル**: `crates/backend-metal/src/
// constant_pad_model.rs`（`constant_pad_model`）が本ファイルの
// アルゴリズムのホスト側逐語再現であり、CPU 参照実装
// （`fandhe_ai_backend_cpu::CpuBackendOps`）との bit 一致をユニット
// テスト（Linux 実行可能）で網羅検証する。本ファイルを変更した場合は
// 同モジュールも追従させること。実機でのカーネル出力 bit 一致は
// `tests/constant_pad_parity.rs`（`#[ignore]`）で確認する。
//
// **添字計算・境界検査（REQ-8）**: shape 配列（`out_shape`／`in_shape`／
// `before`／`in_strides`）はすべて `constant uint*` で渡し、添字計算は
// `long`（64bit 符号付き。`before[a]` を引いた際のアンダーフローを
// 負値として素直に検出するため `ulong` ではなく符号付きを使う）で行う。
// `gid >= numel` の早期 return（REQ-8「手動境界チェックを省略しない」）
// を必ず持つ。

#include <metal_stdlib>
using namespace metal;

// `input(0)`／`out(1)`／`shapes(2, uint: out_shape[rank] ++ in_shape[rank]
// ++ before[rank] ++ in_strides[rank])`／`constant uint& rank(3)`／
// `numel(4)`／`constant float& value(5)`。
kernel void constant_pad_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint* shapes [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant uint& numel [[buffer(4)]],
    constant float& value [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // REQ-8: grid は `ceil(numel/W)` threadgroup のため端で `numel` を
    // はみ出しうる（手動境界チェックを省略しない）。
    if (gid >= numel) {
        return;
    }
    constant uint* out_shape = shapes;
    constant uint* in_shape = shapes + rank;
    constant uint* before = shapes + 2u * rank;

    // `in_strides`（行優先ストライド）はカーネル起動側で事前計算して
    // 渡す（`crates/backend-cuda/src/kernels_constant_pad.rs` と同じ
    // 設計。カーネル内で毎スレッド再計算しない）。
    constant uint* in_strides = shapes + 3u * rank;

    long rem = (long)gid;
    long in_flat = 0;
    bool inside = true;
    for (int a = (int)rank - 1; a >= 0; a--) {
        long axis_size = (long)out_shape[a];
        long c = rem % axis_size;
        rem /= axis_size;
        long src_c = c - (long)before[a];
        if (src_c < 0 || src_c >= (long)in_shape[a]) {
            inside = false;
            break;
        }
        in_flat += src_c * (long)in_strides[a];
    }
    out[gid] = inside ? input[in_flat] : value;
}
