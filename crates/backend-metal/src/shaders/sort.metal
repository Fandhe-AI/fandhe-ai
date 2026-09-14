// sort／topk（`torch.sort`／`torch.topk` 相当。イシュー #1741）の
// `crates/backend-cuda/src/kernels_sort.rs` の Metal 対応版。ホスト
// モデル・鍵設計は `crate::sort_model`（doc コメント「鍵設計」節）を
// 正とする。
//
// `crate::sort::MetalSort`（`sort.rs`）から実行時コンパイルされ、
// `ops.rs::MetalBackendOps::sort`／`topk` から呼ばれる。
//
// 3 カーネル: `sort_build_keys_u64`（入力 → 64bit 合成キー配列を構築）
// → `bitonic_step_u64`（ホスト側ループが k／j を変えながら繰り返し
// エンコードするビットニックソート 1 ステップ）→ `sort_finalize_f32`
// （ソート済みキー配列の先頭 out_len 要素から values／index を復元）。
//
// MSL は `double` 非対応だが `ulong`（64bit 無符号整数）はネイティブ
// 対応（`gather_scatter.metal::bias_f64_widen` 等が既に使用）。合成
// キーの算術は整数演算のみ（浮動小数点を経由しない）ため決定的。
//
// ---- REQ-8（カーネル境界検査規約）----
//
// 3 カーネルとも `gid`（グローバルスレッド添字）を対象要素数
// （`total`）で境界検査したうえで早期 return する。最適化を理由に
// 省略しない。

#include <metal_stdlib>
using namespace metal;

// `sort_model::value_key` の逐語 GPU 実装。NaN は符号・payload に
// 関わらず常に最大キー、±0 は同値化する。
inline uint value_key_msl(float v) {
    if (isnan(v)) {
        return 0xFFFFFFFFu;
    }
    float normalized = (v == 0.0f) ? 0.0f : v;
    uint bits = as_type<uint>(normalized);
    if ((bits >> 31) == 1u) {
        return ~bits;
    }
    return bits | 0x80000000u;
}

// 入力 `input`（行優先 contiguous）から `lines * padded` 長の合成キー
// 配列を構築する（`sort_model::build_keys_host` の逐語 GPU 実装）。
// `gid` はライン内添字方向にもライン方向にもまたがるフラットな
// `lines * padded` 空間の添字（`line = gid / padded`・`i = gid %
// padded`）。`i >= dim_size`（パディング域）は `PADDING_KEY`
// （`u64::MAX`）を書く。
kernel void sort_build_keys_u64(
    device const float* input [[buffer(0)]],
    device ulong* keys [[buffer(1)]],
    constant uint& lines [[buffer(2)]],
    constant uint& dim_size [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& padded [[buffer(5)]],
    constant uint& descending [[buffer(6)]],
    constant uint& numel_in [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)lines * (ulong)padded;
    if ((ulong)gid >= total) {
        return;
    }
    uint line = gid / padded;
    uint i = gid % padded;
    if (i >= dim_size) {
        keys[gid] = 0xFFFFFFFFFFFFFFFFUL;
        return;
    }
    uint in_pos = (line / inner) * dim_size * inner + (line % inner) + i * inner;
    if (in_pos >= numel_in) {
        return;
    }
    float v = input[in_pos];
    uint vk = value_key_msl(v);
    uint hi = (descending != 0) ? (~vk) : vk;
    ulong key = ((ulong)hi << 32) | (ulong)i;
    keys[gid] = key;
}

// `keys`（`lines` 本・各 `padded` 長）の 1 ライン分・1 比較ステップ
// （`sort_model::bitonic_step_u64_host` の逐語 GPU 実装）。`gid` は
// `lines * padded` 空間の添字。
kernel void bitonic_step_u64(
    device ulong* keys [[buffer(0)]],
    constant uint& j [[buffer(1)]],
    constant uint& k [[buffer(2)]],
    constant uint& padded [[buffer(3)]],
    constant uint& lines [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)lines * (ulong)padded;
    if ((ulong)gid >= total) {
        return;
    }
    uint line = gid / padded;
    uint i = gid % padded;
    uint ixj = i ^ j;
    if (ixj <= i || ixj >= padded) {
        return;
    }
    uint base = line * padded;
    ulong a = keys[base + i];
    ulong b = keys[base + ixj];
    bool ascending = (i & k) == 0;
    bool should_swap = ascending ? (a > b) : (a < b);
    if (should_swap) {
        keys[base + i] = b;
        keys[base + ixj] = a;
    }
}

// ソート済みキー配列の先頭 `out_len` 要素から `values`／`index` を
// 復元する（`sort_model::finalize_host` の逐語 GPU 実装）。`gid` は
// `lines * out_len` 空間の添字。
kernel void sort_finalize_f32(
    device const float* input [[buffer(0)]],
    device const ulong* keys [[buffer(1)]],
    device float* values [[buffer(2)]],
    device int* index [[buffer(3)]],
    constant uint& lines [[buffer(4)]],
    constant uint& dim_size [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    constant uint& padded [[buffer(7)]],
    constant uint& out_len [[buffer(8)]],
    constant uint& numel_in [[buffer(9)]],
    constant uint& numel_out [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)lines * (ulong)out_len;
    if ((ulong)gid >= total) {
        return;
    }
    uint line = gid / out_len;
    uint o = gid % out_len;
    // `keys` は `lines * padded` 長（本カーネル doc 参照）。ホスト側
    // （`sort.rs::MetalSort::run_sort_f32`）は起動前に `out_len <=
    // dim_size <= padded` を検証済みだが、REQ-8（カーネル境界検査
    // 規約）に従い本カーネルでも `o < padded` を読み出し前に自前で
    // 検査する（PR #1844 codex-review P0 是正）。
    if (o >= padded) {
        return;
    }
    ulong key = keys[(ulong)line * (ulong)padded + (ulong)o];
    uint idx = (uint)(key & 0xFFFFFFFFUL);
    if (idx >= dim_size) {
        return;
    }
    uint in_pos = (line / inner) * dim_size * inner + (line % inner) + idx * inner;
    uint out_pos = (line / inner) * out_len * inner + (line % inner) + o * inner;
    if (in_pos >= numel_in || out_pos >= numel_out) {
        return;
    }
    values[out_pos] = input[in_pos];
    index[out_pos] = (int)idx;
}
