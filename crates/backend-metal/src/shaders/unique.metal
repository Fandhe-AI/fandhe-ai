// unique（`torch.unique(input, sorted=True)` の values のみ相当。
// イシュー #1734。`crates/backend-cuda/src/kernels_unique.rs` の
// Metal 対応版。ホストモデルは `crate::unique_model`）。
//
// `crate::unique::MetalUnique`（`unique.rs`）から実行時コンパイルされ、
// `ops.rs::MetalBackendOps::unique` から呼ばれる。
//
// ---- ビットニックソート 1 ステップ（`bitonic_step_u32`）----
//
// グローバルメモリ上の `u32` キー配列に対しビットニックソートの
// 1 比較ステップを実行する（`unique.rs::MetalUnique::run_unique_f32`
// がホスト側ループで `j`／`k` を変えながら本カーネルを繰り返し
// エンコードする）。整数 compare/swap のみを行い浮動小数点演算を
// 一切含まないため決定的である。
//
// キー自体は `u32` の昇順が IEEE 754 totalOrder と一致するよう
// ホスト側で変換済み（`unique_model::total_order_key` 参照。カーネル
// 自体は単純な整数比較のみを行う）。
//
// ビットニックソートは 2 のべき乗長の配列を要求するため、呼び出し元は
// ホスト側で実データを `u32::MAX`（totalOrder 上の最大キー）で
// パディングして `padded`（2 のべき乗）長にしてから本カーネルを
// 繰り返しディスパッチする。カーネル自身は `padded` 全域（パディング
// 要素も含む）に対して一様にビットニックソートの比較を行い、パディング
// 値は最大キーであるため自然にソート後の末尾へ集まる（呼び出し元が
// 先頭 `n`〈実データ長〉要素だけを読み出す）。
//
// ---- REQ-8（カーネル境界検査規約）----
//
// `n` 引数は `padded`（ソート対象配列全長。2 のべき乗）を指す
// （実データ長ではない——グリッドの末尾 threadgroup が `padded` を
// 超えてスレッドを起動しうるための境界検査用）。`i`（自スレッドの
// 添字）・`ixj`（比較相手の添字）の両方をこの `n`（＝`padded`）未満か
// 検査する。最適化を理由に省略しない。

#include <metal_stdlib>
using namespace metal;

kernel void bitonic_step_u32(
    device uint* keys [[buffer(0)]],
    constant uint& j [[buffer(1)]],
    constant uint& k [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint i = gid;
    if (i >= n) {
        return;
    }
    uint ixj = i ^ j;
    if (ixj <= i || ixj >= n) {
        return;
    }
    uint a = keys[i];
    uint b = keys[ixj];
    bool ascending = (i & k) == 0;
    bool should_swap = ascending ? (a > b) : (a < b);
    if (should_swap) {
        keys[i] = b;
        keys[ixj] = a;
    }
}
