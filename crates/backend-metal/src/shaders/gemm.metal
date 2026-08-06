// GEMM カーネル 3 段（naive・tiled・simdgroup_matrix。TASK-1.8b・#39〜
// TASK-1.8c・#40）。
//
// 移植元: docs/spec/03-poc/poc-v2-4-metal-gemm/code/rust/src/shaders/gemm.metal
// の `gemm_naive`/`gemm_tiled`/`gemm_simdgroup`。naive 段は #39 で
// productize 済み。本ファイルは #40 で tiled・simdgroup を追加し 3 段
// すべてを揃えた。
//
// `crate::pipeline` が `include_str!` で本ファイルを取り込み、実行時に
// `MTLCompileOptions`（Safe/Precise。PoC-v2-5 実測構成）でコンパイルする。
// `crate::gemm` がバッファ・`Dims` を結線してディスパッチする
// （`GemmVariant::Simdgroup` の場合、`crate::pad` で 8 の倍数へパディング
// 済みの実効次元を `Dims` として渡す契約）。

#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

// `crate::gemm::Dims`（`#[repr(C)]`）とレイアウトを一致させる（12 バイト）。
// `GemmVariant::Simdgroup` の場合、m/n/k は `crate::pad::pad8` によって
// 8 の倍数に切り上げられた実効次元（呼び出し元の元形状ではない）。
struct Dims {
    uint m;
    uint n;
    uint k;
};

// 素朴な 3 重ループ（タイル化なし）。gid.y = 行、gid.x = 列。
//
// 手動境界チェック（`gid.y >= dims.m || gid.x >= dims.n` で早期 return）は
// 性能上の下限・最適化の達成を理由に省略しない（REQ-8・
// `.claude/rules/coding-rust.md`「カーネル実装の境界検査」）。dispatch 側
// （`crate::gemm::gemm_naive`）は grid を `div_ceil(16)` で切り上げるため、
// m・n が threadgroup サイズ（16）の倍数でない場合にこの境界チェックが
// 実際に効く（はみ出したスレッドの書き込みを防ぐ）。
//
// 内積の丸め方針（FMA 契約。REQ-2）: CPU 参照実装
// （`backend_cpu::parity::matmul_reference_fma`）は `f32::mul_add`・k 昇順
// 逐次加算を用いる。ここでも `fma()` を明示し、コンパイラの自動 FMA 融合
// （かかる場合とかからない場合がある）に丸め方針を委ねない
// （PoC の `acc += a*b` から変更。PoC-v2-5 の K=4096 ストレスケースで
// mul_add 化により CPU/GPU 間 fail_cells=0 を実測確認済み）。
kernel void gemm_naive(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant Dims& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.y >= dims.m || gid.x >= dims.n) {
        return;
    }
    float acc = 0.0;
    for (uint p = 0; p < dims.k; p++) {
        acc = fma(a[gid.y * dims.k + p], b[p * dims.n + gid.x], acc);
    }
    c[gid.y * dims.n + gid.x] = acc;
}

// threadgroup 共有メモリによるタイル化（TASK-1.8c・#40）。WGSL 版
// （`docs/spec/.../gemm.wgsl`）と同一のタイルサイズ・アルゴリズムで、
// 「同じアルゴリズムを Metal 直接で書いた場合の差」を切り分けられるよう
// にしている（PoC-v2-4 README「計測結果」。size=4096 で naive 比
// 約 1.67 倍）。
//
// 手動境界チェック（REQ-8）: タイルロード時は `(row < m && a_col < k)` /
// `(b_row < k && col < n)` の条件を満たさない要素を 0 埋めし、ストア時は
// `row < m && col < n` を満たす場合のみ書き込む。m・n・k が TILE（16）の
// 倍数でない場合にこの境界チェックが実際に効く（タイル端のはみ出し
// アクセス・書き込みを防ぐ。性能上の下限・最適化を理由に省略しない）。
//
// 内積の丸め方針（FMA 契約。REQ-2）: naive 段と同じ理由で `acc += a*b`
// （PoC 原文）ではなく `fma()` を明示する（CPU 参照実装 `f32::mul_add`
// との丸め方針統一。PoC-v2-5 実測確認済み）。
//
// バッファオフセットの 64-bit 化（PR #246 Bugbot 指摘対応）: `row * k`・
// `b_row * n`・`row * n` は `dims.m`/`n`/`k` が個々に `u32::MAX` 未満でも
// 積が `u32::MAX` を超えうる（例: m=n=k=100000 は各次元は収まるが
// `row * n` は最大 約 1.0e10 で 32-bit `uint` 溢れによりオフセットが
// ラップアラウンドし、書き込み先行が黙って不正になる）。`row`/`a_col`/
// `b_row`/`col` を `size_t`（64-bit）へ昇格してから乗算し、`u32::MAX` 超の
// 有効な行列サイズでも溢れないようにする。
constant uint TILE = 16;

kernel void gemm_tiled(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant Dims& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float tile_a[TILE][TILE];
    threadgroup float tile_b[TILE][TILE];

    uint m = dims.m, n = dims.n, k = dims.k;
    uint row = gid.y, col = gid.x;
    float acc = 0.0;

    uint num_tiles = (k + TILE - 1) / TILE;
    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE + lid.x;
        uint b_row = t * TILE + lid.y;

        tile_a[lid.y][lid.x] =
            (row < m && a_col < k) ? a[(size_t)row * (size_t)k + (size_t)a_col] : 0.0;
        tile_b[lid.y][lid.x] =
            (b_row < k && col < n) ? b[(size_t)b_row * (size_t)n + (size_t)col] : 0.0;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE; i++) {
            acc = fma(tile_a[lid.y][i], tile_b[i][lid.x], acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < m && col < n) {
        c[(size_t)row * (size_t)n + (size_t)col] = acc;
    }
}

// `simdgroup_matrix`（8x8 ハードウェア行列演算命令。TASK-1.8c・#40）。
//
// 1 threadgroup = 1 simdgroup（32 スレッド）とし、各 simdgroup が C の
// 8x8 タイルを 1 つ担当する。`simdgroup_load`/`simdgroup_multiply_
// accumulate`/`simdgroup_store` は Apple GPU の行列専用命令にディスパッチ
// され、WGSL では表現できない（`crate::lib` クレートコメント・PoC-v2-4
// README「経路選定の比較判断」節が objc2-metal 直接経路を選定した中心的
// 根拠）。命令自体が Apple GPU の既定 FMA 契約で累積するため、CPU 参照
// 実装（`f32::mul_add`）との丸め方針統一は本命令のハードウェア契約に
// 委ねる（PoC-v2-5 の K=4096 ストレスケースで実測確認済み。naive/tiled
// 段のような `fma()` の明示呼び出しは不要）。
//
// **制約と手動境界チェック（REQ-8）**: `dims.m`/`n`/`k` は
// `crate::gemm::MetalGemm::dispatch_variant` が `crate::pad::pad8` で
// 8 の倍数に切り上げた実効次元であり、呼び出し元はこの契約を守って
// バッファ（A・B・C）も実効次元ぶん確保・0 パディング済みでなければ
// ならない。この前提が成立する限り 8x8 タイル内部の `simdgroup_load`/
// `_store` は常に確保領域内に収まるが、性能上の下限・最適化の達成を
// 理由に境界チェック自体を省略しない方針（REQ-8）に従い、タイル原点
// （`row0`/`col0`）が実効次元を超える場合の早期 return を明示的に残す
// （通常の dispatch では `n_eff/8 × m_eff/8` の grid により到達しない
// 防御的チェックだが、将来 dispatch 側の grid 計算に誤りがあっても
// 未確保領域への書き込みを起こさないための境界検査として維持する）。
//
// バッファオフセットの 64-bit 化（PR #246 Bugbot 指摘対応）: `row0 * dims.k`・
// `p0 * dims.n`・`row0 * dims.n` は `dims.m`/`n`/`k` が個々に `u32::MAX`
// 未満でも積が `u32::MAX` を超えうる（tiled 段と同じ懸念。上記コメント
// 参照）。`row0`/`col0`/`p0` を `size_t`（64-bit）へ昇格してから乗算し、
// `u32::MAX` 超の有効な行列サイズでもポインタオフセットが溢れないように
// する。
kernel void gemm_simdgroup(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant Dims& dims [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint row0 = tgid.y * 8;
    uint col0 = tgid.x * 8;

    if (row0 >= dims.m || col0 >= dims.n) {
        return;
    }

    simdgroup_float8x8 acc(0.0f);
    uint k_tiles = dims.k / 8;
    for (uint t = 0; t < k_tiles; t++) {
        uint p0 = t * 8;
        simdgroup_float8x8 a_tile;
        simdgroup_float8x8 b_tile;
        simdgroup_load(a_tile, a + (size_t)row0 * (size_t)dims.k + (size_t)p0, dims.k);
        simdgroup_load(b_tile, b + (size_t)p0 * (size_t)dims.n + (size_t)col0, dims.n);
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
    }
    simdgroup_store(acc, c + (size_t)row0 * (size_t)dims.n + (size_t)col0, dims.n);
}
