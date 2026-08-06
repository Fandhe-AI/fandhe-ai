// naive GEMM カーネル（TASK-1.8b・#39）。
//
// 移植元: docs/spec/03-poc/poc-v2-4-metal-gemm/code/rust/src/shaders/gemm.metal
// の `gemm_naive`（3 重ループ、タイル化なし。PoC の naive/tiled/simdgroup
// 3 段のうち naive 段のみを本イシューで productize する）。simdgroup 版は
// #40（TASK-1.8c）のスコープ。
//
// `crate::pipeline` が `include_str!` で本ファイルを取り込み、実行時に
// `MTLCompileOptions`（Safe/Precise。PoC-v2-5 実測構成）でコンパイルする。
// `crate::gemm` がバッファ・`Dims` を結線してディスパッチする。

#include <metal_stdlib>
using namespace metal;

// `crate::gemm::Dims`（`#[repr(C)]`）とレイアウトを一致させる（12 バイト）。
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
