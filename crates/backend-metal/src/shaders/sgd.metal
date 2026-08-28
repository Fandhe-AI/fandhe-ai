// デバイス上パラメータ更新（SGD in-place）の MSL カーネルソース（イシュー
// #935・CUDA 側 `backend-cuda::kernels_sgd`〈同イシュー〉の Metal 対応版）。
//
// `crate::sgd` が `include_str!` で本ファイルを取り込み、
// `MTLCompileOptions`（Safe/Precise。`crate::pipeline::compile_options`。
// `shaders/elementwise.metal` と同一設定）で実行時コンパイルする。1
// スレッド = 1 要素の 1 次元グリッドで `if (idx < numel)` の手動境界
// チェックを維持する（REQ-8。`.claude/rules/coding-rust.md`）。
//
// # 意味論の正
//
// `fandhe_ai_autodiff::optim::sgd::Sgd::step`（ホスト参照実装）が意味論の
// 正。項順序（weight_decay → momentum〈`is_first_step` で `b ← g` 分岐〉→
// nesterov → 減算）を厳密に踏襲し、`fma`（単精度 FMA）を用いる（CPU
// `f32::mul_add`／CUDA `fmaf` と同じ FMA 契約統一方針。`.claude/rules/
// coding-rust.md`）。
//
// # in-place 更新（ホスト往復排除。本イシューの主目的）
//
// `elementwise.metal` の 5 カーネルと異なり、本カーネルは独立した `out`
// バッファを持たない: `param` を直接書き換える（`velocity` も momentum
// 有効時は直接書き換える）。`StorageModeShared`（Apple Silicon の UMA）の
// ため CUDA のような明示的な非同期転送は元より不要だが、本カーネルの目的
// は毎ステップのホスト側再アップロード（`param` の再構築）そのものを
// 排除することにある（`crate::memory::MetalMemory::upload` を毎ステップ
// 呼ばない）。

#include <metal_stdlib>
using namespace metal;

kernel void sgd_step_f32(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* velocity [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& momentum [[buffer(5)]],
    constant float& dampening [[buffer(6)]],
    constant float& weight_decay [[buffer(7)]],
    constant int& nesterov [[buffer(8)]],
    constant int& is_first_step [[buffer(9)]],
    constant int& use_momentum [[buffer(10)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        float p = param[idx];
        float g = grad[idx];
        if (weight_decay != 0.0f) {
            g = fma(weight_decay, p, g);
        }
        if (use_momentum) {
            float prev = velocity[idx];
            float b;
            if (is_first_step) {
                b = g;
            } else {
                b = fma(momentum, prev, (1.0f - dampening) * g);
            }
            velocity[idx] = b;
            if (nesterov) {
                g = fma(momentum, b, g);
            } else {
                g = b;
            }
        }
        param[idx] = p - lr * g;
    }
}
