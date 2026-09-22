// デバイス上パラメータ更新（Adam・AdamW in-place）の MSL カーネルソース
// （イシュー #2070・`docs/device-resident-update-design.md`「Adam／
// AdamW の常駐 step 結線」節の追補。CUDA 側
// `backend-cuda::kernels_adam`〈イシュー #2069〉の Metal 対応版）。
//
// `crate::adam` が `include_str!` で本ファイルを取り込み、
// `MTLCompileOptions`（Safe/Precise。`crate::pipeline::compile_options`。
// `shaders/sgd.metal` と同一設定）で実行時コンパイルする。1 スレッド =
// 1 要素の 1 次元グリッドで `if (idx < numel)` の手動境界チェックを
// 維持する（REQ-8。`.claude/rules/coding-rust.md`）。
//
// # 意味論の正
//
// `fandhe_ai_autodiff::nn::optim::{adam::Adam::step, adamw::AdamW::step}`
// （ホスト参照実装。`backend-cpu::ops::CpuBackendOps::adam_step_device`
// がこれと bit 一致済み）が意味論の正。演算列（`g_eff` → `p_eff` →
// `m`／`v` 更新 → `denom` → 減算）は `crates/backend-cpu/src/ops.rs::
// adam_step_device`・`crates/backend-cuda/src/kernels_adam.rs::
// ADAM_STEP_F32` を逐語再現する（括弧の結合順序も同一に揃える）。
//
// # FP 縮約禁止（bit 一致の設計目標）
//
// MSL の既定 FP 縮約モードは `fast`（文をまたぐ FMA 縮約を許可。MSL
// 仕様 v4.1 p.16-17）で、`MTLMathMode::Safe`（`crate::pipeline::
// compile_options` が指定）は縮約を `on`（同一文内のみ）へ下げるが
// `off` にはしない。CPU 参照実装が `f32::mul_add` を**明示的に**使う
// 2 箇所（`m`／`v` の指数移動平均更新。下記 `fma(` 呼び出し）以外を
// コンパイラが暗黙に FMA 縮約してしまうと CPU と丸めが変わりうる
// （`p * decay_factor` の直後の減算が典型例）。よって本ファイル冒頭で
// `#pragma clang fp contract(off)`（ファイルスコープ）を明示し、CUDA
// 側の非縮約 intrinsic（`__fmul_rn` 等）方針の Metal 対応とする。
//
// # 分岐構造の一致
//
// `AdamStepKind::Coupled`／`Decoupled` の分岐は呼び出し元
// （`crate::adam_model::validate_adam_step_inputs`）がホストで
// `use_coupled_wd`／`kind_decoupled` フラグへ事前計算してから渡す
// （CPU 参照実装の match 分岐と同じ条件式 `kind == Coupled &&
// weight_decay != 0.0` をホスト側で 1 回だけ評価する）。Decoupled 側の
// `p_eff = p * decay_factor` はカーネル内で**無条件**に適用する
// （`weight_decay == 0.0` でも `decay_factor == 1.0` として同じ演算列を
// 通る CPU 契約と同一。`AdamStepConfig` doc コメント参照）。
//
// # in-place 更新（ホスト往復排除）
//
// `shaders/sgd.metal` と同様、`param`／`m`／`v` を直接書き換える
// （独立した `out` バッファを持たない）。呼び出し元はデバイス上に既に
// 存在する `MetalBuffer` へ直接束縛するため、`grad` のみを毎ステップ
// アップロードすれば `param`／`m`／`v` はステップをまたいでデバイス上
// に常駐できる。

#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

kernel void adam_step_f32(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant uint& numel [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& decay_factor [[buffer(9)]],
    constant float& step_size [[buffer(10)]],
    constant float& bias_correction2_sqrt [[buffer(11)]],
    constant int& kind_decoupled [[buffer(12)]],
    constant int& use_coupled_wd [[buffer(13)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < numel) {
        float p = param[idx];
        float g = grad[idx];

        float g_eff = g;
        if (use_coupled_wd) {
            g_eff = fma(weight_decay, p, g);
        }

        float p_eff = p;
        if (kind_decoupled) {
            p_eff = p * decay_factor;
        }

        float t1 = (1.0f - beta1) * g_eff;
        float m_new = fma(beta1, m[idx], t1);

        float t2 = ((1.0f - beta2) * g_eff) * g_eff;
        float v_new = fma(beta2, v[idx], t2);

        m[idx] = m_new;
        v[idx] = v_new;

        float denom = (precise::sqrt(v_new) / bias_correction2_sqrt) + eps;
        float step = (step_size * m_new) / denom;
        param[idx] = p_eff - step;
    }
}
