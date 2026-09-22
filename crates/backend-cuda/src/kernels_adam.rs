//! デバイス上パラメータ更新（Adam・AdamW in-place）の CUDA C カーネル
//! ソース（NVRTC 実行時コンパイル用の静的文字列。イシュー #2069・
//! `docs/device-resident-update-design.md`「Adam／AdamW の常駐 step
//! 結線」節）。
//!
//! `adam.rs`（呼び出し元）は本モジュールの定数を `nvrtc::compile_ptx` に
//! 渡し `CudaFunction` を得る。`kernels_sgd.rs` と同じ理由でソースを
//! `nvcc` 事前コンパイルせず文字列のまま埋め込む（ビルド時に nvcc/CUDA
//! ヘッダを一切要求しない。「CUDA toolkit 非搭載環境でも `cargo build
//! --workspace` が成立する」契約を維持する。`.claude/rules/
//! deps-policy.md`）。
//!
//! # 意味論の正
//!
//! `fandhe_ai_autodiff::nn::optim::{adam::Adam::step, adamw::AdamW::step}`
//! （ホスト参照実装。`backend-cpu::ops::CpuBackendOps::adam_step_device`
//! がこれと bit 一致済み）が意味論の正。演算列（`g_eff` → `p_eff` →
//! `m`／`v` 更新 → `denom` → 減算）は `crates/backend-cpu/src/ops.rs::
//! adam_step_device` を逐語再現する。
//!
//! # FMA 契約・非縮約 intrinsic の使い分け（bit 一致の設計目標）
//!
//! NVRTC は `--fmad` 既定 ON（`nvrtc.rs::compile_ptx` doc コメント参照）
//! のため、何も対策しないと `p * decay_factor` → 減算のような mul→sub
//! 列がコンパイラ最適化で FMA へ縮約され、CPU（Rust は明示しない限り
//! 縮約しない）と丸めが変わりうる。よって:
//!
//! - CPU 参照実装が **明示的に** `f32::mul_add` を使う 2 箇所（`m`／`v`
//!   の指数移動平均更新）のみ `fmaf(...)` を使う。
//! - それ以外の全ての四則演算は **非縮約 intrinsic**
//!   （`__fmul_rn`／`__fadd_rn`／`__fsub_rn`／`__fdiv_rn`。CUDA C++
//!   Programming Guide の IEEE 754 round-to-nearest-even 丸め single
//!   instruction intrinsic）で書き、コンパイラによる暗黙の FMA 縮約を
//!   避ける。
//! - `sqrtf` は既定 `--prec-sqrt=true`（IEEE 754 正確丸め）のまま使う
//!   （`rsqrtf`／`__fdividef` 等の近似 intrinsic は使わない。イシュー
//!   #1105／#1893 の教訓を踏襲）。
//!
//! # 分岐構造の一致
//!
//! `AdamStepKind::Coupled`／`Decoupled` の分岐は呼び出し元
//! （`adam.rs::CudaAdam::run`）がホストで `use_coupled_wd` フラグへ
//! 事前計算してから渡す（CPU 参照実装の match 分岐と同じ条件式
//! `kind == Coupled && weight_decay != 0.0` をホスト側で 1 回だけ評価
//! する）。Decoupled 側の `p_eff = p * decay_factor` はカーネル内で
//! **無条件** に適用する（`weight_decay == 0.0` でも `decay_factor ==
//! 1.0` として同じ演算列を通る CPU 契約と同一。`backend_ops.rs::
//! AdamStepConfig` doc コメント参照）。
//!
//! # in-place 更新（ホスト往復排除）
//!
//! `kernels_sgd.rs` と同様、`param`／`m`／`v` を直接書き換える
//! （独立した `out` バッファを持たない）。呼び出し元はデバイス上に
//! 既に存在する `CudaSlice<f32>` へ直接ポインタを渡すため、`grad` の
//! みを毎ステップアップロードすれば `param`／`m`／`v` はステップを
//! またいでデバイス上に常駐できる。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `if (idx < numel)` の手動境界チェックを維持する（`kernels_sgd.rs`
//! と同じ理由。`.claude/rules/coding-rust.md`）。

/// 1 スレッドブロックあたりのスレッド数（1 次元）。`kernels_sgd::
/// SGD_BLOCK_DIM` と同じ値・同じ理由（PoC 実測なしの保守的な固定値）。
pub const ADAM_BLOCK_DIM: u32 = 256;

/// Adam・AdamW 1 ステップの in-place 更新カーネル（f32）。
///
/// パラメータ:
/// - `param`（読み書き）: 更新対象。
/// - `grad`（読み取り専用）: このステップの勾配。
/// - `m`／`v`（読み書き）: 1 次／2 次モーメントバッファ。
/// - `numel`: 要素数。
/// - `beta1`／`beta2`／`eps`／`weight_decay`／`decay_factor`／
///   `step_size`／`bias_correction2_sqrt`: `AdamStepConfig`
///   （`tensor-core::backend_ops`）と同一のハイパーパラメータ
///   （`step_size`／`bias_correction2_sqrt`／`decay_factor` はホストで
///   `beta^t` 等を事前計算済みの値。カーネル内では再計算しない）。
/// - `kind_decoupled`／`use_coupled_wd`: 0/1 の bool 相当フラグ
///   （NVRTC 側で `bool` 型を安定して扱うため `int` にしている。
///   `kernels_sgd.rs` と同じ慣習）。`kind_decoupled` が真なら
///   `AdamStepKind::Decoupled`（`p_eff = p * decay_factor`）、偽なら
///   `AdamStepKind::Coupled`（`p_eff = p`）。`use_coupled_wd` は
///   `Coupled && weight_decay != 0.0` をホストで事前評価した値
///   （`g_eff` に coupled L2 decay を適用するか）。
pub const ADAM_STEP_F32: &str = r#"
extern "C" __global__ void adam_step_f32(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    int numel,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float decay_factor,
    float step_size,
    float bias_correction2_sqrt,
    int kind_decoupled,
    int use_coupled_wd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float p = param[idx];
        float g = grad[idx];

        float g_eff = g;
        if (use_coupled_wd) {
            g_eff = fmaf(weight_decay, p, g);
        }

        float p_eff = p;
        if (kind_decoupled) {
            p_eff = __fmul_rn(p, decay_factor);
        }

        float one_minus_beta1 = __fsub_rn(1.0f, beta1);
        float t1 = __fmul_rn(one_minus_beta1, g_eff);
        float m_new = fmaf(beta1, m[idx], t1);

        float one_minus_beta2 = __fsub_rn(1.0f, beta2);
        float g_eff_sq = __fmul_rn(g_eff, g_eff);
        float t2 = __fmul_rn(one_minus_beta2, g_eff_sq);
        float v_new = fmaf(beta2, v[idx], t2);

        m[idx] = m_new;
        v[idx] = v_new;

        float denom = __fadd_rn(__fdiv_rn(sqrtf(v_new), bias_correction2_sqrt), eps);
        float step = __fdiv_rn(__fmul_rn(step_size, m_new), denom);
        param[idx] = __fsub_rn(p_eff, step);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// カーネルソースの丸め契約後退検出（イシュー #1105／#1893 の教訓を
    /// 踏襲した後退検出テスト。実 GPU を要さず Linux CI で常時 green に
    /// なる契約テスト）。
    ///
    /// - CPU が `f32::mul_add` を使う 2 箇所（`m`／`v` 更新）は `fmaf(`
    ///   を使うこと。
    /// - それ以外の四則演算は非縮約 intrinsic を使うこと（`__fmul_rn(`／
    ///   `__fadd_rn(`／`__fsub_rn(`／`__fdiv_rn(` がいずれも出現する）。
    /// - `sqrtf(` を使い、近似 intrinsic（`rsqrtf(`／`__fdividef(`）を
    ///   使わないこと。
    /// - ビルド時に CUDA ヘッダを要求しない契約（`#include`／`INFINITY`
    ///   を使わない）。
    #[test]
    fn adam_step_f32_source_uses_non_fused_intrinsics_and_no_headers() {
        let src = ADAM_STEP_F32;
        assert!(src.contains("fmaf("), "m/v 更新は fmaf を使うはず");
        assert!(
            src.contains("__fmul_rn("),
            "非縮約乗算 intrinsic を使うはず"
        );
        assert!(
            src.contains("__fadd_rn("),
            "非縮約加算 intrinsic を使うはず"
        );
        assert!(
            src.contains("__fsub_rn("),
            "非縮約減算 intrinsic を使うはず"
        );
        assert!(
            src.contains("__fdiv_rn("),
            "非縮約除算 intrinsic を使うはず"
        );
        assert!(src.contains("sqrtf("), "IEEE 正確丸め sqrtf を使うはず");
        assert!(
            !src.contains("rsqrtf("),
            "近似 intrinsic rsqrtf は使わないはず"
        );
        assert!(
            !src.contains("__fdividef("),
            "近似 intrinsic __fdividef は使わないはず"
        );
        assert!(!src.contains("#include"), "CUDA ヘッダを要求しないはず");
        assert!(!src.contains("INFINITY"), "INFINITY マクロを使わないはず");
        assert!(
            src.contains("if (idx < numel)"),
            "REQ-8 の手動境界チェックを維持するはず"
        );
    }
}
