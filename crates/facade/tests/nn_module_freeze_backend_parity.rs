//! 層別 `requires_grad` 凍結（`fandhe_ai_autodiff::nn::Module::freeze`／
//! `set_requires_grad`）の facade 到達経路 parity テスト（イシュー
//! #2137。`no_grad_detach_backend_parity.rs`／`nn_conv_backend_parity.rs`
//! と同型）。
//!
//! `requires_grad` は算術を一切伴わないテープ側メタデータ
//! （`crates/autodiff/src/tape.rs::TapeNode::requires_grad`）のため、
//! 同一バックエンド内で「凍結あり」と「凍結なし」の出力・非凍結
//! パラメータの勾配が **bit 完全一致**することを確認する
//! （`.claude/rules/coding-rust.md` の複合判定の対象外——両者とも
//! 同じバックエンド演算列を通した値の比較であり、本テストが検証する
//! のは凍結の有無が forward の演算列・累積順序を一切変えないことその
//! ものである）。
//!
//! **facade 公開面（意図的な非変更）**: `fandhe_ai::nn::Module` 自体は
//! まだ facade へ公開されていない（イシュー #2133 OPEN・
//! `docs/facade-nn-module-exposure-decision.md` §10 承認待ち）。本
//! テストは内部クレート `fandhe_ai_autodiff::nn`・具体バックエンド
//! クレート（`fandhe_ai_backend_cpu`／`fandhe_ai_backend_cuda`／
//! `fandhe_ai_backend_metal`）を facade の依存経由で直接使う
//! （`nn_conv_backend_parity.rs` の `RawTape` 構成と同じ）ため、facade
//! の公開面（`api_surface.rs` の否定ガード）には影響しない。
//!
//! - 属性なし: `RawTape::new_with_ops(Box::new(CpuBackendOps::new()))`
//!   上で CI 実行する。
//! - `#[ignore]`: `CudaBackendOps::new(0)`／`MetalBackendOps::new()`
//!   （`cfg(target_os = "macos")` 限定）の同経路。実機実測は本エージェ
//!   ント実行環境に到達手段が無いため未実施（PR 本文に申し送り）。

use fandhe_ai_autodiff::Tape as RawTape;
use fandhe_ai_autodiff::nn::{Linear, Module};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn contiguous_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// 同一バックエンド（`ops`）上で、凍結あり／なしの `Linear` の
/// 出力・x 勾配が bit 一致し、凍結側の weight／bias 勾配は
/// `Err(GradientTrackingDisabled)` を返すことを確認する共通本体。
/// `ops` を 2 回分（凍結あり／なしの各テープ用）呼び出すクロージャで
/// 受け取る（`Box<dyn BackendOps + Send>` は `Tape::new_with_ops` に
/// 一度渡すと所有権が移るため）。
fn assert_freeze_parity_with(make_ops: impl Fn() -> Box<dyn BackendOps + Send>) {
    let x_data = t(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]);

    let tape_unfrozen = RawTape::new_with_ops(make_ops());
    let linear_unfrozen = Linear::new(2, 3, true, 7).expect("seed=7 は有効");
    let x_unfrozen = tape_unfrozen.var(&x_data);
    let vars_unfrozen = linear_unfrozen.bind(&tape_unfrozen);
    let out_unfrozen = vars_unfrozen
        .forward(&x_unfrozen)
        .expect("shape が一致するため forward は成功する");
    let loss_unfrozen = out_unfrozen.sum(None).expect("全軸縮約は失敗しない");
    let grads_unfrozen = tape_unfrozen
        .backward(&loss_unfrozen)
        .expect("x が追跡対象のため成功する");
    let dx_unfrozen = grads_unfrozen
        .get(&x_unfrozen)
        .expect("x は requires_grad=true")
        .expect("x は loss に到達する");

    let tape_frozen = RawTape::new_with_ops(make_ops());
    let mut linear_frozen = Linear::new(2, 3, true, 7).expect("同じ seed で同じ初期値");
    linear_frozen
        .freeze()
        .expect("Linear は set_requires_grad をオーバーライド済み");
    let x_frozen = tape_frozen.var(&x_data);
    let vars_frozen = linear_frozen.bind(&tape_frozen);
    let out_frozen = vars_frozen
        .forward(&x_frozen)
        .expect("shape が一致するため forward は成功する");

    assert_eq!(
        contiguous_bits(&out_unfrozen.to_tensor()),
        contiguous_bits(&out_frozen.to_tensor()),
        "freeze は forward の値に影響しない（同一バックエンド内で bit 一致）"
    );

    let loss_frozen = out_frozen.sum(None).expect("全軸縮約は失敗しない");
    let grads_frozen = tape_frozen
        .backward(&loss_frozen)
        .expect("x が追跡対象のため成功する");
    let dx_frozen = grads_frozen
        .get(&x_frozen)
        .expect("x は requires_grad=true")
        .expect("x は loss に到達する");

    assert_eq!(
        contiguous_bits(dx_unfrozen),
        contiguous_bits(dx_frozen),
        "凍結の有無で x の勾配は同一バックエンド内で bit 一致するはず"
    );

    let err_w = grads_frozen.get(&vars_frozen.weight).unwrap_err();
    assert!(matches!(
        err_w,
        fandhe_ai_autodiff::AutodiffError::GradientTrackingDisabled
    ));
    let err_b = grads_frozen
        .get(vars_frozen.bias.as_ref().expect("bias=true で構築した"))
        .unwrap_err();
    assert!(matches!(
        err_b,
        fandhe_ai_autodiff::AutodiffError::GradientTrackingDisabled
    ));
}

/// CPU 本番 ops（`CpuBackendOps`）上での凍結 parity（CI で実行）。
#[test]
fn cpu_freeze_linear_output_and_input_grad_match_unfrozen() {
    assert_freeze_parity_with(|| Box::new(CpuBackendOps::new()));
}

// --- 実機横断（`#[ignore]`。CUDA／Metal） ---
//
// `no_grad_detach_backend_parity.rs` と同じ理由で Metal 依存テストのみ
// `cfg(target_os = "macos")` でコンパイル自体を限定する。実機実測は
// 本エージェント実行環境に到達手段が無いため未実施のまま（PR 本文に
// 申し送り）。

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_freeze_linear_output_and_input_grad_match_unfrozen() {
    assert_freeze_parity_with(|| Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(0)));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_freeze_linear_output_and_input_grad_match_unfrozen() {
    assert_freeze_parity_with(|| Box::new(fandhe_ai_backend_metal::MetalBackendOps::new()));
}
