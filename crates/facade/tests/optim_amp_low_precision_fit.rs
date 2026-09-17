//! 低精度 forward（`Linear` 層限定・#1960）と `GradScaler`（AMP・#1721）
//! を `compat::Sequential::compile_with_amp`（イシュー #1961・親 #1958）
//! で統合した学習ループの**例**（CI 実行可能なサンプル兼テスト。
//! `crates/autodiff/src/create_graph.rs` 系の HVP 例〈#2003〉と同じ
//! 「例としてのテスト」慣行に従う）。
//!
//! **本ファイルは `fandhe_ai` と `bench_harness::rng` 以外を import
//! しない**（`crates/facade/tests/optim_amp_train_loop.rs` と同型の
//! 契約。レビュー・CI では `grep -n
//! "fandhe_ai_autodiff\|fandhe_ai_tensor_core"
//! tests/optim_amp_low_precision_fit.rs` がヒット 0 件であることを
//! 確認する）——`compile_with_amp` の低精度 dtype 指定は facade
//! ローカルの [`fandhe_ai::compat::AmpDType`] のみで完結するため、
//! `fandhe_ai_tensor_core::ScalarDType` を呼び出し側が直接扱う必要は
//! ない。
//!
//! 適用順序契約（forward の低精度化 → scale_loss → backward → unscale
//! → skip 判定 → optimizer step → update）の正は `compile_with_amp` doc
//! （`crates/facade/src/compat/training.rs`）を参照。bit 一致検証・
//! skip／backoff 経路の検証は `compat_sequential_fit_amp.rs` を参照
//! （本ファイルは「使い方の例」として最終 loss の収束のみを見る）。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない（`fit` は
//! 常に CPU `tape()` 固定）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::{AmpConfig, AmpDType, FitConfig, Loss, Optimizer, Sequential};
use fandhe_ai::optim::SgdConfig;

const BATCH: usize = 8;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (
        Tensor::new(x, &[BATCH, D_IN])
            .unwrap_or_else(|e| panic!("test fixture: x の shape 構築に失敗: {e}")),
        Tensor::new(y, &[BATCH, D_OUT])
            .unwrap_or_else(|e| panic!("test fixture: y の shape 構築に失敗: {e}")),
    )
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap_or_else(|e| panic!("test fixture: 層 1 の構築に失敗: {e}"))
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap_or_else(|e| panic!("test fixture: 層 2 の構築に失敗: {e}"))
}

/// **学習ループ例**: `Sequential::compile_with_amp` → `fit` の 2 呼び出し
/// だけで、Linear 層 forward を f16 で計算しつつ f32 master weight・
/// `GradScaler` による損失スケーリングを併用した学習ループが組める
/// ことを示す。
#[test]
fn amp_f16_fit_converges_via_facade_only() {
    const EPOCHS: usize = 50;
    let (x, y) = gen_regression_data(SEED_DATA);

    let mut model = build_model();
    model
        .compile_with_amp(
            Optimizer::Sgd(SgdConfig::new(0.05)),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16),
        )
        .unwrap_or_else(|e| panic!("test fixture: compile_with_amp が失敗した: {e}"));

    let history = model
        .fit(&x, &y, FitConfig::new(EPOCHS, BATCH))
        .unwrap_or_else(|e| panic!("test fixture: fit が失敗した: {e}"));

    assert_eq!(history.loss.len(), EPOCHS);
    let initial = history.loss[0];
    let final_loss = *history
        .loss
        .last()
        .unwrap_or_else(|| unreachable!("history.loss は空でない"));
    assert!(final_loss.is_finite(), "final loss が非有限: {final_loss}");
    assert!(
        final_loss < initial,
        "loss did not decrease: initial={initial} final={final_loss}"
    );

    // `growth_interval`（既定 2000）に対して EPOCHS=50 は十分小さいため
    // growth は一度も起きず、backoff（overflow）も本シナリオでは起き
    // ない前提のもとで scale は既定 `init_scale`（65536.0）のまま。
    assert_eq!(
        model.amp_loss_scale(),
        Some(65536.0),
        "既定 GradScalerConfig では 50 epoch 以内に growth／backoff は \
         起きないはず"
    );
}
