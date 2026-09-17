//! `compat::Sequential::compile_with_amp`（イシュー #1961・親 #1958）の
//! bit 一致・構造テスト。
//!
//! `compat_sequential_fit.rs` と異なり、本ファイルは低精度 forward
//! （`fandhe_ai_autodiff::nn::linear_forward_low_precision`。#1960）と
//! `ScalarDType`（`fandhe_ai_tensor_core`）を直接 import する——手動
//! ループ比較対象を「`fandhe_ai` のみ」ではなく「`compile_with_amp` が
//! 内部で呼ぶのと同一の関数を直接呼ぶ」形で組み立て、`fit` 側の統合
//! 経路が既存 `GradScaler`／`linear_forward_low_precision` 単体と
//! **bit 完全一致**であることを検証する（AC-a。`docs/autodiff-low-
//! precision-linear-design.md` §7・`docs/compat-fit-evaluate-design.md`
//! の bit 一致検証方針と同型）。
//!
//! **AC-b**（MNIST 規模・REQ-2 統一複合判定）は
//! `mnist_amp_low_precision_parity.rs` を参照。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない
//! （`fit` は常に CPU `tape()` 固定）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::{AmpConfig, AmpDType, FitConfig, Loss, Optimizer, Sequential};
use fandhe_ai::optim::{GradScaler, GradScalerConfig, Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, Tensor};
use fandhe_ai_autodiff::nn::linear_forward_low_precision;
use fandhe_ai_tensor_core::{Activation, ScalarDType};

const N: usize = 16;
const D_IN: usize = 4;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 2;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(N * D_IN);
    let y = rng.fill_vec(N * D_OUT);
    (
        Tensor::new(x, &[N, D_IN])
            .unwrap_or_else(|e| panic!("test fixture: x の shape 構築に失敗: {e}")),
        Tensor::new(y, &[N, D_OUT])
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

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

fn params_bit_exact(a: &[&Tensor<f32>], b: &[&Tensor<f32>]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (x, y) in a.iter().zip(b.iter()) {
        let xd = x
            .contiguous()
            .as_slice()
            .expect("test fixture: contiguous 化済み")
            .to_vec();
        let yd = y
            .contiguous()
            .as_slice()
            .expect("test fixture: contiguous 化済み")
            .to_vec();
        if xd.len() != yd.len() {
            return false;
        }
        for (xv, yv) in xd.iter().zip(yd.iter()) {
            if xv.to_bits() != yv.to_bits() {
                return false;
            }
        }
    }
    true
}

// =====================================================================
// 1. compile()（AMP なし）は AMP 配線導入前と bit 同一のまま
// =====================================================================

#[test]
fn fit_without_amp_is_unchanged_by_amp_wiring() {
    const STEPS: usize = 5;
    const LR: f32 = 0.05;
    let (x, y) = gen_regression_data(SEED_DATA);

    let mut manual_model = build_model();
    let mut manual_sgd = Sgd::new(SgdConfig::new(LR))
        .unwrap_or_else(|e| panic!("test fixture: Sgd::new が失敗した: {e}"));
    let mut manual_losses = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = manual_model.bind(&tape);
            let x_var = tape.var(&x);
            let y_var = tape.var_no_grad(&y);

            let pred = bound
                .forward(&tape, &x_var)
                .unwrap_or_else(|e| panic!("test fixture: forward が失敗した: {e}"));
            let loss = pred
                .mse_loss(&y_var)
                .unwrap_or_else(|e| panic!("test fixture: mse_loss が失敗した: {e}"));
            manual_losses.push(scalar(&loss.to_tensor()));

            let grads = tape
                .backward(&loss)
                .unwrap_or_else(|e| panic!("test fixture: backward が失敗した: {e}"));
            let grad_refs = bound
                .trainable_grads(&grads)
                .unwrap_or_else(|e| panic!("test fixture: trainable_grads が失敗した: {e}"));
            let param_refs = manual_model.trainable_parameters();
            manual_sgd
                .step(&param_refs, &grad_refs)
                .unwrap_or_else(|e| panic!("test fixture: Sgd::step が失敗した: {e}"))
        };
        manual_model
            .apply_parameters(updated)
            .unwrap_or_else(|e| panic!("test fixture: apply_parameters が失敗した: {e}"));
    }

    let mut fit_model = build_model();
    assert_eq!(fit_model.amp_loss_scale(), None);
    fit_model
        .compile(Optimizer::Sgd(SgdConfig::new(LR)), Loss::Mse)
        .unwrap_or_else(|e| panic!("test fixture: compile が失敗した: {e}"));
    let history = fit_model
        .fit(&x, &y, FitConfig::new(STEPS, N))
        .unwrap_or_else(|e| panic!("test fixture: fit が失敗した: {e}"));
    // AMP を使わない compile() では amp_loss_scale() は常に None
    // （実装計画 §2.2「AmpState」節）。
    assert_eq!(fit_model.amp_loss_scale(), None);

    assert_eq!(history.loss.len(), manual_losses.len());
    for (m, f) in manual_losses.iter().zip(history.loss.iter()) {
        assert_eq!(
            m.to_bits(),
            f.to_bits(),
            "loss diverged: manual={m} fit={f}"
        );
    }

    let manual_params = manual_model.trainable_parameters();
    let fit_params = fit_model.trainable_parameters();
    assert!(
        params_bit_exact(&manual_params, &fit_params),
        "AMP 配線導入により AMP 未使用時の fit() 挙動が変わった"
    );
}

// =====================================================================
// 2. compile_with_amp() は手動 GradScaler + linear_forward_low_precision
//    ループと bit 完全一致（F16・Bf16 双方）
// =====================================================================

fn run_amp_scenario(amp_dtype: AmpDType, scalar_dtype: ScalarDType) {
    const STEPS: usize = 5;
    const LR: f32 = 0.05;
    let (x, y) = gen_regression_data(SEED_DATA);
    let scaler_config = GradScalerConfig {
        init_scale: 256.0,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 1000,
    };

    // 手動ループ: `compile_with_amp` が `run_fit` 内で行うのと同一の
    // 演算列（bind → forward_with_precision 相当 → mse_loss → scale_loss
    // → backward → trainable_grads → unscale → should_skip_step が
    // false のはず〈このシナリオでは backoff を誘発しない〉→ Sgd::step
    // → apply_parameters → update）。`SequentialVars::forward_with_
    // precision` 自体は `pub(super)` のためテストからは呼べず、その
    // 内部で行っているのと同じ `linear_forward_low_precision` 呼び出し
    // 列を `bound.linears()` を介して直接組み立てる。
    let mut manual_model = build_model();
    let mut manual_sgd = Sgd::new(SgdConfig::new(LR))
        .unwrap_or_else(|e| panic!("test fixture: Sgd::new が失敗した: {e}"));
    let mut manual_scaler = GradScaler::new(scaler_config)
        .unwrap_or_else(|e| panic!("test fixture: GradScaler::new が失敗した: {e}"));
    let mut manual_losses = Vec::with_capacity(STEPS);
    let mut manual_scales = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = manual_model.bind(&tape);
            let x_var = tape.var(&x);
            let y_var = tape.var_no_grad(&y);

            let linears = bound.linears();
            let h =
                linear_forward_low_precision(&linears[0], &x_var, Activation::Relu, scalar_dtype)
                    .unwrap_or_else(|e| {
                        panic!("test fixture: 層 1 の低精度 forward が失敗した: {e}")
                    });
            let pred =
                linear_forward_low_precision(&linears[1], &h, Activation::None, scalar_dtype)
                    .unwrap_or_else(|e| {
                        panic!("test fixture: 層 2 の低精度 forward が失敗した: {e}")
                    });
            let loss = pred
                .mse_loss(&y_var)
                .unwrap_or_else(|e| panic!("test fixture: mse_loss が失敗した: {e}"));
            manual_losses.push(scalar(&loss.to_tensor()));

            let scaled_loss = manual_scaler
                .scale_loss(&loss)
                .unwrap_or_else(|e| panic!("test fixture: scale_loss が失敗した: {e}"));
            let grads = tape
                .backward(&scaled_loss)
                .unwrap_or_else(|e| panic!("test fixture: backward が失敗した: {e}"));
            let grad_refs = bound
                .trainable_grads(&grads)
                .unwrap_or_else(|e| panic!("test fixture: trainable_grads が失敗した: {e}"));
            let unscale_result = manual_scaler
                .unscale(&grad_refs)
                .unwrap_or_else(|e| panic!("test fixture: unscale が失敗した: {e}"));
            assert!(
                !unscale_result.should_skip_step(),
                "このシナリオでは skip は想定していない"
            );
            let unscaled_refs: Vec<&Tensor<f32>> = unscale_result.grads.iter().collect();
            let param_refs = manual_model.trainable_parameters();
            let updated = manual_sgd
                .step(&param_refs, &unscaled_refs)
                .unwrap_or_else(|e| panic!("test fixture: Sgd::step が失敗した: {e}"));
            manual_scaler
                .update(false)
                .unwrap_or_else(|e| panic!("test fixture: GradScaler::update が失敗した: {e}"));
            updated
        };
        manual_model
            .apply_parameters(updated)
            .unwrap_or_else(|e| panic!("test fixture: apply_parameters が失敗した: {e}"));
        manual_scales.push(manual_scaler.scale());
    }

    let mut fit_model = build_model();
    fit_model
        .compile_with_amp(
            Optimizer::Sgd(SgdConfig::new(LR)),
            Loss::Mse,
            AmpConfig::new(amp_dtype).grad_scaler(scaler_config),
        )
        .unwrap_or_else(|e| panic!("test fixture: compile_with_amp が失敗した: {e}"));
    let history = fit_model
        .fit(&x, &y, FitConfig::new(STEPS, N))
        .unwrap_or_else(|e| panic!("test fixture: fit が失敗した: {e}"));

    assert_eq!(history.loss.len(), manual_losses.len());
    for (m, f) in manual_losses.iter().zip(history.loss.iter()) {
        assert_eq!(
            m.to_bits(),
            f.to_bits(),
            "loss diverged: manual={m} fit={f}"
        );
    }
    assert_eq!(
        fit_model.amp_loss_scale(),
        Some(*manual_scales.last().unwrap_or_else(|| unreachable!())),
        "GradScaler の最終 scale が手動ループと一致しない"
    );

    let manual_params = manual_model.trainable_parameters();
    let fit_params = fit_model.trainable_parameters();
    assert!(
        params_bit_exact(&manual_params, &fit_params),
        "compile_with_amp が手動 GradScaler + linear_forward_low_precision ループと \
         bit 一致しない（dtype={amp_dtype:?}）"
    );
}

#[test]
fn fit_amp_matches_manual_low_precision_gradscaler_loop_bit_exact_f16() {
    run_amp_scenario(AmpDType::F16, ScalarDType::F16);
}

#[test]
fn fit_amp_matches_manual_low_precision_gradscaler_loop_bit_exact_bf16() {
    run_amp_scenario(AmpDType::Bf16, ScalarDType::Bf16);
}

// =====================================================================
// 3. skip／backoff の挙動が手動 GradScaler と bit 一致する（AC-a の核）
// =====================================================================

/// `init_scale` を極端に大きく取り、scaled 勾配を意図的に overflow
/// させて最初の step を skip させる（`GradScaler` が本来扱う運用パス。
/// dtype 非依存——`AmpDType::F16` で固定し dtype 側の非有限化と混同
/// しない。実装計画 §5.1「主トリガ」）。
#[test]
fn fit_amp_skip_and_backoff_matches_gradscaler_bit_exact() {
    const STEPS: usize = 8;
    const LR: f32 = 0.05;
    // 通常のデータ生成に加え、勾配 overflow を誘発するため target を
    // 大きくスケールする（forward の loss 自体は有限のまま・scale_loss
    // 後の勾配のみが `init_scale` の大きさにより非有限化する）。
    let (x, y_base) = gen_regression_data(SEED_DATA);
    let y_scaled: Vec<f32> = y_base
        .contiguous()
        .as_slice()
        .expect("test fixture: contiguous 化済み")
        .iter()
        .map(|v| v * 1.0e3)
        .collect();
    let y = Tensor::new(y_scaled, y_base.shape())
        .unwrap_or_else(|e| panic!("test fixture: y の shape 構築に失敗: {e}"));

    let scaler_config = GradScalerConfig {
        init_scale: 2.0f32.powi(127),
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 2,
    };

    let mut manual_model = build_model();
    let mut manual_sgd = Sgd::new(SgdConfig::new(LR))
        .unwrap_or_else(|e| panic!("test fixture: Sgd::new が失敗した: {e}"));
    let mut manual_scaler = GradScaler::new(scaler_config)
        .unwrap_or_else(|e| panic!("test fixture: GradScaler::new が失敗した: {e}"));
    let mut manual_scales = Vec::with_capacity(STEPS);
    let mut manual_skips = Vec::with_capacity(STEPS);
    let mut manual_params_snapshots: Vec<Vec<Tensor<f32>>> = Vec::with_capacity(STEPS);

    for _ in 0..STEPS {
        let outcome = {
            let tape = fandhe_ai::tape();
            let bound = manual_model.bind(&tape);
            let x_var = tape.var(&x);
            let y_var = tape.var_no_grad(&y);

            // `fit_model` は AMP（`AmpDType::F16`）で compile するため、
            // 手動ループの forward も同じ低精度経路（`linear_forward_
            // low_precision`。§2 の `run_amp_scenario` と同型）に揃える
            // 必要がある（f32 の `bound.forward` を使うと skip 判定・
            // scale 推移自体は一致しても勾配値が異なり、後続 step で
            // パラメータが分岐する）。
            let linears = bound.linears();
            let h = linear_forward_low_precision(
                &linears[0],
                &x_var,
                Activation::Relu,
                ScalarDType::F16,
            )
            .unwrap_or_else(|e| panic!("test fixture: 層 1 の低精度 forward が失敗した: {e}"));
            let pred =
                linear_forward_low_precision(&linears[1], &h, Activation::None, ScalarDType::F16)
                    .unwrap_or_else(|e| {
                        panic!("test fixture: 層 2 の低精度 forward が失敗した: {e}")
                    });
            let loss = pred
                .mse_loss(&y_var)
                .unwrap_or_else(|e| panic!("test fixture: mse_loss が失敗した: {e}"));
            let scaled_loss = manual_scaler
                .scale_loss(&loss)
                .unwrap_or_else(|e| panic!("test fixture: scale_loss が失敗した: {e}"));
            let grads = tape
                .backward(&scaled_loss)
                .unwrap_or_else(|e| panic!("test fixture: backward が失敗した: {e}"));
            let grad_refs = bound
                .trainable_grads(&grads)
                .unwrap_or_else(|e| panic!("test fixture: trainable_grads が失敗した: {e}"));
            let unscale_result = manual_scaler
                .unscale(&grad_refs)
                .unwrap_or_else(|e| panic!("test fixture: unscale が失敗した: {e}"));

            if unscale_result.should_skip_step() {
                None
            } else {
                let unscaled_refs: Vec<&Tensor<f32>> = unscale_result.grads.iter().collect();
                let param_refs = manual_model.trainable_parameters();
                Some(
                    manual_sgd
                        .step(&param_refs, &unscaled_refs)
                        .unwrap_or_else(|e| panic!("test fixture: Sgd::step が失敗した: {e}")),
                )
            }
        };

        let skipped = outcome.is_none();
        if let Some(updated) = outcome {
            manual_model
                .apply_parameters(updated)
                .unwrap_or_else(|e| panic!("test fixture: apply_parameters が失敗した: {e}"));
        }
        manual_scaler
            .update(skipped)
            .unwrap_or_else(|e| panic!("test fixture: GradScaler::update が失敗した: {e}"));
        manual_scales.push(manual_scaler.scale());
        manual_skips.push(skipped);
        manual_params_snapshots.push(
            manual_model
                .trainable_parameters()
                .into_iter()
                .cloned()
                .collect(),
        );
    }

    // 手動ループ自体が「少なくとも 1 回は skip・少なくとも 1 回は非
    // skip」を経る構成であることを確認する（さもなくば本テストが
    // skip 経路を検証していないことになる）。
    assert!(
        manual_skips.iter().any(|&s| s),
        "このシナリオでは少なくとも 1 回 skip が発生するはず"
    );
    assert!(
        manual_skips.iter().any(|&s| !s),
        "このシナリオでは少なくとも 1 回は非 skip step が発生するはず"
    );
    // 最初の step は skip（`init_scale` が極端に大きいため）——
    // パラメータは不変のまま・scale は半減する。
    assert!(manual_skips[0], "最初の step は skip されるはずの構成");

    let mut fit_model = build_model();
    let initial_params: Vec<Tensor<f32>> = fit_model
        .trainable_parameters()
        .into_iter()
        .cloned()
        .collect();
    fit_model
        .compile_with_amp(
            Optimizer::Sgd(SgdConfig::new(LR)),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16).grad_scaler(scaler_config),
        )
        .unwrap_or_else(|e| panic!("test fixture: compile_with_amp が失敗した: {e}"));

    for step_idx in 0..STEPS {
        fit_model
            .fit(&x, &y, FitConfig::new(1, N))
            .unwrap_or_else(|e| panic!("test fixture: fit(1) が失敗した (step={step_idx}): {e}"));
        assert_eq!(
            fit_model.amp_loss_scale(),
            Some(manual_scales[step_idx]),
            "scale が手動ループと一致しない (step={step_idx})"
        );
        let fit_params: Vec<Tensor<f32>> = fit_model
            .trainable_parameters()
            .into_iter()
            .cloned()
            .collect();
        let fit_params_refs: Vec<&Tensor<f32>> = fit_params.iter().collect();
        let manual_params_refs: Vec<&Tensor<f32>> =
            manual_params_snapshots[step_idx].iter().collect();
        assert!(
            params_bit_exact(&manual_params_refs, &fit_params_refs),
            "パラメータが手動ループと一致しない (step={step_idx})"
        );
    }

    // 最初の step は skip されたため、fit 側のパラメータも初期値から
    // 不変のままのはず（skip 契約の直接検証）。
    let step0_params_refs: Vec<&Tensor<f32>> = manual_params_snapshots[0].iter().collect();
    let initial_params_refs: Vec<&Tensor<f32>> = initial_params.iter().collect();
    assert!(
        params_bit_exact(&step0_params_refs, &initial_params_refs),
        "最初の（skip された）step でパラメータが変化してしまった"
    );
}

// =====================================================================
// 4. compile_with_amp の状態は fit 呼び出しをまたいで継続する
// =====================================================================

#[test]
fn fit_amp_state_persists_across_fit_calls() {
    const LR: f32 = 0.05;
    let (x, y) = gen_regression_data(SEED_DATA);
    let scaler_config = GradScalerConfig {
        init_scale: 256.0,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 3,
    };

    let mut model_split = build_model();
    model_split
        .compile_with_amp(
            Optimizer::Sgd(SgdConfig::new(LR)),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16).grad_scaler(scaler_config),
        )
        .unwrap_or_else(|e| panic!("test fixture: compile_with_amp が失敗した: {e}"));
    model_split
        .fit(&x, &y, FitConfig::new(4, N))
        .unwrap_or_else(|e| panic!("test fixture: fit(4) #1 が失敗した: {e}"));
    model_split
        .fit(&x, &y, FitConfig::new(4, N))
        .unwrap_or_else(|e| panic!("test fixture: fit(4) #2 が失敗した: {e}"));

    let mut model_once = build_model();
    model_once
        .compile_with_amp(
            Optimizer::Sgd(SgdConfig::new(LR)),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16).grad_scaler(scaler_config),
        )
        .unwrap_or_else(|e| panic!("test fixture: compile_with_amp が失敗した: {e}"));
    model_once
        .fit(&x, &y, FitConfig::new(8, N))
        .unwrap_or_else(|e| panic!("test fixture: fit(8) が失敗した: {e}"));

    assert_eq!(
        model_split.amp_loss_scale(),
        model_once.amp_loss_scale(),
        "fit(4)+fit(4) の最終 scale が fit(8) と一致しない"
    );
    let params_split = model_split.trainable_parameters();
    let params_once = model_once.trainable_parameters();
    assert!(
        params_bit_exact(&params_split, &params_once),
        "fit(4)+fit(4) が fit(8) と bit 一致しない（AMP 状態が fit 呼び出しを \
         またいで継続していない）"
    );
}

// =====================================================================
// 5. compile_with_amp の検証失敗は既存 compiled 状態を保持する
// =====================================================================

#[test]
fn compile_with_amp_rejects_invalid_scaler_config_and_keeps_previous_compiled() {
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap_or_else(|e| panic!("test fixture: compile が失敗した: {e}"));
    assert!(model.is_compiled());
    assert_eq!(model.amp_loss_scale(), None);

    let invalid_scaler = GradScalerConfig {
        init_scale: 0.0,
        ..GradScalerConfig::default()
    };
    let err = model.compile_with_amp(
        Optimizer::Sgd(SgdConfig::new(0.2)),
        Loss::Mse,
        AmpConfig::new(AmpDType::F16).grad_scaler(invalid_scaler),
    );
    assert!(matches!(err, Err(AutodiffError::InvalidArgument(_))));

    // 失敗前の compile() 状態（AMP なし）がそのまま保持されている
    // （`compile_with_amp` doc「全構築成功後にのみ代入する」規約）。
    assert!(model.is_compiled());
    assert_eq!(model.amp_loss_scale(), None);
}
