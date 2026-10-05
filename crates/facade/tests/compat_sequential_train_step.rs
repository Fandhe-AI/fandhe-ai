//! `Sequential::fit_with_train_step`（イシュー #2568・親 #2499。Keras
//! `Model.train_step()` 相当）の公開経路テスト。
//!
//! 役割: #2184 で `training.rs` 内部に置いていたカスタム学習 step フックの
//! 単体テスト（`#[cfg(test)] fit_custom_step_for_test` 経由）を、確定した
//! 公開形（`docs/compat-train-step-hook-decision.md` §8.1。`TrainStepFn`／
//! `TrainStepOptimizer`／`TrainStepOutput`・`Sequential::fit_with_train_step`）
//! 経由の統合テストへ移設し、公開面固有の検証（`TrainStepOptimizer::lr`／
//! `step` の長さ検査・フック中の `is_compiled()`・エラー文言のメソッド名）を
//! 加えたもの（決定記録 §8.4）。新規演算はなく既存 `fit` と同一の
//! `bind → forward → loss → backward → trainable_grads → optimizer.step →
//! apply_parameters` 経路を公開 API だけで再構成するため、CPU 参照実装・
//! 実機 parity は不要。
//!
//! `#[ignore]` 分離は行わない（CPU バックエンドのみで実行可能）。

use fandhe_ai::compat::{
    AmpConfig, AmpDType, Callback, EarlyStopping, FitConfig, Loss, LrSchedule, Optimizer,
    Sequential, TrainStepFn, TrainStepOptimizer, TrainStepOutput,
};
use fandhe_ai::optim::{AdamWConfig, LbfgsConfig, SgdConfig, StepLr};
use fandhe_ai::{AutodiffError, Tensor};
use fandhe_ai_autodiff::Reduction;

const D_IN: usize = 3;
const D_HIDDEN: usize = 4;
const D_OUT: usize = 2;
const SEED_L1: u64 = 0x7570_1111;
const SEED_L2: u64 = 0x7570_2222;

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap()
}

/// splitmix64 による決定的な充填（値域 `(-0.5, 0.5)`）。
fn deterministic_fill(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        })
        .collect()
}

fn gen_regression_data(seed: u64, n: usize) -> (Tensor<f32>, Tensor<f32>) {
    let x = deterministic_fill(seed, n * D_IN);
    let y = deterministic_fill(seed ^ 0x5555_5555_5555_5555, n * D_OUT);
    (
        Tensor::new(x, &[n, D_IN]).unwrap(),
        Tensor::new(y, &[n, D_OUT]).unwrap(),
    )
}

fn params_bit_exact(a: &[&Tensor<f32>], b: &[&Tensor<f32>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(x, y)| {
            let (xd, yd) = (x.host_slice(), y.host_slice());
            xd.len() == yd.len()
                && xd
                    .iter()
                    .zip(yd.iter())
                    .all(|(p, q)| p.to_bits() == q.to_bits())
        })
}

fn compile_sgd(model: &mut Sequential) {
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
}

fn snapshot(model: &Sequential) -> Vec<Tensor<f32>> {
    model.trainable_parameters().into_iter().cloned().collect()
}

/// そのバッチの損失（スカラー Var）を f32 へ取り出す。
fn scalar(loss: &fandhe_ai::Var<'_>) -> f32 {
    loss.to_tensor().get(&[]).unwrap()
}

/// 既定 step（`mse_loss(Reduction::Mean)` → backward → `opt.step`）を
/// 公開 API だけで手書きしたフック。T1 が使い、既定の `fit` と bit 完全一致
/// することが公開経路の配線（n_batch 重み付け・step/apply 順序）の証明になる。
fn default_step_hook(
    m: &Sequential,
    x_batch: &Tensor<f32>,
    y_batch: &Tensor<f32>,
    opt: &mut TrainStepOptimizer<'_>,
) -> Result<TrainStepOutput, AutodiffError> {
    let tape = fandhe_ai::tape();
    let bound = m.bind(&tape);
    let x_var = tape.var(x_batch);
    let pred = bound.forward(&tape, &x_var)?;
    let target_var = tape.var(y_batch);
    let loss_var = pred.mse_loss_with(&target_var, Reduction::Mean)?;
    let loss_scalar = scalar(&loss_var);
    let grads = tape.backward(&loss_var)?;
    let grad_refs = bound.trainable_grads(&grads)?;
    let param_refs = m.trainable_parameters();
    let stepped = opt.step(&param_refs, &grad_refs)?;
    Ok(TrainStepOutput::new(loss_scalar).with_updated(stepped))
}

/// 損失計算のみ `Reduction::Sum`（`Loss` enum にない組合せ）で行い、
/// optimizer を使わずホスト側で `p - lr * g` を計算する（T2。optimizer 引数を
/// 取らないため、手動ループからも同じ関数を呼べる）。
fn custom_loss_host_sgd(
    m: &Sequential,
    x_batch: &Tensor<f32>,
    y_batch: &Tensor<f32>,
) -> Result<(f32, Vec<Tensor<f32>>), AutodiffError> {
    const LR: f32 = 0.02;
    let tape = fandhe_ai::tape();
    let bound = m.bind(&tape);
    let pred = bound.forward(&tape, &tape.var(x_batch))?;
    let loss_var = pred.mse_loss_with(&tape.var(y_batch), Reduction::Sum)?;
    let loss_scalar = scalar(&loss_var);
    let grads = tape.backward(&loss_var)?;
    let grad_refs = bound.trainable_grads(&grads)?;
    let mut updated = Vec::new();
    for (p, g) in m.trainable_parameters().iter().zip(grad_refs.iter()) {
        let new_vals: Vec<f32> = p
            .host_slice()
            .iter()
            .zip(g.host_slice().iter())
            .map(|(pv, gv)| pv - LR * gv)
            .collect();
        updated.push(Tensor::new(new_vals, p.shape())?);
    }
    Ok((loss_scalar, updated))
}

fn custom_loss_host_sgd_hook(
    m: &Sequential,
    x_batch: &Tensor<f32>,
    y_batch: &Tensor<f32>,
    _opt: &mut TrainStepOptimizer<'_>,
) -> Result<TrainStepOutput, AutodiffError> {
    let (loss, updated) = custom_loss_host_sgd(m, x_batch, y_batch)?;
    Ok(TrainStepOutput::new(loss).with_updated(updated))
}

fn assert_default_equivalence(optimizer: impl Fn() -> Optimizer, seed: u64, label: &str) {
    const N: usize = 7;
    const BATCH: usize = 2;
    const EPOCHS: usize = 3;
    let (x, y) = gen_regression_data(seed, N);

    let mut default_model = build_model();
    default_model.compile(optimizer(), Loss::Mse).unwrap();
    let default_history = default_model
        .fit(&x, &y, FitConfig::new(EPOCHS, BATCH))
        .unwrap();

    let mut hook_model = build_model();
    hook_model.compile(optimizer(), Loss::Mse).unwrap();
    let mut hook = default_step_hook;
    let hook_history = hook_model
        .fit_with_train_step(
            &x,
            &y,
            FitConfig::new(EPOCHS, BATCH),
            None,
            &mut [],
            &[],
            &mut hook,
        )
        .unwrap();

    assert_eq!(
        hook_history.loss, default_history.loss,
        "{label}: 既定 step を再実装したフックの History.loss が既定の fit と bit 一致しない"
    );
    assert!(
        params_bit_exact(
            &default_model.trainable_parameters(),
            &hook_model.trainable_parameters()
        ),
        "{label}: 学習後パラメータが既定の fit と bit 一致しない"
    );
}

/// T1: 既定 step の再実装フックは既定の `fit` と bit 完全一致する（SGD）。
/// N=7・batch=2（端数バッチ）・epochs=3。
#[test]
fn fit_with_train_step_reimplementing_default_matches_fit_bit_exact_sgd() {
    assert_default_equivalence(|| Optimizer::Sgd(SgdConfig::new(0.1)), 0x7570_AAAA, "SGD");
}

/// T1 の AdamW 構成版（optimizer 種別に依存しないことの確認）。
#[test]
fn fit_with_train_step_reimplementing_default_matches_fit_bit_exact_adamw() {
    assert_default_equivalence(
        || {
            Optimizer::AdamW(AdamWConfig {
                lr: 0.01,
                ..Default::default()
            })
        },
        0x7570_BBBB,
        "AdamW",
    );
}

/// T2: 自作損失（`Reduction::Sum`）とホスト SGD 更新。(a) 手動ループと
/// `History.loss`・パラメータが bit 完全一致し、(b) 固定シードで収束する。
#[test]
fn fit_with_train_step_custom_loss_and_host_sgd_converges() {
    const N: usize = 16;
    const BATCH: usize = 4;
    const EPOCHS: usize = 30;
    let (x, y) = gen_regression_data(0x7570_CCCC, N);

    let mut hook_model = build_model();
    compile_sgd(&mut hook_model);
    let mut hook = custom_loss_host_sgd_hook;
    let hook_history = hook_model
        .fit_with_train_step(
            &x,
            &y,
            FitConfig::new(EPOCHS, BATCH),
            None,
            &mut [],
            &[],
            &mut hook,
        )
        .unwrap();

    // (a) 手動ループ（N が BATCH で割り切れるため行スライスで同じバッチ列になる）。
    let mut manual_model = build_model();
    let mut manual_loss: Vec<f32> = Vec::new();
    for _ in 0..EPOCHS {
        let mut weighted_sum = 0.0f64;
        let mut count = 0usize;
        for b in 0..N / BATCH {
            let xb = Tensor::new(
                x.host_slice()[b * BATCH * D_IN..(b + 1) * BATCH * D_IN].to_vec(),
                &[BATCH, D_IN],
            )
            .unwrap();
            let yb = Tensor::new(
                y.host_slice()[b * BATCH * D_OUT..(b + 1) * BATCH * D_OUT].to_vec(),
                &[BATCH, D_OUT],
            )
            .unwrap();
            let (loss, updated) = custom_loss_host_sgd(&manual_model, &xb, &yb).unwrap();
            weighted_sum += loss as f64 * BATCH as f64;
            count += BATCH;
            manual_model.apply_parameters(updated).unwrap();
        }
        manual_loss.push((weighted_sum / count as f64) as f32);
    }
    assert_eq!(
        hook_history.loss, manual_loss,
        "フック fit の History.loss が手動ループと bit 一致しない"
    );
    assert!(
        params_bit_exact(
            &manual_model.trainable_parameters(),
            &hook_model.trainable_parameters()
        ),
        "フック fit の学習後パラメータが手動ループと bit 一致しない"
    );

    // (b) 収束。
    let first = hook_history.loss[0];
    let last = *hook_history.loss.last().unwrap();
    assert!(last < first, "収束していない（first={first}, last={last}）");
}

/// T3: 更新なし（`TrainStepOutput::new` のみ）はパラメータを bit 単位で
/// 変えず、loss だけ記録される。
#[test]
fn fit_with_train_step_none_update_leaves_params_unchanged() {
    const EPOCHS: usize = 2;
    let (x, y) = gen_regression_data(0x7570_DDDD, 6);
    let mut model = build_model();
    compile_sgd(&mut model);
    let before = snapshot(&model);

    let mut hook = |m: &Sequential,
                    xb: &Tensor<f32>,
                    yb: &Tensor<f32>,
                    _opt: &mut TrainStepOptimizer<'_>|
     -> Result<TrainStepOutput, AutodiffError> {
        let tape = fandhe_ai::tape();
        let pred = m.bind(&tape).forward(&tape, &tape.var(xb))?;
        let loss = pred.mse_loss(&tape.var(yb))?;
        Ok(TrainStepOutput::new(scalar(&loss)))
    };
    let history = model
        .fit_with_train_step(
            &x,
            &y,
            FitConfig::new(EPOCHS, 2),
            None,
            &mut [],
            &[],
            &mut hook,
        )
        .unwrap();

    assert_eq!(history.loss.len(), EPOCHS);
    let before_refs: Vec<&Tensor<f32>> = before.iter().collect();
    assert!(params_bit_exact(
        &before_refs,
        &model.trainable_parameters()
    ));
}

/// T4: フックの `Err` は fit の `Err` になり、compiled 状態と training
/// モードは維持・復元される。
#[test]
fn fit_with_train_step_error_propagates_and_keeps_compiled() {
    let (x, y) = gen_regression_data(0x7570_EEEE, 4);
    let mut model = build_model();
    compile_sgd(&mut model);
    let prev_training = model.training();

    let mut hook = |_m: &Sequential,
                    _xb: &Tensor<f32>,
                    _yb: &Tensor<f32>,
                    _opt: &mut TrainStepOptimizer<'_>|
     -> Result<TrainStepOutput, AutodiffError> {
        Err(AutodiffError::InvalidArgument("意図的な失敗".to_string()))
    };
    let err = model
        .fit_with_train_step(&x, &y, FitConfig::new(1, 2), None, &mut [], &[], &mut hook)
        .expect_err("フックの Err は fit の Err のはず");
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(model.is_compiled());
    assert_eq!(model.training(), prev_training);
}

/// 併用拒否系の共通検証: `InvalidArgument`・compiled 維持・パラメータ不変・
/// エラー文言が公開メソッド名 `Sequential::fit_with_train_step` を名乗る。
fn assert_rejected(mut model: Sequential, config: FitConfig, what: &str) {
    let (x, y) = gen_regression_data(0x7570_1AAA, 4);
    let before = snapshot(&model);
    let mut hook = default_step_hook;
    let err = model
        .fit_with_train_step(&x, &y, config, None, &mut [], &[], &mut hook)
        .expect_err(what);
    match &err {
        AutodiffError::InvalidArgument(msg) => assert!(
            msg.contains("Sequential::fit_with_train_step"),
            "{what}: エラー文言がメソッド名を含まない: {msg}"
        ),
        other => panic!("{what}: InvalidArgument 以外: {other:?}"),
    }
    assert!(model.is_compiled(), "{what}: compiled 状態が失われた");
    let before_refs: Vec<&Tensor<f32>> = before.iter().collect();
    assert!(params_bit_exact(
        &before_refs,
        &model.trainable_parameters()
    ));
}

#[test]
fn fit_with_train_step_rejected_with_amp() {
    let mut model = build_model();
    model
        .compile_with_amp(
            Optimizer::Sgd(SgdConfig::new(0.1)),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16),
        )
        .unwrap();
    assert_rejected(model, FitConfig::new(1, 2), "AMP との併用は Err のはず");
}

#[test]
fn fit_with_train_step_rejected_with_accumulate_steps_gt_one() {
    let mut model = build_model();
    compile_sgd(&mut model);
    assert_rejected(
        model,
        FitConfig::new(1, 2).accumulate_steps(2),
        "accumulate_steps > 1 との併用は Err のはず",
    );
}

#[test]
fn fit_with_train_step_rejected_with_lbfgs() {
    let mut model = build_model();
    model
        .compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse)
        .unwrap();
    assert_rejected(model, FitConfig::new(1, 2), "Lbfgs との併用は Err のはず");
}

/// 未 compile は `InvalidArgument`。
#[test]
fn fit_with_train_step_rejected_when_not_compiled() {
    let (x, y) = gen_regression_data(0x7570_1DDD, 4);
    let mut model = build_model();
    let mut hook = default_step_hook;
    let err = model
        .fit_with_train_step(&x, &y, FitConfig::new(1, 2), None, &mut [], &[], &mut hook)
        .expect_err("未 compile は Err のはず");
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

/// T6: validation・callbacks 併用でも `val_loss` が epoch 数分埋まる。
#[test]
fn fit_with_train_step_with_validation_and_callbacks() {
    const EPOCHS: usize = 3;
    let (x, y) = gen_regression_data(0x7570_2CCC, 8);
    let (x_val, y_val) = gen_regression_data(0x7570_2DDD, 4);
    let mut model = build_model();
    compile_sgd(&mut model);

    let mut callbacks = [Callback::EarlyStopping(EarlyStopping::new(100))];
    let mut hook = default_step_hook;
    let history = model
        .fit_with_train_step(
            &x,
            &y,
            FitConfig::new(EPOCHS, 2),
            Some((&x_val, &y_val)),
            &mut callbacks,
            &[],
            &mut hook,
        )
        .unwrap();
    assert_eq!(history.val_loss.len(), EPOCHS);
    assert_eq!(history.loss.len(), EPOCHS);
}

/// T7: shape の違う更新は `apply_parameters` の検査で `Err` になる。
#[test]
fn fit_with_train_step_shape_mismatch_update_is_rejected() {
    let (x, y) = gen_regression_data(0x7570_3EEE, 4);
    let mut model = build_model();
    compile_sgd(&mut model);

    let mut hook = |_m: &Sequential,
                    _xb: &Tensor<f32>,
                    _yb: &Tensor<f32>,
                    _opt: &mut TrainStepOptimizer<'_>|
     -> Result<TrainStepOutput, AutodiffError> {
        let bogus = Tensor::new(vec![0.0f32; 1], &[1])?;
        Ok(TrainStepOutput::new(0.0).with_updated(vec![bogus]))
    };
    let err = model
        .fit_with_train_step(&x, &y, FitConfig::new(1, 2), None, &mut [], &[], &mut hook)
        .expect_err("shape の違う更新は Err のはず");
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

/// `TrainStepOptimizer::lr` は epoch 開始時に `LrSchedule` が同期した値で、
/// `History.lr` と一致する。
#[test]
fn train_step_optimizer_lr_matches_history_lr() {
    const EPOCHS: usize = 3;
    const BATCH: usize = 2;
    let (x, y) = gen_regression_data(0x7570_4AAA, 4);
    let mut model = build_model();
    compile_sgd(&mut model);

    let mut seen: Vec<f32> = Vec::new();
    let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(
        StepLr::new(0.1, 1, 0.5).unwrap(),
    ))];
    let history = {
        let mut hook = |m: &Sequential,
                        xb: &Tensor<f32>,
                        yb: &Tensor<f32>,
                        opt: &mut TrainStepOptimizer<'_>|
         -> Result<TrainStepOutput, AutodiffError> {
            seen.push(opt.lr());
            default_step_hook(m, xb, yb, opt)
        };
        let hook: &mut TrainStepFn<'_, f32> = &mut hook;
        model
            .fit_with_train_step(
                &x,
                &y,
                FitConfig::new(EPOCHS, BATCH),
                None,
                &mut callbacks,
                &[],
                hook,
            )
            .unwrap()
    };

    let batches_per_epoch = 2;
    assert_eq!(seen.len(), EPOCHS * batches_per_epoch);
    for (i, lr) in seen.iter().enumerate() {
        assert_eq!(
            lr.to_bits(),
            history.lr[i / batches_per_epoch].to_bits(),
            "バッチ {i}: opt.lr() が History.lr と一致しない"
        );
    }
}

/// `TrainStepOptimizer::step` は params／grads の長さ不一致を
/// `InvalidArgument` で拒否し、fit も `Err` になる。
#[test]
fn train_step_optimizer_step_rejects_length_mismatch() {
    let (x, y) = gen_regression_data(0x7570_4BBB, 4);
    let mut model = build_model();
    compile_sgd(&mut model);

    let mut hook = |m: &Sequential,
                    xb: &Tensor<f32>,
                    yb: &Tensor<f32>,
                    opt: &mut TrainStepOptimizer<'_>|
     -> Result<TrainStepOutput, AutodiffError> {
        let tape = fandhe_ai::tape();
        let bound = m.bind(&tape);
        let pred = bound.forward(&tape, &tape.var(xb))?;
        let loss = pred.mse_loss(&tape.var(yb))?;
        let grads = tape.backward(&loss)?;
        let grad_refs = bound.trainable_grads(&grads)?;
        let params = m.trainable_parameters();
        let stepped = opt.step(&params, &grad_refs[..grad_refs.len() - 1])?;
        Ok(TrainStepOutput::new(scalar(&loss)).with_updated(stepped))
    };
    let err = model
        .fit_with_train_step(&x, &y, FitConfig::new(1, 2), None, &mut [], &[], &mut hook)
        .expect_err("長さ不一致は Err のはず");
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(model.is_compiled());
}

/// フック実行中は `is_compiled()` が false、fit 後は true に戻る。
#[test]
fn model_is_not_compiled_inside_hook() {
    let (x, y) = gen_regression_data(0x7570_4CCC, 4);
    let mut model = build_model();
    compile_sgd(&mut model);

    let mut compiled_inside: Vec<bool> = Vec::new();
    {
        let mut hook = |m: &Sequential,
                        xb: &Tensor<f32>,
                        yb: &Tensor<f32>,
                        opt: &mut TrainStepOptimizer<'_>|
         -> Result<TrainStepOutput, AutodiffError> {
            compiled_inside.push(m.is_compiled());
            default_step_hook(m, xb, yb, opt)
        };
        model
            .fit_with_train_step(&x, &y, FitConfig::new(1, 2), None, &mut [], &[], &mut hook)
            .unwrap();
    }
    assert!(!compiled_inside.is_empty());
    assert!(compiled_inside.iter().all(|c| !c));
    assert!(model.is_compiled());
}
