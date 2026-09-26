//! イシュー #2176（親 #2131）: LR scheduler 5 種
//! （[`MultiStepLr`]・[`CosineAnnealingWarmRestarts`]・[`CyclicLr`]・
//! [`LambdaLr`]・[`SequentialLr`]）を `Sequential::fit_with_callbacks`
//! （`LrSchedule::per_epoch`）経由で駆動した際の統合テスト。
//!
//! facade（`fandhe_ai::optim`）への公開は本イシューでは保留のため
//! （`crates/facade/src/lib.rs::LrSchedulerExtHoldDoctestGuard`）、
//! 5 種は `fandhe_ai_autodiff::nn::optim` から直接 import する
//! （`compat_sequential_callbacks.rs::lr_schedule_per_epoch_matches_
//! manual_set_lr_loop` と同じ検証方式・同じ決定的 fixture）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::{Callback, FitConfig, Loss, LrSchedule, Optimizer, Sequential};
use fandhe_ai::{Tensor, tape};
use fandhe_ai_autodiff::nn::optim::{
    CosineAnnealingWarmRestarts, CyclicLr, LambdaLr, LrScheduler, MultiStepLr, StepLr,
};
use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};

const N: usize = 16;
const D_IN: usize = 4;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 2;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

/// `compat_sequential_callbacks.rs::gen_regression_data` と同型の決定的
/// 生成（本ファイルはテストバイナリが分かれるため独立に定義する）。
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

/// `LrSchedule::per_epoch(sched)` を経由した `fit_with_callbacks` の
/// `history.lr` が `sched.lr_at(e)` と bit 一致し、最終パラメータが
/// 手動 `set_lr` ループと bit 一致することを検証する共通ヘルパー
/// （`compat_sequential_callbacks.rs::lr_schedule_per_epoch_matches_
/// manual_set_lr_loop` と同じ演算列比較方式）。
fn assert_per_epoch_matches_manual_loop(
    name: &str,
    make_sched_for_check: impl Fn() -> Box<dyn LrScheduler>,
    make_sched_for_fit: impl Fn() -> Box<dyn LrScheduler>,
    epochs: usize,
    initial_lr: f32,
) {
    let (x, y) = gen_regression_data(SEED_DATA);
    let sched_for_check = make_sched_for_check();

    let mut manual_model = build_model();
    let mut manual_sgd = Sgd::new(SgdConfig::new(initial_lr)).unwrap();
    for e in 0..epochs {
        manual_sgd.set_lr(sched_for_check.lr_at(e)).unwrap();
        let updated = {
            let t = tape();
            let bound = manual_model.bind(&t);
            let x_var = t.var(&x);
            let y_var = t.var_no_grad(&y);
            let pred = bound.forward(&t, &x_var).unwrap();
            let loss = pred.mse_loss(&y_var).unwrap();
            let grads = t.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = manual_model.trainable_parameters();
            manual_sgd.step(&param_refs, &grad_refs).unwrap()
        };
        manual_model.apply_parameters(updated).unwrap();
    }

    let mut fit_model = build_model();
    fit_model
        .compile(Optimizer::Sgd(SgdConfig::new(initial_lr)), Loss::Mse)
        .unwrap();
    let sched_for_fit = make_sched_for_fit();
    let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(BoxedScheduler(
        sched_for_fit,
    )))];
    let history = fit_model
        .fit_with_callbacks(&x, &y, FitConfig::new(epochs, N), None, &mut callbacks)
        .unwrap();

    assert_eq!(history.lr.len(), epochs, "{name}: history.lr の長さ");
    for (e, lr) in history.lr.iter().enumerate() {
        assert_eq!(
            lr.to_bits(),
            sched_for_check.lr_at(e).to_bits(),
            "{name}: epoch {e} で history.lr が lr_at と bit 一致しない"
        );
    }

    let manual_params = manual_model.trainable_parameters();
    let fit_params = fit_model.trainable_parameters();
    assert!(
        params_bit_exact(&manual_params, &fit_params),
        "{name}: LrSchedule::per_epoch 経由の fit が手動 set_lr ループと\
         bit 一致しない"
    );

    // 学習曲線が下降することも確認する（既存の収束テストと同じ判定
    // 方式。新しい許容誤差は設けない）。
    assert!(
        history.loss[history.loss.len() - 1] < history.loss[0],
        "{name}: 学習曲線が下降していない（loss[0]={} loss[last]={}）",
        history.loss[0],
        history.loss[history.loss.len() - 1]
    );
}

/// `LrSchedule::per_epoch` は `impl LrScheduler + 'static` を要求する
/// ため、`Box<dyn LrScheduler>` をそのまま渡せない（`Box<dyn
/// LrScheduler>` 自体が `LrScheduler` を実装していないため）。本
/// ラッパーは `Box<dyn LrScheduler>` を `LrScheduler` として再委譲する
/// 薄いアダプタで、複数の具体型を単一のヘルパー関数
/// （[`assert_per_epoch_matches_manual_loop`]）へ渡すためだけに使う
/// テスト専用の橋渡し（本番コードには存在しない）。
struct BoxedScheduler(Box<dyn LrScheduler>);
impl LrScheduler for BoxedScheduler {
    fn lr_at(&self, step: usize) -> f32 {
        self.0.lr_at(step)
    }
}

const EPOCHS: usize = 5;

#[test]
fn multi_step_lr_per_epoch_matches_manual_loop() {
    assert_per_epoch_matches_manual_loop(
        "MultiStepLr",
        || Box::new(MultiStepLr::new(0.1, &[2, 4], 0.5).unwrap()),
        || Box::new(MultiStepLr::new(0.1, &[2, 4], 0.5).unwrap()),
        EPOCHS,
        0.1,
    );
}

#[test]
fn cosine_annealing_warm_restarts_per_epoch_matches_manual_loop() {
    assert_per_epoch_matches_manual_loop(
        "CosineAnnealingWarmRestarts",
        || Box::new(CosineAnnealingWarmRestarts::new(0.1, 2, 2, 0.01).unwrap()),
        || Box::new(CosineAnnealingWarmRestarts::new(0.1, 2, 2, 0.01).unwrap()),
        EPOCHS,
        0.1,
    );
}

#[test]
fn cyclic_lr_per_epoch_matches_manual_loop() {
    assert_per_epoch_matches_manual_loop(
        "CyclicLr",
        || Box::new(CyclicLr::new(0.02, 0.1, 2, Some(3)).unwrap()),
        || Box::new(CyclicLr::new(0.02, 0.1, 2, Some(3)).unwrap()),
        EPOCHS,
        0.02,
    );
}

#[test]
fn lambda_lr_per_epoch_matches_manual_loop() {
    assert_per_epoch_matches_manual_loop(
        "LambdaLr",
        || Box::new(LambdaLr::new(0.1, |e: usize| 0.9f64.powi(e as i32)).unwrap()),
        || Box::new(LambdaLr::new(0.1, |e: usize| 0.9f64.powi(e as i32)).unwrap()),
        EPOCHS,
        0.1,
    );
}

#[test]
fn sequential_lr_per_epoch_matches_manual_loop() {
    let make = || -> Box<dyn LrScheduler> {
        let s1: Box<dyn LrScheduler> = Box::new(StepLr::new(0.1, 1, 0.5).unwrap());
        let s2: Box<dyn LrScheduler> = Box::new(StepLr::new(0.1, 1, 0.9).unwrap());
        Box::new(fandhe_ai_autodiff::nn::optim::SequentialLr::new(vec![s1, s2], vec![2]).unwrap())
    };
    assert_per_epoch_matches_manual_loop("SequentialLr", make, make, EPOCHS, 0.1);
}
