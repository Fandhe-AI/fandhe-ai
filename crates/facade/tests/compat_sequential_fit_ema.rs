//! EMA の facade 公開（イシュー #2560・親 #2558。決定記録 `docs/autodiff-ema-decision.md`
//! §10・§13）の統合テスト。`fandhe_ai` の公開パス
//! （`optim::ExponentialMovingAverage`・`compat::{Callback::Ema, EmaCallback}`）のみを
//! import し、内部クレートは直接 import しない。
//!
//! 判定はすべて bit 完全一致で行い、許容誤差定数は追加・変更しない。実機（CUDA／Metal）
//! 非依存のホスト計算のみのため `#[ignore]` 分離は行わない。

use std::collections::{BTreeMap, HashMap};

use fandhe_ai::compat::{
    AmpConfig, AmpDType, Callback, EarlyStopping, EmaCallback, FitConfig, LambdaCallback, Loss,
    ModelCheckpoint, Monitor, Optimizer, Sequential, TrainStepFn, TrainStepOptimizer,
    TrainStepOutput,
};
use fandhe_ai::nn::Module;
use fandhe_ai::optim::{ExponentialMovingAverage, LbfgsConfig, SgdConfig};
use fandhe_ai::{AutodiffError, TapeRef, Tensor, Var};

const N: usize = 16;
const D_IN: usize = 4;
const D_OUT: usize = 2;
const DECAY: f32 = 0.5;

fn gen_data() -> (Tensor<f32>, Tensor<f32>) {
    // 決定的な疑似データ（シード固定の線形合同法）。
    let mut state = 0x2560_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let x: Vec<f32> = (0..N * D_IN).map(|_| next()).collect();
    let y: Vec<f32> = (0..N * D_OUT).map(|_| next()).collect();
    (
        Tensor::new(x, &[N, D_IN]).expect("test fixture"),
        Tensor::new(y, &[N, D_OUT]).expect("test fixture"),
    )
}

fn build_model() -> Sequential {
    let mut m = Sequential::new()
        .add_linear(D_IN, 8, 0x11)
        .expect("test fixture")
        .add_relu()
        .add_linear(8, D_OUT, 0x22)
        .expect("test fixture");
    m.compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .expect("test fixture");
    m
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("test fixture: contiguous 化済み")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn dict_bits(d: &HashMap<String, Tensor<f32>>) -> BTreeMap<String, Vec<u32>> {
    d.iter().map(|(k, v)| (k.clone(), bits(v))).collect()
}

fn model_bits(m: &Sequential) -> BTreeMap<String, Vec<u32>> {
    dict_bits(&m.state_dict())
}

fn shadow_of(cb: &Callback) -> HashMap<String, Tensor<f32>> {
    let Callback::Ema(e) = cb else {
        panic!("test fixture: Callback::Ema のはず");
    };
    e.shadow_state_dict()
        .expect("test fixture: fit 後は初期化済み")
}

fn ema_of(cb: &Callback) -> &EmaCallback {
    let Callback::Ema(e) = cb else {
        panic!("test fixture: Callback::Ema のはず");
    };
    e
}

fn ema_cb(decay: f32) -> Callback {
    Callback::Ema(EmaCallback::new(decay).expect("test fixture: 有効な decay"))
}

fn is_invalid_arg<T: std::fmt::Debug>(r: Result<T, AutodiffError>) -> bool {
    matches!(r, Err(AutodiffError::InvalidArgument(_)))
}

// 1. 手動ループ（全バッチ 1 step の epoch を 1 回ずつ fit → update_named）と bit 一致。
#[test]
fn fit_ema_matches_manual_loop_bit_exact() {
    let (x, y) = gen_data();
    let epochs = 3;

    let mut model_a = build_model();
    let mut cbs = [ema_cb(DECAY)];
    model_a
        .fit_with_callbacks(&x, &y, FitConfig::new(epochs, N), None, &mut cbs)
        .expect("fit");

    let mut model_b = build_model();
    let mut ema = ExponentialMovingAverage::from_named(DECAY, model_b.named_parameters())
        .expect("from_named");
    for _ in 0..epochs {
        model_b.fit(&x, &y, FitConfig::new(1, N)).expect("fit");
        ema.update_named(model_b.named_parameters())
            .expect("update");
    }

    assert_eq!(
        dict_bits(&shadow_of(&cbs[0])),
        dict_bits(&ema.shadow_state_dict())
    );
    let e = ema_of(&cbs[0]);
    assert_eq!(e.num_updates(), epochs as u64);
    assert_eq!(e.num_updates(), ema.num_updates());
    assert_eq!(e.decay(), DECAY);
}

// 2. opt-in が学習を乱さない・fit 後に重みを自動上書きしない。
#[test]
fn ema_does_not_disturb_training_nor_overwrite_weights() {
    let (x, y) = gen_data();
    let mut with = build_model();
    let mut cbs = [ema_cb(DECAY)];
    let h_with = with
        .fit_with_callbacks(&x, &y, FitConfig::new(3, 4), None, &mut cbs)
        .expect("fit");

    let mut without = build_model();
    let h_without = without
        .fit_with_callbacks(&x, &y, FitConfig::new(3, 4), None, &mut [])
        .expect("fit");

    assert_eq!(model_bits(&with), model_bits(&without));
    let a: Vec<u32> = h_with.loss.iter().map(|v| v.to_bits()).collect();
    let b: Vec<u32> = h_without.loss.iter().map(|v| v.to_bits()).collect();
    assert_eq!(a, b);
    // 4 バッチ × 3 epoch = 12 step。
    assert_eq!(ema_of(&cbs[0]).num_updates(), 12);
    // shadow は生の重みとは別物（自動上書きされていない）。
    assert_ne!(dict_bits(&shadow_of(&cbs[0])), model_bits(&with));
}

// 3. validation は shadow の下で評価され、epoch 後は生の重みへ復帰する。
#[test]
fn validation_runs_under_shadow_and_restores_raw_weights() {
    let (x, y) = gen_data();
    let mut model = build_model();
    let mut cbs = [ema_cb(DECAY)];
    let h = model
        .fit_with_callbacks(&x, &y, FitConfig::new(1, N), Some((&x, &y)), &mut cbs)
        .expect("fit");

    // 参照: 同一条件で EMA なしに 1 epoch 学習し、手動で shadow を作って評価する。
    let mut reference = build_model();
    let mut ema = ExponentialMovingAverage::from_named(DECAY, reference.named_parameters())
        .expect("from_named");
    reference.fit(&x, &y, FitConfig::new(1, N)).expect("fit");
    ema.update_named(reference.named_parameters())
        .expect("update");

    // epoch 後は生の重み。
    assert_eq!(model_bits(&model), model_bits(&reference));
    let raw = reference.state_dict();
    reference
        .load_state_dict(ema.shadow_state_dict())
        .expect("load");
    let expected: f32 = reference.evaluate(&x, &y, N).expect("evaluate");
    assert_eq!(h.val_loss[0].to_bits(), expected.to_bits());
    reference.load_state_dict(raw).expect("load");
}

// 4. accumulate_steps > 1: 更新は実際に step した時だけ（端数 flush 込み）。
#[test]
fn accumulate_steps_updates_only_on_real_steps() {
    let (x, y) = gen_data();
    let mut model = build_model();
    let mut cbs = [ema_cb(DECAY)];
    // バッチ 4 件 → 4 バッチ／epoch。accumulate 3 → 窓 1 回＋端数 flush 1 回 = 2 step／epoch。
    model
        .fit_with_callbacks(
            &x,
            &y,
            FitConfig::new(2, 4).accumulate_steps(3),
            None,
            &mut cbs,
        )
        .expect("fit");
    assert_eq!(ema_of(&cbs[0]).num_updates(), 4);
}

// 5. エラー経路でも差し替えが残らない（epoch 末 callback の Err）。
#[test]
fn error_in_epoch_end_callback_restores_raw_weights() {
    let (x, y) = gen_data();
    let mut model = build_model();
    let mut cbs = [
        ema_cb(DECAY),
        Callback::Lambda(LambdaCallback::on_epoch_end(|_, _| {
            Err(AutodiffError::InvalidArgument(
                "test: epoch 末で失敗".into(),
            ))
        })),
    ];
    let err = model.fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut cbs);
    assert!(is_invalid_arg(err));

    let mut reference = build_model();
    reference.fit(&x, &y, FitConfig::new(1, N)).expect("fit");
    assert_eq!(model_bits(&model), model_bits(&reference));
}

// 6. ModelCheckpoint／EarlyStopping は EMA 重みの snapshot を持つ。
#[test]
fn checkpoint_and_early_stopping_observe_ema_weights() {
    let (x, y) = gen_data();
    let mut model = build_model();
    let mut cbs = [
        ema_cb(DECAY),
        Callback::ModelCheckpoint(ModelCheckpoint::new().monitor(Monitor::Loss)),
        Callback::EarlyStopping(
            EarlyStopping::new(5)
                .monitor(Monitor::Loss)
                .restore_best_weights(true),
        ),
    ];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(1, N), None, &mut cbs)
        .expect("fit");
    let shadow = dict_bits(&shadow_of(&cbs[0]));
    let Callback::ModelCheckpoint(mc) = &cbs[1] else {
        panic!("test fixture");
    };
    let best = mc.best_state_dict().expect("snapshot");
    assert_eq!(dict_bits(best), shadow);
    // restore_best_weights が書き戻すのも EMA 重みの snapshot。
    assert_eq!(model_bits(&model), shadow);
}

// 7. 拒否条件: InvalidArgument・モデル重み・compile 状態・train/eval モード不変。
#[test]
fn rejected_combinations_leave_model_untouched() {
    let (x, y) = gen_data();

    // (a) Callback::Ema の複数指定。
    let mut m = build_model();
    let before = model_bits(&m);
    let mode = m.training();
    let mut cbs = [ema_cb(0.9), ema_cb(0.8)];
    assert!(is_invalid_arg(m.fit_with_callbacks(
        &x,
        &y,
        FitConfig::new(1, N),
        None,
        &mut cbs
    )));
    assert_eq!(model_bits(&m), before);
    assert_eq!(m.training(), mode);
    m.evaluate(&x, &y, N).expect("compile 状態が保持されている");

    // (b) L-BFGS。
    let mut m = Sequential::new()
        .add_linear(D_IN, D_OUT, 1)
        .expect("fixture");
    m.compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse)
        .expect("compile");
    let before = model_bits(&m);
    let mut cbs = [ema_cb(0.9)];
    assert!(is_invalid_arg(m.fit_with_callbacks(
        &x,
        &y,
        FitConfig::new(1, N),
        None,
        &mut cbs
    )));
    assert_eq!(model_bits(&m), before);
    m.evaluate(&x, &y, N).expect("compile 状態が保持されている");

    // (c) カスタム train_step フック。
    let mut m = build_model();
    let before = model_bits(&m);
    let mut cbs = [ema_cb(0.9)];
    let mut step = |_: &Sequential,
                    _: &Tensor<f32>,
                    _: &Tensor<f32>,
                    _: &mut TrainStepOptimizer<'_>|
     -> Result<TrainStepOutput, AutodiffError> { Ok(TrainStepOutput::new(0.0)) };
    let hook: &mut TrainStepFn<'_, f32> = &mut step;
    assert!(is_invalid_arg(m.fit_with_train_step(
        &x,
        &y,
        FitConfig::new(1, N),
        None,
        &mut cbs,
        &[],
        hook
    )));
    assert_eq!(model_bits(&m), before);

    // (d) AMP（skip step の扱いが決定記録 §10.2 (g) で未決のため拒否）。
    let mut m = Sequential::new()
        .add_linear(D_IN, D_OUT, 1)
        .expect("fixture");
    m.compile_with_amp(
        Optimizer::Sgd(SgdConfig::new(0.05)),
        Loss::Mse,
        AmpConfig::new(AmpDType::Bf16),
    )
    .expect("compile_with_amp");
    let before = model_bits(&m);
    let mut cbs = [ema_cb(0.9)];
    assert!(is_invalid_arg(m.fit_with_callbacks(
        &x,
        &y,
        FitConfig::new(1, N),
        None,
        &mut cbs
    )));
    assert_eq!(model_bits(&m), before);
    m.evaluate(&x, &y, N).expect("compile 状態が保持されている");
}

// 8. decay 検証と fit 前の状態。
#[test]
fn ema_callback_validates_decay_and_starts_uninitialized() {
    for bad in [f32::NAN, -0.1, 1.1, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(is_invalid_arg(EmaCallback::new(bad)), "decay={bad}");
    }
    for ok in [0.0f32, 1.0, 0.999] {
        assert!(EmaCallback::new(ok).is_ok(), "decay={ok}");
    }
    let e = EmaCallback::new(0.9).expect("ok");
    assert!(e.shadow_state_dict().is_none());
    assert_eq!(e.num_updates(), 0);
    assert_eq!(e.decay(), 0.9);
}

// 9. fit をまたいで shadow が継続し、別構成のモデルへの使い回しは拒否される。
#[test]
fn shadow_persists_across_fits_and_rejects_other_model() {
    let (x, y) = gen_data();
    let mut model = build_model();
    let mut cbs = [ema_cb(DECAY)];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(1, N), None, &mut cbs)
        .expect("fit");
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(1, N), None, &mut cbs)
        .expect("fit");
    assert_eq!(ema_of(&cbs[0]).num_updates(), 2);

    // 構成が異なるモデルでは名前集合の不一致で拒否される（shadow は不変）。
    let before = dict_bits(&shadow_of(&cbs[0]));
    let mut other = Sequential::new()
        .add_linear(D_IN, D_OUT, 3)
        .expect("fixture");
    other
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .expect("compile");
    assert!(is_invalid_arg(other.fit_with_callbacks(
        &x,
        &y,
        FitConfig::new(1, N),
        None,
        &mut cbs
    )));
    assert_eq!(dict_bits(&shadow_of(&cbs[0])), before);
}

// 10. ラッパー単体: 閉形式一致・失敗時 shadow 不変・facade Module 経由の往復。
#[test]
fn wrapper_closed_form_and_atomicity() {
    let w0 = Tensor::new(vec![1.0f32, -2.0, 3.5], &[3]).expect("fixture");
    let mut ema = ExponentialMovingAverage::new(0.9, &[&w0]).expect("new");
    let w1 = Tensor::new(vec![0.5f32, 4.0, -1.0], &[3]).expect("fixture");
    ema.update(&[&w1]).expect("update");
    let one_minus = 1.0f32 - 0.9;
    let expected: Vec<u32> = [1.0f32, -2.0, 3.5]
        .iter()
        .zip([0.5f32, 4.0, -1.0])
        .map(|(s, p)| f32::mul_add(0.9, *s, one_minus * p).to_bits())
        .collect();
    assert_eq!(bits(ema.shadow("0").expect("shadow")), expected);

    // shape 不一致は shadow を変えずに Err。
    let before = bits(ema.shadow("0").expect("shadow"));
    let bad = Tensor::new(vec![1.0f32, 2.0], &[2]).expect("fixture");
    assert!(matches!(ema.update(&[&bad]), Err(AutodiffError::Shape(_))));
    assert_eq!(bits(ema.shadow("0").expect("shadow")), before);
    assert_eq!(ema.num_updates(), 1);
}

struct Pair {
    a: Tensor<f32>,
    b: Tensor<f32>,
}

impl Module for Pair {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("a".into(), &self.a), ("b".into(), &self.b)]
    }
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        match name {
            "a" => self.a = value,
            "b" => self.b = value,
            _ => return Err(AutodiffError::InvalidArgument(format!("unknown {name}"))),
        }
        Ok(())
    }
}

#[test]
fn wrapper_module_apply_restore_round_trip() {
    let t = |v: f32| Tensor::new(vec![v, v + 1.0], &[2]).expect("fixture");
    let mut m = Pair {
        a: t(1.0),
        b: t(10.0),
    };
    let mut ema = ExponentialMovingAverage::from_module(0.5, &m).expect("from_module");
    m.a = t(3.0);
    m.b = t(30.0);
    ema.update_from_module(&m).expect("update_from_module");

    let live = dict_bits(&m.state_dict());
    let backup = ema.apply(&mut m).expect("apply");
    // apply 後は shadow の値（0.5 * old + 0.5 * new）。
    assert_eq!(
        dict_bits(&m.state_dict()),
        dict_bits(&ema.shadow_state_dict())
    );
    ExponentialMovingAverage::restore(&mut m, backup).expect("restore");
    assert_eq!(dict_bits(&m.state_dict()), live);
}
