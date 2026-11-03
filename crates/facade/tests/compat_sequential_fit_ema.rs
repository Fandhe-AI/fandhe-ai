//! EMA の facade 公開（イシュー #2560・親 #2558。決定記録 `docs/autodiff-ema-decision.md`
//! §10・§13）の統合テスト。`fandhe_ai` の公開パス
//! （`optim::ExponentialMovingAverage`・`compat::{Callback::Ema, EmaCallback}`）のみを
//! import し、内部クレートは直接 import しない。
//!
//! `Monitor::Loss` 併用の受理（決定記録 §15・イシュー #2844）もここで固定する。
//!
//! 判定はすべて bit 完全一致で行い、許容誤差定数は追加・変更しない。実機（CUDA／Metal）
//! 非依存のホスト計算のみのため `#[ignore]` 分離は行わない。

use std::collections::{BTreeMap, HashMap};

use fandhe_ai::compat::{
    AmpConfig, AmpDType, Callback, EarlyStopping, EmaCallback, FitConfig, LambdaCallback, Loss,
    ModelCheckpoint, Monitor, Optimizer, Sequential, SwaCallback, TrainStepFn, TrainStepOptimizer,
    TrainStepOutput,
};
use fandhe_ai::nn::Module;
use fandhe_ai::optim::SwaAnneal;
use fandhe_ai::optim::{ExponentialMovingAverage, GradScalerConfig, LbfgsConfig, SgdConfig};
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

// 2b. 利用手順（イシュー #2561）: fit 後は生の重みのまま。呼び出し側が shadow を
// `load_state_dict` で適用して推論へ使う。適用後の重みは shadow と bit 一致し、予測は有限。
#[test]
fn ema_weights_applied_after_fit_match_shadow_and_predict_finite() {
    let (x, y) = gen_data();
    let mut model = build_model();
    let mut cbs = [ema_cb(0.9)];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(3, 4), None, &mut cbs)
        .expect("test fixture: fit");

    let raw = model_bits(&model);
    let shadow = shadow_of(&cbs[0]);
    assert_ne!(
        raw,
        dict_bits(&shadow),
        "fit 終了時に重みを自動上書きしない"
    );

    model
        .load_state_dict(shadow.clone())
        .expect("test fixture: shadow は同一モデル由来");
    assert_eq!(model_bits(&model), dict_bits(&shadow));

    let pred = model.predict(&x).expect("test fixture: predict");
    assert!(
        pred.contiguous()
            .as_slice()
            .expect("test fixture: contiguous 化済み")
            .iter()
            .all(|v| v.is_finite())
    );
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
        Callback::ModelCheckpoint(ModelCheckpoint::new().monitor(Monitor::ValLoss)),
        Callback::EarlyStopping(
            EarlyStopping::new(5)
                .monitor(Monitor::ValLoss)
                .restore_best_weights(true),
        ),
    ];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(1, N), Some((&x, &y)), &mut cbs)
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

    // (d) AMP は #3831 で受理へ反転（決定記録 §17・§19）。受理側は下記テスト 8a〜8d が
    // 正のテストとして担保し、残る拒否（AMP×accumulate／カスタム step／SWA）はテスト 8e が担保する。
    // (e) の Monitor::Loss 併用は #2844 で拒否を撤去済み（決定記録 §15）。受理側は下記
    // テスト 7b／7c が正のテストとして担保する。
}

/// EMA なしの参照モデルを 1 epoch（全件 1 バッチ = 1 step）ずつ `fit` し、各 epoch 末の
/// (shadow, 生の重み) を記録する（テスト 1 と同じ手法。SGD は状態を持たないため 1 epoch ずつの
/// `fit` と k epoch の `fit` は bit 一致する）。戻り値の最後の要素は参照モデル自身。
type DictBits = BTreeMap<String, Vec<u32>>;
fn reference_trace(epochs: usize) -> (Vec<DictBits>, Vec<DictBits>, Sequential) {
    let (x, y) = gen_data();
    let mut reference = build_model();
    let mut ema = ExponentialMovingAverage::from_named(DECAY, reference.named_parameters())
        .expect("from_named");
    let (mut shadows, mut raws) = (Vec::new(), Vec::new());
    for _ in 0..epochs {
        reference.fit(&x, &y, FitConfig::new(1, N)).expect("fit");
        ema.update_named(reference.named_parameters())
            .expect("update");
        shadows.push(dict_bits(&ema.shadow_state_dict()));
        raws.push(model_bits(&reference));
    }
    (shadows, raws, reference)
}

// 7b. 決定記録 §15.2: Ema + ModelCheckpoint(Monitor::Loss)（validation なし）。
// 判定値 = 生の重みの訓練損失（History::loss）、保存重み = その epoch 末の EMA shadow。
// 併用の拒否撤去は #2844。ValLoss 併用の不変はテスト 6、Err 経路の復帰はテスト 5 が担保。
#[test]
fn monitor_loss_checkpoint_judges_raw_train_loss_and_saves_ema_weights() {
    let (x, y) = gen_data();
    let epochs = 4;
    let (shadows, raws, reference) = reference_trace(epochs);

    for save_best_only in [true, false] {
        let mut model = build_model();
        let mut cbs = [
            ema_cb(DECAY),
            Callback::ModelCheckpoint(
                ModelCheckpoint::new()
                    .monitor(Monitor::Loss)
                    .save_best_only(save_best_only),
            ),
        ];
        let h = model
            .fit_with_callbacks(&x, &y, FitConfig::new(epochs, N), None, &mut cbs)
            .expect("併用は受理される（#2844）");
        // 前提: EMA 併用でも学習は乱れない（追加の順伝播なし・History::loss 不変）。
        assert_eq!(model_bits(&model), model_bits(&reference));
        let mut plain = build_model();
        let h_plain = plain
            .fit_with_callbacks(&x, &y, FitConfig::new(epochs, N), None, &mut [])
            .expect("fit");
        let lb = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(lb(&h.loss), lb(&h_plain.loss));

        let Callback::ModelCheckpoint(mc) = &cbs[1] else {
            panic!("test fixture");
        };
        let be = mc.best_epoch().expect("best あり");
        assert_eq!(
            mc.best_value().expect("best あり").to_bits(),
            h.loss[be].to_bits()
        );
        let saved = dict_bits(mc.best_state_dict().expect("snapshot"));
        // 保存重みは EMA 重み（save_best_only(false) では最終 epoch 末の shadow）。
        let saved_epoch = if save_best_only { be } else { epochs - 1 };
        assert_eq!(saved, shadows[saved_epoch]);
        assert_ne!(saved, raws[saved_epoch]);
    }
}

// 7c. 決定記録 §15.2: Ema + EarlyStopping(Monitor::Loss).restore_best_weights（validation なし）。
#[test]
fn monitor_loss_early_stopping_without_validation_restores_ema_snapshot() {
    let (x, y) = gen_data();
    let epochs = 4;
    let (shadows, raws, _) = reference_trace(epochs);

    // 完走: 訓練損失は改善し続け、fit 後の重みはベスト epoch 末の shadow に戻る。
    let mut model = build_model();
    let mut cbs = [
        ema_cb(DECAY),
        Callback::EarlyStopping(
            EarlyStopping::new(epochs)
                .monitor(Monitor::Loss)
                .restore_best_weights(true),
        ),
    ];
    let h = model
        .fit_with_callbacks(&x, &y, FitConfig::new(epochs, N), None, &mut cbs)
        .expect("validation なしでも受理される（#2844）");
    let Callback::EarlyStopping(es) = &cbs[1] else {
        panic!("test fixture");
    };
    let be = es.best_epoch().expect("best あり");
    assert_eq!(
        es.best_value().expect("best あり").to_bits(),
        h.loss[be].to_bits()
    );
    assert_eq!(model_bits(&model), shadows[be]);
    assert_ne!(model_bits(&model), raws[be]);

    // 打ち切り: 巨大な min_delta で 2 epoch 目以降を「改善なし」にし patience=1 で停止させる。
    // 打ち切り後も差し替えは残らず、restore_best_weights の書き戻しだけが効く。
    let mut model = build_model();
    let mut cbs = [
        ema_cb(DECAY),
        Callback::EarlyStopping(
            EarlyStopping::new(1)
                .monitor(Monitor::Loss)
                .min_delta(1.0e9)
                .expect("有効な min_delta")
                .restore_best_weights(true),
        ),
    ];
    let h = model
        .fit_with_callbacks(&x, &y, FitConfig::new(epochs, N), None, &mut cbs)
        .expect("fit");
    let Callback::EarlyStopping(es) = &cbs[1] else {
        panic!("test fixture");
    };
    let stopped = es.stopped_epoch().expect("打ち切られる");
    assert!(h.loss.len() < epochs);
    assert_eq!(h.loss.len(), stopped + 1);
    let be = es.best_epoch().expect("best あり");
    assert_eq!(model_bits(&model), shadows[be]);
    assert_ne!(model_bits(&model), raws[be]);
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
    let other_before = model_bits(&other);
    assert!(is_invalid_arg(other.fit_with_callbacks(
        &x,
        &y,
        FitConfig::new(1, N),
        None,
        &mut cbs
    )));
    assert_eq!(dict_bits(&shadow_of(&cbs[0])), before);
    // fit 開始時に拒否され、最初の step が走らず重みが変わらない（副作用なし）。
    assert_eq!(model_bits(&other), other_before);

    // 名前集合が同じで shape だけ異なるモデルも fit 開始時に Shape エラーで拒否される。
    let mut shaped = Sequential::new()
        .add_linear(D_IN, 8, 0x11)
        .expect("fixture")
        .add_relu()
        .add_linear(8, 3, 0x22)
        .expect("fixture");
    shaped
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .expect("compile");
    let shaped_before = model_bits(&shaped);
    let y3 = Tensor::new(vec![0.0f32; N * 3], &[N, 3]).expect("fixture");
    assert!(matches!(
        shaped.fit_with_callbacks(&x, &y3, FitConfig::new(1, N), None, &mut cbs),
        Err(AutodiffError::Shape(_))
    ));
    assert_eq!(model_bits(&shaped), shaped_before);
    assert_eq!(dict_bits(&shadow_of(&cbs[0])), before);
    assert_eq!(ema_of(&cbs[0]).num_updates(), 2);
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

// =====================================================================
// 8. EMA × AMP（イシュー #3831・決定記録 §17）。skip した step では shadow も
//    `num_updates` も変えない。skip の判定は `amp_loss_scale()` の半減で行う
//    （`growth_interval` を試行数より大きくし、増加と混ざらないようにする）。
//    低精度 forward と手書き GradScaler ループの bit 一致は
//    `compat_sequential_fit_amp.rs` が担保しており、ここはその上に推移的に乗る
//    （内部クレートを import しないための構成）。
// =====================================================================

const AMP_STEPS: usize = 8;

/// 勾配 overflow を誘発するため target を 1e3 倍にしたデータ。
fn amp_data() -> (Tensor<f32>, Tensor<f32>) {
    let (x, y) = gen_data();
    let ys: Vec<f32> = y
        .contiguous()
        .as_slice()
        .expect("test fixture: contiguous 化済み")
        .iter()
        .map(|v| v * 1.0e3)
        .collect();
    (x, Tensor::new(ys, y.shape()).expect("test fixture"))
}

fn amp_model(dtype: AmpDType) -> Sequential {
    let mut m = Sequential::new()
        .add_linear(D_IN, 8, 0x11)
        .expect("test fixture")
        .add_relu()
        .add_linear(8, D_OUT, 0x22)
        .expect("test fixture");
    m.compile_with_amp(
        Optimizer::Sgd(SgdConfig::new(0.05)),
        Loss::Mse,
        AmpConfig::new(dtype).grad_scaler(GradScalerConfig {
            init_scale: 2.0f32.powi(127),
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 1000,
        }),
    )
    .expect("test fixture");
    m
}

fn scale_of(m: &Sequential) -> f32 {
    m.amp_loss_scale().expect("test fixture: AMP compile 済み")
}

/// 参照: EMA なしの AMP モデルを 1 step ずつ `fit` し、skip でない step だけ手動で
/// `update_named` する。戻り値は (参照モデル, 手動 EMA, 各 step の skip 判定)。
fn amp_reference(
    dtype: AmpDType,
    x: &Tensor<f32>,
    y: &Tensor<f32>,
    batch: usize,
    steps: usize,
) -> (Sequential, ExponentialMovingAverage, Vec<bool>) {
    let mut m = amp_model(dtype);
    let mut ema =
        ExponentialMovingAverage::from_named(DECAY, m.named_parameters()).expect("from_named");
    let mut skips = Vec::new();
    for _ in 0..steps {
        let before = scale_of(&m);
        m.fit(x, y, FitConfig::new(1, batch)).expect("fit");
        let skipped = scale_of(&m) < before;
        if !skipped {
            ema.update_named(m.named_parameters()).expect("update");
        }
        skips.push(skipped);
    }
    (m, ema, skips)
}

fn run_amp_matches_reference(dtype: AmpDType) {
    let (x, y) = amp_data();
    let (reference, ema, skips) = amp_reference(dtype, &x, &y, N, AMP_STEPS);
    assert!(skips.iter().any(|&s| s), "skip が 1 回以上ある構成のはず");
    assert!(
        skips.iter().any(|&s| !s),
        "非 skip が 1 回以上ある構成のはず"
    );

    let mut m = amp_model(dtype);
    let mut cbs = [ema_cb(DECAY)];
    m.fit_with_callbacks(&x, &y, FitConfig::new(AMP_STEPS, N), None, &mut cbs)
        .expect("EMA×AMP は受理される");
    assert_eq!(
        dict_bits(&shadow_of(&cbs[0])),
        dict_bits(&ema.shadow_state_dict())
    );
    assert_eq!(ema_of(&cbs[0]).num_updates(), ema.num_updates());
    assert!(ema.num_updates() < AMP_STEPS as u64);
    assert_eq!(model_bits(&m), model_bits(&reference));
}

// 8a. 参照（skip を除いた手動 EMA）と shadow・num_updates が bit 一致する。
#[test]
fn ema_with_amp_matches_fit_amp_reference_bit_exact_f16() {
    run_amp_matches_reference(AmpDType::F16);
}

#[test]
fn ema_with_amp_matches_fit_amp_reference_bit_exact_bf16() {
    run_amp_matches_reference(AmpDType::Bf16);
}

// 8b. skip した step の前後で shadow・num_updates が変わらない。
#[test]
fn ema_with_amp_skips_shadow_update_on_skipped_steps() {
    let (x, y) = amp_data();
    let mut m = amp_model(AmpDType::F16);
    let initial = model_bits(&m);
    let mut cbs = [ema_cb(DECAY)];
    let (mut skipped_n, mut applied_n) = (0, 0);
    for i in 0..AMP_STEPS {
        let before_scale = scale_of(&m);
        let before_shadow = ema_of(&cbs[0]).shadow_state_dict().map(|d| dict_bits(&d));
        let before_n = ema_of(&cbs[0]).num_updates();
        m.fit_with_callbacks(&x, &y, FitConfig::new(1, N), None, &mut cbs)
            .expect("fit");
        let skipped = scale_of(&m) < before_scale;
        let after_n = ema_of(&cbs[0]).num_updates();
        if skipped {
            skipped_n += 1;
            assert_eq!(
                after_n, before_n,
                "skip step で num_updates が変化 (step={i})"
            );
            if let Some(b) = before_shadow {
                assert_eq!(
                    dict_bits(&shadow_of(&cbs[0])),
                    b,
                    "skip step で shadow が変化"
                );
            }
        } else {
            applied_n += 1;
            assert_eq!(after_n, before_n + 1);
        }
        if i == 0 {
            assert!(skipped, "先頭 step は skip する構成のはず");
            assert_eq!(ema_of(&cbs[0]).num_updates(), 0);
            assert_eq!(dict_bits(&shadow_of(&cbs[0])), initial);
        }
    }
    assert!(skipped_n >= 1 && applied_n >= 1);
}

// 8c. 1 epoch の全 step が skip しても、shadow・num_updates は不変・val_loss は push・
//     fit 後の重みは生の重み（EMA なしの AMP fit と bit 一致）。
#[test]
fn ema_with_amp_all_skipped_epoch_keeps_shadow_and_restores_raw() {
    let (x, y) = amp_data();
    let batch = N / 4;
    // 参照: EMA なしの AMP で 1 epoch（4 batch = 4 step）。全 step が skip なら scale は
    // 4 回半減し、重みは初期値のまま。
    let mut reference = amp_model(AmpDType::F16);
    let initial_scale = scale_of(&reference);
    let initial = model_bits(&reference);
    reference
        .fit(&x, &y, FitConfig::new(1, batch))
        .expect("fit");
    assert_eq!(
        scale_of(&reference),
        initial_scale / 16.0,
        "4 step 全 skip の構成のはず"
    );
    assert_eq!(model_bits(&reference), initial);
    let mut m = amp_model(AmpDType::F16);
    let mut cbs = [ema_cb(DECAY)];
    let h = m
        .fit_with_callbacks(&x, &y, FitConfig::new(1, batch), Some((&x, &y)), &mut cbs)
        .expect("fit");
    assert_eq!(ema_of(&cbs[0]).num_updates(), 0);
    assert_eq!(dict_bits(&shadow_of(&cbs[0])), initial);
    assert_eq!(h.val_loss.len(), 1);
    assert_eq!(model_bits(&m), model_bits(&reference));
}

// 8d. EMA は生の学習軌道（重み・loss・lr・scale）を乱さない。
#[test]
fn ema_with_amp_does_not_change_raw_trajectory() {
    let (x, y) = amp_data();
    let mut plain = amp_model(AmpDType::Bf16);
    let hp = plain
        .fit(&x, &y, FitConfig::new(AMP_STEPS, N))
        .expect("fit");
    let mut with = amp_model(AmpDType::Bf16);
    let mut cbs = [ema_cb(DECAY)];
    let hw = with
        .fit_with_callbacks(&x, &y, FitConfig::new(AMP_STEPS, N), None, &mut cbs)
        .expect("fit");
    assert_eq!(model_bits(&plain), model_bits(&with));
    let b = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(b(&hp.loss), b(&hw.loss));
    assert_eq!(b(&hp.lr), b(&hw.lr));
    assert_eq!(scale_of(&plain).to_bits(), scale_of(&with).to_bits());
}

// 8e. 残る拒否は fit 開始前・モデル不変・EMA 未初期化のまま。
#[test]
fn ema_with_amp_remaining_rejections_leave_state_untouched() {
    let (x, y) = gen_data();

    // AMP×EMA×accumulate_steps > 1。
    let mut m = amp_model(AmpDType::Bf16);
    let before = model_bits(&m);
    let scale = scale_of(&m);
    let mut cbs = [ema_cb(0.9)];
    assert!(is_invalid_arg(m.fit_with_callbacks(
        &x,
        &y,
        FitConfig::new(1, 4).accumulate_steps(2),
        None,
        &mut cbs
    )));
    assert_eq!(model_bits(&m), before);
    assert_eq!(scale_of(&m).to_bits(), scale.to_bits());
    assert_eq!(ema_of(&cbs[0]).num_updates(), 0);
    assert!(ema_of(&cbs[0]).shadow_state_dict().is_none());

    // AMP×EMA×カスタム step。
    let mut m = amp_model(AmpDType::Bf16);
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
    assert!(ema_of(&cbs[0]).shadow_state_dict().is_none());
}

// 8f. AMP×EMA×SWA は SWA 側の理由で引き続き拒否される（SWA×AMP は §17.3 で維持）。
#[test]
fn ema_with_amp_and_swa_is_still_rejected() {
    let (x, y) = gen_data();
    let mut m = amp_model(AmpDType::Bf16);
    let before = model_bits(&m);
    let mut cbs = [
        ema_cb(0.9),
        Callback::Swa(SwaCallback::new(0, 0.01, 0, SwaAnneal::Cos).expect("test fixture")),
    ];
    assert!(is_invalid_arg(m.fit_with_callbacks(
        &x,
        &y,
        FitConfig::new(1, N),
        None,
        &mut cbs
    )));
    assert_eq!(model_bits(&m), before);
    assert!(ema_of(&cbs[0]).shadow_state_dict().is_none());
}
