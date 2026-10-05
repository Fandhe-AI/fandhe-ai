//! callbacks（`EarlyStopping`／`ModelCheckpoint`／LR scheduler 連携）・
//! `validation_data`（イシュー #1763・親 #1618）の統合テスト。
//! `compat_sequential_fit.rs` と同じ方針で `fandhe_ai` のみを import
//! し、比較対象の手動ループも facade 経由の既存メソッドだけで組み立てる
//! （`fit_with_callbacks` が手動ループと**同一の演算列**であることを
//! bit 完全一致で検証する）。
//!
//! **決定的シード**: 重み初期化・データ生成は固定シードで駆動する
//! （`.claude/rules/coding-rust.md`）。実機（CUDA/Metal）非依存のため
//! `#[ignore]` 分離は行わない。

use std::sync::Mutex;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::{
    Callback, CsvLogger, EarlyStopping, FitConfig, JsonLogger, LambdaCallback, Loss, LrSchedule,
    Metrics, ModelCheckpoint, Monitor, Optimizer, Sequential,
};
use fandhe_ai::optim::{
    LrScheduler, PlateauMode, ReduceLrOnPlateau, ReduceLrOnPlateauConfig, Sgd, SgdConfig, StepLr,
    ThresholdMode,
};
use fandhe_ai::{AutodiffError, ShapeError, Tensor};

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

const N: usize = 16;
const D_IN: usize = 4;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 2;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_VAL: u64 = 0xBADA55;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

/// `compat_sequential_fit.rs::gen_regression_data` と同型の決定的生成
/// （本ファイルはテストバイナリが分かれるため独立に定義する）。
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

// =====================================================================
// 1. callbacks なし（`&mut []`）は既存 `fit` と bit 完全一致
// =====================================================================

#[test]
fn fit_with_callbacks_empty_matches_fit_bit_exact() {
    let (x, y) = gen_regression_data(SEED_DATA);

    let mut model_a = build_model();
    model_a
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let history_a = model_a.fit(&x, &y, FitConfig::new(4, N)).unwrap();

    let mut model_b = build_model();
    model_b
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let history_b = model_b
        .fit_with_callbacks(&x, &y, FitConfig::new(4, N), None, &mut [])
        .unwrap();

    assert_eq!(history_a.loss.len(), history_b.loss.len());
    for (a, b) in history_a.loss.iter().zip(history_b.loss.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert!(history_a.val_loss.is_empty());
    assert!(history_b.val_loss.is_empty());

    let params_a = model_a.trainable_parameters();
    let params_b = model_b.trainable_parameters();
    assert!(params_bit_exact(&params_a, &params_b));
}

// =====================================================================
// 2. callbacks なしでも history.lr は毎 epoch記録される
// =====================================================================

#[test]
fn history_lr_records_optimizer_lr_without_callbacks() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    let lr = 0.03f32;
    model
        .compile(Optimizer::Sgd(SgdConfig::new(lr)), Loss::Mse)
        .unwrap();
    let history = model.fit(&x, &y, FitConfig::new(3, N)).unwrap();
    assert_eq!(history.lr, vec![lr; 3]);
}

// =====================================================================
// 3. EarlyStopping: 損失一定（lr=0）で patience 経過後に停止する
// =====================================================================

#[test]
fn early_stopping_stops_after_patience_with_constant_loss() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    // lr=0.0（有効な学習率）で optimizer step を no-op にし、forward が
    // 決定的なため各 epoch の loss が bit 完全一致で一定になる
    // （比較の tolerance 不要）。
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.0)), Loss::Mse)
        .unwrap();

    let mut callbacks = [Callback::EarlyStopping(
        EarlyStopping::new(2)
            .monitor(Monitor::Loss)
            .min_delta(1e-6)
            .unwrap(),
    )];
    let history = model
        .fit_with_callbacks(&x, &y, FitConfig::new(10, N), None, &mut callbacks)
        .unwrap();

    // epoch0: 初回観測で改善（wait=0）。epoch1: 非改善（wait=1<patience）。
    // epoch2: 非改善（wait=2>=patience）で停止。
    assert_eq!(history.loss.len(), 3);
    match &callbacks[0] {
        Callback::EarlyStopping(es) => {
            assert_eq!(es.stopped_epoch(), Some(2));
            assert_eq!(es.best_epoch(), Some(0));
        }
        _ => panic!("expected EarlyStopping"),
    }
}

// =====================================================================
// 4. EarlyStopping: patience=0 は最初の非改善 epoch で停止する
//    （改善 epoch 直後には停止しないことを固定する）
// =====================================================================

#[test]
fn early_stopping_patience_zero_stops_at_first_non_improving_epoch() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.0)), Loss::Mse)
        .unwrap();

    let mut callbacks = [Callback::EarlyStopping(
        EarlyStopping::new(0).monitor(Monitor::Loss),
    )];
    let history = model
        .fit_with_callbacks(&x, &y, FitConfig::new(10, N), None, &mut callbacks)
        .unwrap();

    // epoch0: 初回観測で改善 → 停止しない。epoch1: 非改善 → 即停止。
    assert_eq!(history.loss.len(), 2);
    match &callbacks[0] {
        Callback::EarlyStopping(es) => assert_eq!(es.stopped_epoch(), Some(1)),
        _ => panic!("expected EarlyStopping"),
    }
}

// =====================================================================
// 5. EarlyStopping: 状態は fit 呼び出しごとにリセットされる
// =====================================================================

#[test]
fn early_stopping_state_resets_at_fit_start() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.0)), Loss::Mse)
        .unwrap();

    let es = EarlyStopping::new(2)
        .monitor(Monitor::Loss)
        .min_delta(1e-6)
        .unwrap();
    let mut callbacks = [Callback::EarlyStopping(es)];

    let history1 = model
        .fit_with_callbacks(&x, &y, FitConfig::new(10, N), None, &mut callbacks)
        .unwrap();
    assert_eq!(history1.loss.len(), 3);
    let es_after_1 = match callbacks.into_iter().next().unwrap() {
        Callback::EarlyStopping(inner) => inner,
        _ => panic!("expected EarlyStopping"),
    };
    assert_eq!(es_after_1.stopped_epoch(), Some(2));

    // 同一の（停止済み）EarlyStopping を 2 回目の fit_with_callbacks で
    // 再利用しても、reset_for_fit により即座には停止しない
    // （patience 分の epoch を再び回す）。
    let mut callbacks2 = [Callback::EarlyStopping(es_after_1)];
    let history2 = model
        .fit_with_callbacks(&x, &y, FitConfig::new(10, N), None, &mut callbacks2)
        .unwrap();
    assert_eq!(history2.loss.len(), 3);
    match &callbacks2[0] {
        Callback::EarlyStopping(es) => assert_eq!(es.stopped_epoch(), Some(2)),
        _ => panic!("expected EarlyStopping"),
    }
}

// =====================================================================
// 6. EarlyStopping: min_delta の検証（非有限・負値を拒否）
// =====================================================================

#[test]
fn early_stopping_min_delta_rejects_nan_and_negative() {
    assert!(EarlyStopping::new(1).min_delta(f32::NAN).is_err());
    assert!(EarlyStopping::new(1).min_delta(f32::INFINITY).is_err());
    assert!(EarlyStopping::new(1).min_delta(-0.1).is_err());
    assert!(EarlyStopping::new(1).min_delta(0.0).is_ok());
}

// =====================================================================
// 7. EarlyStopping: restore_best_weights は発散モデルで best（epoch 0）
//    のパラメータへ復元する
// =====================================================================

#[test]
fn early_stopping_restore_best_weights_bit_exact() {
    let (x, y) = gen_regression_data(SEED_DATA);
    // 大きな lr（momentum なし）で発散させ、epoch 0 が常に best になる
    // ことを前提検査で確認する（発散しない場合はテスト前提が崩れて
    // いるため panic で明示し、silent に別の意味論を検証してしまう
    // ことを避ける）。
    const LR: f32 = 50.0;
    const EPOCHS: usize = 5;

    let mut probe = build_model();
    probe
        .compile(Optimizer::Sgd(SgdConfig::new(LR)), Loss::Mse)
        .unwrap();
    let probe_history = probe.fit(&x, &y, FitConfig::new(EPOCHS, N)).unwrap();
    assert!(
        probe_history.loss.iter().all(|l| l.is_finite()),
        "test fixture 前提: 発散させる lr のはずが非有限値が出た: {:?}",
        probe_history.loss
    );
    assert!(
        probe_history.loss[1..]
            .iter()
            .all(|l| *l > probe_history.loss[0]),
        "test fixture 前提: epoch 0 以降のすべての epoch で loss が epoch 0 を\
         上回る（epoch 0 が best のまま）はずが崩れた: {:?}",
        probe_history.loss
    );

    // 双子モデル（同一シード）を 1 epoch だけ fit した状態 = 期待される
    // best スナップショット。
    let mut twin = build_model();
    twin.compile(Optimizer::Sgd(SgdConfig::new(LR)), Loss::Mse)
        .unwrap();
    twin.fit(&x, &y, FitConfig::new(1, N)).unwrap();
    let expected_best_params = twin.trainable_parameters();

    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(LR)), Loss::Mse)
        .unwrap();
    let mut callbacks = [Callback::EarlyStopping(
        EarlyStopping::new(EPOCHS) // 発散が続く限り停止しない = 5 epoch 完走
            .monitor(Monitor::Loss)
            .restore_best_weights(true),
    )];
    let history = model
        .fit_with_callbacks(&x, &y, FitConfig::new(EPOCHS, N), None, &mut callbacks)
        .unwrap();
    assert_eq!(history.loss.len(), EPOCHS);

    let restored_params = model.trainable_parameters();
    assert!(
        params_bit_exact(&expected_best_params, &restored_params),
        "restore_best_weights 後のパラメータが epoch 0（best）の \
         双子モデルと bit 一致しない"
    );
}

// =====================================================================
// 8. ModelCheckpoint: best_state_dict／best_epoch が改善時のみ更新される
// =====================================================================

#[test]
fn model_checkpoint_keeps_best_state_dict() {
    let (x, y) = gen_regression_data(SEED_DATA);
    const LR: f32 = 50.0;
    const EPOCHS: usize = 5;

    // 双子モデル（epoch 0 = best 相当）を用意。
    let mut twin = build_model();
    twin.compile(Optimizer::Sgd(SgdConfig::new(LR)), Loss::Mse)
        .unwrap();
    twin.fit(&x, &y, FitConfig::new(1, N)).unwrap();
    let expected_best = twin.state_dict();

    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(LR)), Loss::Mse)
        .unwrap();
    let mut callbacks = [Callback::ModelCheckpoint(
        ModelCheckpoint::new().monitor(Monitor::Loss),
    )];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(EPOCHS, N), None, &mut callbacks)
        .unwrap();

    match &callbacks[0] {
        Callback::ModelCheckpoint(mc) => {
            assert_eq!(mc.best_epoch(), Some(0));
            let best = mc
                .best_state_dict()
                .expect("test fixture: best_state_dict は Some のはず");
            let mut keys: Vec<_> = best.keys().collect();
            keys.sort();
            let mut expected_keys: Vec<_> = expected_best.keys().collect();
            expected_keys.sort();
            assert_eq!(keys, expected_keys);
            for k in keys {
                let a = &best[k];
                let b = &expected_best[k];
                assert!(
                    params_bit_exact(&[a], &[b]),
                    "ModelCheckpoint の best スナップショットがキー {k} で \
                     双子モデルの epoch 0 と bit 一致しない"
                );
            }
        }
        _ => panic!("expected ModelCheckpoint"),
    }

    // best 値がこの callback インスタンスをまたいで継続することも固定
    // する（`save_best_only(false)` で最終 epoch を上書きしても best 値
    // 自体は epoch 0 のまま）。
    let mut model2 = build_model();
    model2
        .compile(Optimizer::Sgd(SgdConfig::new(LR)), Loss::Mse)
        .unwrap();
    let mc_only = match callbacks.into_iter().next().unwrap() {
        Callback::ModelCheckpoint(mc) => mc,
        _ => panic!("expected ModelCheckpoint"),
    };
    let mut callbacks2 = [Callback::ModelCheckpoint(mc_only.save_best_only(false))];
    model2
        .fit_with_callbacks(&x, &y, FitConfig::new(1, N), None, &mut callbacks2)
        .unwrap();
    match &callbacks2[0] {
        Callback::ModelCheckpoint(mc) => {
            // save_best_only(false) の 1 epoch 追加実行で state は
            // 上書きされる（save_best_only 判定に関わらず毎 epoch 更新）
            // が、best_epoch（=0）自体は継続したまま変わらない。
            assert_eq!(mc.best_epoch(), Some(0));
            assert!(mc.best_state_dict().is_some());
        }
        _ => panic!("expected ModelCheckpoint"),
    }
}

// =====================================================================
// 9. LrSchedule::per_epoch: history.lr が StepLr::lr_at と bit 一致し、
//    最終パラメータが手動 set_lr ループと bit 一致する
// =====================================================================

#[test]
fn lr_schedule_per_epoch_matches_manual_set_lr_loop() {
    let (x, y) = gen_regression_data(SEED_DATA);
    const EPOCHS: usize = 3;
    let step_lr_for_check = StepLr::new(0.1, 1, 0.5).unwrap();

    // 手動ループ: 毎 epoch `Sgd::set_lr(step_lr.lr_at(e))` を呼んでから
    // fit_with_callbacks と同一の演算列（bind → forward → mse_loss →
    // backward → trainable_grads → Sgd::step → apply_parameters）を
    // 1 バッチ（batch_size=N）で回す。
    let mut manual_model = build_model();
    let mut manual_sgd = Sgd::new(SgdConfig::new(0.1)).unwrap();
    for e in 0..EPOCHS {
        manual_sgd.set_lr(step_lr_for_check.lr_at(e)).unwrap();
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = manual_model.bind(&tape);
            let x_var = tape.var(&x);
            let y_var = tape.var_no_grad(&y);
            let pred = bound.forward(&tape, &x_var).unwrap();
            let loss = pred.mse_loss(&y_var).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = manual_model.trainable_parameters();
            manual_sgd.step(&param_refs, &grad_refs).unwrap()
        };
        manual_model.apply_parameters(updated).unwrap();
    }

    let mut fit_model = build_model();
    fit_model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
    let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(
        StepLr::new(0.1, 1, 0.5).unwrap(),
    ))];
    let history = fit_model
        .fit_with_callbacks(&x, &y, FitConfig::new(EPOCHS, N), None, &mut callbacks)
        .unwrap();

    assert_eq!(history.lr.len(), EPOCHS);
    for (e, lr) in history.lr.iter().enumerate() {
        assert_eq!(
            lr.to_bits(),
            step_lr_for_check.lr_at(e).to_bits(),
            "epoch {e}: history.lr が StepLr::lr_at と bit 一致しない"
        );
    }

    let manual_params = manual_model.trainable_parameters();
    let fit_params = fit_model.trainable_parameters();
    assert!(
        params_bit_exact(&manual_params, &fit_params),
        "LrSchedule::per_epoch 経由の fit が手動 set_lr ループと bit 一致しない"
    );
}

// =====================================================================
// 10. LrSchedule::per_epoch: epoch カウンタは fit 呼び出しをまたいで継続する
// =====================================================================

#[test]
fn lr_schedule_epoch_counter_persists_across_fit_calls() {
    let (x, y) = gen_regression_data(SEED_DATA);

    // 2 回に分けて 2 epoch ずつ fit_with_callbacks を呼ぶ。
    let mut model_twice = build_model();
    model_twice
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
    let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(
        StepLr::new(0.1, 1, 0.5).unwrap(),
    ))];
    let h1 = model_twice
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut callbacks)
        .unwrap();
    let h2 = model_twice
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut callbacks)
        .unwrap();
    let lr_twice: Vec<f32> = h1.lr.into_iter().chain(h2.lr).collect();
    match &callbacks[0] {
        Callback::LrSchedule(ls) => assert_eq!(ls.epoch(), 4),
        _ => panic!("expected LrSchedule"),
    }

    // 新規 LrSchedule で 4 epoch を 1 回の fit_with_callbacks と比較する。
    let mut model_once = build_model();
    model_once
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
    let mut callbacks_once = [Callback::LrSchedule(LrSchedule::per_epoch(
        StepLr::new(0.1, 1, 0.5).unwrap(),
    ))];
    let h_once = model_once
        .fit_with_callbacks(&x, &y, FitConfig::new(4, N), None, &mut callbacks_once)
        .unwrap();

    assert_eq!(lr_twice.len(), h_once.lr.len());
    for (a, b) in lr_twice.iter().zip(h_once.lr.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    let params_twice = model_twice.trainable_parameters();
    let params_once = model_once.trainable_parameters();
    assert!(
        params_bit_exact(&params_twice, &params_once),
        "fit_with_callbacks(2)+fit_with_callbacks(2) が \
         fit_with_callbacks(4) と bit 一致しない（LrSchedule の epoch \
         カウンタが fit をまたいで継続していない可能性がある）"
    );
}

// =====================================================================
// 11. LrSchedule::plateau: 学習率が毎 epoch 半減し history.lr／
//     最終パラメータが手動ループと bit 一致する
// =====================================================================

#[test]
fn lr_schedule_plateau_reduces_lr_and_history_reflects_it() {
    let (x, y) = gen_regression_data(SEED_DATA);
    const EPOCHS: usize = 4;
    const BASE_LR: f32 = 0.1;

    // patience=0・threshold 巨大（Abs）にすることで「初回観測のみ改善・
    // 以降は毎回非改善」を決定的に固定する（loss の実際の増減に依存
    // しない設計）。これにより epoch 1 以降は毎 epoch 末に発火し lr が
    // 半減し続ける。
    let plateau_config = ReduceLrOnPlateauConfig {
        mode: PlateauMode::Min,
        patience: 0,
        factor: 0.5,
        threshold_mode: ThresholdMode::Abs,
        threshold: 1e30,
        cooldown: 0,
        min_lr: 0.0,
        eps: 1e-8,
    };

    // 手動ループ: epoch 開始時に `sgd.set_lr(sched.current_lr())` を
    // 適用してから 1 バッチ学習し、epoch 末に `sched.step(loss)`。
    let mut manual_model = build_model();
    let mut manual_sgd = Sgd::new(SgdConfig::new(BASE_LR)).unwrap();
    let mut manual_sched = ReduceLrOnPlateau::new(BASE_LR, plateau_config).unwrap();
    let mut manual_lr = Vec::with_capacity(EPOCHS);
    for _ in 0..EPOCHS {
        manual_sgd.set_lr(manual_sched.current_lr()).unwrap();
        manual_lr.push(manual_sched.current_lr());
        let (loss_scalar, updated) = {
            let tape = fandhe_ai::tape();
            let bound = manual_model.bind(&tape);
            let x_var = tape.var(&x);
            let y_var = tape.var_no_grad(&y);
            let pred = bound.forward(&tape, &x_var).unwrap();
            let loss = pred.mse_loss(&y_var).unwrap();
            let loss_scalar = loss
                .to_tensor()
                .get(&[])
                .expect("test fixture: loss の shape は [] のはず");
            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = manual_model.trainable_parameters();
            let updated = manual_sgd.step(&param_refs, &grad_refs).unwrap();
            (loss_scalar, updated)
        };
        manual_model.apply_parameters(updated).unwrap();
        manual_sched.step(loss_scalar).unwrap();
    }

    let mut fit_model = build_model();
    fit_model
        .compile(Optimizer::Sgd(SgdConfig::new(BASE_LR)), Loss::Mse)
        .unwrap();
    let mut callbacks = [Callback::LrSchedule(LrSchedule::plateau_with_monitor(
        ReduceLrOnPlateau::new(BASE_LR, plateau_config).unwrap(),
        Monitor::Loss,
    ))];
    let history = fit_model
        .fit_with_callbacks(&x, &y, FitConfig::new(EPOCHS, N), None, &mut callbacks)
        .unwrap();

    assert_eq!(history.lr.len(), manual_lr.len());
    for (e, (a, b)) in history.lr.iter().zip(manual_lr.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "epoch {e} で lr が不一致");
    }
    // 半減が実際に起きていることも確認する（退行検出）。
    assert!(history.lr[0] > history.lr[EPOCHS - 1]);

    let manual_params = manual_model.trainable_parameters();
    let fit_params = fit_model.trainable_parameters();
    assert!(
        params_bit_exact(&manual_params, &fit_params),
        "LrSchedule::plateau 経由の fit が手動ループと bit 一致しない"
    );
}

// =====================================================================
// 12. validation: history.val_loss.last() が fit 後の evaluate と bit 一致
// =====================================================================

#[test]
fn validation_val_loss_matches_evaluate_bit_exact() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let (x_val, y_val) = gen_regression_data(SEED_VAL);

    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let history = model
        .fit_with_callbacks(
            &x,
            &y,
            FitConfig::new(3, N),
            Some((&x_val, &y_val)),
            &mut [],
        )
        .unwrap();

    assert_eq!(history.val_loss.len(), 3);

    let direct = model.evaluate(&x_val, &y_val, N).unwrap();
    assert_eq!(
        history.val_loss.last().copied().unwrap().to_bits(),
        direct.to_bits(),
        "history.val_loss の最終値が fit 後の evaluate と bit 一致しない"
    );
}

// =====================================================================
// 13. Monitor::ValLoss（既定）× validation=None は拒否される
// =====================================================================

#[test]
fn monitor_val_loss_without_validation_is_rejected() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();

    let prev_training = model.training();
    let mut callbacks = [Callback::EarlyStopping(EarlyStopping::new(1))]; // 既定 monitor = ValLoss
    let err = model
        .fit_with_callbacks(&x, &y, FitConfig::new(3, N), None, &mut callbacks)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(model.is_compiled());
    assert_eq!(model.training(), prev_training);
}

// =====================================================================
// 14. callback 内エラーでも compile 状態・train/eval モードは復元される
// =====================================================================

struct NanAfterFirst;
impl LrScheduler for NanAfterFirst {
    fn lr_at(&self, step: usize) -> f32 {
        if step == 0 { 0.1 } else { f32::NAN }
    }
}

#[test]
fn callbacks_error_keeps_model_compiled_and_restores_mode() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();

    let prev_training = model.training();
    let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(NanAfterFirst))];
    // epoch 0 は lr=0.1 で正常に進むが、epoch 1 開始時の LR 同期で
    // `Sgd::set_lr(NaN)` が InvalidArgument を返し fit 全体が失敗する。
    let err = model
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut callbacks)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(model.is_compiled());
    assert_eq!(model.training(), prev_training);
}

// =====================================================================
// 15. shuffle(true) との併用時もグローバル RNG を直列化して安全に動く
//     ことの smoke test（`fit_minibatch_shuffle_is_reproducible_under_
//     manual_seed` と同型のロック方針）
// =====================================================================

#[test]
fn fit_with_callbacks_alongside_shuffle_is_reproducible_under_manual_seed() {
    let _guard = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let (x, y) = gen_regression_data(SEED_DATA);

    let run = || {
        fandhe_ai::manual_seed(7);
        let mut model = build_model();
        model
            .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
            .unwrap();
        let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(
            StepLr::new(0.05, 1, 0.9).unwrap(),
        ))];
        model
            .fit_with_callbacks(
                &x,
                &y,
                FitConfig::new(3, N / 4).shuffle(true),
                None,
                &mut callbacks,
            )
            .unwrap()
    };

    let history1 = run();
    let history2 = run();
    assert_eq!(history1.loss.len(), history2.loss.len());
    for (a, b) in history1.loss.iter().zip(history2.loss.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

// =====================================================================
// 16. EarlyStopping: `restore_best_weights` は epoch 途中のコール
//     バックエラー（`fit_with_callbacks` の `Err` 早期 return）を
//     またいでも適用される（イシュー #1763 PR #1883 レビュー指摘の
//     回帰テスト。従来は `LrSchedule::advance` 等のエラーが
//     `run_fit` 末尾の `restore_best_weights` を素通りしていたため、
//     `EarlyStopping` が保持していた best スナップショットが失われて
//     いた）
// =====================================================================

/// epoch 3 開始時の LR 同期でのみ `NaN` を返す（`step == 3`）。
/// `step 0..=2` は `early_stopping_restore_best_weights_bit_exact` と
/// 同じ発散用の固定 lr（`LR`）を返し、epoch 0〜2 は正常に完走させる。
struct NanAtStep3 {
    lr: f32,
}
impl LrScheduler for NanAtStep3 {
    fn lr_at(&self, step: usize) -> f32 {
        if step == 3 { f32::NAN } else { self.lr }
    }
}

#[test]
fn early_stopping_restore_best_weights_survives_mid_fit_error() {
    let (x, y) = gen_regression_data(SEED_DATA);
    // `early_stopping_restore_best_weights_bit_exact` と同じ発散用 lr
    // （momentum なしの大きな lr で epoch 0 が常に best になる）。
    const LR: f32 = 50.0;
    const EPOCHS: usize = 5;

    // 双子モデル（同一シード）を 1 epoch だけ fit した状態 = 期待される
    // best スナップショット（epoch 0）。
    let mut twin = build_model();
    twin.compile(Optimizer::Sgd(SgdConfig::new(LR)), Loss::Mse)
        .unwrap();
    twin.fit(&x, &y, FitConfig::new(1, N)).unwrap();
    let expected_best_params = twin.trainable_parameters();

    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(LR)), Loss::Mse)
        .unwrap();
    let mut callbacks = [
        Callback::LrSchedule(LrSchedule::per_epoch(NanAtStep3 { lr: LR })),
        Callback::EarlyStopping(
            // patience は EPOCHS 超のため発散による非改善では停止せず、
            // epoch 3 開始時の LR 同期エラーで fit 全体が打ち切られる
            // 経路を確実に踏む。
            EarlyStopping::new(EPOCHS + 1)
                .monitor(Monitor::Loss)
                .restore_best_weights(true),
        ),
    ];
    let err = model
        .fit_with_callbacks(&x, &y, FitConfig::new(EPOCHS, N), None, &mut callbacks)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));

    // 本題: エラー経路でも `restore_best_weights` が適用され、
    // 現在のパラメータは epoch 0（best）の双子モデルと bit 一致する。
    let restored_params = model.trainable_parameters();
    assert!(
        params_bit_exact(&expected_best_params, &restored_params),
        "epoch 途中のコールバックエラーで fit が失敗しても \
         restore_best_weights が適用されるはずが、\
         best（epoch 0）のパラメータへ復元されていない"
    );
}

// =====================================================================
// 17. イシュー #2249: 確保失敗（巨大 epochs）は非アロケーションな
//    `Shape(ElementCountOverflow)` を返し、復元経路（compile 済み状態・
//    train／eval モード）も機能する。`validation_data` ありで
//    `fit_with_callbacks` 経由の到達性を確認する。
// =====================================================================

#[test]
fn fit_with_callbacks_rejects_huge_epochs_without_panicking() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let (x_val, y_val) = gen_regression_data(SEED_VAL);

    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let was_training = model.training();

    let err = model
        .fit_with_callbacks(
            &x,
            &y,
            FitConfig::new(usize::MAX, N),
            Some((&x_val, &y_val)),
            &mut [],
        )
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ElementCountOverflow)
    ));
    assert!(
        model.is_compiled(),
        "Err 後も compile 済み状態が維持されること"
    );
    assert_eq!(
        model.training(),
        was_training,
        "Err 後も train／eval モードが呼び出し前の値へ復元されること"
    );
}
// =====================================================================
// 18. CsvLogger／JsonLogger／LambdaCallback（イシュー #2571・親 #2570・
//     ルート #2499 本文「承認範囲」節の一括承認。決定記録
//     `docs/compat-callbacks-loggers-decision.md` §5）
// =====================================================================

/// 一意な一時ディレクトリ（pid＋ナノ秒＋タグ）。`Drop` で必ず削除する。
struct TmpDir(std::path::PathBuf);

impl TmpDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        Self(std::env::temp_dir().join(format!(
            "fandhe-callbacks-2571-{tag}-{}-{nanos}",
            std::process::id()
        )))
    }
    fn path(&self, name: &str) -> std::path::PathBuf {
        self.0.join(name)
    }
}

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn gen_classification_data(seed: u64) -> (Tensor<f32>, Tensor<i32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(N * D_IN);
    let y: Vec<i32> = (0..N).map(|i| (i % D_OUT) as i32).collect();
    (
        Tensor::new(x, &[N, D_IN]).expect("test fixture: x"),
        Tensor::new(y, &[N]).expect("test fixture: y"),
    )
}

fn compiled_regression_model(lr: f32) -> Sequential {
    let mut m = build_model();
    m.compile(Optimizer::Sgd(SgdConfig::new(lr)), Loss::Mse)
        .unwrap();
    m
}

fn param_bits(m: &Sequential) -> Vec<Vec<u32>> {
    m.trainable_parameters()
        .iter()
        .map(|t| {
            t.contiguous()
                .as_slice()
                .expect("test fixture: contiguous 化済み")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect()
}

/// CSV を読み戻す（ヘッダ・各行のセル）。
fn read_csv(path: &std::path::Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let header: Vec<String> = lines.next().unwrap().split(',').map(String::from).collect();
    let rows = lines
        .map(|l| l.split(',').map(String::from).collect())
        .collect();
    (header, rows)
}

/// JSON ログを読み戻す（本実装の出力形式〈1 要素 1 行・固定キー〉前提の
/// 最小パーサ）。各要素は `(key, token)` の列。
fn read_json(path: &std::path::Path) -> Vec<Vec<(String, String)>> {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .map(str::trim)
        .filter(|l| l.starts_with('{'))
        .map(|l| {
            l.trim_end_matches(',')
                .trim_start_matches('{')
                .trim_end_matches('}')
                .split(',')
                .map(|kv| {
                    let (k, v) = kv.split_once(':').unwrap();
                    (k.trim_matches('"').to_string(), v.to_string())
                })
                .collect()
        })
        .collect()
}

/// CSV トークン（Rust `Display`）が `v` と一致するか。有限値は bit 一致、
/// 非有限値は値クラス一致。
fn csv_token_matches(tok: &str, v: f32) -> bool {
    match tok {
        "NaN" => v.is_nan(),
        "inf" => v == f32::INFINITY,
        "-inf" => v == f32::NEG_INFINITY,
        _ => tok
            .parse::<f32>()
            .map(|p| p.to_bits() == v.to_bits())
            .unwrap_or(false),
    }
}

/// JSON トークン（有限値は number、非有限値はクォート付き文字列）が `v` と
/// 一致するか。
fn json_token_matches(tok: &str, v: f32) -> bool {
    match tok {
        "\"NaN\"" => v.is_nan(),
        "\"Infinity\"" => v == f32::INFINITY,
        "\"-Infinity\"" => v == f32::NEG_INFINITY,
        _ => tok
            .parse::<f32>()
            .map(|p| p.to_bits() == v.to_bits())
            .unwrap_or(false),
    }
}

#[test]
fn loggers_and_lambda_do_not_change_history_or_params_bit_exact() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let (xv, yv) = gen_regression_data(SEED_VAL);
    let tmp = TmpDir::new("bitexact");

    let mut plain = compiled_regression_model(0.05);
    let h_plain = plain
        .fit_with_callbacks(&x, &y, FitConfig::new(3, 8), Some((&xv, &yv)), &mut [])
        .unwrap();

    let mut logged = compiled_regression_model(0.05);
    let mut cbs = [
        Callback::CsvLogger(CsvLogger::new(tmp.path("a.csv"))),
        Callback::JsonLogger(JsonLogger::new(tmp.path("a.json"))),
        Callback::Lambda(LambdaCallback::on_epoch_end(|_, _| Ok(()))),
    ];
    let h_logged = logged
        .fit_with_callbacks(&x, &y, FitConfig::new(3, 8), Some((&xv, &yv)), &mut cbs)
        .unwrap();

    assert_eq!(h_plain, h_logged);
    assert_eq!(param_bits(&plain), param_bits(&logged));
}

#[test]
fn csv_and_json_read_back_match_history_with_validation_and_metrics() {
    let (x, y) = gen_classification_data(SEED_DATA);
    let (xv, yv) = gen_classification_data(SEED_VAL);
    let tmp = TmpDir::new("readback");
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let mut cbs = [
        Callback::CsvLogger(CsvLogger::new(tmp.path("log.csv"))),
        Callback::JsonLogger(JsonLogger::new(tmp.path("nested/log.json"))),
    ];
    // metrics の並び・重複・ConfusionMatrix は列集合に影響しない。
    let metrics = [
        Metrics::F1,
        Metrics::ConfusionMatrix,
        Metrics::Accuracy,
        Metrics::F1,
    ];
    let h = model
        .fit_with_metrics(
            &x,
            &y,
            FitConfig::new(3, N),
            Some((&xv, &yv)),
            &mut cbs,
            &metrics,
        )
        .unwrap();

    let expect_cols = ["epoch", "loss", "lr", "val_loss", "val_accuracy", "val_f1"];
    let (header, rows) = read_csv(&tmp.path("log.csv"));
    assert_eq!(header, expect_cols);
    assert_eq!(rows.len(), 3);
    let json = read_json(&tmp.path("nested/log.json"));
    assert_eq!(json.len(), 3);
    for e in 0..3 {
        let want = [
            h.loss[e],
            h.lr[e],
            h.val_loss[e],
            h.val_metrics[e].accuracy.unwrap(),
            h.val_metrics[e].f1.unwrap(),
        ];
        assert_eq!(rows[e][0], e.to_string());
        assert_eq!(json[e][0], ("epoch".to_string(), e.to_string()));
        for (i, v) in want.iter().enumerate() {
            assert!(csv_token_matches(&rows[e][i + 1], *v), "csv e={e} col={i}");
            assert_eq!(json[e][i + 1].0, expect_cols[i + 1]);
            assert!(
                json_token_matches(&json[e][i + 1].1, *v),
                "json e={e} col={i}"
            );
        }
    }
}

#[test]
fn logger_columns_without_validation_are_epoch_loss_lr() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let tmp = TmpDir::new("cols");
    let mut model = compiled_regression_model(0.05);
    let mut cbs = [Callback::CsvLogger(CsvLogger::new(tmp.path("c.csv")))];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut cbs)
        .unwrap();
    let (header, rows) = read_csv(&tmp.path("c.csv"));
    assert_eq!(header, ["epoch", "loss", "lr"]);
    assert_eq!(rows.len(), 2);
}

#[test]
fn csv_append_false_overwrites_and_true_keeps_one_header() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let tmp = TmpDir::new("csvappend");
    let path = tmp.path("log.csv");
    let run = |append: bool, epochs: usize| {
        let mut m = compiled_regression_model(0.05);
        let mut cbs = [Callback::CsvLogger(CsvLogger::new(&path).append(append))];
        m.fit_with_callbacks(&x, &y, FitConfig::new(epochs, N), None, &mut cbs)
            .unwrap();
    };
    run(false, 2);
    run(false, 3);
    assert_eq!(read_csv(&path).1.len(), 3, "append=false は上書き");
    run(true, 2);
    let (header, rows) = read_csv(&path);
    assert_eq!(header, ["epoch", "loss", "lr"]);
    assert_eq!(rows.len(), 5, "ヘッダ 1 行 + 全行が残る");
    // 追記された fit の epoch は fit ローカル（0 始まり）。
    assert_eq!(rows[3][0], "0");
    assert_eq!(rows[4][0], "1");
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(text.matches("epoch,loss,lr").count(), 1);
}

#[test]
fn csv_append_to_file_without_trailing_newline_does_not_corrupt_rows() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let tmp = TmpDir::new("csvnonl");
    let path = tmp.path("log.csv");
    // 前回 fit が行の途中で中断した状態（末尾に改行なし）を作る。
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "epoch,loss,lr\n0,0.5,0.01\n1,0.4,0.01").unwrap();
    let mut m = compiled_regression_model(0.05);
    let mut cbs = [Callback::CsvLogger(CsvLogger::new(&path).append(true))];
    m.fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut cbs)
        .unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.starts_with("epoch,loss,lr\n0,0.5,0.01\n1,0.4,0.01\n"));
    let (_, rows) = read_csv(&path);
    assert_eq!(rows.len(), 4, "既存 2 行 + 追記 2 行が壊れず別行になる");
    assert_eq!(rows[2][0], "0");
    assert_eq!(rows[3][0], "1");
}

#[test]
fn csv_append_with_mismatched_header_fails_closed_and_keeps_model_state() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let (xv, yv) = gen_regression_data(SEED_VAL);
    let tmp = TmpDir::new("csvmismatch");
    let path = tmp.path("log.csv");
    // validation なし（epoch,loss,lr）で既存ログを作る。
    let mut m = compiled_regression_model(0.05);
    m.fit_with_callbacks(
        &x,
        &y,
        FitConfig::new(1, N),
        None,
        &mut [Callback::CsvLogger(CsvLogger::new(&path))],
    )
    .unwrap();
    let before = std::fs::read_to_string(&path).unwrap();

    // validation あり（val_loss 列が増える）で append すると列集合が不一致。
    let mut m2 = compiled_regression_model(0.05);
    let params_before = param_bits(&m2);
    let prev_training = m2.training();
    let mut cbs = [Callback::CsvLogger(CsvLogger::new(&path).append(true))];
    let err = m2
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), Some((&xv, &yv)), &mut cbs)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(m2.is_compiled());
    assert_eq!(m2.training(), prev_training);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
    assert_eq!(params_before, param_bits(&m2), "学習は行われない");
}

#[test]
fn json_append_merges_existing_array_and_keeps_raw_elements() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let tmp = TmpDir::new("jsonappend");
    let path = tmp.path("log.json");
    let run = |append: bool, epochs: usize| {
        let mut m = compiled_regression_model(0.05);
        let mut cbs = [Callback::JsonLogger(JsonLogger::new(&path).append(append))];
        m.fit_with_callbacks(&x, &y, FitConfig::new(epochs, N), None, &mut cbs)
            .unwrap();
    };
    run(false, 2);
    run(false, 1);
    assert_eq!(read_json(&path).len(), 1, "append=false は空配列から開始");
    run(true, 2);
    let rows = read_json(&path);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[1][0].1, "0");
    assert_eq!(rows[2][0].1, "1");

    // 外部で作った（整形の異なる）既存要素も原文のまま保持される。
    let foreign = "{\"epoch\": 9, \"loss\": 1.0, \"lr\": 0.5, \"extra\": [1,{\"a\":null}]}";
    std::fs::write(&path, format!("[ {foreign} ]")).unwrap();
    run(true, 1);
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains(foreign));
    assert_eq!(text.matches("\"epoch\"").count(), 2);
}

#[test]
fn json_append_with_invalid_existing_file_fails_closed() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let tmp = TmpDir::new("jsonbad");
    let path = tmp.path("log.json");
    std::fs::create_dir_all(&tmp.0).unwrap();
    for bad in [
        "not json",
        "{\"epoch\":0}",
        "[1, 2]",
        "[{\"epoch\":0,\"loss\":1.0}]", // lr キー不足
        "[{\"epoch\":0,\"loss\":1.0,\"lr\":0.1}] trailing",
    ] {
        std::fs::write(&path, bad).unwrap();
        let mut m = compiled_regression_model(0.05);
        let mut cbs = [Callback::JsonLogger(JsonLogger::new(&path).append(true))];
        let err = m
            .fit_with_callbacks(&x, &y, FitConfig::new(1, N), None, &mut cbs)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)), "{bad}");
        assert!(m.is_compiled());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            bad,
            "既存ファイルは不変"
        );
    }
}

#[test]
fn lambda_receives_local_epoch_and_growing_history() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let sink = std::rc::Rc::clone(&seen);
    let mut cbs = [Callback::Lambda(LambdaCallback::on_epoch_end(
        move |e, h| {
            sink.borrow_mut().push((e, h.loss.len(), h.lr.len()));
            Ok(())
        },
    ))];
    let mut m = compiled_regression_model(0.05);
    m.fit_with_callbacks(&x, &y, FitConfig::new(3, N), None, &mut cbs)
        .unwrap();
    assert_eq!(*seen.borrow(), vec![(0, 1, 1), (1, 2, 2), (2, 3, 3)]);
    // 2 回目の fit でも epoch は fit ローカル（0 始まり）。
    seen.borrow_mut().clear();
    m.fit_with_callbacks(&x, &y, FitConfig::new(1, N), None, &mut cbs)
        .unwrap();
    assert_eq!(*seen.borrow(), vec![(0, 1, 1)]);
    let dbg = format!("{:?}", cbs[0]);
    assert!(
        dbg.contains("LambdaCallback") && dbg.contains("calls: 4"),
        "{dbg}"
    );
}

#[test]
fn lambda_error_aborts_fit_and_restores_state_and_best_weights() {
    let (x, y) = gen_regression_data(SEED_DATA);
    const LR: f32 = 50.0; // epoch 0 が常に best になる発散 lr
    let mut twin = compiled_regression_model(LR);
    twin.fit(&x, &y, FitConfig::new(1, N)).unwrap();
    let expected_best = param_bits(&twin);

    let mut model = compiled_regression_model(LR);
    let prev_training = model.training();
    let calls = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let c = std::rc::Rc::clone(&calls);
    let mut cbs = [
        Callback::Lambda(LambdaCallback::on_epoch_end(move |e, _| {
            c.set(c.get() + 1);
            if e == 2 {
                Err(AutodiffError::InvalidArgument("lambda stop".to_string()))
            } else {
                Ok(())
            }
        })),
        Callback::EarlyStopping(
            EarlyStopping::new(10)
                .monitor(Monitor::Loss)
                .restore_best_weights(true),
        ),
    ];
    let err = model
        .fit_with_callbacks(&x, &y, FitConfig::new(6, N), None, &mut cbs)
        .unwrap_err();
    match err {
        AutodiffError::InvalidArgument(msg) => assert_eq!(msg, "lambda stop"),
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(calls.get(), 3, "epoch 2 で打ち切られる");
    assert!(model.is_compiled());
    assert_eq!(model.training(), prev_training);
    assert_eq!(expected_best, param_bits(&model));
}

#[test]
fn unwritable_log_path_fails_closed_before_training() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let tmp = TmpDir::new("badpath");
    std::fs::create_dir_all(&tmp.0).unwrap();
    // 親パスが通常ファイル（ディレクトリを作れない）。
    let blocker = tmp.path("blocker");
    std::fs::write(&blocker, "x").unwrap();
    let bad = blocker.join("log.out");
    for cb in [
        Callback::CsvLogger(CsvLogger::new(&bad)),
        Callback::JsonLogger(JsonLogger::new(&bad)),
    ] {
        let mut m = compiled_regression_model(0.05);
        let before = param_bits(&m);
        let prev_training = m.training();
        let mut cbs = [cb];
        let err = m
            .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut cbs)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        assert!(m.is_compiled());
        assert_eq!(m.training(), prev_training);
        assert_eq!(before, param_bits(&m));
    }
}

#[test]
fn loggers_write_the_epoch_where_early_stopping_stops() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let tmp = TmpDir::new("earlystop");
    let mut m = compiled_regression_model(0.0); // loss 一定 → 3 epoch で停止
    let mut cbs = [
        Callback::EarlyStopping(
            EarlyStopping::new(2)
                .monitor(Monitor::Loss)
                .min_delta(1e-6)
                .unwrap(),
        ),
        Callback::CsvLogger(CsvLogger::new(tmp.path("e.csv"))),
        Callback::JsonLogger(JsonLogger::new(tmp.path("e.json"))),
    ];
    let h = m
        .fit_with_callbacks(&x, &y, FitConfig::new(10, N), None, &mut cbs)
        .unwrap();
    assert_eq!(h.loss.len(), 3);
    assert_eq!(read_csv(&tmp.path("e.csv")).1.len(), 3);
    assert_eq!(read_json(&tmp.path("e.json")).len(), 3);
}
