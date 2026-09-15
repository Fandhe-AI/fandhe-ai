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
    Callback, EarlyStopping, FitConfig, Loss, LrSchedule, ModelCheckpoint, Monitor, Optimizer,
    Sequential,
};
use fandhe_ai::optim::{
    LrScheduler, PlateauMode, ReduceLrOnPlateau, ReduceLrOnPlateauConfig, Sgd, SgdConfig, StepLr,
    ThresholdMode,
};
use fandhe_ai::{AutodiffError, Tensor};

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
