//! param groups（層別学習率・weight decay。イシュー #2173・親 #2131）の
//! facade CPU 学習ループ統合テスト（受入条件 R4「CPU の学習ループで、
//! 層別設定の収束がベースラインと同等であることを確認する」）。
//!
//! `ParamGroup`／`ParamGroupStep` は #2553 で facade（`fandhe_ai::optim`）へ公開済み
//! （`docs/autodiff-param-groups-decision.md` §9.2・§12）。本ファイルは facade の公開パス
//! だけを使い、`Sequential::compile_with_param_groups` 経由の `fit`・手動ループ・拒否経路を
//! 検証する（既存 4 テストは公開型へ import を切り替えただけで検証内容を保つ）。
//!
//! **決定的シード**: `compat_sequential_train.rs` と同一。
//!
//! **数値判定の規律**: (a)/(b) はビット一致（tolerance 不使用）。
//! (c)/(d) の収束判定は本テスト固有の新しい判定基準として定義し、
//! 既存 tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）
//! とは無関係（`.claude/rules/coding-rust.md`「許容誤差を単独で緩和
//! しない」の対象外。新設のため）。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない（ホスト
//! 経路のみで GPU 分岐がないため）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::{
    Callback, FitConfig, Loss, LrSchedule, ModelIoError, Optimizer, Sequential, TrainStepFn,
    TrainStepOptimizer, TrainStepOutput, load_model, save_model,
};
use fandhe_ai::optim::{
    AdamW, AdamWConfig, Adamax, AdamaxConfig, LbfgsConfig, ParamGroup, ParamGroupStep as _, Sgd,
    SgdConfig, StepLr,
};
use fandhe_ai::{AutodiffError, Tensor};

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap()
}

/// 第 1 Linear（named_parameters の `"0."` 接頭辞）・第 2 Linear
/// （`"2."` 接頭辞）それぞれのスロット添字を、`named_parameters` の
/// 接頭辞から導出する（`trainable_parameters` と完全同順であることを
/// `Sequential::named_parameters` doc「順序契約」節が保証する）。
fn layer_slot_indices(model: &Sequential, layer_prefix: &str) -> Vec<usize> {
    model
        .named_parameters()
        .iter()
        .enumerate()
        .filter_map(|(idx, (name, _))| name.starts_with(layer_prefix).then_some(idx))
        .collect()
}

/// `AdamW::step_with_groups` を使う 1 step。`groups` が空なら
/// `step_with_slot_hparams`／`step()` と bit 一致する契約
/// （`param_group` モジュール冒頭 doc）を利用し、(a) 手動 `step()` と
/// (b) 空グループ経由の `step_with_groups` を同一関数で切り替える。
fn train_step(
    model: &mut Sequential,
    opt: &mut AdamW,
    x_data: &Tensor<f32>,
    y_data: &Tensor<f32>,
    groups: &[ParamGroup],
    log: &mut Vec<f32>,
) -> Result<(), AutodiffError> {
    let updated = {
        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let x = tape.var(x_data);
        let y = tape.var(y_data);

        let pred = bound.forward(&tape, &x)?;
        let loss = pred.mse_loss(&y)?;
        log.push(scalar(&loss.to_tensor()));

        let grads = tape.backward(&loss)?;
        let grad_refs = bound.trainable_grads(&grads)?;
        let param_refs = model.trainable_parameters();
        opt.step_with_groups(&param_refs, &grad_refs, groups)?
    };
    model.apply_parameters(updated)?;
    Ok(())
}

fn train_step_sgd(
    model: &mut Sequential,
    opt: &mut Sgd,
    x_data: &Tensor<f32>,
    y_data: &Tensor<f32>,
    groups: &[ParamGroup],
    log: &mut Vec<f32>,
) -> Result<(), AutodiffError> {
    let updated = {
        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let x = tape.var(x_data);
        let y = tape.var(y_data);

        let pred = bound.forward(&tape, &x)?;
        let loss = pred.mse_loss(&y)?;
        log.push(scalar(&loss.to_tensor()));

        let grads = tape.backward(&loss)?;
        let grad_refs = bound.trainable_grads(&grads)?;
        let param_refs = model.trainable_parameters();
        opt.step_with_groups(&param_refs, &grad_refs, groups)?
    };
    model.apply_parameters(updated)?;
    Ok(())
}

// =====================================================================
// (a) vs (b): 全層を既定値と同じ 1 グループへ入れた場合、`step_with_
// groups` の手動ループは空グループの手動ループ（＝既存 `AdamW::step`）
// と bit 完全一致する。
// =====================================================================

#[test]
fn adamw_all_default_group_bit_matches_empty_groups() {
    const STEPS: usize = 20;
    let cfg = AdamWConfig::default();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let mut model_a = build_model();
    let mut opt_a = AdamW::new(cfg).unwrap();
    let mut log_a = Vec::new();

    let mut model_b = build_model();
    let mut opt_b = AdamW::new(cfg).unwrap();
    let mut log_b = Vec::new();
    let n_slots = model_b.trainable_parameters().len();
    let all_default_group = vec![ParamGroup::new(
        (0..n_slots).collect(),
        cfg.lr,
        cfg.weight_decay,
    )];

    for _ in 0..STEPS {
        train_step(&mut model_a, &mut opt_a, &x_data, &y_data, &[], &mut log_a).unwrap();
        train_step(
            &mut model_b,
            &mut opt_b,
            &x_data,
            &y_data,
            &all_default_group,
            &mut log_b,
        )
        .unwrap();
    }

    assert_eq!(log_a, log_b, "loss 履歴が bit 一致しない");
    let final_a = model_a.named_parameters();
    let final_b = model_b.named_parameters();
    for ((name_a, ta), (name_b, tb)) in final_a.iter().zip(final_b.iter()) {
        assert_eq!(name_a, name_b);
        assert_eq!(
            ta.as_slice().unwrap(),
            tb.as_slice().unwrap(),
            "パラメータ {name_a} が bit 一致しない"
        );
    }
}

// =====================================================================
// (c) 層別 lr で収束すること（第 1 Linear は lr×0.5、第 2 Linear は
// lr×2）。
// =====================================================================

#[test]
fn adamw_per_layer_lr_converges() {
    const STEPS: usize = 300;
    let cfg = AdamWConfig::default();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    // (a) ベースライン: 既定 lr の空グループループ。
    let mut model_a = build_model();
    let mut opt_a = AdamW::new(cfg).unwrap();
    let mut log_a = Vec::new();
    for _ in 0..STEPS {
        train_step(&mut model_a, &mut opt_a, &x_data, &y_data, &[], &mut log_a).unwrap();
    }

    // (c) 層別 lr: 第 1 Linear は lr×0.5、第 2 Linear は lr×2。
    let mut model_c = build_model();
    let l1_slots = layer_slot_indices(&model_c, "0.");
    let l2_slots = layer_slot_indices(&model_c, "2.");
    assert_eq!(l1_slots, vec![0, 1]);
    assert_eq!(l2_slots, vec![2, 3]);
    let groups_c = vec![
        ParamGroup::new(l1_slots, cfg.lr * 0.5, cfg.weight_decay),
        ParamGroup::new(l2_slots, cfg.lr * 2.0, cfg.weight_decay),
    ];
    let mut opt_c = AdamW::new(cfg).unwrap();
    let mut log_c = Vec::new();
    for _ in 0..STEPS {
        train_step(
            &mut model_c,
            &mut opt_c,
            &x_data,
            &y_data,
            &groups_c,
            &mut log_c,
        )
        .unwrap();
    }

    let initial_a = log_a[0];
    let final_a = *log_a.last().unwrap();
    let initial_c = log_c[0];
    let final_c = *log_c.last().unwrap();

    // 両方とも十分収束していること（CPU 実測: 既定構成で lr=1e-3 の
    // AdamW・300 step の regression MLP は initial の 10% 未満まで
    // 減少する〈100 step では約 30% までしか下がらなかったため、
    // 実測に基づき STEPS を 300 へ引き上げた〉。閾値には十分な余裕を
    // 持たせる）。
    assert!(
        final_a <= 0.1 * initial_a,
        "ベースラインが収束していない: initial={initial_a} final={final_a}"
    );
    assert!(
        final_c <= 0.1 * initial_c,
        "層別 lr 構成が収束していない: initial={initial_c} final={final_c}"
    );
    // 層別 lr の収束点がベースラインと同等（劣化が 2 倍以内）であること
    // （R4「層別設定の収束がベースラインと同等」の判定基準）。
    assert!(
        final_c <= 2.0 * final_a,
        "層別 lr 構成の収束がベースラインより著しく悪化している: \
         final_a={final_a} final_c={final_c}"
    );
}

// =====================================================================
// (d) `lr=0, weight_decay=0` のグループへ入れた層は bit 不変（層凍結の
// 近似）。
// =====================================================================

#[test]
fn adamw_frozen_layer_group_stays_bit_unchanged_while_other_layer_trains() {
    const STEPS: usize = 20;
    let cfg = AdamWConfig::default();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let mut model = build_model();
    let l1_slots = layer_slot_indices(&model, "0.");
    let l1_before: Vec<Vec<f32>> = model
        .trainable_parameters()
        .iter()
        .enumerate()
        .filter(|(idx, _)| l1_slots.contains(idx))
        .map(|(_, t)| t.as_slice().unwrap().to_vec())
        .collect();

    let groups = vec![ParamGroup::new(l1_slots.clone(), 0.0, 0.0)];
    let mut opt = AdamW::new(cfg).unwrap();
    let mut log = Vec::new();
    for _ in 0..STEPS {
        train_step(&mut model, &mut opt, &x_data, &y_data, &groups, &mut log).unwrap();
    }

    let params_after = model.trainable_parameters();
    for (idx, expected) in l1_slots.iter().zip(l1_before.iter()) {
        assert_eq!(
            params_after[*idx].as_slice().unwrap(),
            expected.as_slice(),
            "凍結レイヤーのスロット {idx} が変化してしまった"
        );
    }
    assert!(
        log.last().unwrap() < &log[0],
        "非凍結レイヤーのみでも loss は減少するはず"
    );
}

// =====================================================================
// Sgd（momentum あり）でも (a)=(b) の bit 一致を 1 ケース確認する。
// =====================================================================

#[test]
fn sgd_momentum_all_default_group_bit_matches_empty_groups() {
    const STEPS: usize = 20;
    let cfg = SgdConfig::new(0.05).with_momentum(0.9);
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let mut model_a = build_model();
    let mut opt_a = Sgd::new(cfg).unwrap();
    let mut log_a = Vec::new();

    let mut model_b = build_model();
    let mut opt_b = Sgd::new(cfg).unwrap();
    let mut log_b = Vec::new();
    let n_slots = model_b.trainable_parameters().len();
    let all_default_group = vec![ParamGroup::new(
        (0..n_slots).collect(),
        cfg.lr,
        cfg.weight_decay,
    )];

    for _ in 0..STEPS {
        train_step_sgd(&mut model_a, &mut opt_a, &x_data, &y_data, &[], &mut log_a).unwrap();
        train_step_sgd(
            &mut model_b,
            &mut opt_b,
            &x_data,
            &y_data,
            &all_default_group,
            &mut log_b,
        )
        .unwrap();
    }

    assert_eq!(log_a, log_b, "loss 履歴が bit 一致しない");
}
// =====================================================================
// #2553: `compile_with_param_groups`（公開経路）。
// =====================================================================

const EPOCHS: usize = 5;

fn fit_config() -> FitConfig {
    // 全バッチ 1 step / epoch・shuffle なし（決定的）。
    FitConfig::new(EPOCHS, BATCH)
}

fn snapshot(model: &Sequential) -> Vec<Vec<f32>> {
    model
        .trainable_parameters()
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect()
}

fn assert_invalid_argument<T: std::fmt::Debug>(r: Result<T, AutodiffError>, what: &str) {
    assert!(
        matches!(r, Err(AutodiffError::InvalidArgument(_))),
        "{what}: InvalidArgument のはず: {r:?}"
    );
}

/// T1: groups が空なら `compile()` と bit 一致する（Sgd・AdamW）。
#[test]
fn empty_groups_bit_match_plain_compile() {
    let (x, y) = gen_regression_data(SEED_DATA);
    for optimizer in [
        Optimizer::Sgd(SgdConfig::new(0.05).with_momentum(0.9)),
        Optimizer::AdamW(AdamWConfig::default()),
    ] {
        let mut a = build_model();
        a.compile(optimizer, Loss::Mse).unwrap();
        let ha = a.fit(&x, &y, fit_config()).unwrap();
        let mut b = build_model();
        b.compile_with_param_groups(optimizer, Loss::Mse, &[])
            .unwrap();
        let hb = b.fit(&x, &y, fit_config()).unwrap();
        assert_eq!(ha.loss, hb.loss);
        assert_eq!(ha.lr, hb.lr);
        assert_eq!(snapshot(&a), snapshot(&b));
    }
}

/// T2: 層別 lr の `fit` が手動ループ（`step_with_groups`）と bit 一致する。
#[test]
fn fit_with_layer_groups_bit_matches_manual_loop() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let cfg = SgdConfig::new(0.05);
    let probe = build_model();
    let groups = vec![
        ParamGroup::new(layer_slot_indices(&probe, "0."), 0.01, 0.0),
        ParamGroup::new(layer_slot_indices(&probe, "2."), 0.2, 0.0),
    ];

    let mut fitted = build_model();
    fitted
        .compile_with_param_groups(Optimizer::Sgd(cfg), Loss::Mse, &groups)
        .unwrap();
    let history = fitted.fit(&x, &y, fit_config()).unwrap();

    let mut manual = build_model();
    let mut opt = Sgd::new(cfg).unwrap();
    let mut log = Vec::new();
    for _ in 0..EPOCHS {
        train_step_sgd(&mut manual, &mut opt, &x, &y, &groups, &mut log).unwrap();
    }
    assert_eq!(snapshot(&fitted), snapshot(&manual));
    assert_eq!(history.loss, log);
}

/// T3: `lr = 0, wd = 0` のグループへ入れた層は fit 後も不変で、他の層は変わる。
#[test]
fn fit_freezes_grouped_layer() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    let l1 = layer_slot_indices(&model, "0.");
    let l2 = layer_slot_indices(&model, "2.");
    let before = snapshot(&model);
    model
        .compile_with_param_groups(
            Optimizer::AdamW(AdamWConfig::default()),
            Loss::Mse,
            &[ParamGroup::new(l1.clone(), 0.0, 0.0)],
        )
        .unwrap();
    model.fit(&x, &y, fit_config()).unwrap();
    let after = snapshot(&model);
    for &i in &l1 {
        assert_eq!(after[i], before[i], "凍結層のスロット {i} が変化した");
    }
    assert!(
        l2.iter().any(|&i| after[i] != before[i]),
        "他の層が更新されていない"
    );
}

/// T4: groups 非空 × `LrSchedule` は拒否。状態は保たれ、groups 空なら従来どおり動く。
#[test]
fn lr_schedule_is_rejected_only_when_groups_present() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let sched = || {
        [Callback::LrSchedule(LrSchedule::per_epoch(
            StepLr::new(0.1, 1, 0.5).unwrap(),
        ))]
    };
    let mut model = build_model();
    let before = snapshot(&model);
    model
        .compile_with_param_groups(
            Optimizer::Sgd(SgdConfig::new(0.1)),
            Loss::Mse,
            &[ParamGroup::new(vec![0], 0.1, 0.0)],
        )
        .unwrap();
    assert_invalid_argument(
        model.fit_with_callbacks(&x, &y, fit_config(), None, &mut sched()),
        "groups 非空 × LrSchedule",
    );
    assert_eq!(snapshot(&model), before, "拒否時にパラメータが変化した");
    assert!(model.is_compiled(), "拒否後も compile 状態が残るはず");
    model.fit(&x, &y, fit_config()).unwrap();

    let mut plain = build_model();
    plain
        .compile_with_param_groups(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse, &[])
        .unwrap();
    plain
        .fit_with_callbacks(&x, &y, fit_config(), None, &mut sched())
        .unwrap();
}

/// T5: `Lbfgs` × groups 非空は compile で拒否され直前の compile 状態が残る。`&[]` は成功する。
#[test]
fn lbfgs_with_groups_is_rejected_at_compile() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
    assert_invalid_argument(
        model.compile_with_param_groups(
            Optimizer::Lbfgs(LbfgsConfig::default()),
            Loss::Mse,
            &[ParamGroup::new(vec![0], 0.1, 0.0)],
        ),
        "Lbfgs × groups",
    );
    assert!(model.is_compiled());
    model.fit(&x, &y, fit_config()).unwrap();
    model
        .compile_with_param_groups(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse, &[])
        .unwrap();
}

/// T6: 検証違反は `fit` から `InvalidArgument` で返り、パラメータは不変。
#[test]
fn invalid_groups_are_rejected_by_fit_without_touching_params() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let cases: Vec<(&str, Vec<ParamGroup>)> = vec![
        ("範囲外", vec![ParamGroup::new(vec![99], 0.1, 0.0)]),
        ("空 params", vec![ParamGroup::new(vec![], 0.1, 0.0)]),
        (
            "重複",
            vec![
                ParamGroup::new(vec![0], 0.1, 0.0),
                ParamGroup::new(vec![0, 1], 0.1, 0.0),
            ],
        ),
        ("lr 非有限", vec![ParamGroup::new(vec![0], f32::NAN, 0.0)]),
        ("lr 負値", vec![ParamGroup::new(vec![0], -0.1, 0.0)]),
        ("wd 負値", vec![ParamGroup::new(vec![0], 0.1, -1.0)]),
    ];
    for (label, groups) in cases {
        let mut model = build_model();
        let before = snapshot(&model);
        model
            .compile_with_param_groups(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse, &groups)
            .unwrap();
        assert_invalid_argument(model.fit(&x, &y, fit_config()), label);
        assert_eq!(snapshot(&model), before, "{label}: パラメータが変化した");
    }
}

/// T7: 勾配累積と併用でき、手動で累積した結果と bit 一致する。
#[test]
fn accumulate_steps_with_groups_matches_manual_accumulation() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let cfg = SgdConfig::new(0.05);
    let probe = build_model();
    let l1 = layer_slot_indices(&probe, "0.");
    let groups = vec![ParamGroup::new(l1.clone(), 0.0, 0.0)];

    let mut fitted = build_model();
    let before = snapshot(&fitted);
    fitted
        .compile_with_param_groups(Optimizer::Sgd(cfg), Loss::Mse, &groups)
        .unwrap();
    // batch 2 × accumulate 2 = 全 4 サンプルで 1 step / epoch。
    fitted
        .fit(&x, &y, FitConfig::new(EPOCHS, 2).accumulate_steps(2))
        .unwrap();
    let after = snapshot(&fitted);
    for &i in &l1 {
        assert_eq!(after[i], before[i], "凍結層のスロット {i} が変化した");
    }

    let xs = x.as_slice().unwrap();
    let ys = y.as_slice().unwrap();
    let mut manual = build_model();
    let mut opt = Sgd::new(cfg).unwrap();
    for _ in 0..EPOCHS {
        let mut acc: Option<Vec<Tensor<f32>>> = None;
        for half in 0..2 {
            let xb = tensor(
                xs[half * 2 * D_IN..(half + 1) * 2 * D_IN].to_vec(),
                &[2, D_IN],
            );
            let yb = tensor(
                ys[half * 2 * D_OUT..(half + 1) * 2 * D_OUT].to_vec(),
                &[2, D_OUT],
            );
            let tape = fandhe_ai::tape();
            let bound = manual.bind(&tape);
            let pred = bound.forward(&tape, &tape.var(&xb)).unwrap();
            let loss = pred.mse_loss(&tape.var(&yb)).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            acc = Some(match acc {
                None => grad_refs.iter().map(|g| (*g).clone()).collect(),
                Some(prev) => prev
                    .iter()
                    .zip(grad_refs.iter())
                    .map(|(a, g)| {
                        let v: Vec<f32> = a
                            .as_slice()
                            .unwrap()
                            .iter()
                            .zip(g.as_slice().unwrap())
                            .map(|(p, q)| p + q)
                            .collect();
                        tensor(v, a.shape())
                    })
                    .collect(),
            });
        }
        let acc = acc.unwrap();
        let acc_refs: Vec<&Tensor<f32>> = acc.iter().collect();
        let updated = {
            let params = manual.trainable_parameters();
            opt.step_with_groups(&params, &acc_refs, &groups).unwrap()
        };
        manual.apply_parameters(updated).unwrap();
    }
    assert_eq!(snapshot(&fitted), snapshot(&manual));
}

/// T8: `fit_with_train_step` のフック内 `opt.step` に compile 時の groups が効く。
#[test]
fn custom_train_step_applies_compile_time_groups() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    let l1 = layer_slot_indices(&model, "0.");
    let before = snapshot(&model);
    model
        .compile_with_param_groups(
            Optimizer::Sgd(SgdConfig::new(0.1)),
            Loss::Mse,
            &[ParamGroup::new(l1.clone(), 0.0, 0.0)],
        )
        .unwrap();
    let mut step = |m: &Sequential,
                    xb: &Tensor<f32>,
                    yb: &Tensor<f32>,
                    opt: &mut TrainStepOptimizer<'_>|
     -> Result<TrainStepOutput, AutodiffError> {
        let tape = fandhe_ai::tape();
        let bound = m.bind(&tape);
        let pred = bound.forward(&tape, &tape.var(xb))?;
        let loss = pred.mse_loss(&tape.var(yb))?;
        let value = loss.to_tensor().get(&[]).unwrap_or(f32::NAN);
        let grads = tape.backward(&loss)?;
        let grad_refs = bound.trainable_grads(&grads)?;
        let stepped = opt.step(&m.trainable_parameters(), &grad_refs)?;
        Ok(TrainStepOutput::new(value).with_updated(stepped))
    };
    let hook: &mut TrainStepFn<'_, f32> = &mut step;
    model
        .fit_with_train_step(&x, &y, fit_config(), None, &mut [], &[], hook)
        .unwrap();
    let after = snapshot(&model);
    for &i in &l1 {
        assert_eq!(after[i], before[i], "フック経由でも凍結層が不変のはず");
    }
    assert_ne!(after[2], before[2], "グループ外の層は更新される");
}

/// T9: groups 非空では `save_model` が失敗し何も残さない。`&[]` は従来どおり保存・復元できる。
#[test]
fn save_model_rejects_groups_and_keeps_empty_groups_working() {
    let dir = std::env::temp_dir().join(format!("fandhe_ai_2553_save_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut model = build_model();
    model
        .compile_with_param_groups(
            Optimizer::Sgd(SgdConfig::new(0.1)),
            Loss::Mse,
            &[ParamGroup::new(vec![0], 0.0, 0.0)],
        )
        .unwrap();
    let err = save_model(&model, &dir).unwrap_err();
    assert!(
        matches!(
            err,
            ModelIoError::Autodiff(AutodiffError::InvalidArgument(_))
        ),
        "{err:?}"
    );
    assert!(!dir.exists(), "拒否時に保存先を作ってはならない");

    let mut plain = build_model();
    plain
        .compile_with_param_groups(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse, &[])
        .unwrap();
    save_model(&plain, &dir).unwrap();
    let restored = load_model(&dir).unwrap();
    assert_eq!(snapshot(&restored), snapshot(&plain));
    let _ = std::fs::remove_dir_all(&dir);
}

/// T10: `Optimizer` enum に無い種（Adamax）の `step_with_groups` を facade の import だけで呼べる。
#[test]
fn adamax_step_with_groups_is_reachable_from_facade() {
    let cfg = AdamaxConfig::default();
    let mut opt = Adamax::new(cfg).unwrap();
    let p0 = tensor(vec![1.0, 1.0], &[2]);
    let p1 = tensor(vec![1.0, 1.0], &[2]);
    let g = tensor(vec![0.5, 0.5], &[2]);
    let groups = [ParamGroup::new(vec![1], 0.0, 0.0)];
    let out = opt
        .step_with_groups(&[&p0, &p1], &[&g, &g], &groups)
        .unwrap();
    assert_ne!(out[0].as_slice().unwrap(), p0.as_slice().unwrap());
    assert_eq!(out[1].as_slice().unwrap(), p1.as_slice().unwrap());
}
/// #2554: 決定記録 §9.2 項目 4 の契約を固定する。スロット添字の公開ヘルパーを足さない
/// 代わりに、`named_parameters()` の列挙位置が `trainable_parameters()`（step に渡る
/// `params`）の位置、すなわちスロット添字と一致する。3 層モデルの中間層だけを
/// `lr = 0`・`weight_decay = 0` のグループへ入れ、その層のスロットだけが bit 不変で
/// 他の層が更新されることを、名前接頭辞から導いた添字で確認する。
#[test]
fn named_parameters_position_is_slot_index_for_groups() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_HIDDEN, SEED_L2)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2 ^ SEED_L1)
        .unwrap();

    // 列挙位置 = スロット添字: 長さ・順序・値が bit 一致する。
    let named = model.named_parameters();
    let trainable = model.trainable_parameters();
    assert_eq!(named.len(), trainable.len());
    for (i, ((_, np), tp)) in named.iter().zip(trainable.iter()).enumerate() {
        assert_eq!(
            np.as_slice().unwrap(),
            tp.as_slice().unwrap(),
            "スロット {i} の named_parameters と trainable_parameters が不一致"
        );
    }

    let middle = layer_slot_indices(&model, "2.");
    let others: Vec<usize> = (0..trainable.len())
        .filter(|i| !middle.contains(i))
        .collect();
    assert!(!middle.is_empty() && !others.is_empty());
    let before = snapshot(&model);
    model
        .compile_with_param_groups(
            Optimizer::AdamW(AdamWConfig::default()),
            Loss::Mse,
            &[ParamGroup::new(middle.clone(), 0.0, 0.0)],
        )
        .unwrap();
    model.fit(&x, &y, fit_config()).unwrap();
    let after = snapshot(&model);
    for &i in &middle {
        assert_eq!(
            after[i], before[i],
            "凍結した中間層のスロット {i} が変化した"
        );
    }
    for &i in &others {
        assert_ne!(
            after[i], before[i],
            "凍結対象外のスロット {i} が更新されていない"
        );
    }
}
