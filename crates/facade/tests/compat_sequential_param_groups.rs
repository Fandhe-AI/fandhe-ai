//! param groups（層別学習率・weight decay。イシュー #2173・親 #2131）の
//! facade CPU 学習ループ統合テスト（受入条件 R4「CPU の学習ループで、
//! 層別設定の収束がベースラインと同等であることを確認する」）。
//!
//! `ParamGroup`／`ParamGroupStep` は facade（`fandhe_ai::optim`）へは
//! 未公開（承認待ち。`crates/facade/src/lib.rs::
//! ParamGroupsHoldDoctestGuard`・`docs/autodiff-param-groups-decision.md`
//! §5 参照）のため、本ファイルは `crates/facade/tests/
//! compat_sequential_train.rs` と同じく `fandhe_ai_autodiff`／
//! `fandhe_ai_tensor_core` を直接 import する（内部クレート限定の
//! 新規公開面を検証する統合テストのため、facade のみ import の契約
//! （`tests/optim_train_loop.rs` 冒頭 doc）は適用しない）。
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
use fandhe_ai::compat::Sequential;
use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::nn::optim::{AdamW, AdamWConfig, ParamGroup, ParamGroupStep as _};
use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};
use fandhe_ai_tensor_core::Tensor;

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
