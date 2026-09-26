//! param groups（層別学習率・weight decay。イシュー #2173・親 #2131）の
//! 統合テスト。`fandhe_ai_autodiff::nn::optim::{ParamGroup, ParamGroupStep}`
//! を 6 optimizer（`AdamW`・`Adam`・`RmsProp`・`Adagrad`・`Lamb`・
//! `fandhe_ai_autodiff::optim::Sgd`）へ適用したときの契約を固定する。
//!
//! **`step()` との bit 一致契約**（R1・R2）: `step_with_groups(params,
//! grads, &[])` を繰り返した結果は、同じ入力の既存 `step()` と
//! **bit 完全一致**する（`ParamGroup` 未対応スロットは optimizer の
//! 既定 config を使うという仕様。`param_group` モジュール冒頭 doc
//! 「既定スロットの扱い」節）。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。

use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::nn::optim::{
    Adagrad, AdagradConfig, Adam, AdamConfig, AdamW, AdamWConfig, Lamb, LambConfig, ParamGroup,
    ParamGroupStep, RmsProp, RmsPropConfig,
};
use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn to_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.as_slice()
        .expect("test fixture: 生成直後の Tensor は contiguous のはず")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn vals(t: &Tensor<f32>) -> Vec<f32> {
    t.as_slice()
        .expect("test fixture: 生成直後の Tensor は contiguous のはず")
        .to_vec()
}

// =====================================================================
// tuple 形 optimizer（AdamW・Adam・RmsProp・Adagrad・Lamb）共通の
// bit 一致検査（`groups = &[]` vs `step()`）。
// =====================================================================

macro_rules! tuple_optimizer_empty_groups_bit_matches_step {
    ($test_name:ident, $opt_ty:ty, $cfg:expr) => {
        #[test]
        fn $test_name() {
            let mut opt_a: $opt_ty = <$opt_ty>::new($cfg).unwrap();
            let mut opt_b: $opt_ty = <$opt_ty>::new($cfg).unwrap();

            let p1 = t(vec![1.0, -2.0, 3.0], &[3]);
            let p2 = t(vec![0.5, 0.25], &[2]);
            let g1 = t(vec![0.1, 0.2, -0.3], &[3]);
            let g2 = t(vec![0.05, -0.1], &[2]);

            for _ in 0..5 {
                let out_a = opt_a.step(&[(&p1, &g1), (&p2, &g2)]).unwrap();
                let out_b =
                    ParamGroupStep::step_with_groups(&mut opt_b, &[&p1, &p2], &[&g1, &g2], &[])
                        .unwrap();
                assert_eq!(to_bits(&out_a[0]), to_bits(&out_b[0]));
                assert_eq!(to_bits(&out_a[1]), to_bits(&out_b[1]));
            }
        }
    };
}

tuple_optimizer_empty_groups_bit_matches_step!(
    adamw_empty_groups_bit_matches_step,
    AdamW,
    AdamWConfig::default()
);
tuple_optimizer_empty_groups_bit_matches_step!(
    adam_empty_groups_bit_matches_step,
    Adam,
    AdamConfig::default()
);
tuple_optimizer_empty_groups_bit_matches_step!(
    rmsprop_empty_groups_bit_matches_step,
    RmsProp,
    RmsPropConfig::default()
);
tuple_optimizer_empty_groups_bit_matches_step!(
    adagrad_empty_groups_bit_matches_step,
    Adagrad,
    AdagradConfig::default()
);
tuple_optimizer_empty_groups_bit_matches_step!(
    lamb_empty_groups_bit_matches_step,
    Lamb,
    LambConfig::default()
);

#[test]
fn sgd_empty_groups_bit_matches_step() {
    let cfg = SgdConfig::new(0.1).with_momentum(0.9);
    let mut opt_a = Sgd::new(cfg).unwrap();
    let mut opt_b = Sgd::new(cfg).unwrap();

    let p1 = t(vec![1.0, -2.0, 3.0], &[3]);
    let g1 = t(vec![0.1, 0.2, -0.3], &[3]);

    for _ in 0..5 {
        let out_a = opt_a.step(&[&p1], &[&g1]).unwrap();
        let out_b = ParamGroupStep::step_with_groups(&mut opt_b, &[&p1], &[&g1], &[]).unwrap();
        assert_eq!(to_bits(&out_a[0]), to_bits(&out_b[0]));
    }
}

// =====================================================================
// 単一グループが既定 config と同値なら `step()` と bit 一致する。
// =====================================================================

#[test]
fn adamw_single_group_matching_config_bit_matches_step() {
    let cfg = AdamWConfig {
        lr: 0.02,
        weight_decay: 0.05,
        ..AdamWConfig::default()
    };
    let mut opt_a = AdamW::new(cfg).unwrap();
    let mut opt_b = AdamW::new(cfg).unwrap();

    let p1 = t(vec![1.0, -2.0], &[2]);
    let g1 = t(vec![0.1, 0.2], &[2]);
    let groups = vec![ParamGroup::new(vec![0], cfg.lr, cfg.weight_decay)];

    let out_a = opt_a.step(&[(&p1, &g1)]).unwrap();
    let out_b = ParamGroupStep::step_with_groups(&mut opt_b, &[&p1], &[&g1], &groups).unwrap();
    assert_eq!(to_bits(&out_a[0]), to_bits(&out_b[0]));
}

// =====================================================================
// 2 グループで lr を変えると対象スロットのみ別更新量になる（AdamW）。
// =====================================================================

#[test]
fn adamw_two_groups_apply_independent_lr() {
    let cfg = AdamWConfig::default();
    let mut opt = AdamW::new(cfg).unwrap();

    let p0 = t(vec![1.0, 2.0], &[2]);
    let p1 = t(vec![3.0, 4.0], &[2]);
    let g0 = t(vec![0.1, 0.1], &[2]);
    let g1 = t(vec![0.1, 0.1], &[2]);

    // slot 0: lr * 0.1、slot 1: 既定 lr のまま。
    let groups = vec![ParamGroup::new(vec![0], cfg.lr * 0.1, cfg.weight_decay)];
    let out =
        ParamGroupStep::step_with_groups(&mut opt, &[&p0, &p1], &[&g0, &g1], &groups).unwrap();

    // 参照: slot0 のみを lr*0.1 の独立 AdamW インスタンスで 1 step。
    let cfg0 = AdamWConfig {
        lr: cfg.lr * 0.1,
        ..cfg
    };
    let mut ref0 = AdamW::new(cfg0).unwrap();
    let ref_out0 = ref0.step(&[(&p0, &g0)]).unwrap();

    // 参照: slot1 は既定 config の独立 AdamW インスタンスで 1 step。
    let mut ref1 = AdamW::new(cfg).unwrap();
    let ref_out1 = ref1.step(&[(&p1, &g1)]).unwrap();

    assert_eq!(to_bits(&out[0]), to_bits(&ref_out0[0]));
    assert_eq!(to_bits(&out[1]), to_bits(&ref_out1[0]));
    // 2 つの slot が異なる更新量になっている（lr が異なるため）。
    assert_ne!(vals(&out[0]), vals(&ref_out1[0]));
}

// =====================================================================
// lr=0・weight_decay=0 のグループはスロットを bit 不変に保つ
// （層凍結の近似。R2 の「層別学習率」の極端ケース）。
// =====================================================================

#[test]
fn adamw_frozen_group_leaves_slot_bit_unchanged() {
    let cfg = AdamWConfig::default();
    let mut opt = AdamW::new(cfg).unwrap();

    let p0 = t(vec![1.0, 2.0], &[2]);
    let p1 = t(vec![3.0, 4.0], &[2]);
    let g0 = t(vec![0.1, 0.1], &[2]);
    let g1 = t(vec![0.1, 0.1], &[2]);

    let groups = vec![ParamGroup::new(vec![0], 0.0, 0.0)];
    let out =
        ParamGroupStep::step_with_groups(&mut opt, &[&p0, &p1], &[&g0, &g1], &groups).unwrap();

    assert_eq!(to_bits(&out[0]), to_bits(&p0), "凍結スロットは不変のはず");
    assert_ne!(
        vals(&out[1]),
        vals(&p1),
        "非凍結スロットは更新されているはず"
    );
}

// =====================================================================
// SGD（momentum あり）でも (a)=(b) の bit 一致を確認する。
// =====================================================================

#[test]
fn sgd_momentum_two_groups_apply_independent_lr() {
    let cfg = SgdConfig::new(0.1).with_momentum(0.9);
    let mut opt = Sgd::new(cfg).unwrap();

    let p0 = t(vec![1.0], &[1]);
    let p1 = t(vec![2.0], &[1]);
    let g0 = t(vec![0.3], &[1]);
    let g1 = t(vec![0.3], &[1]);

    let groups = vec![ParamGroup::new(vec![0], 0.01, cfg.weight_decay)];
    let out =
        ParamGroupStep::step_with_groups(&mut opt, &[&p0, &p1], &[&g0, &g1], &groups).unwrap();

    let mut ref0 = Sgd::new(SgdConfig { lr: 0.01, ..cfg }).unwrap();
    let ref_out0 = ref0.step(&[&p0], &[&g0]).unwrap();
    let mut ref1 = Sgd::new(cfg).unwrap();
    let ref_out1 = ref1.step(&[&p1], &[&g1]).unwrap();

    assert_eq!(to_bits(&out[0]), to_bits(&ref_out0[0]));
    assert_eq!(to_bits(&out[1]), to_bits(&ref_out1[0]));
}

// =====================================================================
// 拒否ケース（すべて `InvalidArgument`）。
// =====================================================================

#[test]
fn adamw_rejects_empty_group_params() {
    let mut opt = AdamW::new(AdamWConfig::default()).unwrap();
    let p = t(vec![1.0], &[1]);
    let g = t(vec![0.1], &[1]);
    let groups = vec![ParamGroup::new(vec![], 0.1, 0.0)];
    let err = ParamGroupStep::step_with_groups(&mut opt, &[&p], &[&g], &groups).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn adamw_rejects_out_of_range_index() {
    let mut opt = AdamW::new(AdamWConfig::default()).unwrap();
    let p = t(vec![1.0], &[1]);
    let g = t(vec![0.1], &[1]);
    let groups = vec![ParamGroup::new(vec![5], 0.1, 0.0)];
    let err = ParamGroupStep::step_with_groups(&mut opt, &[&p], &[&g], &groups).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn adamw_rejects_duplicate_slot_across_groups() {
    let mut opt = AdamW::new(AdamWConfig::default()).unwrap();
    let p = t(vec![1.0], &[1]);
    let g = t(vec![0.1], &[1]);
    let groups = vec![
        ParamGroup::new(vec![0], 0.1, 0.0),
        ParamGroup::new(vec![0], 0.2, 0.0),
    ];
    let err = ParamGroupStep::step_with_groups(&mut opt, &[&p], &[&g], &groups).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn adamw_rejects_negative_group_lr() {
    let mut opt = AdamW::new(AdamWConfig::default()).unwrap();
    let p = t(vec![1.0], &[1]);
    let g = t(vec![0.1], &[1]);
    let groups = vec![ParamGroup::new(vec![0], -0.1, 0.0)];
    let err = ParamGroupStep::step_with_groups(&mut opt, &[&p], &[&g], &groups).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn adamw_rejects_nan_group_weight_decay() {
    let mut opt = AdamW::new(AdamWConfig::default()).unwrap();
    let p = t(vec![1.0], &[1]);
    let g = t(vec![0.1], &[1]);
    let groups = vec![ParamGroup::new(vec![0], 0.1, f32::NAN)];
    let err = ParamGroupStep::step_with_groups(&mut opt, &[&p], &[&g], &groups).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn adamw_rejects_params_grads_length_mismatch() {
    let mut opt = AdamW::new(AdamWConfig::default()).unwrap();
    let p = t(vec![1.0], &[1]);
    let g0 = t(vec![0.1], &[1]);
    let g1 = t(vec![0.2], &[1]);
    let err = ParamGroupStep::step_with_groups(&mut opt, &[&p], &[&g0, &g1], &[]).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// =====================================================================
// Err 時に状態が変わらないこと（不正グループ検出後も次の正常 step が
// 破損した状態から学習しないことの間接確認。`adamw.rs::
// state_not_mutated_after_failed_step` と同型）。
// =====================================================================

#[test]
fn adamw_state_not_mutated_after_failed_group_validation() {
    let cfg = AdamWConfig::default();
    let mut opt = AdamW::new(cfg).unwrap();
    let p1 = t(vec![1.0, 2.0], &[2]);
    let g1 = t(vec![0.1, 0.1], &[2]);

    // 1 step 目は成功させる。
    ParamGroupStep::step_with_groups(&mut opt, &[&p1], &[&g1], &[]).unwrap();
    let step_count_before = opt.step_count();

    // 2 step 目は不正グループで失敗させる。
    let bad_groups = vec![ParamGroup::new(vec![5], 0.1, 0.0)];
    let err = ParamGroupStep::step_with_groups(&mut opt, &[&p1], &[&g1], &bad_groups).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert_eq!(
        opt.step_count(),
        step_count_before,
        "不正グループ検出時に step_count が進んではならない"
    );

    // 状態が破損していないことを、同じ入力で 3 回目 step を呼んだ結果が
    // 「不正グループが渡されなかった場合の 2 回目の step」と一致する
    // ことで間接的に確認する。
    let mut opt_ref = AdamW::new(cfg).unwrap();
    opt_ref.step(&[(&p1, &g1)]).unwrap();

    let out_after_failed = ParamGroupStep::step_with_groups(&mut opt, &[&p1], &[&g1], &[]).unwrap();
    let out_ref = opt_ref.step(&[(&p1, &g1)]).unwrap();
    assert_eq!(to_bits(&out_after_failed[0]), to_bits(&out_ref[0]));
}

// =====================================================================
// 既存の単体テスト・fixture テストは無修正で green のはず（本ファイル
// では再実行しないが、`cargo test -p fandhe-ai-autodiff` 全体で担保
// する。実装計画 §8「検証方法」参照）。
// =====================================================================
