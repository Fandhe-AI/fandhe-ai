//! SWA の `fit_with_callbacks` 結線（イシュー #3058・親 #3057。決定記録
//! `docs/autodiff-swa-decision.md` §16・§17）の統合テスト。`fandhe_ai` の公開パスのみを
//! import し、内部クレートは直接 import しない。
//!
//! 判定は bit 完全一致を基本とし、許容誤差定数は追加・変更しない（f64 平均との補助比較のみ
//! 統一複合判定「相対 1e-3 未満 または 絶対 1e-5 未満」を既存規約どおり使う）。
//! 実機（CUDA／Metal）非依存のホスト計算のみのため `#[ignore]` 分離は行わない。
//! 学習率アニールと拒否条件は #3059 の範囲でありここでは扱わない。

use std::collections::{BTreeMap, HashMap};

use fandhe_ai::compat::{Callback, FitConfig, Loss, Optimizer, Sequential, SwaCallback};
use fandhe_ai::optim::{AveragedModel, SgdConfig, SwaAnneal};
use fandhe_ai::{AutodiffError, Tensor};

const N: usize = 16;
const D_IN: usize = 4;
const D_OUT: usize = 2;
const SWA_LR: f32 = 0.05;

fn gen_data() -> (Tensor<f32>, Tensor<f32>) {
    // 決定的な疑似データ（シード固定の線形合同法）。
    let mut state = 0x3058_u64;
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

fn build_model_with_hidden(hidden: usize) -> Sequential {
    let mut m = Sequential::new()
        .add_linear(D_IN, hidden, 0x11)
        .expect("test fixture")
        .add_relu()
        .add_linear(hidden, D_OUT, 0x22)
        .expect("test fixture");
    m.compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .expect("test fixture");
    m
}

fn build_model() -> Sequential {
    build_model_with_hidden(8)
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

fn swa_cb(start: usize) -> Callback {
    Callback::Swa(SwaCallback::new(start, SWA_LR, 0, SwaAnneal::Cos).expect("test fixture"))
}

fn swa_of(cb: &Callback) -> &SwaCallback {
    let Callback::Swa(s) = cb else {
        panic!("test fixture: Callback::Swa のはず");
    };
    s
}

fn is_invalid_arg<T: std::fmt::Debug>(r: Result<T, AutodiffError>) -> bool {
    matches!(r, Err(AutodiffError::InvalidArgument(_)))
}

// 1・2. 手動ループと bit 一致。fit 後の重み = 平均化重み、raw_state_dict = 最終の生の重み。
#[test]
fn fit_swa_matches_manual_loop_bit_exact() {
    let (x, y) = gen_data();
    let (epochs, start) = (5usize, 2usize);

    let mut model_a = build_model();
    let mut cbs = [swa_cb(start)];
    model_a
        .fit_with_callbacks(&x, &y, FitConfig::new(epochs, N), None, &mut cbs)
        .expect("test fixture: fit");

    let mut model_b = build_model();
    let mut avg = AveragedModel::from_named(model_b.named_parameters()).expect("test fixture");
    let mut snapshots: Vec<HashMap<String, Tensor<f32>>> = Vec::new();
    for e in 0..epochs {
        model_b
            .fit(&x, &y, FitConfig::new(1, N))
            .expect("test fixture: fit");
        if e >= start {
            avg.update_named(model_b.named_parameters())
                .expect("test fixture");
            snapshots.push(model_b.state_dict());
        }
    }

    let s = swa_of(&cbs[0]);
    assert_eq!(s.n_averaged(), (epochs - start) as u64);
    assert_eq!(s.n_averaged(), avg.n_averaged());
    let averaged = s.averaged_state_dict().expect("test fixture");
    assert_eq!(dict_bits(&averaged), dict_bits(&avg.averaged_state_dict()));
    // 受入基準 1: Ok 終了後のモデルの重み = 平均化重み。
    assert_eq!(model_bits(&model_a), dict_bits(&averaged));
    // raw は最終の生の重み（参照モデルの最終重み）と bit 一致する。
    let raw = s.raw_state_dict().expect("test fixture");
    assert_eq!(dict_bits(&raw), model_bits(&model_b));

    // 補助: epoch スナップショットの f64 平均と統一複合判定で一致する。
    for (name, got) in &averaged {
        let g = got.contiguous();
        let g = g.as_slice().expect("test fixture");
        for (i, gv) in g.iter().enumerate() {
            let mean: f64 = snapshots
                .iter()
                .map(|sn| {
                    let t = sn[name].contiguous();
                    f64::from(t.as_slice().expect("test fixture")[i])
                })
                .sum::<f64>()
                / snapshots.len() as f64;
            let abs = (f64::from(*gv) - mean).abs();
            let rel = abs / mean.abs().max(f64::MIN_POSITIVE);
            assert!(rel < 1e-3 || abs < 1e-5, "{name}[{i}]: {gv} vs {mean}");
        }
    }

    // 退避を適用すると生の重みへ戻る。
    model_a.load_state_dict(raw).expect("test fixture");
    assert_eq!(model_bits(&model_a), model_bits(&model_b));
}

// 3. swa_start 以上の epoch 数（n_averaged == 0）: 上書きせず raw も None。
#[test]
fn swa_start_beyond_epochs_does_not_overwrite() {
    let (x, y) = gen_data();
    let mut with = build_model();
    let mut cbs = [swa_cb(10)];
    with.fit_with_callbacks(&x, &y, FitConfig::new(3, N), None, &mut cbs)
        .expect("test fixture: fit");
    let mut without = build_model();
    without
        .fit_with_callbacks(&x, &y, FitConfig::new(3, N), None, &mut [])
        .expect("test fixture: fit");
    assert_eq!(model_bits(&with), model_bits(&without));
    let s = swa_of(&cbs[0]);
    assert_eq!(s.n_averaged(), 0);
    assert!(s.raw_state_dict().is_none());
}

// 4. Err 経路では上書きしない。
#[test]
fn err_path_does_not_overwrite() {
    let (x, y) = gen_data();
    let mut model = build_model();
    let before = model_bits(&model);
    let mut cbs = [swa_cb(0)];
    // バッチ 32 > サンプル 16 かつ drop_last で全バッチが落ち、count == 0 の Err になる。
    let r = model.fit_with_callbacks(
        &x,
        &y,
        FitConfig::new(2, 32).drop_last(true),
        None,
        &mut cbs,
    );
    assert!(is_invalid_arg(r));
    assert_eq!(model_bits(&model), before);
    let s = swa_of(&cbs[0]);
    assert_eq!(s.n_averaged(), 0);
    assert!(s.raw_state_dict().is_none());
}

// 5. 継続学習: 2 + 2 epoch と 4 epoch 一括が bit 一致する。
#[test]
fn split_fit_matches_single_fit() {
    let (x, y) = gen_data();
    let mut one = build_model();
    let mut cbs_one = [swa_cb(1)];
    one.fit_with_callbacks(&x, &y, FitConfig::new(4, N), None, &mut cbs_one)
        .expect("test fixture: fit");

    let mut two = build_model();
    let mut cbs_two = [swa_cb(1)];
    for _ in 0..2 {
        two.fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut cbs_two)
            .expect("test fixture: fit");
    }

    let (a, b) = (swa_of(&cbs_one[0]), swa_of(&cbs_two[0]));
    assert_eq!(a.n_averaged(), b.n_averaged());
    assert_eq!(
        dict_bits(&a.averaged_state_dict().expect("test fixture")),
        dict_bits(&b.averaged_state_dict().expect("test fixture"))
    );
    assert_eq!(
        dict_bits(&a.raw_state_dict().expect("test fixture")),
        dict_bits(&b.raw_state_dict().expect("test fixture"))
    );
    assert_eq!(model_bits(&one), model_bits(&two));
}

// 5b. 1 回目と 2 回目の間で利用者が重みを変えたら退避を破棄し、現在の重みから継続する。
#[test]
fn user_changed_weights_discard_backup() {
    let (x, y) = gen_data();
    let mut model = build_model();
    let mut cbs = [swa_cb(0)];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut cbs)
        .expect("test fixture: fit");
    let raw1 = swa_of(&cbs[0]).raw_state_dict().expect("test fixture");

    // 利用者が（平均化重みではなく）別の重みへ戻した。
    let other = Sequential::new()
        .add_linear(D_IN, 8, 0x33)
        .expect("test fixture")
        .add_relu()
        .add_linear(8, D_OUT, 0x44)
        .expect("test fixture")
        .state_dict();
    model.load_state_dict(other.clone()).expect("test fixture");

    // 参照: 同じ重みから SWA なしで 2 epoch 学習した結果。
    let mut reference = build_model();
    reference.load_state_dict(other).expect("test fixture");
    reference
        .fit(&x, &y, FitConfig::new(2, N))
        .expect("test fixture: fit");

    model
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut cbs)
        .expect("test fixture: fit");
    let raw2 = swa_of(&cbs[0]).raw_state_dict().expect("test fixture");
    // 破棄後に現在の重みから継続したので raw2 は旧退避ではなく参照の最終重みと一致する。
    assert_eq!(dict_bits(&raw2), model_bits(&reference));
    assert_ne!(dict_bits(&raw2), dict_bits(&raw1));
}

// 6. 別構成のモデルへの使い回し: モデル・callback とも不変。
#[test]
fn reuse_with_other_architecture_is_rejected_without_side_effects() {
    let (x, y) = gen_data();
    let mut model = build_model();
    let mut cbs = [swa_cb(0)];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut cbs)
        .expect("test fixture: fit");
    let n = swa_of(&cbs[0]).n_averaged();

    // 層構成が違う（パラメータ数が違う）。
    let mut other = Sequential::new()
        .add_linear(D_IN, D_OUT, 1)
        .expect("test fixture");
    other
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .expect("test fixture");
    let before = model_bits(&other);
    let r = other.fit_with_callbacks(&x, &y, FitConfig::new(1, N), None, &mut cbs);
    assert!(is_invalid_arg(r));
    assert_eq!(model_bits(&other), before);
    assert_eq!(swa_of(&cbs[0]).n_averaged(), n);

    // 名前集合は同じで shape が違う。
    let mut shaped = build_model_with_hidden(6);
    let before = model_bits(&shaped);
    let r = shaped.fit_with_callbacks(&x, &y, FitConfig::new(1, N), None, &mut cbs);
    assert!(matches!(r, Err(AutodiffError::Shape(_))));
    assert_eq!(model_bits(&shaped), before);
    assert_eq!(swa_of(&cbs[0]).n_averaged(), n);
}

// 7. 既存の挙動が変わらない: 損失・val_loss は SWA なしと bit 一致（生の重みで評価）。
#[test]
fn swa_does_not_disturb_training_or_validation() {
    let (x, y) = gen_data();
    let mut with = build_model();
    let mut cbs = [swa_cb(1)];
    let h_with = with
        .fit_with_callbacks(&x, &y, FitConfig::new(4, 4), Some((&x, &y)), &mut cbs)
        .expect("test fixture: fit");
    let mut without = build_model();
    let h_without = without
        .fit_with_callbacks(&x, &y, FitConfig::new(4, 4), Some((&x, &y)), &mut [])
        .expect("test fixture: fit");
    let b = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(b(&h_with.loss), b(&h_without.loss));
    assert_eq!(b(&h_with.val_loss), b(&h_without.val_loss));
    // 重みは平均化重みへ上書きされ、退避は SWA なしの最終重みと一致する。
    assert_ne!(model_bits(&with), model_bits(&without));
    assert_eq!(
        dict_bits(&swa_of(&cbs[0]).raw_state_dict().expect("test fixture")),
        model_bits(&without)
    );
}

// 8. 構築時の検証と初期状態。
#[test]
fn construction_validates_swa_lr_and_starts_empty() {
    for bad in [f32::NAN, f32::INFINITY, 0.0, -0.1] {
        assert!(is_invalid_arg(SwaCallback::new(0, bad, 0, SwaAnneal::Cos)));
    }
    let s = SwaCallback::new(3, 0.07, 0, SwaAnneal::Linear).expect("test fixture");
    assert_eq!(s.swa_start(), 3);
    assert_eq!(s.swa_lr(), 0.07);
    assert_eq!(s.anneal_epochs(), 0);
    assert_eq!(s.anneal(), SwaAnneal::Linear);
    assert_eq!(s.n_averaged(), 0);
    assert!(s.averaged_state_dict().is_none());
    assert!(s.raw_state_dict().is_none());
}
