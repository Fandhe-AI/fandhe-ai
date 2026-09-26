//! optimizer state_dict（イシュー #2174・親 #2131）と facade の
//! safetensors 入出力（[`fandhe_ai::interop::safetensors`]）を組み合わせた
//! 統合テスト。
//!
//! `fandhe_ai_autodiff::nn::optim::OptimizerStateDict` は facade
//! 非公開の内部クレート限定 API（`crates/autodiff/src/nn/optim/
//! state_dict.rs` モジュール冒頭 doc「facade 公開の保留」節）だが、
//! `state_dict()` が返す `HashMap<String, Tensor<f32>>` 自体は facade の
//! safetensors 入出力（純再エクスポート）へそのまま渡せる
//! （`interop_safetensors_roundtrip.rs`・`compat_sequential_train.rs`
//! と同じく、内部の 9 optimizer 型は `fandhe_ai_autodiff` から直接
//! import する）。
//!
//! 1. 9 optimizer 全種で、`state_dict() → save_safetensors_f32_to_bytes
//!    → load_safetensors_f32_from_bytes` の往復が全キー bit 完全一致
//!    すること・同一マップの 2 回保存が同一バイト列になること
//!    （決定性）を固定する。
//! 2. `AdamW` を代表として、`compat::Sequential` の学習ループ（`bind →
//!    forward → backward → trainable_grads → step →
//!    apply_parameters`。`compat_sequential_train.rs` と同一パターン）
//!    を用いた「checkpoint からの再開軌跡が中断しない学習と一致する」
//!    ことを、safetensors ファイル往復（`save_safetensors_f32`／
//!    `load_safetensors_f32`。一時ファイル経由）で固定する（イシュー
//!    #2174 受け入れ基準 4）。9 optimizer 全種の Sequential 結線は
//!    `nn_optim_state_dict.rs::roundtrip_bit_exact_and_continues_
//!    training_identically`（optimizer 直接操作・tensor 単位）が
//!    すでに固定しているため、本ファイルでは safetensors ファイル
//!    往復という別レイヤの契約に絞って代表 1 種で固定する（範囲の
//!    判断は `docs/autodiff-optimizer-state-dict-decision.md` §6
//!    参照）。
//!
//! `Sequential::fit` 自体の再開は facade API の追加（承認事項。
//! `docs/autodiff-optimizer-state-dict-decision.md` §5）が必要なため
//! 対象外とする。`fit` と手動ループが同一であることは既存の
//! `compat_sequential_train.rs::
//! sequential_training_loop_matches_manual_loop_bit_exact` を根拠に
//! 橋渡しする。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。

use std::collections::HashMap;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::safetensors::{
    load_safetensors_f32, load_safetensors_f32_from_bytes, save_safetensors_f32,
    save_safetensors_f32_to_bytes,
};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_autodiff::nn::optim::{
    Adadelta, AdadeltaConfig, Adagrad, AdagradConfig, Adam, AdamConfig, AdamW, AdamWConfig, Adamax,
    AdamaxConfig, Lamb, LambConfig, NAdam, NAdamConfig, OptimizerStateDict, RAdam, RAdamConfig,
    RmsProp, RmsPropConfig,
};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn tensor_bits(tensor: &Tensor<f32>) -> (Vec<usize>, Vec<u32>) {
    let data = tensor
        .as_slice()
        .expect("test fixture: 生成直後の Tensor は contiguous のはず")
        .iter()
        .map(|v| v.to_bits())
        .collect();
    (tensor.shape().to_vec(), data)
}

fn assert_state_dicts_bit_equal(
    a: &HashMap<String, Tensor<f32>>,
    b: &HashMap<String, Tensor<f32>>,
) {
    let ka: std::collections::BTreeSet<&String> = a.keys().collect();
    let kb: std::collections::BTreeSet<&String> = b.keys().collect();
    assert_eq!(
        ka, kb,
        "state_dict のキー集合が safetensors 往復で一致しない"
    );
    for k in ka {
        assert_eq!(
            tensor_bits(&a[k]),
            tensor_bits(&b[k]),
            "state_dict のキー `{k}` の値が safetensors 往復で bit 一致しない"
        );
    }
}

fn temp_dir_for(test_name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "fandhe-ai-optimizer-state-safetensors-{}-{test_name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// =========================================================================
// 1. 9 optimizer 全種: safetensors バイト列往復の bit 完全一致・決定性。
// =========================================================================

fn two_slot_state_dict<T: OptimizerStateDict>(
    opt: &mut T,
    step: impl Fn(
        &mut T,
        &[(&Tensor<f32>, &Tensor<f32>)],
    ) -> Result<Vec<Tensor<f32>>, fandhe_ai_autodiff::AutodiffError>,
) -> HashMap<String, Tensor<f32>> {
    let p0 = t(vec![1.0, -2.0], &[2]);
    let g0 = t(vec![0.1, -0.05], &[2]);
    let p1 = t(vec![0.5, 1.5, -1.0], &[3]);
    let g1 = t(vec![0.02, -0.01, 0.03], &[3]);
    for _ in 0..3 {
        step(opt, &[(&p0, &g0), (&p1, &g1)]).unwrap();
    }
    opt.state_dict().unwrap()
}

macro_rules! safetensors_bit_roundtrip_test {
    ($fn_name:ident, $Ty:ty, $cfg:expr) => {
        #[test]
        fn $fn_name() {
            let mut opt = <$Ty>::new($cfg).unwrap();
            let sd = two_slot_state_dict(&mut opt, |o, x| o.step(x));

            let bytes1 = save_safetensors_f32_to_bytes(&sd, None).unwrap();
            let bytes2 = save_safetensors_f32_to_bytes(&sd, None).unwrap();
            assert_eq!(bytes1, bytes2, "同一マップの保存は決定的でなければならない");

            let loaded = load_safetensors_f32_from_bytes(&bytes1).unwrap();
            assert_state_dicts_bit_equal(&sd, &loaded);
        }
    };
}

safetensors_bit_roundtrip_test!(
    adamw_state_dict_safetensors_bytes_roundtrip,
    AdamW,
    AdamWConfig::default()
);
safetensors_bit_roundtrip_test!(
    adam_state_dict_safetensors_bytes_roundtrip,
    Adam,
    AdamConfig::default()
);
safetensors_bit_roundtrip_test!(
    rmsprop_state_dict_safetensors_bytes_roundtrip,
    RmsProp,
    RmsPropConfig::default()
);
safetensors_bit_roundtrip_test!(
    adagrad_state_dict_safetensors_bytes_roundtrip,
    Adagrad,
    AdagradConfig::default()
);
safetensors_bit_roundtrip_test!(
    lamb_state_dict_safetensors_bytes_roundtrip,
    Lamb,
    LambConfig::default()
);
safetensors_bit_roundtrip_test!(
    adadelta_state_dict_safetensors_bytes_roundtrip,
    Adadelta,
    AdadeltaConfig::default()
);
safetensors_bit_roundtrip_test!(
    adamax_state_dict_safetensors_bytes_roundtrip,
    Adamax,
    AdamaxConfig::default()
);
safetensors_bit_roundtrip_test!(
    nadam_state_dict_safetensors_bytes_roundtrip,
    NAdam,
    NAdamConfig::default()
);
safetensors_bit_roundtrip_test!(
    radam_state_dict_safetensors_bytes_roundtrip,
    RAdam,
    RAdamConfig::default()
);

/// 空 state（初回 `step()` 前）の safetensors 往復も bit 完全一致する
/// ことを固定する（追加ケース。イシュー #2174 実装計画 §5.3）。
#[test]
fn adamw_empty_state_dict_safetensors_bytes_roundtrip() {
    let opt = AdamW::new(AdamWConfig::default()).unwrap();
    let sd = opt.state_dict().unwrap();
    let bytes = save_safetensors_f32_to_bytes(&sd, None).unwrap();
    let loaded = load_safetensors_f32_from_bytes(&bytes).unwrap();
    assert_state_dicts_bit_equal(&sd, &loaded);
}

/// ファイル往復（`save_safetensors_f32`／`load_safetensors_f32`。
/// 一時ファイル経由）の 1 ケース（`interop_safetensors_roundtrip.rs::
/// temp_dir_for` と同型）。
#[test]
fn adamw_state_dict_safetensors_file_roundtrip() {
    let mut opt = AdamW::new(AdamWConfig::default()).unwrap();
    let sd = two_slot_state_dict(&mut opt, |o, x| o.step(x));

    let dir = temp_dir_for("adamw_state_dict_file_roundtrip");
    let path = dir.join("adamw_state.safetensors");
    save_safetensors_f32(&path, &sd).unwrap();
    let loaded = load_safetensors_f32(&path).unwrap();
    assert_state_dicts_bit_equal(&sd, &loaded);
    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================================
// 2. `AdamW` を代表とした、safetensors ファイル往復を経由する checkpoint
//    再開の軌跡一致（`compat::Sequential` 学習ループ）。
// =========================================================================

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 6;
const D_OUT: usize = 4;
const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;
const LR: f32 = 0.01;

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (t(x, &[BATCH, D_IN]), t(y, &[BATCH, D_OUT]))
}

fn build_model(seed_l1: u64, seed_l2: u64) -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, seed_l1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, seed_l2)
        .unwrap()
}

/// `compat_sequential_train.rs::train_with_sgd` と同型の 1 step
/// （`AdamW` 版）。
fn train_step(
    model: &mut Sequential,
    opt: &mut AdamW,
    x_data: &Tensor<f32>,
    y_data: &Tensor<f32>,
) -> f32 {
    let (loss_value, updated) = {
        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let x = tape.var(x_data);
        let y = tape.var(y_data);

        let pred = bound.forward(&tape, &x).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let loss_value = scalar(&loss.to_tensor());

        let grads = tape.backward(&loss).unwrap();
        let grad_refs = bound.trainable_grads(&grads).unwrap();
        let param_refs = model.trainable_parameters();
        // `AdamW::step` は `(param, grad)` のタプル列を受け取る
        // シグネチャ（`Sgd::step` の 2 引数形とは異なる。`nn/optim/
        // mod.rs` doc「内部配置の不統一・シグネチャ差異」節）。
        let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
            param_refs.into_iter().zip(grad_refs.into_iter()).collect();
        (loss_value, opt.step(&pairs).unwrap())
    };
    model.apply_parameters(updated).unwrap();
    loss_value
}

/// イシュー #2174 受け入れ基準 4「checkpoint から再開した学習の軌跡が、
/// 中断しない学習と一致する」を、safetensors ファイル往復（モデル
/// state_dict・optimizer state_dict の双方）で固定する。
#[test]
fn checkpoint_resume_via_safetensors_files_matches_uninterrupted_training() {
    const TOTAL_STEPS: usize = 8;
    const CHECKPOINT_AT: usize = 3;

    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    // 経路 A: 中断しない学習（連続 TOTAL_STEPS step）。
    let mut model_a = build_model(SEED_L1, SEED_L2);
    let mut opt_a = AdamW::new(AdamWConfig {
        lr: LR,
        ..AdamWConfig::default()
    })
    .unwrap();
    let mut losses_a = Vec::with_capacity(TOTAL_STEPS);
    for _ in 0..TOTAL_STEPS {
        losses_a.push(train_step(&mut model_a, &mut opt_a, &x_data, &y_data));
    }

    // 経路 B: CHECKPOINT_AT step 進めてから safetensors ファイルへ
    // checkpoint（モデル state_dict・optimizer state_dict）を書き出し、
    // 別シードで構築したモデル・新規 optimizer へ読み込んでから残りを
    // 進める。
    let mut model_b = build_model(SEED_L1, SEED_L2);
    let mut opt_b = AdamW::new(AdamWConfig {
        lr: LR,
        ..AdamWConfig::default()
    })
    .unwrap();
    let mut losses_b = Vec::with_capacity(TOTAL_STEPS);
    for _ in 0..CHECKPOINT_AT {
        losses_b.push(train_step(&mut model_b, &mut opt_b, &x_data, &y_data));
    }

    let dir = temp_dir_for("checkpoint_resume_via_safetensors_files");
    let model_path = dir.join("model.safetensors");
    let optim_path = dir.join("optimizer.safetensors");
    save_safetensors_f32(&model_path, &model_b.state_dict()).unwrap();
    save_safetensors_f32(&optim_path, &opt_b.state_dict().unwrap()).unwrap();

    // 別シード（意図的に異なる重み）で構築してから checkpoint を読み込む
    // ことで、「checkpoint の値がそのまま反映される」ことを確認する
    // （同じシードのままだと重みが偶然一致していても見分けが付かない）。
    let mut model_resumed = build_model(SEED_L1 ^ 0xFFFF_FFFF, SEED_L2 ^ 0xFFFF_FFFF);
    let loaded_model_state = load_safetensors_f32(&model_path).unwrap();
    model_resumed.load_state_dict(loaded_model_state).unwrap();

    let mut opt_resumed = AdamW::new(AdamWConfig {
        lr: LR,
        ..AdamWConfig::default()
    })
    .unwrap();
    let loaded_optim_state = load_safetensors_f32(&optim_path).unwrap();
    opt_resumed.load_state_dict(loaded_optim_state).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    for _ in CHECKPOINT_AT..TOTAL_STEPS {
        losses_b.push(train_step(
            &mut model_resumed,
            &mut opt_resumed,
            &x_data,
            &y_data,
        ));
    }

    assert_eq!(losses_a.len(), losses_b.len());
    for (step, (a, b)) in losses_a.iter().zip(losses_b.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "step {step} の loss が中断しない学習と再開後の学習で bit 一致しない\
             （a={a}, b={b}）"
        );
    }

    // 最終パラメータも bit 完全一致すること。
    let final_a = model_a.state_dict();
    let final_b = model_resumed.state_dict();
    assert_state_dicts_bit_equal(&final_a, &final_b);
}

/// checkpoint を step 0（optimizer 状態が空）で取った場合も再開軌跡が
/// 一致することを固定する（追加ケース。イシュー #2174 実装計画
/// §5.3）。
#[test]
fn checkpoint_resume_at_step_zero_matches_uninterrupted_training() {
    const TOTAL_STEPS: usize = 4;

    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let mut model_a = build_model(SEED_L1, SEED_L2);
    let mut opt_a = AdamW::new(AdamWConfig {
        lr: LR,
        ..AdamWConfig::default()
    })
    .unwrap();
    let mut losses_a = Vec::with_capacity(TOTAL_STEPS);
    for _ in 0..TOTAL_STEPS {
        losses_a.push(train_step(&mut model_a, &mut opt_a, &x_data, &y_data));
    }

    // 経路 B: 学習前（step 0）に checkpoint を取る。
    let model_b = build_model(SEED_L1, SEED_L2);
    let opt_b = AdamW::new(AdamWConfig {
        lr: LR,
        ..AdamWConfig::default()
    })
    .unwrap();

    let dir = temp_dir_for("checkpoint_resume_at_step_zero");
    let model_path = dir.join("model.safetensors");
    let optim_path = dir.join("optimizer.safetensors");
    save_safetensors_f32(&model_path, &model_b.state_dict()).unwrap();
    save_safetensors_f32(&optim_path, &opt_b.state_dict().unwrap()).unwrap();

    let mut model_resumed = build_model(SEED_L1 ^ 0xFFFF_FFFF, SEED_L2 ^ 0xFFFF_FFFF);
    model_resumed
        .load_state_dict(load_safetensors_f32(&model_path).unwrap())
        .unwrap();
    let mut opt_resumed = AdamW::new(AdamWConfig {
        lr: LR,
        ..AdamWConfig::default()
    })
    .unwrap();
    opt_resumed
        .load_state_dict(load_safetensors_f32(&optim_path).unwrap())
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    let mut losses_b = Vec::with_capacity(TOTAL_STEPS);
    for _ in 0..TOTAL_STEPS {
        losses_b.push(train_step(
            &mut model_resumed,
            &mut opt_resumed,
            &x_data,
            &y_data,
        ));
    }

    assert_eq!(losses_a.len(), losses_b.len());
    for (a, b) in losses_a.iter().zip(losses_b.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
