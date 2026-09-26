//! イシュー #2171（親 #2131「PyTorch／TF 置き換えの API 網羅」）の
//! facade 側テスト: `compat::Sequential` 経由の手動 step 学習ループと
//! `nn::Linear` 直組みの手動ループが、Adadelta／Adamax／NAdam／RAdam
//! （いずれも `fandhe_ai_autodiff::nn::optim` 直接 import。facade
//! 再エクスポートは未承認のため保留——`crates/facade/src/lib.rs::
//! OptimizerExtHoldDoctestGuard` 参照）で bit 完全一致すること、および
//! 各 optimizer が損失を減少させることを確認する。
//!
//! `crates/facade/tests/compat_sequential_train.rs::
//! sequential_training_loop_matches_manual_loop_bit_exact`
//! （`Sgd`／`AdamW` 版）と同型の構成。**本ファイルは
//! `fandhe_ai_autodiff` を直接 import する契約**であり、`facade::
//! compat::optim_train_loop.rs`（facade 再エクスポートのみを使う契約の
//! 別ファイル）へは混入させない。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::nn::Linear;
use fandhe_ai_autodiff::nn::activation::Relu;
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_autodiff::nn::optim::{
    Adadelta, AdadeltaConfig, Adamax, AdamaxConfig, NAdam, NAdamConfig, RAdam, RAdamConfig,
};
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

/// `nn::Linear` 直組みの手動ループ 1 step。`compat_sequential_train.rs::
/// manual_sgd_step` と同型だが、optimizer が `(param, grad)` タプル列を
/// 受け取るシグネチャ（`Adadelta`／`Adamax`／`NAdam`／`RAdam` 共通）に
/// 合わせて呼び出し方が異なる。呼び出し元がジェネリック閉包で optimizer
/// の `step` を渡す。
fn manual_loop_step<F>(
    l1: &Linear,
    l2: &Linear,
    x_data: &Tensor<f32>,
    y_data: &Tensor<f32>,
    step_fn: &mut F,
) -> Result<(f32, [Tensor<f32>; 4]), AutodiffError>
where
    F: FnMut(&[(&Tensor<f32>, &Tensor<f32>)]) -> Result<Vec<Tensor<f32>>, AutodiffError>,
{
    let tape = fandhe_ai_autodiff::Tape::new_with_ops(Box::new(
        fandhe_ai_backend_cpu::CpuBackendOps::new(),
    ));
    let x = tape.var(x_data);
    let y = tape.var(y_data);

    let l1v = l1.bind(&tape);
    let l2v = l2.bind(&tape);
    let relu = Relu;

    let h1 = l1v.forward(&x)?;
    let a1 = <Relu as fandhe_ai_autodiff::nn::Module>::forward(&relu, &tape, &h1)?;
    let h2 = l2v.forward(&a1)?;
    let loss = MseLoss::new(Reduction::Mean).forward(&h2, &y)?;
    let loss_value = scalar(&loss.to_tensor());

    let grads = tape.backward(&loss)?;
    let l1_weight_grad = grads.get(&l1v.weight)?.unwrap();
    let l1_bias_grad = grads.get(l1v.bias.as_ref().unwrap())?.unwrap();
    let l2_weight_grad = grads.get(&l2v.weight)?.unwrap();
    let l2_bias_grad = grads.get(l2v.bias.as_ref().unwrap())?.unwrap();

    let params_and_grads: Vec<(&Tensor<f32>, &Tensor<f32>)> = vec![
        (l1.weight(), l1_weight_grad),
        (l1.bias().unwrap(), l1_bias_grad),
        (l2.weight(), l2_weight_grad),
        (l2.bias().unwrap(), l2_bias_grad),
    ];
    let updated = step_fn(&params_and_grads)?;
    let updated: [Tensor<f32>; 4] = updated
        .try_into()
        .map_err(|_| AutodiffError::InvalidArgument("test fixture: expected 4".to_string()))?;
    Ok((loss_value, updated))
}

/// `compat::Sequential` 経由の手動 step ループ 1 step。`step_fn` が
/// optimizer の `step` を包む（`manual_loop_step` と同じ理由）。
fn sequential_loop_step<F>(
    model: &mut Sequential,
    x_data: &Tensor<f32>,
    y_data: &Tensor<f32>,
    step_fn: &mut F,
) -> f32
where
    F: FnMut(&[(&Tensor<f32>, &Tensor<f32>)]) -> Result<Vec<Tensor<f32>>, AutodiffError>,
{
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
        let params_and_grads: Vec<(&Tensor<f32>, &Tensor<f32>)> =
            param_refs.into_iter().zip(grad_refs).collect();
        (loss_value, step_fn(&params_and_grads).unwrap())
    };
    model.apply_parameters(updated).unwrap();
    loss_value
}

/// 手動ループと Sequential 経由ループが bit 完全一致することを確認する
/// 共通本体。`new_opt` は optimizer の新しいインスタンスを返す（手動・
/// Sequential 双方で独立したインスタンスを使うため 2 回呼ばれる）。
fn assert_manual_and_sequential_loops_bit_exact<O, F>(steps: usize, mut new_opt: F, mut step: O)
where
    O: FnMut(
        &mut dyn std::any::Any,
        &[(&Tensor<f32>, &Tensor<f32>)],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError>,
    F: FnMut() -> Box<dyn std::any::Any>,
{
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    // 手動ループ。
    let mut l1 = Linear::new(D_IN, D_HIDDEN, true, SEED_L1).unwrap();
    let mut l2 = Linear::new(D_HIDDEN, D_OUT, true, SEED_L2).unwrap();
    let mut manual_opt = new_opt();
    let mut manual_losses = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (loss_value, updated) = manual_loop_step(&l1, &l2, &x_data, &y_data, &mut |pg| {
            step(manual_opt.as_mut(), pg)
        })
        .unwrap();
        let [w1, b1, w2, b2] = updated;
        l1 = Linear::from_parameters(w1, Some(b1)).unwrap();
        l2 = Linear::from_parameters(w2, Some(b2)).unwrap();
        manual_losses.push(loss_value);
    }

    // Sequential 経由ループ（同一シード・同一 optimizer 構成）。
    let mut model = build_model();
    let mut seq_opt = new_opt();
    let mut seq_losses = Vec::with_capacity(steps);
    for _ in 0..steps {
        let loss_value = sequential_loop_step(&mut model, &x_data, &y_data, &mut |pg| {
            step(seq_opt.as_mut(), pg)
        });
        seq_losses.push(loss_value);
    }

    assert_eq!(manual_losses.len(), seq_losses.len());
    for (m, s) in manual_losses.iter().zip(seq_losses.iter()) {
        assert_eq!(m.to_bits(), s.to_bits(), "loss series diverged: {m} != {s}");
    }

    let manual_params = [
        l1.weight(),
        l1.bias().unwrap(),
        l2.weight(),
        l2.bias().unwrap(),
    ];
    let seq_params = model.trainable_parameters();
    assert_eq!(manual_params.len(), seq_params.len());
    for (m, s) in manual_params.iter().zip(seq_params.iter()) {
        let m_data = m.contiguous().as_slice().unwrap().to_vec();
        let s_data = s.contiguous().as_slice().unwrap().to_vec();
        assert_eq!(m_data.len(), s_data.len());
        for (a, b) in m_data.iter().zip(s_data.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

// =====================================================================
// Adadelta
// =====================================================================

#[test]
fn adadelta_sequential_training_loop_matches_manual_loop_bit_exact() {
    assert_manual_and_sequential_loops_bit_exact(
        10,
        || Box::new(Adadelta::new(AdadeltaConfig::default()).unwrap()) as Box<dyn std::any::Any>,
        |opt, pg| opt.downcast_mut::<Adadelta>().unwrap().step(pg),
    );
}

#[test]
fn adadelta_manual_loop_is_deterministic() {
    fn run() -> Vec<f32> {
        let (x_data, y_data) = gen_regression_data(SEED_DATA);
        let mut l1 = Linear::new(D_IN, D_HIDDEN, true, SEED_L1).unwrap();
        let mut l2 = Linear::new(D_HIDDEN, D_OUT, true, SEED_L2).unwrap();
        let mut opt = Adadelta::new(AdadeltaConfig::default()).unwrap();
        let mut losses = Vec::new();
        for _ in 0..5 {
            let (loss_value, updated) =
                manual_loop_step(&l1, &l2, &x_data, &y_data, &mut |pg| opt.step(pg)).unwrap();
            let [w1, b1, w2, b2] = updated;
            l1 = Linear::from_parameters(w1, Some(b1)).unwrap();
            l2 = Linear::from_parameters(w2, Some(b2)).unwrap();
            losses.push(loss_value);
        }
        losses
    }
    assert_eq!(run(), run());
}

#[test]
fn sequential_adadelta_training_loop_reduces_loss() {
    const STEPS: usize = 50;
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    let mut opt = Adadelta::new(AdadeltaConfig::default()).unwrap();
    let mut losses = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        losses.push(sequential_loop_step(
            &mut model,
            &x_data,
            &y_data,
            &mut |pg| opt.step(pg),
        ));
    }
    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "Adadelta loop did not reduce loss: initial={initial} final={final_loss}"
    );
}

// =====================================================================
// Adamax
// =====================================================================

#[test]
fn adamax_sequential_training_loop_matches_manual_loop_bit_exact() {
    assert_manual_and_sequential_loops_bit_exact(
        10,
        || Box::new(Adamax::new(AdamaxConfig::default()).unwrap()) as Box<dyn std::any::Any>,
        |opt, pg| opt.downcast_mut::<Adamax>().unwrap().step(pg),
    );
}

#[test]
fn adamax_manual_loop_is_deterministic() {
    fn run() -> Vec<f32> {
        let (x_data, y_data) = gen_regression_data(SEED_DATA);
        let mut l1 = Linear::new(D_IN, D_HIDDEN, true, SEED_L1).unwrap();
        let mut l2 = Linear::new(D_HIDDEN, D_OUT, true, SEED_L2).unwrap();
        let mut opt = Adamax::new(AdamaxConfig::default()).unwrap();
        let mut losses = Vec::new();
        for _ in 0..5 {
            let (loss_value, updated) =
                manual_loop_step(&l1, &l2, &x_data, &y_data, &mut |pg| opt.step(pg)).unwrap();
            let [w1, b1, w2, b2] = updated;
            l1 = Linear::from_parameters(w1, Some(b1)).unwrap();
            l2 = Linear::from_parameters(w2, Some(b2)).unwrap();
            losses.push(loss_value);
        }
        losses
    }
    assert_eq!(run(), run());
}

#[test]
fn sequential_adamax_training_loop_reduces_loss() {
    const STEPS: usize = 50;
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    let mut opt = Adamax::new(AdamaxConfig::default()).unwrap();
    let mut losses = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        losses.push(sequential_loop_step(
            &mut model,
            &x_data,
            &y_data,
            &mut |pg| opt.step(pg),
        ));
    }
    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "Adamax loop did not reduce loss: initial={initial} final={final_loss}"
    );
}

// =====================================================================
// NAdam
// =====================================================================

#[test]
fn nadam_sequential_training_loop_matches_manual_loop_bit_exact() {
    assert_manual_and_sequential_loops_bit_exact(
        10,
        || Box::new(NAdam::new(NAdamConfig::default()).unwrap()) as Box<dyn std::any::Any>,
        |opt, pg| opt.downcast_mut::<NAdam>().unwrap().step(pg),
    );
}

#[test]
fn nadam_manual_loop_is_deterministic() {
    fn run() -> Vec<f32> {
        let (x_data, y_data) = gen_regression_data(SEED_DATA);
        let mut l1 = Linear::new(D_IN, D_HIDDEN, true, SEED_L1).unwrap();
        let mut l2 = Linear::new(D_HIDDEN, D_OUT, true, SEED_L2).unwrap();
        let mut opt = NAdam::new(NAdamConfig::default()).unwrap();
        let mut losses = Vec::new();
        for _ in 0..5 {
            let (loss_value, updated) =
                manual_loop_step(&l1, &l2, &x_data, &y_data, &mut |pg| opt.step(pg)).unwrap();
            let [w1, b1, w2, b2] = updated;
            l1 = Linear::from_parameters(w1, Some(b1)).unwrap();
            l2 = Linear::from_parameters(w2, Some(b2)).unwrap();
            losses.push(loss_value);
        }
        losses
    }
    assert_eq!(run(), run());
}

#[test]
fn sequential_nadam_training_loop_reduces_loss() {
    const STEPS: usize = 50;
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    let mut opt = NAdam::new(NAdamConfig::default()).unwrap();
    let mut losses = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        losses.push(sequential_loop_step(
            &mut model,
            &x_data,
            &y_data,
            &mut |pg| opt.step(pg),
        ));
    }
    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "NAdam loop did not reduce loss: initial={initial} final={final_loss}"
    );
}

// =====================================================================
// RAdam
// =====================================================================

#[test]
fn radam_sequential_training_loop_matches_manual_loop_bit_exact() {
    assert_manual_and_sequential_loops_bit_exact(
        10,
        || Box::new(RAdam::new(RAdamConfig::default()).unwrap()) as Box<dyn std::any::Any>,
        |opt, pg| opt.downcast_mut::<RAdam>().unwrap().step(pg),
    );
}

#[test]
fn radam_manual_loop_is_deterministic() {
    fn run() -> Vec<f32> {
        let (x_data, y_data) = gen_regression_data(SEED_DATA);
        let mut l1 = Linear::new(D_IN, D_HIDDEN, true, SEED_L1).unwrap();
        let mut l2 = Linear::new(D_HIDDEN, D_OUT, true, SEED_L2).unwrap();
        let mut opt = RAdam::new(RAdamConfig::default()).unwrap();
        let mut losses = Vec::new();
        for _ in 0..5 {
            let (loss_value, updated) =
                manual_loop_step(&l1, &l2, &x_data, &y_data, &mut |pg| opt.step(pg)).unwrap();
            let [w1, b1, w2, b2] = updated;
            l1 = Linear::from_parameters(w1, Some(b1)).unwrap();
            l2 = Linear::from_parameters(w2, Some(b2)).unwrap();
            losses.push(loss_value);
        }
        losses
    }
    assert_eq!(run(), run());
}

#[test]
fn sequential_radam_training_loop_reduces_loss() {
    const STEPS: usize = 50;
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    // RAdam の既定 `lr=1e-3` では 50 step では収束判定に届かないため
    // `nn_optim_radam.rs::mlp_converges_with_radam` と同じ経験的な
    // `lr` を使う。
    let cfg = RAdamConfig {
        lr: 0.01,
        ..RAdamConfig::default()
    };
    let mut opt = RAdam::new(cfg).unwrap();
    let mut losses = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        losses.push(sequential_loop_step(
            &mut model,
            &x_data,
            &y_data,
            &mut |pg| opt.step(pg),
        ));
    }
    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "RAdam loop did not reduce loss: initial={initial} final={final_loss}"
    );
}
