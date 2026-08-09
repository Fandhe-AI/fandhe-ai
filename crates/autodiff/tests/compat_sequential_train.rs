//! イシュー #294（feat(compat): Sequential の学習 API〈optimizer 接続〉
//! 対応）の受け入れ条件 2「optimizer（`crate::optim::Sgd`・
//! `crate::nn::optim::AdamW`）と組み合わせた学習ループの動作を確認する
//! テスト」を公開 API（`autodiff::compat`）経由で検証する統合テスト。
//!
//! `tests/nn_train_convergence.rs` は `nn::Linear` を直接使う手動学習
//! ループ（optimizer 未実装〈#192 着手前〉当時のテストローカル SGD
//! 代替）を固定するのに対し、本ファイルは **`compat::Sequential::bind`
//! が返す `SequentialVars` 経由の学習ループ**と、実装済みの
//! `crate::optim::Sgd`/`crate::nn::optim::AdamW` を接続する点が異なる。
//!
//! **決定的シード**: 重み初期化（`Sequential::add_linear` の `seed`
//! 引数）・データ生成（`bench_harness::rng::Xorshift64Star`）の双方を
//! 固定シードで駆動する（`coding-rust.md`「学習系回帰テストには決定的
//! シード設定ユーティリティを使う」）。
//!
//! **数値判定の規律**: 収束判定は `nn_train_convergence.rs` と同じ
//! 判定様式（最終 loss が初期 loss から十分減少すること）を踏襲し、
//! 新規の許容誤差は設けない。手動ループとのパリティ確認はビット一致
//! （tolerance 不使用。`coding-rust.md`「許容誤差を単独で緩和しない」）。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。

mod common;

use autodiff::compat::Sequential;
use autodiff::nn::Linear;
use autodiff::nn::activation::Relu;
use autodiff::nn::loss::{MseLoss, Reduction};
use autodiff::nn::optim::{AdamW, AdamWConfig};
use autodiff::optim::{Sgd, SgdConfig};
use autodiff::{AutodiffError, Tape};
use bench_harness::rng::Xorshift64Star;
use tensor_core::Tensor;

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

/// `x`（`[BATCH, D_IN]`）・`y`（`[BATCH, D_OUT]`）を Xorshift64Star から
/// 生成する（`nn_train_convergence.rs::gen_regression_data` と同一生成順）。
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

// =====================================================================
// 受け入れ条件 2-1: Sgd 学習ループ収束
// =====================================================================

/// `Sequential::bind` → `SequentialVars::forward` → `Tape::backward` →
/// `SequentialVars::trainable_grads` → `Sgd::step` →
/// `Sequential::apply_parameters` の 1 ステップを `steps` 回繰り返し、
/// 各 step の loss を返す。
///
/// **ブロックスコープでの借用解放**: `SequentialVars`（`bound`）は
/// `&model`/`&tape` の両方を借用するため、`apply_parameters`（`&mut
/// model`）を呼ぶ前に必ずスコープを抜けて借用を解放する（`compat/
/// sequential.rs::Sequential::bind` doc 参照）。
fn train_with_sgd(model: &mut Sequential, steps: usize, lr: f32) -> Vec<f32> {
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let mut sgd = Sgd::new(SgdConfig::new(lr)).unwrap();
    let mut log = Vec::with_capacity(steps);

    for _ in 0..steps {
        let updated = {
            let tape = Tape::new_with_ops(common::naive_ops());
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);

            let pred = bound.forward(&tape, &x).unwrap();
            let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
            log.push(scalar(&loss.to_tensor()));

            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs).unwrap()
        };
        model.apply_parameters(updated).unwrap();
    }

    log
}

/// 受け入れ条件 2 の本体（Sgd）: `compat::Sequential` + `Sgd` の学習
/// ループで loss が減少すること。
///
/// **収束判定の根拠**: `nn_train_convergence.rs::regression_mlp_converges`
/// と同一の同一形状・同一シード・同一データのため、実測は
/// `initial ≈ 0.3743819`・`final(lr=0.05, steps=100) ≈ 0.131` の系列と
/// 整合する（同ファイル docstring 参照）。ここでは新規閾値
/// `final < 0.5 * initial` を据える（緩和ではなく新設の収束判定）。
#[test]
fn sequential_sgd_training_loop_converges() {
    const STEPS: usize = 100;
    const LR: f32 = 0.05;

    let mut model = build_model();
    let log = train_with_sgd(&mut model, STEPS, LR);

    assert_eq!(log.len(), STEPS);
    let initial = log[0];
    let final_loss = *log.last().unwrap();
    assert!(
        final_loss < 0.5 * initial,
        "loss did not converge sufficiently: initial={initial} final={final_loss}"
    );
}

// =====================================================================
// 受け入れ条件 2-2: 手動ループとのビット一致パリティ
// =====================================================================

/// `nn::Linear` を直接使う手動ループ（`nn_train_convergence.rs` と同型）
/// を 1 ステップ実行し、更新後の `(weight, bias)` を層ごとに返す。
fn manual_sgd_step(
    l1: &Linear,
    l2: &Linear,
    x_data: &Tensor<f32>,
    y_data: &Tensor<f32>,
    sgd: &mut Sgd,
) -> Result<(f32, [Tensor<f32>; 4]), AutodiffError> {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(x_data);
    let y = tape.var(y_data);

    let l1v = l1.bind(&tape);
    let l2v = l2.bind(&tape);
    let relu = Relu;

    let h1 = l1v.forward(&x)?;
    let a1 = <Relu as autodiff::nn::Module>::forward(&relu, &tape, &h1)?;
    let h2 = l2v.forward(&a1)?;
    let loss = MseLoss::new(Reduction::Mean).forward(&h2, &y)?;
    let loss_value = scalar(&loss.to_tensor());

    let grads = tape.backward(&loss)?;
    let l1_weight_grad = grads.get(&l1v.weight)?.unwrap();
    let l1_bias_grad = grads.get(l1v.bias.as_ref().unwrap())?.unwrap();
    let l2_weight_grad = grads.get(&l2v.weight)?.unwrap();
    let l2_bias_grad = grads.get(l2v.bias.as_ref().unwrap())?.unwrap();

    let params = [
        l1.weight(),
        l1.bias().unwrap(),
        l2.weight(),
        l2.bias().unwrap(),
    ];
    let grad_refs = [l1_weight_grad, l1_bias_grad, l2_weight_grad, l2_bias_grad];
    let updated = sgd.step(&params, &grad_refs)?;
    let updated: [Tensor<f32>; 4] = updated
        .try_into()
        .map_err(|_| AutodiffError::InvalidArgument("test fixture: expected 4".to_string()))?;
    Ok((loss_value, updated))
}

/// 受け入れ条件 2 の本体（パリティ）: 同一シード・同一 ops・同一更新式
/// で「手動ループ」と「`Sequential` 経由ループ」を並走させ、各 step の
/// loss と最終パラメータがビット一致することを確認する（compat 層の
/// 「薄いラッパー性」の実証。REQ-9）。
#[test]
fn sequential_training_loop_matches_manual_loop_bit_exact() {
    const STEPS: usize = 10;
    const LR: f32 = 0.05;

    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    // 手動ループ。
    let mut l1 = Linear::new(D_IN, D_HIDDEN, true, SEED_L1).unwrap();
    let mut l2 = Linear::new(D_HIDDEN, D_OUT, true, SEED_L2).unwrap();
    let mut manual_sgd = Sgd::new(SgdConfig::new(LR)).unwrap();
    let mut manual_losses = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let (loss_value, updated) =
            manual_sgd_step(&l1, &l2, &x_data, &y_data, &mut manual_sgd).unwrap();
        let [w1, b1, w2, b2] = updated;
        l1 = Linear::from_parameters(w1, Some(b1)).unwrap();
        l2 = Linear::from_parameters(w2, Some(b2)).unwrap();
        manual_losses.push(loss_value);
    }

    // Sequential 経由ループ（同一シード・同一 lr・同一 optimizer 構成）。
    let mut model = build_model();
    let mut seq_sgd = Sgd::new(SgdConfig::new(LR)).unwrap();
    let mut seq_losses = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let updated = {
            let tape = Tape::new_with_ops(common::naive_ops());
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);

            let pred = bound.forward(&tape, &x).unwrap();
            let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
            seq_losses.push(scalar(&loss.to_tensor()));

            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = model.trainable_parameters();
            seq_sgd.step(&param_refs, &grad_refs).unwrap()
        };
        model.apply_parameters(updated).unwrap();
    }

    assert_eq!(manual_losses.len(), seq_losses.len());
    for (m, s) in manual_losses.iter().zip(seq_losses.iter()) {
        assert_eq!(m.to_bits(), s.to_bits(), "loss series diverged: {m} != {s}");
    }

    // 最終パラメータもビット一致すること。
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
// 受け入れ条件 2-3: AdamW 学習ループ
// =====================================================================

/// AdamW（`crate::nn::optim::AdamW`）でも `Sequential` 経由の学習
/// ループが動作し loss が減少すること。`AdamW::step` はタプル列
/// `&[(&Tensor, &Tensor)]` を受け取る（`Sgd::step` の 2 列 API とは
/// シグネチャが異なる）ため、`trainable_parameters()` と
/// `trainable_grads()` の順序契約が両 optimizer で共通に使えることの
/// 実証を兼ねる。
#[test]
fn sequential_adamw_training_loop_reduces_loss() {
    const STEPS: usize = 50;

    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    let mut adamw = AdamW::new(AdamWConfig::default()).unwrap();
    let mut losses = Vec::with_capacity(STEPS);

    for _ in 0..STEPS {
        let updated = {
            let tape = Tape::new_with_ops(common::naive_ops());
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);

            let pred = bound.forward(&tape, &x).unwrap();
            let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
            losses.push(scalar(&loss.to_tensor()));

            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = model.trainable_parameters();
            let params_and_grads: Vec<(&Tensor<f32>, &Tensor<f32>)> =
                param_refs.into_iter().zip(grad_refs).collect();
            adamw.step(&params_and_grads).unwrap()
        };
        model.apply_parameters(updated).unwrap();
    }

    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "AdamW loop did not reduce loss: initial={initial} final={final_loss}"
    );
}

// =====================================================================
// 受け入れ条件 2-4: 勾配欠損時の fail-closed
// =====================================================================

/// loss に寄与しないパラメータを含む構成で `trainable_grads` が
/// `InvalidArgument` を返すこと（黙って除外し optimizer の位置対応
/// 契約を壊さないことの確認）。第 2 層の出力を経由せず第 1 層の出力
/// のみを loss にする（第 2 層の weight/bias は forward グラフから
/// 到達不能になる）ことで未到達勾配を作る。
#[test]
fn trainable_grads_rejects_unreached_parameter() {
    let model = build_model();

    let tape = Tape::new_with_ops(common::naive_ops());
    let bound = model.bind(&tape);
    let x_data = tensor(vec![0.1_f32; BATCH * D_IN], &[BATCH, D_IN]);
    let x = tape.var(&x_data);

    // Sequential::forward ではなく最初の Linear 層のみを使って loss を
    // 組む（第 2 層の LinearVars が backward グラフに登場しない）。
    let first_linear_output = bound.linears()[0].forward(&x).unwrap();
    let loss = first_linear_output.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();

    let err = bound.trainable_grads(&grads).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}
