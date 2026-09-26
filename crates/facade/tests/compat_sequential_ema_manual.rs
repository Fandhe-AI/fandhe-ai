//! イシュー #2179（親 #2131「PyTorch／TF 置き換えの API 網羅」）の
//! `fandhe_ai_autodiff::nn::ExponentialMovingAverage` facade 公開保留
//! （`crates/facade/src/lib.rs::EmaHoldDoctestGuard`）下での受け入れ
//! 条件 R5（MNIST 規模での EMA なし／ありの accuracy 定性確認）の
//! 部分的な裏付け。
//!
//! **本ファイルは `fandhe_ai_autodiff::nn::ExponentialMovingAverage`／
//! `Reduction` を直接 import する契約ファイル**であり
//! （`compat_sequential_lbfgs_manual.rs` と同型の位置づけ）、facade
//! 再エクスポートのみを使う契約のテストファイルへ混入させない。
//! モデル構築・学習ループ本体（`add_linear`／`add_relu`／`bind`／
//! `forward`／`trainable_parameters`／`trainable_grads`／
//! `apply_parameters`／`predict`）・optimizer（`fandhe_ai::optim::
//! AdamW`）はすべて公開 compat API を使う。`compat::Sequential` は
//! [`fandhe_ai_autodiff::nn::Module`] trait を実装しないため
//! `ExponentialMovingAverage::from_module`／`apply`／`restore`（
//! `&mut dyn Module` を要する）は使えない。代わりに公開済みの
//! `named_parameters()`／`state_dict()`／`load_state_dict()` と
//! [`fandhe_ai_autodiff::nn::ExponentialMovingAverage::from_named`]／
//! `update_named`／`shadow_state_dict` を結線する（
//! `docs/autodiff-ema-decision.md` §2「facade 結線」節）。
//! 承認後（`docs/autodiff-ema-decision.md` §4）に `fit(use_ema=true)`
//! 統合を実装する際は、facade 再エクスポート版の別ファイル
//! （`compat_sequential_fit_ema.rs` 想定）を新設する。
//!
//! **数値判定の規律**: AC R5 は「定性的」（厳密な accuracy 改善の
//! アサートはしない。flaky 化を避ける）。両 accuracy（EMA なし／あり）
//! がチャンスレベル（10 クラス分類で 0.1）を明確に上回ること・
//! 退避／復帰後のパラメータが bit 完全一致すること・EMA 評価
//! （shadow への一時差し替え）が学習状態を変化させないことのみを
//! アサートする。実測 accuracy は `eprintln!` で出力し
//! `docs/autodiff-ema-decision.md` §6 へ転記する
//! （`cargo test -p fandhe-ai --test compat_sequential_ema_manual --
//! --nocapture` で確認）。
//!
//! 実機（CUDA／Metal）非依存・ホスト計算のみのため `#[ignore]` 分離は
//! 行わない。

use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{AdamW, AdamWConfig};
use fandhe_ai_autodiff::Reduction;
use fandhe_ai_autodiff::nn::ExponentialMovingAverage;
use fandhe_ai_tensor_core::Tensor;

const D_IN: usize = 784;
const D_HIDDEN: usize = 64;
const D_OUT: usize = 10;
const N_TRAIN: usize = 256;
const N_EVAL: usize = 128;
const EPOCHS: usize = 12;
const EMA_DECAY: f32 = 0.9;

const SEED_L1: u64 = 0xE5A1_0001;
const SEED_L2: u64 = 0xE5A1_0002;
const SEED_DATA_TRAIN: u64 = 0xE5A1_1111;
const SEED_DATA_EVAL: u64 = 0xE5A1_2222;

/// `optim_lbfgs_closure.rs`／`compat_sequential_lbfgs_manual.rs` と同型の
/// 決定的シード生成（facade テストは `bench-harness` を dev-dependency に
/// 持たないためローカルに再実装する）。`[-1, 1)` の一様分布に写像する。
fn xorshift_fill(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.max(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let unit = (bits >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
        out.push((unit * 2.0 - 1.0) as f32);
    }
    out
}

/// 10 クラス分のクラス中心ベクトル（`D_IN` 次元）。学習データ・評価
/// データの双方が同じ中心集合を共有する（クラス `c` ごとに固定
/// シードで xorshift 生成。`gen_classification_data` の呼び出しごとに
/// 再生成すると学習データと評価データで異なる中心を学習・評価する
/// ことになり、正しく学習できていても評価が意味を成さなくなる）。
const CENTER_SEED_BASE: u64 = 0xE5A1_C0DE;

fn class_centers() -> Vec<Vec<f32>> {
    (0..D_OUT)
        .map(|c| xorshift_fill(CENTER_SEED_BASE.wrapping_add(c as u64), D_IN))
        .collect()
}

/// クラス依存の平均＋ノイズで学習可能な合成 MNIST 形状分類データを
/// 生成する（784 次元入力・10 クラス）。`centers`（[`class_centers`]）は
/// 学習・評価で共有し、`seed` はノイズ列のみを差し替える。
fn gen_classification_data(
    n: usize,
    seed: u64,
    centers: &[Vec<f32>],
) -> (Tensor<f32>, Tensor<i32>) {
    let noise = xorshift_fill(seed, n * D_IN);

    let mut x_data = Vec::with_capacity(n * D_IN);
    let mut y_data = Vec::with_capacity(n);
    for i in 0..n {
        let class = i % D_OUT;
        let center = &centers[class];
        for j in 0..D_IN {
            let n_val = noise[i * D_IN + j] * 0.3;
            x_data.push(center[j] + n_val);
        }
        y_data.push(class as i32);
    }

    let x =
        Tensor::new(x_data, &[n, D_IN]).expect("test fixture: shape とデータ長は一致させている");
    let y = Tensor::new(y_data, &[n]).expect("test fixture: shape とデータ長は一致させている");
    (x, y)
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap()
}

/// `logits`（`[n, D_OUT]`）の行ごと argmax と `targets`（`[n]`）の
/// 一致率。
fn accuracy(logits: &Tensor<f32>, targets: &Tensor<i32>) -> f32 {
    let n = targets.shape()[0];
    let mut correct = 0usize;
    for i in 0..n {
        let mut best_class = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for c in 0..D_OUT {
            let v = logits.get(&[i, c]).unwrap();
            if v > best_value {
                best_value = v;
                best_class = c;
            }
        }
        let target = targets.get(&[i]).unwrap();
        if best_class as i32 == target {
            correct += 1;
        }
    }
    correct as f32 / n as f32
}

#[test]
fn ema_shadow_weights_achieve_above_chance_accuracy_without_mutating_training_state() {
    let centers = class_centers();
    let (x_train, y_train) = gen_classification_data(N_TRAIN, SEED_DATA_TRAIN, &centers);
    let (x_eval, y_eval) = gen_classification_data(N_EVAL, SEED_DATA_EVAL, &centers);

    let mut model = build_model();
    let mut opt = AdamW::new(AdamWConfig {
        lr: 5e-3,
        ..AdamWConfig::default()
    })
    .unwrap();
    let mut ema =
        ExponentialMovingAverage::from_named(EMA_DECAY, model.named_parameters()).unwrap();

    for _ in 0..EPOCHS {
        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let x = tape.var(&x_train);
        let pred = bound.forward(&tape, &x).unwrap();
        let loss = pred
            .cross_entropy_loss(&y_train, 1, Reduction::Mean)
            .unwrap();
        let grads = tape.backward(&loss).unwrap();
        let grad_refs = bound.trainable_grads(&grads).unwrap();

        let params_and_grads: Vec<(&Tensor<f32>, &Tensor<f32>)> = model
            .trainable_parameters()
            .into_iter()
            .zip(grad_refs)
            .collect();
        let updated = opt.step(&params_and_grads).unwrap();
        model.apply_parameters(updated).unwrap();

        // 各 step 後に shadow を更新する（AC R2「`fit` の各 step 後に
        // 更新」相当の手動結線。`apply_parameters` 直後の重みで
        // `update_named` を呼ぶ）。
        ema.update_named(model.named_parameters()).unwrap();
    }

    // 素の重みでの評価。
    let raw_logits = model.predict(&x_eval).unwrap();
    let raw_acc = accuracy(&raw_logits, &y_eval);

    // shadow 重みでの評価: state_dict を退避 → shadow へ差し替え →
    // predict → 退避値で復帰。
    let backup = model.state_dict();
    let backup_for_compare = backup.clone();
    model.load_state_dict(ema.shadow_state_dict()).unwrap();
    let ema_logits = model.predict(&x_eval).unwrap();
    let ema_acc = accuracy(&ema_logits, &y_eval);
    model.load_state_dict(backup).unwrap();

    eprintln!(
        "compat_sequential_ema_manual: raw_acc={raw_acc:.4} ema_acc={ema_acc:.4} \
         decay={EMA_DECAY} epochs={EPOCHS} n_train={N_TRAIN} n_eval={N_EVAL}"
    );

    // AC R5: 定性確認（チャンスレベル 1/10 を明確に上回ること）。
    const CHANCE_LEVEL: f32 = 1.0 / D_OUT as f32;
    assert!(
        raw_acc > CHANCE_LEVEL * 2.0,
        "raw_acc={raw_acc} がチャンスレベルの 2 倍を上回っていない"
    );
    assert!(
        ema_acc > CHANCE_LEVEL * 2.0,
        "ema_acc={ema_acc} がチャンスレベルの 2 倍を上回っていない"
    );

    // EMA 評価（退避 → shadow 差し替え → 復帰）は学習状態（重み）を
    // 変化させない: 復帰後のパラメータが退避値と bit 完全一致する。
    let after_restore = model.state_dict();
    for (name, tensor) in &backup_for_compare {
        assert_eq!(
            tensor.as_slice().unwrap(),
            after_restore[name].as_slice().unwrap(),
            "EMA 評価が学習状態（`{name}`）を変化させている"
        );
    }
}
