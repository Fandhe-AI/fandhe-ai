//! TASK-9.1c（親イシュー #90・本イシュー #93）: `nn` モジュール
//! （`Linear`・活性化関数。TASK-9.1a/#91・TASK-9.1b/#92）を経由した
//! 学習収束テスト。
//!
//! `tests/poc_v2_2_parity.rs`（TASK-1.5d・#19）は生の `Var` 演算
//! （`matmul`/`add`/`relu`/`mse_loss`）を直接組んだ 50 step SGD の
//! 決定性テストを持つが、本ファイルは **`autodiff::nn::Linear`/
//! `autodiff::nn::activation::{Relu, Sigmoid}` を経由した学習ループ**
//! が収束・再現することを固定する点が異なる（互換 API 層の受け入れ
//! 検証。REQ-9・受け入れ条件「決定的シードで再現可能な収束テストが
//! green」）。
//!
//! **loss/optimizer をテストローカルに閉じる理由**: 損失関数の集約
//! （CrossEntropy 等・#189）・optimizer（SGD/AdamW・#192）は本イシュー
//! の依存ではなく未実装のため、`Var::mse_loss`（既存 API）と
//! `sgd_step`（`poc_v2_2_parity.rs:439` の先例と同一パターンをテスト
//! ローカルに再掲）で代替する。`Linear` はパラメータの `Tensor<f32>`
//! を不変に保持し、更新後の値は `Linear::from_parameters` で新しい
//! `Linear` を都度構築して差し替える（`nn/linear.rs` の
//! `LinearVars::forward` が勾配取得 API を持たない設計に合わせた運用。
//! optimizer 未実装下でパラメータ更新を成立させる唯一の経路）。
//!
//! **決定的シード**: 重み初期化（`Linear::new` の `seed` 引数）・
//! データ生成（`bench_harness::rng::Xorshift64Star`）の双方を固定シード
//! で駆動する（`coding-rust.md`「学習系回帰テストには決定的シード設定
//! ユーティリティを使う」）。`bench-harness` は dev-dependency 経由の
//! 再利用のみで新規依存追加なし（`Cargo.toml` は変更しない）。
//!
//! **数値判定の規律**: バックエンド間数値一致の統一複合判定・grad
//! check 閾値（`coding-rust.md`）は本ファイルで新設・緩和しない。
//! 収束判定（最終 loss が初期 loss から十分減少すること）は本イシュー
//! で新設する判定だが、既存 tolerance の緩和ではなく、実測値に対し
//! 十分な余裕を持つ値をコメントで根拠付ける。
//!
//! **契約: CI（self-hosted）は `docs/spec`（submodule）を checkout
//! しない**（`poc_v2_2_parity.rs` 冒頭コメントと同じ制約）。本ファイル
//! の全テストは `docs/spec` 配下のいかなるファイルにも依存しない。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。

use autodiff::Tape;
use autodiff::nn::Linear;
use autodiff::nn::activation::{Relu, Sigmoid};
use tensor_core::Tensor;

use bench_harness::rng::Xorshift64Star;

// =====================================================================
// ケース 1: 回帰タスクの学習収束
// =====================================================================

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

// Linear::new の seed 引数（重み初期化）と、入出力データ生成用の
// Xorshift64Star seed を分離する（同一シードの使い回しによる
// 相関を避ける。`nn/linear.rs::Linear::new` の weight/bias 間の
// seed ずらしと同じ考え方）。
const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

/// `Tensor::get` は多次元添字を要求するため、行優先の平坦添字 `i` を
/// `t.shape()` から多次元添字へ復元してから読む（`poc_v2_2_parity.rs`
/// の `flat_get` と同一パターン。`autodiff`/`tensor-core` の公開 API は
/// この変換を提供しないため、テスト側で完結させる）。
fn flat_get(t: &Tensor<f32>, i: usize) -> f32 {
    let shape = t.shape();
    if shape.is_empty() {
        return t.get(&[]).expect("test fixture: スカラー shape [] のはず");
    }
    let mut idx = vec![0usize; shape.len()];
    let mut rem = i;
    for d in (0..shape.len()).rev() {
        idx[d] = rem % shape[d];
        rem /= shape[d];
    }
    t.get(&idx)
        .expect("test fixture: 平坦添字は shape から復元しているため範囲内のはず")
}

/// `poc_v2_2_parity.rs::sgd_step` と同一のフルバッチ SGD 更新（テスト
/// ローカル。optimizer（#192）未実装のため代替する）。
fn sgd_step(param: &Tensor<f32>, grad: &Tensor<f32>, lr: f32) -> Tensor<f32> {
    let shape = param.shape().to_vec();
    let data: Vec<f32> = (0..param.numel())
        .map(|i| flat_get(param, i) - lr * flat_get(grad, i))
        .collect();
    tensor(data, &shape)
}

/// `x`（`[BATCH, D_IN]`）・`y`（`[BATCH, D_OUT]`）を Xorshift64Star から
/// 生成する（`bench_harness::rng::Xorshift64Star::fill_vec` は `[-1,
/// 1)` の一様分布。`poc_v2_2_parity.rs::gen_vec` と同じ生成順 x→y）。
fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

/// `Linear(D_IN→D_HIDDEN)` → `ReLU` → `Linear(D_HIDDEN→D_OUT)` の
/// 2 層 MLP（PoC-v2-2 と同形状。`poc_v2_2_parity.rs` の
/// `D_IN`/`D_HIDDEN`/`D_OUT` 定数と揃え、比較可能性を確保する）を
/// `steps` 回フルバッチ SGD で学習し、各 step の `(loss, loss.to_bits())`
/// を返す。パラメータ更新は `Linear::from_parameters` で都度差し替える
/// （モジュール doc 参照）。
fn run_regression_training(steps: usize, lr: f32) -> Vec<(f32, u32)> {
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let relu = Relu;

    let mut l1 =
        Linear::new(D_IN, D_HIDDEN, true, SEED_L1).expect("test fixture: shape は事前に妥当");
    let mut l2 =
        Linear::new(D_HIDDEN, D_OUT, true, SEED_L2).expect("test fixture: shape は事前に妥当");

    let mut log = Vec::with_capacity(steps);

    for _ in 0..steps {
        let tape = Tape::new();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);

        let l1v = l1.bind(&tape);
        let l2v = l2.bind(&tape);

        let h1 = l1v.forward(&x).unwrap();
        let a1 = relu.forward(&h1);
        let h2 = l2v.forward(&a1).unwrap();
        let loss = h2.mse_loss(&y).unwrap();

        let loss_value = scalar(&loss.to_tensor());
        let grads = tape.backward(&loss).unwrap();

        let l1_weight_grad = grads.get(&l1v.weight).unwrap().unwrap();
        let l1_bias_grad = grads
            .get(l1v.bias.as_ref().expect("test fixture: bias=true で構築"))
            .unwrap()
            .unwrap();
        let l2_weight_grad = grads.get(&l2v.weight).unwrap().unwrap();
        let l2_bias_grad = grads
            .get(l2v.bias.as_ref().expect("test fixture: bias=true で構築"))
            .unwrap()
            .unwrap();

        let new_l1_weight = sgd_step(l1.weight(), l1_weight_grad, lr);
        let new_l1_bias = sgd_step(
            l1.bias().expect("test fixture: bias=true で構築"),
            l1_bias_grad,
            lr,
        );
        let new_l2_weight = sgd_step(l2.weight(), l2_weight_grad, lr);
        let new_l2_bias = sgd_step(
            l2.bias().expect("test fixture: bias=true で構築"),
            l2_bias_grad,
            lr,
        );

        l1 = Linear::from_parameters(new_l1_weight, Some(new_l1_bias))
            .expect("test fixture: shape は sgd_step で保存されている");
        l2 = Linear::from_parameters(new_l2_weight, Some(new_l2_bias))
            .expect("test fixture: shape は sgd_step で保存されている");

        log.push((loss_value, loss_value.to_bits()));
    }

    log
}

/// 受け入れ条件の本体: `Linear`/`Relu` を経由した 2 層 MLP が小規模
/// 回帰データに収束すること（loss が単調非増加・最終 loss が初期 loss
/// から十分減少）を確認する。
///
/// **収束判定の根拠**: `lr=0.05`・`STEPS=100` でローカル実測した結果、
/// `initial=0.3743819` → `final=0.13100824`（約 35% まで減少。
/// `poc_v2_2_parity.rs` の 50 step SGD が同形状・同オーダーの lr で
/// 単調減少することを既に確認済みの先例と整合）。閾値
/// `final < 0.5 * initial` はこの実測に対し余裕を持たせた値であり、
/// 既存 tolerance の緩和ではなく本イシューで新設する収束判定である。
#[test]
fn regression_mlp_converges() {
    const STEPS: usize = 100;
    const LR: f32 = 0.05;

    let log = run_regression_training(STEPS, LR);
    assert_eq!(log.len(), STEPS);

    for w in log.windows(2) {
        assert!(
            w[1].0 <= w[0].0,
            "loss が単調減少していない: {} -> {}",
            w[0].0,
            w[1].0
        );
    }

    let initial = log[0].0;
    let final_loss = log[STEPS - 1].0;
    assert!(
        final_loss < 0.5 * initial,
        "収束が不十分: initial={initial} final={final_loss}"
    );
}

/// 受け入れ条件「再現可能」の直接検証: 同一シードで学習ループを独立に
/// 2 回実行し、各 step の loss 系列がビット完全一致すること
/// （`poc_v2_2_parity.rs::poc_train_repro_determinism` と同一方式）。
/// あわせて異なるシード（データ生成側）では loss 系列が異なることも
/// 固定し、シードが実際に学習結果へ効いていることを確認する。
#[test]
fn regression_mlp_reproducible_with_same_seed() {
    const STEPS: usize = 20;
    const LR: f32 = 0.05;

    let run1 = run_regression_training(STEPS, LR);
    let run2 = run_regression_training(STEPS, LR);

    assert_eq!(
        run1.iter().map(|(_, bits)| *bits).collect::<Vec<_>>(),
        run2.iter().map(|(_, bits)| *bits).collect::<Vec<_>>(),
        "同一シード・同一ステップ数の 2 回の学習実行で loss 系列がビット一致しない\
         （HashMap 等の非決定的走査混入の疑い）"
    );
}

/// 異なるシードでは学習軌跡（loss 系列）が異なることを確認し、
/// `SEED_DATA`/`Linear::new` の `seed` が実際にデータ・初期重みへ
/// 反映されていることを固定する（シードが無視されて常に同じ結果を
/// 返す退行を検出する）。
#[test]
fn regression_mlp_diverges_with_different_seed() {
    const STEPS: usize = 5;
    const LR: f32 = 0.05;

    let (x_data, y_data) = gen_regression_data(SEED_DATA.wrapping_add(1));
    let relu = Relu;
    let mut l1 = Linear::new(D_IN, D_HIDDEN, true, SEED_L1.wrapping_add(1))
        .expect("test fixture: shape は事前に妥当");
    let mut l2 = Linear::new(D_HIDDEN, D_OUT, true, SEED_L2.wrapping_add(1))
        .expect("test fixture: shape は事前に妥当");

    let mut other_log: Vec<(f32, u32)> = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let tape = Tape::new();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let l1v = l1.bind(&tape);
        let l2v = l2.bind(&tape);
        let h1 = l1v.forward(&x).unwrap();
        let a1 = relu.forward(&h1);
        let h2 = l2v.forward(&a1).unwrap();
        let loss = h2.mse_loss(&y).unwrap();
        let loss_value = scalar(&loss.to_tensor());
        let grads = tape.backward(&loss).unwrap();

        let new_l1_weight = sgd_step(l1.weight(), grads.get(&l1v.weight).unwrap().unwrap(), LR);
        let new_l1_bias = sgd_step(
            l1.bias().expect("test fixture: bias=true で構築"),
            grads
                .get(l1v.bias.as_ref().expect("test fixture: bias=true で構築"))
                .unwrap()
                .unwrap(),
            LR,
        );
        let new_l2_weight = sgd_step(l2.weight(), grads.get(&l2v.weight).unwrap().unwrap(), LR);
        let new_l2_bias = sgd_step(
            l2.bias().expect("test fixture: bias=true で構築"),
            grads
                .get(l2v.bias.as_ref().expect("test fixture: bias=true で構築"))
                .unwrap()
                .unwrap(),
            LR,
        );
        l1 = Linear::from_parameters(new_l1_weight, Some(new_l1_bias))
            .expect("test fixture: shape は sgd_step で保存されている");
        l2 = Linear::from_parameters(new_l2_weight, Some(new_l2_bias))
            .expect("test fixture: shape は sgd_step で保存されている");

        other_log.push((loss_value, loss_value.to_bits()));
    }

    let baseline_log = run_regression_training(STEPS, LR);
    assert_ne!(
        baseline_log
            .iter()
            .map(|(_, bits)| *bits)
            .collect::<Vec<_>>(),
        other_log.iter().map(|(_, bits)| *bits).collect::<Vec<_>>(),
        "異なるシードで loss 系列が一致した（シードが学習結果へ反映されていない疑い）"
    );
}

// =====================================================================
// ケース 3: Sigmoid 経路の収束（#92 の成果物に Sigmoid が含まれるため
// 実施する。二値回帰〈XOR〉を Linear→Sigmoid→mse_loss で学習する。
// CrossEntropy は #189 未実装のため使わない）。
// =====================================================================

const XOR_BATCH: usize = 4;
const XOR_IN: usize = 2;
const XOR_HIDDEN: usize = 8;

fn xor_data() -> (Tensor<f32>, Tensor<f32>) {
    // XOR の 4 パターンを固定入力とする（乱数ではなく問題設定そのもの
    // が決定的なため、これは「決定的シード」の対象外の定数データ）。
    let x = tensor(
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        &[XOR_BATCH, XOR_IN],
    );
    let y = tensor(vec![0.0, 1.0, 1.0, 0.0], &[XOR_BATCH, 1]);
    (x, y)
}

/// `Linear(2→8)` → `Sigmoid` → `Linear(8→1)` → `Sigmoid` で XOR を学習し、
/// 最終 loss が初期 loss から十分減少することを確認する（ケース 3）。
///
/// **収束判定の根拠**: `lr=2.0`・`STEPS=2000` でローカル実測した結果、
/// `initial=0.25184217` → `final=0.00080815406`（初期の約 1/300）。
/// sigmoid 飽和域からの脱出に回帰ケースより大きい lr・多くの step を
/// 要するが、実測は閾値 `final < 0.5 * initial` に対し大幅な余裕を
/// 持つ（新設判定・既存 tolerance の緩和ではない）。
#[test]
fn sigmoid_xor_converges() {
    const STEPS: usize = 2000;
    const LR: f32 = 2.0;

    let (x_data, y_data) = xor_data();
    let sigmoid = Sigmoid;

    let mut l1 = Linear::new(XOR_IN, XOR_HIDDEN, true, 0x3333_3333)
        .expect("test fixture: shape は事前に妥当");
    let mut l2 =
        Linear::new(XOR_HIDDEN, 1, true, 0x4444_4444).expect("test fixture: shape は事前に妥当");

    let mut losses = Vec::with_capacity(STEPS);

    for _ in 0..STEPS {
        let tape = Tape::new();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);

        let l1v = l1.bind(&tape);
        let l2v = l2.bind(&tape);

        let h1 = l1v.forward(&x).unwrap();
        let a1 = sigmoid.forward(&h1);
        let h2 = l2v.forward(&a1).unwrap();
        let a2 = sigmoid.forward(&h2);
        let loss = a2.mse_loss(&y).unwrap();

        let loss_value = scalar(&loss.to_tensor());
        let grads = tape.backward(&loss).unwrap();

        let new_l1_weight = sgd_step(l1.weight(), grads.get(&l1v.weight).unwrap().unwrap(), LR);
        let new_l1_bias = sgd_step(
            l1.bias().expect("test fixture: bias=true で構築"),
            grads
                .get(l1v.bias.as_ref().expect("test fixture: bias=true で構築"))
                .unwrap()
                .unwrap(),
            LR,
        );
        let new_l2_weight = sgd_step(l2.weight(), grads.get(&l2v.weight).unwrap().unwrap(), LR);
        let new_l2_bias = sgd_step(
            l2.bias().expect("test fixture: bias=true で構築"),
            grads
                .get(l2v.bias.as_ref().expect("test fixture: bias=true で構築"))
                .unwrap()
                .unwrap(),
            LR,
        );
        l1 = Linear::from_parameters(new_l1_weight, Some(new_l1_bias))
            .expect("test fixture: shape は sgd_step で保存されている");
        l2 = Linear::from_parameters(new_l2_weight, Some(new_l2_bias))
            .expect("test fixture: shape は sgd_step で保存されている");

        losses.push(loss_value);
    }

    let initial = losses[0];
    let final_loss = losses[STEPS - 1];
    assert!(
        final_loss < 0.5 * initial,
        "XOR 学習の収束が不十分: initial={initial} final={final_loss}"
    );
}
