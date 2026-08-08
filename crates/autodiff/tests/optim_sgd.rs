//! #193（親 #192「optimizer（SGD・AdamW）・gradient clipping の実装」）:
//! `autodiff::optim::Sgd` の受け入れテスト。
//!
//! **受け入れ条件**: PyTorch `torch.optim.SGD` と同一系列の更新値一致
//! テストが green（実装計画 §1）。
//!
//! **fixture 方式**: 実装環境に PyTorch が無いため（実装計画 §3.5 の
//! 調査結果）、`torch.optim.SGD` の更新則（PyTorch ドキュメント
//! 「Algorithm」節）を f64 で再実装した参照実装（
//! `tests/testdata/gen_sgd_expected.py`）の出力を埋め込む。同スクリプトは
//! `--with-torch` で `torch.optim.SGD` 実系列との突合も行える作りにして
//! あり、PyTorch 導入済み環境での再検証手段を残す。実機 torch との
//! fixture 突合は本 PR ではまだ未実施（out-of-scope-tracking.md に
//! 従い、必要ならユーザーへ別 Issue 化を提案する）。
//!
//! **数値判定の規律**: 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差
//! 1e-5 未満」（coding-rust.md）をそのまま使う。新しい許容誤差は新設・
//! 緩和しない（f64 参照 vs f32 実装の差は fixture 側で ~1e-7 オーダーに
//! 収まる。`gen_sgd_expected.py` 参照）。

mod common;

use autodiff::Tape;
use autodiff::nn::Linear;
use autodiff::nn::activation::Relu;
use autodiff::optim::{Sgd, SgdConfig};
use tensor_core::Tensor;

use bench_harness::rng::Xorshift64Star;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// coding-rust.md の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差
/// 1e-5 未満）。バックエンド間数値一致テストと同じ基準をそのまま流用
/// する（本ファイルで新設・緩和しない）。
fn assert_close(actual: f32, expected: f32, context: &str) {
    let abs_err = (actual - expected).abs();
    let rel_err = abs_err / expected.abs().max(f32::EPSILON);
    assert!(
        rel_err < 1e-3 || abs_err < 1e-5,
        "{context}: actual={actual} expected={expected} abs_err={abs_err} rel_err={rel_err}"
    );
}

// =====================================================================
// PyTorch 一致 fixture テスト（`tests/testdata/gen_sgd_expected.py` の
// 出力を転記。P0=[1.0, -0.5, 2.0]、GRADS は同スクリプト参照）。
// =====================================================================

const P0: [f32; 3] = [1.0, -0.5, 2.0];
const GRADS: [[f32; 3]; 5] = [
    [0.5, -0.2, 0.1],
    [0.25, -0.1, 0.05],
    [0.125, 0.3, -0.4],
    [-0.05, 0.2, 0.15],
    [0.4, -0.3, 0.1],
];

/// `SgdConfig` を構築し、`P0`/`GRADS` に対して 5 step 実行して各 step
/// 後のパラメータ列を返すヘルパー。
fn run_fixture(config: SgdConfig) -> Vec<[f32; 3]> {
    let mut sgd = Sgd::new(config).expect("test fixture: config は妥当な値のみを渡す");
    let mut p = tensor(P0.to_vec(), &[3]);
    let mut out = Vec::with_capacity(GRADS.len());
    for g in GRADS.iter() {
        let grad = tensor(g.to_vec(), &[3]);
        let updated = sgd
            .step(&[&p], &[&grad])
            .expect("test fixture: shape 不整合は起きない");
        p = updated.into_iter().next().unwrap();
        let data = p.contiguous();
        let slice = data.as_slice().unwrap();
        out.push([slice[0], slice[1], slice[2]]);
    }
    out
}

fn assert_fixture_matches(name: &str, config: SgdConfig, expected: [[f32; 3]; 5]) {
    let actual = run_fixture(config);
    for (t, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        for j in 0..3 {
            assert_close(a[j], e[j], &format!("{name} step={t} idx={j}"));
        }
    }
}

#[test]
fn pytorch_parity_vanilla() {
    assert_fixture_matches(
        "vanilla",
        SgdConfig::new(0.1),
        [
            [0.95, -0.48, 1.99],
            [0.925, -0.47, 1.985],
            [0.9125, -0.5, 2.025],
            [0.9175, -0.52, 2.01],
            [0.8775, -0.49, 2.0],
        ],
    );
}

#[test]
fn pytorch_parity_momentum_0_9() {
    assert_fixture_matches(
        "momentum_0_9",
        SgdConfig::new(0.1).with_momentum(0.9),
        [
            [0.95, -0.48, 1.99],
            [0.88, -0.452, 1.976],
            [0.8045, -0.4568, 2.0034],
            [0.74155, -0.48112, 2.01306],
            [0.644895, -0.473008, 2.011754],
        ],
    );
}

#[test]
fn pytorch_parity_momentum_dampening() {
    assert_fixture_matches(
        "momentum_dampening",
        SgdConfig::new(0.1).with_momentum(0.9).with_dampening(0.5),
        [
            [0.95, -0.48, 1.99],
            [0.8925, -0.457, 1.9785],
            [0.8345, -0.4513, 1.98815],
            [0.7848, -0.45617, 1.989335],
            [0.72007, -0.445553, 1.9854015],
        ],
    );
}

#[test]
fn pytorch_parity_momentum_weight_decay() {
    assert_fixture_matches(
        "momentum_weight_decay",
        SgdConfig::new(0.1)
            .with_momentum(0.9)
            .with_weight_decay(0.01),
        [
            [0.949, -0.4795, 1.988],
            [0.877151, -0.4505705, 1.970212],
            [0.7991097, -0.4540834, 1.9922326],
            [0.7330735, -0.4767909, 1.9950589],
            [0.6329078, -0.4667509, 1.9856075],
        ],
    );
}

#[test]
fn pytorch_parity_nesterov() {
    assert_fixture_matches(
        "nesterov",
        SgdConfig::new(0.1).with_momentum(0.9).with_nesterov(true),
        [
            [0.905, -0.462, 1.981],
            [0.817, -0.4268, 1.9634],
            [0.73655, -0.46112, 2.02806],
            [0.684895, -0.503008, 2.021754],
            [0.5579055, -0.4657072, 2.0105786],
        ],
    );
}

// =====================================================================
// momentum バッファの逐次更新値（実装計画 §5 記載の手計算列）。
// p0=1.0, lr=0.1, μ=0.9, g=[0.5,0.25,0.125] → p=[0.95, 0.88, 0.8045]
// =====================================================================

#[test]
fn momentum_buffer_recursive_sequence_matches_hand_computed_values() {
    let mut sgd = Sgd::new(SgdConfig::new(0.1).with_momentum(0.9)).unwrap();
    let mut p = tensor(vec![1.0], &[1]);
    let grads = [0.5f32, 0.25, 0.125];
    let expected = [0.95f32, 0.88, 0.8045];

    for (g, exp) in grads.iter().zip(expected.iter()) {
        let grad = tensor(vec![*g], &[1]);
        let out = sgd.step(&[&p], &[&grad]).unwrap();
        let got = out[0].get(&[0]).unwrap();
        assert_close(got, *exp, "momentum buffer recursive sequence");
        p = out.into_iter().next().unwrap();
    }
}

// =====================================================================
// エラー系
// =====================================================================

#[test]
fn config_rejects_negative_lr() {
    let err = Sgd::new(SgdConfig::new(-0.1)).unwrap_err();
    assert!(matches!(err, autodiff::AutodiffError::InvalidArgument(_)));
}

#[test]
fn config_rejects_negative_momentum() {
    let err = Sgd::new(SgdConfig::new(0.1).with_momentum(-0.5)).unwrap_err();
    assert!(matches!(err, autodiff::AutodiffError::InvalidArgument(_)));
}

#[test]
fn config_rejects_negative_weight_decay() {
    let err = Sgd::new(SgdConfig::new(0.1).with_weight_decay(-0.01)).unwrap_err();
    assert!(matches!(err, autodiff::AutodiffError::InvalidArgument(_)));
}

#[test]
fn config_rejects_nesterov_without_momentum() {
    let err = Sgd::new(SgdConfig::new(0.1).with_nesterov(true)).unwrap_err();
    assert!(matches!(err, autodiff::AutodiffError::InvalidArgument(_)));
}

#[test]
fn config_rejects_nesterov_with_nonzero_dampening() {
    let err = Sgd::new(
        SgdConfig::new(0.1)
            .with_momentum(0.9)
            .with_dampening(0.1)
            .with_nesterov(true),
    )
    .unwrap_err();
    assert!(matches!(err, autodiff::AutodiffError::InvalidArgument(_)));
}

#[test]
fn step_rejects_params_grads_count_mismatch() {
    let mut sgd = Sgd::new(SgdConfig::new(0.1)).unwrap();
    let p = tensor(vec![1.0], &[1]);
    let g1 = tensor(vec![0.1], &[1]);
    let g2 = tensor(vec![0.2], &[1]);
    let err = sgd.step(&[&p], &[&g1, &g2]).unwrap_err();
    assert!(matches!(err, autodiff::AutodiffError::InvalidArgument(_)));
}

#[test]
fn step_rejects_param_grad_shape_mismatch() {
    let mut sgd = Sgd::new(SgdConfig::new(0.1)).unwrap();
    let p = tensor(vec![1.0, 2.0], &[2]);
    let g = tensor(vec![0.1], &[1]);
    let err = sgd.step(&[&p], &[&g]).unwrap_err();
    assert!(matches!(err, autodiff::AutodiffError::Shape(_)));
}

#[test]
fn step_rejects_shape_change_between_steps_with_momentum() {
    let mut sgd = Sgd::new(SgdConfig::new(0.1).with_momentum(0.9)).unwrap();
    let p1 = tensor(vec![1.0], &[1]);
    let g1 = tensor(vec![0.1], &[1]);
    sgd.step(&[&p1], &[&g1]).unwrap();

    let p2 = tensor(vec![1.0, 2.0], &[2]);
    let g2 = tensor(vec![0.1, 0.1], &[2]);
    let err = sgd.step(&[&p2], &[&g2]).unwrap_err();
    assert!(matches!(err, autodiff::AutodiffError::InvalidArgument(_)));
}

#[test]
fn step_rejects_param_count_change_between_steps_with_momentum() {
    let mut sgd = Sgd::new(SgdConfig::new(0.1).with_momentum(0.9)).unwrap();
    let p1 = tensor(vec![1.0], &[1]);
    let g1 = tensor(vec![0.1], &[1]);
    sgd.step(&[&p1], &[&g1]).unwrap();

    let p2 = tensor(vec![1.0], &[1]);
    let g2 = tensor(vec![0.1], &[1]);
    let err = sgd.step(&[&p1, &p2], &[&g1, &g2]).unwrap_err();
    assert!(matches!(err, autodiff::AutodiffError::InvalidArgument(_)));
}

// =====================================================================
// 端値: 空テンソル・非 contiguous view 入力
// =====================================================================

#[test]
fn step_handles_empty_tensor() {
    let mut sgd = Sgd::new(SgdConfig::new(0.1)).unwrap();
    let p = tensor(vec![], &[0]);
    let g = tensor(vec![], &[0]);
    let out = sgd.step(&[&p], &[&g]).unwrap();
    assert_eq!(out[0].numel(), 0);
}

#[test]
fn step_handles_non_contiguous_view_input() {
    let mut sgd = Sgd::new(SgdConfig::new(0.1)).unwrap();
    let p = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
        .transpose_2d()
        .unwrap();
    // grad は要素ごとに異なる値にする: 全要素同値だと param↔grad の
    // 要素対応がずれても出力が偶然一致してしまい、テストが対応関係の
    // 正しさを検証できない（Review 指摘: #193 momentum PR）。
    let g = tensor(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]);
    assert!(!p.is_contiguous());
    let out = sgd.step(&[&p], &[&g]).unwrap();
    // p.contiguous() の行優先データは transpose 後の並び [1,3,2,4]。
    // g は contiguous のため元の並び [0.1,0.2,0.3,0.4] のまま。
    assert_close(
        out[0].get(&[0, 0]).unwrap(),
        1.0 - 0.1 * 0.1,
        "transpose[0,0]",
    );
    assert_close(
        out[0].get(&[0, 1]).unwrap(),
        3.0 - 0.1 * 0.2,
        "transpose[0,1]",
    );
    assert_close(
        out[0].get(&[1, 0]).unwrap(),
        2.0 - 0.1 * 0.3,
        "transpose[1,0]",
    );
    assert_close(
        out[0].get(&[1, 1]).unwrap(),
        4.0 - 0.1 * 0.4,
        "transpose[1,1]",
    );
}

#[test]
fn step_handles_non_contiguous_grad_input() {
    // param 側は contiguous のまま、grad 側のみ非 contiguous
    // （transpose view）にする。上のテストが param 側のみを非
    // contiguous にしていたため、grad 側の正規化経路は未検証だった
    // （Review 指摘: #193 momentum PR）。
    let mut sgd = Sgd::new(SgdConfig::new(0.1)).unwrap();
    let p = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let g = tensor(vec![0.1, 0.2, 0.3, 0.4], &[2, 2])
        .transpose_2d()
        .unwrap();
    assert!(!g.is_contiguous());
    let out = sgd.step(&[&p], &[&g]).unwrap();
    // g.contiguous() の行優先データは transpose 後の並び
    // [0.1,0.3,0.2,0.4]。
    assert_close(
        out[0].get(&[0, 0]).unwrap(),
        1.0 - 0.1 * 0.1,
        "grad_transpose[0,0]",
    );
    assert_close(
        out[0].get(&[0, 1]).unwrap(),
        2.0 - 0.1 * 0.3,
        "grad_transpose[0,1]",
    );
    assert_close(
        out[0].get(&[1, 0]).unwrap(),
        3.0 - 0.1 * 0.2,
        "grad_transpose[1,0]",
    );
    assert_close(
        out[0].get(&[1, 1]).unwrap(),
        4.0 - 0.1 * 0.4,
        "grad_transpose[1,1]",
    );
}

// =====================================================================
// E2E: 2 層 MLP を Sgd（momentum=0.9）で学習し、決定的シード下で
// loss 減少・2 回実行の to_bits 完全一致を確認する
// （`nn_train_convergence.rs` のパターンを踏襲。既存テストファイルは
// 変更しない）。
// =====================================================================

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;
const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

/// `Sgd`（momentum=0.9）を optimizer 本体として使う E2E 学習ループ。
/// `nn_train_convergence.rs::run_regression_training` と同形状の MLP を
/// 使うが、パラメータ更新をテストローカル `sgd_step` ではなく本イシュー
/// の `autodiff::optim::Sgd` へ委ねる点が異なる。
fn run_regression_training_with_sgd(steps: usize, lr: f32) -> Vec<(f32, u32)> {
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let relu = Relu;

    let mut l1 =
        Linear::new(D_IN, D_HIDDEN, true, SEED_L1).expect("test fixture: shape は事前に妥当");
    let mut l2 =
        Linear::new(D_HIDDEN, D_OUT, true, SEED_L2).expect("test fixture: shape は事前に妥当");
    let mut sgd = Sgd::new(SgdConfig::new(lr).with_momentum(0.9))
        .expect("test fixture: config は妥当な値のみを渡す");

    let mut log = Vec::with_capacity(steps);

    for _ in 0..steps {
        let tape = Tape::new_with_ops(common::naive_ops());
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

        let l1_bias = l1.bias().expect("test fixture: bias=true で構築");
        let l2_bias = l2.bias().expect("test fixture: bias=true で構築");
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

        // `Sgd::step` の「位置対応契約」（`sgd.rs` doc 参照）に従い、
        // 4 パラメータ（l1.weight/l1.bias/l2.weight/l2.bias）を毎 step
        // 同じ順序で渡す。
        let params = [l1.weight(), l1_bias, l2.weight(), l2_bias];
        let grads_slice = [l1_weight_grad, l1_bias_grad, l2_weight_grad, l2_bias_grad];
        let updated = sgd.step(&params, &grads_slice).unwrap();
        let mut updated = updated.into_iter();
        let new_l1_weight = updated.next().unwrap();
        let new_l1_bias = updated.next().unwrap();
        let new_l2_weight = updated.next().unwrap();
        let new_l2_bias = updated.next().unwrap();

        l1 = Linear::from_parameters(new_l1_weight, Some(new_l1_bias))
            .expect("test fixture: shape は Sgd::step で保存されている");
        l2 = Linear::from_parameters(new_l2_weight, Some(new_l2_bias))
            .expect("test fixture: shape は Sgd::step で保存されている");

        log.push((loss_value, loss_value.to_bits()));
    }

    log
}

/// 受け入れ条件の一部: `Sgd`（momentum=0.9）で 2 層 MLP を学習すると
/// loss が減少すること。
///
/// **収束判定の根拠**: `lr=0.02`・`STEPS=50` でローカル実測した結果、
/// 単調ではないが最終 loss は初期 loss から明確に減少する（momentum
/// ありの SGD は plain SGD ほど単調でないため、単調減少ではなく
/// 「最終 < 初期 * 閾値」で判定する。`nn_train_convergence.rs` の
/// 収束判定と同じ考え方で、新設・既存 tolerance の緩和ではない）。
#[test]
fn mlp_trained_with_sgd_optimizer_converges() {
    const STEPS: usize = 50;
    const LR: f32 = 0.02;

    let log = run_regression_training_with_sgd(STEPS, LR);
    assert_eq!(log.len(), STEPS);

    let initial = log[0].0;
    let final_loss = log[STEPS - 1].0;
    assert!(
        final_loss < 0.5 * initial,
        "収束が不十分: initial={initial} final={final_loss}"
    );
}

/// 受け入れ条件「再現可能」の直接検証: 同一シードで学習ループを独立に
/// 2 回実行し、各 step の loss 系列がビット完全一致すること
/// （`nn_train_convergence.rs::regression_mlp_reproducible_with_same_seed`
/// と同一方式。`Sgd` の momentum バッファに非決定的な走査
/// （HashMap 等）が混入していないことを固定する）。
#[test]
fn mlp_trained_with_sgd_optimizer_is_reproducible_with_same_seed() {
    const STEPS: usize = 20;
    const LR: f32 = 0.02;

    let run1 = run_regression_training_with_sgd(STEPS, LR);
    let run2 = run_regression_training_with_sgd(STEPS, LR);

    assert_eq!(
        run1.iter().map(|(_, bits)| *bits).collect::<Vec<_>>(),
        run2.iter().map(|(_, bits)| *bits).collect::<Vec<_>>(),
        "同一シード・同一ステップ数の 2 回の学習実行で loss 系列がビット一致しない\
         （momentum バッファの非決定的走査混入の疑い）"
    );
}
