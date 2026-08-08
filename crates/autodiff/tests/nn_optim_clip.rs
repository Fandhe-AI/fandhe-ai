//! #195（親 #192）: gradient clipping（global norm 方式）のノルム計算・
//! clip 動作・適用順序契約（backward → unscale → clip）の受け入れテスト。
//!
//! `autodiff::nn::optim::clip` は `Gradients`/`Var` に依存しない純関数
//! （`clip.rs` doc 参照）のため、ここでは手組みの `Tensor<f32>` を直接
//! 渡してテストする。ミニ学習ステップとの統合（backward 由来の実勾配へ
//! の適用）は `nn_train_convergence.rs` と同型の 2 層 MLP で検証する。

mod common;

use autodiff::Tape;
use autodiff::nn::optim::{clip_grad_norm, global_grad_norm};
use tensor_core::Tensor;

use bench_harness::rng::Xorshift64Star;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

// =====================================================================
// ノルム計算（受け入れ条件「ノルム計算のテスト」）
// =====================================================================

#[test]
fn global_grad_norm_single_tensor() {
    // 3-4-5 の直角三角形。手計算固定値: sqrt(3^2 + 4^2) = 5.0
    let g = tensor(vec![3.0, 4.0], &[2]);
    let norm = global_grad_norm(&[&g]).unwrap();
    assert!((norm - 5.0).abs() < 1e-6, "norm={norm}");
}

#[test]
fn global_grad_norm_multiple_tensors_is_pooled_across_tensors() {
    // PyTorch clip_grad_norm_ の定義: 全パラメータ勾配をひとつの仮想
    // ベクトルとみなした L2 ノルム。[3,4] と [12] を横断すると
    // sqrt(3^2+4^2+12^2) = sqrt(25+144) = sqrt(169) = 13.0
    // （個々のテンソル毎のノルム 5.0・12.0 の単純和 17.0 とは異なる
    // ことを固定し、pooled 方式であることを回帰化する）。
    let g1 = tensor(vec![3.0, 4.0], &[2]);
    let g2 = tensor(vec![12.0], &[1]);
    let norm = global_grad_norm(&[&g1, &g2]).unwrap();
    assert!((norm - 13.0).abs() < 1e-6, "norm={norm}");
}

#[test]
fn global_grad_norm_empty_slice_is_zero() {
    let norm = global_grad_norm(&[]).unwrap();
    assert_eq!(norm, 0.0);
}

#[test]
fn global_grad_norm_rejects_non_finite_result() {
    let g = tensor(vec![f32::NAN, 1.0], &[2]);
    let err = global_grad_norm(&[&g]).unwrap_err();
    assert!(format!("{err}").contains("非有限"), "err={err}");
}

// =====================================================================
// clip 動作
// =====================================================================

#[test]
fn clip_grad_norm_noop_when_within_max_norm() {
    // total_norm = 5.0 <= max_norm = 10.0 → 無変更・scaled == false
    let g = tensor(vec![3.0, 4.0], &[2]);
    let result = clip_grad_norm(&[&g], 10.0).unwrap();
    assert!(!result.scaled);
    assert!((result.total_norm - 5.0).abs() < 1e-6);
    assert_eq!(result.grads[0].as_slice().unwrap(), &[3.0, 4.0]);
}

#[test]
fn clip_grad_norm_scales_down_when_exceeding_max_norm() {
    // total_norm = 5.0 > max_norm = 1.0 → clip_coef = 1.0/(5.0+1e-6) ≈ 0.2
    let g = tensor(vec![3.0, 4.0], &[2]);
    let result = clip_grad_norm(&[&g], 1.0).unwrap();
    assert!(result.scaled);
    assert!((result.total_norm - 5.0).abs() < 1e-6);

    // clip 後の global norm は max_norm 未満にわずかに収まる
    // （分母の 1e-6 ゼロ除算回避定数分。PyTorch clip_grad_norm_ と
    // 同一の定義・同一の余裕）。
    let clipped_norm = global_grad_norm(&[&result.grads[0]]).unwrap();
    assert!(clipped_norm <= 1.0, "clipped_norm={clipped_norm}");
    assert!(
        (clipped_norm - 1.0).abs() < 1e-5,
        "clipped_norm={clipped_norm}"
    );
}

#[test]
fn clip_grad_norm_preserves_direction_uniformly_across_tensors() {
    // スケール係数が全テンソルへ一様に掛かること（方向保存）を、
    // 各成分が clip 前後で同一比率になることで確認する。
    let g1 = tensor(vec![3.0, 4.0], &[2]);
    let g2 = tensor(vec![12.0], &[1]);
    let result = clip_grad_norm(&[&g1, &g2], 1.0).unwrap();
    assert!(result.scaled);

    let ratio0 = result.grads[0].as_slice().unwrap()[0] / 3.0;
    let ratio1 = result.grads[0].as_slice().unwrap()[1] / 4.0;
    let ratio2 = result.grads[1].as_slice().unwrap()[0] / 12.0;
    assert!((ratio0 - ratio1).abs() < 1e-6);
    assert!((ratio1 - ratio2).abs() < 1e-6);
}

// =====================================================================
// エラー系（fail-closed）
// =====================================================================

#[test]
fn clip_grad_norm_rejects_non_positive_max_norm() {
    let g = tensor(vec![1.0], &[1]);
    assert!(clip_grad_norm(&[&g], 0.0).is_err());
    assert!(clip_grad_norm(&[&g], -1.0).is_err());
    assert!(clip_grad_norm(&[&g], f32::NAN).is_err());
    assert!(clip_grad_norm(&[&g], f32::INFINITY).is_err());
}

#[test]
fn clip_grad_norm_propagates_non_finite_grad_error() {
    let g = tensor(vec![f32::INFINITY], &[1]);
    assert!(clip_grad_norm(&[&g], 1.0).is_err());
}

// =====================================================================
// 適用順序（受け入れ条件「clip の適用順序のテスト」）
// =====================================================================

/// AMP 導入前提の疑似 unscale（`grad / scale`）。損失スケーリング本体
/// は本イシューのスコープ外（`optim/mod.rs` doc）だが、「unscale 後に
/// clip する」契約をここで固定する。
fn unscale(grad: &Tensor<f32>, scale: f32) -> Tensor<f32> {
    let data: Vec<f32> = grad
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| v / scale)
        .collect();
    Tensor::from_slice(&data, grad.shape()).unwrap()
}

#[test]
fn clip_after_unscale_matches_clip_on_unscaled_grad() {
    // S 倍にスケールされた勾配 g*S を unscale してから clip した結果は、
    // 最初から S=1 で計算した clip(g) と一致する（正順の契約）。
    let base = tensor(vec![3.0, 4.0], &[2]);
    let scale = 8.0f32;
    let scaled = tensor(vec![3.0 * scale, 4.0 * scale], &[2]);

    let unscaled = unscale(&scaled, scale);
    let clip_after_unscale = clip_grad_norm(&[&unscaled], 1.0).unwrap();
    let clip_direct = clip_grad_norm(&[&base], 1.0).unwrap();

    assert!(
        (clip_after_unscale.total_norm - clip_direct.total_norm).abs() < 1e-4,
        "clip_after_unscale.total_norm={} clip_direct.total_norm={}",
        clip_after_unscale.total_norm,
        clip_direct.total_norm
    );
    for (a, b) in clip_after_unscale.grads[0]
        .as_slice()
        .unwrap()
        .iter()
        .zip(clip_direct.grads[0].as_slice().unwrap())
    {
        assert!((a - b).abs() < 1e-4, "a={a} b={b}");
    }
}

#[test]
fn clip_before_unscale_diverges_from_correct_order() {
    // 逆順（scale が残ったまま clip → unscale）は、正順の結果と
    // 一致しないことを固定する（「unscale 後に clip」の契約を破ると
    // 別の結果になることの回帰テスト）。
    let base = tensor(vec![3.0, 4.0], &[2]);
    let scale = 8.0f32;
    let scaled = tensor(vec![3.0 * scale, 4.0 * scale], &[2]);

    // 逆順: 先に（スケール済みの）勾配へ clip を適用し、その後 unscale する。
    let clip_before_unscale = clip_grad_norm(&[&scaled], 1.0).unwrap();
    let wrong_order = unscale(&clip_before_unscale.grads[0], scale);

    // 正順: unscale してから clip する（契約どおりの結果）。
    let correct_order = clip_grad_norm(&[&base], 1.0).unwrap();

    let wrong_norm = global_grad_norm(&[&wrong_order]).unwrap();
    let correct_norm = global_grad_norm(&[&correct_order.grads[0]]).unwrap();
    // 逆順の結果は正順の 1/scale になる（clip はスケール込みの勾配へ
    // 掛かるため max_norm が実質 max_norm*scale として作用し、その後の
    // unscale で 1/scale 倍されるため）。この関係式そのものを固定する
    // ことで、「wrong_norm と correct_norm が単に異なる」だけでなく
    // clip_grad_norm がスケール共変であることに起因した差であることを
    // 検証する（scale=1 でも false negative にならず、`clip_grad_norm`
    // を恒等関数に差し替えても検知できる回帰テスト）。
    assert!(
        (wrong_norm - correct_norm / scale).abs() < 1e-4,
        "wrong_norm={wrong_norm} correct_norm/scale={} が一致しない\
         （unscale 順序の契約が崩れている）",
        correct_norm / scale
    );
}

// =====================================================================
// ミニ学習ステップ統合: backward → clip → SGD 更新
// =====================================================================

/// `nn_train_convergence.rs::sgd_step` と同一パターン（optimizer 本体
/// 未実装のためテストローカルに代替）。
fn sgd_step(param: &Tensor<f32>, grad: &Tensor<f32>, lr: f32) -> Tensor<f32> {
    let p = param.as_slice().unwrap();
    let g = grad.as_slice().unwrap();
    let data: Vec<f32> = p.iter().zip(g).map(|(&pv, &gv)| pv - lr * gv).collect();
    Tensor::from_slice(&data, param.shape()).unwrap()
}

#[test]
fn clip_applied_to_real_backward_gradients_bounds_update_norm() {
    use autodiff::nn::Linear;

    const BATCH: usize = 4;
    const D_IN: usize = 8;
    const D_OUT: usize = 4;
    const LR: f32 = 0.05;
    const MAX_NORM: f32 = 0.01;

    let mut rng = Xorshift64Star::new(0xC0FFEE);
    let x_data = tensor(rng.fill_vec(BATCH * D_IN), &[BATCH, D_IN]);
    let y_data = tensor(rng.fill_vec(BATCH * D_OUT), &[BATCH, D_OUT]);

    let l1 = Linear::new(D_IN, D_OUT, true, 0x1111_1111).expect("test fixture: shape は事前に妥当");

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let l1v = l1.bind(&tape);

    let h1 = l1v.forward(&x).unwrap();
    let loss = h1.mse_loss(&y).unwrap();
    let grads = tape.backward(&loss).unwrap();

    let weight_grad = grads.get(&l1v.weight).unwrap().unwrap();
    let bias_grad = grads
        .get(l1v.bias.as_ref().expect("test fixture: bias=true で構築"))
        .unwrap()
        .unwrap();

    // clip 前は max_norm=0.01 を大きく超えるはず（通常初期化の勾配は
    // O(1) オーダーのため）。超えていなければ本テストの前提が崩れる。
    let raw_norm = global_grad_norm(&[weight_grad, bias_grad]).unwrap();
    assert!(
        raw_norm > MAX_NORM,
        "raw_norm={raw_norm} が MAX_NORM 以下では clip の効果を検証できない"
    );

    let clipped = clip_grad_norm(&[weight_grad, bias_grad], MAX_NORM).unwrap();
    assert!(clipped.scaled);

    let new_weight = sgd_step(l1.weight(), &clipped.grads[0], LR);
    let new_bias = sgd_step(
        l1.bias().expect("test fixture: bias=true で構築"),
        &clipped.grads[1],
        LR,
    );

    // パラメータ更新量 (new - old) の global norm は lr * max_norm 近傍
    // に抑制される（clip 後の勾配ノルムが max_norm 以下に収まっている
    // ため、SGD 更新則 delta = -lr * grad の性質上の上界）。
    let delta_weight = tensor(
        new_weight
            .as_slice()
            .unwrap()
            .iter()
            .zip(l1.weight().as_slice().unwrap())
            .map(|(&n, &o)| n - o)
            .collect(),
        new_weight.shape(),
    );
    let delta_bias = tensor(
        new_bias
            .as_slice()
            .unwrap()
            .iter()
            .zip(l1.bias().unwrap().as_slice().unwrap())
            .map(|(&n, &o)| n - o)
            .collect(),
        new_bias.shape(),
    );
    let delta_norm = global_grad_norm(&[&delta_weight, &delta_bias]).unwrap();

    // ゼロ除算回避定数 1e-6（clip.rs）により clip 後の勾配ノルムは
    // max_norm を厳密に下回るため、delta_norm は理論上 lr*max_norm を
    // 厳密に下回る。1% の余裕（`* 1.01`）は f32 丸め誤差のみを吸収する
    // 目的で、束縛そのものを緩めない値（既存 tolerance の流用ではなく
    // 本テスト用に新設。coding-rust.md「既存 tolerance の緩和禁止」
    // には抵触しない）。
    let bound = LR * MAX_NORM * 1.01;
    assert!(
        delta_norm <= bound,
        "delta_norm={delta_norm} が上界 {bound}（lr*max_norm={} の 1% 余裕込み）を超えている",
        LR * MAX_NORM
    );
}
