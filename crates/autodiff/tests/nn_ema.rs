//! イシュー #2179（親 #2131「PyTorch／TF 置き換えの API 網羅」）の
//! [`fandhe_ai_autodiff::nn::ExponentialMovingAverage`] 統合テスト。
//!
//! 単体テスト（`crates/autodiff/src/nn/ema.rs` 内 `#[cfg(test)]`）が
//! `update`／`update_named` の数値契約・検証ロジックを検証するのに対し、
//! 本ファイルは [`Module`] trait 経由（`from_module`／
//! `update_from_module`／`apply`／`restore`）の統合動作——実際の
//! `nn::Sequential`（`Linear`→`Relu`→`Linear`）に対する接続——を
//! 検証する。実機（CUDA／Metal）非依存・ホスト計算のみのため
//! `#[ignore]` 分離は行わない。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::activation::Relu;
use fandhe_ai_autodiff::nn::{ExponentialMovingAverage, Linear, Module, Sequential};
use fandhe_ai_tensor_core::Tensor;

fn build_model() -> Sequential {
    Sequential::new()
        .add(Linear::new(4, 8, true, 0x1111).unwrap())
        .add(Relu)
        .add(Linear::new(8, 2, true, 0x2222).unwrap())
}

fn input() -> Tensor<f32> {
    Tensor::new(vec![0.1, -0.2, 0.3, -0.4], &[1, 4]).unwrap()
}

fn forward_output(model: &Sequential, x: &Tensor<f32>) -> Vec<f32> {
    let tape = Tape::new();
    let var = tape.var(x);
    let out = model.forward(&tape, &var).unwrap();
    out.value().as_slice().unwrap().to_vec()
}

/// テスト用のダミー optimizer step（各 `Linear` の `weight` へ定数
/// オフセットを加算する）。`Module::set_parameter`（trait 公開メソッド）
/// 経由で書き戻す（`Linear::set_parameter` は `pub(crate)` のため統合
/// テストからは呼べない）。
fn bump_linear_weights(model: &mut Sequential, offset: f32) {
    for layer in model.layers_mut() {
        if let Some(linear) = layer.as_linear() {
            let w = linear.weight();
            let bumped: Vec<f32> = w.as_slice().unwrap().iter().map(|v| v + offset).collect();
            let new_w = Tensor::new(bumped, w.shape()).unwrap();
            layer.set_parameter("weight", new_w).unwrap();
        }
    }
}

#[test]
fn from_module_and_update_from_module_matches_manual_update() {
    let model = build_model();
    let ema = ExponentialMovingAverage::from_module(0.9, &model).unwrap();

    // `from_module` は構築時パラメータをそのまま shadow へ clone する。
    let initial_named = model.named_parameters();
    let mut initial_sorted = initial_named.clone();
    initial_sorted.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, tensor) in &initial_sorted {
        assert_eq!(
            ema.shadow(name).unwrap().as_slice().unwrap(),
            tensor.as_slice().unwrap()
        );
    }

    // 2 回目以降のパラメータ更新（optimizer step 相当。ここではダミーで
    // 全要素へ定数オフセットを加算する）と `update_from_module` が、
    // `update_named` を直接呼んだ場合と同じ shadow を導くことを確認する。
    let mut model2 = build_model();
    let mut ema2 = ExponentialMovingAverage::from_module(0.9, &model2).unwrap();

    for step in 0..3 {
        bump_linear_weights(&mut model2, 0.01 * (step as f32 + 1.0));
        ema2.update_from_module(&model2).unwrap();
    }
    assert_eq!(ema2.num_updates(), 3);

    // 独立に同じ更新列を `update_named` 経由で再現し、bit 完全一致する
    // ことを確認する（`from_module`/`update_from_module` が
    // `named_parameters()` の単純な委譲であることの裏付け）。
    let mut model3 = build_model();
    let mut ema3 = ExponentialMovingAverage::from_named(0.9, model3.named_parameters()).unwrap();
    for step in 0..3 {
        bump_linear_weights(&mut model3, 0.01 * (step as f32 + 1.0));
        ema3.update_named(model3.named_parameters()).unwrap();
    }

    let names = model2.named_parameters();
    for (name, _) in &names {
        assert_eq!(
            ema2.shadow(name).unwrap().as_slice().unwrap(),
            ema3.shadow(name).unwrap().as_slice().unwrap(),
        );
    }
}

#[test]
fn apply_then_restore_round_trips_to_bit_identical_weights() {
    let mut model = build_model();
    // decay=0.5 で構築し、別のパラメータへ 1 回 update することで
    // shadow を現在の重みから明確にずらす。
    let mut ema = ExponentialMovingAverage::from_module(0.5, &model).unwrap();
    let bumped_named: Vec<(String, Tensor<f32>)> = model
        .named_parameters()
        .into_iter()
        .map(|(name, tensor)| {
            let data: Vec<f32> = tensor.as_slice().unwrap().iter().map(|v| v + 1.0).collect();
            (name, Tensor::new(data, tensor.shape()).unwrap())
        })
        .collect();
    let bumped_refs: Vec<(String, &Tensor<f32>)> = bumped_named
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor))
        .collect();
    ema.update_named(bumped_refs).unwrap();

    let before_apply = model.state_dict();
    let x = input();
    let output_before = forward_output(&model, &x);

    let backup = ema.apply(&mut model).unwrap();

    // 差し替え後の重みは shadow と一致する（forward の出力も変わる
    // はず。decay=0.5・オフセット +1.0 なので shadow は元と異なる）。
    for (name, shadow_tensor) in ema.shadow_state_dict() {
        assert_eq!(
            model.state_dict()[&name].as_slice().unwrap(),
            shadow_tensor.as_slice().unwrap()
        );
    }
    let output_during = forward_output(&model, &x);
    assert_ne!(output_before, output_during);

    ExponentialMovingAverage::restore(&mut model, backup).unwrap();

    // 復元後は元の重みと bit 完全一致する。
    let after_restore = model.state_dict();
    for (name, tensor) in &before_apply {
        assert_eq!(
            tensor.as_slice().unwrap(),
            after_restore[name].as_slice().unwrap()
        );
    }
    let output_after = forward_output(&model, &x);
    assert_eq!(output_before, output_after);
}

#[test]
fn apply_with_mismatched_shape_model_fails_and_leaves_model_unchanged() {
    let model_a = build_model();
    let ema = ExponentialMovingAverage::from_module(0.9, &model_a).unwrap();

    let before = model_a.state_dict();

    // shadow は model_a（4->8->2）由来なので、shape の異なる
    // model_b（3->8->2）へ apply すると shape 不一致で失敗する。
    let mut model_b = Sequential::new()
        .add(Linear::new(3, 8, true, 0x3333).unwrap())
        .add(Relu)
        .add(Linear::new(8, 2, true, 0x4444).unwrap());
    let before_b = model_b.state_dict();
    let err = ema.apply(&mut model_b);
    assert!(err.is_err());

    // `apply` は shape 不一致を検出した時点（`Module::load_state_dict`
    // のパス 1・検証のみ）で `Err` を返し、`model_b` を一切変更しない
    // （two-pass 契約。AC「`Err` かつモデル不変」の裏付け）。
    let after_b = model_b.state_dict();
    for (name, tensor) in &before_b {
        assert_eq!(
            tensor.as_slice().unwrap(),
            after_b[name].as_slice().unwrap()
        );
    }

    // model_a 自体も変化しない（apply は model_b に対してのみ実行した）。
    let after = model_a.state_dict();
    for (name, tensor) in &before {
        assert_eq!(tensor.as_slice().unwrap(), after[name].as_slice().unwrap());
    }
}

#[test]
fn deterministic_repeated_runs_are_bit_identical() {
    let run = || -> Vec<f32> {
        let mut model = build_model();
        let mut ema = ExponentialMovingAverage::from_module(0.8, &model).unwrap();
        for step in 0..3 {
            bump_linear_weights(&mut model, 0.02 * (step as f32 + 1.0));
            ema.update_from_module(&model).unwrap();
        }
        let mut names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        names.sort();
        names
            .into_iter()
            .flat_map(|name| ema.shadow(&name).unwrap().as_slice().unwrap().to_vec())
            .collect()
    };

    assert_eq!(run(), run());
}
