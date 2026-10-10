//! `fandhe_ai::models::{ResNet, ResNetBlock, TransformerClassifier, TransformerClassifierConfig}`
//! の公開パス限定テスト（イシュー #2975・親 #2541 の Phase 11-2）。
//!
//! `examples/` の `#[path]` 取り込みを使わず、公開パス（`fandhe_ai::models`・`fandhe_ai::optim`・
//! `fandhe_ai::{Tensor, AutodiffError}`）と `std` だけで、`new`・`predict`/`forward` の bit 一致・
//! `train_step`（損失が有限・train モードへの切替・型付きエラー）・名前衝突の不在を確かめる。
//! AC4 相当の 10 epoch 学習判定は `example_resnet_cifar10.rs`／`example_transformer_cifar10.rs`
//! が担う。
//!
//! 保存（R5）: 新しい型は `compat::Sequential` を外へ出さず `nn::Module` も実装しないため、
//! `compat::save_model`／`load_model` に渡す経路が存在しない。fail-closed の実行テストは作れず、
//! 経路の不在は `api_surface.rs` のインベントリ（メソッド集合の完全一致）と
//! `models_unapproved_paths_are_absent` が固定する。実機依存の `#[ignore]` テストは持たない。

use fandhe_ai::models::{ResNet, ResNetBlock, TransformerClassifier, TransformerClassifierConfig};
use fandhe_ai::optim::{Adam, AdamConfig};
use fandhe_ai::{AutodiffError, Tensor};

/// R7 の名前衝突プローブ。glob 同士の曖昧性は「参照したとき」に初めてエラーになるため、
/// 4 名を実際に型位置で参照する。正のプローブで、コンパイルが通ること自体が検査になる
/// （`compile_fail` のコード照合には頼らない）。
mod glob_probe {
    use fandhe_ai::models::*;
    use fandhe_ai::nn::*;

    pub fn probe(
        _: Option<&Transformer>,
        _: Option<&TransformerClassifier>,
        _: Option<&TransformerClassifierConfig>,
        _: Option<&ResNet>,
        _: Option<&ResNetBlock>,
    ) {
    }
}

/// 決定的な擬似入力（`sin` 系の固定生成）。
fn det_input(len: usize, shape: &[usize]) -> Tensor<f32> {
    let data: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();
    Tensor::new(data, shape).unwrap()
}

fn labels(n: usize, num_classes: usize) -> Tensor<i32> {
    Tensor::new((0..n).map(|i| (i % num_classes) as i32).collect(), &[n]).unwrap()
}

fn small_config() -> TransformerClassifierConfig {
    TransformerClassifierConfig {
        seq_len: 4,
        in_features: 6,
        embed_dim: 8,
        num_heads: 2,
        num_layers: 1,
        dim_feedforward: 16,
        mlp_hidden: 16,
        num_classes: 3,
    }
}

#[test]
fn glob_import_of_nn_and_models_compiles() {
    glob_probe::probe(None, None, None, None, None);
}

#[test]
fn resnet_train_step_returns_finite_loss_and_forces_train_mode() {
    let mut model = ResNet::new(8, 4, 10, 0xA1).unwrap();
    model.set_training(false);
    assert!(!model.training());
    let x = det_input(4 * 3 * 32 * 32, &[4, 3, 32, 32]);
    let y = labels(4, 10);
    let mut opt = Adam::new(AdamConfig::default()).unwrap();
    let loss = model.train_step(&x, &y, &mut opt).unwrap();
    assert!(loss.is_finite(), "loss = {loss}");
    assert!(model.training(), "train_step は冒頭で train モードにする");
}

#[test]
fn transformer_classifier_train_step_returns_finite_loss_and_forces_train_mode() {
    let mut model = TransformerClassifier::new(small_config(), 0xB2).unwrap();
    model.set_training(false);
    assert!(!model.training());
    let x = det_input(5 * 4 * 6, &[5, 4, 6]);
    let y = labels(5, 3);
    let mut opt = Adam::new(AdamConfig::default()).unwrap();
    let loss = model.train_step(&x, &y, &mut opt).unwrap();
    assert!(loss.is_finite(), "loss = {loss}");
    assert!(model.training());
    assert_eq!(model.config().num_classes, 3);
}

#[test]
fn train_step_rejects_invalid_inputs_with_typed_errors() {
    let mut opt = Adam::new(AdamConfig::default()).unwrap();

    let mut resnet = ResNet::new(8, 2, 10, 1).unwrap();
    let x = det_input(2 * 3 * 8 * 8, &[2, 3, 8, 8]);
    let out_of_range = Tensor::<i32>::new(vec![0, 10], &[2]).unwrap();
    let rank2 = Tensor::<i32>::new(vec![0, 1], &[2, 1]).unwrap();
    let bad_x = det_input(2 * 5, &[2, 5]);
    for (x, y) in [(&x, &out_of_range), (&x, &rank2), (&bad_x, &labels(2, 10))] {
        assert!(matches!(
            resnet.train_step(x, y, &mut opt),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    let mut tc = TransformerClassifier::new(small_config(), 2).unwrap();
    let x = det_input(2 * 4 * 6, &[2, 4, 6]);
    let out_of_range = Tensor::<i32>::new(vec![0, 3], &[2]).unwrap();
    let bad_x = det_input(2 * 4 * 5, &[2, 4, 5]);
    for (x, y) in [(&x, &out_of_range), (&x, &rank2), (&bad_x, &labels(2, 3))] {
        assert!(matches!(
            tc.train_step(x, y, &mut opt),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
}

#[test]
fn predict_matches_forward_bit_for_bit_in_eval() {
    let mut resnet = ResNet::new(8, 2, 10, 3).unwrap();
    resnet.set_training(false);
    let x = det_input(2 * 3 * 8 * 8, &[2, 3, 8, 8]);
    let via_predict = resnet.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_forward = resnet.forward(&tape, &xv).unwrap().to_tensor();
    assert_eq!(via_predict.shape(), &[2, 10]);
    assert_eq!(
        via_predict.contiguous().as_slice().unwrap(),
        via_forward.contiguous().as_slice().unwrap()
    );

    let mut tc = TransformerClassifier::new(small_config(), 4).unwrap();
    tc.set_training(false);
    let x = det_input(2 * 4 * 6, &[2, 4, 6]);
    let via_predict = tc.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_forward = tc.forward(&tape, &xv).unwrap().to_tensor();
    assert_eq!(via_predict.shape(), &[2, 3]);
    assert_eq!(
        via_predict.contiguous().as_slice().unwrap(),
        via_forward.contiguous().as_slice().unwrap()
    );
}

#[test]
fn named_parameters_prefixes_and_getters() {
    let resnet = ResNet::new(8, 2, 10, 5).unwrap();
    let names: Vec<String> = resnet
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    for prefix in ["stem.", "layer.", "head."] {
        assert!(names.iter().any(|n| n.starts_with(prefix)), "{prefix}");
    }
    assert_eq!(
        (resnet.depth(), resnet.width(), resnet.num_classes()),
        (8, 2, 10)
    );
    let projections: Vec<bool> = resnet
        .blocks()
        .iter()
        .map(ResNetBlock::has_projection_shortcut)
        .collect();
    assert_eq!(projections, vec![false, true, true]);

    let tc = TransformerClassifier::new(small_config(), 6).unwrap();
    let names: Vec<String> = tc.named_parameters().into_iter().map(|(n, _)| n).collect();
    for prefix in ["embed.", "encoder.", "head."] {
        assert!(names.iter().any(|n| n.starts_with(prefix)), "{prefix}");
    }
}
