//! `TransformerDecoderLayer`・`Transformer`（イシュー #2165・親 #2131・
//! #2068 の対）の統合テスト。`nn::Sequential`（autodiff 汎用コンテナ。
//! `nn::container::Sequential::add<M: Module + 'static>` は任意の
//! `Module` 実装を受け付けるため、facade `compat::Sequential`（本 2 層
//! の追加は承認待ち・`TransformerDecoderHoldDoctestGuard` で保留）を
//! 経由せずとも積める）に積んで forward・backward・`state_dict`／
//! `load_state_dict` 往復・`freeze` を検証する。

mod common;

use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::nn::{
    FeedForwardActivation, LAYER_NORM_DEFAULT_EPS, Module, Sequential, Transformer,
    TransformerConfig, TransformerDecoderLayer,
};
use fandhe_ai_tensor_core::Tensor;

const D_MODEL: usize = 4;
const NUM_HEADS: usize = 2;
const DIM_FF: usize = 8;

fn decoder_layer(seed: u64) -> TransformerDecoderLayer {
    TransformerDecoderLayer::new(
        D_MODEL,
        NUM_HEADS,
        DIM_FF,
        FeedForwardActivation::Relu,
        LAYER_NORM_DEFAULT_EPS,
        seed,
    )
    .unwrap()
}

fn small_transformer(seed: u64) -> Transformer {
    let config = TransformerConfig::new(D_MODEL, NUM_HEADS)
        .with_num_encoder_layers(1)
        .with_num_decoder_layers(1)
        .with_dim_feedforward(DIM_FF);
    Transformer::new(&config, seed).unwrap()
}

// --- `nn::Sequential` へ積んで forward・backward（Module::forward の
// 単一入力慣習: tgt=memory=input・src=tgt=input） ---

#[test]
fn sequential_with_transformer_decoder_layer_forward_and_backward() {
    let tape = fandhe_ai_autodiff::Tape::new_with_ops(common::naive_ops());
    let seq = Sequential::new().add(decoder_layer(11));
    let x = tape.var(&Tensor::new(vec![0.1f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap());
    let out = seq.forward(&tape, &x).unwrap();
    assert_eq!(out.to_tensor().shape().to_vec(), vec![2, 3, D_MODEL]);

    let loss = out.mean(None).unwrap();
    // `Sequential::forward`（`Module::forward` 経由。`tgt = memory =
    // input` の単一入力慣習）から得た出力に対して `backward` が
    // panic せず成功することを確認する（個々のパラメータへの勾配
    // 到達自体は `nn/transformer_decoder_layer.rs::
    // backward_reaches_all_twenty_six_parameters` が単体テストとして
    // 担保済み）。
    tape.backward(&loss).unwrap();
    assert!(loss.to_tensor().contiguous().as_slice().unwrap()[0].is_finite());
}

#[test]
fn sequential_with_transformer_forward_and_backward() {
    let tape = fandhe_ai_autodiff::Tape::new_with_ops(common::naive_ops());
    let seq = Sequential::new().add(small_transformer(13));
    let x = tape.var(&Tensor::new(vec![0.1f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap());
    let out = seq.forward(&tape, &x).unwrap();
    assert_eq!(out.to_tensor().shape().to_vec(), vec![2, 3, D_MODEL]);

    let loss = out.mean(None).unwrap();
    tape.backward(&loss).unwrap();
}

// --- `state_dict`／`load_state_dict` 往復（bit 同一） ---

#[test]
fn transformer_decoder_layer_state_dict_round_trip_is_bit_identical() {
    let layer_a = decoder_layer(17);
    let dict = layer_a.state_dict();

    let mut layer_b = decoder_layer(19); // 別シードで構築（元と異なる重み）。
    layer_b.load_state_dict(dict.clone()).unwrap();

    let names_a: Vec<String> = layer_a
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    let names_b: Vec<String> = layer_b
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(names_a, names_b);

    for name in names_a {
        let a = dict.get(&name).unwrap();
        let b = layer_b
            .named_parameters()
            .into_iter()
            .find(|(n, _)| n == &name)
            .unwrap()
            .1;
        assert_eq!(
            a.contiguous().as_slice().unwrap(),
            b.contiguous().as_slice().unwrap(),
            "load_state_dict 後にパラメータ `{name}` が bit 同一でない"
        );
    }
}

#[test]
fn transformer_state_dict_round_trip_is_bit_identical() {
    let model_a = small_transformer(23);
    let dict = model_a.state_dict();

    let mut model_b = small_transformer(29);
    model_b.load_state_dict(dict).unwrap();

    for (name, tensor_a) in model_a.named_parameters() {
        let tensor_b = model_b
            .named_parameters()
            .into_iter()
            .find(|(n, _)| n == &name)
            .unwrap()
            .1;
        assert_eq!(
            tensor_a.contiguous().as_slice().unwrap(),
            tensor_b.contiguous().as_slice().unwrap(),
            "load_state_dict 後にパラメータ `{name}` が bit 同一でない"
        );
    }
}

/// `load_state_dict` は未知のキーを拒否する（**strict 限定**実装。
/// （`Module::load_state_dict` の既定契約。fail-closed）。
#[test]
fn transformer_decoder_layer_load_state_dict_strict_rejects_unknown_key() {
    let mut layer = decoder_layer(31);
    let mut dict = layer.state_dict();
    dict.insert(
        "bogus.weight".to_string(),
        Tensor::new(vec![0.0f32; 1], &[1]).unwrap(),
    );
    let err = layer.load_state_dict(dict).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// --- `freeze`（`set_requires_grad(false)`）で学習対象から外れる ---

#[test]
fn transformer_decoder_layer_freeze_sets_requires_grad_false_for_all_children() {
    let mut layer = decoder_layer(37);
    assert!(Module::requires_grad(&layer));
    Module::set_requires_grad(&mut layer, false).unwrap();
    assert!(!Module::requires_grad(&layer));
    assert!(!Module::requires_grad(layer.self_attn()));
    assert!(!Module::requires_grad(layer.multihead_attn()));
}

#[test]
fn transformer_freeze_sets_requires_grad_false_for_all_children() {
    let mut model = small_transformer(41);
    assert!(Module::requires_grad(&model));
    Module::set_requires_grad(&mut model, false).unwrap();
    assert!(!Module::requires_grad(&model));
    assert!(!Module::requires_grad(&model.encoder_layers()[0]));
    assert!(!Module::requires_grad(&model.decoder_layers()[0]));
}
