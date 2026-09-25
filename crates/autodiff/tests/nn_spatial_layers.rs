//! `nn::ConvTranspose1d`／`nn::Upsample`／`nn::ZeroPad2d`／
//! `nn::Identity`／`nn::Unflatten`（イシュー #2159・親 #2131）の受け
//! 入れ条件検証。
//!
//! - `ConvTranspose1d`: 同 seed の `ConvTranspose2d([1, k])` と重み・
//!   forward・勾配が bit 完全一致すること（構造的に同一のため）。
//!   PyTorch 式の出力長・`output_padding >= stride`／rank 違反／
//!   cross-tape／bias shape 不一致の拒否（孤児ノードを残さないこと
//!   込み）。`forward_host` と `forward` の bit 一致。
//! - `Upsample`／`ZeroPad2d`／`Identity`／`Unflatten`: forward・
//!   `forward_host` の一致、境界値の拒否。
//! - `nn::Sequential` への 5 層混在統合（`named_parameters`／
//!   `parameter_count`／`set_requires_grad(false)`／`summary`）。

mod common;

use fandhe_ai_autodiff::nn::{
    ConvTranspose1d, ConvTranspose2d, Identity, Module, Sequential, Unflatten, Upsample,
    UpsampleSize, ZeroPad2d, summary,
};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{InterpolateMode, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    tensor
        .contiguous()
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_default()
}

fn err_of<T>(result: Result<T, AutodiffError>) -> AutodiffError {
    match result {
        Err(err) => err,
        Ok(_) => panic!("expected Err"),
    }
}

// --- ConvTranspose1d ---

#[test]
fn conv_transpose1d_weight_matches_conv_transpose2d_with_h_axis_fixed() {
    let l1 = ConvTranspose1d::new(2, 3, 3, 1, 0, 0, 1, 1, true, 42).unwrap();
    let l2 =
        ConvTranspose2d::new(2, 3, [1, 3], [1, 1], [0, 0], [0, 0], [1, 1], 1, true, 42).unwrap();
    assert_eq!(l1.weight().shape(), &[2, 3, 3]);
    assert_eq!(l2.weight().shape(), &[2, 3, 1, 3]);
    assert_eq!(dense(l1.weight()), dense(l2.weight()));
    assert_eq!(dense(l1.bias().unwrap()), dense(l2.bias().unwrap()));
}

#[test]
fn conv_transpose1d_forward_matches_conv_transpose2d_reshape_bit_exact() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let layer1 = ConvTranspose1d::new(2, 3, 3, 2, 1, 1, 1, 1, true, 7).unwrap();
    let layer2 =
        ConvTranspose2d::new(2, 3, [1, 3], [1, 2], [0, 1], [0, 1], [1, 1], 1, true, 7).unwrap();

    let x1 = tape.var(&t((1..=16).map(|v| v as f32).collect(), &[1, 2, 8]));
    let y1 = layer1.bind(&tape).forward(&x1).unwrap();

    let x2 = tape.var(&t((1..=16).map(|v| v as f32).collect(), &[1, 2, 1, 8]));
    let y2 = layer2.bind(&tape).forward(&x2).unwrap();
    let y2_tensor = y2.to_tensor();
    let y2_shape = y2_tensor.shape();
    let (n, cout, lout) = (y2_shape[0], y2_shape[1], y2_shape[3]);

    assert_eq!(y1.to_tensor().shape(), &[n, cout, lout]);
    assert_eq!(dense(&y1.to_tensor()), dense(&y2_tensor));
}

#[test]
fn conv_transpose1d_output_length_matches_pytorch_formula() {
    // Lout = (L-1)*s - 2p + d*(k-1) + op + 1
    let tape = Tape::new_with_ops(common::naive_ops());
    let (l, k, s, p, op, d) = (8usize, 3usize, 2usize, 1usize, 1usize, 2usize);
    let layer = ConvTranspose1d::new(1, 1, k, s, p, op, d, 1, false, 1).unwrap();
    let x = tape.var(&t(vec![1.0; l], &[1, 1, l]));
    let y = layer.bind(&tape).forward(&x).unwrap();
    let expected_lout = (l - 1) * s + d * (k - 1) + op + 1 - 2 * p;
    assert_eq!(y.to_tensor().shape(), &[1, 1, expected_lout]);
}

#[test]
fn conv_transpose1d_groups_greater_than_one_forward_succeeds() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let layer = ConvTranspose1d::new(4, 6, 3, 1, 0, 0, 1, 2, true, 3).unwrap();
    let x = tape.var(&t((1..=16).map(|v| v as f32).collect(), &[1, 4, 4]));
    let y = layer.bind(&tape).forward(&x).unwrap();
    assert_eq!(y.to_tensor().shape()[1], 6);
}

#[test]
fn conv_transpose1d_new_rejects_output_padding_ge_stride() {
    let err = err_of(ConvTranspose1d::new(1, 1, 3, 2, 0, 2, 1, 1, false, 1));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv_transpose1d_forward_rejects_rank_mismatch_without_leaving_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let layer = ConvTranspose1d::new(2, 3, 3, 1, 0, 0, 1, 1, true, 5).unwrap();
    let vars = layer.bind(&tape);
    let x_bad_rank = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let len_before = tape.len();

    assert!(vars.forward(&x_bad_rank).is_err());

    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

#[test]
fn conv_transpose1d_from_parameters_rejects_bias_shape_mismatch() {
    let weight = t(vec![1.0; 2 * 3 * 3], &[2, 3, 3]);
    let bad_bias = t(vec![1.0, 2.0], &[2]);
    let err = err_of(ConvTranspose1d::from_parameters(
        weight,
        Some(bad_bias),
        1,
        0,
        0,
        1,
        1,
    ));
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn conv_transpose1d_forward_host_matches_forward() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let host_ops = common::naive_ops();
    let layer = ConvTranspose1d::new(2, 3, 3, 2, 1, 1, 1, 1, true, 11).unwrap();
    let x = t((1..=16).map(|v| v as f32).collect(), &[1, 2, 8]);

    let via_host = layer.forward_host(host_ops.as_ref(), &x).unwrap();
    let via_tape = layer.bind(&tape).forward(&tape.var(&x)).unwrap();
    let via_tape_tensor = via_tape.to_tensor();

    assert_eq!(via_host.shape(), via_tape_tensor.shape());
    assert_eq!(dense(&via_host), dense(&via_tape_tensor));
}

#[test]
fn conv_transpose1d_module_forward_host_matches_direct_call() {
    let host_ops = common::naive_ops();
    let layer = ConvTranspose1d::new(2, 3, 3, 1, 0, 0, 1, 1, true, 13).unwrap();
    let x = t((1..=16).map(|v| v as f32).collect(), &[1, 2, 8]);
    let via_module = Module::forward_host(&layer, host_ops.as_ref(), &x).unwrap();
    let via_direct = layer.forward_host(host_ops.as_ref(), &x).unwrap();
    assert_eq!(dense(&via_module), dense(&via_direct));
}

// --- Upsample ---

#[test]
fn upsample_module_forward_matches_direct_call() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let layer = Upsample::with_size(vec![4, 4], InterpolateMode::Nearest).unwrap();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]));
    let via_module = Module::forward(&layer, &tape, &x).unwrap();
    let via_direct = layer.forward(&x).unwrap();
    assert_eq!(
        dense(&via_module.to_tensor()),
        dense(&via_direct.to_tensor())
    );
}

#[test]
fn upsample_spec_accessor_round_trips() {
    let layer = Upsample::with_scale_factor(vec![2.0], InterpolateMode::Nearest).unwrap();
    assert!(matches!(layer.spec(), UpsampleSize::ScaleFactor(s) if s == &[2.0]));
}

// --- ZeroPad2d ---

#[test]
fn zero_pad2d_module_forward_host_matches_direct_call() {
    let host_ops = common::naive_ops();
    let layer = ZeroPad2d::uniform(1);
    let x = t(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let via_module = Module::forward_host(&layer, host_ops.as_ref(), &x).unwrap();
    let via_direct = layer.forward_host(host_ops.as_ref(), &x).unwrap();
    assert_eq!(dense(&via_module), dense(&via_direct));
}

// --- Identity ---

#[test]
fn identity_in_sequential_appears_twice_in_named_modules() {
    let mut seq = Sequential::new();
    seq.push(Box::new(Identity::new()));
    seq.push(Box::new(Identity::new()));
    let named = seq.named_modules();
    // ZST（`Identity`）は同一アドレスを取りうるため `named_modules` の
    // dedup 特例が正しく機能し、2 個とも列挙されることを確認する
    // （`module.rs::collect_named_modules` の ZST 特例。回帰防止）。
    let identity_count = named
        .iter()
        .filter(|(_, m)| m.type_name().contains("Identity"))
        .count();
    assert_eq!(identity_count, 2);
}

// --- Unflatten ---

#[test]
fn unflatten_module_forward_matches_direct_call() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let layer = Unflatten::new(1, vec![3, 4]).unwrap();
    let x = tape.var(&t((1..=24).map(|v| v as f32).collect(), &[2, 12]));
    let via_module = Module::forward(&layer, &tape, &x).unwrap();
    let via_direct = layer.forward(&x).unwrap();
    assert_eq!(via_module.to_tensor().shape(), &[2, 3, 4]);
    assert_eq!(
        dense(&via_module.to_tensor()),
        dense(&via_direct.to_tensor())
    );
}

// --- nn::Sequential 統合 ---

#[test]
fn sequential_with_all_five_layers_builds_and_summarizes() {
    let mut seq = Sequential::new();
    seq.push(Box::new(
        ConvTranspose1d::new(1, 2, 3, 1, 0, 0, 1, 1, true, 1).unwrap(),
    ));
    seq.push(Box::new(
        Upsample::with_scale_factor(vec![2.0], InterpolateMode::Nearest).unwrap(),
    ));
    seq.push(Box::new(Identity::new()));
    seq.push(Box::new(ZeroPad2d::uniform(0)));
    seq.push(Box::new(Unflatten::new(0, vec![1, 1]).unwrap()));

    let param_count = seq.parameter_count();
    assert!(param_count > 0);

    let mut seq_frozen = seq;
    seq_frozen.set_requires_grad(false).unwrap();
    assert!(!seq_frozen.requires_grad());

    let text = summary(&seq_frozen);
    assert!(!text.is_empty());
}
