//! `nn::{MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, AdaptiveAvgPool1d,
//! AdaptiveAvgPool2d}`（イシュー #1728）の受け入れ条件検証。
//!
//! - `Module::forward` ≡ `forward_host` の bit 一致
//!   （`nn_gelu_softplus.rs` と同方針。`common::naive_ops()` 経由）。
//! - `Module::forward` ≡ `Var::*` 直呼びの bit 一致。
//! - 構築時検査（`Pool2dParams::new`／`output_size >= 1`）。
//! - `named_parameters` が空であること（無状態層）。

mod common;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d, Module,
};
use fandhe_ai_tensor_core::Tensor;

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

fn assert_bit_exact(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape が一致しない");
    let av = dense(a);
    let bv = dense(b);
    for (i, (&x, &y)) in av.iter().zip(bv.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}[{i}]: bit 不一致（a={x}, b={y}）"
        );
    }
}

#[test]
fn max_pool2d_module_forward_matches_forward_host_and_var_direct() {
    let x = t(
        vec![1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0, 7.0, 6.0],
        &[1, 1, 3, 3],
    );
    let layer = MaxPool2d::new([2, 2], Some([1, 1]), [0, 0], [1, 1]).unwrap();
    let ops = common::naive_ops();

    let tape_direct = Tape::new_with_ops(common::naive_ops());
    let xv_direct = tape_direct.var(&x);
    let (via_direct, _idx) = layer.forward(&xv_direct).unwrap();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_module = <MaxPool2d as Module>::forward(&layer, &tape, &xv).unwrap();
    let via_host = layer.forward_host(ops.as_ref(), &x).unwrap();
    assert_bit_exact(
        "max_pool2d direct vs module",
        &via_direct.to_tensor(),
        &via_module.to_tensor(),
    );
    assert_bit_exact(
        "max_pool2d module vs forward_host",
        &via_module.to_tensor(),
        &via_host,
    );
}

#[test]
fn max_pool1d_module_forward_matches_forward_host() {
    let x = t(vec![1.0, 5.0, 2.0, 8.0, 3.0, 9.0], &[1, 1, 6]);
    let layer = MaxPool1d::new(2, Some(2), 0, 1).unwrap();
    let ops = common::naive_ops();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_module = <MaxPool1d as Module>::forward(&layer, &tape, &xv).unwrap();
    let via_host = layer.forward_host(ops.as_ref(), &x).unwrap();

    assert_bit_exact(
        "max_pool1d module vs forward_host",
        &via_module.to_tensor(),
        &via_host,
    );
}

#[test]
fn avg_pool2d_module_forward_matches_forward_host_and_var_direct() {
    let x = t((0..16).map(|v| v as f32 * 0.25).collect(), &[1, 1, 4, 4]);
    let layer = AvgPool2d::new([2, 2], Some([2, 2]), [1, 1], true).unwrap();
    let ops = common::naive_ops();

    let tape_direct = Tape::new_with_ops(common::naive_ops());
    let xv_direct = tape_direct.var(&x);
    let via_direct = layer.forward(&xv_direct).unwrap();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_module = <AvgPool2d as Module>::forward(&layer, &tape, &xv).unwrap();
    let via_host = layer.forward_host(ops.as_ref(), &x).unwrap();

    assert_bit_exact(
        "avg_pool2d direct vs module",
        &via_direct.to_tensor(),
        &via_module.to_tensor(),
    );
    assert_bit_exact(
        "avg_pool2d module vs forward_host",
        &via_module.to_tensor(),
        &via_host,
    );
}

#[test]
fn avg_pool1d_module_forward_matches_forward_host() {
    let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
    let layer = AvgPool1d::new(2, Some(2), 0, true).unwrap();
    let ops = common::naive_ops();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_module = <AvgPool1d as Module>::forward(&layer, &tape, &xv).unwrap();
    let via_host = layer.forward_host(ops.as_ref(), &x).unwrap();

    assert_bit_exact(
        "avg_pool1d module vs forward_host",
        &via_module.to_tensor(),
        &via_host,
    );
}

#[test]
fn adaptive_avg_pool2d_module_forward_matches_forward_host_and_var_direct() {
    let x = t((0..16).map(|v| v as f32).collect(), &[1, 1, 4, 4]);
    let layer = AdaptiveAvgPool2d::new([2, 2]).unwrap();
    let ops = common::naive_ops();

    let tape_direct = Tape::new_with_ops(common::naive_ops());
    let xv_direct = tape_direct.var(&x);
    let via_direct = layer.forward(&xv_direct).unwrap();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_module = <AdaptiveAvgPool2d as Module>::forward(&layer, &tape, &xv).unwrap();
    let via_host = layer.forward_host(ops.as_ref(), &x).unwrap();

    assert_bit_exact(
        "adaptive_avg_pool2d direct vs module",
        &via_direct.to_tensor(),
        &via_module.to_tensor(),
    );
    assert_bit_exact(
        "adaptive_avg_pool2d module vs forward_host",
        &via_module.to_tensor(),
        &via_host,
    );
}

#[test]
fn adaptive_avg_pool1d_module_forward_matches_forward_host() {
    let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 1, 7]);
    let layer = AdaptiveAvgPool1d::new(2).unwrap();
    let ops = common::naive_ops();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_module = <AdaptiveAvgPool1d as Module>::forward(&layer, &tape, &xv).unwrap();
    let via_host = layer.forward_host(ops.as_ref(), &x).unwrap();

    assert_bit_exact(
        "adaptive_avg_pool1d module vs forward_host",
        &via_module.to_tensor(),
        &via_host,
    );
}

// --- 構築時検査 ---

#[test]
fn max_pool2d_new_rejects_padding_over_half_kernel() {
    assert!(MaxPool2d::new([2, 2], None, [2, 2], [1, 1]).is_err());
}

#[test]
fn max_pool2d_new_rejects_zero_kernel() {
    assert!(MaxPool2d::new([0, 2], None, [0, 0], [1, 1]).is_err());
}

#[test]
fn avg_pool2d_new_rejects_padding_over_half_kernel() {
    assert!(AvgPool2d::new([2, 2], None, [2, 2], true).is_err());
}

#[test]
fn adaptive_avg_pool2d_new_rejects_output_size_zero() {
    assert!(AdaptiveAvgPool2d::new([0, 2]).is_err());
    assert!(AdaptiveAvgPool2d::new([2, 0]).is_err());
}

#[test]
fn adaptive_avg_pool1d_new_rejects_output_size_zero() {
    assert!(AdaptiveAvgPool1d::new(0).is_err());
}

// --- named_parameters は空（無状態層） ---

#[test]
fn pooling_layers_have_no_named_parameters() {
    let max2d = MaxPool2d::new([2, 2], None, [0, 0], [1, 1]).unwrap();
    let avg2d = AvgPool2d::new([2, 2], None, [0, 0], true).unwrap();
    let adaptive2d = AdaptiveAvgPool2d::new([2, 2]).unwrap();
    assert!(max2d.named_parameters().is_empty());
    assert!(avg2d.named_parameters().is_empty());
    assert!(adaptive2d.named_parameters().is_empty());
}
