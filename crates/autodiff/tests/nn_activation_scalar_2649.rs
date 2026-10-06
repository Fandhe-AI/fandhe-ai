//! `nn::activation::{Selu, Celu, Softsign, Hardsigmoid, LogSigmoid}`
//! （イシュー #2649）の層としての統合テスト（公開 API 経由）。
//!
//! 層の `forward` が自由関数（`activation_scalar_ops`）と bit 一致すること、
//! `Module::forward`／`forward_host` が一致すること、`Celu::new` の引数検査、
//! 層を通した backward が到達することを確認する。層型は facade 非公開で、
//! 公開は承認依頼 #2677 の承認後に #2679 が担当する。

mod common;

use fandhe_ai_autodiff::activation_scalar_ops as free;
use fandhe_ai_autodiff::nn::Module;
use fandhe_ai_autodiff::nn::activation::{Celu, Hardsigmoid, LogSigmoid, Selu, Softsign};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

const XS: [f32; 6] = [-4.0, -1.0, 0.0, 0.5, 2.0, 5.0];

#[test]
fn layers_match_free_functions_bit_for_bit() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(XS.to_vec(), &[2, 3]));
    let celu = Celu::new(1.5).unwrap();
    let pairs = [
        (Selu.forward(&x).unwrap(), free::selu(&x).unwrap()),
        (celu.forward(&x).unwrap(), free::celu(&x, 1.5).unwrap()),
        (Softsign.forward(&x).unwrap(), free::softsign(&x).unwrap()),
        (
            Hardsigmoid.forward(&x).unwrap(),
            free::hardsigmoid(&x).unwrap(),
        ),
        (
            LogSigmoid.forward(&x).unwrap(),
            free::log_sigmoid(&x).unwrap(),
        ),
    ];
    for (layer, func) in pairs {
        let (a, b) = (layer.to_tensor(), func.to_tensor());
        assert_eq!(a.shape(), b.shape());
        let bits =
            |t: &Tensor<f32>| -> Vec<u32> { t.host_slice().iter().map(|v| v.to_bits()).collect() };
        assert_eq!(bits(&a), bits(&b));
    }
}

#[test]
fn module_forward_and_forward_host_agree() {
    let ops = common::naive_ops();
    let input = t(XS.to_vec(), &[6]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&input);
    let layers: Vec<Box<dyn Module>> = vec![
        Box::new(Selu),
        Box::new(Celu::new(2.0).unwrap()),
        Box::new(Softsign),
        Box::new(Hardsigmoid),
        Box::new(LogSigmoid),
    ];
    for layer in &layers {
        assert!(layer.named_parameters().is_empty(), "パラメータを持たない");
        let via_tape = layer.forward(&tape, &xv).unwrap().to_tensor();
        let via_host = layer.forward_host(ops.as_ref(), &input).unwrap();
        assert_eq!(
            via_tape.host_slice().as_ref(),
            via_host.host_slice().as_ref()
        );
    }
}

#[test]
fn celu_new_rejects_invalid_alpha_and_default_is_one() {
    for alpha in [0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            Celu::new(alpha),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
    assert!(Celu::new(-1.5).is_ok());
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-1.0, 2.0], &[2]));
    let d = Celu::default().forward(&x).unwrap().to_tensor();
    let e = free::celu(&x, 1.0).unwrap().to_tensor();
    assert_eq!(d.host_slice().as_ref(), e.host_slice().as_ref());
}

#[test]
fn backward_through_layers_reaches_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-2.0, -0.5, 0.7, 1.5], &[4]));
    let celu = Celu::new(1.5).unwrap();
    let ys = [
        Selu.forward(&x).unwrap(),
        celu.forward(&x).unwrap(),
        Softsign.forward(&x).unwrap(),
        Hardsigmoid.forward(&x).unwrap(),
        LogSigmoid.forward(&x).unwrap(),
    ];
    for y in ys {
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().expect("入力へ勾配が届く").clone();
        assert!(dx.host_slice().iter().all(|v| v.is_finite()));
        assert!(dx.host_slice().iter().any(|v| *v != 0.0));
    }
}
