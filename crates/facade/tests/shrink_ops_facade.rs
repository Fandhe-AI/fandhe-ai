//! facade 公開面だけで `Var::hardshrink`／`Var::softshrink`／`Sequential::add_hardshrink`／
//! `add_softshrink` に到達できることの固定（イシュー #3817・親 #3813）。`fandhe_ai::*` のみを use する。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{AutodiffError, Tensor, Var};

/// シグネチャを関数ポインタ型の注釈で固定する（承認形からの逸脱をコンパイルエラーにする）。
#[test]
fn approved_signatures_are_pinned() {
    fn pin<'a, 't>(_: fn(&'a Var<'t>, f32) -> Result<Var<'t>, AutodiffError>) {}
    pin(Var::hardshrink);
    pin(Var::softshrink);
    let _: fn(Sequential, f32) -> Result<Sequential, AutodiffError> = Sequential::add_hardshrink;
    let _: fn(Sequential, f32) -> Result<Sequential, AutodiffError> = Sequential::add_softshrink;
}

fn grad_and_value(
    f: impl for<'a> Fn(&Var<'a>) -> Result<Var<'a>, AutodiffError>,
) -> (Vec<f32>, Vec<f32>) {
    let tape = fandhe_ai::tape();
    let x = tape.var(&Tensor::new(vec![-1.0f32, -0.5, 0.25, 0.5, 0.75], &[5]).unwrap());
    let y = f(&x).unwrap();
    let out = y.to_tensor().host_slice().to_vec();
    let loss = y.sum(None).unwrap();
    let g = tape.backward(&loss).unwrap();
    (out, g.get(&x).unwrap().unwrap().host_slice().to_vec())
}

#[test]
fn hardshrink_via_facade_forward_backward() {
    let (y, g) = grad_and_value(|x| x.hardshrink(0.5));
    assert_eq!(y, vec![-1.0, 0.0, 0.0, 0.0, 0.75]);
    assert_eq!(g, vec![1.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn softshrink_via_facade_forward_backward() {
    let (y, g) = grad_and_value(|x| x.softshrink(0.5));
    assert_eq!(y, vec![-0.5, 0.0, 0.0, 0.0, 0.25]);
    assert_eq!(g, vec![1.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn invalid_lambd_is_rejected_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&Tensor::new(vec![1.0f32], &[1]).unwrap());
    for bad in [-0.5f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            x.hardshrink(bad),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            x.softshrink(bad),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(Sequential::new().add_hardshrink(bad).is_err());
        assert!(Sequential::new().add_softshrink(bad).is_err());
    }
}
