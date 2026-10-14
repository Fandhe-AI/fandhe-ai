//! `nn::{Unfold, Fold}`（イシュー #3039）の受け入れ条件検証。
//!
//! - 層 `forward` ≡ `Var::unfold`／`fold` 直呼びの forward 値・入力勾配の bit 一致
//!   （`common::naive_ops()` 経由）。
//! - 構築時・forward 時の不正引数が型付きエラーで、テープに孤児ノードを残さないこと。
//! - 無状態層（`named_parameters` 空）・`as_*` フック・`supports_forward_host`。

mod common;

use fandhe_ai_autodiff::nn::activation::Relu;
use fandhe_ai_autodiff::nn::{Fold, Module, Unfold};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::{BackendError, ShapeError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn bits(x: &Tensor<f32>) -> Vec<u32> {
    x.contiguous()
        .as_slice()
        .map(|s| s.iter().map(|v| v.to_bits()).collect())
        .unwrap_or_default()
}

fn seq(n: usize, shape: &[usize]) -> Tensor<f32> {
    let data: Vec<f32> = (0..n)
        .map(|i| (((i * 37 + 11) % 23) as f32 - 9.0) * 0.5)
        .collect();
    t(data, shape)
}

/// `(out * g).sum()` の forward 値・shape・入力勾配を返す。
fn run<F>(x: &Tensor<f32>, f: F) -> (Vec<u32>, Vec<usize>, Vec<u32>)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Var<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let y = f(&tape, &xv);
    let out = y.to_tensor();
    let g: Vec<f32> = (0..out.shape().iter().product::<usize>())
        .map(|i| 0.25 + (i % 7) as f32)
        .collect();
    let gv = tape.var_no_grad(&t(g, out.shape()));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("入力へ勾配が届く").clone();
    (bits(&out), out.shape().to_vec(), bits(&dx))
}

type Arr = [usize; 2];
// (kernel, stride, padding, dilation)
const CASES: [(Arr, Arr, Arr, Arr); 5] = [
    ([2, 2], [2, 2], [0, 0], [1, 1]),
    ([3, 3], [1, 1], [0, 0], [1, 1]),
    ([3, 3], [1, 1], [1, 1], [1, 1]),
    ([2, 2], [1, 1], [0, 0], [2, 2]),
    ([2, 3], [1, 2], [1, 0], [1, 2]),
];

fn unfold_input() -> Tensor<f32> {
    seq(2 * 3 * 6 * 7, &[2, 3, 6, 7])
}

#[test]
fn unfold_layer_matches_var_direct_forward_and_grad() {
    let x = unfold_input();
    for (k, s, p, d) in CASES {
        let layer = Unfold::new(k, s, p, d).unwrap();
        let l = run(&x, |tape, v| {
            <Unfold as Module>::forward(&layer, tape, v).unwrap()
        });
        let r = run(&x, |_, v| v.unfold(k, s, p, d).unwrap());
        assert_eq!(l.1, r.1);
        assert_eq!(l.0, r.0, "unfold forward {k:?} {s:?} {p:?} {d:?}");
        assert_eq!(l.2, r.2, "unfold grad {k:?} {s:?} {p:?} {d:?}");
        let inherent = run(&x, |_, v| layer.forward(v).unwrap());
        assert_eq!(inherent, r);
    }
}

#[test]
fn fold_layer_matches_var_direct_via_roundtrip_input() {
    // Unfold の出力 [N, K, L] を Fold の入力として使う（往復）。
    for (k, s, p, d) in CASES {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&unfold_input());
        let cols = xv.unfold(k, s, p, d).unwrap().to_tensor();
        let fold_layer = Fold::new([6, 7], k, s, p, d).unwrap();
        let l = run(&cols, |tape, v| {
            <Fold as Module>::forward(&fold_layer, tape, v).unwrap()
        });
        let r = run(&cols, |_, v| v.fold([6, 7], k, s, p, d).unwrap());
        assert_eq!(l.1, vec![2, 3, 6, 7]);
        assert_eq!(l.1, r.1);
        assert_eq!(l.0, r.0, "fold forward {k:?} {s:?} {p:?} {d:?}");
        assert_eq!(l.2, r.2, "fold grad {k:?} {s:?} {p:?} {d:?}");

        // 層同士の往復（unfold → fold）が Var 直呼びの往復と一致する。
        let unfold_layer = Unfold::new(k, s, p, d).unwrap();
        let a = run(&unfold_input(), |tape, v| {
            let c = <Unfold as Module>::forward(&unfold_layer, tape, v).unwrap();
            <Fold as Module>::forward(&fold_layer, tape, &c).unwrap()
        });
        let b = run(&unfold_input(), |_, v| {
            v.unfold(k, s, p, d)
                .unwrap()
                .fold([6, 7], k, s, p, d)
                .unwrap()
        });
        assert_eq!(a, b);
    }
}

#[test]
fn unfold_non_finite_values_are_bit_identical() {
    let mut data = vec![0.5f32; 4 * 4];
    data[0] = f32::NAN;
    data[1] = f32::INFINITY;
    data[2] = f32::NEG_INFINITY;
    data[3] = -0.0;
    let x = t(data, &[1, 1, 4, 4]);
    let layer = Unfold::new([2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
    let l = run(&x, |tape, v| {
        <Unfold as Module>::forward(&layer, tape, v).unwrap()
    });
    let r = run(&x, |_, v| v.unfold([2, 2], [1, 1], [0, 0], [1, 1]).unwrap());
    assert_eq!(l.0, r.0);
}

#[test]
fn layers_are_stateless_and_hooks_are_exact() {
    let u = Unfold::new([2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
    let f = Fold::new([4, 4], [2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
    assert!(u.named_parameters().is_empty());
    assert!(f.named_parameters().is_empty());

    assert!(u.as_unfold().is_some());
    assert!(u.as_fold().is_none());
    assert!(f.as_fold().is_some());
    assert!(f.as_unfold().is_none());
    assert!(Relu.as_unfold().is_none());
    assert!(Relu.as_fold().is_none());

    assert!(!u.is_pooling());
    assert!(!f.is_pooling());
    assert!(!u.supports_forward_host());
    assert!(!f.supports_forward_host());

    let ops = common::naive_ops();
    for r in [
        u.forward_host(ops.as_ref(), &unfold_input()),
        f.forward_host(ops.as_ref(), &unfold_input()),
    ] {
        assert!(matches!(
            r,
            Err(AutodiffError::Backend(BackendError::Unsupported(_)))
        ));
    }
}

fn is_invalid_arg<T>(r: Result<T, AutodiffError>) -> bool {
    matches!(
        r,
        Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
    )
}

#[test]
fn constructors_reject_invalid_arguments() {
    let (o, z) = ([1, 1], [0, 0]);
    assert!(is_invalid_arg(Unfold::new([0, 2], o, z, o)));
    assert!(is_invalid_arg(Unfold::new([2, 2], [1, 0], z, o)));
    assert!(is_invalid_arg(Unfold::new([2, 2], o, z, [0, 1])));
    assert!(is_invalid_arg(Unfold::new(
        [2, 2],
        o,
        [usize::MAX / 2 + 1, 0],
        o
    )));
    assert!(is_invalid_arg(Fold::new([4, 4], [0, 2], o, z, o)));
    assert!(is_invalid_arg(Fold::new([4, 4], [2, 2], [0, 1], z, o)));
    assert!(is_invalid_arg(Fold::new([4, 4], [2, 2], o, z, [1, 0])));
    assert!(is_invalid_arg(Fold::new(
        [4, 4],
        [2, 2],
        o,
        [0, usize::MAX / 2 + 1],
        o
    )));
    assert!(matches!(
        Fold::new([0, 4], [2, 2], o, z, o),
        Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
    ));
    assert!(matches!(
        Fold::new([4, 0], [2, 2], o, z, o),
        Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
    ));
}

#[test]
fn forward_rejects_bad_shapes_without_orphan_nodes() {
    let o = [1, 1];
    let z = [0, 0];
    let u = Unfold::new([2, 2], o, z, o).unwrap();
    let big_k = Unfold::new([9, 9], o, z, o).unwrap();
    let f = Fold::new([4, 4], [2, 2], o, z, o).unwrap();
    let huge = Fold::new([1 << 31, 1 << 31], [2, 2], o, z, o).unwrap();

    let tape = Tape::new_with_ops(common::naive_ops());
    let rank3 = tape.var(&t(vec![0.0; 12], &[1, 3, 4]));
    let img = tape.var(&seq(2 * 3 * 4 * 4, &[2, 3, 4, 4]));
    // K=5 は kH*kW=4 で割り切れない。
    let bad_k = tape.var(&t(vec![0.0; 5 * 9], &[1, 5, 9]));
    // K=4 は OK だが L=8 は窓数 9 と不一致。
    let bad_l = tape.var(&t(vec![0.0; 4 * 8], &[1, 4, 8]));
    let before = tape.len();

    assert!(matches!(
        u.forward(&rank3),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert!(big_k.forward(&img).is_err());
    assert!(matches!(
        f.forward(&img),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert!(f.forward(&bad_k).is_err());
    assert!(f.forward(&bad_l).is_err());
    assert!(huge.forward(&bad_l).is_err());
    assert_eq!(
        tape.len(),
        before,
        "失敗した forward がテープへノードを残した"
    );
}
