//! `nn::LocalResponseNorm`（イシュー #3043）の受け入れ条件検証。
//!
//! - 層 `forward` ≡ `Var::local_response_norm` 直呼びの forward 値・入力勾配の bit 一致。
//! - 構築時・forward 時の不正引数が型付きエラーで、テープに孤児ノードを残さないこと。
//! - 無状態層（`named_parameters` 空）・`as_*` フック・`supports_forward_host`。
//!
//! 受入条件の「kernel 超過」は LRN には該当しない（`size > C` は PyTorch と同じく受理が正）ため、
//! `size = 0`・非有限パラメータ・rank 不足に読み替える。

mod common;

use fandhe_ai_autodiff::nn::activation::Relu;
use fandhe_ai_autodiff::nn::{Fold, LocalResponseNorm, Module, Unfold};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::{BackendError, Tensor};

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

fn assert_layer_matches_direct(x: &Tensor<f32>, size: usize, alpha: f32, beta: f32, k: f32) {
    let layer = LocalResponseNorm::new(size, alpha, beta, k).unwrap();
    let l = run(x, |tape, v| {
        <LocalResponseNorm as Module>::forward(&layer, tape, v).unwrap()
    });
    let r = run(x, |_, v| {
        v.local_response_norm(size, alpha, beta, k).unwrap()
    });
    assert_eq!(l.1, r.1, "shape size={size}");
    assert_eq!(l.0, r.0, "forward size={size} {alpha} {beta} {k}");
    assert_eq!(l.2, r.2, "grad size={size} {alpha} {beta} {k}");
    let inherent = run(x, |_, v| layer.forward(v).unwrap());
    assert_eq!(inherent, r);
    // 同一層の再 forward は決定的（状態を持たない）。
    let again = run(x, |_, v| layer.forward(v).unwrap());
    assert_eq!(again, r);
}

#[test]
fn layer_matches_var_direct_for_sizes_and_params() {
    let x = seq(2 * 3 * 4 * 5, &[2, 3, 4, 5]);
    // 奇数・偶数（前後非対称）・1・size > C(=3)。
    for size in [1usize, 2, 3, 4, 5, 7] {
        assert_layer_matches_direct(&x, size, 1e-4, 0.75, 1.0);
    }
    assert_layer_matches_direct(&x, 3, 2.0, 0.5, 0.0);
}

#[test]
fn layer_matches_var_direct_for_ranks_and_empty_spatial() {
    assert_layer_matches_direct(&seq(2 * 4 * 6, &[2, 4, 6]), 3, 1e-4, 0.75, 1.0);
    assert_layer_matches_direct(
        &seq(2 * 4 * 3 * 2 * 2, &[2, 4, 3, 2, 2]),
        5,
        1e-3,
        0.75,
        2.0,
    );
    // 空間軸 0 要素は受理され空出力。
    let empty = t(vec![], &[2, 3, 0]);
    let layer = LocalResponseNorm::new(3, 1e-4, 0.75, 1.0).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let out = layer.forward(&tape.var(&empty)).unwrap().to_tensor();
    assert_eq!(out.shape(), &[2, 3, 0]);
}

#[test]
fn nonfinite_inputs_are_bit_identical_to_direct() {
    let mut data: Vec<f32> = seq(2 * 3 * 4, &[2, 3, 4]).as_slice().unwrap().to_vec();
    data[0] = f32::NAN;
    data[5] = f32::INFINITY;
    data[9] = f32::NEG_INFINITY;
    data[13] = -0.0;
    let x = t(data, &[2, 3, 4]);
    let layer = LocalResponseNorm::new(3, 1e-4, 0.75, 1.0).unwrap();
    let l = run(&x, |_, v| layer.forward(v).unwrap());
    let r = run(&x, |_, v| {
        v.local_response_norm(3, 1e-4, 0.75, 1.0).unwrap()
    });
    // NaN は payload を含め同一経路なので bit 一致で比較する。
    assert_eq!(l, r);
}

#[test]
fn layer_is_stateless_and_hooks_are_exact() {
    let l = LocalResponseNorm::new(3, 1e-4, 0.75, 1.0).unwrap();
    assert!(l.named_parameters().is_empty());
    assert_eq!((l.size(), l.alpha(), l.beta(), l.k()), (3, 1e-4, 0.75, 1.0));

    assert!(l.as_local_response_norm().is_some());
    assert!(l.as_unfold().is_none());
    assert!(l.as_fold().is_none());
    assert!(Relu.as_local_response_norm().is_none());
    let u = Unfold::new([2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
    let f = Fold::new([4, 4], [2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
    assert!(u.as_local_response_norm().is_none());
    assert!(f.as_local_response_norm().is_none());

    assert!(!l.is_pooling());
    assert!(!l.supports_forward_host());
    let ops = common::naive_ops();
    assert!(matches!(
        l.forward_host(ops.as_ref(), &seq(2 * 3 * 4, &[2, 3, 4])),
        Err(AutodiffError::Backend(BackendError::Unsupported(_)))
    ));
}

#[test]
fn construction_rejects_invalid_arguments() {
    let is_invalid = |r: Result<LocalResponseNorm, AutodiffError>| {
        matches!(r, Err(AutodiffError::InvalidArgument(_)))
    };
    assert!(is_invalid(LocalResponseNorm::new(0, 1e-4, 0.75, 1.0)));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(is_invalid(LocalResponseNorm::new(3, bad, 0.75, 1.0)));
        assert!(is_invalid(LocalResponseNorm::new(3, 1e-4, bad, 1.0)));
        assert!(is_invalid(LocalResponseNorm::new(3, 1e-4, 0.75, bad)));
    }
    // Var 直呼びと同じエラー種別。
    let tape = Tape::new_with_ops(common::naive_ops());
    let v = tape.var(&seq(2 * 3 * 4, &[2, 3, 4]));
    assert!(matches!(
        v.local_response_norm(0, 1e-4, 0.75, 1.0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        v.local_response_norm(3, f32::NAN, 0.75, 1.0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // k <= 0・負の alpha は契約どおり構築できる。
    assert!(LocalResponseNorm::new(3, -1.0, 0.75, -1.0).is_ok());
}

#[test]
fn forward_rejects_low_rank_without_orphan_nodes() {
    let layer = LocalResponseNorm::new(3, 1e-4, 0.75, 1.0).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    for shape in [vec![4usize], vec![2, 3]] {
        let n = shape.iter().product();
        let v = tape.var(&seq(n, &shape));
        let before = tape.len();
        assert!(matches!(layer.forward(&v), Err(AutodiffError::Shape(_))));
        assert_eq!(tape.len(), before, "孤児ノード無し shape={shape:?}");
    }
}
