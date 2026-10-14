//! `nn::{MaxPool3d, AvgPool3d}`（イシュー #3027）の受け入れ条件検証。
//!
//! - `Module::forward`（層経由）≡ `Var::max_pool3d`／`avg_pool3d` 直呼びの forward・索引・
//!   入力勾配の bit 一致（`common::naive_ops()` 経由）。
//! - 構築時・forward 時の不正引数が型付きエラーで、テープに孤児ノードを残さないこと。
//! - 無状態層（`named_parameters` 空）・`as_*` フック・`is_pooling`・`supports_forward_host`。

mod common;

use fandhe_ai_autodiff::nn::activation::Relu;
use fandhe_ai_autodiff::nn::{AvgPool3d, MaxPool3d, Module};
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

fn input() -> Tensor<f32> {
    // 非自明（負値・重複値を含む）な [1, 2, 4, 4, 4]。
    let data: Vec<f32> = (0..128)
        .map(|i| (((i * 37 + 11) % 23) as f32 - 9.0) * 0.5)
        .collect();
    t(data, &[1, 2, 4, 4, 4])
}

/// `(out * g).sum()` の forward 値・shape・入力勾配を返す。`f` が層経由／直呼びの forward。
fn run<F>(f: F) -> (Vec<u32>, Vec<usize>, Vec<u32>)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Var<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&input());
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

type MaxCase = ([usize; 3], Option<[usize; 3]>, [usize; 3], [usize; 3]);

const MAX_CASES: [MaxCase; 3] = [
    ([2, 2, 2], None, [0, 0, 0], [1, 1, 1]),
    ([3, 3, 3], Some([1, 1, 1]), [1, 1, 1], [1, 1, 1]),
    ([2, 2, 2], Some([1, 2, 1]), [1, 0, 1], [2, 1, 2]),
];

#[test]
fn max_pool3d_layer_matches_var_direct_forward_index_and_grad() {
    for (k, s, p, d) in MAX_CASES {
        let st = s.unwrap_or(k);
        let layer = MaxPool3d::new(k, s, p, d).unwrap();
        let l = run(|tape, x| <MaxPool3d as Module>::forward(&layer, tape, x).unwrap());
        let r = run(|_, x| x.max_pool3d(k, Some(st), p, d, false).unwrap().0);
        assert_eq!(l.1, r.1);
        assert_eq!(l.0, r.0, "max_pool3d forward {k:?} {s:?} {p:?} {d:?}");
        assert_eq!(l.2, r.2, "max_pool3d grad {k:?} {s:?} {p:?} {d:?}");

        // 固有 forward の索引が直呼びと完全一致する。
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&input());
        let (_, i_l) = layer.forward(&xv).unwrap();
        let (_, i_d) = xv.max_pool3d(k, Some(st), p, d, false).unwrap();
        assert_eq!(i_l.shape(), i_d.shape());
        assert_eq!(
            i_l.contiguous().as_slice().unwrap(),
            i_d.contiguous().as_slice().unwrap()
        );
    }
}

#[test]
fn avg_pool3d_layer_matches_var_direct_for_both_count_include_pad() {
    for cip in [true, false] {
        for (k, s, p) in [
            ([2, 2, 2], None, [0, 0, 0]),
            ([3, 3, 3], Some([1, 1, 1]), [1, 1, 1]),
            ([2, 2, 2], Some([2, 1, 2]), [1, 0, 1]),
        ] {
            let st = s.unwrap_or(k);
            let layer = AvgPool3d::new(k, s, p, cip).unwrap();
            let l = run(|tape, x| <AvgPool3d as Module>::forward(&layer, tape, x).unwrap());
            let r = run(|_, x| x.avg_pool3d(k, Some(st), p, false, cip).unwrap());
            assert_eq!(l.1, r.1);
            assert_eq!(l.0, r.0, "avg_pool3d forward cip={cip} {k:?} {s:?} {p:?}");
            assert_eq!(l.2, r.2, "avg_pool3d grad cip={cip} {k:?} {s:?} {p:?}");
        }
    }
}

#[test]
fn stride_none_equals_kernel_size() {
    let layer = MaxPool3d::new([2, 2, 2], None, [0, 0, 0], [1, 1, 1]).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&input());
    let y = <MaxPool3d as Module>::forward(&layer, &tape, &xv).unwrap();
    assert_eq!(y.to_tensor().shape(), &[1, 2, 2, 2, 2]);
}

#[test]
fn layers_are_stateless_and_hooks_are_exact() {
    let mx = MaxPool3d::new([2, 2, 2], None, [0, 0, 0], [1, 1, 1]).unwrap();
    let av = AvgPool3d::new([2, 2, 2], None, [0, 0, 0], true).unwrap();
    assert!(mx.named_parameters().is_empty());
    assert!(av.named_parameters().is_empty());

    assert!(mx.as_max_pool3d().is_some());
    assert!(mx.as_avg_pool3d().is_none());
    assert!(av.as_avg_pool3d().is_some());
    assert!(av.as_max_pool3d().is_none());
    assert!(Relu.as_max_pool3d().is_none());
    assert!(Relu.as_avg_pool3d().is_none());

    assert!(mx.is_pooling());
    assert!(av.is_pooling());
    assert!(!mx.supports_forward_host());
    assert!(!av.supports_forward_host());

    let ops = common::naive_ops();
    for r in [
        mx.forward_host(ops.as_ref(), &input()),
        av.forward_host(ops.as_ref(), &input()),
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
    let one = [1, 1, 1];
    let z = [0, 0, 0];
    // 0 次元（kernel／stride／dilation）。
    assert!(is_invalid_arg(MaxPool3d::new([2, 0, 2], None, z, one)));
    assert!(is_invalid_arg(MaxPool3d::new(
        [2, 2, 2],
        Some([1, 0, 1]),
        z,
        one
    )));
    assert!(is_invalid_arg(MaxPool3d::new(
        [2, 2, 2],
        None,
        z,
        [1, 1, 0]
    )));
    assert!(is_invalid_arg(AvgPool3d::new([0, 2, 2], None, z, true)));
    assert!(is_invalid_arg(AvgPool3d::new(
        [2, 2, 2],
        Some([0, 1, 1]),
        z,
        true
    )));
    // padding > kernel/2。
    assert!(is_invalid_arg(MaxPool3d::new(
        [2, 2, 2],
        None,
        [2, 0, 0],
        one
    )));
    assert!(is_invalid_arg(AvgPool3d::new(
        [3, 3, 3],
        None,
        [0, 2, 0],
        false
    )));
    // 2 * padding のオーバーフロー。
    assert!(is_invalid_arg(MaxPool3d::new(
        [2, 2, 2],
        None,
        [usize::MAX, 0, 0],
        one
    )));
    assert!(is_invalid_arg(AvgPool3d::new(
        [2, 2, 2],
        None,
        [0, usize::MAX / 2 + 1, 0],
        true
    )));
}

#[test]
fn forward_rejects_bad_shapes_without_orphan_nodes() {
    let mx = MaxPool3d::new([2, 2, 2], None, [0, 0, 0], [1, 1, 1]).unwrap();
    let av = AvgPool3d::new([2, 2, 2], None, [0, 0, 0], true).unwrap();
    let big_k = MaxPool3d::new([5, 5, 5], None, [0, 0, 0], [1, 1, 1]).unwrap();
    let big_k_avg = AvgPool3d::new([5, 5, 5], None, [0, 0, 0], true).unwrap();
    let dil = MaxPool3d::new([2, 2, 2], None, [0, 0, 0], [4, 4, 4]).unwrap();

    let tape = Tape::new_with_ops(common::naive_ops());
    let rank4 = tape.var(&t(vec![0.0; 16], &[1, 2, 2, 4]));
    let empty = tape.var(&t(vec![], &[1, 1, 0, 2, 2]));
    let small = tape.var(&input());
    let before = tape.len();

    assert!(matches!(
        mx.forward(&rank4),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert!(matches!(
        av.forward(&rank4),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert!(mx.forward(&empty).is_err());
    assert!(av.forward(&empty).is_err());
    assert!(big_k.forward(&small).is_err());
    assert!(big_k_avg.forward(&small).is_err());
    assert!(dil.forward(&small).is_err());
    assert_eq!(
        tape.len(),
        before,
        "失敗した forward がテープへノードを残した"
    );
}
