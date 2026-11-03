//! `nn::{AdaptiveAvgPool3d, AdaptiveMaxPool3d}`（イシュー #3824・設計 `docs/autodiff-pool3d-ops-decision.md`
//! §18）の受け入れ条件検証。`nn_pooling3d.rs` と同型。
//!
//! - `Module::forward`（層経由）≡ `Var::adaptive_*_pool3d` 直呼びの forward・索引・入力勾配の bit 一致。
//! - 構築時の 0 軸と forward 時の rank 違反が型付きエラーで、テープに孤児ノードを残さないこと。
//! - 無状態層・`is_pooling`・`supports_forward_host`。

mod common;

use fandhe_ai_autodiff::nn::{AdaptiveAvgPool3d, AdaptiveMaxPool3d, Module};
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

/// 非自明（負値・重複値を含む）な `[1, 2, 5, 4, 6]`。
fn input() -> Tensor<f32> {
    let data: Vec<f32> = (0..240)
        .map(|i| (((i * 37 + 11) % 23) as f32 - 9.0) * 0.5)
        .collect();
    t(data, &[1, 2, 5, 4, 6])
}

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

/// 割り切れる・重なる・出力 1・入力と同形の各ケース。
const CASES: [[usize; 3]; 4] = [[1, 1, 1], [2, 2, 3], [3, 3, 4], [5, 4, 6]];

#[test]
fn adaptive_avg_pool3d_layer_matches_var_direct_forward_and_grad() {
    for os in CASES {
        let layer = AdaptiveAvgPool3d::new(os).unwrap();
        let l = run(|tape, x| <AdaptiveAvgPool3d as Module>::forward(&layer, tape, x).unwrap());
        let r = run(|_, x| x.adaptive_avg_pool3d(os).unwrap());
        assert_eq!(l.1, r.1);
        assert_eq!(l.0, r.0, "adaptive_avg_pool3d forward {os:?}");
        assert_eq!(l.2, r.2, "adaptive_avg_pool3d grad {os:?}");
    }
}

#[test]
fn adaptive_max_pool3d_layer_matches_var_direct_forward_index_and_grad() {
    for os in CASES {
        let layer = AdaptiveMaxPool3d::new(os).unwrap();
        let l = run(|tape, x| <AdaptiveMaxPool3d as Module>::forward(&layer, tape, x).unwrap());
        let r = run(|_, x| x.adaptive_max_pool3d(os).unwrap().0);
        assert_eq!(l.1, r.1);
        assert_eq!(l.0, r.0, "adaptive_max_pool3d forward {os:?}");
        assert_eq!(l.2, r.2, "adaptive_max_pool3d grad {os:?}");

        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&input());
        let (_, i_l) = layer.forward(&xv).unwrap();
        let (_, i_d) = xv.adaptive_max_pool3d(os).unwrap();
        assert_eq!(i_l.shape(), i_d.shape());
        assert_eq!(
            i_l.contiguous().as_slice().unwrap(),
            i_d.contiguous().as_slice().unwrap()
        );
    }
}

#[test]
fn layers_are_stateless_and_declare_pooling_without_host_path() {
    let av = AdaptiveAvgPool3d::new([1, 2, 2]).unwrap();
    let mx = AdaptiveMaxPool3d::new([1, 2, 2]).unwrap();
    assert!(av.named_parameters().is_empty());
    assert!(mx.named_parameters().is_empty());
    assert!(av.is_pooling());
    assert!(mx.is_pooling());
    assert!(!av.supports_forward_host());
    assert!(!mx.supports_forward_host());

    let ops = common::naive_ops();
    for r in [
        av.forward_host(ops.as_ref(), &input()),
        mx.forward_host(ops.as_ref(), &input()),
    ] {
        assert!(matches!(
            r,
            Err(AutodiffError::Backend(BackendError::Unsupported(_)))
        ));
    }
}

#[test]
fn constructors_reject_zero_axes() {
    for bad in [[0, 1, 1], [1, 0, 1], [1, 1, 0]] {
        assert!(matches!(
            AdaptiveAvgPool3d::new(bad),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            AdaptiveMaxPool3d::new(bad),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
}

#[test]
fn forward_rejects_bad_rank_without_orphan_nodes() {
    let av = AdaptiveAvgPool3d::new([1, 1, 1]).unwrap();
    let mx = AdaptiveMaxPool3d::new([1, 1, 1]).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let rank4 = tape.var(&t(vec![0.0; 16], &[1, 2, 2, 4]));
    let before = tape.len();
    assert!(matches!(
        av.forward(&rank4),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert!(matches!(
        mx.forward(&rank4),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert_eq!(
        tape.len(),
        before,
        "失敗した forward がテープへノードを残した"
    );
}
