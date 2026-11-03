//! `nn::{FractionalMaxPool2d, FractionalMaxPool3d}`（イシュー #3827・設計
//! `docs/autodiff-pool3d-ops-decision.md` §20・§25）の受け入れ条件検証。`nn_adaptive_pooling3d.rs` と同型に、
//! 乱数契約（§20.12）の検査を上乗せする。
//!
//! - `manual_seed(S)` 直後の層 forward ≡ `manual_seed(S)` 直後の `Var::fractional_max_pool*` 直呼びの
//!   値・索引・入力勾配の bit 一致（alpha が整数にならない形状を含む）。
//! - 構築時の 0 軸と forward 時の rank 違反・実行不能が型付きエラーで、RNG を消費しないこと。
//! - 無状態層・`is_pooling`・`supports_forward_host`。
//!
//! グローバル RNG を共有するため全テストを `LOCK` で直列化する。

mod common;

use std::sync::{Mutex, MutexGuard};

use fandhe_ai_autodiff::nn::{FractionalMaxPool2d, FractionalMaxPool3d, Module};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var, manual_seed, rand};
use fandhe_ai_tensor_core::{BackendError, Tensor};

static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const SEED: u64 = 0x3827_0A01;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn bits(x: &Tensor<f32>) -> Vec<u32> {
    x.contiguous()
        .as_slice()
        .map(|s| s.iter().map(|v| v.to_bits()).collect())
        .unwrap_or_default()
}

fn input2() -> Tensor<f32> {
    let data: Vec<f32> = (0..2 * 3 * 7 * 7)
        .map(|i| (((i * 37 + 11) % 23) as f32 - 9.0) * 0.5)
        .collect();
    t(data, &[2, 3, 7, 7])
}

fn input3() -> Tensor<f32> {
    let data: Vec<f32> = (0..2 * 6 * 6 * 6)
        .map(|i| (((i * 37 + 11) % 23) as f32 - 9.0) * 0.5)
        .collect();
    t(data, &[1, 2, 6, 6, 6])
}

fn run<F>(x: &Tensor<f32>, f: F) -> (Vec<u32>, Vec<usize>, Vec<u32>)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Var<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    manual_seed(SEED);
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

/// (kernel, output)。alpha = (in - k) / (out - 1) が整数にならないケースを含む（in = 7, k = 2）。
const CASES2: [([usize; 2], [usize; 2]); 3] =
    [([2, 2], [1, 1]), ([2, 2], [4, 4]), ([2, 3], [3, 5])];
const CASES3: [([usize; 3], [usize; 3]); 3] = [
    ([2, 2, 2], [1, 1, 1]),
    ([2, 2, 2], [4, 3, 4]),
    ([1, 2, 3], [5, 3, 2]),
];

#[test]
fn fractional_max_pool2d_layer_matches_var_direct_value_index_and_grad() {
    let _g = lock();
    let x = input2();
    for (k, o) in CASES2 {
        let layer = FractionalMaxPool2d::new(k, o).unwrap();
        let l = run(&x, |tape, v| {
            <FractionalMaxPool2d as Module>::forward(&layer, tape, v).unwrap()
        });
        let r = run(&x, |_, v| v.fractional_max_pool2d(k, o).unwrap().0);
        assert_eq!(l, r, "2d {k:?} {o:?}");

        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        manual_seed(SEED);
        let (_, i_l) = layer.forward(&xv).unwrap();
        manual_seed(SEED);
        let (_, i_d) = xv.fractional_max_pool2d(k, o).unwrap();
        assert_eq!(
            i_l.contiguous().as_slice().unwrap(),
            i_d.contiguous().as_slice().unwrap()
        );
    }
}

#[test]
fn fractional_max_pool3d_layer_matches_var_direct_value_index_and_grad() {
    let _g = lock();
    let x = input3();
    for (k, o) in CASES3 {
        let layer = FractionalMaxPool3d::new(k, o).unwrap();
        let l = run(&x, |tape, v| {
            <FractionalMaxPool3d as Module>::forward(&layer, tape, v).unwrap()
        });
        let r = run(&x, |_, v| v.fractional_max_pool3d(k, o).unwrap().0);
        assert_eq!(l, r, "3d {k:?} {o:?}");

        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        manual_seed(SEED);
        let (_, i_l) = layer.forward(&xv).unwrap();
        manual_seed(SEED);
        let (_, i_d) = xv.fractional_max_pool3d(k, o).unwrap();
        assert_eq!(
            i_l.contiguous().as_slice().unwrap(),
            i_d.contiguous().as_slice().unwrap()
        );
    }
}

#[test]
fn new_rejects_each_zero_axis() {
    let _g = lock();
    for i in 0..2 {
        let (mut k, mut o) = ([2usize; 2], [1usize; 2]);
        k[i] = 0;
        assert!(matches!(
            FractionalMaxPool2d::new(k, [1, 1]),
            Err(AutodiffError::InvalidArgument(_))
        ));
        o[i] = 0;
        assert!(matches!(
            FractionalMaxPool2d::new([2, 2], o),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
    for i in 0..3 {
        let (mut k, mut o) = ([2usize; 3], [1usize; 3]);
        k[i] = 0;
        assert!(matches!(
            FractionalMaxPool3d::new(k, [1; 3]),
            Err(AutodiffError::InvalidArgument(_))
        ));
        o[i] = 0;
        assert!(matches!(
            FractionalMaxPool3d::new([2; 3], o),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
}

#[test]
fn rank_violation_and_infeasible_shape_err_without_orphans_or_rng_use() {
    let _g = lock();
    let l2 = FractionalMaxPool2d::new([2, 2], [4, 4]).unwrap();
    let too_big = FractionalMaxPool2d::new([2, 2], [7, 7]).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let x3 = tape.var(&input3());
    let x2 = tape.var(&input2());
    for r in [
        l2.forward(&x3).map(|_| ()),
        too_big.forward(&x2).map(|_| ()),
    ] {
        assert!(r.is_err());
    }
    manual_seed(SEED);
    let _ = l2.forward(&x3);
    let after_err = rand(&[1]).unwrap().contiguous().as_slice().unwrap()[0];
    manual_seed(SEED);
    let fresh = rand(&[1]).unwrap().contiguous().as_slice().unwrap()[0];
    assert_eq!(after_err, fresh);
}

#[test]
fn layers_are_stateless_and_declare_pooling_without_host_path() {
    let _g = lock();
    let f2 = FractionalMaxPool2d::new([2, 2], [3, 3]).unwrap();
    let f3 = FractionalMaxPool3d::new([2, 2, 2], [3, 3, 3]).unwrap();
    assert!(f2.named_parameters().is_empty());
    assert!(f3.named_parameters().is_empty());
    assert!(f2.is_pooling());
    assert!(f3.is_pooling());
    assert!(!f2.supports_forward_host());
    assert!(!f3.supports_forward_host());

    let ops = common::naive_ops();
    for r in [
        f2.forward_host(ops.as_ref(), &input2()),
        f3.forward_host(ops.as_ref(), &input3()),
    ] {
        assert!(matches!(
            r,
            Err(AutodiffError::Backend(BackendError::Unsupported(_)))
        ));
    }
}
