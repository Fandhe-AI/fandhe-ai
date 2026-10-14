//! `nn::{MaxUnpool1d, MaxUnpool2d, MaxUnpool3d}`（イシュー #3035）の受け入れ条件検証。
//!
//! - 層 `forward` ≡ `Var::max_unpool{1,2,3}d` 直呼びの forward 値・入力勾配の bit 一致
//!   （`common::naive_ops()` 経由。重複索引・`output_size` 指定あり／なしを含む）。
//! - 構築時・forward 時の不正引数が型付きエラーで、テープに孤児ノードを残さないこと。
//! - 層が無状態で再利用でき、`Clone`／`PartialEq`／`Debug` が機能すること。
//!
//! `Module` は実装しない（索引が第 2 入力のため。`docs/compat-sequential-3d-fold-lrn-layers-decision.md`
//! §3・§12）ので `Module` 経由の検査は無い。

mod common;

use fandhe_ai_autodiff::nn::{MaxUnpool1d, MaxUnpool2d, MaxUnpool3d};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn bits(x: &Tensor<f32>) -> Vec<u32> {
    x.contiguous()
        .as_slice()
        .map(|s| s.iter().map(|v| v.to_bits()).collect())
        .unwrap_or_default()
}

fn wave(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (((i * 37 + 11) % 23) as f32 - 9.0) * 0.5)
        .collect();
    t(data, shape)
}

/// 入力 `y` に `f` を適用し、`(out * g).sum()` の forward 値・shape・入力勾配を返す。
fn run_unpool<F>(y: &Tensor<f32>, f: F) -> (Vec<u32>, Vec<usize>, Vec<u32>)
where
    F: for<'t> Fn(&Var<'t>) -> Result<Var<'t>, AutodiffError>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let yv = tape.var(y);
    let out = f(&yv).expect("test: 有効な引数");
    let o = out.to_tensor();
    let g: Vec<f32> = (0..o.shape().iter().product::<usize>())
        .map(|i| 0.25 + (i % 7) as f32)
        .collect();
    let gv = tape.var_no_grad(&t(g, o.shape()));
    let loss = out.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dy = grads.get(&yv).unwrap().expect("入力へ勾配が届く").clone();
    (bits(&o), o.shape().to_vec(), bits(&dy))
}

fn pool1d(k: usize, s: usize) -> (Tensor<f32>, Tensor<i32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&wave(&[1, 2, 8]));
    let (y, idx) = x.max_pool1d(k, Some(s), 0, 1, false).unwrap();
    (y.to_tensor(), idx)
}

fn pool2d(k: usize, s: usize) -> (Tensor<f32>, Tensor<i32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&wave(&[1, 2, 6, 6]));
    let (y, idx) = x
        .max_pool2d([k, k], Some([s, s]), [0, 0], [1, 1], false)
        .unwrap();
    (y.to_tensor(), idx)
}

fn pool3d(k: usize, s: usize) -> (Tensor<f32>, Tensor<i32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&wave(&[1, 2, 4, 4, 4]));
    let (y, idx) = x
        .max_pool3d([k; 3], Some([s; 3]), [0; 3], [1; 3], false)
        .unwrap();
    (y.to_tensor(), idx)
}

#[test]
fn bit_identical_to_var_methods_1d() {
    // (kernel, stride, output_size): 非重複・output_size 指定・重複索引（stride < kernel）。
    for (k, s, os) in [(2, 2, None), (2, 2, Some(8)), (3, 1, None), (3, 1, Some(8))] {
        let (y, idx) = pool1d(k, s);
        let layer = MaxUnpool1d::new(k, Some(s), 0).unwrap();
        let a = run_unpool(&y, |v| layer.forward(v, &idx, os));
        let b = run_unpool(&y, |v| v.max_unpool1d(&idx, k, Some(s), 0, os));
        assert_eq!(a, b, "k={k} s={s} os={os:?}");
    }
}

#[test]
fn bit_identical_to_var_methods_2d() {
    for (k, s, os) in [(2, 2, None), (2, 2, Some([6, 6])), (3, 1, None)] {
        let (y, idx) = pool2d(k, s);
        let layer = MaxUnpool2d::new([k, k], Some([s, s]), [0, 0]).unwrap();
        let a = run_unpool(&y, |v| layer.forward(v, &idx, os));
        let b = run_unpool(&y, |v| {
            v.max_unpool2d(&idx, [k, k], Some([s, s]), [0, 0], os)
        });
        assert_eq!(a, b, "k={k} s={s} os={os:?}");
    }
}

#[test]
fn bit_identical_to_var_methods_3d() {
    for (k, s, os) in [(2, 2, None), (2, 2, Some([4, 4, 4])), (3, 1, None)] {
        let (y, idx) = pool3d(k, s);
        let layer = MaxUnpool3d::new([k; 3], Some([s; 3]), [0; 3]).unwrap();
        let a = run_unpool(&y, |v| layer.forward(v, &idx, os));
        let b = run_unpool(&y, |v| {
            v.max_unpool3d(&idx, [k; 3], Some([s; 3]), [0; 3], os)
        });
        assert_eq!(a, b, "k={k} s={s} os={os:?}");
    }
}

#[test]
fn stride_none_equals_kernel_size() {
    let (y, idx) = pool2d(2, 2);
    let none = MaxUnpool2d::new([2, 2], None, [0, 0]).unwrap();
    let some = MaxUnpool2d::new([2, 2], Some([2, 2]), [0, 0]).unwrap();
    assert_eq!(none, some);
    assert_eq!(none.stride(), [2, 2]);
    let a = run_unpool(&y, |v| none.forward(v, &idx, None));
    let b = run_unpool(&y, |v| v.max_unpool2d(&idx, [2, 2], None, [0, 0], None));
    assert_eq!(a, b);
}

#[test]
fn padding_layer_matches_direct_call() {
    // padding 1 の pool 由来の索引を padding 1 の層で戻す。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&wave(&[1, 2, 8]));
    let (y, idx) = x.max_pool1d(3, Some(2), 1, 1, false).unwrap();
    let y = y.to_tensor();
    let layer = MaxUnpool1d::new(3, Some(2), 1).unwrap();
    // 既定長は (4-1)*2-2+3 = 7。元入力長 8 は default ± stride の開区間内なので output_size で戻す。
    let a = run_unpool(&y, |v| layer.forward(v, &idx, Some(8)));
    let b = run_unpool(&y, |v| v.max_unpool1d(&idx, 3, Some(2), 1, Some(8)));
    assert_eq!(a, b);
}

#[test]
fn pool_unpool_round_trip_places_winners() {
    let (y, idx) = pool2d(2, 2);
    let tape = Tape::new_with_ops(common::naive_ops());
    let yv = tape.var(&y);
    let layer = MaxUnpool2d::new([2, 2], None, [0, 0]).unwrap();
    let out = layer.forward(&yv, &idx, None).unwrap().to_tensor();
    assert_eq!(out.shape(), &[1, 2, 6, 6]);
    let (o, ys, ix) = (out.host_slice(), y.host_slice(), idx.host_slice());
    // 非重複窓: 勝者位置は pool 出力値、それ以外は 0。
    for (p, (&v, &i)) in ys.iter().zip(ix.iter()).enumerate() {
        let plane = p / 9;
        assert_eq!(o[plane * 36 + i as usize].to_bits(), v.to_bits());
    }
    assert_eq!(
        o.iter().filter(|v| **v != 0.0).count(),
        ys.iter().filter(|v| **v != 0.0).count()
    );
}

#[test]
fn new_rejects_invalid_arguments() {
    let bad = |r: Result<_, AutodiffError>| matches!(r, Err(AutodiffError::InvalidArgument(_)));
    // 1d
    assert!(bad(MaxUnpool1d::new(0, None, 0).map(|_| ())));
    assert!(bad(MaxUnpool1d::new(2, Some(0), 0).map(|_| ())));
    assert!(bad(MaxUnpool1d::new(2, None, usize::MAX).map(|_| ())));
    // 2d（1 軸だけ 0）
    assert!(bad(MaxUnpool2d::new([2, 0], None, [0, 0]).map(|_| ())));
    assert!(bad(
        MaxUnpool2d::new([2, 2], Some([1, 0]), [0, 0]).map(|_| ())
    ));
    assert!(bad(
        MaxUnpool2d::new([2, 2], None, [0, usize::MAX]).map(|_| ())
    ));
    // 3d
    assert!(bad(MaxUnpool3d::new([0, 2, 2], None, [0; 3]).map(|_| ())));
    assert!(bad(
        MaxUnpool3d::new([2; 3], Some([1, 1, 0]), [0; 3]).map(|_| ())
    ));
    assert!(bad(
        MaxUnpool3d::new([2; 3], None, [usize::MAX, 0, 0]).map(|_| ())
    ));
    // 正常系
    assert!(MaxUnpool1d::new(2, None, 0).is_ok());
    assert!(MaxUnpool3d::new([2; 3], Some([1; 3]), [1; 3]).is_ok());
}

#[test]
fn forward_rejects_invalid_arguments_without_orphan_nodes() {
    let (y, idx) = pool1d(2, 2); // y/idx: [1, 2, 4]
    let tape = Tape::new_with_ops(common::naive_ops());
    let yv = tape.var(&y);
    let layer = MaxUnpool1d::new(2, None, 0).unwrap();
    let is_shape = |r: Result<Var<'_>, AutodiffError>| matches!(r, Err(AutodiffError::Shape(_)));
    let is_invalid =
        |r: Result<Var<'_>, AutodiffError>| matches!(r, Err(AutodiffError::InvalidArgument(_)));
    let y2 = tape.var(&wave(&[2, 4]));
    let before = tape.len();

    // indices の shape 不一致
    let bad_idx = ti(vec![0; 6], &[1, 2, 3]);
    assert!(is_shape(layer.forward(&yv, &bad_idx, None)));
    // 入力 rank 違い
    assert!(is_shape(layer.forward(&y2, &ti(vec![0; 8], &[2, 4]), None)));
    // output_size が既定長 8 ± stride(2) の開区間外・0
    assert!(is_shape(layer.forward(&yv, &idx, Some(6))));
    assert!(is_shape(layer.forward(&yv, &idx, Some(10))));
    assert!(is_shape(layer.forward(&yv, &idx, Some(0))));
    // padding が大きすぎて既定出力長が 1 未満
    let big_pad = MaxUnpool1d::new(2, None, 5).unwrap();
    assert!(is_shape(big_pad.forward(&yv, &idx, None)));
    // 索引の負値・出力平面長以上
    let neg = ti(vec![-1, 0, 0, 0, 0, 0, 0, 0], &[1, 2, 4]);
    assert!(is_invalid(layer.forward(&yv, &neg, None)));
    let over = ti(vec![8, 0, 0, 0, 0, 0, 0, 0], &[1, 2, 4]);
    assert!(is_invalid(layer.forward(&yv, &over, None)));

    assert_eq!(tape.len(), before, "拒否時にノードを残さない");
}

#[test]
fn layer_is_stateless_and_reusable() {
    let layer = MaxUnpool2d::new([2, 2], None, [0, 0]).unwrap();
    let (y, idx) = pool2d(2, 2);
    let a = run_unpool(&y, |v| layer.forward(v, &idx, None));
    let b = run_unpool(&y, |v| layer.forward(v, &idx, None));
    assert_eq!(a, b);
    // 別の output_size（既定 6 に対し 7）でも同一インスタンスを使える。
    let c = run_unpool(&y, |v| layer.forward(v, &idx, Some([7, 7])));
    assert_eq!(c.1, vec![1, 2, 7, 7]);
    let d = run_unpool(&y, |v| layer.forward(v, &idx, None));
    assert_eq!(a, d);
}

#[test]
fn clone_eq_debug() {
    let a = MaxUnpool3d::new([2; 3], None, [0; 3]).unwrap();
    assert_eq!(a, a.clone());
    assert_ne!(a, MaxUnpool3d::new([2; 3], Some([1; 3]), [0; 3]).unwrap());
    assert!(format!("{a:?}").contains("MaxUnpool3d"));
    assert_eq!(a.kernel_size(), [2; 3]);
    assert_eq!(a.padding(), [0; 3]);
    let b = MaxUnpool1d::new(3, None, 1).unwrap();
    assert_eq!((b.kernel_size(), b.stride(), b.padding()), (3, 3, 1));
}
