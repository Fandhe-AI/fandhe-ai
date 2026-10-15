//! `functional_ops::vmap_grad`（イシュー #3071・設計記録 §32）の autodiff 内部単体テスト。
//!
//! 役割: 子テープで勾配を取るクロージャ型 `vmap` が、サンプルごとに新しいテープで `backward` した
//! 明示ループ参照と REQ-2 統一複合判定（`common::req2_close`）で一致すること、入口検査の順序、
//! 失敗後に子テープが空・入口の葉プレフィックス状態へ戻ることを固定する。
//!
//! 扱わないもの: facade 委譲（#3072）・facade テストと実機 `#[ignore]`（#3073）・微分可能版。
//! 結果要素数のオーバーフロー（Phase A の 3）は実在の `Tensor` では作れないためテストしない。
//! 世代の進み方は `Tape::epoch` が `pub(crate)` のため結合テストでは観測しない。

mod common;

use fandhe_ai_autodiff::functional_ops::vmap_grad as run_grad_map;
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

type R<'c> = Result<Var<'c>, AutodiffError>;

fn new_tape() -> Tape {
    Tape::new_with_ops(common::naive_ops())
}

fn tn(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test: shape とデータ長は一致させている")
}

fn host(x: &Tensor<f32>) -> Vec<f32> {
    x.host_slice().into_owned()
}

fn seq(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.37).sin() * 1.5)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: 要素数");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            common::req2_close(f64::from(a), f64::from(e)),
            "{ctx}[{i}]: {a} vs {e}"
        );
    }
}

fn weight() -> Tensor<f32> {
    tn(seq(6, 3.0), &[3, 2])
}

fn loss_sq<'c>(_t: &'c Tape, s: &Var<'c>) -> R<'c> {
    s.mul(s)?.sum(None)
}

fn loss_tanh<'c>(_t: &'c Tape, s: &Var<'c>) -> R<'c> {
    s.tanh().mul(s)?.sum(None)
}

/// 要素数 3 のスライスを `[1, 3] @ [3, 2]` で射影して総和する。
fn loss_matmul<'c>(t: &'c Tape, s: &Var<'c>) -> R<'c> {
    let w = t.var_no_grad(&weight());
    let flat = s.reshape(&[1, 3])?;
    flat.matmul(&w)?.tanh().sum(None)
}

type LossFn = for<'c> fn(&'c Tape, &Var<'c>) -> R<'c>;

fn slice_shape_of(shape: &[usize], in_dim: usize) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .filter(|&(a, _)| a != in_dim)
        .map(|(_, &d)| d)
        .collect()
}

/// 参照: スライスごとに新しいテープで backward して dim 0 に並べる。
fn reference(input: &Tensor<f32>, in_dim: usize, f: LossFn) -> Vec<f32> {
    let b = input.shape()[in_dim];
    let slice_shape = slice_shape_of(input.shape(), in_dim);
    let mut out = Vec::new();
    for i in 0..b {
        let sl = input
            .narrow(in_dim, i, 1)
            .unwrap()
            .contiguous()
            .reshape(&slice_shape)
            .unwrap();
        let tape = new_tape();
        let s = tape.var(&sl);
        let loss = f(&tape, &s).unwrap();
        let g = tape.backward(&loss).unwrap();
        out.extend(host(g.get(&s).unwrap().unwrap()));
    }
    out
}

#[test]
fn matches_loop_reference_across_losses_and_dims() {
    let cases: [(&str, LossFn); 3] = [
        ("sq", loss_sq),
        ("tanh", loss_tanh),
        ("matmul", loss_matmul),
    ];
    let shapes: [(&[usize], usize); 5] = [
        (&[1, 3], 0),
        (&[4, 3], 0),
        (&[3, 4], 1),
        (&[2, 3, 1], 0),
        (&[3, 2, 1], 1),
    ];
    for (name, f) in cases {
        for (shape, in_dim) in shapes {
            let n: usize = shape.iter().product();
            let x = tn(seq(n, 1.0), shape);
            let mut child = new_tape();
            let got = run_grad_map(&x, in_dim, &mut child, f).unwrap();
            let mut exp_shape = vec![shape[in_dim]];
            exp_shape.extend(slice_shape_of(shape, in_dim));
            assert_eq!(got.shape(), exp_shape, "{name} {shape:?} {in_dim}: shape");
            assert_close(
                &host(&got),
                &reference(&x, in_dim, f),
                &format!("{name} {shape:?} in_dim={in_dim}"),
            );
            assert!(child.is_empty(), "{name}: 戻り時の子テープは空");
        }
    }
}

#[test]
fn sum_of_squares_matches_closed_form() {
    let x = tn(seq(8, 2.0), &[2, 4]);
    let mut child = new_tape();
    let got = run_grad_map(&x, 0, &mut child, loss_sq).unwrap();
    let exp: Vec<f32> = host(&x).iter().map(|v| 2.0 * v).collect();
    assert_close(&host(&got), &exp, "2x");
}

#[test]
fn rank0_slice_and_loss_shapes_accepted() {
    let x = tn(vec![1.0, -2.0, 3.0], &[3]);
    let mut child = new_tape();
    let got = run_grad_map(&x, 0, &mut child, |_t, s| s.mul(s)).unwrap();
    assert_eq!(got.shape(), vec![3]);
    assert_close(&host(&got), &[2.0, -4.0, 6.0], "rank0");
    let got = run_grad_map(&x, 0, &mut child, |_t, s| {
        let r = s.reshape(&[1])?;
        r.mul(&r)
    })
    .unwrap();
    assert_close(&host(&got), &[2.0, -4.0, 6.0], "[1]");
    let got = run_grad_map(&x, 0, &mut child, |_t, s| {
        let r = s.reshape(&[1, 1])?;
        r.mul(&r)
    })
    .unwrap();
    assert_close(&host(&got), &[2.0, -4.0, 6.0], "[1,1]");
}

#[test]
fn phase_a_errors_leave_child_untouched() {
    let x = tn(seq(6, 0.0), &[2, 3]);
    let mut child = new_tape();
    let sc = tn(vec![1.0], &[]);
    assert!(matches!(
        run_grad_map(&sc, 0, &mut child, loss_sq),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
    ));
    for d in [2, 5] {
        assert!(matches!(
            run_grad_map(&x, d, &mut child, loss_sq),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
        ));
    }
    let empty = tn(vec![], &[0, 3]);
    assert!(matches!(
        run_grad_map(&empty, 0, &mut child, loss_sq),
        Err(AutodiffError::InvalidArgument(_))
    ));
    let _leaf = child.var(&tn(vec![1.0], &[1]));
    let before = child.len();
    assert!(matches!(
        run_grad_map(&x, 0, &mut child, loss_sq),
        Err(AutodiffError::Backward(_))
    ));
    // 順序: 軸範囲外と空バッチは子テープ非空検査より先。
    assert!(matches!(
        run_grad_map(&x, 9, &mut child, loss_sq),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
    ));
    assert!(matches!(
        run_grad_map(&empty, 0, &mut child, loss_sq),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert_eq!(child.len(), before);
}

#[test]
fn phase_b_errors_and_recovery() {
    let x = tn(seq(9, 0.5), &[3, 3]);
    let mut child = new_tape();
    let ok_ref = reference(&x, 0, loss_sq);

    assert!(matches!(
        run_grad_map(&x, 0, &mut child, |_t, s| Ok(s.tanh())),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(child.is_empty());
    assert_close(
        &host(&run_grad_map(&x, 0, &mut child, loss_sq).unwrap()),
        &ok_ref,
        "after B7",
    );

    // 別テープの非スカラー損失は InvalidArgument ではなく TapeMismatch（B6 が B7 より先）。
    let other: &'static Tape = Box::leak(Box::new(new_tape()));
    let ov: Var<'static> = other.var(&tn(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        run_grad_map(&x, 0, &mut child, |_t, _s| Ok(ov)),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(child.is_empty());
    assert_close(
        &host(&run_grad_map(&x, 0, &mut child, loss_sq).unwrap()),
        &ok_ref,
        "after B6",
    );

    let r = run_grad_map(&x, 0, &mut child, |t, s| {
        t.register_backward_hook(s, |_| Err(AutodiffError::Backward("hook".into())))?;
        s.mul(s)?.sum(None)
    });
    assert!(matches!(r, Err(AutodiffError::Backward(_))));
    assert!(child.is_empty());
    assert_close(
        &host(&run_grad_map(&x, 0, &mut child, loss_sq).unwrap()),
        &ok_ref,
        "after B9",
    );
}

#[test]
fn closure_error_midway_restores_child_and_calls_in_order() {
    let x = tn(seq(9, 0.5), &[3, 3]);
    let mut child = new_tape();
    let mut calls = 0usize;
    let r = run_grad_map(&x, 0, &mut child, |t, s| {
        calls += 1;
        if calls == 2 {
            return Err(AutodiffError::InvalidArgument("boom".into()));
        }
        loss_sq(t, s)
    });
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(m)) if m == "boom"));
    assert_eq!(calls, 2, "失敗サンプルで打ち切る");
    assert!(child.is_empty());
    assert_eq!(child.leaf_count(), 0);
    let mut n = 0usize;
    let got = run_grad_map(&x, 0, &mut child, |t, s| {
        n += 1;
        loss_sq(t, s)
    })
    .unwrap();
    assert_eq!(n, 3, "サンプル数ちょうど");
    assert_close(&host(&got), &reference(&x, 0, loss_sq), "retry");
}

#[test]
fn leaf_prefix_is_restored_after_success() {
    let x = tn(seq(6, 0.0), &[3, 2]);
    let mut child = new_tape();
    run_grad_map(&x, 0, &mut child, loss_sq).unwrap();
    let a = child.var(&tn(vec![1.0], &[1]));
    let b = child.var(&tn(vec![2.0], &[1]));
    let _c = a.mul(&b).unwrap();
    child.reset();
    assert_eq!(child.len(), 2);
    assert!(child.leaf(0).is_some() && child.leaf(1).is_some());
}

#[test]
fn zero_rows_for_untracked_or_independent_loss() {
    let x = tn(seq(6, 0.0), &[3, 2]);
    let mut child = new_tape();
    let got = run_grad_map(&x, 0, &mut child, |t, _s| {
        t.var_no_grad(&tn(vec![1.0], &[1])).sum(None)
    })
    .unwrap();
    assert_eq!(host(&got), vec![0.0; 6]);
    let mut i = 0usize;
    let got = run_grad_map(&x, 0, &mut child, |t, s| {
        i += 1;
        if i == 2 {
            let o = t.var(&tn(vec![1.0, 2.0], &[2]));
            o.mul(&o)?.sum(None)
        } else {
            loss_sq(t, s)
        }
    })
    .unwrap();
    let exp = reference(&x, 0, loss_sq);
    let h = host(&got);
    assert_close(&h[0..2], &exp[0..2], "row0");
    assert_eq!(&h[2..4], &[0.0, 0.0]);
    assert_close(&h[4..6], &exp[4..6], "row2");
}

#[test]
fn zero_numel_slice_does_not_panic() {
    let x = tn(vec![], &[3, 0]);
    let mut child = new_tape();
    let got = run_grad_map(&x, 0, &mut child, |t, _s| {
        t.var_no_grad(&tn(vec![1.0], &[1])).sum(None)
    })
    .unwrap();
    assert_eq!(got.shape(), vec![3, 0]);
}
