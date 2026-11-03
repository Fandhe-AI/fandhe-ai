//! `Var::lp_pool1d`／`lp_pool2d`／`lp_pool3d`（イシュー #3825・設計 `docs/autodiff-pool3d-ops-decision.md`
//! §19・承認 #3768）の受け入れテスト（CPU・`NaiveOps`）。
//!
//! 合成のみ（新規 `Op` なし）なので、ここでは (1) 閉形式の値、(2) 中央差分との勾配突合、(3) 拒否系と
//! 孤児ノードが残らないこと、(4) 決定性、(5) `create_graph` が panic せず型付きエラーになること
//! （承認 Q3 (a)）を確かめる。PyTorch 参照値との突合は `fixtures/lp-pool-pytorch-reference/` を参照。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn flat(tensor: &Tensor<f32>) -> Vec<f32> {
    tensor.contiguous().as_slice().unwrap().to_vec()
}

fn positive(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 + ((i as f32) * 0.37).sin().abs())
        .collect()
}

#[test]
fn closed_form_3d_with_stride() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // [1,1,2,2,4] を stride=[1,1,2] の [2,2,2] 窓で走査 → 出力 [1,1,1,1,2]。
    let x = tape.var(&t(vec![1.0; 16], &[1, 1, 2, 2, 4]));
    let y = x.lp_pool3d(2.0, [2, 2, 2], Some([1, 1, 2]), false).unwrap();
    assert_eq!(y.to_tensor().shape(), &[1, 1, 1, 1, 2]);
    let v = flat(&y.to_tensor());
    for e in v {
        assert!((e - 8.0f32.sqrt()).abs() < 1e-6);
    }
}

#[test]
fn gradient_matches_central_difference() {
    let shape = [1usize, 1, 4, 4];
    let n: usize = shape.iter().product();
    let base = positive(n);
    let loss_at = |x: &[f32], p: f32| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), &shape));
        let y = xv.lp_pool2d(p, [2, 2], None, false).unwrap();
        flat(&y.to_tensor()).iter().map(|&v| v as f64).sum()
    };
    for p in [1.0f32, 2.0, 3.0, 1.5] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(base.clone(), &shape));
        let loss = xv
            .lp_pool2d(p, [2, 2], None, false)
            .unwrap()
            .sum(None)
            .unwrap();
        let grads = tape.backward(&loss).unwrap();
        let g = flat(grads.get(&xv).unwrap().unwrap());
        let h = 1e-2f32;
        for i in 0..n {
            let (mut xp, mut xm) = (base.clone(), base.clone());
            xp[i] += h;
            xm[i] -= h;
            let num = (loss_at(&xp, p) - loss_at(&xm, p)) / (2.0 * h as f64);
            assert!(
                common::req2_close(g[i] as f64, num),
                "p={p} i={i}: analytic={} numeric={num}",
                g[i]
            );
        }
    }
}

#[test]
fn zero_window_gradient_is_zero_not_nan() {
    for p in [2.0f32, 1.5] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            &[1, 1, 2, 4],
        ));
        let loss = xv
            .lp_pool2d(p, [1, 4], None, false)
            .unwrap()
            .sum(None)
            .unwrap();
        let grads = tape.backward(&loss).unwrap();
        let g = flat(grads.get(&xv).unwrap().unwrap());
        assert_eq!(&g[..4], &[0.0; 4], "p={p}");
        assert!(g[4..].iter().all(|v| v.is_finite()));
    }
}

#[test]
fn fractional_p_below_one_with_zero_in_window_gives_nonfinite_gradient() {
    // 記録用: 0 < p < 1 かつ窓内に 0 があると x^p の勾配が非有限になる（PyTorch も同様）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&t(vec![0.0, 1.0], &[1, 1, 2]));
    let loss = xv
        .lp_pool1d(0.5, 2, None, false)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    let g = flat(grads.get(&xv).unwrap().unwrap());
    assert!(g.iter().any(|v| !v.is_finite()), "{g:?}");
}

#[test]
fn invalid_arguments_are_rejected_without_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0; 16], &[1, 1, 4, 4]));
    let x3 = tape.var(&t(vec![1.0; 8], &[1, 1, 2, 2, 2]));
    let before = tape.len();
    for bad in [0.0f32, -2.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            x.lp_pool2d(bad, [2, 2], None, false),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(x.lp_pool1d(bad, 2, None, false).is_err());
        assert!(x3.lp_pool3d(bad, [2, 2, 2], None, false).is_err());
    }
    assert!(x.lp_pool2d(2.0, [2, 2], None, true).is_err());
    assert!(x.lp_pool2d(2.0, [0, 2], None, false).is_err());
    assert!(x.lp_pool2d(2.0, [2, 2], Some([0, 1]), false).is_err());
    assert!(x.lp_pool2d(2.0, [8, 8], None, false).is_err());
    assert!(x.lp_pool1d(2.0, 2, None, false).is_err());
    assert!(x.lp_pool3d(2.0, [2, 2, 2], None, false).is_err());
    assert!(x3.lp_pool2d(2.0, [2, 2], None, false).is_err());
    assert!(x3.lp_pool3d(2.0, [usize::MAX, 2, 2], None, false).is_err());
    assert!(
        x3.lp_pool3d(2.0, [1 << 13, 1 << 13, 1], None, false)
            .is_err()
    );
    assert_eq!(tape.len(), before, "拒否時にノードを積まない");
}

#[test]
fn forward_is_deterministic() {
    let run = || {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(positive(32), &[1, 2, 4, 4]));
        flat(
            &x.lp_pool2d(3.0, [2, 2], Some([1, 1]), false)
                .unwrap()
                .to_tensor(),
        )
        .iter()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

#[test]
fn create_graph_returns_typed_error_instead_of_panic() {
    // 承認 Q3 (a): `Sign` が replay できないため、二階微分は型付きエラーになる（現行挙動の固定）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(positive(4), &[1, 1, 4]));
    let loss = x.lp_pool1d(2.0, 2, None, false).unwrap().sum(None).unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
}
