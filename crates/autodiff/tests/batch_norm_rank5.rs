//! `Var::batch_norm` 系の rank 5（`[N, C, D, H, W]`。BatchNorm3d 相当）受理の
//! 統合テスト（イシュー #3819・承認 #3768・`docs/autodiff-batch-norm-3d-decision.md`）。
//!
//! 契約: レイアウト導出（`batch_norm_layout`）は `(n, c, spatial = D*H*W)` だけを
//! 下流へ渡すため、同一バックエンドでは rank 5 の結果が同じデータを
//! `[N, C, D*H*W]` へ reshape した rank 3 入力の結果と bit 一致する。
//! GPU カーネルは無変更で、本ファイルは naive ops（CPU 参照）のみを使う。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().expect("contiguous").to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

const N: usize = 2;
const C: usize = 3;
const S: [usize; 3] = [2, 3, 2];
const SP: usize = 12;

struct Run {
    out: Vec<f32>,
    mean: Vec<f32>,
    var: Vec<f32>,
    dx: Vec<f32>,
    dw: Option<Vec<f32>>,
    db: Option<Vec<f32>>,
}

fn run_train(rank5: bool, affine: bool) -> Run {
    let tape = Tape::new_with_ops(common::naive_ops());
    let total = N * C * SP;
    let x = common::det_seq(total, 0.3);
    let shape: Vec<usize> = if rank5 {
        vec![N, C, S[0], S[1], S[2]]
    } else {
        vec![N, C, SP]
    };
    let xv = tape.var(&Tensor::new(x, &shape).unwrap());
    let coef = tape.var(&Tensor::new(common::det_seq(total, 1.7), &shape).unwrap());
    let w = tape.var(&Tensor::new(vec![0.5f32, 1.25, -0.75], &[C]).unwrap());
    let b = tape.var(&Tensor::new(vec![0.1f32, -0.2, 0.3], &[C]).unwrap());
    let (wo, bo) = if affine {
        (Some(&w), Some(&b))
    } else {
        (None, None)
    };
    let (y, m, v) = xv.batch_norm_with_batch_stats(wo, bo, 1e-5).unwrap();
    let loss = y.mul(&coef).unwrap().sum(None).unwrap();
    let g = tape.backward(&loss).unwrap();
    Run {
        out: dense(&y.to_tensor()),
        mean: dense(&m),
        var: dense(&v),
        dx: dense(g.get(&xv).unwrap().unwrap()),
        dw: affine.then(|| dense(g.get(&w).unwrap().unwrap())),
        db: affine.then(|| dense(g.get(&b).unwrap().unwrap())),
    }
}

#[test]
fn rank5_train_and_grads_are_bit_identical_to_rank3_reshape() {
    for affine in [true, false] {
        let a = run_train(true, affine);
        let r = run_train(false, affine);
        assert_eq!(bits(&a.out), bits(&r.out), "out affine={affine}");
        assert_eq!(bits(&a.mean), bits(&r.mean));
        assert_eq!(bits(&a.var), bits(&r.var));
        assert_eq!(bits(&a.dx), bits(&r.dx), "dx affine={affine}");
        assert_eq!(a.dw.as_deref().map(bits), r.dw.as_deref().map(bits));
        assert_eq!(a.db.as_deref().map(bits), r.db.as_deref().map(bits));
    }
}

#[test]
fn rank5_train_matches_f64_naive_reference() {
    let a = run_train(true, false);
    let x = common::det_seq(N * C * SP, 0.3);
    let m = (N * SP) as f64;
    for c in 0..C {
        let mut s = 0.0f64;
        for n in 0..N {
            for i in 0..SP {
                s += x[(n * C + c) * SP + i] as f64;
            }
        }
        let mean = s / m;
        let mut q = 0.0f64;
        for n in 0..N {
            for i in 0..SP {
                let d = x[(n * C + c) * SP + i] as f64 - mean;
                q += d * d;
            }
        }
        let var = q / m;
        assert!(common::req2_close(a.mean[c] as f64, mean), "mean c={c}");
        assert!(common::req2_close(a.var[c] as f64, var), "var c={c}");
        let rstd = 1.0 / (var + 1e-5f32 as f64).sqrt();
        for n in 0..N {
            for i in 0..SP {
                let idx = (n * C + c) * SP + i;
                let e = (x[idx] as f64 - mean) * rstd;
                assert!(common::req2_close(a.out[idx] as f64, e), "out idx={idx}");
            }
        }
    }
}

#[test]
fn rank5_infer_is_bit_identical_to_rank3_reshape() {
    let rm = Tensor::new(vec![0.2f32, -0.1, 0.05], &[C]).unwrap();
    let rv = Tensor::new(vec![1.1f32, 0.9, 2.0], &[C]).unwrap();
    let go = |rank5: bool| {
        let tape = Tape::new_with_ops(common::naive_ops());
        let shape: Vec<usize> = if rank5 {
            vec![N, C, S[0], S[1], S[2]]
        } else {
            vec![N, C, SP]
        };
        let xv = tape.var(&Tensor::new(common::det_seq(N * C * SP, 0.9), &shape).unwrap());
        let w = tape.var(&Tensor::new(vec![0.5f32, 1.25, -0.75], &[C]).unwrap());
        let b = tape.var(&Tensor::new(vec![0.1f32, -0.2, 0.3], &[C]).unwrap());
        dense(
            &xv.batch_norm_infer(Some(&w), Some(&b), &rm, &rv, 1e-5)
                .unwrap()
                .to_tensor(),
        )
    };
    assert_eq!(bits(&go(true)), bits(&go(false)));
}

#[test]
fn rank6_is_rejected_with_expected_5() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&Tensor::new(vec![0.5f32; 64], &[2; 6]).unwrap());
    let err = xv.batch_norm(None, None, 1e-5).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 5,
            actual: 6
        })
    ));
}

#[test]
fn rank5_train_with_single_reduction_element_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&Tensor::new(vec![0.5f32; 3], &[1, 3, 1, 1, 1]).unwrap());
    let err = xv.batch_norm(None, None, 1e-5).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)), "{err:?}");
}

/// 空テンソル（`N=0`）の扱いは rank 4 と同じ規約になる。
#[test]
fn rank5_empty_tensor_matches_rank4_convention() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x5 = tape.var(&Tensor::new(Vec::<f32>::new(), &[0, 2, 2, 2, 2]).unwrap());
    let x4 = tape.var(&Tensor::new(Vec::<f32>::new(), &[0, 2, 2, 2]).unwrap());
    let r5 = x5.batch_norm(None, None, 1e-5).map(|_| ());
    let r4 = x4.batch_norm(None, None, 1e-5).map(|_| ());
    assert_eq!(r5.is_ok(), r4.is_ok(), "{r5:?} vs {r4:?}");
    if let (Err(e5), Err(e4)) = (r5, r4) {
        assert_eq!(std::mem::discriminant(&e5), std::mem::discriminant(&e4));
    }
}
