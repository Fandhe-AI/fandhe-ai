//! `Var::interpolate` の新規 5 モード（`NearestExact`／`Area`／
//! `Linear`／`Trilinear`／`Bicubic`。イシュー #2152）の統合テスト。
//!
//! `Tape::new_with_ops(common::naive_ops())` を使い（`NaiveOps` は
//! `interpolate` 未実装のため `BackendError::Unsupported` を返し、
//! `grad::interpolate_with_fallback` がホスト参照実装
//! （`autodiff::eval::interpolate_*`）へ必ずフォールバックする——
//! `autodiff` は具体バックエンドクレートへ依存しない設計上の不変条件
//! のため、3 バックエンド parity は `crates/facade/tests/
//! interpolate_backend_parity.rs` 側で扱う。本ファイルは forward の
//! 手計算値照合・backward の中央差分照合・shape 拒否契約を担う）。
//!
//! 許容誤差は `crates/autodiff/tests/einsum_batch_parity.rs` の
//! grad-check 定数（`H=1e-3`・相対 1e-2 または絶対 1e-3・`τ=1e-4`。
//! Issue #223 承認済み）をそのまま再利用し新しい閾値は導入しない。

mod common;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_tensor_core::{InterpolateMode, ShapeError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn assert_close_slice(got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len());
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= tol,
            "index {i}: got={g} expected={e} tol={tol}"
        );
    }
}

// --- forward: 手計算値照合 ---

#[test]
fn nearest_exact_forward_matches_hand_computed_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((1..=8).map(|v| v as f32).collect(), &[8]));
    let out = x.interpolate(&[3], InterpolateMode::NearestExact).unwrap();
    // floor((dst+0.5)*8/3): dst=0->1(idx1=2.0), dst=1->4(idx4=5.0), dst=2->6(idx6=7.0)。
    let got = out.to_tensor();
    assert_close_slice(got.contiguous().as_slice().unwrap(), &[2.0, 5.0, 7.0], 0.0);
}

#[test]
fn nearest_exact_identity_size_is_passthrough() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let out = x.interpolate(&[4], InterpolateMode::NearestExact).unwrap();
    let got = out.to_tensor();
    assert_close_slice(
        got.contiguous().as_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
        0.0,
    );
}

#[test]
fn area_forward_matches_adaptive_avg_pool_hand_computed_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // 4 要素を 2 出力へ縮小: window(0)=[0,2) mean=1.5, window(1)=[2,4) mean=3.5。
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let out = x.interpolate(&[2], InterpolateMode::Area).unwrap();
    let got = out.to_tensor();
    assert_close_slice(got.contiguous().as_slice().unwrap(), &[1.5, 3.5], 1e-6);
}

#[test]
fn area_forward_2d_matches_hand_computed_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // 4x4 -> 2x2: 各出力は 2x2 ブロックの平均。
    let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let x = tape.var(&t(data, &[4, 4]));
    let out = x.interpolate(&[2, 2], InterpolateMode::Area).unwrap();
    let got = out.to_tensor();
    // block(0,0)=[1,2,5,6] mean=3.5, block(0,1)=[3,4,7,8] mean=5.5,
    // block(1,0)=[9,10,13,14] mean=11.5, block(1,1)=[11,12,15,16] mean=13.5。
    assert_close_slice(
        got.contiguous().as_slice().unwrap(),
        &[3.5, 5.5, 11.5, 13.5],
        1e-6,
    );
}

#[test]
fn linear_forward_matches_hand_computed_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 3.0], &[2]));
    let mode = InterpolateMode::Linear {
        align_corners: true,
    };
    let out = x.interpolate(&[3], mode).unwrap();
    // align_corners=true: dst=0->src=0(1.0), dst=1->src=0.5(2.0), dst=2->src=1(3.0)。
    let got = out.to_tensor();
    assert_close_slice(got.contiguous().as_slice().unwrap(), &[1.0, 2.0, 3.0], 1e-6);
}

#[test]
fn trilinear_forward_identity_size_is_passthrough() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let x = tape.var(&t(data.clone(), &[2, 2, 2]));
    let mode = InterpolateMode::Trilinear {
        align_corners: true,
    };
    let out = x.interpolate(&[2, 2, 2], mode).unwrap();
    let got = out.to_tensor();
    assert_close_slice(got.contiguous().as_slice().unwrap(), &data, 1e-5);
}

#[test]
fn trilinear_forward_center_point_is_average_of_all_eight_corners() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let x = tape.var(&t(data, &[2, 2, 2]));
    let mode = InterpolateMode::Trilinear {
        align_corners: false,
    };
    // 2x2x2 -> 1x1x1: half-pixel 変換の唯一の出力点は全 8 コーナーの
    // 中心 -> 単純平均 (1+..+8)/8 = 4.5。
    let out = x.interpolate(&[1, 1, 1], mode).unwrap();
    let got = out.to_tensor();
    let v = got.get(&[0, 0, 0]).unwrap();
    assert!((v - 4.5).abs() < 1e-4, "got {v}");
}

#[test]
fn bicubic_forward_identity_size_is_passthrough() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let x = tape.var(&t(data.clone(), &[4, 4]));
    let mode = InterpolateMode::Bicubic {
        align_corners: true,
    };
    let out = x.interpolate(&[4, 4], mode).unwrap();
    let got = out.to_tensor();
    // align_corners=true・恒等 size では各出力座標の t=0 となり
    // 中央 tap（重み 1）を選ぶため、値がそのまま複製される。
    assert_close_slice(got.contiguous().as_slice().unwrap(), &data, 1e-3);
}

#[test]
fn bicubic_forward_constant_input_is_constant_output() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![7.0; 16], &[4, 4]));
    let mode = InterpolateMode::Bicubic {
        align_corners: false,
    };
    let out = x.interpolate(&[6, 6], mode).unwrap();
    let got = out.to_tensor();
    for v in got.contiguous().as_slice().unwrap() {
        assert!((v - 7.0).abs() < 1e-3, "got {v}");
    }
}

// --- backward: 中央差分照合 ---

const H: f64 = 1e-3;
const TAU: f32 = 1e-4;
const REL_TOL: f32 = 1e-2;
const ABS_TOL: f32 = 1e-3;

fn numeric_grad(target: &Tensor<f32>, perturb: impl Fn(&Tensor<f32>) -> f32) -> Tensor<f32> {
    let shape = target.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut data: Vec<f32> = (0..numel)
        .map(|flat| {
            let mut idx = vec![0usize; shape.len()];
            let mut rem = flat;
            for axis in (0..shape.len()).rev() {
                idx[axis] = rem % shape[axis];
                rem /= shape[axis];
            }
            target.get(&idx).unwrap_or(0.0)
        })
        .collect();
    let mut grad = vec![0f32; numel];
    for i in 0..numel {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = perturb(&t(data.clone(), &shape)) as f64;
        data[i] = (orig - H) as f32;
        let lm = perturb(&t(data.clone(), &shape)) as f64;
        data[i] = orig as f32;
        grad[i] = ((lp - lm) / (2.0 * H)) as f32;
    }
    t(grad, &shape)
}

fn assert_grad_close(label: &str, analytic: &Tensor<f32>, numeric: &Tensor<f32>) {
    assert_eq!(
        analytic.shape(),
        numeric.shape(),
        "{label}: shape が一致しない"
    );
    let shape = analytic.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..numel.max(1) {
        let av = analytic.get(&idx).unwrap_or(0.0);
        let nv = numeric.get(&idx).unwrap_or(0.0);
        let diff = (av - nv).abs();
        let rel = diff / av.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[{idx:?}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
        );
        if shape.is_empty() {
            break;
        }
        for axis in (0..shape.len()).rev() {
            idx[axis] += 1;
            if idx[axis] < shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
}

fn loss_of_interpolate(x: &Tensor<f32>, size: &[usize], mode: InterpolateMode) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let v = tape.var(x);
    let out = v.interpolate(size, mode).unwrap();
    // 各出力要素へ異なる重みを与えることで、backward の scatter_add
    // 集約が「単純総和」に潰れて誤差を隠さないようにする。
    let out_t = out.to_tensor();
    let out_shape = out_t.shape().to_vec();
    let weight: Vec<f32> = (0..out_t.numel()).map(|i| 0.3 + 0.1 * (i as f32)).collect();
    let w = tape.var(&t(weight, &out_shape));
    let weighted = out.mul(&w).unwrap();
    let loss = weighted.sum(None).unwrap();
    loss.to_tensor()
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

fn analytic_grad(x: &Tensor<f32>, size: &[usize], mode: InterpolateMode) -> Tensor<f32> {
    let tape = Tape::new_with_ops(common::naive_ops());
    let v = tape.var(x);
    let out = v.interpolate(size, mode).unwrap();
    let out_t = out.to_tensor();
    let out_shape = out_t.shape().to_vec();
    let weight: Vec<f32> = (0..out_t.numel()).map(|i| 0.3 + 0.1 * (i as f32)).collect();
    let w = tape.var(&t(weight, &out_shape));
    let weighted = out.mul(&w).unwrap();
    let loss = weighted.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    grads
        .get(&v)
        .unwrap()
        .expect("x は loss に到達する")
        .clone()
}

#[test]
fn nearest_exact_backward_matches_numeric_grad() {
    let x = t((1..=8).map(|v| v as f32 * 0.7 - 2.0).collect(), &[8]);
    let mode = InterpolateMode::NearestExact;
    let da = analytic_grad(&x, &[5], mode);
    let num = numeric_grad(&x, |px| loss_of_interpolate(px, &[5], mode));
    assert_grad_close("nearest_exact", &da, &num);
}

#[test]
fn area_backward_matches_numeric_grad_downsample() {
    let x = t((1..=8).map(|v| v as f32 * 0.5 - 1.0).collect(), &[8]);
    let mode = InterpolateMode::Area;
    let da = analytic_grad(&x, &[3], mode);
    let num = numeric_grad(&x, |px| loss_of_interpolate(px, &[3], mode));
    assert_grad_close("area", &da, &num);
}

#[test]
fn area_backward_matches_numeric_grad_upsample() {
    let x = t(vec![1.0, -2.0, 3.0], &[3]);
    let mode = InterpolateMode::Area;
    let da = analytic_grad(&x, &[7], mode);
    let num = numeric_grad(&x, |px| loss_of_interpolate(px, &[7], mode));
    assert_grad_close("area_upsample", &da, &num);
}

#[test]
fn linear_backward_matches_numeric_grad() {
    let x = t(vec![1.0, -2.0, 3.0, 0.5], &[4]);
    for align_corners in [false, true] {
        let mode = InterpolateMode::Linear { align_corners };
        let da = analytic_grad(&x, &[7], mode);
        let num = numeric_grad(&x, |px| loss_of_interpolate(px, &[7], mode));
        assert_grad_close("linear", &da, &num);
    }
}

#[test]
fn trilinear_backward_matches_numeric_grad() {
    let x = t((1..=8).map(|v| v as f32 * 0.3 - 1.0).collect(), &[2, 2, 2]);
    for align_corners in [false, true] {
        let mode = InterpolateMode::Trilinear { align_corners };
        let da = analytic_grad(&x, &[3, 3, 3], mode);
        let num = numeric_grad(&x, |px| loss_of_interpolate(px, &[3, 3, 3], mode));
        assert_grad_close("trilinear", &da, &num);
    }
}

#[test]
fn bicubic_backward_matches_numeric_grad() {
    let x = t((1..=16).map(|v| v as f32 * 0.2 - 1.5).collect(), &[4, 4]);
    for align_corners in [false, true] {
        let mode = InterpolateMode::Bicubic { align_corners };
        let da = analytic_grad(&x, &[6, 6], mode);
        let num = numeric_grad(&x, |px| loss_of_interpolate(px, &[6, 6], mode));
        assert_grad_close("bicubic", &da, &num);
    }
}

#[test]
fn bicubic_backward_matches_numeric_grad_downsample() {
    let x = t((1..=16).map(|v| v as f32 * 0.2 - 1.5).collect(), &[4, 4]);
    let mode = InterpolateMode::Bicubic {
        align_corners: false,
    };
    let da = analytic_grad(&x, &[2, 3], mode);
    let num = numeric_grad(&x, |px| loss_of_interpolate(px, &[2, 3], mode));
    assert_grad_close("bicubic_downsample", &da, &num);
}

#[test]
fn interpolate_backward_with_leading_batch_axis() {
    // 先頭の残り軸（batch）が空間軸の対応を変えないことを勾配側でも
    // 確認する（`outer` 軸の扱い）。
    let x = t((1..=16).map(|v| v as f32 * 0.25 - 1.0).collect(), &[2, 8]);
    let mode = InterpolateMode::NearestExact;
    let da = analytic_grad(&x, &[3], mode);
    let num = numeric_grad(&x, |px| loss_of_interpolate(px, &[3], mode));
    assert_grad_close("nearest_exact_batch", &da, &num);
}

// --- shape 拒否契約 ---

#[test]
fn linear_mode_rejects_rank_other_than_one() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((1..=8).map(|v| v as f32).collect(), &[2, 4]));
    let mode = InterpolateMode::Linear {
        align_corners: false,
    };
    let err = x.interpolate(&[3, 3], mode).unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_autodiff::AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 1,
            actual: 2
        })
    ));
}

#[test]
fn trilinear_mode_rejects_rank_other_than_three() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((1..=8).map(|v| v as f32).collect(), &[2, 4]));
    let mode = InterpolateMode::Trilinear {
        align_corners: false,
    };
    let err = x.interpolate(&[3], mode).unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_autodiff::AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 3,
            actual: 1
        })
    ));
}

#[test]
fn bicubic_mode_rejects_rank_other_than_two() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((1..=8).map(|v| v as f32).collect(), &[8]));
    let mode = InterpolateMode::Bicubic {
        align_corners: false,
    };
    let err = x.interpolate(&[3], mode).unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_autodiff::AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn nearest_exact_and_area_accept_arbitrary_rank() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((1..=24).map(|v| v as f32).collect(), &[2, 3, 4]));
    let out1 = x.interpolate(&[2], InterpolateMode::NearestExact).unwrap();
    assert_eq!(out1.to_tensor().shape(), &[2, 3, 2]);
    let out2 = x.interpolate(&[3, 2], InterpolateMode::Area).unwrap();
    assert_eq!(out2.to_tensor().shape(), &[2, 3, 2]);
}

#[test]
fn interpolate_size_from_scale_factor_reaches_size_arg() {
    // `interpolate_size_from_scale_factor`（tensor-core の純関数）の
    // 戻り値を既存の `size` 引数へそのまま渡す設計（実装計画 §3.6）
    // の到達確認。
    let x = t((1..=4).map(|v| v as f32).collect(), &[4]);
    let size = fandhe_ai_tensor_core::interpolate_size_from_scale_factor(&[4], &[2.0]).unwrap();
    assert_eq!(size, vec![8]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let v = tape.var(&x);
    let out = v.interpolate(&size, InterpolateMode::NearestExact).unwrap();
    assert_eq!(out.to_tensor().shape(), &[8]);
}
