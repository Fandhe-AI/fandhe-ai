//! `Var::var`／`std`／`norm_l1`／`norm_l2` の統合テスト（イシュー #1723）。
//!
//! `common::naive_ops()`（`BackendOps::var`／`vector_norm` を
//! オーバーライドしない `NaiveOps`）を使うため、`Var` 側のフォールバック
//! 経路（`BackendError::Unsupported` → `eval::var_along`／
//! `vector_norm_along`）を実質的に検証する（`backend-cpu` 側の実装
//! 自体の bit 一致検証は `crates/backend-cpu/src/reduction.rs` の
//! 単体テストが担当する。`docs/autodiff-linalg-design.md` の
//! 「eval と CPU の意図的複製」と同方針）。
//!
//! - 解析値（既知の分散／ノルム）が `sum`／`max` 等の既存統合テストと
//!   同じ手計算方式で一致することを確認する。
//! - 中央差分による数値微分と解析勾配（`Tape::backward`）を突合する
//!   （`tests/backward.rs::numeric_grad` と同方式・同許容誤差）。
//! - エラー経路（空縮約・自由度不足）が `AutodiffError::InvalidArgument`
//!   を返すことを確認する。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

const H: f64 = 1e-3;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

// =====================================================================
// 解析値（既知値）
// =====================================================================

#[test]
fn var_full_unbiased_matches_known_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let v = x.var(None, 1).unwrap();
    // [1,2,3,4] の不偏分散（mean=2.5・Σ(x-mean)^2=5・/(4-1)）= 5/3
    assert!((scalar(&v.to_tensor()) - 5.0 / 3.0).abs() < 1e-6);
}

#[test]
fn std_full_matches_sqrt_of_var() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], &[8]));
    let s = x.std(None, 0).unwrap();
    // 母分散 4（教科書的な標準偏差の例）→ std = 2
    assert!((scalar(&s.to_tensor()) - 2.0).abs() < 1e-5);
}

#[test]
fn var_axis_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // shape [2, 3]: 行 0 = [1,2,3]・行 1 = 定数 [4,4,4]（分散 0）
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 4.0, 4.0], &[2, 3]));
    let v = x.var(Some(1), 0).unwrap();
    assert_eq!(v.to_tensor().shape(), &[2]);
    assert!((v.to_tensor().get(&[0]).unwrap() - (2.0 / 3.0)).abs() < 1e-6);
    assert!((v.to_tensor().get(&[1]).unwrap() - 0.0).abs() < 1e-6);
}

#[test]
fn norm_l1_full_matches_known_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-1.0, 2.0, -3.0, 4.0], &[4]));
    let n = x.norm_l1(None).unwrap();
    assert!((scalar(&n.to_tensor()) - 10.0).abs() < 1e-6);
}

#[test]
fn norm_l2_full_matches_known_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 4.0], &[2]));
    let n = x.norm_l2(None).unwrap();
    assert!((scalar(&n.to_tensor()) - 5.0).abs() < 1e-6);
}

#[test]
fn norm_axis_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // shape [2, 2]: [[3,4],[-1,-1]]
    let x = tape.var(&t(vec![3.0, 4.0, -1.0, -1.0], &[2, 2]));
    let n_l2 = x.norm_l2(Some(1)).unwrap();
    assert!((n_l2.to_tensor().get(&[0]).unwrap() - 5.0).abs() < 1e-6);
    assert!((n_l2.to_tensor().get(&[1]).unwrap() - 2.0f32.sqrt()).abs() < 1e-6);

    let n_l1 = x.norm_l1(Some(1)).unwrap();
    assert!((n_l1.to_tensor().get(&[0]).unwrap() - 7.0).abs() < 1e-6);
    assert!((n_l1.to_tensor().get(&[1]).unwrap() - 2.0).abs() < 1e-6);
}

// =====================================================================
// エラー経路
// =====================================================================

#[test]
fn var_empty_reduction_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(Vec::new(), &[0]));
    let err = x.var(None, 1).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn var_insufficient_degrees_of_freedom_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0], &[1]));
    let err = x.var(None, 1).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn norm_l1_empty_reduction_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(Vec::new(), &[0]));
    let err = x.norm_l1(None).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// =====================================================================
// 数値微分突合（中央差分。`tests/backward.rs::numeric_grad` と同方式・
// 同許容誤差〈H=1e-3・相対 1e-2 または絶対 1e-3〉）。
// =====================================================================

fn assert_close(actual: &Tensor<f32>, expected: &Tensor<f32>) {
    assert_eq!(actual.shape(), expected.shape());
    let numel: usize = actual.shape().iter().product();
    for flat in 0..numel {
        let shape = actual.shape().to_vec();
        let mut idx = vec![0usize; shape.len()];
        let mut rem = flat;
        for axis in (0..shape.len()).rev() {
            idx[axis] = rem % shape[axis];
            rem /= shape[axis];
        }
        let a = actual.get(&idx).unwrap();
        let e = expected.get(&idx).unwrap();
        let diff = (a - e).abs();
        let rel_ok = diff <= 1e-2 * e.abs();
        let abs_ok = diff <= 1e-3;
        assert!(
            rel_ok || abs_ok,
            "数値微分不一致（idx={idx:?}）: actual={a}, expected={e}, diff={diff}"
        );
    }
}

fn numeric_grad(target_tensor: &Tensor<f32>, perturb: impl Fn(Tensor<f32>) -> f32) -> Tensor<f32> {
    let shape = target_tensor.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut data: Vec<f32> = (0..numel)
        .map(|flat| {
            let mut idx = vec![0usize; shape.len()];
            let mut rem = flat;
            for axis in (0..shape.len()).rev() {
                idx[axis] = rem % shape[axis];
                rem /= shape[axis];
            }
            target_tensor.get(&idx).unwrap_or(0.0)
        })
        .collect();
    let mut grad = vec![0f32; numel];
    for i in 0..numel {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = perturb(t(data.clone(), &shape)) as f64;
        data[i] = (orig - H) as f32;
        let lm = perturb(t(data.clone(), &shape)) as f64;
        data[i] = orig as f32;
        grad[i] = ((lp - lm) / (2.0 * H)) as f32;
    }
    t(grad, &shape)
}

// x を意図的に非対称な値にし、var の平均中心化項が消えないようにする
// （`H=1e-3` の摂動で `var`／`norm` の局所線形近似が成り立つ範囲）。
fn var_fixture() -> Tensor<f32> {
    t(vec![1.3, -0.7, 2.1, 0.4, -1.5, 0.9], &[6])
}

#[test]
fn var_full_grad_matches_numeric() {
    let x_val = var_fixture();

    let forward_loss = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let loss = xv.var(None, 1).unwrap();
        scalar(&loss.to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x_val);
    let loss = xv.var(None, 1).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let numeric = numeric_grad(&x_val, forward_loss);
    assert_close(&dx, &numeric);
}

#[test]
fn var_axis_grad_matches_numeric() {
    // shape [2, 3]（軸 1 沿いで分散を取る）
    let x_val = t(vec![1.3, -0.7, 2.1, 0.4, -1.5, 0.9], &[2, 3]);

    let forward_loss = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let loss = xv.var(Some(1), 1).unwrap().sum(None).unwrap();
        scalar(&loss.to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x_val);
    let loss = xv.var(Some(1), 1).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let numeric = numeric_grad(&x_val, forward_loss);
    assert_close(&dx, &numeric);
}

#[test]
fn std_full_grad_matches_numeric() {
    let x_val = var_fixture();

    let forward_loss = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let loss = xv.std(None, 1).unwrap();
        scalar(&loss.to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x_val);
    let loss = xv.std(None, 1).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let numeric = numeric_grad(&x_val, forward_loss);
    assert_close(&dx, &numeric);
}

#[test]
fn norm_l1_full_grad_matches_numeric() {
    // L1 は各要素の符号が摂動で反転しないよう十分に 0 から離す。
    let x_val = t(vec![1.3, -0.7, 2.1, 0.4, -1.5, 0.9], &[6]);

    let forward_loss = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let loss = xv.norm_l1(None).unwrap();
        scalar(&loss.to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x_val);
    let loss = xv.norm_l1(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let numeric = numeric_grad(&x_val, forward_loss);
    assert_close(&dx, &numeric);
}

#[test]
fn norm_l2_full_grad_matches_numeric() {
    let x_val = var_fixture();

    let forward_loss = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let loss = xv.norm_l2(None).unwrap();
        scalar(&loss.to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x_val);
    let loss = xv.norm_l2(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let numeric = numeric_grad(&x_val, forward_loss);
    assert_close(&dx, &numeric);
}

// =====================================================================
// 退化ケース（勾配ゼロの規約確認）
// =====================================================================

#[test]
fn var_constant_input_gradient_is_zero() {
    // 定数テンソルの分散は 0（勾配も理論上 0。mean 中心化項が
    // すべての要素で完全相殺するため）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![4.0, 4.0, 4.0, 4.0], &[4]));
    let loss = x.var(None, 1).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    for v in [
        dx.get(&[0]).unwrap(),
        dx.get(&[1]).unwrap(),
        dx.get(&[2]).unwrap(),
        dx.get(&[3]).unwrap(),
    ] {
        assert!((v).abs() < 1e-5);
    }
}

#[test]
fn norm_l2_zero_vector_gradient_is_zero() {
    // ‖0‖ = 0 の点では L2 ノルムの勾配は定義上 0（`matrix_norm_vjp` の
    // ゼロノルム分岐と同じ規約）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0, 0.0, 0.0], &[3]));
    let loss = x.norm_l2(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    for v in [
        dx.get(&[0]).unwrap(),
        dx.get(&[1]).unwrap(),
        dx.get(&[2]).unwrap(),
    ] {
        assert_eq!(v, 0.0);
    }
}
