//! double-VJP 法の実現可能性検証（イシュー #2880）のうち、`pub(crate)` でしか作れない
//! Op variant（`Var::scalar_unary`／`scalar_binary`／`contiguous`）の単体検証。
//!
//! 統合テスト `tests/double_vjp_feasibility.rs` が公開 API から到達できる区分を担い、本
//! モジュールはその残りを同じ手順（設計 `docs/autodiff-functional-transforms-design.md`
//! §8）・同じ判定（REQ-2 統一複合判定。`create_graph.rs` の既存単体テストと同じ形でテスト内に
//! 置き、新しい定数名・閾値は作らない）で確かめる。`#[cfg(test)]` 専用で本番ビルドに入らない。
//! `jvp`／`jacfwd` の API は作らない。キンク（0・タイ）から離した入力を使う。

use fandhe_ai_tensor_core::{ScalarBinaryOp, ScalarUnaryOp, Tensor};

use crate::jacobian_ops::jacobian;
use crate::test_support::test_ops;
use crate::{AutodiffError, Tape, Var};

fn new_tape() -> Tape {
    Tape::new_with_ops(test_ops())
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test: shape とデータ長は一致させている")
}

fn seq(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.37).sin() * 1.5)
        .collect()
}

/// REQ-2 統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）。
fn close(actual: f64, expected: f64) -> bool {
    let diff = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs()).max(1e-12);
    diff / scale < 1e-3 || diff < 1e-5
}

fn double_vjp_probe(tape: &Tape, y: &Var<'_>, x: &Var<'_>, v: &Tensor<f32>, ctx: &str) -> Vec<f32> {
    let y_shape = y.shape();
    let m: usize = y_shape.iter().product();
    let len_before = tape.len();
    let u = tape.var(&t(seq(m, 7.0), &y_shape));
    let s = y.mul(&u).unwrap();
    let child = new_tape();
    let cg = tape
        .backward_create_graph(&s, &child)
        .unwrap_or_else(|e| panic!("{ctx}: backward_create_graph 失敗: {e:?}"));
    assert_eq!(tape.len(), len_before + 2, "{ctx}: 親テープの増分");
    let cu = cg
        .child_var(&u)
        .unwrap()
        .unwrap_or_else(|| panic!("{ctx}: child_var(u) が None"));
    let g = cg
        .grad(x)
        .unwrap()
        .unwrap_or_else(|| panic!("{ctx}: grad(x) が None"));
    assert!(g.requires_grad(), "{ctx}: grad(x) が勾配追跡なし");
    let prod = g.mul(&child.var_no_grad(v)).unwrap();
    let grads = child.backward(&prod).unwrap();
    let ju = grads
        .get(&cu)
        .unwrap()
        .unwrap_or_else(|| panic!("{ctx}: ∂t/∂u が None"));
    assert_eq!(ju.shape(), y_shape.as_slice(), "{ctx}: shape");
    ju.host_slice().into_owned()
}

fn check<F>(shape: &[usize], xv: Vec<f32>, ctx: &str, build: F)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Result<Var<'t>, AutodiffError>,
{
    let n: usize = shape.iter().product();
    let vv = seq(n, 5.0);

    let tb = new_tape();
    let xb = tb.var(&t(xv.clone(), shape));
    let yb = build(&tb, &xb).unwrap();
    let jh = jacobian(&tb, &yb, &xb).unwrap().host_slice().into_owned();
    let m: usize = yb.shape().iter().product();
    assert_eq!(jh.len(), m * n, "{ctx}: jacobian 要素数");

    let ta = new_tape();
    let xa = ta.var(&t(xv, shape));
    let ya = build(&ta, &xa).unwrap();
    let got = double_vjp_probe(&ta, &ya, &xa, &t(vv.clone(), shape), ctx);
    assert_eq!(got.len(), m, "{ctx}: 要素数");
    for (i, &a) in got.iter().enumerate() {
        let e: f64 = (0..n)
            .map(|k| f64::from(jh[i * n + k]) * f64::from(vv[k]))
            .sum();
        assert!(close(f64::from(a), e), "{ctx}[{i}]: {a} vs {e}");
    }
}

#[test]
fn scalar_unary_core_variants() {
    let xv = seq(6, 1.0);
    for (op, name) in [
        (ScalarUnaryOp::Relu, "relu"),
        (ScalarUnaryOp::Exp, "exp"),
        (ScalarUnaryOp::Tanh, "tanh"),
        (ScalarUnaryOp::Sigmoid, "sigmoid"),
    ] {
        check(&[6], xv.clone(), &format!("scalar_unary {name}"), |_, x| {
            x.scalar_unary(op)
        });
    }
}

#[test]
fn scalar_binary_add_mul_maximum_minimum() {
    let xv = seq(6, 1.0);
    // 定数側は x とタイにならない値（a 勝ち・b 勝ちが混在する）。
    for (op, name) in [
        (ScalarBinaryOp::Add, "add"),
        (ScalarBinaryOp::Mul, "mul"),
        (ScalarBinaryOp::Maximum, "maximum"),
        (ScalarBinaryOp::Minimum, "minimum"),
    ] {
        check(
            &[6],
            xv.clone(),
            &format!("scalar_binary {name} (x, c)"),
            |tp, x| x.scalar_binary(&k(tp), op),
        );
        check(
            &[6],
            xv.clone(),
            &format!("scalar_binary {name} (c, x)"),
            |tp, x| k(tp).scalar_binary(x, op),
        );
    }
    // 両オペランドが x に依存する Maximum（x と x²。タイは x が 0 か 1 の点のみで、入力はそこから離れている）。
    check(&[6], xv, "maximum (x, x²)", |_, x| {
        x.scalar_binary(&x.mul(x)?, ScalarBinaryOp::Maximum)
    });
}

#[test]
fn contiguous_after_permute() {
    check(&[2, 3], seq(6, 1.0), "permute→contiguous", |_, x| {
        x.permute(&[1, 0])?.contiguous()
    });
}

fn k<'t>(tp: &'t Tape) -> Var<'t> {
    tp.var_no_grad(&t(vec![0.3, -2.0, 0.9, 1.7, -0.4, 0.05], &[6]))
}
