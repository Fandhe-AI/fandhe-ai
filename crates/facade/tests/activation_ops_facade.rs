//! facade（`fandhe_ai::Var`）経由の `mish`／`hardtanh`／`relu6`／`prelu`／`glu`
//! 利用例と単体テスト（イシュー #2516。実装は #2146・親 #2500・ルート #2499）。
//!
//! `Var::mish` 等は `fandhe_ai_autodiff::activation_ops` の同名自由関数への
//! 1 行委譲メソッドで、facade は `Var` を再エクスポートするため追加の公開
//! 経路を持たない。本テストは `fandhe_ai::` のパスだけで forward・backward・
//! エラー伝播を CPU tape 上で確認する。委譲が自由関数と bit 一致する検査
//! だけは自由関数を直接 use する（facade の dev 依存に autodiff あり）。
//! 厳密に表せる値は完全一致で、超越関数を含む値（mish・glu の sigmoid 等）は
//! REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で比較し、
//! tolerance は新設・変更しない。GPU との比較側は
//! `activation_ops_backend_parity.rs`。

use fandhe_ai::{AutodiffError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn vals(v: &fandhe_ai::Var<'_>) -> Vec<f32> {
    v.to_tensor().host_slice().into_owned()
}

fn bits(v: &fandhe_ai::Var<'_>) -> Vec<u32> {
    vals(v).iter().map(|f| f.to_bits()).collect()
}

/// REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。
fn assert_req2_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        let abs = (a - e).abs();
        let rel = abs / e.abs().max(f32::MIN_POSITIVE);
        assert!(
            abs < 1e-5 || rel < 1e-3,
            "REQ-2 判定外: actual={a} expected={e}"
        );
    }
}

fn grad_of(tape: &fandhe_ai::Tape, loss: &fandhe_ai::Var<'_>, x: &fandhe_ai::Var<'_>) -> Vec<f32> {
    let grads = tape.backward(loss).unwrap();
    grads.get(x).unwrap().unwrap().host_slice().into_owned()
}

#[test]
fn var_activation_forward_via_facade() {
    let tape = fandhe_ai::tape();
    let z = tape.var(&t(vec![0.0, 1.0, -1.0], &[3]));
    // mish(x) = x * tanh(softplus(x))。厳密値は 0 のみ、他は REQ-2 判定。
    assert_req2_close(&vals(&z.mish().unwrap()), &[0.0, 0.865_098_4, -0.303_401_4]);

    let x = tape.var(&t(vec![-2.0, 0.5, 2.0], &[3]));
    assert_eq!(vals(&x.hardtanh(-1.0, 1.0).unwrap()), [-1.0, 0.5, 1.0]);

    let r = tape.var(&t(vec![-1.0, 3.0, 7.0], &[3]));
    assert_eq!(vals(&r.relu6().unwrap()), [0.0, 3.0, 6.0]);

    let p = tape.var(&t(vec![-2.0, 3.0], &[2]));
    let w = tape.var(&t(vec![0.25], &[1]));
    assert_eq!(vals(&p.prelu(&w).unwrap()), [-0.5, 3.0]);

    // gate 側が 0 なら sigmoid = 0.5。
    let g = tape.var(&t(vec![2.0, 4.0, 0.0, 0.0], &[4]));
    assert_eq!(vals(&g.glu(0).unwrap()), [1.0, 2.0]);
}

#[test]
fn var_hardtanh_relu6_match_clamp_bitwise() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![-3.0, -1.0, 0.25, 1.0, 6.0, 9.0], &[6]));
    assert_eq!(
        bits(&x.hardtanh(-1.0, 1.0).unwrap()),
        bits(&x.clamp(-1.0, 1.0).unwrap())
    );
    assert_eq!(bits(&x.relu6().unwrap()), bits(&x.clamp(0.0, 6.0).unwrap()));
}

#[test]
fn var_hardtanh_relu6_backward_boundaries_are_zero() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![-2.0, -1.0, 0.5, 1.0, 2.0], &[5]));
    let loss = x.hardtanh(-1.0, 1.0).unwrap().sum(None).unwrap();
    assert_eq!(grad_of(&tape, &loss, &x), [0.0, 0.0, 1.0, 0.0, 0.0]);

    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![-1.0, 0.0, 3.0, 6.0, 7.0], &[5]));
    let loss = x.relu6().unwrap().sum(None).unwrap();
    assert_eq!(grad_of(&tape, &loss, &x), [0.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn var_prelu_backward_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![-2.0, 3.0, -4.0], &[3]));
    let w = tape.var(&t(vec![0.25], &[1]));
    let loss = x.prelu(&w).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
    assert_eq!(dx, [0.25, 1.0, 0.25]);
    // weight 勾配は負側入力の和（-2 + -4）。
    let dw = grads.get(&w).unwrap().unwrap().host_slice().into_owned();
    assert_req2_close(&dw, &[-6.0]);
}

#[test]
fn var_mish_glu_backward_is_finite_and_distributed() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![-1.0, 0.0, 2.0], &[3]));
    let loss = x.mish().unwrap().sum(None).unwrap();
    assert!(grad_of(&tape, &loss, &x).iter().all(|g| g.is_finite()));

    // glu: out = a * sigmoid(b)。da = sigmoid(b)、db = a * s * (1 - s)。
    let tape = fandhe_ai::tape();
    let g = tape.var(&t(vec![2.0, 4.0, 0.0, 0.0], &[4]));
    let loss = g.glu(0).unwrap().sum(None).unwrap();
    assert_req2_close(&grad_of(&tape, &loss, &g), &[0.5, 0.5, 0.5, 1.0]);
}

#[test]
fn var_activation_ops_match_free_functions_bitwise() {
    use fandhe_ai_autodiff::activation_ops as free;
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.5, -2.0, f32::NAN, 4.25], &[4]));
    assert_eq!(bits(&x.mish().unwrap()), bits(&free::mish(&x).unwrap()));
    assert_eq!(
        bits(&x.hardtanh(-1.0, 1.0).unwrap()),
        bits(&free::hardtanh(&x, -1.0, 1.0).unwrap())
    );
    assert_eq!(bits(&x.relu6().unwrap()), bits(&free::relu6(&x).unwrap()));
    let w = tape.var(&t(vec![0.25], &[1]));
    assert_eq!(
        bits(&x.prelu(&w).unwrap()),
        bits(&free::prelu(&x, &w).unwrap())
    );
    assert_eq!(bits(&x.glu(0).unwrap()), bits(&free::glu(&x, 0).unwrap()));
}

#[test]
fn var_activation_ops_propagate_errors() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    assert!(matches!(
        x.hardtanh(1.0, 1.0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        x.hardtanh(f32::NAN, 1.0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    let w2 = tape.var(&t(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]));
    assert!(matches!(x.prelu(&w2), Err(AutodiffError::Shape(_))));
    let w3 = tape.var(&t(vec![0.1, 0.2, 0.3], &[3]));
    assert!(matches!(x.prelu(&w3), Err(AutodiffError::Shape(_))));
    assert!(matches!(x.glu(2), Err(AutodiffError::Shape(_))));
    let odd = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    assert!(matches!(odd.glu(0), Err(AutodiffError::InvalidArgument(_))));
}
