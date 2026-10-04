//! facade（`fandhe_ai::Var`）経由の `eigh`／`slogdet`／`pinv`／`matrix_rank`／
//! `lstsq` 利用例と単体テスト（イシュー #2515。実装は #2150・親 #2500・
//! ルート #2499）。
//!
//! `Var::eigh` 等は `fandhe_ai_autodiff::linalg_ops` の同名自由関数への
//! 1 行委譲メソッドで、facade は `Var` を再エクスポートするため追加の公開
//! 経路を持たない（`linalg_ops` モジュール・`EighVars`／`SlogdetVars` 型は
//! facade から再エクスポートしないため、本テストは `pub` フィールドで結果へ
//! 到達する）。期待値は厳密に表せる入力での完全一致とし、tolerance は新設
//! しない（REQ-2 の統一複合判定は GPU 比較側 `linalg_ops_backend_parity.rs`）。
//! 委譲が自由関数と bit 一致する検査だけは自由関数を直接 use する
//! （facade の dev 依存に autodiff あり）。

use fandhe_ai::{AutodiffError, Tensor, Var};
use fandhe_ai_autodiff::linalg_ops as free;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn vals(v: &Var<'_>) -> Vec<f32> {
    v.to_tensor().host_slice().into_owned()
}

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|f| f.to_bits()).collect()
}

#[test]
fn var_eigh_diagonal_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![3.0, 0.0, 0.0, 1.0], &[2, 2]));
    let r = x.eigh().unwrap();
    assert_eq!(vals(&r.eigenvalues), [1.0, 3.0]);
    assert_eq!(r.eigenvalues.to_tensor().shape(), &[2]);
    assert_eq!(r.eigenvectors.to_tensor().shape(), &[2, 2]);
    // 符号は不定なので絶対値が置換行列になることで確認する。
    let abs: Vec<f32> = vals(&r.eigenvectors).iter().map(|v| v.abs()).collect();
    assert_eq!(abs, [0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn var_slogdet_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 0.0, 0.0, -2.0], &[2, 2]));
    let r = x.slogdet().unwrap();
    assert_eq!(vals(&r.sign), [-1.0]);
    assert_eq!(r.sign.to_tensor().shape(), &[] as &[usize]);
    let free_r = free::slogdet(&x).unwrap();
    assert_eq!(bits(&vals(&r.logabsdet)), bits(&vals(&free_r.logabsdet)));

    let sing = tape.var(&t(vec![1.0, 0.0, 0.0, 0.0], &[2, 2]));
    let s = sing.slogdet().unwrap();
    assert_eq!(vals(&s.sign), [0.0]);
    assert_eq!(vals(&s.logabsdet), [f32::NEG_INFINITY]);
}

#[test]
fn var_pinv_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![2.0, 0.0, 0.0, 4.0], &[2, 2]));
    assert_eq!(vals(&x.pinv(None).unwrap()), [0.5, 0.0, 0.0, 0.25]);
    let rect = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 3]));
    assert_eq!(rect.pinv(None).unwrap().to_tensor().shape(), &[3, 2]);
}

#[test]
fn var_matrix_rank_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 0.0, 0.0, 0.0], &[2, 2]));
    let r = x.matrix_rank(None).unwrap();
    assert_eq!(vals(&r), [1.0]);
    assert_eq!(r.to_tensor().shape(), &[] as &[usize]);
    let y = tape.var(&t(vec![2.0, 0.0, 0.0, 1.0], &[2, 2]));
    assert_eq!(vals(&y.matrix_rank(Some(0.75)).unwrap()), [1.0]);
}

#[test]
fn var_lstsq_via_facade() {
    let tape = fandhe_ai::tape();
    let a = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0, 7.0, 1.0, 2.0], &[2, 2]));
    let x = a.lstsq(&b, None).unwrap();
    assert_eq!(vals(&x), [5.0, 7.0, 1.0, 2.0]);
    assert_eq!(x.to_tensor().shape(), &[2, 2]);
}

#[test]
fn var_linalg_ops_match_free_functions_bitwise() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![2.0, 1.0, 1.0, 3.0], &[2, 2]));
    let b = tape.var(&t(vec![1.0, 2.0], &[2, 1]));
    let e = x.eigh().unwrap();
    let fe = free::eigh(&x).unwrap();
    assert_eq!(bits(&vals(&e.eigenvalues)), bits(&vals(&fe.eigenvalues)));
    assert_eq!(bits(&vals(&e.eigenvectors)), bits(&vals(&fe.eigenvectors)));
    assert_eq!(
        bits(&vals(&x.pinv(None).unwrap())),
        bits(&vals(&free::pinv(&x, None).unwrap()))
    );
    assert_eq!(
        bits(&vals(&x.matrix_rank(None).unwrap())),
        bits(&vals(&free::matrix_rank(&x, None).unwrap()))
    );
    assert_eq!(
        bits(&vals(&x.lstsq(&b, None).unwrap())),
        bits(&vals(&free::lstsq(&x, &b, None).unwrap()))
    );
}

fn grad_bits(tape: &fandhe_ai::Tape, loss: &Var<'_>, wrt: &Var<'_>) -> Vec<u32> {
    let grads = tape.backward(loss).unwrap();
    let g = grads.get(wrt).unwrap().unwrap();
    bits(&g.host_slice())
}

#[test]
fn var_linalg_ops_backward_matches_free_functions_bitwise() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![2.0, 1.0, 1.0, 3.0], &[2, 2]));
    let b = tape.var(&t(vec![1.0, 2.0], &[2, 1]));

    let l1 = x.eigh().unwrap().eigenvalues.sum(None).unwrap();
    let l2 = free::eigh(&x).unwrap().eigenvalues.sum(None).unwrap();
    assert_eq!(grad_bits(&tape, &l1, &x), grad_bits(&tape, &l2, &x));

    let l1 = x.slogdet().unwrap().logabsdet;
    let l2 = free::slogdet(&x).unwrap().logabsdet;
    assert_eq!(grad_bits(&tape, &l1, &x), grad_bits(&tape, &l2, &x));

    let l1 = x.pinv(None).unwrap().sum(None).unwrap();
    let l2 = free::pinv(&x, None).unwrap().sum(None).unwrap();
    assert_eq!(grad_bits(&tape, &l1, &x), grad_bits(&tape, &l2, &x));

    let l1 = x.lstsq(&b, None).unwrap().sum(None).unwrap();
    let l2 = free::lstsq(&x, &b, None).unwrap().sum(None).unwrap();
    assert_eq!(grad_bits(&tape, &l1, &x), grad_bits(&tape, &l2, &x));
    assert_eq!(grad_bits(&tape, &l1, &b), grad_bits(&tape, &l2, &b));
}

#[test]
fn var_matrix_rank_has_zero_gradient() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![2.0, 1.0, 1.0, 3.0], &[2, 2]));
    let loss = x.matrix_rank(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    if let Some(g) = grads.get(&x).unwrap() {
        assert!(g.host_slice().iter().all(|v| *v == 0.0));
    }
}

#[test]
fn var_linalg_ops_error_propagation() {
    let tape = fandhe_ai::tape();
    let nan = tape.var(&t(vec![f32::NAN, 0.0, 0.0, 1.0], &[2, 2]));
    assert!(matches!(nan.eigh(), Err(AutodiffError::InvalidArgument(_))));
    assert!(matches!(
        nan.pinv(None),
        Err(AutodiffError::InvalidArgument(_))
    ));
    let x = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    assert!(x.pinv(Some(-1.0)).is_err());
    assert!(x.pinv(Some(f32::NAN)).is_err());
    let rect = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    assert!(rect.eigh().is_err());
    assert!(rect.slogdet().is_err());
}
