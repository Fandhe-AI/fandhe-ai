//! facade（`fandhe_ai::Var`）経由の `tril`／`triu`／`diag`／`trace`／`outer`／
//! `dot` 利用例と単体テスト（イシュー #2513。実装は #2144・親 #2500・
//! ルート #2499）。
//!
//! `Var::tril` 等は `fandhe_ai_autodiff::matrix_ops` の同名自由関数への
//! 1 行委譲メソッドで、facade は `Var` を再エクスポートするため追加の公開
//! 経路を持たない。本テストは `fandhe_ai::` のパスだけで forward・backward・
//! エラー伝播を CPU tape 上で確認する。委譲が自由関数と bit 一致する検査
//! だけは自由関数を直接 use する（facade の dev 依存に autodiff あり）。
//! 数値比較は完全一致（厳密に表せる値）とし、tolerance は新設しない
//! （REQ-2 の統一複合判定は GPU との比較側 `matrix_ops_backend_parity.rs`）。

use fandhe_ai::{AutodiffError, ShapeError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn vals(v: &fandhe_ai::Var<'_>) -> Vec<f32> {
    v.to_tensor().host_slice().into_owned()
}

fn seq(n: usize) -> Vec<f32> {
    (1..=n).map(|v| v as f32).collect()
}

#[test]
fn var_tril_triu_forward_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(seq(9), &[3, 3]));
    assert_eq!(
        vals(&x.tril(0).unwrap()),
        [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
    );
    assert_eq!(
        vals(&x.tril(1).unwrap()),
        [1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
    assert_eq!(
        vals(&x.tril(-1).unwrap()),
        [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0]
    );
    assert_eq!(
        vals(&x.triu(0).unwrap()),
        [1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
    );
    assert_eq!(
        vals(&x.triu(1).unwrap()),
        [0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        vals(&x.triu(-1).unwrap()),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]
    );
}

#[test]
fn var_tril_triu_batched_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(seq(8), &[2, 2, 2]));
    let l = x.tril(0).unwrap();
    assert_eq!(l.to_tensor().shape(), &[2, 2, 2]);
    assert_eq!(vals(&l), [1.0, 0.0, 3.0, 4.0, 5.0, 0.0, 7.0, 8.0]);
    let u = x.triu(0).unwrap();
    assert_eq!(vals(&u), [1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 0.0, 8.0]);
}

#[test]
fn var_diag_extract_and_build_via_facade() {
    let tape = fandhe_ai::tape();
    let m = tape.var(&t(seq(9), &[3, 3]));
    assert_eq!(vals(&m.diag(0).unwrap()), [1.0, 5.0, 9.0]);
    assert_eq!(vals(&m.diag(1).unwrap()), [2.0, 6.0]);
    assert_eq!(vals(&m.diag(-1).unwrap()), [4.0, 8.0]);
    // 範囲外の対角は空（形状 [0]）。
    let empty = m.diag(5).unwrap();
    assert_eq!(empty.to_tensor().shape(), &[0]);

    let v = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let d0 = v.diag(0).unwrap();
    assert_eq!(d0.to_tensor().shape(), &[3, 3]);
    assert_eq!(vals(&d0), [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    let dp = v.diag(1).unwrap();
    assert_eq!(dp.to_tensor().shape(), &[4, 4]);
    assert_eq!(dp.to_tensor().host_slice()[1], 1.0);
    let dn = v.diag(-1).unwrap();
    assert_eq!(dn.to_tensor().shape(), &[4, 4]);
    assert_eq!(dn.to_tensor().host_slice()[4], 1.0);
}

#[test]
fn var_trace_via_facade() {
    let tape = fandhe_ai::tape();
    let sq = tape.var(&t(seq(9), &[3, 3]));
    assert_eq!(vals(&sq.trace().unwrap()), [15.0]);
    let rect = tape.var(&t(seq(6), &[2, 3]));
    assert_eq!(vals(&rect.trace().unwrap()), [6.0]);
}

#[test]
fn var_outer_dot_via_facade() {
    let tape = fandhe_ai::tape();
    let a = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let b = tape.var(&t(vec![4.0, 5.0], &[2]));
    let o = a.outer(&b).unwrap();
    assert_eq!(o.to_tensor().shape(), &[3, 2]);
    assert_eq!(vals(&o), [4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);

    let c = tape.var(&t(vec![4.0, 5.0, 6.0], &[3]));
    assert_eq!(vals(&a.dot(&c).unwrap()), [32.0]);
}

#[test]
fn var_tril_and_dot_backward_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(seq(4), &[2, 2]));
    let loss = x.tril(0).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    assert_eq!(dx.host_slice().into_owned(), [1.0, 0.0, 1.0, 1.0]);

    let tape = fandhe_ai::tape();
    let a = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let b = tape.var(&t(vec![4.0, 5.0, 6.0], &[3]));
    let loss = a.dot(&b).unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert_eq!(
        grads.get(&a).unwrap().unwrap().host_slice().into_owned(),
        [4.0, 5.0, 6.0]
    );
    assert_eq!(
        grads.get(&b).unwrap().unwrap().host_slice().into_owned(),
        [1.0, 2.0, 3.0]
    );
}

#[test]
fn var_matrix_ops_match_free_functions_bitwise() {
    use fandhe_ai_autodiff::matrix_ops as free;
    let tape = fandhe_ai::tape();
    let m = tape.var(&t(
        vec![1.5, -2.0, f32::NAN, 4.25, 5.0, 6.0, 7.5, 8.0, 9.0],
        &[3, 3],
    ));
    let v = tape.var(&t(vec![1.5, -2.0, 3.25], &[3]));
    let w = tape.var(&t(vec![0.5, 4.0, -6.0], &[3]));
    let bits =
        |x: &fandhe_ai::Var<'_>| -> Vec<u32> { vals(x).iter().map(|f| f.to_bits()).collect() };
    assert_eq!(bits(&m.tril(1).unwrap()), bits(&free::tril(&m, 1).unwrap()));
    assert_eq!(
        bits(&m.triu(-1).unwrap()),
        bits(&free::triu(&m, -1).unwrap())
    );
    assert_eq!(bits(&m.diag(1).unwrap()), bits(&free::diag(&m, 1).unwrap()));
    assert_eq!(
        bits(&v.diag(-1).unwrap()),
        bits(&free::diag(&v, -1).unwrap())
    );
    assert_eq!(bits(&m.trace().unwrap()), bits(&free::trace(&m).unwrap()));
    assert_eq!(
        bits(&v.outer(&w).unwrap()),
        bits(&free::outer(&v, &w).unwrap())
    );
    assert_eq!(bits(&v.dot(&w).unwrap()), bits(&free::dot(&v, &w).unwrap()));
}

#[test]
fn var_matrix_ops_propagate_errors() {
    let tape = fandhe_ai::tape();
    let v = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let m = tape.var(&t(seq(4), &[2, 2]));
    let r3 = tape.var(&t(seq(8), &[2, 2, 2]));

    assert!(matches!(
        v.tril(0),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert!(matches!(
        v.triu(0),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert!(matches!(
        v.trace(),
        Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            ..
        }))
    ));
    assert!(matches!(
        r3.trace(),
        Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            ..
        }))
    ));
    assert!(matches!(r3.diag(0), Err(AutodiffError::InvalidArgument(_))));
    assert!(matches!(
        m.outer(&v),
        Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 1,
            ..
        }))
    ));
    assert!(matches!(
        m.dot(&m),
        Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 1,
            ..
        }))
    ));
    let short = tape.var(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        v.dot(&short),
        Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
    ));
}
