//! facade（`fandhe_ai::Var`）経由の `topk_with_options`／`unique_with_options`／
//! `unique_consecutive` 利用例と単体テスト（イシュー #2519。実装は #2153・
//! 親 #2500・ルート #2499。決定記録は `docs/autodiff-topk-unique-ops-decision.md`）。
//!
//! これらは `fandhe_ai_autodiff::topk_unique_ops` の同名自由関数への 1 行委譲
//! メソッドで、入出力型（`TopkOptions`／`UniqueOptions`／`UniqueOutput`）は facade
//! ルートから再エクスポートされる。本テストは `fandhe_ai::` のパスだけで
//! forward・backward・エラー伝播を CPU tape 上で確認する。委譲が自由関数と
//! bit 一致する検査だけは自由関数を直接 use する（facade の dev 依存に autodiff あり）。
//! 選択演算（丸めなし）のため数値比較は完全一致とし、tolerance は新設しない
//! （GPU との比較側は `topk_unique_ops_backend_parity.rs`）。

use fandhe_ai::{AutodiffError, ShapeError, Tensor, TopkOptions, UniqueOptions};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn vals(v: &fandhe_ai::Var<'_>) -> Vec<f32> {
    v.to_tensor().host_slice().into_owned()
}

#[test]
fn var_topk_with_options_forward_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![4.0, 5.0, 1.0], &[3]));
    // sorted=true（既定）: 値の大小順。
    let (v, i) = x.topk_with_options(2, TopkOptions::default()).unwrap();
    assert_eq!(vals(&v), [5.0, 4.0]);
    assert_eq!(i.host_slice().into_owned(), [1, 0]);
    // sorted=false: 選んだ k 個を元添字の昇順に並べる（決定的契約）。
    let opts = TopkOptions::default().with_sorted(false);
    let (v, i) = x.topk_with_options(2, opts).unwrap();
    assert_eq!(vals(&v), [4.0, 5.0]);
    assert_eq!(i.host_slice().into_owned(), [0, 1]);
    // 負の dim と largest=false。
    let m = tape.var(&t(vec![1.0, 3.0, 2.0, 6.0, 4.0, 5.0], &[2, 3]));
    let opts = TopkOptions::default().with_dim(-1).with_largest(false);
    let (v, i) = m.topk_with_options(1, opts).unwrap();
    assert_eq!(vals(&v), [1.0, 4.0]);
    assert_eq!(i.host_slice().into_owned(), [0, 1]);
    let opts = TopkOptions::default().with_dim(-2).with_largest(true);
    let (v, _) = m.topk_with_options(1, opts).unwrap();
    assert_eq!(vals(&v), [6.0, 4.0, 5.0]);
}

#[test]
fn var_topk_with_options_backward_matches_sorted_via_facade() {
    let grad_of = |sorted: bool| {
        let tape = fandhe_ai::tape();
        let x = tape.var(&t(vec![4.0, 5.0, 1.0], &[3]));
        let opts = TopkOptions::default().with_sorted(sorted);
        let (v, _) = x.topk_with_options(2, opts).unwrap();
        let loss = v.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        grads.get(&x).unwrap().unwrap().host_slice().into_owned()
    };
    assert_eq!(grad_of(false), [1.0, 1.0, 0.0]);
    assert_eq!(grad_of(true), grad_of(false));
}

#[test]
fn var_topk_with_options_sorted_matches_existing_topk() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![2.0, 9.0, 4.0, 7.0, 7.0, 1.0], &[2, 3]));
    let (v0, i0) = x.topk(2, 1, true).unwrap();
    let (v1, i1) = x.topk_with_options(2, TopkOptions::default()).unwrap();
    assert_eq!(vals(&v0), vals(&v1));
    assert_eq!(i0.host_slice().into_owned(), i1.host_slice().into_owned());
}

#[test]
fn var_unique_with_options_via_facade() {
    let tape = fandhe_ai::tape();
    // dim=None・追加出力なしは既存 Var::unique と bit 同一。
    let x = tape.var(&t(vec![3.0, 1.0, 3.0, 2.0, 1.0], &[5]));
    let base = x.unique().unwrap();
    let out = x.unique_with_options(UniqueOptions::default()).unwrap();
    assert_eq!(
        out.values.host_slice().into_owned(),
        base.host_slice().into_owned()
    );
    assert!(out.inverse.is_none() && out.counts.is_none());
    // dim=Some(-2)（行単位）に inverse／counts。
    let m = tape.var(&t(vec![1.0, 2.0, 1.0, 2.0, 0.0, 5.0], &[3, 2]));
    let opts = UniqueOptions::default()
        .with_dim(-2)
        .with_return_inverse(true)
        .with_return_counts(true);
    let out = m.unique_with_options(opts).unwrap();
    assert_eq!(out.values.shape(), &[2, 2]);
    assert_eq!(out.values.host_slice().into_owned(), [0.0, 5.0, 1.0, 2.0]);
    assert_eq!(out.inverse.unwrap().host_slice().into_owned(), [1, 1, 0]);
    assert_eq!(out.counts.unwrap().host_slice().into_owned(), [1, 2]);
}

#[test]
fn var_unique_consecutive_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 1.0, 2.0, 2.0, 1.0], &[5]));
    let opts = UniqueOptions::default()
        .with_return_inverse(true)
        .with_return_counts(true);
    let out = x.unique_consecutive(opts).unwrap();
    assert_eq!(out.values.host_slice().into_owned(), [1.0, 2.0, 1.0]);
    assert_eq!(
        out.inverse.unwrap().host_slice().into_owned(),
        [0, 0, 1, 1, 2]
    );
    assert_eq!(out.counts.unwrap().host_slice().into_owned(), [2, 2, 1]);
}

#[test]
fn var_topk_unique_ops_propagate_errors() {
    let tape = fandhe_ai::tape();
    let scalar = tape.var(&t(vec![1.0], &[]));
    let v = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    // topk の 0-d は rank 0 での dim 正規化段階で拒否される（Shape か InvalidArgument）。
    assert!(matches!(
        scalar.topk_with_options(1, TopkOptions::default()),
        Err(AutodiffError::Shape(_) | AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        scalar.unique_with_options(UniqueOptions::default()),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert!(matches!(
        scalar.unique_consecutive(UniqueOptions::default()),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    // 範囲外 dim（非負は AxisOutOfRange、`usize` で表せない負値は InvalidArgument）。
    assert!(matches!(
        v.topk_with_options(1, TopkOptions::default().with_dim(5)),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
    ));
    assert!(matches!(
        v.topk_with_options(1, TopkOptions::default().with_dim(-5)),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn var_topk_unique_ops_match_free_functions() {
    use fandhe_ai_autodiff::topk_unique_ops as free;

    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![4.0, 5.0, 1.0, 5.0], &[4]));
    let opts = TopkOptions::default().with_sorted(false);
    let (v0, i0) = x.topk_with_options(3, opts).unwrap();
    let (v1, i1) = free::topk_with_options(&x, 3, opts).unwrap();
    assert_eq!(vals(&v0), vals(&v1));
    assert_eq!(i0.host_slice().into_owned(), i1.host_slice().into_owned());

    let uo = UniqueOptions::default()
        .with_return_inverse(true)
        .with_return_counts(true);
    let a = x.unique_with_options(uo).unwrap();
    let b = free::unique_with_options(&x, uo).unwrap();
    assert_eq!(a.values.host_slice(), b.values.host_slice());
    assert_eq!(
        a.counts.unwrap().host_slice().into_owned(),
        b.counts.unwrap().host_slice().into_owned()
    );
    let a = x.unique_consecutive(uo).unwrap();
    let b = free::unique_consecutive(&x, uo).unwrap();
    assert_eq!(a.values.host_slice(), b.values.host_slice());
}
