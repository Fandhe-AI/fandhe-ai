//! facade（`fandhe_ai::Var`）経由の `advanced_indexing`／`index_put`／`index_put_`
//! 利用例と単体テスト（イシュー #2518。実装は #2148・親 #2500・ルート #2499）。
//!
//! `Var::advanced_indexing` 等は `fandhe_ai_autodiff::indexing_ops` の同名自由関数
//! への 1 行委譲メソッドで、facade は `Var` を再エクスポートするため追加の公開
//! 経路を持たない。本テストは `fandhe_ai::` のパスだけで forward・backward・
//! エラー伝播・再束縛の意味論を CPU tape 上で確認する。委譲が自由関数と bit 一致する
//! 検査だけは自由関数を直接 use する（facade の dev 依存に autodiff あり）。
//! 数値比較は完全一致（厳密に表せる値）とし、tolerance は新設しない
//! （REQ-2 の統一複合判定は GPU との比較側 `indexing_ops_backend_parity.rs`）。

use fandhe_ai::{AutodiffError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn vals(v: &fandhe_ai::Var<'_>) -> Vec<f32> {
    v.to_tensor().host_slice().into_owned()
}

#[test]
fn var_advanced_indexing_forward_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    // k = 1（先頭軸のみ。残り軸 R = [3] を素通し）。
    let r = x.advanced_indexing(&[ti(vec![1, 0], &[2])]).unwrap();
    assert_eq!(r.to_tensor().shape(), &[2, 3]);
    assert_eq!(vals(&r), [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    // k = 2（全軸。要素ごとの読み出し）。
    let r = x
        .advanced_indexing(&[ti(vec![0, 1, 1], &[3]), ti(vec![2, 0, 2], &[3])])
        .unwrap();
    assert_eq!(r.to_tensor().shape(), &[3]);
    assert_eq!(vals(&r), [3.0, 4.0, 6.0]);
    // 添字同士の broadcast（[2, 1] と [1, 2] → [2, 2]）。
    let r = x
        .advanced_indexing(&[ti(vec![0, 1], &[2, 1]), ti(vec![0, 2], &[1, 2])])
        .unwrap();
    assert_eq!(r.to_tensor().shape(), &[2, 2]);
    assert_eq!(vals(&r), [1.0, 3.0, 4.0, 6.0]);
}

#[test]
fn var_index_put_overwrite_and_accumulate_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let idx = [ti(vec![0, 2, 0], &[3])];
    let v = tape.var(&t(vec![10.0, 20.0, 30.0], &[3]));
    // overwrite: 重複添字は最後の書き手が勝つ。
    assert_eq!(
        vals(&x.index_put(&idx, &v, false).unwrap()),
        [30.0, 2.0, 20.0]
    );
    // accumulate: 重複添字は加算。
    assert_eq!(
        vals(&x.index_put(&idx, &v, true).unwrap()),
        [41.0, 2.0, 23.0]
    );
    // values のスカラー相当（[1]）は索引形状へ broadcast される。
    let s = tape.var(&t(vec![7.0], &[1]));
    assert_eq!(
        vals(&x.index_put(&idx, &s, false).unwrap()),
        [7.0, 2.0, 7.0]
    );
    // 非破壊: 元の x は変わらない。
    assert_eq!(vals(&x), [1.0, 2.0, 3.0]);
}

#[test]
fn var_index_put_in_place_rebinds_handle_via_facade() {
    let tape = fandhe_ai::tape();
    let mut x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let old = x;
    let idx = [ti(vec![1], &[1])];
    let v = tape.var(&t(vec![50.0], &[1]));
    x.index_put_(&idx, &v, false).unwrap();
    // 再束縛: 旧ハンドルは古い値のまま、x は新ノード。
    assert_eq!(vals(&old), [1.0, 2.0, 3.0]);
    assert_eq!(vals(&x), [1.0, 50.0, 3.0]);

    let w = tape.var(&t(vec![1.0, 10.0, 100.0], &[3]));
    let loss = x.mul(&w).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    // 新ノード x の勾配は w そのもの。
    let dx = grads.get(&x).unwrap().unwrap();
    assert_eq!(dx.host_slice().into_owned(), [1.0, 10.0, 100.0]);
    // 旧ノードは上書き位置の勾配が 0（index_put overwrite の VJP）。
    let dold = grads.get(&old).unwrap().unwrap();
    assert_eq!(dold.host_slice().into_owned(), [1.0, 0.0, 100.0]);
}

#[test]
fn var_indexing_backward_via_facade() {
    // advanced_indexing: 重複添字の勾配は加算される。
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let loss = x
        .advanced_indexing(&[ti(vec![0, 0, 2], &[3])])
        .unwrap()
        .sum(None)
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    assert_eq!(dx.host_slice().into_owned(), [2.0, 0.0, 1.0]);

    // index_put overwrite: 上書き位置の d_x は 0、values 側は 1。
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let v = tape.var(&t(vec![9.0], &[1]));
    let loss = x
        .index_put(&[ti(vec![1], &[1])], &v, false)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    assert_eq!(dx.host_slice().into_owned(), [1.0, 0.0, 1.0]);
    let dv = grads.get(&v).unwrap().unwrap();
    assert_eq!(dv.host_slice().into_owned(), [1.0]);
}

#[test]
fn var_indexing_methods_bit_match_free_functions() {
    use fandhe_ai_autodiff::indexing_ops as free;
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.5, -2.0, f32::NAN, 4.25, 5.0, 6.0], &[2, 3]));
    let idx = [ti(vec![1, 0, 1], &[3])];
    let v = tape.var(&t(
        vec![0.5, 0.25, 0.125, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[3, 3],
    ));
    let bits =
        |v: &fandhe_ai::Var<'_>| -> Vec<u32> { vals(v).iter().map(|f| f.to_bits()).collect() };
    assert_eq!(
        bits(&x.advanced_indexing(&idx).unwrap()),
        bits(&free::advanced_indexing(&x, &idx).unwrap())
    );
    for accumulate in [false, true] {
        assert_eq!(
            bits(&x.index_put(&idx, &v, accumulate).unwrap()),
            bits(&free::index_put(&x, &idx, &v, accumulate).unwrap())
        );
        let mut a = x;
        let mut b = x;
        a.index_put_(&idx, &v, accumulate).unwrap();
        free::index_put_(&mut b, &idx, &v, accumulate).unwrap();
        assert_eq!(bits(&a), bits(&b));
    }
}

#[test]
fn var_indexing_errors_propagate_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let v = tape.var(&t(vec![1.0], &[1]));
    // 負の添字・範囲外は InvalidArgument（wrap-around しない）。
    assert!(matches!(
        x.advanced_indexing(&[ti(vec![-1], &[1])]),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        x.advanced_indexing(&[ti(vec![2], &[1])]),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // k = 0・k > rank は InvalidArgument。
    assert!(matches!(
        x.advanced_indexing(&[]),
        Err(AutodiffError::InvalidArgument(_))
    ));
    let three = [ti(vec![0], &[1]), ti(vec![0], &[1]), ti(vec![0], &[1])];
    assert!(matches!(
        x.advanced_indexing(&three),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // values の broadcast 不能は Shape。
    let bad = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    assert!(matches!(
        x.index_put(&[ti(vec![0, 1], &[2])], &bad, false),
        Err(AutodiffError::Shape(_))
    ));
    // 別 tape の values は TapeMismatch。
    let other = fandhe_ai::tape();
    let foreign = other.var(&t(vec![1.0], &[1]));
    assert!(matches!(
        x.index_put(&[ti(vec![0], &[1])], &foreign, false),
        Err(AutodiffError::TapeMismatch)
    ));
    // 失敗した index_put_ は x を変更しない。
    let mut y = x;
    assert!(y.index_put_(&[ti(vec![-1], &[1])], &v, false).is_err());
    assert_eq!(vals(&y), [1.0, 2.0, 3.0, 4.0]);
}
