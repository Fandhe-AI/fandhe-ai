//! facade（`fandhe_ai::Var`）経由の `flip`／`roll`／`repeat`／`tile` 利用例と
//! 単体テスト（イシュー #2511。実装は #2143・親 #2500・ルート #2499）。
//!
//! `Var::flip` 等は `fandhe_ai_autodiff::rearrange_ops` の同名自由関数への
//! 1 行委譲メソッドで、facade は `Var` を再エクスポートするため追加の公開
//! 経路を持たない。本テストは `fandhe_ai::` のパスだけで forward・backward・
//! エラー伝播を CPU tape 上で確認する。委譲が自由関数と bit 一致する検査
//! だけは自由関数を直接 use する（facade の dev 依存に autodiff あり）。
//! 数値比較は完全一致（厳密に表せる値）とし、tolerance は新設しない
//! （REQ-2 の統一複合判定は GPU との比較側 `rearrange_ops_backend_parity.rs`）。

use fandhe_ai::{AutodiffError, ShapeError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn vals(v: &fandhe_ai::Var<'_>) -> Vec<f32> {
    v.to_tensor().host_slice().into_owned()
}

#[test]
fn var_flip_forward_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    assert_eq!(vals(&x.flip(&[1]).unwrap()), [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    assert_eq!(
        vals(&x.flip(&[0, 1]).unwrap()),
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    );
    assert_eq!(vals(&x.flip(&[]).unwrap()), vals(&x));
}

#[test]
fn var_roll_forward_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
    assert_eq!(
        vals(&x.roll(&[2], &[0]).unwrap()),
        [4.0, 5.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(
        vals(&x.roll(&[-1], &[0]).unwrap()),
        [2.0, 3.0, 4.0, 5.0, 1.0]
    );
    let m = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let r = m.roll(&[1, 1], &[0, 1]).unwrap();
    assert_eq!(r.to_tensor().shape(), &[2, 3]);
    assert_eq!(vals(&r), [6.0, 4.0, 5.0, 3.0, 1.0, 2.0]);
}

#[test]
fn var_repeat_forward_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let r = x.repeat(&[2, 1]).unwrap();
    assert_eq!(r.to_tensor().shape(), &[4, 2]);
    assert_eq!(vals(&r), [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    // rank を広げる repeat（先頭に新軸）。
    let w = x.repeat(&[2, 1, 2]).unwrap();
    assert_eq!(w.to_tensor().shape(), &[2, 2, 4]);
    assert_eq!(
        vals(&w),
        [
            1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0
        ]
    );
}

#[test]
fn var_tile_forward_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    // reps.len() < rank は先頭を 1 で埋める（torch.tile 互換）。
    let r = x.tile(&[2]).unwrap();
    assert_eq!(r.to_tensor().shape(), &[2, 4]);
    assert_eq!(vals(&r), [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
}

#[test]
fn var_flip_repeat_backward_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let w = tape.var(&t(vec![10.0, 20.0, 30.0, 40.0], &[4]));
    let loss = x.flip(&[0]).unwrap().mul(&w).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    assert_eq!(dx.host_slice().into_owned(), [40.0, 30.0, 20.0, 10.0]);

    // repeat: 各入力要素の勾配は繰り返し数（ここでは 3）倍（整数値のため厳密）。
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let loss = x.repeat(&[3]).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    assert_eq!(dx.host_slice().into_owned(), [3.0, 3.0]);
}

#[test]
fn var_rearrange_ops_match_free_functions_bitwise() {
    use fandhe_ai_autodiff::rearrange_ops as free;
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.5, -2.0, f32::NAN, 4.25, 5.0, 6.0], &[2, 3]));
    let bits =
        |v: &fandhe_ai::Var<'_>| -> Vec<u32> { vals(v).iter().map(|f| f.to_bits()).collect() };
    assert_eq!(
        bits(&x.flip(&[1]).unwrap()),
        bits(&free::flip(&x, &[1]).unwrap())
    );
    assert_eq!(
        bits(&x.roll(&[1], &[1]).unwrap()),
        bits(&free::roll(&x, &[1], &[1]).unwrap())
    );
    assert_eq!(
        bits(&x.repeat(&[2, 2]).unwrap()),
        bits(&free::repeat(&x, &[2, 2]).unwrap())
    );
    assert_eq!(
        bits(&x.tile(&[2]).unwrap()),
        bits(&free::tile(&x, &[2]).unwrap())
    );
}

#[test]
fn var_rearrange_ops_propagate_errors() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    assert!(matches!(
        x.flip(&[2]),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
    ));
    assert!(matches!(
        x.flip(&[0, 0]),
        Err(AutodiffError::Shape(ShapeError::DuplicateAxis { .. }))
    ));
    assert!(matches!(
        x.roll(&[1, 2], &[0]),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        x.repeat(&[2]),
        Err(AutodiffError::InvalidArgument(_))
    ));
}
