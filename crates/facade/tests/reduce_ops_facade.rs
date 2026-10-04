//! facade（`fandhe_ai::Var`）経由の `prod`／`logsumexp`／`any`／`all`／`norm_p`
//! 利用例と単体テスト（イシュー #2514。実装は #2147・親 #2500・ルート #2499）。
//!
//! 利用例: `let y = x.prod(None)?;`・`let n = x.norm_p(2.0, Some(0))?;`。
//! `Var::prod` 等は `fandhe_ai_autodiff::reduce_ops` の同名自由関数への 1 行
//! 委譲メソッドで、facade は `Var` を再エクスポートするため追加の公開経路を
//! 持たない。本テストは `fandhe_ai::` のパスだけで forward・backward・エラー
//! 伝播を CPU tape 上で確認し、委譲が自由関数と bit 一致することだけは
//! 自由関数を直接 use して検査する（facade の dev 依存に autodiff あり）。
//! 数値比較は厳密に表せる値の完全一致とし、tolerance は新設しない
//! （GPU との比較は `reduce_ops_backend_parity.rs`）。

use fandhe_ai::{AutodiffError, Tensor, Var};
use fandhe_ai_autodiff::reduce_ops as free;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn vals(v: &Var<'_>) -> Vec<f32> {
    v.to_tensor().host_slice().into_owned()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// 勾配 `d sum(f(x)) / dx` を取り出す。
fn grad_of(
    f: impl for<'t> Fn(&Var<'t>) -> Result<Var<'t>, AutodiffError>,
    data: Vec<f32>,
) -> Vec<f32> {
    let tape = fandhe_ai::tape();
    let n = data.len();
    let x = tape.var(&t(data, &[n]));
    let loss = f(&x).expect("forward").sum(None).expect("sum");
    let grads = tape.backward(&loss).expect("backward");
    grads
        .get(&x)
        .expect("grad")
        .expect("grad some")
        .host_slice()
        .into_owned()
}

#[test]
fn prod_forward_all_and_axis() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    assert_eq!(vals(&x.prod(None).expect("prod")), [720.0]);
    assert_eq!(vals(&x.prod(Some(1)).expect("prod")), [6.0, 120.0]);
    assert_eq!(vals(&x.prod(Some(0)).expect("prod")), [4.0, 10.0, 18.0]);
}

#[test]
fn prod_backward_is_product_of_others() {
    let g = grad_of(|x| x.prod(None), vec![2.0, 3.0, 4.0]);
    assert_eq!(g, [12.0, 8.0, 6.0]);
}

#[test]
fn logsumexp_forward_and_gradient_is_softmax() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![0.0, 0.0], &[2]));
    let lse = vals(&x.logsumexp(None).expect("logsumexp"))[0];
    assert!((lse - std::f32::consts::LN_2).abs() < 1e-6, "lse={lse}");
    let g = grad_of(|x| x.logsumexp(None), vec![1.0, 1.0]);
    assert_eq!(g, [0.5, 0.5]);
}

#[test]
fn any_all_masks_and_zero_gradient() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![0.0, -0.0, f32::NAN, 1.0], &[4]));
    assert_eq!(vals(&x.any(None).expect("any")), [1.0]);
    assert_eq!(vals(&x.all(None).expect("all")), [0.0]);
    let zeros = tape.var(&t(vec![0.0, -0.0], &[2]));
    assert_eq!(vals(&zeros.any(None).expect("any")), [0.0]);
    let ones = tape.var(&t(vec![1.0, f32::NAN], &[2]));
    assert_eq!(vals(&ones.all(None).expect("all")), [1.0]);
    assert_eq!(grad_of(|x| x.any(None), vec![1.0, 0.0]), [0.0, 0.0]);
    assert_eq!(grad_of(|x| x.all(None), vec![1.0, 2.0]), [0.0, 0.0]);
}

#[test]
fn norm_p_exact_values_and_gradient() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![3.0, 4.0], &[2]));
    assert_eq!(vals(&x.norm_p(2.0, None).expect("norm_p")), [5.0]);
    assert_eq!(vals(&x.norm_p(1.0, None).expect("norm_p")), [7.0]);
    let g = grad_of(|x| x.norm_p(2.0, None), vec![3.0, 4.0]);
    assert!(
        (g[0] - 0.6).abs() < 1e-6 && (g[1] - 0.8).abs() < 1e-6,
        "g={g:?}"
    );
}

#[test]
fn delegation_is_bit_identical_to_free_functions() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![0.5, -1.5, 2.0, 3.25, -0.75, 1.0], &[2, 3]));
    for dim in [None, Some(0), Some(1)] {
        assert_eq!(
            bits(&vals(&x.prod(dim).expect("m"))),
            bits(&vals(&free::prod(&x, dim).expect("f")))
        );
        assert_eq!(
            bits(&vals(&x.logsumexp(dim).expect("m"))),
            bits(&vals(&free::logsumexp(&x, dim).expect("f")))
        );
        assert_eq!(
            bits(&vals(&x.any(dim).expect("m"))),
            bits(&vals(&free::any(&x, dim).expect("f")))
        );
        assert_eq!(
            bits(&vals(&x.all(dim).expect("m"))),
            bits(&vals(&free::all(&x, dim).expect("f")))
        );
        assert_eq!(
            bits(&vals(&x.norm_p(2.0, dim).expect("m"))),
            bits(&vals(&free::norm_p(&x, 2.0, dim).expect("f")))
        );
    }
}

#[test]
fn errors_propagate_through_delegation() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    assert!(x.prod(Some(5)).is_err());
    assert!(x.logsumexp(Some(5)).is_err());
    assert!(x.any(Some(5)).is_err());
    assert!(x.all(Some(5)).is_err());
    assert!(x.norm_p(2.0, Some(5)).is_err());
    for p in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            matches!(x.norm_p(p, None), Err(AutodiffError::InvalidArgument(_))),
            "p={p}"
        );
    }
    let empty = tape.var(&t(vec![], &[0]));
    assert!(matches!(
        empty.logsumexp(None),
        Err(AutodiffError::InvalidArgument(_))
    ));
}
