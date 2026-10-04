//! facade（`fandhe_ai::Var`）経由の `amax`／`amin` 利用例と単体テスト
//! （イシュー #2514。実装は #2154・親 #2500・ルート #2499）。
//!
//! 利用例: `let m = x.amax(Some(1))?;`。`Var::amax`／`amin` は
//! `fandhe_ai_autodiff::extremum_ops` の同名自由関数への 1 行委譲メソッドで、
//! 同値タイに勾配を `g/k` で均等分配する（`Var::max`／`min` は先勝ち）。
//! 数値比較は厳密に表せる値の完全一致とし、tolerance は新設しない
//! （GPU との比較は `extremum_ops_backend_parity.rs`）。

use fandhe_ai::{AutodiffError, Tensor, Var};
use fandhe_ai_autodiff::extremum_ops as free;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn vals(v: &Var<'_>) -> Vec<f32> {
    v.to_tensor().host_slice().into_owned()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

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
fn forward_matches_var_max_min_bitwise() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![3.0, 1.0, 3.0, -2.0, 0.5, 7.0], &[2, 3]));
    for dim in [None, Some(0), Some(1)] {
        assert_eq!(
            bits(&vals(&x.amax(dim).expect("amax"))),
            bits(&vals(&x.max(dim).expect("max")))
        );
        assert_eq!(
            bits(&vals(&x.amin(dim).expect("amin"))),
            bits(&vals(&x.min(dim).expect("min")))
        );
    }
}

#[test]
fn tie_gradient_is_distributed_evenly() {
    assert_eq!(
        grad_of(|x| x.amax(None), vec![3.0, 3.0, 1.0]),
        [0.5, 0.5, 0.0]
    );
    let third = 1.0_f32 / 3.0;
    let g = grad_of(|x| x.amin(None), vec![1.0, 2.0, 1.0, 1.0]);
    assert_eq!(g, [third, 0.0, third, third]);
}

#[test]
fn no_tie_matches_first_wins() {
    let data = vec![1.0, 5.0, 2.0];
    let a = grad_of(|x| x.amax(None), data.clone());
    let b = grad_of(|x| x.max(None), data);
    assert_eq!(a, b);
    assert_eq!(a, [0.0, 1.0, 0.0]);
}

#[test]
fn var_max_gradient_stays_first_wins() {
    assert_eq!(
        grad_of(|x| x.max(None), vec![3.0, 3.0, 1.0]),
        [1.0, 0.0, 0.0]
    );
    assert_eq!(
        grad_of(|x| x.min(None), vec![1.0, 2.0, 1.0]),
        [1.0, 0.0, 0.0]
    );
}

#[test]
fn delegation_is_bit_identical_to_free_functions() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![0.5, -1.5, 2.0, 3.25, -0.75, 1.0], &[2, 3]));
    for dim in [None, Some(0), Some(1)] {
        assert_eq!(
            bits(&vals(&x.amax(dim).expect("m"))),
            bits(&vals(&free::amax(&x, dim).expect("f")))
        );
        assert_eq!(
            bits(&vals(&x.amin(dim).expect("m"))),
            bits(&vals(&free::amin(&x, dim).expect("f")))
        );
    }
}

#[test]
fn errors_propagate_through_delegation() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    assert!(x.amax(Some(5)).is_err());
    assert!(x.amin(Some(5)).is_err());
}
