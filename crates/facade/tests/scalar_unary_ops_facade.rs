//! facade（`fandhe_ai::Var`）経由の `floor`／`ceil`／`round`／`sign`／
//! `reciprocal`／`rsqrt`／`erf`／`pow_scalar` 利用例と単体テスト
//! （イシュー #2512。実装は #2145・親 #2500・ルート #2499）。
//!
//! `Var::floor` 等は `fandhe_ai_autodiff::scalar_unary_ops` の同名自由関数への
//! 1 行委譲メソッドで、facade は `Var` を再エクスポートするため追加の公開
//! 経路を持たない。本テストは `fandhe_ai::` のパスだけで forward・backward・
//! エラー伝播を CPU tape 上で確認する。委譲が自由関数と bit 一致する検査
//! だけは自由関数を直接 use する（facade の dev 依存に autodiff あり）。
//! 数値比較は完全一致（厳密に表せる値）とし、tolerance は新設しない
//! （REQ-2 の統一複合判定は GPU との比較側 `scalar_unary_ops_backend_parity.rs`）。

use fandhe_ai::{AutodiffError, Tensor, Var};
use fandhe_ai_autodiff::scalar_unary_ops as free;

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
fn var_piecewise_forward_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![-1.5, 2.5, 0.5, -0.0], &[4]));
    assert_eq!(vals(&x.floor().unwrap()), [-2.0, 2.0, 0.0, -0.0]);
    assert_eq!(vals(&x.ceil().unwrap()), [-1.0, 3.0, 1.0, -0.0]);
    // 偶数丸め
    assert_eq!(vals(&x.round().unwrap()), [-2.0, 2.0, 0.0, -0.0]);
}

#[test]
fn var_sign_forward_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![3.0, -3.0, 0.0, -0.0, f32::NAN], &[5]));
    let y = vals(&x.sign().unwrap());
    assert_eq!(&y[..4], &[1.0, -1.0, 0.0, 0.0]);
    assert!(y[4].is_nan());
}

#[test]
fn var_reciprocal_rsqrt_pow_forward_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![0.5, 4.0, 0.0], &[3]));
    let r = vals(&x.reciprocal().unwrap());
    assert_eq!(r[0], 2.0);
    assert_eq!(r[1], 0.25);
    assert_eq!(r[2], f32::INFINITY);
    assert_eq!(vals(&x.rsqrt().unwrap())[1], 0.5);
    let neg = tape.var(&t(vec![-1.0], &[1]));
    assert!(vals(&neg.rsqrt().unwrap())[0].is_nan());
    let p = tape.var(&t(vec![2.0], &[1]));
    assert_eq!(vals(&p.pow_scalar(3.0).unwrap()), [8.0]);
}

#[test]
fn var_methods_match_free_functions_bitwise() {
    let data = vec![
        -2.5_f32,
        -1.0,
        -0.0,
        0.0,
        0.5,
        1.5,
        4.0,
        f32::INFINITY,
        f32::NAN,
    ];
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(data.clone(), &[data.len()]));
    let pairs: [(&str, Var<'_>, Var<'_>); 7] = [
        ("floor", x.floor().unwrap(), free::floor(&x).unwrap()),
        ("ceil", x.ceil().unwrap(), free::ceil(&x).unwrap()),
        ("round", x.round().unwrap(), free::round(&x).unwrap()),
        ("sign", x.sign().unwrap(), free::sign(&x).unwrap()),
        (
            "reciprocal",
            x.reciprocal().unwrap(),
            free::reciprocal(&x).unwrap(),
        ),
        ("rsqrt", x.rsqrt().unwrap(), free::rsqrt(&x).unwrap()),
        ("erf", x.erf().unwrap(), free::erf(&x).unwrap()),
    ];
    for (name, m, f) in &pairs {
        assert_eq!(bits(&vals(m)), bits(&vals(f)), "{name}");
    }
    assert_eq!(
        bits(&vals(&x.pow_scalar(2.0).unwrap())),
        bits(&vals(&free::pow_scalar(&x, 2.0).unwrap()))
    );
}

#[test]
fn var_piecewise_backward_is_zero_even_with_infinite_upstream() {
    let data = vec![-1.5_f32, 0.5, 2.5, 3.0];
    for which in 0..4 {
        let tape = fandhe_ai::tape();
        let x = tape.var(&t(data.clone(), &[4]));
        let y = match which {
            0 => x.floor(),
            1 => x.ceil(),
            2 => x.round(),
            _ => x.sign(),
        }
        .unwrap();
        // upstream に inf を流しても `0 * inf = NaN` にならない。
        let up = tape.var(&t(vec![f32::INFINITY; 4], &[4]));
        let loss = y.mul(&up).unwrap().sum(None).unwrap();
        let g = tape.backward(&loss).unwrap();
        let gx = g
            .get(&x)
            .unwrap()
            .expect("grad some")
            .host_slice()
            .into_owned();
        assert_eq!(gx, [0.0; 4]);
    }
}

#[test]
fn var_smooth_backward_via_facade() {
    assert_eq!(grad_of(|x| x.reciprocal(), vec![0.5]), [-4.0]);
    assert_eq!(grad_of(|x| x.rsqrt(), vec![4.0]), [-0.0625]);
    assert_eq!(grad_of(|x| x.pow_scalar(3.0), vec![2.0]), [12.0]);
    assert_eq!(grad_of(|x| x.pow_scalar(0.0), vec![2.0]), [0.0]);
    let data = vec![-0.5_f32, 0.0, 0.75];
    let via_method = grad_of(|x| x.erf(), data.clone());
    let via_free = grad_of(free::erf, data);
    assert_eq!(bits(&via_method), bits(&via_free));
}

#[test]
fn var_scalar_unary_cross_tape_error_propagates() {
    // 別 tape の Var と合成すると `AutodiffError` が返る（委譲先の検査が生きている）。
    let tape_a = fandhe_ai::tape();
    let tape_b = fandhe_ai::tape();
    let a = tape_a.var(&t(vec![1.0, 2.0], &[2]));
    let b = tape_b.var(&t(vec![1.0, 2.0], &[2]));
    let y = a.floor().unwrap();
    assert!(y.add(&b).is_err());
}
