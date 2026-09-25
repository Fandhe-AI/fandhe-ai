//! `fandhe_ai_autodiff::reduce_ops`（イシュー #2147）の NaiveOps
//! （`Tape::new()`）上での閉形式・数値検算テスト（`crates/autodiff/
//! tests/var_norm.rs` と同型の統合テスト層）。
//!
//! autodiff クレートは `backend-cpu`／`backend-cuda`／`backend-metal`
//! を dev-dependency に持たないため、3 バックエンドの parity は
//! `crates/facade/tests/reduce_ops_backend_parity.rs` 側が担う
//! （`var_norm.rs`／`var_norm_backend_parity.rs` と同じ分担）。本ファイル
//! は「NaiveOps 単体で見た数値契約が正しいか」（閉形式・`f64` 参照値との
//! 突合・bit 同一契約・中心差分による勾配検査）に責務を絞る。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::reduce_ops::{all, any, logsumexp, norm_p, prod};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

// --- prod: 閉形式・f64 参照値との突合 ---

#[test]
fn prod_matches_f64_reference_product() {
    let tape = Tape::new();
    let data = vec![1.3f32, -2.7, 0.5, 4.1];
    let x = tape.var(&t(data.clone(), &[4]));
    let out = prod(&x, None).unwrap();
    let expected = data.iter().fold(1.0f64, |acc, &v| acc * v as f64) as f32;
    assert!((out.to_tensor().host_slice()[0] - expected).abs() < 1e-4);
}

#[test]
fn prod_dim_axis_matches_row_products() {
    let tape = Tape::new();
    // [[2,3],[4,5]] -> dim=1: [6, 20]
    let x = tape.var(&t(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]));
    let out = prod(&x, Some(1)).unwrap();
    assert_eq!(out.to_tensor().host_slice().into_owned(), vec![6.0, 20.0]);
}

// --- logsumexp: 閉形式・f64 参照値との突合 ---

#[test]
fn logsumexp_matches_f64_reference() {
    let tape = Tape::new();
    let data = vec![0.1f32, -0.5, 2.3, 1.0];
    let x = tape.var(&t(data.clone(), &[4]));
    let out = logsumexp(&x, None).unwrap();
    let expected = data
        .iter()
        .fold(0.0f64, |acc, &v| acc + (v as f64).exp())
        .ln() as f32;
    assert!((out.to_tensor().host_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn logsumexp_single_element_is_identity() {
    let tape = Tape::new();
    let x = tape.var(&t(vec![3.5], &[1]));
    let out = logsumexp(&x, None).unwrap();
    assert!((out.to_tensor().host_slice()[0] - 3.5).abs() < 1e-6);
}

// --- norm_p: bit 同一契約（p=1/2 は既存 norm_l1/norm_l2 に委譲） ---

#[test]
fn norm_p_one_is_bit_identical_to_norm_l1() {
    let tape = Tape::new();
    let x = tape.var(&t(vec![-3.0, 1.5, -2.0, 4.0], &[4]));
    let a = norm_p(&x, 1.0, None).unwrap().to_tensor();
    let b = x.norm_l1(None).unwrap().to_tensor();
    assert_eq!(a.host_slice()[0].to_bits(), b.host_slice()[0].to_bits());
}

#[test]
fn norm_p_two_is_bit_identical_to_norm_l2() {
    let tape = Tape::new();
    let x = tape.var(&t(vec![-3.0, 1.5, -2.0, 4.0], &[4]));
    let a = norm_p(&x, 2.0, None).unwrap().to_tensor();
    let b = x.norm_l2(None).unwrap().to_tensor();
    assert_eq!(a.host_slice()[0].to_bits(), b.host_slice()[0].to_bits());
}

#[test]
fn norm_p_dim_axis_two_is_bit_identical_to_norm_l2_dim_axis() {
    let tape = Tape::new();
    let x = tape.var(&t(vec![3.0, 4.0, 6.0, 8.0], &[2, 2]));
    let a = norm_p(&x, 2.0, Some(1)).unwrap().to_tensor();
    let b = x.norm_l2(Some(1)).unwrap().to_tensor();
    assert_eq!(
        a.host_slice()
            .into_owned()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        b.host_slice()
            .into_owned()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
    );
}

// --- norm_p: 閉形式（p=4） ---

#[test]
fn norm_p_four_matches_f64_reference() {
    let tape = Tape::new();
    let data = vec![1.0f32, 2.0, -3.0];
    let x = tape.var(&t(data.clone(), &[3]));
    let out = norm_p(&x, 4.0, None).unwrap();
    let sum: f64 = data.iter().map(|&v| (v as f64).abs().powf(4.0)).sum();
    let expected = sum.powf(0.25) as f32;
    assert!((out.to_tensor().host_slice()[0] - expected).abs() < 1e-4);
}

// --- any / all: NaiveOps 上の閉形式 ---

#[test]
fn any_all_naive_ops_basic() {
    let tape = Tape::new();
    let x = tape.var(&t(vec![0.0, 0.0, 5.0, 0.0], &[4]));
    assert_eq!(any(&x, None).unwrap().to_tensor().host_slice()[0], 1.0);
    assert_eq!(all(&x, None).unwrap().to_tensor().host_slice()[0], 0.0);

    let y = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    assert_eq!(any(&y, None).unwrap().to_tensor().host_slice()[0], 1.0);
    assert_eq!(all(&y, None).unwrap().to_tensor().host_slice()[0], 1.0);
}

// --- 中心差分による勾配検査（NaiveOps） ---

fn central_diff_grad<F>(base: &[f32], f: F) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let eps = 1e-3f32;
    base.iter()
        .enumerate()
        .map(|(i, _)| {
            let mut plus = base.to_vec();
            plus[i] += eps;
            let mut minus = base.to_vec();
            minus[i] -= eps;
            (f(&plus) - f(&minus)) / (2.0 * eps)
        })
        .collect()
}

#[test]
fn prod_gradient_matches_central_difference_naive_ops() {
    let base = vec![1.7f32, -0.8, 2.2];
    let numeric = central_diff_grad(&base, |data| {
        let tape = Tape::new();
        let x = tape.var(&t(data.to_vec(), &[3]));
        prod(&x, None).unwrap().to_tensor().host_slice()[0]
    });

    let tape = Tape::new();
    let x = tape.var(&t(base.clone(), &[3]));
    let y = prod(&x, None).unwrap();
    let grads = tape.backward(&y).unwrap();
    let analytic = grads.get(&x).unwrap().unwrap().host_slice().into_owned();

    for i in 0..base.len() {
        assert!(
            (numeric[i] - analytic[i]).abs() < 1e-2,
            "i={i} numeric={} analytic={}",
            numeric[i],
            analytic[i]
        );
    }
}

#[test]
fn logsumexp_gradient_matches_central_difference_naive_ops() {
    let base = vec![-0.3f32, 1.1, 0.4];
    let numeric = central_diff_grad(&base, |data| {
        let tape = Tape::new();
        let x = tape.var(&t(data.to_vec(), &[3]));
        logsumexp(&x, None).unwrap().to_tensor().host_slice()[0]
    });

    let tape = Tape::new();
    let x = tape.var(&t(base.clone(), &[3]));
    let y = logsumexp(&x, None).unwrap();
    let grads = tape.backward(&y).unwrap();
    let analytic = grads.get(&x).unwrap().unwrap().host_slice().into_owned();

    for i in 0..base.len() {
        assert!(
            (numeric[i] - analytic[i]).abs() < 1e-2,
            "i={i} numeric={} analytic={}",
            numeric[i],
            analytic[i]
        );
    }
}

#[test]
fn norm_p_gradient_matches_central_difference_naive_ops() {
    let base = vec![2.0f32, -1.5, 0.7];
    let numeric = central_diff_grad(&base, |data| {
        let tape = Tape::new();
        let x = tape.var(&t(data.to_vec(), &[3]));
        norm_p(&x, 4.0, None).unwrap().to_tensor().host_slice()[0]
    });

    let tape = Tape::new();
    let x = tape.var(&t(base.clone(), &[3]));
    let y = norm_p(&x, 4.0, None).unwrap();
    let grads = tape.backward(&y).unwrap();
    let analytic = grads.get(&x).unwrap().unwrap().host_slice().into_owned();

    for i in 0..base.len() {
        assert!(
            (numeric[i] - analytic[i]).abs() < 1e-2,
            "i={i} numeric={} analytic={}",
            numeric[i],
            analytic[i]
        );
    }
}

// --- 共通: 範囲外 dim・空縮約はエラー ---

#[test]
fn out_of_range_dim_and_empty_reduction_are_errors() {
    let tape = Tape::new();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    assert!(prod(&x, Some(9)).is_err());
    assert!(logsumexp(&x, Some(9)).is_err());
    assert!(norm_p(&x, 3.0, Some(9)).is_err());

    let empty = tape.var(&t(vec![], &[0]));
    assert!(logsumexp(&empty, None).is_err());
    assert!(norm_p(&empty, 3.0, None).is_err());
    assert_eq!(prod(&empty, None).unwrap().to_tensor().host_slice()[0], 1.0);
    assert_eq!(any(&empty, None).unwrap().to_tensor().host_slice()[0], 0.0);
    assert_eq!(all(&empty, None).unwrap().to_tensor().host_slice()[0], 1.0);
}
