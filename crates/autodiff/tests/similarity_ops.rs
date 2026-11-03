//! `Var::cosine_similarity`／`Var::pairwise_distance`（イシュー #3821・親 #3813。設計記録
//! `docs/autodiff-similarity-distance-decision.md` §4）の勾配突合テスト。
//!
//! 解析勾配（`Tape::backward`）を、テスト内の f64 参照実装に対する中央差分と REQ-2 統一複合判定
//! （`common::req2_close`。tolerance は新設しない）で突合する。クランプ境界（`‖x‖ ≈ eps`）と
//! 絶対値の kink（差が 0）は避けた点だけで比べる。クランプ内（`0 < ‖x‖ < eps`）の勾配は
//! 中央差分では **straight-through と一致しない**（クランプの真の微分は第 2 項が 0）ため、
//! 手計算の straight-through 値と照合する。

mod common;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致")
}

/// f64 の cosine 参照（`(outer, len, inner)` 分解。出力は (outer, inner) 順）。
fn cosine_ref(x1: &[f64], x2: &[f64], shape: &[usize], dim: usize, eps: f64) -> Vec<f64> {
    let outer: usize = shape[..dim].iter().product();
    let len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let mut out = Vec::new();
    for o in 0..outer {
        for i in 0..inner {
            let at = |k: usize| o * len * inner + k * inner + i;
            let n1 = (0..len).map(|k| x1[at(k)] * x1[at(k)]).sum::<f64>().sqrt();
            let n2 = (0..len).map(|k| x2[at(k)] * x2[at(k)]).sum::<f64>().sqrt();
            let (c1, c2) = (n1.max(eps), n2.max(eps));
            out.push((0..len).map(|k| (x1[at(k)] / c1) * (x2[at(k)] / c2)).sum());
        }
    }
    out
}

/// f64 の pairwise 参照（最後の軸を縮約）。
fn pairwise_ref(x1: &[f64], x2: &[f64], d: usize, p: f64, eps: f64) -> Vec<f64> {
    x1.chunks(d)
        .zip(x2.chunks(d))
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(u, v)| (u - v + eps).abs().powf(p))
                .sum::<f64>()
                .powf(1.0 / p)
        })
        .collect()
}

/// f64 参照実装（x1, x2 -> 出力）の型。
type RefFn<'a> = dyn Fn(&[f64], &[f64]) -> Vec<f64> + 'a;

/// 重み付き和 `Σ c_j · y_j` の中央差分勾配（x1 側・x2 側）。
fn numeric_grad(f: &RefFn, x1: &[f64], x2: &[f64], c: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let h = 1e-6;
    let loss = |a: &[f64], b: &[f64]| -> f64 { f(a, b).iter().zip(c).map(|(y, w)| y * w).sum() };
    let mut g1 = vec![0.0; x1.len()];
    let mut g2 = vec![0.0; x2.len()];
    for i in 0..x1.len() {
        let (mut p, mut m) = (x1.to_vec(), x1.to_vec());
        p[i] += h;
        m[i] -= h;
        g1[i] = (loss(&p, x2) - loss(&m, x2)) / (2.0 * h);
        let (mut p, mut m) = (x2.to_vec(), x2.to_vec());
        p[i] += h;
        m[i] -= h;
        g2[i] = (loss(x1, &p) - loss(x1, &m)) / (2.0 * h);
    }
    (g1, g2)
}

fn weights(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 + (i as f32 * 0.9).cos() * 0.4).collect()
}

#[test]
fn cosine_gradient_matches_central_difference_for_each_dim() {
    let shape = [2usize, 3, 4];
    let n: usize = shape.iter().product();
    let a = common::det_seq(n, 0.3);
    let b = common::det_seq(n, 5.1);
    for dim in 0..3 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let va = tape.var(&t(a.clone(), &shape));
        let vb = tape.var(&t(b.clone(), &shape));
        let y = va.cosine_similarity(&vb, dim, 1e-8).unwrap();
        let y_shape = y.value().shape().to_vec();
        let out_n: usize = y_shape.iter().product();
        let c = weights(out_n);
        let w = tape.var_no_grad(&t(c.clone(), &y_shape));
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let g = tape.backward(&loss).unwrap();
        let af: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let bf: Vec<f64> = b.iter().map(|&v| v as f64).collect();
        let cf: Vec<f64> = c.iter().map(|&v| v as f64).collect();
        let (n1, n2) = numeric_grad(&|p, q| cosine_ref(p, q, &shape, dim, 1e-8), &af, &bf, &cf);
        common::assert_all_req2_close(
            &g.get(&va).unwrap().unwrap().host_slice(),
            &n1,
            &format!("cosine dim={dim} dx1"),
        );
        common::assert_all_req2_close(
            &g.get(&vb).unwrap().unwrap().host_slice(),
            &n2,
            &format!("cosine dim={dim} dx2"),
        );
    }
}

#[test]
fn pairwise_gradient_matches_central_difference() {
    let (rows, d) = (3usize, 4usize);
    let a = common::det_seq(rows * d, 0.7);
    let b = common::det_seq(rows * d, 9.2);
    for p in [1.0f32, 2.0, 3.0] {
        for keepdim in [false, true] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let va = tape.var(&t(a.clone(), &[rows, d]));
            let vb = tape.var(&t(b.clone(), &[rows, d]));
            let y = va.pairwise_distance(&vb, p, 1e-6, keepdim).unwrap();
            let y_shape = y.value().shape().to_vec();
            assert_eq!(y_shape, if keepdim { vec![rows, 1] } else { vec![rows] });
            let c = weights(rows);
            let w = tape.var_no_grad(&t(c.clone(), &y_shape));
            let loss = y.mul(&w).unwrap().sum(None).unwrap();
            let g = tape.backward(&loss).unwrap();
            let af: Vec<f64> = a.iter().map(|&v| v as f64).collect();
            let bf: Vec<f64> = b.iter().map(|&v| v as f64).collect();
            let cf: Vec<f64> = c.iter().map(|&v| v as f64).collect();
            let (n1, n2) =
                numeric_grad(&|x, z| pairwise_ref(x, z, d, p as f64, 1e-6), &af, &bf, &cf);
            let ctx = format!("pairwise p={p} keepdim={keepdim}");
            common::assert_all_req2_close(
                &g.get(&va).unwrap().unwrap().host_slice(),
                &n1,
                &format!("{ctx} dx1"),
            );
            common::assert_all_req2_close(
                &g.get(&vb).unwrap().unwrap().host_slice(),
                &n2,
                &format!("{ctx} dx2"),
            );
        }
    }
}

/// `0 < ‖x1‖ < eps` の straight-through 勾配（クランプの真の微分と区別するため手計算で照合する）。
#[test]
fn cosine_straight_through_gradient_inside_clamp() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![0.3, 0.4], &[2])); // ‖a‖ = 0.5 < eps
    let b = tape.var(&t(vec![1.0, 2.0], &[2])); // ‖b‖ = sqrt5 > eps
    let eps = 1.0f32;
    let y = a.cosine_similarity(&b, 0, eps).unwrap();
    let g = tape.backward(&y).unwrap();
    let (n1, c1, c2) = (0.5f64, 1.0f64, 5f64.sqrt());
    let s = (0.3 * 1.0 + 0.4 * 2.0) / (c1 * c2);
    let want = [
        1.0 / (c1 * c2) - s * 0.3 / (c1 * n1),
        2.0 / (c1 * c2) - s * 0.4 / (c1 * n1),
    ];
    common::assert_all_req2_close(&g.get(&a).unwrap().unwrap().host_slice(), &want, "dx1");
    // クランプの真の微分（第 2 項 0）とは異なることを確かめる。
    let true_clamp = [1.0 / (c1 * c2), 2.0 / (c1 * c2)];
    assert!(
        (g.get(&a).unwrap().unwrap().host_slice()[0] as f64 - true_clamp[0]).abs() > 1e-3,
        "straight-through 勾配はクランプの真の微分と異なる"
    );
}
