//! `fandhe_ai::Var::einsum` の batch 添字付き縮約（例 `"bij,bjk->bik"`）の
//! facade 利用例テスト（イシュー #2517。親 #2500・ルート #2499）。
//!
//! `fandhe_ai` 公開面だけを使い（`fandhe_ai_autodiff` は import しない）、
//! 公開入口 `Var::einsum` が rank≥3 `Var::matmul`（`gemm_batched`）への
//! 分解で batch 縮約を受理することを示す。新しい公開名は追加していない
//! （既存 `Var::einsum` の受理範囲の非破壊拡張。決定記録
//! `docs/autodiff-einsum-batch-decision.md` §11）。数値判定は REQ-2 統一
//! 複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`。新しい許容
//! 誤差は導入しない）。

use fandhe_ai::{AutodiffError, Tape, Tensor, Var};
use fandhe_ai_backend_cpu::parity::assert_parity;

fn data(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n).map(|i| i as f32 * scale + offset).collect()
}

fn host(v: &Var<'_>) -> Vec<f32> {
    v.to_tensor().host_slice().into_owned()
}

fn leaf<'t>(tape: &'t Tape, d: Vec<f32>, shape: &[usize]) -> Var<'t> {
    tape.var(&Tensor::new(d, shape).expect("tensor"))
}

fn assert_bits_eq(label: &str, x: &[f32], y: &[f32]) {
    assert_eq!(x.len(), y.len(), "{label}: 長さ不一致");
    assert!(
        x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
        "{label}: bit 不一致"
    );
}

/// 素朴参照: `out[o..] = sum_k a[..k] * b[k..]` を f32 `mul_add` で K 軸
/// 逐次蓄積する（FMA 契約。`coding-rust.md`）。`a_idx(o, k)`／`b_idx(o, k)`
/// は出力要素番号 `o` と縮約番号 `k` から入力の平坦添字を返す。
fn reference(
    out_len: usize,
    k_len: usize,
    a: &[f32],
    b: &[f32],
    a_idx: impl Fn(usize, usize) -> usize,
    b_idx: impl Fn(usize, usize) -> usize,
) -> Vec<f32> {
    (0..out_len)
        .map(|o| {
            let mut acc = 0.0_f32;
            for k in 0..k_len {
                acc = a[a_idx(o, k)].mul_add(b[b_idx(o, k)], acc);
            }
            acc
        })
        .collect()
}

#[test]
fn single_batch_axis_forward_matches_var_matmul() {
    let tape = fandhe_ai::tape();
    let a = leaf(&tape, data(24, 0.25, -2.0), &[2, 3, 4]);
    let b = leaf(&tape, data(40, 0.125, -1.0), &[2, 4, 5]);
    let out = Var::einsum("bij,bjk->bik", &[&a, &b]).expect("batch 縮約は受理される");
    assert_eq!(out.to_tensor().shape(), &[2, 3, 5]);
    let direct = a.matmul(&b).expect("matmul");
    assert_bits_eq("forward", &host(&out), &host(&direct));
}

#[test]
fn single_batch_axis_backward_matches_var_matmul() {
    let tape = fandhe_ai::tape();
    let a = leaf(&tape, data(24, 0.25, -2.0), &[2, 3, 4]);
    let b = leaf(&tape, data(40, 0.125, -1.0), &[2, 4, 5]);
    let via_einsum = Var::einsum("bij,bjk->bik", &[&a, &b]).expect("einsum");
    let direct = a.matmul(&b).expect("matmul");
    let g1 = tape
        .backward(&via_einsum.sum(None).expect("sum"))
        .expect("backward");
    let g2 = tape
        .backward(&direct.sum(None).expect("sum"))
        .expect("backward");
    for (label, v) in [("da", &a), ("db", &b)] {
        let x = g1.get(v).expect("get").expect("到達する");
        let y = g2.get(v).expect("get").expect("到達する");
        assert_bits_eq(
            label,
            x.contiguous().as_slice().expect("slice"),
            y.contiguous().as_slice().expect("slice"),
        );
    }
}

#[test]
fn non_leading_batch_axis_forward_matches_reference() {
    // "ibj,bjk->bik": a[i,b,j]（I=3,B=2,J=4）・b[b,j,k]（K=5）
    let (bn, i, j, k) = (2usize, 3usize, 4usize, 5usize);
    let ad = data(i * bn * j, 0.1, -1.0);
    let bd = data(bn * j * k, 0.05, -0.5);
    let tape = fandhe_ai::tape();
    let a = leaf(&tape, ad.clone(), &[i, bn, j]);
    let b = leaf(&tape, bd.clone(), &[bn, j, k]);
    let out = Var::einsum("ibj,bjk->bik", &[&a, &b]).expect("einsum");
    assert_eq!(out.to_tensor().shape(), &[bn, i, k]);
    // 出力平坦添字 o = (bb * I + ii) * K + kk
    let expected = reference(
        bn * i * k,
        j,
        &ad,
        &bd,
        |o, jj| {
            let (bb, ii) = (o / (k * i), (o / k) % i);
            (ii * bn + bb) * j + jj
        },
        |o, jj| {
            let (bb, kk) = (o / (k * i), o % k);
            (bb * j + jj) * k + kk
        },
    );
    assert_parity("ibj,bjk->bik forward", &host(&out), &expected);
}

#[test]
fn multi_batch_axes_forward_matches_reference() {
    // "abij,abjk->abik"（bit 同一は主張せず統一複合判定で比較）
    let (an, bn, i, j, k) = (2usize, 2usize, 3usize, 4usize, 2usize);
    let ad = data(an * bn * i * j, 0.07, -1.0);
    let bd = data(an * bn * j * k, 0.03, -0.4);
    let tape = fandhe_ai::tape();
    let a = leaf(&tape, ad.clone(), &[an, bn, i, j]);
    let b = leaf(&tape, bd.clone(), &[an, bn, j, k]);
    let out = Var::einsum("abij,abjk->abik", &[&a, &b]).expect("einsum");
    assert_eq!(out.to_tensor().shape(), &[an, bn, i, k]);
    let expected = reference(
        an * bn * i * k,
        j,
        &ad,
        &bd,
        |o, jj| (o / k) * j + jj,
        |o, jj| (o / (i * k)) * j * k + jj * k + o % k,
    );
    assert_parity("abij,abjk->abik forward", &host(&out), &expected);
}

#[test]
fn batch_contraction_dimension_mismatch_is_rejected() {
    let tape = fandhe_ai::tape();
    // batch 次元（2 vs 3）の不一致
    let a = leaf(&tape, data(24, 1.0, 0.0), &[2, 3, 4]);
    let b = leaf(&tape, data(60, 1.0, 0.0), &[3, 4, 5]);
    assert!(matches!(
        Var::einsum("bij,bjk->bik", &[&a, &b]),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // contract 次元（4 vs 5）の不一致
    let c = leaf(&tape, data(50, 1.0, 0.0), &[2, 5, 5]);
    assert!(matches!(
        Var::einsum("bij,bjk->bik", &[&a, &c]),
        Err(AutodiffError::InvalidArgument(_))
    ));
}
