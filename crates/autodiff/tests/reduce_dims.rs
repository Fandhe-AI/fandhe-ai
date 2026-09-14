//! `Var::sum_dims`／`max_dims`／`mean`／`mean_dims`（複数軸・`keepdim`
//! 対応の縮約。イシュー #1719・親 #1601「Phase 2（Tier 1）」）の統合
//! テスト。
//!
//! - ブルートフォース（多重ループ）参照実装との forward 突合
//!   （連続・非連続 `dims` × `keepdim` on/off）。
//! - `sum_dims(&[d], false)` が `sum(Some(d))` と**bit 同一**になる
//!   契約（`crate::reduce_dims::merge_for_reduction` の単一軸直接委譲
//!   分岐）。
//! - 非連続 `dims` が `permute`／`contiguous` を経由すること（`Tape::len()`
//!   によるノード数の間接確認。`crate::einsum` の統合テストと同型）。
//! - 中央差分（数値微分）との勾配突合。
//! - `max_dims` の同値タイの決定性（run-to-run 同一）。
//! - 空テンソル・拒否系（重複・範囲外・空 `dims`・`mean` の `n == 0`）。
//! - `Tape::checkpoint` 経由での `Op::Mean` 再計算が forward と bit 同一。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

const H: f64 = 1e-3;
const TAU: f32 = 1e-4;
const REL_TOL: f32 = 1e-2;
const ABS_TOL: f32 = 1e-3;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    let c = tensor.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

fn assert_tensor_close(label: &str, analytic: &Tensor<f32>, numeric: &Tensor<f32>) {
    assert_eq!(
        analytic.shape(),
        numeric.shape(),
        "{label}: shape が一致しない"
    );
    let shape = analytic.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut index = vec![0usize; shape.len()];
    for flat in 0..numel {
        let av = analytic.get(&index).unwrap_or(0.0);
        let nv = numeric.get(&index).unwrap_or(0.0);
        let diff = (av - nv).abs();
        let rel = diff / av.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[{flat:?} idx={index:?}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
        );
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

/// ブルートフォース参照実装（多重ループ）による複数軸縮約。`op` は
/// `sum`／`max` いずれかの二項結合子、`init` は単位元。`torch.sum
/// (dim=[...])`／`torch.amax(dim=[...])` と同じ意味論（`keepdim` は
/// 呼び出し元が別途 reshape する）。
fn brute_reduce(
    input: &Tensor<f32>,
    dims: &[usize],
    op: impl Fn(f32, f32) -> f32,
    init: f32,
) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    let rank = shape.len();
    let reduced: std::collections::HashSet<usize> = dims.iter().copied().collect();
    let kept_axes: Vec<usize> = (0..rank).filter(|a| !reduced.contains(a)).collect();
    let kept_shape: Vec<usize> = kept_axes.iter().map(|&a| shape[a]).collect();
    let out_numel: usize = kept_shape.iter().product::<usize>().max(1);
    let mut out = vec![init; out_numel];

    let numel: usize = shape.iter().product();
    let mut idx = vec![0usize; rank];
    for _ in 0..numel {
        let v = input.get(&idx).unwrap_or(0.0);
        let out_idx: Vec<usize> = kept_axes.iter().map(|&a| idx[a]).collect();
        let mut flat = 0usize;
        for (k, &size) in kept_shape.iter().enumerate() {
            flat = flat * size + out_idx[k];
        }
        out[flat] = op(out[flat], v);
        for axis in (0..rank).rev() {
            idx[axis] += 1;
            if idx[axis] < shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
    if kept_shape.is_empty() {
        Tensor::new(out, &[]).expect("test fixture: スカラー shape")
    } else {
        Tensor::new(out, &kept_shape).expect("test fixture: kept_shape と要素数は一致するはず")
    }
}

fn insert_ones(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    // `brute_reduce` の出力（縮約軸を除去した shape）へ `keepdim` の
    // サイズ 1 軸を挿入し直す（`Var::sum_dims` 等の `keepdim_shape` と
    // 同じ意味）。
    let mut sorted = dims.to_vec();
    sorted.sort_unstable();
    let mut out = shape.to_vec();
    for &axis in &sorted {
        out[axis] = 1;
    }
    out
}

// --- 1. forward 突合（ブルートフォース） ---

#[test]
fn sum_dims_matches_brute_force_non_contiguous_dims() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t(
        (0..(2 * 3 * 4 * 5))
            .map(|v| (v as f32) * 0.1 - 3.0)
            .collect(),
        &[2, 3, 4, 5],
    );
    let xv = tape.var(&x);
    let dims = [1usize, 2usize]; // kept=[0,3]（非連続。permute 必須）
    let got = xv.sum_dims(&dims, false).unwrap();
    let expect = brute_reduce(&x, &dims, |a, b| a + b, 0.0);
    assert_tensor_close("sum_dims non-contig", &got.to_tensor(), &expect);
}

#[test]
fn sum_dims_matches_brute_force_trailing_dims_keepdim() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t(
        (0..(2 * 3 * 4)).map(|v| v as f32 * 0.5 - 2.0).collect(),
        &[2, 3, 4],
    );
    let xv = tape.var(&x);
    let dims = [1usize, 2usize]; // kept=[0]（先頭）・恒等順列で perm 不要
    let got = xv.sum_dims(&dims, true).unwrap();
    let expect_squeezed = brute_reduce(&x, &dims, |a, b| a + b, 0.0);
    let expect_shape = insert_ones(&[2, 3, 4], &dims);
    let expect = expect_squeezed
        .reshape(&expect_shape)
        .expect("test fixture: 要素数は一致するはず");
    assert_tensor_close("sum_dims keepdim", &got.to_tensor(), &expect);
    assert_eq!(got.to_tensor().shape(), expect_shape);
}

#[test]
fn max_dims_matches_brute_force() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t(
        vec![1.0, 5.0, 2.0, 5.0, 3.0, -1.0, 0.0, 5.0, 4.0, 4.0, 4.0, 4.0],
        &[2, 2, 3],
    );
    let xv = tape.var(&x);
    let dims = [0usize, 2usize];
    let got = xv.max_dims(&dims, false).unwrap();
    let expect = brute_reduce(&x, &dims, f32::max, f32::NEG_INFINITY);
    assert_tensor_close("max_dims", &got.to_tensor(), &expect);
}

#[test]
fn mean_dims_matches_sum_dims_divided_by_count() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t((0..24).map(|v| v as f32).collect(), &[2, 3, 4]);
    let xv = tape.var(&x);
    let dims = [0usize, 2usize];
    let got = xv.mean_dims(&dims, false).unwrap();
    let sum = xv.sum_dims(&dims, false).unwrap();
    let count = (2 * 4) as f32;
    let expect: Vec<f32> = dense(&sum.to_tensor())
        .into_iter()
        .map(|v| v / count)
        .collect();
    let expect = t(expect, sum.to_tensor().shape());
    assert_tensor_close("mean_dims", &got.to_tensor(), &expect);
}

// --- 2. 単一軸の bit 同一契約 ---

#[test]
fn sum_dims_single_axis_is_bit_identical_to_sum() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t((0..24).map(|v| v as f32 * 0.37 - 1.0).collect(), &[2, 3, 4]);
    let xv = tape.var(&x);
    let via_dims = xv.sum_dims(&[1], false).unwrap().to_tensor();
    let via_single = xv.sum(Some(1)).unwrap().to_tensor();
    assert_eq!(dense(&via_dims), dense(&via_single));
    assert_eq!(via_dims.shape(), via_single.shape());
}

#[test]
fn max_dims_single_axis_is_bit_identical_to_max() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t((0..24).map(|v| ((v * 7) % 13) as f32).collect(), &[2, 3, 4]);
    let xv = tape.var(&x);
    let via_dims = xv.max_dims(&[0], false).unwrap().to_tensor();
    let via_single = xv.max(Some(0)).unwrap().to_tensor();
    assert_eq!(dense(&via_dims), dense(&via_single));
}

// --- 3. 非連続 dims が permute／contiguous を経由すること ---

#[test]
fn non_contiguous_dims_push_more_nodes_than_trailing_dims() {
    // kept=[0,3] が縮約軸の間に挟まる非連続ケースは permute+contiguous
    // +reshape+sum の 4 ノード（恒等順列でないため contiguous も
    // 実体化コピーを伴う）、末尾側の連続ケースは reshape+sum の 2 ノード
    // （permute 省略・contiguous は no-op）で完了する。ノード数の差で
    // 経路の違いを間接確認する。
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let x = t((0..120).map(|v| v as f32).collect(), &[2, 3, 4, 5]);
    let xv = tape_a.var(&x);
    let before = tape_a.len();
    xv.sum_dims(&[1, 2], false).unwrap();
    let non_contig_added = tape_a.len() - before;

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let yv = tape_b.var(&x);
    let before_b = tape_b.len();
    yv.sum_dims(&[2, 3], false).unwrap();
    let trailing_added = tape_b.len() - before_b;

    assert!(
        non_contig_added > trailing_added,
        "non_contig_added={non_contig_added} trailing_added={trailing_added}"
    );
}

// --- 4. 数値微分との突合 ---

fn numeric_grad(target_tensor: &Tensor<f32>, perturb: impl Fn(Tensor<f32>) -> f32) -> Tensor<f32> {
    let shape = target_tensor.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut data: Vec<f32> = (0..numel)
        .map(|flat| {
            let mut idx = vec![0usize; shape.len()];
            let mut rem = flat;
            for axis in (0..shape.len()).rev() {
                idx[axis] = rem % shape[axis];
                rem /= shape[axis];
            }
            target_tensor.get(&idx).unwrap_or(0.0)
        })
        .collect();
    let mut grad = vec![0f32; numel];
    for i in 0..numel {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = perturb(t(data.clone(), &shape)) as f64;
        data[i] = (orig - H) as f32;
        let lm = perturb(t(data.clone(), &shape)) as f64;
        data[i] = orig as f32;
        grad[i] = ((lp - lm) / (2.0 * H)) as f32;
    }
    t(grad, &shape)
}

#[test]
fn sum_dims_gradient_matches_numeric() {
    let x = t((0..24).map(|v| v as f32 * 0.2 - 2.0).collect(), &[2, 3, 4]);
    let forward = |xt: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(xt);
        let y = xv.sum_dims(&[0, 2], false).unwrap();
        scalar(&y.sum(None).unwrap().to_tensor())
    };
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let loss = xv.sum_dims(&[0, 2], false).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let numeric = numeric_grad(&x, |xt| forward(&xt));
    assert_tensor_close("sum_dims grad", dx, &numeric);
}

#[test]
fn mean_dims_gradient_matches_numeric() {
    let x = t((0..24).map(|v| v as f32 * 0.2 - 2.0).collect(), &[2, 3, 4]);
    let forward = |xt: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(xt);
        let y = xv.mean_dims(&[0, 2], true).unwrap();
        scalar(&y.sum(None).unwrap().to_tensor())
    };
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let loss = xv.mean_dims(&[0, 2], true).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let numeric = numeric_grad(&x, |xt| forward(&xt));
    assert_tensor_close("mean_dims grad", dx, &numeric);
}

#[test]
fn mean_single_axis_gradient_matches_numeric() {
    let x = t((0..12).map(|v| v as f32 * 0.3 - 1.5).collect(), &[3, 4]);
    let forward = |xt: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(xt);
        let y = xv.mean(Some(1)).unwrap();
        scalar(&y.sum(None).unwrap().to_tensor())
    };
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let loss = xv.mean(Some(1)).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let numeric = numeric_grad(&x, |xt| forward(&xt));
    assert_tensor_close("mean(Some(1)) grad", dx, &numeric);
}

#[test]
fn mean_none_gradient_matches_numeric() {
    let x = t((0..12).map(|v| v as f32 * 0.3 - 1.5).collect(), &[3, 4]);
    let forward = |xt: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(xt);
        scalar(&xv.mean(None).unwrap().to_tensor())
    };
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let loss = xv.mean(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let numeric = numeric_grad(&x, |xt| forward(&xt));
    assert_tensor_close("mean(None) grad", dx, &numeric);
}

// --- 5. max_dims の同値タイの決定性 ---

#[test]
fn max_dims_tie_is_deterministic_across_runs() {
    let x = t(vec![3.0, 3.0, 1.0, 3.0, 2.0, 0.0], &[2, 3]);
    let mut results = Vec::new();
    for _ in 0..5 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let y = xv.max_dims(&[0, 1], false).unwrap();
        let loss = y; // すでにスカラー
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");
        results.push(dense(dx));
    }
    for r in &results[1..] {
        assert_eq!(&results[0], r, "max_dims のタイ分配が run ごとに揺れている");
    }
}

// --- 6. 空テンソル・拒否系 ---

#[test]
fn sum_dims_on_empty_axis_returns_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t(vec![], &[0, 3]);
    let xv = tape.var(&x);
    let got = xv.sum_dims(&[0], false).unwrap();
    assert_eq!(dense(&got.to_tensor()), vec![0.0, 0.0, 0.0]);
}

#[test]
fn mean_dims_on_empty_axis_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t(vec![], &[0, 3]);
    let xv = tape.var(&x);
    let err = xv.mean_dims(&[0], false).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn mean_on_empty_axis_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t(vec![], &[0, 3]);
    let xv = tape.var(&x);
    let err = xv.mean(Some(0)).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn sum_dims_rejects_empty_dims() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let xv = tape.var(&x);
    let err = xv.sum_dims(&[], false).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn sum_dims_rejects_duplicate_axis() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let xv = tape.var(&x);
    let err = xv.sum_dims(&[0, 0], false).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn max_dims_rejects_axis_out_of_range() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let xv = tape.var(&x);
    let err = xv.max_dims(&[2], false).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::AxisOutOfRange { axis: 2, rank: 2 })
    ));
}

// --- 7. checkpoint 経由の Op::Mean 再計算が forward と bit 同一 ---

#[test]
fn mean_recompute_via_checkpoint_is_bit_identical_to_forward() {
    let x = t((0..12).map(|v| v as f32 * 0.13 - 0.5).collect(), &[3, 4]);

    let tape_plain = Tape::new_with_ops(common::naive_ops());
    let xv = tape_plain.var(&x);
    let m = xv.mean(Some(1)).unwrap();
    let loss_plain = m.sum(None).unwrap();
    let grads_plain = tape_plain.backward(&loss_plain).unwrap();
    let dx_plain = grads_plain.get(&xv).unwrap().expect("x は loss に到達する");

    let tape_ckpt = Tape::new_with_ops(common::naive_ops());
    let xv2 = tape_ckpt.var(&x);
    let out = tape_ckpt
        .checkpoint(|| xv2.mean(Some(1)))
        .expect("checkpoint 区間は成功するはず");
    let loss_ckpt = out.sum(None).unwrap();
    let grads_ckpt = tape_ckpt.backward(&loss_ckpt).unwrap();
    let dx_ckpt = grads_ckpt.get(&xv2).unwrap().expect("x は loss に到達する");

    assert_eq!(
        dense(dx_plain),
        dense(dx_ckpt),
        "checkpoint 有無で Op::Mean の逆伝播値が bit 一致しない"
    );
}
