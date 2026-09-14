//! `CpuBackendOps::nll_loss`／`nll_loss_backward`（融合カーネル。イシュー
//! #1738）と素朴な参照実装（本ファイル内 `naive_nll_loss`／
//! `naive_nll_loss_backward`）の数値一致検証（`mse_parity.rs` と同型
//! 構成）。
//!
//! 突合は統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`。
//! 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`.claude/rules/
//! coding-rust.md`）で行う。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::{BackendOps, MseReduction, Tensor};

/// `nll.rs::nll_sum_f32` と数式的に同一だが、丸め手順を分離した素朴な
/// 参照実装（単純逐次累積。`fandhe_ai_autodiff::eval::nll_loss` と同型）。
fn naive_nll_loss(
    input: &[f32],
    targets: &[i32],
    outer: usize,
    axis_len: usize,
    inner: usize,
    reduction: MseReduction,
) -> f32 {
    let n = outer * inner;
    let mut total = 0f32;
    for o in 0..outer {
        for i in 0..inner {
            let t = targets[o * inner + i] as usize;
            total -= input[(o * axis_len + t) * inner + i];
        }
    }
    match reduction {
        MseReduction::Mean => {
            if n == 0 {
                0.0
            } else {
                total / n as f32
            }
        }
        MseReduction::Sum => total,
        _ => total,
    }
}

fn naive_nll_loss_backward(
    targets: &[i32],
    outer: usize,
    axis_len: usize,
    inner: usize,
    scale: f32,
) -> Vec<f32> {
    let mut grad = vec![0f32; outer * axis_len * inner];
    for o in 0..outer {
        for i in 0..inner {
            let t = targets[o * inner + i] as usize;
            grad[(o * axis_len + t) * inner + i] = -scale;
        }
    }
    grad
}

/// `(outer, axis_len, inner)` の形状スイープ: `class_dim` が先頭・末尾・
/// 中間のケース、`nll.rs::CHUNK`（4096）境界跨ぎ、非対称な大 shape。
fn shapes() -> Vec<(usize, usize, usize)> {
    vec![
        (2, 3, 1),    // class_dim = 1（末尾軸）・[2,3] 相当
        (1, 5, 4),    // class_dim = 0（先頭軸）・[5,4] 相当
        (3, 4, 2),    // class_dim = 1（中間軸）・[3,4,2] 相当
        (0, 3, 1),    // outer=0（空バッチ）
        (1, 3, 0),    // inner=0（空バッチ）
        (1, 1, 4095), // CHUNK 直前
        (1, 1, 4096), // CHUNK ちょうど
        (1, 1, 4097), // CHUNK 直後
        (1, 1, 8193), // 大 n
    ]
}

fn make_input(outer: usize, axis_len: usize, inner: usize) -> Vec<f32> {
    let n = outer * axis_len * inner;
    (0..n).map(|i| -((i as f32) * 0.001 + 0.01)).collect()
}

fn make_targets(outer: usize, axis_len: usize, inner: usize) -> Vec<i32> {
    let n = outer * inner;
    (0..n).map(|i| (i % axis_len.max(1)) as i32).collect()
}

#[test]
fn nll_loss_forward_matches_naive_mean() {
    let ops = CpuBackendOps::new();
    for (outer, axis_len, inner) in shapes() {
        let class_dim = 1usize; // 呼び出し規約に合わせ rank=3 の [outer, axis_len, inner] を使う
        let shape = [outer, axis_len, inner];
        let input_data = make_input(outer, axis_len, inner);
        let targets_data = make_targets(outer, axis_len, inner);
        let input = Tensor::new(input_data.clone(), &shape).unwrap();
        let targets_shape = [outer, inner];
        let targets = Tensor::new(targets_data.clone(), &targets_shape).unwrap();

        let got = ops
            .nll_loss(&input, &targets, class_dim, MseReduction::Mean)
            .unwrap_or_else(|e| panic!("nll_loss failed for {outer:?},{axis_len},{inner}: {e:?}"));
        assert_eq!(got.shape(), &[] as &[usize]);

        let expected = naive_nll_loss(
            &input_data,
            &targets_data,
            outer,
            axis_len,
            inner,
            MseReduction::Mean,
        );
        assert_parity(
            &format!("nll_loss forward mean outer={outer} axis_len={axis_len} inner={inner}"),
            got.as_slice().unwrap(),
            &[expected],
        );
    }
}

#[test]
fn nll_loss_forward_matches_naive_sum() {
    let ops = CpuBackendOps::new();
    for (outer, axis_len, inner) in shapes() {
        let class_dim = 1usize;
        let shape = [outer, axis_len, inner];
        let input_data = make_input(outer, axis_len, inner);
        let targets_data = make_targets(outer, axis_len, inner);
        let input = Tensor::new(input_data.clone(), &shape).unwrap();
        let targets_shape = [outer, inner];
        let targets = Tensor::new(targets_data.clone(), &targets_shape).unwrap();

        let got = ops
            .nll_loss(&input, &targets, class_dim, MseReduction::Sum)
            .unwrap_or_else(|e| panic!("nll_loss failed for {outer:?},{axis_len},{inner}: {e:?}"));
        let expected = naive_nll_loss(
            &input_data,
            &targets_data,
            outer,
            axis_len,
            inner,
            MseReduction::Sum,
        );
        assert_parity(
            &format!("nll_loss forward sum outer={outer} axis_len={axis_len} inner={inner}"),
            got.as_slice().unwrap(),
            &[expected],
        );
    }
}

#[test]
fn nll_loss_backward_matches_naive() {
    let ops = CpuBackendOps::new();
    for (outer, axis_len, inner) in shapes() {
        let class_dim = 1usize;
        let shape = [outer, axis_len, inner];
        let targets_data = make_targets(outer, axis_len, inner);
        let targets_shape = [outer, inner];
        let targets = Tensor::new(targets_data.clone(), &targets_shape).unwrap();
        let scale = 1.7f32;

        let got = ops
            .nll_loss_backward(&shape, &targets, class_dim, scale)
            .unwrap_or_else(|e| {
                panic!("nll_loss_backward failed for {outer:?},{axis_len},{inner}: {e:?}")
            });
        assert_eq!(got.shape(), &shape);

        let expected = naive_nll_loss_backward(&targets_data, outer, axis_len, inner, scale);
        assert_parity(
            &format!("nll_loss backward outer={outer} axis_len={axis_len} inner={inner}"),
            got.as_slice().unwrap(),
            &expected,
        );
    }
}

/// `class_dim` が先頭軸（0）・末尾軸（rank-1）のケースを個別に確認する
/// （`shapes()` の主スイープは rank=3 固定のため、rank=2 での境界条件を
/// 別途カバーする）。
#[test]
fn nll_loss_class_dim_at_boundaries_matches_naive() {
    let ops = CpuBackendOps::new();

    // class_dim = 0（先頭軸。outer=1）: shape=[3,4] → axis_len=3, inner=4
    {
        let input_data = make_input(1, 3, 4);
        let targets_data = make_targets(1, 3, 4);
        let input = Tensor::new(input_data.clone(), &[3, 4]).unwrap();
        let targets = Tensor::new(targets_data.clone(), &[4]).unwrap();
        let got = ops
            .nll_loss(&input, &targets, 0, MseReduction::Mean)
            .unwrap();
        let expected = naive_nll_loss(&input_data, &targets_data, 1, 3, 4, MseReduction::Mean);
        assert_parity("nll_loss class_dim=0", got.as_slice().unwrap(), &[expected]);
    }

    // class_dim = rank-1（末尾軸。inner=1）: shape=[2,5] → outer=2, axis_len=5
    {
        let input_data = make_input(2, 5, 1);
        let targets_data = make_targets(2, 5, 1);
        let input = Tensor::new(input_data.clone(), &[2, 5]).unwrap();
        let targets = Tensor::new(targets_data.clone(), &[2]).unwrap();
        let got = ops
            .nll_loss(&input, &targets, 1, MseReduction::Mean)
            .unwrap();
        let expected = naive_nll_loss(&input_data, &targets_data, 2, 5, 1, MseReduction::Mean);
        assert_parity(
            "nll_loss class_dim=rank-1",
            got.as_slice().unwrap(),
            &[expected],
        );
    }
}

#[test]
fn nll_loss_rejects_class_dim_out_of_range() {
    use fandhe_ai_tensor_core::device::BackendError;

    let ops = CpuBackendOps::new();
    let input = Tensor::new(vec![-0.1, -2.0], &[2]).unwrap();
    let targets = Tensor::new(vec![0i32], &[1]).unwrap();

    let forward = ops.nll_loss(&input, &targets, 5, MseReduction::Mean);
    assert!(matches!(forward, Err(BackendError::ShapeMismatch(_))));

    let backward = ops.nll_loss_backward(&[2], &targets, 5, 1.0);
    assert!(matches!(backward, Err(BackendError::ShapeMismatch(_))));
}
