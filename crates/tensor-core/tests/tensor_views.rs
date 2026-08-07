//! `tensor-core` 公開 API の統合テスト（TASK-1.4d・#14）。
//!
//! `tensor.rs`／`broadcast.rs`／`ops_shape.rs` の inline `#[cfg(test)]`
//! は各モジュール単体の白箱テストであり、本ファイルはそれらと重複しない
//! 「複数 API の組合せ」観点（レイアウト連鎖・ブロードキャストと view の
//! 合成・shape 検査の一気通貫・`ops_shape` と `Tensor::shape()` の連携・
//! 端点ケース）で公開 API 契約を固定する。`autodiff`（#15 以降）・
//! backend 入口（`docs/public-api-design.md` §4.2 `DeviceBuffer`）が
//! 本クレートを消費する際の想定利用経路を検証する。
//!
//! CI（self-hosted）は `docs/spec`（submodule）を checkout しないため、
//! 本ファイルは `docs/spec` 配下のいかなるファイルにも依存しない。

use tensor_core::{ShapeError, Tensor};

// --- レイアウト: narrow -> transpose -> contiguous の連鎖 ---

#[test]
fn narrow_transpose_contiguous_chain_preserves_values() {
    // shape [4, 5] の連番データから narrow(0, 1, 2) -> transpose -> contiguous
    // の連鎖を経ても、各要素が元データの対応位置と一致することを確認する。
    let src: Vec<f32> = (0..20).map(|v| v as f32).collect();
    let t = Tensor::<f32>::from_slice(&src, &[4, 5]).unwrap();

    let n = t.narrow(0, 1, 2).unwrap(); // shape [2, 5]、元 row 1..3
    assert_eq!(n.offset(), 5);
    assert!(n.is_contiguous());

    let nt = n.transpose(0, 1).unwrap(); // shape [5, 2]
    assert!(!nt.is_contiguous());
    assert_eq!(nt.offset(), n.offset()); // transpose は offset を変えない

    let c = nt.contiguous();
    assert!(c.is_contiguous());
    assert_eq!(c.shape(), &[5, 2]);
    for row in 0..2 {
        for col in 0..5 {
            // 元テンソルの [1+row, col] は narrow 後 n の [row, col]、
            // transpose 後 nt の [col, row] に対応する。
            let expected = t.get(&[1 + row, col]).unwrap();
            assert_eq!(nt.get(&[col, row]).unwrap(), expected);
            assert_eq!(c.get(&[col, row]).unwrap(), expected);
        }
    }
}

#[test]
fn is_contiguous_transitions_across_chain() {
    let t = Tensor::<f32>::zeros(&[3, 4]).unwrap();
    assert!(t.is_contiguous());

    let tt = t.transpose(0, 1).unwrap();
    assert!(!tt.is_contiguous());

    let back = tt.transpose(0, 1).unwrap();
    assert!(back.is_contiguous());

    let n = t.narrow(0, 0, 1).unwrap();
    assert!(n.is_contiguous());
}

#[test]
fn offset_propagates_through_narrow_and_transpose() {
    let t = Tensor::<f32>::zeros(&[6, 4]).unwrap();
    assert_eq!(t.offset(), 0);
    let n = t.narrow(0, 2, 3).unwrap();
    // row-major strides: narrow(dim=0, start=2) は offset += 2 * stride[0](=4)
    assert_eq!(n.offset(), 8);
    let nt = n.transpose(0, 1).unwrap();
    // transpose は strides/shape の入れ替えのみで offset は不変。
    assert_eq!(nt.offset(), 8);
}

// --- ブロードキャスト: broadcast_to の stride 0 view と実体化 ---

#[test]
fn broadcast_to_stride_zero_view_then_contiguous_matches_expected() {
    let t = Tensor::<f32>::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let b = t.broadcast_to(&[3, 3]).unwrap();
    assert_eq!(b.strides(), &[0, 1]);
    for row in 0..3 {
        for col in 0..3 {
            assert_eq!(b.get(&[row, col]).unwrap(), t.get(&[col]).unwrap());
        }
    }
    let c = b.contiguous();
    assert!(c.is_contiguous());
    let expected = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    for (i, &v) in expected.iter().enumerate() {
        assert_eq!(c.get(&[i / 3, i % 3]).unwrap(), v);
    }
}

#[test]
fn broadcast_with_expands_both_operands_bidirectionally() {
    // [3, 1] と [1, 4] は互いに拡張し合い、共通 shape [3, 4] になる
    // （NumPy 互換ブロードキャストの双方向拡張ケース）。
    let a = Tensor::<f32>::from_slice(&[10.0, 20.0, 30.0], &[3, 1]).unwrap();
    let b = Tensor::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let (ba, bb) = a.broadcast_with(&b).unwrap();
    assert_eq!(ba.shape(), &[3, 4]);
    assert_eq!(bb.shape(), &[3, 4]);
    for row in 0..3 {
        for col in 0..4 {
            assert_eq!(ba.get(&[row, col]).unwrap(), a.get(&[row, 0]).unwrap());
            assert_eq!(bb.get(&[row, col]).unwrap(), b.get(&[0, col]).unwrap());
        }
    }
}

// --- shape 検査: 各エラー経路 ---

#[test]
fn reshape_rejects_non_contiguous() {
    let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
    let tt = t.transpose(0, 1).unwrap();
    let err = tt.reshape(&[6]).unwrap_err();
    assert!(matches!(err, ShapeError::NonContiguousReshape));
}

#[test]
fn narrow_out_of_bounds_rejected() {
    let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
    let err = t.narrow(0, 1, 5).unwrap_err();
    assert!(matches!(
        err,
        ShapeError::NarrowOutOfBounds {
            dim: 0,
            start: 1,
            len: 5,
            dim_size: 2
        }
    ));
}

#[test]
fn axis_out_of_range_rejected_for_transpose_and_narrow() {
    let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
    assert!(matches!(
        t.transpose(0, 9).unwrap_err(),
        ShapeError::AxisOutOfRange { axis: 9, rank: 2 }
    ));
    assert!(matches!(
        t.narrow(9, 0, 1).unwrap_err(),
        ShapeError::AxisOutOfRange { axis: 9, rank: 2 }
    ));
}

#[test]
fn element_count_overflow_rejected_at_construction() {
    let err = Tensor::<f32>::zeros(&[usize::MAX, 2]).unwrap_err();
    assert!(matches!(err, ShapeError::ElementCountOverflow));
    let err = Tensor::<f32>::new(vec![0.0; 1], &[usize::MAX, 2]).unwrap_err();
    assert!(matches!(err, ShapeError::ElementCountOverflow));
}

// --- ops_shape 連携: Tensor::shape() と組み合わせた利用経路 ---

#[test]
fn ops_shape_matmul_out_shape_matches_tensor_shapes() {
    let lhs = Tensor::<f32>::zeros(&[2, 3]).unwrap();
    let rhs = Tensor::<f32>::zeros(&[3, 4]).unwrap();
    let out = tensor_core::matmul_out_shape(lhs.shape(), rhs.shape()).unwrap();
    assert_eq!(out, vec![2, 4]);
}

#[test]
fn ops_shape_reduce_out_shape_matches_tensor_shape() {
    let t = Tensor::<f32>::zeros(&[2, 3, 4]).unwrap();
    let out = tensor_core::reduce_out_shape(t.shape(), Some(1)).unwrap();
    assert_eq!(out, vec![2, 4]);
    let out_full = tensor_core::reduce_out_shape(t.shape(), None).unwrap();
    assert!(out_full.is_empty());
}

#[test]
fn ops_shape_require_same_shape_matches_tensor_shapes() {
    let a = Tensor::<f32>::zeros(&[2, 3]).unwrap();
    let b = Tensor::<f32>::zeros(&[2, 3]).unwrap();
    tensor_core::require_same_shape(a.shape(), b.shape()).unwrap();

    let c = Tensor::<f32>::zeros(&[3, 2]).unwrap();
    let err = tensor_core::require_same_shape(a.shape(), c.shape()).unwrap_err();
    assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
}

// --- 端点: 空テンソル・rank 0（スカラー）・要素数 1 のブロードキャスト ---

#[test]
fn empty_axis_tensor_round_trips_through_reshape() {
    let t = Tensor::<f32>::zeros(&[0, 3]).unwrap();
    assert!(t.is_empty());
    assert!(t.is_contiguous());
    let r = t.reshape(&[0]).unwrap();
    assert_eq!(r.shape(), &[0]);
    assert_eq!(r.numel(), 0);
}

#[test]
fn rank_zero_scalar_get_and_broadcast() {
    let t = Tensor::<f32>::new(vec![7.0], &[]).unwrap();
    assert_eq!(t.rank(), 0);
    assert_eq!(t.numel(), 1);
    assert_eq!(t.get(&[]).unwrap(), 7.0);

    let b = t.broadcast_to(&[2, 2]).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(b.get(&[i, j]).unwrap(), 7.0);
        }
    }
}

#[test]
fn single_element_tensor_broadcast_with_larger_shape() {
    let t = Tensor::<f32>::new(vec![9.0], &[1, 1]).unwrap();
    let (bt, _) = t
        .broadcast_with(&Tensor::<f32>::zeros(&[3, 5]).unwrap())
        .unwrap();
    assert_eq!(bt.shape(), &[3, 5]);
    for i in 0..3 {
        for j in 0..5 {
            assert_eq!(bt.get(&[i, j]).unwrap(), 9.0);
        }
    }
}

// --- transpose_2d: #73（TASK-7.1a）safetensors ローダーが
// `nn.Linear.weight` の明示転置に用いる公開 API の契約確認 ---

#[test]
fn transpose_2d_swaps_shape_and_preserves_values() {
    // [out=3, in=2] の PyTorch Linear weight 相当を [in=2, out=3] へ転置する。
    let src: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let t = Tensor::<f32>::from_slice(&src, &[3, 2]).unwrap();

    let tt = t.transpose_2d().unwrap();
    assert_eq!(tt.shape(), &[2, 3]);
    for i in 0..3 {
        for j in 0..2 {
            assert_eq!(tt.get(&[j, i]).unwrap(), t.get(&[i, j]).unwrap());
        }
    }
}

#[test]
fn transpose_2d_rejects_non_rank_2() {
    let t = Tensor::<f32>::zeros(&[2, 3, 4]).unwrap();
    let err = t.transpose_2d().unwrap_err();
    assert!(matches!(
        err,
        ShapeError::RankMismatch {
            expected: 2,
            actual: 3
        }
    ));

    let v = Tensor::<f32>::zeros(&[5]).unwrap();
    let err = v.transpose_2d().unwrap_err();
    assert!(matches!(
        err,
        ShapeError::RankMismatch {
            expected: 2,
            actual: 1
        }
    ));
}
