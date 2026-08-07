//! ONNX `MatMul` オペ（TASK-7.3c・#84）。`numpy.matmul` 準拠のバッチ行列積。
//!
//! Transformer の Attention 計算（`Q @ K^T`・`softmax(...) @ V`）はバッチ・ヘッド軸を
//! 持つ 3〜4 次元テンソル同士の行列積を要求するため、`Gemm`（2 次元専用。TASK-7.2c）
//! とは別関数として提供する。末尾 2 軸を行列積対象とし、それより前の軸は NumPy 互換
//! ブロードキャスト（`tensor_core::broadcast_shape`）でバッチ次元を揃える。
//! 内積の累積は丸め方針（FMA 契約）統一方針（`.claude/rules/coding-rust.md`）に従い
//! `f32::mul_add` を用いる（`gemm.rs` と同一方針）。

use tensor_core::{Tensor, broadcast_shape};

use super::error::OpError;

/// バッチ行列積の要素数（`batch`・出力要素数 `out_len`）を `checked_mul` で検査する。
///
/// `batch_shape`（ブロードキャスト後のバッチ次元）・`m`（A の行数）・`k`（内部次元）・
/// `n`（B の列数）から `batch`・`batch*m*n`（出力）・`batch*m*k`（A 実体化サイズ）・
/// `batch*k*n`（B 実体化サイズ）のいずれかが usize をあふれる場合、外部フォーマット
/// （ONNX shape 属性）由来の巨大 shape とみなし [`OpError::MatMulElementCountOverflow`]
/// で拒否する（OWASP A03。`.claude/rules/security.md`）。`matmul` から実体化前に呼ばれる。
fn checked_matmul_element_counts(
    batch_shape: &[usize],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(usize, usize), OpError> {
    let batch: usize = batch_shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(OpError::MatMulElementCountOverflow)?;
    let out_len = batch
        .checked_mul(m)
        .and_then(|v| v.checked_mul(n))
        .ok_or(OpError::MatMulElementCountOverflow)?;
    batch
        .checked_mul(m)
        .and_then(|v| v.checked_mul(k))
        .ok_or(OpError::MatMulElementCountOverflow)?;
    batch
        .checked_mul(k)
        .and_then(|v| v.checked_mul(n))
        .ok_or(OpError::MatMulElementCountOverflow)?;
    Ok((batch, out_len))
}

/// `MatMul` を計算する（`numpy.matmul` セマンティクス）。
///
/// - rank 0（スカラー）入力は [`OpError::RankMismatch`]（`op: "MatMul(A)"` 等）で拒否する。
/// - 1 次元入力は NumPy 仕様どおり一時的に軸を挿入して計算し（lhs は先頭に `1`・rhs は
///   末尾に `1`）、計算後に挿入した軸を出力 shape から除去する。
/// - 末尾 2 軸を除くバッチ次元は `broadcast_shape` で NumPy 互換に解決する
///   （非互換 shape は `ShapeError::BroadcastIncompatible` を [`OpError::Shape`] として透過）。
/// - 内部次元（`a` の最終軸と `b` の最後から 2 番目の軸）の不一致は
///   [`OpError::MatMulDimMismatch`]。
pub fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    if a.rank() == 0 {
        return Err(OpError::RankMismatch {
            op: "MatMul(A)",
            expected: 1,
            actual: 0,
        });
    }
    if b.rank() == 0 {
        return Err(OpError::RankMismatch {
            op: "MatMul(B)",
            expected: 1,
            actual: 0,
        });
    }

    // 1-D 特例: NumPy 仕様上「片方または両方が 1-D の場合、計算のために軸を 1 つ挿入し、
    // 計算後に取り除く」。挿入位置は lhs が先頭（行ベクトル扱い）・rhs が末尾（列ベクトル
    // 扱い）で、通常の 2 次元行列積の形へ揃えるため。
    let a_expanded_lhs = a.rank() == 1;
    let b_expanded_rhs = b.rank() == 1;

    let a_shape: Vec<usize> = if a_expanded_lhs {
        let mut s = vec![1usize];
        s.extend_from_slice(a.shape());
        s
    } else {
        a.shape().to_vec()
    };
    let b_shape: Vec<usize> = if b_expanded_rhs {
        let mut s = b.shape().to_vec();
        s.push(1);
        s
    } else {
        b.shape().to_vec()
    };

    // 1-D 特例のみ実際に軸を挿入する。rank ≥ 2 の入力（transpose 後の非 contiguous view を
    // 含みうる）を無条件に `reshape` すると `ShapeError::NonContiguousReshape` を誤って
    // 返しうるため、形状変更が不要な場合は元のテンソルをそのまま使う。1-D 側も
    // `broadcast_to` 由来の非 contiguous view（stride 0）でありうるため、`unsqueeze` と
    // 同様に `contiguous()` してから `reshape` する（PR #276 レビュー指摘）。
    let a_r = if a_expanded_lhs {
        a.contiguous().reshape(&a_shape).map_err(OpError::from)?
    } else {
        a.clone()
    };
    let b_r = if b_expanded_rhs {
        b.contiguous().reshape(&b_shape).map_err(OpError::from)?
    } else {
        b.clone()
    };

    let a_batch = &a_shape[..a_shape.len() - 2];
    let b_batch = &b_shape[..b_shape.len() - 2];
    let batch_shape = broadcast_shape(a_batch, b_batch).map_err(OpError::from)?;

    let (m, k) = (a_shape[a_shape.len() - 2], a_shape[a_shape.len() - 1]);
    let (k2, n) = (b_shape[b_shape.len() - 2], b_shape[b_shape.len() - 1]);
    if k != k2 {
        return Err(OpError::MatMulDimMismatch {
            a: a_shape.clone(),
            b: b_shape.clone(),
        });
    }

    // batch * m * n（出力要素数）・batch * m * k・batch * k * n（入力走査量）を
    // `checked_mul` で検査し、外部フォーマット由来の巨大 shape による usize あふれを
    // 未然に拒否する（OWASP A03。`.claude/rules/security.md`）。
    // `broadcast_to`/`contiguous` による実体化（次段）より前に検査することで、
    // あふれるほど巨大な shape に対して確保を試みてから失敗する（Vec capacity
    // overflow によるパニック等）事態を避ける。
    let (batch, out_len) = checked_matmul_element_counts(&batch_shape, m, k, n)?;

    // バッチ次元 + 行列 2 軸へブロードキャストしてから実体化する（`as_slice` は連続
    // 領域を要求するため。`gemm.rs` の `trans_a`/`trans_b` 実体化と同一方針）。
    let mut a_bcast_shape = batch_shape.clone();
    a_bcast_shape.push(m);
    a_bcast_shape.push(k);
    let mut b_bcast_shape = batch_shape.clone();
    b_bcast_shape.push(k);
    b_bcast_shape.push(n);

    let a_full = a_r.broadcast_to(&a_bcast_shape)?.contiguous();
    let b_full = b_r.broadcast_to(&b_bcast_shape)?.contiguous();

    let a_slice = a_full
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("MatMul(A)"))?;
    let b_slice = b_full
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("MatMul(B)"))?;

    let mut out = vec![0f32; out_len];
    let a_mat_stride = m * k;
    let b_mat_stride = k * n;
    let out_mat_stride = m * n;
    for bi in 0..batch {
        let a_mat = &a_slice[bi * a_mat_stride..(bi + 1) * a_mat_stride];
        let b_mat = &b_slice[bi * b_mat_stride..(bi + 1) * b_mat_stride];
        let out_mat = &mut out[bi * out_mat_stride..(bi + 1) * out_mat_stride];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    acc = a_mat[i * k + p].mul_add(b_mat[p * n + j], acc);
                }
                out_mat[i * n + j] = acc;
            }
        }
    }

    // 1-D 特例で挿入した軸を出力 shape から除去する（NumPy 仕様: 両方 1-D ならスカラー
    // `[]`、片方のみ 1-D なら該当軸を除いたベクトル/テンソル）。
    let mut out_shape = batch_shape;
    if !a_expanded_lhs {
        out_shape.push(m);
    }
    if !b_expanded_rhs {
        out_shape.push(n);
    }

    Tensor::new(out, &out_shape).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2d_matches_gemm_known_values() {
        // gemm.rs の plain_matmul_no_bias と同一想定値（PyTorch/NumPy A @ B 一致）。
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::<f32>::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
        let y = matmul(&a, &b).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 58.0);
        assert_eq!(y.get(&[0, 1]).unwrap(), 64.0);
        assert_eq!(y.get(&[1, 0]).unwrap(), 139.0);
        assert_eq!(y.get(&[1, 1]).unwrap(), 154.0);
    }

    #[test]
    fn matmul_3d_same_batch() {
        // [2,2,3] @ [2,3,2] -> [2,2,2]。バッチごとに独立な 2-D 行列積。
        let a = Tensor::<f32>::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch0: [[1,2,3],[4,5,6]]
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // batch1: [[1,0,0],[0,1,0]]
            ],
            &[2, 2, 3],
        )
        .unwrap();
        let b = Tensor::<f32>::new(
            vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch0: [[7,8],[9,10],[11,12]]
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch1: [[1,2],[3,4],[5,6]]
            ],
            &[2, 3, 2],
        )
        .unwrap();
        let y = matmul(&a, &b).unwrap();
        assert_eq!(y.shape(), &[2, 2, 2]);
        // batch0: 同じく [[58,64],[139,154]]
        assert_eq!(y.get(&[0, 0, 0]).unwrap(), 58.0);
        assert_eq!(y.get(&[0, 1, 1]).unwrap(), 154.0);
        // batch1: 単位行列 2x3 部分抽出 -> [[1,2],[3,4]]
        assert_eq!(y.get(&[1, 0, 0]).unwrap(), 1.0);
        assert_eq!(y.get(&[1, 0, 1]).unwrap(), 2.0);
        assert_eq!(y.get(&[1, 1, 0]).unwrap(), 3.0);
        assert_eq!(y.get(&[1, 1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn matmul_batch_broadcast_lhs_no_batch() {
        // [2,3,4] @ [4,5] -> [2,3,5]（b にバッチ軸なし。全バッチで同じ b を共有）。
        let a = Tensor::<f32>::ones(&[2, 3, 4]).unwrap();
        let b = Tensor::<f32>::ones(&[4, 5]).unwrap();
        let y = matmul(&a, &b).unwrap();
        assert_eq!(y.shape(), &[2, 3, 5]);
        for bi in 0..2 {
            for i in 0..3 {
                for j in 0..5 {
                    assert_eq!(y.get(&[bi, i, j]).unwrap(), 4.0);
                }
            }
        }
    }

    #[test]
    fn matmul_batch_broadcast_size_one() {
        // [1,2,3] @ [4,3,2] -> [4,2,2]（lhs バッチ軸 1 が rhs バッチ軸 4 へブロードキャスト）。
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]).unwrap();
        let b = Tensor::<f32>::ones(&[4, 3, 2]).unwrap();
        let y = matmul(&a, &b).unwrap();
        assert_eq!(y.shape(), &[4, 2, 2]);
        for bi in 0..4 {
            // 各バッチで a は同一値: row0 = [1,2,3] の総和 6、row1 = [4,5,6] の総和 15
            assert_eq!(y.get(&[bi, 0, 0]).unwrap(), 6.0);
            assert_eq!(y.get(&[bi, 0, 1]).unwrap(), 6.0);
            assert_eq!(y.get(&[bi, 1, 0]).unwrap(), 15.0);
            assert_eq!(y.get(&[bi, 1, 1]).unwrap(), 15.0);
        }
    }

    #[test]
    fn matmul_1d_vector_times_matrix() {
        // [k] @ [k,n] -> [n]（先頭に軸挿入して計算後に除去）。
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<f32>::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]).unwrap();
        let y = matmul(&a, &b).unwrap();
        assert_eq!(y.shape(), &[2]);
        // [1,2,3]·[1,0,1] = 4, [1,2,3]·[0,1,1] = 5
        assert_eq!(y.get(&[0]).unwrap(), 4.0);
        assert_eq!(y.get(&[1]).unwrap(), 5.0);
    }

    #[test]
    fn matmul_matrix_times_1d_vector() {
        // [m,k] @ [k] -> [m]（末尾に軸挿入して計算後に除去）。
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::<f32>::new(vec![1.0, 0.0, 1.0], &[3]).unwrap();
        let y = matmul(&a, &b).unwrap();
        assert_eq!(y.shape(), &[2]);
        assert_eq!(y.get(&[0]).unwrap(), 4.0); // 1+0+3
        assert_eq!(y.get(&[1]).unwrap(), 10.0); // 4+0+6
    }

    #[test]
    fn matmul_1d_vector_times_matrix_accepts_non_contiguous_broadcast_input() {
        // PR #276 レビュー指摘: broadcast_to 由来の非 contiguous な rank-1 入力
        // （stride 0）でも、内部で `contiguous()` してから `reshape` するため
        // `ShapeError::NonContiguousReshape` を誤って返さないことを確認する。
        let scalar = Tensor::<f32>::new(vec![2.0], &[]).unwrap();
        let a = scalar.broadcast_to(&[3]).unwrap();
        assert!(!a.is_contiguous());
        let b = Tensor::<f32>::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]).unwrap();
        let y = matmul(&a, &b).unwrap();
        assert_eq!(y.shape(), &[2]);
        // [2,2,2]·[1,0,1] = 4, [2,2,2]·[0,1,1] = 4
        assert_eq!(y.get(&[0]).unwrap(), 4.0);
        assert_eq!(y.get(&[1]).unwrap(), 4.0);
    }

    #[test]
    fn matmul_1d_dot_1d_yields_scalar() {
        // [k] @ [k] -> スカラー（shape []）。
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<f32>::new(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let y = matmul(&a, &b).unwrap();
        assert_eq!(y.shape(), &[] as &[usize]);
        assert_eq!(y.get(&[]).unwrap(), 32.0); // 4+10+18
    }

    #[test]
    fn inner_dim_mismatch_rejected() {
        let a = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let b = Tensor::<f32>::zeros(&[4, 2]).unwrap();
        let err = matmul(&a, &b).unwrap_err();
        assert!(matches!(err, OpError::MatMulDimMismatch { .. }));
    }

    #[test]
    fn rank_zero_rejected() {
        let a = Tensor::<f32>::new(vec![1.0], &[]).unwrap();
        let b = Tensor::<f32>::zeros(&[2, 2]).unwrap();
        let err = matmul(&a, &b).unwrap_err();
        assert!(matches!(
            err,
            OpError::RankMismatch {
                expected: 1,
                actual: 0,
                ..
            }
        ));
    }

    #[test]
    fn element_count_overflow_rejected_via_batch_product() {
        // batch_shape 自体の総積が usize をあふれるケース（`try_fold` の checked_mul）。
        let err = checked_matmul_element_counts(&[usize::MAX, 2], 1, 1, 1).unwrap_err();
        assert!(matches!(err, OpError::MatMulElementCountOverflow));
    }

    #[test]
    fn element_count_overflow_rejected_via_out_len() {
        // batch は収まるが batch * m * n（出力要素数）があふれるケース。
        let err = checked_matmul_element_counts(&[2], usize::MAX / 2 + 1, 1, 2).unwrap_err();
        assert!(matches!(err, OpError::MatMulElementCountOverflow));
    }

    #[test]
    fn element_count_overflow_rejected_via_a_realization_size() {
        // out_len（batch*m*n）は収まるが batch*m*k（A 実体化サイズ）があふれるケース。
        let err = checked_matmul_element_counts(&[1], 2, usize::MAX / 2 + 1, 1).unwrap_err();
        assert!(matches!(err, OpError::MatMulElementCountOverflow));
    }

    #[test]
    fn element_count_overflow_rejected_via_b_realization_size() {
        // out_len・batch*m*k は収まるが batch*k*n（B 実体化サイズ）があふれるケース。
        let err = checked_matmul_element_counts(&[1], 1, 2, usize::MAX / 2 + 1).unwrap_err();
        assert!(matches!(err, OpError::MatMulElementCountOverflow));
    }

    #[test]
    fn element_count_within_bounds_accepted() {
        let (batch, out_len) = checked_matmul_element_counts(&[2, 3], 4, 5, 6).unwrap();
        assert_eq!(batch, 6);
        assert_eq!(out_len, 6 * 4 * 6);
    }

    #[test]
    fn batch_incompatible_rejected() {
        // バッチ軸 2 と 3 は NumPy 互換ブロードキャスト非互換。
        let a = Tensor::<f32>::zeros(&[2, 2, 3]).unwrap();
        let b = Tensor::<f32>::zeros(&[3, 3, 2]).unwrap();
        let err = matmul(&a, &b).unwrap_err();
        assert!(matches!(err, OpError::Shape(_)));
    }
}
