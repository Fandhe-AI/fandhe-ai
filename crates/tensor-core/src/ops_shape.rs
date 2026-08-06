//! 演算時（matmul・elementwise・reduction 等）の shape 検査（TASK-1.4c・#13）。
//!
//! TASK-1.4a（#11）までの shape 検査は `Tensor` の生成・view 操作
//! （`new`/`reshape`/`transpose`/`narrow`）に限られていた。本モジュールは
//! 演算実行時の shape 検査を「shape のみ（`&[usize]`）を入力とする純粋
//! 関数群」として提供する。`Tensor<T>` のメソッドにしない理由: 呼び出し元は
//! `autodiff` の `Var`（テープ内 Tensor。#15・TASK-1.5）と backend 入口の
//! `DeviceBuffer`（shape メタデータのみ保持し `Tensor<T>` 実体を持たない。
//! `docs/public-api-design.md` §4.2 `BackendOps`）の両方であり、
//! `Tensor` 実体を経由しない `DeviceBuffer` からも再利用できるようにする
//! ためである。
//!
//! 各関数は「検査 + 出力 shape の確定」を一体で行う（PoC-v2-1
//! `tensor.rs` の「まず shape 検査を経て結果 shape を確定し、その後
//! データを埋める」方針の踏襲。`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/`）。
//! すべての経路が `Result` を返し、本番経路で `unwrap()`/`expect()` は
//! 使わない（`.claude/rules/coding-rust.md`）。
//!
//! ブロードキャスト規則（NumPy 互換）は本来 #12（TASK-1.4b）の成果物に
//! 委譲する設計だが、着手時点で #12 は未マージだったため
//! `elementwise_out_shape` は暫定的に厳密一致のみを検査する
//! （下記 TODO 参照）。#12 マージ後、本モジュールを rebase してブロード
//! キャスト機構へ委譲するよう差し替える。

use crate::error::ShapeError;
use crate::tensor::checked_numel;

/// matmul（2 次元前提。`docs/public-api-design.md` §3.2）の出力 shape を
/// 検査・計算する。
///
/// `autodiff::Var::matmul`（#15）・backend 入口の `BackendOps::matmul`
/// （`docs/public-api-design.md` §4.2）から呼ばれ、カーネル実行前に
/// 呼び出し元が shape 前提を確認する契約点となる。
///
/// - `lhs`/`rhs` の rank が 2 でない場合 `ShapeError::RankMismatch`
///   （`expected: 2`）を返す。
/// - 内部次元（`lhs[1]` と `rhs[0]`）が一致しない場合
///   `ShapeError::MatmulDimMismatch` を返す。
/// - 出力 shape `[lhs[0], rhs[1]]` の要素数積のオーバーフローは
///   `checked_numel`（`tensor.rs` と共有）で検査し
///   `ShapeError::ElementCountOverflow` を返す。
pub fn matmul_out_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, ShapeError> {
    if lhs.len() != 2 {
        return Err(ShapeError::RankMismatch {
            expected: 2,
            actual: lhs.len(),
        });
    }
    if rhs.len() != 2 {
        return Err(ShapeError::RankMismatch {
            expected: 2,
            actual: rhs.len(),
        });
    }
    if lhs[1] != rhs[0] {
        return Err(ShapeError::MatmulDimMismatch {
            lhs: lhs.to_vec(),
            rhs: rhs.to_vec(),
        });
    }
    let out = vec![lhs[0], rhs[1]];
    checked_numel(&out)?;
    Ok(out)
}

/// elementwise 二項演算（`add`・`mul`。`docs/public-api-design.md` §3.2）の
/// 出力 shape を検査・計算する。
///
/// TODO(#12 マージ後): NumPy 互換ブロードキャスト規則（`broadcast_shape`
/// 相当）へ委譲する。#12（TASK-1.4b）着手時点で未マージのため、本関数は
/// 暫定的に「shape の完全一致」のみを成立条件とする安全側の実装とし、
/// 不一致は `ShapeError::ShapeMismatch` で返す。ブロードキャスト成立
/// ケース（例: `[3, 1]` と `[1, 4]` → `[3, 4]`）は #12 マージ後に対応する。
pub fn elementwise_out_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, ShapeError> {
    require_same_shape(lhs, rhs)?;
    Ok(lhs.to_vec())
}

/// 厳密一致を要求する演算（例: `mse_loss` の予測値と target。
/// `docs/public-api-design.md` §3.2）の shape 検査。
///
/// `elementwise_out_shape` の暫定実装（ブロードキャスト未対応）からも
/// 呼ばれる（上記 TODO 参照）。
pub fn require_same_shape(lhs: &[usize], rhs: &[usize]) -> Result<(), ShapeError> {
    if lhs != rhs {
        return Err(ShapeError::ShapeMismatch {
            lhs: lhs.to_vec(),
            rhs: rhs.to_vec(),
        });
    }
    Ok(())
}

/// reduction（`sum`・`max`。`BackendOps` の `dim: Option<usize>` シグネチャ
/// と対応。`docs/public-api-design.md` §3.2/§4.2）の出力 shape を検査・
/// 計算する。
///
/// - `dim: None` は全軸縮約であり、出力 shape は空（`[]`。rank 0 スカラー
///   相当）を返す。
/// - `dim: Some(axis)` は `axis` が `shape` の rank 範囲外の場合
///   `ShapeError::AxisOutOfRange` を返す（#11 で定義済みの variant を
///   `transpose`/`narrow` と共通利用する）。範囲内の場合、その軸を
///   除いた shape を返す（例: shape `[2, 3, 4]`・`axis=1` → `[2, 4]`）。
pub fn reduce_out_shape(shape: &[usize], dim: Option<usize>) -> Result<Vec<usize>, ShapeError> {
    match dim {
        None => Ok(Vec::new()),
        Some(axis) => {
            if axis >= shape.len() {
                return Err(ShapeError::AxisOutOfRange {
                    axis,
                    rank: shape.len(),
                });
            }
            let out = shape
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| if i == axis { None } else { Some(d) })
                .collect();
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- matmul_out_shape ---

    #[test]
    fn matmul_ok() {
        let out = matmul_out_shape(&[2, 3], &[3, 4]).unwrap();
        assert_eq!(out, vec![2, 4]);
    }

    #[test]
    fn matmul_rank_mismatch_lhs() {
        let err = matmul_out_shape(&[2, 3, 4], &[3, 4]).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 3
            }
        ));
    }

    #[test]
    fn matmul_rank_mismatch_rhs() {
        let err = matmul_out_shape(&[2, 3], &[3]).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn matmul_dim_mismatch() {
        let err = matmul_out_shape(&[2, 3], &[4, 5]).unwrap_err();
        match err {
            ShapeError::MatmulDimMismatch { lhs, rhs } => {
                assert_eq!(lhs, vec![2, 3]);
                assert_eq!(rhs, vec![4, 5]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn matmul_overflow() {
        // usize::MAX に近い次元同士の積は checked_numel でオーバーフロー検出される。
        let big = usize::MAX / 2 + 1;
        let err = matmul_out_shape(&[big, big], &[big, big]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
    }

    #[test]
    fn matmul_zero_size_axis() {
        // 空テンソル（サイズ 0 軸）は形状として妥当。
        let out = matmul_out_shape(&[0, 3], &[3, 4]).unwrap();
        assert_eq!(out, vec![0, 4]);
    }

    // --- elementwise_out_shape ---

    #[test]
    fn elementwise_same_shape_ok() {
        let out = elementwise_out_shape(&[2, 3], &[2, 3]).unwrap();
        assert_eq!(out, vec![2, 3]);
    }

    #[test]
    fn elementwise_mismatch_errors() {
        let err = elementwise_out_shape(&[2, 3], &[3, 2]).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn elementwise_scalar_rank0_ok() {
        let out = elementwise_out_shape(&[], &[]).unwrap();
        assert_eq!(out, Vec::<usize>::new());
    }

    // --- require_same_shape ---

    #[test]
    fn require_same_shape_ok() {
        require_same_shape(&[5], &[5]).unwrap();
    }

    #[test]
    fn require_same_shape_mismatch() {
        let err = require_same_shape(&[5], &[6]).unwrap_err();
        match err {
            ShapeError::ShapeMismatch { lhs, rhs } => {
                assert_eq!(lhs, vec![5]);
                assert_eq!(rhs, vec![6]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    // --- reduce_out_shape ---

    #[test]
    fn reduce_full_ok() {
        let out = reduce_out_shape(&[2, 3, 4], None).unwrap();
        assert_eq!(out, Vec::<usize>::new());
    }

    #[test]
    fn reduce_axis_ok() {
        let out = reduce_out_shape(&[2, 3, 4], Some(1)).unwrap();
        assert_eq!(out, vec![2, 4]);
    }

    #[test]
    fn reduce_axis_first_ok() {
        let out = reduce_out_shape(&[2, 3, 4], Some(0)).unwrap();
        assert_eq!(out, vec![3, 4]);
    }

    #[test]
    fn reduce_axis_last_ok() {
        let out = reduce_out_shape(&[2, 3, 4], Some(2)).unwrap();
        assert_eq!(out, vec![2, 3]);
    }

    #[test]
    fn reduce_axis_out_of_range() {
        let err = reduce_out_shape(&[2, 3, 4], Some(3)).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::AxisOutOfRange { axis: 3, rank: 3 }
        ));
    }

    #[test]
    fn reduce_rank0_full_ok() {
        // rank 0（スカラー）テンソルの全縮約は空 shape を返す。
        let out = reduce_out_shape(&[], None).unwrap();
        assert_eq!(out, Vec::<usize>::new());
    }

    #[test]
    fn reduce_empty_tensor_axis_ok() {
        // サイズ 0 軸を含む shape でも軸検査・出力 shape 計算は成立する。
        let out = reduce_out_shape(&[0, 3], Some(0)).unwrap();
        assert_eq!(out, vec![3]);
    }

    // --- Display panic-freedom ---

    #[test]
    fn display_does_not_panic_for_new_variants() {
        let errs = [
            ShapeError::MatmulDimMismatch {
                lhs: vec![2, 3],
                rhs: vec![4, 5],
            },
            ShapeError::ShapeMismatch {
                lhs: vec![1],
                rhs: vec![2],
            },
            ShapeError::RankMismatch {
                expected: 2,
                actual: 3,
            },
        ];
        for err in errs {
            let _ = format!("{err}");
        }
    }
}
