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
//! ブロードキャスト規則（NumPy 互換）は #12（TASK-1.4b・`broadcast.rs`）の
//! 成果物に委譲する。着手時点（TASK-1.4c）では #12 が未マージだったため
//! `elementwise_out_shape` は一時的に厳密一致のみを検査していたが、#12 が
//! caaf3c0 でマージ済みとなったため本イシュー（#22・TASK-1.6b）で
//! `broadcast_shape` への委譲へ差し替える。

use crate::broadcast::broadcast_shape;
use crate::error::ShapeError;
use crate::tensor::checked_numel;

/// matmul（2 次元前提。`docs/public-api-design.md` §3.2）の出力 shape を
/// 検査・計算する。
///
/// `fandhe_ai_autodiff::Var::matmul`（#15）・backend 入口の `BackendOps::matmul`
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
/// NumPy 互換ブロードキャスト規則（`broadcast_shape`。#12・TASK-1.4b）へ
/// 委譲する。不一致（末尾軸から比較して「両者同一」または「片方が 1」の
/// いずれも満たさない軸がある）は `ShapeError::BroadcastIncompatible` を
/// 返す。呼び出し元（`backend-cpu` の elementwise カーネル入口。#22・
/// TASK-1.6b）はここで確定した出力 shape を `Tensor::broadcast_with` へ
/// 渡し、両オペランドの zero-copy view（stride 0）を取得する想定。
///
/// ブロードキャストは出力の各軸を入力軸の最大値まで拡張しうるため、
/// 出力要素数は `lhs`・`rhs` いずれの要素数より大きくなりうる
/// （例: `[1, N]` と `[N, 1]` → `[N, N]`）。`matmul_out_shape` と同様に
/// `checked_numel` で要素数積の `usize` オーバーフローを検査し、
/// オーバーフロー時は `ShapeError::ElementCountOverflow` を返す。
pub fn elementwise_out_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, ShapeError> {
    let out = broadcast_shape(lhs, rhs)?;
    checked_numel(&out)?;
    Ok(out)
}

/// 厳密一致を要求する演算（例: `mse_loss` の予測値と target。
/// `docs/public-api-design.md` §3.2）の shape 検査。
///
/// `elementwise_out_shape`（ブロードキャスト委譲）とは独立した検査であり、
/// ブロードキャストを許容しない演算から呼ばれる。
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

/// `cat`（`Var::cat`。イシュー #1598）の出力 shape を検査・計算する。
///
/// `shapes` は連結対象の各テンソルの shape（空リストは呼び出し元
/// `Var::cat` が `vars[0]` に触れる前に `AutodiffError::InvalidArgument`
/// で弾く契約のため、本関数側では `ShapeError::RankMismatch { expected:
/// 1, actual: 0 }` を fail-closed な代替として返す）。
///
/// - 全 shape の rank が一致しない場合 `ShapeError::RankMismatch`
///   （`expected` は先頭要素の rank）。
/// - `dim` が rank 範囲外の場合 `ShapeError::AxisOutOfRange`。
/// - `dim` 軸以外の各軸が全 shape で一致しない場合
///   `ShapeError::ShapeMismatch`（`lhs`＝先頭 shape・`rhs`＝不一致の
///   あった shape）。
/// - 出力 shape（`dim` 軸のみ各 shape の合計・他軸は共通値）の要素数積
///   オーバーフローは `checked_numel` で検査し
///   `ShapeError::ElementCountOverflow` を返す。
pub fn concat_out_shape(shapes: &[&[usize]], dim: usize) -> Result<Vec<usize>, ShapeError> {
    let first = match shapes.first() {
        Some(s) => *s,
        None => {
            return Err(ShapeError::RankMismatch {
                expected: 1,
                actual: 0,
            });
        }
    };
    let rank = first.len();
    if dim >= rank {
        return Err(ShapeError::AxisOutOfRange { axis: dim, rank });
    }
    let mut dim_sum: usize = 0;
    for &shape in shapes {
        if shape.len() != rank {
            return Err(ShapeError::RankMismatch {
                expected: rank,
                actual: shape.len(),
            });
        }
        for (axis, (&s, &f)) in shape.iter().zip(first.iter()).enumerate() {
            if axis == dim {
                continue;
            }
            if s != f {
                return Err(ShapeError::ShapeMismatch {
                    lhs: first.to_vec(),
                    rhs: shape.to_vec(),
                });
            }
        }
        dim_sum = dim_sum
            .checked_add(shape[dim])
            .ok_or(ShapeError::ElementCountOverflow)?;
    }
    let mut out = first.to_vec();
    out[dim] = dim_sum;
    checked_numel(&out)?;
    Ok(out)
}

/// LayerNorm／RMSNorm（イシュー #1596。`BackendOps::layer_norm`／
/// `rmsnorm` の共通入口）が起動前に `(rows, hidden)` を導出するための
/// shape 検査。CPU／CUDA／Metal の各 `BackendOps` 実装が本関数の結果を
/// 再導出せず共有する単一情報源とする。`row_softmax_layout`
/// （イシュー #1594）とは独立した関数とする——softmax は任意軸・非最終軸を
/// `Ok(None)` で区別する契約だが、LayerNorm／RMSNorm は最終軸限定
/// （PyTorch `nn.LayerNorm`／`nn.RMSNorm` の `normalized_shape` は
/// 常に末尾次元群。多次元 `normalized_shape` は本イシューのスコープ外
/// のため、`hidden` は最終軸 1 個に限定する）であり、非最終軸という
/// 概念自体を持たないため `Option` を返す必要がない。
///
/// - `shape` が rank 0（スカラー）の場合 `ShapeError::RankMismatch`
///   （`expected: 1`）を返す（最終軸自体が存在しないため）。
/// - `hidden = shape[rank-1]`（最終軸のサイズ）、`rows` は残りの
///   先頭次元群の積（`shape[..rank-1]` の要素数積）とする。
///   `checked_numel`（`Tensor::new` 等が使う単一情報源と同じ検査）で
///   `usize` 範囲の乗算オーバーフローを検出し
///   `ShapeError::ElementCountOverflow` を返す。`rows` を
///   `numel / hidden` の除算ではなく先頭次元群の積として直接計算する
///   ことで、`hidden == 0` のゼロ除算を経由しない（`row_softmax_layout`
///   の `checked_div` 吸収より単純）。
pub fn row_norm_layout(shape: &[usize]) -> Result<(usize, usize), ShapeError> {
    let rank = shape.len();
    if rank == 0 {
        return Err(ShapeError::RankMismatch {
            expected: 1,
            actual: 0,
        });
    }
    let hidden = shape[rank - 1];
    let rows = checked_numel(&shape[..rank - 1])?;
    Ok((rows, hidden))
}

/// softmax／log_softmax（イシュー #1594。`BackendOps::softmax`／
/// `log_softmax` の共通入口）が起動前に `(rows, cols)` を導出するための
/// shape 検査。CPU／CUDA／Metal の各 `BackendOps` 実装が本関数の結果を
/// 再導出せず共有する単一情報源とする（3 バックエンドで同じ軸判定
/// ロジックを重複実装しない）。
///
/// - `dim` が `shape` の rank 範囲外の場合 `ShapeError::AxisOutOfRange`
///   を返す（[`reduce_out_shape`] と同じ判定。rank 0 は常に範囲外）。
/// - `dim` が最終軸でない場合（中間軸 softmax）は `Ok(None)` を返す。
///   既存の行カーネル（`backend-cpu::softmax`・`backend-cuda::softmax`・
///   `backend-metal::softmax`）はいずれも最終軸専用であり、呼び出し元
///   （`fandhe_ai_autodiff::var::Var::softmax`／`log_softmax`）はこの
///   `None` を「バックエンドが未対応の軸」の合図としてホスト参照実装
///   （`eval::softmax_along`／`log_softmax_along`）へフォールバックする。
/// - `dim` が最終軸の場合、行優先連続データとして `cols = shape[dim]`・
///   `rows = numel / cols` を返す。`cols == 0` は `numel` も必然的に
///   `0` になるため `checked_div` で `None` を吸収し `rows = 0` とする
///   （ゼロ除算回避。行カーネル側の `rows == 0 || cols == 0` 早期
///   return 契約〈`run_softmax_f32` 等〉と整合する）。
pub fn row_softmax_layout(
    shape: &[usize],
    dim: usize,
) -> Result<Option<(usize, usize)>, ShapeError> {
    let rank = shape.len();
    if dim >= rank {
        return Err(ShapeError::AxisOutOfRange { axis: dim, rank });
    }
    if dim != rank - 1 {
        return Ok(None);
    }
    let cols = shape[dim];
    // `shape.iter().product()` の素朴な乗算は overflow しても panic
    // せず（release ビルドではラップして誤った `numel` を返す）、本番
    // 経路 panic 禁止規約（`.claude/rules/coding-rust.md`）に反する。
    // `checked_numel`（`crate::tensor`。`Tensor::new` 等が使う単一
    // 情報源と同じ検査）で `usize` 範囲の乗算オーバーフローを検出し
    // `ShapeError::ElementCountOverflow` を返す。
    let numel = checked_numel(shape)?;
    let rows = numel.checked_div(cols).unwrap_or(0);
    Ok(Some((rows, cols)))
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
        // [2,3] と [3,2] は末尾軸（3 vs 2）が「同一」「片方が 1」の
        // いずれも満たさないためブロードキャスト非互換。
        let err = elementwise_out_shape(&[2, 3], &[3, 2]).unwrap_err();
        assert!(matches!(err, ShapeError::BroadcastIncompatible { .. }));
    }

    #[test]
    fn elementwise_scalar_rank0_ok() {
        let out = elementwise_out_shape(&[], &[]).unwrap();
        assert_eq!(out, Vec::<usize>::new());
    }

    #[test]
    fn elementwise_broadcast_row_and_column() {
        // 受け入れ条件対象: NumPy 互換ブロードキャストが期待値と一致する
        // 代表例（[3,1] と [1,4] → [3,4]）。
        let out = elementwise_out_shape(&[3, 1], &[1, 4]).unwrap();
        assert_eq!(out, vec![3, 4]);
    }

    #[test]
    fn elementwise_broadcast_rank_difference() {
        // rank 差分（[2,3] と [3]）の暗黙先頭軸補完。
        let out = elementwise_out_shape(&[2, 3], &[3]).unwrap();
        assert_eq!(out, vec![2, 3]);
    }

    #[test]
    fn elementwise_broadcast_incompatible_returns_broadcast_incompatible() {
        let err = elementwise_out_shape(&[2, 3], &[4]).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::BroadcastIncompatible { lhs, rhs }
                if lhs == vec![2, 3] && rhs == vec![4]
        ));
    }

    #[test]
    fn elementwise_broadcast_overflow() {
        // ブロードキャストは出力軸を入力の最大値まで拡張しうるため、
        // 両入力自体は小さくても出力要素数が usize::MAX を超えうる
        // （例: [1, big] と [big, 1] → [big, big]）。matmul_overflow と
        // 同様に checked_numel でオーバーフロー検出されることを確認する
        // （Cursor Bugbot 指摘対応。#22 PR #220）。
        let big = usize::MAX / 2 + 1;
        let err = elementwise_out_shape(&[1, big], &[big, 1]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
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

    // --- row_norm_layout ---

    #[test]
    fn row_norm_layout_rank1() {
        let out = row_norm_layout(&[8]).unwrap();
        assert_eq!(out, (1, 8));
    }

    #[test]
    fn row_norm_layout_2d() {
        let out = row_norm_layout(&[3, 8]).unwrap();
        assert_eq!(out, (3, 8));
    }

    #[test]
    fn row_norm_layout_3d() {
        let out = row_norm_layout(&[2, 3, 8]).unwrap();
        assert_eq!(out, (6, 8));
    }

    #[test]
    fn row_norm_layout_rank0_is_rank_mismatch() {
        let err = row_norm_layout(&[]).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 1,
                actual: 0
            }
        ));
    }

    #[test]
    fn row_norm_layout_zero_hidden_yields_rows_from_leading_dims() {
        // hidden == 0 でも rows は先頭次元群の積として直接計算するため
        // ゼロ除算を経由しない（`row_softmax_layout` の `checked_div`
        // 吸収と異なるアプローチ）。
        let out = row_norm_layout(&[3, 0]).unwrap();
        assert_eq!(out, (3, 0));
    }

    #[test]
    fn row_norm_layout_zero_leading_dim() {
        let out = row_norm_layout(&[0, 4]).unwrap();
        assert_eq!(out, (0, 4));
    }

    #[test]
    fn row_norm_layout_element_count_overflow_is_typed_error() {
        let err = row_norm_layout(&[usize::MAX, 2, 2]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
    }

    // --- row_softmax_layout ---

    #[test]
    fn row_softmax_layout_rank1_last_axis() {
        let out = row_softmax_layout(&[8], 0).unwrap();
        assert_eq!(out, Some((1, 8)));
    }

    #[test]
    fn row_softmax_layout_2d_last_axis() {
        let out = row_softmax_layout(&[2, 8], 1).unwrap();
        assert_eq!(out, Some((2, 8)));
    }

    #[test]
    fn row_softmax_layout_non_final_axis_returns_none() {
        // 中間軸（非最終軸）softmax は行カーネル未対応のためホスト
        // フォールバックの合図として `None` を返す。
        let out = row_softmax_layout(&[2, 8], 0).unwrap();
        assert_eq!(out, None);
    }

    #[test]
    fn row_softmax_layout_3d_non_final_axis_returns_none() {
        let out = row_softmax_layout(&[2, 3, 4], 1).unwrap();
        assert_eq!(out, None);
    }

    #[test]
    fn row_softmax_layout_dim_out_of_range() {
        let err = row_softmax_layout(&[2, 8], 2).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::AxisOutOfRange { axis: 2, rank: 2 }
        ));
    }

    #[test]
    fn row_softmax_layout_rank0_always_out_of_range() {
        let err = row_softmax_layout(&[], 0).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::AxisOutOfRange { axis: 0, rank: 0 }
        ));
    }

    #[test]
    fn row_softmax_layout_zero_cols_yields_zero_rows() {
        // `cols == 0` は `numel` も 0 になるため `checked_div` を `0` へ
        // 吸収する（ゼロ除算回避。行カーネルの `rows == 0 || cols == 0`
        // 早期 return 契約と整合）。
        let out = row_softmax_layout(&[3, 0], 1).unwrap();
        assert_eq!(out, Some((0, 0)));
    }

    #[test]
    fn row_softmax_layout_zero_numel_nonzero_cols() {
        let out = row_softmax_layout(&[0, 4], 1).unwrap();
        assert_eq!(out, Some((0, 4)));
    }

    // codex-review 指摘（PR #1664）の回帰検証: `shape.iter().product()`
    // の素朴な乗算は overflow を検出せず（release ビルドではラップして
    // 誤った `numel` を返す）、本番経路 panic 禁止規約
    // （`.claude/rules/coding-rust.md`）に反していた。`checked_numel`
    // による検査で `ShapeError::ElementCountOverflow` を返すことを
    // 確認する。
    #[test]
    fn row_softmax_layout_element_count_overflow_is_typed_error() {
        let err = row_softmax_layout(&[usize::MAX, 2], 1).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
    }
    #[test]
    fn concat_out_shape_empty_list_is_rank_mismatch() {
        let err = concat_out_shape(&[], 0).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 1,
                actual: 0
            }
        ));
    }

    #[test]
    fn concat_out_shape_basic_dim0() {
        let a: &[usize] = &[2, 3];
        let b: &[usize] = &[4, 3];
        let out = concat_out_shape(&[a, b], 0).unwrap();
        assert_eq!(out, vec![6, 3]);
    }

    #[test]
    fn concat_out_shape_basic_dim1() {
        let a: &[usize] = &[2, 3];
        let b: &[usize] = &[2, 5];
        let out = concat_out_shape(&[a, b], 1).unwrap();
        assert_eq!(out, vec![2, 8]);
    }

    #[test]
    fn concat_out_shape_single_element_is_identity() {
        let a: &[usize] = &[2, 3];
        let out = concat_out_shape(&[a], 0).unwrap();
        assert_eq!(out, vec![2, 3]);
    }

    #[test]
    fn concat_out_shape_rank_mismatch() {
        let a: &[usize] = &[2, 3];
        let b: &[usize] = &[2, 3, 4];
        let err = concat_out_shape(&[a, b], 0).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 3
            }
        ));
    }

    #[test]
    fn concat_out_shape_axis_out_of_range() {
        let a: &[usize] = &[2, 3];
        let err = concat_out_shape(&[a], 2).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::AxisOutOfRange { axis: 2, rank: 2 }
        ));
    }

    #[test]
    fn concat_out_shape_axis_mismatch_on_other_dim() {
        let a: &[usize] = &[2, 3];
        let b: &[usize] = &[2, 4];
        let err = concat_out_shape(&[a, b], 0).unwrap_err();
        match err {
            ShapeError::ShapeMismatch { lhs, rhs } => {
                assert_eq!(lhs, vec![2, 3]);
                assert_eq!(rhs, vec![2, 4]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn concat_out_shape_zero_length_dim_mixed() {
        let a: &[usize] = &[0, 3];
        let b: &[usize] = &[2, 3];
        let out = concat_out_shape(&[a, b], 0).unwrap();
        assert_eq!(out, vec![2, 3]);
    }

    #[test]
    fn concat_out_shape_element_count_overflow() {
        let a: &[usize] = &[usize::MAX, 2];
        let b: &[usize] = &[1, 2];
        let err = concat_out_shape(&[a, b], 0).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
    }
}
