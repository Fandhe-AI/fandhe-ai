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

use crate::backend_ops::{Conv2dParams, Pool2dParams};
use crate::broadcast::broadcast_shape;
use crate::error::ShapeError;
use crate::tensor::{checked_numel, checked_numel_for};

/// matmul の 2 次元厳密版（`docs/public-api-design.md` §3.2）の出力 shape
/// を検査・計算する。
///
/// バッチ対応前（イシュー #1715 以前）の `matmul_out_shape` 本体そのもの。
/// 各バックエンドの GEMM カーネル入口（`backend-cpu`/`backend-cuda`/
/// `backend-metal` の `gemm`/`gemm_checksum`/`gemm_fp32_strict_into`/
/// `gemm_bias_act` 等）はバッチ次元を持たない 2 次元スライス
/// （`BackendOps::gemm_batched` の既定合成実装がバッチをほどいた後の
/// 各バッチ）のみを受け取るため、rank≠2 を明示的に拒否する本関数へ
/// 移行済み（イシュー #1715。#1600 ツリー）。`matmul_out_shape`（本モジュール
/// 下方）はバッチ次元を許容する一般化版であり、2 次元カーネル入口を
/// 誤って rank≥3 で呼び出し「panic せず誤った結果を返す」経路を防ぐため
/// 両者を分離している。
///
/// - `lhs`/`rhs` の rank が 2 でない場合 `ShapeError::RankMismatch`
///   （`expected: 2`）を返す。
/// - 内部次元（`lhs[1]` と `rhs[0]`）が一致しない場合
///   `ShapeError::MatmulDimMismatch` を返す。
/// - 出力 shape `[lhs[0], rhs[1]]` の要素数積のオーバーフローは
///   `checked_numel`（`tensor.rs` と共有）で検査し
///   `ShapeError::ElementCountOverflow` を返す。
pub fn gemm_out_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, ShapeError> {
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

/// matmul（rank≥2。バッチ次元は NumPy 互換ブロードキャスト。
/// `docs/public-api-design.md` §3.2・spec REQ-9 2026-09-12 追記 Tier 1
/// 「バッチ行列積」・`docs/compat-api-scope.md` §1.2）の出力 shape を
/// 検査・計算する。
///
/// `fandhe_ai_autodiff::Var::matmul`（#15。イシュー #1715 でバッチ対応）・
/// `BackendOps::gemm_batched`（既定合成実装。`backend_ops.rs`）から呼ばれ、
/// カーネル実行前に呼び出し元が shape 前提を確認する契約点となる。
/// **カーネル入口（各バックエンドの `gemm`/`gemm_checksum`/`gemm_bias_act`
/// 等）はこの関数を呼ばない**（rank≥3 を誤って受理してしまうため）。
/// カーネル入口は 2 次元厳密版 [`gemm_out_shape`] を使う。
///
/// - `lhs`/`rhs` いずれかの rank が 2 未満の場合 `ShapeError::RankMismatch`
///   （`expected: 2`）を返す（rank 2 同士の挙動・エラーは
///   [`gemm_out_shape`] と完全に一致する）。
/// - 内部次元（`lhs` の最終軸と `rhs` の最後から 2 番目の軸）が一致しない
///   場合 `ShapeError::MatmulDimMismatch` を返す。
/// - バッチ次元（先頭の rank−2 軸）は [`crate::broadcast::broadcast_shape`]
///   で NumPy 互換ブロードキャストする（rank が異なる場合は短い方の先頭に
///   暗黙の軸長 1 を補完。不一致は `ShapeError::BroadcastIncompatible` を
///   そのまま伝播する）。
/// - 出力 shape `batch ++ [m, n]` の要素数積のオーバーフローは
///   `checked_numel` で検査し `ShapeError::ElementCountOverflow` を返す。
pub fn matmul_out_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, ShapeError> {
    Ok(batched_matmul_plan(lhs, rhs)?.out_shape())
}

/// [`matmul_out_shape`] が検査済みの形状情報を保持する計画。
///
/// バッチ行列積の実行（`BackendOps::gemm_batched` 既定合成実装・
/// `CpuBackendOps::gemm_batched`・`matmul_vjp` のバッチ縮約）が共有する。
/// `lhs_batch_shape`/`rhs_batch_shape` は各オペランド自身のバッチ形状
/// （ブロードキャスト前）であり、出力バッチ添字から各オペランドの
/// （ブロードキャスト後）フラット添字への写像に使う。
///
/// 全フィールドを非公開（`pub(self)` 相当）にし、構築は
/// [`batched_matmul_plan`]（本モジュール内で不変条件を検査したうえで
/// 構築する唯一の経路）に限定する。読み取りは下記のアクセサ経由のみ
/// 許可し、フィールドを個別に差し替え可能な `pub` にしない（PR #1810
/// codex-review 指摘。全フィールド `pub` だと呼び出し側が
/// `batch_shape.clear()` 等で `lhs_batch_shape`／`rhs_batch_shape` より
/// rank の小さい不整合な `batch_shape` を作れてしまい、その後
/// `operand_batch_index` を呼ぶと `flat_index` 内の `rank -
/// operand_shape.len()` が `usize` 減算アンダーフローして panic する
/// （`.claude/rules/coding-rust.md` の本番経路 panic 禁止方針）ため）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchedMatmulPlan {
    /// 出力のバッチ形状（`broadcast_shape(lhs_batch, rhs_batch)`）。
    batch_shape: Vec<usize>,
    /// 行列積の m（`lhs` の最後から 2 番目の軸）。
    m: usize,
    /// 行列積の内部次元 k（`lhs` の最終軸・`rhs` の最後から 2 番目の軸）。
    k: usize,
    /// 行列積の n（`rhs` の最終軸）。
    n: usize,
    /// `lhs` 自身のバッチ形状（ブロードキャスト前。`lhs[..rank-2]`）。
    lhs_batch_shape: Vec<usize>,
    /// `rhs` 自身のバッチ形状（ブロードキャスト前。`rhs[..rank-2]`）。
    rhs_batch_shape: Vec<usize>,
}

impl BatchedMatmulPlan {
    /// 出力のバッチ形状（`broadcast_shape(lhs_batch, rhs_batch)`）。
    pub fn batch_shape(&self) -> &[usize] {
        &self.batch_shape
    }

    /// 行列積の m（`lhs` の最後から 2 番目の軸）。
    pub fn m(&self) -> usize {
        self.m
    }

    /// 行列積の内部次元 k（`lhs` の最終軸・`rhs` の最後から 2 番目の軸）。
    pub fn k(&self) -> usize {
        self.k
    }

    /// 行列積の n（`rhs` の最終軸）。
    pub fn n(&self) -> usize {
        self.n
    }

    /// `lhs` 自身のバッチ形状（ブロードキャスト前。`lhs[..rank-2]`）。
    pub fn lhs_batch_shape(&self) -> &[usize] {
        &self.lhs_batch_shape
    }

    /// `rhs` 自身のバッチ形状（ブロードキャスト前。`rhs[..rank-2]`）。
    pub fn rhs_batch_shape(&self) -> &[usize] {
        &self.rhs_batch_shape
    }

    /// 出力 shape（`batch_shape ++ [m, n]`）を返す。
    pub fn out_shape(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.batch_shape.len() + 2);
        out.extend_from_slice(&self.batch_shape);
        out.push(self.m);
        out.push(self.n);
        out
    }

    /// 出力バッチのフラット添字（`batch_shape` を row-major で辿った通し
    /// 番号）から、`lhs`/`rhs` それぞれの（ブロードキャスト後）フラット
    /// バッチ添字を計算する。
    ///
    /// ブロードキャストされた軸（オペランド側の軸長が 1）は常に添字 0
    /// を指す（NumPy の broadcast read と同じ「同一要素の繰り返し読み」。
    /// `docs/public-api-design.md` §2.1）。オペランドのバッチ rank が
    /// 出力より短い場合は、先頭に補完された暗黙の軸長 1 分だけ
    /// `batch_shape` の先頭軸を読み飛ばす。
    pub fn operand_batch_index(&self, out_batch_flat: usize) -> (usize, usize) {
        let rank = self.batch_shape.len();
        let mut multi = vec![0usize; rank];
        let mut rem = out_batch_flat;
        for i in (0..rank).rev() {
            let dim = self.batch_shape[i];
            if dim == 0 {
                multi[i] = 0;
            } else {
                multi[i] = rem % dim;
                rem /= dim;
            }
        }
        let lhs_idx = Self::flat_index(&self.lhs_batch_shape, &multi);
        let rhs_idx = Self::flat_index(&self.rhs_batch_shape, &multi);
        (lhs_idx, rhs_idx)
    }

    /// `operand_shape`（`out_multi` と同じ長さへ右揃えで補完・軸長 1 は
    /// broadcast 添字 0 固定）に対する row-major フラット添字を計算する。
    fn flat_index(operand_shape: &[usize], out_multi: &[usize]) -> usize {
        let rank = out_multi.len();
        let offset = rank - operand_shape.len();
        let mut idx = 0usize;
        for (axis, &dim) in operand_shape.iter().enumerate() {
            let out_axis = axis + offset;
            let coord = if dim == 1 { 0 } else { out_multi[out_axis] };
            idx = idx * dim + coord;
        }
        idx
    }
}

/// [`matmul_out_shape`] の検査本体。出力 shape だけでなく `m`/`k`/`n`・
/// バッチ形状を含む [`BatchedMatmulPlan`] を返す（`BackendOps::gemm_batched`
/// 既定合成実装・`CpuBackendOps::gemm_batched`・`matmul_vjp` から共有）。
pub fn batched_matmul_plan(lhs: &[usize], rhs: &[usize]) -> Result<BatchedMatmulPlan, ShapeError> {
    if lhs.len() < 2 {
        return Err(ShapeError::RankMismatch {
            expected: 2,
            actual: lhs.len(),
        });
    }
    if rhs.len() < 2 {
        return Err(ShapeError::RankMismatch {
            expected: 2,
            actual: rhs.len(),
        });
    }
    let lhs_rank = lhs.len();
    let rhs_rank = rhs.len();
    let m = lhs[lhs_rank - 2];
    let k_lhs = lhs[lhs_rank - 1];
    let k_rhs = rhs[rhs_rank - 2];
    let n = rhs[rhs_rank - 1];
    if k_lhs != k_rhs {
        return Err(ShapeError::MatmulDimMismatch {
            lhs: lhs.to_vec(),
            rhs: rhs.to_vec(),
        });
    }
    let lhs_batch_shape = lhs[..lhs_rank - 2].to_vec();
    let rhs_batch_shape = rhs[..rhs_rank - 2].to_vec();
    let batch_shape = broadcast_shape(&lhs_batch_shape, &rhs_batch_shape)?;

    let mut out = Vec::with_capacity(batch_shape.len() + 2);
    out.extend_from_slice(&batch_shape);
    out.push(m);
    out.push(n);
    checked_numel(&out)?;

    Ok(BatchedMatmulPlan {
        batch_shape,
        m,
        k: k_lhs,
        n,
        lhs_batch_shape,
        rhs_batch_shape,
    })
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

/// `flatten`（`Var::flatten`・`nn::Flatten`。イシュー #2065）の出力
/// shape を検査・計算する。
///
/// PyTorch `torch.flatten(input, start_dim, end_dim)` と同じ規約で
/// `[start_dim, end_dim]`（両端含む）の連続軸を 1 軸へ潰す。
/// `Var::flatten`（`crates/autodiff/src/var.rs`）のインライン実装
/// （イシュー #2065 以前）をそのまま切り出したもので、判定基準は
/// 変更しない——`Var::flatten`（tape 経路）と新設
/// `nn::Flatten::forward_host`（tape 不要経路。`compat::Sequential::
/// predict` が使う）の両方から本関数を呼ぶことで、2 経路間の
/// 判定基準が食い違う「判定迂回経路」を作らない（`Softmax::
/// forward_host` が [`reduce_out_shape`] を tape 経路と共有する
/// 既存パターンの踏襲。`.claude/rules/security.md` A08）。
///
/// - `shape` が rank 0（スカラー）の場合: `start_dim == 0 &&
///   end_dim == 0` のみ許容し出力 shape `[1]` を返す（PyTorch の
///   `torch.flatten` がスカラーを 1 要素ベクトルへ変換する挙動に
///   揃える）。それ以外は `ShapeError::AxisOutOfRange { axis: end_dim,
///   rank: 0 }` を返す。
/// - `end_dim >= rank` の場合 `ShapeError::AxisOutOfRange`。
/// - `start_dim > end_dim` の場合 `ShapeError::AxisOutOfRange`
///   （`axis: start_dim`）。
/// - 潰す軸区間 `shape[start_dim..=end_dim]` の部分積は
///   `checked_mul` で計算し、オーバーフロー時（ゼロ長軸を含む形状
///   でも debug panic・release ラップを起こさないための境界検査。
///   REQ-8 趣旨の境界検査。`.claude/rules/coding-rust.md`）
///   `ShapeError::ElementCountOverflow` を返す。
/// - 出力 shape は `shape[..start_dim]` ++ `[flattened]` ++
///   `shape[end_dim + 1..]`。
pub fn flatten_out_shape(
    shape: &[usize],
    start_dim: usize,
    end_dim: usize,
) -> Result<Vec<usize>, ShapeError> {
    let rank = shape.len();
    if rank == 0 {
        if start_dim == 0 && end_dim == 0 {
            return Ok(vec![1]);
        }
        return Err(ShapeError::AxisOutOfRange {
            axis: end_dim,
            rank,
        });
    }
    if end_dim >= rank {
        return Err(ShapeError::AxisOutOfRange {
            axis: end_dim,
            rank,
        });
    }
    if start_dim > end_dim {
        return Err(ShapeError::AxisOutOfRange {
            axis: start_dim,
            rank,
        });
    }
    let flattened = match shape[start_dim..=end_dim]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
    {
        Some(n) => n,
        None => return Err(ShapeError::ElementCountOverflow),
    };
    let mut out_shape: Vec<usize> = shape[..start_dim].to_vec();
    out_shape.push(flattened);
    out_shape.extend_from_slice(&shape[end_dim + 1..]);
    Ok(out_shape)
}

#[cfg(test)]
mod flatten_out_shape_tests {
    use super::*;

    #[test]
    fn rank0_start_end_zero_wraps_to_singleton() {
        assert_eq!(flatten_out_shape(&[], 0, 0).unwrap(), vec![1]);
    }

    #[test]
    fn rank0_nonzero_dims_are_rejected() {
        assert!(matches!(
            flatten_out_shape(&[], 0, 1),
            Err(ShapeError::AxisOutOfRange { axis: 1, rank: 0 })
        ));
    }

    #[test]
    fn full_flatten_collapses_all_axes() {
        assert_eq!(flatten_out_shape(&[2, 3, 4], 0, 2).unwrap(), vec![24]);
    }

    #[test]
    fn partial_flatten_preserves_untouched_axes() {
        // PyTorch `torch.flatten(x, 1, 2)` 相当: [N, C, H, W] → [N, C*H, W]
        assert_eq!(
            flatten_out_shape(&[2, 3, 4, 5], 1, 2).unwrap(),
            vec![2, 12, 5]
        );
    }

    #[test]
    fn single_axis_range_is_identity() {
        assert_eq!(flatten_out_shape(&[2, 3, 4], 1, 1).unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn end_dim_out_of_range_is_rejected() {
        assert!(matches!(
            flatten_out_shape(&[2, 3], 0, 5),
            Err(ShapeError::AxisOutOfRange { axis: 5, rank: 2 })
        ));
    }

    #[test]
    fn start_dim_greater_than_end_dim_is_rejected() {
        assert!(matches!(
            flatten_out_shape(&[2, 3, 4], 2, 0),
            Err(ShapeError::AxisOutOfRange { axis: 2, rank: 3 })
        ));
    }

    #[test]
    fn element_count_overflow_is_rejected() {
        assert!(matches!(
            flatten_out_shape(&[usize::MAX, 2], 0, 1),
            Err(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn zero_length_axis_in_range_yields_zero_without_panic() {
        // ゼロ長軸を含む区間の部分積は 0（`checked_mul` はオーバーフロー
        // しない）。REQ-8 趣旨の境界検査が panic・ラップを起こさないこと
        // の確認。
        assert_eq!(
            flatten_out_shape(&[0, usize::MAX, 2], 0, 1).unwrap(),
            vec![0, 2]
        );
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

/// `gather`（`Var::gather`／`index_select`。イシュー #1776）の出力
/// shape を検査・計算する。`torch.gather` と同じ意味論: 出力 shape は
/// `index_shape` そのもの（各出力位置ごとに `input` から 1 要素を
/// 独立に読み出す）。
///
/// - `dim >= input_shape` の rank の場合 `ShapeError::AxisOutOfRange`
///   （`input_shape` の rank を基準とする）。
/// - `index_shape` の rank が `input_shape` と一致しない場合
///   `ShapeError::RankMismatch`（`expected` は `input_shape` の rank）。
/// - `dim` 軸以外の各軸で `input_shape[axis] != index_shape[axis]`
///   の場合 `ShapeError::ShapeMismatch`（`lhs`=`input_shape`・
///   `rhs`=`index_shape`）。`dim` 軸自体は `index_shape[dim]` が
///   `input_shape[dim]` と異なってもよい（各出力位置の添字値が
///   `input` の `dim` 軸範囲内かどうかは値検査〈呼び出し元が
///   データを走査して行う〉の対象であり、本関数の shape 検査対象
///   ではない）。
///
/// 出力要素数は `index_shape` と同一のため、`concat_out_shape` と
/// 異なり `checked_numel` によるオーバーフロー検査は不要（`index_shape`
/// 自体が既に有効なテンソル shape として構築済みであることを呼び出し元
/// が保証する）。
pub fn gather_out_shape(
    input_shape: &[usize],
    index_shape: &[usize],
    dim: usize,
) -> Result<Vec<usize>, ShapeError> {
    let rank = input_shape.len();
    if dim >= rank {
        return Err(ShapeError::AxisOutOfRange { axis: dim, rank });
    }
    if index_shape.len() != rank {
        return Err(ShapeError::RankMismatch {
            expected: rank,
            actual: index_shape.len(),
        });
    }
    for (axis, (&in_s, &idx_s)) in input_shape.iter().zip(index_shape.iter()).enumerate() {
        if axis == dim {
            continue;
        }
        if in_s != idx_s {
            return Err(ShapeError::ShapeMismatch {
                lhs: input_shape.to_vec(),
                rhs: index_shape.to_vec(),
            });
        }
    }
    Ok(index_shape.to_vec())
}

/// `scatter`／`scatter_add`（`Var::scatter`／`scatter_add`。イシュー
/// #1776）の出力 shape を検査・計算する。`torch.scatter` と同じ
/// 意味論のうち、本実装は簡略化のため `index_shape == src_shape` を
/// 要求する（PyTorch の「`index.size(d) <= src.size(d)`」という緩い
/// 制約は対象外。`docs/` 実装計画のスコープ外事項）。
///
/// - `dim >= input_shape` の rank の場合 `ShapeError::AxisOutOfRange`。
/// - `index_shape` の rank が `input_shape` と一致しない場合
///   `ShapeError::RankMismatch`。
/// - `index_shape != src_shape` の場合 `ShapeError::ShapeMismatch`
///   （`lhs`=`index_shape`・`rhs`=`src_shape`）。
/// - `dim` 軸以外の各軸で `index_shape[axis] > input_shape[axis]` の
///   場合 `ShapeError::ShapeMismatch`（`lhs`=`input_shape`・
///   `rhs`=`index_shape`）。`dim` 軸自体は値検査（添字が
///   `[0, input_shape[dim])` の範囲内か）の対象であり本関数の
///   shape 検査対象ではない。
///
/// 出力 shape は常に `input_shape`（scatter は shape を変えない）。
pub fn scatter_out_shape(
    input_shape: &[usize],
    index_shape: &[usize],
    src_shape: &[usize],
    dim: usize,
) -> Result<Vec<usize>, ShapeError> {
    let rank = input_shape.len();
    if dim >= rank {
        return Err(ShapeError::AxisOutOfRange { axis: dim, rank });
    }
    if index_shape.len() != rank {
        return Err(ShapeError::RankMismatch {
            expected: rank,
            actual: index_shape.len(),
        });
    }
    if index_shape != src_shape {
        return Err(ShapeError::ShapeMismatch {
            lhs: index_shape.to_vec(),
            rhs: src_shape.to_vec(),
        });
    }
    for (axis, (&in_s, &idx_s)) in input_shape.iter().zip(index_shape.iter()).enumerate() {
        if axis == dim {
            continue;
        }
        if idx_s > in_s {
            return Err(ShapeError::ShapeMismatch {
                lhs: input_shape.to_vec(),
                rhs: index_shape.to_vec(),
            });
        }
    }
    Ok(input_shape.to_vec())
}

/// `pad`（`Var::pad`。`torch.nn.functional.pad(mode='constant')` 相当。
/// イシュー #1756）の出力 shape を検査・計算する。各軸を
/// `(before, after)` だけ定数値で拡張する（負パディング・
/// `reflect`／`replicate` モードは対象外。`docs/compat-feature-gap.md`
/// 追補参照）。
///
/// - `pads.len() != shape` の rank の場合 `ShapeError::RankMismatch`
///   （`expected`=`shape` の rank・`actual`=`pads.len()`）。rank 0
///   （`shape` が空・`pads` も空）は恒等（空 `Vec` を返す）。
/// - 各軸で `shape[axis] + before + after` を `checked_add` で 2 回
///   検査し、オーバーフローする場合 `ShapeError::ElementCountOverflow`。
/// - 最後に `checked_numel_for::<f32>`（`crate::tensor`。`randn`/`rand`
///   等が使う単一情報源と同じ検査）で出力要素数積の `usize`
///   オーバーフローに加え、`f32` で確保した場合のバイトサイズが
///   `Vec` の allocation 上限（`isize::MAX` バイト）に収まるかも
///   検査する。全バックエンド（CPU／CUDA／Metal／`autodiff::eval`）は
///   `Tensor<f32>` を確保するため出力バッファは常に `f32` 単位であり、
///   `checked_numel` 単体（`usize` 要素数積のみの検査）では
///   `shape=[1]`・`pads=[(0, usize::MAX-1)]` のように要素数自体は
///   オーバーフローしない shape を通過させてしまい、後続の
///   `Vec::with_capacity(numel)` が `numel * size_of::<f32>() >
///   isize::MAX` で capacity overflow パニックする（本番経路 panic
///   禁止規約 `.claude/rules/coding-rust.md` に反する DoS 経路。
///   `checked_numel_for` の doc と同じ理由。イシュー #1756・
///   PR #1831 codex-review P1 是正）。
///
/// `pads` の各要素は先頭次元から順に対応する（PyTorch `F.pad` の
/// 「末尾次元から逆順の平坦リスト」とは異なる意図的な設計。
/// `docs/compat-feature-gap.md` 追補参照）。
pub fn pad_out_shape(shape: &[usize], pads: &[(usize, usize)]) -> Result<Vec<usize>, ShapeError> {
    let rank = shape.len();
    if pads.len() != rank {
        return Err(ShapeError::RankMismatch {
            expected: rank,
            actual: pads.len(),
        });
    }
    let mut out = Vec::with_capacity(rank);
    for (&s, &(before, after)) in shape.iter().zip(pads.iter()) {
        let with_before = s
            .checked_add(before)
            .ok_or(ShapeError::ElementCountOverflow)?;
        let total = with_before
            .checked_add(after)
            .ok_or(ShapeError::ElementCountOverflow)?;
        out.push(total);
    }
    checked_numel_for::<f32>(&out)?;
    Ok(out)
}

/// `sort`（`Var::sort`／`argsort`。`torch.sort` 相当。イシュー #1733）の
/// 出力 shape を検査する。sort は `dim` 軸のみを並べ替える純粋な
/// 並べ替えであり、出力 shape は `shape` と恒等（`gather` と異なり
/// index 側の shape も常に `shape` と一致するため、本関数は
/// [`gather_out_shape`] のような index shape 引数を取らない）。
///
/// - `dim >= shape` の rank の場合 `ShapeError::AxisOutOfRange` を返す
///   （[`reduce_out_shape`] と同じ判定）。
pub fn sort_out_shape(shape: &[usize], dim: usize) -> Result<Vec<usize>, ShapeError> {
    let rank = shape.len();
    if dim >= rank {
        return Err(ShapeError::AxisOutOfRange { axis: dim, rank });
    }
    Ok(shape.to_vec())
}

/// `topk`（`Var::topk`。`torch.topk` 相当。イシュー #1733）の出力
/// shape を検査・計算する。`topk` は `dim` 軸を `k` 個へ narrow した
/// 形（`Tensor::narrow(dim, 0, k)` と同じ範囲意味論）を返すため、
/// `k` の範囲検査は新規 variant を追加せず既存の
/// [`ShapeError::NarrowOutOfBounds`] を流用する（`narrow` と topk は
/// いずれも「`dim` 軸を `[0, k)` の範囲へ切り詰める」操作として同一
/// エラー分類が妥当と判断したため）。
///
/// - `dim >= shape` の rank の場合 `ShapeError::AxisOutOfRange`。
/// - `k > shape[dim]` の場合 `ShapeError::NarrowOutOfBounds`
///   （`start: 0`・`len: k`・`dim_size: shape[dim]`）。`k == 0` は
///   出力の `dim` 軸が 0 要素になるだけで許容する（`Tensor::narrow`
///   の `len == 0` と同じ扱い）。
pub fn topk_out_shape(shape: &[usize], dim: usize, k: usize) -> Result<Vec<usize>, ShapeError> {
    let rank = shape.len();
    if dim >= rank {
        return Err(ShapeError::AxisOutOfRange { axis: dim, rank });
    }
    let dim_size = shape[dim];
    if k > dim_size {
        return Err(ShapeError::NarrowOutOfBounds {
            dim,
            start: 0,
            len: k,
            dim_size,
        });
    }
    let mut out = shape.to_vec();
    out[dim] = k;
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

/// BatchNorm1d／2d（`Var::batch_norm`／`batch_norm_infer`。PyTorch
/// `nn.BatchNorm1d`／`BatchNorm2d` 相当。イシュー #1732・親 #1608）が
/// 起動前に `(n, c, spatial)` を導出するための shape 検査。チャネル軸は
/// 常に dim 1（NCHW／NCL 固定。`docs/conv-ops-design.md` と同じレイアウト
/// 契約）。
///
/// - rank 2（`[N, C]`。`BatchNorm1d` の非空間入力）は `spatial = 1`。
/// - rank 3（`[N, C, L]`。`BatchNorm1d` の空間入力）は `spatial = L`。
/// - rank 4（`[N, C, H, W]`。`BatchNorm2d`）は `spatial = H*W`。
///   `N`／`C`／`H`／`W` の**いずれか**が `0` の場合（`x` 全体が空テンソル
///   になる場合）は積を計算せず `spatial = 0` を返す。**4 軸すべてが
///   非ゼロの場合に限り** `checked_numel(&shape[2..])` で `usize`
///   範囲の乗算オーバーフローを検出し `ShapeError::
///   ElementCountOverflow` を返す（`H*W` だけでは `N`／`C` 側の `0` を
///   反映できないため、`N=0, C=1, H=usize::MAX, W=2` のような有効な
///   空テンソルが `H*W` 単体のオーバーフロー検査で不当に拒否される
///   のを防ぐ。`interpolate_out_shape` の「非ゼロ次元のみ
///   `checked_numel`」規約と同型。Cursor Bugbot 指摘・イシュー
///   #1732・PR #1874 fix ループ）。
/// - rank 0・1・5 以上は `ShapeError::RankMismatch`（`expected` は
///   最も近い許容 rank。rank 0・1 は `2`、rank 5 以上は `4`）を返す。
///
/// `BatchNorm1d`／`BatchNorm2d` それぞれの rank 限定（1d はさらに
/// rank 2/3 のみ・2d は rank 4 のみに絞る）は `nn::BatchNorm1d`／
/// `BatchNorm2d` 側が検査する（`Var::batch_norm`／`batch_norm_infer`
/// 自体は rank 2〜4 を一様に受理する）。`M = n * spatial`
/// （チャネルごとの縮約要素数）の導出は呼び出し側の責務とする
/// （train モードは `M <= 1` を追加で拒否する契約のため）。
pub fn batch_norm_layout(shape: &[usize]) -> Result<(usize, usize, usize), ShapeError> {
    let rank = shape.len();
    match rank {
        0 | 1 => Err(ShapeError::RankMismatch {
            expected: 2,
            actual: rank,
        }),
        2 => Ok((shape[0], shape[1], 1)),
        3 => Ok((shape[0], shape[1], shape[2])),
        4 => {
            // `shape` の 4 軸（`N`／`C`／`H`／`W`）のいずれかが `0` なら
            // `x` 全体が空テンソルであり `spatial`（`H*W`）は数学的に
            // 意味を持たないため積の計算自体を避ける（doc comment
            // 参照）。`N`／`C` は `shape[2..]` に現れないため、
            // `checked_numel(&shape[2..])` 単体の overflow 検査だけでは
            // `N=0` や `C=0` を伴う空テンソルを正しく救えない。
            let spatial = if shape.contains(&0) {
                0
            } else {
                checked_numel(&shape[2..])?
            };
            Ok((shape[0], shape[1], spatial))
        }
        _ => Err(ShapeError::RankMismatch {
            expected: 4,
            actual: rank,
        }),
    }
}

#[cfg(test)]
mod batch_norm_layout_tests {
    use super::*;

    #[test]
    fn rank2_is_n_c_spatial_one() {
        assert_eq!(batch_norm_layout(&[8, 3]).unwrap(), (8, 3, 1));
    }

    #[test]
    fn rank3_spatial_is_length() {
        assert_eq!(batch_norm_layout(&[8, 3, 16]).unwrap(), (8, 3, 16));
    }

    #[test]
    fn rank4_spatial_is_h_times_w() {
        assert_eq!(batch_norm_layout(&[8, 3, 4, 5]).unwrap(), (8, 3, 20));
    }

    #[test]
    fn rank0_and_rank1_are_rejected() {
        assert!(matches!(
            batch_norm_layout(&[]).unwrap_err(),
            ShapeError::RankMismatch {
                expected: 2,
                actual: 0
            }
        ));
        assert!(matches!(
            batch_norm_layout(&[4]).unwrap_err(),
            ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn rank5_is_rejected() {
        assert!(matches!(
            batch_norm_layout(&[1, 2, 3, 4, 5]).unwrap_err(),
            ShapeError::RankMismatch {
                expected: 4,
                actual: 5
            }
        ));
    }

    #[test]
    fn rank4_spatial_overflow_is_element_count_overflow() {
        let err = batch_norm_layout(&[1, 1, usize::MAX, 2]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
    }

    #[test]
    fn zero_sized_axes_are_accepted() {
        assert_eq!(batch_norm_layout(&[0, 3, 4]).unwrap(), (0, 3, 4));
        assert_eq!(batch_norm_layout(&[8, 3, 0, 5]).unwrap(), (8, 3, 0));
    }

    /// `N=0` かつ空間軸（`H`）が巨大な rank-4 の空テンソルは、`H*W`
    /// 単体では `usize` オーバーフローするが `x` 全体は空であり有効な
    /// shape である。積を計算せず `spatial=0` を返す（`checked_numel`
    /// を経由しない）ことを確認する（Cursor Bugbot 指摘・イシュー
    /// #1732・PR #1874 fix ループ）。
    #[test]
    fn rank4_empty_leading_axis_with_huge_spatial_does_not_overflow() {
        assert_eq!(
            batch_norm_layout(&[0, 3, usize::MAX, 2]).unwrap(),
            (0, 3, 0)
        );
    }

    /// `C=0`（`N`／空間軸は非ゼロかつ空間軸が巨大）の同型ケース。
    #[test]
    fn rank4_empty_channel_axis_with_huge_spatial_does_not_overflow() {
        assert_eq!(
            batch_norm_layout(&[5, 0, usize::MAX, 2]).unwrap(),
            (5, 0, 0)
        );
    }

    /// 4 軸すべてが非ゼロの場合は従来どおり `checked_numel(&shape[2..])`
    /// によるオーバーフロー拒否が維持されることを確認する（非空
    /// テンソルでの誤った救済がないことの回帰）。
    #[test]
    fn rank4_nonempty_spatial_overflow_is_still_rejected() {
        let err = batch_norm_layout(&[1, 1, usize::MAX, 2]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
        let err = batch_norm_layout(&[2, 3, usize::MAX, 2]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
    }
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

/// `interpolate`（`Var::interpolate`。`torch.nn.functional.interpolate`
/// 相当。イシュー #1757）の出力 shape を検査・計算する。空間軸は
/// **末尾 `size.len()` 軸**（先頭の残り軸——batch／channel 等——は
/// 素通しで shape のまま残る。`BackendOps::interpolate` doc 参照）。
///
/// - `size.is_empty()` の場合 `ShapeError::RankMismatch { expected: 1,
///   actual: 0 }`（空間軸は最低 1 軸必要。[`row_norm_layout`] の
///   rank-0 拒否と同じ「最小必要軸数」の表現方針を踏襲）。
/// - `size.len() > shape.len()` の場合 `ShapeError::RankMismatch
///   { expected: shape.len(), actual: size.len() }`（空間軸数が入力の
///   rank を超えられない）。
/// - 空間軸のいずれかで `shape[axis] == 0` または `size[i] == 0` の
///   場合 `ShapeError::ShapeMismatch { lhs: shape.to_vec(), rhs:
///   size.to_vec() }` を返す（新規 variant を追加せず既存の最も近い
///   variant を流用する方針。`sort_out_shape`／`topk_out_shape` が
///   `NarrowOutOfBounds` を流用するのと同じ判断）。添字式
///   `src = (dst * in) / out`（整数除算）はゼロ除算を避けるため
///   `out == 0` を、対応する `src` 座標が存在しないことを避けるため
///   `in == 0` を、それぞれ事前に拒否する。**先頭の残り軸（空間軸
///   以外）が 0 の空入力は許容する**（この場合出力も先頭軸が 0 の
///   空 shape になるだけで、空間軸の走査自体が発生しないため）。
/// - 出力 shape は `shape[..shape.len()-size.len()]`（先頭の残り軸を
///   そのまま）に `size`（空間軸）を連結した形。要素数積のオーバー
///   フローに加え、出力実体（`eval::interpolate_nearest`・CPU／CUDA／
///   Metal 実装いずれも `f32` 出力）を確保した際のバイトサイズが
///   `Vec` の allocation 上限（`isize::MAX` バイト）を超えないかも
///   `checked_numel_for::<f32>`（`one_hot_out_shape` と同じ単一情報源）
///   で検査する。要素数積が `usize` に収まっても、例えば入力
///   `shape=[1]` を `size=[usize::MAX]` へ interpolate する場合の
///   ように出力バイトサイズが超過する shape を確保前に拒否し、
///   `ElementCountOverflow` を返す（本番経路 panic 禁止規約
///   `.claude/rules/coding-rust.md` に反する capacity overflow
///   panic の防止。イシュー #1834 codex-review P1 是正）。
/// - 上記の `checked_numel_for::<f32>(&out)` は出力 shape**全体**の
///   積（先頭軸のいずれかが `0` なら必ず `0`）しか検査しない。
///   このため入力 `shape=[0, 3, 4]`・`size=[usize::MAX, 2]` のように
///   **先頭の残り軸が `0` で空間軸（`size`）側が巨大な多軸**の組合せは
///   出力 shape 全体の積が `0` に短絡し通過してしまう
///   （`checked_mul` は `0 * x` を正確に `0` と評価でき overflow しない
///   ため——PoC-v2-1 起源の `checked_numel` 自体は正しい。「飽和」では
///   なく数学的に正しいゼロ）。しかし各バックエンド・ホスト参照実装が
///   出力**全体**の要素数ではなく空間軸のみ・先頭軸を除いた部分の
///   積（例: 行優先ストライドの再計算・空間平面サイズ）を個別に
///   計算する経路を将来追加した場合、その部分積は `0` を含まないため
///   `usize::MAX * 2` のように**真に `usize` 乗算がオーバーフロー**
///   しうる（本番経路 panic 禁止規約 `.claude/rules/coding-rust.md`
///   に反する DoS 経路。`concat_out_shape`／`softmax_along` 等で既に
///   発生した「先頭軸の `0` が部分積の overflow 検査を素通りさせる」
///   bug と同型。`crates/autodiff/src/eval.rs` の
///   `concat_empty_out_shape_overflow_tests`・
///   `softmax_empty_tensor_overflow_tests` 参照）。よって単一情報源
///   である本関数で**出力 shape のうち `0` でない次元のみを集めた
///   列**に対しても `checked_numel`（`usize` 乗算オーバーフロー検査
///   のみ）を追加適用し、あらゆる部分積（`0` を含む部分積は自明に
///   `0` で安全）が `usize` 乗算として成立することまで検査する。
///   `0` でない次元はすべて `1` 以上のため、それらの任意の部分列の積
///   は全体（フィルタ後）の積以下になり、全体（フィルタ後）の積が
///   `usize` に収まれば任意の部分積も必ず収まる（Cursor Bugbot 指摘。
///   イシュー #1834）。
///
///   **`checked_numel_for::<f32>` ではなく `checked_numel` を使う
///   理由**: バイトサイズ上限（`isize::MAX` バイト）検査は、その
///   shape 丸ごとのバッファが実際に確保される場合にのみ意味を持つ。
///   `out` 全体の要素数が `0`（先頭軸に `0` を含む）の場合は、
///   `checked_numel_for::<f32>(&out)` が既に判定したとおり実際の
///   バッファは確保されない（各バックエンド・ホスト参照実装は
///   `numel == 0` で空 `Vec` を早期 return する契約）。よって単一の
///   巨大な空間軸（例: `shape=[0, 1]` を `size=[usize::MAX]` へ
///   interpolate。`crates/autodiff/tests/backward.rs::
///   interpolate_nearest_backward_empty_leading_axis_with_huge_size_
///   does_not_panic` が確立済みの契約）まで拒否するのは過剰であり、
///   `nonzero_dims` 側は「`usize` 乗算として成立するか」のみを検査
///   し、バイトサイズ上限は課さない。
pub fn interpolate_out_shape(shape: &[usize], size: &[usize]) -> Result<Vec<usize>, ShapeError> {
    let rank = shape.len();
    if size.is_empty() {
        return Err(ShapeError::RankMismatch {
            expected: 1,
            actual: 0,
        });
    }
    if size.len() > rank {
        return Err(ShapeError::RankMismatch {
            expected: rank,
            actual: size.len(),
        });
    }
    let spatial_start = rank - size.len();
    for (i, &out_sp) in size.iter().enumerate() {
        let axis = spatial_start + i;
        if shape[axis] == 0 || out_sp == 0 {
            return Err(ShapeError::ShapeMismatch {
                lhs: shape.to_vec(),
                rhs: size.to_vec(),
            });
        }
    }
    let mut out = shape.to_vec();
    out[spatial_start..].copy_from_slice(size);
    checked_numel_for::<f32>(&out)?;
    // 上記の全体積検査だけでは `0` を含む出力 shape の任意の部分積
    // （空間軸のみ等）の `usize` 乗算オーバーフローを検出できない
    // （doc comment 参照）。`0` 次元を除いた列に対して `checked_numel`
    // （バイトサイズ上限は課さない。`checked_numel_for` と使い分ける
    // 理由も doc comment 参照）を適用し、その部分積以下になるあらゆる
    // 部分積が `usize` 乗算として成立することを保証する。
    let nonzero_dims: Vec<usize> = out.iter().copied().filter(|&d| d != 0).collect();
    checked_numel(&nonzero_dims)?;
    Ok(out)
}

/// [`interpolate_out_shape`] に `mode`（[`crate::InterpolateMode`]。
/// イシュー #1762）別の追加検査を重ねた版。`Nearest` は
/// `interpolate_out_shape` と完全に同じ（追加検査なし）。`Bilinear`
/// は空間軸がちょうど 2 軸（`size.len() == 2`。末尾 2 軸 = `(H, W)`）
/// であることを追加で要求し、それ以外は
/// `ShapeError::RankMismatch { expected: 2, actual: size.len() }`
/// で拒否する（`linear`〈1 次元〉／`trilinear`〈3 次元〉／`bicubic`
/// は対象外。実装計画「設計判断」§3.1）。
pub fn interpolate_out_shape_for_mode(
    shape: &[usize],
    size: &[usize],
    mode: crate::backend_ops::InterpolateMode,
) -> Result<Vec<usize>, ShapeError> {
    if let crate::backend_ops::InterpolateMode::Bilinear { .. } = mode
        && size.len() != 2
    {
        return Err(ShapeError::RankMismatch {
            expected: 2,
            actual: size.len(),
        });
    }
    interpolate_out_shape(shape, size)
}

#[cfg(test)]
mod interpolate_out_shape_for_mode_tests {
    use super::*;
    use crate::backend_ops::InterpolateMode;

    #[test]
    fn nearest_mode_is_identical_to_mode_agnostic_fn() {
        let a = interpolate_out_shape(&[2, 8], &[3]).unwrap();
        let b = interpolate_out_shape_for_mode(&[2, 8], &[3], InterpolateMode::Nearest).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn bilinear_mode_rejects_rank_other_than_two() {
        let mode = InterpolateMode::Bilinear {
            align_corners: false,
        };
        let err = interpolate_out_shape_for_mode(&[2, 3, 4], &[9], mode).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }
        ));
        let err = interpolate_out_shape_for_mode(&[2, 3, 4], &[9, 9, 9], mode).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 3
            }
        ));
    }

    #[test]
    fn bilinear_mode_accepts_rank_two_and_matches_base_shape_calc() {
        let mode = InterpolateMode::Bilinear {
            align_corners: true,
        };
        let out = interpolate_out_shape_for_mode(&[2, 4, 4], &[9, 9], mode).unwrap();
        let base = interpolate_out_shape(&[2, 4, 4], &[9, 9]).unwrap();
        assert_eq!(out, base);
        assert_eq!(out, vec![2, 9, 9]);
    }
}

/// `one_hot`（`Var::one_hot`。PyTorch `F.one_hot`／TF `tf.one_hot` 相当。
/// イシュー #1755）の出力 shape を検査・計算する。非微分演算（VJP は
/// ゼロ扱い。`crates/autodiff/src/grad.rs::vjp` の `Op::OneHot` 分岐）
/// であり、本関数は `index_shape`（クラス id を保持する整数値
/// テンソルの shape）へ末尾軸として `num_classes` を付加した shape を
/// 返す。`gather_out_shape` と異なり `index_shape` と `input_shape` を
/// 突き合わせる対象が無いため rank 検査は不要。
///
/// - `num_classes == 0` の場合、付加される軸のサイズが 0 になり
///   如何なる添字も範囲外になる（“0 個のクラスに 1 つを立てる”操作が
///   意味を持たない）ため、新規 `ShapeError` variant を追加せず
///   `ShapeError::IndexOutOfRange { dim: index_shape.len(), index: 0,
///   dim_size: 0 }` を返す（`dim` は付加される軸の位置。既存
///   `IndexOutOfRange` の「範囲外添字」という意味論を「`num_classes`
///   軸そのものが空」というケースへ拡張する解釈）。
/// - 出力 shape `index_shape ++ [num_classes]` の要素数積のオーバー
///   フローは `checked_numel_for::<f32>` で検査し
///   `ShapeError::ElementCountOverflow` を返す（`matmul_out_shape` 等の
///   `checked_numel` 単体とは異なり、出力の実体が常に `Tensor<f32>`
///   （`eval::one_hot`・CPU／CUDA／Metal 実装いずれも `f32` 出力）で
///   あることを踏まえ、要素数積が `usize` に収まっても確保バイト数が
///   `Vec` の allocation 上限（`isize::MAX` バイト）を超えるケース
///   （例: `index_shape = [1]`・`num_classes = usize::MAX` は要素数積
///   としては overflow しないが `f32` 4 バイト換算で必ず超過する）まで
///   ここで一括検査する。`one_hot_out_shape` の戻り値は `Var::one_hot`・
///   CPU（`gather_scatter.rs::one_hot`）・CUDA／Metal 各実装が
///   「呼び出し元で検査済み」の前提で `Vec::with_capacity`／
///   `vec![0f32; numel]` へそのまま渡す単一情報源のため、ここで
///   バイトサイズまで検査しないと下流の確保が capacity overflow で
///   panic しうる（本番経路 panic 禁止規約 `.claude/rules/coding-rust.md`
///   に反する DoS 経路。イシュー #1755・codex-review P1 是正）。
pub fn one_hot_out_shape(
    index_shape: &[usize],
    num_classes: usize,
) -> Result<Vec<usize>, ShapeError> {
    if num_classes == 0 {
        return Err(ShapeError::IndexOutOfRange {
            dim: index_shape.len(),
            index: 0,
            dim_size: 0,
        });
    }
    let mut out = index_shape.to_vec();
    out.push(num_classes);
    checked_numel_for::<f32>(&out)?;
    Ok(out)
}

/// Conv2d の出力空間長を計算する（PyTorch `ConvUtils.h::
/// _conv_output_size` と同式。イシュー #1764・設計 `docs/conv-ops-
/// design.md` §3／§4）:
///
/// ```text
/// conv_out_len(in, k, s, p, d) = floor((in + 2p − d(k−1) − 1) / s) + 1
/// ```
///
/// [`pool_out_len`]（pooling 用。イシュー #1728 で実装済み）と同式
/// だが検査規則（padding 上限の有無）が異なるため別関数として定義し
/// 共有しない（設計 doc §4）。
///
/// - `s == 0` または `k == 0` は呼び出し元（[`Conv2dParams::new`]）が
///   構築時点で拒否する契約だが、本関数は `Conv2dParams` を経由しない
///   直接呼び出しでも panic しないよう独立に検査し
///   [`ShapeError::ElementCountOverflow`] を返す。
/// - 分子 `in + 2p − d(k−1) − 1` が負になる（カーネルの実効受容野が
///   パディング済み入力を超える）場合は `checked_sub` で検出し
///   [`ShapeError::ShapeMismatch`]（`lhs`＝パディング済み入力長・
///   `rhs`＝実効受容野長）を返す（負分子拒否ゲート。設計 doc §3）。
///   非負なら整数除算＝floor。
/// - `2p`・`in + 2p`・`d(k−1)` の `usize` オーバーフローは
///   `checked_mul`／`checked_add` で検査し
///   [`ShapeError::ElementCountOverflow`] を返す。
pub fn conv_out_len(
    in_len: usize,
    k: usize,
    s: usize,
    p: usize,
    d: usize,
) -> Result<usize, ShapeError> {
    if s == 0 || k == 0 {
        return Err(ShapeError::ElementCountOverflow);
    }
    let two_p = p.checked_mul(2).ok_or(ShapeError::ElementCountOverflow)?;
    let in_plus_2p = in_len
        .checked_add(two_p)
        .ok_or(ShapeError::ElementCountOverflow)?;
    let k_minus_1 = k - 1;
    let dk = d
        .checked_mul(k_minus_1)
        .ok_or(ShapeError::ElementCountOverflow)?;
    match in_plus_2p.checked_sub(dk).and_then(|v| v.checked_sub(1)) {
        Some(numerator) => Ok(numerator / s + 1),
        None => Err(ShapeError::ShapeMismatch {
            lhs: vec![in_plus_2p],
            rhs: vec![dk.saturating_add(1)],
        }),
    }
}

/// Conv2d（`BackendOps::conv2d`）の出力 shape を検査・計算する
/// （イシュー #1764・設計 `docs/conv-ops-design.md` §3／§4）。
///
/// `input_shape: [N, Cin, H, W]`・`weight_shape: [Cout, Cin/groups,
/// kH, kW]`（`params.kernel_size` ではなく `weight_shape[2..4]` を
/// カーネル空間サイズとして用いる——`Var::conv2d` は `weight` から
/// `kernel_size` を導出して `Conv2dParams` を構築するため、両者は
/// 呼び出し元の契約により常に一致する）。
///
/// 検査順序（PyTorch `check_shape_forward` 相当。設計 doc §3）:
/// rank（input／weight とも 4）→ `Cin % groups == 0` →
/// `weight_shape[1] * groups == Cin` → `Cout % groups == 0` →
/// `Cout >= groups`（`Cout_g >= 1`）→ 空間軸 `H`／`W == 0` 拒否
/// （`N == 0` は受理）→ [`conv_out_len`]（負分子拒否ゲート）→
/// 出力要素数積オーバーフロー（`checked_numel_for`。`f32` 確保時の
/// バイトサイズ上限検査を含む。`crate::tensor` 内非公開）。
pub fn conv2d_out_shape(
    input_shape: &[usize],
    weight_shape: &[usize],
    params: &Conv2dParams,
) -> Result<Vec<usize>, ShapeError> {
    if input_shape.len() != 4 {
        return Err(ShapeError::RankMismatch {
            expected: 4,
            actual: input_shape.len(),
        });
    }
    if weight_shape.len() != 4 {
        return Err(ShapeError::RankMismatch {
            expected: 4,
            actual: weight_shape.len(),
        });
    }
    let (n, cin, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let (cout, cin_g, kh, kw) = (
        weight_shape[0],
        weight_shape[1],
        weight_shape[2],
        weight_shape[3],
    );
    let groups = params.groups();
    if !cin.is_multiple_of(groups) {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![cin],
            rhs: vec![groups],
        });
    }
    let expected_cin_g = cin / groups;
    if cin_g != expected_cin_g {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![cin_g],
            rhs: vec![expected_cin_g],
        });
    }
    if !cout.is_multiple_of(groups) {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![cout],
            rhs: vec![groups],
        });
    }
    if cout < groups {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![cout],
            rhs: vec![groups],
        });
    }
    if h == 0 || w == 0 {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![h, w],
            rhs: vec![1, 1],
        });
    }
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();
    let hout = conv_out_len(h, kh, sh, ph, dh)?;
    let wout = conv_out_len(w, kw, sw, pw, dw)?;
    let out_shape = vec![n, cout, hout, wout];
    checked_numel_for::<f32>(&out_shape)?;
    Ok(out_shape)
}

/// Conv2d の im2col（[`crate::backend_ops::BackendOps::im2col`]）の
/// 出力 shape を検査・計算する（イシュー #1764・設計 `docs/conv-ops-
/// design.md` §4／§8）。`input_shape: [N, Cin, H, W]` を
/// `[N, G, Cin_g·kH·kW, Hout·Wout]` へ展開する（`K_g` 軸は
/// `(c_in_g, kh, kw)` の row-major・`P` 軸は `(oh, ow)` の
/// row-major）。
pub fn im2col_out_shape(
    input_shape: &[usize],
    params: &Conv2dParams,
) -> Result<Vec<usize>, ShapeError> {
    if input_shape.len() != 4 {
        return Err(ShapeError::RankMismatch {
            expected: 4,
            actual: input_shape.len(),
        });
    }
    let (n, cin, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let groups = params.groups();
    if !cin.is_multiple_of(groups) {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![cin],
            rhs: vec![groups],
        });
    }
    if h == 0 || w == 0 {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![h, w],
            rhs: vec![1, 1],
        });
    }
    let cin_g = cin / groups;
    let [kh, kw] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();
    let hout = conv_out_len(h, kh, sh, ph, dh)?;
    let wout = conv_out_len(w, kw, sw, pw, dw)?;
    let k_g = cin_g
        .checked_mul(kh)
        .and_then(|v| v.checked_mul(kw))
        .ok_or(ShapeError::ElementCountOverflow)?;
    let p = hout
        .checked_mul(wout)
        .ok_or(ShapeError::ElementCountOverflow)?;
    let out_shape = vec![n, groups, k_g, p];
    checked_numel_for::<f32>(&out_shape)?;
    Ok(out_shape)
}

/// Pooling（`MaxPool2d`／`AvgPool2d`）の出力空間長を計算する
/// （イシュー #1728・設計 `docs/pooling-ops-design.md` §4）:
///
/// ```text
/// pool_out_len(in, k, s, p, d) = floor((in + 2p − d(k−1) − 1) / s) + 1
/// ```
///
/// [`conv_out_len`] と同式だが検査規則（padding 上限の有無・空窓拒否
/// 検査の要否）が異なるため呼び出し元（[`pool2d_out_shape`]）を分離
/// し共有しない（設計 doc §4「`pool_out_len`（未実装）と同式だが検査
/// 規則が異なるため別関数として定義し共有しない」）。
///
/// 本関数自体の契約は [`conv_out_len`] と同一: `s == 0`／`k == 0` は
/// [`ShapeError::ElementCountOverflow`]・分子（`in + 2p − d(k−1) − 1`）
/// が負になる場合は floor 契約を保つため除算せず
/// [`ShapeError::ShapeMismatch`]（負分子拒否ゲート）・`2p`／`in + 2p`／
/// `d(k−1)` の `usize` オーバーフローは [`ShapeError::
/// ElementCountOverflow`] を返す。
pub fn pool_out_len(
    in_len: usize,
    k: usize,
    s: usize,
    p: usize,
    d: usize,
) -> Result<usize, ShapeError> {
    if s == 0 || k == 0 {
        return Err(ShapeError::ElementCountOverflow);
    }
    let two_p = p.checked_mul(2).ok_or(ShapeError::ElementCountOverflow)?;
    let in_plus_2p = in_len
        .checked_add(two_p)
        .ok_or(ShapeError::ElementCountOverflow)?;
    let k_minus_1 = k - 1;
    let dk = d
        .checked_mul(k_minus_1)
        .ok_or(ShapeError::ElementCountOverflow)?;
    match in_plus_2p.checked_sub(dk).and_then(|v| v.checked_sub(1)) {
        Some(numerator) => Ok(numerator / s + 1),
        None => Err(ShapeError::ShapeMismatch {
            lhs: vec![in_plus_2p],
            rhs: vec![dk.saturating_add(1)],
        }),
    }
}

/// Pooling（`BackendOps::max_pool2d`／`avg_pool2d`）の出力 shape を
/// 検査・計算する（イシュー #1728・設計 `docs/pooling-ops-design.md`
/// §3／§4）。
///
/// `input_shape: [N, C, H, W]`。検査順序（設計 doc §3）: rank（4）→
/// 空間軸 `H`／`W == 0` 拒否（`N`／`C == 0` は受理。padding のみで
/// 構成された窓を出力として通過させる誤りを防ぐため、この検査は
/// [`pool_out_len`] の負分子ゲートより前に独立して行う——設計 doc
/// §3「非 adaptive 側は `padding > 0` のとき負分子ゲートだけでは
/// 不十分」）→ [`pool_out_len`]（負分子拒否ゲート）→ `dilation` に
/// よる空窓拒否（`kernel = 2` かつ `dilation > in_len` の構成のみが
/// 到達しうる。`padding <= floor(kernel/2)` 契約〈[`crate::
/// backend_ops::Pool2dParams::new`]〉の下で他の `kernel` では発生
/// しないことを設計 doc §3 で導出済み）→ 出力要素数積オーバーフロー
/// （`checked_numel_for`）。
pub fn pool2d_out_shape(
    input_shape: &[usize],
    params: &Pool2dParams,
) -> Result<Vec<usize>, ShapeError> {
    if input_shape.len() != 4 {
        return Err(ShapeError::RankMismatch {
            expected: 4,
            actual: input_shape.len(),
        });
    }
    let (n, c, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    if h == 0 || w == 0 {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![h, w],
            rhs: vec![1, 1],
        });
    }
    let [kh, kw] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();
    let hout = pool_out_len(h, kh, sh, ph, dh)?;
    let wout = pool_out_len(w, kw, sw, pw, dw)?;
    if kh == 2 && dh > h {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![h],
            rhs: vec![dh],
        });
    }
    if kw == 2 && dw > w {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![w],
            rhs: vec![dw],
        });
    }
    let out_shape = vec![n, c, hout, wout];
    checked_numel_for::<f32>(&out_shape)?;
    Ok(out_shape)
}

/// Adaptive pooling（`BackendOps::adaptive_avg_pool2d`）の出力 shape
/// を検査・計算する（イシュー #1728・設計 `docs/pooling-ops-
/// design.md` §3／§4）。
///
/// `input_shape: [N, C, H, W]`・`output_size: [Hout, Wout]`。検査
/// 順序: rank（4）→ 空間軸 `H`／`W == 0` 拒否（adaptive の
/// `divisor = end − start` 導出が `in = 0` のとき常に `0` になり
/// 0 除算を招くため。`N`／`C == 0` は受理。設計 doc §3「adaptive
/// 側の動機」）→ `output_size` の両軸 `>= 1` 拒否 → 出力要素数積
/// オーバーフロー。
pub fn adaptive_pool2d_out_shape(
    input_shape: &[usize],
    output_size: [usize; 2],
) -> Result<Vec<usize>, ShapeError> {
    if input_shape.len() != 4 {
        return Err(ShapeError::RankMismatch {
            expected: 4,
            actual: input_shape.len(),
        });
    }
    let (n, c, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    if h == 0 || w == 0 {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![h, w],
            rhs: vec![1, 1],
        });
    }
    let [oh, ow] = output_size;
    if oh == 0 || ow == 0 {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![oh, ow],
            rhs: vec![1, 1],
        });
    }
    let out_shape = vec![n, c, oh, ow];
    checked_numel_for::<f32>(&out_shape)?;
    Ok(out_shape)
}

/// Adaptive pooling の動的窓（PyTorch と同式・整数演算のみで決定的。
/// イシュー #1728・設計 `docs/pooling-ops-design.md` §4）:
///
/// ```text
/// start = floor(o * in_len / out_len)
/// end   = ceil((o + 1) * in_len / out_len)
/// ```
///
/// forward（`backend-cpu::pooling`）・VJP（`grad.rs`）双方が本関数を
/// 単一情報源として共有する（`eval::nearest_src_coord` と同じ理由）。
/// `out_len == 0` または `usize` オーバーフローは `None` を返す
/// （呼び出し元は [`adaptive_pool2d_out_shape`] で `out_len >= 1` を
/// 事前検査済みの前提だが、本関数単体でも panic しないよう独立に
/// 検査する）。
pub fn adaptive_window(o: usize, in_len: usize, out_len: usize) -> Option<(usize, usize)> {
    if out_len == 0 {
        return None;
    }
    let start = o.checked_mul(in_len)? / out_len;
    let end_num = o.checked_add(1)?.checked_mul(in_len)?;
    let end = end_num.checked_add(out_len - 1)?.checked_div(out_len)?;
    Some((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- gemm_out_shape（2 次元厳密版。カーネル入口用） ---

    #[test]
    fn gemm_ok() {
        let out = gemm_out_shape(&[2, 3], &[3, 4]).unwrap();
        assert_eq!(out, vec![2, 4]);
    }

    #[test]
    fn gemm_rank_mismatch_lhs_rank3_rejected() {
        // カーネル入口用の 2 次元厳密版はバッチ次元（rank 3）を拒否する
        // （`matmul_out_shape` の一般化以前の挙動そのもの。イシュー #1715）。
        let err = gemm_out_shape(&[2, 3, 4], &[3, 4]).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 3
            }
        ));
    }

    #[test]
    fn gemm_rank_mismatch_rhs() {
        let err = gemm_out_shape(&[2, 3], &[3]).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn gemm_dim_mismatch() {
        let err = gemm_out_shape(&[2, 3], &[4, 5]).unwrap_err();
        match err {
            ShapeError::MatmulDimMismatch { lhs, rhs } => {
                assert_eq!(lhs, vec![2, 3]);
                assert_eq!(rhs, vec![4, 5]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn gemm_overflow() {
        // usize::MAX に近い次元同士の積は checked_numel でオーバーフロー検出される。
        let big = usize::MAX / 2 + 1;
        let err = gemm_out_shape(&[big, big], &[big, big]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
    }

    #[test]
    fn gemm_zero_size_axis() {
        // 空テンソル（サイズ 0 軸）は形状として妥当。
        let out = gemm_out_shape(&[0, 3], &[3, 4]).unwrap();
        assert_eq!(out, vec![0, 4]);
    }

    // --- matmul_out_shape（rank≥2・バッチ次元 broadcast 対応） ---

    #[test]
    fn matmul_ok_rank2() {
        // rank 2 同士は gemm_out_shape と完全に同じ挙動（結果・エラーとも）。
        let out = matmul_out_shape(&[2, 3], &[3, 4]).unwrap();
        assert_eq!(out, vec![2, 4]);
    }

    #[test]
    fn matmul_batched_equal_batch_shape() {
        let out = matmul_out_shape(&[5, 2, 3], &[5, 3, 4]).unwrap();
        assert_eq!(out, vec![5, 2, 4]);
    }

    #[test]
    fn matmul_batched_broadcast_leading_axis() {
        let out = matmul_out_shape(&[1, 2, 3], &[5, 3, 4]).unwrap();
        assert_eq!(out, vec![5, 2, 4]);
    }

    #[test]
    fn matmul_batched_broadcast_rank_mismatch_lhs_2d() {
        // lhs が 2 次元（暗黙のバッチ rank 0）・rhs がバッチ次元を持つ場合、
        // lhs のバッチ次元は全ブロードキャストされる。
        let out = matmul_out_shape(&[2, 3], &[5, 3, 4]).unwrap();
        assert_eq!(out, vec![5, 2, 4]);
    }

    #[test]
    fn matmul_batched_broadcast_rank_mismatch_rhs_2d() {
        let out = matmul_out_shape(&[5, 2, 3], &[3, 4]).unwrap();
        assert_eq!(out, vec![5, 2, 4]);
    }

    #[test]
    fn matmul_batched_broadcast_multi_axis() {
        // lhs バッチ [2, 1]・rhs バッチ [2, 3] → broadcast_shape で [2, 3]。
        let out = matmul_out_shape(&[2, 1, 2, 3], &[2, 3, 3, 4]).unwrap();
        assert_eq!(out, vec![2, 3, 2, 4]);
    }

    #[test]
    fn matmul_batched_broadcast_incompatible() {
        let err = matmul_out_shape(&[2, 2, 3], &[3, 3, 4]).unwrap_err();
        assert!(matches!(err, ShapeError::BroadcastIncompatible { .. }));
    }

    #[test]
    fn matmul_rank_below_2_rejected_lhs() {
        // rank 1 は 2 次元厳密版と同様に拒否する（rank≥2 未満を許容しない）。
        let err = matmul_out_shape(&[3], &[3, 4]).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn matmul_rank_below_2_rejected_rhs() {
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
    fn matmul_rank3_accepted_now() {
        // バッチ対応前（#1715 以前）は rank 3 を拒否していたが、
        // 本イシューで受理されるようになった（lhs のバッチ軸 [2] を
        // 2 次元 rhs〈暗黙のバッチ rank 0〉へブロードキャスト）。
        let out = matmul_out_shape(&[2, 3, 4], &[4, 5]).unwrap();
        assert_eq!(out, vec![2, 3, 5]);
    }

    #[test]
    fn matmul_batched_dim_mismatch() {
        let err = matmul_out_shape(&[5, 2, 3], &[5, 4, 4]).unwrap_err();
        assert!(matches!(err, ShapeError::MatmulDimMismatch { .. }));
    }

    #[test]
    fn matmul_batched_overflow() {
        let big = usize::MAX / 2 + 1;
        let err = matmul_out_shape(&[2, big, big], &[2, big, big]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
    }

    #[test]
    fn matmul_batched_zero_batch() {
        let out = matmul_out_shape(&[0, 2, 3], &[0, 3, 4]).unwrap();
        assert_eq!(out, vec![0, 2, 4]);
    }

    #[test]
    fn matmul_zero_size_axis() {
        // 空テンソル（サイズ 0 軸）は形状として妥当。
        let out = matmul_out_shape(&[0, 3], &[3, 4]).unwrap();
        assert_eq!(out, vec![0, 4]);
    }

    // --- batched_matmul_plan / BatchedMatmulPlan ---

    #[test]
    fn batched_matmul_plan_fields() {
        let plan = batched_matmul_plan(&[5, 2, 3], &[5, 3, 4]).unwrap();
        assert_eq!(plan.batch_shape(), &[5]);
        assert_eq!(plan.m(), 2);
        assert_eq!(plan.k(), 3);
        assert_eq!(plan.n(), 4);
        assert_eq!(plan.lhs_batch_shape(), &[5]);
        assert_eq!(plan.rhs_batch_shape(), &[5]);
        assert_eq!(plan.out_shape(), vec![5, 2, 4]);
    }

    #[test]
    fn batched_matmul_plan_operand_batch_index_broadcast() {
        // lhs は [1, 2, 3]（バッチ 1）、rhs は [5, 3, 4]（バッチ 5）。
        // 出力バッチ添字 i に対し lhs は常に 0、rhs は i を指す。
        let plan = batched_matmul_plan(&[1, 2, 3], &[5, 3, 4]).unwrap();
        assert_eq!(plan.batch_shape(), &[5]);
        for i in 0..5 {
            assert_eq!(plan.operand_batch_index(i), (0, i));
        }
    }

    #[test]
    fn batched_matmul_plan_operand_batch_index_rank_mismatch() {
        // lhs は 2 次元（暗黙のバッチ rank 0）。rhs は [5, 3, 4]。
        let plan = batched_matmul_plan(&[2, 3], &[5, 3, 4]).unwrap();
        assert_eq!(plan.batch_shape(), &[5]);
        for i in 0..5 {
            assert_eq!(plan.operand_batch_index(i), (0, i));
        }
    }

    #[test]
    fn batched_matmul_plan_operand_batch_index_multi_axis() {
        // lhs=[2,1,2,3]（バッチ [2,1]）・rhs=[3,1,3,4]（バッチ [3,1]、
        // rhs 側は暗黙補完で先頭に 1 軸が付き [1,3,1] 相当ではなく
        // 実際は rank 一致なので lhs_batch=[2,1]・rhs_batch=[3,1]）。
        // 出力バッチは broadcast_shape([2,1],[3,1]) = [2,3] のはず……
        // ではなく末尾軸比較なので [2,1] vs [3,1] は軸0: 2 vs 3 で
        // 不一致になってしまうため、ここでは一致する形状で検証する。
        let plan = batched_matmul_plan(&[2, 3, 2, 3], &[2, 1, 3, 4]).unwrap();
        assert_eq!(plan.batch_shape(), &[2, 3]);
        // out batch (0, 0) -> multi [0, 0] -> lhs idx 0*3+0=0, rhs idx 0*1+0=0
        assert_eq!(plan.operand_batch_index(0), (0, 0));
        // out batch flat 4 -> multi [1, 1] (row-major over [2,3]) ->
        // lhs idx 1*3+1=4, rhs idx 1*1+0=1 (rhs 軸1 は長さ1なので broadcast)
        assert_eq!(plan.operand_batch_index(4), (4, 1));
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

    // --- gather_out_shape ---

    #[test]
    fn gather_out_shape_basic() {
        let out = gather_out_shape(&[3, 4], &[3, 2], 1).unwrap();
        assert_eq!(out, vec![3, 2]);
    }

    #[test]
    fn gather_out_shape_dim_axis_can_grow() {
        // `dim` 軸自体は index 側が input より大きくてもよい（重複読み
        // 出しを許すため）。
        let out = gather_out_shape(&[3, 4], &[3, 10], 1).unwrap();
        assert_eq!(out, vec![3, 10]);
    }

    #[test]
    fn gather_out_shape_axis_out_of_range() {
        let err = gather_out_shape(&[3, 4], &[3, 4], 2).unwrap_err();
        assert_eq!(err, ShapeError::AxisOutOfRange { axis: 2, rank: 2 });
    }

    #[test]
    fn gather_out_shape_rank_mismatch() {
        let err = gather_out_shape(&[3, 4], &[3, 4, 1], 0).unwrap_err();
        assert_eq!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 3
            }
        );
    }

    #[test]
    fn gather_out_shape_other_axis_mismatch() {
        let err = gather_out_shape(&[3, 4], &[2, 4], 1).unwrap_err();
        match err {
            ShapeError::ShapeMismatch { lhs, rhs } => {
                assert_eq!(lhs, vec![3, 4]);
                assert_eq!(rhs, vec![2, 4]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    // --- scatter_out_shape ---

    #[test]
    fn scatter_out_shape_basic() {
        let out = scatter_out_shape(&[3, 4], &[3, 2], &[3, 2], 1).unwrap();
        assert_eq!(out, vec![3, 4]);
    }

    #[test]
    fn scatter_out_shape_axis_out_of_range() {
        let err = scatter_out_shape(&[3, 4], &[3, 4], &[3, 4], 2).unwrap_err();
        assert_eq!(err, ShapeError::AxisOutOfRange { axis: 2, rank: 2 });
    }

    #[test]
    fn scatter_out_shape_rank_mismatch() {
        let err = scatter_out_shape(&[3, 4], &[3, 4, 1], &[3, 4, 1], 0).unwrap_err();
        assert_eq!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 3
            }
        );
    }

    #[test]
    fn scatter_out_shape_index_src_mismatch() {
        let err = scatter_out_shape(&[3, 4], &[3, 2], &[3, 3], 1).unwrap_err();
        match err {
            ShapeError::ShapeMismatch { lhs, rhs } => {
                assert_eq!(lhs, vec![3, 2]);
                assert_eq!(rhs, vec![3, 3]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn scatter_out_shape_other_axis_exceeds_input() {
        let err = scatter_out_shape(&[3, 4], &[5, 2], &[5, 2], 1).unwrap_err();
        match err {
            ShapeError::ShapeMismatch { lhs, rhs } => {
                assert_eq!(lhs, vec![3, 4]);
                assert_eq!(rhs, vec![5, 2]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn scatter_out_shape_other_axis_smaller_than_input_is_ok() {
        // dim 以外の軸で index が input より小さいのは許容（PyTorch の
        // `index.size(d) <= input.size(d)` 契約と整合）。
        let out = scatter_out_shape(&[3, 4], &[2, 2], &[2, 2], 1).unwrap();
        assert_eq!(out, vec![3, 4]);
    }

    // --- pad_out_shape ---

    #[test]
    fn pad_out_shape_1d_basic() {
        let out = pad_out_shape(&[3], &[(1, 2)]).unwrap();
        assert_eq!(out, vec![6]);
    }

    #[test]
    fn pad_out_shape_2d_both_axes() {
        let out = pad_out_shape(&[3, 4], &[(1, 1), (2, 0)]).unwrap();
        assert_eq!(out, vec![5, 6]);
    }

    #[test]
    fn pad_out_shape_rank_mismatch() {
        let err = pad_out_shape(&[3, 4], &[(1, 1)]).unwrap_err();
        match err {
            ShapeError::RankMismatch { expected, actual } => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn pad_out_shape_before_overflow() {
        let err = pad_out_shape(&[usize::MAX], &[(1, 0)]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
    }

    #[test]
    fn pad_out_shape_after_overflow() {
        // before 側は加算できるが after 側の加算でオーバーフローする
        // ケースを個別に検証する（2 回の checked_add の両方を通す）。
        let err = pad_out_shape(&[usize::MAX - 1], &[(1, 1)]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
    }

    #[test]
    fn pad_out_shape_rank_zero_identity() {
        let out = pad_out_shape(&[], &[]).unwrap();
        assert_eq!(out, Vec::<usize>::new());
    }

    #[test]
    fn pad_out_shape_rejects_byte_size_overflow_without_numel_overflow() {
        // `shape=[1]`・`pads=[(0, usize::MAX-1)]` は `usize` の要素数積
        // （`checked_numel`）としてはオーバーフローしない
        // （`1 + (usize::MAX-1) = usize::MAX`）が、`f32`（4 バイト）で
        // 確保すると `numel * 4` が `isize::MAX` バイトを大幅に超える。
        // `checked_numel_for::<f32>` によるバイトサイズ検査がなければ
        // ここを通過し、後続の `Vec::with_capacity` が capacity
        // overflow で panic する（PR #1831 codex-review P1 是正の
        // 回帰テスト）。
        let err = pad_out_shape(&[1], &[(0, usize::MAX - 1)]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
    }

    #[test]
    fn pad_out_shape_all_zero_pads_is_identity() {
        let out = pad_out_shape(&[3, 4], &[(0, 0), (0, 0)]).unwrap();
        assert_eq!(out, vec![3, 4]);
    }

    // --- sort_out_shape ---

    #[test]
    fn sort_out_shape_basic() {
        let out = sort_out_shape(&[3, 4], 1).unwrap();
        assert_eq!(out, vec![3, 4]);
    }

    #[test]
    fn sort_out_shape_axis_out_of_range() {
        let err = sort_out_shape(&[3, 4], 2).unwrap_err();
        assert_eq!(err, ShapeError::AxisOutOfRange { axis: 2, rank: 2 });
    }

    #[test]
    fn sort_out_shape_empty_dim() {
        let out = sort_out_shape(&[3, 0], 1).unwrap();
        assert_eq!(out, vec![3, 0]);
    }

    // --- topk_out_shape ---

    #[test]
    fn topk_out_shape_basic() {
        let out = topk_out_shape(&[3, 4], 1, 2).unwrap();
        assert_eq!(out, vec![3, 2]);
    }

    #[test]
    fn topk_out_shape_axis_out_of_range() {
        let err = topk_out_shape(&[3, 4], 2, 2).unwrap_err();
        assert_eq!(err, ShapeError::AxisOutOfRange { axis: 2, rank: 2 });
    }

    #[test]
    fn topk_out_shape_k_exceeds_dim_size() {
        let err = topk_out_shape(&[3, 4], 1, 5).unwrap_err();
        assert_eq!(
            err,
            ShapeError::NarrowOutOfBounds {
                dim: 1,
                start: 0,
                len: 5,
                dim_size: 4,
            }
        );
    }

    #[test]
    fn topk_out_shape_k_equals_dim_size() {
        let out = topk_out_shape(&[3, 4], 1, 4).unwrap();
        assert_eq!(out, vec![3, 4]);
    }

    #[test]
    fn topk_out_shape_k_zero() {
        let out = topk_out_shape(&[3, 4], 1, 0).unwrap();
        assert_eq!(out, vec![3, 0]);
    }

    // --- interpolate_out_shape ---

    #[test]
    fn interpolate_out_shape_1d_all_axes_spatial() {
        let out = interpolate_out_shape(&[3], &[6]).unwrap();
        assert_eq!(out, vec![6]);
    }

    #[test]
    fn interpolate_out_shape_2d_trailing_axis_only() {
        // batch 軸（先頭）は素通し、末尾 1 軸のみ空間軸。
        let out = interpolate_out_shape(&[2, 3], &[5]).unwrap();
        assert_eq!(out, vec![2, 5]);
    }

    #[test]
    fn interpolate_out_shape_3d_two_spatial_axes() {
        let out = interpolate_out_shape(&[2, 4, 4], &[8, 8]).unwrap();
        assert_eq!(out, vec![2, 8, 8]);
    }

    #[test]
    fn interpolate_out_shape_downsample() {
        let out = interpolate_out_shape(&[1, 8], &[3]).unwrap();
        assert_eq!(out, vec![1, 3]);
    }

    #[test]
    fn interpolate_out_shape_identity_size() {
        let out = interpolate_out_shape(&[2, 3], &[2, 3]).unwrap();
        assert_eq!(out, vec![2, 3]);
    }

    #[test]
    fn interpolate_out_shape_rejects_empty_size() {
        let err = interpolate_out_shape(&[2, 3], &[]).unwrap_err();
        assert_eq!(
            err,
            ShapeError::RankMismatch {
                expected: 1,
                actual: 0
            }
        );
    }

    #[test]
    fn interpolate_out_shape_rejects_size_longer_than_rank() {
        let err = interpolate_out_shape(&[3], &[3, 3]).unwrap_err();
        assert_eq!(
            err,
            ShapeError::RankMismatch {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn interpolate_out_shape_rejects_zero_output_spatial_axis() {
        let err = interpolate_out_shape(&[2, 3], &[0]).unwrap_err();
        assert_eq!(
            err,
            ShapeError::ShapeMismatch {
                lhs: vec![2, 3],
                rhs: vec![0],
            }
        );
    }

    #[test]
    fn interpolate_out_shape_rejects_zero_input_spatial_axis() {
        let err = interpolate_out_shape(&[2, 0], &[4]).unwrap_err();
        assert_eq!(
            err,
            ShapeError::ShapeMismatch {
                lhs: vec![2, 0],
                rhs: vec![4],
            }
        );
    }

    #[test]
    fn interpolate_out_shape_allows_empty_leading_axis() {
        // 空間軸ではない先頭軸（batch）が 0 の空入力は許容する。
        let out = interpolate_out_shape(&[0, 3], &[6]).unwrap();
        assert_eq!(out, vec![0, 6]);
    }

    #[test]
    fn interpolate_out_shape_element_count_overflow() {
        // `out = [2, usize::MAX]` の要素数積が overflow する。
        let err = interpolate_out_shape(&[2, usize::MAX], &[usize::MAX]).unwrap_err();
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }

    #[test]
    fn interpolate_out_shape_rejects_byte_size_overflow_without_numel_overflow() {
        // 入力 shape=[1] を size=[usize::MAX] へ interpolate する場合、
        // 出力の要素数積 `1 * usize::MAX = usize::MAX` は `usize` の乗算
        // オーバーフローとしては検出されない（`checked_numel` 単体では
        // 通過する）が、`f32`（4 バイト）換算のバイトサイズは必ず
        // `isize::MAX` を超えるため `checked_numel_for::<f32>` が
        // `ElementCountOverflow` を返す（イシュー #1834 codex-review
        // P1 是正: `Vec::with_capacity` の capacity overflow panic を
        // 防ぐ）。
        let err = interpolate_out_shape(&[1], &[usize::MAX]).unwrap_err();
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }

    #[test]
    fn interpolate_out_shape_rejects_empty_leading_axis_with_huge_multi_axis_size() {
        // Cursor Bugbot 指摘（イシュー #1834・PR #1834 レビュー）の
        // 回帰テスト。先頭の残り軸（batch）が `0` の空入力に対して
        // 空間軸（`size`）側を巨大な多軸にした場合、出力 shape
        // `[0, usize::MAX, 2]` **全体**の要素数積は先頭の `0` に短絡し
        // `0` になる（`checked_numel_for::<f32>(&out)` 単体は通過する）。
        // しかし `0` 次元を除いた部分積（`usize::MAX * 2`）は overflow
        // するため、新設した `nonzero_dims` 検査が
        // `ElementCountOverflow` を返すことを確認する。
        let err = interpolate_out_shape(&[0, 3, 4], &[usize::MAX, 2]).unwrap_err();
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }

    #[test]
    fn interpolate_out_shape_rejects_middle_zero_axis_with_huge_multi_axis_size() {
        // 上記と同型だが、`0` 次元が先頭（axis 0）ではなく非空間軸の
        // 中間（axis 1）にあるケース。空間軸自体（axis 2・3）はいずれも
        // 非ゼロのため既存のゼロ検査（`shape[axis] == 0` の判定）を
        // 素通りし、出力 shape `[2, 0, usize::MAX, 2]` 全体の積も
        // 先頭でない `0` を経由して `0` に短絡する。`nonzero_dims`
        // 検査（`[2, usize::MAX, 2]` の部分積）が overflow を検出する
        // ことを確認する。
        let err = interpolate_out_shape(&[2, 0, 3, 4], &[usize::MAX, 2]).unwrap_err();
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }

    #[test]
    fn interpolate_out_shape_allows_empty_leading_axis_with_safe_multi_axis_size() {
        // 上記 2 件との対比: 先頭軸が `0` でも空間軸側が小さく
        // `nonzero_dims` の部分積が overflow しない場合は従来どおり
        // 成功する（`interpolate_out_shape_allows_empty_leading_axis`
        // の 2 空間軸版）。
        let out = interpolate_out_shape(&[0, 3, 4], &[8, 8]).unwrap();
        assert_eq!(out, vec![0, 8, 8]);
    }

    // --- one_hot_out_shape ---

    #[test]
    fn one_hot_out_shape_basic() {
        let out = one_hot_out_shape(&[2, 2], 3).unwrap();
        assert_eq!(out, vec![2, 2, 3]);
    }

    #[test]
    fn one_hot_out_shape_1d_index() {
        let out = one_hot_out_shape(&[4], 5).unwrap();
        assert_eq!(out, vec![4, 5]);
    }

    #[test]
    fn one_hot_out_shape_num_classes_zero() {
        let err = one_hot_out_shape(&[2, 2], 0).unwrap_err();
        assert_eq!(
            err,
            ShapeError::IndexOutOfRange {
                dim: 2,
                index: 0,
                dim_size: 0,
            }
        );
    }

    #[test]
    fn one_hot_out_shape_element_count_overflow() {
        let err = one_hot_out_shape(&[usize::MAX], 2).unwrap_err();
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }

    // codex-review P1 是正（イシュー #1755）の回帰: `index_shape = [1]`・
    // `num_classes = usize::MAX` は要素数積（`1 * usize::MAX`）としては
    // `usize` オーバーフローしない（`checked_numel` 単体では通過する）が、
    // `f32` 4 バイト換算の確保バイト数は `isize::MAX` を大幅に超える。
    // `checked_numel_for::<f32>` による検査でここが拒否されることを
    // 確認し、下流（`Var::one_hot`・`eval::one_hot`・CPU
    // `gather_scatter::one_hot` 等）の `Vec::with_capacity`／
    // `vec![0f32; numel]` が capacity overflow で panic する経路を
    // 塞げていることを担保する。
    #[test]
    fn one_hot_out_shape_byte_size_overflow_without_element_count_overflow() {
        let err = one_hot_out_shape(&[1], usize::MAX).unwrap_err();
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }

    // --- conv_out_len / conv2d_out_shape / im2col_out_shape ---

    fn conv_params(
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Conv2dParams {
        Conv2dParams::new(kernel_size, stride, padding, dilation, groups).unwrap()
    }

    #[test]
    fn conv_out_len_basic_no_pad_no_dilation() {
        // PyTorch `_conv_output_size` 参照値: in=4, k=3, s=1, p=0, d=1 -> 2
        assert_eq!(conv_out_len(4, 3, 1, 0, 1).unwrap(), 2);
    }

    #[test]
    fn conv_out_len_with_padding_preserves_size() {
        // 'same' 相当（k=3, p=1, s=1, d=1）
        assert_eq!(conv_out_len(5, 3, 1, 1, 1).unwrap(), 5);
    }

    #[test]
    fn conv_out_len_with_stride() {
        assert_eq!(conv_out_len(7, 3, 2, 0, 1).unwrap(), 3);
    }

    #[test]
    fn conv_out_len_with_dilation() {
        // 実効カーネル幅 = d*(k-1)+1 = 2*2+1 = 5
        assert_eq!(conv_out_len(5, 3, 1, 0, 2).unwrap(), 1);
    }

    #[test]
    fn conv_out_len_negative_numerator_rejected() {
        // in=1, k=3, p=0, d=1 -> 分子 1 - 2 - 1 = -2 < 0
        let err = conv_out_len(1, 3, 1, 0, 1).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn conv_out_len_zero_input_with_padding_only_window() {
        // H=W=1, kernel=1, padding=1 は「有効入力を含まない窓」を許容
        // する（設計 doc §3「訂正」）。in=1 なので H=0 拒否ゲートには
        // 掛からない。
        assert_eq!(conv_out_len(1, 1, 1, 1, 1).unwrap(), 3);
    }

    #[test]
    fn conv_out_len_rejects_zero_stride() {
        let err = conv_out_len(4, 3, 0, 0, 1).unwrap_err();
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }

    #[test]
    fn conv_out_len_rejects_zero_kernel() {
        let err = conv_out_len(4, 0, 1, 0, 1).unwrap_err();
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }

    #[test]
    fn conv2d_out_shape_basic() {
        let params = conv_params([3, 3], [1, 1], [0, 0], [1, 1], 1);
        let out = conv2d_out_shape(&[2, 3, 8, 8], &[4, 3, 3, 3], &params).unwrap();
        assert_eq!(out, vec![2, 4, 6, 6]);
    }

    #[test]
    fn conv2d_out_shape_groups() {
        let params = conv_params([3, 3], [1, 1], [1, 1], [1, 1], 2);
        // groups=2: Cin=4 -> Cin_g=2, Cout=6 -> Cout_g=3
        let out = conv2d_out_shape(&[1, 4, 5, 5], &[6, 2, 3, 3], &params).unwrap();
        assert_eq!(out, vec![1, 6, 5, 5]);
    }

    #[test]
    fn conv2d_out_shape_depthwise_groups_eq_cin() {
        let params = conv_params([3, 3], [1, 1], [1, 1], [1, 1], 4);
        let out = conv2d_out_shape(&[1, 4, 5, 5], &[4, 1, 3, 3], &params).unwrap();
        assert_eq!(out, vec![1, 4, 5, 5]);
    }

    #[test]
    fn conv2d_out_shape_n_zero_is_accepted() {
        let params = conv_params([3, 3], [1, 1], [0, 0], [1, 1], 1);
        let out = conv2d_out_shape(&[0, 3, 8, 8], &[4, 3, 3, 3], &params).unwrap();
        assert_eq!(out, vec![0, 4, 6, 6]);
    }

    #[test]
    fn conv2d_out_shape_h_zero_is_rejected() {
        let params = conv_params([3, 3], [1, 1], [0, 0], [1, 1], 1);
        let err = conv2d_out_shape(&[1, 3, 0, 8], &[4, 3, 3, 3], &params).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn conv2d_out_shape_padding_greater_than_half_kernel_is_accepted() {
        // pooling と異なり Conv は padding 上限を持たない（設計 doc
        // §0.2／§3 の意図的な差異の回帰）。
        let params = conv_params([3, 3], [1, 1], [5, 5], [1, 1], 1);
        let out = conv2d_out_shape(&[1, 1, 4, 4], &[1, 1, 3, 3], &params).unwrap();
        assert_eq!(out, vec![1, 1, 12, 12]);
    }

    #[test]
    fn conv2d_out_shape_rank_mismatch_input() {
        let params = conv_params([3, 3], [1, 1], [0, 0], [1, 1], 1);
        let err = conv2d_out_shape(&[3, 8, 8], &[4, 3, 3, 3], &params).unwrap_err();
        assert_eq!(
            err,
            ShapeError::RankMismatch {
                expected: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn conv2d_out_shape_cin_not_divisible_by_groups() {
        let params = conv_params([3, 3], [1, 1], [0, 0], [1, 1], 2);
        let err = conv2d_out_shape(&[1, 3, 8, 8], &[4, 2, 3, 3], &params).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn conv2d_out_shape_weight_cin_mismatch() {
        let params = conv_params([3, 3], [1, 1], [0, 0], [1, 1], 1);
        let err = conv2d_out_shape(&[1, 3, 8, 8], &[4, 2, 3, 3], &params).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn conv2d_out_shape_cout_not_divisible_by_groups() {
        let params = conv_params([3, 3], [1, 1], [0, 0], [1, 1], 2);
        let err = conv2d_out_shape(&[1, 4, 8, 8], &[5, 2, 3, 3], &params).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn im2col_out_shape_basic() {
        let params = conv_params([3, 3], [1, 1], [0, 0], [1, 1], 1);
        let out = im2col_out_shape(&[2, 3, 8, 8], &params).unwrap();
        // K_g = 3*3*3 = 27, P = 6*6 = 36
        assert_eq!(out, vec![2, 1, 27, 36]);
    }

    #[test]
    fn im2col_out_shape_groups() {
        let params = conv_params([3, 3], [1, 1], [1, 1], [1, 1], 2);
        let out = im2col_out_shape(&[1, 4, 5, 5], &params).unwrap();
        // Cin_g=2, K_g=2*3*3=18, P=5*5=25
        assert_eq!(out, vec![1, 2, 18, 25]);
    }

    #[test]
    fn im2col_out_shape_rejects_rank_mismatch() {
        let params = conv_params([3, 3], [1, 1], [0, 0], [1, 1], 1);
        let err = im2col_out_shape(&[3, 8, 8], &params).unwrap_err();
        assert_eq!(
            err,
            ShapeError::RankMismatch {
                expected: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn im2col_out_shape_w_zero_is_rejected() {
        let params = conv_params([3, 3], [1, 1], [0, 0], [1, 1], 1);
        let err = im2col_out_shape(&[1, 3, 8, 0], &params).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    // --- pool_out_len / pool2d_out_shape / adaptive_pool2d_out_shape /
    // adaptive_window（イシュー #1728）---

    fn pool_params(
        kernel_size: [usize; 2],
        stride: Option<[usize; 2]>,
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Pool2dParams {
        Pool2dParams::new(kernel_size, stride, padding, dilation).unwrap()
    }

    #[test]
    fn pool_out_len_padding_boundary_k2_p1_allowed() {
        // k=2, d=1, p=1 は許可される padding 上限境界（floor(2/2)=1）。
        assert_eq!(pool_out_len(4, 2, 2, 1, 1).unwrap(), 3);
    }

    #[test]
    fn pool_out_len_padding_boundary_k3_p1_allowed() {
        // k=3, d=2, p=1 は許可される padding 上限境界（floor(3/2)=1）。
        // 分子 = 6 + 2*1 - 2*(3-1) - 1 = 3 -> 3/1 + 1 = 4。
        assert_eq!(pool_out_len(6, 3, 1, 1, 2).unwrap(), 4);
    }

    #[test]
    fn pool_out_len_negative_numerator_rejected() {
        // in=1, k=2, s=2, p=0, d=1 -> 分子 -1（負分子拒否ゲート）。
        let err = pool_out_len(1, 2, 2, 0, 1).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn pool_out_len_rejects_zero_stride_or_kernel() {
        assert!(pool_out_len(4, 2, 0, 0, 1).is_err());
        assert!(pool_out_len(4, 0, 1, 0, 1).is_err());
    }

    #[test]
    fn pool2d_out_shape_basic() {
        let params = pool_params([2, 2], None, [0, 0], [1, 1]);
        let out = pool2d_out_shape(&[2, 3, 4, 4], &params).unwrap();
        assert_eq!(out, vec![2, 3, 2, 2]);
    }

    #[test]
    fn pool2d_out_shape_h_zero_is_rejected() {
        let params = pool_params([2, 2], None, [0, 0], [1, 1]);
        let err = pool2d_out_shape(&[1, 3, 0, 4], &params).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn pool2d_out_shape_n_zero_is_accepted() {
        let params = pool_params([2, 2], None, [0, 0], [1, 1]);
        let out = pool2d_out_shape(&[0, 3, 4, 4], &params).unwrap();
        assert_eq!(out, vec![0, 3, 2, 2]);
    }

    #[test]
    fn pool2d_out_shape_c_zero_is_accepted() {
        let params = pool_params([2, 2], None, [0, 0], [1, 1]);
        let out = pool2d_out_shape(&[1, 0, 4, 4], &params).unwrap();
        assert_eq!(out, vec![1, 0, 2, 2]);
    }

    #[test]
    fn pool2d_out_shape_padding_only_window_is_rejected() {
        // in=0, k=2, s=2, p=1, d=1: 負分子ゲートは素通りするが H==0 の
        // 事前検査で拒否される（設計 doc §3「非 adaptive 側は padding>0
        // のとき負分子ゲートだけでは不十分」）。
        let params = pool_params([2, 2], None, [1, 1], [1, 1]);
        let err = pool2d_out_shape(&[1, 1, 0, 4], &params).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn pool2d_out_shape_empty_window_dilation_rejected() {
        // in=1, kernel=2, stride=1, padding=1, dilation=2 は
        // padding 上限を満たすが両タップとも範囲外（空窓）。
        let params = pool_params([2, 2], Some([1, 1]), [1, 1], [2, 2]);
        let err = pool2d_out_shape(&[1, 1, 1, 8], &params).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn pool2d_out_shape_kernel3_dilation_not_empty_window() {
        // kernel>=3 は設計 doc §3 の導出により空窓が起こり得ない。
        let params = pool_params([3, 1], Some([1, 1]), [1, 0], [2, 1]);
        let out = pool2d_out_shape(&[1, 1, 3, 4], &params).unwrap();
        assert_eq!(out[2], 1);
    }

    #[test]
    fn adaptive_pool2d_out_shape_basic() {
        let out = adaptive_pool2d_out_shape(&[2, 3, 8, 8], [2, 2]).unwrap();
        assert_eq!(out, vec![2, 3, 2, 2]);
    }

    #[test]
    fn adaptive_pool2d_out_shape_output_larger_than_input_accepted() {
        let out = adaptive_pool2d_out_shape(&[1, 1, 2, 2], [4, 4]).unwrap();
        assert_eq!(out, vec![1, 1, 4, 4]);
    }

    #[test]
    fn adaptive_pool2d_out_shape_h_zero_is_rejected() {
        let err = adaptive_pool2d_out_shape(&[1, 1, 0, 4], [2, 2]).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn adaptive_pool2d_out_shape_output_size_zero_is_rejected() {
        let err = adaptive_pool2d_out_shape(&[1, 1, 4, 4], [0, 2]).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn adaptive_pool2d_out_shape_rank_mismatch() {
        let err = adaptive_pool2d_out_shape(&[1, 4, 4], [2, 2]).unwrap_err();
        assert!(matches!(err, ShapeError::RankMismatch { .. }));
    }

    #[test]
    fn adaptive_window_basic() {
        // in=8, out=2: 窓 [0,4)/[4,8)。
        assert_eq!(adaptive_window(0, 8, 2), Some((0, 4)));
        assert_eq!(adaptive_window(1, 8, 2), Some((4, 8)));
    }

    #[test]
    fn adaptive_window_non_divisible() {
        // in=7, out=2: start=floor(0*7/2)=0, end=ceil(7/2)=4 /
        // start=floor(7/2)=3, end=ceil(14/2)=7。
        assert_eq!(adaptive_window(0, 7, 2), Some((0, 4)));
        assert_eq!(adaptive_window(1, 7, 2), Some((3, 7)));
    }

    #[test]
    fn adaptive_window_zero_out_len_is_none() {
        assert_eq!(adaptive_window(0, 8, 0), None);
    }
}
