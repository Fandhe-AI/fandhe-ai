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
}
