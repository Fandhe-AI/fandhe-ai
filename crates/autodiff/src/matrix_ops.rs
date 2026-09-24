//! `tril`・`triu`・`diag`・`trace`・`outer`・`dot` の 6 種形状・行列演算
//! （イシュー #2144・親 #2131「5-B 演算」）。
//!
//! **新規 `Op` はゼロ（受け入れ条件）**: いずれも既存の `Var::masked_fill`
//! （`Op::MaskedFill`）・`Var::gather`（`Op::Gather`）・`Var::narrow`
//! （`Op::Narrow`）・`Var::broadcast_to`（`Op::BroadcastTo`）・
//! `Var::transpose`（`Op::Transpose`）・`Var::pad`（`Op::Pad`）・
//! `Var::squeeze`（`Var::reshape` へ委譲）・`Var::mul`（`Op::Mul`）・
//! `Var::sum`（`Op::Sum`）の合成のみで構成する。いずれも CPU・CUDA・
//! Metal の全バックエンドに経路があり（`gather`／`pad` は既定
//! `Unsupported` でホスト参照実装へフォールバック）、専用カーネルなしで
//! 到達可能。`crates/backend-*`・`crates/tensor-core` は変更しない
//! （`crates/autodiff/src/rearrange_ops.rs` の境界検査ヘルパー 2 点
//! （[`crate::rearrange_ops::checked_axis_len_as_i32`]・
//! [`crate::rearrange_ops::checked_index_alloc_len`]）のみ `pub(crate)`
//! へ昇格して共有する）。
//!
//! **facade 非公開（意図的）**: `crates/autodiff/src/rearrange_ops.rs`
//! モジュール doc と同じ理由・同じ判断枠組みによる。`Var` は facade
//! （`fandhe_ai` クレート）から直接再エクスポートされるため、`Var` への
//! inherent メソッド追加は即座に facade 公開面へ出てしまう。イシュー
//! #2144 本文は facade 公開面を承認事項として明示し、親 #2131 はこの
//! ツリーに限り「設計判断記録 → 承認 → 実装」の 2 段階を定めるため、
//! 承認が取れるまでは自由関数として `Var` の外に置き到達不能にする
//! （`docs/autodiff-matrix-ops-decision.md` §2.1）。承認後は
//! `Var::tril` 等の薄い委譲メソッドを追加し、facade 側の保留ガード
//! （`crates/facade/src/lib.rs::VarMatrixOpsHoldDoctestGuard`）を撤去
//! する。
//!
//! **数値契約**（§ ごとの詳細は `docs/autodiff-matrix-ops-decision.md`
//! §3 の表を参照）:
//! - `tril`／`triu`／`diag`（両方向）: forward はコピーまたは定数 0 の
//!   埋め込みのみ（算術を含まない）のため 3 バックエンド間で構造的に
//!   bit 完全一致する（`masked_fill`／`gather`／`narrow`／`pad` はいずれ
//!   もコピー系演算。`NaN` の payload も保存される）。backward は
//!   `Op::MaskedFill`／`Op::Gather` の VJP（fill 位置はゼロ、それ以外は
//!   素通し・scatter で寄与は各 1 つ）のため同じく bit 一致する。
//! - `trace`: `diag(x, 0)`（bit 一致）→ `sum(None)`。`sum` の縮約順序は
//!   バックエンドで異なりうるため REQ-2 の統一複合判定（相対誤差 1e-3
//!   未満 または 絶対誤差 1e-5 未満）で比較する。
//! - `outer`: 乗算 1 回のみ（`mul`）のため forward は bit 一致する。
//!   backward は `broadcast_to` の VJP（軸方向の `reduce_to_shape`
//!   縮約）を経由するため REQ-2 の統一複合判定で比較する。
//! - `dot`: `mul` → `sum(None)`。`mul` 自体は bit 一致するが、続く
//!   `sum` の縮約順序差により forward 全体としては REQ-2 の統一複合
//!   判定で比較する。backward（`g * other`。乗算 1 回のみ）は bit 一致
//!   する。
//!
//! **PyTorch との差分（doc に明記。承認後の facade 版でも据え置く
//! 予定）**:
//! - `trace`・`diag` の 2-D 入力は rank 2 限定（`torch.diagonal` の
//!   rank 3 以上・バッチ trace は非対応）。
//! - 軸番号を持つ引数はない（`tril`／`triu`／`diag` の `diagonal` の
//!   みが可変パラメータ）。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: 外部から渡る
//! `diagonal: isize` は `unsigned_abs()`（`isize::MIN` 対策）・
//! `checked_add`・`saturating_sub` のみで扱い、生の `usize ± isize` 演算
//! は行わない。マスク・添字の確保前サイズ検査は
//! [`crate::rearrange_ops::checked_axis_len_as_i32`]・
//! [`crate::rearrange_ops::checked_index_alloc_len`] を経由する
//! （`rearrange_ops` と同じ 1 GiB 実用上限）。本番経路で `unwrap()`／
//! `expect()` は使わない。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::rearrange_ops::{checked_axis_len_as_i32, checked_index_alloc_len};
use crate::var::Var;

/// `tril`／`triu` 共用のマスク構築（末尾 2 軸 `[m, n]`）。`fill_where`
/// が真を返す位置（`(i, j)` の 0-based 行・列添字）を `true`（＝
/// [`Var::masked_fill`] で 0 に落とす対象）とする `Tensor<bool>` を
/// ホスト側で組み立てる。`checked_index_alloc_len` で `m * n` 要素の
/// 確保前サイズを検査してから `Vec<bool>` を確保する（`broadcast_to`
/// 由来の stride-0 view で `m`／`n` が実体を伴わず巨大になりうるため。
/// `rearrange_ops::flip_on_broadcast_view_...` と同じ脅威モデル）。
fn build_tril_triu_mask(
    m: usize,
    n: usize,
    fill_where: impl Fn(i64, i64) -> bool,
) -> Result<Tensor<bool>, AutodiffError> {
    let total = m
        .checked_mul(n)
        .ok_or(ShapeError::ElementCountOverflow)
        .map_err(AutodiffError::Shape)?;
    checked_index_alloc_len(total)?;
    let mut data = Vec::with_capacity(total);
    for i in 0..m {
        let i64_i = i as i64;
        for j in 0..n {
            data.push(fill_where(i64_i, j as i64));
        }
    }
    Tensor::new(data, &[m, n]).map_err(AutodiffError::Shape)
}

/// `diagonal: isize` を `i64` へ変換する（`checked_index_alloc_len` と
/// 同様に本番経路 panic を避けるための明示変換。`isize` は 64bit
/// ターゲットで `i64` と同じ範囲のため `as` で失われる情報はない——
/// `TryFrom` を使わない理由は `isize` から `i64` への変換が全ターゲット
/// で無損失〈`isize` は `i64` 以下の幅〉であり、`unwrap()` を要する
/// `TryFrom` 経路より単純なため）。
fn diagonal_as_i64(diagonal: isize) -> i64 {
    diagonal as i64
}

/// 末尾 2 軸について下三角を残し、それ以外を 0 にする（`torch.tril`
/// 相当。イシュー #2144）。先頭のバッチ軸は保つ。`diagonal` は主対角
/// からのオフセット（PyTorch と同じ規約: 正で右上へ、負で左下へ）。
///
/// rank は 2 以上を要求する（`RankMismatch { expected: 2, .. }`。
/// `fandhe_ai_tensor_core::batched_matmul_plan` と同じ「`expected: 2`
/// は『2 以上』を表す」慣習）。
///
/// # 早期リターン（ノードを積まない）
///
/// `diagonal >= n - 1`（`n` は列数）のときは全要素が下三角に含まれる
/// ため `x` をそのまま返す（`rearrange_ops::flip` の `dims.is_empty()`
/// と同型）。
pub fn tril<'t>(x: &Var<'t>, diagonal: isize) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let rank = shape.len();
    if rank < 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: rank,
        }));
    }
    let m = shape[rank - 2];
    let n = shape[rank - 1];
    let k = diagonal_as_i64(diagonal);
    // `j - i > k` が常に偽 ⟺ 最大値 `(n-1) - 0` が `k` 以下。
    if n == 0 || (n as i64 - 1) <= k {
        return Ok(*x);
    }
    let mask = build_tril_triu_mask(m, n, |i, j| j - i > k)?;
    x.masked_fill(&mask, 0.0)
}

/// 末尾 2 軸について上三角を残し、それ以外を 0 にする（`torch.triu`
/// 相当。イシュー #2144）。[`tril`] と対称。
///
/// # 早期リターン（ノードを積まない）
///
/// `diagonal <= -(m - 1)`（`m` は行数）のときは全要素が上三角に含まれ
/// るため `x` をそのまま返す。
pub fn triu<'t>(x: &Var<'t>, diagonal: isize) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let rank = shape.len();
    if rank < 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: rank,
        }));
    }
    let m = shape[rank - 2];
    let n = shape[rank - 1];
    let k = diagonal_as_i64(diagonal);
    // `j - i < k` が常に偽 ⟺ 最小値 `0 - (m-1)` が `k` 以上。
    if m == 0 || -((m as i64) - 1) >= k {
        return Ok(*x);
    }
    let mask = build_tril_triu_mask(m, n, |i, j| j - i < k)?;
    x.masked_fill(&mask, 0.0)
}

/// 1-D 入力なら対角行列（2-D）を作り、2-D 入力なら対角成分（1-D）を
/// 取り出す（`torch.diag` 相当。rank で分岐。イシュー #2144）。
///
/// rank は 1 か 2 に限る（それ以外は `AutodiffError::InvalidArgument`。
/// `RankMismatch` は単一の `expected` しか表現できないため、2 値のいず
/// れかを許容する本チェックには使わない）。
pub fn diag<'t>(x: &Var<'t>, diagonal: isize) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    match shape.len() {
        1 => diag_1d_to_2d(x, &shape, diagonal),
        2 => diag_2d_to_1d(x, &shape, diagonal),
        rank => Err(AutodiffError::InvalidArgument(format!(
            "matrix_ops::diag: rank は 1 か 2 のみ対応（torch.diag と同じ\
             規約）、got rank={rank}"
        ))),
    }
}

/// [`diag`] の 1-D → 2-D 経路。`n = shape[0]` から `N = n + |k|` を
/// `checked_add` で確定し、`x.broadcast_to([n, n])`（`out[i][j] = x[j]`。
/// 行方向へ複製する stride-0 view）→ `masked_fill(i != j, 0)`（対角
/// 以外を 0 化）→ `k != 0` のときのみ `pad` で `N x N` へ埋め込む
/// （設計判断: §2.3 の合成方式表を参照。行を `(0,k)`／列を `(k,0)`
/// で埋めると `new[i][i+k] = old[i][i]`、行を `(|k|,0)`／列を
/// `(0,|k|)` で埋めると `new[i+|k|][i] = old[i][i]` になり、
/// `k` の正負それぞれの目的位置と一致する）。
fn diag_1d_to_2d<'t>(
    x: &Var<'t>,
    shape: &[usize],
    diagonal: isize,
) -> Result<Var<'t>, AutodiffError> {
    let n = shape[0];
    let k_abs = diagonal.unsigned_abs();
    checked_axis_len_as_i32(n)?;
    let square = x.broadcast_to(&[n, n])?;
    let mask = build_tril_triu_mask(n, n, |i, j| i != j)?;
    let diag_square = square.masked_fill(&mask, 0.0)?;
    if k_abs == 0 {
        return Ok(diag_square);
    }
    // `N = n + k_abs` は下記 `pad` 内で `pad_out_shape` が checked_add で
    // 確定させるため、ここでの独自オーバーフロー検査は不要（`Var::pad`
    // の契約に委ねる）。
    if diagonal > 0 {
        diag_square.pad(&[(0, k_abs), (k_abs, 0)], 0.0)
    } else {
        diag_square.pad(&[(k_abs, 0), (0, k_abs)], 0.0)
    }
}

/// [`diag`] の 2-D → 1-D 経路。抽出長 `L`（`k >= 0` なら
/// `min(m, n.saturating_sub(k))`・`k < 0` なら
/// `min(m.saturating_sub(|k|), n)`）を先に確定し、`x.narrow(0, r0, L)`
/// （開始行 `r0`: `k >= 0` なら `0`・`k < 0` なら `|k|`）→
/// `gather(1, idx[L,1])`（`idx[i] = i + c0`。`c0`: `k >= 0` なら `k`・
/// `k < 0` なら `0`）→ `squeeze(Some(1))`。`L == 0` のときも
/// `narrow(0, 0, 0)` で開始位置を 0 に固定するため範囲外オフセットでも
/// `NarrowOutOfBounds` にならない（設計判断 §2.3・§2.4）。
fn diag_2d_to_1d<'t>(
    x: &Var<'t>,
    shape: &[usize],
    diagonal: isize,
) -> Result<Var<'t>, AutodiffError> {
    let m = shape[0];
    let n = shape[1];
    let k_abs = diagonal.unsigned_abs();
    let (r0, c0, l) = if diagonal >= 0 {
        let l = m.min(n.saturating_sub(k_abs));
        (0usize, k_abs, l)
    } else {
        let l = m.saturating_sub(k_abs).min(n);
        (k_abs, 0usize, l)
    };

    if l == 0 {
        let empty = x.narrow(0, 0, 0)?;
        let idx = Tensor::new(Vec::new(), &[0, 1]).map_err(AutodiffError::Shape)?;
        let gathered = empty.gather(1, &idx)?;
        return gathered.squeeze(Some(1));
    }

    checked_axis_len_as_i32(n)?;
    checked_index_alloc_len(l)?;
    let narrowed = x.narrow(0, r0, l)?;
    let idx_data: Vec<i32> = (0..l).map(|i| (i + c0) as i32).collect();
    let idx = Tensor::new(idx_data, &[l, 1]).map_err(AutodiffError::Shape)?;
    let gathered = narrowed.gather(1, &idx)?;
    gathered.squeeze(Some(1))
}

/// 2-D の主対角和（`torch.trace` 相当。イシュー #2144）。
/// [`diag`]`(x, 0)`（bit 一致）→ `sum(None)`（REQ-2 統一複合判定。
/// モジュール doc「数値契約」参照）。
///
/// rank は 2 限定（`RankMismatch { expected: 2, actual }`）。非正方
/// 行列・空（`[0, n]`／`[m, 0]`）も許容し、後者は空の対角 `[0]` の
/// `sum(None)` に委ねる（`sum` は単位元 0.0 を持つ縮約のため
/// エラーにはならない——`Var::min` のような「単位元なし」制約はない）。
pub fn trace<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let rank = x.shape().len();
    if rank != 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: rank,
        }));
    }
    diag(x, 0)?.sum(None)
}

/// 1-D 同士の外積 `[n, m]`（`torch.outer` 相当。イシュー #2144）。
/// `outer(a, b)[i, j] = a[i] * b[j]`。
///
/// `a.broadcast_to([1, n]).transpose(0, 1)`（`[n, 1]` の view）
/// `.mul(&b.broadcast_to([1, m]))`（`[n, 1]` と `[1, m]` の NumPy
/// ブロードキャストで `[n, m]` に展開。`Var::mul` が `check_same_tape`
/// を行うため本関数側での再検査は不要）。forward は乗算 1 回のみで
/// bit 一致するが、backward（`broadcast_to` の VJP）は REQ-2 統一複合
/// 判定で比較する（モジュール doc「数値契約」参照）。
///
/// `a`／`b`ともに rank 1 でなければ `RankMismatch { expected: 1, .. }`。
pub fn outer<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let a_rank = a.shape().len();
    if a_rank != 1 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 1,
            actual: a_rank,
        }));
    }
    let b_rank = b.shape().len();
    if b_rank != 1 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 1,
            actual: b_rank,
        }));
    }
    let n = a.shape()[0];
    let m = b.shape()[0];
    let a_col = a.broadcast_to(&[1, n])?.transpose(0, 1)?;
    let b_row = b.broadcast_to(&[1, m])?;
    a_col.mul(&b_row)
}

/// 同じ長さの 1-D 同士の内積（`torch.dot` 相当。イシュー #2144）。
/// `a.mul(b).sum(None)`。PyTorch と同じく broadcast はしない（長さ
/// 不一致は暗黙 broadcast に任せず明示的に拒否する）。
///
/// `a`／`b`ともに rank 1 でなければ `RankMismatch { expected: 1, .. }`。
/// 長さが一致しなければ `ShapeError::ShapeMismatch`。forward
/// （`mul` → `sum`）は REQ-2 統一複合判定、backward（`g * other`。
/// 乗算 1 回のみ）は bit 一致する（モジュール doc「数値契約」参照）。
pub fn dot<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let a_shape = a.shape();
    let a_rank = a_shape.len();
    if a_rank != 1 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 1,
            actual: a_rank,
        }));
    }
    let b_shape = b.shape();
    let b_rank = b_shape.len();
    if b_rank != 1 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 1,
            actual: b_rank,
        }));
    }
    if a_shape != b_shape {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: a_shape,
            rhs: b_shape,
        }));
    }
    a.mul(b)?.sum(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    // --- tril: k=0・正負・範囲外・非正方・rank 3（バッチ）---

    #[test]
    fn tril_k0_2x2() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = tril(&x, 0).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 0.0, 3.0, 4.0]
        );
    }

    #[test]
    fn tril_positive_diagonal() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = tril(&x, 1).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn tril_negative_diagonal() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = tril(&x, -1).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![0.0, 0.0, 3.0, 0.0]
        );
    }

    #[test]
    fn tril_large_positive_diagonal_is_noop_no_new_node() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let before = tape.len();
        let out = tril(&x, 100).unwrap();
        assert_eq!(tape.len(), before);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            x.to_tensor().host_slice().into_owned()
        );
    }

    #[test]
    fn tril_large_negative_diagonal_zeroes_everything() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = tril(&x, -100).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn tril_non_square() {
        let tape = Tape::new();
        // [[1,2,3],[4,5,6]]
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let out = tril(&x, 0).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0]
        );
    }

    #[test]
    fn tril_batched_rank3() {
        let tape = Tape::new();
        // 2 バッチの 2x2: [[1,2],[3,4]] と [[5,6],[7,8]]
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]));
        let out = tril(&x, 0).unwrap();
        assert_eq!(out.to_tensor().shape(), &[2, 2, 2]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 0.0, 3.0, 4.0, 5.0, 0.0, 7.0, 8.0]
        );
    }

    #[test]
    fn tril_rank1_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            tril(&x, 0),
            Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }))
        ));
    }

    #[test]
    fn tril_preserves_nan_payload_and_inf() {
        let tape = Tape::new();
        let nan = f32::from_bits(0x7fc0_1234);
        let x = tape.var(&t(vec![nan, f32::INFINITY, 1.0, 2.0], &[2, 2]));
        let out = tril(&x, 0).unwrap();
        let out_data = out.to_tensor().host_slice().into_owned();
        assert_eq!(out_data[0].to_bits(), nan.to_bits());
        assert_eq!(out_data[1], 0.0);
        assert!(out_data[1].is_sign_positive());
        assert_eq!(out_data[2], 1.0);
        assert_eq!(out_data[3], 2.0);
    }

    // --- tril: 勾配（マスク。bit 一致） ---

    #[test]
    fn tril_gradient_matches_mask() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let y = tril(&x, 0).unwrap();
        let w = tape.var(&t(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]));
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        // tril(k=0) は位置 [0][1] のみ 0 化 → 勾配もそこだけ 0。
        assert_eq!(
            bits(dx.host_slice().as_ref()),
            bits(&[10.0, 0.0, 30.0, 40.0])
        );
    }

    // --- triu: 勾配（マスク。bit 一致） ---

    #[test]
    fn triu_gradient_matches_mask() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let y = triu(&x, 0).unwrap();
        let w = tape.var(&t(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]));
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        // triu(k=0) は位置 [1][0] のみ 0 化 → 勾配もそこだけ 0。
        assert_eq!(
            bits(dx.host_slice().as_ref()),
            bits(&[10.0, 20.0, 0.0, 40.0])
        );
    }

    // --- triu: k=0・正負・範囲外・非正方 ---

    #[test]
    fn triu_k0_2x2() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = triu(&x, 0).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 0.0, 4.0]
        );
    }

    #[test]
    fn triu_positive_diagonal() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = triu(&x, 1).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![0.0, 2.0, 0.0, 0.0]
        );
    }

    #[test]
    fn triu_negative_diagonal() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = triu(&x, -1).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn triu_large_negative_diagonal_is_noop_no_new_node() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let before = tape.len();
        let out = triu(&x, -100).unwrap();
        assert_eq!(tape.len(), before);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            x.to_tensor().host_slice().into_owned()
        );
    }

    #[test]
    fn triu_large_positive_diagonal_zeroes_everything() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = triu(&x, 100).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn triu_non_square() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let out = triu(&x, 0).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0]
        );
    }

    #[test]
    fn triu_rank1_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            triu(&x, 0),
            Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }))
        ));
    }

    // --- diag: 1-D -> 2-D（k=0・正負・isize::MIN/MAX） ---

    #[test]
    fn diag_1d_to_2d_k0() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let out = diag(&x, 0).unwrap();
        assert_eq!(out.to_tensor().shape(), &[3, 3]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
        );
    }

    #[test]
    fn diag_1d_to_2d_positive_k() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = diag(&x, 1).unwrap();
        assert_eq!(out.to_tensor().shape(), &[3, 3]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn diag_1d_to_2d_negative_k() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = diag(&x, -1).unwrap();
        assert_eq!(out.to_tensor().shape(), &[3, 3]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]
        );
    }

    #[test]
    fn diag_1d_empty() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0]));
        let out = diag(&x, 0).unwrap();
        assert_eq!(out.to_tensor().shape(), &[0, 0]);
    }

    // --- diag: 2-D -> 1-D（両方向・範囲外オフセット・空） ---

    #[test]
    fn diag_2d_to_1d_k0_square() {
        let tape = Tape::new();
        let x = tape.var(&t(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
        ));
        let out = diag(&x, 0).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 5.0, 9.0]
        );
    }

    #[test]
    fn diag_2d_to_1d_positive_k() {
        let tape = Tape::new();
        // [[1,2,3],[4,5,6]]
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let out = diag(&x, 1).unwrap();
        assert_eq!(out.to_tensor().shape(), &[2]);
        assert_eq!(out.to_tensor().host_slice().into_owned(), vec![2.0, 6.0]);
    }

    #[test]
    fn diag_2d_to_1d_negative_k() {
        let tape = Tape::new();
        // [[1,2],[3,4],[5,6]]
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
        let out = diag(&x, -1).unwrap();
        assert_eq!(out.to_tensor().shape(), &[2]);
        assert_eq!(out.to_tensor().host_slice().into_owned(), vec![3.0, 6.0]);
    }

    #[test]
    fn diag_2d_to_1d_offset_beyond_bounds_is_empty() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = diag(&x, 100).unwrap();
        assert_eq!(out.to_tensor().shape(), &[0]);
        let out_neg = diag(&x, -100).unwrap();
        assert_eq!(out_neg.to_tensor().shape(), &[0]);
    }

    #[test]
    fn diag_2d_non_square() {
        let tape = Tape::new();
        // [[1,2,3],[4,5,6],[7,8,9],[10,11,12]] (4x3)
        let x = tape.var(&t(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[4, 3],
        ));
        let out = diag(&x, 0).unwrap();
        assert_eq!(out.to_tensor().shape(), &[3]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 5.0, 9.0]
        );
    }

    #[test]
    fn diag_rank3_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0; 8], &[2, 2, 2]));
        assert!(matches!(
            diag(&x, 0),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    // --- diag: 勾配（bit 一致。gather の scatter-add は寄与 1 つ） ---

    #[test]
    fn diag_2d_to_1d_gradient_matches_reference() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let y = diag(&x, 0).unwrap();
        let w = tape.var(&t(vec![10.0, 20.0], &[2]));
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(
            bits(dx.host_slice().as_ref()),
            bits(&[10.0, 0.0, 0.0, 20.0])
        );
    }

    // --- trace: 正方・非正方・空 ---

    #[test]
    fn trace_square() {
        let tape = Tape::new();
        let x = tape.var(&t(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
        ));
        let out = trace(&x).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 15.0);
    }

    #[test]
    fn trace_non_square() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let out = trace(&x).unwrap();
        // diag(k=0) = [1, 5] → sum = 6
        assert_eq!(out.to_tensor().host_slice()[0], 6.0);
    }

    #[test]
    fn trace_empty_is_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0, 3]));
        let out = trace(&x).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn trace_rank1_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            trace(&x),
            Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }))
        ));
    }

    // --- trace: 勾配（単位行列） ---

    #[test]
    fn trace_gradient_is_identity() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let loss = trace(&x).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(bits(dx.host_slice().as_ref()), bits(&[1.0, 0.0, 0.0, 1.0]));
    }

    // --- outer: 基本・n=0・非 contiguous ---

    #[test]
    fn outer_basic() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0], &[2]));
        let b = tape.var(&t(vec![10.0, 20.0, 30.0], &[3]));
        let out = outer(&a, &b).unwrap();
        assert_eq!(out.to_tensor().shape(), &[2, 3]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0]
        );
    }

    #[test]
    fn outer_zero_length() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![], &[0]));
        let b = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = outer(&a, &b).unwrap();
        assert_eq!(out.to_tensor().shape(), &[0, 2]);
    }

    #[test]
    fn outer_non_contiguous_input() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let xt = x.transpose(0, 1).unwrap(); // [[1,4],[2,5],[3,6]]
        let a = xt.narrow(1, 0, 1).unwrap().squeeze(Some(1)).unwrap(); // [1,2,3]
        let b = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = outer(&a, &b).unwrap();
        assert_eq!(out.to_tensor().shape(), &[3, 2]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]
        );
    }

    #[test]
    fn outer_rank_mismatch_is_error() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let b = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            outer(&a, &b),
            Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 1,
                actual: 2
            }))
        ));
    }

    // --- outer: 勾配（行和・列和。有限差分検算） ---

    #[test]
    fn outer_gradient_matches_finite_difference() {
        let eps = 1e-3f32;
        let a_base = vec![0.7f32, -1.3];
        let b_base = vec![2.1f32, 0.4, -0.9];

        let eval = |a_data: &[f32], b_data: &[f32]| -> f32 {
            let tape = Tape::new();
            let a = tape.var(&t(a_data.to_vec(), &[2]));
            let b = tape.var(&t(b_data.to_vec(), &[3]));
            let y = outer(&a, &b).unwrap();
            let loss = y.sum(None).unwrap();
            loss.to_tensor().host_slice()[0]
        };

        let tape = Tape::new();
        let a = tape.var(&t(a_base.clone(), &[2]));
        let b = tape.var(&t(b_base.clone(), &[3]));
        let y = outer(&a, &b).unwrap();
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let da = grads.get(&a).unwrap().unwrap().host_slice().into_owned();

        for i in 0..a_base.len() {
            let mut plus = a_base.clone();
            plus[i] += eps;
            let mut minus = a_base.clone();
            minus[i] -= eps;
            let numeric = (eval(&plus, &b_base) - eval(&minus, &b_base)) / (2.0 * eps);
            assert!(
                (numeric - da[i]).abs() < 1e-2,
                "outer 勾配の有限差分検算が解析解と乖離: i={i}, numeric={numeric}, analytic={}",
                da[i]
            );
        }
    }

    // --- dot: 基本・n=0・非 contiguous・長さ不一致 ---

    #[test]
    fn dot_basic() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let b = tape.var(&t(vec![4.0, 5.0, 6.0], &[3]));
        let out = dot(&a, &b).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 32.0);
    }

    #[test]
    fn dot_zero_length() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![], &[0]));
        let b = tape.var(&t(vec![], &[0]));
        let out = dot(&a, &b).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn dot_length_mismatch_is_error() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0], &[2]));
        let b = tape.var(&t(vec![1.0], &[1]));
        assert!(matches!(
            dot(&a, &b),
            Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
        ));
    }

    #[test]
    fn dot_rank_mismatch_is_error() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let b = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            dot(&a, &b),
            Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 1,
                actual: 2
            }))
        ));
    }

    // --- dot: 勾配（相手ベクトル。bit 一致） ---

    #[test]
    fn dot_gradient_matches_other_operand() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let b = tape.var(&t(vec![4.0, 5.0, 6.0], &[3]));
        let loss = dot(&a, &b).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let da = grads.get(&a).unwrap().unwrap();
        let db = grads.get(&b).unwrap().unwrap();
        assert_eq!(bits(da.host_slice().as_ref()), bits(&[4.0, 5.0, 6.0]));
        assert_eq!(bits(db.host_slice().as_ref()), bits(&[1.0, 2.0, 3.0]));
    }

    // --- エラー系: マスク確保上限（REQ-8。`broadcast_to` 由来の巨大軸長）---

    #[test]
    fn tril_on_broadcast_view_rejects_practically_unallocatable_size() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0], &[1, 1]));
        let big = x.broadcast_to(&[1, 1_000_000_000]).unwrap();
        let err = tril(&big, 0).expect_err("broadcast 由来の巨大軸長も確保前に拒否されるはず");
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ElementCountOverflow)
        ));
    }
}
