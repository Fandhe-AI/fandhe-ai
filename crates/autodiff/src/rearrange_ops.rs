//! `repeat`・`tile`・`flip`・`roll` の 4 種形状演算（イシュー #2143・親
//! #2131「5-B 演算」）。
//!
//! **新規 `Op` はゼロ（受け入れ条件）**: いずれも既存の `Var::index_select`
//! （実体は `Var::gather` → `Op::Gather`）と `Var::broadcast_to`
//! （`Op::BroadcastTo`）の合成のみで構成する。両演算は CPU・CUDA・
//! Metal の全バックエンドに経路があり（`gather` は既定 `Unsupported` で
//! ホスト参照実装へフォールバック）、専用カーネルなしで到達可能。
//!
//! **facade 非公開（意図的）**: `crates/autodiff/src/bool_ops.rs`
//! モジュール doc と同じ理由・同じ判断枠組みによる。`Var` は facade
//! （`fandhe_ai` クレート）から直接再エクスポートされるため、`Var` への
//! inherent メソッド追加は即座に facade 公開面へ出てしまう。イシュー
//! #2143 本文は facade 公開面を承認事項として明示し、親 #2131 はこの
//! ツリーに限り「設計判断記録 → 承認 → 実装」の 2 段階を定めるため、
//! 承認が取れるまでは自由関数として `Var` の外に置き到達不能にする
//! （`docs/autodiff-rearrange-ops-decision.md` §2.1）。承認後は
//! `Var::repeat` 等の薄い委譲メソッドを追加し、facade 側の保留ガード
//! （`crates/facade/src/lib.rs::VarRearrangeOpsHoldDoctestGuard`）を
//! 撤去する。
//!
//! **数値契約**: forward は値のコピーのみ（算術を含まない）ため 3
//! バックエンド間で構造的に bit 完全一致する（`NaN` の payload も保存
//! される）。backward（`Op::Gather` の VJP。`grad.rs`）は scatter-add で
//! 行う。`flip`／`roll` は各入力要素への寄与が常に 1 つのため
//! `0 + g` の単純代入になり bit 一致する。`repeat`／`tile` は `r` 個の
//! コピーの勾配を合算するため、GPU の scatter 加算順序次第では REQ-2
//! の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で
//! 比較する（`.claude/rules/coding-rust.md`）。
//!
//! **PyTorch との差分（doc に明記。承認後の facade 版でも据え置く
//! 予定）**:
//! - `roll` の `dims: None`（flatten してから roll する形）は非対応。
//!   `Var::reshape` の contiguous 制約（`ShapeError::
//!   NonContiguousReshape`）を回避する設計上の理由による
//!   （`Var::expand` の「-1 非対応」注記と同じ扱い）。
//! - 軸番号は非負のみ（PyTorch の負軸表記は非対応。`Var::squeeze`／
//!   `unsqueeze` 等、本クレートの他の形状演算と同じ規約）。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: 外部から渡る
//! `repeats`／`reps`／`dims`／`shifts` はバックエンドを呼ぶ前にすべて
//! 検査する（長さ・軸範囲・重複・`checked_mul` によるオーバーフロー・
//! `i32` 範囲）。巨大な繰り返し数による添字ベクタの過大確保は、確保前に
//! 出力要素数を検査して拒否する（本番経路 panic 禁止規約）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::var::Var;

/// 軸長 `n` を添字 `Tensor<i32>` の要素値として使える範囲（`0..=
/// i32::MAX`）に収まるか検査する。`Var::gather`（`index_select` の
/// 委譲先）は添字値を `i32` として受け取るため、軸長がこれを超える
/// 場合は添字を作る前に拒否する（`.claude/rules/coding-rust.md` 本番
/// 経路 panic 禁止・REQ-8 境界検査の趣旨）。
fn checked_axis_len_as_i32(n: usize) -> Result<(), AutodiffError> {
    i32::try_from(n)
        .map(|_| ())
        .map_err(|_| AutodiffError::Shape(ShapeError::ElementCountOverflow))
}

/// 添字ベクタ `Vec<i32>`（長さ `len`）を確保する前のサイズ検査
/// （`crate::bool_ops::checked_bytes_for` と同型の独立複製。同じ理由
/// による複製——モジュールをまたいで `pub(crate)` 化するほどの共有価値
/// がなく、検査対象の型が固定〈`i32`〉のため専用化した）。要素数積の
/// `usize` オーバーフローに加え、`Vec` の allocation 上限（`isize::MAX`
/// バイト）に収まるかも検査する。
fn checked_index_alloc_len(len: usize) -> Result<(), AutodiffError> {
    let bytes = len
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(ShapeError::ElementCountOverflow)
        .map_err(AutodiffError::Shape)?;
    if bytes > isize::MAX as usize {
        return Err(AutodiffError::Shape(ShapeError::ElementCountOverflow));
    }
    Ok(())
}

/// 軸 `d` が `rank` 範囲内かを検査する（`Var::squeeze`／`gather` 等
/// 既存の形状演算と同じ `ShapeError::AxisOutOfRange` 契約）。
fn checked_axis_in_range(d: usize, rank: usize) -> Result<(), AutodiffError> {
    if d >= rank {
        return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            axis: d,
            rank,
        }));
    }
    Ok(())
}

/// 指定軸を反転する（`torch.flip` 相当。イシュー #2143）。
///
/// `dims` は 0 個以上・重複不可（PyTorch と同じ。重複は
/// `ShapeError::DuplicateAxis`）。`dims` が空なら新しいノードを積まず
/// `x` をそのまま返す（`Var::dropout` の早期リターンと同型）。
///
/// 実装は各軸へ逆順添字（`[n-1, …, 0]`）の [`Var::index_select`] を
/// 順に適用する合成。`n == 0` の軸は添字長 0 になり自然に空テンソルを
/// 保つ（除算を伴わないため特別扱い不要）。
pub fn flip<'t>(x: &Var<'t>, dims: &[usize]) -> Result<Var<'t>, AutodiffError> {
    if dims.is_empty() {
        return Ok(*x);
    }
    let rank = x.shape().len();
    let mut seen = std::collections::HashSet::with_capacity(dims.len());
    for &d in dims {
        checked_axis_in_range(d, rank)?;
        if !seen.insert(d) {
            return Err(AutodiffError::Shape(ShapeError::DuplicateAxis { axis: d }));
        }
    }

    let mut cur = *x;
    for &d in dims {
        let n = cur.shape()[d];
        checked_axis_len_as_i32(n)?;
        checked_index_alloc_len(n)?;
        let idx_data: Vec<i32> = (0..n).rev().map(|i| i as i32).collect();
        let idx = Tensor::new(idx_data, &[n]).map_err(AutodiffError::Shape)?;
        cur = cur.index_select(d, &idx)?;
    }
    Ok(cur)
}

/// 指定軸を循環シフトする（`torch.roll` 相当。イシュー #2143）。
///
/// `shifts.len() == dims.len()` かつ 1 個以上を必須とする（`torch.roll`
/// の `dims=None`〈flatten してから roll する形〉は非対応。モジュール
/// doc「PyTorch との差分」参照）。`dims` の重複は PyTorch と同じく
/// 許容し、指定順に逐次適用する。
///
/// 軸ごとに `s = shift.rem_euclid(n)` を正規化し、`n == 0` または
/// `s == 0` の軸は添字を作らずスキップする（no-op）。それ以外は添字
/// `idx[j] = (j + n - s) % n` で [`Var::index_select`] する。
pub fn roll<'t>(x: &Var<'t>, shifts: &[isize], dims: &[usize]) -> Result<Var<'t>, AutodiffError> {
    if shifts.is_empty() || shifts.len() != dims.len() {
        return Err(AutodiffError::InvalidArgument(format!(
            "rearrange_ops::roll: shifts と dims は同じ長さ・1 個以上が\
             必要（shifts.len()={}, dims.len()={}）",
            shifts.len(),
            dims.len()
        )));
    }
    let rank = x.shape().len();
    for &d in dims {
        checked_axis_in_range(d, rank)?;
    }

    let mut cur = *x;
    for (&d, &shift) in dims.iter().zip(shifts) {
        let n = cur.shape()[d];
        if n == 0 {
            continue;
        }
        let n_isize = isize::try_from(n)
            .map_err(|_| AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
        let s = shift.rem_euclid(n_isize) as usize;
        if s == 0 {
            continue;
        }
        checked_axis_len_as_i32(n)?;
        checked_index_alloc_len(n)?;
        let idx_data: Vec<i32> = (0..n).map(|j| ((j + n - s) % n) as i32).collect();
        let idx = Tensor::new(idx_data, &[n]).map_err(AutodiffError::Shape)?;
        cur = cur.index_select(d, &idx)?;
    }
    Ok(cur)
}

/// 軸ごとに繰り返す（`torch.Tensor.repeat` 相当。イシュー #2143）。
///
/// `repeats.len() < rank` は `AutodiffError::InvalidArgument`（PyTorch と
/// 同じ）。`repeats.len() > rank` の場合は [`Var::broadcast_to`] で先頭に
/// 長さ 1 の軸を追加してから rank を揃える（view のため元が非 contiguous
/// でもよい）。
///
/// 軸ごとに `r == 1` ならスキップし、それ以外は添字
/// `idx = (0..n*r).map(|j| j % n)` で [`Var::index_select`] する。
/// `r == 0` または `n == 0` の軸は添字長 0 になり、対応する出力軸が
/// 長さ 0 になる（エラーにしない。PyTorch と同じ）。全軸が `r == 1` で
/// rank も変わらない場合は新しいノードを積まず `x` をそのまま返す。
pub fn repeat<'t>(x: &Var<'t>, repeats: &[usize]) -> Result<Var<'t>, AutodiffError> {
    let in_shape = x.shape();
    let rank = in_shape.len();
    if repeats.len() < rank {
        return Err(AutodiffError::InvalidArgument(format!(
            "rearrange_ops::repeat: repeats の長さ {} が rank {} 未満で\
             PyTorch 仕様に反する",
            repeats.len(),
            rank
        )));
    }

    let mut cur = *x;
    if repeats.len() > rank {
        let extra = repeats.len() - rank;
        let mut padded_shape = vec![1usize; extra];
        padded_shape.extend_from_slice(&in_shape);
        cur = cur.broadcast_to(&padded_shape)?;
    }

    // 各軸の添字ベクタ（`Vec<i32>`）を確保する前に、全軸を通した最終
    // 出力の総要素数を `checked_mul` で確定させる（codex-review 指摘・
    // PR #2256。従来は軸ごとに `n * r` のみ検証してから確保していたため、
    // 後続軸を検査する前に先頭軸だけで数 GB 規模の確保を試み、型付き
    // `ElementCountOverflow` を返す前に allocation failure で abort し
    // 得た。`cur.shape()` は broadcast 由来の stride-0 view でも論理
    // shape を返すため、ここで軸ごとの `n_d * r_d` と全軸積の両方を
    // 検査してから、後続ループで初めて確保に入る）。
    let cur_shape = cur.shape();
    let mut total_out_elems: usize = 1;
    for (d, &r) in repeats.iter().enumerate() {
        let n = cur_shape[d];
        checked_axis_len_as_i32(n)?;
        let axis_total = n
            .checked_mul(r)
            .ok_or(ShapeError::ElementCountOverflow)
            .map_err(AutodiffError::Shape)?;
        checked_index_alloc_len(axis_total)?;
        total_out_elems = total_out_elems
            .checked_mul(axis_total)
            .ok_or(ShapeError::ElementCountOverflow)
            .map_err(AutodiffError::Shape)?;
    }
    checked_index_alloc_len(total_out_elems)?;

    for (d, &r) in repeats.iter().enumerate() {
        if r == 1 {
            continue;
        }
        let n = cur.shape()[d];
        // `n * r` は上記の事前検証ループで既に overflow・確保上限の
        // 両方を確認済みのため、ここでは再検証せず素の乗算でよい。
        let total = n * r;
        // `total == 0`（`n == 0` または `r == 0`）のときは `0..total` が
        // 空のため、クロージャ内の `j % n` は評価されず `n == 0` による
        // ゼロ除算は起きない（`Iterator::map` の遅延評価による）。
        let idx_data: Vec<i32> = (0..total).map(|j| (j % n) as i32).collect();
        let idx = Tensor::new(idx_data, &[total]).map_err(AutodiffError::Shape)?;
        cur = cur.index_select(d, &idx)?;
    }
    Ok(cur)
}

/// 繰り返す（`torch.tile` 相当。イシュー #2143）。
///
/// `reps.len() < rank` の場合は PyTorch と同じく先頭を `1` で埋めて
/// rank に合わせてから [`repeat`] へ委譲する。`reps.len() >= rank` の
/// 場合は [`repeat`] にそのまま委譲する（`reps.len() > rank` の rank
/// 拡張は `repeat` 側の先頭軸追加に任せる）。
pub fn tile<'t>(x: &Var<'t>, reps: &[usize]) -> Result<Var<'t>, AutodiffError> {
    let rank = x.shape().len();
    if reps.len() >= rank {
        return repeat(x, reps);
    }
    let pad = rank - reps.len();
    let mut padded = vec![1usize; pad];
    padded.extend_from_slice(reps);
    repeat(x, &padded)
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

    // --- flip: 単一軸・複数軸・rank 0・軸長 0 ---

    #[test]
    fn flip_single_axis() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        let out = flip(&x, &[0]).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![4.0, 3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn flip_multiple_axes_2d() {
        let tape = Tape::new();
        // [[1,2,3],[4,5,6]] を両軸反転すると [[6,5,4],[3,2,1]]。
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let out = flip(&x, &[0, 1]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[2, 3]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn flip_empty_dims_is_noop_no_new_node() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let before = tape.len();
        let out = flip(&x, &[]).unwrap();
        assert_eq!(tape.len(), before);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            x.to_tensor().host_slice().into_owned()
        );
    }

    #[test]
    fn flip_zero_length_axis() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0, 2]));
        let out = flip(&x, &[0]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[0, 2]);
    }

    #[test]
    fn flip_axis_out_of_range_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            flip(&x, &[1]),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: 1,
                rank: 1
            }))
        ));
    }

    #[test]
    fn flip_duplicate_axis_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        assert!(matches!(
            flip(&x, &[0, 0]),
            Err(AutodiffError::Shape(ShapeError::DuplicateAxis { axis: 0 }))
        ));
    }

    // --- flip: 非 contiguous 入力（transpose の view） ---

    #[test]
    fn flip_non_contiguous_input() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let xt = x.transpose(0, 1).unwrap();
        let out = flip(&xt, &[0]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[3, 2]);
        // transpose: [[1,4],[2,5],[3,6]]、軸 0 反転: [[3,6],[2,5],[1,4]]。
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![3.0, 6.0, 2.0, 5.0, 1.0, 4.0]
        );
    }

    // --- flip: 勾配（各入力要素への寄与は常に 1 つ・bit 一致） ---

    #[test]
    fn flip_gradient_matches_reference() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        let y = flip(&x, &[0]).unwrap();
        let w = tape.var(&t(vec![10.0, 20.0, 30.0, 40.0], &[4]));
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        // dloss/dx[i] = w[flip 後の対応位置] = w の逆順。
        assert_eq!(
            bits(dx.host_slice().as_ref()),
            bits(&[40.0, 30.0, 20.0, 10.0])
        );
    }

    // --- roll: 単一軸・正シフト・負シフト・軸長超過シフト ---

    #[test]
    fn roll_positive_shift() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
        let out = roll(&x, &[2], &[0]).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![4.0, 5.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn roll_negative_shift() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
        let out = roll(&x, &[-2], &[0]).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![3.0, 4.0, 5.0, 1.0, 2.0]
        );
    }

    #[test]
    fn roll_shift_exceeding_axis_len_wraps() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let out = roll(&x, &[7], &[0]).unwrap(); // 7 % 3 == 1
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![3.0, 1.0, 2.0]
        );
    }

    #[test]
    fn roll_zero_shift_is_noop_no_new_node() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let before = tape.len();
        let out = roll(&x, &[0], &[0]).unwrap();
        assert_eq!(tape.len(), before);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            x.to_tensor().host_slice().into_owned()
        );
    }

    #[test]
    fn roll_duplicate_axis_applies_sequentially() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        // 同じ軸へ shift=1 を 2 回適用 == shift=2 相当。
        let out = roll(&x, &[1, 1], &[0, 0]).unwrap();
        let expected = roll(&x, &[2], &[0]).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            expected.to_tensor().host_slice().into_owned()
        );
    }

    #[test]
    fn roll_length_mismatch_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            roll(&x, &[1, 2], &[0]),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn roll_empty_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            roll(&x, &[], &[]),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn roll_axis_out_of_range_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            roll(&x, &[1], &[1]),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: 1,
                rank: 1
            }))
        ));
    }

    #[test]
    fn roll_zero_length_axis_is_noop() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0, 2]));
        let out = roll(&x, &[3], &[0]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[0, 2]);
    }

    // --- roll: 勾配（bit 一致） ---

    #[test]
    fn roll_gradient_matches_reference() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        let y = roll(&x, &[1], &[0]).unwrap(); // [4,1,2,3]
        let w = tape.var(&t(vec![10.0, 20.0, 30.0, 40.0], &[4]));
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        // y[i] = x[(i-1) mod 4] なので dloss/dx[j] = w[(j+1) mod 4]。
        assert_eq!(
            bits(dx.host_slice().as_ref()),
            bits(&[20.0, 30.0, 40.0, 10.0])
        );
    }

    // --- repeat: 基本・r==0・rank 拡張 ---

    #[test]
    fn repeat_basic_1d() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = repeat(&x, &[3]).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn repeat_2d_per_axis() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = repeat(&x, &[2, 1]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[4, 2]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn repeat_zero_produces_zero_length_axis() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = repeat(&x, &[0]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[0]);
    }

    #[test]
    fn repeat_rank_extension_adds_leading_axis() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = repeat(&x, &[2, 1]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[2, 2]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn repeat_all_ones_same_rank_is_noop_no_new_node() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let before = tape.len();
        let out = repeat(&x, &[1]).unwrap();
        assert_eq!(tape.len(), before);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            x.to_tensor().host_slice().into_owned()
        );
    }

    #[test]
    fn repeat_too_short_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        assert!(matches!(
            repeat(&x, &[2]),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn repeat_non_contiguous_input() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let xt = x.transpose(0, 1).unwrap(); // [[1,4],[2,5],[3,6]]
        let out = repeat(&xt, &[1, 2]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[3, 4]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 4.0, 1.0, 4.0, 2.0, 5.0, 2.0, 5.0, 3.0, 6.0, 3.0, 6.0]
        );
    }

    // --- repeat: 勾配（r 個のコピーの合算。整数値なので合算順序に
    // 依らず bit 一致） ---

    #[test]
    fn repeat_gradient_sums_contributions() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let y = repeat(&x, &[3]).unwrap(); // [1,2,1,2,1,2]
        let w = tape.var(&t(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[6]));
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        // 各入力要素は 3 回コピーされ、重み全て 1 なので dx = [3, 3]。
        assert_eq!(bits(dx.host_slice().as_ref()), bits(&[3.0, 3.0]));
    }

    // --- tile: 基本・reps 短い／長い ---

    #[test]
    fn tile_reps_shorter_than_rank_pads_leading_ones() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = tile(&x, &[2]).unwrap(); // reps=[2] は [1,2] へパディング
        assert_eq!(out.to_tensor().shape(), &[2, 4]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
        );
    }

    #[test]
    fn tile_reps_longer_than_rank_delegates_to_repeat() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = tile(&x, &[2, 1]).unwrap();
        let expected = repeat(&x, &[2, 1]).unwrap();
        assert_eq!(out.to_tensor().shape(), expected.to_tensor().shape());
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            expected.to_tensor().host_slice().into_owned()
        );
    }

    #[test]
    fn tile_matches_repeat_when_same_rank() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = tile(&x, &[2, 1]).unwrap();
        let expected = repeat(&x, &[2, 1]).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            expected.to_tensor().host_slice().into_owned()
        );
    }

    // --- 有限差分検算（1 ケース。repeat の合成勾配を独立に検証） ---

    #[test]
    fn repeat_gradient_matches_finite_difference() {
        let eps = 1e-3f32;
        let base = vec![0.7f32, -1.3, 2.1];
        let weights = vec![1.0f32, 0.5, -0.25, 2.0, 0.1, -1.0];

        let eval = |data: &[f32]| -> f32 {
            let tape = Tape::new();
            let x = tape.var(&t(data.to_vec(), &[3]));
            let w = tape.var(&t(weights.clone(), &[6]));
            let y = repeat(&x, &[2]).unwrap();
            let loss = y.mul(&w).unwrap().sum(None).unwrap();
            loss.to_tensor().host_slice()[0]
        };

        let tape = Tape::new();
        let x = tape.var(&t(base.clone(), &[3]));
        let w = tape.var(&t(weights.clone(), &[6]));
        let y = repeat(&x, &[2]).unwrap();
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        let dx = dx.host_slice().into_owned();

        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;
            let numeric = (eval(&plus) - eval(&minus)) / (2.0 * eps);
            assert!(
                (numeric - dx[i]).abs() < 1e-2,
                "有限差分検算が解析解と乖離: i={i}, numeric={numeric}, analytic={}",
                dx[i]
            );
        }
    }

    // --- エラー系: オーバーフロー（境界検査。REQ-8） ---

    #[test]
    fn repeat_rejects_element_count_overflow_without_panicking() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0; 2], &[2]));
        let err = repeat(&x, &[usize::MAX]).expect_err("巨大な repeats は確保前に拒否されるはず");
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ElementCountOverflow)
        ));
    }
}
