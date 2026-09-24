//! bool を返す比較 6 種・logical 3 種・`masked_select`（イシュー #2141・
//! 親 #2131「5-B 演算」）。
//!
//! 既存の `Var::gt`／`ge`／`lt`／`le`／`eq`／`ne`（`var.rs`）は f32 の
//! `0.0`／`1.0` マスクを `Var`（勾配ゼロの tape ノード）として返す。
//! 本モジュールはそれとは独立に、`torch.gt` 等と同じく厳密な
//! `Tensor<bool>` を返す非微分版を提供する。`Var::where_cond`／
//! `Var::masked_fill`（同ファイル）は既に条件として `&Tensor<bool>` を
//! 受け取るため、ここで作った bool マスクをそのまま接続できる。
//!
//! **facade 非公開（意図的）**: 本モジュールは `crate::lib::pub mod
//! bool_ops` として crate ルートから到達可能だが、`facade`
//! （`fandhe_ai` クレート）はこれを再エクスポートしない。`Var`
//! そのものは facade から直接再エクスポートされる（`crates/facade/
//! src/lib.rs`）ため、`Var` への inherent メソッド追加は即座に facade
//! 公開面へ出てしまう。イシュー #2141 本文は facade 公開面（10 項目）
//! を承認事項として明示列挙しており、親 #2131 はこのツリーに限り
//! 「設計判断記録 → 承認 → 実装」の 2 段階を定めるため、承認が
//! 取れるまでは自由関数として `Var` の外に置き到達不能にする
//! （`docs/autodiff-bool-ops-exposure-decision.md` §3・
//! `docs/unique-facade-exposure-decision.md` §4 と同じ判断枠組み）。
//! 承認後は `Var::gt_bool` 等の薄い委譲メソッドを追加し、facade 側の
//! 保留ガード（`crates/facade/src/lib.rs::VarBoolOpsHoldDoctestGuard`）
//! を撤去する。
//!
//! **数値契約**: 比較 6 種は IEEE 754 準拠（`NaN` を含む比較は `eq` を
//! 含め常に偽・`ne` のみ真。`-0.0 == +0.0` は真）。出力は既存の f32
//! マスク比較と同じ `crate::grad::scalar_binary_with_fallback` を経由
//! し、その 0.0/1.0 出力を `crate::grad::cast_from_f32_with_fallback`
//! （`v != 0.0` 契約。`docs/tensor-core-cast-design.md`）で bool 化する。
//! 丸めは一切入らない。
//!
//! **非微分・tape 非記録**: `Var::unique`／`Var::argmax`／`Var::cast`
//! と同型で、`Tape::push_eager`／`push_lazy` は一切呼ばない。出力が
//! bool・動的 shape・非 Var のいずれかであり、既存の tape ノード表現に
//! 乗らないため（`docs/unique-facade-exposure-decision.md` §3 案 A）。

use fandhe_ai_tensor_core::{BackendError, ScalarBinaryOp, ShapeError, Tensor, broadcast_shape};

use crate::error::AutodiffError;
use crate::grad::{cast_from_f32_with_fallback, scalar_binary_with_fallback};
use crate::tape::materialize_fallible;
use crate::var::Var;

/// `a`／`b`（同一 `Tape` 上の `Var`）を層 1 で実体化した `Tensor<f32>`
/// の組を返す（`Var::scalar_binary`／`Var::where_cond` と同じ「`nodes`
/// の `RefCell` 借用をこのブロック内に閉じ込め、返す前に解放する」
/// パターン。呼び出し後は `ops.*` を自由に呼べる）。
fn materialize_pair<'t>(
    a: &Var<'t>,
    b: &Var<'t>,
) -> Result<(Tensor<f32>, Tensor<f32>), AutodiffError> {
    let nodes = a.tape().nodes.borrow();
    let ops = a.tape().ops();
    let a_val = materialize_fallible(&nodes, ops, a.node_id())?.clone();
    let b_val = materialize_fallible(&nodes, ops, b.node_id())?.clone();
    Ok((a_val, b_val))
}

/// 比較 6 種の共通実装。`op` に応じた要素ごとの比較を NumPy 互換
/// ブロードキャストで評価し、厳密な `Tensor<bool>` を返す。
///
/// 手順: ①`check_same_tape`（テープ不一致は `TapeMismatch`）→
/// ②`broadcast_shape` で出力 shape を先に求める（shape 検証と実行の
/// 分離。`docs/fusion-graph-design.md` §3.5.1 と同じ設計方針）→
/// ③層 1 実体化 → ④`scalar_binary_with_fallback`（既存 CPU／CUDA／
/// Metal 比較カーネルを再利用。`Unsupported` のみホスト参照実装へ
/// フォールバック）→ ⑤`cast_from_f32_with_fallback::<bool>`（既存
/// cast カーネルを再利用）→ ⑥戻り値 shape の事後検査。
fn compare_bool<'t>(
    a: &Var<'t>,
    b: &Var<'t>,
    op: ScalarBinaryOp,
) -> Result<Tensor<bool>, AutodiffError> {
    a.check_same_tape(b)?;
    let out_shape = broadcast_shape(&a.shape(), &b.shape()).map_err(AutodiffError::Shape)?;
    let (a_val, b_val) = materialize_pair(a, b)?;
    let mask_f32 = scalar_binary_with_fallback(a.tape().ops(), op, &a_val, &b_val, &out_shape)?;
    let mask_bool = cast_from_f32_with_fallback::<bool>(a.tape().ops(), &mask_f32)?;
    if mask_bool.shape() != out_shape.as_slice() {
        return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: mask_bool.shape().to_vec(),
                rhs: out_shape,
            },
        )));
    }
    Ok(mask_bool)
}

/// `self > other` の bool 出力版（`torch.gt` 相当）。数値・非微分契約は
/// モジュール doc を参照。イシュー #2141（親 #2131）。
pub fn gt_bool<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Tensor<bool>, AutodiffError> {
    compare_bool(a, b, ScalarBinaryOp::Gt)
}

/// `self >= other` の bool 出力版（`torch.ge` 相当）。数値契約は
/// [`gt_bool`] を参照。イシュー #2141。
pub fn ge_bool<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Tensor<bool>, AutodiffError> {
    compare_bool(a, b, ScalarBinaryOp::Ge)
}

/// `self < other` の bool 出力版（`torch.lt` 相当）。数値契約は
/// [`gt_bool`] を参照。イシュー #2141。
pub fn lt_bool<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Tensor<bool>, AutodiffError> {
    compare_bool(a, b, ScalarBinaryOp::Lt)
}

/// `self <= other` の bool 出力版（`torch.le` 相当）。数値契約は
/// [`gt_bool`] を参照。イシュー #2141。
pub fn le_bool<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Tensor<bool>, AutodiffError> {
    compare_bool(a, b, ScalarBinaryOp::Le)
}

/// `self == other` の bool 出力版（`torch.eq` 相当。`NaN` 同士は偽）。
/// 数値契約は [`gt_bool`] を参照。イシュー #2141。
pub fn eq_bool<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Tensor<bool>, AutodiffError> {
    compare_bool(a, b, ScalarBinaryOp::Eq)
}

/// `self != other` の bool 出力版（`torch.ne` 相当。`NaN` が絡む比較は
/// 常に真）。数値契約は [`gt_bool`] を参照。イシュー #2141。
pub fn ne_bool<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Tensor<bool>, AutodiffError> {
    compare_bool(a, b, ScalarBinaryOp::Ne)
}

/// `checked_numel_for::<T>`（`fandhe_ai_tensor_core::tensor`。
/// `pub(crate)` のためクレートを跨いで共有できない）・`backend-cuda::
/// cast::checked_bytes_for`／`backend-metal::cast::checked_bytes_for`
/// と同型の独立複製（同じ理由による複製。可視性の意味論が異なる
/// クレートを跨ぐため個別に持つ）。要素数積の `usize` オーバーフロー
/// に加え、要素型 `T` 換算のバイトサイズが `Vec` の allocation 上限
/// （`isize::MAX` バイト）に収まるかも検査する。`Tensor::contiguous()`
/// （内部で無検査の `numel()` 乗算・`Vec::with_capacity(numel)` を
/// 呼ぶ）を呼ぶ直前に必ず通す。本モジュールはホスト常駐データを直接
/// 実体化するため（`Vec<usize>` に切り出す `broadcast_to`
/// と同型の view を巨大 shape へ broadcast してから `.contiguous()`
/// する経路は `checked_shape_numel` 単体では検出できない）、確保前に
/// 型付きエラーで拒否する（本番経路 panic 禁止規約
/// `.claude/rules/coding-rust.md`。codex-review P1 指摘の是正・
/// イシュー #2141・PR #2241）。
fn checked_bytes_for<T>(shape: &[usize]) -> Result<(), AutodiffError> {
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(ShapeError::ElementCountOverflow)
        .map_err(AutodiffError::Shape)?;
    let elem_size = std::mem::size_of::<T>();
    if elem_size > 0 {
        let bytes = numel
            .checked_mul(elem_size)
            .ok_or(ShapeError::ElementCountOverflow)
            .map_err(AutodiffError::Shape)?;
        if bytes > isize::MAX as usize {
            return Err(AutodiffError::Shape(ShapeError::ElementCountOverflow));
        }
    }
    Ok(())
}

/// `a`／`b`（`Tensor<bool>`）を共通形状へブロードキャストした
/// `Vec<bool>` の組を返す（[`logical_and`]／[`logical_or`] 共通実装）。
/// ホスト常駐のデータに対してのみ計算し、GPU 専用カーネル・tape は
/// 経由しない（イシュー #2141 の契約どおり本イシューの対象外）。
type BoolPairData = (Vec<usize>, Vec<bool>, Vec<bool>);

fn broadcast_bool_pair(a: &Tensor<bool>, b: &Tensor<bool>) -> Result<BoolPairData, AutodiffError> {
    let out_shape = broadcast_shape(a.shape(), b.shape()).map_err(AutodiffError::Shape)?;
    checked_bytes_for::<bool>(&out_shape)?;
    let a_bc = a
        .broadcast_to(&out_shape)
        .map_err(AutodiffError::Shape)?
        .contiguous();
    let b_bc = b
        .broadcast_to(&out_shape)
        .map_err(AutodiffError::Shape)?
        .contiguous();
    let a_vec = a_bc.host_slice().into_owned();
    let b_vec = b_bc.host_slice().into_owned();
    Ok((out_shape, a_vec, b_vec))
}

/// 要素ごとの論理積（`torch.logical_and` の bool 入力版）。NumPy 互換
/// ブロードキャストに対応する。イシュー #2141。
pub fn logical_and(a: &Tensor<bool>, b: &Tensor<bool>) -> Result<Tensor<bool>, AutodiffError> {
    let (out_shape, a_vec, b_vec) = broadcast_bool_pair(a, b)?;
    let data: Vec<bool> = a_vec.into_iter().zip(b_vec).map(|(x, y)| x && y).collect();
    Tensor::new(data, &out_shape).map_err(AutodiffError::Shape)
}

/// 要素ごとの論理和（`torch.logical_or` の bool 入力版）。NumPy 互換
/// ブロードキャストに対応する。イシュー #2141。
pub fn logical_or(a: &Tensor<bool>, b: &Tensor<bool>) -> Result<Tensor<bool>, AutodiffError> {
    let (out_shape, a_vec, b_vec) = broadcast_bool_pair(a, b)?;
    let data: Vec<bool> = a_vec.into_iter().zip(b_vec).map(|(x, y)| x || y).collect();
    Tensor::new(data, &out_shape).map_err(AutodiffError::Shape)
}

/// 要素ごとの否定（`torch.logical_not` の bool 入力版）。イシュー
/// #2141。
pub fn logical_not(a: &Tensor<bool>) -> Result<Tensor<bool>, AutodiffError> {
    checked_bytes_for::<bool>(a.shape())?;
    let a_c = a.contiguous();
    let data: Vec<bool> = a_c.host_slice().iter().map(|&x| !x).collect();
    Tensor::new(data, a.shape()).map_err(AutodiffError::Shape)
}

/// `mask`（`Tensor<bool>`）が真の位置の `x` の値を row-major 順に集めた
/// rank 1 の `Tensor<f32>` を返す（`torch.masked_select` 相当）。
/// イシュー #2141。
///
/// `x` と `mask` は共通形状へブロードキャストしてから評価する
/// （PyTorch と同じ）。**非微分・detached**: 出力 shape が `mask` の
/// 値（実行時にしか分からない真の個数）で動的に決まるため、静的
/// shape を前提とする既存の tape ノード表現には乗らない
/// （`docs/unique-facade-exposure-decision.md` §3 案 A と同じ理由）。
/// **PyTorch の `masked_select` との差異**: 本家は微分可能だが、本
/// 実装は非微分（`Var::unique`／`argmax` と同型の設計判断。差分は
/// モジュール doc・決定記録に明記）。`mask` が全て偽なら shape `[0]`
/// の空テンソルを返す（エラーにしない。PyTorch と同じ）。
pub fn masked_select<'t>(x: &Var<'t>, mask: &Tensor<bool>) -> Result<Tensor<f32>, AutodiffError> {
    let out_shape = broadcast_shape(&x.shape(), mask.shape()).map_err(AutodiffError::Shape)?;
    checked_bytes_for::<f32>(&out_shape)?;
    checked_bytes_for::<bool>(&out_shape)?;

    let x_val = {
        let nodes = x.tape().nodes.borrow();
        materialize_fallible(&nodes, x.tape().ops(), x.node_id())?.clone()
    };
    let x_bc = x_val
        .broadcast_to(&out_shape)
        .map_err(AutodiffError::Shape)?
        .contiguous();
    let mask_bc = mask
        .broadcast_to(&out_shape)
        .map_err(AutodiffError::Shape)?
        .contiguous();

    let x_slice = x_bc.host_slice();
    let mask_slice = mask_bc.host_slice();
    let selected: Vec<f32> = x_slice
        .iter()
        .zip(mask_slice.iter())
        .filter_map(|(&v, &m)| if m { Some(v) } else { None })
        .collect();
    let n = selected.len();
    Tensor::new(selected, &[n]).map_err(AutodiffError::Shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    fn tb(data: Vec<bool>, shape: &[usize]) -> Tensor<bool> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    // --- 比較 6 種: 同形状 ---

    #[test]
    fn compare_bool_same_shape() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let b = tape.var(&t(vec![2.0, 2.0, 2.0], &[3]));

        assert_eq!(
            gt_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![false, false, true]
        );
        assert_eq!(
            ge_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![false, true, true]
        );
        assert_eq!(
            lt_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![true, false, false]
        );
        assert_eq!(
            le_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![true, true, false]
        );
        assert_eq!(
            eq_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![false, true, false]
        );
        assert_eq!(
            ne_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![true, false, true]
        );
    }

    // --- 比較 6 種: ブロードキャスト（[2,1] と [3] → [2,3]） ---

    #[test]
    fn compare_bool_broadcasts() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0], &[2, 1]));
        let b = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let out = gt_bool(&a, &b).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(
            out.host_slice().into_owned(),
            vec![false, false, false, true, false, false]
        );
    }

    // --- 比較 6 種: スカラー（shape []） ---

    #[test]
    fn compare_bool_scalar() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![5.0], &[]));
        let b = tape.var(&t(vec![3.0], &[]));
        let out = gt_bool(&a, &b).unwrap();
        assert_eq!(out.shape(), &[] as &[usize]);
        assert_eq!(out.host_slice().into_owned(), vec![true]);
    }

    // --- IEEE の境界: NaN ---

    #[test]
    fn compare_bool_nan_is_false_except_ne() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![f32::NAN], &[1]));
        let b = tape.var(&t(vec![1.0], &[1]));
        assert_eq!(
            gt_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![false]
        );
        assert_eq!(
            ge_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![false]
        );
        assert_eq!(
            lt_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![false]
        );
        assert_eq!(
            le_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![false]
        );
        assert_eq!(
            eq_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![false]
        );
        assert_eq!(
            ne_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![true]
        );
    }

    // --- IEEE の境界: -0.0 と +0.0・±inf ---

    #[test]
    fn compare_bool_signed_zero_and_inf() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![-0.0, f32::INFINITY, f32::NEG_INFINITY], &[3]));
        let b = tape.var(&t(vec![0.0, f32::INFINITY, 0.0], &[3]));
        assert_eq!(
            eq_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![true, true, false]
        );
        assert_eq!(
            gt_bool(&a, &b).unwrap().host_slice().into_owned(),
            vec![false, false, false]
        );
    }

    // --- 非微分: tape.len() が変わらない ---

    #[test]
    fn compare_bool_does_not_push_tape_node() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0], &[2]));
        let b = tape.var(&t(vec![2.0, 1.0], &[2]));
        let before = tape.len();
        let _ = gt_bool(&a, &b).unwrap();
        let _ = masked_select(&a, &tb(vec![true, false], &[2])).unwrap();
        assert_eq!(tape.len(), before);
    }

    // --- エラー: テープ不一致・ブロードキャスト不可 ---

    #[test]
    fn compare_bool_errors() {
        let tape1 = Tape::new();
        let tape2 = Tape::new();
        let a = tape1.var(&t(vec![1.0], &[1]));
        let b = tape2.var(&t(vec![1.0], &[1]));
        assert!(matches!(gt_bool(&a, &b), Err(AutodiffError::TapeMismatch)));

        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let b = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(gt_bool(&a, &b), Err(AutodiffError::Shape(_))));
    }

    // --- logical: 真理値表 ---

    #[test]
    fn logical_truth_table() {
        let a = tb(vec![true, true, false, false], &[4]);
        let b = tb(vec![true, false, true, false], &[4]);
        assert_eq!(
            logical_and(&a, &b).unwrap().host_slice().into_owned(),
            vec![true, false, false, false]
        );
        assert_eq!(
            logical_or(&a, &b).unwrap().host_slice().into_owned(),
            vec![true, true, true, false]
        );
        assert_eq!(
            logical_not(&a).unwrap().host_slice().into_owned(),
            vec![false, false, true, true]
        );
    }

    // --- logical: ブロードキャスト ---

    #[test]
    fn logical_broadcasts() {
        let a = tb(vec![true, false], &[2, 1]);
        let b = tb(vec![true, true, false], &[3]);
        let out = logical_and(&a, &b).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(
            out.host_slice().into_owned(),
            vec![true, true, false, false, false, false]
        );
    }

    // --- logical: 非 contiguous な入力（transpose の view） ---

    #[test]
    fn logical_non_contiguous_input() {
        let a = tb(vec![true, false, false, true], &[2, 2])
            .transpose(0, 1)
            .expect("test fixture: transpose は rank 2 で常に成功する");
        let b = tb(vec![true, true, true, true], &[2, 2]);
        let out = logical_and(&a, &b).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        // transpose([[T,F],[F,T]]) = [[T,F],[F,T]]（対称のため見かけ上
        // 不変だが、経路として非 contiguous view を通すことを確認する）。
        assert_eq!(
            out.host_slice().into_owned(),
            vec![true, false, false, true]
        );
    }

    // --- logical: ブロードキャスト不可 ---

    #[test]
    fn logical_broadcast_incompatible_is_shape_error() {
        let a = tb(vec![true, false, true], &[3]);
        let b = tb(vec![true, false], &[2]);
        assert!(matches!(logical_and(&a, &b), Err(AutodiffError::Shape(_))));
    }

    // --- logical_not を 2 回適用すると元に戻る ---

    #[test]
    fn logical_not_involution() {
        let a = tb(vec![true, false, true], &[3]);
        let once = logical_not(&a).unwrap();
        let twice = logical_not(&once).unwrap();
        assert_eq!(twice.host_slice().into_owned(), a.host_slice().into_owned());
    }

    // --- masked_select: 全真・全偽・部分的 ---

    #[test]
    fn masked_select_all_true_false_partial() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![10.0, 20.0, 30.0], &[3]));

        let all_true = tb(vec![true, true, true], &[3]);
        assert_eq!(
            masked_select(&x, &all_true)
                .unwrap()
                .host_slice()
                .into_owned(),
            vec![10.0, 20.0, 30.0]
        );

        let all_false = tb(vec![false, false, false], &[3]);
        let empty = masked_select(&x, &all_false).unwrap();
        assert_eq!(empty.shape(), &[0]);

        let partial = tb(vec![true, false, true], &[3]);
        assert_eq!(
            masked_select(&x, &partial)
                .unwrap()
                .host_slice()
                .into_owned(),
            vec![10.0, 30.0]
        );
    }

    // --- masked_select: mask 側のブロードキャスト（x が [3]・mask が
    // [2,3]） ---

    #[test]
    fn masked_select_mask_broadcasts_over_x() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let mask = tb(vec![true, false, true, false, true, false], &[2, 3]);
        let out = masked_select(&x, &mask).unwrap();
        // x が [3]→[2,3] へブロードキャストされ [[1,2,3],[1,2,3]] になり、
        // mask [[T,F,T],[F,T,F]] で row-major に抽出すると [1,3,2]。
        assert_eq!(out.host_slice().into_owned(), vec![1.0, 3.0, 2.0]);
    }

    // --- masked_select: NaN payload の bit 保存 ---

    #[test]
    fn masked_select_preserves_nan_bits() {
        let tape = Tape::new();
        let nan_bits: u32 = 0x7fc0_1234;
        let x = tape.var(&t(vec![f32::from_bits(nan_bits), 1.0], &[2]));
        let mask = tb(vec![true, false], &[2]);
        let out = masked_select(&x, &mask).unwrap();
        assert_eq!(out.host_slice()[0].to_bits(), nan_bits);
    }

    // --- masked_select: shape 不整合エラー（ブロードキャスト不可） ---

    #[test]
    fn masked_select_shape_mismatch_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let mask = tb(vec![true, false], &[2]);
        assert!(matches!(
            masked_select(&x, &mask),
            Err(AutodiffError::Shape(_))
        ));
    }

    // --- codex-review P1 是正の回帰テスト（イシュー #2141・PR #2241）:
    // 巨大 broadcast view の `.contiguous()` 実体化は確保前に型付き
    // エラーで拒否され panic しない。`backend-cuda::cast::
    // checked_contiguous_rejects_huge_broadcast_view_input_without_panicking`
    // と同型。---

    #[test]
    fn logical_not_rejects_huge_broadcast_view_input_without_panicking() {
        let base = tb(vec![true], &[1usize]);
        let huge_len = (isize::MAX as usize) / std::mem::size_of::<bool>() + 10;
        let huge = base.broadcast_to(&[huge_len]).unwrap();
        assert_eq!(huge.shape(), &[huge_len]);

        let err =
            logical_not(&huge).expect_err("huge broadcast view の実体化は確保前に拒否されるはず");
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn logical_and_rejects_huge_broadcast_view_input_without_panicking() {
        let base = tb(vec![true], &[1usize]);
        let huge_len = (isize::MAX as usize) / std::mem::size_of::<bool>() + 10;
        let huge = base.broadcast_to(&[huge_len]).unwrap();
        let small = tb(vec![true], &[1usize]);

        let err = logical_and(&huge, &small)
            .expect_err("huge broadcast view の実体化は確保前に拒否されるはず");
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn masked_select_rejects_huge_broadcast_shape_without_panicking() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0], &[1usize]));
        let huge_len = (isize::MAX as usize) / std::mem::size_of::<f32>() + 10;
        let base = tb(vec![true], &[1usize]);
        let mask = base.broadcast_to(&[huge_len]).unwrap();

        let err = masked_select(&x, &mask)
            .expect_err("huge broadcast shape の実体化は確保前に拒否されるはず");
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ElementCountOverflow)
        ));
    }
}
