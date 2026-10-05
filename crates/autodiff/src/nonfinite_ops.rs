//! 非有限値の判定 3 種（`isnan`・`isinf`・`isfinite`）と置換 `nan_to_num`
//! （イシュー #2635・親 #2625「Phase 4」）。
//!
//! **新規 `Op` はゼロ**: forward の数式は `tensor_core::ScalarUnaryOp` の 4
//! variant（`IsNan`／`IsInf`／`IsFinite`／`NanToNum`）が単一情報源
//! （`ScalarUnaryOp::apply`）で、バックエンド dispatch（既定 `Unsupported`
//! → ホスト参照実装フォールバック）は `grad::scalar_unary_with_fallback`、
//! `nan_to_num` の tape 記録は `Var::scalar_unary`（`pub(crate)`）が担う。
//!
//! **PyTorch 相当**: `torch.isnan`／`torch.isinf`／`torch.isfinite`／
//! `torch.nan_to_num`（PyTorch 2.14.0 の実行値 fixture
//! `tests/fixtures/nonfinite-pytorch-reference/` と突合）。
//!
//! **判定 3 種は非微分・tape 非記録**: `bool_ops::compare_bool` と同じ
//! 手順（確保前サイズ検査 → 層 1 実体化 → f32 マスク計算 → 既存 cast 経路で
//! bool 化 → shape 事後検査）で厳密な `Tensor<bool>` を返し、
//! `Tape::push_eager`／`push_lazy` は呼ばない（出力が bool で tape ノード
//! 表現に乗らないため。`Var::argmax`／`bool_ops` と同型）。
//!
//! **`nan_to_num` は微分可能**: 有限入力位置は上流勾配をそのまま通し、
//! 非有限入力位置（置換値が出力された位置）は 0。乗算を経由せず要素選択で
//! 返すため上流が `inf`／`NaN` でも汚染されない（`grad.rs` の
//! `Op::ScalarUnary` 分岐。差分は `docs/autodiff-nonfinite-ops-decision.md`
//! §5）。引数 `None` は PyTorch 既定（`nan = 0.0`・`posinf = f32::MAX`・
//! `neginf = f32::MIN`）へ解決する。置換値自体が非有限でも拒否せず書き込む。
//!
//! **公開形は未承認（保留）**: facade（`fandhe_ai`）へは公開しない。推奨案は
//! `Var` の委譲メソッドで、承認依頼は #2677・公開は承認後の #2678。保留は
//! facade の `NonfiniteOpsHoldDoctestGuard` と `tests/api_surface.rs` の
//! 否定ガードで機械固定している（`docs/autodiff-nonfinite-ops-decision.md`
//! §7・§9）。
//!
//! **CUDA／Metal**: 専用カーネルはスコープ外で `unary_kernel_source` が明示
//! `None` を返し、`Unsupported` からホスト参照実装へフォールバックする。
//!
//! **対象外**: `create_graph`（高階微分）・activation checkpoint・f64
//! 自動微分経路・GPU 専用カーネル。

use fandhe_ai_tensor_core::{BackendError, ScalarUnaryOp, ShapeError, Tensor};

use crate::bool_ops::checked_bytes_for;
use crate::error::AutodiffError;
use crate::grad::{cast_from_f32_with_fallback, scalar_unary_with_fallback};
use crate::tape::materialize_fallible;
use crate::var::Var;

/// 判定 3 種の共通実装。`op`（`IsNan`／`IsInf`／`IsFinite`）の 0/1
/// マスクを既存の scalar_unary 経路で求め、cast 経路で `Tensor<bool>` 化する。
///
/// 手順: ①`checked_bytes_for::<f32>`／`::<bool>` による確保前サイズ検査
/// （要素数 1 の `Var` を巨大 shape へ `broadcast_to` した view を渡された
/// 場合に、下流の実体化が capacity overflow で panic するのを防ぐ。
/// 本番経路 panic 禁止規約）→ ②層 1 実体化（`nodes` の借用はブロック内で
/// 閉じる）→ ③`scalar_unary_with_fallback`（`Unsupported` のみホスト
/// 参照実装へフォールバック）→ ④`cast_from_f32_with_fallback::<bool>` →
/// ⑤戻り値 shape の事後検査。
fn predicate<'t>(x: &Var<'t>, op: ScalarUnaryOp) -> Result<Tensor<bool>, AutodiffError> {
    let shape = x.shape();
    checked_bytes_for::<f32>(&shape)?;
    checked_bytes_for::<bool>(&shape)?;
    let input = {
        let nodes = x.tape().nodes.borrow();
        materialize_fallible(&nodes, x.tape().ops(), x.node_id())?.clone()
    };
    let mask_f32 = scalar_unary_with_fallback(x.tape().ops(), op, &input)?;
    let mask_bool = cast_from_f32_with_fallback::<bool>(x.tape().ops(), &mask_f32)?;
    if mask_bool.shape() != shape.as_slice() {
        return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: mask_bool.shape().to_vec(),
                rhs: shape,
            },
        )));
    }
    Ok(mask_bool)
}

/// 要素ごとの NaN 判定（`torch.isnan` 相当）。符号・ペイロード問わず
/// `NaN` のみ真。非微分・tape 非記録（モジュール doc 参照）。
pub fn isnan<'t>(x: &Var<'t>) -> Result<Tensor<bool>, AutodiffError> {
    predicate(x, ScalarUnaryOp::IsNan)
}

/// 要素ごとの無限大判定（`torch.isinf` 相当）。`±inf` のみ真（`NaN` は
/// 偽）。非微分・tape 非記録。
pub fn isinf<'t>(x: &Var<'t>) -> Result<Tensor<bool>, AutodiffError> {
    predicate(x, ScalarUnaryOp::IsInf)
}

/// 要素ごとの有限判定（`torch.isfinite` 相当）。`NaN`／`±inf` 以外
/// （`±0`・非正規化数を含む）が真。非微分・tape 非記録。
pub fn isfinite<'t>(x: &Var<'t>) -> Result<Tensor<bool>, AutodiffError> {
    predicate(x, ScalarUnaryOp::IsFinite)
}

/// 非有限値の置換（`torch.nan_to_num(input, nan, posinf, neginf)` 相当）。
/// `None` は PyTorch 既定（`nan = 0.0`・`posinf = f32::MAX`・
/// `neginf = f32::MIN`）。有限値（`±0`・非正規化数を含む）はビットを
/// 変えずに通す。置換値自体が非有限でも拒否せず書き込む。微分可能
/// （有限入力位置のみ勾配 1。モジュール doc 参照）。
pub fn nan_to_num<'t>(
    x: &Var<'t>,
    nan: Option<f32>,
    posinf: Option<f32>,
    neginf: Option<f32>,
) -> Result<Var<'t>, AutodiffError> {
    checked_bytes_for::<f32>(&x.shape())?;
    x.scalar_unary(ScalarUnaryOp::NanToNum {
        nan: nan.unwrap_or(0.0),
        posinf: posinf.unwrap_or(f32::MAX),
        neginf: neginf.unwrap_or(f32::MIN),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn bools(t: &Tensor<bool>) -> Vec<bool> {
        t.contiguous().host_slice().into_owned()
    }

    #[test]
    fn predicates_match_apply_and_do_not_record_nodes() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(
            &Tensor::new(
                vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, 1.5],
                &[2, 3],
            )
            .unwrap(),
        );
        let before = tape.len();
        let n = isnan(&x).unwrap();
        let i = isinf(&x).unwrap();
        let f = isfinite(&x).unwrap();
        assert_eq!(tape.len(), before, "判定は tape へ積まない");
        assert_eq!(n.shape(), &[2, 3]);
        assert_eq!(bools(&n), [true, false, false, false, false, false]);
        assert_eq!(bools(&i), [false, true, true, false, false, false]);
        assert_eq!(bools(&f), [false, false, false, true, true, true]);
    }

    #[test]
    fn nan_to_num_defaults_and_explicit_values() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(
            &Tensor::new(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.5], &[4]).unwrap(),
        );
        let d = nan_to_num(&x, None, None, None).unwrap();
        let dv: Vec<f32> = (0..4).map(|k| d.value().get(&[k]).unwrap()).collect();
        assert_eq!(dv, [0.0, f32::MAX, f32::MIN, 2.5]);
        let e = nan_to_num(&x, Some(1.0), Some(2.0), Some(-3.0)).unwrap();
        let ev: Vec<f32> = (0..4).map(|k| e.value().get(&[k]).unwrap()).collect();
        assert_eq!(ev, [1.0, 2.0, -3.0, 2.5]);
    }

    #[test]
    fn huge_broadcast_view_is_rejected_before_allocation() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0], &[1]).unwrap());
        let big = x.broadcast_to(&[1usize << 61]).unwrap();
        assert!(matches!(
            isnan(&big),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
        assert!(nan_to_num(&big, None, None, None).is_err());
    }
}
