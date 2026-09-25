//! `floor`・`ceil`・`round`・`sign`・`reciprocal`・`rsqrt`・`erf`・
//! `pow_scalar` の 8 種要素演算（イシュー #2145・親 #2131「Tier 1」の
//! 補完）。
//!
//! **新規 `Op` はゼロ（受け入れ条件）**: いずれも既存の `Op::ScalarUnary`
//! （`tensor_core::ScalarUnaryOp` の `Floor`／`Ceil`／`Round`／`Sign`／
//! `Reciprocal`／`Rsqrt`／`Erf`／`PowScalar` variant への dispatch。
//! `PowScalar` はイシュー #1634 で既に定義済みで、本モジュールは入口
//! 関数 `pow_scalar` を追加するのみ）への薄い委譲のみで構成する。
//! `crate::var::Var::scalar_unary`（`pub(crate)`）が forward の実体化・
//! バックエンド dispatch（既定 `Unsupported` → ホスト参照実装
//! フォールバック）・tape 記録を担う共通経路であり、本モジュールは
//! それぞれの `ScalarUnaryOp` variant を選ぶだけの自由関数を並べる。
//!
//! **facade 非公開（意図的）**: `crates/autodiff/src/rearrange_ops.rs`・
//! `bool_ops.rs` モジュール doc と同じ理由・同じ判断枠組みによる。
//! `Var` は facade（`fandhe_ai` クレート）から直接再エクスポートされる
//! ため、`Var` への inherent メソッド追加は即座に facade 公開面へ出て
//! しまう。親 #2131 はこのツリーに限り「設計判断記録 → 承認 → 実装」
//! の 2 段階を定め、イシュー #2145 本文も facade 公開面（`Var::floor`
//! 等の委譲メソッド追加）を承認事項として明示するため、承認が取れる
//! まではこれらを自由関数として `Var` の外に置き到達不能にする
//! （`docs/autodiff-scalar-unary-ops-decision.md` §9「承認事項」）。
//! 承認後は `Var::floor` 等の薄い委譲メソッドを追加し、facade 側の
//! 保留ガード（`crates/facade/src/lib.rs::
//! VarScalarUnaryOpsHoldDoctestGuard`）を撤去する。
//!
//! **数値規約（`docs/autodiff-scalar-unary-ops-decision.md` §3 が正。
//! forward 数式の単一情報源は `tensor_core::scalar_op::ScalarUnaryOp::
//! apply`）**:
//! - `floor`／`ceil`／`sign` は IEEE のまま伝播し、`round` は偶数丸め
//!   （`f32::round_ties_even`。PyTorch `torch.round` と同じタイブレーク
//!   規則）。いずれも区分定数のため勾配は恒等的に `0`
//!   （`ScalarUnaryOp::is_piecewise_constant`）。`autodiff::grad` 側の
//!   `Op::ScalarUnary` VJP 分岐は upstream の `inf`／`NaN` による
//!   `0 * inf = NaN` 汚染を避けるため、乗算を経由せずゼロテンソルを
//!   直接生成する（`ScalarBinaryOp::is_comparison` と同型。PR #1823
//!   の教訓・`eval/scalar.rs` モジュール doc §4）。
//! - `sign` は `x > 0 → 1`／`x < 0 → -1`／`±0` を含むそれ以外 → `0`
//!   （`f32::signum` は `±0` に対し `±1` を返すため使わない）。`NaN`
//!   入力は `NaN` を返す。
//! - `reciprocal`（`1/x`）・`rsqrt`（`1/sqrt(x)`）は定義域外・`0`
//!   入力でも IEEE のまま（`inf`／`NaN`）panic しない。
//! - `erf` は `Gelu`（誤差関数版）と同じ `f64` 精度の自作近似
//!   （Abramowitz–Stegun 7.1.26。`erf_f64`）を `f64` で計算してから
//!   `f32` へ 1 回だけ downcast する。
//! - `pow_scalar`（`x.powf(exponent)`）は `exponent == 0.0` で勾配が
//!   常に `0` にマスクされる（イシュー #1634／#1686 是正済み）。
//!
//! **CUDA／Metal**: 専用カーネルはスコープ外で、`ScalarUnaryOp` の
//! `unary_kernel_source` が新 7 kind に対し明示的に `None` を返し
//! （`crates/backend-cuda/src/kernels_scalar_op.rs`・`crates/
//! backend-metal/src/scalar_op_source.rs`）、既定の `Unsupported` から
//! ホスト参照実装（`autodiff::eval::scalar::unary`）へフォールバック
//! する。
//!
//! **`create_graph`（二階微分）は非対応**: `crate::create_graph::
//! scalar_unary_replayable` は新 7 kind に触れておらず（末尾
//! ワイルドカードで `false`）、`PowScalar` のみ既に replayable。
//! 二階微分対応は別イシューのスコープ（本モジュール doc「スコープ外」
//! 参照）。

use crate::error::AutodiffError;
use crate::var::Var;
use fandhe_ai_tensor_core::ScalarUnaryOp;

/// 要素ごとの床関数（PyTorch `torch.floor` 相当）。勾配は恒等的に `0`
/// （区分定数）。
pub fn floor<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Floor)
}

/// 要素ごとの天井関数（PyTorch `torch.ceil` 相当）。勾配は恒等的に `0`
/// （区分定数）。
pub fn ceil<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Ceil)
}

/// 要素ごとの偶数丸め（PyTorch `torch.round` 相当。`f32::round`〈0 から
/// 遠い側への丸め〉ではなく `round_ties_even` を使う）。勾配は恒等的に
/// `0`（区分定数）。
pub fn round<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Round)
}

/// 要素ごとの符号関数（PyTorch `torch.sign` 相当）。`sign(±0) = 0`・
/// `sign(NaN) = NaN`。勾配は恒等的に `0`（区分定数）。
pub fn sign<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Sign)
}

/// 要素ごとの逆数 `1/x`（PyTorch `torch.reciprocal` 相当）。`x == 0` は
/// `inf` を返し panic しない。
pub fn reciprocal<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Reciprocal)
}

/// 要素ごとの逆平方根 `1/sqrt(x)`（PyTorch `torch.rsqrt` 相当）。`x < 0`
/// は `NaN`、`x == 0` は `inf`（いずれも IEEE のまま）。
pub fn rsqrt<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Rsqrt)
}

/// 要素ごとの誤差関数 `erf(x)`（PyTorch `torch.erf` 相当）。`Gelu`
/// （誤差関数版）と同じ `f64` 精度の自作近似を経由する。
pub fn erf<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Erf)
}

/// 定数指数へのスカラー累乗 `x.powf(exponent)`（PyTorch `x ** c` 相当。
/// `ScalarUnaryOp::PowScalar` はイシュー #1634 で既に定義済み。本関数は
/// その入口のみを追加する）。`exponent == 0.0` は勾配が常に `0` に
/// マスクされる（`eval::scalar::unary_grad_factor` 参照）。
pub fn pow_scalar<'t>(x: &Var<'t>, exponent: f32) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::PowScalar { exponent })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;
    use fandhe_ai_tensor_core::Tensor;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn each_entry_forwards_to_expected_scalar_unary_op() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[-1.5, 2.5, 0.0, 4.0], &[4]));

        let expected_floor = ScalarUnaryOp::Floor.apply(-1.5);
        assert_eq!(
            floor(&x).unwrap().value().get(&[0]).unwrap(),
            expected_floor
        );

        let expected_ceil = ScalarUnaryOp::Ceil.apply(-1.5);
        assert_eq!(ceil(&x).unwrap().value().get(&[0]).unwrap(), expected_ceil);

        let expected_round = ScalarUnaryOp::Round.apply(2.5);
        assert_eq!(
            round(&x).unwrap().value().get(&[1]).unwrap(),
            expected_round
        );

        let expected_sign = ScalarUnaryOp::Sign.apply(-1.5);
        assert_eq!(sign(&x).unwrap().value().get(&[0]).unwrap(), expected_sign);

        let expected_reciprocal = ScalarUnaryOp::Reciprocal.apply(4.0);
        assert_eq!(
            reciprocal(&x).unwrap().value().get(&[3]).unwrap(),
            expected_reciprocal
        );

        let expected_rsqrt = ScalarUnaryOp::Rsqrt.apply(4.0);
        assert_eq!(
            rsqrt(&x).unwrap().value().get(&[3]).unwrap(),
            expected_rsqrt
        );

        let expected_erf = ScalarUnaryOp::Erf.apply(2.5);
        assert_eq!(erf(&x).unwrap().value().get(&[1]).unwrap(), expected_erf);

        let expected_pow = ScalarUnaryOp::PowScalar { exponent: 3.0 }.apply(2.5);
        assert_eq!(
            pow_scalar(&x, 3.0).unwrap().value().get(&[1]).unwrap(),
            expected_pow
        );
    }
}
