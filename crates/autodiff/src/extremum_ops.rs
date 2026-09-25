//! `amax`・`amin`（イシュー #2154・親 #2131「Phase 5」）の均等分配
//! VJP 実装。決定 doc `docs/autodiff-amax-grad-distribution-decision.md`
//! §5「確定方針」が定める 3 点をそのまま実装する: (1) 既存
//! `Var::max`／`min`／`max_dims`（先勝ち決定的方式。イシュー #1718 で
//! 出荷済み挙動として維持を確定）とは別の `Op`（`crate::tape::Op::
//! Amax`／`Op::Amin`。クレート非公開のためリンクにはしない）を新設し、
//! (2) 別 VJP ヘルパー（`crate::grad` 内 `extremum_even_split_vjp`。
//! 同じく非公開。タイに `g / k` を均等分配）を実装し、(3) 演算が
//! `g / k` の除算のみで加算縮約を経由しないため
//! `.claude/rules/coding-rust.md` の f64 長軸縮約契約は対象外とする。
//!
//! **facade 非公開（意図的）**: `crates/autodiff/src/reduce_ops.rs`
//! モジュール doc（イシュー #2147）・`matrix_ops.rs` と同じ理由・同じ
//! 判断枠組みによる。`Var` は facade（`fandhe_ai` クレート）から直接
//! 再エクスポートされるため、`Var` への inherent メソッド追加は即座に
//! facade 公開面へ出てしまう。イシュー #2154 本文は facade 公開面
//! （`Var::amax`／`amin` の委譲メソッド追加）を承認事項として明示して
//! おり、承認が取れるまでは自由関数として `Var` の外に置き到達不能に
//! する。承認後は `Var::amax`／`amin` の薄い委譲メソッドを追加し、
//! facade 側の保留ガード（`crates/facade/src/lib.rs::
//! VarExtremumOpsHoldDoctestGuard`）を撤去する。
//!
//! **PyTorch 相当・数値契約**:
//!
//! | 演算 | PyTorch 相当 | forward | 勾配 |
//! |---|---|---|---|
//! | [`amax`] | `torch.amax(dim)` | `Var::max` と bit 同一 | 均等分配 `g/k` |
//! | [`amin`] | `torch.amin(dim)` | `Var::min` と bit 同一 | 均等分配 `g/k` |
//!
//! `k` は forward 記録値 `out_value` と IEEE 754 の `==` で一致する
//! 要素数（縮約軸上の整数カウント、丸めなし）。一致した各位置に
//! `g / (k as f32)` を置き、それ以外は 0 とする。`k == 0`（`amax` の
//! NaN 伝播 forward で `out_value` が `NaN` になった場合等）は全ゼロ
//! とする——先勝ち方式（`Var::max`／`min`）の契約違反時と同じ安全側の
//! 扱い（決定 doc §5・§6）。forward は既存の `max`／`min` の経路
//! （`self.tape.ops().max`／`crate::grad::min_with_fallback`）を
//! そのまま使うため `BackendOps` は拡張しない——forward 値は
//! `Var::max`／`Var::min` と bit 同一。
//!
//! **NaN の非対称性（決定 doc §6・対象外）**: `amax` の forward は
//! `Op::Max`（NaN 伝播）経由のため NaN を含む lane の出力は NaN・勾配は
//! 全ゼロ、`amin` は `Op::Min`（NaN 非伝播）経由のため NaN 以外の値を
//! 返す。既存 `max`／`min` の性質をそのまま引き継ぐ非対称性であり、
//! 本イシューでは是正しない。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**:
//! `checked_bytes_for::<f32>`（`crate::bool_ops`）による確保前のバイト
//! 数上限検査を、`reduce_ops.rs` の統一契約（モジュール doc「確保前の
//! バイト数上限検査は全公開入口の冒頭で一律に行う契約」）と同じ規律で
//! 適用する: [`amax`]・[`amin`] の両入口は、`dim` 範囲外検査
//! （`reduce_out_shape`）の直後・かつ実体化（`materialize_fallible`）
//! より前に、入力 shape・`out_shape` の両方を検査する。本番経路で
//! `unwrap()`／`expect()` は使わない。

use fandhe_ai_tensor_core::reduce_out_shape;

use crate::bool_ops::checked_bytes_for;
use crate::error::AutodiffError;
use crate::grad::min_with_fallback;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

/// [`amax`]／[`amin`] の両公開入口が冒頭で呼ぶ、確保前バイト数上限
/// 検査ヘルパー（`reduce_ops::ensure_alloc_fits_f32` と同型の複製。
/// `pub(crate)` 化を避け本モジュール内に複製する理由も同じ——
/// モジュールをまたいだ private ヘルパーの共有はしない方針）。
fn ensure_alloc_fits_f32(input_shape: &[usize], out_shape: &[usize]) -> Result<(), AutodiffError> {
    checked_bytes_for::<f32>(input_shape)?;
    checked_bytes_for::<f32>(out_shape)?;
    Ok(())
}

/// `dim` に沿った縮約最大値・タイの均等分配 VJP 版（`torch.amax(dim)`
/// 相当。イシュー #2154）。`dim: None` は全軸縮約（スカラー）。forward
/// は [`Var::max`] と同一経路（`self.tape.ops().max`）で bit 同一の値を
/// 返す。差分は VJP のみ——`crate::tape::Op::Amax`（クレート非公開の
/// ためリンクにはしない）として登録し、`grad::extremum_even_split_vjp`
/// による均等分配勾配を使う。
///
/// 空縮約（`Var::max` と同じ挙動）は `ops.max` が返すエラーをそのまま
/// 伝播する（新しい規約を作らない）。
pub fn amax<'t>(x: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let out_shape = reduce_out_shape(&shape, dim)?;
    ensure_alloc_fits_f32(&shape, &out_shape)?;
    let input_val = {
        let nodes = x.tape().nodes.borrow();
        materialize_fallible(&nodes, x.tape().ops(), x.node_id())?.clone()
    };
    let value = x.tape().ops().max(&input_val, dim)?;
    let id = x.tape().push_eager(
        Op::Amax {
            input: x.node_id(),
            dim,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}

/// `dim` に沿った縮約最小値・タイの均等分配 VJP 版（`torch.amin(dim)`
/// 相当。イシュー #2154）。[`amax`] と対称。forward は [`Var::min`]
/// と同一経路（`crate::grad::min_with_fallback`。`BackendOps::min` →
/// `Unsupported` のときのみホスト参照実装へフォールバック）で bit
/// 同一の値を返す。
///
/// 空縮約は [`Var::min`] と同じく [`AutodiffError::InvalidArgument`]
/// （`min` は単位元を持たないため。`min_with_fallback` doc 参照）。
pub fn amin<'t>(x: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let out_shape = reduce_out_shape(&shape, dim)?;
    ensure_alloc_fits_f32(&shape, &out_shape)?;
    let input_val = {
        let nodes = x.tape().nodes.borrow();
        materialize_fallible(&nodes, x.tape().ops(), x.node_id())?.clone()
    };
    let value = min_with_fallback(x.tape().ops(), &input_val, dim, &out_shape)?;
    let id = x.tape().push_eager(
        Op::Amin {
            input: x.node_id(),
            dim,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;
    use fandhe_ai_tensor_core::Tensor;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    #[test]
    fn amax_matches_max_forward() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 5.0, 3.0, 5.0], &[4]));
        let a = amax(&x, None).unwrap();
        let b = x.max(None).unwrap();
        assert_eq!(
            a.to_tensor().host_slice()[0].to_bits(),
            b.to_tensor().host_slice()[0].to_bits()
        );
    }

    #[test]
    fn amin_matches_min_forward() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, -5.0, 3.0, -5.0], &[4]));
        let a = amin(&x, None).unwrap();
        let b = x.min(None).unwrap();
        assert_eq!(
            a.to_tensor().host_slice()[0].to_bits(),
            b.to_tensor().host_slice()[0].to_bits()
        );
    }

    #[test]
    fn amax_gradient_splits_evenly_across_ties() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 5.0, 3.0, 5.0], &[4]));
        let y = amax(&x, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.0, 0.5, 0.0, 0.5]);
    }

    #[test]
    fn amin_gradient_splits_evenly_across_ties() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, -5.0, 3.0, -5.0], &[4]));
        let y = amin(&x, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.0, 0.5, 0.0, 0.5]);
    }

    #[test]
    fn amax_gradient_matches_first_match_when_no_tie() {
        // タイが無いときは `Var::max`（先勝ち）と `amax`（均等分配）の
        // 勾配が一致する（`k=1` なら `g/1 == g`）。
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 9.0, 3.0], &[3]));
        let y = amax(&x, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn max_gradient_unchanged_first_match_with_ties() {
        // 回帰: 既存 `Var::max`（先勝ち）の勾配は変わらない。
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 5.0, 3.0, 5.0], &[4]));
        let y = x.max(None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn amax_dim_axis_splits_per_lane() {
        let tape = Tape::new();
        // [[1,5],[5,5]] -> amax(dim=1) = [5, 5]（各行のタイ数は 1・2）
        let x = tape.var(&t(vec![1.0, 5.0, 5.0, 5.0], &[2, 2]));
        let y = amax(&x, Some(1)).unwrap();
        assert_eq!(y.to_tensor().host_slice().into_owned(), vec![5.0, 5.0]);
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.0, 1.0, 0.5, 0.5]);
    }

    #[test]
    fn amax_upstream_gradient_is_distributed_not_just_one() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![2.0, 2.0, 2.0, 2.0], &[4]));
        let y = amax(&x, None).unwrap();
        let four = tape.var(&Tensor::scalar(4.0f32));
        let z = y.mul(&four).unwrap();
        let grads = tape.backward(&z).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        // 上流勾配 4.0 が k=4 個の要素へ均等分配される -> 各 1.0。
        assert_eq!(dx, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn amax_out_of_range_dim_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(amax(&x, Some(5)).is_err());
        assert!(amin(&x, Some(5)).is_err());
    }

    #[test]
    fn amin_empty_reduction_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0]));
        assert!(matches!(
            amin(&x, None),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
}
