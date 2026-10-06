//! `local_response_norm` の自由関数（`F.local_response_norm` 相当。入力 `[N, C, *S]`・チャネル軸は
//! dim 1。イシュー #2646・親 #2625「Phase 4」・ルート #2499）。
//!
//! **facade 非公開（保留）**: 公開形（`Var::local_response_norm` の委譲メソッド）は未承認で、承認依頼は
//! #2677（公開自体は承認後の #2678）。層化（`nn::LocalResponseNorm`・`Module` impl・
//! `Sequential::add_*`）は #2679 の対象で本イシューでは作らない。本モジュールは内部クレート限定の
//! 入口で、`Var` に inherent メソッドを足さない。保留は `crates/facade/src/lib.rs` の
//! `LrnWeightReparamHoldDoctestGuard` と `crates/facade/tests/api_surface.rs` の否定ガードが機械的に
//! 固定する（`docs/autodiff-lrn-weight-reparam-decision.md`）。
//!
//! **経路**: ① パラメータ（`size >= 1`・`alpha`／`beta`／`k` 有限）・rank・確保サイズの検査（実体化・
//! tape 操作より前。エラー時に孤児ノードを残さない）→ ② 入力の実体化 → ③ `BackendOps::lrn_forward`
//! （`Unsupported` のときだけ共有ホストカーネル
//! `fandhe_ai_tensor_core::lrn::local_response_norm_host` へフォールバックし、他のエラーは伝播する。
//! 戻り shape も検証する）→ ④ 専用 `Op`（`Op::LocalResponseNorm`）を積む。VJP は `grad.rs`。
//!
//! **既存 Op の合成にしない理由**: `x.mul(x)` → プールの合成では二乗が `f32` で先に確定し、正規化統計の
//! 「先に `f64` へ昇格してから二乗」契約（`.claude/rules/coding-rust.md`）に抵触するため、専用 `Op` と
//! 共有ホストカーネルとした。
//!
//! **数値契約**: 窓内二乗和・`d^β`・除算は `f64`、最後に 1 回だけ `f32` へ downcast（規則の正は
//! `fandhe_ai_tensor_core::lrn`）。非有限入力は拒否せず伝播する。高階微分（`create_graph`）・
//! activation checkpoint・f64 自動微分経路は対象外。
//!
//! **PyTorch との意図的な差分**: 非有限の `alpha`／`beta`／`k` は拒否する（PyTorch は無検査）。
//! `size == 0` は拒否する。詳細は決定記録 §5。

use fandhe_ai_tensor_core::lrn::{self, LrnParams};
use fandhe_ai_tensor_core::{BackendError, ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

fn materialize_one<'t>(x: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = x.tape().nodes.borrow();
    let ops = x.tape().ops();
    Ok(materialize_fallible(&nodes, ops, x.node_id())?.clone())
}

/// LocalResponseNorm（`F.local_response_norm(x, size, alpha, beta, k)` 相当）。
///
/// `input`: `[N, C, *S]`（rank 3 以上）。窓は `[c − ⌊size/2⌋, c + ⌊(size−1)/2⌋] ∩ [0, C)`・除数は
/// 常に `size`・偶数 `size` は前後非対称（PyTorch と同じ）。検査順: `LrnParams::new`（`size == 0`・
/// 非有限パラメータ）→ `lrn_layout`（rank・確保サイズ）→ 実体化 → バックエンド／ホスト →
/// 戻り shape 再検証 → `push_eager`。
pub fn local_response_norm<'t>(
    input: &Var<'t>,
    size: usize,
    alpha: f32,
    beta: f32,
    k: f32,
) -> Result<Var<'t>, AutodiffError> {
    let params = LrnParams::new(size, alpha, beta, k).map_err(|e| match e {
        BackendError::InvalidArgument(msg) => AutodiffError::InvalidArgument(msg),
        other => AutodiffError::Backend(other),
    })?;
    let in_shape = input.shape();
    let layout = lrn::lrn_layout(&in_shape).map_err(AutodiffError::Shape)?;
    let x = materialize_one(input)?;
    let value = match input.tape().ops().lrn_forward(&x, &params) {
        Ok(v) => {
            if v.shape() != layout.shape() {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: layout.shape().to_vec(),
                    },
                )));
            }
            v
        }
        Err(BackendError::Unsupported(_)) => {
            let data =
                lrn::local_response_norm_host(&x.contiguous().host_slice(), &layout, &params)
                    .map_err(AutodiffError::Shape)?;
            Tensor::new(data, layout.shape()).map_err(AutodiffError::Shape)?
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    let id = input.tape().push_eager(
        Op::LocalResponseNorm {
            input: input.node_id(),
            params,
        },
        value,
    );
    Ok(Var::from_raw(input.tape(), id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    #[test]
    fn matches_hand_computed_value() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3, 1]));
        // size=3・alpha=3・beta=1・k=0 → d_c = S_c（c=0: 5, c=1: 14, c=2: 13）。
        let y = local_response_norm(&x, 3, 3.0, 1.0, 0.0).unwrap();
        let out = y.to_tensor().host_slice().into_owned();
        let want = [1.0 / 5.0, 2.0 / 14.0, 3.0 / 13.0];
        for (a, b) in out.iter().zip(want) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn invalid_arguments_leave_no_orphan_nodes() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0; 6], &[1, 3, 2]));
        let flat = tape.var(&t(vec![0.0; 6], &[2, 3]));
        let before = tape.len();
        assert!(matches!(
            local_response_norm(&x, 0, 1.0, 1.0, 1.0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            local_response_norm(&x, 2, f32::NAN, 1.0, 1.0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            local_response_norm(&flat, 2, 1.0, 1.0, 1.0),
            Err(AutodiffError::Shape(_))
        ));
        assert_eq!(tape.len(), before);
    }
}
