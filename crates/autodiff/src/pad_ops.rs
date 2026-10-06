//! `pad` の非定数モード（reflect／replicate／circular）の自由関数
//! （イシュー #2642・親 #2625「Phase 4」・ルート #2499）。
//!
//! **facade 非公開（保留）**: 公開形（`Var::pad_with_mode` の委譲メソッドと
//! `PadMode` の再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の
//! #2678・#2679）。本モジュールは内部クレート限定の入口で、`Var` に inherent
//! メソッドを足さない。保留は `crates/facade/src/lib.rs` の
//! `PadModesHoldDoctestGuard` と `crates/facade/tests/api_surface.rs` の否定ガードが
//! 機械的に固定する（`docs/autodiff-pad-modes-decision.md`）。
//!
//! **PyTorch 相当**: [`pad_with_mode`] は `torch.nn.functional.pad(x, pad, mode)` の
//! `mode = "reflect" | "replicate" | "circular"` に相当する。`pads` は既存 `Var::pad`
//! と同じく先頭軸から順の `(before, after)` を rank 個（PyTorch の末尾軸からの平坦
//! リストとは異なる既存設計を踏襲）。任意軸・任意 rank を受ける上位集合で、負の
//! パディング（クロップ）は `usize` のため表現できない。定数埋めは既存 `Var::pad`。
//!
//! **経路**: ① 出力 shape の検査と確保サイズ上限の検査（実体化より前。巨大 pad を
//! 拒否）→ ② `tensor_core::pad_modes::pad_modes_layout`（rank・モード別 pad 上限）→
//! ③ 入力の実体化 → ④ `BackendOps::pad_modes_forward`（`Unsupported` のときだけ共有
//! ホストカーネル `pad_modes_host` へフォールバックし、他のエラーは伝播する。戻り値
//! shape も検証する）→ ⑤ 専用 `Op::PadMode` を積む。VJP は `grad.rs`。エラー時は
//! tape を一切操作しない。
//!
//! **数値契約**: forward は算術を含まない純粋なコピーで bit 完全一致（規則の正は
//! `fandhe_ai_tensor_core::pad_modes`）。backward は添字重複の scatter-add を `f64`
//! アキュムレータで蓄積し 1 回だけ `f32` へ downcast する。高階微分（`create_graph`）・
//! activation checkpoint・f64 自動微分経路は対象外。

use fandhe_ai_tensor_core::pad_modes::{self, PadMode};
use fandhe_ai_tensor_core::{BackendError, ShapeError, Tensor, pad_out_shape};

use crate::error::AutodiffError;
use crate::rearrange_ops::checked_index_alloc_len;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

fn verify_shape(actual: &[usize], expected: &[usize]) -> Result<(), AutodiffError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: actual.to_vec(),
                rhs: expected.to_vec(),
            },
        )))
    }
}

/// 非定数モードの pad（`F.pad(x, pad, mode=...)` 相当）。
///
/// `pads` は先頭軸から順の `(before, after)` を rank 個。モード別の pad 上限
/// （reflect: `before`・`after` < 軸長、circular: `<=` 軸長、replicate: 軸長 1 以上）
/// 違反・rank 不一致・出力サイズ超過は型付きエラー（panic しない）。
pub fn pad_with_mode<'t>(
    x: &Var<'t>,
    pads: &[(usize, usize)],
    mode: PadMode,
) -> Result<Var<'t>, AutodiffError> {
    let in_shape = x.shape();
    // 出力サイズの上限検査を実体化・レイアウト構築より前に行う（巨大 pad の拒否）。
    let out_shape = pad_out_shape(&in_shape, pads).map_err(AutodiffError::Shape)?;
    let out_numel = out_shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
    checked_index_alloc_len(out_numel)?;
    let layout = pad_modes::pad_modes_layout(&in_shape, pads, mode)?;
    let input = {
        let nodes = x.tape().nodes.borrow();
        let ops = x.tape().ops();
        materialize_fallible(&nodes, ops, x.node_id())?.clone()
    };
    let value = match x.tape().ops().pad_modes_forward(&input, pads, mode) {
        Ok(v) => {
            verify_shape(v.shape(), layout.out_shape())?;
            v
        }
        Err(BackendError::Unsupported(_)) => {
            let data = pad_modes::pad_modes_host(&input.contiguous().host_slice(), &layout)?;
            Tensor::new(data, layout.out_shape()).map_err(AutodiffError::Shape)?
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    let id = x.tape().push_eager(
        Op::PadMode {
            input: x.node_id(),
            pads: pads.to_vec(),
            mode,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}
