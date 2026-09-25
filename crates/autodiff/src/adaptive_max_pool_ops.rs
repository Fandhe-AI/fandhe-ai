//! AdaptiveMaxPool（1d／2d。PyTorch `nn.AdaptiveMaxPool2d`／`1d`
//! 相当。イシュー #2160・設計 `docs/pooling-ops-design.md` §11）。
//!
//! **facade 非公開（意図的・保留）**: [`crate::conv3d_ops`] モジュール
//! doc と同じ理由・同じ判断枠組みによる。`Var` は facade
//! （`fandhe_ai` クレート）から直接再エクスポートされるため、`Var`
//! への inherent メソッド追加は即座に facade 公開面へ出てしまう。
//! イシュー #2160 は `Var::adaptive_max_pool2d`（委譲メソッド）・
//! `compat::Sequential::add_adaptive_max_pool2d`（facade 公開）を
//! 承認事項として明示するが、本 PR 時点で承認の記録が無いため、
//! `conv3d_ops` よりもさらに一段保守的に、`Var` の外の自由関数
//! （非 `pub mod`・`pub(crate) fn`）として実装し、`nn::pooling` の
//! 層のみから呼ばれる経路に限定する（承認前は `Var` への衝突
//! プローブすら成立させない）。承認後は `Var::adaptive_max_pool2d`
//! の薄い委譲メソッドと `compat::Sequential::add_adaptive_max_pool2d`
//! を追加し、facade 側の保留ガード（`crates/facade/src/lib.rs::
//! AdaptiveMaxGlobalPoolHoldDoctestGuard`）を撤去する。
//!
//! **既存 `Op` の再利用**: 新規 `Op` は追加しない。forward は
//! `crate::tape::Op::MaxPool2d { input, index }` をそのまま記録する
//! （`Op::MaxPool2d` の doc comment 参照。VJP が `params` に非依存
//! なため adaptive max pooling の VJP も数学的に同一）。
//!
//! **数値契約**: 純粋な選択演算（丸めなし）のため 3 バックエンド間
//! （GPU 側はホストフォールバック経由）で値・索引とも bit 完全一致が
//! 受入基準（`Var::max_pool2d` と同一の REQ-2 判定）。

use fandhe_ai_tensor_core::{ShapeError, Tensor, adaptive_pool2d_out_shape};

use crate::error::AutodiffError;
use crate::grad::adaptive_max_pool2d_with_fallback;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

/// 索引の表現可能範囲検査（`H·W <= i32::MAX`。索引は `i32` のため）。
/// [`adaptive_max_pool2d`]／[`adaptive_max_pool1d`] の tape 経路
/// （`Var::forward`）だけでなく、[`crate::nn::module::Module::forward_host`]
/// の host 経路（`AdaptiveMaxPool2d`／`AdaptiveMaxPool1d`／
/// `GlobalPool(Max)`）からも呼ばれる共通ヘルパー（codex-review・Cursor
/// Bugbot 指摘・イシュー #2160）。host 経路がこの検査を欠くと、極端
/// 形状（例: 空バッチかつ `H·W` が `i32::MAX` 超）で tape 経路は
/// `IndexRangeOverflow` を返す一方 host 経路は成功してしまい、両経路の
/// 契約が食い違う。
pub(crate) fn check_max_index_range(h: usize, w: usize) -> Result<(), AutodiffError> {
    let hw = h
        .checked_mul(w)
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
    if hw > i32::MAX as usize {
        return Err(AutodiffError::Shape(ShapeError::IndexRangeOverflow {
            index: hw,
        }));
    }
    Ok(())
}

/// 2 次元 adaptive max pooling（`torch.nn.AdaptiveMaxPool2d` 相当。
/// NCHW 固定。イシュー #2160）。`input`: `[N, C, H, W]`・
/// `output_size: [Hout, Wout]`。戻り値は `(values, index)` で
/// `index`（`(n,c)` 平面内 flat 添字 `h·W+w`）は [`Var::max_pool2d`]
/// と同じ意味論。
///
/// 検査順序（[`Var::max_pool2d`]／[`Var::adaptive_avg_pool2d`] と
/// 同型）: ①[`adaptive_pool2d_out_shape`]（rank・空間軸ゼロ拒否・
/// `output_size >= 1`）→ ②`H·W <= i32::MAX`（索引は `i32` のため。
/// `Var::max_pool2d` にある Max 固有の追加検査を踏襲）→ ③`input` を
/// 層 1 で実体化（`RefCell` 借用を閉じてから push）→ ④
/// `adaptive_max_pool2d_with_fallback` → ⑤戻り shape 再検証
/// （`.claude/rules/security.md` A08）→ ⑥`push_eager`
/// （`Op::MaxPool2d` を再利用。非融合・常実体化）。
pub(crate) fn adaptive_max_pool2d<'t>(
    input: &Var<'t>,
    output_size: [usize; 2],
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    let in_shape = input.shape();
    let out_shape =
        adaptive_pool2d_out_shape(&in_shape, output_size).map_err(AutodiffError::Shape)?;

    check_max_index_range(
        in_shape.get(2).copied().unwrap_or(0),
        in_shape.get(3).copied().unwrap_or(0),
    )?;

    let input_val = {
        let nodes = input.tape().nodes.borrow();
        materialize_fallible(&nodes, input.tape().ops(), input.node_id())?.clone()
    };

    let (value, index) =
        adaptive_max_pool2d_with_fallback(input.tape().ops(), &input_val, output_size, &out_shape)?;
    if value.shape() != out_shape.as_slice() || index.shape() != out_shape.as_slice() {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: value.shape().to_vec(),
            rhs: out_shape,
        }));
    }
    let id = input.tape().push_eager(
        Op::MaxPool2d {
            input: input.node_id(),
            index: index.clone(),
        },
        value,
    );
    Ok((Var::from_raw(input.tape(), id), index))
}

/// 1 次元 adaptive max pooling。[`adaptive_max_pool2d`] を `H` 軸固定
/// （`output_size[0]=1`）で呼び出す reshape 併合の薄いラッパー
/// （[`Var::adaptive_avg_pool1d`] と同型。イシュー #2160）。`input`:
/// `[N, C, L]`。索引は `H=1` のため flat `w` そのもの。
///
/// 検査は reshape より前に完了させる（`Var::adaptive_avg_pool1d` と
/// 同じ規律。reshape 後にエラーを返すと孤立した view ノードがテープに
/// 残るため）。Max 系は索引が `i32` のため `adaptive_pool2d_out_shape`
/// の形状検査に加え `l <= i32::MAX` も reshape 前に検査する
/// （`adaptive_max_pool2d` の `hw <= i32::MAX` 検査と同型。`h=1` 固定
/// のため `hw == l`。要素数ゼロでも `l` 自体が `i32::MAX` を超えれば
/// 索引が表現不能なため、形状検査を通過しても reshape 前に弾く）。
/// `adaptive_max_pool2d` 側の再検査はフェイルクローズドの二重化として
/// 残す。
pub(crate) fn adaptive_max_pool1d<'t>(
    input: &Var<'t>,
    output_size: usize,
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    let in_shape = input.shape();
    if in_shape.len() != 3 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 3,
            actual: in_shape.len(),
        }));
    }
    let (n, c, l) = (in_shape[0], in_shape[1], in_shape[2]);
    adaptive_pool2d_out_shape(&[n, c, 1, l], [1, output_size]).map_err(AutodiffError::Shape)?;
    check_max_index_range(1, l)?;

    let x4 = input.contiguous()?.reshape(&[n, c, 1, l])?;
    let (out4, index) = adaptive_max_pool2d(&x4, [1, output_size])?;
    let out_shape4 = out4.shape();
    let lout = out_shape4[3];
    let out = out4.reshape(&[n, c, lout])?;
    let index = index.reshape(&[n, c, lout]).map_err(AutodiffError::Shape)?;
    Ok((out, index))
}
