//! `Op::IndexedScatterReduce`（`Var::scatter_reduce`）の `create_graph` 対応
//! （イシュー #3802・親 #3719 系・追跡表 #3717 項目 4。設計
//! `docs/autodiff-higher-order-grad-decision.md` §41.10、実装記録は §61）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `validate_ancestors`（入口検査）・
//! `replay_op`（forward 再生）・`build_cgrads`（1 階 VJP の子テープ記録）から呼ばれる。
//! 対象 mode は `Sum`・`Mean`・`Amax`・`Amin`。`Prod` は §41.11 が定める区分つき scan 補助
//! （#3803 ①）の上に載る設計のため本イシューでは実装せず、fail-closed のままにする
//! （`Op::supports_create_graph` が false。実装記録 §61 に理由を残す）。
//!
//! **方式**（値は 1 階 `grad.rs` の VJP と同じホスト関数 `scatter_reduce_vjp_host` の結果で、
//! `push_v3` で載せる。構造は二階微分を流すためだけの線形な `Var` 合成）:
//! - `d_src` の構造は `Gather(g, dim, index)`（`index` の非 dim 軸が input より小さいときは
//!   gather の前に `g` を narrow する。`index_cum::scatter_vjp` と同じ）。
//! - `Sum`: 構造は恒等。`d_input` は `include_self` なら `g`、そうでなければ触れられた位置を 0。
//! - `Mean`: 位置ごとの寄与数 `m = cnt + include_self` から作る定数 `1/m`（`<= 1`）を掛ける。
//! - `Amax`／`Amin`: 選択マスク（`src[p] == result[q]`。1 階 VJP と同じ f32 `==`）と
//!   `1/max(ties, 1)` を定数にし、非選択位置は `where_cond` で 0 にする（`inf * 0` を作らない）。
//!   タイは均等分配の劣勾配、NaN の結果は ties = 0 で勾配 0。
//! - 触れられていない位置の `d_input` は全 mode 共通で `g` を素通しする。
//!
//! 子テープへ積む Op は `Narrow`・`Gather`・`Mul`・`Where`・`Contiguous`・定数 `Leaf` で、すべて
//! `supports_create_graph() == true`（閉包性。§21.1(4)）。`Mean` の `g / m` を f32 の `Var` 値の
//! まま使わない理由は、寄与数が 2^24 を超えると f64 割り算の 1 階値と食い違うため（値は V3）。
//!
//! **総括判定**（§41.12）: 構造の各段は `|g|` 以下（定数は 0/1 か `1/n`）で押さえられるため、
//! 上流 `g` が有限かつ `STAGE_BOUND_GUARD` 未満であることだけを、子テープへ書く前に課す。
//! 入力・`src` の値に `±inf` を含むものは [`validate`] が拒否する（NaN は伝播。§33.14）。
//! 外れる入力は型付きエラーで、親の rollback により `child` は空へ戻る。
//!
//! GPU: `indexed_scatter_reduce` は CPU のみ実装で CUDA／Metal ではホストへフォールバックする
//! （経路区分 `H`）。実機 parity は未実測（Phase 10 の #3857・#3859）。

use fandhe_ai_tensor_core::indexed_update::{
    scatter_reduce_host, scatter_reduce_layout, scatter_reduce_vjp_host,
};
use fandhe_ai_tensor_core::{BackendOps, ScatterReduceMode, Tensor};

use super::{child_value, push_v3};
use crate::create_graph_norm::STAGE_BOUND_GUARD;
use crate::error::AutodiffError;
use crate::eval::{dense_vec, dense_vec_i32};
use crate::tape::{NodeId, Op, Tape, TapeNode, materialize_fallible};
use crate::var::Var;

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!("create_graph: {msg}"))
}

/// create_graph が受理する mode か（`Prod` と `#[non_exhaustive]` の未知 mode は false）。
pub(crate) fn mode_supported(mode: ScatterReduceMode) -> bool {
    matches!(
        mode,
        ScatterReduceMode::Sum
            | ScatterReduceMode::Mean
            | ScatterReduceMode::Amax
            | ScatterReduceMode::Amin
    )
}

/// 入口検査（`validate_ancestors` から、子テープへ書く前に呼ばれる）。
///
/// 未知 mode と `±inf` を含む入力・`src` を拒否する（NaN は伝播させる）。
pub(super) fn validate(
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    id: NodeId,
    op: &Op,
) -> Result<(), AutodiffError> {
    let Op::IndexedScatterReduce {
        input, src, mode, ..
    } = op
    else {
        return Ok(());
    };
    if !mode_supported(*mode) {
        return Err(invalid(format!(
            "IndexedScatterReduce の mode {mode:?}（NodeId({}))は非対応（Prod は #3803 の scan 補助待ち。\
             docs/autodiff-higher-order-grad-decision.md §41・§61 参照）",
            id.0
        )));
    }
    for operand in [*input, *src] {
        let value = materialize_fallible(parent_nodes, parent_ops, operand)?;
        if dense_vec(value).iter().any(|v| v.is_infinite()) {
            return Err(invalid(format!(
                "IndexedScatterReduce の入力に ±inf がある（NodeId({}))ため拒否（\
                 §33.14・§41.12 参照。NaN は伝播させる）",
                id.0
            )));
        }
    }
    Ok(())
}

/// forward の写し（`replay_op` から）。forward の検証（レイアウト・index 値域）を
/// 迂回しないよう公開の `scatter_reduce` へ委譲する。
pub(super) fn replay<'c>(
    input: &Var<'c>,
    src: &Var<'c>,
    dim: usize,
    index: &Tensor<i32>,
    mode: ScatterReduceMode,
    include_self: bool,
) -> Result<Var<'c>, AutodiffError> {
    if !mode_supported(mode) {
        return Err(invalid(format!(
            "IndexedScatterReduce の未知または非対応の mode {mode:?}"
        )));
    }
    crate::indexed_update_ops::scatter_reduce(input, dim, index, src, mode, include_self)
}

/// `IndexedScatterReduce` ノードの静的な形（親テープの shape と `Op` payload）。
pub(super) struct Spec<'a> {
    pub(super) input: NodeId,
    pub(super) src: NodeId,
    pub(super) dim: usize,
    pub(super) index: &'a Tensor<i32>,
    pub(super) mode: ScatterReduceMode,
    pub(super) include_self: bool,
}

/// `(d_input, d_src)`。必要な側だけ構築する。
pub(super) struct Grads<'c> {
    pub(super) d_input: Option<Var<'c>>,
    pub(super) d_src: Option<Var<'c>>,
}

/// ホストで作る構造用の定数群（1 階 VJP と同じ位置規則・f32 `==` タイ規則）。
struct Masks {
    /// 入力位置ごとに触れられたか。
    touched: Vec<bool>,
    /// `d_src` の構造に掛ける係数（Sum は 1、Mean は `1/m`、Amax／Amin は `1/max(ties, 1)`）。
    src_scale: Vec<f32>,
    /// `d_src` を通す位置（Amax／Amin のみ選択マスク。それ以外は全 true）。
    src_keep: Vec<bool>,
    /// `d_input` の構造に掛ける係数（未タッチは 1）。
    in_scale: Vec<f32>,
    /// `d_input` を通す位置（未タッチは true。触れられた位置は mode と `include_self` で決まる）。
    in_keep: Vec<bool>,
}

fn build_masks(
    pos: &[usize],
    x: &[f32],
    s: &[f32],
    result: Option<&[f32]>,
    mode: ScatterReduceMode,
    include_self: bool,
) -> Result<Masks, AutodiffError> {
    let n = x.len();
    let mut cnt = vec![0usize; n];
    for &q in pos {
        cnt[q] += 1;
    }
    let touched: Vec<bool> = cnt.iter().map(|&c| c > 0).collect();
    let mut src_scale = vec![1.0f32; pos.len()];
    let mut src_keep = vec![true; pos.len()];
    let mut in_scale = vec![1.0f32; n];
    let mut in_keep = vec![true; n];
    match mode {
        ScatterReduceMode::Sum => {
            for q in 0..n {
                in_keep[q] = !touched[q] || include_self;
            }
        }
        ScatterReduceMode::Mean => {
            for (p, &q) in pos.iter().enumerate() {
                src_scale[p] = 1.0 / (cnt[q] + usize::from(include_self)) as f32;
            }
            for q in 0..n {
                if touched[q] {
                    in_keep[q] = include_self;
                    in_scale[q] = 1.0 / (cnt[q] + 1) as f32;
                }
            }
        }
        ScatterReduceMode::Amax | ScatterReduceMode::Amin => {
            let result = result
                .ok_or_else(|| invalid("Amax／Amin の結果が未計算（内部契約違反）".to_string()))?;
            // 1 階 VJP（`scatter_reduce_vjp_host`）と同じ規則: 結果と f32 `==` な寄与の個数。
            let mut ties = vec![0usize; n];
            for (&q, &v) in pos.iter().zip(s) {
                if v == result[q] {
                    ties[q] += 1;
                }
            }
            if include_self {
                for q in 0..n {
                    if touched[q] && x[q] == result[q] {
                        ties[q] += 1;
                    }
                }
            }
            for (p, &q) in pos.iter().enumerate() {
                src_keep[p] = s[p] == result[q] && ties[q] > 0;
                src_scale[p] = 1.0 / ties[q].max(1) as f32;
            }
            for q in 0..n {
                if touched[q] {
                    in_keep[q] = include_self && x[q] == result[q] && ties[q] > 0;
                    in_scale[q] = 1.0 / ties[q].max(1) as f32;
                }
            }
        }
        _ => {
            return Err(invalid(format!(
                "IndexedScatterReduce の未知の mode {mode:?}"
            )));
        }
    }
    Ok(Masks {
        touched,
        src_scale,
        src_keep,
        in_scale,
        in_keep,
    })
}

/// 上流 `g` の総括判定（モジュール doc）。非有限、または `max|g| >= STAGE_BOUND_GUARD` を拒否する。
fn check_upstream_bound(g_value: &Tensor<f32>) -> Result<(), AutodiffError> {
    for &v in &dense_vec(g_value) {
        if !v.is_finite() {
            return Err(invalid(
                "IndexedScatterReduce の上流勾配が非有限のため拒否（§41.12 参照）".into(),
            ));
        }
        if f64::from(v).abs() >= STAGE_BOUND_GUARD {
            return Err(invalid(
                "IndexedScatterReduce の上流勾配が大きすぎるため拒否（§41.12 参照）".into(),
            ));
        }
    }
    Ok(())
}

/// `IndexedScatterReduce` の VJP を子テープへ記録する（`build_cgrads` から）。
pub(super) fn cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: &Spec<'_>,
    g: &Var<'c>,
    need_input: bool,
    need_src: bool,
) -> Result<Grads<'c>, AutodiffError> {
    let Spec {
        input,
        src,
        dim,
        index,
        mode,
        include_self,
    } = *spec;
    if !mode_supported(mode) {
        return Err(invalid(format!(
            "IndexedScatterReduce の mode {mode:?} は非対応（内部契約違反。入口検査で拒否済みのはず）"
        )));
    }
    let input_shape = parent_nodes[input.0].shape.clone();
    let src_shape = parent_nodes[src.0].shape.clone();
    if g.shape() != input_shape {
        return Err(invalid(format!(
            "IndexedScatterReduce の上流 shape {:?} が入力 shape {input_shape:?} と不一致（内部契約違反）",
            g.shape()
        )));
    }
    let layout = scatter_reduce_layout(&input_shape, index.shape(), &src_shape, dim)
        .map_err(AutodiffError::Shape)?;
    let zeros_src = child.var_no_grad(&Tensor::zeros(&src_shape)?);
    // 何も書かれない場合（index／src が空）は全位置が未タッチで、1 階は `d_input = g`・`d_src = []`。
    if layout.index_numel() == 0 {
        return Ok(Grads {
            d_input: need_input.then_some(*g),
            d_src: need_src.then_some(zeros_src),
        });
    }
    let g_value = child_value(child, g)?;
    check_upstream_bound(&g_value)?;

    let x_val = materialize_fallible(parent_nodes, parent.ops(), input)?;
    let s_val = materialize_fallible(parent_nodes, parent.ops(), src)?;
    let x = dense_vec(x_val);
    let s = dense_vec(s_val);
    let idx = dense_vec_i32(index);
    let g_host = dense_vec(&g_value);
    let pos = layout.positions(&idx).map_err(AutodiffError::Shape)?;
    let result = match mode {
        ScatterReduceMode::Amax | ScatterReduceMode::Amin => Some(
            scatter_reduce_host(&x, &idx, &s, &layout, mode, include_self)
                .map_err(AutodiffError::Shape)?,
        ),
        _ => None,
    };
    let masks = build_masks(&pos, &x, &s, result.as_deref(), mode, include_self)?;
    let (d_in_host, d_src_host) =
        scatter_reduce_vjp_host(&x, &idx, &s, &g_host, &layout, mode, include_self)
            .map_err(AutodiffError::Shape)?;

    let d_src = if need_src {
        // 非 dim 軸が input より小さい場合は g を先頭から narrow してから gather する。
        let mut gs = *g;
        for (axis, &in_s) in input_shape.iter().enumerate() {
            if axis != dim && src_shape[axis] < in_s {
                gs = gs.narrow(axis, 0, src_shape[axis])?;
            }
        }
        let mut composed = gs.gather(dim, index)?;
        if matches!(
            mode,
            ScatterReduceMode::Mean | ScatterReduceMode::Amax | ScatterReduceMode::Amin
        ) {
            let scale = child.var_no_grad(&Tensor::new(masks.src_scale.clone(), &src_shape)?);
            composed = composed.mul(&scale)?;
        }
        if matches!(mode, ScatterReduceMode::Amax | ScatterReduceMode::Amin) {
            let keep = Tensor::new(masks.src_keep.clone(), &src_shape)?;
            composed = Var::where_cond(&keep, &composed, &zeros_src)?;
        }
        Some(push_v3(
            child,
            &composed,
            Tensor::new(d_src_host, &src_shape)?,
        )?)
    } else {
        None
    };

    let d_input = if need_input {
        let mut composed = *g;
        if matches!(
            mode,
            ScatterReduceMode::Mean | ScatterReduceMode::Amax | ScatterReduceMode::Amin
        ) {
            let scale = child.var_no_grad(&Tensor::new(masks.in_scale.clone(), &input_shape)?);
            composed = composed.mul(&scale)?;
        }
        if masks.in_keep.iter().any(|&k| !k) {
            let keep = Tensor::new(masks.in_keep.clone(), &input_shape)?;
            let zeros = child.var_no_grad(&Tensor::zeros(&input_shape)?);
            composed = Var::where_cond(&keep, &composed, &zeros)?;
        }
        // `touched` は構造の組み立てには使わない（`in_keep`／`in_scale` に畳み込み済み）が、
        // 未タッチ位置が g を素通しすることを debug で固定する。
        debug_assert!(
            masks
                .touched
                .iter()
                .zip(&masks.in_keep)
                .all(|(&t, &k)| t || k)
        );
        Some(push_v3(
            child,
            &composed,
            Tensor::new(d_in_host, &input_shape)?,
        )?)
    } else {
        None
    };
    Ok(Grads { d_input, d_src })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_ops::naive_ops;

    /// 構造側（定数係数と `where_cond`）の値が、ホスト VJP と 4 mode で一致することを固定する。
    /// 1 階の値は `push_v3` がホスト値を載せるため bit 一致だが、孫テープの再生は構造を
    /// 再計算するので、構造値が REQ-2 の範囲でホスト値に一致する前提を単体で押さえる。
    #[test]
    fn masks_agree_with_first_order_vjp() {
        let layout = scatter_reduce_layout(&[4], &[5], &[5], 0).unwrap();
        let idx = [0i32, 0, 2, 2, 2];
        let x = [1.0f32, 5.0, 3.0, 2.0];
        let s = [1.0f32, 4.0, 3.0, 3.0, 0.5];
        let g = [2.0f32, 7.0, 3.0, 11.0];
        let pos = layout.positions(&idx).unwrap();
        for mode in [
            ScatterReduceMode::Sum,
            ScatterReduceMode::Mean,
            ScatterReduceMode::Amax,
            ScatterReduceMode::Amin,
        ] {
            for include_self in [false, true] {
                let result =
                    scatter_reduce_host(&x, &idx, &s, &layout, mode, include_self).unwrap();
                let m = build_masks(&pos, &x, &s, Some(&result), mode, include_self).unwrap();
                let (d_in, d_src) =
                    scatter_reduce_vjp_host(&x, &idx, &s, &g, &layout, mode, include_self).unwrap();
                for (p, &q) in pos.iter().enumerate() {
                    let want = if m.src_keep[p] {
                        g[q] * m.src_scale[p]
                    } else {
                        0.0
                    };
                    assert!(
                        (want - d_src[p]).abs() <= 1e-6 * d_src[p].abs().max(1.0),
                        "{mode:?} include_self={include_self} d_src[{p}] {want} vs {}",
                        d_src[p]
                    );
                }
                for q in 0..x.len() {
                    let want = if m.in_keep[q] {
                        g[q] * m.in_scale[q]
                    } else {
                        0.0
                    };
                    assert!(
                        (want - d_in[q]).abs() <= 1e-6 * d_in[q].abs().max(1.0),
                        "{mode:?} include_self={include_self} d_in[{q}] {want} vs {}",
                        d_in[q]
                    );
                }
            }
        }
    }

    /// 既知 mode の写しが forward と同じ値を返す。
    #[test]
    fn replay_matches_forward() {
        let tape = Tape::new_with_ops(naive_ops());
        let x = tape.var(&Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap());
        let s = tape.var(&Tensor::new(vec![4.0f32, 5.0], &[2]).unwrap());
        let idx = Tensor::<i32>::new(vec![0, 0], &[2]).unwrap();
        let y = replay(&x, &s, 0, &idx, ScatterReduceMode::Sum, true).unwrap();
        assert_eq!(
            y.to_tensor().host_slice().into_owned(),
            vec![10.0, 2.0, 3.0]
        );
        assert!(replay(&x, &s, 0, &idx, ScatterReduceMode::Prod, true).is_err());
    }
}
