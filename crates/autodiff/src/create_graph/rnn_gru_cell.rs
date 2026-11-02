//! `RnnCell`・`GruCell` の `create_graph` 対応（イシュー #3795・親 #3719 系・
//! 決定記録 `docs/autodiff-higher-order-grad-decision.md` §37・§58）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `replay_op`（forward 再生）と
//! `build_cgrads`（1 階 VJP の子テープ記録）から呼ばれる。facade の `hvp`・`jvp`・
//! `jacfwd`・`jvp_create_graph`・`backward_create_graph` が RNN／GRU を含むモデルを
//! 拒否しないための前提となる（facade の公開 RNN API はどれも fused Op を積む）。
//!
//! **方式**（1 階 `grad.rs` の `Op::RnnCell`／`Op::GruCell` 腕と同じオペランド順・同じ
//! 掛ける順で組む）:
//! - **V1**: 1 階 VJP の式を子テープ上の `Var` 演算（`mul`・`sub`・`sigmoid`・`tanh`・
//!   `narrow`・`cat`・`matmul_fp32_strict`・`transpose`）の合成として記録する。GEMM は
//!   forward の `gemm_bias_act`・1 階 `affine_vjp` と同じく常に FP32 厳密
//!   （`matmul_fp32_strict`。CUDA の TF32 opt-in が有効でも二階の記録経路で破らない）。
//! - **V3（bias のみ）**: `db_ih`・`db_hh` は 1 階 `affine_vjp` と同じ f64 縮約ホスト関数
//!   `eval::reduce_bias_grad_rows` を [`super::reduce_bias_grad_var`] 経由で使う
//!   （`.claude/rules/coding-rust.md` の勾配長軸縮約契約）。
//! - `RnnCell` は自ノードの出力 `h_t` の写し（`tanh` の出力）から `1 − h²` を作る。
//! - `GruCell` は payload（`gates_rzn`・`q`）が x・h_prev・W の関数であり定数として
//!   使うと二階が落ちるため、子テープ上で `pre_i`・`pre_h` からゲート `r,z,n`・`q` を
//!   作り直す（reset-after 規約。§37.10 の例外）。GRU の `affine_vjp` は 1 階でも全幅
//!   `[B,3H]` で呼ばれるため `Pad`・列埋め込みは不要。
//!
//! **閉包性**: 子テープへ積む Op は `supports_create_graph() == true` な Op
//! （`MatMul`・`Transpose`・`Narrow`・`Concat`・`Add`・`Sub`〈`ScalarBinary`〉・`Mul`・
//! `Sigmoid`・`Tanh`・`Sum`）に限る。
//!
//! **数値契約**: GRU の再構築は forward の `mul_add` と丸めがずれうるため、cgrad の値は
//! 1 階と REQ-2 統一複合判定で一致する（bit 一致は主張しない。`first_order()` 側は素の
//! `Tape::backward` と bit 同一）。tolerance・baseline は変えない。

use super::{accumulate, get_mirror, reduce_bias_grad_var};
use crate::error::AutodiffError;
use crate::tape::{NodeId, Tape, TapeNode};
use crate::var::{GateParams, Var};
use fandhe_ai_tensor_core::Tensor;

/// セル Op の入力ノード束（`Op::RnnCell`／`Op::GruCell` の共通フィールド）。
#[derive(Clone, Copy)]
pub(super) struct CellIds {
    pub(super) x: NodeId,
    pub(super) h_prev: NodeId,
    pub(super) w_ih: NodeId,
    pub(super) w_hh: NodeId,
    pub(super) b_ih: Option<NodeId>,
    pub(super) b_hh: Option<NodeId>,
}

/// 子テープ上の写しをまとめて取り出したもの。
struct Mirrors<'c> {
    x: Var<'c>,
    h_prev: Var<'c>,
    w_ih: Var<'c>,
    w_hh: Var<'c>,
    b_ih: Option<Var<'c>>,
    b_hh: Option<Var<'c>>,
}

fn mirrors<'c>(mirror: &[Option<Var<'c>>], ids: &CellIds) -> Result<Mirrors<'c>, AutodiffError> {
    Ok(Mirrors {
        x: get_mirror(mirror, ids.x)?,
        h_prev: get_mirror(mirror, ids.h_prev)?,
        w_ih: get_mirror(mirror, ids.w_ih)?,
        w_hh: get_mirror(mirror, ids.w_hh)?,
        b_ih: ids.b_ih.map(|b| get_mirror(mirror, b)).transpose()?,
        b_hh: ids.b_hh.map(|b| get_mirror(mirror, b)).transpose()?,
    })
}

/// `RnnCell` の forward 再生。子テープでも同じ `rnn_cell_forward_value` を通る。
pub(super) fn replay_rnn_cell<'c>(
    mirror: &[Option<Var<'c>>],
    ids: &CellIds,
) -> Result<Var<'c>, AutodiffError> {
    let m = mirrors(mirror, ids)?;
    m.x.rnn_cell(
        &m.h_prev,
        GateParams {
            w_ih: &m.w_ih,
            w_hh: &m.w_hh,
            b_ih: m.b_ih.as_ref(),
            b_hh: m.b_hh.as_ref(),
        },
    )
}

/// `GruCell` の forward 再生（payload は使わず子テープ側が自分で計算し直す）。
pub(super) fn replay_gru_cell<'c>(
    mirror: &[Option<Var<'c>>],
    ids: &CellIds,
) -> Result<Var<'c>, AutodiffError> {
    let m = mirrors(mirror, ids)?;
    m.x.gru_cell(
        &m.h_prev,
        GateParams {
            w_ih: &m.w_ih,
            w_hh: &m.w_hh,
            b_ih: m.b_ih.as_ref(),
            b_hh: m.b_hh.as_ref(),
        },
    )
}

fn ones_like<'c>(child: &'c Tape, shape: &[usize]) -> Result<Var<'c>, AutodiffError> {
    Ok(child.var_no_grad(&Tensor::full(shape, 1.0f32)?))
}

/// `RnnCell` の VJP を子テープへ記録し、入力ごとの寄与を 1 階 `contributions` と同じ順
/// （x、h_prev、w_ih、w_hh、b_ih、b_hh）で `accumulate` する。
pub(super) fn rnn_cell_cgrad<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    cgrads: &mut [Option<Var<'c>>],
    mirror: &[Option<Var<'c>>],
    id: NodeId,
    g: &Var<'c>,
    ids: &CellIds,
) -> Result<(), AutodiffError> {
    let m = mirrors(mirror, ids)?;
    // 自ノードの写し（= tanh の出力 h_t）。`Op::Tanh` の腕と同じく再計算しない。
    let h_t = get_mirror(mirror, id)?;
    let hidden = h_t.shape().get(1).copied().ok_or_else(|| {
        AutodiffError::Backward("create_graph: RnnCell の出力が rank 2 でない".into())
    })?;
    let one = ones_like(child, &h_t.shape())?;
    let factor = one.sub(&h_t.mul(&h_t)?)?;
    let d_pre = g.mul(&factor)?;

    let dx = d_pre.matmul_fp32_strict(&m.w_ih.transpose(0, 1)?)?;
    let dw_ih = m.x.transpose(0, 1)?.matmul_fp32_strict(&d_pre)?;
    let dh_prev = d_pre.matmul_fp32_strict(&m.w_hh.transpose(0, 1)?)?;
    let dw_hh = m.h_prev.transpose(0, 1)?.matmul_fp32_strict(&d_pre)?;

    accumulate(parent_nodes, cgrads, ids.x, dx)?;
    accumulate(parent_nodes, cgrads, ids.h_prev, dh_prev)?;
    accumulate(parent_nodes, cgrads, ids.w_ih, dw_ih)?;
    accumulate(parent_nodes, cgrads, ids.w_hh, dw_hh)?;
    if let Some(b) = ids.b_ih {
        let db = reduce_bias_grad_var(child, &d_pre, &[hidden])?;
        accumulate(parent_nodes, cgrads, b, db)?;
    }
    if let Some(b) = ids.b_hh {
        let db = reduce_bias_grad_var(child, &d_pre, &[hidden])?;
        accumulate(parent_nodes, cgrads, b, db)?;
    }
    Ok(())
}

/// `GruCell` の VJP を子テープへ記録する。ゲートは子テープ上で `pre_i`・`pre_h` から
/// 作り直す（payload を定数として使うと二階が落ちるため）。寄与の順序は 1 階と同じ
/// （x、w_ih、w_hh、h_prev〈direct〉、h_prev〈affine〉、b_ih、b_hh）。
pub(super) fn gru_cell_cgrad<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    cgrads: &mut [Option<Var<'c>>],
    mirror: &[Option<Var<'c>>],
    id: NodeId,
    g: &Var<'c>,
    ids: &CellIds,
) -> Result<(), AutodiffError> {
    let m = mirrors(mirror, ids)?;
    let out_shape = get_mirror(mirror, id)?.shape();
    let hidden = out_shape.get(1).copied().ok_or_else(|| {
        AutodiffError::Backward("create_graph: GruCell の出力が rank 2 でない".into())
    })?;
    let width3 = hidden.checked_mul(3).ok_or_else(|| {
        AutodiffError::Backward("create_graph: GruCell のゲート幅 3H がオーバーフローする".into())
    })?;
    let w_ih_cols = m.w_ih.shape().get(1).copied();
    if w_ih_cols != Some(width3) {
        return Err(AutodiffError::Backward(format!(
            "create_graph: GruCell の w_ih 列数 {w_ih_cols:?} が 3H = {width3} と一致しない"
        )));
    }

    // 再構築（forward の gemm_bias_act と同じく FP32 厳密 GEMM + bias 加算）。
    let mut pre_i = m.x.matmul_fp32_strict(&m.w_ih)?;
    if let Some(b) = m.b_ih.as_ref() {
        pre_i = pre_i.add(b)?;
    }
    let mut pre_h = m.h_prev.matmul_fp32_strict(&m.w_hh)?;
    if let Some(b) = m.b_hh.as_ref() {
        pre_h = pre_h.add(b)?;
    }
    let blk = |v: &Var<'c>, k: usize| v.narrow(1, k * hidden, hidden);
    let r = blk(&pre_i, 0)?.add(&blk(&pre_h, 0)?)?.sigmoid();
    let z = blk(&pre_i, 1)?.add(&blk(&pre_h, 1)?)?.sigmoid();
    let q = blk(&pre_h, 2)?;
    let n = r.mul(&q)?.add(&blk(&pre_i, 2)?)?.tanh();

    // 1 階 `eval::gru_backward` と同じ式・同じ掛ける順。
    let one = ones_like(child, &out_shape)?;
    let dn = g.mul(&one.sub(&z)?)?;
    let dz = g.mul(&m.h_prev.sub(&n)?)?;
    let dh_prev_direct = g.mul(&z)?;
    let d_pre_n = dn.mul(&one.sub(&n.mul(&n)?)?)?;
    let dr = d_pre_n.mul(&q)?;
    let d_pre_r = dr.mul(&r)?.mul(&one.sub(&r)?)?;
    let d_pre_z = dz.mul(&z)?.mul(&one.sub(&z)?)?;
    let d_pre_i = Var::cat(&[d_pre_r, d_pre_z, d_pre_n], 1)?;
    let d_pre_h = Var::cat(&[d_pre_r, d_pre_z, d_pre_n.mul(&r)?], 1)?;

    let dx = d_pre_i.matmul_fp32_strict(&m.w_ih.transpose(0, 1)?)?;
    let dw_ih = m.x.transpose(0, 1)?.matmul_fp32_strict(&d_pre_i)?;
    let dh_prev_affine = d_pre_h.matmul_fp32_strict(&m.w_hh.transpose(0, 1)?)?;
    let dw_hh = m.h_prev.transpose(0, 1)?.matmul_fp32_strict(&d_pre_h)?;

    accumulate(parent_nodes, cgrads, ids.x, dx)?;
    accumulate(parent_nodes, cgrads, ids.w_ih, dw_ih)?;
    accumulate(parent_nodes, cgrads, ids.w_hh, dw_hh)?;
    accumulate(parent_nodes, cgrads, ids.h_prev, dh_prev_direct)?;
    accumulate(parent_nodes, cgrads, ids.h_prev, dh_prev_affine)?;
    if let Some(b) = ids.b_ih {
        let db = reduce_bias_grad_var(child, &d_pre_i, &[width3])?;
        accumulate(parent_nodes, cgrads, b, db)?;
    }
    if let Some(b) = ids.b_hh {
        let db = reduce_bias_grad_var(child, &d_pre_h, &[width3])?;
        accumulate(parent_nodes, cgrads, b, db)?;
    }
    Ok(())
}
