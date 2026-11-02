//! `LstmCell`・`LstmHidden` の `create_graph` 対応（イシュー #3796・親 #3719 系・
//! 決定記録 `docs/autodiff-higher-order-grad-decision.md` §37・§61）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `validate_ancestors`（入口検査）・
//! `replay_op`（forward 再生）・`build_cgrads`（1 階 VJP の子テープ記録）から呼ばれる。
//! facade の `hvp`・`jvp`・`jacfwd`・`jvp_create_graph`・`backward_create_graph` が
//! LSTM を含むモデルを拒否しないための前提となる（facade の公開 LSTM API は
//! 必ず fused Op の `LstmCell`→`LstmHidden` を積む）。兄弟の
//! [`super::rnn_gru_cell`]（RNN・GRU。#3795）と同じ方式で、共通ヘルパー
//! （`guard_overflow`・`ones_like`）はそちらを再利用する。
//!
//! **方式**（1 階 `grad.rs` の `Op::LstmCell`／`Op::LstmHidden` 腕と同じオペランド順・
//! 同じ掛ける順で組む。決定記録 §37.10 の案 (a)・自己再生）:
//! - **V1**: 1 階 VJP の式を子テープ上の `Var` 演算（`mul`・`sub`・`sigmoid`・`tanh`・
//!   `narrow`・`cat`・`matmul_fp32_strict`・`transpose`）の合成として記録する。GEMM は
//!   forward の `gemm_bias_act` と同じく常に FP32 厳密。
//! - **V3（bias のみ）**: `db_ih`・`db_hh` は 1 階 `affine_vjp` と同じ f64 縮約ホスト関数を
//!   [`super::reduce_bias_grad_var`] 経由で使う（`.claude/rules/coding-rust.md`）。
//! - payload（`gates_ifg`・`gate_o`）は x・h_prev・W の関数であり定数として使うと二階が
//!   落ちるため、子テープ上で `pre = x·W_ih + h·W_hh (+bias)` からゲートを作り直す。
//! - **部分幅**: 1 階 `affine_vjp` は LSTM では W の列を部分幅で扱う（`LstmCell` は
//!   `i,f,g` の 3H 列、`LstmHidden` は `o` の H 列）。ここでは `d_pre` を全幅へゼロ埋めして
//!   全幅 W と掛ける形を取らず、W を `narrow` した部分幅と掛ける（W の他列に非有限値が
//!   あると `0 × inf = NaN` になり、1 階が触れない列で値が食い違うため）。dW・db だけを
//!   `cat` でゼロ埋めして全幅へ戻す（`Pad` は §37.11 の再確認対象なので使わない）。
//! - **push 順序契約**: 子テープでも `LstmCell` の写しは `(h, c)` を続けて積むため、
//!   `LstmHidden` の写しは `c_m.id + 1` に存在する。その位置の Op と shape を再生時に検査する。
//!
//! **閉包性**: 子テープへ積む Op は `supports_create_graph() == true` な Op
//! （`MatMul`・`Transpose`・`Narrow`・`Concat`・`Add`・`Sub`・`Mul`・`Sigmoid`・`Tanh`・
//! `Sum`・定数 `Leaf`、および写しの `LstmCell`・`LstmHidden` 自身）に限る。
//!
//! **数値契約**: forward の FMA（`c = f.mul_add(c_prev, i·g)`）は `c` の写しが同じ
//! カーネルで再生されるため作り直さない。cgrad の値は 1 階と REQ-2 統一複合判定で一致する
//! （bit 一致は主張しない。`first_order()` 側は素の `Tape::backward` と bit 同一）。
//! tolerance・baseline は変えない。
//!
//! **型付きの拒否**: `LstmHidden.cell` が `LstmCell` でない、`w_ih` の列数が `4H` でない、
//! 二階の中間積が overflow しうる入力は、子テープへ書く前（または呼び出し側の rollback で
//! 子テープが空へ戻る位置）で `AutodiffError::Backward` にする。

use super::rnn_gru_cell::{guard_overflow, ones_like};
use super::{accumulate, get_mirror, reduce_bias_grad_var};
use crate::error::AutodiffError;
use crate::tape::{NodeId, Op, Tape, TapeNode};
use crate::var::{GateParams, Var};
use fandhe_ai_tensor_core::Tensor;

/// `Op::LstmCell` の入力ノード束（`c_prev` を含むため RNN/GRU の `CellIds` とは別）。
#[derive(Clone, Copy)]
pub(super) struct LstmIds {
    pub(super) x: NodeId,
    pub(super) h_prev: NodeId,
    pub(super) c_prev: NodeId,
    pub(super) w_ih: NodeId,
    pub(super) w_hh: NodeId,
    pub(super) b_ih: Option<NodeId>,
    pub(super) b_hh: Option<NodeId>,
}

impl LstmIds {
    /// 親テープ上の `Op::LstmCell` から束を取り出す。`LstmCell` でなければ型付きエラー
    /// （push 順序契約違反。`grad.rs` の 1 階 VJP と同じ fail-closed）。
    fn from_cell_op(op: &Op) -> Result<Self, AutodiffError> {
        match op {
            Op::LstmCell {
                x,
                h_prev,
                c_prev,
                w_ih,
                w_hh,
                b_ih,
                b_hh,
                ..
            } => Ok(Self {
                x: *x,
                h_prev: *h_prev,
                c_prev: *c_prev,
                w_ih: *w_ih,
                w_hh: *w_hh,
                b_ih: *b_ih,
                b_hh: *b_hh,
            }),
            _ => Err(AutodiffError::Backward(
                "create_graph: LstmHidden.cell が LstmCell を指さない（push 順序契約違反）".into(),
            )),
        }
    }
}

/// 子テープ上の写しをまとめて取り出したもの。
struct Mirrors<'c> {
    x: Var<'c>,
    h_prev: Var<'c>,
    c_prev: Var<'c>,
    w_ih: Var<'c>,
    w_hh: Var<'c>,
    b_ih: Option<Var<'c>>,
    b_hh: Option<Var<'c>>,
}

fn mirrors<'c>(mirror: &[Option<Var<'c>>], ids: &LstmIds) -> Result<Mirrors<'c>, AutodiffError> {
    Ok(Mirrors {
        x: get_mirror(mirror, ids.x)?,
        h_prev: get_mirror(mirror, ids.h_prev)?,
        c_prev: get_mirror(mirror, ids.c_prev)?,
        w_ih: get_mirror(mirror, ids.w_ih)?,
        w_hh: get_mirror(mirror, ids.w_hh)?,
        b_ih: ids.b_ih.map(|b| get_mirror(mirror, b)).transpose()?,
        b_hh: ids.b_hh.map(|b| get_mirror(mirror, b)).transpose()?,
    })
}

/// 入口検査（`validate_ancestors` から子テープへ書く前に呼ばれる）。`LstmHidden.cell` が
/// 親テープの範囲内で `Op::LstmCell` を指すことを確かめる（§37.3）。
pub(super) fn check_lstm_hidden(
    parent_nodes: &[TapeNode],
    cell: NodeId,
) -> Result<(), AutodiffError> {
    let node = parent_nodes.get(cell.0).ok_or_else(|| {
        AutodiffError::Backward("create_graph: LstmHidden.cell が親テープの範囲外".into())
    })?;
    LstmIds::from_cell_op(&node.op).map(|_| ())
}

/// `LstmCell` の forward 再生。`(h, c)` を積み、`c`（`LstmCell` ノードの値）を返す。
/// `h` は `c.id + 1` に残り、続く `LstmHidden` の再生が [`replay_lstm_hidden`] で拾う。
pub(super) fn replay_lstm_cell<'c>(
    mirror: &[Option<Var<'c>>],
    ids: &LstmIds,
) -> Result<Var<'c>, AutodiffError> {
    let m = mirrors(mirror, ids)?;
    let (_h, c) = m.x.lstm_cell(
        &m.h_prev,
        &m.c_prev,
        GateParams {
            w_ih: &m.w_ih,
            w_hh: &m.w_hh,
            b_ih: m.b_ih.as_ref(),
            b_hh: m.b_hh.as_ref(),
        },
    )?;
    Ok(c)
}

/// `LstmHidden` の forward 再生。`cell` の写し `c_m` の直後（`id + 1`）に `lstm_cell` が
/// 積んだ `h` ノードを、push 順序契約どおりの Op・shape かを検査したうえで返す。
pub(super) fn replay_lstm_hidden<'c>(
    mirror: &[Option<Var<'c>>],
    cell: NodeId,
    shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    let c_m = get_mirror(mirror, cell)?;
    let child = c_m.tape();
    let hidden_id = NodeId(c_m.node_id().0 + 1);
    let ok = {
        let nodes = child.nodes.borrow();
        nodes.get(hidden_id.0).is_some_and(|n| {
            matches!(&n.op, Op::LstmHidden { cell: c, .. } if *c == c_m.node_id())
                && n.shape.as_slice() == shape
        })
    };
    if !ok {
        return Err(AutodiffError::Backward(
            "create_graph: LstmHidden の写しが LstmCell の写しの直後に無い（push 順序契約違反）"
                .into(),
        ));
    }
    Ok(Var::from_raw(child, hidden_id))
}

/// `pre = x·W_ih (+b_ih) + h·W_hh (+b_hh)`（`[B, 4H]`）を子テープ上で作り直す。
/// forward の `gemm_bias_act` 2 回の和と同じ加算順。
fn rebuild_pre<'c>(m: &Mirrors<'c>) -> Result<Var<'c>, AutodiffError> {
    let mut pre_i = m.x.matmul_fp32_strict(&m.w_ih)?;
    if let Some(b) = m.b_ih.as_ref() {
        pre_i = pre_i.add(b)?;
    }
    let mut pre_h = m.h_prev.matmul_fp32_strict(&m.w_hh)?;
    if let Some(b) = m.b_hh.as_ref() {
        pre_h = pre_h.add(b)?;
    }
    pre_i.add(&pre_h)
}

/// 隠れ幅 `H`（`c_prev` の列数）を返す。`w_ih`／`w_hh` の列数が `4H` でなければ拒否する。
fn hidden_width(m: &Mirrors<'_>, what: &str) -> Result<usize, AutodiffError> {
    let hidden = m.c_prev.shape().get(1).copied().ok_or_else(|| {
        AutodiffError::Backward(format!("create_graph: {what} の c_prev が rank 2 でない"))
    })?;
    let width4 = hidden.checked_mul(4).ok_or_else(|| {
        AutodiffError::Backward(format!(
            "create_graph: {what} のゲート幅 4H がオーバーフローする"
        ))
    })?;
    for (name, w) in [("w_ih", &m.w_ih), ("w_hh", &m.w_hh)] {
        let cols = w.shape().get(1).copied();
        if cols != Some(width4) {
            return Err(AutodiffError::Backward(format!(
                "create_graph: {what} の {name} 列数 {cols:?} が 4H = {width4} と一致しない"
            )));
        }
    }
    Ok(hidden)
}

fn zeros<'c>(child: &'c Tape, shape: &[usize]) -> Result<Var<'c>, AutodiffError> {
    Ok(child.var_no_grad(&Tensor::zeros(shape)?))
}

/// W の部分幅 `[col_start, col_start + width)` の転置（`[width, rows]`）。
fn narrowed_wt<'c>(w: &Var<'c>, col_start: usize, width: usize) -> Result<Var<'c>, AutodiffError> {
    w.narrow(1, col_start, width)?.transpose(0, 1)
}

/// `[前のゼロ列, 本体, 後ろのゼロ列]` を `dim` 方向に連結する（ゼロ幅の部分は省く）。
fn pad_with_zeros<'c>(
    child: &'c Tape,
    body: Var<'c>,
    lead: &[usize],
    trail: &[usize],
    dim: usize,
) -> Result<Var<'c>, AutodiffError> {
    let mut parts = Vec::with_capacity(3);
    if lead.iter().all(|&d| d > 0) {
        parts.push(zeros(child, lead)?);
    }
    parts.push(body);
    if trail.iter().all(|&d| d > 0) {
        parts.push(zeros(child, trail)?);
    }
    if parts.len() == 1 {
        Ok(body)
    } else {
        Var::cat(&parts, dim)
    }
}

/// 部分幅の `d_pre` から作る入力・パラメータ勾配（dW・db は全幅へゼロ埋め済み）。
struct AffineGrads<'c> {
    dx: Var<'c>,
    dh_prev: Var<'c>,
    dw_ih: Var<'c>,
    dw_hh: Var<'c>,
    db: Option<Var<'c>>,
}

/// `d_pre`（`[B, width]`）が全幅 `4H` 中の `col_start` 列から始まるときの勾配一式。
fn affine_partial<'c>(
    child: &'c Tape,
    m: &Mirrors<'c>,
    d_pre: &Var<'c>,
    col_start: usize,
    hidden: usize,
) -> Result<AffineGrads<'c>, AutodiffError> {
    let width = d_pre.shape().get(1).copied().ok_or_else(|| {
        AutodiffError::Backward("create_graph: LSTM の d_pre が rank 2 でない".into())
    })?;
    let after = hidden
        .checked_mul(4)
        .and_then(|w4| w4.checked_sub(col_start.checked_add(width)?))
        .ok_or_else(|| {
            AutodiffError::Backward("create_graph: LSTM の部分幅が 4H を超える".into())
        })?;
    let d_in = m.x.shape().get(1).copied().unwrap_or(0);
    let dx = d_pre.matmul_fp32_strict(&narrowed_wt(&m.w_ih, col_start, width)?)?;
    let dh_prev = d_pre.matmul_fp32_strict(&narrowed_wt(&m.w_hh, col_start, width)?)?;
    let dw_ih = pad_with_zeros(
        child,
        m.x.transpose(0, 1)?.matmul_fp32_strict(d_pre)?,
        &[d_in, col_start],
        &[d_in, after],
        1,
    )?;
    let dw_hh = pad_with_zeros(
        child,
        m.h_prev.transpose(0, 1)?.matmul_fp32_strict(d_pre)?,
        &[hidden, col_start],
        &[hidden, after],
        1,
    )?;
    let db = if m.b_ih.is_some() || m.b_hh.is_some() {
        let part = reduce_bias_grad_var(child, d_pre, &[width])?;
        Some(pad_with_zeros(child, part, &[col_start], &[after], 0)?)
    } else {
        None
    };
    Ok(AffineGrads {
        dx,
        dh_prev,
        dw_ih,
        dw_hh,
        db,
    })
}

/// `LstmCell` の VJP（上流 `g` = `dc`）を子テープへ記録する。寄与の順序は 1 階と同じ
/// （x、h_prev、c_prev、w_ih、w_hh、b_ih、b_hh）。
pub(super) fn lstm_cell_cgrad<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    cgrads: &mut [Option<Var<'c>>],
    mirror: &[Option<Var<'c>>],
    g: &Var<'c>,
    ids: &LstmIds,
) -> Result<(), AutodiffError> {
    let m = mirrors(mirror, ids)?;
    let hidden = hidden_width(&m, "LstmCell")?;
    let mut operands = vec![&m.x, &m.h_prev, &m.c_prev, &m.w_ih, &m.w_hh];
    operands.extend(m.b_ih.iter().chain(m.b_hh.iter()));
    guard_overflow(child, g, &operands, "LstmCell")?;

    let pre = rebuild_pre(&m)?;
    let blk = |k: usize| pre.narrow(1, k * hidden, hidden);
    let i = blk(0)?.sigmoid();
    let f = blk(1)?.sigmoid();
    let gc = blk(2)?.tanh();
    let one = ones_like(child, &i.shape())?;

    // 1 階 `eval::lstm_cell_backward` と同じ式・同じ掛ける順。
    let d_pre_i = g.mul(&gc)?.mul(&i)?.mul(&one.sub(&i)?)?;
    let d_pre_f = g.mul(&m.c_prev)?.mul(&f)?.mul(&one.sub(&f)?)?;
    let d_pre_g = g.mul(&i)?.mul(&one.sub(&gc.mul(&gc)?)?)?;
    let dc_prev = g.mul(&f)?;
    let d_pre = Var::cat(&[d_pre_i, d_pre_f, d_pre_g], 1)?;
    let a = affine_partial(child, &m, &d_pre, 0, hidden)?;

    accumulate(parent_nodes, cgrads, ids.x, a.dx)?;
    accumulate(parent_nodes, cgrads, ids.h_prev, a.dh_prev)?;
    accumulate(parent_nodes, cgrads, ids.c_prev, dc_prev)?;
    accumulate(parent_nodes, cgrads, ids.w_ih, a.dw_ih)?;
    accumulate(parent_nodes, cgrads, ids.w_hh, a.dw_hh)?;
    if let (Some(b), Some(db)) = (ids.b_ih, a.db) {
        accumulate(parent_nodes, cgrads, b, db)?;
    }
    if let (Some(b), Some(db)) = (ids.b_hh, a.db) {
        accumulate(parent_nodes, cgrads, b, db)?;
    }
    Ok(())
}

/// `LstmHidden` の VJP（上流 `g` = `dh`）を子テープへ記録する。x 等の NodeId は
/// 親テープの `nodes[cell].op`（`LstmCell`）から読む。寄与の順序は 1 階と同じ
/// （cell、x、h_prev、w_ih、w_hh、b_ih、b_hh）。
pub(super) fn lstm_hidden_cgrad<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    cgrads: &mut [Option<Var<'c>>],
    mirror: &[Option<Var<'c>>],
    cell: NodeId,
    g: &Var<'c>,
) -> Result<(), AutodiffError> {
    let cell_node = parent_nodes.get(cell.0).ok_or_else(|| {
        AutodiffError::Backward("create_graph: LstmHidden.cell が親テープの範囲外".into())
    })?;
    let ids = LstmIds::from_cell_op(&cell_node.op)?;
    let m = mirrors(mirror, &ids)?;
    let hidden = hidden_width(&m, "LstmHidden")?;
    let c_m = get_mirror(mirror, cell)?;
    let mut operands = vec![&m.x, &m.h_prev, &m.w_ih, &m.w_hh];
    operands.extend(m.b_ih.iter().chain(m.b_hh.iter()));
    guard_overflow(child, g, &operands, "LstmHidden")?;

    let pre = rebuild_pre(&m)?;
    let o = pre.narrow(1, 3 * hidden, hidden)?.sigmoid();
    let tc = c_m.tanh();
    let one = ones_like(child, &o.shape())?;

    // 1 階 `eval::lstm_hidden_backward` と同じ式・同じ掛ける順。
    let d_pre_o = g.mul(&tc)?.mul(&o)?.mul(&one.sub(&o)?)?;
    let dc = g.mul(&o)?.mul(&one.sub(&tc.mul(&tc)?)?)?;
    let a = affine_partial(child, &m, &d_pre_o, 3 * hidden, hidden)?;

    accumulate(parent_nodes, cgrads, cell, dc)?;
    accumulate(parent_nodes, cgrads, ids.x, a.dx)?;
    accumulate(parent_nodes, cgrads, ids.h_prev, a.dh_prev)?;
    accumulate(parent_nodes, cgrads, ids.w_ih, a.dw_ih)?;
    accumulate(parent_nodes, cgrads, ids.w_hh, a.dw_hh)?;
    if let (Some(b), Some(db)) = (ids.b_ih, a.db) {
        accumulate(parent_nodes, cgrads, b, db)?;
    }
    if let (Some(b), Some(db)) = (ids.b_hh, a.db) {
        accumulate(parent_nodes, cgrads, b, db)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
    }

    fn pseudo(n: usize, seed: f32, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.37).sin() * 1.5 * scale)
            .collect()
    }

    /// `LstmHidden.cell` が `LstmCell` でない親は入口検査で型付き拒否される。
    #[test]
    fn check_lstm_hidden_rejects_non_cell_target() {
        let tape = Tape::new();
        let leaf = tape.var(&t(vec![0.5], &[1, 1]));
        let hidden = tape.push_eager(
            Op::LstmHidden {
                cell: leaf.node_id(),
                gate_o: Tensor::<f32>::zeros(&[1, 1]).unwrap(),
            },
            Tensor::<f32>::zeros(&[1, 1]).unwrap(),
        );
        let nodes = tape.nodes.borrow();
        let err = check_lstm_hidden(&nodes, leaf.node_id()).unwrap_err();
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
        // 範囲外の cell も panic せず型付きエラー。
        let err = check_lstm_hidden(&nodes, NodeId(nodes.len() + 5)).unwrap_err();
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
        assert!(matches!(nodes[hidden.0].op, Op::LstmHidden { .. }));
    }

    /// 子テープの `c_m.id + 1` が `LstmHidden` でない場合、再生は型付き拒否になる。
    #[test]
    fn replay_lstm_hidden_rejects_broken_push_order() {
        let child = Tape::new();
        let leaf = child.var(&t(vec![0.5], &[1, 1]));
        let mirror = vec![Some(leaf)];
        let err = replay_lstm_hidden(&mirror, leaf.node_id(), &[1, 1]).unwrap_err();
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
        // 直後に別の Op があっても（LstmHidden でなければ）拒否する。
        let _next = leaf.mul(&leaf).unwrap();
        let err = replay_lstm_hidden(&mirror, leaf.node_id(), &[1, 1]).unwrap_err();
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
    }

    /// 正のプローブ: 正しく積まれた `lstm_cell` では `h` ノードを拾え、shape 違いは拒否する。
    #[test]
    fn replay_lstm_hidden_accepts_correct_push_order_and_checks_shape() {
        let tape = Tape::new();
        let (b, d, h) = (2usize, 3usize, 2usize);
        let x = tape.var(&t(pseudo(b * d, 1.0, 1.0), &[b, d]));
        let hp = tape.var(&t(pseudo(b * h, 2.0, 1.0), &[b, h]));
        let cp = tape.var(&t(pseudo(b * h, 3.0, 1.0), &[b, h]));
        let wi = tape.var(&t(pseudo(d * 4 * h, 4.0, 0.5), &[d, 4 * h]));
        let wh = tape.var(&t(pseudo(h * 4 * h, 5.0, 0.5), &[h, 4 * h]));
        let (hv, cv) = x
            .lstm_cell(
                &hp,
                &cp,
                GateParams {
                    w_ih: &wi,
                    w_hh: &wh,
                    b_ih: None,
                    b_hh: None,
                },
            )
            .unwrap();
        let mut mirror = vec![None; cv.node_id().0 + 1];
        mirror[cv.node_id().0] = Some(cv);
        let got = replay_lstm_hidden(&mirror, cv.node_id(), &[b, h]).unwrap();
        assert_eq!(got.node_id(), hv.node_id());
        let err = replay_lstm_hidden(&mirror, cv.node_id(), &[b, h + 1]).unwrap_err();
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
    }

    /// 子テープで作り直すゲートが forward の payload（`gates_ifg`・`gate_o`）と一致するか
    /// 掃引で観測する（§61.2。一致すれば forward 揃えの補正は不要）。
    #[test]
    fn rebuilt_gates_match_payload_bitwise() {
        let (b, d, h) = (3usize, 4usize, 5usize);
        let mut checked = 0usize;
        for bias in [false, true] {
            for scale in [1e-2f32, 1.0, 3.0, 10.0, 1e2, 1e3, 1e4] {
                let tape = Tape::new();
                let x = tape.var(&t(pseudo(b * d, 1.0, scale), &[b, d]));
                let hp = tape.var(&t(pseudo(b * h, 2.0, scale), &[b, h]));
                let cp = tape.var(&t(pseudo(b * h, 3.0, scale), &[b, h]));
                let wi = tape.var(&t(pseudo(d * 4 * h, 4.0, scale), &[d, 4 * h]));
                let wh = tape.var(&t(pseudo(h * 4 * h, 5.0, scale), &[h, 4 * h]));
                let bi = tape.var(&t(pseudo(4 * h, 6.0, scale), &[4 * h]));
                let bh = tape.var(&t(pseudo(4 * h, 7.0, scale), &[4 * h]));
                let (hv, cv) = x
                    .lstm_cell(
                        &hp,
                        &cp,
                        GateParams {
                            w_ih: &wi,
                            w_hh: &wh,
                            b_ih: bias.then_some(&bi),
                            b_hh: bias.then_some(&bh),
                        },
                    )
                    .unwrap();
                let m = Mirrors {
                    x,
                    h_prev: hp,
                    c_prev: cp,
                    w_ih: wi,
                    w_hh: wh,
                    b_ih: bias.then_some(bi),
                    b_hh: bias.then_some(bh),
                };
                let pre = rebuild_pre(&m).unwrap();
                let gates = |k: usize| pre.narrow(1, k * h, h).unwrap();
                let (i, f, o) = (gates(0).sigmoid(), gates(1).sigmoid(), gates(3).sigmoid());
                let gc = gates(2).tanh();
                let (gi, go) = {
                    let nodes = tape.nodes.borrow();
                    let (Op::LstmCell { gates_ifg, .. }, Op::LstmHidden { gate_o, .. }) =
                        (&nodes[cv.node_id().0].op, &nodes[hv.node_id().0].op)
                    else {
                        panic!("push 順序契約");
                    };
                    (
                        gates_ifg.contiguous().as_slice().unwrap().to_vec(),
                        gate_o.contiguous().as_slice().unwrap().to_vec(),
                    )
                };
                let val = |v: &Var<'_>| v.value().contiguous().as_slice().unwrap().to_vec();
                let (vi, vf, vg, vo) = (val(&i), val(&f), val(&gc), val(&o));
                for r in 0..b {
                    for j in 0..h {
                        let base = r * 3 * h;
                        let pairs = [
                            (gi[base + j], vi[r * h + j]),
                            (gi[base + h + j], vf[r * h + j]),
                            (gi[base + 2 * h + j], vg[r * h + j]),
                            (go[r * h + j], vo[r * h + j]),
                        ];
                        for (p, q) in pairs {
                            assert_eq!(
                                p.to_bits(),
                                q.to_bits(),
                                "bias={bias} scale={scale}: payload {p} vs rebuilt {q}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 100);
    }
}
