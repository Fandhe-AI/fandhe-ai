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
//! **数値契約**: GRU の再構築は forward の `mul_add` と丸めがずれうるため、`n` の値には
//! 丸めが割れた要素を forward の FMA 値・傾き `1 − n_fwd²` の一次式へ置き換え（`1 − n²` の増幅と二階の不整合を防ぐ）、cgrad の値は
//! 1 階と REQ-2 統一複合判定で一致する（bit 一致は主張しない。`first_order()` 側は素の
//! `Tape::backward` と bit 同一）。tolerance・baseline は変えない。
//!
//! **型付きの拒否（PR #3952 の codex 指摘。§37.12 の「違反入力には拒否を足す」に従う）**:
//! - FMA 相殺: forward は `r.mul_add(q, pre_i_n)`、再構築は積を先に f32 へ丸める。相殺で
//!   `tanh` 引数が REQ-2 の複合判定外へずれる入力（`n` 値が不一致）は `Err` にする。
//! - 中間積 overflow: 二階の随伴は上流 `g` に各オペランド（x・h_prev・W・q 等）を掛けるため、
//!   `max|g| × max(1, オペランドの最大絶対値)` が `STAGE_BOUND_GUARD`（2^125）以上なら、
//!   真値が 0 でも `0 × inf = NaN` になりうるので `Err` にする（保守的。cotangent の
//!   絶対値は 1 以下と仮定）。いずれも構造を積む前後で判定し、`Err` 時は呼び出し側の
//!   ロールバックで子テープは空へ戻る。

use super::{accumulate, get_mirror, reduce_bias_grad_var, value_of};
use crate::create_graph_norm::STAGE_BOUND_GUARD;
use crate::error::AutodiffError;
use crate::tape::{NodeId, Tape, TapeNode};
use crate::var::{GateParams, Var};
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::tolerance::{ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE};

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

fn reject(what: &str) -> AutodiffError {
    AutodiffError::Backward(format!(
        "create_graph: {what}（f32 の子テープ構造が 1 階と食い違いうる入力。決定記録 §37.12。\
         拒否は保守的で、実際に誤るとは限らない）"
    ))
}

fn max_abs_of(t: &Tensor<f32>) -> f64 {
    t.host_slice()
        .iter()
        .fold(0.0f64, |m, &v| m.max(f64::from(v).abs()))
}

/// 二階の随伴が `g × オペランド` で overflow しうる入力を拒否する（RNN・GRU 共通）。
/// `operands` は `g` に掛かりうる値（入力・重み・bias・GRU の `q` 等）。
fn guard_overflow<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    operands: &[&Var<'c>],
    what: &str,
) -> Result<(), AutodiffError> {
    let gv = value_of(child, g)?;
    if !gv.host_slice().iter().all(|v| v.is_finite()) {
        return Err(reject(&format!("{what}: 上流 g に非有限要素がある")));
    }
    let mut m = 1.0f64;
    for v in operands {
        let t = value_of(child, v)?;
        let a = max_abs_of(&t);
        if !a.is_finite() {
            return Err(reject(&format!("{what}: オペランドに非有限要素がある")));
        }
        m = m.max(a);
    }
    let bound = max_abs_of(&gv) * m;
    if bound.is_finite() && bound < STAGE_BOUND_GUARD {
        Ok(())
    } else {
        Err(reject(&format!(
            "{what}: 二階の逆伝播の中間積（上流 g × オペランド）の上界が 2^125 以上"
        )))
    }
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
    let mut operands = vec![&m.x, &m.h_prev, &m.w_ih, &m.w_hh];
    operands.extend(m.b_ih.iter().chain(m.b_hh.iter()));
    guard_overflow(child, g, &operands, "RnnCell")?;
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
    let n_pre = r.mul(&q)?.add(&blk(&pre_i, 2)?)?;
    let n = n_pre.tanh();

    // FMA 契約の補正と判定: forward は `r.mul_add(q, pre_i_n)`、上の再構築は積を先に丸める。
    // (1) 丸めが割れる要素（`n_fwd != n_rec`）は値だけでなく微分因子も forward に合わせる。
    //     値だけ `n_fwd − n_rec` を足すと、`d_pre_n` の `1 − n²` は forward 基準でも、その `n` を
    //     子テープで微分する Tanh の逆伝播が補正前の出力（`1 − n_rec²`）を使い、二階（reset 列への
    //     混合二階等）が forward 基準（`1 − n_fwd²`）から大きく外れる（PR #3952 codex P1・再指摘）。
    //     そこで割れた要素だけ、値 `n_fwd`・傾き `1 − n_fwd²` の 1 次式
    //     `n_fwd + (a − a₀)·(1 − n_fwd²)`（`a₀` は `a` の値の定数写し）へ置き換える。割れない要素は
    //     `tanh` のまま（一次式は割れた要素の 3 階以上の曲率だけを落とす。二階までは forward と整合）。
    // (2) 相殺で `n` が REQ-2 複合判定外へずれる入力は従来どおり拒否する（tolerance は緩めない）。
    let n = {
        let av = value_of(child, &n_pre)?;
        let pv = value_of(child, &blk(&pre_i, 2)?)?;
        let rv = value_of(child, &r)?;
        let qv = value_of(child, &q)?;
        let nv = value_of(child, &n)?;
        let (rs, qs, ps, ns, avs) = (
            rv.host_slice(),
            qv.host_slice(),
            pv.host_slice(),
            nv.host_slice(),
            av.host_slice(),
        );
        let len = ns.len();
        let (mut keep, mut mask_n, mut slope) = (
            Vec::with_capacity(len),
            Vec::with_capacity(len),
            Vec::with_capacity(len),
        );
        let mut any_split = false;
        for i in 0..len {
            let (rr, qq, pp, nn) = (rs[i], qs[i], ps[i], ns[i]);
            let fused_f = rr.mul_add(qq, pp).tanh();
            let fused = f64::from(fused_f);
            let split = f64::from((rr * qq + pp).tanh());
            if fused.is_nan() && split.is_nan() {
                keep.push(1.0f32);
                mask_n.push(0.0f32);
                slope.push(0.0f32);
                continue;
            }
            let diff = (fused - split).abs();
            if !(diff < ABSOLUTE_RESCUE_THRESHOLD || diff < RELATIVE_TOLERANCE * fused.abs()) {
                return Err(reject(
                    "GruCell: r·q + pre_i_n の FMA と分離丸めで n が REQ-2 の許容を超えてずれる（桁落ち）",
                ));
            }
            if fused_f != nn && fused_f.is_finite() {
                any_split = true;
                keep.push(0.0);
                mask_n.push(fused_f);
                slope.push(1.0 - fused_f * fused_f);
            } else {
                keep.push(1.0);
                mask_n.push(0.0);
                slope.push(0.0);
            }
        }
        if any_split {
            // 一次式の `(a − a₀)·0` が非有限の `a` で NaN を撒かないよう、非有限があれば拒否する。
            if avs.iter().any(|v| !v.is_finite()) {
                return Err(reject(
                    "GruCell: n の丸め補正が必要な入力に非有限の tanh 引数が混在する",
                ));
            }
            let keep = child.var_no_grad(&Tensor::new(keep, &out_shape)?);
            let mask_n = child.var_no_grad(&Tensor::new(mask_n, &out_shape)?);
            let slope = child.var_no_grad(&Tensor::new(slope, &out_shape)?);
            let a0 = child.var_no_grad(&av);
            n.mul(&keep)?
                .add(&n_pre.sub(&a0)?.mul(&slope)?)?
                .add(&mask_n)?
        } else {
            n
        }
    };
    let mut operands = vec![&m.x, &m.h_prev, &m.w_ih, &m.w_hh, &q];
    operands.extend(m.b_ih.iter().chain(m.b_hh.iter()));
    guard_overflow(child, g, &operands, "GruCell")?;

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
