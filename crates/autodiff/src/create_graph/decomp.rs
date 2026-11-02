//! 分解族の多出力ノードのうち `QrQ`・`QrR` の `create_graph` 対応（イシュー #3807・親 #3734・
//! 決定記録 `docs/autodiff-higher-order-grad-decision.md` §44・§67）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `validate_ancestors`（入口検査 [`check_qr_pair`]）・
//! `build_mirror` 段 2（再生 [`replay_qr`]）・`build_cgrads`（1 階 VJP の子テープ記録 [`qr_cgrad`]）
//! から呼ばれる。後続の `Svd*`・`Eigh*`（#3808）も [`sibling_of`] を再利用する想定。
//!
//! **多出力ノードの扱い（push 順序契約）**: `Var::qr` は `QrQ` → `QrR` を連続で `push_eager` する
//! ため、「`QrQ` が `id` なら `QrR` は `id + 1` で同じ `input` を持つ」が親テープでも子テープでも
//! 成り立つ。`QrQ.r`・`QrR.q` の payload は追跡されない定数なので構造には使わず（`dR/dA`・`dQ/dA` が
//! 要るため）、兄弟は子テープ上の隣接ノードとして取る。再生は公開メソッド `Var::qr` への自己再生で
//! 行い（forward の検証を迂回しない）、`qr()` は 1 回しか呼ばない。
//!
//! **方式（V3 + f32 構造。§44.10）**: 値は 1 階と同じホスト関数 `eval::linalg::qr_vjp` の結果で、
//! `Op::Contiguous` ノード（恒等 VJP）として積む（[`super::push_v3`]）。ゆえに `cg.grad` は CPU で
//! 1 階と bit 一致する。構造は `m ≥ n`（`k = n`）に限り、`M = R dRᵀ − dQᵀ Q`・`C = copyltu(M)`・
//! `dA = (dQ + Q C) R⁻ᵀ` を f32 の `Var` 合成で積む。`QrQ` は `dR = 0`、`QrR` は `dQ = 0` の部分寄与を
//! ノードごとに別々に計算し、`accumulate` が合算する（まとめると丸めが変わる）。`m×m` を作らない結合順
//! （`gᵀ Q` は `n×n`、`Q C` は `m×n`）にする。積む Op は `Transpose`・`Contiguous`・`MatMul`・`Where`・
//! `Add`・`ScalarUnary(Neg)`・`Solve` と自己再生の `QrQ`／`QrR` で、すべて `supports_create_graph() == true`。
//!
//! **拒否条件**: 子テープへ書く前に [`check_qr_pair`] が、push 順序契約の破れ・因子の不整合・
//! 横長（`m < n`。1 階も未対応）・空分解・rank 落ち（`R` の特異）を拒否する。構造を積む前に
//! [`guard_qr`] が総括判定（`g` の非有限・段の overflow／ゼロ化域・桁落ち・二階の上界）で拒否する。
//! いずれも `AutodiffError::Backward` で、呼び出し側のロールバックにより `child` は空へ戻る。
//!
//! 保証範囲: 受理した入力で `cg.grad` は 1 階と bit 一致し、二階と孫テープは有限差分と REQ-2 判定で
//! 一致する（seed の各要素の絶対値が 1 以下）。GPU は opt-in ゲートの後ろのカーネルが既定 OFF でホストへ
//! フォールバックするため CPU と同値の想定だが、実機の値は未実測（#3857・#3859）。

use fandhe_ai_tensor_core::{BackendOps, Tensor};

use super::linalg_fft::{
    abs_vec, all_stage_ok, inverse_of, max_abs, mm, rounding_ok, second_order_ok, to_f64, transpose,
};
use super::{get_mirror, push_v3, value_of};
use crate::error::AutodiffError;
use crate::eval::linalg::qr_vjp;
use crate::tape::{NodeId, Op, Tape, TapeNode, materialize_fallible};
use crate::var::Var;

fn reject(what: &str) -> AutodiffError {
    AutodiffError::Backward(format!(
        "create_graph: {what}（QrQ／QrR の対象外入力。m ≥ n かつフルランクの入力だけが対象。\
         決定記録 §44・§67。拒否は保守的で、実際に破綻するとは限らない）"
    ))
}

/// 子テープ上で `base` から `offset` ずれた兄弟ノードを取り出す（push 順序契約。#3808 の SVD・Eigh も使う）。
///
/// 範囲外・Op の不一致・`input` の不一致は `Err(Backward)`。`expect` は兄弟の `Op` と、その `input` が
/// `input_cid` と一致するかを判定する。
pub(super) fn sibling_of<'c>(
    child: &'c Tape,
    base: &Var<'c>,
    offset: isize,
    input_cid: NodeId,
    expect: impl Fn(&Op, NodeId) -> bool,
) -> Result<Var<'c>, AutodiffError> {
    let idx = base.node_id().0 as isize + offset;
    let nodes = child.nodes.borrow();
    if idx < 0 || idx as usize >= nodes.len() || !expect(&nodes[idx as usize].op, input_cid) {
        return Err(reject(
            "兄弟ノードが push 順序契約（QrQ の直後に QrR）を満たさない",
        ));
    }
    drop(nodes);
    Ok(Var::from_raw(child, NodeId(idx as usize)))
}

fn is_qr_r_of(op: &Op, input: NodeId) -> bool {
    matches!(op, Op::QrR { input: i, .. } if *i == input)
}

fn is_qr_q_of(op: &Op, input: NodeId) -> bool {
    matches!(op, Op::QrQ { input: i, .. } if *i == input)
}

fn bit_equal(a: &Tensor<f32>, b: &Tensor<f32>) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    let (x, y) = (to_f64(a), to_f64(b));
    x.len() == y.len()
        && x.iter()
            .zip(&y)
            .all(|(p, q)| (*p as f32).to_bits() == (*q as f32).to_bits())
}

/// 親テープ側の入口検査（`validate_ancestors` から、子テープへ書く前に呼ばれる）。
/// `id` は `QrQ`／`QrR` ノード、`op` はその Op。
pub(super) fn check_qr_pair(
    nodes: &[TapeNode],
    ops: &dyn BackendOps,
    id: NodeId,
    op: &Op,
) -> Result<(), AutodiffError> {
    let (input, q_id, r_id) = match op {
        Op::QrQ { input, .. } => (*input, id.0, id.0 + 1),
        Op::QrR { input, .. } => {
            if id.0 == 0 {
                return Err(reject("QrR の直前に QrQ が無い"));
            }
            (*input, id.0 - 1, id.0)
        }
        _ => return Ok(()),
    };
    if r_id >= nodes.len()
        || !is_qr_q_of(&nodes[q_id].op, input)
        || !is_qr_r_of(&nodes[r_id].op, input)
    {
        return Err(reject(
            "QrQ／QrR が push 順序契約（連続した対で同じ input）を満たさない",
        ));
    }
    // 因子の整合: payload（兄弟の forward 値）が兄弟ノードの記録値と bit 一致すること。
    let q_val = materialize_fallible(nodes, ops, NodeId(q_id))?;
    let r_val = materialize_fallible(nodes, ops, NodeId(r_id))?;
    let (Op::QrQ { r: r_payload, .. }, Op::QrR { q: q_payload, .. }) =
        (&nodes[q_id].op, &nodes[r_id].op)
    else {
        return Err(reject("QrQ／QrR の対が不正"));
    };
    if !bit_equal(r_payload, r_val) || !bit_equal(q_payload, q_val) {
        return Err(reject(
            "QrQ／QrR の payload が兄弟ノードの記録値と一致しない",
        ));
    }
    let in_shape = &nodes[input.0].shape;
    if in_shape.len() != 2 {
        return Err(reject("入力が rank 2 でない"));
    }
    let (m, n) = (in_shape[0], in_shape[1]);
    if m < n {
        return Err(reject("横長（m < n）は 1 階の逆伝播も未対応"));
    }
    if n == 0 {
        return Err(reject("空の分解（m == 0 または n == 0）"));
    }
    if r_val.shape() != [n, n] {
        return Err(reject("R の shape が [n,n] でない"));
    }
    // rank 落ち: R の対角が 0、または R が f64 で特異。
    let rv = to_f64(r_val);
    if (0..n).any(|i| rv[i * n + i] == 0.0) {
        return Err(reject("R の対角に 0 がある（rank 落ち）"));
    }
    inverse_of(r_val, "QR")?;
    Ok(())
}

/// `QrQ`／`QrR` の再生（`build_mirror` 段 2 から）。`qr()` は 1 回しか呼ばない。
/// 兄弟がすでに再生済み（昇順走査で `QrQ` が先）なら、その隣接ノードから導く。
pub(super) fn replay_qr<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    id: NodeId,
    mirror: &[Option<Var<'c>>],
) -> Result<Var<'c>, AutodiffError> {
    match &parent_nodes[id.0].op {
        Op::QrQ { input, .. } => Ok(get_mirror(mirror, *input)?.qr()?.q),
        Op::QrR { input, .. } => {
            let in_m = get_mirror(mirror, *input)?;
            if id.0 >= 1
                && is_qr_q_of(&parent_nodes[id.0 - 1].op, *input)
                && let Some(q_m) = mirror[id.0 - 1]
            {
                return sibling_of(child, &q_m, 1, in_m.node_id(), is_qr_r_of);
            }
            Ok(in_m.qr()?.r)
        }
        _ => Err(reject(
            "replay_qr に QrQ／QrR 以外が渡された（内部契約違反）",
        )),
    }
}

// ---------------------------------------------------------------------
// 総括的な適用判定（f64 参照）
// ---------------------------------------------------------------------

/// 行優先 `[n,n]` の `copyltu`（下三角〈対角を含む〉を上三角へ複製して対称化）。
fn copyltu(m: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = if i >= j { m[i * n + j] } else { m[j * n + i] };
        }
    }
    out
}

/// `QrQ`（`is_q_node = true`）／`QrR`（`false`）の判定。`q: [m,n]`・`r: [n,n]`、`g` は上流
/// （`QrQ` は `[m,n]`、`QrR` は `[n,n]`）。`m×m` を確保しない結合順で `qr_cgrad` と同じ段を調べる。
pub(super) fn guard_qr(
    q: &Tensor<f32>,
    r: &Tensor<f32>,
    g: &Tensor<f32>,
    is_q_node: bool,
) -> Result<(), AutodiffError> {
    let (m, n) = (q.shape()[0], q.shape()[1]);
    if n == 0 || m < n {
        return Err(reject("空の分解または横長"));
    }
    let gv = to_f64(g);
    if !gv.iter().all(|x| x.is_finite()) {
        return Err(reject("QR: 上流 g に非有限要素がある"));
    }
    let (qv, rv) = (to_f64(q), to_f64(r));
    let (x, _) = inverse_of(r, "QR")?; // R^{-1}
    let xt = transpose(&x, n, n);
    let (mval, mag_m): (Vec<f64>, Vec<f64>) = if is_q_node {
        let gt = transpose(&gv, m, n);
        let gtq = mm(&gt, &qv, n, m, n);
        let mag = mm(&abs_vec(&gt), &abs_vec(&qv), n, m, n);
        (gtq.iter().map(|v| -v).collect(), mag)
    } else {
        let gt = transpose(&gv, n, n);
        (
            mm(&rv, &gt, n, n, n),
            mm(&abs_vec(&rv), &abs_vec(&gt), n, n, n),
        )
    };
    let c = copyltu(&mval, n);
    let mag_c = copyltu(&mag_m, n);
    let qc = mm(&qv, &c, m, n, n);
    let mag_qc = mm(&abs_vec(&qv), &mag_c, m, n, n);
    let (inner, mag_inner): (Vec<f64>, Vec<f64>) = if is_q_node {
        (
            gv.iter().zip(&qc).map(|(a, b)| a + b).collect(),
            gv.iter().zip(&mag_qc).map(|(a, b)| a.abs() + b).collect(),
        )
    } else {
        (qc.clone(), mag_qc)
    };
    let da = mm(&inner, &xt, m, n, n);
    let mag_da = mm(&mag_inner, &abs_vec(&xt), m, n, n);
    if !(all_stage_ok(&x)
        && all_stage_ok(&mval)
        && all_stage_ok(&c)
        && all_stage_ok(&qc)
        && all_stage_ok(&inner)
        && all_stage_ok(&da))
    {
        return Err(reject("QR: 構造の段が overflow／ゼロ化域に入る"));
    }
    // 積和の段長: `QrQ` は gᵀQ（長さ m）・QC（n）・solve（n）、`QrR` は R gᵀ・QC・solve（各 n）。
    let terms = if is_q_node { m + 2 * n } else { 3 * n };
    if !rounding_ok(&da, &mag_da, terms) {
        return Err(reject(
            "QR: f32 構造の丸め誤差が REQ-2 の許容を超えうる（桁落ち）",
        ));
    }
    let (bx, br, bq) = (
        n as f64 * max_abs(&x),
        n as f64 * max_abs(&rv),
        m as f64 * max_abs(&qv),
    );
    if !second_order_ok(&[bx, br, bq], 4, max_abs(&gv).max(1.0)) {
        return Err(reject("QR: 二階の逆伝播の上界が 2^125 以上"));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// 1 階 VJP の子テープ記録（build_cgrads から呼ばれる）
// ---------------------------------------------------------------------

/// `QrQ`／`QrR` ノード `id` の d_input（V3）。構造を積む前に [`guard_qr`] で判定する。
/// 兄弟の写しは子テープ上の隣接ノードから取る。
pub(super) fn qr_cgrad<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    mirror: &[Option<Var<'c>>],
    id: NodeId,
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let node_op = parent_nodes[id.0].op.clone();
    let (input, is_q_node) = match &node_op {
        Op::QrQ { input, .. } => (*input, true),
        Op::QrR { input, .. } => (*input, false),
        _ => {
            return Err(reject(
                "qr_cgrad に QrQ／QrR 以外が渡された（内部契約違反）",
            ));
        }
    };
    let in_cid = get_mirror(mirror, input)?.node_id();
    let own = materialize_fallible(parent_nodes, parent_ops, id)?;
    let (q_val, r_val, q_m, r_m) = match &node_op {
        Op::QrQ { r, .. } => {
            let q_m = get_mirror(mirror, id)?;
            let r_m = sibling_of(child, &q_m, 1, in_cid, is_qr_r_of)?;
            (own.clone(), r.clone(), q_m, r_m)
        }
        Op::QrR { q, .. } => {
            let r_m = get_mirror(mirror, id)?;
            let q_m = sibling_of(child, &r_m, -1, in_cid, is_qr_q_of)?;
            (q.clone(), own.clone(), q_m, r_m)
        }
        _ => return Err(reject("内部契約違反")),
    };
    let g_val = value_of(child, g)?;
    guard_qr(&q_val, &r_val, &g_val, is_q_node)?;
    // 値（V3）: 1 階 `grad.rs` と逐語で同じ呼び出し。
    let host = if is_q_node {
        let dr_zero = Tensor::<f32>::zeros(r_val.shape())?;
        qr_vjp(&q_val, &r_val, &g_val, &dr_zero)?
    } else {
        let dq_zero = Tensor::<f32>::zeros(q_val.shape())?;
        qr_vjp(&q_val, &r_val, &dq_zero, &g_val)?
    };
    let n = r_val.shape()[1];
    let gc = g.contiguous()?;
    let gt = gc.transpose(0, 1)?.contiguous()?;
    let mval = if is_q_node {
        gt.matmul_fp32_strict(&q_m)?.neg()?
    } else {
        r_m.matmul_fp32_strict(&gt)?
    };
    let upper: Vec<bool> = (0..n * n).map(|i| i % n > i / n).collect();
    let upper = Tensor::new(upper, &[n, n])?;
    let mt = mval.transpose(0, 1)?.contiguous()?;
    let c = Var::where_cond(&upper, &mt, &mval)?;
    let qc = q_m.matmul_fp32_strict(&c)?;
    let inner = if is_q_node { gc.add(&qc)? } else { qc };
    let x = r_m.solve(&inner.transpose(0, 1)?.contiguous()?)?;
    let composed = x.transpose(0, 1)?.contiguous()?;
    push_v3(child, &composed, host)
}
#[cfg(test)]
mod tests {
    use super::*;

    fn data(m: usize, n: usize, scale: f32) -> Tensor<f32> {
        let v: Vec<f32> = (0..m * n)
            .map(|i| {
                (1.0 + ((i * 7) % 11) as f32 * 0.37) * scale
                    + if i / n == i % n { 3.0 } else { 0.0 }
            })
            .collect();
        Tensor::new(v, &[m, n]).unwrap()
    }

    /// 親テープに `QrQ`／`QrR` を手で積み、push 順序契約を破った 3 通りが入口で拒否され、
    /// `child` が空のまま保たれること。
    #[test]
    fn push_order_violations_are_rejected_before_child_write() {
        for case in ["gap", "other-input", "orphan-r"] {
            let tape = Tape::new();
            let child = Tape::new();
            let a = data(3, 2, 1.0);
            let x = tape.var(&a);
            let y = tape.var(&a);
            let (q, r) = crate::eval::linalg::qr(&a).unwrap();
            let rid = match case {
                "gap" => {
                    tape.push_eager(
                        Op::QrQ {
                            input: x.node_id(),
                            r: r.clone(),
                        },
                        q.clone(),
                    );
                    let _ = x.exp();
                    tape.push_eager(
                        Op::QrR {
                            input: x.node_id(),
                            q,
                        },
                        r,
                    )
                }
                "other-input" => {
                    tape.push_eager(
                        Op::QrQ {
                            input: x.node_id(),
                            r: r.clone(),
                        },
                        q.clone(),
                    );
                    tape.push_eager(
                        Op::QrR {
                            input: y.node_id(),
                            q,
                        },
                        r,
                    )
                }
                _ => tape.push_eager(
                    Op::QrR {
                        input: x.node_id(),
                        q,
                    },
                    r,
                ),
            };
            let loss = Var::from_raw(&tape, rid).sum(None).unwrap();
            let res = tape.backward_create_graph(&loss, &child);
            assert!(
                matches!(res, Err(AutodiffError::Backward(_))),
                "{case}: Err(Backward) のはず"
            );
            assert!(child.is_empty(), "{case}: child は空");
        }
    }

    /// 兄弟が祖先に無い場合・両方が祖先の場合のいずれでも `qr()` は 1 回しか呼ばれない
    /// （子テープの `QrQ`・`QrR` ノードは各 1 個）。
    #[test]
    fn replay_calls_qr_once() {
        for which in ["q", "r", "both"] {
            let tape = Tape::new();
            let child = Tape::new();
            let x = tape.var(&data(4, 2, 1.0));
            let f = x.qr().unwrap();
            let loss = match which {
                "q" => f.q.mul(&f.q).unwrap().sum(None).unwrap(),
                "r" => f.r.mul(&f.r).unwrap().sum(None).unwrap(),
                _ => {
                    f.q.sum(None)
                        .unwrap()
                        .add(&f.r.mul(&f.r).unwrap().sum(None).unwrap())
                        .unwrap()
                }
            };
            let _cg = tape.backward_create_graph(&loss, &child).unwrap();
            let nodes = child.nodes.borrow();
            let qs = nodes
                .iter()
                .filter(|n| matches!(n.op, Op::QrQ { .. }))
                .count();
            let rs = nodes
                .iter()
                .filter(|n| matches!(n.op, Op::QrR { .. }))
                .count();
            assert_eq!((qs, rs), (1, 1), "{which}: qr() は 1 回だけ");
        }
    }

    /// 子テープに `m×m`（長辺側の正方行列）のノードを作らない（`[100000,1]` で 10^10 要素を避ける）。
    #[test]
    fn qr_cgrad_has_no_long_side_square_node() {
        for (m, n, scale) in [(100_000usize, 1usize, 100.0f32), (5, 2, 1.0)] {
            let tape = Tape::new();
            let child = Tape::new();
            let x = tape.var(&data(m, n, scale));
            let f = x.qr().unwrap();
            let weights: Vec<f32> = (0..m * n).map(|i| ((i as f32) * 0.9).sin() + 0.2).collect();
            let cq = tape.var_no_grad(&Tensor::new(weights, &[m, n]).unwrap());
            let loss = f.q.mul(&cq).unwrap().sum(None).unwrap();
            tape.backward_create_graph(&loss, &child)
                .unwrap_or_else(|e| panic!("({m},{n}): 受理されるはず: {e}"));
            let long_sq = m * m;
            let nodes = child.nodes.borrow();
            assert!(
                nodes
                    .iter()
                    .all(|node| node.shape.iter().product::<usize>() < long_sq),
                "({m},{n}): m×m（{long_sq} 要素以上）のノードがある"
            );
        }
    }
}
