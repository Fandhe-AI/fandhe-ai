//! 分解族の多出力ノードのうち `QrQ`・`QrR` の `create_graph` 対応（イシュー #3807・親 #3734・
//! 決定記録 `docs/autodiff-higher-order-grad-decision.md` §44・§67）。後半に `SvdU`・`SvdS`・`SvdVh`・
//! `EighValues`・`EighVectors` の対応（イシュー #3808・§68。QR と同じ方式。縮退・σ == 0・空分解は拒否し、
//! 横長 SVD は受理する）を同居させる。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `validate_ancestors`（入口検査 [`check_qr_pair`]）・
//! `build_mirror` 段 2（再生 [`replay_qr`]）・`build_cgrads`（1 階 VJP の子テープ記録 [`qr_cgrad`]）
//! から呼ばれる。`Svd*`・`Eigh*`（#3808・§68）も [`sibling_of`] を共用する（本ファイル後半）。
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
use crate::eval::linalg::{EIGH_DEGENERATE_RTOL, eigh_vjp, qr_vjp, svd_vjp};
use crate::tape::{NodeId, Op, Tape, TapeNode, materialize_fallible};
use crate::var::Var;

fn reject(what: &str) -> AutodiffError {
    AutodiffError::Backward(format!(
        "create_graph: {what}（分解族〈QrQ／QrR・SvdU／SvdS／SvdVh・EighValues／EighVectors〉の\
         対象外入力。決定記録 §44・§67・§68。拒否は保守的で、実際に破綻するとは限らない）"
    ))
}

/// 子テープ上で `base` から `offset` ずれた兄弟ノードを取り出す（push 順序契約。QR・SVD・Eigh 共用）。
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
            "兄弟ノードが push 順序契約（QrQ→QrR・SvdU→SvdS→SvdVh・EighValues→EighVectors の連続 push）を満たさない",
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

// =====================================================================
// SvdU・SvdS・SvdVh・EighValues・EighVectors（イシュー #3808・決定記録 §44・§68）
// =====================================================================
//
// QR と同じ方式（push 順序契約・公開メソッドへの自己再生・V3 の値 + f32 の `Var` 合成構造・
// 事前拒否）を SVD・Eigh へ広げる。SVD は `SvdU`→`SvdS`→`SvdVh`（`Var::svd`）、Eigh は
// `EighValues`→`EighVectors`（`linalg_ops::eigh`）が連続 push される。縮退（SVD は 1 階と同じ絶対判定
// 1e-9、Eigh は `EIGH_DEGENERATE_RTOL` の相対判定）と SVD の σ == 0 は、兄弟のどれを使う損失でも
// 拒否する（1 階の Eigh は固有値だけの損失なら縮退でも受理するが、二階は `1/E²` で破綻するため）。

fn is_svd_u_of(op: &Op, input: NodeId) -> bool {
    matches!(op, Op::SvdU { input: i, .. } if *i == input)
}

fn is_svd_s_of(op: &Op, input: NodeId) -> bool {
    matches!(op, Op::SvdS { input: i, .. } if *i == input)
}

fn is_svd_vh_of(op: &Op, input: NodeId) -> bool {
    matches!(op, Op::SvdVh { input: i, .. } if *i == input)
}

fn is_eigh_values_of(op: &Op, input: NodeId) -> bool {
    matches!(op, Op::EighValues { input: i, .. } if *i == input)
}

fn is_eigh_vectors_of(op: &Op, input: NodeId) -> bool {
    matches!(op, Op::EighVectors { input: i, .. } if *i == input)
}

/// SVD 3 ノードのうち自ノードの位置（`SvdU` = 0・`SvdS` = 1・`SvdVh` = 2）と `input`。
fn svd_position(op: &Op) -> Option<(NodeId, usize)> {
    match op {
        Op::SvdU { input, .. } => Some((*input, 0)),
        Op::SvdS { input, .. } => Some((*input, 1)),
        Op::SvdVh { input, .. } => Some((*input, 2)),
        _ => None,
    }
}

/// 1 階 `eval::linalg::svd_vjp` と同じ絶対判定（`|σj² − σi²| < 1e-9`。同関数の F 行列の分母検査）。
/// 定数は新設せず述語だけを写す（承認 §44 点 4）。単体テストで `svd_vjp` との一致を固定する。
pub(super) fn svd_gap_degenerate(s: &[f64]) -> bool {
    let k = s.len();
    (0..k).any(|i| (0..k).any(|j| i != j && (s[j] * s[j] - s[i] * s[i]).abs() < 1e-9))
}

/// 1 階 `eval::linalg::eigh_vjp` と同じ相対判定（`|λj − λi| <= RTOL · max|λ|`）。
fn eigh_gap_degenerate(lambda: &[f64]) -> bool {
    let n = lambda.len();
    let max_abs_lambda = lambda.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
    let threshold = EIGH_DEGENERATE_RTOL * max_abs_lambda.max(f64::MIN_POSITIVE);
    (0..n).any(|i| (0..n).any(|j| i != j && (lambda[j] - lambda[i]).abs() <= threshold))
}

/// SVD の親側入口検査（`validate_ancestors` から、子テープへ書く前に呼ばれる）。
/// 3 ノードの連続・同一 input・因子の bit 整合・空分解・縮退・σ == 0 を拒否する。横長（`m < n`）は
/// forward が転置で対応するため拒否しない（QR との非対称。§68）。
pub(super) fn check_svd_triple(
    nodes: &[TapeNode],
    ops: &dyn BackendOps,
    id: NodeId,
    op: &Op,
) -> Result<(), AutodiffError> {
    let Some((input, pos)) = svd_position(op) else {
        return Ok(());
    };
    if id.0 < pos {
        return Err(reject("SvdS／SvdVh の直前に SvdU が無い"));
    }
    let u_id = id.0 - pos;
    if u_id + 2 >= nodes.len()
        || !is_svd_u_of(&nodes[u_id].op, input)
        || !is_svd_s_of(&nodes[u_id + 1].op, input)
        || !is_svd_vh_of(&nodes[u_id + 2].op, input)
    {
        return Err(reject(
            "SvdU／SvdS／SvdVh が push 順序契約（連続した 3 ノードで同じ input）を満たさない",
        ));
    }
    let u_val = materialize_fallible(nodes, ops, NodeId(u_id))?;
    let s_val = materialize_fallible(nodes, ops, NodeId(u_id + 1))?;
    let vh_val = materialize_fallible(nodes, ops, NodeId(u_id + 2))?;
    let (
        Op::SvdU {
            s: s_p, vh: vh_p, ..
        },
        Op::SvdS {
            u: u_p, vh: vh_p2, ..
        },
        Op::SvdVh {
            u: u_p3, s: s_p3, ..
        },
    ) = (&nodes[u_id].op, &nodes[u_id + 1].op, &nodes[u_id + 2].op)
    else {
        return Err(reject("SVD の 3 ノードが不正"));
    };
    if !(bit_equal(s_p, s_val)
        && bit_equal(vh_p, vh_val)
        && bit_equal(u_p, u_val)
        && bit_equal(vh_p2, vh_val)
        && bit_equal(u_p3, u_val)
        && bit_equal(s_p3, s_val))
    {
        return Err(reject("SVD の payload が兄弟ノードの記録値と一致しない"));
    }
    let in_shape = &nodes[input.0].shape;
    if in_shape.len() != 2 {
        return Err(reject("入力が rank 2 でない"));
    }
    let (m, n) = (in_shape[0], in_shape[1]);
    let k = m.min(n);
    if k == 0 {
        return Err(reject("空の分解（m == 0 または n == 0）"));
    }
    if u_val.shape() != [m, k] || s_val.shape() != [k] || vh_val.shape() != [k, n] {
        return Err(reject("SVD の因子 shape が [m,k]・[k]・[k,n] でない"));
    }
    let s = to_f64(s_val);
    if s.contains(&0.0) {
        return Err(reject(
            "σ == 0 がある（二階は 1/σ が定義されない。1 階は 0 として扱う）",
        ));
    }
    if svd_gap_degenerate(&s) {
        return Err(reject(
            "特異値が近接／重複している（|σj² − σi²| < 1e-9。1 階の svd_vjp と同じ判定）",
        ));
    }
    Ok(())
}

/// Eigh の親側入口検査。2 ノードの連続・同一 input・因子の bit 整合・空分解・縮退を拒否する。
pub(super) fn check_eigh_pair(
    nodes: &[TapeNode],
    ops: &dyn BackendOps,
    id: NodeId,
    op: &Op,
) -> Result<(), AutodiffError> {
    let (input, v_id) = match op {
        Op::EighValues { input, .. } => (*input, id.0),
        Op::EighVectors { input, .. } => {
            if id.0 == 0 {
                return Err(reject("EighVectors の直前に EighValues が無い"));
            }
            (*input, id.0 - 1)
        }
        _ => return Ok(()),
    };
    if v_id + 1 >= nodes.len()
        || !is_eigh_values_of(&nodes[v_id].op, input)
        || !is_eigh_vectors_of(&nodes[v_id + 1].op, input)
    {
        return Err(reject(
            "EighValues／EighVectors が push 順序契約（連続した対で同じ input）を満たさない",
        ));
    }
    let vals = materialize_fallible(nodes, ops, NodeId(v_id))?;
    let vecs = materialize_fallible(nodes, ops, NodeId(v_id + 1))?;
    let (Op::EighValues { vectors: vec_p, .. }, Op::EighVectors { values: val_p, .. }) =
        (&nodes[v_id].op, &nodes[v_id + 1].op)
    else {
        return Err(reject("EighValues／EighVectors の対が不正"));
    };
    if !bit_equal(vec_p, vecs) || !bit_equal(val_p, vals) {
        return Err(reject(
            "EighValues／EighVectors の payload が兄弟ノードの記録値と一致しない",
        ));
    }
    let in_shape = &nodes[input.0].shape;
    if in_shape.len() != 2 || in_shape[0] != in_shape[1] {
        return Err(reject("入力が正方 rank 2 でない"));
    }
    let n = in_shape[0];
    if n == 0 {
        return Err(reject("空の分解（n == 0）"));
    }
    if vals.shape() != [n] || vecs.shape() != [n, n] {
        return Err(reject("Eigh の因子 shape が [n]・[n,n] でない"));
    }
    if eigh_gap_degenerate(&to_f64(vals)) {
        return Err(reject(
            "固有値が縮退している（|λj − λi| <= EIGH_DEGENERATE_RTOL · max|λ|。固有値だけの損失でも拒否する）",
        ));
    }
    Ok(())
}

/// `SvdU`／`SvdS`／`SvdVh` の再生。`svd()` は 1 回しか呼ばない（より前の兄弟の写しがあれば導く）。
pub(super) fn replay_svd<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    id: NodeId,
    mirror: &[Option<Var<'c>>],
) -> Result<Var<'c>, AutodiffError> {
    let Some((input, pos)) = svd_position(&parent_nodes[id.0].op) else {
        return Err(reject("replay_svd に SVD 以外が渡された（内部契約違反）"));
    };
    let in_m = get_mirror(mirror, input)?;
    let expect = match pos {
        0 => is_svd_u_of,
        1 => is_svd_s_of,
        _ => is_svd_vh_of,
    };
    for back in 1..=pos {
        if id.0 >= back
            && let Some(earlier) = mirror[id.0 - back]
        {
            return sibling_of(child, &earlier, back as isize, in_m.node_id(), expect);
        }
    }
    let f = in_m.svd()?;
    Ok(match pos {
        0 => f.u,
        1 => f.s,
        _ => f.vh,
    })
}

/// `EighValues`／`EighVectors` の再生。`eigh()` は 1 回しか呼ばない。
pub(super) fn replay_eigh<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    id: NodeId,
    mirror: &[Option<Var<'c>>],
) -> Result<Var<'c>, AutodiffError> {
    match &parent_nodes[id.0].op {
        Op::EighValues { input, .. } => Ok(get_mirror(mirror, *input)?.eigh()?.eigenvalues),
        Op::EighVectors { input, .. } => {
            let in_m = get_mirror(mirror, *input)?;
            if id.0 >= 1
                && is_eigh_values_of(&parent_nodes[id.0 - 1].op, *input)
                && let Some(v_m) = mirror[id.0 - 1]
            {
                return sibling_of(child, &v_m, 1, in_m.node_id(), is_eigh_vectors_of);
            }
            Ok(in_m.eigh()?.eigenvectors)
        }
        _ => Err(reject(
            "replay_eigh に EighValues／EighVectors 以外が渡された（内部契約違反）",
        )),
    }
}

// ---------------------------------------------------------------------
// 総括的な適用判定（f64 参照）
// ---------------------------------------------------------------------

fn add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn sub_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn mul_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

/// 行優先 `[r,c]` の各列 `j` を `w[j]` 倍する。
fn scale_cols(a: &[f64], r: usize, c: usize, w: &[f64]) -> Vec<f64> {
    (0..r * c).map(|i| a[i] * w[i % c]).collect()
}

/// 行優先 `[r,c]` の各行 `i` を `w[i]` 倍する。
fn scale_rows(a: &[f64], r: usize, c: usize, w: &[f64]) -> Vec<f64> {
    (0..r * c).map(|i| a[i] * w[i / c]).collect()
}

/// `F_ij = 1/(σj² − σi²)`（`i ≠ j`）、対角 0。縮退は入口検査で拒否済みのため分母は非零。
fn svd_f_matrix(s: &[f64]) -> Vec<f64> {
    let k = s.len();
    let mut f = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            if i != j {
                f[i * k + j] = 1.0 / (s[j] * s[j] - s[i] * s[i]);
            }
        }
    }
    f
}

/// `SvdU`／`SvdS`／`SvdVh`（`pos` = 0／1／2）の判定。`u: [m,k]`・`s: [k]`・`vh: [k,n]`、`g` は上流
/// （`SvdU` は `[m,k]`、`SvdS` は `[k]`、`SvdVh` は `[k,n]`）。`m×m`・`n×n` を確保しない結合順で
/// [`svd_cgrad`] と同じ段を調べる。
pub(super) fn guard_svd(
    u: &Tensor<f32>,
    s: &Tensor<f32>,
    vh: &Tensor<f32>,
    g: &Tensor<f32>,
    pos: usize,
) -> Result<(), AutodiffError> {
    let (m, k) = (u.shape()[0], u.shape()[1]);
    let n = vh.shape()[1];
    if k == 0 {
        return Err(reject("空の分解"));
    }
    let gv = to_f64(g);
    if !gv.iter().all(|x| x.is_finite()) {
        return Err(reject("SVD: 上流 g に非有限要素がある"));
    }
    let (uv, sv, vhv) = (to_f64(u), to_f64(s), to_f64(vh));
    let (au, ash, avh) = (abs_vec(&uv), abs_vec(&sv), abs_vec(&vhv));
    let f = svd_f_matrix(&sv);
    let af = abs_vec(&f);
    let inv_s: Vec<f64> = sv.iter().map(|x| 1.0 / x).collect();
    let ainv_s = abs_vec(&inv_s);
    let agv = abs_vec(&gv);
    let utr = transpose(&uv, m, k);
    let autr = abs_vec(&utr);
    let (stages_ok, da, mag_da, terms): (bool, Vec<f64>, Vec<f64>, usize) = match pos {
        0 => {
            let utg = mm(&utr, &gv, k, m, k);
            let mag_utg = mm(&autr, &agv, k, m, k);
            let jraw = sub_vec(&utg, &transpose(&utg, k, k));
            let mag_jraw = add_vec(&mag_utg, &transpose(&mag_utg, k, k));
            let j = mul_vec(&jraw, &f);
            let inner = scale_cols(&j, k, k, &sv);
            let mag_inner = scale_cols(&mul_vec(&mag_jraw, &af), k, k, &ash);
            let term1 = mm(&mm(&uv, &inner, m, k, k), &vhv, m, k, n);
            let mag_t1 = mm(&mm(&au, &mag_inner, m, k, k), &avh, m, k, n);
            let gs = scale_cols(&gv, m, k, &inv_s);
            let t2a = mm(&gs, &vhv, m, k, n);
            let ut_gs = mm(&utr, &gs, k, m, k);
            let t2b = mm(&mm(&uv, &ut_gs, m, k, k), &vhv, m, k, n);
            let mag_gs = scale_cols(&agv, m, k, &ainv_s);
            let mag_t2a = mm(&mag_gs, &avh, m, k, n);
            let mag_utgs = mm(&autr, &mag_gs, k, m, k);
            let mag_t2b = mm(&mm(&au, &mag_utgs, m, k, k), &avh, m, k, n);
            let da = sub_vec(&add_vec(&term1, &t2a), &t2b);
            let mag = add_vec(&add_vec(&mag_t1, &mag_t2a), &mag_t2b);
            (
                all_stage_ok(&f)
                    && all_stage_ok(&j)
                    && all_stage_ok(&inner)
                    && all_stage_ok(&term1)
                    && all_stage_ok(&gs)
                    && all_stage_ok(&t2a)
                    && all_stage_ok(&t2b)
                    && all_stage_ok(&da),
                da,
                mag,
                m + 4,
            )
        }
        1 => {
            let ug = scale_cols(&uv, m, k, &gv);
            let da = mm(&ug, &vhv, m, k, n);
            let mag = mm(&scale_cols(&au, m, k, &agv), &avh, m, k, n);
            (all_stage_ok(&ug) && all_stage_ok(&da), da, mag, k + 1)
        }
        _ => {
            let dv = transpose(&gv, k, n);
            let vv = transpose(&vhv, k, n);
            let vdv = mm(&vhv, &dv, k, n, k);
            let mag_vdv = mm(&avh, &abs_vec(&dv), k, n, k);
            let kraw = sub_vec(&vdv, &transpose(&vdv, k, k));
            let mag_kraw = add_vec(&mag_vdv, &transpose(&mag_vdv, k, k));
            let kk = mul_vec(&kraw, &f);
            let inner = scale_rows(&kk, k, k, &sv);
            let mag_inner = scale_rows(&mul_vec(&mag_kraw, &af), k, k, &ash);
            let term1 = mm(&mm(&uv, &inner, m, k, k), &vhv, m, k, n);
            let mag_t1 = mm(&mm(&au, &mag_inner, m, k, k), &avh, m, k, n);
            let sg = scale_rows(&gv, k, n, &inv_s);
            let t3a = mm(&uv, &sg, m, k, n);
            let sgv = mm(&sg, &vv, k, n, k);
            let t3b = mm(&mm(&uv, &sgv, m, k, k), &vhv, m, k, n);
            let mag_sg = scale_rows(&agv, k, n, &ainv_s);
            let mag_t3a = mm(&au, &mag_sg, m, k, n);
            let mag_sgv = mm(&mag_sg, &abs_vec(&vv), k, n, k);
            let mag_t3b = mm(&mm(&au, &mag_sgv, m, k, k), &avh, m, k, n);
            let da = sub_vec(&add_vec(&term1, &t3a), &t3b);
            let mag = add_vec(&add_vec(&mag_t1, &mag_t3a), &mag_t3b);
            (
                all_stage_ok(&f)
                    && all_stage_ok(&kk)
                    && all_stage_ok(&inner)
                    && all_stage_ok(&term1)
                    && all_stage_ok(&sg)
                    && all_stage_ok(&t3a)
                    && all_stage_ok(&t3b)
                    && all_stage_ok(&da),
                da,
                mag,
                n + 4,
            )
        }
    };
    if !stages_ok {
        return Err(reject("SVD: 構造の段が overflow／ゼロ化域に入る"));
    }
    if !rounding_ok(&da, &mag_da, terms) {
        return Err(reject(
            "SVD: f32 構造の丸め誤差が REQ-2 の許容を超えうる（桁落ち）",
        ));
    }
    // 二階の上界（保守的な見積り）: 因子は U・Vh の長辺倍の最大値、F·σ²（ギャップ項の微分）、
    // 1/σ_min、σ_max。冪は 4（U・Vh・F・σ の 4 因子が二階で 1 回ずつ増える想定）。
    let bu = m.max(n) as f64 * max_abs(&uv).max(max_abs(&vhv));
    let fa = max_abs(&f) * max_abs(&sv).max(1.0).powi(2);
    if !second_order_ok(
        &[bu, fa, max_abs(&inv_s), max_abs(&sv)],
        4,
        max_abs(&gv).max(1.0),
    ) {
        return Err(reject("SVD: 二階の逆伝播の上界が 2^125 以上"));
    }
    Ok(())
}

/// `MatrixNorm`（`Nuc`・`Spectral`）の総括判定（イシュー #3991・決定記録 §76。
/// `linalg_norm_rank::nuc_spectral_cgrad` が構造を積む前に呼ぶ）。`u: [m,k]`・`s: [k]`・`vh: [k,n]` は
/// 子テープ上の SVD 因子の値、`g_scalar` は上流スカラー。構造は `g·U Vh`（`Nuc`）／`g·u₀v₀ᵀ`
/// （`Spectral`）で、`m×m`・`n×n` を確保しない。
///
/// [`guard_svd`] はシグネチャが因子 shape の `g` 前提のためスカラー上流には使えず、同じ判定式を
/// 写した別関数にした（新しい定数は作らない）。値（V3）は 1 階と bit 一致するホスト値のため、構造の
/// f32 丸め（`rounding_ok`）は `cg.grad` の値に効かず、逆伝播の経路だけを決める。よって `rounding_ok`
/// は使わず、段の有限性（`all_stage_ok`）と二階の上界（`second_order_ok`。`guard_svd` と同じ 4 因子）
/// だけを見る。
pub(super) fn guard_matrix_norm_svd(
    u: &Tensor<f32>,
    s: &Tensor<f32>,
    vh: &Tensor<f32>,
    g_scalar: f32,
    spectral: bool,
) -> Result<(), AutodiffError> {
    let (m, k) = (u.shape()[0], u.shape()[1]);
    let n = vh.shape()[1];
    if k == 0 {
        return Err(reject("空の分解"));
    }
    if !g_scalar.is_finite() {
        return Err(reject("MatrixNorm: 上流 g が非有限"));
    }
    let g = f64::from(g_scalar);
    let (uv, sv, vhv) = (to_f64(u), to_f64(s), to_f64(vh));
    let p = if spectral {
        // u₀ v₀ᵀ: 先頭列 × 先頭行。
        let u0: Vec<f64> = (0..m).map(|r| uv[r * k]).collect();
        mm(&u0, &vhv[..n], m, 1, n)
    } else {
        mm(&uv, &vhv, m, k, n)
    };
    let gp: Vec<f64> = p.iter().map(|x| g * x).collect();
    if !(all_stage_ok(&p) && all_stage_ok(&gp)) {
        return Err(reject("MatrixNorm: 構造の段が overflow／ゼロ化域に入る"));
    }
    let f = svd_f_matrix(&sv);
    let inv_s: Vec<f64> = sv.iter().map(|x| 1.0 / x).collect();
    let bu = m.max(n) as f64 * max_abs(&uv).max(max_abs(&vhv));
    let fa = max_abs(&f) * max_abs(&sv).max(1.0).powi(2);
    if !second_order_ok(
        &[bu, fa, max_abs(&inv_s), max_abs(&sv)],
        4,
        g.abs().max(1.0),
    ) {
        return Err(reject("MatrixNorm: 二階の逆伝播の上界が 2^125 以上"));
    }
    Ok(())
}

/// `EighValues`（`is_vectors_node = false`）／`EighVectors`（`true`）の判定。`vecs: [n,n]`・
/// `lam: [n]`、`g` は上流（`EighValues` は `[n]`、`EighVectors` は `[n,n]`）。
pub(super) fn guard_eigh(
    vecs: &Tensor<f32>,
    lam: &Tensor<f32>,
    g: &Tensor<f32>,
    is_vectors_node: bool,
) -> Result<(), AutodiffError> {
    let n = vecs.shape()[0];
    if n == 0 {
        return Err(reject("空の分解"));
    }
    let gv = to_f64(g);
    if !gv.iter().all(|x| x.is_finite()) {
        return Err(reject("Eigh: 上流 g に非有限要素がある"));
    }
    let (v, l) = (to_f64(vecs), to_f64(lam));
    let (av, vt) = (abs_vec(&v), transpose(&v, n, n));
    let avt = abs_vec(&vt);
    let (da, mag, ok, terms): (Vec<f64>, Vec<f64>, bool, usize) = if is_vectors_node {
        let x = mm(&vt, &gv, n, n, n);
        let mag_x = mm(&avt, &abs_vec(&gv), n, n, n);
        let skew: Vec<f64> = sub_vec(&x, &transpose(&x, n, n))
            .iter()
            .map(|a| 0.5 * a)
            .collect();
        let mag_skew: Vec<f64> = add_vec(&mag_x, &transpose(&mag_x, n, n))
            .iter()
            .map(|a| 0.5 * a)
            .collect();
        let mut middle = vec![0.0; n * n];
        let mut mag_middle = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let e = l[j] - l[i];
                    middle[i * n + j] = skew[i * n + j] / e;
                    mag_middle[i * n + j] = mag_skew[i * n + j] / e.abs();
                }
            }
        }
        let vm = mm(&v, &middle, n, n, n);
        let da = mm(&vm, &vt, n, n, n);
        let mag = mm(&mm(&av, &mag_middle, n, n, n), &avt, n, n, n);
        let ok =
            all_stage_ok(&x) && all_stage_ok(&middle) && all_stage_ok(&vm) && all_stage_ok(&da);
        (da, mag, ok, 3 * n + 2)
    } else {
        let vg = scale_cols(&v, n, n, &gv);
        let da = mm(&vg, &vt, n, n, n);
        let mag = mm(&scale_cols(&av, n, n, &abs_vec(&gv)), &avt, n, n, n);
        let ok = all_stage_ok(&vg) && all_stage_ok(&da);
        (da, mag, ok, n + 1)
    };
    if !ok {
        return Err(reject("Eigh: 構造の段が overflow／ゼロ化域に入る"));
    }
    if !rounding_ok(&da, &mag, terms) {
        return Err(reject(
            "Eigh: f32 構造の丸め誤差が REQ-2 の許容を超えうる（桁落ち）",
        ));
    }
    // 二階の上界（保守的な見積り）: 因子は V の長辺倍の最大値と 1/min|E|。冪は 4。
    let mut min_gap = f64::INFINITY;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                min_gap = min_gap.min((l[j] - l[i]).abs());
            }
        }
    }
    let inv_gap = if min_gap.is_finite() {
        1.0 / min_gap
    } else {
        0.0
    };
    let bv = n as f64 * max_abs(&v);
    if !second_order_ok(&[bv, inv_gap], 4, max_abs(&gv).max(1.0)) {
        return Err(reject("Eigh: 二階の逆伝播の上界が 2^125 以上"));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// 1 階 VJP の子テープ記録（build_cgrads から呼ばれる）
// ---------------------------------------------------------------------

fn bool_mask(
    rows: usize,
    cols: usize,
    f: impl Fn(usize, usize) -> bool,
) -> Result<Tensor<bool>, AutodiffError> {
    let v: Vec<bool> = (0..rows * cols).map(|i| f(i / cols, i % cols)).collect();
    Ok(Tensor::new(v, &[rows, cols])?)
}

fn const_full<'c>(child: &'c Tape, shape: &[usize], v: f32) -> Result<Var<'c>, AutodiffError> {
    let n: usize = shape.iter().product();
    Ok(child.var_no_grad(&Tensor::new(vec![v; n], shape)?))
}

/// `SvdU`／`SvdS`／`SvdVh` ノード `id` の d_input（V3）。構造を積む前に [`guard_svd`] で判定する。
/// 3 つの写し（U・σ・Vh）は子テープ上の隣接ノードから取る。
pub(super) fn svd_cgrad<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    mirror: &[Option<Var<'c>>],
    id: NodeId,
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let node_op = parent_nodes[id.0].op.clone();
    let Some((input, pos)) = svd_position(&node_op) else {
        return Err(reject("svd_cgrad に SVD 以外が渡された（内部契約違反）"));
    };
    let in_cid = get_mirror(mirror, input)?.node_id();
    let own = materialize_fallible(parent_nodes, parent_ops, id)?;
    let own_m = get_mirror(mirror, id)?;
    let (u_val, s_val, vh_val) = match &node_op {
        Op::SvdU { s, vh, .. } => (own.clone(), s.clone(), vh.clone()),
        Op::SvdS { u, vh, .. } => (u.clone(), own.clone(), vh.clone()),
        Op::SvdVh { u, s, .. } => (u.clone(), s.clone(), own.clone()),
        _ => return Err(reject("内部契約違反")),
    };
    let pick = |target: usize, expect: fn(&Op, NodeId) -> bool| -> Result<Var<'c>, AutodiffError> {
        if target == pos {
            Ok(own_m)
        } else {
            sibling_of(
                child,
                &own_m,
                target as isize - pos as isize,
                in_cid,
                expect,
            )
        }
    };
    let u_m = pick(0, is_svd_u_of)?;
    let s_m = pick(1, is_svd_s_of)?;
    let vh_m = pick(2, is_svd_vh_of)?;
    let g_val = value_of(child, g)?;
    guard_svd(&u_val, &s_val, &vh_val, &g_val, pos)?;
    // 値（V3）: 1 階 `grad.rs` と逐語で同じ呼び出し。
    let host = match pos {
        0 => svd_vjp(&u_val, &s_val, &vh_val, Some(&g_val), None, None)?,
        1 => svd_vjp(&u_val, &s_val, &vh_val, None, Some(&g_val), None)?,
        _ => svd_vjp(&u_val, &s_val, &vh_val, None, None, Some(&g_val))?,
    };
    let k = s_val.shape()[0];
    let gc = g.contiguous()?;
    let composed = if pos == 1 {
        let gb = gc.reshape(&[1, k])?;
        u_m.mul(&gb)?.matmul_fp32_strict(&vh_m)?
    } else {
        let eye = bool_mask(k, k, |i, j| i == j)?;
        let ones = const_full(child, &[k, k], 1.0)?;
        let zeros = const_full(child, &[k, k], 0.0)?;
        // 分母 `σj² − σi²` は `(σj − σi)(σj + σi)` の形で作る。f32 で二乗してから引くと、近接した
        // 特異値（例 1.400001 と 1.4）で桁落ちし F とその微分（二階）が REQ-2 を外れる。
        // 差は近接域で厳密（Sterbenz）、和・積の相対誤差は数 ulp に収まる。
        let sr = s_m.reshape(&[1, k])?;
        let sc = s_m.reshape(&[k, 1])?;
        let d = sr.sub(&sc)?.mul(&sr.add(&sc)?)?;
        let d_safe = Var::where_cond(&eye, &ones, &d)?;
        let f = Var::where_cond(&eye, &zeros, &ones.div(&d_safe)?)?;
        let inv_s = const_full(child, &[k], 1.0)?.div(&s_m)?;
        if pos == 0 {
            let ut = u_m.transpose(0, 1)?.contiguous()?;
            let utg = ut.matmul_fp32_strict(&gc)?;
            let j = utg.sub(&utg.transpose(0, 1)?.contiguous()?)?.mul(&f)?;
            let inner = j.mul(&s_m.reshape(&[1, k])?)?;
            let term1 = u_m.matmul_fp32_strict(&inner)?.matmul_fp32_strict(&vh_m)?;
            let gs = gc.mul(&inv_s.reshape(&[1, k])?)?;
            let t2a = gs.matmul_fp32_strict(&vh_m)?;
            let t2b = u_m
                .matmul_fp32_strict(&ut.matmul_fp32_strict(&gs)?)?
                .matmul_fp32_strict(&vh_m)?;
            term1.add(&t2a)?.sub(&t2b)?
        } else {
            let dv = gc.transpose(0, 1)?.contiguous()?;
            let vdv = vh_m.matmul_fp32_strict(&dv)?;
            let kk = vdv.sub(&vdv.transpose(0, 1)?.contiguous()?)?.mul(&f)?;
            let inner = kk.mul(&s_m.reshape(&[k, 1])?)?;
            let term1 = u_m.matmul_fp32_strict(&inner)?.matmul_fp32_strict(&vh_m)?;
            let sg = gc.mul(&inv_s.reshape(&[k, 1])?)?;
            let t3a = u_m.matmul_fp32_strict(&sg)?;
            let v_m = vh_m.transpose(0, 1)?.contiguous()?;
            let sgv = sg.matmul_fp32_strict(&v_m)?;
            let t3b = u_m.matmul_fp32_strict(&sgv)?.matmul_fp32_strict(&vh_m)?;
            term1.add(&t3a)?.sub(&t3b)?
        }
    };
    push_v3(child, &composed, host)
}

/// `EighValues`／`EighVectors` ノード `id` の d_input（V3）。構造を積む前に [`guard_eigh`] で判定する。
pub(super) fn eigh_cgrad<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    mirror: &[Option<Var<'c>>],
    id: NodeId,
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let node_op = parent_nodes[id.0].op.clone();
    let (input, is_vectors) = match &node_op {
        Op::EighValues { input, .. } => (*input, false),
        Op::EighVectors { input, .. } => (*input, true),
        _ => {
            return Err(reject(
                "eigh_cgrad に EighValues／EighVectors 以外が渡された（内部契約違反）",
            ));
        }
    };
    let in_cid = get_mirror(mirror, input)?.node_id();
    let own = materialize_fallible(parent_nodes, parent_ops, id)?;
    let own_m = get_mirror(mirror, id)?;
    let (vals_val, vecs_val, vals_m, vecs_m) = match &node_op {
        Op::EighValues { vectors, .. } => {
            let vecs_m = sibling_of(child, &own_m, 1, in_cid, is_eigh_vectors_of)?;
            (own.clone(), vectors.clone(), own_m, vecs_m)
        }
        Op::EighVectors { values, .. } => {
            let vals_m = sibling_of(child, &own_m, -1, in_cid, is_eigh_values_of)?;
            (values.clone(), own.clone(), vals_m, own_m)
        }
        _ => return Err(reject("内部契約違反")),
    };
    let g_val = value_of(child, g)?;
    guard_eigh(&vecs_val, &vals_val, &g_val, is_vectors)?;
    // 値（V3）: 1 階 `grad.rs` と逐語で同じ呼び出し。
    let host = if is_vectors {
        eigh_vjp(&vals_val, &vecs_val, None, Some(&g_val))?
    } else {
        eigh_vjp(&vals_val, &vecs_val, Some(&g_val), None)?
    };
    let n = vals_val.shape()[0];
    let gc = g.contiguous()?;
    let vt = vecs_m.transpose(0, 1)?;
    let eye = bool_mask(n, n, |i, j| i == j)?;
    let zeros = const_full(child, &[n, n], 0.0)?;
    let ga = if is_vectors {
        let ones = const_full(child, &[n, n], 1.0)?;
        let half = const_full(child, &[1], 0.5)?;
        let x = vt.contiguous()?.matmul_fp32_strict(&gc)?;
        let skew = x.sub(&x.transpose(0, 1)?.contiguous()?)?.mul(&half)?;
        let e = vals_m.reshape(&[1, n])?.sub(&vals_m.reshape(&[n, 1])?)?;
        let e_safe = Var::where_cond(&eye, &ones, &e)?;
        let middle = Var::where_cond(&eye, &zeros, &skew.div(&e_safe)?)?;
        vecs_m
            .matmul_fp32_strict(&middle)?
            .matmul_fp32_strict(&vt)?
    } else {
        let gb = gc.reshape(&[1, n])?;
        vecs_m.mul(&gb)?.matmul_fp32_strict(&vt)?
    };
    // forward が下三角だけを読む契約に合わせ、下三角へ集約する（`eigh_vjp` と同じ）。
    let lower = bool_mask(n, n, |i, j| i > j)?;
    let sum = ga.add(&ga.transpose(0, 1)?.contiguous()?)?;
    let diag_only = Var::where_cond(&eye, &ga, &zeros)?;
    let composed = Var::where_cond(&lower, &sum, &diag_only)?;
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

    fn sq<'t>(v: &Var<'t>) -> Var<'t> {
        v.tanh().sum(None).unwrap()
    }

    fn sv_data() -> Tensor<f32> {
        Tensor::new(
            vec![3.0, 0.2, 0.1, 0.3, 2.0, 0.4, 0.1, 0.5, 1.0, 0.2, 0.1, 0.3],
            &[4, 3],
        )
        .unwrap()
    }

    fn eigh_data() -> Tensor<f32> {
        Tensor::new(vec![4.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.5, 0.3, 1.0], &[3, 3]).unwrap()
    }

    /// SVD の push 順序契約の破れ（隙間・別 input・孤立 SvdS／SvdVh・順序入れ替え）は入口で拒否され、
    /// `child` は空のまま。Eigh の孤立 EighVectors・別 input も同様。
    #[test]
    fn svd_eigh_push_order_violations_are_rejected_before_child_write() {
        let a = sv_data();
        let (u, s, vh) = crate::eval::linalg::svd(&a).unwrap();
        let u_op = |input| Op::SvdU {
            input,
            s: s.clone(),
            vh: vh.clone(),
        };
        let s_op = |input| Op::SvdS {
            input,
            u: u.clone(),
            vh: vh.clone(),
        };
        let vh_op = |input| Op::SvdVh {
            input,
            u: u.clone(),
            s: s.clone(),
        };
        for case in [
            "gap",
            "other-input",
            "orphan-s",
            "orphan-vh",
            "swapped",
            "eigh-orphan",
        ] {
            let tape = Tape::new();
            let child = Tape::new();
            let x = tape.var(&a);
            let y = tape.var(&a);
            let xi = x.node_id();
            let last = match case {
                "gap" => {
                    tape.push_eager(u_op(xi), u.clone());
                    let _ = x.exp();
                    tape.push_eager(s_op(xi), s.clone());
                    tape.push_eager(vh_op(xi), vh.clone())
                }
                "other-input" => {
                    tape.push_eager(u_op(xi), u.clone());
                    tape.push_eager(s_op(y.node_id()), s.clone());
                    tape.push_eager(vh_op(xi), vh.clone())
                }
                "orphan-s" => tape.push_eager(s_op(xi), s.clone()),
                "orphan-vh" => tape.push_eager(vh_op(xi), vh.clone()),
                "swapped" => {
                    tape.push_eager(u_op(xi), u.clone());
                    tape.push_eager(vh_op(xi), vh.clone());
                    tape.push_eager(s_op(xi), s.clone())
                }
                _ => {
                    let e = eigh_data();
                    let (vals, vecs) = crate::eval::linalg::eigh(&e).unwrap();
                    let ex = tape.var(&e);
                    tape.push_eager(
                        Op::EighVectors {
                            input: ex.node_id(),
                            values: vals,
                        },
                        vecs,
                    )
                }
            };
            let loss = Var::from_raw(&tape, last).sum(None).unwrap();
            let res = tape.backward_create_graph(&loss, &child);
            assert!(
                matches!(res, Err(AutodiffError::Backward(_))),
                "{case}: Err(Backward) のはず"
            );
            assert!(child.is_empty(), "{case}: child は空");
        }
    }

    /// 兄弟の組み合わせによらず `svd()`／`eigh()` の再生は 1 回で、子テープの各 Op は 1 個。
    #[test]
    fn replay_calls_svd_and_eigh_once() {
        for which in ["u", "s", "vh", "u+vh", "all"] {
            let tape = Tape::new();
            let child = Tape::new();
            let x = tape.var(&sv_data());
            let f = x.svd().unwrap();
            let loss = match which {
                "u" => sq(&f.u),
                "s" => sq(&f.s),
                "vh" => sq(&f.vh),
                "u+vh" => sq(&f.u).add(&sq(&f.vh)).unwrap(),
                _ => sq(&f.u).add(&sq(&f.s)).unwrap().add(&sq(&f.vh)).unwrap(),
            };
            let _cg = tape.backward_create_graph(&loss, &child).unwrap();
            let nodes = child.nodes.borrow();
            let cnt = |p: fn(&Op) -> bool| nodes.iter().filter(|n| p(&n.op)).count();
            assert_eq!(
                (
                    cnt(|o| matches!(o, Op::SvdU { .. })),
                    cnt(|o| matches!(o, Op::SvdS { .. })),
                    cnt(|o| matches!(o, Op::SvdVh { .. })),
                ),
                (1, 1, 1),
                "{which}: svd() は 1 回だけ"
            );
        }
        for which in ["values", "vectors", "both"] {
            let tape = Tape::new();
            let child = Tape::new();
            let x = tape.var(&eigh_data());
            let f = x.eigh().unwrap();
            let loss = match which {
                "values" => sq(&f.eigenvalues),
                "vectors" => sq(&f.eigenvectors),
                _ => sq(&f.eigenvalues).add(&sq(&f.eigenvectors)).unwrap(),
            };
            let _cg = tape.backward_create_graph(&loss, &child).unwrap();
            let nodes = child.nodes.borrow();
            let cnt = |p: fn(&Op) -> bool| nodes.iter().filter(|n| p(&n.op)).count();
            assert_eq!(
                (
                    cnt(|o| matches!(o, Op::EighValues { .. })),
                    cnt(|o| matches!(o, Op::EighVectors { .. })),
                ),
                (1, 1),
                "{which}: eigh() は 1 回だけ"
            );
        }
    }

    /// SVD の縮退述語は 1 階 `svd_vjp` の分母検査と、しきい値のすぐ内側・すぐ外側で一致する。
    #[test]
    fn svd_degenerate_predicate_matches_first_order_svd_vjp() {
        // σ² の差が 1e-9 の内側（拒否）と外側（受理）。σ1 = 1e-3、σ2² = 1e-6 - d（f32 でも差を表現できる桁）。
        for (d, degenerate) in [(0.5e-9f64, true), (2.0e-9, false)] {
            let s2 = (1.0e-6 - d).sqrt() as f32;
            let sv = [1.0e-3f32, s2];
            let s64: Vec<f64> = sv.iter().map(|&v| f64::from(v)).collect();
            // f32 への丸めで差が変わるため、判定は f64 化した f32 値で比較する（実装と同じ入力）。
            let predicted = svd_gap_degenerate(&s64);
            let u = Tensor::new(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
            let s = Tensor::new(sv.to_vec(), &[2]).unwrap();
            let vh = u.clone();
            let du = Tensor::new(vec![0.1f32, 0.2, 0.3, 0.4], &[2, 2]).unwrap();
            let first = svd_vjp(&u, &s, &vh, Some(&du), None, None);
            assert_eq!(
                first.is_err(),
                predicted,
                "d={d}: 述語と svd_vjp の判定が一致する"
            );
            assert_eq!(predicted, degenerate, "d={d}: 期待どおりの側に入る");
        }
    }
}
