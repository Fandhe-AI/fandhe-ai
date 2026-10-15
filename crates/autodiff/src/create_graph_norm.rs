//! `Op::LayerNorm`・`Op::RmsNorm` の `create_graph`（子テープ方式の二階微分）対応
//! （イシュー #3078・親 #3075。方針の正は
//! `docs/autodiff-higher-order-grad-decision.md` §21.1・§21.3「#3078」、実装記録は §26）。
//!
//! [`crate::create_graph`] の `validate_ancestors`／`replay_op`／`build_cgrads` から呼ばれる
//! 行方向正規化専用の補助モジュール。`create_graph.rs` 本体を肥大させず、並行する
//! #3077（Softmax）・#3079（BatchNorm）との差分衝突を避けるために分離している。
//!
//! # 方式（§21.1 の V3＋S1）
//!
//! - **値（V3）**: 子テープ上の 1 階勾配ノードの値は 1 階と同じホスト関数
//!   （`grad.rs::rmsnorm_vjp_rows`／`layer_norm_vjp_rows`。f64 縮約契約を持つ）で計算する。
//!   CPU／`naive_ops` では 1 階 VJP も同じ関数へフォールバックするため、`g` が親の
//!   upstream と bit 一致する範囲では `cg.grad` も 1 階と bit 一致する。CUDA は 1 階が
//!   専用カーネルを通るので REQ-2 統一複合判定の範囲の一致のみを主張する（実機 parity は
//!   本イシューの対象外）。
//! - **構造（S1）**: 二階微分は、行ごとの 2 のべき乗スケール `s`（ホスト定数）で `x/s` を作って
//!   から二乗和・`rstd` を合成する f32 の `Var` 演算列 `S_k` を流れる。`x=1e20` のような
//!   有限入力でも f32 二乗が overflow しない。`S_k` の出力ごとに
//!   `Op::Contiguous`（VJP は恒等）のノードを `Tape::push_eager` で 1 個積み、値だけを
//!   ホスト関数の結果に差し替える（`create_graph.rs::reduce_bias_grad_var` と同型）。
//! - **閉包性（§21.1 (4)）**: 積む Op は `Reshape`・`Contiguous`・`Narrow`・`Mean`・`Sum`・
//!   `Mul`・`Add`・`ScalarBinary{Sub,Div}`・`ScalarUnary{Sqrt}` のみで、すべて
//!   `supports_create_graph == true`（孫テープ再生・`jvp_create_graph` が通る）。
//!   `Rsqrt`／`Reciprocal`／`Op::Var` は再生不可のため使わない。
//!
//! # `LayerNorm` の安定な中心化
//!
//! `Mean` の f32 出力を `x` から直接引くと `x=[16777216,16777218]` で偏差が `[0,2]` に
//! 狂う。そこで行の先頭要素 `x_ref` を先に引いた `d = x − x_ref` から `m = mean(d)`・
//! `dev = d − m` を作り、補正パス `dev2 = dev − mean(dev)` を 1 回加える。`s` は中心化前の
//! `x` ではなく `dev2` の実値から決める（定数行で `rstd_scaled=inf` になるのを避ける）。
//!
//! # 拒否条件（§21.3 の「S2 へ切り替える」より狭い fail-closed。所有者へ戻す論点）
//!
//! S2（ホストの統計定数＋二階項の手書き合成）は本イシューでは実装しない。代わりに
//! 子テープへ書く前に [`validate_norm_s1`] が親テープの `x` から次の入力を検出し
//! `Err(AutodiffError::Backward)` で拒否する（`child` は空のまま保たれる）。
//!
//! - `LayerNorm`: (a) `x` に非有限要素 (b) f32 の `x_i − x_0` が非有限
//!   (c) `hidden · max|x_i − x_0| ≥ 2^125`（`Mean` の f32 部分和が overflow しうる）
//!   (d) `eps == 0` で `max|x_i − x_0|` が `2^-100` 未満（定数行・極小偏差行。`rstd=inf`）
//! - `RmsNorm`: (a) 非有限要素 (d') `eps == 0` の全ゼロ行
//!
//! # 既知の限界
//!
//! `dxhat = g·w` は行ごとの 2 のべき乗 `gs` で割ってから縮約し、最後に `gs` を掛けて復元する
//! （`g=[3e38,3e38]` でも `Mean(dxhat⊙xhat)` が overflow しない）。`g` が非有限、または
//! `g·w` が f32 に収まらない、または `hidden·max(1,rstd)·max|g·w|` が `2^125` 以上の場合は構築時に
//! `Err(Backward)` で拒否する（逆伝播の行方向縮約の overflow を避けるため）。
//! `dw`／`db` の行方向 `Sum` は 1 階も f32 要素積のため、行をまたぐ合計の overflow は残る
//! （決定記録 §26.3）。

use fandhe_ai_tensor_core::{BackendOps, Tensor, row_norm_layout};

use crate::error::AutodiffError;
use crate::eval::{build_tensor, dense_vec};
use crate::grad::{layer_norm_vjp_rows, rmsnorm_vjp_rows};
use crate::tape::{NodeId, Op, Tape, TapeNode, materialize_fallible};
use crate::var::Var;

/// `Mean`（f32 部分和）が overflow しうるとみなす閾値 `2^125`（f32 最大値 `2^128` に対する余裕）。
const MEAN_OVERFLOW_GUARD: f64 = 4.253_529_586_511_73e37;
/// `eps == 0` の `LayerNorm` で「偏差が極小で `rstd` を安全に作れない」とみなす閾値 `2^-100`。
const TINY_DEVIATION_GUARD: f64 = 7.888_609_052_210_118e-31;

fn reject(what: &str) -> AutodiffError {
    AutodiffError::Backward(format!(
        "create_graph: {what}（LayerNorm／RmsNorm の S1 構造が安全に扱えない入力。\
         docs/autodiff-higher-order-grad-decision.md §26 参照）"
    ))
}

fn shape_err(e: fandhe_ai_tensor_core::ShapeError) -> AutodiffError {
    AutodiffError::Shape(e)
}

/// `validate_ancestors` から `requires_grad == true` かつ対象 Op のノードごとに呼ばれる入口検査。
/// `RmsNorm`／`LayerNorm` 以外の Op では何もしない。親テープの `x` 値だけで判定し、
/// バックエンドに依存しない保守的な拒否条件を適用する（モジュール doc「拒否条件」）。
pub(crate) fn validate_norm_s1(
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    op: &Op,
) -> Result<(), AutodiffError> {
    let (input, eps, is_layer_norm) = match op {
        Op::RmsNorm { input, eps, .. } => (*input, *eps, false),
        Op::LayerNorm { input, eps, .. } => (*input, *eps, true),
        _ => return Ok(()),
    };
    let x = materialize_fallible(parent_nodes, parent_ops, input)?;
    let (rows, hidden) = row_norm_layout(x.shape()).map_err(shape_err)?;
    if rows == 0 || hidden == 0 {
        return Ok(());
    }
    let data = dense_vec(x);
    if data.iter().any(|v| !v.is_finite()) {
        return Err(reject("正規化の入力に非有限要素がある"));
    }
    for r in 0..rows {
        let row = &data[r * hidden..(r + 1) * hidden];
        if is_layer_norm {
            let x0 = row[0];
            let mut max_dev = 0.0f64;
            for &v in row {
                if !(v - x0).is_finite() {
                    return Err(reject("LayerNorm の x_i − x_0 が f32 で非有限になる"));
                }
                max_dev = max_dev.max((v as f64 - x0 as f64).abs());
            }
            if hidden as f64 * max_dev >= MEAN_OVERFLOW_GUARD {
                return Err(reject(
                    "LayerNorm の偏差が大きく Mean の f32 部分和が overflow しうる",
                ));
            }
            if eps == 0.0 && max_dev < TINY_DEVIATION_GUARD {
                return Err(reject("eps=0 の LayerNorm で行が定数（または偏差が極小）"));
            }
        } else if eps == 0.0 && row.iter().all(|&v| v == 0.0) {
            return Err(reject("eps=0 の RmsNorm で行が全ゼロ"));
        }
    }
    Ok(())
}

/// `replay_op` から呼ばれる forward 再生（`Var::rms_norm`／`layer_norm` への薄い委譲）。
pub(crate) fn replay_norm<'c>(
    op: &Op,
    mirror: &[Option<Var<'c>>],
) -> Result<Var<'c>, AutodiffError> {
    match op {
        Op::RmsNorm { input, weight, eps } => {
            let x = get(mirror, *input)?;
            let w = weight.map(|id| get(mirror, id)).transpose()?;
            x.rms_norm(w.as_ref(), *eps)
        }
        Op::LayerNorm {
            input,
            weight,
            bias,
            eps,
        } => {
            let x = get(mirror, *input)?;
            let w = weight.map(|id| get(mirror, id)).transpose()?;
            let b = bias.map(|id| get(mirror, id)).transpose()?;
            x.layer_norm(w.as_ref(), b.as_ref(), *eps)
        }
        _ => Err(AutodiffError::Backward(
            "create_graph: replay_norm: RmsNorm／LayerNorm 以外の Op（内部契約違反）".into(),
        )),
    }
}

fn get<'c>(mirror: &[Option<Var<'c>>], id: NodeId) -> Result<Var<'c>, AutodiffError> {
    mirror[id.0].ok_or_else(|| {
        AutodiffError::Backward(format!(
            "create_graph: 祖先ノード（NodeId({}))の写しが未構築（内部不変条件違反）",
            id.0
        ))
    })
}

/// 2 のべき乗 `2^e`（`e` は `[-126, 127]` で f32 の正規数として厳密に表現できる）の指数を
/// `t` 以上になる最小の整数として求め、表現可能域へクランプする。`t` が 0・非有限なら 0（`s=1`）。
fn pow2_ceil_exponent(t: f64) -> i32 {
    if t.is_nan() || t.is_infinite() || t <= 0.0 {
        return 0;
    }
    let mut e = t.log2().ceil() as i32;
    while 2f64.powi(e) < t {
        e += 1;
    }
    while 2f64.powi(e - 1) >= t {
        e -= 1;
    }
    e.clamp(-126, 127)
}

/// 行ごとのスケール指数。`eps>0` は `max(maxabs, √eps)`、`eps=0` は `maxabs` を 2 のべき乗へ
/// 切り上げる（§21.1 (5)）。
fn scale_exponent(maxabs: f64, eps: f32) -> i32 {
    let t = if eps > 0.0 {
        maxabs.max((eps as f64).sqrt())
    } else {
        maxabs
    };
    pow2_ceil_exponent(t)
}

fn row_maxabs(data: &[f32], rows: usize, hidden: usize) -> Vec<f64> {
    (0..rows)
        .map(|r| {
            data[r * hidden..(r + 1) * hidden]
                .iter()
                .fold(0.0f64, |m, &v| m.max((v as f64).abs()))
        })
        .collect()
}

/// 行ごとの指数から `s`（`[rows, 1]`）と `eps/s²`（`[rows, 1]`。f64 計算→ f32 丸め）の定数葉を作る。
fn scale_consts<'c>(
    child: &'c Tape,
    exps: &[i32],
    eps: f32,
) -> Result<(Var<'c>, Var<'c>), AutodiffError> {
    let s: Vec<f32> = exps.iter().map(|&e| 2f64.powi(e) as f32).collect();
    let eps_s: Vec<f32> = exps
        .iter()
        .map(|&e| (eps as f64 / 2f64.powi(2 * e)) as f32)
        .collect();
    let shape = [exps.len(), 1];
    let s_t = Tensor::new(s, &shape)?;
    let e_t = Tensor::new(eps_s, &shape)?;
    Ok((child.var_no_grad(&s_t), child.var_no_grad(&e_t)))
}

/// 行ごとの真の `rstd = 1/√(var_or_ms + eps)`（f64）。`centered == true` は LayerNorm
/// （平均を引いた分散）、`false` は RmsNorm（二乗平均）。`upstream_scale` の増幅項に使う。
fn row_rstd(data: &[f32], rows: usize, hidden: usize, eps: f32, centered: bool) -> Vec<f64> {
    (0..rows)
        .map(|r| {
            let row = &data[r * hidden..(r + 1) * hidden];
            let n = hidden as f64;
            let mean = if centered {
                row.iter().map(|&v| f64::from(v)).sum::<f64>() / n
            } else {
                0.0
            };
            let ms = row
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / n;
            1.0 / (ms + f64::from(eps)).sqrt()
        })
        .collect()
}

/// upstream 側のスケール `gs`（行ごとの 2 のべき乗。`[rows, 1]`）の定数葉を作る。
/// `dxhat = g·w` の行最大絶対値（f64）から指数を決める。`g` が非有限、または `dxhat` が
/// f32 に収まらない（1 階側も overflow する）場合は型付きエラーで拒否する。`dxhat/gs` を縮約へ
/// 渡すことで `Mean(dxhat⊙xhat)` の f32 部分和が `g=3e38` のような有限入力で overflow するのを
/// 避ける（レビュー指摘 PR #3346。復元は `gs` を最後に掛ける）。
fn upstream_scale<'c>(
    child: &'c Tape,
    dy: &[f32],
    w: Option<&[f32]>,
    rstd: &[f64],
    rows: usize,
    hidden: usize,
) -> Result<Var<'c>, AutodiffError> {
    let mut exps = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut mx = 0.0f64;
        for i in 0..hidden {
            let g = f64::from(dy[r * hidden + i]);
            if !g.is_finite() {
                return Err(reject("upstream に非有限要素がある"));
            }
            let d = match w {
                Some(w) => g * f64::from(w[i]),
                None => g,
            };
            if d.abs() > f64::from(f32::MAX) {
                return Err(reject("dxhat = g·w が f32 に収まらない"));
            }
            mx = mx.max(d.abs());
        }
        // 逆伝播は cotangent を行方向に足し合わせる（`mean_dot` 等への縮約）ため、f32 の和が
        // `hidden · max|g·w|` に達しうる。さらに `xhat`／`rstd` の cotangent は
        // `v·(rstd_s/s) = v·rstd`（真の rstd）経由で流れるため、実際に overflow する積は
        // `hidden · rstd · max|g·w|` 規模になる（低分散行で増幅。PR #3346 Bugbot 指摘）。
        // 増幅は 1 以上のときだけ見て、これが 2^125 以上の行は型付きエラーで拒否する
        // （`gs` を縮約の手前まで持ち越しても cotangent 和は避けられず、実測で NaN になる）。
        if hidden as f64 * rstd[r].max(1.0) * mx >= MEAN_OVERFLOW_GUARD {
            return Err(reject(
                "hidden · max(1, rstd) · max|g·w| が 2^125 以上で逆伝播の行方向縮約が overflow しうる",
            ));
        }
        exps.push(pow2_ceil_exponent(mx));
    }
    let gs: Vec<f32> = exps.iter().map(|&e| 2f64.powi(e) as f32).collect();
    Ok(child.var_no_grad(&Tensor::new(gs, &[rows, 1])?))
}

/// `gs/hidden`（行ごと。`[rows, 1]`）の定数葉。縮約（`Sum`）の直後にこれを掛けて `gs` を復元する。
/// `Mean` の逆伝播が cotangent を行方向に足し合わせる前に `gs` を掛けないため、`gs`・hidden 倍の
/// 中間 overflow を避ける（レビュー指摘 PR #3346 の二巡目。`gs/hidden` は f64 で計算し f32 へ丸める）。
fn gs_over_hidden<'c>(
    child: &'c Tape,
    gs: &Var<'c>,
    hidden: usize,
) -> Result<Var<'c>, AutodiffError> {
    let v = dense_vec(&child_value(child, gs)?);
    let c: Vec<f32> = v
        .iter()
        .map(|&g| (f64::from(g) / hidden as f64) as f32)
        .collect();
    Ok(child.var_no_grad(&Tensor::new(c, &[v.len(), 1])?))
}

/// 構造 `structure` の VJP 値を `value`（ホスト関数の結果）へ差し替えた `Op::Contiguous`
/// ノードを子テープへ積む。`Contiguous` の VJP は恒等なので二階の微分は `structure` を流れる。
fn tag_with_host_value<'c>(
    child: &'c Tape,
    structure: &Var<'c>,
    value: Tensor<f32>,
) -> Result<Var<'c>, AutodiffError> {
    if structure.shape() != value.shape() {
        return Err(AutodiffError::Backward(
            "create_graph: LayerNorm／RmsNorm の構造と値の shape が不一致（内部契約違反）".into(),
        ));
    }
    let id = child.push_eager(
        Op::Contiguous {
            input: structure.node_id(),
        },
        value,
    );
    Ok(Var::from_raw(child, id))
}

/// 子テープ上の `v` の実値（密）を読む。
fn child_value(child: &Tape, v: &Var<'_>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = child.nodes.borrow();
    Ok(materialize_fallible(&nodes, child.ops(), v.node_id())?.clone())
}

fn rows_hidden_2d<'c>(v: &Var<'c>, rows: usize, hidden: usize) -> Result<Var<'c>, AutodiffError> {
    v.contiguous()?.reshape(&[rows, hidden])
}

fn keep_dim<'c>(v: &Var<'c>) -> Result<Var<'c>, AutodiffError> {
    v.unsqueeze(1)
}

/// 空テンソル（`rows == 0 || hidden == 0`）の寄与。1 階と同じくゼロ定数を返す。
fn zero_contributions<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    x_shape: &[usize],
    hidden: usize,
    targets: &Targets,
) -> Result<Vec<(NodeId, Var<'c>)>, AutodiffError> {
    let mut out = Vec::new();
    let tracked = |id: NodeId| parent_nodes[id.0].requires_grad;
    if tracked(targets.input) {
        out.push((targets.input, child.var_no_grad(&Tensor::zeros(x_shape)?)));
    }
    for id in [targets.weight, targets.bias].into_iter().flatten() {
        if tracked(id) {
            out.push((id, child.var_no_grad(&Tensor::zeros(&[hidden])?)));
        }
    }
    Ok(out)
}

/// 寄与先の親ノード id（`weight`／`bias` は存在する場合のみ）。
pub(crate) struct Targets {
    pub(crate) input: NodeId,
    pub(crate) weight: Option<NodeId>,
    pub(crate) bias: Option<NodeId>,
}

/// `Op::RmsNorm` の子テープ上の VJP（V3 値＋S1 構造）。`g` は出力ノードの子テープ上の upstream。
/// 戻り値は `(親ノード id, 子テープ上の寄与)` の列で、呼び出し側（`build_cgrads`）が
/// `accumulate` する。追跡なしの親へは寄与を作らない。
pub(crate) fn rms_norm_cgrads<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    mirror: &[Option<Var<'c>>],
    g: Var<'c>,
    targets: &Targets,
    eps: f32,
) -> Result<Vec<(NodeId, Var<'c>)>, AutodiffError> {
    let x_val = materialize_fallible(parent_nodes, parent_ops, targets.input)?.clone();
    let x_shape = x_val.shape().to_vec();
    let (rows, hidden) = row_norm_layout(&x_shape).map_err(shape_err)?;
    if rows == 0 || hidden == 0 {
        return zero_contributions(child, parent_nodes, &x_shape, hidden, targets);
    }
    let need_x = parent_nodes[targets.input.0].requires_grad;
    let need_w = targets
        .weight
        .is_some_and(|id| parent_nodes[id.0].requires_grad);

    // 値（V3）: 1 階と同じホスト関数。
    let x_slice = dense_vec(&x_val);
    let w_val = match targets.weight {
        Some(id) => Some(materialize_fallible(parent_nodes, parent_ops, id)?.clone()),
        None => None,
    };
    let w_slice = w_val.as_ref().map(dense_vec);
    let dy = dense_vec(&child_value(child, &g)?);
    let (dx_host, dw_host) = rmsnorm_vjp_rows(&x_slice, w_slice.as_deref(), eps, rows, hidden, &dy);

    // 構造（S1）。
    let x_m = get(mirror, targets.input)?;
    let w_m = targets.weight.map(|id| get(mirror, id)).transpose()?;
    let x2 = rows_hidden_2d(&x_m, rows, hidden)?;
    let g2 = rows_hidden_2d(&g, rows, hidden)?;
    let exps: Vec<i32> = row_maxabs(&x_slice, rows, hidden)
        .into_iter()
        .map(|m| scale_exponent(m, eps))
        .collect();
    let (s_c, eps_s) = scale_consts(child, &exps, eps)?;
    let one = child.var_no_grad(&Tensor::full(&[1], 1.0f32)?);

    let xs = x2.div(&s_c)?;
    let ms = keep_dim(&xs.mul(&xs)?.mean(Some(1))?)?;
    let rstd_s = one.div(&ms.add(&eps_s)?.sqrt()?)?;
    let xhat = xs.mul(&rstd_s)?;

    let mut out = Vec::new();
    if need_x {
        let gs_c = upstream_scale(
            child,
            &dy,
            w_slice.as_deref(),
            &row_rstd(&x_slice, rows, hidden, eps, false),
            rows,
            hidden,
        )?;
        let gw = match &w_m {
            Some(w) => g2.mul(w)?,
            None => g2,
        };
        let dxhat = gw.div(&gs_c)?;
        // `mean_dot·gs` は縮約の直後に `gs/hidden` を掛けて作る（逆伝播で縮約前に大きな
        // cotangent が復元されない）。
        let gsh = gs_over_hidden(child, &gs_c, hidden)?;
        let dot_gs = keep_dim(&dxhat.mul(&xhat)?.sum(Some(1))?)?.mul(&gsh)?;
        let inner = gw.sub(&xhat.mul(&dot_gs)?)?;
        let dx_s = rstd_s.div(&s_c)?.mul(&inner)?.reshape(&x_shape)?;
        let v = tag_with_host_value(child, &dx_s, build_tensor(dx_host, &x_shape))?;
        out.push((targets.input, v));
    }
    if need_w && let (Some(wid), Some(dw)) = (targets.weight, dw_host) {
        let dw_s = g2.mul(&xhat)?.sum(Some(0))?;
        let v = tag_with_host_value(child, &dw_s, build_tensor(dw, &[hidden]))?;
        out.push((wid, v));
    }
    Ok(out)
}

/// `Op::LayerNorm` の子テープ上の VJP（V3 値＋S1 構造＋安定な中心化）。契約は
/// [`rms_norm_cgrads`] と同じ。
pub(crate) fn layer_norm_cgrads<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    mirror: &[Option<Var<'c>>],
    g: Var<'c>,
    targets: &Targets,
    eps: f32,
) -> Result<Vec<(NodeId, Var<'c>)>, AutodiffError> {
    let x_val = materialize_fallible(parent_nodes, parent_ops, targets.input)?.clone();
    let x_shape = x_val.shape().to_vec();
    let (rows, hidden) = row_norm_layout(&x_shape).map_err(shape_err)?;
    if rows == 0 || hidden == 0 {
        return zero_contributions(child, parent_nodes, &x_shape, hidden, targets);
    }
    let need_x = parent_nodes[targets.input.0].requires_grad;
    let need_w = targets
        .weight
        .is_some_and(|id| parent_nodes[id.0].requires_grad);
    let need_b = targets
        .bias
        .is_some_and(|id| parent_nodes[id.0].requires_grad);

    // 値（V3）。
    let x_slice = dense_vec(&x_val);
    let w_val = match targets.weight {
        Some(id) => Some(materialize_fallible(parent_nodes, parent_ops, id)?.clone()),
        None => None,
    };
    let w_slice = w_val.as_ref().map(dense_vec);
    let dy = dense_vec(&child_value(child, &g)?);
    let (dx_host, dw_host, db_host) = layer_norm_vjp_rows(
        &x_slice,
        w_slice.as_deref(),
        targets.bias.is_some(),
        eps,
        rows,
        hidden,
        &dy,
    );

    // 構造: 安定な中心化（x_ref を先に引き、補正パスを 1 回加える）。
    let x_m = get(mirror, targets.input)?;
    let w_m = targets.weight.map(|id| get(mirror, id)).transpose()?;
    let x2 = rows_hidden_2d(&x_m, rows, hidden)?;
    let g2 = rows_hidden_2d(&g, rows, hidden)?;
    let x_ref = x2.narrow(1, 0, 1)?;
    let d = x2.sub(&x_ref)?;
    let m = keep_dim(&d.mean(Some(1))?)?;
    let dev = d.sub(&m)?;
    let dev2 = dev.sub(&keep_dim(&dev.mean(Some(1))?)?)?;

    // `s` は中心化後偏差 `dev2` の子テープ上の実値から決める（中心化前の `x` からは取らない）。
    let dev2_host = dense_vec(&child_value(child, &dev2)?);
    let exps: Vec<i32> = row_maxabs(&dev2_host, rows, hidden)
        .into_iter()
        .map(|mx| scale_exponent(mx, eps))
        .collect();
    let (s_c, eps_s) = scale_consts(child, &exps, eps)?;
    let one = child.var_no_grad(&Tensor::full(&[1], 1.0f32)?);

    let devs = dev2.div(&s_c)?;
    let var_s = keep_dim(&devs.mul(&devs)?.mean(Some(1))?)?;
    let rstd_s = one.div(&var_s.add(&eps_s)?.sqrt()?)?;
    let xhat = devs.mul(&rstd_s)?;

    let mut out = Vec::new();
    if need_x {
        let gs_c = upstream_scale(
            child,
            &dy,
            w_slice.as_deref(),
            &row_rstd(&x_slice, rows, hidden, eps, true),
            rows,
            hidden,
        )?;
        let gw = match &w_m {
            Some(w) => g2.mul(w)?,
            None => g2,
        };
        let dxhat = gw.div(&gs_c)?;
        let gsh = gs_over_hidden(child, &gs_c, hidden)?;
        let sum_gs = keep_dim(&dxhat.sum(Some(1))?)?.mul(&gsh)?;
        let dot_gs = keep_dim(&dxhat.mul(&xhat)?.sum(Some(1))?)?.mul(&gsh)?;
        let inner = gw.sub(&sum_gs)?.sub(&xhat.mul(&dot_gs)?)?;
        let dx_s = rstd_s.div(&s_c)?.mul(&inner)?.reshape(&x_shape)?;
        let v = tag_with_host_value(child, &dx_s, build_tensor(dx_host, &x_shape))?;
        out.push((targets.input, v));
    }
    if need_w && let (Some(wid), Some(dw)) = (targets.weight, dw_host) {
        let dw_s = g2.mul(&xhat)?.sum(Some(0))?;
        let v = tag_with_host_value(child, &dw_s, build_tensor(dw, &[hidden]))?;
        out.push((wid, v));
    }
    if need_b && let (Some(bid), Some(db)) = (targets.bias, db_host) {
        // 構造は `g2.sum(0)`、値は `layer_norm_vjp_rows` の db（`rows == 1` の符号付きゼロの
        // 扱いまで 1 階と bit 一致させるため `reduce_bias_grad_var` ではなく V3 で書く）。
        let db_s = g2.sum(Some(0))?;
        let v = tag_with_host_value(child, &db_s, build_tensor(db, &[hidden]))?;
        out.push((bid, v));
    }
    Ok(out)
}
