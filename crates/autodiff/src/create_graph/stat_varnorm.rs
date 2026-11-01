//! 統計縮約 4 Op（`Var`・`Std`・`VectorNorm`〈`L1`／`L2`〉・`PNorm`）の `create_graph` 対応
//! （イシュー #3791・親 #3725・設計 `docs/autodiff-higher-order-grad-decision.md` §33、実装記録 §51）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `validate_ancestors`（入口検査
//! [`validate_stat_varnorm`]）・`replay_op`（forward 再生。公開 `Var::var`／`std`／`norm_l1`／
//! `norm_l2`／`norm_p` への 1 行委譲なので親側に置く）・`build_cgrads`（1 階 VJP の子テープ記録
//! [`stat_varnorm_cgrad`]）から呼ばれる。`LogSumExp`・`Nansum`・`Nanmean` は兄弟イシュー #3792 の
//! 担当でここには含めない。
//!
//! # 方式（値は 1 階 `grad.rs` の VJP と bit 一致。契約は REQ-2 のまま）
//!
//! どの Op も **V3**: 値（`cg.grad` の実体）は 1 階と同じホスト関数（`var_vjp`・`std_vjp`・
//! `vector_norm_vjp`・`pnorm_vjp`。f64 アキュムレータ）を [`push_v3`] で載せ、構造（二階の
//! 微分が流れる `Var` 演算グラフ）は f32 の合成で組む。縮約軸を最後へ寄せた `[lanes, n]` に
//! 正準化して軸 1 で扱い（`dim=None` は `[1, numel]`）、終わったら元の並びへ戻す。
//!
//! - `Var`: `cg = g_b · 2·dev2 / (n−c)`。`dev2` は LayerNorm と共有する安定な中心化
//!   （`create_graph_norm::stable_centering`）。`dev2` について線形なのでスケールは要らない。
//! - `Std`: 2 段スケール（前スケール S0 → 中心化 → 子テープ上の `dev2` の実値から決める `s1`）で
//!   `|dev_s / std_s| ≤ √(n−c)` に収め、`cg = g_b · dev_s / (std_s · (n−c))`。`std == 0` の
//!   レーンは 1 階が 0 勾配にマスクするので、`Sqrt` の入力側と分母を `where_cond` で正の定数へ
//!   先に置換してから出力を 0 にする（`Sqrt` の微分係数 `0.5/y` が `y=0` で `0×inf=NaN` に
//!   なるのを防ぐ）。S0 は §27.2 が将来案に残したもので、本族で採る（#3767 の承認）。
//! - `VectorNorm::L1`: 値は V3、構造は `where_cond(x>0, g_b, where_cond(x<0, −g_b, 0))`
//!   （0/1 定数を掛けない。`inf*0` を作らない）。x についての二階は 0。
//! - `VectorNorm::L2`: `x_s = x/s`（`s` はレーンの `max|x|` 以上の 2 のべき乗）→
//!   `cg = g_b · x_s / ‖x_s‖₂`。全ゼロレーンは `Sqrt` の入力側を守って出力を 0 にする。
//! - `PNorm`: `x_s = x/s`、`|x_s|` のゼロ要素は `where_cond` で 1 に置換してから `PowScalar` へ
//!   入れる（`p<1` で `0^負` が inf になるのを防ぐ。1 階の劣勾配の選び方と同じ）。
//!   `cg = g_b · sign(x) · (|x_s| / ‖x_s‖_p)^(p−1)`。
//!
//! 子テープへ積む Op は `Reshape`・`Permute`・`Contiguous`・`Narrow`・`BroadcastTo`・`Sum`・
//! `Mean`・`Mul`・`Add`・`Where`・`ScalarBinary{Sub,Div}`・`ScalarUnary{Sqrt,Abs,PowScalar,Neg}`・
//! 定数 `Leaf` のみで、すべて `supports_create_graph() == true`（閉包性。孫テープでも閉じる）。
//!
//! # 拒否条件（fail-closed。保証範囲は §27.3 と同じ書き方）
//!
//! - **入口**（[`validate_stat_varnorm`]。子テープへ書く前）: `Var`／`Std`／`L2`／`PNorm` の入力に
//!   `±inf` がある（NaN は伝播させるので通す。#3767 の承認）。`L1` は `±inf` も NaN も受理する
//!   （構造が 3 分岐の `where_cond` で `inf` に触れず、NaN は 1 階と同じく符号 0 になる）。
//!   未知の `VectorNormOrd` は拒否する。`Var` は `x_i − x_0` が f32 で有限であることと
//!   `n·max|x_i − x_0| < 2^125`（`Mean` の f32 部分和）を要求する。なお `pnorm_vjp` の
//!   `input_has_inf` 分岐（`±inf` 要素への均等分配）は、この入口拒否により create_graph からは
//!   到達しない。
//! - **総括判定**（[`check_var`] など。`g` が手に入った時点で構造を積む前に呼ぶ。`Std` だけは
//!   `dev2` の実値が要るため中心化まで積んでから呼ぶ）: 構造の forward の各段と二階の逆伝播
//!   （seed の各要素の絶対値 ≤ 1）の「cotangent × 微分係数」を f64 で積み上げ、どれか 1 段でも
//!   `2^125` 以上か非有限なら拒否する。非有限の上流 `g` も拒否する。NaN を含むレーンは判定から
//!   外す（出力は 1 階でも NaN になり、縮約はレーンごとに閉じている）。
//! - 拒否は `AutodiffError::Backward`。`backward_create_graph` のロールバックで `child` は
//!   空へ戻り、安全な入力で再試行できる。
//!
//! 保証は判定を通った入力に限り、判定を通った入力では 1 階・`cg.grad`・二階（seed の各要素の
//! 絶対値 ≤ 1）・孫テープ再生が有限で、`cg.grad` は 1 階と bit 一致する（V3）。`g` を作る上流
//! の Op の逆伝播、絶対値 1 を超える seed、三階以上の微分は対象外。構造は二階だけを担うので、
//! 相対的に極小な要素が f32 で 0 に落ちる場合の二階の精度は保証しない（§45 の S0 と同じ扱い）。
//!
//! GPU: `BackendOps::var`／`vector_norm`／`vector_norm_p` を実装するのは `backend-cpu` だけで、
//! `Std` は常にホスト実装（経路区分 `H`）。実機 parity は未実測（Phase 10 の #3856・#3858）。

use fandhe_ai_tensor_core::{BackendOps, Tensor, VectorNormOrd, reduce_out_shape};

use super::select_reduce::bcast_upstream;
use super::{child_value, push_v3};
use crate::create_graph_norm::{STAGE_BOUND_GUARD, pow2_ceil_exponent, stable_centering};
use crate::error::AutodiffError;
use crate::eval::dense_vec;
use crate::grad::{pnorm_vjp, std_vjp, var_vjp, vector_norm_vjp};
use crate::tape::{NodeId, Op, Tape, TapeNode, materialize_fallible};
use crate::var::Var;

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!(
        "create_graph: {msg}（統計縮約 Var／Std／VectorNorm／PNorm の構造が安全に扱えない入力。\
         docs/autodiff-higher-order-grad-decision.md §33・§51 参照）"
    ))
}

/// 構造の 1 段の上界 `v` を判定する。非有限または `2^125` 以上なら型付きエラー。
fn st(v: f64, what: &str) -> Result<(), AutodiffError> {
    if v.is_finite() && v.abs() < STAGE_BOUND_GUARD {
        Ok(())
    } else {
        Err(invalid(format!(
            "構造段の上界判定: {what} の上界 {v:e} が 2^125 以上で f32 の安全域を外れる"
        )))
    }
}

// ---------------------------------------------------------------------
// 入口検査（validate_ancestors から呼ぶ。子テープへ書く前に拒否する）
// ---------------------------------------------------------------------

/// 縮約軸 `dim` の `(outer, n, inner)`。`dim=None` は `(1, numel, 1)`。
fn outer_n_inner(
    shape: &[usize],
    dim: Option<usize>,
) -> Result<(usize, usize, usize), AutodiffError> {
    match dim {
        None => Ok((1, shape.iter().product(), 1)),
        Some(axis) => {
            if axis >= shape.len() {
                return Err(invalid(format!(
                    "縮約軸 dim={axis} が入力 rank {} の範囲外",
                    shape.len()
                )));
            }
            Ok((
                shape[..axis].iter().product(),
                shape[axis],
                shape[axis + 1..].iter().product(),
            ))
        }
    }
}

/// `validate_ancestors` から `requires_grad == true` の非葉ノードごとに呼ばれる入口検査。
/// 4 Op 以外では何もしない。
pub(super) fn validate_stat_varnorm(
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    op: &Op,
) -> Result<(), AutodiffError> {
    let (input, dim, kind) = match op {
        Op::Var {
            input,
            dim,
            correction,
        } => (*input, *dim, Kind::Var(*correction)),
        Op::Std {
            input,
            dim,
            correction,
        } => (*input, *dim, Kind::Std(*correction)),
        Op::VectorNorm { input, ord, dim } => match ord {
            // L1 は ±inf も NaN も受理する（モジュール doc）。
            VectorNormOrd::L1 => return Ok(()),
            VectorNormOrd::L2 => (*input, *dim, Kind::L2),
            _ => return Err(invalid(format!("未知の VectorNormOrd {ord:?}"))),
        },
        Op::PNorm { input, dim, .. } => (*input, *dim, Kind::PNorm),
        _ => return Ok(()),
    };
    let x = materialize_fallible(parent_nodes, parent_ops, input)?;
    if x.shape().contains(&0) {
        return Ok(());
    }
    let (outer, n, inner) = outer_n_inner(x.shape(), dim)?;
    let data = dense_vec(x);
    if data.iter().any(|v| v.is_infinite()) {
        return Err(invalid(
            "入力に ±inf がある（NaN は伝播するが ±inf は拒否する）".into(),
        ));
    }
    if let Kind::Var(c) | Kind::Std(c) = kind
        && n <= c
    {
        return Err(invalid(format!("自由度不足（n={n} <= correction={c}）")));
    }
    if let Kind::Var(_) = kind {
        let xl = lane_major(&data, outer, n, inner);
        for lane in xl.chunks(n.max(1)) {
            if lane.iter().any(|v| v.is_nan()) {
                continue;
            }
            let x0 = lane[0];
            let mut maxd = 0.0f64;
            for &v in lane {
                if !(v - x0).is_finite() {
                    return Err(invalid("Var の x_i − x_0 が f32 で非有限になる".into()));
                }
                maxd = maxd.max((v as f64 - x0 as f64).abs());
            }
            if n as f64 * maxd >= STAGE_BOUND_GUARD {
                return Err(invalid(
                    "Var の偏差が大きく Mean の f32 部分和が overflow しうる".into(),
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Kind {
    Var(usize),
    Std(usize),
    L2,
    PNorm,
}

// ---------------------------------------------------------------------
// レーン正準化
// ---------------------------------------------------------------------

/// 元の並び（outer×axis×inner）をレーン主順 `[lanes = outer·inner, n]` に並べ替える。
fn lane_major(data: &[f32], outer: usize, n: usize, inner: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for o in 0..outer {
        for i in 0..inner {
            for a in 0..n {
                out.push(data[(o * n + a) * inner + i]);
            }
        }
    }
    out
}

/// 縮約軸を最後へ寄せた `[lanes, n]` への正準化（`Reshape`・`Permute`・`Contiguous`）。
struct Lanes {
    in_shape: Vec<usize>,
    dim: Option<usize>,
    outer: usize,
    n: usize,
    inner: usize,
    /// 軸を最後へ寄せる置換（恒等のときは空）。
    perm: Vec<usize>,
}

impl Lanes {
    fn new(in_shape: &[usize], dim: Option<usize>) -> Result<Self, AutodiffError> {
        let (outer, n, inner) = outer_n_inner(in_shape, dim)?;
        let perm = match dim {
            Some(axis) if axis + 1 != in_shape.len() => (0..in_shape.len())
                .filter(|&d| d != axis)
                .chain(std::iter::once(axis))
                .collect(),
            _ => Vec::new(),
        };
        Ok(Self {
            in_shape: in_shape.to_vec(),
            dim,
            outer,
            n,
            inner,
            perm,
        })
    }

    fn lanes(&self) -> usize {
        self.outer * self.inner
    }

    fn to_lane_form<'c>(&self, v: &Var<'c>) -> Result<Var<'c>, AutodiffError> {
        let src = if self.perm.is_empty() {
            *v
        } else {
            v.permute(&self.perm)?
        };
        src.contiguous()?.reshape(&[self.lanes(), self.n])
    }

    fn restore_lanes<'c>(&self, v: &Var<'c>) -> Result<Var<'c>, AutodiffError> {
        let v = v.contiguous()?;
        if self.dim.is_none() || self.perm.is_empty() {
            return v.reshape(&self.in_shape);
        }
        let permuted: Vec<usize> = self.perm.iter().map(|&d| self.in_shape[d]).collect();
        let mut inv = vec![0usize; self.perm.len()];
        for (i, &p) in self.perm.iter().enumerate() {
            inv[p] = i;
        }
        v.reshape(&permuted)?.permute(&inv)?.contiguous()
    }

    fn lane_major(&self, data: &[f32]) -> Vec<f32> {
        lane_major(data, self.outer, self.n, self.inner)
    }
}

fn col_const<'c>(child: &'c Tape, vals: Vec<f32>) -> Result<Var<'c>, AutodiffError> {
    let len = vals.len();
    Ok(child.var_no_grad(&Tensor::new(vals, &[len, 1])?))
}

fn scalar_const(child: &Tape, v: f32) -> Result<Var<'_>, AutodiffError> {
    Ok(child.var_no_grad(&Tensor::full(&[1], v)?))
}

fn pow2(e: i32) -> f32 {
    2f64.powi(e) as f32
}

fn lane_has_nan(lane: &[f32]) -> bool {
    lane.iter().any(|v| v.is_nan())
}

/// レーンごとの `max|x|` 以上の 2 のべき乗の指数（NaN レーン・全ゼロは 0）。
fn lane_exps(xl: &[f32], n: usize) -> Vec<i32> {
    xl.chunks(n)
        .map(|lane| {
            if lane_has_nan(lane) {
                0
            } else {
                pow2_ceil_exponent(lane.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs())))
            }
        })
        .collect()
}

fn bool_tensor(flags: Vec<bool>, shape: &[usize]) -> Result<Tensor<bool>, AutodiffError> {
    Ok(Tensor::new(flags, shape)?)
}

// ---------------------------------------------------------------------
// 総括判定（モジュール doc「拒否条件」）
// ---------------------------------------------------------------------

fn check_g_finite(g: &[f32]) -> Result<(), AutodiffError> {
    if g.iter().any(|v| !v.is_finite()) {
        return Err(invalid("upstream に非有限要素がある".into()));
    }
    Ok(())
}

fn check_var(n: usize, nc: f64, xl: &[f32], g: &[f32]) -> Result<(), AutodiffError> {
    let nf = n as f64;
    let kc = 2.0 / nc;
    for (l, lane) in xl.chunks(n).enumerate() {
        if lane_has_nan(lane) {
            continue;
        }
        let x0 = lane[0] as f64;
        let maxd = lane
            .iter()
            .fold(0.0f64, |m, &v| m.max((v as f64 - x0).abs()));
        // forward: d → Mean(d) → dev → Mean(dev) → dev2 → t → t·g
        st(nf * maxd, "Var の Mean(d)")?;
        let b1 = 2.0 * maxd;
        st(nf * b1, "Var の Mean(dev)")?;
        let b2 = 2.0 * b1;
        st(b2, "Var の dev2")?;
        let t = b2 * kc;
        st(t, "Var の t")?;
        let gl = g[l].abs() as f64;
        st(t * gl, "Var の t·g")?;
        // 二階の逆伝播（seed の各要素 ≤ 1）: g へは Σ|t|、dev2 へは k=kc·|g|、2 回の中心化で 4k、
        // x_ref の narrow は n 個の cotangent を吸う。
        st(nf * t, "Var の g への cotangent")?;
        let k = kc * gl;
        st(4.0 * k, "Var の x への cotangent")?;
        st(nf * 4.0 * k, "Var の x_ref への cotangent")?;
    }
    Ok(())
}

struct StdCheck<'a> {
    n: usize,
    nc: f64,
    dev2: &'a [f32],
    e0: &'a [i32],
    e1: &'a [i32],
    masked: &'a [bool],
    g: &'a [f32],
}

fn check_std(c: &StdCheck<'_>) -> Result<(), AutodiffError> {
    let nf = c.n as f64;
    let kappa = 1.0 / c.nc;
    for (l, lane) in c.dev2.chunks(c.n).enumerate() {
        if c.masked[l] || lane_has_nan(lane) {
            continue;
        }
        let s0 = 2f64.powi(c.e0[l]);
        let s1 = 2f64.powi(c.e1[l]);
        let dev_s: Vec<f64> = lane.iter().map(|&v| (v / pow2(c.e1[l])) as f64).collect();
        let ssv = dev_s.iter().map(|d| d * d).sum::<f64>() / c.nc;
        if ssv.is_nan() || ssv <= 0.0 {
            return Err(invalid(
                "Std の二乗和が 0 以下（構造が f32 で表現できない低分散）".into(),
            ));
        }
        let std_s = ssv.sqrt();
        let gl = c.g[l].abs() as f64;
        let max_ds = dev_s.iter().fold(0.0f64, |m, d| m.max(d.abs()));
        let maxq = max_ds / std_s * kappa;
        st(maxq, "Std の q")?;
        st(maxq * gl, "Std の q·g")?;
        st(nf * maxq, "Std の g への cotangent")?;
        let a = gl * kappa;
        let direct = a / std_s;
        st(direct, "Std の dev_s への直接 cotangent")?;
        let cot_den = dev_s.iter().map(|d| a * d.abs()).sum::<f64>() / (std_s * std_s);
        st(cot_den, "Std の分母 den への cotangent")?;
        let cot_ss = cot_den * 0.5 / std_s;
        st(cot_ss, "Std の ss への cotangent")?;
        let cot_each = cot_ss / c.nc;
        let m = direct + 2.0 * max_ds * cot_each;
        st(m, "Std の dev_s への cotangent")?;
        let cot_dev2 = m / s1;
        st(cot_dev2, "Std の dev2 への cotangent")?;
        let cot_x0 = (nf + 1.0) * 4.0 * cot_dev2;
        st(cot_x0, "Std の x0 への cotangent")?;
        st(cot_x0 / s0, "Std の x への cotangent")?;
    }
    Ok(())
}

fn check_l2(
    n: usize,
    xs: &[f32],
    zero: &[bool],
    exps: &[i32],
    g: &[f32],
) -> Result<(), AutodiffError> {
    let nf = n as f64;
    for (l, lane) in xs.chunks(n).enumerate() {
        if zero[l] || lane_has_nan(lane) {
            continue;
        }
        let ss = lane.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
        if ss.is_nan() || ss <= 0.0 {
            return Err(invalid(
                "L2 の二乗和が 0（スケール後に表現できない低振幅）".into(),
            ));
        }
        st(ss, "L2 の二乗和")?;
        let nrm = ss.sqrt();
        let gl = g[l].abs() as f64;
        let max_x = lane.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()));
        st(max_x / nrm * gl, "L2 の q·g")?;
        st(nf * max_x / nrm, "L2 の g への cotangent")?;
        let direct = gl / nrm;
        st(direct, "L2 の x_s への直接 cotangent")?;
        let cot_nrm = lane.iter().map(|&v| gl * (v as f64).abs()).sum::<f64>() / ss;
        let cot_ss = cot_nrm * 0.5 / nrm;
        st(cot_ss, "L2 の ss への cotangent")?;
        let m = direct + 2.0 * max_x * cot_ss;
        st(m, "L2 の x_s への cotangent")?;
        st(m / 2f64.powi(exps[l]), "L2 の x への cotangent")?;
    }
    Ok(())
}

struct PNormCheck<'a> {
    n: usize,
    p: f64,
    xs: &'a [f32],
    zelem: &'a [bool],
    exps: &'a [i32],
    g: &'a [f32],
}

fn check_pnorm(c: &PNormCheck<'_>) -> Result<(), AutodiffError> {
    let p = c.p;
    for (l, lane) in c.xs.chunks(c.n).enumerate() {
        let zl = &c.zelem[l * c.n..(l + 1) * c.n];
        if zl.iter().all(|&z| z) || lane_has_nan(lane) {
            continue;
        }
        let gl = c.g[l].abs() as f64;
        let a: Vec<f64> = lane
            .iter()
            .zip(zl)
            .map(|(&v, &z)| if z { 0.0 } else { (v as f64).abs() })
            .collect();
        let s: f64 = a
            .iter()
            .map(|&v| if v > 0.0 { v.powf(p) } else { 0.0 })
            .sum();
        st(s, "PNorm の Σ|x_s|^p")?;
        if s.is_nan() || s <= 0.0 {
            return Err(invalid(
                "PNorm の Σ|x_s|^p が 0（スケール後に表現できない）".into(),
            ));
        }
        // 実際の f32 グラフと同じ丸めで Σ・ノルム・a/ノルムを再現する。f64 では正でも
        // f32 のべき乗和がアンダーフローで 0 になると nrm=0・a/nrm=inf となり、
        // 二階微分が非有限になるため（個々の要素の 0 化は寄与が無視できるので許容）。
        let s32: f32 = a
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    (v as f32).powf(p as f32)
                } else {
                    0.0
                }
            })
            .sum();
        if !s32.is_finite() || s32 < f32::MIN_POSITIVE {
            return Err(invalid(
                "PNorm の f32 べき乗和 Σ|x_s|^p が 0／非正規／非有限（f32 でアンダーフロー）"
                    .into(),
            ));
        }
        let nrm32 = s32.powf(1.0 / p as f32);
        if !nrm32.is_finite() || nrm32 < f32::MIN_POSITIVE {
            return Err(invalid(
                "PNorm の f32 ノルムが 0／非正規／非有限（f32 でアンダーフロー）".into(),
            ));
        }
        if a.iter().any(|&v| !(v as f32 / nrm32).is_finite()) {
            return Err(invalid("PNorm の f32 の a/ノルムが非有限".into()));
        }
        let nrm = s.powf(1.0 / p);
        st(nrm, "PNorm のノルム (Σ)^(1/p)")?;
        if nrm.is_nan() || nrm <= 0.0 {
            return Err(invalid(
                "PNorm のノルムが 0（スケール後に表現できない）".into(),
            ));
        }
        let mut cot_nrm = 0.0f64;
        let mut cot_a_direct = vec![0.0f64; a.len()];
        let mut sum_rp = 0.0f64;
        for (i, &ai) in a.iter().enumerate() {
            if zl[i] {
                continue;
            }
            let r = ai / nrm;
            let rp = r.powf(p - 1.0);
            st(rp, "PNorm の (|x_s|/‖x_s‖)^(p-1)")?;
            st(rp * gl, "PNorm の rp·g")?;
            sum_rp += rp;
            let cot_r = gl * (p - 1.0).abs() * r.powf(p - 2.0);
            st(cot_r, "PNorm の r への cotangent")?;
            cot_a_direct[i] = cot_r / nrm;
            cot_nrm += cot_r * r / nrm;
        }
        st(sum_rp, "PNorm の g への cotangent")?;
        st(cot_nrm, "PNorm のノルムへの cotangent")?;
        let cot_s = cot_nrm * (1.0 / p) * s.powf(1.0 / p - 1.0);
        st(cot_s, "PNorm の Σ への cotangent")?;
        let s_scale = 2f64.powi(c.exps[l]);
        for (i, &ai) in a.iter().enumerate() {
            if zl[i] {
                continue;
            }
            let cot_a = cot_a_direct[i] + cot_s * p * ai.powf(p - 1.0);
            st(cot_a, "PNorm の |x_s| への cotangent")?;
            st(cot_a / s_scale, "PNorm の x への cotangent")?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// 1 階 VJP の子テープ記録
// ---------------------------------------------------------------------

/// 統計縮約ノードの方式（`build_cgrads` の腕が組み立てる）。
#[derive(Clone, Copy)]
pub(super) enum StatKind {
    Var { correction: usize },
    Std { correction: usize },
    L1,
    L2,
    PNorm { p: f32 },
}

/// 統計縮約ノードの記述。
#[derive(Clone, Copy)]
pub(super) struct StatSpec {
    pub(super) input: NodeId,
    pub(super) dim: Option<usize>,
    pub(super) kind: StatKind,
}

/// 統計縮約ノード（入力 `spec.input`）の d_input を子テープへ記録する。
pub(super) fn stat_varnorm_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    x_m: &Var<'c>,
    spec: StatSpec,
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let StatSpec { input, dim, kind } = spec;
    let x_val = materialize_fallible(parent_nodes, parent.ops(), input)?;
    let in_shape = x_val.shape().to_vec();
    let expect = reduce_out_shape(&in_shape, dim).map_err(AutodiffError::Shape)?;
    if g.shape() != expect || x_m.shape() != in_shape {
        return Err(invalid(format!(
            "統計縮約の shape 不整合（入力 {in_shape:?}・dim {dim:?}・期待 {expect:?}・写し {:?}・上流 {:?}。内部契約違反）",
            x_m.shape(),
            g.shape()
        )));
    }
    if in_shape.contains(&0) {
        return Ok(child.var_no_grad(&Tensor::zeros(&in_shape)?));
    }
    let g_val = child_value(child, g)?;
    let g_data = dense_vec(&g_val);
    check_g_finite(&g_data)?;
    let lay = Lanes::new(&in_shape, dim)?;
    let (lanes, n) = (lay.lanes(), lay.n);
    if g_data.len() != lanes {
        return Err(invalid(
            "統計縮約の上流の要素数がレーン数と不一致（内部契約違反）".into(),
        ));
    }
    let x_data = dense_vec(x_val);
    let xl = lay.lane_major(&x_data);
    let g2 = g.contiguous()?.reshape(&[lanes, 1])?;
    let ones_col = col_const(child, vec![1.0; lanes])?;
    let zeros_ln = child.var_no_grad(&Tensor::zeros(&[lanes, n])?);

    let (composed, host) = match kind {
        StatKind::L1 => {
            // 構造は x に触れない 3 分岐（モジュール doc）。x の符号は親テープの記録値から作る。
            let pos = bool_tensor(x_data.iter().map(|&v| v > 0.0).collect(), &in_shape)?;
            let neg = bool_tensor(x_data.iter().map(|&v| v < 0.0).collect(), &in_shape)?;
            let g_b = bcast_upstream(g, &in_shape, dim)?;
            let zeros = child.var_no_grad(&Tensor::zeros(&in_shape)?);
            let inner = Var::where_cond(&neg, &g_b.neg()?, &zeros)?;
            let composed = Var::where_cond(&pos, &g_b, &inner)?;
            let host = vector_norm_vjp(x_val, VectorNormOrd::L1, dim, &g_val);
            (composed, host)
        }
        StatKind::Var { correction } => {
            let nc = n as f64 - correction as f64;
            if nc.is_nan() || nc <= 0.0 {
                return Err(invalid(format!(
                    "自由度不足（n={n}・correction={correction}）"
                )));
            }
            check_var(n, nc, &xl, &g_data)?;
            let x2 = lay.to_lane_form(x_m)?;
            let dev2 = stable_centering(&x2)?;
            let t = dev2.mul(&scalar_const(child, (2.0 / nc) as f32)?)?;
            let cg2 = t.mul(&g2)?;
            let host = var_vjp(x_val, dim, correction, &g_val);
            (lay.restore_lanes(&cg2)?, host)
        }
        StatKind::Std { correction } => {
            let nc = n as f64 - correction as f64;
            if nc.is_nan() || nc <= 0.0 {
                return Err(invalid(format!(
                    "自由度不足（n={n}・correction={correction}）"
                )));
            }
            let x2 = lay.to_lane_form(x_m)?;
            // S0: x0 = x / s0（s0 はレーンの max|x| 以上の 2 のべき乗）。
            let e0 = lane_exps(&xl, n);
            let x0 = x2.div(&col_const(child, e0.iter().map(|&e| pow2(e)).collect())?)?;
            let dev2 = stable_centering(&x0)?;
            // S1: 子テープ上の dev2 の実値から決める（LayerNorm と同じ「実値から決める」規約）。
            let dev2_data = dense_vec(&child_value(child, &dev2)?);
            let e1 = lane_exps(&dev2_data, n);
            let masked: Vec<bool> = xl
                .chunks(n)
                .zip(dev2_data.chunks(n))
                .map(|(lane, dl)| {
                    std_zero_lane(lane, nc) || (!lane_has_nan(dl) && dl.iter().all(|&v| v == 0.0))
                })
                .collect();
            check_std(&StdCheck {
                n,
                nc,
                dev2: &dev2_data,
                e0: &e0,
                e1: &e1,
                masked: &masked,
                g: &g_data,
            })?;
            let dev_s = dev2.div(&col_const(child, e1.iter().map(|&e| pow2(e)).collect())?)?;
            let nc_c = scalar_const(child, nc as f32)?;
            let ss = dev_s.mul(&dev_s)?.sum(Some(1))?.unsqueeze(1)?.div(&nc_c)?;
            let mask_col = bool_tensor(masked, &[lanes, 1])?;
            let ss_safe = Var::where_cond(&mask_col, &ones_col, &ss)?;
            let std_s = ss_safe.sqrt()?;
            let den = Var::where_cond(&mask_col, &ones_col, &std_s)?;
            let q = dev_s
                .div(&den)?
                .mul(&scalar_const(child, (1.0 / nc) as f32)?)?;
            let cg2 = Var::where_cond(&mask_col, &zeros_ln, &q.mul(&g2)?)?;
            let host = std_vjp(x_val, dim, correction, &g_val);
            (lay.restore_lanes(&cg2)?, host)
        }
        StatKind::L2 => {
            let x2 = lay.to_lane_form(x_m)?;
            let exps = lane_exps(&xl, n);
            let scales: Vec<f32> = exps.iter().map(|&e| pow2(e)).collect();
            let xs_host: Vec<f32> = xl
                .chunks(n)
                .zip(&scales)
                .flat_map(|(lane, &s)| lane.iter().map(move |&v| v / s))
                .collect();
            let zero: Vec<bool> = xl
                .chunks(n)
                .map(|lane| lane.iter().all(|&v| v == 0.0))
                .collect();
            check_l2(n, &xs_host, &zero, &exps, &g_data)?;
            let x_s = x2.div(&col_const(child, scales)?)?;
            let ss = x_s.mul(&x_s)?.sum(Some(1))?.unsqueeze(1)?;
            let zero_col = bool_tensor(zero, &[lanes, 1])?;
            let nrm = Var::where_cond(&zero_col, &ones_col, &ss)?.sqrt()?;
            let q = x_s.div(&nrm)?;
            let cg2 = Var::where_cond(&zero_col, &zeros_ln, &q.mul(&g2)?)?;
            let host = vector_norm_vjp(x_val, VectorNormOrd::L2, dim, &g_val);
            (lay.restore_lanes(&cg2)?, host)
        }
        StatKind::PNorm { p } => {
            let x2 = lay.to_lane_form(x_m)?;
            let exps = lane_exps(&xl, n);
            let scales: Vec<f32> = exps.iter().map(|&e| pow2(e)).collect();
            let xs_host: Vec<f32> = xl
                .chunks(n)
                .zip(&scales)
                .flat_map(|(lane, &s)| lane.iter().map(move |&v| v / s))
                .collect();
            let zelem: Vec<bool> = xs_host.iter().map(|&v| v == 0.0).collect();
            let zlane: Vec<bool> = zelem.chunks(n).map(|z| z.iter().all(|&b| b)).collect();
            check_pnorm(&PNormCheck {
                n,
                p: p as f64,
                xs: &xs_host,
                zelem: &zelem,
                exps: &exps,
                g: &g_data,
            })?;
            let sign: Vec<f32> = xs_host
                .iter()
                .map(|&v| {
                    if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let zelem_t = bool_tensor(zelem, &[lanes, n])?;
            let zlane_t = bool_tensor(zlane, &[lanes, 1])?;
            let ones_ln = child.var_no_grad(&Tensor::full(&[lanes, n], 1.0f32)?);
            let x_s = x2.div(&col_const(child, scales)?)?;
            let a = Var::where_cond(&zelem_t, &ones_ln, &x_s.abs()?)?;
            let pw = Var::where_cond(&zelem_t, &zeros_ln, &a.pow_scalar(p)?)?;
            let s_sum = pw.sum(Some(1))?.unsqueeze(1)?;
            let s_safe = Var::where_cond(&zlane_t, &ones_col, &s_sum)?;
            let nrm = s_safe.pow_scalar((1.0 / p as f64) as f32)?;
            let rp = a.div(&nrm)?.pow_scalar((p as f64 - 1.0) as f32)?;
            let sign_c = child.var_no_grad(&Tensor::new(sign, &[lanes, n])?);
            let cg2 = Var::where_cond(&zelem_t, &zeros_ln, &sign_c.mul(&rp)?.mul(&g2)?)?;
            let host = pnorm_vjp(x_val, p, dim, &g_val);
            (lay.restore_lanes(&cg2)?, host)
        }
    };
    push_v3(child, &composed, host)
}

/// `std_vjp` と同じ順序・精度で `std` の f32 丸めがゼロになるレーンか。
fn std_zero_lane(lane: &[f32], nc: f64) -> bool {
    let n = lane.len() as f64;
    let mut mean_acc = 0.0f64;
    for &v in lane {
        mean_acc += v as f64;
    }
    let mean = mean_acc / n;
    let mut sq_acc = 0.0f64;
    for &v in lane {
        let d = v as f64 - mean;
        sq_acc += d * d;
    }
    ((sq_acc / nc).sqrt() as f32) == 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_major_reorders_outer_axis_inner() {
        // shape [2, 3, 2]・axis 1: レーン (o,i) ごとに axis 方向 3 要素が連続する。
        let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let out = lane_major(&data, 2, 3, 2);
        assert_eq!(
            out,
            vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0, 6.0, 8.0, 10.0, 7.0, 9.0, 11.0]
        );
    }

    #[test]
    fn stage_bound_rejects_large_or_nonfinite() {
        assert!(st(1.0, "x").is_ok());
        assert!(st(STAGE_BOUND_GUARD, "x").is_err());
        assert!(st(f64::NAN, "x").is_err());
        assert!(st(f64::INFINITY, "x").is_err());
    }

    #[test]
    fn check_var_accepts_benign_and_rejects_huge_lane() {
        let g = [1.0f32];
        assert!(check_var(2, 1.0, &[1.0, 3.0], &g).is_ok());
        assert!(check_var(2, 1.0, &[-1e38, 1e38], &g).is_err());
        // NaN レーンは判定から外れる
        assert!(check_var(2, 1.0, &[f32::NAN, 1e38], &g).is_ok());
    }

    #[test]
    fn check_pnorm_rejects_tiny_p_with_long_lane() {
        let xs: Vec<f32> = (1..=33).map(|v| v as f32 / 33.0).collect();
        let zelem = vec![false; xs.len()];
        let c = |p: f64| PNormCheck {
            n: xs.len(),
            p,
            xs: &xs,
            zelem: &zelem,
            exps: &[0],
            g: &[1.0],
        };
        assert!(check_pnorm(&c(3.0)).is_ok());
        assert!(check_pnorm(&c(0.01)).is_err());
    }

    /// 回帰: x=[0.75,0.75], p=400, loss=norm_p(x)*1e-30 は f64 では正だが f32 のべき乗和が
    /// 0 になる。型付きエラーで拒否されること（PR #3942 Codex P1）。
    #[test]
    fn check_pnorm_rejects_f32_power_sum_underflow() {
        let xs = [0.75f32, 0.75];
        let zelem = [false, false];
        let c = PNormCheck {
            n: 2,
            p: 400.0,
            xs: &xs,
            zelem: &zelem,
            exps: &[0],
            g: &[1e-30],
        };
        assert!(check_pnorm(&c).is_err());
    }

    #[test]
    fn std_zero_lane_matches_constant_lane() {
        assert!(std_zero_lane(&[2.0, 2.0, 2.0], 2.0));
        assert!(!std_zero_lane(&[1.0, 2.0, 3.0], 2.0));
    }
}
