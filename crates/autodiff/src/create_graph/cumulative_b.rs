//! 累積 2 Op（`Cumprod`・`Logcumsumexp`）の `create_graph` 対応
//! （イシュー #3803・親 #3732・追跡表 #3717 項目 4。設計
//! `docs/autodiff-higher-order-grad-decision.md` §41、実装記録は §63）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `build_cgrads`（1 階 VJP の子テープ記録）から
//! 呼ばれる。forward の再生（`replay_op`）は公開 `Var::cumprod`／`logcumsumexp` への
//! 1 行委譲で足りるため親側に置く。線形再帰の展開は兄弟モジュール [`super::scan`] を使う。
//!
//! **方式**（値は 1 階 `grad.rs`／`tensor_core::cumulative` の VJP と同じホスト関数の結果を
//! `push_v3` で載せる。契約は REQ-2 のまま。構造は二階微分を流すための replayable な `Var` 列）:
//! - `Cumprod`（V3 + 除算を使わない scan）: 1 階は `d_x = L ⊙ S`。`S[a] = g[a] + x[a+1]·S[a+1]`
//!   は係数 `A[a] = x[a+1]` の線形再帰（scan 補助で `⌈log2 n⌉` 段）、`L` は排他的 prefix 積で
//!   `Concat([1], Narrow(y, 0, n-1))`（`y` は forward の出力の写し。追加の scan を要さない）。
//!   除算を使わないので零が何個あっても一階・二階とも厳密（除算形＋最初の零マスクでは
//!   `x=[2,0,3]`・`g=[0,0,1]` の `H[j,z]` を取りこぼす）。
//! - `Logcumsumexp`（V3 + 係数が [0,1] の再帰 scan）: 1 階は `d_x = w ⊙ T`、`w = exp(x - out)`・
//!   `T[j] = g[j] + c[j]·T[j+1]`・`c[j] = exp(out[j] - out[j+1])`（`out` は forward の出力の写し）。
//!   `c`・`w` は [0,1] に収まり overflow しない。
//!
//! 子テープへ積む Op は `Narrow`・`Concat`・`Mul`・`Add`・`Sub`・`Exp`・`Cumprod`／`Logcumsumexp`
//! （写し）・定数葉で、すべて `supports_create_graph() == true`（閉包性。§41.10）。
//!
//! **総括判定（§41.12 の具体化）**: 子テープへ構造を積む**前**に cgrad 関数の中で行う
//! （upstream `g` は `build_cgrads` でしか得られないため）。拒否は型付き `Err(Backward)` で、
//! 親の rollback により `child` は空に戻る。
//! - 入口規則（§33.14 論点 3）: `g` の非有限と `x` の `±inf` は拒否、`x` の NaN は伝播（NaN の lane は
//!   以降の判定を行わない）。`max|g|·axis_len` が `STAGE_BOUND_GUARD`（`2^125`）以上も拒否。
//! - `Cumprod`: 構造の各段（`A`・`S`・`L`・`L⊙S`）をホストで同じ f32 演算で再計算し、非有限または
//!   `2^125` 以上なら拒否。さらに二階の逆伝播の各段（「cotangent × 微分係数」）を、seed の各要素の
//!   絶対値を 1 として絶対値の f64 で歩いた上界（scan 側は `scan_cotangent_bound`、`L` 側は
//!   絶対値の入力を 1 階と同じ cumprod の VJP に通す）も同じ閾値で判定する。0 への underflow は
//!   拒否しない（underflow した項の絶対誤差は 1e-5 の救済内に収まる）。
//! - `Logcumsumexp`: f64 prefix の参照係数 `exp(out64[j]-out64[j+1])`・`exp(x-out64)` と f32 構造の
//!   係数 `c`・`w` を REQ-2 の統一複合判定（`tolerance::composite_close_f32`。新しい定数は作らない）で
//!   比べ、さらに Hessian 側の項 `c[j]·T[j+1]`・`w[j]·T[j]` にも同じ判定を課す。外れたら拒否。
//!
//! GPU: `cumprod` は CUDA／Metal にカーネルがあり（経路区分 `K`）、`logcumsumexp` はホストへ
//! フォールバックする（経路区分 `H`）。実機 parity は未実測（Phase 10 の #3857・#3859）。

use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::cumulative::{cumulative_layout, logcumsumexp_vjp_host};
use fandhe_ai_tensor_core::tolerance::composite_close_f32;

use super::scan::{linear_scan, scan_cotangent_bound, scan_lane_host};
use super::{child_value, push_v3};
use crate::create_graph_norm::STAGE_BOUND_GUARD;
use crate::error::AutodiffError;
use crate::grad::cumprod_vjp_along;
use crate::tape::{NodeId, Tape, TapeNode, materialize_fallible};
use crate::var::Var;

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!("create_graph: {msg}"))
}

/// 累積軸まわりの (outer, axis_len, inner)。lane は `(outer, inner)` の組ごと。
#[derive(Clone, Copy)]
struct Dims {
    outer: usize,
    n: usize,
    inner: usize,
}

impl Dims {
    fn new(shape: &[usize], dim: usize) -> Self {
        Dims {
            outer: shape[..dim].iter().product(),
            n: shape[dim],
            inner: shape[dim + 1..].iter().product(),
        }
    }

    fn lane<T: Copy>(&self, data: &[T], o: usize, i: usize) -> Vec<T> {
        (0..self.n)
            .map(|a| data[(o * self.n + a) * self.inner + i])
            .collect()
    }
}

/// 段の上界 `v` が `STAGE_BOUND_GUARD`（`2^125`）以上、または NaN なら `true`（拒否側に倒す）。
fn over_guard(v: f64) -> bool {
    v.is_nan() || v >= STAGE_BOUND_GUARD
}

/// 入口規則。`g` に非有限があれば拒否、`x` に `±inf` があれば拒否（NaN は通す）、
/// `max|g|·max(1, n)` が `STAGE_BOUND_GUARD` 以上でも拒否する。
fn check_entry(x: &[f32], g: &[f32], n: usize, what: &str) -> Result<(), AutodiffError> {
    let mut max_g = 0f64;
    for &v in g {
        if !v.is_finite() {
            return Err(invalid(format!(
                "{what} の上流勾配が非有限のため拒否（§41・§63 参照）"
            )));
        }
        max_g = max_g.max(f64::from(v).abs());
    }
    if x.iter().any(|v| v.is_infinite()) {
        return Err(invalid(format!(
            "{what} の入力に ±inf がある（NaN は伝播するが ±inf は拒否する。§41・§63 参照）"
        )));
    }
    if over_guard(max_g * (n.max(1) as f64)) {
        return Err(invalid(format!(
            "{what} の上流勾配が大きすぎ中間値が f32 範囲を超えうるため拒否（§41・§63 参照）"
        )));
    }
    Ok(())
}

/// ノード `id` の shape 整合を確かめ、入力 shape を返す。
fn checked_shape(
    parent_nodes: &[TapeNode],
    id: NodeId,
    input: NodeId,
    dim: usize,
    g: &Var<'_>,
) -> Result<Vec<usize>, AutodiffError> {
    let in_shape = parent_nodes[input.0].shape.clone();
    if dim >= in_shape.len() || parent_nodes[id.0].shape != in_shape || g.shape() != in_shape {
        return Err(invalid(format!(
            "累積 Op の shape 不整合（入力 {in_shape:?}・dim {dim}・記録値 {:?}・上流 {:?}。内部契約違反）",
            parent_nodes[id.0].shape,
            g.shape()
        )));
    }
    Ok(in_shape)
}

/// `dim` 軸の長さ 1 の定数（`fill` で満たす）を `shape` に合わせて作る。
fn unit_slab<'c>(
    child: &'c Tape,
    shape: &[usize],
    dim: usize,
    fill: f32,
) -> Result<Var<'c>, AutodiffError> {
    let mut s = shape.to_vec();
    s[dim] = 1;
    Ok(child.var_no_grad(&Tensor::full(&s, fill)?))
}

// ---------------------------------------------------------------------
// Cumprod
// ---------------------------------------------------------------------

/// `Cumprod` の総括判定（モジュール doc）。`x`・`g`・`y` は同 shape の密な値。
fn check_cumprod_stages(x: &[f32], g: &[f32], y: &[f32], d: Dims) -> Result<(), AutodiffError> {
    let n = d.n;
    for o in 0..d.outer {
        for i in 0..d.inner {
            let xl = d.lane(x, o, i);
            if xl.iter().any(|v| v.is_nan()) {
                continue;
            }
            let gl = d.lane(g, o, i);
            let yl = d.lane(y, o, i);
            let mut a = vec![0f32; n];
            a[..n - 1].copy_from_slice(&xl[1..]);
            let (s, stage_max) = scan_lane_host(&gl, &a, None);
            if over_guard(stage_max) {
                return Err(invalid(
                    "Cumprod の scan 構造の段が 2^125 以上または非有限のため拒否（§41・§63 参照）"
                        .into(),
                ));
            }
            let mut l = vec![1f32; n];
            l[1..].copy_from_slice(&yl[..n - 1]);
            let mut cot_s = vec![0f64; n];
            let mut cot_y = vec![0f32; n];
            for j in 0..n {
                let prod = l[j] * s[j];
                if over_guard(f64::from(l[j]).abs()) || over_guard(f64::from(prod).abs()) {
                    return Err(invalid(
                        "Cumprod の排他的 prefix 積または出力段が 2^125 以上または非有限のため拒否（§41・§63 参照）"
                            .into(),
                    ));
                }
                // seed の各要素の絶対値 1 以下の前提で、`d_x = L ⊙ S` への cotangent は
                // `S` には `|L|`、`L` には `|S|` が流れる（`L[j] = y[j-1]` なので `y[j-1]` へ）。
                cot_s[j] = f64::from(l[j]).abs();
                if j >= 1 {
                    cot_y[j - 1] = s[j].abs();
                }
            }
            // 二階の逆伝播の各段（「cotangent × 微分係数」）の上界を、絶対値の f64 で歩く。
            let (_, cot_a, walk_max) = scan_cotangent_bound(&gl, &a, &cot_s);
            if over_guard(walk_max) {
                return Err(invalid(
                    "Cumprod の二階の逆伝播の中間値が 2^125 以上または非有限のため拒否（§41・§63 参照）"
                        .into(),
                ));
            }
            // `y` 経由の cotangent は 1 階と同じ cumprod の VJP（`WideFloat`）に、絶対値の入力で通す。
            let abs_x = Tensor::new(xl.iter().map(|v| v.abs()).collect::<Vec<f32>>(), &[n])?;
            let cot_y_t = Tensor::new(cot_y, &[n])?;
            let cot_x_l = crate::eval::dense_vec(&cumprod_vjp_along(&abs_x, &cot_y_t, 0));
            for (m, v) in cot_x_l.iter().enumerate() {
                // `A[a] = x[a+1]` への cotangent も同じ `x[m]` に加わる。
                let from_a = if m >= 1 { cot_a[m - 1] } else { 0.0 };
                if over_guard(f64::from(*v).abs() + from_a) {
                    return Err(invalid(
                        "Cumprod の二階の入力 x への cotangent が 2^125 以上または非有限のため拒否（§41・§63 参照）"
                            .into(),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// `Cumprod` の d_input を子テープへ記録する（V3）。`x_m`・`y_m` は入力・出力の子テープ上の写し。
pub(super) fn cumprod_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, usize),
    mirrors: (&Var<'c>, &Var<'c>),
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (id, input, dim) = spec;
    let (x_m, y_m) = mirrors;
    let in_shape = checked_shape(parent_nodes, id, input, dim, g)?;
    if in_shape.contains(&0) {
        return Ok(child.var_no_grad(&Tensor::zeros(&in_shape)?));
    }
    let d = Dims::new(&in_shape, dim);
    let input_val = materialize_fallible(parent_nodes, parent.ops(), input)?;
    let out_val = materialize_fallible(parent_nodes, parent.ops(), id)?;
    let g_value = child_value(child, g)?;
    let (x, y, gv) = (
        crate::eval::dense_vec(input_val),
        crate::eval::dense_vec(out_val),
        crate::eval::dense_vec(&g_value),
    );
    check_entry(&x, &gv, d.n, "Cumprod")?;
    check_cumprod_stages(&x, &gv, &y, d)?;

    let n = d.n;
    let (l, s) = if n == 1 {
        (unit_slab(child, &in_shape, dim, 1.0)?, *g)
    } else {
        let a = Var::cat(
            &[
                x_m.narrow(dim, 1, n - 1)?,
                unit_slab(child, &in_shape, dim, 0.0)?,
            ],
            dim,
        )?;
        let l = Var::cat(
            &[
                unit_slab(child, &in_shape, dim, 1.0)?,
                y_m.narrow(dim, 0, n - 1)?,
            ],
            dim,
        )?;
        (l, linear_scan(child, g, &a, dim, None)?)
    };
    let composed = l.mul(&s)?;
    let host = cumprod_vjp_along(input_val, &g_value, dim);
    push_v3(child, &composed, host)
}

// ---------------------------------------------------------------------
// Logcumsumexp
// ---------------------------------------------------------------------

/// `log(exp(a) + exp(b))` を f64 で安定に求める（`tensor_core::cumulative` の私的関数と
/// 同じ分岐順。参照値の再計算専用で、tensor-core の公開面は増やさない）。
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b && !a.is_finite() {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

fn reject_coeff(what: &str) -> AutodiffError {
    invalid(format!(
        "Logcumsumexp の {what} が f64 参照と REQ-2 統一複合判定で一致せず（f32 の出力の丸めによる）拒否（§41・§63 参照）"
    ))
}

/// `Logcumsumexp` の総括判定（モジュール doc）。`out` は forward の記録値（f32）。
fn check_logcumsumexp_stages(
    x: &[f32],
    g: &[f32],
    out: &[f32],
    d: Dims,
) -> Result<(), AutodiffError> {
    let n = d.n;
    for o in 0..d.outer {
        for i in 0..d.inner {
            let xl = d.lane(x, o, i);
            if xl.iter().any(|v| v.is_nan()) {
                continue;
            }
            let gl = d.lane(g, o, i);
            let ol = d.lane(out, o, i);
            if ol.iter().any(|v| !v.is_finite()) {
                return Err(invalid(
                    "Logcumsumexp の出力が f32 で非有限のため拒否（§41・§63 参照）".into(),
                ));
            }
            let mut out64 = vec![0f64; n];
            for a in 0..n {
                let v = f64::from(xl[a]);
                out64[a] = if a == 0 {
                    v
                } else {
                    log_add_exp(out64[a - 1], v)
                };
            }
            // 係数 c（j < n-1）と w の f32 構造値・f64 参照値。
            let mut c32 = vec![0f32; n];
            let mut c64 = vec![0f64; n];
            for j in 0..n - 1 {
                c32[j] = (ol[j] - ol[j + 1]).exp();
                c64[j] = (out64[j] - out64[j + 1]).exp();
                if !composite_close_f32(c32[j], c64[j] as f32) {
                    return Err(reject_coeff("係数 exp(out[j]-out[j+1])"));
                }
            }
            let mut w32 = vec![0f32; n];
            let mut w64 = vec![0f64; n];
            for j in 0..n {
                w32[j] = (xl[j] - ol[j]).exp();
                w64[j] = (f64::from(xl[j]) - out64[j]).exp();
                if !composite_close_f32(w32[j], w64[j] as f32) {
                    return Err(reject_coeff("係数 exp(x-out)"));
                }
            }
            // T の f32 構造（scan の木順）と f64 逐次参照。
            let (t32, stage_max) = scan_lane_host(&gl, &c32, None);
            if over_guard(stage_max) {
                return Err(invalid(
                    "Logcumsumexp の scan 構造の段が 2^125 以上または非有限のため拒否（§41・§63 参照）"
                        .into(),
                ));
            }
            let mut t64 = vec![0f64; n];
            for j in (0..n).rev() {
                let next = if j + 1 < n { c64[j] * t64[j + 1] } else { 0.0 };
                t64[j] = f64::from(gl[j]) + next;
            }
            for j in 0..n {
                if j + 1 < n {
                    let h32 = c32[j] * t32[j + 1];
                    let h64 = c64[j] * t64[j + 1];
                    if !composite_close_f32(h32, h64 as f32) {
                        return Err(reject_coeff("Hessian 項 c[j]·T[j+1]"));
                    }
                }
                let h32 = w32[j] * t32[j];
                let h64 = w64[j] * t64[j];
                if !composite_close_f32(h32, h64 as f32) {
                    return Err(reject_coeff("Hessian 項 w[j]·T[j]"));
                }
            }
        }
    }
    Ok(())
}

/// `Logcumsumexp` の d_input を子テープへ記録する（V3）。`x_m`・`y_m` は入力・出力の子テープ上の写し。
pub(super) fn logcumsumexp_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, usize),
    mirrors: (&Var<'c>, &Var<'c>),
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (id, input, dim) = spec;
    let (x_m, y_m) = mirrors;
    let in_shape = checked_shape(parent_nodes, id, input, dim, g)?;
    if in_shape.contains(&0) {
        return Ok(child.var_no_grad(&Tensor::zeros(&in_shape)?));
    }
    let d = Dims::new(&in_shape, dim);
    let input_val = materialize_fallible(parent_nodes, parent.ops(), input)?;
    let out_val = materialize_fallible(parent_nodes, parent.ops(), id)?;
    let g_value = child_value(child, g)?;
    let (x, y, gv) = (
        crate::eval::dense_vec(input_val),
        crate::eval::dense_vec(out_val),
        crate::eval::dense_vec(&g_value),
    );
    check_entry(&x, &gv, d.n, "Logcumsumexp")?;
    check_logcumsumexp_stages(&x, &gv, &y, d)?;

    let n = d.n;
    let t = if n == 1 {
        *g
    } else {
        let c = y_m
            .narrow(dim, 0, n - 1)?
            .sub(&y_m.narrow(dim, 1, n - 1)?)?
            .exp();
        let a = Var::cat(&[c, unit_slab(child, &in_shape, dim, 0.0)?], dim)?;
        linear_scan(child, g, &a, dim, None)?
    };
    let w = x_m.sub(y_m)?.exp();
    let composed = w.mul(&t)?;
    let layout = cumulative_layout(&in_shape, dim).map_err(AutodiffError::Shape)?;
    let host = logcumsumexp_vjp_host(&x, &gv, &layout).map_err(AutodiffError::Shape)?;
    push_v3(child, &composed, Tensor::new(host, &in_shape)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d1(n: usize) -> Dims {
        Dims {
            outer: 1,
            n,
            inner: 1,
        }
    }

    fn cumprod_y(x: &[f32]) -> Vec<f32> {
        let mut acc = 1f32;
        x.iter()
            .map(|v| {
                acc *= *v;
                acc
            })
            .collect()
    }

    fn lse_out(x: &[f32]) -> Vec<f32> {
        let mut acc = 0f64;
        x.iter()
            .enumerate()
            .map(|(a, v)| {
                acc = if a == 0 {
                    f64::from(*v)
                } else {
                    log_add_exp(acc, f64::from(*v))
                };
                acc as f32
            })
            .collect()
    }

    fn cp(x: &[f32], g: &[f32]) -> Result<(), AutodiffError> {
        check_entry(x, g, x.len(), "Cumprod")?;
        check_cumprod_stages(x, g, &cumprod_y(x), d1(x.len()))
    }

    fn lse(x: &[f32], g: &[f32]) -> Result<(), AutodiffError> {
        check_entry(x, g, x.len(), "Logcumsumexp")?;
        check_logcumsumexp_stages(x, g, &lse_out(x), d1(x.len()))
    }

    #[test]
    fn cumprod_check_accepts_underflow_and_zeros_rejects_overflow() {
        // 0 への underflow（L[2] = 1e-60 → 0）は拒否しない。
        assert!(cp(&[1e-30, 1e-30, 1e30], &[1.0, 1.0, 1.0]).is_ok());
        assert!(cp(&[2.0, 3.0, 4.0], &[1.0, 1.0, 1.0]).is_ok());
        assert!(cp(&[2.0, 0.0, 3.0], &[0.0, 0.0, 1.0]).is_ok());
        assert!(cp(&[0.0, 0.0, 3.0], &[1.0, 1.0, 1.0]).is_ok());
        assert!(cp(&[0.0, 2.0, 0.0, 1.0], &[1.0, 1.0, 1.0, 1.0]).is_ok());
        // NaN の lane は伝播（判定しない）。
        assert!(cp(&[1.0, f32::NAN, 2.0], &[1.0, 1.0, 1.0]).is_ok());
        // 部分積が 2^125 を超えるものは拒否（零を含んでも W の上界で拒否）。
        let mut big = vec![1e30f32; 40];
        big[0] = 0.0;
        assert!(cp(&big, &[1.0; 40]).is_err());
        assert!(cp(&[1e30, 1e30, 1e30], &[1.0, 1.0, 1.0]).is_err());
        // 二階の cotangent が実際に f32 に収まる大きな入力は受理される（部分積 1e60 を作らない）。
        assert!(cp(&[1e30, 1e30], &[1.0, 1.0]).is_ok());
        // 入口規則。
        assert!(cp(&[1.0, f32::INFINITY], &[1.0, 1.0]).is_err());
        assert!(cp(&[1.0, 2.0], &[f32::NAN, 1.0]).is_err());
        assert!(cp(&[1.0, 2.0], &[3e38, 1.0]).is_err());
    }

    #[test]
    fn logcumsumexp_check_accepts_moderate_rejects_rounding_loss() {
        assert!(lse(&[0.0, 0.0], &[0.0, 1.0]).is_ok());
        assert!(lse(&[-1000.0, 1000.0], &[1.0, 1.0]).is_ok());
        assert!(lse(&[-100.0, -100.0, 80.0], &[1.0, 1.0, 1.0]).is_ok());
        assert!(lse(&[3.0, 3.0, 3.0, 3.0], &[1.0, -1.0, 1.0, 0.5]).is_ok());
        assert!(lse(&[1.0, f32::NAN], &[1.0, 1.0]).is_ok());
        // f32 の出力が ln2 の差を失う（w=1 対 参照 0.5）。
        assert!(lse(&[1e8, 1e8], &[0.0, 1.0]).is_err());
        assert!(lse(&[1e8, 1e8 + 1.0, 1e8], &[1.0, 1.0, 1.0]).is_err());
        // 係数の相対誤差が 1e-3 を超える。
        assert!(lse(&[5e4, 5e4], &[1.0, 1.0]).is_err());
        // 入口規則。
        assert!(lse(&[f32::NEG_INFINITY, 0.0], &[1.0, 1.0]).is_err());
        assert!(lse(&[0.0, 0.0], &[3e38, 1.0]).is_err());
        assert!(lse(&[0.0, 0.0], &[3e38, -3e38]).is_err());
    }
}
