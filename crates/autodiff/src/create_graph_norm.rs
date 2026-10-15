//! `Op::BatchNorm` の create_graph（二階微分）用 VJP 構築
//! （イシュー #3079・親 #3075。方針は `docs/autodiff-higher-order-grad-decision.md`
//! §21.1 (5)・§21.3、実装記録は §23）。
//!
//! **役割**: `create_graph.rs::build_cgrads` の `Op::BatchNorm` 腕から呼ばれ、
//! 1 階 VJP（`grad.rs::batch_norm_vjp_channels`）の結果を子テープ上の
//! `Var` として再現する。値は 1 階と同じホスト関数で計算し（V3。f64 縮約契約
//! を維持するため bit 同一）、二階の微分が流れる構造は replayable な `Var`
//! 演算（`Mul`・`Sub`・`Mean`・`Sum`・`Sqrt`・`Div`・`Reshape`・`Permute`・
//! `Contiguous` のみ。`Op::supports_create_graph` が `true` の Op に閉じる＝
//! 孫テープ再生で閉じる）の合成で組む。
//!
//! **方式の割り当て（出力ごと）**:
//! - train の `dx`・両モードの `dw`／`db`: V3（値はホスト関数、構造は合成グラフ）
//! - eval の `dx`: V1（統計量が定数なので `dx̂ ⊙ rstd` の要素積だけで足りる）
//!
//! **数値契約（S1。§21.1 (5)）**: 構造側の f32 中間値が 1 階で有限な入力
//! （`x=1e20` 系・`[3e38,-3e38]`・巨大な上流勾配）で overflow しないよう、
//! すべて 2 のべき乗のホスト定数でスケールしてから統計量を作る。2 のべき乗
//! 倍は（非正規化域を除き）厳密なので、`x̂ = u·rstd_s` はスケールに依らず
//! 真の `x̂` と一致し、これらの定数を微分上の定数として扱っても勾配は厳密
//! に正しい。
//!
//! running stats（`nn::BatchNorm*` の buffer・`fixed_stats` payload）は
//! 本モジュールも `replay_op` も更新しない（`Var::` プリミティブとホスト関数
//! だけを呼び、`RefCell` の buffer へ触れる経路がない）。

use fandhe_ai_tensor_core::{ScalarBinaryOp, ScalarUnaryOp, Tensor};

use crate::error::AutodiffError;
use crate::eval::{build_tensor, dense_vec};
use crate::grad::batch_norm_vjp_channels;
use crate::tape::{Op, Tape, materialize_fallible};
use crate::var::Var;

/// 2 のべき乗スケールに使う指数の下限・上限（f32 の正規数範囲）。
const POW2_MIN_EXP: i32 = -126;
const POW2_MAX_EXP: i32 = 127;
/// `mul_pow2_per_row` が 1 段で掛ける指数の絶対値の上限（`2^100` は f32 で有限）。
const POW2_CHUNK: i32 = 100;
/// 入力側の前置スケール `p` の指数を `[-BAL, BAL]` に制限する値。`p` をそのまま
/// （最大 127）使うと、逆伝播で `2^(k+p)` 級の倍率が `inv_p`（forward 先頭）へ
/// 到達する前に先に掛かり、巨大定数入力（`x=mean=3e38`・`var=0` 等）で二階勾配が
/// overflow する（forward 値・最終倍率 `rstd` は有限）。制限すると逆伝播の各段の
/// 累積倍率が `rstd·2^BAL` 以下に収まる（決定記録 §23 の有限性契約）。`|p|<=BAL`
/// の入力では従来と bit 同一。
const POW2_BAL: i32 = 64;

/// `max_abs` 以上の最小の 2 のべき乗の指数を `[-126, 127]` に clamp して返す。
/// 0・NaN は 0（スケール 1）、`inf` は 127。
pub(crate) fn pow2_ceil_exp(max_abs: f64) -> i32 {
    if max_abs.is_nan() || max_abs == 0.0 {
        return 0;
    }
    if max_abs.is_infinite() {
        return POW2_MAX_EXP;
    }
    let bits = max_abs.abs().to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i32;
    if exp_field == 0 {
        // f64 の非正規数は f32 の下限より遥かに小さい。
        return POW2_MIN_EXP;
    }
    let e = exp_field - 1023;
    let mant = bits & ((1u64 << 52) - 1);
    let ceil = if mant == 0 { e } else { e + 1 };
    ceil.clamp(POW2_MIN_EXP, POW2_MAX_EXP)
}

/// `2^e`（`e` は `[-1022, 1023]`）を f64 で返す。
fn pow2_f64(e: i32) -> f64 {
    let e = e.clamp(-1022, 1023);
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// 子テープの `v` の実体値を平坦化して返す（`RefCell` 借用はこの関数内で閉じる）。
fn host_values<'c>(child: &'c Tape, v: &Var<'c>) -> Result<Vec<f32>, AutodiffError> {
    let nodes = child.nodes.borrow();
    Ok(dense_vec(materialize_fallible(
        &nodes,
        child.ops(),
        v.node_id(),
    )?))
}

/// 子テープの `v` の実体値（Tensor）を返す（借用はこの関数内で閉じる）。
fn host_tensor<'c>(child: &'c Tape, v: &Var<'c>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = child.nodes.borrow();
    Ok(materialize_fallible(&nodes, child.ops(), v.node_id())?.clone())
}

/// 長さ `c` の定数列を `[c, 1]` の定数葉（勾配なし）にする。
fn const_col<'c>(child: &'c Tape, vals: &[f32]) -> Result<Var<'c>, AutodiffError> {
    let t = Tensor::new(vals.to_vec(), &[vals.len(), 1])?;
    Ok(child.var_no_grad(&t))
}

/// `[c]` の `Var` を `[c, 1]` へ（`Mean`／`Sum` の出力を行へ broadcast する前段）。
fn as_col<'c>(v: &Var<'c>) -> Result<Var<'c>, AutodiffError> {
    let c = v.shape().iter().product::<usize>();
    v.contiguous()?.reshape(&[c, 1])
}

/// 各チャネル行（`[c, m]` の値 `vals`）の絶対値最大を返す。NaN は無視し、
/// `inf` を含むとき `saw_inf` を立てる。
fn row_abs_max(vals: &[f32], c: usize, m: usize) -> Vec<(f64, bool)> {
    (0..c)
        .map(|ch| {
            let mut mx = 0.0f64;
            let mut inf = false;
            for &v in &vals[ch * m..(ch + 1) * m] {
                if v.is_nan() {
                    continue;
                }
                if v.is_infinite() {
                    inf = true;
                    continue;
                }
                mx = mx.max((v as f64).abs());
            }
            (mx, inf)
        })
        .collect()
}

/// `[n, c, spatial]` 相当の `v`（任意 rank のチャネル第 1 軸テンソル）を
/// チャネル行 `[c, n*spatial]` へ並べ替える。
pub(crate) fn to_channel_rows<'c>(
    v: &Var<'c>,
    n: usize,
    c: usize,
    spatial: usize,
) -> Result<Var<'c>, AutodiffError> {
    v.contiguous()?
        .reshape(&[n, c, spatial])?
        .permute(&[1, 0, 2])?
        .contiguous()?
        .reshape(&[c, n * spatial])
}

/// [`to_channel_rows`] の逆変換（`x_shape` へ戻す）。
pub(crate) fn from_channel_rows<'c>(
    v: &Var<'c>,
    n: usize,
    c: usize,
    spatial: usize,
    x_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    v.contiguous()?
        .reshape(&[c, n, spatial])?
        .permute(&[1, 0, 2])?
        .contiguous()?
        .reshape(x_shape)
}

/// チャネル `ch` の行 `v: [c, m]` に `2^exps[ch]` を掛ける。指数の絶対値が
/// 大きい場合は `±100` ずつの段に分け、各チャネルの符号を段を通じて一定に
/// 保つ（最終値が表現可能なら途中で overflow／underflow しない）。
pub(crate) fn mul_pow2_per_row<'c>(
    child: &'c Tape,
    v: &Var<'c>,
    exps: &[i32],
) -> Result<Var<'c>, AutodiffError> {
    let max_abs = exps.iter().map(|e| e.unsigned_abs()).max().unwrap_or(0) as i32;
    if max_abs == 0 {
        return Ok(*v);
    }
    let stages = (max_abs + POW2_CHUNK - 1) / POW2_CHUNK;
    let mut rem = exps.to_vec();
    let mut cur = *v;
    for _ in 0..stages {
        let step: Vec<i32> = rem
            .iter()
            .map(|&r| r.clamp(-POW2_CHUNK, POW2_CHUNK))
            .collect();
        for (r, s) in rem.iter_mut().zip(step.iter()) {
            *r -= *s;
        }
        let consts: Vec<f32> = step.iter().map(|&s| pow2_f64(s) as f32).collect();
        cur = cur.mul(&const_col(child, &consts)?)?;
    }
    Ok(cur)
}

/// train モードのチャネル行 `x_rows: [c, m]` から、スケール不変な正規化値
/// `x̂ = u·rstd_s`（`[c, m]`）・`rstd_s`（`[c, 1]`）・実効 `rstd` へ戻すための
/// 指数 `lg_p + lg_s2`（チャネルごと）を S1 方式で組む。
///
/// 手順（plan §3.4）: ①`p`（`max|x|` 以上の 2 のべき乗）で割る → ②先頭要素
/// 基準の安定な中心化と補正パス → ③`s2`（`max(max|dev2|, √eps/p)` 以上）で割って
/// `u` を作る → ④`eps_eff = eps/(p·s2)²` で `rstd_s = 1/√(mean(u²)+eps_eff)`。
/// 実効 `rstd = rstd_s / (p·s2)`。
pub(crate) fn scaled_row_xhat<'c>(
    child: &'c Tape,
    x_rows: &Var<'c>,
    eps: f32,
) -> Result<(Var<'c>, Var<'c>, Vec<i32>), AutodiffError> {
    let shape = x_rows.shape();
    let (c, m) = (shape[0], shape[1]);
    let xv = host_values(child, x_rows)?;
    let p_exp: Vec<i32> = row_abs_max(&xv, c, m)
        .iter()
        .map(|&(mx, inf)| {
            (if inf { POW2_MAX_EXP } else { pow2_ceil_exp(mx) }).clamp(-POW2_BAL, POW2_BAL)
        })
        .collect();
    let inv_p: Vec<f32> = p_exp.iter().map(|&e| pow2_f64(-e) as f32).collect();
    let xs = x_rows.mul(&const_col(child, &inv_p)?)?;
    // 先頭要素を基準に引いてから平均を引く（f32 の `Mean` 出力を早期に
    // 引いて桁落ちする入力 `[16777216, 16777218]` の対策。§21.3 #3078）。
    let x_ref = xs.narrow(1, 0, 1)?;
    let d = xs.sub(&x_ref)?;
    let dev = d.sub(&as_col(&d.mean(Some(1))?)?)?;
    let dev2 = dev.sub(&as_col(&dev.mean(Some(1))?)?)?;

    let dv = host_values(child, &dev2)?;
    let sqrt_eps = (eps as f64).sqrt();
    let s2_exp: Vec<i32> = row_abs_max(&dv, c, m)
        .iter()
        .zip(p_exp.iter())
        .map(|(&(mx, inf), &pe)| {
            if inf {
                POW2_MAX_EXP
            } else {
                pow2_ceil_exp(mx.max(sqrt_eps / pow2_f64(pe)))
            }
        })
        .collect();
    let inv_s2: Vec<f32> = s2_exp.iter().map(|&e| pow2_f64(-e) as f32).collect();
    let u = dev2.mul(&const_col(child, &inv_s2)?)?;

    let eps_eff: Vec<f32> = p_exp
        .iter()
        .zip(s2_exp.iter())
        .map(|(&pe, &se)| {
            let scale = pow2_f64(pe) * pow2_f64(se);
            (eps as f64 / (scale * scale)) as f32
        })
        .collect();
    let var_s = as_col(&u.mul(&u)?.mean(Some(1))?)?;
    let denom = var_s
        .add(&const_col(child, &eps_eff)?)?
        .scalar_unary(ScalarUnaryOp::Sqrt)?;
    let ones = const_col(child, &vec![1.0f32; c])?;
    let rstd_s = ones.scalar_binary(&denom, ScalarBinaryOp::Div)?;
    let xhat = u.mul(&rstd_s)?;
    let exps: Vec<i32> = p_exp
        .iter()
        .zip(s2_exp.iter())
        .map(|(a, b)| a + b)
        .collect();
    Ok((xhat, rstd_s, exps))
}

/// eval モード（固定統計）の正規化値 `x̂ = (x−mean)·rstd`（`[c, m]`）を、
/// `x`・`mean` を 2 のべき乗 `p` で割った差 `q`（`|q|≤2`）と `rstd` の
/// 仮数・指数分解から組む（`x` と `mean` が逆符号の大値でも差が overflow
/// しない）。
fn eval_row_xhat<'c>(
    child: &'c Tape,
    x_rows: &Var<'c>,
    mean: &[f32],
    rstd: &[f64],
) -> Result<Var<'c>, AutodiffError> {
    let shape = x_rows.shape();
    let (c, m) = (shape[0], shape[1]);
    let xv = host_values(child, x_rows)?;
    let p_exp: Vec<i32> = row_abs_max(&xv, c, m)
        .iter()
        .zip(mean.iter())
        .map(|(&(mx, inf), &mu)| {
            if inf || !mu.is_finite() {
                POW2_MAX_EXP
            } else {
                pow2_ceil_exp(mx.max((mu as f64).abs()))
            }
            .clamp(-POW2_BAL, POW2_BAL)
        })
        .collect();
    let inv_p: Vec<f32> = p_exp.iter().map(|&e| pow2_f64(-e) as f32).collect();
    let mean_p: Vec<f32> = mean
        .iter()
        .zip(p_exp.iter())
        .map(|(&mu, &e)| (mu as f64 * pow2_f64(-e)) as f32)
        .collect();
    let q = x_rows
        .mul(&const_col(child, &inv_p)?)?
        .sub(&const_col(child, &mean_p)?)?;
    // rstd = mant · 2^k（mant ∈ [1, 2)）。非有限・0 は分解せずそのまま掛ける。
    let mut mant = Vec::with_capacity(c);
    let mut exps = Vec::with_capacity(c);
    for (&r, &pe) in rstd.iter().zip(p_exp.iter()) {
        if r.is_finite() && r > 0.0 {
            let k = ((r.to_bits() >> 52) & 0x7ff) as i32 - 1023;
            mant.push((r / pow2_f64(k)) as f32);
            exps.push(k + pe);
        } else {
            mant.push(r as f32);
            exps.push(pe);
        }
    }
    let scaled = q.mul(&const_col(child, &mant)?)?;
    mul_pow2_per_row(child, &scaled, &exps)
}

/// `S`（構造グラフ）に `value`（ホスト関数で計算した値）を載せた
/// `Op::Contiguous`（VJP が恒等で前方値に依存しない）ノードを積む（V3）。
fn with_host_value<'c>(
    child: &'c Tape,
    structure: &Var<'c>,
    value: Tensor<f32>,
) -> Result<Var<'c>, AutodiffError> {
    if structure.shape() != value.shape() {
        return Err(AutodiffError::Backward(
            "create_graph: BatchNorm の構造グラフとホスト値の shape が不一致（内部契約違反）"
                .into(),
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

/// [`batch_norm_cgrads`] の入力（親テープ側の実測値と子テープの写し）。
pub(crate) struct BatchNormCgradInputs<'a, 'c> {
    pub x_mirror: Var<'c>,
    pub w_mirror: Option<Var<'c>>,
    pub has_bias: bool,
    pub eps: f32,
    pub fixed_stats: Option<&'a (Tensor<f32>, Tensor<f32>)>,
    /// 親テープ上の `x` の実体値（`[n, c, ...]`）。
    pub x_host: &'a Tensor<f32>,
    /// 親テープ上の `weight` の実体値。
    pub w_host: Option<&'a Tensor<f32>>,
    pub need_dx: bool,
    pub n: usize,
    pub c: usize,
    pub spatial: usize,
}

/// 子テープ上の `dx`／`dw`／`db`（要求されなかった出力・存在しない
/// パラメータは `None`）。
pub(crate) struct BatchNormCgrads<'c> {
    pub dx: Option<Var<'c>>,
    pub dw: Option<Var<'c>>,
    pub db: Option<Var<'c>>,
}

/// `Op::BatchNorm` の VJP を子テープ上の `Var` として構築する。`g` は
/// 出力 `y` への上流勾配（子テープ上。`x` と同 shape）。
pub(crate) fn batch_norm_cgrads<'c>(
    child: &'c Tape,
    inp: &BatchNormCgradInputs<'_, 'c>,
    g: &Var<'c>,
) -> Result<BatchNormCgrads<'c>, AutodiffError> {
    let (n, c, spatial) = (inp.n, inp.c, inp.spatial);
    let x_shape = inp.x_host.shape().to_vec();
    let g_rows = to_channel_rows(g, n, c, spatial)?;
    let x_rows = to_channel_rows(&inp.x_mirror, n, c, spatial)?;

    // 1 階 VJP と同じホスト関数で一度だけ値を得る（V3）。`dx` は train のみ使う。
    let g_val = host_tensor(child, g)?;
    let x_dense = dense_vec(inp.x_host);
    let w_dense = inp.w_host.map(dense_vec);
    let dy_dense = dense_vec(&g_val);
    let fixed_dense = inp
        .fixed_stats
        .map(|(mu, var)| (dense_vec(mu), dense_vec(var)));
    let (dx_host, dw_host, db_host) = batch_norm_vjp_channels(
        &x_dense,
        w_dense.as_deref(),
        inp.has_bias,
        inp.eps,
        n,
        c,
        spatial,
        &dy_dense,
        fixed_dense
            .as_ref()
            .map(|(mu, var)| (mu.as_slice(), var.as_slice())),
    );

    // `dx̂ = g ⊙ w`（1 階と同じ f32 積）。
    let dxhat = match &inp.w_mirror {
        Some(w) => g_rows.mul(&as_col(w)?)?,
        None => g_rows,
    };

    // 正規化値 x̂ と、train の実効 rstd 指数。
    let (xhat, train_ctx) = match inp.fixed_stats {
        None => {
            let (xhat, rstd_s, exps) = scaled_row_xhat(child, &x_rows, inp.eps)?;
            (xhat, Some((rstd_s, exps)))
        }
        Some((mu, var)) => {
            let mean = dense_vec(mu);
            let rstd: Vec<f64> = dense_vec(var)
                .iter()
                .map(|&v| 1.0f64 / (v as f64 + inp.eps as f64).sqrt())
                .collect();
            (eval_row_xhat(child, &x_rows, &mean, &rstd)?, None)
        }
    };

    let dx = if inp.need_dx {
        match &train_ctx {
            Some((rstd_s, p_s2_exps)) => {
                // dx̂ を行ごとの 2 のべき乗 s_g で割ってから縮約する（巨大な
                // 上流勾配での `Σdx̂` overflow を避ける。dx は dx̂ について線形）。
                let dv = host_values(child, &dxhat)?;
                let sg_exp: Vec<i32> = row_abs_max(&dv, c, n * spatial)
                    .iter()
                    .map(|&(mx, inf)| if inf { POW2_MAX_EXP } else { pow2_ceil_exp(mx) })
                    .collect();
                let inv_sg: Vec<f32> = sg_exp.iter().map(|&e| pow2_f64(-e) as f32).collect();
                let t = dxhat.mul(&const_col(child, &inv_sg)?)?;
                let mean_t = as_col(&t.mean(Some(1))?)?;
                let mean_tx = as_col(&t.mul(&xhat)?.mean(Some(1))?)?;
                let inner = t.sub(&mean_t)?.sub(&xhat.mul(&mean_tx)?)?;
                let scaled = inner.mul(rstd_s)?;
                let exps: Vec<i32> = sg_exp
                    .iter()
                    .zip(p_s2_exps.iter())
                    .map(|(sg, ps)| sg - ps)
                    .collect();
                let rows = mul_pow2_per_row(child, &scaled, &exps)?;
                let s = from_channel_rows(&rows, n, c, spatial, &x_shape)?;
                Some(with_host_value(child, &s, build_tensor(dx_host, &x_shape))?)
            }
            None => {
                // eval の dx は V1: 定数 rstd との要素積のみ。
                let (_, var) = inp.fixed_stats.ok_or_else(|| {
                    AutodiffError::Backward("create_graph: BatchNorm eval 統計欠落".into())
                })?;
                let rstd: Vec<f32> = dense_vec(var)
                    .iter()
                    .map(|&v| (1.0f64 / (v as f64 + inp.eps as f64).sqrt()) as f32)
                    .collect();
                let rows = dxhat.mul(&const_col(child, &rstd)?)?;
                Some(from_channel_rows(&rows, n, c, spatial, &x_shape)?)
            }
        }
    } else {
        None
    };

    let dw = match (&inp.w_mirror, dw_host) {
        (Some(_), Some(dw_host)) => {
            let s = g_rows.mul(&xhat)?.sum(Some(1))?;
            Some(with_host_value(child, &s, build_tensor(dw_host, &[c]))?)
        }
        _ => None,
    };
    let db = match db_host {
        Some(db_host) => {
            let s = g_rows.sum(Some(1))?;
            Some(with_host_value(child, &s, build_tensor(db_host, &[c]))?)
        }
        None => None,
    };
    Ok(BatchNormCgrads { dx, dw, db })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_tape() -> Tape {
        Tape::new_with_ops(crate::default_ops::naive_ops())
    }

    #[test]
    fn pow2_ceil_exp_rules() {
        assert_eq!(pow2_ceil_exp(0.0), 0);
        assert_eq!(pow2_ceil_exp(f64::NAN), 0);
        assert_eq!(pow2_ceil_exp(f64::INFINITY), POW2_MAX_EXP);
        assert_eq!(pow2_ceil_exp(1.0), 0);
        assert_eq!(pow2_ceil_exp(1.5), 1);
        assert_eq!(pow2_ceil_exp(0.5), -1);
        assert_eq!(pow2_ceil_exp(3.0e38), 128.min(POW2_MAX_EXP));
        assert_eq!(pow2_ceil_exp(1e-60), POW2_MIN_EXP);
        // eps の √ floor は呼び出し側で max を取る前提。
        assert_eq!(pow2_ceil_exp((1e-5f64).sqrt()), -8);
    }

    #[test]
    fn mul_pow2_per_row_handles_large_exponents_without_overflow() {
        let tape = new_tape();
        let v = tape.var_no_grad(&Tensor::new(vec![1.0f32, 1.0, 1.0, 1.0], &[2, 2]).unwrap());
        // 行 0: 2^-250（f32 では 0 へ underflow）。行 1: 2^120。
        let out = mul_pow2_per_row(&tape, &v, &[-120, 120]).unwrap();
        let vals = out.to_tensor();
        assert_eq!(vals.as_slice().unwrap()[0], 2.0f32.powi(-120));
        assert_eq!(vals.as_slice().unwrap()[2], 2.0f32.powi(120));
        // 2 段に分かれる指数（|e| > 100）でも、最終値が表現可能なら厳密。
        let big = tape
            .var_no_grad(&Tensor::new(vec![2.0f32.powi(-100), 2.0f32.powi(100)], &[2, 1]).unwrap());
        let out = mul_pow2_per_row(&tape, &big, &[200, -200]).unwrap();
        let vals = out.to_tensor();
        assert_eq!(vals.as_slice().unwrap()[0], 2.0f32.powi(100));
        assert_eq!(vals.as_slice().unwrap()[1], 2.0f32.powi(-100));
    }

    #[test]
    fn channel_rows_round_trip_for_rank2_rank3_rank4() {
        for shape in [vec![3usize, 2], vec![2, 3, 4], vec![2, 2, 2, 3]] {
            let (n, c, spatial) = fandhe_ai_tensor_core::batch_norm_layout(&shape).unwrap();
            let numel: usize = shape.iter().product();
            let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
            let tape = new_tape();
            let v = tape.var_no_grad(&Tensor::new(data.clone(), &shape).unwrap());
            let rows = to_channel_rows(&v, n, c, spatial).unwrap();
            assert_eq!(rows.shape(), vec![c, n * spatial]);
            // 行 ch の先頭要素は x[0, ch, 0...]。
            let rv = rows.to_tensor();
            assert_eq!(rv.as_slice().unwrap()[0], 0.0);
            if c > 1 {
                assert_eq!(rv.as_slice().unwrap()[n * spatial], spatial as f32);
            }
            let back = from_channel_rows(&rows, n, c, spatial, &shape).unwrap();
            assert_eq!(back.to_tensor().as_slice().unwrap(), data.as_slice());
        }
    }
}
