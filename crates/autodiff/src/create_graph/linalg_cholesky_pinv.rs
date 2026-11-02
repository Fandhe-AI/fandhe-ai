//! `Cholesky`・`Pinv`・`Lstsq` の `create_graph` 対応（イシュー #3805・親 #3734・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §43・§65）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `build_cgrads`（1 階 VJP の子テープ記録）から
//! 呼ばれる。再生（`replay_op`）は親側が公開メソッド（`cholesky`・`pinv`・`lstsq`）への自己再生で
//! 行うため、forward の検証を迂回しない。構造に積む Op はすべて `supports_create_graph() == true`
//! な Op（`Transpose`・`Contiguous`・`MatMul`・`Where`・`Mul`・`Add`・`Sub`・`Neg`・`Solve`、それに
//! 自己再生の `Cholesky`・`Pinv`）に限り、閉包性（子テープ上の勾配をさらに
//! `backward_create_graph` で微分できること。§21.1(4)）を保つ。`Pinv` を `Lstsq` より先に有効化する
//! のは、`Lstsq` の構造が子テープへ `Pinv` ノードを積むため。
//!
//! **方式（V3 + f32 構造）**: 値は 1 階 VJP と同じホスト関数（`eval::linalg::cholesky_vjp`・
//! `lstsq_vjp`・`pinv_vjp`）の結果で、`Op::Contiguous` ノード（恒等 VJP）として `push_eager` する
//! （[`super::push_v3`]）。ゆえに `cg.grad` は CPU で 1 階と bit 一致する。構造（二階微分を流す
//! f32 の `Var` 演算の連鎖）は値に影響せず、二階と孫テープにだけ効く。
//! - `Cholesky`: `Φ = tril(Lᵀ g) ⊙ W`（`W` は対角 0.5・他 1。上三角は `where_cond` で 0 にする。
//!   0 定数との `Mul` は `inf·0 = NaN` を生むため使わない）、`Y = solve(Lᵀ, Φ)`、
//!   `S = solve(Lᵀ, Yᵀ)ᵀ`、`dA = 0.5 (S + Sᵀ)`（`L` は `Cholesky` ノード自身の写し）。
//! - `Pinv`（打ち切りなし `rank == min(m,n)` の入力だけ）: `P` は `Pinv` ノードの写し、`G = g`。
//!   縦長（`m > n`。`PA ≡ I_n` なので第 3 項は落ちる）は `gA = (M − A (P M)) − Pᵀ (G Pᵀ)`、
//!   `M = Gᵀ (P Pᵀ)`。横長（`m < n`。`AP ≡ I_m` で第 2 項が落ちる）は
//!   `gA = (M − (M P) A) − (Pᵀ G) Pᵀ`、`M = (Pᵀ P) Gᵀ`。正方は `gA = −Pᵀ G Pᵀ`。
//!   **長辺側の `m×m`（`n×n`）を実体化しない結合順**（縦長 `[100000,1]` で 10^10 要素の確保を避ける）。
//! - `Lstsq`: 子テープへ新しく `P = pinv(A)` を積み、`gB = Pᵀ gX`、`gA` は `G = gX Bᵀ` の `Pinv` 構造。
//!
//! **総括的な適用判定**（[`guard_cholesky`]・[`guard_pinv`]・[`guard_lstsq`]。`linalg_fft` の
//! `guard_inv`／`guard_solve` と同型。新しい定数は作らない）: 構造を積む前に f64 参照で各段を求め、
//! 非有限・`2^125` 以上・非零なのに f32 正規化最小値未満・f32 積和の丸め誤差上界が REQ-2 の複合判定を
//! 超えうる（桁落ち）・二階の上界が `2^125` 以上・`g` に非有限要素、のいずれかなら
//! `Err(AutodiffError::Backward)` で拒否する（呼び出し側のロールバックで `child` は空へ戻る）。
//! `Pinv`・`Lstsq` はさらに、`rank != min(m,n)`（打ち切りあり・rank 落ち）と、特異値が `rcond` の閾値の
//! f32 隣接値の範囲内にある入力も拒否する（再生で rank が反転しうるため）。判定は保守的で、拒否されても
//! 実際に破綻するとは限らない。
//!
//! 保証範囲: 判定を通った入力で、seed の各要素の絶対値が 1 以下の二階と孫テープへの再生が有限で
//! 有限差分と一致する。対象外は三階以上と、打ち切りありの `Pinv`・`Lstsq` の二階（§8 のスコープ外課題）。

use fandhe_ai_tensor_core::Tensor;

use super::linalg_fft::{
    abs_vec, all_stage_ok, empty_grad, inverse_of, max_abs, mm, rounding_ok, second_order_ok,
    to_f64, transpose,
};
use super::{push_v3, value_of};
use crate::error::AutodiffError;
use crate::eval::linalg::{
    cholesky_vjp, lstsq_vjp, resolve_rcond, singular_values_and_pinv_f64, truncated_svd_rank,
};
use crate::tape::Tape;
use crate::var::Var;

fn reject(what: &str) -> AutodiffError {
    AutodiffError::Backward(format!(
        "create_graph: {what}（f32 の子テープ構造が f64 の 1 階と食い違いうる入力、または打ち切りあり／\
         閾値付近の Pinv・Lstsq。決定記録 §43・§65。拒否は保守的で、実際に破綻するとは限らない）"
    ))
}

// ---------------------------------------------------------------------
// 総括的な適用判定（f64 参照）
// ---------------------------------------------------------------------

/// 行優先 `[r,c]` の下三角（対角を含む）以外を 0 にし、対角を 0.5 倍する（Cholesky の `Φ` の重み）。
fn tril_half_diag(v: &mut [f64], n: usize) {
    for r in 0..n {
        for c in 0..n {
            if c > r {
                v[r * n + c] = 0.0;
            } else if c == r {
                v[r * n + c] *= 0.5;
            }
        }
    }
}

/// `Cholesky` の判定。`l` は forward の `L`（`[n,n]`）、`g` は上流 cotangent（子テープ上の値）。
pub(super) fn guard_cholesky(l: &Tensor<f32>, g: &Tensor<f32>) -> Result<(), AutodiffError> {
    let n = l.shape()[0];
    if n == 0 {
        return Ok(());
    }
    let gv = to_f64(g);
    if !gv.iter().all(|x| x.is_finite()) {
        return Err(reject("Cholesky: 上流 g に非有限要素がある"));
    }
    let lv = to_f64(l);
    let (x, _) = inverse_of(l, "Cholesky")?; // L^{-1}
    let xt = transpose(&x, n, n);
    let lt = transpose(&lv, n, n);
    let mut phi = mm(&lt, &gv, n, n, n);
    tril_half_diag(&mut phi, n);
    let y = mm(&xt, &phi, n, n, n);
    let st = mm(&xt, &transpose(&y, n, n), n, n, n);
    let s = transpose(&st, n, n);
    let da: Vec<f64> = (0..n * n)
        .map(|i| 0.5 * (s[i] + s[(i % n) * n + i / n]))
        .collect();
    if !(all_stage_ok(&x)
        && all_stage_ok(&phi)
        && all_stage_ok(&y)
        && all_stage_ok(&st)
        && all_stage_ok(&da))
    {
        return Err(reject("Cholesky: 構造の段が overflow／ゼロ化域に入る"));
    }
    let mut mag_phi = mm(&abs_vec(&lt), &abs_vec(&gv), n, n, n);
    tril_half_diag(&mut mag_phi, n);
    let mag_s = mm(&mm(&abs_vec(&xt), &mag_phi, n, n, n), &abs_vec(&x), n, n, n);
    let mag: Vec<f64> = (0..n * n)
        .map(|i| 0.5 * (mag_s[i] + mag_s[(i % n) * n + i / n]))
        .collect();
    if !rounding_ok(&da, &mag, 2 * n) {
        return Err(reject(
            "Cholesky: f32 構造の丸め誤差が REQ-2 の許容を超えうる（桁落ち）",
        ));
    }
    let (bx, bl) = (n as f64 * max_abs(&x), n as f64 * max_abs(&lv));
    if !second_order_ok(&[bx, bl], 4, max_abs(&gv).max(1.0)) {
        return Err(reject("Cholesky: 二階の逆伝播の上界が 2^125 以上"));
    }
    Ok(())
}

/// 隣接する f32 値（`rcond` の閾値付近の判定に使う。新しい tolerance 定数は作らない）。
pub(super) fn prev_f32(r: f32) -> f32 {
    if r <= 0.0 {
        0.0
    } else {
        f32::from_bits(r.to_bits() - 1)
    }
}

pub(super) fn next_f32(r: f32) -> f32 {
    if r.is_finite() && r >= 0.0 {
        f32::from_bits(r.to_bits() + 1)
    } else {
        r
    }
}

/// `Pinv`・`Lstsq` 共通の前段: 打ち切りなし（`rank == min(m,n)`）と閾値付近でないことを確かめ、
/// f64 の擬似逆行列 `P`（行優先 `[n,m]`）を返す。
fn pinv_prelude(a: &Tensor<f32>, rcond: f32) -> Result<Vec<f64>, AutodiffError> {
    let shape = a.shape();
    let (m, n) = (shape[0], shape[1]);
    let k = m.min(n);
    // 1 階 VJP は `Some(payload)` を `resolve_rcond` で f64 へ解く。同じ解決を使う。
    let rc = resolve_rcond(Some(rcond), m, n)?;
    let (s, p) = singular_values_and_pinv_f64(a, rc)?;
    if truncated_svd_rank(&s, rc) != k {
        return Err(reject(
            "Pinv／Lstsq: 打ち切りあり（rank が min(m,n) 未満）の入力は二階の対象外",
        ));
    }
    let smax = s.first().copied().unwrap_or(0.0);
    let (lo, hi) = (
        smax * f64::from(prev_f32(rcond)),
        smax * f64::from(next_f32(rcond)),
    );
    if s.iter().any(|&v| v >= lo && v <= hi) {
        return Err(reject(
            "Pinv／Lstsq: 特異値が rcond の閾値付近にある（再生で rank が反転しうる）",
        ));
    }
    Ok(p)
}

/// `a`（`[m,n]`）・`p`（`[n,m]`）・`g`（`[n,m]`）の f64 で、[`pinv_struct`] と同じ結合順の各段と
/// 丸め上界を調べる。長辺側の `m×m`（`n×n`）を確保しない。
fn pinv_struct_check(
    a: &[f64],
    p: &[f64],
    g: &[f64],
    m: usize,
    n: usize,
) -> Result<(), AutodiffError> {
    let pt = transpose(p, n, m); // [m,n]
    let (apt, ag, agt, aa, ap) = (
        abs_vec(&pt),
        abs_vec(g),
        abs_vec(&transpose(g, n, m)),
        abs_vec(a),
        abs_vec(p),
    );
    let (ga, mag): (Vec<f64>, Vec<f64>);
    let mut stages: Vec<Vec<f64>> = Vec::new();
    if m == n {
        // gA = -Pᵀ (G Pᵀ)
        let gpt = mm(g, &pt, n, m, n);
        let t1 = mm(&pt, &gpt, m, n, n);
        mag = mm(&apt, &mm(&ag, &apt, n, m, n), m, n, n);
        ga = t1.iter().map(|v| -v).collect();
        stages.push(gpt);
        stages.push(t1);
    } else if m > n {
        let gpt = mm(g, &pt, n, m, n); // [n,n]
        let t1 = mm(&pt, &gpt, m, n, n); // [m,n]
        let ppt = mm(p, &pt, n, m, n); // [n,n]
        let mval = mm(&transpose(g, n, m), &ppt, m, n, n); // [m,n]
        let pm = mm(p, &mval, n, m, n); // [n,n]
        let apm = mm(a, &pm, m, n, n); // [m,n]
        let t2: Vec<f64> = mval.iter().zip(&apm).map(|(x, y)| x - y).collect();
        ga = t2.iter().zip(&t1).map(|(x, y)| x - y).collect();
        let mag_m = mm(&agt, &mm(&ap, &apt, n, m, n), m, n, n);
        let mag_t2: Vec<f64> = mag_m
            .iter()
            .zip(mm(&aa, &mm(&ap, &mag_m, n, m, n), m, n, n))
            .map(|(x, y)| x + y)
            .collect();
        let mag_t1 = mm(&apt, &mm(&ag, &apt, n, m, n), m, n, n);
        mag = mag_t2.iter().zip(&mag_t1).map(|(x, y)| x + y).collect();
        stages.extend([gpt, t1, ppt, mval, pm, apm, t2]);
    } else {
        let ptg = mm(&pt, g, m, n, m); // [m,m]
        let t1 = mm(&ptg, &pt, m, m, n); // [m,n]
        let ptp = mm(&pt, p, m, n, m); // [m,m]
        let mval = mm(&ptp, &transpose(g, n, m), m, m, n); // [m,n]
        let mp = mm(&mval, p, m, n, m); // [m,m]
        let mpa = mm(&mp, a, m, m, n); // [m,n]
        let t3: Vec<f64> = mval.iter().zip(&mpa).map(|(x, y)| x - y).collect();
        ga = t3.iter().zip(&t1).map(|(x, y)| x - y).collect();
        let mag_m = mm(&mm(&apt, &ap, m, n, m), &agt, m, m, n);
        let mag_t3: Vec<f64> = mag_m
            .iter()
            .zip(mm(&mm(&mag_m, &ap, m, n, m), &aa, m, m, n))
            .map(|(x, y)| x + y)
            .collect();
        let mag_t1 = mm(&mm(&apt, &ag, m, n, m), &apt, m, m, n);
        mag = mag_t3.iter().zip(&mag_t1).map(|(x, y)| x + y).collect();
        stages.extend([ptg, t1, ptp, mval, mp, mpa, t3]);
    }
    stages.push(ga.clone());
    if !(all_stage_ok(p) && stages.iter().all(|s| all_stage_ok(s))) {
        return Err(reject("Pinv: 構造の段が overflow／ゼロ化域に入る"));
    }
    if !rounding_ok(&ga, &mag, 2 * m.max(n)) {
        return Err(reject(
            "Pinv: f32 構造の丸め誤差が REQ-2 の許容を超えうる（桁落ち）",
        ));
    }
    let dim = m.max(n) as f64;
    if !second_order_ok(
        &[dim * max_abs(p), dim * max_abs(a)],
        4,
        max_abs(g).max(1.0),
    ) {
        return Err(reject("Pinv: 二階の逆伝播の上界が 2^125 以上"));
    }
    Ok(())
}

/// `Pinv` の判定。`a` は入力 `[m,n]`、`rcond` は payload、`g` は上流 `[n,m]`。
pub(super) fn guard_pinv(
    a: &Tensor<f32>,
    rcond: f32,
    g: &Tensor<f32>,
) -> Result<(), AutodiffError> {
    let shape = a.shape();
    let (m, n) = (shape[0], shape[1]);
    if m == 0 || n == 0 {
        return Ok(());
    }
    let gv = to_f64(g);
    if !gv.iter().all(|x| x.is_finite()) {
        return Err(reject("Pinv: 上流 g に非有限要素がある"));
    }
    let p = pinv_prelude(a, rcond)?;
    pinv_struct_check(&to_f64(a), &p, &gv, m, n)
}

/// `Lstsq` の判定。`b` は `[m,k]`、`g_x` は上流 `[n,k]`。
pub(super) fn guard_lstsq(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    rcond: f32,
    g_x: &Tensor<f32>,
) -> Result<(), AutodiffError> {
    let shape = a.shape();
    let (m, n) = (shape[0], shape[1]);
    let k = b.shape()[1];
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let gx = to_f64(g_x);
    if !gx.iter().all(|x| x.is_finite()) {
        return Err(reject("Lstsq: 上流 g に非有限要素がある"));
    }
    let bv = to_f64(b);
    if !bv.iter().all(|x| x.is_finite()) {
        return Err(reject("Lstsq: b に非有限要素がある"));
    }
    let p = pinv_prelude(a, rcond)?;
    let g = mm(&gx, &transpose(&bv, m, k), n, k, m); // G = gX Bᵀ: [n,m]
    if !all_stage_ok(&g) {
        return Err(reject("Lstsq: G = gX Bᵀ が overflow／ゼロ化域に入る"));
    }
    // G の f32 積和（縮約長 k）自体の桁落ちも検査する。f64 で通っても子テープの f32 では
    // 打ち消し合う列（例: B = [1e8, 1, -1e8]）で G が 0 になり、二階の構造が崩れる。
    if !rounding_ok(
        &g,
        &mm(&abs_vec(&gx), &abs_vec(&transpose(&bv, m, k)), n, k, m),
        k,
    ) {
        return Err(reject(
            "Lstsq: G = gX Bᵀ の f32 積和の丸め誤差が REQ-2 の許容を超えうる（桁落ち）",
        ));
    }
    pinv_struct_check(&to_f64(a), &p, &g, m, n)?;
    // G = gX Bᵀ・gB = Pᵀ gX の追加演算の逆伝播。tanh 飽和などで値が gX=G=gB=0 でも、二階の
    // cotangent は seed から「Pinv 構造の逆伝播（P・A の dim 倍の 4 乗）→ G から gX へ（B の
    // m 倍）」「gB から gX へ（P の m 倍）」と流れるため、値に依らない上界で検査する。
    let dim = m.max(n) as f64;
    let boundary = (m as f64 * max_abs(&bv))
        .max(m as f64 * max_abs(&p))
        .max(1.0);
    if !second_order_ok(&[dim * max_abs(&p), dim * max_abs(&to_f64(a))], 4, boundary) {
        return Err(reject(
            "Lstsq: G・gB から B・gX への逆伝播の cotangent 上界が 2^125 以上",
        ));
    }
    let pt = transpose(&p, n, m);
    let gb = mm(&pt, &gx, m, n, k);
    if !all_stage_ok(&gb) {
        return Err(reject("Lstsq: gB = Pᵀ gX が overflow／ゼロ化域に入る"));
    }
    if !rounding_ok(&gb, &mm(&abs_vec(&pt), &abs_vec(&gx), m, n, k), n) {
        return Err(reject(
            "Lstsq: gB の f32 構造の丸め誤差が REQ-2 の許容を超えうる（桁落ち）",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// 1 階 VJP の子テープ記録（build_cgrads から呼ばれる。判定は呼び出し側が先に行う）
// ---------------------------------------------------------------------

/// `Cholesky` の d_input（V3）。`l_m` は `Cholesky` ノード自身の写し、`l_val` は forward の `L`。
pub(super) fn cholesky_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    l_m: &Var<'c>,
    l_val: &Tensor<f32>,
) -> Result<Var<'c>, AutodiffError> {
    let n = l_m.shape()[0];
    if n == 0 {
        return empty_grad(child, &[n, n]);
    }
    let host = cholesky_vjp(l_val, &value_of(child, g)?)?;
    let upper: Vec<bool> = (0..n * n).map(|i| i % n > i / n).collect();
    let upper = Tensor::new(upper, &[n, n])?;
    let w: Vec<f32> = (0..n * n)
        .map(|i| if i % n == i / n { 0.5 } else { 1.0 })
        .collect();
    let w = child.var_no_grad(&Tensor::new(w, &[n, n])?);
    let half = child.var_no_grad(&Tensor::new(vec![0.5f32; n * n], &[n, n])?);
    let lt = l_m.transpose(0, 1)?.contiguous()?;
    let t1 = lt.matmul_fp32_strict(&g.contiguous()?)?;
    let zeros = child.var_no_grad(&Tensor::zeros(&[n, n])?);
    let tri = Var::where_cond(&upper, &zeros, &t1)?;
    let phi = tri.mul(&w)?;
    let y = lt.solve(&phi)?;
    let st = lt.solve(&y.transpose(0, 1)?.contiguous()?)?;
    let s = st.transpose(0, 1)?.contiguous()?;
    let composed = s.add(&s.transpose(0, 1)?.contiguous()?)?.mul(&half)?;
    push_v3(child, &composed, host)
}

/// `Pinv` の構造（`Lstsq` と共有）: `a_m: [m,n]`・`p_m: [n,m]`・`g: [n,m]` から `gA: [m,n]`。
/// 長辺側の `m×m`（`n×n`）を実体化しない結合順（モジュール doc）。
fn pinv_struct<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    a_m: &Var<'c>,
    p_m: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let a_shape = a_m.shape();
    let (m, n) = (a_shape[0], a_shape[1]);
    if m == 0 || n == 0 {
        return empty_grad(child, &a_shape);
    }
    let g = g.contiguous()?;
    let p = p_m.contiguous()?;
    let pt = p.transpose(0, 1)?.contiguous()?; // [m,n]
    if m == n {
        return pt.matmul_fp32_strict(&g.matmul_fp32_strict(&pt)?)?.neg();
    }
    let gt = g.transpose(0, 1)?.contiguous()?; // [m,n]
    if m > n {
        let t1 = pt.matmul_fp32_strict(&g.matmul_fp32_strict(&pt)?)?;
        let ppt = p.matmul_fp32_strict(&pt)?; // [n,n]
        let mval = gt.matmul_fp32_strict(&ppt)?; // [m,n]
        let pm = p.matmul_fp32_strict(&mval)?; // [n,n]
        let t2 = mval.sub(&a_m.matmul_fp32_strict(&pm)?)?;
        t2.sub(&t1)
    } else {
        let t1 = pt.matmul_fp32_strict(&g)?.matmul_fp32_strict(&pt)?;
        let ptp = pt.matmul_fp32_strict(&p)?; // [m,m]
        let mval = ptp.matmul_fp32_strict(&gt)?; // [m,n]
        let mp = mval.matmul_fp32_strict(&p)?; // [m,m]
        let t3 = mval.sub(&mp.matmul_fp32_strict(a_m)?)?;
        t3.sub(&t1)
    }
}

/// `Pinv` の d_input（V3）。`p_m` は `Pinv` ノード自身の写し、`a_val` は入力値。
pub(super) fn pinv_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    a_m: &Var<'c>,
    p_m: &Var<'c>,
    a_val: &Tensor<f32>,
    rcond: f32,
) -> Result<Var<'c>, AutodiffError> {
    let a_shape = a_m.shape();
    if a_shape[0] == 0 || a_shape[1] == 0 {
        return empty_grad(child, &a_shape);
    }
    let host = crate::eval::linalg::pinv_vjp(a_val, Some(rcond), &value_of(child, g)?)?;
    let composed = pinv_struct(child, g, a_m, p_m)?;
    push_v3(child, &composed, host)
}

/// `Lstsq` の `(d_a, d_b)`（V3）。子テープへ `P = pinv(A)` を新しく積む。
pub(super) fn lstsq_cgrad<'c>(
    child: &'c Tape,
    g_x: &Var<'c>,
    a_m: &Var<'c>,
    b_m: &Var<'c>,
    vals: (&Tensor<f32>, &Tensor<f32>),
    rcond: f32,
) -> Result<(Var<'c>, Var<'c>), AutodiffError> {
    let a_shape = a_m.shape();
    let b_shape = b_m.shape();
    if a_shape[0] == 0 || a_shape[1] == 0 {
        return Ok((empty_grad(child, &a_shape)?, empty_grad(child, &b_shape)?));
    }
    let (host_a, host_b) = lstsq_vjp(vals.0, vals.1, Some(rcond), &value_of(child, g_x)?)?;
    let p = a_m.pinv(Some(rcond))?;
    let gx = g_x.contiguous()?;
    let gb = p.transpose(0, 1)?.contiguous()?.matmul_fp32_strict(&gx)?;
    let g_pinv = gx.matmul_fp32_strict(&b_m.transpose(0, 1)?.contiguous()?)?;
    let ga = pinv_struct(child, &g_pinv, a_m, &p)?;
    Ok((push_v3(child, &ga, host_a)?, push_v3(child, &gb, host_b)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_f32_values_bracket_the_input() {
        let r = 0.75f32;
        assert!(prev_f32(r) < r && r < next_f32(r));
        assert_eq!(prev_f32(0.0), 0.0);
        assert!(next_f32(0.0) > 0.0);
    }

    #[test]
    fn tril_half_diag_masks_upper_and_halves_diagonal() {
        let mut v = vec![1.0f64; 4];
        tril_half_diag(&mut v, 2);
        assert_eq!(v, vec![0.5, 0.0, 1.0, 0.5]);
    }

    /// 縦長・横長の `Pinv` 構造は長辺側の `m×m`（`n×n`）の中間行列を子テープに作らない
    /// （`pinv_vjp_tall_matrix_does_not_allocate_full_m_by_m_intermediate` の構造版。§43.4）。
    #[test]
    fn pinv_cgrad_has_no_long_side_square_node() {
        for (m, n) in [(3000usize, 1usize), (1, 3000)] {
            let data: Vec<f32> = (0..m * n).map(|i| 1.0 + (i % 5) as f32 * 0.1).collect();
            let tape = Tape::new();
            let child = Tape::new();
            let x = tape.var(&Tensor::new(data, &[m, n]).unwrap());
            let loss = x.pinv(None).unwrap().sum(None).unwrap();
            tape.backward_create_graph(&loss, &child)
                .expect("打ち切りなしの縦長／横長は受理される");
            let long_sq = m.max(n) * m.max(n);
            let nodes = child.nodes.borrow();
            assert!(
                nodes
                    .iter()
                    .all(|node| node.shape.iter().product::<usize>() < long_sq),
                "({m},{n}): 長辺側の正方行列（{long_sq} 要素以上）のノードがある"
            );
        }
    }
}
