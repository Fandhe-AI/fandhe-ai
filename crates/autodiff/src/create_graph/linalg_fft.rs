//! 線形代数 Op（`Inv`・`Solve`・`Det`・`SlogdetLogAbsDet`）と FFT 4 件（`Rfft`・`Irfft`・
//! `Fft`・`Ifft`）の `create_graph` 対応（イシュー #3086・親 #3076・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §21.2・§30）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `replay_op`（forward 再生）・`build_cgrads`
//! （1 階 VJP の子テープ記録）・`validate_ancestors`（入口の事前拒否）から呼ばれる。
//! 子テープへ積む Op はすべて `supports_create_graph() == true` な Op（`Fft`・`Ifft`・`Rfft`・
//! `Irfft`・`Inv`・`Solve`・`Det`・`Transpose`・`MatMul`・`Mul`・`Neg`〈`ScalarUnary`〉・
//! `BroadcastTo`・`Narrow`・`Concat`・`Contiguous`）に限り、閉包性（子テープ上の勾配を
//! さらに `backward_create_graph` で微分できること。§21.1(4)）を保つ。
//!
//! **FFT の方式（V1。値は 1 階 VJP と REQ-2 で一致するが bit 同一は主張しない）**: 4 件とも
//! 入力値に依存しない線形写像で、随伴は別の FFT 1 回とゼロ詰め／切り詰めで書ける。
//! `norm` は随伴で `Backward <-> Forward` を入れ替える（`Ortho` は不変）。
//! - `Fft`: `ifft(g, n, dim, swap(norm))` の後に実軸を入力長 `L` へ合わせる。
//! - `Ifft`: `fft(g, n, dim, swap(norm))` の後に `L` へ合わせる。
//! - `Rfft`: Hermitian の倍加を打ち消す重み `h`（DC と〈`n` 偶数の〉Nyquist は 1、それ以外は
//!   0.5。0 を含まないので `inf * 0` は生じない）を掛けて `irfft(.., n, dim, swap(norm))`、その後 `L` へ。
//! - `Irfft`: `rfft(g, n, dim, swap(norm))` に重み `c`（DC・Nyquist は 1、それ以外は 2）を掛け、
//!   bin 軸を入力の bin 数 `m` へ合わせる。
//!
//! 掛ける定数は 2 のべき乗で丸めに影響しない。overflow の上界判定は付けない（真の値が f32 を
//! 超える場合だけ 1 階と同じく `inf` になる。FFT の結合は有限和で、段をまたぐ増幅がない）。
//! `FftNorm` は `#[non_exhaustive]` のため未知 variant は [`check_fft_norm`] が入口で拒否する。
//!
//! **線形代数の方式（V1 + 総括的な適用判定）**: 1 階 VJP（`eval::linalg`）は f64 の LU で計算し
//! f32 へ丸めるのは最後の 1 回だけだが、子テープの構造は f32 の `Var` 演算の連鎖である。
//! そのため構造を積む前に f64 参照で各段の値と上界を求め（[`LinalgGuard`]）、1 段でも
//! 「有限でない」「`2^125` 以上」「非零なのに f32 正規化最小値未満（ゼロ化・subnormal 化しうる）」
//! 「二階の逆伝播の上界が `2^125` 以上」「`g` に非有限要素」のいずれかなら
//! `Err(AutodiffError::Backward)` で拒否する（呼び出し側のロールバックで `child` は空へ戻る）。
//! 判定は保守的で、拒否されても実際に overflow するとは限らない。`Det` は §21.2 の行別スケール＋
//! 2 のべき乗チャンク構造（V3）を実装せず、同じ判定を付けた V1（`g * det(A) * inv(A)^T`）とし、
//! V3 が必要になる入力（`det` が `2^125` 以上など）は拒否する。
//! - `Inv`: `dA = -(X^T g X^T)`（`X` は `Inv` ノードの写し）。
//! - `Solve`: `dB = solve(A^T, g)`、`dA = -dB X^T`（`X` は `Solve` ノードの写し）。
//! - `SlogdetLogAbsDet`: `dA = broadcast(g) * inv(A)^T`。
//! - `Det`: `dA = broadcast(g * det) * inv(A)^T`。
//!
//! 保証範囲: 判定を通った入力では、`cg.grad` が 1 階と REQ-2 で一致し、seed の絶対値が 1 以下の
//! 二階・孫テープへの再生が有限になる。対象外は上流 Op・|seed|>1・三階以上・真の値が f32 を超える場合。

use fandhe_ai_tensor_core::{FftNorm, Tensor};

use crate::create_graph_norm::STAGE_BOUND_GUARD;
use crate::error::AutodiffError;
use crate::eval::linalg::inv_and_det_f64;
use crate::tape::Tape;
use crate::var::Var;

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!("create_graph: {msg}"))
}

/// 随伴で使う `norm`: `Backward <-> Forward`、`Ortho` は不変。未知 variant は `Err`。
pub(super) fn swap_norm(norm: FftNorm) -> Result<FftNorm, AutodiffError> {
    match norm {
        FftNorm::Backward => Ok(FftNorm::Forward),
        FftNorm::Forward => Ok(FftNorm::Backward),
        FftNorm::Ortho => Ok(FftNorm::Ortho),
        other => Err(invalid(format!("未知の FftNorm variant {other:?}"))),
    }
}

/// `validate_ancestors` から呼ぶ入口検査。子テープへ書き込む前に未知 `norm` を拒否する。
pub(super) fn check_fft_norm(norm: FftNorm) -> Result<(), AutodiffError> {
    swap_norm(norm).map(|_| ())
}

/// FFT 4 件の種別（[`fft_vjp`] の分岐用）。
#[derive(Clone, Copy)]
pub(super) enum FftKind {
    Rfft,
    Irfft,
    Fft,
    Ifft,
}

/// `axis` の長さを `from` から `to` へ合わせる（縮めるなら `narrow`、伸ばすならゼロ定数との
/// `concat`。`Pad` は再生できないため使わない）。
fn resize_axis<'c>(
    child: &'c Tape,
    v: &Var<'c>,
    axis: usize,
    from: usize,
    to: usize,
) -> Result<Var<'c>, AutodiffError> {
    if to == from {
        return Ok(*v);
    }
    if to < from {
        return v.narrow(axis, 0, to);
    }
    let mut shape = v.shape();
    shape[axis] = to - from;
    let zeros = child.var_no_grad(&Tensor::zeros(&shape)?);
    Var::cat(&[*v, zeros], axis)
}

/// bin 軸（`dim`）に沿う Hermitian 重みの定数葉（形状は `dim` だけ `m`、他は 1）。
/// DC と〈`n` 偶数の〉Nyquist は 1、それ以外は `other`。
fn hermitian_weights<'c>(
    child: &'c Tape,
    rank: usize,
    dim: usize,
    n: usize,
    other: f32,
) -> Result<Var<'c>, AutodiffError> {
    let m = n / 2 + 1;
    let data: Vec<f32> = (0..m)
        .map(|k| {
            if k == 0 || (n.is_multiple_of(2) && k == n / 2) {
                1.0
            } else {
                other
            }
        })
        .collect();
    let mut shape = vec![1usize; rank];
    shape[dim] = m;
    Ok(child.var_no_grad(&Tensor::new(data, &shape)?))
}

/// FFT 4 件の VJP を子テープへ記録する（方式は本モジュール doc）。`input_shape` は入力ノードの
/// shape、`n`／`dim`／`norm` は `Op` の解決済み payload。
pub(super) fn fft_vjp<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    kind: FftKind,
    input_shape: &[usize],
    n: usize,
    dim: usize,
    norm: FftNorm,
) -> Result<Var<'c>, AutodiffError> {
    let sw = swap_norm(norm)?;
    let l = *input_shape.get(dim).ok_or_else(|| {
        invalid(format!(
            "FFT の dim={dim} が入力 shape {input_shape:?} の範囲外"
        ))
    })?;
    match kind {
        FftKind::Fft => {
            let r = g.ifft(Some(n), Some(dim), sw)?;
            resize_axis(child, &r, dim, n, l)
        }
        FftKind::Ifft => {
            let r = g.fft(Some(n), Some(dim), sw)?;
            resize_axis(child, &r, dim, n, l)
        }
        FftKind::Rfft => {
            let h = hermitian_weights(child, g.shape().len(), dim, n, 0.5)?;
            let r = g.mul(&h)?.irfft(Some(n), Some(dim), sw)?;
            resize_axis(child, &r, dim, n, l)
        }
        FftKind::Irfft => {
            let c = hermitian_weights(child, input_shape.len(), dim, n, 2.0)?;
            let r = g.rfft(Some(n), Some(dim), sw)?.mul(&c)?;
            resize_axis(child, &r, dim, n / 2 + 1, l)
        }
    }
}

// ---------------------------------------------------------------------
// 線形代数: 総括的な適用判定（f64 参照）
// ---------------------------------------------------------------------

/// 構造の 1 段の値が安全か: 有限・`2^125` 未満・非零なら f32 正規化最小値以上。
fn stage_ok(v: f64) -> bool {
    v.is_finite()
        && v.abs() < STAGE_BOUND_GUARD
        && (v == 0.0 || v.abs() >= f64::from(f32::MIN_POSITIVE))
}

fn reject(what: &str) -> AutodiffError {
    invalid(format!(
        "{what}（f32 の子テープ構造が f64 の 1 階と食い違いうる入力。決定記録 §30。\
         拒否は保守的で、実際に overflow するとは限らない）"
    ))
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// 行優先 `a: [r,k]` と `b: [k,c]` の f64 積。
fn mm(a: &[f64], b: &[f64], r: usize, k: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r * c];
    for i in 0..r {
        for p in 0..k {
            let x = a[i * k + p];
            if x == 0.0 {
                continue;
            }
            for j in 0..c {
                out[i * c + j] += x * b[p * c + j];
            }
        }
    }
    out
}

fn transpose(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = a[i * c + j];
        }
    }
    out
}

fn to_f64(t: &Tensor<f32>) -> Vec<f64> {
    t.host_slice().iter().map(|&v| f64::from(v)).collect()
}

fn all_stage_ok(v: &[f64]) -> bool {
    v.iter().all(|&x| stage_ok(x))
}

/// `A` の f64 逆行列（特異は拒否）。
fn inverse_of(a: &Tensor<f32>, what: &str) -> Result<(Vec<f64>, f64), AutodiffError> {
    inv_and_det_f64(a).ok_or_else(|| reject(&format!("{what}: 行列が特異")))
}

/// 二階の逆伝播の上界（seed の各要素の絶対値 1 以下を仮定した保守的な見積もり）が閾値未満か。
fn second_order_ok(factors: &[f64], power: i32, extra: f64) -> bool {
    let m = factors.iter().fold(1.0f64, |m, &x| m.max(x));
    let bound = m.powi(power) * extra;
    bound.is_finite() && bound < STAGE_BOUND_GUARD
}

/// `Inv` の判定。`g` は上流の cotangent（子テープ上の値）。
pub(super) fn guard_inv(a: &Tensor<f32>, g: &Tensor<f32>) -> Result<(), AutodiffError> {
    let n = a.shape()[0];
    if n == 0 {
        return Ok(());
    }
    let gv = to_f64(g);
    if !gv.iter().all(|x| x.is_finite()) {
        return Err(reject("Inv: 上流 g に非有限要素がある"));
    }
    let (x, _) = inverse_of(a, "Inv")?;
    let xt = transpose(&x, n, n);
    let s1 = mm(&xt, &gv, n, n, n);
    let s2 = mm(&s1, &xt, n, n, n);
    if !(all_stage_ok(&x) && all_stage_ok(&s1) && all_stage_ok(&s2)) {
        return Err(reject("Inv: 構造の段が overflow／ゼロ化域に入る"));
    }
    let bx = n as f64 * max_abs(&x);
    if !second_order_ok(&[bx], 4, max_abs(&gv).max(1.0)) {
        return Err(reject("Inv: 二階の逆伝播の上界が 2^125 以上"));
    }
    Ok(())
}

/// `Solve` の判定。
pub(super) fn guard_solve(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    g: &Tensor<f32>,
) -> Result<(), AutodiffError> {
    let n = a.shape()[0];
    let k = b.shape()[1];
    if n == 0 || k == 0 {
        return Ok(());
    }
    let gv = to_f64(g);
    if !gv.iter().all(|x| x.is_finite()) {
        return Err(reject("Solve: 上流 g に非有限要素がある"));
    }
    let bv = to_f64(b);
    if !bv.iter().all(|x| x.is_finite()) {
        return Err(reject("Solve: b に非有限要素がある"));
    }
    let (xinv, _) = inverse_of(a, "Solve")?;
    let xs = mm(&xinv, &bv, n, n, k);
    let xinv_t = transpose(&xinv, n, n);
    let db = mm(&xinv_t, &gv, n, n, k);
    let da = mm(&db, &transpose(&xs, n, k), n, k, n);
    if !(all_stage_ok(&xinv) && all_stage_ok(&xs) && all_stage_ok(&db) && all_stage_ok(&da)) {
        return Err(reject("Solve: 構造の段が overflow／ゼロ化域に入る"));
    }
    let dim = n.max(k) as f64;
    let (bxi, bxs) = (dim * max_abs(&xinv), dim * max_abs(&xs));
    if !second_order_ok(&[bxi, bxs], 4, max_abs(&gv).max(1.0)) {
        return Err(reject("Solve: 二階の逆伝播の上界が 2^125 以上"));
    }
    Ok(())
}

/// `SlogdetLogAbsDet`（`scale = 1`）と `Det`（`scale = det`）の判定。`g` は形状 `[]`。
pub(super) fn guard_det_like(
    a: &Tensor<f32>,
    g: &Tensor<f32>,
    with_det: bool,
) -> Result<(), AutodiffError> {
    let n = a.shape()[0];
    if n == 0 {
        return Ok(());
    }
    let gv = g.host_slice().first().map_or(0.0, |&v| f64::from(v));
    if !gv.is_finite() {
        return Err(reject("上流 g が非有限"));
    }
    let (x, det) = inverse_of(a, "行列式系")?;
    let scale = if with_det { gv * det } else { gv };
    let rows: Vec<f64> = x.iter().map(|&v| scale * v).collect();
    if !(all_stage_ok(&x)
        && all_stage_ok(&rows)
        && (!with_det || (stage_ok(det) && stage_ok(scale))))
    {
        return Err(reject("行列式系: 構造の段が overflow／ゼロ化域に入る"));
    }
    let bx = n as f64 * max_abs(&x);
    let extra = gv.abs().max(1.0) * if with_det { det.abs().max(1.0) } else { 1.0 };
    if !second_order_ok(&[bx], 3, extra) {
        return Err(reject("行列式系: 二階の逆伝播の上界が 2^125 以上"));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// 線形代数: 1 階 VJP の子テープ記録（build_cgrads から呼ばれる）
// ---------------------------------------------------------------------

fn empty_grad<'c>(child: &'c Tape, shape: &[usize]) -> Result<Var<'c>, AutodiffError> {
    Ok(child.var_no_grad(&Tensor::zeros(shape)?))
}

/// `Inv` の d_input（`x` は `Inv` ノード自身の写し）。判定は呼び出し側が先に行う。
pub(super) fn inv_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    x: &Var<'c>,
    a_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    if a_shape[0] == 0 {
        return empty_grad(child, a_shape);
    }
    let xt = x.transpose(0, 1)?;
    xt.matmul_fp32_strict(g)?.matmul_fp32_strict(&xt)?.neg()
}

/// `Solve` の `(d_a, d_b)`（`x` は `Solve` ノード自身の写し）。
pub(super) fn solve_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    a_m: &Var<'c>,
    x: &Var<'c>,
) -> Result<(Var<'c>, Var<'c>), AutodiffError> {
    let a_shape = a_m.shape();
    let x_shape = x.shape();
    if a_shape[0] == 0 || x_shape[1] == 0 {
        return Ok((empty_grad(child, &a_shape)?, empty_grad(child, &x_shape)?));
    }
    let at = a_m.transpose(0, 1)?.contiguous()?;
    let db = at.solve(&g.contiguous()?)?;
    let da = db.matmul_fp32_strict(&x.transpose(0, 1)?)?.neg()?;
    Ok((da, db))
}

/// `SlogdetLogAbsDet`／`Det` の d_input: `broadcast(scale) * inv(A)^T`（`scale` は形状 `[]`）。
pub(super) fn det_like_cgrad<'c>(
    child: &'c Tape,
    scale: &Var<'c>,
    a_m: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let a_shape = a_m.shape();
    if a_shape[0] == 0 {
        return empty_grad(child, &a_shape);
    }
    let inv_t = a_m.inv()?.transpose(0, 1)?;
    scale.broadcast_to(&a_shape)?.mul(&inv_t)
}
