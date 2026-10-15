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
//!   0.5。0 を含まない）を掛けて `irfft(.., n, dim, swap(norm))`、その後 `L` へ。DC・Nyquist の虚部は
//!   `irfft` が読まないため、1 階 VJP と同じ非有限値の伝播になるよう レーンごとに 0 倍して当該レーンの有効入力位置へ加える（`Irfft` の DC・Nyquist 虚部は 1 階が +0.0 固定のため定数ゼロへ置換）。
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
//! 「二階の逆伝播の上界が `2^125` 以上」「`g` に非有限要素」「`Inv`・`Solve` の f32 積和の丸め誤差上界が
//! REQ-2 の複合判定を超えうる（桁落ち。`rounding_ok`）」のいずれかなら
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
            let r = resize_axis(child, &r, dim, n, l)?;
            // `irfft` は DC・Nyquist の虚部を読まないが、1 階 VJP（`rfft_vjp_host`）は
            // レーンごとに `gi * sin` を足すため、そのレーンの非有限値（inf／NaN）が当該レーンの
            // 有効入力位置（`j < min(n, L)`）だけへ伝播する。同じ扱いに揃えるため、各レーンの
            // その虚部を要素ごとに 0 倍（有限なら厳密に 0、非有限なら NaN）してから `dim` 軸へ
            // 広げて有効位置へ加える。全レーンのスカラー縮約はしない（無関係なレーンへの汚染と
            // 有限値の和の overflow を避ける）。
            let mut bins = vec![0usize];
            if n.is_multiple_of(2) && n / 2 > 0 {
                bins.push(n / 2);
            }
            let im_axis = g.shape().len() - 1;
            let mut poison: Option<Var<'c>> = None;
            for k in bins {
                let im = g.narrow(dim, k, 1)?.narrow(im_axis, 1, 1)?;
                let zero = child.var_no_grad(&Tensor::zeros(&im.shape())?);
                let z = im.mul(&zero)?;
                poison = Some(match poison {
                    Some(p) => p.add(&z)?,
                    None => z,
                });
            }
            match poison {
                Some(p) => {
                    let mut lane_shape = p.shape();
                    lane_shape.pop();
                    let p = p.reshape(&lane_shape)?;
                    let valid = l.min(n);
                    let mut full = lane_shape.clone();
                    full[dim] = valid;
                    let p = p.broadcast_to(&full)?;
                    let p = resize_axis(child, &p, dim, valid, l)?;
                    r.add(&p)
                }
                None => Ok(r),
            }
        }
        FftKind::Irfft => {
            let c = hermitian_weights(child, input_shape.len(), dim, n, 2.0)?;
            let r = g.rfft(Some(n), Some(dim), sw)?;
            // 1 階 VJP（`irfft_vjp_host`）は DC・〈`n` 偶数の〉Nyquist の虚部勾配をリテラル +0.0 に
            // 固定する。`rfft` の当該虚部は `g * sin(0)` を含み上流 `g` が非有限だと NaN になるため、
            // 定数ゼロへ置換（`narrow` + `cat`）して揃える。
            let r = zero_real_bin_imag(child, &r, dim, n)?.mul(&c)?;
            resize_axis(child, &r, dim, n / 2 + 1, l)
        }
    }
}

/// `r: [.., n/2+1 (dim), .., 2]` の実数 bin（DC と〈`n` 偶数の〉Nyquist）の虚部を定数ゼロへ置換する。
fn zero_real_bin_imag<'c>(
    child: &'c Tape,
    r: &Var<'c>,
    dim: usize,
    n: usize,
) -> Result<Var<'c>, AutodiffError> {
    let im_axis = r.shape().len() - 1;
    let m = n / 2 + 1;
    let re = r.narrow(im_axis, 0, 1)?;
    let im = r.narrow(im_axis, 1, 1)?;
    let zero_bin = || -> Result<Var<'c>, AutodiffError> {
        let mut shape = im.shape();
        shape[dim] = 1;
        Ok(child.var_no_grad(&Tensor::zeros(&shape)?))
    };
    let nyq = n.is_multiple_of(2) && n / 2 > 0;
    let mut parts = vec![zero_bin()?];
    let mid_end = if nyq { m - 1 } else { m };
    if mid_end > 1 {
        parts.push(im.narrow(dim, 1, mid_end - 1)?);
    }
    if nyq {
        parts.push(zero_bin()?);
    }
    let im_fixed = Var::cat(&parts, dim)?;
    Var::cat(&[re, im_fixed], im_axis)
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

/// REQ-2 統一複合判定の許容誤差定数（正本は `tensor-core::tolerance`。値は変えない）。
use fandhe_ai_tensor_core::tolerance::{
    ABSOLUTE_RESCUE_THRESHOLD as REQ2_ABSOLUTE_RESCUE_THRESHOLD,
    RELATIVE_TOLERANCE as REQ2_RELATIVE_TOLERANCE,
};
/// 保守的な相対判定の閾値。許容の半分を使い、f32 構造の見積り上界に余裕を持たせる（許容自体は不変）。
const CONSERVATIVE_RELATIVE_TOLERANCE: f64 = REQ2_RELATIVE_TOLERANCE / 2.0;

/// f32 の子テープ構造の丸め誤差が REQ-2 の複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）に
/// 収まるかの保守的検査。`reference` は f64 の真値、`magnitude` は同じ式の絶対値積（`|X||G||X|` 等）、
/// `terms` は構造内の積和の総項数（段の長さ）。誤差上界を `(2 * terms + 4) * 2^-23 * magnitude`
/// （入力 `X` 自体の f32 丸め 2 回分を含む）と見積もり、要素ごとに許容内かを調べる。
/// 桁落ち（`magnitude` が `|reference|` より桁違いに大きい入力）を f32 構造のまま通さない。
fn rounding_ok(reference: &[f64], magnitude: &[f64], terms: usize) -> bool {
    let coeff = (2 * terms + 4) as f64 * (-23f64).exp2();
    reference.iter().zip(magnitude).all(|(&r, &m)| {
        let bound = coeff * m;
        bound < REQ2_ABSOLUTE_RESCUE_THRESHOLD || bound <= CONSERVATIVE_RELATIVE_TOLERANCE * r.abs()
    })
}

fn abs_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| x.abs()).collect()
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
    let (ax, ag) = (abs_vec(&xt), abs_vec(&gv));
    let mag = mm(&mm(&ax, &ag, n, n, n), &ax, n, n, n);
    if !rounding_ok(&s2, &mag, n) {
        return Err(reject(
            "Inv: f32 構造の丸め誤差が REQ-2 の許容を超えうる（桁落ち）",
        ));
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
    let mag_da = mm(&abs_vec(&db), &abs_vec(&transpose(&xs, n, k)), n, k, n);
    if !rounding_ok(&da, &mag_da, k) {
        return Err(reject(
            "Solve: f32 構造の丸め誤差が REQ-2 の許容を超えうる（桁落ち）",
        ));
    }
    let dim = n.max(k) as f64;
    let (bxi, bxs) = (dim * max_abs(&xinv), dim * max_abs(&xs));
    if !second_order_ok(&[bxi, bxs], 4, max_abs(&gv).max(1.0)) {
        return Err(reject("Solve: 二階の逆伝播の上界が 2^125 以上"));
    }
    Ok(())
}

/// `Det` の二階微分 `gv * det * (tr(X S) * X^T - X^T S^T X^T)`（`X = inv(A)`、`S` は seed、`gv` は
/// 上流勾配）は、det(A) と inv(A) の経路が大きな同符号反対項を打ち消し合う（`det` が小さい悪条件
/// 入力で桁落ち。例: f32 の `[[1,2],[3,6.001]]` は厳密に 0 の二階が約 4e-3 残る）。2 項は別経路の
/// f32 丸めを持ち相殺しないため、基底 seed `E_kl` ごとの絶対値積（2 項の大きさ）の総和（`|S| <= 1`
/// の任意 seed の上界）に `|gv|`（相殺誤差は上流勾配に比例する）と丸め誤差係数 `(2n+4) * 2^-23` を
/// 掛けて見積もり、全要素で絶対 1e-5 未満に収まらなければ拒否する（tolerance は緩めない。保守的で、
/// 実際には許容内の入力も拒否しうる）。
/// 基底 seed の総和は因数分解でき、`mag[a][b] = |det| * |gv| * (T * |X^T|[a][b] + R[a] * C[b])`
/// （`T` は `|X|` の全要素和、`R`・`C` は `|X^T|` の行和・列和）となる。O(n^2)。
fn det_second_order_rounding_ok(x: &[f64], det: f64, gv: f64, n: usize) -> bool {
    let coeff = (2 * n + 4) as f64 * (-23f64).exp2();
    let ax = abs_vec(&transpose(x, n, n));
    let total: f64 = ax.iter().sum();
    let rows: Vec<f64> = ax.chunks(n).map(|r| r.iter().sum()).collect();
    let mut cols = vec![0.0f64; n];
    for r in ax.chunks(n) {
        for (c, v) in cols.iter_mut().zip(r) {
            *c += v;
        }
    }
    let scale = det.abs() * gv.abs();
    ax.iter().enumerate().all(|(i, &v)| {
        let m = scale * (total * v + rows[i / n] * cols[i % n]);
        coeff * m < REQ2_ABSOLUTE_RESCUE_THRESHOLD
    })
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
    if with_det && !det_second_order_rounding_ok(&x, det, gv, n) {
        return Err(reject(
            "Det: 二階微分の det と inv の経路間の桁落ちが REQ-2 の許容を超えうる",
        ));
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
