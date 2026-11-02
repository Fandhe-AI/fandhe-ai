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
//! 判定は保守的で、拒否されても実際に overflow するとは限らない。
//! `Det` はイシュー #3810（決定記録 §46・§70）で §21.2 の行別スケール＋2 のべき乗チャンク構造を実装した
//! （値は全入力で V3〈`det_vjp` を `push_v3`。`cg.grad` が 1 階と bit 一致〉）。[`det_row_scale_plan`] が
//! 非スケール構造（V1: `g * det(A) * inv(A)^T`）の検査を従来と同じ順で行い、区分 (ii)（途中の段が `2^125`
//! 以上など）だけで落ちた入力に限りスケール域で判定をやり直して行別スケール構造（[`det_row_scaled_cgrad`]）へ
//! 回す。(i)（真の値が f32 を超える）・(iii)（桁落ち）・(iv) の拒否は残す。`Inv`・`Solve`・
//! `SlogdetLogAbsDet` は対象外として恒久化した（§46.14・#3767 の承認）。
//! 孫テープ再生: 子テープの `det(B)` への cotangent は `2^c` 倍になるが、A 側へ戻る際に `2^-c` 倍へ
//! 戻る。`det(B)` ノードへ下流の縮小率を登録し (iii) の `mult` へ渡すため、再生は有限に受理される（§70.3）。
//! - `Inv`: `dA = -(X^T g X^T)`（`X` は `Inv` ノードの写し）。
//! - `Solve`: `dB = solve(A^T, g)`、`dA = -dB X^T`（`X` は `Solve` ノードの写し）。
//! - `SlogdetLogAbsDet`: `dA = broadcast(g) * inv(A)^T`。
//! - `Det`: `dA = broadcast(g * det) * inv(A)^T`。
//!
//! `Cholesky`・`Pinv`・`Lstsq` は同型の判定を `linalg_cholesky_pinv` サブモジュール（イシュー #3805）が
//! 持ち、本モジュールの判定用ヘルパー（`stage_ok`・`rounding_ok`・`mm` など）を共有する。
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
pub(super) fn stage_ok(v: f64) -> bool {
    v.is_finite()
        && v.abs() < STAGE_BOUND_GUARD
        && (v == 0.0 || v.abs() >= f64::from(f32::MIN_POSITIVE))
}

pub(super) fn reject(what: &str) -> AutodiffError {
    invalid(format!(
        "{what}（f32 の子テープ構造が f64 の 1 階と食い違いうる入力。決定記録 §30。\
         拒否は保守的で、実際に overflow するとは限らない）"
    ))
}

pub(super) fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// 行優先 `a: [r,k]` と `b: [k,c]` の f64 積。
pub(super) fn mm(a: &[f64], b: &[f64], r: usize, k: usize, c: usize) -> Vec<f64> {
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

pub(super) fn transpose(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = a[i * c + j];
        }
    }
    out
}

pub(super) fn to_f64(t: &Tensor<f32>) -> Vec<f64> {
    t.host_slice().iter().map(|&v| f64::from(v)).collect()
}

pub(super) fn all_stage_ok(v: &[f64]) -> bool {
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
pub(super) fn rounding_ok(reference: &[f64], magnitude: &[f64], terms: usize) -> bool {
    let coeff = (2 * terms + 4) as f64 * (-23f64).exp2();
    reference.iter().zip(magnitude).all(|(&r, &m)| {
        let bound = coeff * m;
        bound < REQ2_ABSOLUTE_RESCUE_THRESHOLD || bound <= CONSERVATIVE_RELATIVE_TOLERANCE * r.abs()
    })
}

pub(super) fn abs_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| x.abs()).collect()
}

/// `A` の f64 逆行列（特異は拒否）。
pub(super) fn inverse_of(a: &Tensor<f32>, what: &str) -> Result<(Vec<f64>, f64), AutodiffError> {
    inv_and_det_f64(a).ok_or_else(|| reject(&format!("{what}: 行列が特異")))
}

/// 二階の逆伝播の上界（seed の各要素の絶対値 1 以下を仮定した保守的な見積もり）が閾値未満か。
pub(super) fn second_order_ok(factors: &[f64], power: i32, extra: f64) -> bool {
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
fn det_second_order_rounding_ok(x: &[f64], det: f64, gv: f64, n: usize, mult: f64) -> bool {
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
    // `mult` は行別スケール経路（B 域の量を最終域へ戻す復元係数。§70）。非スケール経路は 1.0。
    let scale = det.abs() * gv.abs() * mult;
    ax.iter().enumerate().all(|(i, &v)| {
        let m = scale * (total * v + rows[i / n] * cols[i % n]);
        coeff * m < REQ2_ABSOLUTE_RESCUE_THRESHOLD
    })
}

/// 非スケール構造（V1。`g * det * inv(A)^T`）の段の判定（区分 (ii)）。`SlogdetLogAbsDet`
/// （`with_det = false`）と `Det`（`with_det = true`）で共有する。
fn det_like_stage_pass(x: &[f64], det: f64, gv: f64, with_det: bool) -> bool {
    let scale = if with_det { gv * det } else { gv };
    let rows: Vec<f64> = x.iter().map(|&v| scale * v).collect();
    all_stage_ok(x) && all_stage_ok(&rows) && (!with_det || (stage_ok(det) && stage_ok(scale)))
}

/// 区分 (iii)（桁落ち）と (iv)（二階の逆伝播の上界）。段の判定を通った非スケール構造に対する検査。
fn det_like_tail(
    x: &[f64],
    det: f64,
    gv: f64,
    n: usize,
    with_det: bool,
    mult: f64,
) -> Result<(), AutodiffError> {
    if with_det && !det_second_order_rounding_ok(x, det, gv, n, mult) {
        return Err(reject(
            "Det: 二階微分の det と inv の経路間の桁落ちが REQ-2 の許容を超えうる",
        ));
    }
    let bx = n as f64 * max_abs(x);
    let extra = gv.abs().max(1.0) * if with_det { det.abs().max(1.0) } else { 1.0 };
    if !second_order_ok(&[bx], 3, extra) {
        return Err(reject("行列式系: 二階の逆伝播の上界が 2^125 以上"));
    }
    Ok(())
}

/// `SlogdetLogAbsDet`（`scale = 1`）と `Det`（`scale = det`）の判定。`g` は形状 `[]`。
/// `SlogdetLogAbsDet` は対象外として恒久化した（決定記録 §46.14・§70）ため、本関数が唯一の判定。
/// `Det` は [`det_row_scale_plan`] が同じ検査を先に行い、区分 (ii) だけで落ちた入力を行別スケール経路へ回す。
pub(super) fn guard_det_like(
    a: &Tensor<f32>,
    g: &Tensor<f32>,
    with_det: bool,
    mult: f64,
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
    if !det_like_stage_pass(&x, det, gv, with_det) {
        return Err(reject("行列式系: 構造の段が overflow／ゼロ化域に入る"));
    }
    det_like_tail(&x, det, gv, n, with_det, mult)
}

// ---------------------------------------------------------------------
// 線形代数: 1 階 VJP の子テープ記録（build_cgrads から呼ばれる）
// ---------------------------------------------------------------------

pub(super) fn empty_grad<'c>(child: &'c Tape, shape: &[usize]) -> Result<Var<'c>, AutodiffError> {
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
// ---------------------------------------------------------------------
// Det: 行別スケール＋2 のべき乗チャンク（イシュー #3810・決定記録 §46・§70）
// ---------------------------------------------------------------------

/// 1 回の `Mul` で掛ける 2 のべき乗の指数の上限（絶対値）。`2^100` は f32 で有限かつ厳密で、
/// 実行中の値が最終値へ向かって単調に動く限り途中で overflow／ゼロ化しない。
const CHUNK_EXP: i64 = 100;
/// チャンクの段数の上限。f32 で意味を持つ指数幅（約 ±300）の数倍で、作り込みの入力が子テープへ
/// 大量のノードを積むのを防ぐ（A04）。超えたら拒否する。
const MAX_CHUNK_STEPS: usize = 64;

/// `Det` の行別スケール計画（[`det_row_scale_plan`] の結果）。判定と構造が同じ値を見るための
/// 単一の源で、`create_graph.rs` の `Op::Det` 分岐が [`det_row_scaled_cgrad`] へ渡す。
pub(super) struct DetRowScale {
    /// 行 `i` のスケール指数 `e_i`（`r_i = 2^e_i` は行の最大絶対値以上で最小の 2 のべき乗。全ゼロ行は 0）。
    row_exp: Vec<i32>,
    /// `g` を割る共通スケール `s_g = 2^log2_sg`（`[2^-126, 2^127]` へクランプ。`g = 0` は 0）。
    log2_sg: i32,
    /// 復元指数 `c_i = E_i + log2_sg`、`E_i = Σ_{k≠i} e_k`。
    out_exp: Vec<i64>,
    /// 下流の縮小率 `max 2^-e_i`（`det(B)` ノードへ登録し、孫テープ再生の (iii) が使う）。
    r_inv: f64,
}

/// `x`（正の有限値）以上で最小の 2 のべき乗の指数。
fn pow2_ceil_exp(x: f64) -> i32 {
    let mut e = x.log2().ceil() as i32;
    while f64::powi(2.0, e) < x {
        e += 1;
    }
    while f64::powi(2.0, e - 1) >= x {
        e -= 1;
    }
    e
}

/// `2^k`（`|k| <= 100`）の厳密な f32。
fn pow2_f32(k: i32) -> f32 {
    f32::from_bits(((k + 127) as u32) << 23)
}

/// `exp` を同符号で各 `CHUNK_EXP` 以下の指数列へ分ける（係数を 1 個の f32 定数にまとめない）。
fn chunk_list(exp: i64) -> Vec<i32> {
    let sign: i64 = exp.signum();
    let mut rem = exp.abs();
    let mut out = Vec::new();
    while rem > 0 {
        let k = rem.min(CHUNK_EXP);
        out.push((sign * k) as i32);
        rem -= k;
    }
    out
}

/// 区分 (ii) だけで落ちる `Det` の入力に対し行別スケールの計画を作る。`Ok(None)` は「非スケール構造
/// （現行の V1）のまま」の合図で、子テープへスケール用のノードを足さない。`Err` は拒否（呼び出し側の
/// ロールバックで `child` は空へ戻る）。判定と構造の両方がこの結果を使う。
///
/// 手順は決定記録 §70.2。非スケール経路の検査（段・(iii)・(iv)）を従来と同じ順と式で行い、段（ii）だけが
/// 落ちた入力に限りスケール域で判定をやり直す。`g_requires_grad` は `g` が子テープ上で勾配を要する
/// （`g` へ戻る経路の cotangent を (iv) に含める）かどうか。
pub(super) fn det_row_scale_plan(
    a: &Tensor<f32>,
    g: &Tensor<f32>,
    g_requires_grad: bool,
    mult: f64,
) -> Result<Option<DetRowScale>, AutodiffError> {
    let n = a.shape()[0];
    if n == 0 {
        return Ok(None);
    }
    let gv = g.host_slice().first().map_or(0.0, |&v| f64::from(v));
    if !gv.is_finite() {
        return Err(reject("上流 g が非有限"));
    }
    let (x, det) = inverse_of(a, "行列式系")?;
    #[cfg(test)]
    let forced = tests::forced_scale();
    #[cfg(not(test))]
    let forced = false;
    if !forced {
        if det_like_stage_pass(&x, det, gv, true) {
            det_like_tail(&x, det, gv, n, true, mult)?;
            return Ok(None);
        }
        // (i) 真の 1 階値 `g * det * X^T`（`det_vjp` と同じ f64 計算）が f32 に収まらないなら拒否。
        let scale = gv * det;
        if !x.iter().all(|&v| stage_ok(scale * v)) {
            return Err(reject(
                "Det: 真の 1 階値が f32 の範囲を超える（区分 (i)。行別スケールでも救えない）",
            ));
        }
    }
    scale_plan_checked(a, gv, n, g_requires_grad).map(Some)
}

/// 行別スケールの計画本体とスケール域での判定（段・桁落ち・cotangent 上界）。
fn scale_plan_checked(
    a: &Tensor<f32>,
    gv: f64,
    n: usize,
    g_requires_grad: bool,
) -> Result<DetRowScale, AutodiffError> {
    let av = to_f64(a);
    let row_exp: Vec<i32> = av
        .chunks(n)
        .map(|r| {
            let m = max_abs(r);
            if m > 0.0 { pow2_ceil_exp(m) } else { 0 }
        })
        .collect();
    // B = D^-1 A（f64 で厳密。非零要素が正規化域に収まれば f32 でも厳密に表せる）。
    let b64: Vec<f64> = av
        .iter()
        .enumerate()
        .map(|(k, &v)| v * f64::powi(2.0, -row_exp[k / n]))
        .collect();
    if !all_stage_ok(&b64) {
        return Err(reject(
            "Det: 行別スケール後の B が f32 の正規化域に収まらない（行内の桁差が表現幅超過）",
        ));
    }
    let b32: Vec<f32> = b64.iter().map(|&v| v as f32).collect();
    let b_t = Tensor::new(b32, &[n, n])?;
    let (xb, det_b) = inverse_of(&b_t, "Det（スケール後）")?;
    if !(stage_ok(det_b) && all_stage_ok(&xb)) {
        return Err(reject(
            "Det: スケール後の det(B)／inv(B) が overflow／ゼロ化域に入る",
        ));
    }
    let log2_sg = if gv == 0.0 {
        0
    } else {
        pow2_ceil_exp(gv.abs()).clamp(-126, 127)
    };
    let gt = gv * f64::powi(2.0, -log2_sg);
    let q = gt * det_b;
    // 復元指数 c_i = E_i + log2_sg（checked 演算。範囲外は拒否）。
    let total: i64 = row_exp
        .iter()
        .try_fold(0i64, |acc, &e| acc.checked_add(i64::from(e)))
        .ok_or_else(|| reject("Det: 指数の和が範囲外"))?;
    let out_exp: Vec<i64> = row_exp
        .iter()
        .map(|&e| (total - i64::from(e)) + i64::from(log2_sg))
        .collect();
    // スケール域の段: g̃・g̃·det(B)・チャンク前の積 p_ij と、各チャンクを掛けた後の値。
    if !(stage_ok(gt) && stage_ok(q)) {
        return Err(reject(
            "Det: スケール後の g̃／g̃·det(B) が overflow／ゼロ化域に入る",
        ));
    }
    for (i, &c) in out_exp.iter().enumerate() {
        let list = chunk_list(c);
        if list.len() > MAX_CHUNK_STEPS {
            return Err(reject("Det: 復元指数のチャンク段数が上限を超える"));
        }
        for j in 0..n {
            let mut v = q * xb[j * n + i];
            if !stage_ok(v) {
                return Err(reject(
                    "Det: チャンク前の積が overflow／ゼロ化域に入る（指数復元前のアンダーフロー等）",
                ));
            }
            for &k in &list {
                v *= f64::powi(2.0, k);
                if !stage_ok(v) {
                    return Err(reject("Det: チャンク適用後の値が overflow／ゼロ化域に入る"));
                }
            }
        }
    }
    // 復元係数: 出力側の最大 2^{c_i}（保守側）、シード側の最大 1/r_k（B = D^-1 A の微分）。
    let k_out = out_exp.iter().fold(0.0f64, |m, &c| {
        m.max(f64::powi(2.0, c.clamp(-2000, 2000) as i32))
    });
    let r_inv = row_exp
        .iter()
        .fold(0.0f64, |m, &e| m.max(f64::powi(2.0, -e)));
    // (iii) 桁落ち。B 域の量で求め、復元係数を掛けた最終域で絶対 1e-5 と比べる（g̃ だけで評価しない）。
    if !det_second_order_rounding_ok(&xb, det_b, gt, n, k_out * r_inv) {
        return Err(reject(
            "Det: 二階微分の det と inv の経路間の桁落ちが REQ-2 の許容を超えうる（スケール域）",
        ));
    }
    // (iv) 二階の逆伝播の cotangent 上界（冪 3 は不変）。B 経由（出力側の 2^c と 1/r）と、
    // g が勾配を要する場合の g 経由（2^c_i / s_g = 2^E_i）の大きい方を使う。
    let e_max = out_exp
        .iter()
        .map(|&c| c - i64::from(log2_sg))
        .max()
        .unwrap_or(0);
    let path_b = k_out.max(1.0) * r_inv.max(1.0);
    let path_g = if g_requires_grad {
        f64::powi(2.0, e_max.clamp(-2000, 2000) as i32).max(1.0)
    } else {
        1.0
    };
    let bx = n as f64 * max_abs(&xb);
    let extra = gt.abs().max(1.0) * det_b.abs().max(1.0) * path_b.max(path_g);
    if !second_order_ok(&[bx], 3, extra) {
        return Err(reject(
            "Det: 二階の逆伝播の cotangent 上界が 2^125 以上（スケール域）",
        ));
    }
    Ok(DetRowScale {
        row_exp,
        log2_sg,
        out_exp,
        r_inv,
    })
}

/// 行列 `v`（`[n, m]`）の行 `i` に `2^{exps[i]}` を、`CHUNK_EXP` 以下のチャンクの `Mul` 列で掛ける。
/// 各行のチャンクは同符号なので、実行中の値は最終値へ向かって単調に動く（途中で最終値を超えない）。
fn scale_rows<'c>(child: &'c Tape, v: &Var<'c>, exps: &[i64]) -> Result<Var<'c>, AutodiffError> {
    let n = exps.len();
    let lists: Vec<Vec<i32>> = exps.iter().map(|&e| chunk_list(e)).collect();
    let steps = lists.iter().map(Vec::len).max().unwrap_or(0);
    let shape = v.shape();
    let mut cur = *v;
    for s in 0..steps {
        let col: Vec<f32> = lists
            .iter()
            .map(|l| pow2_f32(l.get(s).copied().unwrap_or(0)))
            .collect();
        let c = child
            .var_no_grad(&Tensor::new(col, &[n, 1])?)
            .broadcast_to(&shape)?;
        cur = cur.mul(&c)?;
    }
    Ok(cur)
}

/// スカラー `v`（形状 `[]`）に `2^exp` をチャンク列の `Mul` で掛ける。
fn scale_scalar<'c>(child: &'c Tape, v: &Var<'c>, exp: i64) -> Result<Var<'c>, AutodiffError> {
    let mut cur = *v;
    for k in chunk_list(exp) {
        let c = child.var_no_grad(&Tensor::new(vec![pow2_f32(k)], &[])?);
        cur = cur.mul(&c)?;
    }
    Ok(cur)
}

/// `Det` の行別スケール構造（§21.2 の `Det` の数値契約）。`B = D^-1 A`、`g̃ = g / s_g` から
/// `p = broadcast(g̃ · det(B)) ⊙ inv(B)^T` を作り、行 `i` に `2^{c_i}` をチャンクで掛け戻す。
/// 積む Op は `Mul`・`Det`・`Inv`・`Transpose`・`BroadcastTo` だけ（いずれも再生可能）。値は呼び出し側が
/// V3（`det_vjp`）で入れる。
pub(super) fn det_row_scaled_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    a_m: &Var<'c>,
    plan: &DetRowScale,
) -> Result<Var<'c>, AutodiffError> {
    let a_shape = a_m.shape();
    let neg_e: Vec<i64> = plan.row_exp.iter().map(|&e| -i64::from(e)).collect();
    let b = scale_rows(child, a_m, &neg_e)?;
    let gt = scale_scalar(child, g, -i64::from(plan.log2_sg))?;
    let det_b = b.det()?;
    // 孫テープ再生で当該 det(B) の桁落ち判定が下流の縮小（B = D^-1 A）を織り込めるよう登録する。
    child
        .det_downstream_gain
        .borrow_mut()
        .insert(det_b.node_id().0, plan.r_inv);
    let q = gt.mul(&det_b)?;
    let inv_t = b.inv()?.transpose(0, 1)?;
    let p = q.broadcast_to(&a_shape)?.mul(&inv_t)?;
    scale_rows(child, &p, &plan.out_exp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        /// スケール経路の強制フック（テストビルド限定。#3809 の `FORCED_PRESCALE` と同型）。
        static FORCED_SCALE: Cell<bool> = const { Cell::new(false) };
    }

    pub(super) fn forced_scale() -> bool {
        FORCED_SCALE.with(Cell::get)
    }

    /// 強制フックを設定して `f` を実行し、終了時に必ず解除する。
    pub(crate) fn with_forced_scale<R>(on: bool, f: impl FnOnce() -> R) -> R {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                FORCED_SCALE.with(|c| c.set(false));
            }
        }
        FORCED_SCALE.with(|c| c.set(on));
        let _reset = Reset;
        f()
    }

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).unwrap()
    }

    #[test]
    fn pow2_helpers_are_exact() {
        assert_eq!(pow2_ceil_exp(1.0), 0);
        assert_eq!(pow2_ceil_exp(1.5), 1);
        assert_eq!(pow2_ceil_exp(2.0), 1);
        assert_eq!(pow2_ceil_exp(0.25), -2);
        assert_eq!(pow2_ceil_exp(3e38), 128);
        assert_eq!(pow2_f32(0), 1.0);
        assert_eq!(pow2_f32(-100), 2f32.powi(-100));
        assert_eq!(pow2_f32(100), 2f32.powi(100));
        assert_eq!(chunk_list(250), vec![100, 100, 50]);
        assert_eq!(chunk_list(-101), vec![-100, -1]);
        assert!(chunk_list(0).is_empty());
    }

    #[test]
    fn plan_is_none_for_inputs_accepted_by_the_unscaled_guard() {
        let one = t(vec![1.0], &[]);
        for a in [
            t(vec![2.0, 0.5, -0.3, 1.5], &[2, 2]),
            t(vec![1.0, 0.0, 0.0, 2.0], &[2, 2]),
        ] {
            let p = with_forced_scale(false, || det_row_scale_plan(&a, &one, false, 1.0));
            assert!(matches!(p, Ok(None)), "非スケールのまま");
        }
    }

    #[test]
    fn plan_exists_for_large_diagonal_and_rejects_true_overflow() {
        let a = t(vec![1e20, 0.0, 0.0, 1e20], &[2, 2]);
        let one = t(vec![1.0], &[]);
        let p = det_row_scale_plan(&a, &one, false, 1.0).unwrap().unwrap();
        assert_eq!(p.row_exp, vec![67, 67]);
        assert_eq!(p.out_exp, vec![67, 67]);
        let eye = t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let huge = t(vec![3e38], &[]);
        assert!(det_row_scale_plan(&eye, &huge, false, 1.0).is_err());
    }

    #[test]
    fn forced_scale_cancellation_regression_is_still_rejected() {
        // 1e10 * det(dense) は強制スケールでも桁落ちで拒否される（§46.12 の落とし穴）。
        let a = t(vec![2.0, 0.5, -0.3, 1.5], &[2, 2]);
        let g = t(vec![1e10], &[]);
        let r = with_forced_scale(true, || det_row_scale_plan(&a, &g, false, 1.0));
        assert!(r.is_err(), "強制スケールでも拒否のはず");
    }

    fn new_tape() -> Tape {
        Tape::new_with_ops(crate::default_ops::naive_ops())
    }

    fn req2(a: f32, b: f32) -> bool {
        let (a, b) = (f64::from(a), f64::from(b));
        let diff = (a - b).abs();
        diff / a.abs().max(b.abs()).max(1e-12) < 1e-3 || diff < 1e-5
    }

    /// `loss = g * det(A)` の（1 階 `cg.grad`、二階 Hessian 行列）を子テープで求める。
    fn det_grad_and_hessian(
        a0: &[f32],
        n: usize,
        g: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), AutodiffError> {
        let tape = new_tape();
        let child = new_tape();
        let x = tape.var(&t(a0.to_vec(), &[n, n]));
        let gc = tape.var_no_grad(&t(vec![g], &[]));
        let loss = x.det()?.mul(&gc)?;
        let cg = tape.backward_create_graph(&loss, &child)?;
        let gx = cg.grad(&x)?.expect("到達");
        let cx = cg.child_var(&x)?.expect("写し");
        let grad = gx.to_tensor().contiguous().as_slice().expect("c").to_vec();
        let mut hess = Vec::new();
        for j in 0..n * n {
            let s = gx.narrow(0, j / n, 1)?.narrow(1, j % n, 1)?.sum(None)?;
            let rg = child.backward(&s)?;
            let row = rg.get(&cx)?.expect("到達").contiguous();
            hess.extend_from_slice(row.as_slice().expect("c"));
        }
        Ok((grad, hess))
    }

    /// 強制スケールとスケールなしの比較（§45 の「強制 k」と同じ確かめ方）: ほどほどの入力で、
    /// 行別スケール構造の `cg.grad` が 1 階と bit 一致し、二階（Hessian）が非スケール構造と REQ-2 で一致する。
    #[test]
    fn forced_scale_matches_unscaled_for_moderate_inputs() {
        let cases: [(&[f32], usize, f32); 3] = [
            (&[2.0, 0.5, -0.3, 1.5], 2, 1.0),
            (&[2.0, 0.5, -0.3, 1.5], 2, 0.25),
            (&[2.0, 0.5, 0.1, -0.3, 1.5, 0.2, 0.1, -0.2, 1.8], 3, 0.25),
        ];
        for (a0, n, g) in cases {
            let base = with_forced_scale(false, || det_grad_and_hessian(a0, n, g)).unwrap();
            let forced = with_forced_scale(true, || det_grad_and_hessian(a0, n, g)).unwrap();
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&base.0),
                bits(&forced.0),
                "cg.grad は 1 階と bit 一致 (n={n} g={g})"
            );
            for (i, (&a, &b)) in base.1.iter().zip(&forced.1).enumerate() {
                assert!(req2(a, b), "hessian[{i}] base={a} forced={b} (n={n} g={g})");
            }
        }
    }

    /// 強制スケールでも桁落ちの回帰（`1e10 * det(dense)`）は `Err(Backward)` で、`child` は空のまま。
    #[test]
    fn forced_scale_cancellation_regression_leaves_child_empty() {
        let tape = new_tape();
        let child = new_tape();
        let x = tape.var(&t(vec![2.0, 0.5, -0.3, 1.5], &[2, 2]));
        let gc = tape.var_no_grad(&t(vec![1e10], &[]));
        let loss = x.det().unwrap().mul(&gc).unwrap();
        let r = with_forced_scale(true, || tape.backward_create_graph(&loss, &child));
        assert!(matches!(r, Err(AutodiffError::Backward(_))));
        assert!(child.is_empty());
    }
}
