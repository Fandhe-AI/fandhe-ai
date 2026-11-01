//! `TapeF64::jvp`／`jacfwd`（イシュー #3787・親 #3719。設計記録
//! `docs/autodiff-functional-transforms-design.md` §36／§39）の CPU 統合テスト。
//!
//! 同じグラフを f32 の `Tape` と `TapeF64` に組み、`functional_ops::{jvp, jacfwd}` との
//! REQ-2 統一複合判定（`common::req2_close`。tolerance は新設しない）で突き合わせる。
//! 線形 Op だけで二階が 0 になる検証（§34.3）は作らず非線形な包みを入れる。入力は f32 で
//! 厳密に表せる値にする。実機の `#[ignore]` テストは #3788 の担当（本ファイルは CPU のみ）。

mod common;

use fandhe_ai_autodiff::f64_autograd::{TapeF64, VarF64};
use fandhe_ai_autodiff::functional_ops;
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

type R<T> = Result<T, AutodiffError>;

fn t64(data: &[f32], shape: &[usize]) -> R<Tensor<f64>> {
    let v: Vec<f64> = data.iter().map(|&x| f64::from(x)).collect();
    Tensor::new(v, shape).map_err(AutodiffError::Shape)
}

fn t32(data: &[f32], shape: &[usize]) -> R<Tensor<f32>> {
    Tensor::new(data.to_vec(), shape).map_err(AutodiffError::Shape)
}

fn vals64(x: &Tensor<f64>) -> Vec<f64> {
    x.contiguous().host_slice().to_vec()
}

fn assert_f32_f64_close(f32v: &Tensor<f32>, f64v: &Tensor<f64>, ctx: &str) {
    assert_eq!(f32v.shape(), f64v.shape(), "{ctx}: shape");
    let a = f32v.contiguous();
    let b = vals64(f64v);
    for (i, (x, y)) in a.host_slice().iter().zip(&b).enumerate() {
        assert!(
            common::req2_close(f64::from(*x), *y),
            "{ctx}[{i}]: f32 {x} f64 {y}"
        );
    }
}

fn new_tape() -> Tape {
    Tape::new_with_ops(common::naive_ops())
}

const X: [f32; 6] = [0.5, 1.5, 2.0, 1.0, 0.75, 3.0];
const S: [f32; 3] = [1.25, 0.5, 1.75];
const V: [f32; 6] = [1.0, -0.5, 2.0, 0.25, 1.5, -1.0];

/// add／mul／div／pow とブロードキャストを通す f64 グラフ。
fn g1_f64<'g, 't>(g: &'g TapeF64<'t>) -> R<(VarF64<'g, 't>, VarF64<'g, 't>)> {
    let x = g.var(&t64(&X, &[2, 3])?);
    let s = g.var_no_grad(&t64(&S, &[3])?);
    let two = g.var_no_grad(&t64(&[2.0], &[])?);
    let num = x.mul(&s)?.add(&x)?;
    let den = s.add(&two)?;
    let y = num.div(&den)?.add(&x.pow(&s)?)?;
    Ok((x, y))
}

fn g1_f32<'t>(g: &'t Tape) -> R<(Var<'t>, Var<'t>)> {
    let x = g.var(&t32(&X, &[2, 3])?);
    let s = g.var_no_grad(&t32(&S, &[3])?);
    let two = g.var_no_grad(&t32(&[2.0], &[])?);
    let num = x.mul(&s)?.add(&x)?;
    let den = s.add(&two)?;
    let y = num.div(&den)?.add(&x.pow(&s)?)?;
    Ok((x, y))
}

/// matmul／sum／mean を通す f64 グラフ（出力 shape [2]）。
fn g2_f64<'g, 't>(g: &'g TapeF64<'t>) -> R<(VarF64<'g, 't>, VarF64<'g, 't>)> {
    let x = g.var(&t64(&X, &[2, 3])?);
    let w = g.var_no_grad(&t64(&[0.5, -1.0, 1.5, 0.25, 2.0, 1.0], &[3, 2])?);
    let h = x.matmul(&w)?;
    let hh = h.mul(&h)?;
    let y = hh.sum(Some(1))?.add(&hh.mean(Some(1))?)?;
    Ok((x, y))
}

fn g2_f32<'t>(g: &'t Tape) -> R<(Var<'t>, Var<'t>)> {
    let x = g.var(&t32(&X, &[2, 3])?);
    let w = g.var_no_grad(&t32(&[0.5, -1.0, 1.5, 0.25, 2.0, 1.0], &[3, 2])?);
    let h = x.matmul(&w)?;
    let hh = h.mul(&h)?;
    let y = hh.sum(Some(1))?.add(&hh.mean(Some(1))?)?;
    Ok((x, y))
}

#[test]
fn jvp_matches_f32_add_mul_div_pow() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let (x, y) = g1_f64(&p)?;
    let got = p.jvp(&y, &x, &t64(&V, &[2, 3])?, &c)?;
    let (pt, ct) = (new_tape(), new_tape());
    let (x32, y32) = g1_f32(&pt)?;
    let want = functional_ops::jvp(&pt, &y32, &x32, &t32(&V, &[2, 3])?, &ct)?;
    assert_f32_f64_close(&want, &got, "jvp g1");
    Ok(())
}

#[test]
fn jvp_matches_f32_matmul_sum_mean() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let (x, y) = g2_f64(&p)?;
    let got = p.jvp(&y, &x, &t64(&V, &[2, 3])?, &c)?;
    let (pt, ct) = (new_tape(), new_tape());
    let (x32, y32) = g2_f32(&pt)?;
    let want = functional_ops::jvp(&pt, &y32, &x32, &t32(&V, &[2, 3])?, &ct)?;
    assert_f32_f64_close(&want, &got, "jvp g2");
    Ok(())
}

#[test]
fn jacfwd_matches_f32() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let (x, y) = g1_f64(&p)?;
    let got = p.jacfwd(&y, &x, &c)?;
    assert_eq!(got.shape(), &[2, 3, 2, 3]);
    let (pt, ct) = (new_tape(), new_tape());
    let (x32, y32) = g1_f32(&pt)?;
    let want = functional_ops::jacfwd(&pt, &y32, &x32, &ct)?;
    assert_f32_f64_close(&want, &got, "jacfwd g1");

    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let (x, y) = g2_f64(&p)?;
    let got = p.jacfwd(&y, &x, &c)?;
    assert_eq!(got.shape(), &[2, 2, 3]);
    let (pt, ct) = (new_tape(), new_tape());
    let (x32, y32) = g2_f32(&pt)?;
    let want = functional_ops::jacfwd(&pt, &y32, &x32, &ct)?;
    assert_f32_f64_close(&want, &got, "jacfwd g2");
    Ok(())
}

/// 閉形式（`y = x⊙x` → `2x⊙v`）と、`backward` の繰り返しで組んだヤコビアンの `J·v`。
#[test]
fn jvp_matches_closed_form_and_backward_jacobian() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let x = p.var(&t64(&X, &[6])?);
    let y = x
        .mul(&x)?
        .add(&x.pow(&p.var_no_grad(&t64(&[2.0; 6], &[6])?))?)?;
    let v = t64(&V, &[6])?;
    let got = vals64(&p.jvp(&y, &x, &v, &c)?);
    // y = 2x^2 → J = diag(4x)
    for i in 0..6 {
        let want = 4.0 * f64::from(X[i]) * f64::from(V[i]);
        assert!(
            common::req2_close(got[i], want),
            "closed[{i}]: {} {want}",
            got[i]
        );
    }
    // backward の繰り返しで J を行ごとに組む。
    let mut jv = [0.0f64; 6];
    for r in 0..6 {
        let pb = TapeF64::new(&base);
        let xb = pb.var(&t64(&X, &[6])?);
        let yb = xb
            .mul(&xb)?
            .add(&xb.pow(&pb.var_no_grad(&t64(&[2.0; 6], &[6])?))?)?;
        let mut onehot = [0.0f32; 6];
        onehot[r] = 1.0;
        let w = pb.var_no_grad(&t64(&onehot, &[6])?);
        let loss = yb.mul(&w)?.sum(None)?;
        let grads = pb.backward(&loss)?;
        let row = vals64(
            grads
                .get(&xb)?
                .ok_or(AutodiffError::GradientTrackingDisabled)?,
        );
        jv[r] = row.iter().zip(&V).map(|(a, b)| a * f64::from(*b)).sum();
    }
    for i in 0..6 {
        assert!(common::req2_close(got[i], jv[i]), "backward-J[{i}]");
    }
    Ok(())
}

#[test]
fn jacfwd_columns_match_one_hot_jvp() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let (x, y) = g2_f64(&p)?;
    let j = vals64(&p.jacfwd(&y, &x, &c)?);
    let (m, n) = (2usize, 6usize);
    for k in 0..n {
        let mut e = [0.0f32; 6];
        e[k] = 1.0;
        let (pk, ck) = (TapeF64::new(&base), TapeF64::new(&base));
        let (xk, yk) = g2_f64(&pk)?;
        let col = vals64(&pk.jvp(&yk, &xk, &t64(&e, &[2, 3])?, &ck)?);
        for i in 0..m {
            assert!(common::req2_close(j[i * n + k], col[i]), "J[{i},{k}]");
        }
    }
    Ok(())
}

#[test]
fn zero_paths_return_zeros_without_touching_tapes() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let x = p.var(&t64(&[1.0, 2.0], &[2])?);
    let konst = p.var_no_grad(&t64(&[3.0, 4.0], &[2])?);
    let v = t64(&[1.0, 1.0], &[2])?;
    // 追跡なしの出力
    let out = p.jvp(&konst, &x, &v, &c)?;
    assert_eq!(vals64(&out), vec![0.0, 0.0]);
    let j = p.jacfwd(&konst, &x, &c)?;
    assert_eq!(j.shape(), &[2, 2]);
    assert!(vals64(&j).iter().all(|&e| e == 0.0));
    // 要素数 0 の入力
    let empty = p.var(&t64(&[], &[0])?);
    let y0 = p.var(&t64(&[1.0, 2.0], &[2])?).mul(&konst)?;
    let z = p.jvp(&y0, &empty, &t64(&[], &[0])?, &c)?;
    assert_eq!(z.shape(), &[2]);
    assert!(vals64(&z).iter().all(|&e| e == 0.0));
    // J ≡ 0（input が output に届かない）
    let other = p.var(&t64(&[5.0, 6.0], &[2])?);
    let y = other.mul(&other)?;
    let z = p.jvp(&y, &x, &v, &c)?;
    assert!(vals64(&z).iter().all(|&e| e == 0.0));
    // 成功した前段は子へ写しを残すため、2 回目は新しい子を使う。
    let c2 = TapeF64::new(&base);
    let j = p.jacfwd(&y, &x, &c2)?;
    assert!(vals64(&j).iter().all(|&e| e == 0.0));
    Ok(())
}

/// 空入力の jacfwd は出力側の検査より先に零返却経路へ入り、結果 shape は `out ++ in`（0 要素）になる
/// （Bugbot 指摘の回帰）。要素数 0 の巨大 shape `[0, usize::MAX, 2]` を出力に使う。
/// 注: 要素数が非 0 で巨大な出力（バイト数溢れ）はデータなしでは構築できないため、
/// その分岐は `check_entry` の `jacobian && n == 0` 早期 return の構造で担保する。
#[test]
fn jacfwd_empty_input_with_huge_shape_output_returns_zeros() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let empty = p.var(&t64(&[], &[0])?);
    let huge = p.var(&t64(&[], &[0, usize::MAX, 2])?);
    let j = p.jacfwd(&huge, &empty, &c)?;
    assert_eq!(j.shape(), &[0, usize::MAX, 2, 0]);
    assert!(j.host_slice().is_empty());
    Ok(())
}

#[test]
fn entry_checks_reject_in_order() -> R<()> {
    let base = new_tape();
    let (p, c, o) = (
        TapeF64::new(&base),
        TapeF64::new(&base),
        TapeF64::new(&base),
    );
    let x = p.var(&t64(&[1.0, 2.0], &[2])?);
    let y = x.mul(&x)?;
    let ox = o.var(&t64(&[1.0, 2.0], &[2])?);
    let nog = p.var_no_grad(&t64(&[1.0, 2.0], &[2])?);
    let bad_v = t64(&[1.0], &[1])?;
    // 別テープ + 追跡なし + shape 不一致 → TapeMismatch が先
    assert!(matches!(
        p.jvp(&y, &ox, &bad_v, &c),
        Err(AutodiffError::TapeMismatch)
    ));
    // 追跡なし + shape 不一致 → GradientTrackingDisabled が先
    assert!(matches!(
        p.jvp(&y, &nog, &bad_v, &c),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    assert!(matches!(
        p.jvp(&y, &x, &bad_v, &c),
        Err(AutodiffError::Shape(_))
    ));
    assert!(matches!(
        p.jacfwd(&y, &ox, &c),
        Err(AutodiffError::TapeMismatch)
    ));
    Ok(())
}

#[test]
fn max_in_tracked_ancestors_is_rejected_and_child_stays_reusable() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let x = p.var(&t64(&[1.0, 2.0], &[2])?);
    let bad = x.max(None)?;
    let v = t64(&[1.0, 1.0], &[2])?;
    assert!(matches!(
        p.jvp(&bad, &x, &v, &c),
        Err(AutodiffError::Backward(_))
    ));
    // 同じ child で正しいグラフが通る
    let good = x.mul(&x)?;
    let out = p.jvp(&good, &x, &v, &c)?;
    assert_eq!(vals64(&out).len(), 2);
    Ok(())
}

#[test]
fn child_tape_errors_propagate() -> R<()> {
    let base = new_tape();
    let p = TapeF64::new(&base);
    let x = p.var(&t64(&[1.0, 2.0], &[2])?);
    let y = x.mul(&x)?;
    let v = t64(&[1.0, 1.0], &[2])?;
    // 親と同一の子
    assert!(matches!(
        p.jvp(&y, &x, &v, &p),
        Err(AutodiffError::Backward(_))
    ));
    // 空でない子
    let c = TapeF64::new(&base);
    let _ = c.var(&t64(&[1.0], &[1])?);
    assert!(matches!(
        p.jvp(&y, &x, &v, &c),
        Err(AutodiffError::Backward(_))
    ));
    assert!(matches!(
        p.jacfwd(&y, &x, &c),
        Err(AutodiffError::Backward(_))
    ));
    Ok(())
}
