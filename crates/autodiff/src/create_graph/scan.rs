//! 線形再帰 `S[a] = b[a] + A[a]·S[a+1]` の replayable な scan 補助
//! （イシュー #3803・親 #3732。決定記録 `docs/autodiff-higher-order-grad-decision.md`
//! §41.10・実装記録 §62）。
//!
//! 親モジュール [`super`] 配下の `cumulative_b`（`Cumprod`・`Logcumsumexp` の cgrad）から
//! 呼ばれる。将来 `IndexedScatterReduce` の `Prod`（イシュー #3802）が区切りマスク付きで
//! 同じ補助を使う（`cut` 引数がそのフック）。
//!
//! **方式**: Hillis–Steele 型の `⌈log2 n⌉` 段展開。段 `k = 1, 2, 4, …`（`k < n`）で
//! `S ← S + A ⊙ shift(S, k)`・`A ← A ⊙ shift(A, k)` を行う。`shift` は `dim` 方向に
//! `k` だけ前へ詰めて末尾を 0 で埋める。子テープへ積む Op は `Narrow`・`Concat`・`Mul`・
//! `Add`・`Where`・定数葉だけで、すべて `supports_create_graph() == true`（閉包性）。
//! 逐次ループ展開（O(n) ノード）と O(n²) の行列形は採らない（§41.10）。
//! 最終段では `A` を更新しない（使わない積で余計な overflow を作らないため）。
//!
//! ホスト側の再計算 [`scan_lane_host`] は同じ f32 演算順で各段を再現し、総括判定
//! （`cumulative_b` の `check_*`）と単体テストが共用する。

use fandhe_ai_tensor_core::Tensor;

use crate::error::AutodiffError;
use crate::tape::Tape;
use crate::var::Var;

/// `n` 要素の lane を畳み込むのに要する段数（`⌈log2 n⌉`。`n <= 1` は 0 段）。
pub(super) fn scan_steps(n: usize) -> usize {
    let mut steps = 0;
    let mut k = 1usize;
    while k < n {
        steps += 1;
        k = k.saturating_mul(2);
    }
    steps
}

/// `dim` 方向へ `k` だけ前へ詰め、末尾 `k` 個を 0 で埋める（`k < shape[dim]`）。
fn shift_left<'c>(
    child: &'c Tape,
    x: &Var<'c>,
    dim: usize,
    k: usize,
) -> Result<Var<'c>, AutodiffError> {
    let shape = x.shape();
    let n = shape[dim];
    let body = x.narrow(dim, k, n - k)?;
    let mut pad_shape = shape.clone();
    pad_shape[dim] = k;
    let pad = child.var_no_grad(&Tensor::zeros(&pad_shape)?);
    Var::cat(&[body, pad], dim)
}

/// 線形再帰 `S[a] = b[a] + A[a]·S[a+1]`（`S[n] = 0`）を `dim` 軸に沿って解く。
///
/// `cut` が `Some` なら、真の位置の `A` を先に 0 にする（区間の境界をまたぐ係数を断つ。
/// `where_cond` で選ぶので `inf·0` を作らない）。`b` と `A` は同 shape。
pub(super) fn linear_scan<'c>(
    child: &'c Tape,
    b: &Var<'c>,
    a: &Var<'c>,
    dim: usize,
    cut: Option<&Tensor<bool>>,
) -> Result<Var<'c>, AutodiffError> {
    let shape = b.shape();
    if a.shape() != shape || dim >= shape.len() {
        return Err(AutodiffError::Backward(format!(
            "create_graph: scan の shape 不整合（b {shape:?}・A {:?}・dim {dim}。内部契約違反）",
            a.shape()
        )));
    }
    let n = shape[dim];
    if n == 0 {
        return Ok(child.var_no_grad(&Tensor::zeros(&shape)?));
    }
    let mut a_cur = match cut {
        Some(mask) => {
            let zeros = child.var_no_grad(&Tensor::zeros(&shape)?);
            Var::where_cond(mask, &zeros, a)?
        }
        None => *a,
    };
    let mut s_cur = *b;
    let steps = scan_steps(n);
    let mut k = 1usize;
    for step in 0..steps {
        let s_shift = shift_left(child, &s_cur, dim, k)?;
        s_cur = s_cur.add(&a_cur.mul(&s_shift)?)?;
        if step + 1 < steps {
            let a_shift = shift_left(child, &a_cur, dim, k)?;
            a_cur = a_cur.mul(&a_shift)?;
        }
        k *= 2;
    }
    Ok(s_cur)
}

/// 1 lane のホスト再計算。[`linear_scan`] と同じ f32 演算順（積を f32 に丸めてから和）で
/// 各段を再現し、最終の `S` と全段（`S`・`A`）の絶対値の最大を返す。
/// 非有限値が 1 つでも現れたら最大は `f64::INFINITY` にする（NaN も INFINITY 扱い）。
/// `cut` は `A` を先に 0 にする位置（`None` なら切らない）。
pub(super) fn scan_lane_host(b: &[f32], a: &[f32], cut: Option<&[bool]>) -> (Vec<f32>, f64) {
    let n = b.len();
    let mut s: Vec<f32> = b.to_vec();
    let mut a_cur: Vec<f32> = a.to_vec();
    if let Some(mask) = cut {
        for (v, &c) in a_cur.iter_mut().zip(mask) {
            if c {
                *v = 0.0;
            }
        }
    }
    let mut max_abs = 0f64;
    let mut note = |v: &[f32]| {
        for &x in v {
            if x.is_finite() {
                max_abs = max_abs.max(f64::from(x).abs());
            } else {
                max_abs = f64::INFINITY;
            }
        }
    };
    note(&s);
    note(&a_cur);
    let steps = scan_steps(n);
    let mut k = 1usize;
    for step in 0..steps {
        let s_old = s.clone();
        for i in 0..n {
            let shifted = if i + k < n { s_old[i + k] } else { 0.0 };
            let prod = a_cur[i] * shifted;
            s[i] = s_old[i] + prod;
        }
        note(&s);
        if step + 1 < steps {
            let a_old = a_cur.clone();
            for i in 0..n {
                let shifted = if i + k < n { a_old[i + k] } else { 0.0 };
                a_cur[i] = a_old[i] * shifted;
            }
            note(&a_cur);
        }
        k *= 2;
    }
    (s, max_abs)
}

/// 1 lane の scan の逆伝播を、絶対値の f64 で歩いた上界。二階の逆伝播
/// （「cotangent × 微分係数」）の各段の大きさの総括判定が使う（決定記録 §27.3 と同じ流儀）。
///
/// `b`・`a` は scan の入力（`A` は `cut` 適用後の値を渡す）、`cot_out` は最終 `S` に流れ込む
/// cotangent の絶対値（seed の各要素の絶対値 1 以下の前提で呼び出し側が組む）。
/// 返り値は `(b への cotangent の上界, A への cotangent の上界, 歩いた全配列の絶対値の最大)`。
/// 各段の演算は [`scan_lane_host`] の順（S 更新→A 更新）の逆順で、`|·|` を取った非負値だけを
/// 扱うので、符号付きの実値の逆伝播の各中間値はここで求めた値以下になる。
pub(super) fn scan_cotangent_bound(
    b: &[f32],
    a: &[f32],
    cot_out: &[f64],
) -> (Vec<f64>, Vec<f64>, f64) {
    let n = b.len();
    let mut s: Vec<f64> = b.iter().map(|v| f64::from(*v).abs()).collect();
    let mut a_cur: Vec<f64> = a.iter().map(|v| f64::from(*v).abs()).collect();
    let steps = scan_steps(n);
    // 各段の更新前の (S, A)。逆伝播で係数として使う。
    let mut history: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(steps);
    let mut k = 1usize;
    for step in 0..steps {
        history.push((s.clone(), a_cur.clone()));
        let s_old = s.clone();
        for i in 0..n {
            let shifted = if i + k < n { s_old[i + k] } else { 0.0 };
            s[i] = s_old[i] + a_cur[i] * shifted;
        }
        if step + 1 < steps {
            let a_old = a_cur.clone();
            for i in 0..n {
                let shifted = if i + k < n { a_old[i + k] } else { 0.0 };
                a_cur[i] = a_old[i] * shifted;
            }
        }
        k *= 2;
    }
    let mut max_abs = 0f64;
    let mut note = |v: &[f64]| {
        for &x in v {
            max_abs = if x.is_nan() {
                f64::INFINITY
            } else {
                max_abs.max(x.abs())
            };
        }
    };
    let mut cot_s: Vec<f64> = cot_out.to_vec();
    let mut cot_a: Vec<f64> = vec![0.0; n];
    note(&cot_s);
    for step in (0..steps).rev() {
        let k = 1usize << step;
        let (s_old, a_old) = &history[step];
        let mut next_a = vec![0.0f64; n];
        // A 更新（最終段は更新しない）の逆: A_new[i] = A[i]·A[i+k]。
        if step + 1 < steps {
            for i in 0..n {
                if i + k < n {
                    next_a[i] += cot_a[i] * a_old[i + k];
                    next_a[i + k] += cot_a[i] * a_old[i];
                }
            }
        }
        // S 更新の逆: S_new[i] = S[i] + A[i]·S[i+k]。
        let mut next_s = cot_s.clone();
        for i in 0..n {
            if i + k < n {
                next_a[i] += cot_s[i] * s_old[i + k];
                next_s[i + k] += cot_s[i] * a_old[i];
            }
        }
        note(&next_s);
        note(&next_a);
        cot_s = next_s;
        cot_a = next_a;
    }
    (cot_s, cot_a, max_abs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_ops::naive_ops;

    fn lane_data(n: usize, seed: f32) -> (Vec<f32>, Vec<f32>) {
        let b: Vec<f32> = (0..n).map(|i| ((i as f32 + seed) * 0.7).sin()).collect();
        let a: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + seed) * 1.3).cos() * 0.9)
            .collect();
        (b, a)
    }

    fn sequential(b: &[f32], a: &[f32]) -> Vec<f64> {
        let n = b.len();
        let mut s = vec![0f64; n];
        let mut next = 0f64;
        for i in (0..n).rev() {
            next = f64::from(b[i]) + f64::from(a[i]) * next;
            s[i] = next;
        }
        s
    }

    #[test]
    fn scan_steps_is_ceil_log2() {
        let got: Vec<usize> = [0usize, 1, 2, 3, 4, 5, 8, 9, 1024, 1025]
            .iter()
            .map(|&n| scan_steps(n))
            .collect();
        assert_eq!(got, vec![0, 0, 1, 2, 2, 3, 3, 4, 10, 11]);
    }

    /// 構造（`Var` 演算列）の値が、ホストの逐次計算と REQ-2 で一致し、
    /// ホスト再計算（同じ f32 演算順）とは bit 一致する。rank 1 と中間軸の rank 3 を含む。
    #[test]
    fn scan_structure_matches_sequential_host() {
        for n in [1usize, 2, 3, 5, 8, 9] {
            let (b, a) = lane_data(n, 0.3);
            let tape = Tape::new_with_ops(naive_ops());
            let bv = tape.var(&Tensor::new(b.clone(), &[n]).unwrap());
            let av = tape.var(&Tensor::new(a.clone(), &[n]).unwrap());
            let s = linear_scan(&tape, &bv, &av, 0, None).unwrap();
            let got = s.to_tensor().host_slice().into_owned();
            let (host, _) = scan_lane_host(&b, &a, None);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&host), "n={n}");
            let seq = sequential(&b, &a);
            for (g, r) in got.iter().zip(&seq) {
                assert!(
                    fandhe_ai_tensor_core::tolerance::composite_close_f32(*g, *r as f32),
                    "n={n} got={g} ref={r}"
                );
            }
        }
        // rank 3・中間軸（dim = 1）。lane は (outer, inner) の組ごと。
        let (outer, n, inner) = (2usize, 5usize, 3usize);
        let (b, a) = lane_data(outer * n * inner, 1.1);
        let tape = Tape::new_with_ops(naive_ops());
        let bv = tape.var(&Tensor::new(b.clone(), &[outer, n, inner]).unwrap());
        let av = tape.var(&Tensor::new(a.clone(), &[outer, n, inner]).unwrap());
        let s = linear_scan(&tape, &bv, &av, 1, None).unwrap();
        let got = s.to_tensor().host_slice().into_owned();
        for o in 0..outer {
            for i in 0..inner {
                let idx = |p: usize| (o * n + p) * inner + i;
                let lb: Vec<f32> = (0..n).map(|p| b[idx(p)]).collect();
                let la: Vec<f32> = (0..n).map(|p| a[idx(p)]).collect();
                let (host, _) = scan_lane_host(&lb, &la, None);
                for p in 0..n {
                    assert_eq!(got[idx(p)].to_bits(), host[p].to_bits());
                }
            }
        }
    }

    /// 区切りマスクで境界をまたぐ係数を断つと、区間ごとの独立な再帰になる。
    #[test]
    fn scan_with_cut_splits_segments() {
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let a = vec![0.5f32, 0.5, 0.5, 0.5];
        let cut = vec![false, true, false, false];
        let tape = Tape::new_with_ops(naive_ops());
        let bv = tape.var(&Tensor::new(b.clone(), &[4]).unwrap());
        let av = tape.var(&Tensor::new(a.clone(), &[4]).unwrap());
        let mask = Tensor::new(cut.clone(), &[4]).unwrap();
        let s = linear_scan(&tape, &bv, &av, 0, Some(&mask)).unwrap();
        let got = s.to_tensor().host_slice().into_owned();
        // 区間 [0,1] と [2,3]。S[3]=4・S[2]=3+0.5·4=5・S[1]=2・S[0]=1+0.5·2=2。
        assert_eq!(got, vec![2.0, 2.0, 5.0, 4.0]);
        let (host, _) = scan_lane_host(&b, &a, Some(&cut));
        assert_eq!(host, got);
    }

    /// 全要素が正なら絶対値の歩きは実値の逆伝播と一致する（上界が実値を下回らないことの確認）。
    #[test]
    fn scan_cotangent_bound_matches_backward_for_positive_inputs() {
        for n in [1usize, 2, 3, 5, 8, 9] {
            let b: Vec<f32> = (0..n).map(|i| 0.5 + 0.1 * i as f32).collect();
            let a: Vec<f32> = (0..n).map(|i| 0.9 - 0.05 * i as f32).collect();
            let cot: Vec<f32> = (0..n).map(|i| 1.0 + 0.2 * i as f32).collect();
            let tape = Tape::new_with_ops(naive_ops());
            let bv = tape.var(&Tensor::new(b.clone(), &[n]).unwrap());
            let av = tape.var(&Tensor::new(a.clone(), &[n]).unwrap());
            let s = linear_scan(&tape, &bv, &av, 0, None).unwrap();
            let cv = tape.var_no_grad(&Tensor::new(cot.clone(), &[n]).unwrap());
            let loss = s.mul(&cv).unwrap().sum(None).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let gb = grads.get(&bv).unwrap().unwrap().host_slice().into_owned();
            // n = 1 では `A` が使われず、勾配は存在しない（上界は 0）。
            let ga = match grads.get(&av).unwrap() {
                Some(g) => g.host_slice().into_owned(),
                None => vec![0.0; n],
            };
            let cot64: Vec<f64> = cot.iter().map(|v| f64::from(*v)).collect();
            let (wb, wa, max_abs) = scan_cotangent_bound(&b, &a, &cot64);
            for i in 0..n {
                assert!(
                    fandhe_ai_tensor_core::tolerance::composite_close_f32(wb[i] as f32, gb[i]),
                    "n={n} b[{i}] bound={} grad={}",
                    wb[i],
                    gb[i]
                );
                // 最終段で更新しない A は、使われた段の係数だけが勾配に寄与する。
                assert!(
                    fandhe_ai_tensor_core::tolerance::composite_close_f32(wa[i] as f32, ga[i]),
                    "n={n} a[{i}] bound={} grad={}",
                    wa[i],
                    ga[i]
                );
            }
            assert!(max_abs.is_finite());
        }
    }

    #[test]
    fn scan_host_reports_non_finite_as_infinite_bound() {
        let (_, m) = scan_lane_host(&[f32::MAX, f32::MAX], &[2.0, 0.0], None);
        assert!(m.is_infinite());
        let (_, m) = scan_lane_host(&[1.0, 2.0], &[0.5, 0.0], None);
        assert!(m.is_finite());
    }
}
