//! `kernels_loss.rs` のカーネルを CPU 上で (block, thread) ごとに逐語再現するテスト専用モデル
//! （イシュー #3436・R5）。
//!
//! 役割: AMD 実機がない環境で、(1) grid-stride の被覆と `LOSS_MAX_BLOCKS` 上限での 2 周以上の
//! ループ、(2) wave 幅 32／64 での XOR butterfly → `warp_sums[WARPS_PER_BLOCK]`（未書き込みの
//! スロットを読んだらエラー）→ wave 0 の再縮約、(3) finalize の 1 ブロック縮約と `factor`、
//! (4) backward の書き込み回数（各出力がちょうど 1 回）を検証する。要素関数は device 関数と同じ
//! 式・同じ演算順序で写す（CPU 参照 `backend-cpu` の elem 関数と同じ）。`CpuBackendOps` と
//! 統一複合判定（forward）または bit 一致（backward）で突合する。HIPRTC のコンパイル結果や
//! 数学関数の精度の検証ではない（実機確認は #3159）。`ops.rs` の `HostModel` 経路もここを呼ぶ。

use crate::kernels_loss::{LOSS_BLOCK_DIM, LOSS_MAX_BLOCKS};
use crate::launch::for_each_thread_gid;
use crate::loss::num_blocks;
use crate::warp_geometry::WarpGeometry;

/// 縮約中に起きうるモデル上の違反（実機では未初期化読みや境界外読みになる）。
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ModelError {
    UninitRead(usize),
    OutOfBounds(usize),
}

const BLOCK: usize = LOSS_BLOCK_DIM as usize;

/// `__shfl_xor(v, off, width)` を `half` から 1 まで: 全 lane が旧値を同時に読む。
fn butterfly(vals: &mut [f32], width: usize, half: usize) {
    let mut off = half;
    while off > 0 {
        let old = vals.to_vec();
        for (i, v) in vals.iter_mut().enumerate() {
            let src = i ^ off;
            let same_group = src / width == i / width && src < old.len();
            let other = if same_group { old[src] } else { old[i] };
            *v = old[i] + other;
        }
        off >>= 1;
    }
}

/// `block_reduce_sum`（カーネルの device 関数）: スレッドごとの値 256 個から wave 0 の値を返す。
///
/// `buf_len`・`read_bound` は `WARPS_PER_BLOCK` を渡すのが正しい形で、CUDA 由来の 8 固定を
/// 渡すと wave64 で未初期化読みになることをテストで固定する。
pub(crate) fn block_reduce_with(
    g: WarpGeometry,
    thread_vals: &[f32],
    buf_len: usize,
    read_bound: usize,
) -> Result<f32, ModelError> {
    let w = g.width() as usize;
    let mut v = thread_vals.to_vec();
    butterfly(&mut v, w, g.half_width() as usize);
    let mut warp_sums: Vec<Option<f32>> = vec![None; buf_len];
    for warp in 0..v.len() / w {
        *warp_sums
            .get_mut(warp)
            .ok_or(ModelError::OutOfBounds(warp))? = Some(v[warp * w]);
    }
    let mut second = vec![0.0f32; w];
    for (lane, slot) in second.iter_mut().enumerate() {
        if lane < read_bound {
            *slot = warp_sums
                .get(lane)
                .ok_or(ModelError::OutOfBounds(lane))?
                .ok_or(ModelError::UninitRead(lane))?;
        }
    }
    butterfly(&mut second, w, g.half_width() as usize);
    Ok(second[0])
}

fn block_reduce(g: WarpGeometry, thread_vals: &[f32]) -> f32 {
    let n = g.warps_per_block(LOSS_BLOCK_DIM).expect("検証済みの幅") as usize;
    block_reduce_with(g, thread_vals, n, n).expect("WARPS_PER_BLOCK 導出なら違反しない")
}

/// 2 段縮約の逐語モデル。`step(idx, acc)` が grid-stride の 1 要素分の更新（partial カーネルの
/// ループ本体）。`work` は 1 段目が処理する要素数（1 以上）。
fn forward(g: WarpGeometry, work: usize, factor: f32, step: impl Fn(usize, f32) -> f32) -> f32 {
    if work == 0 {
        return 0.0;
    }
    let blocks = num_blocks(work);
    assert!(blocks <= LOSS_MAX_BLOCKS as usize);
    let stride = blocks * BLOCK;
    let mut partial = vec![0.0f32; blocks];
    for (b, slot) in partial.iter_mut().enumerate() {
        let mut thread_vals = vec![0.0f32; BLOCK];
        for (t, tv) in thread_vals.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            let mut idx = b * BLOCK + t;
            while idx < work {
                acc = step(idx, acc);
                idx += stride;
            }
            *tv = acc;
        }
        *slot = block_reduce(g, &thread_vals);
    }
    // finalize: 1 ブロック。thread t が partial[t], partial[t+256], ... を順に加算する。
    let mut thread_vals = vec![0.0f32; BLOCK];
    for (t, tv) in thread_vals.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        let mut idx = t;
        while idx < partial.len() {
            acc += partial[idx];
            idx += BLOCK;
        }
        *tv = acc;
    }
    block_reduce(g, &thread_vals) * factor
}

/// 1 スレッド 1 要素の backward カーネルの逐語モデル。範囲外アクセスは panic、書き込み回数は検査する。
fn backward(numel: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    let mut out = vec![f32::NAN; numel];
    let mut writes = vec![0u8; numel];
    for_each_thread_gid(numel, |gid| {
        if gid < numel as i64 {
            let i = gid as usize;
            out[i] = f(i);
            writes[i] += 1;
        }
    });
    assert!(
        writes.iter().all(|&w| w == 1),
        "各出力はちょうど 1 回書かれる"
    );
    out
}

// ---- 要素関数（device 関数の逐語写し。CPU 参照と同じ式・同じ順序） ----

fn huber_elem_loss(d: f32, kind: i32, delta: f32) -> f32 {
    let abs_d = d.abs();
    if kind == 1 {
        if abs_d < delta {
            return 0.5 * (d / delta) * d;
        }
        return abs_d - 0.5 * delta;
    }
    if abs_d < delta {
        return 0.5 * d * d;
    }
    delta * (abs_d - 0.5 * delta)
}

fn huber_elem_grad(d: f32, kind: i32, delta: f32) -> f32 {
    if d.is_nan() {
        return d;
    }
    let abs_d = d.abs();
    if kind == 1 {
        if abs_d < delta {
            return d / delta;
        }
        return 1.0f32.copysign(d);
    }
    if abs_d < delta {
        return d;
    }
    delta.copysign(d)
}

fn bce_elem_loss(input: f32, target: f32, kind: i32) -> f32 {
    if kind == 0 {
        let log_p = input.ln().max(-100.0);
        let log_1mp = (1.0 - input).ln().max(-100.0);
        return -(target * log_p + (1.0 - target) * log_1mp);
    }
    if input >= 0.0 {
        return (1.0 - target) * input + (-input).exp().ln_1p();
    }
    -target * input + input.exp().ln_1p()
}

fn bce_elem_grad_input(input: f32, target: f32, kind: i32) -> f32 {
    if kind == 0 {
        let denom = (input * (1.0 - input)).max(1e-12);
        return (input - target) / denom;
    }
    let sigmoid = if input >= 0.0 {
        1.0 / (1.0 + (-input).exp())
    } else {
        let e = input.exp();
        e / (1.0 + e)
    };
    sigmoid - target
}

fn kl_div_elem_loss(input: f32, target: f32, kind: i32) -> f32 {
    if kind == 0 {
        if target == 0.0 {
            return 0.0;
        }
        return target * (target.ln() - input);
    }
    target.exp() * (target - input)
}

fn kl_div_elem_grad_input(target: f32, kind: i32) -> f32 {
    if kind == 0 { -target } else { -target.exp() }
}

// ---- 公開（クレート内）モデル。ops.rs の HostModel 経路が呼ぶ ----

pub(crate) fn mse(g: WarpGeometry, pred: &[f32], target: &[f32], factor: f32) -> f32 {
    forward(g, pred.len(), factor, |i, acc| {
        let diff = pred[i] - target[i];
        diff.mul_add(diff, acc)
    })
}

pub(crate) fn mse_backward(pred: &[f32], target: &[f32], scale: f32) -> Vec<f32> {
    backward(pred.len(), |i| scale * (pred[i] - target[i]))
}

pub(crate) fn huber(
    g: WarpGeometry,
    pred: &[f32],
    target: &[f32],
    kind: i32,
    delta: f32,
    factor: f32,
) -> f32 {
    forward(g, pred.len(), factor, |i, acc| {
        acc + huber_elem_loss(pred[i] - target[i], kind, delta)
    })
}

pub(crate) fn huber_backward(
    pred: &[f32],
    target: &[f32],
    kind: i32,
    delta: f32,
    scale: f32,
) -> Vec<f32> {
    backward(pred.len(), |i| {
        scale * huber_elem_grad(pred[i] - target[i], kind, delta)
    })
}

pub(crate) fn bce(g: WarpGeometry, input: &[f32], target: &[f32], kind: i32, factor: f32) -> f32 {
    forward(g, input.len(), factor, |i, acc| {
        acc + bce_elem_loss(input[i], target[i], kind)
    })
}

pub(crate) fn bce_backward(input: &[f32], target: &[f32], kind: i32, scale: f32) -> Vec<f32> {
    backward(input.len(), |i| {
        scale * bce_elem_grad_input(input[i], target[i], kind)
    })
}

pub(crate) fn kl_div(
    g: WarpGeometry,
    input: &[f32],
    target: &[f32],
    kind: i32,
    factor: f32,
) -> f32 {
    forward(g, input.len(), factor, |i, acc| {
        acc + kl_div_elem_loss(input[i], target[i], kind)
    })
}

pub(crate) fn kl_div_backward(input: &[f32], target: &[f32], kind: i32, scale: f32) -> Vec<f32> {
    let _ = input;
    backward(target.len(), |i| {
        scale * kl_div_elem_grad_input(target[i], kind)
    })
}

pub(crate) fn nll(
    g: WarpGeometry,
    input: &[f32],
    targets: &[i32],
    outer: usize,
    num_classes: usize,
    inner: usize,
    factor: f32,
) -> f32 {
    forward(g, outer * inner, factor, |idx, acc| {
        let o = idx / inner;
        let i = idx % inner;
        let t = targets[idx];
        if t < 0 || t as usize >= num_classes {
            return acc;
        }
        acc - input[(o * num_classes + t as usize) * inner + i]
    })
}

pub(crate) fn nll_backward(
    targets: &[i32],
    outer: usize,
    num_classes: usize,
    inner: usize,
    scale: f32,
) -> Vec<f32> {
    let numel = outer * num_classes * inner;
    let n = outer * inner;
    let mut out = vec![0.0f32; numel];
    let mut writes = vec![0u8; numel];
    for_each_thread_gid(n, |gid| {
        if gid < n as i64 {
            let idx = gid as usize;
            let o = idx / inner;
            let i = idx % inner;
            let t = targets[idx];
            if t >= 0 && (t as usize) < num_classes {
                let at = (o * num_classes + t as usize) * inner + i;
                out[at] = -scale;
                writes[at] += 1;
            }
        }
    });
    assert!(writes.iter().all(|&w| w <= 1), "書き込み先は一意");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};
    use fandhe_ai_tensor_core::{
        BackendOps, BceKind, HuberKind, KlDivTarget, MseReduction, Tensor,
    };

    const SIZES: [usize; 7] = [1, 7, 255, 256, 257, 4097, 256 * 1024 + 3];

    fn geoms() -> [WarpGeometry; 2] {
        [
            WarpGeometry::new(32).unwrap(),
            WarpGeometry::new(64).unwrap(),
        ]
    }

    fn lcg(len: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                lo + (hi - lo) * ((s >> 8) as f32 / (1u32 << 24) as f32)
            })
            .collect()
    }

    fn t1(v: &[f32]) -> Tensor<f32> {
        Tensor::new(v.to_vec(), &[v.len()]).unwrap()
    }

    fn factor(red: MseReduction, n: usize) -> f32 {
        match red {
            MseReduction::Mean => 1.0 / n as f32,
            _ => 1.0,
        }
    }

    fn same_bits(label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{label}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let ok = (g.is_nan() && w.is_nan()) || g.to_bits() == w.to_bits();
            assert!(ok, "{label}[{i}]: {g} vs {w}");
        }
    }

    #[test]
    fn mse_matches_cpu_for_both_widths() {
        let cpu = CpuBackendOps;
        for g in geoms() {
            for &n in &SIZES {
                let p = lcg(n, 1, -2.0, 2.0);
                let t = lcg(n, 2, -2.0, 2.0);
                for red in [MseReduction::Mean, MseReduction::Sum] {
                    let want = cpu.mse_loss(&t1(&p), &t1(&t), red).unwrap();
                    let got = mse(g, &p, &t, factor(red, n));
                    assert_parity(
                        &format!("mse w{} n{n}", g.width()),
                        &[got],
                        want.as_slice().unwrap(),
                    );
                }
                let want = cpu.mse_loss_backward(&t1(&p), &t1(&t), 0.37).unwrap();
                same_bits(
                    "mse bwd",
                    &mse_backward(&p, &t, 0.37),
                    want.as_slice().unwrap(),
                );
            }
        }
    }

    #[test]
    fn huber_matches_cpu_including_boundaries_and_nan() {
        let cpu = CpuBackendOps;
        for g in geoms() {
            for &n in &SIZES {
                let mut p = lcg(n, 3, -3.0, 3.0);
                let mut t = lcg(n, 4, -3.0, 3.0);
                // |d| == delta と NaN を混ぜる。
                if n >= 3 {
                    p[0] = 1.5;
                    t[0] = 0.5;
                    p[1] = f32::NAN;
                    p[2] = -1.0;
                    t[2] = 0.0;
                }
                for (kind, kind_i) in [(HuberKind::Huber, 0), (HuberKind::SmoothL1, 1)] {
                    for delta in [1.0f32, 0.5] {
                        for red in [MseReduction::Mean, MseReduction::Sum] {
                            let want = cpu.huber_loss(&t1(&p), &t1(&t), kind, delta, red).unwrap();
                            let got = huber(g, &p, &t, kind_i, delta, factor(red, n));
                            let w = want.as_slice().unwrap();
                            if w[0].is_nan() {
                                assert!(got.is_nan());
                            } else {
                                assert_parity(&format!("huber w{} n{n}", g.width()), &[got], w);
                            }
                        }
                        let want = cpu
                            .huber_loss_backward(&t1(&p), &t1(&t), kind, delta, 0.5)
                            .unwrap();
                        same_bits(
                            "huber bwd",
                            &huber_backward(&p, &t, kind_i, delta, 0.5),
                            want.as_slice().unwrap(),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bce_matches_cpu_including_extremes() {
        let cpu = CpuBackendOps;
        for g in geoms() {
            for &n in &SIZES {
                let tg: Vec<f32> = lcg(n, 6, 0.0, 1.0);
                // Probabilities は (0,1) の中に端点 0/1 を混ぜる。Logits は大きな正負の値を混ぜる。
                let mut probs = lcg(n, 5, 0.01, 0.99);
                let mut logits = lcg(n, 7, -8.0, 8.0);
                if n >= 4 {
                    probs[0] = 0.0;
                    probs[1] = 1.0;
                    logits[0] = 80.0;
                    logits[1] = -80.0;
                }
                for (kind, kind_i, input) in [
                    (BceKind::Probabilities, 0, &probs),
                    (BceKind::Logits, 1, &logits),
                ] {
                    for red in [MseReduction::Mean, MseReduction::Sum] {
                        let want = cpu.bce_loss(&t1(input), &t1(&tg), kind, red).unwrap();
                        let got = bce(g, input, &tg, kind_i, factor(red, n));
                        assert_parity(
                            &format!("bce k{kind_i} w{} n{n}", g.width()),
                            &[got],
                            want.as_slice().unwrap(),
                        );
                    }
                    let want = cpu
                        .bce_loss_backward(&t1(input), &t1(&tg), kind, 0.25)
                        .unwrap();
                    same_bits(
                        "bce bwd",
                        &bce_backward(input, &tg, kind_i, 0.25),
                        want.as_slice().unwrap(),
                    );
                }
            }
        }
    }

    #[test]
    fn kl_div_matches_cpu_including_zero_target() {
        let cpu = CpuBackendOps;
        for g in geoms() {
            for &n in &SIZES {
                let input = lcg(n, 8, -4.0, 0.0);
                let mut probs = lcg(n, 9, 0.0, 1.0);
                let logp = lcg(n, 10, -4.0, 0.0);
                if n >= 2 {
                    probs[0] = 0.0;
                }
                for (kind, kind_i, tg) in [
                    (KlDivTarget::Probabilities, 0, &probs),
                    (KlDivTarget::LogProbabilities, 1, &logp),
                ] {
                    for red in [MseReduction::Mean, MseReduction::Sum] {
                        let want = cpu.kl_div_loss(&t1(&input), &t1(tg), kind, red).unwrap();
                        let got = kl_div(g, &input, tg, kind_i, factor(red, n));
                        assert_parity(
                            &format!("kl_div k{kind_i} w{} n{n}", g.width()),
                            &[got],
                            want.as_slice().unwrap(),
                        );
                    }
                    let want = cpu
                        .kl_div_loss_backward(&t1(&input), &t1(tg), kind, 0.5)
                        .unwrap();
                    same_bits(
                        "kl_div bwd",
                        &kl_div_backward(&input, tg, kind_i, 0.5),
                        want.as_slice().unwrap(),
                    );
                }
            }
        }
    }

    #[test]
    fn nll_matches_cpu_for_class_dims_and_inner() {
        let cpu = CpuBackendOps;
        // (shape, class_dim): 先頭・中間・末尾と inner > 1。
        let cases: [(&[usize], usize); 5] = [
            (&[5], 0),
            (&[4, 6], 0),
            (&[3, 5, 4], 1),
            (&[37, 11], 1),
            (&[300, 9, 2], 1),
        ];
        for g in geoms() {
            for (shape, cd) in cases {
                let numel: usize = shape.iter().product();
                let outer: usize = shape[..cd].iter().product();
                let c = shape[cd];
                let inner: usize = shape[cd + 1..].iter().product();
                let n = outer * inner;
                let input = lcg(numel, 11, -5.0, 0.0);
                let targets: Vec<i32> = (0..n).map(|s| ((s * 7 + 3) % c) as i32).collect();
                let it = Tensor::new(input.clone(), shape).unwrap();
                let tt = Tensor::new(targets.clone(), &[n]).unwrap();
                for red in [MseReduction::Mean, MseReduction::Sum] {
                    let want = cpu.nll_loss(&it, &tt, cd, red).unwrap();
                    let got = nll(g, &input, &targets, outer, c, inner, factor(red, n));
                    assert_parity(
                        &format!("nll {shape:?} cd{cd} w{}", g.width()),
                        &[got],
                        want.as_slice().unwrap(),
                    );
                }
                let want = cpu.nll_loss_backward(shape, &tt, cd, 0.125).unwrap();
                same_bits(
                    "nll bwd",
                    &nll_backward(&targets, outer, c, inner, 0.125),
                    want.as_slice().unwrap(),
                );
            }
        }
    }

    #[test]
    fn nll_large_sample_count_exercises_grid_stride_wraparound() {
        let cpu = CpuBackendOps;
        let n = 256 * 1024 + 3;
        let c = 3;
        let input = lcg(n * c, 12, -3.0, 0.0);
        let targets: Vec<i32> = (0..n).map(|s| (s % c) as i32).collect();
        let it = Tensor::new(input.clone(), &[n, c]).unwrap();
        let tt = Tensor::new(targets.clone(), &[n]).unwrap();
        let want = cpu.nll_loss(&it, &tt, 1, MseReduction::Sum).unwrap();
        for g in geoms() {
            let got = nll(g, &input, &targets, n, c, 1, 1.0);
            assert_parity("nll big", &[got], want.as_slice().unwrap());
        }
    }

    #[test]
    fn empty_inputs_are_zero_and_empty() {
        for g in geoms() {
            assert_eq!(mse(g, &[], &[], 1.0), 0.0);
            assert_eq!(nll(g, &[], &[], 0, 3, 4, 1.0), 0.0);
        }
        assert!(mse_backward(&[], &[], 1.0).is_empty());
        assert_eq!(nll_backward(&[], 0, 3, 4, 1.0), Vec::<f32>::new());
    }

    #[test]
    fn cuda_fixed_eight_warps_breaks_on_wave64() {
        let g64 = WarpGeometry::new(64).unwrap();
        let input = vec![1.0f32; 256];
        assert_eq!(
            block_reduce_with(g64, &input, 8, 8),
            Err(ModelError::UninitRead(4))
        );
        assert_eq!(block_reduce_with(g64, &input, 4, 4).unwrap(), 256.0);
        let g32 = WarpGeometry::new(32).unwrap();
        assert_eq!(block_reduce_with(g32, &input, 8, 8).unwrap(), 256.0);
        assert_eq!(
            block_reduce_with(g32, &input, 4, 4),
            Err(ModelError::OutOfBounds(4))
        );
    }

    #[test]
    fn integer_partial_sums_are_exact_for_both_widths() {
        for g in geoms() {
            let p: Vec<f32> = (0..4097).map(|i| (i % 5) as f32).collect();
            let t = vec![0.0f32; p.len()];
            let want: f32 = p.iter().map(|x| x * x).sum();
            assert_eq!(mse(g, &p, &t, 1.0), want);
        }
    }
}
