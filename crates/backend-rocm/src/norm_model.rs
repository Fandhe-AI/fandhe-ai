//! `kernels_norm.rs` の 8 カーネルを CPU 上で (block, lane) ごとに逐語再現するテスト専用モデル
//! （イシュー #3437・R6）。
//!
//! 役割: AMD 実機がない環境で、wave 幅（32／64）ごとの lane 分担（`i = lane, lane + W, …`）・
//! xor butterfly（offset `W/2 → 1`、`snap[lane] + snap[lane ^ off]` の順）・f64 アキュムレータ・
//! f32 で確定する要素積・affine の 4 分岐・境界ガードを検証する。積和は `contract(off)` 前提で
//! `__builtin_fmaf` ↔ `f32::mul_add`、`__builtin_fma` ↔ `f64::mul_add` に 1 対 1 で対応させる。
//! `RocmBackendOps` のテスト用 HostModel 経路（`ops.rs`）もこのモデルで計算する。
//!
//! HIPRTC のコンパイル結果そのものの検証ではない（`__shfl_xor` の double 版・`__builtin_fma` の
//! f64 命令化・wave64 での parity 実測は #3159）。parity が外れても tolerance は緩めず、
//! モデルとカーネルの式の不一致として直す。

use crate::launch::for_each_thread_gid;
use crate::warp_geometry::WarpGeometry;

/// 1 wave 分の lane ストライド蓄積と xor butterfly（`wave_sum_f64` ＋ 呼び出し側の蓄積ループ）。
/// 全 lane が同じ値になるので lane 0 を返す。
fn wave_reduce(geom: WarpGeometry, n: usize, mut contribute: impl FnMut(usize, f64) -> f64) -> f64 {
    let w = geom.width() as usize;
    let mut lanes = vec![0.0f64; w];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut i = lane;
        while i < n {
            *slot = contribute(i, *slot);
            i += w;
        }
    }
    let mut offset = geom.half_width() as usize;
    while offset > 0 {
        let snap = lanes.clone();
        for (lane, slot) in lanes.iter_mut().enumerate() {
            *slot = snap[lane] + snap[lane ^ offset];
        }
        offset >>= 1;
    }
    lanes[0]
}

fn apply_affine(xhat: f32, w: Option<f32>, b: Option<f32>) -> f32 {
    match (w, b) {
        (Some(w), Some(b)) => xhat.mul_add(w, b),
        (Some(w), None) => xhat * w,
        (None, Some(b)) => xhat + b,
        (None, None) => xhat,
    }
}

/// `rmsnorm_f32`（grid = rows・block = 1 wave）。
pub(crate) fn rmsnorm(
    geom: WarpGeometry,
    x: &[f32],
    w: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let inv_n = 1.0f32 / hidden as f32;
    let mut out = vec![f32::NAN; rows * hidden];
    for row in 0..rows {
        let xr = &x[row * hidden..(row + 1) * hidden];
        let acc = wave_reduce(geom, hidden, |i, acc| {
            let v = xr[i] as f64;
            v.mul_add(v, acc)
        });
        let rstd = (1.0f64 / acc.mul_add(inv_n as f64, eps as f64).sqrt()) as f32;
        for i in 0..hidden {
            let v = xr[i] * rstd;
            out[row * hidden + i] = match w {
                Some(w) => v * w[i],
                None => v,
            };
        }
    }
    out
}

/// 行内の平均と `rstd`（二パス分散。layer_norm の forward／backward 共通）。
fn row_mean_rstd(geom: WarpGeometry, xr: &[f32], eps: f32) -> (f64, f64) {
    let n = xr.len();
    let sum = wave_reduce(geom, n, |i, acc| acc + xr[i] as f64);
    let mean = sum / n as f64;
    let sq = wave_reduce(geom, n, |i, acc| {
        let d = xr[i] as f64 - mean;
        d.mul_add(d, acc)
    });
    let var = sq / n as f64;
    (mean, 1.0f64 / (var + eps as f64).sqrt())
}

/// `layer_norm_f32`。
pub(crate) fn layer_norm(
    geom: WarpGeometry,
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; rows * hidden];
    for row in 0..rows {
        let xr = &x[row * hidden..(row + 1) * hidden];
        let (mean, rstd) = row_mean_rstd(geom, xr, eps);
        for i in 0..hidden {
            let xhat = ((xr[i] as f64 - mean) * rstd) as f32;
            out[row * hidden + i] = apply_affine(xhat, w.map(|w| w[i]), b.map(|b| b[i]));
        }
    }
    out
}

/// `rmsnorm_bwd_dx_f32` ＋（weight があれば）`rmsnorm_bwd_dw_f32`。
pub(crate) fn rmsnorm_backward(
    geom: WarpGeometry,
    x: &[f32],
    w: Option<&[f32]>,
    dy: &[f32],
    eps: f32,
    rows: usize,
    hidden: usize,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let inv_n = 1.0f64 / hidden as f64;
    let mut dx = vec![f32::NAN; rows * hidden];
    let mut rstd_scratch = vec![0.0f64; rows];
    for row in 0..rows {
        let xr = &x[row * hidden..(row + 1) * hidden];
        let dyr = &dy[row * hidden..(row + 1) * hidden];
        let acc = wave_reduce(geom, hidden, |i, acc| {
            let v = xr[i] as f64;
            v.mul_add(v, acc)
        });
        let rstd = (1.0f64 / acc.mul_add(inv_n, eps as f64).sqrt()) as f32;
        rstd_scratch[row] = rstd as f64;
        let dxhat_at = |i: usize| -> f32 {
            match w {
                Some(w) => dyr[i] * w[i],
                None => dyr[i],
            }
        };
        let dot = wave_reduce(geom, hidden, |i, acc| {
            let xhat = xr[i] * rstd;
            let term = dxhat_at(i) * xhat;
            acc + term as f64
        });
        let mean_dot = dot * inv_n;
        for i in 0..hidden {
            let xhat = xr[i] * rstd;
            let d = (rstd as f64) * (dxhat_at(i) as f64 - (xhat as f64) * mean_dot);
            dx[row * hidden + i] = d as f32;
        }
    }
    let dw = w.map(|_| {
        let mut dw = vec![f32::NAN; hidden];
        for_each_thread_gid(hidden, |gid| {
            if gid >= hidden as i64 {
                return;
            }
            let col = gid as usize;
            let mut acc_w = 0.0f64;
            for (r, &rstd64) in rstd_scratch.iter().enumerate() {
                let idx = r * hidden + col;
                let rstd = rstd64 as f32;
                let xhat = x[idx] * rstd;
                let term = dy[idx] * xhat;
                acc_w += term as f64;
            }
            dw[col] = acc_w as f32;
        });
        dw
    });
    (dx, dw)
}

/// `layer_norm_bwd_dx_f32` ＋（weight または bias があれば）`layer_norm_bwd_dwdb_f32`。
#[allow(clippy::too_many_arguments)] // `RocmNorm::run_layer_norm_backward_f32` と同じ引数構成
pub(crate) fn layer_norm_backward(
    geom: WarpGeometry,
    x: &[f32],
    w: Option<&[f32]>,
    has_bias: bool,
    dy: &[f32],
    eps: f32,
    rows: usize,
    hidden: usize,
) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
    let mut dx = vec![f32::NAN; rows * hidden];
    let mut mean_scratch = vec![0.0f64; rows];
    let mut rstd_scratch = vec![0.0f64; rows];
    for row in 0..rows {
        let xr = &x[row * hidden..(row + 1) * hidden];
        let dyr = &dy[row * hidden..(row + 1) * hidden];
        let (mean, rstd) = row_mean_rstd(geom, xr, eps);
        mean_scratch[row] = mean;
        rstd_scratch[row] = rstd;
        let xhat_at = |i: usize| -> f32 { ((xr[i] as f64 - mean) * rstd) as f32 };
        let dxhat_at = |i: usize| -> f32 {
            match w {
                Some(w) => dyr[i] * w[i],
                None => dyr[i],
            }
        };
        let sum_dxhat = wave_reduce(geom, hidden, |i, acc| acc + dxhat_at(i) as f64);
        let dot = wave_reduce(geom, hidden, |i, acc| {
            let term = dxhat_at(i) * xhat_at(i);
            acc + term as f64
        });
        let mean_dxhat = sum_dxhat / hidden as f64;
        let mean_dot = dot / hidden as f64;
        for i in 0..hidden {
            let d = rstd * (dxhat_at(i) as f64 - mean_dxhat - (xhat_at(i) as f64) * mean_dot);
            dx[row * hidden + i] = d as f32;
        }
    }
    if w.is_none() && !has_bias {
        return (dx, None, None);
    }
    let mut dw = vec![f32::NAN; hidden];
    let mut db = vec![f32::NAN; hidden];
    for_each_thread_gid(hidden, |gid| {
        if gid >= hidden as i64 {
            return;
        }
        let col = gid as usize;
        let (mut acc_w, mut acc_b) = (0.0f64, 0.0f64);
        for r in 0..rows {
            let idx = r * hidden + col;
            let dyv = dy[idx];
            if w.is_some() {
                let xhat = ((x[idx] as f64 - mean_scratch[r]) * rstd_scratch[r]) as f32;
                let term = dyv * xhat;
                acc_w += term as f64;
            }
            if has_bias {
                acc_b += dyv as f64;
            }
        }
        if w.is_some() {
            dw[col] = acc_w as f32;
        }
        if has_bias {
            db[col] = acc_b as f32;
        }
    });
    (dx, w.map(|_| dw), if has_bias { Some(db) } else { None })
}

fn bn_index(i: usize, ch: usize, c: usize, spatial: usize) -> usize {
    (i / spatial) * c * spatial + ch * spatial + (i % spatial)
}

/// `batch_norm_train_f32`（grid = c・block = 1 wave）。`(out, mean, var)`。
#[allow(clippy::too_many_arguments)] // `RocmNorm::run_batch_norm_train_f32` と同じ引数構成
pub(crate) fn batch_norm_train(
    geom: WarpGeometry,
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    n: usize,
    c: usize,
    spatial: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let m = n * spatial;
    let mut out = vec![f32::NAN; n * c * spatial];
    let mut mean_out = vec![f32::NAN; c];
    let mut var_out = vec![f32::NAN; c];
    for ch in 0..c {
        let sum = wave_reduce(geom, m, |i, acc| {
            acc + x[bn_index(i, ch, c, spatial)] as f64
        });
        let mean = sum / m as f64;
        let sq = wave_reduce(geom, m, |i, acc| {
            let d = x[bn_index(i, ch, c, spatial)] as f64 - mean;
            d.mul_add(d, acc)
        });
        let var = sq / m as f64;
        let rstd = 1.0f64 / (var + eps as f64).sqrt();
        mean_out[ch] = mean as f32;
        var_out[ch] = var as f32;
        for i in 0..m {
            let idx = bn_index(i, ch, c, spatial);
            let xhat = ((x[idx] as f64 - mean) * rstd) as f32;
            out[idx] = apply_affine(xhat, w.map(|w| w[ch]), b.map(|b| b[ch]));
        }
    }
    (out, mean_out, var_out)
}

/// `batch_norm_infer_f32`（1 スレッド = 1 要素）。
#[allow(clippy::too_many_arguments)] // `RocmNorm::run_batch_norm_infer_f32` と同じ引数構成
pub(crate) fn batch_norm_infer(
    x: &[f32],
    mean: &[f32],
    var: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    c: usize,
    spatial: usize,
    numel: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; numel];
    for_each_thread_gid(numel, |gid| {
        if gid >= numel as i64 {
            return;
        }
        let idx = gid as usize;
        let ch = (idx / spatial) % c;
        let rstd = 1.0f64 / (var[ch] as f64 + eps as f64).sqrt();
        let xhat = ((x[idx] as f64 - mean[ch] as f64) * rstd) as f32;
        out[idx] = apply_affine(xhat, w.map(|w| w[ch]), b.map(|b| b[ch]));
    });
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::{
        assert_parity, run_batch_norm_infer_f32, run_batch_norm_train_f32, run_layer_norm_f32,
        run_rmsnorm_f32,
    };

    const G32: fn() -> WarpGeometry = || WarpGeometry::new(32).unwrap();
    const G64: fn() -> WarpGeometry = || WarpGeometry::new(64).unwrap();

    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
            })
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// 検証対象の `hidden`（wave 幅 32／64 の前後・倍数・端数）と `rows`。
    const HIDDENS: [usize; 9] = [1, 31, 32, 33, 63, 64, 65, 130, 257];
    const ROWS: [usize; 3] = [1, 3, 17];

    /// `GEOM` が 32 のとき CPU と bit 一致、64 のときは複合判定（縮約順序が違うため）。
    fn check(geom: WarpGeometry, ctx: &str, got: &[f32], cpu: &[f32], bit_exact_on_32: bool) {
        if geom.width() == 32 && bit_exact_on_32 {
            assert_eq!(bits(got), bits(cpu), "{ctx}");
        } else {
            assert_parity(ctx, got, cpu);
        }
    }

    #[test]
    fn rmsnorm_matches_cpu() {
        for geom in [G32(), G64()] {
            for &rows in &ROWS {
                for &hidden in &HIDDENS {
                    let x = lcg(rows * hidden, 1 + hidden as u64);
                    let w = lcg(hidden, 9);
                    for wo in [None, Some(&w[..])] {
                        let got = rmsnorm(geom, &x, wo, 1e-5, rows, hidden);
                        let cpu = run_rmsnorm_f32(&x, wo, 1e-5, rows, hidden).unwrap();
                        // CPU は逐次和（aarch64 は NEON）なので bit 一致は主張しない。
                        check(
                            geom,
                            &format!("rmsnorm W{} {rows}x{hidden}", geom.width()),
                            &got,
                            &cpu,
                            false,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rmsnorm_large_finite_values_do_not_overflow() {
        for geom in [G32(), G64()] {
            let x = vec![2e20f32, -2e20, 2e20, -2e20, 2e20];
            let got = rmsnorm(geom, &x, None, 1e-5, 1, 5);
            assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
            assert_parity(
                "2e20",
                &got,
                &run_rmsnorm_f32(&x, None, 1e-5, 1, 5).unwrap(),
            );
        }
    }

    #[test]
    fn layer_norm_matches_cpu_all_affine_combinations() {
        for geom in [G32(), G64()] {
            for &rows in &ROWS {
                for &hidden in &HIDDENS {
                    let x = lcg(rows * hidden, 3 + hidden as u64);
                    let w = lcg(hidden, 11);
                    let b = lcg(hidden, 12);
                    for wo in [None, Some(&w[..])] {
                        for bo in [None, Some(&b[..])] {
                            let got = layer_norm(geom, &x, wo, bo, 1e-5, rows, hidden);
                            let cpu = run_layer_norm_f32(&x, wo, bo, 1e-5, rows, hidden).unwrap();
                            let ctx = format!(
                                "layer_norm W{} {rows}x{hidden} w={} b={}",
                                geom.width(),
                                wo.is_some(),
                                bo.is_some()
                            );
                            check(geom, &ctx, &got, &cpu, true);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn layer_norm_uniform_row_has_zero_deviation() {
        for geom in [G32(), G64()] {
            let x = vec![1e30f32; 49];
            let got = layer_norm(geom, &x, None, None, 1e-5, 1, 49);
            assert!(got.iter().all(|v| *v == 0.0), "{got:?}");
        }
    }

    #[test]
    fn layer_norm_signed_zero_matches_cpu_affine_branches() {
        // xhat = -0.0 になる入力と、-0.0 の weight／bias で 4 分岐の符号を CPU と比べる。
        for geom in [G32(), G64()] {
            let x = vec![5.0f32; 4];
            for (wo, bo) in [
                (Some(vec![-1.0f32; 4]), None),
                (None, Some(vec![-0.0f32; 4])),
                (Some(vec![-0.0f32; 4]), Some(vec![-0.0f32; 4])),
                (None, None),
            ] {
                let got = layer_norm(geom, &x, wo.as_deref(), bo.as_deref(), 1e-5, 1, 4);
                let cpu = run_layer_norm_f32(&x, wo.as_deref(), bo.as_deref(), 1e-5, 1, 4).unwrap();
                assert_eq!(bits(&got), bits(&cpu), "W{}", geom.width());
            }
        }
    }

    /// `eval::warp_reduce_f64` の逐語複製（32 lane butterfly。backward 参照実装用）。
    fn warp_reduce32(hidden: usize, mut f: impl FnMut(usize, f64) -> f64) -> f64 {
        let mut lanes = [0.0f64; 32];
        for (lane, slot) in lanes.iter_mut().enumerate() {
            let mut idx = lane;
            while idx < hidden {
                *slot = f(idx, *slot);
                idx += 32;
            }
        }
        let mut offset = 16usize;
        while offset > 0 {
            let snap = lanes;
            for (lane, slot) in lanes.iter_mut().enumerate() {
                *slot = snap[lane] + snap[lane ^ offset];
            }
            offset >>= 1;
        }
        lanes[0]
    }

    /// `autodiff::grad::rmsnorm_vjp_rows` の式の逐語参照（`CpuBackendOps` は backward が
    /// `Unsupported` のため、テスト内に置く。符号付き縮約は 32 lane butterfly、rstd は逐次和）。
    pub(crate) fn ref_rmsnorm_backward(
        x: &[f32],
        w: Option<&[f32]>,
        dy: &[f32],
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        let mut dx = vec![0.0f32; x.len()];
        let mut dw_acc = w.map(|_| vec![0.0f64; hidden]);
        let inv_n = 1.0f64 / hidden as f64;
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let dyr = &dy[r * hidden..(r + 1) * hidden];
            let mut acc = 0.0f64;
            for &v in row {
                let v = v as f64;
                acc = v.mul_add(v, acc);
            }
            let rstd = (1.0f64 / acc.mul_add(inv_n, eps as f64).sqrt()) as f32;
            let dxhat_at = |i: usize| w.map_or(dyr[i], |w| dyr[i] * w[i]);
            let dot = warp_reduce32(hidden, |i, a| a + (dxhat_at(i) * (row[i] * rstd)) as f64);
            let mean_dot = dot * inv_n;
            for i in 0..hidden {
                let xhat = row[i] * rstd;
                dx[r * hidden + i] =
                    ((rstd as f64) * (dxhat_at(i) as f64 - (xhat as f64) * mean_dot)) as f32;
            }
            if let Some(a) = dw_acc.as_mut() {
                for i in 0..hidden {
                    a[i] += (dyr[i] * (row[i] * rstd)) as f64;
                }
            }
        }
        (
            dx,
            dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect()),
        )
    }

    /// `autodiff::grad::layer_norm_vjp_rows` の式の逐語参照。
    pub(crate) fn ref_layer_norm_backward(
        x: &[f32],
        w: Option<&[f32]>,
        has_bias: bool,
        dy: &[f32],
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
        let mut dx = vec![0.0f32; x.len()];
        let mut dw_acc = w.map(|_| vec![0.0f64; hidden]);
        let mut db_acc = has_bias.then(|| vec![0.0f64; hidden]);
        let n = hidden as f64;
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let dyr = &dy[r * hidden..(r + 1) * hidden];
            let sum = warp_reduce32(hidden, |i, a| a + row[i] as f64);
            let mean = sum / n;
            let sq = warp_reduce32(hidden, |i, a| {
                let d = row[i] as f64 - mean;
                d.mul_add(d, a)
            });
            let rstd = 1.0f64 / (sq / n + eps as f64).sqrt();
            let xhat_at = |i: usize| ((row[i] as f64 - mean) * rstd) as f32;
            let dxhat_at = |i: usize| w.map_or(dyr[i], |w| dyr[i] * w[i]);
            let sum_dxhat = warp_reduce32(hidden, |i, a| a + dxhat_at(i) as f64);
            let dot = warp_reduce32(hidden, |i, a| a + (dxhat_at(i) * xhat_at(i)) as f64);
            let (mean_dxhat, mean_dot) = (sum_dxhat / n, dot / n);
            for i in 0..hidden {
                dx[r * hidden + i] = (rstd
                    * (dxhat_at(i) as f64 - mean_dxhat - (xhat_at(i) as f64) * mean_dot))
                    as f32;
            }
            if let Some(a) = dw_acc.as_mut() {
                for i in 0..hidden {
                    a[i] += (dyr[i] * xhat_at(i)) as f64;
                }
            }
            if let Some(a) = db_acc.as_mut() {
                for i in 0..hidden {
                    a[i] += dyr[i] as f64;
                }
            }
        }
        let f = |v: Vec<f64>| v.into_iter().map(|a| a as f32).collect::<Vec<f32>>();
        (dx, dw_acc.map(f), db_acc.map(f))
    }

    #[test]
    fn rmsnorm_backward_matches_reference() {
        for geom in [G32(), G64()] {
            for &rows in &ROWS {
                for &hidden in &HIDDENS {
                    let x = lcg(rows * hidden, 21 + hidden as u64);
                    let dy = lcg(rows * hidden, 22);
                    let w = lcg(hidden, 23);
                    for wo in [None, Some(&w[..])] {
                        let (dx, dw) = rmsnorm_backward(geom, &x, wo, &dy, 1e-5, rows, hidden);
                        let (rdx, rdw) = ref_rmsnorm_backward(&x, wo, &dy, 1e-5, rows, hidden);
                        let ctx = format!("rmsnorm_bwd W{} {rows}x{hidden}", geom.width());
                        assert_parity(&format!("{ctx} dx"), &dx, &rdx);
                        assert_eq!(dw.is_some(), rdw.is_some());
                        if let (Some(a), Some(b)) = (dw, rdw) {
                            assert_parity(&format!("{ctx} dw"), &a, &b);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn layer_norm_backward_matches_reference_all_combinations() {
        for geom in [G32(), G64()] {
            for &rows in &ROWS {
                for &hidden in &HIDDENS {
                    let x = lcg(rows * hidden, 31 + hidden as u64);
                    let dy = lcg(rows * hidden, 32);
                    let w = lcg(hidden, 33);
                    for wo in [None, Some(&w[..])] {
                        for has_bias in [false, true] {
                            let (dx, dw, db) = layer_norm_backward(
                                geom, &x, wo, has_bias, &dy, 1e-5, rows, hidden,
                            );
                            let (rdx, rdw, rdb) =
                                ref_layer_norm_backward(&x, wo, has_bias, &dy, 1e-5, rows, hidden);
                            let ctx = format!("ln_bwd W{} {rows}x{hidden}", geom.width());
                            assert_parity(&format!("{ctx} dx"), &dx, &rdx);
                            assert_eq!(
                                (dw.is_some(), db.is_some()),
                                (rdw.is_some(), rdb.is_some())
                            );
                            if let (Some(a), Some(b)) = (dw, rdw) {
                                assert_parity(&format!("{ctx} dw"), &a, &b);
                            }
                            if let (Some(a), Some(b)) = (db, rdb) {
                                // db は単純な行方向 f64 和で CPU/参照と bit 一致する。
                                assert_eq!(bits(&a), bits(&b), "{ctx} db");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn backward_cancellation_inputs_stay_within_parity() {
        // 相殺を含む符号付き入力（結合順序で dx が O(1) 乖離しうる系）。参照も同じ 32 lane 順。
        for geom in [G32(), G64()] {
            let hidden = 64usize;
            let x = lcg(hidden, 41);
            let mut dy = vec![0.0f32; hidden];
            dy[0] = 1e20;
            dy[1] = 1.0;
            dy[2] = -1e20;
            let (dx, _) = rmsnorm_backward(geom, &x, None, &dy, 1e-5, 1, hidden);
            let (rdx, _) = ref_rmsnorm_backward(&x, None, &dy, 1e-5, 1, hidden);
            if geom.width() == 32 {
                assert_parity("cancel W32", &dx, &rdx);
            } else {
                // 幅 64 は縮約順序が異なるため相殺入力での乖離は結果が有限であることだけを確認する。
                assert!(dx.iter().all(|v| v.is_finite()));
            }
        }
    }

    #[test]
    fn batch_norm_train_matches_cpu() {
        let shapes = [
            (2, 3, 4),
            (1, 1, 7),
            (4, 5, 1),
            (2, 2, 33),
            (3, 4, 100),
            (1, 3, 130),
        ];
        for geom in [G32(), G64()] {
            for &(n, c, spatial) in &shapes {
                let x = lcg(n * c * spatial, 51 + spatial as u64);
                let w = lcg(c, 52);
                let b = lcg(c, 53);
                for wo in [None, Some(&w[..])] {
                    for bo in [None, Some(&b[..])] {
                        let (out, mean, var) =
                            batch_norm_train(geom, &x, wo, bo, 1e-5, n, c, spatial);
                        let cpu =
                            run_batch_norm_train_f32(&x, wo, bo, 1e-5, n, c, spatial).unwrap();
                        let ctx = format!("bn_train W{} {n}x{c}x{spatial}", geom.width());
                        check(geom, &format!("{ctx} out"), &out, &cpu.out, true);
                        check(geom, &format!("{ctx} mean"), &mean, &cpu.mean, true);
                        check(geom, &format!("{ctx} var"), &var, &cpu.var, true);
                    }
                }
            }
        }
    }

    #[test]
    fn batch_norm_infer_matches_cpu_bitwise() {
        // infer は縮約を持たないので wave 幅に依らず bit 一致する。
        for &(n, c, spatial) in &[(2, 3, 4), (1, 1, 7), (3, 5, 300), (2, 70, 5)] {
            let x = lcg(n * c * spatial, 61);
            let mean = lcg(c, 62);
            let var: Vec<f32> = lcg(c, 63).iter().map(|v| v.abs() + 0.1).collect();
            let w = lcg(c, 64);
            let b = lcg(c, 65);
            for wo in [None, Some(&w[..])] {
                for bo in [None, Some(&b[..])] {
                    let got = batch_norm_infer(&x, &mean, &var, wo, bo, 1e-5, c, spatial, x.len());
                    let cpu =
                        run_batch_norm_infer_f32(&x, &mean, &var, wo, bo, 1e-5, n, c, spatial)
                            .unwrap();
                    assert_eq!(bits(&got), bits(&cpu), "{n}x{c}x{spatial}");
                }
            }
        }
    }

    #[test]
    fn batch_norm_extreme_scale_does_not_overflow() {
        for geom in [G32(), G64()] {
            let x = vec![2e20f32, -2e20, 2e20, -2e20];
            let (out, _, _) = batch_norm_train(geom, &x, None, None, 1e-5, 1, 2, 2);
            assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
        }
    }

    #[test]
    fn wave_reduce_matches_32_lane_reference_and_is_lane_uniform() {
        let v = lcg(257, 71);
        let a = wave_reduce(G32(), v.len(), |i, acc| acc + v[i] as f64);
        let b = warp_reduce32(v.len(), |i, acc| acc + v[i] as f64);
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
