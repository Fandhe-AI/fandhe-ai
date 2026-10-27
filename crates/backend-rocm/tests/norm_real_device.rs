//! ROCm 実機（MI300X・gfx942・wave64）での正規化 6 メソッドを CPU 参照と突き合わせる実機 parity
//! テスト（イシュー #3437・R6。実測は #3159）。
//!
//! - 役割: `RocmBackendOps` の `rmsnorm`／`layer_norm`／`batch_norm_train`／`batch_norm_infer` を
//!   `CpuBackendOps` と、backward 2 つを本ファイル内の f64 参照と `compare`（`assert_parity` と同値。REQ-2 統一複合判定。
//!   相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。許容誤差は変更しない）で照合する。
//! - 呼び出し文脈: CI では実行しない（`#[ignore]`）。HIPRTC のコンパイル（`__shfl_xor` の double 版・
//!   `__builtin_fma` の f64 命令化）と wave64 の縮約順序の確認は、実機で本テストを実行して行う。
//!   parity が外れても tolerance は緩めず、カーネルとホスト側モデル（`norm_model.rs`）の式の不一致として直す。

use std::cell::Cell;

use fandhe_ai_backend_cpu::{CpuBackendOps, compare};
use fandhe_ai_backend_rocm::{GfxTarget, RocmBackendOps};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

const EPS: f32 = 1e-5;

fn data(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

fn t(shape: &[usize], seed: u64) -> Tensor<f32> {
    Tensor::new(data(shape.iter().product(), seed), shape).unwrap()
}

fn flat(x: &Tensor<f32>) -> Vec<f32> {
    x.contiguous().as_slice().unwrap().to_vec()
}

thread_local! {
    /// テスト関数（= スレッド）ごとの複合判定 fail 要素数の累計。`finish` が 0 を要求する。
    static FAILS: Cell<usize> = const { Cell::new(0) };
    /// 同一テスト内で行名が衝突しないよう、記録順の連番を付ける。
    static SEQ: Cell<usize> = const { Cell::new(0) };
}

/// 1 件ぶんの記録行（`ROCM_PARITY_NORM`）を出力し、fail 数を累計する。
///
/// 判定は `assert_parity` と同値（`compare` の `fail_count == 0`）。外れても最初の 1 件で止めず、
/// 全ケースの fail 要素数と最大誤差をログに残してから `finish` で落とす（#3159 作業 4）。
/// 行名は `<label>#<連番>` とし、README の表へ転記するときに行が衝突しないようにする。
/// tolerance 定数・`compare` 本体は変更しない。
fn record(label: &str, got: &[f32], want: &[f32]) {
    let r = compare(got, want).unwrap_or_else(|e| panic!("compare {label}: {e:?}"));
    let bit_mismatch = got
        .iter()
        .zip(want)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count();
    let seq = SEQ.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    eprintln!(
        "ROCM_PARITY_NORM\t{label}#{seq}\ttotal={}\tfail={}\tmax_abs={:e}\tmax_rel={:e}\tmean_abs={:e}\tp99_abs={:e}\tbit_mismatch={bit_mismatch}",
        r.total, r.fail_count, r.max_abs_diff, r.max_rel_err, r.mean_abs_diff, r.p99_abs_diff
    );
    FAILS.with(|f| f.set(f.get() + r.fail_count));
}

/// 累計 fail が 0 でなければ落とす（REQ-2 複合判定を外れた要素がある）。
fn finish() {
    let fails = FAILS.with(Cell::get);
    assert_eq!(fails, 0, "REQ-2 複合判定を外れた要素がある");
}

/// 行ごとの f64 参照（RMSNorm backward）。
fn ref_rmsnorm_backward(
    x: &[f32],
    w: Option<&[f32]>,
    dy: &[f32],
    rows: usize,
    hidden: usize,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let mut dx = vec![0.0f32; x.len()];
    let mut dw = vec![0.0f64; hidden];
    for r in 0..rows {
        let xr = &x[r * hidden..(r + 1) * hidden];
        let dyr = &dy[r * hidden..(r + 1) * hidden];
        let ms: f64 = xr.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hidden as f64;
        let rstd = 1.0 / (ms + EPS as f64).sqrt();
        let dxhat: Vec<f64> = (0..hidden)
            .map(|i| dyr[i] as f64 * w.map_or(1.0, |w| w[i] as f64))
            .collect();
        let mean_dot: f64 = (0..hidden)
            .map(|i| dxhat[i] * xr[i] as f64 * rstd)
            .sum::<f64>()
            / hidden as f64;
        for i in 0..hidden {
            let xhat = xr[i] as f64 * rstd;
            dx[r * hidden + i] = (rstd * (dxhat[i] - xhat * mean_dot)) as f32;
            dw[i] += dyr[i] as f64 * xhat;
        }
    }
    (dx, w.map(|_| dw.into_iter().map(|v| v as f32).collect()))
}

/// 行ごとの f64 参照（LayerNorm backward）。
fn ref_layer_norm_backward(
    x: &[f32],
    w: Option<&[f32]>,
    has_bias: bool,
    dy: &[f32],
    rows: usize,
    hidden: usize,
) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
    let mut dx = vec![0.0f32; x.len()];
    let (mut dw, mut db) = (vec![0.0f64; hidden], vec![0.0f64; hidden]);
    let n = hidden as f64;
    for r in 0..rows {
        let xr = &x[r * hidden..(r + 1) * hidden];
        let dyr = &dy[r * hidden..(r + 1) * hidden];
        let mean: f64 = xr.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var: f64 = xr.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let rstd = 1.0 / (var + EPS as f64).sqrt();
        let xhat: Vec<f64> = xr.iter().map(|&v| (v as f64 - mean) * rstd).collect();
        let dxhat: Vec<f64> = (0..hidden)
            .map(|i| dyr[i] as f64 * w.map_or(1.0, |w| w[i] as f64))
            .collect();
        let mean_dxhat = dxhat.iter().sum::<f64>() / n;
        let mean_dot = (0..hidden).map(|i| dxhat[i] * xhat[i]).sum::<f64>() / n;
        for i in 0..hidden {
            dx[r * hidden + i] = (rstd * (dxhat[i] - mean_dxhat - xhat[i] * mean_dot)) as f32;
            dw[i] += dyr[i] as f64 * xhat[i];
            db[i] += dyr[i] as f64;
        }
    }
    let f = |v: Vec<f64>| v.into_iter().map(|a| a as f32).collect::<Vec<f32>>();
    (dx, w.map(|_| f(dw)), has_bias.then(|| f(db)))
}

#[test]
#[ignore = "ROCm 実機が必要（#3159 で実行）"]
fn forward_matches_cpu_on_device() {
    let o = RocmBackendOps::new(GfxTarget::Gfx942);
    let c = CpuBackendOps;
    for shape in [
        vec![1usize, 1],
        vec![3, 31],
        vec![5, 64],
        vec![4, 65],
        vec![2, 3, 257],
    ] {
        let hidden = *shape.last().unwrap();
        let x = t(&shape, 1);
        let w = t(&[hidden], 2);
        let b = t(&[hidden], 3);
        for (wo, bo) in [
            (None, None),
            (Some(&w), None),
            (None, Some(&b)),
            (Some(&w), Some(&b)),
        ] {
            let got = o.rmsnorm(&x, wo, EPS).unwrap();
            let want = c.rmsnorm(&x, wo, EPS).unwrap();
            record("rmsnorm", &flat(&got), &flat(&want));
            let got = o.layer_norm(&x, wo, bo, EPS).unwrap();
            let want = c.layer_norm(&x, wo, bo, EPS).unwrap();
            record("layer_norm", &flat(&got), &flat(&want));
        }
    }
    finish();
}

#[test]
#[ignore = "ROCm 実機が必要（#3159 で実行）"]
fn backward_matches_reference_on_device() {
    let o = RocmBackendOps::new(GfxTarget::Gfx942);
    for (rows, hidden) in [(1usize, 1usize), (3, 31), (5, 64), (4, 65), (6, 257)] {
        let shape = [rows, hidden];
        let x = t(&shape, 11);
        let dy = t(&shape, 12);
        let w = t(&[hidden], 13);
        for wo in [None, Some(&w)] {
            let wf = wo.map(flat);
            let (dx, dw) = o.rmsnorm_backward(&x, wo, &dy, EPS).unwrap();
            let (rdx, rdw) =
                ref_rmsnorm_backward(&flat(&x), wf.as_deref(), &flat(&dy), rows, hidden);
            record("rmsnorm_backward dx", &flat(&dx), &rdx);
            if let (Some(a), Some(b)) = (dw, rdw) {
                record("rmsnorm_backward dw", &flat(&a), &b);
            }
            for has_bias in [false, true] {
                let (dx, dw, db) = o.layer_norm_backward(&x, wo, has_bias, &dy, EPS).unwrap();
                let (rdx, rdw, rdb) = ref_layer_norm_backward(
                    &flat(&x),
                    wf.as_deref(),
                    has_bias,
                    &flat(&dy),
                    rows,
                    hidden,
                );
                record("layer_norm_backward dx", &flat(&dx), &rdx);
                if let (Some(a), Some(b)) = (dw, rdw) {
                    record("layer_norm_backward dw", &flat(&a), &b);
                }
                if let (Some(a), Some(b)) = (db, rdb) {
                    record("layer_norm_backward db", &flat(&a), &b);
                }
            }
        }
    }
    finish();
}

#[test]
#[ignore = "ROCm 実機が必要（#3159 で実行）"]
fn batch_norm_matches_cpu_on_device() {
    let o = RocmBackendOps::new(GfxTarget::Gfx942);
    let c = CpuBackendOps;
    for shape in [vec![4usize, 3], vec![2, 3, 17], vec![2, 5, 4, 33]] {
        let ch = shape[1];
        let x = t(&shape, 21);
        let w = t(&[ch], 22);
        let b = t(&[ch], 23);
        let mean = t(&[ch], 24);
        let var = Tensor::new(data(ch, 25).iter().map(|v| v.abs() + 0.1).collect(), &[ch]).unwrap();
        for (wo, bo) in [
            (None, None),
            (Some(&w), None),
            (None, Some(&b)),
            (Some(&w), Some(&b)),
        ] {
            let got = o.batch_norm_train(&x, wo, bo, EPS).unwrap();
            let want = c.batch_norm_train(&x, wo, bo, EPS).unwrap();
            record("bn out", &flat(&got.output), &flat(&want.output));
            record("bn mean", &flat(&got.batch_mean), &flat(&want.batch_mean));
            record("bn var", &flat(&got.batch_var), &flat(&want.batch_var));
            let got = o.batch_norm_infer(&x, &mean, &var, wo, bo, EPS).unwrap();
            let want = c.batch_norm_infer(&x, &mean, &var, wo, bo, EPS).unwrap();
            record("bn infer", &flat(&got), &flat(&want));
        }
    }
    finish();
}
