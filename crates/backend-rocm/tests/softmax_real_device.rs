//! ROCm 実機（MI300X・gfx942）での softmax 系 3 演算を CPU 参照と突き合わせる実機 parity テスト
//! （イシュー #3435。実測は #3159 段 B）。
//!
//! - 役割: `RocmSoftmax`（softmax・log_softmax）と `RocmLogSoftmaxBackward` の出力を
//!   `CpuBackendOps::softmax`／`log_softmax`（backward は同式のテスト内参照）と
//!   `fandhe_ai_backend_cpu::compare`（REQ-2 統一複合判定。相対誤差 1e-3 未満 または 絶対誤差 1e-5
//!   未満。定数は変更しない）で判定する。`exp2f`／`expf`／`logf` の丸めと wave64 の結合順が CPU と
//!   異なるため bit 一致は assert せず、観測値（`bit_mismatch`）として出力するだけにする。
//! - 呼び出し文脈: CI では実行しない（`#[ignore]`）。`run_rocm_parity.sh` はクレートの `--ignored`
//!   テストを全件実行するため本ファイルも自動で含まれ、本ファイルの失敗は run の終了コードに影響する。
//!   記録は `docs/perf/logs/rocm-parity-3159/README.md` の「softmax 系（#3435）」節。
//! - ログ行は `ROCM_PARITY_SOFTMAX` で始まる（GEMM の `ROCM_PARITY` 行と同形式）。

use fandhe_ai_backend_cpu::{CpuBackendOps, compare};
use fandhe_ai_backend_rocm::{GfxTarget, RocmLogSoftmaxBackward, RocmSoftmax};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn data(len: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * amp
        })
        .collect()
}

fn cpu_fwd(x: &[f32], rows: usize, cols: usize, log: bool) -> Vec<f32> {
    let t = Tensor::new(x.to_vec(), &[rows, cols]).unwrap();
    let o = if log {
        CpuBackendOps::new().log_softmax(&t, 1)
    } else {
        CpuBackendOps::new().softmax(&t, 1)
    }
    .unwrap();
    o.as_slice().unwrap().to_vec()
}

/// `autodiff::grad::log_softmax_vjp_along` と同式（f64 逐次和・最後に 1 回 f32）。
fn cpu_bwd(y: &[f32], g: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let b = r * cols;
        let sum: f64 = (0..cols).map(|i| f64::from(g[b + i])).sum();
        for i in 0..cols {
            out[b + i] = (f64::from(g[b + i]) - f64::from(y[b + i].exp()) * sum) as f32;
        }
    }
    out
}

/// 1 件ぶんの記録行を出力し、複合判定の fail 数を返す。
fn record(name: &str, got: &[f32], want: &[f32]) -> usize {
    let r = compare(got, want).unwrap_or_else(|e| panic!("compare {name}: {e:?}"));
    let bits = got
        .iter()
        .zip(want)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count();
    eprintln!(
        "ROCM_PARITY_SOFTMAX\t{name}\ttotal={}\tfail={}\tmax_abs={:e}\tmax_rel={:e}\tmean_abs={:e}\tp99_abs={:e}\tbit_mismatch={bits}",
        r.total, r.fail_count, r.max_abs_diff, r.max_rel_err, r.mean_abs_diff, r.p99_abs_diff
    );
    r.fail_count
}

const SHAPES: [(usize, usize); 9] = [
    (1, 1),
    (3, 4),
    (7, 63),
    (5, 64),
    (5, 65),
    (4, 128),
    (3, 1000),
    (2, 4096),
    (70000, 3),
];

#[test]
#[ignore = "ROCm 実機が必要（#3159 段 B で実行）"]
fn softmax_and_log_softmax_match_cpu_reference() {
    let sm = RocmSoftmax::new(GfxTarget::Gfx942).expect("HIPRTC コンパイル・ロードに成功すること");
    let mut fails = 0;
    for (rows, cols) in SHAPES {
        for amp in [1.0f32, 30.0] {
            let x = data(rows * cols, 1 + cols as u64, amp);
            let name = format!("softmax {rows}x{cols} amp={amp}");
            fails += record(
                &name,
                &sm.run_softmax_f32(&x, rows, cols).unwrap(),
                &cpu_fwd(&x, rows, cols, false),
            );
            let name = format!("log_softmax {rows}x{cols} amp={amp}");
            fails += record(
                &name,
                &sm.run_log_softmax_f32(&x, rows, cols).unwrap(),
                &cpu_fwd(&x, rows, cols, true),
            );
        }
    }
    // 極値（有限のまま）。
    let x = [f32::MAX, 0.0, f32::MAX, 1.0, -f32::MAX];
    let got = sm.run_softmax_f32(&x, 1, 5).unwrap();
    assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
    fails += record("softmax extreme", &got, &cpu_fwd(&x, 1, 5, false));
    let x = [-f32::MAX; 5];
    let got = sm.run_softmax_f32(&x, 1, 5).unwrap();
    assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
    fails += record("softmax all-min", &got, &cpu_fwd(&x, 1, 5, false));
    assert_eq!(fails, 0, "REQ-2 複合判定を外れた要素がある");
}

#[test]
#[ignore = "ROCm 実機が必要（#3159 段 B で実行）"]
fn log_softmax_backward_matches_cpu_reference() {
    let bw = RocmLogSoftmaxBackward::new(GfxTarget::Gfx942)
        .expect("HIPRTC コンパイル・ロードに成功すること");
    let mut fails = 0;
    for (rows, cols) in SHAPES {
        let x = data(rows * cols, 3 + cols as u64, 4.0);
        let y = cpu_fwd(&x, rows, cols, true);
        let g = data(rows * cols, 5 + rows as u64, 1.0);
        let name = format!("log_softmax_backward {rows}x{cols}");
        fails += record(
            &name,
            &bw.run_log_softmax_backward_f32(&y, &g, rows, cols).unwrap(),
            &cpu_bwd(&y, &g, rows, cols),
        );
    }
    // 相殺入力と大きな上流勾配。
    let y = vec![-(4.0f32).ln(); 4];
    for g in [[1e20f32, -1e20, 1.0, 0.0], [2e38, 2e38, -2e38, 1.0]] {
        let got = bw.run_log_softmax_backward_f32(&y, &g, 1, 4).unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
        fails += record("log_softmax_backward cancel", &got, &cpu_bwd(&y, &g, 1, 4));
    }
    assert_eq!(fails, 0, "REQ-2 複合判定を外れた要素がある");
}
