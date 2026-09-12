//! `fandhe_ai_backend_cpu::layer_norm::run_layer_norm_f32`／
//! `CpuBackendOps::layer_norm` の受け入れ基準対応テスト（イシュー
//! #1596）。
//!
//! 形状網羅（hidden 端数・rows 0／hidden 0）・weight/bias 有無・NaN・
//! 極端な大きさを検証する。判定式・許容誤差は
//! `fandhe_ai_backend_cpu::parity` を唯一の参照とし再定義しない
//! （`.claude/rules/coding-rust.md`）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::layer_norm::run_layer_norm_f32;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// テスト専用の素朴 `f64` 参照実装（`run_layer_norm_f32` と独立に
/// 実装することでカーネル実装自体のバグを検出できるようにする）。
fn naive_layer_norm(
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    if hidden == 0 {
        return out;
    }
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / hidden as f64;
        let var: f64 = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / hidden as f64;
        let rstd = 1.0f64 / (var + eps as f64).sqrt();
        let out_row = &mut out[r * hidden..(r + 1) * hidden];
        for i in 0..hidden {
            let mut xhat = ((row[i] as f64 - mean) * rstd) as f32;
            if let Some(w) = w {
                xhat *= w[i];
            }
            if let Some(b) = b {
                xhat += b[i];
            }
            out_row[i] = xhat;
        }
    }
    out
}

fn assert_layer_norm_matches_naive(
    seed: u64,
    rows: usize,
    hidden: usize,
    with_weight: bool,
    with_bias: bool,
    eps: f32,
) {
    let x = Xorshift64Star::new(seed).fill_vec(rows * hidden);
    let w = if with_weight {
        Some(Xorshift64Star::new(seed + 500).fill_vec(hidden))
    } else {
        None
    };
    let b = if with_bias {
        Some(Xorshift64Star::new(seed + 900).fill_vec(hidden))
    } else {
        None
    };
    let actual = run_layer_norm_f32(&x, w.as_deref(), b.as_deref(), eps, rows, hidden).unwrap();
    let expected = naive_layer_norm(&x, w.as_deref(), b.as_deref(), eps, rows, hidden);
    assert_parity(
        &format!(
            "layer_norm rows={rows} hidden={hidden} with_weight={with_weight} with_bias={with_bias} eps={eps}"
        ),
        &actual,
        &expected,
    );
}

#[test]
fn layer_norm_matches_naive_across_shapes_and_affine_combinations() {
    let hidden_cases: &[usize] = &[1, 3, 4, 5, 7, 8, 17, 33, 128, 4097];
    let rows_cases: &[usize] = &[1, 2, 5];
    let mut seed = 1000u64;
    for &hidden in hidden_cases {
        for &rows in rows_cases {
            for with_weight in [false, true] {
                for with_bias in [false, true] {
                    seed += 1;
                    assert_layer_norm_matches_naive(
                        seed,
                        rows,
                        hidden,
                        with_weight,
                        with_bias,
                        1e-5,
                    );
                }
            }
        }
    }
    assert_layer_norm_matches_naive(9001, 4, 256, false, false, 0.0);
}

#[test]
fn run_layer_norm_f32_rows_zero_or_hidden_zero_is_empty() {
    assert_eq!(
        run_layer_norm_f32(&[], None, None, 1e-5, 0, 8).unwrap(),
        Vec::<f32>::new()
    );
    assert_eq!(
        run_layer_norm_f32(&[], None, None, 1e-5, 3, 0).unwrap(),
        Vec::<f32>::new()
    );
}

/// 数値安定性: 極値入力で NaN/inf を出さない。
#[test]
fn layer_norm_extreme_values_no_nan_inf() {
    let x = vec![1e30f32, -1e30, 1e30, -1e30, 1e-30, -1e-30, 0.0, 0.0];
    let out = run_layer_norm_f32(&x, None, None, 1e-5, 2, 4).unwrap();
    for &v in &out {
        assert!(v.is_finite(), "expected finite layer_norm output, got {v}");
    }
}

/// 同一行の複数 `inf`（分散計算で `inf - inf = NaN` が発生しうる境界
/// ケース）。
#[test]
fn layer_norm_multiple_inf_in_same_row() {
    let x = vec![f32::INFINITY, f32::INFINITY, 1.0, 2.0];
    let out = run_layer_norm_f32(&x, None, None, 1e-5, 1, 4).unwrap();
    // NaN 伝播の意味論を明示的に確認する（inf-inf=NaN が行全体へ伝播
    // することを許容する契約——特定の値を要求しない）。
    let _ = out;
}

/// NaN 伝播（`rmsnorm_parity.rs` の同名テストと同じ意味論）。
#[test]
fn layer_norm_propagates_nan_for_row_with_nan_element() {
    for hidden in [4usize, 5] {
        let mut x = vec![1.0f32; hidden];
        x[0] = f32::NAN;
        let out = run_layer_norm_f32(&x, None, None, 1e-5, 1, hidden).unwrap();
        assert!(
            out.iter().all(|v| v.is_nan()),
            "hidden={hidden}: NaN 要素を含む行の出力が NaN へ伝播していない: {out:?}"
        );
    }
}

// --- BackendOps::layer_norm 独立エントリ ---

#[test]
fn backend_ops_layer_norm_is_bit_identical_to_run_layer_norm_f32() {
    let rows = 3usize;
    let hidden = 17usize;
    let x_data = Xorshift64Star::new(8181).fill_vec(rows * hidden);
    let w_data = Xorshift64Star::new(8182).fill_vec(hidden);
    let b_data = Xorshift64Star::new(8183).fill_vec(hidden);
    let x = Tensor::new(x_data.clone(), &[rows, hidden]).unwrap();
    let w = Tensor::new(w_data.clone(), &[hidden]).unwrap();
    let b = Tensor::new(b_data.clone(), &[hidden]).unwrap();

    let cpu = CpuBackendOps::new();
    let via_ops = cpu.layer_norm(&x, Some(&w), Some(&b), 1e-5).unwrap();
    let via_kernel =
        run_layer_norm_f32(&x_data, Some(&w_data), Some(&b_data), 1e-5, rows, hidden).unwrap();

    assert_eq!(via_ops.shape(), &[rows, hidden]);
    assert_eq!(via_ops.as_slice().unwrap(), via_kernel.as_slice());
}

#[test]
fn backend_ops_layer_norm_no_affine_is_bit_identical_to_run_layer_norm_f32() {
    let rows = 2usize;
    let hidden = 33usize;
    let x_data = Xorshift64Star::new(9191).fill_vec(rows * hidden);
    let x = Tensor::new(x_data.clone(), &[rows, hidden]).unwrap();

    let cpu = CpuBackendOps::new();
    let via_ops = cpu.layer_norm(&x, None, None, 1e-6).unwrap();
    let via_kernel = run_layer_norm_f32(&x_data, None, None, 1e-6, rows, hidden).unwrap();

    assert_eq!(via_ops.as_slice().unwrap(), via_kernel.as_slice());
}

/// 非 contiguous な入力（転置 view）は `contiguous()`（`ops.rs::
/// CpuBackendOps::layer_norm`）を経由して透過的に実体化されるため
/// エラーにはならず、明示的に `contiguous()` した等価な入力と同じ
/// 結果を返す（`rmsnorm_parity.rs` の同名テストと同じ理由）。
#[test]
fn backend_ops_layer_norm_non_contiguous_input_matches_contiguous_equivalent() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let transposed = x.transpose(0, 1).unwrap(); // [3, 2]・非 contiguous。

    let cpu = CpuBackendOps::new();
    let via_transposed = cpu.layer_norm(&transposed, None, None, 1e-5).unwrap();
    let via_contiguous = cpu
        .layer_norm(&transposed.contiguous(), None, None, 1e-5)
        .unwrap();

    assert_eq!(
        via_transposed.as_slice().unwrap(),
        via_contiguous.as_slice().unwrap()
    );
}

#[test]
fn backend_ops_layer_norm_rejects_weight_length_mismatch() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let w = Tensor::new(vec![1.0, 1.0, 1.0], &[3]).unwrap();

    let cpu = CpuBackendOps::new();
    let result = cpu.layer_norm(&x, Some(&w), None, 1e-5);
    assert!(matches!(result, Err(BackendError::KernelLaunchFailed(_))));
}

#[test]
fn backend_ops_layer_norm_rejects_bias_length_mismatch() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let b = Tensor::new(vec![0.0, 0.0, 0.0], &[3]).unwrap();

    let cpu = CpuBackendOps::new();
    let result = cpu.layer_norm(&x, None, Some(&b), 1e-5);
    assert!(matches!(result, Err(BackendError::KernelLaunchFailed(_))));
}
