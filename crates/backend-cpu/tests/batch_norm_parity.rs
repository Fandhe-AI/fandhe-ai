//! `fandhe_ai_backend_cpu::batch_norm::run_batch_norm_train_f32`／
//! `run_batch_norm_infer_f32`／`CpuBackendOps::batch_norm_train`／
//! `batch_norm_infer` の受け入れ基準対応テスト（イシュー #1732・親
//! #1608）。
//!
//! 形状網羅（rank 2／3／4・N/C/spatial 端数・並列閾値跨ぎ）・
//! weight/bias 有無・NaN・極端な大きさを検証する。判定式・許容誤差は
//! `fandhe_ai_backend_cpu::parity` を唯一の参照とし再定義しない
//! （`.claude/rules/coding-rust.md`）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_cpu::{run_batch_norm_infer_f32, run_batch_norm_train_f32};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// テスト専用の素朴 `f64` 参照実装（`run_batch_norm_train_f32` と
/// 独立に実装することでカーネル実装自体のバグを検出できるようにする。
/// `layer_norm_parity.rs::naive_layer_norm` と同じ位置付け）。
fn naive_batch_norm_train(
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    n: usize,
    c: usize,
    spatial: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; x.len()];
    let mut mean = vec![0.0f32; c];
    let mut var = vec![0.0f32; c];
    if n == 0 || c == 0 || spatial == 0 {
        return (out, mean, var);
    }
    let m = (n * spatial) as f64;
    for ch in 0..c {
        let mut vals = Vec::with_capacity(n * spatial);
        for batch in 0..n {
            for sp in 0..spatial {
                vals.push(x[batch * (c * spatial) + ch * spatial + sp] as f64);
            }
        }
        let mean_c: f64 = vals.iter().sum::<f64>() / m;
        let var_c: f64 = vals.iter().map(|v| (v - mean_c).powi(2)).sum::<f64>() / m;
        mean[ch] = mean_c as f32;
        var[ch] = var_c as f32;
        let rstd = 1.0f64 / (var_c + eps as f64).sqrt();
        let wv = w.map_or(1.0f32, |w| w[ch]);
        let bv = b.map_or(0.0f32, |b| b[ch]);
        for batch in 0..n {
            for sp in 0..spatial {
                let idx = batch * (c * spatial) + ch * spatial + sp;
                let xhat = ((x[idx] as f64 - mean_c) * rstd) as f32;
                out[idx] = xhat.mul_add(wv, bv);
            }
        }
    }
    (out, mean, var)
}

#[allow(clippy::too_many_arguments)]
fn assert_batch_norm_train_matches_naive(
    seed: u64,
    n: usize,
    c: usize,
    spatial: usize,
    with_weight: bool,
    with_bias: bool,
    eps: f32,
) {
    let numel = n * c * spatial;
    let x = Xorshift64Star::new(seed).fill_vec(numel);
    let w = if with_weight {
        Some(Xorshift64Star::new(seed + 500).fill_vec(c))
    } else {
        None
    };
    let b = if with_bias {
        Some(Xorshift64Star::new(seed + 900).fill_vec(c))
    } else {
        None
    };
    let actual =
        run_batch_norm_train_f32(&x, w.as_deref(), b.as_deref(), eps, n, c, spatial).unwrap();
    let (expected_out, expected_mean, expected_var) =
        naive_batch_norm_train(&x, w.as_deref(), b.as_deref(), eps, n, c, spatial);
    let label = format!(
        "batch_norm_train n={n} c={c} spatial={spatial} with_weight={with_weight} \
         with_bias={with_bias} eps={eps}"
    );
    assert_parity(&format!("{label} out"), &actual.out, &expected_out);
    assert_parity(&format!("{label} mean"), &actual.mean, &expected_mean);
    assert_parity(&format!("{label} var"), &actual.var, &expected_var);
}

#[test]
fn batch_norm_train_matches_naive_across_shapes_and_affine_combinations() {
    // (n, c, spatial) 組: rank 2 相当（spatial=1）・rank 3（1d 空間入力）・
    // rank 4（2d。H*W が spatial）を網羅する。
    let shape_cases: &[(usize, usize, usize)] = &[
        (2, 1, 1),
        (5, 3, 1),
        (4, 7, 5),
        (3, 4, 1),
        (2, 5, 17),
        (8, 2, 1),
    ];
    let mut seed = 2000u64;
    for &(n, c, spatial) in shape_cases {
        for with_weight in [false, true] {
            for with_bias in [false, true] {
                seed += 1;
                assert_batch_norm_train_matches_naive(
                    seed,
                    n,
                    c,
                    spatial,
                    with_weight,
                    with_bias,
                    1e-5,
                );
            }
        }
    }
    assert_batch_norm_train_matches_naive(9002, 4, 3, 8, false, false, 0.0);
}

/// rayon 並列閾値（`PARALLEL_THRESHOLD`。チャネル方向並列）を跨ぐ
/// 大きめの形状での正しさを確認する。
#[test]
fn batch_norm_train_matches_naive_across_parallel_threshold() {
    assert_batch_norm_train_matches_naive(3001, 4, 8, 1 << 11, true, true, 1e-5);
}

#[test]
fn run_batch_norm_train_f32_n_zero_or_c_zero_or_spatial_zero_is_empty() {
    let raw = run_batch_norm_train_f32(&[], None, None, 1e-5, 0, 3, 4).unwrap();
    assert_eq!(raw.out, Vec::<f32>::new());
    assert_eq!(raw.mean, vec![0.0f32; 3]);
    assert_eq!(raw.var, vec![0.0f32; 3]);

    let raw = run_batch_norm_train_f32(&[], None, None, 1e-5, 2, 0, 4).unwrap();
    assert_eq!(raw.out, Vec::<f32>::new());
    assert!(raw.mean.is_empty());
}

/// 数値安定性: 極値入力で NaN/inf を出さない。
#[test]
fn batch_norm_train_extreme_values_no_nan_inf() {
    let x = vec![1e30f32, -1e30, 1e30, -1e30, 1e-30, -1e-30, 0.0, 0.0];
    let raw = run_batch_norm_train_f32(&x, None, None, 1e-5, 2, 2, 2).unwrap();
    for &v in &raw.out {
        assert!(v.is_finite(), "expected finite batch_norm output, got {v}");
    }
}

/// NaN 伝播: NaN を含むチャネルのみ NaN になり、他チャネルは無関係の
/// まま（`eval::batch_norm_train_channels` の同名テストと同じ意味論）。
#[test]
fn batch_norm_train_propagates_nan_for_channel_with_nan_element() {
    let x = vec![f32::NAN, 1.0, 1.0, 1.0];
    let raw = run_batch_norm_train_f32(&x, None, None, 1e-5, 1, 2, 2).unwrap();
    assert!(raw.out[0].is_nan() && raw.out[1].is_nan());
    assert!(!raw.out[2].is_nan() && !raw.out[3].is_nan());
}

// --- BackendOps::batch_norm_train／batch_norm_infer 独立エントリ ---

#[test]
fn backend_ops_batch_norm_train_is_bit_identical_to_run_batch_norm_train_f32() {
    let n = 3usize;
    let c = 5usize;
    let spatial = 4usize;
    let numel = n * c * spatial;
    let x_data = Xorshift64Star::new(8281).fill_vec(numel);
    let w_data = Xorshift64Star::new(8282).fill_vec(c);
    let b_data = Xorshift64Star::new(8283).fill_vec(c);
    let x = Tensor::new(x_data.clone(), &[n, c, spatial]).unwrap();
    let w = Tensor::new(w_data.clone(), &[c]).unwrap();
    let b = Tensor::new(b_data.clone(), &[c]).unwrap();

    let cpu = CpuBackendOps::new();
    let via_ops = cpu.batch_norm_train(&x, Some(&w), Some(&b), 1e-5).unwrap();
    let via_kernel =
        run_batch_norm_train_f32(&x_data, Some(&w_data), Some(&b_data), 1e-5, n, c, spatial)
            .unwrap();

    assert_eq!(via_ops.output.shape(), &[n, c, spatial]);
    assert_eq!(
        via_ops.output.as_slice().unwrap(),
        via_kernel.out.as_slice()
    );
    assert_eq!(
        via_ops.batch_mean.as_slice().unwrap(),
        via_kernel.mean.as_slice()
    );
    assert_eq!(
        via_ops.batch_var.as_slice().unwrap(),
        via_kernel.var.as_slice()
    );
}

#[test]
fn backend_ops_batch_norm_infer_is_bit_identical_to_run_batch_norm_infer_f32() {
    let n = 2usize;
    let c = 3usize;
    let spatial = 5usize;
    let numel = n * c * spatial;
    let x_data = Xorshift64Star::new(9281).fill_vec(numel);
    let mean_data = Xorshift64Star::new(9282).fill_vec(c);
    let var_data: Vec<f32> = Xorshift64Star::new(9283)
        .fill_vec(c)
        .into_iter()
        .map(|v| v.abs() + 0.1)
        .collect();
    let x = Tensor::new(x_data.clone(), &[n, c, spatial]).unwrap();
    let mean = Tensor::new(mean_data.clone(), &[c]).unwrap();
    let var = Tensor::new(var_data.clone(), &[c]).unwrap();

    let cpu = CpuBackendOps::new();
    let via_ops = cpu
        .batch_norm_infer(&x, &mean, &var, None, None, 1e-5)
        .unwrap();
    let via_kernel = run_batch_norm_infer_f32(
        &x_data, &mean_data, &var_data, None, None, 1e-5, n, c, spatial,
    )
    .unwrap();

    assert_eq!(via_ops.as_slice().unwrap(), via_kernel.as_slice());
}

/// 非 contiguous な入力（転置 view）は `contiguous()`（`ops.rs::
/// CpuBackendOps::batch_norm_train`）を経由して透過的に実体化される
/// ため、エラーにはならず、明示的に `contiguous()` した等価な入力と
/// 同じ結果を返す（`layer_norm_parity.rs` の同名テストと同じ理由）。
#[test]
fn backend_ops_batch_norm_train_non_contiguous_input_matches_contiguous_equivalent() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let transposed = x.transpose(0, 1).unwrap(); // [3, 2]・非 contiguous。

    let cpu = CpuBackendOps::new();
    let via_transposed = cpu.batch_norm_train(&transposed, None, None, 1e-5).unwrap();
    let via_contiguous = cpu
        .batch_norm_train(&transposed.contiguous(), None, None, 1e-5)
        .unwrap();

    assert_eq!(
        via_transposed.output.as_slice().unwrap(),
        via_contiguous.output.as_slice().unwrap()
    );
}

#[test]
fn backend_ops_batch_norm_train_rejects_weight_length_mismatch() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let w = Tensor::new(vec![1.0, 1.0, 1.0], &[3]).unwrap();

    let cpu = CpuBackendOps::new();
    let result = cpu.batch_norm_train(&x, Some(&w), None, 1e-5);
    assert!(matches!(result, Err(BackendError::KernelLaunchFailed(_))));
}

#[test]
fn backend_ops_batch_norm_infer_rejects_stats_length_mismatch() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let mean = Tensor::new(vec![0.0, 0.0, 0.0], &[3]).unwrap();
    let var = Tensor::new(vec![1.0, 1.0], &[2]).unwrap();

    let cpu = CpuBackendOps::new();
    let result = cpu.batch_norm_infer(&x, &mean, &var, None, None, 1e-5);
    assert!(matches!(result, Err(BackendError::KernelLaunchFailed(_))));
}
