//! イシュー #1735: BatchNorm1d／2d 順伝播カーネル（NVRTC・1 warp = 1
//! channel。`kernels_batch_norm.rs` の train／infer 2 カーネル構成）
//! の CPU-CUDA 数値一致検証。
//!
//! `tests/layer_norm_parity.rs` と同じ構成方針を踏襲する: 環境適応
//! スモーク（属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `fandhe_ai_backend_cuda::CudaError::DriverUnavailable`／
//! `NvrtcUnavailable` を確認して panic しないことのみ検証）と、実機
//! 必須の形状網羅（`#[ignore]`。DGX Spark GB10 等）を分離する。判定
//! 式・許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の
//! 参照とする（`.claude/rules/coding-rust.md`）。
//!
//! **本エージェント実行環境には CUDA 実機への到達手段がないため、
//! `#[ignore]` テストは未実測のまま記入欄を残す**（`docs/batch-norm-
//! ops-design.md` §9「実装記録」）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test batch_norm_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaBatchNorm, CudaDevice, CudaError};

mod common;

/// テスト専用 CPU 参照実装（`crates/backend-cpu/tests/batch_norm_
/// parity.rs::naive_batch_norm_train` と独立に実装することでカーネル
/// 実装自体のバグを検出できるようにする。正規化統計は `f64` で計算し
/// GPU 側カーネル〈`kernels_batch_norm.rs` の `double` アキュムレータ・
/// `fma`〉と揃える）。`out = (x-mean(x))*rsqrt(var(x)+eps)*w+b`（`w`／
/// `b` が `None` の場合はそれぞれの演算をスキップ）。分散は biased
/// ÷M。
#[allow(clippy::too_many_arguments)]
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
fn naive_batch_norm_infer(
    x: &[f32],
    mean: &[f32],
    var: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    n: usize,
    c: usize,
    spatial: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    if n == 0 || c == 0 || spatial == 0 {
        return out;
    }
    for ch in 0..c {
        let mean_c = mean[ch] as f64;
        let rstd = 1.0f64 / (var[ch] as f64 + eps as f64).sqrt();
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
    out
}

#[allow(clippy::too_many_arguments)]
fn assert_batch_norm_train_parity(
    batch_norm: &CudaBatchNorm,
    seed_x: u64,
    seed_w: u64,
    seed_b: u64,
    n: usize,
    c: usize,
    spatial: usize,
    with_weight: bool,
    with_bias: bool,
    eps: f32,
) {
    let numel = n * c * spatial;
    let x_data = Xorshift64Star::new(seed_x).fill_vec(numel);
    let w_data = if with_weight {
        Some(Xorshift64Star::new(seed_w).fill_vec(c))
    } else {
        None
    };
    let b_data = if with_bias {
        Some(Xorshift64Star::new(seed_b).fill_vec(c))
    } else {
        None
    };

    let (gpu_out, gpu_mean, gpu_var) = batch_norm
        .run_batch_norm_train_f32(
            &x_data,
            w_data.as_deref(),
            b_data.as_deref(),
            eps,
            n,
            c,
            spatial,
        )
        .expect(
            "CudaBatchNorm::run_batch_norm_train_f32 must succeed on CUDA-equipped test runner",
        );
    let (expected_out, expected_mean, expected_var) = naive_batch_norm_train(
        &x_data,
        w_data.as_deref(),
        b_data.as_deref(),
        eps,
        n,
        c,
        spatial,
    );

    let label = format!(
        "batch_norm_train n={n} c={c} spatial={spatial} with_weight={with_weight} with_bias={with_bias} eps={eps}"
    );
    fandhe_ai_backend_cpu::parity::assert_parity(&format!("{label} out"), &gpu_out, &expected_out);
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("{label} mean"),
        &gpu_mean,
        &expected_mean,
    );
    fandhe_ai_backend_cpu::parity::assert_parity(&format!("{label} var"), &gpu_var, &expected_var);
}

#[allow(clippy::too_many_arguments)]
fn assert_batch_norm_infer_parity(
    batch_norm: &CudaBatchNorm,
    seed_x: u64,
    seed_mean: u64,
    seed_var: u64,
    seed_w: u64,
    seed_b: u64,
    n: usize,
    c: usize,
    spatial: usize,
    with_weight: bool,
    with_bias: bool,
    eps: f32,
) {
    let numel = n * c * spatial;
    let x_data = Xorshift64Star::new(seed_x).fill_vec(numel);
    let mean_data = Xorshift64Star::new(seed_mean).fill_vec(c);
    let var_data: Vec<f32> = Xorshift64Star::new(seed_var)
        .fill_vec(c)
        .into_iter()
        .map(|v| v.abs() + 0.1)
        .collect();
    let w_data = if with_weight {
        Some(Xorshift64Star::new(seed_w).fill_vec(c))
    } else {
        None
    };
    let b_data = if with_bias {
        Some(Xorshift64Star::new(seed_b).fill_vec(c))
    } else {
        None
    };

    let gpu_out = batch_norm
        .run_batch_norm_infer_f32(
            &x_data,
            &mean_data,
            &var_data,
            w_data.as_deref(),
            b_data.as_deref(),
            eps,
            n,
            c,
            spatial,
        )
        .expect(
            "CudaBatchNorm::run_batch_norm_infer_f32 must succeed on CUDA-equipped test runner",
        );
    let expected = naive_batch_norm_infer(
        &x_data,
        &mean_data,
        &var_data,
        w_data.as_deref(),
        b_data.as_deref(),
        eps,
        n,
        c,
        spatial,
    );

    let label = format!(
        "batch_norm_infer n={n} c={c} spatial={spatial} with_weight={with_weight} with_bias={with_bias} eps={eps}"
    );
    fandhe_ai_backend_cpu::parity::assert_parity(&label, &gpu_out, &expected);
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA／NVRTC 非搭載
/// 環境では既知の variant（`DriverUnavailable`／`NvrtcUnavailable`）を
/// 確認して panic しないことのみ検証し、実機なら小規模な parity まで
/// 実行する。
#[test]
fn batch_norm_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaBatchNorm::new(&device) {
        Ok(batch_norm) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            assert_batch_norm_train_parity(
                &batch_norm,
                8001,
                8002,
                8003,
                2,
                3,
                1,
                false,
                false,
                1e-5,
            );
            assert_batch_norm_train_parity(
                &batch_norm,
                8004,
                8005,
                8006,
                4,
                3,
                5,
                true,
                true,
                1e-5,
            );
            assert_batch_norm_infer_parity(
                &batch_norm,
                8007,
                8008,
                8009,
                8010,
                8011,
                2,
                3,
                1,
                true,
                true,
                1e-5,
            );
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {
            // NVRTC 非搭載環境（driver はあるが nvrtc が無い）。panic
            // しないことのみ確認する。
        }
        Err(other) => panic!("unexpected error variant for CudaBatchNorm::new: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`(n, c, spatial)` 組は
/// rank 2 相当（spatial=1）・rank 3（1d 空間入力）・rank 4（2d。H*W
/// が spatial）を横断し、warp 幅 32 の端数（33・非倍数）を含む。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn batch_norm_train_matches_naive_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let batch_norm = CudaBatchNorm::new(&device)
        .expect("CudaBatchNorm::new must succeed on CUDA-equipped test runner");

    let shape_cases: &[(usize, usize, usize)] = &[
        (2, 1, 1),
        (5, 3, 1),
        (4, 7, 5),
        (3, 4, 1),
        (2, 5, 17),
        (8, 2, 1),
        (1, 33, 64),
        (3, 32, 4097),
        (17, 1, 1),
    ];
    let mut seed = 9000u64;
    for &(n, c, spatial) in shape_cases {
        for with_weight in [false, true] {
            for with_bias in [false, true] {
                seed += 1;
                assert_batch_norm_train_parity(
                    &batch_norm,
                    seed,
                    seed + 500,
                    seed + 900,
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
    // eps=0.0 の 1 件。
    assert_batch_norm_train_parity(&batch_norm, 9500, 9501, 9502, 4, 3, 5, true, true, 0.0);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn batch_norm_infer_matches_naive_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let batch_norm = CudaBatchNorm::new(&device)
        .expect("CudaBatchNorm::new must succeed on CUDA-equipped test runner");

    let shape_cases: &[(usize, usize, usize)] = &[
        (2, 1, 1),
        (5, 3, 1),
        (4, 7, 5),
        (3, 4, 1),
        (2, 5, 17),
        (8, 2, 1),
        (1, 33, 64),
        (3, 32, 4097),
        (17, 1, 1),
    ];
    let mut seed = 10000u64;
    for &(n, c, spatial) in shape_cases {
        for with_weight in [false, true] {
            for with_bias in [false, true] {
                seed += 1;
                assert_batch_norm_infer_parity(
                    &batch_norm,
                    seed,
                    seed + 300,
                    seed + 600,
                    seed + 900,
                    seed + 1200,
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
}

/// 数値安定性: 極値入力で NaN/inf を出さない（実機必須）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn batch_norm_train_extreme_values_no_nan_inf() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let batch_norm = CudaBatchNorm::new(&device)
        .expect("CudaBatchNorm::new must succeed on CUDA-equipped test runner");

    let x = vec![1e30f32, -1e30, 1e30, -1e30, 1e-30, -1e-30, 0.0, 0.0];
    let (out, mean, var) = batch_norm
        .run_batch_norm_train_f32(&x, None, None, 1e-5, 2, 1, 4)
        .expect("run_batch_norm_train_f32 must succeed");
    for &v in out.iter().chain(mean.iter()).chain(var.iter()) {
        assert!(v.is_finite(), "expected finite batch_norm output, got {v}");
    }
}

/// 極端なスケール（`2e20`）でも overflow しない（`.claude/rules/
/// coding-rust.md` の正規化統計 `f64` 二乗和契約）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn batch_norm_train_extreme_scale_does_not_overflow() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let batch_norm = CudaBatchNorm::new(&device)
        .expect("CudaBatchNorm::new must succeed on CUDA-equipped test runner");

    let x = vec![2e20f32, -2e20, 2e20, -2e20];
    let (out, _mean, var) = batch_norm
        .run_batch_norm_train_f32(&x, None, None, 1e-5, 1, 2, 2)
        .expect("run_batch_norm_train_f32 must succeed");
    for &v in out.iter().chain(var.iter()) {
        assert!(v.is_finite(), "expected finite batch_norm output, got {v}");
    }
}

/// NaN はそれが属するチャネルのみへ伝播する（他チャネルは無汚染。
/// `channel_index` によるチャネル分離の直接検証）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn batch_norm_train_propagates_nan_only_for_channel_with_nan() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let batch_norm = CudaBatchNorm::new(&device)
        .expect("CudaBatchNorm::new must succeed on CUDA-equipped test runner");

    // n=1,c=2,spatial=2: ch0=[NaN,1.0]・ch1=[1.0,1.0]。
    let x = vec![f32::NAN, 1.0, 1.0, 1.0];
    let (out, mean, _var) = batch_norm
        .run_batch_norm_train_f32(&x, None, None, 1e-5, 1, 2, 2)
        .expect("run_batch_norm_train_f32 must succeed");
    assert!(out[0].is_nan() && out[1].is_nan());
    assert!(mean[0].is_nan());
    assert!(!out[2].is_nan() && !out[3].is_nan());
    assert!(!mean[1].is_nan());
}

/// CPU-CUDA 直接突合（実機必須）: `fandhe_ai_backend_cpu::batch_norm::
/// run_batch_norm_train_f32` を GPU 出力と直接比較する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn batch_norm_train_matches_backend_cpu_directly() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let batch_norm = CudaBatchNorm::new(&device)
        .expect("CudaBatchNorm::new must succeed on CUDA-equipped test runner");

    let n = 3usize;
    let c = 5usize;
    let spatial = 17usize;
    let eps = 1e-5f32;
    let x_data = Xorshift64Star::new(51_101).fill_vec(n * c * spatial);
    let w_data = Xorshift64Star::new(51_102).fill_vec(c);
    let b_data = Xorshift64Star::new(51_103).fill_vec(c);

    let (gpu_out, gpu_mean, gpu_var) = batch_norm
        .run_batch_norm_train_f32(&x_data, Some(&w_data), Some(&b_data), eps, n, c, spatial)
        .expect(
            "CudaBatchNorm::run_batch_norm_train_f32 must succeed on CUDA-equipped test runner",
        );
    let cpu_raw = fandhe_ai_backend_cpu::run_batch_norm_train_f32(
        &x_data,
        Some(&w_data),
        Some(&b_data),
        eps,
        n,
        c,
        spatial,
    )
    .expect("fandhe_ai_backend_cpu::run_batch_norm_train_f32 must succeed");

    fandhe_ai_backend_cpu::parity::assert_parity(
        "batch_norm_train cpu(backend_cpu)-cuda direct parity out",
        &gpu_out,
        &cpu_raw.out,
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        "batch_norm_train cpu(backend_cpu)-cuda direct parity mean",
        &gpu_mean,
        &cpu_raw.mean,
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        "batch_norm_train cpu(backend_cpu)-cuda direct parity var",
        &gpu_var,
        &cpu_raw.var,
    );
}

/// 同一入力の 2 回連続実行が bit 同一であること（決定性は契約。
/// `.claude/rules/coding-rust.md` の run-to-run bit 同一契約）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn batch_norm_train_is_run_to_run_deterministic() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let batch_norm = CudaBatchNorm::new(&device)
        .expect("CudaBatchNorm::new must succeed on CUDA-equipped test runner");

    let x_data = Xorshift64Star::new(52_001).fill_vec(3 * 5 * 17);
    let w_data = Xorshift64Star::new(52_002).fill_vec(5);

    let (out1, mean1, var1) = batch_norm
        .run_batch_norm_train_f32(&x_data, Some(&w_data), None, 1e-5, 3, 5, 17)
        .expect("run_batch_norm_train_f32 must succeed (run 1)");
    let (out2, mean2, var2) = batch_norm
        .run_batch_norm_train_f32(&x_data, Some(&w_data), None, 1e-5, 3, 5, 17)
        .expect("run_batch_norm_train_f32 must succeed (run 2)");

    assert_eq!(out1, out2, "run-to-run output must be bit identical");
    assert_eq!(mean1, mean2, "run-to-run mean must be bit identical");
    assert_eq!(var1, var2, "run-to-run var must be bit identical");
}

// --- BackendOps::batch_norm_train／batch_norm_infer 独立エントリ ---

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA 非搭載環境では
/// `BackendError::CudaUnavailable` を確認して panic しないことのみ
/// 検証する。
#[test]
fn backend_ops_batch_norm_smoke_env_adaptive() {
    use fandhe_ai_tensor_core::device::BackendError;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    let n = 2usize;
    let c = 3usize;
    let x_data = Xorshift64Star::new(4101).fill_vec(n * c);
    let x = Tensor::new(x_data, &[n, c]).expect("valid tensor");

    let cuda = fandhe_ai_backend_cuda::CudaBackendOps::new(0);
    match cuda.batch_norm_train(&x, None, None, 1e-5) {
        Ok(via_ops) => {
            assert_eq!(via_ops.output.shape(), &[n, c]);
            assert_eq!(via_ops.batch_mean.shape(), &[c]);
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => {
            panic!("unexpected error variant for CudaBackendOps::batch_norm_train: {other}")
        }
    }
}

/// `CudaBackendOps::batch_norm_train`／`batch_norm_infer` を
/// `fandhe_ai_backend_cpu::CpuBackendOps` と実機で直接 `assert_parity`
/// 突合する（形状網羅。rank 2／3／4）。実機必須（`#[ignore]`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn backend_ops_batch_norm_matches_cpu_backend_ops_across_shapes() {
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    common::parity_baseline::assert_tolerance_constants_pinned();

    let cuda = fandhe_ai_backend_cuda::CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();

    // (n, c, spatial): rank 2・rank 3・rank 4 を横断。
    let cases: &[(usize, usize, usize, &[usize])] = &[
        (4, 3, 1, &[4, 3]),
        (2, 5, 6, &[2, 5, 6]),
        (2, 3, 12, &[2, 3, 4, 3]),
    ];
    let mut seed = 11000u64;
    for &(n, c, spatial, shape) in cases {
        seed += 1;
        let x_data = Xorshift64Star::new(seed).fill_vec(n * c * spatial);
        let w_data = Xorshift64Star::new(seed + 500).fill_vec(c);
        let b_data = Xorshift64Star::new(seed + 900).fill_vec(c);
        let x = Tensor::new(x_data, shape).expect("valid tensor");
        let w = Tensor::new(w_data, &[c]).expect("valid tensor");
        let b = Tensor::new(b_data, &[c]).expect("valid tensor");

        let gpu_train = cuda
            .batch_norm_train(&x, Some(&w), Some(&b), 1e-5)
            .expect("BackendOps::batch_norm_train must succeed on CUDA-equipped test runner");
        let cpu_train = cpu
            .batch_norm_train(&x, Some(&w), Some(&b), 1e-5)
            .expect("BackendOps::batch_norm_train must succeed on CPU");

        assert_eq!(gpu_train.output.shape(), shape);
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("BackendOps::batch_norm_train cpu-cuda direct parity out shape={shape:?}"),
            gpu_train.output.as_slice().expect("contiguous"),
            cpu_train.output.as_slice().expect("contiguous"),
        );
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("BackendOps::batch_norm_train cpu-cuda direct parity mean shape={shape:?}"),
            gpu_train.batch_mean.as_slice().expect("contiguous"),
            cpu_train.batch_mean.as_slice().expect("contiguous"),
        );

        let gpu_infer = cuda
            .batch_norm_infer(
                &x,
                &gpu_train.batch_mean,
                &gpu_train.batch_var,
                Some(&w),
                Some(&b),
                1e-5,
            )
            .expect("BackendOps::batch_norm_infer must succeed on CUDA-equipped test runner");
        let cpu_infer = cpu
            .batch_norm_infer(
                &x,
                &cpu_train.batch_mean,
                &cpu_train.batch_var,
                Some(&w),
                Some(&b),
                1e-5,
            )
            .expect("BackendOps::batch_norm_infer must succeed on CPU");
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("BackendOps::batch_norm_infer cpu-cuda direct parity shape={shape:?}"),
            gpu_infer.as_slice().expect("contiguous"),
            cpu_infer.as_slice().expect("contiguous"),
        );
    }
}

/// 非連続入力（転置 view）でも contiguous 化後の結果と一致する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn backend_ops_batch_norm_train_non_contiguous_input_matches_contiguous_equivalent() {
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    let cuda = fandhe_ai_backend_cuda::CudaBackendOps::new(0);

    // [c, n] を転置した [n, c] view（非連続）と、同じ値の連続 [n, c]
    // テンソルを突合する。
    let c = 3usize;
    let n = 4usize;
    let data = Xorshift64Star::new(12001).fill_vec(n * c);
    let base = Tensor::new(data.clone(), &[c, n]).expect("valid tensor");
    let transposed = base.transpose(0, 1).expect("transpose must succeed");

    let mut contiguous_data = vec![0.0f32; n * c];
    for row in 0..n {
        for col in 0..c {
            contiguous_data[row * c + col] = data[col * n + row];
        }
    }
    let contiguous = Tensor::new(contiguous_data, &[n, c]).expect("valid tensor");

    let out_transposed = cuda.batch_norm_train(&transposed, None, None, 1e-5).expect(
        "BackendOps::batch_norm_train must succeed on CUDA-equipped test runner (transposed)",
    );
    let out_contiguous = cuda.batch_norm_train(&contiguous, None, None, 1e-5).expect(
        "BackendOps::batch_norm_train must succeed on CUDA-equipped test runner (contiguous)",
    );

    fandhe_ai_backend_cpu::parity::assert_parity(
        "batch_norm_train non-contiguous vs contiguous equivalent",
        out_transposed.output.as_slice().expect("contiguous"),
        out_contiguous.output.as_slice().expect("contiguous"),
    );
}

/// 長さ不一致（weight）は `BackendError::KernelLaunchFailed` を返す
/// （判定迂回経路を作らない。`map_batch_norm_error` の
/// `InvalidBatchNormShape` 分岐）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn backend_ops_batch_norm_rejects_length_mismatch() {
    use fandhe_ai_tensor_core::device::BackendError;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    let cuda = fandhe_ai_backend_cuda::CudaBackendOps::new(0);
    let x = Tensor::new(vec![1.0f32; 8], &[4, 2]).expect("valid tensor");
    let w = Tensor::new(vec![1.0f32; 3], &[3]).expect("valid tensor: mismatched length");

    let err = cuda
        .batch_norm_train(&x, Some(&w), None, 1e-5)
        .expect_err("length mismatch must be rejected");
    assert!(matches!(err, BackendError::KernelLaunchFailed(_)));
}
