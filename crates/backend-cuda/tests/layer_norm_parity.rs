//! イシュー #1596: LayerNorm 順伝播カーネル（NVRTC・warp 内 reduction。
//! `kernels_layer_norm.rs` の単純な 1 CTA = 1 行構成）の CPU-CUDA 数値
//! 一致検証。
//!
//! `tests/rmsnorm_parity.rs` と同じ構成方針を踏襲する: 環境適応スモーク
//! （属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `fandhe_ai_backend_cuda::CudaError::DriverUnavailable`／
//! `NvrtcUnavailable` を確認して panic しないことのみ検証）と、実機必須
//! の形状網羅（`#[ignore]`。DGX Spark GB10 等）を分離する。判定式・
//! 許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の参照と
//! する（`.claude/rules/coding-rust.md`）。
//!
//! **本エージェント実行環境には CUDA 実機への到達手段がないため、
//! `#[ignore]` テストは未実測のまま記入欄を残す**（`docs/norm-ops-design.md`
//! 実機実測状況節）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test layer_norm_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaLayerNorm};

mod common;

/// テスト専用 CPU 参照実装（`f32::mul_add` を使用し、GPU 側 `fma` と
/// 丸め方針を揃える。`.claude/rules/coding-rust.md`）。
/// `out = (x-mean(x))*rsqrt(var(x)+eps)*w+b`（`w`／`b` が `None` の場合は
/// それぞれの演算をスキップ）。分散は biased ÷N。
fn cpu_layer_norm_reference(
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
    let inv_n = 1.0f32 / hidden as f32;
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mut sum = 0.0f32;
        for &v in row {
            sum += v;
        }
        let mean = sum * inv_n;
        let mut sq_acc = 0.0f32;
        for &v in row {
            let d = v - mean;
            sq_acc = d.mul_add(d, sq_acc);
        }
        let var = sq_acc * inv_n;
        let rstd = 1.0f32 / (var + eps).sqrt();
        let out_row = &mut out[r * hidden..(r + 1) * hidden];
        for i in 0..hidden {
            let mut xhat = (row[i] - mean) * rstd;
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

#[allow(clippy::too_many_arguments)]
fn assert_layer_norm_parity(
    layer_norm: &CudaLayerNorm,
    seed_x: u64,
    seed_w: u64,
    seed_b: u64,
    rows: usize,
    hidden: usize,
    with_weight: bool,
    with_bias: bool,
    eps: f32,
) {
    let x_data = Xorshift64Star::new(seed_x).fill_vec(rows * hidden);
    let w_data = if with_weight {
        Some(Xorshift64Star::new(seed_w).fill_vec(hidden))
    } else {
        None
    };
    let b_data = if with_bias {
        Some(Xorshift64Star::new(seed_b).fill_vec(hidden))
    } else {
        None
    };

    let gpu_out = layer_norm
        .run_layer_norm_f32(
            &x_data,
            w_data.as_deref(),
            b_data.as_deref(),
            eps,
            rows,
            hidden,
        )
        .expect("CudaLayerNorm::run_layer_norm_f32 must succeed on CUDA-equipped test runner");
    let expected = cpu_layer_norm_reference(
        &x_data,
        w_data.as_deref(),
        b_data.as_deref(),
        eps,
        rows,
        hidden,
    );

    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!(
            "layer_norm rows={rows} hidden={hidden} with_weight={with_weight} with_bias={with_bias} eps={eps}"
        ),
        &gpu_out,
        &expected,
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA／NVRTC 非搭載
/// 環境では既知の variant（`DriverUnavailable`／`NvrtcUnavailable`）を
/// 確認して panic しないことのみ検証し、実機なら小規模な parity まで
/// 実行する（`rmsnorm_parity.rs::rmsnorm_parity_smoke_env_adaptive` と
/// 同じ分岐パターン）。
#[test]
fn layer_norm_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaLayerNorm::new(&device) {
        Ok(layer_norm) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            assert_layer_norm_parity(&layer_norm, 901, 902, 903, 1, 8, false, false, 1e-5);
            assert_layer_norm_parity(&layer_norm, 904, 905, 906, 3, 128, true, true, 1e-5);
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {
            // NVRTC 非搭載環境（driver はあるが nvrtc が無い）。panic
            // しないことのみ確認する。
        }
        Err(other) => panic!("unexpected error variant for CudaLayerNorm::new: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。
///
/// hidden の網羅: 1（最小）・8・33（warp 幅 32 の非倍数）・128・1024・
/// 4097（大規模・非 4 の倍数）。rows の網羅: 1・3・17。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn layer_norm_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let layer_norm = CudaLayerNorm::new(&device)
        .expect("CudaLayerNorm::new must succeed on CUDA-equipped test runner");

    let hidden_cases: &[usize] = &[1, 8, 33, 128, 1024, 4097];
    let rows_cases: &[usize] = &[1, 3, 17];
    let mut seed = 5000u64;
    for &hidden in hidden_cases {
        for &rows in rows_cases {
            for with_weight in [false, true] {
                for with_bias in [false, true] {
                    seed += 1;
                    assert_layer_norm_parity(
                        &layer_norm,
                        seed,
                        seed + 500,
                        seed + 900,
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
}

/// 数値安定性: 極値入力で NaN/inf を出さない（実機必須）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn layer_norm_extreme_values_no_nan_inf() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let layer_norm = CudaLayerNorm::new(&device)
        .expect("CudaLayerNorm::new must succeed on CUDA-equipped test runner");

    let x = vec![1e30f32, -1e30, 1e30, -1e30, 1e-30, -1e-30, 0.0, 0.0];
    let out = layer_norm
        .run_layer_norm_f32(&x, None, None, 1e-5, 2, 4)
        .expect("run_layer_norm_f32 must succeed");
    for &v in &out {
        assert!(v.is_finite(), "expected finite layer_norm output, got {v}");
    }
}

/// CPU-CUDA 直接突合（実機必須）: `fandhe_ai_backend_cpu::layer_norm::
/// run_layer_norm_f32` を GPU 出力と直接比較する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn layer_norm_matches_backend_cpu_directly() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let layer_norm = CudaLayerNorm::new(&device)
        .expect("CudaLayerNorm::new must succeed on CUDA-equipped test runner");

    let rows = 3usize;
    let hidden = 4097usize;
    let eps = 1e-5f32;
    let x_data = Xorshift64Star::new(51_001).fill_vec(rows * hidden);
    let w_data = Xorshift64Star::new(51_002).fill_vec(hidden);
    let b_data = Xorshift64Star::new(51_003).fill_vec(hidden);

    let gpu_out = layer_norm
        .run_layer_norm_f32(&x_data, Some(&w_data), Some(&b_data), eps, rows, hidden)
        .expect("CudaLayerNorm::run_layer_norm_f32 must succeed on CUDA-equipped test runner");
    let cpu_out = fandhe_ai_backend_cpu::layer_norm::run_layer_norm_f32(
        &x_data,
        Some(&w_data),
        Some(&b_data),
        eps,
        rows,
        hidden,
    )
    .expect("fandhe_ai_backend_cpu::layer_norm::run_layer_norm_f32 must succeed");

    fandhe_ai_backend_cpu::parity::assert_parity(
        "layer_norm cpu(backend_cpu)-cuda direct parity",
        &gpu_out,
        &cpu_out,
    );
}

// --- BackendOps::layer_norm 独立エントリ ---

/// 環境適応スモーク（属性なし。通常 CI で実行）。
#[test]
fn backend_ops_layer_norm_smoke_env_adaptive() {
    use fandhe_ai_tensor_core::device::BackendError;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    let rows = 3usize;
    let hidden = 8usize;
    let x_data = Xorshift64Star::new(4001).fill_vec(rows * hidden);
    let x = Tensor::new(x_data.clone(), &[rows, hidden]).expect("valid tensor");

    let cuda = fandhe_ai_backend_cuda::CudaBackendOps::new(0);
    match cuda.layer_norm(&x, None, None, 1e-5) {
        Ok(via_ops) => {
            let expected = cpu_layer_norm_reference(&x_data, None, None, 1e-5, rows, hidden);
            assert_eq!(via_ops.shape(), &[rows, hidden]);
            fandhe_ai_backend_cpu::parity::assert_parity(
                "BackendOps::layer_norm vs cpu naive reference",
                via_ops.as_slice().expect("contiguous"),
                &expected,
            );
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::layer_norm: {other}"),
    }
}

/// `CudaBackendOps::layer_norm` を `fandhe_ai_backend_cpu::CpuBackendOps::
/// layer_norm` と実機で直接 `assert_parity` 突合する（形状網羅）。実機
/// 必須（`#[ignore]`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn backend_ops_layer_norm_matches_cpu_backend_ops_across_shapes() {
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    common::parity_baseline::assert_tolerance_constants_pinned();

    let cuda = fandhe_ai_backend_cuda::CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();

    let rows_cases: &[usize] = &[1, 3, 17];
    let hidden_cases: &[usize] = &[1, 31, 32, 33, 1024, 4097];
    let mut seed = 6000u64;
    for &rows in rows_cases {
        for &hidden in hidden_cases {
            seed += 1;
            let x_data = Xorshift64Star::new(seed).fill_vec(rows * hidden);
            let x = Tensor::new(x_data, &[rows, hidden]).expect("valid tensor");

            let gpu_out = cuda
                .layer_norm(&x, None, None, 1e-5)
                .expect("BackendOps::layer_norm must succeed on CUDA-equipped test runner");
            let cpu_out = cpu
                .layer_norm(&x, None, None, 1e-5)
                .expect("BackendOps::layer_norm must succeed on CPU");

            assert_eq!(gpu_out.shape(), &[rows, hidden]);
            fandhe_ai_backend_cpu::parity::assert_parity(
                &format!(
                    "BackendOps::layer_norm cpu-cuda direct parity rows={rows} hidden={hidden}"
                ),
                gpu_out.as_slice().expect("contiguous"),
                cpu_out.as_slice().expect("contiguous"),
            );
        }
    }
}
