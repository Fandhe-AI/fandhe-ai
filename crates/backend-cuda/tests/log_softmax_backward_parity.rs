//! イシュー #1949: `log_softmax` backward カーネル（1 warp = 1 行・
//! `Σ_dim(g)` の `double` butterfly reduction）の CPU-CUDA 数値一致
//! 検証。
//!
//! `softmax_parity.rs`（#594・#1594）と同じ構成方針を踏襲する:
//! 環境適応スモーク（属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `CudaError::DriverUnavailable`／`NvrtcUnavailable`
//! （`BackendOps` 経由は `BackendError::CudaUnavailable`）を確認して
//! panic しないことのみ検証）と、実機必須の形状網羅（`#[ignore]`。
//! DGX Spark GB10 等）を分離する。判定式・許容誤差は再定義せず
//! `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2 統一複合判定。
//! `crates/tensor-core/src/backend_ops.rs::BackendOps::
//! log_softmax_backward` doc の「bit 完全一致は要求しない」契約どおり）
//! を唯一の参照とする（`.claude/rules/coding-rust.md`）。
//!
//! CPU 参照実装は本ファイル内のテスト専用関数（`dx = g − exp(y)·Σ_dim(g)`。
//! `Σ_dim` は `dim` 添字昇順の `f64` 逐次和。`grad::log_softmax_vjp_along`
//! と同じ式・同じ結合順序）である。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test log_softmax_backward_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaLogSoftmaxBackward};

mod common;

/// `dx = g − exp(y)·Σ_dim(g)` の独立参照実装（`dim` はテンソルの最終軸
/// 限定。`Σ_dim` は添字昇順の `f64` 逐次和。`grad::
/// log_softmax_vjp_along` と同じ式・同じ結合順序をテストローカルに
/// 複製する）。
fn cpu_log_softmax_backward_reference(y: &[f32], g: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut dx = vec![0.0f32; y.len()];
    if cols == 0 {
        return dx;
    }
    for r in 0..rows {
        let y_row = &y[r * cols..(r + 1) * cols];
        let g_row = &g[r * cols..(r + 1) * cols];
        let sum: f64 = g_row.iter().map(|&v| v as f64).sum();
        let dx_row = &mut dx[r * cols..(r + 1) * cols];
        for i in 0..cols {
            dx_row[i] = (g_row[i] as f64 - y_row[i].exp() as f64 * sum) as f32;
        }
    }
    dx
}

fn assert_log_softmax_backward_parity(
    kernel: &CudaLogSoftmaxBackward,
    seed: u64,
    rows: usize,
    cols: usize,
) {
    // `y` は log_softmax 出力の値域（負の実数）を模した入力にする
    // （必ずしも真の log_softmax 出力である必要はない。カーネルは
    // `y`／`g` から `dx` を計算する純粋な式のため）。
    let y_data: Vec<f32> = Xorshift64Star::new(seed)
        .fill_vec(rows * cols)
        .iter()
        .map(|&v| -(v.abs()) - 1e-3) // 常に負・0 を避ける。
        .collect();
    let g_data = Xorshift64Star::new(seed.wrapping_add(1)).fill_vec(rows * cols);

    let gpu_out = kernel
        .run_log_softmax_backward_f32(&y_data, &g_data, rows, cols)
        .expect("CudaLogSoftmaxBackward::run_log_softmax_backward_f32 must succeed on CUDA-equipped test runner");
    let cpu_out = cpu_log_softmax_backward_reference(&y_data, &g_data, rows, cols);

    assert_eq!(gpu_out.len(), cpu_out.len());
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("log_softmax_backward cpu-cuda parity rows={rows} cols={cols}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。`softmax_parity.rs::
/// softmax_parity_smoke_env_adaptive` と同じ分岐パターン。
#[test]
fn log_softmax_backward_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaLogSoftmaxBackward::new(&device) {
        Ok(kernel) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            assert_log_softmax_backward_parity(&kernel, 901, 1, 8);
            assert_log_softmax_backward_parity(&kernel, 903, 3, 1024);
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {
            // NVRTC 非搭載環境。panic しないことのみ確認する。
        }
        Err(other) => panic!("unexpected error variant for CudaLogSoftmaxBackward::new: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`cols` の網羅: 1（行長 1。
/// 縮約は自明）・8・31／32／33（warp 境界の両側）・1024（1 warp 複数
/// 反復）・4097（NEON 端要素相当。GPU 側は境界検査のみ影響）。`rows` の
/// 網羅: 1・3・33（grid をまたぐ行数）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn log_softmax_backward_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let kernel = CudaLogSoftmaxBackward::new(&device)
        .expect("log_softmax_backward kernel compile must succeed");

    let cols_cases: &[usize] = &[1, 8, 31, 32, 33, 1024, 4097];
    let rows_cases: &[usize] = &[1, 3, 33];

    let mut seed = 9000u64;
    for &cols in cols_cases {
        for &rows in rows_cases {
            seed += 2;
            assert_log_softmax_backward_parity(&kernel, seed, rows, cols);
        }
    }
}

/// `rows == 0 || cols == 0` は空結果を返す（driver に触れない早期
/// return。`layer_norm.rs::run_layer_norm_f32` と同じ 0 要素契約）。
/// driver 非依存のため実機不要。
#[test]
fn log_softmax_backward_empty_shape_returns_empty_without_driver() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    let kernel = match CudaLogSoftmaxBackward::new(&device) {
        Ok(kernel) => kernel,
        Err(CudaError::NvrtcUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaLogSoftmaxBackward::new: {other}"),
    };
    assert_eq!(
        kernel
            .run_log_softmax_backward_f32(&[], &[], 0, 4)
            .expect("rows=0 must succeed"),
        Vec::<f32>::new()
    );
    assert_eq!(
        kernel
            .run_log_softmax_backward_f32(&[], &[], 4, 0)
            .expect("cols=0 must succeed"),
        Vec::<f32>::new()
    );
}

/// run-to-run bit 同一性（決定的カーネル。butterfly reduction 自体は
/// bit 完全一致を主張しないが、同一入力に対しては毎回同じ bit
/// パターンを返す契約）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn log_softmax_backward_is_run_to_run_bit_identical() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let kernel = CudaLogSoftmaxBackward::new(&device)
        .expect("log_softmax_backward kernel compile must succeed");

    let rows = 5usize;
    let cols = 1024usize;
    let y_data: Vec<f32> = Xorshift64Star::new(1234)
        .fill_vec(rows * cols)
        .iter()
        .map(|&v| -(v.abs()) - 1e-3)
        .collect();
    let g_data = Xorshift64Star::new(5678).fill_vec(rows * cols);

    let first = kernel
        .run_log_softmax_backward_f32(&y_data, &g_data, rows, cols)
        .expect("run 1 must succeed");
    let second = kernel
        .run_log_softmax_backward_f32(&y_data, &g_data, rows, cols)
        .expect("run 2 must succeed");
    assert_eq!(
        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "run-to-run bit 同一性が崩れている"
    );
}

// --- BackendOps::log_softmax_backward 独立エントリ ---

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA 非搭載環境では
/// `BackendError::CudaUnavailable` を確認して panic しないことのみ
/// 検証する。
#[test]
fn backend_ops_log_softmax_backward_smoke_env_adaptive() {
    use fandhe_ai_tensor_core::device::BackendError;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    let rows = 3usize;
    let cols = 8usize;
    let y_data: Vec<f32> = Xorshift64Star::new(4101)
        .fill_vec(rows * cols)
        .iter()
        .map(|&v| -(v.abs()) - 1e-3)
        .collect();
    let g_data = Xorshift64Star::new(4102).fill_vec(rows * cols);
    let y = Tensor::new(y_data.clone(), &[rows, cols]).expect("valid tensor");
    let g = Tensor::new(g_data.clone(), &[rows, cols]).expect("valid tensor");

    let cuda = fandhe_ai_backend_cuda::CudaBackendOps::new(0);
    match cuda.log_softmax_backward(&y, &g, 1) {
        Ok(dx) => {
            let expected = cpu_log_softmax_backward_reference(&y_data, &g_data, rows, cols);
            assert_eq!(dx.shape(), &[rows, cols]);
            fandhe_ai_backend_cpu::parity::assert_parity(
                "BackendOps::log_softmax_backward vs cpu naive reference",
                dx.as_slice().expect("contiguous"),
                &expected,
            );
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => {
            panic!("unexpected error variant for CudaBackendOps::log_softmax_backward: {other}")
        }
    }
}

/// 非最終軸は `Unsupported`（`row_softmax_layout` がドライバ非依存で
/// 先に軸検査を行うため、実機なしでも検証できる。`softmax_parity.rs::
/// backend_ops_softmax_non_final_axis_is_unsupported` と同型）。
#[test]
fn backend_ops_log_softmax_backward_non_final_axis_is_unsupported() {
    use fandhe_ai_tensor_core::device::BackendError;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    let y = Tensor::new(vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], &[2, 3]).expect("valid tensor");
    let g = Tensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]).expect("valid tensor");
    let cuda = fandhe_ai_backend_cuda::CudaBackendOps::new(0);
    let result = cuda.log_softmax_backward(&y, &g, 0);
    assert!(matches!(result, Err(BackendError::Unsupported(_))));
}

/// `out`／`upstream` の shape 不一致は driver 非依存で拒否される
/// （fail-closed。`row_softmax_layout` 検査の前段で完結する経路）。
#[test]
fn backend_ops_log_softmax_backward_rejects_shape_mismatch_without_driver() {
    use fandhe_ai_tensor_core::device::BackendError;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    let y = Tensor::new(vec![-1.0, -2.0, -3.0, -4.0], &[2, 2]).expect("valid tensor");
    let g = Tensor::new(vec![0.1, 0.2, 0.3], &[3]).expect("valid tensor");
    let cuda = fandhe_ai_backend_cuda::CudaBackendOps::new(0);
    let result = cuda.log_softmax_backward(&y, &g, 1);
    assert!(matches!(result, Err(BackendError::ShapeMismatch(_))));
}

/// `CudaBackendOps::log_softmax_backward` を CPU 独立参照実装と実機で
/// 直接 `assert_parity` 突合する（形状網羅）。実機必須（`#[ignore]`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn backend_ops_log_softmax_backward_matches_cpu_reference_across_shapes() {
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    common::parity_baseline::assert_tolerance_constants_pinned();

    let cuda = fandhe_ai_backend_cuda::CudaBackendOps::new(0);

    let rows_cases: &[usize] = &[1, 3, 17];
    let cols_cases: &[usize] = &[1, 31, 32, 33, 1024, 4097];
    let mut seed = 6000u64;
    for &rows in rows_cases {
        for &cols in cols_cases {
            seed += 2;
            let y_data: Vec<f32> = Xorshift64Star::new(seed)
                .fill_vec(rows * cols)
                .iter()
                .map(|&v| -(v.abs()) - 1e-3)
                .collect();
            let g_data = Xorshift64Star::new(seed + 1).fill_vec(rows * cols);
            let y = Tensor::new(y_data.clone(), &[rows, cols]).expect("valid tensor");
            let g = Tensor::new(g_data.clone(), &[rows, cols]).expect("valid tensor");

            let gpu_out = cuda.log_softmax_backward(&y, &g, 1).expect(
                "BackendOps::log_softmax_backward must succeed on CUDA-equipped test runner",
            );
            let cpu_out = cpu_log_softmax_backward_reference(&y_data, &g_data, rows, cols);

            assert_eq!(gpu_out.shape(), &[rows, cols]);
            fandhe_ai_backend_cpu::parity::assert_parity(
                &format!(
                    "BackendOps::log_softmax_backward cpu-cuda direct parity rows={rows} cols={cols}"
                ),
                gpu_out.as_slice().expect("contiguous"),
                &cpu_out,
            );
        }
    }
}
