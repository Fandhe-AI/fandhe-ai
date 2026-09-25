//! イシュー #2155: `log_softmax` forward カーネル（NVRTC・1 パス／
//! 2 パス。`kernels_softmax.rs::LOG_SOFTMAX_F32_ONEPASS`／
//! `LOG_SOFTMAX_F32_TWOPASS`）の CPU-CUDA 数値一致検証。
//!
//! `softmax_parity.rs`（#594）と同じ構成方針を踏襲する: 環境適応
//! スモーク（属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `fandhe_ai_backend_cuda::CudaError::DriverUnavailable`／
//! `NvrtcUnavailable` を確認して panic しないことのみ検証）と、実機
//! 必須の形状網羅・極値入力（`#[ignore]`。DGX Spark GB10 等）を分離
//! する。判定式・許容誤差は再定義せず `fandhe_ai_backend_cpu::parity`
//! を唯一の参照とする（`.claude/rules/coding-rust.md`）。
//!
//! CPU 参照実装は `fandhe_ai_backend_cpu::softmax::run_log_softmax_f32`
//! （NEON/rayon 参照実装）を直接使う（`softmax_parity.rs::
//! softmax_matches_backend_cpu_directly` と同じ「参照実装クレートを
//! 直接呼ぶ」方式。テスト専用の素朴 CPU 実装は用意しない——`log_softmax`
//! は既に CPU 側に融合カーネルがあるため二重実装を避ける）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test log_softmax_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaSoftmax};

mod common;

fn assert_log_softmax_parity(softmax: &CudaSoftmax, seed: u64, rows: usize, cols: usize) {
    let x_data = Xorshift64Star::new(seed).fill_vec(rows * cols);

    let gpu_out = softmax
        .run_log_softmax_f32(&x_data, rows, cols)
        .expect("CudaSoftmax::run_log_softmax_f32 must succeed on CUDA-equipped test runner");
    let cpu_out = fandhe_ai_backend_cpu::softmax::run_log_softmax_f32(&x_data, rows, cols)
        .expect("cpu log_softmax reference must succeed");

    assert_eq!(gpu_out.len(), cpu_out.len());
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("log_softmax cpu-cuda parity rows={rows} cols={cols}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。`softmax_parity.rs::
/// softmax_parity_smoke_env_adaptive` と同じ分岐パターン。
#[test]
fn log_softmax_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaSoftmax::new(&device) {
        Ok(softmax) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            assert_log_softmax_parity(&softmax, 22_701, 1, 8);
            assert_log_softmax_parity(&softmax, 22_703, 3, 1024);
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {}
        Err(other) => panic!("unexpected error variant for CudaSoftmax::new: {other}"),
    }
}

/// 実機必須の形状網羅（`softmax_matches_cpu_across_shapes` と同じ
/// cols／rows 網羅。1 パス／2 パス経路・vec4 端要素・非倍数 cols を
/// 含む）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn log_softmax_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let softmax = CudaSoftmax::new(&device).expect("softmax kernel compile must succeed");

    let cols_cases: &[usize] = &[1, 8, 1024, 4096, 4097, 8192, 16384];
    let rows_cases: &[usize] = &[1, 3, 33];

    let mut seed = 22_000u64;
    for &cols in cols_cases {
        for &rows in rows_cases {
            seed += 1;
            assert_log_softmax_parity(&softmax, seed, rows, cols);
        }
    }
}

/// 極値・非有限行の入力（実装計画 §2.2「境界の入力」）: 全要素
/// `-inf`・`+inf` 混在・NaN 混在・`±f32::MAX` 混在の各行が CPU 参照
/// 実装と NaN クラス一致することを検証する（非有限入力は数学的に
/// 未定義のため REQ-2 統一複合判定ではなく NaN クラス一致で判定する。
/// `softmax_numerically_stable_for_extreme_inputs` と同じ「実機必須の
/// 数値安定性検証」の役割）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn log_softmax_extreme_and_nonfinite_rows() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let softmax = CudaSoftmax::new(&device).expect("softmax kernel compile must succeed");

    let cols = 8usize;
    let rows_data: &[Vec<f32>] = &[
        vec![f32::NEG_INFINITY; cols],
        {
            let mut row = vec![1.0f32; cols];
            row[0] = f32::INFINITY;
            row
        },
        {
            let mut row = vec![1.0f32; cols];
            row[3] = f32::NAN;
            row
        },
        {
            let mut row = vec![0.0f32; cols];
            row[0] = f32::MAX;
            row[1] = -f32::MAX;
            row
        },
    ];
    let mut x_data = Vec::with_capacity(rows_data.len() * cols);
    for row in rows_data {
        x_data.extend_from_slice(row);
    }
    let rows = rows_data.len();

    let gpu_out = softmax
        .run_log_softmax_f32(&x_data, rows, cols)
        .expect("log_softmax must not error on non-finite rows");
    let cpu_out = fandhe_ai_backend_cpu::softmax::run_log_softmax_f32(&x_data, rows, cols)
        .expect("cpu log_softmax reference must succeed on non-finite rows");

    for r in 0..rows {
        let g_row = &gpu_out[r * cols..(r + 1) * cols];
        let c_row = &cpu_out[r * cols..(r + 1) * cols];
        for (i, (&g, &c)) in g_row.iter().zip(c_row.iter()).enumerate() {
            assert_eq!(
                g.is_nan(),
                c.is_nan(),
                "row={r} idx={i}: NaN クラス不一致（cuda={g:?}, cpu={c:?}）"
            );
            if !g.is_nan() {
                assert_eq!(
                    g, c,
                    "row={r} idx={i}: 非有限値の一致が崩れている（cuda={g:?}, cpu={c:?}）"
                );
            }
        }
    }
}

/// run-to-run の bit 同一性（決定性）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn log_softmax_run_to_run_is_bit_identical() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let softmax = CudaSoftmax::new(&device).expect("softmax kernel compile must succeed");

    let rows = 5usize;
    let cols = 16384usize; // 2 パス経路を強制。
    let x_data = Xorshift64Star::new(22_500).fill_vec(rows * cols);

    let out1 = softmax
        .run_log_softmax_f32(&x_data, rows, cols)
        .expect("log_softmax run1 must succeed");
    let out2 = softmax
        .run_log_softmax_f32(&x_data, rows, cols)
        .expect("log_softmax run2 must succeed");

    assert_eq!(
        out1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        out2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "run-to-run で bit 同一のはず"
    );
}
