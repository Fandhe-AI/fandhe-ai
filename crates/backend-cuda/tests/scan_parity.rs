//! イシュー #1740: `BackendOps::cumsum`／`cumprod`（`torch.cumsum`／
//! `torch.cumprod` 相当。親イシュー #1731）の CPU-CUDA 数値一致検証。
//!
//! `unique_parity.rs` と同じ構成方針を踏襲する: 環境適応スモーク
//! （属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `BackendError::CudaUnavailable` を確認して panic しないことのみ
//! 検証）と、実機必須の形状網羅（`#[ignore]`。DGX Spark GB10 等）を
//! 分離する。
//!
//! **契約は bit 同一**（lane ごとの `f64`／`double` アキュムレータ
//! 逐次計算・CPU 参照実装と bit 完全一致。`fandhe_ai_tensor_core::
//! BackendOps::cumsum`／`cumprod` doc 参照）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test scan_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// `cumsum`／`cumprod` の CPU-CUDA parity を bit 単位で確認する共通
/// ヘルパー。`shape`・`dim` を指定し、`f(&cpu, x, dim)` の形で
/// `BackendOps::cumsum`／`cumprod` を呼ぶクロージャを受け取る。
fn assert_scan_parity(
    cpu: &CpuBackendOps,
    cuda: &CudaBackendOps,
    x: &Tensor<f32>,
    dim: usize,
    f: impl Fn(
        &dyn BackendOps,
        &Tensor<f32>,
        usize,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::device::BackendError>,
    label: &str,
) {
    let cpu_out = f(cpu, x, dim).expect("cpu scan always succeeds");
    let cuda_out = f(cuda, x, dim).expect("cuda scan must succeed on CUDA-equipped test runner");

    assert_eq!(cuda_out.shape(), cpu_out.shape(), "{label}: shape 不一致");
    assert_eq!(bits(&cuda_out), bits(&cpu_out), "{label}: bit 不一致");

    // run-to-run 決定性。
    let cuda_out2 = f(cuda, x, dim).expect("cuda scan run2");
    assert_eq!(
        bits(&cuda_out2),
        bits(&cuda_out),
        "{label}: run-to-run で bit 同一のはず"
    );
}

fn cumsum_fn(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    dim: usize,
) -> Result<Tensor<f32>, fandhe_ai_tensor_core::device::BackendError> {
    ops.cumsum(x, dim)
}

fn cumprod_fn(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    dim: usize,
) -> Result<Tensor<f32>, fandhe_ai_tensor_core::device::BackendError> {
    ops.cumprod(x, dim)
}

#[test]
fn scan_parity_smoke_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("valid tensor");

    match cuda.cumsum(&x, 0) {
        Ok(_) => {
            // 1-D。
            assert_scan_parity(&cpu, &cuda, &x, 0, cumsum_fn, "cumsum 1-D");
            assert_scan_parity(&cpu, &cuda, &x, 0, cumprod_fn, "cumprod 1-D");

            // 2-D・各軸。
            let x2 = Tensor::new(Xorshift64Star::new(4001).fill_vec(24), &[4usize, 6usize])
                .expect("valid tensor");
            assert_scan_parity(&cpu, &cuda, &x2, 0, cumsum_fn, "cumsum 2-D dim0");
            assert_scan_parity(&cpu, &cuda, &x2, 1, cumsum_fn, "cumsum 2-D dim1");
            assert_scan_parity(&cpu, &cuda, &x2, 0, cumprod_fn, "cumprod 2-D dim0");
            assert_scan_parity(&cpu, &cuda, &x2, 1, cumprod_fn, "cumprod 2-D dim1");

            // 3-D・中間軸。
            let x3 = Tensor::new(
                Xorshift64Star::new(4002).fill_vec(60),
                &[3usize, 4usize, 5usize],
            )
            .expect("valid tensor");
            assert_scan_parity(&cpu, &cuda, &x3, 1, cumsum_fn, "cumsum 3-D dim1");
            assert_scan_parity(&cpu, &cuda, &x3, 1, cumprod_fn, "cumprod 3-D dim1");

            // axis_len=1（scan は恒等写像）。
            let x_axis1 =
                Tensor::new(vec![5.0, 6.0, 7.0], &[3usize, 1usize]).expect("valid tensor");
            assert_scan_parity(&cpu, &cuda, &x_axis1, 1, cumsum_fn, "cumsum axis_len=1");

            // transpose view 入力（非 contiguous）。
            let base = Tensor::new(Xorshift64Star::new(4003).fill_vec(12), &[3usize, 4usize])
                .expect("valid tensor");
            let transposed = base.transpose(0, 1).unwrap();
            assert_scan_parity(&cpu, &cuda, &transposed, 0, cumsum_fn, "cumsum transposed");
            assert_scan_parity(
                &cpu,
                &cuda,
                &transposed,
                1,
                cumprod_fn,
                "cumprod transposed",
            );

            // 契約証明ベクトル（本ファイル冒頭・Issue #1740 計画 §3.2）:
            // f64 アキュムレータでなければ成立しない値。
            let big = Tensor::new(vec![1e8, 1.0, -1e8], &[3]).expect("valid tensor");
            let cpu_big = cpu.cumsum(&big, 0).expect("cpu cumsum");
            let cuda_big = cuda.cumsum(&big, 0).expect("cuda cumsum");
            assert_eq!(bits(&cuda_big), bits(&cpu_big));
            let cuda_big_vals = cuda_big.contiguous().as_slice().unwrap().to_vec();
            assert_eq!(cuda_big_vals, vec![1e8, 1e8, 1.0]);

            let tiny = Tensor::new(vec![1e-30, 1e-30, 1e30], &[3]).expect("valid tensor");
            let cpu_tiny = cpu.cumprod(&tiny, 0).expect("cpu cumprod");
            let cuda_tiny = cuda.cumprod(&tiny, 0).expect("cuda cumprod");
            assert_eq!(bits(&cuda_tiny), bits(&cpu_tiny));
            let cuda_tiny_vals = cuda_tiny.contiguous().as_slice().unwrap().to_vec();
            assert_ne!(
                cuda_tiny_vals[2], 0.0,
                "f64 アキュムレータなら underflow しないはず"
            );

            // -0.0 先頭。
            let neg_zero = Tensor::new(vec![-0.0, 1.0], &[2]).expect("valid tensor");
            assert_scan_parity(&cpu, &cuda, &neg_zero, 0, cumsum_fn, "cumsum -0.0 leading");

            // inf + -inf → NaN、0 * inf → NaN（クラス一致で比較。
            // `soft_f64` 系と同じ NaN クラス一致方針だが、CUDA は
            // native `double` のため通常は CPU と bit も一致する。
            // ここでは NaN が発生する事実のみを検証する）。
            let inf_pair =
                Tensor::new(vec![f32::INFINITY, f32::NEG_INFINITY], &[2]).expect("valid tensor");
            let cuda_inf_sum = cuda.cumsum(&inf_pair, 0).expect("cuda cumsum");
            let cuda_inf_vals = cuda_inf_sum.contiguous().as_slice().unwrap().to_vec();
            assert!(cuda_inf_vals[1].is_nan(), "inf + -inf は NaN のはず");

            let zero_inf = Tensor::new(vec![0.0, f32::INFINITY], &[2]).expect("valid tensor");
            let cuda_zero_inf = cuda.cumprod(&zero_inf, 0).expect("cuda cumprod");
            let cuda_zero_inf_vals = cuda_zero_inf.contiguous().as_slice().unwrap().to_vec();
            assert!(cuda_zero_inf_vals[1].is_nan(), "0 * inf は NaN のはず");

            // 空出力。
            let empty = Tensor::new(Vec::new(), &[0usize, 3usize]).expect("valid tensor");
            let cpu_empty = cpu.cumsum(&empty, 0).expect("cpu cumsum(empty)");
            let cuda_empty = cuda.cumsum(&empty, 0).expect("cuda cumsum(empty)");
            assert_eq!(cpu_empty.shape(), cuda_empty.shape());
        }
        Err(fandhe_ai_tensor_core::device::BackendError::CudaUnavailable(_)) => {
            // CUDA 非搭載環境（通常 CI）。panic せず終了する。
        }
        Err(other) => panic!("unexpected error on CUDA-equipped runner: {other}"),
    }
}

/// 大きめ lane 数／axis 長を実機で確認する（DGX Spark GB10 等。
/// `#[ignore]`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn scan_matches_cpu_for_large_shapes() {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let x = Tensor::new(
        Xorshift64Star::new(5001).fill_vec(2000 * 300),
        &[2000usize, 300usize],
    )
    .expect("valid tensor");
    assert_scan_parity(&cpu, &cuda, &x, 0, cumsum_fn, "cumsum large dim0");
    assert_scan_parity(&cpu, &cuda, &x, 1, cumsum_fn, "cumsum large dim1");
    assert_scan_parity(&cpu, &cuda, &x, 1, cumprod_fn, "cumprod large dim1");

    let x1d = Tensor::new(Xorshift64Star::new(5002).fill_vec(1 << 16), &[1usize << 16])
        .expect("valid tensor");
    assert_scan_parity(&cpu, &cuda, &x1d, 0, cumsum_fn, "cumsum large 1-D");
}
