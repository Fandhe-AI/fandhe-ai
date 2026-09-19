//! `TypedOps<f64>` の CUDA 統合テスト（イシュー #2060・親 #1650）。
//!
//! (a) 実機非依存（GPU・CUDA driver 不要・CI 常時実行）は
//!     `typed_ops_f64_contract.rs` が担う（accessor 契約・shape 検証）。
//!
//! (b) `#[ignore]` 実機依存（GB10 実機。`typed_ops_f16_parity.rs` と
//!     同型の `real_device_or_skip` env-adaptive パターン）: CPU
//!     参照実装（`crates/backend-cpu/src/typed_f64.rs`）との数値契約を
//!     `kernels_typed_f64.rs` モジュール doc の分類（bit 完全一致／
//!     REQ-2 統一複合判定）どおりに検証する。
//!     - `gemm`：`matmul_reference_fma_f64`（累積順序が
//!       `p` 昇順で GPU カーネルと同一構造）と bit 完全一致
//!     - `add`／`mul`／`relu`：順序非依存の純粋算術のため bit 完全一致
//!     - `exp`／`tanh`：デバイス側 libm の丸め差のため REQ-2 複合判定
//!     - 軸指定 `sum`／`max`：昇順逐次累積が CPU と同一構造のため
//!       bit 完全一致
//!     - 全軸縮約 `sum`／`max`：GPU 2 段木縮約 対 CPU `CHUNK` 分割の
//!       構造差のため REQ-2 複合判定

use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity_f64, matmul_reference_fma_f64};
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice, CudaError};
use fandhe_ai_tensor_core::{Tensor, TypedOps};

fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::new(data.to_vec(), shape).unwrap()
}

/// `#[ignore]` テスト共通: CUDA driver 不在環境では早期 return する
/// （`typed_ops_f16_parity.rs::real_device_or_skip` と同型）。
fn real_device_or_skip() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(dev) => Some(dev),
        Err(CudaError::DriverUnavailable { .. }) | Err(CudaError::Driver(_)) => None,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    }
}

/// `TypedOps<f64>::gemm` が `matmul_reference_fma_f64`
/// （`f64::mul_add` を `p` 昇順で累積する参照実装）と bit 完全一致する
/// ことを確認する（`kernels_typed_f64.rs::GEMM_NAIVE_F64` の
/// `fma(double,double,double)` が同一の累積順序を持つため）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn typed_f64_gemm_is_bit_identical_to_reference_fma() {
    let Some(_device) = real_device_or_skip() else {
        return;
    };
    let ops = CudaBackendOps::new(0);

    for (m, n, k) in [(2usize, 2usize, 2usize), (7, 5, 11), (32, 32, 64)] {
        let a: Vec<f64> = (0..(m * k)).map(|i| (i % 13) as f64 * 0.1 - 0.5).collect();
        let b: Vec<f64> = (0..(k * n)).map(|i| (i % 11) as f64 * 0.2 - 0.7).collect();
        let a_t = t(&a, &[m, k]);
        let b_t = t(&b, &[k, n]);

        let gpu = TypedOps::<f64>::gemm(&ops, &a_t, &b_t)
            .expect("TypedOps::<f64>::gemm succeeds on real hardware");

        let mut reference = vec![0.0f64; m * n];
        matmul_reference_fma_f64(&a, &b, &mut reference, m, n, k)
            .expect("matmul_reference_fma_f64 shape validation must pass");

        assert_eq!(
            gpu.host_slice().as_ref(),
            reference.as_slice(),
            "TypedOps<f64>::gemm must be bit-identical to matmul_reference_fma_f64 for \
             m={m} n={n} k={k}"
        );
    }
}

/// `add`／`mul`／`relu`（bit 完全一致目標）・`exp`／`tanh`（REQ-2 複合
/// 判定目標）・軸指定 `sum`／`max`（bit 完全一致目標）・全軸縮約
/// `sum`／`max`（REQ-2 複合判定目標）を CPU `TypedOps<f64>`
/// （`backend-cpu::typed_f64`）と突き合わせる。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn typed_f64_elementwise_and_reduction_match_cpu_reference() {
    let Some(_device) = real_device_or_skip() else {
        return;
    };
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();

    // [2,3] + [3]（行方向ブロードキャスト）。
    let a = vec![0.5f64, -0.25, 0.75, -0.5, 0.25, -0.75];
    let b = vec![0.1f64, 0.2, 0.3];
    let a_t = t(&a, &[2, 3]);
    let b_t = t(&b, &[3]);

    // add / mul（bit 完全一致。broadcast 込み）。
    let cuda_add = TypedOps::<f64>::add(&cuda, &a_t, &b_t).expect("cuda add succeeds");
    let cpu_add = TypedOps::<f64>::add(&cpu, &a_t, &b_t).expect("cpu add succeeds");
    assert_eq!(
        cuda_add.host_slice().as_ref(),
        cpu_add.host_slice().as_ref(),
        "typed f64 add must be bit-identical to cpu reference"
    );

    let cuda_mul = TypedOps::<f64>::mul(&cuda, &a_t, &b_t).expect("cuda mul succeeds");
    let cpu_mul = TypedOps::<f64>::mul(&cpu, &a_t, &b_t).expect("cpu mul succeeds");
    assert_eq!(
        cuda_mul.host_slice().as_ref(),
        cpu_mul.host_slice().as_ref(),
        "typed f64 mul must be bit-identical to cpu reference"
    );

    // relu（bit 完全一致。NaN 混入時の非伝播規約も併せて確認する）。
    let a_with_nan = t(&[1.0, -2.0, f64::NAN, 0.0, 3.5, -0.001], &[2, 3]);
    let cuda_relu = TypedOps::<f64>::relu(&cuda, &a_with_nan).expect("cuda relu succeeds");
    let cpu_relu = TypedOps::<f64>::relu(&cpu, &a_with_nan).expect("cpu relu succeeds");
    assert_eq!(
        cuda_relu.host_slice().as_ref(),
        cpu_relu.host_slice().as_ref(),
        "typed f64 relu must be bit-identical to cpu reference (including NaN handling)"
    );

    // exp / tanh（REQ-2 複合判定。デバイス側 libm の丸め差を許容する）。
    let cuda_exp = TypedOps::<f64>::exp(&cuda, &a_t).expect("cuda exp succeeds");
    let cpu_exp = TypedOps::<f64>::exp(&cpu, &a_t).expect("cpu exp succeeds");
    assert_parity_f64(
        "typed f64 exp vs cpu",
        cuda_exp.host_slice().as_ref(),
        cpu_exp.host_slice().as_ref(),
    );

    let cuda_tanh = TypedOps::<f64>::tanh(&cuda, &a_t).expect("cuda tanh succeeds");
    let cpu_tanh = TypedOps::<f64>::tanh(&cpu, &a_t).expect("cpu tanh succeeds");
    assert_parity_f64(
        "typed f64 tanh vs cpu",
        cuda_tanh.host_slice().as_ref(),
        cpu_tanh.host_slice().as_ref(),
    );

    // sum / max 軸指定（bit 完全一致。CPU 側は `axis_reduce_f64` の
    // 昇順逐次累積で GPU の 1 スレッド 1 出力要素の昇順ループと一致する）。
    let cuda_sum_axis = TypedOps::<f64>::sum(&cuda, &a_t, Some(0)).expect("cuda sum axis succeeds");
    let cpu_sum_axis = TypedOps::<f64>::sum(&cpu, &a_t, Some(0)).expect("cpu sum axis succeeds");
    assert_eq!(
        cuda_sum_axis.host_slice().as_ref(),
        cpu_sum_axis.host_slice().as_ref(),
        "typed f64 sum(Some(0)) must be bit-identical to cpu reference"
    );

    let cuda_max_axis = TypedOps::<f64>::max(&cuda, &a_t, Some(0)).expect("cuda max axis succeeds");
    let cpu_max_axis = TypedOps::<f64>::max(&cpu, &a_t, Some(0)).expect("cpu max axis succeeds");
    assert_eq!(
        cuda_max_axis.host_slice().as_ref(),
        cpu_max_axis.host_slice().as_ref(),
        "typed f64 max(Some(0)) must be bit-identical to cpu reference"
    );

    // sum / max 全軸縮約（REQ-2 複合判定。GPU 2 段木縮約と CPU CHUNK
    // 分割の構造差のため bit 完全一致は目標としない）。
    let cuda_sum_all = TypedOps::<f64>::sum(&cuda, &a_t, None).expect("cuda sum all succeeds");
    let cpu_sum_all = TypedOps::<f64>::sum(&cpu, &a_t, None).expect("cpu sum all succeeds");
    assert_parity_f64(
        "typed f64 sum(None) vs cpu",
        cuda_sum_all.host_slice().as_ref(),
        cpu_sum_all.host_slice().as_ref(),
    );

    let cuda_max_all = TypedOps::<f64>::max(&cuda, &a_t, None).expect("cuda max all succeeds");
    let cpu_max_all = TypedOps::<f64>::max(&cpu, &a_t, None).expect("cpu max all succeeds");
    assert_parity_f64(
        "typed f64 max(None) vs cpu",
        cuda_max_all.host_slice().as_ref(),
        cpu_max_all.host_slice().as_ref(),
    );
}

/// 空縮約・境界形状（`numel == 0`／`k == 0`）が CPU 参照実装と同一の
/// 意味論を持つことを確認する（`kernels_typed_f64.rs` モジュール doc・
/// `typed_f64.rs` 各 `run_*` の doc コメント参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn typed_f64_empty_and_boundary_shapes_match_cpu_semantics() {
    let Some(_device) = real_device_or_skip() else {
        return;
    };
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();

    // k == 0: 全 0 出力。
    let a = Tensor::new(Vec::<f64>::new(), &[2, 0]).unwrap();
    let b = Tensor::new(Vec::<f64>::new(), &[0, 3]).unwrap();
    let gpu_c = TypedOps::<f64>::gemm(&cuda, &a, &b).expect("gemm with k=0 succeeds");
    let cpu_c = TypedOps::<f64>::gemm(&cpu, &a, &b).expect("cpu gemm with k=0 succeeds");
    assert_eq!(gpu_c.host_slice().as_ref(), cpu_c.host_slice().as_ref());

    // sum(None) の空縮約は 0.0。
    let empty = Tensor::new(Vec::<f64>::new(), &[0, 3]).unwrap();
    let gpu_sum = TypedOps::<f64>::sum(&cuda, &empty, None).expect("sum over empty succeeds");
    assert_eq!(gpu_sum.host_slice().as_ref(), &[0.0f64]);

    // max(None) の空縮約はエラー。
    let err = TypedOps::<f64>::max(&cuda, &empty, None).unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::device::BackendError::KernelLaunchFailed(_)
    ));
}
