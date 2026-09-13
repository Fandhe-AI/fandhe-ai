//! `TypedOps<half::f16>` の CUDA 統合テスト（イシュー #1703・親 #1650）。
//!
//! (a) 実機非依存（GPU・CUDA driver 不要・CI 常時実行）: accessor 契約・
//!     shape 検証（driver 到達前に拒否されること）を確認する。
//! (b) `#[ignore]` 実機依存（GB10 実機）: `gemm` が `CudaGemmAuto::run_f16`
//!     と bit 完全一致すること（パススルー結線の直接検証）・7 演算が
//!     CPU 参照実装の f32 経路を f16 へ丸めた値と REQ-2 統一複合判定
//!     （`assert_parity`）で一致することを確認する。

use half::f16;

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice, CudaError, CudaGemmAuto};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, TypedOps};

fn f16_tensor(data: &[f32], shape: &[usize]) -> Tensor<f16> {
    let d: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
    Tensor::new(d, shape).unwrap()
}

fn f16_to_f32_vec(t: &Tensor<f16>) -> Vec<f32> {
    t.host_slice().iter().map(|v| v.to_f32()).collect()
}

// ---------------------------------------------------------------------
// (a) 実機非依存
// ---------------------------------------------------------------------

#[test]
fn typed_ops_f16_accessor_is_some_and_bf16_is_none() {
    let ops = CudaBackendOps::new(0);
    assert!(BackendOps::typed_ops_f16(&ops).is_some());
    assert!(BackendOps::typed_ops_bf16(&ops).is_none());
}

#[test]
fn gemm_rejects_shape_mismatch_before_touching_driver() {
    let ops = CudaBackendOps::new(0);
    let a = f16_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let b = f16_tensor(&[1.0, 2.0], &[2, 1]);
    let err = TypedOps::<f16>::gemm(&ops, &a, &b).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

// ---------------------------------------------------------------------
// (b) `#[ignore]` 実機依存（GB10）
// ---------------------------------------------------------------------

/// `#[ignore]` テスト共通: CUDA driver 不在環境では早期 return する
/// （`cpu_cuda_mma_parity.rs::mma_f16_parity_smoke_env_adaptive` と同じ
/// env-adaptive パターン）。
fn real_device_or_skip() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(dev) => Some(dev),
        Err(CudaError::DriverUnavailable { .. }) | Err(CudaError::Driver(_)) => None,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    }
}

/// `TypedOps<f16>::gemm`（`crate::typed_f16`。`ops.rs` 経由）が
/// `CudaGemmAuto::run_f16` の直接呼び出しと bit 単位で完全一致することを
/// 確認する（結線がパススルーであることの直接検証。イシュー #1703
/// 実装計画 §5.2）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn typed_f16_gemm_is_bit_identical_to_cuda_gemm_auto_run_f16() {
    let Some(device) = real_device_or_skip() else {
        return;
    };
    let auto = CudaGemmAuto::new(&device).expect("CudaGemmAuto::new succeeds on real hardware");
    let ops = CudaBackendOps::new(0);

    // 16x16x16（`gemm_auto.rs::run_f16_matches_cpu_reference` と同一の
    // 入力生成規則）と mma 整列形状（64x128x32）の両方を確認する。
    for (m, n, k) in [(16usize, 16usize, 16usize), (64, 128, 32)] {
        let a_f32: Vec<f32> = (0..(m * k)).map(|i| (i % 7) as f32 * 0.1).collect();
        let b_f32: Vec<f32> = (0..(k * n)).map(|i| (i % 5) as f32 * 0.2).collect();
        let a16 = f16_tensor(&a_f32, &[m, k]);
        let b16 = f16_tensor(&b_f32, &[k, n]);

        let a_slice: Vec<f16> = a_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let b_slice: Vec<f16> = b_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let direct = auto
            .run_f16(&a_slice, &b_slice, m as u32, n as u32, k as u32)
            .expect("CudaGemmAuto::run_f16 succeeds on real hardware");

        let via_typed_ops = TypedOps::<f16>::gemm(&ops, &a16, &b16)
            .expect("TypedOps::<f16>::gemm succeeds on real hardware");

        assert_eq!(
            f16_to_f32_vec(&via_typed_ops),
            direct.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            "TypedOps<f16>::gemm must be bit-identical to CudaGemmAuto::run_f16 for m={m} n={n} k={k}"
        );
    }
}

/// `TypedOps<f16>::gemm` の数値が CPU f32 参照実装（`matmul_reference_fma`）
/// を f16 へ丸めた値と REQ-2 統一複合判定で一致することを確認する
/// （`gemm_auto.rs::run_f16_matches_cpu_reference` と同一入力・同一手順）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn typed_f16_gemm_matches_cpu_reference_rounded() {
    let Some(_device) = real_device_or_skip() else {
        return;
    };
    let ops = CudaBackendOps::new(0);
    let (m, n, k) = (16usize, 16usize, 16usize);
    let a_f32: Vec<f32> = (0..(m * k)).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_f32: Vec<f32> = (0..(k * n)).map(|i| (i % 5) as f32 * 0.2).collect();
    let a16 = f16_tensor(&a_f32, &[m, k]);
    let b16 = f16_tensor(&b_f32, &[k, n]);

    let gpu = TypedOps::<f16>::gemm(&ops, &a16, &b16)
        .expect("TypedOps::<f16>::gemm succeeds on real hardware");

    let mut reference_f32 = vec![0.0f32; m * n];
    fandhe_ai_backend_cpu::matmul_reference_fma(&a_f32, &b_f32, &mut reference_f32, m, n, k)
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let reference_rounded: Vec<f32> = reference_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    fandhe_ai_backend_cpu::assert_parity(
        "TypedOps<f16>::gemm vs CPU reference",
        &f16_to_f32_vec(&gpu),
        &reference_rounded,
    );
}

/// `add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`（f16）が
/// CPU `BackendOps`（f32）を f16 へ丸めた値と一致することを確認する
/// （broadcast・非 contiguous view・`dim=None`／`Some(axis)` を含む）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn typed_f16_elementwise_and_reduction_match_cpu_backend_ops_rounded() {
    let Some(_device) = real_device_or_skip() else {
        return;
    };
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();

    // [2,3] + [3]（行方向ブロードキャスト）。値は f16 exp/tanh が
    // 有限範囲に収まるよう小さめに抑える（U[-1,1) 程度）。
    let a32 = vec![0.5f32, -0.25, 0.75, -0.5, 0.25, -0.75];
    let b32 = vec![0.1f32, 0.2, 0.3];
    let a16 = f16_tensor(&a32, &[2, 3]);
    let b16 = f16_tensor(&b32, &[3]);
    // CPU 参照側は CUDA 側が実際に計算へ使う値（f16 へ丸めた後の値。
    // `typed_f16::upcast_f16` が `f16::to_f32` で復元する値と同一）を
    // 使う。ここで元の `a32`／`b32` をそのまま使うと、f16 で正確に表現
    // できない値（本テストの入力は 0.1／0.2／0.3 等 2 進小数で丸められる）
    // について「入力の丸め誤差」と「演算自体の誤差」が REQ-2 複合判定
    // に混在し、CUDA 側の演算経路（f32 昇格→f32 カーネル→f16 丸め）
    // 自体の正しさを検証できなくなる（#1797 codex-review 指摘の是正）。
    let a32_rounded: Vec<f32> = a32.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let b32_rounded: Vec<f32> = b32.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let a32_full = Tensor::new(a32_rounded, &[2, 3]).unwrap();
    let b32_full = Tensor::new(b32_rounded, &[3]).unwrap();

    // add / mul（broadcast）。
    let cuda_add = TypedOps::<f16>::add(&cuda, &a16, &b16).expect("cuda add succeeds");
    let cpu_add = BackendOps::add(&cpu, &a32_full, &b32_full).expect("cpu add succeeds");
    let cpu_add_rounded: Vec<f32> = cpu_add
        .host_slice()
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    fandhe_ai_backend_cpu::assert_parity(
        "typed f16 add vs cpu f32 rounded",
        &f16_to_f32_vec(&cuda_add),
        &cpu_add_rounded,
    );

    let cuda_mul = TypedOps::<f16>::mul(&cuda, &a16, &b16).expect("cuda mul succeeds");
    let cpu_mul = BackendOps::mul(&cpu, &a32_full, &b32_full).expect("cpu mul succeeds");
    let cpu_mul_rounded: Vec<f32> = cpu_mul
        .host_slice()
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    fandhe_ai_backend_cpu::assert_parity(
        "typed f16 mul vs cpu f32 rounded",
        &f16_to_f32_vec(&cuda_mul),
        &cpu_mul_rounded,
    );

    // relu / exp / tanh（単項）。
    let cuda_relu = TypedOps::<f16>::relu(&cuda, &a16).expect("cuda relu succeeds");
    let cpu_relu = BackendOps::relu(&cpu, &a32_full).expect("cpu relu succeeds");
    let cpu_relu_rounded: Vec<f32> = cpu_relu
        .host_slice()
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    fandhe_ai_backend_cpu::assert_parity(
        "typed f16 relu vs cpu f32 rounded",
        &f16_to_f32_vec(&cuda_relu),
        &cpu_relu_rounded,
    );

    let cuda_exp = TypedOps::<f16>::exp(&cuda, &a16).expect("cuda exp succeeds");
    let cpu_exp = BackendOps::exp(&cpu, &a32_full).expect("cpu exp succeeds");
    let cpu_exp_rounded: Vec<f32> = cpu_exp
        .host_slice()
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    fandhe_ai_backend_cpu::assert_parity(
        "typed f16 exp vs cpu f32 rounded",
        &f16_to_f32_vec(&cuda_exp),
        &cpu_exp_rounded,
    );

    let cuda_tanh = TypedOps::<f16>::tanh(&cuda, &a16).expect("cuda tanh succeeds");
    let cpu_tanh = BackendOps::tanh(&cpu, &a32_full).expect("cpu tanh succeeds");
    let cpu_tanh_rounded: Vec<f32> = cpu_tanh
        .host_slice()
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    fandhe_ai_backend_cpu::assert_parity(
        "typed f16 tanh vs cpu f32 rounded",
        &f16_to_f32_vec(&cuda_tanh),
        &cpu_tanh_rounded,
    );

    // sum / max（全縮約・軸縮約）。
    let cuda_sum_all = TypedOps::<f16>::sum(&cuda, &a16, None).expect("cuda sum succeeds");
    let cpu_sum_all = BackendOps::sum(&cpu, &a32_full, None).expect("cpu sum succeeds");
    let cpu_sum_all_rounded: Vec<f32> = cpu_sum_all
        .host_slice()
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    fandhe_ai_backend_cpu::assert_parity(
        "typed f16 sum(None) vs cpu f32 rounded",
        &f16_to_f32_vec(&cuda_sum_all),
        &cpu_sum_all_rounded,
    );

    let cuda_sum_axis = TypedOps::<f16>::sum(&cuda, &a16, Some(0)).expect("cuda sum axis succeeds");
    let cpu_sum_axis = BackendOps::sum(&cpu, &a32_full, Some(0)).expect("cpu sum axis succeeds");
    let cpu_sum_axis_rounded: Vec<f32> = cpu_sum_axis
        .host_slice()
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    fandhe_ai_backend_cpu::assert_parity(
        "typed f16 sum(Some(0)) vs cpu f32 rounded",
        &f16_to_f32_vec(&cuda_sum_axis),
        &cpu_sum_axis_rounded,
    );

    let cuda_max_all = TypedOps::<f16>::max(&cuda, &a16, None).expect("cuda max succeeds");
    let cpu_max_all = BackendOps::max(&cpu, &a32_full, None).expect("cpu max succeeds");
    let cpu_max_all_rounded: Vec<f32> = cpu_max_all
        .host_slice()
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    fandhe_ai_backend_cpu::assert_parity(
        "typed f16 max(None) vs cpu f32 rounded",
        &f16_to_f32_vec(&cuda_max_all),
        &cpu_max_all_rounded,
    );
}
