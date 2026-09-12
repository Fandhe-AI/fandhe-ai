//! `CudaBackendOps::sum`／`max`（`reduce::CudaReduce`。イシュー #1584・
//! 親イシュー #1571）の実機必須テスト。`linear_forward_device_real_device.rs`
//! と同じ構成方針（`#[ignore]` 分離。CPU 参照実装との統一複合判定・
//! 一部は bit 同一判定）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test reduce_parity -- --ignored --nocapture
//! ```

use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// `Xorshift64Star` 同等の決定的疑似乱数（U[-0.5, 0.5)。`linear_forward_
/// device_real_device.rs` と同じ生成器）。強い相殺を起こさない系列
/// （`.claude/rules/coding-rust.md`「sum の parity は `assert_parity`
/// で行い bit 一致を要求しない」の前提）。
fn xorshift_fill(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        })
        .collect()
}

fn assert_scalar_close(actual: &Tensor<f32>, expected: &Tensor<f32>, ctx: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{ctx}: shape mismatch");
    let a = actual.contiguous();
    let e = expected.contiguous();
    fandhe_ai_backend_cpu::assert_parity(ctx, a.as_slice().unwrap(), e.as_slice().unwrap());
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// 対象形状（全軸・単一軸。`inner==1` の lastaxis 経路・`inner!=1` の
/// 汎用軸経路・中間軸の両方を網羅する。§5.2「reduce_parity」参照）。
const SHAPES: &[(&[usize], Option<usize>)] = &[
    (&[1], None),
    (&[7], None),
    (&[256], None),
    (&[4097], None),
    (&[3, 4096], Some(0)),
    (&[4096, 3], Some(1)),
    (&[33, 17, 5], Some(1)),
    (&[64, 784], Some(1)),
];

/// (a) `sum` が CPU 参照実装（`reduction::sum`）と統一複合判定内で
/// 一致することを全対象形状で確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn sum_matches_cpu_reference_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    for &(shape, dim) in SHAPES {
        let numel: usize = shape.iter().product();
        let a = tensor(xorshift_fill(0x1234_5678 ^ numel as u64, numel), shape);

        let expected = cpu_ops.sum(&a, dim).unwrap();
        let actual = cuda_ops.sum(&a, dim).unwrap();
        assert_scalar_close(
            &actual,
            &expected,
            &format!("sum: shape={shape:?}, dim={dim:?}"),
        );
    }
}

/// (b) `max` が CPU 参照実装（`reduction::max`）と統一複合判定内で一致
/// することを全対象形状で確認する。同値タイがない乱数入力のため
/// bit 同一（厳密選択・丸めなし）も併せて確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn max_matches_cpu_reference_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    for &(shape, dim) in SHAPES {
        let numel: usize = shape.iter().product();
        let a = tensor(xorshift_fill(0x9abc_def0 ^ numel as u64, numel), shape);

        let expected = cpu_ops.max(&a, dim).unwrap();
        let actual = cuda_ops.max(&a, dim).unwrap();
        assert_scalar_close(
            &actual,
            &expected,
            &format!("max: shape={shape:?}, dim={dim:?}"),
        );
        assert_eq!(
            bits(&actual),
            bits(&expected),
            "max is a strict selection (no rounding); tie-free random input should be bit \
             exact: shape={shape:?}, dim={dim:?}"
        );
    }
}

/// (c) 空縮約の意味論（`sum`: 0.0／`max`: `EmptyReduction`）が CPU と
/// 同一であることを確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn empty_reduction_semantics_match_cpu_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());

    // 全軸: numel == 0。
    let empty_all = tensor(Vec::new(), &[0]);
    let sum_all = cuda_ops.sum(&empty_all, None).unwrap();
    assert_eq!(sum_all.contiguous().as_slice().unwrap(), &[0.0f32]);
    assert!(matches!(
        cuda_ops.max(&empty_all, None),
        Err(BackendError::KernelLaunchFailed(msg)) if msg.contains("empty reduction for op \"max\"")
    ));

    // 軸指定: axis_len == 0 かつ outer*inner > 0（出力は非空、縮約長が 0）。
    let empty_axis = tensor(Vec::new(), &[2, 0, 3]);
    let sum_axis = cuda_ops.sum(&empty_axis, Some(1)).unwrap();
    assert_eq!(sum_axis.shape(), &[2, 3]);
    assert!(
        sum_axis
            .contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
    assert!(matches!(
        cuda_ops.max(&empty_axis, Some(1)),
        Err(BackendError::KernelLaunchFailed(msg)) if msg.contains("empty reduction for op \"max\"")
    ));

    // outer*inner == 0（出力自体が空）は vacuous に成功する。
    let vacuous = tensor(Vec::new(), &[0, 5]);
    let sum_vacuous = cuda_ops.sum(&vacuous, Some(0)).unwrap();
    assert_eq!(sum_vacuous.shape(), &[5]);
    let max_vacuous = cuda_ops.max(&vacuous, Some(0)).unwrap();
    assert_eq!(max_vacuous.shape(), &[5]);
}

/// (d) NaN／±inf 入力の意味論が CPU（`f32::max` の NaN 非伝播）と一致
/// することを確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn nan_and_infinity_semantics_match_cpu_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    let cases: &[&[f32]] = &[
        &[f32::NAN, 1.0, 2.0],
        &[f32::NAN, f32::NAN],
        &[f32::INFINITY, 1.0, -f32::INFINITY],
        &[1.0, f32::NAN],
    ];
    for &data in cases {
        let a = tensor(data.to_vec(), &[data.len()]);

        let cpu_max = cpu_ops.max(&a, None).unwrap();
        let cuda_max = cuda_ops.max(&a, None).unwrap();
        assert_eq!(
            bits(&cuda_max),
            bits(&cpu_max),
            "max NaN/inf semantics must match CPU bit-exactly: data={data:?}"
        );

        let cpu_sum = cpu_ops.sum(&a, None).unwrap();
        let cuda_sum = cuda_ops.sum(&a, None).unwrap();
        // NaN 伝播は両者一致するが sum は累積順序が異なるため、NaN の
        // bit パターン自体は比較せず「NaN か否か」で判定する。
        assert_eq!(
            cuda_sum.contiguous().as_slice().unwrap()[0].is_nan(),
            cpu_sum.contiguous().as_slice().unwrap()[0].is_nan(),
            "sum NaN propagation must match CPU: data={data:?}"
        );
    }
}

/// (e) run-to-run 決定性: 同一入力を 3 回実行して bit 同一であることを
/// 確認する（`atomicAdd`／`atomicMax` 不使用の非決定性回避が実際に
/// 機能していることの実機検証）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn sum_and_max_are_run_to_run_deterministic_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());

    let a = tensor(xorshift_fill(0x1357_9bdf, 1 << 16), &[1 << 16]);

    let sum_runs: Vec<Vec<u32>> = (0..3)
        .map(|_| bits(&cuda_ops.sum(&a, None).unwrap()))
        .collect();
    assert_eq!(sum_runs[0], sum_runs[1]);
    assert_eq!(sum_runs[1], sum_runs[2]);

    let max_runs: Vec<Vec<u32>> = (0..3)
        .map(|_| bits(&cuda_ops.max(&a, None).unwrap()))
        .collect();
    assert_eq!(max_runs[0], max_runs[1]);
    assert_eq!(max_runs[1], max_runs[2]);
}
