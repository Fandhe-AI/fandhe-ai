//! `CudaBackendOps::sum`／`max`／`min`（`reduce::CudaReduce`。イシュー
//! #1584・親イシュー #1571。`min` はイシュー #1720）・`argmax`／
//! `argmin`（`arg_reduce::CudaArgReduce`。イシュー #1948・親イシュー
//! #1947）の実機必須テスト。`linear_forward_device_real_device.rs` と
//! 同じ構成方針（`#[ignore]` 分離。CPU 参照実装との統一複合判定・一部
//! は bit 同一判定）。`argmax`／`argmin` は添字（`i32`）を返す走査
//! 演算のため厳密な添字完全一致（tolerance 不使用）で判定する。
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

/// (b') `min` が CPU 参照実装（`reduction::min`）と統一複合判定内で
/// 一致することを全対象形状で確認する（イシュー #1720。[`max_matches_
/// cpu_reference_on_real_device`] と対称）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn min_matches_cpu_reference_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    for &(shape, dim) in SHAPES {
        let numel: usize = shape.iter().product();
        let a = tensor(xorshift_fill(0x2468_ace0 ^ numel as u64, numel), shape);

        let expected = cpu_ops.min(&a, dim).unwrap();
        let actual = cuda_ops.min(&a, dim).unwrap();
        assert_scalar_close(
            &actual,
            &expected,
            &format!("min: shape={shape:?}, dim={dim:?}"),
        );
        assert_eq!(
            bits(&actual),
            bits(&expected),
            "min is a strict selection (no rounding); tie-free random input should be bit \
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
    assert!(matches!(
        cuda_ops.min(&empty_all, None),
        Err(BackendError::KernelLaunchFailed(msg)) if msg.contains("empty reduction for op \"min\"")
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
    assert!(matches!(
        cuda_ops.min(&empty_axis, Some(1)),
        Err(BackendError::KernelLaunchFailed(msg)) if msg.contains("empty reduction for op \"min\"")
    ));

    // outer*inner == 0（出力自体が空）は vacuous に成功する。shape
    // [0, 5]・axis=1 は outer=0（shape[..1]）・axis_len=5・inner=1 で
    // total_out=outer*inner=0（出力 shape は [0]）。axis=0 だと
    // outer=1・axis_len=0・inner=5 で total_out=5 となり出力が非空の
    // ため上記の「軸指定」ケースと同一になってしまう（vacuous ではない。
    // advisor 指摘により是正）。
    let vacuous = tensor(Vec::new(), &[0, 5]);
    let sum_vacuous = cuda_ops.sum(&vacuous, Some(1)).unwrap();
    assert_eq!(sum_vacuous.shape(), &[0]);
    let max_vacuous = cuda_ops.max(&vacuous, Some(1)).unwrap();
    assert_eq!(max_vacuous.shape(), &[0]);
    let min_vacuous = cuda_ops.min(&vacuous, Some(1)).unwrap();
    assert_eq!(min_vacuous.shape(), &[0]);
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

        let cpu_min = cpu_ops.min(&a, None).unwrap();
        let cuda_min = cuda_ops.min(&a, None).unwrap();
        assert_eq!(
            bits(&cuda_min),
            bits(&cpu_min),
            "min NaN/inf semantics must match CPU bit-exactly: data={data:?}"
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

    let min_runs: Vec<Vec<u32>> = (0..3)
        .map(|_| bits(&cuda_ops.min(&a, None).unwrap()))
        .collect();
    assert_eq!(min_runs[0], min_runs[1]);
    assert_eq!(min_runs[1], min_runs[2]);
}

/// `argmax`／`argmin` 対象形状（`SHAPES` に加え、全軸縮約でチャンク境界
/// を跨ぐ大きさ・`axis_len` がチャンク粒度と無関係な非倍数の形状を
/// 追加する。イシュー #1948）。
const ARG_SHAPES: &[(&[usize], Option<usize>)] = &[
    (&[1], None),
    (&[7], None),
    (&[256], None),
    (&[4097], None),
    (&[10_000], None),
    (&[3, 4096], Some(0)),
    (&[4096, 3], Some(1)),
    (&[33, 17, 5], Some(1)),
    (&[64, 784], Some(1)),
    (&[2, 251, 3], Some(1)),
];

fn bits_i32(t: &Tensor<i32>) -> Vec<i32> {
    t.contiguous().as_slice().unwrap().to_vec()
}

/// (f) `argmax`／`argmin` が CPU 参照実装（`reduction::argmax`／
/// `argmin`）と添字完全一致することを全対象形状で確認する（タイ・NaN
/// を含む乱数入力で「最初の添字」契約を検証するため、値域を狭くして
/// タイを多発させ確率的に NaN を混入する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn argmax_and_argmin_match_cpu_reference_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    for &(shape, dim) in ARG_SHAPES {
        let numel: usize = shape.iter().product();
        let mut state = 0x1122_3344_5566_7788u64 ^ numel as u64;
        let data: Vec<f32> = (0..numel)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                // 値域を [-3, 3] へ丸めてタイを多発させる。
                let v = (-3.0 + 6.0 * u as f32).round();

                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
                // 確率的に NaN を混入する。
                if u2 < 0.05 { f32::NAN } else { v }
            })
            .collect();
        let a = tensor(data, shape);

        let expected_max = cpu_ops.argmax(&a, dim).unwrap();
        let actual_max = cuda_ops.argmax(&a, dim).unwrap();
        assert_eq!(actual_max.shape(), expected_max.shape());
        assert_eq!(
            bits_i32(&actual_max),
            bits_i32(&expected_max),
            "argmax: shape={shape:?}, dim={dim:?}"
        );

        let expected_min = cpu_ops.argmin(&a, dim).unwrap();
        let actual_min = cuda_ops.argmin(&a, dim).unwrap();
        assert_eq!(actual_min.shape(), expected_min.shape());
        assert_eq!(
            bits_i32(&actual_min),
            bits_i32(&expected_min),
            "argmin: shape={shape:?}, dim={dim:?}"
        );
    }
}

/// (g) `argmax`／`argmin` の NaN 混在・全 NaN・全 `±inf`・`±0.0`
/// 混在ケースが CPU と一致することを確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn argmax_and_argmin_nan_and_infinity_semantics_match_cpu_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    let cases: &[&[f32]] = &[
        &[f32::NAN, 1.0, 2.0],
        &[f32::NAN, f32::NAN, f32::NAN],
        &[f32::INFINITY, 1.0, -f32::INFINITY],
        &[1.0, f32::NAN],
        &[0.0, -0.0, 0.0],
        &[f32::NEG_INFINITY; 10],
        &[f32::INFINITY; 10],
    ];
    for &data in cases {
        let a = tensor(data.to_vec(), &[data.len()]);

        let cpu_max = cpu_ops.argmax(&a, None).unwrap();
        let cuda_max = cuda_ops.argmax(&a, None).unwrap();
        assert_eq!(
            bits_i32(&cuda_max),
            bits_i32(&cpu_max),
            "argmax NaN/inf semantics must match CPU: data={data:?}"
        );

        let cpu_min = cpu_ops.argmin(&a, None).unwrap();
        let cuda_min = cuda_ops.argmin(&a, None).unwrap();
        assert_eq!(
            bits_i32(&cuda_min),
            bits_i32(&cpu_min),
            "argmin NaN/inf semantics must match CPU: data={data:?}"
        );
    }
}

/// (h) 空縮約の意味論（`EmptyReduction`。`op` フィールドの文言が CPU
/// と一致）・vacuous 成功（出力自体が空）が `sum`／`max`／`min` と同一
/// であることを確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn argmax_and_argmin_empty_reduction_semantics_match_cpu_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());

    let empty_all = tensor(Vec::new(), &[0]);
    assert!(matches!(
        cuda_ops.argmax(&empty_all, None),
        Err(BackendError::KernelLaunchFailed(msg)) if msg.contains("empty reduction for op \"argmax\"")
    ));
    assert!(matches!(
        cuda_ops.argmin(&empty_all, None),
        Err(BackendError::KernelLaunchFailed(msg)) if msg.contains("empty reduction for op \"argmin\"")
    ));

    let empty_axis = tensor(Vec::new(), &[2, 0, 3]);
    assert!(matches!(
        cuda_ops.argmax(&empty_axis, Some(1)),
        Err(BackendError::KernelLaunchFailed(msg)) if msg.contains("empty reduction for op \"argmax\"")
    ));
    assert!(matches!(
        cuda_ops.argmin(&empty_axis, Some(1)),
        Err(BackendError::KernelLaunchFailed(msg)) if msg.contains("empty reduction for op \"argmin\"")
    ));

    let vacuous = tensor(Vec::new(), &[0, 5]);
    let max_vacuous = cuda_ops.argmax(&vacuous, Some(1)).unwrap();
    assert_eq!(max_vacuous.shape(), &[0]);
    let min_vacuous = cuda_ops.argmin(&vacuous, Some(1)).unwrap();
    assert_eq!(min_vacuous.shape(), &[0]);
}

/// (i) 非 contiguous 入力（transpose view）でも `argmax`／`argmin` が
/// contiguous 実体化後の CPU 参照実装と一致することを確認する
/// （`arg_reduce_dispatch` の `a.contiguous()` 経路の検証）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn argmax_and_argmin_match_cpu_reference_for_transposed_view_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    let data = xorshift_fill(0x4321_dcba, 12 * 7);
    let a = tensor(data, &[12, 7]).transpose_2d().unwrap();
    assert!(
        a.as_slice().is_none(),
        "transpose_2d は非 contiguous であるべき"
    );

    for dim in [None, Some(0), Some(1)] {
        let expected_max = cpu_ops.argmax(&a, dim).unwrap();
        let actual_max = cuda_ops.argmax(&a, dim).unwrap();
        assert_eq!(
            bits_i32(&actual_max),
            bits_i32(&expected_max),
            "dim={dim:?}"
        );

        let expected_min = cpu_ops.argmin(&a, dim).unwrap();
        let actual_min = cuda_ops.argmin(&a, dim).unwrap();
        assert_eq!(
            bits_i32(&actual_min),
            bits_i32(&expected_min),
            "dim={dim:?}"
        );
    }
}

/// (j) run-to-run 決定性（`sum_and_max_are_run_to_run_deterministic_
/// on_real_device` と同型）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn argmax_and_argmin_are_run_to_run_deterministic_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());

    let a = tensor(xorshift_fill(0x2468_1357, 1 << 16), &[1 << 16]);

    let argmax_runs: Vec<Vec<i32>> = (0..3)
        .map(|_| bits_i32(&cuda_ops.argmax(&a, None).unwrap()))
        .collect();
    assert_eq!(argmax_runs[0], argmax_runs[1]);
    assert_eq!(argmax_runs[1], argmax_runs[2]);

    let argmin_runs: Vec<Vec<i32>> = (0..3)
        .map(|_| bits_i32(&cuda_ops.argmin(&a, None).unwrap()))
        .collect();
    assert_eq!(argmin_runs[0], argmin_runs[1]);
    assert_eq!(argmin_runs[1], argmin_runs[2]);
}
