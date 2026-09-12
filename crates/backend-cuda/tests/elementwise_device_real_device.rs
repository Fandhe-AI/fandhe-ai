//! `CudaBackendOps::binary_elementwise_device`／
//! `unary_elementwise_device`（イシュー #1584）の実機必須テスト。
//! `linear_forward_device_real_device.rs` と同じ構成方針（`#[ignore]`
//! 分離。ホスト版と bit 同一の判定）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test elementwise_device_real_device -- --ignored --nocapture
//! ```

use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{
    BackendOps, BinaryElementwiseOp, MemoryOps, Tensor, UnaryElementwiseOp,
};

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

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

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// (a) 5 演算それぞれについて、`upload` → 常駐版 → `download` がホスト版
/// `BackendOps::add`／`mul`／`relu`／`exp`／`tanh` と bit 同一であること
/// を複数形状で確認する（同一カーネルを再利用する契約）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn binary_and_unary_elementwise_device_match_host_bit_exact_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let ops = CudaBackendOps::new(device.ordinal());
    let mem = ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");

    for &len in &[1usize, 7, 256, 4097, 1 << 20] {
        let a = tensor(xorshift_fill(0x1111_1111 ^ len as u64, len), &[len]);
        let b = tensor(xorshift_fill(0x2222_2222 ^ len as u64, len), &[len]);

        let a_buf = mem.upload(&a).unwrap();
        let b_buf = mem.upload(&b).unwrap();

        let add_dev = ops
            .binary_elementwise_device(BinaryElementwiseOp::Add, &a_buf, &b_buf)
            .unwrap();
        let add_host = ops.add(&a, &b).unwrap();
        assert_eq!(
            bits(&mem.download(&add_dev).unwrap()),
            bits(&add_host),
            "add: len={len}"
        );

        let mul_dev = ops
            .binary_elementwise_device(BinaryElementwiseOp::Mul, &a_buf, &b_buf)
            .unwrap();
        let mul_host = ops.mul(&a, &b).unwrap();
        assert_eq!(
            bits(&mem.download(&mul_dev).unwrap()),
            bits(&mul_host),
            "mul: len={len}"
        );

        let relu_dev = ops
            .unary_elementwise_device(UnaryElementwiseOp::Relu, &a_buf)
            .unwrap();
        let relu_host = ops.relu(&a).unwrap();
        assert_eq!(
            bits(&mem.download(&relu_dev).unwrap()),
            bits(&relu_host),
            "relu: len={len}"
        );

        let exp_dev = ops
            .unary_elementwise_device(UnaryElementwiseOp::Exp, &a_buf)
            .unwrap();
        let exp_host = ops.exp(&a).unwrap();
        assert_eq!(
            bits(&mem.download(&exp_dev).unwrap()),
            bits(&exp_host),
            "exp: len={len}"
        );

        let tanh_dev = ops
            .unary_elementwise_device(UnaryElementwiseOp::Tanh, &a_buf)
            .unwrap();
        let tanh_host = ops.tanh(&a).unwrap();
        assert_eq!(
            bits(&mem.download(&tanh_dev).unwrap()),
            bits(&tanh_host),
            "tanh: len={len}"
        );
    }
}

/// (b) `add → relu → exp` の 3 段チェーンを `download` 1 回で実行し、
/// per-op ホスト版の合成と bit 同一であることを確認する（H2D／D2H・
/// 同期が中間段で発生しないことの機能的検証。厳密なタイミング計測は
/// 別イシューのスコープ）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn elementwise_device_chain_matches_per_op_host_composition_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let ops = CudaBackendOps::new(device.ordinal());
    let mem = ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");

    let len = 513usize;
    let a = tensor(xorshift_fill(0x3333_3333, len), &[len]);
    let b = tensor(xorshift_fill(0x4444_4444, len), &[len]);

    let a_buf = mem.upload(&a).unwrap();
    let b_buf = mem.upload(&b).unwrap();

    let sum_buf = ops
        .binary_elementwise_device(BinaryElementwiseOp::Add, &a_buf, &b_buf)
        .unwrap();
    let relu_buf = ops
        .unary_elementwise_device(UnaryElementwiseOp::Relu, &sum_buf)
        .unwrap();
    let exp_buf = ops
        .unary_elementwise_device(UnaryElementwiseOp::Exp, &relu_buf)
        .unwrap();
    let chained = mem.download(&exp_buf).unwrap();

    let sum_host = ops.add(&a, &b).unwrap();
    let relu_host = ops.relu(&sum_host).unwrap();
    let exp_host = ops.exp(&relu_host).unwrap();

    assert_eq!(bits(&chained), bits(&exp_host));
}

/// (c) `numel == 0` は空バッファを返す（カーネル起動を回避する契約）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn binary_elementwise_device_handles_empty_numel_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let ops = CudaBackendOps::new(device.ordinal());
    let mem = ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");

    let a = tensor(Vec::new(), &[0]);
    let b = tensor(Vec::new(), &[0]);
    let a_buf = mem.upload(&a).unwrap();
    let b_buf = mem.upload(&b).unwrap();

    let out = ops
        .binary_elementwise_device(BinaryElementwiseOp::Add, &a_buf, &b_buf)
        .unwrap();
    assert_eq!(out.shape(), &[0]);
    assert_eq!(out.numel(), 0);
}

/// (d) shape 不一致・device 不一致は起動前に fail-closed で拒否される。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn binary_elementwise_device_rejects_shape_and_device_mismatch_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let ops = CudaBackendOps::new(device.ordinal());
    let mem = ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    let a = tensor(vec![1.0, 2.0], &[2]);
    let b = tensor(vec![1.0, 2.0, 3.0], &[3]);
    let a_buf = mem.upload(&a).unwrap();
    let b_buf = mem.upload(&b).unwrap();
    let result = ops.binary_elementwise_device(BinaryElementwiseOp::Add, &a_buf, &b_buf);
    assert!(matches!(result, Err(BackendError::ShapeMismatch(_))));

    let a_cpu = cpu_ops.upload(&a).unwrap();
    assert_eq!(a_cpu.device(), Device::Cpu);
    let a_cuda = mem.upload(&a).unwrap();
    let result = ops.binary_elementwise_device(BinaryElementwiseOp::Add, &a_cuda, &a_cpu);
    assert!(matches!(result, Err(BackendError::DeviceMismatch)));
}
