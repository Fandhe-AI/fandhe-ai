//! `CudaBackendOps::gemm_resident_rhs_act` の ReLU epilogue 融合 override
//! （イシュー #3202）の実機必須テスト。
//!
//! 融合（`gemm_resident_rhs_act(Relu)`）と手動合成（`gemm_resident_rhs` →
//! `relu`）が bit 一致すること（tolerance 不使用）と、CPU 参照との統一複合判定
//! を検証する。override 後は `gemm_resident_rhs_act` が trait 既定合成へ届かない
//! ため、参照は必ず手動合成で組む。`#[ignore]` 分離（GB10 等の実機必須）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test gemm_resident_rhs_act_fused_real_device -- --ignored --nocapture
//! ```

use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::buffer::DeviceBufferView;
use fandhe_ai_tensor_core::{Activation, BackendOps, Tensor};

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// 決定的疑似乱数（`-0.5` シフトで負値を含め ReLU が恒等化しないようにする）。
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

/// 融合 vs 手動合成の bit 一致と、CPU 参照との統一複合判定（1 ケース分）。
fn check_case(a: &Tensor<f32>, w: &Tensor<f32>, bias: Option<&Tensor<f32>>, ctx: &str) {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let ops = CudaBackendOps::new(device.ordinal());
    let mem = ops.memory_ops().expect("MemoryOps");
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let (k, n) = (w.shape()[0], w.shape()[1]);

    let w_dev = mem.upload(w).unwrap();
    let b_dev = bias.map(|b| mem.upload(b).unwrap());
    let w_shape = [k, n];
    let b_shape = [n];
    let view_w = || DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
    let view_b = || {
        b_dev
            .as_ref()
            .map(|b| DeviceBufferView::new(b, 0, &b_shape).unwrap())
    };

    for act in [Activation::None, Activation::Relu] {
        let fused = ops
            .gemm_resident_rhs_act(a, view_w(), view_b(), act)
            .unwrap();

        // 手動合成（参照）。override は trait 既定に届かないため明示的に組む。
        let plain = ops.gemm_resident_rhs(a, view_w(), view_b()).unwrap();
        let composed = match act {
            Activation::Relu => ops.relu(&plain).unwrap(),
            _ => plain,
        };
        assert_eq!(
            bits(&fused),
            bits(&composed),
            "{ctx} act={act:?}: bit 不一致"
        );

        let expected = cpu.gemm_bias_act(a, w, bias, act).unwrap();
        fandhe_ai_backend_cpu::assert_parity(
            ctx,
            fused.contiguous().as_slice().unwrap(),
            expected.contiguous().as_slice().unwrap(),
        );
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn fused_relu_epilogue_bit_identical_to_manual_composition() {
    for &(m, k, n) in &[
        (1, 1, 1),
        (4, 8, 4),
        (37, 65, 33),
        (64, 784, 256),
        (64, 256, 10),
    ] {
        for has_bias in [false, true] {
            let a = tensor(xorshift_fill(0x1234_5678 ^ (m as u64), m * k), &[m, k]);
            let w = tensor(xorshift_fill(0x9abc_def0 ^ (k as u64), k * n), &[k, n]);
            let bias = has_bias.then(|| tensor(xorshift_fill(0x0f0f_0f0f ^ (n as u64), n), &[n]));
            check_case(
                &a,
                &w,
                bias.as_ref(),
                &format!("m{m} k{k} n{n} bias={has_bias}"),
            );
        }
    }
}

/// NaN・-0.0・大きな負値を混ぜても bit 一致する（NaN→0・-0.0→+0.0 を固定）。
/// NaN を含むため CPU 参照の複合判定は行わず bit 比較のみとする。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn fused_relu_epilogue_bit_identical_on_special_values() {
    let (m, k, n) = (4, 8, 4);
    let mut a = xorshift_fill(1, m * k);
    a[0] = f32::NAN;
    a[9] = -0.0;
    a[17] = -1.0e30;
    let w = xorshift_fill(2, k * n);
    let mut bias = xorshift_fill(3, n);
    bias[1] = -0.0;
    bias[2] = -1.0e30;
    let (a, w, bias) = (tensor(a, &[m, k]), tensor(w, &[k, n]), tensor(bias, &[n]));

    let device = CudaDevice::new(0).expect("CUDA device 0");
    let ops = CudaBackendOps::new(device.ordinal());
    let mem = ops.memory_ops().expect("MemoryOps");
    let w_dev = mem.upload(&w).unwrap();
    let b_dev = mem.upload(&bias).unwrap();
    let w_shape = [k, n];
    let b_shape = [n];
    let view_w = || DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
    let view_b = || Some(DeviceBufferView::new(&b_dev, 0, &b_shape).unwrap());
    let fused = ops
        .gemm_resident_rhs_act(&a, view_w(), view_b(), Activation::Relu)
        .unwrap();
    let composed = ops
        .relu(&ops.gemm_resident_rhs(&a, view_w(), view_b()).unwrap())
        .unwrap();
    assert_eq!(bits(&fused), bits(&composed));
}
