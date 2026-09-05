//! `CudaBackendOps::linear_forward_device`（イシュー #1216・`docs/
//! inference-forward-fixed-cost-design.md` §3.2「段階 B」）の実機必須
//! テスト。`crates/backend-cuda/tests/gemm_resident_real_device.rs` と
//! 同じ構成方針（`#[ignore]` 分離。CPU 参照実装との統一複合判定）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test linear_forward_device_real_device -- --ignored --nocapture
//! ```

use std::time::Instant;

use bench_harness::median_q1_q3;
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::buffer::DeviceBufferView;
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{Activation, BackendOps, Tensor};

fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn assert_tensor_close(actual: &Tensor<f32>, expected: &Tensor<f32>, ctx: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{ctx}: shape mismatch");
    let a = actual.contiguous();
    let e = expected.contiguous();
    for (i, (av, ev)) in a
        .as_slice()
        .unwrap()
        .iter()
        .zip(e.as_slice().unwrap())
        .enumerate()
    {
        assert_close(*av, *ev, &format!("{ctx}: element {i}"));
    }
}

/// `Xorshift64Star` 同等の決定的疑似乱数（`-0.5` シフトで ReLU 恒等化を
/// 避ける。`docs/inference-forward-fixed-cost-design.md` の parity テスト
/// と同じ方針）。
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

/// (a) `linear_forward_device`（CUDA）が CPU 参照実装（`gemm_bias_act`）
/// と統一複合判定内で一致することを、bias 有無 × `Activation::{None,
/// Relu}` の組み合わせで検証する（イシュー #1216 実装計画 Step 5 (a)）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn linear_forward_device_matches_cpu_reference_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cuda_mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    for &(m, k, n) in &[
        (1, 1, 1),
        (4, 8, 4),
        (37, 65, 33),
        (64, 784, 256),
        (64, 256, 10),
    ] {
        for has_bias in [false, true] {
            for act in [Activation::None, Activation::Relu] {
                let a = tensor(xorshift_fill(0x1234_5678 ^ (m as u64), m * k), &[m, k]);
                let w = tensor(xorshift_fill(0x9abc_def0 ^ (k as u64), k * n), &[k, n]);
                let bias =
                    has_bias.then(|| tensor(xorshift_fill(0x0f0f_0f0f ^ (n as u64), n), &[n]));

                let expected = cpu_ops.gemm_bias_act(&a, &w, bias.as_ref(), act).unwrap();

                let a_dev = cuda_mem.upload(&a).unwrap();
                let w_dev = cuda_mem.upload(&w).unwrap();
                let w_shape = [k, n];
                let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
                let bias_dev = bias.as_ref().map(|b| cuda_mem.upload(b).unwrap());
                let bias_shape = [n];
                let bias_view = bias_dev
                    .as_ref()
                    .map(|buf| DeviceBufferView::new(buf, 0, &bias_shape).unwrap());

                let actual_dev = cuda_ops
                    .linear_forward_device(&a_dev, w_view, bias_view, act)
                    .unwrap();
                let actual = cuda_mem.download(&actual_dev).unwrap();

                assert_tensor_close(
                    &actual,
                    &expected,
                    &format!(
                        "linear_forward_device m={m} k={k} n={n} has_bias={has_bias} act={act:?}"
                    ),
                );
            }
        }
    }
}

/// (b) 同バックエンドの `gemm_resident_rhs_act`（`a`／戻り値がホスト
/// 常駐）と bit 完全一致することを確認する（同一カーネル・同一 launch
/// config・epilogue ReLU の恒等性。`act_relu` 配線ミスを零許容で検出する
/// 回帰テスト。イシュー #1216 実装計画 Step 5 (b)）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn linear_forward_device_matches_gemm_resident_rhs_act_bit_exact_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cuda_mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");

    for &(m, k, n) in &[(1, 1, 1), (4, 8, 4), (37, 65, 33), (64, 784, 256)] {
        for has_bias in [false, true] {
            for act in [Activation::None, Activation::Relu] {
                let a = tensor(xorshift_fill(0x2468_ace0 ^ (m as u64), m * k), &[m, k]);
                let w = tensor(xorshift_fill(0x1357_9bdf ^ (k as u64), k * n), &[k, n]);
                let bias =
                    has_bias.then(|| tensor(xorshift_fill(0xa5a5_a5a5 ^ (n as u64), n), &[n]));

                let w_dev = cuda_mem.upload(&w).unwrap();
                let w_shape = [k, n];
                let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
                let bias_dev = bias.as_ref().map(|b| cuda_mem.upload(b).unwrap());
                let bias_shape = [n];
                let bias_view = bias_dev
                    .as_ref()
                    .map(|buf| DeviceBufferView::new(buf, 0, &bias_shape).unwrap());

                // 参照: `gemm_resident_rhs_act`（`a` を毎回 upload・結果を
                // 毎回 download）。
                let expected = cuda_ops
                    .gemm_resident_rhs_act(&a, w_view, bias_view, act)
                    .unwrap();

                // 対象: `linear_forward_device`（`a` も呼び出し元が常駐
                // させる）。
                let a_dev = cuda_mem.upload(&a).unwrap();
                let actual_dev = cuda_ops
                    .linear_forward_device(&a_dev, w_view, bias_view, act)
                    .unwrap();
                let actual = cuda_mem.download(&actual_dev).unwrap();

                assert_eq!(
                    actual.shape(),
                    expected.shape(),
                    "shape mismatch m={m} k={k} n={n} has_bias={has_bias} act={act:?}"
                );
                let a_slice = actual.contiguous();
                let e_slice = expected.contiguous();
                assert_eq!(
                    a_slice.as_slice().unwrap(),
                    e_slice.as_slice().unwrap(),
                    "linear_forward_device と gemm_resident_rhs_act は同一カーネルの \
                     はずのため bit 完全一致するはず: m={m} k={k} n={n} has_bias={has_bias} \
                     act={act:?}"
                );
            }
        }
    }
}

/// (c) 2 層チェーン（batch 64・784→256→ReLU→10）を `linear_forward_device`
/// の連鎖で計算し、CPU 参照（`gemm_bias_act` の 2 段連鎖）と一致する
/// ことを確認する。CUDA は同一ストリーム FIFO により前層出力を次層の
/// 入力として読む順序が保証されることの実機確認を兼ねる（`docs/
/// backend-cuda-async-execution-design.md` の契約）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn linear_forward_device_two_layer_chain_matches_cpu_reference_on_real_device() {
    const BATCH: usize = 64;
    const D_IN: usize = 784;
    const D_HIDDEN: usize = 256;
    const D_OUT: usize = 10;

    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cuda_mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    let x = tensor(xorshift_fill(0x1111_2222, BATCH * D_IN), &[BATCH, D_IN]);
    let w1 = tensor(
        xorshift_fill(0x3333_4444, D_IN * D_HIDDEN),
        &[D_IN, D_HIDDEN],
    );
    let b1 = tensor(xorshift_fill(0x5555_6666, D_HIDDEN), &[D_HIDDEN]);
    let w2 = tensor(
        xorshift_fill(0x7777_8888, D_HIDDEN * D_OUT),
        &[D_HIDDEN, D_OUT],
    );
    let b2 = tensor(xorshift_fill(0x9999_aaaa, D_OUT), &[D_OUT]);

    // CPU 参照: `gemm_bias_act`（融合 epilogue 版）の 2 段連鎖。
    let h1_ref = cpu_ops
        .gemm_bias_act(&x, &w1, Some(&b1), Activation::Relu)
        .unwrap();
    let expected = cpu_ops
        .gemm_bias_act(&h1_ref, &w2, Some(&b2), Activation::None)
        .unwrap();

    // CUDA: upload 1 回 → linear_forward_device ×2 → download 1 回。
    let x_dev = cuda_mem.upload(&x).unwrap();
    let w1_dev = cuda_mem.upload(&w1).unwrap();
    let w1_view = DeviceBufferView::new(&w1_dev, 0, &[D_IN, D_HIDDEN]).unwrap();
    let b1_dev = cuda_mem.upload(&b1).unwrap();
    let b1_view = DeviceBufferView::new(&b1_dev, 0, &[D_HIDDEN]).unwrap();
    let w2_dev = cuda_mem.upload(&w2).unwrap();
    let w2_view = DeviceBufferView::new(&w2_dev, 0, &[D_HIDDEN, D_OUT]).unwrap();
    let b2_dev = cuda_mem.upload(&b2).unwrap();
    let b2_view = DeviceBufferView::new(&b2_dev, 0, &[D_OUT]).unwrap();

    let h1_dev = cuda_ops
        .linear_forward_device(&x_dev, w1_view, Some(b1_view), Activation::Relu)
        .unwrap();
    let out_dev = cuda_ops
        .linear_forward_device(&h1_dev, w2_view, Some(b2_view), Activation::None)
        .unwrap();
    let actual = cuda_mem.download(&out_dev).unwrap();

    assert_tensor_close(&actual, &expected, "linear_forward_device 2 層チェーン");
}

/// (d) fail-closed 系列: `w` 形状不整合・`bias` 形状不一致・空形状
/// （`m == 0`）の早期 return が空 `[0, n]` を返すこと・CPU バッファを
/// `a` に渡すと `DeviceMismatch` になることを確認する（実装計画 Step 5
/// (d)）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn linear_forward_device_rejects_shape_mismatches_and_handles_empty_input_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cuda_mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");

    let a = tensor(xorshift_fill(0xdead_beef, 4 * 8), &[4, 8]);
    let a_dev = cuda_mem.upload(&a).unwrap();

    // `w` の行数が `a` の列数と不一致（k=8 に対し w は [7, 4]）。
    let w_bad = tensor(xorshift_fill(0xfeed_face, 7 * 4), &[7, 4]);
    let w_bad_dev = cuda_mem.upload(&w_bad).unwrap();
    let w_bad_view = DeviceBufferView::new(&w_bad_dev, 0, &[7, 4]).unwrap();
    let result = cuda_ops.linear_forward_device(&a_dev, w_bad_view, None, Activation::None);
    assert!(
        matches!(result, Err(BackendError::ShapeMismatch(_))),
        "w 形状不整合は ShapeMismatch で拒否されるはず: {result:?}"
    );

    // `bias` が `[n]` ちょうどでない（n=4 に対し bias は [3]）。
    let w = tensor(xorshift_fill(0x1a2b_3c4d, 8 * 4), &[8, 4]);
    let w_dev = cuda_mem.upload(&w).unwrap();
    let w_view = DeviceBufferView::new(&w_dev, 0, &[8, 4]).unwrap();
    let bias_bad = tensor(xorshift_fill(0x5e6f_7a8b, 3), &[3]);
    let bias_bad_dev = cuda_mem.upload(&bias_bad).unwrap();
    let bias_bad_view = DeviceBufferView::new(&bias_bad_dev, 0, &[3]).unwrap();
    let result =
        cuda_ops.linear_forward_device(&a_dev, w_view, Some(bias_bad_view), Activation::None);
    assert!(
        matches!(result, Err(BackendError::ShapeMismatch(_))),
        "bias 形状不一致は ShapeMismatch で拒否されるはず: {result:?}"
    );

    // `m == 0`（空入力）の早期 return は空 `[0, n]` を返す。
    let a_empty = tensor(Vec::new(), &[0, 8]);
    let a_empty_dev = cuda_mem.upload(&a_empty).unwrap();
    let out_dev = cuda_ops
        .linear_forward_device(&a_empty_dev, w_view, None, Activation::None)
        .unwrap();
    assert_eq!(out_dev.shape(), &[0usize, 4]);

    // CPU デバイスの `a` は `DeviceMismatch` で拒否される。
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cpu_mem = cpu_ops
        .memory_ops()
        .expect("CpuBackendOps must implement MemoryOps");
    let a_cpu_dev = cpu_mem.upload(&a).unwrap();
    let result = cuda_ops.linear_forward_device(&a_cpu_dev, w_view, None, Activation::None);
    assert!(
        matches!(result, Err(BackendError::DeviceMismatch)),
        "CPU デバイスの a は DeviceMismatch で拒否されるはず: {result:?}"
    );
    assert_eq!(a_cpu_dev.device(), Device::Cpu);
}

/// backend レベル before/after 記録用ベンチ（record-only。hard assert
/// なし。`device_param_store_bench.rs` と同方針）。2 層チェーン（batch
/// 64・784→256→ReLU→10）で、旧経路（層ごとに `gemm_resident_rhs_act`
/// を呼び毎回 D2H→H2D）と新経路（`mem.upload` 1 回 → `linear_forward_
/// device` ×2 → `mem.download` 1 回）の per-forward 時間を比較する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須（record only）"]
fn linear_forward_device_bench_cuda() {
    const BATCH: usize = 64;
    const D_IN: usize = 784;
    const D_HIDDEN: usize = 256;
    const D_OUT: usize = 10;
    const WARMUP: usize = 20;
    const ITERS: usize = 20;
    const TRIALS: usize = 5;

    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cuda_mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");

    let x = tensor(xorshift_fill(0x1111_2222, BATCH * D_IN), &[BATCH, D_IN]);
    let w1 = tensor(
        xorshift_fill(0x3333_4444, D_IN * D_HIDDEN),
        &[D_IN, D_HIDDEN],
    );
    let b1 = tensor(xorshift_fill(0x5555_6666, D_HIDDEN), &[D_HIDDEN]);
    let w2 = tensor(
        xorshift_fill(0x7777_8888, D_HIDDEN * D_OUT),
        &[D_HIDDEN, D_OUT],
    );
    let b2 = tensor(xorshift_fill(0x9999_aaaa, D_OUT), &[D_OUT]);

    let w1_dev = cuda_mem.upload(&w1).unwrap();
    let w1_view = DeviceBufferView::new(&w1_dev, 0, &[D_IN, D_HIDDEN]).unwrap();
    let b1_dev = cuda_mem.upload(&b1).unwrap();
    let b1_view = DeviceBufferView::new(&b1_dev, 0, &[D_HIDDEN]).unwrap();
    let w2_dev = cuda_mem.upload(&w2).unwrap();
    let w2_view = DeviceBufferView::new(&w2_dev, 0, &[D_HIDDEN, D_OUT]).unwrap();
    let b2_dev = cuda_mem.upload(&b2).unwrap();
    let b2_view = DeviceBufferView::new(&b2_dev, 0, &[D_OUT]).unwrap();

    let run_before = || {
        let h1 = cuda_ops
            .gemm_resident_rhs_act(&x, w1_view, Some(b1_view), Activation::Relu)
            .unwrap();
        let out = cuda_ops
            .gemm_resident_rhs_act(&h1, w2_view, Some(b2_view), Activation::None)
            .unwrap();
        std::hint::black_box(out);
    };
    let run_after = || {
        let x_dev = cuda_mem.upload(&x).unwrap();
        let h1_dev = cuda_ops
            .linear_forward_device(&x_dev, w1_view, Some(b1_view), Activation::Relu)
            .unwrap();
        let out_dev = cuda_ops
            .linear_forward_device(&h1_dev, w2_view, Some(b2_view), Activation::None)
            .unwrap();
        let out = cuda_mem.download(&out_dev).unwrap();
        std::hint::black_box(out);
    };

    for _ in 0..WARMUP {
        run_before();
        run_after();
    }

    let mut before_medians = Vec::with_capacity(TRIALS);
    let mut after_medians = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let mut before_samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t0 = Instant::now();
            run_before();
            before_samples.push(t0.elapsed().as_secs_f64());
        }
        let median = median_q1_q3(&before_samples)
            .expect("non-empty samples")
            .median;
        before_medians.push(median);

        let mut after_samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t0 = Instant::now();
            run_after();
            after_samples.push(t0.elapsed().as_secs_f64());
        }
        let median = median_q1_q3(&after_samples)
            .expect("non-empty samples")
            .median;
        after_medians.push(median);
    }

    let before_median = median_q1_q3(&before_medians)
        .expect("non-empty samples")
        .median;
    let after_median = median_q1_q3(&after_medians)
        .expect("non-empty samples")
        .median;
    let speedup = before_median / after_median;
    println!(
        "[linear_forward_device_bench:cuda] before_median_s={before_median:.6} \
         after_median_s={after_median:.6} speedup_x={speedup:.3}"
    );
}
