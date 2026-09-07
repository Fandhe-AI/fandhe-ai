//! イシュー #1352: `crate::placement::set_managed_placement_enabled` の
//! opt-in で切り替わる `CudaMemory::alloc_zeroed`／`upload`／`download`
//! （managed memory 配置）の実機必須テスト。
//!
//! 検証内容:
//!
//! 1. managed 配置での upload → download roundtrip が bit 完全一致する
//!    こと（`memory_real_device.rs::upload_download_roundtrip_is_bit_
//!    exact_on_real_hardware` の managed 版）。
//! 2. managed 配置での `alloc_zeroed` が全 0 バッファを返すこと。
//! 3. device-only 配置と managed 配置とで、`gemm_resident_rhs`／
//!    `gemm_resident_lhs`／`linear_forward_device`／`sgd_step_device`
//!    チェーンの出力が bit 完全一致すること（本イシューの核心契約:
//!    配置はメモリの物理的な置き場所のみを変え、カーネル本体・起動
//!    config は完全に共有するため出力は配置に依らず bit 同一となる）。
//! 4. `mem_get_info`（`CudaDevice::context()` 経由）でリークがないことを
//!    確認する（`memory_real_device.rs` と同型）。
//! 5. managed ハンドルに対する `download` が、呼び出し元による明示
//!    `synchronize()` なしの直後呼び出しでも bit 一致すること
//!    （`memory.rs::host_readback` の同期契約の実機確認。単一ストリーム
//!    構成下では `UnifiedSlice` 内部 event に何も記録されないため、
//!    本関数内の明示 `stream.synchronize()` が唯一の同期点であることを
//!    実機で裏取りする）。
//!
//! 対象外（本イシューのスコープ外。`docs/backend-cuda-managed-placement-
//! decision.md` 参照）: fresh モードの素の `CudaBackendOps::gemm`
//! （`CudaMemory` を経由しない）・`gemm_resident_lhs` の NT 転置分岐・
//! 性能比較（5 回中央値）は #1353 が担当する。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --all-features \
//!     --test managed_placement_real_device -- --ignored --nocapture
//! ```

use fandhe_ai_backend_cuda::placement::{managed_placement_enabled, set_managed_placement_enabled};
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice, CudaMemory};
use fandhe_ai_tensor_core::buffer::{DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::device::Device;
use fandhe_ai_tensor_core::{Activation, BackendOps, SgdStepConfig, Tensor};

/// フラグはプロセスグローバル（`crate::placement`）のため、`cargo test`
/// の既定並列実行下での相互干渉を避けて直列化・原状復帰する RAII ガード
/// （`gemm_tf32_optin.rs::Tf32FlagGuard` と同型。統合テストは
/// `crate::placement::test_support`〈`pub(crate)` 限定〉へ到達できない
/// ため、本ファイル専用のロックを持つ）。
static PLACEMENT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct PlacementFlagGuard {
    original: bool,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl PlacementFlagGuard {
    fn acquire(enabled: bool) -> Self {
        let lock = PLACEMENT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let original = managed_placement_enabled();
        set_managed_placement_enabled(enabled);
        Self {
            original,
            _lock: lock,
        }
    }
}

impl Drop for PlacementFlagGuard {
    fn drop(&mut self) {
        set_managed_placement_enabled(self.original);
    }
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn assert_tensor_bit_exact(actual: &Tensor<f32>, expected: &Tensor<f32>, ctx: &str) {
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
        assert_eq!(
            av.to_bits(),
            ev.to_bits(),
            "{ctx}: element {i} must be bit-exact across placements (actual={av} expected={ev})"
        );
    }
}

/// 検証項目 1: managed 配置での upload → download roundtrip が bit 完全一致する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等。managed memory 対応デバイス）必須"]
fn managed_upload_download_roundtrip_is_bit_exact_on_real_hardware() {
    let _guard = PlacementFlagGuard::acquire(true);
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    if !device.managed_memory_supported() {
        eprintln!("skip: device does not support managed memory");
        return;
    }
    let mem = CudaMemory::new(&device);

    let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.5 - 100.0).collect();
    let t = tensor(data.clone(), &[64, 64]);

    let buf = mem
        .upload(&t)
        .expect("managed upload must succeed on real hardware");
    let back = mem
        .download(&buf)
        .expect("managed download must succeed on real hardware");

    for i in 0..64 {
        for j in 0..64 {
            let expected = data[i * 64 + j];
            let actual = back.get(&[i, j]).unwrap();
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "managed roundtrip must be bit exact at [{i}, {j}]"
            );
        }
    }
}

/// 検証項目 2: managed 配置での `alloc_zeroed` は全 0 バッファを返す。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等。managed memory 対応デバイス）必須"]
fn managed_alloc_zeroed_returns_all_zero_on_real_hardware() {
    let _guard = PlacementFlagGuard::acquire(true);
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    if !device.managed_memory_supported() {
        eprintln!("skip: device does not support managed memory");
        return;
    }
    let mem = CudaMemory::new(&device);

    let buf = mem
        .alloc_zeroed(&[128, 128])
        .expect("managed alloc_zeroed must succeed on real hardware");
    let host = mem
        .download(&buf)
        .expect("managed download must succeed on real hardware");
    let data = host.contiguous();
    for (i, v) in data.as_slice().unwrap().iter().enumerate() {
        assert_eq!(v.to_bits(), 0.0f32.to_bits(), "element {i} must be zero");
    }
}

/// 検証項目 5: managed ハンドルへの `download` が明示 `synchronize()` なしの
/// 直後呼び出しでも bit 一致すること（`host_readback` の同期契約）。
/// `alloc_zeroed`（memset カーネル投入） → 即 `download` という、
/// 呼び出し元がストリーム完了を待たない最短経路で検証する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等。managed memory 対応デバイス）必須"]
fn managed_download_is_correct_without_caller_side_synchronize_on_real_hardware() {
    let _guard = PlacementFlagGuard::acquire(true);
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    if !device.managed_memory_supported() {
        eprintln!("skip: device does not support managed memory");
        return;
    }
    let mem = CudaMemory::new(&device);

    // 大きめの要素数にして memset カーネルの実行時間を稼ぎ、
    // `download` 側の明示 `synchronize()` が本当に効いていることを
    // 検出しやすくする（同期が抜けていれば未完了の中身が読める）。
    const NUMEL: usize = 4 * 1024 * 1024;
    let buf = mem
        .alloc_zeroed(&[NUMEL])
        .expect("managed alloc_zeroed must succeed on real hardware");
    // ここで呼び出し元は synchronize() を挟まない（`download` 内部の
    // `host_readback` が唯一の同期点である契約を検証する）。
    let host = mem
        .download(&buf)
        .expect("managed download must succeed on real hardware");
    let data = host.contiguous();
    for (i, v) in data.as_slice().unwrap().iter().enumerate().take(1024) {
        assert_eq!(v.to_bits(), 0.0f32.to_bits(), "element {i} must be zero");
    }
}

/// 検証項目 3: device-only 配置と managed 配置とで、常駐 GEMM／SGD チェーンの
/// 出力が bit 完全一致すること（本イシューの核心契約）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等。managed memory 対応デバイス）必須"]
fn device_and_managed_placement_produce_bit_identical_resident_gemm_output_on_real_hardware() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    if !device.managed_memory_supported() {
        eprintln!("skip: device does not support managed memory");
        return;
    }

    fn run_chain(
        ordinal: usize,
        managed: bool,
    ) -> (Tensor<f32>, Tensor<f32>, Tensor<f32>, Tensor<f32>) {
        let _guard = PlacementFlagGuard::acquire(managed);
        let ops = CudaBackendOps::new(ordinal);
        let mem = ops.memory_ops().expect("MemoryOps must be available");

        let (m, k, n) = (17usize, 33usize, 9usize);
        let a = tensor(
            (0..m * k).map(|i| (i as f32) * 0.01 - 0.5).collect(),
            &[m, k],
        );
        let w = tensor(
            (0..k * n).map(|i| (i as f32) * 0.02 - 0.3).collect(),
            &[k, n],
        );
        let bias = tensor((0..n).map(|i| i as f32 * 0.1).collect(), &[n]);

        let w_dev = mem.upload(&w).unwrap();
        let w_shape = [k, n];
        let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
        let bias_dev = mem.upload(&bias).unwrap();
        let bias_shape = [n];
        let bias_view = DeviceBufferView::new(&bias_dev, 0, &bias_shape).unwrap();

        // gemm_resident_rhs（bias あり）。
        let rhs_out = ops
            .gemm_resident_rhs(&a, w_view, Some(bias_view))
            .expect("gemm_resident_rhs must succeed");

        // linear_forward_device（a をデバイス常駐のまま渡す経路）。
        let a_dev = mem.upload(&a).unwrap();
        let forward_out = ops
            .linear_forward_device(&a_dev, w_view, Some(bias_view), Activation::Relu)
            .expect("linear_forward_device must succeed");
        let forward_out = mem.download(&forward_out).unwrap();

        // gemm_resident_lhs（フォールバック経路。b が dense 転置と
        // 判定されない形状を使い、常に `launch_tiled_f32_resident` を
        // 通す）。
        let (p, q, r) = (11usize, 23usize, 7usize);
        let w2 = tensor(
            (0..p * q).map(|i| (i as f32) * 0.015 - 0.4).collect(),
            &[p, q],
        );
        let b2 = tensor(
            (0..q * r).map(|i| (i as f32) * 0.025 - 0.2).collect(),
            &[q, r],
        );
        let w2_dev = mem.upload(&w2).unwrap();
        let w2_shape = [p, q];
        let w2_view = DeviceBufferView::new(&w2_dev, 0, &w2_shape).unwrap();
        let lhs_out = ops
            .gemm_resident_lhs(w2_view, &b2)
            .expect("gemm_resident_lhs must succeed");

        // sgd_step_device（in-place 更新後の param を download）。
        let mut param = mem.upload(&tensor(vec![1.0, 2.0, 3.0, 4.0], &[4])).unwrap();
        let grad = mem.upload(&tensor(vec![0.1, 0.2, 0.3, 0.4], &[4])).unwrap();
        let config = SgdStepConfig {
            lr: 0.5,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            is_first_step: true,
        };
        ops.sgd_step_device(&mut param, &grad, None, &config)
            .expect("sgd_step_device must succeed");
        let sgd_out = mem.download(&param).unwrap();

        // `gemm_resident_rhs`／`gemm_resident_lhs` は `BackendOps` の
        // 公開シグネチャ上すでに `Tensor`（download 済み）を返す。
        // 4 系統すべてを比較対象として返す。
        (rhs_out, forward_out, lhs_out, sgd_out)
    }

    let (device_rhs, device_forward, device_lhs, device_sgd) = run_chain(device.ordinal(), false);
    let (managed_rhs, managed_forward, managed_lhs, managed_sgd) =
        run_chain(device.ordinal(), true);

    assert_tensor_bit_exact(&managed_rhs, &device_rhs, "gemm_resident_rhs");
    assert_tensor_bit_exact(&managed_forward, &device_forward, "linear_forward_device");
    assert_tensor_bit_exact(&managed_lhs, &device_lhs, "gemm_resident_lhs");
    assert_tensor_bit_exact(&managed_sgd, &device_sgd, "sgd_step_device");
}

/// 検証項目 4: `mem_get_info`（`CudaContext::mem_get_info`）でリークがないことを
/// 確認する（`memory_real_device.rs::repeated_alloc_drop_cycles_do_not_
/// leak_device_memory` と同型の判定方針: リークがあれば途中の
/// `alloc_zeroed` が失敗して本テスト自体が panic する。加えて
/// `mem_get_info` の free bytes がループ前後でおおむね戻ることも
/// 確認する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等。managed memory 対応デバイス）必須"]
fn managed_placement_does_not_leak_on_real_hardware() {
    let _guard = PlacementFlagGuard::acquire(true);
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    if !device.managed_memory_supported() {
        eprintln!("skip: device does not support managed memory");
        return;
    }
    let mem = CudaMemory::new(&device);

    // 64MiB 相当（`memory_real_device.rs` と同一サイズ）でウォームアップ
    // 1 回してからループする。
    let numel = 16 * 1024 * 1024;
    {
        let buf = mem
            .alloc_zeroed(&[numel])
            .expect("warmup alloc must succeed");
        drop(buf);
    }

    let (free_before, _total) = device
        .context()
        .mem_get_info()
        .expect("mem_get_info must succeed on real hardware");

    for _ in 0..100 {
        let buf = mem
            .alloc_zeroed(&[numel])
            .expect("managed alloc_zeroed must succeed within the loop (no leak)");
        drop(buf);
    }

    let final_buf = mem
        .alloc_zeroed(&[numel])
        .expect("allocation after 100 managed alloc/drop cycles must still succeed (no leak)");
    drop(final_buf);

    let (free_after, _total) = device
        .context()
        .mem_get_info()
        .expect("mem_get_info must succeed on real hardware");
    // managed 配置の解放は同期 `cuMemFree`（`UnifiedSlice::drop`）の
    // ため、ループ終了時点で全て解放済みのはず。多少のドライバ内部
    // フラグメンテーションを許容する緩い閾値（8 MiB）で検証する。
    let leaked = free_before.saturating_sub(free_after);
    assert!(
        leaked < 8 * 1024 * 1024,
        "managed placement should not leak across repeated alloc/free rounds: leaked={leaked} bytes"
    );
    let _ = Device::Cuda(device.ordinal());
}
