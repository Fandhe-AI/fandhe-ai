//! イシュー #3250: 常駐入力の行列積（`BackendOps::gemm_device`）と推論チェーンの層列が、
//! 計測窓の中で入力の H2D を起こさないこと、`gemm_device` がホスト入力の `gemm` と bit 一致
//! （NaN はクラス一致）することを確かめる実機診断テスト（`docs/compat-resident-train-step-decision.md`
//! §4・§10）。
//!
//! # 配置理由
//! 診断カウンタ（`gemm.rs` の `GemmLaunchDiagCounters`。`cfg(test)` で有効。`pub(crate)`）へ
//! 到達する必要があるため integration test ではなく `lib.rs` の兄弟モジュールとして置く
//! （`train_step_parts_sync_diag_tests_3224` と同じ判断）。カウンタはスレッドローカルなので、
//! 計測区間の呼び出しは同じスレッドで行う。
//!
//! # 計数
//! 入力の H2D は `MemoryOps::upload`（`mem_uploads`）と `upload_h2d_new`（`h2d_calls`）の両方を
//! 0 と断言する。D2H は結果の読み出し（`download` の同期。先読みの有無で件数が揺れうるため
//! 1 以上で確かめ、カウンタが生きていることの対照を兼ねる）。
//!
//! 実行: `cargo test -p fandhe-ai-backend-cuda --release resident_input_diag_tests_3250 -- --ignored --test-threads=1`

use fandhe_ai_tensor_core::{
    Activation, BackendOps, DeviceBuffer, DeviceBufferView, DispatchFailureCell, Tensor,
};

use crate::device::CudaDevice;
use crate::gemm::{gemm_launch_diag_reset, gemm_launch_diag_snapshot};
use crate::ops::CudaBackendOps;

fn make_ops() -> CudaBackendOps {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    CudaBackendOps::new(device.ordinal())
}

fn pattern(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + salt * 17) % 23) as f32 - 11.0) * 0.37)
        .collect()
}

fn tensor(salt: usize, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(pattern(n, salt), shape).expect("tensor")
}

fn upload(ops: &CudaBackendOps, t: &Tensor<f32>) -> DeviceBuffer<f32> {
    ops.memory_ops()
        .expect("memory_ops")
        .upload(t)
        .expect("upload")
}

fn download(ops: &CudaBackendOps, buf: &DeviceBuffer<f32>) -> Tensor<f32> {
    ops.memory_ops()
        .expect("memory_ops")
        .download(buf)
        .expect("download")
}

fn assert_bits_or_nan_class(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: len");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{what} idx {i}: expected NaN, got {a}");
        } else {
            assert_eq!(a.to_bits(), e.to_bits(), "{what} idx {i}: {a} vs {e}");
        }
    }
}

/// `gemm_device` が `gemm`（FP32 厳密の既定モード）と bit 一致する。N = 256〜4096 と非整列形状。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_device_matches_host_gemm_bitwise() {
    let ops = make_ops();
    for &(m, k, n) in &[
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (64, 784, 256),
        (7, 13, 5),
        (63, 129, 65),
        (1, 1, 1),
    ] {
        let a = tensor(1, &[m, k]);
        let b = tensor(2, &[k, n]);
        let want = ops.gemm(&a, &b).expect("host gemm");
        let (ab, bb) = (upload(&ops, &a), upload(&ops, &b));
        let out = ops.gemm_device(&ab, &bb).expect("gemm_device");
        let got = download(&ops, &out);
        assert_eq!(got.shape(), &[m, n]);
        assert_bits_or_nan_class(
            got.as_slice().expect("contiguous"),
            want.as_slice().expect("contiguous"),
            &format!("{m}x{k}x{n}"),
        );
    }
}

/// 窓の中（`gemm_device` と結果の `download`）で入力の H2D が 0、D2H が 1 回以上。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_device_window_has_no_input_h2d() {
    let ops = make_ops();
    let n = 1024usize;
    let (ab, bb) = (
        upload(&ops, &tensor(1, &[n, n])),
        upload(&ops, &tensor(2, &[n, n])),
    );
    // warm-up（NVRTC 構築）と同期の済ませ。
    let warm = ops.gemm_device(&ab, &bb).expect("warmup");
    let _ = download(&ops, &warm);

    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    let out = ops.gemm_device(&ab, &bb).expect("gemm_device");
    let _ = download(&ops, &out);
    let d = gemm_launch_diag_snapshot().since(before);
    assert_eq!(d.mem_uploads, 0, "input H2D (MemoryOps::upload): {d:?}");
    assert_eq!(d.h2d_calls, 0, "input H2D (upload_h2d_new): {d:?}");
    assert!(d.kernel_launches >= 1, "kernel must launch: {d:?}");
    assert!(d.d2h_calls >= 1, "download must count D2H: {d:?}");
}

/// 推論チェーンの層列（`predict_device_chain_resident` が呼ぶ列そのもの）が、upload 済みの
/// 入力から窓の中で H2D を起こさない。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn chain_layers_window_has_no_input_h2d() {
    let ops = make_ops();
    let (batch, d_in, d_hid, d_out) = (64usize, 784usize, 256usize, 10usize);
    let x = upload(&ops, &tensor(1, &[batch, d_in]));
    let w1 = upload(&ops, &tensor(2, &[d_in, d_hid]));
    let b1 = upload(&ops, &tensor(3, &[d_hid]));
    let w2 = upload(&ops, &tensor(4, &[d_hid, d_out]));
    let token = DispatchFailureCell::new();
    let w1_shape = [d_in, d_hid];
    let b1_shape = [d_hid];
    let w2_shape = [d_hid, d_out];
    let run = || {
        let w1v = DeviceBufferView::new(&w1, 0, &w1_shape).unwrap();
        let b1v = DeviceBufferView::new(&b1, 0, &b1_shape).unwrap();
        let w2v = DeviceBufferView::new(&w2, 0, &w2_shape).unwrap();
        let h = ops
            .linear_forward_device_tracked(&x, w1v, Some(b1v), Activation::Relu, &token)
            .expect("layer 1");
        ops.linear_forward_device_tracked(&h, w2v, None, Activation::None, &token)
            .expect("layer 2")
    };
    let warm = run();
    let _ = download(&ops, &warm);

    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    let out = run();
    let _ = download(&ops, &out);
    let d = gemm_launch_diag_snapshot().since(before);
    assert_eq!(d.mem_uploads, 0, "input H2D (MemoryOps::upload): {d:?}");
    assert_eq!(d.h2d_calls, 0, "input H2D (upload_h2d_new): {d:?}");
    assert!(d.d2h_calls >= 1, "download must count D2H: {d:?}");
}
