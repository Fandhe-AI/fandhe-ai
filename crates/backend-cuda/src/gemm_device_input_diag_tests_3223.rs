//! イシュー #3223: `gemm_fp32_strict_into_device`（dW）・`gemm_resident_lhs_device`
//! （d_input）が「H2D を出さず・同期を出さず・D2H を出さない」ことを、診断カウンタ
//! （`gemm.rs` の `GemmLaunchDiagCounters`。`cfg(test)` で有効）の前後差分で確かめる
//! 実機診断テスト。対照として同じ形状のホスト版が H2D と同期を出すことも確かめ、
//! カウンタが実際に計数できていること（常に 0 を返す壊れ方をしていないこと）を担保する。
//!
//! # 配置理由
//! カウンタ（`pub(crate)`）へ到達する必要があるため integration test ではなく
//! `lib.rs` の兄弟モジュールとして配置する（`gemm_small_launch_cost_diag_tests` と
//! 同じ判断）。実機（CUDA）依存のため `#[ignore]`。正式な実機記録は #3224。
//!
//! 実行: `cargo test -p fandhe-ai-backend-cuda --release gemm_device_input_diag_tests_3223 -- --ignored --test-threads=1`

use fandhe_ai_tensor_core::{
    BackendOps, DeviceBuffer, DeviceBufferView, DeviceMatrixLayout, DispatchFailureCell, Tensor,
};

use crate::device::CudaDevice;
use crate::gemm::{gemm_launch_diag_reset, gemm_launch_diag_snapshot};
use crate::ops::CudaBackendOps;

fn pattern(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + salt * 17) % 23) as f32 - 11.0) * 0.37)
        .collect()
}

fn upload(ops: &CudaBackendOps, data: &[f32]) -> DeviceBuffer<f32> {
    ops.memory_ops()
        .expect("memory_ops")
        .upload(&Tensor::new(data.to_vec(), &[data.len()]).expect("tensor"))
        .expect("upload")
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn device_input_gemm_issues_no_h2d_no_sync_no_d2h() {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    let ops = CudaBackendOps::new(device.ordinal());
    // 形状は MLP 形状。dW は m = in・k = B・n = out。
    let (bsz, inf, outf) = (64usize, 128usize, 256usize);
    let (m, k, n) = (inf, bsz, outf);

    let x_data = pattern(m * k, 1);
    let g_data = pattern(k * n, 2);
    let w_data = pattern(m * n, 3);
    let x_buf = upload(&ops, &x_data);
    let g_buf = upload(&ops, &g_data);
    let w_buf = upload(&ops, &w_data);
    let (xs, gs, ws) = ([k, m], [k, n], [m, n]);
    let x = DeviceBufferView::new(&x_buf, 0, &xs).expect("view");
    let g = DeviceBufferView::new(&g_buf, 0, &gs).expect("view");
    let w = DeviceBufferView::new(&w_buf, 0, &ws).expect("view");
    let total = m * n + n + 8;
    let mut out = upload(&ops, &vec![0.5f32; total]);
    let bias = Some((m * n + 2, n));

    // ワークスペース拡張と NVRTC 構築を済ませる（定常状態の計測にするため。
    // 拡張時の確保は計数対象外だが、起動前に cached_* を温めておく）。
    let token = DispatchFailureCell::new();
    ops.gemm_fp32_strict_into_device_tracked(
        x,
        DeviceMatrixLayout::Transposed,
        g,
        DeviceMatrixLayout::RowMajor,
        &mut out,
        0,
        bias,
        &token,
    )
    .expect("warmup dW");
    let _ = ops
        .gemm_resident_lhs_device_tracked(w, g, DeviceMatrixLayout::Transposed, &token)
        .expect("warmup d_input");
    ops.memory_ops()
        .expect("memory_ops")
        .download(&out)
        .expect("sync");

    // デバイス入力版: H2D・同期・D2H がすべて 0。
    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    ops.gemm_fp32_strict_into_device(
        x,
        DeviceMatrixLayout::Transposed,
        g,
        DeviceMatrixLayout::RowMajor,
        &mut out,
        0,
        bias,
    )
    .expect("device dW");
    let din = ops
        .gemm_resident_lhs_device(w, g, DeviceMatrixLayout::Transposed)
        .expect("device d_input");
    let d = gemm_launch_diag_snapshot().since(before);
    assert_eq!(d.h2d_calls, 0, "h2d: {d:?}");
    assert_eq!(d.stream_syncs, 0, "sync: {d:?}");
    assert_eq!(d.d2h_calls, 0, "d2h: {d:?}");
    drop(din);

    // 対照: 同じ dW のホスト版は H2D と同期を出す（カウンタが壊れていないことの確認）。
    let x_host = Tensor::new(x_data, &xs)
        .expect("tensor")
        .transpose_2d()
        .expect("transpose");
    let g_host = Tensor::new(g_data, &gs).expect("tensor");
    let mut out_ref = upload(&ops, &vec![0.5f32; total]);
    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    ops.gemm_fp32_strict_into_with_bias_reduce_tracked(
        &x_host,
        &g_host,
        &mut out_ref,
        0,
        bias,
        &token,
    )
    .expect("host dW");
    let d = gemm_launch_diag_snapshot().since(before);
    assert!(d.h2d_calls >= 2, "host path must H2D both inputs: {d:?}");
    assert!(d.stream_syncs >= 1, "host path must synchronize: {d:?}");
}
