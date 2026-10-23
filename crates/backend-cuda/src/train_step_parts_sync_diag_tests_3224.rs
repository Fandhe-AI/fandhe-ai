//! イシュー #3224: 学習ステップ常駐化の CUDA 3 部品（ReLU マスク・MSE・dW／d_input GEMM）の
//! デバイス入出力版が、MLP（784→256→10・batch 64）の形状で「同期を出さず・D2H を出さない」
//! （GEMM は H2D も出さない）ことを、診断カウンタ（`gemm.rs` の `GemmLaunchDiagCounters`。
//! `cfg(test)` で有効）の前後差分で確かめる実機診断テスト。あわせて backward 部分列を
//! `_tracked` 版だけで連結した区間全体でも同期・D2H が 0 であることと、各部品の出力が
//! ホスト版（設計 `docs/train-step-resident-parts-design.md` §4 の比較先）と bit 一致
//! （NaN はクラス一致。MSE のみ CPU 参照との統一複合判定ではなく、同じバックエンドの
//! ホスト版との比較）であることを確かめる。
//!
//! # 配置理由
//! カウンタ（`pub(crate)`）へ到達する必要があるため integration test ではなく `lib.rs` の
//! 兄弟モジュールとして配置する（`gemm_device_input_diag_tests_3223` と同じ判断）。
//! カウンタはスレッドローカルなので、計測区間の呼び出しはすべて同じスレッドで行う。
//!
//! # 計数の範囲（記録上の注記）
//! `diag_count_h2d` は `gemm.rs` の H2D だけを数え、ReLU・MSE のホスト版の `clone_htod` は
//! 数えない。新版は入力が `DeviceBuffer` なので構造上 H2D がなく、確認は `stream_syncs`／
//! `d2h_calls` の 0 と、ソース上 `clone_htod`・`readback` を呼ばないこと（設計 §3）による。
//! 対照として `download`（`readback` を通る）が同期と D2H を出すことを確かめ、
//! カウンタが常に 0 を返す壊れ方をしていないことを担保する。
//!
//! 正式な実機記録（RULE.txt・5 run）は `docs/perf/logs/train-step-resident-parts-parity-3224/`。
//!
//! 実行: `cargo test -p fandhe-ai-backend-cuda --release train_step_parts_sync_diag_tests_3224 -- --ignored --test-threads=1`

use fandhe_ai_tensor_core::{
    BackendOps, DeviceBuffer, DeviceBufferView, DeviceMatrixLayout, DispatchFailureCell,
    MseReduction, Tensor,
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

fn upload(ops: &CudaBackendOps, data: &[f32], shape: &[usize]) -> DeviceBuffer<f32> {
    ops.memory_ops()
        .expect("memory_ops")
        .upload(&Tensor::new(data.to_vec(), shape).expect("tensor"))
        .expect("upload")
}

fn download(ops: &CudaBackendOps, buf: &DeviceBuffer<f32>) -> Vec<f32> {
    ops.memory_ops()
        .expect("memory_ops")
        .download(buf)
        .expect("download")
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

fn to_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.as_slice().map_or_else(
        || t.contiguous().as_slice().unwrap().to_vec(),
        <[f32]>::to_vec,
    )
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

fn transpose(stored: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = stored[i * c + j];
        }
    }
    out
}

/// 対照: `download` が同期を数えること（カウンタが生きていることの確認）。
/// D2H 回数は検査しない: readback 先読み既定 ON（#3227）では `d2h_calls` が 0 のまま
/// 同期だけが 1 回出る経路があり、D2H の有無は対照として不安定なため。
fn assert_download_is_counted(ops: &CudaBackendOps, buf: &DeviceBuffer<f32>) {
    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    let _ = download(ops, buf);
    let d = gemm_launch_diag_snapshot().since(before);
    assert!(d.stream_syncs >= 1, "download must synchronize: {d:?}");
}

/// ReLU マスク: `[64, 256]` の RowMajor／Transposed で、非 tracked・tracked とも同期・D2H が 0。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn relu_mask_device_issues_no_sync_no_d2h() {
    let ops = make_ops();
    let (r, c) = (64usize, 256usize);
    let n = r * c;
    let mut last_out: Option<DeviceBuffer<f32>> = None;
    for layout in [DeviceMatrixLayout::RowMajor, DeviceMatrixLayout::Transposed] {
        let g_store = pattern(n, 1);
        let m_store = pattern(n, 2);
        let g_buf = upload(&ops, &g_store, &[r, c]);
        let m_buf = upload(&ops, &m_store, &[r, c]);
        let (logical, g_logical) = match layout {
            DeviceMatrixLayout::Transposed => ([c, r], transpose(&g_store, r, c)),
            _ => ([r, c], g_store.clone()),
        };
        let g_shape = [r, c];
        let g = DeviceBufferView::new(&g_buf, 0, &g_shape).unwrap();
        let m = DeviceBufferView::new(&m_buf, 0, &logical).unwrap();
        let want: Vec<f32> = g_logical
            .iter()
            .zip(&m_store)
            .map(|(&g, &m)| if m > 0.0 { g } else { 0.0 })
            .collect();
        let token = DispatchFailureCell::new();

        // warm-up（NVRTC 構築）と同期の済ませ。
        let warm = ops
            .relu_backward_mask_device_tracked(g, layout, m, &token)
            .expect("warmup");
        let _ = download(&ops, &warm);

        gemm_launch_diag_reset();
        let before = gemm_launch_diag_snapshot();
        let out = ops.relu_backward_mask_device(g, layout, m).expect("sync");
        let out_t = ops
            .relu_backward_mask_device_tracked(g, layout, m, &token)
            .expect("tracked");
        let d = gemm_launch_diag_snapshot().since(before);
        assert_eq!(d.stream_syncs, 0, "{layout:?} sync: {d:?}");
        assert_eq!(d.d2h_calls, 0, "{layout:?} d2h: {d:?}");
        assert_bits_or_nan_class(&download(&ops, &out), &want, &format!("{layout:?}"));
        assert_bits_or_nan_class(
            &download(&ops, &out_t),
            &want,
            &format!("tracked {layout:?}"),
        );
        assert!(!token.is_set());
        last_out = Some(out);
    }
    // 対照は全 layout（Transposed 周回を含む）の検査を終えてからループ外で 1 回取る。
    assert_download_is_counted(&ops, last_out.as_ref().expect("loop ran"));
}

/// MSE: `[64, 10]` の forward（Mean／Sum）・backward が、非 tracked・tracked とも同期・D2H 0。
/// 出力はホスト版 `mse_loss`／`mse_loss_backward` と bit 一致。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn mse_device_issues_no_sync_no_d2h() {
    let ops = make_ops();
    let shape = [64usize, 10];
    let n = 640usize;
    let p = pattern(n, 3);
    let t = pattern(n, 4);
    let p_d = upload(&ops, &p, &shape);
    let t_d = upload(&ops, &t, &shape);
    let p_h = Tensor::new(p, &shape).unwrap();
    let t_h = Tensor::new(t, &shape).unwrap();
    let scale = 2.0 / n as f32;
    let token = DispatchFailureCell::new();

    // warm-up（NVRTC 構築）と同期の済ませ。
    let warm = ops
        .mse_loss_backward_device_tracked(&p_d, &t_d, scale, &token)
        .expect("warmup");
    let _ = download(&ops, &warm);
    let _ = ops
        .mse_loss_device_tracked(&p_d, &t_d, MseReduction::Mean, &token)
        .expect("warmup fwd");

    for reduction in [MseReduction::Mean, MseReduction::Sum] {
        let want = ops.mse_loss(&p_h, &t_h, reduction).unwrap();
        gemm_launch_diag_reset();
        let before = gemm_launch_diag_snapshot();
        let a = ops.mse_loss_device(&p_d, &t_d, reduction).unwrap();
        let b = ops
            .mse_loss_device_tracked(&p_d, &t_d, reduction, &token)
            .unwrap();
        let d = gemm_launch_diag_snapshot().since(before);
        assert_eq!(d.stream_syncs, 0, "fwd {reduction:?} sync: {d:?}");
        assert_eq!(d.d2h_calls, 0, "fwd {reduction:?} d2h: {d:?}");
        assert_bits_or_nan_class(&download(&ops, &a), want.as_slice().unwrap(), "loss");
        assert_bits_or_nan_class(
            &download(&ops, &b),
            want.as_slice().unwrap(),
            "tracked loss",
        );
    }

    let want = ops.mse_loss_backward(&p_h, &t_h, scale).unwrap();
    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    let a = ops.mse_loss_backward_device(&p_d, &t_d, scale).unwrap();
    let b = ops
        .mse_loss_backward_device_tracked(&p_d, &t_d, scale, &token)
        .unwrap();
    let d = gemm_launch_diag_snapshot().since(before);
    assert_eq!(d.stream_syncs, 0, "bwd sync: {d:?}");
    assert_eq!(d.d2h_calls, 0, "bwd d2h: {d:?}");
    assert_bits_or_nan_class(&download(&ops, &a), want.as_slice().unwrap(), "grad");
    assert_bits_or_nan_class(
        &download(&ops, &b),
        want.as_slice().unwrap(),
        "tracked grad",
    );
    assert!(!token.is_set());

    // 対照: ホスト版は readback を通るので同期・D2H を数える。
    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    let _ = ops.mse_loss_backward(&p_h, &t_h, scale).unwrap();
    let d = gemm_launch_diag_snapshot().since(before);
    // D2H 回数は readback 先読み既定 ON（#3227）で 0 になりうるため検査しない。
    assert!(d.stream_syncs >= 1, "host mse must synchronize: {d:?}");
}

/// dW（TN・bias あり）と d_input（`Transposed`）を MLP の 2 形状で呼び、H2D・同期・D2H が 0。
/// 3223 の判定（`(64,128,256)` のみ）を MLP の実形状へ広げる。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_device_input_mlp_shapes_issue_no_h2d_no_sync_no_d2h() {
    let ops = make_ops();
    for (bsz, inf, outf) in [(64usize, 784usize, 256usize), (64, 256, 10)] {
        let (m, k, n) = (inf, bsz, outf);
        let x_buf = upload(&ops, &pattern(m * k, 1), &[k, m]);
        let g_buf = upload(&ops, &pattern(k * n, 2), &[k, n]);
        let w_buf = upload(&ops, &pattern(m * n, 3), &[m, n]);
        let (xs, gs, ws) = ([k, m], [k, n], [m, n]);
        let x = DeviceBufferView::new(&x_buf, 0, &xs).unwrap();
        let g = DeviceBufferView::new(&g_buf, 0, &gs).unwrap();
        let w = DeviceBufferView::new(&w_buf, 0, &ws).unwrap();
        let total = m * n + n + 8;
        let mut out = upload(&ops, &vec![0.5f32; total], &[total]);
        let bias = Some((m * n + 2, n));
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
        let _ = download(&ops, &out);

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
        let what = format!("(B,in,out)=({bsz},{inf},{outf})");
        assert_eq!(d.h2d_calls, 0, "{what} h2d: {d:?}");
        assert_eq!(d.stream_syncs, 0, "{what} sync: {d:?}");
        assert_eq!(d.d2h_calls, 0, "{what} d2h: {d:?}");
        drop(din);
    }
}

/// backward 部分列（MSE backward → dW L2 → d_input L2 → ReLU マスク → dW L1）を `_tracked` 版
/// だけで連結し、区間全体で H2D・同期・D2H が 0。最後に download して各出力をホスト版の
/// 連鎖結果と bit 比較する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn mlp_backward_parts_tracked_chain_issues_no_sync() {
    let ops = make_ops();
    let (bsz, d_in, d_hid, d_out) = (64usize, 784usize, 256usize, 10usize);
    let pred = pattern(bsz * d_out, 3);
    let target = pattern(bsz * d_out, 4);
    let h1 = pattern(bsz * d_hid, 5);
    let x0 = pattern(bsz * d_in, 6);
    let w2 = pattern(d_hid * d_out, 7);
    let pred_d = upload(&ops, &pred, &[bsz, d_out]);
    let target_d = upload(&ops, &target, &[bsz, d_out]);
    let h1_d = upload(&ops, &h1, &[bsz, d_hid]);
    let x0_d = upload(&ops, &x0, &[bsz, d_in]);
    let w2_d = upload(&ops, &w2, &[d_hid, d_out]);
    let n_dw2 = d_hid * d_out + d_out;
    let n_dw1 = d_in * d_hid + d_hid;
    let mut dw2_d = upload(&ops, &vec![0.0f32; n_dw2], &[n_dw2]);
    let mut dw1_d = upload(&ops, &vec![0.0f32; n_dw1], &[n_dw1]);
    let scale = 2.0 / (bsz * d_out) as f32;
    let token = DispatchFailureCell::new();

    let h1_shape = [bsz, d_hid];
    let dy2_shape = [bsz, d_out];
    let w2_shape = [d_hid, d_out];
    let din2_shape = [d_hid, bsz];
    let x0_shape = [bsz, d_in];
    let h1_view = DeviceBufferView::new(&h1_d, 0, &h1_shape).unwrap();
    let w2_view = DeviceBufferView::new(&w2_d, 0, &w2_shape).unwrap();
    let x0_view = DeviceBufferView::new(&x0_d, 0, &x0_shape).unwrap();

    // 1 回ぶんを温めて（NVRTC 構築・ワークスペース拡張）同期を済ませる。
    let chain = |ops: &CudaBackendOps,
                 dw2_d: &mut DeviceBuffer<f32>,
                 dw1_d: &mut DeviceBuffer<f32>|
     -> (DeviceBuffer<f32>, DeviceBuffer<f32>, DeviceBuffer<f32>) {
        let dy2_d = ops
            .mse_loss_backward_device_tracked(&pred_d, &target_d, scale, &token)
            .unwrap();
        let dy2_view = DeviceBufferView::new(&dy2_d, 0, &dy2_shape).unwrap();
        ops.gemm_fp32_strict_into_device_tracked(
            h1_view,
            DeviceMatrixLayout::Transposed,
            dy2_view,
            DeviceMatrixLayout::RowMajor,
            dw2_d,
            0,
            Some((d_hid * d_out, d_out)),
            &token,
        )
        .unwrap();
        let din2_d = ops
            .gemm_resident_lhs_device_tracked(
                w2_view,
                dy2_view,
                DeviceMatrixLayout::Transposed,
                &token,
            )
            .unwrap();
        let din2_view = DeviceBufferView::new(&din2_d, 0, &din2_shape).unwrap();
        let dh1_d = ops
            .relu_backward_mask_device_tracked(
                din2_view,
                DeviceMatrixLayout::Transposed,
                h1_view,
                &token,
            )
            .unwrap();
        let dh1_view = DeviceBufferView::new(&dh1_d, 0, &h1_shape).unwrap();
        ops.gemm_fp32_strict_into_device_tracked(
            x0_view,
            DeviceMatrixLayout::Transposed,
            dh1_view,
            DeviceMatrixLayout::RowMajor,
            dw1_d,
            0,
            Some((d_in * d_hid, d_hid)),
            &token,
        )
        .unwrap();
        (dy2_d, din2_d, dh1_d)
    };
    let _ = chain(&ops, &mut dw2_d, &mut dw1_d);
    let _ = download(&ops, &dw1_d);

    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    let (dy2_d, din2_d, dh1_d) = chain(&ops, &mut dw2_d, &mut dw1_d);
    let d = gemm_launch_diag_snapshot().since(before);
    assert_eq!(d.h2d_calls, 0, "chain h2d: {d:?}");
    assert_eq!(d.stream_syncs, 0, "chain sync: {d:?}");
    assert_eq!(d.d2h_calls, 0, "chain d2h: {d:?}");
    assert!(!token.is_set());

    // ホスト版の連鎖結果と比較（download が同期する）。
    let dy2_ref = to_vec(
        &ops.mse_loss_backward(
            &Tensor::new(pred, &dy2_shape).unwrap(),
            &Tensor::new(target, &dy2_shape).unwrap(),
            scale,
        )
        .unwrap(),
    );
    assert_bits_or_nan_class(&download(&ops, &dy2_d), &dy2_ref, "dy2");
    let host_dw = |x: &[f32], g: &[f32], (b, i, o): (usize, usize, usize)| -> Vec<f32> {
        let xt = Tensor::new(x.to_vec(), &[b, i])
            .unwrap()
            .transpose_2d()
            .unwrap();
        let gt = Tensor::new(g.to_vec(), &[b, o]).unwrap();
        let total = i * o + o;
        let mut out = upload(&ops, &vec![0.0f32; total], &[total]);
        ops.gemm_fp32_strict_into_with_bias_reduce_tracked(
            &xt,
            &gt,
            &mut out,
            0,
            Some((i * o, o)),
            &DispatchFailureCell::new(),
        )
        .unwrap();
        download(&ops, &out)
    };
    assert_bits_or_nan_class(
        &download(&ops, &dw2_d),
        &host_dw(&h1, &dy2_ref, (bsz, d_hid, d_out)),
        "dW L2",
    );
    let g_host = Tensor::new(dy2_ref, &dy2_shape)
        .unwrap()
        .transpose_2d()
        .unwrap();
    let din2_ref = to_vec(&ops.gemm_resident_lhs(w2_view, &g_host).unwrap());
    assert_bits_or_nan_class(&download(&ops, &din2_d), &din2_ref, "d_input L2");
    let g_logical = transpose(&din2_ref, d_hid, bsz);
    let dh1_ref: Vec<f32> = g_logical
        .iter()
        .zip(&h1)
        .map(|(&g, &m)| if m > 0.0 { g } else { 0.0 })
        .collect();
    assert_bits_or_nan_class(&download(&ops, &dh1_d), &dh1_ref, "relu mask dh1");
    assert_bits_or_nan_class(
        &download(&ops, &dw1_d),
        &host_dw(&x0, &dh1_ref, (bsz, d_in, d_hid)),
        "dW L1",
    );
}
