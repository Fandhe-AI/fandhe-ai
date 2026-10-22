//! イシュー #3224: 学習ステップ常駐化の Metal 3 部品（ReLU マスク・MSE backward・
//! dW／d_input GEMM）を MLP（784→256→10・batch 64）の backward 部分列として
//! `_tracked` 版だけで連結し、区間全体で `wait_until_completed` が 0 であること、
//! および各部品の出力が比較先（設計 `docs/train-step-resident-parts-design.md` §4。
//! 同じ入力をホスト版へ通した結果）と bit 一致（NaN はクラス一致）することを
//! 実機カウンタ（`__diagnostic_batch_counters_snapshot`）で確かめる。
//!
//! 個々の部品の parity は #3218〜#3220 の各テストが担い、本ファイルは「連結しても
//! 途中で待たない」ことの証跡を担う（常駐チェーン #3248・#3251・#3252 の前提）。
//! 正式な実機記録（RULE.txt・5 run・`.rev-stamp`）は
//! `docs/perf/logs/train-step-resident-parts-parity-3224/` を正とする。
//!
//! 実行: `cargo test -p fandhe-ai-backend-metal --release --test train_step_parts_no_wait_3224 -- --ignored --test-threads=1 --nocapture`

#![cfg(target_os = "macos")]

use fandhe_ai_backend_metal::{__diagnostic_batch_counters_snapshot, MetalBackendOps};
use fandhe_ai_tensor_core::{
    BackendOps, DeviceBuffer, DeviceBufferView, DeviceMatrixLayout, DispatchFailureCell, Tensor,
};

/// 診断カウンタはプロセス共有のため、ファイル内のテストを直列化する。
fn serialize() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Finite,
    /// pred に NaN・±inf を混ぜる。
    Special,
}

fn pattern(n: usize, salt: usize, kind: Kind) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let base = (((i * 31 + salt * 17) % 23) as f32 - 11.0) * 0.37;
            match kind {
                Kind::Finite => base,
                Kind::Special => match (i + salt) % 41 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => f32::NEG_INFINITY,
                    _ => base,
                },
            }
        })
        .collect()
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

fn upload(ops: &MetalBackendOps, data: &[f32], shape: &[usize]) -> DeviceBuffer<f32> {
    ops.memory_ops()
        .expect("memory_ops")
        .upload(&Tensor::new(data.to_vec(), shape).unwrap())
        .expect("upload")
}

fn download(ops: &MetalBackendOps, buf: &DeviceBuffer<f32>) -> Vec<f32> {
    ops.memory_ops()
        .unwrap()
        .download(buf)
        .unwrap()
        .as_slice()
        .unwrap()
        .to_vec()
}

fn to_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.as_slice().map_or_else(
        || t.contiguous().as_slice().unwrap().to_vec(),
        <[f32]>::to_vec,
    )
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

/// ホスト版 dW（bias あり）。出力は `[in*out]` の weight 勾配と `[out]` の bias 勾配を
/// 連結したバッファ（weight は offset 0、bias は `in*out` から）。
fn host_dw(
    ops: &MetalBackendOps,
    x_stored: &[f32],
    g_stored: &[f32],
    (bsz, inf, outf): (usize, usize, usize),
) -> Vec<f32> {
    let x = Tensor::new(x_stored.to_vec(), &[bsz, inf])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let g = Tensor::new(g_stored.to_vec(), &[bsz, outf]).unwrap();
    let total = inf * outf + outf;
    let mut out = upload(ops, &vec![0.0f32; total], &[total]);
    let token = DispatchFailureCell::new();
    ops.gemm_fp32_strict_into_with_bias_reduce_tracked(
        &x,
        &g,
        &mut out,
        0,
        Some((inf * outf, outf)),
        &token,
    )
    .expect("host dW");
    download(ops, &out)
}

fn run_chain(kind: Kind) {
    let ops = MetalBackendOps::new();
    let (bsz, d_in, d_hid, d_out) = (64usize, 784usize, 256usize, 10usize);

    // 入力（すべて先に upload して同期しておく）。
    let pred = pattern(bsz * d_out, 3, kind);
    let target = pattern(bsz * d_out, 4, Kind::Finite);
    let h1 = pattern(bsz * d_hid, 5, Kind::Finite); // L1 出力（ReLU マスク元・L2 の x）
    let x0 = pattern(bsz * d_in, 6, Kind::Finite);
    let w2 = pattern(d_hid * d_out, 7, Kind::Finite);
    let pred_d = upload(&ops, &pred, &[bsz, d_out]);
    let target_d = upload(&ops, &target, &[bsz, d_out]);
    let h1_d = upload(&ops, &h1, &[bsz, d_hid]);
    let x0_d = upload(&ops, &x0, &[bsz, d_in]);
    let w2_d = upload(&ops, &w2, &[d_hid, d_out]);
    let n_dw2 = d_hid * d_out + d_out;
    let n_dw1 = d_in * d_hid + d_hid;
    let mut dw2_d = upload(&ops, &vec![0.0f32; n_dw2], &[n_dw2]);
    let mut dw1_d = upload(&ops, &vec![0.0f32; n_dw1], &[n_dw1]);
    // 先行する GPU 仕事を同期しておく（open バッチを残さない）。
    let _ = download(&ops, &w2_d);

    let scale = 2.0 / (bsz * d_out) as f32;
    let token = DispatchFailureCell::new();
    let before = __diagnostic_batch_counters_snapshot().unwrap();

    // 1) MSE backward → dy2 [64, 10]
    let dy2_d = ops
        .mse_loss_backward_device_tracked(&pred_d, &target_d, scale, &token)
        .unwrap();
    // 2) dW L2（TN、(B,in,out) = (64,256,10)、bias あり）
    let h1_shape = [bsz, d_hid];
    let h1_view = DeviceBufferView::new(&h1_d, 0, &h1_shape).unwrap();
    let dy2_shape = [bsz, d_out];
    let dy2_view = DeviceBufferView::new(&dy2_d, 0, &dy2_shape).unwrap();
    ops.gemm_fp32_strict_into_device_tracked(
        h1_view,
        DeviceMatrixLayout::Transposed,
        dy2_view,
        DeviceMatrixLayout::RowMajor,
        &mut dw2_d,
        0,
        Some((d_hid * d_out, d_out)),
        &token,
    )
    .unwrap();
    // 3) d_input L2 → [256, 64]（転置格納）
    let w2_shape = [d_hid, d_out];
    let w2_view = DeviceBufferView::new(&w2_d, 0, &w2_shape).unwrap();
    let din2_d = ops
        .gemm_resident_lhs_device_tracked(w2_view, dy2_view, DeviceMatrixLayout::Transposed, &token)
        .unwrap();
    // 4) ReLU マスク（g は転置格納 [256, 64]、論理 [64, 256]）
    let din2_shape = [d_hid, bsz];
    let din2_view = DeviceBufferView::new(&din2_d, 0, &din2_shape).unwrap();
    let dh1_d = ops
        .relu_backward_mask_device_tracked(
            din2_view,
            DeviceMatrixLayout::Transposed,
            h1_view,
            &token,
        )
        .unwrap();
    // 5) dW L1（TN、(B,in,out) = (64,784,256)、bias あり）
    let x0_shape = [bsz, d_in];
    let x0_view = DeviceBufferView::new(&x0_d, 0, &x0_shape).unwrap();
    let dh1_shape = [bsz, d_hid];
    let dh1_view = DeviceBufferView::new(&dh1_d, 0, &dh1_shape).unwrap();
    ops.gemm_fp32_strict_into_device_tracked(
        x0_view,
        DeviceMatrixLayout::Transposed,
        dh1_view,
        DeviceMatrixLayout::RowMajor,
        &mut dw1_d,
        0,
        Some((d_in * d_hid, d_hid)),
        &token,
    )
    .unwrap();

    let after = __diagnostic_batch_counters_snapshot().unwrap();
    assert_eq!(
        after.wait_until_completed - before.wait_until_completed,
        0,
        "{kind:?}: tracked 連結区間は wait を出さない"
    );
    assert!(!token.is_set(), "{kind:?}: token");

    // ここから 1 回だけ download（内部同期）して、ホスト版の連鎖結果と比較する。
    let dy2_ref = to_vec(
        &ops.mse_loss_backward(
            &Tensor::new(pred.clone(), &[bsz, d_out]).unwrap(),
            &Tensor::new(target.clone(), &[bsz, d_out]).unwrap(),
            scale,
        )
        .unwrap(),
    );
    assert_bits_or_nan_class(&download(&ops, &dy2_d), &dy2_ref, "mse backward dy2");

    let dw2_ref = host_dw(&ops, &h1, &dy2_ref, (bsz, d_hid, d_out));
    assert_bits_or_nan_class(&download(&ops, &dw2_d), &dw2_ref, "dW L2");

    let g_host = Tensor::new(dy2_ref.clone(), &[bsz, d_out])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let din2_ref = to_vec(&ops.gemm_resident_lhs(w2_view, &g_host).unwrap());
    assert_bits_or_nan_class(&download(&ops, &din2_d), &din2_ref, "d_input L2");

    // ReLU マスクの参照: 論理 g = din2 の転置、`m > 0 ? g : 0`。
    let g_logical = transpose(&din2_ref, d_hid, bsz);
    let dh1_ref: Vec<f32> = g_logical
        .iter()
        .zip(&h1)
        .map(|(&g, &m)| if m > 0.0 { g } else { 0.0 })
        .collect();
    assert_bits_or_nan_class(&download(&ops, &dh1_d), &dh1_ref, "relu mask dh1");

    let dw1_ref = host_dw(&ops, &x0, &dh1_ref, (bsz, d_in, d_hid));
    assert_bits_or_nan_class(&download(&ops, &dw1_d), &dw1_ref, "dW L1");
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn mlp_backward_parts_tracked_chain_issues_no_wait() {
    let _g = serialize();
    run_chain(Kind::Finite);
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn mlp_backward_parts_tracked_chain_with_nan_inf_issues_no_wait() {
    let _g = serialize();
    run_chain(Kind::Special);
}
