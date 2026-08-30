//! `CpuBackendOps::linear_forward_device` の受け入れ基準対応テスト
//! （イシュー #1028・`docs/inference-forward-fixed-cost-design.md`）。
//!
//! `fandhe_ai_tensor_core::BackendOps` に非破壊追加した
//! `linear_forward_device`（デフォルトは `BackendError::Unsupported`。
//! `crates/tensor-core/src/backend_ops.rs`）の CPU 実装が、旧経路
//! （`Sequential::predict` 等が `tape.ops()` 経由で呼ぶ非融合 `gemm` →
//! `add`（bias 行方向複製）→ `relu` の 3 段合成。`fandhe_ai_autodiff::
//! nn::linear::LinearVars::forward` と同型）と **bit 完全一致**すること
//! を検証する（§3.3 (b) の bit-exactness 契約）。`a`／`w`／`bias`／
//! 戻り値をいずれもデバイス常駐バッファへ upload/download してから
//! 呼ぶ点のみが異なり、数式は同一のため REQ-2 統一複合判定ではなく
//! bit 完全一致で比較する（`gemm_resident_parity.rs` と同じ方針）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{CpuBackendOps, CpuMemory};
use fandhe_ai_tensor_core::buffer::{DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::{Activation, BackendError, BackendOps, Tensor};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// 旧経路（`gemm` → `add` → `act`。`LinearVars::forward` + 活性化層と
/// 同型の非融合合成）をホスト常駐 `Tensor` で計算する参照実装。
fn old_path(
    ops: &CpuBackendOps,
    a: &Tensor<f32>,
    w: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    act: Activation,
) -> Tensor<f32> {
    let mut y = ops.gemm(a, w).unwrap();
    if let Some(bias) = bias {
        y = ops.add(&y, bias).unwrap();
    }
    y = match act {
        Activation::None => y,
        Activation::Relu => ops.relu(&y).unwrap(),
        _ => unreachable!("test only exercises None/Relu"),
    };
    y
}

/// `linear_forward_device` が旧経路（`gemm` → `add` → `act`）と
/// bit 完全一致することを、MR/NR/MC/KC/NC 境界を跨ぐ複数形状・
/// bias 有無・activation（None/Relu）の組み合わせで確認する。
#[test]
fn linear_forward_device_matches_old_path_across_shapes() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();

    for &(m, k, n) in &[(1, 1, 1), (4, 8, 4), (37, 65, 33), (128, 129, 96)] {
        for has_bias in [false, true] {
            for act in [Activation::None, Activation::Relu] {
                // ReLU が層の出力符号に対して恒等に近くなる境界ケースを
                // 減らすため、bias 込みの分布を中心 0 付近からわずかに
                // ずらす（`Xorshift64Star::fill_vec` は [0,1) 一様分布の
                // ため、そのままでは gemm 出力が正に偏り ReLU が全恒等に
                // なりがちで parity 検証として弱くなる）。
                let a_raw = random_matrix(0x1000 + m as u64, m * k);
                let a_data: Vec<f32> = a_raw.iter().map(|v| v - 0.5).collect();
                let w_raw = random_matrix(0x2000 + n as u64, k * n);
                let w_data: Vec<f32> = w_raw.iter().map(|v| v - 0.5).collect();
                let a = tensor(a_data, &[m, k]);
                let w = tensor(w_data, &[k, n]);
                let bias = has_bias.then(|| {
                    let raw = random_matrix(0x3000 + n as u64, n);
                    tensor(raw.iter().map(|v| v - 0.5).collect(), &[n])
                });

                let expected = old_path(&ops, &a, &w, bias.as_ref(), act);

                let a_dev = mem.upload(&a).unwrap();
                let w_dev = mem.upload(&w).unwrap();
                let w_shape = [k, n];
                let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
                let bias_dev = bias.as_ref().map(|b| mem.upload(b).unwrap());
                let bias_shape = [n];
                let bias_view = bias_dev
                    .as_ref()
                    .map(|buf| DeviceBufferView::new(buf, 0, &bias_shape).unwrap());

                let actual_dev = ops
                    .linear_forward_device(&a_dev, w_view, bias_view, act)
                    .unwrap();
                let actual = mem.download(&actual_dev).unwrap();

                assert_eq!(
                    actual.shape(),
                    expected.shape(),
                    "m={m} k={k} n={n} has_bias={has_bias} act={act:?}"
                );
                let a_slice = actual.contiguous();
                let e_slice = expected.contiguous();
                assert_eq!(
                    a_slice.as_slice().unwrap(),
                    e_slice.as_slice().unwrap(),
                    "linear_forward_device は旧経路（gemm→add→act）と bit 完全一致するはず \
                     （m={m} k={k} n={n} has_bias={has_bias} act={act:?}）"
                );
            }
        }
    }
}

/// `w`（デバイス常駐）の shape が `a` の列数と噛み合わない場合は
/// カーネル本体へ触れる前に `ShapeMismatch` を返す（REQ-8・OWASP A03）。
#[test]
fn linear_forward_device_rejects_incompatible_shapes() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let a = tensor(vec![1.0, 2.0, 3.0], &[1, 3]);
    let w = tensor(vec![1.0, 2.0], &[2, 1]);
    let a_dev = mem.upload(&a).unwrap();
    let w_dev = mem.upload(&w).unwrap();
    let w_shape = [2, 1];
    let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();

    let err = ops
        .linear_forward_device(&a_dev, w_view, None, Activation::None)
        .unwrap_err();

    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// `bias` の shape が `[n]` 厳密一致でない場合も同様に拒否する。
#[test]
fn linear_forward_device_rejects_bias_shape_mismatch() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let a = tensor(vec![1.0, 2.0], &[1, 2]);
    let w = tensor(vec![1.0, 2.0], &[2, 1]);
    let bias = tensor(vec![1.0, 2.0], &[2]); // n=1 のはずが [2]

    let a_dev = mem.upload(&a).unwrap();
    let w_dev = mem.upload(&w).unwrap();
    let w_shape = [2, 1];
    let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
    let bias_dev = mem.upload(&bias).unwrap();
    let bias_shape = [2];
    let bias_view = DeviceBufferView::new(&bias_dev, 0, &bias_shape).unwrap();

    let err = ops
        .linear_forward_device(&a_dev, w_view, Some(bias_view), Activation::None)
        .unwrap_err();

    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}
