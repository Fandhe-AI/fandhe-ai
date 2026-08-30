//! `CpuBackendOps::mse_loss`／`mse_loss_backward`（融合カーネル。イシュー
//! #1045）と素朴な参照実装（本ファイル内 `naive_mse_loss`／
//! `naive_mse_loss_backward`。逐次 `diff*diff` 累積・`mul_add` 不使用）の
//! 数値一致検証。
//!
//! `mse.rs` 側は決定的固定チャンク＋`mul_add` 累積、本ファイルの素朴実装は
//! 単純逐次累積であり丸め手順が異なるため、突合は統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`。相対誤差 1e-3 未満
//! または絶対誤差 1e-5 未満。`.claude/rules/coding-rust.md`）で行う
//! （`sgd_device_parity.rs`・`tests/fma_contract.rs` と同方針。判定式は
//! 唯一の参照点 `parity::assert_parity` を再定義しない）。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::{BackendOps, MseReduction, Tensor};

/// `mse.rs::mse_loss_f32` と数式的に同一だが、丸め手順を分離した素朴な
/// 参照実装（単純逐次累積・`mul_add` 不使用。`fandhe_ai_autodiff::eval::
/// mse_loss` と同型）。
fn naive_mse_loss(pred: &[f32], target: &[f32], reduction: MseReduction) -> f32 {
    let numel = pred.len();
    if numel == 0 {
        return 0.0;
    }
    let sum_sq: f32 = pred
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| {
            let diff = p - t;
            diff * diff
        })
        .sum();
    match reduction {
        MseReduction::Mean => sum_sq / numel as f32,
        MseReduction::Sum => sum_sq,
        _ => sum_sq,
    }
}

/// `mse.rs::mse_loss_backward_f32` と数式的に同一の素朴参照実装。
fn naive_mse_loss_backward(pred: &[f32], target: &[f32], scale: f32) -> Vec<f32> {
    pred.iter()
        .zip(target.iter())
        .map(|(&p, &t)| scale * (p - t))
        .collect()
}

/// 形状スイープ: 空・単一要素・`mse.rs::CHUNK`（4096）境界跨ぎ（±1）・
/// 大 n（8193）。CPU 融合カーネルの決定性契約（`mse.rs` モジュール doc）
/// が対象とするチャンク境界を forward/backward 両方でカバーする。
fn shapes() -> Vec<usize> {
    vec![0, 1, 2, 100, 4095, 4096, 4097, 8193]
}

fn make_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let pred: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let target: Vec<f32> = (0..n).map(|i| (i as f32) * 0.005 + 0.5).collect();
    (pred, target)
}

#[test]
fn mse_loss_forward_matches_naive_mean() {
    let ops = CpuBackendOps::new();
    for n in shapes() {
        let (pred_data, target_data) = make_inputs(n);
        let pred = Tensor::new(pred_data.clone(), &[n]).unwrap();
        let target = Tensor::new(target_data.clone(), &[n]).unwrap();

        let got = ops
            .mse_loss(&pred, &target, MseReduction::Mean)
            .unwrap_or_else(|e| panic!("mse_loss failed for n={n}: {e:?}"));
        assert_eq!(got.shape(), &[] as &[usize], "n={n}: 出力 shape はスカラー");

        let expected = naive_mse_loss(&pred_data, &target_data, MseReduction::Mean);
        assert_parity(
            &format!("mse_loss forward mean n={n}"),
            got.as_slice().unwrap(),
            &[expected],
        );
    }
}

#[test]
fn mse_loss_forward_matches_naive_sum() {
    let ops = CpuBackendOps::new();
    for n in shapes() {
        let (pred_data, target_data) = make_inputs(n);
        let pred = Tensor::new(pred_data.clone(), &[n]).unwrap();
        let target = Tensor::new(target_data.clone(), &[n]).unwrap();

        let got = ops
            .mse_loss(&pred, &target, MseReduction::Sum)
            .unwrap_or_else(|e| panic!("mse_loss failed for n={n}: {e:?}"));
        let expected = naive_mse_loss(&pred_data, &target_data, MseReduction::Sum);
        assert_parity(
            &format!("mse_loss forward sum n={n}"),
            got.as_slice().unwrap(),
            &[expected],
        );
    }
}

#[test]
fn mse_loss_backward_matches_naive() {
    let ops = CpuBackendOps::new();
    for n in shapes() {
        let (pred_data, target_data) = make_inputs(n);
        let pred = Tensor::new(pred_data.clone(), &[n]).unwrap();
        let target = Tensor::new(target_data.clone(), &[n]).unwrap();
        let scale = 1.7f32;

        let got = ops
            .mse_loss_backward(&pred, &target, scale)
            .unwrap_or_else(|e| panic!("mse_loss_backward failed for n={n}: {e:?}"));
        assert_eq!(got.shape(), &[n], "n={n}: dpred の shape は pred と一致");

        let expected = naive_mse_loss_backward(&pred_data, &target_data, scale);
        assert_parity(
            &format!("mse_loss backward n={n}"),
            got.as_slice().unwrap(),
            &expected,
        );
    }
}

#[test]
fn mse_loss_rejects_shape_mismatch() {
    use fandhe_ai_tensor_core::device::BackendError;

    let ops = CpuBackendOps::new();
    let pred = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let target = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();

    let forward = ops.mse_loss(&pred, &target, MseReduction::Mean);
    assert!(matches!(forward, Err(BackendError::ShapeMismatch(_))));

    let backward = ops.mse_loss_backward(&pred, &target, 1.0);
    assert!(matches!(backward, Err(BackendError::ShapeMismatch(_))));
}
