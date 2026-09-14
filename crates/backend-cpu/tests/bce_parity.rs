//! `CpuBackendOps::bce_loss`／`bce_loss_backward`（融合カーネル。イシュー
//! #1737）と素朴な参照実装（本ファイル内 `naive_bce_loss`／
//! `naive_bce_loss_backward`。逐次 `f32` 累積）の数値一致検証。
//!
//! `bce.rs` 側は決定的固定チャンク累積、本ファイルの素朴実装は単純逐次
//! 累積であり丸め手順が異なるため、突合は統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`。相対誤差 1e-3 未満
//! または絶対誤差 1e-5 未満。`.claude/rules/coding-rust.md`）で行う
//! （`mse_parity.rs` と同方針。判定式は唯一の参照点 `parity::
//! assert_parity` を再定義しない）。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::{BackendOps, BceKind, MseReduction, Tensor};

/// `bce.rs` の要素式と数式的に同一の素朴参照実装（単純逐次累積。
/// `fandhe_ai_autodiff::eval::bce_elem_loss` と同型）。
fn naive_bce_elem_loss(input: f32, target: f32, kind: BceKind) -> f32 {
    match kind {
        BceKind::Probabilities => {
            let log_p = input.ln().max(-100.0);
            let log_1mp = (1.0 - input).ln().max(-100.0);
            -(target * log_p + (1.0 - target) * log_1mp)
        }
        _ => input.max(0.0) - input * target + (-input.abs()).exp().ln_1p(),
    }
}

fn naive_bce_elem_grad_input(input: f32, target: f32, kind: BceKind) -> f32 {
    match kind {
        BceKind::Probabilities => {
            let denom = (input * (1.0 - input)).max(1e-12);
            (input - target) / denom
        }
        _ => {
            let sigmoid = if input >= 0.0 {
                1.0 / (1.0 + (-input).exp())
            } else {
                let e = input.exp();
                e / (1.0 + e)
            };
            sigmoid - target
        }
    }
}

fn naive_bce_loss(input: &[f32], target: &[f32], kind: BceKind, reduction: MseReduction) -> f32 {
    let numel = input.len();
    if numel == 0 {
        return 0.0;
    }
    let sum: f32 = input
        .iter()
        .zip(target.iter())
        .map(|(&p, &y)| naive_bce_elem_loss(p, y, kind))
        .sum();
    match reduction {
        MseReduction::Mean => sum / numel as f32,
        MseReduction::Sum => sum,
        _ => sum,
    }
}

fn naive_bce_loss_backward(input: &[f32], target: &[f32], kind: BceKind, scale: f32) -> Vec<f32> {
    input
        .iter()
        .zip(target.iter())
        .map(|(&p, &y)| scale * naive_bce_elem_grad_input(p, y, kind))
        .collect()
}

/// 形状スイープ: 空・単一要素・`bce.rs::CHUNK`（4096）境界跨ぎ（±1）・
/// 大 n（8193）。`mse_parity.rs::shapes` と同型。
fn shapes() -> Vec<usize> {
    vec![0, 1, 2, 100, 4095, 4096, 4097, 8193]
}

/// `Probabilities`（`kind`）向けの決定的入力（`(0, 1)` 開区間に収める。
/// `n == 0` は空配列）。
fn make_probabilities_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let input: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 + 1.0) / 99.0).collect();
    let target: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    (input, target)
}

/// `Logits`（`kind`）向けの決定的入力（範囲制約なし。負値・大きな正値を
/// 含む）。
fn make_logits_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 4.0).collect();
    let target: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
    (input, target)
}

#[test]
fn bce_loss_forward_probabilities_matches_naive_mean() {
    let ops = CpuBackendOps::new();
    for n in shapes() {
        let (input_data, target_data) = make_probabilities_inputs(n);
        let input = Tensor::new(input_data.clone(), &[n]).unwrap();
        let target = Tensor::new(target_data.clone(), &[n]).unwrap();

        let got = ops
            .bce_loss(&input, &target, BceKind::Probabilities, MseReduction::Mean)
            .unwrap_or_else(|e| panic!("bce_loss failed for n={n}: {e:?}"));
        assert_eq!(got.shape(), &[] as &[usize], "n={n}: 出力 shape はスカラー");

        let expected = naive_bce_loss(
            &input_data,
            &target_data,
            BceKind::Probabilities,
            MseReduction::Mean,
        );
        assert_parity(
            &format!("bce_loss forward probabilities mean n={n}"),
            got.as_slice().unwrap(),
            &[expected],
        );
    }
}

#[test]
fn bce_loss_forward_probabilities_matches_naive_sum() {
    let ops = CpuBackendOps::new();
    for n in shapes() {
        let (input_data, target_data) = make_probabilities_inputs(n);
        let input = Tensor::new(input_data.clone(), &[n]).unwrap();
        let target = Tensor::new(target_data.clone(), &[n]).unwrap();

        let got = ops
            .bce_loss(&input, &target, BceKind::Probabilities, MseReduction::Sum)
            .unwrap_or_else(|e| panic!("bce_loss failed for n={n}: {e:?}"));
        let expected = naive_bce_loss(
            &input_data,
            &target_data,
            BceKind::Probabilities,
            MseReduction::Sum,
        );
        assert_parity(
            &format!("bce_loss forward probabilities sum n={n}"),
            got.as_slice().unwrap(),
            &[expected],
        );
    }
}

#[test]
fn bce_loss_forward_logits_matches_naive_mean() {
    let ops = CpuBackendOps::new();
    for n in shapes() {
        let (input_data, target_data) = make_logits_inputs(n);
        let input = Tensor::new(input_data.clone(), &[n]).unwrap();
        let target = Tensor::new(target_data.clone(), &[n]).unwrap();

        let got = ops
            .bce_loss(&input, &target, BceKind::Logits, MseReduction::Mean)
            .unwrap_or_else(|e| panic!("bce_loss failed for n={n}: {e:?}"));
        let expected = naive_bce_loss(
            &input_data,
            &target_data,
            BceKind::Logits,
            MseReduction::Mean,
        );
        assert_parity(
            &format!("bce_loss forward logits mean n={n}"),
            got.as_slice().unwrap(),
            &[expected],
        );
    }
}

#[test]
fn bce_loss_backward_probabilities_matches_naive() {
    let ops = CpuBackendOps::new();
    for n in shapes() {
        let (input_data, target_data) = make_probabilities_inputs(n);
        let input = Tensor::new(input_data.clone(), &[n]).unwrap();
        let target = Tensor::new(target_data.clone(), &[n]).unwrap();
        let scale = 1.7f32;

        let got = ops
            .bce_loss_backward(&input, &target, BceKind::Probabilities, scale)
            .unwrap_or_else(|e| panic!("bce_loss_backward failed for n={n}: {e:?}"));
        assert_eq!(got.shape(), &[n], "n={n}: dinput の shape は input と一致");

        let expected =
            naive_bce_loss_backward(&input_data, &target_data, BceKind::Probabilities, scale);
        assert_parity(
            &format!("bce_loss backward probabilities n={n}"),
            got.as_slice().unwrap(),
            &expected,
        );
    }
}

#[test]
fn bce_loss_backward_logits_matches_naive() {
    let ops = CpuBackendOps::new();
    for n in shapes() {
        let (input_data, target_data) = make_logits_inputs(n);
        let input = Tensor::new(input_data.clone(), &[n]).unwrap();
        let target = Tensor::new(target_data.clone(), &[n]).unwrap();
        let scale = -0.5f32;

        let got = ops
            .bce_loss_backward(&input, &target, BceKind::Logits, scale)
            .unwrap_or_else(|e| panic!("bce_loss_backward failed for n={n}: {e:?}"));
        let expected = naive_bce_loss_backward(&input_data, &target_data, BceKind::Logits, scale);
        assert_parity(
            &format!("bce_loss backward logits n={n}"),
            got.as_slice().unwrap(),
            &expected,
        );
    }
}

#[test]
fn bce_loss_rejects_shape_mismatch() {
    use fandhe_ai_tensor_core::device::BackendError;

    let ops = CpuBackendOps::new();
    let input = Tensor::new(vec![0.2, 0.5, 0.8], &[3]).unwrap();
    let target = Tensor::new(vec![0.0, 1.0], &[2]).unwrap();

    let forward = ops.bce_loss(&input, &target, BceKind::Probabilities, MseReduction::Mean);
    assert!(matches!(forward, Err(BackendError::ShapeMismatch(_))));

    let backward = ops.bce_loss_backward(&input, &target, BceKind::Probabilities, 1.0);
    assert!(matches!(backward, Err(BackendError::ShapeMismatch(_))));
}
