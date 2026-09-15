//! `CpuBackendOps::kl_div_loss`／`kl_div_loss_backward`（融合カーネル。
//! イシュー #1738）と素朴な参照実装（本ファイル内 `naive_kl_div_loss`／
//! `naive_kl_div_loss_backward`）の数値一致検証（`mse_parity.rs` と
//! 同型構成）。
//!
//! 突合は統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`。
//! 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`.claude/rules/
//! coding-rust.md`）で行う。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::{BackendOps, KlDivTarget, MseReduction, Tensor};

/// `kl_div.rs::kl_div_sum_f32` と数式的に同一だが丸め手順を分離した
/// 素朴な参照実装（`fandhe_ai_autodiff::eval::kl_div_elem_loss` と
/// 同型）。
fn naive_kl_div_elem(x: f32, tv: f32, kind: KlDivTarget) -> f32 {
    match kind {
        KlDivTarget::Probabilities => {
            if tv == 0.0 {
                0.0
            } else {
                tv * (tv.ln() - x)
            }
        }
        KlDivTarget::LogProbabilities => tv.exp() * (tv - x),
        _ => tv.exp() * (tv - x),
    }
}

fn naive_kl_div_loss(
    input: &[f32],
    target: &[f32],
    kind: KlDivTarget,
    reduction: MseReduction,
) -> f32 {
    let numel = input.len();
    if numel == 0 {
        return 0.0;
    }
    let sum: f32 = input
        .iter()
        .zip(target.iter())
        .map(|(&x, &tv)| naive_kl_div_elem(x, tv, kind))
        .sum();
    match reduction {
        MseReduction::Mean => sum / numel as f32,
        MseReduction::Sum => sum,
        _ => sum,
    }
}

fn naive_kl_div_grad_input(_x: f32, tv: f32, kind: KlDivTarget) -> f32 {
    match kind {
        KlDivTarget::Probabilities => -tv,
        KlDivTarget::LogProbabilities => -tv.exp(),
        _ => -tv.exp(),
    }
}

fn naive_kl_div_loss_backward(
    input: &[f32],
    target: &[f32],
    kind: KlDivTarget,
    scale: f32,
) -> Vec<f32> {
    input
        .iter()
        .zip(target.iter())
        .map(|(&x, &tv)| scale * naive_kl_div_grad_input(x, tv, kind))
        .collect()
}

/// 形状スイープ: 空・単一要素・`kl_div.rs::CHUNK`（4096）境界跨ぎ（±1）・
/// 大 n（8193）。
fn shapes() -> Vec<usize> {
    vec![0, 1, 2, 100, 4095, 4096, 4097, 8193]
}

/// `input` は log 確率想定（負値中心）、`target` は確率（`(0, 1)`
/// 開区間。`target == 0` の分岐は別テストで単独確認する）。
fn make_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let input: Vec<f32> = (0..n).map(|i| -0.5 - (i as f32) * 0.0003).collect();
    let target: Vec<f32> = (0..n).map(|i| 0.1 + ((i % 7) as f32) * 0.1).collect();
    (input, target)
}

#[test]
fn kl_div_loss_forward_matches_naive_mean_probabilities() {
    let ops = CpuBackendOps::new();
    for n in shapes() {
        let (input_data, target_data) = make_inputs(n);
        let input = Tensor::new(input_data.clone(), &[n]).unwrap();
        let target = Tensor::new(target_data.clone(), &[n]).unwrap();

        let got = ops
            .kl_div_loss(
                &input,
                &target,
                KlDivTarget::Probabilities,
                MseReduction::Mean,
            )
            .unwrap_or_else(|e| panic!("kl_div_loss failed for n={n}: {e:?}"));
        assert_eq!(got.shape(), &[] as &[usize]);

        let expected = naive_kl_div_loss(
            &input_data,
            &target_data,
            KlDivTarget::Probabilities,
            MseReduction::Mean,
        );
        assert_parity(
            &format!("kl_div_loss forward mean(probabilities) n={n}"),
            got.as_slice().unwrap(),
            &[expected],
        );
    }
}

#[test]
fn kl_div_loss_forward_matches_naive_sum_probabilities() {
    let ops = CpuBackendOps::new();
    for n in shapes() {
        let (input_data, target_data) = make_inputs(n);
        let input = Tensor::new(input_data.clone(), &[n]).unwrap();
        let target = Tensor::new(target_data.clone(), &[n]).unwrap();

        let got = ops
            .kl_div_loss(
                &input,
                &target,
                KlDivTarget::Probabilities,
                MseReduction::Sum,
            )
            .unwrap_or_else(|e| panic!("kl_div_loss failed for n={n}: {e:?}"));
        let expected = naive_kl_div_loss(
            &input_data,
            &target_data,
            KlDivTarget::Probabilities,
            MseReduction::Sum,
        );
        assert_parity(
            &format!("kl_div_loss forward sum(probabilities) n={n}"),
            got.as_slice().unwrap(),
            &[expected],
        );
    }
}

#[test]
fn kl_div_loss_forward_matches_naive_log_target() {
    let ops = CpuBackendOps::new();
    for n in [0usize, 1, 2, 4096, 4097] {
        let input_data: Vec<f32> = (0..n).map(|i| -0.5 - (i as f32) * 0.0003).collect();
        let target_data: Vec<f32> = (0..n).map(|i| -0.3 - (i as f32) * 0.0002).collect();
        let input = Tensor::new(input_data.clone(), &[n]).unwrap();
        let target = Tensor::new(target_data.clone(), &[n]).unwrap();

        let got = ops
            .kl_div_loss(
                &input,
                &target,
                KlDivTarget::LogProbabilities,
                MseReduction::Mean,
            )
            .unwrap_or_else(|e| panic!("kl_div_loss failed for n={n}: {e:?}"));
        let expected = naive_kl_div_loss(
            &input_data,
            &target_data,
            KlDivTarget::LogProbabilities,
            MseReduction::Mean,
        );
        assert_parity(
            &format!("kl_div_loss forward mean(log_target) n={n}"),
            got.as_slice().unwrap(),
            &[expected],
        );
    }
}

#[test]
fn kl_div_loss_backward_matches_naive() {
    let ops = CpuBackendOps::new();
    for n in shapes() {
        let (input_data, target_data) = make_inputs(n);
        let input = Tensor::new(input_data.clone(), &[n]).unwrap();
        let target = Tensor::new(target_data.clone(), &[n]).unwrap();
        let scale = 1.7f32;

        let got = ops
            .kl_div_loss_backward(&input, &target, KlDivTarget::Probabilities, scale)
            .unwrap_or_else(|e| panic!("kl_div_loss_backward failed for n={n}: {e:?}"));
        assert_eq!(got.shape(), &[n]);

        let expected = naive_kl_div_loss_backward(
            &input_data,
            &target_data,
            KlDivTarget::Probabilities,
            scale,
        );
        assert_parity(
            &format!("kl_div_loss backward(probabilities) n={n}"),
            got.as_slice().unwrap(),
            &expected,
        );
    }
}

#[test]
fn kl_div_loss_target_zero_contributes_zero_and_backward_matches() {
    let ops = CpuBackendOps::new();
    let input = Tensor::new(vec![-0.5, -1.0, -0.3], &[3]).unwrap();
    let target = Tensor::new(vec![0.0, 0.4, 0.0], &[3]).unwrap();

    let loss = ops
        .kl_div_loss(
            &input,
            &target,
            KlDivTarget::Probabilities,
            MseReduction::Sum,
        )
        .unwrap();
    let expected = 0.4f32 * (0.4f32.ln() - (-1.0f32));
    assert_parity(
        "kl_div_loss target_zero forward",
        loss.as_slice().unwrap(),
        &[expected],
    );

    let dinput = ops
        .kl_div_loss_backward(&input, &target, KlDivTarget::Probabilities, 1.0)
        .unwrap();
    let expected_dinput = [0.0f32, -0.4, 0.0];
    assert_parity(
        "kl_div_loss target_zero backward",
        dinput.as_slice().unwrap(),
        &expected_dinput,
    );
}

#[test]
fn kl_div_loss_rejects_shape_mismatch() {
    use fandhe_ai_tensor_core::device::BackendError;

    let ops = CpuBackendOps::new();
    let input = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let target = Tensor::new(vec![0.1, 0.2], &[2]).unwrap();

    let forward = ops.kl_div_loss(
        &input,
        &target,
        KlDivTarget::Probabilities,
        MseReduction::Mean,
    );
    assert!(matches!(forward, Err(BackendError::ShapeMismatch(_))));

    let backward = ops.kl_div_loss_backward(&input, &target, KlDivTarget::Probabilities, 1.0);
    assert!(matches!(backward, Err(BackendError::ShapeMismatch(_))));
}
