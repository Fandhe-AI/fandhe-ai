//! `CpuBackendOps::huber_loss`／`huber_loss_backward`（融合カーネル。
//! イシュー #1739）と素朴な参照実装（本ファイル内 `naive_huber_loss`／
//! `naive_huber_loss_backward`。逐次累積）の数値一致検証
//! （`mse_parity.rs` と同型の構成）。
//!
//! `huber.rs` 側は決定的固定チャンク累積、本ファイルの素朴実装は
//! 単純逐次累積であり丸め手順が異なるため、突合は統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`。相対誤差 1e-3 未満
//! または絶対誤差 1e-5 未満。`.claude/rules/coding-rust.md`）で行う
//! （`mse_parity.rs` と同方針。判定式は唯一の参照点
//! `parity::assert_parity` を再定義しない）。backward は要素独立の map
//! 演算のため丸めの発生源が存在せず `to_bits()` 厳密比較で固定する。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::{BackendOps, HuberKind, MseReduction, Tensor};

/// `huber.rs` 側の融合カーネルと数式的に同一だが丸め手順を分離した
/// 素朴な参照実装（単純逐次累積。`fandhe_ai_autodiff::eval::huber_loss`
/// と同型）。
fn naive_elem_loss(d: f32, kind: HuberKind, delta: f32) -> f32 {
    let abs_d = d.abs();
    match kind {
        HuberKind::Huber => {
            if abs_d < delta {
                0.5 * d * d
            } else {
                delta * (abs_d - 0.5 * delta)
            }
        }
        HuberKind::SmoothL1 => {
            if abs_d < delta {
                0.5 * d * d / delta
            } else {
                abs_d - 0.5 * delta
            }
        }
        _ => 0.0,
    }
}

fn naive_elem_grad(d: f32, kind: HuberKind, delta: f32) -> f32 {
    let abs_d = d.abs();
    match kind {
        HuberKind::Huber => {
            if abs_d < delta {
                d
            } else {
                delta.copysign(d)
            }
        }
        HuberKind::SmoothL1 => {
            if abs_d < delta {
                d / delta
            } else {
                1.0f32.copysign(d)
            }
        }
        _ => 0.0,
    }
}

fn naive_huber_loss(
    pred: &[f32],
    target: &[f32],
    kind: HuberKind,
    delta: f32,
    reduction: MseReduction,
) -> f32 {
    let numel = pred.len();
    if numel == 0 {
        return 0.0;
    }
    let sum: f32 = pred
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| naive_elem_loss(p - t, kind, delta))
        .sum();
    match reduction {
        MseReduction::Mean => sum / numel as f32,
        MseReduction::Sum => sum,
        _ => sum,
    }
}

fn naive_huber_loss_backward(
    pred: &[f32],
    target: &[f32],
    kind: HuberKind,
    delta: f32,
    scale: f32,
) -> Vec<f32> {
    pred.iter()
        .zip(target.iter())
        .map(|(&p, &t)| scale * naive_elem_grad(p - t, kind, delta))
        .collect()
}

/// 形状スイープ: 空・単一要素・`huber.rs::CHUNK`（4096）境界跨ぎ（±1）・
/// 大 n（8193）。`mse_parity.rs::shapes` と同一。
fn shapes() -> Vec<usize> {
    vec![0, 1, 2, 100, 4095, 4096, 4097, 8193]
}

/// `mse_parity.rs::make_inputs` と同一の progression（diff が両分岐
/// 〈二次・線形〉を横断する範囲をカバーする）。
fn make_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let pred: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let target: Vec<f32> = (0..n).map(|i| (i as f32) * 0.005 + 0.5).collect();
    (pred, target)
}

fn kinds_and_deltas() -> Vec<(HuberKind, f32)> {
    vec![
        (HuberKind::Huber, 0.5),
        (HuberKind::Huber, 1.0),
        (HuberKind::Huber, 2.0),
        (HuberKind::SmoothL1, 0.5),
        (HuberKind::SmoothL1, 1.0),
        (HuberKind::SmoothL1, 2.0),
    ]
}

#[test]
fn huber_loss_forward_matches_naive_mean() {
    let ops = CpuBackendOps::new();
    for (kind, delta) in kinds_and_deltas() {
        for n in shapes() {
            let (pred_data, target_data) = make_inputs(n);
            let pred = Tensor::new(pred_data.clone(), &[n]).unwrap();
            let target = Tensor::new(target_data.clone(), &[n]).unwrap();

            let got = ops
                .huber_loss(&pred, &target, kind, delta, MseReduction::Mean)
                .unwrap_or_else(|e| {
                    panic!("huber_loss failed for kind={kind:?} delta={delta} n={n}: {e:?}")
                });
            assert_eq!(got.shape(), &[] as &[usize], "n={n}: 出力 shape はスカラー");

            let expected =
                naive_huber_loss(&pred_data, &target_data, kind, delta, MseReduction::Mean);
            assert_parity(
                &format!("huber_loss forward mean kind={kind:?} delta={delta} n={n}"),
                got.as_slice().unwrap(),
                &[expected],
            );
        }
    }
}

#[test]
fn huber_loss_forward_matches_naive_sum() {
    let ops = CpuBackendOps::new();
    for (kind, delta) in kinds_and_deltas() {
        for n in shapes() {
            let (pred_data, target_data) = make_inputs(n);
            let pred = Tensor::new(pred_data.clone(), &[n]).unwrap();
            let target = Tensor::new(target_data.clone(), &[n]).unwrap();

            let got = ops
                .huber_loss(&pred, &target, kind, delta, MseReduction::Sum)
                .unwrap_or_else(|e| {
                    panic!("huber_loss failed for kind={kind:?} delta={delta} n={n}: {e:?}")
                });
            let expected =
                naive_huber_loss(&pred_data, &target_data, kind, delta, MseReduction::Sum);
            assert_parity(
                &format!("huber_loss forward sum kind={kind:?} delta={delta} n={n}"),
                got.as_slice().unwrap(),
                &[expected],
            );
        }
    }
}

#[test]
fn huber_loss_backward_bit_matches_naive() {
    // backward は要素独立の map 演算のため丸めの発生源が存在せず
    // `to_bits()` 厳密比較で固定する（`mse_parity.rs::
    // mse_loss_backward_bit_matches_naive` と同型）。
    let ops = CpuBackendOps::new();
    for (kind, delta) in kinds_and_deltas() {
        for n in shapes() {
            let (pred_data, target_data) = make_inputs(n);
            let pred = Tensor::new(pred_data.clone(), &[n]).unwrap();
            let target = Tensor::new(target_data.clone(), &[n]).unwrap();
            let scale = 1.7f32;

            let got = ops
                .huber_loss_backward(&pred, &target, kind, delta, scale)
                .unwrap_or_else(|e| {
                    panic!(
                        "huber_loss_backward failed for kind={kind:?} delta={delta} n={n}: {e:?}"
                    )
                });
            assert_eq!(got.shape(), &[n], "n={n}: dpred の shape は pred と一致");

            let expected = naive_huber_loss_backward(&pred_data, &target_data, kind, delta, scale);
            let got_slice = got.as_slice().unwrap();
            assert_eq!(got_slice.len(), expected.len(), "n={n}");
            for i in 0..n {
                assert_eq!(
                    got_slice[i].to_bits(),
                    expected[i].to_bits(),
                    "kind={kind:?} delta={delta} n={n} i={i}: bit mismatch"
                );
            }
        }
    }
}

#[test]
fn huber_loss_rejects_shape_mismatch() {
    use fandhe_ai_tensor_core::device::BackendError;

    let ops = CpuBackendOps::new();
    let pred = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let target = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();

    let forward = ops.huber_loss(&pred, &target, HuberKind::Huber, 1.0, MseReduction::Mean);
    assert!(matches!(forward, Err(BackendError::ShapeMismatch(_))));

    let backward = ops.huber_loss_backward(&pred, &target, HuberKind::Huber, 1.0, 1.0);
    assert!(matches!(backward, Err(BackendError::ShapeMismatch(_))));
}
