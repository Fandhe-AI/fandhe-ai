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

/// [`crate::bce::CHUNK`]（4096）と同値。`autodiff::eval::bce_loss`
/// （ホストフォールバック。`BCE_HOST_FALLBACK_CHUNK`）が同じチャンクサイズ・
/// 同じ縮約順序を使うことを、逆方向（本クレート側から見た固定チャンク
/// 逐次縮約の意図的複製）で突合する（`autodiff` は `backend-cpu` へ
/// 依存できない設計上の不変条件のため cross-crate 呼び出しはしない。
/// PR #1848 codex-review 指摘の再発防止・イシュー #1737）。
const HOST_FALLBACK_REFERENCE_CHUNK: usize = 4096;

/// `eval::bce_elem_loss`（`crates/autodiff/src/eval.rs`）と数式的に同一
/// の要素式（`bce.rs::bce_elem_loss` の意図的複製の複製。モジュール doc
/// 「eval と CPU 実装の意図的複製」節と同型）。本ファイル冒頭の
/// `naive_bce_elem_loss` は Logits 側が旧式（桁落ちしうる）ままのため、
/// ここでは `Unsupported` フォールバック側と同じ現行の数値安定式を使う。
fn stable_bce_elem_loss(input: f32, target: f32, kind: BceKind) -> f32 {
    match kind {
        BceKind::Probabilities => {
            let log_p = input.ln().max(-100.0);
            let log_1mp = (1.0 - input).ln().max(-100.0);
            -(target * log_p + (1.0 - target) * log_1mp)
        }
        _ => {
            if input >= 0.0 {
                (1.0 - target) * input + (-input).exp().ln_1p()
            } else {
                -target * input + input.exp().ln_1p()
            }
        }
    }
}

fn host_fallback_reference_sum(input: &[f32], target: &[f32], kind: BceKind) -> f32 {
    input
        .chunks(HOST_FALLBACK_REFERENCE_CHUNK)
        .zip(target.chunks(HOST_FALLBACK_REFERENCE_CHUNK))
        .map(|(i_chunk, t_chunk)| {
            i_chunk
                .iter()
                .zip(t_chunk.iter())
                .fold(0.0f32, |acc, (&p, &y)| {
                    acc + stable_bce_elem_loss(p, y, kind)
                })
        })
        .fold(0.0f32, |acc, v| acc + v)
}

/// codex-review 指摘（PR #1848）の再発防止回帰テスト: 本クレートの
/// CPU 融合カーネル（`CpuBackendOps::bce_loss` → `bce::bce_sum_f32`）が
/// `HOST_FALLBACK_REFERENCE_CHUNK`（= `bce::CHUNK`／`eval::
/// BCE_HOST_FALLBACK_CHUNK` と同値）の固定チャンク縮約と bit 完全一致
/// することを、チャンク境界を跨ぐ複数の `n`（`CHUNK-1`・`CHUNK`・
/// `2*CHUNK+1`・大入力 `1<<20`）で検証する。
#[test]
fn bce_loss_forward_matches_host_fallback_reduction_bit_exact() {
    let shapes = [
        HOST_FALLBACK_REFERENCE_CHUNK - 1,
        HOST_FALLBACK_REFERENCE_CHUNK,
        2 * HOST_FALLBACK_REFERENCE_CHUNK + 1,
        1usize << 20,
    ];
    for &n in &shapes {
        for kind in [BceKind::Probabilities, BceKind::Logits] {
            let (input_data, target_data): (Vec<f32>, Vec<f32>) = match kind {
                BceKind::Probabilities => (vec![0.5f32; n], vec![0.0f32; n]),
                _ => (vec![0.0f32; n], vec![0.0f32; n]),
            };
            let input = Tensor::new(input_data.clone(), &[n]).unwrap();
            let target = Tensor::new(target_data.clone(), &[n]).unwrap();
            for reduction in [MseReduction::Mean, MseReduction::Sum] {
                let ops = CpuBackendOps::new();
                let got = ops.bce_loss(&input, &target, kind, reduction).unwrap();
                let actual = got.get(&[]).unwrap();

                let sum = host_fallback_reference_sum(&input_data, &target_data, kind);
                let numel = input_data.len();
                let expected = match reduction {
                    MseReduction::Mean => {
                        if numel == 0 {
                            0.0
                        } else {
                            sum / numel as f32
                        }
                    }
                    MseReduction::Sum => sum,
                    _ => sum,
                };
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "bce_loss forward CPU fusion (n={n}, kind={kind:?}, reduction={reduction:?}): \
                     actual={actual} expected={expected} bit 完全一致しない"
                );
            }
        }
    }
}
