//! `fandhe_ai_autodiff::margin_focal_loss_ops`（イシュー #2653。facade 非公開のため本テストは
//! `fandhe_ai_autodiff::margin_focal_loss_ops::*` を直接 use する。facade の dev 依存に
//! `fandhe-ai-autodiff` が既に含まれている。`crates/autodiff/src/margin_focal_loss_ops.rs` モジュール doc 参照）の
//! バックエンド間 parity テスト（`elementwise_loss_ops_backend_parity.rs` と同型）。
//!
//! 本ファイルの契約は MultiMargin・MultiLabelMargin・MultiLabelSoftMargin・sigmoid focal loss の各演算 ×
//! {forward, backward} × {CPU vs NaiveOps, CUDA vs CPU, Metal vs CPU} の各セルを埋めることであり、各テストは
//! 全演算を網羅する（代表 1 演算での省略をしない。#2146 の教訓）。
//!
//! **判定方式**: 新規 Op 4 種はすべてホスト参照実装（`eval::margin_focal_loss`）へ常にフォールバックし、
//! どのバックエンドの tape からも同一コードで計算されるため、**bit 完全一致**で判定する
//! （REQ-2 統一複合判定を使う経路はない）。
//!
//! - 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`〉と `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`〉）:
//!   `cpu_bit_exact_forward_matches_naive_reference`・`cpu_bit_exact_backward_matches_naive_reference`
//! - `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os = "macos")` 限定〉
//!   で同じ経路を CPU tape と比較）: 上記 2 種を `cuda_*`／`metal_*` の接頭辞で対称に置く（計 4 件）。
//!   新規 GPU カーネルは無く、ホスト経路の bit 一致確認である。実機への到達手段が本エージェント実行環境に
//!   ないため未実施のまま申し送る（`docs/perf/logs/margin-focal-loss-ops-2653/README.md`）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Reduction;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::margin_focal_loss_ops::{
    MultiLabelSoftMarginOptions, MultiMarginOptions, SigmoidFocalLossOptions, multi_margin_loss,
    multilabel_margin_loss, multilabel_soft_margin_loss, sigmoid_focal_loss,
};
use fandhe_ai_tensor_core::Tensor;

trait VarSource {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_>;
    fn run_backward(&self, loss: &Var<'_>, inputs: &[&Var<'_>]) -> Vec<Tensor<f32>>;
}

impl VarSource for fandhe_ai::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
    fn run_backward(&self, loss: &Var<'_>, inputs: &[&Var<'_>]) -> Vec<Tensor<f32>> {
        let grads = self.backward(loss).unwrap();
        inputs
            .iter()
            .map(|v| grads.get(v).unwrap().unwrap().clone())
            .collect()
    }
}

impl VarSource for fandhe_ai_autodiff::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
    fn run_backward(&self, loss: &Var<'_>, inputs: &[&Var<'_>]) -> Vec<Tensor<f32>> {
        let grads = self.backward(loss).unwrap();
        inputs
            .iter()
            .map(|v| grads.get(v).unwrap().unwrap().clone())
            .collect()
    }
}

/// 検査対象の演算。
#[derive(Clone, Copy, Debug)]
enum Op {
    MultiMargin,
    MultiMarginP2Weighted,
    MultiLabelMargin,
    MultiLabelSoftMargin,
    SigmoidFocal,
}

const BIT_EXACT_OPS: [Op; 5] = [
    Op::MultiMargin,
    Op::MultiMarginP2Weighted,
    Op::MultiLabelMargin,
    Op::MultiLabelSoftMargin,
    Op::SigmoidFocal,
];

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).unwrap()
}

/// 正負・0・マージン超過と未満を含む入力（shape `[3, 4]`）。
fn x_in() -> Tensor<f32> {
    t(
        vec![
            -2.5, -0.5, 0.0, 1.0, 0.75, 3.0, -1.25, 0.5, 2.0, -0.75, 0.25, 1.5,
        ],
        &[3, 4],
    )
}

fn labels() -> Tensor<f32> {
    t(
        vec![0.0, 1.0, 0.3, 0.9, 0.5, 1.0, 0.0, 0.25, 1.0, 0.0, 0.75, 0.1],
        &[3, 4],
    )
}

/// `(forward 出力, 追跡入力の勾配列)` を実行する。
fn run(tape: &impl VarSource, op: Op) -> (Tensor<f32>, Vec<Tensor<f32>>) {
    let x = tape.make_var(&x_in());
    let loss = match op {
        Op::MultiMargin => multi_margin_loss(
            &x,
            &ti(vec![0, 3, 1], &[3]),
            &MultiMarginOptions::default(),
            Reduction::Mean,
        ),
        Op::MultiMarginP2Weighted => multi_margin_loss(
            &x,
            &ti(vec![2, 0, 3], &[3]),
            &MultiMarginOptions::default()
                .p(2)
                .margin(0.5)
                .weight(t(vec![0.5, 2.0, 1.0, 3.0], &[4])),
            Reduction::Sum,
        ),
        // 行 0: target {3, 0}・行 1: 空集合・行 2: 全クラス（終端なし）
        Op::MultiLabelMargin => multilabel_margin_loss(
            &x,
            &ti(vec![3, 0, -1, 1, -1, 0, 0, 1, 2, 3, 0, 0], &[3, 4]),
            Reduction::Mean,
        ),
        Op::MultiLabelSoftMargin => multilabel_soft_margin_loss(
            &x,
            &labels(),
            &MultiLabelSoftMarginOptions::default().weight(t(vec![1.0, 0.5, 2.0, 1.5], &[4])),
            Reduction::Mean,
        ),
        Op::SigmoidFocal => sigmoid_focal_loss(
            &x,
            &labels(),
            &SigmoidFocalLossOptions::default().gamma(1.5),
            Reduction::Sum,
        ),
    }
    .unwrap();
    let out = loss.to_tensor();
    (out, tape.run_backward(&loss, &[&x]))
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn assert_bits(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(bits(a), bits(b), "{label}");
}

fn assert_bits_all(label: &str, a: &[Tensor<f32>], b: &[Tensor<f32>]) {
    assert_eq!(a.len(), b.len(), "{label}: 勾配本数");
    for (i, (ga, gb)) in a.iter().zip(b).enumerate() {
        assert_bits(&format!("{label}[{i}]"), ga, gb);
    }
}

// ---------------------------------------------------------------------
// CPU vs NaiveOps（属性なし）
// ---------------------------------------------------------------------

#[test]
fn cpu_bit_exact_forward_matches_naive_reference() {
    for op in BIT_EXACT_OPS {
        let cpu = fandhe_ai::tape();
        let naive = fandhe_ai_autodiff::Tape::new();
        let (a, _) = run(&cpu, op);
        let (b, _) = run(&naive, op);
        assert_bits(&format!("{op:?} forward: cpu vs naive"), &a, &b);
    }
}

#[test]
fn cpu_bit_exact_backward_matches_naive_reference() {
    for op in BIT_EXACT_OPS {
        let cpu = fandhe_ai::tape();
        let naive = fandhe_ai_autodiff::Tape::new();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&naive, op);
        assert_bits_all(&format!("{op:?} backward: cpu vs naive"), &a, &b);
    }
}

// ---------------------------------------------------------------------
// CUDA vs CPU（実機 `#[ignore]`）
// ---------------------------------------------------------------------

fn cuda_tape() -> fandhe_ai::Tape {
    fandhe_ai::tape_for(Device::Cuda(0)).expect("実機が利用可能な前提のテストのため成功するはず")
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/margin-focal-loss-ops-2653/README.md 参照"]
fn cuda_bit_exact_forward_matches_cpu_reference() {
    for op in BIT_EXACT_OPS {
        let cpu = fandhe_ai::tape();
        let dev = cuda_tape();
        let (a, _) = run(&cpu, op);
        let (b, _) = run(&dev, op);
        assert_bits(&format!("{op:?} forward: cpu vs cuda"), &a, &b);
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/margin-focal-loss-ops-2653/README.md 参照"]
fn cuda_bit_exact_backward_matches_cpu_reference() {
    for op in BIT_EXACT_OPS {
        let cpu = fandhe_ai::tape();
        let dev = cuda_tape();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&dev, op);
        assert_bits_all(&format!("{op:?} backward: cpu vs cuda"), &a, &b);
    }
}

// ---------------------------------------------------------------------
// Metal vs CPU（実機 `#[ignore]`・macOS 限定）
// ---------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn metal_tape() -> fandhe_ai::Tape {
    fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず")
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/margin-focal-loss-ops-2653/README.md 参照"]
fn metal_bit_exact_forward_matches_cpu_reference() {
    for op in BIT_EXACT_OPS {
        let cpu = fandhe_ai::tape();
        let dev = metal_tape();
        let (a, _) = run(&cpu, op);
        let (b, _) = run(&dev, op);
        assert_bits(&format!("{op:?} forward: cpu vs metal"), &a, &b);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/margin-focal-loss-ops-2653/README.md 参照"]
fn metal_bit_exact_backward_matches_cpu_reference() {
    for op in BIT_EXACT_OPS {
        let cpu = fandhe_ai::tape();
        let dev = metal_tape();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&dev, op);
        assert_bits_all(&format!("{op:?} backward: cpu vs metal"), &a, &b);
    }
}
