//! `fandhe_ai_autodiff::elementwise_loss_ops`（イシュー #2652。facade 非公開のため本テストは
//! `fandhe_ai_autodiff::elementwise_loss_ops::*` を直接 use する。facade の dev 依存に
//! `fandhe-ai-autodiff` が既に含まれている。`crates/autodiff/src/elementwise_loss_ops.rs` モジュール doc 参照）の
//! バックエンド間 parity テスト（`softmin_threshold_ops_backend_parity.rs` と同型）。
//!
//! 本ファイルの契約は pos_weight 付き BCEWithLogits・HingeEmbedding・SoftMargin・GaussianNLL の各演算 ×
//! {forward, backward} × {CPU vs NaiveOps, CUDA vs CPU, Metal vs CPU} の各セルを埋めることであり、各テストは
//! 全演算を網羅する（代表 1 演算での省略をしない。#2146 の教訓）。
//!
//! **判定方式の割り当て**（モジュール doc「数値契約」参照）:
//! - bit 完全一致: 新規 Op 4 種。ホスト参照実装（`eval::elementwise_loss`）へ常にフォールバックし、
//!   どのバックエンドの tape からも同一コードで計算されるため。
//! - REQ-2 統一複合判定: `pos_weight` なしの委譲経路（既存 `Op::BceLoss`。CPU 融合カーネルを通りうる）。
//!
//! - 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`〉と `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`〉）:
//!   - `cpu_bit_exact_forward_matches_naive_reference`・`cpu_bit_exact_backward_matches_naive_reference`
//!   - `cpu_req2_forward_matches_naive_reference_within_tolerance`・
//!     `cpu_req2_backward_matches_naive_reference_within_tolerance`
//! - `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os = "macos")` 限定〉
//!   で同じ経路を CPU tape と比較）: 上記 4 種を `cuda_*`／`metal_*` の接頭辞で対称に置く（計 8 件）。
//!   新規 GPU カーネルは無く、ホスト経路の bit 一致確認である。実機への到達手段が本エージェント実行環境に
//!   ないため未実施のまま申し送る（`docs/perf/logs/elementwise-loss-ops-2652/README.md`）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Reduction;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::elementwise_loss_ops::{
    BceWithLogitsOptions, GaussianNllOptions, bce_with_logits_loss_with, gaussian_nll_loss,
    hinge_embedding_loss, soft_margin_loss,
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
    BcePosWeight,
    BceNoPosWeight,
    HingeEmbedding,
    SoftMargin,
    GaussianNll,
}

const BIT_EXACT_OPS: [Op; 4] = [
    Op::BcePosWeight,
    Op::HingeEmbedding,
    Op::SoftMargin,
    Op::GaussianNll,
];
const REQ2_OPS: [Op; 1] = [Op::BceNoPosWeight];

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// 正負・0・境界（hinge の `x == margin`）・`var < eps` を含む入力（shape `[2, 3]`）。
fn x_in() -> Tensor<f32> {
    t(vec![-2.5, -0.5, 0.0, 1.0, 0.75, 3.0], &[2, 3])
}

fn y_soft() -> Tensor<f32> {
    t(vec![0.0, 1.0, 0.3, 0.9, 0.5, 1.0], &[2, 3])
}

fn y_pm() -> Tensor<f32> {
    t(vec![1.0, -1.0, -1.0, 1.0, -1.0, 1.0], &[2, 3])
}

fn target_in() -> Tensor<f32> {
    t(vec![0.2, -0.4, 0.1, 1.5, 0.75, 2.0], &[2, 3])
}

fn var_in() -> Tensor<f32> {
    t(vec![0.0, 0.5, 2.0, 1e-7, 1.0, 0.25], &[2, 3])
}

/// `(forward 出力, 追跡入力の勾配列)` を実行する。
fn run(tape: &impl VarSource, op: Op) -> (Tensor<f32>, Vec<Tensor<f32>>) {
    let x = tape.make_var(&x_in());
    match op {
        Op::BcePosWeight | Op::BceNoPosWeight => {
            let y = tape.make_var(&y_soft());
            let options = if matches!(op, Op::BcePosWeight) {
                BceWithLogitsOptions::default().pos_weight(t(vec![0.5, 2.0, 3.0], &[3]))
            } else {
                BceWithLogitsOptions::default()
            };
            let loss = bce_with_logits_loss_with(&x, &y, Reduction::Mean, &options).unwrap();
            let out = loss.to_tensor();
            (out, tape.run_backward(&loss, &[&x, &y]))
        }
        Op::HingeEmbedding => {
            let loss = hinge_embedding_loss(&x, &y_pm(), 0.75, Reduction::Sum).unwrap();
            let out = loss.to_tensor();
            (out, tape.run_backward(&loss, &[&x]))
        }
        Op::SoftMargin => {
            let loss = soft_margin_loss(&x, &y_pm(), Reduction::Mean).unwrap();
            let out = loss.to_tensor();
            (out, tape.run_backward(&loss, &[&x]))
        }
        Op::GaussianNll => {
            let tg = tape.make_var(&target_in());
            let v = tape.make_var(&var_in());
            let options = GaussianNllOptions::default().full(true);
            let loss = gaussian_nll_loss(&x, &tg, &v, &options, Reduction::Mean).unwrap();
            let out = loss.to_tensor();
            (out, tape.run_backward(&loss, &[&x, &tg, &v]))
        }
    }
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

fn assert_req2(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    fandhe_ai_backend_cpu::parity::assert_parity(
        label,
        a.host_slice().as_ref(),
        b.host_slice().as_ref(),
    );
}

fn assert_req2_all(label: &str, a: &[Tensor<f32>], b: &[Tensor<f32>]) {
    assert_eq!(a.len(), b.len(), "{label}: 勾配本数");
    for (i, (ga, gb)) in a.iter().zip(b).enumerate() {
        assert_req2(&format!("{label}[{i}]"), ga, gb);
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

#[test]
fn cpu_req2_forward_matches_naive_reference_within_tolerance() {
    for op in REQ2_OPS {
        let cpu = fandhe_ai::tape();
        let naive = fandhe_ai_autodiff::Tape::new();
        let (a, _) = run(&cpu, op);
        let (b, _) = run(&naive, op);
        assert_req2(&format!("{op:?} forward: cpu vs naive"), &a, &b);
    }
}

#[test]
fn cpu_req2_backward_matches_naive_reference_within_tolerance() {
    for op in REQ2_OPS {
        let cpu = fandhe_ai::tape();
        let naive = fandhe_ai_autodiff::Tape::new();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&naive, op);
        assert_req2_all(&format!("{op:?} backward: cpu vs naive"), &a, &b);
    }
}

// ---------------------------------------------------------------------
// CUDA vs CPU（実機 `#[ignore]`）
// ---------------------------------------------------------------------

fn cuda_tape() -> fandhe_ai::Tape {
    fandhe_ai::tape_for(Device::Cuda(0)).expect("実機が利用可能な前提のテストのため成功するはず")
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/elementwise-loss-ops-2652/README.md 参照"]
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
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/elementwise-loss-ops-2652/README.md 参照"]
fn cuda_bit_exact_backward_matches_cpu_reference() {
    for op in BIT_EXACT_OPS {
        let cpu = fandhe_ai::tape();
        let dev = cuda_tape();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&dev, op);
        assert_bits_all(&format!("{op:?} backward: cpu vs cuda"), &a, &b);
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/elementwise-loss-ops-2652/README.md 参照"]
fn cuda_req2_forward_matches_cpu_reference_within_tolerance() {
    for op in REQ2_OPS {
        let cpu = fandhe_ai::tape();
        let dev = cuda_tape();
        let (a, _) = run(&cpu, op);
        let (b, _) = run(&dev, op);
        assert_req2(&format!("{op:?} forward: cpu vs cuda"), &a, &b);
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/elementwise-loss-ops-2652/README.md 参照"]
fn cuda_req2_backward_matches_cpu_reference_within_tolerance() {
    for op in REQ2_OPS {
        let cpu = fandhe_ai::tape();
        let dev = cuda_tape();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&dev, op);
        assert_req2_all(&format!("{op:?} backward: cpu vs cuda"), &a, &b);
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
#[ignore = "Metal 実機が必要。docs/perf/logs/elementwise-loss-ops-2652/README.md 参照"]
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
#[ignore = "Metal 実機が必要。docs/perf/logs/elementwise-loss-ops-2652/README.md 参照"]
fn metal_bit_exact_backward_matches_cpu_reference() {
    for op in BIT_EXACT_OPS {
        let cpu = fandhe_ai::tape();
        let dev = metal_tape();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&dev, op);
        assert_bits_all(&format!("{op:?} backward: cpu vs metal"), &a, &b);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/elementwise-loss-ops-2652/README.md 参照"]
fn metal_req2_forward_matches_cpu_reference_within_tolerance() {
    for op in REQ2_OPS {
        let cpu = fandhe_ai::tape();
        let dev = metal_tape();
        let (a, _) = run(&cpu, op);
        let (b, _) = run(&dev, op);
        assert_req2(&format!("{op:?} forward: cpu vs metal"), &a, &b);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/elementwise-loss-ops-2652/README.md 参照"]
fn metal_req2_backward_matches_cpu_reference_within_tolerance() {
    for op in REQ2_OPS {
        let cpu = fandhe_ai::tape();
        let dev = metal_tape();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&dev, op);
        assert_req2_all(&format!("{op:?} backward: cpu vs metal"), &a, &b);
    }
}
