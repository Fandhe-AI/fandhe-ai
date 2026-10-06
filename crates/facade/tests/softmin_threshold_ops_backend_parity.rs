//! `fandhe_ai_autodiff::softmin_threshold_ops`（イシュー #2650。facade 非公開のため本テストは
//! `fandhe_ai_autodiff::softmin_threshold_ops::*` を直接 use する。facade の dev 依存に
//! `fandhe-ai-autodiff` が既に含まれている。`crates/autodiff/src/softmin_threshold_ops.rs` モジュール doc 参照）の
//! バックエンド間 parity テスト（`activation_ops_backend_parity.rs` と同型）。
//!
//! 本ファイルの契約は `softmin`／`tanhshrink`／`threshold`／`rrelu`（固定 noise の `rrelu_with_noise` と
//! 推論 `rrelu(training = false)`）の各演算 × {forward, backward} × {CPU vs NaiveOps, CUDA vs CPU,
//! Metal vs CPU} の各セルを埋めることであり、各テストは全演算を網羅する（代表 1 演算での省略をしない。
//! #2146 の教訓）。関数名は下の網羅表と一対一に対応する。
//!
//! **判定方式の割り当て**（モジュール doc「数値契約」参照）:
//! - bit 完全一致: `threshold`・`rrelu_with_noise`（選択と IEEE 乗算 1 回のみ）・`rrelu` 推論
//!   （同一入力・同一パラメータからの同一路のため bit 一致。PyTorch との比較で 1 ulp ずれうる点は
//!   autodiff 側 `softmin_threshold_parity.rs` を参照）
//! - REQ-2 統一複合判定: `softmin`（`softmax` の超越関数）・`tanhshrink`（`tanh`）
//!
//! - 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`〉と `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`〉）:
//!   - `cpu_bit_exact_forward_matches_naive_reference`（`threshold`／`rrelu_with_noise`／`rrelu` 推論）
//!   - `cpu_bit_exact_backward_matches_naive_reference`（同上）
//!   - `cpu_req2_forward_matches_naive_reference_within_tolerance`（`softmin`／`tanhshrink`）
//!   - `cpu_req2_backward_matches_naive_reference_within_tolerance`（同上）
//! - `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os = "macos")` 限定〉
//!   で同じ経路を CPU tape と比較）: 上記 4 種を `cuda_*`／`metal_*` の接頭辞で対称に置く（計 8 件）。
//!   新規 GPU カーネルは無く、既存カーネルとホストフォールバック経路の確認である。実機への到達手段が
//!   本エージェント実行環境にないため未実施のまま申し送る
//!   （`docs/perf/logs/softmin-threshold-ops-2650/README.md`）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::softmin_threshold_ops::{
    rrelu, rrelu_with_noise, softmin, tanhshrink, threshold,
};
use fandhe_ai_tensor_core::Tensor;

trait VarSource {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_>;
    fn run_backward(&self, loss: &Var<'_>, x: &Var<'_>) -> Tensor<f32>;
}

impl VarSource for fandhe_ai::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
    fn run_backward(&self, loss: &Var<'_>, x: &Var<'_>) -> Tensor<f32> {
        let grads = self.backward(loss).unwrap();
        grads.get(x).unwrap().unwrap().clone()
    }
}

impl VarSource for fandhe_ai_autodiff::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
    fn run_backward(&self, loss: &Var<'_>, x: &Var<'_>) -> Tensor<f32> {
        let grads = self.backward(loss).unwrap();
        grads.get(x).unwrap().unwrap().clone()
    }
}

/// 検査対象の演算。
#[derive(Clone, Copy, Debug)]
enum Op {
    Softmin,
    Tanhshrink,
    Threshold,
    RreluWithNoise,
    RreluEval,
}

const BIT_EXACT_OPS: [Op; 3] = [Op::Threshold, Op::RreluWithNoise, Op::RreluEval];
const REQ2_OPS: [Op; 2] = [Op::Softmin, Op::Tanhshrink];

/// 境界・正負・0 を含む入力（rank 1）。
fn fixture() -> Tensor<f32> {
    Tensor::new(vec![-3.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.5], &[8]).unwrap()
}

/// `rrelu_with_noise` 用の固定 noise（正領域 1・負領域は異なる傾き）。
fn noise() -> Tensor<f32> {
    Tensor::new(vec![0.3, 0.2, 0.15, 0.25, 1.0, 1.0, 1.0, 1.0], &[8]).unwrap()
}

fn apply<'t>(op: Op, x: &Var<'t>) -> Var<'t> {
    match op {
        Op::Softmin => softmin(x, 0).unwrap(),
        Op::Tanhshrink => tanhshrink(x).unwrap(),
        Op::Threshold => threshold(x, 0.25, 7.0).unwrap(),
        Op::RreluWithNoise => rrelu_with_noise(x, &noise()).unwrap(),
        Op::RreluEval => rrelu(x, 0.1, 0.3, false).unwrap(),
    }
}

/// `(forward 出力, 入力勾配)` を実行する。損失は `sum(y * y_weight)`（`softmin` の和は定数 1 で勾配が
/// 0 になるため、要素ごとに異なる重みを掛けて勾配を非自明にする）。
fn run(tape: &impl VarSource, op: Op) -> (Tensor<f32>, Tensor<f32>) {
    let x = tape.make_var(&fixture());
    let y = apply(op, &x);
    let out = y.to_tensor();
    let w = tape
        .make_var(&Tensor::new(vec![0.5, -1.0, 2.0, 0.75, -0.25, 1.5, -2.0, 0.1], &[8]).unwrap());
    let loss = y.mul(&w).unwrap().sum(None).unwrap();
    let dx = tape.run_backward(&loss, &x);
    (out, dx)
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

fn assert_req2(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    fandhe_ai_backend_cpu::parity::assert_parity(
        label,
        a.host_slice().as_ref(),
        b.host_slice().as_ref(),
    );
}

// ---------------------------------------------------------------------
// CPU vs NaiveOps（属性なし）
// ---------------------------------------------------------------------

#[test]
fn cpu_bit_exact_forward_matches_naive_reference() {
    let cpu = fandhe_ai::tape();
    let naive = fandhe_ai_autodiff::Tape::new();
    for op in BIT_EXACT_OPS {
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
        assert_bits(&format!("{op:?} backward: cpu vs naive"), &a, &b);
    }
}

#[test]
fn cpu_req2_forward_matches_naive_reference_within_tolerance() {
    let cpu = fandhe_ai::tape();
    let naive = fandhe_ai_autodiff::Tape::new();
    for op in REQ2_OPS {
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
        assert_req2(&format!("{op:?} backward: cpu vs naive"), &a, &b);
    }
}

// ---------------------------------------------------------------------
// CUDA vs CPU（実機 `#[ignore]`）
// ---------------------------------------------------------------------

fn cuda_tape() -> fandhe_ai::Tape {
    fandhe_ai::tape_for(Device::Cuda(0)).expect("実機が利用可能な前提のテストのため成功するはず")
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/softmin-threshold-ops-2650/README.md 参照"]
fn cuda_bit_exact_forward_matches_cpu_reference() {
    let cpu = fandhe_ai::tape();
    let dev = cuda_tape();
    for op in BIT_EXACT_OPS {
        let (a, _) = run(&cpu, op);
        let (b, _) = run(&dev, op);
        assert_bits(&format!("{op:?} forward: cpu vs cuda"), &a, &b);
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/softmin-threshold-ops-2650/README.md 参照"]
fn cuda_bit_exact_backward_matches_cpu_reference() {
    for op in BIT_EXACT_OPS {
        let cpu = fandhe_ai::tape();
        let dev = cuda_tape();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&dev, op);
        assert_bits(&format!("{op:?} backward: cpu vs cuda"), &a, &b);
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/softmin-threshold-ops-2650/README.md 参照"]
fn cuda_req2_forward_matches_cpu_reference_within_tolerance() {
    let cpu = fandhe_ai::tape();
    let dev = cuda_tape();
    for op in REQ2_OPS {
        let (a, _) = run(&cpu, op);
        let (b, _) = run(&dev, op);
        assert_req2(&format!("{op:?} forward: cpu vs cuda"), &a, &b);
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/softmin-threshold-ops-2650/README.md 参照"]
fn cuda_req2_backward_matches_cpu_reference_within_tolerance() {
    for op in REQ2_OPS {
        let cpu = fandhe_ai::tape();
        let dev = cuda_tape();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&dev, op);
        assert_req2(&format!("{op:?} backward: cpu vs cuda"), &a, &b);
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
#[ignore = "Metal 実機が必要。docs/perf/logs/softmin-threshold-ops-2650/README.md 参照"]
fn metal_bit_exact_forward_matches_cpu_reference() {
    let cpu = fandhe_ai::tape();
    let dev = metal_tape();
    for op in BIT_EXACT_OPS {
        let (a, _) = run(&cpu, op);
        let (b, _) = run(&dev, op);
        assert_bits(&format!("{op:?} forward: cpu vs metal"), &a, &b);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/softmin-threshold-ops-2650/README.md 参照"]
fn metal_bit_exact_backward_matches_cpu_reference() {
    for op in BIT_EXACT_OPS {
        let cpu = fandhe_ai::tape();
        let dev = metal_tape();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&dev, op);
        assert_bits(&format!("{op:?} backward: cpu vs metal"), &a, &b);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/softmin-threshold-ops-2650/README.md 参照"]
fn metal_req2_forward_matches_cpu_reference_within_tolerance() {
    let cpu = fandhe_ai::tape();
    let dev = metal_tape();
    for op in REQ2_OPS {
        let (a, _) = run(&cpu, op);
        let (b, _) = run(&dev, op);
        assert_req2(&format!("{op:?} forward: cpu vs metal"), &a, &b);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/softmin-threshold-ops-2650/README.md 参照"]
fn metal_req2_backward_matches_cpu_reference_within_tolerance() {
    for op in REQ2_OPS {
        let cpu = fandhe_ai::tape();
        let dev = metal_tape();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&dev, op);
        assert_req2(&format!("{op:?} backward: cpu vs metal"), &a, &b);
    }
}
