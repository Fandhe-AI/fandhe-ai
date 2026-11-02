//! `Var::hardshrink`／`Var::softshrink`（イシュー #3817・親 #3813）のバックエンド間 parity テスト
//! （`softmin_threshold_ops_backend_parity.rs` と同型）。
//!
//! 両演算は選択と IEEE 減算のみで構成されるため、forward／backward とも bit 完全一致を契約とする
//! （`crates/autodiff/src/shrink_ops.rs` モジュール doc「数値契約」）。
//!
//! - 属性なし: CPU（`fandhe_ai::tape()`）と NaiveOps（`fandhe_ai_autodiff::Tape::new()`）の bit 一致。
//! - `#[ignore]`: CUDA 実機（`cuda_*`）・Metal 実機（`metal_*`・macOS 限定）と CPU の bit 一致。
//!   実機への到達手段が実装環境にないため未実測のまま申し送る
//!   （`docs/perf/logs/shrink-ops-3817/README.md`。実測は Phase 10 の #3860・#3861）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
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

#[derive(Clone, Copy, Debug)]
enum Op {
    Hardshrink,
    Softshrink,
}

const OPS: [Op; 2] = [Op::Hardshrink, Op::Softshrink];
const LAMBD: f32 = 0.5;

/// 境界（`|x| == lambd`）・範囲内・範囲外・0 を含む入力。
fn fixture() -> Tensor<f32> {
    Tensor::new(
        vec![
            -3.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 2.5, -2.0, 0.125,
        ],
        &[12],
    )
    .unwrap()
}

fn apply<'t>(op: Op, x: &Var<'t>) -> Var<'t> {
    match op {
        Op::Hardshrink => x.hardshrink(LAMBD).unwrap(),
        Op::Softshrink => x.softshrink(LAMBD).unwrap(),
    }
}

/// `(forward 出力, 入力勾配)`。損失は要素ごとに異なる重みを掛けた和（勾配を非自明にする）。
fn run(tape: &impl VarSource, op: Op) -> (Tensor<f32>, Tensor<f32>) {
    let x = tape.make_var(&fixture());
    let y = apply(op, &x);
    let out = y.to_tensor();
    let w = tape.make_var(
        &Tensor::new(
            vec![
                0.5, -1.0, 2.0, 0.75, -0.25, 1.5, -2.0, 0.1, 3.0, -0.5, 1.25, 0.6,
            ],
            &[12],
        )
        .unwrap(),
    );
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

#[test]
fn cpu_bit_exact_forward_matches_naive_reference() {
    let cpu = fandhe_ai::tape();
    let naive = fandhe_ai_autodiff::Tape::new();
    for op in OPS {
        let (a, _) = run(&cpu, op);
        let (b, _) = run(&naive, op);
        assert_bits(&format!("{op:?} forward: cpu vs naive"), &a, &b);
    }
}

#[test]
fn cpu_bit_exact_backward_matches_naive_reference() {
    for op in OPS {
        let cpu = fandhe_ai::tape();
        let naive = fandhe_ai_autodiff::Tape::new();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&naive, op);
        assert_bits(&format!("{op:?} backward: cpu vs naive"), &a, &b);
    }
}

fn cuda_tape() -> fandhe_ai::Tape {
    fandhe_ai::tape_for(Device::Cuda(0)).expect("実機が利用可能な前提のテストのため成功するはず")
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/shrink-ops-3817/README.md 参照"]
fn cuda_bit_exact_forward_matches_cpu_reference() {
    let cpu = fandhe_ai::tape();
    let dev = cuda_tape();
    for op in OPS {
        let (a, _) = run(&cpu, op);
        let (b, _) = run(&dev, op);
        assert_bits(&format!("{op:?} forward: cpu vs cuda"), &a, &b);
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/shrink-ops-3817/README.md 参照"]
fn cuda_bit_exact_backward_matches_cpu_reference() {
    for op in OPS {
        let cpu = fandhe_ai::tape();
        let dev = cuda_tape();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&dev, op);
        assert_bits(&format!("{op:?} backward: cpu vs cuda"), &a, &b);
    }
}

#[cfg(target_os = "macos")]
fn metal_tape() -> fandhe_ai::Tape {
    fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず")
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/shrink-ops-3817/README.md 参照"]
fn metal_bit_exact_forward_matches_cpu_reference() {
    let cpu = fandhe_ai::tape();
    let dev = metal_tape();
    for op in OPS {
        let (a, _) = run(&cpu, op);
        let (b, _) = run(&dev, op);
        assert_bits(&format!("{op:?} forward: cpu vs metal"), &a, &b);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/shrink-ops-3817/README.md 参照"]
fn metal_bit_exact_backward_matches_cpu_reference() {
    for op in OPS {
        let cpu = fandhe_ai::tape();
        let dev = metal_tape();
        let (_, a) = run(&cpu, op);
        let (_, b) = run(&dev, op);
        assert_bits(&format!("{op:?} backward: cpu vs metal"), &a, &b);
    }
}
