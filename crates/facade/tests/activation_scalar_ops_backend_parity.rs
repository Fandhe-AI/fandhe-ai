//! `fandhe_ai_autodiff::activation_scalar_ops`（イシュー #2649・`selu`／`celu`／
//! `softsign`／`hardsigmoid`／`log_sigmoid`。facade 非公開のため
//! `fandhe_ai_autodiff::activation_scalar_ops::*` を直接 use する。
//! `crates/autodiff/src/activation_scalar_ops.rs` モジュール doc 参照）の
//! バックエンド間 parity テスト（`trig_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps::scalar_unary`〉と
//! `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`＝既定 `Unsupported` →
//! ホスト参照実装へフォールバック〉の突き合わせ）: forward は bit 一致、
//! backward は REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）で検証する。
//!
//! `#[ignore]`（`tape_for(Device::Metal)`〈`cfg(target_os = "macos")` 限定〉／
//! `tape_for(Device::Cuda(0))` で同じ経路を CPU tape と比較）: 実機への到達手段
//! が本エージェント実行環境にないため未実施のまま Mac／GB10 セッションへ申し送る
//! （`docs/perf/logs/activation-scalar-ops-2649/README.md`）。CUDA／Metal は
//! いずれも本 5 演算の GPU カーネルを持たないため（明示 `None` → `Unsupported`）、
//! この比較は「ホストへのフォールバック経路が CPU tape と同じ結果になること」を
//! 確認するものであり、GPU カーネル自体の parity ではない。

use fandhe_ai::Device;
use fandhe_ai_autodiff::activation_scalar_ops::{celu, hardsigmoid, log_sigmoid, selu, softsign};
use fandhe_ai_autodiff::{AutodiffError, Var};
use fandhe_ai_tensor_core::Tensor;

trait VarSource {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_>;
}

impl VarSource for fandhe_ai::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

impl VarSource for fandhe_ai_autodiff::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn assert_parity(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape");
    fandhe_ai_backend_cpu::parity::assert_parity(
        label,
        a.host_slice().as_ref(),
        b.host_slice().as_ref(),
    );
}

type UnaryFn = for<'t> fn(&Var<'t>) -> Result<Var<'t>, AutodiffError>;

fn celu_1_5<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    celu(x, 1.5)
}

const UNARY: [(&str, UnaryFn); 5] = [
    ("selu", selu),
    ("celu", celu_1_5),
    ("softsign", softsign),
    ("hardsigmoid", hardsigmoid),
    ("log_sigmoid", log_sigmoid),
];

/// 負・正・`hardsigmoid` の領域外／内を含む入力（形状 `[2, 3]`。kink の `0`・`±3` を避ける）。
fn input() -> Tensor<f32> {
    t(vec![-4.0, -1.0, -0.3, 0.4, 1.2, 4.5], &[2, 3])
}

/// 損失 `Σ y²` の forward 出力と入力勾配（CPU tape）。
fn run_cpu(f: UnaryFn, x: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
    let tape = fandhe_ai::tape();
    let xv = tape.make_var(x);
    let y = f(&xv).unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let dx = tape
        .backward(&loss)
        .unwrap()
        .get(&xv)
        .unwrap()
        .unwrap()
        .clone();
    (y.to_tensor(), dx)
}

fn run_naive(f: UnaryFn, x: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
    let tape = fandhe_ai_autodiff::Tape::new();
    let xv = tape.make_var(x);
    let y = f(&xv).unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let dx = tape
        .backward(&loss)
        .unwrap()
        .get(&xv)
        .unwrap()
        .unwrap()
        .clone();
    (y.to_tensor(), dx)
}

#[test]
fn cpu_forward_is_bit_identical_and_backward_matches_naive() {
    let x = input();
    for (name, f) in UNARY {
        let (y_cpu, dx_cpu) = run_cpu(f, &x);
        let (y_naive, dx_naive) = run_naive(f, &x);
        let bits =
            |t: &Tensor<f32>| -> Vec<u32> { t.host_slice().iter().map(|v| v.to_bits()).collect() };
        assert_eq!(bits(&y_cpu), bits(&y_naive), "{name}: forward bit 一致");
        assert_parity(
            &format!("{name} backward: cpu vs naive"),
            &dx_cpu,
            &dx_naive,
        );
    }
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/activation-scalar-ops-2649/README.md`）。
// 5 演算すべてを CPU 版と同じセル数で網羅する（代表 1 演算で省略しない）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = input();
    for (name, f) in UNARY {
        let (y_cpu, dx_cpu) = run_cpu(f, &x);
        let xv = device_tape.make_var(&x);
        let y = f(&xv).unwrap();
        assert_parity(
            &format!("{name} forward: cpu vs {label}"),
            &y_cpu,
            &y.to_tensor(),
        );
        let loss = y.mul(&y).unwrap().sum(None).unwrap();
        let dx = device_tape
            .backward(&loss)
            .unwrap()
            .get(&xv)
            .unwrap()
            .unwrap()
            .clone();
        assert_parity(&format!("{name} backward: cpu vs {label}"), &dx_cpu, &dx);
    }
}

/// 5 演算の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/activation-scalar-ops-2649/README.md 参照"]
fn metal_activation_scalar_ops_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// 5 演算の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/activation-scalar-ops-2649/README.md 参照"]
fn cuda_activation_scalar_ops_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
