//! `fandhe_ai_autodiff::trig_ops`（イシュー #2634・`atan`／`asin`／`acos`／
//! `sinh`／`cosh`／`asinh`／`acosh`／`atanh`／`atan2`。facade 非公開のため
//! `fandhe_ai_autodiff::trig_ops::*` を直接 use する。`crates/autodiff/src/
//! trig_ops.rs` モジュール doc 参照）のバックエンド間 parity テスト
//! （`fft_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps::scalar_unary`／
//! `scalar_binary`〉と `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`＝既定
//! `Unsupported` → ホスト参照実装へフォールバック〉の突き合わせ）: forward は
//! bit 一致、backward は REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）で検証する。
//!
//! `#[ignore]`（`tape_for(Device::Metal)`〈`cfg(target_os = "macos")` 限定〉／
//! `tape_for(Device::Cuda(0))` で同じ経路を CPU tape と比較）: 実機への到達手段
//! が本エージェント実行環境にないため未実施のまま Mac／GB10 セッションへ申し送る
//! （`docs/perf/logs/trig-ops-2634/README.md`）。CUDA／Metal はいずれも本 9 演算
//! の GPU カーネルを持たないため（明示 `None` → `Unsupported`）、この比較は
//! 「ホストへのフォールバック経路が CPU tape と同じ結果になること」を確認する
//! ものであり、GPU カーネル自体の parity ではない。

use fandhe_ai::Device;
use fandhe_ai_autodiff::trig_ops::{acos, acosh, asin, asinh, atan, atan2, atanh, cosh, sinh};
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

const UNARY: [(&str, UnaryFn); 8] = [
    ("atan", atan),
    ("asin", asin),
    ("acos", acos),
    ("sinh", sinh),
    ("cosh", cosh),
    ("asinh", asinh),
    ("acosh", acosh),
    ("atanh", atanh),
];

/// 各演算の定義域の内側の入力（形状 `[2, 3]`）。
fn domain_input(name: &str) -> Tensor<f32> {
    let data = match name {
        "asin" | "acos" | "atanh" => vec![-0.8, -0.4, -0.1, 0.2, 0.5, 0.9],
        "acosh" => vec![1.1, 1.5, 2.0, 3.0, 4.5, 8.0],
        _ => vec![-2.5, -1.0, -0.3, 0.4, 1.2, 2.8],
    };
    t(data, &[2, 3])
}

/// 損失 `Σ y²` の forward 出力と入力勾配（CPU tape）。
fn run_unary_cpu(f: UnaryFn, x: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
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

fn run_unary_naive(f: UnaryFn, x: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
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

fn atan2_inputs() -> (Tensor<f32>, Tensor<f32>) {
    (
        t(vec![1.3, 0.7, -1.1, -0.9, 1.6, -2.0], &[2, 3]),
        t(vec![0.6, -1.8, 1.2], &[3]),
    )
}

fn run_atan2_cpu() -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    let (a, b) = atan2_inputs();
    let tape = fandhe_ai::tape();
    let (av, bv) = (tape.make_var(&a), tape.make_var(&b));
    let y = atan2(&av, &bv).unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&av).unwrap().unwrap().clone();
    let db = grads.get(&bv).unwrap().unwrap().clone();
    (y.to_tensor(), da, db)
}

fn run_atan2_naive() -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    let (a, b) = atan2_inputs();
    let tape = fandhe_ai_autodiff::Tape::new();
    let (av, bv) = (tape.make_var(&a), tape.make_var(&b));
    let y = atan2(&av, &bv).unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&av).unwrap().unwrap().clone();
    let db = grads.get(&bv).unwrap().unwrap().clone();
    (y.to_tensor(), da, db)
}

#[test]
fn cpu_unary_forward_is_bit_identical_and_backward_matches_naive() {
    for (name, f) in UNARY {
        let x = domain_input(name);
        let (y_cpu, dx_cpu) = run_unary_cpu(f, &x);
        let (y_naive, dx_naive) = run_unary_naive(f, &x);
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

#[test]
fn cpu_atan2_forward_and_backward_match_naive_reference() {
    let (y_cpu, da_cpu, db_cpu) = run_atan2_cpu();
    let (y_naive, da_naive, db_naive) = run_atan2_naive();
    assert_parity("atan2 forward: cpu vs naive", &y_cpu, &y_naive);
    assert_parity("atan2 da: cpu vs naive", &da_cpu, &da_naive);
    assert_parity("atan2 db: cpu vs naive", &db_cpu, &db_naive);
    assert_eq!(
        db_cpu.shape(),
        &[3],
        "broadcast した b の勾配は元 shape へ縮約"
    );
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/trig-ops-2634/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    for (name, f) in UNARY {
        let x = domain_input(name);
        let (y_cpu, dx_cpu) = run_unary_cpu(f, &x);
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
    let (a, b) = atan2_inputs();
    let (y_cpu, da_cpu, db_cpu) = run_atan2_cpu();
    let (av, bv) = (device_tape.make_var(&a), device_tape.make_var(&b));
    let y = atan2(&av, &bv).unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let grads = device_tape.backward(&loss).unwrap();
    assert_parity(
        &format!("atan2 forward: cpu vs {label}"),
        &y_cpu,
        &y.to_tensor(),
    );
    assert_parity(
        &format!("atan2 da: cpu vs {label}"),
        &da_cpu,
        grads.get(&av).unwrap().unwrap(),
    );
    assert_parity(
        &format!("atan2 db: cpu vs {label}"),
        &db_cpu,
        grads.get(&bv).unwrap().unwrap(),
    );
}

/// 9 演算の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/trig-ops-2634/README.md 参照"]
fn metal_trig_ops_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// 9 演算の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/trig-ops-2634/README.md 参照"]
fn cuda_trig_ops_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
