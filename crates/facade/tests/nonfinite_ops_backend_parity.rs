//! `fandhe_ai_autodiff::nonfinite_ops`（イシュー #2635・`isnan`／`isinf`／
//! `isfinite`／`nan_to_num`。facade 非公開のため
//! `fandhe_ai_autodiff::nonfinite_ops::*` を直接 use する。`crates/autodiff/src/
//! nonfinite_ops.rs` モジュール doc 参照）のバックエンド間 parity テスト
//! （`trig_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps::scalar_unary`〉と
//! `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`＝既定 `Unsupported` →
//! ホスト参照実装へフォールバック〉の突き合わせ）: 判定 3 種は bool 完全一致、
//! `nan_to_num` の forward は bit 一致（NaN はクラス一致）、backward は REQ-2
//! 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）で検証する。
//!
//! `#[ignore]`（`tape_for(Device::Metal)`〈`cfg(target_os = "macos")` 限定〉／
//! `tape_for(Device::Cuda(0))` で同じ経路を CPU tape と比較）: 実機への到達手段が
//! 本エージェント実行環境にないため未実施のまま Mac／GB10 セッションへ申し送る
//! （`docs/perf/logs/nonfinite-ops-2635/README.md`）。CUDA／Metal はいずれも本 4 演算の
//! GPU カーネルを持たないため（明示 `None` → `Unsupported`）、この比較は
//! 「ホストへのフォールバック経路が CPU tape と同じ結果になること」を確認する
//! ものであり、GPU カーネル自体の parity ではない（bool 化は既存 cast 経路を通る）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::nonfinite_ops::{isfinite, isinf, isnan, nan_to_num};
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

/// NaN・`±inf`・`±0`・非正規化数・巨大有限値（`±1e30`。`Σ y²` の勾配 `2y` が
/// overflow しない範囲。`f32::MAX` は fixture 側で検証済み）・通常値を含む `[2, 5]` 入力。
fn mixed_input() -> Tensor<f32> {
    t(
        vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::MIN_POSITIVE / 4.0,
            1.0e30,
            -1.0e30,
            1.25,
            -3.5,
        ],
        &[2, 5],
    )
}

fn bools(x: &Tensor<bool>) -> Vec<bool> {
    x.contiguous().host_slice().into_owned()
}

/// NaN は NaN 同士、それ以外は bit 一致。
fn assert_class_or_bits_eq(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape");
    for (i, (x, y)) in a.host_slice().iter().zip(b.host_slice().iter()).enumerate() {
        if x.is_nan() || y.is_nan() {
            assert!(x.is_nan() && y.is_nan(), "{label}[{i}]: {x} vs {y}");
        } else {
            assert_eq!(x.to_bits(), y.to_bits(), "{label}[{i}]: {x} vs {y}");
        }
    }
}

fn assert_parity(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape");
    fandhe_ai_backend_cpu::parity::assert_parity(
        label,
        a.host_slice().as_ref(),
        b.host_slice().as_ref(),
    );
}

/// 判定 3 種の bool 出力と、`nan_to_num` の forward／入力勾配（損失 `Σ y²`）。
struct Outputs {
    isnan: Vec<bool>,
    isinf: Vec<bool>,
    isfinite: Vec<bool>,
    y: Tensor<f32>,
    dx: Tensor<f32>,
}

fn cpu_outputs() -> Outputs {
    let tape = fandhe_ai::tape();
    let x = tape.make_var(&mixed_input());
    let y = nan_to_num(&x, Some(1.0), Some(2.0), Some(-3.0)).unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let dx = tape
        .backward(&loss)
        .unwrap()
        .get(&x)
        .unwrap()
        .unwrap()
        .clone();
    Outputs {
        isnan: bools(&isnan(&x).unwrap()),
        isinf: bools(&isinf(&x).unwrap()),
        isfinite: bools(&isfinite(&x).unwrap()),
        y: y.to_tensor(),
        dx,
    }
}

fn naive_outputs() -> Outputs {
    let tape = fandhe_ai_autodiff::Tape::new();
    let x = tape.make_var(&mixed_input());
    let y = nan_to_num(&x, Some(1.0), Some(2.0), Some(-3.0)).unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let dx = tape
        .backward(&loss)
        .unwrap()
        .get(&x)
        .unwrap()
        .unwrap()
        .clone();
    Outputs {
        isnan: bools(&isnan(&x).unwrap()),
        isinf: bools(&isinf(&x).unwrap()),
        isfinite: bools(&isfinite(&x).unwrap()),
        y: y.to_tensor(),
        dx,
    }
}

fn assert_outputs_match(label: &str, a: &Outputs, b: &Outputs) {
    assert_eq!(a.isnan, b.isnan, "{label}: isnan");
    assert_eq!(a.isinf, b.isinf, "{label}: isinf");
    assert_eq!(a.isfinite, b.isfinite, "{label}: isfinite");
    assert_class_or_bits_eq(&format!("{label}: nan_to_num forward"), &a.y, &b.y);
    assert_parity(&format!("{label}: nan_to_num backward"), &a.dx, &b.dx);
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = cpu_outputs();
    let naive = naive_outputs();
    assert_outputs_match("cpu vs naive", &cpu, &naive);
    // 期待値の健全性（両者が同じ誤りで一致していないこと）。
    assert_eq!(
        cpu.isnan,
        [
            true, false, false, false, false, false, false, false, false, false
        ]
    );
    assert_eq!(cpu.y.host_slice()[0], 1.0);
    assert_eq!(cpu.y.host_slice()[1], 2.0);
    assert_eq!(cpu.y.host_slice()[2], -3.0);
    let dx = cpu.dx.host_slice();
    assert_eq!(dx[..3], [0.0, 0.0, 0.0], "非有限入力位置の勾配は 0");
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/nonfinite-ops-2635/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let x = device_tape.make_var(&mixed_input());
    let y = nan_to_num(&x, Some(1.0), Some(2.0), Some(-3.0)).unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let dx = device_tape
        .backward(&loss)
        .unwrap()
        .get(&x)
        .unwrap()
        .unwrap()
        .clone();
    let dev = Outputs {
        isnan: bools(&isnan(&x).unwrap()),
        isinf: bools(&isinf(&x).unwrap()),
        isfinite: bools(&isfinite(&x).unwrap()),
        y: y.to_tensor(),
        dx,
    };
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// 4 演算の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/nonfinite-ops-2635/README.md 参照"]
fn metal_nonfinite_ops_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// 4 演算の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/nonfinite-ops-2635/README.md 参照"]
fn cuda_nonfinite_ops_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
