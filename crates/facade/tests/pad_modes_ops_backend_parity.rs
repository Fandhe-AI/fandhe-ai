//! `fandhe_ai_autodiff::pad_ops::pad_with_mode`（イシュー #2642・reflect／replicate／
//! circular。facade 非公開のため `fandhe_ai_autodiff::pad_ops::*` を直接 use する。
//! `crates/autodiff/src/pad_ops.rs` モジュール doc 参照）のバックエンド間 parity
//! テスト（`cumulative_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps::pad_modes_forward`〉と
//! `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`＝既定 `Unsupported` → 共有ホスト
//! カーネルへフォールバック〉の突き合わせ）: forward は bit 一致（算術を含まない
//! コピー）、backward は REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::
//! assert_parity`）で検証する。両経路とも同じ共有カーネルを呼ぶため、ここで確認
//! するのは「`BackendOps` 経由の配線（形状検証・フォールバック）が結果を変えない
//! こと」である。
//!
//! `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os
//! = "macos")` 限定〉で同じ経路を CPU tape と比較）: 実機への到達手段が本エージェント
//! 実行環境にないため未実施のまま GB10／Mac セッションへ申し送る
//! （`docs/perf/logs/pad-modes-2642/README.md`）。CUDA／Metal はいずれも本演算の
//! GPU カーネルを持たないため（既定 `Unsupported`）、この比較は「ホストへの
//! フォールバック経路が CPU tape と同じ結果になること」を確認するものであり、GPU
//! カーネル自体の parity ではない。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::pad_ops::pad_with_mode;
use fandhe_ai_tensor_core::{PadMode, Tensor};

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

const MODES: [PadMode; 3] = [PadMode::Reflect, PadMode::Replicate, PadMode::Circular];
const IN_SHAPE: [usize; 4] = [2, 3, 4, 5];
/// 先頭軸は無パディング、残り 3 軸は 3 モードすべてで有効な幅（軸長 3・4・5 に対して
/// reflect は `< len`、circular は `<= len`）。
const PADS: [(usize, usize); 4] = [(0, 0), (2, 1), (1, 3), (3, 2)];

fn input() -> Tensor<f32> {
    let data: Vec<f32> = (0..120)
        .map(|i| (((i * 7 + 3) % 13) as f32) * 0.4 - 2.0)
        .collect();
    t(data, &IN_SHAPE)
}

fn out_numel() -> usize {
    IN_SHAPE
        .iter()
        .zip(PADS)
        .map(|(&s, (b, a))| s + b + a)
        .product()
}

/// 損失 `Σ w ⊙ y` の重み（forward 値と同 shape）。
fn weights() -> Vec<f32> {
    (0..out_numel())
        .map(|i| ((i % 5) as f32) * 0.3 - 0.6)
        .collect()
}

struct Outputs {
    values: Vec<Tensor<f32>>,
    grads: Vec<Tensor<f32>>,
}

macro_rules! outputs_on {
    ($tape:expr) => {{
        let tape = $tape;
        let mut values = Vec::new();
        let mut grads = Vec::new();
        for mode in MODES {
            let x = tape.make_var(&input());
            let y = pad_with_mode(&x, &PADS, mode).unwrap();
            let yt = y.to_tensor();
            let w = tape.make_var(&t(weights(), yt.shape()));
            let loss = y.mul(&w).unwrap().sum(None).unwrap();
            let g = tape
                .backward(&loss)
                .unwrap()
                .get(&x)
                .unwrap()
                .unwrap()
                .clone();
            values.push(yt);
            grads.push(g);
        }
        Outputs { values, grads }
    }};
}

fn cpu_outputs() -> Outputs {
    outputs_on!(fandhe_ai::tape())
}

fn naive_outputs() -> Outputs {
    outputs_on!(fandhe_ai_autodiff::Tape::new())
}

fn assert_outputs_match(label: &str, a: &Outputs, b: &Outputs) {
    for (k, mode) in MODES.iter().enumerate() {
        let (va, vb) = (&a.values[k], &b.values[k]);
        assert_eq!(va.shape(), vb.shape(), "{label}: {mode:?} forward shape");
        for (i, (x, y)) in va
            .host_slice()
            .iter()
            .zip(vb.host_slice().iter())
            .enumerate()
        {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label}: {mode:?} forward[{i}]: {x} vs {y}"
            );
        }
        let (ga, gb) = (&a.grads[k], &b.grads[k]);
        assert_eq!(ga.shape(), gb.shape(), "{label}: {mode:?} backward shape");
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("{label}: {mode:?} backward"),
            ga.host_slice().as_ref(),
            gb.host_slice().as_ref(),
        );
    }
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = cpu_outputs();
    let naive = naive_outputs();
    assert_outputs_match("cpu vs naive", &cpu, &naive);
    // 期待値の健全性（両者が同じ誤りで一致していないこと）: 出力 shape と、
    // replicate 出力の先頭要素が入力の先頭要素に一致すること。
    let want: Vec<usize> = IN_SHAPE
        .iter()
        .zip(PADS)
        .map(|(&s, (b, a))| s + b + a)
        .collect();
    assert_eq!(cpu.values[0].shape(), want.as_slice());
    assert_eq!(
        cpu.values[1].host_slice()[0].to_bits(),
        input().host_slice()[0].to_bits()
    );
    // 勾配の総和は損失の重み付き総和の保存則（Σ d_x = Σ w）に一致する。
    let sum_w: f64 = weights().iter().map(|&v| f64::from(v)).sum();
    for g in &cpu.grads {
        let sum_g: f64 = g.host_slice().iter().map(|&v| f64::from(v)).sum();
        assert!((sum_g - sum_w).abs() < 1e-3, "{sum_g} vs {sum_w}");
    }
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/pad-modes-2642/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let dev = outputs_on!(device_tape);
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// 3 モードの forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/pad-modes-2642/README.md 参照"]
fn metal_pad_modes_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// 3 モードの forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/pad-modes-2642/README.md 参照"]
fn cuda_pad_modes_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
