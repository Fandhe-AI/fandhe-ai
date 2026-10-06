//! `fandhe_ai_autodiff::indexed_update_ops`（イシュー #2641・`scatter_reduce`／
//! `index_add`／`index_copy`／`masked_scatter`。facade 非公開のため
//! `fandhe_ai_autodiff::indexed_update_ops::*` を直接 use する。
//! `crates/autodiff/src/indexed_update_ops.rs` モジュール doc 参照）の
//! バックエンド間 parity テスト（`cumulative_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`。`indexed_scatter_reduce`・`scatter`〉と
//! `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`＝既定 `Unsupported` → ホスト参照実装へ
//! フォールバック〉の突き合わせ）: forward と各 backward を REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）で検証し、選択・コピーのみの
//! `index_copy`／`masked_scatter` の forward は bit 一致も確認する。
//!
//! `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os
//! = "macos")` 限定〉で同じ経路を CPU tape と比較）: 実機への到達手段が本エージェント
//! 実行環境にないため未実施のまま GB10／Mac セッションへ申し送る
//! （`docs/perf/logs/indexed-update-ops-2641/README.md`）。比較対象の意味は演算の系統で
//! 異なる: `scatter_reduce` は CUDA／Metal とも専用カーネルを持たず（既定
//! `Unsupported`）ホストフォールバックの確認、合成 3 演算は既存の GPU scatter カーネル
//! 経由（`index_add` は `Add` の `f64` 相当の決定的集約契約、`index_copy`／
//! `masked_scatter` はコピーのみ）の確認である。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::indexed_update_ops::{
    index_add, index_copy, masked_scatter, scatter_reduce,
};
use fandhe_ai_tensor_core::{ScatterReduceMode, Tensor};

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

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

/// タイ・NaN・0 を含まない決定的な入力（`n` 要素）。
fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + salt * 5 + 3) % 13) as f32) * 0.37 - 2.1)
        .collect()
}

const MODES: [ScatterReduceMode; 5] = [
    ScatterReduceMode::Sum,
    ScatterReduceMode::Prod,
    ScatterReduceMode::Mean,
    ScatterReduceMode::Amax,
    ScatterReduceMode::Amin,
];

/// 1 演算ぶんの forward 値と入力・src の勾配。
struct Out {
    label: String,
    value: Tensor<f32>,
    d_x: Tensor<f32>,
    d_s: Tensor<f32>,
    exact: bool,
}

macro_rules! run_op {
    ($tape:expr, $label:expr, $exact:expr, $xs:expr, $ss:expr, |$x:ident, $s:ident| $body:expr) => {{
        let tape = &$tape;
        let $x = tape.make_var(&$xs);
        let $s = tape.make_var(&$ss);
        let y = $body;
        let value = y.to_tensor();
        let w_data: Vec<f32> = (0..value.shape().iter().product::<usize>())
            .map(|i| ((i % 5) as f32) * 0.3 - 0.6)
            .collect();
        let w = tape.make_var(&t(w_data, value.shape()));
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        Out {
            label: $label,
            value,
            d_x: grads.get(&$x).unwrap().unwrap().clone(),
            d_s: grads.get(&$s).unwrap().unwrap().clone(),
            exact: $exact,
        }
    }};
}

macro_rules! outputs_on {
    ($make:expr) => {{
        let shared_tape = $make;
        let mut outs: Vec<Out> = Vec::new();
        let idx_sr = ti(vec![0, 3, 1, 1, 2, 0], &[3, 2]);
        for mode in MODES {
            for inc in [true, false] {
                outs.push(run_op!(
                    shared_tape,
                    format!("scatter_reduce {mode:?} include_self={inc}"),
                    matches!(mode, ScatterReduceMode::Amax | ScatterReduceMode::Amin),
                    t(data(12, 1), &[3, 4]),
                    t(data(6, 2), &[3, 2]),
                    |x, s| scatter_reduce(&x, 1, &idx_sr, &s, mode, inc).unwrap()
                ));
            }
        }
        let idx_add = ti(vec![2, 0, 2], &[3]);
        outs.push(run_op!(
            shared_tape,
            "index_add".to_string(),
            false,
            t(data(12, 3), &[4, 3]),
            t(data(9, 4), &[3, 3]),
            |x, s| index_add(&x, 0, &idx_add, &s).unwrap()
        ));
        let idx_copy = ti(vec![4, 0], &[2]);
        outs.push(run_op!(
            shared_tape,
            "index_copy".to_string(),
            true,
            t(data(15, 5), &[3, 5]),
            t(data(6, 6), &[3, 2]),
            |x, s| index_copy(&x, 1, &idx_copy, &s).unwrap()
        ));
        let mask = Tensor::new(vec![true, false, true, false, true, false], &[2, 3]).unwrap();
        outs.push(run_op!(
            shared_tape,
            "masked_scatter".to_string(),
            true,
            t(data(6, 7), &[2, 3]),
            t(data(5, 8), &[5]),
            |x, s| masked_scatter(&x, &mask, &s).unwrap()
        ));
        outs
    }};
}

fn cpu_outputs() -> Vec<Out> {
    outputs_on!(fandhe_ai::tape())
}

fn naive_outputs() -> Vec<Out> {
    outputs_on!(fandhe_ai_autodiff::Tape::new())
}

fn assert_parity(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape");
    fandhe_ai_backend_cpu::parity::assert_parity(
        label,
        a.host_slice().as_ref(),
        b.host_slice().as_ref(),
    );
}

fn assert_bits_eq(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape");
    for (i, (x, y)) in a.host_slice().iter().zip(b.host_slice().iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label}[{i}]: {x} vs {y}");
    }
}

fn assert_outputs_match(label: &str, a: &[Out], b: &[Out]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.label, y.label);
        let l = format!("{label}: {}", x.label);
        if x.exact {
            assert_bits_eq(&format!("{l} forward"), &x.value, &y.value);
        } else {
            assert_parity(&format!("{l} forward"), &x.value, &y.value);
        }
        assert_parity(&format!("{l} d_x"), &x.d_x, &y.d_x);
        assert_parity(&format!("{l} d_src"), &x.d_s, &y.d_s);
    }
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = cpu_outputs();
    let naive = naive_outputs();
    assert_outputs_match("cpu vs naive", &cpu, &naive);
    // 期待値の健全性（両者が同じ誤りで一致していないこと）: 全演算で勾配が非自明。
    assert_eq!(cpu.len(), 13);
    for o in &cpu {
        assert!(
            o.d_s.host_slice().iter().any(|&v| v != 0.0),
            "{}: src 勾配が全て 0",
            o.label
        );
    }
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/indexed-update-ops-2641/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let dev = outputs_on!(device_tape);
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// 4 演算の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/indexed-update-ops-2641/README.md 参照"]
fn metal_indexed_update_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// 4 演算の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/indexed-update-ops-2641/README.md 参照"]
fn cuda_indexed_update_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
