//! `fandhe_ai_autodiff::cumulative_ops`（イシュー #2636・`cummax`／`cummin`／
//! `logcumsumexp`。facade 非公開のため `fandhe_ai_autodiff::cumulative_ops::*` を
//! 直接 use する。`crates/autodiff/src/cumulative_ops.rs` モジュール doc 参照）の
//! バックエンド間 parity テスト（`nonfinite_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps::scan_*`〉と
//! `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`＝既定 `Unsupported` →
//! 共有ホストカーネルへフォールバック〉の突き合わせ）: `cummax`／`cummin` の値は
//! bit 一致（NaN はクラス一致）・索引は完全一致、`logcumsumexp` の forward と各
//! backward は REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）で
//! 検証する。両経路とも同じ共有カーネルを呼ぶため、ここで確認するのは
//! 「`BackendOps` 経由の配線（形状検証・フォールバック）が結果を変えないこと」である。
//!
//! `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os
//! = "macos")` 限定〉で同じ経路を CPU tape と比較）: 実機への到達手段が本エージェント
//! 実行環境にないため未実施のまま GB10／Mac セッションへ申し送る
//! （`docs/perf/logs/cumulative-ops-2636/README.md`）。CUDA／Metal はいずれも本 3 演算の
//! GPU カーネルを持たないため（既定 `Unsupported`）、この比較は「ホストへの
//! フォールバック経路が CPU tape と同じ結果になること」を確認するものであり、GPU
//! カーネル自体の parity ではない。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::cumulative_ops::{cummax, cummin, logcumsumexp};
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

/// タイ・NaN を含まない `[2, 3, 4]` 入力（`dim = 1` で非末尾軸を通す）。
fn input() -> Tensor<f32> {
    let data: Vec<f32> = (0..24)
        .map(|i| (((i * 7 + 3) % 11) as f32) * 0.4 - 2.0)
        .collect();
    t(data, &[2, 3, 4])
}

/// 損失 `Σ w ⊙ y` の重み（forward 値と同 shape）。
fn weights() -> Tensor<f32> {
    let data: Vec<f32> = (0..24).map(|i| ((i % 5) as f32) * 0.3 - 0.6).collect();
    t(data, &[2, 3, 4])
}

struct Outputs {
    max_v: Tensor<f32>,
    max_i: Vec<i32>,
    min_v: Tensor<f32>,
    min_i: Vec<i32>,
    lse: Tensor<f32>,
    d_max: Tensor<f32>,
    d_min: Tensor<f32>,
    d_lse: Tensor<f32>,
}

fn indices(i: &Tensor<i32>) -> Vec<i32> {
    i.contiguous().host_slice().into_owned()
}

macro_rules! outputs_on {
    ($tape:expr) => {{
        let tape = $tape;
        let w = tape.make_var(&weights());
        // `$f` は `x` を受けて損失の対象 `Var` を返す式。入力ごとに新しい葉を作る。
        macro_rules! grad_of {
            (|$x:ident| $body:expr) => {{
                let $x = tape.make_var(&input());
                let y = $body;
                let loss = y.mul(&w).unwrap().sum(None).unwrap();
                tape.backward(&loss)
                    .unwrap()
                    .get(&$x)
                    .unwrap()
                    .unwrap()
                    .clone()
            }};
        }
        let x = tape.make_var(&input());
        let (mv, mi) = cummax(&x, 1).unwrap();
        let (nv, ni) = cummin(&x, 1).unwrap();
        let l = logcumsumexp(&x, 1).unwrap();
        Outputs {
            max_v: mv.to_tensor(),
            max_i: indices(&mi),
            min_v: nv.to_tensor(),
            min_i: indices(&ni),
            lse: l.to_tensor(),
            d_max: grad_of!(|x| cummax(&x, 1).unwrap().0),
            d_min: grad_of!(|x| cummin(&x, 1).unwrap().0),
            d_lse: grad_of!(|x| logcumsumexp(&x, 1).unwrap()),
        }
    }};
}

fn cpu_outputs() -> Outputs {
    outputs_on!(fandhe_ai::tape())
}

fn naive_outputs() -> Outputs {
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

fn assert_outputs_match(label: &str, a: &Outputs, b: &Outputs) {
    assert_bits_eq(&format!("{label}: cummax 値"), &a.max_v, &b.max_v);
    assert_eq!(a.max_i, b.max_i, "{label}: cummax 索引");
    assert_bits_eq(&format!("{label}: cummin 値"), &a.min_v, &b.min_v);
    assert_eq!(a.min_i, b.min_i, "{label}: cummin 索引");
    assert_parity(&format!("{label}: logcumsumexp forward"), &a.lse, &b.lse);
    assert_parity(&format!("{label}: cummax backward"), &a.d_max, &b.d_max);
    assert_parity(&format!("{label}: cummin backward"), &a.d_min, &b.d_min);
    assert_parity(
        &format!("{label}: logcumsumexp backward"),
        &a.d_lse,
        &b.d_lse,
    );
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = cpu_outputs();
    let naive = naive_outputs();
    assert_outputs_match("cpu vs naive", &cpu, &naive);
    // 期待値の健全性（両者が同じ誤りで一致していないこと）: 先頭 lane は索引 0 から
    // 始まり、cummax 値は単調非減少・cummin 値は単調非増加。
    assert_eq!(cpu.max_i.len(), 24);
    let v = cpu.max_v.host_slice();
    let n = cpu.min_v.host_slice();
    for lane_base in [0usize, 1, 2, 3, 12, 13, 14, 15] {
        for a in 1..3 {
            assert!(v[lane_base + a * 4] >= v[lane_base + (a - 1) * 4]);
            assert!(n[lane_base + a * 4] <= n[lane_base + (a - 1) * 4]);
        }
    }
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/cumulative-ops-2636/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let dev = outputs_on!(device_tape);
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// 3 演算の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/cumulative-ops-2636/README.md 参照"]
fn metal_cumulative_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// 3 演算の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/cumulative-ops-2636/README.md 参照"]
fn cuda_cumulative_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
