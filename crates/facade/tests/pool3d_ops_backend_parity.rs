//! `fandhe_ai_autodiff::pool3d_ops`（イシュー #2643・`max_pool3d`／`avg_pool3d`。facade
//! 非公開のため `fandhe_ai_autodiff::pool3d_ops::*` を直接 use する。
//! `crates/autodiff/src/pool3d_ops.rs` モジュール doc 参照）のバックエンド間 parity テスト
//! （`cumulative_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps::pool3d_*`〉と
//! `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`＝既定 `Unsupported` → 共有ホストカーネルへ
//! フォールバック〉の突き合わせ）: Max の値は bit 一致・索引は完全一致、Avg の forward と
//! 各 backward は REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）で
//! 検証する。両経路とも同じ共有カーネルを呼ぶため、ここで確認するのは
//! 「`BackendOps` 経由の配線（形状検証・フォールバック）が結果を変えないこと」である。
//!
//! `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os
//! = "macos")` 限定〉で同じ経路を CPU tape と比較）: 実機への到達手段が本エージェント
//! 実行環境にないため未実施のまま GB10／Mac セッションへ申し送る
//! （`docs/perf/logs/pool3d-ops-2643/README.md`）。CUDA／Metal はいずれも本 2 演算の
//! GPU カーネルを持たないため（既定 `Unsupported`）、この比較は「ホストへの
//! フォールバック経路が CPU tape と同じ結果になること」を確認するものであり、GPU
//! カーネル自体の parity ではない。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::pool3d_ops::{avg_pool3d, max_pool3d};
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

const IN_SHAPE: [usize; 5] = [2, 2, 4, 5, 4];
const KERNEL: [usize; 3] = [2, 3, 2];
const STRIDE: [usize; 3] = [1, 1, 2];
const PADDING: [usize; 3] = [1, 1, 1];
// 出力 shape は [2, 2, 5, 5, 3]。
const OUT_NUMEL: usize = 2 * 2 * 5 * 5 * 3;

/// 窓内タイを含みうる決定的な入力（`7` と `11` は互いに素）。
fn input() -> Tensor<f32> {
    let n: usize = IN_SHAPE.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (((i * 7 + 3) % 11) as f32) * 0.4 - 2.0)
        .collect();
    t(data, &IN_SHAPE)
}

/// 損失 `Σ w ⊙ y` の重み（forward 値と同 shape）。
fn weights() -> Tensor<f32> {
    let data: Vec<f32> = (0..OUT_NUMEL)
        .map(|i| ((i % 5) as f32) * 0.3 - 0.6)
        .collect();
    t(data, &[2, 2, 5, 5, 3])
}

struct Outputs {
    max_v: Tensor<f32>,
    max_i: Vec<i32>,
    avg_incl: Tensor<f32>,
    avg_excl: Tensor<f32>,
    d_max: Tensor<f32>,
    d_avg_incl: Tensor<f32>,
    d_avg_excl: Tensor<f32>,
}

fn indices(i: &Tensor<i32>) -> Vec<i32> {
    i.contiguous().host_slice().into_owned()
}

macro_rules! outputs_on {
    ($tape:expr) => {{
        let tape = $tape;
        let w = tape.make_var(&weights());
        // 入力ごとに新しい葉を作り、損失 `Σ w ⊙ f(x)` の入力勾配を返す。
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
        let (mv, mi) = max_pool3d(&x, KERNEL, Some(STRIDE), PADDING, [1, 1, 1], false).unwrap();
        let ai = avg_pool3d(&x, KERNEL, Some(STRIDE), PADDING, false, true).unwrap();
        let ae = avg_pool3d(&x, KERNEL, Some(STRIDE), PADDING, false, false).unwrap();
        Outputs {
            max_v: mv.to_tensor(),
            max_i: indices(&mi),
            avg_incl: ai.to_tensor(),
            avg_excl: ae.to_tensor(),
            d_max: grad_of!(|x| {
                max_pool3d(&x, KERNEL, Some(STRIDE), PADDING, [1, 1, 1], false)
                    .unwrap()
                    .0
            }),
            d_avg_incl: grad_of!(|x| {
                avg_pool3d(&x, KERNEL, Some(STRIDE), PADDING, false, true).unwrap()
            }),
            d_avg_excl: grad_of!(|x| {
                avg_pool3d(&x, KERNEL, Some(STRIDE), PADDING, false, false).unwrap()
            }),
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
    assert_bits_eq(&format!("{label}: max_pool3d 値"), &a.max_v, &b.max_v);
    assert_eq!(a.max_i, b.max_i, "{label}: max_pool3d 索引");
    assert_parity(
        &format!("{label}: avg_pool3d(count_include_pad=true) forward"),
        &a.avg_incl,
        &b.avg_incl,
    );
    assert_parity(
        &format!("{label}: avg_pool3d(count_include_pad=false) forward"),
        &a.avg_excl,
        &b.avg_excl,
    );
    assert_parity(&format!("{label}: max_pool3d backward"), &a.d_max, &b.d_max);
    assert_parity(
        &format!("{label}: avg_pool3d(include) backward"),
        &a.d_avg_incl,
        &b.d_avg_incl,
    );
    assert_parity(
        &format!("{label}: avg_pool3d(exclude) backward"),
        &a.d_avg_excl,
        &b.d_avg_excl,
    );
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = cpu_outputs();
    let naive = naive_outputs();
    assert_outputs_match("cpu vs naive", &cpu, &naive);
    // 期待値の健全性（両者が同じ誤りで一致していないこと）:
    // 出力要素数・索引範囲・Max 値が入力の最大値以下・padding を含む窓の
    // count_include_pad 差（角の窓は include の平均の絶対値が exclude 以下）。
    assert_eq!(cpu.max_i.len(), OUT_NUMEL);
    let plane: i32 = 4 * 5 * 4;
    assert!(cpu.max_i.iter().all(|&i| (0..plane).contains(&i)));
    let x = input();
    let xmax = x
        .host_slice()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(cpu.max_v.host_slice().iter().all(|&v| v <= xmax));
    let (inc, exc) = (cpu.avg_incl.host_slice(), cpu.avg_excl.host_slice());
    // 先頭出力（od=0, oh=0, ow=0）の窓は padding を含む: divisor は include=12・exclude<12 で
    // 同じ総和なので |include| < |exclude|（総和が 0 でない限り）。
    assert!(inc[0].abs() < exc[0].abs() + 1e-6);
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/pool3d-ops-2643/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let dev = outputs_on!(device_tape);
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// 2 演算の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/pool3d-ops-2643/README.md 参照"]
fn metal_pool3d_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// 2 演算の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/pool3d-ops-2643/README.md 参照"]
fn cuda_pool3d_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
