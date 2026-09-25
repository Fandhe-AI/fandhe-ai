//! `fandhe_ai_autodiff::topk_unique_ops`（イシュー #2153・facade 非公開
//! の内部入口。`crates/autodiff/src/topk_unique_ops.rs` モジュール doc
//! 参照）のバックエンド間 parity テスト（`reduce_ops_backend_parity.rs`
//! と同型）。
//!
//! `topk_unique_ops` は facade から再エクスポートされないため、本
//! テストは `fandhe_ai_autodiff::topk_unique_ops::*` を直接 use する。
//!
//! 選択演算（丸めなし）のため REQ-2 複合判定ではなく **bit 完全一致**
//! で突合する（tolerance を持ち込まない。`fandhe_ai_tensor_core::
//! BackendOps::unique_ext`／`topk` doc の数値契約）。
//!
//! - 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`〉と
//!   `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`〉の突き合わせ）:
//!   `topk_with_options`（`sorted=false`・負 dim）の forward・backward、
//!   `unique_with_options`（`dim` 有無 × `inverse`／`counts`）、
//!   `unique_consecutive` の forward。
//! - `#[ignore]`（`tape_for(Device::Metal)`〈`cfg(target_os =
//!   "macos")` 限定〉／`tape_for(Device::Cuda(0))` で同じ経路を CPU
//!   tape と比較。CUDA／Metal は `BackendOps::unique_ext` を override
//!   しないため既定 `Unsupported` からホストフォールバック経由で CPU
//!   側と同じ計算結果になる契約）。
//!
//!   実機（DGX Spark GB10／Apple Silicon）への到達手段が本エージェント
//!   実行環境にないため未実施のまま Mac／GB10 セッションへ申し送る
//!   （`docs/perf/logs/topk-unique-2153/README.md`）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::topk_unique_ops::{
    TopkOptions, UniqueOptions, topk_with_options, unique_consecutive, unique_with_options,
};
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

fn f32_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn i32_vec(t: &Tensor<i32>) -> Vec<i32> {
    t.contiguous().host_slice().into_owned()
}

/// `topk`／`unique` 共通の fixture（正負混在・重複値ありの 1×6）。
fn fixture() -> Tensor<f32> {
    Tensor::new(vec![3.0, 1.0, 4.0, 1.5, 9.0, 1.0], &[1, 6]).expect("test fixture: shape 一致")
}

/// `unique` の `dim` 指定用 fixture（shape [3, 2]。row0 と row2 が
/// 重複）。
fn unique_dim_fixture() -> Tensor<f32> {
    Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0], &[3, 2]).expect("test fixture: shape 一致")
}

/// `topk_with_options`（`sorted=false`・負 dim）forward が CPU
/// （`fandhe_ai::tape()`）と NaiveOps（`fandhe_ai_autodiff::
/// Tape::new()`）で bit 完全一致することを確認する。
#[test]
fn cpu_topk_sorted_false_negative_dim_forward_bit_matches_naive_reference() {
    let data = fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    let opts = TopkOptions::default().with_dim(-1).with_sorted(false);
    let (out_cpu, idx_cpu) = topk_with_options(&x_cpu, 3, opts).unwrap();
    let (out_naive, idx_naive) = topk_with_options(&x_naive, 3, opts).unwrap();

    assert_eq!(
        f32_bits(&out_cpu.to_tensor()),
        f32_bits(&out_naive.to_tensor()),
        "topk sorted=false values"
    );
    assert_eq!(
        i32_vec(&idx_cpu),
        i32_vec(&idx_naive),
        "topk sorted=false index"
    );
}

/// `topk_with_options`（`sorted=false`）backward が CPU と NaiveOps で
/// bit 完全一致することを確認する（scatter ベース VJP のため）。
#[test]
fn cpu_topk_sorted_false_gradient_bit_matches_naive_reference() {
    let data = fixture();
    let opts = TopkOptions::default().with_dim(1).with_sorted(false);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let (out_cpu, _) = topk_with_options(&x_cpu, 3, opts).unwrap();
    let loss_cpu = out_cpu.mul(&out_cpu).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu
        .get(&x_cpu)
        .unwrap()
        .expect("x は loss に到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let (out_naive, _) = topk_with_options(&x_naive, 3, opts).unwrap();
    let loss_naive = out_naive.mul(&out_naive).unwrap().sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive
        .get(&x_naive)
        .unwrap()
        .expect("x は loss に到達する");

    assert_eq!(f32_bits(dx_cpu), f32_bits(dx_naive), "topk sorted=false dX");
}

/// `unique_with_options`（`dim=None`・`return_inverse`／
/// `return_counts` あり）が CPU と NaiveOps で bit 完全一致する。
#[test]
fn cpu_unique_with_options_dim_none_bit_matches_naive_reference() {
    let data = fixture();
    let opts = UniqueOptions::default()
        .with_return_inverse(true)
        .with_return_counts(true);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let out_cpu = unique_with_options(&x_cpu, opts).unwrap();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let out_naive = unique_with_options(&x_naive, opts).unwrap();

    assert_eq!(f32_bits(&out_cpu.values), f32_bits(&out_naive.values));
    assert_eq!(
        i32_vec(out_cpu.inverse.as_ref().unwrap()),
        i32_vec(out_naive.inverse.as_ref().unwrap())
    );
    assert_eq!(
        i32_vec(out_cpu.counts.as_ref().unwrap()),
        i32_vec(out_naive.counts.as_ref().unwrap())
    );
}

/// `unique_with_options`（`dim` 指定）が CPU と NaiveOps で bit 完全
/// 一致する。
#[test]
fn cpu_unique_with_options_dim_specified_bit_matches_naive_reference() {
    let data = unique_dim_fixture();
    let opts = UniqueOptions::default()
        .with_dim(0)
        .with_return_inverse(true);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let out_cpu = unique_with_options(&x_cpu, opts).unwrap();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let out_naive = unique_with_options(&x_naive, opts).unwrap();

    assert_eq!(out_cpu.values.shape(), out_naive.values.shape());
    assert_eq!(f32_bits(&out_cpu.values), f32_bits(&out_naive.values));
    assert_eq!(
        i32_vec(out_cpu.inverse.as_ref().unwrap()),
        i32_vec(out_naive.inverse.as_ref().unwrap())
    );
}

/// `unique_consecutive` が CPU と NaiveOps で bit 完全一致する。
#[test]
fn cpu_unique_consecutive_bit_matches_naive_reference() {
    let data = Tensor::new(vec![1.0, 1.0, 2.0, 1.0, 1.0], &[5]).expect("test fixture: shape 一致");
    let opts = UniqueOptions::default()
        .with_return_inverse(true)
        .with_return_counts(true);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let out_cpu = unique_consecutive(&x_cpu, opts).unwrap();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let out_naive = unique_consecutive(&x_naive, opts).unwrap();

    assert_eq!(f32_bits(&out_cpu.values), f32_bits(&out_naive.values));
    assert_eq!(
        i32_vec(out_cpu.inverse.as_ref().unwrap()),
        i32_vec(out_naive.inverse.as_ref().unwrap())
    );
    assert_eq!(
        i32_vec(out_cpu.counts.as_ref().unwrap()),
        i32_vec(out_naive.counts.as_ref().unwrap())
    );
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/topk-unique-2153/README.md`）。
// ---------------------------------------------------------------------

/// `unique_with_options`（`dim` 指定）の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/topk-unique-2153/README.md 参照"]
fn metal_unique_with_options_matches_cpu_reference() {
    let data = unique_dim_fixture();
    let opts = UniqueOptions::default()
        .with_dim(0)
        .with_return_inverse(true);
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);

    let out_cpu = unique_with_options(&x_cpu, opts).unwrap();
    let out_metal = unique_with_options(&x_metal, opts).unwrap();
    assert_eq!(f32_bits(&out_cpu.values), f32_bits(&out_metal.values));
    assert_eq!(
        i32_vec(out_cpu.inverse.as_ref().unwrap()),
        i32_vec(out_metal.inverse.as_ref().unwrap())
    );
}

/// `unique_with_options`（`dim` 指定）の CPU／CUDA 実機（DGX Spark
/// GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/topk-unique-2153/README.md 参照"]
fn cuda_unique_with_options_matches_cpu_reference() {
    let data = unique_dim_fixture();
    let opts = UniqueOptions::default()
        .with_dim(0)
        .with_return_inverse(true);
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);

    let out_cpu = unique_with_options(&x_cpu, opts).unwrap();
    let out_cuda = unique_with_options(&x_cuda, opts).unwrap();
    assert_eq!(f32_bits(&out_cpu.values), f32_bits(&out_cuda.values));
    assert_eq!(
        i32_vec(out_cpu.inverse.as_ref().unwrap()),
        i32_vec(out_cuda.inverse.as_ref().unwrap())
    );
}

/// `topk_with_options`（`sorted=false`）の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/topk-unique-2153/README.md 参照"]
fn metal_topk_sorted_false_matches_cpu_reference() {
    let data = fixture();
    let opts = TopkOptions::default().with_dim(1).with_sorted(false);
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);

    let (out_cpu, idx_cpu) = topk_with_options(&x_cpu, 3, opts).unwrap();
    let (out_metal, idx_metal) = topk_with_options(&x_metal, 3, opts).unwrap();
    assert_eq!(
        f32_bits(&out_cpu.to_tensor()),
        f32_bits(&out_metal.to_tensor())
    );
    assert_eq!(i32_vec(&idx_cpu), i32_vec(&idx_metal));
}

/// `topk_with_options`（`sorted=false`）の CPU／CUDA 実機（DGX Spark
/// GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/topk-unique-2153/README.md 参照"]
fn cuda_topk_sorted_false_matches_cpu_reference() {
    let data = fixture();
    let opts = TopkOptions::default().with_dim(1).with_sorted(false);
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);

    let (out_cpu, idx_cpu) = topk_with_options(&x_cpu, 3, opts).unwrap();
    let (out_cuda, idx_cuda) = topk_with_options(&x_cuda, 3, opts).unwrap();
    assert_eq!(
        f32_bits(&out_cpu.to_tensor()),
        f32_bits(&out_cuda.to_tensor())
    );
    assert_eq!(i32_vec(&idx_cpu), i32_vec(&idx_cuda));
}
