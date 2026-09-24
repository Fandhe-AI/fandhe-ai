//! `fandhe_ai_autodiff::bool_ops`（イシュー #2141・facade 非公開の内部
//! 入口。`crates/autodiff/src/bool_ops.rs` モジュール doc 参照）の
//! バックエンド間 parity テスト（`cast_backend_parity.rs`／
//! `unique_backend_parity.rs` と同型）。
//!
//! `bool_ops` は facade から再エクスポートされないため、本テストは
//! `fandhe_ai_autodiff::bool_ops::*` を直接 use する（facade の dev
//! 依存に `fandhe-ai-autodiff` が既に含まれている。`cast_backend_
//! parity.rs` と同じ経路）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`。バックエンド実装
//!   `None` の場合は常にホスト参照実装へフォールバックする）で、比較
//!   6 種・logical の合成・`masked_select` の出力を突き合わせる。
//!   比較 6 種は `scalar_binary_with_fallback`（既存比較カーネル
//!   再利用）→ `cast_from_f32_with_fallback::<bool>`（既存 cast カーネル
//!   再利用）の合成のため、両カーネルが既に bit 完全一致契約
//!   （`docs/tensor-core-cast-design.md`）を持つ。判定は bit 完全一致
//!   （bool は値一致、f32 は `to_bits()` 一致。NaN はクラス一致——
//!   `cast_backend_parity.rs` と同じ扱い）。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` で同じ経路を CPU
//!   tape と比較する。実機（DGX Spark GB10／Apple Silicon）への到達
//!   手段が本エージェント実行環境にないため未実施のまま Mac／GB10
//!   セッションへ申し送る（`docs/perf/logs/bool-ops-2141/README.md`）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::bool_ops::{
    eq_bool, ge_bool, gt_bool, le_bool, logical_and, logical_not, logical_or, lt_bool,
    masked_select, ne_bool,
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

fn f32_fixture() -> Tensor<f32> {
    Tensor::new(
        vec![
            1.5,
            -2.5,
            0.0,
            -0.0,
            3.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ],
        &[8],
    )
    .expect("test fixture: shape 一致")
}

fn f32_fixture_b() -> Tensor<f32> {
    Tensor::new(
        vec![1.5, -3.0, 0.0, 0.0, 2.0, 1.0, f32::INFINITY, 0.0],
        &[8],
    )
    .expect("test fixture: shape 一致")
}

fn bool_bits(t: &Tensor<bool>) -> Vec<bool> {
    t.contiguous().host_slice().into_owned()
}

fn f32_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// 比較 6 種を CPU（`fandhe_ai::tape()`）と NaiveOps
/// （`fandhe_ai_autodiff::Tape::new()`）の両方で評価し、bit 完全一致を
/// 確認する共通ヘルパー。
#[test]
fn cpu_compare_bool_matches_naive_reference() {
    let a_data = f32_fixture();
    let b_data = f32_fixture_b();

    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&a_data);
    let b_cpu = cpu_tape.make_var(&b_data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&a_data);
    let b_naive = naive_tape.make_var(&b_data);

    assert_eq!(
        bool_bits(&gt_bool(&a_cpu, &b_cpu).unwrap()),
        bool_bits(&gt_bool(&a_naive, &b_naive).unwrap())
    );
    assert_eq!(
        bool_bits(&ge_bool(&a_cpu, &b_cpu).unwrap()),
        bool_bits(&ge_bool(&a_naive, &b_naive).unwrap())
    );
    assert_eq!(
        bool_bits(&lt_bool(&a_cpu, &b_cpu).unwrap()),
        bool_bits(&lt_bool(&a_naive, &b_naive).unwrap())
    );
    assert_eq!(
        bool_bits(&le_bool(&a_cpu, &b_cpu).unwrap()),
        bool_bits(&le_bool(&a_naive, &b_naive).unwrap())
    );
    assert_eq!(
        bool_bits(&eq_bool(&a_cpu, &b_cpu).unwrap()),
        bool_bits(&eq_bool(&a_naive, &b_naive).unwrap())
    );
    assert_eq!(
        bool_bits(&ne_bool(&a_cpu, &b_cpu).unwrap()),
        bool_bits(&ne_bool(&a_naive, &b_naive).unwrap())
    );
}

/// logical の合成（`logical_and(gt_bool(x,a), lt_bool(x,b))` 相当）を
/// CPU と NaiveOps で比較する。logical 自体はホストのみで計算する
/// ため（モジュール doc 参照）、入力側の比較 6 種の parity が担保
/// できれば合成結果も一致するはずだが、経路全体を通して固定する。
#[test]
fn cpu_logical_composition_matches_naive_reference() {
    let x_data = f32_fixture();
    let a_data = Tensor::new(vec![0.0f32; 8], &[8]).unwrap();
    let b_data = f32_fixture_b();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&x_data);
    let a_cpu = cpu_tape.make_var(&a_data);
    let b_cpu = cpu_tape.make_var(&b_data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&x_data);
    let a_naive = naive_tape.make_var(&a_data);
    let b_naive = naive_tape.make_var(&b_data);

    let mask_cpu = logical_and(
        &gt_bool(&x_cpu, &a_cpu).unwrap(),
        &lt_bool(&x_cpu, &b_cpu).unwrap(),
    )
    .unwrap();
    let mask_naive = logical_and(
        &gt_bool(&x_naive, &a_naive).unwrap(),
        &lt_bool(&x_naive, &b_naive).unwrap(),
    )
    .unwrap();
    assert_eq!(bool_bits(&mask_cpu), bool_bits(&mask_naive));

    let or_cpu = logical_or(
        &gt_bool(&x_cpu, &a_cpu).unwrap(),
        &lt_bool(&x_cpu, &b_cpu).unwrap(),
    )
    .unwrap();
    let or_naive = logical_or(
        &gt_bool(&x_naive, &a_naive).unwrap(),
        &lt_bool(&x_naive, &b_naive).unwrap(),
    )
    .unwrap();
    assert_eq!(bool_bits(&or_cpu), bool_bits(&or_naive));

    let not_cpu = logical_not(&mask_cpu).unwrap();
    let not_naive = logical_not(&mask_naive).unwrap();
    assert_eq!(bool_bits(&not_cpu), bool_bits(&not_naive));
}

/// `masked_select` を CPU と NaiveOps で比較する。マスクは比較 6 種の
/// 出力をそのまま使う（`Var::where_cond` 等と同じ接続）。
#[test]
fn cpu_masked_select_matches_naive_reference() {
    let x_data = f32_fixture();
    let zero = Tensor::new(vec![0.0f32; 8], &[8]).unwrap();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&x_data);
    let zero_cpu = cpu_tape.make_var(&zero);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&x_data);
    let zero_naive = naive_tape.make_var(&zero);

    let mask_cpu = gt_bool(&x_cpu, &zero_cpu).unwrap();
    let mask_naive = gt_bool(&x_naive, &zero_naive).unwrap();

    let sel_cpu = masked_select(&x_cpu, &mask_cpu).unwrap();
    let sel_naive = masked_select(&x_naive, &mask_naive).unwrap();
    assert_eq!(f32_bits(&sel_cpu), f32_bits(&sel_naive));
}

// --- `#[ignore]`: 実機依存（CUDA／Metal） ---

fn compare_bool_on(device: Device, a_data: &Tensor<f32>, b_data: &Tensor<f32>) -> Tensor<bool> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = tape.var(a_data);
    let b = tape.var(b_data);
    gt_bool(&a, &b).expect("gt_bool は常に成功する")
}

fn masked_select_on(device: Device, x_data: &Tensor<f32>, mask: &Tensor<bool>) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.var(x_data);
    masked_select(&x, mask).expect("masked_select は常に成功する")
}

/// `device` 上で比較（`gt_bool`）・`masked_select` が CPU と bit 完全
/// 一致することを検証する共通ヘルパー。
fn assert_device_matches_cpu(device: Device) {
    let a_data = f32_fixture();
    let b_data = f32_fixture_b();

    let dev_cmp = compare_bool_on(device, &a_data, &b_data);
    let cpu_cmp = compare_bool_on(Device::Cpu, &a_data, &b_data);
    assert_eq!(bool_bits(&dev_cmp), bool_bits(&cpu_cmp));

    let dev_sel = masked_select_on(device, &a_data, &dev_cmp);
    let cpu_sel = masked_select_on(Device::Cpu, &a_data, &cpu_cmp);
    assert_eq!(f32_bits(&dev_sel), f32_bits(&cpu_sel));
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`cast_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない。bool_ops は\
            既存の比較・cast カーネル（イシュー #1712／#1751）を再利用する\
            合成のため実装は既に存在するが、本エージェント実行環境に\
            Apple Silicon 実機への到達手段がないため未実測（イシュー #2141）"]
fn metal_bool_ops_matches_cpu() {
    assert_device_matches_cpu(Device::Metal);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。bool_ops は既存の比較・cast\
            カーネル（イシュー #1712／#1751）を再利用する合成のため実装は\
            既に存在するが、本エージェント実行環境に CUDA 実機への到達\
            手段がないため未実測（イシュー #2141）"]
fn cuda_bool_ops_matches_cpu() {
    assert_device_matches_cpu(Device::Cuda(0));
}
