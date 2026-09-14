//! `Var::unique`（イシュー #1734）の facade 到達経路（既存 `Var`
//! 再エクスポート経由。新規 `pub use`／`pub fn` は追加していない。
//! `docs/unique-facade-exposure-decision.md` 参照）の受け入れ条件対応
//! テスト（`index_ops_backend_parity.rs` と同型）。
//!
//! `unique` は非微分演算（`Op` を tape に記録しない・detached な
//! `Tensor<f32>` を返す）ため、他の facade parity テストと異なり
//! backward の突合は対象外——forward（値そのもの）のみを検証する。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で `Var::unique` の
//!   出力を突き合わせる（選択演算のため bit 同一）。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU tape
//!   と比較する（bit 同一）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
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

fn leaf_with_dups(seed: u64, numel: usize) -> Tensor<f32> {
    // 値域を狭くして重複を意図的に発生させる（`[-1, 1)` の一様乱数を
    // 10 段階へ丸める）。
    let raw = Xorshift64Star::new(seed).fill_vec(numel);
    let data: Vec<f32> = raw.into_iter().map(|v| ((v * 5.0).round()) / 5.0).collect();
    Tensor::new(data, &[numel]).expect("leaf_with_dups: shape 一致")
}

fn contiguous_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

// --- 属性なし: CPU vs NaiveOps ---

/// `Var::unique` forward の CPU（`BackendOps::unique`）と NaiveOps
/// （ホスト `eval::unique` 参照実装）の parity。選択演算のため bit
/// 同一を確認する。
#[test]
fn cpu_unique_matches_naive_reference() {
    let x_data = leaf_with_dups(1, 200);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&x_data);
    let cpu_out = x_cpu.unique().expect("unique は常に成功する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&x_data);
    let naive_out = x_naive.unique().expect("unique は常に成功する");

    assert_eq!(cpu_out.shape(), naive_out.shape());
    assert_eq!(
        contiguous_bits(&cpu_out),
        contiguous_bits(&naive_out),
        "unique: 選択演算は丸めを伴わないため bit 同一のはず"
    );
}

/// `Var::unique` が tape ノードを追加しないこと（`docs/
/// unique-facade-exposure-decision.md` の非微分演算契約）を facade
/// 経由でも確認する。
/// `facade::Tape` はノード数アクセサを公開しないため、tape ノードを
/// 追加しないという非微分演算契約自体の検証は
/// `crates/autodiff/tests/tape_recording.rs::
/// unique_does_not_record_a_tape_node`（`fandhe_ai_autodiff::Tape` を
/// 直接使う層）で行う。本テストは facade 経由でも同じ出力形状契約
/// （`0 <= m <= numel`）が成り立つことのみを確認する。
#[test]
fn facade_unique_output_shape_is_bounded_by_input_numel() {
    let tape = fandhe_ai::tape();
    let x = tape.make_var(&leaf_with_dups(2, 50));

    let out = x.unique().expect("unique は常に成功する");
    assert!(out.shape()[0] <= 50);
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn unique_on(device: Device, seed: u64, numel: usize) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf_with_dups(seed, numel));
    x.unique().expect("unique は常に成功する")
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`index_ops_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_unique_matches_cpu() {
    let metal_out = unique_on(Device::Metal, 10, 300);
    let cpu_out = unique_on(Device::Cpu, 10, 300);

    assert_eq!(metal_out.shape(), cpu_out.shape());
    assert_eq!(
        contiguous_bits(&metal_out),
        contiguous_bits(&cpu_out),
        "unique: 選択演算は丸めを伴わないため bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_unique_matches_cpu() {
    let cuda_out = unique_on(Device::Cuda(0), 20, 300);
    let cpu_out = unique_on(Device::Cpu, 20, 300);

    assert_eq!(cuda_out.shape(), cpu_out.shape());
    assert_eq!(
        contiguous_bits(&cuda_out),
        contiguous_bits(&cpu_out),
        "unique: 選択演算は丸めを伴わないため bit 同一のはず"
    );
}
