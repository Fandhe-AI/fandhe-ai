//! `Var::cast`／`Tape::var_from`（イシュー #1750）の facade 到達経路
//! （既存 `Var` 再エクスポート経由。新規 `pub use`／`pub fn` は
//! `Tape::var_from` の 1 メソッドのみ追加。`docs/tensor-core-cast-
//! design.md` 参照）の受け入れ条件対応テスト（`unique_backend_
//! parity.rs` と同型）。
//!
//! cast は算術を含まない変換のため bit 完全一致契約（非 NaN。NaN の
//! みクラス一致）——`crates/tensor-core/src/cast.rs` モジュール doc
//! 「数値契約」参照。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`。`cast_ops` accessor
//!   既定 `None`）で 8 方向の cast 出力を突き合わせる。CPU 実装
//!   （`crates/backend-cpu/src/cast.rs`）・NaiveOps のフォールバック
//!   （accessor `None`）は両方とも `tensor_core::cast` の同一ホスト
//!   参照実装へ帰着するため bit 完全一致するはず。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU tape
//!   と比較する。**#1750 時点では CUDA／Metal 側の `CastOps` accessor
//!   が `None`（未実装）のため、これらのテストは実質「両バックエンドとも
//!   ホスト参照実装へフォールバックする」ことの確認に留まる**——#1751
//!   がカーネルを実装した後も同じテストで契約検証できる形にしている
//!   （実機実測は #1751 側へ申し送り）。

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

fn f64_bits(t: &Tensor<f64>) -> Vec<u64> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn f32_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

// --- 属性なし: CPU vs NaiveOps ---

/// f32→{f64,i32,i64,bool} の 4 方向すべてで CPU（`BackendOps::cast_ops`
/// 経由）と NaiveOps（ホスト参照実装フォールバック）の出力が一致する
/// ことを確認する。
#[test]
fn cpu_cast_from_f32_matches_naive_reference() {
    let x_data = f32_fixture();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&x_data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&x_data);

    let cpu_f64: Tensor<f64> = x_cpu.cast().expect("cast は常に成功する");
    let naive_f64: Tensor<f64> = x_naive.cast().expect("cast は常に成功する");
    assert_eq!(cpu_f64.shape(), naive_f64.shape());
    assert_eq!(f64_bits(&cpu_f64), f64_bits(&naive_f64));

    let cpu_i32: Tensor<i32> = x_cpu.cast().expect("cast は常に成功する");
    let naive_i32: Tensor<i32> = x_naive.cast().expect("cast は常に成功する");
    assert_eq!(
        cpu_i32.contiguous().host_slice().into_owned(),
        naive_i32.contiguous().host_slice().into_owned()
    );

    let cpu_i64: Tensor<i64> = x_cpu.cast().expect("cast は常に成功する");
    let naive_i64: Tensor<i64> = x_naive.cast().expect("cast は常に成功する");
    assert_eq!(
        cpu_i64.contiguous().host_slice().into_owned(),
        naive_i64.contiguous().host_slice().into_owned()
    );

    let cpu_bool: Tensor<bool> = x_cpu.cast().expect("cast は常に成功する");
    let naive_bool: Tensor<bool> = x_naive.cast().expect("cast は常に成功する");
    assert_eq!(
        cpu_bool.contiguous().host_slice().into_owned(),
        naive_bool.contiguous().host_slice().into_owned()
    );
}

/// {f64,i32,i64,bool}→f32 の 4 方向すべてで `Tape::var_from` が CPU と
/// NaiveOps の間で一致することを確認する。
#[test]
fn cpu_var_from_matches_naive_reference() {
    let cpu_tape = fandhe_ai::tape();
    let naive_tape = fandhe_ai_autodiff::Tape::new();

    let f64_in = Tensor::new(vec![1.5f64, -2.5, f64::MAX, f64::MIN], &[4]).unwrap();
    let cpu_out = cpu_tape.var_from(&f64_in).expect("var_from は常に成功する");
    let naive_out = naive_tape
        .var_from(&f64_in)
        .expect("var_from は常に成功する");
    assert_eq!(
        f32_bits(&cpu_out.to_tensor()),
        f32_bits(&naive_out.to_tensor())
    );

    let i32_in = Tensor::new(vec![i32::MIN, 0, i32::MAX], &[3]).unwrap();
    let cpu_out = cpu_tape.var_from(&i32_in).expect("var_from は常に成功する");
    let naive_out = naive_tape
        .var_from(&i32_in)
        .expect("var_from は常に成功する");
    assert_eq!(
        f32_bits(&cpu_out.to_tensor()),
        f32_bits(&naive_out.to_tensor())
    );

    let i64_in = Tensor::new(vec![i64::MIN, 0, i64::MAX], &[3]).unwrap();
    let cpu_out = cpu_tape.var_from(&i64_in).expect("var_from は常に成功する");
    let naive_out = naive_tape
        .var_from(&i64_in)
        .expect("var_from は常に成功する");
    assert_eq!(
        f32_bits(&cpu_out.to_tensor()),
        f32_bits(&naive_out.to_tensor())
    );

    let bool_in = Tensor::new(vec![true, false, true], &[3]).unwrap();
    let cpu_out = cpu_tape
        .var_from(&bool_in)
        .expect("var_from は常に成功する");
    let naive_out = naive_tape
        .var_from(&bool_in)
        .expect("var_from は常に成功する");
    assert_eq!(
        f32_bits(&cpu_out.to_tensor()),
        f32_bits(&naive_out.to_tensor())
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn cast_i32_on(device: Device, x_data: &Tensor<f32>) -> Tensor<i32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(x_data);
    x.cast().expect("cast は常に成功する")
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`unique_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない。#1750 時点では CastOps 未実装のためホストフォールバック経路の確認に留まる（#1751 へ引き継ぎ）"]
fn metal_cast_matches_cpu() {
    let x_data = f32_fixture();
    let metal_out = cast_i32_on(Device::Metal, &x_data);
    let cpu_out = cast_i32_on(Device::Cpu, &x_data);

    assert_eq!(metal_out.shape(), cpu_out.shape());
    assert_eq!(
        metal_out.contiguous().host_slice().into_owned(),
        cpu_out.contiguous().host_slice().into_owned()
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。#1750 時点では CastOps 未実装のためホストフォールバック経路の確認に留まる（#1751 へ引き継ぎ）"]
fn cuda_cast_matches_cpu() {
    let x_data = f32_fixture();
    let cuda_out = cast_i32_on(Device::Cuda(0), &x_data);
    let cpu_out = cast_i32_on(Device::Cpu, &x_data);

    assert_eq!(cuda_out.shape(), cpu_out.shape());
    assert_eq!(
        cuda_out.contiguous().host_slice().into_owned(),
        cpu_out.contiguous().host_slice().into_owned()
    );
}
