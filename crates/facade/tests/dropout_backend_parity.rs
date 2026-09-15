//! `Var::dropout`（イシュー #1603）の facade 到達経路（既存 `Var`
//! 再エクスポート経由。新規 `pub use`／`pub fn` は `compat::
//! Sequential::add_dropout` の 1 件のみ）の 3 バックエンド受け入れ
//! 条件対応テスト（`index_ops_backend_parity.rs` と同型）。
//!
//! マスク生成はホスト側のグローバル RNG（`docs/rng-global-contract-
//! design.md`）のため `manual_seed` を各バックエンド呼び出し直前で
//! 打ち直せば同一マスクを再現できる——`mask` は `BackendOps::mul`
//! （forward）・そのVJP（backward）にのみ渡るため、3 バックエンドとも
//! bit 完全一致（乗算 1 回のみで丸め差が生じない）を主張できる
//! （`.claude/rules/coding-rust.md` の複合判定〈相対誤差／絶対誤差〉
//! ではなく厳密ゼロ fail 判定を用いる）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で forward／
//!   backward を bit 同一＋`assert_parity` 併記で突き合わせる。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` を CPU `tape_for`
//!   と比較する（各バックエンド呼び出し直前に `manual_seed` を打ち直す）。

use std::sync::Mutex;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// グローバル RNG 状態を書き換えるため、本ファイル内のテスト同士を
/// 直列化する（`crates/facade/tests/rng_tensor_generation.rs` と同型）。
fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`index_ops_backend_parity.rs`
/// の `VarSource` と同じ理由・同じ構成）。
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

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

// --- 属性なし（CPU vs NaiveOps） -----------------------------------------

/// dropout forward の CPU（`BackendOps::mul`）と NaiveOps（ホスト
/// `eval::mul` 参照実装）の parity。マスク乗算は丸めを伴わないため
/// bit 同一も確認する。
#[test]
fn cpu_dropout_forward_matches_naive_reference() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let shape = [4usize, 5];
    let x_data = leaf(11, &shape);

    fandhe_ai::manual_seed(9001);
    let cpu_tape = fandhe_ai::tape();
    let cpu_x = cpu_tape.make_var(&x_data);
    let cpu_out = cpu_x
        .dropout(0.4, true)
        .expect("dropout: 有効な p")
        .to_tensor();

    fandhe_ai::manual_seed(9001);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_x = naive_tape.make_var(&x_data);
    let naive_out = naive_x
        .dropout(0.4, true)
        .expect("dropout: 有効な p")
        .to_tensor();

    let cpu_slice = contiguous_slice(&cpu_out);
    let naive_slice = contiguous_slice(&naive_out);
    assert_parity("dropout forward: CPU vs NaiveOps", &cpu_slice, &naive_slice);
    assert_eq!(
        cpu_slice, naive_slice,
        "dropout forward: マスク乗算は丸めを伴わないため bit 同一のはず"
    );
}

/// dropout backward（`d_input = upstream ⊙ mask`）の CPU と NaiveOps の
/// parity。同一マスクを固定した状態で `loss = sum(dropout(x))` の勾配
/// を比較する。
#[test]
fn cpu_dropout_backward_matches_naive_reference() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let shape = [3usize, 4];
    let x_data = leaf(12, &shape);

    fandhe_ai::manual_seed(9002);
    let cpu_tape = fandhe_ai::tape();
    let cpu_x = cpu_tape.make_var(&x_data);
    let cpu_out = cpu_x.dropout(0.5, true).expect("dropout: 有効な p");
    let cpu_loss = cpu_out.sum(None).expect("全軸縮約は失敗しない");
    let cpu_grads = cpu_tape
        .backward(&cpu_loss)
        .expect("x は requires_grad の葉");
    let cpu_dx = cpu_grads
        .get(&cpu_x)
        .expect("x は requires_grad=true の葉")
        .expect("x は loss に到達する");

    fandhe_ai::manual_seed(9002);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_x = naive_tape.make_var(&x_data);
    let naive_out = naive_x.dropout(0.5, true).expect("dropout: 有効な p");
    let naive_loss = naive_out.sum(None).expect("全軸縮約は失敗しない");
    let naive_grads = naive_tape
        .backward(&naive_loss)
        .expect("x は requires_grad の葉");
    let naive_dx = naive_grads
        .get(&naive_x)
        .expect("x は requires_grad=true の葉")
        .expect("x は loss に到達する");

    let cpu_slice = contiguous_slice(cpu_dx);
    let naive_slice = contiguous_slice(naive_dx);
    assert_parity(
        "dropout backward: CPU vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
    assert_eq!(
        cpu_slice, naive_slice,
        "dropout backward: マスク乗算は丸めを伴わないため bit 同一のはず"
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---------------------------------

fn dropout_forward_on(device: Device, seed: u64, x: &Tensor<f32>) -> Tensor<f32> {
    fandhe_ai::manual_seed(seed);
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let v = tape.make_var(x);
    v.dropout(0.5, true).expect("dropout: 有効な p").to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`index_ops_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_dropout_forward_matches_cpu() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let x = leaf(21, &[4, 5]);
    let metal_out = dropout_forward_on(Device::Metal, 9101, &x);
    let cpu_out = dropout_forward_on(Device::Cpu, 9101, &x);

    let metal_slice = contiguous_slice(&metal_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "dropout forward: Metal tape_for vs CPU tape_for",
        &metal_slice,
        &cpu_slice,
    );
    assert_eq!(
        metal_slice, cpu_slice,
        "dropout: マスク乗算は丸めを伴わないため bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_dropout_forward_matches_cpu() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let x = leaf(22, &[4, 5]);
    let cuda_out = dropout_forward_on(Device::Cuda(0), 9102, &x);
    let cpu_out = dropout_forward_on(Device::Cpu, 9102, &x);

    let cuda_slice = contiguous_slice(&cuda_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "dropout forward: CUDA tape_for vs CPU tape_for",
        &cuda_slice,
        &cpu_slice,
    );
    assert_eq!(
        cuda_slice, cpu_slice,
        "dropout: マスク乗算は丸めを伴わないため bit 同一のはず"
    );
}
