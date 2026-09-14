//! `nn::MultiheadAttention`（イシュー #1640。親 #1605 sub-issue (b)）の
//! facade 横断 parity テスト（`norm_backend_parity.rs`・
//! `softmax_backend_parity.rs` と同型）。
//!
//! `MultiheadAttentionVars::new`（`crates/autodiff/src/nn/attention.rs`
//! §D7）を facade からの到達経路として使う: facade の `Tape`（`fandhe_ai::
//! Tape`）は内部 `fandhe_ai_autodiff::Tape` を `pub(crate)` フィールドに
//! 保持するのみでクレート外へ公開しないため、`MultiheadAttention::bind`
//! （crate-internal）を facade テストから直接呼べない。代わりに
//! `LinearVars`（pub フィールド）を `tape.var(&tensor)` で自前構築し、
//! `MultiheadAttentionVars::new` へ渡す。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`。`gemm_batched`
//!   オーバーライド・softmax 行カーネル）と `fandhe_ai_autodiff::Tape::
//!   new()`（NaiveOps。`gemm_batched` は per-batch 合成既定・softmax は
//!   ホストフォールバック）で同一の固定パラメータから forward／backward
//!   を実行し parity を突合する（mask なし・causal の 2 ケース）。
//! - `#[ignore]`: `tape_for(Device::Metal)`／`tape_for(Device::Cuda(0))`
//!   の同経路を CPU tape と突合（forward のみ。実機必須）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::nn::{LinearVars, MultiheadAttentionVars};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`norm_backend_parity.rs` の
/// `VarSource` と同じ理由・同じ構成）。
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

const B: usize = 2;
const L: usize = 3;
const S: usize = 4;
const E: usize = 4;
const NUM_HEADS: usize = 2;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture: shape とデータ長は事前に一致させている")
}

fn seq(seed: i64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((seed + i as i64) % 17 - 8) as f32 * 0.03)
        .collect()
}

fn query_fixture() -> Tensor<f32> {
    t(seq(101, B * L * E), &[B, L, E])
}

fn key_fixture() -> Tensor<f32> {
    t(seq(111, B * S * E), &[B, S, E])
}

fn value_fixture() -> Tensor<f32> {
    t(seq(121, B * S * E), &[B, S, E])
}

fn q_weight() -> Tensor<f32> {
    t(seq(1, E * E), &[E, E])
}
fn q_bias() -> Tensor<f32> {
    t(seq(2, E), &[E])
}
fn k_weight() -> Tensor<f32> {
    t(seq(3, E * E), &[E, E])
}
fn k_bias() -> Tensor<f32> {
    t(seq(4, E), &[E])
}
fn v_weight() -> Tensor<f32> {
    t(seq(5, E * E), &[E, E])
}
fn v_bias() -> Tensor<f32> {
    t(seq(6, E), &[E])
}
fn out_weight() -> Tensor<f32> {
    t(seq(7, E * E), &[E, E])
}
fn out_bias() -> Tensor<f32> {
    t(seq(8, E), &[E])
}

fn target_fixture() -> Tensor<f32> {
    t(seq(131, B * L * E), &[B, L, E])
}

/// 8 パラメータを与えた `tape` 上へ登録し `MultiheadAttentionVars` を
/// 構築する（`Tape::var` を 4 層 × 2〈weight/bias〉分呼ぶだけの薄い
/// ヘルパー。モジュール doc 参照）。
fn build_vars<'t>(tape: &'t impl VarSource) -> MultiheadAttentionVars<'t> {
    let q = LinearVars {
        weight: tape.make_var(&q_weight()),
        bias: Some(tape.make_var(&q_bias())),
    };
    let k = LinearVars {
        weight: tape.make_var(&k_weight()),
        bias: Some(tape.make_var(&k_bias())),
    };
    let v = LinearVars {
        weight: tape.make_var(&v_weight()),
        bias: Some(tape.make_var(&v_bias())),
    };
    let out = LinearVars {
        weight: tape.make_var(&out_weight()),
        bias: Some(tape.make_var(&out_bias())),
    };
    MultiheadAttentionVars::new(NUM_HEADS, q, k, v, out)
        .expect("fixture: 4 層とも [E,E]・bias 全 Some・E % NUM_HEADS == 0")
}

// --- forward parity（属性なし。CPU vs NaiveOps）------------------------

#[test]
fn cpu_forward_matches_naive_reference_no_mask() {
    let cpu_tape = fandhe_ai::tape();
    let vars_cpu = build_vars(&cpu_tape);
    let q_cpu = cpu_tape.make_var(&query_fixture());
    let k_cpu = cpu_tape.make_var(&key_fixture());
    let v_cpu = cpu_tape.make_var(&value_fixture());
    let out_cpu = vars_cpu
        .forward(&q_cpu, &k_cpu, &v_cpu, None, false)
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let vars_naive = build_vars(&naive_tape);
    let q_naive = naive_tape.make_var(&query_fixture());
    let k_naive = naive_tape.make_var(&key_fixture());
    let v_naive = naive_tape.make_var(&value_fixture());
    let out_naive = vars_naive
        .forward(&q_naive, &k_naive, &v_naive, None, false)
        .unwrap()
        .to_tensor();

    assert_parity(
        "MultiheadAttention forward（no mask）: CpuBackendOps vs NaiveOps",
        out_cpu.as_slice().expect("contiguous"),
        out_naive.as_slice().expect("contiguous"),
    );
}

#[test]
fn cpu_forward_matches_naive_reference_causal_self_attention() {
    let x = t(seq(141, B * L * E), &[B, L, E]);

    let cpu_tape = fandhe_ai::tape();
    let vars_cpu = build_vars(&cpu_tape);
    let x_cpu = cpu_tape.make_var(&x);
    let out_cpu = vars_cpu
        .forward(&x_cpu, &x_cpu, &x_cpu, None, true)
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let vars_naive = build_vars(&naive_tape);
    let x_naive = naive_tape.make_var(&x);
    let out_naive = vars_naive
        .forward(&x_naive, &x_naive, &x_naive, None, true)
        .unwrap()
        .to_tensor();

    assert_parity(
        "MultiheadAttention forward（causal self-attention）: CpuBackendOps vs NaiveOps",
        out_cpu.as_slice().expect("contiguous"),
        out_naive.as_slice().expect("contiguous"),
    );
}

// --- backward parity（属性なし。CPU vs NaiveOps）-----------------------

#[test]
fn cpu_backward_matches_naive_reference_no_mask() {
    let cpu_tape = fandhe_ai::tape();
    let vars_cpu = build_vars(&cpu_tape);
    let q_cpu = cpu_tape.make_var(&query_fixture());
    let k_cpu = cpu_tape.make_var(&key_fixture());
    let v_cpu = cpu_tape.make_var(&value_fixture());
    let t_cpu = cpu_tape.make_var(&target_fixture());
    let out_cpu = vars_cpu
        .forward(&q_cpu, &k_cpu, &v_cpu, None, false)
        .unwrap();
    let loss_cpu = out_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let vars_naive = build_vars(&naive_tape);
    let q_naive = naive_tape.make_var(&query_fixture());
    let k_naive = naive_tape.make_var(&key_fixture());
    let v_naive = naive_tape.make_var(&value_fixture());
    let t_naive = naive_tape.make_var(&target_fixture());
    let out_naive = vars_naive
        .forward(&q_naive, &k_naive, &v_naive, None, false)
        .unwrap();
    let loss_naive = out_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();

    for (label, cpu_var, naive_var) in [
        ("q.weight", &vars_cpu.q.weight, &vars_naive.q.weight),
        ("k.weight", &vars_cpu.k.weight, &vars_naive.k.weight),
        ("v.weight", &vars_cpu.v.weight, &vars_naive.v.weight),
        ("out.weight", &vars_cpu.out.weight, &vars_naive.out.weight),
    ] {
        let g_cpu = grads_cpu.get(cpu_var).unwrap().expect("到達する");
        let g_naive = grads_naive.get(naive_var).unwrap().expect("到達する");
        assert_parity(
            &format!("MultiheadAttention backward（{label}）: CpuBackendOps vs NaiveOps"),
            g_cpu.as_slice().expect("contiguous"),
            g_naive.as_slice().expect("contiguous"),
        );
    }
    let dq_cpu = grads_cpu.get(&q_cpu).unwrap().expect("到達する");
    let dq_naive = grads_naive.get(&q_naive).unwrap().expect("到達する");
    assert_parity(
        "MultiheadAttention backward（query 入力）: CpuBackendOps vs NaiveOps",
        dq_cpu.as_slice().expect("contiguous"),
        dq_naive.as_slice().expect("contiguous"),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。forward のみ）-----------------

fn forward_on(device: Device, causal: bool) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let vars = build_vars(&tape);
    if causal {
        let x = tape.make_var(&t(seq(141, B * L * E), &[B, L, E]));
        vars.forward(&x, &x, &x, None, true).unwrap().to_tensor()
    } else {
        let q = tape.make_var(&query_fixture());
        let k = tape.make_var(&key_fixture());
        let v = tape.make_var(&value_fixture());
        vars.forward(&q, &k, &v, None, false).unwrap().to_tensor()
    }
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある（`#[ignore]`
// は実行のみをスキップしコンパイルはスキップしないため。
// `norm_backend_parity.rs`・`softmax_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_forward_matches_cpu_no_mask() {
    let metal_out = forward_on(Device::Metal, false);
    let cpu_out = forward_on(Device::Cpu, false);
    assert_parity(
        "MultiheadAttention forward（no mask）: Metal vs CPU",
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_forward_matches_cpu_causal() {
    let metal_out = forward_on(Device::Metal, true);
    let cpu_out = forward_on(Device::Cpu, true);
    assert_parity(
        "MultiheadAttention forward（causal）: Metal vs CPU",
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_forward_matches_cpu_no_mask() {
    let cuda_out = forward_on(Device::Cuda(0), false);
    let cpu_out = forward_on(Device::Cpu, false);
    assert_parity(
        "MultiheadAttention forward（no mask）: CUDA vs CPU",
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_forward_matches_cpu_causal() {
    let cuda_out = forward_on(Device::Cuda(0), true);
    let cpu_out = forward_on(Device::Cpu, true);
    assert_parity(
        "MultiheadAttention forward（causal）: CUDA vs CPU",
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}
