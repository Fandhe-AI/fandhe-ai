//! `Var::scaled_dot_product_attention`（イシュー #1639）の facade 到達
//! 経路（既存 `Var` 再エクスポート経由。新規 `pub use`／`pub fn` は
//! 追加していない）の受け入れ条件対応テスト（`einsum_backend_parity.rs`
//! と同型）。
//!
//! `crate::attention` は新規カーネルを追加せず既存の `matmul`／
//! `transpose`／`mul`／`masked_fill`／`softmax` への分解として実装して
//! いるため、ここでは「該当バックエンドすべてに実装」を分解によって
//! 自動的に充足していることを確認する。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`。CPU の
//!   `gemm_batched`／`masked_fill`／`softmax` 行カーネル経由）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`。ホスト参照実装）で
//!   forward・backward（dq/dk/dv）を REQ-2 統一複合判定で突き合わせる。
//!   `is_causal` ありのケースも含む（CPU softmax カーネルが `-inf`
//!   入力を受ける経路を Linux CI で通す）。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` を CPU tape と
//!   `assert_parity` で比較する（GEMM カーネルが異なるため bit 同一は
//!   主張しない）。本エージェント実行環境に実機がないため未実測の
//!   まま Mac／GB10 セッションへ申し送る。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`einsum_backend_parity.rs`
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

const L: usize = 3;
const S: usize = 4;
const E: usize = 2;
const EV: usize = 2;

/// forward の CPU（`CpuBackendOps` 経由）と NaiveOps（ホスト参照実装）
/// の parity（`is_causal` の有無で 2 パターン）。
fn run_forward_parity(is_causal: bool) {
    let cpu_tape = fandhe_ai::tape();
    let q_cpu = cpu_tape.make_var(&leaf(1, &[L, E]));
    let k_cpu = cpu_tape.make_var(&leaf(2, &[S, E]));
    let v_cpu = cpu_tape.make_var(&leaf(3, &[S, EV]));
    let out_cpu = Var::scaled_dot_product_attention(&q_cpu, &k_cpu, &v_cpu, None, is_causal, None)
        .expect("sdpa: 形状適合")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let q_naive = naive_tape.make_var(&leaf(1, &[L, E]));
    let k_naive = naive_tape.make_var(&leaf(2, &[S, E]));
    let v_naive = naive_tape.make_var(&leaf(3, &[S, EV]));
    let out_naive =
        Var::scaled_dot_product_attention(&q_naive, &k_naive, &v_naive, None, is_causal, None)
            .expect("sdpa: 形状適合")
            .to_tensor();

    assert_parity(
        &format!("fandhe_ai::tape()（CpuBackendOps 経由 sdpa。is_causal={is_causal}）vs NaiveOps"),
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

#[test]
fn cpu_sdpa_forward_matches_naive_reference_no_mask() {
    run_forward_parity(false);
}

#[test]
fn cpu_sdpa_forward_matches_naive_reference_is_causal() {
    run_forward_parity(true);
}

/// backward（dq/dk/dv）の CPU と NaiveOps の parity。
fn run_backward_parity(is_causal: bool) {
    let cpu_tape = fandhe_ai::tape();
    let q_cpu = cpu_tape.make_var(&leaf(1, &[L, E]));
    let k_cpu = cpu_tape.make_var(&leaf(2, &[S, E]));
    let v_cpu = cpu_tape.make_var(&leaf(3, &[S, EV]));
    let loss_cpu = Var::scaled_dot_product_attention(&q_cpu, &k_cpu, &v_cpu, None, is_causal, None)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dq_cpu = grads_cpu.get(&q_cpu).unwrap().expect("到達する").clone();
    let dk_cpu = grads_cpu.get(&k_cpu).unwrap().expect("到達する").clone();
    let dv_cpu = grads_cpu.get(&v_cpu).unwrap().expect("到達する").clone();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let q_naive = naive_tape.make_var(&leaf(1, &[L, E]));
    let k_naive = naive_tape.make_var(&leaf(2, &[S, E]));
    let v_naive = naive_tape.make_var(&leaf(3, &[S, EV]));
    let loss_naive =
        Var::scaled_dot_product_attention(&q_naive, &k_naive, &v_naive, None, is_causal, None)
            .unwrap()
            .sum(None)
            .unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dq_naive = grads_naive.get(&q_naive).unwrap().expect("到達する");
    let dk_naive = grads_naive.get(&k_naive).unwrap().expect("到達する");
    let dv_naive = grads_naive.get(&v_naive).unwrap().expect("到達する");

    assert_parity(
        &format!("sdpa dq: CPU vs NaiveOps（is_causal={is_causal}）"),
        &contiguous_slice(&dq_cpu),
        &contiguous_slice(dq_naive),
    );
    assert_parity(
        &format!("sdpa dk: CPU vs NaiveOps（is_causal={is_causal}）"),
        &contiguous_slice(&dk_cpu),
        &contiguous_slice(dk_naive),
    );
    assert_parity(
        &format!("sdpa dv: CPU vs NaiveOps（is_causal={is_causal}）"),
        &contiguous_slice(&dv_cpu),
        &contiguous_slice(dv_naive),
    );
}

#[test]
fn cpu_sdpa_backward_matches_naive_reference_no_mask() {
    run_backward_parity(false);
}

#[test]
fn cpu_sdpa_backward_matches_naive_reference_is_causal() {
    run_backward_parity(true);
}

/// `fandhe_ai::Var::scaled_dot_product_attention` として facade から
/// 到達できること（`api_surface.rs` の既存ガードが `pub use` 追加なし
/// を機械的に確認する）。関連関数として直接呼び出せることを本テスト
/// でも型検査レベルで裏付ける。
#[test]
fn scaled_dot_product_attention_reachable_via_facade_var_reexport() {
    let tape = fandhe_ai::tape();
    let q = tape.make_var(&leaf(1, &[L, E]));
    let k = tape.make_var(&leaf(2, &[S, E]));
    let v = tape.make_var(&leaf(3, &[S, EV]));
    let out: fandhe_ai::Var<'_> =
        fandhe_ai::Var::scaled_dot_product_attention(&q, &k, &v, None, false, None)
            .expect("sdpa: 形状適合");
    assert_eq!(out.to_tensor().shape(), &[L, EV]);
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn sdpa_forward_on(device: Device, is_causal: bool) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let q = tape.make_var(&leaf(1, &[L, E]));
    let k = tape.make_var(&leaf(2, &[S, E]));
    let v = tape.make_var(&leaf(3, &[S, EV]));
    Var::scaled_dot_product_attention(&q, &k, &v, None, is_causal, None)
        .expect("sdpa: 形状適合")
        .to_tensor()
}

fn sdpa_backward_dq_on(device: Device, is_causal: bool) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let q = tape.make_var(&leaf(1, &[L, E]));
    let k = tape.make_var(&leaf(2, &[S, E]));
    let v = tape.make_var(&leaf(3, &[S, EV]));
    let loss = Var::scaled_dot_product_attention(&q, &k, &v, None, is_causal, None)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    grads.get(&q).unwrap().expect("到達する").clone()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`einsum_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_sdpa_forward_matches_cpu() {
    let metal_out = sdpa_forward_on(Device::Metal, false);
    let cpu_out = sdpa_forward_on(Device::Cpu, false);

    assert_parity(
        "sdpa forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_sdpa_backward_dq_matches_cpu() {
    let metal_dq = sdpa_backward_dq_on(Device::Metal, false);
    let cpu_dq = sdpa_backward_dq_on(Device::Cpu, false);

    assert_parity(
        "sdpa backward dq: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_dq),
        &contiguous_slice(&cpu_dq),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_sdpa_forward_matches_cpu() {
    let cuda_out = sdpa_forward_on(Device::Cuda(0), false);
    let cpu_out = sdpa_forward_on(Device::Cpu, false);

    assert_parity(
        "sdpa forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_sdpa_backward_dq_matches_cpu() {
    let cuda_dq = sdpa_backward_dq_on(Device::Cuda(0), false);
    let cpu_dq = sdpa_backward_dq_on(Device::Cpu, false);

    assert_parity(
        "sdpa backward dq: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_dq),
        &contiguous_slice(&cpu_dq),
    );
}
