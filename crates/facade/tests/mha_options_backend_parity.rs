//! `nn::MultiheadAttention` のオプション（イシュー #2163・親 #2131。
//! `batch_first`・`kdim`/`vdim`・`key_padding_mask`）の facade 横断
//! parity テスト（`mha_backend_parity.rs` と同型）。
//!
//! `MultiheadAttentionVars::new_with_config`（`crates/autodiff/src/nn/
//! attention.rs`）を facade からの到達経路として使う（`mha_backend_
//! parity.rs` doc 参照。`MultiheadAttention::bind` は crate-internal
//! のため facade テストから直接呼べない）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で forward を突合
//!   する: (a) `kdim`/`vdim` 非対称の cross-attention、(b)
//!   `batch_first=false`、(c) `key_padding_mask` + `is_causal`。
//! - `#[ignore]`: `tape_for(Device::Metal)`／`tape_for(Device::Cuda(0))`
//!   の forward を CPU と突合する（実機必須。CUDA・Metal 実機は本環境
//!   では未実測——`docs/perf/logs/mha-options-2163/README.md` へ申し
//!   送る）。
//!
//! 既存 `mha_backend_parity.rs`・`kv_cache_backend_parity.rs`・
//! `checkpoint_backend_bit_identity.rs` 等は無修正で green を維持する
//! （既定経路〈`batch_first=true`・`key_padding_mask=None`〉の bit 同一
//! 保証の CI 側根拠。`attention.rs` モジュール doc 参照）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::nn::{LinearVars, MultiheadAttentionConfig, MultiheadAttentionVars};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`mha_backend_parity.rs` の
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

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture: shape とデータ長は事前に一致させている")
}

fn seq(seed: i64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((seed + i as i64) % 17 - 8) as f32 * 0.03)
        .collect()
}

/// `config` に従って `LinearVars` 4 個を組み立てる（`mha_backend_
/// parity.rs::build_vars` の config 対応版）。
fn build_vars_with_config<'t>(
    tape: &'t impl VarSource,
    config: &MultiheadAttentionConfig,
) -> MultiheadAttentionVars<'t> {
    let e = config.embed_dim();
    let kdim = config.kdim();
    let vdim = config.vdim();
    let q = LinearVars {
        weight: tape.make_var(&t(seq(1, e * e), &[e, e])),
        bias: Some(tape.make_var(&t(seq(2, e), &[e]))),
    };
    let k = LinearVars {
        weight: tape.make_var(&t(seq(3, kdim * e), &[kdim, e])),
        bias: Some(tape.make_var(&t(seq(4, e), &[e]))),
    };
    let v = LinearVars {
        weight: tape.make_var(&t(seq(5, vdim * e), &[vdim, e])),
        bias: Some(tape.make_var(&t(seq(6, e), &[e]))),
    };
    let out = LinearVars {
        weight: tape.make_var(&t(seq(7, e * e), &[e, e])),
        bias: Some(tape.make_var(&t(seq(8, e), &[e]))),
    };
    MultiheadAttentionVars::new_with_config(config, q, k, v, out)
        .expect("fixture: config と 4 層の shape は事前に一致させている")
}

// --- (a) kdim/vdim 非対称の cross-attention（属性なし。CPU vs NaiveOps）

#[test]
fn cpu_forward_matches_naive_reference_asymmetric_kdim_vdim() {
    let (b, l, s, e, h) = (2, 3, 4, 4, 2);
    let cfg = MultiheadAttentionConfig::new(e, h)
        .with_kdim(6)
        .with_vdim(5);

    let cpu_tape = fandhe_ai::tape();
    let vars_cpu = build_vars_with_config(&cpu_tape, &cfg);
    let q_cpu = cpu_tape.make_var(&t(seq(101, b * l * e), &[b, l, e]));
    let k_cpu = cpu_tape.make_var(&t(seq(111, b * s * 6), &[b, s, 6]));
    let v_cpu = cpu_tape.make_var(&t(seq(121, b * s * 5), &[b, s, 5]));
    let out_cpu = vars_cpu
        .forward(&q_cpu, &k_cpu, &v_cpu, None, false)
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let vars_naive = build_vars_with_config(&naive_tape, &cfg);
    let q_naive = naive_tape.make_var(&t(seq(101, b * l * e), &[b, l, e]));
    let k_naive = naive_tape.make_var(&t(seq(111, b * s * 6), &[b, s, 6]));
    let v_naive = naive_tape.make_var(&t(seq(121, b * s * 5), &[b, s, 5]));
    let out_naive = vars_naive
        .forward(&q_naive, &k_naive, &v_naive, None, false)
        .unwrap()
        .to_tensor();

    assert_parity(
        "MultiheadAttention forward（kdim/vdim 非対称）: CpuBackendOps vs NaiveOps",
        out_cpu.as_slice().expect("contiguous"),
        out_naive.as_slice().expect("contiguous"),
    );
}

// --- (b) batch_first=false（属性なし。CPU vs NaiveOps）------------------

#[test]
fn cpu_forward_matches_naive_reference_batch_first_false() {
    let (b, l, e, h) = (2, 3, 4, 2);
    let cfg = MultiheadAttentionConfig::new(e, h).with_batch_first(false);
    let x = t(seq(141, l * b * e), &[l, b, e]);

    let cpu_tape = fandhe_ai::tape();
    let vars_cpu = build_vars_with_config(&cpu_tape, &cfg);
    let x_cpu = cpu_tape.make_var(&x);
    let out_cpu = vars_cpu
        .forward(&x_cpu, &x_cpu, &x_cpu, None, false)
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let vars_naive = build_vars_with_config(&naive_tape, &cfg);
    let x_naive = naive_tape.make_var(&x);
    let out_naive = vars_naive
        .forward(&x_naive, &x_naive, &x_naive, None, false)
        .unwrap()
        .to_tensor();

    assert_parity(
        "MultiheadAttention forward（batch_first=false）: CpuBackendOps vs NaiveOps",
        out_cpu.as_slice().expect("contiguous"),
        out_naive.as_slice().expect("contiguous"),
    );
}

// --- (c) key_padding_mask + is_causal（属性なし。CPU vs NaiveOps）------

#[test]
fn cpu_forward_matches_naive_reference_key_padding_mask_with_causal() {
    let (b, l, s, e, h) = (2, 3, 3, 4, 2);
    let cfg = MultiheadAttentionConfig::new(e, h);
    let x = t(seq(151, b * l * e), &[b, l, e]);
    // バッチ 0 は末尾 key を無効化、バッチ 1 は全 key 有効。
    let kpm = Tensor::new(vec![true, true, false, true, true, true], &[b, s]).unwrap();

    let cpu_tape = fandhe_ai::tape();
    let vars_cpu = build_vars_with_config(&cpu_tape, &cfg);
    let x_cpu = cpu_tape.make_var(&x);
    let out_cpu = vars_cpu
        .forward_with_key_padding_mask(&x_cpu, &x_cpu, &x_cpu, None, Some(&kpm), true)
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let vars_naive = build_vars_with_config(&naive_tape, &cfg);
    let x_naive = naive_tape.make_var(&x);
    let out_naive = vars_naive
        .forward_with_key_padding_mask(&x_naive, &x_naive, &x_naive, None, Some(&kpm), true)
        .unwrap()
        .to_tensor();

    assert_parity(
        "MultiheadAttention forward（key_padding_mask + is_causal）: CpuBackendOps vs NaiveOps",
        out_cpu.as_slice().expect("contiguous"),
        out_naive.as_slice().expect("contiguous"),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。forward のみ）-----------------

fn forward_asymmetric_on(device: Device) -> Tensor<f32> {
    let (b, l, s, e, h) = (2, 3, 4, 4, 2);
    let cfg = MultiheadAttentionConfig::new(e, h)
        .with_kdim(6)
        .with_vdim(5);
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let vars = build_vars_with_config(&tape, &cfg);
    let q = tape.make_var(&t(seq(101, b * l * e), &[b, l, e]));
    let k = tape.make_var(&t(seq(111, b * s * 6), &[b, s, 6]));
    let v = tape.make_var(&t(seq(121, b * s * 5), &[b, s, 5]));
    vars.forward(&q, &k, &v, None, false).unwrap().to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`mha_backend_parity.rs` の同一コメント参照）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_forward_matches_cpu_asymmetric_kdim_vdim() {
    let metal_out = forward_asymmetric_on(Device::Metal);
    let cpu_out = forward_asymmetric_on(Device::Cpu);
    assert_parity(
        "MultiheadAttention forward（kdim/vdim 非対称）: Metal vs CPU",
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_forward_matches_cpu_asymmetric_kdim_vdim() {
    let cuda_out = forward_asymmetric_on(Device::Cuda(0));
    let cpu_out = forward_asymmetric_on(Device::Cpu);
    assert_parity(
        "MultiheadAttention forward（kdim/vdim 非対称）: CUDA vs CPU",
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}
