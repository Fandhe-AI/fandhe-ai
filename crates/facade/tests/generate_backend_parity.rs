//! `fandhe_ai_autodiff::generate`（イシュー #2191）の facade 横断 parity
//! テスト（`kv_cache_backend_parity.rs` と同型）。
//!
//! facade は `generate`／`GenerateConfig` 相当の新規 `pub fn`／`pub
//! struct` を追加していない（`docs/facade-generate-decision.md` §8。
//! `crates/facade/tests/api_surface.rs` の否定ガードで固定）ため、本
//! テストは内部クレート `fandhe_ai_autodiff` を直接使う（facade
//! （`fandhe-ai`）の通常の `[dependencies]` であり、これは facade の
//! テストコードが内部クレートへ直接依存すること自体を妨げない設計
//! ——`kv_cache_backend_parity.rs` と同じ位置づけ）。
//!
//! - 属性なし: `CpuBackendOps`（`Tape::new_with_ops`）を使うモデルと
//!   `NaiveOps`（`Tape::new()`）を使うモデルとで、同一 prompt・同一
//!   `GenerateConfig`（Greedy）の生成 token 列が一致することを検証する。
//! - `#[ignore]`: Metal／CUDA バックエンドを使うモデルと CPU の生成
//!   token 列を突合する（実機必須）。

use fandhe_ai_autodiff::generate::{
    AutoregressiveModel, GenerateConfig, SamplingStrategy, generate,
};
use fandhe_ai_autodiff::nn::{KvCache, LinearVars, MultiheadAttentionVars};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

const V: usize = 5;
const E: usize = 4;
const NUM_HEADS: usize = 2;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture: shape とデータ長は事前に一致させている")
}

fn seq(seed: i64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((seed + i as i64 * 7) % 13 - 6) as f32 * 0.05)
        .collect()
}

/// `forward_step` ごとに新規 [`Tape`] をどのバックエンドで作るかを選ぶ
/// （`NaiveOps` は `pub(crate)` のため facade テストから直接構築できず、
/// [`Tape::new`] 経由で使う）。
#[derive(Clone, Copy)]
enum TapeBackend {
    /// `Tape::new()`（`default_ops::naive_ops`。ホスト参照実装）。
    Naive,
    /// `Tape::new_with_ops(Box::new(CpuBackendOps::new()))`。
    Cpu,
    /// `cfg(target_os = "macos")` 限定。Metal 実機必須。
    #[cfg(target_os = "macos")]
    Metal,
    /// CUDA 実機必須。
    Cuda,
}

impl TapeBackend {
    fn new_tape(self) -> Tape {
        match self {
            TapeBackend::Naive => Tape::new(),
            TapeBackend::Cpu => {
                Tape::new_with_ops(Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()))
            }
            #[cfg(target_os = "macos")]
            TapeBackend::Metal => {
                Tape::new_with_ops(Box::new(fandhe_ai_backend_metal::MetalBackendOps::new()))
            }
            TapeBackend::Cuda => {
                Tape::new_with_ops(Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(0)))
            }
        }
    }
}

/// [`fandhe_ai_autodiff::tests::nn_generate`]（`crates/autodiff/tests/
/// nn_generate.rs::TinyCausalLm`）と同型だが、`forward_step` ごとに
/// 使う [`TapeBackend`] を差し替えられるようにした版（facade 横断
/// parity のため。異なるバックエンドで同一 token 列が得られることを
/// 確認する）。
struct BackendParametrizedLm {
    embedding: Tensor<f32>,
    q_weight: Tensor<f32>,
    q_bias: Tensor<f32>,
    k_weight: Tensor<f32>,
    k_bias: Tensor<f32>,
    v_weight: Tensor<f32>,
    v_bias: Tensor<f32>,
    out_weight: Tensor<f32>,
    out_bias: Tensor<f32>,
    lm_weight: Tensor<f32>,
    lm_bias: Tensor<f32>,
    backend: TapeBackend,
}

impl BackendParametrizedLm {
    fn new(backend: TapeBackend) -> BackendParametrizedLm {
        BackendParametrizedLm {
            embedding: t(seq(1, V * E), &[V, E]),
            q_weight: t(seq(2, E * E), &[E, E]),
            q_bias: t(seq(3, E), &[E]),
            k_weight: t(seq(4, E * E), &[E, E]),
            k_bias: t(seq(5, E), &[E]),
            v_weight: t(seq(6, E * E), &[E, E]),
            v_bias: t(seq(7, E), &[E]),
            out_weight: t(seq(8, E * E), &[E, E]),
            out_bias: t(seq(9, E), &[E]),
            lm_weight: t(seq(10, E * V), &[E, V]),
            lm_bias: t(seq(11, V), &[V]),
            backend,
        }
    }
}

impl AutoregressiveModel for BackendParametrizedLm {
    fn num_kv_layers(&self) -> usize {
        1
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let tape = self.backend.new_tape();
        let embed_weight = tape.var(&self.embedding);
        let x = embed_weight.embedding(new_ids, None)?;

        let q = LinearVars {
            weight: tape.var(&self.q_weight),
            bias: Some(tape.var(&self.q_bias)),
        };
        let k = LinearVars {
            weight: tape.var(&self.k_weight),
            bias: Some(tape.var(&self.k_bias)),
        };
        let v = LinearVars {
            weight: tape.var(&self.v_weight),
            bias: Some(tape.var(&self.v_bias)),
        };
        let out = LinearVars {
            weight: tape.var(&self.out_weight),
            bias: Some(tape.var(&self.out_bias)),
        };
        let mha = MultiheadAttentionVars::new(NUM_HEADS, q, k, v, out)?;
        let attn_out = mha.forward_with_cache(&x, &x, &x, &mut caches[0])?;

        let (b, l) = (new_ids.shape()[0], new_ids.shape()[1]);
        let flat = attn_out.reshape(&[b * l, E])?;
        let lm = LinearVars {
            weight: tape.var(&self.lm_weight),
            bias: Some(tape.var(&self.lm_bias)),
        };
        let logits = lm.forward(&flat)?.reshape(&[b, l, V])?;
        Ok(logits.to_tensor())
    }
}

fn prompt_1d(data: Vec<i32>) -> Tensor<i32> {
    let len = data.len();
    Tensor::new(data, &[len]).unwrap()
}

#[test]
fn cpu_backend_ops_matches_naive_ops_for_greedy_generation() {
    let prompt = prompt_1d(vec![0, 1, 2]);
    let config = GenerateConfig::new(8, SamplingStrategy::Greedy);

    let cpu_model = BackendParametrizedLm::new(TapeBackend::Cpu);
    let naive_model = BackendParametrizedLm::new(TapeBackend::Naive);

    let cpu_out = generate(&cpu_model, &prompt, &config).unwrap();
    let naive_out = generate(&naive_model, &prompt, &config).unwrap();

    assert_eq!(
        cpu_out.contiguous().host_slice().into_owned(),
        naive_out.contiguous().host_slice().into_owned(),
        "CpuBackendOps と NaiveOps で generate() の token 列が一致しない"
    );
}

#[test]
fn cpu_backend_ops_matches_naive_ops_for_top_k_generation_with_same_seed() {
    let prompt = prompt_1d(vec![0, 1, 2]);
    let config = GenerateConfig::new(8, SamplingStrategy::TopK(3)).with_seed(77);

    let cpu_model = BackendParametrizedLm::new(TapeBackend::Cpu);
    let naive_model = BackendParametrizedLm::new(TapeBackend::Naive);

    let cpu_out = generate(&cpu_model, &prompt, &config).unwrap();
    let naive_out = generate(&naive_model, &prompt, &config).unwrap();

    assert_eq!(
        cpu_out.contiguous().host_slice().into_owned(),
        naive_out.contiguous().host_slice().into_owned(),
        "CpuBackendOps と NaiveOps で generate()（TopK・同一 seed）の \
         token 列が一致しない"
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。CPU と token 列を突合）--------

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_backend_ops_matches_cpu_backend_ops_for_greedy_generation() {
    let prompt = prompt_1d(vec![0, 1, 2]);
    let config = GenerateConfig::new(8, SamplingStrategy::Greedy);

    let metal_model = BackendParametrizedLm::new(TapeBackend::Metal);
    let cpu_model = BackendParametrizedLm::new(TapeBackend::Cpu);

    let metal_out = generate(&metal_model, &prompt, &config).unwrap();
    let cpu_out = generate(&cpu_model, &prompt, &config).unwrap();

    assert_eq!(
        metal_out.contiguous().host_slice().into_owned(),
        cpu_out.contiguous().host_slice().into_owned(),
        "Metal と CPU で generate() の token 列が一致しない"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_backend_ops_matches_cpu_backend_ops_for_greedy_generation() {
    let prompt = prompt_1d(vec![0, 1, 2]);
    let config = GenerateConfig::new(8, SamplingStrategy::Greedy);

    let cuda_model = BackendParametrizedLm::new(TapeBackend::Cuda);
    let cpu_model = BackendParametrizedLm::new(TapeBackend::Cpu);

    let cuda_out = generate(&cuda_model, &prompt, &config).unwrap();
    let cpu_out = generate(&cpu_model, &prompt, &config).unwrap();

    assert_eq!(
        cuda_out.contiguous().host_slice().into_owned(),
        cpu_out.contiguous().host_slice().into_owned(),
        "CUDA と CPU で generate() の token 列が一致しない"
    );
}
