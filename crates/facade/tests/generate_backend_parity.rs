//! `fandhe_ai_autodiff::generate`（イシュー #2191）の facade 横断 parity
//! テスト（`kv_cache_backend_parity.rs` と同型）。
//!
//! facade は #2575 で `fandhe_ai::inference::{generate, GenerateConfig, ..}` を純再エクスポート
//! として公開済み（`docs/facade-generate-decision.md` §17。公開経路の単体テストは
//! `generate_facade.rs`）。本テストは `Tape::new_with_ops` でバックエンドを選ぶ必要があり、
//! facade には `BackendOps` 注入経路が無い（REQ-12）ため、引き続き内部クレート
//! `fandhe_ai_autodiff` を直接使う（facade（`fandhe-ai`）の通常の `[dependencies]` であり、
//! これは facade のテストコードが内部クレートへ直接依存すること自体を妨げない設計
//! ——`kv_cache_backend_parity.rs` と同じ位置づけ）。
//!
//! - 属性なし: `CpuBackendOps`（`Tape::new_with_ops`）を使うモデルと
//!   `NaiveOps`（`Tape::new()`）を使うモデルとで、同一 prompt・同一
//!   `GenerateConfig`（Greedy・TopK）の生成 token 列が一致することを
//!   検証する。token 列一致だけでは同じ token が選ばれつつ logits が
//!   REQ-2 の統一複合判定から外れるケースを見逃すため（codex-review
//!   指摘・PR #2324）、`generate()` が実際に辿った token 列を
//!   `replay_capturing_logits` で再生し、各 `forward_step` の logits も
//!   `assert_parity` で突合する。
//! - `#[ignore]`: Metal／CUDA バックエンドを使うモデルと CPU の生成
//!   token 列・各 `forward_step` の logits を突合する（実機必須）。

use fandhe_ai_autodiff::generate::{
    AutoregressiveModel, GenerateConfig, SamplingStrategy, generate,
};
use fandhe_ai_autodiff::nn::{KvCache, LinearVars, MultiheadAttentionVars};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_backend_cpu::parity::assert_parity;
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

/// タイの最小 index を選ぶ最大値添字（`generate/mod.rs::greedy_argmax` と
/// 同じタイ規約。同モジュールの実体は `pub(crate)` のため本テストでは
/// 独立に再実装する）。
fn greedy_argmax(row: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = row[0];
    for (idx, &v) in row.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    best_idx
}

/// `generate()` の Greedy 経路を `AutoregressiveModel::forward_step`
/// 単位で手動再実装し（rank1・`B == 1` 限定。既存テストの prompt は
/// すべて `prompt_1d` のためこの前提で十分）、各ステップの生 logits
/// （host `Vec<f32>`。shape `[1, l_new, V]` をそのままフラット化した
/// もの）を token 列とあわせて返す。
///
/// codex-review 指摘（PR #2324・イシュー #2191）: 既存のバックエンド
/// 横断比較（`metal_backend_ops_matches_cpu_backend_ops_for_greedy_
/// generation`・`cuda_backend_ops_matches_cpu_backend_ops_for_greedy_
/// generation`）は最終 token 列のみを突合しており、同じ token が
/// 選ばれれば logits が REQ-2 の統一複合判定の許容誤差を超えて
/// 異なっていても検出できない。本関数はその不足を埋めるため、
/// `generate()` の呼び出しに加えてこの手動ループでも同一 prompt を
/// 流し、ステップごとの logits を呼び出し元へ返す（呼び出し元が
/// [`assert_parity`] でバックエンド間の logits 数値一致を検証する）。
fn run_greedy_capturing_logits<M: AutoregressiveModel>(
    model: &M,
    prompt: &Tensor<i32>,
    max_length: usize,
) -> (Vec<i32>, Vec<Vec<f32>>) {
    let prompt_len = prompt.shape()[0];
    let mut caches: Vec<KvCache> = (0..model.num_kv_layers()).map(|_| KvCache::new()).collect();
    let mut ids: Vec<i32> = prompt.contiguous().host_slice().into_owned();
    let mut step_logits: Vec<Vec<f32>> = Vec::new();

    let prompt_ids = Tensor::new(ids.clone(), &[1, prompt_len]).unwrap();
    let logits = model.forward_step(&prompt_ids, &mut caches).unwrap();
    let logits_flat: Vec<f32> = logits.contiguous().host_slice().into_owned();
    let vocab = logits.shape()[2];
    let last_row = &logits_flat[(prompt_len - 1) * vocab..prompt_len * vocab];
    let mut next_id = greedy_argmax(last_row) as i32;
    step_logits.push(logits_flat);
    ids.push(next_id);

    while ids.len() < max_length {
        let step_ids = Tensor::new(vec![next_id], &[1, 1]).unwrap();
        let logits = model.forward_step(&step_ids, &mut caches).unwrap();
        let logits_flat: Vec<f32> = logits.contiguous().host_slice().into_owned();
        next_id = greedy_argmax(&logits_flat) as i32;
        step_logits.push(logits_flat);
        ids.push(next_id);
    }

    (ids, step_logits)
}

/// `generate()` が実際に選んだ token 列（`generated_ids`。prompt を
/// 含む・呼び出し元は事前に `cpu_out == naive_out` を確認済みの前提）
/// をそのまま各 `forward_step` へ再投入し（`generate()` 本体の
/// prefill／decode ループと同じ呼び出し回数・同じ入力 token）、
/// ステップごとの生 logits を返す。
///
/// [`run_greedy_capturing_logits`] は独自に Greedy で次 token を選び
/// 直す（`SamplingStrategy` に依存しない実機〈Metal／CUDA〉向けの
/// 再導出）のに対し、本関数は `generate()` が辿った経路をそのまま
/// 再生するため `SamplingStrategy::TopK`／`Temperature` の乱数選択
/// ロジック（`sample_step`・`Generator::multinomial` 等はいずれも
/// `pub(crate)`／内部実装のためテストクレートから直接再現できない）を
/// 再実装せずに済む。codex-review 指摘（PR #2324・イシュー #2191）:
/// CPU 間比較（`cpu_backend_ops_matches_naive_ops_for_greedy_
/// generation`／`..._top_k_generation_with_same_seed`）が最終 token 列
/// のみを突合しており、同じ token が選ばれても logits が REQ-2 の
/// 統一複合判定の許容誤差を超えて異なっていても検出できない不足を
/// 埋める。
fn replay_capturing_logits<M: AutoregressiveModel>(
    model: &M,
    prompt: &Tensor<i32>,
    generated_ids: &[i32],
) -> Vec<Vec<f32>> {
    let prompt_len = prompt.shape()[0];
    let mut caches: Vec<KvCache> = (0..model.num_kv_layers()).map(|_| KvCache::new()).collect();
    let mut step_logits: Vec<Vec<f32>> = Vec::new();

    let prompt_ids = Tensor::new(generated_ids[..prompt_len].to_vec(), &[1, prompt_len])
        .expect("fixture: prompt_len は generated_ids の長さ以下");
    let logits = model.forward_step(&prompt_ids, &mut caches).unwrap();
    step_logits.push(logits.contiguous().host_slice().into_owned());

    // decode: 実際に生成された token を 1 つずつ再投入する（末尾の
    // token は generate() の decode ループでも forward_step へ渡され
    // ないため対象外——`run_greedy_capturing_logits` と同じ呼び出し
    // 回数になる）。
    for &id in &generated_ids[prompt_len..generated_ids.len() - 1] {
        let step_ids = Tensor::new(vec![id], &[1, 1]).unwrap();
        let logits = model.forward_step(&step_ids, &mut caches).unwrap();
        step_logits.push(logits.contiguous().host_slice().into_owned());
    }

    step_logits
}

#[test]
fn cpu_backend_ops_matches_naive_ops_for_greedy_generation() {
    let prompt = prompt_1d(vec![0, 1, 2]);
    let config = GenerateConfig::new(8, SamplingStrategy::Greedy);

    let cpu_model = BackendParametrizedLm::new(TapeBackend::Cpu);
    let naive_model = BackendParametrizedLm::new(TapeBackend::Naive);

    let cpu_out = generate(&cpu_model, &prompt, &config).unwrap();
    let naive_out = generate(&naive_model, &prompt, &config).unwrap();

    let cpu_ids: Vec<i32> = cpu_out.contiguous().host_slice().into_owned();
    let naive_ids: Vec<i32> = naive_out.contiguous().host_slice().into_owned();
    assert_eq!(
        cpu_ids, naive_ids,
        "CpuBackendOps と NaiveOps で generate() の token 列が一致しない"
    );

    // token 列一致だけでは logits 自体の乖離を見逃すため（codex-review
    // 指摘・PR #2324）、generate() が辿った token 列を再生して各
    // forward_step の logits を REQ-2 の統一複合判定（`assert_parity`）
    // で突合する。
    let cpu_logits = replay_capturing_logits(&cpu_model, &prompt, &cpu_ids);
    let naive_logits = replay_capturing_logits(&naive_model, &prompt, &cpu_ids);
    assert_eq!(cpu_logits.len(), naive_logits.len());
    for (step, (c, n)) in cpu_logits.iter().zip(naive_logits.iter()).enumerate() {
        assert_parity(
            &format!("generate step {step}: CpuBackendOps vs NaiveOps logits (Greedy)"),
            c,
            n,
        );
    }
}

#[test]
fn cpu_backend_ops_matches_naive_ops_for_top_k_generation_with_same_seed() {
    let prompt = prompt_1d(vec![0, 1, 2]);
    let config = GenerateConfig::new(8, SamplingStrategy::TopK(3)).with_seed(77);

    let cpu_model = BackendParametrizedLm::new(TapeBackend::Cpu);
    let naive_model = BackendParametrizedLm::new(TapeBackend::Naive);

    let cpu_out = generate(&cpu_model, &prompt, &config).unwrap();
    let naive_out = generate(&naive_model, &prompt, &config).unwrap();

    let cpu_ids: Vec<i32> = cpu_out.contiguous().host_slice().into_owned();
    let naive_ids: Vec<i32> = naive_out.contiguous().host_slice().into_owned();
    assert_eq!(
        cpu_ids, naive_ids,
        "CpuBackendOps と NaiveOps で generate()（TopK・同一 seed）の \
         token 列が一致しない"
    );

    // token 列一致だけでは logits 自体の乖離を見逃すため（codex-review
    // 指摘・PR #2324）、generate() が辿った token 列を再生して各
    // forward_step の logits を REQ-2 の統一複合判定（`assert_parity`）
    // で突合する。
    let cpu_logits = replay_capturing_logits(&cpu_model, &prompt, &cpu_ids);
    let naive_logits = replay_capturing_logits(&naive_model, &prompt, &cpu_ids);
    assert_eq!(cpu_logits.len(), naive_logits.len());
    for (step, (c, n)) in cpu_logits.iter().zip(naive_logits.iter()).enumerate() {
        assert_parity(
            &format!("generate step {step}: CpuBackendOps vs NaiveOps logits (TopK seed 77)"),
            c,
            n,
        );
    }
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。CPU と token 列を突合）--------

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_backend_ops_matches_cpu_backend_ops_for_greedy_generation() {
    let prompt = prompt_1d(vec![0, 1, 2]);
    let config = GenerateConfig::new(8, SamplingStrategy::Greedy);
    let max_length = config.max_length;

    let metal_model = BackendParametrizedLm::new(TapeBackend::Metal);
    let cpu_model = BackendParametrizedLm::new(TapeBackend::Cpu);

    let metal_out = generate(&metal_model, &prompt, &config).unwrap();
    let cpu_out = generate(&cpu_model, &prompt, &config).unwrap();

    assert_eq!(
        metal_out.contiguous().host_slice().into_owned(),
        cpu_out.contiguous().host_slice().into_owned(),
        "Metal と CPU で generate() の token 列が一致しない"
    );

    // token 列一致だけでは logits 自体の乖離を見逃すため（codex-review
    // 指摘・PR #2324）、各 forward_step の logits を REQ-2 の統一複合
    // 判定（`assert_parity`）で突合する。
    let (metal_ids, metal_logits) = run_greedy_capturing_logits(&metal_model, &prompt, max_length);
    let (cpu_ids, cpu_logits) = run_greedy_capturing_logits(&cpu_model, &prompt, max_length);
    assert_eq!(
        metal_ids, cpu_ids,
        "Metal と CPU で手動 Greedy ループの token 列が一致しない"
    );
    assert_eq!(metal_logits.len(), cpu_logits.len());
    for (step, (m, c)) in metal_logits.iter().zip(cpu_logits.iter()).enumerate() {
        assert_parity(&format!("generate step {step}: Metal vs CPU logits"), m, c);
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_backend_ops_matches_cpu_backend_ops_for_greedy_generation() {
    let prompt = prompt_1d(vec![0, 1, 2]);
    let config = GenerateConfig::new(8, SamplingStrategy::Greedy);
    let max_length = config.max_length;

    let cuda_model = BackendParametrizedLm::new(TapeBackend::Cuda);
    let cpu_model = BackendParametrizedLm::new(TapeBackend::Cpu);

    let cuda_out = generate(&cuda_model, &prompt, &config).unwrap();
    let cpu_out = generate(&cpu_model, &prompt, &config).unwrap();

    assert_eq!(
        cuda_out.contiguous().host_slice().into_owned(),
        cpu_out.contiguous().host_slice().into_owned(),
        "CUDA と CPU で generate() の token 列が一致しない"
    );

    // token 列一致だけでは logits 自体の乖離を見逃すため（codex-review
    // 指摘・PR #2324）、各 forward_step の logits を REQ-2 の統一複合
    // 判定（`assert_parity`）で突合する。
    let (cuda_ids, cuda_logits) = run_greedy_capturing_logits(&cuda_model, &prompt, max_length);
    let (cpu_ids, cpu_logits) = run_greedy_capturing_logits(&cpu_model, &prompt, max_length);
    assert_eq!(
        cuda_ids, cpu_ids,
        "CUDA と CPU で手動 Greedy ループの token 列が一致しない"
    );
    assert_eq!(cuda_logits.len(), cpu_logits.len());
    for (step, (g, c)) in cuda_logits.iter().zip(cpu_logits.iter()).enumerate() {
        assert_parity(&format!("generate step {step}: CUDA vs CPU logits"), g, c);
    }
}
