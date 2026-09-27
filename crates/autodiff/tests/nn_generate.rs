//! `fandhe_ai_autodiff::generate`（イシュー #2191）の統合テスト。
//!
//! [`TinyCausalLm`] は `Embedding → MultiheadAttentionVars（KV キャッシュ
//! 付き） → Linear（lm head）` の最小構成で [`AutoregressiveModel`] を
//! 実装するテスト専用モデル（`crates/facade/tests/kv_cache_backend_parity.rs`
//! の `build_vars` と同じ「`LinearVars` を直接組み立てる」パターン）。
//! 各 `forward_step` 呼び出しごとに新規 `Tape::new()`（`docs/
//! kv-cache-design.md` §3.2 が推奨する運用）を作る。

use fandhe_ai_autodiff::generate::{
    AutoregressiveModel, GenerateConfig, SamplingStrategy, generate,
};
use fandhe_ai_autodiff::nn::{KvCache, LinearVars, MultiheadAttentionVars};
use fandhe_ai_autodiff::{AutodiffError, Tape, manual_seed, rand};
use fandhe_ai_tensor_core::Tensor;
use std::cell::RefCell;

const V: usize = 4;
const E: usize = 4;
const NUM_HEADS: usize = 2;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture: shape とデータ長は事前に一致させている")
}

/// 決定的な擬似乱数列（テストフィクスチャ専用。`rng.rs` の xorshift64*
/// とは無関係の単純な合成式で十分——本番経路には使わない）。
fn seq(seed: i64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((seed + i as i64 * 7) % 13 - 6) as f32 * 0.05)
        .collect()
}

/// `Embedding → MultiheadAttentionVars（KV キャッシュ付き） → Linear`
/// の最小 `AutoregressiveModel`。`dominant_token`（`Some` の場合）を
/// 指定すると、lm head の bias に極端な値を入れて常にその token が
/// argmax になるようにし、「手計算可能な小モデル」（実装計画 §5.1
/// 受入条件 4）として使える。
struct TinyCausalLm {
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
}

impl TinyCausalLm {
    fn new(dominant_token: Option<usize>) -> TinyCausalLm {
        let lm_bias = match dominant_token {
            Some(tok) => {
                let mut b = vec![0.0f32; V];
                b[tok] = 100.0;
                b
            }
            None => seq(9, V),
        };
        let lm_weight = match dominant_token {
            // bias だけで argmax が決まるよう weight を 0 にし、attention
            // 出力（入力依存の値）の影響を完全に消す。
            Some(_) => vec![0.0f32; E * V],
            None => seq(10, E * V),
        };
        TinyCausalLm {
            embedding: t(seq(1, V * E), &[V, E]),
            q_weight: t(seq(2, E * E), &[E, E]),
            q_bias: t(seq(3, E), &[E]),
            k_weight: t(seq(4, E * E), &[E, E]),
            k_bias: t(seq(5, E), &[E]),
            v_weight: t(seq(6, E * E), &[E, E]),
            v_bias: t(seq(7, E), &[E]),
            out_weight: t(seq(8, E * E), &[E, E]),
            out_bias: t(seq(9, E), &[E]),
            lm_weight: t(lm_weight, &[E, V]),
            lm_bias: t(lm_bias, &[V]),
        }
    }
}

impl AutoregressiveModel for TinyCausalLm {
    fn num_kv_layers(&self) -> usize {
        1
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        assert_eq!(caches.len(), 1, "TinyCausalLm は 1 attention 層のみ持つ");
        let tape = Tape::new();
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

        // `LinearVars::forward` は rank 2 入力限定（イシュー #1715
        // スコープ外）のため、lm head 通過前に `[B, L, E]` → `[B*L, E]`
        // へ平坦化し、通過後に `[B, L, V]` へ戻す。`Var::shape` は
        // `pub(crate)` のため（テストは別クレート扱い）`new_ids` の
        // 既知 shape から `b`／`l` を導出する。
        let (b, l) = (new_ids.shape()[0], new_ids.shape()[1]);
        let e = E;
        let flat = attn_out.reshape(&[b * l, e])?;
        let lm = LinearVars {
            weight: tape.var(&self.lm_weight),
            bias: Some(tape.var(&self.lm_bias)),
        };
        let logits_flat = lm.forward(&flat)?;
        let vocab = self.lm_bias.shape()[0];
        let logits = logits_flat.reshape(&[b, l, vocab])?;
        Ok(logits.to_tensor())
    }
}

/// [`TinyCausalLm::forward_step`] への各呼び出しの `new_ids` shape・
/// 呼び出し前後の `caches[0].seq_len()` を記録するラッパー（受入条件 3
/// 「KV キャッシュ結線」の構造検証用）。
struct RecordingModel {
    inner: TinyCausalLm,
    l_new_per_call: RefCell<Vec<usize>>,
    seq_len_before: RefCell<Vec<usize>>,
    seq_len_after: RefCell<Vec<usize>>,
    logits_per_call: RefCell<Vec<Tensor<f32>>>,
}

impl RecordingModel {
    fn new(inner: TinyCausalLm) -> RecordingModel {
        RecordingModel {
            inner,
            l_new_per_call: RefCell::new(Vec::new()),
            seq_len_before: RefCell::new(Vec::new()),
            seq_len_after: RefCell::new(Vec::new()),
            logits_per_call: RefCell::new(Vec::new()),
        }
    }
}

impl AutoregressiveModel for RecordingModel {
    fn num_kv_layers(&self) -> usize {
        self.inner.num_kv_layers()
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        self.l_new_per_call.borrow_mut().push(new_ids.shape()[1]);
        self.seq_len_before.borrow_mut().push(caches[0].seq_len());
        let logits = self.inner.forward_step(new_ids, caches)?;
        self.seq_len_after.borrow_mut().push(caches[0].seq_len());
        self.logits_per_call.borrow_mut().push(logits.clone());
        Ok(logits)
    }
}

/// `forward_step` が呼ばれるたびに vocab（末尾軸）を変えて返す不正モデル
/// （エラー系「モデル戻り shape 不一致」の検証用）。
struct DriftingVocabModel;

impl AutoregressiveModel for DriftingVocabModel {
    fn num_kv_layers(&self) -> usize {
        0
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        assert!(caches.is_empty());
        let b = new_ids.shape()[0];
        let l = new_ids.shape()[1];
        // 呼ばれるたびに vocab を変える（最初は V、以降は V+1）。
        static CALL_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let vocab = if n == 0 { V } else { V + 1 };
        Tensor::new(vec![0.0f32; b * l * vocab], &[b, l, vocab]).map_err(AutodiffError::Shape)
    }
}

/// 常に非有限 logits を返す不正モデル（エラー系「非有限 logits」検証用）。
struct NonFiniteLogitsModel;

impl AutoregressiveModel for NonFiniteLogitsModel {
    fn num_kv_layers(&self) -> usize {
        0
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        assert!(caches.is_empty());
        let b = new_ids.shape()[0];
        let l = new_ids.shape()[1];
        Tensor::new(vec![f32::NAN; b * l * V], &[b, l, V]).map_err(AutodiffError::Shape)
    }
}

fn prompt_2d(data: Vec<i32>, b: usize, t: usize) -> Tensor<i32> {
    Tensor::new(data, &[b, t]).unwrap()
}

// --- 受入条件 1: GenerateConfig の end-to-end 検証拒否 ------------------

#[test]
fn generate_rejects_max_length_less_than_prompt_len() {
    let model = TinyCausalLm::new(Some(2));
    let prompt = prompt_2d(vec![0, 1, 2], 1, 3);
    let config = GenerateConfig::new(2, SamplingStrategy::Greedy);
    assert!(generate(&model, &prompt, &config).is_err());
}

#[test]
fn generate_rejects_empty_prompt() {
    let model = TinyCausalLm::new(Some(2));
    let prompt = Tensor::new(Vec::<i32>::new(), &[1, 0]).unwrap();
    let config = GenerateConfig::new(4, SamplingStrategy::Greedy);
    assert!(generate(&model, &prompt, &config).is_err());
}

#[test]
fn generate_rejects_empty_batch() {
    let model = TinyCausalLm::new(Some(2));
    let prompt = Tensor::new(Vec::<i32>::new(), &[0, 3]).unwrap();
    let config = GenerateConfig::new(4, SamplingStrategy::Greedy);
    assert!(generate(&model, &prompt, &config).is_err());
}

#[test]
fn generate_rejects_rank0_and_rank3_input() {
    let model = TinyCausalLm::new(Some(2));
    let config = GenerateConfig::new(4, SamplingStrategy::Greedy);

    let rank0 = Tensor::new(vec![1i32], &[]).unwrap();
    assert!(generate(&model, &rank0, &config).is_err());

    let rank3 = Tensor::new(vec![1i32; 8], &[1, 2, 4]).unwrap();
    assert!(generate(&model, &rank3, &config).is_err());
}

#[test]
fn generate_rejects_top_k_zero_or_greater_than_vocab() {
    let model = TinyCausalLm::new(Some(2));
    let prompt = prompt_2d(vec![0, 1], 1, 2);

    let k_zero = GenerateConfig::new(4, SamplingStrategy::TopK(0));
    assert!(generate(&model, &prompt, &k_zero).is_err());

    let k_too_large = GenerateConfig::new(4, SamplingStrategy::TopK(V + 1));
    assert!(generate(&model, &prompt, &k_too_large).is_err());
}

#[test]
fn generate_rejects_non_finite_or_non_positive_temperature() {
    let model = TinyCausalLm::new(Some(2));
    let prompt = prompt_2d(vec![0, 1], 1, 2);

    for bad in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
        let config = GenerateConfig::new(4, SamplingStrategy::Temperature(bad));
        assert!(generate(&model, &prompt, &config).is_err());
    }
}

#[test]
fn generate_propagates_model_shape_drift_error() {
    let model = DriftingVocabModel;
    let prompt = prompt_2d(vec![0, 1], 1, 2);
    let config = GenerateConfig::new(4, SamplingStrategy::Greedy);
    assert!(generate(&model, &prompt, &config).is_err());
}

#[test]
fn generate_propagates_non_finite_logits_error() {
    let model = NonFiniteLogitsModel;
    let prompt = prompt_2d(vec![0, 1], 1, 2);
    let config = GenerateConfig::new(4, SamplingStrategy::Greedy);
    assert!(generate(&model, &prompt, &config).is_err());
}

// --- 受入条件 4: loop cycle（prompt → logits → token id の往復） ------

#[test]
fn generate_max_length_equal_to_prompt_len_returns_prompt_unchanged_rank1() {
    let model = TinyCausalLm::new(Some(2));
    let prompt = Tensor::new(vec![0i32, 1, 3], &[3]).unwrap();
    let config = GenerateConfig::new(3, SamplingStrategy::Greedy);
    let out = generate(&model, &prompt, &config).unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.contiguous().host_slice().into_owned(), vec![0, 1, 3]);
}

#[test]
fn generate_greedy_with_dominant_token_always_appends_that_token_rank1() {
    let model = TinyCausalLm::new(Some(2));
    let prompt = Tensor::new(vec![0i32, 1], &[2]).unwrap();
    let config = GenerateConfig::new(6, SamplingStrategy::Greedy);
    let out = generate(&model, &prompt, &config).unwrap();
    assert_eq!(out.shape(), &[6]);
    let data = out.contiguous().host_slice().into_owned();
    assert_eq!(&data[..2], &[0, 1]);
    assert!(data[2..].iter().all(|&id| id == 2));
}

#[test]
fn generate_greedy_with_dominant_token_rank2_batch() {
    let model = TinyCausalLm::new(Some(1));
    let prompt = prompt_2d(vec![0, 1, /* row 1 */ 2, 3], 2, 2);
    let config = GenerateConfig::new(5, SamplingStrategy::Greedy);
    let out = generate(&model, &prompt, &config).unwrap();
    assert_eq!(out.shape(), &[2, 5]);
    let data = out.contiguous().host_slice().into_owned();
    assert_eq!(&data[0..2], &[0, 1]);
    assert!(data[2..5].iter().all(|&id| id == 1));
    assert_eq!(&data[5..7], &[2, 3]);
    assert!(data[7..10].iter().all(|&id| id == 1));
}

#[test]
fn generate_output_token_ids_are_within_vocab_range() {
    let model = TinyCausalLm::new(None);
    let prompt = Tensor::new(vec![0i32, 1, 2], &[3]).unwrap();
    for strategy in [
        SamplingStrategy::Greedy,
        SamplingStrategy::TopK(2),
        SamplingStrategy::Temperature(0.8),
    ] {
        let config = GenerateConfig::new(8, strategy).with_seed(7);
        let out = generate(&model, &prompt, &config).unwrap();
        assert_eq!(out.shape(), &[8]);
        for &id in out.contiguous().host_slice().iter() {
            assert!((0..V as i32).contains(&id), "id={id} out of range");
        }
    }
    // TopK(1) は Greedy と一致する（実装計画 §5.1 受入条件 2）。
    let greedy_out = generate(
        &model,
        &prompt,
        &GenerateConfig::new(8, SamplingStrategy::Greedy),
    )
    .unwrap();
    let topk1_out = generate(
        &model,
        &prompt,
        &GenerateConfig::new(8, SamplingStrategy::TopK(1)),
    )
    .unwrap();
    assert_eq!(
        greedy_out.contiguous().host_slice().into_owned(),
        topk1_out.contiguous().host_slice().into_owned()
    );
}

// --- 受入条件 2: TopK の生成 token が毎ステップの top-k 集合に含まれる ---

#[test]
fn generate_top_k_selected_tokens_are_within_recorded_top_k_set() {
    let k = 2;
    let model = RecordingModel::new(TinyCausalLm::new(None));
    let prompt = prompt_2d(vec![0, 1], 1, 2);
    let config = GenerateConfig::new(6, SamplingStrategy::TopK(k)).with_seed(123);
    let out = generate(&model, &prompt, &config).unwrap();
    let generated = out.contiguous().host_slice().into_owned();

    let logits_per_call = model.logits_per_call.borrow();
    // generated[2..] が新規生成分（各呼び出し 1 個ずつ対応）。
    for (call_idx, logits) in logits_per_call.iter().enumerate() {
        let shape = logits.shape();
        let l_new = shape[1];
        let vocab = shape[2];
        let data = logits.contiguous().host_slice().into_owned();
        let last_row = &data[(l_new - 1) * vocab..l_new * vocab];
        let mut idxs: Vec<usize> = (0..vocab).collect();
        idxs.sort_by(|&a, &b| last_row[b].partial_cmp(&last_row[a]).unwrap());
        let top_set: Vec<usize> = idxs[..k].to_vec();
        let generated_id = generated[2 + call_idx] as usize;
        assert!(
            top_set.contains(&generated_id),
            "call {call_idx}: generated id {generated_id} not in top-{k} set {top_set:?}"
        );
    }
}

// --- 受入条件 3: KV キャッシュ結線の構造検証 ---------------------------

#[test]
fn generate_calls_forward_step_with_prefill_then_single_token_decode_shape() {
    let model = RecordingModel::new(TinyCausalLm::new(Some(2)));
    let prompt = prompt_2d(vec![0, 1, 2], 1, 3);
    let config = GenerateConfig::new(6, SamplingStrategy::Greedy);
    generate(&model, &prompt, &config).unwrap();

    let l_new = model.l_new_per_call.borrow();
    assert_eq!(l_new[0], 3, "prefill は new_ids に prompt 全体を渡す");
    for &l in l_new.iter().skip(1) {
        assert_eq!(l, 1, "decode は new_ids に新規 1 トークンのみを渡す");
    }
    assert_eq!(
        l_new.len(),
        3,
        "prompt 長 3・max_length 6 なら prefill 1 回（+2 トークン）+ decode 2 回で到達する"
    );

    let seq_before = model.seq_len_before.borrow();
    let seq_after = model.seq_len_after.borrow();
    assert_eq!(seq_before[0], 0, "prefill 前の cache は空");
    for i in 0..seq_before.len() {
        assert_eq!(
            seq_after[i],
            seq_before[i] + l_new[i],
            "cache の seq_len は各呼び出しで l_new だけ増える"
        );
    }
    for i in 1..seq_before.len() {
        assert_eq!(
            seq_before[i],
            seq_after[i - 1],
            "次の呼び出し前の seq_len は直前の呼び出し後と一致する"
        );
    }
}

/// KV キャッシュ経由の decode（1 トークンずつ）で得た greedy token 列が、
/// 全系列再計算（`MultiheadAttentionVars::forward`・causal）を毎ステップ
/// 繰り返して得た token 列と一致することを確認する（受入条件 3
/// 「等価性」）。
#[test]
fn generate_greedy_matches_full_recompute_without_cache() {
    let model = TinyCausalLm::new(None);
    let prompt_ids = vec![0i32, 1, 2];
    let config = GenerateConfig::new(6, SamplingStrategy::Greedy);
    let prompt = Tensor::new(prompt_ids.clone(), &[1, 3]).unwrap();
    let out = generate(&model, &prompt, &config).unwrap();
    let generated = out.contiguous().host_slice().into_owned();

    // 全系列再計算で同じ token 列を独立に再現する。
    let mut all_ids = prompt_ids.clone();
    while all_ids.len() < 6 {
        let tape = Tape::new();
        let embed_weight = tape.var(&model.embedding);
        let ids = Tensor::new(all_ids.clone(), &[1, all_ids.len()]).unwrap();
        let x = embed_weight.embedding(&ids, None).unwrap();
        let q = LinearVars {
            weight: tape.var(&model.q_weight),
            bias: Some(tape.var(&model.q_bias)),
        };
        let k = LinearVars {
            weight: tape.var(&model.k_weight),
            bias: Some(tape.var(&model.k_bias)),
        };
        let v = LinearVars {
            weight: tape.var(&model.v_weight),
            bias: Some(tape.var(&model.v_bias)),
        };
        let out_proj = LinearVars {
            weight: tape.var(&model.out_weight),
            bias: Some(tape.var(&model.out_bias)),
        };
        let mha = MultiheadAttentionVars::new(NUM_HEADS, q, k, v, out_proj).unwrap();
        let attn_out = mha.forward(&x, &x, &x, None, true).unwrap();
        let (b, l_full, e) = (1usize, all_ids.len(), E);
        let flat = attn_out.reshape(&[b * l_full, e]).unwrap();
        let lm = LinearVars {
            weight: tape.var(&model.lm_weight),
            bias: Some(tape.var(&model.lm_bias)),
        };
        let vocab_size = model.lm_bias.shape()[0];
        let logits = lm
            .forward(&flat)
            .unwrap()
            .reshape(&[b, l_full, vocab_size])
            .unwrap()
            .to_tensor();
        let shape = logits.shape();
        let (l, vocab) = (shape[1], shape[2]);
        let data = logits.contiguous().host_slice().into_owned();
        let last_row = &data[(l - 1) * vocab..l * vocab];
        let mut best_idx = 0usize;
        let mut best_val = last_row[0];
        for (idx, &val) in last_row.iter().enumerate().skip(1) {
            if val > best_val {
                best_val = val;
                best_idx = idx;
            }
        }
        all_ids.push(best_idx as i32);
    }

    assert_eq!(generated, all_ids);
}

// --- 受入条件 5: 再現性・グローバル RNG 非消費 --------------------------

#[test]
fn generate_same_seed_produces_bit_identical_output_for_top_k_and_temperature() {
    let model = TinyCausalLm::new(None);
    let prompt = prompt_2d(vec![0, 1], 1, 2);

    for strategy in [
        SamplingStrategy::TopK(3),
        SamplingStrategy::Temperature(0.9),
    ] {
        let config_a = GenerateConfig::new(8, strategy).with_seed(999);
        let config_b = GenerateConfig::new(8, strategy).with_seed(999);
        let out_a = generate(&model, &prompt, &config_a).unwrap();
        let out_b = generate(&model, &prompt, &config_b).unwrap();
        assert_eq!(
            out_a.contiguous().host_slice().into_owned(),
            out_b.contiguous().host_slice().into_owned()
        );
    }
}

#[test]
fn generate_greedy_is_seed_independent() {
    let model = TinyCausalLm::new(Some(2));
    let prompt = prompt_2d(vec![0, 1], 1, 2);
    let out_a = generate(
        &model,
        &prompt,
        &GenerateConfig::new(6, SamplingStrategy::Greedy).with_seed(1),
    )
    .unwrap();
    let out_b = generate(
        &model,
        &prompt,
        &GenerateConfig::new(6, SamplingStrategy::Greedy).with_seed(2),
    )
    .unwrap();
    assert_eq!(
        out_a.contiguous().host_slice().into_owned(),
        out_b.contiguous().host_slice().into_owned()
    );
}

#[test]
fn generate_does_not_consume_global_rng() {
    let model = TinyCausalLm::new(None);
    let prompt = prompt_2d(vec![0, 1], 1, 2);
    let config = GenerateConfig::new(8, SamplingStrategy::TopK(2)).with_seed(55);

    manual_seed(1234);
    let baseline = rand(&[4]).unwrap();

    manual_seed(1234);
    let _ = generate(&model, &prompt, &config).unwrap();
    let after_generate = rand(&[4]).unwrap();

    assert_eq!(
        baseline.contiguous().host_slice().into_owned(),
        after_generate.contiguous().host_slice().into_owned(),
        "generate はグローバル RNG（manual_seed が触れる状態）を消費してはならない"
    );
}
