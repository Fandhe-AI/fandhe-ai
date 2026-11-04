//! `generate_speculative_stateful`（内部状態を持つモデル版 speculative decoding。
//! イシュー #3848。設計記録 `docs/facade-speculative-decoding-batching-design.md` §29・§30）の
//! 統合テスト。autodiff の公開経路（`fandhe_ai_autodiff::generate::stateful`）だけを使い、
//! `backend-cpu` へは依存しない（`architecture_boundaries.rs` の不変条件）。
//!
//! # 構成
//!
//! - **フィクスチャ A（整数の隠れ状態）**: 状態が処理済みトークン数だけの純 RNN 相当モデル。
//!   同じ規則を `StatefulAutoregressiveModel`（外部 `State`）と、`RefCell` に状態を持つ
//!   `AutoregressiveModel`（`num_kv_layers() == 0`。参照の `generate` 用）で実装し、Greedy の
//!   token 列が完全一致することを確かめる。logits は状態だけで決まる one-hot で形状に依らない
//! - **フィクスチャ B（`State = Vec<KvCache>`）**: KV を持つモデルを新しい trait で表せることと、
//!   既存 `generate_speculative` と **bit 単位で一致**すること（Greedy・同一シードの
//!   TopK／Temperature）を確かめる。後者は RNG 消費順が同じであることの直接検査になる
//! - サンプリングの C1（決定性）・C2（グローバル RNG 非消費）・C4（draft == target で全受理）・
//!   C5（`TopK(1)` は Greedy と一致）。C6（fail-closed）は `stateful.rs` の単体テスト、
//!   C3（受理確率・残差の純関数）は共有する `speculative.rs` の既存単体テストの担当で重複させない
//!
//! 対象外: 分布検定（設計 §20.6。契約外）、実機（CUDA／Metal）parity（新しいカーネル・
//! バックエンド経路が無いため `#[ignore]` で分けるものが無い。設計 §29.4）。

use fandhe_ai_autodiff::generate::speculative::{SpeculativeConfig, generate_speculative};
use fandhe_ai_autodiff::generate::stateful::{
    StatefulAutoregressiveModel, generate_speculative_stateful,
};
use fandhe_ai_autodiff::generate::{
    AutoregressiveModel, GenerateConfig, SamplingStrategy, generate,
};
use fandhe_ai_autodiff::nn::{KvCache, LinearVars, MultiheadAttentionVars};
use fandhe_ai_autodiff::{AutodiffError, Tape, manual_seed, rand};
use fandhe_ai_tensor_core::Tensor;
use std::cell::{Cell, RefCell};
use std::sync::Mutex;

const V: usize = 7;
const E: usize = 4;
const HEADS: usize = 2;

/// グローバル RNG を読み書きするテストの直列化。
static GLOBAL_RNG_LOCK: Mutex<()> = Mutex::new(());

type NextFn = fn(i32, usize) -> i32;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture: shape とデータ長は事前に一致させている")
}

fn seq(seed: i64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((seed + i as i64 * 7) % 13 - 6) as f32 * 0.05)
        .collect()
}

fn target_next(tok: i32, pos: usize) -> i32 {
    ((3 * tok as usize + pos + 1) % V) as i32
}
fn wrong_next(tok: i32, pos: usize) -> i32 {
    (target_next(tok, pos) + 1) % V as i32
}
fn periodic_next(tok: i32, pos: usize) -> i32 {
    if pos.is_multiple_of(3) {
        wrong_next(tok, pos)
    } else {
        target_next(tok, pos)
    }
}

fn prompt(len: usize, rank1: bool) -> Tensor<i32> {
    let v: Vec<i32> = (0..len).map(|i| (i % V) as i32).collect();
    let shape: &[usize] = if rank1 { &[len] } else { &[1, len] };
    Tensor::new(v, shape).expect("fixture")
}

fn vals(x: &Tensor<i32>) -> Vec<i32> {
    x.contiguous().host_slice().into_owned()
}

fn cfg(max_length: usize) -> GenerateConfig {
    GenerateConfig::new(max_length, SamplingStrategy::Greedy)
}

/// `(token, 絶対位置)` の one-hot logits（`[1, L, V]`）。
fn one_hot_logits(toks: &[i32], pos0: usize, next: NextFn) -> Tensor<f32> {
    let mut data = vec![0.0f32; toks.len() * V];
    for (i, &tok) in toks.iter().enumerate() {
        data[i * V + next(tok, pos0 + i) as usize] = 10.0;
    }
    t(data, &[1, toks.len(), V])
}

// ---- フィクスチャ A: 整数の隠れ状態 ----------------------------------------

/// 外部 `State`（処理済みトークン数）を持つ純 RNN 相当モデル。
struct RnnModel {
    next: NextFn,
    restores: Cell<usize>,
}

impl RnnModel {
    fn new(next: NextFn) -> RnnModel {
        RnnModel {
            next,
            restores: Cell::new(0),
        }
    }
}

impl StatefulAutoregressiveModel for RnnModel {
    type State = usize;

    fn init_state(&self) -> Result<usize, AutodiffError> {
        Ok(0)
    }

    fn forward_step_stateful(
        &self,
        new_ids: &Tensor<i32>,
        state: &mut usize,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let toks = vals(new_ids);
        let out = one_hot_logits(&toks, *state, self.next);
        *state += toks.len();
        Ok(out)
    }

    fn save_state(&self, state: &usize) -> Result<usize, AutodiffError> {
        Ok(*state)
    }

    fn restore_state(&self, state: &mut usize, saved: usize) -> Result<(), AutodiffError> {
        self.restores.set(self.restores.get() + 1);
        *state = saved;
        Ok(())
    }
}

/// 同じ規則を `RefCell` に状態を持つ `AutoregressiveModel`（`num_kv_layers() == 0`）で
/// 実装した参照。`generate` 用なので巻き戻しは不要。実行ごとに新しいインスタンスを使う。
struct RefCellRnn {
    next: NextFn,
    pos: RefCell<usize>,
}

impl AutoregressiveModel for RefCellRnn {
    fn num_kv_layers(&self) -> usize {
        0
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        _caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let toks = vals(new_ids);
        let mut pos = self.pos.borrow_mut();
        let out = one_hot_logits(&toks, *pos, self.next);
        *pos += toks.len();
        Ok(out)
    }
}

fn reference(prompt_ids: &Tensor<i32>, max: usize) -> Vec<i32> {
    let m = RefCellRnn {
        next: target_next,
        pos: RefCell::new(0),
    };
    vals(&generate(&m, prompt_ids, &cfg(max)).unwrap())
}

#[test]
fn rnn_greedy_matches_generate() {
    let drafts: [NextFn; 3] = [target_next, wrong_next, periodic_next];
    for draft_next in drafts {
        for rank1 in [true, false] {
            for k in [1usize, 2, 4] {
                for prompt_len in [1usize, 3] {
                    for max in [prompt_len, prompt_len + 1, prompt_len + k + 1, 20] {
                        let ids = prompt(prompt_len, rank1);
                        let target = RnnModel::new(target_next);
                        let draft = RnnModel::new(draft_next);
                        let out = generate_speculative_stateful(
                            &target,
                            &draft,
                            &ids,
                            &cfg(max),
                            &SpeculativeConfig::new(k),
                        )
                        .unwrap();
                        let want_shape: Vec<usize> = if rank1 { vec![max] } else { vec![1, max] };
                        assert_eq!(out.shape(), want_shape.as_slice());
                        assert_eq!(vals(&out), reference(&ids, max), "k={k} max={max}");
                    }
                }
            }
        }
    }
}

#[test]
fn same_instance_as_target_and_draft_works() {
    let m = RnnModel::new(target_next);
    let ids = prompt(2, true);
    let out =
        generate_speculative_stateful(&m, &m, &ids, &cfg(15), &SpeculativeConfig::new(4)).unwrap();
    assert_eq!(vals(&out), reference(&ids, 15));
    assert_eq!(
        m.restores.get(),
        0,
        "draft == target は全受理で restore 0 回"
    );
}

// ---- フィクスチャ B: State = Vec<KvCache> ----------------------------------

/// `TableKvModel`（`nn_generate_speculative.rs`）と同型。KV を実 MHA で進める。
struct TableKvModel {
    layers: usize,
    next: NextFn,
}

impl TableKvModel {
    fn step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let pos0 = caches[0].seq_len();
        let tape = Tape::new();
        let emb = tape.var(&t(seq(1, V * E), &[V, E]));
        let x = emb.embedding(new_ids, None)?;
        for cache in caches.iter_mut() {
            let lin = |s: i64| LinearVars {
                weight: tape.var(&t(seq(s, E * E), &[E, E])),
                bias: Some(tape.var(&t(seq(s + 1, E), &[E]))),
            };
            let mha = MultiheadAttentionVars::new(HEADS, lin(2), lin(4), lin(6), lin(8))?;
            mha.forward_with_cache(&x, &x, &x, cache)?;
        }
        Ok(one_hot_logits(&vals(new_ids), pos0, self.next))
    }
}

impl AutoregressiveModel for TableKvModel {
    fn num_kv_layers(&self) -> usize {
        self.layers
    }
    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        self.step(new_ids, caches)
    }
}

impl StatefulAutoregressiveModel for TableKvModel {
    type State = Vec<KvCache>;

    fn init_state(&self) -> Result<Vec<KvCache>, AutodiffError> {
        Ok((0..self.layers).map(|_| KvCache::new()).collect())
    }

    fn forward_step_stateful(
        &self,
        new_ids: &Tensor<i32>,
        state: &mut Vec<KvCache>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        self.step(new_ids, state)
    }

    // KvCache の clone は Arc 共有で O(1)。
    fn save_state(&self, state: &Vec<KvCache>) -> Result<Vec<KvCache>, AutodiffError> {
        Ok(state.clone())
    }

    fn restore_state(
        &self,
        state: &mut Vec<KvCache>,
        saved: Vec<KvCache>,
    ) -> Result<(), AutodiffError> {
        *state = saved;
        Ok(())
    }
}

fn kv(layers: usize, next: NextFn) -> TableKvModel {
    TableKvModel { layers, next }
}

fn topk_cfg(max_length: usize, k: usize) -> GenerateConfig {
    GenerateConfig::new(max_length, SamplingStrategy::TopK(k)).with_seed(21)
}
fn temp_cfg(max_length: usize) -> GenerateConfig {
    GenerateConfig::new(max_length, SamplingStrategy::Temperature(0.8)).with_seed(21)
}

#[test]
fn kv_state_matches_generate_speculative_bit_for_bit() {
    let drafts: [NextFn; 3] = [target_next, wrong_next, periodic_next];
    let configs = [cfg(14), topk_cfg(14, 3), temp_cfg(14)];
    for config in &configs {
        for draft_next in drafts {
            for layers in [1usize, 2] {
                for k in [1usize, 3] {
                    for rank1 in [true, false] {
                        let ids = prompt(3, rank1);
                        let spec = SpeculativeConfig::new(k);
                        let a = generate_speculative_stateful(
                            &kv(layers, target_next),
                            &kv(layers, draft_next),
                            &ids,
                            config,
                            &spec,
                        )
                        .unwrap();
                        let b = generate_speculative(
                            &kv(layers, target_next),
                            &kv(layers, draft_next),
                            &ids,
                            config,
                            &spec,
                        )
                        .unwrap();
                        assert_eq!(a.shape(), b.shape());
                        assert_eq!(vals(&a), vals(&b), "k={k} layers={layers}");
                    }
                }
            }
        }
    }
}

// ---- サンプリング（C1・C2・C4・C5） -----------------------------------------

#[test]
fn c1_sampling_is_deterministic_for_same_seed() {
    let drafts: [NextFn; 3] = [target_next, wrong_next, periodic_next];
    for config in [temp_cfg(14), topk_cfg(14, 3)] {
        for draft_next in drafts {
            let run = || {
                generate_speculative_stateful(
                    &RnnModel::new(target_next),
                    &RnnModel::new(draft_next),
                    &prompt(3, true),
                    &config,
                    &SpeculativeConfig::new(3),
                )
                .unwrap()
            };
            assert_eq!(vals(&run()), vals(&run()));
        }
    }
}

#[test]
fn c2_sampling_does_not_consume_global_rng() {
    let _guard = GLOBAL_RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for strategy in [
        SamplingStrategy::TopK(3),
        SamplingStrategy::Temperature(0.8),
    ] {
        let config = GenerateConfig::new(12, strategy).with_seed(21);

        manual_seed(1234);
        let baseline = rand(&[4]).unwrap();

        manual_seed(1234);
        let _ = generate_speculative_stateful(
            &RnnModel::new(target_next),
            &RnnModel::new(wrong_next),
            &prompt(3, true),
            &config,
            &SpeculativeConfig::new(3),
        )
        .unwrap();
        let after = rand(&[4]).unwrap();

        assert_eq!(
            baseline.contiguous().host_slice().into_owned(),
            after.contiguous().host_slice().into_owned(),
            "stateful 版 speculative sampling はグローバル RNG を消費してはならない"
        );
    }
}

#[test]
fn c4_identical_draft_accepts_everything() {
    for config in [temp_cfg(12), topk_cfg(12, 3)] {
        let target = RnnModel::new(target_next);
        let draft = RnnModel::new(target_next);
        generate_speculative_stateful(
            &target,
            &draft,
            &prompt(2, true),
            &config,
            &SpeculativeConfig::new(3),
        )
        .unwrap();
        assert_eq!(target.restores.get(), 0);
        assert_eq!(draft.restores.get(), 0);
    }
}

#[test]
fn c5_top_k_one_matches_greedy() {
    let drafts: [NextFn; 3] = [target_next, wrong_next, periodic_next];
    for draft_next in drafts {
        for (prompt_len, max, k) in [(3usize, 12usize, 1usize), (2, 12, 3), (3, 6, 8), (3, 4, 8)] {
            let ids = prompt(prompt_len, true);
            let run = |config: &GenerateConfig| {
                generate_speculative_stateful(
                    &RnnModel::new(target_next),
                    &RnnModel::new(draft_next),
                    &ids,
                    config,
                    &SpeculativeConfig::new(k),
                )
                .unwrap()
            };
            let sampled = run(&topk_cfg(max, 1));
            assert_eq!(vals(&sampled), vals(&run(&cfg(max))));
            assert_eq!(vals(&sampled), reference(&ids, max));
        }
    }
}
