//! paged attention（K-3）の facade 公開経路（`Tape::new_paged_kv_cache`・
//! `Tape::paged_attention_forward`。イシュー #3112・親 #3108）の統合テスト。
//!
//! 設計正本は `docs/device-resident-kv-paged-attention-design.md` §6・§13。facade の公開面
//! だけを使い、K-1（`stateful_attention_forward_with_cache`）との出力一致を REQ-2 統一複合判定
//! （`assert_parity`。tolerance 定数は不変）で固定する。あわせて原子性・容量超過・
//! `L_new == 0` を固定する。すべて CPU で動く（実機 parity は #3113・#3171・#3172）。

use fandhe_ai::Tensor;
use fandhe_ai::nn::kv_cache::{
    KvCache, MultiheadAttentionConfig, PagedKvCacheConfig, StatefulAttention,
};
use fandhe_ai_backend_cpu::parity::assert_parity;

const E: usize = 4;
const H: usize = 2;
const SEED: u64 = 11;

fn sa() -> StatefulAttention {
    StatefulAttention::from_config(&MultiheadAttentionConfig::new(E, H), SEED).expect("from_config")
}

fn input(b: usize, total: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..b * total * E)
        .map(|i| ((211 + i as i64) % 17 - 8) as f32 * 0.03)
        .collect();
    Tensor::new(data, &[b, total, E]).expect("fixture")
}

fn data_of(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().expect("contiguous").to_vec()
}

fn slice_time(x: &Tensor<f32>, from: usize, to: usize) -> Tensor<f32> {
    let b = x.shape()[0];
    let total = x.shape()[1];
    let d = data_of(x);
    let mut out = Vec::new();
    for bi in 0..b {
        out.extend_from_slice(&d[(bi * total + from) * E..(bi * total + to) * E]);
    }
    Tensor::new(out, &[b, to - from, E]).expect("slice")
}

/// K-1 の外部キャッシュ版と paged の各ステップ出力が統一複合判定で一致する。
fn compare(b: usize, steps: &[usize], block_size: usize, max_blocks: usize) {
    let total: usize = steps.iter().sum();
    let x = input(b, total);
    let tape = fandhe_ai::tape();
    let sa = sa();
    let mut k1 = KvCache::new();
    let mut paged = tape
        .new_paged_kv_cache(PagedKvCacheConfig::new(block_size, max_blocks, E).expect("cfg"))
        .expect("new");
    let mut at = 0;
    for (i, &len) in steps.iter().enumerate() {
        let xs = slice_time(&x, at, at + len);
        let want = tape
            .stateful_attention_forward_with_cache(&sa, &tape.var(&xs), &mut k1)
            .expect("k1")
            .to_tensor();
        let got = tape
            .paged_attention_forward(&sa, &xs, &mut paged)
            .expect("paged");
        assert_eq!(got.shape(), want.shape());
        assert_parity(
            &format!("paged vs K-1 (b={b}, bs={block_size}, step {i})"),
            &data_of(&got),
            &data_of(&want),
        );
        at += len;
        assert_eq!(paged.seq_len(), at);
        assert_eq!(sa.seq_len(), 0, "内部キャッシュは不変");
    }
}

#[test]
fn prefill_decode_and_multi_token_append_match_k1() {
    compare(1, &[3, 1, 2], 2, 8);
}

#[test]
fn multi_block_and_batch_match_k1() {
    compare(2, &[5, 1, 1, 3], 2, 16);
}

#[test]
fn block_size_boundaries_match_k1() {
    // block_size = 4: 3（境界 -1）→ +1（ちょうど）→ +1（境界 +1）
    compare(1, &[3, 1, 1], 4, 4);
    compare(2, &[4, 1], 4, 4);
}

#[test]
fn invalid_input_leaves_cache_unchanged() {
    let tape = fandhe_ai::tape();
    let sa = sa();
    let mut cache = tape
        .new_paged_kv_cache(PagedKvCacheConfig::new(2, 8, E).unwrap())
        .unwrap();
    tape.paged_attention_forward(&sa, &input(1, 3), &mut cache)
        .unwrap();
    let used = cache.num_used_blocks();
    let bad_rank = Tensor::new(vec![0.0f32; E], &[1, E]).unwrap();
    assert!(
        tape.paged_attention_forward(&sa, &bad_rank, &mut cache)
            .is_err()
    );
    let bad_e = Tensor::new(vec![0.0f32; 6], &[1, 1, 6]).unwrap();
    assert!(
        tape.paged_attention_forward(&sa, &bad_e, &mut cache)
            .is_err()
    );
    assert!(
        tape.paged_attention_forward(&sa, &input(2, 1), &mut cache)
            .is_err()
    );
    assert_eq!(cache.seq_len(), 3);
    assert_eq!(cache.num_used_blocks(), used);
    assert!(!cache.is_poisoned());
    tape.paged_attention_forward(&sa, &input(1, 1), &mut cache)
        .unwrap();
    assert_eq!(cache.seq_len(), 4);
}

#[test]
fn exceeding_max_blocks_errs_and_keeps_state() {
    let tape = fandhe_ai::tape();
    let sa = sa();
    let mut cache = tape
        .new_paged_kv_cache(PagedKvCacheConfig::new(2, 2, E).unwrap())
        .unwrap();
    tape.paged_attention_forward(&sa, &input(1, 4), &mut cache)
        .unwrap();
    assert_eq!(cache.num_free_blocks(), 0);
    assert!(
        tape.paged_attention_forward(&sa, &input(1, 1), &mut cache)
            .is_err()
    );
    assert_eq!(cache.seq_len(), 4);
    assert!(!cache.is_poisoned());
}

#[test]
fn zero_length_step_is_a_noop() {
    let tape = fandhe_ai::tape();
    let sa = sa();
    let mut cache = tape
        .new_paged_kv_cache(PagedKvCacheConfig::new(2, 4, E).unwrap())
        .unwrap();
    tape.paged_attention_forward(&sa, &input(1, 2), &mut cache)
        .unwrap();
    let empty = Tensor::new(Vec::<f32>::new(), &[1, 0, E]).unwrap();
    let y = tape
        .paged_attention_forward(&sa, &empty, &mut cache)
        .unwrap();
    assert_eq!(y.shape(), &[1usize, 0, E]);
    assert_eq!(cache.seq_len(), 2);
}
