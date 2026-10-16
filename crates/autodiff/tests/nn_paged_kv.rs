//! `nn::PagedKvCacheReference`／`paged_attention_reference`（イシュー #3110・親 #3108。
//! 設計正本 `docs/device-resident-kv-paged-attention-design.md` §8）の受け入れ条件テスト。
//!
//! K-1（`KvCache` + `forward_with_cache`）と REQ-2 統一複合判定
//! （`common::req2_close`。tolerance は不変）で比較する。`gather` は算術を伴わない
//! コピーなのでビット一致を要求する。具体バックエンドには依存しない（`common/mod.rs`）。

mod common;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::{
    KvCache, Linear, MultiheadAttention, PagedKvCacheConfig, PagedKvCacheReference,
    paged_attention_reference,
};
use fandhe_ai_tensor_core::Tensor;

const E: usize = 8;
const H: usize = 2;
const BS: usize = 4;
const SEED: u64 = 7;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn fixture(b: usize, len: usize) -> Tensor<f32> {
    let data = (0..b * len * E).map(|i| (i as f32 * 0.37).sin()).collect();
    t(data, &[b, len, E])
}

fn data_of(x: &Tensor<f32>) -> Vec<f32> {
    x.contiguous().as_slice().unwrap().to_vec()
}

fn assert_close(label: &str, a: &Tensor<f32>, e: &Tensor<f32>) {
    assert_eq!(a.shape(), e.shape(), "{label}: shape");
    for (i, (x, y)) in data_of(a).iter().zip(data_of(e)).enumerate() {
        assert!(
            common::req2_close(*x as f64, y as f64),
            "{label}[{i}]: actual={x} expected={y}"
        );
    }
}

fn new_cache(max_blocks: usize) -> PagedKvCacheReference {
    PagedKvCacheReference::new(PagedKvCacheConfig::new(BS, max_blocks, E).unwrap()).unwrap()
}

/// 公開 API だけで 1 つの Linear 射影（`[B, L, E]` → `[B, L, E]`）を適用する。
fn apply(lin: &Linear, x: &Tensor<f32>) -> Tensor<f32> {
    let (b, l) = (x.shape()[0], x.shape()[1]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let flat = tape.var(x).reshape(&[b * l, E]).unwrap();
    lin.bind(&tape)
        .forward(&flat)
        .unwrap()
        .reshape(&[b, l, E])
        .unwrap()
        .to_tensor()
}

fn project(mha: &MultiheadAttention, x: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    (
        apply(mha.q_proj(), x),
        apply(mha.k_proj(), x),
        apply(mha.v_proj(), x),
    )
}

fn scale() -> f32 {
    1.0 / ((E / H) as f32).sqrt()
}

/// `[B, total, E]` の `start..start+len` を切り出す。
fn chunk(x: &Tensor<f32>, start: usize, len: usize) -> Tensor<f32> {
    let (b, total) = (x.shape()[0], x.shape()[1]);
    let xd = data_of(x);
    let mut d = Vec::new();
    for bi in 0..b {
        let off = (bi * total + start) * E;
        d.extend_from_slice(&xd[off..off + len * E]);
    }
    t(d, &[b, len, E])
}

/// K-1 と paged を同じチャンク列で駆動して各ステップの出力を比較する。
fn run_compare(b: usize, chunks: &[usize], max_blocks: usize) {
    let total: usize = chunks.iter().sum();
    let x = fixture(b, total);
    let mha = MultiheadAttention::new(E, H, true, SEED).unwrap();
    let mut k1 = KvCache::new();
    let mut paged = new_cache(max_blocks);
    let mut start = 0;
    for (n, &len) in chunks.iter().enumerate() {
        let xc = chunk(&x, start, len);
        start += len;

        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&xc);
        let expected = mha
            .bind(&tape)
            .forward_with_cache(&xv, &xv, &xv, &mut k1)
            .unwrap()
            .to_tensor();

        let (q, k, v) = project(&mha, &xc);
        paged.append(&k, &v).unwrap();
        let y = paged_attention_reference(&q, &paged, H, scale()).unwrap();
        let actual = apply(mha.out_proj(), &y);
        assert_close(&format!("step{n}"), &actual, &expected);

        let (gk, gv) = paged.gather().unwrap();
        assert_close("gather_k", &gk, k1.k().unwrap());
        assert_close("gather_v", &gv, k1.v().unwrap());
        assert_eq!(paged.seq_len(), k1.seq_len());
    }
}

#[test]
fn prefill_only_matches_k1() {
    run_compare(1, &[5], 8);
    run_compare(2, &[5], 8);
}

#[test]
fn prefill_then_decode_matches_k1() {
    run_compare(1, &[3, 1, 1, 1, 1, 1], 8);
    run_compare(2, &[3, 1, 1, 1, 1, 1], 16);
}

#[test]
fn multi_token_append_matches_k1() {
    run_compare(1, &[2, 3, 4], 8);
    run_compare(2, &[2, 3, 4], 16);
}

#[test]
fn block_boundaries_and_used_blocks() {
    for (len, blocks) in [(BS - 1, 1), (BS, 1), (BS + 1, 2), (3 * BS + 1, 4)] {
        let x = fixture(1, len);
        let mut c = new_cache(8);
        c.append(&x, &x).unwrap();
        assert_eq!(c.num_used_blocks(), blocks, "len={len}");
        assert_eq!(c.num_free_blocks(), 8 - blocks);
        let (gk, gv) = c.gather().unwrap();
        assert_eq!(data_of(&gk), data_of(&x), "len={len} K bit");
        assert_eq!(data_of(&gv), data_of(&x), "len={len} V bit");
        run_compare(1, &[len], 8);
    }
}

#[test]
fn empty_cache_behaviour() {
    let c = new_cache(2);
    assert_eq!(c.seq_len(), 0);
    assert_eq!(c.batch(), None);
    assert!(c.gather().is_err());
    let q0 = t(vec![], &[1, 0, E]);
    let out = paged_attention_reference(&q0, &c, H, scale()).unwrap();
    assert_eq!(out.shape(), &[1, 0, E]);
    let q1 = fixture(1, 1);
    assert!(paged_attention_reference(&q1, &c, H, scale()).is_err());
}

#[derive(Debug, PartialEq)]
struct Snap {
    seq: usize,
    batch: Option<usize>,
    free: usize,
    tables: Vec<Option<Vec<u32>>>,
    k: Option<Vec<f32>>,
}

fn snap(c: &PagedKvCacheReference) -> Snap {
    Snap {
        seq: c.seq_len(),
        batch: c.batch(),
        free: c.num_free_blocks(),
        tables: (0..3)
            .map(|r| c.block_table(r).map(<[u32]>::to_vec))
            .collect(),
        k: c.gather().ok().map(|(k, _)| data_of(&k)),
    }
}

#[test]
fn errors_leave_state_unchanged() {
    let mut c = new_cache(3);
    let x = fixture(1, 5);
    c.append(&x, &x).unwrap(); // 2 ブロック使用
    let before = snap(&c);

    // max_blocks 不足（5 + 9 = 14 トークンは 4 ブロック必要で max_blocks=3 を超える）。
    let big = fixture(1, 9);
    assert!(c.append(&big, &big).is_err());
    assert_eq!(before, snap(&c));
    // rank 不正・shape 不一致・B／E 不一致。
    let r2 = t(vec![0.0; E], &[1, E]);
    assert!(c.append(&r2, &r2).is_err());
    let a = fixture(1, 1);
    let b2 = fixture(1, 2);
    assert!(c.append(&a, &b2).is_err());
    let bb = fixture(2, 1);
    assert!(c.append(&bb, &bb).is_err());
    let ee = t(vec![0.0; 4], &[1, 1, 4]);
    assert!(c.append(&ee, &ee).is_err());
    assert_eq!(before, snap(&c));
    // rewind 超過。
    assert!(c.rewind_to(6).is_err());
    assert_eq!(before, snap(&c));
    // attention 側。
    let q = fixture(1, 6);
    assert!(paged_attention_reference(&q, &c, H, scale()).is_err()); // L_new > seq_len
    let q = fixture(1, 1);
    assert!(paged_attention_reference(&q, &c, 3, scale()).is_err());
    assert!(paged_attention_reference(&q, &c, 0, scale()).is_err());
    assert!(paged_attention_reference(&q, &c, H, f32::NAN).is_err());
    assert!(paged_attention_reference(&q, &c, H, f32::INFINITY).is_err());
    assert_eq!(before, snap(&c));
}

#[test]
fn invalid_config_is_rejected() {
    for (bs, mb, e) in [
        (0, 4, E),
        (3, 4, E),
        (512, 4, E),
        (4, 0, E),
        (4, u32::MAX as usize + 1, E),
        (4, 4, 0),
        (256, u32::MAX as usize, usize::MAX / 8),
    ] {
        assert!(PagedKvCacheConfig::new(bs, mb, e).is_err(), "{bs} {mb} {e}");
    }
}

#[test]
fn rewind_frees_blocks_and_reappend_matches_k1() {
    let mha = MultiheadAttention::new(E, H, true, SEED).unwrap();
    let x = fixture(1, 10);
    let mut c = new_cache(8);
    let (_, k, v) = project(&mha, &x);
    c.append(&k, &v).unwrap();
    assert_eq!(c.num_used_blocks(), 3);
    c.rewind_to(5).unwrap();
    assert_eq!(
        (c.seq_len(), c.num_used_blocks(), c.num_free_blocks()),
        (5, 2, 6)
    );
    c.rewind_to(4).unwrap();
    assert_eq!((c.num_used_blocks(), c.num_free_blocks()), (1, 7));

    // 巻き戻した位置から別トークンを追記し、同じ系列の K-1 と比較する。
    let y2 = fixture(1, 3);
    let mut seq = data_of(&x)[..4 * E].to_vec();
    seq.extend(data_of(&y2));
    let full = t(seq, &[1, 7, E]);
    let (q, k2, v2) = project(&mha, &y2);
    c.append(&k2, &v2).unwrap();
    let y = paged_attention_reference(&q, &c, H, scale()).unwrap();
    let actual = apply(mha.out_proj(), &y);

    let mut k1 = KvCache::new();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&full);
    let pre = xv.narrow(1, 0, 4).unwrap();
    let mha_b = mha.bind(&tape);
    mha_b.forward_with_cache(&pre, &pre, &pre, &mut k1).unwrap();
    let tail = xv.narrow(1, 4, 3).unwrap();
    let expected = mha_b
        .forward_with_cache(&tail, &tail, &tail, &mut k1)
        .unwrap()
        .to_tensor();
    assert_close("rewind_reappend", &actual, &expected);
    let (gk, gv) = c.gather().unwrap();
    assert_close("rewind gather k", &gk, k1.k().unwrap());
    assert_close("rewind gather v", &gv, k1.v().unwrap());
}

#[test]
fn rewind_to_zero_allows_new_batch() {
    let mut c = new_cache(8);
    let x = fixture(1, 6);
    c.append(&x, &x).unwrap();
    c.rewind_to(0).unwrap();
    assert_eq!(c.batch(), None);
    assert_eq!(c.num_free_blocks(), 8);
    let y = fixture(2, 3);
    c.append(&y, &y).unwrap();
    assert_eq!((c.batch(), c.seq_len()), (Some(2), 3));
}
