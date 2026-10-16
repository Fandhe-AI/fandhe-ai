//! `CpuBackendOps` のデバイス常駐 KV（K-3。イシュー #3111）と、#3110 の CPU 参照
//! （`PagedKvCacheReference`・`paged_attention_reference`）の端から端までの一致テスト。
//!
//! 役割: CI（GPU なし）で動く唯一の「`PagedKvCache`（デバイス版 owner）→
//! `BackendOps::paged_kv_append_tracked`／`paged_attention_forward_tracked` →
//! `MemoryOps::download`」の通し検証。CUDA・Metal の実機 parity（`#[ignore]`。#3113 で網羅）が
//! 比べる期待値の妥当性も、ここで先に固める。判定は REQ-2 の統一複合判定
//! （`assert_parity`。tolerance 定数は変更しない）。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::{
    PagedKvCache, PagedKvCacheConfig, PagedKvCacheReference, paged_attention_reference,
};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::assert_parity;
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, Device, DeviceBuffer, PagedAttentionArgs, PagedKvLayout, Tensor,
};

/// 決定的な擬似乱数（xorshift）。値域は約 [-1, 1)。
fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn tensor(b: usize, l: usize, e: usize, seed: u64) -> Tensor<f32> {
    Tensor::new(data(b * l * e, seed), &[b, l, e]).unwrap()
}

fn flat(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().unwrap().to_vec()
}

struct Case {
    block_size: usize,
    batch: usize,
    embed: usize,
    heads: usize,
    /// 各ステップの追記トークン数（先頭が prefill）。
    steps: &'static [usize],
}

fn run_case(c: &Case, rewind_after: Option<(usize, usize)>) {
    let max_blocks = 64;
    let cfg = PagedKvCacheConfig::new(c.block_size, max_blocks, c.embed).unwrap();
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let mut dev = PagedKvCache::new(&tape, cfg).unwrap();
    let mut reference = PagedKvCacheReference::new(cfg).unwrap();
    let scale = 1.0 / ((c.embed / c.heads) as f32).sqrt();
    for (si, &l) in c.steps.iter().enumerate() {
        let seed = 100 + si as u64;
        let k = tensor(c.batch, l, c.embed, seed);
        let v = tensor(c.batch, l, c.embed, seed + 1000);
        let q = tensor(c.batch, l, c.embed, seed + 2000);
        dev.append(&tape, &k, &v).unwrap();
        reference.append(&k, &v).unwrap();
        let got = dev.attention(&tape, &q, c.heads, scale).unwrap();
        let want = paged_attention_reference(&q, &reference, c.heads, scale).unwrap();
        assert_eq!(got.shape(), want.shape());
        assert_parity(
            &format!("bs={} B={} step={si} L_new={l}", c.block_size, c.batch),
            &flat(&got),
            &flat(&want),
        );
        assert_eq!(dev.seq_len(), reference.seq_len());
        assert_eq!(dev.num_used_blocks(), reference.num_used_blocks());
        if let Some((at, to)) = rewind_after
            && at == si
        {
            dev.rewind_to(to).unwrap();
            reference.rewind_to(to).unwrap();
            assert_eq!(dev.num_free_blocks(), reference.num_free_blocks());
        }
    }
}

#[test]
fn matches_reference_across_block_boundaries() {
    // ブロックサイズ ±1・複数ブロックにまたがる長さ・prefill 後の L_new = 1 の連続。
    for &(bs, steps) in &[
        (1usize, &[3usize, 1, 1][..]),
        (2, &[3, 1, 1, 2]),
        (4, &[3, 1, 1, 1, 1, 5]),
        (4, &[4, 1]),
        (4, &[5, 1]),
        (16, &[15, 1, 1, 17]),
    ] {
        for &(batch, embed, heads) in &[(1usize, 8usize, 2usize), (2, 12, 3), (3, 4, 1)] {
            run_case(
                &Case {
                    block_size: bs,
                    batch,
                    embed,
                    heads,
                    steps,
                },
                None,
            );
        }
    }
}

#[test]
fn matches_reference_after_rewind_and_reappend() {
    run_case(
        &Case {
            block_size: 2,
            batch: 2,
            embed: 8,
            heads: 2,
            steps: &[5, 1, 3, 1],
        },
        Some((1, 3)),
    );
}

// --- 直接 BackendOps 呼び出しの検証エラー（本番経路で panic せず Err） ---

fn buffers(rows: usize, e: usize, layout: &PagedKvLayout) -> [DeviceBuffer<f32>; 4] {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let pool = [layout.num_blocks, layout.block_size, layout.embed_dim];
    [
        mem.upload(&tensor(1, rows, e, 1)).unwrap(),
        mem.upload(&tensor(1, rows, e, 2)).unwrap(),
        mem.alloc_zeroed(&pool).unwrap(),
        mem.alloc_zeroed(&pool).unwrap(),
    ]
}

#[test]
fn append_rejects_out_of_range_and_duplicate_slots() {
    let ops = CpuBackendOps::new();
    let layout = PagedKvLayout::new(2, 2, 4).unwrap();
    let [k, v, mut kp, mut vp] = buffers(2, 4, &layout);
    assert!(
        ops.paged_kv_append(&k, &v, &mut kp, &mut vp, &[0, 3], &layout)
            .is_ok()
    );
    for bad in [[0u32, 4], [0, u32::MAX], [1, 1]] {
        let r = ops.paged_kv_append(&k, &v, &mut kp, &mut vp, &bad, &layout);
        assert!(
            matches!(r, Err(BackendError::InvalidArgument(_))),
            "{bad:?}"
        );
    }
    // 行数と slot_mapping の長さの不一致。
    let r = ops.paged_kv_append(&k, &v, &mut kp, &mut vp, &[0], &layout);
    assert!(matches!(r, Err(BackendError::InvalidArgument(_))));
}

#[test]
fn attention_rejects_bad_block_ids_and_sentinel_in_use() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let layout = PagedKvLayout::new(2, 2, 4).unwrap();
    let args = PagedAttentionArgs::new(layout, 1, 1, 2, 2, 0.5).unwrap();
    let [q, _, kp, vp] = buffers(1, 4, &layout);
    let _ = mem;
    assert!(
        ops.paged_attention_forward(&q, &kp, &vp, &[0, 1], &[3], &args)
            .is_ok()
    );
    for tables in [[0u32, 2], [0, u32::MAX]] {
        let r = ops.paged_attention_forward(&q, &kp, &vp, &tables, &[3], &args);
        assert!(
            matches!(r, Err(BackendError::InvalidArgument(_))),
            "{tables:?}"
        );
    }
    // seq_len が範囲外（上限 2 枠 · block_size 2 = 4）。
    let r = ops.paged_attention_forward(&q, &kp, &vp, &[0, 1], &[5], &args);
    assert!(matches!(r, Err(BackendError::InvalidArgument(_))));
    assert_eq!(q.device(), Device::Cpu);
}
