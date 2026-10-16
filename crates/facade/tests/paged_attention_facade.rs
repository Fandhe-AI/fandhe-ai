//! paged attention（K-3）の facade 公開経路（`Tape::new_paged_kv_cache`・
//! `Tape::paged_attention_forward`。イシュー #3112・親 #3108）の統合テスト。
//!
//! 設計正本は `docs/device-resident-kv-paged-attention-design.md` §6・§13。facade の公開面
//! だけを使い、K-1（`stateful_attention_forward_with_cache`）との出力一致を REQ-2 統一複合判定
//! （`assert_parity`。tolerance 定数は不変）で固定する。あわせて原子性・容量超過・
//! `L_new == 0` を固定する。CPU で動くテスト（形状マトリクス・空キャッシュ・範囲外 `Err`・巻き戻し。#3113）と、
//! 末尾の実機端から端まで `#[ignore]`（`cuda_`／`metal_` 接頭辞。手順は
//! `docs/perf/logs/paged-attention-parity-3113/README.md`、実測は #3171・#3172）からなる。

use fandhe_ai::Tensor;
use fandhe_ai::nn::kv_cache::{
    KvCache, MultiheadAttentionConfig, PagedKvCacheConfig, StatefulAttention,
};
use fandhe_ai_backend_cpu::parity::assert_parity;

const E: usize = 4;
const H: usize = 2;
const SEED: u64 = 11;

fn sa() -> StatefulAttention {
    sa_eh(E, H)
}

fn sa_eh(e: usize, h: usize) -> StatefulAttention {
    StatefulAttention::from_config(&MultiheadAttentionConfig::new(e, h), SEED).expect("from_config")
}

fn input(b: usize, total: usize) -> Tensor<f32> {
    input_e(b, total, E, 0)
}

/// 決定的な入力。`salt` を変えると別系列になる（巻き戻し後の再追記用）。
fn input_e(b: usize, total: usize, e: usize, salt: i64) -> Tensor<f32> {
    let data: Vec<f32> = (0..b * total * e)
        .map(|i| ((211 + salt * 7 + i as i64) % 17 - 8) as f32 * 0.03)
        .collect();
    Tensor::new(data, &[b, total, e]).expect("fixture")
}

fn data_of(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().expect("contiguous").to_vec()
}

fn slice_time(x: &Tensor<f32>, from: usize, to: usize) -> Tensor<f32> {
    let b = x.shape()[0];
    let total = x.shape()[1];
    let e = x.shape()[2];
    let d = data_of(x);
    let mut out = Vec::new();
    for bi in 0..b {
        out.extend_from_slice(&d[(bi * total + from) * e..(bi * total + to) * e]);
    }
    Tensor::new(out, &[b, to - from, e]).expect("slice")
}

/// K-1 の外部キャッシュ版と paged の各ステップ出力が統一複合判定で一致する。
fn compare(b: usize, steps: &[usize], block_size: usize, max_blocks: usize) {
    compare_eh(b, steps, block_size, max_blocks, E, H);
}

fn compare_eh(b: usize, steps: &[usize], block_size: usize, max_blocks: usize, e: usize, h: usize) {
    let total: usize = steps.iter().sum();
    let x = input_e(b, total, e, 0);
    let tape = fandhe_ai::tape();
    let sa = sa_eh(e, h);
    let mut k1 = KvCache::new();
    let mut paged = tape
        .new_paged_kv_cache(PagedKvCacheConfig::new(block_size, max_blocks, e).expect("cfg"))
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
            &format!("paged vs K-1 (b={b}, bs={block_size}, e={e}, h={h}, step {i})"),
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
/// ブロック境界（bs-1／bs／bs+1 の prefill → decode 2 回）× バッチ × (E, H)。
#[test]
fn block_boundary_matrix_matches_k1() {
    for bs in [1usize, 2, 4, 16] {
        for prefill in [bs.saturating_sub(1), bs, bs + 1] {
            if prefill == 0 {
                continue;
            }
            for b in [1usize, 2] {
                for (e, h) in [(4usize, 2usize), (8, 2), (12, 3)] {
                    compare_eh(b, &[prefill, 1, 1], bs, 64, e, h);
                }
            }
        }
    }
}

/// 複数ブロックにまたがる長い系列（prefill と decode を混ぜる）。
#[test]
fn long_multi_block_sequence_matches_k1() {
    compare(2, &[7, 1, 1, 9, 1, 12, 1, 1, 7], 4, 64);
}

/// 空キャッシュへの L_new=0 は何も確保せず `[B, 0, E]` を返し、その後の通常利用が K-1 と一致する。
#[test]
fn empty_cache_zero_length_step_then_prefill_matches_k1() {
    let tape = fandhe_ai::tape();
    let sa = sa();
    let mut cache = tape
        .new_paged_kv_cache(PagedKvCacheConfig::new(2, 16, E).unwrap())
        .unwrap();
    let empty = Tensor::new(Vec::<f32>::new(), &[2, 0, E]).unwrap();
    let y = tape
        .paged_attention_forward(&sa, &empty, &mut cache)
        .expect("空キャッシュへの L_new=0");
    assert_eq!(y.shape(), &[2usize, 0, E]);
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.num_used_blocks(), 0);
    assert!(!cache.is_poisoned());

    let x = input(2, 5);
    let mut k1 = KvCache::new();
    for (i, (from, to)) in [(0usize, 3usize), (3, 4), (4, 5)].into_iter().enumerate() {
        let xs = slice_time(&x, from, to);
        let want = tape
            .stateful_attention_forward_with_cache(&sa, &tape.var(&xs), &mut k1)
            .unwrap()
            .to_tensor();
        let got = tape.paged_attention_forward(&sa, &xs, &mut cache).unwrap();
        assert_parity(
            &format!("empty-then-prefill step {i}"),
            &data_of(&got),
            &data_of(&want),
        );
    }
}

/// `clear()` 後に別のバッチ数で使い直しても、新規の K-1 と一致する。
#[test]
fn clear_then_reuse_matches_fresh_k1() {
    let tape = fandhe_ai::tape();
    let sa = sa();
    let mut cache = tape
        .new_paged_kv_cache(PagedKvCacheConfig::new(2, 16, E).unwrap())
        .unwrap();
    tape.paged_attention_forward(&sa, &input(1, 5), &mut cache)
        .unwrap();
    cache.clear();
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.num_used_blocks(), 0);
    assert_eq!(cache.batch(), None);

    let x = input_e(3, 4, E, 5);
    let mut k1 = KvCache::new();
    for (i, (from, to)) in [(0usize, 3usize), (3, 4)].into_iter().enumerate() {
        let xs = slice_time(&x, from, to);
        let want = tape
            .stateful_attention_forward_with_cache(&sa, &tape.var(&xs), &mut k1)
            .unwrap()
            .to_tensor();
        let got = tape.paged_attention_forward(&sa, &xs, &mut cache).unwrap();
        assert_parity(
            &format!("clear-then-reuse step {i}"),
            &data_of(&got),
            &data_of(&want),
        );
    }
}

/// 巻き戻し後の再追記は、切り詰めた系列を新規に流した K-1 と一致する
/// （K-1 の `kv_rewind` は非公開のため使わず、新しい `KvCache` へ流し直して比較する）。
#[test]
fn rewind_then_reappend_matches_k1_on_truncated_sequence() {
    for (bs, keep) in [(2usize, 3usize), (2, 4), (4, 3), (4, 4)] {
        let tape = fandhe_ai::tape();
        let sa = sa();
        let mut cache = tape
            .new_paged_kv_cache(PagedKvCacheConfig::new(bs, 16, E).unwrap())
            .unwrap();
        let x = input(2, 5);
        let mut k1_full = KvCache::new();
        let want_full = tape
            .stateful_attention_forward_with_cache(&sa, &tape.var(&x), &mut k1_full)
            .unwrap()
            .to_tensor();
        let got_full = tape.paged_attention_forward(&sa, &x, &mut cache).unwrap();
        assert_parity(
            "rewind: first pass",
            &data_of(&got_full),
            &data_of(&want_full),
        );

        cache.rewind_to(keep).expect("rewind");
        assert_eq!(cache.seq_len(), keep);
        let x2 = input_e(2, 3, E, 9);
        let got = tape.paged_attention_forward(&sa, &x2, &mut cache).unwrap();

        let mut k1 = KvCache::new();
        tape.stateful_attention_forward_with_cache(
            &sa,
            &tape.var(&slice_time(&x, 0, keep)),
            &mut k1,
        )
        .unwrap();
        let want = tape
            .stateful_attention_forward_with_cache(&sa, &tape.var(&x2), &mut k1)
            .unwrap()
            .to_tensor();
        assert_parity(
            &format!("rewind(bs={bs}, keep={keep}) then reappend"),
            &data_of(&got),
            &data_of(&want),
        );
        assert_eq!(cache.seq_len(), keep + 3);
    }
}

/// 範囲外の設定・巻き戻し・形状不一致は `Err` で、失敗後も状態が変わらない（fail-closed）。
#[test]
fn out_of_range_inputs_are_err_and_keep_state() {
    for (bs, mb) in [(0usize, 4usize), (3, 4), (512, 4), (2, 0)] {
        assert!(
            PagedKvCacheConfig::new(bs, mb, E).is_err(),
            "block_size={bs}, max_blocks={mb} は Err"
        );
    }
    let tape = fandhe_ai::tape();
    let sa = sa();
    let mut cache = tape
        .new_paged_kv_cache(PagedKvCacheConfig::new(2, 8, E).unwrap())
        .unwrap();
    tape.paged_attention_forward(&sa, &input(1, 3), &mut cache)
        .unwrap();
    let used = cache.num_used_blocks();
    assert!(cache.rewind_to(4).is_err(), "seq_len を超える巻き戻し");
    assert_eq!(cache.seq_len(), 3);
    assert_eq!(cache.num_used_blocks(), used);

    // config の embed_dim が attention の E と違う
    let mut wrong = tape
        .new_paged_kv_cache(PagedKvCacheConfig::new(2, 8, 6).unwrap())
        .unwrap();
    assert!(
        tape.paged_attention_forward(&sa, &input(1, 2), &mut wrong)
            .is_err()
    );
    assert_eq!(wrong.seq_len(), 0);
    assert_eq!(wrong.num_used_blocks(), 0);
    assert!(!wrong.is_poisoned());
    assert!(!cache.is_poisoned());
}

// --- 実機（CUDA／Metal）端から端まで。本エージェント実行環境に実機がないため未実測の
// まま申し送る（実測は #3171・#3172）。q/k/v と out_proj の射影 GEMM を含む合成経路なので
// 判定は厳密（統一複合判定のゼロ fail）のみ。fail した場合は backend 層マトリクス
// （`crates/backend-*/tests/paged_kv_real_device.rs`）の結果と合わせて事実を記録して申し送る。
// baseline への切り替えは backend 層の `BASELINES` 表で扱う。 ---

/// 同じ重み・同じ入力を CPU tape と `device` tape の paged 経路へ流し、ステップごとに比べる。
fn assert_device_matches_cpu(device: fandhe_ai::Device) {
    for &(bs, b, e, h, steps) in &[
        (2usize, 1usize, 8usize, 2usize, &[3usize, 1, 1][..]),
        (4, 2, 8, 2, &[5, 1, 2]),
    ] {
        let sa = sa_eh(e, h);
        let x = input_e(b, steps.iter().sum(), e, 0);
        let cpu_tape = fandhe_ai::tape();
        let dev_tape =
            fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
        let cfg = || PagedKvCacheConfig::new(bs, 32, e).unwrap();
        let mut cpu_cache = cpu_tape.new_paged_kv_cache(cfg()).unwrap();
        let mut dev_cache = dev_tape.new_paged_kv_cache(cfg()).unwrap();
        let mut at = 0;
        for (i, &len) in steps.iter().enumerate() {
            let xs = slice_time(&x, at, at + len);
            let want = cpu_tape
                .paged_attention_forward(&sa, &xs, &mut cpu_cache)
                .unwrap();
            let got = dev_tape
                .paged_attention_forward(&sa, &xs, &mut dev_cache)
                .unwrap();
            assert_parity(
                &format!("paged facade {device:?} vs cpu (bs={bs}, b={b}, step {i})"),
                &data_of(&got),
                &data_of(&want),
            );
            at += len;
        }
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。docs/perf/logs/paged-attention-parity-3113/README.md 参照"]
fn cuda_paged_attention_forward_matches_cpu_tape() {
    assert_device_matches_cpu(fandhe_ai::Device::Cuda(0));
}

// `Device::Metal` は `cfg(target_os = "macos")` 限定の variant のため、参照する関数ごと macOS 限定。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。docs/perf/logs/paged-attention-parity-3113/README.md 参照"]
fn metal_paged_attention_forward_matches_cpu_tape() {
    assert_device_matches_cpu(fandhe_ai::Device::Metal);
}
