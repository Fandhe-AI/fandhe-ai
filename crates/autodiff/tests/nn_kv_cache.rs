//! `nn::KvCache`／`MultiheadAttentionVars::forward_with_cache`／
//! `nn::StatefulAttention`（イシュー #2084・親 #2059。設計正本
//! `docs/kv-cache-design.md`）の受け入れ条件対応テスト。
//!
//! `tests/nn_attention.rs` と同じ構成方針: `common::naive_ops()`
//! （`BackendOps` の softmax／gemm_batched 等は既定合成フォールバック
//! 経由）のみを使い、`backend-cpu` 等の具体バックエンドへ依存しない
//! （`common/mod.rs` 冒頭コメントの設計上の不変条件）。
//!
//! 「全系列再計算」と「prefill → decode の各ステップ出力を連結した
//! もの」の比較は REQ-2 統一複合判定（`common::req2_close`）でハード
//! assert する。bit 一致は `docs/kv-cache-design.md` §3.5 が「事前登録の
//! 仮説」と位置づける（CPU GEMM の shape 依存ブロッキングパラメータの
//! ため）ため、ハード assert しない。(a) 経路（空 cache からの
//! prefill）は `forward`／`forward_with_cache` が完全に同一の
//! `sdpa_compose` 呼び出しへ帰着するため bit 一致を期待する。

mod common;

use fandhe_ai_autodiff::nn::{KvCache, MultiheadAttention, StatefulAttention};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// REQ-2 統一複合判定（`common::req2_close` に委譲。`nn_attention.rs::
/// assert_req2_close` と同型）。
fn assert_req2_close(label: &str, actual: &Tensor<f32>, expected: &Tensor<f32>) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label}: shape が一致しない"
    );
    let shape = actual.shape().to_vec();
    let numel: usize = shape.iter().product();
    if numel == 0 {
        return;
    }
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..numel {
        let av = actual.get(&idx).unwrap_or(0.0);
        let ev = expected.get(&idx).unwrap_or(0.0);
        assert!(
            common::req2_close(av as f64, ev as f64),
            "{label}[{idx:?}]: actual={av} expected={ev}"
        );
        for axis in (0..shape.len()).rev() {
            idx[axis] += 1;
            if idx[axis] < shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
}

/// `[B, len_i, E]` の列を `dim=1` で連結した `[B, sum(len_i), E]` を
/// ホスト側で組み立てる（比較専用のテストヘルパー。`Var::cat` は別
/// `Tape` をまたいだ入力を受け付けないため使えない）。
fn concat_seq(chunks: &[Tensor<f32>]) -> Tensor<f32> {
    let b = chunks[0].shape()[0];
    let e = chunks[0].shape()[2];
    let total_len: usize = chunks.iter().map(|c| c.shape()[1]).sum();
    let mut data = Vec::with_capacity(b * total_len * e);
    for bi in 0..b {
        for chunk in chunks {
            let len = chunk.shape()[1];
            for li in 0..len {
                for ei in 0..e {
                    data.push(chunk.get(&[bi, li, ei]).unwrap());
                }
            }
        }
    }
    t(data, &[b, total_len, e])
}

/// 決定的な入力データ（`sin` を使い、負値・正値が混在するようにする）。
fn fixture_sequence(b: usize, len: usize, e: usize) -> Tensor<f32> {
    let mut data = Vec::with_capacity(b * len * e);
    for i in 0..(b * len * e) {
        data.push((i as f32 * 0.37).sin());
    }
    t(data, &[b, len, e])
}

const B: usize = 1;
const E: usize = 4;
const H: usize = 2;
const SEED: u64 = 11;

#[test]
fn prefill_then_single_token_decode_matches_full_recompute() {
    let total_len = 5;
    let t0 = 2;
    let x = fixture_sequence(B, total_len, E);
    let mha = MultiheadAttention::new(E, H, true, SEED).unwrap();

    // 全系列再計算。
    let tape_full = Tape::new_with_ops(common::naive_ops());
    let xv_full = tape_full.var(&x);
    let full_out = mha
        .bind(&tape_full)
        .forward(&xv_full, &xv_full, &xv_full, None, true)
        .unwrap()
        .to_tensor();

    // prefill → decode（ステップごとに新規 Tape。`docs/kv-cache-design.md`
    // §3.2 の運用）。
    let mut cache = KvCache::new();
    let mut outputs = Vec::new();
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let x_prefill = xv.narrow(1, 0, t0).unwrap();
        let y = mha
            .bind(&tape)
            .forward_with_cache(&x_prefill, &x_prefill, &x_prefill, &mut cache)
            .unwrap();
        outputs.push(y.to_tensor());
    }
    for step in t0..total_len {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let x_t = xv.narrow(1, step, 1).unwrap();
        let y = mha
            .bind(&tape)
            .forward_with_cache(&x_t, &x_t, &x_t, &mut cache)
            .unwrap();
        outputs.push(y.to_tensor());
    }

    let combined = concat_seq(&outputs);
    assert_req2_close("prefill_then_decode", &combined, &full_out);
    assert_eq!(cache.seq_len(), total_len);
}

#[test]
fn prefill_then_multi_token_append_matches_full_recompute() {
    // 規則 (c): cache 非空・L_new > 1（複数トークンの追記）。
    let total_len = 6;
    let t0 = 2;
    let append_len = 3; // t0 + append_len < total_len にして端を残す。
    let x = fixture_sequence(B, total_len, E);
    let mha = MultiheadAttention::new(E, H, true, SEED).unwrap();

    let target_len = t0 + append_len;
    let tape_full = Tape::new_with_ops(common::naive_ops());
    let xv_full = tape_full.var(&x);
    let x_target = xv_full.narrow(1, 0, target_len).unwrap();
    let full_out = mha
        .bind(&tape_full)
        .forward(&x_target, &x_target, &x_target, None, true)
        .unwrap()
        .to_tensor();

    let mut cache = KvCache::new();
    let mut outputs = Vec::new();
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let x_prefill = xv.narrow(1, 0, t0).unwrap();
        let y = mha
            .bind(&tape)
            .forward_with_cache(&x_prefill, &x_prefill, &x_prefill, &mut cache)
            .unwrap();
        outputs.push(y.to_tensor());
    }
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let x_append = xv.narrow(1, t0, append_len).unwrap();
        let y = mha
            .bind(&tape)
            .forward_with_cache(&x_append, &x_append, &x_append, &mut cache)
            .unwrap();
        outputs.push(y.to_tensor());
    }

    let combined = concat_seq(&outputs);
    assert_req2_close("prefill_then_multi_append", &combined, &full_out);
    assert_eq!(cache.seq_len(), target_len);
}

#[test]
fn empty_cache_prefill_bit_matches_forward_with_is_causal_true() {
    // 規則 (a): 空 cache からの forward_with_cache は forward(..., None,
    // true) と完全に同一の sdpa_compose 呼び出しへ帰着するため bit 一致
    // を期待する（`docs/kv-cache-design.md` §3.5）。
    let len = 3;
    let x = fixture_sequence(B, len, E);
    let mha = MultiheadAttention::new(E, H, true, SEED).unwrap();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&x);
    let forward_out = mha
        .bind(&tape_a)
        .forward(&xv_a, &xv_a, &xv_a, None, true)
        .unwrap()
        .to_tensor();

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&x);
    let mut cache = KvCache::new();
    let cache_out = mha
        .bind(&tape_b)
        .forward_with_cache(&xv_b, &xv_b, &xv_b, &mut cache)
        .unwrap()
        .to_tensor();

    assert_eq!(
        forward_out.contiguous().as_slice().unwrap(),
        cache_out.contiguous().as_slice().unwrap(),
        "空 cache からの prefill は forward(..., None, true) と bit 一致するはず"
    );
}

#[test]
fn seq_len_grows_and_clear_resets_to_empty() {
    let x = fixture_sequence(B, 4, E);
    let mha = MultiheadAttention::new(E, H, true, SEED).unwrap();
    let mut cache = KvCache::new();
    assert_eq!(cache.seq_len(), 0);

    for step in 0..3 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let x_t = xv.narrow(1, step, 1).unwrap();
        mha.bind(&tape)
            .forward_with_cache(&x_t, &x_t, &x_t, &mut cache)
            .unwrap();
        assert_eq!(cache.seq_len(), step + 1);
        assert_eq!(cache.k().unwrap().shape(), &[B, step + 1, E]);
    }

    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);

    // clear 後は再度 prefill できる。
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let x_prefill = xv.narrow(1, 0, 2).unwrap();
    mha.bind(&tape)
        .forward_with_cache(&x_prefill, &x_prefill, &x_prefill, &mut cache)
        .unwrap();
    assert_eq!(cache.seq_len(), 2);
}

// --- エラー経路（panic せず型付きエラー・cache は不変） ---

fn snapshot(cache: &KvCache) -> (usize, Option<Vec<f32>>) {
    (
        cache.seq_len(),
        cache
            .k()
            .map(|k| k.contiguous().as_slice().unwrap().to_vec()),
    )
}

#[test]
fn forward_with_cache_rejects_batch_mismatch_and_leaves_cache_unchanged() {
    let mha = MultiheadAttention::new(E, H, true, SEED).unwrap();
    let mut cache = KvCache::new();

    // まず batch=1 で prefill してキャッシュを非空にする。
    let x1 = fixture_sequence(1, 2, E);
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x1);
        mha.bind(&tape)
            .forward_with_cache(&xv, &xv, &xv, &mut cache)
            .unwrap();
    }
    let before = snapshot(&cache);

    // batch=2 の新規トークンを渡すと B 不一致で拒否される。
    let x2 = fixture_sequence(2, 1, E);
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x2);
    let err = mha
        .bind(&tape)
        .forward_with_cache(&xv, &xv, &xv, &mut cache)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
    assert_eq!(
        snapshot(&cache),
        before,
        "エラー時に cache が変化してはならない"
    );
}

#[test]
fn forward_with_cache_rejects_rank_mismatch() {
    let mha = MultiheadAttention::new(E, H, true, SEED).unwrap();
    let mut cache = KvCache::new();
    let bad = t(vec![0.0; E], &[E]); // rank 1（rank 3 を要求）
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&bad);
    let err = mha
        .bind(&tape)
        .forward_with_cache(&xv, &xv, &xv, &mut cache)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
    assert!(cache.is_empty());
}

#[test]
fn forward_with_cache_rejects_key_value_shape_mismatch() {
    let mha = MultiheadAttention::new(E, H, true, SEED).unwrap();
    let mut cache = KvCache::new();
    let q = fixture_sequence(B, 2, E);
    let k = fixture_sequence(B, 3, E); // key の系列長が query と食い違う
    let tape = Tape::new_with_ops(common::naive_ops());
    let qv = tape.var(&q);
    let kv = tape.var(&k);
    let err = mha
        .bind(&tape)
        .forward_with_cache(&qv, &kv, &kv, &mut cache)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(cache.is_empty());
}

#[test]
fn forward_with_cache_rejects_different_tape_vars() {
    let mha = MultiheadAttention::new(E, H, true, SEED).unwrap();
    let mut cache = KvCache::new();
    let x = fixture_sequence(B, 2, E);
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let qv = tape_a.var(&x);
    let kv = tape_b.var(&x);
    let err = mha
        .bind(&tape_a)
        .forward_with_cache(&qv, &kv, &kv, &mut cache)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
    assert!(cache.is_empty());
}

// --- StatefulAttention ---

#[test]
fn stateful_attention_forward_matches_manual_forward_with_cache() {
    let x = fixture_sequence(B, 3, E);
    let mha_a = MultiheadAttention::new(E, H, true, SEED).unwrap();
    let mha_b = MultiheadAttention::new(E, H, true, SEED).unwrap();
    let mut stateful = StatefulAttention::new(mha_a);
    let mut cache = KvCache::new();

    for step in 0..3 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let x_t = xv.narrow(1, step, 1).unwrap();

        let via_stateful = stateful.forward(&tape, &x_t).unwrap().to_tensor();
        let via_manual = mha_b
            .bind(&tape)
            .forward_with_cache(&x_t, &x_t, &x_t, &mut cache)
            .unwrap()
            .to_tensor();

        assert_eq!(
            via_stateful.contiguous().as_slice().unwrap(),
            via_manual.contiguous().as_slice().unwrap(),
            "step={step}: StatefulAttention::forward は forward_with_cache への 1 行委譲のため bit 一致するはず"
        );
    }
    assert_eq!(stateful.seq_len(), 3);

    stateful.reset_cache();
    assert_eq!(stateful.seq_len(), 0);
    assert!(stateful.cache().is_empty());
}

// --- 勾配 ---

#[test]
fn backward_on_decode_step_flows_to_current_projection_params() {
    let x = fixture_sequence(B, 2, E);
    let mha = MultiheadAttention::new(E, H, true, SEED).unwrap();
    let mut cache = KvCache::new();

    // prefill（勾配追跡なし。`var_no_grad` 相当ではなく通常 var だが、
    // このステップの Tape は decode ステップとは別のため backward の
    // 対象外）。
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let x_prefill = xv.narrow(1, 0, 1).unwrap();
        mha.bind(&tape)
            .forward_with_cache(&x_prefill, &x_prefill, &x_prefill, &mut cache)
            .unwrap();
    }

    // decode ステップで backward し、現ステップの q_proj weight に
    // 勾配が出ることを確認する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let x_t = xv.narrow(1, 1, 1).unwrap();
    let vars = mha.bind(&tape);
    let y = vars
        .forward_with_cache(&x_t, &x_t, &x_t, &mut cache)
        .unwrap();
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let q_weight_grad = grads
        .get(&vars.q.weight)
        .unwrap()
        .expect("現ステップの q_proj.weight は loss に寄与しているため勾配が存在するはず");
    assert_eq!(q_weight_grad.shape(), &[E, E]);
}
