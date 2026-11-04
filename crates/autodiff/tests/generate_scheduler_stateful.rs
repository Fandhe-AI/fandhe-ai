//! `BatchScheduler::step_stateful`（内部状態を持つモデルの連続バッチング。イシュー #3849・親 #3719。
//! 設計記録 `docs/facade-speculative-decoding-batching-design.md` §29・§31）の統合テスト。
//! autodiff の公開経路だけを使い、`backend-cpu` へは依存しない（`architecture_boundaries.rs` の不変条件）。
//!
//! # 検査内容
//!
//! - 要求ごとの出力が単独 `generate`（`RefCell` に状態を持つ参照）と token 列で完全一致する
//!   （Greedy・TopK・Temperature。参加のずれ・スロット再利用を含む）
//! - `State = Vec<KvCache>` のアダプタで `step_stateful` が `step` と bit 一致する（RNG 消費順の直接検査）
//! - fail-closed (a)〜(d)（要求の種類の取り違え・`State` 型違い・`init_state` 失敗）と forward の `Err` が
//!   要求単位の失敗になり、ほかの要求を巻き込まない
//! - `init_state` の呼び出し回数・上限・`Debug` が状態を出さないこと
//!
//! 対象外: 実機（CUDA／Metal）parity（新しいカーネル・バックエンド経路が無く `#[ignore]` で分けるものが
//! 無い。設計 §29.4）。

use fandhe_ai_autodiff::generate::scheduler::{BatchScheduler, RequestId, SchedulerLimits};
use fandhe_ai_autodiff::generate::stateful::StatefulAutoregressiveModel;
use fandhe_ai_autodiff::generate::{
    AutoregressiveModel, GenerateConfig, SamplingStrategy, generate,
};
use fandhe_ai_autodiff::nn::{KvCache, LinearVars, MultiheadAttentionVars};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;
use std::cell::{Cell, RefCell};

const V: usize = 8;
const E: usize = 4;
const HEADS: usize = 2;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture: shape とデータ長は事前に一致させている")
}

fn seq(seed: i64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((seed + i as i64 * 7) % 13 - 6) as f32 * 0.05)
        .collect()
}

fn ids1(v: &[i32]) -> Tensor<i32> {
    Tensor::new(v.to_vec(), &[v.len()]).expect("fixture")
}

fn vals(x: &Tensor<i32>) -> Vec<i32> {
    x.contiguous().host_slice().into_owned()
}

/// `(token, 位置)` だけで決まるなだらかな logits（同値なし。TopK／Temperature でも RNG 消費が効く）。
fn soft_row(tok: i32, pos: usize) -> [f32; V] {
    let mut row = [0.0f32; V];
    for (j, r) in row.iter_mut().enumerate() {
        *r = ((tok as usize * 7 + pos * 3 + j * 5) % 11) as f32 * 0.4;
    }
    row
}

/// `toks` を絶対位置 `pos0` から並べた `[1, L, V]` の logits。
fn soft_logits(toks: &[i32], pos0: usize) -> Tensor<f32> {
    let mut data = Vec::with_capacity(toks.len() * V);
    for (i, &tok) in toks.iter().enumerate() {
        data.extend_from_slice(&soft_row(tok, pos0 + i));
    }
    t(data, &[1, toks.len(), V])
}

fn injected(msg: &str) -> AutodiffError {
    AutodiffError::InvalidArgument(format!("fixture: {msg}"))
}

// ---- フィクスチャ: 整数の隠れ状態 --------------------------------------------

/// 状態が目印付きの処理済みトークン数。`Debug` は目印を含む（スケジューラの `Debug` に出ないことを検査する）。
#[derive(Debug)]
struct Hidden {
    pos: usize,
    /// 最初の prefill の先頭 token（要求の識別用）。
    tag: i32,
    #[allow(dead_code)]
    note: &'static str,
}

const SECRET: &str = "SECRET_STATE_MARKER_3849";

struct Rnn {
    init_calls: Cell<usize>,
    forward_calls: Cell<usize>,
    /// n 回目（1 始まり）の `init_state` だけ `Err`。
    fail_init_nth: Option<usize>,
    /// 先頭 token が `tag` で、`pos >= 2` の forward を `Err` にする。
    poison_tag: Option<i32>,
}

impl Rnn {
    fn new() -> Rnn {
        Rnn {
            init_calls: Cell::new(0),
            forward_calls: Cell::new(0),
            fail_init_nth: None,
            poison_tag: None,
        }
    }
}

impl StatefulAutoregressiveModel for Rnn {
    type State = Hidden;

    fn init_state(&self) -> Result<Hidden, AutodiffError> {
        let n = self.init_calls.get() + 1;
        self.init_calls.set(n);
        if self.fail_init_nth == Some(n) {
            return Err(injected("init_state"));
        }
        Ok(Hidden {
            pos: 0,
            tag: -1,
            note: SECRET,
        })
    }

    fn forward_step_stateful(
        &self,
        new_ids: &Tensor<i32>,
        state: &mut Hidden,
    ) -> Result<Tensor<f32>, AutodiffError> {
        self.forward_calls.set(self.forward_calls.get() + 1);
        let toks = vals(new_ids);
        if state.pos == 0 {
            state.tag = toks[0];
        }
        if self.poison_tag == Some(state.tag) && state.pos >= 2 {
            return Err(injected("forward"));
        }
        let out = soft_logits(&toks, state.pos);
        state.pos += toks.len();
        Ok(out)
    }

    fn save_state(&self, state: &Hidden) -> Result<Hidden, AutodiffError> {
        Ok(Hidden {
            pos: state.pos,
            tag: state.tag,
            note: SECRET,
        })
    }

    fn restore_state(&self, state: &mut Hidden, saved: Hidden) -> Result<(), AutodiffError> {
        *state = saved;
        Ok(())
    }
}

/// `State` の型だけが `Rnn` と異なるモデル（(c) の downcast 失敗用）。
struct OtherRnn;

impl StatefulAutoregressiveModel for OtherRnn {
    type State = u64;

    fn init_state(&self) -> Result<u64, AutodiffError> {
        Ok(0)
    }

    fn forward_step_stateful(
        &self,
        new_ids: &Tensor<i32>,
        state: &mut u64,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let toks = vals(new_ids);
        let out = soft_logits(&toks, *state as usize);
        *state += toks.len() as u64;
        Ok(out)
    }

    fn save_state(&self, state: &u64) -> Result<u64, AutodiffError> {
        Ok(*state)
    }

    fn restore_state(&self, state: &mut u64, saved: u64) -> Result<(), AutodiffError> {
        *state = saved;
        Ok(())
    }
}

/// 同じ規則を `RefCell` に状態を持つ `AutoregressiveModel`（`num_kv_layers() == 0`）で実装した参照。
/// 実行ごとに新しいインスタンスを使う。`forward` が一度も呼ばれていないことの検査にも使う。
struct RefRnn {
    pos: RefCell<usize>,
}

impl RefRnn {
    fn new() -> RefRnn {
        RefRnn {
            pos: RefCell::new(0),
        }
    }
}

impl AutoregressiveModel for RefRnn {
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
        let out = soft_logits(&toks, *pos);
        *pos += toks.len();
        Ok(out)
    }
}

/// 位置に依存しない状態なしの表引きモデル（`step`／`step_batched` で進める KV 形の要求用）。
struct Table;

impl AutoregressiveModel for Table {
    fn num_kv_layers(&self) -> usize {
        0
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        _caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (b, l) = (new_ids.shape()[0], new_ids.shape()[1]);
        let ids = vals(new_ids);
        let mut data = Vec::with_capacity(b * l * V);
        for &tok in &ids {
            data.extend_from_slice(&soft_row(tok, 0));
        }
        Ok(t(data, &[b, l, V]))
    }
}

/// `State = Vec<KvCache>` のアダプタ。KV を実 MHA で進め、位置は `caches[0].seq_len()` から取る
/// （`nn_generate_speculative_stateful.rs` の `TableKvModel` と同型）。
struct KvAdapter;

impl KvAdapter {
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
        Ok(soft_logits(&vals(new_ids), pos0))
    }
}

impl AutoregressiveModel for KvAdapter {
    fn num_kv_layers(&self) -> usize {
        1
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        self.step(new_ids, caches)
    }
}

impl StatefulAutoregressiveModel for KvAdapter {
    type State = Vec<KvCache>;

    fn init_state(&self) -> Result<Vec<KvCache>, AutodiffError> {
        Ok(vec![KvCache::new()])
    }

    fn forward_step_stateful(
        &self,
        new_ids: &Tensor<i32>,
        state: &mut Vec<KvCache>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        self.step(new_ids, state)
    }

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

// ---- 補助 ----------------------------------------------------------------

fn limits(max_active: usize, max_queued: usize) -> SchedulerLimits {
    SchedulerLimits::new(max_active, max_queued, 32).unwrap()
}

fn greedy(max_length: usize) -> GenerateConfig {
    GenerateConfig::new(max_length, SamplingStrategy::Greedy)
}

/// 単独 `generate` の参照出力（新しい `RefRnn` で実行する）。
fn reference(prompt: &Tensor<i32>, config: &GenerateConfig) -> Vec<i32> {
    vals(&generate(&RefRnn::new(), prompt, config).unwrap())
}

type Outputs = Vec<(RequestId, Vec<i32>)>;

/// 指定の step 関数でスケジューラが空になるまで回し、(完了, 失敗した id) を返す。
fn drain<F>(s: &mut BatchScheduler, mut step: F) -> (Outputs, Vec<RequestId>)
where
    F: FnMut(&mut BatchScheduler) -> Result<usize, AutodiffError>,
{
    let mut done: Outputs = Vec::new();
    let mut failed = Vec::new();
    for _ in 0..200 {
        step(s).unwrap();
        done.extend(s.take_finished().into_iter().map(|(id, o)| (id, vals(&o))));
        failed.extend(s.take_failed().into_iter().map(|(id, _)| id));
        if s.queued_len() + s.active_len() == 0 {
            return (done, failed);
        }
    }
    panic!("スケジューラが 200 step で空にならない");
}

fn out_of(done: &Outputs, id: RequestId) -> &Vec<i32> {
    &done
        .iter()
        .find(|(i, _)| *i == id)
        .expect("完了していない")
        .1
}

// ---- テスト --------------------------------------------------------------

#[test]
fn interleaved_requests_match_single_generate() {
    let configs = [
        greedy(9),
        GenerateConfig::new(9, SamplingStrategy::TopK(3)).with_seed(5),
        GenerateConfig::new(9, SamplingStrategy::Temperature(0.8)).with_seed(11),
    ];
    for base in configs {
        // max_active = 2 で 3 要求（長さ・prompt が異なる）。3 つ目は枠の再利用で途中参加する。
        let rnn = Rnn::new();
        let mut s = BatchScheduler::new(limits(2, 8));
        let mut cfg_a = base.clone();
        cfg_a.max_length = 9;
        let mut cfg_b = base.clone();
        cfg_b.max_length = 5;
        let mut cfg_c = base.clone();
        cfg_c.max_length = 7;
        let p_a = ids1(&[0, 1, 2]);
        let p_b = ids1(&[3]);
        let p_c = Tensor::new(vec![5, 4], &[1, 2]).unwrap();
        let ia = s.submit(&p_a, &cfg_a).unwrap();
        let ib = s.submit(&p_b, &cfg_b).unwrap();
        let ic = s.submit(&p_c, &cfg_c).unwrap();
        let (done, failed) = drain(&mut s, |s| s.step_stateful(&rnn));
        assert!(failed.is_empty());
        assert_eq!(done.len(), 3);
        assert_eq!(out_of(&done, ia), &reference(&p_a, &cfg_a));
        assert_eq!(out_of(&done, ib), &reference(&p_b, &cfg_b));
        assert_eq!(out_of(&done, ic), &reference(&p_c, &cfg_c));
        // init_state は参加した要求ごとに 1 回。
        assert_eq!(rnn.init_calls.get(), 3);
    }
}

#[test]
fn kv_state_adapter_matches_step_bit_for_bit() {
    let configs = [
        greedy(8),
        GenerateConfig::new(8, SamplingStrategy::TopK(3)).with_seed(21),
        GenerateConfig::new(8, SamplingStrategy::Temperature(0.8)).with_seed(21),
    ];
    for cfg in configs {
        let prompts = [ids1(&[0, 1]), ids1(&[4]), ids1(&[6, 2, 3])];
        let run = |stateful: bool| -> Outputs {
            let mut s = BatchScheduler::new(limits(2, 8));
            let ids: Vec<RequestId> = prompts.iter().map(|p| s.submit(p, &cfg).unwrap()).collect();
            let (done, failed) = if stateful {
                drain(&mut s, |s| s.step_stateful(&KvAdapter))
            } else {
                drain(&mut s, |s| s.step(&KvAdapter))
            };
            assert!(failed.is_empty());
            ids.iter()
                .map(|&id| (id, out_of(&done, id).clone()))
                .collect()
        };
        assert_eq!(run(true), run(false), "{cfg:?}");
    }
}

#[test]
fn stateful_request_stepped_by_step_fails_alone_without_forward() {
    // (a): num_kv_layers() == 0 のモデルでは caches 数の検査だけでは見逃すため、種類の検査が要る。
    let rnn = Rnn::new();
    let mut s = BatchScheduler::new(limits(2, 4));
    let id = s.submit(&ids1(&[1, 2]), &greedy(6)).unwrap();
    s.step_stateful(&rnn).unwrap();
    assert_eq!(s.active_len(), 1);
    let reference_model = RefRnn::new();
    s.step(&reference_model).unwrap();
    let failed = s.take_failed();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].0, id);
    assert!(matches!(failed[0].1, AutodiffError::InvalidArgument(_)));
    assert_eq!(
        *reference_model.pos.borrow(),
        0,
        "forward を呼んではならない"
    );
    assert_eq!(s.active_len(), 0);
    assert!(s.take_finished().is_empty());
}

#[test]
fn step_batched_fails_only_the_stateful_request() {
    let rnn = Rnn::new();
    let mut s = BatchScheduler::new(limits(3, 8));
    let sid = s.submit(&ids1(&[1, 2]), &greedy(6)).unwrap();
    s.step_stateful(&rnn).unwrap();
    let p1 = ids1(&[0]);
    let p2 = ids1(&[3]);
    let k1 = s.submit(&p1, &greedy(6)).unwrap();
    let k2 = s.submit(&p2, &greedy(6)).unwrap();
    let (done, failed) = drain(&mut s, |s| s.step_batched(&Table));
    assert_eq!(failed, vec![sid]);
    let want = |p: &Tensor<i32>| vals(&generate(&Table, p, &greedy(6)).unwrap());
    assert_eq!(out_of(&done, k1), &want(&p1));
    assert_eq!(out_of(&done, k2), &want(&p2));
}

#[test]
fn kv_request_stepped_by_step_stateful_fails_alone() {
    // (b): KV 形で参加した要求を step_stateful で進めると、その要求だけが失敗する。
    let rnn = Rnn::new();
    let mut s = BatchScheduler::new(limits(3, 8));
    let kid = s.submit(&ids1(&[1]), &greedy(6)).unwrap();
    s.step(&Table).unwrap();
    assert_eq!(s.active_len(), 1);
    let p = ids1(&[2, 3]);
    let sid = s.submit(&p, &greedy(6)).unwrap();
    let (done, failed) = drain(&mut s, |s| s.step_stateful(&rnn));
    assert_eq!(failed, vec![kid]);
    assert_eq!(out_of(&done, sid), &reference(&p, &greedy(6)));
}

#[test]
fn state_type_mismatch_fails_only_that_request() {
    // (c): step 間で State 型の違うモデルを渡すと、downcast に失敗した要求だけが失敗する。
    let rnn = Rnn::new();
    let mut s = BatchScheduler::new(limits(3, 8));
    let first = s.submit(&ids1(&[1, 2]), &greedy(6)).unwrap();
    s.step_stateful(&rnn).unwrap();
    let p = ids1(&[4]);
    let second = s.submit(&p, &greedy(6)).unwrap();
    s.step_stateful(&OtherRnn).unwrap();
    let failed = s.take_failed();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].0, first);
    // 新しく参加した要求は OtherRnn の State で進み、続けて OtherRnn で完了できる。
    let (done, failed) = drain(&mut s, |s| s.step_stateful(&OtherRnn));
    assert!(failed.is_empty());
    assert_eq!(out_of(&done, second), &reference(&p, &greedy(6)));
}

#[test]
fn init_state_failure_isolates_one_request() {
    // (d): init_state が Err を返した要求だけが失敗し、ほかは完了する。
    let mut rnn = Rnn::new();
    rnn.fail_init_nth = Some(2);
    let mut s = BatchScheduler::new(limits(3, 8));
    let prompts = [ids1(&[0]), ids1(&[1]), ids1(&[2])];
    let ids: Vec<RequestId> = prompts
        .iter()
        .map(|p| s.submit(p, &greedy(5)).unwrap())
        .collect();
    let (done, failed) = drain(&mut s, |s| s.step_stateful(&rnn));
    assert_eq!(failed, vec![ids[1]]);
    assert_eq!(out_of(&done, ids[0]), &reference(&prompts[0], &greedy(5)));
    assert_eq!(out_of(&done, ids[2]), &reference(&prompts[2], &greedy(5)));
    assert_eq!(rnn.init_calls.get(), 3);
}

#[test]
fn forward_error_isolates_one_request() {
    let mut rnn = Rnn::new();
    rnn.poison_tag = Some(6);
    let mut s = BatchScheduler::new(limits(3, 8));
    let prompts = [ids1(&[0]), ids1(&[6]), ids1(&[2])];
    let ids: Vec<RequestId> = prompts
        .iter()
        .map(|p| s.submit(p, &greedy(7)).unwrap())
        .collect();
    let (done, failed) = drain(&mut s, |s| s.step_stateful(&rnn));
    assert_eq!(failed, vec![ids[1]]);
    assert_eq!(out_of(&done, ids[0]), &reference(&prompts[0], &greedy(7)));
    assert_eq!(out_of(&done, ids[2]), &reference(&prompts[2], &greedy(7)));
}

#[test]
fn init_state_is_not_called_for_prompt_only_or_empty_scheduler() {
    let rnn = Rnn::new();
    let mut s = BatchScheduler::new(limits(2, 4));
    // 空のスケジューラはモデルを呼ばない。
    assert_eq!(s.step_stateful(&rnn).unwrap(), 0);
    assert_eq!(rnn.init_calls.get(), 0);
    assert_eq!(rnn.forward_calls.get(), 0);
    // prompt だけの要求（max_length == prompt 長）は forward も init_state も呼ばずに完了する。
    let p = ids1(&[1, 2, 3]);
    let id = s.submit(&p, &greedy(3)).unwrap();
    assert_eq!(s.step_stateful(&rnn).unwrap(), 1);
    assert_eq!(rnn.init_calls.get(), 0);
    assert_eq!(rnn.forward_calls.get(), 0);
    let done = s.take_finished();
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].0, id);
    assert_eq!(vals(&done[0].1), vec![1, 2, 3]);
}

#[test]
fn max_active_is_respected() {
    let rnn = Rnn::new();
    let mut s = BatchScheduler::new(limits(1, 2));
    s.submit(&ids1(&[0]), &greedy(5)).unwrap();
    s.submit(&ids1(&[1]), &greedy(5)).unwrap();
    // 待ち行列の上限 (2) を超える submit は従来どおり Err。
    assert!(s.submit(&ids1(&[2]), &greedy(5)).is_err());
    s.step_stateful(&rnn).unwrap();
    assert_eq!(s.active_len(), 1);
    assert_eq!(s.queued_len(), 1);
    assert_eq!(rnn.init_calls.get(), 1);
}

#[test]
fn debug_does_not_expose_state_or_tokens() {
    let rnn = Rnn::new();
    let mut s = BatchScheduler::new(limits(2, 4));
    s.submit(&ids1(&[1, 2]), &greedy(6)).unwrap();
    s.step_stateful(&rnn).unwrap();
    assert_eq!(s.active_len(), 1);
    let dbg = format!("{s:?}");
    assert!(!dbg.contains(SECRET), "{dbg}");
    assert!(!dbg.contains("Hidden"), "{dbg}");
    assert!(dbg.contains("active"), "{dbg}");
}
