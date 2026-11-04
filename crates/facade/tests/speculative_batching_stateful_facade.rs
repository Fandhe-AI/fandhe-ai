//! facade の `fandhe_ai::inference` 公開面だけで、内部状態を持つモデルの speculative decoding と
//! 連続バッチングを通す受け入れテスト（イシュー #3849・親 #3719。設計記録
//! `docs/facade-speculative-decoding-batching-design.md` §29・§31）。
//!
//! 内部クレート（`fandhe_ai_autodiff`）を `use` しない。公開パスで書けること自体が受け入れ条件で、
//! 広い掃引は autodiff 側の `generate_scheduler_stateful.rs`・`nn_generate_speculative_stateful.rs` が担う。
//! 実機（CUDA／Metal）依存のテストは無い（新しいカーネル・バックエンド経路が無い。設計 §29.4）。

use fandhe_ai::inference::{
    AutoregressiveModel, BatchScheduler, GenerateConfig, SamplingStrategy, SchedulerLimits,
    SpeculativeConfig, StatefulAutoregressiveModel, generate, generate_speculative_stateful,
};
use fandhe_ai::nn::kv_cache::KvCache;
use fandhe_ai::{AutodiffError, Tensor};
use std::cell::{Cell, RefCell};

const V: usize = 7;

fn next_tok(tok: i32, pos: usize) -> i32 {
    ((3 * tok as usize + pos + 1) % V) as i32
}

fn one_hot(toks: &[i32], pos0: usize) -> Tensor<f32> {
    let mut data = vec![0.0f32; toks.len() * V];
    for (i, &tok) in toks.iter().enumerate() {
        data[i * V + next_tok(tok, pos0 + i) as usize] = 10.0;
    }
    Tensor::new(data, &[1, toks.len(), V]).unwrap()
}

fn vals(x: &Tensor<i32>) -> Vec<i32> {
    x.contiguous().host_slice().into_owned()
}

fn prompt(v: &[i32]) -> Tensor<i32> {
    Tensor::new(v.to_vec(), &[v.len()]).unwrap()
}

/// 状態が処理済みトークン数だけのモデル。`init_calls` で呼び出し回数を数える。
struct Rnn {
    init_calls: Cell<usize>,
    fail_init: bool,
}

impl Rnn {
    fn new() -> Rnn {
        Rnn {
            init_calls: Cell::new(0),
            fail_init: false,
        }
    }
}

impl StatefulAutoregressiveModel for Rnn {
    type State = usize;

    fn init_state(&self) -> Result<usize, AutodiffError> {
        self.init_calls.set(self.init_calls.get() + 1);
        if self.fail_init {
            return Err(AutodiffError::InvalidArgument("init".to_string()));
        }
        Ok(0)
    }

    fn forward_step_stateful(
        &self,
        new_ids: &Tensor<i32>,
        state: &mut usize,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let toks = vals(new_ids);
        let out = one_hot(&toks, *state);
        *state += toks.len();
        Ok(out)
    }

    fn save_state(&self, state: &usize) -> Result<usize, AutodiffError> {
        Ok(*state)
    }

    fn restore_state(&self, state: &mut usize, saved: usize) -> Result<(), AutodiffError> {
        *state = saved;
        Ok(())
    }
}

/// 同じ規則の参照（`RefCell` に状態を持つ `AutoregressiveModel`）。
struct RefRnn(RefCell<usize>);

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
        let mut pos = self.0.borrow_mut();
        let out = one_hot(&toks, *pos);
        *pos += toks.len();
        Ok(out)
    }
}

fn reference(p: &Tensor<i32>, cfg: &GenerateConfig) -> Vec<i32> {
    vals(&generate(&RefRnn(RefCell::new(0)), p, cfg).unwrap())
}

#[test]
fn speculative_stateful_matches_reference_through_facade_path() {
    let cfg = GenerateConfig::new(9, SamplingStrategy::Greedy);
    for k in [1usize, 2, 4] {
        let p = prompt(&[1, 2]);
        let out = generate_speculative_stateful(
            &Rnn::new(),
            &Rnn::new(),
            &p,
            &cfg,
            &SpeculativeConfig::new(k),
        )
        .unwrap();
        assert_eq!(vals(&out), reference(&p, &cfg), "k={k}");
    }
}

#[test]
fn speculative_stateful_rejects_invalid_inputs() {
    let spec = SpeculativeConfig::new(2);
    let cfg = GenerateConfig::new(5, SamplingStrategy::Greedy);
    // k = 0 は拒否される。
    let r = generate_speculative_stateful(
        &Rnn::new(),
        &Rnn::new(),
        &prompt(&[0]),
        &cfg,
        &SpeculativeConfig::new(0),
    );
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))));
    // 空 prompt は拒否される。
    let empty = Tensor::<i32>::new(vec![], &[0]).unwrap();
    let r = generate_speculative_stateful(&Rnn::new(), &Rnn::new(), &empty, &cfg, &spec);
    assert!(r.is_err());
    // max_length < prompt 長は拒否される。
    let r = generate_speculative_stateful(
        &Rnn::new(),
        &Rnn::new(),
        &prompt(&[0, 1, 2]),
        &GenerateConfig::new(2, SamplingStrategy::Greedy),
        &spec,
    );
    assert!(r.is_err());
    // init_state の失敗は伝播する。
    let mut bad = Rnn::new();
    bad.fail_init = true;
    let r = generate_speculative_stateful(&bad, &Rnn::new(), &prompt(&[0]), &cfg, &spec);
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn step_stateful_matches_reference_and_isolates_failures() {
    let cfg = GenerateConfig::new(6, SamplingStrategy::Greedy);
    let rnn = Rnn::new();
    let mut s = BatchScheduler::new(SchedulerLimits::new(2, 4, 8).unwrap());
    let p0 = prompt(&[0]);
    let p1 = prompt(&[2, 3]);
    let i0 = s.submit(&p0, &cfg).unwrap();
    let i1 = s.submit(&p1, &cfg).unwrap();
    let mut finished = Vec::new();
    for _ in 0..16 {
        s.step_stateful(&rnn).unwrap();
        finished.extend(s.take_finished());
        if s.queued_len() + s.active_len() == 0 {
            break;
        }
    }
    assert!(s.take_failed().is_empty());
    assert_eq!(finished.len(), 2);
    for (id, out) in &finished {
        let p = if *id == i0 { &p0 } else { &p1 };
        assert_eq!(vals(out), reference(p, &cfg));
    }
    assert_eq!(rnn.init_calls.get(), 2);
    let _ = i1;

    // 種類の取り違え: 内部状態で参加した要求を `step` で進めると、その要求だけが失敗する。
    let mut s = BatchScheduler::new(SchedulerLimits::new(2, 4, 8).unwrap());
    let id = s.submit(&p0, &cfg).unwrap();
    s.step_stateful(&Rnn::new()).unwrap();
    s.step(&RefRnn(RefCell::new(0))).unwrap();
    let failed = s.take_failed();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].0, id);

    // init_state の失敗は要求単位の失敗になる。
    let mut bad = Rnn::new();
    bad.fail_init = true;
    let mut s = BatchScheduler::new(SchedulerLimits::new(2, 4, 8).unwrap());
    let id = s.submit(&p0, &cfg).unwrap();
    s.step_stateful(&bad).unwrap();
    let failed = s.take_failed();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].0, id);
    assert_eq!(s.active_len(), 0);
}
