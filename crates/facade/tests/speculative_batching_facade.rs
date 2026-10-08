//! facade `fandhe_ai::inference` 経由の speculative decoding（`generate_speculative`）と連続バッチング
//! スケジューラ（`BatchScheduler`）の結合テスト（イシュー #2934・親 #2932。設計記録
//! `docs/facade-speculative-decoding-batching-design.md` §6・§17・§18）。
//!
//! 公開 API（`generate_speculative`・`SpeculativeConfig`・`BatchScheduler`・`RequestId`・`SchedulerLimits`）は
//! すべて `fandhe_ai::inference::…` のパスで呼ぶ。承認根拠はリポジトリ所有者本人の承認コメント
//! issuecomment-6067263650 の項 1。KV キャッシュを進めるモデルは facade の公開面だけでは組めない
//! （`forward_with_cache` が非公開。保留論点 3）ため、フィクスチャに限り内部クレート
//! `fandhe_ai_autodiff::nn::{MultiheadAttentionVars, LinearVars}`（`forward_with_cache`）を使う
//! （`speculative_batching_backend_parity.rs`・`generate_backend_parity.rs` と同じ扱い。テスト限定）。
//!
//! # 判定
//!
//! 形状に依らない logits を返す「表引き KV モデル」（MHA で `caches` を進めつつ、logits は
//! 直前トークンと位置だけで決まる one-hot）を使うため、token 列は単独 `generate` と完全一致を要求する。
//! 数値比較は無く、REQ-2 統一複合判定・tolerance・baseline は新設しない。実 MHA の logits を使う
//! token 列一致の仮説（H1）と CPU 対 CUDA／Metal の parity は `speculative_batching_backend_parity.rs`
//! （属性なし分は CI、`#[ignore]` 分は実機）が担う。facade の再エクスポートはデバイス挙動を変えないため、
//! 本ファイルに新しい `#[ignore]` テストは足さない（実機は未測定。
//! `docs/perf/logs/speculative-batching-facade-2934/README.md` に申し送り）。
//!
//! 乱数は独立した `Generator`（xorshift64*）のみで、決定的 seed を使う（非暗号。OWASP A02）。

use fandhe_ai::inference::{
    AutoregressiveModel, BatchScheduler, GenerateConfig, RequestId, SamplingStrategy,
    SchedulerLimits, SpeculativeConfig, generate, generate_speculative,
};
use fandhe_ai::nn::kv_cache::KvCache;
use fandhe_ai::{AutodiffError, Tensor};
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::{LinearVars, MultiheadAttentionVars};

const V: usize = 7;
const E: usize = 4;
const HEADS: usize = 2;
/// スケジューラ駆動ループの上限（ハング防止の fail-closed）。
const TICK_CAP: usize = 256;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture: shape とデータ長は事前に一致させている")
}

fn seq(seed: i64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((seed + i as i64 * 7) % 13 - 6) as f32 * 0.05)
        .collect()
}

fn ids(v: &[i32], rank1: bool) -> Tensor<i32> {
    let shape: Vec<usize> = if rank1 {
        vec![v.len()]
    } else {
        vec![1, v.len()]
    };
    Tensor::new(v.to_vec(), &shape).expect("fixture")
}

fn toks(x: &Tensor<i32>) -> Vec<i32> {
    x.contiguous().host_slice().into_owned()
}

fn greedy(max_length: usize) -> GenerateConfig {
    GenerateConfig::new(max_length, SamplingStrategy::Greedy)
}

/// 表引きモデル。`layers == 0` は状態なし（位置は常に 0）、`layers > 0` は MHA で `caches` を進める
/// 表引き KV モデル（`caches[0].seq_len()` を位置として使う。巻き戻し後の再生が位置を狂わせれば
/// 出力が単独 `generate` からずれる）。logits は `next(tok, pos)` の one-hot で、形状に依存しない。
struct TableLm {
    layers: usize,
    /// 次トークン規則の係数（target と draft で変えて棄却を起こす）。
    mul: usize,
}

impl TableLm {
    fn next(&self, tok: i32, pos: usize) -> usize {
        (self.mul * tok as usize + pos + 1) % V
    }
}

impl AutoregressiveModel for TableLm {
    fn num_kv_layers(&self) -> usize {
        self.layers
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let pos0 = caches.first().map_or(0, |c| c.seq_len());
        if !caches.is_empty() {
            let tape = Tape::new();
            let emb = tape.var(&t(seq(1, V * E), &[V, E]));
            let mut x = emb.embedding(new_ids, None)?;
            for (li, cache) in caches.iter_mut().enumerate() {
                let o = 20 * li as i64;
                let lin = |d: i64| LinearVars {
                    weight: tape.var(&t(seq(o + d, E * E), &[E, E])),
                    bias: Some(tape.var(&t(seq(o + d + 1, E), &[E]))),
                };
                let mha = MultiheadAttentionVars::new(HEADS, lin(2), lin(4), lin(6), lin(8))?;
                x = mha.forward_with_cache(&x, &x, &x, cache)?;
            }
        }
        let l = new_ids.shape()[new_ids.shape().len() - 1];
        let mut data = vec![0.0f32; l * V];
        for (i, &tok) in toks(new_ids).iter().enumerate() {
            data[i * V + self.next(tok, pos0 + i)] = 1.0;
        }
        Ok(t(data, &[1, l, V]))
    }
}

/// prompt に `poison` を含む要求だけ `forward_step` が `Err` を返すモデル（要求単位の失敗の切り離し検査用）。
struct PoisonLm {
    inner: TableLm,
    poison: i32,
}

impl AutoregressiveModel for PoisonLm {
    fn num_kv_layers(&self) -> usize {
        self.inner.num_kv_layers()
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        // prefill（キャッシュ長 0）の入力だけを見る（生成 token に poison が現れても他要求を巻き込まない）。
        let prefill = caches.first().is_some_and(|c| c.seq_len() == 0);
        if prefill && toks(new_ids).contains(&self.poison) {
            return Err(AutodiffError::InvalidArgument("poison".to_string()));
        }
        self.inner.forward_step(new_ids, caches)
    }
}

/// `scheduler` を完了まで駆動し、完了した `(RequestId, 出力)` を返す（上限付きループ）。
fn drive<M: AutoregressiveModel + ?Sized>(
    scheduler: &mut BatchScheduler,
    model: &M,
) -> Vec<(RequestId, Tensor<i32>)> {
    let mut finished = Vec::new();
    for _ in 0..TICK_CAP {
        scheduler.step(model).expect("step");
        finished.extend(scheduler.take_finished());
        if scheduler.queued_len() + scheduler.active_len() == 0 {
            return finished;
        }
    }
    panic!("スケジューラが {TICK_CAP} step 以内に完了しない");
}

/// 3 要求（Greedy／TopK／Temperature。seed は別々）を交互に進め、各出力が単独 `generate` と
/// token 列で完全一致することを確かめる（設計 §6.3）。
fn check_scheduler_matches_alone(model: &TableLm) {
    let cfgs = [
        greedy(6),
        GenerateConfig::new(7, SamplingStrategy::TopK(3)).with_seed(11),
        GenerateConfig::new(5, SamplingStrategy::Temperature(0.7)).with_seed(23),
    ];
    let prompts = [vec![0], vec![2, 3], vec![4]];
    let limits = SchedulerLimits::new(2, 4, 8).expect("limits");
    let mut scheduler = BatchScheduler::new(limits);
    let mut submitted = Vec::new();
    for (p, c) in prompts.iter().zip(cfgs.iter()) {
        let id = scheduler.submit(&ids(p, true), c).expect("submit");
        submitted.push(id);
    }
    // 3 要求 > max_active(2)。待ち行列からの参加が起きる。
    assert_eq!(scheduler.queued_len(), 3);
    let finished = drive(&mut scheduler, model);
    assert!(scheduler.take_failed().is_empty());
    assert_eq!(finished.len(), 3);
    for (i, id) in submitted.iter().enumerate() {
        let (_, out) = finished
            .iter()
            .find(|(fid, _)| fid == id)
            .expect("完了に含まれる");
        let alone = generate(model, &ids(&prompts[i], true), &cfgs[i]).expect("単独 generate");
        assert_eq!(out.shape(), alone.shape(), "要求 {i} の shape");
        assert_eq!(toks(out), toks(&alone), "要求 {i} の token 列");
    }
}

#[test]
fn scheduler_matches_standalone_generate_for_stateless_model() {
    check_scheduler_matches_alone(&TableLm { layers: 0, mul: 3 });
}

#[test]
fn scheduler_matches_standalone_generate_for_kv_model() {
    check_scheduler_matches_alone(&TableLm { layers: 1, mul: 3 });
    check_scheduler_matches_alone(&TableLm { layers: 2, mul: 3 });
}

#[test]
fn scheduler_fail_closed_on_limits_and_bad_requests() {
    // 各上限の 0 は拒否（暗黙の無制限を作らない）。
    for (a, q, m) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
        assert!(
            matches!(
                SchedulerLimits::new(a, q, m),
                Err(AutodiffError::InvalidArgument(_))
            ),
            "({a},{q},{m})"
        );
    }
    let limits = SchedulerLimits::new(1, 2, 4).expect("limits");
    assert_eq!(
        (
            limits.max_active(),
            limits.max_queued(),
            limits.max_length()
        ),
        (1, 2, 4)
    );
    let mut s = BatchScheduler::new(limits);
    // max_length 超過・B = 2 は submit で拒否され、状態は変わらない。
    assert!(matches!(
        s.submit(&ids(&[0], true), &greedy(5)),
        Err(AutodiffError::InvalidArgument(_))
    ));
    let b2 = Tensor::<i32>::new(vec![0, 1], &[2, 1]).expect("fixture");
    assert!(matches!(
        s.submit(&b2, &greedy(3)),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert_eq!(s.queued_len(), 0);
    // 待ち行列長の超過。
    s.submit(&ids(&[0], true), &greedy(3)).expect("1 件目");
    s.submit(&ids(&[1], true), &greedy(3)).expect("2 件目");
    assert!(matches!(
        s.submit(&ids(&[2], true), &greedy(3)),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert_eq!(s.queued_len(), 2);
}

#[test]
fn scheduler_isolates_per_request_failure() {
    let model = PoisonLm {
        inner: TableLm { layers: 1, mul: 3 },
        poison: 5,
    };
    let limits = SchedulerLimits::new(3, 4, 8).expect("limits");
    let mut s = BatchScheduler::new(limits);
    let good = [vec![0], vec![2, 3]];
    let a = s.submit(&ids(&good[0], true), &greedy(6)).expect("a");
    let bad = s.submit(&ids(&[5], true), &greedy(6)).expect("bad");
    let c = s.submit(&ids(&good[1], true), &greedy(6)).expect("c");
    let finished = drive(&mut s, &model);
    let failed = s.take_failed();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].0, bad);
    assert_eq!(finished.len(), 2);
    for (id, prompt) in [(a, &good[0]), (c, &good[1])] {
        let (_, out) = finished.iter().find(|(f, _)| *f == id).expect("完了");
        let alone = generate(&model, &ids(prompt, true), &greedy(6)).expect("単独");
        assert_eq!(toks(out), toks(&alone), "失敗要求の影響を受けない");
    }
}

/// speculative 成功経路: 出力が Greedy の単独 `generate` と完全一致する（形状に依らない logits の
/// 表引き KV モデルなので完全一致を要求する。設計 §6.1）。
#[test]
fn speculative_matches_greedy_generate_on_kv_table_model() {
    let target = TableLm { layers: 1, mul: 3 };
    // 別重み（別層数）で、規則が target と同じ全受理の draft と、規則が違う（棄却が起きる）draft。
    let drafts = [TableLm { layers: 2, mul: 3 }, TableLm { layers: 1, mul: 2 }];
    let mut runs = 0usize;
    for rank1 in [true, false] {
        for prompt_len in [1usize, 3] {
            let prompt: Vec<i32> = (0..prompt_len).map(|i| (i % V) as i32).collect();
            for max in [prompt_len, prompt_len + 1, prompt_len + 2, 10] {
                let alone = generate(&target, &ids(&prompt, rank1), &greedy(max)).expect("alone");
                for k in [1usize, 2, 4] {
                    for draft in &drafts {
                        let out = generate_speculative(
                            &target,
                            draft,
                            &ids(&prompt, rank1),
                            &greedy(max),
                            &SpeculativeConfig::new(k),
                        )
                        .unwrap_or_else(|e| {
                            panic!("rank1={rank1} T={prompt_len} max={max} k={k}: {e:?}")
                        });
                        assert_eq!(out.shape(), alone.shape());
                        assert_eq!(
                            toks(&out),
                            toks(&alone),
                            "rank1={rank1} T={prompt_len} max={max} k={k}"
                        );
                        runs += 1;
                    }
                }
            }
        }
    }
    assert!(runs > 0);
    // draft == target（同一インスタンス）でも動作する。
    let prompt = ids(&[1, 2], true);
    let out = generate_speculative(
        &target,
        &target,
        &prompt,
        &greedy(8),
        &SpeculativeConfig::new(3),
    )
    .expect("同一インスタンス");
    assert_eq!(
        toks(&out),
        toks(&generate(&target, &prompt, &greedy(8)).expect("alone"))
    );
}

#[test]
fn speculative_fails_closed_on_unsupported_or_invalid_inputs() {
    let kv = TableLm { layers: 1, mul: 3 };
    let stateless = TableLm { layers: 0, mul: 3 };
    let p = ids(&[0, 1], true);
    let spec = SpeculativeConfig::new(2);
    let invalid = |r: Result<Tensor<i32>, AutodiffError>, what: &str| {
        assert!(
            matches!(r, Err(AutodiffError::InvalidArgument(_))),
            "{what}: {r:?}"
        );
    };
    // サンプリング版は存在しない（Greedy 以外は Err）。
    invalid(
        generate_speculative(
            &kv,
            &kv,
            &p,
            &GenerateConfig::new(6, SamplingStrategy::TopK(3)),
            &spec,
        ),
        "TopK",
    );
    invalid(
        generate_speculative(
            &kv,
            &kv,
            &p,
            &GenerateConfig::new(6, SamplingStrategy::Temperature(0.8)),
            &spec,
        ),
        "Temperature",
    );
    invalid(
        generate_speculative(&kv, &kv, &p, &greedy(6), &SpeculativeConfig::new(0)),
        "k == 0",
    );
    let b2 = Tensor::<i32>::new(vec![0, 1, 2, 3], &[2, 2]).expect("fixture");
    invalid(
        generate_speculative(&kv, &kv, &b2, &greedy(6), &spec),
        "B = 2",
    );
    invalid(
        generate_speculative(&stateless, &kv, &p, &greedy(6), &spec),
        "target が num_kv_layers == 0",
    );
    invalid(
        generate_speculative(&kv, &stateless, &p, &greedy(6), &spec),
        "draft が num_kv_layers == 0",
    );
    let empty = Tensor::<i32>::new(vec![], &[0]).expect("fixture");
    invalid(
        generate_speculative(&kv, &kv, &empty, &greedy(6), &spec),
        "空 prompt",
    );
    invalid(
        generate_speculative(&kv, &kv, &p, &greedy(1), &spec),
        "max_length < T",
    );
}
