//! facade `fandhe_ai::inference` 経由の speculative decoding（`generate_speculative`）と連続バッチング
//! スケジューラ（`BatchScheduler`）の結合テスト（イシュー #2934・親 #2932。設計記録
//! `docs/facade-speculative-decoding-batching-design.md` §6・§17・§18）。
//!
//! 公開 API（`generate_speculative`・`SpeculativeConfig`・`BatchScheduler`・`RequestId`・`SchedulerLimits`）は
//! すべて `fandhe_ai::inference::…` のパスで呼ぶ。承認根拠はリポジトリ所有者本人の承認コメント
//! issuecomment-6067263650 の項 1。KV キャッシュを進めるモデルは、イシュー #3097（親 #3094）以降は
//! facade の公開面だけで組む: `Tape::stateful_attention_forward_with_cache`（#3096）で
//! `forward_step` の `caches[i]` を進める形にする（`num_kv_layers()` は層数。設計記録 §19.6）。
//! 本ファイルは内部クレート（autodiff）を一切 import しない（完了の検証シグナル）。
//! 数値一致の判定ヘルパー（REQ-2 定数・`assert_parity`）だけは `fandhe_ai_backend_cpu::parity` を
//! テスト専用に使う（`generate_facade.rs`・`kv_cache_facade.rs` と同じ扱い）。
//!
//! # 判定
//!
//! 形状に依らない logits を返す「表引き KV モデル」（MHA で `caches` を進めつつ、logits は
//! 直前トークンと位置だけで決まる one-hot）を使うため、token 列は単独 `generate` と完全一致を要求する。
//! 表引き側に数値比較は無い。attention 依存 logits の KV モデル（`AttnCachesLm`）は位置ごとの logits を
//! REQ-2 統一複合判定（`assert_parity`。tolerance・baseline は新設しない）で単独 `generate` と突き合わせ、
//! token 列一致の仮説（H1。margin 前提下）も確かめる。CPU 対 CUDA／Metal の parity は
//! `speculative_batching_backend_parity.rs`（属性なし分は CI、`#[ignore]` 分は実機）が担う。facade の再エクスポートはデバイス挙動を変えないため、
//! 本ファイルに新しい `#[ignore]` テストは足さない（実機は未測定。
//! `docs/perf/logs/speculative-batching-facade-2934/README.md` に申し送り）。
//!
//! 乱数は独立した `Generator`（xorshift64*）のみで、決定的 seed を使う（非暗号。OWASP A02）。

use fandhe_ai::inference::{
    AutoregressiveModel, BatchScheduler, GenerateConfig, RequestId, SamplingStrategy,
    SchedulerLimits, SpeculativeConfig, generate, generate_speculative,
};
use fandhe_ai::nn::kv_cache::{KvCache, MultiheadAttentionConfig, StatefulAttention};
use fandhe_ai::{AutodiffError, LinearVars, Tensor};
use fandhe_ai_backend_cpu::parity::{ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE, assert_parity};
use std::cell::RefCell;

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

/// `layers` 層分の `StatefulAttention` を作る（層 `li` の seed は `seed_base + li`）。状態は持たず、
/// `forward_step` では `caches[li]` だけを進める（`sa` 自身の内部キャッシュは読み書きしない。設計 §19.5）。
fn attention_layers(layers: usize, seed_base: u64) -> Vec<StatefulAttention> {
    (0..layers)
        .map(|li| {
            StatefulAttention::from_config(
                &MultiheadAttentionConfig::new(E, HEADS),
                seed_base + li as u64,
            )
            .expect("fixture: from_config")
        })
        .collect()
}

/// 表引きモデル。`layers == 0` は状態なし（位置は常に 0）、`layers > 0` は facade の
/// `stateful_attention_forward_with_cache` で `caches` を進める表引き KV モデル（`caches[0].seq_len()` を
/// 位置として使う。巻き戻し後の再生が位置を狂わせれば出力が単独 `generate` からずれる）。
/// logits は `next(tok, pos)` の one-hot で、形状に依存しない。`sas` は `&self` 入口でのみ使うため、
/// 1 つのモデルを複数要求で共有しても状態は混ざらない。
struct TableLm {
    sas: Vec<StatefulAttention>,
    /// 次トークン規則の係数（target と draft で変えて棄却を起こす）。
    mul: usize,
}

impl TableLm {
    fn new(layers: usize, mul: usize) -> Self {
        Self {
            sas: attention_layers(layers, 1),
            mul,
        }
    }

    fn next(&self, tok: i32, pos: usize) -> usize {
        (self.mul * tok as usize + pos + 1) % V
    }
}

impl AutoregressiveModel for TableLm {
    fn num_kv_layers(&self) -> usize {
        self.sas.len()
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let pos0 = caches.first().map_or(0, |c| c.seq_len());
        if !caches.is_empty() {
            let tape = fandhe_ai::tape();
            let emb = tape.var(&t(seq(1, V * E), &[V, E]));
            let mut x = emb.embedding(new_ids, None)?;
            for (sa, cache) in self.sas.iter().zip(caches.iter_mut()) {
                x = tape.stateful_attention_forward_with_cache(sa, &x, cache)?;
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
    check_scheduler_matches_alone(&TableLm::new(0, 3));
}

#[test]
fn scheduler_matches_standalone_generate_for_kv_model() {
    check_scheduler_matches_alone(&TableLm::new(1, 3));
    check_scheduler_matches_alone(&TableLm::new(2, 3));
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
        inner: TableLm::new(1, 3),
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
    let target = TableLm::new(1, 3);
    // 別重み（別層数）で、規則が target と同じ全受理の draft と、規則が違う（棄却が起きる）draft。
    let drafts = [TableLm::new(2, 3), TableLm::new(1, 2)];
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
    let kv = TableLm::new(1, 3);
    let stateless = TableLm::new(0, 3);
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
// ---------------------------------------------------------------------
// イシュー #3097: `caches` 形モデルの交互進行一致と、attention 依存 KV モデルの speculative 一致
// ---------------------------------------------------------------------

/// margin 確保用 skip 項の大きさ。上位 2 候補の差を attention 経路の数値揺らぎより十分大きく取る
/// フィクスチャ定数（判定閾値ではない）。結果を見て調整しない（H1 の事前登録。
/// `crates/autodiff/tests/nn_generate_speculative.rs` の `SKIP_C` と同値）。
const SKIP_C: f32 = 3.0;

type NextFn = Box<dyn Fn(i32, usize) -> i32>;

fn target_next(tok: i32, pos: usize) -> i32 {
    ((3 * tok as usize + pos + 1) % V) as i32
}

fn wrong_next(tok: i32, pos: usize) -> i32 {
    (target_next(tok, pos) + 1) % V as i32
}

/// 位置が `period` の倍数のときだけ外す draft 規則（部分受理を起こす）。
fn periodic_next(period: usize) -> NextFn {
    Box::new(move |tok, pos| {
        if pos.is_multiple_of(period) {
            wrong_next(tok, pos)
        } else {
            target_next(tok, pos)
        }
    })
}

/// 埋め込み → `stateful_attention_forward_with_cache`（層ごと直列）→ lm head の attention 依存 KV モデル。
/// margin 確保のため `SKIP_C · onehot(next(tok, pos))` をホスト側で足す。logits は KV 経路の数値に依存する。
struct AttnCachesLm {
    sas: Vec<StatefulAttention>,
    seed_base: i64,
    next: NextFn,
}

impl AttnCachesLm {
    fn new(layers: usize, seed_base: i64, next: NextFn) -> Self {
        Self {
            sas: attention_layers(layers, 500 + seed_base as u64),
            seed_base,
            next,
        }
    }
}

impl AutoregressiveModel for AttnCachesLm {
    fn num_kv_layers(&self) -> usize {
        self.sas.len()
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let pos0 = caches.first().map_or(0, |c| c.seq_len());
        let s = self.seed_base;
        let tape = fandhe_ai::tape();
        let emb = tape.var(&t(seq(s + 1, V * E), &[V, E]));
        let mut x = emb.embedding(new_ids, None)?;
        for (sa, cache) in self.sas.iter().zip(caches.iter_mut()) {
            x = tape.stateful_attention_forward_with_cache(sa, &x, cache)?;
        }
        let l = new_ids.shape()[new_ids.shape().len() - 1];
        let flat = x.reshape(&[l, E])?;
        let lm = LinearVars {
            weight: tape.var(&t(seq(s + 10, E * V), &[E, V])),
            bias: Some(tape.var(&t(seq(s + 9, V), &[V]))),
        };
        let logits = lm.forward(&flat)?.reshape(&[1, l, V])?.to_tensor();
        let mut data = logits.contiguous().host_slice().into_owned();
        for (i, &tok) in toks(new_ids).iter().enumerate() {
            data[i * V + (self.next)(tok, pos0 + i) as usize] += SKIP_C;
        }
        Ok(t(data, &[1, l, V]))
    }
}

/// 1 回の `forward_step` の記録（`seq_before` は呼び出し前の `caches[0].seq_len()`）。
struct Call {
    seq_before: usize,
    ids: Vec<i32>,
    logits: Vec<f32>,
}

/// 任意モデルを包み、各 `forward_step` の入出力を記録する（位置ごとの logits 突き合わせ用）。
struct Recording<M> {
    inner: M,
    log: RefCell<Vec<Call>>,
}

impl<M> Recording<M> {
    fn new(inner: M) -> Self {
        Self {
            inner,
            log: RefCell::new(Vec::new()),
        }
    }
}

impl<M: AutoregressiveModel> AutoregressiveModel for Recording<M> {
    fn num_kv_layers(&self) -> usize {
        self.inner.num_kv_layers()
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let seq_before = caches[0].seq_len();
        let logits = self.inner.forward_step(new_ids, caches)?;
        self.log.borrow_mut().push(Call {
            seq_before,
            ids: toks(new_ids),
            logits: logits.contiguous().host_slice().into_owned(),
        });
        Ok(logits)
    }
}

/// 参照（単独 `generate`）の記録から絶対位置ごとの logits 行を作る。重複・欠落は panic（fail-closed）。
fn reference_rows(log: &[Call]) -> Vec<Vec<f32>> {
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for call in log {
        for (i, row) in call.logits.chunks(V).enumerate() {
            assert_eq!(call.seq_before + i, rows.len(), "参照の位置が連続しない");
            rows.push(row.to_vec());
        }
    }
    rows
}

/// speculative の target 記録から、prefix が最終出力と一致する行だけを `(位置, 行, L_new)` で返す。
fn comparable_rows(log: &[Call], output: &[i32]) -> Vec<(usize, Vec<f32>, usize)> {
    let mut out = Vec::new();
    for call in log {
        for (i, row) in call.logits.chunks(V).enumerate() {
            let p = call.seq_before + i;
            if p < output.len() && output[call.seq_before..=p] == call.ids[..=i] {
                out.push((p, row.to_vec(), call.ids.len()));
            }
        }
    }
    out
}

/// 直前の呼び出しと同じ `seq_before` から始まる呼び出し（= 巻き戻し後の再 forward）の回数。
fn replay_count(log: &[Call]) -> usize {
    log.windows(2)
        .filter(|w| w[1].seq_before == w[0].seq_before)
        .count()
}

/// 入力の妥当性検査（新しい tolerance ではない）。参照のトークン選択位置 `T-1 ..= max-2` で
/// `top1 - top2 > 2·(rtol·M/(1-rtol) ∨ abs)` を要求する（REQ-2 定数から導いた帰結。
/// 導出は `crates/autodiff/tests/nn_generate_speculative.rs::assert_margin`）。
fn assert_margin(label: &str, ref_rows: &[Vec<f32>], prompt_len: usize, max_length: usize) {
    for (p, row) in ref_rows
        .iter()
        .enumerate()
        .take(max_length - 1)
        .skip(prompt_len - 1)
    {
        let mut sorted = row.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).expect("fixture: logits は有限"));
        let m = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs())) as f64;
        let rtol = RELATIVE_TOLERANCE;
        let bound = (rtol * m / (1.0 - rtol)).max(ABSOLUTE_RESCUE_THRESHOLD);
        let margin = f64::from(sorted[0] - sorted[1]);
        assert!(
            margin > 2.0 * bound,
            "{label}: フィクスチャ不正（margin 前提を満たさない）位置 {p}: margin={margin}"
        );
    }
}

/// 交互進行（設計 §19.6）: `caches` を進める形に書き直した KV 付きモデルを 1 つ共有し、
/// `max_active` 未満の枠で 3 要求を交互に進めても各出力が単独 `generate` と token 列で完全一致する。
/// attention 依存 logits のモデルでも、スケジューラ経路は単独と同じ forward 形状（B=1・prefill→L_new=1）
/// なので完全一致を要求する。完了後、モデルの `StatefulAttention` 内部キャッシュは未使用（長さ 0）のまま。
#[test]
fn scheduler_interleaves_caches_models_like_standalone_generate() {
    let cfgs = [
        greedy(6),
        GenerateConfig::new(7, SamplingStrategy::TopK(3)).with_seed(11),
        GenerateConfig::new(5, SamplingStrategy::Temperature(0.7)).with_seed(23),
    ];
    let prompts = [vec![0], vec![2, 3], vec![4]];
    let models: Vec<(&str, Box<dyn AutoregressiveModel>)> = vec![
        ("table1", Box::new(TableLm::new(1, 3))),
        ("table2", Box::new(TableLm::new(2, 3))),
        (
            "attn1",
            Box::new(AttnCachesLm::new(1, 100, Box::new(target_next))),
        ),
        (
            "attn2",
            Box::new(AttnCachesLm::new(2, 100, Box::new(target_next))),
        ),
    ];
    for (name, model) in &models {
        assert!(model.num_kv_layers() > 0, "{name}");
        let limits = SchedulerLimits::new(2, 4, 8).expect("limits");
        let mut scheduler = BatchScheduler::new(limits);
        let mut submitted = Vec::new();
        for (p, c) in prompts.iter().zip(cfgs.iter()) {
            submitted.push(scheduler.submit(&ids(p, true), c).expect("submit"));
        }
        assert_eq!(scheduler.queued_len(), 3, "{name}: 3 要求 > max_active(2)");
        let finished = drive(&mut scheduler, model.as_ref());
        assert!(scheduler.take_failed().is_empty(), "{name}");
        assert_eq!(finished.len(), 3, "{name}");
        for (i, id) in submitted.iter().enumerate() {
            let (_, out) = finished
                .iter()
                .find(|(fid, _)| fid == id)
                .expect("完了に含まれる");
            let alone = generate(model.as_ref(), &ids(&prompts[i], true), &cfgs[i]).expect("単独");
            assert_eq!(out.shape(), alone.shape(), "{name}: 要求 {i} の shape");
            assert_eq!(toks(out), toks(&alone), "{name}: 要求 {i} の token 列");
        }
    }
    // `sa` の内部キャッシュは読みも書きもされない（設計 §19.5）。
    let model = TableLm::new(2, 3);
    let mut scheduler = BatchScheduler::new(SchedulerLimits::new(2, 4, 8).expect("limits"));
    for (p, c) in prompts.iter().zip(cfgs.iter()) {
        scheduler.submit(&ids(p, true), c).expect("submit");
    }
    drive(&mut scheduler, &model);
    assert!(model.sas.iter().all(|sa| sa.seq_len() == 0));
}

/// speculative（Greedy）: attention 依存 logits の KV モデルでも、位置ごとの logits が単独 `generate` と
/// REQ-2 統一複合判定（tolerance 不変）で一致し、仮説 H1（margin 前提下の token 列一致）を確かめる。
/// H1 は契約ではなく仮説で、破れた場合は判定を緩めず設計 §10 論点 2 として報告する。
#[test]
fn speculative_matches_generate_on_attention_caches_model() {
    let (mut runs, mut verify_rows, mut replays) = (0usize, 0usize, 0usize);
    for layers in [1usize, 2] {
        for k in [1usize, 3] {
            for prompt_len in [1usize, 3] {
                for max in [prompt_len + 1, prompt_len + k + 2, 12] {
                    let prompt: Vec<i32> = (0..prompt_len).map(|i| (i % V) as i32).collect();
                    let p = ids(&prompt, false);
                    let ref_model =
                        Recording::new(AttnCachesLm::new(layers, 100, Box::new(target_next)));
                    let reference = generate(&ref_model, &p, &greedy(max)).expect("参照");
                    let ref_rows = reference_rows(&ref_model.log.borrow());
                    let base = format!("layers={layers} k={k} T={prompt_len} max={max}");
                    assert_margin(&base, &ref_rows, prompt_len, max);

                    let drafts: Vec<(&str, i64, NextFn)> = vec![
                        ("全受理(別重み)", 300, Box::new(target_next)),
                        ("全棄却", 300, Box::new(wrong_next)),
                        ("部分受理", 300, periodic_next(3)),
                    ];
                    for (name, draft_seed, draft_next) in drafts {
                        let label = format!("{base} draft={name}");
                        let target =
                            Recording::new(AttnCachesLm::new(layers, 100, Box::new(target_next)));
                        let draft =
                            Recording::new(AttnCachesLm::new(layers, draft_seed, draft_next));
                        let out = generate_speculative(
                            &target,
                            &draft,
                            &p,
                            &greedy(max),
                            &SpeculativeConfig::new(k),
                        )
                        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
                        let out_vals = toks(&out);
                        let log = target.log.borrow();
                        let rows = comparable_rows(&log, &out_vals);
                        for (pos, row, l_new) in &rows {
                            assert_parity(
                                &format!("{label} 位置 {pos}（L_new={l_new}）"),
                                row,
                                &ref_rows[*pos],
                            );
                        }
                        // 空判定の防止: トークン選択位置 T-1 ..= max-2 がすべて比較されている。
                        for pos in (prompt_len - 1)..=(max - 2) {
                            assert!(
                                rows.iter().any(|r| r.0 == pos),
                                "{label}: 位置 {pos} が比較されていない"
                            );
                        }
                        verify_rows += rows.iter().filter(|r| r.2 > 1).count();
                        replays += replay_count(&log);
                        assert_eq!(out.shape(), reference.shape(), "{label}: shape");
                        assert_eq!(
                            out_vals,
                            toks(&reference),
                            "{label}: 仮説 H1 の破れ。判定を緩めず設計 §10 論点 2 として報告すること"
                        );
                        runs += 1;
                    }
                }
            }
        }
    }
    assert!(runs > 30, "掃引が縮退している（runs={runs}）");
    assert!(verify_rows > 0, "検証 forward の行が比較されていない");
    assert!(replays > 0, "再 forward（rewind-replay）が起きていない");
}
/// 公開面だけで組んだ表引き KV モデル（`inference/mod.rs` の成功例 doctest と同じ形）。層数・次トークン規則
/// （`NextFn`）を差し替えられる。位置は `caches[0].seq_len()` から取り、全層の `caches[i]` を進める（#3098）。
struct RuleTableLm {
    sas: Vec<StatefulAttention>,
    next: NextFn,
}

impl RuleTableLm {
    fn new(layers: usize, seed_base: u64, next: NextFn) -> Self {
        Self {
            sas: attention_layers(layers, seed_base),
            next,
        }
    }
}

impl AutoregressiveModel for RuleTableLm {
    fn num_kv_layers(&self) -> usize {
        self.sas.len()
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let pos0 = caches[0].seq_len();
        let tape = fandhe_ai::tape();
        let emb = tape.var(&t(seq(1, V * E), &[V, E]));
        let mut x = emb.embedding(new_ids, None)?;
        for (sa, cache) in self.sas.iter().zip(caches.iter_mut()) {
            x = tape.stateful_attention_forward_with_cache(sa, &x, cache)?;
        }
        let l = new_ids.shape()[new_ids.shape().len() - 1];
        let mut data = vec![0.0f32; l * V];
        for (i, &tok) in toks(new_ids).iter().enumerate() {
            data[i * V + (self.next)(tok, pos0 + i) as usize] = 1.0;
        }
        Ok(t(data, &[1, l, V]))
    }
}

/// 公開面だけのモデルでの Greedy 一致の掃引（#3098）。軸: k・draft（全受理／全棄却／部分受理）・
/// prompt 長・`max_length`（R=0・1・k 超過を含む）・入力 rank（1／2）・target 層数。
/// tolerance は使わず完全一致を要求する。
#[test]
fn speculative_matches_greedy_generate_on_public_surface_table_model_sweep() {
    let (mut runs, mut replays) = (0usize, 0usize);
    for layers in [1usize, 2] {
        let target = Recording::new(RuleTableLm::new(layers, 1, Box::new(target_next)));
        for rank1 in [true, false] {
            for prompt_len in [1usize, 2, 5] {
                let prompt: Vec<i32> = (0..prompt_len).map(|i| ((i * 2) % V) as i32).collect();
                for k in [1usize, 2, 3, 5, 8] {
                    let mut maxes = vec![
                        prompt_len,
                        prompt_len + 1,
                        prompt_len + 2,
                        prompt_len + k + 1,
                        12,
                    ];
                    maxes.sort_unstable();
                    maxes.dedup();
                    for max in maxes {
                        let alone =
                            generate(&target, &ids(&prompt, rank1), &greedy(max)).expect("alone");
                        let drafts: Vec<(&str, NextFn)> = vec![
                            ("全受理", Box::new(target_next)),
                            ("全棄却", Box::new(wrong_next)),
                            ("部分受理", periodic_next(3)),
                        ];
                        for (name, next) in drafts {
                            let label = format!(
                                "layers={layers} rank1={rank1} T={prompt_len} k={k} max={max} draft={name}"
                            );
                            let draft = RuleTableLm::new(3 - layers, 77, next);
                            target.log.borrow_mut().clear();
                            let out = generate_speculative(
                                &target,
                                &draft,
                                &ids(&prompt, rank1),
                                &greedy(max),
                                &SpeculativeConfig::new(k),
                            )
                            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
                            assert_eq!(out.shape(), alone.shape(), "{label}: shape");
                            assert_eq!(toks(&out), toks(&alone), "{label}");
                            replays += replay_count(&target.log.borrow());
                            runs += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(runs > 300, "掃引が縮退している（runs={runs}）");
    assert!(replays > 0, "棄却後の再 forward が起きていない");
}
