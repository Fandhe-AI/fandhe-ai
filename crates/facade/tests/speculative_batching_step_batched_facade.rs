//! facade `fandhe_ai::inference::BatchScheduler::step_batched`（テンソル単位バッチ。#3106）が、各要求の
//! 出力を単独 `generate`（B = 1）と一致させることの結合テスト（イシュー #3107・親 #3103。設計記録
//! `docs/facade-speculative-decoding-batching-design.md` §6.3・§21.8・§27）。
//!
//! 承認の根拠は D-A の範囲（§21.1）に限り、本ファイルは新しい承認を主張しない。tolerance
//! （`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・baseline・判定規則は変えず、
//! `crates/*/src`・依存・`unsafe` も触らない。
//!
//! 本ファイルは facade の公開面だけで組み、内部クレート（autodiff）を import しない
//! （完了の検証シグナル。`speculative_batching_facade.rs` と同じ方針）。数値一致の判定ヘルパー
//! （REQ-2 定数・`assert_parity`）だけは `fandhe_ai_backend_cpu::parity` をテスト専用に使う。
//! 実機（CUDA／Metal）のバッチ経路は `speculative_batching_backend_parity.rs` の `#[ignore]`
//! テストが担う。
//!
//! # 判定
//!
//! - 位置ごとの logits: REQ-2 統一複合判定（`assert_parity`）。bit 一致は主張しない
//!   （GEMM のブロッキングは形状に依存するため。§21.4）
//! - token 列: Greedy で、参照の全選択位置が margin 前提（`assert_margin`）を満たす入力に限る
//! - TopK／Temperature: token 列の一致は契約にしない。分岐する前の文脈の logits だけ比べる。
//!   logits が形状に依らない表引きモデルだけは token 列の完全一致を要求してよい（§6.1）
//!
//! # フィクスチャの契約
//!
//! 対象モデル契約（§17.5・§21.6）に従い、行ごとの位置は `KvCache::valid_len(r)` から取り
//! （`seq_len()` は padding を含む物理長のため使わない）、`[b, l, V]` を返す。B 非対応のモデルを
//! 渡すと `step_batched` は黙って B = 1 へ戻り、束ねが一度も起きなくてもテストが通ってしまう。
//! そのため束ねを検証するテストはすべて `assert_batched_calls` で B > 1 の呼び出しが実在することを
//! 確かめる（空振り防止）。
//!
//! 乱数は独立した `Generator`（xorshift64*）のみで、決定的 seed を使う（非暗号。OWASP A02）。

use fandhe_ai::inference::{
    AutoregressiveModel, BatchScheduler, GenerateConfig, SamplingStrategy, SchedulerLimits,
    generate,
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
/// `SchedulerLimits` の `max_length`。
const MAX_LEN: usize = 16;
/// margin 確保用 skip 項の大きさ。判定閾値ではなくフィクスチャ定数で、結果を見て調整しない
/// （`speculative_batching_facade.rs` の `SKIP_C` と同値）。
const SKIP_C: f32 = 3.0;

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

fn target_next(tok: i32, pos: usize) -> i32 {
    ((3 * tok as usize + pos + 1) % V) as i32
}

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

/// `new_ids`（`[b, l]` または rank 1 の `[l]`）から `(b, l)` を取る。
fn dims(new_ids: &Tensor<i32>) -> (usize, usize) {
    let s = new_ids.shape();
    let l = s[s.len() - 1];
    (s.iter().product::<usize>() / l, l)
}

/// 呼び出し前の行ごとの有効長。空キャッシュ（prefill）は 0、状態なしモデル（`caches` が空）は空 Vec。
fn valid_before(caches: &[KvCache], b: usize) -> Vec<usize> {
    match caches.first() {
        Some(c) => (0..b).map(|r| c.valid_len(r).unwrap_or(0)).collect(),
        None => Vec::new(),
    }
}

/// attention 依存 logits の B 対応 KV モデル（埋め込み → `stateful_attention_forward_with_cache`
/// → lm head）。margin 確保のため `SKIP_C · onehot(next(tok, pos))` をホスト側で足す。
/// 行 `r` の位置は forward 前の `valid_len(r)` から取る（padding を含む `seq_len()` は使わない）。
struct BatchedAttnLm {
    sas: Vec<StatefulAttention>,
    seed_base: i64,
}

impl BatchedAttnLm {
    fn new(layers: usize, seed_base: i64) -> Self {
        Self {
            sas: attention_layers(layers, 500 + seed_base as u64),
            seed_base,
        }
    }
}

impl AutoregressiveModel for BatchedAttnLm {
    fn num_kv_layers(&self) -> usize {
        self.sas.len()
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (b, l) = dims(new_ids);
        let pos0 = valid_before(caches, b);
        let s = self.seed_base;
        let tape = fandhe_ai::tape();
        let emb = tape.var(&t(seq(s + 1, V * E), &[V, E]));
        let mut x = emb.embedding(new_ids, None)?;
        for (sa, cache) in self.sas.iter().zip(caches.iter_mut()) {
            x = tape.stateful_attention_forward_with_cache(sa, &x, cache)?;
        }
        let flat = x.reshape(&[b * l, E])?;
        let lm = LinearVars {
            weight: tape.var(&t(seq(s + 10, E * V), &[E, V])),
            bias: Some(tape.var(&t(seq(s + 9, V), &[V]))),
        };
        let logits = lm.forward(&flat)?.reshape(&[b, l, V])?.to_tensor();
        let mut data = logits.contiguous().host_slice().into_owned();
        let flat_ids = toks(new_ids);
        for r in 0..b {
            let base = pos0.get(r).copied().unwrap_or(0);
            for i in 0..l {
                let tok = flat_ids[r * l + i];
                data[(r * l + i) * V + target_next(tok, base + i) as usize] += SKIP_C;
            }
        }
        Ok(t(data, &[b, l, V]))
    }
}

/// logits が形状に依らない表引きの B 対応モデル。`layers == 0` は状態なし（位置は常に 0）、
/// `layers > 0` は `caches` を進め位置を `valid_len(r)` から取る。logits は bit 同一になるため、
/// TopK／Temperature でも token 列の完全一致を要求できる。
struct BatchedTableLm {
    sas: Vec<StatefulAttention>,
}

impl AutoregressiveModel for BatchedTableLm {
    fn num_kv_layers(&self) -> usize {
        self.sas.len()
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (b, l) = dims(new_ids);
        let pos0 = valid_before(caches, b);
        if !caches.is_empty() {
            let tape = fandhe_ai::tape();
            let emb = tape.var(&t(seq(1, V * E), &[V, E]));
            let mut x = emb.embedding(new_ids, None)?;
            for (sa, cache) in self.sas.iter().zip(caches.iter_mut()) {
                x = tape.stateful_attention_forward_with_cache(sa, &x, cache)?;
            }
        }
        let mut data = vec![0.0f32; b * l * V];
        let flat_ids = toks(new_ids);
        for r in 0..b {
            let base = pos0.get(r).copied().unwrap_or(0);
            for i in 0..l {
                let tok = flat_ids[r * l + i];
                data[(r * l + i) * V + target_next(tok, base + i) as usize] = 1.0;
            }
        }
        Ok(t(data, &[b, l, V]))
    }
}

/// 1 回の `forward_step` の記録。
struct Call {
    b: usize,
    l: usize,
    /// 呼び出し前の行ごとの有効長（状態なしモデルは空）。
    valid: Vec<usize>,
    /// 呼び出し前の物理長 `caches[0].seq_len()`（状態なしモデルは 0）。
    phys: usize,
    ids: Vec<i32>,
    logits: Vec<f32>,
}

struct BatchedRecording<M> {
    inner: M,
    log: RefCell<Vec<Call>>,
}

impl<M> BatchedRecording<M> {
    fn new(inner: M) -> Self {
        Self {
            inner,
            log: RefCell::new(Vec::new()),
        }
    }
}

impl<M: AutoregressiveModel> AutoregressiveModel for BatchedRecording<M> {
    fn num_kv_layers(&self) -> usize {
        self.inner.num_kv_layers()
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (b, l) = dims(new_ids);
        let valid = valid_before(caches, b);
        let phys = caches.first().map_or(0, |c| c.seq_len());
        let logits = self.inner.forward_step(new_ids, caches)?;
        self.log.borrow_mut().push(Call {
            b,
            l,
            valid,
            phys,
            ids: toks(new_ids),
            logits: logits.contiguous().host_slice().into_owned(),
        });
        Ok(logits)
    }
}

/// 空振り防止: `b > 1 && l == 1` の（束ねた）呼び出しが 1 件以上あることを確かめ、件数を返す。
fn assert_batched_calls(label: &str, log: &[Call]) -> usize {
    let n = log.iter().filter(|c| c.b > 1 && c.l == 1).count();
    assert!(
        n >= 1,
        "{label}: 束ねた forward（B > 1）が 1 件も起きていない（B 非対応モデル等で B = 1 へ戻っている）"
    );
    n
}

/// 行ごとの有効長が揃っていない（= padding を伴う）呼び出しの件数。
fn count_padded_calls(log: &[Call]) -> usize {
    log.iter()
        .filter(|c| c.valid.iter().any(|&v| v != c.valid[0]))
        .count()
}

/// 参照（単独 `generate`）の記録から絶対位置ごとの logits 行を作る。重複・欠落は panic（fail-closed）。
fn reference_rows(log: &[Call]) -> Vec<Vec<f32>> {
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for call in log {
        assert_eq!(call.b, 1, "参照は B = 1");
        for (i, row) in call.logits.chunks(V).enumerate() {
            assert_eq!(call.valid[0] + i, rows.len(), "参照の位置が連続しない");
            rows.push(row.to_vec());
        }
    }
    rows
}

/// 入力の妥当性検査（新しい tolerance ではない）。参照のトークン選択位置 `T-1 ..= max-2` で
/// `top1 - top2 > 2·(rtol·M/(1-rtol) ∨ abs)` を要求する（REQ-2 定数から導いた帰結。
/// 導出は `speculative_batching_facade.rs::assert_margin`）。
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
        let bound =
            (RELATIVE_TOLERANCE * m / (1.0 - RELATIVE_TOLERANCE)).max(ABSOLUTE_RESCUE_THRESHOLD);
        let margin = f64::from(sorted[0] - sorted[1]);
        assert!(
            margin > 2.0 * bound,
            "{label}: フィクスチャ不正（margin 前提を満たさない）位置 {p}: margin={margin}"
        );
    }
}

/// 要求の計画。`(submit する tick, prompt, rank1 か, config)`。
type Plan = Vec<(usize, Vec<i32>, bool, GenerateConfig)>;

/// 1 回の駆動の結果。出力は submit 順。
struct Outcome {
    outputs: Vec<Tensor<i32>>,
}

/// `plan` を tick どおりに submit しつつ、全要求が掃けるまで `step_batched` を回す。
fn run_plan<M: AutoregressiveModel>(model: &M, plan: &Plan, max_active: usize) -> Outcome {
    let limits = SchedulerLimits::new(max_active, plan.len().max(1), MAX_LEN).expect("limits");
    let mut s = BatchScheduler::new(limits);
    let mut submitted = Vec::new();
    let mut finished = Vec::new();
    for tick in 0..TICK_CAP {
        for (at, prompt, rank1, c) in plan {
            if *at == tick {
                submitted.push(s.submit(&ids(prompt, *rank1), c).expect("submit"));
            }
        }
        if submitted.len() == plan.len() && s.queued_len() == 0 && s.active_len() == 0 {
            assert!(s.take_failed().is_empty(), "失敗した要求がある");
            let outputs = submitted
                .iter()
                .map(|id| {
                    finished
                        .iter()
                        .find(|(fid, _): &&(_, Tensor<i32>)| fid == id)
                        .map(|(_, out)| out.clone())
                        .expect("要求が finished にない")
                })
                .collect();
            return Outcome { outputs };
        }
        s.step_batched(model).expect("step_batched");
        finished.extend(s.take_finished());
        assert!(s.take_failed().is_empty(), "失敗した要求がある");
    }
    panic!("TICK_CAP に到達した（スケジューラが掃けない）");
}

/// 単独 `generate` の参照（出力と位置ごとの logits 行）。
fn alone(
    layers: usize,
    prompt: &[i32],
    rank1: bool,
    c: &GenerateConfig,
) -> (Tensor<i32>, Vec<Vec<f32>>) {
    let m = BatchedRecording::new(BatchedAttnLm::new(layers, 100));
    let out = generate(&m, &ids(prompt, rank1), c).expect("単独 generate");
    let rows = reference_rows(&m.log.borrow());
    (out, rows)
}

/// lockstep（全要求を tick 0 で投入・`max_active >= N`・全 Greedy）の計画。prompt 長と
/// `max_length` を混在させる。各要求は decode を 2 回以上行う（`max >= T + 3`）。
fn lockstep_plan() -> Plan {
    vec![
        (0, vec![1], true, greedy(8)),
        (0, vec![2, 3, 4], false, greedy(9)),
        (0, vec![5, 6, 0, 1, 2], true, greedy(11)),
        (0, vec![3, 1], false, greedy(10)),
    ]
}

/// lockstep の `step_batched` 記録を要求へ対応づける: 先頭 N 件は B = 1 の prefill、以降は decode の
/// k 回目に残っている要求（`T + 1 + k < max`。submit 順）を行にする。位置 `T + k` の行ごとに
/// `visit(要求 index, 位置, 行)` を呼び、要求ごとの比較行数を返す。対応づけは fail-closed に検査する。
fn map_lockstep(
    plan: &Plan,
    log: &[Call],
    mut visit: impl FnMut(usize, usize, &[f32], i32),
) -> Vec<usize> {
    let n = plan.len();
    for (i, call) in log.iter().take(n).enumerate() {
        assert_eq!((call.b, call.l), (1, plan[i].1.len()), "prefill {i}");
    }
    let mut counts = vec![0usize; n];
    let mut k = 0usize;
    let mut idx = n;
    loop {
        let alive: Vec<usize> = (0..n)
            .filter(|&i| plan[i].1.len() + 1 + k < plan[i].3.max_length)
            .collect();
        if alive.is_empty() {
            break;
        }
        let call = &log[idx];
        assert_eq!((call.b, call.l), (alive.len(), 1), "decode {k} の b");
        for (r, &i) in alive.iter().enumerate() {
            let pos = plan[i].1.len() + k;
            assert_eq!(call.valid[r], pos, "decode {k} 行 {r} の valid_len");
            visit(i, pos, &call.logits[r * V..(r + 1) * V], call.ids[r]);
            counts[i] += 1;
        }
        k += 1;
        idx += 1;
    }
    assert_eq!(idx, log.len(), "記録に未対応の呼び出しが残っている");
    counts
}

#[test]
fn step_batched_actually_batches_with_fixture() {
    // フィクスチャが B = 1 へ黙って戻っていないことの最小確認。
    let m = BatchedRecording::new(BatchedAttnLm::new(2, 100));
    run_plan(&m, &lockstep_plan(), 4);
    assert!(assert_batched_calls("smoke", &m.log.borrow()) >= 2);
}

/// 作業 1: prompt 長と `max_length` の違う要求を同時に投入し、各要求の出力を単独 `generate` と比べる。
#[test]
fn step_batched_lockstep_mixed_lengths_matches_standalone_generate() {
    let plan = lockstep_plan();
    for layers in [1usize, 2] {
        let label = format!("layers={layers}");
        let refs: Vec<_> = plan
            .iter()
            .map(|(_, p, r1, c)| {
                let (out, rows) = alone(layers, p, *r1, c);
                assert_margin(&label, &rows, p.len(), c.max_length);
                (out, rows)
            })
            .collect();
        let m = BatchedRecording::new(BatchedAttnLm::new(layers, 100));
        let out = run_plan(&m, &plan, plan.len());
        let log = m.log.borrow();
        assert_batched_calls(&label, &log);
        assert!(
            count_padded_calls(&log) >= 1,
            "{label}: padding を伴う呼び出しがない"
        );
        // 退出で B が減る呼び出しがある（B の値が複数）。
        let mut bs: Vec<usize> = log.iter().filter(|c| c.l == 1).map(|c| c.b).collect();
        bs.sort_unstable();
        bs.dedup();
        assert!(bs.len() >= 3, "{label}: 退出で B が変わっていない: {bs:?}");

        let counts = map_lockstep(&plan, &log, |i, pos, row, _| {
            assert_parity(
                &format!("{label} 要求 {i} 位置 {pos}"),
                row,
                &refs[i].1[pos],
            );
        });
        for (i, (_, p, _, c)) in plan.iter().enumerate() {
            // トークンを選ぶ位置 T-1 ..= max-2 のうち decode 側 T ..= max-2 が全部比較された。
            assert_eq!(counts[i], c.max_length - 1 - p.len(), "{label} 要求 {i}");
            assert_eq!(
                out.outputs[i].shape(),
                refs[i].0.shape(),
                "{label} {i} shape"
            );
            assert_eq!(
                toks(&out.outputs[i]),
                toks(&refs[i].0),
                "{label} {i} token 列"
            );
        }
    }
}

/// 作業 3: padding の量だけを変えた同じ要求（R）の出力が変わらない。
#[test]
fn step_batched_padding_amount_does_not_change_output() {
    let r_prompt = vec![1];
    let r_cfg = greedy(6);
    let (r_ref, r_rows) = alone(2, &r_prompt, true, &r_cfg);
    assert_margin("R", &r_rows, r_prompt.len(), 6);

    let mut seen: Vec<Vec<i32>> = Vec::new();
    for lc in [1usize, 2, 4, 7] {
        let c_prompt: Vec<i32> = (0..lc).map(|i| ((i * 3 + 2) % V) as i32).collect();
        let plan: Plan = vec![
            (0, r_prompt.clone(), true, r_cfg.clone()),
            (0, c_prompt, true, greedy(lc + 8)),
        ];
        let m = BatchedRecording::new(BatchedAttnLm::new(2, 100));
        let out = run_plan(&m, &plan, 2);
        let log = m.log.borrow();
        assert_batched_calls(&format!("Lc={lc}"), &log);
        // R の decode 行（B = 2 の呼び出し）。k 回目の位置は T + k。R の padding 量は Lc - 1。
        let mut rows = 0usize;
        for call in log.iter().filter(|c| c.b == 2 && c.l == 1) {
            let pos = call.valid[0];
            assert_eq!(
                call.phys - call.valid[0],
                lc - 1,
                "Lc={lc}: R の padding 量"
            );
            assert_parity(
                &format!("Lc={lc} 位置 {pos}"),
                &call.logits[..V],
                &r_rows[pos],
            );
            rows += 1;
        }
        // R の decode は max - T - 1 = 4 回すべて C と同居している。
        assert_eq!(rows, 4, "Lc={lc}: R の decode 行が欠けている");
        assert_eq!(toks(&out.outputs[0]), toks(&r_ref), "Lc={lc}: 単独との一致");
        seen.push(toks(&out.outputs[0]));
    }
    // 単独 R を step_batched に流す回（decode 要求 1 件 = B = 1 経路）。
    let solo = run_plan(
        &BatchedRecording::new(BatchedAttnLm::new(2, 100)),
        &vec![(0, r_prompt, true, r_cfg)],
        1,
    );
    seen.push(toks(&solo.outputs[0]));
    assert!(
        seen.windows(2).all(|w| w[0] == w[1]),
        "padding 量で出力が変わった"
    );
}

/// 途中参加・退出の混在計画。R は index 0。
fn mid_plan(variant: usize) -> Plan {
    let r = (0usize, vec![2, 3, 1], false, greedy(9));
    match variant {
        // R だけ。
        0 => vec![r],
        // 途中参加あり（max_active 3 < 要求数 5。空いた枠を再利用する）。
        1 => vec![
            r,
            (0, vec![1], true, greedy(6)),
            (1, vec![4, 2], false, greedy(10)),
            (3, vec![0], true, greedy(8)),
            (4, vec![5, 5, 6, 1], false, greedy(12)),
        ],
        // 相手が早く退出する。
        _ => vec![
            r,
            (0, vec![6], true, greedy(4)),
            (0, vec![1, 1], false, greedy(5)),
            (2, vec![3], true, greedy(6)),
        ],
    }
}

/// 作業 2: 途中参加・退出・長さの混在があっても他の要求の出力が変わらない。
/// 位置ごとの logits の対応づけは受け入れの再現が壊れやすいため、出力水準の比較に留める。
#[test]
fn step_batched_mid_join_and_leave_do_not_change_other_outputs() {
    let layers = 2;
    let mut r_outs: Vec<Vec<i32>> = Vec::new();
    for variant in 0..3 {
        let plan = mid_plan(variant);
        let refs: Vec<_> = plan
            .iter()
            .map(|(_, p, r1, c)| {
                let (out, rows) = alone(layers, p, *r1, c);
                assert_margin("mid", &rows, p.len(), c.max_length);
                out
            })
            .collect();
        let m = BatchedRecording::new(BatchedAttnLm::new(layers, 100));
        let out = run_plan(&m, &plan, 3);
        for (i, want) in refs.iter().enumerate() {
            assert_eq!(
                out.outputs[i].shape(),
                want.shape(),
                "variant {variant} 要求 {i}"
            );
            assert_eq!(
                toks(&out.outputs[i]),
                toks(want),
                "variant {variant} 要求 {i}"
            );
        }
        let log = m.log.borrow();
        if variant > 0 {
            assert_batched_calls(&format!("variant {variant}"), &log);
            let mut bs: Vec<usize> = log
                .iter()
                .filter(|c| c.l == 1 && c.b > 1)
                .map(|c| c.b)
                .collect();
            bs.sort_unstable();
            bs.dedup();
            assert!(
                bs.len() >= 2,
                "variant {variant}: 構成が変わっていない: {bs:?}"
            );
        }
        r_outs.push(toks(&out.outputs[0]));
    }
    assert!(
        r_outs.windows(2).all(|w| w[0] == w[1]),
        "他要求の構成で R の出力が変わった"
    );
}

/// §21.8 の TopK／Temperature: token 列は主張しない。分岐前は単独 run の logits と、全 decode 位置は
/// 束ねた run の入力 id 列を単独モデルへ再生した同一文脈の logits と比べる。
#[test]
fn step_batched_sampling_requests_match_before_divergence() {
    let plan: Plan = vec![
        (0, vec![1], true, greedy(9)),
        (
            0,
            vec![2, 3, 4],
            false,
            GenerateConfig::new(10, SamplingStrategy::TopK(3)).with_seed(11),
        ),
        (
            0,
            vec![5, 0],
            true,
            GenerateConfig::new(9, SamplingStrategy::Temperature(0.7)).with_seed(23),
        ),
    ];
    for layers in [1usize, 2] {
        let label = format!("layers={layers}");
        let refs: Vec<_> = plan
            .iter()
            .map(|(_, p, r1, c)| alone(layers, p, *r1, c))
            .collect();
        let m = BatchedRecording::new(BatchedAttnLm::new(layers, 100));
        let out = run_plan(&m, &plan, plan.len());
        let log = m.log.borrow();
        assert_batched_calls(&label, &log);
        let mut compared_first = [false; 3];
        let mut replayed = [0usize; 3];
        map_lockstep(&plan, &log, |i, pos, row, _| {
            let t0 = plan[i].1.len();
            // 位置 pos の文脈 = prompt + 生成 token（pos まで）。束ねた run と単独で一致する間だけ比べる。
            let (b_out, a_out) = (toks(&out.outputs[i]), toks(&refs[i].0));
            if b_out[..=pos] == a_out[..=pos] {
                assert_parity(
                    &format!("{label} 要求 {i} 位置 {pos}"),
                    row,
                    &refs[i].1[pos],
                );
                if pos == t0 {
                    compared_first[i] = true;
                }
            }
            // 分岐後も検査を残すため、束ねた run の実際の入力 id 列（prompt + 生成 token の
            // `..=pos`）を単独モデルへ再生し、同じ文脈の位置 pos の logits と比べる（§21.8）。
            // 先頭呼び出し（prefill）の最終行が位置 pos の logits になる。
            let replay = BatchedRecording::new(BatchedAttnLm::new(layers, 100));
            let cfg = greedy(pos + 2);
            generate(&replay, &ids(&b_out[..=pos], true), &cfg).expect("再生 generate");
            let rl = replay.log.borrow();
            let last = rl[0].logits.chunks(V).last().expect("再生の logits");
            assert_parity(
                &format!("{label} 要求 {i} 位置 {pos}（入力 id 列の再生）"),
                row,
                last,
            );
            replayed[i] += 1;
        });
        // 全 decode 位置（分岐後を含む）が再生比較された。
        for (i, (_, p, _, c)) in plan.iter().enumerate() {
            assert_eq!(
                replayed[i],
                c.max_length - 1 - p.len(),
                "{label} 要求 {i} 再生"
            );
        }
        // prefill は B = 1 で単独と同じ形状なので、最初の decode 位置は必ず比較される。
        assert_eq!(compared_first, [true; 3], "{label}");
    }
}

/// 表引きモデル（logits が形状に依らない）は、全戦略で token 列の完全一致を要求できる。
#[test]
fn step_batched_table_model_matches_exactly_for_all_strategies() {
    let plan: Plan = vec![
        (0, vec![0], true, greedy(8)),
        (
            0,
            vec![2, 3],
            false,
            GenerateConfig::new(9, SamplingStrategy::TopK(3)).with_seed(11),
        ),
        (
            1,
            vec![4, 1, 2, 6],
            true,
            GenerateConfig::new(11, SamplingStrategy::Temperature(0.7)).with_seed(23),
        ),
        (
            2,
            vec![3],
            false,
            GenerateConfig::new(7, SamplingStrategy::TopK(2)).with_seed(5),
        ),
    ];
    for layers in [0usize, 1, 2] {
        let model = || {
            BatchedRecording::new(BatchedTableLm {
                sas: attention_layers(layers, 1),
            })
        };
        let m = model();
        let out = run_plan(&m, &plan, 3);
        let log = m.log.borrow();
        assert_batched_calls(&format!("table layers={layers}"), &log);
        if layers > 0 {
            assert!(
                count_padded_calls(&log) >= 1,
                "layers={layers}: padding を伴う呼び出しがない"
            );
        }
        for (i, (_, p, r1, c)) in plan.iter().enumerate() {
            let want = generate(&model(), &ids(p, *r1), c).expect("単独");
            assert_eq!(
                out.outputs[i].shape(),
                want.shape(),
                "layers={layers} 要求 {i}"
            );
            assert_eq!(
                toks(&out.outputs[i]),
                toks(&want),
                "layers={layers} 要求 {i}"
            );
        }
    }
}

/// 共通の受入基準: 同じ入力・同じ形状での再実行は出力も全呼び出しの logits も bit 一致する。
#[test]
fn step_batched_is_deterministic() {
    let bits = |log: &[Call]| -> Vec<(usize, usize, Vec<u32>)> {
        log.iter()
            .map(|c| (c.b, c.l, c.logits.iter().map(|v| v.to_bits()).collect()))
            .collect()
    };
    for (plan, max_active) in [(lockstep_plan(), 4usize), (mid_plan(1), 3)] {
        let run = || {
            let m = BatchedRecording::new(BatchedAttnLm::new(2, 100));
            let out = run_plan(&m, &plan, max_active);
            let outs: Vec<Vec<i32>> = out.outputs.iter().map(toks).collect();
            let b = bits(&m.log.borrow());
            (outs, b)
        };
        assert_eq!(run(), run());
    }
}
