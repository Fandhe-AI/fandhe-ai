//! `fandhe_ai_autodiff::generate::scheduler`（連続バッチング。イシュー #2888）の
//! 統合テスト（イシュー #2889）。設計の正は
//! `docs/facade-speculative-decoding-batching-design.md` §6.3・§7・§8.3。
//!
//! # 固定する 3 点
//!
//! 1. 要求 A・B を交互・途中参加・同一シード・スロット再利用で進めても、各要求の
//!    出力が同じ要求を単独 `generate` した結果と token 列・shape で完全一致する
//!    （§6.3: 第 1 段階は要求ごとに B = 1 で forward するため完全一致を要求できる）。
//!    KV キャッシュ付きモデルと状態なしモデルの両方で検査する
//! 2. 途中（prefill／decode）で失敗した要求は当該要求だけが切り離され、他要求の出力は
//!    不変（§7）
//! 3. `scheduler.rs` が thread／mpsc／tokio／async／net／fs／rayon に依存しない（§8.3）
//!
//! # 対象外
//!
//! - モデル内部に状態を保持する型（`RefCell<..>` 等）は契約上対象外（§8.1）のため、
//!   成功・失敗のどちらとしても固定しない
//! - CUDA／Metal 実機 parity は #2890 の担当
//! - サンプリングの `Generator` は非暗号の xorshift64* であり、生成 token を
//!   セキュリティ用途に使わない（OWASP A02。scheduler.rs と同じ注記）

use fandhe_ai_autodiff::generate::scheduler::{BatchScheduler, RequestId, SchedulerLimits};
use fandhe_ai_autodiff::generate::{
    AutoregressiveModel, GenerateConfig, SamplingStrategy, generate,
};
use fandhe_ai_autodiff::nn::{KvCache, LinearVars, MultiheadAttentionVars};
use fandhe_ai_autodiff::{AutodiffError, Tape, manual_seed, rand};
use fandhe_ai_tensor_core::Tensor;
use std::path::Path;

const V: usize = 8;
const E: usize = 4;
/// ループの上限（ハング防止。到達したら panic する fail-closed）。
const TICK_CAP: usize = 256;

// ---------------------------------------------------------------------------
// フィクスチャ
// ---------------------------------------------------------------------------

fn seq(seed: i64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((seed + i as i64 * 7) % 13 - 6) as f32 * 0.05)
        .collect()
}

fn ids1(v: &[i32]) -> Tensor<i32> {
    Tensor::new(v.to_vec(), &[v.len()]).expect("fixture")
}

fn ids2(v: &[i32]) -> Tensor<i32> {
    Tensor::new(v.to_vec(), &[1, v.len()]).expect("fixture")
}

fn data(t: &Tensor<i32>) -> Vec<i32> {
    t.contiguous().host_slice().into_owned()
}

fn limits(a: usize, q: usize, m: usize) -> SchedulerLimits {
    SchedulerLimits::new(a, q, m).expect("fixture")
}

fn cfg(max_length: usize, strategy: SamplingStrategy, seed: u64) -> GenerateConfig {
    GenerateConfig::new(max_length, strategy).with_seed(seed)
}

/// 状態なしの表引きモデル（`num_kv_layers() == 0`）。次 token は `(3t+1) % V` を首位にし、
/// 他の語彙にも非 0 の値を置いて TopK／Temperature のサンプリングが意味を持つようにする。
struct TableModel;

impl AutoregressiveModel for TableModel {
    fn num_kv_layers(&self) -> usize {
        0
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        _caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (b, l) = (new_ids.shape()[0], new_ids.shape()[1]);
        let mut out = Vec::new();
        for tok in data(new_ids) {
            let next = (tok * 3 + 1).rem_euclid(V as i32) as usize;
            for v in 0..V {
                out.push(if v == next { 2.0 } else { 0.1 * (v % 3) as f32 });
            }
        }
        Ok(Tensor::new(out, &[b, l, V]).expect("fixture"))
    }
}

/// `Embedding → MHA（KV キャッシュ付き）× 2 層 → lm head` の KV モデル。
/// 履歴（キャッシュの内容）が logits に効く。呼び出しごとに新しい `Tape` を作る。
struct KvLm;

impl AutoregressiveModel for KvLm {
    fn num_kv_layers(&self) -> usize {
        2
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let t = |d: Vec<f32>, s: &[usize]| Tensor::new(d, s).expect("fixture");
        let tape = Tape::new();
        let emb = tape.var(&t(seq(1, V * E), &[V, E]));
        let mut x = emb.embedding(new_ids, None)?;
        for cache in caches.iter_mut() {
            let lin = |s: i64| LinearVars {
                weight: tape.var(&t(seq(s, E * E), &[E, E])),
                bias: Some(tape.var(&t(seq(s + 1, E), &[E]))),
            };
            let mha = MultiheadAttentionVars::new(2, lin(2), lin(4), lin(6), lin(8))?;
            x = mha.forward_with_cache(&x, &x, &x, cache)?;
        }
        let (b, l) = (new_ids.shape()[0], new_ids.shape()[1]);
        let flat = x.reshape(&[b * l, E])?;
        let lm = LinearVars {
            weight: tape.var(&t(seq(10, E * V), &[E, V])),
            bias: Some(tape.var(&t(seq(9, V), &[V]))),
        };
        Ok(lm.forward(&flat)?.reshape(&[b, l, V])?.to_tensor())
    }
}

/// 失敗注入の条件。ラッパーは出力に影響する内部状態を持たない（決定的）。
#[derive(Clone, Copy)]
enum Fail {
    /// decode（`L == 1` かつキャッシュ長 `n`）で失敗する。KV モデル用。
    DecodeAtSeqLen(usize),
    /// prefill の `new_ids` に番兵 token が含まれれば失敗する。
    PromptContains(i32),
    /// decode で受けた token が `tok` のとき失敗する。状態なしモデル用。
    DecodeToken(i32),
}

struct Failing<'a, M: AutoregressiveModel> {
    inner: &'a M,
    fail: Fail,
}

const MARKER: &str = "fixture: injected failure";

impl<M: AutoregressiveModel> AutoregressiveModel for Failing<'_, M> {
    fn num_kv_layers(&self) -> usize {
        self.inner.num_kv_layers()
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let l = new_ids.shape()[1];
        let toks = data(new_ids);
        let prefill = l > 1 || caches.first().is_some_and(|c| c.seq_len() == 0);
        let hit = match self.fail {
            Fail::DecodeAtSeqLen(n) => {
                !prefill && l == 1 && caches.first().is_some_and(|c| c.seq_len() == n)
            }
            Fail::PromptContains(s) => prefill && toks.contains(&s),
            Fail::DecodeToken(t) => !prefill && toks.contains(&t),
        };
        if hit {
            return Err(AutodiffError::InvalidArgument(MARKER.to_string()));
        }
        self.inner.forward_step(new_ids, caches)
    }
}

fn is_injected(e: &AutodiffError) -> bool {
    matches!(e, AutodiffError::InvalidArgument(m) if m.contains(MARKER))
}

// ---------------------------------------------------------------------------
// ドライバ
// ---------------------------------------------------------------------------

type Req = (usize, Tensor<i32>, GenerateConfig);

struct Outcome {
    /// `schedule` と同じ順の RequestId。
    ids: Vec<RequestId>,
    finished: Vec<(RequestId, Tensor<i32>)>,
    failed: Vec<(RequestId, AutodiffError)>,
    completed_sum: usize,
    peak_active: usize,
    peak_queued: usize,
}

/// `schedule` の各要求を指定 tick で submit しながら `step` を回し、全要求が掃けるまで進める。
fn drive<M: AutoregressiveModel>(m: &M, s: &mut BatchScheduler, schedule: &[Req]) -> Outcome {
    let mut o = Outcome {
        ids: Vec::new(),
        finished: Vec::new(),
        failed: Vec::new(),
        completed_sum: 0,
        peak_active: 0,
        peak_queued: 0,
    };
    let mut submitted = 0usize;
    for tick in 0..TICK_CAP {
        for (at, input, c) in schedule {
            if *at == tick {
                o.ids.push(s.submit(input, c).expect("submit"));
                submitted += 1;
            }
        }
        if submitted == schedule.len() && s.queued_len() == 0 && s.active_len() == 0 {
            return o;
        }
        o.completed_sum += s.step(m).expect("step は Ok を返し続ける");
        o.peak_active = o.peak_active.max(s.active_len());
        o.peak_queued = o.peak_queued.max(s.queued_len());
        o.finished.extend(s.take_finished());
        o.failed.extend(s.take_failed());
    }
    panic!("TICK_CAP に到達した（スケジューラが掃けない）");
}

/// `finished` の各要求が、`single`（失敗注入なしのモデル）での単独 `generate` と
/// shape・token 列で完全一致することを確認する。`skip` の要求は対象外。
fn assert_matches_single<M: AutoregressiveModel>(
    single: &M,
    schedule: &[Req],
    o: &Outcome,
    skip: &[usize],
) {
    for (i, (_, input, c)) in schedule.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        let got = o
            .finished
            .iter()
            .find(|(id, _)| *id == o.ids[i])
            .unwrap_or_else(|| panic!("要求 {i} が finished にない"));
        let want = generate(single, input, c).expect("単独 generate");
        assert_eq!(got.1.shape(), want.shape(), "要求 {i}: shape 不一致");
        assert_eq!(data(&got.1), data(&want), "要求 {i}: token 列不一致");
    }
}

fn assert_drained(s: &BatchScheduler, o: &Outcome, expect_ok: usize) {
    assert_eq!(s.queued_len(), 0);
    assert_eq!(s.active_len(), 0);
    assert_eq!(o.finished.len(), expect_ok);
    assert_eq!(o.completed_sum, expect_ok, "step 戻り値の合計 = 成功完了数");
}

// ---------------------------------------------------------------------------
// 単独実行一致（受け入れ条件 1）
// ---------------------------------------------------------------------------

fn req_a(at: usize) -> Req {
    (at, ids1(&[1, 2, 3]), cfg(10, SamplingStrategy::Greedy, 5))
}
/// A と同一シード・別 prompt／strategy・rank 2。
fn req_b(at: usize) -> Req {
    (
        at,
        ids2(&[3, 1, 4, 1]),
        cfg(12, SamplingStrategy::TopK(3), 5),
    )
}

fn scenario_simultaneous<M: AutoregressiveModel>(m: &M) {
    let schedule = vec![req_a(0), req_b(0)];
    let mut s = BatchScheduler::new(limits(2, 8, 32));
    let o = drive(m, &mut s, &schedule);
    assert_drained(&s, &o, 2);
    assert_matches_single(m, &schedule, &o, &[]);
    assert!(o.failed.is_empty());
}

fn scenario_staggered_same_seed<M: AutoregressiveModel>(m: &M) {
    let schedule = vec![req_a(0), req_b(3)];
    let mut s = BatchScheduler::new(limits(2, 8, 32));
    let o = drive(m, &mut s, &schedule);
    assert_drained(&s, &o, 2);
    assert_matches_single(m, &schedule, &o, &[]);
    assert!(o.failed.is_empty());
}

fn scenario_serial<M: AutoregressiveModel>(m: &M) {
    let schedule = vec![req_a(0), req_b(0)];
    let mut s = BatchScheduler::new(limits(1, 8, 32));
    let o = drive(m, &mut s, &schedule);
    assert_drained(&s, &o, 2);
    assert_eq!(o.peak_active, 1, "max_active = 1");
    assert_eq!(o.peak_queued, 1, "B は A の完了まで待ち行列に残る");
    assert_matches_single(m, &schedule, &o, &[]);
}

fn scenario_slot_reuse_second_wave<M: AutoregressiveModel>(m: &M) {
    // A（短い）の完了で空いたスロットに、待ち行列の C が B の進行中に参加する。
    // 流し切ったあと（tick 60）に同じスケジューラへ第 2 波 D・E を投入する。
    let schedule = vec![
        (0, ids1(&[2, 5]), cfg(4, SamplingStrategy::Greedy, 1)),
        (
            0,
            ids2(&[1, 6, 3]),
            cfg(14, SamplingStrategy::Temperature(0.9), 2),
        ),
        (0, ids1(&[7, 0]), cfg(9, SamplingStrategy::TopK(2), 2)),
        (
            60,
            ids1(&[4, 4, 2]),
            cfg(8, SamplingStrategy::Temperature(1.2), 3),
        ),
        (60, ids2(&[6]), cfg(7, SamplingStrategy::Greedy, 3)),
    ];
    let mut s = BatchScheduler::new(limits(2, 8, 32));
    let o = drive(m, &mut s, &schedule);
    assert_drained(&s, &o, 5);
    assert_eq!(o.peak_active, 2);
    assert_matches_single(m, &schedule, &o, &[]);
}

fn scenario_matrix<M: AutoregressiveModel>(m: &M) {
    // rank 1／rank 2、3 戦略、prompt のみ（forward なし）を同じバッチに混ぜる。
    let schedule = vec![
        (0, ids1(&[1, 2, 3]), cfg(9, SamplingStrategy::Greedy, 7)),
        (0, ids2(&[4, 5]), cfg(8, SamplingStrategy::TopK(4), 7)),
        (
            0,
            ids1(&[6, 1]),
            cfg(10, SamplingStrategy::Temperature(0.7), 7),
        ),
        (0, ids1(&[3, 3, 3]), cfg(3, SamplingStrategy::Greedy, 7)),
        (1, ids2(&[2, 2]), cfg(2, SamplingStrategy::TopK(2), 9)),
        (
            2,
            ids2(&[0, 7, 5]),
            cfg(11, SamplingStrategy::Temperature(1.1), 9),
        ),
    ];
    let mut s = BatchScheduler::new(limits(3, 8, 32));
    let o = drive(m, &mut s, &schedule);
    assert_drained(&s, &o, 6);
    assert_matches_single(m, &schedule, &o, &[]);
    for (i, (_, input, c)) in schedule.iter().enumerate() {
        let got = &o.finished.iter().find(|(id, _)| *id == o.ids[i]).unwrap().1;
        let expect_shape: Vec<usize> = if input.shape().len() == 1 {
            vec![c.max_length]
        } else {
            vec![1, c.max_length]
        };
        assert_eq!(got.shape(), expect_shape.as_slice(), "要求 {i}");
    }
}

#[test]
fn simultaneous_matches_single_stateless() {
    scenario_simultaneous(&TableModel);
}
#[test]
fn simultaneous_matches_single_kv() {
    scenario_simultaneous(&KvLm);
}
#[test]
fn staggered_same_seed_matches_single_stateless() {
    scenario_staggered_same_seed(&TableModel);
}
#[test]
fn staggered_same_seed_matches_single_kv() {
    scenario_staggered_same_seed(&KvLm);
}
#[test]
fn serial_matches_single_stateless() {
    scenario_serial(&TableModel);
}
#[test]
fn serial_matches_single_kv() {
    scenario_serial(&KvLm);
}
#[test]
fn slot_reuse_second_wave_matches_single_stateless() {
    scenario_slot_reuse_second_wave(&TableModel);
}
#[test]
fn slot_reuse_second_wave_matches_single_kv() {
    scenario_slot_reuse_second_wave(&KvLm);
}
#[test]
fn shape_strategy_matrix_matches_single_stateless() {
    scenario_matrix(&TableModel);
}
#[test]
fn shape_strategy_matrix_matches_single_kv() {
    scenario_matrix(&KvLm);
}

/// テストが自明に通らないことの担保: キャッシュ内容と RNG が出力に効く。
#[test]
fn fixtures_are_sensitive_to_cache_and_seed() {
    // 最後の token が同じで先頭部分だけ異なる 2 prompt で、最終位置の logits が異なる
    // （キャッシュの中身が出力に効く＝要求間でキャッシュが混入すれば検出できる）。
    let last_logits = |p: &[i32]| {
        let mut caches: Vec<KvCache> = (0..KvLm.num_kv_layers()).map(|_| KvCache::new()).collect();
        let l = KvLm.forward_step(&ids2(p), &mut caches).unwrap();
        let v = data_f32(&l);
        v[v.len() - V..].to_vec()
    };
    assert_ne!(
        last_logits(&[1, 2, 3, 4]),
        last_logits(&[5, 6, 3, 4]),
        "KV モデルの logits が履歴に依存していない"
    );
    // 同じ prompt・異なる seed で出力が変わる（RNG が出力に効く）。
    let seeds = |m: &dyn AutoregressiveModel, st: SamplingStrategy| {
        let o = |seed| data(&generate(m, &ids1(&[1, 2]), &cfg(14, st, seed)).unwrap());
        o(1) != o(2) || o(2) != o(3)
    };
    for st in [
        SamplingStrategy::TopK(3),
        SamplingStrategy::Temperature(1.0),
    ] {
        assert!(seeds(&TableModel, st), "TableModel: seed が出力に効かない");
        assert!(seeds(&KvLm, st), "KvLm: seed が出力に効かない");
    }
}

#[test]
fn scheduler_does_not_consume_global_rng() {
    let run = |drive_it: bool| {
        manual_seed(1234);
        if drive_it {
            let mut s = BatchScheduler::new(limits(2, 4, 32));
            let sched = vec![req_b(0), req_a(0)];
            let _ = drive(&KvLm, &mut s, &sched);
        }
        data_f32(&rand(&[4]).unwrap())
    };
    assert_eq!(
        run(false),
        run(true),
        "スケジューラはグローバル RNG を消費しない"
    );
}

fn data_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().host_slice().into_owned()
}

// ---------------------------------------------------------------------------
// 失敗の切り離し（受け入れ条件 2）
// ---------------------------------------------------------------------------

/// A（B より前）・B（失敗）・C（B より後）・D（B の失敗で空いたスロットに参加）を
/// `max_active = 2` で投入し、流し切ったあとに E を投入する。
fn failure_scenario<M: AutoregressiveModel>(
    inner: &M,
    fail: Fail,
    b: (Tensor<i32>, GenerateConfig),
    c_strategy: SamplingStrategy,
) {
    let schedule: Vec<Req> = vec![
        (0, ids1(&[2, 3]), cfg(8, SamplingStrategy::Greedy, 1)),
        (0, b.0, b.1),
        (0, ids1(&[6, 3, 2]), cfg(9, c_strategy, 2)),
        (0, ids1(&[3, 3]), cfg(7, SamplingStrategy::Greedy, 3)),
        (60, ids1(&[2, 6]), cfg(6, SamplingStrategy::Greedy, 4)),
    ];
    let failing = Failing { inner, fail };

    // 注入が B だけに効く前提（単独 generate でも B のみ Err）。
    for (i, (_, input, c)) in schedule.iter().enumerate() {
        let r = generate(&failing, input, c);
        if i == 1 {
            assert!(matches!(&r, Err(e) if is_injected(e)), "B は注入で失敗する");
            assert!(generate(inner, input, c).is_ok());
        } else {
            assert!(r.is_ok(), "要求 {i} には注入が効かない前提");
        }
    }

    let mut s = BatchScheduler::new(limits(2, 8, 32));
    let o = drive(&failing, &mut s, &schedule);
    assert_eq!(s.queued_len(), 0);
    assert_eq!(s.active_len(), 0);
    assert_eq!(o.finished.len(), 4);
    assert_eq!(o.completed_sum, 4, "失敗は完了数に含めない");
    assert_eq!(o.failed.len(), 1, "take_failed はちょうど 1 件");
    let (fid, err) = &o.failed[0];
    assert_eq!(*fid, o.ids[1]);
    assert!(is_injected(err), "注入マーカーを含む InvalidArgument");
    assert!(s.take_failed().is_empty(), "2 回目の take_failed は空");
    assert!(
        o.finished.iter().all(|(id, _)| *id != o.ids[1]),
        "失敗した B は finished に出ない"
    );
    // 失敗を注入しない内側モデルでの単独 generate と完全一致。
    assert_matches_single(inner, &schedule, &o, &[1]);
}

#[test]
fn decode_failure_isolated_kv() {
    // B は prompt 長 1 → 最初の decode でキャッシュ長 1。他要求は prompt 長 >= 2 の
    // ため decode 時にキャッシュ長 1 を通らない。
    failure_scenario(
        &KvLm,
        Fail::DecodeAtSeqLen(1),
        (ids1(&[5]), cfg(8, SamplingStrategy::Greedy, 9)),
        SamplingStrategy::Temperature(0.9),
    );
}

#[test]
fn decode_failure_isolated_stateless() {
    // B の連鎖 1→4→5: token 5 を decode で受けて失敗。他要求（Greedy）は
    // {2,7,6,3} の巡回のみで 5 を含まないことを下で固定する。
    for p in [&[2, 3][..], &[6, 3, 2], &[3, 3], &[2, 6]] {
        let out =
            data(&generate(&TableModel, &ids1(p), &cfg(12, SamplingStrategy::Greedy, 0)).unwrap());
        assert!(!out.contains(&5), "前提: 他要求の連鎖に token 5 が現れない");
    }
    failure_scenario(
        &TableModel,
        Fail::DecodeToken(5),
        (ids1(&[0, 1]), cfg(8, SamplingStrategy::Greedy, 9)),
        SamplingStrategy::Greedy,
    );
}

#[test]
fn prefill_failure_isolated_kv() {
    failure_scenario(
        &KvLm,
        Fail::PromptContains(7),
        (ids2(&[1, 7, 2]), cfg(8, SamplingStrategy::TopK(3), 9)),
        SamplingStrategy::Temperature(1.0),
    );
}

#[test]
fn prefill_failure_isolated_stateless() {
    failure_scenario(
        &TableModel,
        Fail::PromptContains(7),
        (ids1(&[1, 7, 2]), cfg(8, SamplingStrategy::TopK(3), 9)),
        SamplingStrategy::Greedy,
    );
}

// ---------------------------------------------------------------------------
// §8.3 非依存の検査（受け入れ条件 3）
// ---------------------------------------------------------------------------

const FORBIDDEN: [&str; 7] = [
    "std::thread",
    "std::sync::mpsc",
    "tokio",
    "async ",
    "std::net",
    "std::fs",
    "rayon",
];

/// 禁止リテラルを含む行を `行番号: 内容` で返す（コメント・テストを含む全行が対象。
/// 設計 §8.3 の grep と同じ範囲）。
fn violations(content: &str) -> Vec<String> {
    content
        .lines()
        .enumerate()
        .filter(|(_, l)| FORBIDDEN.iter().any(|w| l.contains(w)))
        .map(|(i, l)| format!("{}: {l}", i + 1))
        .collect()
}

#[test]
fn scheduler_source_has_no_concurrency_or_io_dependency() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/generate/scheduler.rs");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("test fixture: {} が読めない: {e}", path.display()));
    // 正のプローブ: 別ファイル・空ファイルを走査して 0 件で素通りするのを防ぐ。
    assert!(content.contains("pub struct BatchScheduler"));
    assert!(content.contains("pub fn step"));
    let v = violations(&content);
    assert!(
        v.is_empty(),
        "scheduler.rs が §8.3 の禁止依存を含む:\n{}",
        v.join("\n")
    );
}

#[test]
fn violation_detector_catches_each_forbidden_literal() {
    for w in FORBIDDEN {
        let src = format!("fn ok() {{}}\nuse {w}::x;\n");
        assert_eq!(violations(&src).len(), 1, "{w} を検出できない");
    }
    assert!(violations("fn ok() { let asyncx = 1; }\n").is_empty());
}
