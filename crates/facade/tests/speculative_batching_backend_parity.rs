//! speculative decoding（`generate_speculative`。#2886）と連続バッチング
//! スケジューラ（`BatchScheduler`。#2888）の、CUDA／Metal 実機と CPU の
//! parity テスト（イシュー #2890。設計記録
//! `docs/facade-speculative-decoding-batching-design.md` §6.1・§6.3・§6.4・§11 行 10）。
//!
//! 両機能は #2934 で facade `fandhe_ai::inference` へ純再エクスポート済み
//! （結合テストは `speculative_batching_facade.rs`）。ただし facade には `BackendOps` を注入する
//! 経路が無い（REQ-12）ため、`generate_backend_parity.rs`・`kv_cache_backend_parity.rs` と同じ位置づけで
//! facade のテストコードから内部クレート `fandhe_ai_autodiff` を直接使う。
//! CPU 上の一致は #2887（speculative）・#2889（scheduler）が autodiff 側で済ませている。
//!
//! - 属性なし: `CpuBackendOps` と `Tape::new()`（NaiveOps）を同じハーネスで突合する。
//!   ハーネスを CI で動かし、腐敗と dead_code を防ぐ。
//! - `#[ignore]`: CUDA（DGX Spark GB10）／Metal（`cfg(target_os = "macos")` 限定。
//!   Linux ではコンパイルもされない）と CPU を突合する。測定手順と記入欄は
//!   `docs/perf/logs/speculative-batching-2890/README.md`。
//!
//! # 判定
//!
//! 契約は各 `forward_step` 呼び出しの logits に対する REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`。相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満）。検証 forward（`L_new = k + 1`）と rewind 後の再 forward を
//! 実デバイス上でそのまま通すため、`L_new = 1` への再生は使わない。bit 一致は契約にしない
//! （設計 §6.1）。tolerance・baseline・係数は新設しない。
//!
//! # 事前登録の仮説 H1
//!
//! 全行で `top1 - top2` が REQ-2 定数から導いた下限を超える入力（`assert_margin`）では、
//! デバイス側の token 列も CPU 側と一致する。H1 は契約ではなく仮説であり、破れた場合は
//! 判定を緩めず事実を記録して設計 §10 論点 2 として承認依頼に戻す。`SKIP_C` は #2887 と
//! 同値の固定フィクスチャ定数で、結果を見て調整しない。
//!
//! # 対象外
//!
//! バックエンドをまたぐ TopK／Temperature の比較（ホスト `f64` のサンプリングは CDF 境界で
//! logits の微差により反転しうる。設計 §6.2）、サンプリング版 speculative（論点 1）、
//! B > 1、`num_kv_layers() == 0` の speculative（論点 3）、内部状態保持型モデル（論点 8）、
//! facade 公開。`Generator` は非暗号の xorshift64* で、本テストは Greedy のみのため乱数を
//! 使わない（OWASP A02 の注記）。

use fandhe_ai_autodiff::generate::scheduler::{BatchScheduler, SchedulerLimits};
use fandhe_ai_autodiff::generate::speculative::{SpeculativeConfig, generate_speculative};
use fandhe_ai_autodiff::generate::{
    AutoregressiveModel, GenerateConfig, SamplingStrategy, generate,
};
use fandhe_ai_autodiff::nn::{KvCache, LinearVars, MultiheadAttentionVars};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_backend_cpu::parity::{
    ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE, assert_parity, compare,
};
use fandhe_ai_tensor_core::Tensor;
use std::cell::RefCell;

const V: usize = 7;
const E: usize = 4;
const HEADS: usize = 2;
/// 上位 2 候補の margin を数値揺らぎより十分大きく取るためのフィクスチャ定数
/// （判定閾値ではない。#2887 と同値。実行前に固定し結果を見て調整しない）。
const SKIP_C: f32 = 3.0;
/// スケジューラ駆動ループの上限（ハング防止の fail-closed）。
const TICK_CAP: usize = 256;

type NextFn = Box<dyn Fn(i32, usize) -> i32>;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture: shape とデータ長は事前に一致させている")
}

fn seq(seed: i64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((seed + i as i64 * 7) % 13 - 6) as f32 * 0.05)
        .collect()
}

fn target_next(tok: i32, pos: usize) -> i32 {
    ((3 * tok as usize + pos + 1) % V) as i32
}

fn wrong_next(tok: i32, pos: usize) -> i32 {
    (target_next(tok, pos) + 1) % V as i32
}

fn periodic_next(period: usize) -> NextFn {
    Box::new(move |tok, pos| {
        if pos.is_multiple_of(period) {
            wrong_next(tok, pos)
        } else {
            target_next(tok, pos)
        }
    })
}

fn prompt(len: usize, rank1: bool) -> Tensor<i32> {
    let v: Vec<i32> = (0..len).map(|i| (i % V) as i32).collect();
    let shape: &[usize] = if rank1 { &[len] } else { &[1, len] };
    Tensor::new(v, shape).expect("fixture")
}

fn vals(x: &Tensor<i32>) -> Vec<i32> {
    x.contiguous().host_slice().into_owned()
}

fn cfg(max_length: usize) -> GenerateConfig {
    GenerateConfig::new(max_length, SamplingStrategy::Greedy)
}

/// `forward_step` ごとに新規 [`Tape`] をどのバックエンドで作るかを選ぶ
/// （`generate_backend_parity.rs` と同型）。
#[derive(Clone, Copy)]
enum TapeBackend {
    /// `Tape::new()`（ホスト参照実装 NaiveOps）。
    Naive,
    /// `Tape::new_with_ops(Box::new(CpuBackendOps::new()))`。
    Cpu,
    /// `cfg(target_os = "macos")` 限定。Metal 実機必須。
    #[cfg(target_os = "macos")]
    Metal,
    /// CUDA 実機必須。
    Cuda,
}

impl TapeBackend {
    fn new_tape(self) -> Tape {
        match self {
            TapeBackend::Naive => Tape::new(),
            TapeBackend::Cpu => {
                Tape::new_with_ops(Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()))
            }
            #[cfg(target_os = "macos")]
            TapeBackend::Metal => {
                Tape::new_with_ops(Box::new(fandhe_ai_backend_metal::MetalBackendOps::new()))
            }
            TapeBackend::Cuda => {
                Tape::new_with_ops(Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(0)))
            }
        }
    }
}

/// `Embedding → MHA（KV キャッシュ付き・層ごとに直列） → lm head` に、margin 確保用の
/// skip 項 `SKIP_C · onehot(next(tok, pos))` をホスト側で足すモデル。`layers == 0` の
/// ときは MHA を持たない状態なしモデル（`num_kv_layers() == 0`。スケジューラ用。
/// 表引きと違い tape を通るためデバイス比較の意味がある）。
struct AttnLm {
    backend: TapeBackend,
    layers: usize,
    seed_base: i64,
    next: NextFn,
}

impl AutoregressiveModel for AttnLm {
    fn num_kv_layers(&self) -> usize {
        self.layers
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let pos0 = caches.first().map_or(0, |c| c.seq_len());
        let s = self.seed_base;
        let tape = self.backend.new_tape();
        let emb = tape.var(&t(seq(s + 1, V * E), &[V, E]));
        let mut x = emb.embedding(new_ids, None)?;
        for (li, cache) in caches.iter_mut().enumerate() {
            let o = s + 20 * li as i64;
            let lin = |d: i64| LinearVars {
                weight: tape.var(&t(seq(o + d, E * E), &[E, E])),
                bias: Some(tape.var(&t(seq(o + d + 1, E), &[E]))),
            };
            let mha = MultiheadAttentionVars::new(HEADS, lin(2), lin(4), lin(6), lin(8))?;
            x = mha.forward_with_cache(&x, &x, &x, cache)?;
        }
        let l = new_ids.shape()[new_ids.shape().len() - 1];
        let flat = x.reshape(&[l, E])?;
        let lm = LinearVars {
            weight: tape.var(&t(seq(s + 10, E * V), &[E, V])),
            bias: Some(tape.var(&t(seq(s + 9, V), &[V]))),
        };
        let logits = lm.forward(&flat)?.reshape(&[1, l, V])?.to_tensor();
        let mut data = logits.contiguous().host_slice().into_owned();
        for (i, &tok) in vals(new_ids).iter().enumerate() {
            data[i * V + (self.next)(tok, pos0 + i) as usize] += SKIP_C;
        }
        Ok(t(data, &[1, l, V]))
    }
}

fn attn(backend: TapeBackend, layers: usize, seed_base: i64, next: NextFn) -> Recording<AttnLm> {
    Recording::new(AttnLm {
        backend,
        layers,
        seed_base,
        next,
    })
}

/// 1 回の `forward_step` 呼び出しの記録。
struct Call {
    /// 呼び出し前の `caches[0].seq_len()`（キャッシュ無しモデルは `None`）。
    seq_before: Option<usize>,
    ids: Vec<i32>,
    shape: Vec<usize>,
    logits: Vec<f32>,
}

/// 任意の [`AutoregressiveModel`] を包み、各 `forward_step` の入出力を記録する。
struct Recording<M> {
    inner: M,
    log: RefCell<Vec<Call>>,
}

impl<M> Recording<M> {
    fn new(inner: M) -> Recording<M> {
        Recording {
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
        let seq_before = caches.first().map(|c| c.seq_len());
        let logits = self.inner.forward_step(new_ids, caches)?;
        self.log.borrow_mut().push(Call {
            seq_before,
            ids: vals(new_ids),
            shape: logits.shape().to_vec(),
            logits: logits.contiguous().host_slice().into_owned(),
        });
        Ok(logits)
    }
}

/// 入力の妥当性検査（新しい tolerance ではない）。記録された全行で `top1 - top2` が
/// 十分大きいことを確かめる。導出は `nn_generate_speculative.rs::assert_margin` と同じ:
/// 2 行の各要素が REQ-2 を満たすなら `|a-b| < rtol·M/(1-rtol)` または `< abs`。
/// `top1 - top2 > 2·bound` なら argmax は反転しえない。既存 REQ-2 定数の帰結で、
/// 新しい数値は作らない。
fn assert_margin(label: &str, log: &[Call]) {
    for (ci, call) in log.iter().enumerate() {
        for (ri, row) in call.logits.chunks(V).enumerate() {
            let mut sorted = row.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).expect("fixture: logits は有限"));
            let m = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs())) as f64;
            let bound = (RELATIVE_TOLERANCE * m / (1.0 - RELATIVE_TOLERANCE))
                .max(ABSOLUTE_RESCUE_THRESHOLD);
            let margin = (sorted[0] - sorted[1]) as f64;
            assert!(
                margin > 2.0 * bound,
                "{label}: フィクスチャ不正（margin 前提を満たさない）呼び出し {ci} 行 {ri}: \
                 margin={margin} 必要={}",
                2.0 * bound
            );
        }
    }
}

/// 2 つの記録の構造（呼び出し数・`seq_before`・ids・logits shape）が一致し、
/// 呼び出しごとの logits が REQ-2 統一複合判定を満たすことを確かめる。比較した呼び出し数を返す。
fn assert_logs_parity(label: &str, dev: &[Call], reference: &[Call]) -> usize {
    assert_eq!(
        dev.len(),
        reference.len(),
        "{label}: forward_step の呼び出し数が一致しない"
    );
    for (i, (d, r)) in dev.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            d.seq_before, r.seq_before,
            "{label}: 呼び出し {i} の seq_before"
        );
        assert_eq!(d.ids, r.ids, "{label}: 呼び出し {i} の ids");
        assert_eq!(d.shape, r.shape, "{label}: 呼び出し {i} の logits shape");
        assert_parity(&format!("{label} 呼び出し {i}"), &d.logits, &r.logits);
    }
    dev.len()
}

/// 掃引の実行統計（空判定の防止に使う）。
struct SpecStats {
    runs: usize,
    verify_calls: usize,
    replays: usize,
}

/// draft の掃引点。`(名前, seed_base, 規則)`。
fn draft_spec(i: usize) -> (&'static str, i64, NextFn) {
    match i {
        0 => ("全受理(同一重み)", 100, Box::new(target_next)),
        1 => ("全受理(別重み)", 300, Box::new(target_next)),
        2 => ("全棄却", 300, Box::new(wrong_next)),
        _ => ("部分受理(周期 3)", 300, periodic_next(3)),
    }
}

/// speculative の掃引を `make`（比較側）と `reference` の両方で実行して突合する。
fn check_speculative(make: TapeBackend, reference: TapeBackend, label: &str) -> SpecStats {
    let mut st = SpecStats {
        runs: 0,
        verify_calls: 0,
        replays: 0,
    };
    for layers in [1usize, 2] {
        for k in [1usize, 3] {
            for prompt_len in [1usize, 3] {
                for max in [prompt_len + k + 1, 12] {
                    let ids = prompt(prompt_len, false);
                    // 参照 generate（Greedy）。margin 前提を先に評価する。
                    let ref_gen = attn(reference, layers, 100, Box::new(target_next));
                    let gen_out = generate(&ref_gen, &ids, &cfg(max)).expect("参照 generate");
                    assert_margin(
                        &format!("{label} generate layers={layers} T={prompt_len} max={max}"),
                        &ref_gen.log.borrow(),
                    );
                    for di in 0..4 {
                        let name = draft_spec(di).0;
                        let case = format!(
                            "{label} layers={layers} k={k} T={prompt_len} max={max} draft={name}"
                        );
                        let run = |backend: TapeBackend| {
                            let (_, dseed, dnext) = draft_spec(di);
                            let target = attn(backend, layers, 100, Box::new(target_next));
                            let draft = attn(backend, layers, dseed, dnext);
                            let out = generate_speculative(
                                &target,
                                &draft,
                                &ids,
                                &cfg(max),
                                &SpeculativeConfig::new(k),
                            )
                            .expect("generate_speculative");
                            (out, target, draft)
                        };
                        let (r_out, r_target, r_draft) = run(reference);
                        assert_margin(&format!("{case} target"), &r_target.log.borrow());
                        assert_margin(&format!("{case} draft"), &r_draft.log.borrow());
                        assert_eq!(vals(&r_out), vals(&gen_out), "{case}: 参照側 speculative");
                        let (d_out, d_target, d_draft) = run(make);

                        // 仮説 H1（バックエンド横断版）。
                        let h1 = "事前登録の仮説 H1 の破れ。判定を緩めず設計 §10 論点 2 として承認依頼に戻すこと";
                        assert_eq!(d_out.shape(), r_out.shape(), "{case}: shape（{h1}）");
                        assert_eq!(vals(&d_out), vals(&r_out), "{case}: token 列（{h1}）");

                        // 契約: 呼び出しごとの logits の REQ-2 判定。
                        let (dt, rt) = (d_target.log.borrow(), r_target.log.borrow());
                        assert_logs_parity(&format!("{case} target"), &dt, &rt);
                        assert_logs_parity(
                            &format!("{case} draft"),
                            &d_draft.log.borrow(),
                            &r_draft.log.borrow(),
                        );
                        st.verify_calls += dt
                            .iter()
                            .filter(|c| c.ids.len() > 1 && c.seq_before != Some(0))
                            .count();
                        st.replays += dt
                            .windows(2)
                            .filter(|w| w[1].seq_before == w[0].seq_before)
                            .count();
                        st.runs += 1;
                    }
                }
            }
        }
    }
    assert!(
        st.runs >= 60,
        "{label}: 掃引が縮退している（runs={})",
        st.runs
    );
    assert!(
        st.verify_calls > 0,
        "{label}: 検証 forward（L_new > 1）が比較されていない"
    );
    assert!(
        st.replays > 0,
        "{label}: 再 forward（rewind-replay）が 1 度も起きていない"
    );
    st
}

type Req = (usize, Tensor<i32>, GenerateConfig);

fn ids1(v: &[i32]) -> Tensor<i32> {
    Tensor::new(v.to_vec(), &[v.len()]).expect("fixture")
}

fn ids2(v: &[i32]) -> Tensor<i32> {
    Tensor::new(v.to_vec(), &[1, v.len()]).expect("fixture")
}

/// 途中参加・スロット再利用（`max_active` 2 < 要求数 4）・rank 混在・`max_length` 差を含む。
fn schedule() -> Vec<Req> {
    vec![
        (0, ids1(&[1, 2, 3]), cfg(9)),
        (0, ids2(&[3, 1, 4, 1]), cfg(11)),
        (2, ids1(&[5]), cfg(8)),
        (4, ids2(&[2, 6]), cfg(10)),
    ]
}

/// `schedule` を指定 tick で submit しつつ全要求が掃けるまで `step` を回し、
/// submit 順の出力（shape, token 列）を返す。失敗した要求があれば panic する。
fn drive<M: AutoregressiveModel>(m: &M, reqs: &[Req]) -> Vec<(Vec<usize>, Vec<i32>)> {
    let limits = SchedulerLimits::new(2, 4, 16).expect("fixture");
    let mut s = BatchScheduler::new(limits);
    let mut ids = Vec::new();
    let mut finished = Vec::new();
    for tick in 0..TICK_CAP {
        for (at, input, c) in reqs {
            if *at == tick {
                ids.push(s.submit(input, c).expect("submit"));
            }
        }
        if ids.len() == reqs.len() && s.queued_len() == 0 && s.active_len() == 0 {
            assert!(s.take_failed().is_empty(), "失敗した要求がある");
            return ids
                .iter()
                .map(|id| {
                    let (_, out) = finished
                        .iter()
                        .find(|(fid, _): &&(_, Tensor<i32>)| fid == id)
                        .expect("要求が finished にない");
                    (out.shape().to_vec(), vals(out))
                })
                .collect();
        }
        s.step(m).expect("step は Ok を返し続ける");
        finished.extend(s.take_finished());
        assert!(s.take_failed().is_empty(), "失敗した要求がある");
    }
    panic!("TICK_CAP に到達した（スケジューラが掃けない）");
}

/// スケジューラ（Greedy）を `make` と `reference` で突合する。KV モデル（2 層）と
/// 状態なしモデルの両方を通す。
fn check_scheduler(make: TapeBackend, reference: TapeBackend, label: &str) {
    for layers in [2usize, 0] {
        let case = format!("{label} scheduler layers={layers}");
        let reqs = schedule();
        let r_model = attn(reference, layers, 100, Box::new(target_next));
        let d_model = attn(make, layers, 100, Box::new(target_next));
        let r_out = drive(&r_model, &reqs);
        assert_margin(&case, &r_model.log.borrow());
        let d_out = drive(&d_model, &reqs);

        // H1（バックエンド横断版）。
        assert_eq!(
            d_out, r_out,
            "{case}: 出力の shape／token 列（事前登録の仮説 H1 の破れ。判定を緩めず設計 §10 論点 2 として承認依頼に戻すこと）"
        );
        let compared = assert_logs_parity(&case, &d_model.log.borrow(), &r_model.log.borrow());
        assert!(compared > 0, "{case}: 比較した呼び出しが 0 件");

        // §6.3: 各要求の出力は同一バックエンドの単独 generate と token 列一致する
        // （デバイスのカーネル決定性に依存するため、失敗は所見として記録する）。
        for (i, (_, input, c)) in reqs.iter().enumerate() {
            let single = attn(make, layers, 100, Box::new(target_next));
            let want = generate(&single, input, c).expect("単独 generate");
            assert_eq!(
                d_out[i].0,
                want.shape().to_vec(),
                "{case}: 要求 {i} shape（単独実行）"
            );
            assert_eq!(
                d_out[i].1,
                vals(&want),
                "{case}: 要求 {i} token 列（単独実行）"
            );
        }
    }
}

// ---------------------------------------------------------------------
// 属性なし（CI で実行。ハーネスの腐敗と dead_code を防ぐ）
// ---------------------------------------------------------------------

#[test]
fn cpu_backend_ops_matches_naive_ops_for_speculative_greedy() {
    check_speculative(TapeBackend::Cpu, TapeBackend::Naive, "cpu-vs-naive");
}

#[test]
fn cpu_backend_ops_matches_naive_ops_for_scheduler_greedy() {
    check_scheduler(TapeBackend::Cpu, TapeBackend::Naive, "cpu-vs-naive");
}

/// 突合が実際に効くこと: 重みの異なるモデルの logits は REQ-2 判定で不合格になる。
#[test]
fn comparison_detects_logits_divergence() {
    let ids = prompt(3, false);
    let flat = |seed: i64| -> Vec<f32> {
        let m = attn(TapeBackend::Naive, 1, seed, Box::new(target_next));
        generate(&m, &ids, &cfg(6)).expect("generate");
        m.log
            .borrow()
            .iter()
            .flat_map(|c| c.logits.clone())
            .collect()
    };
    let (a, b) = (flat(100), flat(300));
    assert_eq!(a.len(), b.len());
    assert!(!compare(&a, &b).expect("compare").passes());
}

// ---------------------------------------------------------------------
// 実機横断（`#[ignore]`）
// ---------------------------------------------------------------------

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/speculative-batching-2890/README.md 参照"]
fn cuda_speculative_greedy_matches_cpu() {
    check_speculative(TapeBackend::Cuda, TapeBackend::Cpu, "cuda");
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/speculative-batching-2890/README.md 参照"]
fn cuda_scheduler_greedy_matches_cpu() {
    check_scheduler(TapeBackend::Cuda, TapeBackend::Cpu, "cuda");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/speculative-batching-2890/README.md 参照"]
fn metal_speculative_greedy_matches_cpu() {
    check_speculative(TapeBackend::Metal, TapeBackend::Cpu, "metal");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/speculative-batching-2890/README.md 参照"]
fn metal_scheduler_greedy_matches_cpu() {
    check_scheduler(TapeBackend::Metal, TapeBackend::Cpu, "metal");
}
