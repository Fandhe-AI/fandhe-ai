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
//! バックエンドをまたぐ TopK／Temperature の token 列比較（ホスト `f64` のサンプリングは
//! CDF 境界で logits の微差により反転しうる。設計 §6.2）、B > 1、`num_kv_layers() == 0` の
//! speculative（論点 3）、内部状態保持型モデル（論点 8）、facade 公開。
//!
//! # サンプリング版 speculative（#3102。設計 §20.7）
//!
//! サンプリング版は token 列の一致を契約にも仮説（H1）にもしない。推奨方式に従い、参照側で
//! `generate_speculative` を実行して `forward_step` ごとの入力 id 列を記録し、比較側バックエンドの
//! モデルへ**同じ id 列を再生**して、呼び出しごとの logits だけを REQ-2 統一複合判定で比べる
//! （`replay_logs`・`check_speculative_sampling`。巻き戻しは `seq_before` ごとに保存した
//! `KvCache` の複製へ戻して再現する）。tolerance・baseline は新設しない。実測は
//! `docs/perf/logs/speculative-sampling-3102/README.md`。
//! `Generator` は非暗号の xorshift64* で、生成 token をセキュリティ用途に使わない（OWASP A02 の注記）。
//!
//! # #3107 追記（`step_batched` のバッチ経路）
//!
//! `BatchedAttnLm`（B 対応: `[b, l, V]`・位置は `valid_len(r)`）で `step_batched` を同じ `schedule()`
//! により回し、device 側と参照側の記録を呼び出し数・b・行ごとの `valid_len`・ids・shape まで構造ごと
//! 突き合わせ、logits は REQ-2 統一複合判定で比べる（`check_scheduler_batched`）。同じバックエンドの
//! 単独 `generate` との Greedy 一致（§6.3）も確かめる。対象外: TopK／Temperature のバックエンド横断、
//! B > 1 の speculative。facade 側の CPU 結合テストは `speculative_batching_step_batched_facade.rs`。

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
use std::collections::HashMap;

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

/// サンプリング版の掃引設定（`(名前, 戦略)`。seed は呼び出し側が与える）。
fn sampling_cfgs() -> [(&'static str, SamplingStrategy); 2] {
    [
        ("Temperature(0.8)", SamplingStrategy::Temperature(0.8)),
        ("TopK(3)", SamplingStrategy::TopK(3)),
    ]
}

/// `log`（参照側で記録した `forward_step` 呼び出し列）の入力 id 列を、`make` 側モデルへ順に再生して
/// 呼び出しごとの記録を返す（設計 §20.7）。巻き戻しは、呼び出し前のキャッシュ長が `seq_before` と
/// 違うとき、長さごとに保存した最新の `KvCache` 複製へ戻して再現する（巻き戻し後の長さ以下の履歴は
/// 不変なので最新の複製が正しい状態になる）。検証 forward（`L = k + 1`）と再 forward はそのまま通す。
fn replay_logs(
    make: TapeBackend,
    layers: usize,
    seed_base: i64,
    next: NextFn,
    log: &[Call],
) -> Vec<Call> {
    let model = attn(make, layers, seed_base, next);
    let mut states: HashMap<usize, Vec<KvCache>> = HashMap::new();
    let mut caches: Vec<KvCache> = (0..layers).map(|_| KvCache::new()).collect();
    states.insert(0, caches.clone());
    for (i, call) in log.iter().enumerate() {
        let before = call.seq_before.expect("speculative は KV 付きモデルのみ");
        if caches[0].seq_len() != before {
            caches = states
                .get(&before)
                .unwrap_or_else(|| panic!("呼び出し {i}: 長さ {before} の状態が記録にない"))
                .clone();
        }
        let input = Tensor::new(call.ids.clone(), &[1, call.ids.len()]).expect("fixture");
        model.forward_step(&input, &mut caches).expect("再生");
        states.insert(caches[0].seq_len(), caches.clone());
    }
    model.log.into_inner()
}

/// サンプリング版 speculative の掃引。reference で `generate_speculative` を実行して target・draft の
/// 呼び出し列を記録し、`make` 側へ id 列を再生して logits を呼び出しごとに突合する（token 列は比べない）。
fn check_speculative_sampling(make: TapeBackend, reference: TapeBackend, label: &str) -> SpecStats {
    let mut st = SpecStats {
        runs: 0,
        verify_calls: 0,
        replays: 0,
    };
    for layers in [1usize, 2] {
        for k in [1usize, 3] {
            for prompt_len in [1usize, 3] {
                for max in [prompt_len + k + 1, 12] {
                    for (sname, strategy) in sampling_cfgs() {
                        for seed in [1u64, 2] {
                            for di in 0..4 {
                                let (dname, dseed, dnext) = draft_spec(di);
                                let case = format!(
                                    "{label} {sname} seed={seed} layers={layers} k={k} T={prompt_len} max={max} draft={dname}"
                                );
                                let target = attn(reference, layers, 100, Box::new(target_next));
                                let draft = attn(reference, layers, dseed, dnext);
                                let config = GenerateConfig::new(max, strategy).with_seed(seed);
                                generate_speculative(
                                    &target,
                                    &draft,
                                    &prompt(prompt_len, false),
                                    &config,
                                    &SpeculativeConfig::new(k),
                                )
                                .expect("generate_speculative");
                                let (r_t, r_d) = (target.log.borrow(), draft.log.borrow());

                                let d_t =
                                    replay_logs(make, layers, 100, Box::new(target_next), &r_t);
                                let (_, _, dnext) = draft_spec(di);
                                let d_d = replay_logs(make, layers, dseed, dnext, &r_d);
                                assert_logs_parity(&format!("{case} target"), &d_t, &r_t);
                                assert_logs_parity(&format!("{case} draft"), &d_d, &r_d);

                                st.verify_calls += r_t
                                    .iter()
                                    .filter(|c| c.ids.len() > 1 && c.seq_before != Some(0))
                                    .count();
                                st.replays += r_t
                                    .windows(2)
                                    .filter(|w| w[1].seq_before == w[0].seq_before)
                                    .count();
                                st.runs += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        st.runs >= 256,
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
// イシュー #3107: `step_batched`（テンソル単位バッチ。#3106）のバッチ経路
// ---------------------------------------------------------------------

/// `AttnLm` の B 対応版。入力 `[b, l]` に対し `[b, l, V]` を返し、行 `r` の位置は forward 前の
/// `valid_len(r)` から取る（`seq_len()` は padding を含む物理長のため使わない。対象モデル契約
/// §21.6）。B 非対応のモデルでは `step_batched` が黙って B = 1 へ戻るため、こちらを使う。
struct BatchedAttnLm {
    backend: TapeBackend,
    layers: usize,
    seed_base: i64,
}

impl AutoregressiveModel for BatchedAttnLm {
    fn num_kv_layers(&self) -> usize {
        self.layers
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let shape = new_ids.shape();
        let l = shape[shape.len() - 1];
        let b = shape.iter().product::<usize>() / l;
        let pos0: Vec<usize> = match caches.first() {
            Some(c) => (0..b).map(|r| c.valid_len(r).unwrap_or(0)).collect(),
            None => vec![0; b],
        };
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
        let flat = x.reshape(&[b * l, E])?;
        let lm = LinearVars {
            weight: tape.var(&t(seq(s + 10, E * V), &[E, V])),
            bias: Some(tape.var(&t(seq(s + 9, V), &[V]))),
        };
        let logits = lm.forward(&flat)?.reshape(&[b, l, V])?.to_tensor();
        let mut data = logits.contiguous().host_slice().into_owned();
        let flat_ids = vals(new_ids);
        for r in 0..b {
            for i in 0..l {
                let tok = flat_ids[r * l + i];
                data[(r * l + i) * V + target_next(tok, pos0[r] + i) as usize] += SKIP_C;
            }
        }
        Ok(t(data, &[b, l, V]))
    }
}

/// `step_batched` 用の 1 回の `forward_step` の記録。
struct BatchedCall {
    b: usize,
    l: usize,
    /// 呼び出し前の行ごとの有効長（状態なしモデルは空）。
    valid: Vec<usize>,
    ids: Vec<i32>,
    shape: Vec<usize>,
    logits: Vec<f32>,
}

struct BatchedRecording<M> {
    inner: M,
    log: RefCell<Vec<BatchedCall>>,
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
        let shape = new_ids.shape();
        let l = shape[shape.len() - 1];
        let b = shape.iter().product::<usize>() / l;
        let valid = match caches.first() {
            Some(c) => (0..b).map(|r| c.valid_len(r).unwrap_or(0)).collect(),
            None => Vec::new(),
        };
        let logits = self.inner.forward_step(new_ids, caches)?;
        self.log.borrow_mut().push(BatchedCall {
            b,
            l,
            valid,
            ids: vals(new_ids),
            shape: logits.shape().to_vec(),
            logits: logits.contiguous().host_slice().into_owned(),
        });
        Ok(logits)
    }
}

fn batched_attn(
    backend: TapeBackend,
    layers: usize,
    seed_base: i64,
) -> BatchedRecording<BatchedAttnLm> {
    BatchedRecording {
        inner: BatchedAttnLm {
            backend,
            layers,
            seed_base,
        },
        log: RefCell::new(Vec::new()),
    }
}

/// `drive` の `step_batched` 版（同じ計画・同じ上限）。
fn drive_batched<M: AutoregressiveModel>(m: &M, reqs: &[Req]) -> Vec<(Vec<usize>, Vec<i32>)> {
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
        s.step_batched(m).expect("step_batched は Ok を返し続ける");
        finished.extend(s.take_finished());
        assert!(s.take_failed().is_empty(), "失敗した要求がある");
    }
    panic!("TICK_CAP に到達した（スケジューラが掃けない）");
}

/// 空振り防止: 束ねた呼び出し（`b > 1 && l == 1`）の件数。0 件なら B = 1 へ戻っている。
fn batched_call_count(log: &[BatchedCall]) -> usize {
    log.iter().filter(|c| c.b > 1 && c.l == 1).count()
}

/// 行ごとの有効長が揃わない（padding を伴う）呼び出しの件数。
fn padded_call_count(log: &[BatchedCall]) -> usize {
    log.iter()
        .filter(|c| c.valid.iter().any(|&v| v != c.valid[0]))
        .count()
}

/// 記録全行の margin 検査（`assert_margin` と同じ導出。新しい tolerance ではない）。
fn assert_batched_margin(label: &str, log: &[BatchedCall]) {
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
                "{label}: フィクスチャ不正（margin 前提を満たさない）呼び出し {ci} 行 {ri}: margin={margin}"
            );
        }
    }
}

/// `step_batched`（Greedy）を `make` と `reference` で突合する。`layers` = 2（padding を伴う束ね）と
/// 0（状態なし）を通す。呼び出し数・b・行ごとの `valid_len`・ids・shape の構造一致が、片側だけ
/// B = 1 へ戻った場合の検出器を兼ねる。
fn check_scheduler_batched(make: TapeBackend, reference: TapeBackend, label: &str) {
    for layers in [2usize, 0] {
        let case = format!("{label} step_batched layers={layers}");
        let reqs = schedule();
        let r_model = batched_attn(reference, layers, 100);
        let d_model = batched_attn(make, layers, 100);
        let r_out = drive_batched(&r_model, &reqs);
        assert_batched_margin(&case, &r_model.log.borrow());
        let d_out = drive_batched(&d_model, &reqs);

        // H1（バックエンド横断版。破れても判定は緩めず設計 §10 論点 2 として承認依頼に戻す）。
        assert_eq!(
            d_out, r_out,
            "{case}: 出力の shape／token 列（事前登録の仮説 H1 の破れ。判定を緩めず設計 §10 論点 2 として承認依頼に戻すこと）"
        );
        let (dl, rl) = (d_model.log.borrow(), r_model.log.borrow());
        assert_eq!(dl.len(), rl.len(), "{case}: forward_step の呼び出し数");
        for (i, (d, r)) in dl.iter().zip(rl.iter()).enumerate() {
            assert_eq!((d.b, d.l), (r.b, r.l), "{case}: 呼び出し {i} の (b, l)");
            assert_eq!(d.valid, r.valid, "{case}: 呼び出し {i} の valid_len");
            assert_eq!(d.ids, r.ids, "{case}: 呼び出し {i} の ids");
            assert_eq!(d.shape, r.shape, "{case}: 呼び出し {i} の logits shape");
            assert_parity(&format!("{case} 呼び出し {i}"), &d.logits, &r.logits);
        }
        assert!(
            batched_call_count(&dl) >= 1,
            "{case}: device 側が束ねていない"
        );
        assert!(batched_call_count(&rl) >= 1, "{case}: 参照側が束ねていない");
        if layers > 0 {
            assert!(
                padded_call_count(&dl) >= 1,
                "{case}: padding を伴う呼び出しがない"
            );
        }

        // §6.3: 各要求の出力は同一バックエンドの単独 generate と token 列一致する
        // （Greedy＋margin。デバイスのカーネル決定性に依存するため、失敗は所見として記録する）。
        for (i, (_, input, c)) in reqs.iter().enumerate() {
            let single = batched_attn(make, layers, 100);
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

#[test]
fn cpu_backend_ops_matches_naive_ops_for_scheduler_batched_greedy() {
    check_scheduler_batched(TapeBackend::Cpu, TapeBackend::Naive, "cpu-vs-naive");
}

#[test]
fn cpu_backend_ops_matches_naive_ops_for_speculative_sampling() {
    check_speculative_sampling(TapeBackend::Cpu, TapeBackend::Naive, "cpu-vs-naive");
}

/// 再生ハーネスの検出力: 記録の id を 1 つ書き換える、または logits を摂動すると判定が落ちる。
#[test]
fn replay_comparison_detects_tampering() {
    let target = attn(TapeBackend::Naive, 1, 100, Box::new(target_next));
    let draft = attn(TapeBackend::Naive, 1, 300, periodic_next(3));
    let config = GenerateConfig::new(10, SamplingStrategy::Temperature(0.8)).with_seed(1);
    generate_speculative(
        &target,
        &draft,
        &prompt(2, false),
        &config,
        &SpeculativeConfig::new(3),
    )
    .expect("generate_speculative");
    let log = target.log.borrow();
    let flat =
        |calls: &[Call]| -> Vec<f32> { calls.iter().flat_map(|c| c.logits.clone()).collect() };

    // 無改変の再生は合格する（基準）。
    let ok = replay_logs(TapeBackend::Naive, 1, 100, Box::new(target_next), &log);
    assert!(compare(&flat(&ok), &flat(&log)).expect("compare").passes());

    // id の書き換え（最後の呼び出しの末尾 id を別トークンへ）。
    let mut tampered: Vec<Call> = log
        .iter()
        .map(|c| Call {
            seq_before: c.seq_before,
            ids: c.ids.clone(),
            shape: c.shape.clone(),
            logits: c.logits.clone(),
        })
        .collect();
    let last = tampered.last_mut().expect("非空");
    let n = last.ids.len();
    last.ids[n - 1] = (last.ids[n - 1] + 1) % V as i32;
    let replayed = replay_logs(TapeBackend::Naive, 1, 100, Box::new(target_next), &tampered);
    assert!(
        !compare(&flat(&replayed), &flat(&log))
            .expect("compare")
            .passes()
    );

    // 参照 logits の摂動。
    let mut perturbed = flat(&log);
    perturbed[0] += 1.0e-2;
    assert!(!compare(&flat(&ok), &perturbed).expect("compare").passes());
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

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/speculative-sampling-3102/README.md 参照"]
fn cuda_speculative_sampling_matches_cpu() {
    check_speculative_sampling(TapeBackend::Cuda, TapeBackend::Cpu, "cuda");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/speculative-sampling-3102/README.md 参照"]
fn metal_speculative_sampling_matches_cpu() {
    check_speculative_sampling(TapeBackend::Metal, TapeBackend::Cpu, "metal");
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

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/speculative-batching-facade-2934/README.md 参照"]
fn cuda_scheduler_batched_greedy_matches_cpu() {
    check_scheduler_batched(TapeBackend::Cuda, TapeBackend::Cpu, "cuda");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/speculative-batching-facade-2934/README.md 参照"]
fn metal_scheduler_batched_greedy_matches_cpu() {
    check_scheduler_batched(TapeBackend::Metal, TapeBackend::Cpu, "metal");
}
