//! facade `fandhe_ai::inference::generate_speculative` のサンプリング版（`TopK`／`Temperature`）の
//! 決定的シード結合テスト（イシュー #3102・親 #3099）。判定契約の正は設計記録
//! `docs/facade-speculative-decoding-batching-design.md` §20.5（C1〜C6）・§20.6（分布検定は契約外）・
//! §25.2（RNG 消費順）。#3101 が autodiff 側のユニットテストで C1・C3〜C6 を、
//! `crates/autodiff/tests/nn_generate_speculative.rs` で C2 を済ませており、本ファイルは同じ契約を
//! **facade の公開面だけ**（`fandhe_ai::…`）で固定する。内部クレートは import しない。
//!
//! # 判定方式（新しい数値判定を作らない）
//!
//! 判定はすべて完全一致（token 列・RNG 値・記録した呼び出し列）。tolerance・baseline・χ² 等の
//! 閾値つき統計量は導入しない（§20.6。分布検定は選択肢 A として契約外）。
//!
//! 受理確率・残差分布の突合は「厳密入力の再生オラクル」で行う。`SetLm` は logits を
//! `S(tok, pos)` 上で `0.0`、それ以外で `-1.0e4` とし、`S` の大きさは 2 のべき乗（1・2・4・8）に限る。
//! `exp(0) = 1`・`exp(-1e4 / t) = 0`（`t` が任意でも）・`1/m`（`m` は 2 のべき乗）は libm や計算順序に
//! よらず厳密なので、実装が `Generator` へ渡す `f32` 値（サンプリング重み・受理確率 0／0.5／1・残差重み）は
//! プラットフォームをまたいで bit 単位で一意に決まる。`reference_speculative` は同じ `Generator` を
//! §25.2 の消費順で引く独立実装（確率は集合の定義から `f64` で直接導く）で、`generate_speculative` の
//! 出力と token 列で完全一致することを固定掃引で要求する。受理確率・残差分布・`q` を draft 時の重みから
//! 取ること・巻き戻し後の位置・消費順のいずれかがずれれば、32 通りを超える掃引のどこかで token 列が分岐する。
//!
//! # 隔離
//!
//! C2（グローバル RNG 非消費）はプロセス共有の RNG を観測するため、別テストバイナリに置き、
//! かつファイル内の全テストが `GLOBAL_RNG_LOCK` を取って直列化する
//! （`compat_sequential_dropout_embedding_bag.rs` の教訓と同型）。
//!
//! 乱数は独立した `Generator`（xorshift64*。非暗号）で、生成 token をセキュリティ用途に使わない（OWASP A02）。
//! CPU のみで動き、外部通信・実機・ファイル入出力は無い。実機（CUDA／Metal）の logits 突合は
//! `speculative_batching_backend_parity.rs`（`#[ignore]` 分）が担う。

use fandhe_ai::inference::{
    AutoregressiveModel, GenerateConfig, SamplingStrategy, SpeculativeConfig, generate,
    generate_speculative,
};
use fandhe_ai::nn::kv_cache::{KvCache, MultiheadAttentionConfig, StatefulAttention};
use fandhe_ai::{AutodiffError, Generator, LinearVars, Tensor};
use std::cell::RefCell;
use std::sync::{Mutex, MutexGuard};

const V: usize = 8;
const E: usize = 4;
const HEADS: usize = 2;
/// 集合の外の logits（有限値。`exp(-1e4 / t)` は `f64` で厳密に 0 へ潰れる）。
const GAP: f32 = -1.0e4;
/// margin 確保用の skip 項（`AttnLm`。判定閾値ではないフィクスチャ定数。`speculative_batching_facade.rs` と同値）。
const SKIP_C: f32 = 3.0;

/// ファイル内の全テストが先頭で取る直列化ロック（C2 の隠れたグローバル RNG 消費による揺れを防ぐ）。
fn global_rng_lock() -> MutexGuard<'static, ()> {
    static GLOBAL_RNG_LOCK: Mutex<()> = Mutex::new(());
    GLOBAL_RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

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

/// テスト側のサンプリング設定（`SamplingStrategy` を直接掃引せず、再生オラクルへも渡せる形にする）。
#[derive(Clone, Copy, Debug)]
enum Strat {
    Temperature(f32),
    TopK(usize),
}

impl Strat {
    fn config(self, max_length: usize, seed: u64) -> GenerateConfig {
        let s = match self {
            Strat::Temperature(x) => SamplingStrategy::Temperature(x),
            Strat::TopK(k) => SamplingStrategy::TopK(k),
        };
        GenerateConfig::new(max_length, s).with_seed(seed)
    }
}

// ---------------------------------------------------------------------
// フィクスチャ: 集合一様モデル
// ---------------------------------------------------------------------

/// 次トークン分布の台（`S(tok, pos)`）を返す規則。
type SetFn = fn(i32, usize) -> Vec<usize>;
/// `AttnLm` の次トークン規則。
type NextRule = fn(i32, usize) -> usize;

fn base(tok: i32, pos: usize) -> usize {
    (3 * tok as usize + pos) % V
}

/// target 規則: `{(a + j) % 8 : j = 0..4}`（大きさ 4）。
fn rule_target(tok: i32, pos: usize) -> Vec<usize> {
    let a = base(tok, pos);
    (0..4).map(|j| (a + j) % V).collect()
}

/// draft (ii): target の台に含まれる大きさ 2 の部分集合（`p = 1/4`・`q = 1/2` で `a = 0.5`）。
fn rule_subset2(tok: i32, pos: usize) -> Vec<usize> {
    let a = base(tok, pos);
    vec![a % V, (a + 1) % V]
}

/// draft (iii): 2 点だけずらした大きさ 4（重なる 2 点で `a = 1`、外れる点で `a = 0`）。
fn rule_shift2(tok: i32, pos: usize) -> Vec<usize> {
    let a = base(tok, pos);
    (0..4).map(|j| (a + 2 + j) % V).collect()
}

/// draft (iv): 全語彙一様（大きさ 8。`p/q = 2` を clamp して 1、台の外は 0）。
fn rule_all(_tok: i32, _pos: usize) -> Vec<usize> {
    (0..V).collect()
}

/// draft (v): target の台と交わらない 1 点（常に棄却）。
fn rule_disjoint(tok: i32, pos: usize) -> Vec<usize> {
    vec![(base(tok, pos) + 5) % V]
}

/// 大きさ 1 の target 規則（one-hot。C5 用）。
fn rule_one(tok: i32, pos: usize) -> Vec<usize> {
    vec![(3 * tok as usize + pos + 1) % V]
}

/// 大きさ 1 の外れ規則（`rule_one` の次の語彙）。
fn rule_one_wrong(tok: i32, pos: usize) -> Vec<usize> {
    vec![(3 * tok as usize + pos + 2) % V]
}

/// 位置が 3 の倍数のときだけ外す大きさ 1 の規則（部分受理を起こす）。
fn rule_one_periodic(tok: i32, pos: usize) -> Vec<usize> {
    if pos.is_multiple_of(3) {
        rule_one_wrong(tok, pos)
    } else {
        rule_one(tok, pos)
    }
}

const DRAFT_RULES: [(&str, SetFn); 5] = [
    ("(i) target と同一", rule_target),
    ("(ii) 部分集合 2 点", rule_subset2),
    ("(iii) 2 点ずらし", rule_shift2),
    ("(iv) 全語彙一様", rule_all),
    ("(v) 交わらない 1 点", rule_disjoint),
];

/// 1 回の `forward_step` の記録（`seq_before` は呼び出し前の `caches[0].seq_len()`）。
struct Call {
    seq_before: usize,
    len: usize,
}

/// logits が `S(tok, pos)` 上 `0.0`・それ以外 `GAP` の KV 付きモデル（位置は `caches[0].seq_len()`。
/// 巻き戻し後の再生が位置を狂わせれば分布がずれる）。`caches` は facade 公開面の
/// `stateful_attention_forward_with_cache` で進める。
struct SetLm {
    sas: Vec<StatefulAttention>,
    rule: SetFn,
    /// logits の語彙サイズ（`> V` なら余りは `GAP`。語彙サイズ不一致の検査用）。
    vocab: usize,
    /// prefill（`seq_before == 0`）の最終行へ NaN を入れる。
    nan_on_prefill: bool,
    /// decode（`seq_before > 0` かつ 1 token）へ NaN を入れる。
    nan_on_decode: bool,
    log: RefCell<Vec<Call>>,
}

impl SetLm {
    fn new(layers: usize, rule: SetFn) -> Self {
        Self {
            sas: attention_layers(layers, 1),
            rule,
            vocab: V,
            nan_on_prefill: false,
            nan_on_decode: false,
            log: RefCell::new(Vec::new()),
        }
    }

    fn calls(&self) -> Vec<(usize, usize)> {
        self.log
            .borrow()
            .iter()
            .map(|c| (c.seq_before, c.len))
            .collect()
    }
}

impl AutoregressiveModel for SetLm {
    fn num_kv_layers(&self) -> usize {
        self.sas.len()
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let pos0 = caches.first().map_or(0, |c| c.seq_len());
        let tape = fandhe_ai::tape();
        let emb = tape.var(&t(seq(1, V * E), &[V, E]));
        let mut x = emb.embedding(new_ids, None)?;
        for (sa, cache) in self.sas.iter().zip(caches.iter_mut()) {
            x = tape.stateful_attention_forward_with_cache(sa, &x, cache)?;
        }
        let ts = toks(new_ids);
        let l = ts.len();
        self.log.borrow_mut().push(Call {
            seq_before: pos0,
            len: l,
        });
        let mut data = vec![GAP; l * self.vocab];
        for (i, &tok) in ts.iter().enumerate() {
            for s in (self.rule)(tok, pos0 + i) {
                data[i * self.vocab + s] = 0.0;
            }
        }
        if self.nan_on_prefill && pos0 == 0 {
            data[(l - 1) * self.vocab] = f32::NAN;
        }
        if self.nan_on_decode && pos0 > 0 && l == 1 {
            data[0] = f32::NAN;
        }
        Ok(t(data, &[1, l, self.vocab]))
    }
}

/// facade 公開面の実 attention 依存 logits モデル（`speculative_batching_facade.rs` の `AttnCachesLm` と同型）。
/// 非自明な確率で C1 を確かめ、`TopK(1)` と Greedy の一致（C5）に使う。
struct AttnLm {
    sas: Vec<StatefulAttention>,
    seed_base: i64,
    next: NextRule,
}

fn attn_target_next(tok: i32, pos: usize) -> usize {
    (3 * tok as usize + pos + 1) % V
}

fn attn_wrong_next(tok: i32, pos: usize) -> usize {
    (attn_target_next(tok, pos) + 1) % V
}

fn attn_periodic_next(tok: i32, pos: usize) -> usize {
    if pos.is_multiple_of(3) {
        attn_wrong_next(tok, pos)
    } else {
        attn_target_next(tok, pos)
    }
}

impl AttnLm {
    fn new(layers: usize, seed_base: i64, next: NextRule) -> Self {
        Self {
            sas: attention_layers(layers, 500 + seed_base as u64),
            seed_base,
            next,
        }
    }
}

impl AutoregressiveModel for AttnLm {
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
            data[i * V + (self.next)(tok, pos0 + i)] += SKIP_C;
        }
        Ok(t(data, &[1, l, V]))
    }
}

// ---------------------------------------------------------------------
// 再生オラクル（独立実装）
// ---------------------------------------------------------------------

/// 台（TopK なら昇順の先頭 k 点。全要素が同値 0.0 なので top-k は index 昇順の先頭 k 点になる）。
fn support(set: &[usize], strat: Strat) -> Vec<usize> {
    let mut s = set.to_vec();
    s.sort_unstable();
    s.dedup();
    if let Strat::TopK(k) = strat {
        s.truncate(k);
    }
    s
}

/// 集合から求めた確率（`f64`。台の上で一様）。
fn probs(set: &[usize], strat: Strat) -> Vec<f64> {
    let s = support(set, strat);
    let mut p = vec![0.0f64; V];
    for &i in &s {
        p[i] = 1.0 / s.len() as f64;
    }
    p
}

fn weights_f32(p: &[f64]) -> Vec<f32> {
    p.iter().map(|&v| v as f32).collect()
}

/// 参照実装が数えた事象（空判定の防止）。
#[derive(Default, Debug)]
struct Stats {
    accepts: usize,
    half_accept_draws: usize,
    rejects: usize,
    residual_draws: usize,
    bonus_draws: usize,
}

fn draw(g: &mut Generator, w: &[f32]) -> i32 {
    let sum: f32 = w.iter().sum();
    assert_eq!(sum, 1.0, "oracle: 重みの総和が厳密に 1 でない（前提違反）");
    let out = g
        .multinomial(&t(w.to_vec(), &[1, w.len()]), 1, true)
        .expect("oracle: multinomial");
    toks(&out)[0]
}

fn bernoulli(g: &mut Generator, a: f32) -> bool {
    let out = g.bernoulli(&t(vec![a], &[1])).expect("oracle: bernoulli");
    out.contiguous().host_slice()[0] == 1.0
}

/// §25.2 の消費順で `Generator::new(seed)` を 1 個だけ引く独立参照実装（本体コードは写さず、
/// 確率は集合の定義から導く）。`max_length == T` では RNG を作らない。
fn reference_speculative(
    (target, draft): (SetFn, SetFn),
    prompt: &[i32],
    max_length: usize,
    k: usize,
    strat: Strat,
    seed: u64,
    st: &mut Stats,
) -> Vec<i32> {
    let mut row = prompt.to_vec();
    let tlen = prompt.len();
    if max_length == tlen {
        return row;
    }
    let mut g = Generator::new(seed);
    let tp = |tok: i32, pos: usize| probs(&target(tok, pos), strat);
    let dp = |tok: i32, pos: usize| probs(&draft(tok, pos), strat);

    row.push(draw(&mut g, &weights_f32(&tp(row[tlen - 1], tlen - 1))));
    let mut cur = tlen + 1;
    while cur < max_length {
        let last = row[cur - 1];
        let remaining = max_length - cur;
        if remaining == 1 {
            let tok = draw(&mut g, &weights_f32(&tp(last, cur - 1)));
            row.push(tok);
            break;
        }
        let k_eff = k.min(remaining - 1);

        let mut drafts: Vec<i32> = Vec::new();
        let mut qs: Vec<Vec<f64>> = Vec::new();
        let mut x = last;
        for i in 0..k_eff {
            let q = dp(x, cur - 1 + i);
            x = draw(&mut g, &weights_f32(&q));
            drafts.push(x);
            qs.push(q);
        }

        let mut n_acc = 0usize;
        let mut correction: Option<i32> = None;
        for i in 0..k_eff {
            let inp = if i == 0 { last } else { drafts[i - 1] };
            let p = tp(inp, cur - 1 + i);
            let d = drafts[i] as usize;
            let a = (p[d] / qs[i][d]).min(1.0) as f32;
            assert!(
                a == 0.0 || a == 0.5 || a == 1.0,
                "oracle: 受理確率 {a} が厳密入力の期待値でない"
            );
            if bernoulli(&mut g, a) {
                n_acc += 1;
                st.accepts += 1;
                if a == 0.5 {
                    st.half_accept_draws += 1;
                }
                continue;
            }
            st.rejects += 1;
            let r: Vec<f64> = p
                .iter()
                .zip(&qs[i])
                .map(|(a, b)| (a - b).max(0.0))
                .collect();
            let z: f64 = r.iter().sum();
            assert!(z > 0.0, "oracle: 残差の総和が 0（棄却したのに残差が無い）");
            let w: Vec<f32> = r.iter().map(|&v| (v / z) as f32).collect();
            for (j, &wj) in w.iter().enumerate() {
                if p[j] <= qs[i][j] {
                    assert_eq!(wj, 0.0, "oracle: 非支持位置の残差重みが厳密に 0 でない");
                }
            }
            st.residual_draws += 1;
            correction = Some(draw(&mut g, &w));
            break;
        }
        let correction = correction.unwrap_or_else(|| {
            st.bonus_draws += 1;
            let inp = drafts[k_eff - 1];
            draw(&mut g, &weights_f32(&tp(inp, cur - 1 + k_eff)))
        });
        row.extend_from_slice(&drafts[..n_acc]);
        row.push(correction);
        cur += n_acc + 1;
    }
    row
}

fn run_set(
    target: SetFn,
    draft: SetFn,
    prompt: &[i32],
    config: &GenerateConfig,
    k: usize,
    rank1: bool,
    layers: (usize, usize),
) -> Tensor<i32> {
    let tm = SetLm::new(layers.0, target);
    let dm = SetLm::new(layers.1, draft);
    generate_speculative(
        &tm,
        &dm,
        &ids(prompt, rank1),
        config,
        &SpeculativeConfig::new(k),
    )
    .expect("generate_speculative")
}

fn prompt_of(len: usize) -> Vec<i32> {
    (0..len).map(|i| ((i * 3 + 1) % V) as i32).collect()
}

const STRATS: [Strat; 4] = [
    Strat::Temperature(1.0),
    Strat::Temperature(0.8),
    Strat::TopK(4),
    Strat::TopK(8),
];

// ---------------------------------------------------------------------
// C3（受理確率・残差・消費順）: 厳密入力の再生オラクルとの完全一致
// ---------------------------------------------------------------------

#[test]
fn sampling_matches_exact_f64_reference_replay() {
    let _g = global_rng_lock();
    let mut st = Stats::default();
    let mut runs = 0usize;
    for seed in 0..8u64 {
        for strat in STRATS {
            for (name, drule) in DRAFT_RULES {
                for k in [1usize, 2, 3, 5] {
                    for tlen in [1usize, 3] {
                        for max in [tlen, tlen + 1, tlen + 2, 12] {
                            let p = prompt_of(tlen);
                            let rank1 = seed % 2 == 0;
                            let config = strat.config(max, seed);
                            // target 1 層・draft 2 層（層数を変えて KV の取り違えも検出する）。
                            let out = run_set(rule_target, drule, &p, &config, k, rank1, (1, 2));
                            let want = reference_speculative(
                                (rule_target, drule),
                                &p,
                                max,
                                k,
                                strat,
                                seed,
                                &mut st,
                            );
                            let shape = if rank1 { vec![max] } else { vec![1, max] };
                            assert_eq!(out.shape(), &shape[..]);
                            assert_eq!(
                                toks(&out),
                                want,
                                "{name} {strat:?} seed={seed} k={k} T={tlen} max={max}"
                            );
                            runs += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(runs, 8 * 4 * 5 * 4 * 2 * 4);
    assert!(st.accepts > 0, "{st:?}");
    assert!(st.half_accept_draws > 0, "{st:?}");
    assert!(st.rejects > 0, "{st:?}");
    assert!(st.residual_draws > 0, "{st:?}");
    assert!(st.bonus_draws > 0, "{st:?}");
}

// ---------------------------------------------------------------------
// C1（決定性）と、シードを変えると出力が変わること
// ---------------------------------------------------------------------

#[test]
fn sampling_is_deterministic_for_same_seed() {
    let _g = global_rng_lock();
    for strat in [Strat::Temperature(0.8), Strat::TopK(3), Strat::TopK(4)] {
        for k in [1usize, 3] {
            for rank1 in [true, false] {
                for layers in [1usize, 2] {
                    let p = prompt_of(3);
                    let config = strat.config(11, 7);
                    let shape = if rank1 { vec![11] } else { vec![1, 11] };

                    // 集合モデル（draft は部分受理・棄却が混ざる規則）。
                    let a = run_set(
                        rule_target,
                        rule_shift2,
                        &p,
                        &config,
                        k,
                        rank1,
                        (layers, layers),
                    );
                    let b = run_set(
                        rule_target,
                        rule_shift2,
                        &p,
                        &config,
                        k,
                        rank1,
                        (layers, layers),
                    );
                    assert_eq!(a.shape(), &shape[..]);
                    assert_eq!(toks(&a), toks(&b), "SetLm {strat:?} k={k}");

                    // 非自明な確率を持つ実 attention モデル。
                    let run = || {
                        let target = AttnLm::new(layers, 100, attn_target_next);
                        let draft = AttnLm::new(layers, 300, attn_periodic_next);
                        generate_speculative(
                            &target,
                            &draft,
                            &ids(&p, rank1),
                            &config,
                            &SpeculativeConfig::new(k),
                        )
                        .expect("attn speculative")
                    };
                    let (c, d) = (run(), run());
                    assert_eq!(c.shape(), &shape[..]);
                    assert_eq!(toks(&c), toks(&d), "AttnLm {strat:?} k={k}");
                }
            }
        }
    }
}

#[test]
fn different_seeds_can_change_output() {
    let _g = global_rng_lock();
    let p = prompt_of(1);
    for strat in [Strat::Temperature(1.0), Strat::TopK(8)] {
        let mut outs: Vec<Vec<i32>> = Vec::new();
        for seed in 0..8u64 {
            let config = strat.config(12, seed);
            let out = toks(&run_set(
                rule_target,
                rule_subset2,
                &p,
                &config,
                3,
                true,
                (1, 1),
            ));
            let want = reference_speculative(
                (rule_target, rule_subset2),
                &p,
                12,
                3,
                strat,
                seed,
                &mut Stats::default(),
            );
            assert_eq!(out, want, "{strat:?} seed={seed}");
            outs.push(out);
        }
        outs.sort();
        outs.dedup();
        assert!(
            outs.len() >= 2,
            "{strat:?}: 8 個の seed で出力が 1 種類しかない（RNG が出力へ反映されていない）"
        );
    }
}

// ---------------------------------------------------------------------
// C2（グローバル RNG 非消費）
// ---------------------------------------------------------------------

#[test]
fn sampling_does_not_consume_global_rng() {
    let _g = global_rng_lock();
    for strat in [Strat::TopK(4), Strat::Temperature(0.8)] {
        for seed in [3u64, 99] {
            fandhe_ai::manual_seed(seed);
            let before = toks_f32(&fandhe_ai::rand(&[4]).expect("rand"));
            fandhe_ai::manual_seed(seed);
            // draft (v) は常に棄却するので、bernoulli・残差・prefill・最終 step の全経路を通る。
            let out = run_set(
                rule_target,
                rule_disjoint,
                &prompt_of(2),
                &strat.config(10, 5),
                3,
                true,
                (1, 1),
            );
            assert_eq!(out.shape(), &[10]);
            let after = toks_f32(&fandhe_ai::rand(&[4]).expect("rand"));
            assert_eq!(
                before, after,
                "{strat:?}: グローバル RNG の消費が検出された"
            );
        }
    }
}

fn toks_f32(x: &Tensor<f32>) -> Vec<f32> {
    x.contiguous().host_slice().into_owned()
}

// ---------------------------------------------------------------------
// C4（draft == target は全受理）
// ---------------------------------------------------------------------

/// 全受理を仮定した target 呼び出し数（prefill 1 + ラウンド数 + 最終 step）。独立に計算する。
fn expected_target_calls(tlen: usize, max: usize, k: usize) -> usize {
    if max == tlen {
        return 0;
    }
    let mut calls = 1; // prefill
    let mut cur = tlen + 1;
    while cur < max {
        let remaining = max - cur;
        calls += 1; // 最終 step または検証 forward
        if remaining == 1 {
            break;
        }
        cur += k.min(remaining - 1) + 1;
    }
    calls
}

#[test]
fn identical_draft_accepts_everything() {
    let _g = global_rng_lock();
    for strat in [Strat::Temperature(0.8), Strat::TopK(4)] {
        for k in [1usize, 2, 3, 5] {
            for (tlen, max) in [(1usize, 12usize), (3, 12), (3, 4), (1, 2)] {
                let p = prompt_of(tlen);
                let config = strat.config(max, 4);
                // 別インスタンス・別層数。
                let target = SetLm::new(1, rule_target);
                let draft = SetLm::new(2, rule_target);
                let out = generate_speculative(
                    &target,
                    &draft,
                    &ids(&p, false),
                    &config,
                    &SpeculativeConfig::new(k),
                )
                .expect("speculative");
                let mut st = Stats::default();
                let want = reference_speculative(
                    (rule_target, rule_target),
                    &p,
                    max,
                    k,
                    strat,
                    4,
                    &mut st,
                );
                assert_eq!(toks(&out), want);
                assert_eq!(st.rejects, 0, "参照実装が棄却した");
                for (name, m) in [("target", &target), ("draft", &draft)] {
                    let calls = m.calls();
                    // 再 forward（同じ seq_before の繰り返し）や巻き戻しが無い。
                    for w in calls.windows(2) {
                        assert!(w[1].0 > w[0].0, "{name}: 再 forward {calls:?}");
                    }
                }
                assert_eq!(
                    target.calls().len(),
                    expected_target_calls(tlen, max, k),
                    "{strat:?} k={k} T={tlen} max={max}: target の呼び出し数"
                );

                // 同一インスタンスでも同じ。
                let same = SetLm::new(1, rule_target);
                let out2 = generate_speculative(
                    &same,
                    &same,
                    &ids(&p, false),
                    &config,
                    &SpeculativeConfig::new(k),
                );
                // 同一インスタンスは `log` を共有するため、呼び出し列ではなく出力だけを見る。
                assert_eq!(toks(&out2.expect("same instance")), want);
            }
        }
    }
}

// ---------------------------------------------------------------------
// C5（TopK(1)・one-hot logits は Greedy と一致）
// ---------------------------------------------------------------------

#[test]
fn top_k_one_and_one_hot_logits_match_greedy() {
    let _g = global_rng_lock();
    let greedy = |max: usize| GenerateConfig::new(max, SamplingStrategy::Greedy);
    let drafts: [(&str, SetFn); 3] = [
        ("全受理", rule_one),
        ("全棄却", rule_one_wrong),
        ("部分受理", rule_one_periodic),
    ];

    for (name, drule) in drafts {
        for k in [1usize, 3] {
            for (tlen, max) in [(1usize, 9usize), (3, 12)] {
                let p = prompt_of(tlen);
                let alone = {
                    let m = SetLm::new(1, rule_one);
                    toks(&generate(&m, &ids(&p, true), &greedy(max)).expect("alone"))
                };
                let g_spec = toks(&run_set(rule_one, drule, &p, &greedy(max), k, true, (1, 1)));
                assert_eq!(g_spec, alone, "{name} Greedy spec == alone");
                // (a) TopK(1)。
                for seed in [0u64, 1, 2] {
                    let c = Strat::TopK(1).config(max, seed);
                    let out = toks(&run_set(rule_one, drule, &p, &c, k, true, (1, 1)));
                    assert_eq!(out, alone, "{name} TopK(1) seed={seed}");
                }
                // (b) Temperature(0.8) でも gap -1e4 の one-hot は重みが厳密に one-hot。
                for seed in [0u64, 5, 11] {
                    let c = Strat::Temperature(0.8).config(max, seed);
                    let out = toks(&run_set(rule_one, drule, &p, &c, k, true, (1, 1)));
                    assert_eq!(out, alone, "{name} Temperature(0.8) seed={seed}");
                }
            }
        }
    }

    // 実 attention モデルでの TopK(1)。
    let attn_drafts: [(&str, NextRule); 3] = [
        ("全受理", attn_target_next),
        ("全棄却", attn_wrong_next),
        ("部分受理", attn_periodic_next),
    ];
    for (name, next) in attn_drafts {
        for k in [1usize, 3] {
            let p = prompt_of(3);
            let max = 11;
            let target = AttnLm::new(1, 100, attn_target_next);
            let draft = AttnLm::new(1, 300, next);
            let alone = toks(&generate(&target, &ids(&p, false), &greedy(max)).expect("alone"));
            let greedy_spec = toks(
                &generate_speculative(
                    &target,
                    &draft,
                    &ids(&p, false),
                    &greedy(max),
                    &SpeculativeConfig::new(k),
                )
                .expect("greedy spec"),
            );
            assert_eq!(greedy_spec, alone, "{name} AttnLm Greedy spec == alone");
            for seed in [0u64, 4] {
                let out = toks(
                    &generate_speculative(
                        &target,
                        &draft,
                        &ids(&p, false),
                        &Strat::TopK(1).config(max, seed),
                        &SpeculativeConfig::new(k),
                    )
                    .expect("topk1 spec"),
                );
                assert_eq!(out, alone, "{name} AttnLm TopK(1) seed={seed}");
            }
        }
    }
}

// ---------------------------------------------------------------------
// C6（fail-closed）
// ---------------------------------------------------------------------

#[test]
fn sampling_fails_closed() {
    let _g = global_rng_lock();
    let p = prompt_of(2);
    let spec = |k: usize| SpeculativeConfig::new(k);
    let sampling = || Strat::TopK(4).config(10, 1);
    let go = |target: &SetLm, draft: &SetLm, c: &GenerateConfig, k: usize| {
        generate_speculative(target, draft, &ids(&p, true), c, &spec(k))
    };
    let invalid = |r: Result<Tensor<i32>, AutodiffError>, what: &str| match r {
        Err(AutodiffError::InvalidArgument(_)) => {}
        other => panic!("{what}: InvalidArgument を期待したが {other:?}"),
    };

    // TopK(V + 1)。
    let (tm, dm) = (SetLm::new(1, rule_target), SetLm::new(1, rule_target));
    invalid(
        go(&tm, &dm, &Strat::TopK(V + 1).config(10, 1), 3),
        "TopK(V + 1)",
    );
    // k == 0。
    invalid(go(&tm, &dm, &sampling(), 0), "k == 0");
    // 設定の矛盾（TopK(0)・非正／非有限の temperature）。
    invalid(go(&tm, &dm, &Strat::TopK(0).config(10, 1), 3), "TopK(0)");
    invalid(
        go(&tm, &dm, &Strat::Temperature(0.0).config(10, 1), 3),
        "Temperature(0)",
    );
    invalid(
        go(&tm, &dm, &Strat::Temperature(f32::NAN).config(10, 1), 3),
        "Temperature(NaN)",
    );
    // 語彙サイズ不一致（draft だけ V + 1）。
    let mut wide = SetLm::new(1, rule_target);
    wide.vocab = V + 1;
    match go(&tm, &wide, &sampling(), 3) {
        Err(AutodiffError::Shape(_)) => {}
        other => panic!("語彙サイズ不一致: Shape を期待したが {other:?}"),
    }
    // target の prefill 末尾の非有限 logits。
    let mut nan_t = SetLm::new(1, rule_target);
    nan_t.nan_on_prefill = true;
    invalid(go(&nan_t, &dm, &sampling(), 3), "target prefill の NaN");
    // draft decode の非有限 logits。
    let mut nan_d = SetLm::new(1, rule_target);
    nan_d.nan_on_decode = true;
    invalid(go(&tm, &nan_d, &sampling(), 3), "draft decode の NaN");
}
