//! `generate_speculative`（greedy 版 speculative decoding。イシュー #2886）が
//! target 単独の `generate`（`SamplingStrategy::Greedy`）と一致することの
//! 統合テスト（イシュー #2887。設計記録
//! `docs/facade-speculative-decoding-batching-design.md` §6.1・§11）。
//!
//! 別クレート扱いの統合テストとして autodiff の公開経路
//! （`fandhe_ai_autodiff::generate::speculative`）だけを使い、`backend-cpu` へは
//! 依存しない（`architecture_boundaries.rs` の不変条件。REQ-2 判定は
//! `common::req2_close` を使う）。
//!
//! # 判定の 2 段構成
//!
//! 1. **表引きモデル（テスト A）**: KV キャッシュは実 `MultiheadAttentionVars` で
//!    進めるが、logits は `(token, 絶対位置)` の表引き one-hot だけで決まるため
//!    形状（`L_new`）に依らず bit 同一になる。token 列と、比較可能な全位置の
//!    logits を **完全一致**（`f32::to_bits`）で確かめる。
//!    `num_kv_layers() == 0` のモデルは入口で拒否される（設計 §10 論点 3）ため
//!    フィクスチャに含めない。
//! 2. **attention 依存モデル（テスト B）**: logits が KV 経路の数値（CPU GEMM の
//!    形状依存ブロッキング。`docs/kv-cache-design.md` §3.5）に依存する。契約は
//!    位置ごとの logits を REQ-2 統一複合判定（相対誤差 1e-3 未満 または
//!    絶対誤差 1e-5 未満）で確かめること。
//!
//! # 事前登録の仮説 H1
//!
//! margin 前提（`assert_margin`）を満たす入力では、attention 依存モデルでも
//! `generate_speculative` の token 列が `generate`（Greedy）と一致する。
//! H1 は契約ではなく仮説であり、破れた場合は判定を緩めず設計 §10 論点 2 として
//! 承認依頼に戻す。tolerance・baseline は新設しない。margin 前提は既存 REQ-2
//! 定数から導いた入力の妥当性検査であり、新しい判定閾値ではない。
//!
//! 対象外: CUDA／Metal 実機 parity（#2890）、サンプリング版（設計 §10 論点 1）、
//! `num_kv_layers() == 0` のモデル（論点 3）、入口の fail-closed 検査（#2886 の
//! 単体テストの担当）。

mod common;

use fandhe_ai_autodiff::generate::speculative::{SpeculativeConfig, generate_speculative};
use fandhe_ai_autodiff::generate::{
    AutoregressiveModel, GenerateConfig, SamplingStrategy, generate,
};
use fandhe_ai_autodiff::nn::{KvCache, LinearVars, MultiheadAttentionVars};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;
use std::cell::RefCell;

const V: usize = 7;
const E: usize = 4;
const HEADS: usize = 2;
/// テスト B の skip 項の大きさ。上位 2 候補の margin を attention 経路の
/// 数値揺らぎより十分大きく取るためのフィクスチャ定数（判定閾値ではない）。
/// speculative 実行の前に固定してあり、結果を見て調整しない（H1 の事前登録）。
const SKIP_C: f32 = 3.0;

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

/// 位置が `period` の倍数のときだけ外す draft 規則。
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

/// 1 回の `forward_step` 呼び出しの記録。
struct Call {
    /// 呼び出し前の `caches[0].seq_len()`（= `new_ids` 先頭の絶対位置）。
    seq_before: usize,
    ids: Vec<i32>,
    /// `[L_new, V]` を行優先で平坦化した logits。
    logits: Vec<f32>,
}

/// 任意の [`AutoregressiveModel`] を包み、各 `forward_step` の入出力を
/// 記録するラッパー。target 単独 `generate` と speculative の target 呼び出しの
/// 位置ごとの logits を突き合わせるために使う。
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
        let seq_before = caches[0].seq_len();
        let logits = self.inner.forward_step(new_ids, caches)?;
        self.log.borrow_mut().push(Call {
            seq_before,
            ids: vals(new_ids),
            logits: logits.contiguous().host_slice().into_owned(),
        });
        Ok(logits)
    }
}

/// target 単独 `generate` の記録から、絶対位置ごとの logits 行を構築する。
/// 位置の重複・欠落は fail-closed で panic する。
fn reference_rows(log: &[Call]) -> Vec<Vec<f32>> {
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for call in log {
        for (i, row) in call.logits.chunks(V).enumerate() {
            let pos = call.seq_before + i;
            assert_eq!(
                pos,
                rows.len(),
                "参照 generate の位置が連続しない（重複または欠落）"
            );
            rows.push(row.to_vec());
        }
    }
    rows
}

/// speculative の target 記録から、prefix が最終出力と一致する行だけを
/// `(絶対位置, 行, その呼び出しの L_new)` で返す。キャッシュ不変条件により
/// `output[..seq_before]` は確定済み prefix なので、`output[seq_before..=p]` が
/// `new_ids[..=i]` と一致すれば、その行は「最終出力と同じ prefix の logits」になる。
/// 棄却された draft 以降の位置は prefix が異なるため比較しない。
fn comparable_rows(log: &[Call], output: &[i32]) -> Vec<(usize, Vec<f32>, usize)> {
    let mut out = Vec::new();
    for call in log {
        let l_new = call.ids.len();
        for (i, row) in call.logits.chunks(V).enumerate() {
            let p = call.seq_before + i;
            if p < output.len() && output[call.seq_before..=p] == call.ids[..=i] {
                out.push((p, row.to_vec(), l_new));
            }
        }
    }
    out
}

/// 再 forward（rewind-replay）の回数。キャッシュは各ラウンドで必ず前進するため、
/// 直前の呼び出しと同じ `seq_before` から始まる呼び出しは巻き戻し後の再 forward。
fn replay_count(log: &[Call]) -> usize {
    log.windows(2)
        .filter(|w| w[1].seq_before == w[0].seq_before)
        .count()
}

// ---------------------------------------------------------------------
// テスト A: 形状非依存（表引き）モデル
// ---------------------------------------------------------------------

/// KV キャッシュを実 MHA で進めつつ、logits は `(token, 絶対位置)` の one-hot
/// だけで決まるモデル。logits は `L_new` に依らず bit 同一になる。
struct TableKvModel {
    layers: usize,
    next: NextFn,
}

impl AutoregressiveModel for TableKvModel {
    fn num_kv_layers(&self) -> usize {
        self.layers
    }

    fn forward_step(
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
        let toks = vals(new_ids);
        let mut data = vec![0.0f32; toks.len() * V];
        for (i, &tok) in toks.iter().enumerate() {
            data[i * V + (self.next)(tok, pos0 + i) as usize] = 10.0;
        }
        Ok(t(data, &[1, toks.len(), V]))
    }
}

fn table(layers: usize, next: NextFn) -> Recording<TableKvModel> {
    Recording::new(TableKvModel { layers, next })
}

#[test]
fn table_model_matches_generate_exactly() {
    let mut runs = 0usize;
    for layers in [1usize, 2] {
        for rank1 in [true, false] {
            for k in 1..=6usize {
                for prompt_len in [1usize, 2, 4] {
                    let maxes = [
                        prompt_len,
                        prompt_len + 1,
                        prompt_len + 2,
                        prompt_len + k,
                        prompt_len + k + 1,
                        20,
                    ];
                    for max in maxes {
                        let ids = prompt(prompt_len, rank1);
                        let reference_model = table(layers, Box::new(target_next));
                        let reference = generate(&reference_model, &ids, &cfg(max)).unwrap();
                        let ref_rows = reference_rows(&reference_model.log.borrow());

                        let drafts: Vec<(&str, NextFn)> = vec![
                            ("target と同一", Box::new(target_next)),
                            ("常に外す", Box::new(wrong_next)),
                            ("周期 2 で外す", periodic_next(2)),
                            ("周期 5 で外す", periodic_next(5)),
                            ("周期 k+1 で外す", periodic_next(k + 1)),
                        ];
                        for (name, draft_next) in drafts {
                            let label = format!(
                                "layers={layers} rank1={rank1} k={k} T={prompt_len} \
                                 max={max} draft={name}"
                            );
                            let target = table(layers, Box::new(target_next));
                            let draft = table(layers, draft_next);
                            let out = generate_speculative(
                                &target,
                                &draft,
                                &ids,
                                &cfg(max),
                                &SpeculativeConfig::new(k),
                            )
                            .unwrap();
                            assert_eq!(out.shape(), reference.shape(), "{label}: shape");
                            assert_eq!(vals(&out), vals(&reference), "{label}: token 列");

                            let rows = comparable_rows(&target.log.borrow(), &vals(&out));
                            // トークン選択位置（T-1 ..= max-2）が全て比較対象に入る。
                            for p in prompt_len.saturating_sub(1)..max.saturating_sub(1) {
                                if max > prompt_len {
                                    assert!(
                                        rows.iter().any(|r| r.0 == p),
                                        "{label}: 位置 {p} が比較されていない"
                                    );
                                }
                            }
                            for (p, row, l_new) in &rows {
                                let want = &ref_rows[*p];
                                let same = row
                                    .iter()
                                    .zip(want.iter())
                                    .all(|(a, b)| a.to_bits() == b.to_bits());
                                assert!(
                                    same,
                                    "{label}: 位置 {p}（L_new={l_new}）の logits が bit 一致しない"
                                );
                            }
                            runs += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(runs > 1000, "掃引が縮退している（runs={runs}）");
}

#[test]
fn table_model_same_instance_as_draft_and_target() {
    for layers in [1usize, 2] {
        for rank1 in [true, false] {
            let m = table(layers, Box::new(target_next));
            let ids = prompt(3, rank1);
            let out =
                generate_speculative(&m, &m, &ids, &cfg(14), &SpeculativeConfig::new(4)).unwrap();
            let reference =
                generate(&table(layers, Box::new(target_next)), &ids, &cfg(14)).unwrap();
            assert_eq!(out.shape(), reference.shape());
            assert_eq!(vals(&out), vals(&reference));
        }
    }
}

// ---------------------------------------------------------------------
// テスト B: attention 依存 KV モデル
// ---------------------------------------------------------------------

/// `Embedding → MHA（KV キャッシュ付き・層ごとに残差なしで直列） → lm head` に、
/// margin 確保用の skip 項 `SKIP_C · onehot(next(tok, pos))` をホスト側で足すモデル。
/// lm head の出力は必ず logits に入るため、logits は KV 経路の数値に依存する。
struct AttnLm {
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
        let pos0 = caches[0].seq_len();
        let s = self.seed_base;
        let tape = Tape::new();
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
        let toks = vals(new_ids);
        for (i, &tok) in toks.iter().enumerate() {
            data[i * V + (self.next)(tok, pos0 + i) as usize] += SKIP_C;
        }
        Ok(t(data, &[1, l, V]))
    }
}

fn attn(layers: usize, seed_base: i64, next: NextFn) -> Recording<AttnLm> {
    Recording::new(AttnLm {
        layers,
        seed_base,
        next,
    })
}

/// 入力の妥当性検査（新しい tolerance ではない）。参照 `generate` の
/// トークン選択位置 `T-1 ..= max-2` で `top1 - top2` が十分大きいことを確かめる。
///
/// 導出: 2 つの行 a・b の各要素が `common::req2_close` を満たすなら
/// `|a-b| < rtol·max(|a|,|b|)`（相対側）または `|a-b| < abs`（救済側）。
/// 行の絶対値最大を M とすると `max(|a|,|b|) ≤ M + |a-b|` より
/// `|a-b| < rtol·M/(1-rtol)` または `< abs`。argmax が反転するには
/// 上位 2 候補の差が 2 要素分の乖離 `2·bound` 以下になる必要があるため、
/// `top1 - top2 > 2·(rtol·M/(1-rtol) + abs)` なら反転しえない。
/// 既存 REQ-2 定数から導いた帰結で、新しい数値は作らない。
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
        let rtol = common::REQ2_RELATIVE_TOLERANCE;
        let bound = rtol * m / (1.0 - rtol);
        let bound = bound.max(common::REQ2_ABSOLUTE_RESCUE_THRESHOLD);
        let margin = (sorted[0] - sorted[1]) as f64;
        assert!(
            margin > 2.0 * bound,
            "{label}: フィクスチャ不正（margin 前提を満たさない）位置 {p}: \
             margin={margin} 必要={}",
            2.0 * bound
        );
    }
}

fn assert_row_req2(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: 行長");
    for (j, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            common::req2_close(*a as f64, *b as f64),
            "{label}[{j}]: actual={a} expected={b}"
        );
    }
}

#[test]
fn attention_model_logits_req2_and_tokens_match() {
    let mut runs = 0usize;
    let mut verify_rows = 0usize;
    let mut replays = 0usize;
    let mut bit_mismatch_rows = 0usize;
    let mut compared_rows = 0usize;
    for layers in [1usize, 2] {
        for k in [1usize, 2, 3, 5] {
            for prompt_len in [1usize, 3] {
                let maxes = [prompt_len + 1, prompt_len + k + 1, prompt_len + k + 2, 16];
                for max in maxes {
                    let ids = prompt(prompt_len, false);
                    // 参照（speculative を実行する前に margin 前提を評価する）。
                    let ref_model = attn(layers, 100, Box::new(target_next));
                    let reference = generate(&ref_model, &ids, &cfg(max)).unwrap();
                    let ref_rows = reference_rows(&ref_model.log.borrow());
                    assert_margin(
                        &format!("layers={layers} k={k} T={prompt_len} max={max}"),
                        &ref_rows,
                        prompt_len,
                        max,
                    );

                    // (名前, draft の seed_base, draft の規則)
                    let drafts: Vec<(&str, i64, NextFn)> = vec![
                        ("全受理(同一重み)", 100, Box::new(target_next)),
                        ("全受理(別重み)", 300, Box::new(target_next)),
                        ("全棄却", 300, Box::new(wrong_next)),
                        ("部分受理(周期 3)", 300, periodic_next(3)),
                        ("部分受理(周期 k+1)", 300, periodic_next(k + 1)),
                    ];
                    for (name, draft_seed, draft_next) in drafts {
                        let label =
                            format!("layers={layers} k={k} T={prompt_len} max={max} draft={name}");
                        let target = attn(layers, 100, Box::new(target_next));
                        let draft = attn(layers, draft_seed, draft_next);
                        let out = generate_speculative(
                            &target,
                            &draft,
                            &ids,
                            &cfg(max),
                            &SpeculativeConfig::new(k),
                        )
                        .unwrap();
                        let out_vals = vals(&out);
                        let log = target.log.borrow();

                        // 1. 契約（REQ-2）: 位置ごとの logits。
                        let rows = comparable_rows(&log, &out_vals);
                        for (p, row, l_new) in &rows {
                            assert_row_req2(
                                &format!("{label} 位置 {p}（L_new={l_new}）"),
                                row,
                                &ref_rows[*p],
                            );
                            compared_rows += 1;
                            if row
                                .iter()
                                .zip(ref_rows[*p].iter())
                                .any(|(a, b)| a.to_bits() != b.to_bits())
                            {
                                bit_mismatch_rows += 1;
                            }
                        }

                        // 2. 空判定の防止（fail-closed）。
                        for p in (prompt_len - 1)..=(max - 2) {
                            assert!(
                                rows.iter().any(|r| r.0 == p),
                                "{label}: トークン選択位置 {p} が比較されていない"
                            );
                        }
                        verify_rows += rows.iter().filter(|r| r.2 > 1).count();
                        replays += replay_count(&log);

                        // 3. 仮説 H1: token 列一致。
                        assert_eq!(
                            out.shape(),
                            reference.shape(),
                            "{label}: shape（事前登録の仮説 H1 の破れ）"
                        );
                        assert_eq!(
                            out_vals,
                            vals(&reference),
                            "{label}: 事前登録の仮説 H1 の破れ。判定を緩めず設計 §10 論点 2 \
                             として承認依頼に戻すこと"
                        );
                        runs += 1;
                    }
                }
            }
        }
    }
    // 検証 forward（L_new > 1）と再 forward を実際に通した（空判定でない）こと。
    assert!(runs > 100, "掃引が縮退している（runs={runs}）");
    assert!(
        verify_rows > 0,
        "検証 forward の行が 1 つも比較されていない"
    );
    assert!(
        replays > 0,
        "再 forward（rewind-replay）が 1 度も起きていない"
    );
    // bit 一致は契約ではない（設計 §6.1）。参考情報として出すだけで assert しない。
    eprintln!(
        "attention model: 比較 {compared_rows} 行中 bit 不一致 {bit_mismatch_rows} 行 \
         （参考情報。契約は REQ-2 判定）"
    );
}
