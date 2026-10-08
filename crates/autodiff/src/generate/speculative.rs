//! greedy 版 speculative decoding の内部実装（イシュー #2886。設計の正は
//! `docs/facade-speculative-decoding-batching-design.md` §3・§4・§4.1・§6.1・§7・§10）。
//!
//! # 役割
//!
//! 小さい draft モデルが `k` トークン先読みし、target が 1 回の forward で
//! まとめて検証する。`generate` / `GenerateConfig` は変更せず、別関数
//! `generate_speculative` として提供する。KV 巻き戻しは設計 §4.1 の案 (i)
//! （兄弟モジュール `kv_rewind` の `KvSnapshot` と `rewind_and_replay`）だけを使う。
//!
//! # 制限（いずれも設計記録の未承認論点または承認待ち事項に由来）
//!
//! - **facade 未公開**: `docs/compat-api-scope.md` §5.1 の S1 の承認待ち。
//!   本モジュールは `autodiff` 内部の公開に留まり、facade からは到達できない
//! - **B = 1 限定**: 全系列の `S_cached` が共通である前提（`docs/kv-cache-design.md` §7）
//! - **Greedy のみ**: サンプリング版は設計 §10 論点 1（未承認）でブロック中。
//!   黙って greedy へ落とさず `InvalidArgument` で拒否する
//! - **`num_kv_layers() == 0` のモデルは対象外**: 設計 §10 論点 3 が決まるまで拒否する
//!   （`kv_rewind` と同じ扱い）
//! - **token 列は `generate` と bit 一致するとは限らない**: 検証 forward（長さ
//!   `k + 1`）と 1 トークン decode で logits が bit 一致する保証がない（設計 §6.1）。
//!   形状に依存しない logits を返すモデルでだけ完全一致する
//!
//! # アルゴリズムの要点
//!
//! 各ラウンド開始時、行の末尾トークン `last` はどちらのモデルにも未 forward で、
//! 両キャッシュは `cur_len - 1` トークン分を保持する。残り長を `R` とすると
//! `R == 1` は target の 1 step decode、`R >= 2` は `k_eff = min(k, R - 1)` で
//! 丸めて speculative に進む（全受理時の bonus トークンを含めても出力が
//! 丁度 `max_length` に収まり、切り捨てが起きない）。追加トークンは必ず検証
//! forward の logits から決め、巻き戻し後の再 forward の logits は捨てる。
//!
//! 乱数は使わない（グローバル RNG も消費しない）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::KvCache;

use super::kv_rewind::KvSnapshot;
use super::{
    AutoregressiveModel, GenerateConfig, SamplingStrategy, build_output, greedy_argmax, kv_rewind,
    validate_forward_step_output,
};

/// [`generate_speculative`] の設定（先読み長 `k`）。`#[non_exhaustive]` と derive は
/// [`GenerateConfig`] に合わせる（公開クレートの derive は後から外せないため最小限）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SpeculativeConfig {
    /// draft が 1 ラウンドで先読みするトークン数（1 以上）。残り長が短いときは
    /// `R - 1` に丸められる。
    pub k: usize,
}

impl SpeculativeConfig {
    /// 先読み長 `k` で構築する（`k == 0` の拒否は [`generate_speculative`] の入口検査）。
    pub fn new(k: usize) -> SpeculativeConfig {
        SpeculativeConfig { k }
    }
}

/// `logits`（`[1, l_new, vocab]`。形状は `validate_forward_step_output` で検証済み）の
/// 位置 `pos` から greedy にトークンを選ぶ。非有限値を含む行は拒否する。
fn greedy_token_at(
    logits: &Tensor<f32>,
    pos: usize,
    l_new: usize,
    vocab: usize,
) -> Result<i32, AutodiffError> {
    let contiguous = logits.contiguous();
    let data = contiguous.host_slice();
    // pos < l_new かつ data.len() == l_new * vocab は呼び出し元の shape 検証と
    // ループ境界で保証されるが、保証が崩れても panic せず Err にする。
    let row = pos
        .checked_mul(vocab)
        .and_then(|start| Some((start, start.checked_add(vocab)?)))
        .filter(|&(_, end)| pos < l_new && end <= data.len())
        .and_then(|(start, end)| data.get(start..end))
        .ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "generate_speculative: logits の位置が範囲外（内部不整合）".to_string(),
            )
        })?;
    if row.iter().any(|v| !v.is_finite()) {
        return Err(AutodiffError::InvalidArgument(
            "generate_speculative: logits に非有限値が含まれる".to_string(),
        ));
    }
    // vocab <= i32::MAX は validate_forward_step_output が検証済みなので、
    // 添字の `as i32` は折り返さない。
    Ok(greedy_argmax(row) as i32)
}

/// `ids`（長さ `L`）を `[1, L]` で forward し、shape と語彙サイズを検証する。
fn forward_checked<M: AutoregressiveModel + ?Sized>(
    model: &M,
    ids: &[i32],
    caches: &mut [KvCache],
    expected_vocab: usize,
) -> Result<Tensor<f32>, AutodiffError> {
    let l = ids.len();
    let input = Tensor::new(ids.to_vec(), &[1, l]).map_err(AutodiffError::Shape)?;
    let logits = model.forward_step(&input, caches)?;
    let vocab = validate_forward_step_output(&logits, 1, l)?;
    if vocab != expected_vocab {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: vec![1, l, vocab],
            rhs: vec![1, l, expected_vocab],
        }));
    }
    Ok(logits)
}

/// 退避済みキャッシュの長さがラウンド不変条件（`cur_len - 1`）と一致するか検査する。
fn check_snapshot_len(snap: &KvSnapshot, cur_len: usize) -> Result<(), AutodiffError> {
    if snap.seq_len() + 1 != cur_len {
        return Err(AutodiffError::InvalidArgument(format!(
            "generate_speculative: KV キャッシュ長 {} がラウンド不変条件（{}）と一致しない\
             （forward_step がキャッシュを進めていない可能性）",
            snap.seq_len(),
            cur_len.saturating_sub(1)
        )));
    }
    Ok(())
}

/// 巻き戻して `[last, accepted...]` を再 forward する（logits は捨てる）。
fn rewind_to<M: AutoregressiveModel + ?Sized>(
    model: &M,
    caches: &mut [KvCache],
    snap: KvSnapshot,
    replay_ids: &[i32],
    vocab: usize,
) -> Result<(), AutodiffError> {
    let ids =
        Tensor::new(replay_ids.to_vec(), &[1, replay_ids.len()]).map_err(AutodiffError::Shape)?;
    match kv_rewind::rewind_and_replay(model, caches, snap, &ids, vocab)? {
        Some(_) => Ok(()),
        None => Err(AutodiffError::InvalidArgument(
            "generate_speculative: 再生入力が空（内部不整合）".to_string(),
        )),
    }
}

/// greedy 版 speculative decoding。`target` を正とする生成を、`draft` の先読みと
/// target の一括検証で進める。`input_ids` は `[T]` または `[1, T]`（rank 1 なら出力も
/// rank 1）。出力は `[max_length]` または `[1, max_length]`。
///
/// 受理統計は返さない。`draft == target`（同一インスタンス可）でも動作する。
///
/// # Errors
///
/// 次の順に検査し、最初に該当した条件で `Err` を返す（前段で拒否する場合
/// `forward_step` は呼ばれない）。
///
/// 1. `config` が矛盾している
/// 2. `config.strategy` が `Greedy` 以外
/// 3. `spec.k == 0`
/// 4. `input_ids` の rank が 1／2 以外
/// 5. `T == 0`
/// 6. `B != 1`（`B == 0` を含む）
/// 7. `config.max_length < T`
/// 8. `B * max_length` の `usize` オーバーフロー、または出力バッファが
///    `isize::MAX` バイトを超える
/// 9. `target` または `draft` の `num_kv_layers() == 0`
///
/// 以降は forward の戻り shape の不整合、語彙サイズの target／draft 間・ステップ間
/// 不一致、KV キャッシュ長の不整合、トークン選択に使う位置（prefill 末尾・draft の
/// 各 decode・検証 forward の受理判定に使う位置・最終 1 step）の logits の非有限値、
/// およびモデルが返すエラー（そのまま伝播）で `Err` を返す。再 forward・補正 decode の
/// logits と、使わない検証位置の logits は検査しない。
pub fn generate_speculative<T, D>(
    target: &T,
    draft: &D,
    input_ids: &Tensor<i32>,
    config: &GenerateConfig,
    spec: &SpeculativeConfig,
) -> Result<Tensor<i32>, AutodiffError>
where
    T: AutoregressiveModel + ?Sized,
    D: AutoregressiveModel + ?Sized,
{
    config.validate()?;
    if !matches!(config.strategy, SamplingStrategy::Greedy) {
        return Err(AutodiffError::InvalidArgument(
            "generate_speculative: SamplingStrategy::Greedy のみ対応（サンプリング版は未承認）"
                .to_string(),
        ));
    }
    if spec.k == 0 {
        return Err(AutodiffError::InvalidArgument(
            "generate_speculative: SpeculativeConfig.k は 1 以上である必要がある".to_string(),
        ));
    }

    let shape = input_ids.shape();
    let (b, prompt_len, want_rank1) = match *shape {
        [t] => (1usize, t, true),
        [b, t] => (b, t, false),
        _ => {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: shape.len(),
            }));
        }
    };
    if prompt_len == 0 {
        return Err(AutodiffError::InvalidArgument(
            "generate_speculative: prompt（input_ids の系列長）が空である".to_string(),
        ));
    }
    if b != 1 {
        return Err(AutodiffError::InvalidArgument(format!(
            "generate_speculative: B = 1 限定（got B = {b}）"
        )));
    }
    let max_length = config.max_length;
    if max_length < prompt_len {
        return Err(AutodiffError::InvalidArgument(format!(
            "generate_speculative: max_length ({max_length}) は prompt 長 ({prompt_len}) 以上である必要がある"
        )));
    }
    let total_elems = b.checked_mul(max_length).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "generate_speculative: B({b}) * max_length({max_length}) が usize をオーバーフローした"
        ))
    })?;
    let total_bytes = total_elems.checked_mul(std::mem::size_of::<i32>());
    if !matches!(total_bytes, Some(bytes) if bytes <= isize::MAX as usize) {
        return Err(AutodiffError::InvalidArgument(format!(
            "generate_speculative: 出力バッファ（{total_elems} 要素・i32）の確保バイト数が \
             Vec の allocation 上限（isize::MAX バイト）を超える"
        )));
    }
    let (target_layers, draft_layers) = (target.num_kv_layers(), draft.num_kv_layers());
    if target_layers == 0 || draft_layers == 0 {
        return Err(AutodiffError::InvalidArgument(
            "generate_speculative: num_kv_layers() == 0 のモデルは対象外（設計 §10 論点 3）"
                .to_string(),
        ));
    }

    let prompt_contig = input_ids.contiguous();
    let prompt_slice = prompt_contig.host_slice();
    let mut row: Vec<i32> = Vec::with_capacity(max_length);
    row.extend_from_slice(&prompt_slice);
    if max_length == prompt_len {
        return build_output(vec![row], 1, max_length, want_rank1);
    }

    let mut target_caches: Vec<KvCache> = (0..target_layers).map(|_| KvCache::new()).collect();
    let mut draft_caches: Vec<KvCache> = (0..draft_layers).map(|_| KvCache::new()).collect();

    // prefill（target → draft の順。語彙サイズは target で確定し draft を照合する）。
    let prompt_ids =
        Tensor::new(prompt_slice.into_owned(), &[1, prompt_len]).map_err(AutodiffError::Shape)?;
    let t_logits = target.forward_step(&prompt_ids, &mut target_caches)?;
    let vocab = validate_forward_step_output(&t_logits, 1, prompt_len)?;
    config.validate_top_k_le_vocab(vocab)?;
    let d_logits = draft.forward_step(&prompt_ids, &mut draft_caches)?;
    let draft_vocab = validate_forward_step_output(&d_logits, 1, prompt_len)?;
    if draft_vocab != vocab {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: vec![1, prompt_len, draft_vocab],
            rhs: vec![1, prompt_len, vocab],
        }));
    }

    row.push(greedy_token_at(
        &t_logits,
        prompt_len - 1,
        prompt_len,
        vocab,
    )?);
    let mut cur_len = prompt_len + 1;

    while cur_len < max_length {
        let last = row[cur_len - 1];
        let remaining = max_length - cur_len;

        if remaining == 1 {
            let logits = forward_checked(target, &[last], &mut target_caches, vocab)?;
            row.push(greedy_token_at(&logits, 0, 1, vocab)?);
            break;
        }

        // 全受理時の bonus を含めても max_length に収まるよう R - 1 で丸める。
        let k_eff = spec.k.min(remaining - 1);

        // draft の先読み（キャッシュは last, d1..d_{k_eff-1} の分だけ進む）。
        let snap_d = KvSnapshot::capture(draft, &draft_caches)?;
        check_snapshot_len(&snap_d, cur_len)?;
        let mut drafts: Vec<i32> = Vec::with_capacity(k_eff);
        let mut x = last;
        for _ in 0..k_eff {
            let logits = forward_checked(draft, &[x], &mut draft_caches, vocab)?;
            x = greedy_token_at(&logits, 0, 1, vocab)?;
            drafts.push(x);
        }

        // target の一括検証。
        let snap_t = KvSnapshot::capture(target, &target_caches)?;
        check_snapshot_len(&snap_t, cur_len)?;
        let mut verify_ids = Vec::with_capacity(k_eff + 1);
        verify_ids.push(last);
        verify_ids.extend_from_slice(&drafts);
        let logits = forward_checked(target, &verify_ids, &mut target_caches, vocab)?;

        let mut n_acc = 0usize;
        let correction = loop {
            let g = greedy_token_at(&logits, n_acc, k_eff + 1, vocab)?;
            match drafts.get(n_acc) {
                Some(&d) if d == g => n_acc += 1,
                _ => break g,
            }
        };

        row.extend_from_slice(&drafts[..n_acc]);
        row.push(correction);
        cur_len += n_acc + 1;
        if cur_len >= max_length {
            break;
        }

        // 両キャッシュを cur_len_old - 1 + n_acc + 1 へ揃える。
        let mut replay_ids = Vec::with_capacity(n_acc + 1);
        replay_ids.push(last);
        replay_ids.extend_from_slice(&drafts[..n_acc]);
        if n_acc < k_eff {
            rewind_to(target, &mut target_caches, snap_t, &replay_ids, vocab)?;
        }
        if n_acc == k_eff {
            // draft は d_{k_eff} 未 forward。logits は捨てる。
            forward_checked(draft, &[x], &mut draft_caches, vocab)?;
        } else if n_acc + 1 < k_eff {
            rewind_to(draft, &mut draft_caches, snap_d, &replay_ids, vocab)?;
        }
    }

    build_output(vec![row], 1, max_length, want_rank1)
}

#[cfg(test)]
mod tests {
    use super::super::generate;
    use super::*;
    use crate::Tape;
    use crate::nn::{LinearVars, MultiheadAttentionVars};
    use std::cell::Cell;

    const V: usize = 7;
    const E: usize = 4;
    const HEADS: usize = 2;

    type NextFn = fn(i32, usize) -> i32;

    fn target_next(t: i32, p: usize) -> i32 {
        ((3 * t as usize + p + 1) % V) as i32
    }
    fn wrong_next(t: i32, p: usize) -> i32 {
        (target_next(t, p) + 1) % V as i32
    }
    fn mixed_next(t: i32, p: usize) -> i32 {
        if p.is_multiple_of(3) {
            wrong_next(t, p)
        } else {
            target_next(t, p)
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Fault {
        None,
        FailAtCall(usize),
        SkipCache,
        BadVocab,
        BadVocabFrom(usize),
        BadLenAtCall(usize),
        NanAt { call: usize, idx: usize },
    }

    /// logits が `(token, pos)` だけで決まる表引きモデル（形状非依存なので
    /// `generate` と bit 一致を要求できる）。KV キャッシュは実 MHA で進める。
    struct TableModel {
        layers: usize,
        next: NextFn,
        fault: Fault,
        calls: Cell<usize>,
    }

    impl TableModel {
        fn new(layers: usize, next: NextFn, fault: Fault) -> TableModel {
            TableModel {
                layers,
                next,
                fault,
                calls: Cell::new(0),
            }
        }
        fn calls(&self) -> usize {
            self.calls.get()
        }
    }

    fn tf(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("fixture")
    }
    fn seq(seed: i64, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((seed + i as i64 * 7) % 13 - 6) as f32 * 0.05)
            .collect()
    }

    impl AutoregressiveModel for TableModel {
        fn num_kv_layers(&self) -> usize {
            self.layers
        }

        fn forward_step(
            &self,
            new_ids: &Tensor<i32>,
            caches: &mut [KvCache],
        ) -> Result<Tensor<f32>, AutodiffError> {
            let n = self.calls.get() + 1;
            self.calls.set(n);
            if self.fault == Fault::FailAtCall(n) {
                return Err(AutodiffError::InvalidArgument("fixture: 注入失敗".into()));
            }
            let pos0 = caches.first().map_or(0, |c| c.seq_len());
            let tape = Tape::new();
            let emb = tape.var(&tf(seq(1, V * E), &[V, E]));
            let x = emb.embedding(new_ids, None)?;
            let run = if self.fault == Fault::SkipCache {
                0
            } else {
                self.layers
            };
            for cache in caches.iter_mut().take(run) {
                let lin = |s: i64| LinearVars {
                    weight: tape.var(&tf(seq(s, E * E), &[E, E])),
                    bias: Some(tape.var(&tf(seq(s + 1, E), &[E]))),
                };
                let mha = MultiheadAttentionVars::new(HEADS, lin(2), lin(4), lin(6), lin(8))?;
                mha.forward_with_cache(&x, &x, &x, cache)?;
            }
            let toks = new_ids.contiguous().host_slice().into_owned();
            let l = toks.len();
            let vocab = match self.fault {
                Fault::BadVocab => V + 1,
                Fault::BadVocabFrom(from) if n >= from => V + 1,
                _ => V,
            };
            let l_out = if self.fault == Fault::BadLenAtCall(n) {
                l + 1
            } else {
                l
            };
            let mut data = vec![0.0f32; l_out * vocab];
            for (i, &tok) in toks.iter().enumerate() {
                let nx = (self.next)(tok, pos0 + i) as usize;
                data[i * vocab + nx] = 10.0;
            }
            if let Fault::NanAt { call, idx } = self.fault
                && call == n
            {
                data[idx * vocab] = f32::NAN;
            }
            Ok(tf(data, &[1, l_out, vocab]))
        }
    }

    fn prompt(b: usize, t: usize, rank1: bool) -> Tensor<i32> {
        let v: Vec<i32> = (0..b * t).map(|i| (i % V) as i32).collect();
        if rank1 {
            Tensor::new(v, &[t]).unwrap()
        } else {
            Tensor::new(v, &[b, t]).unwrap()
        }
    }

    fn cfg(max_length: usize) -> GenerateConfig {
        GenerateConfig::new(max_length, SamplingStrategy::Greedy)
    }

    fn vals(t: &Tensor<i32>) -> Vec<i32> {
        t.contiguous().host_slice().into_owned()
    }

    /// 規則関数を直接たどった期待列。
    fn expected(prompt: &[i32], max_length: usize) -> Vec<i32> {
        let mut row = prompt.to_vec();
        while row.len() < max_length {
            let p = row.len() - 1;
            row.push(target_next(row[p], p));
        }
        row
    }

    fn err_text(e: AutodiffError) -> String {
        format!("{e}")
    }

    /// generate と一致し、期待列とも一致することを確認する。
    fn check_equal(layers: usize, draft_next: NextFn, t: usize, max: usize, k: usize, rank1: bool) {
        let target = TableModel::new(layers, target_next, Fault::None);
        let draft = TableModel::new(layers, draft_next, Fault::None);
        let ids = prompt(1, t, rank1);
        let out =
            generate_speculative(&target, &draft, &ids, &cfg(max), &SpeculativeConfig::new(k))
                .unwrap();
        let reference = generate(
            &TableModel::new(layers, target_next, Fault::None),
            &ids,
            &cfg(max),
        )
        .unwrap();
        assert_eq!(out.shape(), reference.shape());
        assert_eq!(vals(&out), vals(&reference));
        assert_eq!(vals(&out), expected(&vals(&ids), max));
    }

    #[test]
    fn draft_equals_target_same_instance() {
        for layers in [1, 2] {
            for rank1 in [true, false] {
                let m = TableModel::new(layers, target_next, Fault::None);
                let ids = prompt(1, 3, rank1);
                let out = generate_speculative(&m, &m, &ids, &cfg(14), &SpeculativeConfig::new(4))
                    .unwrap();
                assert_eq!(vals(&out), expected(&vals(&ids), 14));
            }
        }
    }

    #[test]
    fn k_one_and_other_boundaries() {
        for layers in [1, 2] {
            for rank1 in [true, false] {
                for draft_next in [target_next as NextFn, wrong_next, mixed_next] {
                    check_equal(layers, draft_next, 3, 12, 1, rank1);
                    check_equal(layers, draft_next, 2, 12, 3, rank1);
                    // 残り長 < k
                    check_equal(layers, draft_next, 3, 6, 8, rank1);
                    // ラウンド 0 回・R == 1 ステップ
                    check_equal(layers, draft_next, 3, 4, 8, rank1);
                    check_equal(layers, draft_next, 3, 5, 8, rank1);
                }
            }
        }
    }

    #[test]
    fn max_length_equals_prompt_returns_prompt_without_forward() {
        let target = TableModel::new(1, target_next, Fault::None);
        let draft = TableModel::new(1, target_next, Fault::None);
        let ids = prompt(1, 4, false);
        let out = generate_speculative(&target, &draft, &ids, &cfg(4), &SpeculativeConfig::new(3))
            .unwrap();
        assert_eq!(vals(&out), vals(&ids));
        assert_eq!((target.calls(), draft.calls()), (0, 0));
    }

    #[test]
    fn draft_always_right_needs_one_verify_per_round() {
        let target = TableModel::new(1, target_next, Fault::None);
        let draft = TableModel::new(1, target_next, Fault::None);
        let ids = prompt(1, 2, true);
        let out = generate_speculative(&target, &draft, &ids, &cfg(12), &SpeculativeConfig::new(3))
            .unwrap();
        assert_eq!(vals(&out), expected(&vals(&ids), 12));
        // prefill 1 + 検証 2 ラウンド + 最終 1 step
        assert_eq!(target.calls(), 4);
        let plain = TableModel::new(1, target_next, Fault::None);
        generate(&plain, &ids, &cfg(12)).unwrap();
        assert!(target.calls() < plain.calls());
    }

    #[test]
    fn draft_always_wrong_costs_verify_plus_replay() {
        let target = TableModel::new(1, target_next, Fault::None);
        let draft = TableModel::new(1, wrong_next, Fault::None);
        let ids = prompt(1, 2, true);
        let out = generate_speculative(&target, &draft, &ids, &cfg(8), &SpeculativeConfig::new(3))
            .unwrap();
        assert_eq!(vals(&out), expected(&vals(&ids), 8));
        // prefill 1 + (検証 + 再 forward) 4 ラウンド + 最終 1 step
        assert_eq!(target.calls(), 10);
    }

    fn assert_rejected_without_forward(
        target: &TableModel,
        draft: &TableModel,
        ids: &Tensor<i32>,
        config: &GenerateConfig,
        k: usize,
        needle: &str,
    ) {
        let err = generate_speculative(target, draft, ids, config, &SpeculativeConfig::new(k))
            .unwrap_err();
        let text = err_text(err);
        assert!(text.contains(needle), "got: {text}, want: {needle}");
        assert_eq!((target.calls(), draft.calls()), (0, 0));
    }

    fn pair() -> (TableModel, TableModel) {
        (
            TableModel::new(1, target_next, Fault::None),
            TableModel::new(1, target_next, Fault::None),
        )
    }

    #[test]
    fn rejects_invalid_inputs_before_any_forward() {
        let (t, d) = pair();
        let ids = prompt(1, 3, false);
        let topk = GenerateConfig::new(8, SamplingStrategy::TopK(2));
        assert_rejected_without_forward(&t, &d, &ids, &topk, 2, "Greedy");
        let temp = GenerateConfig::new(8, SamplingStrategy::Temperature(0.5));
        assert_rejected_without_forward(&t, &d, &ids, &temp, 2, "Greedy");
        assert_rejected_without_forward(&t, &d, &ids, &cfg(8), 0, "SpeculativeConfig.k");
        let bad = cfg(8).with_temperature(0.5);
        assert_rejected_without_forward(&t, &d, &ids, &bad, 2, "GenerateConfig");
        assert_rejected_without_forward(&t, &d, &prompt(1, 0, false), &cfg(8), 2, "空");
        assert_rejected_without_forward(&t, &d, &prompt(2, 3, false), &cfg(8), 2, "B = 1");
        let empty_b = Tensor::new(Vec::<i32>::new(), &[0, 3]).unwrap();
        assert_rejected_without_forward(&t, &d, &empty_b, &cfg(8), 2, "B = 1");
        assert_rejected_without_forward(&t, &d, &ids, &cfg(2), 2, "max_length");
        assert_rejected_without_forward(&t, &d, &ids, &cfg(usize::MAX), 2, "isize::MAX");
        for shape in [vec![], vec![1, 1, 3]] {
            let n: usize = shape.iter().product();
            let bad_rank = Tensor::new(vec![0i32; n], &shape).unwrap();
            let e = generate_speculative(&t, &d, &bad_rank, &cfg(8), &SpeculativeConfig::new(2))
                .unwrap_err();
            assert!(matches!(
                e,
                AutodiffError::Shape(ShapeError::RankMismatch { .. })
            ));
        }
        assert_eq!((t.calls(), d.calls()), (0, 0));
    }

    #[test]
    fn rejects_models_without_kv_layers() {
        let ids = prompt(1, 3, false);
        for (tl, dl) in [(0, 1), (1, 0)] {
            let t = TableModel::new(tl, target_next, Fault::None);
            let d = TableModel::new(dl, target_next, Fault::None);
            for max in [3, 8] {
                assert_rejected_without_forward(&t, &d, &ids, &cfg(max), 2, "num_kv_layers");
            }
        }
    }

    #[test]
    fn validation_order_is_fixed() {
        let (t, d) = pair();
        let b2 = prompt(2, 3, false);
        let bad = cfg(8).with_temperature(0.5);
        assert_rejected_without_forward(&t, &d, &b2, &bad, 2, "GenerateConfig");
        let topk = GenerateConfig::new(8, SamplingStrategy::TopK(2));
        assert_rejected_without_forward(&t, &d, &b2, &topk, 0, "Greedy");
        let rank3 = Tensor::new(vec![0i32; 3], &[1, 1, 3]).unwrap();
        let e =
            generate_speculative(&t, &d, &rank3, &cfg(8), &SpeculativeConfig::new(0)).unwrap_err();
        assert!(err_text(e).contains("SpeculativeConfig.k"));
        assert_rejected_without_forward(&t, &d, &b2, &cfg(2), 2, "B = 1");
    }

    #[test]
    fn vocab_mismatch_stops_after_draft_prefill() {
        let t = TableModel::new(1, target_next, Fault::None);
        let d = TableModel::new(1, target_next, Fault::BadVocab);
        let e = generate_speculative(
            &t,
            &d,
            &prompt(1, 3, false),
            &cfg(8),
            &SpeculativeConfig::new(2),
        )
        .unwrap_err();
        assert!(matches!(
            e,
            AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
        ));
        assert_eq!((t.calls(), d.calls()), (1, 1));
    }

    #[test]
    fn rejects_bad_shapes_in_verify_and_draft_decode() {
        let ids = prompt(1, 2, true);
        // target 検証 forward（呼び出し 2 回目）の L 軸 +1
        let t = TableModel::new(1, target_next, Fault::BadLenAtCall(2));
        let d = TableModel::new(1, target_next, Fault::None);
        let e =
            generate_speculative(&t, &d, &ids, &cfg(10), &SpeculativeConfig::new(3)).unwrap_err();
        assert!(matches!(
            e,
            AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
        ));
        // draft decode（呼び出し 2 回目以降）の語彙 +1
        let t = TableModel::new(1, target_next, Fault::None);
        let d = TableModel::new(1, target_next, Fault::BadVocabFrom(2));
        let e =
            generate_speculative(&t, &d, &ids, &cfg(10), &SpeculativeConfig::new(3)).unwrap_err();
        assert!(matches!(
            e,
            AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn non_finite_logits_only_rejected_at_used_positions() {
        let ids = prompt(1, 2, true);
        let run = |t: TableModel, d: TableModel| {
            generate_speculative(&t, &d, &ids, &cfg(10), &SpeculativeConfig::new(3))
        };
        let ok = || TableModel::new(1, target_next, Fault::None);
        // prefill 末尾
        let t = TableModel::new(1, target_next, Fault::NanAt { call: 1, idx: 1 });
        assert!(run(t, ok()).is_err());
        // draft decode の位置 0
        let d = TableModel::new(1, target_next, Fault::NanAt { call: 2, idx: 0 });
        assert!(run(ok(), d).is_err());
        // 検証の位置 0
        let t = TableModel::new(1, target_next, Fault::NanAt { call: 2, idx: 0 });
        assert!(run(t, t_wrong()).is_err());
        // 使わない検証位置（常に外す draft では位置 0 だけが使われる）
        let t = TableModel::new(1, target_next, Fault::NanAt { call: 2, idx: 2 });
        let out = run(t, t_wrong()).unwrap();
        assert_eq!(vals(&out), expected(&vals(&ids), 10));
    }

    fn t_wrong() -> TableModel {
        TableModel::new(1, wrong_next, Fault::None)
    }

    #[test]
    fn propagates_forward_errors_from_every_stage() {
        let ids = prompt(1, 2, true);
        // 常に外す draft: target は prefill(1)・検証(2)・再 forward(3)
        for n in 1..=3 {
            let t = TableModel::new(1, target_next, Fault::FailAtCall(n));
            assert!(
                generate_speculative(&t, &t_wrong(), &ids, &cfg(10), &SpeculativeConfig::new(3))
                    .is_err(),
                "target call {n}"
            );
        }
        // 常に外す draft: draft は prefill(1)・decode(2..4)・再 forward(5)
        for n in 1..=5 {
            let d = TableModel::new(1, wrong_next, Fault::FailAtCall(n));
            let t = TableModel::new(1, target_next, Fault::None);
            assert!(
                generate_speculative(&t, &d, &ids, &cfg(10), &SpeculativeConfig::new(3)).is_err(),
                "draft call {n}"
            );
        }
        // 常に当たる draft: draft の補正 decode は 4 回目
        let d = TableModel::new(1, target_next, Fault::FailAtCall(4));
        let t = TableModel::new(1, target_next, Fault::None);
        assert!(generate_speculative(&t, &d, &ids, &cfg(12), &SpeculativeConfig::new(2)).is_err());
    }

    #[test]
    fn cache_not_advancing_model_is_detected() {
        let ids = prompt(1, 2, true);
        let t = TableModel::new(1, target_next, Fault::SkipCache);
        let d = TableModel::new(1, target_next, Fault::None);
        let e =
            generate_speculative(&t, &d, &ids, &cfg(10), &SpeculativeConfig::new(3)).unwrap_err();
        assert!(err_text(e).contains("不変条件"));
        let t = TableModel::new(1, target_next, Fault::None);
        let d = TableModel::new(1, target_next, Fault::SkipCache);
        let e =
            generate_speculative(&t, &d, &ids, &cfg(10), &SpeculativeConfig::new(3)).unwrap_err();
        assert!(err_text(e).contains("不変条件"));
    }
}
