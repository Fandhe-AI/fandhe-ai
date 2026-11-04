//! 内部状態を持つモデル（RNN の隠れ状態・SSM の再帰状態など、`KvCache` で表せない状態）向けの
//! trait と speculative decoding の入口（イシュー #3848。親 #3719。
//! 設計の正は `docs/facade-speculative-decoding-batching-design.md` §29・実装記録は §30。
//! 承認は #3770 の所有者回答 <https://github.com/Fandhe-AI/fandhe-ai/issues/3770#issuecomment-6554565145>）。
//!
//! # 役割
//!
//! 案 B2: モデルは不変（`&self`）のまま、状態を呼び出し側が持って `&mut Self::State` で渡す。
//! 棄却時の巻き戻しは trait の明示メソッド（`save_state`／
//! `restore_state`）で行う。`AutoregressiveModel` の
//! supertrait ではなく、`generate_stateful` は設けない（#3770）。
//!
//! 本ファイルは autodiff 側（trait・入口・単体テスト）だけを受け持つ。facade への公開、
//! `BatchScheduler::step_stateful`、`api_surface.rs` の公開形ガードは #3849 の担当。
//!
//! # 共有するもの
//!
//! 検証・サンプリングの純関数（`verify_sampling` 等）は `speculative` の `pub(super)` を
//! そのまま呼ぶ。RNG は `Generator::new(config.seed)` を呼び出しごとに 1 個だけ作り
//! （Greedy では作らない）、消費順は `generate_speculative` と同じ（設計記録 §20.4・§25.2）。

use fandhe_ai_tensor_core::rng::Generator;
use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;

use super::speculative::{
    SpeculativeConfig, greedy_token_at, row_at, sample_from, verify_sampling,
};
use super::{
    GenerateConfig, SamplingStrategy, build_output, sampling_weights_f32,
    validate_forward_step_output,
};

/// 内部状態を持つ自己回帰モデル（案 B2。設計記録 §29.0.2）。
///
/// モデルは不変で、状態（`Self::State`）は呼び出し側〈[`generate_speculative_stateful`]〉が
/// 所有し `&mut` で貸し出す。`State` に境界は付けない（`Send + Sync + 'static` は #3849 の
/// スケジューラ入口が `M::State` に限って要求する）。KV を持つモデルは
/// `State = Vec<KvCache>` などにして実装できる。
///
/// # 契約
///
/// - [`Self::save_state`] は、以後の `state` の更新に影響されない独立した snapshot を返す
/// - [`Self::restore_state`] の後、`state` は退避時点と同じ振る舞いをする
/// - 状態が保持するトークン数は、フレームワークが数える長さと一致させる。ラウンドの
///   不変条件「末尾トークンはどちらのモデルにも未 forward で、状態は `cur_len - 1` トークン分」
///   は外から検査できないため、利用者の保証事項とする
/// - 状態のサイズは利用者の責任（フレームワークは上限を設けない）
pub trait StatefulAutoregressiveModel {
    /// 呼び出し側が持つ状態の型。
    type State;

    /// 空の状態（トークン 0 個分）を作る。
    fn init_state(&self) -> Result<Self::State, AutodiffError>;

    /// `new_ids: [1, L_new]` を forward して `state` を `L_new` トークン分進め、
    /// logits `[1, L_new, V]` を返す（`AutoregressiveModel::forward_step` と同じ形状契約）。
    fn forward_step_stateful(
        &self,
        new_ids: &Tensor<i32>,
        state: &mut Self::State,
    ) -> Result<Tensor<f32>, AutodiffError>;

    /// `state` の独立した snapshot を返す（棄却時の巻き戻し用）。
    fn save_state(&self, state: &Self::State) -> Result<Self::State, AutodiffError>;

    /// `state` を `saved`（[`Self::save_state`] の戻り値）の時点へ戻す。
    fn restore_state(
        &self,
        state: &mut Self::State,
        saved: Self::State,
    ) -> Result<(), AutodiffError>;
}

/// `ids`（長さ `L`）を `[1, L]` で forward し、shape と語彙サイズを検証する。
fn forward_checked_stateful<M: StatefulAutoregressiveModel + ?Sized>(
    model: &M,
    ids: &[i32],
    state: &mut M::State,
    expected_vocab: usize,
) -> Result<Tensor<f32>, AutodiffError> {
    let l = ids.len();
    let input = Tensor::new(ids.to_vec(), &[1, l]).map_err(AutodiffError::Shape)?;
    let logits = model.forward_step_stateful(&input, state)?;
    let vocab = validate_forward_step_output(&logits, 1, l)?;
    if vocab != expected_vocab {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: vec![1, l, vocab],
            rhs: vec![1, l, expected_vocab],
        }));
    }
    Ok(logits)
}

/// 退避 snapshot へ戻して `[last, accepted...]` を再 forward する（logits は捨てるが
/// shape と語彙は検査する）。失敗は加工せず返す。
fn rewind_stateful<M: StatefulAutoregressiveModel + ?Sized>(
    model: &M,
    state: &mut M::State,
    snap: M::State,
    replay_ids: &[i32],
    vocab: usize,
) -> Result<(), AutodiffError> {
    model.restore_state(state, snap)?;
    forward_checked_stateful(model, replay_ids, state, vocab)?;
    Ok(())
}

/// 内部状態を持つモデル版の speculative decoding（Greedy／TopK／Temperature）。
/// 契約は [`super::speculative::generate_speculative`] と同じで、KV の代わりに
/// [`StatefulAutoregressiveModel`] の状態を `save_state`／`restore_state` で巻き戻す
/// （設計記録 §29・§30）。`input_ids` は `[T]` または `[1, T]`、出力は `[max_length]` または
/// `[1, max_length]`。`draft == target`（同一インスタンス可）でも動作する。
///
/// target・draft のどちらも新しい trait を実装する（KV 形のモデルは `State = Vec<KvCache>`）。
///
/// # 契約
///
/// - token 列が `generate` と一致するのは、形状に依存しない logits を返すモデルに限る
///   （設計記録 §6.1 と同じ）
/// - サンプリング版の契約（受理棄却・残差再抽出・RNG 消費順・グローバル RNG 非消費）は
///   設計記録 §20 を引き継ぐ。生成 token をセキュリティ用途に使わない（xorshift64*。非暗号）
/// - 巻き戻し（`restore_state`）や再 forward が失敗したら直ちに `Err` を返す。状態はこの関数が
///   `init_state` で作ったローカル値で `Err` と共に drop されるため呼び出し側から観測できない。
///   ただし利用者の `State` が `Arc` などで外部と共有されている場合、失敗後の中身は
///   利用者の責任となる（設計記録 §30 の明確化）
/// - 状態が持つトークン数をフレームワークの数える長さと一致させるのは利用者の保証事項
///
/// # Errors
///
/// 次の順に検査し、最初に該当した条件で `Err` を返す（前段で拒否する場合、
/// `init_state` も `forward_step_stateful` も呼ばれない）。
///
/// 1. `config` が矛盾している
/// 2. `spec.k == 0`
/// 3. `input_ids` の rank が 1／2 以外
/// 4. `T == 0`
/// 5. `B != 1`（`B == 0` を含む）
/// 6. `config.max_length < T`
/// 7. `B * max_length` の `usize` オーバーフロー、または出力バッファが `isize::MAX` バイトを超える
/// 8. `target.init_state()` の失敗（そのまま伝播。`max_length == T` でも呼ばれる）
/// 9. `draft.init_state()` の失敗（同上）
///
/// 以降は forward の戻り shape の不整合、語彙サイズの target／draft 間・ステップ間不一致、
/// `top_k > vocab`、トークン選択に使う位置の logits の非有限値、サンプリング時の
/// 確率計算・`RngError` の fail-closed、およびモデルが返すエラー（`save_state`・
/// `restore_state` を含めそのまま伝播）で `Err` を返す。`generate_speculative` と違い
/// `num_kv_layers() == 0` の拒否は無い（純 RNN は KV 層 0 が正当なため）。
pub fn generate_speculative_stateful<T, D>(
    target: &T,
    draft: &D,
    input_ids: &Tensor<i32>,
    config: &GenerateConfig,
    spec: &SpeculativeConfig,
) -> Result<Tensor<i32>, AutodiffError>
where
    T: StatefulAutoregressiveModel + ?Sized,
    D: StatefulAutoregressiveModel + ?Sized,
{
    config.validate()?;
    if spec.k == 0 {
        return Err(AutodiffError::InvalidArgument(
            "generate_speculative_stateful: SpeculativeConfig.k は 1 以上である必要がある"
                .to_string(),
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
            "generate_speculative_stateful: prompt（input_ids の系列長）が空である".to_string(),
        ));
    }
    if b != 1 {
        return Err(AutodiffError::InvalidArgument(format!(
            "generate_speculative_stateful: B = 1 限定（got B = {b}）"
        )));
    }
    let max_length = config.max_length;
    if max_length < prompt_len {
        return Err(AutodiffError::InvalidArgument(format!(
            "generate_speculative_stateful: max_length ({max_length}) は prompt 長 ({prompt_len}) 以上である必要がある"
        )));
    }
    let total_elems = b.checked_mul(max_length).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "generate_speculative_stateful: B({b}) * max_length({max_length}) が usize をオーバーフローした"
        ))
    })?;
    let total_bytes = total_elems.checked_mul(std::mem::size_of::<i32>());
    if !matches!(total_bytes, Some(bytes) if bytes <= isize::MAX as usize) {
        return Err(AutodiffError::InvalidArgument(format!(
            "generate_speculative_stateful: 出力バッファ（{total_elems} 要素・i32）の確保バイト数が \
             Vec の allocation 上限（isize::MAX バイト）を超える"
        )));
    }

    // モデル側の検査は早期 return より前に置く（generate_speculative の検査 8 と同じ位置づけ）。
    let mut target_state = target.init_state()?;
    let mut draft_state = draft.init_state()?;

    let prompt_contig = input_ids.contiguous();
    let prompt_slice = prompt_contig.host_slice();
    let mut row: Vec<i32> = Vec::with_capacity(max_length);
    row.extend_from_slice(&prompt_slice);
    if max_length == prompt_len {
        return build_output(vec![row], 1, max_length, want_rank1);
    }

    // prefill（target → draft の順。語彙サイズは target で確定し draft を照合する）。
    let prompt_ids =
        Tensor::new(prompt_slice.into_owned(), &[1, prompt_len]).map_err(AutodiffError::Shape)?;
    let t_logits = target.forward_step_stateful(&prompt_ids, &mut target_state)?;
    let vocab = validate_forward_step_output(&t_logits, 1, prompt_len)?;
    config.validate_top_k_le_vocab(vocab)?;
    let d_logits = draft.forward_step_stateful(&prompt_ids, &mut draft_state)?;
    let draft_vocab = validate_forward_step_output(&d_logits, 1, prompt_len)?;
    if draft_vocab != vocab {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: vec![1, prompt_len, draft_vocab],
            rhs: vec![1, prompt_len, vocab],
        }));
    }

    let mut rng: Option<Generator> =
        (!matches!(config.strategy, SamplingStrategy::Greedy)).then(|| Generator::new(config.seed));
    let first = match rng.as_mut() {
        None => greedy_token_at(&t_logits, prompt_len - 1, prompt_len, vocab)?,
        Some(g) => {
            let r = row_at(&t_logits, prompt_len - 1, prompt_len, vocab)?;
            sample_from(g, sampling_weights_f32(&r, config), "prefill サンプリング")?
        }
    };
    row.push(first);
    let mut cur_len = prompt_len + 1;

    while cur_len < max_length {
        let last = row[cur_len - 1];
        let remaining = max_length - cur_len;

        if remaining == 1 {
            let logits = forward_checked_stateful(target, &[last], &mut target_state, vocab)?;
            let tok = match rng.as_mut() {
                None => greedy_token_at(&logits, 0, 1, vocab)?,
                Some(g) => {
                    let r = row_at(&logits, 0, 1, vocab)?;
                    sample_from(
                        g,
                        sampling_weights_f32(&r, config),
                        "最終 step サンプリング",
                    )?
                }
            };
            row.push(tok);
            break;
        }

        // 全受理時の bonus を含めても max_length に収まるよう R - 1 で丸める。
        let k_eff = spec.k.min(remaining - 1);

        // draft の先読み（状態は last, d1..d_{k_eff-1} の分だけ進む）。
        let snap_d = draft.save_state(&draft_state)?;
        let mut drafts: Vec<i32> = Vec::with_capacity(k_eff);
        let mut q_ws: Vec<Vec<f32>> = Vec::with_capacity(k_eff);
        let mut x = last;
        for _ in 0..k_eff {
            let logits = forward_checked_stateful(draft, &[x], &mut draft_state, vocab)?;
            match rng.as_mut() {
                None => x = greedy_token_at(&logits, 0, 1, vocab)?,
                Some(g) => {
                    let r = row_at(&logits, 0, 1, vocab)?;
                    let w = sampling_weights_f32(&r, config);
                    x = sample_from(g, w.clone(), "draft サンプリング")?;
                    q_ws.push(w);
                }
            }
            drafts.push(x);
        }

        // target の一括検証。
        let snap_t = target.save_state(&target_state)?;
        let mut verify_ids = Vec::with_capacity(k_eff + 1);
        verify_ids.push(last);
        verify_ids.extend_from_slice(&drafts);
        let logits = forward_checked_stateful(target, &verify_ids, &mut target_state, vocab)?;

        let (n_acc, correction) = match rng.as_mut() {
            None => {
                let mut n_acc = 0usize;
                let correction = loop {
                    let g = greedy_token_at(&logits, n_acc, k_eff + 1, vocab)?;
                    match drafts.get(n_acc) {
                        Some(&d) if d == g => n_acc += 1,
                        _ => break g,
                    }
                };
                (n_acc, correction)
            }
            Some(g) => verify_sampling(g, &logits, &drafts, &q_ws, config, vocab)?,
        };

        row.extend_from_slice(&drafts[..n_acc]);
        row.push(correction);
        cur_len += n_acc + 1;
        if cur_len >= max_length {
            break;
        }

        // 両状態を cur_len_old - 1 + n_acc + 1 トークン分へ揃える。
        let mut replay_ids = Vec::with_capacity(n_acc + 1);
        replay_ids.push(last);
        replay_ids.extend_from_slice(&drafts[..n_acc]);
        if n_acc < k_eff {
            rewind_stateful(target, &mut target_state, snap_t, &replay_ids, vocab)?;
        }
        if n_acc == k_eff {
            // draft は d_{k_eff} 未 forward。logits は捨てる。
            forward_checked_stateful(draft, &[x], &mut draft_state, vocab)?;
        } else if n_acc + 1 < k_eff {
            rewind_stateful(draft, &mut draft_state, snap_d, &replay_ids, vocab)?;
        }
    }

    build_output(vec![row], 1, max_length, want_rank1)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    const V: usize = 7;
    type NextFn = fn(i32, usize) -> i32;

    fn target_next(t: i32, p: usize) -> i32 {
        ((3 * t as usize + p + 1) % V) as i32
    }
    fn wrong_next(t: i32, p: usize) -> i32 {
        (target_next(t, p) + 1) % V as i32
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Fault {
        None,
        InitFail,
        ForwardFailAtCall(usize),
        SaveFailAtCall(usize),
        RestoreFail,
        BadVocab,
        BadVocabFrom(usize),
        BadLenAtCall(usize),
        NanAt { call: usize, idx: usize },
    }

    /// 状態が「処理済みトークン数」だけの純 RNN 相当モデル（KV 層なし）。logits は
    /// `(token, 絶対位置)` の one-hot なので形状に依らず決定的。
    struct RnnModel {
        next: NextFn,
        fault: Fault,
        inits: Cell<usize>,
        forwards: Cell<usize>,
        saves: Cell<usize>,
        restores: Cell<usize>,
    }

    impl RnnModel {
        fn new(next: NextFn, fault: Fault) -> RnnModel {
            RnnModel {
                next,
                fault,
                inits: Cell::new(0),
                forwards: Cell::new(0),
                saves: Cell::new(0),
                restores: Cell::new(0),
            }
        }
        fn counts(&self) -> (usize, usize, usize, usize) {
            (
                self.inits.get(),
                self.forwards.get(),
                self.saves.get(),
                self.restores.get(),
            )
        }
    }

    fn injected() -> AutodiffError {
        AutodiffError::InvalidArgument("fixture: 注入失敗".into())
    }

    impl StatefulAutoregressiveModel for RnnModel {
        type State = usize;

        fn init_state(&self) -> Result<usize, AutodiffError> {
            self.inits.set(self.inits.get() + 1);
            if self.fault == Fault::InitFail {
                return Err(injected());
            }
            Ok(0)
        }

        fn forward_step_stateful(
            &self,
            new_ids: &Tensor<i32>,
            state: &mut usize,
        ) -> Result<Tensor<f32>, AutodiffError> {
            let n = self.forwards.get() + 1;
            self.forwards.set(n);
            if self.fault == Fault::ForwardFailAtCall(n) {
                return Err(injected());
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
                data[i * vocab + (self.next)(tok, *state + i) as usize] = 10.0;
            }
            if let Fault::NanAt { call, idx } = self.fault
                && call == n
            {
                data[idx * vocab] = f32::NAN;
            }
            *state += l;
            Tensor::new(data, &[1, l_out, vocab]).map_err(AutodiffError::Shape)
        }

        fn save_state(&self, state: &usize) -> Result<usize, AutodiffError> {
            let n = self.saves.get() + 1;
            self.saves.set(n);
            if self.fault == Fault::SaveFailAtCall(n) {
                return Err(injected());
            }
            Ok(*state)
        }

        fn restore_state(&self, state: &mut usize, saved: usize) -> Result<(), AutodiffError> {
            self.restores.set(self.restores.get() + 1);
            if self.fault == Fault::RestoreFail {
                return Err(injected());
            }
            *state = saved;
            Ok(())
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
    fn expected(prompt: &[i32], max_length: usize) -> Vec<i32> {
        let mut row = prompt.to_vec();
        while row.len() < max_length {
            let p = row.len() - 1;
            row.push(target_next(row[p], p));
        }
        row
    }
    fn ok_pair() -> (RnnModel, RnnModel) {
        (
            RnnModel::new(target_next, Fault::None),
            RnnModel::new(target_next, Fault::None),
        )
    }
    fn run(
        t: &RnnModel,
        d: &RnnModel,
        ids: &Tensor<i32>,
        config: &GenerateConfig,
        k: usize,
    ) -> Result<Tensor<i32>, AutodiffError> {
        generate_speculative_stateful(t, d, ids, config, &SpeculativeConfig::new(k))
    }

    fn assert_rejected_untouched(
        t: &RnnModel,
        d: &RnnModel,
        ids: &Tensor<i32>,
        config: &GenerateConfig,
        k: usize,
        needle: &str,
    ) {
        let e = run(t, d, ids, config, k).unwrap_err();
        assert!(format!("{e}").contains(needle), "{e} に {needle} が無い");
        assert_eq!(t.counts(), (0, 0, 0, 0));
        assert_eq!(d.counts(), (0, 0, 0, 0));
    }

    #[test]
    fn accepts_pure_rnn_models_without_kv_layers() {
        for draft_next in [target_next as NextFn, wrong_next] {
            for rank1 in [true, false] {
                let t = RnnModel::new(target_next, Fault::None);
                let d = RnnModel::new(draft_next, Fault::None);
                let ids = prompt(1, 3, rank1);
                let out = run(&t, &d, &ids, &cfg(14), 3).unwrap();
                assert_eq!(vals(&out), expected(&vals(&ids), 14));
                assert_eq!(out.shape().len(), if rank1 { 1 } else { 2 });
            }
        }
    }

    #[test]
    fn rejects_invalid_inputs_before_any_model_call() {
        let (t, d) = ok_pair();
        let ids = prompt(1, 3, false);
        assert_rejected_untouched(&t, &d, &ids, &cfg(8), 0, "SpeculativeConfig.k");
        let bad = cfg(8).with_temperature(0.5);
        assert_rejected_untouched(&t, &d, &ids, &bad, 2, "GenerateConfig");
        assert_rejected_untouched(&t, &d, &prompt(1, 0, false), &cfg(8), 2, "空");
        assert_rejected_untouched(&t, &d, &prompt(2, 3, false), &cfg(8), 2, "B = 1");
        let empty_b = Tensor::new(Vec::<i32>::new(), &[0, 3]).unwrap();
        assert_rejected_untouched(&t, &d, &empty_b, &cfg(8), 2, "B = 1");
        assert_rejected_untouched(&t, &d, &ids, &cfg(2), 2, "max_length");
        assert_rejected_untouched(&t, &d, &ids, &cfg(usize::MAX), 2, "isize::MAX");
        for shape in [vec![], vec![1, 1, 3]] {
            let n: usize = shape.iter().product();
            let bad_rank = Tensor::new(vec![0i32; n], &shape).unwrap();
            let e = run(&t, &d, &bad_rank, &cfg(8), 2).unwrap_err();
            assert!(matches!(
                e,
                AutodiffError::Shape(ShapeError::RankMismatch { .. })
            ));
        }
        assert_eq!(t.counts(), (0, 0, 0, 0));
    }

    #[test]
    fn validation_order_is_fixed() {
        let (t, d) = ok_pair();
        let b2 = prompt(2, 3, false);
        let bad = cfg(8).with_temperature(0.5);
        assert_rejected_untouched(&t, &d, &b2, &bad, 2, "GenerateConfig");
        let topk = GenerateConfig::new(8, SamplingStrategy::TopK(2));
        assert_rejected_untouched(&t, &d, &b2, &topk, 0, "SpeculativeConfig.k");
        let rank3 = Tensor::new(vec![0i32; 3], &[1, 1, 3]).unwrap();
        let e = run(&t, &d, &rank3, &cfg(8), 0).unwrap_err();
        assert!(format!("{e}").contains("SpeculativeConfig.k"));
        assert_rejected_untouched(&t, &d, &b2, &cfg(2), 2, "B = 1");
    }

    #[test]
    fn init_state_failures_propagate_in_order_target_then_draft() {
        let ids = prompt(1, 3, false);
        let t = RnnModel::new(target_next, Fault::InitFail);
        let d = RnnModel::new(target_next, Fault::InitFail);
        assert!(run(&t, &d, &ids, &cfg(8), 2).is_err());
        assert_eq!((t.counts().0, d.counts().0), (1, 0));
        let (t, _) = ok_pair();
        let d = RnnModel::new(target_next, Fault::InitFail);
        assert!(run(&t, &d, &ids, &cfg(8), 2).is_err());
        assert_eq!((t.counts().0, d.counts().0), (1, 1));
        assert_eq!(t.counts().1, 0);
    }

    #[test]
    fn max_length_equals_prompt_inits_but_never_forwards() {
        let (t, d) = ok_pair();
        let ids = prompt(1, 3, true);
        let out = run(&t, &d, &ids, &cfg(3), 2).unwrap();
        assert_eq!(vals(&out), vals(&ids));
        assert_eq!(t.counts(), (1, 0, 0, 0));
        assert_eq!(d.counts(), (1, 0, 0, 0));
    }

    #[test]
    fn full_accept_never_restores() {
        let (t, d) = ok_pair();
        run(&t, &d, &prompt(1, 2, true), &cfg(12), 3).unwrap();
        // target: prefill 1 + 検証 2 + 最終 1。save は各ラウンド 1 回、restore は 0 回。
        assert_eq!(t.counts(), (1, 4, 2, 0));
        // draft: prefill 1 + 先読み 3×2 + 全受理後の追加 forward 1×2
        assert_eq!(d.counts(), (1, 9, 2, 0));
    }

    #[test]
    fn first_position_rejection_restores_both() {
        // k_eff = 1（remaining = 2）: n_acc = 0 == k_eff - 1 なので draft は restore しない。
        let t = RnnModel::new(target_next, Fault::None);
        let d = RnnModel::new(wrong_next, Fault::None);
        run(&t, &d, &prompt(1, 2, true), &cfg(5), 3).unwrap();
        assert_eq!((t.counts().2, t.counts().3), (1, 1));
        assert_eq!((d.counts().2, d.counts().3), (1, 0));
        // k_eff = 2 で 1 位置目棄却は両方 restore（n_acc + 1 < k_eff）。
        let t = RnnModel::new(target_next, Fault::None);
        let d = RnnModel::new(wrong_next, Fault::None);
        // 常に外す draft: k_eff = 2 のラウンドが 2 回（restore は両方）、続く k_eff = 1 は draft のみ不要。
        run(&t, &d, &prompt(1, 2, true), &cfg(7), 2).unwrap();
        assert_eq!(t.counts().3, 3);
        assert_eq!(d.counts().3, 2);
    }

    #[test]
    fn last_position_rejection_restores_target_only() {
        fn last_wrong(t: i32, p: usize) -> i32 {
            if p == 3 {
                wrong_next(t, p)
            } else {
                target_next(t, p)
            }
        }
        let t = RnnModel::new(target_next, Fault::None);
        let d = RnnModel::new(last_wrong, Fault::None);
        // k_eff = 2、先読みは位置 2・3。2 個目（位置 3）が外れ n_acc + 1 == k_eff。
        // 2 ラウンド目（k_eff = 1）は全受理で restore なし。
        run(&t, &d, &prompt(1, 2, true), &cfg(7), 2).unwrap();
        assert_eq!((t.counts().2, t.counts().3), (2, 1));
        assert_eq!((d.counts().2, d.counts().3), (2, 0));
    }

    #[test]
    fn model_errors_propagate_from_every_stage() {
        let ids = prompt(1, 2, true);
        for call in 1..=3 {
            let t = RnnModel::new(target_next, Fault::ForwardFailAtCall(call));
            let d = RnnModel::new(wrong_next, Fault::None);
            assert!(run(&t, &d, &ids, &cfg(8), 2).is_err(), "target call {call}");
        }
        for call in 1..=4 {
            let t = RnnModel::new(target_next, Fault::None);
            let d = RnnModel::new(wrong_next, Fault::ForwardFailAtCall(call));
            assert!(run(&t, &d, &ids, &cfg(8), 2).is_err(), "draft call {call}");
        }
        for call in 1..=2 {
            let t = RnnModel::new(target_next, Fault::SaveFailAtCall(call));
            let d = RnnModel::new(wrong_next, Fault::None);
            assert!(run(&t, &d, &ids, &cfg(8), 2).is_err(), "target save {call}");
            let t = RnnModel::new(target_next, Fault::None);
            let d = RnnModel::new(wrong_next, Fault::SaveFailAtCall(call));
            assert!(run(&t, &d, &ids, &cfg(8), 2).is_err(), "draft save {call}");
        }
        let t = RnnModel::new(target_next, Fault::RestoreFail);
        let d = RnnModel::new(wrong_next, Fault::None);
        let e = run(&t, &d, &ids, &cfg(8), 2).unwrap_err();
        assert!(format!("{e}").contains("注入失敗"));
        let t = RnnModel::new(target_next, Fault::None);
        let d = RnnModel::new(wrong_next, Fault::RestoreFail);
        assert!(run(&t, &d, &ids, &cfg(10), 3).is_err());
    }

    #[test]
    fn shape_and_vocab_mismatches_fail_closed() {
        let ids = prompt(1, 2, true);
        let t = RnnModel::new(target_next, Fault::None);
        let d = RnnModel::new(target_next, Fault::BadVocab);
        let e = run(&t, &d, &ids, &cfg(8), 2).unwrap_err();
        assert!(matches!(
            e,
            AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
        ));
        let t = RnnModel::new(target_next, Fault::None);
        let d = RnnModel::new(target_next, Fault::BadVocabFrom(2));
        assert!(run(&t, &d, &ids, &cfg(8), 2).is_err());
        let t = RnnModel::new(target_next, Fault::BadLenAtCall(2));
        let d = RnnModel::new(target_next, Fault::None);
        assert!(run(&t, &d, &ids, &cfg(8), 2).is_err());
        let (t, d) = ok_pair();
        let topk = GenerateConfig::new(8, SamplingStrategy::TopK(V + 1));
        assert!(run(&t, &d, &ids, &topk, 2).is_err());
        assert_eq!((t.counts().1, d.counts().1), (1, 0));
        let t = RnnModel::new(target_next, Fault::NanAt { call: 1, idx: 1 });
        let d = RnnModel::new(target_next, Fault::None);
        assert!(run(&t, &d, &ids, &cfg(8), 2).is_err());
    }

    #[test]
    fn sampling_is_deterministic_and_top_k_one_matches_greedy() {
        let sampled = |config: &GenerateConfig| {
            let (t, d) = ok_pair();
            vals(&run(&t, &d, &prompt(1, 3, true), config, 3).unwrap())
        };
        let temp = GenerateConfig::new(14, SamplingStrategy::Temperature(0.8));
        assert_eq!(sampled(&temp), sampled(&temp));
        let top1 = GenerateConfig::new(12, SamplingStrategy::TopK(1));
        assert_eq!(sampled(&top1), sampled(&cfg(12)));
    }
}
