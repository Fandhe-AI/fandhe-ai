//! 連続バッチングのスケジューラ（第 1 段階。イシュー #2888。設計の正は
//! `docs/facade-speculative-decoding-batching-design.md` §6.3・§7・§8.1・§8.3）。
//!
//! # 役割
//!
//! イテレーション単位で要求の参加・退出ができる同期スケジューラ。要求ごとに
//! `Vec<KvCache>` と独立した [`Generator`] を所有し、[`BatchScheduler::step`] 1 回で
//! 進行中の各要求を 1 トークンずつ進める。各要求は B = 1 で `forward_step` を呼び、
//! prefill／decode の forward 形状・検証・サンプリング・RNG 消費順を親モジュールの
//! `generate` と同一にする（`super::` の非公開ヘルパー `sample_step`・
//! `validate_forward_step_output`・`build_output` を再利用する）。このため各要求の
//! 出力は単独 `generate` と token 列が完全一致する構成になる（網羅的な一致テストは
//! #2889、実機 parity は #2890 の担当）。
//!
//! # 対象モデルの契約（設計 §8.1）
//!
//! | モデルの形 | 扱い |
//! |---|---|
//! | 渡された `caches` だけに状態を持つ（`num_kv_layers() > 0`） | 対象 |
//! | 状態を持たない（`num_kv_layers() == 0` で呼び出し間で結果が独立） | 対象 |
//! | 生成状態を、渡された `caches` の外に持つ型（`RefCell<StatefulAttention>` 等） | **対象外**（出力を保証しない） |
//!
//! 内部状態保持型とは「生成状態を、渡された `caches` の外に持つ型」であり、
//! `num_kv_layers()` の値では決まらない（`> 0` を返しつつ内部にも状態を持つ型も作れる）。
//! 型でも実行時でも検出できず、要求間で状態が混ざるため、利用者が対象モデルの形を
//! 保証すること（設計記録 §17.5）。
//!
//! # 資源上限・失敗・完了条件
//!
//! - 同時実行数・待ち行列長・要求ごとの `max_length` の上限は [`SchedulerLimits::new`]
//!   で必須（0 は拒否。暗黙の無制限を作らない。OWASP A04）。超過は
//!   `Err(AutodiffError::InvalidArgument)`
//! - 要求の途中で `forward_step` 等が `Err` を返した場合は当該要求だけを切り離して
//!   [`BatchScheduler::take_failed`] に積む。他要求のキャッシュ・RNG・生成済み token は
//!   別インスタンスのため構造的に変わらない。`forward_step` 内の panic は捕捉しない
//! - 完了条件は `max_length` 到達のみ（EOS 停止は扱わない）
//! - B > 1 の要求は対象外（要求ごとに B = 1。拒否する）
//! - テンソル単位のバッチ化（設計 §8.2）・性能保証は持たない
//! - 呼び出しスレッド上の同期実行のみ（設計 §8.3 の制約に従い、標準ライブラリの
//!   コレクションのみを使う）
//!
//! # 公開範囲
//!
//! 本モジュールは facade `fandhe_ai::inference` へ純再エクスポート済み（イシュー #2934。
//! `RequestId`・上限の型・`step` の `model` 引数の型・失敗の表現型〈`take_failed`〉は
//! 承認で確定。設計記録 §17.3・§17.4）。 `Generator` は xorshift64* の非暗号 PRNG であり、生成 token を
//! セキュリティ用途に使わないこと（OWASP A02。親モジュールと同じ注記）。
//! 未回収の完了・失敗結果は呼び出し側が `take_*` で回収する前提（件数は submit 済み
//! 要求数で抑えられる）。

use std::collections::VecDeque;

use fandhe_ai_tensor_core::rng::Generator;
use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::KvCache;

use super::{
    AutoregressiveModel, GenerateConfig, build_output, sample_step, validate_forward_step_output,
};

/// スケジューラが採番する要求の識別子（不透明。`submit` ごとに単調増加）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(u64);

/// スケジューラの資源上限（構築時必須。A04）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SchedulerLimits {
    max_active: usize,
    max_queued: usize,
    max_length: usize,
}

impl SchedulerLimits {
    /// 同時実行数・待ち行列長・要求ごとの `max_length` の上限を指定して構築する。
    /// いずれかが 0、または `max_length` 分の `i32` バッファが `isize::MAX` バイトを
    /// 超える場合は `Err(InvalidArgument)`。
    pub fn new(
        max_active: usize,
        max_queued: usize,
        max_length: usize,
    ) -> Result<SchedulerLimits, AutodiffError> {
        if max_active == 0 || max_queued == 0 || max_length == 0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "SchedulerLimits: 各上限は 1 以上である必要がある（max_active={max_active}, \
                 max_queued={max_queued}, max_length={max_length}）"
            )));
        }
        check_buffer_bytes(max_length)?;
        Ok(SchedulerLimits {
            max_active,
            max_queued,
            max_length,
        })
    }

    /// 同時に進行できる要求数の上限。
    pub fn max_active(&self) -> usize {
        self.max_active
    }

    /// 待ち行列に積める要求数の上限。
    pub fn max_queued(&self) -> usize {
        self.max_queued
    }

    /// 要求ごとの `GenerateConfig::max_length` の上限。
    pub fn max_length(&self) -> usize {
        self.max_length
    }
}

/// `max_length` 個の `i32` バッファが `Vec` の allocation 上限内か検査する
/// （`generate` と同方式。確保前に型付きエラーにして capacity overflow の panic を避ける）。
fn check_buffer_bytes(max_length: usize) -> Result<(), AutodiffError> {
    let bytes = max_length.checked_mul(std::mem::size_of::<i32>());
    if matches!(bytes, Some(b) if b <= isize::MAX as usize) {
        Ok(())
    } else {
        Err(AutodiffError::InvalidArgument(format!(
            "scheduler: max_length ({max_length}) 要素・i32 の確保バイト数が Vec の allocation \
             上限（isize::MAX バイト）を超える"
        )))
    }
}

/// 待ち行列上の要求（prompt は所有して複製済み）。
struct Pending {
    id: RequestId,
    prompt: Vec<i32>,
    config: GenerateConfig,
    want_rank1: bool,
}

/// 進行中の要求。キャッシュ・RNG・token 列を要求ごとに独立して所有する。
struct Active {
    id: RequestId,
    /// prompt ＋生成済み token。
    tokens: Vec<i32>,
    caches: Vec<KvCache>,
    rng: Generator,
    /// prefill 前は `None`。prefill の戻り shape で確定する。
    vocab: Option<usize>,
    config: GenerateConfig,
    want_rank1: bool,
}

impl Active {
    /// 1 トークン進める。`Ok(Some(出力))` は `max_length` 到達で完了。
    /// 制御フローは `generate` の prefill／decode を逐語的に再現する。
    fn advance<M: AutoregressiveModel + ?Sized>(
        &mut self,
        model: &M,
    ) -> Result<Option<Tensor<i32>>, AutodiffError> {
        if self.caches.len() != model.num_kv_layers() {
            return Err(AutodiffError::InvalidArgument(format!(
                "scheduler: キャッシュ数 ({}) が model.num_kv_layers() ({}) と一致しない \
                 （step 間で異なるモデルが渡された）",
                self.caches.len(),
                model.num_kv_layers()
            )));
        }
        let next = match self.vocab {
            None => {
                let len = self.tokens.len();
                let ids =
                    Tensor::new(self.tokens.clone(), &[1, len]).map_err(AutodiffError::Shape)?;
                let logits = model.forward_step(&ids, &mut self.caches)?;
                let vocab = validate_forward_step_output(&logits, 1, len)?;
                self.config.validate_top_k_le_vocab(vocab)?;
                let next = sample_step(&logits, 1, len, vocab, &self.config, &mut self.rng)?;
                self.vocab = Some(vocab);
                next
            }
            Some(vocab) => {
                let last = self.tokens.last().copied().ok_or_else(|| {
                    AutodiffError::InvalidArgument("scheduler: token 列が空である".to_string())
                })?;
                let ids = Tensor::new(vec![last], &[1, 1]).map_err(AutodiffError::Shape)?;
                let logits = model.forward_step(&ids, &mut self.caches)?;
                let step_vocab = validate_forward_step_output(&logits, 1, 1)?;
                if step_vocab != vocab {
                    return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                        lhs: vec![1, 1, step_vocab],
                        rhs: vec![1, 1, vocab],
                    }));
                }
                sample_step(&logits, 1, 1, vocab, &self.config, &mut self.rng)?
            }
        };
        let tok = next.first().copied().ok_or_else(|| {
            AutodiffError::InvalidArgument("scheduler: サンプリング結果が空である".to_string())
        })?;
        self.tokens.push(tok);
        if self.tokens.len() >= self.config.max_length {
            let tokens = std::mem::take(&mut self.tokens);
            let out = build_output(vec![tokens], 1, self.config.max_length, self.want_rank1)?;
            return Ok(Some(out));
        }
        Ok(None)
    }
}

/// 連続バッチングのスケジューラ（第 1 段階）。モジュール doc の契約に従う。
///
/// 対象外: 生成状態を、渡された `caches` の外に持つ型（内部状態保持型）は要求間で状態が
/// 混ざるため出力を保証しない。`num_kv_layers()` の値によらず、型でも実行時でも検出できない
/// ため、対象モデルの形は利用者が保証する（設計記録 §17.5）。
pub struct BatchScheduler {
    limits: SchedulerLimits,
    queue: VecDeque<Pending>,
    /// 参加順（＝`RequestId` 昇順）に並ぶ。処理順は決定的。
    active: Vec<Active>,
    finished: Vec<(RequestId, Tensor<i32>)>,
    failed: Vec<(RequestId, AutodiffError)>,
    next_id: u64,
}

impl std::fmt::Debug for BatchScheduler {
    /// token 列を出さず件数のみを出す。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchScheduler")
            .field("limits", &self.limits)
            .field("queued", &self.queue.len())
            .field("active", &self.active.len())
            .field("finished", &self.finished.len())
            .field("failed", &self.failed.len())
            .finish()
    }
}

impl BatchScheduler {
    /// 上限（検証済み）を指定して空のスケジューラを作る。
    pub fn new(limits: SchedulerLimits) -> BatchScheduler {
        BatchScheduler {
            limits,
            queue: VecDeque::new(),
            active: Vec::new(),
            finished: Vec::new(),
            failed: Vec::new(),
            next_id: 0,
        }
    }

    /// 待ち行列にいる要求数。
    pub fn queued_len(&self) -> usize {
        self.queue.len()
    }

    /// 進行中の要求数。
    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    /// 要求を待ち行列へ積む（forward は呼ばない）。検査順は `generate` と同じ
    /// （config → rank → `T == 0` → `B`）に上限検査を加えたもの。いずれの `Err` でも
    /// スケジューラの状態は変えない。`input_ids` は `[T]` または `[1, T]`
    /// （B > 1 は対象外として拒否）。
    pub fn submit(
        &mut self,
        input_ids: &Tensor<i32>,
        config: &GenerateConfig,
    ) -> Result<RequestId, AutodiffError> {
        config.validate()?;
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
                "scheduler: prompt（input_ids の系列長）が空である".to_string(),
            ));
        }
        if b != 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "scheduler: 要求ごとの batch は 1 のみ対象（got {b}）"
            )));
        }
        if config.max_length < prompt_len {
            return Err(AutodiffError::InvalidArgument(format!(
                "scheduler: max_length ({}) は prompt 長 ({prompt_len}) 以上である必要がある",
                config.max_length
            )));
        }
        if config.max_length > self.limits.max_length {
            return Err(AutodiffError::InvalidArgument(format!(
                "scheduler: max_length ({}) が上限 ({}) を超えている",
                config.max_length, self.limits.max_length
            )));
        }
        check_buffer_bytes(config.max_length)?;
        if self.queue.len() >= self.limits.max_queued {
            return Err(AutodiffError::InvalidArgument(format!(
                "scheduler: 待ち行列が上限 ({}) に達している",
                self.limits.max_queued
            )));
        }
        let next = self.next_id.checked_add(1).ok_or_else(|| {
            AutodiffError::InvalidArgument("scheduler: RequestId の採番が尽きた".to_string())
        })?;
        let prompt = input_ids.contiguous().host_slice().into_owned();
        let id = RequestId(self.next_id);
        self.next_id = next;
        self.queue.push_back(Pending {
            id,
            prompt,
            config: config.clone(),
            want_rank1,
        });
        Ok(id)
    }

    /// 1 イテレーション進める: 待ち行列から空きの分だけ参加させ、進行中の各要求を
    /// 1 トークン進める。戻り値はこの呼び出しで成功完了した要求数（失敗は含めない）。
    ///
    /// 要求単位の失敗は `Err` にせず [`Self::take_failed`] に積む。`Err` はスケジューラ
    /// 全体の不変条件破れに予約する（第 1 段階では返す経路はない）。`model` は保持せず
    /// 呼び出しごとに借用する。内部状態保持型モデルは対象外（モジュール doc）。
    pub fn step<M: AutoregressiveModel + ?Sized>(
        &mut self,
        model: &M,
    ) -> Result<usize, AutodiffError> {
        let mut completed = 0usize;

        while self.active.len() < self.limits.max_active {
            let Some(p) = self.queue.pop_front() else {
                break;
            };
            if p.config.max_length == p.prompt.len() {
                // forward を呼ばず prompt をそのまま返す（`generate` の早期 return と同じ）。
                let max_length = p.config.max_length;
                match build_output(vec![p.prompt], 1, max_length, p.want_rank1) {
                    Ok(out) => {
                        self.finished.push((p.id, out));
                        completed += 1;
                    }
                    Err(e) => self.failed.push((p.id, e)),
                }
                continue;
            }
            let mut tokens = Vec::with_capacity(p.config.max_length);
            tokens.extend_from_slice(&p.prompt);
            self.active.push(Active {
                id: p.id,
                tokens,
                caches: (0..model.num_kv_layers()).map(|_| KvCache::new()).collect(),
                rng: Generator::new(p.config.seed),
                vocab: None,
                config: p.config,
                want_rank1: p.want_rank1,
            });
        }

        let current = std::mem::take(&mut self.active);
        for mut a in current {
            match a.advance(model) {
                Ok(None) => self.active.push(a),
                Ok(Some(out)) => {
                    self.finished.push((a.id, out));
                    completed += 1;
                }
                Err(e) => self.failed.push((a.id, e)),
            }
        }
        Ok(completed)
    }

    /// 完了した要求を完了順に取り出す（drain。2 回目は空）。
    pub fn take_finished(&mut self) -> Vec<(RequestId, Tensor<i32>)> {
        std::mem::take(&mut self.finished)
    }

    /// 失敗として切り離された要求を取り出す（drain）。`AutodiffError` は `Clone`
    /// でないため所有権ごと移す（失敗の表現型は未承認の内部暫定選択）。
    pub fn take_failed(&mut self) -> Vec<(RequestId, AutodiffError)> {
        std::mem::take(&mut self.failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;
    use crate::nn::{LinearVars, MultiheadAttentionVars};
    use std::cell::Cell;

    use super::super::{SamplingStrategy, generate};

    const V: usize = 8;

    fn ids1(v: Vec<i32>) -> Tensor<i32> {
        let n = v.len();
        Tensor::new(v, &[n]).expect("fixture")
    }

    fn ids2(v: Vec<i32>) -> Tensor<i32> {
        let n = v.len();
        Tensor::new(v, &[1, n]).expect("fixture")
    }

    /// 状態なしの表引きモデル。各位置の token から次 token を `(3t+1)%V` と定める。
    /// `fail_decode_on`: decode で当該 token を受けたら Err。
    /// `bad_vocab_on`: decode で当該 token を受けたら語彙を V+1 にする。
    struct TableModel {
        fail_decode_on: Option<i32>,
        bad_vocab_on: Option<i32>,
        calls: Cell<usize>,
    }

    impl TableModel {
        fn new() -> TableModel {
            TableModel {
                fail_decode_on: None,
                bad_vocab_on: None,
                calls: Cell::new(0),
            }
        }
    }

    impl AutoregressiveModel for TableModel {
        fn num_kv_layers(&self) -> usize {
            0
        }

        fn forward_step(
            &self,
            new_ids: &Tensor<i32>,
            _caches: &mut [KvCache],
        ) -> Result<Tensor<f32>, AutodiffError> {
            self.calls.set(self.calls.get() + 1);
            let (b, l) = (new_ids.shape()[0], new_ids.shape()[1]);
            let data = new_ids.contiguous().host_slice().into_owned();
            let decode = l == 1;
            let mut vocab = V;
            let mut out = Vec::new();
            for &tok in data.iter() {
                if decode && self.fail_decode_on == Some(tok) {
                    return Err(AutodiffError::InvalidArgument("fixture: 注入失敗".into()));
                }
                if decode && self.bad_vocab_on == Some(tok) {
                    vocab = V + 1;
                }
            }
            for &tok in data.iter() {
                let next = (tok * 3 + 1).rem_euclid(V as i32) as usize;
                for v in 0..vocab {
                    out.push(if v == next { 2.0 } else { 0.1 * (v % 3) as f32 });
                }
            }
            Ok(Tensor::new(out, &[b, l, vocab]).expect("fixture"))
        }
    }

    fn seq(seed: i64, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((seed + i as i64 * 7) % 13 - 6) as f32 * 0.05)
            .collect()
    }

    /// `Embedding → MHA（KV キャッシュ付き）× 2 → lm head` の KV モデル。
    struct KvModel;

    impl AutoregressiveModel for KvModel {
        fn num_kv_layers(&self) -> usize {
            2
        }

        fn forward_step(
            &self,
            new_ids: &Tensor<i32>,
            caches: &mut [KvCache],
        ) -> Result<Tensor<f32>, AutodiffError> {
            const E: usize = 4;
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

    fn limits(a: usize, q: usize, m: usize) -> SchedulerLimits {
        SchedulerLimits::new(a, q, m).expect("fixture")
    }

    fn cfg(max_length: usize, strategy: SamplingStrategy, seed: u64) -> GenerateConfig {
        GenerateConfig::new(max_length, strategy).with_seed(seed)
    }

    fn run_to_end<M: AutoregressiveModel>(
        s: &mut BatchScheduler,
        m: &M,
    ) -> Vec<(RequestId, Tensor<i32>)> {
        let mut out = Vec::new();
        for _ in 0..64 {
            if s.queued_len() == 0 && s.active_len() == 0 {
                break;
            }
            s.step(m).expect("step");
            out.extend(s.take_finished());
        }
        out
    }

    fn data(t: &Tensor<i32>) -> Vec<i32> {
        t.contiguous().host_slice().into_owned()
    }

    #[test]
    fn limits_reject_zero_and_huge() {
        assert!(SchedulerLimits::new(0, 1, 1).is_err());
        assert!(SchedulerLimits::new(1, 0, 1).is_err());
        assert!(SchedulerLimits::new(1, 1, 0).is_err());
        assert!(SchedulerLimits::new(1, 1, usize::MAX).is_err());
        assert!(SchedulerLimits::new(1, 1, isize::MAX as usize / 4 + 1).is_err());
        let l = SchedulerLimits::new(2, 3, 4).expect("ok");
        assert_eq!((l.max_active(), l.max_queued(), l.max_length()), (2, 3, 4));
    }

    #[test]
    fn submit_rejections_leave_state_unchanged() {
        let mut s = BatchScheduler::new(limits(1, 2, 6));
        let g = cfg(4, SamplingStrategy::Greedy, 0);
        let bad_cfg = cfg(4, SamplingStrategy::Greedy, 0).with_temperature(2.0);
        assert!(s.submit(&ids1(vec![1]), &bad_cfg).is_err());
        let rank3 = Tensor::new(vec![1, 2], &[1, 1, 2]).expect("fixture");
        assert!(matches!(
            s.submit(&rank3, &g),
            Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
        ));
        let empty = Tensor::new(vec![], &[0]).expect("fixture");
        assert!(s.submit(&empty, &g).is_err());
        let b2 = Tensor::new(vec![1, 2], &[2, 1]).expect("fixture");
        assert!(s.submit(&b2, &g).is_err());
        assert!(s.submit(&ids1(vec![1, 2, 3, 4, 5]), &g).is_err());
        assert!(
            s.submit(&ids1(vec![1]), &cfg(7, SamplingStrategy::Greedy, 0))
                .is_err()
        );
        assert_eq!(s.queued_len(), 0);
        s.submit(&ids1(vec![1]), &g).expect("1");
        s.submit(&ids1(vec![1]), &g).expect("2");
        assert!(s.submit(&ids1(vec![1]), &g).is_err());
        assert_eq!(s.queued_len(), 2);
    }

    #[test]
    fn request_ids_are_monotonic() {
        let mut s = BatchScheduler::new(limits(1, 3, 6));
        let g = cfg(4, SamplingStrategy::Greedy, 0);
        let a = s.submit(&ids1(vec![1]), &g).expect("a");
        let b = s.submit(&ids1(vec![1]), &g).expect("b");
        assert!(a < b);
    }

    #[test]
    fn admission_is_bounded_by_max_active() {
        let m = TableModel::new();
        let mut s = BatchScheduler::new(limits(1, 2, 6));
        let g = cfg(3, SamplingStrategy::Greedy, 0);
        s.submit(&ids1(vec![1]), &g).expect("a");
        s.submit(&ids1(vec![2]), &g).expect("b");
        s.step(&m).expect("step");
        assert_eq!(m.calls.get(), 1);
        assert_eq!((s.active_len(), s.queued_len()), (1, 1));
        let mut done = s.take_finished();
        done.extend(run_to_end(&mut s, &m));
        assert_eq!(done.len(), 2);
        assert!(done[0].0 < done[1].0);
    }

    #[test]
    fn prompt_only_request_finishes_without_forward() {
        let m = TableModel::new();
        let mut s = BatchScheduler::new(limits(1, 1, 6));
        s.submit(&ids2(vec![3, 4]), &cfg(2, SamplingStrategy::Greedy, 0))
            .expect("a");
        assert_eq!(s.step(&m).expect("step"), 1);
        assert_eq!(m.calls.get(), 0);
        let done = s.take_finished();
        assert_eq!(done[0].1.shape(), &[1, 2]);
        assert_eq!(data(&done[0].1), vec![3, 4]);
        assert!(s.take_finished().is_empty());
    }

    #[test]
    fn empty_step_does_not_call_model() {
        let m = TableModel::new();
        let mut s = BatchScheduler::new(limits(1, 1, 6));
        assert_eq!(s.step(&m).expect("step"), 0);
        assert_eq!(m.calls.get(), 0);
    }

    fn check_matches_single<M: AutoregressiveModel>(m: &M) {
        let reqs = [
            (ids1(vec![1, 2]), cfg(6, SamplingStrategy::Greedy, 0)),
            (ids2(vec![3]), cfg(5, SamplingStrategy::TopK(3), 7)),
            (
                ids1(vec![5, 6, 7]),
                cfg(7, SamplingStrategy::Temperature(0.8), 11),
            ),
        ];
        let mut s = BatchScheduler::new(limits(2, 4, 8));
        let mut idv = Vec::new();
        for (i, c) in &reqs {
            idv.push(s.submit(i, c).expect("submit"));
        }
        let done = run_to_end(&mut s, m);
        assert_eq!(done.len(), reqs.len());
        for (id, (i, c)) in idv.iter().zip(reqs.iter()) {
            let got = &done.iter().find(|(d, _)| d == id).expect("done").1;
            let want = generate(m, i, c).expect("generate");
            assert_eq!(got.shape(), want.shape());
            assert_eq!(data(got), data(&want));
        }
    }

    #[test]
    fn interleaved_matches_single_generate_stateless() {
        check_matches_single(&TableModel::new());
    }

    #[test]
    fn interleaved_matches_single_generate_kv() {
        check_matches_single(&KvModel);
    }

    #[test]
    fn failure_is_isolated_to_one_request() {
        // 連鎖: A は {1,4,5,0}、C は {2,7,6,3}。6 を decode で受けた C だけが失敗する。
        let mut m = TableModel::new();
        m.fail_decode_on = Some(6);
        let a = (ids1(vec![1]), cfg(8, SamplingStrategy::TopK(1), 3));
        let c = (ids1(vec![2]), cfg(8, SamplingStrategy::Greedy, 0));
        let mut s = BatchScheduler::new(limits(2, 2, 8));
        let ia = s.submit(&a.0, &a.1).expect("a");
        let ic = s.submit(&c.0, &c.1).expect("c");
        let done = run_to_end(&mut s, &m);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].0, ia);
        let want = generate(&TableModel::new(), &a.0, &a.1).expect("generate");
        assert_eq!(data(&done[0].1), data(&want));
        let failed = s.take_failed();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, ic);
        assert!(s.take_failed().is_empty());
    }

    #[test]
    fn vocab_mismatch_on_decode_fails_only_that_request() {
        let mut m = TableModel::new();
        m.bad_vocab_on = Some(7);
        let mut s = BatchScheduler::new(limits(2, 2, 8));
        s.submit(&ids1(vec![1]), &cfg(5, SamplingStrategy::Greedy, 0))
            .expect("a");
        let ic = s
            .submit(&ids1(vec![2]), &cfg(5, SamplingStrategy::Greedy, 0))
            .expect("c");
        let done = run_to_end(&mut s, &m);
        assert_eq!(done.len(), 1);
        let failed = s.take_failed();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, ic);
        assert!(matches!(failed[0].1, AutodiffError::Shape(_)));
    }

    #[test]
    fn model_layer_count_change_fails_request() {
        let mut s = BatchScheduler::new(limits(1, 1, 6));
        s.submit(&ids1(vec![1]), &cfg(4, SamplingStrategy::Greedy, 0))
            .expect("a");
        s.step(&KvModel).expect("step");
        s.step(&TableModel::new()).expect("step");
        assert_eq!(s.take_failed().len(), 1);
        assert_eq!(s.active_len(), 0);
    }
}
