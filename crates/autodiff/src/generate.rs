//! LLM 推論向けの自己回帰生成ループ（イシュー #2191。トークナイザは
//! 対象外——入出力は token id 列〈`Tensor<i32>`〉に限る）。
//!
//! **facade 非公開・保留中**: 本モジュールはイシュー #2191 の受入条件
//! （設定型・3 戦略・KV キャッシュ結線・loop 動作・seed 決定性）を
//! すべてここで満たすが、facade（`fandhe_ai`）公開面への追加は
//! `pub fn generate`／`GenerateConfig` の署名がユーザー承認事項のため
//! 保留する（前例: #2084 K-1/K-2・#2156・#2184/#2180 と同型の 2 段構成）。
//! 承認依頼用の設計・保留の多層固定は `docs/facade-generate-decision.md`
//! を正とする。
//!
//! # KV キャッシュ結線（#2084）
//!
//! [`AutoregressiveModel::forward_step`] は `&self`（不変借用）で呼ばれる
//! ため、`MultiheadAttentionVars::forward_with_cache`（`&mut KvCache` を
//! 要求）や `StatefulAttention::forward`（`&mut self`）をそのまま trait
//! メソッドの型に採れない。本モジュールは「モデル自体は不変・KV
//! キャッシュ配列だけを呼び出し側〈[`generate`]〉が所有し `&mut [KvCache]`
//! として貸し出す」設計を採ることで、この不一致を解消する
//! （`docs/kv-cache-design.md` §2 の mask 規則 (a) prefill・(b) decode に
//! そのまま帰着する——`forward_step` の実装は 1 層ごとに
//! `forward_with_cache` を呼ぶだけでよい）。
//!
//! `forward_step` の戻り値をホスト `Tensor<f32>`（`Tape` を介さない）に
//! している理由: (1) facade 越しのテストからは `fandhe_ai::Tape` の内側
//! （`pub(crate)` の実体）へ到達できず、`&mut Tape` を要求する署名では
//! facade 横断 parity テストが書けない。(2) `docs/kv-cache-design.md`
//! §3.2 が推奨する「step ごとに `Tape` を再作成または `reset` する」
//! 運用を trait 実装側の責務として閉じ込められる。`generate` 自体は
//! `Tape` を一切持たない無状態関数である。
//!
//! # サンプリング
//!
//! 3 戦略（[`SamplingStrategy::Greedy`]／[`SamplingStrategy::TopK`]／
//! [`SamplingStrategy::Temperature`]）はいずれもホスト側 `f64` で行う
//! （バックエンド非依存で決定的になる。CPU／CUDA／Metal のどの
//! `forward_step` 実装でも同じ token 列が得られる）。乱数は
//! [`fandhe_ai_tensor_core::rng::Generator`]（独立インスタンス）のみを
//! 使い、**グローバル RNG（[`crate::manual_seed`] が触れる状態）は
//! 一切消費しない**。`Generator` の実体は xorshift64* であり暗号学的に
//! 安全な PRNG ではない（`rng.rs` と同じ注記。OWASP A02。生成した
//! token をセキュリティ用途に使わないこと）。
//!
//! # HuggingFace `generate()` との既知の差分
//!
//! - EOS による早期停止・`pad_token`・repetition penalty は対象外
//!   （出力は常に `max_length` 到達まで生成する）
//! - `top_k > vocab_size` は HF が黙って clamp するのに対し、本実装は
//!   `AutodiffError::InvalidArgument` で拒否する（fail-closed。
//!   `.claude/rules/security.md` A03 方針）
//! - 出力は token id 列限定（`Tensor<i32>`）。トークナイザ結線は対象外

use fandhe_ai_tensor_core::rng::{Generator, RngError};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::KvCache;

/// 次トークンの選び方（イシュー #2191 受入条件 2）。`#[non_exhaustive]`
/// は他の `AutodiffError` 系列挙型と同じ理由（公開 API 非破壊。
/// `.claude/rules/security.md`）で、将来 nucleus（top-p）等を追加しても
/// 非破壊にするため。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SamplingStrategy {
    /// 各ステップで最大 logit の token を決定的に選ぶ
    /// （`torch.argmax` 相当）。RNG を消費しない。
    Greedy,
    /// 上位 `k` 個の logit のみへ確率質量を残し、温度付き softmax で
    /// 抽選する（`k == 0` または `k > vocab_size` は
    /// [`GenerateConfig`] の検証で拒否する）。
    TopK(usize),
    /// 全 vocab に対する温度付き softmax で抽選する
    /// （`τ`。非有限・非正は検証で拒否する）。
    Temperature(f32),
}

/// [`generate`] の設定（イシュー #2191 受入条件 1）。
///
/// `temperature`／`top_k` は `strategy` から一意に導出される値を保持する
/// 冗長構成を採る（[`GenerateConfig::new`] が導出し、[`GenerateConfig::
/// validate`] が矛盾状態を fail-closed で拒否する）。GAT や private
/// フィールド＋getter 方式ではなく受入条件が挙げる 4 フィールドを字義
/// どおり公開フィールドとして持つ設計は、facade 公開時の承認事項として
/// `docs/facade-generate-decision.md` に記録する。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GenerateConfig {
    /// prompt を含む生成後の全長（HF `generate(max_length=..)` と同義）。
    pub max_length: usize,
    /// 温度 `τ`。`Greedy` では常に `1.0`（未使用）。
    pub temperature: f32,
    /// top-k の `k`。`Greedy`／`Temperature` では常に `None`。
    pub top_k: Option<usize>,
    /// サンプリング戦略。
    pub strategy: SamplingStrategy,
    /// [`Generator::new`] へ渡すシード（既定 `0`。決定的再現性のため
    /// 呼び出し元が明示的に変えない限り固定値になる）。
    pub seed: u64,
}

impl GenerateConfig {
    /// `strategy` から `temperature`／`top_k` を導出して構築する
    /// （`seed` の既定は `0`）。
    pub fn new(max_length: usize, strategy: SamplingStrategy) -> GenerateConfig {
        let (temperature, top_k) = match strategy {
            SamplingStrategy::Greedy => (1.0, None),
            SamplingStrategy::TopK(k) => (1.0, Some(k)),
            SamplingStrategy::Temperature(t) => (t, None),
        };
        GenerateConfig {
            max_length,
            temperature,
            top_k,
            strategy,
            seed: 0,
        }
    }

    /// 温度を上書きする（`TopK` ＋温度の組み合わせ用。HF の
    /// `top_k` ＋ `temperature` 相当）。`Greedy`／`Temperature` に対して
    /// 呼ぶと `Self::validate`（非公開）が矛盾として拒否する（`Greedy` は
    /// `temperature == 1.0` 固定、`Temperature(t)` は `t` 自身が唯一の
    /// 温度値であるため）。
    pub fn with_temperature(mut self, temperature: f32) -> GenerateConfig {
        self.temperature = temperature;
        self
    }

    /// シードを上書きする。
    pub fn with_seed(mut self, seed: u64) -> GenerateConfig {
        self.seed = seed;
        self
    }

    /// [`generate`] の入口で必ず呼ぶ検証（`prompt_len` に依存しない部分）。
    /// 矛盾・不正状態を fail-closed で拒否する（`.claude/rules/
    /// security.md` A03）: `temperature` の有限性・正値、`strategy` と
    /// `temperature`／`top_k` フィールドの整合、`TopK` の `k >= 1`。
    /// `k` の vocab 上限検査は prefill 後の [`Self::validate_top_k_le_vocab`]
    /// が担う（vocab は `AutoregressiveModel::forward_step` の戻り値が
    /// 確定するまで分からないため）。
    fn validate(&self) -> Result<(), AutodiffError> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "GenerateConfig: temperature は有限かつ正である必要がある（got {}）",
                self.temperature
            )));
        }
        match self.strategy {
            SamplingStrategy::Greedy => {
                if self.temperature != 1.0 || self.top_k.is_some() {
                    return Err(AutodiffError::InvalidArgument(
                        "GenerateConfig: SamplingStrategy::Greedy は temperature == 1.0 かつ \
                         top_k == None を要求する"
                            .to_string(),
                    ));
                }
            }
            SamplingStrategy::TopK(k) => {
                if self.top_k != Some(k) {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "GenerateConfig: SamplingStrategy::TopK({k}) は top_k == Some({k}) を \
                         要求する（実際: {:?}）",
                        self.top_k
                    )));
                }
                if k == 0 {
                    return Err(AutodiffError::InvalidArgument(
                        "GenerateConfig: SamplingStrategy::TopK の k は 1 以上である必要がある"
                            .to_string(),
                    ));
                }
            }
            SamplingStrategy::Temperature(t) => {
                if self.top_k.is_some() {
                    return Err(AutodiffError::InvalidArgument(
                        "GenerateConfig: SamplingStrategy::Temperature は top_k == None を \
                         要求する"
                            .to_string(),
                    ));
                }
                if self.temperature != t {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "GenerateConfig: SamplingStrategy::Temperature({t}) は \
                         temperature == {t} を要求する（実際: {}）",
                        self.temperature
                    )));
                }
                if !t.is_finite() || t <= 0.0 {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "GenerateConfig: SamplingStrategy::Temperature の τ は有限かつ正である \
                         必要がある（got {t}）"
                    )));
                }
            }
        }
        Ok(())
    }

    /// prefill で確定した `vocab_size` に対する `top_k` の上限検査
    /// （`k > vocab` は HF の clamp と異なり fail-closed でエラーにする。
    /// モジュール doc「HuggingFace `generate()` との既知の差分」参照）。
    fn validate_top_k_le_vocab(&self, vocab_size: usize) -> Result<(), AutodiffError> {
        if let SamplingStrategy::TopK(k) = self.strategy
            && k > vocab_size
        {
            return Err(AutodiffError::InvalidArgument(format!(
                "GenerateConfig: SamplingStrategy::TopK の k ({k}) が vocab_size \
                 ({vocab_size}) を超えている"
            )));
        }
        Ok(())
    }
}

/// [`generate`] が呼び出すモデル抽象（イシュー #2191）。`&self`（不変
/// 借用）で `caches`（呼び出し側が確保・所有する `KvCache` 配列）を
/// 更新することで、`Module::forward`（`&self`・状態なし）と
/// `StatefulAttention::forward`（`&mut self`）のどちらの制約にも縛られず
/// 「モデル自体は不変・KV キャッシュだけを外部から借用する」設計にする
/// （モジュール doc「KV キャッシュ結線」参照）。
pub trait AutoregressiveModel {
    /// [`generate`] が確保する [`KvCache`] の個数（attention 層数）。
    /// [`Self::forward_step`] へ渡す `caches` スライスの長さと必ず
    /// 一致する。
    fn num_kv_layers(&self) -> usize;

    /// 1 ステップ分の forward（prefill は `new_ids: [B, L_new]` ＝
    /// prompt 全体、decode は `[B, 1]`）。戻り値はホスト `Tensor<f32>`
    /// （`[B, L_new, V]`。`V` は語彙サイズで、[`generate`] は最初の
    /// 呼び出しの戻り shape から確定する）。実装は `Tape` の生成・reset
    /// を自身の責務として行う（モジュール doc 参照）。
    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError>;
}

/// `AutoregressiveModel::forward_step` の戻り値を検査し `vocab_size`
/// を返す（イシュー #2191 セキュリティ考慮: モデル実装のバグで shape が
/// ステップ間で変化しても [`generate`] が誤ったオフセットで host メモリを
/// 読まないようにする fail-closed 検査）。
fn validate_forward_step_output(
    logits: &Tensor<f32>,
    expected_batch: usize,
    expected_l_new: usize,
) -> Result<usize, AutodiffError> {
    let shape = logits.shape();
    if shape.len() != 3 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 3,
            actual: shape.len(),
        }));
    }
    let vocab = shape[2];
    if shape[0] != expected_batch || shape[1] != expected_l_new {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![expected_batch, expected_l_new, vocab],
        }));
    }
    if vocab == 0 {
        return Err(AutodiffError::InvalidArgument(
            "generate: AutoregressiveModel::forward_step の語彙サイズ（末尾軸）が 0".to_string(),
        ));
    }
    Ok(vocab)
}

/// タイの最小 index を選ぶ最大値添字（`Var::argmax` と同じタイ規約。
/// 非有限値は呼び出し元 [`sample_step`] が事前に拒否済みの前提）。
fn greedy_argmax(row: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = row[0];
    for (idx, &v) in row.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    best_idx
}

/// `(logits - max) / τ` を `f64` で計算し正規化する（バックエンド非依存
/// で決定的な softmax。有限入力である前提は呼び出し元が事前検証済み）。
fn softmax_weights_f64(row: &[f32], temperature: f32) -> Vec<f32> {
    let max = row
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &v| acc.max(v as f64));
    let t = temperature as f64;
    let exps: Vec<f64> = row.iter().map(|&v| ((v as f64 - max) / t).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| (e / sum) as f32).collect()
}

/// (値降順・index 昇順) で安定に上位 `k` 個の添字を返す（`k` は
/// 呼び出し元が `1 <= k <= vocab` に検証済み）。
fn top_k_indices(row: &[f32], k: usize) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..row.len()).collect();
    idxs.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idxs.truncate(k);
    idxs
}

/// [`RngError`] を [`AutodiffError::InvalidArgument`] へ写す（実装計画
/// §2.1「`RngError` は `Display` 経由で `InvalidArgument` へ写す」）。
fn rng_error_to_autodiff(context: &str, err: RngError) -> AutodiffError {
    AutodiffError::InvalidArgument(format!("generate: {context}: {err}"))
}

/// 1 ステップ分（`logits: [B, L_new, V]`）から次トークン `[B]` を選ぶ
/// （末尾位置 `L_new - 1` のみを見る——prefill では prompt 末尾、decode
/// では新規 1 トークン自身）。`Greedy` は RNG を消費しない。`TopK`／
/// `Temperature` は `rng`（[`generate`] が一度だけ確保する独立
/// [`Generator`]）を消費する。
fn sample_step(
    logits: &Tensor<f32>,
    b: usize,
    l_new: usize,
    vocab: usize,
    config: &GenerateConfig,
    rng: &mut Generator,
) -> Result<Vec<i32>, AutodiffError> {
    let contiguous = logits.contiguous();
    let data = contiguous.host_slice();
    let last = l_new - 1;
    let mut rows: Vec<&[f32]> = Vec::with_capacity(b);
    for bi in 0..b {
        let start = (bi * l_new + last) * vocab;
        rows.push(&data[start..start + vocab]);
    }
    for row in &rows {
        if row.iter().any(|v| !v.is_finite()) {
            return Err(AutodiffError::InvalidArgument(
                "generate: logits に非有限値が含まれる".to_string(),
            ));
        }
    }

    match config.strategy {
        SamplingStrategy::Greedy => Ok(rows.iter().map(|row| greedy_argmax(row) as i32).collect()),
        SamplingStrategy::Temperature(_) => {
            let mut weights = Vec::with_capacity(b * vocab);
            for row in &rows {
                weights.extend(softmax_weights_f64(row, config.temperature));
            }
            let weights_tensor = Tensor::new(weights, &[b, vocab]).map_err(AutodiffError::Shape)?;
            let sampled = rng
                .multinomial(&weights_tensor, 1, true)
                .map_err(|e| rng_error_to_autodiff("Temperature サンプリング", e))?;
            Ok(sampled.contiguous().host_slice().into_owned())
        }
        SamplingStrategy::TopK(k) => {
            let mut weights = vec![0.0f32; b * vocab];
            for (bi, row) in rows.iter().enumerate() {
                let top = top_k_indices(row, k);
                let top_vals: Vec<f32> = top.iter().map(|&idx| row[idx]).collect();
                let top_weights = softmax_weights_f64(&top_vals, config.temperature);
                for (pos, &orig_idx) in top.iter().enumerate() {
                    weights[bi * vocab + orig_idx] = top_weights[pos];
                }
            }
            let weights_tensor = Tensor::new(weights, &[b, vocab]).map_err(AutodiffError::Shape)?;
            let sampled = rng
                .multinomial(&weights_tensor, 1, true)
                .map_err(|e| rng_error_to_autodiff("TopK サンプリング", e))?;
            Ok(sampled.contiguous().host_slice().into_owned())
        }
    }
}

/// `rows`（各バッチの token id 列。長さ `max_length` で揃っている前提）
/// を出力 `Tensor<i32>` へ組み立てる（`want_rank1` なら `[max_length]`、
/// そうでなければ `[b, max_length]`）。
fn build_output(
    rows: Vec<Vec<i32>>,
    b: usize,
    max_length: usize,
    want_rank1: bool,
) -> Result<Tensor<i32>, AutodiffError> {
    let mut flat = Vec::with_capacity(b * max_length);
    for row in rows {
        flat.extend(row);
    }
    let shape: Vec<usize> = if want_rank1 {
        vec![max_length]
    } else {
        vec![b, max_length]
    };
    Tensor::new(flat, &shape).map_err(AutodiffError::Shape)
}

/// 自己回帰生成ループ本体（イシュー #2191）。`input_ids`（`[T]` または
/// `[B, T]`）を prompt として、`config.max_length` に達するまで
/// `model.forward_step` を繰り返し呼び、`config.strategy` に従って次
/// トークンを選ぶ。`input_ids` が rank 1 なら出力も rank 1（`[T]` →
/// `[max_length]`）、rank 2 なら `[B, T]` → `[B, max_length]`。
///
/// `model` は呼び出しごとに無関係（`generate` 自体は無状態関数）:
/// `caches`（`KvCache` の配列。長さ `model.num_kv_layers()`）は本関数が
/// 新規に確保し、呼び出し元の状態には触れない。
///
/// # Errors
///
/// `input_ids` の rank が 1／2 以外、`T == 0`、`B == 0`、
/// `config.max_length < T`、`B * config.max_length` が `usize` を
/// オーバーフローするか出力バッファ（`Vec<i32>`）の確保バイト数が
/// `Vec` allocation 上限（`isize::MAX` バイト）を超える、`config`
/// 自体が矛盾している（`GenerateConfig::validate`）、
/// `model.forward_step` の戻り shape が期待（`[B, L_new, V]`・`V` は
/// ステップ間で不変）と食い違う、サンプリングに使う位置（`sample_step`
/// が検査する末尾位置 `L_new - 1`。prefill では prompt 末尾、decode
/// では新規 1 トークン自身——サンプリングに使わない他位置の logits は
/// 検査対象外）の logits に非有限値が含まれる、のいずれかで `Err` を
/// 返す。`model.forward_step` 自身が返すエラーはそのまま伝播する。
pub fn generate<M: AutoregressiveModel + ?Sized>(
    model: &M,
    input_ids: &Tensor<i32>,
    config: &GenerateConfig,
) -> Result<Tensor<i32>, AutodiffError> {
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
            "generate: prompt（input_ids の系列長）が空である".to_string(),
        ));
    }
    if b == 0 {
        return Err(AutodiffError::InvalidArgument(
            "generate: batch（input_ids のバッチ次元）が空である".to_string(),
        ));
    }
    if config.max_length < prompt_len {
        return Err(AutodiffError::InvalidArgument(format!(
            "generate: max_length ({}) は prompt 長 ({prompt_len}) 以上である必要がある",
            config.max_length
        )));
    }
    // 出力要素数を事前に検証する（`.claude/rules/security.md` A04
    // 「出力バッファは checked_mul で overflow を検出する」）。
    let total_elems = b.checked_mul(config.max_length).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "generate: B({b}) * max_length({}) が usize をオーバーフローした",
            config.max_length
        ))
    })?;
    // `checked_mul` は `usize` オーバーフローしか検出せず、`b == 1`・
    // 短い prompt・`max_length == usize::MAX` のような入力（積が
    // `usize` に収まる）を素通りさせる。この後 `Vec::with_capacity`
    // （`rows` の各行・`build_output` の `flat`。いずれも要素型は
    // `i32`）が実際に確保するバイト数が `Vec` allocation 上限
    // （`isize::MAX` バイト）を超えると capacity overflow で panic
    // する（本番経路 panic 禁止規約 `.claude/rules/coding-rust.md`
    // に反する。codex-review 指摘・PR #2324 是正。他クレートの同型
    // 検査は `tensor-core::checked_numel_for`・`backend-cpu::linalg::
    // checked_numel_for` 等を参照）。確保前にバイト数も検証し、
    // 確保不能な場合は型付きエラーとして返す。
    let total_bytes = total_elems.checked_mul(std::mem::size_of::<i32>());
    if !matches!(total_bytes, Some(bytes) if bytes <= isize::MAX as usize) {
        return Err(AutodiffError::InvalidArgument(format!(
            "generate: 出力バッファ（B({b}) * max_length({}) 要素・i32）の確保バイト数が Vec の allocation 上限（isize::MAX バイト）を超える",
            config.max_length
        )));
    }

    let prompt_contig = input_ids.contiguous();
    let prompt_slice = prompt_contig.host_slice();
    let mut rows: Vec<Vec<i32>> = Vec::with_capacity(b);
    for bi in 0..b {
        let mut row = Vec::with_capacity(config.max_length);
        row.extend_from_slice(&prompt_slice[bi * prompt_len..(bi + 1) * prompt_len]);
        rows.push(row);
    }

    // `max_length == prompt_len`: forward を一切呼ばず prompt をそのまま
    // 返す（実装計画 §2.4 手順 1）。
    if config.max_length == prompt_len {
        return build_output(rows, b, config.max_length, want_rank1);
    }

    let mut caches: Vec<KvCache> = (0..model.num_kv_layers()).map(|_| KvCache::new()).collect();
    let mut rng = Generator::new(config.seed);

    // prefill: prompt 全体を 1 回で forward する。
    let prompt_ids =
        Tensor::new(prompt_slice.into_owned(), &[b, prompt_len]).map_err(AutodiffError::Shape)?;
    let logits = model.forward_step(&prompt_ids, &mut caches)?;
    let vocab = validate_forward_step_output(&logits, b, prompt_len)?;
    config.validate_top_k_le_vocab(vocab)?;

    let mut next_ids = sample_step(&logits, b, prompt_len, vocab, config, &mut rng)?;
    for (bi, row) in rows.iter_mut().enumerate() {
        row.push(next_ids[bi]);
    }

    // decode: 新規 1 トークンずつ forward する
    // （`AutoregressiveModel::forward_step` doc・`docs/kv-cache-design.md`
    // §2 mask 規則 (b)）。
    let mut cur_len = prompt_len + 1;
    while cur_len < config.max_length {
        let step_ids = Tensor::new(next_ids.clone(), &[b, 1]).map_err(AutodiffError::Shape)?;
        let logits = model.forward_step(&step_ids, &mut caches)?;
        let step_vocab = validate_forward_step_output(&logits, b, 1)?;
        if step_vocab != vocab {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![b, 1, step_vocab],
                rhs: vec![b, 1, vocab],
            }));
        }
        next_ids = sample_step(&logits, b, 1, vocab, config, &mut rng)?;
        for (bi, row) in rows.iter_mut().enumerate() {
            row.push(next_ids[bi]);
        }
        cur_len += 1;
    }

    build_output(rows, b, config.max_length, want_rank1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- GenerateConfig::new・validate --------------------------------

    #[test]
    fn new_derives_fields_from_strategy() {
        let greedy = GenerateConfig::new(8, SamplingStrategy::Greedy);
        assert_eq!(greedy.temperature, 1.0);
        assert_eq!(greedy.top_k, None);
        assert_eq!(greedy.seed, 0);

        let topk = GenerateConfig::new(8, SamplingStrategy::TopK(5));
        assert_eq!(topk.temperature, 1.0);
        assert_eq!(topk.top_k, Some(5));

        let temp = GenerateConfig::new(8, SamplingStrategy::Temperature(0.7));
        assert_eq!(temp.temperature, 0.7);
        assert_eq!(temp.top_k, None);
    }

    #[test]
    fn with_temperature_and_with_seed_override_fields() {
        let config = GenerateConfig::new(8, SamplingStrategy::TopK(3))
            .with_temperature(0.5)
            .with_seed(42);
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn validate_rejects_greedy_with_non_default_temperature() {
        let config = GenerateConfig::new(8, SamplingStrategy::Greedy).with_temperature(0.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_greedy_with_top_k() {
        let mut config = GenerateConfig::new(8, SamplingStrategy::Greedy);
        config.top_k = Some(3);
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_topk_zero() {
        let config = GenerateConfig::new(8, SamplingStrategy::TopK(0));
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_topk_field_mismatch() {
        let mut config = GenerateConfig::new(8, SamplingStrategy::TopK(3));
        config.top_k = Some(4);
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_temperature_non_finite_or_non_positive() {
        assert!(
            GenerateConfig::new(8, SamplingStrategy::Temperature(f32::NAN))
                .validate()
                .is_err()
        );
        assert!(
            GenerateConfig::new(8, SamplingStrategy::Temperature(0.0))
                .validate()
                .is_err()
        );
        assert!(
            GenerateConfig::new(8, SamplingStrategy::Temperature(-1.0))
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_rejects_temperature_with_top_k() {
        let mut config = GenerateConfig::new(8, SamplingStrategy::Temperature(0.8));
        config.top_k = Some(2);
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_accepts_well_formed_configs() {
        assert!(
            GenerateConfig::new(8, SamplingStrategy::Greedy)
                .validate()
                .is_ok()
        );
        assert!(
            GenerateConfig::new(8, SamplingStrategy::TopK(4))
                .validate()
                .is_ok()
        );
        assert!(
            GenerateConfig::new(8, SamplingStrategy::Temperature(0.9))
                .validate()
                .is_ok()
        );
        assert!(
            GenerateConfig::new(8, SamplingStrategy::TopK(4))
                .with_temperature(0.6)
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn validate_top_k_le_vocab_rejects_k_greater_than_vocab() {
        let config = GenerateConfig::new(8, SamplingStrategy::TopK(10));
        assert!(config.validate_top_k_le_vocab(5).is_err());
        assert!(config.validate_top_k_le_vocab(10).is_ok());
        assert!(config.validate_top_k_le_vocab(20).is_ok());
    }

    // --- greedy_argmax --------------------------------------------------

    #[test]
    fn greedy_argmax_breaks_ties_with_smallest_index() {
        assert_eq!(greedy_argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(greedy_argmax(&[5.0]), 0);
        assert_eq!(greedy_argmax(&[-1.0, -2.0, -0.5]), 2);
    }

    // --- softmax_weights_f64 --------------------------------------------

    #[test]
    fn softmax_weights_sum_to_one_and_are_non_negative() {
        let weights = softmax_weights_f64(&[1.0, 2.0, 3.0], 1.0);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn softmax_weights_are_numerically_stable_for_large_logits() {
        // 素朴な exp(logit) 実装だと即座に inf/NaN になる大きな値。
        // max 減算で安定するはず（数値安定性の受入条件）。
        let weights = softmax_weights_f64(&[1e30, 1e30, -1e30], 1.0);
        assert!(weights.iter().all(|w| w.is_finite()));
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn softmax_weights_lower_temperature_sharpens_distribution() {
        let sharp = softmax_weights_f64(&[1.0, 2.0], 0.1);
        let flat = softmax_weights_f64(&[1.0, 2.0], 10.0);
        // 温度が低いほど最大 logit（index 1）へ質量が集中する。
        assert!(sharp[1] > flat[1]);
    }

    // --- top_k_indices ----------------------------------------------------

    #[test]
    fn top_k_indices_orders_by_value_desc_then_index_asc() {
        let row = [1.0, 5.0, 5.0, 3.0, 0.0];
        assert_eq!(top_k_indices(&row, 3), vec![1, 2, 3]);
    }

    #[test]
    fn top_k_indices_truncates_to_k() {
        let row = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(top_k_indices(&row, 1), vec![0]);
        assert_eq!(top_k_indices(&row, 4), vec![0, 2, 3, 1]);
    }

    // --- build_output -----------------------------------------------------

    #[test]
    fn build_output_rank1_flattens_single_row() {
        let out = build_output(vec![vec![1, 2, 3]], 1, 3, true).unwrap();
        assert_eq!(out.shape(), &[3]);
    }

    #[test]
    fn build_output_rank2_preserves_batch_shape() {
        let out = build_output(vec![vec![1, 2], vec![3, 4]], 2, 2, false).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
    }

    // --- validate_forward_step_output --------------------------------

    #[test]
    fn validate_forward_step_output_rejects_wrong_rank() {
        let logits = Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap();
        assert!(validate_forward_step_output(&logits, 1, 1).is_err());
    }

    #[test]
    fn validate_forward_step_output_rejects_shape_mismatch() {
        let logits = Tensor::new(vec![0.0f32; 2 * 3], &[2, 1, 3]).unwrap();
        assert!(validate_forward_step_output(&logits, 3, 1).is_err());
        assert!(validate_forward_step_output(&logits, 2, 2).is_err());
    }

    #[test]
    fn validate_forward_step_output_rejects_zero_vocab() {
        let logits = Tensor::new(Vec::<f32>::new(), &[1, 1, 0]).unwrap();
        assert!(validate_forward_step_output(&logits, 1, 1).is_err());
    }

    #[test]
    fn validate_forward_step_output_returns_vocab_on_success() {
        let logits = Tensor::new(vec![0.0f32; 2 * 5], &[1, 2, 5]).unwrap();
        assert_eq!(validate_forward_step_output(&logits, 1, 2).unwrap(), 5);
    }
}
