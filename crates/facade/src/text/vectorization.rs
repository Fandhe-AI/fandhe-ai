//! 公開型 `TextVectorization`／`TextVectorizationConfig`（語彙 lookup 型のテキスト変換）。
//!
//! 役割: `fandhe_ai::text` の入口。内部部品（`vocab::Vocabulary`・`adapt::adapt`・
//! `transform::transform`・`limits::TextLimits`）を設定 1 つにまとめて呼ぶだけの薄い
//! 合成層で、新しいアルゴリズムは持たない（設計記録 §16.4）。呼び出し元は利用者
//! コードで、出力の `Tensor<i32>` は `Var::embedding(.., Some(0))` へ変換なしで渡せる。
//!
//! 構築時検査（設計記録 §8・§16.4）: `max_tokens`・`ngrams`・`output_sequence_length`
//! は `from_vocabulary`／`adapt` の入口で、語彙やコーパスの確保・走査より先に検査し、
//! 不正なら黙って無視も切り詰めもせず型付きエラーで返す（A03・A04）。
//! `transform` 側の再検査は多重防御として残る。`Debug` は設定と語彙件数だけを出し、
//! 語彙の全文は出さない（A09）。
//!
//! 出力モード（#3092。決定記録 `docs/facade-text-output-modes-string-lookup-decision.md`
//! §3・§5）: `transform_multi_hot`／`transform_count`／`transform_tf_idf` は
//! `bag::bag` を呼ぶだけの薄い入口で、`[B, W]` の `Tensor<f32>`（列 i = id i）を返す。
//! tf_idf の idf 重みは設定（`Eq` を保つため）ではなくインスタンスが持ち、
//! `from_vocabulary_with_idf`／`adapt_tf_idf` で作ったものだけが `transform_tf_idf` を呼べる。

use std::fmt;

use super::adapt::{adapt, adapt_with_df};
use super::bag::{BagKind, BagOptions, bag};
use super::error::TextError;
use super::limits::TextLimits;
use super::split::Split;
use super::standardize::Standardize;
use super::transform::{TransformOptions, transform};
use super::vocab::{RESERVED_COUNT, Vocabulary};
use crate::Tensor;

/// `TextVectorization` の設定。`Default` と `with_*` ビルダで組み立てる。
///
/// 既定値は `max_tokens: None`・`standardize: LowerAndStripPunctuation`・
/// `split: Whitespace`・`ngrams: None`・`output_sequence_length: None`・
/// `limits: TextLimits::default()`。`Option` のフィールドは `with_*` で `Some` を
/// 設定する（`None` へ戻すには `Default` かフィールド代入）。値の検査は
/// ビルダでは行わず、`TextVectorization::from_vocabulary`／`adapt` の構築時に行う。
///
/// - `max_tokens`: 予約 2 件を含む語彙数の上限。2 以下は `InvalidMaxTokens`
/// - `ngrams`: `Some(n)` で 1〜n-gram を作る（`n` は 1 以上かつ `limits` の上限以下）
/// - `output_sequence_length`: `Some(L)` で `L` へ切り詰め／0 埋め。`None` は
///   バッチ内の最長に合わせる（出力モード系メソッドでは無視される）
/// - `pad_to_max_tokens`: 既定 `false`。`true` なら出力モード系メソッドの列数を
///   `max_tokens` に固定する（`max_tokens` が `None` だと構築時に
///   `PadToMaxTokensWithoutMaxTokens`）。`false` は語彙数（予約 2 件を含む）
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TextVectorizationConfig {
    /// 予約 2 件を含む語彙数の上限。`None` は無制限（`limits` の上限は適用される）。
    pub max_tokens: Option<usize>,
    /// 標準化の方式（既定は ASCII。Unicode 小文字化は `UnicodeLower` で opt-in）。
    pub standardize: Standardize,
    /// 分割の方式（既定は ASCII 空白。Unicode White_Space は `UnicodeWhitespace` で opt-in）。
    pub split: Split,
    /// `Some(n)` なら 1〜n-gram を作る。
    pub ngrams: Option<usize>,
    /// `Some(L)` なら出力長を `L` に固定する。
    pub output_sequence_length: Option<usize>,
    /// 出力モード系メソッドの列数を `max_tokens` へ固定するか（既定 `false`）。
    pub pad_to_max_tokens: bool,
    /// 非信頼入力の上限。
    pub limits: TextLimits,
}

impl TextVectorizationConfig {
    /// `max_tokens` に `Some(max_tokens)` を設定する。
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
    /// 標準化の方式を設定する。
    pub fn with_standardize(mut self, standardize: Standardize) -> Self {
        self.standardize = standardize;
        self
    }
    /// 分割の方式を設定する。
    pub fn with_split(mut self, split: Split) -> Self {
        self.split = split;
        self
    }
    /// `ngrams` に `Some(n)` を設定する。
    pub fn with_ngrams(mut self, n: usize) -> Self {
        self.ngrams = Some(n);
        self
    }
    /// `output_sequence_length` に `Some(len)` を設定する。
    pub fn with_output_sequence_length(mut self, len: usize) -> Self {
        self.output_sequence_length = Some(len);
        self
    }
    /// `pad_to_max_tokens` を設定する。`true` には `max_tokens` の指定が必要。
    pub fn with_pad_to_max_tokens(mut self, pad: bool) -> Self {
        self.pad_to_max_tokens = pad;
        self
    }
    /// 非信頼入力の上限を設定する。
    pub fn with_limits(mut self, limits: TextLimits) -> Self {
        self.limits = limits;
        self
    }
}

/// 語彙 lookup 型のテキスト変換（Keras `TextVectorization` 相当）。
///
/// 文字列バッチを標準化・分割（任意で n-gram）し、語彙で id 列へ引いて
/// `[B, L]` の `Tensor<i32>` にする。id は 0 = パディング、1 = OOV、2.. = 語彙。
/// 語彙は `from_vocabulary`（直接指定）か `adapt`（コーパスから構築）で作る。
/// 契約の詳細はモジュール doc（`fandhe_ai::text`）を参照。
#[derive(Clone)]
pub struct TextVectorization {
    config: TextVectorizationConfig,
    vocabulary: Vocabulary,
    /// id 順の idf 重み（長さ = 語彙数。`idf[0]` = 0.0、`idf[1]` = OOV、`idf[2..]` = 語彙）。
    /// `from_vocabulary_with_idf`／`adapt_tf_idf` で作ったときだけ `Some`。
    idf: Option<Vec<f32>>,
}

/// `pad_to_max_tokens` と `max_tokens` の整合検査（確保・走査より先）。
fn check_pad(config: &TextVectorizationConfig) -> Result<(), TextError> {
    if config.pad_to_max_tokens && config.max_tokens.is_none() {
        return Err(TextError::PadToMaxTokensWithoutMaxTokens);
    }
    Ok(())
}

/// 語彙の idf 重み（長さ = 語彙の個数）から id 順の重みを組む。
/// `idf[0]` = 0.0、`idf[1]`（OOV）= 語彙の重みの平均（f64 で合計し 1 回だけ f32 へ。空なら 0.0）、
/// `idf[2..]` = 語彙の重み。`adapt_tf_idf`・`from_vocabulary_with_idf` が共通で使う。
fn assemble_idf(vocab_weights: &[f32]) -> Vec<f32> {
    let oov = if vocab_weights.is_empty() {
        0.0f32
    } else {
        let sum: f64 = vocab_weights.iter().map(|&w| f64::from(w)).sum();
        (sum / vocab_weights.len() as f64) as f32
    };
    let mut idf = Vec::with_capacity(vocab_weights.len() + RESERVED_COUNT);
    idf.push(0.0f32);
    idf.push(oov);
    idf.extend_from_slice(vocab_weights);
    idf
}

/// `max_tokens` の検査（`Some(m)` で m が 2 以下は `InvalidMaxTokens`）。
fn check_max_tokens(max_tokens: Option<usize>) -> Result<(), TextError> {
    match max_tokens {
        Some(m) if m <= RESERVED_COUNT => Err(TextError::InvalidMaxTokens { max_tokens: m }),
        _ => Ok(()),
    }
}

/// `ngrams`・`output_sequence_length` の構築時検査（確保より先）。
fn check_config_numbers(config: &TextVectorizationConfig) -> Result<(), TextError> {
    if let Some(n) = config.ngrams {
        config.limits.check_ngrams(n)?;
    }
    if let Some(l) = config.output_sequence_length {
        config.limits.check_output_sequence_length(l)?;
    }
    Ok(())
}

impl TextVectorization {
    /// 語彙を直接指定して作る。id は入力順に 2 から振る（0 = `""`、1 = `"[UNK]"`）。
    ///
    /// 語彙は標準化されず、lookup はバイト完全一致（大小文字を区別する）。
    /// `""` は `EmptyVocabularyToken`、`"[UNK]"` は `ReservedVocabularyToken`、
    /// 重複は `DuplicateVocabularyToken`。`max_tokens` が `Some(m)` のとき、`m` が 2 以下は
    /// `InvalidMaxTokens`、予約 2 件を含む語彙数が `m` を超えると
    /// `VocabularyTooLarge`（黙って切り詰めない。確保より先に検査する）。
    pub fn from_vocabulary<S: AsRef<str>>(
        config: TextVectorizationConfig,
        vocabulary: &[S],
    ) -> Result<Self, TextError> {
        check_pad(&config)?;
        check_max_tokens(config.max_tokens)?;
        if let Some(m) = config.max_tokens {
            let total = vocabulary.len().saturating_add(RESERVED_COUNT);
            if total > m {
                return Err(TextError::VocabularyTooLarge { len: total, max: m });
            }
        }
        check_config_numbers(&config)?;
        let vocabulary = Vocabulary::from_tokens(&config.limits, vocabulary)?;
        Ok(Self {
            config,
            vocabulary,
            idf: None,
        })
    }

    /// 語彙と idf 重みを直接指定して作る（`transform_tf_idf` 用）。
    ///
    /// `idf_weights` は語彙と同じ順・同じ個数（予約 2 件を含まない）。各値は有限かつ 0 以上
    /// （`-0.0` は可）。id 順の重みは `idf[0]` = 0.0、`idf[1]`（OOV）= 与えた重みの平均
    /// （f64 で合計し 1 回だけ f32 へ。語彙が空なら 0.0）、`idf[2..]` = 与えた重み。
    /// 検査順（確保より先）: `pad_to_max_tokens` → `max_tokens`・語彙数・設定値 →
    /// 個数の一致（`IdfWeightsLengthMismatch`）→ 各値（`InvalidIdfWeight`）→ 語彙。
    pub fn from_vocabulary_with_idf<S: AsRef<str>>(
        config: TextVectorizationConfig,
        vocabulary: &[S],
        idf_weights: &[f32],
    ) -> Result<Self, TextError> {
        check_pad(&config)?;
        check_max_tokens(config.max_tokens)?;
        if let Some(m) = config.max_tokens {
            let total = vocabulary.len().saturating_add(RESERVED_COUNT);
            if total > m {
                return Err(TextError::VocabularyTooLarge { len: total, max: m });
            }
        }
        check_config_numbers(&config)?;
        if idf_weights.len() != vocabulary.len() {
            return Err(TextError::IdfWeightsLengthMismatch {
                len: idf_weights.len(),
                expected: vocabulary.len(),
            });
        }
        if let Some(index) = idf_weights
            .iter()
            .position(|w| !w.is_finite() || w.is_sign_negative() && *w != 0.0)
        {
            return Err(TextError::InvalidIdfWeight { index });
        }
        let vocabulary = Vocabulary::from_tokens(&config.limits, vocabulary)?;
        let idf = Some(assemble_idf(idf_weights));
        Ok(Self {
            config,
            vocabulary,
            idf,
        })
    }

    /// コーパスから語彙を構築して作る。
    ///
    /// コーパスは設定の標準化・分割（・n-gram）を通して数える。予約トークン
    /// （`""`・`"[UNK]"`）は数えない。語彙は頻度降順で、同頻度はバイト列の辞書順。
    /// `max_tokens` が `Some(m)` なら上位 `m - 2` 件で切り詰める。語彙 1 件の
    /// バイト数は検査しないため、結果を既定の `TextLimits` の `from_vocabulary` へ
    /// 渡すと読み戻せない場合がある。
    pub fn adapt<S: AsRef<str>>(
        config: TextVectorizationConfig,
        corpus: &[S],
    ) -> Result<Self, TextError> {
        check_pad(&config)?;
        check_max_tokens(config.max_tokens)?;
        check_config_numbers(&config)?;
        let vocabulary = adapt(
            corpus,
            config.standardize,
            config.split,
            config.ngrams,
            config.max_tokens,
            &config.limits,
        )?;
        Ok(Self {
            config,
            vocabulary,
            idf: None,
        })
    }

    /// コーパスから語彙と idf 重みを構築して作る（`transform_tf_idf` 用）。
    ///
    /// 語彙は `adapt` と同一。idf は語 t について `ln(1 + N / (1 + df))` を f64 で計算して
    /// 1 回だけ f32 へ丸める（`N` = コーパスの文書数、`df` = t を含む文書数で、文書内の
    /// 重複は 1 と数える。自然対数）。`idf[0]` = 0.0、`idf[1]`（OOV）は丸めた後の語彙の idf の
    /// 平均（f64 で合計し 1 回だけ f32 へ。語彙が空なら 0.0）で、`idf_weights()` の `[2..]`
    /// を `from_vocabulary_with_idf` へ渡すと同じ重みへ往復できる。
    pub fn adapt_tf_idf<S: AsRef<str>>(
        config: TextVectorizationConfig,
        corpus: &[S],
    ) -> Result<Self, TextError> {
        check_pad(&config)?;
        check_max_tokens(config.max_tokens)?;
        check_config_numbers(&config)?;
        let (vocabulary, df) = adapt_with_df(
            corpus,
            config.standardize,
            config.split,
            config.ngrams,
            config.max_tokens,
            &config.limits,
        )?;
        // 語彙数の検査は adapt 内で済んでいる。ここで初めて idf を確保する。
        let n = corpus.len() as f64;
        let weights: Vec<f32> = df
            .iter()
            .map(|&d| (1.0f64 + n / (1.0f64 + d as f64)).ln() as f32)
            .collect();
        let idf = Some(assemble_idf(&weights));
        Ok(Self {
            config,
            vocabulary,
            idf,
        })
    }

    /// 文字列バッチを `[B, L]` の `Tensor<i32>` へ変換する。
    ///
    /// `L` は `output_sequence_length` が `Some` ならその値、`None` ならバッチ内の
    /// 最長トークン数。短い行は 0 で埋め、長い行は先頭 `L` 個を残す。語彙にない
    /// トークンは 1（OOV）。`B = 0` は `[0, L]`（`None` なら `L = 0`）、`Some(0)` は
    /// `[B, 0]`。`Split::None` と空入力 `""` は id 0（パディングと区別できない）。
    pub fn transform<S: AsRef<str>>(&self, inputs: &[S]) -> Result<Tensor<i32>, TextError> {
        let opts = TransformOptions {
            standardize: self.config.standardize,
            split: self.config.split,
            ngrams: self.config.ngrams,
            output_sequence_length: self.config.output_sequence_length,
        };
        transform(&self.vocabulary, &opts, &self.config.limits, inputs)
    }

    /// 出力モード系メソッドの出力列数 `W`（`pad_to_max_tokens` なら `max_tokens`、
    /// そうでなければ語彙数）。構築時検査で `pad` なら `max_tokens` は `Some`。
    fn bag_width(&self) -> usize {
        match (self.config.pad_to_max_tokens, self.config.max_tokens) {
            (true, Some(m)) => m,
            _ => self.vocabulary.vocabulary_size(),
        }
    }

    fn bag_options(&self) -> BagOptions {
        BagOptions {
            standardize: self.config.standardize,
            split: self.config.split,
            ngrams: self.config.ngrams,
            width: self.bag_width(),
        }
    }

    /// 文字列バッチを `[B, W]` の `Tensor<f32>`（出現すれば 1.0・しなければ 0.0）へ変換する。
    ///
    /// 列 i が id i に対応する（列 0 = パディングは常に 0、列 1 = OOV）。`W` は
    /// `pad_to_max_tokens` が false なら `vocabulary_size()`、true なら `max_tokens`
    /// （`vocabulary_size()` 以上の列は常に 0）。`output_sequence_length` は無視し、
    /// `ngrams` は適用する。id 0 になるトークン（`Split::None` と `""`）は数えない。
    /// `B = 0` は `[0, W]`。dtype を f32 にするのは `Linear`／`Sequential` へ直接渡すため。
    /// 入力の件数・バイト数・出力要素数（`B × W`）は確保より先に検査し、1 行のトークン数が
    /// 2^24 を超えると `TokenCountNotExact`。Keras と違い `W` は予約 2 件ぶん大きい。
    pub fn transform_multi_hot<S: AsRef<str>>(
        &self,
        inputs: &[S],
    ) -> Result<Tensor<f32>, TextError> {
        bag(
            &self.vocabulary,
            &self.bag_options(),
            &self.config.limits,
            inputs,
            BagKind::Presence,
        )
    }

    /// 文字列バッチを `[B, W]` の `Tensor<f32>`（各 id の出現回数）へ変換する。
    ///
    /// 形状・列の意味・検査・`output_sequence_length` の無視は `transform_multi_hot` と同じ。
    /// 回数は 2^24 以下に限って厳密に数える（超える行は `TokenCountNotExact`）。
    pub fn transform_count<S: AsRef<str>>(&self, inputs: &[S]) -> Result<Tensor<f32>, TextError> {
        bag(
            &self.vocabulary,
            &self.bag_options(),
            &self.config.limits,
            inputs,
            BagKind::Count,
        )
    }

    /// 文字列バッチを `[B, W]` の `Tensor<f32>`（出現回数 × idf 重み）へ変換する。
    ///
    /// 値は `count as f32 * idf[id]`（f32 の積 1 回）。idf を持たないインスタンス
    /// （`from_vocabulary`／`adapt` で作ったもの）は、入力を見る前に `MissingIdfWeights`。
    /// 形状・列の意味・検査・`output_sequence_length` の無視は `transform_multi_hot` と同じ。
    /// 出現しない列は計算せず 0 のまま（pad 時の範囲外の列でも idf を引かない）。
    pub fn transform_tf_idf<S: AsRef<str>>(&self, inputs: &[S]) -> Result<Tensor<f32>, TextError> {
        let Some(idf) = self.idf.as_deref() else {
            return Err(TextError::MissingIdfWeights);
        };
        bag(
            &self.vocabulary,
            &self.bag_options(),
            &self.config.limits,
            inputs,
            BagKind::Weighted(idf),
        )
    }

    /// id 順の idf 重み（長さ = `vocabulary_size()`。index 0 = 0.0、1 = OOV、2.. = 語彙）。
    /// `from_vocabulary`／`adapt` で作ったインスタンスは `None`。
    pub fn idf_weights(&self) -> Option<&[f32]> {
        self.idf.as_deref()
    }

    /// id 順の語彙。index 0 = `""`、1 = `"[UNK]"`、2.. = 語彙。
    pub fn vocabulary(&self) -> &[String] {
        self.vocabulary.tokens()
    }

    /// 語彙数（予約 2 件を含む）。
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.vocabulary_size()
    }

    /// 構築時の設定。
    pub fn config(&self) -> &TextVectorizationConfig {
        &self.config
    }
}

/// 設定と語彙件数だけを出す。語彙の全文は出さない（A09）。
impl fmt::Debug for TextVectorization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TextVectorization")
            .field("config", &self.config)
            .field("vocabulary_size", &self.vocabulary.vocabulary_size())
            // idf は件数だけ。値は出さない（A09）。
            .field("idf_weights_len", &self.idf.as_ref().map(Vec::len))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_recorded_defaults() {
        let c = TextVectorizationConfig::default();
        assert_eq!(c.max_tokens, None);
        assert_eq!(c.standardize, Standardize::LowerAndStripPunctuation);
        assert_eq!(c.split, Split::Whitespace);
        assert_eq!(c.ngrams, None);
        assert_eq!(c.output_sequence_length, None);
        assert_eq!(c.limits, TextLimits::default());
    }

    #[test]
    fn builders_set_some_and_fields() {
        let c = TextVectorizationConfig::default()
            .with_max_tokens(10)
            .with_standardize(Standardize::None)
            .with_split(Split::Character)
            .with_ngrams(2)
            .with_output_sequence_length(5);
        assert_eq!(c.max_tokens, Some(10));
        assert_eq!(c.standardize, Standardize::None);
        assert_eq!(c.split, Split::Character);
        assert_eq!(c.ngrams, Some(2));
        assert_eq!(c.output_sequence_length, Some(5));
    }

    #[test]
    fn from_vocabulary_max_tokens_is_fail_closed() {
        let cfg = |m| TextVectorizationConfig::default().with_max_tokens(m);
        let v = ["a", "b", "c"];
        assert!(matches!(
            TextVectorization::from_vocabulary(cfg(2), &v),
            Err(TextError::InvalidMaxTokens { max_tokens: 2 })
        ));
        assert!(matches!(
            TextVectorization::from_vocabulary(cfg(4), &v),
            Err(TextError::VocabularyTooLarge { len: 5, max: 4 })
        ));
        assert_eq!(
            TextVectorization::from_vocabulary(cfg(5), &v)
                .unwrap()
                .vocabulary_size(),
            5
        );
    }

    #[test]
    fn construction_checks_numbers_before_vocabulary() {
        let empty: [&str; 0] = [];
        let bad_n = TextVectorizationConfig::default().with_ngrams(0);
        assert!(matches!(
            TextVectorization::from_vocabulary(bad_n.clone(), &empty),
            Err(TextError::InvalidNgrams { .. })
        ));
        assert!(matches!(
            TextVectorization::adapt(bad_n, &empty),
            Err(TextError::InvalidNgrams { .. })
        ));
        let too_long = TextVectorizationConfig::default().with_output_sequence_length(usize::MAX);
        assert!(matches!(
            TextVectorization::from_vocabulary(too_long.clone(), &empty),
            Err(TextError::OutputSequenceLengthTooLarge { .. })
        ));
        assert!(matches!(
            TextVectorization::adapt(too_long, &empty),
            Err(TextError::OutputSequenceLengthTooLarge { .. })
        ));
    }

    #[test]
    fn debug_hides_vocabulary_contents() {
        let tv = TextVectorization::from_vocabulary(
            TextVectorizationConfig::default(),
            &["SECRET_MARKER_TOKEN"],
        )
        .unwrap();
        let s = format!("{tv:?}");
        assert!(!s.contains("SECRET_MARKER_TOKEN"));
        assert!(s.contains("vocabulary_size"));
    }

    fn idf_of(n: usize, df: usize) -> f32 {
        (1.0f64 + n as f64 / (1.0 + df as f64)).ln() as f32
    }

    fn oov_mean(ws: &[f32]) -> f32 {
        if ws.is_empty() {
            return 0.0;
        }
        (ws.iter().map(|&w| f64::from(w)).sum::<f64>() / ws.len() as f64) as f32
    }

    #[test]
    fn pad_field_defaults_false_and_builder_sets_it() {
        assert!(!TextVectorizationConfig::default().pad_to_max_tokens);
        let c = TextVectorizationConfig::default().with_pad_to_max_tokens(true);
        assert!(c.pad_to_max_tokens);
    }

    #[test]
    fn pad_without_max_tokens_is_rejected_by_all_constructors() {
        let c = TextVectorizationConfig::default().with_pad_to_max_tokens(true);
        let v = ["a"];
        assert!(matches!(
            TextVectorization::from_vocabulary(c.clone(), &v),
            Err(TextError::PadToMaxTokensWithoutMaxTokens)
        ));
        assert!(matches!(
            TextVectorization::adapt(c.clone(), &v),
            Err(TextError::PadToMaxTokensWithoutMaxTokens)
        ));
        assert!(matches!(
            TextVectorization::from_vocabulary_with_idf(c.clone(), &v, &[1.0]),
            Err(TextError::PadToMaxTokensWithoutMaxTokens)
        ));
        assert!(matches!(
            TextVectorization::adapt_tf_idf(c, &v),
            Err(TextError::PadToMaxTokensWithoutMaxTokens)
        ));
    }

    #[test]
    fn width_follows_vocabulary_or_max_tokens_when_padded() {
        let cfg = TextVectorizationConfig::default().with_max_tokens(8);
        let tv = TextVectorization::from_vocabulary(cfg.clone(), &["a", "b"]).unwrap();
        assert_eq!(tv.transform_count(&["a"]).unwrap().shape(), &[1, 4]);
        let padded = cfg.with_pad_to_max_tokens(true);
        let tv = TextVectorization::from_vocabulary(padded, &["a", "b"]).unwrap();
        let out = tv.transform_count(&["a b b"]).unwrap();
        assert_eq!(out.shape(), &[1, 8]);
        assert_eq!(
            out.host_slice().to_vec(),
            [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            tv.transform_multi_hot::<&str>(&[]).unwrap().shape(),
            &[0, 8]
        );
    }

    #[test]
    fn output_sequence_length_is_ignored_by_bag_methods() {
        let a = TextVectorizationConfig::default();
        let b = a.clone().with_output_sequence_length(1);
        let v = ["x", "y"];
        let ta = TextVectorization::from_vocabulary(a, &v).unwrap();
        let tb = TextVectorization::from_vocabulary(b, &v).unwrap();
        let ins = ["x y y z"];
        assert_eq!(
            ta.transform_count(&ins).unwrap().host_slice(),
            tb.transform_count(&ins).unwrap().host_slice()
        );
    }

    #[test]
    fn tf_idf_without_weights_is_missing_even_for_bad_input() {
        let tv =
            TextVectorization::from_vocabulary(TextVectorizationConfig::default(), &["a"]).unwrap();
        assert!(tv.idf_weights().is_none());
        assert!(matches!(
            tv.transform_tf_idf(&["a"]),
            Err(TextError::MissingIdfWeights)
        ));
        let tv = TextVectorization::adapt(TextVectorizationConfig::default(), &["a"]).unwrap();
        assert!(matches!(
            tv.transform_tf_idf::<&str>(&[]),
            Err(TextError::MissingIdfWeights)
        ));
    }

    #[test]
    fn idf_weights_are_validated_in_documented_order() {
        let cfg = TextVectorizationConfig::default;
        assert!(matches!(
            TextVectorization::from_vocabulary_with_idf(cfg(), &["a", "b"], &[1.0]),
            Err(TextError::IdfWeightsLengthMismatch {
                len: 1,
                expected: 2
            })
        ));
        for (i, bad) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0]
            .into_iter()
            .enumerate()
        {
            let r = TextVectorization::from_vocabulary_with_idf(cfg(), &["a", "b"], &[0.5, bad]);
            assert!(
                matches!(r, Err(TextError::InvalidIdfWeight { index: 1 })),
                "{i}"
            );
        }
        assert!(
            TextVectorization::from_vocabulary_with_idf(cfg(), &["a", "b"], &[-0.0, 0.0]).is_ok()
        );
        // 長さ不一致は値の検査より先、語彙の検査（重複）より先。
        assert!(matches!(
            TextVectorization::from_vocabulary_with_idf(cfg(), &["a", "a"], &[f32::NAN]),
            Err(TextError::IdfWeightsLengthMismatch { .. })
        ));
        // 値の検査は語彙の検査（重複）より先。
        assert!(matches!(
            TextVectorization::from_vocabulary_with_idf(cfg(), &["a", "a"], &[f32::NAN, 1.0]),
            Err(TextError::InvalidIdfWeight { index: 0 })
        ));
        // 設定側の検査（max_tokens・pad）は idf の検査より先。
        let small = cfg().with_max_tokens(3);
        assert!(matches!(
            TextVectorization::from_vocabulary_with_idf(small, &["a", "b"], &[f32::NAN]),
            Err(TextError::VocabularyTooLarge { .. })
        ));
        let pad = cfg().with_pad_to_max_tokens(true);
        assert!(matches!(
            TextVectorization::from_vocabulary_with_idf(pad, &["a"], &[f32::NAN, 1.0]),
            Err(TextError::PadToMaxTokensWithoutMaxTokens)
        ));
    }

    #[test]
    fn explicit_idf_layout_and_oov_mean() {
        let tv = TextVectorization::from_vocabulary_with_idf(
            TextVectorizationConfig::default(),
            &["a", "b"],
            &[1.0, 2.0],
        )
        .unwrap();
        assert_eq!(tv.idf_weights().unwrap(), [0.0, 1.5, 1.0, 2.0]);
        let empty: [&str; 0] = [];
        let tv = TextVectorization::from_vocabulary_with_idf(
            TextVectorizationConfig::default(),
            &empty,
            &[],
        )
        .unwrap();
        assert_eq!(tv.idf_weights().unwrap(), [0.0, 0.0]);
    }

    #[test]
    fn adapt_tf_idf_matches_formula_bit_for_bit_and_round_trips() {
        let corpus = ["a a b", "a c", "b b b d"];
        let cfg = TextVectorizationConfig::default();
        let tv = TextVectorization::adapt_tf_idf(cfg.clone(), &corpus).unwrap();
        // freq: b=4, a=3, c=1, d=1 -> 並び b, a, c, d。df: b=2, a=2, c=1, d=1。
        assert_eq!(tv.vocabulary(), ["", "[UNK]", "b", "a", "c", "d"]);
        let dfs = [2usize, 2, 1, 1];
        let ws: Vec<f32> = dfs.iter().map(|&d| idf_of(3, d)).collect();
        let idf = tv.idf_weights().unwrap();
        assert_eq!(idf.len(), tv.vocabulary_size());
        assert_eq!(idf[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(idf[1].to_bits(), oov_mean(&ws).to_bits());
        for (g, w) in idf[2..].iter().zip(&ws) {
            assert_eq!(g.to_bits(), w.to_bits());
        }
        let out = tv.transform_tf_idf(&["b b a zzz"]).unwrap();
        let o = out.host_slice();
        assert_eq!(o[0], 0.0);
        assert_eq!(o[1].to_bits(), (1.0f32 * oov_mean(&ws)).to_bits());
        assert_eq!(o[2].to_bits(), (2.0f32 * ws[0]).to_bits());
        assert_eq!(o[3].to_bits(), (1.0f32 * ws[1]).to_bits());
        assert_eq!(o[4], 0.0);
        // 往復: 語彙と idf[2..] から作り直すと idf・出力が bit 一致する。
        let words: Vec<String> = tv.vocabulary()[2..].to_vec();
        let back = TextVectorization::from_vocabulary_with_idf(cfg, &words, &idf[2..]).unwrap();
        let bi = back.idf_weights().unwrap();
        assert!(idf.iter().zip(bi).all(|(a, b)| a.to_bits() == b.to_bits()));
        let again = back.transform_tf_idf(&["b b a zzz"]).unwrap();
        assert!(
            o.iter()
                .zip(again.host_slice().iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn tf_idf_never_reads_idf_for_pad_columns() {
        let cfg = TextVectorizationConfig::default()
            .with_max_tokens(6)
            .with_pad_to_max_tokens(true);
        let tv = TextVectorization::from_vocabulary_with_idf(cfg, &["a"], &[2.0]).unwrap();
        let out = tv.transform_tf_idf(&["a a q"]).unwrap();
        assert_eq!(out.shape(), &[1, 6]);
        assert_eq!(out.host_slice().to_vec(), [0.0, 2.0, 4.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn debug_shows_only_idf_count() {
        let tv = TextVectorization::from_vocabulary_with_idf(
            TextVectorizationConfig::default(),
            &["SECRET_MARKER_TOKEN"],
            &[0.123_456_79],
        )
        .unwrap();
        let s = format!("{tv:?}");
        assert!(!s.contains("SECRET_MARKER_TOKEN"));
        assert!(!s.contains("0.1234"));
        assert!(s.contains("idf_weights_len: Some(3)"));
    }
}
