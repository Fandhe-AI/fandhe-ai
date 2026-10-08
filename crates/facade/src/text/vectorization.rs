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

use std::fmt;

use super::adapt::adapt;
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
///   バッチ内の最長に合わせる
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TextVectorizationConfig {
    /// 予約 2 件を含む語彙数の上限。`None` は無制限（`limits` の上限は適用される）。
    pub max_tokens: Option<usize>,
    /// 標準化の方式（ASCII 限定）。
    pub standardize: Standardize,
    /// 分割の方式（ASCII 空白限定）。
    pub split: Split,
    /// `Some(n)` なら 1〜n-gram を作る。
    pub ngrams: Option<usize>,
    /// `Some(L)` なら出力長を `L` に固定する。
    pub output_sequence_length: Option<usize>,
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
        check_max_tokens(config.max_tokens)?;
        if let Some(m) = config.max_tokens {
            let total = vocabulary.len().saturating_add(RESERVED_COUNT);
            if total > m {
                return Err(TextError::VocabularyTooLarge { len: total, max: m });
            }
        }
        check_config_numbers(&config)?;
        let vocabulary = Vocabulary::from_tokens(&config.limits, vocabulary)?;
        Ok(Self { config, vocabulary })
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
        Ok(Self { config, vocabulary })
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
}
