//! 非信頼入力の上限 `TextLimits` と上限検査関数（`text` モジュール）。
//!
//! 役割: 語彙 lookup 型のテキスト変換が受け取る非信頼入力（文字列・コーパス・
//! 語彙・設定値）を、**標準化・分割・メモリ確保より先に**長さだけで検査する。
//! 呼び出し元は後続イシューの標準化・語彙構築・adapt・transform（いずれも入口で
//! 本モジュールの `check_*` を呼ぶ）。設計の正は
//! `docs/facade-text-vectorization-design.md` §8（既定値・絶対上限・検査時機）。
//! 本上限は API の入力上限でありガードレール閾値ではない。
//!
//! 設計上の要点:
//! - フィールドは非公開。`with_*` が絶対上限を検査するため、絶対上限を超える
//!   値は構築できない（`TextError::LimitAboveAbsoluteMaximum`）。
//! - 検査の本体は「長さ（`usize`）」を受け取る。巨大な実文字列なしで境界値 ±1 を
//!   検査でき、`&[S]` から長さを取り出す入口（`check_inputs`・`check_corpus`）は
//!   薄い。比較は常に「`> max` で `Err`、`== max` は `Ok`」。
//! - 和・積は `checked_*` を使い、overflow も `Err` に倒す（panic しない）。
//! - 「1 入力から作る n-gram 数」の `checked_mul` 検査は n-gram 生成段
//!   （設計記録 §11 の 4）が担う。
//!
//! 公開するのは `TextLimits` 本体と `with_*` 9 本だけ（設計記録 §16.2）。アクセサ・
//! `check_*`・`DEFAULT_*`／`ABSOLUTE_*` 定数は非公開のまま。

use super::error::TextError;

/// バッチ要素数・コーパス件数の既定上限（設計記録 §8）。
pub(crate) const DEFAULT_MAX_BATCH: usize = 65_536;
/// 1 文字列のバイト数の既定上限（1 MiB）。
pub(crate) const DEFAULT_MAX_INPUT_BYTES: usize = 1 << 20;
/// `adapt` のコーパス総バイト数の既定上限（256 MiB）。
pub(crate) const DEFAULT_MAX_CORPUS_BYTES: usize = 256 << 20;
/// 語彙数（予約 2 件を含む）の既定上限。
pub(crate) const DEFAULT_MAX_VOCABULARY_SIZE: usize = 1 << 24;
/// 語彙 1 件のバイト数の既定上限（4 KiB）。
pub(crate) const DEFAULT_MAX_VOCABULARY_TOKEN_BYTES: usize = 4096;
/// `adapt` 中の異なり語数の既定上限。
pub(crate) const DEFAULT_MAX_DISTINCT_TOKENS: usize = 1 << 24;
/// n-gram の n の既定上限（絶対上限と同値。下げる方向だけ設定可）。
pub(crate) const DEFAULT_MAX_NGRAMS: usize = 8;
/// `output_sequence_length` の既定上限。
pub(crate) const DEFAULT_MAX_OUTPUT_SEQUENCE_LENGTH: usize = 1 << 20;
/// 出力要素数 `B × L` の既定上限。
pub(crate) const DEFAULT_MAX_OUTPUT_ELEMENTS: usize = 1 << 28;

/// バイト数系の絶対上限（`isize::MAX`）。
pub(crate) const ABSOLUTE_MAX_BYTES: usize = isize::MAX as usize;
/// 語彙数の絶対上限（`i32::MAX`。出力 id が `i32` のため）。
pub(crate) const ABSOLUTE_MAX_VOCABULARY_SIZE: usize = i32::MAX as usize;
/// n-gram の n の絶対上限。
pub(crate) const ABSOLUTE_MAX_NGRAMS: usize = 8;
/// 出力要素数の絶対上限（`isize::MAX` バイトを `i32` 要素数に換算）。
pub(crate) const ABSOLUTE_MAX_OUTPUT_ELEMENTS: usize =
    isize::MAX as usize / std::mem::size_of::<i32>();

/// 非信頼入力（文字列バッチ・コーパス・語彙・設定値）の上限集合。
///
/// `TextVectorizationConfig::limits` に載せる。検査は標準化・分割・メモリ確保より
/// 先に長さだけで行い、超過は [`TextError`] で返す。`Default` の既定値と
/// 各 `with_*` の絶対上限は次のとおり（`with_*` は絶対上限を超えると
/// [`TextError::LimitAboveAbsoluteMaximum`] を返す）。
///
/// | `with_*` | 既定値 | 絶対上限 |
/// |---|---|---|
/// | `with_max_batch`（バッチ要素数・コーパス件数） | 65,536 | なし |
/// | `with_max_input_bytes`（1 文字列のバイト数） | 1 MiB | `isize::MAX` |
/// | `with_max_corpus_bytes`（`adapt` のコーパス総バイト数） | 256 MiB | `isize::MAX` |
/// | `with_max_vocabulary_size`（予約 2 件を含む語彙数） | 2^24 | `i32::MAX` |
/// | `with_max_vocabulary_token_bytes`（語彙 1 件のバイト数） | 4 KiB | `isize::MAX` |
/// | `with_max_distinct_tokens`（`adapt` 中の異なり語数） | 2^24 | なし |
/// | `with_max_ngrams`（n-gram の n） | 8 | 8（下げる方向のみ） |
/// | `with_max_output_sequence_length` | 2^20 | なし |
/// | `with_max_output_elements`（出力 `B × L`） | 2^28 | `isize::MAX / 4` |
///
/// 値を読み戻すアクセサは公開しない。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextLimits {
    max_batch: usize,
    max_input_bytes: usize,
    max_corpus_bytes: usize,
    max_vocabulary_size: usize,
    max_vocabulary_token_bytes: usize,
    max_distinct_tokens: usize,
    max_ngrams: usize,
    max_output_sequence_length: usize,
    max_output_elements: usize,
}

impl Default for TextLimits {
    fn default() -> Self {
        Self {
            max_batch: DEFAULT_MAX_BATCH,
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            max_corpus_bytes: DEFAULT_MAX_CORPUS_BYTES,
            max_vocabulary_size: DEFAULT_MAX_VOCABULARY_SIZE,
            max_vocabulary_token_bytes: DEFAULT_MAX_VOCABULARY_TOKEN_BYTES,
            max_distinct_tokens: DEFAULT_MAX_DISTINCT_TOKENS,
            max_ngrams: DEFAULT_MAX_NGRAMS,
            max_output_sequence_length: DEFAULT_MAX_OUTPUT_SEQUENCE_LENGTH,
            max_output_elements: DEFAULT_MAX_OUTPUT_ELEMENTS,
        }
    }
}

/// 設定値が絶対上限以下なら `Ok(value)`、超えたら `LimitAboveAbsoluteMaximum`。
fn within_absolute(limit: &'static str, value: usize, max: usize) -> Result<usize, TextError> {
    if value > max {
        Err(TextError::LimitAboveAbsoluteMaximum { limit, value, max })
    } else {
        Ok(value)
    }
}

impl TextLimits {
    /// バッチ要素数・コーパス件数の上限（絶対上限なし）。
    pub fn with_max_batch(mut self, value: usize) -> Result<Self, TextError> {
        self.max_batch = within_absolute("max_batch", value, usize::MAX)?;
        Ok(self)
    }
    /// 1 文字列のバイト数の上限（絶対上限 `isize::MAX`）。
    pub fn with_max_input_bytes(mut self, value: usize) -> Result<Self, TextError> {
        self.max_input_bytes = within_absolute("max_input_bytes", value, ABSOLUTE_MAX_BYTES)?;
        Ok(self)
    }
    /// コーパス総バイト数の上限（絶対上限 `isize::MAX`）。
    pub fn with_max_corpus_bytes(mut self, value: usize) -> Result<Self, TextError> {
        self.max_corpus_bytes = within_absolute("max_corpus_bytes", value, ABSOLUTE_MAX_BYTES)?;
        Ok(self)
    }
    /// 語彙数（予約 2 件を含む）の上限（絶対上限 `i32::MAX`）。
    pub fn with_max_vocabulary_size(mut self, value: usize) -> Result<Self, TextError> {
        self.max_vocabulary_size =
            within_absolute("max_vocabulary_size", value, ABSOLUTE_MAX_VOCABULARY_SIZE)?;
        Ok(self)
    }
    /// 語彙 1 件のバイト数の上限（絶対上限 `isize::MAX`）。
    pub fn with_max_vocabulary_token_bytes(mut self, value: usize) -> Result<Self, TextError> {
        self.max_vocabulary_token_bytes =
            within_absolute("max_vocabulary_token_bytes", value, ABSOLUTE_MAX_BYTES)?;
        Ok(self)
    }
    /// `adapt` 中の異なり語数の上限（絶対上限なし）。
    pub fn with_max_distinct_tokens(mut self, value: usize) -> Result<Self, TextError> {
        self.max_distinct_tokens = within_absolute("max_distinct_tokens", value, usize::MAX)?;
        Ok(self)
    }
    /// n-gram の n の上限（絶対上限 8。下げる方向のみ）。
    pub fn with_max_ngrams(mut self, value: usize) -> Result<Self, TextError> {
        self.max_ngrams = within_absolute("max_ngrams", value, ABSOLUTE_MAX_NGRAMS)?;
        Ok(self)
    }
    /// `output_sequence_length` の上限（絶対上限なし）。
    pub fn with_max_output_sequence_length(mut self, value: usize) -> Result<Self, TextError> {
        self.max_output_sequence_length =
            within_absolute("max_output_sequence_length", value, usize::MAX)?;
        Ok(self)
    }
    /// 出力要素数 `B × L` の上限（絶対上限 `isize::MAX / size_of::<i32>()`）。
    pub fn with_max_output_elements(mut self, value: usize) -> Result<Self, TextError> {
        self.max_output_elements =
            within_absolute("max_output_elements", value, ABSOLUTE_MAX_OUTPUT_ELEMENTS)?;
        Ok(self)
    }

    // アクセサはテスト専用（公開しない。設計記録 §16.2・§17）。本番経路は `check_*` を使う。
    #[cfg(test)]
    pub(crate) fn max_batch(&self) -> usize {
        self.max_batch
    }
    #[cfg(test)]
    pub(crate) fn max_input_bytes(&self) -> usize {
        self.max_input_bytes
    }
    #[cfg(test)]
    pub(crate) fn max_corpus_bytes(&self) -> usize {
        self.max_corpus_bytes
    }
    #[cfg(test)]
    pub(crate) fn max_vocabulary_size(&self) -> usize {
        self.max_vocabulary_size
    }
    #[cfg(test)]
    pub(crate) fn max_vocabulary_token_bytes(&self) -> usize {
        self.max_vocabulary_token_bytes
    }
    pub(crate) fn max_distinct_tokens(&self) -> usize {
        self.max_distinct_tokens
    }
    #[cfg(test)]
    pub(crate) fn max_ngrams(&self) -> usize {
        self.max_ngrams
    }
    #[cfg(test)]
    pub(crate) fn max_output_sequence_length(&self) -> usize {
        self.max_output_sequence_length
    }
    #[cfg(test)]
    pub(crate) fn max_output_elements(&self) -> usize {
        self.max_output_elements
    }

    /// バッチ要素数（`transform`）・コーパス件数（`adapt`）の検査。
    pub(crate) fn check_batch_len(&self, len: usize) -> Result<(), TextError> {
        if len > self.max_batch {
            return Err(TextError::BatchTooLarge {
                len,
                max: self.max_batch,
            });
        }
        Ok(())
    }

    /// `index` 番目の入力文字列のバイト数 `len` の検査。
    pub(crate) fn check_input_len(&self, index: usize, len: usize) -> Result<(), TextError> {
        if len > self.max_input_bytes {
            return Err(TextError::InputTooLong {
                index,
                len,
                max: self.max_input_bytes,
            });
        }
        Ok(())
    }

    /// `transform` 入口: 件数 → 各要素のバイト数の順に検査する。
    pub(crate) fn check_inputs<S: AsRef<str>>(&self, inputs: &[S]) -> Result<(), TextError> {
        self.check_batch_len(inputs.len())?;
        for (index, s) in inputs.iter().enumerate() {
            self.check_input_len(index, s.as_ref().len())?;
        }
        Ok(())
    }

    /// コーパス総バイト数の検査。`checked_add` で合計し、途中で上限を超えた
    /// 時点で即 `Err`（`total_bytes` はその時点の合計。加算 overflow 時は
    /// `usize::MAX` に飽和させる）。
    pub(crate) fn check_corpus_total_bytes(
        &self,
        lens: impl IntoIterator<Item = usize>,
    ) -> Result<(), TextError> {
        let mut total: usize = 0;
        for len in lens {
            total = match total.checked_add(len) {
                Some(t) => t,
                None => {
                    return Err(TextError::CorpusTooLarge {
                        total_bytes: usize::MAX,
                        max: self.max_corpus_bytes,
                    });
                }
            };
            if total > self.max_corpus_bytes {
                return Err(TextError::CorpusTooLarge {
                    total_bytes: total,
                    max: self.max_corpus_bytes,
                });
            }
        }
        Ok(())
    }

    /// `adapt` 入口: 件数（`max_batch` を共用）→ 各要素 → 総バイト数の順。
    pub(crate) fn check_corpus<S: AsRef<str>>(&self, corpus: &[S]) -> Result<(), TextError> {
        self.check_inputs(corpus)?;
        self.check_corpus_total_bytes(corpus.iter().map(|s| s.as_ref().len()))
    }

    /// 語彙数の検査。`len` は予約 2 件を含む件数（呼び出し側が `checked_add`
    /// で足してから渡す）。
    pub(crate) fn check_vocabulary_size(&self, len: usize) -> Result<(), TextError> {
        if len > self.max_vocabulary_size {
            return Err(TextError::VocabularyTooLarge {
                len,
                max: self.max_vocabulary_size,
            });
        }
        Ok(())
    }

    /// `index` 番目の語彙のバイト数 `len` の検査。
    pub(crate) fn check_vocabulary_token_len(
        &self,
        index: usize,
        len: usize,
    ) -> Result<(), TextError> {
        if len > self.max_vocabulary_token_bytes {
            return Err(TextError::VocabularyTokenTooLong {
                index,
                len,
                max: self.max_vocabulary_token_bytes,
            });
        }
        Ok(())
    }

    /// 新しい語を入れた後の異なり語数 `count` の検査。集計用 `HashMap` へ
    /// 入れる**前**に呼ぶ。
    pub(crate) fn check_distinct_tokens(&self, count: usize) -> Result<(), TextError> {
        if count > self.max_distinct_tokens {
            return Err(TextError::TooManyDistinctTokens {
                max: self.max_distinct_tokens,
            });
        }
        Ok(())
    }

    /// n-gram の n の検査（`n == 0` も `Err`）。
    pub(crate) fn check_ngrams(&self, n: usize) -> Result<(), TextError> {
        if n == 0 || n > self.max_ngrams {
            return Err(TextError::InvalidNgrams {
                n,
                max: self.max_ngrams,
            });
        }
        Ok(())
    }

    /// `output_sequence_length` の検査。
    pub(crate) fn check_output_sequence_length(&self, len: usize) -> Result<(), TextError> {
        if len > self.max_output_sequence_length {
            return Err(TextError::OutputSequenceLengthTooLarge {
                len,
                max: self.max_output_sequence_length,
            });
        }
        Ok(())
    }

    /// 出力要素数 `batch × len` の検査。検査後の要素数を返す。
    /// 積の overflow、要素数の上限超過、`i32` バイト数換算が `isize::MAX` を
    /// 超える場合は `Err`（`crates/autodiff/src/generate.rs` と同型の多重防御）。
    /// overflow 時の `elements` は `usize::MAX` に飽和させる。
    pub(crate) fn check_output_elements(
        &self,
        batch: usize,
        len: usize,
    ) -> Result<usize, TextError> {
        let too_large = |elements: usize| TextError::OutputTooLarge {
            elements,
            max: self.max_output_elements,
        };
        let elements = batch
            .checked_mul(len)
            .ok_or_else(|| too_large(usize::MAX))?;
        if elements > self.max_output_elements {
            return Err(too_large(elements));
        }
        match elements.checked_mul(std::mem::size_of::<i32>()) {
            Some(bytes) if bytes <= ABSOLUTE_MAX_BYTES => Ok(elements),
            _ => Err(too_large(elements)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const I: usize = ABSOLUTE_MAX_BYTES;

    fn small() -> TextLimits {
        TextLimits::default()
            .with_max_batch(3)
            .and_then(|l| l.with_max_input_bytes(4))
            .and_then(|l| l.with_max_corpus_bytes(6))
            .and_then(|l| l.with_max_vocabulary_size(10))
            .and_then(|l| l.with_max_vocabulary_token_bytes(5))
            .and_then(|l| l.with_max_distinct_tokens(7))
            .and_then(|l| l.with_max_ngrams(2))
            .and_then(|l| l.with_max_output_sequence_length(9))
            .and_then(|l| l.with_max_output_elements(20))
            .unwrap_or_default()
    }

    #[test]
    fn defaults_match_design_record() {
        let l = TextLimits::default();
        assert_eq!(l.max_batch(), 65_536);
        assert_eq!(l.max_input_bytes(), 1 << 20);
        assert_eq!(l.max_corpus_bytes(), 256 << 20);
        assert_eq!(l.max_vocabulary_size(), 1 << 24);
        assert_eq!(l.max_vocabulary_token_bytes(), 4096);
        assert_eq!(l.max_distinct_tokens(), 1 << 24);
        assert_eq!(l.max_ngrams(), 8);
        assert_eq!(l.max_output_sequence_length(), 1 << 20);
        assert_eq!(l.max_output_elements(), 1 << 28);
    }

    #[test]
    fn defaults_do_not_exceed_absolute_maxima() {
        const { assert!(DEFAULT_MAX_INPUT_BYTES <= ABSOLUTE_MAX_BYTES) };
        const { assert!(DEFAULT_MAX_CORPUS_BYTES <= ABSOLUTE_MAX_BYTES) };
        const { assert!(DEFAULT_MAX_VOCABULARY_SIZE <= ABSOLUTE_MAX_VOCABULARY_SIZE) };
        const { assert!(DEFAULT_MAX_VOCABULARY_TOKEN_BYTES <= ABSOLUTE_MAX_BYTES) };
        const { assert!(DEFAULT_MAX_NGRAMS <= ABSOLUTE_MAX_NGRAMS) };
        const { assert!(DEFAULT_MAX_OUTPUT_ELEMENTS <= ABSOLUTE_MAX_OUTPUT_ELEMENTS) };
    }

    #[test]
    fn small_limits_builder_applies_every_value() {
        let l = small();
        assert_eq!(l.max_batch(), 3);
        assert_eq!(l.max_input_bytes(), 4);
        assert_eq!(l.max_corpus_bytes(), 6);
        assert_eq!(l.max_vocabulary_size(), 10);
        assert_eq!(l.max_vocabulary_token_bytes(), 5);
        assert_eq!(l.max_distinct_tokens(), 7);
        assert_eq!(l.max_ngrams(), 2);
        assert_eq!(l.max_output_sequence_length(), 9);
        assert_eq!(l.max_output_elements(), 20);
    }

    #[test]
    fn setters_accept_absolute_maximum_and_reject_one_above() {
        let d = TextLimits::default();
        assert!(d.with_max_input_bytes(I).is_ok());
        assert!(matches!(
            d.with_max_input_bytes(I + 1),
            Err(TextError::LimitAboveAbsoluteMaximum {
                limit: "max_input_bytes",
                value,
                max,
            }) if value == I + 1 && max == I
        ));
        assert!(d.with_max_corpus_bytes(I).is_ok());
        assert!(matches!(
            d.with_max_corpus_bytes(I + 1),
            Err(TextError::LimitAboveAbsoluteMaximum { .. })
        ));
        assert!(d.with_max_vocabulary_token_bytes(I).is_ok());
        assert!(matches!(
            d.with_max_vocabulary_token_bytes(I + 1),
            Err(TextError::LimitAboveAbsoluteMaximum { .. })
        ));
        assert!(d.with_max_vocabulary_size(i32::MAX as usize).is_ok());
        assert!(matches!(
            d.with_max_vocabulary_size(i32::MAX as usize + 1),
            Err(TextError::LimitAboveAbsoluteMaximum { .. })
        ));
        assert!(d.with_max_ngrams(8).is_ok());
        assert!(matches!(
            d.with_max_ngrams(9),
            Err(TextError::LimitAboveAbsoluteMaximum { .. })
        ));
        assert!(
            d.with_max_output_elements(ABSOLUTE_MAX_OUTPUT_ELEMENTS)
                .is_ok()
        );
        assert!(matches!(
            d.with_max_output_elements(ABSOLUTE_MAX_OUTPUT_ELEMENTS + 1),
            Err(TextError::LimitAboveAbsoluteMaximum { .. })
        ));
    }

    #[test]
    fn setters_without_absolute_maximum_accept_usize_max() {
        let d = TextLimits::default();
        assert!(d.with_max_batch(usize::MAX).is_ok());
        assert!(d.with_max_distinct_tokens(usize::MAX).is_ok());
        assert!(d.with_max_output_sequence_length(usize::MAX).is_ok());
    }

    #[test]
    fn batch_len_boundary() {
        let l = small();
        assert!(l.check_batch_len(2).is_ok());
        assert!(l.check_batch_len(3).is_ok());
        assert!(matches!(
            l.check_batch_len(4),
            Err(TextError::BatchTooLarge { len: 4, max: 3 })
        ));
    }

    #[test]
    fn input_len_boundary() {
        let l = small();
        assert!(l.check_input_len(0, 3).is_ok());
        assert!(l.check_input_len(0, 4).is_ok());
        assert!(matches!(
            l.check_input_len(7, 5),
            Err(TextError::InputTooLong {
                index: 7,
                len: 5,
                max: 4
            })
        ));
    }

    #[test]
    fn corpus_total_boundary_and_overflow() {
        let l = small();
        assert!(l.check_corpus_total_bytes([3, 2]).is_ok());
        assert!(l.check_corpus_total_bytes([3, 3]).is_ok());
        assert!(matches!(
            l.check_corpus_total_bytes([3, 4]),
            Err(TextError::CorpusTooLarge {
                total_bytes: 7,
                max: 6
            })
        ));
        assert!(l.check_corpus_total_bytes(std::iter::empty()).is_ok());
        assert!(matches!(
            l.check_corpus_total_bytes([usize::MAX, 1]),
            Err(TextError::CorpusTooLarge {
                total_bytes: usize::MAX,
                ..
            })
        ));
    }

    #[test]
    fn vocabulary_size_boundary() {
        let l = small();
        assert!(l.check_vocabulary_size(9).is_ok());
        assert!(l.check_vocabulary_size(10).is_ok());
        assert!(matches!(
            l.check_vocabulary_size(11),
            Err(TextError::VocabularyTooLarge { len: 11, max: 10 })
        ));
    }

    #[test]
    fn vocabulary_token_len_boundary() {
        let l = small();
        assert!(l.check_vocabulary_token_len(0, 4).is_ok());
        assert!(l.check_vocabulary_token_len(0, 5).is_ok());
        assert!(matches!(
            l.check_vocabulary_token_len(2, 6),
            Err(TextError::VocabularyTokenTooLong {
                index: 2,
                len: 6,
                max: 5
            })
        ));
    }

    #[test]
    fn distinct_tokens_boundary() {
        let l = small();
        assert!(l.check_distinct_tokens(6).is_ok());
        assert!(l.check_distinct_tokens(7).is_ok());
        assert!(matches!(
            l.check_distinct_tokens(8),
            Err(TextError::TooManyDistinctTokens { max: 7 })
        ));
    }

    #[test]
    fn ngrams_boundary_including_zero() {
        let l = TextLimits::default();
        assert!(matches!(
            l.check_ngrams(0),
            Err(TextError::InvalidNgrams { n: 0, max: 8 })
        ));
        assert!(l.check_ngrams(1).is_ok());
        assert!(l.check_ngrams(8).is_ok());
        assert!(matches!(
            l.check_ngrams(9),
            Err(TextError::InvalidNgrams { n: 9, max: 8 })
        ));
    }

    #[test]
    fn output_sequence_length_boundary() {
        let l = small();
        assert!(l.check_output_sequence_length(8).is_ok());
        assert!(l.check_output_sequence_length(9).is_ok());
        assert!(matches!(
            l.check_output_sequence_length(10),
            Err(TextError::OutputSequenceLengthTooLarge { len: 10, max: 9 })
        ));
    }

    #[test]
    fn output_elements_boundary_and_overflow() {
        let l = small();
        assert!(matches!(l.check_output_elements(4, 4), Ok(16)));
        assert!(matches!(l.check_output_elements(4, 5), Ok(20)));
        assert!(matches!(
            l.check_output_elements(3, 7),
            Err(TextError::OutputTooLarge {
                elements: 21,
                max: 20
            })
        ));
        assert!(matches!(
            l.check_output_elements(usize::MAX, 2),
            Err(TextError::OutputTooLarge {
                elements: usize::MAX,
                ..
            })
        ));
    }

    #[test]
    fn output_elements_byte_size_guard_at_absolute_maximum() {
        let l = TextLimits::default()
            .with_max_output_elements(ABSOLUTE_MAX_OUTPUT_ELEMENTS)
            .unwrap_or_default();
        assert_eq!(l.max_output_elements(), ABSOLUTE_MAX_OUTPUT_ELEMENTS);
        assert!(matches!(
            l.check_output_elements(ABSOLUTE_MAX_OUTPUT_ELEMENTS, 1),
            Ok(n) if n == ABSOLUTE_MAX_OUTPUT_ELEMENTS
        ));
        assert!(matches!(
            l.check_output_elements(ABSOLUTE_MAX_OUTPUT_ELEMENTS + 1, 1),
            Err(TextError::OutputTooLarge { .. })
        ));
    }

    #[test]
    fn check_inputs_orders_count_then_each_element() {
        let l = small();
        let empty: [&str; 0] = [];
        assert!(l.check_inputs(&empty).is_ok());
        assert!(l.check_inputs(&["abcd", "a", ""]).is_ok());
        // 件数超過が要素超過より先に報告される。
        assert!(matches!(
            l.check_inputs(&["abcde", "a", "b", "c"]),
            Err(TextError::BatchTooLarge { len: 4, max: 3 })
        ));
        assert!(matches!(
            l.check_inputs(&["a", "abcde"]),
            Err(TextError::InputTooLong {
                index: 1,
                len: 5,
                max: 4
            })
        ));
    }

    #[test]
    fn check_corpus_orders_count_each_then_total() {
        let l = small();
        assert!(l.check_corpus(&["abc", "abc"]).is_ok());
        assert!(matches!(
            l.check_corpus(&["a", "b", "c", "d"]),
            Err(TextError::BatchTooLarge { .. })
        ));
        assert!(matches!(
            l.check_corpus(&["abcde"]),
            Err(TextError::InputTooLong { index: 0, .. })
        ));
        assert!(matches!(
            l.check_corpus(&["abcd", "abc"]),
            Err(TextError::CorpusTooLarge {
                total_bytes: 7,
                max: 6
            })
        ));
    }
}
