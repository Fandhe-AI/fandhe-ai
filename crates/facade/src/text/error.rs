//! 語彙 lookup 型のテキスト変換（`text` モジュール）の型付きエラー `TextError`。
//!
//! 役割: `text` 内の各段（上限検査・標準化・語彙構築・変換）が返す失敗を 1 つの
//! enum に集約する。呼び出し元は `limits`（本イシュー）と、後続イシューの語彙・
//! adapt・transform。設計の正は `docs/facade-text-vectorization-design.md` §6
//! （variant 案）・§8（上限）。
//!
//! 契約（OWASP A09）: **入力文字列・語彙の中身を一切持たない**。保持するのは位置
//! （index）・長さ・上限値だけで、`Display`／`Debug` からも文字列が漏れない。
//! `OnnxError`・`ModelError` と同型の自己完結型（内部クレートのエラー enum は
//! 抱えない。例外は形状不整合を表す `ShapeError` のみ）。
//!
//! 公開形は未承認のため `pub(crate)`。facade への公開は設計記録 §11 の 9。

use std::fmt;

use crate::ShapeError;

/// テキスト変換の失敗を表す型付きエラー。`#[non_exhaustive]`: 後続段の variant
/// 追加を非破壊にするため。
#[non_exhaustive]
#[derive(Debug)]
pub(crate) enum TextError {
    /// バッチ要素数（`transform`）またはコーパス件数（`adapt`）が上限超過。
    BatchTooLarge { len: usize, max: usize },
    /// `index` 番目の入力文字列のバイト数が上限超過。
    InputTooLong {
        index: usize,
        len: usize,
        max: usize,
    },
    /// `adapt` のコーパス総バイト数が上限超過（加算 overflow 時は
    /// `total_bytes = usize::MAX`）。
    CorpusTooLarge { total_bytes: usize, max: usize },
    /// 語彙数（予約 2 件を含む）が上限超過。
    VocabularyTooLarge { len: usize, max: usize },
    /// `index` 番目の語彙のバイト数が上限超過。
    VocabularyTokenTooLong {
        index: usize,
        len: usize,
        max: usize,
    },
    /// `index` 番目の語彙が空文字列。
    EmptyVocabularyToken { index: usize },
    /// `index` 番目の語彙が予約語（`""`・`"[UNK]"`）。
    ReservedVocabularyToken { index: usize },
    /// 語彙の `first` 番目と `second` 番目が重複。
    DuplicateVocabularyToken { first: usize, second: usize },
    /// `max_tokens` が 2 以下で語彙が入らない。
    InvalidMaxTokens { max_tokens: usize },
    /// n-gram の `n` が 0 または上限超過。
    InvalidNgrams { n: usize, max: usize },
    /// `output_sequence_length` が上限超過。
    OutputSequenceLengthTooLarge { len: usize, max: usize },
    /// `adapt` 中の異なり語数が上限超過。
    TooManyDistinctTokens { max: usize },
    /// 出力要素数 `B × L`（またはそのバイト数）が上限超過・overflow
    /// （overflow 時は `elements = usize::MAX`）。
    OutputTooLarge { elements: usize, max: usize },
    /// 上限の設定値が絶対上限を超えた（`TextLimits::with_*`）。`limit` は
    /// フィールド名の固定文字列。設計記録 §6 の variant 案にない追加分
    /// （§8「設定値が絶対上限を超えたら `Err`」を満たすため。公開時の承認対象）。
    LimitAboveAbsoluteMaximum {
        limit: &'static str,
        value: usize,
        max: usize,
    },
    /// テンソル形状の不整合。
    Shape(ShapeError),
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextError::BatchTooLarge { len, max } => write!(
                f,
                "バッチ（コーパス）の件数が上限を超えています（{len} 件 > 上限 {max} 件）"
            ),
            TextError::InputTooLong { index, len, max } => write!(
                f,
                "入力文字列が上限を超えています（index={index}・{len} バイト > 上限 {max} バイト）"
            ),
            TextError::CorpusTooLarge { total_bytes, max } => write!(
                f,
                "コーパス総バイト数が上限を超えています（{total_bytes} バイト > 上限 {max} バイト）"
            ),
            TextError::VocabularyTooLarge { len, max } => {
                write!(f, "語彙数が上限を超えています（{len} 件 > 上限 {max} 件）")
            }
            TextError::VocabularyTokenTooLong { index, len, max } => write!(
                f,
                "語彙が上限を超えています（index={index}・{len} バイト > 上限 {max} バイト）"
            ),
            TextError::EmptyVocabularyToken { index } => {
                write!(f, "語彙が空文字列です（index={index}）")
            }
            TextError::ReservedVocabularyToken { index } => {
                write!(f, "語彙が予約語です（index={index}）")
            }
            TextError::DuplicateVocabularyToken { first, second } => {
                write!(f, "語彙が重複しています（index={first} と index={second}）")
            }
            TextError::InvalidMaxTokens { max_tokens } => {
                write!(f, "max_tokens は 3 以上が必要です（指定値 {max_tokens}）")
            }
            TextError::InvalidNgrams { n, max } => {
                write!(f, "n-gram の n が不正です（n={n}・有効範囲 1..={max}）")
            }
            TextError::OutputSequenceLengthTooLarge { len, max } => write!(
                f,
                "output_sequence_length が上限を超えています（{len} > 上限 {max}）"
            ),
            TextError::TooManyDistinctTokens { max } => {
                write!(f, "異なり語数が上限 {max} 件を超えています")
            }
            TextError::OutputTooLarge { elements, max } => write!(
                f,
                "出力要素数が上限を超えています（{elements} 要素 > 上限 {max} 要素）"
            ),
            TextError::LimitAboveAbsoluteMaximum { limit, value, max } => write!(
                f,
                "上限の設定値が絶対上限を超えています（{limit}={value} > 絶対上限 {max}）"
            ),
            TextError::Shape(e) => write!(f, "テンソル形状が不正です: {e}"),
        }
    }
}

impl std::error::Error for TextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TextError::Shape(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    fn all_variants() -> Vec<TextError> {
        vec![
            TextError::BatchTooLarge { len: 3, max: 2 },
            TextError::InputTooLong {
                index: 1,
                len: 9,
                max: 8,
            },
            TextError::CorpusTooLarge {
                total_bytes: 9,
                max: 8,
            },
            TextError::VocabularyTooLarge { len: 5, max: 4 },
            TextError::VocabularyTokenTooLong {
                index: 2,
                len: 7,
                max: 6,
            },
            TextError::EmptyVocabularyToken { index: 0 },
            TextError::ReservedVocabularyToken { index: 1 },
            TextError::DuplicateVocabularyToken {
                first: 0,
                second: 4,
            },
            TextError::InvalidMaxTokens { max_tokens: 2 },
            TextError::InvalidNgrams { n: 0, max: 8 },
            TextError::OutputSequenceLengthTooLarge { len: 11, max: 10 },
            TextError::TooManyDistinctTokens { max: 10 },
            TextError::OutputTooLarge {
                elements: 11,
                max: 10,
            },
            TextError::LimitAboveAbsoluteMaximum {
                limit: "max_ngrams",
                value: 9,
                max: 8,
            },
            TextError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: 3,
            }),
        ]
    }

    #[test]
    fn display_is_nonempty_for_every_variant() {
        for e in all_variants() {
            assert!(!e.to_string().is_empty(), "{e:?}");
        }
    }

    #[test]
    fn source_is_some_only_for_shape() {
        for e in all_variants() {
            let is_shape = matches!(e, TextError::Shape(_));
            assert_eq!(e.source().is_some(), is_shape, "{e:?}");
        }
    }

    #[test]
    fn variants_hold_numbers_only_so_no_input_text_leaks() {
        // 構造上、文字列を持つのは固定の `limit` 名だけ。目印の文字列を入力に
        // 見立てても Display／Debug に現れないことの回帰ガード。
        let marker = "SECRET_MARKER_TOKEN";
        for e in all_variants() {
            assert!(!e.to_string().contains(marker));
            assert!(!format!("{e:?}").contains(marker));
        }
    }
}
