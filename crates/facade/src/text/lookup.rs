//! 分割済みトークンの語彙 lookup 専用型 `StringLookup`（Keras `StringLookup` 相当の最小形）。
//!
//! 役割: `fandhe_ai::text` の公開型。標準化・分割をせず、利用者が渡したトークン 1 件を
//! 1 つの `i32` id へ引く（順引き `lookup`）と、id から文字列へ戻す（逆引き `token`／
//! `invert`）。内部は `TextVectorization` と同じ `vocab::Vocabulary` を使うので、同じ語彙なら
//! 同じ id になる（0 = パディング `""`、1 = OOV `"[UNK]"`、2.. = 語彙）。
//! Keras の既定（OOV = 0）とは意図して違える。`Var::embedding(.., Some(0))` との整合のため
//! で、Keras との bit 互換は契約にしない。
//!
//! 公開形の正は `docs/facade-text-output-modes-string-lookup-decision.md` §4・§5・§7。
//! 実装記録は `docs/facade-text-vectorization-design.md` §20。
//!
//! 契約（OWASP A03・A04・A09）:
//! - 件数・バイト長・範囲はすべて確保・lookup より先に検査し、型付き `Err` で返す
//! - 範囲外 id で panic しない。エラーと `Debug` は入力文字列・語彙の全文を持たない
//! - ホスト側の処理だけで完結し、バックエンドに依存しない

use std::fmt;

use super::adapt::adapt;
use super::error::TextError;
use super::limits::TextLimits;
use super::split::Split;
use super::standardize::Standardize;
use super::vocab::{Vocabulary, check_max_tokens, check_vocabulary_within_max_tokens};
use crate::Tensor;

/// `StringLookup` の構築設定。既定は `max_tokens: None`・`limits: TextLimits::default()`。
///
/// - `max_tokens`: 予約 2 件を含む語彙数の上限。2 以下は `InvalidMaxTokens`。
///   `from_vocabulary` は超過を `VocabularyTooLarge` にし、`adapt` は上位 `m - 2` 件で切り詰める
/// - `limits`: 件数・バイト数の上限（[`TextLimits`]）
///
/// `#[non_exhaustive]` なのでフィールドは構造体リテラルではなく `default()` とビルダで設定する。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StringLookupConfig {
    /// 予約 2 件を含む語彙数の上限。`None` は無制限（`limits` の上限は常に適用）。
    pub max_tokens: Option<usize>,
    /// 件数・バイト数の上限。
    pub limits: TextLimits,
}

impl StringLookupConfig {
    /// `max_tokens` に `Some(max_tokens)` を設定する。
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// 件数・バイト数の上限を設定する。
    pub fn with_limits(mut self, limits: TextLimits) -> Self {
        self.limits = limits;
        self
    }
}

/// 分割済みトークンを id へ引く（順引き）・id から文字列へ戻す（逆引き）語彙表。
///
/// id は 0 = パディング（`""`）、1 = OOV（`"[UNK]"`）、2.. = 語彙。lookup はバイト完全一致で
/// 大小文字を区別する。`Debug` は設定と語彙の件数だけを出し、語彙の中身を出さない。
///
/// ```
/// use fandhe_ai::text::{StringLookup, StringLookupConfig, TextError};
///
/// let sl = StringLookup::from_vocabulary(StringLookupConfig::default(), &["Cat", "dog"]).unwrap();
/// let ids = sl.lookup(&["Cat", "cat", "dog", ""]).unwrap();
/// assert_eq!(ids.shape(), &[4]);
/// assert_eq!(ids.host_slice().to_vec(), [2, 1, 3, 0]);
/// assert_eq!(sl.token(3).unwrap(), "dog");
/// assert!(matches!(sl.token(9), Err(TextError::IdOutOfRange { index: 0 })));
/// assert_eq!(sl.invert(&ids).unwrap(), ["Cat", "[UNK]", "dog", ""]);
///
/// // 予約語は語彙に入れられない。
/// let err = StringLookup::from_vocabulary(StringLookupConfig::default(), &["[UNK]"]).unwrap_err();
/// assert!(matches!(err, TextError::ReservedVocabularyToken { index: 0 }));
/// ```
#[derive(Clone)]
pub struct StringLookup {
    config: StringLookupConfig,
    vocabulary: Vocabulary,
}

impl StringLookup {
    /// 語彙を直接指定して作る。id は入力順に 2 から振る。
    ///
    /// 検査順（確保より先）: `max_tokens`（2 以下は `InvalidMaxTokens`）→ 語彙数が
    /// `max_tokens` を超えるか（`VocabularyTooLarge`。黙って切り詰めない）→ 件数・バイト長・
    /// 空（`EmptyVocabularyToken`）・予約語 `"[UNK]"`（`ReservedVocabularyToken`）・
    /// 重複（`DuplicateVocabularyToken`）。
    pub fn from_vocabulary<S: AsRef<str>>(
        config: StringLookupConfig,
        vocabulary: &[S],
    ) -> Result<Self, TextError> {
        check_max_tokens(config.max_tokens)?;
        check_vocabulary_within_max_tokens(config.max_tokens, vocabulary.len())?;
        let vocabulary = Vocabulary::from_tokens(&config.limits, vocabulary)?;
        Ok(Self { config, vocabulary })
    }

    /// トークン列から頻度順に語彙を構築して作る。
    ///
    /// 各要素を 1 トークンとして数える（空白を含んでも分割しない）。`""`・`"[UNK]"` は数えない。
    /// 語彙は頻度降順で同頻度はバイト列の辞書順。`max_tokens` が `Some(m)` なら上位 `m - 2` 件で
    /// 切り詰める（`from_vocabulary` と違い頻度で選ぶ構築のため）。語彙 1 件のバイト数は
    /// 検査しないので、結果を既定の `TextLimits` の `from_vocabulary` へ渡すと読み戻せない
    /// 場合がある。
    pub fn adapt<S: AsRef<str>>(
        config: StringLookupConfig,
        tokens: &[S],
    ) -> Result<Self, TextError> {
        check_max_tokens(config.max_tokens)?;
        let vocabulary = adapt(
            tokens,
            Standardize::None,
            Split::None,
            None,
            config.max_tokens,
            &config.limits,
        )?;
        Ok(Self { config, vocabulary })
    }

    /// トークン列を id 列 `[N]` の `Tensor<i32>` へ引く（順引き）。
    ///
    /// 語彙にない語は 1（OOV）、`""` は 0。`N = 0` は形状 `[0]`。件数（`BatchTooLarge`）・
    /// 各要素のバイト数（`InputTooLong`）・出力要素数（`OutputTooLarge`）を確保より先に検査する。
    pub fn lookup<S: AsRef<str>>(&self, tokens: &[S]) -> Result<Tensor<i32>, TextError> {
        let limits = &self.config.limits;
        limits.check_inputs(tokens)?;
        let n = limits.check_output_elements(tokens.len(), 1)?;
        let mut ids: Vec<i32> = Vec::with_capacity(n);
        for t in tokens {
            ids.push(self.vocabulary.lookup(t.as_ref()));
        }
        Tensor::new(ids, &[n]).map_err(TextError::Shape)
    }

    /// id から文字列へ戻す（逆引き）。負または語彙数以上の id は `IdOutOfRange`。
    ///
    /// 単独の id なので `index` は常に 0。
    pub fn token(&self, id: i32) -> Result<&str, TextError> {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.vocabulary.tokens().get(i))
            .map(String::as_str)
            .ok_or(TextError::IdOutOfRange { index: 0 })
    }

    /// id のテンソルを行優先で平坦化して文字列列へ戻す（rank は問わない）。
    ///
    /// 検査順（確保より先）: 件数（`OutputTooLarge`。readback より前）→ 各 id の範囲
    /// （`IdOutOfRange`。`index` は最初の範囲外の位置）→ 出力の総バイト数。総バイト数の上限は
    /// `TextLimits` の `max_corpus_bytes` を流用し、超過は `CorpusTooLarge` で返す。
    pub fn invert(&self, ids: &Tensor<i32>) -> Result<Vec<String>, TextError> {
        let limits = &self.config.limits;
        let numel = limits.check_output_elements(ids.numel(), 1)?;
        // `String` は 24 バイトで `i32` より大きいので、確保バイト数の上限を別に検査する。
        match numel.checked_mul(std::mem::size_of::<String>()) {
            Some(bytes) if bytes <= isize::MAX as usize => {}
            _ => {
                return Err(TextError::OutputTooLarge {
                    elements: numel,
                    max: isize::MAX as usize / std::mem::size_of::<String>(),
                });
            }
        }
        let host = ids.host_slice();
        let tokens = self.vocabulary.tokens();
        // 走査 1: 範囲検査のみ（中間 Vec を確保しない）。最初の範囲外の位置を返す。
        let lookup = |id: i32| usize::try_from(id).ok().and_then(|i| tokens.get(i));
        if let Some(index) = host.iter().position(|&id| lookup(id).is_none()) {
            return Err(TextError::IdOutOfRange { index });
        }
        // 走査 2: 総バイト数の検査（範囲は検証済みなので `unwrap_or` は到達しない）。
        limits
            .check_corpus_total_bytes(host.iter().map(|&id| lookup(id).map_or(0, |t| t.len())))?;
        // 走査 3: 全検査後にだけ結果を確保して構築する。
        Ok(host.iter().filter_map(|&id| lookup(id).cloned()).collect())
    }

    /// id 順の語彙（予約 2 件を先頭に含む）。
    pub fn vocabulary(&self) -> &[String] {
        self.vocabulary.tokens()
    }

    /// 語彙数（予約 2 件を含む）。
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.vocabulary_size()
    }

    /// 構築に使った設定。
    pub fn config(&self) -> &StringLookupConfig {
        &self.config
    }
}

/// 設定と語彙の件数だけを出す（語彙の全文を出さない。OWASP A09）。
impl fmt::Debug for StringLookup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StringLookup")
            .field("config", &self.config)
            .field("vocabulary_size", &self.vocabulary.vocabulary_size())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_limits(limits: TextLimits, vocab: &[&str]) -> StringLookup {
        let Ok(sl) =
            StringLookup::from_vocabulary(StringLookupConfig::default().with_limits(limits), vocab)
        else {
            panic!("構築に失敗");
        };
        sl
    }

    fn build(tokens: &[&str]) -> StringLookup {
        with_limits(TextLimits::default(), tokens)
    }

    fn ids(v: Vec<i32>) -> Tensor<i32> {
        let n = v.len();
        let Ok(t) = Tensor::new(v, &[n]) else {
            panic!("テンソル作成に失敗");
        };
        t
    }

    #[test]
    fn ids_follow_input_order_and_reserved() {
        let sl = build(&["b", "a", "日本"]);
        assert_eq!(sl.vocabulary(), ["", "[UNK]", "b", "a", "日本"]);
        assert_eq!(sl.vocabulary_size(), 5);
        let Ok(t) = sl.lookup(&["a", "b", "日本", "A", "", "[UNK]"]) else {
            panic!("lookup 失敗");
        };
        assert_eq!(t.shape(), &[6]);
        assert_eq!(t.host_slice().to_vec(), [3, 2, 4, 1, 0, 1]);
    }

    #[test]
    fn invalid_vocabulary_is_rejected() {
        let c = StringLookupConfig::default;
        assert!(matches!(
            StringLookup::from_vocabulary(c(), &["a", "a"]),
            Err(TextError::DuplicateVocabularyToken {
                first: 0,
                second: 1
            })
        ));
        assert!(matches!(
            StringLookup::from_vocabulary(c(), &["[UNK]"]),
            Err(TextError::ReservedVocabularyToken { index: 0 })
        ));
        assert!(matches!(
            StringLookup::from_vocabulary(c(), &["a", ""]),
            Err(TextError::EmptyVocabularyToken { index: 1 })
        ));
    }

    #[test]
    fn max_tokens_is_fail_closed() {
        let cfg = |m| StringLookupConfig::default().with_max_tokens(m);
        assert!(matches!(
            StringLookup::from_vocabulary(cfg(2), &["a"]),
            Err(TextError::InvalidMaxTokens { max_tokens: 2 })
        ));
        assert!(matches!(
            StringLookup::from_vocabulary(cfg(3), &["a", "b"]),
            Err(TextError::VocabularyTooLarge { len: 4, max: 3 })
        ));
        assert!(StringLookup::from_vocabulary(cfg(4), &["a", "b"]).is_ok());
        assert!(matches!(
            StringLookup::adapt(cfg(1), &["a"]),
            Err(TextError::InvalidMaxTokens { max_tokens: 1 })
        ));
    }

    #[test]
    fn lookup_empty_batch_and_limits() {
        let sl = build(&["a"]);
        let Ok(t) = sl.lookup::<&str>(&[]) else {
            panic!("空バッチ失敗");
        };
        assert_eq!(t.shape(), &[0]);

        let Ok(l) = TextLimits::default().with_max_batch(1) else {
            panic!("limits");
        };
        assert!(matches!(
            with_limits(l, &["a"]).lookup(&["a", "a"]),
            Err(TextError::BatchTooLarge { len: 2, max: 1 })
        ));

        let Ok(l) = TextLimits::default().with_max_input_bytes(2) else {
            panic!("limits");
        };
        assert!(matches!(
            with_limits(l, &["a"]).lookup(&["a", "abc"]),
            Err(TextError::InputTooLong { index: 1, .. })
        ));

        let Ok(l) = TextLimits::default().with_max_output_elements(1) else {
            panic!("limits");
        };
        assert!(matches!(
            with_limits(l, &["a"]).lookup(&["a", "a"]),
            Err(TextError::OutputTooLarge {
                elements: 2,
                max: 1
            })
        ));
    }

    #[test]
    fn token_bounds() {
        let sl = build(&["a", "b"]);
        assert_eq!(sl.token(0).ok(), Some(""));
        assert_eq!(sl.token(1).ok(), Some("[UNK]"));
        assert_eq!(sl.token(3).ok(), Some("b"));
        for bad in [-1, 4, i32::MAX, i32::MIN] {
            assert!(
                matches!(sl.token(bad), Err(TextError::IdOutOfRange { index: 0 })),
                "{bad}"
            );
        }
    }

    #[test]
    fn invert_round_trip_flatten_and_empty() {
        let sl = build(&["a", "b"]);
        let Ok(t) = Tensor::new(vec![2, 3, 1, 0], &[2, 2]) else {
            panic!("テンソル作成に失敗");
        };
        let expected: Vec<String> = ["a", "b", "[UNK]", ""].map(String::from).to_vec();
        assert_eq!(sl.invert(&t).ok(), Some(expected));
        let Ok(e) = Tensor::<i32>::new(vec![], &[0]) else {
            panic!("空テンソル");
        };
        assert_eq!(sl.invert(&e).ok(), Some(Vec::<String>::new()));
    }

    #[test]
    fn invert_reports_first_out_of_range_position() {
        let sl = build(&["a"]);
        assert!(matches!(
            sl.invert(&ids(vec![2, 0, 9, -1])),
            Err(TextError::IdOutOfRange { index: 2 })
        ));
    }

    #[test]
    fn invert_limits() {
        let Ok(l) = TextLimits::default().with_max_output_elements(2) else {
            panic!("limits");
        };
        // 件数超過は範囲外 id より先に OutputTooLarge。
        assert!(matches!(
            with_limits(l, &["a"]).invert(&ids(vec![99, 99, 99])),
            Err(TextError::OutputTooLarge {
                elements: 3,
                max: 2
            })
        ));

        let Ok(l) = TextLimits::default().with_max_corpus_bytes(3) else {
            panic!("limits");
        };
        let sl = with_limits(l, &["ab"]);
        assert!(sl.invert(&ids(vec![2])).is_ok());
        assert!(matches!(
            sl.invert(&ids(vec![2, 2])),
            Err(TextError::CorpusTooLarge {
                total_bytes: 4,
                max: 3
            })
        ));
    }

    #[test]
    fn adapt_orders_by_frequency_and_truncates() {
        let Ok(sl) = StringLookup::adapt(
            StringLookupConfig::default(),
            &["b", "a b", "a b", "", "[UNK]", "c", "c"],
        ) else {
            panic!("adapt 失敗");
        };
        // "a b" は空白を含む 1 語（2 回）。"c" は 2 回、"b" は 1 回。
        assert_eq!(sl.vocabulary(), ["", "[UNK]", "a b", "c", "b"]);
        let Ok(sl) = StringLookup::adapt(
            StringLookupConfig::default().with_max_tokens(4),
            &["b", "a b", "a b", "c", "c"],
        ) else {
            panic!("adapt 失敗");
        };
        assert_eq!(sl.vocabulary(), ["", "[UNK]", "a b", "c"]);
    }

    #[test]
    fn debug_and_errors_hide_input_text() {
        let marker = "SECRET_MARKER";
        let sl = build(&[marker]);
        assert!(!format!("{sl:?}").contains(marker));
        let errs = [
            StringLookup::from_vocabulary(StringLookupConfig::default(), &[marker, marker]).err(),
            StringLookup::from_vocabulary(StringLookupConfig::default(), &["[UNK]", marker]).err(),
            StringLookup::from_vocabulary(
                StringLookupConfig::default().with_max_tokens(3),
                &[marker, "x"],
            )
            .err(),
        ];
        for e in errs {
            let Some(e) = e else { panic!("Err のはず") };
            assert!(!e.to_string().contains(marker));
            assert!(!format!("{e:?}").contains(marker));
        }
    }
}
