//! 語彙の直接指定と lookup（`text` モジュール）。
//!
//! 役割: 利用者がメモリ上で渡した語彙から lookup 表を作り、トークン文字列を
//! `i32` の id へ引く。id 0 はパディング（トークン `""`）、id 1 は OOV
//! （トークン `"[UNK]"`）、利用者の語彙は id 2 から入力順に振る。呼び出し元は
//! 後続の adapt（#2901。頻度順に並べた語彙を `from_tokens` へ渡す）と
//! transform（#2902。分割後のトークンを `lookup` で id 化する）。設計の正は
//! `docs/facade-text-vectorization-design.md` §3・§6・§8・§13。
//!
//! 設計上の要点:
//! - 構築は件数（予約 2 件込み）→ 要素ごとのバイト長・空・予約・重複の順に検査し、
//!   全件の検査を通るまで所有の `String` を確保しない（設計記録 §8「確保より先」）。
//! - `""` は空トークンであり予約トークンでもあるが、検査順により
//!   `EmptyVocabularyToken` として報告する。`"[UNK]"` は
//!   `ReservedVocabularyToken`。どちらも `Err` であり、この分類は実装上の選択
//!   （設計記録の承認事項ではない）。
//! - 直接指定の語彙は標準化しない。lookup は完全なバイト一致で、大小文字を区別する。
//! - `lookup("")` は 0 を返す。分割なし（`Split::None`）で空入力を変換すると
//!   パディングと区別できなくなるため、transform 側での扱いは #2902 が決める。
//! - エラーと `Debug` は index・長さ・件数だけを持ち、語彙の文字列を漏らさない
//!   （OWASP A09）。
//!
//! 型名・メソッド名は内部実装上の選択で、公開形は設計記録 §11 の 9 の承認時に
//! 決める。facade へは公開しない（`pub(crate)`）。

use std::collections::HashMap;
use std::fmt;

use super::error::TextError;
use super::limits::TextLimits;

/// パディングのトークン（空文字列）。
pub(crate) const PADDING_TOKEN: &str = "";
/// 語彙外（OOV）のトークン。
pub(crate) const OOV_TOKEN: &str = "[UNK]";
/// パディングの id。
pub(crate) const PADDING_ID: i32 = 0;
/// OOV の id。
pub(crate) const OOV_ID: i32 = 1;
/// 予約トークンの件数（利用者の語彙の id はこの値から始まる）。
pub(crate) const RESERVED_COUNT: usize = 2;

/// 予約 2 件と利用者の語彙を持つ lookup 表。
#[derive(Clone)]
pub(crate) struct Vocabulary {
    /// index 0 = `""`、1 = `"[UNK]"`、2.. = 利用者の語彙（入力順）。
    tokens: Vec<String>,
    ids: HashMap<String, i32>,
}

impl Vocabulary {
    /// 利用者の語彙から lookup 表を作る。id は入力順に 2 から振る。
    ///
    /// エラーの index・`first`・`second` は入力 slice 上の位置（予約分を足さない）。
    pub(crate) fn from_tokens<S: AsRef<str>>(
        limits: &TextLimits,
        tokens: &[S],
    ) -> Result<Self, TextError> {
        // 件数の検査を確保より先に行う。加算の overflow は飽和させて Err に倒す。
        let total = tokens.len().saturating_add(RESERVED_COUNT);
        limits.check_vocabulary_size(total)?;

        // 全件の検査が終わるまで String を確保しない（借用で重複を検出する）。
        // 未検査の要素に比例した事前確保はしない（先頭の不正要素で即 Err にできる入力に
        // 件数分の確保が走らないよう、検証済み要素のみを段階的に挿入する。設計記録 §8・§13）。
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            let t = t.as_ref();
            limits.check_vocabulary_token_len(i, t.len())?;
            if t.is_empty() {
                return Err(TextError::EmptyVocabularyToken { index: i });
            }
            if t == OOV_TOKEN {
                return Err(TextError::ReservedVocabularyToken { index: i });
            }
            if let Some(&first) = seen.get(t) {
                return Err(TextError::DuplicateVocabularyToken { first, second: i });
            }
            seen.insert(t, i);
        }

        let mut out_tokens: Vec<String> = Vec::with_capacity(total);
        let mut ids: HashMap<String, i32> = HashMap::with_capacity(total);
        out_tokens.push(PADDING_TOKEN.to_owned());
        out_tokens.push(OOV_TOKEN.to_owned());
        ids.insert(PADDING_TOKEN.to_owned(), PADDING_ID);
        ids.insert(OOV_TOKEN.to_owned(), OOV_ID);
        for (i, t) in tokens.iter().enumerate() {
            // 件数検査で total <= i32::MAX が保証されるため失敗しないが、fail-closed にする。
            let id = i
                .checked_add(RESERVED_COUNT)
                .and_then(|v| i32::try_from(v).ok())
                .ok_or(TextError::VocabularyTooLarge {
                    len: total,
                    max: i32::MAX as usize,
                })?;
            let t = t.as_ref();
            out_tokens.push(t.to_owned());
            ids.insert(t.to_owned(), id);
        }
        Ok(Self {
            tokens: out_tokens,
            ids,
        })
    }

    /// トークンの id。語彙になければ `OOV_ID`。失敗しない。
    pub(crate) fn lookup(&self, token: &str) -> i32 {
        self.ids.get(token).copied().unwrap_or(OOV_ID)
    }

    /// id 順のトークン（予約 2 件を先頭に含む）。
    pub(crate) fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// 語彙数（予約 2 件を含む）。
    pub(crate) fn vocabulary_size(&self) -> usize {
        self.tokens.len()
    }
}

/// 語彙の全文を出さず件数だけを出す（設計記録 §6）。
impl fmt::Debug for Vocabulary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vocabulary")
            .field("len", &self.tokens.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_limits() -> TextLimits {
        TextLimits::default()
            .with_max_vocabulary_size(4)
            .and_then(|l| l.with_max_vocabulary_token_bytes(6))
            .unwrap_or_default()
    }

    fn build(tokens: &[&str]) -> Result<Vocabulary, TextError> {
        Vocabulary::from_tokens(&TextLimits::default(), tokens)
    }

    #[test]
    fn known_unknown_and_reserved_lookup() {
        let Ok(v) = build(&["cat", "dog"]) else {
            panic!("構築に失敗");
        };
        assert_eq!(v.lookup("cat"), 2);
        assert_eq!(v.lookup("dog"), 3);
        assert_eq!(v.lookup("bird"), OOV_ID);
        assert_eq!(v.lookup(PADDING_TOKEN), PADDING_ID);
        assert_eq!(v.lookup(OOV_TOKEN), OOV_ID);
        assert_eq!(v.tokens()[0], PADDING_TOKEN);
        assert_eq!(v.tokens()[1], OOV_TOKEN);
    }

    #[test]
    fn round_trip_and_size() {
        let Ok(v) = build(&["cat", "dog", "emu"]) else {
            panic!("構築に失敗");
        };
        assert_eq!(v.vocabulary_size(), 3 + RESERVED_COUNT);
        assert_eq!(v.vocabulary_size(), v.tokens().len());
        for t in v.tokens() {
            let Ok(id) = usize::try_from(v.lookup(t)) else {
                panic!("id が負");
            };
            assert_eq!(&v.tokens()[id], t);
        }
    }

    #[test]
    fn empty_vocabulary_maps_everything_to_oov() {
        let Ok(v) = build(&[]) else {
            panic!("構築に失敗");
        };
        assert_eq!(v.vocabulary_size(), 2);
        assert_eq!(v.lookup("x"), OOV_ID);
    }

    #[test]
    fn case_sensitive_and_multibyte() {
        let Ok(v) = build(&["a", "日本"]) else {
            panic!("構築に失敗");
        };
        assert_eq!(v.lookup("A"), OOV_ID);
        assert_eq!(v.lookup("日本"), 3);
    }

    #[test]
    fn empty_token_is_empty_error() {
        assert!(matches!(
            build(&["a", ""]),
            Err(TextError::EmptyVocabularyToken { index: 1 })
        ));
    }

    #[test]
    fn oov_token_is_reserved_error_but_lowercase_is_accepted() {
        assert!(matches!(
            build(&["a", "b", "[UNK]"]),
            Err(TextError::ReservedVocabularyToken { index: 2 })
        ));
        assert!(build(&["[unk]"]).is_ok());
    }

    #[test]
    fn duplicate_reports_first_two_occurrences() {
        assert!(matches!(
            build(&["x", "y", "x", "x"]),
            Err(TextError::DuplicateVocabularyToken {
                first: 0,
                second: 2
            })
        ));
    }

    #[test]
    fn vocabulary_size_boundary() {
        let l = small_limits();
        assert!(Vocabulary::from_tokens(&l, &["a", "b"]).is_ok());
        assert!(matches!(
            Vocabulary::from_tokens(&l, &["a", "b", "c"]),
            Err(TextError::VocabularyTooLarge { len: 5, max: 4 })
        ));
    }

    #[test]
    fn token_bytes_boundary() {
        let l = small_limits();
        assert!(Vocabulary::from_tokens(&l, &["日本"]).is_ok());
        assert!(matches!(
            Vocabulary::from_tokens(&l, &["a", "日本a"]),
            Err(TextError::VocabularyTokenTooLong {
                index: 1,
                len: 7,
                max: 6
            })
        ));
    }

    #[test]
    fn invalid_first_token_fails_before_scanning_rest() {
        // 先頭が空文字列なら index 0 で即 Err（残りの大量要素は検査されない）。
        let mut toks: Vec<String> = vec![String::new()];
        toks.extend((0..50_000).map(|i| format!("t{i}")));
        assert!(matches!(
            Vocabulary::from_tokens(&TextLimits::default(), &toks),
            Err(TextError::EmptyVocabularyToken { index: 0 })
        ));
    }

    #[test]
    fn check_order() {
        let l = small_limits();
        // 件数超過が要素の検査より先。
        assert!(matches!(
            Vocabulary::from_tokens(&l, &["日本a", "b", "c"]),
            Err(TextError::VocabularyTooLarge { .. })
        ));
        // 同じ index ではバイト長が予約より先（上限 4 に対し "[UNK]" は 5 バイト）。
        let Ok(l4) = l.with_max_vocabulary_token_bytes(4) else {
            panic!("上限の設定に失敗");
        };
        assert!(matches!(
            Vocabulary::from_tokens(&l4, &["[UNK]"]),
            Err(TextError::VocabularyTokenTooLong { index: 0, .. })
        ));
        // 先に現れる重複が、後ろの予約語より先に報告される。
        assert!(matches!(
            build(&["a", "a", "[UNK]"]),
            Err(TextError::DuplicateVocabularyToken {
                first: 0,
                second: 1
            })
        ));
    }

    #[test]
    fn debug_and_errors_do_not_leak_tokens() {
        let Ok(v) = build(&["SECRET_MARKER_TOKEN"]) else {
            panic!("構築に失敗");
        };
        let dbg = format!("{v:?}");
        assert!(!dbg.contains("SECRET_MARKER_TOKEN"));
        assert!(dbg.contains('3'));

        let l = small_limits();
        let errs = [
            Vocabulary::from_tokens(&l, &["SECRET_LONG_TOKEN"]).err(),
            build(&["SECRET_A", "SECRET_A"]).err(),
            build(&["SECRET_A", ""]).err(),
            build(&["SECRET_A", "[UNK]"]).err(),
        ];
        for e in errs {
            let Some(e) = e else {
                panic!("Err になるはず");
            };
            assert!(!format!("{e:?}").contains("SECRET"));
            assert!(!format!("{e}").contains("SECRET"));
        }
    }

    #[test]
    fn clone_keeps_lookup() {
        let Ok(v) = build(&["cat"]) else {
            panic!("構築に失敗");
        };
        let c = v.clone();
        assert_eq!(c.lookup("cat"), v.lookup("cat"));
        assert_eq!(c.lookup("zzz"), OOV_ID);
    }
}
