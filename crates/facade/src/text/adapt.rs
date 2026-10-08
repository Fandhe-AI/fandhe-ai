//! メモリ上のコーパスから頻度に基づいて語彙を構築する `adapt`（`text` モジュール）。
//!
//! 役割: 標準化（`standardize`）→ 分割（`split`）→ n-gram（`ngram`）の結果を
//! 数え、「頻度の降順、同頻度はバイト列の辞書順（昇順）」で並べた語彙を
//! `Vocabulary::from_tokens` へ渡して作る（設計記録
//! `docs/facade-text-vectorization-design.md` §3）。呼び出し元は後続の transform
//! （#2902）と、公開時の facade（設計記録 §11 の 9）。構築経路は `from_tokens` の
//! 1 本に保ち、検査を飛ばすコンストラクタは作らない。
//!
//! 決定性: 集計は `HashMap` だが、反復順には依存せず全体をソートして並びを確定する。
//! キー（語）は互いに異なるため比較に引き分けがなく、不安定ソートでも結果は一意。
//! `str` の `Ord` はバイト単位の辞書順で、設計記録 §3 の規定と一致する。
//!
//! 検査の時機（設計記録 §8）: 設定値（`max_tokens`・n）→ コーパスの件数・各要素・
//! 総バイト数 → 集計中の異なり語数（`HashMap` へ入れる前）→ 語彙数（ソート・確保の前）。
//! 頻度は `u64` を `checked_add` で加算し、overflow は型付き `Err`。
//!
//! 設計記録が定めていない点（実装上の選択。公開時の承認対象として #2896 へ申し送る）:
//! - 予約トークン（`""`・`"[UNK]"`）がコーパスから出ても数えない。`lookup` が元々
//!   これらを予約 id に引くため整合し、`from_tokens` の予約語エラーで adapt が
//!   内容次第で失敗するのを避ける。
//! - 語彙 1 件のバイト数は adapt では検査しない（§8 は検査を `from_vocabulary` 入口に
//!   定める）。各トークンは入力 1 件の部分文字列か k-gram で、長さは入力上限の
//!   高々 2 倍（`Split::Character` では k-gram の区切りが加わる）。そのため
//!   `from_tokens` への上限だけ絶対上限へ緩める。結果として、adapt が作った語彙は
//!   既定の上限（4 KiB）では `from_tokens` で読み戻せない場合がある。
//! - 頻度 overflow は `TextError::FrequencyOverflow`。
//!
//! 浮動小数点・GPU を通らないため、統一複合判定・FMA 契約は対象外（設計記録 §7）。

use std::borrow::Cow;
use std::collections::HashMap;

use super::error::TextError;
use super::limits::{ABSOLUTE_MAX_BYTES, TextLimits};
use super::ngram::ngrams;
use super::split::{Split, split};
use super::standardize::{Standardize, standardize};
use super::vocab::{OOV_TOKEN, PADDING_TOKEN, RESERVED_COUNT, Vocabulary};

/// 頻度を 1 増やす。overflow は `FrequencyOverflow`（語の文字列は持たない）。
fn increment(count: &mut u64) -> Result<(), TextError> {
    *count = count.checked_add(1).ok_or(TextError::FrequencyOverflow)?;
    Ok(())
}

/// 1 トークンを頻度表へ数える。予約トークンは飛ばす。新しい語は `HashMap` へ
/// 入れる**前**に異なり語数を検査する（設計記録 §8）。
fn count_token(
    map: &mut HashMap<String, u64>,
    tok: Cow<'_, str>,
    limits: &TextLimits,
) -> Result<(), TextError> {
    if tok == PADDING_TOKEN || tok == OOV_TOKEN {
        return Ok(());
    }
    // `String: Borrow<str>` により、既存の語は出現ごとの確保なしで引ける。
    if let Some(c) = map.get_mut(tok.as_ref()) {
        return increment(c);
    }
    let next = map
        .len()
        .checked_add(1)
        .ok_or(TextError::TooManyDistinctTokens {
            max: limits.max_distinct_tokens(),
        })?;
    limits.check_distinct_tokens(next)?;
    map.insert(tok.into_owned(), 1);
    Ok(())
}

/// コーパスから語彙を構築する。
///
/// - `ngrams`: `None` は分割トークンそのもの（1-gram）だけを数える。`Some(n)` は
///   1〜n-gram を数える。
/// - `max_tokens`: 予約 2 件を含む語彙数の上限。`None` は件数で切り詰めない。
///   2 以下は `InvalidMaxTokens`。
///
/// 戻り値の index 0 = `""`、1 = `"[UNK]"`、2.. = 頻度降順（同頻度はバイト列の
/// 辞書順）。Keras との並び順・`max_tokens` の数え方の bit 互換は契約にしない。
pub(crate) fn adapt<S: AsRef<str>>(
    corpus: &[S],
    standardize_mode: Standardize,
    split_mode: Split,
    ngrams_n: Option<usize>,
    max_tokens: Option<usize>,
    limits: &TextLimits,
) -> Result<Vocabulary, TextError> {
    // 設定値の検査はコーパスを読む前。空コーパスでも不正な設定を見逃さない。
    let keep = match max_tokens {
        Some(m) => Some(
            m.checked_sub(RESERVED_COUNT + 1)
                .map(|v| v + 1)
                .ok_or(TextError::InvalidMaxTokens { max_tokens: m })?,
        ),
        None => None,
    };
    if let Some(n) = ngrams_n {
        limits.check_ngrams(n)?;
    }
    limits.check_corpus(corpus)?;

    let mut map: HashMap<String, u64> = HashMap::new();
    for s in corpus {
        let std = standardize(s.as_ref(), standardize_mode);
        match ngrams_n {
            None => {
                for t in split(&std, split_mode) {
                    count_token(&mut map, Cow::Borrowed(t), limits)?;
                }
            }
            Some(n) => {
                let toks: Vec<&str> = split(&std, split_mode).collect();
                for g in ngrams(&toks, n, limits)? {
                    count_token(&mut map, g, limits)?;
                }
            }
        }
    }

    // 語彙数はソート・確保の前に検査する（加算 overflow は飽和させて Err に倒す）。
    let distinct = map.len();
    let selected = keep.map_or(distinct, |k| distinct.min(k));
    limits.check_vocabulary_size(selected.saturating_add(RESERVED_COUNT))?;

    let mut entries: Vec<(String, u64)> = map.into_iter().collect();
    entries.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    entries.truncate(selected);
    let tokens: Vec<String> = entries.into_iter().map(|(t, _)| t).collect();

    let relaxed = limits.with_max_vocabulary_token_bytes(ABSOLUTE_MAX_BYTES)?;
    Vocabulary::from_tokens(&relaxed, &tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(corpus: &[&str], max_tokens: Option<usize>) -> Result<Vocabulary, TextError> {
        adapt(
            corpus,
            Standardize::None,
            Split::Whitespace,
            None,
            max_tokens,
            &TextLimits::default(),
        )
    }

    fn toks(corpus: &[&str], max_tokens: Option<usize>) -> Vec<String> {
        match run(corpus, max_tokens) {
            Ok(v) => v.tokens().to_vec(),
            Err(e) => panic!("adapt が失敗: {e:?}"),
        }
    }

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    fn permutations(items: &[&'static str]) -> Vec<Vec<&'static str>> {
        if items.len() <= 1 {
            return vec![items.to_vec()];
        }
        let mut out = Vec::new();
        for i in 0..items.len() {
            let mut rest = items.to_vec();
            let head = rest.remove(i);
            for mut p in permutations(&rest) {
                p.insert(0, head);
                out.push(p);
            }
        }
        out
    }

    #[test]
    fn orders_by_frequency_then_bytes() {
        assert_eq!(
            toks(&["b a c", "a b", "a"], None),
            strs(&["", "[UNK]", "a", "b", "c"])
        );
    }

    #[test]
    fn ties_use_byte_order_for_multibyte() {
        // バイト順: "z"(0x7a) < "é"(0xc3..) < "日"(0xe6..)
        assert_eq!(
            toks(&["日 é z"], None),
            strs(&["", "[UNK]", "z", "é", "日"])
        );
    }

    #[test]
    fn corpus_order_does_not_change_result() {
        let base = ["x y", "y z", "z w", "w x z"];
        let expected = toks(&base, None);
        for p in permutations(&base) {
            assert_eq!(toks(&p, None), expected, "{p:?}");
        }
    }

    #[test]
    fn independent_of_hashmap_iteration_order() {
        let corpus = ["d c b a", "h g f e", "a e"];
        let first = toks(&corpus, None);
        for _ in 0..16 {
            assert_eq!(toks(&corpus, None), first);
        }
    }

    #[test]
    fn max_tokens_boundaries() {
        for m in 0..=2 {
            assert!(matches!(
                run(&["a"], Some(m)),
                Err(TextError::InvalidMaxTokens { max_tokens }) if max_tokens == m
            ));
        }
        let c = ["b a c", "a b", "a"];
        assert_eq!(toks(&c, Some(3)), strs(&["", "[UNK]", "a"]));
        assert_eq!(toks(&c, Some(4)), strs(&["", "[UNK]", "a", "b"]));
        assert_eq!(toks(&c, Some(100)), toks(&c, None));
    }

    #[test]
    fn settings_are_checked_before_corpus() {
        let limits = match TextLimits::default().with_max_batch(1) {
            Ok(l) => l,
            Err(e) => panic!("{e:?}"),
        };
        let big = ["a", "b"];
        let r = adapt(
            &big,
            Standardize::None,
            Split::Whitespace,
            None,
            Some(0),
            &limits,
        );
        assert!(matches!(r, Err(TextError::InvalidMaxTokens { .. })));
        for n in [0, 9] {
            let r = adapt::<&str>(
                &[],
                Standardize::None,
                Split::Whitespace,
                Some(n),
                None,
                &TextLimits::default(),
            );
            assert!(matches!(r, Err(TextError::InvalidNgrams { .. })), "n={n}");
        }
    }

    #[test]
    fn corpus_limits_at_boundaries() {
        let go = |c: &[&str], l: &TextLimits| {
            adapt(c, Standardize::None, Split::Whitespace, None, None, l)
        };
        let l = TextLimits::default().with_max_batch(2).unwrap_or_default();
        assert!(go(&["a", "b"], &l).is_ok());
        assert!(matches!(
            go(&["a", "b", "c"], &l),
            Err(TextError::BatchTooLarge { .. })
        ));
        let l = TextLimits::default()
            .with_max_input_bytes(3)
            .unwrap_or_default();
        assert!(go(&["abc"], &l).is_ok());
        assert!(matches!(
            go(&["abcd"], &l),
            Err(TextError::InputTooLong { .. })
        ));
        let l = TextLimits::default()
            .with_max_corpus_bytes(5)
            .unwrap_or_default();
        assert!(go(&["abc", "de"], &l).is_ok());
        assert!(matches!(
            go(&["abc", "def"], &l),
            Err(TextError::CorpusTooLarge { .. })
        ));
    }

    #[test]
    fn distinct_limit_fires_only_on_new_tokens() {
        let l = TextLimits::default()
            .with_max_distinct_tokens(2)
            .unwrap_or_default();
        let go = |c: &[&str]| adapt(c, Standardize::None, Split::Whitespace, None, None, &l);
        assert!(go(&["a b a b a b"]).is_ok());
        assert!(matches!(
            go(&["a b c"]),
            Err(TextError::TooManyDistinctTokens { max: 2 })
        ));
    }

    #[test]
    fn vocabulary_size_is_checked_on_selected_count() {
        let l = TextLimits::default()
            .with_max_vocabulary_size(4)
            .unwrap_or_default();
        let go = |mt| {
            adapt(
                &["a b c"],
                Standardize::None,
                Split::Whitespace,
                None,
                mt,
                &l,
            )
        };
        assert!(matches!(
            go(None),
            Err(TextError::VocabularyTooLarge { .. })
        ));
        assert!(go(Some(4)).is_ok());
    }

    #[test]
    fn ngrams_share_one_frequency_table() {
        let r = adapt(
            &["a b", "a b"],
            Standardize::None,
            Split::Whitespace,
            Some(2),
            None,
            &TextLimits::default(),
        );
        let t = match r {
            Ok(v) => v.tokens().to_vec(),
            Err(e) => panic!("{e:?}"),
        };
        // a=2, b=2, "a b"=2: 同頻度はバイト順（"a" < "a b" < "b"）。
        assert_eq!(t, strs(&["", "[UNK]", "a", "a b", "b"]));
    }

    #[test]
    fn reserved_tokens_in_corpus_are_not_counted() {
        let r = adapt(
            &["", "!!!", "x"],
            Standardize::StripPunctuation,
            Split::None,
            None,
            None,
            &TextLimits::default(),
        );
        assert!(matches!(&r, Ok(v) if v.tokens() == strs(&["", "[UNK]", "x"])));
        assert_eq!(toks(&["[UNK] a"], None), strs(&["", "[UNK]", "a"]));
    }

    #[test]
    fn long_token_is_accepted_but_not_readable_back_with_defaults() {
        let long = "x".repeat(5000);
        let r = adapt(
            &[long.as_str()],
            Standardize::None,
            Split::None,
            None,
            None,
            &TextLimits::default(),
        );
        let v = match r {
            Ok(v) => v,
            Err(e) => panic!("{e:?}"),
        };
        assert_eq!(v.tokens().get(2).map(String::len), Some(5000));
        let back = Vocabulary::from_tokens(&TextLimits::default(), &v.tokens()[2..]);
        assert!(matches!(
            back,
            Err(TextError::VocabularyTokenTooLong { .. })
        ));
    }

    #[test]
    fn character_split_ngram_longer_than_input_is_accepted() {
        // 空白もトークンになり k-gram は区切りを足すため、入力より長くなりうる。
        let l = TextLimits::default()
            .with_max_input_bytes(5)
            .unwrap_or_default();
        let r = adapt(
            &["a b c"],
            Standardize::None,
            Split::Character,
            Some(5),
            None,
            &l,
        );
        assert!(r.is_ok());
    }

    #[test]
    fn frequency_increment_overflow() {
        let mut c = u64::MAX - 1;
        assert!(increment(&mut c).is_ok());
        assert_eq!(c, u64::MAX);
        assert!(matches!(
            increment(&mut c),
            Err(TextError::FrequencyOverflow)
        ));
        assert_eq!(c, u64::MAX);
    }

    #[test]
    fn empty_and_blank_corpus_give_reserved_only() {
        assert_eq!(toks(&[], None).len(), 2);
        assert_eq!(toks(&["   ", "\t"], None).len(), 2);
    }

    #[test]
    fn lookup_matches_token_index() {
        let v = match run(&["b a c", "a b", "a"], None) {
            Ok(v) => v,
            Err(e) => panic!("{e:?}"),
        };
        for (i, t) in v.tokens().iter().enumerate() {
            assert_eq!(usize::try_from(v.lookup(t)).ok(), Some(i), "{t}");
        }
    }

    #[test]
    fn no_corpus_text_in_debug_or_errors() {
        let marker = "SECRET_MARKER_TOKEN";
        let v = run(&[marker], None);
        assert!(matches!(&v, Ok(v) if !format!("{v:?}").contains(marker)));
        let l = TextLimits::default()
            .with_max_input_bytes(3)
            .unwrap_or_default();
        let e = adapt(
            &[marker],
            Standardize::None,
            Split::Whitespace,
            None,
            None,
            &l,
        );
        match e {
            Err(e) => {
                assert!(!e.to_string().contains(marker));
                assert!(!format!("{e:?}").contains(marker));
            }
            Ok(_) => panic!("上限超過が Err にならない"),
        }
    }
}
