//! 分割済みトークン列から 1〜n-gram を作る内部実装。
//!
//! 役割: `split` が返したトークン列を、語彙 lookup の前段で使う n-gram 列へ
//! 展開する。呼び出し元は adapt（#2901）・transform（#2902）。定義の正は
//! `docs/facade-text-vectorization-design.md` §3・§8。
//!
//! - 出力順: 全 1-gram → 全 2-gram → … → 全 n-gram。各 k-gram は連続する k 個の
//!   トークンを半角空白 1 つで連結する。1-gram は入力の借用で確保しない
//! - 連結文字は分割方式によらず固定。`Split::Character` では空白自体がトークンに
//!   なるため、連結結果に空白が続くことがある（正規化しない）
//! - 検査順: `n` の検査 → 生成数 `tokens × n` の `checked_mul` → 確保（検査が先）。
//!   不正な `n` は空入力でも `Err`。トークン数 < n や空入力は `Err` にしない
//!
//! 失敗は型付き `TextError`（入力文字列は保持しない）。論点 4（Keras との順序の
//! 一致）は契約にせず、同値とは主張しない。内部実装のため `pub(crate)`
//! （設計記録 §16.2）。

use std::borrow::Cow;

use super::error::TextError;
use super::limits::TextLimits;

/// 生成数の上限見積り `tokens × n` を `checked_mul` で求める（設計記録 §8）。
fn ngram_count_upper_bound(tokens: usize, n: usize) -> Result<usize, TextError> {
    tokens
        .checked_mul(n)
        .ok_or(TextError::NgramCountOverflow { tokens, n })
}

/// `tokens` 個のトークンから作る 1〜`n`-gram の正確な件数を、文字列を確保せずに求める。
///
/// transform（#2902）が出力長を確保前に決めるために呼ぶ。`ngrams` の件数と一致する。
pub(crate) fn ngram_count(
    tokens: usize,
    n: usize,
    limits: &TextLimits,
) -> Result<usize, TextError> {
    limits.check_ngrams(n)?;
    ngram_count_upper_bound(tokens, n)?;
    let mut total: usize = 0;
    for k in 1..=n.min(tokens) {
        total = total
            .checked_add(tokens - k + 1)
            .ok_or(TextError::NgramCountOverflow { tokens, n })?;
    }
    Ok(total)
}

/// `tokens` から 1〜`n`-gram を設計記録 §3 の順で作る。
pub(crate) fn ngrams<'a>(
    tokens: &[&'a str],
    n: usize,
    limits: &TextLimits,
) -> Result<Vec<Cow<'a, str>>, TextError> {
    limits.check_ngrams(n)?;
    let cap = ngram_count_upper_bound(tokens.len(), n)?;

    // 正確な件数は上限見積り以下。過大確保を避けるため小さい方で確保する。
    let t = tokens.len();
    let exact = ngram_count(t, n, limits)?;
    let mut out: Vec<Cow<'a, str>> = Vec::with_capacity(exact.min(cap));

    // `check_ngrams` により k >= 1（`windows(0)` は panic するため前提として必要）。
    // `windows(k)` は k > t なら空で、手書きの減算のような underflow が起きない。
    for k in 1..=n {
        for w in tokens.windows(k) {
            if k == 1 {
                out.push(Cow::Borrowed(w[0]));
            } else {
                out.push(Cow::Owned(w.join(" ")));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::split::{Split, split};

    fn run(tokens: &[&str], n: usize) -> Result<Vec<String>, TextError> {
        ngrams(tokens, n, &TextLimits::default())
            .map(|v| v.into_iter().map(Cow::into_owned).collect())
    }

    fn ok(tokens: &[&str], n: usize) -> Vec<String> {
        run(tokens, n).unwrap_or_default()
    }

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn order_and_join() {
        assert_eq!(
            ok(&["a", "b", "c"], 3),
            strs(&["a", "b", "c", "a b", "b c", "a b c"])
        );
        assert_eq!(ok(&["a", "b"], 1), strs(&["a", "b"]));
    }

    #[test]
    fn fewer_tokens_than_n() {
        assert_eq!(ok(&["a", "b"], 3), strs(&["a", "b", "a b"]));
        assert_eq!(ok(&["a"], 8), strs(&["a"]));
    }

    #[test]
    fn empty_input() {
        assert!(matches!(run(&[], 1), Ok(v) if v.is_empty()));
        assert!(matches!(run(&[], 8), Ok(v) if v.is_empty()));
        assert!(matches!(
            run(&[], 0),
            Err(TextError::InvalidNgrams { n: 0, max: 8 })
        ));
    }

    #[test]
    fn n_boundary() {
        assert!(matches!(
            run(&["a"], 0),
            Err(TextError::InvalidNgrams { n: 0, max: 8 })
        ));
        assert!(matches!(
            run(&["a"], 9),
            Err(TextError::InvalidNgrams { n: 9, max: 8 })
        ));
        assert!(run(&["a"], 1).is_ok());
        assert!(run(&["a"], 8).is_ok());
    }

    #[test]
    fn lowered_limit() {
        let Ok(l) = TextLimits::default().with_max_ngrams(2) else {
            panic!("with_max_ngrams(2) must succeed");
        };
        assert!(matches!(
            ngrams(&["a", "b", "c"], 3, &l),
            Err(TextError::InvalidNgrams { n: 3, max: 2 })
        ));
        assert!(ngrams(&["a", "b", "c"], 2, &l).is_ok());
    }

    #[test]
    fn ngram_count_matches_ngrams() {
        let l = TextLimits::default();
        for t in 0..=10usize {
            let owned: Vec<String> = (0..t).map(|i| format!("w{i}")).collect();
            let tokens: Vec<&str> = owned.iter().map(String::as_str).collect();
            for n in 1..=8usize {
                let Ok(out) = ngrams(&tokens, n, &l) else {
                    panic!("ngrams must succeed");
                };
                assert!(matches!(ngram_count(t, n, &l), Ok(c) if c == out.len()));
            }
        }
        assert!(matches!(
            ngram_count(1, 0, &l),
            Err(TextError::InvalidNgrams { .. })
        ));
        assert!(matches!(
            ngram_count(usize::MAX, 2, &l),
            Err(TextError::NgramCountOverflow { .. })
        ));
    }

    #[test]
    fn count_upper_bound() {
        assert!(matches!(
            ngram_count_upper_bound(usize::MAX, 2),
            Err(TextError::NgramCountOverflow { n: 2, .. })
        ));
        assert!(matches!(
            ngram_count_upper_bound(usize::MAX, 1),
            Ok(usize::MAX)
        ));
        assert!(matches!(ngram_count_upper_bound(0, 8), Ok(0)));
        assert!(matches!(ngram_count_upper_bound(3, 3), Ok(9)));
    }

    #[test]
    fn count_identity_and_structure() {
        for t in 0..=10usize {
            let owned: Vec<String> = (0..t).map(|i| format!("w{i}")).collect();
            let tokens: Vec<&str> = owned.iter().map(String::as_str).collect();
            for n in 1..=8usize {
                let Ok(out) = ngrams(&tokens, n, &TextLimits::default()) else {
                    panic!("ngrams must succeed for t={t} n={n}");
                };
                let expected = if t >= n {
                    n * t - n * (n - 1) / 2
                } else {
                    t * (t + 1) / 2
                };
                assert_eq!(out.len(), expected, "t={t} n={n}");
                assert!(matches!(ngram_count_upper_bound(t, n), Ok(b) if out.len() <= b));
                // 1-gram は借用、k-gram は空白をちょうど k-1 個持つ。
                let mut idx = 0;
                for k in 1..=n.min(t) {
                    for _ in 0..(t - k + 1) {
                        let g = &out[idx];
                        assert_eq!(g.matches(' ').count(), k - 1);
                        assert_eq!(k == 1, matches!(g, Cow::Borrowed(_)));
                        idx += 1;
                    }
                }
            }
        }
    }

    #[test]
    fn multibyte() {
        assert_eq!(
            ok(&["日本", "語", "é"], 2),
            strs(&["日本", "語", "é", "日本 語", "語 é"])
        );
    }

    #[test]
    fn with_split() {
        let toks: Vec<&str> = split("a  b c", Split::Whitespace).collect();
        assert_eq!(ok(&toks, 2), strs(&["a", "b", "c", "a b", "b c"]));
        let toks: Vec<&str> = split("", Split::None).collect();
        assert_eq!(ok(&toks, 2), strs(&[""]));
    }
}
