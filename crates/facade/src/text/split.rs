//! 文字列の分割（空白・文字・なし）の内部実装。
//!
//! 役割: 標準化済みの文字列をトークン列へ分割する。呼び出し元は n-gram（#2899）・
//! adapt（#2901）・transform（#2902）。定義の正は
//! `docs/facade-text-vectorization-design.md` §3。
//!
//! - `Whitespace`: ASCII 空白（SP・TAB・LF・FF・CR）で区切る。VT・NBSP・全角空白は
//!   区切りにならない。連続空白で空トークンは出ない
//! - `Character`: Unicode スカラー値（`char`）単位。書記素クラスタ単位ではなく、
//!   結合文字・ZWJ 絵文字は複数トークンに分かれる
//! - `None`: 入力全体を常に 1 トークンにする。空入力でも空文字列の 1 トークンを
//!   返す（設計記録に空入力の特記がないための素直な解釈）。空文字列はパディング
//!   予約語と一致するため、lookup 段（#2900・#2902）での扱いは後続で決める
//!
//! 戻り値は入力を借用するイテレータでトークンごとの確保をしない。上限検査
//! （`TextLimits`）は呼び出し側が入口で先に行う前提（設計記録 §8）。論点 3 は
//! 未承認で、本実装は設計記録 §3 の推奨定義に従う。失敗しない純関数。
//! derive 一式は内部実装上の選択で、公開時の derive は承認時の決定事項。

use std::str::{CharIndices, SplitAsciiWhitespace};

/// 分割の方式。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Split {
    /// 分割しない（入力全体を 1 トークン）。
    None,
    /// ASCII 空白で分割する（既定）。
    #[default]
    Whitespace,
    /// Unicode スカラー値単位で分割する。
    Character,
}

/// `split` が返すトークンのイテレータ（入力の部分スライスを返す）。
pub(crate) enum SplitIter<'a> {
    /// `Split::None`: 未返却なら全体を 1 回だけ返す。
    Whole(Option<&'a str>),
    /// `Split::Whitespace`。
    Whitespace(SplitAsciiWhitespace<'a>),
    /// `Split::Character`: 元文字列とバイト位置付きの文字列走査。
    Character(&'a str, CharIndices<'a>),
}

impl<'a> Iterator for SplitIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        match self {
            SplitIter::Whole(rest) => rest.take(),
            SplitIter::Whitespace(it) => it.next(),
            SplitIter::Character(src, it) => {
                let (i, c) = it.next()?;
                // `char_indices` の境界は常に文字境界で、`get` は失敗しない。
                src.get(i..i + c.len_utf8())
            }
        }
    }
}

/// `input` を `mode` に従ってトークンへ分割する。
pub(crate) fn split(input: &str, mode: Split) -> SplitIter<'_> {
    match mode {
        Split::None => SplitIter::Whole(Some(input)),
        Split::Whitespace => SplitIter::Whitespace(input.split_ascii_whitespace()),
        Split::Character => SplitIter::Character(input, input.char_indices()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str, m: Split) -> Vec<&str> {
        split(s, m).collect()
    }

    fn is_ws(c: char) -> bool {
        c.is_ascii_whitespace()
    }

    fn all_strings() -> Vec<String> {
        let alphabet = ['a', 'é', ' ', '\t', '\n', '\u{3000}', '\u{A0}', '日'];
        let mut all = vec![String::new()];
        let mut frontier = vec![String::new()];
        for _ in 0..4 {
            let mut next = Vec::new();
            for s in &frontier {
                for c in alphabet {
                    let mut t = s.clone();
                    t.push(c);
                    next.push(t);
                }
            }
            all.extend(next.iter().cloned());
            frontier = next;
        }
        all
    }

    #[test]
    fn 空白分割() {
        assert_eq!(toks("  a  b\tc\n", Split::Whitespace), ["a", "b", "c"]);
        assert!(toks("", Split::Whitespace).is_empty());
        assert!(toks(" \t\n", Split::Whitespace).is_empty());
        assert_eq!(toks("a\u{3000}b", Split::Whitespace), ["a\u{3000}b"]);
        assert_eq!(toks("a\u{00A0}b", Split::Whitespace), ["a\u{00A0}b"]);
        assert_eq!(toks("a\u{0B}b", Split::Whitespace), ["a\u{0B}b"]);
        assert_eq!(toks("日本 語", Split::Whitespace), ["日本", "語"]);
    }

    #[test]
    fn 文字分割() {
        assert_eq!(toks("aé日", Split::Character), ["a", "é", "日"]);
        assert!(toks("", Split::Character).is_empty());
        assert_eq!(toks("e\u{301}", Split::Character).len(), 2);
        assert_eq!(toks("👨\u{200D}👩", Split::Character).len(), 3);
        assert_eq!(toks("a b", Split::Character), ["a", " ", "b"]);
    }

    #[test]
    fn 分割なし() {
        assert_eq!(toks("a b", Split::None), ["a b"]);
        assert_eq!(toks("", Split::None), [""]);
        assert_eq!(toks(" \t", Split::None), [" \t"]);
    }

    #[test]
    fn 長さ不変条件() {
        for s in all_strings() {
            let ch = toks(&s, Split::Character);
            assert_eq!(ch.len(), s.chars().count());
            assert!(ch.iter().all(|t| t.chars().count() == 1));
            assert_eq!(ch.concat(), s);
            assert_eq!(ch.iter().map(|t| t.len()).sum::<usize>(), s.len());

            let ws = toks(&s, Split::Whitespace);
            assert!(ws.iter().all(|t| !t.is_empty() && !t.contains(is_ws)));
            let stripped: String = s.chars().filter(|c| !is_ws(*c)).collect();
            assert_eq!(ws.concat(), stripped);
            assert!(ws.iter().map(|t| t.len()).sum::<usize>() <= s.len());

            assert_eq!(toks(&s, Split::None), [s.as_str()]);
        }
    }

    #[test]
    fn 既定値は空白() {
        assert_eq!(Split::default(), Split::Whitespace);
    }
}
