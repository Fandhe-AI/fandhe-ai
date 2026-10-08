//! 文字列の標準化（小文字化・句読点除去）の内部実装。
//!
//! 役割: 語彙 lookup 型テキスト変換の前段として、入力文字列を ASCII の範囲で
//! 正規化する。呼び出し元は n-gram（#2899）・adapt（#2901）・transform（#2902）。
//! 定義の正は `docs/facade-text-vectorization-design.md` §3。
//!
//! - 小文字化は `str::to_ascii_lowercase` 相当（ASCII 大文字のみ。非 ASCII は不変）
//! - 句読点除去は `char::is_ascii_punctuation` の 32 文字を削除する（空白へは置換しない）
//! - `LowerAndStripPunctuation` は小文字化 → 句読点除去の順で適用する
//!
//! 論点 3（ASCII に限るか・空白の定義）は `docs/compat-api-scope.md` 上で未承認。
//! 本実装は設計記録 §3 の推奨定義に従う内部実装であり、承認結論が Unicode 側へ
//! 変わった場合は本モジュールの差し替えで吸収する。
//!
//! 上限検査（`TextLimits`）は呼び出し側が入口で標準化・確保より先に行う前提
//! （設計記録 §8）。出力バイト長は入力以下のため入口の上限は標準化後も保たれる。
//! 失敗しない純関数であり、入力文字列をエラーへ出す経路を持たない。
//! derive 一式は内部実装上の選択で、公開時の derive は承認時の決定事項。

use std::borrow::Cow;

/// 標準化の方式。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Standardize {
    /// 何もしない。
    None,
    /// ASCII 大文字のみ小文字化する。
    Lower,
    /// ASCII 句読点を削除する。
    StripPunctuation,
    /// 小文字化してから句読点を削除する（既定）。
    #[default]
    LowerAndStripPunctuation,
}

/// `input` を `mode` に従って標準化する。変更が不要なら借用のまま返す。
pub(crate) fn standardize(input: &str, mode: Standardize) -> Cow<'_, str> {
    let (lower, strip) = match mode {
        Standardize::None => return Cow::Borrowed(input),
        Standardize::Lower => (true, false),
        Standardize::StripPunctuation => (false, true),
        Standardize::LowerAndStripPunctuation => (true, true),
    };
    let needs_change = input
        .chars()
        .any(|c| (lower && c.is_ascii_uppercase()) || (strip && c.is_ascii_punctuation()));
    if !needs_change {
        return Cow::Borrowed(input);
    }
    // 出力は入力以下のバイト長（小文字化は同長・句読点除去は減るのみ）。
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        if strip && c.is_ascii_punctuation() {
            continue;
        }
        out.push(if lower { c.to_ascii_lowercase() } else { c });
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODES: [Standardize; 4] = [
        Standardize::None,
        Standardize::Lower,
        Standardize::StripPunctuation,
        Standardize::LowerAndStripPunctuation,
    ];

    fn run(s: &str, m: Standardize) -> String {
        standardize(s, m).into_owned()
    }

    fn punct_count(s: &str) -> usize {
        s.chars().filter(|c| c.is_ascii_punctuation()).count()
    }

    /// 長さ 0〜4 の全文字列を列挙する。
    fn all_strings() -> Vec<String> {
        let alphabet = ['a', 'A', 'Z', ' ', '\t', '.', '!', 'é', 'あ', '\u{3000}'];
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
    fn 代表入力の期待値() {
        let s = "Hello, World!";
        assert_eq!(run(s, Standardize::None), s);
        assert_eq!(run(s, Standardize::Lower), "hello, world!");
        assert_eq!(run(s, Standardize::StripPunctuation), "Hello World");
        assert_eq!(run(s, Standardize::LowerAndStripPunctuation), "hello world");
    }

    #[test]
    fn 空入力と空白のみは不変() {
        for m in MODES {
            assert_eq!(run("", m), "");
            assert_eq!(run(" \t\n", m), " \t\n");
        }
        assert_eq!(run("!?.,", Standardize::StripPunctuation), "");
        assert_eq!(run("!?.,", Standardize::LowerAndStripPunctuation), "");
    }

    #[test]
    fn ascii_限定の帰結を固定する() {
        assert_eq!(run("ÉSSAİ ß", Standardize::Lower), "Éssaİ ß");
        assert_eq!(run("！、。", Standardize::StripPunctuation), "！、。");
        assert_eq!(
            run("ＡＢＣ", Standardize::LowerAndStripPunctuation),
            "ＡＢＣ"
        );
        assert_eq!(run("日本語ABC", Standardize::Lower), "日本語abc");
    }

    #[test]
    fn 変更不要なら借用を返す() {
        assert!(matches!(
            standardize("Abc!", Standardize::None),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            standardize("abc", Standardize::Lower),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn 長さ不変条件_lower() {
        for s in all_strings() {
            let o = run(&s, Standardize::Lower);
            assert_eq!(o.len(), s.len());
            assert_eq!(o.chars().count(), s.chars().count());
            assert_eq!(run(&o, Standardize::Lower), o);
        }
    }

    #[test]
    fn 長さ不変条件_strip() {
        for s in all_strings() {
            let o = run(&s, Standardize::StripPunctuation);
            assert_eq!(o.len(), s.len() - punct_count(&s));
            assert_eq!(punct_count(&o), 0);
            assert_eq!(run(&o, Standardize::StripPunctuation), o);
        }
    }

    #[test]
    fn 長さ不変条件_両方() {
        for s in all_strings() {
            let both = run(&s, Standardize::LowerAndStripPunctuation);
            assert!(both.len() <= s.len());
            let a = run(&run(&s, Standardize::Lower), Standardize::StripPunctuation);
            let b = run(&run(&s, Standardize::StripPunctuation), Standardize::Lower);
            assert_eq!(a, b);
            assert_eq!(both, a);
        }
    }

    #[test]
    fn 既定値は両方() {
        assert_eq!(
            Standardize::default(),
            Standardize::LowerAndStripPunctuation
        );
    }
}
