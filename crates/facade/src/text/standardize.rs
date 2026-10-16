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
//! 論点 3 のうち ASCII 部分は承認済み（`docs/facade-text-vectorization-design.md`
//! §16.7）。Unicode 小文字化は #3089 で opt-in の `UnicodeLower` として追加した
//! （設計記録 §18。既定と既存 4 値は不変）。
//!
//! 上限検査（`TextLimits`）は呼び出し側が入口で先に行う。ASCII 系 4 値は出力が
//! 入力以下のため入口の上限が標準化後も保たれる。`UnicodeLower` は最大 3/2 倍に
//! 伸びうる（`İ` 2 byte → 3 byte）ため、標準化後のバイト数を数える pass で
//! `max_input_bytes` と比べ、合格後にだけバッファを確保する（fail-closed。
//! 超過は既存の `TextError::InputTooLong`、`len` は標準化後のバイト数）。
//! エラーは入力文字列を保持しない。
//! 公開型 `Standardize` は `fandhe_ai::text` へ再エクスポートされる（§16.2）。
//! 自由関数 `standardize` は非公開のまま。

use std::borrow::Cow;

use super::error::TextError;
use super::limits::TextLimits;

/// 標準化の方式。既定と既存 4 値の小文字化・句読点除去は ASCII に限る。
///
/// 非 ASCII 文字（`À`・全角句読点等）は既存 4 値では小文字化も除去もされない。
/// Unicode の小文字化が要る場合は `UnicodeLower` を opt-in で選ぶ。
/// `#[non_exhaustive]` のため variant の追加は非破壊。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Standardize {
    /// 何もしない。
    None,
    /// ASCII 大文字（`A`〜`Z`）だけを小文字化する（`to_ascii_lowercase` 相当）。
    /// 非 ASCII の大文字は変えない。
    Lower,
    /// ASCII 句読点 32 文字（`char::is_ascii_punctuation`）だけを削除する。空白へは
    /// 置換しない。非 ASCII の句読点は残る。
    StripPunctuation,
    /// ASCII の小文字化をしてから ASCII 句読点を削除する（既定）。
    #[default]
    LowerAndStripPunctuation,
    /// 各 `char` に `char::to_lowercase`（Unicode の単純な小文字化）を適用する。
    /// 句読点は除去しない。文脈非依存のため語末の `Σ` も `σ`（`ς` にならない）。
    /// `ß` は不変・`ẞ` は `ß`・`İ` は `i` + U+0307（バイト長・文字数が増えうる）。
    /// 標準化後のバイト数にも `TextLimits` の入力上限を適用する（超過は
    /// `InputTooLong`）。Unicode 正規化は行わない。
    UnicodeLower,
}

/// `input` を `mode` に従って標準化する。変更が不要なら借用のまま返す。
///
/// 呼び出し元は adapt・transform。`UnicodeLower` のみ標準化後の長さが入力を超え
/// うるため、数える pass で `limits` と比べてから確保する（失敗時は `index` 番目の
/// 入力として `InputTooLong`）。ASCII 系 4 値は `Err` を返さない。
pub(crate) fn standardize<'a>(
    input: &'a str,
    mode: Standardize,
    limits: &TextLimits,
    index: usize,
) -> Result<Cow<'a, str>, TextError> {
    let (lower, strip) = match mode {
        Standardize::None => return Ok(Cow::Borrowed(input)),
        Standardize::UnicodeLower => return unicode_lower(input, limits, index),
        Standardize::Lower => (true, false),
        Standardize::StripPunctuation => (false, true),
        Standardize::LowerAndStripPunctuation => (true, true),
    };
    let needs_change = input
        .chars()
        .any(|c| (lower && c.is_ascii_uppercase()) || (strip && c.is_ascii_punctuation()));
    if !needs_change {
        return Ok(Cow::Borrowed(input));
    }
    // 出力は入力以下のバイト長（小文字化は同長・句読点除去は減るのみ）。
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        if strip && c.is_ascii_punctuation() {
            continue;
        }
        out.push(if lower { c.to_ascii_lowercase() } else { c });
    }
    Ok(Cow::Owned(out))
}

/// `Standardize::UnicodeLower` の本体。文字ごとの `char::to_lowercase` を使う
/// （`str::to_lowercase` は語末シグマを文脈依存で扱い、事前に数えた長さと一致
/// しなくなるため使わない）。
fn unicode_lower<'a>(
    input: &'a str,
    limits: &TextLimits,
    index: usize,
) -> Result<Cow<'a, str>, TextError> {
    let changes = |c: char| {
        let mut it = c.to_lowercase();
        !(it.next() == Some(c) && it.next().is_none())
    };
    if !input.chars().any(changes) {
        return Ok(Cow::Borrowed(input));
    }
    // 数える pass（確保なし）。加算 overflow は飽和させて必ず上限超過にする。
    let mut len = 0usize;
    for c in input.chars() {
        for l in c.to_lowercase() {
            len = len.saturating_add(l.len_utf8());
        }
    }
    limits.check_input_len(index, len)?;
    let mut out = String::with_capacity(len);
    for c in input.chars() {
        out.extend(c.to_lowercase());
    }
    Ok(Cow::Owned(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODES: [Standardize; 5] = [
        Standardize::UnicodeLower,
        Standardize::None,
        Standardize::Lower,
        Standardize::StripPunctuation,
        Standardize::LowerAndStripPunctuation,
    ];

    fn std_ok(s: &str, m: Standardize) -> Cow<'_, str> {
        let Ok(c) = standardize(s, m, &TextLimits::default(), 0) else {
            panic!("default limits must accept");
        };
        c
    }

    fn run(s: &str, m: Standardize) -> String {
        std_ok(s, m).into_owned()
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
            std_ok("Abc!", Standardize::None),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            std_ok("abc", Standardize::Lower),
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
    fn unicode_小文字化の代表値() {
        let u = |s| run(s, Standardize::UnicodeLower);
        assert_eq!(u("ß"), "ß");
        assert_eq!(u("ẞ"), "ß");
        assert_eq!(u("İ"), "i\u{307}");
        assert_eq!(u("ÉSSAİ"), "éssai\u{307}");
        assert_eq!(u("Hello, World!"), "hello, world!");
        // 文脈非依存: 語末の Σ も σ（str::to_lowercase は ς にする）。
        assert_eq!(u("ΟΔΟΣ"), "οδοσ");
        assert_eq!("ΟΔΟΣ".to_lowercase(), "οδος");
    }

    #[test]
    fn unicode_小文字化は変更不要なら借用() {
        for s in ["", " \t", "abc", "ß", "日本語"] {
            assert!(matches!(
                std_ok(s, Standardize::UnicodeLower),
                Cow::Borrowed(_)
            ));
        }
    }

    #[test]
    fn unicode_小文字化は冪等() {
        for s in all_strings() {
            let o = run(&s, Standardize::UnicodeLower);
            assert_eq!(run(&o, Standardize::UnicodeLower), o);
        }
    }

    #[test]
    fn unicode_小文字化の上限は標準化後のバイト数で境界検査() {
        let lim = |n: usize| {
            let Ok(l) = TextLimits::default().with_max_input_bytes(n) else {
                panic!("limit must build");
            };
            l
        };
        let m = Standardize::UnicodeLower;
        assert!(standardize("İ", m, &lim(3), 0).is_ok());
        assert!(matches!(
            standardize("İ", m, &lim(2), 7),
            Err(TextError::InputTooLong {
                index: 7,
                len: 3,
                max: 2
            })
        ));
    }

    #[test]
    fn unicode_小文字化の膨張率は_3_2_以下() {
        for c in char::MIN..=char::MAX {
            let o: usize = c.to_lowercase().map(char::len_utf8).sum();
            assert!(o * 2 <= c.len_utf8() * 3, "{c:?}");
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
