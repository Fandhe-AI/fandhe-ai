//! 文字列バッチを `[B, L]` の `Tensor<i32>` へ変換する内部実装（語彙 lookup 型の最終段）。
//!
//! 役割: 標準化（`standardize`）→ 分割（`split`）→ 任意の n-gram（`ngram`）→
//! 語彙 lookup（`vocab`）→ 切り詰め・0 埋めを 1 関数にまとめる。呼び出し元は
//! 結合テスト・embedding 結線（#2903）と、公開時の facade（設計記録 §11 の 9）。
//! 定義の正は `docs/facade-text-vectorization-design.md` §3・§8。
//!
//! 検査順（確保より先）: 設定値（n・出力長）→ 入力（件数・要素ごとのバイト数）→
//! 出力要素数（`B × L` の `checked_mul`・要素数上限・`isize::MAX` バイト）→ 一括確保。
//! `output_sequence_length` が `None` のときは 2 pass にする。pass 1 は行ごとに
//! 長さだけを数え（id 列も n-gram 文字列も確保しない）、pass 2 で書き込む。
//! 1 pass で `Vec<Vec<i32>>` に貯めると、出力要素数の検査より前に中間データが
//! 入力バイト数 × n 規模へ膨らむため（OWASP A04）。
//!
//! 実装上の選択（設計記録の承認事項ではない。公開承認の際に確認を求める）:
//! - `B = 0` は形状 `[0, L]`（`None` なら `L = 0`）。設定値の検査は先に行う
//! - `Split::None` と空入力 `""` は特別扱いせず、トークン `""` を lookup して id 0
//!   （パディングと区別できない）
//! - 切り詰めは先頭 `L` 個を残し末尾を捨てる。`Some(0)` は `[B, 0]`（`Err` にしない）
//! - `max_output_sequence_length` は設定値 `Some(L)` にだけ適用し、`None` で導いた
//!   `L` は出力要素数の検査で抑える
//!
//! 数値一致は整数の完全一致で判定する（浮動小数点向けの統一複合判定は適用しない）。
//! エラーは入力文字列を保持しない（OWASP A09）。内部型で `fandhe_ai::text` へは公開しない。

use super::error::TextError;
use super::limits::TextLimits;
use super::ngram::{ngram_count, ngrams};
use super::split::{Split, split};
use super::standardize::{Standardize, standardize};
use super::vocab::{PADDING_ID, Vocabulary};

/// transform の設定（内部型）。公開の `TextVectorizationConfig` から組み立てられる。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct TransformOptions {
    /// 標準化の方式。
    pub(crate) standardize: Standardize,
    /// 分割の方式。
    pub(crate) split: Split,
    /// `Some(n)` なら 1〜n-gram を作る。`None` は n-gram なし。
    pub(crate) ngrams: Option<usize>,
    /// `Some(L)` なら `L` へ切り詰め／0 埋め。`None` はバッチ内の最長に合わせる。
    pub(crate) output_sequence_length: Option<usize>,
}

/// 1 行ぶんの標準化済みトークン数を数える（確保しない）。
fn token_count(input: &str, opts: &TransformOptions) -> usize {
    let std = standardize(input, opts.standardize);
    split(&std, opts.split).count()
}

/// 1 行を id 列へ変換して `dst`（長さ `L`、0 初期化済み）の先頭から書く。
fn fill_row(
    input: &str,
    vocab: &Vocabulary,
    opts: &TransformOptions,
    limits: &TextLimits,
    dst: &mut [i32],
) -> Result<(), TextError> {
    let std = standardize(input, opts.standardize);
    let toks: Vec<&str> = split(&std, opts.split).collect();
    match opts.ngrams {
        Some(n) => {
            let grams = ngrams(&toks, n, limits)?;
            for (slot, g) in dst.iter_mut().zip(grams.iter()) {
                *slot = vocab.lookup(g);
            }
        }
        None => {
            for (slot, t) in dst.iter_mut().zip(toks.iter()) {
                *slot = vocab.lookup(t);
            }
        }
    }
    Ok(())
}

/// 文字列バッチを `[B, L]` の `Tensor<i32>` へ変換する。
pub(crate) fn transform<S: AsRef<str>>(
    vocab: &Vocabulary,
    opts: &TransformOptions,
    limits: &TextLimits,
    inputs: &[S],
) -> Result<crate::Tensor<i32>, TextError> {
    // 1. 設定値の検査（B = 0 でも行う）。
    if let Some(n) = opts.ngrams {
        limits.check_ngrams(n)?;
    }
    if let Some(l) = opts.output_sequence_length {
        limits.check_output_sequence_length(l)?;
    }
    // 2. 入力の検査。
    limits.check_inputs(inputs)?;

    // 3. 出力長 L を確保なしで決める。
    let len = match opts.output_sequence_length {
        Some(l) => l,
        None => {
            let mut max = 0usize;
            for s in inputs {
                let t = token_count(s.as_ref(), opts);
                let row = match opts.ngrams {
                    Some(n) => ngram_count(t, n, limits)?,
                    None => t,
                };
                max = max.max(row);
            }
            max
        }
    };

    // 4. 出力要素数の検査（確保より先）→ 5. 一括確保。
    let batch = inputs.len();
    let elements = limits.check_output_elements(batch, len)?;
    let mut data = vec![PADDING_ID; elements];

    // 6. pass 2。`chunks_exact_mut(0)` は panic するため `len == 0` は飛ばす。
    if len > 0 {
        for (row, s) in data.chunks_exact_mut(len).zip(inputs.iter()) {
            fill_row(s.as_ref(), vocab, opts, limits, row)?;
        }
    }
    crate::Tensor::new(data, &[batch, len]).map_err(TextError::Shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &str = "SECRET_MARKER_TOKEN";

    fn vocab(words: &[&str]) -> Vocabulary {
        let Ok(v) = Vocabulary::from_tokens(&TextLimits::default(), words) else {
            panic!("vocab must build");
        };
        v
    }

    fn run(
        v: &Vocabulary,
        opts: &TransformOptions,
        limits: &TextLimits,
        inputs: &[&str],
    ) -> Result<(Vec<usize>, Vec<i32>), TextError> {
        transform(v, opts, limits, inputs).map(|t| (t.shape().to_vec(), t.host_slice().to_vec()))
    }

    fn base() -> Vocabulary {
        vocab(&["cat", "dog", "the"])
    }

    fn go(opts: TransformOptions, inputs: &[&str]) -> (Vec<usize>, Vec<i32>) {
        match run(&base(), &opts, &TextLimits::default(), inputs) {
            Ok(r) => r,
            Err(e) => panic!("transform failed: {e}"),
        }
    }

    #[test]
    fn basic_pads_to_longest() {
        let (s, d) = go(TransformOptions::default(), &["the cat", "dog"]);
        assert_eq!(s, vec![2, 2]);
        assert_eq!(d, vec![4, 2, 3, 0]);
    }

    #[test]
    fn truncate_and_pad_with_fixed_length() {
        let o = TransformOptions {
            output_sequence_length: Some(2),
            ..Default::default()
        };
        let (s, d) = go(o, &["the cat dog", "dog"]);
        assert_eq!(s, vec![2, 2]);
        assert_eq!(d, vec![4, 2, 3, 0]);
        let o = TransformOptions {
            output_sequence_length: Some(4),
            ..Default::default()
        };
        let (s, d) = go(o, &["the cat"]);
        assert_eq!((s, d), (vec![1, 4], vec![4, 2, 0, 0]));
    }

    #[test]
    fn zero_length() {
        let o = TransformOptions {
            output_sequence_length: Some(0),
            ..Default::default()
        };
        let (s, d) = go(o, &["a", "b"]);
        assert_eq!((s, d.len()), (vec![2, 0], 0));
        let (s, d) = go(TransformOptions::default(), &["", "  "]);
        assert_eq!((s, d.len()), (vec![2, 0], 0));
    }

    #[test]
    fn empty_batch() {
        let (s, d) = go(TransformOptions::default(), &[]);
        assert_eq!((s, d.len()), (vec![0, 0], 0));
        let o = TransformOptions {
            output_sequence_length: Some(3),
            ..Default::default()
        };
        let (s, d) = go(o, &[]);
        assert_eq!((s, d.len()), (vec![0, 3], 0));
    }

    #[test]
    fn empty_batch_still_validates_config() {
        let l = TextLimits::default();
        let o = TransformOptions {
            ngrams: Some(0),
            ..Default::default()
        };
        assert!(matches!(
            run(&base(), &o, &l, &[]),
            Err(TextError::InvalidNgrams { .. })
        ));
        let o = TransformOptions {
            output_sequence_length: Some(usize::MAX),
            ..Default::default()
        };
        assert!(matches!(
            run(&base(), &o, &l, &[]),
            Err(TextError::OutputSequenceLengthTooLarge { .. })
        ));
    }

    #[test]
    fn oov_and_standardize_before_lookup() {
        let (_, d) = go(TransformOptions::default(), &["Cat! zebra"]);
        assert_eq!(d, vec![2, 1]);
        let o = TransformOptions {
            standardize: Standardize::None,
            ..Default::default()
        };
        let (_, d) = go(o, &["Cat!"]);
        assert_eq!(d, vec![1]);
    }

    #[test]
    fn split_modes() {
        let v = vocab(&["é", "a", " "]);
        let o = TransformOptions {
            split: Split::Character,
            standardize: Standardize::None,
            ..Default::default()
        };
        let Ok((s, d)) = run(&v, &o, &TextLimits::default(), &["éa b"]) else {
            panic!("must succeed");
        };
        assert_eq!((s, d), (vec![1, 4], vec![2, 3, 4, 1]));
        let o = TransformOptions {
            split: Split::None,
            ..Default::default()
        };
        let (s, d) = go(o, &[""]);
        assert_eq!((s, d), (vec![1, 1], vec![0]));
    }

    #[test]
    fn ngrams_order_and_lookup() {
        let v = vocab(&["a", "b", "a b"]);
        let o = TransformOptions {
            ngrams: Some(2),
            ..Default::default()
        };
        let Ok((s, d)) = run(&v, &o, &TextLimits::default(), &["a b", "a"]) else {
            panic!("must succeed");
        };
        assert_eq!(s, vec![2, 3]);
        assert_eq!(d, vec![2, 3, 4, 2, 0, 0]);
    }

    #[test]
    fn output_too_large_both_paths() {
        let Ok(l) = TextLimits::default().with_max_output_elements(3) else {
            panic!("limit");
        };
        let none = TransformOptions::default();
        assert!(matches!(
            run(&base(), &none, &l, &["a b", "a b"]),
            Err(TextError::OutputTooLarge {
                elements: 4,
                max: 3
            })
        ));
        let some = TransformOptions {
            output_sequence_length: Some(2),
            ..Default::default()
        };
        assert!(matches!(
            run(&base(), &some, &l, &["a", "b"]),
            Err(TextError::OutputTooLarge {
                elements: 4,
                max: 3
            })
        ));
        assert!(run(&base(), &some, &l, &["a"]).is_ok());
    }

    #[test]
    fn input_limit_errors() {
        let Ok(l) = TextLimits::default()
            .with_max_batch(2)
            .and_then(|l| l.with_max_input_bytes(3))
        else {
            panic!("limit");
        };
        let o = TransformOptions::default();
        assert!(matches!(
            run(&base(), &o, &l, &["a", "b", "c"]),
            Err(TextError::BatchTooLarge { .. })
        ));
        assert!(matches!(
            run(&base(), &o, &l, &["a", "abcd"]),
            Err(TextError::InputTooLong { index: 1, .. })
        ));
    }

    #[test]
    fn config_errors_precede_input_errors() {
        let Ok(l) = TextLimits::default().with_max_batch(1) else {
            panic!("limit");
        };
        let o = TransformOptions {
            ngrams: Some(9),
            ..Default::default()
        };
        assert!(matches!(
            run(&base(), &o, &l, &["a", "b"]),
            Err(TextError::InvalidNgrams { .. })
        ));
        let o = TransformOptions::default();
        assert!(matches!(
            run(&base(), &o, &l, &["a", "b"]),
            Err(TextError::BatchTooLarge { .. })
        ));
    }

    #[test]
    fn errors_do_not_leak_input() {
        let Ok(l) = TextLimits::default().with_max_batch(1) else {
            panic!("limit");
        };
        let inputs = [SECRET, SECRET];
        let cases = [
            TransformOptions::default(),
            TransformOptions {
                ngrams: Some(0),
                ..Default::default()
            },
            TransformOptions {
                output_sequence_length: Some(usize::MAX),
                ..Default::default()
            },
        ];
        for o in cases {
            let Err(e) = run(&base(), &o, &l, &inputs) else {
                panic!("must fail");
            };
            assert!(!format!("{e}").contains(SECRET));
            assert!(!format!("{e:?}").contains(SECRET));
        }
        let Ok(l2) = TextLimits::default().with_max_output_elements(1) else {
            panic!("limit");
        };
        let Err(e) = run(
            &base(),
            &TransformOptions::default(),
            &l2,
            &[SECRET, SECRET],
        ) else {
            panic!("must fail");
        };
        assert!(!format!("{e}").contains(SECRET) && !format!("{e:?}").contains(SECRET));
    }
}
