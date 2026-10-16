//! 文字列バッチを `[B, W]` の `Tensor<f32>`（bag-of-words 系の特徴量）へ変換する内部実装。
//!
//! 役割: `TextVectorization` の出力モード系メソッド（`transform_multi_hot`・
//! `transform_count`・`transform_tf_idf`）の共通実装。標準化 → 分割 → 任意の n-gram →
//! 語彙 lookup の経路は `transform` と同じで、結果を id 列ではなく「列 i = id i」の
//! 行ベクトルへ数え上げる点だけが違う。呼び出し元は `vectorization`（公開メソッド）で、
//! 公開項目はここに置かない。契約の正は
//! `docs/facade-text-output-modes-string-lookup-decision.md` §3・§5。
//!
//! 検査順（確保より先にすべて）: n-gram の `n` → 入力（件数・要素ごとのバイト数）→
//! 事前走査（行ごとのトークン数だけを数える。中間の `Vec` も n-gram 文字列も作らない。
//! 1 行でも 2^24 を超えたら `TokenCountNotExact`）→ 出力要素数 `B × W`（`checked_mul`・
//! 要素数上限・`isize::MAX` バイト）→ 一括確保。事前走査で 1 行 2^24 以下と確定する
//! ため、f32 への `+= 1.0` は丸められず厳密（count の `==` 判定の根拠）。
//!
//! 列 0（パディング）は常に 0 とし、id 0 になるトークンは数えずに捨てる。`W` が
//! `vocabulary_size()` を超える pad 時、超過分の列は 0 のまま。n-gram 文字列は行内で
//! 使い回す 1 本のバッファで組み立て、確保を増やさない（A04）。エラーは入力文字列を
//! 保持しない（A09）。`output_sequence_length` は参照しない。

use super::error::TextError;
use super::limits::{TextLimits, check_exact_count};
use super::ngram::ngram_count;
use super::split::{Split, split};
use super::standardize::{Standardize, standardize};
use super::vocab::{PADDING_ID, Vocabulary};

/// bag 系変換の設定（内部型）。公開の `TextVectorizationConfig` から組み立てられる。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BagOptions {
    /// 標準化の方式。
    pub(crate) standardize: Standardize,
    /// 分割の方式。
    pub(crate) split: Split,
    /// `Some(n)` なら 1〜n-gram を作る。
    pub(crate) ngrams: Option<usize>,
    /// 出力の列数 `W`（`pad_to_max_tokens` なら `max_tokens`、そうでなければ語彙数）。
    pub(crate) width: usize,
}

/// 出力値の作り方（内部型）。公開しない。
#[derive(Debug, Clone, Copy)]
pub(crate) enum BagKind<'a> {
    /// 出現すれば 1.0。
    Presence,
    /// 出現回数。
    Count,
    /// 出現回数 × 重み（id 順の idf。長さは語彙数）。
    Weighted(&'a [f32]),
}

/// 1 行ぶんの（n-gram 込みの）トークン数を、確保せずに数える（事前走査用）。
fn row_token_count(
    input: &str,
    opts: &BagOptions,
    limits: &TextLimits,
    index: usize,
) -> Result<usize, TextError> {
    let std = standardize(input, opts.standardize, limits, index)?;
    let tokens = split(&std, opts.split).count();
    match opts.ngrams {
        Some(n) => ngram_count(tokens, n, limits),
        None => Ok(tokens),
    }
}

/// 1 トークンの id を `row` へ 1 加算する。id 0（パディング）は捨てる。
/// 範囲外の列は何もしない（多重防御。通常は `id < 語彙数 <= W` で到達しない）。
fn bump(row: &mut [f32], vocab: &Vocabulary, token: &str) {
    let id = vocab.lookup(token);
    if id == PADDING_ID {
        return;
    }
    if let Some(slot) = usize::try_from(id).ok().and_then(|i| row.get_mut(i)) {
        *slot += 1.0;
    }
}

/// 1 行を数え上げて `row`（長さ `W`、0 初期化済み）へ書く。
fn count_row(
    input: &str,
    vocab: &Vocabulary,
    opts: &BagOptions,
    limits: &TextLimits,
    index: usize,
    row: &mut [f32],
) -> Result<(), TextError> {
    let std = standardize(input, opts.standardize, limits, index)?;
    match opts.ngrams {
        None => {
            for t in split(&std, opts.split) {
                bump(row, vocab, t);
            }
        }
        Some(n) => {
            let toks: Vec<&str> = split(&std, opts.split).collect();
            // `ngram::ngrams` と同じ順序・同じ連結（半角空白 1 つ）。行内で 1 本のバッファを
            // 使い回し、n-gram ごとの `String` を作らない。
            let mut buf = String::new();
            for k in 1..=n {
                for w in toks.windows(k) {
                    buf.clear();
                    for (i, t) in w.iter().enumerate() {
                        if i > 0 {
                            buf.push(' ');
                        }
                        buf.push_str(t);
                    }
                    bump(row, vocab, &buf);
                }
            }
        }
    }
    Ok(())
}

/// 文字列バッチを `[B, W]` の `Tensor<f32>` へ変換する。
pub(crate) fn bag<S: AsRef<str>>(
    vocab: &Vocabulary,
    opts: &BagOptions,
    limits: &TextLimits,
    inputs: &[S],
    kind: BagKind<'_>,
) -> Result<crate::Tensor<f32>, TextError> {
    // 1. 設定値の検査（B = 0 でも行う）。
    if let Some(n) = opts.ngrams {
        limits.check_ngrams(n)?;
    }
    // 2. 入力の検査。
    limits.check_inputs(inputs)?;
    // 3. 事前走査（確保なし）。f32 で正確に数えられない行は確保前に拒否する。
    for (index, s) in inputs.iter().enumerate() {
        check_exact_count(index, row_token_count(s.as_ref(), opts, limits, index)?)?;
    }
    // 4. 出力要素数の検査（確保より先）→ 5. 一括確保。
    let batch = inputs.len();
    let width = opts.width;
    let elements = limits.check_output_elements(batch, width)?;
    let mut data = vec![0.0f32; elements];

    // 6. 行ごとに in-place で数え、モード別の後処理を同じ行バッファの上で行う。
    //    `chunks_exact_mut(0)` は panic するため `width == 0` は飛ばす。
    if width > 0 {
        for (index, (row, s)) in data.chunks_exact_mut(width).zip(inputs.iter()).enumerate() {
            count_row(s.as_ref(), vocab, opts, limits, index, row)?;
            match kind {
                BagKind::Presence => {
                    for v in row.iter_mut().filter(|v| **v > 0.0) {
                        *v = 1.0;
                    }
                }
                BagKind::Count => {}
                BagKind::Weighted(weights) => {
                    // 重みは語彙数ぶんだけ。pad で語彙数以上の列は常に 0 のため引かない。
                    for (v, w) in row.iter_mut().zip(weights.iter()) {
                        if *v != 0.0 {
                            *v *= *w;
                        }
                    }
                }
            }
        }
    }
    crate::Tensor::new(data, &[batch, width]).map_err(TextError::Shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(words: &[&str]) -> Vocabulary {
        let Ok(v) = Vocabulary::from_tokens(&TextLimits::default(), words) else {
            panic!("vocab must build");
        };
        v
    }

    fn opts(v: &Vocabulary) -> BagOptions {
        BagOptions {
            standardize: Standardize::None,
            split: Split::Whitespace,
            ngrams: None,
            width: v.vocabulary_size(),
        }
    }

    fn run(
        v: &Vocabulary,
        o: &BagOptions,
        inputs: &[&str],
        kind: BagKind<'_>,
    ) -> Result<(Vec<usize>, Vec<f32>), TextError> {
        bag(v, o, &TextLimits::default(), inputs, kind)
            .map(|t| (t.shape().to_vec(), t.host_slice().to_vec()))
    }

    #[test]
    fn count_and_presence_use_id_as_column() {
        let v = vocab(&["cat", "dog"]); // 0 "", 1 UNK, 2 cat, 3 dog
        let o = opts(&v);
        let (s, d) = run(&v, &o, &["cat cat dog bird", "bird"], BagKind::Count).unwrap();
        assert_eq!(s, vec![2, 4]);
        assert_eq!(d, vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        let (_, d) = run(&v, &o, &["cat cat dog bird", "bird"], BagKind::Presence).unwrap();
        assert_eq!(d, vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn id_zero_tokens_are_dropped_from_column_zero() {
        let v = vocab(&["cat"]);
        let mut o = opts(&v);
        o.split = Split::None;
        let (_, d) = run(&v, &o, &["", "cat"], BagKind::Count).unwrap();
        assert_eq!(d, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let (_, d) = run(&v, &opts(&v), &["  "], BagKind::Count).unwrap();
        assert_eq!(d, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn ngrams_are_counted_with_space_join() {
        let v = vocab(&["a", "a b"]); // 2 a, 3 "a b"
        let mut o = opts(&v);
        o.ngrams = Some(2);
        let (_, d) = run(&v, &o, &["a b a b"], BagKind::Count).unwrap();
        // 1-gram: a, b(OOV), a, b(OOV)。2-gram: "a b", "b a"(OOV), "a b"。
        assert_eq!(d, vec![0.0, 3.0, 2.0, 2.0]);
    }

    #[test]
    fn weighted_multiplies_count_by_weight_for_nonzero_columns_only() {
        let v = vocab(&["x", "y"]);
        let o = opts(&v);
        let w = [0.0f32, 0.5, 2.0, 4.0];
        let (_, d) = run(&v, &o, &["x x y z"], BagKind::Weighted(&w)).unwrap();
        assert_eq!(d, vec![0.0, 0.5, 4.0, 4.0]);
    }

    #[test]
    fn empty_batch_keeps_width_and_pad_columns_stay_zero() {
        let v = vocab(&["x"]);
        let mut o = opts(&v);
        let (s, d) = run(&v, &o, &[], BagKind::Count).unwrap();
        assert_eq!((s, d.len()), (vec![0, 3], 0));
        o.width = 6;
        let (s, d) = run(&v, &o, &["x q"], BagKind::Count).unwrap();
        assert_eq!(s, vec![1, 6]);
        assert_eq!(d, vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn output_size_is_rejected_before_allocation() {
        let v = vocab(&["x"]);
        let mut o = opts(&v);
        o.width = 1 << 30;
        let limits = TextLimits::default();
        let inputs = ["x"; 4];
        assert!(matches!(
            bag(&v, &o, &limits, &inputs, BagKind::Count),
            Err(TextError::OutputTooLarge { .. })
        ));
        let small = TextLimits::default().with_max_output_elements(6).unwrap();
        o.width = 3;
        assert!(bag(&v, &o, &small, &["x", "x"], BagKind::Count).is_ok());
        assert!(matches!(
            bag(&v, &o, &small, &["x", "x", "x"], BagKind::Count),
            Err(TextError::OutputTooLarge { .. })
        ));
    }
}
