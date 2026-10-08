//! テキスト変換パイプラインの結合テストと `Var::embedding` との結線テスト（#2903）。
//!
//! 役割: 標準化 → 分割 → n-gram → 語彙（直接指定／adapt）→ transform を通しで動かし、
//! 出力 id 列を手書きの期待値と整数の完全一致で確かめる。続けて transform の出力
//! `Tensor<i32>` を `Var::embedding(.., Some(0))` へ渡し、形状 `[B, L, D]` とパディング行
//! （id 0）の weight 勾配 0 を確かめる。各段の単体テスト（`adapt`・`transform` の
//! `tests`）が見ない「段をまたぐ並び順」と「下流 API との契約」が対象。
//!
//! 判定方式（設計記録 `docs/facade-text-vectorization-design.md` §7）: id は整数のため
//! `==` の完全一致で判定し、tolerance は新設しない（id が大きいと 1 ずれても相対誤差が
//! 小さくなり、複合判定では見逃すため）。Keras との bit 互換は契約にしない。ホスト側のみの
//! 処理で GPU 経路を通らないため、実機 parity テストと perf ログへの申し送りは対象外。
//! CPU の `crate::tape()` だけを使う。
//!
//! 期待値は実装の関数で組み立てず、設計記録 §3 の定義（頻度の降順・同頻度はバイト順・
//! n-gram は全 1-gram → 全 2-gram の順）から手で導いた値を直書きする。
//! `tests/text_module_hygiene.rs` が本ファイルのコメントも含めて走査する。

use super::adapt::adapt;
use super::error::TextError;
use super::limits::TextLimits;
use super::split::Split;
use super::standardize::Standardize;
use super::transform::{TransformOptions, transform};
use super::vocab::Vocabulary;

const SECRET: &str = "SECRET_MARKER_TOKEN";

fn limits() -> TextLimits {
    TextLimits::default()
}

fn tokens_of(v: &Vocabulary) -> Vec<&str> {
    v.tokens().iter().map(String::as_str).collect()
}

fn build(
    corpus: &[&str],
    st: Standardize,
    sp: Split,
    ngrams: Option<usize>,
    max_tokens: Option<usize>,
) -> Vocabulary {
    match adapt(corpus, st, sp, ngrams, max_tokens, &limits()) {
        Ok(v) => v,
        Err(e) => panic!("adapt failed: {e}"),
    }
}

fn run(v: &Vocabulary, opts: &TransformOptions, inputs: &[&str]) -> (Vec<usize>, Vec<i32>) {
    match transform(v, opts, &limits(), inputs) {
        Ok(t) => (t.shape().to_vec(), t.host_slice().to_vec()),
        Err(e) => panic!("transform failed: {e}"),
    }
}

fn corpus3() -> [&'static str; 3] {
    ["The cat sat.", "the dog sat", "A cat!"]
}

#[test]
fn direct_vocabulary_standardizes_before_lookup() {
    // cat=2・dog=3・the=4。標準化で "The CAT!" -> "the cat"。bird は OOV。
    let Ok(v) = Vocabulary::from_tokens(&limits(), &["cat", "dog", "the"]) else {
        panic!("vocab must build");
    };
    let (s, d) = run(
        &v,
        &TransformOptions::default(),
        &["The CAT!", "dog, bird?"],
    );
    assert_eq!(s, vec![2, 2]);
    assert_eq!(d, vec![4, 2, 3, 1]);
}

#[test]
fn adapt_order_decides_ids() {
    // 頻度 the=2・cat=2・sat=2・dog=1・a=1。同頻度はバイト順: cat < sat < the、a < dog。
    let v = build(
        &corpus3(),
        Standardize::default(),
        Split::default(),
        None,
        None,
    );
    assert_eq!(
        tokens_of(&v),
        vec!["", "[UNK]", "cat", "sat", "the", "a", "dog"]
    );
    // "THE dog!" -> [the=4, dog=6]。"a bird sat on the cat" -> [5, 1, 3, 1, 4, 2]。
    let inputs = ["THE dog!", "a bird sat on the cat"];
    let (s, d) = run(&v, &TransformOptions::default(), &inputs);
    assert_eq!(s, vec![2, 6]);
    assert_eq!(d, vec![4, 6, 0, 0, 0, 0, 5, 1, 3, 1, 4, 2]);

    let o = TransformOptions {
        output_sequence_length: Some(3),
        ..Default::default()
    };
    let (s, d) = run(&v, &o, &inputs);
    assert_eq!(s, vec![2, 3]);
    assert_eq!(d, vec![4, 6, 0, 5, 1, 3]);
}

#[test]
fn max_tokens_truncation_turns_rare_words_into_oov() {
    // max_tokens = 5: 予約 2 件 + 上位 3 語（cat・sat・the）。a・dog は OOV。
    let v = build(
        &corpus3(),
        Standardize::default(),
        Split::default(),
        None,
        Some(5),
    );
    assert_eq!(tokens_of(&v), vec!["", "[UNK]", "cat", "sat", "the"]);
    let (s, d) = run(
        &v,
        &TransformOptions::default(),
        &["THE dog!", "a bird sat on the cat"],
    );
    assert_eq!(s, vec![2, 6]);
    assert_eq!(d, vec![4, 1, 0, 0, 0, 0, 1, 1, 3, 1, 4, 2]);
}

#[test]
fn bigram_pipeline_through_adapt_and_transform() {
    // 数える語: a=2・b=2・"a b"=2・c=1・"b c"=1。頻度 2 は a < "a b" < b、頻度 1 は "b c" < c。
    let v = build(
        &["a b", "a b c"],
        Standardize::None,
        Split::Whitespace,
        Some(2),
        None,
    );
    assert_eq!(
        tokens_of(&v),
        vec!["", "[UNK]", "a", "a b", "b", "b c", "c"]
    );
    let o = TransformOptions {
        standardize: Standardize::None,
        ngrams: Some(2),
        ..Default::default()
    };
    // "a b c" -> 1-gram [a=2, b=4, c=6] の後に 2-gram ["a b"=3, "b c"=5]。
    // "c a" -> 1-gram [c=6, a=2] の後に "c a"（語彙外）=1。残りは 0 埋め。
    let (s, d) = run(&v, &o, &["a b c", "c a"]);
    assert_eq!(s, vec![2, 5]);
    assert_eq!(d, vec![2, 4, 6, 3, 5, 6, 2, 1, 0, 0]);
}

#[test]
fn split_modes_through_adapt_and_transform() {
    let v = build(&["aab"], Standardize::None, Split::Character, None, None);
    assert_eq!(tokens_of(&v), vec!["", "[UNK]", "a", "b"]);
    let o = TransformOptions {
        standardize: Standardize::None,
        split: Split::Character,
        ..Default::default()
    };
    let (s, d) = run(&v, &o, &["abc"]);
    assert_eq!(s, vec![1, 3]);
    assert_eq!(d, vec![2, 3, 1]);

    // Split::None は入力全体が 1 トークン。
    let v = build(
        &["Hello World", "x"],
        Standardize::None,
        Split::None,
        None,
        None,
    );
    assert_eq!(tokens_of(&v), vec!["", "[UNK]", "Hello World", "x"]);
    let o = TransformOptions {
        standardize: Standardize::None,
        split: Split::None,
        ..Default::default()
    };
    let (s, d) = run(&v, &o, &["Hello World", "hello world"]);
    assert_eq!(s, vec![2, 1]);
    assert_eq!(d, vec![2, 1]);
}

#[test]
fn corpus_order_does_not_change_output() {
    let inputs = ["THE dog!", "a bird sat on the cat"];
    let want = vec![4, 6, 0, 0, 0, 0, 5, 1, 3, 1, 4, 2];
    for corpus in [
        ["The cat sat.", "the dog sat", "A cat!"],
        ["A cat!", "the dog sat", "The cat sat."],
        ["the dog sat", "A cat!", "The cat sat."],
    ] {
        let v = build(
            &corpus,
            Standardize::default(),
            Split::default(),
            None,
            None,
        );
        let (_, d) = run(&v, &TransformOptions::default(), &inputs);
        assert_eq!(d, want);
    }
}

#[test]
fn pipeline_failures_are_typed_and_do_not_leak_input() {
    let Ok(tight) = limits().with_max_distinct_tokens(1) else {
        panic!("limits");
    };
    let r = adapt(
        &[SECRET, "other"],
        Standardize::None,
        Split::Whitespace,
        None,
        None,
        &tight,
    );
    let Err(e) = r else {
        panic!("adapt must fail on distinct-token limit");
    };
    assert!(matches!(e, TextError::TooManyDistinctTokens { .. }));
    assert!(!format!("{e}").contains(SECRET));
    assert!(!format!("{e:?}").contains(SECRET));

    let Ok(small) = limits().with_max_corpus_bytes(4) else {
        panic!("limits");
    };
    let r = adapt(
        &[SECRET],
        Standardize::None,
        Split::Whitespace,
        None,
        None,
        &small,
    );
    let Err(e) = r else {
        panic!("adapt must fail on corpus bytes");
    };
    assert!(!format!("{e}").contains(SECRET));
    assert!(!format!("{e:?}").contains(SECRET));

    // 語彙の Debug は件数だけで、語彙の全文を出さない。
    let Ok(v) = Vocabulary::from_tokens(&limits(), &[SECRET, "b"]) else {
        panic!("vocab must build");
    };
    assert!(!format!("{v:?}").contains(SECRET));
    let Ok(out_small) = limits().with_max_output_elements(3) else {
        panic!("limits");
    };
    let r = transform(
        &v,
        &TransformOptions::default(),
        &out_small,
        &[SECRET, "b b"],
    );
    let Err(e) = r else {
        panic!("transform must fail on output elements");
    };
    assert!(!format!("{e}").contains(SECRET));
    assert!(!format!("{e:?}").contains(SECRET));
}

// ---- embedding との結線（CPU のみ） ----

const DIM: usize = 3;

/// adapt した語彙（語彙数 7 = 予約 2 + 5 語）と、固定の 2 入力を transform した ids。
fn wired_ids(len: Option<usize>) -> (Vocabulary, crate::Tensor<i32>) {
    let v = build(
        &corpus3(),
        Standardize::default(),
        Split::default(),
        None,
        None,
    );
    let o = TransformOptions {
        output_sequence_length: len,
        ..Default::default()
    };
    match transform(&v, &o, &limits(), &["THE dog!", "a bird sat on the cat"]) {
        Ok(t) => (v, t),
        Err(e) => panic!("transform failed: {e}"),
    }
}

fn weight(rows: usize) -> crate::Tensor<f32> {
    match crate::Tensor::<f32>::new((0..rows * DIM).map(|i| i as f32).collect(), &[rows, DIM]) {
        Ok(t) => t,
        Err(e) => panic!("weight: {e}"),
    }
}

/// embedding の forward 形状・値と weight 勾配（行優先の平坦列）を返す。
fn embed_grad(
    ids: &crate::Tensor<i32>,
    rows: usize,
    padding_idx: Option<usize>,
) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let tape = crate::tape();
    let w = tape.var(&weight(rows));
    let Ok(out) = w.embedding(ids, padding_idx) else {
        panic!("embedding must accept transform output");
    };
    let value = out.to_tensor();
    let shape = value.shape().to_vec();
    let fwd = value.host_slice().to_vec();
    let Ok(loss) = out.sum(None) else {
        panic!("sum");
    };
    let Ok(grads) = tape.backward(&loss) else {
        panic!("backward");
    };
    let Ok(Some(dw)) = grads.get(&w) else {
        panic!("weight grad must exist");
    };
    (shape, fwd, dw.host_slice().to_vec())
}

#[test]
fn transform_output_feeds_embedding_and_zeroes_padding_grad() {
    let (v, ids) = wired_ids(None);
    assert_eq!(ids.shape(), &[2, 6]);
    let rows = v.vocabulary_size();
    assert_eq!(rows, 7);
    let (shape, fwd, dw) = embed_grad(&ids, rows, Some(0));
    assert_eq!(shape, vec![2, 6, DIM]);

    // forward: 各位置が weight の該当行（値は行 * DIM + 列）。
    let want_ids = [4usize, 6, 0, 0, 0, 0, 5, 1, 3, 1, 4, 2];
    let mut want_fwd = Vec::new();
    for id in want_ids {
        for c in 0..DIM {
            want_fwd.push((id * DIM + c) as f32);
        }
    }
    assert_eq!(fwd, want_fwd);

    // 出現回数: id0=4・id1=2・id2=1・id3=1・id4=2・id5=1・id6=1。行 0 は padding_idx で 0。
    let counts = [0.0f32, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0];
    let want_dw: Vec<f32> = counts.iter().flat_map(|&c| [c; DIM]).collect();
    assert_eq!(dw, want_dw);
}

#[test]
fn without_padding_idx_row_zero_receives_gradient() {
    let (v, ids) = wired_ids(None);
    let (_, _, dw) = embed_grad(&ids, v.vocabulary_size(), None);
    // 対照: padding_idx なしでは id 0 の 4 回分が行 0 に積まれる。
    assert_eq!(&dw[..DIM], &[4.0, 4.0, 4.0]);
}

#[test]
fn fixed_length_output_also_wires_into_embedding() {
    let (v, ids) = wired_ids(Some(3));
    assert_eq!(ids.shape(), &[2, 3]);
    let (shape, _, dw) = embed_grad(&ids, v.vocabulary_size(), Some(0));
    assert_eq!(shape, vec![2, 3, DIM]);
    // ids = [4, 6, 0, 5, 1, 3]。行 0 は 0、出現した行は 1、出ない行（2）は 0。
    let counts = [0.0f32, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let want_dw: Vec<f32> = counts.iter().flat_map(|&c| [c; DIM]).collect();
    assert_eq!(dw, want_dw);
}
