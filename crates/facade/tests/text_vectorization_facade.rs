//! `fandhe_ai::text`（語彙 lookup 型のテキスト変換）の公開パス経由の結合テスト。
//!
//! 役割: イシュー #2937 の公開面を、内部モジュールに触れず `fandhe_ai::text::…` だけで検証する
//! （契約の正は `docs/facade-text-vectorization-design.md` §16.5）。CPU のみ・ホスト側処理のみで、
//! 実機（CUDA／Metal）依存はない。判定は整数の完全一致。

use fandhe_ai::Tensor;
use fandhe_ai::text::{
    Split, Standardize, TextError, TextLimits, TextVectorization, TextVectorizationConfig,
};

fn cfg() -> TextVectorizationConfig {
    TextVectorizationConfig::default()
}

fn ids(t: &Tensor<i32>) -> Vec<i32> {
    t.host_slice().to_vec()
}

#[test]
fn default_config_matches_recorded_defaults() {
    let c = cfg();
    assert_eq!(c.max_tokens, None);
    assert_eq!(c.standardize, Standardize::LowerAndStripPunctuation);
    assert_eq!(c.split, Split::Whitespace);
    assert_eq!(c.ngrams, None);
    assert_eq!(c.output_sequence_length, None);
    assert_eq!(c.limits, TextLimits::default());
}

#[test]
fn from_vocabulary_assigns_ids_in_input_order_and_is_case_sensitive() {
    let c = cfg().with_standardize(Standardize::None);
    let tv = TextVectorization::from_vocabulary(c, &["Zed", "alpha", "beta"]).unwrap();
    assert_eq!(tv.vocabulary(), ["", "[UNK]", "Zed", "alpha", "beta"]);
    assert_eq!(tv.vocabulary_size(), 5);
    let out = tv.transform(&["Zed zed alpha beta gamma"]).unwrap();
    assert_eq!(out.shape(), &[1, 5]);
    assert_eq!(ids(&out), [2, 1, 3, 4, 1]);
}

#[test]
fn from_vocabulary_does_not_standardize_the_vocabulary() {
    // 語彙の "Cat" は標準化されない。入力は小文字化されるため "cat" になり一致しない。
    let tv = TextVectorization::from_vocabulary(cfg(), &["Cat"]).unwrap();
    assert_eq!(ids(&tv.transform(&["Cat"]).unwrap()), [1]);
}

#[test]
fn from_vocabulary_classifies_reserved_and_invalid_tokens() {
    assert!(matches!(
        TextVectorization::from_vocabulary(cfg(), &["a", ""]),
        Err(TextError::EmptyVocabularyToken { index: 1 })
    ));
    assert!(matches!(
        TextVectorization::from_vocabulary(cfg(), &["[UNK]"]),
        Err(TextError::ReservedVocabularyToken { index: 0 })
    ));
    assert!(matches!(
        TextVectorization::from_vocabulary(cfg(), &["a", "b", "a"]),
        Err(TextError::DuplicateVocabularyToken {
            first: 0,
            second: 2
        })
    ));
}

#[test]
fn max_tokens_is_fail_closed_at_construction() {
    let c = |m| cfg().with_max_tokens(m);
    assert!(matches!(
        TextVectorization::from_vocabulary(c(2), &["a"]),
        Err(TextError::InvalidMaxTokens { max_tokens: 2 })
    ));
    assert!(matches!(
        TextVectorization::adapt(c(2), &["a"]),
        Err(TextError::InvalidMaxTokens { max_tokens: 2 })
    ));
    assert!(matches!(
        TextVectorization::from_vocabulary(c(3), &["a", "b"]),
        Err(TextError::VocabularyTooLarge { len: 4, max: 3 })
    ));
    assert_eq!(
        TextVectorization::from_vocabulary(c(4), &["a", "b"])
            .unwrap()
            .vocabulary_size(),
        4
    );
}

#[test]
fn construction_rejects_bad_ngrams_and_output_length() {
    let empty: [&str; 0] = [];
    for bad in [cfg().with_ngrams(0), cfg().with_ngrams(9)] {
        assert!(matches!(
            TextVectorization::from_vocabulary(bad.clone(), &empty),
            Err(TextError::InvalidNgrams { .. })
        ));
        assert!(matches!(
            TextVectorization::adapt(bad, &empty),
            Err(TextError::InvalidNgrams { .. })
        ));
    }
    let limits = TextLimits::default()
        .with_max_output_sequence_length(4)
        .unwrap();
    let bad = cfg().with_limits(limits).with_output_sequence_length(5);
    assert!(matches!(
        TextVectorization::from_vocabulary(bad.clone(), &empty),
        Err(TextError::OutputSequenceLengthTooLarge { len: 5, max: 4 })
    ));
    assert!(matches!(
        TextVectorization::adapt(bad, &empty),
        Err(TextError::OutputSequenceLengthTooLarge { len: 5, max: 4 })
    ));
}

#[test]
fn adapt_orders_by_frequency_then_bytes_and_skips_reserved_tokens() {
    let corpus = ["b a a c", "c [UNK] a", "d"];
    // 標準化で "[UNK]" の句読点が除かれ "unk" になる。予約トークンは数えない挙動は None 標準化で確認する。
    let tv = TextVectorization::adapt(cfg(), &corpus).unwrap();
    assert_eq!(tv.vocabulary(), ["", "[UNK]", "a", "c", "b", "d", "unk"]);
    let raw = TextVectorization::adapt(cfg().with_standardize(Standardize::None), &corpus).unwrap();
    assert_eq!(raw.vocabulary(), ["", "[UNK]", "a", "c", "b", "d"]);
    let capped = TextVectorization::adapt(cfg().with_max_tokens(4), &corpus).unwrap();
    assert_eq!(capped.vocabulary(), ["", "[UNK]", "a", "c"]);
}

#[test]
fn transform_shapes_padding_and_truncation() {
    let tv = TextVectorization::from_vocabulary(cfg(), &["a", "b", "c"]).unwrap();
    let none: [&str; 0] = [];
    // B = 0
    assert_eq!(tv.transform(&none).unwrap().shape(), &[0, 0]);
    let fixed =
        TextVectorization::from_vocabulary(cfg().with_output_sequence_length(3), &["a"]).unwrap();
    assert_eq!(fixed.transform(&none).unwrap().shape(), &[0, 3]);
    // L = Some(0)
    let zero =
        TextVectorization::from_vocabulary(cfg().with_output_sequence_length(0), &["a"]).unwrap();
    assert_eq!(zero.transform(&["a b"]).unwrap().shape(), &[1, 0]);
    // 0 埋めと先頭からの切り詰め。
    let out = tv.transform(&["a b c a", "c"]).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    assert_eq!(ids(&out), [2, 3, 4, 2, 4, 0, 0, 0]);
    let cut =
        TextVectorization::from_vocabulary(cfg().with_output_sequence_length(2), &["a", "b", "c"])
            .unwrap();
    assert_eq!(ids(&cut.transform(&["a b c"]).unwrap()), [2, 3]);
}

#[test]
fn split_none_and_empty_input_map_to_padding_id() {
    let tv = TextVectorization::from_vocabulary(cfg().with_split(Split::None), &["a"]).unwrap();
    assert_eq!(ids(&tv.transform(&[""]).unwrap()), [0]);
    let ws =
        TextVectorization::from_vocabulary(cfg().with_output_sequence_length(2), &["a"]).unwrap();
    assert_eq!(ids(&ws.transform(&[""]).unwrap()), [0, 0]);
}

#[test]
fn lowercasing_and_whitespace_are_ascii_only() {
    let lower = cfg()
        .with_standardize(Standardize::Lower)
        .with_split(Split::None);
    let tv = TextVectorization::from_vocabulary(lower, &["Àb"]).unwrap();
    // "ÀB" は ASCII の B だけ小文字化され "Àb" になる（À は変わらない）。
    assert_eq!(ids(&tv.transform(&["ÀB"]).unwrap()), [2]);

    // 全角空白・NBSP では分割しない（1 トークンのまま OOV）。
    let ws = TextVectorization::from_vocabulary(cfg(), &["a", "b"]).unwrap();
    assert_eq!(ids(&ws.transform(&["a\u{3000}b"]).unwrap()), [1]);
    assert_eq!(ids(&ws.transform(&["a\u{00A0}b"]).unwrap()), [1]);
    assert_eq!(ids(&ws.transform(&["a\tb\n"]).unwrap()), [2, 3]);
}

#[test]
fn character_split_and_ngrams() {
    let ch = TextVectorization::adapt(cfg().with_split(Split::Character), &["ab"]).unwrap();
    assert_eq!(ch.vocabulary(), ["", "[UNK]", "a", "b"]);
    let ng = TextVectorization::adapt(cfg().with_ngrams(2), &["a b"]).unwrap();
    assert_eq!(ng.vocabulary_size(), 2 + 3); // a, b, "a b"
    assert_eq!(ng.transform(&["a b"]).unwrap().shape(), &[1, 3]);
}

#[test]
fn text_limits_with_methods_reject_above_absolute_maximum() {
    let l = TextLimits::default();
    assert!(matches!(
        l.with_max_ngrams(9),
        Err(TextError::LimitAboveAbsoluteMaximum {
            limit: "max_ngrams",
            value: 9,
            max: 8
        })
    ));
    assert!(matches!(
        l.with_max_input_bytes(usize::MAX),
        Err(TextError::LimitAboveAbsoluteMaximum { .. })
    ));
    assert!(matches!(
        l.with_max_vocabulary_size(usize::MAX),
        Err(TextError::LimitAboveAbsoluteMaximum { .. })
    ));
    assert!(l.with_max_batch(usize::MAX).is_ok());
    // 下げた上限が入力検査に効く。
    let small = l.with_max_batch(1).unwrap();
    let tv = TextVectorization::from_vocabulary(cfg().with_limits(small), &["a"]).unwrap();
    assert!(matches!(
        tv.transform(&["a", "a"]),
        Err(TextError::BatchTooLarge { len: 2, max: 1 })
    ));
}

#[test]
fn errors_and_debug_do_not_leak_input_or_vocabulary() {
    let marker = "SECRET_MARKER_TOKEN";
    let tv = TextVectorization::from_vocabulary(cfg(), &[marker]).unwrap();
    assert!(!format!("{tv:?}").contains(marker));
    let small = TextLimits::default().with_max_input_bytes(3).unwrap();
    let tv2 = TextVectorization::from_vocabulary(cfg().with_limits(small), &["a"]).unwrap();
    let err = tv2.transform(&[marker]).unwrap_err();
    assert!(!err.to_string().contains(marker));
    assert!(!format!("{err:?}").contains(marker));
    let dup = TextVectorization::from_vocabulary(cfg(), &[marker, marker]).unwrap_err();
    assert!(!dup.to_string().contains(marker));
    assert!(!format!("{dup:?}").contains(marker));
}

#[test]
fn transform_output_feeds_embedding_and_zeroes_padding_gradient() {
    let tv = TextVectorization::adapt(cfg(), &["the cat sat", "the dog"]).unwrap();
    let out = tv.transform(&["the cat", "dog"]).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(ids(&out), [2, 3, 4, 0]);

    let dim = 3;
    let rows = tv.vocabulary_size();
    let table = Tensor::<f32>::new(
        (0..rows * dim).map(|i| i as f32 + 1.0).collect(),
        &[rows, dim],
    )
    .unwrap();
    let tape = fandhe_ai::tape();
    let w = tape.var(&table);
    let emb = w.embedding(&out, Some(0)).unwrap();
    assert_eq!(emb.to_tensor().shape(), &[2, 2, dim]);
    let loss = emb.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&w).unwrap().unwrap();
    let dw = dw.host_slice();
    // padding_idx = Some(0) の行は勾配 0、使われた id の行は 1。
    assert!(dw[..dim].iter().all(|&g| g == 0.0));
    for id in [2usize, 3, 4] {
        assert!(dw[id * dim..(id + 1) * dim].iter().all(|&g| g == 1.0));
    }
}
