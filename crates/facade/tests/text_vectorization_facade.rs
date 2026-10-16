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
fn unicode_lower_and_unicode_whitespace_are_opt_in() {
    let uc = || {
        cfg()
            .with_standardize(Standardize::UnicodeLower)
            .with_split(Split::None)
    };
    let tv =
        TextVectorization::from_vocabulary(uc(), &["i\u{307}stanbul", "straße", "οδοσ"]).unwrap();
    assert_eq!(
        ids(&tv.transform(&["İSTANBUL", "STRAßE", "ΟΔΟΣ"]).unwrap()),
        [2, 3, 4]
    );
    // 文脈非依存: 語末の Σ も σ になるので ς の語彙とは一致しない。
    let tv = TextVectorization::from_vocabulary(uc(), &["οδος"]).unwrap();
    assert_eq!(ids(&tv.transform(&["ΟΔΟΣ"]).unwrap()), [1]);

    let ws = cfg()
        .with_split(Split::UnicodeWhitespace)
        .with_output_sequence_length(3);
    let tv = TextVectorization::from_vocabulary(ws, &["a", "b", "c"]).unwrap();
    let out = tv
        .transform(&["a\u{3000}b\u{00A0}c", "a\u{0B}b", "", "\u{3000}"])
        .unwrap();
    assert_eq!(ids(&out), [2, 3, 4, 2, 3, 0, 0, 0, 0, 0, 0, 0]);

    let both = cfg()
        .with_standardize(Standardize::UnicodeLower)
        .with_split(Split::UnicodeWhitespace);
    let tv = TextVectorization::adapt(both, &["Σ\u{3000}σ"]).unwrap();
    assert_eq!(tv.vocabulary(), ["", "[UNK]", "σ"]);
}

#[test]
fn unicode_lower_rejects_expanded_input_over_limit() {
    let limits = TextLimits::default().with_max_input_bytes(2).unwrap();
    let c = cfg()
        .with_standardize(Standardize::UnicodeLower)
        .with_split(Split::None)
        .with_limits(limits);
    let is_want = |e: TextError| {
        matches!(
            e,
            TextError::InputTooLong {
                index: 0,
                len: 3,
                max: 2
            }
        )
    };
    let tv = TextVectorization::from_vocabulary(c.clone(), &["a"]).unwrap();
    assert!(is_want(tv.transform(&["İ"]).unwrap_err()));
    let fixed =
        TextVectorization::from_vocabulary(c.clone().with_output_sequence_length(2), &["a"])
            .unwrap();
    assert!(is_want(fixed.transform(&["İ"]).unwrap_err()));
    assert!(is_want(TextVectorization::adapt(c, &["İ"]).unwrap_err()));
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
// ---------------------------------------------------------------------
// イシュー #3092: 出力モード（`transform_multi_hot`／`transform_count`／`transform_tf_idf`）。
// 公開形の正は `docs/facade-text-output-modes-string-lookup-decision.md`。判定は count・
// multi_hot が `==`、tf_idf が同じ式から作った値との bit 一致（新しい tolerance は作らない）。

fn f32s(t: &Tensor<f32>) -> Vec<f32> {
    t.host_slice().to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|w| w.to_bits()).collect()
}

fn idf_of(n: usize, df: usize) -> f32 {
    (1.0f64 + n as f64 / (1.0 + df as f64)).ln() as f32
}

#[test]
fn bag_modes_match_hand_computed_values() {
    let c = cfg().with_standardize(Standardize::None);
    let tv = TextVectorization::from_vocabulary(c, &["cat", "dog"]).unwrap();
    let ins = ["cat cat dog bird", "bird", ""];
    let count = tv.transform_count(&ins).unwrap();
    assert_eq!(count.shape(), &[3, 4]);
    assert_eq!(
        f32s(&count),
        [0., 1., 2., 1., 0., 1., 0., 0., 0., 0., 0., 0.]
    );
    let hot = tv.transform_multi_hot(&ins).unwrap();
    assert_eq!(f32s(&hot), [0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0.]);
}

#[test]
fn bag_ngrams_apply_and_output_sequence_length_is_ignored() {
    let c = cfg()
        .with_standardize(Standardize::None)
        .with_ngrams(2)
        .with_output_sequence_length(1);
    let tv = TextVectorization::from_vocabulary(c, &["a", "a b"]).unwrap();
    let out = tv.transform_count(&["a b a b"]).unwrap();
    assert_eq!(f32s(&out), [0., 3., 2., 2.]);
}

#[test]
fn bag_zero_column_ignores_id_zero_tokens() {
    let c = cfg().with_split(Split::None);
    let tv = TextVectorization::from_vocabulary(c, &["x"]).unwrap();
    let out = tv.transform_count(&["", "x"]).unwrap();
    assert_eq!(f32s(&out), [0., 0., 0., 0., 0., 1.]);
}

#[test]
fn bag_count_agrees_with_id_histogram_of_transform() {
    let corpus = ["the cat sat on the mat", "a dog and a cat", "zebra the end"];
    for (split, ngrams) in [
        (Split::Whitespace, None),
        (Split::Whitespace, Some(2)),
        (Split::Character, Some(3)),
        (Split::None, None),
    ] {
        let mut c = cfg().with_split(split);
        if let Some(n) = ngrams {
            c = c.with_ngrams(n);
        }
        let tv = TextVectorization::adapt(c, &corpus).unwrap();
        let ins = ["the cat the dog", "unseen words here", "a a a"];
        let ids = tv.transform(&ins).unwrap();
        let len = ids.shape()[1];
        let counts = tv.transform_count(&ins).unwrap();
        let w = tv.vocabulary_size();
        assert_eq!(counts.shape(), &[ins.len(), w]);
        for (r, row) in counts.host_slice().chunks(w).enumerate() {
            let mut hist = vec![0.0f32; w];
            for &id in &ids.host_slice()[r * len..(r + 1) * len] {
                if id != 0 {
                    hist[id as usize] += 1.0;
                }
            }
            assert_eq!(
                row,
                hist.as_slice(),
                "split={split:?} ngrams={ngrams:?} row={r}"
            );
        }
    }
}

#[test]
fn bag_pad_to_max_tokens_fixes_width() {
    let c = cfg().with_max_tokens(7).with_pad_to_max_tokens(true);
    let tv = TextVectorization::from_vocabulary(c, &["a", "b"]).unwrap();
    let out = tv.transform_count(&["a b b c"]).unwrap();
    assert_eq!(out.shape(), &[1, 7]);
    assert_eq!(f32s(&out), [0., 1., 1., 2., 0., 0., 0.]);
    assert_eq!(tv.transform_count::<&str>(&[]).unwrap().shape(), &[0, 7]);

    let no_pad = cfg().with_max_tokens(7);
    let tv = TextVectorization::adapt(no_pad, &["a b"]).unwrap();
    assert_eq!(tv.transform_count(&["a"]).unwrap().shape(), &[1, 4]);
}

#[test]
fn bag_pad_without_max_tokens_is_rejected_everywhere() {
    let c = cfg().with_pad_to_max_tokens(true);
    assert!(matches!(
        TextVectorization::from_vocabulary(c.clone(), &["a"]),
        Err(TextError::PadToMaxTokensWithoutMaxTokens)
    ));
    assert!(matches!(
        TextVectorization::adapt(c.clone(), &["a"]),
        Err(TextError::PadToMaxTokensWithoutMaxTokens)
    ));
    assert!(matches!(
        TextVectorization::from_vocabulary_with_idf(c.clone(), &["a"], &[1.0]),
        Err(TextError::PadToMaxTokensWithoutMaxTokens)
    ));
    assert!(matches!(
        TextVectorization::adapt_tf_idf(c, &["a"]),
        Err(TextError::PadToMaxTokensWithoutMaxTokens)
    ));
}

#[test]
fn weighted_adapt_matches_formula_and_round_trips() {
    let corpus = ["a a b", "a c", "b b b d"];
    let tv = TextVectorization::adapt_tf_idf(cfg(), &corpus).unwrap();
    assert_eq!(tv.vocabulary(), ["", "[UNK]", "b", "a", "c", "d"]);
    let idf = tv.idf_weights().unwrap().to_vec();
    let ws: Vec<f32> = [2usize, 2, 1, 1].iter().map(|&d| idf_of(3, d)).collect();
    let oov = (ws.iter().map(|&w| f64::from(w)).sum::<f64>() / ws.len() as f64) as f32;
    assert_eq!(idf[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(idf[1].to_bits(), oov.to_bits());
    assert_eq!(bits(&idf[2..]), bits(&ws));
    let out = tv.transform_tf_idf(&["b b a q"]).unwrap();
    let o = f32s(&out);
    assert_eq!(o[1].to_bits(), oov.to_bits());
    assert_eq!(o[2].to_bits(), (2.0f32 * ws[0]).to_bits());
    assert_eq!(o[3].to_bits(), ws[1].to_bits());

    let words: Vec<String> = tv.vocabulary()[2..].to_vec();
    let back = TextVectorization::from_vocabulary_with_idf(cfg(), &words, &idf[2..]).unwrap();
    assert_eq!(bits(back.idf_weights().unwrap()), bits(&idf));
    let again = f32s(&back.transform_tf_idf(&["b b a q"]).unwrap());
    assert_eq!(bits(&again), bits(&o));
}

#[test]
fn weighted_requires_weights_and_validates_them() {
    let tv = TextVectorization::from_vocabulary(cfg(), &["a"]).unwrap();
    assert!(tv.idf_weights().is_none());
    assert!(matches!(
        tv.transform_tf_idf(&["a"]),
        Err(TextError::MissingIdfWeights)
    ));
    assert!(matches!(
        TextVectorization::from_vocabulary_with_idf(cfg(), &["a", "b"], &[1.0]),
        Err(TextError::IdfWeightsLengthMismatch {
            len: 1,
            expected: 2
        })
    ));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.5] {
        assert!(matches!(
            TextVectorization::from_vocabulary_with_idf(cfg(), &["a", "b"], &[1.0, bad]),
            Err(TextError::InvalidIdfWeight { index: 1 })
        ));
    }
    assert!(TextVectorization::from_vocabulary_with_idf(cfg(), &["a", "b"], &[-0.0, 0.0]).is_ok());
}

#[test]
fn bag_rejects_oversized_output_before_allocating() {
    // 語彙を実際には作らず、`B × W` の超過だけを確かめる。
    let c = cfg().with_max_tokens(1 << 30).with_pad_to_max_tokens(true);
    let tv = TextVectorization::from_vocabulary(c, &["a"]).unwrap();
    assert!(matches!(
        tv.transform_count(&["a", "a", "a"]),
        Err(TextError::OutputTooLarge { .. })
    ));
    let limits = TextLimits::default().with_max_output_elements(6).unwrap();
    let tv = TextVectorization::from_vocabulary(cfg().with_limits(limits), &["a"]).unwrap();
    assert!(tv.transform_count(&["a", "a"]).is_ok()); // 2 × 3 = 6
    assert!(matches!(
        tv.transform_count(&["a", "a", "a"]),
        Err(TextError::OutputTooLarge {
            elements: 9,
            max: 6
        })
    ));
}

#[test]
fn bag_rejects_inputs_over_limits_before_allocating() {
    let limits = TextLimits::default()
        .with_max_batch(2)
        .unwrap()
        .with_max_input_bytes(3)
        .unwrap();
    let tv = TextVectorization::from_vocabulary(cfg().with_limits(limits), &["a"]).unwrap();
    assert!(matches!(
        tv.transform_count(&["a", "a", "a"]),
        Err(TextError::BatchTooLarge { len: 3, max: 2 })
    ));
    assert!(matches!(
        tv.transform_count(&["a", "abcd"]),
        Err(TextError::InputTooLong { index: 1, .. })
    ));
}

#[test]
fn bag_rejects_token_counts_f32_cannot_count_exactly() {
    // Split::Character + 8-gram: t 文字から 8t - 28 個。t = 2^21 + 4 で 2^24 を超える。
    let limits = TextLimits::default().with_max_input_bytes(4 << 20).unwrap();
    let c = cfg()
        .with_standardize(Standardize::None)
        .with_split(Split::Character)
        .with_ngrams(8)
        .with_limits(limits);
    let tv = TextVectorization::from_vocabulary(c, &["a"]).unwrap();
    let big = "a".repeat((1 << 21) + 4);
    match tv.transform_count(&["a", big.as_str()]) {
        Err(TextError::TokenCountNotExact { index, count, max }) => {
            assert_eq!((index, max), (1, 1 << 24));
            assert!(count > max);
        }
        other => panic!("TokenCountNotExact を期待: {other:?}"),
    }
}

#[test]
fn bag_errors_and_debug_do_not_leak_values() {
    let tv = TextVectorization::from_vocabulary_with_idf(
        cfg(),
        &["SECRET_MARKER_TOKEN"],
        &[0.314_159_27],
    )
    .unwrap();
    let dbg = format!("{tv:?}");
    assert!(!dbg.contains("SECRET_MARKER_TOKEN") && !dbg.contains("0.3141"));
    let err =
        TextVectorization::from_vocabulary_with_idf(cfg(), &["SECRET_MARKER_TOKEN"], &[f32::NAN])
            .unwrap_err();
    let shown = format!("{err} {err:?}");
    assert!(!shown.contains("SECRET_MARKER_TOKEN") && !shown.contains("NaN"));
}
