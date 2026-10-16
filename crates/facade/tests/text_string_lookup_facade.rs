//! `fandhe_ai::text::StringLookup`（イシュー #3093・親 #3090）の公開パス結合テスト。
//!
//! 役割: 公開パス（`fandhe_ai::text`）だけを使い、`TextVectorization` との id 一致・往復・
//! 不正入力の `Err`・エラーと `Debug` に入力文字列が出ないこと・`Var::embedding` への接続を
//! 固定する。判定は整数の完全一致（統一複合判定・tolerance は使わない）。ホスト処理のみで
//! 実機（CUDA／Metal）は対象外。公開形の正は
//! `docs/facade-text-output-modes-string-lookup-decision.md` §4・§5。

use fandhe_ai::Tensor;
use fandhe_ai::text::{
    Split, Standardize, StringLookup, StringLookupConfig, TextError, TextVectorization,
    TextVectorizationConfig,
};

const VOCAB: [&str; 4] = ["Cat", "dog", "日本", "x y"];
const PROBE: [&str; 8] = ["Cat", "cat", "dog", "", "日本", "[UNK]", "bird", "x y"];

fn tv_config() -> TextVectorizationConfig {
    TextVectorizationConfig::default()
        .with_standardize(Standardize::None)
        .with_split(Split::None)
}

#[test]
fn ids_match_text_vectorization_for_the_same_vocabulary() {
    let sl = StringLookup::from_vocabulary(StringLookupConfig::default(), &VOCAB).unwrap();
    let tv = TextVectorization::from_vocabulary(tv_config(), &VOCAB).unwrap();
    let a = sl.lookup(&PROBE).unwrap();
    let b = tv.transform(&PROBE).unwrap();
    assert_eq!(a.shape(), &[PROBE.len()]);
    assert_eq!(b.shape(), &[PROBE.len(), 1]);
    assert_eq!(a.host_slice().to_vec(), b.host_slice().to_vec());
    assert_eq!(a.host_slice().to_vec(), [2, 1, 3, 0, 4, 1, 1, 5]);
    assert_eq!(sl.vocabulary(), tv.vocabulary());
}

#[test]
fn adapt_matches_text_vectorization_adapt() {
    let corpus = ["b", "a b", "a b", "c", "c", "", "[UNK]"];
    let sl = StringLookup::adapt(StringLookupConfig::default(), &corpus).unwrap();
    let tv = TextVectorization::adapt(tv_config(), &corpus).unwrap();
    assert_eq!(sl.vocabulary(), tv.vocabulary());
    let sl =
        StringLookup::adapt(StringLookupConfig::default().with_max_tokens(4), &corpus).unwrap();
    let tv = TextVectorization::adapt(tv_config().with_max_tokens(4), &corpus).unwrap();
    assert_eq!(sl.vocabulary(), tv.vocabulary());
}

#[test]
fn lookup_then_invert_round_trips_and_token_agrees() {
    let sl = StringLookup::from_vocabulary(StringLookupConfig::default(), &VOCAB).unwrap();
    let ids = sl.lookup(&PROBE).unwrap();
    let back = sl.invert(&ids).unwrap();
    let want: Vec<&str> = PROBE
        .iter()
        .map(|t| {
            if VOCAB.contains(t) || t.is_empty() {
                *t
            } else {
                "[UNK]"
            }
        })
        .collect();
    assert_eq!(back, want);
    for (i, id) in ids.host_slice().iter().enumerate() {
        assert_eq!(sl.token(*id).unwrap(), back[i]);
        assert_eq!(sl.token(*id).unwrap(), sl.vocabulary()[*id as usize]);
    }
}

#[test]
fn empty_batch_gives_shape_zero() {
    let sl = StringLookup::from_vocabulary(StringLookupConfig::default(), &VOCAB).unwrap();
    let ids = sl.lookup::<&str>(&[]).unwrap();
    assert_eq!(ids.shape(), &[0]);
    assert!(sl.invert(&ids).unwrap().is_empty());
}

#[test]
fn invalid_inputs_are_errors_without_leaking_text() {
    let marker = "SECRET_MARKER";
    let c = StringLookupConfig::default;
    let errs: Vec<TextError> = vec![
        StringLookup::from_vocabulary(c(), &[marker, marker]).unwrap_err(),
        StringLookup::from_vocabulary(c(), &[marker, "[UNK]"]).unwrap_err(),
        StringLookup::from_vocabulary(c(), &[marker, ""]).unwrap_err(),
        StringLookup::from_vocabulary(c().with_max_tokens(3), &[marker, "y"]).unwrap_err(),
        StringLookup::from_vocabulary(c().with_max_tokens(2), &[marker]).unwrap_err(),
        StringLookup::from_vocabulary(c(), &[marker])
            .unwrap()
            .token(-1)
            .unwrap_err(),
    ];
    for e in &errs {
        assert!(!e.to_string().contains(marker), "{e}");
        assert!(!format!("{e:?}").contains(marker), "{e:?}");
    }
    let sl = StringLookup::from_vocabulary(c(), &[marker]).unwrap();
    assert!(!format!("{sl:?}").contains(marker));
    assert!(!format!("{:?}", sl.config()).contains(marker));

    let bad = Tensor::<i32>::new(vec![0, 99], &[2]).unwrap();
    assert!(matches!(
        sl.invert(&bad),
        Err(TextError::IdOutOfRange { index: 1 })
    ));
}

#[test]
fn lookup_output_feeds_embedding() {
    let sl = StringLookup::from_vocabulary(StringLookupConfig::default(), &VOCAB).unwrap();
    let ids = sl.lookup(&PROBE).unwrap();
    let (rows, dim) = (sl.vocabulary_size(), 3);
    let table =
        Tensor::<f32>::new((0..rows * dim).map(|i| i as f32).collect(), &[rows, dim]).unwrap();
    let tape = fandhe_ai::tape();
    let weight = tape.var(&table);
    let out = weight.embedding(&ids, Some(0)).unwrap();
    assert_eq!(out.to_tensor().shape(), &[PROBE.len(), dim]);
}
