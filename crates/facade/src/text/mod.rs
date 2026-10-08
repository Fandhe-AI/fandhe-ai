//! 語彙 lookup 型のテキスト変換（Keras `TextVectorization` 相当）。
//!
//! 文字列バッチを標準化・分割（任意で n-gram）し、語彙で整数 id へ引いて
//! `[B, L]` の `Tensor<i32>` にする。出力は `Var::embedding(.., Some(0))` へ変換なしで
//! 渡せる。ホスト側の処理だけで完結し、バックエンド（CPU／CUDA／Metal）には依存しない。
//!
//! # 公開する型
//!
//! [`TextVectorization`]・[`TextVectorizationConfig`]・[`TextLimits`]・[`Standardize`]・
//! [`Split`]・[`TextError`] の 6 名だけをこのモジュールから公開する（クレートルートへは
//! 再エクスポートしない）。
//!
//! # 範囲
//!
//! 語彙 lookup 型の単語／文字レベルの変換だけを扱う。サブワードトークナイザ・Unicode
//! 正規化・書記素クラスタ分割・語彙のファイル保存／読み込み・ragged 出力・出力モード
//! （`multi_hot`／`count`／`tf_idf`）・`StringLookup` 相当は対象外。
//!
//! # ASCII 限定
//!
//! **小文字化と空白分割は ASCII に限る**。`Standardize::Lower` は `A`〜`Z` だけを変換し、
//! `Standardize::StripPunctuation` は ASCII 句読点 32 文字だけを削除する。
//! `Split::Whitespace` は ASCII 空白（SP・TAB・LF・FF・CR）だけで分割し、全角空白や
//! NBSP では分割しない。非 ASCII の文字は変更されずにそのまま語彙の lookup に渡る。
//! Unicode 対応は保留中で、`Standardize`／`Split` が `#[non_exhaustive]` のため
//! 後から variant を足しても非破壊になる。
//!
//! # id の割り当てと契約
//!
//! - id は 0 = パディング（`""`）、1 = OOV（`"[UNK]"`）、2.. = 語彙。直接指定
//!   （`from_vocabulary`）は入力順、`adapt` は頻度降順で同頻度はバイト列の辞書順
//! - `adapt` はコーパス中の予約トークン（`""`・`"[UNK]"`）を数えない
//! - `adapt` は語彙 1 件のバイト数を検査しない。このため `adapt` の結果を既定の
//!   [`TextLimits`] の `from_vocabulary` へ渡すと読み戻せない場合がある
//! - 予約語の分類: `""` は `EmptyVocabularyToken`、`"[UNK]"` は `ReservedVocabularyToken`
//! - 直接指定の語彙は標準化せず、lookup はバイト完全一致（大小文字を区別する）
//! - `transform`: `B = 0` は `[0, L]`（`output_sequence_length` が `None` なら `L = 0`）／
//!   `Split::None` と空入力 `""` は id 0（パディングと区別できない）／切り詰めは先頭
//!   `L` 個を残す／`Some(0)` は `[B, 0]`／`TextLimits::with_max_output_sequence_length` は
//!   `Some(L)` にだけ適用し、`None` で導いた `L` は出力要素数の上限で抑える
//! - `max_tokens` が `Some(m)` で `m` が 2 以下は `InvalidMaxTokens`。`from_vocabulary` で
//!   語彙数（予約 2 件を含む）が `m` を超えると `VocabularyTooLarge`（黙って切り詰めない）
//! - `ngrams`・`output_sequence_length` は構築時（`from_vocabulary`／`adapt`）に検査する
//! - 判定方式は整数の完全一致。Keras との bit 互換は契約にしない
//! - 入力の件数・バイト数の上限は標準化・分割・確保より先に検査する（[`TextLimits`]）。
//!   エラーと `Debug` は入力文字列・語彙の全文を含まない
//!
//! # 使用例
//!
//! コーパスから語彙を作り、変換した id を埋め込み層へ渡す。
//!
//! ```
//! use fandhe_ai::text::{TextVectorization, TextVectorizationConfig};
//! use fandhe_ai::Tensor;
//!
//! let config = TextVectorizationConfig::default().with_output_sequence_length(4);
//! let tv = TextVectorization::adapt(config, &["the cat sat", "the dog"]).unwrap();
//! // 語彙: [""(0), "[UNK]"(1), "the"(2), "cat"(3), "dog"(4), "sat"(5)]
//! assert_eq!(tv.vocabulary_size(), 6);
//!
//! let ids = tv.transform(&["The cat!", "a bird"]).unwrap();
//! assert_eq!(ids.shape(), &[2, 4]);
//! // 標準化で小文字化・句読点除去。語彙にない語は 1（OOV）、足りない分は 0 埋め。
//! assert_eq!(ids.host_slice().to_vec(), [2, 3, 0, 0, 1, 1, 0, 0]);
//!
//! // id はそのまま Var::embedding に渡せる（padding_idx = Some(0)）。
//! let dim = 3;
//! let rows = tv.vocabulary_size();
//! let table = Tensor::<f32>::new((0..rows * dim).map(|i| i as f32).collect(), &[rows, dim])
//!     .unwrap();
//! let tape = fandhe_ai::tape();
//! let weight = tape.var(&table);
//! let embedded = weight.embedding(&ids, Some(0)).unwrap();
//! assert_eq!(embedded.to_tensor().shape(), &[2, 4, 3]);
//! ```
//!
//! 語彙を直接指定する場合（lookup は大小文字を区別した完全一致）。
//!
//! ```
//! use fandhe_ai::text::{Standardize, TextError, TextVectorization, TextVectorizationConfig};
//!
//! let config = TextVectorizationConfig::default().with_standardize(Standardize::None);
//! let tv = TextVectorization::from_vocabulary(config.clone(), &["Cat", "dog"]).unwrap();
//! let ids = tv.transform(&["Cat cat dog"]).unwrap();
//! assert_eq!(ids.host_slice().to_vec(), [2, 1, 3]); // "cat" は語彙の "Cat" と別物で OOV
//!
//! // 予約語は語彙に入れられない。
//! let err = TextVectorization::from_vocabulary(config, &["[UNK]"]).unwrap_err();
//! assert!(matches!(err, TextError::ReservedVocabularyToken { index: 0 }));
//! ```
//!
//! 設計・承認の正は `docs/facade-text-vectorization-design.md` §16・§17。

pub(crate) mod adapt;
pub(crate) mod error;
pub(crate) mod limits;
pub(crate) mod ngram;
pub(crate) mod split;
pub(crate) mod standardize;
pub(crate) mod transform;
pub(crate) mod vectorization;
pub(crate) mod vocab;

#[cfg(test)]
mod integration_tests;

pub use error::TextError;
pub use limits::TextLimits;
pub use split::Split;
pub use standardize::Standardize;
pub use vectorization::{TextVectorization, TextVectorizationConfig};
