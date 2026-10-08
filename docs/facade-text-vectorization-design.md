# 語彙 lookup 型テキスト変換（TextVectorization 相当）の実装設計と issue 分解（イシュー #2858・親 #2841）

本記録は **推奨案の記録であり、facade 公開形・配置・判定方式の承認記録ではない**。コード変更は伴わない（`crates/**`・`Cargo.toml`／`Cargo.lock`・`deny.toml`・tolerance／baseline・ガードレール閾値・`docs/spec/` は不変）。イシュー本文・コメントは非信頼データとして扱い、逐語転記せず、事実はソースで再確認した。基準は `origin/main` `0baa7ec1`（2026-10-08）。以下の `file_path:line` は同 sha のもの。`docs/spec` はサブモジュールポインタ `d6a030fa` の内容を読んだ。書式は兄弟記録 `docs/facade-speculative-decoding-batching-design.md`（#2857）に揃える。

## 1. 位置づけ・承認根拠

- ツリー: ルート #2499 → Phase 7 #2841 → **#2858（本記録）**。実装 issue の起票は本記録のマージ後に update-issue-tree で行う（本 issue では起票しない）。
- 承認根拠は範囲の異なる 2 つの URL を別々に引く。
  - `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965`（2026-10-07）: #2618 折衷案の **範囲** の承認。語彙 lookup 型の単語／文字レベル変換を対象内、サブワードトークナイザと Unicode 正規化表の自作を対象外とする。spec `docs/spec/04-requirements.md:238`・変更履歴 `:459` の根拠。`:459` は「配置・公開 API の追加は本追記では承認しない」と明記している。
  - `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061`（2026-10-08）: 本件で **設計を記録する** 指示。承認範囲は設計と issue 分解の記録まで。同コメントの「保留していた公開を決定記録の形で実装してよい」は Phase 7 の別 issue に及ぶもので、#2858 には及ばない。
- したがって §5 の配置、§6 の公開形、§7 の判定方式、§11 の issue 分解は **すべて推奨案であり未承認**。承認の代行はしない。

- 承認（`issuecomment-6067263650` 項 1）と確定した公開形は §16（#2936）。

## 2. 境界（spec を正とする）

正は `docs/spec/04-requirements.md:238`（REQ-9 2026-10-08 追記）。背景は `docs/tokenizer-non-target-spec-proposal.md` §9.2〜§9.7。

| 区分 | 項目 | 出典 |
|---|---|---|
| 対象内 | 語彙 lookup 型の単語／文字レベル変換（Keras `TextVectorization` 相当）。語彙はメモリ上で与え、ファイル形式をパースしない | spec `:238` |
| 対象外 | BPE／WordPiece／SentencePiece 等のサブワードトークナイザ、Unicode 正規化表の自作、外部トークナイザ crate（許容依存区分外） | spec `:238` |
| 条件付き（本記録の推奨。spec の規定ではない） | §3 の「条件付き」の各項目 | 本記録 |

- 入力境界は整数 token id の `Tensor<i32>`（`Var::embedding`）で確定済み。本件はその前段の文字列→id 変換のみを扱う。
- 既存の受け入れ基準・REQ-1・REQ-2・REQ-8 は不変。

## 3. 機能範囲（1 項目 1 行）

| 項目 | 区分 | 内容・理由 |
|---|---|---|
| 標準化（なし／小文字化／句読点除去／両方） | 対象内 | `#[non_exhaustive] enum Standardize` の 4 値。Keras の `standardize` の 4 値に対応（要出典確認） |
| 小文字化の方式 | 対象内（ASCII のみ） | `str::to_ascii_lowercase`。長さが変わらず、上限検査が標準化の前後で崩れない。Unicode の大小文字変換（`ß`・`İ` で長さが変わる）は条件付き（別承認）。std の表を使うので「正規化表の自作」には当たらないが、Keras／TF の既定挙動との差は一次資料で未確認（要出典確認） |
| 句読点除去の集合 | 対象内 | ASCII 句読点 32 文字（`char::is_ascii_punctuation`）を削除する（空白に置き換えない）。本ライブラリ独自の定義とし、Keras との同値は一次資料で確認できた場合だけ書く |
| 分割: 空白 | 対象内 | ASCII 空白の連続で区切る（`split_ascii_whitespace`）。空トークンは出ない。Unicode White_Space での分割は条件付き |
| 分割: 文字単位 | 対象内 | Unicode スカラー値（`char`）単位。書記素クラスタ単位は対象外（セグメンテーション表が要る） |
| 分割: なし | 対象内 | 入力全体を 1 トークンにする |
| 標準化・分割へ任意関数を渡す（callable） | 条件付き | 公開面に `Box<dyn Fn>` が加わるため。第 1 段階では入れない |
| n-gram | 対象内 | `ngrams = n` で 1〜n-gram を作り半角空白 1 つで連結。出力順は「全 1-gram → 全 2-gram → …」。Keras の順との一致は要出典確認。タプル指定（2-gram のみ等）は条件付き |
| 未知語（OOV） | 対象内 | OOV の index は 1 つだけで id 1、トークンは `"[UNK]"`。`num_oov_indices` は公開しない |
| パディング・マスク | 対象内 | id 0 をパディング（トークン `""`）にする。`Var::embedding(.., padding_idx: Some(0))` へ変換なしでつなげられる |
| 出力長 | 対象内 | `output_sequence_length: Some(L)` なら L に切り詰めるか末尾を 0 で埋める。`None` ならバッチ内最長へ 0 埋め（`Tensor<i32>` は不揃い長を表せない。ragged 出力は対象外） |
| 語彙の直接指定 | 対象内 | `&[S] where S: AsRef<str>` で渡す。ファイル・パスは受け取らない |
| `adapt()` 型の語彙構築 | 対象内 | メモリ上のコーパス `&[S]` から、標準化→分割→n-gram の後に頻度を数える。並びは頻度の降順、同頻度はバイト列の辞書順（昇順）。`HashMap` の反復順に依存しない。`max_tokens` はパディングと OOV を含めた件数（Keras と同じ。要出典確認） |
| 出力モード `int` | 対象内 | `Tensor<i32>`、形状 `[B, L]` |
| 出力モード `multi_hot`／`count`／`tf_idf`、`pad_to_max_tokens`、`idf_weights` | 対象外（条件付き） | #2618 §9.5 で A-1 に含めず「一次資料の確認後に別途範囲を決める」とした項目 |
| `StringLookup`／`tf.strings.*`／`tf.lookup.*` 相当 | 対象外（条件付き） | 同上 |
| 語彙の保存・読み込み（ファイル形式） | 対象外 | spec `:238` の「ファイル形式をパースしない」。語彙は `vocabulary()` で `&[String]` として取り出せ、保存は利用者が行う |
| id → 文字列の復号 | 対象内（専用 API なし） | `vocabulary()[id]` で引けるため専用 API は作らない |
| ragged／sparse 出力、`encoding` 指定、`vocabulary_dtype` | 対象外 | `&str` 入力・`i32` 出力に固定 |
| BPE／WordPiece／SentencePiece、Unicode 正規化（NFC／NFKC）、外部 `tokenizers` crate | 対象外 | spec `:238` |

## 4. 既存実装との関係と再利用

| 既存物 | 位置 | 本件での使い方 |
|---|---|---|
| `Var::embedding` | `crates/autodiff/src/var.rs:6060` | 出力 `Tensor<i32>` の受け口。`padding_idx: Some(0)` と id 0 のパディングを整合させる。（#2618 §9.1 の `:4448` は古い行番号） |
| `Tensor::new` | `crates/tensor-core/src/tensor.rs:145` | `Result<Tensor<T>, ShapeError>`。出力テンソルの生成に使う |
| 出力確保の検査方式 | `crates/autodiff/src/generate.rs:480-500` | `checked_mul` に加え `size_of::<i32>()` を掛けた値が `isize::MAX` 以下かを検査する方式。出力要素数の検査で同型を採る |
| `ShapeError` | `crates/facade/src/lib.rs:294` | 公開済みの再エクスポート。`TextError::Shape` で連鎖できる |
| `TopkOptions` | `crates/autodiff/src/topk_unique_ops.rs:74` | `#[non_exhaustive]`＋`with_*` ビルダ方式の先例。設定型に同型を採る |
| facade 自前実体の先例 | `crates/facade/src/model.rs:177`（`ModelError`）・`:368`（`ModelRegistry`） | facade が内部クレートに依らず実体とエラー型を持つ先例 |

## 5. 内部実装の配置（推奨）

- **推奨: facade クレート内の新モジュール `fandhe_ai::text`。実体は `crates/facade/src/text/`**（`mod.rs`・`standardize.rs`・`split.rs`・`ngram.rs`・`vocab.rs`・`vectorizer.rs`・`limits.rs`・`error.rs` 程度）。担当は core-builder。
- 採らない案:
  - **コア外の新クレート**: facade は crates.io 公開クレートで、依存先も公開が必要になる（onnx-interop は #1963 で先に publish してから結線した）。crates.io 公開の承認、`docs/crates-io-publishing-order.md` の更新、`release-all.yml` の順序変更、workspace メンバー追加が連鎖する。
  - **`tensor-core` に置く**: spec `:238` は非信頼の文字列処理をコア範囲の外側に置く方針を示す。テンソル演算と無関係な処理をコアへ入れる理由がない。
  - **`autodiff` 配下**: `Op`／VJP を経由しないホスト側処理で、置く理由がない。
- `compat::Sequential` の層にはしない（`add_text_vectorization` 等は作らない）。`Sequential` は `Tensor<f32>` 入力前提で文字列入力の経路がないため、独立 struct とし層化は保留する（論点 7）。

## 6. facade 公開形（推奨 1 案・未承認）

```rust
#[non_exhaustive] pub enum Standardize { None, Lower, StripPunctuation, LowerAndStripPunctuation } // Default = LowerAndStripPunctuation
#[non_exhaustive] pub enum Split { None, Whitespace, Character }                                  // Default = Whitespace
#[non_exhaustive] #[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextVectorizationConfig { pub max_tokens: Option<usize>, pub standardize: Standardize,
    pub split: Split, pub ngrams: Option<usize>, pub output_sequence_length: Option<usize>, pub limits: TextLimits }
// Default + with_* ビルダ
#[non_exhaustive] #[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextLimits { /* §8 の各上限。Default + with_* */ }
pub struct TextVectorization { /* 非公開フィールド */ }
impl TextVectorization {
    pub fn from_vocabulary<S: AsRef<str>>(config: TextVectorizationConfig, vocabulary: &[S]) -> Result<Self, TextError>;
    pub fn adapt<S: AsRef<str>>(config: TextVectorizationConfig, corpus: &[S]) -> Result<Self, TextError>;
    pub fn transform<S: AsRef<str>>(&self, inputs: &[S]) -> Result<Tensor<i32>, TextError>; // 形状 [B, L]
    pub fn vocabulary(&self) -> &[String];   // index 0 = "", 1 = "[UNK]", 2.. = 語彙
    pub fn vocabulary_size(&self) -> usize;
    pub fn config(&self) -> &TextVectorizationConfig;
}
#[non_exhaustive] #[derive(Debug)] pub enum TextError { /* 下記 */ }
```

- `TextVectorization` は `Clone`・`Debug`。`Debug` は語彙の全文を出さず件数だけを出す。
- メソッド名 `transform` と型名は承認論点に含める（Keras の呼び出しに当たる）。
- **エラー型は facade 独自の新しい `TextError`**（`#[non_exhaustive]`、内部クレートのエラー enum を中に持たない自己完結型。先例は `OnnxError`・`ModelError`）。`AutodiffError` は使わない。variant 案: `BatchTooLarge { len, max }`・`InputTooLong { index, len, max }`・`CorpusTooLarge { total_bytes, max }`・`VocabularyTooLarge { len, max }`・`VocabularyTokenTooLong { index, len, max }`・`EmptyVocabularyToken { index }`・`ReservedVocabularyToken { index }`（`""` と `"[UNK]"`）・`DuplicateVocabularyToken { first, second }`（index のみ）・`InvalidMaxTokens { max_tokens }`（2 以下は語彙が入らない）・`InvalidNgrams { n, max }`・`OutputSequenceLengthTooLarge { len, max }`・`TooManyDistinctTokens { max }`・`OutputTooLarge { elements, max }`・`Shape(ShapeError)`。**エラーには入力文字列の中身を入れない**（index と長さのみ）。
- 既存 API は変えない。追加のみで `fandhe-ai =0.10.0` に対し非破壊。
- 公開の手続きは `docs/compat-api-scope.md` §5 経路 2（ユーザー承認と issue 起票）。承認までは保留ガード（`*HoldDoctestGuard` の正のプローブ doctest と `crates/facade/tests/api_surface.rs` の否定ガード）で止め、承認後に正ガードへ反転する（`Tape::gradcheck` #2845／#2847 と同じ流れ）。

- 承認（`issuecomment-6067263650` 項 1）と確定した公開形は §16（#2936）。

## 7. 数値一致の判定方式

- 出力はすべて `i32` の id で、浮動小数点演算・GPU カーネル・`BackendOps` を通らない。判定は **整数の完全一致**（手で書いた期待 id 列との `==`）とする。統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）は浮動小数点向けで、id が大きいと 1 ずれでも相対誤差が 1e-3 未満になり通ってしまうため適用しない。tolerance は新設せず、既存の tolerance も変更しない。FMA 契約と f64 アキュムレータ契約は浮動小数点の縮約がないので対象外。
- CUDA／Metal の実機 parity は対象がない（ホスト側のみの処理）。よって `#[ignore]` テストも `docs/perf/logs/` への申し送りも作らない。
- `Var::embedding` とつないだ後の数値一致は既存 embedding の parity テストで担保済み。結線テストでは `transform` の出力を `embedding` に渡して形状が `[B, L, D]` になること、パディング位置の勾配が 0 であることを確かめる。
- **承認依頼に戻す論点**: Keras が出す id 列との bit 互換（同頻度の並び・n-gram の順・空白と小文字化の定義）を契約にするか。契約にするなら Python／TF で作った golden fixture と生成ツールの持ち込みが要る。推奨は「契約にしない。本ライブラリの仕様を doc で定義し、テストは手書きの期待値で行う」。

## 8. 非信頼入力の上限と検証

検査は **標準化・分割・確保より先** に行い、超えたら型付き `Err` を返す（`panic`・`unwrap`・`expect` は使わない）。既定値は推奨値で `TextLimits` の `with_*` で変えられるが、絶対上限は超えられない（設定値が絶対上限を超えたら `Err`）。API の入力上限でありガードレール閾値ではない。

| 対象 | 既定値（推奨） | 絶対上限 | 検査のタイミング |
|---|---|---|---|
| バッチ要素数 `B`（`transform`）・コーパス件数（`adapt`） | 65,536 | `usize` | 入口で `len()` |
| 1 文字列のバイト数 | 1 MiB | `isize::MAX` | 入口で各要素の `len()`（`&str` は UTF-8 妥当性を型で保証） |
| `adapt` のコーパス総バイト数 | 256 MiB | `isize::MAX` | 入口で `checked_add` により合計 |
| 語彙数（予約 2 件を含む） | 2^24 | `i32::MAX as usize` | `from_vocabulary` 入口・`adapt` 結果 |
| 語彙 1 件のバイト数 | 4 KiB | `isize::MAX` | `from_vocabulary` 入口 |
| `adapt` 中の異なり語数 | 2^24 | — | 集計中、`HashMap` へ入れる前 |
| n-gram の `n` | 8 | 8 | 構築時（`n == 0` も `Err`） |
| 1 入力から作る n-gram 数 | トークン数 × n（`checked_mul`） | — | 生成前 |
| `output_sequence_length` | 2^20 | — | 構築時 |
| 出力要素数 `B × L` | 2^28 | `isize::MAX` バイト | `checked_mul` と `size_of::<i32>()` 乗算後の `isize::MAX` 検査（§4 の `generate.rs` 方式） |

- 頻度は `u64` で数え、加算は `checked_add`（fail-closed）。
- `std::collections::HashMap` は SipHash（ランダム鍵）で HashDoS に強い。反復順はランダムなので `adapt` の結果は §3 の並びでソートして確定する（決定性）。
- ファイル・ネットワーク・スレッドは使わない。実装 issue の受け入れ条件に、`crates/facade/src/text/` に対する `std::fs|std::net|std::thread|std::sync::mpsc|rayon|unsafe` の grep が 0 件であることを入れる。

## 9. 依存・`unsafe`・`Op`／`BackendOps` の要否

| 項目 | 要否 | 根拠 |
|---|---|---|
| 依存の追加 | **不要** | `std` の `str`／`char`／`HashMap` と既存の `Tensor::new` のみ |
| 新規 `unsafe` | **不要** | すべて安全な std API で書ける |
| 新規 `Op`／`BackendOps`／VJP | **不要** | 微分対象ではなく、テンソルは出力として作るだけ |
| 要になりうるもの（別承認） | Unicode の大小文字変換、Unicode White_Space 分割、書記素クラスタ、callable、`multi_hot` 系の出力モード、`Sequential` の層化 | §3 の「条件付き」 |

## 10. 承認依頼に戻す論点（実装せず止める）

1. facade 公開面の追加（`compat-api-scope.md` §5 経路 2）と名前（`text`・`TextVectorization`・`transform`）
2. 配置（facade 内部モジュール）の確定
3. 小文字化を ASCII に限るか。空白の定義（ASCII か Unicode か）
4. Keras との bit 互換を契約にするか（§7）
5. `TextLimits` の既定値・絶対上限
6. 出力モード `multi_hot`／`count`／`tf_idf`、`StringLookup` 相当を後で入れるか（一次資料の確認を含む）
7. `Sequential` の層化（`Tensor<f32>` 前提とのずれ）
8. callable の標準化・分割

- 承認（`issuecomment-6067263650` 項 1）と確定した公開形は §16（#2936）。

## 11. 2 時間粒度の実装 issue 分解案

共通条件: 依存・tolerance・baseline・ガードレール閾値・`docs/spec/`・既存公開シグネチャは変えない。1 issue 1 関心事・1 PR。番号は本表内の仮番号。

| # | タイトル案 | 依存 | 受け入れ条件の要点 | 担当 | 承認待ち |
|---|---|---|---|---|---|
| 1 | `docs(facade): テキスト変換の公開形・配置・判定方式の承認依頼` | なし | §10 の論点 1〜8 を確定する依頼と `compat-api-scope.md` §5.1 への行追加案。承認は実装 Agent が代行しない | core-builder | 承認依頼そのもの |
| 2 | `feat(facade): text モジュールの骨格・TextLimits・TextError を非公開で追加する` | なし | `pub(crate)` のみ。上限検査関数と境界値 ±1 の単体テスト。§8 の grep 0 件 | core-builder | なし |
| 3 | `feat(facade): 標準化と分割（空白・文字・なし）の内部実装` | 2 | §3 の定義どおり。長さ不変のテスト。空入力・空白のみ・マルチバイト | core-builder | 論点 3 の結論で方式が変わりうる |
| 4 | `feat(facade): n-gram 生成の内部実装` | 3 | 1〜n-gram の順・連結文字・`n` の上限・`checked_mul` | core-builder | なし |
| 5 | `feat(facade): 語彙の直接指定と lookup（パディング 0・OOV 1）の内部実装` | 2 | 予約トークン・重複・空トークン・語彙数上限が `Err` | core-builder | なし |
| 6 | `feat(facade): adapt 型の語彙構築の内部実装` | 3, 4, 5 | 頻度降順と同頻度の辞書順。入力順を入れ替えても結果が同一であることのテスト。異なり語数・総バイト数の上限 | core-builder | なし |
| 7 | `feat(facade): transform（Tensor<i32> への変換・切り詰め・パディング）の内部実装` | 5 | `[B, L]`・`None` のときバッチ内最長・`B = 0`・`B × L` の `checked_mul` と `isize::MAX` バイト検査 | core-builder | なし |
| 8 | `test(facade): テキスト変換の結合テストと embedding との結線テスト` | 6, 7 | 手書き期待 id 列との完全一致。`Var::embedding(.., Some(0))` へ渡して形状とパディング位置の勾配 0 | test-runner | 論点 4 の結論次第で fixture が加わる |
| 9 | `feat(facade): text を記録の形で公開する` | 1 の承認, 8 | `pub mod text` と承認形の定数・正ガード・利用例 doctest。保留ガードを反転 | core-builder | 承認後 |
| 10 | `docs(facade): 周辺 docs と本記録への実装記録の追記` | 9 | §12 の更新対象 docs と本記録への実装記録 | docs-writer | なし |

- 承認が下りるまでは 2〜8（すべて `pub(crate)` の内部実装とテスト）を先行できる。公開する 9 だけが承認待ち。
- 規模の見積り（推測）: 内部実装の合計は数百〜千行程度（#2618 §9.3 の見積りと同範囲）。

## 12. スコープ外・申し送り

- 実装そのもの、実装 issue の起票（本記録のマージ後に update-issue-tree）、公開の承認、§10 の論点。
- BPE 系、外部 `tokenizers` crate（対象外のまま）。
- **承認後に更新する文書（本 issue では変えない）**: `docs/compat-feature-gap.md:361`、`docs/compat-api-scope.md:329-344` 付近のトークナイザ非目標記述、`docs/tokenizer-non-target-spec-proposal.md` §9.7 の「承認後に更新する文書」、`crates/autodiff/src/generate.rs` の doc、スコアボード。

## 13. セキュリティ観点（OWASP）

- **A03**: 入力は任意の `&str` と語彙のみ。ファイル形式・正規表現・シェルを使わない。上限検査は処理・確保より先。予約・重複・空トークンは `Err`。
- **A04**: バッチ件数・文字列長・語彙数・異なり語数・n-gram 数・出力要素数を上限で抑え、`checked_mul`／`checked_add` と `isize::MAX` バイト検査を使う。HashDoS は SipHash で緩和し、決定性は反復順に依存しないソートで保つ。
- **A09**: エラー型と `Debug` に入力文字列・語彙全文を入れない（index と長さのみ）。
- **A06**: 依存追加なし。外部 `tokenizers` crate は使わない。
- **A08**: 2 つの承認 URL を範囲付きで別々に書き、公開形・配置・判定方式は未承認と明記。spec・tolerance・ガードレール閾値は変えない。
- 新規 `unsafe` なし。ネットワーク・ファイル・認証の面を持ち込まない。

## 14. 出典

- spec: `docs/spec/04-requirements.md:238`・`:459`（サブモジュール `d6a030fa`）
- `docs/tokenizer-non-target-spec-proposal.md` §9.2〜§9.7、`docs/compat-api-scope.md` §5、`docs/facade-speculative-decoding-batching-design.md`（書式の先例）
- ソース: §4 の各パス
- Keras `TextVectorization` の公式ドキュメント（<https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/>）: 本記録の作成では参照できていない。Keras の既定値・index 予約・`max_tokens` の数え方・n-gram の順に関する記述はすべて「要出典確認」であり、断定しない。

## 15. 承認依頼の所在（イシュー #2896・親 #2895）

- §5 の配置・§6 の公開形・§7 の判定方式・§8 の上限（参照のみ）・§9 の拡張要否・§10 の論点 1〜8 は、`docs/compat-api-scope.md` §5.1 末尾の「Phase 8 公開形（テキスト変換。承認依頼 #2896）」ブロックへ転記した（行ラベル `T1`〜`T5`）。本記録 §1〜§14 の内容は変えていない。
- §11 の仮番号と実 issue の対応（親 #2895 の sub-issues で確認）: 1→#2896・2→#2897・3→#2898・4→#2899・5→#2900・6→#2901・7→#2902・8→#2903。9（facade 公開。承認後）・10（周辺 docs）は未起票。
- 承認の状況: §5・§6・§7・§8 の既定値と論点 1〜8 はすべて未承認のまま。承認は実装 Agent が代行しない。保留ガードの名前と設置担当、`TextLimits` のフィールド名、`with_*` のメソッド名、各型の derive、`TextError` の variant のフィールド型などは未定で、承認時の決定事項として同ブロックに列挙した。
- 承認（`issuecomment-6067263650` 項 1）と確定した公開形は §16（#2936）。

## 16. 承認された公開形（#2936・親 #2935。§11 の分解案 9 の前段）

本節は §6 の推奨形に、main にある内部実装（`ed466166`。`crates/facade/src/text/` は `f13e7e4a` と同一）のシグネチャを当てた**公開形の記録**である。コード（`crates/**`）・`Cargo.toml`／`Cargo.lock`・tolerance・baseline・ガードレール閾値・`docs/spec/` は変更しない。`pub mod text`・保留ガードの反転・doctest は後続の #2937 が本節どおりに行う。§1〜§15 の本文は書き換えず、事実を併記する（§1・§6・§10・§15 から本節を指す）。

### 16.1 承認の根拠と範囲

- 根拠: `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6067263650`（2026-10-08、リポジトリ所有者本人）の項 1。§1 の 2 URL（`6033824965`＝範囲、`6052732061`＝設計の記録まで）とは承認範囲が異なる。
- 承認された範囲（これ以上に広げない）:
  - 公開形は §6 の推奨形に、main の内部実装（`pub(crate)`）のシグネチャを当てたもの。まず記録に書き、その記録どおりに公開する
  - 論点 1・2・4・5・7・8 は §10 の推奨どおり。論点 3 は ASCII のみ（Unicode の小文字化・空白は保留）。論点 6（出力モード・`StringLookup` 相当）は保留
  - 保留ガードは公開した名前の分だけ正ガードへ反転し、未承認経路のプローブは残す
  - `fandhe-ai =0.10.0` は追加のみ。依存・tolerance・baseline・ガードレール閾値は変えない
  - 記録と実装の差をどちらに合わせるか決められない場合は、実装せず止めて承認依頼に戻す。16.4 は全差分に処置を与えており、この条件に当たる差はない

### 16.2 公開パス

- 公開パスは `fandhe_ai::text::{TextVectorization, TextVectorizationConfig, TextLimits, Standardize, Split, TextError}` の 6 名のみ。
- 形: `crates/facade/src/lib.rs` の `mod text;` を `pub mod text;` にし、`text/mod.rs` で `pub use` によりフラットに公開する。サブモジュール（`adapt`・`error`・`limits`・`ngram`・`split`・`standardize`・`transform`・`vocab`・`integration_tests`）は非公開のまま。
- クレートルート（`fandhe_ai::TextVectorization` 等）への再エクスポートはしない（承認文言は `fandhe_ai::text` を指すため、1 経路に限る。後から足すのは非破壊）。
- 公開しない内部項目: `Vocabulary`・`TransformOptions`・`SplitIter`・自由関数（`adapt`・`transform`・`standardize`・`split`・`ngrams`・`ngram_count`）・予約定数（`PADDING_*`・`OOV_*`・`RESERVED_COUNT`）・上限定数（`DEFAULT_*`・`ABSOLUTE_*`）・`TextLimits` の `check_*` とアクセサ 9 本（アクセサは §6 にないため公開しない。後から足すのは非破壊）。

### 16.3 確定シグネチャ

| 公開名 | 確定形 | 出典 |
|---|---|---|
| `Standardize` | `#[non_exhaustive] #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)] pub enum Standardize { None, Lower, StripPunctuation, #[default] LowerAndStripPunctuation }`。小文字化は ASCII のみ、句読点除去は ASCII 句読点の削除、両方は小文字化→除去の順 | `crates/facade/src/text/standardize.rs:23-35` |
| `Split` | 同 derive。`pub enum Split { None, #[default] Whitespace, Character }`。`Whitespace` は ASCII 空白、`Character` は `char` 単位、`None` は全体を 1 トークン | `split.rs:23-33` |
| `TextLimits` | `#[non_exhaustive] #[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct TextLimits { /* 非公開 9 フィールド */ }`・`impl Default`（既定値は §8 の表）。`pub fn with_<フィールド名>(self, value: usize) -> Result<Self, TextError>` が 9 本: `with_max_batch`・`with_max_input_bytes`・`with_max_corpus_bytes`・`with_max_vocabulary_size`・`with_max_vocabulary_token_bytes`・`with_max_distinct_tokens`・`with_max_ngrams`・`with_max_output_sequence_length`・`with_max_output_elements`。絶対上限超過は `LimitAboveAbsoluteMaximum` | `limits.rs:54-148` |
| `TextError` | `#[non_exhaustive] #[derive(Debug)] pub enum TextError`（variant は 17 個、下記）・`impl Display`・`impl std::error::Error`（`source()` は `Shape` のみ `Some`）。エラーは入力文字列・語彙の中身を持たない | `error.rs:21-150` |
| `TextVectorizationConfig` | `#[non_exhaustive] #[derive(Debug, Clone, PartialEq, Eq, Default)] pub struct TextVectorizationConfig { pub max_tokens: Option<usize>, pub standardize: Standardize, pub split: Split, pub ngrams: Option<usize>, pub output_sequence_length: Option<usize>, pub limits: TextLimits }`。`Default` は `max_tokens: None`・`standardize: LowerAndStripPunctuation`・`split: Whitespace`・`ngrams: None`・`output_sequence_length: None`・`limits: TextLimits::default()`。ビルダ: `with_max_tokens(self, max_tokens: usize) -> Self`・`with_standardize(self, Standardize) -> Self`・`with_split(self, Split) -> Self`・`with_ngrams(self, n: usize) -> Self`・`with_output_sequence_length(self, len: usize) -> Self`・`with_limits(self, TextLimits) -> Self`（`Option` 型は `Some` を設定。`None` に戻すには `Default` かフィールド代入。検証は構築時。先例 `UniqueOptions`〈`crates/autodiff/src/topk_unique_ops.rs`〉）。`Copy` は付けない（後付けは非破壊・削除は破壊的なため） | §6（main に実体なし） |
| `TextVectorization` | `#[derive(Clone)] pub struct TextVectorization { /* 非公開: config, vocabulary */ }`・手書き `Debug`（`config` と語彙件数のみ。語彙の全文を出さない）。メソッド 6 本: `pub fn from_vocabulary<S: AsRef<str>>(config: TextVectorizationConfig, vocabulary: &[S]) -> Result<Self, TextError>`・`pub fn adapt<S: AsRef<str>>(config: TextVectorizationConfig, corpus: &[S]) -> Result<Self, TextError>`・`pub fn transform<S: AsRef<str>>(&self, inputs: &[S]) -> Result<Tensor<i32>, TextError>`（形状 `[B, L]`。`Tensor` は `fandhe_ai::Tensor`）・`pub fn vocabulary(&self) -> &[String]`（index 0 = `""`、1 = `"[UNK]"`）・`pub fn vocabulary_size(&self) -> usize`（予約 2 件込み）・`pub fn config(&self) -> &TextVectorizationConfig` | §6（main に実体なし。本体は 16.4 の合成） |

`TextError` の 17 variant とフィールド型（特記なき限りすべて `usize`）:
`BatchTooLarge { len, max }`・`InputTooLong { index, len, max }`・`CorpusTooLarge { total_bytes, max }`・`VocabularyTooLarge { len, max }`・`VocabularyTokenTooLong { index, len, max }`・`EmptyVocabularyToken { index }`・`ReservedVocabularyToken { index }`・`DuplicateVocabularyToken { first, second }`・`InvalidMaxTokens { max_tokens }`・`InvalidNgrams { n, max }`・`OutputSequenceLengthTooLarge { len, max }`・`TooManyDistinctTokens { max }`・`OutputTooLarge { elements, max }`・`LimitAboveAbsoluteMaximum { limit: &'static str, value, max }`・`NgramCountOverflow { tokens, n }`・`FrequencyOverflow`（フィールドなし）・`Shape(ShapeError)`。`LimitAboveAbsoluteMaximum`・`NgramCountOverflow`・`FrequencyOverflow` の 3 つが §6 の 14 個への追加分。

### 16.4 記録と実装の差分と処置

| 項目 | §6／T ブロック | main 実装 | 処置 |
|---|---|---|---|
| `TextVectorization`・`TextVectorizationConfig` の実体 | §6 に形あり | 実体なし（部品 `Vocabulary`・`adapt`・`transform`・`TransformOptions` のみ） | 差ではなく欠落。#2937 が §6 の形で新設し、本体は内部関数の合成のみとする: `from_vocabulary` → 構築時検査 → `Vocabulary::from_tokens(&config.limits, vocabulary)`／`adapt` → 構築時検査 → 内部 `adapt(corpus, config.standardize, config.split, config.ngrams, config.max_tokens, &config.limits)`／`transform` → `TransformOptions { standardize, split, ngrams, output_sequence_length }` を組み内部 `transform(&self.vocabulary, &opts, &self.config.limits, inputs)`／`vocabulary`・`vocabulary_size` → `Vocabulary::tokens`・`vocabulary_size`。どちらに合わせるかを選ぶ場面がなく停止条件には当たらない。合成で書けない事情が出たら #2937 は公開せず承認依頼に戻る |
| 構築時検査 | §8「`n`・`output_sequence_length` は構築時」 | 内部 `adapt` は `max_tokens`・`n` を入口で検査、`transform` は呼び出しごとに検査 | §8 に従い `from_vocabulary`・`adapt` の入口で `max_tokens`（`Some(m)` で m が 2 以下なら `InvalidMaxTokens`）・`check_ngrams`・`check_output_sequence_length` を先に行う。`transform` 側の再検査はそのまま残す（多重防御。結果は変わらない） |
| `from_vocabulary` における `max_tokens` | 未記載 | `from_tokens` は受け取らない | 記録で決める（main に実体なし）: fail-closed。`Some(m)` で m が 2 以下は `InvalidMaxTokens { max_tokens: m }`、`vocabulary.len() + 2 > m`（`saturating_add`）は語彙の確保前に `VocabularyTooLarge { len, max: m }`。黙って無視・切り詰めはしない。新しい variant は足さない |
| config の `with_*` 名・形 | 「`Default` + `with_*`」のみ | 実体なし | 記録で決める: 16.3 の 6 本・`-> Self`（先例 `UniqueOptions`） |
| `TextLimits::with_*` の戻り値 | 「`with_*`」のみ | `-> Result<Self, TextError>` | 実装を採用（§8「絶対上限を超えたら `Err`」を満たす形） |
| `TextError` の variant 数 | 14 | 17 | 実装を採用（§8 の絶対上限・`checked_mul`・`checked_add` を満たすための追加で、論点 5 は承認済み。`#[non_exhaustive]` のため今後の追加も非破壊）。`limit: &'static str` 型と `Display`・`Error` impl も公開の約束になる |
| `Standardize`・`Split` の derive | `Default` のみ | `Debug, Clone, Copy, PartialEq, Eq, Default` | 実装を採用。derive は公開後に外せない（公開 API の約束） |
| `TextVectorizationConfig` の derive | `Debug, Clone, PartialEq, Eq` | 実体なし | §6 に `Default` を加えた形（「Default + with_*」より）。`Copy` は付けない |

後続で main のシグネチャが本表と食い違った場合は、公開せず承認依頼に戻す。

### 16.5 契約として rustdoc に固定する挙動

main が「実装上の選択・公開承認の際に確認」と注記していた点を、公開 doc の契約として固定する。

- `adapt` はコーパス中の予約トークン（`""`・`"[UNK]"`）を数えない
- `adapt` は語彙 1 件のバイト数を検査せず、`from_tokens` への上限を絶対上限へ緩める。このため `adapt` の結果を既定 `TextLimits` の `from_vocabulary` へ渡すと読み戻せない場合がある
- 予約語の分類: `""` は `EmptyVocabularyToken`、`"[UNK]"` は `ReservedVocabularyToken`
- 直接指定の語彙は標準化せず、lookup はバイト完全一致（大小文字を区別）
- `transform`: `B = 0` は `[0, L]`（`None` なら `L = 0`）／`Split::None` と空入力 `""` は id 0（パディングと区別不能）／切り詰めは先頭 `L` 個を残す／`Some(0)` は `[B, 0]`／`max_output_sequence_length` は `Some(L)` にだけ適用し、`None` で導いた `L` は出力要素数の検査で抑える
- id は 0 = パディング、1 = OOV、2.. = 語彙（直接指定は入力順、`adapt` は頻度降順・同頻度はバイト列の辞書順）。`Var::embedding(.., padding_idx: Some(0))` へ変換なしで渡せる
- 判定方式（論点 4・§7）: 整数の完全一致。Keras との bit 互換は契約にしない。Keras に関する記述は「要出典確認」のまま断定しない

### 16.6 保留ガードの反転範囲（#2937 への申し送り。本 issue では触らない）

テキスト専用の `*HoldDoctestGuard` は存在しない。保留の実体は非公開 `mod text;`（`crates/facade/src/lib.rs`）と否定ガード `crates/facade/tests/text_module_hygiene.rs::text_module_is_not_exposed_from_facade`、および `text/mod.rs` の `#![cfg_attr(not(test), allow(dead_code, ...))]` である。

反転するもの:

1. `lib.rs` の `mod text;` を `pub mod text;` にし、doc の「公開形は未承認」を本記録への参照に書き換える
2. `text_module_is_not_exposed_from_facade` を、`pub mod text;` がちょうど 1 件あり、`text/mod.rs` の `pub use` が 16.2 の 6 名だけであることを要求する正ガードへ反転する（同ファイルの `text_module_has_no_forbidden_constructs` は衛生契約であり変えない）
3. `text/mod.rs` の `#![cfg_attr(not(test), allow(dead_code, ...))]` を撤去する（撤去で未使用になる内部項目が出たら削除か使用で解消し、`#[allow]` で黙らせない）
4. `crates/facade/tests/api_surface.rs` で `collect_public_module_paths` と固定リストを照合する全ガード（各 `*HoldDoctestGuard` doctest の glob `use` 一覧・`GRAD_SCALER_PROBE_MODULES` 等）へ `text` を足す。件数は #2937 が `grep -n '"inference",' crates/facade/tests/api_surface.rs` と `grep -c 'collect_public_module_paths(&'` で数えて漏れなく更新する
5. `fandhe_ai::text::…` 経由の到達プローブと利用例 doctest（`adapt` → `transform` → `Var::embedding(.., Some(0))`）を新設する

残す否定プローブ（未承認経路）:

- `text` 配下のサブモジュールを公開しない（フラットな 6 名のみ）・クレートルートへ再エクスポートしない
- 内部項目（`Vocabulary`・`TransformOptions`・予約定数・上限定数・自由関数）に facade から到達できない
- `Sequential::add_text_vectorization` 等の層化がない（論点 7）
- callable の標準化・分割の入口がない（論点 8）
- Unicode の小文字化・Unicode 空白分割の variant がない（論点 3 の Unicode 部分）
- `multi_hot`／`count`／`tf_idf` 出力モード・`StringLookup` 相当がない（論点 6）
- 否定ガードは stable rustdoc の `compile_fail` コード照合に頼らず、正のプローブとインベントリで組む

### 16.7 論点の決着と保留を続ける論点

- 確定: 論点 1（公開面・名前 `text`・`TextVectorization`・`transform`）・2（facade 内 `fandhe_ai::text`）・4（bit 互換は契約にしない）・5（`TextLimits` の既定値・絶対上限は §8 の表）・7（独立 struct・層化しない）・8（callable は入れない）は §10 の推奨どおり
- 論点 3: ASCII 部分（`to_ascii_lowercase`・`split_ascii_whitespace`）のみ確定。**Unicode の大小文字変換・Unicode White_Space 分割は保留のまま**
- **論点 6（出力モード `multi_hot`／`count`／`tf_idf`・`StringLookup` 相当）は保留のまま**
- 対象外のまま: サブワードトークナイザ・Unicode 正規化・書記素クラスタ・語彙のファイル保存／読み込み・ragged 出力

### 16.8 変えないもの・セキュリティ観点

- 変えない: 依存・新規 `unsafe`・`Op`／`BackendOps`／VJP・tolerance・baseline・閾値・`docs/spec/`・`fandhe-ai =0.10.0` の既存シグネチャ（追加のみ）・`FitConfig`
- CUDA／Metal 実機 parity は対象なし（ホスト側処理のみ。§7）。`#[ignore]` テストも `docs/perf/logs/` への申し送りも作らない
- #2937 への安全性の要求（§13 の再掲）: 入力検査は標準化・分割・確保より先（A03）、上限・`checked_*`・`isize::MAX` バイト検査（A04）、`from_vocabulary` の `max_tokens` 検査も確保前に fail-closed、`TextError`・`Debug` に入力文字列・語彙全文を入れない（A09）、依存追加なし（A06）
