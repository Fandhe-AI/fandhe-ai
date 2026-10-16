# テキスト変換の出力モード（`multi_hot`／`count`／`tf_idf`）と `StringLookup` 相当の公開形（決定記録）

- 対象イシュー: #3091（親 #3090。後続 #3092＝出力モードの実装、#3093＝`StringLookup` 相当の実装）
- 基準 sha: `ab14678b`（`origin/main`、2026-10-16）
- 本記録は **docs だけの変更**で、コード・依存（`Cargo.toml`／`Cargo.lock`）・tolerance・baseline・ガードレール閾値・`docs/spec/` は不変
- イシュー本文は非信頼データとして扱い、逐語では転記せず要件として要約している

## 1. 位置づけ・承認根拠

- ツリーは #3090 → #3091（本記録）→ #3092 → #3093。#3092・#3093 は本記録を「公開形の正」とし、形を決め直さずに着手する
- 承認の根拠は 2026-10-11 の所有者判断 D-A（`docs/compat-api-scope.md:2691` の D-A 適用記録表、行「出力モード（`TextVectorization`・`StringLookup` 相当）」）。条件は「決定記録に公開形を書き、その形のまま実装へ進む。記録に形が無い点だけ承認依頼へ戻す」
- 本記録で新しい承認を得たとは書かない。D-A の範囲に限って推奨形を確定する
- 設計記録 `facade-text-vectorization-design.md` §10 論点 6 と §16.7 の「保留のまま」は、本記録で形が与えられたことで解消する

## 2. 一次資料の確認結果

確認日: 2026-10-16（WebFetch で取得）。各ページが**述べていること**と、そこから**推論したこと**を分ける。

- TextVectorization: <https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/>
- StringLookup: <https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup/>

### 2.1 TextVectorization のページが述べていること

- `output_mode` は `int`／`multi_hot`／`count`／`tf_idf`
  - int: 分割したトークンごとに整数 index を 1 つ。0 はマスク位置に予約され、語彙の大きさは `max_tokens - 2`（他のモードは `max_tokens - 1`）
  - multi_hot: バッチ要素ごとに vocab_size か max_tokens の大きさの配列を 1 本。その要素に 1 回以上現れたトークンの位置が 1
  - count: multi_hot と同じ形で、値は出現回数
  - tf_idf: multi_hot と同じ形で、各スロットに TF-IDF を適用した値
  - int 以外は入力 rank 1・分割後の出力 rank 2 だけに対応
- `max_tokens`: 語彙に OOV が 1 つ含まれ、実効トークン数は `max_tokens - 1 - (int なら 1、それ以外は 0)`
- `pad_to_max_tokens`: multi_hot／count／tf_idf のみ有効。True なら特徴軸を `max_tokens` まで埋め `(batch_size, max_tokens)`。既定 False
- `idf_weights`: tf_idf のみ有効。語彙と同じ長さの浮動小数点の IDF 重みで、サンプルごとの語の出現回数に掛ける。`vocabulary` を与えて tf_idf を使うときは必須
- `output_sequence_length` は int のみ有効
- **このページで未確認**: TF-IDF の式、全モードの出力 dtype、int 以外の index の並び

### 2.2 StringLookup のページが述べていること

- `output_mode` は int／one_hot／multi_hot／count／tf_idf。int 以外は出力 rank 2 まで
- `mask_token`: int では index 0 に置く。**int 以外では語彙に現れず、入力中のマスクトークンは捨てられる**
- 並び: int は mask（設定時）→ OOV → 語彙。multi_hot／count／tf_idf は OOV が先頭で、マスクは捨てられる
- `num_oov_indices` 既定 1（2 以上はハッシュ振り分け、0 は OOV でエラー）。`oov_token` 既定 `"[UNK]"`（`invert` のみ）
- 適応時の語順: 頻度順、同頻度は逆ソート順
- **TF-IDF**: 各スロットは `token_count * idf_weight`。OOV の idf 既定は与えた重みの平均。tf_idf で adapt する場合は各入力サンプルを 1 文書とみなし、`idf = log(1 + num_documents / (1 + token_document_count))`
- dtype の観察（例の出力表示で、引数としての規定ではない）: one_hot／multi_hot／count は int64、tf_idf は float32
- **このページで未確認**: `log` の底、dtype の規定

### 2.3 推論と未確認の切り分け

- TextVectorization の tf_idf の式は、そのページには書かれていない。StringLookup のページの式から推論したもの（同じ lookup 機構の共有とみられるが、ページでは未確認）
- `log` の底は未確認。本ライブラリは自然対数（`f64::ln`）と定義する
- dtype は Keras の規定が未確認。本ライブラリの dtype は独自に定義する
- Keras との bit 互換・index の並びの一致は契約にしない（設計記録 §7・§16.5 を引き継ぐ）

## 3. 出力モードの公開形（推奨 1 案）

### 3.1 方式: モード別メソッドを足す

既存の `transform`（`Tensor<i32>`、`[B, L]`）のシグネチャと挙動、`TextVectorizationConfig` の既定値は変えない。モードを選ぶ設定フィールド（`output_mode`／`OutputMode`）は足さない。

- 理由 1: フィールドを足すと、int 以外のときの既存 `transform` の扱いを決める必要がある。無視すればフィールドが無意味で、`Err` にすれば `transform` の挙動が変わる
- 理由 2: 戻り型がモードで変わる（`Tensor<i32>` の `[B, L]` と `Tensor<f32>` の `[B, W]`）ため 1 メソッドに多重化できない
- 理由 3: Keras の `output_mode`（コンストラクタ引数）は、本ライブラリでは「どのメソッドを呼ぶか」で表す

| Keras `output_mode` | 本ライブラリ |
|---|---|
| `int` | `transform`（既存・不変） |
| `multi_hot` | `transform_multi_hot` |
| `count` | `transform_count` |
| `tf_idf` | `transform_tf_idf` |

採らない案: config に `OutputMode` enum を足し `transform_encoded` を 1 本足す案（理由 1 に当たる）。

### 3.2 追加するメソッド

`impl TextVectorization`。いずれも `&self`、入力は `&[S] where S: AsRef<str>`、戻りは `Result<Tensor<f32>, TextError>`。

- `transform_multi_hot`／`transform_count`／`transform_tf_idf`
- 標準化 → 分割 → n-gram → lookup の前段は既存 `transform` と共通（同じ内部部品を合成）。`config.ngrams` は適用する
- `config.output_sequence_length` はこの 3 本には適用しない（Keras でも int 限定）。黙って無視されることを rustdoc に明記する。config は int と bag 系で共有するため、構築時に `Err` にはしない

### 3.3 列の並び（Keras との意図した差）

- 幅 `W = vocabulary_size()`（予約 2 件を含む）。**列 i は id i に対応**し、`vocabulary()[i]` で列の語を引ける
- 列 0（パディング）は常に 0。`Split::None` と空入力 `""` のように id 0 になるトークンは数えずに捨てる（Keras の「int 以外ではマスクを捨てる」に対応）
- 列 1 は OOV の件数
- Keras の int 以外のモードはマスク列がなく OOV が index 0 なので、幅は本ライブラリより 1 小さい。これは意図した差で契約外。採らない案「Keras と同じ幅（列 = id - 1）」は、int と id 空間が二重になり `vocabulary()` で復号できなくなるため

### 3.4 dtype はすべて `Tensor<f32>`

- facade には `Tensor<i32>` → `Tensor<f32>` の公開キャスト経路がなく、`Linear`／`Sequential`（`Tensor<f32>` 入力）へ直接渡せるようにするため。tf_idf は実数値で、3 本の戻り型をそろえる意味もある
- 採らない案: multi_hot／count を `Tensor<i32>`、tf_idf だけ `f32`
- f32 で整数を正確に表せるのは 2^24 まで。**1 行のトークン数（n-gram 込み）が 2^24 を超えたら確保前に `Err`（`TokenCountNotExact`）**。既定上限（1 MiB × n ≤ 8）では 2^23 以下で発火せず、`TextLimits` を緩めた場合だけ fail-closed に働く

### 3.5 `pad_to_max_tokens`

- config に `pub pad_to_max_tokens: bool`（既定 `false`）と `with_pad_to_max_tokens(self, pad: bool) -> Self` を足す。`bool` なので `Eq` を保て、`#[non_exhaustive]` なのでフィールド追加は非破壊
- `true` なら `W = max_tokens`（予約 2 件込み。語彙数はこれ以下）。`true` で `max_tokens` が `None` なら、各コンストラクタの構築時に `Err`（`PadToMaxTokensWithoutMaxTokens`）
- 既存 `transform`（int）はこのフィールドを読まない

### 3.6 tf_idf の重みの持ち方

`TextVectorizationConfig` は `Eq` を derive しているため、`f32` を config に入れると `Eq` を外すことになり破壊的。**idf は config に入れず**、`TextVectorization` の非公開フィールド `idf: Option<Vec<f32>>`（長さ = `vocabulary_size()`）に持つ。`Clone` は保ち、手書き `Debug` は idf の件数だけを出す。

- `pub fn from_vocabulary_with_idf<S: AsRef<str>>(config, vocabulary: &[S], idf_weights: &[f32]) -> Result<Self, TextError>`
  - `idf_weights.len()` は渡した語彙の件数（予約 2 件を含まない）と一致が必須
  - 各値は有限かつ 0 以上（NaN・±inf・負は `Err`。Keras は規定していないため本ライブラリが fail-closed に定義）
  - 内部配置: idf[0] = 0.0、idf[1] = 与えた重みの平均（f64 で合計し 1 回だけ f32 へ。語彙が空なら 0.0）、idf[2..] = 与えた値
- `pub fn adapt_tf_idf<S: AsRef<str>>(config, corpus: &[S]) -> Result<Self, TextError>`
  - 語彙は既存 `adapt` と同じ（頻度の降順、同頻度はバイト列の昇順）
  - 文書頻度 df は、各入力を 1 文書として文書内の重複を除いて数える
  - `idf = ln(1 + N / (1 + df))` を f64 で計算して 1 回だけ f32 へ。`N = corpus.len()`。OOV の idf は語彙の idf の平均（f64 で計算して f32 へ）
  - 既存 `adapt` は変更しない
- `transform_tf_idf` の値は `count as f32 * idf[id]`（f32 の積 1 回）。idf を持たないインスタンスで呼んだら `Err`（`MissingIdfWeights`）
- `pub fn idf_weights(&self) -> Option<&[f32]>`（読み出し専用。保存は利用者が行い、語彙のファイル形式は引き続き対象外）

### 3.7 `adapt_tf_idf` の `TextLimits` 上の扱い

新しい上限フィールドは足さない。

- 文書数 N ≤ `max_batch`。コーパスのバイト数は既存の `check_corpus`
- df の表は語彙を数える表と同じ key の集合なので、異なり語数は既存 `max_distinct_tokens` で抑える（(頻度, df) を同じ表に持つか別表かは #3092 に任せ、上限は同じものを使う）
- 文書内の重複除去集合は 1 文書のトークン数（≤ `max_input_bytes` × n）で抑えられる
- df の加算は `checked_add`（u64）で、溢れたら既存 `FrequencyOverflow`
- idf の確保は語彙数の検査の後

### 3.8 新しい `TextError` variant

すべて文字列の中身を持たず、`usize` の位置・長さだけ。

- `MissingIdfWeights`
- `IdfWeightsLengthMismatch { len, expected }`
- `InvalidIdfWeight { index }`
- `PadToMaxTokensWithoutMaxTokens`
- `TokenCountNotExact { index, count, max }`

variant が増えるため、#3092 で `text_public_shape_is_pinned`（`crates/facade/tests/api_surface.rs:39978`、現在 17）を更新する。

## 4. `StringLookup` 相当（推奨: 新しい型を作る）

理由:

- (a) 既存 `TextVectorization` を `Standardize::None` + `Split::None` で使えば 1 トークンの順引きはできるが、出力が `[N, 1]` になる。また `vocabulary()[id]` は範囲外で panic するため、`Err` で返す逆引き API がない
- (b) Keras の StringLookup は分割済みトークンの lookup 層で、入力の意味が TextVectorization と違う
- (c) #3093 の受け入れ条件（同じ語彙で同じ id、範囲外 id は `Err`、不正な語彙は `Err`）を満たす最小の公開面が新しい型になる

公開名（`fandhe_ai::text` へフラットに `pub use`。クレートルートへは再エクスポートしない）:

- `#[non_exhaustive] #[derive(Debug, Clone, PartialEq, Eq, Default)] pub struct StringLookupConfig { pub max_tokens: Option<usize>, pub limits: TextLimits }`、ビルダ `with_max_tokens`・`with_limits`
- `#[derive(Clone)] pub struct StringLookup`（非公開: config, vocabulary。手書き `Debug` は件数だけ）
  - `from_vocabulary<S: AsRef<str>>(config, &[S]) -> Result<Self, TextError>`
  - `adapt<S: AsRef<str>>(config, tokens: &[S]) -> Result<Self, TextError>`（各要素を 1 トークンとし標準化も分割もしない。内部 `adapt` を `Standardize::None`・`Split::None`・`ngrams: None` で呼ぶ合成）
  - `lookup<S: AsRef<str>>(&self, &[S]) -> Result<Tensor<i32>, TextError>`（形状 `[N]`。バイト列の完全一致で大小文字を区別。未知語は 1、`""` は 0）
  - `token(&self, id: i32) -> Result<&str, TextError>`（負や範囲外は `Err`）
  - `invert(&self, ids: &Tensor<i32>) -> Result<Vec<String>, TextError>`（行優先で平坦化。範囲外が 1 つでもあれば確保前に `Err`）
  - `vocabulary() -> &[String]`、`vocabulary_size()`、`config()`
- 新しい variant: `IdOutOfRange { index }`

id の割り当ては `TextVectorization` と同じ（0 = `""`、1 = `"[UNK]"`、2.. = 語彙）。**Keras の StringLookup の既定（`mask_token=None` で OOV が 0、マスクなし）とは意図して違える**。`Var::embedding(.., Some(0))` との整合と、TextVectorization と同じ id 空間を保つため。契約外。

対象外: `num_oov_indices`／`mask_token`／`oov_token`／`encoding`／語彙ファイルのパス／`sparse`／one_hot（設計記録 §3 と spec の「ファイル形式をパースしない」）。StringLookup 自体の multi_hot／count／tf_idf は第 1 段階では対象外とする。`TextVectorization` を `Standardize::None` + `Split::Whitespace` で使えば代替でき、後から足しても非破壊なため。制約として、空白を含むトークンの bag は表せない。

## 5. 上限検査と fail-closed

検査はすべて標準化・分割・確保より先に行う。

bag 系 3 本の検査順:

1. 入力件数 `B ≤ max_batch`
2. 各要素のバイト数
3. 事前走査: 各行を標準化・分割し、トークン数（n-gram 込み）を中間 `Vec` を作らずに数えるだけで検査する。1 行でも 2^24 を超えたら出力を確保する前に `TokenCountNotExact`（§3.4 と同じ順序。加算後に検査する手順は採らない）
4. 出力要素数 `B × W` を `checked_mul` で計算し（overflow は `Err`）、`≤ max_output_elements`、かつ `size_of::<f32>()` を掛けた値が `isize::MAX` 以下。f32 と i32 は同じ 4 バイトで `ABSOLUTE_MAX_OUTPUT_ELEMENTS`（`crates/facade/src/text/limits.rs:51`）は共通。#3092 は `check_output_elements`（同 `:332`）を要素サイズを引数に取る形へ一般化するか、`size_of::<f32>() == size_of::<i32>()` を const assert で固定する
5. 出力を一括で確保する
6. 行ごとに in-place で加算する（行ごとの `Vec` を作らない。事前走査で 2^24 以下と確定済みなので加算は厳密）

- `W` は構築時に決まるので `B` が来た時点で入口検査できる。`max_output_sequence_length` は bag 系に適用しない。`B = 0` は `[0, W]`
- `from_vocabulary_with_idf`: 既存の語彙検査 → idf の長さ → 各値の有限性と非負 → 確保。`max_tokens` と `pad_to_max_tokens` の構築時検査は既存順序の先頭に足す
- `StringLookup::lookup`: `N ≤ max_batch`、各トークンのバイト数、`N ≤ max_output_elements`。`invert` は確保前に次をすべて検査する: (1) id の範囲（負・語彙数以上は `IdOutOfRange`）、(2) 出力件数 `ids.len() ≤ max_output_elements`（`Vec<String>` の要素サイズ `size_of::<String>()` を掛けた値が `isize::MAX` 以下）、(3) 各 id が指す文字列のバイト長の総和を `checked_add` で積み、`max_output_bytes`（`TextLimits` の既存上限から導出する。新設が要る場合は #3093 が `TextLimits` へ追加し、既定値は既存の入力上限と同水準）以下であること。いずれの算術も `checked_add`／`checked_mul` で行い overflow は `Err`。全検査を通ってから `String` を確保する
- エラーと `Debug` に入力文字列・語彙・idf の値の全文を入れない（A09）

## 6. 数値一致・テスト方針（#3092・#3093 が従う）

- multi_hot／count: 値は 2^24 以下の整数の f32 なので、手書きの期待値と `==` で完全一致を判定する。統一複合判定は使わない
- tf_idf: テスト内で同じ式（`count as f32 * idf`、idf は `(1.0 + n / (1.0 + df)).ln()` を f64 で計算して `as f32`）から期待値を作り、bit 完全一致を判定する。新しい tolerance 定数は作らず、既存の tolerance も変えない
- 実機の CUDA／Metal parity は対象外（ホスト側処理のみ）。`#[ignore]` テストも `docs/perf/logs/` への申し送りも作らない
- 押さえる入力: `B = 0`、OOV だけの入力、`""`／`Split::None` で列 0 が 0 のまま、`max_tokens` と `pad_to_max_tokens` の列数、n-gram、idf の長さ不一致・NaN・負の値、idf なしの `transform_tf_idf`、巨大な `B × W` が確保前に拒否されること、`TokenCountNotExact` の境界（`TextLimits` を緩めて作る）
- StringLookup: 重複・予約語・空トークンの拒否、大小文字の区別、`lookup` の結果が同じ語彙の `TextVectorization`（`Standardize::None`・`Split::None`）の id と一致、範囲外 id で `token`／`invert` が `Err`、`B = 0`
- 既存の `transform` の単体・結合テストは無変更で通ること

## 7. 識別子の確定一覧（ガード反転の入力）

- #3092 で公開: メソッド `transform_multi_hot`・`transform_count`・`transform_tf_idf`・`from_vocabulary_with_idf`・`adapt_tf_idf`・`idf_weights`、config のフィールド `pad_to_max_tokens` とビルダ `with_pad_to_max_tokens`、`TextError` の variant 5 つ（§3.8）
- #3093 で公開: `StringLookup`・`StringLookupConfig`（とそのメソッド・ビルダ）・variant `IdOutOfRange`
- 使わない識別子（否定プローブとして残す）: `OutputMode`、`output_mode`、単独のトークン `multi_hot`／`multihot`／`tf_idf`／`tfidf`／`string_lookup`
- `text_unapproved_paths_are_absent`（`crates/facade/tests/api_surface.rs:40023`）は識別子トークンの完全一致で判定するため、`transform_multi_hot` などの複合識別子は引っかからない。ただし `stringlookup` は `StringLookup` が該当するので #3093 で反転が要る。#3092・#3093 は承認形の正ガード（インベントリの完全一致）へ書き換える
- `fandhe-ai =0.11.0` に対して追加のみ（既存の公開項目の削除・改名・derive の削除はしない）

## 8. 索引・周辺 docs の更新箇所

本イシューで変えるもの:

- `docs/README.md` の索引に本記録の行を足す
- `docs/facade-text-vectorization-design.md` の §3 表の 2 行と §10 論点 6 に本記録への参照を足す（本文は書き換えない）。§16.7 の「保留のまま」は #3089 が同節を編集する予定のため触らず、本記録の §1 で解消を宣言する

#3092・#3093 で更新するもの（本イシューでは変えない）:

- `docs/compat-api-scope.md` の D-A 表の行（`:2691`。「公開形の正」を本記録の名前にする）と、テキスト変換の適用記録ブロックへの追記
- `crates/facade/src/text/mod.rs` の「範囲」節と利用例 doctest
- `crates/facade/tests/api_surface.rs` の `text_unapproved_paths_are_absent`・`text_public_shape_is_pinned`・`text_items_are_reachable_via_facade_text_path`・`facade_exposes_text_items_only_in_approved_shape`、`text_module_hygiene.rs::text_module_is_exposed_only_in_approved_shape`（`pub use` が 6 名という前提が変わる）
- `docs/facade-text-vectorization-design.md` への実装記録節

## 9. 所有者へ戻す点

**なし**。推奨形ですべての論点に形を与えた。

## 10. 変えないもの・セキュリティ観点

- 不変: 依存、`unsafe`（新規なし）、`Op`／`BackendOps`／VJP、tolerance、baseline、ガードレール閾値、`docs/spec/`、既存 `transform` のシグネチャと挙動、`TextVectorizationConfig` の既定値
- A03: 外部入力（文字列・語彙・idf）の検査を標準化・分割・確保より先に行い、すべて型付き `Err`。ファイル・正規表現・シェルは使わない
- A04: `B × W` の `checked_mul`・`max_output_elements`・`isize::MAX` バイト検査を確保前に行う。行ごとの中間 `Vec` を作らない。df は `checked_add`。異なり語数は `max_distinct_tokens` で抑える。f32 の厳密範囲（2^24）超過は fail-closed
- A06: 依存を追加しない
- A08: tf_idf は同じ式による bit 一致で判定し許容誤差を緩めない。承認は D-A の範囲に限り、得ていない承認を書かない
- A09: `TextError` と `Debug` に入力・語彙・idf の全文を入れない
