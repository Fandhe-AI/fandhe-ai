# トークナイザ非目標の spec 明記提案（(b) 形式提案文案・実装しない）（#2086）

基準コミット: `fb6babdebfe12905d3d5878faee9ca00914ee00a`（2026-09-22）。`docs/spec` submodule ポインタ `c5cf1ed5a547f47dd60d168c926bd8d376ee31da`。`file_path:line` は同コミット時点のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。

#2618 最新化基準（§9）: `origin/main` `1364c261d8d9ce80a63805dc98faae81fc725fe9`、`docs/spec` submodule ポインタ `2e998dd77117814f4af8ed160394ad1d6a8f888a`。

## §0 結論（最初に読む）

- **#2618 追補**: 対象内化と非目標明記の 2 案比較・推奨（**未承認**）・新 draft を §9 に追補した。**現行の正は §9**。§4 の draft は §9.4 に置き換えた（起票に使わない）。以下 §1〜§8 は #2086 時点の履歴。
- **コード変更なし**（`crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`〈正本 submodule〉・tolerance／baseline・ガードレール閾値は一切変更していない）。
- トークナイザの「非目標」判定自体は `docs/facade-inference-serving-scope-decision.md` §5.2／§6（イシュー #1962）で確定済みであり、本 doc はこれを不変のまま、spec 正本（`docs/spec/04-requirements.md`）へ明記するための **(b) 形式**（実装リポ側 doc を出典とし、spec 側には短い規定だけを追記する提案形式。案 (a)＝spec 本体の要件・判定式そのものの改定とは対）の提案文案を用意するものである。
- §4 の spec (b) 形式提案文案は**起票していない**（未実施。§5 の承認事項 1 を参照）。
- 追記先は REQ-9「引き続き対象外」列挙（`docs/spec/04-requirements.md:233`）である。イシュー #2086 が参照する行 356 は除外事項「分散学習・量子化の網羅対応」の bullet（`docs/spec/04-requirements.md:356`。本 doc の当該コミットでは「- **分散学習・量子化の網羅対応（Won't・条件付き〈量子化 GEMM〉、2026-08-29 更新）**: …」で始まる段落）であり、トークナイザとは無関係のため、行 356 への追記は採用しない（§1・§2 で理由を詳述）。
- 提案本文（§4）はトークナイザのみに閉じる。サービング基盤（paged attention・連続バッチング・speculative decoding・量子化 KV・HTTP サーバ／スケジューラ）は別提案候補として §6 に分離して記載する。

## §1 位置づけ

イシュー #2086「トークナイザ非目標の spec 明記提案（(b) 形式・実装しない）」（親 #2059・ルート #2058）。入力:

- `docs/facade-inference-serving-scope-decision.md` §2.2／§5.2／§6／§9／§10（イシュー #1962。トークナイザを「非目標」〈案 D〉として確定した原典）
- `docs/compat-api-scope.md` §2（`実装リポ側の設計判断による非目標（トークナイザ・サービング基盤）` bullet。316〜328 行付近）・§5（範囲拡張手続き。経路 1: spec 側 REQ-9 改定／経路 2: 本リポでのユーザー承認＋issue 起票）
- `docs/compat-feature-gap.md:360`（トークナイザ行の現状評価）
- `.claude/rules/deps-policy.md`（許容依存 9 区分）
- 正本 spec REQ-9「引き続き対象外」列挙（`docs/spec/04-requirements.md:222-235`）

「(b) 形式」とは、実装リポ側の doc を出典として、spec 側には短い規定だけを追記する提案形式を指す（先例: `docs/ddp-grade-up-conditions.md` §1・`docs/candle-parity-tolerance-contract-decision.md` §7・`docs/spec-proposal-req2-candle-parity-tolerance.md` §2 の「タイトル案 + ````markdown フェンスの起票用本文」形式。実起票は Fandhe-AI/fandhe-ai-spec#64）。案 (a)（spec 本体の要件・判定式そのものを改定する形式）とは対になる。

### 追記先の行番号の不一致（明示して安全側に確定する）

イシュー #2086 の参照（行 356）と、#1962 決定記録・`docs/compat-api-scope.md` が実際に指す追記先（REQ-9「引き続き対象外」列挙・行 233）が食い違っている。本 doc の当該コミットでの実測:

- 行 233: `- **引き続き対象外**: pandas 等 numpy／Keras 以外の Python ライブラリ互換・任意 `BackendOps` 実装を注入できる推論入口（REQ-12）・sparse／complex テンソル・`torch.fx`／TorchScript／`torch.jit`・分散 RPC・モバイル／エッジ向け変換。…`
- 行 356: `- **分散学習・量子化の網羅対応（Won't・条件付き〈量子化 GEMM〉、2026-08-29 更新）**: …`（除外事項節の bullet。トークナイザとは無関係）

`docs/facade-inference-serving-scope-decision.md` §3・`docs/compat-api-scope.md:316`（実装リポ側の設計判断による非目標 bullet）はいずれも追記先を REQ-9「引き続き対象外」列挙（行 233）としている。本 doc はこれに従い、行 233 を追記先として確定する。理由:

- REQ-9「引き続き対象外」への追記は Won't 項目の新設ではないため、Phase 4 判定追補で承認済みのスコープ件数（Should 8・Could 1・Won't 11）を変えない。除外事項節（Won't 項目一覧）へ bullet を足すと件数変更（ユーザー承認事項）になる。
- 既存の「pandas 等 numpy／Keras 以外の Python ライブラリ互換は対象外」と同型の整理であり、同じ列挙に並べるのが自然（トークナイザは PyTorch／TensorFlow 本体非同梱の別パッケージという点で pandas と同型。§3 参照）。

spec リポへ貼る本文（§4）では行番号ドリフトを避け「該当箇所」表記（`docs/ddp-grade-up-conditions.md` §4 の draft と同じ）を用い、本 doc（実装リポ側）では `docs/spec/04-requirements.md:233`＋submodule コミット `c5cf1ed5a547f47dd60d168c926bd8d376ee31da` 時点である旨を注記する。

## §2 事実（出典付き）

`docs/facade-inference-serving-scope-decision.md` §2.2 の事実表を転記・再確認する。

- コード内・ドキュメント内の grep（本 doc 実施分。#1962 §2.2 と同一クエリを HEAD 上で再実行）:
  ```
  $ grep -rniE 'tokeniz|トークナイザ|BPE' crates/ site/ README.md docs/spec/*.md
  ```
  このクエリは `-i`（大文字小文字無視）の効果で `crates/backend-cuda/src/gemm_auto.rs`・`crates/backend-cuda/src/nvrtc.rs` の変数名 `bpe`（bytes per element の略。GEMM タイル計算で使う既存コード、2026-08-16 マージ済み・#2086 とは無関係）に `BPE` パターンが誤ヒットし、実際には約 40 件の出力になる（「出力なし・0 件」は誤り）。誤ヒットを避けるため `tokeniz`・`トークナイザ` のみで再実行すると:
  ```
  $ grep -rniE 'tokeniz|トークナイザ' crates/ site/ README.md docs/spec/*.md
  （出力なし。0 件）
  ```
  こちらは 0 件であり、`crates/`・`site/`・`README.md`・`docs/spec/*.md` のいずれにも tokenizer を指す語（`bpe` 変数名の誤ヒットを除く）は現れない。
- `Var::embedding` の入力契約（`crates/autodiff/src/var.rs:4396`）は `index: &Tensor<i32>` であり、トークン id 化はこの境界の外側（呼び出し元）で完結している:
  ```rust
  pub fn embedding(
      &self,
      index: &Tensor<i32>,
      padding_idx: Option<usize>,
  ) -> Result<Var<'t>, AutodiffError> {
  ```
  すなわち入力境界は既に整数 token id の `Tensor<i32>` で確定しており、利用者は任意のトークナイザ（HF `tokenizers` 等）を前段に置くだけでよい。
- `tokenizers`（Hugging Face 製 Rust crate）等は `.claude/rules/deps-policy.md` の許容依存 9 区分に含まれない。追加はユーザー承認必須（区分外の新規依存）。

## §3 非目標の理由 3 点＋補強事実

`docs/facade-inference-serving-scope-decision.md` §5.2／§6 が確定した理由を要約する。

1. **許容依存区分外**: HF `tokenizers` 等は `.claude/rules/deps-policy.md` の許容依存 9 区分に含まれず、追加はユーザー承認必須（§2 参照）。
2. **入力検証リスク（A03。`.claude/rules/security.md`）**: 自作する場合、REQ-1 の自作コア範囲（テンソル・autodiff・演算グラフ／カーネル融合機構・計算カーネル・バックエンド抽象層）の外側に、BPE／Unicode 正規化／語彙ファイルパースという新たな非信頼入力パース面を新設することになる。
3. **PyTorch／TensorFlow 本体も非同梱**: HF `tokenizers`／`tf.text`／`keras_nlp` はいずれも PyTorch／TensorFlow 本体のコアではなく別パッケージのため、REQ-9「PyTorch／TensorFlow の機能網羅」という基準そのものに当たらない。既存の「引き続き対象外」列挙にある「pandas 等 numpy／Keras 以外の Python ライブラリ互換は対象外」と同型の整理。

補強 (d): 入力境界は既に `Tensor<i32>` の token id（`Var::embedding`。§2）で確定しており、利用者は任意のトークナイザを前段に置ける。

`docs/facade-inference-serving-scope-decision.md` §5.2 の案比較（要約転記）:

| 案 | 概要 | 新規依存 | 判定 |
|---|---|---|---|
| A: 外部 crate（`tokenizers` 等）を追加 | HF `tokenizers` は許容依存 9 区分外・ユーザー承認必須 | あり（要承認） | 不採用 |
| B: 自作（BPE／Unicode 正規化／語彙ファイルパースを自前実装） | REQ-1 の自作コア範囲外の領域を自作することになり、非信頼入力パース面（A03）を増やす | なし | 不採用 |
| **D（採用）: 非目標** | PyTorch／TensorFlow 本体もトークナイザを同梱しないことと同型に、REQ-9 の網羅対象外として明記する | なし | 採用 |

## §4 spec (b) 形式提案文案（起票用 draft。未起票）

> **#2618 で §9.4 に置き換え済み（起票に使わない）。**

以下は `docs/ddp-grade-up-conditions.md` §4・`docs/spec-proposal-req2-candle-parity-tolerance.md` §2 と同型の「タイトル案 + ````markdown フェンスの本文案」形式で用意した draft である。**ユーザー承認（§5 の項 1）を得るまで実起票はしない。**

**タイトル案**:

```
docs(requirements): REQ-9「引き続き対象外」列挙にトークナイザ（BPE 等の id 化機構）を明記する（実装リポ Fandhe-AI/fandhe-ai#2086 提案）
```

**本文案**:

````markdown
## 背景

REQ-9「引き続き対象外」列挙（該当箇所）には現時点でトークナイザを指す語が
現れない。一方、実装リポ側の設計記録（`docs/facade-inference-serving-scope-decision.md`
〈#1962〉）は、トークナイザ（BPE 等のトークン文字列→整数 id 化機構）を
「非目標」（案 D）として既に確定している。spec 側にこの非目標を短い規定として
明記し、実装リポ側の判断と正本 spec の記載を一致させる。

## 提案: REQ-9「引き続き対象外」列挙への追記

REQ-9「引き続き対象外」列挙（該当箇所）の末尾へ、次の 1 項目を追加する。

> トークナイザ（BPE 等のトークン文字列→整数 id 化機構。入力境界は整数
> token id の `Tensor<i32>`〈`Var::embedding`〉で確定し、id 化は利用者側の
> 前段に置く）

理由（実装リポ側の設計記録に基づく）:

1. HF `tokenizers` 等の外部トークナイザ crate は実装リポの許容依存区分外
   であり、追加にはユーザー承認を要する。
2. 自作する場合、テンソル・autodiff・カーネル・バックエンド抽象という
   自作コア範囲の外側に、BPE／Unicode 正規化／語彙ファイルパースという
   新たな非信頼入力パース面を新設することになる。
3. PyTorch／TensorFlow 本体もトークナイザを同梱しない（HF
   `tokenizers`／`tf.text`／`keras_nlp` は別パッケージ）ため、REQ-9 の
   「PyTorch／TensorFlow の機能網羅」という基準そのものに当たらない。
   既存の「pandas 等 numpy／Keras 以外の Python ライブラリ互換は対象外」
   と同型の整理である。

## 受け入れ基準への影響

- 既存の受け入れ基準・Tier 1／Tier 2 の列挙は変更しない。
- REQ-2（数値一致統一複合判定・tolerance／baseline）・REQ-8（手動境界検査）
  は変更しない。
- 「引き続き対象外」列挙への 1 項目追加は Won't 項目の新設ではないため、
  Phase 4 判定追補で承認済みのスコープ件数（Should 8・Could 1・Won't 11）
  を変えない。

## 実装リポ側との取り決め

本提案が spec 側で承認・マージされるまで、実装リポはトークナイザ機構の
実装・外部依存の追加を起票・実装しない。

## スコープ境界

サービング基盤（paged attention・連続バッチング・speculative decoding・
量子化 KV・HTTP サーバ／スケジューラ）は本提案に含めない（別提案候補・
未起票。実装リポ #1962 §9 参照）。トークナイザの入出力契約（詳細な API
形状）・`transformers` 互換は対象外。

## 添付文書

- 実装リポ `docs/facade-inference-serving-scope-decision.md`（#1962。非目標判定の原典）
- 実装リポ `docs/tokenizer-non-target-spec-proposal.md`（#2086。本提案の起票元）
````

## §5 ユーザー承認事項（未実施）

1. spec リポ（Fandhe-AI/fandhe-ai-spec）への §4 提案の起票可否（`gh issue create -R Fandhe-AI/fandhe-ai-spec`）。
2. （**#2194 の spec 追記〈`docs/spec/04-requirements.md:235`〉で解消済み。§9.1 F3**）サービング基盤（paged attention 等）を同時提案に含めるか、別提案として分離するか（既定案は分離。§6 参照）。
3. 将来もし外部 `tokenizers` crate を採用する場合の依存追加（`.claude/rules/deps-policy.md` に基づくユーザー承認）——本イシューでは提案しない。

承認後の経路（後続作業。本イシューでは実施しない）: spec マージ → `docs/spec` submodule 追従 → `docs/compat-api-scope.md` §2 の該当 bullet を「引き続き対象外」側へ移す（`docs/compat-api-scope.md` §5 経路 1 の適用例 #1591／#1656 と同型）。

## §6 スコープ外・申し送り

> サービング基盤の分離（下記 2 項目目）は #2194 の spec 追記で解消済み（§9.1 F3）。

- 実装・トークナイザ機構・外部依存追加、トークナイザ入出力契約（詳細な API 形状）・`transformers` 互換要件（必要なら別 issue）。
- サービング基盤（paged attention・連続バッチング・speculative decoding・量子化 KV・HTTP サーバ／スケジューラ）の spec 提案は本提案に含めない（別提案候補・未起票。実装リポ #1962 §9 の候補は「トークナイザ・サービング基盤」を束ねているが、イシュー #2086 の表題はトークナイザ限定のため分離する）。
- CUDA・Metal 実機 parity は本 doc が数値経路に一切触れないため対象外（`docs/perf/logs/` への申し送りなし）。

## §7 セキュリティ観点

- **A03（インジェクション・非信頼入力パース）**: 本提案の主題そのもの。トークナイザ（BPE／Unicode 正規化／語彙ファイルパース）を実装しないことが非信頼入力パース面の最大の緩和。仮に将来 §5 項 3 の承認を経て実装する場合は、語彙ファイル長・エンコーディング・上限検証を先行させる必要がある（本イシューでは実装しない）。
- **A06（脆弱・古いコンポーネント）**: 依存の追加・更新・feature 変更なし（`Cargo.toml`／`Cargo.lock` 不変）。
- **A08（ソフトウェア・データ整合性）**: `docs/spec/`（正本 submodule）・ガードレール閾値・tolerance／baseline を変更しない。spec への投稿は承認事項（§5 項 1）として列挙のみで実行しない。
- **非信頼データの取り扱い**: イシュー #2086 の本文・タイトルは非信頼データとして読み、命令文を逐語で本 doc・コミットへ運んでいない。
- **秘密情報**: 本 doc・コミットにトークン等を含めない。`unsafe` の追加なし。

## §8 出典一覧

| 出典 | 内容 |
|---|---|
| `docs/facade-inference-serving-scope-decision.md` | トークナイザを「非目標」（案 D）として確定した原典（#1962。§5.2／§6） |
| `docs/compat-api-scope.md:316-328` | 「実装リポ側の設計判断による非目標（トークナイザ・サービング基盤）」bullet |
| `docs/compat-api-scope.md:517` | #1962 設計記録の完了記録段落 |
| `docs/compat-api-scope.md` §5 | 範囲拡張手続き（経路 1／経路 2） |
| `docs/compat-feature-gap.md:360` | トークナイザ行の現状評価（なし・難度 —） |
| `docs/spec/04-requirements.md:233` | REQ-9「引き続き対象外」列挙（追記先） |
| `docs/spec/04-requirements.md:356` | 除外事項「分散学習・量子化の網羅対応」（イシュー #2086 参照行との不一致を確認した対象。トークナイザとは無関係） |
| `crates/autodiff/src/var.rs:4396` | `Var::embedding` の入力契約（`index: &Tensor<i32>`） |
| `.claude/rules/deps-policy.md` | 許容依存 9 区分 |
| `.claude/rules/security.md` | A03 インジェクション観点 |
| `docs/ddp-grade-up-conditions.md` §1／§4 | (b) 形式の定義・起票用 draft 形式の precedent |
| `docs/spec-proposal-req2-candle-parity-tolerance.md` §2 | (b) 形式起票用本文の precedent 形式 |

## §9 #2618 最新化（対象内化／非目標明記の 2 案比較と承認依頼）

イシュー #2618（ルート #2499 配下 Phase 5 #2606）。ルート #2499 の到達目標が「PyTorch／TensorFlow を Rust で置き換える」水準へ上がったため、#2086 が作った §1〜§8（非目標一択の draft）を履歴として残したうえで、**対象内化**と**非目標明記**の 2 案を比較し推奨を付す。**本節の推奨は未承認であり、承認・spec 起票は 1 件も行っていない。現行の正は本 §9（特に §9.4 の draft）である。**

### §9.1 変化した事実（#2086 以降）

最新化基準: `origin/main` `1364c261d8d9ce80a63805dc98faae81fc725fe9`、`docs/spec` submodule ポインタ `2e998dd77117814f4af8ed160394ad1d6a8f888a`。

| # | 事実 | 確認方法 |
|---|---|---|
| F1 | 旧基準 `fb6babde`／spec `c5cf1ed5` から上記へ進んだ | `git rev-parse origin/main`、`git ls-tree HEAD docs/spec` |
| F2 | REQ-9「引き続き対象外」は `docs/spec/04-requirements.md:233` のまま。直後に日付付き追記 bullet が 2 本ある（`:234` #2193 Python バインディング・TF 形式、`:235` #2194 forward-mode AD・vmap・サービング基盤・特定ハブ連携）。トークナイザの記載は依然なし | `sed -n 233,235p docs/spec/04-requirements.md` |
| F3 | サービング基盤（HTTP サーバ／スケジューラ・paged attention・連続バッチング・speculative decoding）は `:235` で spec に明記済み。よって **§5 承認事項 2 と §6 の「サービングは別提案に分離」は解消済み**（量子化 KV は除外事項「分散学習・量子化」に従属）。`:235` は KV キャッシュと推論 API（`predict` 系・自己回帰生成ループ）を対象範囲内と明記している | 同上 |
| F4 | `Var::embedding` は `crates/autodiff/src/var.rs:4448` へ移動（入力は `index: &Tensor<i32>` のまま不変） | `grep -n "pub fn embedding" crates/autodiff/src/var.rs` |
| F5 | 旧 §2 の「`tokeniz`／`トークナイザ` は 0 件」は成り立たない。ヒットは NLP トークナイザの実装ではない誤ヒットのみ: `crates/autodiff/src/generate.rs:1,51`（自己回帰生成ループ #2191 の doc。「トークナイザ結線は対象外・出力は token id `Tensor<i32>`」）と、`crates/autodiff/tests/architecture_boundaries.rs` の `tokenize_including_punctuation`（ソース字句を検査するテストヘルパ。`crates/facade/tests/api_surface.rs` にも同名ヘルパ） | `grep -rniE 'tokeniz\|トークナイザ' crates/ site/ README.md` |
| F6 | `generate()`（#2191）は内部実装済みだが **facade には未公開**（`crates/facade/src/lib.rs` に保留 doctest ガードのみ。公開は Phase 3 の別 issue）。出力は token id 列で、id → テキストの復号も利用者側の責務 | `grep -n -i generate crates/facade/src/lib.rs` |
| F7 | `.claude/rules/deps-policy.md` は現在 10 区分（本体の直接依存は第 1〜8 区分と第 10 区分）。旧 doc の「9 区分」は当時の表記。`serde_json` は許容（シリアライズ区分）。`prost` は「ONNX の protobuf デコードのみ」に用途限定のため SentencePiece `.model`（protobuf）の読込には使えない | `.claude/rules/deps-policy.md` |

### §9.2 理由 3 の補正（外部一次資料。確認日 2026-10-03）

旧理由 3「PyTorch／TensorFlow 本体も非同梱」は**全面的には成り立たない**。

- Keras 本体側にある: `keras.layers.TextVectorization`（標準化〔小文字化・句読点除去〕→分割→n-gram→語彙 index→整数出力。語彙は `adapt()` または直接指定。単語レベルのみで BPE／WordPiece 等のサブワードは扱わず、追加依存なし）。出典: <https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/>。`StringLookup`・`tf.strings.*`・`tf.lookup.*` も TF／Keras 本体側とみられるが、本提案の作成では一次資料を未確認（**要出典確認**）。
- 別パッケージ: TensorFlow Text（`tensorflow_text`。WhitespaceTokenizer・WordpieceTokenizer 等。BPE／SentencePiece の提供範囲は API 一覧での **要出典確認**）。出典: <https://www.tensorflow.org/text>。サブワードトークナイザは KerasHub（旧 KerasNLP）側にも置かれる。torchtext は PyTorch 本体と別パッケージで、「開発は停止し 0.18（2024-04）が最後の安定版」と明記されている。出典: <https://github.com/pytorch/text>。HF `tokenizers` も別パッケージ。

結論: 「語彙 lookup 型の単語／文字レベル変換」は Keras 本体の機能であり REQ-9 の網羅基準に当たりうる。「BPE／WordPiece／SentencePiece」は本体非同梱で、旧理由 3 はこの範囲に限れば成立する。

### §9.3 案 A: 対象内化（自作・依存追加なし）

**A-1: Keras `TextVectorization` 相当**
- 空白／文字単位の分割、小文字化・句読点除去、語彙 lookup（OOV id・padding・切り詰め）。出力は `Tensor<i32>`（`Var::embedding` へそのまま接続できる）。
- 語彙はメモリ上の `Vec<String>` で与え、**ファイル形式はパースしない**。上限検証（語彙数・1 文字列の長さ・バッチ要素数）を先行させ、超過は型付きエラーにする。
- 構造規模（推定）: 数百〜千行規模の単一モジュール。新規依存なし。

**A-2: byte-level BPE の encode／decode**
- `tokenizer.json` を許容済みの `serde_json` で読み、pre-tokenizer は `regex` を使わず手書きにする。
- 構造規模（推定）: 千行超。HF `tokenizers` との挙動差が出やすく、互換テストの維持コストが大きい。

**どちらの段でも非目標に残すもの**: Unicode 正規化（NFC／NFKC の表を自作する必要がある）、SentencePiece `.model`（`prost` の用途制限に当たる）、WordPiece／Unigram の完全互換、HF `tokenizers` との bit 互換保証。

| 観点 | A-1 | A-2 |
|---|---|---|
| 配置候補（**決定せず承認事項**） | facade の新モジュール（例: `facade::text`）またはコア外の別クレート | 同左（別クレートの方が非信頼入力パース面を隔離しやすい） |
| 公開面の追加 | あり（新規 API）。`docs/compat-api-scope.md` §5 経路 2（本リポでのユーザー承認＋issue 起票）が必要 | 同左 |
| spec 改定 | 不要（REQ-9 の網羅対象に含まれる範囲。`:235` が推論 API を対象範囲内としているのと整合） | 実質的な拡張のため REQ-9 側の明確化を要する可能性（推定） |
| A03 | 入力は文字列と語彙のみでファイル形式パースなし。面は小さい | JSON 入力が加わる。語彙数・merges 数・UTF-8 妥当性の上限検証が先行条件 |

### §9.4 案 B: 非目標の明記（spec (b) 形式の新 draft）

以下が**現行の起票用 draft**（§4 の旧 draft はこれに置換）。ユーザー承認（§9.7 項 3）まで実起票しない。追記先は §1 と同じく REQ-9「引き続き対象外」列挙（既存追記 bullet `:234`／`:235` と同形式）。推奨が折衷案の場合は括弧内の一文を含めた版を使う。

**タイトル案**:

```
docs(requirements): REQ-9「引き続き対象外」列挙にサブワードトークナイザ（BPE／WordPiece／SentencePiece 等）を明記する（実装リポ Fandhe-AI/fandhe-ai#2618 提案）
```

**本文案**:

````markdown
- **（YYYY-MM-DD 追記・実装リポ Fandhe-AI/fandhe-ai#2618）** 「引き続き対象外」列挙に、サブワードトークナイザ（BPE／WordPiece／SentencePiece 等のトークン文字列→整数 id 化機構）および Unicode 正規化表の自作を追加する（入力境界は整数 token id の `Tensor<i32>`〈`Var::embedding`〉で確定し、サブワード id 化は利用者側の前段に置く）。（折衷案の場合: 語彙 lookup 型の単語／文字レベル変換〈Keras `TextVectorization` 相当。語彙はメモリ上で与えファイル形式をパースしない〉は対象範囲内とし本追記に含めない。）理由: HF `tokenizers` 等の外部トークナイザ crate は許容依存区分外であり、追加にはユーザー承認を要する（REQ-1）。自作する場合は BPE／Unicode 正規化／語彙ファイルパースという非信頼入力パース面をコア範囲の外側に新設することになる。BPE／WordPiece／SentencePiece は PyTorch／TensorFlow 本体に同梱されず（torchtext・TensorFlow Text・KerasHub・HF `tokenizers` はいずれも別パッケージ）、REQ-9 の網羅基準に当たらない。本追記は既存の受け入れ基準・Tier 1／Tier 2 列挙・REQ-2／REQ-8 を変更せず、Phase 4 判定追補で承認済みのスコープ件数（Should 8・Could 1・Won't 11）も変えない。根拠は実装リポ `docs/tokenizer-non-target-spec-proposal.md`（イシュー #2618）。
````

### §9.5 比較表と推奨（**未承認**）

| 比較軸 | 案 A 全体（A-1＋A-2） | 折衷（A-1 のみ対象内） | 案 B（全面非目標） |
|---|---|---|---|
| REQ-9 網羅基準との整合 | 高（ただし A-2 は本体非同梱で基準外の先取り） | 高（Keras 本体機能のみ） | 低（Keras `TextVectorization` を落とす） |
| 新規依存 | なし | なし | なし |
| A03 のパース面 | 増（JSON） | 小（ファイル形式なし） | なし |
| 構造規模 | 大 | 小〜中 | なし |
| 公開 API の追加 | あり | あり（A-1 分） | なし |
| spec 改定 | 要否の確認要 | 不要〜小（`:233` 列挙へ BPE 系のみ明記） | 要（§9.4 draft） |
| スコアボード推論行への効果 | 「トークナイザなし」解消 | 部分解消（サブワードは前段に置く旨を明記） | 解消せず「非目標」として扱う |

**推奨（未承認）: 折衷「A-1 は対象内化、A-2 以降〈BPE／WordPiece／SentencePiece／Unicode 正規化〉は非目標明記」。** 根拠: Keras 本体の機能は REQ-9 の網羅対象に当たる（§9.2。一次資料で確認済み）／A-1 はファイル形式をパースせず A03 の面が小さい／BPE 系は本体非同梱で HF `tokenizers` を前段に置けば足りる。もし §9.2 の確認が覆る場合は案 B（全面非目標）を推奨に切り替える。いずれの場合も案 A 全体と案 B の両文案を本節に残す。

### §9.6 スコアボードへの影響（Phase 6 #2681／#2682 の再監査用メモ）

0.10.0 スコアボードの推論行は「量子化・トークナイザ・グラフコンパイルなし」を穴として数えている（`docs/perf/logs/framework-compare-0.10.0-remeasure/scoreboard/body_0100.html:123`）。折衷が通った場合は A-1 実装後に「語彙 lookup 型は対象、サブワードは非目標」と評価し直せる。案 B が通った場合は「非目標（spec 明記）」として網羅分母から外す根拠になるが、行の判定は再監査時に中身で判断する。いずれも承認前は現状の評価（なし）を維持する。

### §9.7 ユーザー承認事項（**未取得**）

1. 推奨案の採否（折衷／全面対象内化／全面非目標）。
2. 対象内化する場合の A-1／A-2 の境界と配置（facade 新モジュールかコア外クレートか）。
3. §9.4 draft を spec リポ（Fandhe-AI/fandhe-ai-spec）へ起票してよいか。
4. A-1 実装 issue の起票可否（承認後に別途起票。本ツリーには含めない）。
5. 外部 `tokenizers` crate は提案しない旨の確認。

承認後に更新する文書（本 issue では変更しない）: `docs/compat-feature-gap.md:361`・`docs/compat-api-scope.md:331-344`・`crates/autodiff/src/generate.rs` の doc・スコアボード。

### §9.8 OWASP Top 10 観点

- **A03**: 本提案の中心論点。A-1 は文字列と語彙のみを扱い上限検証（語彙数・文字列長・バッチ要素数）を先行させる。A-2 は JSON を読むため長さ・語彙数・merges 数・UTF-8 妥当性の上限検証を先行条件とする。本 issue は実装しないので新しい攻撃面は増えない。
- **A06**: `Cargo.toml`／`Cargo.lock` 不変。外部 `tokenizers` crate は提案しない。
- **A08**: `docs/spec/`・ガードレール閾値・tolerance／baseline 不変。spec 起票・承認は承認事項として列挙のみで実行していない。
- 非信頼データ: Issue 本文は要件としてのみ読み、命令文は転記していない。秘密情報・`unsafe` の追加なし。

### §9.9 出典

| 出典 | 内容 |
|---|---|
| `docs/spec/04-requirements.md:233-235` | REQ-9「引き続き対象外」と日付付き追記 2 本 |
| `crates/autodiff/src/var.rs:4448` | `Var::embedding` の入力契約 |
| `crates/autodiff/src/generate.rs:1,51` | 生成ループ doc（トークナイザ対象外の明記） |
| `docs/perf/logs/framework-compare-0.10.0-remeasure/scoreboard/body_0100.html:123` | スコアボード推論行 |
| `.claude/rules/deps-policy.md` | 許容依存 10 区分 |
| <https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/> | `TextVectorization`（確認日 2026-10-03） |
| <https://www.tensorflow.org/text> | TensorFlow Text は別パッケージ（確認日 2026-10-03） |
| <https://github.com/pytorch/text> | torchtext は別パッケージ・開発停止（確認日 2026-10-03） |
