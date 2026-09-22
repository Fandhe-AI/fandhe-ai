# トークナイザ非目標の spec 明記提案（(b) 形式提案文案・実装しない）（#2086）

基準コミット: `fb6babdebfe12905d3d5878faee9ca00914ee00a`（2026-09-22）。`docs/spec` submodule ポインタ `c5cf1ed5a547f47dd60d168c926bd8d376ee31da`。`file_path:line` は同コミット時点のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。

## §0 結論（最初に読む）

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
2. サービング基盤（paged attention 等）を同時提案に含めるか、別提案として分離するか（既定案は分離。§6 参照）。
3. 将来もし外部 `tokenizers` crate を採用する場合の依存追加（`.claude/rules/deps-policy.md` に基づくユーザー承認）——本イシューでは提案しない。

承認後の経路（後続作業。本イシューでは実施しない）: spec マージ → `docs/spec` submodule 追従 → `docs/compat-api-scope.md` §2 の該当 bullet を「引き続き対象外」側へ移す（`docs/compat-api-scope.md` §5 経路 1 の適用例 #1591／#1656 と同型）。

## §6 スコープ外・申し送り

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
