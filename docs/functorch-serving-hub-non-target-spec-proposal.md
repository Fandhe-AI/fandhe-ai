# forward-mode AD・関数変換・サービング基盤・特定ハブ連携の非目標明記提案（(b) 形式提案文案・実装しない）（#2194）

基準コミット: `9fe4b5231ad53883e4aff072e44a589ac548baaf`（2026-09-27）。`docs/spec` submodule ポインタ `e43704a7baefd1489d3f1716571064ab65c5eed6`。`file_path:line` は同コミット時点のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。

## §0 結論（最初に読む）

- **コード変更なし**（`crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`〈正本 submodule〉・tolerance／baseline・ガードレール閾値は一切変更していない）。
- 追記先は REQ-9「引き続き対象外」列挙（`docs/spec/04-requirements.md:233`）である。
- sparse／complex は同列挙に既記載のため（`docs/tensor-core-sparse-complex-decision.md` §3。#1633）、**本提案では spec 改定を行わず、新規 spec issue を起票しない**。既存の統合引用に留める。
- §4 の spec (b) 形式提案文案は**起票していない**（未実施。§5 の承認事項 1 を参照）。
- 以下は本提案に含めない: トークナイザ（`docs/tokenizer-non-target-spec-proposal.md`・#2086）、言語バインディング・TensorFlow SavedModel 形式（#2193）、分散学習・量子化の網羅対応（既存の除外事項。従属関係のみ §3-C で言及する）。

## §1 位置づけ

イシュー #2194「docs(spec): forward-mode AD・vmap・functorch・sparse・complex・HTTP 非目標明記提案」（親 #2131）。

「(b) 形式」とは、実装リポ側の doc を出典として、spec 側には短い規定だけを追記する提案形式を指す（先例: `docs/tokenizer-non-target-spec-proposal.md`〈#2086〉・`docs/ddp-grade-up-conditions.md`・`docs/spec-proposal-req2-candle-parity-tolerance.md`）。案 (a)（spec 本体の要件・判定式そのものを改定する形式）とは対になる。

### イシュー本文の前提と、リポジトリ実態の不一致（是正して確定する）

イシュー本文は「4 件の決定記録が forward-mode AD・vmap・functorch・HTTP サービングを各々非対応と明記済み」という前提に立つが、実測すると次の食い違いがある。本 doc は是正後の文言を採用する。

| イシュー本文の前提 | 実態（出典） | 是正後の扱い |
|---|---|---|
| forward-mode AD・vmap・functorch を非対応と明記した doc が既存 | `crates/` 内の `jvp`／`forward_ad`／`vmap`／`functorch` の grep は 0 件（§2）。明記した doc は存在しない | 「既存 doc の判断を再掲する」のではなく、「既存契約（reverse-mode テープ・VJP のみ・区分 C）から導かれる非目標を、本 doc で初めて明文化する」と位置づける |
| `docs/autodiff-higher-order-grad-decision.md` §3 が forward-mode 非対応の根拠 | 同 doc §4・§5 の案 C（forward-over-reverse の JVP。HVP 限定）は、段階 1 の代替案として記録されている。forward-mode を否定していない。HVP は reverse-over-reverse の `Tape::backward_create_graph`（`crates/autodiff/src/create_graph.rs:268`。内部クレート限定・facade 公開は #2063 承認事項待ち）で提供済み | 非目標は「**利用者向けの汎用 forward-mode AD API**」（`torch.func.jvp`／`jacfwd`／`torch.autograd.forward_ad` 相当）に限定する。案 C（HVP 用途の内部 JVP）は妨げないと明記する |
| `docs/autodiff-graph-optimization-scope-decision.md` §4-5 区分 B が根拠 | 同 doc に vmap・functorch の記述はない。関連づけられるのは区分 C（汎用 JIT／トレース再コンパイル・グラフ書き換えは非目標）のみ | vmap・`torch.func` の合成関数変換は、区分 C と同型の「プログラム変換」として位置づける（区分 B は引用しない） |
| `docs/facade-inference-serving-scope-decision.md` §4 が HTTP 非対応の根拠 | 該当記述は §5.1 案 C と §6（KV キャッシュ行の「明確に非目標とするもの」） | §5.1／§6 を引用する |
| 「モデルハブ非対応」 | `docs/model-distribution-design.md` §5（ユーザー決定 2026-09-24「他ライブラリと同じにする」）は、汎用の HTTPS ダウンロード＋ローカルキャッシュ＋ハッシュ検証を**コアの責務**（`ModelRegistry` は `crates/facade/src/model.rs:464` に実装済み。リモート取得は #2088 の設計記録のみ）とし、特定ハブ（Hugging Face Hub）連携は**コア外の別クレート**（#2243。OPEN）に分離している | 「特定モデルハブ（HF Hub 等）連携を facade（コア）へ組み込まない。提供する場合は別クレート（#2243。依存追加・workspace 追加はユーザー承認）」と書く。「モデルハブ非対応」という一括表現は用いない |
| functorch は「研究段階技術のため stable ライブラリ対象外」 | PyTorch 2.x の `torch.func` は安定 API 系統であり、この表現は事実として裏付けがない | 「研究段階」とは書かない。理由はリポジトリ契約（§3-A）で構成する |

## §2 事実（出典付き。基準コミット `9fe4b5231a` で再実行）

- `grep -rniE '\bjvp\b|forward_ad|vmap|functorch' crates/ --include='*.rs'` は **0 件**。
- `Tape`（`crates/autodiff/`）は eager の reverse-mode 記録器であり、勾配規則は VJP のみ（`crates/autodiff/src/grad.rs`）。
- HVP は `Tape::backward_create_graph`（`crates/autodiff/src/create_graph.rs:268`。§8 対象 Op 限定・facade 未公開）で提供済み。
- バッチ次元は明示的な軸として扱う: rank≥3 の `Var::matmul`（batched gemm）と einsum の batch 縮約（`docs/autodiff-einsum-batch-decision.md`）。汎用の任意関数へのバッチ次元自動挿入（vmap 相当）は存在しない。
- HTTP サーバ／クライアント系の依存・コードは本体 workspace に存在しない（`Cargo.toml`・`crates/` の依存を確認。許容依存 9 区分〈`.claude/rules/deps-policy.md`〉に HTTP クライアント区分は含まれない）。
- `ModelRegistry`（`crates/facade/src/model.rs:464`）は存在する。ローカルキャッシュ限定・読み取り専用の入口である。
- sparse／complex の事実は `docs/tensor-core-sparse-complex-decision.md` §2・§6（段階 0・非対応の明文化。#1633）に整理済みであり、本 doc では再確認のみ行い変更しない。

## §3 項目別の非目標理由

### A. 汎用 forward-mode AD API・vmap・`torch.func` 合成関数変換

1. **AD 実行モデルの単一系統性**: AD 実行モデルは reverse-mode テープ＋VJP の単一系統である。2 つ目の AD モード（JVP 規則を全 `Op` へ追加すること）は `Op`／`BackendOps` の横断拡張になり、`docs/autodiff-higher-order-grad-decision.md` §10 と同型の承認事項に当たる。
2. **二重保守コスト**: 2 系統の AD を並行維持すると、`Op`（現行 69 variant。`docs/autodiff-higher-order-grad-decision.md` §5）全体に JVP と VJP の二重保守コストが恒久的に生じる。数値契約（FMA 契約・`f64` アキュムレータ契約・REQ-2 統一複合判定）も 2 系統分定義する必要が生まれる。
3. **プログラム変換としての vmap**: vmap・`torch.func` の合成関数変換はプログラム変換であり、`docs/autodiff-graph-optimization-scope-decision.md` §5 区分 C（汎用 JIT／トレース再コンパイル・グラフ書き換えは非目標）と同型の構造を持つ。
4. **HVP・高階微分は対象範囲内**: HVP・高階微分は Tier 2（高階微分）として reverse-over-reverse で対象範囲内にある。案 C（HVP 用途の内部 JVP。`docs/autodiff-higher-order-grad-decision.md` §4・§5）は本非目標の対象に含めない。

### B. sparse／complex（既記載の確認・変更なし）

`docs/tensor-core-sparse-complex-decision.md` §3・§6（#1633）の結論（案 A・段階 0・非対応の明文化）を統合引用する。FFT は実部・虚部の対で表す実テンソル表現（`docs/autodiff-fft-design.md`）であり、complex dtype の非目標を再開しない。

**本提案は sparse／complex について spec 改定を行わず、新規 spec issue を起票しない**。REQ-9「引き続き対象外」列挙（`docs/spec/04-requirements.md:233`）に既に明記されており、`docs/tensor-core-sparse-complex-decision.md` §3 が spec 整合を確認済みである。

### C. サービング基盤（HTTP サーバ／スケジューラ・paged attention・連続バッチング・speculative decoding）

根拠は `docs/facade-inference-serving-scope-decision.md` §5.1 案 C・§6 の整理。

1. REQ-1 の自作コア範囲（テンソル・autodiff・カーネル・バックエンド抽象）の外側にある領域である。
2. HTTP スタックは許容依存 9 区分（`.claude/rules/deps-policy.md`）の外にあり、未承認の新規区分に当たる（`docs/model-download-design.md`・`docs/model-distribution-design.md` §4）。
3. ネットワーク入力を受ける面（OWASP A03／A05 の攻撃面）を新設することになる。
4. 量子化 KV は独立項目にせず、既存の除外事項「分散学習・量子化の網羅対応」（Won't・条件付き）に従属する。

次のものは対象範囲内として明記し、非目標に巻き込まない: KV キャッシュ（`docs/kv-cache-design.md`。#2084 実装済み・内部クレート）、`Sequential::predict`／`predict_resident`。

### D. 特定モデルハブ連携のコア非搭載

`docs/model-distribution-design.md` §5 の方針を引用する: 汎用 HTTPS 取得＋ローカルレジストリ（`ModelRegistry`）はコア、Hugging Face Hub 等の特定ハブ連携はコア外の別クレート（#2243）。

PyTorch（`torch.hub`／`huggingface_hub`）、TensorFlow/Keras（`tf.keras.utils.get_file`／`tensorflow_hub`）と同型の分離である（出典 URL は `docs/model-distribution-design.md` §5 を参照）。#2246 の申し送り（「コア非対応・別クレートで提供」）に揃える。

## §4 代替手段

- **A**: reverse-mode AD（`Tape::backward`）。HVP は `Tape::backward_create_graph`。バッチ化は明示的なバッチ次元（batched matmul・einsum batch・broadcast）で行う。関数変換が必須の研究用途は Python の PyTorch `torch.func` で行い、成果を safetensors／ONNX で持ち込む（REQ-7 の相互運用経路）。
- **B**: 実部・虚部の 2 テンソル分解（`docs/tensor-core-sparse-complex-decision.md` §5 案 D・FFT 設計）。sparse は dense 化してから使う。
- **C**: 利用者のアプリ側で `predict`／`Sequential::predict_resident` を自前の HTTP サーバに組み込む。前段にリバースプロキシを置く。ONNX export（`OnnxModel::to_bytes`）で外部推論サーバへ持ち出す。
- **D**: 汎用 HTTPS 取得＋`ModelRegistry`（リモート取得機構は依存承認待ち）、または別クレート（#2243、承認後）。

## §5 spec (b) 形式提案文案（起票用 draft。未起票）

以下は `docs/tokenizer-non-target-spec-proposal.md` §4 と同型の「タイトル案 + ````markdown フェンスの本文案」形式で用意した draft である。**ユーザー承認（§6 の項 1）を得るまで実起票はしない。**

**タイトル案**:

```
docs(requirements): REQ-9「引き続き対象外」列挙に汎用 forward-mode AD／関数変換・サービング基盤・特定モデルハブのコア組み込みを明記する（実装リポ Fandhe-AI/fandhe-ai#2194 提案）
```

**本文案**:

````markdown
## 背景

実装リポ側の複数の設計判断記録が、PyTorch／TensorFlow の一部機能を個別に
「非目標」「コア外」と整理している。REQ-9「引き続き対象外」列挙（該当箇所）
には現時点でこれらの一部（forward-mode AD・vmap・functorch・サービング
基盤・特定モデルハブ連携）を指す語が現れない。sparse／complex テンソルは
既に明記済みであり、本提案では変更しない。

## 提案: REQ-9「引き続き対象外」列挙への追記

REQ-9「引き続き対象外」列挙（該当箇所）の末尾へ、次の 3 項目を追加する。

> 利用者向けの汎用 forward-mode AD API（`torch.func.jvp`／`jacfwd`／
> `torch.autograd.forward_ad` 相当）・vmap・`torch.func` 相当の合成関数
> 変換（reverse-mode テープ＋VJP の単一系統を維持するため。HVP 用途の
> 内部 JVP は対象外としない）

> サービング基盤（HTTP サーバ／スケジューラ・paged attention・連続
> バッチング・speculative decoding。自作コア範囲外・許容依存区分外の
> HTTP スタックを要するため。KV キャッシュ・推論 API 自体は対象範囲内）

> 特定モデルハブ（Hugging Face Hub 等）連携のコア組み込み（汎用 HTTPS
> 取得＋ローカルキャッシュ＋ハッシュ検証はコアの責務。特定ハブ API への
> 特化はコア外の別クレートで提供する）

## 既記載項目の確認（sparse／complex。変更なし・新規 issue なし）

REQ-9「引き続き対象外」列挙は既に sparse／complex テンソルを含んでいる
（実装リポ #1633 で spec 整合確認済み）。本提案はこれを変更せず、新規
spec issue も起票しない。

## 理由（実装リポ側の設計記録に基づく）

1. forward-mode AD／vmap: AD 実行モデルが reverse-mode テープ＋VJP の
   単一系統であり、2 系統目の追加は `Op`／`BackendOps` の横断拡張・数値
   契約の二重定義を要する（承認事項）。vmap 等の関数変換は汎用グラフ JIT
   と同型のプログラム変換であり非目標。
2. サービング基盤: 自作コア範囲（テンソル・autodiff・カーネル・バック
   エンド抽象）の外側にあり、HTTP スタックは許容依存区分外の新規区分と
   なる。ネットワーク入力面の新設を伴う。
3. 特定モデルハブ連携: PyTorch（`torch.hub`／`huggingface_hub`）・
   TensorFlow/Keras（`tf.keras.utils.get_file`／`tensorflow_hub`）と同型に、
   汎用取得はコア・特定ハブ API への特化は別パッケージという分離を踏襲
   する。

## 受け入れ基準への影響

- 既存の受け入れ基準・Tier 1／Tier 2 の列挙は変更しない。
- REQ-2（数値一致統一複合判定・tolerance／baseline）・REQ-8（手動境界検査）
  は変更しない。
- 「引き続き対象外」列挙への項目追加は Won't 項目の新設ではないため、
  Phase 4 判定追補で承認済みのスコープ件数（Should 8・Could 1・Won't 11）
  を変えない。

## 各項目の再開条件（参考。正式な格上げ条件表ではない）

- forward-mode AD／vmap: 利用者需要と `Op`／`BackendOps` 拡張の承認。
- サービング基盤: HTTP 依存区分の承認と REQ-1 範囲の再定義。
- 特定モデルハブ連携: #2243 の承認。

## 実装リポ側との取り決め

本提案が spec 側で承認・マージされるまで、実装リポは上記 3 項目の実装・
外部依存の追加を起票・実装しない。

## スコープ境界

トークナイザ（別提案。実装リポ #2086）・言語バインディング／TensorFlow
SavedModel 形式（実装リポ #2193）とは独立した項目であり、起票順に依存
しない。KV キャッシュ・推論 API（`predict`／`predict_resident`）・高階
微分（HVP）は対象範囲内であり本提案に含めない。

## 添付文書

- 実装リポ `docs/autodiff-higher-order-grad-decision.md`（forward-over-reverse 案 C の記録）
- 実装リポ `docs/autodiff-graph-optimization-scope-decision.md`（区分 C の記録）
- 実装リポ `docs/facade-inference-serving-scope-decision.md`（サービング基盤非目標の記録）
- 実装リポ `docs/model-distribution-design.md`（特定モデルハブ分離の記録）
- 実装リポ `docs/tensor-core-sparse-complex-decision.md`（sparse／complex 既記載の確認元）
- 実装リポ `docs/functorch-serving-hub-non-target-spec-proposal.md`（#2194。本提案の起票元）
````

## §6 ユーザー承認事項（未実施）

1. spec リポ（Fandhe-AI/fandhe-ai-spec）への §5 提案の起票可否（`gh issue create -R Fandhe-AI/fandhe-ai-spec`）。
2. #2086（トークナイザ）・#2193（言語バインディング／TF 形式）の文案と束ねて起票するか、個別に起票するか（既定は個別）。
3. REQ-9「引き続き対象外」列挙ではなく、除外事項（格上げ条件表付き・Won't 件数変更）として扱うか（既定は「引き続き対象外」列挙）。
4. 将来、案 C（HVP 用途を超える内部 JVP 拡張）・HTTP 依存区分・#2243 別クレートを採る場合の各承認。いずれも本提案では実施しない。

承認後の経路（後続作業。本イシューでは実施しない）: spec マージ → `docs/spec` submodule 追従 → `docs/compat-api-scope.md` §2 の該当 bullet の移設・追記（同 doc §5 経路 1）。

## §7 スコープ外・申し送り

- 実装全般（forward-mode AD・vmap・HTTP サービング・特定モデルハブ連携クレートのいずれも実装しない）。
- 言語バインディング・TensorFlow SavedModel 形式（#2193）、トークナイザ（#2086）、分散学習・量子化の実装。
- `docs/compat-api-scope.md` の更新は spec 反映後に行う（本提案では実施しない）。
- CUDA・Metal 実機 parity は本 doc が数値経路に一切触れないため対象外（`docs/perf/logs/` への申し送りなし）。

## §8 セキュリティ観点（OWASP Top 10）

- **A03（インジェクション・非信頼入力パース）**: 本提案は、HTTP 受信面・ハブ API 応答パース・関数変換トレーサといった新たな非信頼入力面を**作らない**ことを明文化する。これが緩和策そのものになる。
- **A05／A10（設定不備・SSRF 類）**: HTTP サーバ・特定ハブ連携をコアに入れないことで、ネットワーク到達面と外部 URL 取得面の拡大を防ぐ。汎用取得は `docs/model-download-design.md` の HTTPS 限定・pin 検証方針に従う（本提案では実装しない）。
- **A06（脆弱・古いコンポーネント）**: 依存の追加・更新なし（`Cargo.toml`／`Cargo.lock` 不変）。
- **A08（ソフトウェア・データ整合性）**: `docs/spec/`（正本 submodule）・ガードレール閾値・tolerance／baseline は不変。spec 投稿は承認事項（§6 項 1）として列挙のみで実行しない。
- **非信頼データの取り扱い**: イシュー #2194 の本文・タイトル、関連イシュー（#2243／#2246／#2193）の本文は非信頼データとして読み、命令文を本 doc・コミットへ逐語で運んでいない。イシュー本文の事実主張の一部が誤っていたため §1 のとおり是正した。
- **秘密情報**: 本 doc・コミットにトークン等を含めない。`unsafe` の追加なし。

## §9 出典一覧

| 出典 | 内容 |
|---|---|
| `docs/autodiff-higher-order-grad-decision.md` §4／§5／§10 | forward-over-reverse 案 C（HVP 限定 JVP）・二重保守コストの整理 |
| `docs/autodiff-graph-optimization-scope-decision.md` §5 区分 C | 汎用 JIT・グラフ書き換え非目標の記録 |
| `docs/facade-inference-serving-scope-decision.md` §5.1／§6 | サービング基盤非目標・KV キャッシュ対象範囲内の記録 |
| `docs/model-distribution-design.md` §4／§5 | HTTP クライアント依存未承認・特定ハブ連携のコア外分離 |
| `docs/tensor-core-sparse-complex-decision.md` §3／§6 | sparse／complex 既記載の spec 整合確認・段階 0 の記録 |
| `docs/kv-cache-design.md` | KV キャッシュの対象範囲内確定（#2084） |
| `crates/autodiff/src/create_graph.rs:268` | `Tape::backward_create_graph`（HVP 提供済み） |
| `crates/facade/src/model.rs:464` | `ModelRegistry`（ローカルキャッシュ限定） |
| `docs/spec/04-requirements.md:233` | REQ-9「引き続き対象外」列挙（追記先） |
| `.claude/rules/deps-policy.md` | 許容依存 9 区分 |
| `.claude/rules/security.md` | A03 インジェクション観点 |
| `docs/tokenizer-non-target-spec-proposal.md` | (b) 形式・起票用 draft 形式の precedent（#2086） |
