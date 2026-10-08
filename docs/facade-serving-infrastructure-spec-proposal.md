# サービング基盤（HTTP サーバ・連続バッチング・paged attention・speculative decoding・量子化 KV）の REQ-9 改定提案（提案文案・未起票）（#2624）

> 基準コミット: `origin/main` `163b7da1`（2026-10-04）／`docs/spec` submodule ポインタ `2e998dd7`。本書の
> `file_path:line` は同コミット時点の値（実装時に再計測した）。

## §0 結論（最初に読む）

- **コード変更なし**（`crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`・tolerance／baseline・ガードレール閾値は一切変更していない）。
- 本書は、#2194（`docs/functorch-serving-hub-non-target-spec-proposal.md`・PR #2330）が REQ-9「引き続き対象外」へ加えた (2)「サービング基盤（HTTP サーバ／スケジューラ・paged attention・連続バッチング・speculative decoding）」を、**構成要素ごとに見直す**改定提案である（`docs/spec/04-requirements.md:235`、変更履歴 `:436`）。ルート #2499 の目標（PyTorch／TF の置き換え）に対し、スコアボード「推論・サービング」行の網羅に必要なため。
- 推奨は**案 C（段階化。推奨候補）**。依存を追加せずに済む speculative decoding とネットワーク非依存の連続バッチング用ライブラリ API を Tier 2 へ移し、paged attention は K-3（デバイス常駐 KV）の充足を条件にする。HTTP サーバ／API サーバと量子化 KV は対象外に残す。**決定はユーザーに委ねる**（§6）。
- 規模見積りは本書の推測であり、確定値ではない（§3）。
- spec 文案（§5）は**未起票**。spec リポ（Fandhe-AI/fandhe-ai-spec）への投稿はユーザー承認事項であり、本 issue では実施しない。承認の代行もしていない。

## §1 位置づけ

- Issue #2624、親 #2606（仕様外・依存承認が要る項目の提案と条件検証）、ルート #2499。直接の先例は `docs/autodiff-forward-mode-vmap-spec-proposal.md`（#2617。#2194 の (1) を見直す提案）。
- #2194 が (2) を非目標とした理由は次の 3 点に整理できる。(i) 自作コアの範囲外。(ii) 許容依存区分外の HTTP スタックを要する。(iii) ネットワーク入力面を持ち込む（A03／A05）。量子化 KV は (iv) として、除外事項「分散学習・量子化の網羅対応」（Won't・条件付き）に従属する。
- 本書は各案がどの理由を解消し、どれを維持するかを §3.2 の表で示す。
- KV キャッシュ・推論 API（`predict` 系・自己回帰生成ループ）自体は対象範囲内のままで、本書の対象は (2) の列挙項目に限る。

## §2 事実（出典付き。基準コミットで再計測）

| 事実 | 計測・出典 |
|---|---|
| spec の (2) は 4 要素を非目標にしている | `docs/spec/04-requirements.md:235`。KV キャッシュ・`predict` 系・自己回帰生成ループは対象内と明記 |
| 量子化は除外事項で Won't（条件付き） | `docs/spec/04-requirements.md:358-365`。int8 の格上げ判断は #2609・#2610（Phase 5）、実機側は #2607（#2683 へ分離済み） |
| paged／speculative／continuous batching の実装はない | `grep -rniE 'paged\|speculative\|continuous.?batch' crates --include='*.rs'` が 1 件。`crates/backend-cuda/src/pool.rs:420` のコメント中の一般語（「speculative な variant」）で、機能とは無関係 |
| 本体 workspace に HTTP サーバ／クライアント依存はない | `grep -rniE 'hyper\|axum\|tiny_http\|tokio' Cargo.toml crates/*/Cargo.toml` が 0 件 |
| `KvCache` はホスト上の contiguous 保持 | `crates/autodiff/src/nn/attention.rs:1554`。射影済み K/V を `[B, S_cached, E]` で保持し、バッチ内の全系列が同じ `S_cached` を共有する。系列ごとの可変長・ブロック管理はない。デバイス常駐は K-3 に切り出し済みで未着手（`docs/kv-cache-design.md` §2 案 B） |
| `KvCache` の facade 公開は保留（記録時点。2026-10-07 に承認され PR #2816〈#2579〉で `fandhe_ai::nn::kv_cache` として公開。`docs/kv-cache-design.md` §14） | `KvCacheHoldDoctestGuard` と `crates/facade/tests/api_surface.rs` の否定ガード（#2579 で正ガードへ置換済み）。公開形の確定は Phase 3 の #2577〜#2580 |
| `generate()` は 3 戦略のみ | `crates/autodiff/src/generate/mod.rs`（`pub fn generate`）。greedy／top-k／temperature。EOS 早期停止・pad・repetition penalty は対象外（同ファイル冒頭 `:46`）、top-p・draft model を差す口はない。facade は未公開（`GenerateHoldDoctestGuard`）。公開形は #2573〜#2576（open） |
| `predict_batches` は内部実装済み・facade 未公開 | `docs/facade-predict-batches-phase-metrics-decision.md`。公開形は #2581〜#2583（open） |
| `docs/model-download-design.md` §14 の「第 11 区分相当」は HTTP **クライアント**用 | #2621 の起案で、`ModelRegistry::download` のためのもの。サーバ用途を含まないため、**サーバ依存の根拠としては引用しない**。区分起案の書式の先例としてのみ参照する |

## §3 構成要素ごとの比較と規模見積り

概算は 2h 粒度の issue 数で、**推測**である。実測・実装前調査に基づく値ではない。

| 要素 | 内容と論点 | 依存追加 | 前提 issue | 新規 `Op`／`BackendOps` | OWASP 面 | 概算（推測） |
|---|---|---|---|---|---|---|
| (a) 生成ループの拡張 | EOS 早期停止・top-p・repetition penalty・バッチ内可変長（pad と `attn_mask`）。spec 上は「自己回帰生成ループ」として既に対象内のため改定不要。新しい公開面は `docs/compat-api-scope.md` §5 の承認事項 | 不要 | #2573〜#2576 | 不要（ホスト側ロジック） | なし | 6〜10 |
| (b) speculative decoding | `generate()` の上に組むアルゴリズム。draft model を差し込む口が要る。greedy では target 単独の greedy 結果と bit 一致させる判定案（新しい tolerance は導入しない）。サンプリング版の分布一致判定は承認事項 | 不要 | #2573〜#2576 | 不要 | A02（非暗号 PRNG の扱い） | 6〜10 |
| (c) 連続バッチング（イテレーション単位のスケジューラ） | ネットワークに依存しないライブラリ API として、ホストで動く純 Rust 実装。現行 `KvCache` は全系列が同じ `S_cached` を共有するため、系列ごとの長さ管理（可変長キャッシュまたはマスク）が要る。性能目標は掲げない | 不要 | #2577〜#2583 の公開形（先取りしない） | 原則不要 | なし | 8〜14 |
| (d) paged attention（ブロックテーブル化した KV） | 承認済みの KV 設計（ホスト上 contiguous `[B,S,E]`）と構造が衝突する。現実的には K-3（デバイス常駐 KV）が前提。CPU 参照実装（gather と sdpa の合成）を先に作り、CUDA／Metal は既定の `Unsupported` フォールバック。専用カーネル・`BackendOps` 拡張が要るなら別承認。REQ-8 の境界検査が適用される | 不要 | K-3 | 専用カーネルなら要（別承認） | なし | 12〜20（K-3 を除く） |
| (e) 量子化 KV | 独立項目にせず、除外事項「分散学習・量子化の網羅対応」（Won't・条件付き）に従属する。#2609／#2610／#2607（#2683）の格上げ判断を待つ | 不要 | #2609・#2610 | 要になりうる | なし | 本書では見積らない |
| (f) HTTP サーバ／API サーバ | サーバ用 crate（hyper／axum＋tokio／tiny_http 等）が要る。候補名を挙げるのみで、ライセンスと推移依存は**未実測**（実測は承認後の別 issue）。手書きの HTTP/1.1 パーサは A03 の非信頼入力パース面を新設する。公開面の追加は A01／A05／A07（認証・リクエストサイズ上限・同時実行数上限・TLS 終端）にも及ぶ。PyTorch／TF 本体は HTTP 推論サーバを同梱せず、TorchServe・TF Serving は別プロジェクトである。提供するならコア外の別クレート（`docs/hf-hub-integration-design.md` と同型）が候補。facade が唯一の公開面であること（`docs/compat-api-scope.md` §0）との整合、新しい依存区分（第 12 区分相当。区分番号は承認時に確定）は承認事項 | **要（区分新設）** | 区分の承認 | 不要 | A01／A02／A03／A05／A07 | 10〜20（推測） |

連続バッチング・paged attention は、PyTorch／TF 本体ではなく vLLM／TGI 等の別プロジェクトが主に提供している。speculative decoding（assisted generation）は HF `transformers` 側の機能である。バージョンごとの API 有無・保守状況は確認しておらず、本書は断定しない。

### 3.1 量子化 KV・speculative decoding を含めるか

- speculative decoding は**含める候補**。依存不要で、生成ループの延長として実装でき、#2194 の理由 (i)〜(iii) のいずれにも抵触しない。
- 量子化 KV は**含めない**。除外事項に従属し、格上げ条件（`docs/spec/04-requirements.md:358-365` の (a)〜(g)）は未達。

### 3.2 改定形の候補

| 案 | 内容 | (i) 自作コア範囲外 | (ii) HTTP 依存 | (iii) ネットワーク入力面 | (iv) 量子化従属 | REQ-2／8／12 |
|---|---|---|---|---|---|---|
| A | (2) を削除し、HTTP サーバまで全面的に対象内化 | 解消 | 解消（区分新設が必須） | 新設（A01／A03／A05／A07） | 要別途判断 | 影響が広く、REQ-8 の再審が要る |
| B | 格上げ条件表方式（`docs/ddp-grade-up-conditions.md`・`docs/rocm-grade-up-conditions-v2-spec-proposal.md` と同型）で要素ごとに着手条件を定める | 要素ごとに解消 | 維持（条件に含める） | 維持 | 維持 | 不変 |
| **C（推奨候補）** | 段階化。Tier 2 へ移す: (b)、(c)。条件付き: (d)（K-3 の充足と CPU 参照実装の成立が移行条件）。対象外に残す: (f)（コアには入れず、提供するならコア外。依存区分の新設は承認事項）、(e) | (b)(c)(d) は解消 | 維持 | 維持 | 維持 | 不変（新しい tolerance／baseline なし） |
| D | 現状維持。「推論・サービング」行は「部分的」のまま残る | 維持 | 維持 | 維持 | 維持 | 不変 |

案 C を推す理由は、依存を追加せずに網羅へ近づけられ、(ii)(iii)(iv) を維持できるため。公開 API 互換は、いずれの案も `fandhe-ai =0.10.0` を壊さず、追加 API・opt-in のみとし、`FitConfig`（`Copy + Eq`）にフィールドを足さない前提とする。

## §4 既存機構との関係

- 前提とする設計: `docs/kv-cache-design.md`（K-3）、`docs/facade-generate-decision.md`、`docs/facade-predict-batches-phase-metrics-decision.md`、Phase 3 の公開 issue 群（#2573〜#2583）。公開形は先取りしない。
- 数値契約: 全系列を再計算した結果との一致は、既存の sdpa・matmul の parity 契約（REQ-2 統一複合判定・FMA 契約）へそのまま帰着させる。新しい tolerance／baseline は導入しない。
- GPU 経路に触れる実装（(d) の専用カーネル等）を将来行う場合は、実機 parity を `#[ignore]` で分離し、`docs/perf/logs/` へ申し送る。

## §5 spec 改定文案（起票用 draft。未起票）

タイトル案: `REQ-9: speculative decoding・ネットワーク非依存の連続バッチングを対象内へ移し、paged attention は K-3 充足を条件に移す（HTTP サーバ／API サーバ・量子化 KV は対象外のまま）`

````markdown
## 背景
2026-09-29 追記（実装リポ #2194）で REQ-9「引き続き対象外」に加えたサービング基盤（HTTP サーバ／スケジューラ・paged attention・連続バッチング・speculative decoding）を、構成要素ごとに見直す。目的は REQ-9 の PyTorch／TensorFlow 水準の機能網羅（実装リポ #2499）。

## 提案
1. 「引き続き対象外」(2) を次へ改める。
   - 対象外: HTTP サーバ／API サーバ（コアには搭載しない。提供する場合はコア外とし、依存区分の新設は実装リポ側の承認事項）、量子化 KV（「分散学習・量子化の網羅対応」の除外事項に従属する）。
   - 対象内（Tier 2）: speculative decoding、ネットワーク非依存のバッチ生成スケジューラ（連続バッチング）。依存追加なし・性能保証なし。
   - 条件付き（移行条件の充足を要する）: paged attention。デバイス常駐 KV キャッシュ（K-3）が成立し、CPU 参照実装が成立していること。
2. KV キャッシュ・推論 API（`predict` 系・自己回帰生成ループ）は従前どおり対象範囲内である。

## 受け入れ基準への影響
REQ-2（統一複合判定）・REQ-8・REQ-12 は変更しない。新機能の数値一致は既存の統一複合判定の範囲で扱い、新しい判定契約が必要になった場合は別途承認を得る。Phase 4 判定追補で承認済みのスコープ件数（Should 8・Could 1・Won't 11）も変えない。

## 各項目の再開条件（対象外に残す項目）
- HTTP サーバ／API サーバ: 依存区分の新設承認、ライセンスと推移依存の実測、ネットワーク公開面の脅威整理（認証・サイズ上限・同時実行数上限）が揃った場合。
- 量子化 KV: 除外事項「分散学習・量子化の網羅対応」の格上げ条件が成立した場合。

## 実装リポ側との取り決め
公開面の追加は `docs/compat-api-scope.md` §5 の手続きで承認を得る。`fandhe-ai =0.10.0` の公開 API は壊さず、追加 API・opt-in のみとする。
````

案 A を選ぶ場合は 1 を「(2) を削除」へ差し替える。案 B の場合は 1 を要素ごとの格上げ条件表へ差し替える。案 D は文案不要。

## §6 ユーザー承認事項（未実施）

1. 改定形（案 A〜D）の選択。
2. spec リポへの投稿（本書は未起票）。
3. HTTP サーバ依存区分の新設と、コア外クレートの可否（案 A、または将来の再開時）。
4. paged attention を進めるときの `Op`／`BackendOps` 拡張と K-3 の着手。
5. speculative decoding（サンプリング版）・連続バッチングの数値判定方式（tolerance／baseline は変えない前提）。
6. facade 公開面の追加（`docs/compat-api-scope.md` §5）。
7. 承認後の実装 issue 起票（本ツリー外で別途行う）。

## §7 スコープ外・申し送り

- 実機 parity: コード変更がないため `docs/perf/logs/` への申し送りは発生しない。
- スコアボードと feature matrix（`docs/perf/framework-compare-feature-matrix-0.9.0.md`）の再評価は Phase 6（#2680）。
- サーバ crate のライセンス実測は承認後。
- #2615（グラフコンパイル）・#2618（トークナイザ）・#2622（HF Hub）との境界を維持する。
- #2194 の文書（`docs/functorch-serving-hub-non-target-spec-proposal.md`）は編集しない。

## §8 セキュリティ観点（OWASP Top 10）

- A08: 承認を代行しない。spec リポへの起票・PR、`docs/spec/` の編集、ruleset 変更はしていない。得ていない承認を本書・コミット・PR・コメントに書かない。
- A06: `Cargo.toml`／`Cargo.lock` は不変。提案も依存追加を前提にしない。サーバ crate は候補名のみで未実測。
- A03: issue 本文は非信頼データとして扱い、逐語転記していない。HTTP サーバを手書きパーサで作ると非信頼入力パース面が増える点を、対象外に残す根拠にした。
- A01／A05／A07: ネットワーク公開面（認証・リクエストサイズ上限・同時実行数上限・TLS 終端）の新設リスクを、HTTP サーバをコアに入れない理由として整理した。
- A02: speculative decoding・サンプリングは非暗号 PRNG（xorshift64*）を使う。生成した token をセキュリティ用途に使わないという既存の注記を引き継ぐ。

## §9 出典一覧

- `docs/spec/04-requirements.md:235,358-365,436`（読むのみ）
- `docs/functorch-serving-hub-non-target-spec-proposal.md`（#2194。本書で編集しない）
- `docs/autodiff-forward-mode-vmap-spec-proposal.md`（#2617。構成の先例）
- `docs/facade-inference-serving-scope-decision.md`・`docs/kv-cache-design.md`・`docs/facade-generate-decision.md`・`docs/facade-predict-batches-phase-metrics-decision.md`
- `docs/model-download-design.md` §14（HTTP クライアント用。書式の先例のみ）
- `docs/hf-hub-integration-design.md`・`docs/compat-api-scope.md` §0・§5
- `docs/ddp-grade-up-conditions.md`・`docs/rocm-grade-up-conditions-v2-spec-proposal.md`
- `crates/autodiff/src/generate/mod.rs`・`crates/autodiff/src/nn/attention.rs`
- `.claude/rules/deps-policy.md`・`.claude/rules/coding-rust.md`・`.claude/rules/security.md`
