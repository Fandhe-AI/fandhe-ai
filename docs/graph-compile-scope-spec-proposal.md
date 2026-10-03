# 汎用グラフコンパイル（torch.compile 相当）の範囲提案と承認依頼（#2615）

基準コミット: `7aab882f`（`origin/main`）。`docs/spec` submodule は本 worktree では未取得のため、spec の行番号はメイン checkout の `docs/spec/04-requirements.md` で確認した値（REQ-9 対象外 `:233`、REQ-12 `:279-289`）。submodule 更新で行番号はずれうるため、参照時は §2 の確認コマンドで再確認すること。

## §0 結論（最初に読む）

- **コード変更なし**（`crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`・tolerance／baseline・ガードレール閾値は変更していない）。成果物は本 doc、決定記録への #2615 追補、`docs/README.md` の索引、#2615 への承認依頼コメントである。
- **区分 B（既存の融合・CUDA Graph capture の延長）は spec 改定が不要**で、承認事項ごとに着手可否が決まる（§3）。ただし B-3（学習 step 部分）と B-5 は前提ゲートが未充足のため、承認だけでは着手できない。
- **汎用グラフ JIT（トレース・グラフ書き換え・コード生成・利用者向け `compile` 入口）は spec 改定が要る**（§4）。REQ-9「引き続き対象外」（`:233`）と REQ-12「利用者が明示的に融合を制御する API は提供しない」（`:285` 付近）の両方に触れるため。
- spec 改定案は 2 分岐を用意した（§5）。**推奨は分岐 (i)「対象外として確定」**（未承認）。分岐 (ii)「範囲を限った汎用 JIT を対象内にする」の文案も残す。**いずれも spec リポへ起票していない**。
- 承認依頼は #2615 へのコメントで行う（§7）。**承認は 1 件も得ていない**。承認の代行はしない。

## §1 位置づけ

- イシュー #2615（親 #2606 Phase 5・ルート #2499）。Phase 5 は「仕様外・依存承認が要る項目の提案と条件検証」で、成果物は提案文書とユーザー承認依頼まで。
- 先行: #1632（`docs/autodiff-graph-optimization-scope-decision.md`。区分 A 実装済み／B 延長候補／C 非目標）、#1962（`docs/facade-inference-serving-scope-decision.md` §2.3・§6。区分 B のゲート状況）、#2085（B-1 実装）、#2115（推論チェーンの CUDA Graph forward capture）、#2194（`docs/functorch-serving-hub-non-target-spec-proposal.md`。vmap 等のプログラム変換の非目標化提案）。
- 提案形式 (b) は #2086（`docs/tokenizer-non-target-spec-proposal.md`）・#2194 と同じ。実装リポ側 doc を出典とし、spec 側には短い規定だけを追記する形式（案 (a)＝spec 本体の要件・判定式の改定とは対）。
- spec 側 REQ-9 の文言は「新規 JIT を作らず既存の融合・CUDA Graph capture の延長で扱う範囲を実装リポ側で整理する」。実装リポ側の整理は #1632 で済んでいるが、spec 側の「整理する」は閉じていない。これを閉じるのが本提案である。

## §2 事実（基準コミット `7aab882f` で確認）

| # | 事実 | 確認コマンド |
|---|---|---|
| H1 | `Op::is_lazy_elementwise` は `Add`／`Mul`／`Relu`／`Exp`／`Tanh` の 5 演算のまま（B-2 未着手）。`crates/autodiff/src/tape.rs:1466` | `grep -n "fn is_lazy_elementwise" -A6 crates/autodiff/src/tape.rs` |
| H2 | B-1 は #2085 で実装済み。opt-in は `backend-cuda`／`backend-metal` の `fused_elementwise::set_gpu_elementwise_fusion_enabled`（クレート内 `pub`）のみで、`crates/facade/src` に再公開はない。GB10／M4 Max は未実測（`docs/perf/logs/gpu-elementwise-fusion-2085/README.md`） | `grep -rn set_gpu_elementwise_fusion_enabled crates/facade/src`（0 件） |
| H3 | #2085 の決定記録追補は新規 `unsafe` 4 か所を「本 PR のユーザー承認事項 (3)」と書く。**承認が実際に得られたかは記録上確認できない**。本 doc は「取得済み」とは記さない | 決定記録 `#2085 追補` |
| H4 | 推論チェーン（`predict_device_chain`）の CUDA Graph forward capture が #2115 で別機構として実装済み（環境変数 `FANDHE_AI_CUDA_GRAPH_INFER`・既定 OFF・facade 公開面変更なし・GB10 未実測）。学習 step の forward／backward capture は `d_input`／loss の常駐化ゲートが未充足 | `docs/perf/infer-chain-graphcapture-cuda-ab.md`・`docs/inference-chain-single-sync-design.md`・`docs/facade-inference-serving-scope-decision.md` §2.3 |
| H5 | B-4 の `cuGraphExecUpdate_v2` は cudarc の安全 API になく、新規 `unsafe` が必要（未実装） | `sed -n 1,20p crates/backend-cuda/src/graph.rs` |
| H6 | B-6（Metal encode-only 延長）は #1690・#1912 が REJECT の最新記録 | `docs/backend-metal-command-batching-design.md`・`docs/facade-inference-serving-scope-decision.md` §6 |
| H7 | facade の既存 opt-in setter は composition root の `AtomicBool` 型。GPU elementwise 融合・推論 capture の setter はない | `grep -n "^pub fn set_" crates/facade/src/lib.rs` |
| H8 | spec: REQ-9 対象外 `:233`、REQ-12 `:279-289`（2026-08-08 注記・「利用者が明示的に融合を制御する API は提供しない」） | メイン checkout で `grep -n "汎用グラフ JIT" docs/spec/04-requirements.md` |
| H9 | スコアボード「推論・サービング」行の fandhe-ai 列に「グラフコンパイルなし」、「固有の仕組み」行に PyTorch `torch.compile／Inductor`・TF `XLA・tf.function グラフ` | `grep -n "グラフコンパイル" docs/perf/logs/framework-compare-0.10.0-remeasure/scoreboard/body_0100.html` |
| H10 | `docs/compat-api-scope.md` §2 は「汎用 JIT は引き続き対象外・区分 B は承認事項付きの別 issue」と書き、HEAD でも正しい | 同 doc |

## §3 区分 B: 承認だけで着手できる延長（spec 改定は全項目で不要）

| 候補 | HEAD 状況 | 残る前提ゲート | 承認事項 | 承認だけで着手できるか |
|---|---|---|---|---|
| B-1 GPU elementwise 融合 | 実装済み（#2085）。facade 未公開・実機未測 | 実機 A/B 計測（#2683 側の測定対象） | (1) facade への opt-in setter 再公開（`docs/compat-api-scope.md` §5 の手続き）、(2) 計測後の既定 ON 化の判断、(3) 新規 `unsafe` 4 か所の承認状況の確認（H3。記録上不明） | 公開は可。既定 ON 化は実測後 |
| B-2 `is_lazy_elementwise` への `Sub`／`Div`／`Rsqrt` 追加 | 未着手（H1） | CPU カーネル allowlist への追随、`ScalarBinaryOp`／`ScalarUnaryOp`（#1634）との重複整理 | `FusedOpKind`／`Op::is_lazy_elementwise` の拡張、数値判定が REQ-2 判定か bit 一致か | 可 |
| B-3 forward／backward capture | 推論チェーンの forward capture は #2115 で別機構として実装済み（H4）。学習 step 部分は未充足 | `d_input` の直接計算・loss のデバイス常駐化 | 学習 step 部分は前提ゲート充足後に再提案 | 学習 step 部分は**不可**（ゲート待ち） |
| 推論チェーン capture（#2115） | 実装済み・既定 OFF・facade 公開なし | GB10 での ADOPT 判定（#2683 側） | facade への opt-in setter 公開、ADOPT 後の既定化 | 公開は可。既定化は実測後 |
| B-4 `cuGraphExecUpdate_v2` | 未実装（H5） | なし | 新規 `unsafe` FFI の追加（`security.md`「unsafe」節のレビュー対象） | 可 |
| B-5 tape レベル構造キー | 未着手 | step 全体 capture との同時導入 | B-3 のゲート充足後に再提案 | **不可**（ゲート待ち） |
| B-6 Metal 側の同型延長 | #1690・#1912 REJECT（H6） | 新規の測定根拠 | なし（現時点で着手根拠がない） | 不可（根拠待ち） |

どの行も REQ-9／REQ-12 の改定を必要としない。理由: 新規 JIT を作らず、利用者向けの融合制御 API を作らず（opt-in は composition root の setter に限る。`docs/autodiff-graph-optimization-scope-decision.md` F9 の型）、既存の allowlist 方式の融合・capture の延長に収まるため。

## §4 汎用グラフ JIT: spec 改定が要るもの

対象（区分 C の範囲）: トレース → グラフ書き換え（演算子再配置・定数畳み込み・代数簡約）→ コード生成の一連、`compile(model)` 相当の利用者向け入口、dynamic shape の guard 再コンパイル。

区分 B で届かない理由と、衝突する要件:

- **REQ-9 対象外（`:233`）**: 汎用グラフ JIT は「新規 JIT を作らず」の範囲外。
- **REQ-12（`:285` 付近）**: 利用者が明示的に融合を制御する API を提供しない、という要件と、利用者向け `compile` 入口は両立しない。
- **REQ-1**: 完全自作コアで汎用コード生成器を抱える保守負担が大きい。
- **REQ-2**: 融合・書き換えによる丸め差が、バックエンド間の複合判定（相対 1e-3 または絶対 1e-5）へ広く波及する。単一の連続 K ループと異なる結合順序の扱い（`coding-rust.md` の baseline 方式）が全演算へ広がりうる。
- 隣接事項: REQ-12 の CubeCL 前提文言の v2 全面書き直しは spec 側が「次回 Phase 4 見直し」に委ねている。本提案には含めない。

## §5 spec (b) 形式文案（いずれも未起票）

行番号は基準時点の値。再確認コマンドは §2 H8。

### 分岐 (i)（推奨・未承認）: 汎用グラフ JIT を対象外として確定する

タイトル案: `REQ-9: 汎用グラフ JIT（torch.compile／tf.function 相当）を対象外として確定し、延長範囲を実装リポの区分 B に委ねる`

````markdown
## 背景
REQ-9「引き続き対象外」（基準時点 `:233`）は、汎用グラフ JIT を「新規 JIT を作らず既存の融合・CUDA Graph capture の延長で扱う範囲を実装リポ側で整理する」と書く。実装リポ側の整理は Fandhe-AI/fandhe-ai の `docs/autodiff-graph-optimization-scope-decision.md`（#1632）および `docs/graph-compile-scope-spec-proposal.md`（#2615）で完了した。

## 提案
`:233` の末尾の文を次に置き換える。
「汎用グラフ JIT（トレース・グラフ書き換え・コード生成・利用者向け `compile` 入口。`torch.compile`／`tf.function` 相当）は対象外とする。既存の融合・CUDA Graph capture の延長として扱える範囲は、実装リポ `docs/autodiff-graph-optimization-scope-decision.md` の区分 B を正とする。」

## 理由
- REQ-1: 完全自作コアで汎用コード生成器を抱える保守負担
- REQ-12: 利用者向けの融合制御 API を提供しない方針との整合
- REQ-2: 融合・書き換えによる丸め差が複合判定へ広く波及する

## 受け入れ基準への影響
なし（REQ-2 の判定式・tolerance は変更しない）。区分 B の各項目は個別の承認と issue で扱う。

## スコープ境界
Keras 風 `Sequential::compile(optimizer, loss)`（学習設定 API）は別物であり対象外に含まない。vmap 等のプログラム変換は #2194 の提案で扱う。

## 添付文書
実装リポの `docs/autodiff-graph-optimization-scope-decision.md`・`docs/graph-compile-scope-spec-proposal.md`
````

### 分岐 (ii)（対案・未承認）: 範囲を限った汎用 JIT を対象内にする

タイトル案: `REQ-9／REQ-12: 静的形状に限ったトレース再生と既存融合 IR の自動適用を対象内にする`

````markdown
## 背景
利用者が `torch.compile` 相当の入口でグラフ最適化を得られないことが、PyTorch／TensorFlow 水準への到達の差として残る。

## 提案（最小形）
- 対象内にする範囲: 静的形状に限ったトレースの再生と、既存カーネル・既存融合 IR（`crates/tensor-core/src/fusion/`）の自動適用。新規のコード生成器、dynamic shape の guard 再コンパイル、外部コンパイラは含めない。
- REQ-9: `:233` の汎用グラフ JIT を対象外列挙から外し、上記の範囲を対象内に追記する。
- REQ-12: 「利用者が明示的に融合を制御する API は提供しない」を、「融合の個別制御（融合の on／off、融合境界の指定）は提供しない。`compile` 相当の単一入口は composition root（facade）に限り提供する」に改める。
- REQ-2: 自動適用で結合順序が変わる経路は、`coding-rust.md` の baseline 非後退方式（実機実測値・人間承認必須）で判定する。tolerance 定数は変更しない。

## 承認事項
新規 `unsafe` の有無、融合カーネルキャッシュの上限と fail-closed（#2085 の先例: 256 エントリ）、allowlist 方式の維持（`security.md` A08 の迂回経路を作らない）、依存追加の有無（外部コンパイラが要る場合は `deps-policy.md` の承認が別途必要）。

## 添付文書
実装リポの `docs/graph-compile-scope-spec-proposal.md`
````

## §6 スコアボードへの影響（Phase 6 #2681／#2682 の再監査用）

対応表自体は Phase 6 の担当で、本 issue では編集しない。

- 分岐 (i): 「推論・サービング」行の「グラフコンパイルなし」と「固有の仕組み」行は、「対象外（spec で確定）」と注記できる。区分 B の実装は行の中身（推論の高速化）の改善として別に評価する。
- 分岐 (ii): 実装完了まで「部分的」のまま。静的形状に限るため、PyTorch／TF の dynamic shape 対応との差は残る。
- どちらの分岐でも、「行の判定値を『ある』にする」だけでは足りず、行の中身で判断するという #2499 の物差しに照らすと、(i) は「対象外と確定した差」、(ii) は「範囲を限った実装」と記録される。

## §7 ユーザー承認事項

承認済みの項目はない。

1. spec の扱いは分岐 (i)（推奨）と (ii) のどちらにするか。
2. 選んだ文案を Fandhe-AI/fandhe-ai-spec へ起票してよいか（実装エージェントは起票していない）。
3. 区分 B のうち着手を承認する項目はどれか（B-1 facade 公開・B-2・B-4 の新規 `unsafe`・#2115 facade 公開。B-3 学習 step 部分・B-5 は前提ゲート未充足のため対象外）。
4. 承認いただいた項目の実装 issue を起票してよいか。

承認依頼コメントは #2615 に同内容で投稿する。

## §8 スコープ外・申し送り

- 区分 B・汎用 JIT の実装、facade 公開面の拡張、既定 ON 化
- spec リポへの起票・PR、`docs/spec/` の編集
- 依存追加、tolerance／baseline／閾値の変更、新規 issue の起票（承認後に別途）
- 実機計測（B-1・#2115 の GB10／M4 Max は #2683 側）
- スコアボード・feature-matrix の更新（Phase 6）
- `docs/kernel-fusion.md` の陳腐化記述の是正（`docs/autodiff-graph-optimization-scope-decision.md` §8 草案 6 のまま）

## §9 OWASP Top 10 観点

- A01: ruleset・リポジトリ設定・spec リポには触れない。承認状況は事実だけを書く。
- A04: 分岐 (ii) はカーネル生成面・キャッシュ面を広げるため、キャッシュ上限・fail-closed・allowlist 維持を承認事項に明記した。
- A06: 依存は変えない。分岐 (ii) で外部コンパイラが要る場合は `deps-policy.md` の承認事項として記すのみ。
- A08: B-4 と #2085 の新規 `unsafe` の承認状況を事実どおりに書いた（H3）。
- A09: 実測値は推測で書かない。実機未測は「未実測（#2683 側）」と記す。

## §10 出典

- `docs/autodiff-graph-optimization-scope-decision.md`（#1632・#1962・#2085・#2615 追補）
- `docs/facade-inference-serving-scope-decision.md` §2.3・§6
- `docs/compat-api-scope.md` §2・§5
- `docs/functorch-serving-hub-non-target-spec-proposal.md`（#2194）
- `docs/tokenizer-non-target-spec-proposal.md`（#2086）
- `docs/perf/infer-chain-graphcapture-cuda-ab.md`・`docs/inference-chain-single-sync-design.md`
- `docs/perf/logs/gpu-elementwise-fusion-2085/README.md`・`docs/backend-metal-command-batching-design.md`
- `docs/perf/logs/framework-compare-0.10.0-remeasure/scoreboard/body_0100.html`
- `docs/spec/04-requirements.md` REQ-9・REQ-12（メイン checkout で参照。編集しない）
