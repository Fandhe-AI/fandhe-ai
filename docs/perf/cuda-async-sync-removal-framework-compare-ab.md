# CUDA 都度同期除去（#1011）framework-compare 実践規模 A/B 計測記録

## 0. 位置づけ

`docs/backend-cuda-async-execution-design.md`（#1012 設計・#1013 実装）が
除去した CUDA 側の都度 `stream.synchronize()` について、#1011 の受入条件 2
「MLP 学習 1 step（CUDA）が実測で短縮し数値一致複合判定を維持する」を、
`scripts/bench/framework-compare` の実践規模ワークロード
（`bench-fandhe train cuda 64`。784→256→10 の 3 層 MLP・バッチ 64）で確認する
（イシュー #1083）。

`docs/perf/cuda-async-sync-removal-rtx3060.md`（RTX 3060・トイモデル
`BATCH=4, D_IN=8, D_HIDDEN=16, D_OUT=4`）は update フェーズ単体では約 1.7 倍
高速化を示したが、1 step 全体では明確な高速化を示せなかった。本文書は
「実践規模のワークロードで再計測しないと結論が出ない」という同記録の
申し送りを引き継ぐ。

**本 PR（イシュー #1083 の Phase A）時点の状態**: 本文書は手順・記入欄の
雛形のみを整備する。実測（§4）は次回 crates.io 公開後、`fandhe-ai` ピンの
更新（ユーザー承認必須。`.claude/rules/deps-policy.md` 第 9 区分）を経た
DGX Spark GB10 実機セッションで行う（§2 の Phase B/C）。

## 1. 計測環境（記入欄）

実機セッションで記入する。

| 項目 | 値 |
| --- | --- |
| ホスト | （DGX Spark GB10。実ホスト名は `docs/real-hardware-verification-env.local.md` 参照） |
| GPU | （sm_121 等） |
| driver / CUDA | |
| 計測日 | |
| before バージョン | `fandhe-ai =0.4.0`（crates.io。都度同期あり） |
| after バージョン | （次回公開版。都度同期なし・#1064 適用済み） |

## 2. 手順

### Phase A（本 PR。完了）

`scripts/bench/framework-compare/run_ab_train_cuda.sh`（fresh/reuse 各 5 回
起動）・`compare_ab.py`（before/after の 5 回中央値比較。fail-closed 判定）
を整備した。詳細は `scripts/bench/framework-compare/README.md`「A/B 計測
（都度同期廃止・イシュー #1083）」節を参照。

### Phase B（次回 crates.io 公開後・ユーザー承認後）

1. crates.io で新版公開を確認する（`docs/crates-io-publishing-order.md` §11）
2. `scripts/bench/framework-compare` の `fandhe-ai` ピンを新版へ更新する
   （`bench-fandhe/Cargo.toml`・`bench-fandhe/src/main.rs` の `VERSION`・
   `Cargo.lock`・`scripts/check-forbidden-deps.sh` の承認済みピン等。
   別 PR・ユーザー承認）

### Phase C（DGX Spark 実機セッション）

1. before（ピン更新前のツリー = 0.4.0）を実機へ転送し
   `bash run_ab_train_cuda.sh before-0.4.0` を実行する
2. after（ピン更新後のツリー）を同手順で転送し
   `bash run_ab_train_cuda.sh after-<新版>` を実行する
3. GPU 使用率 0%・他プロセス不在を事前確認し、before/after を同日・同一
   ノードで取得する
4. raw JSONL・skipped ログ・実行ログを回収する

### Phase D（記録・判定）

1. `python3 compare_ab.py results/raw/results-dgx-ab-before-0.4.0.jsonl results/raw/results-dgx-ab-after-<新版>.jsonl`
   の出力を §4〜6 と `results/summary.md`（環境として追記）へ記録する
2. 数値一致（最終 loss の複合判定）・`cargo test -p fandhe-ai-backend-cuda -- --ignored`
   （#1067 の順序依存回帰テスト）を実機で通し記録する
3. #1011 へ結果をコメントし、受入条件 2 を満たせばクローズ可否を判断する

## 3. before 参考値（既存データからの転記・参考情報）

**未転記（本 PR 時点）**: #1050（`results-dgx-0.4.0.jsonl` のコミット・
`results/summary.md` 環境追記）が本リポジトリへマージされた時点でコミット
済みの 0.4.0 実測 JSONL から転記する。捏造を避けるため、コミットされていない
値をここへ書き写さない（A08）。

参考までに、本リポジトリに既にコミット済みの `results-dgx.jsonl`
（`framework_version: "0.3.0"`。#1083 の A/B 比較対象である 0.4.0 とは別
バージョンで、CUDA コンテキスト構築コストが毎回計測に乗る旧プロトコルの
可能性がある。単純比較不可）の train cuda fresh 実測は
`scripts/bench/framework-compare/results/raw/results-dgx.jsonl` を出典として
参照できるが、本 A/B の before 値としては使わない（§4 で 0.4.0 起点の
before を実測し直す）。

## 4. after 実測表（記入欄）

実機セッション（Phase C/D）で `compare_ab.py` の出力を転記する。

| mode | before version | after version | before median (n=5) | after median (n=5) | after/before | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| fresh | | | | | | |
| reuse | | | | | | |

### フェーズ分解（診断用。§2 の同期点分析の裏付け）

| phase | before | after | after/before |
| --- | --- | --- | --- |
| | | | |

## 5. 数値一致確認（記入欄）

- 最終 loss（checksum）の複合判定（相対誤差 1e-3 未満 または 絶対誤差
  1e-5 未満）: `compare_ab.py` の判定結果を転記
- `cargo test -p fandhe-ai-backend-cuda -- --ignored`（#1067）: 実機実行結果

## 6. #1011 受入条件 2 の判定（記入欄）

- 短縮の有無・比率:
- 判定（クローズ可否）:

## 7. 未実施事項

- 本文書公開時点では §4〜6 は未実測（記入欄のまま）
- `has_async_alloc` プローブ（設計文書 §3 保留・#1014 T4）は Phase C で
  同時取得可能だが、記録先は設計文書 I4（別 PR。本文書のスコープ外）
- Metal 側の同種 A/B 計測はスコープ外（本文書は CUDA 限定）

## 8. 出典

- `docs/backend-cuda-async-execution-design.md`（#1012 設計・§12 実装記録）
- `docs/perf/cuda-async-sync-removal-rtx3060.md`（RTX 3060 トイモデル計測・
  申し送り事項の出典）
- `scripts/bench/framework-compare/README.md`「A/B 計測（都度同期廃止・
  イシュー #1083）」節
- `scripts/bench/framework-compare/run_ab_train_cuda.sh`・`compare_ab.py`
- `.claude/rules/deps-policy.md` 第 9 区分（`fandhe-ai` ピン更新の承認契約）
- `docs/crates-io-publishing-order.md`（crates.io 公開順序・版数運用）
