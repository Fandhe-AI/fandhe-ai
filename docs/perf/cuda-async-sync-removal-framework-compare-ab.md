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

**追補（2026-09-01）**: Phase B（`fandhe-ai =0.5.0` へのピン更新。PR #1104。
イシュー #1011 の A/B 計測前提としてユーザー承認済み）・Phase C（DGX Spark
GB10 実機での before/after 計測）・Phase D（本文書 §4〜6 への記録）が完了
した。以下は Phase A 完了時点（本文書初版）の記述を原文のまま残す。

## 1. 計測環境

| 項目 | 値 |
| --- | --- |
| ホスト | DGX Spark GB10（実ホスト名は `docs/real-hardware-verification-env.local.md` 参照。本文書では非公開） |
| GPU | NVIDIA GB10（sm_121） |
| driver / CUDA | CUDA 13.0 系 |
| rustc | 1.97.0 |
| 計測日 | 2026-08-31 〜 2026-09-01 |
| before バージョン | `fandhe-ai =0.4.0`（crates.io。都度同期あり。PR #1104 の pin 更新コミット `e3682d5` 直前の main ツリー） |
| after バージョン | `fandhe-ai =0.5.0`（crates.io。2026-08-31 公開・都度同期なし。#1011 実装 + `006eeff`〈bench-fandhe の reuse 経路を 0.5.0 の `backward_device_param_store` 契約へ追従〉適用済みツリー） |

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

**追補（2026-09-01）**: before/after の実測は §4 で本 A/B 専用に
`results-dgx-ab-before-0.4.0.jsonl`／`results-dgx-ab-after-0.5.0.jsonl`
（`scripts/bench/framework-compare/results/raw/` にコミット済み）として
直接取得済みのため、本節（既存 `results-dgx-0.4.0.jsonl` 等からの転記）は
使用しなかった。以下は Phase A 完了時点の記述を原文のまま残す。

参考までに、本リポジトリに既にコミット済みの `results-dgx.jsonl`
（`framework_version: "0.3.0"`。#1083 の A/B 比較対象である 0.4.0 とは別
バージョンで、CUDA コンテキスト構築コストが毎回計測に乗る旧プロトコルの
可能性がある。単純比較不可）の train cuda fresh 実測は
`scripts/bench/framework-compare/results/raw/results-dgx.jsonl` を出典として
参照できるが、本 A/B の before 値としては使わない（§4 で 0.4.0 起点の
before を実測し直す）。

## 4. after 実測表

DGX Spark GB10 実機で `run_ab_train_cuda.sh`（fresh/reuse 各 5 回起動）を
before（0.4.0）/after（0.5.0）それぞれで実行し、`compare_ab.py` で 5 回
計測中央値を比較した結果（raw JSONL は
`scripts/bench/framework-compare/results/raw/results-dgx-ab-{before-0.4.0,after-0.5.0}.jsonl`
にコミット済み。skipped ログはいずれも空 = 失敗 0 件）。

### 1 step 総和（5 回計測の中央値）

| mode | before version | after version | before median | after median | after/before | 判定 |
|---|---|---|---|---|---|---|
| fresh | 0.4.0 | 0.5.0 | 12.404 ms (n=5) | 11.507 ms (n=5) | 0.928 | ok |
| reuse | 0.4.0 | 0.5.0 | 12.362 ms (n=5) | 5.436 ms (n=5) | 0.440 | ok |

### フェーズ分解（診断用・fresh・単発計測）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 4.1 us | 3.3 us | 0.810 |
| leaf_register | 0.9 us | 0.9 us | 1.026 |
| forward | 238.4 us | 177.5 us | 0.745 |
| loss_readout | 0.0 us | 0.0 us | 1.500 |
| backward | 12.057 ms | 11.239 ms | 0.932 |
| param_readout | 46.4 us | 26.0 us | 0.560 |
| host_sgd | 60.4 us | 53.0 us | 0.877 |
| apply_params | 0.3 us | 0.2 us | 0.853 |
| tape_drop | 2.7 us | 2.0 us | 0.745 |
| step_total | 12.414 ms | 11.504 ms | 0.927 |

### フェーズ分解（診断用・reuse・単発計測）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 3.3 us | 3.1 us | 0.942 |
| leaf_register | 0.1 us | 0.1 us | 1.000 |
| forward_resident | 266.7 us | 150.1 us | 0.563 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 12.043 ms | 5.243 ms | 0.435 |
| device_update | 59.5 us | 40.2 us | 0.676 |
| tape_drop | 3.3 us | 1.0 us | 0.312 |
| step_total | 12.379 ms | 5.439 ms | 0.439 |

reuse の `backward`（12.043 ms → 5.243 ms・0.435）が 1 step 短縮の支配項で
あり、都度同期除去（#1011）の効果が `stream.synchronize()` を最も高頻度に
挟んでいた backward 経路に集中して現れている。fresh 側は forward・
param_readout 等にも短縮が分散するが、backward の絶対時間（約 11〜12 ms）
自体が支配的なため 1 step 総和の短縮率（0.928）は reuse（0.440）より小さい。
**計測条件の注意**: after 側の reuse 計測は `006eeff`（`bench-fandhe` の
`--mode reuse` を 0.5.0 の `Op::LinearResident`／`backward_device_param_store`
契約へ追従させた修正）適用後のツリーで取得した。before 側は 0.4.0 の
素の `Tape::backward` のままであり、この API 呼び出し差自体は本 A/B が
比較したい「都度同期の有無」とは独立な契約変更だが、before/after 間で
計測対象コード（`main.rs` の `run_train_reuse`）が完全に同一ではない点は
計測条件として明記する。

## 5. 数値一致確認

- 最終 loss（checksum）の複合判定（相対誤差 1e-3 未満 または 絶対誤差
  1e-5 未満）: `compare_ab.py` が fresh・reuse 双方で `ok` と判定
  （`python3 compare_ab.py results/raw/results-dgx-ab-before-0.4.0.jsonl
  results/raw/results-dgx-ab-after-0.5.0.jsonl` の終了コード 0）
- `cargo test -p fandhe-ai-backend-cuda -- --ignored`（#1067）: GB10 実機で
  20 passed / 9 failed。失敗内訳は以下のとおりで、いずれも #1011 の
  同期契約変更とは独立の既知事象:
  - `wmma_tf32` 系: provenance 未確定の fail-closed 判定（#1102 で追跡中）
  - `jit_cache_bench` 系: キャッシュルートの pin 設定に起因する
    `os error 22`（実行環境要因。#1011 の変更対象外）

## 6. #1011 受入条件 2 の判定

- 短縮の有無・比率: fresh は after/before 0.928（約 1.08 倍）、reuse は
  after/before 0.440（約 2.3 倍短縮）。いずれも 1 step 総和が短縮しており、
  受入条件 2「MLP 学習 1 step（CUDA）が実測で短縮する」を実践規模の
  ワークロードで確認できた
- 判定（クローズ可否）: 短縮確認済み・クローズ可（最終判断は #1011 側で
  main が行う）

## 7. 未実施事項

- `has_async_alloc` プローブ（設計文書 §3 保留・#1014 T4）は本セッションで
  取得していない。記録先は設計文書 I4（別 PR。本文書のスコープ外）のまま
- Metal 側の同種 A/B 計測はスコープ外（本文書は CUDA 限定）
- reuse 側の after 計測は `006eeff`（`backward_device_param_store` 追従）
  適用後のツリーであり、before（0.4.0・素の `Tape::backward`）との
  API 呼び出し差は §4 の計測条件の注意に明記した

## 8. 出典

- `docs/backend-cuda-async-execution-design.md`（#1012 設計・§12 実装記録）
- `docs/perf/cuda-async-sync-removal-rtx3060.md`（RTX 3060 トイモデル計測・
  申し送り事項の出典）
- `scripts/bench/framework-compare/README.md`「A/B 計測（都度同期廃止・
  イシュー #1083）」節
- `scripts/bench/framework-compare/run_ab_train_cuda.sh`・`compare_ab.py`
- `.claude/rules/deps-policy.md` 第 9 区分（`fandhe-ai` ピン更新の承認契約）
- `docs/crates-io-publishing-order.md`（crates.io 公開順序・版数運用）
