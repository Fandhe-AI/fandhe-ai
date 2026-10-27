# 学習ステップ全体の Graph capture の GB10 A/B（イシュー #3261）

## 状態

**実測済み・判定 ADOPT**（2026-10-25・GB10 `<cuda-node-2>`）。chain の ratio 0.836（5/5 round で < 1.00）・reuse 0.948・fresh 0.972・hard 条件と R0〜R2 はすべて合格。詳細は `judge-3261.md`（終了コード `judge-3261.exit` = 0）と設計 doc `docs/backend-cuda-graph-step-capture-design.md` §8.9。ADOPT でも既定（`TRAIN_GRAPH_DEFAULT_ENABLED`）は反転していない（設計 §8.6・所有者判断）。

## 収録物

- `RULE.txt`: 事前登録規則（計測前に固定。事後に緩めない）。
- `gb10/`: マスク済み生ログ・JSONL・`env_info`・`.rev-stamp`・前提ゲート R0〜R2 のログ。判定出力は `judge-3261.md`／`judge-3261.exit`。

## 設計の要点

- on／off は同一バイナリ・同一コミット。腕の差は環境変数 `FANDHE_AI_CUDA_GRAPH_TRAIN` だけ（イシュー本文の「on／off 腕のコミット」は 1 つの `.rev-stamp` にまとまる）。
- 判定セルは `--mode chain`。capture は `DeviceParamStore::train_step_device_chain` の中にしかなく、bench の `reuse`／`fresh` は届かない。この 2 セルは stream 変更のみの対照セル。
- 基準値は #3257 の常駐チェーン（train CUDA 0.1343 ms、candle 比 2.02x。`../resident-chain-ab-3257/judge-3257-gb10.md`）。イシュー起票時の 0.331 ms は古い。
- ADOPT でも既定の反転（`TRAIN_GRAPH_DEFAULT_ENABLED`）は本件に含めない。設計 §8.6 で所有者判断のため、影響評価を含む別 PR とする。

## 実行手順（GB10。実施は `<cuda-node-2>`。`docs/real-hardware-verification-env.md` §7）

1. 実機ロック `~/work/.fandhe-hw-lock` を取り、owner に `#3261 <UTC>` を書く。
2. 規則コミットを `git bundle` で転送して clone し、`.rev-stamp`（HEAD と `dirty=0`）を作って `.git/info/exclude` に足す。`bench-resident-chain.patch` を `git apply` する。
3. torch の専用 venv（2.14.0+cu130）を用意する。
4. `AB_FACADE_PATH=<abs>/crates/facade AB_TORCH_PY=<abs>/venv/bin/python AB_BENCH_PY=<abs>/bench_py.py bash scripts/bench/framework-compare/run_ab_train_graph_cuda.sh 3261-gb10` を `setsid nohup` で起動する。計測中はビルドを並走させない。
5. 回収後、ホスト名・`$HOME`・ユーザー名をマスクし、ロックを外す。
6. `python3 -I scripts/bench/framework-compare/judge_train_graph_ab.py --rule docs/perf/logs/train-step-graph-ab-3261/RULE.txt --dir docs/perf/logs/train-step-graph-ab-3261/gb10 --label 3261-gb10` の出力と終了コードを `judge-3261.md`／`judge-3261.exit` に保存する。

## 実施メモ

- runner コミット: `6a97087ac7e29381eb932584783dc2495bd889e5`（`git bundle` で転送した実 `.git` つきツリー。`.rev-stamp` は `gb10/rev-stamp-runner.txt`）。
- ノード: `<cuda-node-2>`（GPU 使用率 0%・load1 < 1.0 を事前確認。`.local.md` の `<cuda-node>` は他ワークロード常駐のため避けた）。実機ロックは 2026-10-25T22:55Z に取得し、回収後 23:07Z に解除。
- torch: 専用 venv に 2.14.0+cu130 を導入（実機の環境構築であり、リポジトリの依存変更ではない。版は `gb10/torch-version.txt`）。
- マスク: ホスト名を `<host>`、`$HOME` を `<home>` に置換した。JSONL・`rule_sha256`・`patch_sha256`・`bin_sha256`・PARAM_DIGEST・BITS 行は無加工。
- 収録しないもの: bench バイナリ・`target`・venv・`.local.md`。
- 回収日: 2026-10-26。run の差し替え・追加起動はしていない。
