# 学習ステップ全体の Graph capture の GB10 A/B（イシュー #3261）

## 状態

**未実測**（規則・ランナー・判定ツールの整備まで。GB10 での 5 round 計測と判定は未実施）。
実測後は本欄を判定（ADOPT／REJECT／UNDETERMINED）と `gb10/` の収録物で更新する。

## 収録物

- `RULE.txt`: 事前登録規則（計測前に固定。事後に緩めない）。
- `gb10/`（実測後に追加）: マスク済み生ログ・JSONL・`env_info`・`.rev-stamp`・前提ゲート R0〜R2 のログ・判定出力。

## 設計の要点

- on／off は同一バイナリ・同一コミット。腕の差は環境変数 `FANDHE_AI_CUDA_GRAPH_TRAIN` だけ（イシュー本文の「on／off 腕のコミット」は 1 つの `.rev-stamp` にまとまる）。
- 判定セルは `--mode chain`。capture は `DeviceParamStore::train_step_device_chain` の中にしかなく、bench の `reuse`／`fresh` は届かない。この 2 セルは stream 変更のみの対照セル。
- 基準値は #3257 の常駐チェーン（train CUDA 0.1343 ms、candle 比 2.02x。`../resident-chain-ab-3257/judge-3257-gb10.md`）。イシュー起票時の 0.331 ms は古い。
- ADOPT でも既定の反転（`TRAIN_GRAPH_DEFAULT_ENABLED`）は本件に含めない。設計 §8.6 で所有者判断のため、影響評価を含む別 PR とする。

## 実行手順（GB10 `spark-dbd9`。`docs/real-hardware-verification-env.md` §7）

1. 実機ロック `~/work/.fandhe-hw-lock` を取り、owner に `#3261 <UTC>` を書く。
2. 規則コミットを `git bundle` で転送して clone し、`.rev-stamp`（HEAD と `dirty=0`）を作って `.git/info/exclude` に足す。`bench-resident-chain.patch` を `git apply` する。
3. torch の専用 venv（2.14.0+cu130）を用意する。
4. `AB_FACADE_PATH=<abs>/crates/facade AB_TORCH_PY=<abs>/venv/bin/python AB_BENCH_PY=<abs>/bench_py.py bash scripts/bench/framework-compare/run_ab_train_graph_cuda.sh 3261-gb10` を `setsid nohup` で起動する。計測中はビルドを並走させない。
5. 回収後、ホスト名・`$HOME`・ユーザー名をマスクし、ロックを外す。
6. `python3 -I scripts/bench/framework-compare/judge_train_graph_ab.py --rule docs/perf/logs/train-step-graph-ab-3261/RULE.txt --dir docs/perf/logs/train-step-graph-ab-3261/gb10 --label 3261-gb10` の出力と終了コードを `judge-3261.md`／`judge-3261.exit` に保存する。
