# cpu-predict-resident-fixedcost-2105

イシュー #2105（CPU `predict_resident` 固定費の切り分け）の計測ログ置き場。
背景・仮説・限界は `docs/perf/cpu-predict-resident-fixedcost.md`。

## 構成

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録の判定規則（実測前に固定。事後に緩めない） |
| `orchestrate.sh` | テストバイナリを 1 回ビルドし、独立 5 プロセスで実行して `run{1..5}.jsonl`・`env_info.txt`・`load_gate.log`・`load_gate_status.txt`（run ごとの負荷ゲート状態。未通過は集計出力に「参考扱い」と表示）を残す |
| `aggregate.py` | 5 run を集計（python3 標準ライブラリのみ。欠落・不正・n≠80・非有限/非正の median_s・checksum 不一致・ゲート状態欠落は fail-closed。`--self-test` あり） |
| `m4max/`・`gb10/` | 実機実測時の生ログ（現状は未実測のため存在しない） |

## 実行手順（実機セッション）

```bash
# M4 Max
bash docs/perf/logs/cpu-predict-resident-fixedcost-2105/orchestrate.sh m4max
python3 docs/perf/logs/cpu-predict-resident-fixedcost-2105/aggregate.py \
  docs/perf/logs/cpu-predict-resident-fixedcost-2105/m4max > .../m4max/aggregate.md
# GB10（load1 < 1.0 の専有ゲート付き）
bash docs/perf/logs/cpu-predict-resident-fixedcost-2105/orchestrate.sh gb10
# H3 の補助（別ディレクトリ・1 run 相当。系列には混ぜない）
FIXEDCOST_MINFLT=1 bash .../orchestrate.sh gb10 --out <別ディレクトリ>
```

- 既存の `run{N}.jsonl` があると `orchestrate.sh` は中止する（差し替え禁止）
- 収録前にホスト名は `masked`、`$HOME` は `<home>` へ置換される（`env_info.txt`・`run*.err`）
- 結果は `docs/perf/cpu-predict-resident-fixedcost.md` §6 に RULE.txt の規則で記入する
