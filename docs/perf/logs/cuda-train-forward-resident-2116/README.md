# cuda-train-forward-resident-2116

イシュー #2116（CUDA train `forward_resident` 内訳診断と fresh `param_readout` の診断）の計測ログ置き場。
背景・仮説・限界は `docs/perf/cuda-train-forward-resident-diagnosis.md`。

## 構成

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録の判定規則（実測前に固定。事後に緩めない） |
| `orchestrate.sh` | 上位（facade）・下位（backend-cuda lib）の 2 テストバイナリを 1 回ずつビルドし、独立 5 回ずつ実行して `run{1..5}.facade.jsonl`・`run{1..5}.backend.jsonl`・`env_info.txt`・`load_gate.log`・`load_gate_status.txt` を残す |
| `aggregate.py` | 5 run を集計（python3 標準ライブラリのみ。欠落・不正・n≠80・非有限の median_s・checksum 不一致・ゲート状態欠落は fail-closed。`--self-test` あり） |
| `gb10/`・`rtx3060/` | 実機実測時の生ログ（現状は未実測のため存在しない） |

## 実行手順（実機セッション）

```bash
# GB10（load1 < 1.0 かつ GPU utilization 0% の専有ゲート付き。判定対象）
bash docs/perf/logs/cuda-train-forward-resident-2116/orchestrate.sh gb10
python3 docs/perf/logs/cuda-train-forward-resident-2116/aggregate.py \
  docs/perf/logs/cuda-train-forward-resident-2116/gb10 > docs/perf/logs/cuda-train-forward-resident-2116/gb10/aggregate.md
# GB10 以外（record_only・参考）
bash docs/perf/logs/cuda-train-forward-resident-2116/orchestrate.sh rtx3060
```

- 既存の `run{N}.*.jsonl` があると `orchestrate.sh` は中止する（差し替え禁止）
- 収録前にホスト名は `masked`、`$HOME` は `<home>` へ置換される（`env_info.txt`・`run*.err`）
- 結果は `docs/perf/cuda-train-forward-resident-diagnosis.md` §6 に RULE.txt の規則で記入する
- 出力形式の動作確認のみ CPU でできる: `TRAIN_FWD_DIAG_KIND=cpu cargo test --release -p fandhe-ai --test cuda_train_forward_resident_diag cuda_train_forward_resident_phases -- --ignored --nocapture --test-threads=1`（判定外）
