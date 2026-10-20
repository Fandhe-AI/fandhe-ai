# cpu-gemm-reuse-fresh-3187

イシュー #3187（GB10 の CPU GEMM で reuse が fresh より遅い原因の帰属）の計測ログ置き場。
背景・仮説・限界は `docs/perf/cpu-gemm-reuse-fresh-inversion.md`。状態: **未実測**。

## 構成

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録の判定規則（実測前に固定。事後に緩めない） |
| `orchestrate.sh` | テストバイナリを 1 回ビルドし、独立 5 プロセスで実行して `run{1..5}.jsonl`・`env_info.txt`（`.rev-stamp`・RULE.txt の sha256 を含む）・`load_gate.log`・`load_gate_status.txt` を残す。`.rev-stamp` が無ければ開始前に停止。`RAYON_NUM_THREADS` 設定時は拒否 |
| `aggregate.py` | 5 run を集計（python3 標準ライブラリのみ。欠落・不正・n_samples≠80・非有限/非正の median_s・checksum の bit 不一致・ゲート状態欠落は fail-closed。`--self-test` あり） |
| `gb10/` | 実機実測時の生ログ（現状は未実測のため存在しない） |

## 実行手順（GB10 セッション）

```bash
# 計測ツリーのルートに .rev-stamp（1 行目 HEAD・2 行目 dirty=<件数>）を置く
bash docs/perf/logs/cpu-gemm-reuse-fresh-3187/orchestrate.sh gb10
python3 docs/perf/logs/cpu-gemm-reuse-fresh-3187/aggregate.py \
  docs/perf/logs/cpu-gemm-reuse-fresh-3187/gb10 > docs/perf/logs/cpu-gemm-reuse-fresh-3187/gb10/aggregate.md
# ページ確保の補助（別ディレクトリ・1 run 相当。系列には混ぜない）
GEMM_REUSE_MINFLT=1 bash docs/perf/logs/cpu-gemm-reuse-fresh-3187/orchestrate.sh gb10 --out <別ディレクトリ>
```

- 既存の `run{N}.jsonl` があると `orchestrate.sh` は中止する（差し替え禁止）
- 収録前にホスト名は `masked`、`$HOME` は `<home>`、ユーザー名は `<user>` へ置換される
- 計測中にビルドを並走させない（テストバイナリは開始時に 1 回だけビルド）
- affinity A/B（#2117）は `docs/perf/logs/cpu-gb10-affinity-ab-2117/README.md` の手順に従う
