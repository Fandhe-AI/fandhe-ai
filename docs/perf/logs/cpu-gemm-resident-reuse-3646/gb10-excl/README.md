# GB10 専有系列（#3646・3 回目の reopen。2026-10-28）

親ディレクトリ（`../gb10/` ほか）は常駐 GPU プロセス 3 件がある機体で取った**参考記録**（PR #3669）。本ディレクトリは
compute プロセス 0 件の GB10（`<cuda-node>`）で、`RULE.txt`（ブリッジ節のみ `RULE.v2.txt`）どおりに取り直した**専有系列**で、
親ディレクトリの回収物は書き換えていない。閾値・帰属規則・`aggregate.py`・各 judge は不変。

## 状態

**正式系列。** 6 系列すべてが `excl=pass`（`session-scripts/excl_status.txt`）、診断主系列の全 run が `gate=pass`、checksum は全 run bit 一致、
5 本の judge（`excl_check.py`・`bridge_judge.py`・`aggregate.py`・`gcpuinf_judge.py`・補助系列の `aggregate.py`）が exit 0。

| 項目 | 結果 |
|---|---|
| 専有（`excl_check.py`） | main・minflt・tunables・taskset・bridge・gcpuinf すべて pass。`gpu_poll.log` の 173 件（30 秒間隔・全系列の前・途中・後）が `rc=0 apps=0`。境界スナップショット 12 件も 0 件 |
| 診断主系列（5 run） | N=256〜4096 すべて再現（ratio 1.064・1.064・1.021・1.038・1.203）。帰属は N=4096 のみ H1（`download`）支持（rho_DL 0.72）。N=256〜2048 はブリッジ fail のため保留 |
| ブリッジ v2 | N=4096 のみ pass（G_bench / G_diag = 1.135）。N=256〜2048 は fail（8.1〜20.1 倍） |
| 補助系列 | minflt 1 run・tunables 5 run・taskset 5 run（判定外。各 `aggregate.md`） |
| G-CPU-INF | 純カーネル差 / D = 1.34〜1.92、構成差 / D は負。相手経路は追加確認（shim）で NVPL と確認。分類は doc §9.4 |
| 実行できなかった項目 | perf（`perf_event_paranoid=4` で拒否。sudo は使っていない）。RULE 固定手順の LD_DEBUG は sgemm 系の解決先を拾えなかった |

解釈・帰属の表・限界は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §9 を正とする。

## 手順（`../rev-stamp-verification-3646.md` の「専有系列」節に同定情報）

- ノード: 事前に compute-apps 0 件・load1 0.03・hw-lock なし・torch 2.14.0+cu130 を確認（計数しない preflight）。計測用ツリーは固定コミットの `git archive` のみ
- 自プロセス由来の compute-apps が出ないことを、正式セッション前に計数しないスモークで確認した（診断テスト・bench-fandhe・PyTorch 経路。1 秒間隔 64 回すべて 0 件。出力は削除済み）
- `session-scripts/session-3646-excl.sh`（開始前に固定コミット `555a445a`）。poller（30 秒間隔）と系列ラベル `current_series` で、系列ごとの途中の出現を判定する
- G-CPU-INF (iii) の補足 shim（`gb10-gcpuinf/blas_probe/probe_out.txt`）はセッション後に同じ `blas_probe/run_probe.sh` を実行した追加確認（計測外・時間は測らない）。
  出力は `../gb10-gcpuinf/blas_probe/probe_out.txt`（PR #3669 の参考記録）と sha256 まで同一
- 回収物は `$HOME` を `<home>` に置換済み（数値は書き換えていない）

## 構成

`gb10/`（診断主系列 5 run）・`gb10-minflt/`・`gb10-tunables/`・`gb10-taskset/`・`gb10-bridge/`（bench-fandhe の生 JSONL・`bridge_judge.md`）・
`gb10-gcpuinf/`（`gcpuinf_judge.md`・`fixedcost-2105/`・`blas_path.log`・`blas_probe/`）・`session-scripts/`（実行スクリプト・`steps.log`・`gpu_poll.log`・`gpu_procs.txt`・`excl_status.txt`・`load_gate_bench.log`・`orch-*.log`・`versions.txt`）
