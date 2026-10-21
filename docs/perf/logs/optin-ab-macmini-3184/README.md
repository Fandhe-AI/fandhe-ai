# optin-ab-macmini-3184

イシュー #3184（未計測の opt-in 候補 7 件を Mac mini で A/B）の補遺規則・実測ログ・補助ツールの収録先。
状態: **実測済み（2026-10-21）。全ゲートの Mac mini 判定は undetermined（補遺 RULE 3 の round 別 load1 < 3.0 を満たす正式系列なし）。G6 は INCOMPLETE、G7 は集計器 fail-closed。結論は `docs/perf/optin-ab-macmini.md`**。

| パス | 役割 |
|---|---|
| `RULE.txt` | 事前登録の補遺規則（継承 RULE は変更しない） |
| `on-host-arena.patch`・`on-reduction.patch`・`on-train-forward-encode.patch` | after 腕 patch。`on-reduction.patch` の `@T@` は Stage 0 の T_macmini = 16384 を記入済み（`reduction/on-reduction-filled.sha256`） |
| `aggregate_sweep_single.py` | Stage 0 の単機集計（継承 `aggregate_sweep.py` を import。`--self-test` あり） |
| `machine_gate_check.py` | 継承ゲートログから round 別 load1 < 3.0 を判定（`--self-test` あり）。結果は `macmini/machine_gate_ok.txt` |
| `split_by_task.py` | 混在 JSONL を task だけで分割する前処理（G4。2112 README §3 のコマンドを実行可能にする） |
| `session-scripts/` | Mac mini 上で実行したセッションスクリプト（`cpu_session.sh`・`cpu_session2.sh`・`metal_session.sh`・`lib.sh`・`prep.sh`） |
| `macmini/` | `uname.txt`（ホスト名マスク）・`rev-stamp.txt`（before の HEAD）・`machine_gate.log`・`preflight-procs.txt`・`df-*.txt`・`steps.log` ほか |
| `host-arena/`（G1）・`reduction/stage0|stage1/`（G2）・`encode-only/`（G3。`prereq/` に前提ゲート）・`readback/`（G4）・`readout/`（G5）・`steel/`（G6）・`verify-cache/`（G7） | 生ログ（マスク済み） |

## ファイル名と機体の対応

- 実機は全て **Mac mini（M4）**。`readback/`・`readout/` のファイル名にある `m4max` はスクリプトの直書きで、label に `macmini` を含めて区別した。
- G7 の `orchestrate.sh` は `host_kind` として `m4max` しか受け付けないため、トークンを流用して `--out` で出力先を指定した（`verify-cache/env_info.txt` の `host_kind: m4max` はこの流用）。
- 系列 label: G1 は `g1`（初回）・`g1w`（ウォーム）・`g1f`（再実行）。`g1f` の 1 回目は clone 内に未追跡の target があり継承スクリプトが停止した（`host-arena/stdout-g1f-attempt1-untracked-stop.log`）ため、`.git/info/exclude` に target を除外してから同じ label を再実行した（計測は 1 回目で始まっていない）。G2 Stage 1 は `g2w`（クリーンアップ前に中断した走行。結果未完）・`g2f`（正式 label）。
- 再実行は別 label で行い旧系列も全て残した。

## 実行手順（Mac mini 上。継承スクリプトは編集しない）

1. `git bundle`（main の commit）を rsync し、`before`／`after`／`diag` の 3 clone を作る（`.git` 付き）。
2. `ssh <node> 'uname -a'` で到達を確認し結果を保存する。
3. CPU 系: `session-scripts/cpu_session.sh`（G1 → G2 Stage 0）→ `aggregate_sweep_single.py --macmini <stage0>` → `cpu_session2.sh <T>`（G1 再走・G2 Stage 1）。
4. Metal 系: `metal_session.sh g4g5` → `g3g6g7`（G3 前提ゲート）→ `g3ab` → `g6g7`。
5. 回収後にマスク（ホスト名 → `masked`、`$HOME` → `<home>`、ユーザー名を除去）し、`machine_gate_check.py` で round 別の load1 を判定する。
6. G4 は `split_by_task.py` で分割後、2112 README §3 のコマンドで比較。G5 は `compare_readout_ab.py --device metal --sizes gate --threshold 1.00`。
