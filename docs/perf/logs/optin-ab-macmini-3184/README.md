# optin-ab-macmini-3184

イシュー #3184（未計測の opt-in 候補 7 件を Mac mini で A/B）の補遺規則と収録先。
状態: **規則と after 腕 patch まで完了・実測は未実施（全ゲート undetermined〈未実測〉）**。

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 事前登録の補遺規則（継承 RULE は変更しない） |
| `on-host-arena.patch` | G1 after 腕（`HOST_ARENA_DEFAULT_ENABLED` を true） |
| `on-reduction.patch` | G2 after 腕。`@T@` は Stage 0 確定後に記入（未確定のままでは `git apply` できない） |
| `on-train-forward-encode.patch` | G3 after 腕 |

## 実行手順（Mac mini 上。継承スクリプトは編集しない）

1. 外側ゲート: `sysctl -n vm.loadavg` が load1 < 3.0 になるまで最大 30 分待つ。値を `machine_gate.log` へ記録。
2. `git bundle` で送り、`before`／`after` の 2 clone を作る（`.git` 付き）。after には対象 patch だけを当て、系列ごとに `git checkout -- .` で戻す。
3. CPU 系: G1 `run_ab_tape_arena_cpu.sh` → G2 Stage 0 `run_reduction_sweep.sh` → Stage 1 `run_ab_reduction_threshold_cpu.sh`（`AB_THRESHOLD_T`）。
4. Metal 系: G4 `run_ab_readback_metal.sh` → G5 `run_ab_readout_metal.sh` → G3 前提 → `run_ab_train_forward_encode_metal.sh` → G6 `orchestrate.sh gate`／`1..5` → G7 `orchestrate.sh m4max --out <dir>`。
5. 各系列の後に継承ゲートログの round 別 load1 から `machine_gate_ok` を記録し、`--target-dir` を削除して空き容量を確認。
6. 生ログをマスク（RULE 6）して本ディレクトリへ収録。結論は `docs/perf/optin-ab-macmini.md` へ。

実測ログ（`host-arena/` など）は未生成。
