# エンジン単体の対照の Mac mini 測り直し（v0.15.0・#4314）

状態: **5/5 ラウンド実測済み・集計済み**（系列 2026-11-15T03:56:37Z〜04:24:51Z・収録 PR #4316）

スコアボード 0.15.0（`docs/perf/logs/framework-compare-0.15.0-remeasure/scoreboard/`）の「エンジン単体の対照」節の 2 カード
（CPU GEMM vs gemm crate／matrixmultiply・Metal GEMM vs MLX f32／PyTorch MPS f32）は、v0.14.0 のソースで 2026-11-08 に測った値を
「0.15.0 未再計測」と明記して流用している（所有者判断 #4179）。0.15.0 は SME 既定 ON と Metal STEEL が入ったため、
タグ v0.15.0（`22a475e2d`）のソースで Mac mini（`<metal-node>`。Apple M4・10 コア・16 GB）で測り直す。

- 規則: `RULE.txt`（本ディレクトリを足す PR のマージ時刻で固定）
- 前回: `../2026-11-08-macmini/`（v0.14.0・PR #4139）。ハーネスの src・example・Python スクリプトは v0.14.0 と v0.15.0 で差分なし
- 実行: `scripts/orchestrate.sh`（prepare → series）。前回からの変更点はスクリプト冒頭のコメントにある
  - 他ジョブゲート（#4177 と同じ。`scripts/foreign_gate_wrapper.sh` は #4177 の byte コピー）
  - 環境変数検査（`scripts/env_guard.sh` も #4177 の byte コピー）
  - venv の freeze の byte 一致検査
  - 打ち切りは系列全体で、5/5 だけを採る
- 集計: `scripts/aggregate.py`（前回と byte 一致）。ALL-DONE の系列だけに行う
- 道具の検査: `bash scripts/self-test.sh`（Linux 上のモック。計測しない）
- 計測後に `env/`・`out/` を収録し、結果をこの README と `docs/perf/oss-gemm-comparison-baseline.md`・スコアボード 0.15.0 のカードに反映する

## 結果（2026-11-15）

集計表は `docs/perf/oss-gemm-comparison-baseline.md`「Mac mini（Apple M4）での測り直し（v0.15.0・2026-11-15）」節。

- 規則: `RULE.txt`（2026-11-15T03:51:57Z 固定・PR #4315 のマージで計測前に main へ）
- 対象: タグ v0.15.0 の git archive。ツリー指紋 `014866ab…d8f4`（2705 行）は `env/tree-fingerprint-{prepare,before,after}.txt` の 3 本で一致し、RULE の値と同じ
- venv の freeze（venv-torch・venv-mlx）は prepare・開始時・終了時とも参照と byte 一致（`env/pip-freeze-*.txt`）
- 他ジョブゲート: 5 ラウンドとも `gate=pass`・`waited_s=0`（load1 2.13・2.42・0.68・0.92・1.48）、foreign-before／after は全ラウンド clean（`out/gate.log`）。`loop.log` は `ALL-DONE completed=5/5`
- 終了コード: oss-gemm-compare は全ラウンド exit=1（出力不一致による。規則どおり記録のみ）。他の 4 ハーネスは全ラウンド exit=0
- 集計: `python3 -I scripts/aggregate.py out 5` → `aggregate.out`（`status=formal`）。`--self-test` も統合側で再実行し `self-test ok`
- SME: `env/env_info_*.txt` の sysctl（FEAT_SME・SME_F32F32）は 1・1。SME 経路と Metal STEEL が実際に選ばれたかはこの計測では確かめていない
- `env/` のホームディレクトリのパスは `~`、ユーザー名は `<user>` に置き換えた（`out/`・`loop.log` は回収したまま）
