# エンジン単体の対照の Mac mini 測り直し（v0.15.0・#4314）

状態: **RULE 固定・未計測**

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
