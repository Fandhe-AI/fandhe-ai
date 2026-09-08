#!/usr/bin/env bash
# イシュー #1253（PR #1459 codex-review P1 指摘の是正）:
# 排他計測契約の閾値（load average の gate 判定）は、待機フェーズ
# （wait_gate.sh）と実行中監視フェーズ（orchestrate.sh の run 監視ループ）
# の双方で同一でなければならない（同一契約の 2 箇所表現）。
# 従来は wait_gate.sh の GATE_THRESHOLD と orchestrate.sh の
# MONITOR_GATE_THRESHOLD を別名の変数として重複定義しており、片方だけ
# override すると待機時と実行中監視で異なる閾値になり得た
# （例: GATE_THRESHOLD=1.0 で再実行しても実行中監視は既定 2.0 のまま）。
# 本ファイルを両スクリプトが source することで閾値を一箇所に集約する。
# 呼び出し側が環境変数 GATE_THRESHOLD を明示指定していればそれを優先し
# （bash パラメータ展開の `:-` 既定値構文）、未指定時のみ既定値 2.0 を使う。
GATE_THRESHOLD="${GATE_THRESHOLD:-2.0}"
