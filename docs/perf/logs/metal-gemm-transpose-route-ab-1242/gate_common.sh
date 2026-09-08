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

# pgrep_or_fail: pgrep を実行し、終了コード 0（該当あり）・1（該当なし）は
# 正常として一致 PID を標準出力へ返し 0 を返す。2 以上（構文エラー・
# プロセス一覧の取得失敗）は「判定不能」として 1 を返す（出力なし）。
# PR #1459 codex-review 七度目の指摘の是正（イシュー #1253・P2）: 従来は
# `pgrep ... | wc -l`／`for pid in $(pgrep ...)` で出力だけを数えていた
# ため、取得失敗も「該当プロセスなし」と同じ 0 件になり、低負荷なら
# ゲート PASSED・実行中監視 ok と誤判定して排他条件を確認できていない
# 計測を有効扱いしうる。呼び出し側（wait_gate.sh・orchestrate.sh）は本
# 関数の失敗を fail-closed に扱う（ゲート不成立／run 判定不能）。
pgrep_or_fail() {
  local out rc
  out=$(pgrep "$@" 2>/dev/null)
  rc=$?
  case "$rc" in
    0|1)
      if [ -n "$out" ]; then printf '%s\n' "$out"; fi
      return 0
      ;;
    *) return 1 ;;
  esac
}
