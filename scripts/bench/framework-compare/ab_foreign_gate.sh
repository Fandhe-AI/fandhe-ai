# shellcheck shell=sh
# イシュー #4160: Mac mini（<metal-node>）の各計測プロセスを他ジョブの前提ゲートで前後から挟む関数群（source 専用）。
# 役割: orchestrate_macmini.sh（docs/perf/logs/metal-steel-large-n-ab-macmini-4160/）と
#   run_ab_steel_large_n_gemm_metal.sh から source され、計測プロセスの前に「他ジョブが CLEAN かつ load1 < 閾値」を
#   同じループで待ち、後に他ジョブの有無を 1 回確かめる。判定そのものは docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py
#   （本スクリプトは変えず呼ぶだけ。FOREIGN_GATE にその絶対パスを渡す）。経緯: 2026-11-10 に hw-lock を取らない他ジョブが
#   計測と重なり load1 では検出できなかった（#4151・#4153・#4154。所有者指示「各 round を他ジョブの前提ゲートで前後から挟む。
#   ゲートは厳しくする方向にだけ変える」）。load1 の閾値・待ちの上限（30 秒間隔×60 回＝30 分）は呼び出し元の既存のまま。
# 記録の文法: 呼び出し元の load ゲートのログ（`<label> gate=<status> ...`。aggregate.py が読む）に乗せる。通過は従来と同じ
#   `gate=pass` の 1 行のまま、通過以外だけ `gate=timeout|invalid|unavailable` に `reason=...` を足す。終了後の FOREIGN は同じ label の
#   `gate=invalid reason=foreign-after` を足す。部分系列で続ける経路は作らない（記録して続けるのは既存の GATE-TIMEOUT／GATE-INVALID と同じ。
#   そうした行を含む系列は aggregate.py が正式系列にしない）。
# 出力: 各回のゲート出力（マスク済み）は呼び出し元が決める FG_DIR 以下に `<name>-before-a<k>.txt`・`<name>-after.txt` で残す。
#   FOREIGN_GATE の絶対パス（$HOME を含みうる）はログに書かない（sha256 のみ）。ゲートの標準出力・標準エラーは捨てる。
# 解釈の失敗は fail-closed: 終了コードが 0／1 以外、出力が無い、1 行目が終了コードと食い違う（0 は CLEAN、1 は FOREIGN）なら error。
# 前提: POSIX sh 互換（bash 3.2・dash）。変数は ${VAR}。呼び出し元が set -u でも動く。

# shellcheck disable=SC2034  # FG_* は source 元が読む
FG_DIR=""
FG_RC=2
FG_STATUS=""
FG_REASON=""
FG_L1="NA"
FG_WAITED=0

# fg_require <FOREIGN_GATE> <FG_DIR>: パス検査と出力ディレクトリの新規作成（既存は差し替え禁止で拒否）。失敗は 1。
fg_require() {
  if [ -z "${1:-}" ]; then
    echo "error: FOREIGN_GATE（foreign_job_gate.py の絶対パス）が未指定" >&2; return 1
  fi
  case "$1" in
    /*) ;;
    *) echo "error: FOREIGN_GATE は絶対パスのみ" >&2; return 1 ;;
  esac
  case "$1" in
    *\"*|*\'*|*\\*|*\$*|*\`*|*' '*) echo "error: FOREIGN_GATE に禁止文字を含む" >&2; return 1 ;;
  esac
  if [ ! -f "$1" ]; then
    echo "error: FOREIGN_GATE のファイルが無い" >&2; return 1
  fi
  if [ -z "${2:-}" ]; then
    echo "error: ゲート出力ディレクトリが空" >&2; return 1
  fi
  if [ -e "$2" ]; then
    echo "error: ゲート出力ディレクトリが既にある（差し替え禁止）" >&2; return 1
  fi
  mkdir -p "$2" || return 1
  FG_DIR="$2"
  return 0
}

# fg_sha256 <file>: sha256 を標準出力へ。
fg_sha256() {
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# fg_run <before|after> <出力ファイル>: ゲートを 1 回実行し FG_RC に 0=CLEAN・1=FOREIGN・2=error を入れる。
fg_run() {
  FG_RC=2
  python3 -I "${FOREIGN_GATE}" --phase "$1" --out "$2" > /dev/null 2>&1
  fg_rc=$?
  if [ "${fg_rc}" -ne 0 ] && [ "${fg_rc}" -ne 1 ]; then return 0; fi
  [ -f "$2" ] || return 0
  fg_first=$(head -n 1 "$2" | awk '{print $1 " " $2}')
  if [ "${fg_rc}" -eq 0 ] && [ "${fg_first}" = "CLEAN $1" ]; then
    FG_RC=0
  elif [ "${fg_rc}" -eq 1 ] && [ "${fg_first}" = "FOREIGN $1" ]; then
    FG_RC=1
  fi
  return 0
}

# fg_wait_before <name> <load1 閾値>: 他ジョブが CLEAN かつ load1 < 閾値になるまで 30 秒間隔・最大 60 回（30 分）待つ。
# 結果: FG_STATUS=pass|timeout|invalid、FG_REASON（pass 以外。foreign-job|load1|foreign-gate-error|load1-unavailable）、
#   FG_L1（最後に読んだ load1。読めなければ NA）、FG_WAITED（待った秒数）。
fg_wait_before() {
  fg_name=$1
  fg_thr=$2
  FG_STATUS=timeout
  FG_REASON=""
  FG_L1=NA
  FG_WAITED=0
  fg_k=0
  while [ "${FG_WAITED}" -lt 1800 ]; do
    fg_k=$((fg_k + 1))
    fg_run before "${FG_DIR}/${fg_name}-before-a${fg_k}.txt"
    if [ "${FG_RC}" -eq 2 ]; then
      FG_STATUS=invalid; FG_REASON=foreign-gate-error; return 0
    fi
    FG_L1=$(sysctl -n vm.loadavg 2> /dev/null | awk '{print $2}')
    case "${FG_L1}" in
      ''|*[!0-9.]*) FG_STATUS=invalid; FG_REASON=load1-unavailable; FG_L1=NA; return 0 ;;
    esac
    fg_ok=$(awk -v a="${FG_L1}" -v t="${fg_thr}" 'BEGIN{print (a<t)?1:0}')
    if [ "${FG_RC}" -eq 0 ] && [ "${fg_ok}" = "1" ]; then
      FG_STATUS=pass; FG_REASON=""; return 0
    fi
    FG_REASON=load1
    [ "${FG_RC}" -eq 1 ] && FG_REASON=foreign-job
    sleep 30
    FG_WAITED=$((FG_WAITED + 30))
  done
  FG_STATUS=timeout
  return 0
}

# fg_after <name>: 計測プロセスの直後に 1 回確かめる。FG_RC=0 CLEAN／1 FOREIGN／2 error。
# 異常時に記録する reason は fg_after_reason が返す（foreign-after|foreign-gate-error）。
fg_after() {
  fg_run after "${FG_DIR}/$1-after.txt"
}
fg_after_reason() {
  if [ "${FG_RC}" -eq 1 ]; then echo foreign-after; else echo foreign-gate-error; fi
}
