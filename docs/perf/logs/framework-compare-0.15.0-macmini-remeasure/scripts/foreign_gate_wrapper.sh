#!/usr/bin/env bash
# 0.15.0 Mac mini 同席計測（#4177）: 他ジョブゲートの薄いラッパー（source 専用）。
# 役割: sha256 固定済みの docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py と
#   scripts/bench/framework-compare/ab_foreign_gate.sh（fg_run・fg_wait_before・fg_after）を、中身を変えずに呼ぶ。
#   呼び出し前に両方の sha256 を検査し、一致しなければ使わない（fail-closed）。扱いは #4168 と同じ
#   （docs/perf/logs/prepost-0150-4168/RULE.txt の [負荷ゲート]。ランナー run_ab_prepost_0150_4168.sh の gate_wait／ラウンド後の確認と同じ文法）:
#   - 各ラウンドの前: 他ジョブが CLEAN かつ load1 < 閾値（3.0）になるまで 30 秒間隔・最大 30 分待つ。超えたら GATE-TIMEOUT。
#   - 各ラウンドの後: FOREIGN（またはゲート自身の失敗）ならそのラウンドを無効にする。呼び出し元（macmini-0150b-loop.sh）が系列を打ち切る。
#   - 5/5 完走だけを採る（採否は collect.sh 以降・判定側の責務）。
# gate.log の文法（#4168 と同じ。通過は 2 行、異常は 1 行）:
#   round<N> gate=pass threshold=<閾値> waited_s=<秒> load1=<値>／round<N> foreign-before=clean／round<N> foreign-after=clean
#   round<N> GATE-TIMEOUT reason=<foreign-job|load1> load1=<値>／round<N> GATE-INVALID reason=<foreign-gate-error|load1-unavailable|foreign-after>
# 前提: Mac mini（sysctl・python3）。ゲートの出力は <FG_DIR>/r<N>-before-a<k>.txt・r<N>-after.txt（マスクは collect.sh）。
# 呼び出し元: macmini-0150b-loop.sh（本番）・self-test.sh（モックで判定分岐を検査）。
# bash 3.2 互換。

FGW_GATE_REL=docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py
FGW_LIB_REL=scripts/bench/framework-compare/ab_foreign_gate.sh
# RULE（#4168）が固定した値と同じ（変えるときは所有者承認。ゲートの中身は変えない）
FGW_GATE_SHA256=d8cf0fd60693be5ed16e2d0947e8a5a5a5c1918999eeb61dd635d9dbd7a91e04
FGW_LIB_SHA256=b7ae69bc4b32fa63887867822b97e1eeb9e900ed1a51a81053b231e8ff10f975

fgw_sha256() { if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }

# fgw_init <ROOT> <FG_DIR>: 両ファイルの sha256 を検査し、ab_foreign_gate.sh を source して出力ディレクトリを作る。失敗は 1。
fgw_init() {
  local root=$1 rel want got
  case "${root}" in /*) ;; *) echo "FOREIGN-GATE: ROOT は絶対パス" >&2; return 1 ;; esac
  for rel in "${FGW_GATE_REL}" "${FGW_LIB_REL}"; do
    if [ "${rel}" = "${FGW_GATE_REL}" ]; then want="${FGW_GATE_SHA256}"; else want="${FGW_LIB_SHA256}"; fi
    [ -f "${root}/${rel}" ] || { echo "FOREIGN-GATE: ファイルが無い: ${rel}" >&2; return 1; }
    got="$(fgw_sha256 "${root}/${rel}")" || { echo "FOREIGN-GATE: sha256 を取れない: ${rel}" >&2; return 1; }
    [ "${got}" = "${want}" ] || { echo "FOREIGN-GATE: sha256 不一致（固定値と違う）: ${rel}" >&2; return 1; }
  done
  # shellcheck source=/dev/null
  . "${root}/${FGW_LIB_REL}" || return 1
  FOREIGN_GATE="${root}/${FGW_GATE_REL}"
  fg_require "${FOREIGN_GATE}" "$2" || return 1
}

# fgw_before <round> <load1 閾値> <gate.log>: 戻り値 0 = 通過／1 = GATE-TIMEOUT／2 = GATE-INVALID。
fgw_before() {
  fg_wait_before "r$1" "$2"
  case "${FG_STATUS}" in
    pass)
      echo "round$1 gate=pass threshold=$2 waited_s=${FG_WAITED} load1=${FG_L1}" >> "$3"
      echo "round$1 foreign-before=clean" >> "$3"
      return 0 ;;
    timeout)
      echo "round$1 GATE-TIMEOUT reason=${FG_REASON} load1=${FG_L1}" >> "$3"
      return 1 ;;
    *)
      echo "round$1 GATE-INVALID reason=${FG_REASON}" >> "$3"
      return 2 ;;
  esac
}

# fgw_after <round> <gate.log>: 戻り値 0 = CLEAN／1 = FOREIGN／2 = ゲート自身の失敗。0 以外はそのラウンドを無効にする。
fgw_after() {
  fg_after "r$1"
  if [ "${FG_RC}" -eq 0 ]; then
    echo "round$1 foreign-after=clean" >> "$2"
    return 0
  fi
  echo "round$1 GATE-INVALID reason=$(fg_after_reason)" >> "$2"
  return "${FG_RC}"
}
