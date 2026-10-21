#!/usr/bin/env bash
# イシュー #3184 セッション共通ヘルパー（Mac mini 上で source する）。
# 役割: 外側負荷ゲート（RULE 3: load1 < 3.0 を 30 秒間隔・最大 30 分待ち machine_gate.log へ記録）、
# 並走プロセス件数の記録、step ログ。継承スクリプトの閾値は変えない。bash 3.2 互換。
R=${HOME}/work/optin-3184
COL=${R}/collect
mkdir -p "${COL}/macmini"
step() { echo "$(date -u +%FT%TZ) $*" >>"${COL}/macmini/steps.log"; }
outer_gate() { # outer_gate <series>
  local s=$1 i=0 l1 ok=0
  while [ "${i}" -le 60 ]; do
    l1=$(sysctl -n vm.loadavg | awk '{print $2}')
    if awk -v a="${l1}" 'BEGIN{exit !(a<3.0)}'; then ok=1; break; fi
    sleep 30; i=$((i+1))
  done
  echo "series=${s} outer_gate_ok=${ok} load1=${l1} at=$(date -u +%FT%TZ)" >>"${COL}/macmini/machine_gate.log"
  echo "series=${s} watchlist_procs=$(ps -axo comm= | grep -ciE 'cargo|rustc|bench|python|gemm_')" >>"${COL}/macmini/preflight-procs.txt"
  df -h "${HOME}" | tail -1 >>"${COL}/macmini/df-${s}.txt"
}
