#!/usr/bin/env bash
# イシュー #3184 再開分セッション共通ヘルパー（Mac mini 上で source する。RULE-reopen.txt 2・8）。
# 役割: 外側負荷ゲート（load1 < 3.0 を 30 秒間隔・最大 30 分待ち）、他セッションの並走プロセスが 0 件に
#   なるまでの待機（#3183 など同じ Mac mini を使う他の計測との衝突回避）、step ログ。
#   継承 A/B スクリプトの round 前クールダウン hook は AB_ROUND_COOLDOWN_MAX_LOAD1=3.0 で有効化する
#   （継承の閾値 8.0・gate_ok の意味は変えない）。bash 3.2 互換。
R=${HOME}/work/optin-3184
COL=${R}/collect-r2
M=${COL}/macmini-r2
mkdir -p "${M}"
export AB_ROUND_COOLDOWN_MAX_LOAD1=3.0
step() { echo "$(date -u +%FT%TZ) $*" >>"${M}/steps.log"; }
watch_procs() { ps -axo comm= | grep -ciE 'cargo|rustc|bench|python|gemm_'; }
outer_gate() { # outer_gate <series>
  local s=$1 i=0 l1 ok=0 p
  while [ "${i}" -le 60 ]; do
    l1=$(sysctl -n vm.loadavg | awk '{print $2}')
    p=$(watch_procs)
    if awk -v a="${l1}" 'BEGIN{exit !(a<3.0)}' && [ "${p}" = 0 ]; then ok=1; break; fi
    sleep 30; i=$((i+1))
  done
  echo "series=${s} outer_gate_ok=${ok} load1=${l1} watchlist_procs=${p} waited_s=$((i*30)) at=$(date -u +%FT%TZ)" >>"${M}/machine_gate.log"
  echo "series=${s} watchlist_procs=${p}" >>"${M}/preflight-procs.txt"
  df -h "${HOME}" | tail -1 >>"${M}/df-${s}.txt"
}
# cooldown_wait <logfile> <round>: G6 など継承スクリプトの外側で round 前に待つ（float の load1 を記録）
cooldown_wait() {
  # shellcheck source=/dev/null
  . "${R}/before/scripts/bench/framework-compare/ab_round_cooldown.sh"
  if [ "$2" = 1 ]; then ab_round_cooldown_init "$1"; else AB_COOLDOWN_LOG=$1; fi
  ab_round_cooldown "$2"
}
