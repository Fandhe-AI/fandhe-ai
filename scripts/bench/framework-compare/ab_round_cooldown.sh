#!/usr/bin/env bash
# イシュー #3184（再開分）: A/B 継承スクリプトの round 前クールダウン hook（既定 OFF）。
# 役割: 継承 A/B スクリプト（run_ab_tape_arena_cpu.sh・run_ab_reduction_threshold_cpu.sh・
#   run_ab_train_forward_encode_metal.sh・run_ab_readback_metal.sh・run_ab_readout_metal.sh）から source され、
#   各 round の先頭（継承の load ゲートより前）で load1 が閾値未満になるまで待ち、その値を専用ログへ記録する。
#   待機を足すだけで、継承の閾値（8.0 等）・gate_ok・series 判定は変えない（条件を厳しくする方向のみ）。
#   直前の cargo build や前 round の負荷が次 round の load1 に乗る問題（#3184 初回で補遺 RULE 3 の
#   事後検査が成立しなかった原因）を、計測の外側で吸収する。
# 既定 OFF: 環境変数 AB_ROUND_COOLDOWN_MAX_LOAD1 が未設定または空なら即 return し、ファイルも作らない
#   （既存の事前登録が依存する挙動を変えない）。
# 規約: docs/perf/logs/optin-ab-macmini-3184/RULE-reopen.txt。bash 3.2 互換・eval 不使用。
# 閾値は有限の正数かつ継承上限 8.0 以下のみ受理し、それ以外は fail-closed で exit する。
# 読取・待機は関数（ab_cooldown_read_load1／ab_cooldown_sleep）に分離し、self-test は関数の上書きで
# 差し替える（環境変数で負荷値を偽装できる経路は作らない）。

AB_COOLDOWN_INTERVAL_S=30
AB_COOLDOWN_MAX_WAIT_S=1800
AB_COOLDOWN_UPPER_BOUND=8.0
AB_COOLDOWN_LOG=""

ab_cooldown_read_load1() {
  local l1
  l1=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
  if [[ -z "$l1" && -r /proc/loadavg ]]; then l1=$(awk '{print $1}' /proc/loadavg); fi
  printf '%s' "$l1"
}
ab_cooldown_sleep() { sleep "$1"; }

# ab_round_cooldown_init <logfile>: 閾値検証とログ新規作成（既存ログは拒否）。OFF なら何もしない。
ab_round_cooldown_init() {
  AB_COOLDOWN_LOG=""
  [[ -n "${AB_ROUND_COOLDOWN_MAX_LOAD1:-}" ]] || return 0
  local thr=$AB_ROUND_COOLDOWN_MAX_LOAD1 log=${1:-}
  if [[ ! "$thr" =~ ^[0-9]+(\.[0-9]+)?$ ]] \
    || [[ "$(awk -v t="$thr" -v u="$AB_COOLDOWN_UPPER_BOUND" 'BEGIN{print (t>0 && t<=u)?1:0}')" != "1" ]]; then
    echo "error: AB_ROUND_COOLDOWN_MAX_LOAD1 は 0 より大きく ${AB_COOLDOWN_UPPER_BOUND} 以下の数値のみ（got: ${thr}）" >&2
    exit 1
  fi
  if [[ -z "$log" ]]; then echo "error: cooldown ログのパスが空" >&2; exit 1; fi
  if [[ -e "$log" ]]; then echo "error: cooldown ログが既に存在する（上書き禁止）: $log" >&2; exit 1; fi
  ( set -C; printf 'threshold=%s interval_s=%s max_wait_s=%s\n' "$thr" "$AB_COOLDOWN_INTERVAL_S" "$AB_COOLDOWN_MAX_WAIT_S" >"$log" ) \
    || { echo "error: cooldown ログの排他作成に失敗: $log" >&2; exit 1; }
  AB_COOLDOWN_LOG=$log
}

# ab_round_cooldown <round>: load1 < 閾値まで待つ。結果は pass／timeout／unavailable で記録する。
ab_round_cooldown() {
  [[ -n "${AB_COOLDOWN_LOG:-}" ]] || return 0
  local round=$1 thr=$AB_ROUND_COOLDOWN_MAX_LOAD1 waited=0 status=timeout l1 ok
  while :; do
    l1=$(ab_cooldown_read_load1)
    if [[ -z "$l1" || ! "$l1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      status=unavailable
      l1=NA
      break
    fi
    ok=$(awk -v a="$l1" -v t="$thr" 'BEGIN{print (a<t)?1:0}')
    if [[ "$ok" == "1" ]]; then status=pass; break; fi
    if ((waited >= AB_COOLDOWN_MAX_WAIT_S)); then break; fi
    ab_cooldown_sleep "$AB_COOLDOWN_INTERVAL_S"
    waited=$((waited + AB_COOLDOWN_INTERVAL_S))
  done
  echo "round${round} cooldown=${status} threshold=${thr} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$AB_COOLDOWN_LOG"
}
