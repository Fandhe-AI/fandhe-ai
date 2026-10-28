#!/usr/bin/env bash
# イシュー #3672 計測セッション（RULE.txt ＋ RULE.v2.txt が正）。1 機体分を 1 本の切り離しセッションで直列に実行する。
#   (1) 実機ロック（外部の mkdir 方式。他イシューの計測と排他）  (2) GB10 のみ: 30 秒間隔の compute-apps poller と pre／post スナップショット
#   (3) orchestrate.sh（before／after の ABBA・5 run）  (4) GB10 のみ: excl_check.py で専有の合否を excl_status.txt へ
#   (5) ab_judge.py の機体別判定（参考。最終判定は両機体の回収後に judge-3672.md として出す）
# ビルドは事前に build_ab_arms.sh で完了済み（計測中のビルド並走は禁止）。各ステップの rc を steps.log へ追記し最後に終端行を書く。
# 使い方: session-3672.sh <gb10|macmini> <label>   （label は ^[A-Za-z0-9._-]+$。出力は $HOME/work/r3672/out-<label>。既存なら中止）
# 配置: $HOME/work/r3672/{runner,bin}（runner = 部品を入れたコミットの git archive 展開木・bin = build_ab_arms.sh の出力）
# 値は RULE の固定定数だけを使い、外部入力から組み立てない。ホスト名・ユーザー名のマスクは orchestrate.sh と回収側で行う。
export PATH=$HOME/.cargo/bin:$PATH
export LC_ALL=C
unset RAYON_NUM_THREADS CARGO_TARGET_DIR GEMM_REUSE_MINFLT GLIBC_TUNABLES

host_kind="${1:-}"
label="${2:-}"
case "${host_kind}" in gb10 | macmini) ;; *) echo "usage: $0 <gb10|macmini> <label>" >&2; exit 2 ;; esac
case "${label}" in '' | *[!A-Za-z0-9._-]*) echo "label が不正" >&2; exit 2 ;; esac

R=$HOME/work/r3672
T=$R/runner
BIN=$R/bin
O=$R/out-${label}
LOCK_DIR=$HOME/work/.fandhe-hw-lock
LOG=$O/steps.log
GPU=$O/gpu_procs.txt
POLL=$O/gpu_poll.log
POLL_INTERVAL=30
RES=$T/docs/perf/logs/cpu-gemm-download-owned-3672
EXCL=$T/docs/perf/logs/cpu-gemm-resident-reuse-3646/excl_check.py

if [ -e "${O}" ]; then echo "output dir exists: ${O}" >&2; exit 5; fi
mkdir -p "${O}" || exit 1
step() { echo "$(date -u +%FT%TZ) $*" >>"${LOG}"; }

# ---- 実機ロック（取れるまで 60 秒おき。最大 3 時間）----
got=0
i=0
while [ "${i}" -lt 180 ]; do
  if mkdir "${LOCK_DIR}" 2>/dev/null; then got=1; break; fi
  sleep 60
  i=$((i + 1))
done
if [ "${got}" != 1 ]; then step "hwlock timeout"; echo "done. rc=lock" >>"${LOG}"; exit 3; fi
echo "#3672 ${host_kind} $(date -u +%FT%TZ)" >"${LOCK_DIR}/owner"
POLLER_PID=""
trap '[ -z "${POLLER_PID}" ] || kill "${POLLER_PID}" 2>/dev/null; rm -rf "${LOCK_DIR}"; step "hwlock released"' EXIT
step "hwlock acquired host=${host_kind}"

# ---- GB10: compute-apps poller（30 秒固定間隔。停止も変更もしない観測のみ。pid が出たら ps を添える）----
if [ "${host_kind}" = "gb10" ]; then
  poller() {
    local rc out n pids
    while :; do
      out="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1)"
      rc=$?
      if [ "${rc}" = 0 ]; then
        if [ -z "${out}" ]; then n=0; else n="$(printf '%s\n' "${out}" | wc -l | tr -d ' ')"; fi
      else
        n=NA
      fi
      echo "$(date -u +%FT%TZ) series=main rc=${rc} apps=${n} $(printf '%s' "${out}" | tr '\n' ';')" >>"${POLL}"
      if [ "${rc}" = 0 ] && [ "${n}" != 0 ]; then
        pids="$(printf '%s\n' "${out}" | cut -d, -f1 | tr -d ' ' | paste -sd, -)"
        ps -o pid,ppid,comm -p "${pids}" >>"${POLL}" 2>&1
      fi
      sleep "${POLL_INTERVAL}"
    done
  }
  snap() { # $1=ラベル（excl_check.py の `<series>:pre|post` 形式）。
    {
      echo "== $1 $(date -u +%FT%TZ)"
      local o r pids
      o="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1)"
      r=$?
      echo "rc=${r}"
      printf '%s\n' "${o}"
      nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
      pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)"
      [ -z "${pids}" ] || ps -o pid,%cpu,etime,comm -p "${pids}"
      uptime
    } >>"${GPU}" 2>&1
  }
  poller &
  POLLER_PID=$!
fi

cd "${T}" || exit 1
step "start runner=$(sed -n 1p "${T}/.rev-stamp") before=$(sed -n 1p "${R}/before/.rev-stamp") after=$(sed -n 1p "${R}/after/.rev-stamp")"

[ "${host_kind}" != "gb10" ] || snap "main:pre"
bash "${RES}/orchestrate.sh" "${host_kind}" --bindir "${BIN}" --out "${O}/${host_kind}" >"${O}/orch-${host_kind}.log" 2>&1
orc_rc=$?
step "orchestrate rc=${orc_rc}"
if [ "${host_kind}" = "gb10" ]; then
  snap "main:post"
  kill "${POLLER_PID}" 2>/dev/null
  POLLER_PID=""
  python3 -I "${EXCL}" "${O}" --series main >"${O}/excl_check.log" 2>&1
  step "excl_check rc=$?"
  [ ! -f "${O}/excl_status.txt" ] || cp "${O}/excl_status.txt" "${O}/${host_kind}/excl_status.txt"
fi
if [ "${orc_rc}" = 0 ]; then
  python3 -I "${RES}/ab_judge.py" --host "${host_kind}=${O}/${host_kind}" --out "${O}/judge-${host_kind}.md" >"${O}/judge-${host_kind}.log" 2>&1
  step "ab_judge rc=$?"
fi
step "session body done"
echo "done. rc=${orc_rc}" >>"${LOG}"
