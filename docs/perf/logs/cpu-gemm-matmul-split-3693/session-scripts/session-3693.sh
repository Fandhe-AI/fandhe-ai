#!/usr/bin/env bash
# イシュー #3693 計測セッション（RULE.txt が正）。1 機体・1 系列を 1 本の切り離しセッションで直列に実行する。
#   (1) 実機ロック（外部の mkdir 方式。他イシューの計測と排他）  (2) GB10 のみ: 30 秒間隔の compute-apps poller と pre／post スナップショット
#   (3) orchestrate.sh（診断のみ・ABBA・5 run）  (4) GB10 のみ: excl_check.py で専有の合否を excl_status.txt へ
#   (5) split_judge.py の単一系列確認（参考。最終判定は系列を揃えて judge-3693.md として出す）
# ビルドは事前に build_ab_arms.sh で完了済み（計測中のビルド並走は禁止）。各ステップの rc を steps.log へ追記し最後に終端行を書く。
# 使い方: session-3693.sh <gb10|macmini> <label> <before|after>
#   label は ^[A-Za-z0-9._-]+$。第 3 引数は系列の先頭腕（RULE.txt: s1 before・s2 after・s3 before・…）。
#   出力は $HOME/work/r3693/out-<label>。既存なら中止。
# 配置: $HOME/work/r3693/{runner,bin}（runner = 部品を入れたコミットの git archive 展開木・bin = build_ab_arms.sh の出力）
# 終了コード: orchestrate／excl_check／split_judge のいずれかが非 0 ならセッションも非 0
#   （終端行 done. rc= と exit に最初の失敗 rc を反映する）。
# 値は RULE の固定定数だけを使い、外部入力から組み立てない。ホスト名・ユーザー名のマスクは orchestrate.sh と回収側で行う。
export PATH=$HOME/.cargo/bin:$PATH
export LC_ALL=C
unset RAYON_NUM_THREADS CARGO_TARGET_DIR GEMM_REUSE_MINFLT GLIBC_TUNABLES

host_kind="${1:-}"
label="${2:-}"
first_arm="${3:-}"
case "${host_kind}" in gb10 | macmini) ;; *) echo "usage: $0 <gb10|macmini> <label> <before|after>" >&2; exit 2 ;; esac
case "${label}" in '' | *[!A-Za-z0-9._-]*) echo "label が不正" >&2; exit 2 ;; esac
case "${first_arm}" in before | after) ;; *) echo "第 3 引数は before|after" >&2; exit 2 ;; esac

R=$HOME/work/r3693
T=$R/runner
BIN=$R/bin
O=$R/out-${label}
LOCK_DIR=$HOME/work/.fandhe-hw-lock
LOG=$O/steps.log
GPU=$O/gpu_procs.txt
POLL=$O/gpu_poll.log
POLL_INTERVAL=30
RES=$T/docs/perf/logs/cpu-gemm-matmul-split-3693
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
echo "#3693 ${host_kind} $(date -u +%FT%TZ)" >"${LOCK_DIR}/owner"
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
fi

cd "${T}" || exit 1
step "start runner=$(sed -n 1p "${T}/.rev-stamp") before=$(sed -n 1p "${R}/before/.rev-stamp") after=$(sed -n 1p "${R}/after/.rev-stamp") first=${first_arm}"

# poller は pre の取得後に起動し、post の前に停止して wait する（観測時刻を pre／post の境界内に収める。
# excl_check.py の gap_failures は範囲外の poll を負の間隔として拒否するため）。
if [ "${host_kind}" = "gb10" ]; then
  snap "main:pre"
  poller &
  POLLER_PID=$!
fi
bash "${RES}/orchestrate.sh" "${host_kind}" --bindir "${BIN}" --out "${O}/${host_kind}" --first "${first_arm}" >"${O}/orch-${host_kind}.log" 2>&1
orc_rc=$?
step "orchestrate rc=${orc_rc}"
final_rc=${orc_rc}
if [ "${host_kind}" = "gb10" ]; then
  kill "${POLLER_PID}" 2>/dev/null
  wait "${POLLER_PID}" 2>/dev/null
  POLLER_PID=""
  snap "main:post"
  python3 -I "${EXCL}" "${O}" --series main >"${O}/excl_check.log" 2>&1
  excl_rc=$?
  step "excl_check rc=${excl_rc}"
  [ "${final_rc}" != 0 ] || final_rc=${excl_rc}
  [ ! -f "${O}/excl_status.txt" ] || cp "${O}/excl_status.txt" "${O}/${host_kind}/excl_status.txt"
fi
if [ "${orc_rc}" = 0 ]; then
  # 単一系列では「系列不足」の判定不能になるのが正常（参考出力。最終判定は系列を揃えて別途出す）。
  python3 -I "${RES}/split_judge.py" --host "${host_kind}=${O}/${host_kind}" --out "${O}/judge-${host_kind}.md" >"${O}/judge-${host_kind}.log" 2>&1
  judge_rc=$?
  step "split_judge rc=${judge_rc}"
  [ "${final_rc}" != 0 ] || final_rc=${judge_rc}
fi
step "session body done"
echo "done. rc=${final_rc}" >>"${LOG}"
exit "${final_rc}"
