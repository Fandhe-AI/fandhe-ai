#!/usr/bin/env bash
# イシュー #3179: Mac mini で PyTorch の純カーネル時間と checksum を測る実行ループ。
# 規則の正は同ディレクトリの RULE.txt。cpu-jobgrid-3177 / run_ab_thread_policy_macmini_cpu.sh と同方針
# （負荷ゲート・run ごとの起動順ローテーション・ログ非上書き）。bash 3.2（macOS 標準）で動く。
#
# 使い方（Mac mini 上。作業ディレクトリ D に venv-torch と bench_py.py がある前提）:
#   VENV_PY=<venv-torch/bin/python> BENCH_PY=<bench_py.py> DIAG=<opp_kernel_torch_diag.py> \
#   REF_JSONL=<results-macmini-py-0.11.0-median5.jsonl> OUT=<出力ディレクトリ(未作成)> \
#   [BENCH_FANDHE=<ビルド済み bench-fandhe>] bash run-macmini.sh
set -euo pipefail

: "${VENV_PY:?}" "${BENCH_PY:?}" "${DIAG:?}" "${REF_JSONL:?}" "${OUT:?}"
GATE_LOAD=3.0
if [ -e "${OUT}" ]; then
  echo "OUT が既にある（上書きしない）: ${OUT}" >&2
  exit 2
fi
mkdir -p "${OUT}"
GATE_LOG="${OUT}/gate.log"

# 版の固定（RULE 項 2）。違えば停止。
"${VENV_PY}" -c 'import sys,torch,numpy; v=(sys.version.split()[0],torch.__version__.split("+")[0],numpy.__version__); print(*v); sys.exit(0 if v==("3.12.15","2.14.0","2.5.4") else 4)' > "${OUT}/versions.txt" \
  || { echo "venv の版が RULE 項 2 と不一致" >&2; exit 4; }
{
  sw_vers; sysctl -n machdep.cpu.brand_string; sysctl hw.perflevel0.logicalcpu hw.perflevel1.logicalcpu
  "${VENV_PY}" -c 'import torch,numpy; print(torch.__config__.show()); print(torch.get_num_threads()); numpy.show_config()'
  shasum -a 256 "${BENCH_PY}"
} > "${OUT}/env.txt" 2>&1

wait_gate() { # $1=ラベル。load1 < GATE_LOAD になるまで 30 秒間隔で最大 30 分。
  local i=0 load
  while [ "${i}" -le 60 ]; do
    load="$(uptime | sed -E 's/.*load averages?: *([0-9.]+).*/\1/')"
    case "${load}" in
      ''|*[!0-9.]*) echo "GATE-INVALID $1 load=${load}" >> "${GATE_LOG}"; exit 5 ;;
    esac
    if awk -v l="${load}" -v t="${GATE_LOAD}" 'BEGIN{exit !(l<t)}'; then
      echo "GATE-PASS $1 load1=${load}" >> "${GATE_LOG}"
      return 0
    fi
    i=$((i + 1))
    sleep 30
  done
  echo "GATE-TIMEOUT $1" >> "${GATE_LOG}"
  exit 6
}

# 腕: "arm:device:env"。env は VECLIB_MAXIMUM_THREADS の値（空は未設定）。
ARMS="base:cpu: base:mps: threads1:cpu: threads4:cpu: threads10:cpu: base:cpu:1 base:cpu:4 numpy:cpu:"
N_ARMS=$(echo ${ARMS} | wc -w | tr -d ' ')
run=1
while [ "${run}" -le 5 ]; do
  mkdir -p "${OUT}/run${run}"
  rot=$(( (run - 1) % N_ARMS ))
  idx=0
  ordered=""
  for pass in 1 2; do
    j=0
    for a in ${ARMS}; do
      if { [ "${pass}" = 1 ] && [ "${j}" -ge "${rot}" ]; } || { [ "${pass}" = 2 ] && [ "${j}" -lt "${rot}" ]; }; then
        ordered="${ordered} ${a}"
      fi
      j=$((j + 1))
    done
  done
  for a in ${ordered}; do
    arm="${a%%:*}"; rest="${a#*:}"; dev="${rest%%:*}"; vl="${rest#*:}"
    idx=$((idx + 1))
    tag="${arm}-${dev}-vl${vl:-unset}"
    wait_gate "run${run}/${tag}"
    uptime > "${OUT}/run${run}/uptime_before_${tag}.txt"
    if [ -n "${vl}" ]; then export VECLIB_MAXIMUM_THREADS="${vl}"; else unset VECLIB_MAXIMUM_THREADS || true; fi
    shapes=all
    rc=0
    "${VENV_PY}" "${DIAG}" measure --bench-py "${BENCH_PY}" --device "${dev}" --shapes "${shapes}" \
      --arm "${arm}" --run "${run}" --out "${OUT}/run${run}/${tag}.jsonl" || rc=$?
    if [ "${rc}" = 3 ]; then
      echo "unavailable ${tag}" >> "${OUT}/run${run}/unavailable.txt"
    elif [ "${rc}" != 0 ]; then
      exit "${rc}"
    fi
    uptime > "${OUT}/run${run}/uptime_after_${tag}.txt"
  done
  run=$((run + 1))
done

# Python の計測を終えてから Rust 側の総和（ビルド済みバイナリのみ。cargo build はここでは走らせない）。
if [ -n "${BENCH_FANDHE:-}" ]; then
  mkdir -p "${OUT}/rust"
  for n in 256 512 1024 2048 4096; do
    for r in 1 2 3 4 5; do
      wait_gate "rust/n${n}/r${r}"
      # --out を明示しないと bench-common の Cli が既定の results/raw/results.jsonl へ追記し、
      # 作業ディレクトリの既存結果へ診断行が混入する（RULE 項 11）。専用パスへ出し stdout は別ログへ。
      "${BENCH_FANDHE}" --task gemm --device cpu --mode reuse --phases --size "${n}" \
        --out "${OUT}/rust/n${n}_r${r}.jsonl" > "${OUT}/rust/n${n}_r${r}.stdout.log"
    done
  done
fi

"${VENV_PY}" "${DIAG}" aggregate --runs-dir "${OUT}" --reference-py-jsonl "${REF_JSONL}" \
  --out-json "${OUT}/aggregate.json" --out-md "${OUT}/aggregate.md"
