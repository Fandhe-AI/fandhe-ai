#!/usr/bin/env bash
# 役割: イシュー #2997 の Mac mini（Apple M4・<metal-node>）CPU 参考行で「1 腕 × 1 run = 1 プロセス」を実行する。
# 呼び出し文脈: 開発機から `ssh <metal-node> 'zsh -lc "./docs/perf/logs/int8-quant-baseline-2609/orchestrate_mac.sh <arm> <run>"'`
# の形でノード側同期ツリー上で呼ぶ（docs/real-hardware-verification-env.md §7）。規則の正は RULE.txt。
# 変数は ${VAR} 形式（macOS の bash 3.2 は $VAR 直後の全角文字を変数名に取り込むため）。完了マーカーは行頭 "done." 。
set -euo pipefail

ARM="${1:-}"
RUN="${2:-}"
case "${ARM}" in rust-cpu|pytorch-cpu) ;; *) echo "bad arm: ${ARM}" >&2; exit 2 ;; esac
case "${RUN}" in 1|2|3|4|5) ;; *) echo "bad run: ${RUN}" >&2; exit 2 ;; esac

ROOT="${ROOT:-${HOME}/work/rust-ai-library-run}"
OUT="${OUT:-${HOME}/work/int8-2997-logs}"
PYVENV="${PYVENV:-${HOME}/work/venv-torch-cpu}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${HOME}/work/target-fandhe-ai}"
TIMEOUT_SECS="${TIMEOUT_SECS:-1800}"
mkdir -p "${OUT}"
LOG="${OUT}/${ARM}-run${RUN}.log"
JSON="${OUT}/${ARM}-run${RUN}.json"
if [ -e "${LOG}" ]; then echo "既存ログがあるため停止: ${LOG}" >&2; exit 3; fi

# 開始前ゲート: load1 < 2.0 を最大 30 分待つ。満たさなければ「参考扱い」と記録して実行する（RULE.txt DISCARD_MAC）。
waited=0
while :; do
  load1="$(sysctl -n vm.loadavg | awk '{print $2}')"
  if awk -v l="${load1}" 'BEGIN{exit !(l < 2.0)}'; then note="load1_ok"; break; fi
  if [ "${waited}" -ge 1800 ]; then note="REFERENCE_ONLY_load1_high"; break; fi
  sleep 20; waited=$((waited + 20))
done
echo "pre load1=${load1} waited=${waited}s note=${note}" > "${OUT}/load-${ARM}-run${RUN}.txt"

# macOS には coreutils の timeout が無いため perl の alarm で外部タイムアウトを掛ける。
with_timeout() { local secs="$1"; shift; perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "${secs}" "$@"; }

cd "${ROOT}"
set +e
case "${ARM}" in
  rust-cpu)
    BENCH_TRANSFORMER_REPORT="${JSON}" with_timeout "${TIMEOUT_SECS}" \
      cargo test -p bench-harness --release --test transformer_workload \
      -- transformer_block_forward_bench_cpu --exact --ignored --nocapture > "${LOG}" 2>&1 ;;
  pytorch-cpu)
    with_timeout "${TIMEOUT_SECS}" "${PYVENV}/bin/python" scripts/bench-transformer-pytorch.py \
      --device cpu --output "${JSON}" > "${LOG}" 2>&1 ;;
esac
status=$?
set -e
echo "post load1=$(sysctl -n vm.loadavg | awk '{print $2}')" >> "${OUT}/load-${ARM}-run${RUN}.txt"
echo "done. arm=${ARM} run=${RUN} status=${status}"
exit "${status}"
