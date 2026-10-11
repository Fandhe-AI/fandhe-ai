#!/usr/bin/env bash
# 役割: イシュー #2997 の GB10（<cuda-node>）実測で「1 腕 × 1 run = 1 プロセス」を実行する。
# 呼び出し文脈: 開発機から `ssh <cuda-node> './docs/perf/logs/int8-quant-baseline-2609/orchestrate_cuda.sh <arm> <run>'`
# の形でノード側同期ツリー上で呼ぶ（docs/real-hardware-verification-env.md §3〜§6 の方式）。
# 規則の正は RULE.txt（腕・順序・破棄条件）。腕名・run 番号は allowlist と照合し eval は使わない。
# 変数は ${VAR} 形式（bash 3.2 の全角文字連結事故を避ける）。完了マーカーは行頭 "done." （rc= は使わない）。
set -euo pipefail

ARM="${1:-}"
RUN="${2:-}"
# RULE.txt の腕名 pytorch-cuda を受理し、ログ・JSON・集計の腕名は記録済みの pytorch に正規化する
# （aggregate.py --arms ...,pytorch と一致させる。既存の pytorch-run*.{log,json} を改名しない）。
if [ "${ARM}" = "pytorch-cuda" ]; then ARM="pytorch"; fi
case "${ARM}" in prefusion|fused|parity|pytorch) ;; *) echo "bad arm: ${ARM}" >&2; exit 2 ;; esac
case "${RUN}" in 1|2|3|4|5) ;; *) echo "bad run: ${RUN}" >&2; exit 2 ;; esac

ROOT="${ROOT:-${HOME}/work/rust-ai-library-run}"
OUT="${OUT:-${HOME}/work/int8-2997-logs}"
PYVENV="${PYVENV:-${HOME}/poc5-pytorch}"
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${HOME}/work/target-fandhe-ai}"
TIMEOUT_SECS="${TIMEOUT_SECS:-900}"
mkdir -p "${OUT}"
LOG="${OUT}/${ARM}-run${RUN}.log"
JSON="${OUT}/${ARM}-run${RUN}.json"
OCC="${OUT}/gpu-occupancy-${ARM}-run${RUN}.txt"
if [ -e "${LOG}" ]; then echo "既存ログがあるため停止: ${LOG}" >&2; exit 3; fi

# 開始前ゲート: utilization.gpu <= 5% になるまで最大 300 秒待つ（RULE.txt DISCARD_CUDA）。
waited=0
while :; do
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [ "${util}" -le 5 ]; then break; fi
  if [ "${waited}" -ge 300 ]; then echo "GPU が混雑(${util}%)のため実行しない" >&2; exit 4; fi
  sleep 10; waited=$((waited + 10))
done
# nvidia-smi の採取失敗は握りつぶさず ${OCC}.smifail へ記録する。監視不能な run は
# 「別プロセス不在」と区別できないため RUN_INVALID として集計から除外する（RULE.txt DISCARD_CUDA）。
SMIFAIL="${OCC}.smifail"
SMIOK="${OCC}.smiok"
rm -f "${SMIFAIL}" "${SMIOK}"
{
  echo "pre util=${util}% waited=${waited}s"
  echo "pre compute-apps:"
  if ! nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; then
    echo "pre compute-apps query failed" >> "${SMIFAIL}"
  fi
} > "${OCC}"

# run 中の compute-apps を 1 秒間隔で採取する（別プロセス判定用）。
( while :; do
    if nvidia-smi --query-compute-apps=process_name --format=csv,noheader >> "${OCC}.samples" 2>/dev/null; then
      echo 1 >> "${SMIOK}"
    else
      echo "sample query failed" >> "${SMIFAIL}"
    fi
    sleep 1
  done ) &
SAMPLER=$!
trap 'kill ${SAMPLER} 2>/dev/null || true' EXIT

cd "${ROOT}"
set +e
case "${ARM}" in
  prefusion)
    BENCH_TRANSFORMER_REPORT_CUDA_PREFUSION="${JSON}" timeout "${TIMEOUT_SECS}" \
      cargo test -p bench-harness --release --test transformer_workload \
      -- transformer_block_forward_bench_cuda_prefusion --exact --ignored --nocapture > "${LOG}" 2>&1 ;;
  fused)
    BENCH_TRANSFORMER_REPORT_CUDA_FUSED="${JSON}" timeout "${TIMEOUT_SECS}" \
      cargo test -p bench-harness --release --test transformer_workload \
      -- transformer_block_forward_bench_cuda_fused --exact --ignored --nocapture > "${LOG}" 2>&1 ;;
  parity)
    timeout "${TIMEOUT_SECS}" \
      cargo test -p bench-harness --release --test transformer_workload \
      -- transformer_block_forward_cuda_fused_parity --exact --ignored --nocapture > "${LOG}" 2>&1 ;;
  pytorch)
    timeout "${TIMEOUT_SECS}" "${PYVENV}/bin/python" scripts/bench-transformer-pytorch.py \
      --device cuda --output "${JSON}" > "${LOG}" 2>&1 ;;
esac
status=$?
set -e

kill "${SAMPLER}" 2>/dev/null || true
wait "${SAMPLER}" 2>/dev/null || true
{
  echo "post compute-apps:"
  if ! nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; then
    echo "post compute-apps query failed" >> "${SMIFAIL}"
  fi
  if post_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1)" && [ -n "${post_util}" ]; then
    echo "post util=${post_util}"
  else
    echo "post util query failed" >> "${SMIFAIL}"
    echo "post util=UNKNOWN"
  fi
  echo "foreign process names during run (transformer_workload / python 以外):"
  if [ -e "${OCC}.samples" ]; then
    grep -v -e '^$' -e 'transformer_workload' -e 'python' "${OCC}.samples" | sort -u || true
  fi
} >> "${OCC}"
# 異種プロセス判定: 採取結果から transformer_workload / python 以外が現れた run は無効（RULE.txt DISCARD_CUDA）。
foreign=""
if [ -e "${OCC}.samples" ]; then
  foreign="$(grep -v -e '^$' -e 'transformer_workload' -e 'python' "${OCC}.samples" | sort -u || true)"
fi
rm -f "${OCC}.samples"
# 監視不能判定: 採取失敗が 1 件でもある、または run 中に成功した採取が 0 回の run は無効。
smi_unmonitored=""
if [ -s "${SMIFAIL}" ]; then
  smi_unmonitored="nvidia-smi 採取失敗 $(wc -l < "${SMIFAIL}" | tr -d ' ') 件: $(sort -u "${SMIFAIL}" | tr '\n' ';')"
elif [ ! -s "${SMIOK}" ]; then
  smi_unmonitored="run 中に成功した nvidia-smi 採取が 0 回"
fi
rm -f "${SMIFAIL}" "${SMIOK}"
if [ -n "${smi_unmonitored}" ]; then
  echo "SMI_UNMONITORED: ${smi_unmonitored}" >> "${OCC}"
fi
if [ -n "${foreign}" ] || [ -n "${smi_unmonitored}" ]; then
  # 無効 run として記録し、破棄ログを *.discarded-<epoch> へ退避する（取り直し時に既存ログ停止へ掛からないようにする）。
  # aggregate.py は gpu-occupancy-*.txt の RUN_INVALID 行も拒否するため、退避しない手運用でも集計に混入しない。
  if [ -n "${foreign}" ]; then
    echo "RUN_INVALID: foreign process detected (discarded; rerun required)" >> "${OCC}"
  fi
  if [ -n "${smi_unmonitored}" ]; then
    echo "RUN_INVALID: nvidia-smi unmonitorable (discarded; rerun required)" >> "${OCC}"
  fi
  stamp="$(date +%s)"
  for f in "${LOG}" "${JSON}" "${OCC}"; do
    if [ -e "${f}" ]; then mv "${f}" "${f}.discarded-${stamp}"; fi
  done
  echo "done. arm=${ARM} run=${RUN} status=${status} RUN_INVALID (discarded)"
  exit 5
fi
# 計測コマンド自体が失敗（タイムアウト 124 等）した run も、通常名の JSON が残って集計されないよう退避する。
echo "${status}" > "${OUT}/${ARM}-run${RUN}.status"
if [ "${status}" -ne 0 ]; then
  stamp="$(date +%s)"
  for f in "${LOG}" "${JSON}" "${OCC}" "${OUT}/${ARM}-run${RUN}.status"; do
    if [ -e "${f}" ]; then mv "${f}" "${f}.discarded-${stamp}"; fi
  done
  echo "done. arm=${ARM} run=${RUN} status=${status} FAILED (discarded)"
  exit "${status}"
fi
echo "done. arm=${ARM} run=${RUN} status=${status}"
exit "${status}"
