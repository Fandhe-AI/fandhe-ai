#!/usr/bin/env bash
# イシュー #1489: 正式系列 fandhe-ai =0.8.0 の CUDA GEMM candle 比ゲート再計測（GB10）の
# オーケストレーション。専有ゲート（事前宣言）→ run_gemm_gate_cuda.sh → compare_gemm_gate.py。
#
# 事前宣言ゲート（計測開始前に固定。結果を見てから緩和しない）:
#   - 1 分間 load average < 1.0
#   - nvidia-smi utilization.gpu == 0 %
#   - 上記 2 条件を 30 秒間隔で連続 3 サンプル満たしたら通過
#   - 最大 20 サンプル（約 10 分）で通過しなければ verdict=undetermined を 1 回記録して終了
#     （再試行ループで待ち続けない。イシュー #1489 ユーザー承認節・#1242 ツリーの教訓）
# 前提: bench-fandhe / bench-candle(cuda) は本スクリプト起動前に prebuild 済み（ビルド残余負荷を
#       計測区間から分離する。§14.6 の再発防止）。run_gemm_gate.sh 側の cargo build は no-op
#       で通過し、manifest 記録（record_manifest）だけが行われる（GEMM_GATE_SKIP_BUILD は使わない）。
# 実行ディレクトリ: `run_gemm_gate_cuda.sh`・`compare_gemm_gate.py` がある
#   `scripts/bench/framework-compare/`（`FRAMEWORK_COMPARE_DIR` で上書き可。未指定時は本ファイルの
#   保存先 `docs/perf/logs/<...>/` からリポジトリ相対で解決する）。ログ（gate／run／compare 出力）も
#   同ディレクトリへ書き出し、計測後に `docs/perf/logs/cuda-gemm-candle-gate-0.8.0-1489/` へ回収する。
#   実測時（2026-09-10）は本ファイルを GB10 上の `scripts/bench/framework-compare/orchestrate-1489.sh`
#   へコピーして `(setsid nohup bash ./orchestrate-1489.sh 0.8.0-1489 > orchestrate-1489.out 2>&1 < /dev/null &)`
#   で切り離し実行した（その配置でも上記の相対解決で同じディレクトリに到達する）。
set -u
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
LABEL=${1:-0.8.0-1489}
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -n "${FRAMEWORK_COMPARE_DIR:-}" ]]; then
  WORK_DIR="$FRAMEWORK_COMPARE_DIR"
elif [[ -f "$SELF_DIR/run_gemm_gate_cuda.sh" ]]; then
  WORK_DIR="$SELF_DIR"
else
  WORK_DIR="$SELF_DIR/../../../../scripts/bench/framework-compare"
fi
if [[ ! -f "$WORK_DIR/run_gemm_gate_cuda.sh" || ! -f "$WORK_DIR/compare_gemm_gate.py" ]]; then
  echo "ERROR: run_gemm_gate_cuda.sh / compare_gemm_gate.py が見つからない: $WORK_DIR" >&2
  echo "  FRAMEWORK_COMPARE_DIR=<scripts/bench/framework-compare の絶対パス> を指定すること" >&2
  exit 1
fi
cd "$WORK_DIR"
GATE_LOG="gate-${LABEL}.log"
RUN_LOG="run_gemm_gate_cuda-dgx-${LABEL}.log"
CMP_OUT="compare_gemm_gate-${LABEL}.md"
MAX_SAMPLES=20
NEED_CONSEC=3
INTERVAL=30
LOAD_MAX="1.0"
{
  echo "gate protocol: load1 < ${LOAD_MAX} && util.gpu == 0% for ${NEED_CONSEC} consecutive samples (interval ${INTERVAL}s, max ${MAX_SAMPLES})"
  echo "start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$GATE_LOG"
consec=0
passed=0
for i in $(seq 1 "$MAX_SAMPLES"); do
  load1=$(cut -d' ' -f1 /proc/loadavg)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
  ok=0
  if awk -v l="$load1" -v m="$LOAD_MAX" 'BEGIN{exit !(l<m)}' && [[ "$util" == "0" ]]; then ok=1; fi
  echo "sample=$i ts=$(date -u +%H:%M:%SZ) load1=$load1 util_gpu=${util}% compute_apps=$apps gate_ok=$ok" >> "$GATE_LOG"
  if [[ "$ok" == "1" ]]; then consec=$((consec+1)); else consec=0; fi
  if [[ "$consec" -ge "$NEED_CONSEC" ]]; then passed=1; break; fi
  sleep "$INTERVAL"
done
if [[ "$passed" != "1" ]]; then
  echo "verdict=undetermined (gate not satisfied within ${MAX_SAMPLES} samples)" >> "$GATE_LOG"
  echo "done. verdict=undetermined" >> "$GATE_LOG"
  exit 0
fi
echo "gate passed at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$GATE_LOG"
uptime >> "$GATE_LOG"
bash ./run_gemm_gate_cuda.sh "$LABEL" > "$RUN_LOG" 2>&1
echo "run_gemm_gate exit=$?" >> "$GATE_LOG"
uptime >> "$GATE_LOG"
python3 compare_gemm_gate.py --device cuda "results/raw/results-dgx-gemm-gate-${LABEL}.jsonl" --out "$CMP_OUT" > "compare_gemm_gate-${LABEL}.stdout.log" 2>&1
echo "compare_gemm_gate exit=$?" >> "$GATE_LOG"
echo "end: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$GATE_LOG"
echo "done." >> "$GATE_LOG"
