#!/usr/bin/env bash
# イシュー #1560: CUDA resident weight 勾配経路（#1559）の GB10 実機
# オーケストレーション。専有ゲート（事前宣言。#1489 と同型）→
# `run_ab_resident_grad_cuda.sh` の順に実行する。
#
# 事前宣言ゲート（計測開始前に固定。結果を見てから緩和しない）:
#   - 1 分間 load average < 1.0
#   - nvidia-smi utilization.gpu == 0 %
#   - 上記 2 条件を 30 秒間隔で連続 3 サンプル満たしたら通過
#   - 最大 20 サンプル（約 10 分）で通過しなければ verdict=undetermined を
#     1 回記録して終了（再試行ループで待ち続けない。イシュー #1489 承認節・
#     #1242 ツリーの教訓）
#   - `AB_LOAD_GATE_MODE=record_only` を指定すると専有ゲートを opt-out
#     できる（ユーザーの明示指示がある場合のみ使う。#1519 の運用）
#
# 前提: before/after 両ツリーの `bench-fandhe` は本スクリプト起動前に
#   prebuild 済みでなくてよい（`run_ab_resident_grad_cuda.sh` が両腕を
#   ビルドする）。ただし計測負荷をビルド区間から分離したい場合は事前に
#   1 回 `run_ab_resident_grad_cuda.sh` のビルド部分のみを流してキャッシュを
#   温めておくとよい（必須ではない）。
#
# 実行ディレクトリ: `run_ab_resident_grad_cuda.sh` がある
#   `scripts/bench/framework-compare/`（`FRAMEWORK_COMPARE_DIR` で上書き可。
#   未指定時は本ファイルの保存先からリポジトリ相対で解決する）。
#
# 使い方（GB10 実機。ユーザー承認・別セッション）:
#   AB_BEFORE_FACADE_PATH=/home/<user>/work/rust-ai-library-run-1560-before/crates/facade \
#   AB_AFTER_FACADE_PATH=/home/<user>/work/rust-ai-library-run-1560-after/crates/facade \
#     ./orchestrate.sh 1560
#
# `--dry-run` で経路解決のみ検証できる（実機不要。Linux で自己検証可能）。
set -u
LABEL=${1:-1560}
# A03 インジェクション対策: ラベルはファイル名に直接埋め込む
# （`gate-${LABEL}.log`／`run-${LABEL}.log`）ため、`run_ab_resident_grad_
# cuda.sh` と同じ allowlist で検証する（パストラバーサル・コマンド注入
# の防止。`--dry-run` はラベル検証より前に処理する）。
if [[ "$LABEL" != "--dry-run" && ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label> [--dry-run]  (label must match [A-Za-z0-9._-]+)" >&2
  exit 1
fi
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -n "${FRAMEWORK_COMPARE_DIR:-}" ]]; then
  WORK_DIR="$FRAMEWORK_COMPARE_DIR"
elif [[ -f "$SELF_DIR/../../../../scripts/bench/framework-compare/run_ab_resident_grad_cuda.sh" ]]; then
  WORK_DIR="$(cd "$SELF_DIR/../../../../scripts/bench/framework-compare" && pwd)"
else
  WORK_DIR=""
fi
if [[ -z "$WORK_DIR" || ! -f "$WORK_DIR/run_ab_resident_grad_cuda.sh" || ! -f "$WORK_DIR/compare_gemm_ab.py" ]]; then
  echo "ERROR: run_ab_resident_grad_cuda.sh / compare_gemm_ab.py が見つからない: ${WORK_DIR:-<unresolved>}" >&2
  echo "  FRAMEWORK_COMPARE_DIR=<scripts/bench/framework-compare の絶対パス> を指定すること" >&2
  exit 1
fi

if [[ "${2:-}" == "--dry-run" || "${1:-}" == "--dry-run" ]]; then
  echo "dry-run: WORK_DIR resolved to $WORK_DIR"
  echo "dry-run: label=$LABEL"
  ls -la "$WORK_DIR/run_ab_resident_grad_cuda.sh" "$WORK_DIR/compare_gemm_ab.py"
  exit 0
fi

cd "$WORK_DIR"
GATE_LOG="${SELF_DIR}/gate-${LABEL}.log"
MAX_SAMPLES=20
NEED_CONSEC=3
INTERVAL=30
LOAD_MAX="1.0"
GATE_MODE="${AB_LOAD_GATE_MODE:-gated}"

if [[ "$GATE_MODE" == "record_only" ]]; then
  {
    echo "gate protocol: record_only（専有ゲート opt-out。ユーザー明示指示。イシュー #1519 運用）"
    echo "start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    uptime
  } >"$GATE_LOG"
else
  {
    echo "gate protocol: load1 < ${LOAD_MAX} && util.gpu == 0% for ${NEED_CONSEC} consecutive samples (interval ${INTERVAL}s, max ${MAX_SAMPLES})"
    echo "start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$GATE_LOG"
  consec=0
  passed=0
  for i in $(seq 1 "$MAX_SAMPLES"); do
    load1=$(cut -d' ' -f1 /proc/loadavg)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    ok=0
    if awk -v l="$load1" -v m="$LOAD_MAX" 'BEGIN{exit !(l<m)}' && [[ "$util" == "0" ]]; then ok=1; fi
    echo "sample=$i ts=$(date -u +%H:%M:%SZ) load1=$load1 util_gpu=${util}% compute_apps=$apps gate_ok=$ok" >>"$GATE_LOG"
    if [[ "$ok" == "1" ]]; then consec=$((consec+1)); else consec=0; fi
    if [[ "$consec" -ge "$NEED_CONSEC" ]]; then passed=1; break; fi
    sleep "$INTERVAL"
  done
  if [[ "$passed" != "1" ]]; then
    echo "verdict=undetermined (gate not satisfied within ${MAX_SAMPLES} samples)" >>"$GATE_LOG"
    echo "done. verdict=undetermined" >>"$GATE_LOG"
    exit 0
  fi
  echo "gate passed at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$GATE_LOG"
  uptime >>"$GATE_LOG"
fi

bash ./run_ab_resident_grad_cuda.sh "$LABEL" >"${SELF_DIR}/run-${LABEL}.log" 2>&1
RC=$?
echo "run_ab_resident_grad_cuda exit=$RC" >>"$GATE_LOG"
uptime >>"$GATE_LOG"
echo "end: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$GATE_LOG"
echo "done." >>"$GATE_LOG"
exit "$RC"
