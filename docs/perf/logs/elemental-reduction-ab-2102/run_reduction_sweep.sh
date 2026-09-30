#!/usr/bin/env bash
# イシュー #2102: Phase 0（`reduction::tests::reduction_threshold_sweep`）を 5 プロセス独立に
# 起動し、生ログ run1..5.log と env_info.txt を出力先へ保存する。
# 規則の正は `elemental-reduction-threshold-2101/RULE.txt`（計測コマンド・5 プロセス・
# RAYON_NUM_THREADS 未設定）と `RULE.txt`（本ディレクトリ。集計は aggregate_sweep.py）。
# 上書き禁止（出力先に既存ファイルがあれば停止。run の差し替え・追加起動はしない）。
#
# 使い方: bash run_reduction_sweep.sh <machine-label> <絶対パスの出力先>
#   例: bash run_reduction_sweep.sh m4max <repo>/docs/perf/logs/elemental-reduction-threshold-2101/m4max
set -u

LABEL=${1:-}
OUTDIR=${2:-}
# A03 対策: label は英数字と `._-` のみ、出力先は絶対パス。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <machine-label> <absolute-outdir>  (label: [A-Za-z0-9._-]+)" >&2
  exit 1
fi
if [[ -z "$OUTDIR" || "$OUTDIR" != /* ]]; then
  echo "error: outdir must be an absolute path (got: ${OUTDIR:-empty})" >&2
  exit 1
fi
if [[ -d "$OUTDIR" ]] && [[ -n "$(find "$OUTDIR" -mindepth 1 -maxdepth 1 ! -name README.md -print -quit)" ]]; then
  echo "error: $OUTDIR に既存ファイルがあります。上書き禁止のため別の出力先を使ってください" >&2
  exit 1
fi
mkdir -p "$OUTDIR" || exit 1

ENV_INFO="$OUTDIR/env_info.txt"
{
  echo "label=$LABEL"
  echo "uname=$(uname -a)"
  echo "rustc=$(rustc --version 2>&1)"
  echo "head=$(git rev-parse HEAD 2>&1)"
  if [[ -r /proc/cpuinfo ]]; then
    echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //')"
    echo "nproc=$(nproc)"
  else
    echo "cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    echo "nproc=$(sysctl -n hw.ncpu 2>/dev/null)"
  fi
  echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS-<unset>}"
} >"$ENV_INFO"

# 規則上 RAYON_NUM_THREADS は未設定（本番と同一）。設定されていたら拒否する。
if [[ -n "${RAYON_NUM_THREADS:-}" ]]; then
  echo "error: RAYON_NUM_THREADS must be unset (RULE.txt)" >&2
  exit 1
fi

load1() {
  local l
  l=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
  [[ -z "$l" && -r /proc/loadavg ]] && l=$(awk '{print $1}' /proc/loadavg)
  echo "${l:-NA}"
}

FAILED=0
for i in 1 2 3 4 5; do
  echo "run${i}_start load1=$(load1) at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ENV_INFO"
  if ! cargo test -p fandhe-ai-backend-cpu --release --lib \
    reduction::tests::reduction_threshold_sweep -- --ignored --nocapture \
    2>"$OUTDIR/run${i}.log" >/dev/null; then
    echo "error: run${i} failed (see $OUTDIR/run${i}.log)" >&2
    FAILED=1
    break
  fi
  echo "run${i}_end load1=$(load1) at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ENV_INFO"
done
exit "$FAILED"
