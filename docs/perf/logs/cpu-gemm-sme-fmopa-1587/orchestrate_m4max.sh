#!/bin/sh
# イシュー #1978（#1587 事前登録規則 R4）: SME vs NEON マイクロ A/B の正式
# 16 格子点（`gemm_blis::tests::sme_vs_neon_ab_r4_grid`）を 5 プロセス独立
# 起動する。各 run 開始前に load1 < 8.0 を 30 秒間隔・最大 30 分待つ
# （RULE.txt）。`--dry-run` は実行内容の表示のみ。事前に
# `cargo test -p fandhe-ai-backend-cpu --release --lib --no-run` を済ませて
# おく（計測中にビルドを並走させない）。bash 3.2 安全のため ${VAR} を使う。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${HERE}/../../../.." && pwd)
CMD="cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored sme_vs_neon_ab_r4_grid --nocapture --test-threads=1"
if [ "${1:-}" = "--dry-run" ]; then
  echo "[dry-run] ${CMD} x5 -> ${HERE}/sme_r4_grid_run{1..5}.log"
  exit 0
fi
for i in 1 2 3 4 5; do
  if [ -e "${HERE}/sme_r4_grid_run${i}.log" ]; then
    echo "sme_r4_grid_run${i}.log が既に存在する（差し替え禁止）" >&2
    exit 1
  fi
done
cd "${ROOT}" || exit 1
for i in 1 2 3 4 5; do
  waited=0
  status="timeout"
  while [ "${waited}" -lt 1800 ]; do
    l1=$(sysctl -n vm.loadavg | awk '{print $2}')
    ok=$(awk -v a="${l1}" 'BEGIN{print (a<8.0)?1:0}')
    if [ "${ok}" = "1" ]; then status="pass"; break; fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "run${i} gate=${status} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${HERE}/load_gate_r4.log"
  ${CMD} 2>&1 | grep -E "^(variant=|test |SME )" > "${HERE}/sme_r4_grid_run${i}.log"
  echo "run${i} end_load1=$(sysctl -n vm.loadavg | awk '{print $2}')" >> "${HERE}/load_gate_r4.log"
done
echo "series done" >> "${HERE}/load_gate_r4.log"
