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
  for f in "sme_r4_grid_run${i}.log" "sme_r4_grid_run${i}.raw.log"; do
    if [ -e "${HERE}/${f}" ]; then
      echo "${f} が既に存在する（差し替え禁止）" >&2
      exit 1
    fi
  done
done
cd "${ROOT}" || exit 1
for i in 1 2 3 4 5; do
  waited=0
  status="timeout"
  while [ "${waited}" -lt 1800 ]; do
    # 負荷を取得できない（sysctl 失敗・空・非数値）場合はゲート通過と
    # 見なさず `unavailable` として記録する（RULE.txt の「5/5 通過のみ正式」
    # に対し参考系列へ分類するための fail-closed。run_ab_sme_cpu.sh と同型）
    l1=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
    case "${l1}" in
      ''|*[!0-9.]*) status="unavailable"; l1="NA"; break ;;
    esac
    ok=$(awk -v a="${l1}" 'BEGIN{print (a<8.0)?1:0}')
    if [ "${ok}" = "1" ]; then status="pass"; break; fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "run${i} gate=${status} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${HERE}/load_gate_r4.log"
  # 未加工の出力（ビルド・panic 本文を含む）は .raw.log へ保存し、集計用の
  # 抽出行のみ .log へ書く。計測プロセスの失敗（非ゼロ終了）・抽出行 0 件は
  # 非ゼロ終了で伝播し、残りの run を続行しない（ログは差し替え禁止のため
  # 障害原因は .raw.log から確認する）
  ${CMD} > "${HERE}/sme_r4_grid_run${i}.raw.log" 2>&1
  rc=$?
  grep -E "^(variant=|test |SME )" "${HERE}/sme_r4_grid_run${i}.raw.log" > "${HERE}/sme_r4_grid_run${i}.log"
  grc=$?
  echo "run${i} exit=${rc} grep_exit=${grc} end_load1=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')" >> "${HERE}/load_gate_r4.log"
  if [ "${rc}" -ne 0 ] || [ "${grc}" -ne 0 ]; then
    echo "run${i}: 計測プロセス失敗（exit=${rc} grep_exit=${grc}）。${HERE}/sme_r4_grid_run${i}.raw.log を確認" >&2
    exit 1
  fi
done
echo "series done" >> "${HERE}/load_gate_r4.log"
