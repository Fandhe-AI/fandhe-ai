#!/bin/sh
# イシュー #3209: Mac mini で m=64 細タイル候補（#3208）の到達追跡（trace）と純カーネル時間（kern）を
# 5 run 取る計測本体。規則の正は同ディレクトリの RULE.txt。雛形は metal-tile-macmini-3181/
# orchestrate_macmini.sh（.rev-stamp・差し替え禁止・ビルド失敗の即中止・負荷ゲート）。
# train/infer セル A/B は別プロセス（scripts/bench/framework-compare/run_ab_m64_thin_tile_metal.sh）で、
# 本スクリプトと同時に走らせない（RULE 項 14）。
# 使い方（Mac mini 上。展開ツリーの docs/perf/logs/metal-m64-thin-tile-ab-macmini-3209/ から）:
#   CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai OUT=<出力ディレクトリ> sh orchestrate_macmini.sh
# bash 3.2／sh 互換のため変数は必ず ${VAR} で参照する。外部入力を eval・展開しない（テスト名は固定）。
# ホスト名（uname -n）・ユーザー名・$HOME は出力しない（RULE 項 2 のマスク方針）。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${HERE}/../../../.." && pwd)
: "${OUT:?OUT（出力ディレクトリ）が未指定}"
: "${CARGO_TARGET_DIR:?CARGO_TARGET_DIR が未指定（target は 1 つに保つ）}"
export CARGO_TARGET_DIR
GATE_LOAD=3.0
MOD="gemm_m64_thin_tile_ab_diag_tests"
PREREQ_MOD="gemm_m64_thin_tile_diag_tests"

[ -f "${ROOT}/.rev-stamp" ] || { echo ".rev-stamp が無い（腕の同定不能）" >&2; exit 1; }
mkdir -p "${OUT}"
for f in prereq_bitident.log parity.log load_gate.log rev-stamp.txt env_info.txt build_tests.log; do
  if [ -e "${OUT}/${f}" ]; then
    echo "${f} が既に存在する（差し替え禁止。再実行は別の OUT で行う）" >&2
    exit 1
  fi
done
for i in 1 2 3 4 5; do
  for f in trace_run${i}.log kern_run${i}.log; do
    if [ -e "${OUT}/${f}" ]; then
      echo "${f} が既に存在する（差し替え禁止）" >&2
      exit 1
    fi
  done
done
cp "${ROOT}/.rev-stamp" "${OUT}/rev-stamp.txt"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  sw_vers
  sysctl -n machdep.cpu.brand_string
  sysctl hw.perflevel0.logicalcpu hw.perflevel1.logicalcpu hw.memsize
  rustc --version
  uptime
} > "${OUT}/env_info.txt" 2>&1

# ---- ビルド（計測より前に全て終える。失敗なら即中止） ----
cd "${ROOT}" || exit 1
if ! cargo test --release -p fandhe-ai-backend-metal --lib --no-run > "${OUT}/build_tests.log" 2>&1; then
  echo "BUILD FAILED（backend-metal テストバイナリ）" >&2; tail -20 "${OUT}/build_tests.log" >&2; exit 1
fi
echo "build-done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"

# ---- 負荷ゲート（load1 < 3.0 を 30 秒間隔・最大 30 分。$1 = ラベル） ----
# 通過以外（GATE-TIMEOUT／GATE-INVALID）は gate= に記録し、aggregate.py が正式系列から外す。
gate() {
  waited=0
  status="timeout"
  while [ "${waited}" -lt 1800 ]; do
    l1=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
    case "${l1}" in
      ''|*[!0-9.]*) status="invalid"; l1="NA"; break ;;
    esac
    ok=$(awk -v a="${l1}" -v t="${GATE_LOAD}" 'BEGIN{print (a<t)?1:0}')
    if [ "${ok}" = "1" ]; then status="pass"; break; fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "$1 gate=${status} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
}

run_diag() { # $1 = モジュール、$2 = テスト名（空なら全件）、$3 = 出力ファイル
  if [ -n "$2" ]; then
    cargo test --release -p fandhe-ai-backend-metal --lib "$1::$2" \
      -- --ignored --nocapture --exact --test-threads=1 > "$3" 2>&1
  else
    cargo test --release -p fandhe-ai-backend-metal --lib "$1" \
      -- --ignored --nocapture --test-threads=1 > "$3" 2>&1
  fi
}

# ---- 前提ゲート（RULE 項 6。性能 run の前に 1 回。失敗なら系列を開始しない） ----
gate "prereq-bitident"
if ! run_diag "${PREREQ_MOD}" "" "${OUT}/prereq_bitident.log"; then
  echo "prereq_bitident FAILED（系列を開始しない）" >> "${OUT}/load_gate.log"
  echo "PREREQ BIT-IDENT FAILED" >&2; tail -20 "${OUT}/prereq_bitident.log" >&2; exit 1
fi
gate "parity"
if ! run_diag "${MOD}" m64_thin_candidates_match_cpu_reference "${OUT}/parity.log"; then
  echo "parity FAILED（系列を開始しない）" >> "${OUT}/load_gate.log"
  echo "PARITY FAILED" >&2; tail -20 "${OUT}/parity.log" >&2; exit 1
fi

# ---- 5 run（各 run: 到達追跡 → 純カーネル計測） ----
for i in 1 2 3 4 5; do
  gate "run${i}-trace"
  run_diag "${MOD}" m64_thin_reach_trace "${OUT}/trace_run${i}.log" \
    || echo "run${i} FAILED trace" >> "${OUT}/load_gate.log"
  gate "run${i}-kern"
  run_diag "${MOD}" m64_thin_candidates_kernel_gpu "${OUT}/kern_run${i}.log" \
    || echo "run${i} FAILED kern" >> "${OUT}/load_gate.log"
  echo "run${i} end_load1=$(sysctl -n vm.loadavg | awk '{print $2}')" >> "${OUT}/load_gate.log"
done
echo "series done" >> "${OUT}/load_gate.log"
