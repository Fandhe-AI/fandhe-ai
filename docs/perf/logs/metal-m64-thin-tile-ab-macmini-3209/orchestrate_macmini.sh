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
mkdir -p "${OUT}" || exit 1
# 相対指定の OUT は以降の cd（ROOT へ移動）で別基準になり、項 14 のログ上書き禁止検査を迂回しうるため、
# 検査・保存を始める前に絶対パスへ正規化する。
OUT=$(cd "${OUT}" && pwd) || exit 1
for f in prereq_bitident.log bitident_report.log parity.log load_gate.log rev-stamp.txt env_info.txt build_tests.log; do
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
# 既存の厳密診断（候補 1 件の不一致・fallback で panic する）は維持するが、その失敗で系列全体は止めない。
# 候補別の bit 一致は m64_thin_prereq_bitident_report が記録し、不一致候補は aggregate.py が
# NOT_ADOPTABLE として除外して残りを判定する。rc は load_gate.log へ記録する（FAILED の語は使わない。
# aggregate.py の正式系列判定が FAILED を含む行を系列不成立とするため）。
gate "prereq-bitident"
run_diag "${PREREQ_MOD}" "" "${OUT}/prereq_bitident.log"
echo "prereq_bitident_strict rc=$? （非 0 でも系列は継続。候補別の結果は bitident_report.log）" >> "${OUT}/load_gate.log"
# 候補別記録の失敗は基準 cur 側の異常か基盤の異常なので系列を開始しない。
if ! run_diag "${MOD}" m64_thin_prereq_bitident_report "${OUT}/bitident_report.log"; then
  echo "bitident_report FAILED（系列を開始しない）" >> "${OUT}/load_gate.log"
  echo "BITIDENT REPORT FAILED" >&2; tail -20 "${OUT}/bitident_report.log" >&2; exit 1
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
