#!/bin/sh
# イシュー #3182: Mac mini で Metal GEMM reuse 計測窓の内訳（Layer A: bench-fandhe --phases／窓合計、
# candle 窓合計・転送分離、Layer B: backend-metal 診断テスト）を取る計測本体。規則の正は同ディレクトリの
# RULE.txt。合成元は train-infer-phases-macmini-3180/orchestrate_macmini.sh（.rev-stamp・tree.txt 検査・
# 負荷ゲート・差し替え禁止）。
# 使い方（Mac mini 上。展開ツリーの docs/perf/logs/metal-gemm-window-macmini-3182/ から）:
#   CARGO_TARGET_DIR=$HOME/work/target-3182 OUT=<出力ディレクトリ> sh orchestrate_macmini.sh
# bash 3.2／sh 互換のため変数は必ず ${VAR} で参照する。外部入力を eval・展開しない（サイズ・テスト名は固定）。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${HERE}/../../../.." && pwd)
FC="${ROOT}/scripts/bench/framework-compare"
: "${OUT:?OUT（出力ディレクトリ）が未指定}"
: "${CARGO_TARGET_DIR:?CARGO_TARGET_DIR が未指定}"
export CARGO_TARGET_DIR
FC_TARGET="${CARGO_TARGET_DIR}-fc"
GATE_LOAD=3.0
SIZES="256 512 1024 2048 4096"

if [ -n "${RAYON_NUM_THREADS:-}" ]; then
  echo "RAYON_NUM_THREADS が設定されている（既定スレッド方針の計測に影響するため中止）" >&2
  exit 1
fi
[ -f "${ROOT}/.rev-stamp" ] || { echo ".rev-stamp が無い（腕の同定不能）" >&2; exit 1; }
mkdir -p "${OUT}"
for i in 1 2 3 4 5; do
  for f in layerA window-reuse fresh candle; do
    if [ -e "${OUT}/${f}-run${i}.jsonl" ]; then
      echo "${f}-run${i}.jsonl が既に存在する（差し替え禁止）" >&2
      exit 1
    fi
  done
  if [ -e "${OUT}/layerB-run${i}.log" ]; then
    echo "layerB-run${i}.log が既に存在する（差し替え禁止）" >&2
    exit 1
  fi
done
cp "${ROOT}/.rev-stamp" "${OUT}/rev-stamp.txt"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  sw_vers
  sysctl -n machdep.cpu.brand_string
  sysctl hw.ncpu hw.perflevel0.logicalcpu hw.perflevel1.logicalcpu hw.memsize
  rustc --version
  uptime
  echo "-- pmset -g therm (before)"
  pmset -g therm 2>&1
} > "${OUT}/env_info.txt" 2>&1

# ---- ビルド（計測より前に全て終える。失敗なら即中止） ----
cd "${FC}" || exit 1
# shellcheck disable=SC1091
. ./bench_fandhe_lock_restore.sh
bench_fandhe_setup_lock_restore_trap
if ! CARGO_TARGET_DIR="${FC_TARGET}" cargo build --release -p bench-fandhe -p bench-candle > "${OUT}/build_bench.log" 2>&1; then
  echo "BUILD FAILED（bench-fandhe／bench-candle）" >&2; tail -20 "${OUT}/build_bench.log" >&2; exit 1
fi
CARGO_TARGET_DIR="${FC_TARGET}" cargo tree -p bench-fandhe --depth 1 | grep -E 'fandhe-ai v' > "${OUT}/tree.txt"
if ! grep -q 'fandhe-ai v0.11.0$' "${OUT}/tree.txt" || [ "$(wc -l < "${OUT}/tree.txt" | tr -d ' ')" != 1 ]; then
  echo "registry pin 0.11.0 not resolved（腕 A の同定に失敗）" >&2; exit 1
fi
BIN_F="${FC_TARGET}/release/bench-fandhe"
BIN_C="${FC_TARGET}/release/bench-candle"
[ -x "${BIN_F}" ] || { echo "bench-fandhe が無い" >&2; exit 1; }
[ -x "${BIN_C}" ] || { echo "bench-candle が無い" >&2; exit 1; }

cd "${ROOT}" || exit 1
if ! cargo test --release --no-run -p fandhe-ai-backend-metal --lib > "${OUT}/build_tests.log" 2>&1; then
  echo "BUILD FAILED（テストバイナリ）" >&2; tail -20 "${OUT}/build_tests.log" >&2; exit 1
fi
echo "build-done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"

# ---- 負荷ゲート（load1 < 3.0 を 30 秒間隔・最大 30 分） ----
gate() {
  waited=0
  status="timeout"
  l1="NA"
  while [ "${waited}" -lt 1800 ]; do
    l1=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
    case "${l1}" in
      ''|*[!0-9.]*) status="unavailable"; l1="NA"; break ;;
    esac
    ok=$(awk -v a="${l1}" -v t="${GATE_LOAD}" 'BEGIN{print (a<t)?1:0}')
    if [ "${ok}" = "1" ]; then status="pass"; break; fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "$1 gate=${status} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
}

fail() { echo "$1 FAILED" >> "${OUT}/load_gate.log"; }

# ---- 5 run（各 run は固定順。失敗は load_gate.log に記録し、集計側が欠損として拒否する） ----
for i in 1 2 3 4 5; do
  gate "run${i}"
  for n in ${SIZES}; do
    "${BIN_F}" --task gemm --device metal --size "${n}" --mode reuse --phases \
      --out "${OUT}/layerA-run${i}.jsonl" 2>> "${OUT}/run${i}.err" > /dev/null || fail "run${i} layerA n=${n}"
  done
  for n in ${SIZES}; do
    "${BIN_F}" --task gemm --device metal --size "${n}" --mode reuse \
      --out "${OUT}/window-reuse-run${i}.jsonl" 2>> "${OUT}/run${i}.err" > /dev/null || fail "run${i} window-reuse n=${n}"
  done
  for n in ${SIZES}; do
    "${BIN_F}" --task gemm --device metal --size "${n}" --mode fresh \
      --out "${OUT}/fresh-run${i}.jsonl" 2>> "${OUT}/run${i}.err" > /dev/null || fail "run${i} fresh n=${n}"
  done
  for n in ${SIZES}; do
    "${BIN_C}" --task gemm --device metal --size "${n}" \
      --out "${OUT}/candle-run${i}.jsonl" 2>> "${OUT}/run${i}.err" > /dev/null || fail "run${i} candle-gemm n=${n}"
  done
  for n in ${SIZES}; do
    "${BIN_C}" --task gemm-transfer-split --device metal --size "${n}" \
      --out "${OUT}/candle-run${i}.jsonl" 2>> "${OUT}/run${i}.err" > /dev/null || fail "run${i} candle-split n=${n}"
  done
  (cd "${ROOT}" && cargo test --release -p fandhe-ai-backend-metal --lib -- --ignored --nocapture --test-threads=1 --exact \
    gemm_reuse_phase_diag_tests::gemm_window_route_precheck \
    gemm_reuse_phase_diag_tests::gemm_window_phase_diag_small_sizes \
    gemm_reuse_phase_diag_tests::gemm_reuse_phase_diag_production_batch) > "${OUT}/layerB-run${i}.log" 2>&1 \
    || fail "run${i} layerB"
  echo "run${i} end_load1=$(sysctl -n vm.loadavg | awk '{print $2}')" >> "${OUT}/load_gate.log"
done
{
  echo "-- pmset -g therm (after)"
  pmset -g therm 2>&1
  uptime
} >> "${OUT}/env_info.txt" 2>&1
echo "series done" >> "${OUT}/load_gate.log"
