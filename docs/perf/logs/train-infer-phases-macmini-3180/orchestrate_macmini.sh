#!/bin/sh
# イシュー #3180: Mac mini で train／infer の phase 内訳（腕 A）・同期回数（腕 B）・同期 1 回の固定費を
# 取る計測本体。規則の正は同ディレクトリの RULE.txt。合成元は
#   - framework-compare-0.11.0-macmini-remeasure/scripts/macmini-run-0110.sh（.rev-stamp・tree.txt 検査・失敗検知）
#   - train-infer-phases-0.9.0-1980-1981/orchestrate_m4max.sh（8 セルのループ・差し替え禁止）
# 使い方（Mac mini 上。展開ツリーの docs/perf/logs/train-infer-phases-macmini-3180/ から）:
#   CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai OUT=<出力ディレクトリ> sh orchestrate_macmini.sh
# bash 3.2／sh 互換のため変数は必ず ${VAR} で参照する。外部入力を eval・展開しない（セル列は固定）。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${HERE}/../../../.." && pwd)
FC="${ROOT}/scripts/bench/framework-compare"
: "${OUT:?OUT（出力ディレクトリ）が未指定}"
: "${CARGO_TARGET_DIR:?CARGO_TARGET_DIR が未指定（target は 1 つに保つ）}"
export CARGO_TARGET_DIR
# framework-compare は独自の [workspace]・Cargo.lock を持つため target を分ける（main workspace の成果物と混ぜない）
FC_TARGET="${CARGO_TARGET_DIR}-fc"
GATE_LOAD=3.0

[ -f "${ROOT}/.rev-stamp" ] || { echo ".rev-stamp が無い（腕の同定不能）" >&2; exit 1; }
mkdir -p "${OUT}/counters" "${OUT}/sync-fixed-cost"
for i in 1 2 3 4 5; do
  if [ -e "${OUT}/run${i}.jsonl" ]; then
    echo "run${i}.jsonl が既に存在する（差し替え禁止）" >&2
    exit 1
  fi
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
cd "${FC}" || exit 1
# shellcheck disable=SC1091
. ./bench_fandhe_lock_restore.sh
bench_fandhe_setup_lock_restore_trap
if ! CARGO_TARGET_DIR="${FC_TARGET}" cargo build --release -p bench-fandhe > "${OUT}/build_bench.log" 2>&1; then
  echo "BUILD FAILED（bench-fandhe）" >&2; tail -20 "${OUT}/build_bench.log" >&2; exit 1
fi
CARGO_TARGET_DIR="${FC_TARGET}" cargo tree -p bench-fandhe --depth 1 | grep -E 'fandhe-ai v' > "${OUT}/tree.txt"
if ! grep -q 'fandhe-ai v0.11.0$' "${OUT}/tree.txt" || [ "$(wc -l < "${OUT}/tree.txt" | tr -d ' ')" != 1 ]; then
  echo "registry pin 0.11.0 not resolved（腕 A の同定に失敗）" >&2; exit 1
fi
BIN="${FC_TARGET}/release/bench-fandhe"
[ -x "${BIN}" ] || { echo "bench-fandhe が無い" >&2; exit 1; }

cd "${ROOT}" || exit 1
if ! cargo test --release --no-run -p fandhe-ai --test mnist_scale_train_reuse_bench > "${OUT}/build_tests.log" 2>&1 \
  || ! cargo test --release --no-run -p fandhe-ai-backend-metal --test sync_fixed_cost_diag >> "${OUT}/build_tests.log" 2>&1 \
  || ! cargo test --release --no-run -p fandhe-ai-backend-metal --lib >> "${OUT}/build_tests.log" 2>&1; then
  echo "BUILD FAILED（テストバイナリ）" >&2; tail -20 "${OUT}/build_tests.log" >&2; exit 1
fi
echo "build-done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"

# ---- 負荷ゲート（load1 < 3.0 を 30 秒間隔・最大 30 分） ----
gate() {
  waited=0
  status="timeout"
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

# ---- 腕 A（phase 内訳）＋ 同期 1 回の固定費（各 run の後に 1 プロセス） ----
for i in 1 2 3 4 5; do
  gate "run${i}"
  for dev in cpu metal; do
    for task in train infer; do
      for mode in fresh reuse; do
        "${BIN}" --task "${task}" --device "${dev}" --mode "${mode}" --phases \
          --out "${OUT}/run${i}.jsonl" 2>> "${OUT}/run${i}.err" > /dev/null \
          || echo "run${i} FAILED dev=${dev} task=${task} mode=${mode}" >> "${OUT}/load_gate.log"
      done
    done
  done
  echo "run${i} end_load1=$(sysctl -n vm.loadavg | awk '{print $2}')" >> "${OUT}/load_gate.log"
  gate "run${i}-syncfc"
  (cd "${ROOT}" && cargo test --release -p fandhe-ai-backend-metal --test sync_fixed_cost_diag \
    -- --ignored --nocapture --test-threads=1) > "${OUT}/sync-fixed-cost/run${i}.log" 2>&1 \
    || echo "run${i} FAILED sync_fixed_cost_diag" >> "${OUT}/load_gate.log"
done

# ---- 腕 B（同期回数。決定的な値。1 回ずつ。assert は緩めない） ----
cd "${ROOT}" || exit 1
for t in mnist_scale_train_reuse_metal_batch_counters:batch_counters mnist_scale_train_reuse_metal_backward_dinput_phase:backward_dinput_phase; do
  name="${t%%:*}"
  file="${t#*:}"
  cargo test --release -p fandhe-ai --test mnist_scale_train_reuse_bench "${name}" \
    -- --ignored --nocapture --exact --test-threads=1 > "${OUT}/counters/${file}.log" 2>&1 \
    || echo "counters ${name} rc!=0（assert 不一致を含む。ログに実測値）" >> "${OUT}/load_gate.log"
done
# クロスチェック（wall − GPU busy）
cargo test --release -p fandhe-ai-backend-metal --lib infer_fixed_cost_diag \
  -- --ignored --nocapture --test-threads=1 > "${OUT}/sync-fixed-cost/infer_fixed_cost_diag.log" 2>&1 \
  || echo "FAILED infer_fixed_cost_diag" >> "${OUT}/load_gate.log"
echo "series done" >> "${OUT}/load_gate.log"
