#!/usr/bin/env bash
# イシュー #3189: GB10 で CUDA train／infer の phase 内訳（腕 A）・#2116 forward 診断（B-0）・
# infer 内部内訳（B-2）・launch／同期の固定費（B-3）・同期回数（B-1。nsys）・unified_memory 属性（B-4）を
# 取る計測本体。規則の正は同ディレクトリの RULE.txt（計測前に固定）。構成は
#   - train-infer-phases-macmini-3180/orchestrate_macmini.sh（腕 A のループ・.rev-stamp・tree.txt 検査）
#   - cuda-train-forward-resident-2116/orchestrate.sh（負荷ゲート・マスク・差し替え禁止）
# と同型。
# 使い方（GB10 上。展開ツリーのルートから）:
#   PATH=/usr/local/cuda/bin:$PATH CARGO_TARGET_DIR=<ツリー外の絶対パス> OUT=<出力ディレクトリ> \
#     bash docs/perf/logs/cuda-train-infer-gb10-3189/orchestrate_gb10.sh
# 変数は必ず ${VAR} で参照し、外部入力を eval・展開しない（セル列・窓・K は固定）。
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
FC="${ROOT}/scripts/bench/framework-compare"
: "${OUT:?OUT（出力ディレクトリ）が未指定}"
: "${CARGO_TARGET_DIR:?CARGO_TARGET_DIR が未指定（ツリーの外に置く）}"
export CARGO_TARGET_DIR
FC_TARGET="${CARGO_TARGET_DIR}-fc"

[ -f "${ROOT}/.rev-stamp" ] || { echo ".rev-stamp が無い（腕の同定不能）" >&2; exit 1; }
command -v nsys >/dev/null 2>&1 || NSYS_MISSING=1
mkdir -p "${OUT}/armA" "${OUT}/infer-diag" "${OUT}/fixed-cost" "${OUT}/census"
# 相対指定の OUT は最初の cd より前に絶対パス化する（cd 後の相対解決で出力先がずれるのを防ぐ）
OUT="$(cd "${OUT}" && pwd)" || exit 1
for i in 1 2 3 4 5; do
  for f in "armA/run${i}.jsonl" "infer-diag/run${i}.facade.jsonl" "infer-diag/run${i}.backend.jsonl" "fixed-cost/run${i}.log"; do
    if [ -e "${OUT}/${f}" ]; then
      echo "${f} が既に存在する（差し替え禁止）" >&2
      exit 1
    fi
  done
done
cp "${ROOT}/.rev-stamp" "${OUT}/rev-stamp.txt"

mask() {
  sed -e "s|${HOME}|<home>|g" -e "s|$(hostname)|masked|g" -e "s|$(whoami)|user|g"
}

{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname: $(uname -srm)"
  if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|Architecture' || true
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader || true
    nvidia-smi | grep -o 'CUDA Version: [0-9.]*' || true
  else
    echo "nvidia-smi: unavailable"
  fi
  echo "rustc: $(rustc --version)"
  echo "cargo: $(cargo --version)"
  command -v nsys >/dev/null 2>&1 && echo "nsys: $(nsys --version)"
} 2>&1 | mask > "${OUT}/env_info.txt"

# ---- ビルド（計測より前に全て終える。失敗なら即中止） ----
cd "${FC}" || exit 1
# shellcheck disable=SC1091
. ./bench_fandhe_lock_restore.sh
bench_fandhe_setup_lock_restore_trap
# cargo の "Compiling … (/home/…)" 行が絶対パスを含むためビルドログも mask を通す（pipefail で cargo の失敗を保持）
# subshell 内では親の EXIT trap（Cargo.lock 復元）を解除する（subshell 終了時の二重復元・バックアップ不在での非 0 を防ぐ）
if ! (trap - EXIT ERR; set -o pipefail; CARGO_TARGET_DIR="${FC_TARGET}" cargo build --release -p bench-fandhe 2>&1 | mask > "${OUT}/build_bench.log"); then
  echo "BUILD FAILED（bench-fandhe）" >&2; tail -20 "${OUT}/build_bench.log" >&2; exit 1
fi
CARGO_TARGET_DIR="${FC_TARGET}" cargo tree -p bench-fandhe --depth 1 | grep -E 'fandhe-ai v' > "${OUT}/tree.txt"
if ! grep -q 'fandhe-ai v0.11.0$' "${OUT}/tree.txt" || [ "$(wc -l < "${OUT}/tree.txt" | tr -d ' ')" != 1 ]; then
  echo "registry pin 0.11.0 not resolved（腕 A の同定に失敗）" >&2; exit 1
fi
BIN="${FC_TARGET}/release/bench-fandhe"
[ -x "${BIN}" ] || { echo "bench-fandhe が無い" >&2; exit 1; }

cd "${ROOT}" || exit 1
# テストバイナリのパスを cargo の JSON 出力から特定する（第 1 引数: test ターゲット名。lib は "lib:<crate>"）。
find_bin() {
  # $(find_bin …) の command substitution で呼ぶ前提。subshell 側で親の EXIT trap（lock 復元）を解除する
  trap - EXIT ERR
  local want="$1"
  shift
  cargo test --release --locked "$@" --no-run --message-format=json 2> /dev/null |
    python3 -I -c '
import json, sys
want = sys.argv[1]
exe = None
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") != "compiler-artifact" or not m.get("executable"):
        continue
    t = m.get("target", {})
    if want.startswith("lib:"):
        if t.get("name", "").replace("-", "_") == want[4:] and "lib" in t.get("kind", []) and m.get("profile", {}).get("test"):
            exe = m["executable"]
    elif t.get("name") == want:
        exe = m["executable"]
if exe is None:
    sys.exit(1)
print(exe)
' "${want}"
}
CENSUS_BIN="$(find_bin cuda_train_infer_sync_census_3189 -p fandhe-ai --test cuda_train_infer_sync_census_3189)" || { echo "census バイナリ特定失敗" >&2; exit 1; }
INFER_BIN="$(find_bin cuda_infer_reuse_phase_diag_3189 -p fandhe-ai --test cuda_infer_reuse_phase_diag_3189)" || { echo "infer facade バイナリ特定失敗" >&2; exit 1; }
BACKEND_BIN="$(find_bin infer_chain_phase_diag_3189 -p fandhe-ai-backend-cuda --test infer_chain_phase_diag_3189)" || { echo "infer backend バイナリ特定失敗" >&2; exit 1; }
LIB_BIN="$(find_bin lib:fandhe_ai_backend_cuda -p fandhe-ai-backend-cuda --lib)" || { echo "backend lib バイナリ特定失敗" >&2; exit 1; }
if ! (trap - EXIT ERR; set -o pipefail; cargo build --release --locked -p fandhe-ai-backend-cuda --features internal-diagnostics --example unified_memory_probe 2>&1 | mask > "${OUT}/build_probe.log"); then
  echo "BUILD FAILED（unified_memory_probe）" >&2; tail -20 "${OUT}/build_probe.log" >&2; exit 1
fi
PROBE_BIN="${CARGO_TARGET_DIR}/release/examples/unified_memory_probe"
[ -x "${PROBE_BIN}" ] || { echo "unified_memory_probe が無い" >&2; exit 1; }
echo "build-done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"

load1() {
  uptime | sed -e 's/.*load averages*: *//' -e 's/,/ /g' | awk '{print $1}'
}
gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2> /dev/null | head -n1 | tr -d ' '
}
# 負荷ゲート（load1 < 1.0 かつ GPU utilization 0%。30 秒間隔・最大 30 分）。結果を load_gate.log へ残す。
gate() {
  local label="$1" waited=0 status=timeout l u
  while [ "${waited}" -lt 1800 ]; do
    l="$(load1)"
    u="$(gpu_util || true)"
    if [[ "${l}" =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ "${u}" =~ ^[0-9]+$ ]]; then
      if awk -v l="${l}" 'BEGIN{exit !(l < 1.0)}' && [ "${u}" -eq 0 ]; then
        status=pass
        break
      fi
    else
      status=unavailable
      break
    fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "${label} gate=${status} load1=${l:-NA} gpu_util=${u:-na} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
}

# ---- 腕 A ＋ B-2 ＋ B-3（各 run に 1 回ずつ） ----
for i in 1 2 3 4 5; do
  gate "run${i}"
  for dev in cuda cpu; do
    for task in train infer; do
      for mode in fresh reuse; do
        "${BIN}" --task "${task}" --device "${dev}" --mode "${mode}" --phases \
          --out "${OUT}/armA/run${i}.jsonl" 2>> "${OUT}/armA/run${i}.err" > /dev/null \
          || echo "run${i} FAILED armA dev=${dev} task=${task} mode=${mode}" >> "${OUT}/load_gate.log"
      done
    done
  done
  gate "run${i}-infer"
  RAW="$(mktemp)"
  "${INFER_BIN}" --ignored --exact cuda_infer_reuse_phases --nocapture --test-threads=1 > "${RAW}" 2>> "${OUT}/infer-diag/run${i}.facade.err" \
    || echo "run${i} FAILED infer facade" >> "${OUT}/load_gate.log"
  grep -o 'DIAG_JSON .*' "${RAW}" | sed -e 's/^DIAG_JSON //' > "${OUT}/infer-diag/run${i}.facade.jsonl"
  "${BACKEND_BIN}" --ignored --exact infer_chain_phase_diag_3189 --nocapture --test-threads=1 > "${RAW}" 2>> "${OUT}/infer-diag/run${i}.backend.err" \
    || echo "run${i} FAILED infer backend" >> "${OUT}/load_gate.log"
  grep -o 'BACKEND_JSON .*' "${RAW}" | sed -e 's/^BACKEND_JSON //' > "${OUT}/infer-diag/run${i}.backend.jsonl"
  gate "run${i}-fixedcost"
  "${LIB_BIN}" --ignored --exact gemm_small_launch_cost_diag_tests::gemm_small_launch_cost_diag --nocapture --test-threads=1 > "${RAW}" 2>&1 \
    || echo "run${i} FAILED fixedcost" >> "${OUT}/load_gate.log"
  grep -o 'DIAG2109 .*' "${RAW}" > "${OUT}/fixed-cost/run${i}.log"
  rm -f "${RAW}"
  echo "run${i} end_load1=$(load1)" >> "${OUT}/load_gate.log"
done

# ---- 腕 B-0（#2116。RULE を継承。無改変で実行） ----
bash "${ROOT}/docs/perf/logs/cuda-train-forward-resident-2116/orchestrate.sh" gb10 \
  || echo "FAILED 2116 orchestrate" >> "${OUT}/load_gate.log"

# ---- 腕 B-4（unified_memory_probe。決定的な値） ----
# pipefail を subshell に閉じ、probe 自体の失敗が mask の終了状態で隠れないようにする
(trap - EXIT ERR; set -o pipefail; "${PROBE_BIN}" 2>&1 | mask > "${OUT}/unified_memory_probe.txt") \
  || echo "FAILED unified_memory_probe" >> "${OUT}/load_gate.log"

# ---- 腕 B-1（nsys による回数計測。時間の統計には使わない。最後に実行） ----
if [ "${NSYS_MISSING:-0}" = 1 ]; then
  echo "nsys が無い（B-1 は未実測）" >> "${OUT}/load_gate.log"
else
  TMPD="$(mktemp -d)"
  for r in 1 2 3 4 5; do
    for w in fwd fwd_bwd full infer_reuse; do
      for k in 20 120; do
        base="${w}-K${k}-r${r}"
        CENSUS_KIND=cuda CENSUS_WINDOW="${w}" CENSUS_STEPS="${k}" \
          nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
          --output "${TMPD}/${base}" \
          "${CENSUS_BIN}" --ignored --exact cuda_train_infer_sync_census --nocapture --test-threads=1 \
          > "${TMPD}/${base}.out" 2>&1 || echo "FAILED census ${base}" >> "${OUT}/load_gate.log"
        grep -o 'CENSUS_JSON .*' "${TMPD}/${base}.out" > "${OUT}/census/${base}.json"
        for rep in cuda_api_sum cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum; do
          nsys stats --report "${rep}" --format csv --force-export=true "${TMPD}/${base}.nsys-rep" 2> /dev/null |
            mask > "${OUT}/census/${base}.${rep}.csv" || echo "FAILED nsys stats ${base} ${rep}" >> "${OUT}/load_gate.log"
        done
        rm -f "${TMPD}/${base}".nsys-rep "${TMPD}/${base}".sqlite
      done
    done
  done
  rm -rf "${TMPD}"
fi
echo "series done" >> "${OUT}/load_gate.log"
