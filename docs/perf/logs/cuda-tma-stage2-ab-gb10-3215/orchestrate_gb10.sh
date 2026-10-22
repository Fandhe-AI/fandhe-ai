#!/usr/bin/env bash
# イシュー #3215: GB10 上で TMA Stage 2 の A/B 一式を順に実行する（規則の正は同ディレクトリの RULE.txt）。
# 役割: ツリー指紋 → env 記録 → ビルド（計測前に完了）→ ゲート A → ゲート B → 純カーネル 5 プロセス →
#   cuBLAS 5 run（fp32/tf32）→ gemm セル 5 round を、各プロセスの前に専有ゲートを通して直列に実行する。
#   判定はしない（集計は aggregate.py）。失敗した段は rc を stage_rc.txt に記録して継続し、最後に固有マーカー
#   `ORCHESTRATE-DONE` を出す（完了待ちはこの行で行う。rc の grep で待たない）。
# 継承元: docs/perf/logs/opp-kernel-gb10-3185/run-gb10.sh（専有ゲート・版固定）・cuda-tma-stage1-1975（ゲート構成）。
#
# 使い方（GB10 上。ツリーは Mac で `git archive <計測 SHA>` を展開して転送済みで REPO_DIR/.rev-stamp がある）:
#   REPO_DIR=<転送ツリー絶対パス> OUT=<出力ディレクトリ(未作成の絶対パス)> VENV_PY=<~/work/.venv-bench/bin/python> \
#     setsid nohup bash orchestrate_gb10.sh > <OUT の外のログ> 2>&1 &
set -uo pipefail
export LC_ALL=C

: "${REPO_DIR:?}" "${OUT:?}" "${VENV_PY:?}"
if [[ "${REPO_DIR}" != /* || "${OUT}" != /* || -e "${OUT}" ]]; then
  echo "REPO_DIR/OUT は絶対パス、OUT は未作成であること（上書き禁止）" >&2
  exit 2
fi
HERE="${REPO_DIR}/docs/perf/logs/cuda-tma-stage2-ab-gb10-3215"
FC="${REPO_DIR}/scripts/bench/framework-compare"
BENCH_PY="${REPO_DIR}/docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py"
BENCH_PY_SHA256="9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b"
GATE_LOAD=1.0
mkdir -p "${OUT}/cublas"
GATE_LOG="${OUT}/load_gate.log"
STAGE_RC="${OUT}/stage_rc.txt"
: >"${GATE_LOG}"
: >"${STAGE_RC}"

mask() { # ホスト名・ユーザー名・ホームを伏せる（RULE 項 1）
  sed -e "s|${HOME}|<home>|g" -e "s|$(whoami)|<user>|g" -e "s|$(hostname)|<node>|g" -e "s|$(hostname -s)|<node>|g"
}

wait_gate() { # $1=ラベル。load1 < GATE_LOAD かつ GPU 利用率 0% になるまで 30 秒間隔で最大 30 分。
  local i=0 load util
  while [[ "${i}" -le 60 ]]; do
    load="$(uptime | sed -E 's/.*load averages?: *([0-9.]+).*/\1/')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    case "${load}" in '' | *[!0-9.]*) echo "GATE-INVALID $1 load=${load}" >>"${GATE_LOG}"; return 5 ;; esac
    case "${util}" in '' | *[!0-9]*) echo "GATE-INVALID $1 util=${util}" >>"${GATE_LOG}"; return 5 ;; esac
    if awk -v l="${load}" -v t="${GATE_LOAD}" 'BEGIN{exit !(l<t)}' && [[ "${util}" == 0 ]]; then
      echo "GATE-PASS $1 load1=${load} gpu_util=${util}" >>"${GATE_LOG}"
      return 0
    fi
    i=$((i + 1))
    sleep 30
  done
  echo "GATE-TIMEOUT $1" >>"${GATE_LOG}"
  return 6
}

stage() { # stage <名前> <ログ> <コマンド...>: 専有ゲート → 実行。不通過は実行せず rc を記録（正式系列から外す）。
  local name=$1 log=$2
  shift 2
  if ! wait_gate "${name}"; then
    echo "${name} rc=GATE-FAIL" >>"${STAGE_RC}"
    return 0
  fi
  uptime >"${OUT}/uptime_before_${name}.txt"
  "$@" >"${log}" 2>&1
  local rc=$?
  uptime >"${OUT}/uptime_after_${name}.txt"
  echo "${name} rc=${rc}" >>"${STAGE_RC}"
  return 0
}

# --- ツリー指紋・env 記録 ---
python3 -I "${HERE}/tree_fingerprint.py" "${REPO_DIR}" "${OUT}/tree-fingerprint.txt" >"${OUT}/tree-fingerprint.log" 2>&1
cp "${REPO_DIR}/.rev-stamp" "${OUT}/rev-stamp.txt"
{
  nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
  nvcc --version || true
  uname -m
  rustc -V
  cargo -V
  "${VENV_PY}" -c 'import sys,torch,numpy; print(sys.version.split()[0], torch.__version__, numpy.__version__, torch.version.cuda)'
  echo "bench_py.py sha256: $(sha256sum <"${BENCH_PY}" | cut -d' ' -f1)"
  echo "lscpu model: $(lscpu | grep -m1 'Model name' || true)"
} 2>&1 | mask >"${OUT}/env_info.txt"

# --- 版固定の検査（fail-closed。別版へ読み替えない） ---
if ! "${VENV_PY}" -c 'import sys,torch; sys.exit(0 if torch.__version__=="2.14.0+cu130" else 4)'; then
  echo "torch が 2.14.0+cu130 でない" >&2
  echo "venv rc=FAIL" >>"${STAGE_RC}"
  echo "ORCHESTRATE-DONE aborted=torch-version"
  exit 4
fi
if [[ "$(sha256sum <"${BENCH_PY}" | cut -d' ' -f1)" != "${BENCH_PY_SHA256}" ]]; then
  echo "bench_py.py の sha256 が RULE と不一致" >&2
  echo "bench_py rc=FAIL" >>"${STAGE_RC}"
  echo "ORCHESTRATE-DONE aborted=bench-py-sha"
  exit 4
fi

# --- ビルド（計測前に完了させる。RULE 項 12） ---
cd "${REPO_DIR}" || exit 2
{
  cargo test -p fandhe-ai-backend-cuda --release --features internal-diagnostics --no-run &&
    cargo build -p fandhe-ai-backend-cuda --release --features internal-diagnostics --example gemm_candidate_event_ab_3215
} >"${OUT}/build.log" 2>&1
BUILD_RC=$?
echo "build rc=${BUILD_RC}" >>"${STAGE_RC}"
if [[ "${BUILD_RC}" != 0 ]]; then
  echo "ORCHESTRATE-DONE aborted=build"
  exit 1
fi

TEST=(cargo test -p fandhe-ai-backend-cuda --release --features internal-diagnostics)

# --- ゲート A ---
stage gateA "${OUT}/gateA_cpu_cuda_tiled_pipeline_tma_parity.log" \
  "${TEST[@]}" --test cpu_cuda_tiled_pipeline_tma_parity -- --ignored --nocapture --test-threads=1

# --- ゲート B（6 バイナリ） ---
for t in parity_nonregression cpu_cuda_parity cpu_cuda_tiled_pipeline_parity gemm_transposed_parity transpose_parity gemm_tiled; do
  stage "gateB_${t}" "${OUT}/gateB_${t}.log" "${TEST[@]}" --test "${t}" -- --ignored --nocapture
done

# --- 純カーネル 5 プロセス（主判定） ---
for run in 1 2 3 4 5; do
  stage "kern_run${run}" "${OUT}/kern_run${run}.log" \
    cargo run -p fandhe-ai-backend-cuda --release --features internal-diagnostics --example gemm_candidate_event_ab_3215
done

# --- cuBLAS 5 run（fp32/tf32。run ごとに起動順を反転） ---
for run in 1 2 3 4 5; do
  mkdir -p "${OUT}/cublas/run${run}"
  if [[ $((run % 2)) -eq 1 ]]; then arms="fp32 tf32"; else arms="tf32 fp32"; fi
  for arm in ${arms}; do
    stage "cublas_run${run}_${arm}" "${OUT}/cublas/run${run}/${arm}.stdout.log" \
      "${VENV_PY}" "${FC}/opp_kernel_cuda_diag.py" measure --bench-py "${BENCH_PY}" --arm "${arm}" --run "${run}" \
      --out "${OUT}/cublas/run${run}/${arm}.jsonl"
  done
done

# --- gemm セル（5 round。専有ゲートはスクリプト内で round ごとに通す） ---
CELL_OUT="${OUT}/cell" bash "${FC}/run_cell_gemm_cuda_3215.sh" head-3215 "${REPO_DIR}/crates/facade" \
  >"${OUT}/cell.stdout.log" 2>&1
echo "cell rc=$?" >>"${STAGE_RC}"
echo "ORCHESTRATE-DONE stages=$(wc -l <"${STAGE_RC}")"
