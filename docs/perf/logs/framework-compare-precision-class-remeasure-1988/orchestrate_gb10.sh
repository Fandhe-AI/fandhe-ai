#!/usr/bin/env bash
# GB10 #1988: 精度クラス（#2046）反映後の burn cuda 5 セル・PyTorch cpu 4096 の 5 run 再計測。
# 判定規則は同ディレクトリ RULE.txt（事前登録）。ノード上で実行する（Mac から rsync 転送後）。
set -uo pipefail
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
ROOT="${HOME}/work/rust-ai-library-run"; FC="${ROOT}/scripts/bench/framework-compare"
PY="${HOME}/work/.venv-bench/bin/python"
LOGD="${HOME}/work/gb10-1988"; mkdir -p "${LOGD}"
SHA="$(cat "${ROOT}/.rev-stamp")"
[[ -n "${SHA}" ]] || { echo "ERROR: .rev-stamp が空（転送後に書き込むこと）"; exit 1; }
echo "start $(date -u +%FT%TZ) sha=${SHA}"
uptime > "${LOGD}/uptime_before.txt"

gate() { # 専有ゲート: load1 < 1.0 かつ gpu_util 0%。最大 20 回・30 秒間隔。
  local tag=$1 i l1 gu
  for i in $(seq 1 20); do
    l1=$(cut -d' ' -f1 /proc/loadavg)
    gu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if awk -v l="${l1}" -v g="${gu}" 'BEGIN{exit !(l<1.0 && g==0)}'; then
      echo "${tag} attempt=${i} load1=${l1} gpu_util=${gu} pass" | tee -a "${LOGD}/load_gate.log"; return 0
    fi
    echo "${tag} attempt=${i} load1=${l1} gpu_util=${gu} wait" >> "${LOGD}/load_gate.log"; sleep 30
  done
  echo "${tag} attempt=20 load1=${l1} gpu_util=${gu} fail" | tee -a "${LOGD}/load_gate.log"; return 1
}

cd "${FC}"
echo "-- build 3 binaries ($(date -u +%FT%TZ))"
# #2046 は bench-common を変更しているため、同一 run 内比に使う 3 バイナリを同じツリーから再ビルドする
# （run_all_cuda.sh の build と同じフラグ。bench-fandhe は registry ピン =0.9.0 のまま patch なし）。
ls -l --time-style=full-iso target/release/bench-burn target/release/bench-candle target/release/bench-fandhe > "${LOGD}/build.log" 2>&1
echo "precision_class refs in bench-burn/src/main.rs: $(grep -c precision_class bench-burn/src/main.rs)" >> "${LOGD}/build.log"
# ビルド失敗時は古いバイナリ（target 保持）で計測して rev_stamp と食い違うのを防ぐため、計測開始前に停止する
# （codex-review 指摘・PR #2047。本実測では 3 本とも rc=0。build.log 参照）。
cargo build --release -p bench-fandhe >> "${LOGD}/build.log" 2>&1 || { echo "build bench-fandhe rc=$? -> 計測を停止"; exit 1; }; echo "build bench-fandhe rc=0"
cargo build --release -p bench-candle --no-default-features --features cuda >> "${LOGD}/build.log" 2>&1 || { echo "build bench-candle rc=$? -> 計測を停止"; exit 1; }; echo "build bench-candle rc=0"
cargo build --release -p bench-burn --no-default-features --features cuda >> "${LOGD}/build.log" 2>&1 || { echo "build bench-burn rc=$? -> 計測を停止"; exit 1; }; echo "build bench-burn rc=0"
ls -l --time-style=full-iso target/release/bench-burn target/release/bench-candle target/release/bench-fandhe >> "${LOGD}/build.log" 2>&1
grep -E 'fandhe-ai v' <(cargo tree -p bench-fandhe --depth 1 2>/dev/null) | head -2 >> "${LOGD}/build.log"
"${PY}" -c "import torch;print('torch', torch.__version__, 'threads', torch.get_num_threads())" >> "${LOGD}/build.log" 2>&1
ls "${HOME}/work/cache" >> "${LOGD}/build.log" 2>&1

for r in 1 2 3 4 5; do
  RD="${LOGD}/run${r}"; mkdir -p "${RD}"; OUT="${RD}/results.jsonl"; : > "${OUT}"; ERR="${RD}/err.log"; : > "${ERR}"
  gate "run=${r}"
  echo "-- run${r} ($(date -u +%FT%TZ))"
  for n in 256 512 1024 2048 4096; do
    for mode in reuse fresh; do
      ./target/release/bench-fandhe --task gemm --device cuda --size "${n}" --mode "${mode}" --out "${OUT}" 2>>"${ERR}" || echo "run${r} bench-fandhe ${n} ${mode} FAIL"
    done
    ./target/release/bench-candle --task gemm --device cuda --size "${n}" --mode fresh --out "${OUT}" 2>>"${ERR}" || echo "run${r} bench-candle ${n} FAIL"
    ./target/release/bench-burn --task gemm --device cuda --size "${n}" --mode fresh --out "${OUT}" 2>>"${ERR}" || echo "run${r} bench-burn ${n} FAIL"
  done
  PYOUT="${RD}/py.jsonl"; : > "${PYOUT}"
  ( cd "${HOME}/work" && "${PY}" ./bench_py.py --framework pytorch --task gemm --device cpu --size 4096 --out "${PYOUT}" 2>>"${ERR}" ) || echo "run${r} pytorch cpu 4096 FAIL"
  echo "run${r} rows=$(wc -l < "${OUT}") py_rows=$(wc -l < "${PYOUT}")"
  grep -E '"framework":"burn"' "${OUT}" | sed -E 's/.*"size":([0-9]+).*"parity_fail_count":([0-9]+).*"parity_scaled_abs_rescued":([0-9]+).*/  burn N=\1 fail=\2 rescued=\3/'
  sed -E 's/.*"parity_fail_count": ?([0-9]+).*"parity_scaled_abs_rescued": ?([0-9]+).*/  pytorch cpu 4096 fail=\1 rescued=\2/' "${PYOUT}"
done

{
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "rev_stamp=${SHA}"
  echo "uname=$(uname -srm)"
  echo "rustc=$(rustc -V)"; echo "cargo=$(cargo -V)"
  echo "nvcc=$(nvcc --version | tail -1)"
  nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
  echo "nproc=$(nproc)"
  echo "torch=$("${PY}" -c 'import torch;print(torch.__version__)')"
  echo "tree=<home>/work/rust-ai-library-run"
  echo "load_gate=load1<1.0 && gpu_util==0 (see load_gate.log)"
} > "${LOGD}/env_info.txt"
uptime > "${LOGD}/uptime_after.txt"
echo "done. $(date -u +%FT%TZ)"
