#!/usr/bin/env bash
# エンジン単体の対照（スコアボード 0.14.0 の「エンジン単体の対照」カード）を Mac mini で測る実行役。
# 規則の正は同ディレクトリ親の RULE.txt。手順の正は docs/real-hardware-verification-env.md §7（zsh -lc で包む・nohup で切り離す）。
#   prepare <owner> : ロックの owner を確かめ、2 つの target を新規に作って release ビルドし、版を env/ へ記録する（計測しない）
#   series  <owner> : 5 ラウンド。各ラウンドの前に load1 < 3.0 を最大 30 分待つゲート（60 秒間隔）。
#                     ラウンド内は oss-gemm-compare → metal_gemm_bench（examples/gemm_bench.rs・転送込み）→ gemm_f32_prepared_bench → MLX → PyTorch MPS の順。
# 環境変数: D（作業ディレクトリの絶対パス。tree/・venv-mlx/・venv-torch/ を置く）。ロックは ~/work/.fandhe-hw-lock（取得は送信側）。
# bash 3.2 互換（${VAR} 形式・setsid／timeout なし）。ホスト名は記録しない。
set -uo pipefail
MODE="${1:?mode}"; OWNER="${2:?lock owner}"; D="${D:?}"
ROOT="${D}/tree"; LOCK_DIR="${HOME}/work/.fandhe-hw-lock"; ENVD="${D}/env"; OUT="${D}/out"
ROUNDS=5; GATE_MAX=1800; GATE_STEP=60; GATE_LOAD="3.0"
[ "$(cat "${LOCK_DIR}/owner" 2>/dev/null)" = "${OWNER}" ] || { echo "LOCK-NOT-MINE: $(cat "${LOCK_DIR}/owner" 2>/dev/null)"; exit 2; }
case "${D}" in /*) ;; *) echo "D must be absolute"; exit 2 ;; esac
mkdir -p "${ENVD}" "${OUT}"
OSS_BIN="${D}/target-oss/release/oss-gemm-compare"
TR_BIN="${D}/target-metal/release/examples/metal_gemm_bench"
PR_BIN="${D}/target-metal/release/examples/gemm_f32_prepared_bench"

envinfo() { # $1=出力ファイル
  { date -u +%FT%TZ; sysctl -n machdep.cpu.brand_string; sysctl -n hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.memsize
    uname -srm; sw_vers -productVersion; sw_vers -buildVersion; sysctl -n vm.loadavg
    pmset -g therm 2>/dev/null; pmset -g ps 2>/dev/null | head -2
    echo "logged-in users: $(who | awk '{print $1}' | sort -u | tr '\n' ' ')"
    echo "top cpu:"; ps -Ao pcpu=,comm= | sort -rn | head -5
    rustc -vV; cargo -V
    "${D}/venv-mlx/bin/python" -I -c 'import sys,mlx.core as m,numpy;print("venv-mlx",sys.version.split()[0],"mlx",m.__version__,"numpy",numpy.__version__)'
    "${D}/venv-torch/bin/python" -I -c 'import sys,torch,numpy;print("venv-torch",sys.version.split()[0],"torch",torch.__version__,"numpy",numpy.__version__,"mps",torch.backends.mps.is_available())'
    echo "rev_stamp=$(cat "${ROOT}/.rev-stamp")"
    for b in "${OSS_BIN}" "${TR_BIN}" "${PR_BIN}"; do [ -f "${b}" ] && echo "bin $(shasum -a 256 "${b}" | cut -d' ' -f1) $(stat -f '%Sm' -t '%FT%T' "${b}") ${b#"${D}"/}"; done
  } > "$1" 2>&1
}
fingerprint() { ( cd "${ROOT}" && find crates scripts/bench Cargo.toml Cargo.lock rust-toolchain.toml -type f ! -path '*/target/*' -print | LC_ALL=C sort | while IFS= read -r f; do shasum -a 256 "$f"; done ) > "$1"; }

if [ "${MODE}" = "prepare" ]; then
  [ ! -e "${D}/target-oss" ] && [ ! -e "${D}/target-metal" ] || { echo "target が既にある（新規で作る規則）"; exit 2; }
  ( cd "${ROOT}" && CARGO_TARGET_DIR="${D}/target-oss" cargo build --release --locked --manifest-path scripts/bench/oss-gemm-compare/Cargo.toml ) > "${ENVD}/build-oss.log" 2>&1 || { echo "build oss failed"; exit 1; }
  ( cd "${ROOT}" && CARGO_TARGET_DIR="${D}/target-metal" cargo build --release --locked -p fandhe-ai-backend-metal --example metal_gemm_bench --example gemm_f32_prepared_bench ) > "${ENVD}/build-metal.log" 2>&1 || { echo "build metal failed"; exit 1; }
  "${D}/venv-mlx/bin/pip" freeze > "${ENVD}/pip-freeze-venv-mlx.txt" 2>&1
  "${D}/venv-torch/bin/pip" freeze > "${ENVD}/pip-freeze-venv-torch.txt" 2>&1
  envinfo "${ENVD}/env_info_prepare.txt"; fingerprint "${ENVD}/tree-fingerprint-prepare.txt"
  echo "prepare ok $(date -u +%FT%TZ)"; exit 0
fi
[ "${MODE}" = "series" ] || { echo "unknown mode"; exit 2; }
for b in "${OSS_BIN}" "${TR_BIN}" "${PR_BIN}"; do [ -x "${b}" ] || { echo "missing ${b}"; exit 2; }; done
[ ! -e "${OUT}/round1" ] || { echo "out/round1 が既にある（上書きしない）"; exit 2; }
release() { if [ "$(cat "${LOCK_DIR}/owner" 2>/dev/null)" = "${OWNER}" ]; then rm -rf "${LOCK_DIR}"; echo "lock released $(date -u +%FT%TZ)"; else echo "LOCK-LOST: 解放しない"; fi; }
trap release EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM
envinfo "${ENVD}/env_info_before.txt"; fingerprint "${ENVD}/tree-fingerprint-before.txt"
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
for i in $(seq 1 "${ROUNDS}"); do
  R="${OUT}/round${i}"; mkdir -p "${R}"
  start=$(date +%s); verdict=TIMEOUT
  while :; do
    l=$(load1); now=$(date +%s); echo "$(date -u +%FT%TZ) round${i} load1=${l}" >> "${OUT}/gate.log"
    if awk -v a="${l}" -v b="${GATE_LOAD}" 'BEGIN{exit !(a<b)}'; then verdict=PASS; break; fi
    [ $((now - start)) -ge "${GATE_MAX}" ] && break
    sleep "${GATE_STEP}"
  done
  echo "${verdict} $(date -u +%FT%TZ) load1=${l} waited_s=$(( $(date +%s) - start ))" > "${R}/gate.txt"
  uptime > "${R}/uptime_before.txt"; ps -Ao pcpu=,comm= | sort -rn | head -5 >> "${R}/uptime_before.txt"
  [ "${verdict}" = PASS ] || { echo "round${i} GATE-TIMEOUT"; continue; }
  ( cd "${ROOT}" && "${OSS_BIN}" > "${R}/oss.jsonl" 2> "${R}/oss.stderr.txt"; echo "exit=$?" > "${R}/oss.exit" )
  ( cd "${ROOT}" && "${TR_BIN}" > "${R}/metal-transfer.txt" 2>&1; echo "exit=$?" > "${R}/metal-transfer.exit" )
  ( cd "${ROOT}" && "${PR_BIN}" > "${R}/metal-prepared.txt" 2>&1; echo "exit=$?" > "${R}/metal-prepared.exit" )
  ( cd "${ROOT}" && "${D}/venv-mlx/bin/python" -I scripts/bench/gemm_bench_mlx_f32.py > "${R}/mlx.txt" 2>&1; echo "exit=$?" > "${R}/mlx.exit" )
  ( cd "${ROOT}" && "${D}/venv-torch/bin/python" -I scripts/bench/gemm_bench_torch_mps_f32.py > "${R}/mps.txt" 2>&1; echo "exit=$?" > "${R}/mps.exit" )
  uptime > "${R}/uptime_after.txt"
  echo "round${i} done $(date -u +%FT%TZ)"
done
envinfo "${ENVD}/env_info_after.txt"; fingerprint "${ENVD}/tree-fingerprint-after.txt"
echo "series done $(date -u +%FT%TZ)"
