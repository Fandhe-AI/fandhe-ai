#!/usr/bin/env bash
# エンジン単体の対照（スコアボード 0.15.0 の「エンジン単体の対照」カード）を v0.15.0 のソースで Mac mini で測る実行役（#4314）。
# 規則の正は同ディレクトリ親の RULE.txt。手順の正は docs/real-hardware-verification-env.md §7（zsh -lc で包む・nohup で切り離す）。
#   prepare <owner> : ロックの owner を確かめ、venv の freeze を参照と照合し、2 つの target を新規に作って release ビルドし、版を env/ へ記録する（計測しない）
#   series  <owner> : 5 ラウンド。各ラウンドの前後に他ジョブゲート（#4177 と同じ）。
#                     ラウンド内は oss-gemm-compare → metal_gemm_bench（examples/gemm_bench.rs・転送込み）→ gemm_f32_prepared_bench → MLX → PyTorch MPS の順。
# 前回（docs/perf/oss-comparison/2026-11-08-macmini/scripts/orchestrate.sh・v0.14.0）からの変更点:
#   (1) ゲート: load1 だけの待ち（60 秒間隔）を、#4177 の他ジョブゲートに置き換えた。foreign_gate_wrapper.sh（#4177 の byte コピー）を source し、
#       sha256 固定済みの foreign_job_gate.py・ab_foreign_gate.sh（ROOT=tree 内。中身は変えない）を呼ぶ。
#       各ラウンドの前: CLEAN かつ load1 < 3.0 を 30 秒間隔・最大 30 分待つ。後: 1 回確かめる。
#       前の TIMEOUT・INVALID、後の非 CLEAN では SERIES-ABORTED completed=<n>/5 を出して非 0 で終わる（部分系列で続けない。前回は TIMEOUT のラウンドを飛ばして続けていた）。
#       5/5 で ALL-DONE completed=5/5。集計（aggregate.py）は ALL-DONE の系列だけに行う（RULE）。
#       gate.txt は aggregate.py（2026-11-08 版の byte コピー）が読む形式（1 行目が PASS で始まる）を保つ。
#   (2) env guard: series の開始時に env_guard.sh（#4177 の byte コピー）で FANDHE_AI_*・RAYON_NUM_THREADS が未設定かを確かめる（既定経路を測るため）。
#   (3) venv: venv-torch は VENV_TORCH（#4177 で使った venv をそのまま）、venv-mlx は ${D}/venv-mlx。freeze は `uv pip freeze --python`。
#       prepare・series の開始時・series の終了時に FREEZE_REF_DIR の参照と byte 一致を確かめる（不一致なら開始しない／終了時は無効として非 0）。
#   (4) envinfo に SME の OS 検出（sysctl の 2 値）と GPU（Chipset Model・コア数）を記録として足した。
#   (5) series ではロックの owner を確かめたあと、どの経路で終わってもロックを外す（前回は開始前の検査で止まるとロックが残った）。
# 前回と同じ点: ハーネス 5 つの順と引数、ROUNDS=5、prepare の target 新規作成、fingerprint の定義、ハーネスの終了コードは記録のみ（判定は aggregate.py）。
# 環境変数: D（作業ディレクトリの絶対パス。tree/・venv-mlx/ を置く）・VENV_TORCH（venv-torch の絶対パス）・
#   FREEZE_REF_DIR（pip-freeze-venv-torch.txt・pip-freeze-venv-mlx.txt の参照を置いた絶対パス）。ロックは ~/work/.fandhe-hw-lock（取得は送信側）。
# 呼び出し元: Mac mini での手動起動（nohup zsh -lc）・self-test.sh（モックで分岐を検査）。
# bash 3.2 互換（${VAR} 形式・setsid／timeout なし）。ホスト名は記録しない。
set -uo pipefail
MODE="${1:?mode}"; OWNER="${2:?lock owner}"; D="${D:?}"
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${D}/tree"; LOCK_DIR="${HOME}/work/.fandhe-hw-lock"; ENVD="${D}/env"; OUT="${D}/out"
ROUNDS=5; GATE_LOAD1=3.0
VENV_TORCH="${VENV_TORCH:?}"; VENV_MLX="${D}/venv-mlx"; FREEZE_REF_DIR="${FREEZE_REF_DIR:?}"
[ "$(cat "${LOCK_DIR}/owner" 2>/dev/null)" = "${OWNER}" ] || { echo "LOCK-NOT-MINE: $(cat "${LOCK_DIR}/owner" 2>/dev/null)"; exit 2; }
for p in "${D}" "${VENV_TORCH}" "${FREEZE_REF_DIR}"; do case "${p}" in /*) ;; *) echo "D・VENV_TORCH・FREEZE_REF_DIR は絶対パス"; exit 2 ;; esac; done
mkdir -p "${ENVD}" "${OUT}"
OSS_BIN="${D}/target-oss/release/oss-gemm-compare"
TR_BIN="${D}/target-metal/release/examples/metal_gemm_bench"
PR_BIN="${D}/target-metal/release/examples/gemm_f32_prepared_bench"

envinfo() { # $1=出力ファイル
  { date -u +%FT%TZ; sysctl -n machdep.cpu.brand_string; sysctl -n hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.memsize
    uname -srm; sw_vers -productVersion; sw_vers -buildVersion; sysctl -n vm.loadavg
    echo "sme sysctl (FEAT_SME SME_F32F32):"; sysctl -n hw.optional.arm.FEAT_SME hw.optional.arm.SME_F32F32
    echo "gpu:"; system_profiler SPDisplaysDataType 2>/dev/null | grep -E 'Chipset Model|Total Number of Cores'
    pmset -g therm 2>/dev/null; pmset -g ps 2>/dev/null | head -2
    echo "logged-in users: $(who | awk '{print $1}' | sort -u | tr '\n' ' ')"
    echo "top cpu:"; ps -Ao pcpu=,comm= | sort -rn | head -5
    rustc -vV; cargo -V
    "${VENV_MLX}/bin/python" -I -c 'import sys,mlx.core as m,numpy;print("venv-mlx",sys.version.split()[0],"mlx",m.__version__,"numpy",numpy.__version__)'
    "${VENV_TORCH}/bin/python" -I -c 'import sys,torch,numpy;print("venv-torch",sys.version.split()[0],"torch",torch.__version__,"numpy",numpy.__version__,"mps",torch.backends.mps.is_available())'
    echo "rev_stamp=$(cat "${ROOT}/.rev-stamp")"
    for b in "${OSS_BIN}" "${TR_BIN}" "${PR_BIN}"; do [ -f "${b}" ] && echo "bin $(shasum -a 256 "${b}" | cut -d' ' -f1) $(stat -f '%Sm' -t '%FT%T' "${b}") ${b#"${D}"/}"; done
  } > "$1" 2>&1
}
fingerprint() { ( cd "${ROOT}" && find crates scripts/bench Cargo.toml Cargo.lock rust-toolchain.toml -type f ! -path '*/target/*' -print | LC_ALL=C sort | while IFS= read -r f; do shasum -a 256 "$f"; done ) > "$1"; }
# freeze_check <suffix>: 2 つの venv の freeze を env/pip-freeze-venv-{torch,mlx}<suffix>.txt に取り、参照と byte 一致なら 0。不一致・取得失敗は 1。
freeze_check() {
  local name venv bad=0
  for name in torch mlx; do
    if [ "${name}" = torch ]; then venv="${VENV_TORCH}"; else venv="${VENV_MLX}"; fi
    if ! uv pip freeze --python "${venv}/bin/python" > "${ENVD}/pip-freeze-venv-${name}$1.txt" 2>/dev/null; then echo "FREEZE-FAILED venv-${name}"; bad=1; continue; fi
    cmp -s "${ENVD}/pip-freeze-venv-${name}$1.txt" "${FREEZE_REF_DIR}/pip-freeze-venv-${name}.txt" || { echo "FREEZE-MISMATCH venv-${name}"; bad=1; }
  done
  return "${bad}"
}

if [ "${MODE}" = "prepare" ]; then
  [ ! -e "${D}/target-oss" ] && [ ! -e "${D}/target-metal" ] || { echo "target が既にある（新規で作る規則）"; exit 2; }
  freeze_check "" || { echo "freeze が参照と一致しない（ビルドしない）"; exit 1; }
  ( cd "${ROOT}" && CARGO_TARGET_DIR="${D}/target-oss" cargo build --release --locked --manifest-path scripts/bench/oss-gemm-compare/Cargo.toml ) > "${ENVD}/build-oss.log" 2>&1 || { echo "build oss failed"; exit 1; }
  ( cd "${ROOT}" && CARGO_TARGET_DIR="${D}/target-metal" cargo build --release --locked -p fandhe-ai-backend-metal --example metal_gemm_bench --example gemm_f32_prepared_bench ) > "${ENVD}/build-metal.log" 2>&1 || { echo "build metal failed"; exit 1; }
  envinfo "${ENVD}/env_info_prepare.txt"; fingerprint "${ENVD}/tree-fingerprint-prepare.txt"
  echo "prepare ok $(date -u +%FT%TZ)"; exit 0
fi
[ "${MODE}" = "series" ] || { echo "unknown mode"; exit 2; }
release() { if [ "$(cat "${LOCK_DIR}/owner" 2>/dev/null)" = "${OWNER}" ]; then rm -rf "${LOCK_DIR}"; echo "lock released $(date -u +%FT%TZ)"; else echo "LOCK-LOST: 解放しない"; fi; }
trap release EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM
for b in "${OSS_BIN}" "${TR_BIN}" "${PR_BIN}"; do [ -x "${b}" ] || { echo "missing ${b}"; exit 2; }; done
[ ! -e "${OUT}/round1" ] || { echo "out/round1 が既にある（上書きしない）"; exit 2; }
# shellcheck source=/dev/null
. "${SD}/env_guard.sh"
ENVG="$(env_guard_check)" || { echo "ENV-GUARD FAILED（系列を始めない）"; exit 3; }
freeze_check "-before" || { echo "FREEZE CHECK FAILED（系列を始めない）"; exit 1; }
# shellcheck source=/dev/null
. "${SD}/foreign_gate_wrapper.sh"
fgw_init "${ROOT}" "${OUT}/foreign-gate" || { echo "FOREIGN-GATE PRECHECK FAILED"; exit 1; }
envinfo "${ENVD}/env_info_before.txt"; printf '%s\n' "${ENVG}" >> "${ENVD}/env_info_before.txt"
fingerprint "${ENVD}/tree-fingerprint-before.txt"
done_n=0
for i in $(seq 1 "${ROUNDS}"); do
  R="${OUT}/round${i}"; mkdir -p "${R}"
  grc=0; fgw_before "${i}" "${GATE_LOAD1}" "${OUT}/gate.log" || grc=$?
  if [ "${grc}" -ne 0 ]; then
    if [ "${grc}" -eq 1 ]; then v=TIMEOUT; else v=INVALID; fi
    echo "${v} $(date -u +%FT%TZ) reason=${FG_REASON} load1=${FG_L1} waited_s=${FG_WAITED}" > "${R}/gate.txt"
    echo "GATE-${v} before round${i}"; echo "SERIES-ABORTED completed=${done_n}/5"; exit 1
  fi
  echo "PASS $(date -u +%FT%TZ) load1=${FG_L1} waited_s=${FG_WAITED}" > "${R}/gate.txt"
  uptime > "${R}/uptime_before.txt"; ps -Ao pcpu=,comm= | sort -rn | head -5 >> "${R}/uptime_before.txt"
  ( cd "${ROOT}" && "${OSS_BIN}" > "${R}/oss.jsonl" 2> "${R}/oss.stderr.txt"; echo "exit=$?" > "${R}/oss.exit" )
  ( cd "${ROOT}" && "${TR_BIN}" > "${R}/metal-transfer.txt" 2>&1; echo "exit=$?" > "${R}/metal-transfer.exit" )
  ( cd "${ROOT}" && "${PR_BIN}" > "${R}/metal-prepared.txt" 2>&1; echo "exit=$?" > "${R}/metal-prepared.exit" )
  ( cd "${ROOT}" && "${VENV_MLX}/bin/python" -I scripts/bench/gemm_bench_mlx_f32.py > "${R}/mlx.txt" 2>&1; echo "exit=$?" > "${R}/mlx.exit" )
  ( cd "${ROOT}" && "${VENV_TORCH}/bin/python" -I scripts/bench/gemm_bench_torch_mps_f32.py > "${R}/mps.txt" 2>&1; echo "exit=$?" > "${R}/mps.exit" )
  uptime > "${R}/uptime_after.txt"
  arc=0; fgw_after "${i}" "${OUT}/gate.log" || arc=$?
  if [ "${arc}" -ne 0 ]; then
    echo "INVALID-AFTER $(date -u +%FT%TZ) reason=$(fg_after_reason)" >> "${R}/gate.txt"
    echo "SERIES-ABORTED completed=${done_n}/5 (round${i} は無効: $(fg_after_reason))"; exit 1
  fi
  done_n=$((done_n + 1))
  echo "round${i} done $(date -u +%FT%TZ)"
done
envinfo "${ENVD}/env_info_after.txt"; fingerprint "${ENVD}/tree-fingerprint-after.txt"
freeze_check "-after" || { echo "SERIES-INVALID completed=${done_n}/5 reason=freeze-after"; exit 1; }
[ "${done_n}" = 5 ] || { echo "SERIES-INCOMPLETE completed=${done_n}/5"; exit 1; }
echo "ALL-DONE completed=5/5 $(date -u +%FT%TZ)"
