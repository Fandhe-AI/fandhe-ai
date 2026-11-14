#!/usr/bin/env bash
# 系列 B の起動役（#4177）。実機ロックを引き継ぎ、事前検査と環境の記録を取ってから macmini-0150b-loop.sh を 1 回だけ実行し、
# 終了時に（失敗しても）ロックを外す。計測キット（道具のみ・RULE 未固定・未計測）。0.14.0 の macmini-0140-series.sh の写し。
# 0.14.0 版からの変更点:
#   (1) ロック owner の形式検査: `#4177 rel-0.15.0 <UTC>`（0.14.0 は文字列一致のみ）。
#   (2) 環境変数の検査（env_guard.sh）: FANDHE_AI_* と RAYON_NUM_THREADS が無いこと。結果を env_info に残す。あれば測らず止まる。
#   (3) 比較相手の venv の freeze が 0.14.0 の収録と byte 一致すること（venv_preflight.sh）。不一致・不在なら測らず止まる。
#       venv の場所は PY_VENV_DIR（直下の venv-torch・venv-tf。既定は D）。
#   (4) env_info に他ジョブゲートの 2 ファイルの sha256 を記録する（RULE が固定する値との照合用）。
# 引数: $1=ロック owner（RULE.txt に書いた文字列）。環境変数: ROOT・D・EXPECT_SHA（ループと同じ）・任意 PY_VENV_DIR・FREEZE_REF_DIR。
# ロックは取らない（RULE 固定前の dry-run の時点から送信側が保持している）。owner が違えば何も実行しない。
# bash 3.2 互換（${VAR} 形式・setsid なし）。計測中はビルドも別負荷も並走させない。
set -uo pipefail
OWNER="${1:?lock owner}"; ROOT="${ROOT:?}"; D="${D:?}"; EXPECT_SHA="${EXPECT_SHA:?}"
PY_VENV_DIR="${PY_VENV_DIR:-${D}}"; export PY_VENV_DIR   # VENV_TORCH・VENV_TF（任意）で個別指定もできる（export が必要）
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_DIR="${HOME}/work/.fandhe-hw-lock"; ENVD="${D}/env-series"
printf '%s' "${OWNER}" | grep -Eq '^#4177 rel-0\.15\.0 [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$' || { echo "OWNER-FORMAT-INVALID（#4177 rel-0.15.0 <UTC> の形式）"; exit 2; }
[ "$(cat "${LOCK_DIR}/owner" 2>/dev/null)" = "${OWNER}" ] || { echo "LOCK-NOT-MINE: $(cat "${LOCK_DIR}/owner" 2>/dev/null)"; exit 2; }
# shellcheck disable=SC2329  # trap から呼ばれる
release() { if [ "$(cat "${LOCK_DIR}/owner" 2>/dev/null)" = "${OWNER}" ]; then rm -rf "${LOCK_DIR}"; echo "lock released $(date -u +%FT%TZ)"; else echo "LOCK-LOST: 解放しない"; fi; }
trap release EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM
# shellcheck source=/dev/null
. "${SD}/env_guard.sh"
ENVG="$(env_guard_check)" || { echo "ENV-GUARD FAILED（FANDHE_AI_* か RAYON_NUM_THREADS が設定されている）"; exit 3; }
# 事前検査: venv の freeze（0.14.0 の収録と byte 一致）。不一致なら測らない
bash "${SD}/venv_preflight.sh" macmini "${PY_VENV_DIR}" || { echo "VENV-PREFLIGHT FAILED"; exit 4; }
mkdir -p "${ENVD}"
sha256_of() { if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
envinfo() { # $1=出力ファイル。ホスト名は書かない（uname -n を使わない）
  { date -u +%FT%TZ; sysctl -n machdep.cpu.brand_string; sysctl -n hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.memsize
    uname -srm; sw_vers -productVersion; sw_vers -buildVersion; sysctl -n vm.loadavg
    pmset -g therm 2>/dev/null; pmset -g ps 2>/dev/null | head -2
    rustc -vV; cargo -V; "${VENV_TORCH:-${PY_VENV_DIR}/venv-torch}/bin/python" --version; "${VENV_TF:-${PY_VENV_DIR}/venv-tf}/bin/python" --version
    echo "rev_stamp=$(cat "${ROOT}/.rev-stamp")"
    echo "arm_re_count=$(grep -Ec '^[[:space:]]+let c0 = tape\.matmul_resident\(' "${ROOT}/scripts/bench/framework-compare/bench-fandhe/src/main.rs")"
    echo "bench_py_sha256=$(sha256_of "${D}/py/bench_py.py")"
    echo "foreign_job_gate_sha256=$(sha256_of "${ROOT}/docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py")"
    echo "ab_foreign_gate_sha256=$(sha256_of "${ROOT}/scripts/bench/framework-compare/ab_foreign_gate.sh")"
    printf '%s\n' "${ENVG}"
    echo "CARGO_TARGET_DIR=${CARGO_TARGET_DIR-unset}"; } > "$1" 2>&1
}
fingerprint() { ( cd "${ROOT}" && find scripts/bench/framework-compare -type f ! -path '*/target/*' -print | LC_ALL=C sort | while IFS= read -r f; do shasum -a 256 "$f"; done ) > "$1"; }
envinfo "${ENVD}/env_info_before.txt"; fingerprint "${ENVD}/tree-fingerprint-before.txt"
bash "${SD}/macmini-0150b-loop.sh"; rc=$?
envinfo "${ENVD}/env_info_after.txt"; fingerprint "${ENVD}/tree-fingerprint-after.txt"
echo "series rc=${rc} $(date -u +%FT%TZ)"
exit "${rc}"
