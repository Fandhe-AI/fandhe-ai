#!/usr/bin/env bash
# 系列 B の起動役（#3874）。実機ロックを引き継ぎ、環境の記録を取ってから macmini-0140b-loop.sh を 1 回だけ実行し、
# 終了時に（失敗しても）ロックを外す。ループ本体（macmini-0140b-loop.sh・macmini-run-0140.sh）は 0.13.0 版の版文字列だけを変えたもので、
# ロックと環境の記録は持たない。そのため 0.13.0 では手で行っていたロックの解放を、本スクリプトの trap で行う。
# 引数: $1=ロック owner（RULE.txt に書いた文字列）。環境変数: ROOT・D・EXPECT_SHA（ループと同じ）。
# ロックは取らない（RULE 固定前の行数スモークの時点から送信側が保持している）。owner が違えば何も実行しない。
# bash 3.2 互換（${VAR} 形式・setsid なし）。計測中はビルドも別負荷も並走させない。
set -uo pipefail
OWNER="${1:?lock owner}"; ROOT="${ROOT:?}"; D="${D:?}"; EXPECT_SHA="${EXPECT_SHA:?}"
LOCK_DIR="${HOME}/work/.fandhe-hw-lock"; ENVD="${D}/env-series"
[ "$(cat "${LOCK_DIR}/owner" 2>/dev/null)" = "${OWNER}" ] || { echo "LOCK-NOT-MINE: $(cat "${LOCK_DIR}/owner" 2>/dev/null)"; exit 2; }
release() { if [ "$(cat "${LOCK_DIR}/owner" 2>/dev/null)" = "${OWNER}" ]; then rm -rf "${LOCK_DIR}"; echo "lock released $(date -u +%FT%TZ)"; else echo "LOCK-LOST: 解放しない"; fi; }
trap release EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM
mkdir -p "${ENVD}"
envinfo() { # $1=出力ファイル。ホスト名は書かない（uname -n を使わない）
  { date -u +%FT%TZ; sysctl -n machdep.cpu.brand_string; sysctl -n hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.memsize
    uname -srm; sw_vers -productVersion; sw_vers -buildVersion; sysctl -n vm.loadavg
    pmset -g therm 2>/dev/null; pmset -g ps 2>/dev/null | head -2
    rustc -vV; cargo -V; "${D}/venv-torch/bin/python" --version; "${D}/venv-tf/bin/python" --version
    echo "rev_stamp=$(cat "${ROOT}/.rev-stamp")"
    echo "arm_re_count=$(grep -Ec '^[[:space:]]+let c0 = tape\.matmul_resident\(' "${ROOT}/scripts/bench/framework-compare/bench-fandhe/src/main.rs")"
    echo "bench_py_sha256=$(shasum -a 256 "${D}/py/bench_py.py" | cut -d' ' -f1)"
    echo "CARGO_TARGET_DIR=${CARGO_TARGET_DIR-unset}"; } > "$1" 2>&1
}
fingerprint() { ( cd "${ROOT}" && find scripts/bench/framework-compare -type f ! -path '*/target/*' -print | LC_ALL=C sort | while IFS= read -r f; do shasum -a 256 "$f"; done ) > "$1"; }
envinfo "${ENVD}/env_info_before.txt"; fingerprint "${ENVD}/tree-fingerprint-before.txt"
bash "${D}/macmini-0140b-loop.sh"; rc=$?
envinfo "${ENVD}/env_info_after.txt"; fingerprint "${ENVD}/tree-fingerprint-after.txt"
echo "series rc=${rc} $(date -u +%FT%TZ)"
exit "${rc}"
