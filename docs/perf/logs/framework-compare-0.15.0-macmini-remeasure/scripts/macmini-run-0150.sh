#!/usr/bin/env bash
# Mac mini（Apple M4）0.15.0 計測の 1 round（計測キット。道具のみ・RULE 未固定・未計測。#4177）。
# 0.14.0 の macmini-run-0140.sh（docs/perf/logs/framework-compare-0.14.0-macmini-remeasure/scripts/）の写し。
# 0.14.0 版からの変更点:
#   (1) 版の文字列: 出力ファイル名（*-0.15.0.*）・registry ピンの検査（tree.txt が `fandhe-ai v0.15.0$`）。判定式・閾値・N・期待行数・bench_py.py は不変。
#   (2) 環境変数の検査: run の最初に FANDHE_AI_* と RAYON_NUM_THREADS が 1 つも無いことを確かめ（env_guard.sh）、あれば非 0 で止める。
#       結果（なし）を ${LOGD}/env_guard.txt に残す。0.15.0 は SME・jpw1・Graph・STEEL を既定のまま測る（環境変数で既定を変えない）。
#   (3) 比較相手の venv の場所を PY_VENV_DIR で指定できる（直下の venv-torch・venv-tf。既定は D）。freeze の byte 一致は
#       macmini-0150-series.sh が系列の前に venv_preflight.sh で確かめる（本スクリプトは使うだけ）。
#   他ジョブゲート（各ラウンドの前後）は macmini-0150b-loop.sh が foreign_gate_wrapper.sh 経由で行う。本スクリプトの 1 round の外側。
# 残した不変条件: 機体が Apple M4（完全一致）、.rev-stamp、ARM_RE が計測ツリーにあること、bench_py.py の sha256、ピン検査、
#   期待行数（Rust 48・Python 32）、許容する skipped（burn Metal GEMM 512〜4096 fresh の 4 件のみ）、失敗時の非零終了。
# 環境変数: ROOT=計測ツリー D=作業ディレクトリ EXPECT_SHA=RULE.txt に固定した SHA（フル 40 桁）。任意: SKIP_BUILD=1・BUILD_ONLY=1・PY_VENV_DIR・VENV_TORCH・VENV_TF（個別指定）
# bash 3.2 互換（${VAR} 形式・mapfile なし）。計測中はビルドも別負荷も並走させない。
set -uo pipefail
validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
ROOT="${ROOT:?repo root}"; D="${D:?remeasure dir}"
validate_abs ROOT "${ROOT}"; validate_abs D "${D}"
PY_VENV_DIR="${PY_VENV_DIR:-${D}}"; validate_abs PY_VENV_DIR "${PY_VENV_DIR}"
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "${SD}/env_guard.sh"
ENVG="$(env_guard_check)" || { echo "FANDHE_AI_* か RAYON_NUM_THREADS が設定されている（既定のまま測るため停止）"; exit 3; }
SHA="${EXPECT_SHA:?RULE の SHA}"
BENCH_PY_SHA256="9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b"
FC="${ROOT}/scripts/bench/framework-compare"
LOGD="${1:?logdir}"; validate_abs LOGD "${LOGD}"; mkdir -p "${LOGD}"
OUT="${LOGD}/results-macmini-0.15.0.jsonl"; SKIP="${LOGD}/skipped-macmini-0.15.0.log"
PYOUT="${LOGD}/results-macmini-py-0.15.0.jsonl"
if [[ -z "${BUILD_ONLY:-}" ]]; then : > "${OUT}"; : > "${SKIP}"; : > "${PYOUT}"; fi  # BUILD_ONLY（事前ビルド）では結果ファイルを作らない（空の JSONL を実データと取り違えない）
# ---- precheck（fail-closed）
[ "$(sysctl -n machdep.cpu.brand_string)" = "Apple M4" ] || { echo "機体が Apple M4 ではない（M4 Max 等の値を Mac mini として使わない）"; exit 1; }
[ "$(cat "${ROOT}/.rev-stamp")" = "${SHA}" ] || { echo ".rev-stamp が RULE と不一致"; exit 1; }
ARM_RE='^[[:space:]]+let c0 = tape\.matmul_resident\('  # 行頭アンカー必須（コメント行での誤ヒットを避ける）
grep -Eq "${ARM_RE}" "${FC}/bench-fandhe/src/main.rs" || { echo "計測ツリーに常駐入力の記号がない（bench-fandhe が 0.14.0 から変わっている）"; exit 1; }
[ "$(shasum -a 256 "${D}/py/bench_py.py" | cut -d' ' -f1)" = "${BENCH_PY_SHA256}" ] || { echo "bench_py.py の sha256 不一致"; exit 1; }
cp "${ROOT}/.rev-stamp" "${LOGD}/rev-stamp.txt"
printf '%s\n' "${ENVG}" > "${LOGD}/env_guard.txt"
echo "start $(date -u +%FT%TZ) rev=$(cut -c1-8 "${ROOT}/.rev-stamp")"; uptime | tee "${LOGD}/uptime_before.txt"
cd "${FC}" || exit 1
# shellcheck source=/dev/null
source ./bench_fandhe_lock_restore.sh
bench_fandhe_setup_lock_restore_trap
# ビルドは系列開始前に 1 回だけ（以降の run は SKIP_BUILD=1）
if [[ -z "${SKIP_BUILD:-}" ]]; then
  if ! cargo build --release --locked 2>&1 | tail -5; then echo "BUILD FAILED"; exit 1; fi
fi
cargo tree --locked -p bench-fandhe --depth 1 | grep -E 'fandhe-ai v' | tee "${LOGD}/tree.txt"
grep -q 'fandhe-ai v0\.15\.0$' "${LOGD}/tree.txt" || { echo "registry pin 0.15.0 not resolved (tree)"; exit 1; }
echo "build-done $(date -u +%FT%TZ)"; uptime | tee "${LOGD}/uptime_after_build.txt"
if [[ -n "${BUILD_ONLY:-}" ]]; then echo "build-only done."; exit 0; fi
run() { local bin=$1 task=$2 dev=$3 size=$4 mode=${5:-fresh}
  echo "== ${bin} ${task} ${dev} ${size} ${mode} =="
  "./target/release/${bin}" --task "${task}" --device "${dev}" --size "${size}" --mode "${mode}" --out "${OUT}" 2>"${LOGD}/err.tmp" \
    || echo "${bin} ${task} ${dev} ${size} ${mode}: $(cat "${LOGD}/err.tmp")" >> "${SKIP}"; }
for bin in bench-fandhe bench-candle bench-burn; do
  for n in 256 512 1024 2048; do run "${bin}" gemm cpu "${n}"; done
  for n in 256 512 1024 2048 4096; do run "${bin}" gemm metal "${n}"; done
  for dev in cpu metal; do run "${bin}" train "${dev}" 64; run "${bin}" infer "${dev}" 64; done
done
for n in 256 512 1024 2048 4096; do run bench-fandhe gemm metal "${n}" reuse; done
for n in 256 512 1024 2048; do run bench-fandhe gemm cpu "${n}" reuse; done
for dev in cpu metal; do run bench-fandhe train "${dev}" 64 reuse; run bench-fandhe infer "${dev}" 64 reuse; done
echo "rust-done $(date -u +%FT%TZ)"
# Python 参照 FW（bench_py.py。PyTorch=venv-torch・SciPy=venv-torch・TensorFlow=venv-tf）
pyrun() { local py=$1; shift; echo "== py $* =="
  "${py}" "${D}/py/bench_py.py" "$@" --out "${PYOUT}" > /dev/null 2>"${LOGD}/err.tmp" \
    || echo "py $*: $(tail -3 "${LOGD}/err.tmp" | tr '\n' ' ')" >> "${SKIP}"; }
PT="${VENV_TORCH:-${PY_VENV_DIR}/venv-torch}/bin/python"; TF="${VENV_TF:-${PY_VENV_DIR}/venv-tf}/bin/python"
for pair in "pytorch:${PT}" "tensorflow:${TF}"; do fw="${pair%%:*}"; py="${pair#*:}"
  for n in 256 512 1024 2048; do pyrun "${py}" --framework "${fw}" --task gemm --device cpu --size "${n}"; done
  for n in 256 512 1024 2048 4096; do pyrun "${py}" --framework "${fw}" --task gemm --device metal --size "${n}"; done
  for dev in cpu metal; do pyrun "${py}" --framework "${fw}" --task train --device "${dev}" --size 64; pyrun "${py}" --framework "${fw}" --task infer --device "${dev}" --size 64; done
done
for n in 256 512 1024 2048; do pyrun "${PT}" --framework scipy --task gemm --device cpu --size "${n}"; done
pyrun "${PT}" --framework scipy --task train --device cpu --size 64; pyrun "${PT}" --framework scipy --task infer --device cpu --size 64
rm -f "${LOGD}/err.tmp"
uptime | tee "${LOGD}/uptime_after.txt"
# 期待行数: Rust 48・Python 32。skipped は burn Metal GEMM N=512〜4096 fresh の既知の記録拒否 4 件だけ
# （0.14.0 と同じ。RULE 固定前の dry-run の行数で再確認する）。外れたら非零で終了する
rows=$(wc -l < "${OUT}" | tr -d ' '); pyrows=$(wc -l < "${PYOUT}" | tr -d ' ')
unexpected=$(grep -vcE '^bench-burn gemm metal (512|1024|2048|4096) fresh: MEASURE_ERROR: gemm checksum is degenerate' "${SKIP}")
if [ "${rows}" != 48 ] || [ "${pyrows}" != 32 ] || [ "${unexpected}" != 0 ]; then
  echo "round FAILED rows=${rows} pyrows=${pyrows} unexpected_skipped=${unexpected}"; exit 1
fi
echo "done. rows=${rows} pyrows=${pyrows} skipped=$(wc -l < "${SKIP}" | tr -d ' ') $(date -u +%FT%TZ)"
