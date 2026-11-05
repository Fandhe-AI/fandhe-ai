#!/usr/bin/env bash
# DGX GB10 0.14.0 計測の 1 round（#3875）。0.14.0 の dgx-run-0130.sh からの派生（単腕のまま）。
# 0.14.0 版からの変更点:
#   (1) 版の文字列: registry ピンの検査・出力ファイル名を 0.14.0 にした。期待 SHA は EXPECT_SHA（RULE.txt に固定した計測腕）。
#   (2) 専有の確認（#3875 の所有者指示）: compute プロセスを env_info.txt に記録するだけでなく、1 件でも出たら「専有未達」で
#       この round を止める（nvidia-smi が失敗したら EXCLUSIVE-INVALID で止める）。常駐サービスの有無も env_info.txt に記録する。
#       env_info.txt の先頭行（`<GPU 名>, <driver>`）と `compute-apps:` の形式は 0.14.0 と同じで、その後ろに追記する。
#   (3) 実機ロックの確認: ~/work/.fandhe-hw-lock/owner が RULE の owner 形式でなければ止める（ロックなしの値は採らない）。
# 残した不変条件: ARM_RE が計測ツリーにあること、bench_py.py の sha256、GEMM_GATE_PATCH_FACADE_PATH の拒否、
#   cargo tree でのピン検査（計測前）、期待行数、失敗時の非零終了。
# 期待行数: stage1 112 / skipped 0 / stage2 8 / stage3 28。失敗は FAILED に記録し最後に非零で終了する。
# 環境変数: ROOT LOGD EXPECT_SHA。
set -uo pipefail
validate_abs() { case "$2" in /*) ;; *) echo "$1 は絶対パスのみ"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac; }
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
ROOT="${ROOT:?root}"; LOGD="${LOGD:?logdir}"
validate_abs ROOT "${ROOT}"; validate_abs LOGD "${LOGD}"
# registry v0.14.0 の計測に path patch を継承させない（cargo tree は patch を使わないため、残っていると同定を通った別ソースを測りうる）
[ -z "${GEMM_GATE_PATCH_FACADE_PATH:-}" ] || { echo "GEMM_GATE_PATCH_FACADE_PATH が非空（registry 計測では拒否）"; exit 2; }
SHA="${EXPECT_SHA:?}"
FC="${ROOT}/scripts/bench/framework-compare"
mkdir -p "${LOGD}"
FAILED=0
fail() { echo "FAIL: $*"; FAILED=1; }
expect_rows() { local n; n="$(wc -l < "$1" | tr -d ' ')"; [ "${n}" = "$2" ] || fail "$3 の行数 ${n}（期待 $2）"; }
[ "$(cat "${ROOT}/.rev-stamp")" = "${SHA}" ] || { echo ".rev-stamp が RULE と不一致"; exit 1; }
ARM_RE='^[[:space:]]+let c0 = tape\.matmul_resident\('
grep -Eq "${ARM_RE}" "${FC}/bench-fandhe/src/main.rs" || { echo "計測ツリーに常駐入力の記号がない（bench-fandhe が 0.13.0 から変わっている）"; exit 1; }
PY="${HOME}/work/.venv-bench/bin/python"; BP="${HOME}/work/bench_py.py"
echo "9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b  ${BP}" | sha256sum -c - > /dev/null || { echo "bench_py.py の sha256 不一致"; exit 1; }
cp "${ROOT}/.rev-stamp" "${LOGD}/rev-stamp.txt"
OWNER_F="${HOME}/work/.fandhe-hw-lock/owner"
grep -Eq '^#3875 rel-0\.14\.0 [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$' "${OWNER_F}" 2>/dev/null || { echo "実機ロックの owner が RULE の形式でない（ロックなしの値は採らない）"; exit 1; }
APPS="$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>&1)"; APPS_RC=$?
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; echo "compute-apps:"; [ -z "${APPS}" ] || printf '%s\n' "${APPS}"; echo "resident-services:"; for svc in vllm comfyui kokoro; do printf '%s=%s\n' "${svc}" "$(pgrep -fic "${svc}" || true)"; done; echo "hw-lock-owner: $(cat "${OWNER_F}")"; } > "${LOGD}/env_info.txt" 2>&1
[ "${APPS_RC}" -eq 0 ] || { echo "EXCLUSIVE-INVALID: nvidia-smi --query-compute-apps rc=${APPS_RC}"; exit 1; }
[ -z "${APPS}" ] || { echo "専有未達: compute プロセスあり（env_info.txt に記録）"; exit 1; }
cd "${FC}"
# pin 検査は計測前に行う（誤った pin で計測を終えてから fail しない）。fail() は FAILED を立てるので、ここでは即 exit する
cargo tree --locked -p bench-fandhe --depth 1 | grep -E 'fandhe-ai v' > "${LOGD}/tree.txt"
grep -q 'fandhe-ai v0\.14\.0$' "${LOGD}/tree.txt" || fail "registry pin 0.14.0 not resolved (tree)"
[ "${FAILED}" -eq 0 ] || { echo "pin 検査に失敗（計測前）"; exit 1; }
echo "start $(date -u +%FT%TZ) rev=$(cut -c1-8 "${ROOT}/.rev-stamp")"; uptime > "${LOGD}/uptime_before_all.txt"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader > "${LOGD}/gpu_util_before.txt"
rm -f results/raw/results-cuda.jsonl results/raw/skipped-cuda.log
env -u GEMM_GATE_PATCH_FACADE_PATH bash ./run_all_cuda.sh > "${LOGD}/run_all_cuda.log" 2>&1; rc=$?; echo "run_all rc=${rc}"
[ "${rc}" -eq 0 ] || fail "run_all_cuda.sh rc=${rc}"
[ -f results/raw/results-cuda.jsonl ] || { fail "results-cuda.jsonl なし"; exit 1; }
touch results/raw/skipped-cuda.log
cp results/raw/results-cuda.jsonl "${LOGD}/results-dgx-0.14.0.jsonl"; cp results/raw/skipped-cuda.log "${LOGD}/skipped-dgx-0.14.0.log"
uptime > "${LOGD}/uptime_after_all.txt"
echo "stage1-done $(date -u +%FT%TZ) rows=$(wc -l < results/raw/results-cuda.jsonl) skipped=$(wc -l < results/raw/skipped-cuda.log)"
expect_rows results/raw/results-cuda.jsonl 112 stage1
expect_rows results/raw/skipped-cuda.log 0 "stage1 skipped"
EX="${LOGD}/results-dgx-0.14.0-extra.jsonl"; : > "${EX}"
for n in 256 512 1024 2048 4096; do ./target/release/bench-fandhe --task gemm --device cpu --size "${n}" --mode reuse --out "${EX}" 2>>"${LOGD}/extra.err" || fail "reuse ${n}"; done
./target/release/bench-fandhe --task gemm --device cpu --size 4096 --mode fresh --out "${EX}" 2>>"${LOGD}/extra.err" || fail "fresh 4096"
for bin in bench-candle bench-burn; do ./target/release/${bin} --task gemm --device cpu --size 4096 --mode fresh --out "${EX}" 2>>"${LOGD}/extra.err" || fail "${bin} 4096"; done
echo "stage2-done $(date -u +%FT%TZ) rows=$(wc -l < "${EX}")"
expect_rows "${EX}" 8 stage2
# stage3: Python 参照 FW（PyTorch CPU/CUDA・TensorFlow CPU・SciPy）
PYOUT="${LOGD}/results-dgx-py-0.14.0.jsonl"; : > "${PYOUT}"
uptime > "${LOGD}/uptime_before_py.txt"
prun() { echo "== $* =="; "${PY}" "${BP}" "$@" --out "${PYOUT}" > /dev/null 2>>"${LOGD}/py.err" || fail "python $*"; }
for fw in pytorch tensorflow scipy; do
  for n in 256 512 1024 2048 4096; do prun --framework "${fw}" --task gemm --device cpu --size "${n}"; done
  prun --framework "${fw}" --task train --device cpu --size 64
  prun --framework "${fw}" --task infer --device cpu --size 64
done
for n in 256 512 1024 2048 4096; do prun --framework pytorch --task gemm --device cuda --size "${n}"; done
prun --framework pytorch --task train --device cuda --size 64
prun --framework pytorch --task infer --device cuda --size 64
uptime > "${LOGD}/uptime_after_py.txt"
echo "stage3-done $(date -u +%FT%TZ) pyrows=$(wc -l < "${PYOUT}")"
expect_rows "${PYOUT}" 28 stage3
if [ "${FAILED}" -ne 0 ]; then echo "round FAILED. $(date -u +%FT%TZ)"; exit 1; fi
echo "done. $(date -u +%FT%TZ)"
