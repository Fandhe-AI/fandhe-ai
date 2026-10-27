#!/usr/bin/env bash
# DGX GB10 0.12.0 計測の 1 round（イシュー #3273。草案: 実機未実行）。dgx-run-0110.sh からの派生。
# 変更点:
#   (1) ROOT（新腕）と OLD_ROOT（旧条件腕）を env で受け取り（validate_abs）、SHA は .rev-stamp から読む（.head-sha は廃止）
#   (2) stage1 の後に run ごとの cargo tree --locked を新旧で取り、v0.12.0 でなければ fail（0.11.0 の WARN から強化）
#   (3) stage4（旧腕 28 行）を stage2 の後・stage3（Python）の前に追加。旧腕は run_all_cuda.sh の target dir の外へビルドする
#   (4) nvidia-smi の GPU 名・driver・compute プロセスを env_info.txt に記録
# 期待行数: stage1 112 / skipped 0 / stage2 8 / stage3 28 / stage4 28。失敗は FAILED に記録し最後に非零で終了する。
# 環境変数: ROOT OLD_ROOT LOGD EXPECT_NEW_SHA EXPECT_OLD_SHA。
set -uo pipefail
validate_abs() { case "$2" in /*) ;; *) echo "$1 は絶対パスのみ"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac; }
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
ROOT="${ROOT:?new root}"; OLD_ROOT="${OLD_ROOT:?old root}"; LOGD="${LOGD:?logdir}"
validate_abs ROOT "${ROOT}"; validate_abs OLD_ROOT "${OLD_ROOT}"; validate_abs LOGD "${LOGD}"
# registry v0.12.0 の計測に path patch を継承させない（cargo tree は patch を使わないため、残っていると同定を通った別ソースを測りうる）
[ -z "${GEMM_GATE_PATCH_FACADE_PATH:-}" ] || { echo "GEMM_GATE_PATCH_FACADE_PATH が非空（registry 計測では拒否）"; exit 2; }
NEW_SHA="${EXPECT_NEW_SHA:?}"; OLD_SHA="${EXPECT_OLD_SHA:?}"
FC="${ROOT}/scripts/bench/framework-compare"; OFC="${OLD_ROOT}/scripts/bench/framework-compare"
OLD_TARGET="${OLD_ROOT}/../target-old"
mkdir -p "${LOGD}"
FAILED=0
fail() { echo "FAIL: $*"; FAILED=1; }
expect_rows() { local n; n="$(wc -l < "$1" | tr -d ' ')"; [ "${n}" = "$2" ] || fail "$3 の行数 ${n}（期待 $2）"; }
[ "$(cat "${ROOT}/.rev-stamp")" = "${NEW_SHA}" ] || { echo "新腕 .rev-stamp が RULE と不一致"; exit 1; }
[ "$(cat "${OLD_ROOT}/.rev-stamp")" = "${OLD_SHA}" ] || { echo "旧腕 .rev-stamp が RULE と不一致"; exit 1; }
for p in Cargo.lock Cargo.toml bench-fandhe/Cargo.toml bench-candle bench-burn bench-common; do
  diff -r "${FC}/${p}" "${OFC}/${p}" > /dev/null || { echo "共有ファイル ${p} が新旧で不一致"; exit 1; }
done
ARM_RE='^[[:space:]]+let c0 = tape\.matmul_resident\('
grep -Eq "${ARM_RE}" "${FC}/bench-fandhe/src/main.rs" || { echo "新腕に新条件の記号がない"; exit 1; }
if grep -Eq "${ARM_RE}" "${OFC}/bench-fandhe/src/main.rs"; then echo "旧腕に新条件の記号がある"; exit 1; fi
PY="${HOME}/work/.venv-bench/bin/python"; BP="${HOME}/work/bench_py.py"
echo "9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b  ${BP}" | sha256sum -c - > /dev/null || { echo "bench_py.py の sha256 不一致"; exit 1; }
cp "${ROOT}/.rev-stamp" "${LOGD}/rev-stamp.txt"; cp "${OLD_ROOT}/.rev-stamp" "${LOGD}/rev-stamp-old.txt"
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; echo "compute-apps:"; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; } > "${LOGD}/env_info.txt" 2>&1
cd "${FC}"
# pin 検査は計測前に行う（誤った pin で計測を終えてから fail しない）。fail() は FAILED を立てるので、ここでは即 exit する
cargo tree --locked -p bench-fandhe --depth 1 | grep -E 'fandhe-ai v' > "${LOGD}/tree.txt"
(cd "${OFC}" && cargo tree --locked -p bench-fandhe --depth 1 | grep -E 'fandhe-ai v') > "${LOGD}/tree-old.txt"
for t in tree tree-old; do grep -q 'fandhe-ai v0\.12\.0$' "${LOGD}/${t}.txt" || fail "registry pin 0.12.0 not resolved (${t})"; done
[ "${FAILED}" -eq 0 ] || { echo "pin 検査に失敗（計測前）"; exit 1; }
echo "start $(date -u +%FT%TZ) new=$(cut -c1-8 "${ROOT}/.rev-stamp") old=$(cut -c1-8 "${OLD_ROOT}/.rev-stamp")"; uptime > "${LOGD}/uptime_before_all.txt"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader > "${LOGD}/gpu_util_before.txt"
rm -f results/raw/results-cuda.jsonl results/raw/skipped-cuda.log
env -u GEMM_GATE_PATCH_FACADE_PATH bash ./run_all_cuda.sh > "${LOGD}/run_all_cuda.log" 2>&1; rc=$?; echo "run_all rc=${rc}"
[ "${rc}" -eq 0 ] || fail "run_all_cuda.sh rc=${rc}"
[ -f results/raw/results-cuda.jsonl ] || { fail "results-cuda.jsonl なし"; exit 1; }
touch results/raw/skipped-cuda.log
cp results/raw/results-cuda.jsonl "${LOGD}/results-dgx-0.12.0.jsonl"; cp results/raw/skipped-cuda.log "${LOGD}/skipped-dgx-0.12.0.log"
uptime > "${LOGD}/uptime_after_all.txt"
echo "stage1-done $(date -u +%FT%TZ) rows=$(wc -l < results/raw/results-cuda.jsonl) skipped=$(wc -l < results/raw/skipped-cuda.log)"
expect_rows results/raw/results-cuda.jsonl 112 stage1
expect_rows results/raw/skipped-cuda.log 0 "stage1 skipped"
EX="${LOGD}/results-dgx-0.12.0-extra.jsonl"; : > "${EX}"
for n in 256 512 1024 2048 4096; do ./target/release/bench-fandhe --task gemm --device cpu --size "${n}" --mode reuse --out "${EX}" 2>>"${LOGD}/extra.err" || fail "reuse ${n}"; done
./target/release/bench-fandhe --task gemm --device cpu --size 4096 --mode fresh --out "${EX}" 2>>"${LOGD}/extra.err" || fail "fresh 4096"
for bin in bench-candle bench-burn; do ./target/release/${bin} --task gemm --device cpu --size 4096 --mode fresh --out "${EX}" 2>>"${LOGD}/extra.err" || fail "${bin} 4096"; done
echo "stage2-done $(date -u +%FT%TZ) rows=$(wc -l < "${EX}")"
expect_rows "${EX}" 8 stage2
# stage4: 旧条件腕（判定 14 セル × fresh・reuse = 28 行）。段の位置は stage3 の前（新腕と位置が違う偏りは RULE に明記）。
# 先後は run 番号の奇偶で fresh/reuse を入れ替える
# 旧腕バイナリは系列開始前に dgx-0120-loop.sh がビルドする。測定窓内ではビルドせず、無ければ fail（ビルド負荷・熱の混入防止）
[ -x "${OLD_TARGET}/release/bench-fandhe" ] || { fail "旧腕バイナリなし（dgx-0120-loop.sh の事前ビルドが必要）"; echo "round FAILED."; exit 1; }
OLDOUT="${LOGD}/results-dgx-old-0.12.0.jsonl"; : > "${OLDOUT}"
RUN_NO="$(basename "${LOGD}" | tr -dc '0-9')"; RUN_NO="${RUN_NO:-1}"
if [ $((RUN_NO % 2)) -eq 1 ]; then MODES="fresh reuse"; else MODES="reuse fresh"; fi
orun() { echo "== old $* =="; "${OLD_TARGET}/release/bench-fandhe" --task "$1" --device "$2" --size "$3" --mode "$4" --out "${OLDOUT}" 2>>"${LOGD}/old.err" || fail "old $*"; }
for mode in ${MODES}; do
  for n in 256 512 1024 2048 4096; do orun gemm cpu "${n}" "${mode}"; done
  for n in 256 512 1024 2048 4096; do orun gemm cuda "${n}" "${mode}"; done
  for dev in cpu cuda; do orun train "${dev}" 64 "${mode}"; orun infer "${dev}" 64 "${mode}"; done
done
echo "stage4-done $(date -u +%FT%TZ) rows=$(wc -l < "${OLDOUT}")"
expect_rows "${OLDOUT}" 28 stage4
# stage3: Python 参照 FW（PyTorch CPU/CUDA・TensorFlow CPU・SciPy）
PYOUT="${LOGD}/results-dgx-py-0.12.0.jsonl"; : > "${PYOUT}"
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
