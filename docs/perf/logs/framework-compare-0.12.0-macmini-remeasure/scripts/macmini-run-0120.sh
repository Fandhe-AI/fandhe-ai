#!/usr/bin/env bash
# Mac mini（Apple M4）0.12.0 計測の 1 round（イシュー #3273。草案: 実機未実行）。macmini-run-0110.sh からの派生。
# 変更点:
#   (1) 新腕 ROOT と旧条件腕 OLD_ROOT の 2 ツリーを扱い、旧腕の判定セルを Rust 段の直後・Python 段の前に測る
#   (2) 事前検査（機体・SHA・共有ファイル一致・腕判別 grep・bench_py.py の sha256）を fail-closed にする
#   (3) run ごとに tree.txt / tree-old.txt / rev-stamp*.txt を残し、v0.12.0 でなければ中止する
# 環境変数: ROOT=新腕の展開ツリー OLD_ROOT=旧腕の展開ツリー D=作業ディレクトリ
#           EXPECT_NEW_SHA / EXPECT_OLD_SHA=RULE.txt に固定した SHA（フル 40 桁）
# bash 3.2 互換（${VAR} 形式・mapfile なし）。計測中はビルドも別負荷も並走させない。
set -uo pipefail
validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
ROOT="${ROOT:?new repo root}"; OLD_ROOT="${OLD_ROOT:?old repo root}"; D="${D:?remeasure dir}"
validate_abs ROOT "${ROOT}"; validate_abs OLD_ROOT "${OLD_ROOT}"; validate_abs D "${D}"
NEW_SHA="${EXPECT_NEW_SHA:?RULE の新腕 SHA}"; OLD_SHA="${EXPECT_OLD_SHA:?RULE の旧腕 SHA}"
BENCH_PY_SHA256="9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b"
FC="${ROOT}/scripts/bench/framework-compare"; OFC="${OLD_ROOT}/scripts/bench/framework-compare"
OLD_TARGET="${OLD_ROOT}/../target-old"
LOGD="${1:?logdir}"; validate_abs LOGD "${LOGD}"; mkdir -p "${LOGD}"
OUT="${LOGD}/results-macmini-0.12.0.jsonl"; SKIP="${LOGD}/skipped-macmini-0.12.0.log"
PYOUT="${LOGD}/results-macmini-py-0.12.0.jsonl"; OLDOUT="${LOGD}/results-macmini-old-0.12.0.jsonl"
: > "${OUT}"; : > "${SKIP}"; : > "${PYOUT}"; : > "${OLDOUT}"
# ---- precheck（fail-closed）
[ "$(sysctl -n machdep.cpu.brand_string)" = "Apple M4" ] || { echo "機体が Apple M4 ではない（M4 Max 等の値を Mac mini として使わない）"; exit 1; }
[ "$(cat "${ROOT}/.rev-stamp")" = "${NEW_SHA}" ] || { echo "新腕 .rev-stamp が RULE と不一致"; exit 1; }
[ "$(cat "${OLD_ROOT}/.rev-stamp")" = "${OLD_SHA}" ] || { echo "旧腕 .rev-stamp が RULE と不一致"; exit 1; }
for p in Cargo.lock Cargo.toml bench-fandhe/Cargo.toml bench-candle bench-burn bench-common; do
  diff -r "${FC}/${p}" "${OFC}/${p}" > /dev/null || { echo "共有ファイル ${p} が新旧で不一致"; exit 1; }
done
ARM_RE='^[[:space:]]+let c0 = tape\.matmul_resident\('  # 行頭アンカー必須（コメント行での誤ヒットを避ける）
grep -Eq "${ARM_RE}" "${FC}/bench-fandhe/src/main.rs" || { echo "新腕に新条件の記号がない"; exit 1; }
if grep -Eq "${ARM_RE}" "${OFC}/bench-fandhe/src/main.rs"; then echo "旧腕に新条件の記号がある（腕の取り違え）"; exit 1; fi
[ "$(shasum -a 256 "${D}/py/bench_py.py" | cut -d' ' -f1)" = "${BENCH_PY_SHA256}" ] || { echo "bench_py.py の sha256 不一致"; exit 1; }
cp "${ROOT}/.rev-stamp" "${LOGD}/rev-stamp.txt"; cp "${OLD_ROOT}/.rev-stamp" "${LOGD}/rev-stamp-old.txt"
echo "start $(date -u +%FT%TZ) new=$(cut -c1-8 "${ROOT}/.rev-stamp") old=$(cut -c1-8 "${OLD_ROOT}/.rev-stamp")"; uptime | tee "${LOGD}/uptime_before.txt"
cd "${FC}"
source ./bench_fandhe_lock_restore.sh
bench_fandhe_setup_lock_restore_trap
# ビルドは系列開始前に 1 回だけ（以降は SKIP_BUILD=1）。旧腕は別 target-dir
if [[ -z "${SKIP_BUILD:-}" ]]; then
  if ! cargo build --release --locked 2>&1 | tail -5; then echo "BUILD FAILED"; exit 1; fi
  if ! (cd "${OFC}" && cargo build --release --locked --target-dir "${OLD_TARGET}" 2>&1 | tail -5); then echo "BUILD FAILED (old)"; exit 1; fi
fi
cargo tree --locked -p bench-fandhe --depth 1 | grep -E 'fandhe-ai v' | tee "${LOGD}/tree.txt"
(cd "${OFC}" && cargo tree --locked -p bench-fandhe --depth 1 | grep -E 'fandhe-ai v') | tee "${LOGD}/tree-old.txt"
for t in tree tree-old; do
  grep -q 'fandhe-ai v0\.12\.0$' "${LOGD}/${t}.txt" || { echo "registry pin 0.12.0 not resolved (${t})"; exit 1; }
done
echo "build-done $(date -u +%FT%TZ)"; uptime | tee "${LOGD}/uptime_after_build.txt"
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
# 旧条件腕（判定 13 セル × fresh・reuse = 26 行）。位置は Rust 段の直後・Python 段の前に固定。
# fresh と reuse のどちらを先に測るかは run 番号の奇偶で入れ替える（#3266 と同じ）
orun() { local task=$1 dev=$2 size=$3 mode=$4
  echo "== old ${task} ${dev} ${size} ${mode} =="
  "${OLD_TARGET}/release/bench-fandhe" --task "${task}" --device "${dev}" --size "${size}" --mode "${mode}" --out "${OLDOUT}" 2>"${LOGD}/err.tmp" \
    || echo "old ${task} ${dev} ${size} ${mode}: $(cat "${LOGD}/err.tmp")" >> "${SKIP}"; }
RUN_NO="$(basename "${LOGD}" | tr -dc '0-9')"; RUN_NO="${RUN_NO:-1}"
if [ $((RUN_NO % 2)) -eq 1 ]; then MODES="fresh reuse"; else MODES="reuse fresh"; fi
for mode in ${MODES}; do
  for n in 256 512 1024 2048; do orun gemm cpu "${n}" "${mode}"; done
  for n in 256 512 1024 2048 4096; do orun gemm metal "${n}" "${mode}"; done
  for dev in cpu metal; do orun train "${dev}" 64 "${mode}"; orun infer "${dev}" 64 "${mode}"; done
done
echo "old-done $(date -u +%FT%TZ)"
# Python 参照 FW（bench_py.py。PyTorch=venv-torch・SciPy=venv-torch・TensorFlow=venv-tf）
pyrun() { local py=$1; shift; echo "== py $* =="
  "${py}" "${D}/py/bench_py.py" "$@" --out "${PYOUT}" > /dev/null 2>"${LOGD}/err.tmp" \
    || echo "py $*: $(tail -3 "${LOGD}/err.tmp" | tr '\n' ' ')" >> "${SKIP}"; }
PT="${D}/venv-torch/bin/python"; TF="${D}/venv-tf/bin/python"
for pair in "pytorch:${PT}" "tensorflow:${TF}"; do fw="${pair%%:*}"; py="${pair#*:}"
  for n in 256 512 1024 2048; do pyrun "${py}" --framework "${fw}" --task gemm --device cpu --size "${n}"; done
  for n in 256 512 1024 2048 4096; do pyrun "${py}" --framework "${fw}" --task gemm --device metal --size "${n}"; done
  for dev in cpu metal; do pyrun "${py}" --framework "${fw}" --task train --device "${dev}" --size 64; pyrun "${py}" --framework "${fw}" --task infer --device "${dev}" --size 64; done
done
for n in 256 512 1024 2048; do pyrun "${PT}" --framework scipy --task gemm --device cpu --size "${n}"; done
pyrun "${PT}" --framework scipy --task train --device cpu --size 64; pyrun "${PT}" --framework scipy --task infer --device cpu --size 64
rm -f "${LOGD}/err.tmp"
uptime | tee "${LOGD}/uptime_after.txt"
# 期待行数: Rust 48・Python 32・旧腕 26。skipped は burn Metal GEMM N=512〜4096 の既知の記録拒否 4 件だけ
# （0.11.0 と同じ。行数スモークで再確認してから RULE に固定する）。外れたら非零で終了する
rows=$(wc -l < "${OUT}" | tr -d ' '); pyrows=$(wc -l < "${PYOUT}" | tr -d ' '); oldrows=$(wc -l < "${OLDOUT}" | tr -d ' ')
unexpected=$(grep -vcE '^bench-burn gemm metal (512|1024|2048|4096) fresh: MEASURE_ERROR: gemm checksum is degenerate' "${SKIP}")
if [ "${rows}" != 48 ] || [ "${pyrows}" != 32 ] || [ "${oldrows}" != 26 ] || [ "${unexpected}" != 0 ]; then
  echo "round FAILED rows=${rows} pyrows=${pyrows} oldrows=${oldrows} unexpected_skipped=${unexpected}"; exit 1
fi
echo "done. rows=${rows} pyrows=${pyrows} oldrows=${oldrows} skipped=$(wc -l < "${SKIP}" | tr -d ' ') $(date -u +%FT%TZ)"
