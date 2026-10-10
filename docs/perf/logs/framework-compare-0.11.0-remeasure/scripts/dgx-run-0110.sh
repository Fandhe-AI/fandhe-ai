#!/usr/bin/env bash
# DGX GB10 再計測（registry ピン =0.11.0）: run_all_cuda.sh 全セル ＋ CPU gemm reuse/4096 ＋ Python 参照 FW（同一セッション）
#
# 失敗検知: 各段階の失敗（run_all_cuda.sh の非ゼロ終了・個別計測の失敗・期待行数との不一致）は
# FAILED に記録し、最後に非ゼロで終了する。期待行数は 0.11.0 の構成（stage1 112 行・skipped 0、
# stage2 8 行、stage3 28 行）に固定している。この検知は 0.10.0 の計測後に PR #2498 の codex P2 を受けて追加した。
# 0.11.0 の 5 ラウンド（2026-10-10・main 624d0ee4）は、この検知を含む本版で回した（gb10/run*/run.log）。
# 0.10.0 計測時の経緯（コミット 8c152036 の版で回したこと・事後確認）は `docs/perf/logs/framework-compare-0.10.0-remeasure/README.md` を参照。
set -uo pipefail
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
ROOT="${HOME}/work/rust-ai-library-run"; FC="${ROOT}/scripts/bench/framework-compare"
LOGD="${LOGD:-${HOME}/work/dgx-0110-logs}"; mkdir -p "${LOGD}"
SHA="$(cat "${ROOT}/.head-sha")"
FAILED=0
fail() { echo "FAIL: $*"; FAILED=1; }
expect_rows() { # $1=ファイル $2=期待行数 $3=段階名
  local n; n="$(wc -l < "$1" | tr -d ' ')"
  [ "${n}" = "$2" ] || fail "$3 の行数 ${n}（期待 $2）"
}
cd "${FC}"
echo "start $(date -u +%FT%TZ) sha=${SHA}"; uptime > "${LOGD}/uptime_before_all.txt"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader > "${LOGD}/gpu_util_before.txt"
# 前ラウンドの成果物を誤ってコピーしないよう、実行前に消す
rm -f results/raw/results-cuda.jsonl results/raw/skipped-cuda.log
bash ./run_all_cuda.sh > "${LOGD}/run_all_cuda.log" 2>&1; rc=$?; echo "run_all rc=${rc}"
[ "${rc}" -eq 0 ] || fail "run_all_cuda.sh rc=${rc}"
[ -f results/raw/results-cuda.jsonl ] || { fail "results-cuda.jsonl なし"; exit 1; }
touch results/raw/skipped-cuda.log
cp results/raw/results-cuda.jsonl "${LOGD}/results-dgx-0.11.0.jsonl"; cp results/raw/skipped-cuda.log "${LOGD}/skipped-dgx-0.11.0.log"
uptime > "${LOGD}/uptime_after_all.txt"
echo "stage1-done $(date -u +%FT%TZ) rows=$(wc -l < results/raw/results-cuda.jsonl) skipped=$(wc -l < results/raw/skipped-cuda.log)"
expect_rows results/raw/results-cuda.jsonl 112 stage1
expect_rows results/raw/skipped-cuda.log 0 "stage1 skipped"
EX="${LOGD}/results-dgx-0.11.0-extra.jsonl"; : > "${EX}"
for n in 256 512 1024 2048 4096; do ./target/release/bench-fandhe --task gemm --device cpu --size "${n}" --mode reuse --out "${EX}" 2>>"${LOGD}/extra.err" || fail "reuse ${n}"; done
./target/release/bench-fandhe --task gemm --device cpu --size 4096 --mode fresh --out "${EX}" 2>>"${LOGD}/extra.err" || fail "fresh 4096"
for bin in bench-candle bench-burn; do ./target/release/${bin} --task gemm --device cpu --size 4096 --mode fresh --out "${EX}" 2>>"${LOGD}/extra.err" || fail "${bin} 4096"; done
echo "stage2-done $(date -u +%FT%TZ) rows=$(wc -l < "${EX}")"
expect_rows "${EX}" 8 stage2
# run_all_cuda.sh は cargo tree を出力しないため、registry ピンの確認は事後に別途行う（gb10/tree.txt）
grep -E 'fandhe-ai v0\.11\.0$' "${LOGD}/run_all_cuda.log" | head -2 || echo "WARN: registry 0.11.0 line not found in log"
# stage3: Python 参照 FW（PyTorch CPU/CUDA・TensorFlow CPU・SciPy）
PY="${HOME}/work/.venv-bench/bin/python"; BP="${HOME}/work/bench_py.py"
PYOUT="${LOGD}/results-dgx-py-0.11.0.jsonl"; : > "${PYOUT}"
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
