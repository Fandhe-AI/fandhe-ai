#!/usr/bin/env bash
# イシュー #3618 G7 再測定セッション（RULE-g7-r3.txt）。Mac mini 上で nohup 切り離し起動する。
# 使い方: bash g7_r3_session.sh   （${R}/repo の HEAD が固定コミットを含む clone であること）
# lib_r2.sh は R／COL が r2 固定のため source せず、必要な関数のみ写す（bash 3.2 互換・${VAR} 表記）。
set -u
export PATH=${HOME}/.cargo/bin:${PATH}
unset RAYON_NUM_THREADS CARGO_TARGET_DIR
R=${HOME}/work/g7-3618
REPO=${R}/repo
OUT=${R}/collect-r3/verify-cache-r3
mkdir -p "${OUT}"
step() { echo "$(date -u +%FT%TZ) $*" >>"${OUT}/steps.log"; }
watch_procs() { ps -axo comm= | grep -ciE 'cargo|rustc|bench|python|gemm_'; }
outer_gate() { # outer_gate <series>
  local s=$1 i=0 l1 ok=0 p
  while [ "${i}" -le 60 ]; do
    l1=$(sysctl -n vm.loadavg | awk '{print $2}')
    p=$(watch_procs)
    if awk -v a="${l1}" 'BEGIN{exit !(a<3.0)}' && [ "${p}" = 0 ]; then ok=1; break; fi
    sleep 30; i=$((i+1))
  done
  echo "series=${s} outer_gate_ok=${ok} load1=${l1} watchlist_procs=${p} waited_s=$((i*30)) at=$(date -u +%FT%TZ)" >>"${OUT}/machine_gate.log"
  echo "series=${s} watchlist_procs=${p}" >>"${OUT}/preflight-procs.txt"
}
export CARGO_TARGET_DIR=${R}/target-r3
cd "${REPO}" || exit 1
uname -a >"${OUT}/uname.txt"
{ git rev-parse HEAD; git status --porcelain; } >"${OUT}/rev-stamp.txt"
step "start"
cargo test --release --locked -p fandhe-ai --test metal_infer_tape_build_fixedcost_diag --no-run >"${OUT}/build.log" 2>&1
rc=$?; step "build rc=${rc}"
if [ "${rc}" != 0 ]; then step "ABORT build"; exit 1; fi
T="cargo test --release --locked -p fandhe-ai --test metal_infer_tape_build_fixedcost_diag"
outer_gate g7r3-bitexact
${T} -- --ignored --exact metal_infer_decomposition_matches_public_api_bit_exact --test-threads=1 --nocapture >"${OUT}/bitexact.log" 2>&1
rc=$?; step "bitexact rc=${rc}"
if [ "${rc}" != 0 ]; then step "ABORT bitexact"; exit 1; fi
outer_gate g7r3-res
${T} -- --ignored --exact timer_resolution_probe --test-threads=1 --nocapture >"${OUT}/timer-res.log" 2>&1
step "timer-res rc=$?"
outer_gate g7r3
bash docs/perf/logs/metal-tape-build-infer-phase-2114/orchestrate.sh macmini --out "${OUT}" >"${OUT}/orchestrate.log" 2>&1
step "orchestrate rc=$?"
python3 -I docs/perf/logs/metal-tape-build-infer-phase-2114/aggregate.py "${OUT}" >"${OUT}/aggregate.md" 2>"${OUT}/aggregate.err"
step "aggregate rc=$?"
rm -rf "${R}/target-r3"
step "DONE"
