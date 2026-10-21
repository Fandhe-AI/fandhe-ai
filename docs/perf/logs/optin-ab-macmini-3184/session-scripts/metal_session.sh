#!/usr/bin/env bash
# イシュー #3184 Metal 系セッション（G4 -> G5 -> G3 -> G6 -> G7）。nohup で切り離し起動する。
# 配置: ~/work/optin-3184/session/（clone の外）。継承スクリプトは編集せず呼ぶだけ。
# 各ステップの rc を steps.log に残し、失敗しても次へ進む（判定は RULE のとおり機械的）。
# 使い方: bash metal_session.sh <g4g5|g3g6g7>（RULE 3 の 2 時間超過時の分割に対応）
set -u
export PATH=${HOME}/.cargo/bin:${PATH}
unset RAYON_NUM_THREADS CARGO_TARGET_DIR
. "$(dirname "$0")/lib.sh"
PART=${1:-}
B=${R}/before; A=${R}/after; D=${R}/diag
PATCHES=${B}/docs/perf/logs/optin-ab-macmini-3184
FC=${B}/scripts/bench/framework-compare
SHORT=$(git -C "${B}" rev-parse --short HEAD)

collect_raw() { # collect_raw <glob-fragment> <destdir>
  mkdir -p "$2"
  mv "${FC}"/results/raw/*"$1"* "$2/" 2>/dev/null
}

if [ "${PART}" = g4g5 ]; then
  # ---- G4
  outer_gate g4
  step "G4 start"
  ( cd "${FC}" && AB_PATCH_FACADE_PATH=${B}/crates/facade bash run_ab_readback_metal.sh "head-${SHORT}-3184-macmini-g4" >"${COL}/macmini/stdout-g4.log" 2>&1 )
  step "G4 AB rc=$?"
  collect_raw 3184-macmini-g4 "${COL}/readback"
  mv "${COL}/macmini/stdout-g4.log" "${COL}/readback/" 2>/dev/null
  rm -rf "${FC}"/target-ab-* "${FC}"/target-*3184* 2>/dev/null
  step "status before=$(git -C "${B}" status --porcelain | wc -l)"
  # ---- G5
  outer_gate g5
  step "G5 start"
  ( cd "${FC}" && AB_PATCH_FACADE_PATH=${B}/crates/facade bash run_ab_readout_metal.sh "head-${SHORT}-3184-macmini-g5" >"${COL}/macmini/stdout-g5.log" 2>&1 )
  step "G5 AB rc=$?"
  collect_raw 3184-macmini-g5 "${COL}/readout"
  mv "${COL}/macmini/stdout-g5.log" "${COL}/readout/" 2>/dev/null
  rm -rf "${FC}"/target-ab-* "${FC}"/target-*3184* 2>/dev/null
  step "status before=$(git -C "${B}" status --porcelain | wc -l)"
  step "METAL1 DONE"
  echo done >"${R}/metal1.done"
  exit 0
fi

if [ "${PART}" = g3g6g7 ]; then
  # ---- G3 前提ゲート
  P=${COL}/encode-only/prereq
  mkdir -p "${P}"
  export CARGO_TARGET_DIR=${R}/target-tests
  outer_gate g3
  step "G3 start"
  git -C "${A}" apply --check "${PATCHES}/on-train-forward-encode.patch" >"${COL}/macmini/patch_apply-g3.log" 2>&1 \
    && git -C "${A}" apply "${PATCHES}/on-train-forward-encode.patch" >>"${COL}/macmini/patch_apply-g3.log" 2>&1
  step "G3 patch rc=$?"
  shasum -a 256 "${PATCHES}/on-train-forward-encode.patch" >>"${COL}/macmini/patch_apply-g3.log"
  # (a)
  ( cd "${A}" && cargo test -p fandhe-ai-backend-metal --release --test train_forward_encode_only_parity -- --ignored --nocapture >"${P}/a1-after.log" 2>&1 ); step "G3 a1 rc=$?"
  ( cd "${A}" && cargo test -p fandhe-ai --release --test metal_train_forward_encode_only_bit_identity -- --ignored >"${P}/a2-after.log" 2>&1 ); step "G3 a2 rc=$?"
  # (b)
  ( cd "${A}" && cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture --test-threads=1 >"${P}/b-after.log" 2>&1 ); step "G3 b rc=$?"
  # (c) after / before 両腕
  for arm in after before; do
    if [ "${arm}" = after ]; then W=${A}; else W=${B}; fi
    for t in command_batching command_batching_bench gemm_resident_parity linear_forward_device_parity; do
      ( cd "${W}" && cargo test -p fandhe-ai-backend-metal --release --test "${t}" -- --ignored --nocapture >"${P}/c-${arm}-${t}.log" 2>&1 ); step "G3 c ${arm} ${t} rc=$?"
    done
    ( cd "${W}" && cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture --test-threads=1 >"${P}/c-${arm}-mnist_scale_train_reuse_bench.log" 2>&1 ); step "G3 c ${arm} mnist rc=$?"
    ( cd "${W}" && cargo test -p fandhe-ai --release --test device_param_store_backend_parity -- --ignored --nocapture >"${P}/c-${arm}-device_param_store_backend_parity.log" 2>&1 ); step "G3 c ${arm} dpsbp rc=$?"
  done
  unset CARGO_TARGET_DIR
  rm -rf "${R}/target-tests"
  # A/B は前提ゲートの判定後に別起動（g3ab）で行う。ここでは after patch を残さない
  git -C "${A}" checkout -- .
  step "G3 PREREQ DONE"
  echo done >"${R}/metal2.done"
  exit 0
fi

if [ "${PART}" = g3ab ]; then
  outer_gate g3ab
  step "G3 AB start"
  git -C "${A}" apply "${PATCHES}/on-train-forward-encode.patch" >>"${COL}/macmini/patch_apply-g3.log" 2>&1
  ( cd "${FC}" && AB_DEVICE=metal AB_BEFORE_FACADE_PATH=${B}/crates/facade AB_AFTER_FACADE_PATH=${A}/crates/facade \
    bash run_ab_train_forward_encode_metal.sh 3184-macmini-g3 >"${COL}/macmini/stdout-g3.log" 2>&1 )
  step "G3 AB rc=$?"
  git -C "${A}" checkout -- .
  collect_raw 3184-macmini-g3 "${COL}/encode-only"
  mv "${COL}/macmini/stdout-g3.log" "${COL}/encode-only/" 2>/dev/null
  rm -rf "${FC}"/target-ab-* "${FC}"/target-*3184* 2>/dev/null
  step "G3AB DONE"
  echo done >"${R}/metal3.done"
  exit 0
fi

if [ "${PART}" = g6g7 ]; then
  export CARGO_TARGET_DIR=${R}/target-diag
  # ---- G6
  G6=${D}/docs/perf/logs/metal-gemm-candidate-ab-2111
  outer_gate g6
  step "G6 start"
  {
    echo "hardware: Mac mini (M4, P4+E6)"
    echo "chip: $(sysctl -n machdep.cpu.brand_string)"
    echo "gpu_cores: $(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2; exit}')"
    echo "macos: $(sw_vers -productVersion)"
    echo "rustc: $(rustc --version)"
    echo "git_commit: $(git -C "${D}" rev-parse HEAD)"
    echo "date_utc: $(date -u +%FT%TZ)"
    echo "load_policy: exclusive_gate"
    echo "mtl_architecture_name: (未取得)"
  } >"${G6}/env_info.txt"
  ( cd "${G6}" && ./orchestrate.sh gate >"${COL}/macmini/stdout-g6-gate.log" 2>&1 ); step "G6 gate rc=$?"
  for n in 1 2 3 4 5; do
    ( cd "${G6}" && ./orchestrate.sh "${n}" >"${COL}/macmini/stdout-g6-run${n}.log" 2>&1 ); step "G6 run${n} rc=$?"
  done
  ( cd "${G6}" && python3 -I aggregate.py >aggregate.md 2>aggregate.err ); step "G6 aggregate rc=$?"
  mkdir -p "${COL}/steel"
  for f in "${G6}"/*; do
    case "$(basename "${f}")" in orchestrate.sh|aggregate.py|README.md|RULE.txt) ;; *) cp -R "${f}" "${COL}/steel/" ;; esac
  done
  git -C "${D}" checkout -- .
  git -C "${D}" clean -fdq -- docs/perf/logs/metal-gemm-candidate-ab-2111
  rm -rf "${R}/target-diag"
  step "status diag=$(git -C "${D}" status --porcelain | wc -l)"
  # ---- G7
  outer_gate g7
  step "G7 start"
  export CARGO_TARGET_DIR=${R}/target-diag
  ( cd "${D}" && bash docs/perf/logs/metal-tape-build-infer-phase-2114/orchestrate.sh m4max --out "${COL}/verify-cache" >"${COL}/macmini/stdout-g7.log" 2>&1 ); step "G7 rc=$?"
  ( cd "${D}" && python3 -I docs/perf/logs/metal-tape-build-infer-phase-2114/aggregate.py "${COL}/verify-cache" >"${COL}/verify-cache/aggregate.md" 2>"${COL}/verify-cache/aggregate.err" ); step "G7 aggregate rc=$?"
  rm -rf "${R}/target-diag"
  step "METAL4 DONE"
  echo done >"${R}/metal4.done"
  exit 0
fi
echo "usage: $0 <g4g5|g3g6g7|g3ab|g6g7>" >&2
exit 2
