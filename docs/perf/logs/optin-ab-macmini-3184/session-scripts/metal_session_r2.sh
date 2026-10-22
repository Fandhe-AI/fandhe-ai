#!/usr/bin/env bash
# イシュー #3184 再開分 Metal 系セッション（RULE-reopen.txt 3〜8）。nohup で切り離し起動する。
# 使い方: bash metal_session_r2.sh <g4g5|g3prereq|g3ab|g6|g7>
#   g4g5: G4 -> G5 / g3prereq: G3 前提ゲート (a)(b)(c)（新 rev で取り直す）/ g3ab: G3 A/B /
#   g6: p0_device_attributes -> run ごとの外側クールダウン -> G6 / g7: G7（macmini・H4(c)）。
# 各ステップの rc を steps.log に残し、失敗しても次へ進む（判定は RULE のとおり機械的）。
set -u
export PATH=${HOME}/.cargo/bin:${PATH}
unset RAYON_NUM_THREADS CARGO_TARGET_DIR
. "$(dirname "$0")/lib_r2.sh"
PART=${1:-}
B=${R}/before; A=${R}/after; D=${R}/diag
PATCHES=${B}/docs/perf/logs/optin-ab-macmini-3184
FC=${B}/scripts/bench/framework-compare
SHORT=$(git -C "${B}" rev-parse --short HEAD)
collect_raw() { # collect_raw <glob-fragment> <destdir>
  mkdir -p "$2"
  mv "${FC}"/results/raw/*"$1"* "$2/" 2>/dev/null
}

case "${PART}" in
g4g5)
  outer_gate g4r2; step "G4 start"
  ( cd "${FC}" && AB_PATCH_FACADE_PATH=${B}/crates/facade bash run_ab_readback_metal.sh "head-${SHORT}-3184-macmini-r2-g4" >"${M}/stdout-g4.log" 2>&1 )
  step "G4 AB rc=$?"
  collect_raw 3184-macmini-r2-g4 "${COL}/readback-r2"; mv "${M}/stdout-g4.log" "${COL}/readback-r2/" 2>/dev/null
  rm -rf "${FC}"/target-ab-* "${FC}"/target-*3184* 2>/dev/null
  outer_gate g5r2; step "G5 start"
  ( cd "${FC}" && AB_PATCH_FACADE_PATH=${B}/crates/facade bash run_ab_readout_metal.sh "head-${SHORT}-3184-macmini-r2-g5" >"${M}/stdout-g5.log" 2>&1 )
  step "G5 AB rc=$?"
  collect_raw 3184-macmini-r2-g5 "${COL}/readout-r2"; mv "${M}/stdout-g5.log" "${COL}/readout-r2/" 2>/dev/null
  rm -rf "${FC}"/target-ab-* "${FC}"/target-*3184* 2>/dev/null
  step "status before=$(git -C "${B}" status --porcelain | wc -l)"
  step "METAL-R2 G4G5 DONE"; echo done >"${R}/metal-r2-g4g5.done"
  ;;
g3prereq)
  P=${COL}/encode-only-r2/prereq
  mkdir -p "${P}"
  export CARGO_TARGET_DIR=${R}/target-tests
  outer_gate g3r2; step "G3 prereq start"
  git -C "${A}" checkout -- .
  git -C "${A}" apply --check "${PATCHES}/on-train-forward-encode.patch" >"${M}/patch_apply-g3.log" 2>&1 \
    && git -C "${A}" apply "${PATCHES}/on-train-forward-encode.patch" >>"${M}/patch_apply-g3.log" 2>&1
  step "G3 patch rc=$?"
  shasum -a 256 "${PATCHES}/on-train-forward-encode.patch" >>"${M}/patch_apply-g3.log"
  ( cd "${A}" && cargo test -p fandhe-ai-backend-metal --release --test train_forward_encode_only_parity -- --ignored --nocapture >"${P}/a1-after.log" 2>&1 ); step "G3 a1 rc=$?"
  ( cd "${A}" && cargo test -p fandhe-ai --release --test metal_train_forward_encode_only_bit_identity -- --ignored >"${P}/a2-after.log" 2>&1 ); step "G3 a2 rc=$?"
  ( cd "${A}" && cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture --test-threads=1 >"${P}/b-after.log" 2>&1 ); step "G3 b rc=$?"
  for arm in after before; do
    if [ "${arm}" = after ]; then W=${A}; else W=${B}; fi
    for t in command_batching command_batching_bench gemm_resident_parity linear_forward_device_parity; do
      ( cd "${W}" && cargo test -p fandhe-ai-backend-metal --release --test "${t}" -- --ignored --nocapture >"${P}/c-${arm}-${t}.log" 2>&1 ); step "G3 c ${arm} ${t} rc=$?"
    done
    ( cd "${W}" && cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture --test-threads=1 >"${P}/c-${arm}-mnist_scale_train_reuse_bench.log" 2>&1 ); step "G3 c ${arm} mnist rc=$?"
    ( cd "${W}" && cargo test -p fandhe-ai --release --test device_param_store_backend_parity -- --ignored --nocapture >"${P}/c-${arm}-device_param_store_backend_parity.log" 2>&1 ); step "G3 c ${arm} dpsbp rc=$?"
  done
  unset CARGO_TARGET_DIR; rm -rf "${R}/target-tests"
  git -C "${A}" checkout -- .
  step "G3 PREREQ DONE"; echo done >"${R}/metal-r2-g3prereq.done"
  ;;
g3ab)
  outer_gate g3abr2; step "G3 AB start"
  git -C "${A}" checkout -- .
  git -C "${A}" apply "${PATCHES}/on-train-forward-encode.patch" >>"${M}/patch_apply-g3.log" 2>&1
  ( cd "${FC}" && AB_DEVICE=metal AB_BEFORE_FACADE_PATH=${B}/crates/facade AB_AFTER_FACADE_PATH=${A}/crates/facade \
    bash run_ab_train_forward_encode_metal.sh 3184-macmini-r2-g3 >"${M}/stdout-g3.log" 2>&1 )
  step "G3 AB rc=$?"
  git -C "${A}" checkout -- .
  collect_raw 3184-macmini-r2-g3 "${COL}/encode-only-r2"; mv "${M}/stdout-g3.log" "${COL}/encode-only-r2/" 2>/dev/null
  rm -rf "${FC}"/target-ab-* "${FC}"/target-*3184* 2>/dev/null
  step "G3AB DONE"; echo done >"${R}/metal-r2-g3ab.done"
  ;;
g6)
  export CARGO_TARGET_DIR=${R}/target-diag
  G6=${D}/docs/perf/logs/metal-gemm-candidate-ab-2111
  outer_gate g6r2; step "G6 start"
  # mtl_architecture_name は既存 #[ignore] テスト p0_device_attributes の出力から転記する（コード追加なし）
  ( cd "${D}" && cargo test -p fandhe-ai-backend-metal --release --lib -- --ignored --exact typed_bf16_probe_diag_tests::p0_device_attributes --nocapture >"${M}/stdout-g6-p0.log" 2>&1 ); step "G6 p0 rc=$?"
  ARCH=$(sed -n 's/.*device_architecture=\([^ ]*\).*/\1/p' "${M}/stdout-g6-p0.log" | head -1)
  {
    echo "hardware: Mac mini (M4, P4+E6)"
    echo "chip: $(sysctl -n machdep.cpu.brand_string)"
    echo "gpu_cores: $(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2; exit}')"
    echo "macos: $(sw_vers -productVersion)"
    echo "rustc: $(rustc --version)"
    echo "git_commit: $(git -C "${D}" rev-parse HEAD)"
    echo "date_utc: $(date -u +%FT%TZ)"
    echo "load_policy: exclusive_gate"
    echo "mtl_architecture_name: ${ARCH:-(未取得)}"
  } >"${G6}/env_info.txt"
  ( cd "${G6}" && ./orchestrate.sh gate >"${M}/stdout-g6-gate.log" 2>&1 ); step "G6 gate rc=$?"
  mkdir -p "${COL}/steel-r2"
  for n in 1 2 3 4 5; do
    cooldown_wait "${COL}/steel-r2/cooldown.log" "${n}"
    ( cd "${G6}" && ./orchestrate.sh "${n}" >"${M}/stdout-g6-run${n}.log" 2>&1 ); step "G6 run${n} rc=$?"
  done
  ( cd "${G6}" && python3 -I aggregate.py >aggregate.md 2>aggregate.err ); step "G6 aggregate rc=$?"
  for f in "${G6}"/*; do
    case "$(basename "${f}")" in orchestrate.sh|aggregate.py|README.md|RULE.txt) ;; *) cp -R "${f}" "${COL}/steel-r2/" ;; esac
  done
  git -C "${D}" checkout -- .
  git -C "${D}" clean -fdq -- docs/perf/logs/metal-gemm-candidate-ab-2111
  rm -rf "${R}/target-diag"
  step "status diag=$(git -C "${D}" status --porcelain | wc -l)"
  step "G6 DONE"; echo done >"${R}/metal-r2-g6.done"
  ;;
g7)
  export CARGO_TARGET_DIR=${R}/target-diag
  outer_gate g7r2; step "G7 start"
  mkdir -p "${COL}/verify-cache-r2"
  # 集計器の正値検査は変えない。median_s=0 で fail-closed したら H1〜H3 は未確定と書く（推定で埋めない）
  ( cd "${D}" && bash docs/perf/logs/metal-tape-build-infer-phase-2114/orchestrate.sh macmini --out "${COL}/verify-cache-r2" >"${M}/stdout-g7.log" 2>&1 ); step "G7 rc=$?"
  ( cd "${D}" && python3 -I docs/perf/logs/metal-tape-build-infer-phase-2114/aggregate.py "${COL}/verify-cache-r2" >"${COL}/verify-cache-r2/aggregate.md" 2>"${COL}/verify-cache-r2/aggregate.err" ); step "G7 aggregate rc=$?"
  # H4(c): infer の固定費（wall と GPU busy の比較）。出力は手動で README／doc へ転記する
  outer_gate g7h4c
  ( cd "${D}" && cargo test -p fandhe-ai-backend-metal --release --lib -- --ignored --exact infer_fixed_cost_diag_tests::infer_fixed_cost_diag_wall_vs_gpu_busy --nocapture >"${COL}/verify-cache-r2/h4c.log" 2>&1 ); step "G7 H4c rc=$?"
  rm -rf "${R}/target-diag"
  step "G7 DONE"; echo done >"${R}/metal-r2-g7.done"
  ;;
*)
  echo "usage: $0 <g4g5|g3prereq|g3ab|g6|g7>" >&2
  exit 2
  ;;
esac
