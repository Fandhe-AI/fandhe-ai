#!/usr/bin/env bash
# DGX GB10 0.15.0 の 5 ラウンド（run1〜run5 を同じループで回す。#4178。計測キット: 道具のみ・RULE 未固定・未計測）。
# 0.14.0 の dgx-0140-loop.sh（docs/perf/logs/framework-compare-0.14.0-remeasure/scripts/。#3875）の写し。
# 各 round の前に専有ゲート（load1 < 1.0 かつ GPU 使用率 0% が 3 サンプル連続・30 秒間隔・最大 30 分。0.14.0 と同一。GB10 に他ジョブゲートは足さない）をかける。
# load1 が数値で取れなければ GATE-INVALID、30 分超過なら GATE-TIMEOUT として系列を打ち切る。5/5 完走のときだけ採用する。
# 0.14.0 版からの変更点:
#   (1) 版の文字列（ログ置き場 ~/work/dgx-0150-logs・呼び先 dgx-run-0150.sh・ロック owner の形式 `#4178 rel-0.15.0 <UTC>`）。
#       呼び先は本スクリプトの隣（SD）から取る。
#   (2) 環境変数の検査（env_guard.sh）: FANDHE_AI_* と RAYON_NUM_THREADS が無いこと。あれば系列を始めない（各 round 前の検査は dgx-run-0150.sh）。
#   (3) 比較相手の venv（~/work/.venv-bench）の freeze が 0.14.0 の収録と byte 一致すること（venv_preflight.sh）。不一致・不在なら測らず止まる。
# 0.14.0 から変えない点: 系列の前の専有確認（compute_apps_before_series.txt）・終了時の実機ロック解除（失敗しても外す。lock-released を出す）。
# 環境変数: ROOT EXPECT_SHA（dgx-run-0150.sh へ引き継ぐ）。任意: FREEZE_REF_DIR（venv_preflight.sh。平置きのとき）。
set -uo pipefail
B="${HOME}/work/dgx-0150-logs"; mkdir -p "${B}"
LOCK_D="${HOME}/work/.fandhe-hw-lock"
release_lock() {
  if grep -Eq '^#4178 rel-0\.15\.0 ' "${LOCK_D}/owner" 2>/dev/null; then
    rm -rf "${LOCK_D}"; echo "lock-released $(date -u +%FT%TZ)"
  else
    echo "LOCK-NOT-RELEASED $(date -u +%FT%TZ)（owner が RULE の形式でない）"
  fi; }
trap release_lock EXIT
trap 'exit 143' TERM HUP INT
gate() { local t=0 ok=0 l g
  while :; do
    l=$(awk '{print $1}' /proc/loadavg 2>/dev/null)
    g=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo "gate $(date -u +%FT%TZ) load1=${l} gpu=${g}" >> "${B}/gate.log"
    if ! printf '%s' "${l}" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || ! printf '%s' "${g}" | grep -Eq '^[0-9]+$'; then
      echo "gate $(date -u +%FT%TZ) GATE-INVALID" >> "${B}/gate.log"; return 2; fi
    if awk -v l="${l}" 'BEGIN{exit !(l<1.0)}' && [ "${g}" = 0 ]; then ok=$((ok+1)); else ok=0; fi
    [ "${ok}" -ge 3 ] && return 0
    t=$((t+30)); [ "${t}" -ge 1800 ] && return 1; sleep 30
  done; }
ROOT="${ROOT:?root}"
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "${SD}/env_guard.sh"
ENVG="$(env_guard_check)" || { echo "ENV-GUARD FAILED（FANDHE_AI_* か RAYON_NUM_THREADS が設定されている）"; echo "SERIES-ABORTED completed=0/5"; exit 3; }
echo "${ENVG}" > "${B}/env_guard_before_series.txt"
bash "${SD}/venv_preflight.sh" gb10 "${HOME}/work/.venv-bench" | tee "${B}/venv_preflight.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "VENV-PREFLIGHT FAILED"; echo "SERIES-ABORTED completed=0/5"; exit 4; }
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
# registry 計測のため patch 変数は拒否する（path patch を継承した別ソースを測らない）。
[ -z "${GEMM_GATE_PATCH_FACADE_PATH:-}" ] || { echo "GEMM_GATE_PATCH_FACADE_PATH が非空（registry 計測では拒否）"; exit 2; }
# 系列の前の専有確認（各 round 前の確認は dgx-run-0150.sh）。nvidia-smi の失敗も専有を確かめられないので止める。
PRE_APPS="$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>&1)"; PRE_RC=$?
printf '%s rc=%s\n%s\n' "$(date -u +%FT%TZ)" "${PRE_RC}" "${PRE_APPS}" > "${B}/compute_apps_before_series.txt"
[ "${PRE_RC}" -eq 0 ] || { echo "EXCLUSIVE-INVALID: nvidia-smi --query-compute-apps rc=${PRE_RC}"; echo "SERIES-ABORTED completed=0/5"; exit 1; }
[ -z "${PRE_APPS}" ] || { echo "専有未達: 系列の前に compute プロセスあり"; echo "SERIES-ABORTED completed=0/5"; exit 1; }
# ビルドは系列開始前に 1 回だけ行う（run_all_cuda.sh と同じ package・feature。測定窓・専有ゲート後にビルド負荷・熱を入れない）。
# run_all_cuda.sh 内の cargo build は最新成果物なら即終了する。
# -p 指定が必要（package 指定なしだと bench-candle の macOS 用 candle-core/metal 既定 feature が Linux でビルド失敗する）。
# --locked なので Cargo.lock は書き換わらず、ツリーは dirty にならない。lock と Cargo.toml が不整合なら build が失敗する（fail-closed）。
NFC="${ROOT}/scripts/bench/framework-compare"
(cd "${NFC}" && cargo build --release --locked -p bench-fandhe > "${B}/build.log" 2>&1 \
  && cargo build --release --locked -p bench-candle --no-default-features --features cuda >> "${B}/build.log" 2>&1 \
  && cargo build --release --locked -p bench-burn --no-default-features --features cuda >> "${B}/build.log" 2>&1) || { echo "BUILD FAILED"; exit 1; }
for bin in bench-fandhe bench-candle bench-burn; do [ -x "${NFC}/target/release/${bin}" ] || { echo "BUILD FAILED (${bin} missing)"; exit 1; }; done
done_n=0
for i in 1 2 3 4 5; do
  grc=0; gate || grc=$?
  if [ "${grc}" -ne 0 ]; then echo "GATE-FAILED(rc=${grc}) before run${i}"; echo "SERIES-ABORTED completed=${done_n}/5"; exit 1; fi
  mkdir -p "${B}/run${i}"
  if ! LOGD="${B}/run${i}" bash "${SD}/dgx-run-0150.sh" > "${B}/run${i}.log" 2>&1; then
    echo "run${i}: FAILED $(tail -1 "${B}/run${i}.log")"; echo "SERIES-ABORTED completed=${done_n}/5"; exit 1
  fi
  done_n=$((done_n+1)); echo "run${i}: $(tail -1 "${B}/run${i}.log")"
done
{ date -u +%FT%TZ; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; } > "${B}/compute_apps_after_series.txt" 2>&1
[ "${done_n}" = 5 ] || { echo "SERIES-INCOMPLETE completed=${done_n}/5"; exit 1; }
echo "ALL-DONE completed=5/5"
