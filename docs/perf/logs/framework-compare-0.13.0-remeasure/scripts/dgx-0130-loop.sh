#!/usr/bin/env bash
# DGX GB10 0.13.0 の 5 ラウンド（run1〜run5 を同じループで回す）。計測キット（未実測）。dgx-0120-loop.sh からの派生。
# 各 round の前に専有ゲート（load1 < 1.0 かつ GPU 使用率 0% が 3 サンプル連続・30 秒間隔・最大 30 分。0.12.0 と同一）をかける。
# load1 が数値で取れなければ GATE-INVALID、30 分超過なら GATE-TIMEOUT として系列を打ち切る。5/5 完走のときだけ採用する。
# 0.12.0 版からの変更点: 単腕のため旧条件腕の事前ビルド（OLD_ROOT・target-old）を削除した。
# 環境変数: ROOT EXPECT_SHA（dgx-run-0130.sh へ引き継ぐ）。
set -uo pipefail
B="${HOME}/work/dgx-0130-logs"; mkdir -p "${B}"
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
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
# registry 計測のため patch 変数は拒否する（path patch を継承した別ソースを測らない）。
[ -z "${GEMM_GATE_PATCH_FACADE_PATH:-}" ] || { echo "GEMM_GATE_PATCH_FACADE_PATH が非空（registry 計測では拒否）"; exit 2; }
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
  if ! LOGD="${B}/run${i}" bash "${HOME}/work/dgx-run-0130.sh" > "${B}/run${i}.log" 2>&1; then
    echo "run${i}: FAILED $(tail -1 "${B}/run${i}.log")"; echo "SERIES-ABORTED completed=${done_n}/5"; exit 1
  fi
  done_n=$((done_n+1)); echo "run${i}: $(tail -1 "${B}/run${i}.log")"
done
[ "${done_n}" = 5 ] || { echo "SERIES-INCOMPLETE completed=${done_n}/5"; exit 1; }
echo "ALL-DONE completed=5/5"
