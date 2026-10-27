#!/usr/bin/env bash
# DGX GB10 0.12.0 の 5 ラウンド（run1〜run5 を同じループで回す。0.11.0 の「run1 だけ単独」形は廃止）。イシュー #3273。草案: 実機未実行。
# 各 round の前に専有ゲート（load1 < 1.0 かつ GPU 使用率 0% が 3 サンプル連続・30 秒間隔・最大 30 分）をかける。
# load1 が数値で取れなければ GATE-INVALID、30 分超過なら GATE-TIMEOUT として系列を打ち切る。5/5 完走のときだけ採用する。
# 環境変数: ROOT OLD_ROOT EXPECT_NEW_SHA EXPECT_OLD_SHA（dgx-run-0120.sh へ引き継ぐ）。
set -uo pipefail
B="${HOME}/work/dgx-0120-logs"; mkdir -p "${B}"
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
# 旧条件腕のビルドは系列開始前に 1 回だけ行う（round 内の測定窓へビルド負荷・熱を入れない）。
# trap による Cargo.lock 復元の対象外だが、--locked なので Cargo.lock は書き換わらない（意図的）。
ROOT="${ROOT:?new root}"; OLD_ROOT="${OLD_ROOT:?old root}"; OLD_TARGET="${OLD_ROOT}/../target-old"
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
# 既存バイナリの有無では判断しない（別コミット・別版の成果物を旧腕として測らないため）。毎回 cargo に再利用判断させる（最新なら即終了）。
# -p bench-fandhe で旧腕だけをビルドする（package 指定なしだと bench-candle の macOS 用 candle-core/metal 既定 feature が Linux でビルド失敗する）。
(cd "${OLD_ROOT}/scripts/bench/framework-compare" && cargo build --release --locked -p bench-fandhe --target-dir "${OLD_TARGET}" > "${B}/build-old.log" 2>&1) || { echo "BUILD FAILED (old)"; exit 1; }
[ -x "${OLD_TARGET}/release/bench-fandhe" ] || { echo "BUILD FAILED (old: binary missing)"; exit 1; }
done_n=0
for i in 1 2 3 4 5; do
  grc=0; gate || grc=$?
  if [ "${grc}" -ne 0 ]; then echo "GATE-FAILED(rc=${grc}) before run${i}"; echo "SERIES-ABORTED completed=${done_n}/5"; exit 1; fi
  mkdir -p "${B}/run${i}"
  if ! LOGD="${B}/run${i}" bash "${HOME}/work/dgx-run-0120.sh" > "${B}/run${i}.log" 2>&1; then
    echo "run${i}: FAILED $(tail -1 "${B}/run${i}.log")"; echo "SERIES-ABORTED completed=${done_n}/5"; exit 1
  fi
  done_n=$((done_n+1)); echo "run${i}: $(tail -1 "${B}/run${i}.log")"
done
[ "${done_n}" = 5 ] || { echo "SERIES-INCOMPLETE completed=${done_n}/5"; exit 1; }
echo "ALL-DONE completed=5/5"
