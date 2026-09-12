#!/usr/bin/env bash
# DGX 追加計測: (a) 大コア pin 付き RAYON スイープ（#1319 と同じ taskset -c 5-9,15-19）、(b) CPU gemm reuse／4096 セル
set -u
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
FC="${HOME}/work/rust-ai-library-run/scripts/bench/framework-compare"
LOGD="${HOME}/work/dgx-0.8.0-logs"; cd "${FC}"
echo "start2 $(date -u +%FT%TZ)"; uptime > "${LOGD}/uptime_before_run2.txt"
SW="${LOGD}/rayon-sweep-pinned.jsonl"; : > "${SW}"
for T in 4 8 10; do
  for run in 1 2 3; do
    for task in train infer; do
      for mode in fresh reuse; do
        RAYON_NUM_THREADS="${T}" taskset -c 5-9,15-19 ./target/release/bench-fandhe --task "${task}" --device cpu --size 64 --mode "${mode}" --out "${SW}" 2>>"${LOGD}/rayon-sweep-pinned.err" || echo "T=${T} run=${run} ${task} ${mode} FAIL"
      done
    done
  done
done
echo "pinned-sweep-done $(date -u +%FT%TZ)"
EX="${LOGD}/results-dgx-0.8.0-extra.jsonl"; : > "${EX}"
for n in 256 512 1024 2048 4096; do ./target/release/bench-fandhe --task gemm --device cpu --size "${n}" --mode reuse --out "${EX}" 2>>"${LOGD}/extra.err" || echo "reuse ${n} FAIL"; done
./target/release/bench-fandhe --task gemm --device cpu --size 4096 --mode fresh --out "${EX}" 2>>"${LOGD}/extra.err" || echo "fresh 4096 FAIL"
for bin in bench-candle bench-burn; do ./target/release/${bin} --task gemm --device cpu --size 4096 --mode fresh --out "${EX}" 2>>"${LOGD}/extra.err" || echo "${bin} 4096 FAIL"; done
uptime > "${LOGD}/uptime_after_run2.txt"
echo "done2 $(date -u +%FT%TZ) rows=$(wc -l < "${EX}")"
