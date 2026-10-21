#!/usr/bin/env bash
# イシュー #3190 再開分 GB10 CUDA セッション 3 / 段 2: B5 の TFLOPS A/B (前提ゲート (a)(b) 成立後だけ起動する)。
# 奇数 round は base -> head、偶数 round は head -> base。各プロセスの前に専有ゲート (load1 < 1.0 かつ GPU 0%。30 秒間隔・最大 30 分)。
# ビルドは計測前に済ませ、計測中はビルドしない。CARGO_TARGET_DIR は export しない。
set -u
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
unset RAYON_NUM_THREADS
unset CARGO_TARGET_DIR
R=$HOME/work/r3190-reopen
OUT=$R/out
B5=$OUT/b5
mkdir -p "$B5"
step() { echo "$(date -u +%FT%TZ) $*" >>"$OUT/steps-session3.log"; }

for arm in base head; do
  step "B5 prebuild $arm"
  (cd "$R/b5-$arm" && cargo build -p fandhe-ai-backend-cuda --release --example gemm_mma_bench >"$B5/prebuild-$arm.log" 2>&1)
  step "B5 prebuild $arm rc=$?"
done

: >"$B5/gate.log"
gate() {
  local label="$1" waited=0 load util
  while :; do
    load="$(cut -d' ' -f1 /proc/loadavg)"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if awk -v l="$load" 'BEGIN{exit !(l < 1.0)}' && [[ "$util" == "0" ]]; then
      echo "$label PASS load1=$load gpu_util=$util" >>"$B5/gate.log"
      return 0
    fi
    if (( waited >= 1800 )); then
      echo "$label FAIL(参考扱い) load1=$load gpu_util=$util" >>"$B5/gate.log"
      return 0
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

for r in 1 2 3 4 5; do
  if (( r % 2 == 1 )); then order="base head"; else order="head base"; fi
  for arm in $order; do
    gate "run$r-$arm"
    step "B5 run$r-$arm start"
    (cd "$R/b5-$arm" && cargo run -p fandhe-ai-backend-cuda --release --example gemm_mma_bench >"$B5/run$r-$arm.log" 2>&1)
    step "B5 run$r-$arm rc=$?"
  done
done
step "STAGE2 DONE"
touch "$R/stage2.done"
