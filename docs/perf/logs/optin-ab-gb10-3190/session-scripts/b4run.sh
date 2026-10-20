export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
unset RAYON_NUM_THREADS CARGO_TARGET_DIR
R=$HOME/work/r3190
cd $R/runner/docs/perf/logs/cuda-gemm-small-launch-cost-2109 || exit 1
LOG_DIR=$R/cuda-out3/b4 RUN_NSYS=1 ./orchestrate.sh > $R/cuda-out3/b4-orch.log 2>&1
echo "$(date -u +%FT%TZ) B4 orchestrate rc=$?" >> $R/cuda-out3/steps.log
touch $R/cuda3.done
