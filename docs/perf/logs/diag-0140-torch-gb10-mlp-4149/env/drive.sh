unset RAYON_NUM_THREADS CARGO_TARGET_DIR
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
echo "drive start $(date -u +%FT%TZ) owner=$(cat <home>/work/.fandhe-hw-lock/owner)"
for i in 1 2 3 4 5; do
  echo "round $i start $(date -u +%FT%TZ)"
  ROOT=<home>/work/rust-ai-library-run-4149 PY=<home>/work/.venv-bench/bin/python BP=<home>/work/rust-ai-library-run-4149/docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py EXPECT_SHA=faf0f6bce1ab592e7b6f82e67164e47b48452867 HW_LOCK_OWNER_FILE=<home>/work/.fandhe-hw-lock/owner HW_LOCK_HELD=1 PROFILE_STEPS=50 bash <home>/work/rust-ai-library-run-4149/docs/perf/logs/diag-0140-torch-gb10-mlp-4149/run_round.sh $i <home>/work/out-4149/round$i
  rc=$?
  echo "round $i rc=$rc gate=$(cat <home>/work/out-4149/round$i/gate.txt 2>/dev/null) $(date -u +%FT%TZ)"
  if [ $rc -ne 0 ] && head -c 4 <home>/work/out-4149/round$i/gate.txt 2>/dev/null | grep -q PASS; then echo "STOP: round $i failed after gate PASS"; echo "done. rc=99"; exit 99; fi
done
PY=<home>/work/.venv-bench/bin/python bash <home>/work/rust-ai-library-run-4149/docs/perf/logs/diag-0140-torch-gb10-mlp-4149/probe/run_probe.sh <home>/work/out-4149/env/probe.txt <home>/work/RULE-4149.txt
echo "probe rc=$?"
python3 -I <home>/work/rust-ai-library-run-4149/docs/perf/logs/diag-0140-torch-gb10-mlp-4149/aggregate_4149.py --fingerprint <home>/work/rust-ai-library-run-4149
echo "done. rc=0 $(date -u +%FT%TZ)"
