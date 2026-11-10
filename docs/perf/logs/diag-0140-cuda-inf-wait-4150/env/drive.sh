#!/usr/bin/env bash
set -u
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
unset RAYON_NUM_THREADS
W=$HOME/work/diag-4150; T=$W/tree; D=$T/docs/perf/logs/diag-0140-cuda-inf-wait-4150; R=$W/RULE.txt
echo "start $(date -u +%FT%TZ)"
python3 -I -B $D/aggregate_4150.py --check-rule $R || { echo "check-rule failed"; echo "done. rc=1"; exit 1; }
FP_RULE=$(awk "/^tree_fingerprint:/{print \$2}" $R); FP_NODE=$(python3 -I -B $D/aggregate_4150.py --fingerprint $T | cut -d" " -f1)
echo "fingerprint rule=$FP_RULE node=$FP_NODE"
[ "$FP_RULE" = "$FP_NODE" ] || { echo "fingerprint mismatch"; echo "done. rc=1"; exit 1; }
export ROOT=$T EXPECT_SHA=$(awk "/^tree_sha:/{print \$2}" $R) EXPECT_EXAMPLE_SHA=$(awk "/^example_sha256:/{print \$2}" $R)
export HW_LOCK_OWNER_FILE=$HOME/work/.fandhe-hw-lock/owner HW_LOCK_HELD=1
mkdir -p $W/out
for i in 1 2 3 4 5; do echo "round $i start $(date -u +%FT%TZ)"; bash $D/run_round.sh $i $W/out/round$i; echo "round $i rc=$?"; done
OUT=$W/out/nsys bash $D/nsys_aux_4150.sh; echo "nsys rc=$? $(cat $W/out/nsys/nsys.txt)"
echo "done. rc=0 $(date -u +%FT%TZ)"
