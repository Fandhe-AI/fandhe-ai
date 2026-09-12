#!/bin/zsh
set -u
S=<masked-scratchpad-path>
BIN=${S}/wt-diag/scripts/bench/framework-compare/target/release/bench-fandhe
D=${S}/mac-diag; mkdir -p ${D}
echo "start $(date -u +%FT%TZ)"; uptime > ${D}/uptime_before.txt
# 1) backward 残差（5 プロセス起動 × cpu/metal × fresh/reuse）
for run in 1 2 3 4 5; do
  for dev in cpu metal; do
    for mode in fresh reuse; do
      FANDHE_DIAG_BACKWARD=1 ${BIN} --task train --device ${dev} --size 64 --mode ${mode} --phases --out ${D}/train-phases.jsonl 2> ${D}/diag-${dev}-${mode}-run${run}.err >/dev/null || echo "FAIL ${dev} ${mode} ${run}"
    done
  done
done
echo "stage1-done $(date -u +%FT%TZ)"
# 2) Metal N=1024 reuse readout legacy/borrowed 交互 × 5（--phases）
: > ${D}/readout-phases.jsonl
for run in 1 2 3 4 5; do
  for ro in legacy borrowed; do
    ${BIN} --task gemm --device metal --size 1024 --mode reuse --phases --readout ${ro} --out ${D}/readout-phases.jsonl 2>>${D}/readout.err >/dev/null || echo "FAIL readout ${ro} ${run}"
  done
  for ro in borrowed legacy; do
    ${BIN} --task gemm --device metal --size 1024 --mode fresh --readout ${ro} --out ${D}/readout-phases.jsonl 2>>${D}/readout.err >/dev/null || echo "FAIL readout fresh ${ro} ${run}"
  done
done
uptime > ${D}/uptime_after.txt
echo "done $(date -u +%FT%TZ)"
