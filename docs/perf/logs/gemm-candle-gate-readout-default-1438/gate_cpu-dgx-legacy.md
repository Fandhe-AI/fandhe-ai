未達: results/raw/results-dgx-cpu-gemm-gate-head-fddca17-readout-legacy.jsonl size=512（fandhe-ai 0.002401s > candle 0.001942s）
未達: results/raw/results-dgx-cpu-gemm-gate-head-fddca17-readout-legacy.jsonl size=1024（fandhe-ai 0.006916s > candle 0.005603s）
未達: results/raw/results-dgx-cpu-gemm-gate-head-fddca17-readout-legacy.jsonl size=2048（fandhe-ai 0.035283s > candle 0.033709s）
## GEMM 目標達成ゲート（#1117・device=cpu）: `results/raw/results-dgx-cpu-gemm-gate-head-fddca17-readout-legacy.jsonl`

| N | fandhe-ai reuse median (min–max, n) | candle fresh median (n) | candle/fandhe | GFLOP/s | 判定 | fandhe-ai fresh median（参考。n） |
|---|---|---|---|---|---|---|
| 512 | 2.401 ms (1.924 ms–2.471 ms, n=5) | 1.942 ms (n=5) | 0.809 | 111.79 | 未達 | 2.251 ms (2.196 ms–2.371 ms, n=5) |
| 1024 | 6.916 ms (6.823 ms–7.157 ms, n=5) | 5.603 ms (n=5) | 0.810 | 310.51 | 未達 | 7.738 ms (7.594 ms–7.773 ms, n=5) |
| 2048 | 35.283 ms (34.447 ms–35.458 ms, n=5) | 33.709 ms (n=5) | 0.955 | 486.92 | 未達（candle 救済 2 要素） | 36.191 ms (34.493 ms–36.301 ms, n=5) |

### N=512 要素単位検証の run 別内訳

| framework | run | fail_count/total | max_abs | max_rel | bound | rescued | 判定 |
|---|---|---|---|---|---|---|---|
| fandhe-ai | 1 | 0/262144 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 2 | 0/262144 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 3 | 0/262144 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 4 | 0/262144 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 5 | 0/262144 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| candle | 1 | 0/262144 | 0.000000e+00 | 0.000000e+00 | 3.814663e-06 | 0 | ok |
| candle | 2 | 0/262144 | 0.000000e+00 | 0.000000e+00 | 3.814663e-06 | 0 | ok |
| candle | 3 | 0/262144 | 0.000000e+00 | 0.000000e+00 | 3.814663e-06 | 0 | ok |
| candle | 4 | 0/262144 | 0.000000e+00 | 0.000000e+00 | 3.814663e-06 | 0 | ok |
| candle | 5 | 0/262144 | 0.000000e+00 | 0.000000e+00 | 3.814663e-06 | 0 | ok |

### N=1024 要素単位検証の run 別内訳

| framework | run | fail_count/total | max_abs | max_rel | bound | rescued | 判定 |
|---|---|---|---|---|---|---|---|
| fandhe-ai | 1 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 2 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 3 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 4 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 5 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| candle | 1 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | 7.629375e-06 | 0 | ok |
| candle | 2 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | 7.629375e-06 | 0 | ok |
| candle | 3 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | 7.629375e-06 | 0 | ok |
| candle | 4 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | 7.629375e-06 | 0 | ok |
| candle | 5 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | 7.629375e-06 | 0 | ok |

### N=2048 要素単位検証の run 別内訳

| framework | run | fail_count/total | max_abs | max_rel | bound | rescued | 判定 |
|---|---|---|---|---|---|---|---|
| fandhe-ai | 1 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 2 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 3 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 4 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 5 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| candle | 1 | 0/4194304 | 3.814697e-05 | 3.944416e-01 | 1.525878e-05 | 2 | ok |
| candle | 2 | 0/4194304 | 3.814697e-05 | 3.944416e-01 | 1.525878e-05 | 2 | ok |
| candle | 3 | 0/4194304 | 3.814697e-05 | 3.944416e-01 | 1.525878e-05 | 2 | ok |
| candle | 4 | 0/4194304 | 3.814697e-05 | 3.944416e-01 | 1.525878e-05 | 2 | ok |
| candle | 5 | 0/4194304 | 3.814697e-05 | 3.944416e-01 | 1.525878e-05 | 2 | ok |

