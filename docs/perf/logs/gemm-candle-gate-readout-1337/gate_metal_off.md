未達: results/raw/results-m4max-gemm-gate-head-1c298ff-readout-off.jsonl size=1024（fandhe-ai 0.002291s > candle 0.002102s）
未達: results/raw/results-m4max-gemm-gate-head-1c298ff-readout-off.jsonl size=2048（fandhe-ai 0.010054s > candle 0.007491s）
未達: results/raw/results-m4max-gemm-gate-head-1c298ff-readout-off.jsonl size=4096（fandhe-ai 0.041511s > candle 0.023620s）
## GEMM 目標達成ゲート（#1037・device=metal）: `results/raw/results-m4max-gemm-gate-head-1c298ff-readout-off.jsonl`

| N | fandhe-ai reuse median (min–max, n) | candle fresh median (n) | candle/fandhe | GFLOP/s | 判定 |
|---|---|---|---|---|---|
| 1024 | 2.291 ms (2.041 ms–2.955 ms, n=5) | 2.102 ms (n=5) | 0.918 | 937.31 | 未達 |
| 2048 | 10.054 ms (9.233 ms–10.570 ms, n=5) | 7.491 ms (n=5) | 0.745 | 1708.78 | 未達 |
| 4096 | 41.511 ms (40.111 ms–48.830 ms, n=5) | 23.620 ms (n=5) | 0.569 | 3310.89 | 未達 |

### N=1024 要素単位検証の run 別内訳

| framework | run | fail_count/total | max_abs | max_rel | 判定 |
|---|---|---|---|---|---|
| fandhe-ai | 1 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 2 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 3 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 4 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 5 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 1 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 2 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 3 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 4 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 5 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |

### N=2048 要素単位検証の run 別内訳

| framework | run | fail_count/total | max_abs | max_rel | 判定 |
|---|---|---|---|---|---|
| fandhe-ai | 1 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 2 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 3 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 4 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 5 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 1 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 2 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 3 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 4 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 5 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |

### N=4096 要素単位検証の run 別内訳

| framework | run | fail_count/total | max_abs | max_rel | 判定 |
|---|---|---|---|---|---|
| fandhe-ai | 1 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 2 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 3 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 4 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 5 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 1 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 2 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 3 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 4 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 5 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | ok |

