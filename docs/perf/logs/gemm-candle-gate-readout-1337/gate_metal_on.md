未達: results/raw/results-m4max-gemm-gate-head-1c298ff-readout-on.jsonl size=1024（fandhe-ai 0.003570s > candle 0.002337s）
未達: results/raw/results-m4max-gemm-gate-head-1c298ff-readout-on.jsonl size=2048（fandhe-ai 0.013077s > candle 0.009276s）
未達: results/raw/results-m4max-gemm-gate-head-1c298ff-readout-on.jsonl size=4096（fandhe-ai 0.055336s > candle 0.035790s）
## GEMM 目標達成ゲート（#1037・device=metal）: `results/raw/results-m4max-gemm-gate-head-1c298ff-readout-on.jsonl`

| N | fandhe-ai reuse median (min–max, n) | candle fresh median (n) | candle/fandhe | GFLOP/s | 判定 |
|---|---|---|---|---|---|
| 1024 | 3.570 ms (2.351 ms–5.498 ms, n=5) | 2.337 ms (n=5) | 0.655 | 601.53 | 未達 |
| 2048 | 13.077 ms (10.846 ms–15.814 ms, n=5) | 9.276 ms (n=5) | 0.709 | 1313.73 | 未達 |
| 4096 | 55.336 ms (42.258 ms–83.371 ms, n=5) | 35.790 ms (n=5) | 0.647 | 2483.72 | 未達 |

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

