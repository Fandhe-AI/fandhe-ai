未達: results/raw/results-m4max-cpu-gemm-gate-head-1c298ff-readout-off.jsonl size=512（fandhe-ai 0.001161s > candle 0.001056s）
未達: results/raw/results-m4max-cpu-gemm-gate-head-1c298ff-readout-off.jsonl size=2048（fandhe-ai 0.038176s > candle 0.032661s）
## GEMM 目標達成ゲート（#1117・device=cpu）: `results/raw/results-m4max-cpu-gemm-gate-head-1c298ff-readout-off.jsonl`

| N | fandhe-ai reuse median (min–max, n) | candle fresh median (n) | candle/fandhe | GFLOP/s | 判定 | fandhe-ai fresh median（参考。n） |
|---|---|---|---|---|---|---|
| 512 | 1.161 ms (913.0 us–1.303 ms, n=5) | 1.056 ms (n=5) | 0.909 | 231.22 | 未達 | 1.028 ms (956.6 us–1.075 ms, n=5) |
| 1024 | 5.024 ms (4.879 ms–5.662 ms, n=5) | 5.055 ms (n=5) | 1.006 | 427.47 | 達成 | 5.128 ms (4.633 ms–6.969 ms, n=5) |
| 2048 | 38.176 ms (35.643 ms–76.549 ms, n=5) | 32.661 ms (n=5) | 0.856 | 450.02 | 未達 | 35.676 ms (33.878 ms–40.336 ms, n=5) |

### N=512 要素単位検証の run 別内訳

| framework | run | fail_count/total | max_abs | max_rel | 判定 |
|---|---|---|---|---|---|
| fandhe-ai | 1 | 0/262144 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 2 | 0/262144 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 3 | 0/262144 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 4 | 0/262144 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 5 | 0/262144 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 1 | 0/262144 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 2 | 0/262144 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 3 | 0/262144 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 4 | 0/262144 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 5 | 0/262144 | 0.000000e+00 | 0.000000e+00 | ok |

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

