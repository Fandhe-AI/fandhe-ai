未達: results/raw/results-dgx-gemm-gate-head-1c298ff-readout-off.jsonl size=1024（fandhe-ai 0.002190s > candle 0.000924s）
undeterminable: results/raw/results-dgx-gemm-gate-head-1c298ff-readout-off.jsonl size=2048（要素単位検証が無効（candle#1: 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01; candle#2: 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01; candle#3: 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01; candle#4: 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01; candle#5: 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01））
未達: results/raw/results-dgx-gemm-gate-head-1c298ff-readout-off.jsonl size=4096（fandhe-ai 0.054438s > candle 0.053537s）
## GEMM 目標達成ゲート（#1031・device=cuda）: `results/raw/results-dgx-gemm-gate-head-1c298ff-readout-off.jsonl`

| N | fandhe-ai reuse median (min–max, n) | candle fresh median (n) | candle/fandhe | GFLOP/s | 判定 |
|---|---|---|---|---|---|
| 1024 | 2.190 ms (2.125 ms–2.307 ms, n=5) | 923.9 us (n=5) | 0.422 | 980.37 | 未達 |
| 2048 | - | - | - | - | 判定不能: 要素単位検証が無効（candle#1: 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01; candle#2: 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01; candle#3: 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01; candle#4: 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01; candle#5: 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01） |
| 4096 | 54.438 ms (52.367 ms–60.003 ms, n=5) | 53.537 ms (n=5) | 0.983 | 2524.70 | 未達 |

### N=1024 要素単位検証の run 別内訳

| framework | run | fail_count/total | max_abs | max_rel | 判定 |
|---|---|---|---|---|---|
| fandhe-ai | 1 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 2 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 3 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 4 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 5 | 0/1048576 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 1 | 0/1048576 | 1.811981e-05 | 3.398111e-01 | ok |
| candle | 2 | 0/1048576 | 1.811981e-05 | 3.398111e-01 | ok |
| candle | 3 | 0/1048576 | 1.811981e-05 | 3.398111e-01 | ok |
| candle | 4 | 0/1048576 | 1.811981e-05 | 3.398111e-01 | ok |
| candle | 5 | 0/1048576 | 1.811981e-05 | 3.398111e-01 | ok |

### N=2048 要素単位検証の run 別内訳

| framework | run | fail_count/total | max_abs | max_rel | 判定 |
|---|---|---|---|---|---|
| fandhe-ai | 1 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 2 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 3 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 4 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| fandhe-ai | 5 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | ok |
| candle | 1 | 2/4194304 | 3.623962e-05 | 2.811288e-01 | 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01 |
| candle | 2 | 2/4194304 | 3.623962e-05 | 2.811288e-01 | 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01 |
| candle | 3 | 2/4194304 | 3.623962e-05 | 2.811288e-01 | 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01 |
| candle | 4 | 2/4194304 | 3.623962e-05 | 2.811288e-01 | 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01 |
| candle | 5 | 2/4194304 | 3.623962e-05 | 2.811288e-01 | 要素誤差超過 fail=2/4194304, max_abs=3.623962e-05, max_rel=2.811288e-01 |

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

