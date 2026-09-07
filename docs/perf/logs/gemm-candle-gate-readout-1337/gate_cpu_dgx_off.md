未達: results/raw/results-dgx-cpu-gemm-gate-head-1c298ff-readout-off.jsonl size=512（fandhe-ai 0.002360s > candle 0.001757s）
未達: results/raw/results-dgx-cpu-gemm-gate-head-1c298ff-readout-off.jsonl size=1024（fandhe-ai 0.007120s > candle 0.005554s）
undeterminable: results/raw/results-dgx-cpu-gemm-gate-head-1c298ff-readout-off.jsonl size=2048（要素単位検証が無効（candle#1: 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01; candle#2: 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01; candle#3: 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01; candle#4: 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01; candle#5: 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01））
## GEMM 目標達成ゲート（#1117・device=cpu）: `results/raw/results-dgx-cpu-gemm-gate-head-1c298ff-readout-off.jsonl`

| N | fandhe-ai reuse median (min–max, n) | candle fresh median (n) | candle/fandhe | GFLOP/s | 判定 | fandhe-ai fresh median（参考。n） |
|---|---|---|---|---|---|---|
| 512 | 2.360 ms (2.261 ms–2.612 ms, n=5) | 1.757 ms (n=5) | 0.745 | 113.76 | 未達 | 2.211 ms (2.188 ms–2.462 ms, n=5) |
| 1024 | 7.120 ms (6.982 ms–7.638 ms, n=5) | 5.554 ms (n=5) | 0.780 | 301.62 | 未達 | 7.749 ms (7.432 ms–8.151 ms, n=5) |
| 2048 | - | - | - | - | 判定不能: 要素単位検証が無効（candle#1: 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01; candle#2: 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01; candle#3: 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01; candle#4: 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01; candle#5: 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01） | - |

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
| candle | 1 | 2/4194304 | 3.814697e-05 | 3.944416e-01 | 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01 |
| candle | 2 | 2/4194304 | 3.814697e-05 | 3.944416e-01 | 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01 |
| candle | 3 | 2/4194304 | 3.814697e-05 | 3.944416e-01 | 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01 |
| candle | 4 | 2/4194304 | 3.814697e-05 | 3.944416e-01 | 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01 |
| candle | 5 | 2/4194304 | 3.814697e-05 | 3.944416e-01 | 要素誤差超過 fail=2/4194304, max_abs=3.814697e-05, max_rel=3.944416e-01 |

