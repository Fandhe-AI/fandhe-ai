未達: /private/tmp/claude-501/-Users-nancy-fandhe-library-rust-ai-library/bac57b76-f1a4-4186-aea8-8f5e06b5dc10/scratchpad/base-fddca17/scripts/bench/framework-compare/results/raw/results-m4max-gemm-gate-head-fddca17-readout-legacy.jsonl size=1024（fandhe-ai 0.004838s > candle 0.002981s）
未達: /private/tmp/claude-501/-Users-nancy-fandhe-library-rust-ai-library/bac57b76-f1a4-4186-aea8-8f5e06b5dc10/scratchpad/base-fddca17/scripts/bench/framework-compare/results/raw/results-m4max-gemm-gate-head-fddca17-readout-legacy.jsonl size=2048（fandhe-ai 0.017059s > candle 0.012329s）
未達: /private/tmp/claude-501/-Users-nancy-fandhe-library-rust-ai-library/bac57b76-f1a4-4186-aea8-8f5e06b5dc10/scratchpad/base-fddca17/scripts/bench/framework-compare/results/raw/results-m4max-gemm-gate-head-fddca17-readout-legacy.jsonl size=4096（fandhe-ai 0.058375s > candle 0.034157s）
## GEMM 目標達成ゲート（#1037・device=metal）: `/private/tmp/claude-501/-Users-nancy-fandhe-library-rust-ai-library/bac57b76-f1a4-4186-aea8-8f5e06b5dc10/scratchpad/base-fddca17/scripts/bench/framework-compare/results/raw/results-m4max-gemm-gate-head-fddca17-readout-legacy.jsonl`

| N | fandhe-ai reuse median (min–max, n) | candle fresh median (n) | candle/fandhe | GFLOP/s | 判定 |
|---|---|---|---|---|---|
| 1024 | 4.838 ms (4.373 ms–4.931 ms, n=5) | 2.981 ms (n=5) | 0.616 | 443.85 | 未達 |
| 2048 | 17.059 ms (16.771 ms–19.880 ms, n=5) | 12.329 ms (n=5) | 0.723 | 1007.08 | 未達 |
| 4096 | 58.375 ms (40.857 ms–64.843 ms, n=5) | 34.157 ms (n=5) | 0.585 | 2354.40 | 未達 |

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
| candle | 1 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | 1.525878e-05 | 0 | ok |
| candle | 2 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | 1.525878e-05 | 0 | ok |
| candle | 3 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | 1.525878e-05 | 0 | ok |
| candle | 4 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | 1.525878e-05 | 0 | ok |
| candle | 5 | 0/4194304 | 0.000000e+00 | 0.000000e+00 | 1.525878e-05 | 0 | ok |

### N=4096 要素単位検証の run 別内訳

| framework | run | fail_count/total | max_abs | max_rel | bound | rescued | 判定 |
|---|---|---|---|---|---|---|---|
| fandhe-ai | 1 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 2 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 3 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 4 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| fandhe-ai | 5 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0 | ok |
| candle | 1 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | 3.051757e-05 | 0 | ok |
| candle | 2 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | 3.051757e-05 | 0 | ok |
| candle | 3 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | 3.051757e-05 | 0 | ok |
| candle | 4 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | 3.051757e-05 | 0 | ok |
| candle | 5 | 0/16777216 | 0.000000e+00 | 0.000000e+00 | 3.051757e-05 | 0 | ok |

