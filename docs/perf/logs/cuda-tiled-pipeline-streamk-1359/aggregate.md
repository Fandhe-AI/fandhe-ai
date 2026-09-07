# n_runs=5

## ゲート C（64x64 streamk_over_pipeline3。5 回中央値）

| N | 値（5 run） | 中央値 | 判定基準 | 結果 |
|---|---|---|---|---|
| 1024 | 1.0169, 1.0271, 1.0189, 1.0295, 1.0350 | 1.0271 | >= 1.05 | FAIL |
| 2048 | 1.0157, 0.9355, 0.9050, 1.0145, 0.9395 | 0.9395 | >= 1.0 | FAIL |
| 4096 | 0.9799, 0.9464, 0.9823, 0.9368, 0.9613 | 0.9613 | 参考 | 参考（判定に使わない） |

## ゲート D（64x64.streamk_gpu_only_tflops / 128x64.pipeline3_gpu_only_tflops。5 回中央値）

| N | 値（5 run） | 中央値 | 判定基準 | 結果 |
|---|---|---|---|---|
| 1024 | 0.9735, 0.9801, 0.9701, 0.9819, 0.9829 | 0.9801 | >= 1.0 | FAIL |
| 2048 | 1.2295, 0.8394, 0.8132, 0.9015, 0.8377 | 0.8394 | >= 1.0 | FAIL |
| 4096 | 0.7027, 0.7194, 0.7486, 0.7177, 0.7295 | 0.7194 | 参考 | 参考（判定に使わない） |

## 参考: 各 TFLOPS 列の中央値

| N | tile | 列 | 中央値 |
|---|---|---|---|
| 1024 | 64x64 | pipeline3_gpu_only_tflops | 11.1875 |
| 1024 | 64x64 | persistent_gpu_only_tflops | 11.2522 |
| 1024 | 64x64 | streamk_gpu_only_tflops | 11.4705 |
| 1024 | 128x64 | pipeline3_gpu_only_tflops | 11.7107 |
| 1024 | 128x64 | persistent_gpu_only_tflops | 11.7993 |
| 2048 | 64x64 | pipeline3_gpu_only_tflops | 12.9367 |
| 2048 | 64x64 | persistent_gpu_only_tflops | 12.9651 |
| 2048 | 64x64 | streamk_gpu_only_tflops | 12.1509 |
| 2048 | 128x64 | pipeline3_gpu_only_tflops | 14.4284 |
| 2048 | 128x64 | persistent_gpu_only_tflops | 14.6118 |
| 4096 | 64x64 | pipeline3_gpu_only_tflops | 9.6601 |
| 4096 | 64x64 | persistent_gpu_only_tflops | 9.0707 |
| 4096 | 64x64 | streamk_gpu_only_tflops | 9.1562 |
| 4096 | 128x64 | pipeline3_gpu_only_tflops | 12.6254 |
| 4096 | 128x64 | persistent_gpu_only_tflops | 13.0581 |
