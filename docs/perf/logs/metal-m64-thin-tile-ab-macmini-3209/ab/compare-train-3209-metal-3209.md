| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 2.325 ms (min 2.322 ms / max 2.429 ms) | 2.347 ms (min 2.326 ms / max 2.540 ms) | 1.0093 | 完全一致 | 後退 | 1.0456, 1.0123, 1.0005, 1.0063, 1.0066 | はい |
| 64/reuse | 1.881 ms (min 1.877 ms / max 1.891 ms) | 1.878 ms (min 1.874 ms / max 1.892 ms) | 0.9984 | 完全一致 | 非後退 | 1.0005, 0.9962, 0.9992, 1.0009, 0.9963 | いいえ |
