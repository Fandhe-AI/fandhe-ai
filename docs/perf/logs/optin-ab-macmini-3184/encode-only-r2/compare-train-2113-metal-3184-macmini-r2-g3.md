| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 2.327 ms (min 2.269 ms / max 2.991 ms) | 2.345 ms (min 2.133 ms / max 2.455 ms) | 1.0077 | 完全一致 | 後退 | 0.8209, 0.9400, 1.0075, 1.0094, 1.0024 | いいえ |
| 64/reuse | 1.879 ms (min 1.873 ms / max 1.890 ms) | 1.691 ms (min 1.662 ms / max 1.697 ms) | 0.9000 | 完全一致 | 非後退 | 0.9060, 0.8945, 0.8970, 0.9060, 0.8835 | いいえ |
