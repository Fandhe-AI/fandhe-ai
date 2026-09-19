| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.053 ms (min 1.045 ms / max 1.078 ms) | 1.063 ms (min 1.037 ms / max 1.122 ms) | 1.0102 | 完全一致 | 後退 | 0.9808, 0.9792, 1.0173, 1.0720, 1.0310 | いいえ |
| 64/reuse | 1.058 ms (min 962.5 us / max 1.131 ms) | 986.7 us (min 927.8 us / max 1.070 ms) | 0.9328 | 完全一致 | 非後退 | 0.9725, 0.8331, 1.1121, 0.8771, 0.9501 | いいえ |
