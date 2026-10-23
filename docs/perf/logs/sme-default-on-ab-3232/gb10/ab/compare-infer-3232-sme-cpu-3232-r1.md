| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 176.1 us (min 167.7 us / max 200.7 us) | 169.8 us (min 166.7 us / max 177.4 us) | 0.9645 | 完全一致 | 非後退 | 1.0164, 0.8390, 0.9452, 1.0577, 0.9466 | いいえ |
| 64/reuse | 191.9 us (min 180.7 us / max 225.1 us) | 183.1 us (min 169.4 us / max 193.0 us) | 0.9543 | 完全一致 | 非後退 | 0.9501, 1.0311, 0.8574, 0.8826, 0.9323 | いいえ |
