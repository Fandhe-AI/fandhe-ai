| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.118 ms (min 1.069 ms / max 1.129 ms) | 967.3 us (min 878.7 us / max 1.024 ms) | 0.8648 | 完全一致 | 非後退 | 0.9068, 0.8217, 0.8484, 0.9022, 0.9050 | いいえ |
| 64/reuse | 828.1 us (min 823.8 us / max 850.7 us) | 842.4 us (min 832.1 us / max 863.2 us) | 1.0172 | 完全一致 | 後退 | 0.9902, 1.0074, 1.0162, 0.9950, 1.0424 | いいえ |
