| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.124 ms (min 1.001 ms / max 1.164 ms) | 1.198 ms (min 1.065 ms / max 1.240 ms) | 1.0662 | 完全一致 | 後退 | 0.9471, 1.0522, 1.1968, 1.0515, 1.0957 | いいえ |
| 64/reuse | 851.3 us (min 823.9 us / max 870.7 us) | 946.1 us (min 922.6 us / max 1.004 ms) | 1.1115 | 完全一致 | 後退 | 1.0838, 1.1374, 1.1527, 1.1484, 1.0932 | はい |
