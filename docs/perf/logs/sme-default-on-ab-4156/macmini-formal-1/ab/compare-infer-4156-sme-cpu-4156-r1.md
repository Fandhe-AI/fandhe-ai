| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 200.1 us (min 195.6 us / max 205.4 us) | 200.1 us (min 197.0 us / max 201.6 us) | 1.0002 | 完全一致 | 後退 | 0.9628, 1.0276, 1.0077, 0.9740, 0.9975 | いいえ |
| 64/reuse | 198.1 us (min 197.2 us / max 210.6 us) | 191.1 us (min 188.0 us / max 199.2 us) | 0.9646 | 完全一致 | 非後退 | 0.9545, 1.0061, 0.9692, 0.8928, 0.9592 | いいえ |
