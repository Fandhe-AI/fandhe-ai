| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 169.6 us (min 160.9 us / max 192.7 us) | 180.4 us (min 170.8 us / max 187.4 us) | 1.0638 | 完全一致 | 後退 | 1.0350, 1.0977, 1.1213, 0.9311, 0.9722 | いいえ |
| 64/reuse | 194.2 us (min 177.0 us / max 197.4 us) | 186.7 us (min 179.3 us / max 200.4 us) | 0.9613 | 完全一致 | 非後退 | 1.0133, 0.9692, 0.9165, 1.1325, 0.9456 | いいえ |
