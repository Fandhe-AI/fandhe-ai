| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 203.6 us (min 200.0 us / max 204.1 us) | 202.0 us (min 199.1 us / max 203.8 us) | 0.9923 | 完全一致 | 非後退 | 0.9893, 0.9980, 0.9982, 1.0103, 0.9775 | いいえ |
| 64/reuse | 209.2 us (min 206.4 us / max 210.0 us) | 206.3 us (min 199.5 us / max 211.8 us) | 0.9861 | 完全一致 | 非後退 | 0.9975, 1.0124, 0.9824, 0.9505, 0.9737 | いいえ |
