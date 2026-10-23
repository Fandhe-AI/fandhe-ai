| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 695.7 us (min 686.9 us / max 700.6 us) | 694.5 us (min 690.2 us / max 708.4 us) | 0.9983 | 完全一致 | 非後退 | 0.9851, 1.0030, 0.9902, 1.0197, 1.0182 | いいえ |
| 64/reuse | 673.1 us (min 671.6 us / max 680.1 us) | 677.4 us (min 673.8 us / max 681.0 us) | 1.0063 | 完全一致 | 後退 | 1.0086, 1.0015, 1.0021, 0.9969, 1.0029 | いいえ |
