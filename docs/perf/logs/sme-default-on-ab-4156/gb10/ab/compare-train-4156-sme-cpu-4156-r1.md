| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 944.6 us (min 925.7 us / max 987.4 us) | 956.2 us (min 943.8 us / max 996.0 us) | 1.0122 | 完全一致 | 後退 | 1.0543, 1.0035, 1.0472, 1.0143, 0.9656 | いいえ |
| 64/reuse | 718.4 us (min 658.3 us / max 750.0 us) | 694.2 us (min 635.2 us / max 790.3 us) | 0.9663 | 完全一致 | 非後退 | 1.1001, 0.9201, 1.1912, 0.9255, 0.9038 | いいえ |
