| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 858.8 us (min 825.3 us / max 868.0 us) | 848.5 us (min 814.6 us / max 910.8 us) | 0.9880 | 完全一致 | 非後退 | 0.9485, 0.9810, 1.0281, 1.0494, 1.0032 | いいえ |
| 64/reuse | 689.3 us (min 666.8 us / max 698.0 us) | 670.1 us (min 648.0 us / max 699.7 us) | 0.9722 | 完全一致 | 非後退 | 0.9732, 0.9686, 1.0007, 1.0131, 0.9722 | いいえ |
