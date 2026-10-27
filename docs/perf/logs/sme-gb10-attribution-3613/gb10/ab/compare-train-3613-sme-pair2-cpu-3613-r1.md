| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 848.5 us (min 814.6 us / max 910.8 us) | 908.4 us (min 892.0 us / max 946.4 us) | 1.0706 | 完全一致 | 後退 | 1.1151, 1.0827, 1.0665, 1.0391, 1.0538 | はい |
| 64/reuse | 670.1 us (min 648.0 us / max 699.7 us) | 686.9 us (min 674.4 us / max 720.9 us) | 1.0251 | 完全一致 | 後退 | 0.9980, 1.0601, 1.0802, 1.0054, 1.0064 | いいえ |
