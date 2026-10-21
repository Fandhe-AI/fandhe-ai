| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 713.4 us (min 710.9 us / max 758.9 us) | 721.6 us (min 713.0 us / max 746.1 us) | 1.0115 | 完全一致 | 後退 | 0.9831, 1.0029, 1.0117, 1.0082, 1.0137 | いいえ |
| 64/reuse | 686.9 us (min 683.6 us / max 689.6 us) | 693.1 us (min 688.6 us / max 712.3 us) | 1.0090 | 完全一致 | 後退 | 1.0153, 1.0036, 1.0420, 1.0025, 1.0060 | はい |
