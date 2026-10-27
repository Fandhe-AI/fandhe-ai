| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 858.8 us (min 825.3 us / max 868.0 us) | 908.4 us (min 892.0 us / max 946.4 us) | 1.0578 | 完全一致 | 後退 | 1.0578, 1.0621, 1.0965, 1.0904, 1.0572 | はい |
| 64/reuse | 689.3 us (min 666.8 us / max 698.0 us) | 686.9 us (min 674.4 us / max 720.9 us) | 0.9967 | 完全一致 | 非後退 | 0.9712, 1.0268, 1.0810, 1.0186, 0.9785 | いいえ |
