| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 760.2 us (min 745.8 us / max 767.2 us) | 758.4 us (min 755.5 us / max 774.0 us) | 0.9977 | 完全一致 | 非後退 | 1.0182, 0.9885, 1.0171, 0.9938, 1.0066 | いいえ |
| 64/reuse | 393.4 us (min 329.9 us / max 415.6 us) | 371.1 us (min 333.8 us / max 440.2 us) | 0.9434 | 完全一致 | 非後退 | 1.0080, 1.3342, 1.0937, 0.8485, 0.8972 | いいえ |
