| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 180.7 us (min 169.6 us / max 191.5 us) | 173.3 us (min 158.9 us / max 190.3 us) | 0.9592 | 完全一致 | 非後退 | 0.9050, 1.0375, 0.9753, 0.9076, 0.9749 | いいえ |
| 64/reuse | 177.7 us (min 163.3 us / max 209.9 us) | 156.1 us (min 143.0 us / max 197.3 us) | 0.8787 | 完全一致 | 非後退 | 0.8349, 1.1102, 0.8187, 1.1041, 0.7438 | いいえ |
