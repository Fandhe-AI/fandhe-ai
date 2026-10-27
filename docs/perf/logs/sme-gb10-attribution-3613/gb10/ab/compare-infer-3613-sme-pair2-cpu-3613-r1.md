| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 179.3 us (min 159.5 us / max 180.8 us) | 164.7 us (min 162.2 us / max 185.5 us) | 0.9188 | 完全一致 | 非後退 | 0.9045, 1.0332, 0.9312, 1.0285, 0.9363 | いいえ |
| 64/reuse | 167.4 us (min 152.1 us / max 174.6 us) | 167.4 us (min 161.6 us / max 178.4 us) | 0.9998 | 完全一致 | 非後退 | 0.9254, 1.1152, 0.9435, 1.0520, 1.0658 | いいえ |
