| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 171.2 us (min 169.9 us / max 178.1 us) | 173.6 us (min 159.6 us / max 183.2 us) | 1.0142 | 完全一致 | 後退 | 0.9483, 0.9750, 1.0580, 1.0203, 0.9394 | いいえ |
| 64/reuse | 163.9 us (min 158.4 us / max 167.4 us) | 163.2 us (min 157.9 us / max 175.5 us) | 0.9960 | 完全一致 | 非後退 | 1.0974, 1.0226, 0.9752, 1.0562, 0.9631 | いいえ |
