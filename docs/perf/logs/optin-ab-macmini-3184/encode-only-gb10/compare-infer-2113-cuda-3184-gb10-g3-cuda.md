| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 155.8 us (min 152.5 us / max 156.8 us) | 157.4 us (min 150.8 us / max 157.8 us) | 1.0101 | 完全一致 | 後退 | 1.0063, 0.9679, 1.0280, 1.0036, 1.0321 | いいえ |
| 64/reuse | 99.4 us (min 99.0 us / max 100.0 us) | 99.0 us (min 98.9 us / max 100.0 us) | 0.9962 | 完全一致 | 非後退 | 0.9953, 0.9911, 1.0103, 0.9905, 0.9969 | いいえ |
