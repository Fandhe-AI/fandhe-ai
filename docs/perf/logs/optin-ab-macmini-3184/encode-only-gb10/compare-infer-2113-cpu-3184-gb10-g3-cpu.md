| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 173.2 us (min 170.1 us / max 189.9 us) | 186.8 us (min 182.4 us / max 188.0 us) | 1.0780 | 完全一致 | 後退 | 1.0835, 1.0787, 0.9899, 0.9911, 1.0911 | いいえ |
| 64/reuse | 186.5 us (min 180.4 us / max 203.0 us) | 199.4 us (min 186.5 us / max 202.0 us) | 1.0691 | 完全一致 | 後退 | 1.1198, 1.0749, 0.9823, 1.0757, 0.9893 | いいえ |
