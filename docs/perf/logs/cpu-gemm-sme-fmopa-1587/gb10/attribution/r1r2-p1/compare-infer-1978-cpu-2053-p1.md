| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 177.0 us (min 165.9 us / max 192.2 us) | 177.5 us (min 167.5 us / max 189.4 us) | 1.0032 | 完全一致 | 後退 | 1.0411, 0.8848, 1.0032, 1.0098, 1.1321 | いいえ |
| 64/reuse | 179.9 us (min 175.0 us / max 193.7 us) | 180.3 us (min 177.8 us / max 189.9 us) | 1.0024 | 完全一致 | 後退 | 1.0648, 0.9309, 0.9763, 0.9997, 1.0164 | いいえ |
