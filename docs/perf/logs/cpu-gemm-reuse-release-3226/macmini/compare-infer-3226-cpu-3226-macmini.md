| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 205.7 us (min 193.4 us / max 209.4 us) | 206.2 us (min 202.4 us / max 212.2 us) | 1.0026 | 完全一致 | 後退 | 0.9666, 0.9802, 1.0026, 1.0355, 1.0725 | いいえ |
| 64/reuse | 208.7 us (min 205.8 us / max 216.6 us) | 209.5 us (min 204.7 us / max 215.5 us) | 1.0036 | 完全一致 | 後退 | 0.9686, 0.9947, 0.9971, 1.0184, 1.0036 | いいえ |
