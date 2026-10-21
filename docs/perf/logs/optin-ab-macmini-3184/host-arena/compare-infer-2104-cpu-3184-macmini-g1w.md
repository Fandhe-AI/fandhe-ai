| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 205.1 us (min 200.4 us / max 208.3 us) | 205.2 us (min 198.6 us / max 209.9 us) | 1.0007 | 完全一致 | 後退 | 1.0018, 0.9541, 1.0474, 0.9716, 1.0154 | いいえ |
| 64/reuse | 208.0 us (min 203.6 us / max 213.1 us) | 209.9 us (min 202.0 us / max 212.4 us) | 1.0090 | 完全一致 | 後退 | 0.9914, 0.9712, 0.9852, 1.0433, 1.0308 | いいえ |
