| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 203.6 us (min 199.9 us / max 208.4 us) | 200.0 us (min 193.7 us / max 205.2 us) | 0.9821 | 完全一致 | 非後退 | 0.9484, 1.0078, 0.9689, 0.9934, 0.9836 | いいえ |
| 64/reuse | 202.9 us (min 200.4 us / max 205.9 us) | 205.0 us (min 200.2 us / max 207.4 us) | 1.0100 | 完全一致 | 後退 | 1.0194, 1.0205, 1.0072, 0.9775, 1.0100 | いいえ |
