| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 204.2 us (min 200.1 us / max 208.9 us) | 199.0 us (min 195.5 us / max 203.1 us) | 0.9744 | 完全一致 | 非後退 | 1.0151, 0.9776, 0.9703, 0.9615, 0.9512 | いいえ |
| 64/reuse | 208.4 us (min 203.0 us / max 211.3 us) | 203.0 us (min 195.4 us / max 203.5 us) | 0.9742 | 完全一致 | 非後退 | 0.9923, 0.9742, 0.9714, 0.9247, 0.9808 | いいえ |
