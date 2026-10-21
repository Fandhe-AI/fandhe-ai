| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 207.0 us (min 201.1 us / max 212.2 us) | 203.0 us (min 196.6 us / max 204.4 us) | 0.9808 | 完全一致 | 非後退 | 1.0070, 0.9548, 1.0095, 0.9873, 0.9487 | いいえ |
| 64/reuse | 209.8 us (min 192.2 us / max 217.6 us) | 196.0 us (min 187.5 us / max 202.1 us) | 0.9343 | 完全一致 | 非後退 | 0.9615, 0.8823, 0.9540, 0.9545, 0.9755 | いいえ |
