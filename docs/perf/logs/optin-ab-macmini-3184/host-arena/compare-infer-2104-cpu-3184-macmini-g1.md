| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 203.6 us (min 195.0 us / max 206.8 us) | 203.4 us (min 195.3 us / max 208.6 us) | 0.9993 | 完全一致 | 非後退 | 1.0250, 0.9670, 1.0143, 1.0018, 0.9876 | いいえ |
| 64/reuse | 205.8 us (min 199.7 us / max 210.0 us) | 206.3 us (min 193.8 us / max 207.8 us) | 1.0022 | 完全一致 | 後退 | 0.9962, 0.9704, 1.0194, 0.9867, 0.9861 | いいえ |
