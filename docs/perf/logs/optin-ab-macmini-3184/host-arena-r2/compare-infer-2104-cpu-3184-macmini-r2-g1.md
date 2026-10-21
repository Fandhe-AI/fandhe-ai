| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 200.9 us (min 174.4 us / max 206.2 us) | 200.1 us (min 187.0 us / max 206.2 us) | 0.9961 | 完全一致 | 非後退 | 1.0109, 1.0720, 1.0297, 0.9750, 1.0002 | いいえ |
| 64/reuse | 202.0 us (min 198.6 us / max 205.2 us) | 205.6 us (min 177.5 us / max 208.8 us) | 1.0181 | 完全一致 | 後退 | 1.0194, 0.8939, 1.0163, 0.9997, 1.0181 | いいえ |
