| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 172.5 us (min 170.4 us / max 175.8 us) | 165.6 us (min 159.4 us / max 175.8 us) | 0.9600 | 完全一致 | 非後退 | 0.9093, 0.9901, 1.0192, 0.9351, 0.9500 | いいえ |
| 64/reuse | 188.1 us (min 184.5 us / max 197.1 us) | 189.0 us (min 178.3 us / max 195.2 us) | 1.0049 | 完全一致 | 後退 | 0.9624, 0.9864, 0.9829, 1.0049, 0.9902 | いいえ |
