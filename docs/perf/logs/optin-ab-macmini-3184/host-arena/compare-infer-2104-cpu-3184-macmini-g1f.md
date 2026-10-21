| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 201.3 us (min 198.3 us / max 208.6 us) | 206.9 us (min 198.3 us / max 214.7 us) | 1.0275 | 完全一致 | 後退 | 1.0292, 1.0430, 0.9555, 1.0100, 1.0307 | いいえ |
| 64/reuse | 211.0 us (min 207.4 us / max 213.4 us) | 204.1 us (min 202.1 us / max 207.9 us) | 0.9671 | 完全一致 | 非後退 | 0.9651, 0.9830, 0.9840, 0.9583, 0.9603 | いいえ |
