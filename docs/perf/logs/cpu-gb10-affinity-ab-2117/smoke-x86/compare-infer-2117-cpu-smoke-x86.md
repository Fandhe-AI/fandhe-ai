| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 202.9 us (min 158.6 us / max 6.001 ms) | 201.9 us (min 169.8 us / max 7.999 ms) | 0.9953 | 完全一致 | 非後退（判定注意: before spread > 1.5x・負荷ノイズの疑い） | 0.9953, 0.9476, 0.8817, 1.1619, 1.3330 | いいえ |
| 64/reuse | 203.1 us (min 191.8 us / max 357.2 us) | 189.9 us (min 170.9 us / max 652.4 us) | 0.9349 | 完全一致 | 非後退（判定注意: before spread > 1.5x・負荷ノイズの疑い） | 0.9133, 0.9166, 1.0339, 0.8818, 1.8263 | いいえ |
