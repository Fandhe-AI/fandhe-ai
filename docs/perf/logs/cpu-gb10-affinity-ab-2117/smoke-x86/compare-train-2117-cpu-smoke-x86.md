| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 928.9 us (min 848.3 us / max 4.102 ms) | 1.024 ms (min 861.0 us / max 2.792 ms) | 1.1020 | 完全一致 | 後退（判定注意: before spread > 1.5x・負荷ノイズの疑い） | 1.0345, 1.1364, 1.0150, 1.0102, 0.6806 | いいえ |
| 64/reuse | 901.2 us (min 795.8 us / max 1.760 ms) | 917.5 us (min 855.4 us / max 1.874 ms) | 1.0181 | 完全一致 | 後退（判定注意: before spread > 1.5x・負荷ノイズの疑い） | 1.0040, 0.9475, 1.0750, 1.0449, 1.0650 | いいえ |
