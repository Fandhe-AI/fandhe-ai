| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/fresh | 1.028 ms (min 956.6 us / max 1.075 ms) | 1.624 ms (min 896.5 us / max 2.302 ms) | 1.5789 | 完全一致 | 後退 |
| 512/reuse | 1.161 ms (min 913.0 us / max 1.303 ms) | 1.023 ms (min 881.3 us / max 12.547 ms) | 0.8811 | 完全一致 | 非後退 |
| 1024/fresh | 5.128 ms (min 4.633 ms / max 6.969 ms) | 10.640 ms (min 5.246 ms / max 20.684 ms) | 2.0749 | 完全一致 | 後退（判定注意: before spread > 1.5x・負荷ノイズの疑い） |
| 1024/reuse | 5.024 ms (min 4.879 ms / max 5.662 ms) | 5.748 ms (min 4.242 ms / max 19.758 ms) | 1.1441 | 完全一致 | 後退 |
| 2048/fresh | 35.676 ms (min 33.878 ms / max 40.336 ms) | 43.780 ms (min 29.989 ms / max 50.142 ms) | 1.2272 | 完全一致 | 後退 |
| 2048/reuse | 38.176 ms (min 35.643 ms / max 76.549 ms) | 39.568 ms (min 36.452 ms / max 53.048 ms) | 1.0365 | 完全一致 | 非後退（判定注意: before spread > 1.5x・負荷ノイズの疑い） |
