| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 1024/reuse | 4.838 ms (min 4.373 ms / max 4.931 ms) | 2.288 ms (min 2.178 ms / max 2.824 ms) | 0.4728 | 完全一致 | 非後退 |
| 2048/reuse | 17.059 ms (min 16.771 ms / max 19.880 ms) | 10.487 ms (min 9.250 ms / max 10.899 ms) | 0.6147 | 完全一致 | 非後退 |
| 4096/reuse | 58.375 ms (min 40.857 ms / max 64.843 ms) | 39.367 ms (min 39.051 ms / max 40.284 ms) | 0.6744 | 完全一致 | 非後退（判定注意: before spread > 1.5x・負荷ノイズの疑い） |
