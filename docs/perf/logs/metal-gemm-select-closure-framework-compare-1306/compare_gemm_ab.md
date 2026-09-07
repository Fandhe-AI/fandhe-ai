| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/fresh | 514.2 us (min 479.4 us / max 2.190 ms) | 511.6 us (min 479.6 us / max 1.571 ms) | 0.9950 | 完全一致 | 非後退（判定注意: before spread > 1.5x・負荷ノイズの疑い） |
| 512/reuse | 569.8 us (min 519.4 us / max 1.797 ms) | 553.6 us (min 540.4 us / max 1.789 ms) | 0.9714 | 完全一致 | 非後退（判定注意: before spread > 1.5x・負荷ノイズの疑い） |
| 1024/fresh | 2.677 ms (min 2.114 ms / max 2.813 ms) | 2.556 ms (min 2.192 ms / max 2.755 ms) | 0.9547 | 完全一致 | 非後退 |
| 1024/reuse | 2.923 ms (min 2.827 ms / max 2.976 ms) | 2.907 ms (min 2.849 ms / max 2.951 ms) | 0.9944 | 完全一致 | 非後退 |
| 2048/fresh | 8.369 ms (min 7.613 ms / max 13.190 ms) | 8.324 ms (min 8.132 ms / max 9.368 ms) | 0.9947 | 完全一致 | 非後退（判定注意: before spread > 1.5x・負荷ノイズの疑い） |
| 2048/reuse | 9.187 ms (min 9.030 ms / max 9.547 ms) | 9.522 ms (min 9.210 ms / max 10.352 ms) | 1.0365 | 完全一致 | 非後退 |
| 4096/fresh | 33.776 ms (min 33.683 ms / max 35.087 ms) | 33.926 ms (min 33.614 ms / max 34.729 ms) | 1.0045 | 完全一致 | 非後退 |
| 4096/reuse | 39.622 ms (min 38.694 ms / max 53.898 ms) | 38.868 ms (min 38.558 ms / max 41.099 ms) | 0.9809 | 完全一致 | 非後退 |
