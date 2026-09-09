| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/fresh | 1.793 ms (min 1.639 ms / max 1.965 ms) | 1.752 ms (min 1.597 ms / max 1.860 ms) | 0.9772 | 完全一致 | 非後退 |
| 512/reuse | 2.450 ms (min 2.202 ms / max 2.628 ms) | 2.475 ms (min 2.272 ms / max 2.778 ms) | 1.0102 | 完全一致 | 非後退 |
| 1024/fresh | 4.408 ms (min 3.994 ms / max 4.797 ms) | 4.650 ms (min 3.865 ms / max 4.908 ms) | 1.0549 | 完全一致 | 後退 |
| 1024/reuse | 5.596 ms (min 5.251 ms / max 5.658 ms) | 5.530 ms (min 5.518 ms / max 5.873 ms) | 0.9882 | 完全一致 | 非後退 |
| 2048/fresh | 16.592 ms (min 15.877 ms / max 23.738 ms) | 16.735 ms (min 16.649 ms / max 17.302 ms) | 1.0086 | 完全一致 | 非後退 |
| 2048/reuse | 27.738 ms (min 20.063 ms / max 28.105 ms) | 19.666 ms (min 19.196 ms / max 20.221 ms) | 0.7090 | 完全一致 | 非後退 |
