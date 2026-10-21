| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 2.341 ms (min 2.324 ms / max 2.597 ms) | 2.341 ms (min 2.334 ms / max 2.781 ms) | 1.0002 | 完全一致 | 後退 | 1.0710, 1.0062, 1.0045, 1.0010, 0.9989 | いいえ |
| 64/reuse | 1.886 ms (min 1.878 ms / max 1.901 ms) | 1.686 ms (min 1.681 ms / max 1.691 ms) | 0.8941 | 完全一致 | 非後退 | 0.8867, 0.8944, 0.8965, 0.8991, 0.8887 | いいえ |
