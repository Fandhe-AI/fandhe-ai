| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/fresh | 666.6 us (min 653.4 us / max 684.5 us) | 705.0 us (min 686.7 us / max 723.4 us) | 1.0576 | 完全一致 | 後退 |
| 512/reuse | 672.5 us (min 664.6 us / max 717.3 us) | 724.9 us (min 699.5 us / max 750.7 us) | 1.0780 | 完全一致 | 後退 |
| 1024/fresh | 3.131 ms (min 3.051 ms / max 3.222 ms) | 3.127 ms (min 3.109 ms / max 3.243 ms) | 0.9988 | 完全一致 | 非後退 |
| 1024/reuse | 3.293 ms (min 3.244 ms / max 3.476 ms) | 3.345 ms (min 3.285 ms / max 3.348 ms) | 1.0156 | 完全一致 | 非後退 |
| 2048/fresh | 21.424 ms (min 19.404 ms / max 21.777 ms) | 21.397 ms (min 20.450 ms / max 22.237 ms) | 0.9987 | 完全一致 | 非後退 |
| 2048/reuse | 21.842 ms (min 20.969 ms / max 22.478 ms) | 22.305 ms (min 21.645 ms / max 25.050 ms) | 1.0212 | 完全一致 | 非後退 |
