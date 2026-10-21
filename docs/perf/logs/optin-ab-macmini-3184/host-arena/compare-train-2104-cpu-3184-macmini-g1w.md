| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 725.2 us (min 716.9 us / max 753.1 us) | 746.3 us (min 724.6 us / max 767.2 us) | 1.0291 | 完全一致 | 後退 | 1.0187, 1.0106, 1.0108, 1.0281, 1.0302 | はい |
| 64/reuse | 704.0 us (min 700.6 us / max 707.7 us) | 699.3 us (min 696.1 us / max 708.6 us) | 0.9934 | 完全一致 | 非後退 | 1.0048, 1.0077, 0.9837, 0.9961, 0.9933 | いいえ |
