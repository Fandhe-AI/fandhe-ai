| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 873.0 us (min 827.4 us / max 926.6 us) | 871.1 us (min 814.2 us / max 995.1 us) | 0.9979 | 完全一致 | 非後退 | 1.0053, 1.0528, 0.9403, 1.0739, 0.9626 | いいえ |
| 64/reuse | 825.4 us (min 817.6 us / max 832.7 us) | 859.8 us (min 847.7 us / max 890.9 us) | 1.0417 | 完全一致 | 後退 | 1.0361, 1.0357, 1.0452, 1.0849, 1.0270 | はい |
