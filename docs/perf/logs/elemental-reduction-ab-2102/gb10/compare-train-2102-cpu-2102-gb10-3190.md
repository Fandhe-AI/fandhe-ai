| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.101 ms (min 1.073 ms / max 1.149 ms) | 868.6 us (min 823.2 us / max 878.1 us) | 0.7889 | 完全一致 | 非後退 | 0.7719, 0.7563, 0.8145, 0.7443, 0.8132 | いいえ |
| 64/reuse | 825.5 us (min 800.3 us / max 841.4 us) | 816.4 us (min 777.1 us / max 826.9 us) | 0.9890 | 完全一致 | 非後退 | 1.0017, 0.9838, 0.9477, 0.9607, 1.0304 | いいえ |
