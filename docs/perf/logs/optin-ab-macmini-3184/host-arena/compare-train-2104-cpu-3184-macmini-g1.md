| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 706.2 us (min 705.2 us / max 756.7 us) | 714.4 us (min 711.4 us / max 754.7 us) | 1.0115 | 完全一致 | 後退 | 0.9973, 1.0087, 1.0050, 1.0213, 1.0116 | いいえ |
| 64/reuse | 692.9 us (min 689.2 us / max 694.9 us) | 685.0 us (min 678.4 us / max 694.3 us) | 0.9887 | 完全一致 | 非後退 | 0.9910, 0.9999, 0.9825, 0.9881, 0.9887 | いいえ |
