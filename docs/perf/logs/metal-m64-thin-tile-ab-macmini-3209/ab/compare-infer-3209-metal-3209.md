| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 754.9 us (min 694.3 us / max 790.8 us) | 762.5 us (min 751.3 us / max 771.8 us) | 1.0100 | 完全一致 | 後退 | 1.0100, 0.9521, 1.0208, 1.0821, 1.0211 | いいえ |
| 64/reuse | 366.1 us (min 336.5 us / max 449.4 us) | 374.4 us (min 330.3 us / max 497.8 us) | 1.0229 | 完全一致 | 後退 | 0.9300, 1.2437, 1.4796, 0.7350, 1.0229 | いいえ |
