| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 727.4 us (min 721.2 us / max 743.2 us) | 758.8 us (min 751.3 us / max 764.1 us) | 1.0432 | 完全一致 | 後退 | 1.0520, 1.0595, 1.0151, 1.0329, 1.0474 | はい |
| 64/reuse | 708.5 us (min 699.7 us / max 714.0 us) | 696.5 us (min 687.6 us / max 735.2 us) | 0.9832 | 完全一致 | 非後退 | 0.9633, 0.9954, 0.9706, 1.0357, 1.0310 | いいえ |
