| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 720.1 us (min 716.2 us / max 730.8 us) | 712.0 us (min 692.5 us / max 718.1 us) | 0.9888 | 完全一致 | 非後退 | 0.9649, 0.9617, 0.9900, 0.9969, 1.0024 | いいえ |
| 64/reuse | 700.1 us (min 696.3 us / max 704.4 us) | 669.6 us (min 664.6 us / max 731.3 us) | 0.9564 | 完全一致 | 非後退 | 1.0503, 0.9564, 0.9563, 0.9490, 0.9532 | いいえ |
