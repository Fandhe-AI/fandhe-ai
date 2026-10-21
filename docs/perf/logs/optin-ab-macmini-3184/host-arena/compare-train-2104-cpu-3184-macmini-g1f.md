| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 725.8 us (min 710.9 us / max 762.2 us) | 729.9 us (min 717.2 us / max 777.0 us) | 1.0057 | 完全一致 | 後退 | 1.0195, 0.9882, 0.9997, 1.0219, 1.0267 | いいえ |
| 64/reuse | 698.0 us (min 696.8 us / max 710.3 us) | 702.7 us (min 698.5 us / max 707.4 us) | 1.0067 | 完全一致 | 後退 | 0.9912, 1.0123, 1.0076, 1.0007, 0.9959 | いいえ |
