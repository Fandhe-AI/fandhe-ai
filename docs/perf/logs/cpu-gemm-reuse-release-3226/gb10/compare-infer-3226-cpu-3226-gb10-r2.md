| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 179.4 us (min 160.6 us / max 189.1 us) | 185.8 us (min 165.5 us / max 196.0 us) | 1.0357 | 完全一致 | 後退 | 1.1172, 0.9696, 1.0246, 1.0366, 1.0357 | いいえ |
| 64/reuse | 192.7 us (min 178.0 us / max 198.9 us) | 191.0 us (min 179.8 us / max 201.4 us) | 0.9915 | 完全一致 | 非後退 | 1.0307, 0.9950, 0.9915, 1.0111, 1.0100 | いいえ |
