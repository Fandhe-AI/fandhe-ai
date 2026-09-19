| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 170.3 us (min 163.7 us / max 182.0 us) | 172.6 us (min 165.3 us / max 185.2 us) | 1.0132 | 完全一致 | 後退 | 0.9427, 1.0132, 1.1313, 1.0241, 0.9477 | いいえ |
| 64/reuse | 181.6 us (min 177.4 us / max 194.4 us) | 192.6 us (min 171.1 us / max 195.1 us) | 1.0606 | 完全一致 | 後退 | 1.0528, 1.0434, 1.0606, 1.0998, 0.8800 | いいえ |
