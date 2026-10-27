| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 172.0 us (min 168.5 us / max 182.8 us) | 179.3 us (min 159.5 us / max 180.8 us) | 1.0423 | 完全一致 | 後退 | 1.0527, 0.9465, 0.9888, 1.0285, 1.0157 | いいえ |
| 64/reuse | 169.6 us (min 158.2 us / max 186.2 us) | 167.4 us (min 152.1 us / max 174.6 us) | 0.9870 | 完全一致 | 非後退 | 1.0113, 0.8170, 1.0132, 1.0055, 0.9916 | いいえ |
