| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 178.2 us (min 172.3 us / max 187.2 us) | 178.1 us (min 168.7 us / max 187.2 us) | 0.9995 | 完全一致 | 非後退 | 0.9517, 1.0130, 0.9795, 1.0363, 0.9951 | いいえ |
| 64/reuse | 182.8 us (min 179.0 us / max 196.3 us) | 183.4 us (min 179.6 us / max 202.6 us) | 1.0035 | 完全一致 | 後退 | 1.0649, 0.9949, 1.0496, 0.9914, 0.9344 | いいえ |
