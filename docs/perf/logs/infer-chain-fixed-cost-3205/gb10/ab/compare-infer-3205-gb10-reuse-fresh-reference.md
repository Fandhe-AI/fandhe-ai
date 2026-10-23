| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 153.1 us (min 152.2 us / max 159.8 us) | 156.4 us (min 151.9 us / max 161.2 us) | 1.0216 | 完全一致 | 後退 | 1.0530, 0.9854, 0.9980, 1.0006, 1.0216 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| leaf_register | 0.2 us | 0.3 us | 1.308 |
| forward | 131.9 us | 137.2 us | 1.041 |
| to_tensor | 19.7 us | 19.5 us | 0.991 |
| host_copy | 0.1 us | 0.1 us | 1.000 |
| checksum | 0.6 us | 0.6 us | 1.000 |
| iter_total | 153.1 us | 157.5 us | 1.029 |
