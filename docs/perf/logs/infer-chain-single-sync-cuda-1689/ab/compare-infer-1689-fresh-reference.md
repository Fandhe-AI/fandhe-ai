| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 153.9 us (min 150.2 us / max 157.7 us) | 151.1 us (min 150.6 us / max 152.2 us) | 0.9817 | 完全一致 | 非後退 | 0.9889, 0.9825, 0.9802, 0.9584, 1.0103 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| leaf_register | 0.4 us | 0.3 us | 0.955 |
| forward | 131.0 us | 131.3 us | 1.003 |
| to_tensor | 19.1 us | 19.2 us | 1.005 |
| host_copy | 0.1 us | 0.1 us | 1.000 |
| checksum | 0.6 us | 0.7 us | 1.105 |
| iter_total | 151.4 us | 152.5 us | 1.008 |
