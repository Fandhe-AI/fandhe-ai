| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 152.1 us (min 150.4 us / max 160.2 us) | 154.4 us (min 151.1 us / max 157.9 us) | 1.0148 | 完全一致 | 後退 | 0.9845, 1.0051, 1.0070, 1.0210, 1.0148 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| leaf_register | 0.2 us | 0.3 us | 1.222 |
| forward | 131.9 us | 130.6 us | 0.990 |
| to_tensor | 18.8 us | 19.0 us | 1.007 |
| host_copy | 0.1 us | 0.1 us | 1.000 |
| checksum | 0.6 us | 0.6 us | 1.000 |
| iter_total | 151.6 us | 150.8 us | 0.995 |
