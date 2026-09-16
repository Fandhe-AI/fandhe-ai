| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 141.0 us (min 139.5 us / max 143.2 us) | 99.3 us (min 99.1 us / max 100.0 us) | 0.7041 | 完全一致 | 非後退 | 0.7031, 0.7118, 0.7037, 0.6919, 0.7119 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| predict_resident | 142.1 us | 98.9 us | 0.696 |
| host_copy | 0.1 us | 0.1 us | 1.000 |
| checksum | 0.6 us | 0.9 us | 1.611 |
| iter_total | 142.9 us | 100.1 us | 0.700 |
