| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 140.2 us (min 140.1 us / max 143.1 us) | 99.3 us (min 99.2 us / max 102.8 us) | 0.7084 | 完全一致 | 非後退 | 0.7078, 0.7109, 0.6941, 0.6977, 0.7338 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| predict_resident | 142.5 us | 98.7 us | 0.693 |
| host_copy | 0.1 us | 0.0 us | 0.750 |
| checksum | 0.6 us | 0.8 us | 1.395 |
| iter_total | 143.5 us | 99.8 us | 0.696 |
