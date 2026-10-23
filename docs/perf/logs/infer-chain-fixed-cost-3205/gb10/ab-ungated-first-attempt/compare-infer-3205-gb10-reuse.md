| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 99.1 us (min 99.0 us / max 99.2 us) | 90.0 us (min 89.8 us / max 90.9 us) | 0.9076 | 完全一致 | 非後退 | 0.9071, 0.9064, 0.9166, 0.9147, 0.9084 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| predict_resident | 98.9 us | 89.3 us | 0.903 |
| host_copy | 0.1 us | 0.1 us | 1.000 |
| checksum | 0.9 us | 0.7 us | 0.761 |
| iter_total | 99.9 us | 90.4 us | 0.904 |
