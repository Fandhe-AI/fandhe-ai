| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 99.9 us (min 99.8 us / max 100.0 us) | 90.3 us (min 90.0 us / max 90.7 us) | 0.9038 | 完全一致 | 非後退 | 0.9018, 0.9058, 0.9037, 0.9081, 0.9001 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| predict_resident | 98.8 us | 89.2 us | 0.902 |
| host_copy | 0.1 us | 0.1 us | 1.000 |
| checksum | 0.9 us | 0.6 us | 0.673 |
| iter_total | 99.8 us | 90.0 us | 0.902 |
