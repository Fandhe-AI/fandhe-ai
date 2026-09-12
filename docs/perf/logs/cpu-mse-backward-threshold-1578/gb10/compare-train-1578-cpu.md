| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 980.0 us (min 966.7 us / max 1.043 ms) | 996.4 us (min 921.3 us / max 1.222 ms) | 1.0167 | 完全一致 | 後退 | 1.0578, 1.2644, 1.0117, 0.9734, 0.8833 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 0.2 us | 0.1 us | 0.593 |
| leaf_register | 0.3 us | 0.2 us | 0.639 |
| forward_resident | 276.1 us | 239.9 us | 0.869 |
| loss_readout | 0.2 us | 0.1 us | 0.818 |
| backward | 543.2 us | 449.5 us | 0.827 |
| device_update | 247.3 us | 280.6 us | 1.135 |
| tape_drop | 1.8 us | 1.8 us | 0.996 |
| step_total | 1.140 ms | 1.019 ms | 0.894 |
