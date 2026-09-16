| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 462.7 us (min 458.6 us / max 464.3 us) | 311.4 us (min 309.6 us / max 317.0 us) | 0.6730 | 完全一致 | 非後退 | 0.6716, 0.6805, 0.6851, 0.6694, 0.6706 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 2.3 us | 2.6 us | 1.120 |
| leaf_register | 0.1 us | 0.3 us | 3.000 |
| forward_resident | 156.7 us | 154.5 us | 0.986 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 213.9 us | 151.2 us | 0.707 |
| device_update | 86.2 us | 7.8 us | 0.090 |
| tape_drop | 0.9 us | 0.8 us | 0.877 |
| step_total | 460.8 us | 317.4 us | 0.689 |
