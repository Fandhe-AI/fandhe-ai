| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 311.9 us (min 309.9 us / max 315.3 us) | 277.8 us (min 273.9 us / max 279.1 us) | 0.8905 | 完全一致 | 非後退 | 0.8843, 0.8870, 0.9008, 0.8910, 0.8687 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 2.5 us | 2.3 us | 0.922 |
| leaf_register | 0.3 us | 0.3 us | 0.941 |
| forward_resident | 132.6 us | 130.4 us | 0.984 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 178.5 us | 142.1 us | 0.796 |
| device_update | 2.7 us | 2.6 us | 0.965 |
| tape_drop | 0.8 us | 0.9 us | 1.137 |
| step_total | 317.3 us | 278.5 us | 0.878 |
