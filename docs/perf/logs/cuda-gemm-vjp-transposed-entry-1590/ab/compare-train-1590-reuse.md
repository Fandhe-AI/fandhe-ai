| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 607.8 us (min 606.0 us / max 621.6 us) | 447.6 us (min 443.6 us / max 449.5 us) | 0.7365 | 完全一致 | 非後退 | 0.7403, 0.7300, 0.7208, 0.7354, 0.7321 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 2.8 us | 2.7 us | 0.944 |
| leaf_register | 0.1 us | 0.1 us | 1.167 |
| forward_resident | 156.0 us | 152.3 us | 0.976 |
| loss_readout | 0.0 us | 0.0 us | 0.667 |
| backward | 361.5 us | 197.9 us | 0.547 |
| device_update | 89.9 us | 89.5 us | 0.996 |
| tape_drop | 0.8 us | 0.9 us | 1.100 |
| step_total | 610.6 us | 444.9 us | 0.729 |
