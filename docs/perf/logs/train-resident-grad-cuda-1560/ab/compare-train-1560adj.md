| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 459.5 us (min 458.7 us / max 468.5 us) | 337.2 us (min 335.6 us / max 338.0 us) | 0.7339 | 完全一致 | 非後退 | 0.7230, 0.7354, 0.7303, 0.7191, 0.7361 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 2.6 us | 2.5 us | 0.957 |
| leaf_register | 0.1 us | 0.1 us | 1.000 |
| forward_resident | 155.6 us | 154.8 us | 0.995 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 215.3 us | 170.0 us | 0.789 |
| device_update | 85.6 us | 7.8 us | 0.091 |
| tape_drop | 0.8 us | 0.8 us | 0.981 |
| step_total | 460.5 us | 336.5 us | 0.731 |
