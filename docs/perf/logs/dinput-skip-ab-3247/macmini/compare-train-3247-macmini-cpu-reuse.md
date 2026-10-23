| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 674.0 us (min 667.5 us / max 687.0 us) | 498.0 us (min 489.9 us / max 509.7 us) | 0.7389 | 完全一致 | 非後退 | 0.7637, 0.7229, 0.7406, 0.7198, 0.7453 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 0.0 us | 0.0 us | 0.976 |
| leaf_register | 0.1 us | 0.2 us | 1.328 |
| forward_resident | 205.7 us | 209.8 us | 1.020 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 434.9 us | 256.9 us | 0.591 |
| device_update | 30.4 us | 29.3 us | 0.964 |
| tape_drop | 0.3 us | 0.3 us | 1.068 |
| step_total | 672.4 us | 494.6 us | 0.736 |
