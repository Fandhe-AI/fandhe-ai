| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 933.2 us (min 898.7 us / max 977.9 us) | 855.2 us (min 833.0 us / max 903.0 us) | 0.9163 | 完全一致 | 非後退 | 0.8806, 0.8745, 0.9329, 0.9773, 0.9189 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 0.0 us | 0.0 us | 1.000 |
| leaf_register | 0.2 us | 0.2 us | 1.000 |
| forward_resident | 259.8 us | 257.9 us | 0.993 |
| loss_readout | 0.0 us | 0.0 us | 0.976 |
| backward | 508.3 us | 468.9 us | 0.922 |
| device_update | 123.5 us | 123.2 us | 0.997 |
| tape_drop | 0.5 us | 0.4 us | 0.768 |
| step_total | 909.5 us | 847.4 us | 0.932 |
