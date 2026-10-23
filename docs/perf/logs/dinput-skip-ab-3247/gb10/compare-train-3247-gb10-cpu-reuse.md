| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 918.1 us (min 810.6 us / max 1.101 ms) | 745.6 us (min 727.6 us / max 854.0 us) | 0.8122 | 完全一致 | 非後退 | 0.7167, 0.6751, 0.8564, 0.8976, 0.9701 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 0.2 us | 0.2 us | 1.091 |
| leaf_register | 0.8 us | 0.6 us | 0.737 |
| forward_resident | 246.0 us | 249.3 us | 1.013 |
| loss_readout | 0.1 us | 0.1 us | 0.933 |
| backward | 491.8 us | 316.6 us | 0.644 |
| device_update | 133.2 us | 134.2 us | 1.007 |
| tape_drop | 1.8 us | 1.6 us | 0.891 |
| step_total | 911.3 us | 740.2 us | 0.812 |
