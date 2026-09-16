| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 514.7 us (min 512.0 us / max 518.6 us) | 510.2 us (min 508.2 us / max 536.5 us) | 0.9912 | 完全一致 | 非後退 | 0.9874, 0.9952, 1.0405, 0.9936, 0.9869 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 2.8 us | 2.9 us | 1.049 |
| leaf_register | 1.1 us | 0.8 us | 0.727 |
| forward | 176.5 us | 175.0 us | 0.991 |
| loss_readout | 0.0 us | 0.0 us | 0.800 |
| backward | 201.7 us | 205.2 us | 1.018 |
| param_readout | 65.8 us | 66.2 us | 1.006 |
| host_sgd | 51.8 us | 58.1 us | 1.123 |
| apply_params | 0.2 us | 0.2 us | 1.077 |
| tape_drop | 1.9 us | 1.8 us | 0.933 |
| step_total | 509.0 us | 509.3 us | 1.001 |
