| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 932.7 us (min 862.1 us / max 975.0 us) | 969.8 us (min 956.4 us / max 1.073 ms) | 1.0398 | 完全一致 | 後退 | 1.0488, 1.2444, 0.9809, 1.1095, 1.0398 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 0.2 us | 0.2 us | 0.840 |
| leaf_register | 1.3 us | 1.3 us | 1.025 |
| forward | 237.3 us | 244.3 us | 1.030 |
| loss_readout | 0.1 us | 0.1 us | 1.286 |
| backward | 428.6 us | 425.8 us | 0.993 |
| param_readout | 38.9 us | 38.1 us | 0.980 |
| host_sgd | 54.3 us | 55.0 us | 1.012 |
| apply_params | 0.8 us | 0.7 us | 0.883 |
| tape_drop | 3.9 us | 3.5 us | 0.900 |
| step_total | 914.7 us | 986.2 us | 1.078 |
