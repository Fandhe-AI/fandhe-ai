| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 2.061 ms (min 1.620 ms / max 2.080 ms) | 2.009 ms (min 1.598 ms / max 2.093 ms) | 0.9746 | 完全一致 | 非後退 | 0.9867, 0.9949, 0.9746, 0.8182, 1.0248 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 34.5 us | 31.0 us | 0.900 |
| leaf_register | 0.9 us | 0.8 us | 0.819 |
| forward | 862.3 us | 929.6 us | 1.078 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 949.3 us | 1.015 ms | 1.069 |
| param_readout | 23.5 us | 24.0 us | 1.018 |
| host_sgd | 34.8 us | 34.3 us | 0.986 |
| apply_params | 0.3 us | 0.5 us | 1.497 |
| tape_drop | 0.9 us | 1.0 us | 1.091 |
| step_total | 1.936 ms | 2.063 ms | 1.065 |
