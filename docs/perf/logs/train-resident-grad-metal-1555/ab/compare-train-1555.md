| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.444 ms (min 1.432 ms / max 1.902 ms) | 1.446 ms (min 1.378 ms / max 1.883 ms) | 1.0012 | 完全一致 | 後退 | 0.9896, 0.9485, 0.9719, 1.0092, 1.0024 | いいえ |
| 64/reuse | 1.361 ms (min 1.352 ms / max 1.378 ms) | 1.196 ms (min 1.167 ms / max 1.207 ms) | 0.8784 | 完全一致 | 非後退 | 0.8580, 0.8784, 0.8864, 0.8632, 0.8773 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 13.6 us | 15.2 us | 1.118 |
| leaf_register | 0.5 us | 0.5 us | 0.998 |
| forward | 651.9 us | 639.3 us | 0.981 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 697.3 us | 696.5 us | 0.999 |
| param_readout | 19.4 us | 20.0 us | 1.030 |
| host_sgd | 27.9 us | 28.0 us | 1.005 |
| apply_params | 0.2 us | 0.2 us | 1.202 |
| tape_drop | 0.5 us | 0.6 us | 1.077 |
| step_total | 1.419 ms | 1.408 ms | 0.992 |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 11.8 us | 14.4 us | 1.224 |
| leaf_register | 0.2 us | 0.2 us | 0.994 |
| forward_resident | 600.2 us | 608.1 us | 1.013 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 657.3 us | 444.6 us | 0.676 |
| device_update | 57.8 us | 139.2 us | 2.407 |
| tape_drop | 0.2 us | 0.3 us | 1.168 |
| step_total | 1.338 ms | 1.210 ms | 0.905 |
