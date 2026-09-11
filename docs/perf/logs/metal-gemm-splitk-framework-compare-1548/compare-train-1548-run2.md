| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.646 ms (min 1.534 ms / max 1.814 ms) | 1.562 ms (min 1.414 ms / max 1.913 ms) | 0.9493 | 完全一致 | 非後退 | 0.9014, 1.0182, 1.0550, 0.9948, 0.8592 | いいえ |
| 64/reuse | 1.426 ms (min 1.345 ms / max 1.490 ms) | 1.433 ms (min 1.300 ms / max 1.469 ms) | 1.0049 | 完全一致 | 後退 | 0.9325, 1.0049, 0.9859, 1.0010, 0.9982 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 16.7 us | 18.0 us | 1.081 |
| leaf_register | 0.5 us | 0.6 us | 1.078 |
| forward | 621.9 us | 656.5 us | 1.056 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 674.5 us | 709.6 us | 1.052 |
| param_readout | 21.9 us | 22.0 us | 1.004 |
| host_sgd | 32.4 us | 31.5 us | 0.975 |
| apply_params | 0.2 us | 0.2 us | 1.005 |
| tape_drop | 0.7 us | 0.7 us | 1.063 |
| step_total | 1.378 ms | 1.451 ms | 1.053 |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 13.4 us | 15.4 us | 1.143 |
| leaf_register | 0.2 us | 0.2 us | 1.006 |
| forward_resident | 606.5 us | 791.5 us | 1.305 |
| loss_readout | 0.0 us | 0.0 us | 1.024 |
| backward | 667.1 us | 850.1 us | 1.274 |
| device_update | 62.5 us | 67.0 us | 1.072 |
| tape_drop | 0.3 us | 0.3 us | 1.144 |
| step_total | 1.358 ms | 1.740 ms | 1.282 |
