| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.537 ms (min 1.392 ms / max 1.822 ms) | 1.467 ms (min 1.373 ms / max 1.584 ms) | 0.9545 | 完全一致 | 非後退 | 0.8693, 0.9152, 1.0141, 0.9864, 0.9726 | いいえ |
| 64/reuse | 1.060 ms (min 1.039 ms / max 1.100 ms) | 1.023 ms (min 1.006 ms / max 1.118 ms) | 0.9653 | 完全一致 | 非後退 | 1.0147, 1.0296, 0.9209, 0.9538, 0.9849 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 13.6 us | 12.8 us | 0.946 |
| leaf_register | 0.5 us | 0.5 us | 1.000 |
| forward | 657.6 us | 632.7 us | 0.962 |
| loss_readout | 0.0 us | 0.0 us | 0.976 |
| backward | 718.5 us | 684.7 us | 0.953 |
| param_readout | 21.6 us | 21.3 us | 0.989 |
| host_sgd | 31.7 us | 31.4 us | 0.990 |
| apply_params | 0.2 us | 0.2 us | 0.803 |
| tape_drop | 0.5 us | 0.5 us | 1.000 |
| step_total | 1.462 ms | 1.395 ms | 0.954 |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 13.3 us | 13.8 us | 1.038 |
| leaf_register | 0.2 us | 0.2 us | 1.000 |
| forward_resident | 621.9 us | 562.5 us | 0.904 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 483.8 us | 422.5 us | 0.873 |
| device_update | 3.9 us | 4.4 us | 1.123 |
| tape_drop | 0.2 us | 0.3 us | 1.332 |
| step_total | 1.132 ms | 1.007 ms | 0.890 |
