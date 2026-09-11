| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.519 ms (min 1.503 ms / max 1.586 ms) | 1.511 ms (min 1.385 ms / max 1.723 ms) | 0.9947 | 完全一致 | 非後退 | 0.9881, 0.9478, 0.9527, 1.1338, 0.9153 | いいえ |
| 64/reuse | 1.427 ms (min 1.364 ms / max 1.467 ms) | 1.456 ms (min 1.407 ms / max 1.481 ms) | 1.0207 | 完全一致 | 後退 | 1.0230, 1.0725, 0.9864, 0.9925, 1.0207 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 13.8 us | 14.1 us | 1.021 |
| leaf_register | 0.5 us | 0.5 us | 1.092 |
| forward | 679.4 us | 653.6 us | 0.962 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 715.2 us | 707.4 us | 0.989 |
| param_readout | 21.4 us | 21.5 us | 1.004 |
| host_sgd | 31.1 us | 31.2 us | 1.004 |
| apply_params | 0.2 us | 0.2 us | 0.995 |
| tape_drop | 0.6 us | 0.5 us | 0.963 |
| step_total | 1.497 ms | 1.446 ms | 0.966 |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 12.1 us | 12.7 us | 1.050 |
| leaf_register | 0.1 us | 0.1 us | 1.000 |
| forward_resident | 618.4 us | 613.6 us | 0.992 |
| loss_readout | 0.0 us | 0.0 us | 0.976 |
| backward | 683.0 us | 678.6 us | 0.994 |
| device_update | 62.2 us | 62.9 us | 1.011 |
| tape_drop | 0.2 us | 0.2 us | 1.000 |
| step_total | 1.406 ms | 1.391 ms | 0.990 |
