| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 997.5 us (min 907.9 us / max 1.170 ms) | 850.4 us (min 795.6 us / max 944.1 us) | 0.8526 | 完全一致 | 非後退 | 0.8068, 0.7977, 0.9578, 0.7774, 0.8608 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 0.2 us | 0.2 us | 1.130 |
| leaf_register | 1.7 us | 2.0 us | 1.171 |
| forward | 255.5 us | 229.3 us | 0.897 |
| loss_readout | 0.1 us | 0.1 us | 0.667 |
| backward | 437.7 us | 426.7 us | 0.975 |
| param_readout | 38.7 us | 39.0 us | 1.008 |
| host_sgd | 66.2 us | 49.7 us | 0.751 |
| apply_params | 0.9 us | 1.1 us | 1.315 |
| tape_drop | 4.5 us | 3.6 us | 0.802 |
| step_total | 1.154 ms | 873.3 us | 0.757 |
