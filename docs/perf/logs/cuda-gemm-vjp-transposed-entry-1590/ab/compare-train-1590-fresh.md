| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.079 ms (min 1.068 ms / max 1.090 ms) | 528.1 us (min 514.7 us / max 544.5 us) | 0.4892 | 完全一致 | 非後退 | 0.4768, 0.5034, 0.5010, 0.4926, 0.4816 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 3.6 us | 3.1 us | 0.855 |
| leaf_register | 0.9 us | 1.2 us | 1.327 |
| forward | 176.1 us | 174.9 us | 0.993 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 737.5 us | 196.1 us | 0.266 |
| param_readout | 80.9 us | 72.0 us | 0.889 |
| host_sgd | 64.5 us | 62.3 us | 0.967 |
| apply_params | 0.3 us | 0.2 us | 0.875 |
| tape_drop | 2.1 us | 2.0 us | 0.955 |
| step_total | 1.068 ms | 511.8 us | 0.479 |
