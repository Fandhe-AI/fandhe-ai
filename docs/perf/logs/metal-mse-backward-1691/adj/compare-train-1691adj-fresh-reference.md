| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 2.004 ms (min 1.517 ms / max 2.258 ms) | 1.968 ms (min 1.822 ms / max 2.349 ms) | 0.9818 | 完全一致 | 非後退 | 1.0555, 0.9962, 0.8716, 0.9667, 1.2672 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 19.8 us | 15.8 us | 0.799 |
| leaf_register | 0.5 us | 0.5 us | 0.847 |
| forward | 917.9 us | 708.0 us | 0.771 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 1.014 ms | 891.2 us | 0.879 |
| param_readout | 22.3 us | 23.4 us | 1.050 |
| host_sgd | 32.8 us | 31.7 us | 0.965 |
| apply_params | 0.3 us | 0.2 us | 0.774 |
| tape_drop | 0.8 us | 0.6 us | 0.790 |
| step_total | 2.025 ms | 1.659 ms | 0.819 |
