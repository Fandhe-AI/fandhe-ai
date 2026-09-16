| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.935 ms (min 1.921 ms / max 2.078 ms) | 1.895 ms (min 1.751 ms / max 2.046 ms) | 0.9794 | 完全一致 | 非後退 | 0.9881, 0.9847, 0.9117, 0.9183, 0.9794 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 26.0 us | 24.1 us | 0.927 |
| leaf_register | 0.7 us | 0.9 us | 1.293 |
| forward | 890.9 us | 808.5 us | 0.908 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 937.2 us | 961.8 us | 1.026 |
| param_readout | 23.8 us | 26.5 us | 1.116 |
| host_sgd | 36.4 us | 38.6 us | 1.061 |
| apply_params | 0.3 us | 0.4 us | 1.189 |
| tape_drop | 0.9 us | 0.9 us | 1.000 |
| step_total | 1.959 ms | 1.866 ms | 0.953 |
