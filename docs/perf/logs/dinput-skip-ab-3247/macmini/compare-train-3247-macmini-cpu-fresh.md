| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 695.1 us (min 681.9 us / max 724.5 us) | 695.6 us (min 686.2 us / max 717.7 us) | 1.0006 | 完全一致 | 後退 | 0.9906, 0.9830, 1.0317, 1.0006, 1.0080 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 0.0 us | 0.0 us | 1.000 |
| leaf_register | 0.6 us | 0.6 us | 1.000 |
| forward | 213.9 us | 217.7 us | 1.018 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 420.1 us | 418.3 us | 0.996 |
| param_readout | 22.8 us | 23.1 us | 1.016 |
| host_sgd | 34.9 us | 34.4 us | 0.985 |
| apply_params | 0.2 us | 0.2 us | 1.000 |
| tape_drop | 0.6 us | 0.7 us | 1.142 |
| step_total | 693.6 us | 693.9 us | 1.000 |
