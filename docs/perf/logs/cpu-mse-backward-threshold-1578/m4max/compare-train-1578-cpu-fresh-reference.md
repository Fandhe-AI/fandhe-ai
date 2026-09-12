| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 836.1 us (min 719.7 us / max 866.1 us) | 736.2 us (min 713.7 us / max 748.2 us) | 0.8805 | 完全一致 | 非後退 | 0.8676, 0.8240, 0.8708, 1.0396, 0.9083 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 0.0 us | 0.0 us | 0.976 |
| leaf_register | 0.5 us | 0.4 us | 0.869 |
| forward | 242.4 us | 265.1 us | 1.094 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 546.5 us | 430.8 us | 0.788 |
| param_readout | 23.4 us | 22.9 us | 0.979 |
| host_sgd | 34.2 us | 33.5 us | 0.979 |
| apply_params | 0.2 us | 0.2 us | 1.000 |
| tape_drop | 0.9 us | 0.8 us | 0.863 |
| step_total | 864.8 us | 781.5 us | 0.904 |
