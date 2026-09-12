| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 774.9 us (min 743.4 us / max 852.7 us) | 802.5 us (min 747.6 us / max 806.8 us) | 1.0356 | 完全一致 | 後退 | 1.0625, 0.9831, 0.9223, 0.9412, 1.0852 | いいえ |
| 64/reuse | 960.7 us (min 911.6 us / max 1.025 ms) | 883.4 us (min 860.1 us / max 967.2 us) | 0.9196 | 完全一致 | 非後退 | 0.9434, 0.9120, 0.9461, 0.9848, 0.8774 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 0.0 us | 0.0 us | 1.000 |
| leaf_register | 0.6 us | 0.5 us | 0.928 |
| forward | 211.6 us | 229.5 us | 1.084 |
| loss_readout | 0.0 us | 0.0 us | 1.024 |
| backward | 442.6 us | 488.1 us | 1.103 |
| param_readout | 23.4 us | 23.5 us | 1.004 |
| host_sgd | 33.5 us | 33.8 us | 1.009 |
| apply_params | 0.2 us | 0.2 us | 1.202 |
| tape_drop | 1.0 us | 1.0 us | 0.920 |
| step_total | 708.0 us | 785.5 us | 1.109 |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 0.0 us | 0.0 us | 1.000 |
| leaf_register | 0.2 us | 0.2 us | 1.000 |
| forward_resident | 254.9 us | 247.9 us | 0.973 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 592.8 us | 551.2 us | 0.930 |
| device_update | 123.9 us | 124.2 us | 1.003 |
| tape_drop | 0.7 us | 0.5 us | 0.705 |
| step_total | 973.3 us | 940.3 us | 0.966 |
