| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 514.3 us (min 512.9 us / max 526.3 us) | 515.8 us (min 511.0 us / max 545.6 us) | 1.0030 | 完全一致 | 後退 | 0.9936, 1.0529, 1.0367, 1.0057, 0.9969 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 2.7 us | 2.8 us | 1.021 |
| leaf_register | 0.8 us | 1.7 us | 2.250 |
| forward | 176.0 us | 175.7 us | 0.998 |
| loss_readout | 0.0 us | 0.0 us | 0.800 |
| backward | 187.7 us | 202.1 us | 1.076 |
| param_readout | 81.0 us | 66.7 us | 0.823 |
| host_sgd | 63.4 us | 51.3 us | 0.810 |
| apply_params | 0.2 us | 0.3 us | 1.231 |
| tape_drop | 1.8 us | 2.0 us | 1.142 |
| step_total | 511.2 us | 509.9 us | 0.998 |
