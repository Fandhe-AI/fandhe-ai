| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 526.0 us (min 507.6 us / max 529.2 us) | 522.9 us (min 506.2 us / max 537.0 us) | 0.9941 | 完全一致 | 非後退 | 1.0313, 1.0237, 0.9565, 0.9970, 0.9941 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 3.1 us | 3.2 us | 1.044 |
| leaf_register | 1.9 us | 2.1 us | 1.108 |
| forward | 169.4 us | 169.5 us | 1.000 |
| loss_readout | 0.0 us | 0.0 us | 0.667 |
| backward | 188.6 us | 179.7 us | 0.953 |
| param_readout | 98.8 us | 97.7 us | 0.989 |
| host_sgd | 62.2 us | 61.2 us | 0.985 |
| apply_params | 0.5 us | 0.4 us | 0.897 |
| tape_drop | 1.9 us | 1.9 us | 1.021 |
| step_total | 527.3 us | 518.8 us | 0.984 |
